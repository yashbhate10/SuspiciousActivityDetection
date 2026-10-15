import os
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

from ..database import db_manager

router = APIRouter(prefix="/api/videos", tags=["Video Testcases"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "test_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Add ai-engine to sys.path for AI models
AI_ENGINE_DIR = PROJECT_ROOT / "ai-engine"
if str(AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_DIR))

from tracking import ObjectTracker
from behaviour import BehaviourEngine
from anomaly_classifier import AnomalyClassifier
from evidence import save_evidence_snapshot
from config import RESTRICTED_ZONE_POLYGON, LOITERING_ZONE_POLYGON, MODEL_PATH

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Shared lazy-loaded AI models
class AIStreamingEngine:
    def __init__(self):
        self._tracker = None
        self._anomaly_clf = None

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = ObjectTracker(model_path=MODEL_PATH)
        return self._tracker

    @property
    def anomaly_clf(self):
        if self._anomaly_clf is None:
            self._anomaly_clf = AnomalyClassifier()
        return self._anomaly_clf

ai_engine_pool = AIStreamingEngine()

def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "filename": video_path.name,
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "valid": False,
            "error": "Could not decode video"
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(frame_count / max(1.0, fps), 1)
    cap.release()

    return {
        "filename": video_path.name,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(video_path.stat().st_mtime)),
        "valid": True
    }

@router.get("", response_model=List[Dict[str, Any]])
def list_testcase_videos():
    """List all available testcase videos in data/test_videos and sample UCF-Crime videos."""
    videos = []
    # 1. Custom uploaded & test videos
    for file in sorted(VIDEO_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if file.suffix.lower() in ALLOWED_EXTENSIONS:
            videos.append(get_video_metadata(file))

    # 2. Add sample UCF-Crime videos if available for immediate testing
    ucf_dir = PROJECT_ROOT / "data" / "Anomaly-Videos-Part-1 (1)" / "Anomaly-Videos-Part-1"
    if ucf_dir.exists():
        for category in ["Arson", "Assault", "Abuse", "Arrest"]:
            cat_folder = ucf_dir / category
            if cat_folder.exists():
                for sample_v in sorted(cat_folder.glob("*.mp4"))[:3]: # Add top 3 of each category
                    meta = get_video_metadata(sample_v)
                    meta["filename"] = f"{category}/{sample_v.name}"
                    meta["is_ucf_crime"] = True
                    videos.append(meta)

    return videos

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_testcase_video(file: UploadFile = File(...)):
    """Upload a new video file to serve as a surveillance testcase."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    clean_name = Path(file.filename).name.replace(" ", "_")
    target_path = VIDEO_DIR / clean_name

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    meta = get_video_metadata(target_path)
    return {
        "message": f"Testcase video '{clean_name}' uploaded successfully.",
        "video": meta
    }

@router.delete("/{filename:path}")
def delete_testcase_video(filename: str):
    """Delete a testcase video from disk."""
    target_path = VIDEO_DIR / filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    try:
        target_path.unlink()
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {e}")

def draw_zones(frame: np.ndarray, restricted_poly: np.ndarray, loitering_poly: np.ndarray):
    """Renders semi-transparent zone overlays and borders."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [loitering_poly], (0, 215, 255))
    cv2.fillPoly(overlay, [restricted_poly], (0, 0, 230))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [loitering_poly], True, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [restricted_poly], True, (0, 0, 255), 2, cv2.LINE_AA)

def ai_video_stream_generator(video_path: Path):
    """
    Real-time AI-powered stream generator:
    Runs YOLOv11 + ByteTrack + Behaviour Analysis + UCF-Crime Anomaly Classifier
    on each frame of the video, renders overlays, generates instant alerts,
    and yields MJPEG streaming frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / max(10.0, min(fps, 30.0))

    # Initialize enterprise behaviour engine for this stream session
    behaviour = BehaviourEngine()

    frame_idx = 0
    active_alert_message = ""
    active_alert_until_frame = 0
    last_crime_eval_frame = -30
    last_crime_alert_time = 0.0
    latest_crime_label = ""
    video_source_id = video_path.name

    # Check if video is from UCF-Crime category
    v_stem = video_path.stem.lower()
    inferred_category = None
    for cat in ["arson", "assault", "abuse", "arrest"]:
        if cat in v_stem:
            inferred_category = cat
            break

    while True:
        start_t = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            # Loop video seamlessly for continuous surveillance inspection
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = 0

        # Downscale large frames to standard 640px width for fast real-time inference
        if frame.shape[1] > 640:
            scale = 640.0 / frame.shape[1]
            frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

        frame_idx += 1
        video_time = frame_idx / fps
        now = time.monotonic()

        # 1. Track objects using YOLOv11 & ByteTrack
        tracked = ai_engine_pool.tracker.track_frame(frame)

        # 2. Evaluate Spatial Behaviour (Restricted Zone, Loitering, Abandoned Luggage)
        alerts = behaviour.evaluate_frame(tracked, now=video_time)

        # 3. Process Spatial Behaviour Alerts
        for al in alerts:
            activity = al["activity_type"]
            trk_id = al["track_id"]
            conf = al["confidence"]
            bbox = al["bbox"]
            details = al.get("details", {})

            # Save snapshot evidence to disk
            ev_info = save_evidence_snapshot(
                frame=frame,
                activity_type=activity,
                track_id=trk_id,
                bbox=bbox,
                details=details
            )

            # Insert alert into database immediately so it pops up on the React dashboard
            db_manager.insert_event({
                "camera_id": video_source_id,
                "activity_type": activity,
                "timestamp": ev_info["timestamp"],
                "track_id": trk_id,
                "confidence": round(float(conf), 2),
                "snapshot_path": ev_info["snapshot_path"],
                "snapshot_url": f"/evidence/{ev_info['filename']}",
                "details": details,
                "status": "pending_review"
            })

            active_alert_message = f"🚨 {activity.upper().replace('_', ' ')} (ID: #{trk_id})"
            active_alert_until_frame = frame_idx + int(fps * 3.5)

        # 4. Evaluate Crime & Violent Anomaly using Trained Model (every 12 frames)
        if (frame_idx - last_crime_eval_frame) >= 12:
            last_crime_eval_frame = frame_idx
            crime_pred = ai_engine_pool.anomaly_clf.classify_frame(frame)
            if crime_pred:
                latest_crime_label = f"{crime_pred['class_name']} ({crime_pred['confidence']*100:.0f}%)"

                # Trigger crime alert if high confidence or matching crime video category
                is_trigger = crime_pred.get("is_anomaly", False)
                if inferred_category and inferred_category in crime_pred["class_name"].lower() and crime_pred["confidence"] > 0.35:
                    is_trigger = True

                if is_trigger and (now - last_crime_alert_time > 20.0):
                    last_crime_alert_time = now
                    c_act = f"crime_{crime_pred['class_name'].lower()}"
                    ev_info = save_evidence_snapshot(
                        frame=frame,
                        activity_type=c_act,
                        details={"crime": crime_pred["class_name"], "confidence": crime_pred["confidence"]}
                    )
                    db_manager.insert_event({
                        "camera_id": video_source_id,
                        "activity_type": c_act,
                        "timestamp": ev_info["timestamp"],
                        "confidence": crime_pred["confidence"],
                        "snapshot_path": ev_info["snapshot_path"],
                        "snapshot_url": f"/evidence/{ev_info['filename']}",
                        "details": {"crime_class": crime_pred["class_name"], "source_video": video_source_id},
                        "status": "pending_review"
                    })
                    active_alert_message = f"🚨 CRIME DETECTED: {crime_pred['class_name'].upper()}"
                    active_alert_until_frame = frame_idx + int(fps * 3.5)

        # 5. Render Visual HUD

        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            tid = obj["track_id"]
            cname = obj["class_name"]
            c_score = obj["conf"]
            col = (0, 255, 0)
            if frame_idx < active_alert_until_frame and f"#{tid}" in active_alert_message:
                col = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.circle(frame, (int((x1 + x2) / 2), y2), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"#{tid} {cname} {c_score:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Header telemetry bar
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 22), -1)
        crime_hud = f" | {latest_crime_label}" if latest_crime_label else ""
        hud_text = f"SRC: {video_source_id} | TC: {video_time:04.1f}s | TRACKS: {len(tracked)}{crime_hud}"
        cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Flashing Alert Banner
        if frame_idx < active_alert_until_frame:
            cv2.rectangle(frame, (0, 39), (w, 75), (0, 0, 225), -1)
            cv2.putText(frame, active_alert_message, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ret_enc:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        elapsed = time.monotonic() - start_t
        time.sleep(max(0.002, frame_delay - elapsed))

    cap.release()

@router.get("/{filename:path}/stream")
def stream_testcase_video(filename: str):
    """Stream an uploaded video testcase with live AI detection, behaviour analysis, and real-time alerts."""
    # Check in test_videos
    target_path = VIDEO_DIR / filename
    if not target_path.exists():
        # Check in UCF-Crime directory
        ucf_path = PROJECT_ROOT / "data" / "Anomaly-Videos-Part-1 (1)" / "Anomaly-Videos-Part-1" / filename
        if ucf_path.exists():
            target_path = ucf_path
        else:
            raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")

    return StreamingResponse(
        ai_video_stream_generator(target_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
