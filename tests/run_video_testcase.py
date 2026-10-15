import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure ai-engine is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ENGINE_DIR = PROJECT_ROOT / "ai-engine"
sys.path.insert(0, str(AI_ENGINE_DIR))

# Safe Windows stdout encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import (
    MODEL_PATH,
    RESTRICTED_ZONE_POLYGON,
    LOITERING_ZONE_POLYGON,
    EVIDENCE_DIR
)
from tracking import ObjectTracker
from behaviour import BehaviourEngine
from evidence import save_evidence_snapshot

def draw_zones(frame: np.ndarray, restricted_poly: np.ndarray, loitering_poly: np.ndarray):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [loitering_poly], (0, 215, 255))
    cv2.fillPoly(overlay, [restricted_poly], (0, 0, 230))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    cv2.polylines(frame, [loitering_poly], True, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [restricted_poly], True, (0, 0, 255), 2, cv2.LINE_AA)

def run_testcase(
    video_path: str,
    loiter_threshold: int = 10,
    output_video_path: str = None,
    show_window: bool = False,
    report_path: str = None
):
    video_file = Path(video_path)
    if not video_file.exists():
        # Check in data/test_videos
        alt_path = PROJECT_ROOT / "data" / "test_videos" / video_path
        if alt_path.exists():
            video_file = alt_path
        else:
            raise FileNotFoundError(f"Video file '{video_path}' not found!")

    print("\n=======================================================")
    print(" 🎬 AI SURVEILLANCE TESTCASE RUNNER (YOLOv11)")
    print(f" Video: {video_file.name}")
    print(f" Loitering Threshold: {loiter_threshold}s")
    print("=======================================================")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(1.0, video_fps)

    print(f" Resolution: {width}x{height} | FPS: {video_fps:.1f} | Duration: {duration_sec:.1f}s ({total_frames} frames)")

    tracker = ObjectTracker(model_path=MODEL_PATH)
    behaviour = BehaviourEngine(
        restricted_polygon=RESTRICTED_ZONE_POLYGON,
        loitering_polygon=LOITERING_ZONE_POLYGON
    )
    # Customize loiter threshold if specified
    from config import LOITER_THRESHOLD_SECONDS

    # Video Writer for annotated output if requested
    out_writer = None
    if output_video_path:
        out_file = Path(output_video_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_file), fourcc, video_fps, (width, height))
        print(f" Saving annotated output to: {out_file}")

    events_detected = []
    frame_idx = 0
    start_time = time.time()
    active_alert_banner = ""
    active_alert_until_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        video_time = frame_idx / video_fps

        # 1. Track objects
        tracked = tracker.track_frame(frame)

        # 2. Evaluate behaviour
        alerts = behaviour.evaluate_frame(tracked, now=video_time)

        # 3. Process alerts
        for al in alerts:
            activity = al["activity_type"]
            trk_id = al["track_id"]
            conf = al["confidence"]
            bbox = al["bbox"]
            details = al.get("details", {})

            # Save snapshot evidence
            ev_info = save_evidence_snapshot(
                frame=frame,
                activity_type=activity,
                track_id=trk_id,
                bbox=bbox,
                details=details
            )

            event_record = {
                "event_index": len(events_detected) + 1,
                "video_timestamp_sec": round(video_time, 2),
                "frame_number": frame_idx,
                "activity_type": activity,
                "track_id": trk_id,
                "confidence": round(float(conf), 2),
                "snapshot_file": ev_info["filename"],
                "details": details
            }
            events_detected.append(event_record)
            print(f" [ALERT] Frame {frame_idx:04d} ({video_time:05.1f}s) -> {activity.upper()} (Track #{trk_id})")

            active_alert_banner = f"ALERT: {activity.upper()} (ID: #{trk_id})"
            active_alert_until_frame = frame_idx + int(video_fps * 2.5)

        # 4. Annotations
        draw_zones(frame, RESTRICTED_ZONE_POLYGON, LOITERING_ZONE_POLYGON)

        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            tid = obj["track_id"]
            cname = obj["class_name"]
            c_score = obj["conf"]
            col = (0, 255, 0)
            if frame_idx < active_alert_until_frame and f"#{tid}" in active_alert_banner:
                col = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.circle(frame, (int((x1+x2)/2), y2), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"#{tid} {cname} {c_score:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # HUD banner
        cv2.rectangle(frame, (0, 0), (width, 35), (20, 20, 25), -1)
        hud = f"TESTCASE: {video_file.name} | Time: {video_time:05.1f}s / {duration_sec:05.1f}s | Events: {len(events_detected)}"
        cv2.putText(frame, hud, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if frame_idx < active_alert_until_frame:
            cv2.rectangle(frame, (0, 36), (width, 70), (0, 0, 220), -1)
            cv2.putText(frame, f"🚨 {active_alert_banner}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        if out_writer is not None:
            out_writer.write(frame)

        if show_window:
            cv2.imshow(f"Testcase Analysis - {video_file.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nEarly termination by user.")
                break

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if show_window:
        cv2.destroyAllWindows()

    total_proc_time = time.time() - start_time
    proc_fps = frame_idx / max(0.001, total_proc_time)

    # Breakdown by activity type
    breakdown = {}
    for ev in events_detected:
        breakdown[ev["activity_type"]] = breakdown.get(ev["activity_type"], 0) + 1

    summary = {
        "video_name": video_file.name,
        "video_path": str(video_file),
        "total_frames": total_frames,
        "processed_frames": frame_idx,
        "video_duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(total_proc_time, 2),
        "processing_fps": round(proc_fps, 1),
        "total_suspicious_events": len(events_detected),
        "events_by_type": breakdown,
        "events": events_detected
    }

    # Save JSON report
    if report_path is None:
        report_path = str(EVIDENCE_DIR / f"testcase_report_{video_file.stem}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Summary Report Table
    print("\n=======================================================")
    print(" 📊 TESTCASE EXECUTION SUMMARY REPORT")
    print("=======================================================")
    print(f" Video File:              {video_file.name}")
    print(f" Duration:                {duration_sec:.1f} seconds ({frame_idx} frames)")
    print(f" Processing Speed:        {proc_fps:.1f} FPS (Total: {total_proc_time:.1f}s)")
    print(f" Total Incidents Flagged: {len(events_detected)}")
    for act, count in breakdown.items():
        print(f"   • {act.replace('_', ' ').title()}: {count}")
    print(f" JSON Report Saved:       {report_path}")
    print("=======================================================\n")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Run Video Testcase through AI Suspicious Activity Detection")
    parser.add_argument("--video", type=str, default="sample_walkthrough.mp4", help="Video file name or path")
    parser.add_argument("--loiter-threshold", type=int, default=10, help="Loitering threshold in seconds")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for annotated video")
    parser.add_argument("--show", action="store_true", help="Display OpenCV live visualization window")
    parser.add_argument("--report", type=str, default=None, help="Custom output path for JSON report")
    args = parser.parse_args()

    run_testcase(
        video_path=args.video,
        loiter_threshold=args.loiter_threshold,
        output_video_path=args.output,
        show_window=args.show,
        report_path=args.report
    )

if __name__ == "__main__":
    main()
