"""
Enterprise Real-Time Suspicious Activity Detection Engine.

Pipeline:
  Video Frame → YOLOv11 + ByteTrack → Enterprise Behaviour Engine
             → Crime Classifier (optional, requires retrained 5-class model)
             → Evidence Capture → Backend Alert Dispatch

Run modes:
  python detect.py --source 0                        # webcam
  python detect.py --source video.mp4                # file
  python detect.py --source sim                      # synthetic demo
  python detect.py --source 0 --enable-crime-clf     # with crime classifier
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_PATH, CAMERA_ID
from tracking import ObjectTracker
from behaviour import BehaviourEngine
from evidence import save_evidence_snapshot
from alert_service import alert_dispatcher

# ── Severity colour map for OpenCV overlays ───────────────────────────
SEVERITY_COLOUR = {
    "low":    (0, 200, 0),    # green
    "medium": (0, 165, 255),  # orange
    "high":   (0, 0, 220),    # red
    "crime":  (0, 0, 255),    # bright red
}

ALERT_LABEL_COLOUR = {
    "loitering":        (0, 165, 255),
    "erratic_movement": (0, 165, 255),
    "rapid_approach":   (0, 0, 220),
    "abandoned_object": (0, 0, 220),
    "crowd_surge":      (0, 165, 255),
}


def draw_alert_banner(frame: np.ndarray, message: str, severity: str = "high"):
    """Draws a coloured top banner with the alert message."""
    colour = SEVERITY_COLOUR.get(severity, (0, 0, 220))
    cv2.rectangle(frame, (0, 42), (frame.shape[1], 82), colour, -1)
    # Pulsing border
    cv2.rectangle(frame, (0, 42), (frame.shape[1], 82), (255, 255, 255), 1)
    cv2.putText(
        frame, f"  {message}",
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_track(frame, obj, alert_ids):
    """Draw bounding box and label for a tracked object."""
    x1, y1, x2, y2 = map(int, obj["bbox"])
    trk_id  = obj["track_id"]
    cls_name = obj["class_name"]
    conf    = obj["conf"]

    in_alert = trk_id in alert_ids
    colour   = (0, 0, 220) if in_alert else (0, 220, 0)
    thickness = 3 if in_alert else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    foot_x = (x1 + x2) // 2
    cv2.circle(frame, (foot_x, y2), 5, (0, 255, 255), -1)

    label = f"#{trk_id} {cls_name} {conf:.2f}"
    label_bg_y1 = max(0, y1 - 22)
    cv2.rectangle(frame, (x1, label_bg_y1), (x1 + len(label) * 9, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Suspicious Activity Detection")
    parser.add_argument("--source", default="0",
                        help="Video source: 0=webcam, path=video file, sim=synthetic")
    parser.add_argument("--no-show", action="store_true", help="Headless mode (no OpenCV window)")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream annotated frames to backend")
    parser.add_argument("--enable-crime-clf", action="store_true",
                        help="Enable crime classifier (requires retrained 5-class model)")
    parser.add_argument("--crime-conf", type=float, default=0.85,
                        help="Minimum confidence for crime classifier alerts")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 65)
    print("  ENTERPRISE AI SURVEILLANCE — YOLOv11 + ByteTrack")
    print(f"  Source : {source}")
    print(f"  Behaviours: Loitering | Erratic Movement | Rapid Approach")
    print(f"              Abandoned Object | Crowd Surge")
    crime_status = "ENABLED" if args.enable_crime_clf else "DISABLED (pass --enable-crime-clf)"
    print(f"  Crime Clf : {crime_status}")
    print("=" * 65)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Warning] Cannot open source '{source}'. Using synthetic feed.")
        cap = None

    tracker  = ObjectTracker(model_path=MODEL_PATH)
    behaviour = BehaviourEngine()

    # Crime classifier — only if explicitly enabled AND model has Normal class
    anomaly_clf = None
    if args.enable_crime_clf:
        from anomaly_classifier import AnomalyClassifier
        clf = AnomalyClassifier(conf_threshold=args.crime_conf)
        if clf.is_ready:
            anomaly_clf = clf
            print("  Crime classifier loaded.")
        else:
            print("  Crime classifier failed to load — behaviour-only mode.")

    # Runtime state
    active_alerts: deque = deque(maxlen=5)   # (message, severity, expiry)
    last_crime_eval  = 0.0
    last_crime_alert = 0.0
    latest_crime_tag = ""

    fps_history: deque = deque(maxlen=30)
    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            # ── Frame acquisition ─────────────────────────────────────
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
            else:
                # Synthetic feed — simulated loitering person
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (35, 35, 40)
                for y in range(0, 480, 60):
                    cv2.line(frame, (0, y), (640, y), (50, 50, 55), 1)
                for x in range(0, 640, 60):
                    cv2.line(frame, (x, 0), (x, 480), (50, 50, 55), 1)
                sim_t = time.time()
                cx = int(320 + 30 * np.sin(sim_t * 0.15))   # small movement = loitering
                cy = int(240 + 15 * np.cos(sim_t * 0.15))
                cv2.circle(frame, (cx, cy - 30), 16, (200, 180, 150), -1)
                cv2.rectangle(frame, (cx - 20, cy - 14), (cx + 20, cy + 45), (180, 100, 50), -1)
                time.sleep(0.03)

            now = time.monotonic()
            frame_count += 1
            curr_time = time.time()
            fps_history.append(1.0 / max(0.001, curr_time - prev_time))
            prev_time = curr_time
            fps = sum(fps_history) / len(fps_history)

            # ── Detection & Tracking ──────────────────────────────────
            tracked_objects = tracker.track_frame(frame)

            # ── Behaviour Analysis ────────────────────────────────────
            new_alerts = behaviour.evaluate_frame(tracked_objects, now=now)
            alert_track_ids = set()

            for alert in new_alerts:
                activity  = alert["activity_type"]
                trk_id    = alert["track_id"]
                conf      = alert["confidence"]
                bbox      = alert["bbox"]
                details   = alert.get("details", {})
                severity  = alert.get("severity", "medium")

                if trk_id > 0:
                    alert_track_ids.add(trk_id)

                # Save evidence snapshot
                ev = save_evidence_snapshot(
                    frame=frame,
                    activity_type=activity,
                    track_id=trk_id if trk_id > 0 else None,
                    bbox=bbox,
                    details=details
                )

                # Dispatch to backend
                alert_dispatcher.dispatch_alert(
                    activity_type=activity,
                    timestamp=ev["timestamp"],
                    snapshot_path=ev["snapshot_path"],
                    track_id=trk_id if trk_id > 0 else None,
                    confidence=conf,
                    details=details,
                    camera_id=CAMERA_ID
                )

                label = activity.upper().replace("_", " ")
                trk_label = f" (ID: #{trk_id})" if trk_id > 0 else ""
                conf_label = f" [{conf*100:.0f}%]"
                msg = f"🚨 {label}{trk_label}{conf_label}"

                active_alerts.append((msg, severity, now + 5.0))
                print(f"  [ALERT] {msg} — {details.get('description', '')}")

            # ── Crime Classifier ──────────────────────────────────────
            if anomaly_clf and (now - last_crime_eval > 2.0):
                last_crime_eval = now
                crime_res = anomaly_clf.classify_frame(frame)
                if crime_res:
                    cls   = crime_res["class_name"]
                    cconf = crime_res["confidence"]
                    streak = crime_res.get("streak", 0)
                    latest_crime_tag = f"{cls} {cconf*100:.0f}% (x{streak})"

                    if crime_res["is_anomaly"] and (now - last_crime_alert > 60.0):
                        last_crime_alert = now
                        c_type = f"crime_{cls.lower()}"
                        ev = save_evidence_snapshot(
                            frame=frame,
                            activity_type=c_type,
                            details={"crime_class": cls, "confidence": cconf}
                        )
                        alert_dispatcher.dispatch_alert(
                            activity_type=c_type,
                            timestamp=ev["timestamp"],
                            snapshot_path=ev["snapshot_path"],
                            confidence=cconf,
                            details={"crime_class": cls},
                            camera_id=CAMERA_ID
                        )
                        msg = f"🚨 CRIME: {cls.upper()} [{cconf*100:.0f}%]"
                        active_alerts.append((msg, "crime", now + 6.0))
                        print(f"  [CRIME ALERT] {msg}")

            # ── HUD & Overlays ────────────────────────────────────────
            # Top status bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (12, 12, 20), -1)
            hud = (f"CAM: {CAMERA_ID}  |  FPS: {fps:.1f}  "
                   f"|  TRACKS: {len(tracked_objects)}"
                   f"|  ALERTS: {len(active_alerts)}")
            if latest_crime_tag:
                hud += f"  |  CLF: {latest_crime_tag}"
            cv2.putText(frame, hud, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Bounding boxes
            for obj in tracked_objects:
                draw_track(frame, obj, alert_track_ids)

            # Active alert banners (highest severity first)
            live = [(m, s, e) for m, s, e in active_alerts if now < e]
            active_alerts.clear()
            active_alerts.extend(live)
            if live:
                sev_order = {"crime": 0, "high": 1, "medium": 2, "low": 3}
                top = sorted(live, key=lambda x: sev_order.get(x[1], 9))[0]
                draw_alert_banner(frame, top[0], top[1])

            # Alert count badge
            if len(live) > 1:
                badge = f"+{len(live)-1} more"
                cv2.putText(frame, badge, (frame.shape[1] - 120, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # ── Stream & Display ──────────────────────────────────────
            if args.stream and (frame_count % 2 == 0):
                ret_enc, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ret_enc:
                    alert_dispatcher.push_frame_to_stream(buf.tobytes())

            if not args.no_show:
                cv2.imshow("Enterprise AI Surveillance (YOLOv11 + ByteTrack)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n  Stopping...")
    finally:
        if cap:
            cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        alert_dispatcher.shutdown()
        print("  Detection engine stopped.")


if __name__ == "__main__":
    main()
