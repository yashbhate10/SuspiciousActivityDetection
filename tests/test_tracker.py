import sys
from pathlib import Path
import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parent.parent / "ai-engine"
sys.path.insert(0, str(AI_ENGINE_DIR))

from tracking import ObjectTracker

def test_yolo_tracker():
    print("Testing YOLO11 ByteTrack object tracking on dummy frame...")
    tracker = ObjectTracker()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (40, 40, 40)

    # Run tracking
    results = tracker.track_frame(dummy_frame)
    print(f"[PASS] YOLO11 Tracker initialized and ran inference successfully. Detections on empty frame: {len(results)}")

if __name__ == "__main__":
    test_yolo_tracker()
