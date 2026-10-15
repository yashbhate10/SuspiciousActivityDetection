import sys
import time
from pathlib import Path
import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parent.parent / "ai-engine"
sys.path.insert(0, str(AI_ENGINE_DIR))

from tracking import ObjectTracker
from behaviour import BehaviourEngine
from evidence import save_evidence_snapshot
from config import RESTRICTED_ZONE_POLYGON, LOITERING_ZONE_POLYGON

def test_simulation_frames():
    print("Testing 10 simulation frames through tracker and behaviour engine...")
    tracker = ObjectTracker()
    behaviour = BehaviourEngine(
        restricted_polygon=RESTRICTED_ZONE_POLYGON,
        loitering_polygon=LOITERING_ZONE_POLYGON
    )

    for i in range(10):
        # Synthetic frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 35)

        tracked = tracker.track_frame(frame)
        alerts = behaviour.evaluate_frame(tracked, now=time.monotonic())
        time.sleep(0.01)

    print("[PASS] 10 frames processed seamlessly through YOLO11 ByteTrack & BehaviourEngine!")

if __name__ == "__main__":
    test_simulation_frames()
