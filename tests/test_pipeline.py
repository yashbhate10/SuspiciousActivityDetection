import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Add ai-engine to system path
AI_ENGINE_DIR = Path(__file__).resolve().parent.parent / "ai-engine"
sys.path.insert(0, str(AI_ENGINE_DIR))

from behaviour import BehaviourEngine
from evidence import save_evidence_snapshot
from config import RESTRICTED_ZONE_POLYGON, LOITERING_ZONE_POLYGON

def test_restricted_zone_trespassing():
    print("Testing Restricted Zone Entry detection...")
    engine = BehaviourEngine(
        restricted_polygon=RESTRICTED_ZONE_POLYGON,
        loitering_polygon=LOITERING_ZONE_POLYGON
    )

    # Simulated person entering restricted zone: box center at (300, 250), bottom at (300, 300)
    # Foot point (300, 300) is well inside [(120, 100), (520, 100), (520, 400), (120, 400)]
    tracked_person = [{
        "track_id": 1,
        "class_id": 0,
        "class_name": "person",
        "bbox": (280, 200, 320, 300),
        "conf": 0.92
    }]

    now = 1000.0
    alerts = []
    # Send 5 consecutive frames to exceed debounce threshold
    for i in range(5):
        alerts = engine.evaluate_frame(tracked_person, now=now + (i * 0.1))

    assert len(alerts) == 1, f"Expected 1 trespassing alert after 5 consecutive frames, got {len(alerts)}"
    assert alerts[0]["activity_type"] == "restricted_zone_entry"
    assert alerts[0]["track_id"] == 1
    print("[PASS] Restricted Zone Entry test PASSED!")

def test_loitering_detection():
    print("Testing Loitering dwell time detection...")
    engine = BehaviourEngine(
        restricted_polygon=RESTRICTED_ZONE_POLYGON,
        loitering_polygon=LOITERING_ZONE_POLYGON
    )

    # Foot point at (80, 80) is inside loitering zone (60..580) but OUTSIDE restricted zone (120..520)
    tracked_person = [{
        "track_id": 2,
        "class_id": 0,
        "class_name": "person",
        "bbox": (70, 40, 90, 80),
        "conf": 0.89
    }]

    # Frame at t = 0
    t0 = 2000.0
    alerts_0 = engine.evaluate_frame(tracked_person, now=t0)
    assert len(alerts_0) == 0, "Should not alert immediately upon entry"

    # Frame at t = 10s (under 15s threshold)
    alerts_10 = engine.evaluate_frame(tracked_person, now=t0 + 10.0)
    assert len(alerts_10) == 0, "Should not alert before threshold is reached"

    # Frame at t = 16s (exceeds 15s threshold)
    alerts_16 = engine.evaluate_frame(tracked_person, now=t0 + 16.0)
    assert len(alerts_16) == 1, f"Expected loitering alert after 16s dwell, got {len(alerts_16)}"
    assert alerts_16[0]["activity_type"] == "loitering"
    assert alerts_16[0]["track_id"] == 2
    print("[PASS] Loitering detection test PASSED!")

def test_abandoned_object_detection():
    print("Testing Abandoned Object detection...")
    engine = BehaviourEngine(
        restricted_polygon=RESTRICTED_ZONE_POLYGON,
        loitering_polygon=LOITERING_ZONE_POLYGON
    )

    # Bag at (200, 200)
    # Initially person is standing right next to bag at (210, 200)
    t0 = 3000.0
    tracked_together = [
        {"track_id": 10, "class_id": 24, "class_name": "backpack", "bbox": (190, 190, 210, 210), "conf": 0.85},
        {"track_id": 3, "class_id": 0, "class_name": "person", "bbox": (200, 150, 240, 250), "conf": 0.90}
    ]
    alerts = engine.evaluate_frame(tracked_together, now=t0)
    assert len(alerts) == 0, "Bag with person nearby should not be flagged"

    # At t = 12s, owner walked far away to (500, 200) (distance > 160 px), bag remains at (200, 200)
    tracked_apart = [
        {"track_id": 10, "class_id": 24, "class_name": "backpack", "bbox": (190, 190, 210, 210), "conf": 0.85},
        {"track_id": 3, "class_id": 0, "class_name": "person", "bbox": (500, 150, 540, 250), "conf": 0.90}
    ]
    alerts_abandoned = engine.evaluate_frame(tracked_apart, now=t0 + 12.0)
    assert len(alerts_abandoned) == 1, f"Expected abandoned luggage alert, got {len(alerts_abandoned)}"
    assert alerts_abandoned[0]["activity_type"] == "abandoned_object"
    assert alerts_abandoned[0]["track_id"] == 10
    print("[PASS] Abandoned object detection test PASSED!")

def test_evidence_snapshot():
    print("Testing Evidence Snapshot generation...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (50, 50, 50)

    result = save_evidence_snapshot(
        frame=dummy_frame,
        activity_type="restricted_zone_entry",
        track_id=99,
        bbox=(150, 120, 250, 320),
        details={"test": True}
    )

    saved_path = Path(result["snapshot_path"])
    assert saved_path.exists(), f"Snapshot file not created at {saved_path}"
    assert saved_path.stat().st_size > 0, "Saved snapshot is empty"

    # Verify image is valid and readable
    loaded_img = cv2.imread(str(saved_path))
    assert loaded_img is not None
    assert loaded_img.shape == (480, 640, 3)
    print(f"[PASS] Evidence snapshot saved and verified at: {saved_path.name}")

if __name__ == "__main__":
    print("\n--- Running AI Suspicious Activity Detection Test Suite ---")
    test_restricted_zone_trespassing()
    test_loitering_detection()
    test_abandoned_object_detection()
    test_evidence_snapshot()
    print("\n[SUCCESS] ALL UNIT AND PIPELINE TESTS PASSED SUCCESSFULLY!\n")
