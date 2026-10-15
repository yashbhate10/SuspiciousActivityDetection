import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ai-engine"))

from anomaly_classifier import AnomalyClassifier

def test_anomaly_classification():
    print("Testing AnomalyClassifier on UCF-Crime validation image...")
    clf = AnomalyClassifier()
    assert clf.is_ready, "AnomalyClassifier failed to load weights"

    # Test on a validation image
    val_img_path = PROJECT_ROOT / "data" / "anomaly_classification" / "val" / "Arson"
    val_files = list(val_img_path.glob("*.jpg"))
    if val_files:
        sample_img = cv2.imread(str(val_files[0]))
        pred = clf.classify_frame(sample_img)
        print(f"[PASS] Anomaly prediction on {val_files[0].name}: {pred}")
        assert pred is not None
        assert "class_name" in pred

if __name__ == "__main__":
    test_anomaly_classification()
