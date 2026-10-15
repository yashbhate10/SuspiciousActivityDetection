"""
Enterprise Crime Classifier — Inference Module v3.

5-class model: Normal, Abuse, Arrest, Arson, Assault.

The model NOW has a Normal class, so we can use straightforward logic:
  - If top prediction = "Normal" → no alert
  - If top prediction = crime class AND confidence >= threshold → potential alert
  - Temporal consistency: require N consecutive crime predictions before firing
    (prevents single-frame false positives from noisy scenes)

Key anti-false-alarm safeguards:
  1. High confidence threshold (default 0.65 for 5-class model)
  2. Temporal streak: same crime class must be predicted 4+ times consecutively
  3. Normal-class margin: if Normal probability is within margin of top crime,
     suppress the alert (model is uncertain)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CUSTOM_MODEL_PATH = MODELS_DIR / "anomaly_yolo11_best.pt"


class AnomalyClassifier:
    """
    5-class UCF-Crime frame classifier with temporal consistency.
    
    Classes: Normal (0), Abuse (1), Arrest (2), Arson (3), Assault (4)
    """

    # Require this many consecutive same-crime predictions to fire an alert
    CONSECUTIVE_REQUIRED = 4

    # If Normal probability is within this margin of the top crime prob,
    # treat as uncertain and suppress the alert
    NORMAL_MARGIN = 0.15

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.65):
        self.conf_threshold = conf_threshold

        if model_path is None:
            if CUSTOM_MODEL_PATH.exists():
                self.model_path = str(CUSTOM_MODEL_PATH)
            else:
                self.model_path = None
        else:
            self.model_path = model_path

        # Temporal consistency state
        self._streak_class: Optional[str] = None
        self._streak_count: int = 0

        # Track class name → index mapping
        self._normal_idx: Optional[int] = None

        try:
            if self.model_path is None:
                print("  ⚠️ No anomaly model found. Crime classifier disabled.")
                self.model = None
                self.is_ready = False
                return

            self.model = YOLO(self.model_path)
            self.is_ready = True

            # Detect Normal class index
            if hasattr(self.model, 'names'):
                for idx, name in self.model.names.items():
                    if name.lower() == "normal":
                        self._normal_idx = idx
                        break

            if self._normal_idx is not None:
                print(f"  ✅ Crime classifier loaded (5-class, Normal=idx {self._normal_idx})")
            else:
                print("  ⚠️ Crime classifier loaded but NO Normal class found in model!")
                print("     This will cause false positives. Retrain with Normal class.")

        except Exception as e:
            print(f"  ⚠️ Could not load anomaly classifier ({e}).")
            self.model = None
            self.is_ready = False

    def classify_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Classify a frame for criminal activity.

        Returns dict with:
          - class_name: predicted class
          - confidence: prediction confidence
          - is_anomaly: True ONLY when temporal consistency is met AND
                        the prediction is NOT Normal
          - streak: number of consecutive same-class predictions
          - normal_prob: probability of Normal class (for debugging)
        """
        if not self.is_ready or self.model is None:
            return None

        try:
            results = self.model.predict(frame, imgsz=224, verbose=False)
            if not results or len(results) == 0:
                return None

            res = results[0]
            probs = res.probs
            if probs is None:
                return None

            prob_data = probs.data.cpu().numpy()
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf.cpu().numpy())
            class_name = res.names.get(top1_idx, f"Class_{top1_idx}")

            # Get Normal class probability
            normal_prob = 0.0
            if self._normal_idx is not None and self._normal_idx < len(prob_data):
                normal_prob = float(prob_data[self._normal_idx])

            # ── Decision Logic ──────────────────────────────────────
            is_crime_prediction = class_name.lower() != "normal"

            # Check 1: Is it a crime prediction with enough confidence?
            confident_crime = (
                is_crime_prediction
                and top1_conf >= self.conf_threshold
            )

            # Check 2: Is Normal probability low enough?
            # If Normal is close to the crime prediction, the model is uncertain
            normal_suppressed = False
            if confident_crime and self._normal_idx is not None:
                if normal_prob > (top1_conf - self.NORMAL_MARGIN):
                    normal_suppressed = True
                    confident_crime = False

            # ── Temporal consistency ────────────────────────────────
            if confident_crime and class_name == self._streak_class:
                self._streak_count += 1
            elif confident_crime:
                self._streak_class = class_name
                self._streak_count = 1
            else:
                # Normal prediction or low confidence → break streak
                self._streak_class = None
                self._streak_count = 0

            is_anomaly = self._streak_count >= self.CONSECUTIVE_REQUIRED

            return {
                "class_id": top1_idx,
                "class_name": class_name,
                "confidence": round(top1_conf, 3),
                "is_anomaly": is_anomaly,
                "streak": self._streak_count,
                "normal_prob": round(normal_prob, 3),
                "suppressed": normal_suppressed,
            }

        except Exception:
            return None
