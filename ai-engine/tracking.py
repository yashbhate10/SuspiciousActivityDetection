import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, TRACKER_CONFIG

logger = logging.getLogger("ai_engine.tracking")

class ObjectTracker:
    def __init__(self, model_path: str = MODEL_PATH, conf: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.conf = conf
        logger.info(f"Loading YOLO11 model from {model_path}...")
        self.model = YOLO(model_path)
        logger.info("YOLO11 model loaded successfully.")

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
        """
        Runs YOLO11 detection and ByteTrack tracking on a single frame.
        Returns a list of structured tracked objects.
        """
        results = self.model.track(
            frame,
            persist=persist,
            tracker=TRACKER_CONFIG,
            conf=self.conf,
            imgsz=640,
            verbose=False
        )

        tracked_objects = []
        if not results or len(results) == 0:
            return tracked_objects

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return tracked_objects

        # Extract boxes, ids, classes, confidences
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        # ByteTrack assigns ids; if tracker is warming up or object has no id yet, assign -1
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(len(classes), -1)
        )

        names = res.names
        for i in range(len(classes)):
            trk_id = int(track_ids[i])
            if trk_id < 0:
                # If tracker hasn't assigned an ID yet, use temporary fallback
                continue

            cls_id = int(classes[i])
            cls_name = names.get(cls_id, f"class_{cls_id}")
            bbox = tuple(xyxy[i].tolist())
            conf_val = float(confs[i])

            tracked_objects.append({
                "track_id": trk_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "bbox": bbox,
                "conf": conf_val
            })

        return tracked_objects
