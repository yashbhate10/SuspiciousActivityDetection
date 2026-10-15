import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from config import EVIDENCE_DIR

def save_evidence_snapshot(
    frame: np.ndarray,
    activity_type: str,
    track_id: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Saves an annotated evidence snapshot with visual overlay.
    Returns ISO timestamp and saved file path.
    """
    timestamp = datetime.now(timezone.utc)
    time_str = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
    track_str = f"_trk{track_id}" if track_id is not None else ""
    filename = f"{time_str}_{activity_type}{track_str}.jpg"
    filepath = EVIDENCE_DIR / filename

    # Create a visual copy for evidence inspection
    evidence_img = frame.copy()

    # Draw evidence watermark banner on top
    h, w = evidence_img.shape[:2]
    cv2.rectangle(evidence_img, (0, 0), (w, 55), (20, 20, 35), -1)
    cv2.line(evidence_img, (0, 55), (w, 55), (0, 0, 255), 2)

    alert_title = f"SECURITY ALERT: {activity_type.upper().replace('_', ' ')}"
    cv2.putText(
        evidence_img,
        alert_title,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 70, 255),
        2,
        cv2.LINE_AA
    )

    info_str = f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} | Track ID: {track_id or 'N/A'}"
    cv2.putText(
        evidence_img,
        info_str,
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Highlight bbox if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(evidence_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(
            evidence_img,
            f"SUSPECT [{activity_type}]",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Save to disk
    success = cv2.imwrite(str(filepath), evidence_img)
    if not success:
        raise RuntimeError(f"Could not write evidence image to {filepath}")

    return {
        "timestamp": timestamp.isoformat(),
        "snapshot_path": str(filepath),
        "filename": filename
    }
