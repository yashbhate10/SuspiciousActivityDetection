import cv2
import numpy as np
from pathlib import Path

EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Generate a sample demo evidence snapshot
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:] = (30, 32, 38)

# Security grid pattern
for y in range(0, 480, 40):
    cv2.line(img, (0, y), (640, y), (42, 45, 52), 1)
for x in range(0, 640, 40):
    cv2.line(img, (x, 0), (x, 480), (42, 45, 52), 1)

# Restricted Zone overlay (Red)
pts = np.array([(120, 100), (520, 100), (520, 400), (120, 400)], dtype=np.int32)
overlay = img.copy()
cv2.fillPoly(overlay, [pts], (0, 0, 180))
cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
cv2.polylines(img, [pts], True, (0, 0, 255), 2)
cv2.putText(img, "RESTRICTED PERIMETER (ZONE A)", (130, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# Suspect outline
cv2.rectangle(img, (270, 180), (370, 380), (0, 0, 255), 3)
cv2.putText(img, "SUSPECT [TRESPASSING] 94%", (270, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
cv2.circle(img, (320, 380), 6, (0, 255, 255), -1)

# Header banner
cv2.rectangle(img, (0, 0), (640, 50), (18, 18, 28), -1)
cv2.line(img, (0, 50), (640, 50), (0, 0, 255), 2)
cv2.putText(img, "SECURITY ALERT: RESTRICTED ZONE INTRUSION", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
cv2.putText(img, "CAM-01 (Corridor) | Track #14 | Threshold: 5 frames debounced", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

output_path = EVIDENCE_DIR / "sample_demo.jpg"
cv2.imwrite(str(output_path), img)
print(f"[PASS] Sample evidence created at {output_path}")
