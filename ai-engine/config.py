import os
from pathlib import Path
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
MODELS_DIR = PROJECT_ROOT / "models"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO Model Configuration
if (MODELS_DIR / "best.pt").exists():
    MODEL_PATH = str(MODELS_DIR / "best.pt")
elif (MODELS_DIR / "yolo11n.pt").exists():
    MODEL_PATH = str(MODELS_DIR / "yolo11n.pt")
else:
    MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.35
TRACKER_CONFIG = "bytetrack.yaml"
DEVICE = "0"  # Set to "0" for NVIDIA GPU, or "cpu" if CUDA unavailable

# COCO Class IDs
CLASS_PERSON = 0
CLASSES_VEHICLES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASSES_BAGS = [24, 26, 28]      # backpack, handbag, suitcase

# Behaviour Analysis Parameters
RESTRICTED_ZONE_CONSECUTIVE_FRAMES = 25  # ~1 second at 25fps; person must stay inside zone, not just clip the edge
LOITER_THRESHOLD_SECONDS = 15           # Alert if dwell time exceeds this
LOITER_ALERT_COOLDOWN_SECONDS = 45      # Min time between repeat alerts for same track
RESTRICTED_ALERT_COOLDOWN_SECONDS = 30  # Min time between repeat trespassing alerts
ABANDONED_OBJECT_MIN_SECONDS = 10       # Time luggage must remain stationary without owner
ABANDONED_OBJECT_MAX_OWNER_DIST = 160   # In pixels: owner farther than this considered detached

# Configured Surveillance Zones (Coordinates in 640x480 normalized/default space)
# Zone 1: High Security Restricted Area (Red) — smaller area, left side
RESTRICTED_ZONE_POLYGON = np.array(
    [(50, 150), (250, 150), (250, 380), (50, 380)],
    dtype=np.int32
)

# Zone 2: Monitored Corridor / Loitering Watch Zone (Yellow) — medium area, center-right
LOITERING_ZONE_POLYGON = np.array(
    [(300, 120), (580, 120), (580, 400), (300, 400)],
    dtype=np.int32
)

# Backend Integration
BACKEND_EVENT_URL = "http://localhost:8000/api/events"
BACKEND_STREAM_URL = "http://localhost:8000/api/stream/feed/camera_01"
CAMERA_ID = "camera_01"
