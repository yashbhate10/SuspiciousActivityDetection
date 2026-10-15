from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from ..schemas import CameraConfig, ZoneConfig

router = APIRouter(prefix="/api/cameras", tags=["Cameras & Zones"])

# Default in-memory / configured cameras
DEFAULT_CAMERAS: Dict[str, CameraConfig] = {
    "camera_01": CameraConfig(
        camera_id="camera_01",
        name="Zone A - Main Hallway & Restricted Entrance",
        source="0",
        enabled=True,
        zones=[
            ZoneConfig(
                name="Restricted Vault / Server Room",
                zone_type="restricted",
                points=[(120, 80), (520, 80), (520, 380), (120, 380)],
                loiter_threshold_seconds=0,
                color_bgr=(0, 0, 255)
            ),
            ZoneConfig(
                name="Corridor Loitering Check Zone",
                zone_type="loitering",
                points=[(50, 50), (600, 50), (600, 450), (50, 450)],
                loiter_threshold_seconds=20,
                color_bgr=(0, 255, 255)
            )
        ]
    )
}

@router.get("", response_model=List[CameraConfig])
def get_cameras():
    return list(DEFAULT_CAMERAS.values())

@router.get("/{camera_id}", response_model=CameraConfig)
def get_camera_by_id(camera_id: str):
    if camera_id not in DEFAULT_CAMERAS:
        raise HTTPException(status_code=404, detail="Camera not found")
    return DEFAULT_CAMERAS[camera_id]

@router.post("", response_model=CameraConfig)
def register_or_update_camera(config: CameraConfig):
    DEFAULT_CAMERAS[config.camera_id] = config
    return config

@router.put("/{camera_id}/zones", response_model=CameraConfig)
def update_zones(camera_id: str, zones: List[ZoneConfig]):
    if camera_id not in DEFAULT_CAMERAS:
        raise HTTPException(status_code=404, detail="Camera not found")
    DEFAULT_CAMERAS[camera_id].zones = zones
    return DEFAULT_CAMERAS[camera_id]
