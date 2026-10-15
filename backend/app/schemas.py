from datetime import datetime
from typing import List, Optional, Tuple, Any, Dict
from pydantic import BaseModel, Field

class ZoneConfig(BaseModel):
    name: str = "Restricted Area"
    zone_type: str = Field(default="restricted", description="'restricted' or 'loitering'")
    points: List[Tuple[int, int]] = Field(
        default_factory=lambda: [(100, 100), (500, 100), (500, 400), (100, 400)]
    )
    loiter_threshold_seconds: Optional[int] = 30
    color_bgr: Optional[Tuple[int, int, int]] = (0, 0, 255)

class CameraConfig(BaseModel):
    camera_id: str = "camera_01"
    name: str = "Main Entrance / Corridor"
    source: str = "0"  # Can be 0 for webcam, or video filepath, or rtsp URL
    enabled: bool = True
    zones: List[ZoneConfig] = Field(default_factory=list)

class EventCreate(BaseModel):
    camera_id: str = Field(default="camera_01")
    activity_type: str = Field(..., description="loitering, restricted_zone_entry, abandoned_object, etc.")
    timestamp: str
    track_id: Optional[int] = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    snapshot_path: Optional[str] = None
    snapshot_url: Optional[str] = None
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    status: str = Field(default="pending_review", description="'pending_review', 'verified', 'dismissed', 'resolved'")

class EventUpdateStatus(BaseModel):
    status: str = Field(..., description="'pending_review', 'verified', 'dismissed', 'resolved'")
    notes: Optional[str] = None

class EventResponse(BaseModel):
    id: str
    camera_id: str
    activity_type: str
    timestamp: str
    track_id: Optional[int] = None
    confidence: float
    snapshot_path: Optional[str] = None
    snapshot_url: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    status: str
    received_at: Optional[str] = None

class SystemStats(BaseModel):
    total_events: int
    pending_alerts: int
    verified_events: int
    dismissed_events: int
    events_by_type: Dict[str, int]
    active_cameras: int
