from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, status
from ..schemas import EventCreate, EventResponse, EventUpdateStatus, SystemStats
from ..database import db_manager
from .alerts import set_latest_alert

router = APIRouter(prefix="/api/events", tags=["Events & Alerts"])

@router.get("", response_model=dict)
def list_events(
    limit: int = Query(default=50, ge=1, le=200),
    camera_id: Optional[str] = Query(default=None),
    activity_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    events = db_manager.get_events(
        limit=limit,
        camera_id=camera_id,
        activity_type=activity_type,
        status=status,
    )
    return {"events": events, "count": len(events)}

@router.post("", status_code=status.HTTP_201_CREATED)
def record_event(payload: EventCreate):
    event_dict = payload.model_dump()
    event_dict["received_at"] = datetime.now(timezone.utc).isoformat()
    
    # If snapshot_path exists and starts with evidence, generate web url
    if event_dict.get("snapshot_path"):
        filename = event_dict["snapshot_path"].replace("\\", "/").split("/")[-1]
        event_dict["snapshot_url"] = f"/evidence/{filename}"

    inserted_id = db_manager.insert_event(event_dict)

    # Push to the in-memory latest alert store for real-time dashboard polling
    set_latest_alert(event_dict)

    return {
        "id": inserted_id,
        "status": "stored",
        "message": f"Suspicious activity event '{payload.activity_type}' recorded successfully."
    }

@router.patch("/{event_id}")
def update_event_review(event_id: str, payload: EventUpdateStatus):
    success = db_manager.update_event_status(
        event_id=event_id,
        new_status=payload.status,
        notes=payload.notes,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Event with id '{event_id}' not found.")
    return {"status": "updated", "event_id": event_id, "new_status": payload.status}

@router.get("/stats/summary", response_model=SystemStats)
def get_stats():
    stats = db_manager.get_stats()
    return stats
