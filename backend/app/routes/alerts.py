"""
Real-time alert endpoint.

Stores the most-recent alert in memory so the React dashboard can poll
GET /api/alerts/latest without waiting for a full event-list refresh.
"""

import threading
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/alerts", tags=["Real-Time Alerts"])

# ── In-memory alert store (thread-safe) ──────────────────────────────
_lock = threading.Lock()
_latest_alert: Optional[dict] = None
_alert_history: List[dict] = []          # newest-first, capped at 50
_MAX_HISTORY = 50


def set_latest_alert(alert: dict) -> None:
    """Called from the events route whenever a new event is recorded."""
    global _latest_alert
    enriched = {
        **alert,
        "alerted_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _latest_alert = enriched
        _alert_history.insert(0, enriched)
        # Keep history bounded
        if len(_alert_history) > _MAX_HISTORY:
            del _alert_history[_MAX_HISTORY:]


# ── HTTP endpoints ───────────────────────────────────────────────────
@router.get("/latest")
def get_latest_alert():
    """Return the single most recent alert (or 204 No Content)."""
    with _lock:
        if _latest_alert is None:
            raise HTTPException(status_code=204, detail="No alerts yet")
        return _latest_alert


@router.get("/history")
def get_alert_history(limit: int = 20):
    """Return the last N in-memory alerts (max 50)."""
    with _lock:
        return {"alerts": _alert_history[:min(limit, _MAX_HISTORY)]}
