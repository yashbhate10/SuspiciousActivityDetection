import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import db_manager
from .routes.events import router as events_router
from .routes.cameras import router as cameras_router
from .routes.stream import router as stream_router
from .routes.videos import router as videos_router
from .routes.alerts import router as alerts_router

load_dotenv()

app = FastAPI(
    title="AI-Based Suspicious Activity Detection API",
    description="Backend service for CCTV surveillance, behaviour rules, and alert management using YOLOv11 and ByteTrack.",
    version="1.0.0"
)

# CORS setup
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure evidence directory exists and mount it for static viewing
EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

# Include modular API routers
app.include_router(events_router)
app.include_router(cameras_router)
app.include_router(stream_router)
app.include_router(videos_router)
app.include_router(alerts_router)

@app.get("/health", tags=["Health"])
def health_check():
    db_status = db_manager.ping()
    return {
        "status": "healthy",
        "service": "suspicious-activity-api",
        "database": db_status
    }

@app.get("/", tags=["Root"])
def root_info():
    return {
        "name": "AI-Based Suspicious Activity Detection API",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/api/events",
            "/api/events/stats/summary",
            "/api/alerts/latest",
            "/api/alerts/history",
            "/api/cameras",
            "/api/stream/camera_01",
            "/evidence/{filename}"
        ]
    }
