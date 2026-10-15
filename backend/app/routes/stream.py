import time
import asyncio
import cv2
import numpy as np
from threading import Lock
from typing import Optional
from fastapi import APIRouter, Response, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/stream", tags=["Live Surveillance Stream"])

# Global thread-safe frame buffer for live streams
class StreamManager:
    def __init__(self):
        self._lock = Lock()
        self._frames = {}

    def update_frame(self, camera_id: str, frame_bytes: bytes):
        with self._lock:
            self._frames[camera_id] = frame_bytes

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(camera_id)

stream_manager = StreamManager()

def generate_placeholder_frame(text: str = "Awaiting Live Feed...") -> bytes:
    """Generate a clean dark placeholder frame with timestamp."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (24, 24, 27)  # Dark slate background
    # Grid lines
    for y in range(0, 480, 40):
        cv2.line(img, (0, y), (640, y), (35, 35, 40), 1)
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 480), (35, 35, 40), 1)

    cv2.putText(
        img,
        "AI SURVEILLANCE FEED",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        (40, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 255, 100),
        1,
        cv2.LINE_AA
    )
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes()

async def mjpeg_generator(camera_id: str):
    last_frame = None
    while True:
        frame_bytes = stream_manager.get_frame(camera_id)
        if frame_bytes is None:
            placeholder = generate_placeholder_frame("Live Engine Standby / Starting...")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            )
            await asyncio.sleep(0.4)
            continue

        if frame_bytes is not last_frame:
            last_frame = frame_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(0.015)
        else:
            await asyncio.sleep(0.01)

@router.get("/{camera_id}")
async def live_mjpeg_feed(camera_id: str):
    """MJPEG streaming endpoint for direct browser embedding."""
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/feed/{camera_id}")
async def receive_feed_frame(camera_id: str, request: Request):
    """Endpoint for AI engine to push latest annotated frame for web streaming."""
    frame_bytes = await request.body()
    if frame_bytes:
        stream_manager.update_frame(camera_id, frame_bytes)
    return {"status": "received"}
