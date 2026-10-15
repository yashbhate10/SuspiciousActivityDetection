import sys
import time
import queue
import logging
import threading
from typing import Dict, Any, Optional
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import BACKEND_EVENT_URL, BACKEND_STREAM_URL, CAMERA_ID

logger = logging.getLogger("ai_engine.alerts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AlertDispatcher:
    def __init__(self, backend_url: str = BACKEND_EVENT_URL):
        self.backend_url = backend_url
        self.queue = queue.Queue(maxsize=200)
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def dispatch_alert(
        self,
        activity_type: str,
        timestamp: str,
        snapshot_path: str,
        track_id: Optional[int] = None,
        confidence: float = 0.88,
        details: Optional[Dict[str, Any]] = None,
        camera_id: str = CAMERA_ID
    ):
        """Enqueue alert payload for background delivery without blocking video inference."""
        payload = {
            "camera_id": camera_id,
            "activity_type": activity_type,
            "timestamp": timestamp,
            "track_id": track_id,
            "confidence": round(float(confidence), 2),
            "snapshot_path": snapshot_path,
            "details": details or {},
            "status": "pending_review"
        }
        try:
            self.queue.put_nowait(payload)
            logger.info(f"🚨 Enqueued alert: {activity_type.upper()} for track {track_id}")
        except queue.Full:
            logger.warning("Alert queue is full! Dropping alert to maintain real-time performance.")

    def _process_queue(self):
        session = requests.Session()
        while self.running:
            try:
                payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Attempt transmission to backend with retry
            delivered = False
            for attempt in range(3):
                try:
                    resp = session.post(self.backend_url, json=payload, timeout=3.0)
                    if resp.status_code in [200, 201]:
                        delivered = True
                        break
                    else:
                        logger.warning(f"Backend responded with {resp.status_code}: {resp.text}")
                except Exception as err:
                    time.sleep(0.5 * (attempt + 1))

            if not delivered:
                logger.error(f"Failed to deliver alert to {self.backend_url} after 3 attempts.")
            self.queue.task_done()

    def push_frame_to_stream(self, frame_bytes: bytes, url: str = BACKEND_STREAM_URL):
        """Non-blocking frame push to FastAPI live stream buffer."""
        def _send():
            try:
                requests.post(url, data=frame_bytes, headers={"Content-Type": "image/jpeg"}, timeout=0.1)
            except Exception:
                pass  # Ignore network drops during streaming
        threading.Thread(target=_send, daemon=True).start()

    def shutdown(self):
        self.running = False

alert_dispatcher = AlertDispatcher()
