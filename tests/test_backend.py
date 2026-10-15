import sys
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

def test_api_endpoints():
    client = TestClient(app)

    # 1. Health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"
    print("[PASS] GET /health returned healthy status")

    # 2. Record an event
    sample_payload = {
        "camera_id": "camera_01",
        "activity_type": "restricted_zone_entry",
        "timestamp": "2026-10-14T18:00:00Z",
        "track_id": 42,
        "confidence": 0.95,
        "snapshot_path": "evidence/test_sample.jpg",
        "details": {"test": True},
        "status": "pending_review"
    }
    post_resp = client.post("/api/events", json=sample_payload)
    assert post_resp.status_code == 201
    event_id = post_resp.json()["id"]
    print(f"[PASS] POST /api/events created event: {event_id}")

    # 3. Retrieve events list
    list_resp = client.get("/api/events?limit=10")
    assert list_resp.status_code == 200
    events = list_resp.json()["events"]
    assert len(events) >= 1
    print(f"[PASS] GET /api/events retrieved {len(events)} events")

    # 4. Update status
    patch_resp = client.patch(f"/api/events/{event_id}", json={"status": "verified", "notes": "Confirmed drill test"})
    assert patch_resp.status_code == 200
    print("[PASS] PATCH /api/events/{id} updated status to verified")

    # 5. Check stats summary
    stats_resp = client.get("/api/events/stats/summary")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_events"] >= 1
    print(f"[PASS] GET /api/events/stats/summary: {stats}")

    # 6. Check camera endpoint
    cam_resp = client.get("/api/cameras")
    assert cam_resp.status_code == 200
    print(f"[PASS] GET /api/cameras returned {len(cam_resp.json())} configured camera(s)")

    # 7. Check video testcases endpoint
    vid_resp = client.get("/api/videos")
    assert vid_resp.status_code == 200
    videos = vid_resp.json()
    assert len(videos) >= 1
    print(f"[PASS] GET /api/videos listed {len(videos)} testcase video(s): {[v['filename'] for v in videos]}")

if __name__ == "__main__":
    print("\n--- Running FastAPI Backend Tests ---")
    test_api_endpoints()
    print("\n[SUCCESS] ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY!\n")
