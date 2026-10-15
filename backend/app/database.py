import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("backend.database")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "suspicious_activity")
FALLBACK_DIR = Path(__file__).resolve().parent.parent.parent / "evidence"
FALLBACK_FILE = FALLBACK_DIR / "events_fallback.json"

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.use_mongodb = False
        self._fallback_events: List[Dict[str, Any]] = []
        self._init_connection()

    def _init_connection(self):
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi

            logger.info(f"Attempting MongoDB connection to {MONGODB_URL}...")
            # Set a 2.5s server selection timeout to avoid freezing if Mongo is not running locally
            client = MongoClient(
                MONGODB_URL,
                server_api=ServerApi("1") if "mongodb.net" in MONGODB_URL else None,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500
            )
            client.admin.command("ping")
            self.client = client
            self.db = client[DATABASE_NAME]
            self.use_mongodb = True
            logger.info("Successfully connected to MongoDB!")
        except Exception as e:
            logger.warning(
                f"Could not connect to MongoDB ({e}). Falling back to local file storage for seamless local operation."
            )
            self.use_mongodb = False
            self._load_fallback()

    def _load_fallback(self):
        try:
            FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
            if FALLBACK_FILE.exists():
                with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                    self._fallback_events = json.load(f)
            else:
                self._fallback_events = []
        except Exception as err:
            logger.error(f"Error loading fallback store: {err}")
            self._fallback_events = []

    def _save_fallback(self):
        try:
            FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
            with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(self._fallback_events, f, indent=2, ensure_ascii=False)
        except Exception as err:
            logger.error(f"Error saving fallback store: {err}")

    def ping(self) -> dict:
        if self.use_mongodb and self.client:
            self.client.admin.command("ping")
            return {"status": "connected", "mode": "mongodb", "database": DATABASE_NAME}
        return {"status": "connected", "mode": "fallback_store", "count": len(self._fallback_events)}

    def insert_event(self, event_data: dict) -> str:
        import uuid
        event_id = str(uuid.uuid4())
        event_record = dict(event_data)
        event_record["id"] = event_record.get("id") or event_id

        if self.use_mongodb and self.db is not None:
            try:
                res = self.db.event_logs.insert_one(event_record)
                return str(res.inserted_id)
            except Exception as e:
                logger.error(f"MongoDB write failed, saving to fallback: {e}")

        # Fallback store
        self._fallback_events.insert(0, event_record)
        self._save_fallback()
        return event_record["id"]

    def get_events(self, limit: int = 50, camera_id: Optional[str] = None, activity_type: Optional[str] = None, status: Optional[str] = None) -> List[dict]:
        if self.use_mongodb and self.db is not None:
            try:
                query = {}
                if camera_id:
                    query["camera_id"] = camera_id
                if activity_type:
                    query["activity_type"] = activity_type
                if status:
                    query["status"] = status
                
                cursor = self.db.event_logs.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
                events = list(cursor)
                for ev in events:
                    if "id" not in ev and "_id" in ev:
                        ev["id"] = str(ev["_id"])
                return events
            except Exception as e:
                logger.error(f"MongoDB read failed, reading from fallback: {e}")

        # Fallback filter
        results = []
        for ev in self._fallback_events:
            if camera_id and ev.get("camera_id") != camera_id:
                continue
            if activity_type and ev.get("activity_type") != activity_type:
                continue
            if status and ev.get("status") != status:
                continue
            results.append(ev)
            if len(results) >= limit:
                break
        return results

    def update_event_status(self, event_id: str, new_status: str, notes: Optional[str] = None) -> bool:
        if self.use_mongodb and self.db is not None:
            try:
                from bson.objectid import ObjectId
                update_doc = {"status": new_status}
                if notes:
                    update_doc["review_notes"] = notes

                # Try matching by custom 'id' or Mongo '_id'
                res = self.db.event_logs.update_one({"id": event_id}, {"$set": update_doc})
                if res.matched_count == 0:
                    try:
                        res = self.db.event_logs.update_one({"_id": ObjectId(event_id)}, {"$set": update_doc})
                    except Exception:
                        pass
                if res.matched_count > 0:
                    return True
            except Exception as e:
                logger.error(f"MongoDB update failed: {e}")

        for ev in self._fallback_events:
            if str(ev.get("id")) == str(event_id):
                ev["status"] = new_status
                if notes:
                    ev["review_notes"] = notes
                self._save_fallback()
                return True
        return False

    def get_stats(self) -> dict:
        events = self.get_events(limit=1000)
        events_by_type = {}
        pending = 0
        verified = 0
        dismissed = 0

        for ev in events:
            atype = ev.get("activity_type", "unknown")
            events_by_type[atype] = events_by_type.get(atype, 0) + 1
            st = ev.get("status", "pending_review")
            if st == "pending_review":
                pending += 1
            elif st == "verified" or st == "resolved":
                verified += 1
            elif st == "dismissed":
                dismissed += 1

        return {
            "total_events": len(events),
            "pending_alerts": pending,
            "verified_events": verified,
            "dismissed_events": dismissed,
            "events_by_type": events_by_type,
            "active_cameras": 1
        }

db_manager = DatabaseManager()
