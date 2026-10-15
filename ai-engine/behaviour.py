"""
Enterprise Behaviour Analysis Engine.

Detects suspicious activity purely from OBSERVED BEHAVIOUR over time.
No zone polygons required — alerts fire based on what a person DOES,
not where they stand.

Behaviours detected:
  1. Loitering          — person stays mostly stationary for extended time
  2. Erratic Movement   — person changes direction repeatedly / moves unpredictably
  3. Rapid Approach     — person moves fast toward another person (assault precursor)
  4. Abandoned Object   — bag/suitcase left without owner
  5. Crowd Surge        — sudden large increase in person count in a short time
"""

import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from config import (
    LOITER_THRESHOLD_SECONDS,
    LOITER_ALERT_COOLDOWN_SECONDS,
    ABANDONED_OBJECT_MIN_SECONDS,
    ABANDONED_OBJECT_MAX_OWNER_DIST,
    CLASS_PERSON,
    CLASSES_BAGS
)

# ── Additional enterprise thresholds ─────────────────────────────────
ERRATIC_DIRECTION_CHANGES   = 6      # min direction reversals in window to flag
ERRATIC_WINDOW_SECONDS      = 6.0   # time window for direction-change counting
RAPID_APPROACH_SPEED_PX_S   = 120   # pixels/sec toward another person
RAPID_APPROACH_COOLDOWN_S   = 20.0
CROWD_SURGE_MIN_PERSONS      = 5     # minimum persons to consider a crowd
CROWD_SURGE_DELTA            = 3     # persons appearing within surge window
CROWD_SURGE_WINDOW_SECONDS   = 3.0
SPEED_HISTORY_MAXLEN         = 30    # frames to keep for speed computation


class PersonTrack:
    """Holds temporal state for a single tracked person."""

    def __init__(self, trk_id: int, center: Tuple[int, int], now: float):
        self.trk_id = trk_id
        self.first_seen = now
        self.last_seen  = now

        # Position history: deque of (cx, cy, timestamp)
        self.positions: deque = deque(maxlen=200)
        self.positions.append((center[0], center[1], now))

        # Direction history for erratic detection
        self.directions: deque = deque(maxlen=50)   # unit vectors
        self.direction_changes = 0
        self.direction_change_window_start = now

        # Alert cooldowns
        self.last_loiter_alert   = 0.0
        self.last_erratic_alert  = 0.0
        self.last_approach_alert = 0.0

    def update(self, center: Tuple[int, int], now: float):
        """Update track with new position and compute derived features."""
        prev_cx, prev_cy, _ = self.positions[-1]
        self.positions.append((center[0], center[1], now))
        self.last_seen = now

        # Compute movement direction
        dx, dy = center[0] - prev_cx, center[1] - prev_cy
        dist = np.hypot(dx, dy)
        if dist > 2.0:  # ignore micro-jitter
            unit = (dx / dist, dy / dist)
            if self.directions:
                prev_unit = self.directions[-1]
                # Direction reversal = dot product strongly negative
                dot = unit[0] * prev_unit[0] + unit[1] * prev_unit[1]
                if dot < -0.5:
                    self.direction_changes += 1
            self.directions.append(unit)

        # Reset direction-change counter every window
        if now - self.direction_change_window_start > ERRATIC_WINDOW_SECONDS:
            self.direction_changes = 0
            self.direction_change_window_start = now

    def get_dwell_seconds(self, now: float) -> float:
        return now - self.first_seen

    def get_stationary_ratio(self, window_seconds: float = 30.0) -> float:
        """Fraction of recent samples within 60px of the average position."""
        recent = [(x, y) for x, y, t in self.positions
                  if (self.last_seen - t) <= window_seconds]
        if len(recent) < 5:
            return 0.0
        avg_x = sum(x for x, y in recent) / len(recent)
        avg_y = sum(y for x, y in recent) / len(recent)
        near  = sum(1 for x, y in recent if np.hypot(x - avg_x, y - avg_y) <= 60.0)
        return near / len(recent)

    def get_speed_px_per_sec(self, window_seconds: float = 1.0) -> float:
        """Average speed (px/s) over the last `window_seconds`."""
        cutoff = self.last_seen - window_seconds
        recent = [(x, y, t) for x, y, t in self.positions if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        total_dist = sum(
            np.hypot(recent[i][0] - recent[i-1][0], recent[i][1] - recent[i-1][1])
            for i in range(1, len(recent))
        )
        time_span = recent[-1][2] - recent[0][2]
        return total_dist / max(time_span, 0.01)

    def get_approach_speed_toward(
        self, target_center: Tuple[int, int], window_seconds: float = 1.0
    ) -> float:
        """Speed (px/s) of movement toward a specific target point (positive = approaching)."""
        cutoff = self.last_seen - window_seconds
        recent = [(x, y, t) for x, y, t in self.positions if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        # Distance to target at start and end of window
        dist_start = np.hypot(recent[0][0] - target_center[0], recent[0][1] - target_center[1])
        dist_end   = np.hypot(recent[-1][0] - target_center[0], recent[-1][1] - target_center[1])
        time_span  = recent[-1][2] - recent[0][2]
        # Positive = getting closer
        return (dist_start - dist_end) / max(time_span, 0.01)


class BehaviourEngine:
    """
    Enterprise behaviour analysis engine.
    Alerts fire based on what a person DOES, not where they stand.
    """

    def __init__(self, *args, **kwargs):
        self.tracks: Dict[int, PersonTrack] = {}
        self.bag_history: Dict[int, Dict[str, Any]] = {}
        self.last_bag_alert_at: Dict[int, float] = {}
        self.last_rapid_approach_alert: Dict[Tuple[int, int], float] = {}

        # Crowd surge tracking
        self._person_count_history: deque = deque(maxlen=100)  # (count, timestamp)
        self._last_crowd_alert = 0.0

    def _get_or_create_track(
        self, trk_id: int, center: Tuple[int, int], now: float
    ) -> PersonTrack:
        if trk_id not in self.tracks:
            self.tracks[trk_id] = PersonTrack(trk_id, center, now)
        return self.tracks[trk_id]

    def _prune_tracks(self, now: float, timeout: float = 10.0):
        expired = [tid for tid, t in self.tracks.items()
                   if (now - t.last_seen) > timeout]
        for tid in expired:
            del self.tracks[tid]
            self.bag_history.pop(tid, None)

    def evaluate_frame(
        self,
        tracked_objects: List[Dict[str, Any]],
        now: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all tracked objects and returns a list of alert dicts.
        Each alert has: activity_type, track_id, confidence, bbox, details, severity
        """
        if now is None:
            now = time.monotonic()

        alerts = []
        persons, bags = [], []
        for obj in tracked_objects:
            if obj["class_id"] == CLASS_PERSON:
                persons.append(obj)
            elif obj["class_id"] in CLASSES_BAGS:
                bags.append(obj)

        person_centers = [
            (int((o["bbox"][0] + o["bbox"][2]) / 2),
             int((o["bbox"][1] + o["bbox"][3]) / 2))
            for o in persons
        ]

        # Update all person tracks
        for obj in persons:
            trk_id = obj["track_id"]
            cx = int((obj["bbox"][0] + obj["bbox"][2]) / 2)
            cy = int((obj["bbox"][1] + obj["bbox"][3]) / 2)
            trk = self._get_or_create_track(trk_id, (cx, cy), now)
            trk.update((cx, cy), now)

        # ── 1. LOITERING ─────────────────────────────────────────────
        for obj in persons:
            trk_id = obj["track_id"]
            trk = self.tracks[trk_id]
            dwell = trk.get_dwell_seconds(now)

            if dwell >= LOITER_THRESHOLD_SECONDS:
                ratio = trk.get_stationary_ratio(window_seconds=dwell)
                if ratio >= 0.75:
                    if (now - trk.last_loiter_alert) >= LOITER_ALERT_COOLDOWN_SECONDS:
                        trk.last_loiter_alert = now
                        alerts.append({
                            "activity_type": "loitering",
                            "track_id": trk_id,
                            "confidence": min(0.98, 0.70 + ratio * 0.28),
                            "bbox": obj["bbox"],
                            "severity": "medium",
                            "details": {
                                "dwell_seconds": round(dwell, 1),
                                "stationary_ratio": round(ratio, 2),
                                "description": f"Person #{trk_id} has been stationary for {dwell:.0f}s"
                            }
                        })

        # ── 2. ERRATIC MOVEMENT ───────────────────────────────────────
        for obj in persons:
            trk_id = obj["track_id"]
            trk = self.tracks[trk_id]

            if (trk.direction_changes >= ERRATIC_DIRECTION_CHANGES and
                    trk.get_dwell_seconds(now) >= 3.0):
                if (now - trk.last_erratic_alert) >= 30.0:
                    trk.last_erratic_alert = now
                    alerts.append({
                        "activity_type": "erratic_movement",
                        "track_id": trk_id,
                        "confidence": min(0.95, 0.60 + trk.direction_changes * 0.04),
                        "bbox": obj["bbox"],
                        "severity": "medium",
                        "details": {
                            "direction_changes": trk.direction_changes,
                            "window_seconds": ERRATIC_WINDOW_SECONDS,
                            "description": f"Person #{trk_id} is moving erratically ({trk.direction_changes} direction reversals)"
                        }
                    })

        # ── 3. RAPID APPROACH (assault precursor) ────────────────────
        for i, obj_a in enumerate(persons):
            trk_a = self.tracks[obj_a["track_id"]]
            for j, obj_b in enumerate(persons):
                if i == j:
                    continue
                cx_b = int((obj_b["bbox"][0] + obj_b["bbox"][2]) / 2)
                cy_b = int((obj_b["bbox"][1] + obj_b["bbox"][3]) / 2)

                approach_speed = trk_a.get_approach_speed_toward((cx_b, cy_b))
                if approach_speed >= RAPID_APPROACH_SPEED_PX_S:
                    key = (obj_a["track_id"], obj_b["track_id"])
                    last = self.last_rapid_approach_alert.get(key, 0)
                    if (now - last) >= RAPID_APPROACH_COOLDOWN_S:
                        self.last_rapid_approach_alert[key] = now
                        alerts.append({
                            "activity_type": "rapid_approach",
                            "track_id": obj_a["track_id"],
                            "confidence": min(0.97, 0.65 + approach_speed / 400),
                            "bbox": obj_a["bbox"],
                            "severity": "high",
                            "details": {
                                "approach_speed_px_s": round(approach_speed, 1),
                                "target_track_id": obj_b["track_id"],
                                "description": f"Person #{obj_a['track_id']} rapidly approaching #{obj_b['track_id']} at {approach_speed:.0f}px/s"
                            }
                        })

        # ── 4. ABANDONED OBJECT ───────────────────────────────────────
        for bag in bags:
            trk_id = bag["track_id"]
            cx = int((bag["bbox"][0] + bag["bbox"][2]) / 2)
            cy = int((bag["bbox"][1] + bag["bbox"][3]) / 2)
            center = (cx, cy)

            if trk_id not in self.bag_history:
                self.bag_history[trk_id] = {
                    "last_pos": center,
                    "stationary_since": now,
                    "alerted": False
                }
            else:
                rec = self.bag_history[trk_id]
                displacement = np.hypot(cx - rec["last_pos"][0], cy - rec["last_pos"][1])

                if displacement > 25:
                    rec["last_pos"] = center
                    rec["stationary_since"] = now
                    rec["alerted"] = False
                else:
                    stationary_dur = now - rec["stationary_since"]
                    nearest = min(
                        (np.hypot(cx - px, cy - py) for px, py in person_centers),
                        default=float("inf")
                    )
                    owner_away = (not person_centers or nearest > ABANDONED_OBJECT_MAX_OWNER_DIST)

                    if (stationary_dur >= ABANDONED_OBJECT_MIN_SECONDS
                            and owner_away and not rec["alerted"]):
                        rec["alerted"] = True
                        alerts.append({
                            "activity_type": "abandoned_object",
                            "track_id": trk_id,
                            "confidence": min(0.97, 0.70 + stationary_dur / 60),
                            "bbox": bag["bbox"],
                            "severity": "high",
                            "details": {
                                "stationary_duration": round(stationary_dur, 1),
                                "nearest_person_dist": round(nearest, 1) if nearest != float("inf") else -1,
                                "object_class": bag["class_name"],
                                "description": f"Unattended {bag['class_name']} for {stationary_dur:.0f}s"
                            }
                        })

        # ── 5. CROWD SURGE ────────────────────────────────────────────
        self._person_count_history.append((len(persons), now))
        if len(persons) >= CROWD_SURGE_MIN_PERSONS:
            # Count how many persons entered in the last surge window
            old_counts = [c for c, t in self._person_count_history
                          if (now - t) <= CROWD_SURGE_WINDOW_SECONDS]
            if old_counts:
                surge = len(persons) - min(old_counts)
                if (surge >= CROWD_SURGE_DELTA
                        and (now - self._last_crowd_alert) >= 30.0):
                    self._last_crowd_alert = now
                    alerts.append({
                        "activity_type": "crowd_surge",
                        "track_id": -1,
                        "confidence": min(0.95, 0.65 + surge * 0.05),
                        "bbox": None,
                        "severity": "medium",
                        "details": {
                            "current_count": len(persons),
                            "surge_delta": surge,
                            "window_seconds": CROWD_SURGE_WINDOW_SECONDS,
                            "description": f"Crowd surge detected: {surge} persons entered in {CROWD_SURGE_WINDOW_SECONDS:.0f}s"
                        }
                    })

        # Prune stale tracks
        self._prune_tracks(now)
        return alerts
