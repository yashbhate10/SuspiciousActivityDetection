# 🛡️ AI-Based Suspicious Activity Detection System
## Comprehensive Technical Project Report

**Author:** Yash Bhate  
**Repository:** [github.com/yashbhate10/SuspiciousActivityDetection](https://github.com/yashbhate10/SuspiciousActivityDetection.git)  
**Tech Stack:** Python 3.12, PyTorch (CUDA 12.1), YOLOv11, ByteTrack, OpenCV, FastAPI, Uvicorn, MongoDB, React.js (Vite), Tailwind-inspired CSS  

---

## 1. Executive Summary

Standard CCTV camera systems in public and private infrastructure (banks, stations, airports, server rooms) produce hundreds of hours of raw footage that require human security operators to monitor continuously. Human fatigue leads to missed incidents and delayed responses. Furthermore, basic computer vision solutions that alert simply because "a person is detected" result in overwhelming false alarms.

This project delivers an **Enterprise-Grade AI-Powered Surveillance and Behaviour Analysis Platform**. The system decouples mere object presence from suspicious activity:
- It tracks persons and objects across frames using **YOLOv11 and ByteTrack**.
- It analyzes dynamic physical actions through an advanced **Temporal Behaviour Analysis Engine** that observes what a person *does* over time (movement vector analysis, stationary dwell ratio, approach acceleration, luggage ownership detachment).
- It incorporates a **Crime and Violence Classifier** (Normal, Abuse, Arrest, Arson, Assault) equipped with temporal consistency filters.
- Alerts are streamed in real time to a **FastAPI backend** and displayed on a **React-based tactical surveillance dashboard** with live MJPEG feeds and instant toast notifications.

---

## 2. Core Problem Statement & False Alarm Elimination

### The Problem
Traditional automated surveillance suffers from two fatal flaws:
1. **Premature Alarms:** Alarms firing before any activity actually occurred, triggered by naive static polygon zones covering walking areas.
2. **False Positives from Classification Models:** Pretrained crime anomaly models trained solely on crime categories (e.g., UCF-Crime without a "Normal" class) mathematically *always* predicted a crime on every normal frame, resulting in 100% false alarm rates.

### How We Solved It
1. **Elimination of Arbitrary Static Zones:** Replaced artificial zone-triggering with a pure **Temporal Behaviour Engine** (`behaviour.py`). Alerts only fire if empirical physical thresholds are breached:
   - **Loitering:** Requires a track to remain within a tight stationary radius for $\ge 15\text{ seconds}$ with $\ge 75\%$ stationary ratio (walking through a corridor never triggers an alert).
   - **Erratic Movement:** Flags rapid, repeated directional reversals ($\ge 6$ reversals within 6 seconds), indicating panic or pacing.
   - **Rapid Approach:** Computes closing vector velocity toward other individuals ($\ge 120\text{ px/s}$), detecting sudden lunges or aggressive approaches.
   - **Abandoned Luggage:** Tracks luggage and owner centroids; alerts only when luggage remains stationary and owner distance exceeds 160 pixels for $\ge 10\text{ seconds}$.
   - **Crowd Surge:** Monitors instantaneous person density shifts ($\ge 3$ sudden arrivals in $<3$ seconds with $\ge 5$ total people).

2. **5-Class Crime Model with Normal Ground Truth:**
   Rebuilt the dataset pipeline (`prepare_dataset.py`) to incorporate a calibrated **Normal** class extracted across non-incident segments, combined with a **Normal Probability Margin Check** and temporal streak requirement (requiring 4+ consecutive high-confidence frames) to prevent single-frame anomalies from causing false alarms.

---

## 3. High-Level System Architecture

```
                       [ CCTV Camera / Webcam / Video File ]
                                        │
                                        ▼
                      [ AI Engine: detect.py (RTX 3050 Ti GPU) ]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          [ YOLOv11 + ByteTrack ]               [ Crime Classifier (Optional) ]
        (Person, Bag, Vehicle Tracks)          (5-Class: Normal, Assault, etc.)
                    │                                       │
                    ▼                                       ▼
     [ Temporal Behaviour Engine ]              [ Temporal Streak Gating ]
   (Loiter, Erratic, Approach, Bags)            (Suppresses 1-frame spikes)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                         [ Evidence & Dispatcher ]
                    (Watermarked JPEGs + JSON Alert)
                                        │
               HTTP POST /api/events    │    HTTP POST /api/stream/feed/{cam}
                                        ▼
                             [ FastAPI Backend ]
                           (Port 8000 / Uvicorn)
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       [ In-Memory Alert Store ]                     [ Database Storage ]
     (Live Poll: /api/alerts/latest)              (MongoDB / Local JSON Fallback)
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                        [ React Surveillance Dashboard ]
                               (Port 5173 / Vite)
         - Live MJPEG Surveillance Feed with Tactical HUD
         - Instant Top-Right Alarm Toast Notification (<3s latency)
         - Incident Evidence Review Archive & Status Verifier
```

---

## 4. Module Breakdown

### 4.1 AI Engine (`ai-engine/`)
- **`tracking.py`**: Wraps YOLOv11 object detection and ByteTrack multi-object tracking. Emits persistent tracking IDs, bounding boxes, and object classes. Accelerated via PyTorch CUDA on the NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- **`behaviour.py`**: Maintains `PersonTrack` objects storing temporal position histories, velocity vectors, and centroid trajectories. Zero zone dependencies.
- **`anomaly_classifier.py`**: Loads the trained PyTorch/YOLO classification weights. Performs inference at 224×224 resolution with class probability gating and streak counters.
- **`alert_service.py`**: Multi-threaded non-blocking queue. Decouples video inference frame rates from network I/O, ensuring video processing never drops below 25+ FPS while dispatching alerts to the backend.
- **`evidence.py`**: Annotates and crops forensic snapshot evidence with timestamps, track IDs, and alert categories, saving them to `evidence/`.

### 4.2 Backend (`backend/`)
- **FastAPI Core (`main.py`)**: Asynchronous REST API server configured with CORS middleware and static file serving for forensic evidence snapshots.
- **Streaming Pipeline (`routes/stream.py`)**: Thread-safe `StreamManager` providing multipart MJPEG HTTP streams (`/api/stream/{camera_id}`) directly consumable by web browser `<img>` tags.
- **Live Alerts (`routes/alerts.py`)**: Thread-safe circular buffer for high-priority alerts (`/api/alerts/latest`), enabling lightweight dashboard polling.
- **Incident Storage (`database.py`)**: Robust dual-mode persistence. Automatically connects to MongoDB if available; seamlessly falls back to local structured JSON storage (`evidence/events_fallback.json`) if MongoDB is offline.

### 4.3 Frontend Web Dashboard (`frontend/`)
- Built with **React 19 + Vite**.
- **Tactical Dark HUD**: Metrics grid displaying active cameras, total incidents, verified events, and breakdown by activity type.
- **Live Surveillance Tab**: Real-time MJPEG feed viewer with stream pause/resume and source switching (Webcam / Uploaded test videos).
- **Incident Evidence Archive**: Interactive incident log with thumbnail previews, full-resolution evidence inspector, and security review actions (Verify / Dismiss).
- **Real-Time Toast Alerts**: Floating notification banner that triggers instantly upon suspicious activity detection and auto-dismisses after 6 seconds.

---

## 5. Key Engineering Milestones & Fixes Completed

1. **Hardware Acceleration Fix:**
   - Identified that the virtual environment initially had a CPU-only PyTorch build (`2.14.1+cpu`).
   - Upgraded to `torch 2.5.1+cu121` with CUDA 12.1 runtime, unleashing the 4GB NVIDIA GeForce RTX 3050 Ti Laptop GPU and accelerating training from 6+ hours down to ~6 minutes.

2. **FastAPI Frame Streaming Protocol Fix:**
   - Fixed an endpoint bug where FastAPI was interpreting `frame: bytes` as a query parameter rather than raw body bytes, which caused incoming video frames to be rejected with HTTP 422.
   - Migrated to asynchronous `Request.body()` ingestion, enabling uninterrupted 25 FPS web streaming.

3. **Behavior Engine Zone-Free Refactor:**
   - Completely decoupled spatial behavior rules from predefined polygon zones.
   - Replaced fixed boundary collision checks with vector-based kinematic analysis to guarantee that walking or passing through an area never raises a false alarm.

---

## 6. Verification & How to Run

### Requirements
- Python 3.11 or 3.12 with CUDA 12.1 compatible GPU
- Node.js 18+ and npm

### Execution Steps (3 Terminals)

**Terminal 1 — Backend API Server:**
```powershell
cd D:\Projects\SuspiciousActivityDetection\backend
..\ai-engine\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend Surveillance Dashboard:**
```powershell
cd D:\Projects\SuspiciousActivityDetection\frontend
npm run dev
```

**Terminal 3 — AI Video Processing Engine:**
```powershell
cd D:\Projects\SuspiciousActivityDetection\ai-engine

# Webcam surveillance:
.\.venv\Scripts\python.exe detect.py --source 0

# Video file surveillance:
.\.venv\Scripts\python.exe detect.py --source ..\data\test_videos\Abuse001_x264.mp4

# Synthetic simulation mode:
.\.venv\Scripts\python.exe detect.py --source sim
```

Open your browser at `http://localhost:5173` to access the live dashboard.
