# 🛡️ AI-Based Suspicious Activity Detection System using YOLOv11

An end-to-end intelligent CCTV surveillance and behaviour analysis system powered by **YOLOv11**, **ByteTrack**, **FastAPI**, **MongoDB**, and **React.js**.

> 📄 **Looking for the in-depth engineering documentation?** Check out [PROJECT_REPORT.md](PROJECT_REPORT.md) for full architecture details, false-alarm mitigation strategies, and mathematical thresholds.

The core architecture strictly separates **object detection/tracking** from **behaviour analysis**: detecting a person alone is never suspicious. The behaviour engine monitors movement vectors, dwell times, and luggage proximity across time to detect:
1. **Loitering Detection** ($\ge 15\text{s}$ stationary dwell within 60px perimeter with $\ge 75\%$ stationary ratio).
2. **Erratic Movement Detection** (erratic direction shifts $\ge 6$ reversals in 6s).
3. **Rapid Approach Precursor** (sudden velocity spike $>120\text{ px/s}$ towards another person).
4. **Abandoned Object Detection** (unattended luggage/backpacks remaining stationary while owners depart $>160\text{px}$).
5. **Crowd Surge** (instantaneous arrival of $\ge 3$ persons in $<3$ seconds with $\ge 5$ total people).
6. **Crime & Anomaly Classifier** (optional 5-class UCF-Crime + Normal neural network classifier).

---

## 🏗️ Architecture Overview

```
 [ CCTV / Webcam / Video File ]
               │
               ▼
   [ YOLOv11 + ByteTrack ] ── (PyTorch CUDA on NVIDIA RTX 3050 Ti)
     (Detects persons, luggage, vehicles; assigns persistent tracking IDs)
               │
               ▼
 [ Behaviour Analysis Engine ]
     (Velocity Vectors, Dwell Timers, Owner-Proximity Detachment, Crowd Kinetics)
               │
       ┌───────┴───────┐
       ▼               ▼
[ Evidence Service ] [ Non-Blocking Alert Dispatcher ]
(Timestamped Snapshots) │
                        ▼
               [ FastAPI Backend ]
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   [ MongoDB / Store ]     [ React Surveillance Dashboard ]
 (Incident Logs & Evidence)  (Real-Time Monitoring, Live MJPEG Stream, Toasts)
```

---

## 📁 Project Structure

```
SuspiciousActivityDetection/
├── ai-engine/
│   ├── config.py              # Camera IDs, dwell timers, speed thresholds
│   ├── detect.py              # Main real-time detection & HUD tracking runner
│   ├── tracking.py            # YOLO11 + ByteTrack tracker abstraction
│   ├── behaviour.py           # Enterprise pure temporal behaviour engine
│   ├── anomaly_classifier.py  # 5-class crime classifier with temporal streak gating
│   ├── prepare_dataset.py     # Balanced UCF-Crime + Normal dataset builder
│   ├── train_anomaly.py       # PyTorch/Ultralytics GPU training runner
│   ├── alert_service.py       # Asynchronous queue & HTTP alert dispatcher
│   ├── evidence.py            # Watermarked forensic snapshot capture
│   └── requirements.txt       # AI engine dependencies
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entry point, static evidence mount, CORS
│   │   ├── database.py        # MongoDB connection with zero-config local fallback
│   │   ├── schemas.py         # Pydantic data models for events, zones, stats
│   │   └── routes/
│   │       ├── events.py      # Event logging, stats, and forensic review endpoints
│   │       ├── alerts.py      # In-memory latest alert circular buffer for dashboard
│   │       ├── cameras.py     # Camera metadata endpoints
│   │       ├── videos.py      # Testcase video upload and AI streaming
│   │       └── stream.py      # Live MJPEG streaming buffer with Request.body()
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example           # Example server and database settings
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Dark-mode surveillance dashboard with live toast
│   │   ├── App.css            # Tactical control room styling
│   │   └── main.jsx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── data/
│   └── test_videos/           # Sample video test cases (Abuse, Assault, etc.)
├── models/                    # YOLO11 model weights
├── evidence/                  # High-resolution forensic snapshots
├── tests/
│   ├── test_pipeline.py       # Behaviour engine unit tests
│   ├── test_backend.py        # FastAPI endpoints integration test
│   └── test_tracker.py        # YOLO11 smoke test
├── docker-compose.yml         # Containerized full-stack deployment
├── PROJECT_REPORT.md          # Full technical report
└── README.md
```

---

## 🚀 Quick Start Guide (Windows)

### Prerequisites
- Python 3.11 or 3.12
- Node.js LTS (v18+)
- (Recommended) NVIDIA GPU with CUDA 12.1 for real-time inference acceleration

---

### Step 1: Start the FastAPI Backend
Open a terminal in `backend/`:

```powershell
cd backend
..\ai-engine\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
- API Documentation (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

*Note: The backend connects to MongoDB if available, and automatically falls back to local JSON storage if MongoDB is offline.*

---

### Step 2: Start the React Surveillance Dashboard
Open a new terminal in `frontend/`:

```powershell
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

**Dashboard Features:**
- **Overview & Analytics**: High-priority incident feed, metrics cards, drill simulations.
- **Live Surveillance Feed**: Live MJPEG video stream with real-time HUD and active track counters.
- **Incident Evidence Archive**: Forensic inspector modal with high-res snapshots and verification controls (*Verify* / *Dismiss*).
- **Real-Time Alert Toast**: Instant floating banner appearing upon any detected incident.

---

### Step 3: Run the Real-Time AI Detection Engine
Open a third terminal:

```powershell
# Run with webcam (source 0):
cd ai-engine
.\.venv\Scripts\python.exe detect.py --source 0

# Run with a recorded test video:
.\.venv\Scripts\python.exe detect.py --source ..\data\test_videos\Abuse001_x264.mp4

# Run with crime classifier enabled:
.\.venv\Scripts\python.exe detect.py --source ..\data\test_videos\Abuse001_x264.mp4 --enable-crime-clf

# Run with synthetic simulation mode (if no camera is attached):
.\.venv\Scripts\python.exe detect.py --source sim
```

---

## 🛡️ License & Acknowledgements
- **YOLOv11**: Ultralytics
- **ByteTrack**: ByteDance Multi-Object Tracking
- **UCF-Crime**: University of Central Florida Center for Research in Computer Vision
