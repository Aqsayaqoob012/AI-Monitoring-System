# 🎓 Cheating Guard AI — Online Exam Proctoring System

An AI-powered online proctoring system that monitors students in real-time using computer vision, object detection, and audio analysis to detect suspicious behavior during online exams.

---

## 👥 Team Members & Work Division

| Member  | Modules |
|-------- |---------|
| *Aqsa*  | YOLO Object Detection (Phone + Person), Speech Recognition, Flask Web Server, UI Design |
| *Ayesha*| Head Pose Detection, Suspicious Activity Score Engine, Evidence Capture |

---

## 📁 Project Structure

```
AI-MONITORING-SYSTEM/
│
├── app.py                  # Main Flask application
├── best.pt                 # YOLO trained model
├── main.py                 # Standalone detection script
│
├── ayesha/                 # Ayesha's modules
│   ├── head_pose.py        # Head pose detection (MediaPipe)
│   ├── score_engine.py     # Risk scoring system
│   └── evidence.py         # Screenshot & video clip capture
│
├── templates/
│   ├── home.html           # Landing page
│   └── index.html          # Live dashboard
│
├── screenshots/            # Partner's detection screenshots
├── evidence/               # Ayesha's evidence captures
└── requirements.txt        # All dependencies
```

---

## ⚙️ Features

### Aqsa's Work
- **Person Detection** — Detects if student leaves the frame
- **Phone Detection** — YOLO model detects mobile phones
- **Speech Recognition** — Monitors audio, converts Urdu to Roman script
- **Timeline Logging** — Records all events with timestamps

### Ayesha's Work
- **Head Pose Detection** — Detects if student looks LEFT / RIGHT / UP / DOWN
- **Score Engine** — Weighted risk scoring system (LOW / MEDIUM / HIGH)
- **Evidence Capture** — Auto-saves screenshots and 5-second video clips on suspicious events

---

## 📊 Risk Scoring System

| Event | Score |
|-------|-------|
| Looking Away (5+ sec) | +5 |
| Looking Left | +3 |
| Looking Right | +3 |
| Looking Up | +2 |
| Looking Down | +4 |
| Multiple Persons | +15 |
| No Face Detected | +10 |
| Phone Detected | +10 |
| Voice Detected | +5 |

| Score Range | Risk Level |
|-------------|------------|
| 0 – 20 | 🟢 LOW |
| 21 – 50 | 🟡 MEDIUM |
| 51+ | 🔴 HIGH |

---

## 🖥️ Dashboard Features

- **3 Stat Boxes** — Total Frames, Session Duration, Total Alerts
- **Head Pose Strip** — Real-time LEFT / RIGHT / UP / DOWN / FORWARD indicators
- **Evidence Grid** — Screenshots with event name, time, direction (click to fullscreen)
- **Student Analysis Table** — Avg Score, Max Score, Severity, Event Count
- **Score Events** — Individual event pills with timestamps
- **Event Timeline** — Color-coded live log
- **Download Report** — PDF and JSON export

---

## 🛠️ Installation

### Requirements
- Python 3.10
- Webcam

### Step 1 — Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install Libraries
```bash
pip install flask==3.0.3 opencv-python==4.9.0.80 mediapipe==0.10.9 numpy==1.26.4 ultralytics SpeechRecognition pyaudio requests reportlab
```

> ⚠️ If `pyaudio` fails:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

### Step 3 — Run
```bash
python app.py
```

### Step 4 — Open Browser
```
http://127.0.0.1:5000
```

---

## 🚀 How to Use

1. Open `http://127.0.0.1:5000` in browser
2. Click **"Live Detection"** from navbar
3. Click **"▶ Start Detection"** button
4. Monitor the dashboard in real-time
5. Click **"⏹ Stop"** to end session
6. Click **"⬇ Download Report (PDF)"** to save report

---

## 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Core language |
| Flask | Web framework |
| OpenCV | Video capture & image processing |
| MediaPipe | Face mesh & head pose detection |
| YOLOv8 (Ultralytics) | Phone & person detection |
| SpeechRecognition | Audio monitoring |
| ReportLab | PDF report generation |
| HTML / CSS / JS | Frontend dashboard |

---

## 📸 Evidence Storage

All suspicious activity evidence is automatically saved:

```
evidence/
├── looking_left_20250314_142205.jpg
├── looking_right_20250314_142310.jpg
├── clip_looking_away_20250314_142400.avi
├── phone_detected_20250314_142500.jpg
└── no_face_20250314_142600.jpg
```

---

## 📄 Report Output

Session report includes:
- Risk Score & Level
- Session Duration & Total Frames
- Head Pose Event Log
- Full Event Timeline
- Audio Detections
- Evidence Summary

---

*Cheating Guard AI — Final Project | AI-Based Online Proctoring System*
