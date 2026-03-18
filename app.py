# app.py — Cheating Guard AI (Uploaded Video + Live Camera)
from flask import Flask, render_template, Response, jsonify, request
import cv2, time, os
from ultralytics import YOLO
import speech_recognition as sr
from werkzeug.utils import secure_filename

# ── Your modules ──────────────────────────────────────────────────────────────
from ayesha.head_pose    import get_head_pose
from ayesha.score_engine import fire_event, get_score, get_risk_level, get_events, get_event_count, reset
from ayesha.evidence     import save_screenshot as ev_save, start_clip, record_tick, get_evidence_list, reset_evidence
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# YOLO Setup
model        = YOLO("best.pt")
PERSON_CLASS = 3
PHONE_CLASS  = 1

# ── Global State ──────────────────────────────────────────────────────────────
cap              = None
person_state     = None
person_start     = None
last_person_seen = None
phone_state      = None
phone_start      = None
last_phone_seen  = None
timeline_logs    = []
detected_sentences = []
screenshots_list = []
running          = False
TOLERANCE        = 0.7
start_time       = 0
stop_listening   = None
total_frames     = 0
session_duration = 0
direction_log    = []
last_direction   = "FORWARD"

# Folders
os.makedirs("screenshots", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Speech Recognition
r   = sr.Recognizer()
mic = sr.Microphone()

# ── Video Processing Generator ───────────────────────────────────────────────
def gen_frames(video_path=None):
    global cap, running, person_state, person_start, last_person_seen
    global phone_state, phone_start, last_phone_seen
    global start_time, total_frames, last_direction, direction_log

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("ERROR: Cannot open video source.")
        running = False
        return

    start_time = time.time()
    total_frames = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        total_frames += 1

        # YOLO Detection
        results = model(frame, verbose=False)
        num_persons = 0
        phone_detected = False
        for r_box in results:
            class_ids = r_box.boxes.cls.cpu().numpy()
            for cls in class_ids:
                if cls == PERSON_CLASS: num_persons += 1
                elif cls == PHONE_CLASS: phone_detected = True

        # Head Pose
        pose = get_head_pose(frame)
        direction = pose["direction"]
        if direction != last_direction:
            last_direction = direction
            direction_log.append({
                "direction": direction,
                "time_str": time.strftime("%H:%M:%S"),
                "away_sec": pose["away_seconds"],
            })

        # Evidence / Tick
        record_tick(frame)

        # Encode frame for streaming
        ret2, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    running = False

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/live")
def live_page():
    return render_template("index.html")

@app.route("/upload_video")
def upload_video_page():
    return render_template("upload_video.html")    

@app.route("/video_feed")
def video_feed():
    global running
    if not running:
        running = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload_video", methods=["POST"])
def upload_video():
    global running, cap, start_time, total_frames, session_duration, last_direction, direction_log
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join("uploads", filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 500

    running = True
    start_time = time.time()
    total_frames = 0
    session_duration = 0
    timeline_logs.clear()
    screenshots_list.clear()
    direction_log.clear()
    last_direction = "FORWARD"

    # Process full video for JSON response
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # YOLO & Head Pose
        results = model(frame, verbose=False)
        num_persons = 0
        phone_detected = False
        for r_box in results:
            class_ids = r_box.boxes.cls.cpu().numpy()
            for cls in class_ids:
                if cls == PERSON_CLASS: num_persons += 1
                elif cls == PHONE_CLASS: phone_detected = True

        # Head pose
        pose = get_head_pose(frame)
        direction = pose["direction"]
        if direction != last_direction:
            last_direction = direction
            direction_log.append({
                "direction": direction,
                "time_str": time.strftime("%H:%M:%S"),
                "away_sec": pose["away_seconds"],
            })

        # Tick & evidence
        record_tick(frame)

    session_duration = round(time.time() - start_time, 1)
    cap.release()
    running = False

    response = {
        "total_frames": total_frames,
        "session_duration": session_duration,
        "total_alerts": get_event_count(),
        "risk_level": get_risk_level(),
        "current_direction": last_direction,
        "evidence": get_evidence_list(),
        "student_analysis": get_score(),
        "score_events": get_events(),
        "timeline": timeline_logs,
        "audio": detected_sentences
    }

    return jsonify(response)

# ── Start / Stop / Reset ──────────────────────────────────────────────────────
@app.route("/start")
def start():
    global running
    running = True
    return "Started"

@app.route("/stop")
def stop():
    global running, session_duration
    running = False
    session_duration = round(time.time() - start_time, 1)
    return "Stopped"

@app.route("/reset")
def reset_all():
    global timeline_logs, detected_sentences, screenshots_list
    global person_state, phone_state, total_frames, session_duration, direction_log, last_direction
    reset()
    reset_evidence()
    timeline_logs      = []
    detected_sentences = []
    screenshots_list   = []
    person_state       = None
    phone_state        = None
    total_frames       = 0
    session_duration   = 0
    direction_log      = []
    last_direction     = "FORWARD"
    return "Reset OK"

if __name__ == "__main__":
    running = False
    app.run(debug=True, use_reloader=False)