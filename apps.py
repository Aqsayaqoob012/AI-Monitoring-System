# app.py — Cheating Guard AI (Uploaded Video + Live Camera)
from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2, time, os
from ultralytics import YOLO
import speech_recognition as sr
from werkzeug.utils import secure_filename

# ── Your modules ──────────────────────────────────────────────────────────────
from ayesha.head_pose    import get_head_pose
from ayesha.score_engine import fire_event, get_score, get_risk_level, get_events, get_event_count, reset, set_video_mode
from ayesha.evidence     import save_screenshot as ev_save, start_clip, record_tick, get_evidence_list, get_screenshot_count, reset_evidence
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# YOLO Setup
model = YOLO("best.pt")
# Class IDs (YOLO)
BOOK_CLASS = 0
PHONE_CLASS = 1
HEADPHONE_CLASS = 2
PERSON_CLASS = 3

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
os.makedirs("uploads",     exist_ok=True)
os.makedirs("evidence",    exist_ok=True)

# Speech Recognition
r   = sr.Recognizer()
mic = sr.Microphone()

# ── Video Processing Generator (Live Camera) ─────────────────────────────────
def gen_frames():
    global running, person_state, person_start, last_person_seen
    global phone_state, phone_start, last_phone_seen
    global start_time, total_frames, last_direction, direction_log

    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        if running:
            total_frames += 1
            current = round(time.time() - start_time, 2)

            # YOLO
            results = model(frame, verbose=False)
            num_persons = 0
            phone_detected = False
            for r_box in results:
                class_ids = r_box.boxes.cls.cpu().numpy()
                for cls in class_ids:
                    if cls == PERSON_CLASS: num_persons += 1
                    elif cls == PHONE_CLASS: phone_detected = True

            # Person tracking
            if num_persons > 0:
                if person_state != "Detected":
                    person_state = "Detected"
                    person_start = current
                last_person_seen = current
            else:
                if person_state == "Detected" and last_person_seen and (current - last_person_seen) > TOLERANCE:
                    pts = fire_event("no_face")
                    if pts:
                        timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ No Face +{pts} pts | Score: {get_score()}")
                        ev_save(frame, "no_face")
                    person_state = "Missing"

            # Phone tracking
            if phone_detected:
                if phone_state != "Detected":
                    phone_state = "Detected"
                    pts = fire_event("phone_detected")
                    if pts:
                        timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Phone Detected +{pts} pts | Score: {get_score()}")
                        ev_save(frame, "phone_detected")
                last_phone_seen = current
            else:
                if phone_state == "Detected" and last_phone_seen and (current - last_phone_seen) > TOLERANCE:
                    phone_state = None

            # Multiple persons
            if num_persons > 1:
                pts = fire_event("multiple_persons")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Multiple Persons +{pts} pts | Score: {get_score()}")

            # Head Pose
            pose = get_head_pose(frame)
            direction = pose["direction"]
            if direction != last_direction:
                last_direction = direction
                if direction != "FORWARD":
                    direction_log.append({"direction": direction, "time_str": time.strftime("%H:%M:%S"), "away_sec": pose["away_seconds"]})
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] 👁 Looking {direction}")

            if direction == "LEFT":
                pts = fire_event("looking_left")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking LEFT +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "looking_left", "LEFT")
            elif direction == "RIGHT":
                pts = fire_event("looking_right")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking RIGHT +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "looking_right", "RIGHT")
            elif direction == "UP":
                pts = fire_event("looking_up")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking UP +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "looking_up", "UP")
            elif direction == "DOWN":
                pts = fire_event("looking_down")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking DOWN +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "looking_down", "DOWN")

            if pose["alert"] == "looking_away":
                pts = fire_event("looking_away")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking Away >5s +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "looking_away", direction)

            record_tick(frame)

        ret2, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    running = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── UPLOAD VIDEO — FULL DETECTION ────────────────────────────────────────────
@app.route("/upload_video", methods=["POST"])
def upload_video():
    global total_frames, session_duration, last_direction, direction_log
    global person_state, phone_state, last_person_seen, last_phone_seen

    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Reset everything for fresh analysis
    reset()
    reset_evidence()
    set_video_mode(True)   # ← frame-based cooldown for video
    timeline_logs.clear()
    screenshots_list.clear()
    direction_log.clear()
    last_direction   = "FORWARD"
    person_state     = None
    phone_state      = None
    last_person_seen = None
    last_phone_seen  = None
    total_frames     = 0

    filename   = secure_filename(file.filename)
    video_path = os.path.join("uploads", filename)
    file.save(video_path)

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 500

    fps        = video_cap.get(cv2.CAP_PROP_FPS) or 25
    start      = time.time()
    frame_num  = 0

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_num  += 1
        total_frames = frame_num
        current    = round(frame_num / fps, 2)  # video time in seconds

        # ── YOLO ─────────────────────────────────────────────────────────
        results        = model(frame, verbose=False)
        num_persons    = 0
        phone_detected = False
        for r_box in results:
            class_ids = r_box.boxes.cls.cpu().numpy()
            for cls in class_ids:
                if cls == PERSON_CLASS: num_persons += 1
                elif cls == PHONE_CLASS: phone_detected = True

        # Person tracking
        if num_persons > 0:
            if person_state != "Detected":
                person_state     = "Detected"
                person_start     = current
            last_person_seen = current
        else:
            if person_state == "Detected" and last_person_seen and (current - last_person_seen) > TOLERANCE:
                pts = fire_event("no_face", frame_num)
                if pts:
                    timeline_logs.append(f"[{current}s] ⚠ No Face +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "no_face")
                    start_clip(frame, "no_face")
                person_state = "Missing"

        # Phone tracking
        if phone_detected:
            if phone_state != "Detected":
                phone_state  = "Detected"
                phone_start  = current
                pts = fire_event("phone_detected", frame_num)
                if pts:
                    timeline_logs.append(f"[{current}s] ⚠ Phone Detected +{pts} pts | Score: {get_score()}")
                    ev_save(frame, "phone_detected")
                    start_clip(frame, "phone_detected")
            last_phone_seen = current
        else:
            if phone_state == "Detected" and last_phone_seen and (current - last_phone_seen) > TOLERANCE:
                phone_state = None

        # Multiple persons
        if num_persons > 1:
            pts = fire_event("multiple_persons", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Multiple Persons +{pts} pts | Score: {get_score()}")
                ev_save(frame, "multiple_persons")

        # ── Head Pose ─────────────────────────────────────────────────────
        pose      = get_head_pose(frame)
        direction = pose["direction"]

        if direction != last_direction:
            last_direction = direction
            if direction != "FORWARD":
                direction_log.append({"direction": direction, "time_str": f"{current}s", "away_sec": pose["away_seconds"]})
                timeline_logs.append(f"[{current}s] 👁 Looking {direction}")

        if direction == "LEFT":
            pts = fire_event("looking_left", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Looking LEFT +{pts} pts | Score: {get_score()}")
                ev_save(frame, "looking_left", "LEFT")
        elif direction == "RIGHT":
            pts = fire_event("looking_right", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Looking RIGHT +{pts} pts | Score: {get_score()}")
                ev_save(frame, "looking_right", "RIGHT")
        elif direction == "UP":
            pts = fire_event("looking_up", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Looking UP +{pts} pts | Score: {get_score()}")
                ev_save(frame, "looking_up", "UP")
        elif direction == "DOWN":
            pts = fire_event("looking_down", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Looking DOWN +{pts} pts | Score: {get_score()}")
                ev_save(frame, "looking_down", "DOWN")

        if pose["alert"] == "looking_away":
            pts = fire_event("looking_away", frame_num)
            if pts:
                timeline_logs.append(f"[{current}s] ⚠ Looking Away >5s +{pts} pts | Score: {get_score()}")
                ev_save(frame, "looking_away", direction)

        record_tick(frame)

    video_cap.release()
    session_duration = round(time.time() - start, 1)

    # Build response
    score     = get_score()
    risk      = get_risk_level()
    evt_count = get_event_count()
    avg_score = round(score / max(evt_count, 1), 2) if evt_count else 0

    # Fix evidence paths for browser
    ev_list = []
    for e in get_evidence_list():
        e_copy       = dict(e)
        e_copy["path"] = "/evidence/" + os.path.basename(e["path"])
        ev_list.append(e_copy)

    return jsonify({
        "total_frames":     total_frames,
        "session_duration": session_duration,
        "total_alerts":     get_screenshot_count(),
        "risk_level":       risk,
        "current_direction": last_direction,
        "evidence":         ev_list,
        "student_analysis": {
            "avg_score":   avg_score,
            "max_score":   score,
            "severity":    risk,
            "event_count": evt_count,
        },
        "score_events": get_events(),
        "timeline":     timeline_logs,
        "audio":        detected_sentences,
        "direction_log": direction_log,
    })

# ── Serve evidence images ─────────────────────────────────────────────────────
@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_file(os.path.join("evidence", filename))

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
    global person_state, phone_state, total_frames, session_duration, direction_log, last_direction, running
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
    running            = True
    return "Reset OK"

@app.route("/get_logs")
def get_logs():
    score     = get_score()
    risk      = get_risk_level()
    evt_count = get_event_count()
    avg_score = round(score / max(evt_count, 1), 2) if evt_count else 0
    dur       = session_duration if session_duration else round(time.time()-start_time, 1) if start_time else 0

    ev_list = []
    for e in get_evidence_list():
        e_copy = dict(e)
        e_copy["path"] = "/evidence/" + os.path.basename(e["path"])
        ev_list.append(e_copy)

    return jsonify({
        "screenshots":       screenshots_list,
        "audio":             detected_sentences,
        "timeline":          timeline_logs,
        "score":             score,
        "risk_level":        risk,
        "total_frames":      total_frames,
        "total_alerts":      get_screenshot_count(),
        "session_duration":  dur,
        "evidence":          ev_list,
        "current_direction": last_direction,
        "direction_log":     direction_log,
        "student_analysis": {
            "avg_score":   avg_score,
            "max_score":   score,
            "severity":    risk,
            "event_count": evt_count,
        },
        "score_events": get_events(),
    })

if __name__ == "__main__":
    running = True
    app.run(debug=True, use_reloader=False)
