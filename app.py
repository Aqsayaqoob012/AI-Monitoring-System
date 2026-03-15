# app.py — Cheating Guard AI
# Integrates: YOLO (partner) + HeadPose + ScoreEngine + Evidence (your work)

from flask import Flask, render_template, Response, jsonify, send_file
import cv2, time, os, io, json
from ultralytics import YOLO
import speech_recognition as sr
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ── Your modules ──────────────────────────────────────────────────────────────
from ayesha.head_pose    import get_head_pose
from ayesha.score_engine import fire_event, get_score, get_risk_level, get_events, get_event_count, reset
from ayesha.evidence     import save_screenshot as ev_save, start_clip, record_tick, get_evidence_list, get_screenshot_count, reset_evidence
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# YOLO Setup (partner)
model        = YOLO("best.pt")
PERSON_CLASS = 3
PHONE_CLASS  = 1

# ── Global State ──────────────────────────────────────────────────────────────
cap              = cv2.VideoCapture(0)
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

# Head pose direction tracking for dashboard
direction_log = []   # list of {direction, time_str, away_sec}
last_direction = "FORWARD"

# Folders
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Speech Recognition (partner)
r   = sr.Recognizer()
mic = sr.Microphone()

def urdu_to_roman(text):
    url = "https://inputtools.google.com/request"
    params = {"text": text, "itc": "ur-t-i0-und", "num": 1}
    try:
        res    = requests.get(url, params=params)
        result = res.json()
        if result[0] == "SUCCESS": return result[1][0][1][0]
    except: pass
    return text

def callback(recognizer, audio):
    try:
        txt = recognizer.recognize_google(audio)
        if any('\u0600' <= c <= '\u06FF' for c in txt):
            txt = urdu_to_roman(txt)
        detected_sentences.append(txt)
        pts = fire_event("voice_detected")
        if pts:
            timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] 🎤 Voice Detected  +{pts} pts")
    except: pass

def _save_ss(frame, event, timestamp):
    """Partner's screenshot function."""
    filename = f"screenshots/{event}_{timestamp:.2f}.png"
    cv2.imwrite(filename, frame)
    screenshots_list.append(filename)
    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] 📸 {event.replace('_',' ')}")

# ── Video Feed ────────────────────────────────────────────────────────────────
def gen_frames():
    global person_state, person_start, last_person_seen
    global phone_state, phone_start, last_phone_seen
    global start_time, running, total_frames
    global last_direction, direction_log

    start_time = time.time()
    with mic as source:
        r.adjust_for_ambient_noise(source)
    global stop_listening
    stop_listening = r.listen_in_background(mic, callback)

    while running:
        ret, frame = cap.read()
        if not ret: continue

        frame   = cv2.flip(frame, 1)
        current = round(time.time() - start_time, 2)
        total_frames += 1

        # ── YOLO Detection (partner) ──────────────────────────────────────
        results      = model(frame, verbose=False)
        num_persons  = 0
        phone_detected = False

        for r_box in results:
            class_ids = r_box.boxes.cls.cpu().numpy()
            for cls in class_ids:
                if cls == PERSON_CLASS: num_persons += 1
                elif cls == PHONE_CLASS: phone_detected = True

        # Person tracking
        if num_persons > 0:
            if person_state != "Detected":
                if person_state is not None:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] Person {person_state} from {person_start}s to {current}s")
                person_state = "Detected"
                person_start = current
            last_person_seen = current
        else:
            if person_state == "Detected" and (current - last_person_seen) > TOLERANCE:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] Person left frame ({person_start}s → {last_person_seen}s)")
                _save_ss(frame, "Person_Missing", current)
                pts = fire_event("no_face")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ No Face  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "no_face")
                start_clip(frame, "no_face")
                person_state = "Missing"
                person_start = current

        # Phone tracking
        if phone_detected:
            if phone_state != "Detected":
                phone_state = "Detected"
                phone_start = current
                _save_ss(frame, "Phone_Detected", current)
                pts = fire_event("phone_detected")
                if pts:
                    timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Phone Detected  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "phone_detected")
                start_clip(frame, "phone_detected")
            last_phone_seen = current
        else:
            if phone_state == "Detected" and (current - last_phone_seen) > TOLERANCE:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] Phone removed from view")
                phone_state = None

        # Multiple persons
        if num_persons > 1:
            pts = fire_event("multiple_persons")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Multiple Persons ({num_persons})  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "multiple_persons")

        # ── Head Pose Detection (YOUR WORK) ───────────────────────────────
        pose = get_head_pose(frame)
        direction = pose["direction"]

        # Log direction change
        if direction != last_direction:
            last_direction = direction
            if direction != "FORWARD":
                direction_log.append({
                    "direction": direction,
                    "time_str":  time.strftime("%H:%M:%S"),
                    "away_sec":  pose["away_seconds"],
                })
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] 👁 Student looking {direction}")

        # Fire score event per direction
        if direction == "LEFT":
            pts = fire_event("looking_left")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking LEFT  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "looking_left", "LEFT")
                start_clip(frame, "looking_left")
        elif direction == "RIGHT":
            pts = fire_event("looking_right")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking RIGHT  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "looking_right", "RIGHT")
                start_clip(frame, "looking_right")
        elif direction == "UP":
            pts = fire_event("looking_up")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking UP  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "looking_up", "UP")
        elif direction == "DOWN":
            pts = fire_event("looking_down")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking DOWN  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "looking_down", "DOWN")

        # 5-sec away alert
        if pose["alert"] == "looking_away":
            pts = fire_event("looking_away")
            if pts:
                timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠ Looking Away >5s  +{pts} pts  |  Score: {get_score()}")
                ev_save(frame, "looking_away", direction)
                start_clip(frame, "looking_away")

        # Evidence clip tick
        record_tick(frame)
        # ─────────────────────────────────────────────────────────────────

        ret2, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/live")
def live_page():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    global running
    running = True   # auto-start when feed is accessed
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    if stop_listening: stop_listening(wait_for_stop=False)
    current = round(time.time() - start_time, 2)
    if person_state is not None:
        timeline_logs.append(f"[{time.strftime('%H:%M:%S')}] Session ended — {current}s total")
    timeline_logs.append("=== Session Complete ===")
    for s in detected_sentences:
        timeline_logs.append(f"🎤 {s}")
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

@app.route("/get_logs")
def get_logs():
    score    = get_score()
    risk     = get_risk_level()
    ev_list  = get_evidence_list()
    ss_count = get_screenshot_count()
    dur      = session_duration if session_duration else round(time.time()-start_time,1) if start_time else 0

    evt_count = get_event_count()
    avg_score = round(score / max(evt_count,1), 2) if evt_count else 0

    return jsonify({
        "screenshots":       screenshots_list,
        "audio":             detected_sentences,
        "timeline":          timeline_logs,
        "score":             score,
        "risk_level":        risk,
        "total_frames":      total_frames,
        "total_alerts":      ss_count,
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

@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_file(os.path.join("evidence", filename))

@app.route("/download_report")
def download_report():
    score = get_score()
    risk  = get_risk_level()
    buf   = io.BytesIO()
    doc   = SimpleDocTemplate(buf, pagesize=letter,
                leftMargin=0.75*inch, rightMargin=0.75*inch,
                topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_s = ParagraphStyle('T', parent=styles['Title'],   fontSize=22, textColor=colors.HexColor('#0b1a2c'), spaceAfter=4)
    sub_s   = ParagraphStyle('S', parent=styles['Normal'],  fontSize=10, textColor=colors.grey, spaceAfter=16)
    head_s  = ParagraphStyle('H', parent=styles['Heading2'],fontSize=13, textColor=colors.HexColor('#1f3b6f'), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('B', parent=styles['Normal'],  fontSize=9,  textColor=colors.HexColor('#333'), spaceAfter=3)

    story.append(Paragraph("Cheating Guard AI — Session Report", title_s))
    story.append(Paragraph(f"Generated: {time.strftime('%d %B %Y, %I:%M %p')}", sub_s))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1f3b6f')))
    story.append(Spacer(1, 12))

    # Summary table
    story.append(Paragraph("Session Summary", head_s))
    risk_col = colors.HexColor('#e8453c') if risk=='HIGH' else colors.HexColor('#f0b429') if risk=='MEDIUM' else colors.HexColor('#2e7d32')
    dur = session_duration if session_duration else 0
    s_data = [
        ["Metric",            "Value"],
        ["Total Risk Score",  str(score)],
        ["Risk Level",        risk],
        ["Session Duration",  f"{dur}s"],
        ["Total Frames",      str(total_frames)],
        ["Evidence Captured", str(get_screenshot_count())],
        ["Voice Detections",  str(len(detected_sentences))],
    ]
    t = Table(s_data, colWidths=[2.8*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#1f3b6f')),
        ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
        ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1,2),(1,2),  risk_col),
        ('FONTNAME',      (1,2),(1,2),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,1),(-1,-1), 9),
        ('FONTNAME',      (0,1),(0,-1), 'Helvetica-Bold'),
        ('GRID',          (0,0),(-1,-1), 0.5, colors.HexColor('#ccc')),
        ('TOPPADDING',    (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ('ROWBACKGROUND', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Head pose log
    if direction_log:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))
        story.append(Paragraph("Head Pose Events", head_s))
        hp_data = [["Time", "Direction", "Away Duration"]]
        for d in direction_log:
            hp_data.append([d["time_str"], d["direction"], f"{d['away_sec']}s"])
        ht = Table(hp_data, colWidths=[1.5*inch, 2*inch, 2.8*inch])
        ht.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#1f3b6f')),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('GRID',          (0,0),(-1,-1), 0.5, colors.HexColor('#ccc')),
            ('TOPPADDING',    (0,0),(-1,-1), 5),
            ('BOTTOMPADDING', (0,0),(-1,-1), 5),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('ROWBACKGROUND', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        story.append(ht)
        story.append(Spacer(1, 14))

    # Timeline
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))
    story.append(Paragraph("Event Timeline", head_s))
    for log in timeline_logs:
        clean = log.replace("📸","[SS]").replace("🎤","[Audio]").replace("⚠","[!]").replace("👁","[Eye]")
        if "[SCORE]" in log or "+pts" in log or "pts" in log:
            p = Paragraph(f"<b>{clean}</b>", ParagraphStyle('sc', parent=body_s, textColor=colors.HexColor('#b8860b')))
        elif "[SS]" in clean:
            p = Paragraph(clean, ParagraphStyle('ss', parent=body_s, textColor=colors.HexColor('#1f3b6f')))
        elif "[Audio]" in clean:
            p = Paragraph(clean, ParagraphStyle('au', parent=body_s, textColor=colors.HexColor('#2e7d32')))
        else:
            p = Paragraph(clean, body_s)
        story.append(p)

    story.append(Spacer(1,14))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1f3b6f')))
    story.append(Spacer(1,6))
    story.append(Paragraph("Cheating Guard AI — AI Proctoring System",
        ParagraphStyle('ft', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf')

@app.route("/download_json")
def download_json():
    data = {
        "score":          get_score(),
        "risk_level":     get_risk_level(),
        "session_duration": session_duration,
        "total_frames":   total_frames,
        "evidence":       get_evidence_list(),
        "direction_log":  direction_log,
        "timeline":       timeline_logs,
        "audio":          detected_sentences,
        "score_events":   get_events(),
    }
    buf = io.BytesIO(json.dumps(data, indent=2).encode())
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                     mimetype='application/json')

if __name__ == "__main__":
    running = True
    app.run(debug=True)