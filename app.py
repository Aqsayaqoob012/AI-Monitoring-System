from flask import Flask, render_template, Response, jsonify
import cv2, time, os
from ultralytics import YOLO
import speech_recognition as sr
import requests

app = Flask(__name__)

# YOLO Setup
model = YOLO("best.pt")
PERSON_CLASS = 3
PHONE_CLASS = 1

# Global Variables
cap = cv2.VideoCapture(0)
person_state = None
person_start = None
last_person_seen = None
phone_state = None
phone_start = None
last_phone_seen = None
timeline_logs = []
detected_sentences = []
screenshots_list = []
running = False
TOLERANCE = 0.7
start_time = 0
stop_listening = None

# Folder
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Speech Recognition
r = sr.Recognizer()
mic = sr.Microphone()

def urdu_to_roman(text):
    url = "https://inputtools.google.com/request"
    params = {"text": text, "itc": "ur-t-i0-und", "num": 1}
    try:
        res = requests.get(url, params=params)
        result = res.json()
        if result[0]=="SUCCESS": return result[1][0][1][0]
    except: pass
    return text

def callback(recognizer, audio):
    try:
        txt = recognizer.recognize_google(audio)
        if any('\u0600' <= c <= '\u06FF' for c in txt):
            txt = urdu_to_roman(txt)
        detected_sentences.append(txt)
    except: pass

def save_screenshot(frame, event, timestamp):
    filename = f"screenshots/{event}_{timestamp:.2f}.png"
    cv2.imwrite(filename, frame)
    screenshots_list.append(filename)
    timeline_logs.append(f"📸 Screenshot saved: {filename}")

# Generator for Video Feed
def gen_frames():
    global person_state, person_start, last_person_seen
    global phone_state, phone_start, last_phone_seen, start_time, running

    start_time = time.time()
    with mic as source:
        r.adjust_for_ambient_noise(source)
    global stop_listening
    stop_listening = r.listen_in_background(mic, callback)

    while running:
        ret, frame = cap.read()
        if not ret: continue
        current = round(time.time() - start_time, 2)

        # YOLO detection
        results = model(frame, verbose=False)
        num_persons = 0
        phone_detected = False
        for r_box in results:
            class_ids = r_box.boxes.cls.cpu().numpy()
            for cls in class_ids:
                if cls==PERSON_CLASS: num_persons+=1
                elif cls==PHONE_CLASS: phone_detected=True

        # Person tracking
        if num_persons>0:
            if person_state!="Detected":
                if person_state is not None:
                    timeline_logs.append(f"Person {person_state} from {person_start}s to {current}s")
                person_state="Detected"
                person_start=current
            last_person_seen=current
        else:
            if person_state=="Detected" and (current - last_person_seen)>TOLERANCE:
                timeline_logs.append(f"Person Detected from {person_start}s to {last_person_seen}s")
                save_screenshot(frame,"Person_Missing",current)
                person_state="Missing"
                person_start=current

        # Phone tracking
        if phone_detected:
            if phone_state!="Detected":
                phone_state="Detected"
                phone_start=current
                save_screenshot(frame,"Phone_Detected",current)
            last_phone_seen=current
        else:
            if phone_state=="Detected" and (current - last_phone_seen)>TOLERANCE:
                timeline_logs.append(f"Phone detected from {phone_start}s to {last_phone_seen}s")
                phone_state=None

        # encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/live")
def live_page():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start")
def start():
    global running
    running = True
    return "Started"

@app.route("/stop")
def stop():
    global running
    running=False
    if stop_listening: stop_listening(wait_for_stop=False)
    current = round(time.time()-start_time,2)
    if person_state is not None: timeline_logs.append(f"Person {person_state} from {person_start}s to {current}s")
    if phone_state=="Detected": timeline_logs.append(f"Phone detected from {phone_start}s to {last_phone_seen}s")
    timeline_logs.append("=== Final Timeline Summary ===")
    for s in detected_sentences: timeline_logs.append(f"🎤 {s}")
    return "Stopped"

@app.route("/get_logs")
def get_logs():
    return jsonify({
        "screenshots": screenshots_list,
        "audio": detected_sentences,
        "timeline": timeline_logs
    })

if __name__=="__main__":
    running=True
    app.run(debug=True)