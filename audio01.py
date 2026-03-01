import cv2
import time
import os
import numpy as np
import pyaudio
from ultralytics import YOLO

# ----------------------------
# Setup
# ----------------------------
model = YOLO("best.pt")

PERSON_CLASS = 3
PHONE_CLASS = 1

cap = cv2.VideoCapture(0)

person_state = None
person_start = None
last_person_seen = None

phone_state = None
phone_start = None
last_phone_seen = None

audio_state = None
audio_start = None

timeline_logs = []

start_time = time.time()
TOLERANCE = 0.7
AUDIO_THRESHOLD = 500      # adjust if needed
CONTINUOUS_LIMIT = 3       # seconds

# Folder
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

def save_screenshot(frame, event, timestamp):
    filename = f"screenshots/{event}_{timestamp:.2f}s.png"
    cv2.imwrite(filename, frame)
    print(f"📸 Screenshot saved: {filename}")

# ----------------------------
# AUDIO SETUP
# ----------------------------
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current = round(time.time() - start_time, 2)

    # ================= VIDEO DETECTION =================
    results = model(frame, verbose=False)

    num_persons = 0
    phone_detected = False

    for r in results:
        class_ids = r.boxes.cls.cpu().numpy()
        for cls in class_ids:
            if cls == PERSON_CLASS:
                num_persons += 1
            elif cls == PHONE_CLASS:
                phone_detected = True

    # ================= PERSON =================
    if num_persons > 0:
        last_person_seen = current
        person_state = "Detected"
    else:
        if person_state == "Detected" and (current - last_person_seen) > TOLERANCE:
            save_screenshot(frame, "Person_Missing", current)
            timeline_logs.append(f"❌ Person Missing at {current}s")
            person_state = "Missing"

    # ================= PHONE =================
    if phone_detected:
        if phone_state != "Detected":
            save_screenshot(frame, "Phone_Detected", current)
            timeline_logs.append(f"📱 Phone Detected at {current}s")
            phone_state = "Detected"
        last_phone_seen = current
    else:
        if phone_state == "Detected" and (current - last_phone_seen) > TOLERANCE:
            phone_state = None

    # ================= AUDIO MONITORING =================
    audio_data = stream.read(1024, exception_on_overflow=False)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    rms = np.sqrt(np.mean(audio_np**2))

    if rms > AUDIO_THRESHOLD:
        if audio_state != "Speaking":
            audio_state = "Speaking"
            audio_start = current
        elif (current - audio_start) > CONTINUOUS_LIMIT:
            save_screenshot(frame, "Continuous_Speech", current)
            timeline_logs.append(f"🎤 Continuous Speech at {current}s")
            audio_state = "Flagged"
    else:
        audio_state = None

    cv2.imshow("Proctoring Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()

print("\n=== Final Timeline Summary ===")
for log in timeline_logs:
    print(log)