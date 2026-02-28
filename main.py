import cv2
import time
import os
from ultralytics import YOLO

# Setup

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

timeline_logs = []

start_time = time.time()
TOLERANCE = 0.7  # seconds

# Folder for screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

def save_screenshot(frame, event, timestamp):
    """Save screenshot with event info"""
    filename = f"screenshots/{event}_{timestamp:.2f}s.png"
    cv2.imwrite(filename, frame)
    print(f"📸 Screenshot saved: {filename}")


# Main Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current = round(time.time() - start_time, 2)

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

    # PERSON TRACKING 
    if num_persons > 0:
        if person_state != "Detected":
            if person_state is not None:
                timeline_logs.append(
                    f"Person {person_state} from {person_start}s to {current}s"
                )
            person_state = "Detected"
            person_start = current

        last_person_seen = current

    else:
        if person_state == "Detected" and (current - last_person_seen) > TOLERANCE:
            timeline_logs.append(
                f"Person Detected from {person_start}s to {last_person_seen}s"
            )
            #  Screenshot for Person Missing
            save_screenshot(frame, "Person_Missing", current)
            person_state = "Missing"
            person_start = current

    #  PHONE TRACKING 
    if phone_detected:
        if phone_state != "Detected":
            phone_state = "Detected"
            phone_start = current
            #  Screenshot for Phone Detected
            save_screenshot(frame, "Phone_Detected", current)
        last_phone_seen = current
    else:
        if phone_state == "Detected" and (current - last_phone_seen) > TOLERANCE:
            timeline_logs.append(
                f"Phone detected from {phone_start}s to {last_phone_seen}s"
            )
            phone_state = None

    cv2.imshow("Proctoring Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# END OF SESSION 
end_time = round(time.time() - start_time, 2)

if person_state is not None:
    timeline_logs.append(
        f"Person {person_state} from {person_start}s to {end_time}s"
    )

if phone_state == "Detected":
    timeline_logs.append(
        f"Phone detected from {phone_start}s to {last_phone_seen}s"
    )

cap.release()
cv2.destroyAllWindows()

#  TIMELINE SUMMARY
print("\n Final Timeline Summary")
for log in timeline_logs:
    print(log)