import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    direction = "FORWARD"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # Landmarks
            left_eye  = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose      = face_landmarks.landmark[1]

            # Convert to pixels
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            nx, ny = int(nose.x * w), int(nose.y * h)

            # Eye center
            eye_center_x = (lx + rx) // 2
            eye_center_y = (ly + ry) // 2

            # 🔥 Difference calculate
            dx = nx - eye_center_x
            dy = ny - eye_center_y

            # 🔥 Better thresholds
            if -120 < dx < -20:
                direction = "LEFT"
            elif 20 < dx > 25:
                direction = "RIGHT"
            elif dy > 35:
                direction = "DOWN"
            else:
                direction = "FORWARD"

            # Draw
            cv2.circle(frame, (nx, ny), 4, (0,255,0), -1)
            cv2.circle(frame, (lx, ly), 4, (255,0,0), -1)
            cv2.circle(frame, (rx, ry), 4, (255,0,0), -1)

    cv2.putText(frame, f"Direction: {direction}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()