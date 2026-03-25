''' # head_pose.py
# Detects head direction: LEFT, RIGHT, UP, DOWN, FORWARD
# Uses MediaPipe Face Mesh — no solvePnP, accurate & fast

import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
NOSE_TIP    = 1
FOREHEAD    = 10
CHIN        = 152
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454

# Direction thresholds
YAW_LEFT_TH   = 0.35
YAW_RIGHT_TH  = 0.65
PITCH_UP_TH   = 0.33
PITCH_DOWN_TH = 0.60

AWAY_THRESHOLD_SEC = 5   # seconds before alert fires

_face_mesh          = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
_looking_away_since = None


def get_head_pose(frame):
    """
    Process frame → return pose dict.

    Returns:
        face_count   : int
        direction    : "FORWARD" | "LEFT" | "RIGHT" | "UP" | "DOWN"
        looking_away : bool
        away_seconds : float
        alert        : "looking_away" | None
    """
    global _looking_away_since

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = _face_mesh.process(rgb)

    face_count   = 0
    direction    = "FORWARD"
    looking_away = False
    alert        = None
    away_seconds = 0.0

    if res.multi_face_landmarks:
        face_count = len(res.multi_face_landmarks)
        lm = res.multi_face_landmarks[0].landmark

        nx = lm[NOSE_TIP].x    * w
        ny = lm[NOSE_TIP].y    * h
        lx = lm[LEFT_CHEEK].x  * w
        rx = lm[RIGHT_CHEEK].x * w
        ty = lm[FOREHEAD].y    * h
        by = lm[CHIN].y        * h

        face_w  = max(rx - lx, 1)
        face_h  = max(by - ty, 1)
        h_ratio = (nx - lx) / face_w
        v_ratio = (ny - ty) / face_h

        if   h_ratio < YAW_LEFT_TH:   direction = "LEFT"
        elif h_ratio > YAW_RIGHT_TH:  direction = "RIGHT"
        elif v_ratio < PITCH_UP_TH:   direction = "UP"
        elif v_ratio > PITCH_DOWN_TH: direction = "DOWN"

        looking_away = (direction != "FORWARD")

    if looking_away:
        if _looking_away_since is None:
            _looking_away_since = time.time()
        away_seconds = time.time() - _looking_away_since
        if away_seconds >= AWAY_THRESHOLD_SEC:
            alert = "looking_away"
    else:
        _looking_away_since = None

    return {
        "face_count":   face_count,
        "direction":    direction,
        "looking_away": looking_away,
        "away_seconds": round(away_seconds, 1),
        "alert":        alert,
    } '''




import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
NOSE_TIP    = 1
LEFT_EYE    = 33
RIGHT_EYE   = 263

AWAY_THRESHOLD_SEC = 5   # seconds before alert fires

_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
_looking_away_since = None


def get_head_pose(frame):
    """
    Process frame → return pose dict.

    Returns:
        face_count   : int
        direction    : "FORWARD" | "LEFT" | "RIGHT" | "UP" | "DOWN"
        looking_away : bool
        away_seconds : float
        alert        : "looking_away" | None
    """
    global _looking_away_since

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)

    face_count = 0
    direction = "FORWARD"
    looking_away = False
    alert = None
    away_seconds = 0.0

    if res.multi_face_landmarks:
        face_count = len(res.multi_face_landmarks)
        lm = res.multi_face_landmarks[0].landmark

        # Nose and eyes in pixels
        nx, ny = int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)
        lx, ly = int(lm[LEFT_EYE].x * w), int(lm[LEFT_EYE].y * h)
        rx, ry = int(lm[RIGHT_EYE].x * w), int(lm[RIGHT_EYE].y * h)

        # Eye center
        eye_center_x = (lx + rx) // 2
        eye_center_y = (ly + ry) // 2

        # Difference
        dx = nx - eye_center_x
        dy = ny - eye_center_y

        # Thresholds for direction
        LEFT_THRESHOLD = -20
        RIGHT_THRESHOLD = 20
        DOWN_THRESHOLD = 60

        if -120 < dx < LEFT_THRESHOLD:
            direction = "LEFT"
        elif 20 < dx > RIGHT_THRESHOLD:
            direction = "RIGHT"
        elif dy > DOWN_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FORWARD"

    return {
        "face_count": face_count,
        "direction": direction,
        "looking_away": looking_away,
        "away_seconds": round(away_seconds, 1),
        "alert": alert,
    }
