# evidence.py
# Saves screenshots and 5-second video clips on suspicious events
# All evidence stored in evidence/ folder with timestamps

import cv2
import os
import time

EVIDENCE_DIR = "evidence"

if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

_clip_writer    = None
_clip_start     = None
_clip_recording = False
_clip_path      = None
CLIP_DURATION   = 5
CLIP_FPS        = 20

# Public list — app.py reads this to show on dashboard
evidence_list = []   # each item: {type, event, path, time_str, direction}


def save_screenshot(frame, event_name, direction=""):
    """Save screenshot and add to evidence_list."""
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{EVIDENCE_DIR}/{event_name}_{ts}.jpg"
    cv2.imwrite(filename, frame)
    evidence_list.append({
        "type":      "screenshot",
        "event":     event_name,
        "path":      filename,
        "time_str":  time.strftime("%H:%M:%S"),
        "direction": direction,
    })
    return filename


def start_clip(frame, event_name):
    """Start recording a 5-sec clip (won't start if already recording)."""
    global _clip_writer, _clip_start, _clip_recording, _clip_path
    if _clip_recording:
        return

    h, w      = frame.shape[:2]
    ts        = time.strftime("%Y%m%d_%H%M%S")
    _clip_path = f"{EVIDENCE_DIR}/clip_{event_name}_{ts}.avi"
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    _clip_writer    = cv2.VideoWriter(_clip_path, fourcc, CLIP_FPS, (w, h))
    _clip_start     = time.time()
    _clip_recording = True
    evidence_list.append({
        "type":      "clip",
        "event":     event_name,
        "path":      _clip_path,
        "time_str":  time.strftime("%H:%M:%S"),
        "direction": "",
    })


def record_tick(frame):
    """Call every frame to write clip data. Auto-stops after CLIP_DURATION."""
    global _clip_writer, _clip_recording
    if not _clip_recording or not _clip_writer:
        return
    _clip_writer.write(frame)
    if time.time() - _clip_start >= CLIP_DURATION:
        _clip_writer.release()
        _clip_writer    = None
        _clip_recording = False


def get_evidence_list():
    return list(evidence_list)


def get_screenshot_count():
    return sum(1 for e in evidence_list if e["type"] == "screenshot")


def reset_evidence():
    global evidence_list
    evidence_list = []
