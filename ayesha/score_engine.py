# score_engine.py
# Weighted risk scoring system for exam proctoring
# Live camera: time-based cooldown
# Video mode:  frame-based cooldown (no time.time() dependency)

import time

WEIGHTS = {
    "looking_away":     5,
    "looking_left":     3,
    "looking_right":    3,
    "looking_up":       2,
    "looking_down":     4,
    "multiple_persons": 15,
    "no_face":          10,
    "phone_detected":   10,
    "voice_detected":   5,
}

COOLDOWN_SEC    = 10    # live camera: 10 seconds
COOLDOWN_FRAMES = 150   # video mode: 150 frames (~5 sec at 30fps)

_score       = 0
_events      = []
_last_fired  = {}   # live mode: event -> last real time
_last_frame  = {}   # video mode: event -> last frame number
_video_mode  = False


def set_video_mode(enabled):
    """Call with True before processing uploaded video, False for live camera."""
    global _video_mode
    _video_mode = enabled


def fire_event(event_name, frame_num=0):
    """
    Add score for event (respects cooldown).
    Pass frame_num when in video mode.
    Returns pts added (0 if cooldown active).
    """
    global _score

    if _video_mode:
        last = _last_frame.get(event_name, -COOLDOWN_FRAMES)
        if frame_num - last < COOLDOWN_FRAMES:
            return 0
        _last_frame[event_name] = frame_num
    else:
        now  = time.time()
        last = _last_fired.get(event_name, 0)
        if now - last < COOLDOWN_SEC:
            return 0
        _last_fired[event_name] = now

    pts     = WEIGHTS.get(event_name, 0)
    _score += pts
    _events.append({
        "event":    event_name,
        "pts":      pts,
        "total":    _score,
        "time_str": time.strftime("%H:%M:%S"),
    })
    return pts


def get_score():
    return _score


def get_risk_level():
    if _score <= 20:   return "LOW"
    elif _score <= 50: return "MEDIUM"
    return "HIGH"


def get_events():
    return list(_events)


def get_event_count():
    return len(_events)


def reset():
    global _score, _events, _last_fired, _last_frame, _video_mode
    _score      = 0
    _events     = []
    _last_fired = {}
    _last_frame = {}
    _video_mode = False