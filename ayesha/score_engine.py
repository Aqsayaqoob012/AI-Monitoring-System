# score_engine.py
# Weighted risk scoring system for exam proctoring
# Tracks events, calculates risk level, maintains cooldown

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

COOLDOWN_SEC = 10  # same event won't score twice within this time

_score      = 0
_events     = []          # list of dicts: {event, pts, time_str}
_last_fired = {}


def fire_event(event_name):
    """
    Add score for event (respects cooldown).
    Returns pts added (0 if cooldown active).
    """
    global _score
    now  = time.time()
    last = _last_fired.get(event_name, 0)

    if now - last < COOLDOWN_SEC:
        return 0

    pts     = WEIGHTS.get(event_name, 0)
    _score += pts
    _last_fired[event_name] = now
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
    global _score, _events, _last_fired
    _score      = 0
    _events     = []
    _last_fired = {}
