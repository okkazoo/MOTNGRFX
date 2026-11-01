"""Easing curves and the style.motion -> easing mapping.

snappy = expo-out, smooth = cubic-in-out, bouncy = back-out, glitch = steps (+ seeded
jitter applied by the caller).
"""
from __future__ import annotations

import math

import numpy as np

MOTIONS = ("snappy", "smooth", "bouncy", "glitch")


def clamp01(p: float) -> float:
    return 0.0 if p <= 0.0 else 1.0 if p >= 1.0 else float(p)


def linear(p): return clamp01(p)


def expo_out(p):
    p = clamp01(p)
    return 1.0 if p >= 1.0 else 1.0 - 2.0 ** (-10.0 * p)


def expo_in(p):
    p = clamp01(p)
    return 0.0 if p <= 0.0 else 2.0 ** (10.0 * (p - 1.0))


def expo_in_out(p):
    p = clamp01(p)
    if p <= 0.0 or p >= 1.0:
        return p
    return 0.5 * 2.0 ** (20.0 * p - 10.0) if p < 0.5 else 1.0 - 0.5 * 2.0 ** (-20.0 * p + 10.0)


def cubic_in(p):
    p = clamp01(p)
    return p * p * p


def cubic_out(p):
    p = clamp01(p)
    return 1.0 - (1.0 - p) ** 3


def cubic_in_out(p):
    p = clamp01(p)
    return 4.0 * p * p * p if p < 0.5 else 1.0 - (-2.0 * p + 2.0) ** 3 / 2.0


def back_out(p, s: float = 1.70158):
    p = clamp01(p) - 1.0
    return 1.0 + (s + 1.0) * p ** 3 + s * p ** 2


def back_in(p, s: float = 1.70158):
    p = clamp01(p)
    return (s + 1.0) * p ** 3 - s * p ** 2


def elastic_out(p):
    p = clamp01(p)
    if p <= 0.0 or p >= 1.0:
        return p
    return 2.0 ** (-10.0 * p) * math.sin((p * 10.0 - 0.75) * (2.0 * math.pi) / 3.0) + 1.0


def steps(p, n: int = 5):
    p = clamp01(p)
    return 1.0 if p >= 1.0 else math.floor(p * n) / n


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else float(x >= e1)
    return t * t * (3.0 - 2.0 * t)


def ease_in(p: float, motion: str) -> float:
    """Easing for entrances (0 -> 1)."""
    if motion == "smooth":
        return cubic_in_out(p)
    if motion == "bouncy":
        return back_out(p)
    if motion == "glitch":
        return steps(p, 5)
    return expo_out(p)


def ease_out(p: float, motion: str) -> float:
    """Easing for exits (0 -> 1 = fully gone)."""
    if motion == "smooth":
        return cubic_in_out(p)
    if motion == "bouncy":
        return back_in(p)
    if motion == "glitch":
        return steps(p, 4)
    return expo_in(p)


def ease_in_out(p: float, motion: str) -> float:
    """Easing for transitions."""
    if motion == "snappy":
        return expo_in_out(p)
    if motion == "glitch":
        return steps(p, 6)
    return cubic_in_out(p)


def decay_env(events: np.ndarray, t: float, tau: float = 0.15, attack: float = 0.0) -> float:
    """1 at the most recent event time <= t, decaying exponentially (with a short attack)."""
    if events is None or len(events) == 0:
        return 0.0
    i = int(np.searchsorted(events, t + 1e-6, side="right")) - 1
    if i < 0:
        return 0.0
    dt = t - float(events[i])
    if dt < attack:
        return max(0.0, dt / attack) if attack > 0 else 1.0
    return math.exp(-(dt - attack) / tau)
