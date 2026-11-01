"""Time shaping and cameras: eases by role, closed-form springs, Bezier / Hermite paths, 2D screen<->world
maps, 3D look-at + perspective projection, homographies and warps. Every function is a pure function of
its inputs (seekable); array arguments may be numpy or torch (xp dispatch).

Rules (from _info/ref-motion-rules-web.md):
- ease by role: arrivals long-tail ease-out (power4.out / expo.out), camera moves and colour floods
  power2.inOut, exits and zoom-throughs accelerate (power3.in), linear only for rotation / constant speed.
- springs by damping ratio: titles 0.85-1.0, objects 0.5-0.7, never on opacity / colour.
- tween log(scale) for zooms (perceptually even).
"""
from __future__ import annotations

import math

import numpy as np

import xp


def _clip01(x):
    return xp.clip(x, 0.0, 1.0) if xp.is_t(x) else np.clip(x, 0.0, 1.0)


# ------------------------------------------------------------------------------------------- eases
def smoothstep(a, b, x):
    """Hermite smoothstep from a to b (a > b allowed: a falling step)."""
    u = _clip01((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def smootherstep(x):
    u = _clip01(x)
    return u * u * u * (u * (6 * u - 15) + 10)


def bump(t, t0, t1):
    """0 outside [t0, t1], sin^2 hump inside (peak 1 at the middle)."""
    u = _clip01((t - t0) / (t1 - t0))
    return (xp.sin(math.pi * u) if xp.is_t(u) else np.sin(np.pi * u)) ** 2


def power_out(u, p=4):
    """GSAP powerN.out has exponent N+1: power2 = cubic, power3 = quart, power4 = quint."""
    return 1 - (1 - _clip01(u)) ** (p + 1)


def power_in(u, p=3):
    return _clip01(u) ** (p + 1)


def power_inout(u, p=2):
    u = _clip01(u)
    k = p + 1
    lo = 0.5 * (2 * u) ** k
    hi = 1 - 0.5 * (2 - 2 * u) ** k
    return xp.where(u < 0.5, lo, hi) if xp.is_t(u) else np.where(u < 0.5, lo, hi)


def expo_out(u):
    u = _clip01(u)
    e = xp.exp(-10 * u * math.log(2)) if xp.is_t(u) else np.exp(-10 * u * np.log(2))
    return (1 - e) / (1 - 2 ** -10)


def cubic_bezier(x1, y1, x2, y2):
    """CSS cubic-bezier(x1, y1, x2, y2) as an ease u -> v (Newton on x, numpy or torch)."""
    def f(u):
        u = _clip01(u)
        s = u * 1.0
        for _ in range(8):
            x = 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s * s * x2 + s ** 3 - u
            dx = 3 * (1 - s) ** 2 * x1 + 6 * (1 - s) * s * (x2 - x1) + 3 * s * s * (1 - x2)
            s = _clip01(s - x / (dx + 1e-6))
        return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s ** 3
    return f


ROLES = {
    "arrive": lambda u: power_out(u, 4),        # long-tail ease-out on arrivals
    "arrive_expo": expo_out,
    "camera": lambda u: power_inout(u, 2),      # camera moves, colour floods
    "flood": lambda u: power_inout(u, 2),
    "exit": lambda u: power_in(u, 3),           # exits, zoom-throughs accelerate
    "linear": lambda u: _clip01(u),
    "smooth": lambda u: smootherstep(u),
}


def ease(role: str, u):
    """ease('arrive' | 'arrive_expo' | 'camera' | 'flood' | 'exit' | 'linear' | 'smooth', u in [0, 1])."""
    return ROLES[role](u)


def ease_window(role, t, t0, t1):
    return ease(role, (t - t0) / (t1 - t0))


# ------------------------------------------------------------------------------------------- springs
SPRINGS = {               # role: (zeta, natural frequency Hz)
    "title": (0.9, 1.6),
    "object": (0.6, 1.8),
    "playful": (0.5, 1.2),
    "settle": (0.5, 1.2),  # E4's noise relax
    "camera": (1.0, 0.9),
}


def spring(tau, zeta=0.9, freq=1.6, x0=1.0, v0=0.0):
    """Closed-form damped spring displacement x(tau) released at tau = 0 from x0 with velocity v0, decaying
    to 0 (x0 for tau < 0). zeta < 1 underdamped, = 1 critical, > 1 overdamped. freq in Hz."""
    w = 2 * math.pi * freq
    tau = xp.maximum(tau, 0.0) if xp.is_t(tau) else np.maximum(tau, 0.0)
    exp = xp.exp if xp.is_t(tau) else np.exp
    if zeta < 1:
        wd = w * math.sqrt(1 - zeta * zeta)
        sin, cos = (xp.sin, xp.cos) if xp.is_t(tau) else (np.sin, np.cos)
        return exp(-zeta * w * tau) * (x0 * cos(wd * tau) + (v0 + zeta * w * x0) / wd * sin(wd * tau))
    if zeta == 1:
        return exp(-w * tau) * (x0 + (v0 + w * x0) * tau)
    r = w * math.sqrt(zeta * zeta - 1)
    a, b = -zeta * w + r, -zeta * w - r
    c2 = (v0 - a * x0) / (b - a)
    return (x0 - c2) * exp(a * tau) + c2 * exp(b * tau)


def spring_role(tau, role="title"):
    z, f = SPRINGS[role]
    return spring(tau, z, f)


def spring_decay(tau, zeta, omega):
    """E4's form: spring released from 1 at tau = 0, omega in rad/s (= spring(tau, zeta, omega / 2 pi))."""
    tau = xp.maximum(tau, 0.0)
    wd = omega * math.sqrt(1 - zeta ** 2)
    return xp.exp(-zeta * omega * tau) * (xp.cos(wd * tau) + zeta / math.sqrt(1 - zeta ** 2) * xp.sin(wd * tau))


def spring_to(t, t0, a, b, role="title"):
    """Value moving from a to b with a spring released at t0 (overshoot per role's zeta)."""
    return b + (a - b) * spring_role(t - t0, role)


def log_zoom(t, t0, t1, z0, z1, role="camera"):
    """Zoom tweened in log space (perceptually even)."""
    u = ease(role, (t - t0) / (t1 - t0))
    return math.exp(math.log(z0) + (math.log(z1) - math.log(z0)) * float(u))


# ------------------------------------------------------------------------------------------- paths
def bezier(p0, p1, p2, p3, u):
    """Cubic Bezier at u (scalar or array); points are array-likes of any dimension."""
    u = np.asarray(u, np.float64)[..., None]
    v = 1 - u
    return v ** 3 * np.asarray(p0) + 3 * v * v * u * np.asarray(p1) + 3 * v * u * u * np.asarray(p2) + u ** 3 * np.asarray(p3)


def bezier_path(keys, t, role="camera"):
    """Piecewise cubic Bezier through keys [(t, p), ...]: tangents from neighbours (Catmull-Rom handles),
    each segment eased by `role`. Clamps outside the key range."""
    ts = [k[0] for k in keys]
    P = [np.asarray(k[1], float) for k in keys]
    if t <= ts[0]:
        return P[0].copy()
    if t >= ts[-1]:
        return P[-1].copy()
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    m0 = (P[i + 1] - P[max(i - 1, 0)]) / 6.0
    m1 = (P[min(i + 2, len(P) - 1)] - P[i]) / 6.0
    u = float(ease(role, (t - ts[i]) / (ts[i + 1] - ts[i])))
    return bezier(P[i], P[i] + m0, P[i + 1] - m1, P[i + 1], u)


def hermite(keys, t):
    """Cubic Hermite through (time, pos, vel) keys; linear extrapolation outside (E1's camera keys)."""
    keys = sorted(keys, key=lambda k: k[0])
    if t <= keys[0][0]:
        return keys[0][1] + keys[0][2] * (t - keys[0][0])
    for (t0, p0, v0), (t1, p1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            dt = t1 - t0
            u = (t - t0) / dt
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1
    return keys[-1][1] + keys[-1][2] * (t - keys[-1][0])


# ------------------------------------------------------------------------------------------- 2D camera
def screen_to_world(px, py, cx, cy, zoom=1.0, roll=0.0, tx=0.0, ty=0.0):
    """Screen point -> world: about the screen centre (cx, cy), divide by zoom, rotate by roll, translate.
    (E4's camera: world = R(roll) (p - c) / zoom + c + t.)"""
    cr, sr = math.cos(roll), math.sin(roll)
    u, v = (px - cx) / zoom, (py - cy) / zoom
    return cr * u - sr * v + cx + tx, sr * u + cr * v + cy + ty


def world_to_screen(X, Y, cx, cy, zoom=1.0, roll=0.0, tx=0.0, ty=0.0):
    cr, sr = math.cos(roll), math.sin(roll)
    u, v = X - cx - tx, Y - cy - ty
    return (cr * u + sr * v) * zoom + cx, (-sr * u + cr * v) * zoom + cy


# ------------------------------------------------------------------------------------------- 3D camera
def rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """View matrix rows (right, up, forward) for a camera at eye looking at target."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    return np.stack([r, u, f])


def project(X, eye, Mv, f_px, W, H, near=1e-3):
    """Perspective projection of world points X (..., 3) -> screen (..., 2), depth z, visible mask.
    Mv rows = (right, up, forward). Works on numpy or torch X (Mv / eye as numpy)."""
    if xp.is_t(X):
        d = (X - xp.const(X, eye)) @ xp.const(X, Mv).T
    else:
        d = (np.asarray(X, float) - eye) @ np.asarray(Mv).T
    z = d[..., 2]
    ok = z > near
    zz = xp.where(ok, z, 1.0) if xp.is_t(z) else np.where(ok, z, 1.0)
    sx = W / 2 + f_px * d[..., 0] / zz
    sy = H / 2 - f_px * d[..., 1] / zz
    return (xp.stack([sx, sy], -1) if xp.is_t(sx) else np.stack([sx, sy], -1)), z, ok


def focal_px(fov_deg, W):
    return 0.5 * W / math.tan(math.radians(fov_deg) / 2)


# ------------------------------------------------------------------------------------------- homographies
def homography(src, dst):
    """3x3 H with dst ~ H src for four point pairs (DLT, numpy float64; = cv2.getPerspectiveTransform)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def apply_h(Hm, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(Hm).T
    return p[:, :2] / p[:, 2:3]


def warp(src, Hm, W, H, wrap=False, border="replicate"):
    """dst(x, y) = src(H^-1 (x, y)): cv2.warpPerspective(src, Hm, (W, H)) on numpy, grid_sample on torch
    (pixel centres at integer coords, as cv2). src: (h, w) or (C, h, w) tensor / (h, w) array."""
    if not xp.is_t(src):
        import cv2
        bm = cv2.BORDER_WRAP if wrap else (cv2.BORDER_REPLICATE if border == "replicate" else cv2.BORDER_CONSTANT)
        return cv2.warpPerspective(src, np.asarray(Hm, float), (W, H), flags=cv2.INTER_LINEAR, borderMode=bm)
    T = xp.torch()
    dev = src.device
    Hi = T.as_tensor(np.linalg.inv(Hm), dtype=T.float32, device=dev)
    v, u = T.meshgrid(T.arange(H, dtype=T.float32, device=dev), T.arange(W, dtype=T.float32, device=dev), indexing="ij")
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    x = (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den
    y = (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den
    s = src if src.dim() == 3 else src[None]
    h, w = s.shape[-2:]
    if wrap:
        x, y = T.remainder(x, w), T.remainder(y, h)
        s = T.nn.functional.pad(s[None], (0, 1, 0, 1), mode="circular")[0]
        h, w = h + 1, w + 1
    g = T.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)[None]
    out = T.nn.functional.grid_sample(s[None], g, mode="bilinear",
                                      padding_mode="border" if border == "replicate" else "zeros",
                                      align_corners=True)[0]
    return out if src.dim() == 3 else out[0]
