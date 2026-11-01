"""Scene transitions. Each takes the outgoing frame A and incoming frame B (both
rendered at the same clip time) and a linear progress p in 0..1 across the window
centred on the scene boundary."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import Ctx, affine
from .easing import clamp01, cubic_in_out, ease_in_out, smoothstep
from .shapes import circle_pts, draw_ops, raster

TRANSITIONS = ("cut", "crossfade", "push", "wipe", "zoom_blur", "shape_reveal", "glitch",
               "luma_wipe", "flash")

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _lerp(A, B, m):
    if np.ndim(m) == 2:
        m = m[..., None]
    return A + (B - A) * m


def crossfade(A, B, p, ctx, spec, fi, poly):
    return _lerp(A, B, np.float32(cubic_in_out(p)))


def push(A, B, p, ctx, spec, fi, poly):
    H, W = A.shape[:2]
    e = ease_in_out(p, ctx.motion)
    d = spec.get("dir", "left")
    out = np.empty_like(A)
    if d in ("up", "down"):
        o = int(round(e * H))
        if d == "up":
            out[:H - o], out[H - o:] = A[o:], B[:o]
        else:
            out[o:], out[:o] = A[:H - o], B[H - o:]
    else:
        o = int(round(e * W))
        if d == "right":
            out[:, o:], out[:, :o] = A[:, :W - o], B[:, W - o:]
        else:
            out[:, :W - o], out[:, W - o:] = A[:, o:], B[:, :o]
    return out


def wipe(A, B, p, ctx, spec, fi, poly):
    H, W = A.shape[:2]
    e = ease_in_out(p, ctx.motion)
    d = spec.get("dir", "left")
    soft = max(1.0, 0.015 * ctx.S)
    if d in ("up", "down"):
        y = np.arange(H, dtype=np.float32) + 0.5
        edge = H * (1.0 - e) if d == "up" else H * e
        m = (y - edge) / soft + 0.5 if d == "up" else (edge - y) / soft + 0.5
        m = np.clip(m, 0, 1)[:, None]
        line = np.array([[0, edge], [W, edge]], np.float64)
    else:
        x = np.arange(W, dtype=np.float32) + 0.5
        edge = W * (1.0 - e) if d == "left" else W * e
        m = (x - edge) / soft + 0.5 if d == "left" else (edge - x) / soft + 0.5
        m = np.clip(m, 0, 1)[None, :]
        line = np.array([[edge, 0], [edge, H]], np.float64)
    out = A + (B - A) * m[..., None]
    if 0.0 < e < 1.0:
        draw_ops(out, [("line", line, False, max(2.0, 0.01 * ctx.S))], ctx.accent(0), math.sin(math.pi * p))
    return out


def _zoomblur(img, zoom, strength, n=6):
    H, W = img.shape[:2]
    c = (W / 2.0, H / 2.0)
    acc = None
    for k in range(n):
        z = zoom * (1.0 + strength * k / max(1, n - 1))
        M = affine(scale=z, pivot=c)
        Mt = M.copy()
        Mt[:, 2] = M[:, :2] @ np.array([0.5, 0.5]) + M[:, 2] - 0.5
        w = cv2.warpAffine(img, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc = w if acc is None else acc + w
    return acc * np.float32(1.0 / n)


def zoom_blur(A, B, p, ctx, spec, fi, poly):
    k = math.sin(math.pi * p)
    s = cubic_in_out(p)
    w = smoothstep(0.38, 0.62, p)
    strength = 0.22 * k
    out = None
    if w < 1.0:
        out = _zoomblur(A, 1.0 + 0.45 * s, strength)
    if w > 0.0:
        b = _zoomblur(B, 1.0 + 0.45 * (1.0 - s), strength)
        out = b if out is None else _lerp(out, b, np.float32(w))
    return out


def shape_reveal(A, B, p, ctx, spec, fi, poly):
    H, W = A.shape[:2]
    e = ease_in_out(p, "smooth")
    if e <= 0.0:
        return A.copy()
    if e >= 1.0:
        return B.copy()
    if poly is not None and len(poly) >= 3:
        c = (poly.min(0) + poly.max(0)) / 2.0
        r = np.hypot(*(poly - c).T)
        r_in = max(2.0, float(np.percentile(r, 30)))
        far = max(math.hypot(c[0] - x, c[1] - y) for x in (0, W) for y in (0, H))
        smax = min(400.0, far / r_in * 1.15)
        s = (e / 0.5) if e < 0.5 else smax ** ((e - 0.5) / 0.5)
        shape = (poly - c) * s + c
    else:
        c = np.array([W / 2.0, H / 2.0])
        shape = circle_pts(c[0], c[1], max(1.0, e * math.hypot(W, H) / 2.0 * 1.05))
    r = raster([("fill", shape)], W, H)
    m = np.zeros((H, W), np.float32)
    if r is not None:
        mask, x0, y0 = r
        m[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
    out = _lerp(A, B, m)
    draw_ops(out, [("line", shape, True, max(2.0, 0.008 * ctx.S))], ctx.accent(0), math.sin(math.pi * p))
    return out


def glitch(A, B, p, ctx, spec, fi, poly):
    H, W = A.shape[:2]
    k = math.sin(math.pi * p)
    base, other = (A, B) if p < 0.5 else (B, A)
    out = base.copy()
    r = ctx.rng(fi, 991)
    for _ in range(int(4 + 12 * k)):
        y0 = int(r.integers(0, H))
        hh = int(r.integers(2, max(3, int(0.09 * H))))
        sh = int(r.normal(0.0, 0.07 * W * k))
        src = other if r.random() < 0.35 * k else base
        out[y0:y0 + hh] = np.roll(src[y0:y0 + hh], sh, axis=1)
    dx = int(round(0.022 * W * k))
    if dx:
        out[..., 0] = np.roll(out[..., 0], dx, axis=1)
        out[..., 2] = np.roll(out[..., 2], -dx, axis=1)
    return out


def luma_wipe(A, B, p, ctx, spec, fi, poly):
    L = B @ _LUMA
    L = cv2.GaussianBlur(L, (0, 0), max(1.0, 0.012 * ctx.S))
    lo, hi = float(L.min()), float(L.max())
    L = (L - lo) / max(hi - lo, 1e-3)
    soft = 0.25
    e = cubic_in_out(p)
    m = np.clip((e * (1.0 + soft) - L) / soft, 0.0, 1.0)
    return _lerp(A, B, m)


def flash(A, B, p, ctx, spec, fi, poly):
    f = (1.0 - abs(2.0 * p - 1.0)) ** 0.6
    base = A if p < 0.5 else B
    light = np.maximum(ctx.palette[1] if len(ctx.palette) > 1 else np.ones(3, np.float32), 0.9)
    return base + (light[None, None, :] - base) * np.float32(0.95 * f)


def cut(A, B, p, ctx, spec, fi, poly):
    return A if p < 0.5 else B


FUNCS = {"cut": cut, "crossfade": crossfade, "push": push, "wipe": wipe, "zoom_blur": zoom_blur,
         "shape_reveal": shape_reveal, "glitch": glitch, "luma_wipe": luma_wipe, "flash": flash}


def blend(kind: str, A: np.ndarray, B: np.ndarray, p: float, ctx: Ctx, spec: dict, fi: int,
          poly=None) -> np.ndarray:
    """Unknown transition types fall back to crossfade."""
    fn = FUNCS.get(kind, crossfade)
    return fn(A, B, clamp01(p), ctx, spec or {}, fi, poly)
