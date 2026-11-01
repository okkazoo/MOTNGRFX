"""explore_ops2: round-2 EXPLORATION ops (2026-10-30, `16_app/core_check/explore_masks/GALLERY_round2.md`): dust key from the clean plate, mask zoom (open the letters to full
frame), weighted mask blend. Numpy ops, masks HxW 0..1, images HxWx3 0..1. Registered by `compose._ops_registry` under their own names only (never overriding), no default uses them."""
from __future__ import annotations

import cv2
import numpy as np


def _g(x, h=None, w=None):
    a = np.asarray(x, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[2] == 1 else a.mean(2)
    if h is not None and a.shape[:2] != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return a


def plate_diff_key(footage, plate, matte=None, lo=0.06, hi=0.20, blur=1.5, gain=1.0):
    """what the clean plate does NOT have (dust, spray, shadow the Alpha-Gen matte drops): smoothstep of |footage - plate| (colour distance, lightly blurred), max'ed with `matte`."""
    f = np.asarray(footage, np.float32)
    p = np.asarray(plate, np.float32)
    if p.shape[:2] != f.shape[:2]:
        p = cv2.resize(p, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = np.sqrt(((f - p) ** 2).sum(2) / 3.0)
    if blur > 0:
        d = cv2.GaussianBlur(d, (0, 0), float(blur))
    t = np.clip((d - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    k = np.clip(t * t * (3.0 - 2.0 * t) * float(gain), 0.0, 1.0)
    if matte is not None:
        k = np.maximum(k, np.clip(_g(matte, *f.shape[:2]), 0.0, 1.0))
    return k.astype(np.float32)


def mask_zoom(a, scale=1.0, cx=0.5, cy=0.5):
    """the mask scaled about (cx, cy) (frame fractions): scale may be a timeline, e.g. 1 -> 12 opens the letters until the frame is inside them (zoom through a letter)."""
    m = _g(a)
    H, W = m.shape
    s = max(float(scale), 1e-3)
    px, py = cx * W, cy * H
    M = np.float32([[s, 0, px - s * px], [0, s, py - s * py]])
    return cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mask_lerp(a, b, k=0.0):
    a, b = _g(a), _g(b)
    b = _g(b, *a.shape)
    kk = float(np.clip(k, 0.0, 1.0))
    return (a * (1.0 - kk) + b * kk).astype(np.float32)


def ink_over(base, alpha, color=(1.0, 1.0, 1.0), opacity=1.0):
    """a colour over base by alpha * opacity (opacity may be a timeline; alpha None = everywhere)."""
    im = np.asarray(base, np.float32)
    a0 = np.ones(im.shape[:2], np.float32) if alpha is None else _g(alpha, *im.shape[:2])
    al = np.clip(a0 * float(opacity), 0.0, 1.0)[..., None]
    return im * (1.0 - al) + np.asarray(color, np.float32).reshape(1, 1, 3) * al


def img_over(base, fg, alpha, opacity=1.0):
    """image fg over base by alpha * opacity (opacity may be a timeline; alpha None = everywhere)."""
    im = np.asarray(base, np.float32)
    f = np.asarray(fg, np.float32)
    if f.shape[:2] != im.shape[:2]:
        f = cv2.resize(f, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
    a0 = np.ones(im.shape[:2], np.float32) if alpha is None else _g(alpha, *im.shape[:2])
    al = np.clip(a0 * float(opacity), 0.0, 1.0)[..., None]
    return im * (1.0 - al) + f * al


OPS = {"plate_diff_key": plate_diff_key, "mask_zoom": mask_zoom, "mask_lerp": mask_lerp, "ink_over": ink_over, "img_over": img_over}
