"""glitch_in: the matte-cut subject GLITCHES into existence once a 3D cut has settled (user 2026-10-30 on own-video v003 base2 s2: "it should have done the movement transition just with the
background and the text coming on with no silhouette and then had the matted foreground element glitch into existence once the frame had settled").

During the camera move a shot that has a CLEAN PLATE shows only the plate (no subject card, no blurred hole, no silhouette); when the move has settled the subject (its own playing footage, cut by its
own matte) appears over ~GLITCH_FRAMES frames: a few on / off flickers, 1-3 horizontal slice bands shifted sideways and an RGB split, all crisp (no blur), shrinking to nothing, then it stays solid.
The same style as hold_life's glitch_slice. Pure numpy, deterministic in (seed, frame). `glitch_out` plays the flicker backwards for the A side at the start of a move.
"""
from __future__ import annotations

import random

import numpy as np

GLITCH_FRAMES = 8             # frames of the glitch-in (6-10)
SLICE_DX = (0.02, 0.06)       # slice offset, share of the frame width (x strength)
SPLIT_PX = (1, 3)             # RGB split per channel, px at 360 wide, scaled with the width


def flicker(n: int = GLITCH_FRAMES, seed: int = 0) -> list[float]:
    """alpha per frame of the glitch-in: starts off, a couple of on / off flickers, the last two frames solid (n frames; frame n and later = 1.0)."""
    rng = random.Random(f"glitch_in:{int(seed)}:{n}")
    a = [0.0] * n
    on = 1
    for i in range(1, n - 2):
        if rng.random() < 0.55:
            on = 1 - on
        a[i] = float(on)
    a[1] = 1.0 if a[1] == 0.0 and n > 3 and rng.random() < 0.5 else a[1]
    a[n - 2] = a[n - 1] = 1.0
    if n > 4 and a[n - 3] == a[n - 4]:
        a[n - 3] = 1.0 - a[n - 4]                       # at least one real flicker before it settles
    return a


def alpha_at(k: int, n: int = GLITCH_FRAMES, seed: int = 0) -> float:
    """0 before the glitch, the flicker value inside it, 1 after."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return flicker(n, seed)[k]


def _shifted(layer: np.ndarray, bands: list, split: int, mag: float) -> np.ndarray:
    out = layer.copy()
    for (y0, y1, dx) in bands:
        out[y0:y1] = np.roll(layer[y0:y1], int(round(dx * mag)), axis=1)
    if split > 0 and out.ndim == 3 and out.shape[-1] >= 3:
        r, b = out[..., 0].copy(), out[..., 2].copy()
        s = max(1, int(round(split * mag)))
        out[..., 0], out[..., 2] = np.roll(r, s, axis=1), np.roll(b, -s, axis=1)
    return out


def glitch_layer(rgb: np.ndarray, matte: np.ndarray, mag: float, seed: int = 0, k: int = 0):
    """the subject layer glitched at strength `mag` (1 = full, 0 = none): (rgb uint8 (H,W,3), matte float (H,W)) with the slice offsets and the RGB split applied to both. Crisp: integer
    pixel shifts only, bands and split drawn from (seed, k)."""
    h, w = matte.shape[:2]
    rng = random.Random(f"glitch_in:{int(seed)}:{int(k)}")
    bands = []
    if mag > 0.02:
        for _ in range(rng.randint(1, 3)):
            bh = max(3, int(h * rng.uniform(0.04, 0.14)))
            y0 = rng.randint(0, max(h - bh, 0))
            bands.append((y0, y0 + bh, rng.choice((-1, 1)) * rng.uniform(*SLICE_DX) * w))
    split = int(round(rng.uniform(*SPLIT_PX) * w / 360.0)) if mag > 0.02 else 0
    return _shifted(rgb, bands, split, mag), _shifted(matte[..., None], bands, 0, mag)[..., 0]


def step(k: int, n: int = GLITCH_FRAMES, seed: int = 0, reverse: bool = False) -> tuple[float, float]:
    """(alpha, strength) of glitch step k. Forward (glitch-in): k < 0 -> (0, 0); inside -> (flicker[k], 1 - k/n); k >= n -> (1, 0) = solid. Reverse (glitch-out): k < 0 -> (1, 0) solid; inside ->
    (flicker reversed, k/n); k >= n -> (0, 0) = gone."""
    if not reverse:
        return (0.0, 0.0) if k < 0 else ((1.0, 0.0) if k >= n else (flicker(n, seed)[k], 1.0 - k / float(n)))
    return (1.0, 0.0) if k < 0 else ((0.0, 0.0) if k >= n else (flicker(n, seed)[n - 1 - k], k / float(n)))


def composite(base: np.ndarray, subject: np.ndarray, matte: np.ndarray, k: int, n: int = GLITCH_FRAMES, seed: int = 0, reverse: bool = False):
    """the subject (its own frame, cut by its matte) over `base` (the plate frame) at glitch step k -> (rgb uint8, effective matte float32)."""
    a, mag = step(k, n, seed, reverse)
    m0 = np.clip(np.asarray(matte, np.float32), 0.0, 1.0)
    if a <= 0.0:
        return base.copy(), np.zeros(m0.shape[:2], np.float32)
    rgb, m = (subject, m0) if mag <= 0.0 else glitch_layer(subject, m0, mag, seed, k)
    m = (m * a).astype(np.float32)
    out = base.astype(np.float32) * (1.0 - m[..., None]) + rgb.astype(np.float32) * m[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), m
