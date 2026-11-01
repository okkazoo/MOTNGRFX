"""Print and film treatments: halftone, screenprint, xerox, duotone/tritone, film grade,
flat posterise, paper, grain, vignette, light leak, dust and scratches.

Every treatment takes and returns an HxWx3 float32 RGB image in 0..1 and is deterministic
(seed + frame index). Pixel sizes (halftone cell, misregistration, ...) are given in px at
the full-size 1080 px short side; pass ``scale`` = (short side in px) / 1080 so a draft
render looks the same as a full one (``apply_treatment`` does this for you).

Speed: the per-pixel work runs on uint8 planes with ``cv2.LUT`` tables (float32 tables give
the float32 output in one pass); screens, noise fields, dither tiles and paper textures are
cached per size, so a call costs a handful of full-frame passes.
"""
from __future__ import annotations

import inspect
import math
import zlib
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from scipy.special import ndtr

from .common import luminance, resolve_color

TREATMENTS = ("halftone", "screenprint", "xerox", "duotone", "tritone", "film_grade", "posterize_flat",
              "paper", "grain", "vignette", "light_leak", "dust_scratches")
FILM_PRESETS = ("mono", "teal_orange", "warm", "cool")
HALFTONE_SHAPES = ("dot", "line")
DITHERS = ("ordered", "noise", "diffusion", "none")

PAPER = "#f1ebdd"        # warm uncoated stock
TONER = "#161616"
REF_SHORT = 1080.0

_F32 = (np.arange(256, dtype=np.float32) / np.float32(255.0))
_TEX_K = 1.25            # paper texture is stored as tex / _TEX_K in uint8


# ----------------------------------------------------------------------------- small helpers

def _rgb(c, default=(1.0, 1.0, 1.0)) -> np.ndarray:
    return np.asarray(resolve_color(None, c, default), np.float32)


def _u8(img: np.ndarray) -> np.ndarray:
    """float 0..1 -> uint8 (rounded, saturated)."""
    return cv2.convertScaleAbs(img, alpha=255.0)


def _f32(u8: np.ndarray) -> np.ndarray:
    return cv2.LUT(u8, _F32)


def _gray8(u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)


def _mix_lut(c0, c1, as_u8=True) -> np.ndarray:
    """(256, 1, 3) table: index a -> c0 + (c1 - c0) * a / 255."""
    a = np.arange(256, dtype=np.float32)[:, None] / 255.0
    t = c0[None, :] + (c1 - c0)[None, :] * a
    if as_u8:
        return np.clip(t * 255.0 + 0.5, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return np.clip(t, 0, 1).astype(np.float32).reshape(256, 1, 3)


def _percentiles8(L8: np.ndarray, lo_q: float, hi_q: float):
    """Fast percentiles of a uint8 plane via its histogram."""
    hist = cv2.calcHist([L8], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(hist)
    n = cdf[-1]
    lo = int(np.searchsorted(cdf, lo_q * n))
    hi = int(np.searchsorted(cdf, hi_q * n))
    return lo, hi


def _stretch_lut(lo: int, hi: int, min_span: int = 24) -> np.ndarray:
    if hi - lo < min_span:      # near-flat image: widen the window but keep it inside 0..255
        lo = min(max(0.0, 0.5 * (hi + lo) - min_span / 2), 255.0 - min_span)
        hi = lo + min_span
    v = np.arange(256, dtype=np.float32)
    return np.clip((v - lo) / max(hi - lo, 1e-3) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _seed_rng(seed, *keys) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF] + [int(k) & 0xFFFFFFFF for k in keys])


def _blur_small(L8: np.ndarray, factor: float) -> np.ndarray:
    """Cheap wide blur: area-downscale by ``factor`` then bilinear back up."""
    H, W = L8.shape[:2]
    if factor <= 1.2:
        return L8
    sw, sh = max(2, int(round(W / factor))), max(2, int(round(H / factor)))
    small = cv2.resize(L8, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- cached fields

@lru_cache(maxsize=40)
def _noise_u(H: int, W: int, seed: int, tag: int, sigma: float) -> np.ndarray:
    """Uniform 0..1 float32 noise with spatial correlation ``sigma`` px (Gaussian -> CDF)."""
    rng = _seed_rng(seed, 5150, tag)
    if sigma > 6.0:      # low-frequency: build small, upsample
        f = sigma / 3.0
        sw, sh = max(4, int(W / f) + 2), max(4, int(H / f) + 2)
        n = rng.standard_normal((sh, sw), dtype=np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 3.0)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        n = rng.standard_normal((H, W), dtype=np.float32)
        if sigma > 0.3:
            n = cv2.GaussianBlur(n, (0, 0), sigma)
    n = (n - n.mean()) / max(float(n.std()), 1e-6)
    return ndtr(n).astype(np.float32)


@lru_cache(maxsize=40)
def _noise8(H: int, W: int, seed: int, tag: int, sigma: float) -> np.ndarray:
    return np.clip(_noise_u(H, W, seed, tag, sigma) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _bayer(n: int = 8) -> np.ndarray:
    m = np.array([[0, 2], [3, 1]], np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


@lru_cache(maxsize=8)
def _blue_tile(seed: int, n: int = 64) -> np.ndarray:
    """Blue-ish noise tile (high-passed white noise, rank-normalised to uniform 0..1)."""
    rng = _seed_rng(seed, 616)
    t = rng.random((n, n)).astype(np.float32)
    for _ in range(3):
        p = np.pad(t, 4, mode="wrap")
        t = t - cv2.GaussianBlur(p, (0, 0), 1.2)[4:-4, 4:-4]
        r = np.empty(t.size, np.float32)
        r[np.argsort(t, axis=None)] = (np.arange(t.size, dtype=np.float32) + 0.5) / t.size
        t = r.reshape(n, n)
    return t


@lru_cache(maxsize=16)
def _dither8(H: int, W: int, mode: str, px: int, seed: int) -> np.ndarray:
    tile = _bayer(8) if mode == "ordered" else _blue_tile(seed)
    if px > 1:
        tile = np.kron(tile, np.ones((px, px), np.float32))
    th, tw = tile.shape
    full = np.tile(tile, (H // th + 1, W // tw + 1))[:H, :W]
    return np.clip(full * 255.0, 0, 255).astype(np.uint8)


@lru_cache(maxsize=12)
def _paper_tex(H: int, W: int, fibre: float, speckle: float, seed: int, s: float) -> np.ndarray:
    """Paper luminance multiplier (float32 HxW, mean ~1)."""
    rng = _seed_rng(seed, 3030)
    s = max(s, 0.2)
    mottle = _noise_u(H, W, seed, 31, 60.0 * s) - 0.5
    tooth = _noise_u(H, W, seed, 32, 0.6 * s) - 0.5
    # fibres: anisotropic blurred noise in two orientations
    n = rng.standard_normal((H, W), dtype=np.float32)
    fa = cv2.GaussianBlur(n, (0, 0), sigmaX=7.0 * s, sigmaY=0.7 * s)
    n2 = rng.standard_normal((H, W), dtype=np.float32)
    fb = cv2.GaussianBlur(n2, (0, 0), sigmaX=0.7 * s, sigmaY=5.0 * s)
    fib = fa / max(float(fa.std()), 1e-6) + 0.6 * fb / max(float(fb.std()), 1e-6)
    tex = 1.0 + 0.07 * mottle + fibre * (0.022 * fib + 0.05 * tooth)
    if speckle > 0:
        cnt = int(speckle * H * W * 2.5e-5) + 1
        ys, xs = rng.integers(0, H, cnt), rng.integers(0, W, cnt)
        canvas = np.zeros((H, W), np.float32)
        for y, x, r, dark in zip(ys, xs, rng.uniform(0.5, 1.6, cnt) * s, rng.random(cnt) < 0.8):
            cv2.circle(canvas, (int(x), int(y)), max(1, int(round(r))), -0.35 if dark else 0.12, -1, cv2.LINE_AA)
        if s > 0.5:
            canvas = cv2.GaussianBlur(canvas, (0, 0), 0.5 * s)
        tex += canvas
    return np.clip(tex, 0.55, _TEX_K).astype(np.float32)


@lru_cache(maxsize=12)
def _paper_tex8(H: int, W: int, amount: float, fibre: float, speckle: float, seed: int, s: float) -> np.ndarray | None:
    """3-channel uint8 texture for ``cv2.multiply(.., scale=_TEX_K/255)``; None when amount ~ 0."""
    if amount <= 0.01:
        return None
    tex = 1.0 + amount * (_paper_tex(H, W, fibre, speckle, seed, s) - 1.0)
    t8 = np.clip(tex / _TEX_K * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(t8, cv2.COLOR_GRAY2BGR)


def _finish(out8: np.ndarray, tex8) -> np.ndarray:
    if tex8 is not None:
        out8 = cv2.multiply(out8, tex8, scale=_TEX_K / 255.0)
    return _f32(out8)


def _tex_for(H, W, texture, seed, scale):
    t = round(float(texture), 2)
    return _paper_tex8(H, W, t, 1.0, 0.6, int(seed) & 0xFFFF, round(float(scale), 3)) if t > 0.01 else None


# ----------------------------------------------------------------------------- paper

def paper(size, tone=PAPER, fibre: float = 0.5, speckle: float = 0.3, seed: int = 0, scale: float | None = None) -> np.ndarray:
    """Paper background texture. size = (H, W). Returns HxWx3 float32 (a new array)."""
    H, W = int(size[0]), int(size[1])
    s = min(H, W) / REF_SHORT if scale is None else float(scale)
    tex = _paper_tex(H, W, round(float(fibre), 3), round(float(speckle), 3), int(seed), round(s, 3))
    col = _rgb(tone, (0.945, 0.92, 0.867))
    return np.clip(cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR) * col[None, None, :], 0, 1).astype(np.float32)


def paper_overlay(img, tone=PAPER, fibre: float = 0.5, speckle: float = 0.3, amount: float = 1.0,
                  seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Multiply the image onto paper (tone tint + fibre/speckle texture)."""
    H, W = img.shape[:2]
    lut = _mix_lut(np.ones(3, np.float32), _rgb(tone), as_u8=False)[:, 0, :]
    tint = 1.0 + amount * (lut[255] - 1.0)
    out8 = _u8(img)
    tex8 = _paper_tex8(H, W, round(float(amount), 2), round(float(fibre), 3), round(float(speckle), 3),
                       int(seed) & 0xFFFF, round(float(scale), 3))
    if tex8 is not None:
        out8 = cv2.multiply(out8, tex8, scale=_TEX_K / 255.0)
    out8 = cv2.multiply(out8, tuple(float(v) for v in tint) + (1.0,))
    return _f32(out8)


# ----------------------------------------------------------------------------- halftone

def _dot_area(r: np.ndarray) -> np.ndarray:
    """Area of {|p| < r} inside a unit cell centred on the dot (r in cell units)."""
    r = np.asarray(r, np.float64)
    a = np.pi * r * r
    big = r > 0.5
    rb = r[big]
    a[big] -= 4.0 * (rb * rb * np.arccos(0.5 / rb) - 0.5 * np.sqrt(rb * rb - 0.25))
    return np.minimum(a, 1.0)


@lru_cache(maxsize=12)
def _screen8(H: int, W: int, cell: float, angle: float, shape: str):
    """uint8 distance (px) from each pixel to its dot centre / line centre, encoded as
    (d + 1) * 255 / span. Returns (screen, span)."""
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    ys = np.arange(H, dtype=np.float32)[:, None] + 0.5 - H / 2.0
    xs = np.arange(W, dtype=np.float32)[None, :] + 0.5 - W / 2.0
    v = (-xs * sa + ys * ca) / cell
    fv = v - np.floor(v) - 0.5
    if shape == "line":
        d = np.abs(fv) * cell
    else:
        u = (xs * ca + ys * sa) / cell
        fu = u - np.floor(u) - 0.5
        d = np.sqrt(fu * fu + fv * fv) * cell
    span = 0.75 * cell + 2.0
    return np.clip((d + 1.0) * 255.0 / span + 0.5, 0, 255).astype(np.uint8), span


@lru_cache(maxsize=24)
def _radius_lut(cell: float, shape: str, span: float, contrast: float) -> np.ndarray:
    """luminance byte -> needed dot radius (px), same encoding as ``_screen8``."""
    L = np.arange(256, dtype=np.float64) / 255.0
    D = np.clip((1.0 - L - 0.5) * contrast + 0.5, 0.0, 1.0)
    if shape == "line":
        R = D * 0.5
        R[D > 0.995] = 0.5 + 1.5 / cell
    else:
        rr = np.linspace(0.0, 0.7072, 4000)
        R = np.interp(D, _dot_area(rr), rr)
        R[D > 0.995] = 0.7072 + 1.5 / cell
    Rpx = R * cell
    Reff = Rpx - 0.5 * np.maximum(0.0, 1.0 - Rpx)        # no ink at all for pure white
    return np.clip((Reff + 1.0) * 255.0 / span + 0.5, 0, 255).astype(np.uint8)


def halftone(img, cell: float = 16.0, angle: float = 45.0, shape: str = "dot", ink=TONER, paper=PAPER,
             contrast: float = 1.0, auto_levels: bool = True, texture: float = 0.35, seed: int = 0,
             scale: float = 1.0) -> np.ndarray:
    """AM halftone screen: one ink on paper, dot area = local darkness, anti-aliased edges.

    cell: screen period in px at 1080 short side (min 4 px on screen). angle: screen angle, degrees.
    shape: "dot" (round dots that merge into a checker at ~78 % ink) | "line" (line screen).
    auto_levels stretches the 1st..99th luminance percentiles to the full tonal range first."""
    H, W = img.shape[:2]
    shape = shape if shape in HALFTONE_SHAPES else "dot"
    cpx = max(4.0, float(cell) * float(scale))
    cpx = round(cpx, 2)
    scr, span = _screen8(H, W, cpx, float(angle) % 180.0, shape)
    L8 = _gray8(_u8(img))
    L8 = _blur_small(L8, cpx / 1.6)                         # one tone per cell -> clean round dots
    lut = _radius_lut(cpx, shape, span, round(float(contrast), 3))
    if auto_levels:
        lut = lut[_stretch_lut(*_percentiles8(L8, 0.01, 0.99))]
    need = cv2.LUT(L8, lut)
    a8 = cv2.addWeighted(need, span, scr, -span, 127.5)     # alpha = clip(R - d + 0.5) in px
    out8 = cv2.LUT(cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR), _mix_lut(_rgb(paper), _rgb(ink, (0.08, 0.08, 0.08))))
    return _finish(out8, _tex_for(H, W, texture, seed, scale))


# ----------------------------------------------------------------------------- screenprint

@lru_cache(maxsize=24)
def _holes8(H: int, W: int, seed: int, tag: int, grit: float, s: float) -> np.ndarray:
    """Binary ink-starvation mask (255 = ink transfers, 0 = void): fine voids, denser in dry patches."""
    fine = _noise_u(H, W, seed, 200 + tag, 0.6 * s + 0.2)
    patch = _noise_u(H, W, seed, 300 + tag, 45.0 * s + 6.5)
    thr = 1.0 - grit * (0.025 + 0.35 * patch ** 4)
    return np.where(fine < thr, 255, 0).astype(np.uint8)


def _label_lut(T: float, dark: bool, label: int) -> np.ndarray:
    v = np.arange(256, dtype=np.float32)
    on = v < T if dark else v >= T
    return np.where(on, label, 0).astype(np.uint8)


def _shift8(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Integer translation with reflected borders (a view into a padded copy)."""
    if dx == 0 and dy == 0:
        return m
    H, W = m.shape[:2]
    p = max(abs(dx), abs(dy))
    big = cv2.copyMakeBorder(m, p, p, p, p, cv2.BORDER_REFLECT)
    return big[p - dy:p - dy + H, p - dx:p - dx + W]


def screenprint(img, inks=("#111111", "#ff4d2e"), paper=PAPER, misregister: float = 4.0, grit: float = 0.35,
                smooth: float = 0.35, dither: str = "noise", dither_amount: float = 0.5, dither_px: float = 2.0,
                texture: float = 0.35, seed: int = 0, frame: int = 0, boil: float = 0.0,
                scale: float = 1.0) -> np.ndarray:
    """Spot-colour screen print. Luminance is smoothed, posterised into (paper + inks) bands with
    equal tonal steps, and each ink is printed as an opaque layer, lightest first, with nested
    (trapped) masks so darker inks sit on lighter ones. Each layer is shifted by up to
    ``misregister`` px (at 1080), and ``grit`` adds ragged edges and ink-starved voids.

    dither: "noise" (blue-noise threshold) | "ordered" (Bayer) | "diffusion" (Floyd-Steinberg,
    ~3x slower) | "none"; dither_amount 0..1 is the band-edge transition width.
    boil > 0 re-rolls grit and jitters registration every 2 frames (hand-printed frames)."""
    H, W = img.shape[:2]
    ink_rgb = [_rgb(c, (0.1, 0.1, 0.1)) for c in list(inks)[:4]] or [_rgb("#111111")]
    pap = _rgb(paper)
    tones = [pap] + ink_rgb
    lums = np.array([luminance(t) for t in tones])
    order = list(np.argsort(lums, kind="stable"))            # rank -> tone index (dark -> light)
    nb = len(tones)
    p = order.index(0)
    step = 255.0 / nb
    s = float(scale)
    grit = float(np.clip(grit, 0.0, 1.0))
    bank = (int(frame) // 2) % 3 if boil > 0 else 0

    L8 = _gray8(_u8(img))
    lo, hi = _percentiles8(L8, 0.02, 0.98)
    Ls = cv2.LUT(L8, _stretch_lut(lo, hi))
    sig = (0.6 + 5.0 * float(np.clip(smooth, 0, 1))) * s
    if sig >= 0.5:
        Ls = cv2.GaussianBlur(Ls, (0, 0), sig)
    mode = dither if dither in DITHERS else "noise"
    if mode == "diffusion":
        centres = [int(round((b + 0.5) * step)) for b in range(nb)]
        flat = centres + [centres[-1]] * (256 - nb)
        pal = Image.new("P", (1, 1))
        pal.putpalette(sum(([c, c, c] for c in flat), []))
        idx = np.asarray(Image.fromarray(Ls, "L").convert("RGB").quantize(
            palette=pal, dither=Image.Dither.FLOYDSTEINBERG))
        Ld = cv2.LUT(np.ascontiguousarray(idx, np.uint8), np.array(flat, np.uint8))
    elif mode in ("ordered", "noise") and dither_amount > 0:
        amp = step * float(np.clip(dither_amount, 0, 1))
        dpx = max(1, int(round(float(dither_px) * s)))
        D8 = _dither8(H, W, mode, dpx, int(seed) & 0xFFFF)
        Ld = cv2.addWeighted(Ls, 1.0, D8, amp / 255.0, -0.5 * amp)
    else:
        Ld = Ls

    label = np.zeros((H, W), np.uint8)                       # 0 = paper, k = k-th tone printed
    print_order = sorted((r for r in range(nb) if r != p), key=lambda r: (abs(r - p), r))
    g = grit * step * 0.5
    lut_cols = np.zeros((256, 1, 3), np.uint8)
    lut_cols[0, 0] = np.clip(pap * 255.0 + 0.5, 0, 255).astype(np.uint8)
    for k, r in enumerate(print_order, start=1):
        ti = order[r]
        lut_cols[k, 0] = np.clip(tones[ti] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        dark = r < p
        T = (r + 1) * step if dark else r * step
        x = Ld
        if g > 0.5:     # ragged, slightly bled edges
            n8 = _noise8(H, W, int(seed), 10 + ti + 100 * bank, round(1.4 * s + 0.3, 3))
            x = cv2.addWeighted(Ld, 1.0, n8, g / 255.0, -0.5 * g)
        m = cv2.LUT(x, _label_lut(T, dark, k))
        if grit > 0.01:
            m = cv2.min(m, _holes8(H, W, int(seed), ti + 10 * bank, round(grit, 3), round(s, 3)))
        rng = _seed_rng(seed, 101, ti)
        ang = float(rng.uniform(0, 2 * np.pi))
        mag = float(misregister) * s * float(rng.uniform(0.6, 1.0))
        dx, dy = mag * math.cos(ang), mag * math.sin(ang)
        if boil > 0:
            jr = _seed_rng(seed, 102, ti, int(frame) // 2)
            dx += float(jr.uniform(-1, 1)) * boil * s
            dy += float(jr.uniform(-1, 1)) * boil * s
        label = cv2.max(label, _shift8(m, int(round(dx)), int(round(dy))))
    out8 = cv2.LUT(cv2.cvtColor(label, cv2.COLOR_GRAY2BGR), lut_cols)   # crisp ink edges, like a print
    return _finish(out8, _tex_for(H, W, texture, seed, scale))


# ----------------------------------------------------------------------------- xerox

@lru_cache(maxsize=8)
def _xerox_dirt8(H: int, W: int, seed: int, grit: float, s: float) -> np.ndarray:
    """Toner specks, drum streaks and a lid-edge shadow (uint8 amount to subtract)."""
    rng = _seed_rng(seed, 4040)
    canvas = np.zeros((H, W), np.uint8)
    cnt = int(grit * H * W * 6e-5) + 2
    for y, x, r, v in zip(rng.integers(0, H, cnt), rng.integers(0, W, cnt),
                          rng.uniform(0.4, 1.8, cnt) ** 2 * s, rng.integers(120, 256, cnt)):
        cv2.circle(canvas, (int(x), int(y)), max(1, int(round(r))), int(v), -1, cv2.LINE_AA)
    # a few clustered toner smudges
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        k = int(40 * grit) + 3
        for y, x in zip(rng.normal(cy, 18 * s, k), rng.normal(cx, 18 * s, k)):
            cv2.circle(canvas, (int(x), int(y)), max(1, int(round(rng.uniform(0.5, 1.5) * s))),
                       int(rng.integers(90, 200)), -1, cv2.LINE_AA)
    # vertical drum streaks
    streak = np.zeros((H, W), np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        x0 = int(rng.integers(0, W))
        w = max(1, int(round(rng.uniform(0.6, 3.0) * s)))
        prof = (0.5 + 0.5 * np.sin(ys * rng.uniform(3, 12) + rng.uniform(0, 6))) * rng.uniform(20, 55) * grit
        streak[:, x0:x0 + w] += prof[:, None]
    # lid-edge shadow along one side
    side = int(rng.integers(4))
    n = max(2, int(0.025 * min(H, W)))
    ramp = (np.linspace(1, 0, n, dtype=np.float32) ** 2) * 140 * grit
    if side == 0:
        streak[:n, :] += ramp[:, None]
    elif side == 1:
        streak[-n:, :] += ramp[::-1, None]
    elif side == 2:
        streak[:, :n] += ramp[None, :]
    else:
        streak[:, -n:] += ramp[None, ::-1]
    return cv2.add(canvas, np.clip(streak, 0, 255).astype(np.uint8))


@lru_cache(maxsize=8)
def _xerox_solid8(H: int, W: int, seed: int, grit: float, s: float) -> np.ndarray:
    """Uneven toner density in solids (low-frequency lift) plus tiny voids (uint8 amount to add)."""
    mott = _noise_u(H, W, seed, 4141, 30.0 * s + 6.5)
    fine = _noise_u(H, W, seed, 4142, 0.5 * s + 0.2)
    lift = (mott ** 2) * 38.0 * grit
    voids = np.where(fine > 1.0 - 0.004 * grit, 200.0, 0.0)
    return np.clip(lift + voids, 0, 255).astype(np.uint8)


def xerox(img, threshold: float = 0.5, contrast: float = 0.7, toner_grit: float = 0.45, paper="#efede6",
          crushed_blacks: float = 0.4, ink=TONER, texture: float = 0.3, seed: int = 0, frame: int = 0,
          boil: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Photocopy: auto-levelled, edge-enhanced, near-binary toner with crunchy noisy edges,
    toner specks and drum streaks, crushed solid blacks, on paper."""
    H, W = img.shape[:2]
    s = float(scale)
    grit = float(np.clip(toner_grit, 0.0, 1.0))
    con = float(np.clip(contrast, 0.0, 2.0))
    bank = (int(frame) // 2) % 3 if boil > 0 else 0
    L8 = _gray8(_u8(img))
    lo, hi = _percentiles8(L8, 0.01, 0.99)
    Ls = cv2.LUT(L8, _stretch_lut(lo, hi))
    low = _blur_small(Ls, max(2.0, 14.0 * s))
    k = 0.9 * con
    L2 = cv2.addWeighted(Ls, 1.0 + k, low, -k, 0.0)
    if s * (0.4 + grit) >= 0.4:      # soften edges first so the toner noise can break them up
        L2 = cv2.GaussianBlur(L2, (0, 0), (0.8 + 3.0 * grit) * s)
    g = 110.0 * grit
    if g > 0.5:     # fine toner noise + coarser clumping -> crunchy, broken edges and stippled mids
        n8 = _noise8(H, W, int(seed), 1 + bank, round(0.75 * s + 0.25, 3))
        L2 = cv2.addWeighted(L2, 1.0, n8, g / 255.0, -0.5 * g)
        n8 = _noise8(H, W, int(seed), 4 + bank, round(2.4 * s + 0.3, 3))
        L2 = cv2.addWeighted(L2, 1.0, n8, 1.3 * g / 255.0, -0.65 * g)
    x = np.arange(256, dtype=np.float64) / 255.0
    slope = 5.0 + 28.0 * con
    sig = lambda z: 1.0 / (1.0 + np.exp(-(z - float(threshold)) * slope))  # noqa: E731
    y = (sig(x) - sig(0.0)) / max(sig(1.0) - sig(0.0), 1e-6)
    cb = 0.45 * float(np.clip(crushed_blacks, 0.0, 1.0))
    y = np.clip((y - cb) / max(1.0 - cb - 0.04, 1e-3), 0, 1)              # crush blacks, blow whites
    v8 = cv2.LUT(L2, np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if grit > 0.01:
        v8 = cv2.add(v8, _xerox_solid8(H, W, int(seed) + bank, round(grit, 3), round(s, 3)))
        v8 = cv2.subtract(v8, _xerox_dirt8(H, W, int(seed) + bank, round(grit, 3), round(s, 3)))
    if s >= 0.6:
        v8 = cv2.GaussianBlur(v8, (0, 0), 0.55 * s)                       # toner spread
    out8 = cv2.LUT(cv2.cvtColor(v8, cv2.COLOR_GRAY2BGR), _mix_lut(_rgb(ink, (0.08, 0.08, 0.08)), _rgb(paper)))
    return _finish(out8, _tex_for(H, W, texture, seed, scale))


# ----------------------------------------------------------------------------- duotone / tritone

def duotone(img, colors=("#111111", "#f2f2f2"), exact: bool = False) -> np.ndarray:
    """Gradient map: luminance (2nd..98th percentile stretched) through 2+ evenly spaced colours.

    exact=True with 2 colours reproduces the legacy ``background.treatment.duotone`` maths
    bit-for-bit (float Rec.709 luminance, full-array percentiles); the default path is the fast
    uint8/LUT version (within ~1/255 of it)."""
    cols = [_rgb(c, (0.5, 0.5, 0.5)) for c in (colors or [])]
    if len(cols) < 2:
        cols = [_rgb("#111111"), _rgb("#f2f2f2")] if not cols else [cols[0], cols[0]]
    img = np.asarray(img, np.float32)
    if exact and len(cols) == 2:
        cd, cl = cols
        L = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lo, hi = np.percentile(L, [2, 98])
        L = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)[..., None]
        return (cd[None, None] * (1 - L) + cl[None, None] * L).astype(np.float32)
    L8 = cv2.transform(_u8(img), np.array([[0.2126, 0.7152, 0.0722]], np.float32))
    lo, hi = _percentiles8(L8, 0.02, 0.98)
    v = np.arange(256, dtype=np.float32)
    t = np.clip((v - lo) / max(hi - lo, 1e-3), 0, 1)
    stops = np.linspace(0, 1, len(cols))
    C = np.stack(cols)
    lut = np.stack([np.interp(t, stops, C[:, c]) for c in range(3)], 1).astype(np.float32).reshape(256, 1, 3)
    return cv2.LUT(cv2.cvtColor(L8, cv2.COLOR_GRAY2BGR), lut)


def tritone(img, colors=("#111111", "#ff4d2e", "#f2f2f2")) -> np.ndarray:
    """Three-stop gradient map (shadow, mid, highlight)."""
    return duotone(img, colors)


# ----------------------------------------------------------------------------- film grade

_GRADES = {
    #               saturation, S-curve, shadow tint (r,g,b),      highlight tint
    "teal_orange": (1.15, 0.38, (-0.1, 0.035, 0.1), (0.09, 0.015, -0.1)),
    "warm":        (1.05, 0.25, (0.025, 0.0, -0.035), (0.055, 0.02, -0.05)),
    "cool":        (0.92, 0.25, (-0.03, 0.0, 0.05), (-0.035, 0.005, 0.045)),
    "mono":        (0.0, 0.45, (0.0, 0.0, 0.012), (0.018, 0.008, -0.012)),
}


@lru_cache(maxsize=32)
def _grade_lut(preset: str, crushed_blacks: float, lift: float) -> np.ndarray:
    sat, sc, sh, hl = _GRADES[preset]
    x = np.arange(256, dtype=np.float64) / 255.0
    y = x + sc * (x * x * (3 - 2 * x) - x)                         # filmic S
    cb = 0.14 * crushed_blacks
    y = np.clip((y - cb) / (1.0 - cb), 0, 1)                       # crush
    y = y ** (1.0 + 0.6 * crushed_blacks)                          # deepen the toe
    out = np.empty((256, 3), np.float64)
    for c in range(3):
        out[:, c] = y + sh[c] * (1 - y) ** 2 * (0.25 + y) * 2.0 + hl[c] * y * y
    out = lift + out * (1.0 - lift)                                # lifted (milky) blacks
    return np.clip(out, 0, 1).astype(np.float32).reshape(256, 1, 3)


def film_grade(img, preset: str = "teal_orange", halation: float = 0.35, crushed_blacks: float = 0.15,
               lift: float = 0.03, saturation: float = 1.0) -> np.ndarray:
    """Film-style grade: saturation matrix, per-channel curves (S-curve, split tone by preset,
    crushed / lifted blacks) and halation (red-orange bloom around highlights)."""
    preset = preset if preset in _GRADES else "teal_orange"
    H, W = img.shape[:2]
    u8 = _u8(img)
    sat = _GRADES[preset][0] * float(saturation)
    if abs(sat - 1.0) > 1e-3:
        w = np.array([0.2126, 0.7152, 0.0722], np.float32)
        M = (1.0 - sat) * np.tile(w, (3, 1)) + sat * np.eye(3, dtype=np.float32)
        g8 = cv2.transform(u8, M.astype(np.float32))
    else:
        g8 = u8
    hal = float(np.clip(halation, 0.0, 2.0))
    if hal > 0.01:
        S = min(H, W)
        f = 8 if S >= 200 else 4
        sw, sh = max(2, W // f), max(2, H // f)
        small = cv2.resize(_gray8(u8), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m = np.clip((small - 0.6) / 0.4, 0, 1) ** 1.5
        wide = cv2.GaussianBlur(m, (0, 0), 0.03 * S / f)
        tight = cv2.GaussianBlur(m, (0, 0), 0.009 * S / f)
        c_wide, c_tight = ((np.array([1.0, 0.22, 0.07], np.float32), np.array([1.0, 0.5, 0.2], np.float32))
                           if preset != "mono" else (np.full(3, 0.7, np.float32), np.full(3, 0.7, np.float32)))
        glow = hal * (wide[..., None] * c_wide * 2.2 + tight[..., None] * c_tight * 0.6)
        g_small = np.clip(glow * 255.0 + 0.5, 0, 255).astype(np.uint8)
        g8 = cv2.add(g8, cv2.resize(g_small, (W, H), interpolation=cv2.INTER_LINEAR))
    lut = _grade_lut(preset, round(float(np.clip(crushed_blacks, 0, 1)), 3), round(float(np.clip(lift, 0, 0.5)), 3))
    return cv2.LUT(g8, lut)


# ----------------------------------------------------------------------------- flat posterise

@lru_cache(maxsize=16)
def _nearest_lut(pal_key: tuple) -> np.ndarray:
    """32^3 RGB grid -> index of the nearest palette colour (CIE Lab distance)."""
    pal = np.array(pal_key, np.float32).reshape(-1, 3)
    g = (np.arange(32, dtype=np.float32) * 8 + 4) / 255.0
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([r, gg, b], -1).reshape(-1, 1, 3)
    lab = cv2.cvtColor(grid, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    plab = cv2.cvtColor(pal.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    d = ((lab[:, None, :] - plab[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, 1).astype(np.uint8)


_Q_R = (np.arange(256, dtype=np.uint16) >> 3) << 10
_Q_G = (np.arange(256, dtype=np.uint16) >> 3) << 5
_Q_B = np.arange(256, dtype=np.uint16) >> 3


def posterize_flat(img, palette=("#111111", "#f2f2f2", "#ff4d2e"), smooth: float = 0.5,
                   scale: float = 1.0) -> np.ndarray:
    """Flat-illustration look: edge-preserving smoothing (bilateral + median at reduced size),
    then every pixel snapped to the nearest palette colour (Lab), edges anti-aliased."""
    H, W = img.shape[:2]
    cols = [_rgb(c, (0.5, 0.5, 0.5)) for c in (palette or [])][:16]
    if len(cols) < 2:
        cols = [_rgb("#111111"), _rgb("#f2f2f2")]
    smooth = float(np.clip(smooth, 0.0, 1.0))
    u8 = _u8(img)
    want = 3.0 * float(scale) * (0.5 + smooth)               # smoothing reach in frame px / 5-tap kernel
    f = max(1.5, want)
    sw, sh = max(8, int(round(W / f))), max(8, int(round(H / f)))
    small = cv2.resize(u8, (sw, sh), interpolation=cv2.INTER_AREA)
    passes = 1 + int(smooth > 0.45 and want >= 1.5)
    for _ in range(passes):
        small = cv2.bilateralFilter(small, 5, 30.0 + 60.0 * smooth, 5)
    mk = 3 if (smooth < 0.3 or want < 1.5) else (5 if smooth < 0.75 else 7)
    small = cv2.medianBlur(small, mk)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)              # smooth region boundaries before upsampling
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    r, g, b = cv2.split(up)
    idx = cv2.add(cv2.add(cv2.LUT(r, _Q_R), cv2.LUT(g, _Q_G)), cv2.LUT(b, _Q_B))
    lab = np.take(_nearest_lut(tuple(float(v) for c in cols for v in c)), idx)
    pal_lut = np.zeros((256, 1, 3), np.uint8)
    pal_lut[:len(cols), 0, :] = np.clip(np.stack(cols) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out8 = cv2.LUT(cv2.cvtColor(lab, cv2.COLOR_GRAY2BGR), pal_lut)
    if scale >= 0.6:
        out8 = cv2.GaussianBlur(out8, (3, 3), 0.6)            # AA the region edges (flat interiors unchanged)
    return _f32(out8)


# ----------------------------------------------------------------------------- grain / vignette

@lru_cache(maxsize=8)
def _grain_bank(H: int, W: int, seed: int, amount: float):
    """Same fields as the compositor's grain bank (ctx.rng(4242, k))."""
    S = min(H, W)
    gh, gw = (H // 2, W // 2) if S > 540 else (H, W)
    amp = np.float32(amount * 0.07 * (1.4 if S > 540 else 1.0))
    return [np.random.default_rng([int(seed) & 0xFFFFFFFF, 4242, k]).standard_normal((gh, gw), dtype=np.float32) * amp
            for k in range(12)]


def grain(img, amount: float = 0.15, seed: int = 0, frame: int = 0) -> np.ndarray:
    """Film grain, identical to the compositor's global grain for the same seed and frame."""
    amount = float(amount)
    if amount <= 0:
        return np.array(img, np.float32, copy=True)
    H, W = img.shape[:2]
    bank = _grain_bank(H, W, int(seed), round(amount, 4))
    r = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(frame) & 0xFFFFFFFF, 4242])
    n = bank[int(r.integers(len(bank)))]
    fl = int(r.integers(4))
    if fl < 3:
        n = cv2.flip(n, fl - 1)
    if n.shape[0] != H:
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(np.asarray(img, np.float32), cv2.cvtColor(n, cv2.COLOR_GRAY2BGR))
    return np.clip(out, 0.0, 1.0, out=out)


@lru_cache(maxsize=8)
def _vig(W: int, H: int, amount: float):
    from .compositor import _vignette      # lazy: compositor imports this module's users
    return _vignette(W, H, amount)


def vignette(img, amount: float = 0.25) -> np.ndarray:
    """Darken the corners (same falloff as the compositor's global vignette)."""
    H, W = img.shape[:2]
    v = _vig(W, H, round(float(amount), 4))
    if v is None:
        return np.array(img, np.float32, copy=True)
    return cv2.multiply(np.asarray(img, np.float32), v)


# ----------------------------------------------------------------------------- light leak

_LEAK_COLS = np.array([[1.0, 0.42, 0.08], [1.0, 0.16, 0.06], [1.0, 0.72, 0.3], [0.95, 0.25, 0.35]], np.float32)


def light_leak(img, t: float, seed: int = 0, strength: float = 0.5, speed: float = 1.0) -> np.ndarray:
    """Warm light leaking in from the frame edges: soft blobs that drift and breathe over
    time ``t`` (seconds), screen-blended so highlights burn out first."""
    H, W = img.shape[:2]
    strength = float(strength)
    u8 = _u8(img)
    if strength <= 0.005:
        return _f32(u8)
    f = 16 if min(H, W) >= 256 else 8
    sw, sh = max(8, W // f), max(8, H // f)
    asp = W / H
    xs = (np.arange(sw, dtype=np.float32) + 0.5) / sw * asp
    ys = (np.arange(sh, dtype=np.float32) + 0.5) / sh
    rng = _seed_rng(seed, 707)
    field = np.zeros((sh, sw, 3), np.float32)
    tt = float(t) * float(speed)
    for k in range(3):
        edge = int(rng.integers(4))
        along = float(rng.uniform(0.15, 0.85))
        ph, w1, w2 = rng.uniform(0, 6.283, 3), float(rng.uniform(0.25, 0.6)), float(rng.uniform(0.4, 1.1))
        a = along + 0.22 * math.sin(tt * w1 + ph[0])
        out = -0.05 + 0.12 * math.sin(tt * w2 * 0.7 + ph[1])
        if edge == 0:
            cx, cy = a * asp, out
        elif edge == 1:
            cx, cy = a * asp, 1.0 - out
        elif edge == 2:
            cx, cy = out * asp, a
        else:
            cx, cy = asp - out * asp, a
        rad = float(rng.uniform(0.22, 0.45)) * (1.0 + 0.2 * math.sin(tt * w2 + ph[2]))
        inten = max(0.0, 0.55 + 0.6 * math.sin(tt * w2 * 1.3 + ph[1] * 2.0))
        d2 = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2
        gk = np.exp(-d2 / (2.0 * rad * rad)).astype(np.float32)
        field += gk[..., None] * _LEAK_COLS[int(rng.integers(len(_LEAK_COLS)))] * np.float32(inten)
    leak8 = np.clip(field * strength * 255.0 + 0.5, 0, 255).astype(np.uint8)
    leak8 = cv2.resize(leak8, (W, H), interpolation=cv2.INTER_CUBIC)
    # screen: out = img + leak * (1 - img)
    out8 = cv2.add(u8, cv2.multiply(leak8, cv2.bitwise_not(u8), scale=1.0 / 255.0))
    return _f32(out8)


# ----------------------------------------------------------------------------- dust and scratches

def dust_scratches(img, frame: int, seed: int = 0, amount: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """Per-frame dust specks and hairs (mostly dark, some bright) and vertical scratches that
    persist for a few frames with jitter. Touches only small regions, so it is cheap."""
    out = np.array(img, np.float32, copy=True)
    amount = float(amount)
    if amount <= 0:
        return out
    H, W = out.shape[:2]
    s = max(float(scale), 0.25)
    rng = _seed_rng(seed, 909, frame)
    for _ in range(int(rng.poisson(amount * 7.0))):
        kind = rng.random()
        dark = rng.random() < 0.75
        op = float(rng.uniform(0.45, 0.9))
        if kind < 0.7:     # speck
            r = float(rng.uniform(0.8, 3.5)) * s
            pts = None
            half = int(math.ceil(r)) + 3
        else:              # hair: short random-walk curve
            n = int(rng.integers(6, 14))
            stepl = float(rng.uniform(2.5, 6.0)) * s
            ang = np.cumsum(rng.normal(0, 0.45, n)) + rng.uniform(0, 6.283)
            pts = np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * stepl, 0)
            pts -= pts.mean(0)
            half = int(np.abs(pts).max()) + 4
            r = 0.0
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        x0, y0, x1, y1 = max(0, cx - half), max(0, cy - half), min(W, cx + half + 1), min(H, cy + half + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
        if pts is None:
            cv2.ellipse(patch, ((half, half), (2 * r, 2 * r * float(rng.uniform(0.5, 1.0))),
                                float(rng.uniform(0, 180))), 1.0, -1, cv2.LINE_AA)
        else:
            q = np.round((pts + half) * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(patch, [q], False, 1.0, max(1, int(round(0.9 * s))), cv2.LINE_AA, 4)
        m = patch[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half), None] * np.float32(op)
        roi = out[y0:y1, x0:x1]
        if dark:
            roi *= 1.0 - m
        else:
            roi += (1.0 - roi) * m
    # scratches: re-rolled every 6 frames, jitter each frame
    hold = 6
    srng = _seed_rng(seed, 910, int(frame) // hold)
    for j in range(int(srng.poisson(amount * 1.4))):
        x = float(srng.uniform(0.04, 0.96)) * W + float(rng.normal(0, 0.9 * s))
        w = max(1.0, float(srng.uniform(0.6, 1.6)) * s)
        ya, yb = sorted(srng.uniform(-0.3, 1.3, 2))
        ya, yb = int(max(0, ya * H)), int(min(H, yb * H))
        if yb - ya < 4:
            continue
        op = float(srng.uniform(0.25, 0.6)) * (0.6 + 0.4 * float(rng.random()))
        bright = srng.random() < 0.6
        xi0, xi1 = max(0, int(x - w - 2)), min(W, int(x + w + 3))
        if xi1 <= xi0:
            continue
        xx = np.arange(xi0, xi1, dtype=np.float32) + 0.5
        prof = np.clip(w / 2.0 + 0.5 - np.abs(xx - x), 0, 1)[None, :, None] * np.float32(op)
        roi = out[ya:yb, xi0:xi1]
        if bright:
            roi += (1.0 - roi) * prof
        else:
            roi *= 1.0 - prof
    return out


# ----------------------------------------------------------------------------- dispatcher

def _resolve_refs(v, palette):
    """Replace "palette:N" strings (also inside lists) with the palette entry."""
    if isinstance(v, str) and v.strip().startswith("palette:"):
        pal = list(palette or [])
        if not pal:
            return v
        try:
            i = int(v.split(":", 1)[1])
        except ValueError:
            i = 0
        c = pal[i % len(pal)]
        return c.tolist() if isinstance(c, np.ndarray) else c
    if isinstance(v, (list, tuple)):
        return [_resolve_refs(x, palette) for x in v]
    return v


def _pal(palette, i, fallback):
    pal = list(palette or [])
    if not pal:
        return fallback
    c = pal[i % len(pal)] if i < len(pal) else fallback
    return c.tolist() if isinstance(c, np.ndarray) else c


_FUNCS = {
    "halftone": halftone, "screenprint": screenprint, "xerox": xerox, "duotone": duotone, "tritone": tritone,
    "film_grade": film_grade, "posterize_flat": posterize_flat, "paper": paper_overlay, "grain": grain,
    "vignette": vignette, "light_leak": light_leak, "dust_scratches": dust_scratches,
}
_SIGS = {k: set(inspect.signature(f).parameters) for k, f in _FUNCS.items()}
_NUMERIC = {"cell", "angle", "contrast", "texture", "misregister", "grit", "dither_px", "boil", "threshold",
            "toner_grit", "crushed_blacks", "halation", "lift", "saturation", "smooth", "fibre", "speckle",
            "amount", "strength", "speed"}


def apply_treatment(img: np.ndarray, spec, palette=None, frame: int = 0, seed: int = 0,
                    fps: float = 30.0, scale: float | None = None) -> np.ndarray:
    """Apply one treatment spec (or a list of them, in order) to an HxWx3 float32 image.

    spec: {"type": "screenprint", "inks": ["palette:0", "palette:2", "#e8dcc3"], "misregister": 2, ...}
    palette: plan.style.palette (hex strings or rgb arrays) for "palette:N" refs and defaults.
    frame/fps drive time-varying looks (grain, light_leak, dust_scratches, boil). scale defaults
    to (short side / 1080). Unknown types and keys are ignored; returns a new float32 array."""
    if not spec:
        return img
    if isinstance(spec, (list, tuple)):
        for k, s in enumerate(spec):
            img = apply_treatment(img, s, palette, frame, int(seed) + 7919 * k, fps, scale)
        return img
    if not isinstance(spec, dict):
        return img
    typ = spec.get("type")
    fn = _FUNCS.get(typ)
    if fn is None:
        return img
    img = np.ascontiguousarray(np.asarray(img)[..., :3], dtype=np.float32)
    H, W = img.shape[:2]
    if scale is None:
        scale = min(H, W) / REF_SHORT
    kw = {}
    for k, v in spec.items():
        if k == "type" or k not in _SIGS[typ] or k in ("img", "seed", "frame", "t", "scale", "size"):
            continue
        v = _resolve_refs(v, palette)
        if k in _NUMERIC:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        kw[k] = v
    # palette-aware defaults
    if typ == "halftone":
        kw.setdefault("ink", _pal(palette, 0, TONER))
    elif typ == "screenprint":
        inks = kw.get("inks")
        if not isinstance(inks, (list, tuple)) or not inks:
            kw["inks"] = [_pal(palette, 0, "#111111"), _pal(palette, 2, "#ff4d2e")]
    elif typ in ("duotone", "tritone"):
        cols = kw.get("colors")
        if not isinstance(cols, (list, tuple)) or len(cols) < 2:
            kw["colors"] = ([_pal(palette, 0, "#111111"), _pal(palette, 1, "#f2f2f2")] if typ == "duotone" else
                            [_pal(palette, 0, "#111111"), _pal(palette, 2, "#ff4d2e"), _pal(palette, 1, "#f2f2f2")])
    elif typ == "posterize_flat":
        cols = kw.get("palette")
        if not isinstance(cols, (list, tuple)) or len(cols) < 2:
            kw["palette"] = list(palette) if palette is not None and len(palette) >= 2 else None
            if kw["palette"] is None:
                kw.pop("palette")
    sig = _SIGS[typ]
    # grain keeps the raw seed so it matches the compositor's global grain exactly
    sd = int(seed) if typ == "grain" else (int(seed) * 1000003 + zlib.crc32(str(typ).encode())) & 0x7FFFFFFF
    if "seed" in sig:
        kw["seed"] = sd
    if "frame" in sig:
        kw["frame"] = int(frame)
    if "t" in sig:
        kw["t"] = float(frame) / max(float(fps), 1e-6)
    if "scale" in sig:
        kw["scale"] = float(scale)
    out = fn(img, **kw)
    return np.ascontiguousarray(out, np.float32)


__all__ = ["TREATMENTS", "FILM_PRESETS", "apply_treatment", "halftone", "screenprint", "xerox", "duotone",
           "tritone", "film_grade", "posterize_flat", "paper", "paper_overlay", "grain", "vignette",
           "light_leak", "dust_scratches"]
