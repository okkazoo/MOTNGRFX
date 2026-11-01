"""Compositing library: weaves kinetic type into generated video (behind / in front / seen through / traced).

Conventions
- Frames are float32 arrays in [0,1], layout HxWx3 (RGB). Mattes / alphas / depth are HxW (HxWx1 accepted).
- Sizes are (width, height) everywhere, e.g. (1088, 1920) for 9:16 finals.
- Depth: near = white (1.0), far = black.
- Every public op is a pure function: numpy in, numpy out. The maths runs on torch/CUDA when available, otherwise on
  numpy/cv2; the API is identical. Force one with env FUSION_BACKEND=numpy|torch or set_backend().
- Ops never mutate their inputs. Mismatched mask sizes are bilinearly resized to the footage size.

CLI:  python tools/fusion/compose.py shot recipe.json OUT_DIR [--audio a.wav]
      python tools/fusion/compose.py assemble spec.json
      python tools/fusion/compose.py info | selfcheck
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:  # torch is optional
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

# ======================================================================================================== backends


class _NumpyBackend:
    name = "numpy"
    device = "cpu"

    # -- conversion
    def asarray(self, x):
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.ascontiguousarray(x, dtype=np.float32)

    def tonumpy(self, x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # -- elementwise
    def where(self, c, a, b):
        return np.where(c, a, b).astype(np.float32)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def sqrt(self, x):
        return np.sqrt(x)

    def exp(self, x):
        return np.exp(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def stack(self, xs, axis=-1):
        return np.stack(xs, axis=axis)

    def mean(self, x, axis):
        return x.mean(axis=axis)

    def amax(self, x):
        return float(x.max())

    def percentile(self, x, qs):
        return [float(v) for v in np.percentile(x[::4, ::4], qs)]

    # -- filters
    def blur(self, x, sigma):
        if sigma <= 0.05:
            return x
        if sigma > 6.0:  # big sigmas: blur at reduced res and resample back (same trick as the torch backend)
            h, w = x.shape[:2]
            f = int(sigma // 3)
            small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), float(sigma) / f, borderType=cv2.BORDER_REFLECT_101)
            return _keep_shape(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), x)
        return _keep_shape(cv2.GaussianBlur(x, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT_101), x)

    def box(self, x, r):
        r = int(r)
        if r <= 0:
            return x
        return _keep_shape(cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT_101), x)

    def resize(self, x, h, w, interp="linear"):
        if x.shape[0] == h and x.shape[1] == w:
            return x
        m = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}[interp]
        return _keep_shape(cv2.resize(x, (int(w), int(h)), interpolation=m), x)

    def _dilate1(self, x, cross):
        k = _CROSS if cross else _SQUARE
        return _keep_shape(cv2.dilate(x, k), x)

    def sobel(self, g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0, borderType=cv2.BORDER_REFLECT_101)
        return gx, gy

    def warp(self, img, dx, dy):
        """img sampled at (x+dx, y+dy), bilinear, reflected borders."""
        h, w = img.shape[:2]
        xs, ys = _grid_xy(h, w, self)
        return _keep_shape(cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), img)


def _keep_shape(y, like):
    if like.ndim == 3 and y.ndim == 2:
        y = y[..., None]
    return y


_SQUARE = np.ones((3, 3), np.uint8)
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


class _TorchBackend(_NumpyBackend):
    name = "torch"

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def asarray(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, torch.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

    def tonumpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def where(self, c, a, b):
        return torch.where(c, a, b)

    def maximum(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=self.device)
        return torch.maximum(a, b)

    def minimum(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=self.device)
        return torch.minimum(a, b)

    def sqrt(self, x):
        return torch.sqrt(x)

    def exp(self, x):
        return torch.exp(x)

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def stack(self, xs, axis=-1):
        return torch.stack(xs, dim=axis)

    def mean(self, x, axis):
        return x.mean(dim=axis)

    def amax(self, x):
        return float(x.max())

    def percentile(self, x, qs):
        s = x[::8, ::8].reshape(-1)
        return [float(torch.quantile(s, q / 100.0)) for q in qs]

    # NCHW helpers
    @staticmethod
    def _to4(x):
        return (x[None, None], True) if x.ndim == 2 else (x.permute(2, 0, 1)[None], False)

    @staticmethod
    def _from4(y, was2d):
        y = y[0]
        return y[0] if was2d else y.permute(1, 2, 0).contiguous()

    def _conv_sep(self, x4, kx, ky, pad_mode="reflect"):
        c = x4.shape[1]
        rx, ry = len(kx) // 2, len(ky) // 2
        kh = torch.tensor(kx, dtype=torch.float32, device=x4.device).view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        kv = torch.tensor(ky, dtype=torch.float32, device=x4.device).view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        y = F.conv2d(F.pad(x4, (rx, rx, 0, 0), mode=pad_mode), kh, groups=c)
        return F.conv2d(F.pad(y, (0, 0, ry, ry), mode=pad_mode), kv, groups=c)

    def blur(self, x, sigma):
        if sigma <= 0.05:
            return x
        h, w = x.shape[:2]
        x4, was2d = self._to4(x)
        if sigma > 6.0:  # blur at reduced res (cheap on big sigmas), resample back
            f = int(sigma // 3)
            hh, ww = max(1, h // f), max(1, w // f)
            small = F.interpolate(x4, size=(hh, ww), mode="area")
            y = self._gauss4(small, sigma / f)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        else:
            y = self._gauss4(x4, sigma)
        return self._from4(y, was2d)

    def _gauss4(self, x4, sigma):
        r = int(math.ceil(3 * sigma))
        r = max(1, min(r, min(x4.shape[2], x4.shape[3]) - 1))
        t = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (t / sigma) ** 2)
        k = (k / k.sum()).tolist()
        return self._conv_sep(x4, k, k)

    def box(self, x, r):
        r = int(r)
        if r <= 0:
            return x
        x4, was2d = self._to4(x)
        r = min(r, min(x4.shape[2], x4.shape[3]) - 1)
        y = F.pad(x4, (r, r, r, r), mode="reflect")
        y = F.avg_pool2d(y, 2 * r + 1, stride=1)
        return self._from4(y, was2d)

    def resize(self, x, h, w, interp="linear"):
        if x.shape[0] == h and x.shape[1] == w:
            return x
        x4, was2d = self._to4(x)
        if interp == "area":
            y = F.interpolate(x4, size=(int(h), int(w)), mode="area")
        elif interp == "cubic":
            y = F.interpolate(x4, size=(int(h), int(w)), mode="bicubic", align_corners=False)
        else:
            y = F.interpolate(x4, size=(int(h), int(w)), mode="bilinear", align_corners=False)
        return self._from4(y, was2d)

    def _dilate1(self, x, cross):
        x4, was2d = self._to4(x)
        a = F.max_pool2d(x4, (1, 3), 1, (0, 1))
        b = F.max_pool2d(x4, (3, 1), 1, (1, 0))
        y = torch.maximum(a, b) if cross else F.max_pool2d(a, (3, 1), 1, (1, 0))
        return self._from4(y, was2d)

    def sobel(self, g):
        g4, _ = self._to4(g)
        p = F.pad(g4, (1, 1, 1, 1), mode="reflect")
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=g.device) / 8.0
        ky = kx.t().contiguous()
        gx = F.conv2d(p, kx.view(1, 1, 3, 3))[0, 0]
        gy = F.conv2d(p, ky.view(1, 1, 3, 3))[0, 0]
        return gx, gy

    def warp(self, img, dx, dy):
        h, w = img.shape[:2]
        xs, ys = _grid_xy(h, w, self)
        gx = 2.0 * (xs + dx) / max(w - 1, 1) - 1.0
        gy = 2.0 * (ys + dy) / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]
        x4, was2d = self._to4(img)
        y = F.grid_sample(x4, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        return self._from4(y, was2d)


_GRID = {}


def _grid_xy(h, w, be):
    key = (be.name, str(be.device), h, w)
    g = _GRID.get(key)
    if g is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (be.asarray(xs), be.asarray(ys))
        if len(_GRID) > 16:
            _GRID.clear()
        _GRID[key] = g
    return g


def _make_backend(name=None):
    name = (name or os.environ.get("FUSION_BACKEND", "auto")).lower()
    if name == "numpy" or torch is None:
        return _NumpyBackend()
    if name == "torch":
        return _TorchBackend()
    if torch.cuda.is_available():  # auto: torch only pays off on CUDA; cv2 beats torch-CPU
        return _TorchBackend("cuda")
    return _NumpyBackend()


B = _make_backend()


def set_backend(name):
    """'numpy' | 'torch' | 'auto' - rebinds the global backend; returns its name."""
    global B
    B = _make_backend(name)
    return B.name


def backend_name():
    return f"{B.name}:{B.device}"


def _is_arr(x):
    return isinstance(x, np.ndarray) or (torch is not None and isinstance(x, torch.Tensor))


def _in(x):
    return B.asarray(x) if _is_arr(x) else x


def _out(x):
    if _is_arr(x):
        return B.tonumpy(x)
    if isinstance(x, tuple):
        return tuple(_out(v) for v in x)
    return x


def api(fn):
    """Public op: numpy args -> backend arrays, result back to numpy. `fn.raw` is the backend-array version."""
    @functools.wraps(fn)
    def w(*a, **k):
        a = [_in(v) for v in a]
        k = {n: _in(v) for n, v in k.items()}
        return _out(fn(*a, **k))
    w.raw = fn
    return w


# ======================================================================================================== helpers

def _c3(x):
    """HxW -> HxWx1"""
    return x[..., None]


def _gray(x):
    """any mask-like input -> HxW (RGBA: alpha, RGB: luma, HxWx1: squeeze)."""
    if x.ndim == 2:
        return x
    c = x.shape[-1]
    if c == 1:
        return x[..., 0]
    if c == 4:
        return x[..., 3]
    return _luma(x)


def _luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def _fit(x, h, w, interp="linear"):
    if x.shape[0] == h and x.shape[1] == w:
        return x
    return B.resize(x, h, w, interp)


def _mask(x, h, w):
    return _fit(_gray(x), h, w)


def _color(c, like):
    """colour tuple/array -> HxWx3 image like `like`. Arrays are fitted."""
    if _is_arr(c):
        c = _in(c)
        if c.ndim == 1:
            return _color(tuple(float(v) for v in B.tonumpy(c)), like)
        if c.ndim == 2:
            c = B.stack([c, c, c])
        return _fit(c, like.shape[0], like.shape[1])
    v = np.asarray(c, np.float32)
    if v.ndim == 0:
        v = np.repeat(v, 3)
    base = B.asarray(np.broadcast_to(v[:3], (like.shape[0], like.shape[1], 3)))
    return base


def _vec3(c, like):
    """colour tuple -> broadcastable (1,1,3) backend array"""
    v = np.asarray(c, np.float32).reshape(1, 1, -1)
    if v.shape[-1] == 1:
        v = np.repeat(v, 3, axis=-1)
    return B.asarray(v[..., :3])


def _clip(x, lo=0.0, hi=1.0):
    return x.clip(lo, hi)


def _lerp(a, b, t):
    return a + (b - a) * t


def _smoothstep(x, e0, e1):
    t = _clip((x - e0) / max(e1 - e0, 1e-8))
    return t * t * (3.0 - 2.0 * t)


def _sat(x, s):
    g = _c3(_luma(x))
    return g + (x - g) * s


def _dilate(x, r):
    """Grey dilation by ~r px (octagon: alternating 3x3 square / cross steps). Works on HxW or HxWxC."""
    for i in range(int(round(r))):
        x = B._dilate1(x, cross=bool(i % 2))
    return x


def _erode(x, r):
    return 1.0 - _dilate(1.0 - x, r)


def luma(x):
    """Rec.709 luma of an RGB frame -> HxW."""
    return _out(_luma(_in(x)))


# ======================================================================================================== blend helpers

@api
def screen(a, b, opacity=1.0):
    r = 1.0 - (1.0 - a) * (1.0 - b)
    return _lerp(a, r, opacity) if opacity != 1.0 else r


@api
def multiply(a, b, opacity=1.0):
    r = a * b
    return _lerp(a, r, opacity) if opacity != 1.0 else r


@api
def overlay(a, b, opacity=1.0):
    lo = 2.0 * a * b
    hi = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
    r = B.where(a < 0.5, lo, hi)
    return _lerp(a, r, opacity) if opacity != 1.0 else r


@api
def soft_light(a, b, opacity=1.0):
    """W3C soft-light, base a, blend b."""
    d = B.where(a <= 0.25, ((16.0 * a - 12.0) * a + 4.0) * a, B.sqrt(a))
    lo = a - (1.0 - 2.0 * b) * a * (1.0 - a)
    hi = a + (2.0 * b - 1.0) * (d - a)
    r = B.where(b <= 0.5, lo, hi)
    return _lerp(a, r, opacity) if opacity != 1.0 else r


@api
def over(base, fg, alpha):
    """Standard 'over': fg (HxWx3 or colour tuple) onto base by alpha (HxW)."""
    h, w = base.shape[:2]
    return _lerp(base, _color(fg, base), _c3(_clip(_mask(alpha, h, w))))


@api
def add_glow(img, mask, radius=24.0, intensity=0.8, color=None, passes=3):
    """Bloom from a mask: multi-radius blurred mask (radius, 2r, 4r ...), tinted, screened over img."""
    h, w = img.shape[:2]
    m = _clip(_mask(mask, h, w))
    acc = None
    wsum = 0.0
    for i in range(int(passes)):
        wi = 0.5 ** i
        g = B.blur(m, radius * (2 ** i)) * wi
        acc = g if acc is None else acc + g
        wsum += wi
    g = _clip(acc / wsum * intensity * 2.0)
    if color is not None and getattr(color, "ndim", 0) >= 2:  # per-pixel tint image (e.g. "$tcol")
        col = color[..., :3] if color.ndim == 3 and color.shape[-1] >= 3 else _c3(color)
    else:
        col = _vec3(color if color is not None else (1.0, 1.0, 1.0), img)
    return 1.0 - (1.0 - img) * (1.0 - _c3(g) * col)


@api
def luminosity_key(img, low=0.3, high=0.7, invert=False, gamma=1.0):
    """Alpha from luma: smoothstep(low, high). invert=True keys the darks."""
    k = _smoothstep(_luma(img), low, high)
    if invert:
        k = 1.0 - k
    return k if gamma == 1.0 else k ** gamma


# ======================================================================================================== finishing

@api
def grade(frame, lift=0.0, gamma=1.0, gain=1.0, sat=1.0):
    """gain -> lift (raises blacks toward `lift`) -> gamma (>1 brightens) -> saturation. lift/gain may be 3-tuples."""
    lift_v = _vec3(lift, frame) if isinstance(lift, (tuple, list)) else lift
    gain_v = _vec3(gain, frame) if isinstance(gain, (tuple, list)) else gain
    x = _clip(frame * gain_v)
    x = lift_v + (1.0 - lift_v) * x
    x = _clip(x) ** (1.0 / max(gamma, 1e-4))
    if sat != 1.0:
        x = _clip(_sat(x, sat))
    return x


@api
def halation(frame, threshold=0.7, radius=22.0, strength=0.35, tint=(1.0, 0.45, 0.25)):
    """Red-orange glow bleeding from bright areas (film halation)."""
    hl = _c3(_clip((_luma(frame) - threshold) / max(1.0 - threshold, 1e-4)))
    src = frame * hl
    glow = B.blur(src, radius) + B.blur(src, radius * 2.5) * 0.5
    g = glow * _vec3(tint, frame) * strength
    return 1.0 - (1.0 - frame) * (1.0 - _clip(g))


def _grain_noise(h, w, seed, idx, size):
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(idx) & 0xFFFFFFFF])
    if size <= 1.0:
        return rng.standard_normal((h, w), dtype=np.float32)
    hh, ww = max(2, int(round(h / size))), max(2, int(round(w / size)))
    n = rng.standard_normal((hh, ww), dtype=np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    return n / max(float(n.std()), 1e-6)


@api
def grain(frame, frame_idx=0, seed=0, amount=0.03, size=1.0, midtone=0.7):
    """Deterministic film grain: a different field per frame_idx, identical across backends. amount ~ stddev."""
    h, w = frame.shape[:2]
    n = B.asarray(_grain_noise(h, w, seed, frame_idx, size))
    l = _luma(frame)
    wgt = (1.0 - midtone) + midtone * 4.0 * l * (1.0 - l)
    return _clip(frame + _c3(n * wgt) * amount)


@api
def vignette(frame, strength=0.4, radius=0.5, softness=0.6, center=(0.5, 0.5)):
    """Darken toward the corners. radius/softness in units of the half-diagonal."""
    h, w = frame.shape[:2]
    xs, ys = _grid_xy(h, w, B)
    cx, cy = center[0] * (w - 1), center[1] * (h - 1)
    d = B.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (0.5 * math.hypot(w, h))
    t = _smoothstep(d, radius, radius + softness)
    return frame * _c3(1.0 - strength * t)


# ======================================================================================================== core compositing ops

@api
def sandwich(footage, type_rgb, type_a, matte, matte_gamma=1.0, matte_gain=1.0, fg=None, invert=False,
             type_opacity=1.0, matte_threshold=None, matte_soft=0.08, front=None, legible=True):
    """bottom = footage; type over it by type_a; then (fg or footage) over that by clip(matte)**gamma * gain.
    Default (matte_threshold None): the matte is NOT thresholded, a grey 0.5 matte shows type at 50% (mist / glass / silk /
    water / hair stay see-through: the `behind_sheer` recipe).
    SOLID matte: matte_threshold=0.5, matte_soft=0.08 -> m = smoothstep(m, thr - soft/2, thr + soft/2) (after `invert`, before
    gamma / gain), so a subject hides the type completely and the edge is a soft 0.08-wide ramp (the `behind` recipe).
    invert=True flips the matte (1-matte) before the threshold / gamma / gain, so the region the matte marks as subject stays
    *under* the type and the rest (background / foreground layer) is drawn over it: type in front of the subject,
    behind the background.
    front: optional HxW region (0..1, e.g. a type pass or a drawn mask): where it is 1 the type is IN FRONT of the subject
    (the matte is ignored there), elsewhere it stays behind it: ONE type layer both in front of and behind the subject.
    Type in front of AND behind at once with two layers: chain two sandwich ops on the same matte, the second reading the first's
    output, e.g. out "image": sandwich(footage=$foot, type_a=$ta_back, matte=$m) -> sandwich(footage=$image, type_a=$ta_front,
    matte=$m, invert=true); the second op only shows its type over the subject and keeps the first op's type behind it.
    legible: not read by the op itself. recipe_eval.build_recipe inserts the `legible` guard (letters the matte would hide for the
    whole hold of a word are cleared) before every sandwich unless this is False."""
    h, w = footage.shape[:2]
    ta = _clip(_mask(type_a, h, w) * type_opacity)
    base = _lerp(footage, _color(type_rgb, footage), _c3(ta))
    m = _clip(_mask(matte, h, w))
    if invert:
        m = 1.0 - m
    if matte_threshold is not None:
        m = _smoothstep(m, float(matte_threshold) - 0.5 * float(matte_soft), float(matte_threshold) + 0.5 * float(matte_soft))
    m = _clip(m ** matte_gamma * matte_gain)
    if front is not None:
        m = m * (1.0 - _clip(_mask(front, h, w)))
    top = footage if fg is None else _fit(fg, h, w)
    return _lerp(base, top, _c3(m))


@api
def depth_cover(footage_depth, plane=0.5, softness=0.08, size=None):
    """HxW 0..1: how much the footage (depth map) covers type sitting at depth `plane` (the cover `depth_occlude` uses; `size` = (h, w) to fit the depth to)."""
    h, w = tuple(size) if size is not None else footage_depth.shape[:2]
    d = _mask(footage_depth, h, w)
    pl = _mask(plane, h, w) if _is_arr(plane) else plane
    return _clip((d - pl) / max(softness, 1e-6) + 0.5)


@api
def depth_occlude(footage, footage_depth, type_rgb, type_a, plane=0.5, softness=0.08, type_opacity=1.0, cover=None, legible=True, fg=None, subject_matte=None):
    """Type sits at depth `plane` (0..1, near=white; scalar or HxW map e.g. a type depth pass). Footage pixels nearer
    than the plane cover the type with a soft ramp of total width `softness` centred on the plane.
    cover: optional HxW 0..1 cover used instead of the one computed from the depth (`depth_cover`, possibly adjusted by `legible`).
    fg: optional HxWx3 pixels shown where the footage covers the type (three-layer shots: the clean clip while `footage` is the text-rendered
    background, 16_app/LAYERS.md); None = the footage itself (single clip, unchanged).
    subject_matte: optional HxW 0..1 matte of the SUBJECT (Alpha-Gen, cut from the clean clip): the cover is at least the matte AFTER the legibility guard, so the subject
    stays in front of the type even where the depth map calls it far or the guard cleared the cover (three-layer shots only; None = unchanged).
    legible: not read by the op; recipe_eval.build_recipe inserts depth_cover + the `legible` guard before it unless False."""
    h, w = footage.shape[:2]
    cover = depth_cover.raw(footage_depth, plane, softness, (h, w)) if cover is None else _clip(_mask(cover, h, w))
    if subject_matte is not None:
        cover = B.maximum(cover, _clip(_mask(subject_matte, h, w)))
    ta = _clip(_mask(type_a, h, w) * type_opacity) * (1.0 - cover)
    if fg is not None:
        footage = _lerp(footage, _fit(fg, h, w), _c3(cover))
    return _lerp(footage, _color(type_rgb, footage), _c3(ta))


def _small_np(x, side=180):
    """backend HxW / HxWxC -> small float32 numpy (long side <= `side`), for scalar measurements (contrast checks)."""
    h, w = x.shape[:2]
    f = min(1.0, side / float(max(h, w)))
    if f < 1.0:
        x = B.resize(x, max(2, int(round(h * f))), max(2, int(round(w * f))), "area")
    return np.asarray(B.tonumpy(x), np.float32)


def window_contrast(plate_rgb, type_a, footage, core=0.5, ring=0.035):
    """(Lin, Lout, n): mean luma of `footage` under the letters (type_a > core) and mean luma of `plate_rgb` in a ring around the word
    (a band `ring` x the long side wide, outside the letters). n = letter pixels measured (0 -> nothing to measure). Backend arrays in."""
    a = _small_np(type_a)
    f = _small_np(footage)
    p = _small_np(plate_rgb)
    c = (a > core).astype(np.uint8)
    n = int(c.sum())
    if n < 12:
        return 0.0, 0.0, 0
    r = max(2, int(round(ring * max(a.shape))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = (cv2.dilate(c, k) > 0) & (c == 0)
    lum = lambda x: x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722   # noqa: E731
    lin = float(lum(f)[c > 0].mean())
    lout = float(lum(p)[band].mean()) if band.any() else float(lum(p).mean())
    return lin, lout, n


@api
def knockout(plate_rgb, type_a, footage, fg_matte=None, fg=None, min_contrast=None, face=None):
    """Letters are windows into footage through a flat plate. With fg_matte the subject (footage or fg) steps out
    over the plate: out = plate; footage by type_a; (fg or footage) by fg_matte.
    min_contrast (legibility, default off): the window must read, so its mean luma under the letters has to exceed the surround's by
    at least this much (e.g. 0.22). When the surround is too bright for that, the plate is darkened (down to 0.05 x) until it does (P10-style
    ground); when the window itself is too dark to ever do (the letters' share of the footage < 0.4 x min_contrast) the letters become SOLID
    type instead (`face`: HxWx3 type colour, the beauty pass; default warm white), blended in as the window gets darker. Continuous in the
    measured values, so no pops between frames."""
    h, w = footage.shape[:2]
    ta = _clip(_mask(type_a, h, w))
    plate = _color(plate_rgb, footage)
    solid = 0.0
    if min_contrast is not None:
        mc = float(min_contrast)
        lin, lout, n = window_contrast(plate, ta, footage)
        if n:
            if lin - lout < mc:
                g = float(np.clip((lin - mc) / max(lout, 1e-3), 0.05, 1.0))
                plate = plate * g
            solid = 1.0 - float(np.clip((lin - 0.4 * mc) / max(0.6 * mc, 1e-6), 0.0, 1.0))
    out = _lerp(plate, footage, _c3(ta))
    if solid > 1e-3:
        fc = _color(face if face is not None else (1.0, 0.94, 0.82), footage)
        out = _lerp(out, fc, _c3(ta * solid))
    if fg_matte is not None:
        top = footage if fg is None else _fit(fg, h, w)
        out = _lerp(out, top, _c3(_clip(_mask(fg_matte, h, w))))
    return out


# ======================================================================================================== legibility guard
# A matte (or depth cover) must never hide a word's letters for the whole time the word is showing. Occlusion is fine while a
# word forms / exits and for things passing in front briefly; during the HOLD (hit -> release) every letter must read.
# Design: stateless per frame. The per-hold decision ("which letters are covered too much of the hold") is made ONCE by
# `legible_plan` from a pass over the hold's frames (recipe_eval.compose_frames runs it before the frame loop, so chunked /
# resumed renders all decide the same); the `legible` op then only applies the plan at time t: it clears the matte over the
# flagged letters' live shapes, weighted by a ramp (0 -> 1 over ramp_s ending at the hit, 1 -> 0 over ramp_s after the release),
# so the word reads as coming forward, not as a pop. Outside the holds (+ ramps) the matte is untouched.

LEG_THR = 0.25            # type alpha above this counts as letter body
LEG_MAXSIDE = 360.0       # the analysis runs at this short-side (px); the plan boxes are stored at full res
# HARD FLOOR (2026-11-01, QC of public renders: "WR NG", a second line hidden for 5 s): a word whose letter pixels the subject covers by more than
# `word` (share of the word's letter pixels) in more than `most` of its hold frames, or with ONE letter less than `letter` visible for `most` of the
# hold, comes in front of the subject as a WHOLE word. recipe_eval.guard_ops puts it on every subject-over-type draw (sandwich, also with
# legible=false, and the `over` pairs of the pure-type route); only compose.legible_floor = false turns it off.
# 2026-11-02 (user, frame 406 of an own-video job: "most of the text neither fully over or fully behind the subject - we need moments like these to read the text correctly"): the floor is the
# readable-behind rule, one set of numbers for every mode: a word stays BEHIND the subject only while, in all but 25 % of its hold's frames, every letter keeps >= 70 % of its area visible
# and the whole word >= 85 % (the subject hides <= 15 % of the letter pixels); else the whole word is IN FRONT. Numbers from the job (a glyph fully hidden, two thirds of
# the word's area): the old 25 % / 50 % floor let a word with a quarter hidden stay half behind; tools/app/overlay_mode.VIS_RULE is the same rule used at design time.
LEG_FLOOR = {"word": 0.15, "letter": 0.70, "most": 0.25}
# BEHIND QUOTA (user 2026-11-01, R16 review "none of the text is being masked", a rule for ALL videos): where a subject matte exists, about half of the words (at least one) that the subject
# overlaps stay BEHIND it with part of their letters covered. The strict guard above (letters < 80 % visible, floor) would bring nearly every such word in front; the words chosen here are judged by
# the relaxed floor instead (the subject hides <= 50 % of the letter pixels and no letter drops under 10 % visible, in most of the hold), nearest to SWEET_OCC of the word first. The others keep the strict rule.
LEG_BEHIND = {"share": 0.5, "min": 1, "word": 0.15, "letter": 0.70, "most": 0.25, "sweet": 0.10, "overlap": 0.02}      # 2026-11-02: the behind words are judged by the SAME readable rule as every word (LEG_FLOOR): no relaxed floor


def _leg_scale(h, w):
    return min(1.0, LEG_MAXSIDE / float(max(min(h, w), 1)))


def _leg_small(a, s):
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] == 1 else a.mean(-1)
    if s >= 1.0:
        return a
    return cv2.resize(a, (max(1, int(round(a.shape[1] * s))), max(1, int(round(a.shape[0] * s)))), interpolation=cv2.INTER_AREA)


def _leg_components(ta_s, min_rel=0.08):
    """connected components of the letter body -> (labels, [(label, x0, y0, x1, y1, area)]) without specks (i-dots, noise:
    area < min_rel * median area)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((ta_s > LEG_THR).astype(np.uint8), connectivity=8)
    comps = [(i, int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3]), int(st[i, 4])) for i in range(1, n)]
    if comps:
        med = float(np.median([c[5] for c in comps]))
        comps = [c for c in comps if c[5] >= min_rel * med and c[5] >= 4]
    return lab, comps


def effective_occluder(matte, matte_threshold=None, matte_soft=0.08, matte_gamma=1.0, invert=False):
    """the matte as `sandwich` turns it into an occluder (0 = type fully shown, 1 = fully hidden). numpy HxW."""
    m = np.clip(np.asarray(matte, np.float32), 0.0, 1.0)
    if invert:
        m = 1.0 - m
    if matte_threshold is not None:
        e0, e1 = float(matte_threshold) - 0.5 * float(matte_soft), float(matte_threshold) + 0.5 * float(matte_soft)
        t = np.clip((m - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
        m = t * t * (3.0 - 2.0 * t)
    return np.clip(m ** float(matte_gamma), 0.0, 1.0)


def legible_plan(type_at, matte_at, holds, fps=24, min_vis=0.8, min_frac=0.8, samples=48, dilate=6,
                 matte_threshold=None, matte_soft=0.08, matte_gamma=1.0, invert=False, frame0=0, frame1=None, front_at=None, floor=None, letters=True, behind=None):
    """Decide, per hold, which letters the matte covers too much. `type_at(f)` / `matte_at(f)` -> HxW float (f = piece frame
    index, absolute); holds = [(t0_s, t1_s)] piece seconds (hit, release); frames outside [frame0, frame1) do not exist.
    A letter = connected component of the type alpha at the hold's middle frame. At every sampled frame of the hold (every frame
    up to `samples`, else a stride) its visible fraction is sum(ta * (1 - occluder)) / sum(ta) inside its box; it is FLAGGED
    when that is below `min_vis` in more than (1 - min_frac) of the frames where the letter is on screen.
    -> {"size": [W, H], "holds": [{"a", "b", "boxes": [[x0,y0,x1,y1] full res], "hidden": [fraction per letter], "n": letters}]};
    boxes = flagged letters only (dilated by `dilate` px). `front_at(f)` (optional, like sandwich's `front`) = region where the type is
    already in front of the matte: it does not occlude there.
    floor: None or a dict like LEG_FLOOR (True = LEG_FLOOR): the word-level hard floor (see LEG_FLOOR): when it fires, EVERY letter of the hold is
    flagged (the whole word in front); letters=False skips the per-letter rule above (a sandwich that opted out of the guard keeps only the floor).
    Each hold also reports "occ" (mean share of letter pixels occluded over the sampled frames) and "floor" (True when the floor fired)."""
    if floor is True:
        floor = LEG_FLOOR
    if behind is True:
        behind = LEG_BEHIND
    bh = dict(LEG_BEHIND, **behind) if isinstance(behind, dict) else None
    out, size = [], None
    for a, b in holds:
        f0, f1 = int(round(float(a) * fps)), int(round(float(b) * fps))
        f0 = max(f0, int(frame0))
        if frame1 is not None:
            f1 = min(f1, int(frame1))
        if f1 <= f0:
            out.append(dict(a=float(a), b=float(b), boxes=[], hidden=[], n=0, occ=0.0, floor=False))
            continue
        step = max(1, int(math.ceil((f1 - f0) / float(samples))))
        mid = np.asarray(type_at((f0 + f1) // 2), np.float32)
        if mid.ndim == 3:
            mid = mid[..., 0]
        H, W = mid.shape[:2]
        size = [W, H]
        s = _leg_scale(H, W)
        lab, comps = _leg_components(_leg_small(mid, s))
        hid, seen = np.zeros(len(comps)), np.zeros(len(comps))
        lo_vis = np.zeros(len(comps))                # frames where the letter is under floor["letter"] visible
        occs, n_word = [], 0                         # word occlusion per sampled frame; frames over floor["word"]
        lo_b, n_word_b = np.zeros(len(comps)), 0     # the relaxed (behind) floor's counters
        for f in range(f0, f1, step):
            ta = _leg_small(type_at(f), s)
            o = effective_occluder(_leg_small(matte_at(f), s), matte_threshold, matte_soft, matte_gamma, invert)
            if o.shape != ta.shape:
                o = cv2.resize(o, (ta.shape[1], ta.shape[0]), interpolation=cv2.INTER_LINEAR)
            if front_at is not None:
                fr_ = _leg_small(front_at(f), s)
                if fr_.shape != o.shape:
                    fr_ = cv2.resize(fr_, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)
                o = o * (1.0 - np.clip(fr_, 0.0, 1.0))
            wt_, wo_ = 0.0, 0.0
            for k, (_, x0, y0, x1, y1, _a) in enumerate(comps):
                t = ta[y0:y1, x0:x1]
                tot = float(t.sum())
                if tot < 1e-3:                       # letter not on screen at this sample: not counted
                    continue
                seen[k] += 1
                vis = float((t * (1.0 - o[y0:y1, x0:x1])).sum()) / tot
                wt_, wo_ = wt_ + tot, wo_ + tot * (1.0 - vis)
                if vis < min_vis:
                    hid[k] += 1
                if floor and vis < float(floor.get("letter", LEG_FLOOR["letter"])):
                    lo_vis[k] += 1
                if bh and vis < bh["letter"]:
                    lo_b[k] += 1
            if wt_ > 1e-3:
                occs.append(wo_ / wt_)
                n_word += 1 if floor and occs[-1] > float(floor.get("word", LEG_FLOOR["word"])) else 0
                n_word_b += 1 if bh and occs[-1] > bh["word"] else 0
        frac = np.where(seen > 0, hid / np.maximum(seen, 1), 0.0)
        most = float((floor or {}).get("most", LEG_FLOOR["most"]))
        fired = bool(floor) and bool(occs) and (n_word > most * len(occs) or bool(np.any((seen > 0) & (lo_vis > most * seen))))
        d = int(round(dilate))
        boxes = []
        for k, (_, x0, y0, x1, y1, _a) in enumerate(comps):
            if fired or (letters and seen[k] > 0 and hid[k] > (1.0 - min_frac) * seen[k] + 1e-6):
                boxes.append([max(0, int(round(x0 / s)) - d), max(0, int(round(y0 / s)) - d),
                              min(W, int(round(x1 / s)) + d), min(H, int(round(y1 / s)) + d)])
        row = dict(a=float(a), b=float(b), boxes=boxes, hidden=[round(float(v), 3) for v in frac], n=len(comps),
                   occ=round(float(np.mean(occs)), 3) if occs else 0.0, floor=fired)
        if bh and occs:                              # would the word stay readable BEHIND the subject (relaxed floor)?
            row["_ok_behind"] = not (n_word_b > bh["most"] * len(occs) or bool(np.any((seen > 0) & (lo_b > bh["most"] * seen))))
        out.append(row)
    if bh:                                           # the quota: of the holds the subject overlaps, about `share` (at least `min`) stay behind
        on = [r for r in out if r["occ"] > bh["overlap"]]
        natural = [r for r in on if not r["boxes"] and not r["floor"]]
        need = max(int(bh["min"]), int(bh["share"] * len(on) + 0.5)) - len(natural) if on else 0
        for r in sorted([r for r in on if (r["boxes"] or r["floor"]) and r.get("_ok_behind")], key=lambda r: abs(r["occ"] - bh["sweet"])):
            if need <= 0:
                break
            r["boxes"], r["floor"], r["behind"] = [], False, True
            need -= 1
        for r in natural:
            r["behind"] = True
    for r in out:
        r.pop("_ok_behind", None)
    return {"size": size, "holds": out}


def _leg_weight(t, a, b, ramp):
    """0 outside, 1 over [a, b], smoothstep rise over [a - ramp, a], fall over [b, b + ramp]."""
    r = max(float(ramp), 1e-6)
    up = min(max((t - (a - r)) / r, 0.0), 1.0)
    dn = min(max(((b + r) - t) / r, 0.0), 1.0)
    return (up * up * (3.0 - 2.0 * up)) * (dn * dn * (3.0 - 2.0 * dn))


def legible_clear(type_a, plan, t, ramp_s=0.25, dilate=6):
    """clear amount 0..1 (numpy HxW at the type's size; None when nothing is flagged at time t): the live shape of every
    flagged letter (component of the current type alpha overlapping a flagged box by half of the smaller of the two), dilated (hard edge, not feathered), times the
    hold's ramp weight."""
    if not plan or not plan.get("holds"):
        return None
    act = [(h, _leg_weight(t, h["a"], h["b"], ramp_s)) for h in plan["holds"] if h.get("boxes")]
    act = [(h, w) for h, w in act if w > 1e-4]
    if not act:
        return None
    ta = np.asarray(type_a, np.float32)
    if ta.ndim == 3:
        ta = ta[..., 0]
    H, W = ta.shape[:2]
    s = _leg_scale(H, W)
    lab, comps = _leg_components(_leg_small(ta, s))
    clear_s = np.zeros(lab.shape, np.float32)
    ps = plan.get("size") or [W, H]
    kx, ky = W / float(ps[0]) * s, H / float(ps[1]) * s
    for h, w in act:
        for bx in h["boxes"]:
            b0, b1, b2, b3 = bx[0] * kx, bx[1] * ky, bx[2] * kx, bx[3] * ky
            barea = max((b2 - b0) * (b3 - b1), 1e-6)
            for (i, x0, y0, x1, y1, _a) in comps:
                inter = max(min(x1, b2) - max(x0, b0), 0.0) * max(min(y1, b3) - max(y0, b1), 0.0)
                if inter >= 0.5 * min(barea, max((x1 - x0) * (y1 - y0), 1e-6)):
                    clear_s = np.maximum(clear_s, (lab == i).astype(np.float32) * float(w))
    if not clear_s.any():
        return None
    d = max(1, int(round(dilate * s)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    clear_s = np.clip(cv2.dilate(clear_s, k), 0.0, 1.0)     # 2026-11-01 (user: no soft masks around text): hard-edged dilation of the letter shape, no gaussian feather
    if clear_s.shape != (H, W):
        clear_s = cv2.resize(clear_s, (W, H), interpolation=cv2.INTER_LINEAR)
    return clear_s


@api
def legible(type_a, matte, holds=None, t=0.0, min_vis=0.8, min_frac=0.8, ramp_s=0.25, fps=24, mode="front", plan=None,
            dilate=6, matte_threshold=None, matte_soft=0.08, matte_gamma=1.0, invert=False, front=None, floor=None, letters=True, behind=None):
    """Legibility guard: returns the matte with the occlusion removed over the letters it would hide for (almost) the whole
    hold of a word. `holds` = [(hit_s, release_s)] piece seconds, `t` = this frame's piece time, `plan` = the per-hold decision
    from `legible_plan` (recipe_eval.compose_frames computes it before the frame loop; min_vis / min_frac / fps / matte_* /
    invert / front / floor / letters parametrise that decision (front, floor, letters are only read by the decision) and must mirror the following `sandwich` so it
    sees the same occluder). Without holds or
    plan, or with mode "off", the matte is returned untouched; outside the holds (+ ramp_s either side) it is untouched too.
    mode "front": the flagged letters come in front of the matte (matte -> 0 over them; with invert=True the occluder is
    1 - matte, so matte -> 1), ramped over `ramp_s` (the rise ends at the hit, the fall starts at the release)."""
    if mode not in ("front", "off"):
        raise ValueError(f"legible: mode must be 'front' or 'off', got {mode!r}")
    if mode == "off" or not holds or not plan:
        return matte
    c = legible_clear(B.tonumpy(type_a), plan, float(t), ramp_s, dilate)
    if c is None:
        return matte
    m = _gray(matte)
    h, w = m.shape[:2]
    c = B.asarray(cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) if c.shape != (h, w) else c)
    return 1.0 - (1.0 - m) * (1.0 - c) if invert else m * (1.0 - c)


# ======================================================================================================== type face + hold net
# USER RULE (2026-10-27, v005 previews: 16 of 28 holds unreadable): the visible type is ALWAYS the beauty pass, alpha = the type matte;
# every treatment only modulates it (occlude, glow, window). `type_face` turns the beauty pass into a colour layer (+ a sane alpha);
# recipe_eval.legibility_pass wires it into the ops of a recipe; `hold_net` is the last line of defence: per word hold it scores the
# result against the plain beauty-over-plate and blends toward that when the word does not read.

def _l709(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def _bg_guess(b, a, bg=None):
    """beauty background colour (0..1 RGB): the given one, else the commonest colour of the beauty (the matte is not trusted: it may be inverted)."""
    if bg is not None:
        v = np.asarray(bg, np.float32).reshape(-1)[:3]
        return (v / 255.0 if float(v.max()) > 1.0 else v).astype(np.float32)
    q = (np.clip(b, 0, 1) * 31.0 + 0.5).astype(np.int32).reshape(-1, 3)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    top = int(np.bincount(key).argmax())
    c = np.array([top // 1024, (top // 32) % 32, top % 32], np.int32)
    sel = np.abs(q - c).max(-1) <= 1
    return np.median(b.reshape(-1, 3)[sel], axis=0).astype(np.float32)


TF_MIN_STROKE = 0.010     # 2026-11-01 type floor: letter stroke >= this x the frame's short side during a hold (hairline faces: BungeeHairline 2 px at 576 = unreadable)
TF_MIN_CAP = 0.09         # ... and cap height >= this x the frame height (design guide's absolute floor), scaled up about the word's centre ...
TF_MAX_BOX = 0.92         # ... while the word's box stays inside this share of the frame (a single long word is never sliced into letter rows)


def _stroke_px(body):
    """mean stroke width (px) of a binary letter mask: 2 x area / perimeter (exact for long thin strokes)."""
    cs, _ = cv2.findContours(body.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2.0 * float(body.sum()) / max(per, 1e-6)


def type_floor(col, a, wgt, min_stroke=TF_MIN_STROKE, min_cap=TF_MIN_CAP, max_box=TF_MAX_BOX):
    """the run-time readable-size floor of one frame's type (numpy col HxWx3, a HxW) at hold weight wgt (0..1) -> (col, a, info).
    1 CAP: cap height = median letter height / frame height; under min_cap the type is scaled about its box centre by min_cap / cap, limited so the
      box stays within max_box of the frame (and inside it). 2 STROKE: stroke = 2 x area / perimeter; under min_stroke x short side the alpha is grown
      by the missing half-width (distance field, sub-pixel), the colour already covers it (type_face fills outside the letters). Both eased by wgt."""
    info = {}
    if wgt <= 1e-4 or min_cap is None and min_stroke is None:
        return col, a, info
    h, w = a.shape
    body = a > 0.5
    if int(body.sum()) < 30:
        return col, a, info
    if min_cap:                     # letters closed by 3 % of the height first: dot / pixel faces count as letters, never as dots (biased high = never a false scale-up)
        k = max(3, int(round(0.03 * h)))
        closed = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, _lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        big = [i for i in range(1, n) if st[i, 4] >= 0.08 * max(st[1:, 4].max(), 1)]
        cap = float(np.median([st[i, 3] for i in big])) / h if big else 0.0
        info["cap"] = round(cap, 4)
        if 0.0 < cap < float(min_cap):
            ys, xs = np.nonzero(body)
            bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            s = min(float(min_cap) / cap, max_box * w / max(bx1 - bx0, 1), max_box * h / max(by1 - by0, 1))
            s = 1.0 + (max(s, 1.0) - 1.0) * float(wgt)
            if s > 1.005:
                cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                hw, hh = 0.5 * (bx1 - bx0) * s, 0.5 * (by1 - by0) * s
                cx = min(max(cx, hw + 0.5 * (1 - max_box) * w), w - hw - 0.5 * (1 - max_box) * w) if 2 * hw < w else 0.5 * w
                cy = min(max(cy, hh + 0.5 * (1 - max_box) * h), h - hh - 0.5 * (1 - max_box) * h) if 2 * hh < h else 0.5 * h
                M = np.float32([[s, 0, cx - s * 0.5 * (bx0 + bx1)], [0, s, cy - s * 0.5 * (by0 + by1)]])
                a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
                col = cv2.warpAffine(col, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                body = a > 0.5
                info["scale"] = round(s, 3)
    if min_stroke and int(body.sum()) >= 30:
        sw = _stroke_px(body)
        need = float(min_stroke) * min(h, w)
        info["stroke_px"] = round(sw, 2)
        if sw < need:
            r = 0.5 * (need - sw) * float(wgt)
            dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
            a = np.maximum(a, np.clip(r + 1.0 - dist, 0.0, 1.0)).astype(np.float32)      # dist is to the outside pixel's centre: the edge sits half a pixel further
            info["grow_px"] = round(r, 2)
    return col, a, info


def type_face(beauty, type_a, bg=None, mismatch=0.5, thresh=0.04, holds=None, t=0.0, ramp_s=0.25, min_stroke=None, min_cap=None):
    """beauty pass (HxWx3 on the palette background) + type matte -> HxWx4 float32 RGBA: rgb = the type's own colour (un-premultiplied by the
    matte; outside / on the thin edge of the letters it is filled with the nearest letter colour so a warp or blur never pulls the background in),
    a = the type alpha. The alpha is the matte, unless the matte contradicts the beauty (a rasteriser failure, e.g. the camera inside a tunnel
    gives a matte that is white everywhere except the letters' holes): when more than `mismatch` of the matte lies where the beauty is plain
    background, the alpha comes from the beauty instead (distance from the background colour `bg`, estimated when not given).
    Numpy / cv2 op. bg: 0..255 or 0..1 RGB.
    holds ([(hit_s, release_s)] piece seconds) + t + min_stroke / min_cap (fractions, see TF_*): the readable-size floor `type_floor` inside the holds
    (eased over ramp_s either side); recipe_eval.floor_ops sets them on every type_face of a recipe. None = off (old behaviour)."""
    b = np.asarray(beauty, np.float32)
    h, w = b.shape[:2]
    a = np.asarray(type_a, np.float32)
    if a.ndim == 3:
        a = a[..., 3] if a.shape[-1] == 4 else a[..., 0]
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.clip(a, 0.0, 1.0)
    bgc = _bg_guess(b, a, bg)
    d = np.abs(b - bgc).max(-1)
    ab = np.clip((d - thresh * 0.5) / max(thresh * 1.5, 1e-6), 0.0, 1.0)
    ab = ab * ab * (3.0 - 2.0 * ab)
    tot = float(a.sum())
    if tot > 1e-3:
        near = cv2.dilate((ab > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        if float((a * (1.0 - near)).sum()) / tot > float(mismatch):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            a = np.clip(cv2.morphologyEx(ab, cv2.MORPH_CLOSE, k), 0.0, 1.0)
    pm = np.clip(b - bgc * (1.0 - a[..., None]), 0.0, 1.0)                  # = a * colour
    inner = np.where((a > 0.05)[..., None], pm / np.maximum(a, 0.05)[..., None], 0.0).clip(0.0, 1.0)
    solid = (a >= 0.5).astype(np.float32)
    col = inner
    if float(solid.sum()) > 0.0 and float(solid.min()) < 1.0:
        filled = None
        for sg in (2.0, 8.0, 32.0):
            num = cv2.GaussianBlur(inner * solid[..., None], (0, 0), sg)
            den = cv2.GaussianBlur(solid, (0, 0), sg)[..., None]
            f = num / np.maximum(den, 1e-4)
            filled = f if filled is None else np.where(den > 0.02, filled, f)
        col = np.where((solid > 0)[..., None], inner, filled)
    if holds and (min_stroke or min_cap):
        wgt = max((_leg_weight(float(t), float(h0), float(h1), ramp_s) for h0, h1 in holds), default=0.0)
        col, a, _inf = type_floor(np.ascontiguousarray(col, np.float32), np.ascontiguousarray(a, np.float32), wgt, min_stroke, min_cap)
    return np.concatenate([col.astype(np.float32), a[..., None].astype(np.float32)], axis=-1)


NET_LOG = {}                      # state_key -> {piece frame: (hold weight, score, blend k)}; recipe_eval.compose_frames reads it (net_report)


def hold_score(image, alpha, face=None, plain=None, min_slope=0.12):
    """how well the word reads in `image` (HxWx3 float), 0 (not at all) .. 1 (as well as the plain beauty over the same image).
    Two gates on the word's region (bounding box of `alpha` > 0.25 plus a margin), both measured after a small blur:
      step  |slope| of the image luma regressed on the type alpha (the luminance step between the letters and their surround) against
            0.7 x that of `plain` (the same region with the beauty drawn over it; at least `min_slope` absolute);
      shape |correlation| of the band-passed image luma with the band-passed type alpha (the letters' edges / inner detail) against 0.7 x that
            of `plain` (at least 0.25): the letters' structure, not just a flat slab or a glow in the same place.
    score = min(step, shape). -> dict(score, step, shape, slope, ncc, plain_slope, plain_ncc); score 1.0 when there is nothing to measure."""
    a = np.asarray(alpha, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    img = np.asarray(image, np.float32)
    h, w = a.shape
    core = a > 0.25
    if int(core.sum()) < 12:
        return dict(score=1.0, step=1.0, shape=1.0, slope=0.0, ncc=0.0, plain_slope=0.0, plain_ncc=0.0)
    ys, xs = np.nonzero(core)
    pad = max(3, int(round(0.04 * max(h, w))))
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    sg = max(0.7, 0.0025 * max(h, w))

    def stats(x):
        l0 = _l709(x[y0:y1, x0:x1]).astype(np.float32)
        a0 = a[y0:y1, x0:x1]
        l, aa = cv2.GaussianBlur(l0, (0, 0), sg), cv2.GaussianBlur(a0, (0, 0), sg)
        da, dl = aa - aa.mean(), l - l.mean()
        va, vl = float((da * da).mean()), float((dl * dl).mean())
        if va < 1e-8:
            return 0.0, 0.0
        slope = float((da * dl).mean()) / va
        ha, hl = aa - cv2.GaussianBlur(a0, (0, 0), 4.0 * sg), l - cv2.GaussianBlur(l0, (0, 0), 4.0 * sg)       # band-pass: letter detail / edges
        vha, vhl = float((ha * ha).mean()), float((hl * hl).mean())
        return slope, float((ha * hl).mean()) / max(np.sqrt(vha * vhl), 1e-8)

    sl, nc = stats(img)
    if plain is None:
        fc = np.asarray(face, np.float32) if face is not None else np.array([1.0, 0.94, 0.82], np.float32)
        plain = img * (1.0 - a[..., None]) + (fc if fc.ndim == 3 else fc.reshape(1, 1, 3)) * a[..., None]
    ps, pn = stats(np.asarray(plain, np.float32))
    step = float(np.clip(abs(sl) / max(0.7 * abs(ps), min_slope), 0.0, 1.0))
    shape = float(np.clip(abs(nc) / max(0.7 * abs(pn), 0.25), 0.0, 1.0))
    return dict(score=min(step, shape), step=step, shape=shape, slope=sl, ncc=nc, plain_slope=ps, plain_ncc=pn)


def _ss_np(x, e0, e1):
    t = float(np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def hold_net(image, plate, face, alpha, holds=None, t=0.0, frame_idx=0, ramp_s=0.25, lo=0.45, hi=0.85, min_contrast=0.3,
             state_key="net", enable=True, scrim=False):
    """Hold-legibility safety net (numpy op, on the finished image before the closing grade). During a word's hold (hit .. release, ramped over
    `ramp_s` either side) the word has to read: `hold_score` compares `image` with the plain beauty drawn over it; the blend toward that plain
    version is k = 1 - smoothstep(score, lo, hi) x the hold ramp, applied inside the word's region only (the plate / other lines keep their
    treatment). The plain version = the beauty over the (graded) `plate`, with a SCRIM when the beauty colour is too close in luminance to its
    surround (< `min_contrast`): the surround of the word is darkened (light type) or lightened (dark type) just enough. Stateless (the blend depends on this frame only). Every call is logged in NET_LOG[state_key][frame_idx] =
    (hold weight, score, k): see net_report. holds = [[hit, release], ..] piece seconds, t = this frame's piece time."""
    img = np.asarray(image, np.float32)
    if not enable or not holds:
        return img
    wt = max((_leg_weight(float(t), float(a), float(b), float(ramp_s)) for a, b in holds), default=0.0)
    if wt < 1e-3:
        return img
    h, w = img.shape[:2]
    a = np.asarray(alpha, np.float32)
    if a.ndim == 3:
        a = a[..., 3] if a.shape[-1] == 4 else a[..., 0]
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.clip(a, 0.0, 1.0)
    fc = np.asarray(face, np.float32) if face is not None else np.array([1.0, 0.94, 0.82], np.float32)
    if fc.ndim == 3 and fc.shape[:2] != (h, w):
        fc = cv2.resize(fc, (w, h), interpolation=cv2.INTER_LINEAR)
    fcb = fc if fc.ndim == 3 else fc.reshape(1, 1, 3)
    core = (a > 0.25).astype(np.uint8)
    if int(core.sum()) < 12:
        NET_LOG.setdefault(state_key, {})[int(frame_idx)] = (round(wt, 3), 1.0, 0.0)
        return img
    # the plain version = the beauty over the (graded) PLATE, with a scrim when its luminance is too close to the surround's
    pl = np.asarray(plate, np.float32) if plate is not None else img
    if pl.ndim == 2:
        pl = np.stack([pl] * 3, -1)
    if pl.shape[:2] != (h, w):
        pl = cv2.resize(pl, (w, h), interpolation=cv2.INTER_LINEAR)
    r1 = max(3, int(round(0.03 * max(h, w))))
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    near = cv2.dilate(core, kr)
    ring = (near > 0) & (core == 0)
    lface = _l709(np.broadcast_to(fcb, img.shape))
    fl = float(lface[core > 0].mean())
    lum_p = _l709(pl)
    lr = float(lum_p[ring].mean()) if ring.any() else float(lum_p.mean())
    base = pl
    scrimmed = bool(scrim) and abs(fl - lr) < min_contrast
    if scrimmed:           # 2026-10-30 (user: no soft blur / darkening behind the type): the scrim only when asked for (`scrim`, effect soft_support)
        sc = np.clip(cv2.GaussianBlur(near.astype(np.float32), (0, 0), max(1.0, 0.6 * r1)) * 1.3, 0.0, 1.0)[..., None]
        if fl >= lr:
            g = float(np.clip((fl - min_contrast) / max(lr, 1e-3), 0.1, 1.0))
            base = pl * (1.0 - sc * (1.0 - g))
        else:
            b_ = float(np.clip((min(fl + min_contrast, 1.0) - lr) / max(1.0 - lr, 1e-3), 0.0, 1.0))
            base = pl + (1.0 - pl) * sc * b_
    plain = base * (1.0 - a[..., None]) + fcb * a[..., None]
    sc_ = hold_score(img, a, plain=plain)
    k = (1.0 - _ss_np(sc_["score"], float(lo), float(hi))) * wt
    NET_LOG.setdefault(state_key, {})[int(frame_idx)] = (round(wt, 3), round(float(sc_["score"]), 3), round(float(k), 3))
    if k < 1e-3:
        return img
    # 2026-11-01 (user: "not to use soft masks around text"; R15 s1 outline_to_word: this feathered dilation of the word blended the dark picture toward the plate = a soft window round the word):
    # the net acts on the LETTER SHAPES only (hard alpha, no dilation, no blur): the letters go toward the plain face colour, nothing round them changes
    if scrimmed:                                         # only the explicit opt-in (compose.soft_support) keeps the old feathered region round the word
        reg = np.clip(cv2.GaussianBlur(near.astype(np.float32), (0, 0), max(1.0, 0.5 * r1)) * 1.4, 0.0, 1.0)[..., None]
        return np.clip(img * (1.0 - k * reg) + plain * (k * reg), 0.0, 1.0).astype(np.float32)
    reg = a[..., None]
    return np.clip(img * (1.0 - k * reg) + fcb * (k * reg), 0.0, 1.0).astype(np.float32)


def net_report(state_key, holds, fps=24):
    """per-hold summary of NET_LOG[state_key] for the QC / compose report: [{hit, release, frames, mid_score, min_score, max_k, fired_frames,
    fired}] (`fired` = the net blended anywhere inside the hold proper; mid_score = the score at the hold's middle frame)."""
    log = NET_LOG.get(state_key) or {}
    out = []
    for a, b in holds:
        f0, f1 = int(round(a * fps)), int(round(b * fps))
        fr = [f for f in sorted(log) if f0 <= f < f1]
        rows = [log[f] for f in fr]
        mid = min(fr, key=lambda f: abs(f - (f0 + f1) // 2)) if fr else None
        out.append(dict(hit=round(float(a), 3), release=round(float(b), 3), frames=len(rows),
                        mid_score=(round(log[mid][1], 3) if mid is not None else None),
                        min_score=round(min((r[1] for r in rows), default=1.0), 3), max_k=round(max((r[2] for r in rows), default=0.0), 3),
                        fired_frames=sum(1 for r in rows if r[2] > 0.05), fired=any(r[2] > 0.05 for r in rows)))
    return out


# ======================================================================================================== palette / tint

def _kmeans(x, k, iters=24, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    cent = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(cent)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(x[rng.choice(n, p=p)] if p is not None else x[rng.integers(n)])
    cent = np.asarray(cent, np.float64)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = cent.copy()
        for j in range(k):
            sel = x[lab == j]
            if len(sel):
                new[j] = sel.mean(0)
        if np.allclose(new, cent, atol=1e-4):
            cent = new
            break
        cent = new
    return cent


def palette_from_footage(frames, k=5, seed=0, max_frames=24, sample_px=12000, stride=1):
    """k-means palette of a shot -> (k,3) float32 sorted dark to light. `frames`: HxWx3, THxWx3 array, list or
    generator (generators are read every `stride` frames up to `max_frames`; sequences are sampled evenly)."""
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    if hasattr(frames, "__len__") and hasattr(frames, "__getitem__"):
        n = len(frames)
        idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
        picked = (frames[int(i)] for i in idx)
    else:
        picked = (f for i, f in enumerate(frames) if i % max(stride, 1) == 0)
    rng = np.random.default_rng(seed)
    px = []
    for j, f in enumerate(picked):
        if j >= max_frames:
            break
        f = np.asarray(_out(f), np.float32)
        h, w = f.shape[:2]
        s = 96.0 / max(h, w)
        small = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        px.append(small.reshape(-1, 3))
    x = np.concatenate(px, 0)
    if len(x) > sample_px:
        x = x[rng.choice(len(x), sample_px, replace=False)]
    cent = _kmeans(x.astype(np.float64), int(k), seed=seed)
    lum = cent @ np.array([0.2126, 0.7152, 0.0722])
    return cent[np.argsort(lum)].astype(np.float32)


def _set_luma(c, target, amount, max_gain=4.0):
    l = _c3(_luma(c))
    want = _lerp(l, target, amount)
    return c * _clip(want / (l + 1e-4), 0.0, max_gain)


def _soft_clip_rgb(c):
    m = _c3(B.maximum(B.maximum(c[..., 0], c[..., 1]), c[..., 2]))
    return _clip(c / B.maximum(m, 1.0))


@api
def tint_type(type_a, footage, mode="sample", palette=None, sample_blur=None, sat=1.5, lum_target=0.72,
              lum_amount=0.6, invert=False, local=True, lo=None, hi=None):
    """Colour for the type (HxWx3, full frame - use with type_a). Modes:
    'sample'       blurred footage colour under the letters, saturation boosted, luma pushed toward lum_target
    'complement'   the opposite hue of the (locally) blurred footage, same boosts
    'gradient_map' blurred footage luma (contrast-stretched) mapped through `palette` ((k,3) dark->light; from the
                   footage itself if omitted). invert=True maps light footage to dark palette ends."""
    h, w = footage.shape[:2]
    sb = sample_blur if sample_blur is not None else 0.02 * h
    if mode == "sample":
        c = B.blur(footage, sb)
    elif mode == "complement":
        c = B.blur(footage, sb if local else max(sb, 0.35 * h))
        c = 2.0 * _c3(_luma(c)) - c  # 180 degree hue rotation (chroma negated, luma kept)
    elif mode == "gradient_map":
        pal = palette
        if pal is None:
            pal = palette_from_footage([_out(footage)], k=5)
        pal = B.asarray(np.asarray(_out(pal) if _is_arr(pal) else pal, np.float32))
        k = pal.shape[0]
        l = _luma(B.blur(footage, sb))
        if lo is None or hi is None:
            p = B.percentile(l, [2, 98])
            lo = p[0] if lo is None else lo
            hi = p[1] if hi is None else hi
        t = _clip((l - lo) / max(hi - lo, 1e-4))
        if invert:
            t = 1.0 - t
        pos = t * (k - 1)
        if k == 1:
            return _clip(pal[0][None, None, :] + 0 * _c3(t))
        i0 = _clip(pos.floor() if hasattr(pos, "floor") else np.floor(pos), 0, k - 2)
        fr = _c3(pos - i0)
        ii = i0.long() if hasattr(i0, "long") else i0.astype(np.int64)
        c = _lerp(pal[ii], pal[ii + 1], fr)
        return _clip(c)
    else:
        raise ValueError(f"tint_type mode {mode!r}: expected sample | complement | gradient_map")
    if invert and mode != "gradient_map":
        c = 1.0 - c
    c = _clip(_sat(c, sat))
    c = _set_luma(c, lum_target, lum_amount)
    return _soft_clip_rgb(c)


# ======================================================================================================== edges / trace

class EdgeTracer:
    """Stateful edge extractor: gradient magnitude with temporal hold-decay -> hysteresis-ish soft threshold."""

    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def step(self, footage, low=0.03, high=0.09, temporal_smooth=0.5, blur=1.2):
        return _out(self._step(_in(footage), low, high, temporal_smooth, blur))

    def _step(self, footage, low, high, ts, blur):
        g = _gray(footage) if footage.ndim == 3 else footage
        g = B.blur(g, blur)
        gx, gy = B.sobel(g)
        mag = B.sqrt(gx * gx + gy * gy) * 2.0  # a unit step edge ~ 1.0
        if self.prev is not None and self.prev.shape == mag.shape:
            mag = B.maximum(mag, ts * self.prev)  # hold-with-decay: no dropout flicker, moving edges keep full strength
        self.prev = mag
        strong = _smoothstep(mag, high * 0.8, high)
        weak = _smoothstep(mag, low * 0.8, low)
        return _clip(B.maximum(strong, weak * _dilate(strong, 3)))


def trace_edges(footage, low=0.03, high=0.09, temporal_smooth=0.5, blur=1.2):
    """Clean edge map from footage luma, soft [0,1], temporally smoothed (hold-decay of gradient magnitude, then a
    low/high hysteresis-style threshold; weak edges survive only next to strong ones). temporal_smooth = per-frame
    decay of the held magnitude (max(current, smooth*previous)): removes dropout flicker without dimming moving edges.
    HxWx3 -> HxW; THxWx3 array -> THxW array; any other iterable of frames -> generator of HxW maps."""
    tr = EdgeTracer()
    if isinstance(footage, np.ndarray) and footage.ndim == 3:
        return tr.step(footage, low, high, temporal_smooth, blur)
    if isinstance(footage, np.ndarray) and footage.ndim == 4:
        return np.stack([tr.step(f, low, high, temporal_smooth, blur) for f in footage])
    return (tr.step(f, low, high, temporal_smooth, blur) for f in footage)


@api
def edges_in_type(edges, type_a, gain=1.0, spread=0.0):
    """Footage lines visible only inside the letters -> alpha (HxW). spread>0 grows the letter mask by that many px."""
    h, w = edges.shape[:2]
    ta = _mask(type_a, h, w)
    if spread > 0:
        ta = B.blur(_dilate(ta, spread), max(spread * 0.4, 0.5))
    return _clip(_gray(edges) * _clip(ta) * gain)


@api
def type_from_edges(edges, type_a, grow=10.0, line_width=1.0, density=1.0, fill=0.0, softness=4.0):
    """Letters built from the scene's contours: edges (thickened by line_width) revealed through the dilated type
    matte (grow px, soft edge softness px) -> alpha (HxW). fill>0 adds a faint solid letter body underneath."""
    h, w = edges.shape[:2]
    ta = _clip(_mask(type_a, h, w))
    field = B.blur(_dilate(ta, grow), softness) if grow > 0 else ta
    e = _dilate(_gray(edges), line_width) if line_width > 0 else _gray(edges)
    a = _clip(e * field * density)
    if fill > 0:
        a = B.maximum(a, ta * fill)
    return a


# ======================================================================================================== contour lines

_CL_WORK = 960          # contours are traced on a copy of src whose long side is at most this (then scaled to the output size)


def _cl_gray(src):
    a = np.asarray(src, np.float32)
    if a.ndim == 3:
        c = a.shape[-1]
        a = a[..., 0] if c == 1 else (a[..., 3] if c == 4 else a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722)
    return np.ascontiguousarray(a)


def _cl_find(a, t):
    """closed pixel contours (list of (n,2) float arrays, x y) of a > t; contours along the image border are pushed off-canvas."""
    pad = 3                      # replicate-pad so regions cut by the frame edge trace along the pad ring (just off-canvas), not the edge
    b = (cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_REPLICATE) > t).astype(np.uint8)
    cnts = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    return [c[:, 0, :].astype(np.float32) - pad for c in cnts if len(c) >= 8]


def _cl_smooth(p, k):
    """circular moving average of a closed polyline (k points each side): removes the pixel staircase."""
    if k <= 0 or len(p) < 2 * k + 3:
        return p
    q = np.concatenate([p[-k:], p, p[:k]])
    ker = np.ones(2 * k + 1, np.float32) / (2 * k + 1)
    return np.stack([np.convolve(q[:, i], ker, mode="valid") for i in range(2)], 1)


def _cl_arc(p, closed=True):
    q = np.concatenate([p, p[:1]]) if closed else p
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _cl_cut(p, frac, closed):
    """first `frac` of the arc length of polyline p -> (points, closed_flag)."""
    if frac >= 0.999:
        return p, closed
    if frac <= 0.0:
        return None, False
    arc = _cl_arc(p, closed)
    q = np.concatenate([p, p[:1]]) if closed else p
    end = frac * arc[-1]
    i = int(np.searchsorted(arc, end))
    i = min(max(i, 1), len(q) - 1)
    seg = max(arc[i] - arc[i - 1], 1e-6)
    tip = q[i - 1] + (q[i] - q[i - 1]) * ((end - arc[i - 1]) / seg)
    return np.concatenate([q[:i], tip[None]]), False


def _cl_draw(canvas, p, closed, scale, thick, col):
    if p is None or len(p) < 2:
        return
    pts = np.round(p * np.asarray(scale, np.float32) * 4.0).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], bool(closed), int(col), int(thick), cv2.LINE_AA, 2)


def _cl_scribble(p, rng, jitter, bias, cover, step=5.0):
    """one hand-drawn stroke around the closed contour p: starts at a random point, covers `cover` x the arc length (>1 overshoots
    like a pencil), displaced along the normal by smooth low-frequency noise (px of the work grid)."""
    arc = _cl_arc(p, True)
    total = float(arc[-1])
    if total < 8:
        return None
    pc = np.concatenate([p, p[:1]])
    m = int(max(8, cover * total / step))
    s = (rng.random() * total + np.linspace(0.0, cover * total, m)) % total
    d = np.stack([np.interp(s, arc, pc[:, 0]), np.interp(s, arc, pc[:, 1])], 1)
    tan = np.gradient(d, axis=0)
    ln = np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    nrm = np.stack([tan[:, 1], -tan[:, 0]], 1) / ln
    nc = max(4, int(cover * total / 55.0)) + 1
    grid = np.linspace(0, nc - 1, m)
    off = np.interp(grid, np.arange(nc), rng.uniform(-1.0, 1.0, nc)) * jitter + bias
    offt = np.interp(grid, np.arange(nc), rng.uniform(-1.0, 1.0, nc)) * jitter * 0.6
    return d + nrm * off[:, None] + (tan / ln) * offt[:, None]


def contour_lines(src, mode="iso", levels=6, phase=0.0, width=2.0, reveal=1.0, smooth=2.0, boil=3, frame_idx=0,
                  jitter=4.0, strokes=3, seed=0, lo=0.0, hi=1.0, thresh=0.5, min_len=24.0, max_contours=6,
                  stagger=0.0, size=None):
    """Animated, subject-tracking contour lines -> alpha HxW float32 (lines = 1, anti-aliased). Numpy / cv2 op.
    src     HxW float (depth `$sN_depth` or matte `$sN_matte`; HxWx3/4 is reduced to luma/alpha). Re-contoured on every frame, so
            the lines follow the subject as src moves.
    mode    "iso": `levels` iso-lines of src between lo..hi. "outline": the outline of src > thresh (a matte). "scribble": 2-3 jittered,
            offset hand-drawn strokes (`strokes`) around the largest `max_contours` outlines; re-randomised every `boil` frames.
    levels  number of iso-lines (iso only).      phase  0..1 shifts the iso levels up through lo..hi so the lines flow (loops with
            period 1: lines fade in at one end and out at the other); give it {"timeline":..} / {"anim":..}.
    width   line width in OUT px (int, min 1).   reveal  0..1 fraction of each contour's arc length drawn (draw-on / draw-off;
            `stagger` 0..1 delays contour k of n so they draw in sequence).   smooth  Gaussian sigma on src before contouring.
    jitter  scribble wobble in OUT px.   frame_idx  pass "$frame_idx" (drives the boil).   size  [W, H] output size (default src size).
    min_len drops contours shorter than this many OUT px.
    Use as the `edges` input of `type_from_edges` (letters form out of / dissolve into the contours; gate with reveal or the type
    alpha timeline), or draw the lines over the picture in 2 ops:
        {"op": "contour_lines", "out": "cl", "args": {"src": "$s1_depth", "mode": "iso", "levels": 8, "width": 2,
                                                       "phase": {"timeline": [[0, 0.0], [8, 1.0]]}, "frame_idx": "$frame_idx"}}
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": [1.0, 0.92, 0.75], "alpha": "$cl"}}   # or add_glow(mask=$cl)
    """
    a = _cl_gray(src)
    H, W = a.shape
    ow, oh = (int(size[0]), int(size[1])) if size else (W, H)
    ls = max(H, W)
    if ls > _CL_WORK:
        f = _CL_WORK / float(ls)
        a = cv2.resize(a, (max(8, int(round(W * f))), max(8, int(round(H * f)))), interpolation=cv2.INTER_AREA)
    wh, ww = a.shape
    sc = (ow / float(ww), oh / float(wh))
    s_avg = 0.5 * (sc[0] + sc[1])
    if smooth and smooth > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(smooth) * ww / W if W > ww else float(smooth))
    canvas = np.zeros((oh, ow), np.uint8)
    thick = max(1, int(round(width)))
    reveal = float(np.clip(reveal, 0.0, 1.0))
    stagger = float(np.clip(stagger, 0.0, 0.95))
    min_w = float(min_len) / max(s_avg, 1e-6)
    sm_k = 3

    def frac_for(k, n):
        return float(np.clip((reveal - stagger * (k / max(n, 1))) / (1.0 - stagger), 0.0, 1.0))

    if mode == "iso":
        n = max(1, int(levels))
        todo = []
        for k in range(n + 1):
            fr = (k + float(phase)) / n
            wgt = float(np.clip(min(fr, 1.0 - fr) / (0.5 / n), 0.0, 1.0))
            if wgt > 0.0:
                todo.append((lo + fr * (hi - lo), wgt))
        found = []
        for t, wgt in todo:
            cs = [_cl_smooth(c, sm_k) for c in _cl_find(a, t)]
            found.append((wgt, [c for c in cs if _cl_arc(c)[-1] >= min_w]))
        total = max(sum(len(cs) for _, cs in found), 1)
        k = 0
        for wgt, cs in found:
            for c in cs:
                p, closed = _cl_cut(c, frac_for(k, total), True)
                _cl_draw(canvas, p, closed, sc, thick, 255 * wgt)
                k += 1
    elif mode in ("outline", "scribble"):
        cs = [_cl_smooth(c, sm_k) for c in _cl_find(a, thresh)]
        cs = [c for c in cs if _cl_arc(c)[-1] >= min_w]
        cs.sort(key=lambda c: -_cl_arc(c)[-1])
        cs = cs[:max(1, int(max_contours))]
        if mode == "outline":
            for k, c in enumerate(cs):
                p, closed = _cl_cut(c, frac_for(k, len(cs)), True)
                _cl_draw(canvas, p, closed, sc, thick, 255)
        else:
            bucket = int(frame_idx) // max(1, int(boil))
            jw = float(jitter) / max(s_avg, 1e-6)
            ns = max(1, int(strokes))
            for k, c in enumerate(cs):
                for si in range(ns):
                    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, bucket, si, k])
                    bias = (si - 0.5 * (ns - 1)) * 0.9 * jw
                    st = _cl_scribble(c, rng, jw, bias, cover=float(rng.uniform(0.78, 1.08)))
                    if st is None:
                        continue
                    p, _ = _cl_cut(st, frac_for(k, len(cs)), False)
                    _cl_draw(canvas, p, False, sc, max(1, thick - (1 if si and thick > 1 else 0)), 255)
    else:
        raise ValueError("contour_lines mode must be 'iso', 'outline' or 'scribble'")
    return canvas.astype(np.float32) / 255.0


# ======================================================================================================== refraction

@api
def refract(footage, type_depth_or_matte, strength=12.0, blur=6.0, chroma=0.15, rim=0.5, rim_color=(1.0, 1.0, 1.0),
            light=(-0.6, -0.8), mask=None):
    """Letters as glass. The (blurred) type depth/matte is a height field; footage is displaced by its gradient
    (strength ~ max displacement in px at a step edge; positive pulls inner content outward), R/B channels displaced
    by (1+-chroma) of that for a chromatic split, and a specular + fresnel rim is screened on. Where the height
    field is flat nothing changes (pixel-exact). mask (HxW) limits the effect, e.g. dilated type matte."""
    h, w = footage.shape[:2]
    d = _mask(type_depth_or_matte, h, w)
    hf = B.blur(d, blur)
    gx, gy = B.sobel(hf)
    s = strength * max(blur, 0.5) * 2.5
    dx, dy = gx * s, gy * s
    if chroma != 0:
        chans = []
        for c, f in ((0, 1.0 + chroma), (1, 1.0), (2, 1.0 - chroma)):
            chans.append(B.warp(footage[..., c], dx * f, dy * f))
        warped = B.stack(chans, axis=-1)
    else:
        warped = B.warp(footage, dx, dy)
    wgt = _clip(B.sqrt(dx * dx + dy * dy) * 1000.0)  # exactly 0 where the field is flat
    out = _lerp(footage, warped, _c3(wgt))
    if rim > 0:
        rs = max(1.0, blur * 0.4)
        h2 = B.blur(d, rs)
        rx, ry = B.sobel(h2)
        gm = B.sqrt(rx * rx + ry * ry)
        gmn = _clip(gm * rs * 2.5)
        lx, ly = light
        ln = math.hypot(lx, ly) or 1.0
        spec = _clip(-(rx * lx / ln + ry * ly / ln) / (gm + 1e-6)) ** 2
        r = _c3(_clip(gmn * (0.25 + 0.75 * spec)) * rim) * _vec3(rim_color, footage)
        out = 1.0 - (1.0 - out) * (1.0 - _clip(r))
    if mask is not None:
        out = _lerp(footage, out, _c3(_clip(_mask(mask, h, w))))
    return out


# ======================================================================================================== flow wake (P06 v003)

def _tanh(z):
    return 2.0 / (1.0 + B.exp(-2.0 * z)) - 1.0


def _rs(x, h, w):
    """resize to (h, w): area when shrinking, bilinear when growing (HxW or HxWxC)."""
    if x.shape[0] == h and x.shape[1] == w:
        return x
    return B.resize(x, h, w, "area" if (h <= x.shape[0] and w <= x.shape[1]) else "linear")


# ECHO CAP (user 2026-11-01: "the echo is too strong and covers too much of the image"): the wake is a subtle accent in a small
# band around the letters, never a smear across the frame. Applied in FlowWake._step, so every source (rules WAKE, model designs,
# presets, direct calls) is bounded at run time. radius is a fraction of the compose short side; max_disp is already one.
ECHO_CAP = {"trail": 3.0, "trail_gain": 0.4, "radius_frac": 0.03, "max_disp": 0.04, "strength": 2.0, "decay": 0.85, "bg_drag": 0.05}


def cap_wake_args(args: dict, short_side=None) -> dict:
    """flow_wake args (numbers, or a {'curve', 'lo', 'hi'} strength) -> a copy bounded by ECHO_CAP. radius is capped only when
    short_side (compose px) is known; other dict-valued args ({timeline} / {anim}) are left to the run-time cap."""
    c, out = ECHO_CAP, dict(args)

    def lim(v, hi):
        if isinstance(v, dict):
            v = dict(v)
            for k in ("lo", "hi"):
                if isinstance(v.get(k), (int, float)):
                    v[k] = min(float(v[k]), hi)
            return v
        try:
            return min(float(v), hi)
        except (TypeError, ValueError):
            return v
    for k in ("trail", "trail_gain", "max_disp", "strength", "decay", "bg_drag"):
        if k in out:
            out[k] = lim(out[k], c[k])
    if "radius" in out and short_side:
        out["radius"] = lim(out["radius"], c["radius_frac"] * float(short_side))
    return out


class FlowWake:
    """Stateful 'footage motion pushes the type' op (use via flow_wake / .step). A displacement field P (content offset, in px of a
    ~270 px wide simulation grid) is advected along the injected velocity, relaxes by `decay` every frame, and warps the type:
    out(x) = type(x - P(x)). Velocity is injected where the dilated, feathered subject matte overlaps (the neighbourhood of) the
    type, from the footage's optical flow; a fading echo buffer advected along the same velocity leaves the streaks."""

    SIM_W = 270.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.P = self.T = self.TP = None
        self.last = self.sig = None
        self.trail_live = False

    def step(self, type_a, flow, matte, type_rgb=None, **kw):
        """numpy in / numpy out (HxW alpha, or HxWx4 RGBA when type_rgb is given)."""
        a = _in(type_a), _in(flow), _in(matte)
        return _out(self._step(*a, type_rgb=None if type_rgb is None else _in(type_rgb), **kw))

    def _step(self, type_a, flow, matte, type_rgb=None, strength=1.0, radius=40.0, decay=0.88, trail=6.0, gate=1.0,
              bg_drag=0.12, max_disp=0.10, trail_gain=0.8, settle=0.0, smooth=1.0):
        h, w = type_a.shape[:2]
        cap = cap_wake_args(dict(strength=strength, radius=radius, decay=decay, trail=trail, bg_drag=bg_drag, max_disp=max_disp,
                                 trail_gain=trail_gain), min(h, w))                     # ECHO CAP: a small band around the letters
        strength, radius, decay, trail = cap["strength"], cap["radius"], cap["decay"], cap["trail"]
        bg_drag, max_disp, trail_gain = cap["bg_drag"], cap["max_disp"], cap["trail_gain"]
        ta = _clip(_gray(type_a))
        rgb = None if type_rgb is None else _fit(type_rgb, h, w)
        s = min(1.0, self.SIM_W / w)
        sh, sw = max(8, int(round(h * s))), max(8, int(round(w * s)))
        sig = (h, w, rgb is not None)
        if self.P is None or self.sig != sig:
            self.P = B.asarray(np.zeros((sh, sw, 2), np.float32))
            self.T = B.asarray(np.zeros((h, w), np.float32))
            self.TP = B.asarray(np.zeros((h, w, 3), np.float32)) if rgb is not None else None
            self.sig, self.trail_live = sig, False
        settle = min(max(float(settle), 0.0), 1.0)
        st = float(strength) * max(float(gate), 0.0) * (1.0 - 0.7 * settle)
        if settle >= 0.999:                                                  # fully settled (a word's hold, plan_rules.SETTLE_HOLD): no new push at all, the displacement relaxes (decay ** 4 per frame), no echo is seeded
            st = 0.0
        # footage flow (px/frame at the flow's own grid) -> simulation grid, smoothed, soft speed cap
        fh, fw = flow.shape[:2]
        fs = _rs(flow, sh, sw) * (sw / float(fw))
        fs = B.blur(fs, 1.0)
        spd = B.sqrt(fs[..., 0] ** 2 + fs[..., 1] ** 2)
        vmax = 0.06 * sh
        fs = fs * _c3(vmax * _tanh(spd / vmax) / (spd + 1e-6))
        # influence: dilated + feathered subject matte (plus a weak background drag), only near the type
        rs_ = max(float(radius) * s, 1.0)
        ms = _clip(_rs(_gray(matte), sh, sw))
        infl = _clip(B.blur(_dilate(ms, int(round(rs_ * 0.5))), rs_ * 0.4) * 1.5)
        wgt = infl + float(bg_drag) * (1.0 - infl)
        ts = _rs(ta, sh, sw)
        reach = _clip(B.blur(_dilate(ts, int(round(rs_ * 0.35))), max(rs_ * 0.35, 1.0)) * 1.5)
        u = fs * _c3(wgt * reach * st)                                   # injected velocity, sim px / frame
        # advect the displacement along u, add the push, relax, soft-clamp
        P = B.warp(self.P, -u[..., 0], -u[..., 1]) + u
        P = P * (float(decay) ** (1.0 + 3.0 * settle))
        if smooth > 0:
            P = B.blur(P, float(smooth))
        md = max(float(max_disp) * min(h, w) * s, 1e-3)
        mag = B.sqrt(P[..., 0] ** 2 + P[..., 1] ** 2)
        P = P * _c3(md * _tanh(mag / md) / (mag + 1e-6))
        self.P = P
        moving = B.amax(mag) > 0.02
        if not moving and not self.trail_live:                           # nothing displaced, no echo: exact identity
            return ta if rgb is None else B.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2], ta], axis=-1)
        inv = 1.0 / s
        Pf = _rs(P, h, w) * inv
        ad = B.warp(ta, -Pf[..., 0], -Pf[..., 1]) if moving else ta
        cd = (B.warp(rgb, -Pf[..., 0], -Pf[..., 1]) if moving else rgb) if rgb is not None else None
        # echo buffer: seeded by the displaced type where the subject moves fast, advected by the same velocity, fading
        T, TP = self.T, self.TP
        if trail > 0:
            uf = _rs(u, h, w) * inv
            fade = 1.0 - 1.0 / max(float(trail), 1.0)
            tadv = B.warp(T, -uf[..., 0], -uf[..., 1]) * fade
            sw_ = _clip(B.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2) / (0.008 * sh)) * infl
            seed = ad * _rs(sw_, h, w) * (1.0 - settle)
            T = seed + tadv * (1.0 - seed)
            if rgb is not None:
                tpa = B.warp(TP, -uf[..., 0], -uf[..., 1]) * fade
                TP = seed[..., None] * cd + tpa * _c3(1.0 - seed)
            self.T, self.TP = T, TP
            self.trail_live = B.amax(T) > 0.004
        tg = T * float(trail_gain)
        out_a = ad + tg * (1.0 - ad)
        if rgb is None:
            return out_a
        pm = ad[..., None] * cd + (TP * float(trail_gain)) * _c3(1.0 - ad)
        col = B.where(_c3(out_a) > 1e-3, pm / _c3(B.maximum(out_a, 1e-3)), cd)
        return B.stack([col[..., 0], col[..., 1], col[..., 2], out_a], axis=-1)


_WAKE = {}


def _wake_state(key, frame_idx, u, sig_hw):
    """implicit per-key state for the recipe op: continues while frame_idx advances by 1 inside a shot (u > 0), else starts cold."""
    st = _WAKE.get(key)
    cont = (st is not None and frame_idx is not None and st.last is not None and int(frame_idx) == st.last + 1
            and not (u is not None and float(u) <= 0.0))
    if not cont:
        st = _WAKE[key] = FlowWake()
    st.last = int(frame_idx) if frame_idx is not None else None
    return st


def _flow_wake(type_a, flow, matte, type_rgb=None, strength=1.0, radius=40.0, decay=0.88, trail=6.0, gate=1.0, bg_drag=0.12,
               max_disp=0.10, trail_gain=0.8, settle=0.0, smooth=1.0, frame_idx=None, u=None, state_key="default", state=None):
    st = state if state is not None else _wake_state(state_key, frame_idx, u, None)
    return st._step(type_a, flow, matte, type_rgb=type_rgb, strength=strength, radius=radius, decay=decay, trail=trail,
                    gate=gate, bg_drag=bg_drag, max_disp=max_disp, trail_gain=trail_gain, settle=settle, smooth=smooth)


flow_wake = api(_flow_wake)
flow_wake.__doc__ = """The type reacts to the footage's motion. type_a: type alpha HxW; flow: footage flow [h,w,2] (px/frame of its own grid, see
footage_flow.py); matte: subject matte (white = runner). Optional type_rgb (HxWx3) -> returns HxWx4 RGBA (pull with rgba_part),
else the displaced alpha HxW.
  strength   flow -> push gain (1 = the type moves as fast as the footage under the matte); a {'curve': 'onset'} arg makes it beat-reactive
  radius     px (compose size): dilation + feather of the matte, and of the type, that decide where motion is injected
  decay      per-frame retention of the displacement (0.88 -> settles back in ~15 frames)
  trail      echo length in frames (fade = 1 - 1/trail); 0 = no echo.   trail_gain: echo opacity
  gate       0..1 overall multiplier (e.g. a timeline).  settle 0..1: 1 = settled: no new push, no echo, the displacement relaxes fast (legible holds)
  bg_drag    push from the background flow away from the matte (fraction of strength).  max_disp: soft cap, fraction of min(h, w)
State: carried across frames in a FlowWake. Pass frame_idx (+ u = shot progress) and the op keeps / resets it by itself (consecutive
frame_idx continues, u <= 0 or a jump restarts); state_key separates simultaneous users; or hand in state=FlowWake()."""


@api
def rgba_part(rgba, part="a"):
    """HxWx4 -> alpha (HxW, part='a') or colour (HxWx3, part='rgb'). Companion of flow_wake with type_rgb."""
    return rgba[..., 3] if part == "a" else rgba[..., :3]


# ======================================================================================================== hold layouts

def _shape_polygon(sh, w, h):
    """frame-fraction shape -> float polygon in px (n, 2): rect (centre cx, cy; w, h fractions of the width / height; angle degrees clockwise
    about the centre, in px space) or poly (pts)."""
    if sh["kind"] == "poly":
        return np.array([[float(x) * w, float(y) * h] for x, y in sh["pts"]], np.float64)
    cx, cy = float(sh["cx"]) * w, float(sh["cy"]) * h
    hw, hh = float(sh["w"]) * w / 2.0, float(sh["h"]) * h / 2.0
    a = math.radians(float(sh.get("angle", 0.0)))
    ca, sa = math.cos(a), math.sin(a)
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return np.array([[cx + x * ca - y * sa, cy + x * sa + y * ca] for x, y in pts], np.float64)


def shape_mask(sh, h, w, t=None):
    """anti-aliased coverage (h, w) float32 0..1 of one layout shape (rect / poly / circle, frame fractions). A shape with a ``pose``
    ({pivot, keys [[t, dx, dy, dang]], ease}, tools/fusion/strip_resnap.py) is moved rigidly to its pose at piece time ``t``."""
    m = np.zeros((h, w), np.uint8)
    mv = None
    if t is not None and sh.get("pose"):
        from tools.fusion import strip_resnap as _sr
        ps = sh["pose"]
        dx, dy, da = _sr.pose_at(ps["keys"], float(t), ps.get("ease", "smooth"))
        mv = lambda P: _sr.move_xy(P, dx * w, dy * h, da, float(ps["pivot"][0]) * w, float(ps["pivot"][1]) * h)  # noqa: E731
    if sh["kind"] == "circle":
        r = float(sh["r"]) * min(w, h)
        c = np.array([float(sh["cx"]) * w, float(sh["cy"]) * h])
        c = mv(c) if mv else c
        cv2.circle(m, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(r * 16)), 255, -1, cv2.LINE_AA, 4)
    else:
        poly = _shape_polygon(sh, w, h)
        pts = np.round((mv(poly) if mv else poly) * 16).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255.0


def layout_shapes(img, shapes, t=0.0, fade=0.25, matte=None, opacity=1.0):
    """Flat backing shapes of a hold layout (tools/fusion/hold_layouts.py) painted onto the plate ``img`` (HxWx3).
    shapes   list of {kind: rect|poly|circle, geometry in frame fractions, color [r, g, b] 0..1, t0, t1 (piece seconds)}. A shape is on screen
             between t0 and t1 and fades in / out over ``fade`` s (smoothstep); outside its window it costs nothing (``img`` is returned as is).
    t        piece time in seconds (`$t`).
    matte    optional subject matte HxW: the shapes are never painted over the subject (it stays in front of the strip).
    opacity  overall multiplier (flat solid strips: 1.0).
    Shapes are in SCREEN space, so the hold camera is held while a shape is up (plan_rules.weave_layout / hold_layouts.anchored).
    Numpy op (cv2 raster; recipe_eval detours through numpy), returns float32 HxWx3."""
    base = _out(img) if _is_arr(img) else np.asarray(img, np.float32)
    t = float(t)
    live = []
    for sh in shapes or []:
        t0, t1 = float(sh.get("t0", -1e9)), float(sh.get("t1", 1e9))
        f = max(float(fade), 1e-6)
        a = min(1.0, max(0.0, (t - t0) / f)) * min(1.0, max(0.0, (t1 - t) / f))
        a = a * a * (3.0 - 2.0 * a) * float(opacity)
        if a > 1e-4:
            live.append((sh, a))
    if not live:
        return np.asarray(base, np.float32)
    h, w = base.shape[:2]
    out = np.array(base, np.float32)
    keep = None
    if matte is not None:
        mm = np.asarray(_out(matte) if _is_arr(matte) else matte, np.float32)
        if mm.ndim == 3:
            mm = mm[..., 3] if mm.shape[-1] == 4 else mm[..., 0]
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_LINEAR)
        keep = 1.0 - np.clip(mm, 0.0, 1.0)
    for sh, a in live:
        cov = shape_mask(sh, h, w, t) * a
        if keep is not None:
            cov = cov * keep
        col = np.asarray(sh["color"], np.float32).reshape(1, 1, 3)
        out = out * (1.0 - cov[..., None]) + col * cov[..., None]
    return out


# ======================================================================================================== utilities

@api
def fit_cover(img, size):
    """Scale to cover size=(w,h) and centre-crop (e.g. any aspect -> 9:16)."""
    tw, th = int(size[0]), int(size[1])
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img
    s = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * s))), max(th, int(round(h * s)))
    y = B.resize(img, nh, nw, "area" if s < 1 else "cubic")
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    y = y[y0:y0 + th, x0:x0 + tw]
    return _clip(y) if s >= 1 else y


@api
def crossfade(a, b, t, smooth=False):
    """a -> b. t scalar 0..1 or HxW map; smooth applies smoothstep to t."""
    if _is_arr(t):
        t = _c3(_mask(t, a.shape[0], a.shape[1])) if t.ndim == 2 else t
        if smooth:
            t = _smoothstep(t, 0.0, 1.0)
    else:
        t = float(min(max(t, 0.0), 1.0))
        if smooth:
            t = t * t * (3 - 2 * t)
    return _lerp(a, b, t)


def match_indices(n_src, n, mode="hold"):
    """Source positions (floats; integers except for 'retime') for each of n output frames."""
    j = np.arange(n)
    L = n_src
    if mode == "hold":
        return np.minimum(j, L - 1).astype(float)
    if mode == "loop":
        return (j % L).astype(float)
    if mode == "pingpong":
        if L == 1:
            return np.zeros(n)
        k = j % (2 * L - 2)
        return np.where(k < L, k, 2 * L - 2 - k).astype(float)
    if mode == "retime":
        return j * (L - 1) / (n - 1) if n > 1 else np.zeros(n)
    raise ValueError(f"match_length mode {mode!r}: expected hold | loop | pingpong | retime")


def match_length(seq, n, mode="hold"):
    """Generator of n frames from `seq` (list / array / FrameSource): 'hold' last frame, 'loop', 'pingpong', or
    'retime' (stretch/squeeze the whole clip to n frames, blending neighbours)."""
    pos = match_indices(len(seq), n, mode)
    for p in pos:
        i0 = int(math.floor(p))
        fr = float(p - i0)
        if fr < 1e-6 or i0 + 1 >= len(seq):
            yield seq[i0]
        else:
            yield crossfade(seq[i0], seq[i0 + 1], fr)


@api
def upscale_matte(matte, size, guide=None, radius=None, eps=2e-4):
    """Upscale a low-res matte to size=(w,h) (e.g. 576x1024 -> 1088x1920). With a full-res `guide` (footage frame or
    luma) a guided filter (He et al.) snaps the soft matte to real edges in the footage without thresholding it."""
    tw, th = int(size[0]), int(size[1])
    m0 = _gray(matte)
    up = B.resize(m0, th, tw, "cubic" if (th * tw) > m0.shape[0] * m0.shape[1] else "area")
    up = _clip(up)
    if guide is None:
        return up
    g = _fit(guide, th, tw) if guide.shape[:2] != (th, tw) else guide
    g = _luma(g) if g.ndim == 3 and g.shape[-1] == 3 else _gray(g)
    scale = max(tw / m0.shape[1], th / m0.shape[0])
    r = int(radius if radius is not None else max(2, round(2 * scale)))
    mean = lambda x: B.box(x, r)
    mI, mp = mean(g), mean(up)
    cov = mean(g * up) - mI * mp
    var = mean(g * g) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _clip(mean(a) * g + mean(b))


# ======================================================================================================== frame IO

_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _probe_video(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,nb_frames,r_frame_rate,duration:format=duration", "-of", "json", str(path)]
    j = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    s = j["streams"][0]
    num, den = (s.get("r_frame_rate") or "24/1").split("/")
    fps = float(num) / float(den or 1)
    n = s.get("nb_frames")
    if n in (None, "N/A", "0"):
        dur = s.get("duration")
        dur = float(dur) if dur not in (None, "N/A") else float(j["format"]["duration"])
        n = int(round(dur * fps))
    return int(s["width"]), int(s["height"]), int(n), fps


def audio_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    return float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


def imread(path, kind="rgb"):
    """Read an image -> float32. kind: 'rgb' (HxWx3) | 'gray' (HxW; RGB -> luma) | 'alpha' (alpha channel if present,
    else gray) | 'raw' (as stored). 8/16-bit PNG normalised to [0,1]. .npy files load as-is (e.g. flow HxWx2)."""
    p = str(path)
    if p.lower().endswith(".npy"):
        return np.load(p).astype(np.float32)
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"cannot read image {p}")
    scale = 65535.0 if im.dtype == np.uint16 else (255.0 if im.dtype == np.uint8 else 1.0)
    im = im.astype(np.float32) / scale
    if im.ndim == 3 and im.shape[2] >= 3:
        rgb = im[..., :3][..., ::-1]
        alpha = im[..., 3] if im.shape[2] == 4 else None
    else:
        g = im if im.ndim == 2 else im[..., 0]
        rgb, alpha = np.stack([g, g, g], -1), None
    if kind == "rgb":
        return np.ascontiguousarray(rgb)
    if kind == "gray":
        return np.ascontiguousarray(rgb[..., 0] if (im.ndim == 2 or im.shape[2] == 1) else
                                    rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722)
    if kind == "alpha":
        if alpha is not None:
            return np.ascontiguousarray(alpha)
        return imread(path, "gray")
    if kind == "raw":
        return im
    raise ValueError(kind)


def _to_u8(f):
    return np.clip(np.rint(np.asarray(f, np.float32) * 255.0), 0, 255).astype(np.uint8)


def imwrite(path, arr, bit16=False):
    """Write float [0,1] HxW / HxWx3 as PNG (RGB order)."""
    a = np.asarray(arr, np.float32)
    if bit16:
        q = np.clip(np.rint(a * 65535.0), 0, 65535).astype(np.uint16)
    else:
        q = _to_u8(a)
    if q.ndim == 3 and q.shape[2] == 3:
        q = q[..., ::-1]
    ok, buf = cv2.imencode(".png", q, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"png encode failed {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


class FrameSource:
    """Lazy frame sequence over an mp4 or a directory of images (len + random access + iteration).
    Never loads the whole clip; mp4 is decoded through an ffmpeg pipe (sequential reads are fast, random access
    re-seeks - extract_frames() to PNGs first for heavy random access such as pingpong).
    size=(w,h) cover-crops every frame. kind as in imread (video: 'rgb' or 'gray')."""

    def __init__(self, path, kind="rgb", size=None):
        self.path, self.kind, self.size = Path(path), kind, (tuple(size) if size else None)
        self._cache = OrderedDict()
        self._proc = None
        self._pos = -1
        if self.path.is_dir():
            self.files = sorted((p for p in self.path.iterdir() if p.suffix.lower() in _IMG_EXT | {".npy"}),
                                key=lambda p: _natural_key(p.name))
            if not self.files:
                raise FileNotFoundError(f"no frames in {self.path}")
            self.video, self.n, self.fps = False, len(self.files), None
        elif self.path.suffix.lower() in _VID_EXT:
            self.w0, self.h0, self.n, self.fps = _probe_video(self.path)
            self.video, self.files = True, None
        elif self.path.suffix.lower() in _IMG_EXT | {".npy"}:  # a single still -> one-frame source
            self.files, self.video, self.n, self.fps = [self.path], False, 1, None
        else:
            raise FileNotFoundError(f"unsupported frame source {self.path}")

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def __del__(self):
        self.close()

    def _fit(self, f):
        if self.size is not None and f.shape[:2] != (self.size[1], self.size[0]):
            f = fit_cover(f, self.size)
        return f

    def _open(self, i):
        self.close()
        ss = max(0.0, (i - 0.5) / self.fps) if i > 0 else 0.0
        cmd = ["ffmpeg", "-v", "error", "-nostdin"] + (["-ss", f"{ss:.6f}"] if i > 0 else []) + \
              ["-i", str(self.path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=self.w0 * self.h0 * 3 * 2)
        self._pos = i

    def _next_video(self):
        raw = self._proc.stdout.read(self.w0 * self.h0 * 3)
        if len(raw) < self.w0 * self.h0 * 3:  # EOF early: hold the last frame
            last = next(reversed(self._cache.values()), None)
            if last is None:
                raise IOError(f"no frames decoded from {self.path}")
            self._pos += 1
            return last[1]
        self._pos += 1
        f = np.frombuffer(raw, np.uint8).reshape(self.h0, self.w0, 3).astype(np.float32) / 255.0
        if self.kind in ("gray", "alpha"):
            f = f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
        return f

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        if i in self._cache:
            return self._cache[i][0]
        if self.video:
            if self._proc is None or i != self._pos:
                if self._proc is not None and self._pos < i <= self._pos + 8:  # short forward skip: just decode on
                    while self._pos < i:
                        self._next_video()
                else:
                    self._open(i)
            raw_f = self._next_video()
            f = self._fit(raw_f)
        else:
            f = self._fit(imread(self.files[i], self.kind))
        self._cache[i] = (f, f)
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return f


def read_frames(path, kind="rgb", size=None, start=0, stop=None, step=1):
    """Generator of float32 frames from an mp4 or image directory, never the whole clip in RAM."""
    src = FrameSource(path, kind, size)
    try:
        for i in range(start, src.n if stop is None else min(stop, src.n), step):
            yield src[i]
    finally:
        src.close()


def frame_count(path):
    return len(FrameSource(path))


def write_png_sequence(frames, out_dir, pattern="%05d.png", bit16=False, start=0, workers=2):
    """Write an iterable of frames as PNGs (threaded encode). Returns the number written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = []
        for i, f in enumerate(frames, start):
            futs.append(ex.submit(imwrite, out / (pattern % i), f, bit16))
            n += 1
            if len(futs) > workers * 4:
                futs.pop(0).result()
        for f in futs:
            f.result()
    return n


def extract_frames(mp4, out_dir, pattern="%05d.png"):
    """mp4 -> PNG directory with ffmpeg (fast; use before random access)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), str(Path(out_dir) / pattern)], check=True)
    return len(list(Path(out_dir).glob("*.png")))


def encode_mp4(frames, out_mp4, fps=24, audio=None, crf=16, preset="medium", audio_bitrate="320k", vcodec=None, vopts=None):
    """Encode frames (iterable of float arrays, or a PNG directory) to H.264 yuv420p. With `audio`, AAC 320k is
    muxed and the result lasts exactly the audio length: video is trimmed, or its last frame held, to ceil(dur*fps)
    frames and the output is cut at the audio duration. Returns {'path','frames','seconds'}.
    vcodec (default libx264) / vopts (list of ffmpeg args placed after -c:v, default ["-preset", preset, "-crf", crf]) pick another
    encoder, e.g. vcodec="h264_nvenc", vopts=["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"]."""
    if isinstance(frames, (str, Path)):
        frames = read_frames(frames)
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    adur = audio_duration(audio) if audio else None
    limit = int(math.ceil(adur * fps - 1e-6)) if adur else None
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", str(fps), "-i", "pipe:0"]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", vcodec or "libx264"]
    cmd += list(vopts) if vopts is not None else ["-preset", preset, "-crf", str(crf)]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(fps)]
    if audio:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", audio_bitrate, "-t", f"{adur:.6f}"]
    cmd += ["-movflags", "+faststart", str(out_mp4)]
    err = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
    n, last = 0, first
    try:
        cur = first
        while True:
            p.stdin.write(np.ascontiguousarray(_to_u8(cur)[..., :3]).tobytes())
            n += 1
            last = cur
            if limit is not None and n >= limit:
                break
            try:
                cur = next(it)
            except StopIteration:
                if limit is None:
                    break
                cur = last  # hold last frame until audio ends
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        rc = p.wait()
    if rc != 0:
        err.seek(0)
        raise RuntimeError("ffmpeg failed: " + err.read().decode(errors="replace")[-800:])
    return {"path": str(out_mp4), "frames": n, "seconds": adur if adur else n / fps}


# ======================================================================================================== recipes

def _ease(u, kind):
    return u * u * (3 - 2 * u) if kind == "smooth" else u


def _anim(spec, u):
    """{'anim': [[u0,v0],[u1,v1],...], 'ease': 'smooth'|'linear'}; u = frame/(n-1). Values may be scalars or lists."""
    ks = spec["anim"]
    ease = spec.get("ease", "linear")
    if u <= ks[0][0]:
        return ks[0][1]
    for (u0, v0), (u1, v1) in zip(ks[:-1], ks[1:]):
        if u <= u1:
            t = _ease((u - u0) / max(u1 - u0, 1e-9), ease)
            a, b = np.asarray(v0, float), np.asarray(v1, float)
            r = a + (b - a) * t
            return r.tolist() if r.ndim else float(r)
    return ks[-1][1]


def _timeline(spec, t):
    """{'timeline': [[sec, v], ...], 'step': False, 'ease': 'linear'|'smooth'}: piece-time keyframes (t in seconds). step=True holds
    the value of the last key at or before t (colours, booleans); otherwise like _anim (scalars or lists, linear / smooth)."""
    ks = spec["timeline"]
    if spec.get("step"):
        v = ks[0][1]
        for k, val in ks:
            if t + 1e-9 >= k:
                v = val
        return v
    return _anim({"anim": ks, "ease": spec.get("ease", "linear")}, t)


_STATEFUL = {"trace_edges"}


def _ops_registry():
    reg = _core_ops()
    from tools.fusion.matte_treatments import OPS as _matte_ops     # v005: mask_gate, mask_mul, split_region, matte_reach, matte_ring
    reg.update(_matte_ops)
    from tools.fusion.strip_resnap import OPS as _strip_ops           # block strips_persist_across_cuts: rigid_pose (the type follows the posed strips)
    reg.update(_strip_ops)
    from tools.fusion.edge_run import OPS as _edge_ops                # block edge_run_small_type: edge_run_type (small type along a tracked footage edge)
    reg.update(_edge_ops)
    from tools.fusion.plate_pan import OPS as _pan_ops                # v006: plate_pan (the plate travels with the type camera of a pan transition)
    reg.update(_pan_ops)
    from tools.fusion.legible_color import OPS as _lc_ops             # v005 round 3: legible_color (type colour from the measured background), legible_support
    reg.update(_lc_ops)
    from tools.fusion.window_cut import OPS as _wc_ops                # 2026-10-28: window_cut (letters as windows between two shots: window_other / window_cut / window_swap)
    reg.update(_wc_ops)
    from tools.fusion.hold_life import OPS as _hl_ops                 # 2026-10-30: hold_life (letter flicker / slice glitch / pulse on the drawn type layer during a hold)
    reg.update(_hl_ops)
    from tools.fusion.contour_sweep import OPS as _cs_ops              # 2026-10-30: contour_sweep (contour lines draw on / off along the forms, swept in the transition's direction)
    reg.update(_cs_ops)
    from tools.fusion.word_bridge import OPS as _wb_ops                # 2026-11-01: word_bridge (letters A -> [contour] -> transition pattern -> letters B, one continuous line set)
    reg.update(_wb_ops)
    try:                                                              # 2026-10-30 exploration ops (grounds / masks / brush outlines, core_check/explore_masks): own names only, no default uses them
        from tools.fusion.explore_ops import OPS as _ex_ops
        reg.update({k: v for k, v in _ex_ops.items() if k not in reg})
    except ImportError:
        pass
    try:                                                              # 2026-10-30 round-2 exploration ops (plate_diff_key, mask_zoom, ...): own names only
        from tools.fusion.explore_ops2 import OPS as _ex2_ops
        reg.update({k: v for k, v in _ex2_ops.items() if k not in reg})
    except ImportError:
        pass
    try:                                                              # 2026-10-30 the hearted explore looks as named effects (tools/fusion/explore_effects.py): own op names only
        from tools.fusion.explore_effects import OPS as _ee_ops
        reg.update({k: v for k, v in _ee_ops.items() if k not in reg})
    except ImportError:
        pass
    try:                                                              # v005: the plate as depth cards under the piece's type camera (needs torch)
        from mograph.render.multiplane import compose_op as _multiplane
        reg["multiplane"] = _multiplane
    except ImportError:
        pass
    return reg


def _core_ops():
    return {
        "sandwich": sandwich, "depth_occlude": depth_occlude, "knockout": knockout, "tint_type": tint_type,
        "edges_in_type": edges_in_type, "type_from_edges": type_from_edges, "refract": refract,
        "screen": screen, "multiply": multiply, "overlay": overlay, "soft_light": soft_light, "over": over,
        "add_glow": add_glow, "luminosity_key": luminosity_key, "grade": grade, "halation": halation,
        "grain": grain, "vignette": vignette, "fit_cover": fit_cover, "upscale_matte": upscale_matte,
        "crossfade": crossfade, "blur": _op_blur, "fill": _op_fill, "luma": luma, "fill_holes": fill_holes,
        "legible": legible, "depth_cover": depth_cover,
        "translate": _op_translate, "flow_wake": flow_wake, "rgba_part": rgba_part, "contour_lines": contour_lines,
        "layout_shapes": layout_shapes, "type_face": type_face, "hold_net": hold_net,
    }


def _op_blur(img, sigma=4.0):
    return _out(B.blur(_in(img), sigma))


def _op_fill(like, color=(0.0, 0.0, 0.0)):
    like = _in(like)
    return _out(_color(color, like))


def _op_translate(img, dx=0.0, dy=0.0):
    """(P08, additive) image content moved by (dx, dy) px (+dy = down), reflected borders: out(x, y) = img(x - dx, y - dy)."""
    img = _in(img)
    h, w = img.shape[:2]
    return _out(B.warp(img, B.asarray(np.full((h, w), -float(dx), np.float32)), B.asarray(np.full((h, w), -float(dy), np.float32))))


def fill_holes(mask, amount=1.0, level=1, thresh=0.5, soften=1.0, close=0):
    """Close the enclosed holes of an outline-only matte (e.g. Rock 3D, whose contours overlap so the even-odd fill leaves a ring).
    Holes at nesting depth `level` (1 = holes of the outermost white shapes) are filled; deeper holes (counters inside those) stay
    open. amount 0 returns the mask untouched (and costs nothing), 0..1 blends. Numpy / cv2 op (compose workers run numpy)."""
    m = _out(_in(mask)) if _is_arr(mask) else np.asarray(mask, np.float32)
    m = np.asarray(m, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    amount = float(amount)
    if amount <= 0.0:
        return m
    b = (m > thresh).astype(np.uint8)
    if close > 0:                                   # bridge small gaps in the outline so near-closed loops fill too
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(close) + 1, 2 * int(close) + 1)))
    cnts, hier = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None or not len(cnts):
        return m
    hier = hier[0]
    depth = []
    for i in range(len(cnts)):
        d, j = 0, hier[i][3]
        while j >= 0:
            d, j = d + 1, hier[j][3]
        depth.append(d)
    canvas = np.zeros_like(b)
    for i in sorted(range(len(cnts)), key=lambda k: depth[k]):
        d = depth[i]
        val = 255 if (d % 2 == 0 or d <= level) else 0
        cv2.drawContours(canvas, cnts, i, val, thickness=-1)
    f = canvas.astype(np.float32) / 255.0
    if soften > 0:
        f = cv2.GaussianBlur(f, (0, 0), soften)
    return np.clip(np.maximum(m, f * amount), 0.0, 1.0)


_NAME_GRAY = re.compile(r"(matte|depth|alpha|_a$|^a$|line|mask|edges)")


def _guess_kind(name):
    return "gray" if _NAME_GRAY.search(name) else "rgb"


def _refs(obj, acc):
    if isinstance(obj, str) and obj.startswith("$"):
        acc.add(obj[1:])
    elif isinstance(obj, dict):
        for v in obj.values():
            _refs(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _refs(v, acc)


class _ConstSource:
    def __init__(self, arr):
        self.arr, self.n = arr, 1

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.arr

    def close(self):
        pass


class _NpyStack:
    """one .npy holding a whole clip's per-frame arrays (e.g. <shot>_flow.npy, T x h x w x 2) as a frame source (memory-mapped)."""

    def __init__(self, path):
        self.arr = np.load(path, mmap_mode="r")
        self.n = int(self.arr.shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.ascontiguousarray(self.arr[min(max(int(i), 0), self.n - 1)], dtype=np.float32)

    def close(self):
        self.arr = None


def _open_input(name, spec, size, base):
    if isinstance(spec, (list, tuple)):  # constant colour
        v = np.asarray(spec, np.float32)
        return _ConstSource(np.broadcast_to(v[:3], (size[1], size[0], 3)).copy()), "hold"
    if isinstance(spec, str):
        spec = {"path": spec}
    p = Path(spec["path"])
    p = p if p.is_absolute() else Path(base) / p
    kind = spec.get("kind") or _guess_kind(name)
    if p.suffix.lower() == ".npy" and p.is_file() and np.load(p, mmap_mode="r").ndim == 4:
        return _NpyStack(p), spec.get("mode")
    fit = None if kind == "npy" else size
    if p.suffix.lower() == ".npy" or (p.is_dir() and any(p.glob("*.npy")) and not any(p.glob("*.png"))):
        kind, fit = "raw", None
    return FrameSource(p, kind, fit), spec.get("mode")


def compose_shot(recipe, out_dir, base_dir=None, audio=None, write_png=True, write_mp4=True, progress=None):
    """Render one shot from a JSON-able recipe to out_dir/frames/%05d.png and out_dir/<name>.mp4 (see module doc /
    the report for the schema). Returns {'frames_dir','mp4','n','fps','size','sec_per_frame'}."""
    base = Path(base_dir or os.getcwd())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = recipe.get("name", "shot")
    size = tuple(recipe.get("size", (1088, 1920)))
    fps = float(recipe.get("fps", 24))
    default_mode = recipe.get("length_mode", "hold")
    ops = recipe.get("ops", [])
    out_buf = recipe.get("output", "image")
    reg = _ops_registry()
    for op in ops:
        if op["op"] not in reg and op["op"] not in _STATEFUL:
            raise ValueError(f"unknown op {op['op']!r}; available: {sorted(set(reg) | _STATEFUL)}")

    used = set()
    _refs([o.get("args", {}) for o in ops], used)
    inputs = recipe.get("inputs", {})
    srcs, modes = {}, {}
    for nm, spec in inputs.items():
        if nm in used or nm == "footage" or nm == recipe.get("length_from"):
            srcs[nm], modes[nm] = _open_input(nm, spec, size, base)
    missing = [u for u in used if u not in inputs and u not in {"frame_idx", "t", "u", "n", "palette"}
               and u not in {o.get("out", "image") for o in ops}]
    if missing:
        raise ValueError(f"recipe references undefined buffers: {missing}")

    adur = audio_duration(audio) if audio else None
    if recipe.get("frames"):
        n = int(recipe["frames"])
    elif recipe.get("seconds"):
        n = int(round(float(recipe["seconds"]) * fps))
    elif adur and recipe.get("trim_to_audio"):
        n = int(math.ceil(adur * fps))
    else:
        ref = recipe.get("length_from", "footage")
        n = len(srcs[ref])

    palette = None
    if recipe.get("palette"):
        pr = recipe["palette"]
        s = srcs[pr.get("source", "footage")]
        palette = palette_from_footage(s, k=int(pr.get("k", 5)))

    streams = {nm: iter(match_length(s, n, modes[nm] or default_mode)) for nm, s in srcs.items() if nm in used}
    states = {i: EdgeTracer() for i, o in enumerate(ops) if o["op"] in _STATEFUL}
    frames_dir = out_dir / "frames"
    mp4 = out_dir / f"{name}.mp4"
    stat = {"t": 0.0}

    def frames():
        for i in range(n):
            t0 = time.perf_counter()
            u = i / max(n - 1, 1)
            ctx = {nm: next(st) for nm, st in streams.items()}
            ctx.update(frame_idx=i, t=i / fps, u=u, n=n)
            if palette is not None:
                ctx["palette"] = palette

            def res(v):
                if isinstance(v, str) and v.startswith("$"):
                    return ctx[v[1:]]
                if isinstance(v, dict) and "anim" in v:
                    return _anim(v, u)
                if isinstance(v, list):
                    return [res(x) for x in v]
                return v

            for oi, op in enumerate(ops):
                args = {k: res(v) for k, v in op.get("args", {}).items()}
                if op["op"] == "trace_edges":
                    fo = args.pop("footage")
                    r = states[oi].step(fo, **args)
                else:
                    r = reg[op["op"]](**args)
                ctx[op.get("out", "image")] = r
            img = ctx[out_buf]
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            stat["t"] += time.perf_counter() - t0
            if progress:
                progress(i, n)
            yield img

    tee_ex = ThreadPoolExecutor(2) if write_png else None
    pending = []

    def tee():
        for i, img in enumerate(frames()):
            if write_png:
                pending.append(tee_ex.submit(imwrite, frames_dir / ("%05d.png" % i), img))
                if len(pending) > 8:
                    pending.pop(0).result()
            yield img

    try:
        if write_mp4:
            encode_mp4(tee(), mp4, fps=fps, audio=audio if recipe.get("mux_audio") else None,
                       crf=int(recipe.get("crf", 16)))
        else:
            for _ in tee():
                pass
        for f in pending:
            f.result()
    finally:
        if tee_ex:
            tee_ex.shutdown()
        for s in srcs.values():
            s.close()
    return {"frames_dir": str(frames_dir) if write_png else None, "mp4": str(mp4) if write_mp4 else None,
            "n": n, "fps": fps, "size": size, "sec_per_frame": stat["t"] / max(n, 1)}


def assemble(shots, audio, out_mp4, crossfades=0.0, fps=24, size=None, crf=16, base_dir=None, sources=None, **enc):
    """Join shots on cut times and mux audio. shots: list of paths (mp4 / PNG dir) or dicts
    {'src', 'cut': seconds where the shot begins in the output (default: right after the previous one ends),
     'mode': 'hold'|'loop' (what a shot does if it runs out, default hold)}.
    crossfades: seconds (scalar, or list of len(shots)-1) centred on each cut: the fade is 50% at the cut time, the
    incoming shot starts playing d/2 before it. The output lasts the audio length (or the end of the last shot).
    sources: optional list (one per shot, None = open the shot's src) of in-memory frame sources (len, [i] -> float HxWx3, close()).
    **enc: forwarded to encode_mp4 (preset, vcodec, vopts, audio_bitrate)."""
    base = Path(base_dir or os.getcwd())
    specs = [{"src": s} if isinstance(s, (str, Path)) else dict(s) for s in shots]
    cf = list(crossfades) if isinstance(crossfades, (list, tuple)) else [float(crossfades)] * max(len(specs) - 1, 0)
    srcs = []
    for k, s in enumerate(specs):
        if sources is not None and sources[k] is not None:
            srcs.append(sources[k])
            continue
        p = Path(s["src"])
        srcs.append(FrameSource(p if p.is_absolute() else base / p, "rgb", size))
    if size is None:
        f0 = srcs[0][0]
        size = (f0.shape[1], f0.shape[0])
        for s in srcs:
            s.size = size
    lens = [len(s) / fps for s in srcs]
    cuts, start = [], 0.0
    for k, s in enumerate(specs):
        c = float(s["cut"]) if s.get("cut") is not None else (0.0 if k == 0 else cuts[-1] + lens[k - 1])
        cuts.append(c if k else 0.0)
    d = [0.0] + cf
    starts = [cuts[k] - d[k] / 2.0 for k in range(len(specs))]
    adur = audio_duration(audio) if audio else None
    total = adur if adur else starts[-1] + lens[-1]
    nframes = int(math.ceil(total * fps - 1e-6))

    def local(k, t):
        i = int(round((t - starts[k]) * fps))
        L = len(srcs[k])
        if specs[k].get("mode") == "loop":
            i %= L
        return srcs[k][min(max(i, 0), L - 1)]

    def gen():
        for f in range(nframes):
            t = f / fps
            k = max(j for j in range(len(specs)) if starts[j] <= t + 1e-9)
            img = local(k, t)
            if k > 0 and d[k] > 0 and t < starts[k] + d[k]:
                w = (t - starts[k]) / d[k]
                img = crossfade(local(k - 1, t), img, w, smooth=True)
            yield img

    try:
        return encode_mp4(gen(), out_mp4, fps=fps, audio=audio, crf=crf, **enc)
    finally:
        for s in srcs:
            s.close()


# ======================================================================================================== CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shot", help="render a recipe json")
    s.add_argument("recipe")
    s.add_argument("out_dir")
    s.add_argument("--audio")
    s.add_argument("--no-png", action="store_true")
    a = sub.add_parser("assemble", help="join shots: json {shots:[...], audio, out, crossfades}")
    a.add_argument("spec")
    sub.add_parser("info", help="print backend")
    sub.add_parser("selfcheck", help="synthetic demo + contact sheet")
    args = ap.parse_args(argv)
    if args.cmd == "info":
        print(backend_name())
    elif args.cmd == "shot":
        rp = Path(args.recipe).resolve()
        recipe = json.loads(rp.read_text(encoding="utf-8"))
        r = compose_shot(recipe, args.out_dir, base_dir=rp.parent, audio=args.audio, write_png=not args.no_png,
                         progress=lambda i, n: print(f"\rframe {i + 1}/{n}", end="", file=sys.stderr))
        print(json.dumps(r, indent=2))
    elif args.cmd == "assemble":
        sp = Path(args.spec).resolve()
        spec = json.loads(sp.read_text(encoding="utf-8"))
        print(json.dumps(assemble(spec["shots"], spec.get("audio"), spec["out"], spec.get("crossfades", 0.0),
                                  fps=spec.get("fps", 24), base_dir=sp.parent), indent=2))
    elif args.cmd == "selfcheck":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import selfcheck_compose
        selfcheck_compose.main()


if __name__ == "__main__":
    main()
