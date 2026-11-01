#!/usr/bin/env python
"""Math test D: silk wind (10 s seamless loop, 4:5, no words).

A volume of a few thousand hairline strands seen by a fixed pinhole camera. Nothing is carried
between frames: ``Silk.frame(t)`` is a pure function of t, and frame(0) == frame(T) exactly.

Strand i, in object space (unit cloud):
  u_i(t)  base point riding a gentle wind along the axis of a cylinder tile (one tile length per
          loop, so it wraps exactly once; the wrap happens on the outer shell where alpha is 0);
  D(u,t)  3D curl of a domain-warped fBm vector potential (analytic chain-rule gradients, so the
          flow is exactly divergence-free and reads as air, not jitter);
  E(u,t)  extra high-frequency 3D curl displacement weighted by smoothstep on the radius, so the
          outer shell flutters while the core flows calmly;
  o       = noisy ellipsoid * m(u + D + E); m() packs the core denser than the shell.
World: w(t) = Tr(t) + R(t) (s(t) o). The breath s(t) swells the cloud from a distant dense mass
until it surrounds the camera; R(t) is a slow tumble about a drifting tilted axis; Tr(t) a gentle
drift. Every noise is a sum of random plane waves in (x, y, z, cos 2pi t/T, sin 2pi t/T): time is
sampled on a circle, so the loop is exact.

Trails are spatially honest: sample k is the head's world position at t - k dt *including*
M(t - k dt), projected through the same camera, so expansion streaks outward past the viewer,
rotation sweeps arcs and retreat converges. Trail duration is per strand (long-tailed).

Drawing: additive bilinear splatting into a float canvas (2x supersampled), depth classes with
increasing blur (near = wider, dimmer, defocused), light ~ 1/z, exponential depth fog, a near-plane
fade band, tone map 1 - exp(-x), area downsample. A few small dark-rimmed bubble heads (the A look)
and <=1% accent strands in #e2582a. Sound: tools/mathtype/chords.py (looped), muxed by --mux.

Outputs never overwrite: every run gets a generation number gNN (max existing + 1).
    python tools/mathtype/silkwind.py --draft             # 360x450  -> v003/local/D_silkwind_gNN_draft.mp4 + stills
    python tools/mathtype/silkwind.py --final [--gen NN]  # 1080x1350 -> D_silkwind_gNN_final.mp4 (latest gen)
    python tools/mathtype/silkwind.py --mux [--gen NN]    # + chords.py loop -> D_silkwind_gNN_final_chords.mp4
    python tools/mathtype/silkwind.py --still 0 2.5 5     # scratch PNG stills to v003/review/ (--big: 1080)
    python tools/mathtype/silkwind.py --seam              # loop-seam diffs

Audio-reactive (v004, 12 s, not looping; test name D_silkwind_<mode>; AudioMod holds the mapping):
    python tools/mathtype/silkwind.py --audio x.wav --events x.json --mode beaty|ethereal --preview \
        [--parent ..] [--trigger ..] [--change ..]     # preview + _preview_audio.mp4 + stills + QC strip/sync plot
    python tools/mathtype/silkwind.py --audio x.wav --events x.json --mode beaty --final [--gen NN]   # + _final_audio.mp4

Matte for the sea-shot wipe (v005, test D_wipe_<mode>; composited by tools/mathtype/matte_wipe.py):
    python tools/mathtype/silkwind.py --audio x.wav --events x.json --mode beaty --matte [--preview] \
        --wipe-at 9.6 --wipe-dur 1.2 [--wipe-grow 30 --wipe-far .. --wipe-light .. --wipe-base 3]

GPU: any of the above + --backend torch --device cuda [--frame-batch 4] (tools/mathtype/xp.py; README.md).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xp  # noqa: E402  (array backend: numpy reference / torch cpu|cuda)

REPO =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(REPO, "11_mathtype", "v003")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

DW, DH = 1080, 1350                       # design space (px)
FPS, T = 25, 10.0
NFR = int(round(FPS * T))                 # 250
BG = np.array([5, 5, 5], float)
INK = np.array([0xF3, 0xEA, 0xDB], float)
ACCENT = np.array([0xE2, 0x58, 0x2A], float)
EDGE = (16, 15, 14)                       # bubble rim

SEED = 11
N = 7000                                  # strands
K = 32                                    # trail samples per strand
TAU_MIN, TAU_MAX = 0.08, 1.4              # trail duration (s), long-tailed between these
ACCENT_FRAC = 0.006
HEAD_FRAC = 0.10

# camera: pinhole at the origin looking down +z
FOCAL = 1.2 * DH
CX, CY = DW / 2, DH / 2
Z_NEAR, Z_FADE = 0.22, 0.85               # strands fade out over [Z_NEAR, Z_FADE]
Z_REF = 6.5                               # retreat distance: hairline width / unit light here
FOG_Z, FOG_L = 6.5, 6.0                   # fog: exp(-(z - FOG_Z) / FOG_L) beyond FOG_Z
LINE_W = 0.5 * Z_REF / FOCAL              # world-space strand width (0.5 design px at Z_REF)

# volume transform
S_MIN, S_MAX = 1.25, 5.4                  # breath scale (world units per object unit)
Z_FAR, Z_CLOSE = 6.5, 4.2                 # cloud centre distance at rest / at full breath
TILT_SWING = 0.50                         # tumble amplitude (rad): ~57 deg peak to peak

# object-space flow
TILE = 2.0                                # along-wind tile length; the wind crosses it once per loop
WIND = np.array([1.0, -0.42, 0.30])
WIND /= np.linalg.norm(WIND)
FLOW_F, FLOW_AMP, WARP, FLOW_GAIN = 0.9, 0.14, 0.45, 0.35
RUF_F, RUF_AMP = 3.6, 0.05
EXPOSURE = 3.5
BREATH_GLOW = 0.6                         # light x (s/S_MIN)^k: the surrounding cloud keeps its sheen
WIND_EASE, WIND_PHASE = 0.75, -3.4        # wind slows to 25% at full breath, 175% when retreated

# --matte (v005, test D_wipe_<mode>): greyscale strand matte for tools/mathtype/matte_wipe.py. Strand
# luminance only (no accent, no bubble heads), fog kept. --wipe-at/--wipe-dur: around the wipe moment the
# cloud swells round the camera and strands grow wide and soft (near ones most), so coverage swells from
# threads to near-full white with organic gaps, then holds. All closed-form in t (wipe_env).
MATTE = False
WIPE_AT, WIPE_DUR = None, 1.6
WIPE_PRE = 0.30                           # share of the envelope built over [at - dur, at] (the approach)
WIPE_GROW = 30.0                          # width multiplier for the nearest strands at full wipe
WIPE_BASE = 1.0                           # resting width multiplier for near strands (soft ghost threads)
WIPE_ZN = 2.6                             # strands nearer than this get full growth; (WIPE_ZN / z)^1.5 beyond
WIPE_FAR = 0.25                           # floor of that near weight at full wipe (far strands thicken too)
WIPE_SOFT = 0.45                          # Gaussian sigma = WIPE_SOFT x grown width (design px)
WIPE_LIGHT = 1.0                          # light x grow^WIPE_LIGHT (keeps a widened stroke's peak up)
WIPE_BREATH = 0.85                        # breath floor at full wipe: the cloud surrounds the camera and holds
WIDE_BLUR = (6.0, 12.0, 24.0, 48.0, 96.0)  # extra blur classes (design px sigma) for grown strands

# --backend numpy|torch --device cpu|cuda (tools/mathtype/xp.py). numpy is the reference; torch runs the
# same closed-form geometry in float32 and rasterises with a scatter-add splat + separable blur, several
# frames per batch (--frame-batch).
BACKEND = xp.Backend()
FRAME_BATCH = 4


def smoothstep(a, b, x):
    x = xp.clip((x - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def wipe_env(t):
    """0 -> 1 wipe envelope: WIPE_PRE builds over the approach [at - dur, at], the rest sweeps in over
    [at - 0.15 dur, at + 0.5 dur], so the wipe moment lands on WIPE_AT. 0 when no wipe is set."""
    t = xp.asfloat(t)
    if WIPE_AT is None:
        return xp.zeros_like(t)
    pre = smoothstep(WIPE_AT - WIPE_DUR, WIPE_AT, t)
    x = xp.clip((t - (WIPE_AT - 0.15 * WIPE_DUR)) / (0.65 * WIPE_DUR), 0.0, 1.0)
    return WIPE_PRE * pre + (1 - WIPE_PRE) * x * x * x * (x * (6 * x - 15) + 10)


# ---------------------------------------------------------------------------------------------- noise
class Bank:
    """F independent band-limited noises over (x, y, z) and the time circle (cos th, sin th).
    Each is a sum of random plane waves, octaves at doubling frequency with fresh random
    directions (so the domain is rotated per octave), gain `gain`. `rate` is the time-circle
    radius, i.e. how far the field evolves over one loop. One matmul serves all F fields;
    returns values (P, F) and analytic spatial gradients (P, F, 3)."""

    def __init__(self, rng, fields, f0, octaves=3, modes=6, rate=1.0, gain=0.5):
        rows = []
        for _ in range(fields):
            for o in range(octaves):
                d = rng.normal(size=(modes, 3))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                k = d * (f0 * 2 ** o * rng.uniform(0.8, 1.2, modes))[:, None]
                w = rng.normal(0.0, 1.0, (modes, 2)) * rate * (1 + 0.5 * o)
                rows.append(np.column_stack([k, w, rng.random(modes) * 2 * np.pi, np.full(modes, gain ** o)]))
        m = np.concatenate(rows)
        self.F, self.m = fields, octaves * modes
        self.M = m[:, :6].T.copy()                                  # (6, F*m)
        self.M32 = self.M.astype(np.float32)
        a = m[:, 6].reshape(fields, self.m)
        a = a / np.sqrt(0.5 * np.sum(a ** 2, 1, keepdims=True))     # unit variance per field
        self.a = a.ravel().astype(np.float32)
        self.ak = (self.a[:, None] * m[:, :3]).astype(np.float32)   # (F*m, 3)
        self._dev = {}

    def __call__(self, X):
        """X: (P, 6) rows (x, y, z, ct, st, 1). Torch: float32 values / gradients on X's device."""
        if xp.is_t(X):
            key = str(X.device)
            if key not in self._dev:
                self._dev[key] = (xp.const(X, self.M32), xp.const(X, self.a).view(self.F, self.m),
                                  xp.const(X, self.ak).view(self.F, self.m, 3))
            M, a, ak = self._dev[key]
            arg = X @ M
            s, c = xp.sin(arg), xp.cos(arg)
            P = X.shape[0]
            val = (s.view(P, self.F, self.m) * a).sum(-1)
            grad = xp.einsum("pfm,fmj->pfj", c.view(P, self.F, self.m), ak)
            return val, grad
        arg = X.astype(np.float32) @ self.M32
        s, c = np.sin(arg), np.cos(arg)
        P = X.shape[0]
        val = (s * self.a).reshape(P, self.F, self.m).sum(-1)
        grad = np.stack([(c * self.ak[:, j]).reshape(P, self.F, self.m).sum(-1) for j in range(3)], -1)
        return val.astype(np.float64), grad.astype(np.float64)


def curl(g):
    """curl of a vector potential from its gradients g (P, 3 fields, 3 dims)."""
    return xp.stack([g[:, 2, 1] - g[:, 1, 2], g[:, 0, 2] - g[:, 2, 0], g[:, 1, 0] - g[:, 0, 1]], -1)


# ---------------------------------------------------------------------------------------------- breath
def breath(t):
    """0 = distant dense mass, 1 = cloud surrounds the camera. One slow breath per loop,
    ease-in-out, phase- and shape-modulated by low harmonics of the loop (so still periodic)."""
    th = 2 * np.pi * xp.asfloat(t) / T
    e = 0.5 - 0.5 * xp.cos(th + 0.35 * xp.sin(th + 0.9) + 0.08 * xp.sin(2 * th + 2.0))
    e = 0.55 * e + 0.45 * e * e * (3 - 2 * e)                         # longer, softer holds
    e = e + 0.16 * e * (1 - e) * xp.sin(2 * th + 0.4) + 0.05 * e * (1 - e) * xp.sin(5 * th + 1.3)
    return xp.clip(e, 0.0, 1.0)


def wind(t):
    """Wind travel in tiles: exactly +1 per loop (so the wrap is exact) but eased, slow while the
    cloud surrounds the camera and quicker while it is a distant mass."""
    th = 2 * np.pi * xp.asfloat(t) / T
    w = xp.asfloat(t) / T - WIND_EASE * (xp.sin(th + WIND_PHASE) - np.sin(WIND_PHASE)) / (2 * np.pi)
    return w if MOD is None else w + MOD.wind(t)


def transform(t):
    """Closed-form, T-periodic volume transform: scale s (P,), rotation axis (P, 3) and angle (P,),
    translation (P, 3)."""
    t = xp.asfloat(t)
    th = 2 * np.pi * t / T
    e = breath(t) if MOD is None else MOD.breath(t, breath(t))
    if WIPE_AT is not None:               # --wipe-at: swell round the camera through the wipe, then hold
        e = e + wipe_env(t) * xp.maximum(WIPE_BREATH - e, 0.0)
    s = S_MIN * (S_MAX / S_MIN) ** e
    axis = xp.stack([0.45 + 0.30 * xp.cos(th), xp.ones_like(th), 0.35 + 0.30 * xp.sin(th + 0.5)], -1)
    axis /= xp.norm(axis, axis=-1, keepdims=True)
    ang = TILT_SWING * xp.sin(th - 0.6) + 0.12 * xp.sin(2 * th + 1.4)
    if MOD is not None:                   # audio: radial pulses / drop-burst on s (snare kicks: Silk.world)
        s = s * MOD.scale(t)
    tr = xp.stack([0.22 * xp.sin(th + 0.6), 0.16 * xp.sin(2 * th + 1.9) - 0.05,
                   Z_FAR + (Z_CLOSE - Z_FAR) * e + 0.15 * xp.sin(th + 2.2)], -1)
    return s, axis, ang, tr


# ---------------------------------------------------------------------------------------------- audio
MOD = None          # --audio: an AudioMod. Every hook is a pure function of the *sample* time, so trail
                    # samples at past times show the modulation that was live at those times.
HOOK_DT = 0.001     # hooks are tabulated once on a 1 ms grid and linearly interpolated


def kick_pulse(tau, zeta=0.40, omega=10.0, rise=0.013):
    """Kick radial pulse, peak 1: ~40 ms rise, then a spring decay (envelope tau 1/(zeta omega) = 250 ms,
    zeta 0.4 -> one small rebound lobe)."""
    tau = np.asarray(tau, float)
    tp = np.maximum(tau, 0.0)
    wd = omega * np.sqrt(1 - zeta ** 2)
    h = (1 - np.exp(-tp / rise)) * np.exp(-zeta * omega * tp) * np.cos(wd * tp)
    g = np.arange(0, 0.3, 1e-4)
    peak = np.max((1 - np.exp(-g / rise)) * np.exp(-zeta * omega * g) * np.cos(wd * g))
    return np.where(tau > 0, h / peak, 0.0)


def spring_release(tau, zeta=0.35, omega=12.0):
    """Underdamped step response 1 -> 0 (overshoots below 0 once): the burst out of the drop."""
    tp = np.maximum(np.asarray(tau, float), 0.0)
    wd = omega * np.sqrt(1 - zeta ** 2)
    return np.exp(-zeta * omega * tp) * (np.cos(wd * tp) + zeta * omega / wd * np.sin(wd * tp))


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


class AudioMod:
    """Audio -> silk parameters. Ground-truth hits (beat.py events) drive impulses; audiofeat features
    drive the continuous parameters. Amounts are the module constants below."""
    # beaty
    KICK_AMP = 0.10           # radial scale pulse per kick (x velocity)
    SNARE_STEP, SNARE_TAU = 0.12, 0.12    # tumble angle step per snare (rad), angular-velocity decay (s)
    SNARE_AXIS = np.array([0.25, 0.35, 1.0]) / np.linalg.norm([0.25, 0.35, 1.0])   # near the view axis: a twist
    HAT_AMP, HAT_TAU, HAT_RISE = 1.0, 0.07, 0.025   # edge-ruffle gain per hat, decay, rise (s)
    SUB_AR = (0.015, 0.18)                # sub follower attack / release (s)
    SUB_LIGHT = (0.75, 0.55)              # light x (a + b sub)
    SUB_TRAIL = (0.80, 0.40)              # trail duration x (a + b sub)
    RMS_AR = (0.2, 0.6)
    RMS_WIND = 0.035                      # extra wind travel (tiles/s) at full RMS; base ~0.083
    DROP_DEPTH = 0.30                     # cloud contracts to 70% over the drop, bursts back on the downbeat
    ARC = (0.05, 0.55, 0.55, 0.30)        # beaty breath arc: start, before the drop, after the burst, end
    # ethereal
    BREATH_AR = (0.8, 1.5)
    BREATH_RANGE = (0.20, 0.60)           # body swells span this; the fade-in/out runs 0 -> 0.20
    BREATH_LAG = (1.2, 1.8)               # second slow follower on the mapped breath (no fast swells)
    WARP_STEP, WARP_TIME = 0.6, 1.0       # warp-domain rotation per chord change (rad), over (s)
    WARP_AXIS = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    LOWMID_AR = (0.4, 1.0)
    LOWMID_TRAIL = (0.85, 0.6)
    MID_AR = (0.3, 0.8)
    SHIMMER = 0.35                        # light x (1 + SHIMMER mid sin(travelling band))

    def __init__(self, wav, events, mode):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import audiofeat
        self.F = audiofeat.Features(wav)
        with open(events, encoding="utf-8") as f:
            self.ev = json.load(f)
        self.mode = mode
        self.duration = float(self.ev.get("duration", self.F.duration))
        self.g = np.arange(-3.0, self.duration + 1.0, HOOK_DT)
        g, F, tab = self.g, self.F, {}
        if mode == "beaty":
            hits = lambda k: [(e["t"], e["v"]) for e in self.ev.get(k, [])]
            tab["scale"] = 1 + self.KICK_AMP * sum(v * kick_pulse(g - tk) for tk, v in hits("kick"))
            d0, d1 = self.ev["drop"]
            c = np.where(g < d1, np.clip((g - d0) / (d1 - d0), 0, 1) ** 2, spring_release(g - d1, 0.4, 8.0))
            tab["scale"] = tab["scale"] * (1 - self.DROP_DEPTH * c)
            tab["spin"] = sum(self.SNARE_STEP * v * np.where(g > tk, 1 - np.exp(-(g - tk) / self.SNARE_TAU), 0.0)
                              for tk, v in hits("snare"))
            tab["ruffle"] = 1 + self.HAT_AMP * sum(
                v * np.where(g > th, smoothstep(0.0, self.HAT_RISE, g - th) * np.exp(-(g - th) / self.HAT_TAU), 0.0)
                for th, v in hits("hat"))
            sub = F.feat("sub", g, *self.SUB_AR)
            tab["light"] = self.SUB_LIGHT[0] + self.SUB_LIGHT[1] * sub
            tab["trail"] = self.SUB_TRAIL[0] + self.SUB_TRAIL[1] * sub
            tab["wind"] = self.RMS_WIND * F.integral("rms", g, *self.RMS_AR)
            # breath arc built on the arrangement: distant mass -> a near mass by the drop (which contracts
            # it, via scale), a bigger state after the burst, retreat over the last bar
            a0, a1, a2, a3 = self.ARC
            ramp = smoothstep(0.0, 0.85 * d0, g)
            tab["breath"] = (a0 + (a1 - a0) * ramp + (a2 - a1) * smoothstep(d1, d1 + 0.8, g)
                             + (a3 - a2) * smoothstep(d1 + 0.9, self.duration, g)
                             + 0.04 * np.sin(2 * np.pi * g / (4 * (d1 - d0)) + 1.0) * ramp)
        else:
            body = (g > 1.0) & (g < self.duration - 1.0)

            def contrast(x):      # 0-1 over the body's own range (5th..98th percentile), clipped
                lo, hi = np.percentile(x[body], 5), np.percentile(x[body], 98)
                return np.clip((x - lo) / (hi - lo), 0, 1), lo
            r = F.feat("rms", g, *self.BREATH_AR)
            e, lo = contrast(r)
            e = e * e * (3 - 2 * e)
            b0, b1 = self.BREATH_RANGE
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from audiofeat import follow
            tab["breath"] = follow(np.where(r < lo, b0 * np.clip(r / lo, 0, 1), b0 + (b1 - b0) * e),
                                   *self.BREATH_LAG, HOOK_DT)
            tab["warp"] = sum(self.WARP_STEP * smootherstep((g - c["t"]) / self.WARP_TIME)
                              for c in self.ev.get("chords", []) if c["t"] > 0)
            tab["trail"] = self.LOWMID_TRAIL[0] + self.LOWMID_TRAIL[1] * contrast(F.feat("lowmid", g, *self.LOWMID_AR))[0]
            tab["mid"] = contrast(F.feat("mid", g, *self.MID_AR))[0]
        self.tab = tab
        self._tt = {}                     # (name, device) -> (grid, table) tensors for torch lookups

    def _at(self, name, t, default):
        if name not in self.tab:
            return default
        if xp.is_t(t):
            key = (name, str(t.device))
            if key not in self._tt:
                self._tt[key] = (xp.const(t, self.g), xp.const(t, self.tab[name]))
            return xp.interp(t, None, None, *self._tt[key])
        t = np.asarray(t, float)
        return np.interp(t.ravel(), self.g, self.tab[name]).reshape(t.shape)

    def breath(self, t, base):
        return self._at("breath", t, base)

    def scale(self, t):
        return self._at("scale", t, 1.0)

    def spin(self, t):
        return self._at("spin", t, 0.0)

    def wind(self, t):
        return self._at("wind", t, 0.0)

    def warp(self, t):
        return self._at("warp", t, None)

    def ruffle(self, t):
        return self._at("ruffle", t, 1.0)

    def trail(self, t):
        v = self._at("trail", t, 1.0)
        return v if xp.is_t(v) else float(v)          # torch: per-row tensor (frame batches)

    def light(self, ts, v):
        """Per trail-sample light multiplier at sample times ts (N, k); v object points (N, k, 3)."""
        if "mid" in self.tab:   # slow shimmer: a light band drifting through the volume, depth ~ mid energy
            m = self._at("mid", ts, 0.0)
            band = xp.sin(2.6 * (v @ xp.const(v, [0.55, 0.75, 0.37])) - 2 * np.pi * 0.32 * ts + 0.7)
            return 1 + self.SHIMMER * m * band
        return self._at("light", ts, 1.0)


# ---------------------------------------------------------------------------------------------- scene
class Silk:
    BLUR = (0.0, 0.6, 1.3, 2.6, 5.0)          # design-px sigma per depth class
    EDGES = (0.9, 1.8, 3.6, 7.0)              # projected width (design px) class boundaries

    def __init__(self, preview=False, backend=None):
        self.be = backend or BACKEND
        self.n, self.K = N, K
        rng = np.random.default_rng(SEED)
        self.q = Bank(rng, 3, FLOW_F, 2, 6, rate=0.3, gain=FLOW_GAIN)       # warp
        self.psi = Bank(rng, 3, FLOW_F, 3, 6, rate=0.5, gain=FLOW_GAIN)     # vector potential
        self.ruf = Bank(rng, 3, RUF_F, 2, 6, rate=2.6)                      # shell flutter
        # wind frame and base points: along-wind a0 uniform on the tile, cross-section a unit disc
        e1 = WIND
        e2 = np.cross(e1, [0.0, 0.0, 1.0])
        e2 /= np.linalg.norm(e2)
        self.E = np.stack([e1, e2, np.cross(e1, e2)])
        self.a0 = (rng.random(N) - 0.5) * TILE
        rr = np.sqrt(rng.random(N))
        ph = rng.random(N) * 2 * np.pi
        self.b0, self.c0 = rr * np.cos(ph), rr * np.sin(ph)
        self.tau = TAU_MIN + (TAU_MAX - TAU_MIN) * rng.random(N) ** 3.0     # many short, some long
        self.lum = 0.45 + 0.55 * rng.random(N) ** 1.3
        self.accent = (rng.random(N) < ACCENT_FRAC) & (self.tau > np.median(self.tau))
        self.head = (rng.random(N) < HEAD_FRAC) & ~self.accent
        self.head_r = 0.006 + 0.010 * rng.random(N) ** 2                    # world radius
        p = rng.uniform(-1, 1, (4000, 3))
        th = rng.uniform(0, 2 * np.pi, 4000)
        self.flow_norm = self.ruf_norm = 1.0
        self.flow_norm = float(np.sqrt(np.mean(np.sum(self._flow(p, np.cos(th), np.sin(th)) ** 2, -1))))
        self.ruf_norm = float(np.sqrt(np.mean(np.sum(self._ruffle(p, np.cos(th), np.sin(th)) ** 2, -1))))
        if preview:                                   # --preview: first N/4 strands (same ones as the full run)
            self.n, self.K = N // 4, K // 2
            for nm in ("a0", "b0", "c0", "tau", "lum", "accent", "head", "head_r"):
                setattr(self, nm, getattr(self, nm)[:self.n])
            self.BLUR, self.EDGES = (0.6, 3.0), (2.0,)    # fewest blur classes
        if self.be.torch:                             # per-strand parameters live on the device
            for nm in ("a0", "b0", "c0", "tau", "lum", "accent", "head", "head_r", "E"):
                setattr(self, nm, self.be.asarray(getattr(self, nm)))

    # closed-form fields --------------------------------------------------------------------------
    def _flow(self, u, ct, st, phi=None):
        """curl of psi_j(p + WARP q(p)), p = FLOW_F u, with the exact chain rule through the warp.
        phi (P,): optional rotation of the warp's domain about AudioMod.WARP_AXIS (chord changes)."""
        one = xp.ones_like(ct)
        p = u * FLOW_F
        off = xp.const(p, [5.2, 1.3, 2.7])
        if phi is None:
            qv, qg = self.q(xp.column_stack([p + off, ct, st, one]))   # (P,3), (P,3,3)
        else:
            a = AudioMod.WARP_AXIS
            ax = xp.const(p, [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            c, s = xp.cos(phi)[:, None, None], xp.sin(phi)[:, None, None]
            R = c * xp.const(p, np.eye(3)) + s * ax + (1 - c) * xp.const(p, np.outer(a, a))   # (P,3,3) Rodrigues
            y = xp.einsum("pij,pj->pi", R, p + off)
            qv, qg = self.q(xp.column_stack([y, ct, st, one]))
            qg = xp.einsum("pji,pik->pjk", qg, R)                                    # d q / d p
        w = p + WARP * qv
        _, g = self.psi(xp.column_stack([w, ct, st, one]))                      # (P,3,3)
        g = g + WARP * xp.einsum("pji,pik->pjk", g, qg)                         # (I + A Jq)^T grad
        return curl(g) / self.flow_norm

    def _ruffle(self, u, ct, st):
        _, g = self.ruf(xp.column_stack([u * RUF_F, ct, st, xp.ones_like(ct)]))
        return curl(g) / self.ruf_norm

    def world(self, ts, t_head, rows=None):
        """World positions (N, k, 3), shell alpha (N, k) and displaced object points (N, k, 3)
        at sample times ts (N, k). rows: (a0, b0, c0) per row (torch frame batches tile the strands)."""
        a0, b0, c0 = rows if rows is not None else (self.a0, self.b0, self.c0)
        n, k = ts.shape
        a_un = a0[:, None] + TILE * wind(ts)
        wrap = xp.floor((a0 + TILE * wind(t_head) + TILE / 2) / TILE)   # head's wrap, whole trail
        a = a_un - wrap[:, None] * TILE
        u = (a[..., None] * self.E[0] + b0[:, None, None] * self.E[1]
             + c0[:, None, None] * self.E[2]).reshape(-1, 3)
        r = xp.norm(u, axis=1)
        th = 2 * np.pi * ts.ravel() / T
        ct, st = xp.cos(th), xp.sin(th)
        phi = None if MOD is None else MOD.warp(ts.ravel())
        v = u + FLOW_AMP * self._flow(u, ct, st, phi)
        ruf = RUF_AMP * smoothstep(0.45, 0.9, r) * (1.0 if MOD is None else MOD.ruffle(ts.ravel()))
        v = v + ruf[:, None] * self._ruffle(u, ct, st)
        rv = xp.norm(v, axis=1, keepdims=True)
        d = v / xp.maximum(rv, 1e-9)
        shape = (1 + 0.10 * xp.sin(2.0 * d @ xp.const(d, [0.8, 0.5, 0.3]) + th)
                 + 0.08 * xp.sin(3.0 * d @ xp.const(d, [-0.3, 0.9, 0.3]) - th + 1.0)
                 + 0.05 * xp.sin(5.0 * d @ xp.const(d, [0.2, -0.4, 0.9]) + 2 * th + 2.0))
        o = v * (0.55 + 0.45 * rv) * shape[:, None] * xp.const(v, [1.0, 1.12, 0.92])
        s, axis, ang, tr = transform(ts.ravel())
        o = o * s[:, None]
        c, sn = xp.cos(ang)[:, None], xp.sin(ang)[:, None]
        o = o * c + xp.cross(axis, o) * sn + axis * xp.sum_(axis * o, 1, keepdims=True) * (1 - c)
        if MOD is not None and "spin" in MOD.tab:     # snare kicks: extra tumble about a near-view axis
            a = xp.const(o, AudioMod.SNARE_AXIS)
            kick = MOD.spin(ts.ravel())
            c, sn = xp.cos(kick)[:, None], xp.sin(kick)[:, None]
            o = o * c + xp.cross(a, o) * sn + a * (o @ a)[:, None] * (1 - c)
        wpos = o + tr
        alpha = 1 - smoothstep(0.70, 0.97, r)
        return wpos.reshape(n, k, 3), alpha.reshape(n, k), v.reshape(n, k, 3)

    def strands(self, t):
        """Projected trails (N, K+1, 2), depth z (N, K+1), per-sample light (N, K+1), head alpha.
        Torch: t may be a 1-D tensor of B frame times; rows are then frame-major (B*N)."""
        j = xp.const(self.tau, np.arange(self.K + 1) / self.K)
        th_, tau, lum, rows = t, self.tau, self.lum, None
        if xp.is_t(t):                                # frame batch: tile the strands once per frame
            B = t.numel()
            th_ = t.reshape(-1).repeat_interleave(self.n)
            tau, lum = self.tau.repeat(B), self.lum.repeat(B)
            rows = (self.a0.repeat(B), self.b0.repeat(B), self.c0.repeat(B))
        tr = 1.0 if MOD is None else MOD.trail(th_)
        ts = (th_[:, None] if xp.is_t(th_) else t) - tau[:, None] * j[None, :] * (tr[:, None] if xp.is_t(tr) else tr)
        ts0 = ts
        ts = xp.concat([ts[:, :1], ts[:, :1] - 0.04, ts[:, 1:]], 1)
        Wp, al, v = self.world(ts, th_, rows)
        ospeed = xp.norm(v[:, 0] - v[:, 1], axis=1) / 0.04          # air speed, object units/s
        lmod = 1.0 if MOD is None else MOD.light(ts0, xp.concat([v[:, :1], v[:, 2:]], 1))
        Wp, al = xp.concat([Wp[:, :1], Wp[:, 2:]], 1), xp.concat([al[:, :1], al[:, 2:]], 1)
        z = Wp[..., 2]
        zs = xp.maximum(z, 1e-3)
        P = xp.stack([CX + FOCAL * Wp[..., 0] / zs, CY + FOCAL * Wp[..., 1] / zs], -1)
        near = smoothstep(Z_NEAR, Z_FADE, z)
        fog = xp.exp(-xp.maximum(z - FOG_Z, 0.0) / FOG_L)
        depth = xp.clip(Z_REF / zs, 0.0, 12.0)
        fade = (1 - j) ** 1.3
        sp = 0.4 + 0.6 * smoothstep(0.12, 0.45, ospeed)
        glow = (transform(th_)[0] / S_MIN) ** BREATH_GLOW
        light = (lum * sp * glow)[:, None] * fade[None, :] * al * near * fog * depth * lmod
        return P, z, light, al[:, 0] * near[:, 0] * fog[:, 0]

    # drawing -------------------------------------------------------------------------------------
    def frames(self, ts, W: int, H: int, ss: int = 2) -> list:
        """Frames for a list of times; torch renders them FRAME_BATCH at a time."""
        if not self.be.torch:
            return [self.frame(t, W, H, ss) for t in ts]
        out = []
        for i in range(0, len(ts), max(1, FRAME_BATCH)):
            out.extend(self._frames_torch(list(ts[i:i + max(1, FRAME_BATCH)]), W, H, ss))
        return out

    def frame(self, t: float, W: int, H: int, ss: int = 2) -> np.ndarray:
        if self.be.torch:
            return self._frames_torch([t], W, H, ss)[0]
        CWd, CHd = W * ss, H * ss
        kc = CWd / DW                                         # design px -> canvas px
        P, z, light, head_al = self.strands(t)
        P = P * kc
        lw = CWd / (2 * DW)                                   # light per canvas px, tuned at 2160 wide
        A, B = P[:, :-1], P[:, 1:]
        IA, IB = light[:, :-1], light[:, 1:]
        zm = 0.5 * (z[:, :-1] + z[:, 1:])
        # drop dark segments, segments touching the near plane / behind the camera, and off-canvas
        keep = ((IA + IB) > 0.003) & (z[:, :-1] > Z_NEAR) & (z[:, 1:] > Z_NEAR)
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        keep &= (hi[..., 0] > -8) & (hi[..., 1] > -8) & (lo[..., 0] < CWd + 8) & (lo[..., 1] < CHd + 8)
        sid = np.broadcast_to(np.arange(self.n)[:, None], keep.shape)[keep]
        A, B, IA, IB, zm = A[keep], B[keep], IA[keep], IB[keep], zm[keep]
        width = LINE_W * FOCAL / zm                           # projected width, design px
        cls = np.searchsorted(self.EDGES, width)
        blur = tuple(self.BLUR)
        if MATTE and WIPE_AT is not None:                     # wipe: near strands grow wide and soft
            blur = blur + WIDE_BLUR
            wv = float(wipe_env(t))
            nw = np.clip((WIPE_ZN / zm) ** 1.5, 0.0, 1.0)
            grow = (1 + (WIPE_BASE - 1) * nw * (1 - wv)
                    + (WIPE_GROW - 1) * wv * (nw + (1 - nw) * WIPE_FAR * wv))
            sig = WIPE_SOFT * width * grow
            k = np.clip(np.round(np.log2(sig / WIDE_BLUR[0])), 0, len(WIDE_BLUR) - 1).astype(np.int64)
            cls = np.where(sig > WIDE_BLUR[0] / np.sqrt(2), len(self.BLUR) + k, cls)
            gl = grow ** WIPE_LIGHT
            IA, IB = IA * gl, IB * gl
        nsl = len(blur)
        ln = np.linalg.norm(B - A, axis=1)
        step = np.maximum(0.75, 0.5 * np.array(blur)[cls] * kc)
        n = np.clip(np.ceil(ln / step), 1, 4000).astype(np.int64)
        rep = np.repeat(np.arange(len(n)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        f = (np.arange(len(rep)) - start + 0.5) / n[rep]
        x = A[rep, 0] + (B[rep, 0] - A[rep, 0]) * f
        y = A[rep, 1] + (B[rep, 1] - A[rep, 1]) * f
        w = (IA[rep] + (IB[rep] - IA[rep]) * f) * (ln / n)[rep] * lw
        slot = (cls if MATTE else np.where(self.accent[sid], nsl, cls))[rep]   # matte: no accent
        x0, y0 = np.floor(x - 0.5), np.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < CWd - 1) & (y0 < CHd - 1)
        idx = (y0 * CWd + x0)[ok]
        w, fx, fy, slot = w[ok], fx[ok], fy[ok], slot[ok]
        size = CWd * CHd
        white = np.zeros((CHd, CWd), np.float32)
        accl = np.zeros((CHd, CWd), np.float32)
        for sl in np.unique(slot):
            sel = slot == sl
            i, ww, a, b = idx[sel], w[sel], fx[sel], fy[sel]
            acc = np.zeros(size)
            for di, wt in ((0, (1 - a) * (1 - b)), (1, a * (1 - b)), (CWd, (1 - a) * b), (CWd + 1, a * b)):
                acc += np.bincount(i + di, ww * wt, minlength=size)
            layer = acc.reshape(CHd, CWd).astype(np.float32)
            if sl == nsl:
                accl = cv2.GaussianBlur(layer, (0, 0), max(0.5, 0.5 * kc)) * 1.6
                continue
            sig = blur[sl] * kc
            white += cv2.GaussianBlur(layer, (0, 0), sig) if sig > 0.3 else layer
        tot = (white + accl) * EXPOSURE
        lum = 1 - np.exp(-tot)
        if MATTE:                                             # greyscale matte, 0 = A, 255 = B
            img = np.repeat(np.clip(lum * 255 + 0.5, 0, 255).astype(np.uint8)[..., None], 3, -1)
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if ss > 1 else img
        frac_a = accl / np.maximum(white + accl, 1e-6)
        col = (INK - BG)[None, None, :] * (1 - frac_a[..., None]) + (ACCENT - BG)[None, None, :] * frac_a[..., None]
        img = np.clip(BG + lum[..., None] * col, 0, 255).astype(np.uint8)
        img = self.heads(img, P[:, 0], z[:, 0], head_al, kc)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def heads(self, img, xy, z, al, kc):
        """Small dark-rimmed bubble heads (the A look), radius ~ 1/z, drawn far to near."""
        S = 16.0
        sel = np.where(self.head & (al > 0.08) & (z > Z_NEAR))[0]
        sel = sel[np.argsort(-z[sel], kind="stable")]
        for i in sel:
            R = min(self.head_r[i] * FOCAL / z[i], 16.0) * kc
            rim = max(1.0, 0.9 * kc * min(1.0, Z_REF / z[i]) ** 0.5)
            if not (-R < xy[i, 0] < img.shape[1] + R and -R < xy[i, 1] < img.shape[0] + R):
                continue
            c = (int(round(xy[i, 0] * S)), int(round(xy[i, 1] * S)))
            a = float(al[i]) * float(self.lum[i])
            fill = tuple(float(v) for v in BG + (INK - BG) * a)
            cv2.circle(img, c, int(round((R + rim) * S)), EDGE, -1, cv2.LINE_AA, 4)
            cv2.circle(img, c, int(round(R * S)), fill, -1, cv2.LINE_AA, 4)
        return img

    # torch drawing (same algorithm as frame(): bilinear splat of the densely sampled segments into one
    # layer per blur class, cv2-identical separable Gaussian per layer, tone map, heads, area downsample)
    def _frames_torch(self, tl, W: int, H: int, ss: int = 2) -> list:
        Tm, dev = xp.torch(), self.be.dev
        f32 = Tm.float32
        B, n = len(tl), self.n
        CWd, CHd = W * ss, H * ss
        size = CWd * CHd
        kc = CWd / DW
        tb = Tm.as_tensor(np.asarray(tl, float), dtype=f32, device=dev)
        P, z, light, head_al = self.strands(tb)                     # rows frame-major (B*n)
        P = P * kc
        lw = CWd / (2 * DW)
        A, Bp = P[:, :-1], P[:, 1:]
        IA, IB = light[:, :-1], light[:, 1:]
        zm = 0.5 * (z[:, :-1] + z[:, 1:])
        keep = ((IA + IB) > 0.003) & (z[:, :-1] > Z_NEAR) & (z[:, 1:] > Z_NEAR)
        lo, hi = Tm.minimum(A, Bp), Tm.maximum(A, Bp)
        keep &= (hi[..., 0] > -8) & (hi[..., 1] > -8) & (lo[..., 0] < CWd + 8) & (lo[..., 1] < CHd + 8)
        row = Tm.arange(B * n, device=dev)[:, None].expand_as(keep)[keep]
        sid, fb = row % n, row // n
        A, Bp, IA, IB, zm = A[keep], Bp[keep], IA[keep], IB[keep], zm[keep]
        width = LINE_W * FOCAL / zm
        cls = Tm.searchsorted(Tm.as_tensor(self.EDGES, dtype=f32, device=dev), width)
        blur = tuple(self.BLUR)
        if MATTE and WIPE_AT is not None:                     # wipe: near strands grow wide and soft
            blur = blur + WIDE_BLUR
            wv = Tm.as_tensor(np.asarray(wipe_env(np.asarray(tl, float)), float), dtype=f32, device=dev)[fb]
            nw = Tm.clamp((WIPE_ZN / zm) ** 1.5, 0.0, 1.0)
            grow = (1 + (WIPE_BASE - 1) * nw * (1 - wv)
                    + (WIPE_GROW - 1) * wv * (nw + (1 - nw) * WIPE_FAR * wv))
            sig = WIPE_SOFT * width * grow
            k = Tm.clamp(Tm.round(Tm.log2(sig / WIDE_BLUR[0])), 0, len(WIDE_BLUR) - 1).long()
            cls = Tm.where(sig > WIDE_BLUR[0] / np.sqrt(2), len(self.BLUR) + k, cls)
            gl = grow ** WIPE_LIGHT
            IA, IB = IA * gl, IB * gl
        nsl = len(blur)
        D = Bp - A
        ln = Tm.linalg.vector_norm(D, dim=1)
        step = Tm.clamp(0.5 * Tm.as_tensor(blur, dtype=f32, device=dev)[cls] * kc, min=0.75)
        ns = Tm.clamp(Tm.ceil(ln / step), 1, 4000).long()
        rep = Tm.repeat_interleave(ns)                               # segment of each splat point
        start = (Tm.cumsum(ns, 0) - ns)[rep]
        f = ((Tm.arange(rep.numel(), device=dev) - start).to(f32) + 0.5) / ns[rep].to(f32)
        seg = Tm.stack([A[:, 0], A[:, 1], D[:, 0], D[:, 1], IA, IB - IA, ln / ns.to(f32) * lw], 1)[rep]
        x = seg[:, 0] + seg[:, 2] * f
        y = seg[:, 1] + seg[:, 3] * f
        w = (seg[:, 4] + seg[:, 5] * f) * seg[:, 6]
        del seg
        slot = cls if MATTE else Tm.where(self.accent[sid], nsl, cls)   # matte: no accent
        slot, pfb = slot[rep], fb[rep]
        x0, y0 = Tm.floor(x - 0.5), Tm.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        x0, y0 = x0.long(), y0.long()
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < CWd - 1) & (y0 < CHd - 1)
        x0, y0, w, fx, fy, slot, pfb = x0[ok], y0[ok], w[ok], fx[ok], fy[ok], slot[ok], pfb[ok]
        used = Tm.unique(slot).tolist()                        # only the layers in use get a buffer
        remap = Tm.full((nsl + 1,), -1, dtype=Tm.long, device=dev)
        remap[used] = Tm.arange(len(used), device=dev)
        U = len(used)
        gi = (pfb * U + remap[slot]) * size + y0 * CWd + x0
        idx = Tm.cat([gi, gi + 1, gi + CWd, gi + CWd + 1])
        val = Tm.cat([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
        del gi, x, y, w, fx, fy, x0, y0, rep, start, f
        buf = Tm.zeros(B * U * size, dtype=f32, device=dev)
        buf.index_put_((idx,), val, accumulate=True)          # deterministic under use_deterministic_algorithms
        del idx, val
        buf = buf.view(B, U, CHd, CWd)
        white = Tm.zeros((B, CHd, CWd), dtype=f32, device=dev)
        accl = Tm.zeros_like(white)
        for u, sl in enumerate(used):
            layer = buf[:, u]
            if sl == nsl:
                accl = xp.gaussian_blur(layer, max(0.5, 0.5 * kc)) * 1.6
                continue
            sg = blur[sl] * kc
            white += xp.gaussian_blur(layer, sg) if sg > 0.3 else layer
        del buf
        lum = 1 - Tm.exp(-((white + accl) * EXPOSURE))
        if MATTE:                                             # greyscale matte, 0 = A, 255 = B
            img = Tm.clamp(lum * 255 + 0.5, 0, 255).to(Tm.uint8)
            img = self._area(img[:, None].float(), ss)[:, 0] if ss > 1 else img
            return list(img[..., None].expand(-1, -1, -1, 3).contiguous().cpu().numpy())
        frac_a = accl / Tm.clamp(white + accl, min=1e-6)
        bg = Tm.as_tensor(BG, dtype=Tm.float64, device=dev)
        fa = frac_a.double()[..., None]
        col = (Tm.as_tensor(INK - BG, dtype=Tm.float64, device=dev) * (1 - fa)
               + Tm.as_tensor(ACCENT - BG, dtype=Tm.float64, device=dev) * fa)
        img = Tm.clamp(bg + lum.double()[..., None] * col, 0, 255).to(Tm.uint8)   # truncates, as astype
        del col, fa, frac_a, lum, white, accl
        img = self._heads_torch(img, P[:, 0], z[:, 0], head_al, kc, B)
        if ss > 1:
            img = self._area(img.permute(0, 3, 1, 2).float(), ss).permute(0, 2, 3, 1)
        return list(img.contiguous().cpu().numpy())

    @staticmethod
    def _area(x, ss: int):
        """cv2.resize INTER_AREA by an integer factor on uint8 values: box mean, rounded half up."""
        Tm = xp.torch()
        s = Tm.nn.functional.avg_pool2d(x, ss) * (ss * ss)                  # exact integer sums
        return Tm.div(s + (ss * ss) // 2, ss * ss, rounding_mode="floor").clamp(0, 255).to(Tm.uint8)

    def _heads_torch(self, img, xy, z, al, kc, B):
        """Bubble heads as in heads(): rim disc (EDGE) then fill disc per head, far to near, cv2 LINE_AA-like
        edge coverage clip(R + 1 - d, 0, 1) at integer pixel centres, 1/16 px centres and radii. The
        painter's order is resolved per pixel: sort the stamp pixels by (pixel, layer) and composite with
        a segmented suffix product of (1 - coverage), so all heads of all frames draw in one pass."""
        Tm, dev = xp.torch(), img.device
        CHd, CWd = img.shape[1], img.shape[2]
        S = 16.0
        head, lum, hr = self.head.repeat(B), self.lum.repeat(B), self.head_r.repeat(B)
        sel = Tm.nonzero(head & (al > 0.08) & (z > Z_NEAR))[:, 0]
        zz = z[sel]
        R = Tm.clamp(hr[sel] * FOCAL / zz, max=16.0) * kc
        rim = Tm.clamp(0.9 * kc * Tm.clamp(Z_REF / zz, max=1.0) ** 0.5, min=1.0)
        x, y = xy[sel, 0], xy[sel, 1]
        on = (x > -R) & (x < CWd + R) & (y > -R) & (y < CHd + R)
        sel, zz, R, rim, x, y = sel[on], zz[on], R[on], rim[on], x[on], y[on]
        if sel.numel() == 0:
            return img
        order = Tm.sort(-zz, stable=True)[1]                  # far first (stable: index order on ties)
        rank = Tm.empty_like(order)
        rank[order] = Tm.arange(order.numel(), device=dev)
        a = (al[sel] * lum[sel]).double()
        fill = Tm.round(Tm.as_tensor(BG, dtype=Tm.float64, device=dev)
                        + Tm.as_tensor(INK - BG, dtype=Tm.float64, device=dev) * a[:, None])
        edge = Tm.as_tensor(EDGE, dtype=Tm.float64, device=dev).expand_as(fill)
        # coverage stamps drawn by cv2 itself (LINE_AA discs are polygons from ellipse2Poly: a diamond below
        # 3 px, a 12-gon below 10 px), one small patch per head on the CPU: exact, and only ~400 per frame
        hs = Tm.stack([Tm.round(x * S), Tm.round(y * S), Tm.round((R + rim) * S), Tm.round(R * S)], 1)
        hs = hs.long().cpu().numpy()
        half = int(hs[:, 2].max()) // 16 + 3
        side = 2 * half + 1
        ox, oy = (hs[:, 0] >> 4) - half, (hs[:, 1] >> 4) - half
        atlas = np.zeros((len(hs), 2, side, side), np.uint8)
        for i, (c16x, c16y, ro, ri) in enumerate(hs.tolist()):
            c = (int(c16x - ox[i] * 16), int(c16y - oy[i] * 16))
            cv2.circle(atlas[i, 0], c, ro, 255, -1, cv2.LINE_AA, 4)
            cv2.circle(atlas[i, 1], c, ri, 255, -1, cv2.LINE_AA, 4)
        at = Tm.from_numpy(atlas).to(dev)
        h, layer, ly, lx = Tm.nonzero(at, as_tuple=True)
        cov = at[h, layer, ly, lx].double() / 255.0
        px = Tm.from_numpy(ox).to(dev)[h] + lx
        py = Tm.from_numpy(oy).to(dev)[h] + ly
        ok = (px >= 0) & (py >= 0) & (px < CWd) & (py < CHd)
        h, layer, px, py, cov = h[ok], layer[ok], px[ok], py[ok], cov[ok]
        fbh = sel[h] // self.n
        pix2 = fbh * (CHd * CWd) + py * CWd + px
        lay = 2 * rank[h] + layer                             # rim (2 rank) under fill (2 rank + 1)
        colr = Tm.where((layer == 0)[:, None], edge[h], fill[h])
        LB = 1 << int(2 * order.numel() + 2).bit_length()
        srt = Tm.sort(pix2 * LB + lay)[1]
        pix2, cov, colr = pix2[srt], cov[srt], colr[srt]
        lg = Tm.log1p(-Tm.clamp(cov, max=1 - 1e-7))
        cs = Tm.cumsum(lg, 0)
        upix, inv, counts = Tm.unique_consecutive(pix2, return_inverse=True, return_counts=True)
        ends = Tm.cumsum(counts, 0) - 1
        starts = ends - counts + 1
        after = cs[ends][inv] - cs                             # log transmittance of the later layers
        total = cs[ends] - cs[starts] + lg[starts]             # all layers of the pixel
        flat = img.view(-1, 3)
        base = flat[upix].double()
        c = Tm.cumsum(colr * (cov * Tm.exp(after))[:, None], 0)    # segment sums (sorted: no atomics)
        new = base * Tm.exp(total)[:, None] + c[ends] - c[starts] + (colr * (cov * Tm.exp(after))[:, None])[starts]
        flat[upix] = Tm.clamp(Tm.round(new), 0, 255).to(Tm.uint8)
        return img


# ---------------------------------------------------------------------------------------------- output
def render(path: str, W: int, H: int, ss: int, preview: bool = False) -> float:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc = Silk(preview)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    step = max(1, FRAME_BATCH) if sc.be.torch else 1
    for n in range(0, NFR, step):
        for fr in sc.frames([m / FPS for m in range(n, min(n + step, NFR))], W, H, ss):
            proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"{path}  {W}x{H}  {NFR} frames  {dt:.1f}s")
    return dt


def seam(W: int, H: int, ss: int) -> None:
    """Exactness of the loop: frame(T) vs frame(0), and the 249 -> 0 step vs ordinary steps."""
    sc = Silk()
    f = {n: sc.frame(n / FPS, W, H, ss).astype(float) for n in (0, 1, 124, 125, 248, 249)}
    fT = sc.frame(T, W, H, ss).astype(float)
    d = lambda a, b: float(np.mean(np.abs(a - b)))
    print(f"frame(T) vs frame(0): {d(fT, f[0]):.4f}")
    print(f"step 249->0: {d(f[249], f[0]):.3f}   0->1: {d(f[0], f[1]):.3f}   "
          f"124->125: {d(f[124], f[125]):.3f}   248->249: {d(f[248], f[249]):.3f}")


STEM = "D_silkwind"
CHANGES = "0.44,2.94,5.44,7.94"          # chord changes every 2.5 s, anchored on the breath peak (5.44 s)
GEN_STILLS = (2.0, 6.0, 9.6)              # expansion transit, inside the cloud, retreated mass


def gens(stem: str = STEM) -> list[int]:
    out = []
    for f in glob.glob(os.path.join(LOCAL, f"{stem}_g*_*")):
        m = re.match(rf"{re.escape(stem)}_g(\d+)_", os.path.basename(f))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def next_gen(stem: str = STEM) -> int:
    g = gens(stem)
    return (g[-1] + 1) if g else 1


def free_path(gen: int, kind: str) -> tuple[int, str]:
    """Unique output path for generation `gen`; bumps the number rather than overwrite anything."""
    while True:
        path = os.path.join(LOCAL, f"{STEM}_g{gen:02d}_{kind}.mp4")
        if not os.path.exists(path):
            return gen, path
        gen += 1


def gen_stills(gen: int) -> list[str]:
    """One still per GEN_STILLS time for this generation, <=540 px wide, in review/."""
    os.makedirs(REVIEW, exist_ok=True)
    sc, names = Silk(), []
    for t in GEN_STILLS:
        name = f"{STEM}_g{gen:02d}_still_{t:.1f}.jpg"
        Image.fromarray(sc.frame(t, 540, 675, 2)).save(os.path.join(REVIEW, name), quality=92)
        names.append(name)
    return names


# ---------------------------------------------------------------------------------------------- audio runs
AUDIO_STILLS = {"beaty": (4.84, 5.20, 9.64), "ethereal": (3.6, 5.0, 10.4)}
QC_ROWS = {   # (label, times) per row; all on the 25 fps grid
    "beaty": [("kick 4.80", (4.76, 4.84, 4.88, 5.00, 5.20)),
              ("snare 5.40", (5.36, 5.44, 5.52, 5.60, 5.68)),
              ("drop 8.4-9.6 / burst 9.60", (8.32, 9.00, 9.56, 9.64, 9.72, 10.00))],
    "ethereal": [("chord 3.0", (2.80, 3.20, 3.60, 4.00, 4.40)),
                 ("chord 6.0", (5.80, 6.20, 6.60, 7.00, 7.40)),
                 ("breath", (1.00, 4.80, 8.00, 10.40, 11.60))],
}


def configure(wav: str, events: str, mode: str, matte: bool = False) -> None:
    """Audio mode: 12 s non-looping piece in v004 under the test name D_silkwind_<mode>
    (--matte: v005, test D_wipe_<mode>, greyscale matte for matte_wipe.py)."""
    global MOD, T, NFR, VDIR, LOCAL, REVIEW, STEM, GEN_STILLS, MATTE
    MOD = AudioMod(wav, events, mode)
    T = MOD.duration
    NFR = int(round(FPS * T))
    MATTE = matte
    VDIR = os.path.join(REPO, "11_mathtype", "v005" if matte else "v004")
    LOCAL, REVIEW = os.path.join(VDIR, "local"), os.path.join(VDIR, "review")
    STEM = f"D_wipe_{mode}" if matte else f"D_silkwind_{mode}"
    GEN_STILLS = AUDIO_STILLS[mode]


def matte_main(a) -> None:
    """--matte: render D_wipe_<mode>_gNN_matte.mp4 (360x450; --preview = N/4 strands, else full N at 2x)
    plus a params sidecar json; matte_wipe.py composites it under the same gNN."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gens as G
    prev = gens(STEM)
    parent = a.parent or (f"{STEM} g{prev[-1]:02d}" if prev else "D_silkwind_" + a.mode + " (v004)")
    g, path = free_path(a.gen or next_gen(STEM), "matte")
    G.backfill(LOCAL, STEM, g, G.now_nz(), parent, a.trigger, a.change, [], [])
    dt = render(path, 360, 450, 1 if a.preview else 2, a.preview)
    params = dict(mode=a.mode, wipe_at=WIPE_AT, wipe_dur=WIPE_DUR, wipe_pre=WIPE_PRE, grow=WIPE_GROW, base=WIPE_BASE,
                  zn=WIPE_ZN, far=WIPE_FAR, soft=WIPE_SOFT, light=WIPE_LIGHT, breath=WIPE_BREATH,
                  preview=bool(a.preview), render_s=round(dt, 1), audio=os.path.basename(a.audio))
    with open(path.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=1)
    G.add_files(LOCAL, STEM, g, [os.path.basename(path)], [])
    print("matte", path, params)


def mux_audio(src: str, wav: str) -> str:
    out = src.replace(".mp4", "_audio.mp4")
    if os.path.exists(out):
        sys.exit(f"{out} exists; not overwriting")
    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", src, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    return out


def qc(video: str, gen: int, mode: str) -> list[str]:
    """Labelled frame strip around hits / changes, and a per-frame sync plot of the rendered video
    (cloud spread, brightness, frame difference) against the ground-truth events."""
    sc, rows = Silk(True), []
    tw, th_ = 240, 300
    for label, times in QC_ROWS[mode]:
        tiles = []
        for t in times:
            im = cv2.resize(sc.frame(t, 360, 450, 1), (tw, th_), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{t:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 88, 42), 1, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < 6:
            tiles.append(np.full_like(tiles[0], 5))
        row = np.concatenate(tiles, 1)
        bar = np.full((22, row.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (243, 234, 219), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, row], 0))
    strip = os.path.join(REVIEW, f"{STEM}_g{gen:02d}_qc_strip.jpg")
    Image.fromarray(np.concatenate(rows, 0)).save(strip, quality=90)
    # sync metrics from the encoded video
    cap, L = cv2.VideoCapture(video), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        L.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) - 5)
    L = np.maximum(np.stack(L), 0)
    h, w = L.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.5 * h) ** 2
    tot = L.sum((1, 2)) + 1e-6
    spread = np.sqrt((L * r2).sum((1, 2)) / tot)
    bright = L.mean((1, 2))
    diff = np.concatenate([[0], np.abs(np.diff(L, axis=0)).mean((1, 2))])
    # dense optical flow, split into radial (expansion +) and non-radial (turn / sweep) parts, ink-weighted
    rx, ry = xx - w / 2, yy - h / 2
    rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    rx, ry = rx / rn, ry / rn
    rad, tan = [0.0], [0.0]
    for k in range(1, len(L)):
        a8, b8 = (np.clip(L[k - 1] * 4, 0, 255).astype(np.uint8), np.clip(L[k] * 4, 0, 255).astype(np.uint8))
        fl = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        wt = cv2.GaussianBlur(L[k] + L[k - 1], (0, 0), 3) + 1e-3
        vr = fl[..., 0] * rx + fl[..., 1] * ry
        vt = -fl[..., 0] * ry + fl[..., 1] * rx                      # signed tangential (swirl)
        rad.append(float((vr * wt).sum() / wt.sum()))
        tan.append(float((vt * wt).sum() / wt.sum()))
    rad, tan = np.array(rad), np.array(tan)
    tf = np.arange(len(L)) / FPS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    for a, y, n in zip(ax, (spread, bright, diff, rad, tan),
                       ("spread", "brightness", "frame diff", "radial flow px", "swirl flow px")):
        a.plot(tf, y, lw=1.0, color="0.2")
        a.set_ylabel(n)
        for e in MOD.ev.get("kick", []):
            a.axvline(e["t"], color="C0", lw=0.7, alpha=0.7)
        for e in MOD.ev.get("snare", []):
            a.axvline(e["t"], color="C3", lw=0.7, alpha=0.7, ls="--")
        for c in MOD.ev.get("chords", []):
            a.axvline(c["t"], color="C2", lw=1.3, alpha=0.8)
    ax[-1].set_xlabel("s  (blue kick, red dashed snare, green chord)")
    fig.tight_layout()
    plot = os.path.join(REVIEW, f"{STEM}_g{gen:02d}_qc_sync.png")
    fig.savefig(plot, dpi=80)
    plt.close(fig)
    for inst in ("kick", "snare"):
        ks = [int(round(e["t"] * FPS)) for e in MOD.ev.get(inst, []) if 2 <= e["t"] * FPS < len(L) - 3]
        if ks:
            pre = np.array([spread[k - 1] for k in ks])
            post = np.array([spread[k + 1:k + 3].max() for k in ks])
            dpre = np.array([diff[k - 1] for k in ks])
            dpost = np.array([diff[k + 1:k + 3].max() for k in ks])
            print(f"  {inst}: spread +{100 * np.mean(post / pre - 1):.1f}% (1-2 frames after vs 1 before), "
                  f"frame diff x{np.mean(dpost / np.maximum(dpre, 1e-3)):.2f}, radial flow "
                  f"{np.mean([rad[k - 1] for k in ks]):+.2f} -> {np.mean([rad[k + 1:k + 3].max() for k in ks]):+.2f} px, "
                  f"swirl {np.mean([tan[k - 1] for k in ks]):+.2f} -> {np.mean([tan[k + 1:k + 4][np.argmax(np.abs(tan[k + 1:k + 4]))] for k in ks]):+.2f} px")
    for c in MOD.ev.get("chords", []):
        k = int(round(c["t"] * FPS))
        if 2 <= k < len(L) - 30:
            print(f"  chord {c['name']} @ {c['t']:.1f}: |swirl| 0.4 s before {np.abs(tan[k - 10:k]).mean():.2f}, "
                  f"over the next 1 s {np.abs(tan[k:k + 25]).mean():.2f} px; frame diff {diff[k - 10:k].mean():.2f} -> "
                  f"{diff[k:k + 25].mean():.2f}")
    print("qc", strip, plot)
    return [os.path.basename(strip), os.path.basename(plot)]


def audio_main(a) -> None:
    configure(a.audio, a.events, a.mode, a.matte)
    if a.matte:
        return matte_main(a)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gens as G
    prev = gens(STEM)
    parent = a.parent or (f"{STEM} g{prev[-1]:02d}" if prev else "D_silkwind g04 (v003)")
    wavname = os.path.basename(a.audio)
    if a.preview:
        g, path = free_path(a.gen or next_gen(STEM), "preview")
        G.backfill(LOCAL, STEM, g, G.now_nz(), parent, a.trigger, a.change, [], [])
        render(path, 360, 450, 1, True)
        out = mux_audio(path, a.audio)
        stills = gen_stills(g)
        G.add_files(LOCAL, STEM, g, [os.path.basename(path), os.path.basename(out), wavname], stills)
        G.add_files(LOCAL, STEM, g, [], qc(path, g, a.mode))
    if a.final:
        g = a.gen or gens(STEM)[-1]
        g, path = free_path(g, "final")
        if not G._load(LOCAL, STEM).get(str(g)):
            G.backfill(LOCAL, STEM, g, G.now_nz(), parent, a.trigger, a.change, [], [])
        render(path, 1080, 1350, 2)
        out = mux_audio(path, a.audio)
        G.add_files(LOCAL, STEM, g, [os.path.basename(path), os.path.basename(out), wavname], [])
        print("final", out)
    if a.qc_only:
        g = a.gen or gens(STEM)[-1]
        qc(os.path.join(LOCAL, f"{STEM}_g{g:02d}_preview.mp4"), g, a.mode)


def main():
    global WIPE_AT, WIPE_DUR, WIPE_GROW, WIPE_BASE, WIPE_ZN, WIPE_FAR, WIPE_SOFT, WIPE_LIGHT, WIPE_BREATH
    global BACKEND, FRAME_BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="wav: audio-reactive 12 s piece in v004 (needs --events, --mode)")
    ap.add_argument("--events", help="events json (beat.py --events-json, or chord changes)")
    ap.add_argument("--mode", choices=("beaty", "ethereal"), default="beaty")
    ap.add_argument("--parent", default="", help="GENERATIONS.md parent (audio mode)")
    ap.add_argument("--trigger", default="", help="GENERATIONS.md trigger (audio mode)")
    ap.add_argument("--change", default="", help="GENERATIONS.md change (audio mode)")
    ap.add_argument("--qc-only", action="store_true", help="audio mode: redo the QC strip / sync plot")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--preview", action="store_true", help="fast CPU test: 360x450, N/4 strands, K/2, 1x, 2 blur classes -> _preview.mp4")
    ap.add_argument("--gen", type=int, help="generation number (default: draft = next, final = latest)")
    ap.add_argument("--still", type=float, nargs="*", help="scratch stills to review/")
    ap.add_argument("--gstills", action="store_true", help="per-generation stills for --gen / latest")
    ap.add_argument("--big", action="store_true", help="scratch stills / seam at 1080x1350")
    ap.add_argument("--seam", action="store_true")
    ap.add_argument("--mux", action="store_true", help="gen final + looped chords -> _final_chords.mp4")
    ap.add_argument("--matte", action="store_true", help="audio mode: greyscale strand matte -> v005 D_wipe_<mode>_gNN_matte.mp4")
    ap.add_argument("--wipe-at", type=float, help="matte: wipe moment (s)")
    ap.add_argument("--wipe-dur", type=float, default=WIPE_DUR, help="matte: wipe duration (s)")
    ap.add_argument("--wipe-grow", type=float, default=WIPE_GROW, help="matte: near-strand width multiplier")
    ap.add_argument("--wipe-base", type=float, default=WIPE_BASE, help="matte: resting near-strand width multiplier")
    ap.add_argument("--wipe-zn", type=float, default=WIPE_ZN, help="matte: full-growth depth")
    ap.add_argument("--wipe-far", type=float, default=WIPE_FAR, help="matte: far-strand growth share")
    ap.add_argument("--wipe-soft", type=float, default=WIPE_SOFT, help="matte: sigma / grown width")
    ap.add_argument("--wipe-light", type=float, default=WIPE_LIGHT, help="matte: light x grow^k")
    ap.add_argument("--wipe-breath", type=float, default=WIPE_BREATH, help="matte: breath floor at full wipe")
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy", help="array backend (xp.py)")
    ap.add_argument("--device", default="cpu", help="torch device: cpu | cuda | cuda:N")
    ap.add_argument("--frame-batch", type=int, default=FRAME_BATCH, help="torch: frames rendered per batch")
    a = ap.parse_args()
    BACKEND, FRAME_BATCH = xp.Backend(a.backend, a.device), a.frame_batch
    WIPE_AT, WIPE_DUR, WIPE_GROW, WIPE_ZN = a.wipe_at, a.wipe_dur, a.wipe_grow, a.wipe_zn
    WIPE_FAR, WIPE_SOFT, WIPE_LIGHT, WIPE_BREATH = a.wipe_far, a.wipe_soft, a.wipe_light, a.wipe_breath
    WIPE_BASE = a.wipe_base
    if a.audio:
        return audio_main(a)
    if a.preview:
        parent = gens()[-1] if gens() else 0
        g, path = free_path(a.gen or next_gen(), "preview")
        render(path, 360, 450, 1, True)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gens as G
        G.backfill(LOCAL, STEM, g, G.now_nz(), f"{STEM} g{parent:02d}", "user: fast CPU preview mode",
                   "--preview: N/4 strands, K/2 trail samples, 1x supersample, 2 blur classes", [os.path.basename(path)], [])
    if a.draft:
        g, path = free_path(a.gen or next_gen(), "draft")
        render(path, 360, 450, 3)
        print("stills", gen_stills(g))
    if a.final:
        g = a.gen or (gens()[-1] if gens() else 1)
        g, path = free_path(g, "final")
        render(path, 1080, 1350, 2)
    if a.gstills:
        print("stills", gen_stills(a.gen or gens()[-1]))
    if a.still is not None:
        os.makedirs(REVIEW, exist_ok=True)
        sc = Silk()
        W, H, ss = (1080, 1350, 2) if a.big else (360, 450, 3)
        for t in a.still:
            t0 = time.time()
            Image.fromarray(sc.frame(t, W, H, ss)).save(os.path.join(REVIEW, f"D_still_{W}_{t:05.2f}.png"))
            print(f"t={t:.2f}  {time.time() - t0:.2f}s")
    if a.seam:
        seam(*((1080, 1350, 2) if a.big else (360, 450, 3)))
    if a.mux:
        g = a.gen or gens()[-1]
        src = os.path.join(LOCAL, f"{STEM}_g{g:02d}_final.mp4")
        wav = os.path.join(LOCAL, f"chords_{STEM}_g{g:02d}.wav")
        if not os.path.exists(wav):
            subprocess.run([sys.executable, os.path.join(REPO, "tools", "mathtype", "chords.py"), "--duration",
                            str(T), "--changes", CHANGES, "--out", wav, "--loop"], check=True)
        out = os.path.join(LOCAL, f"{STEM}_g{g:02d}_final_chords.mp4")
        if os.path.exists(out):
            sys.exit(f"{out} exists; not overwriting")
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", src, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
        print("muxed", out)


if __name__ == "__main__":
    main()
