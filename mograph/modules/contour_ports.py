"""mograph.modules.contour_ports: footage contours as entry / exit / via ports of the type modules (v005, 2026-10-26).

A *port* is a tracked contour of the plate that a word module can be born from (entry), dissolve onto (exit) or that a
transition rides along between two words (via). Three sources, all from files the runner already makes per shot:

    matte_outline  outline of the subject matte (closed, largest blobs)      needs the shot's Alpha-Gen matte
    depth_iso      iso-lines of the depth map, subject inpainted away so they run on behind it (open, floor / back bands)
    luma_edges     iso-lines of the blurred plate luminance (videotrace topo; open or closed)

`ContourTrack` extracts them for every frame and keeps the lines in correspondence over time (same start point, same number of
points, arc-length parameter s in 0..1), so a strand that sits at s = 0.3 stays on the same part of the outline while the dancer
moves. `ContourContext` projects them into the type world through the piece camera: a contour point (x, y) of the plate is the point at
distance `depth` along the camera ray of that pixel at the time t, so it stays pinned to the picture whatever the type camera does.

    ctx = ContourContext(PlateData.from_files(plate="s1.mp4", matte="s1_matte.mkv", depth="s1_depth.mkv", fps=24), world=(540, 960))
    ctx.bind_camera(chain)                                       # CameraChain / anything with .at(t) -> cam3d.Camera
    items = apply_ports(items, word_specs, ctx)                  # words with "entry"/"exit"/"via" port dicts; see README

Strand mechanics reuse the existing hand-off (Bundle strands, `PathTransition` routes, `Slide` windows): a contour ride is a route
`A gate -> onto the contour -> along it -> B gate` whose contour part is re-evaluated on every frame (tracking). Sinks: strands travel and
lie down on the contour (optionally leaving a glowing contour that pulses on beats). Sources: strands lie on the contour and peel
off into the word's own entry. `PortedWord` wraps any gallery word module (entry / exit ports carved out of its form / exit
windows, so the beat grid does not move); `ContourRide` is the transition between two words (`via`).

Only numpy / scipy / cv2: type rendering stays on the box like the rest of the engine."""
from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mograph.modules.core import (Bundle, Layer, Port, Route3, Slide, Timing, WordModule, strand_layers, unit)
from mograph.modules.transitions._path import PathTransition, hermite_curve
from mograph.render import cam3d
from mograph.render.cam3d import Camera, smoother

KINDS = ("matte_outline", "depth_iso", "luma_edges")
DIRS = ("up", "along", "around")
MATTE_MODES = ("front", "window", "fill", "wipe", "glow")        # treatments: tools/fusion/matte_treatments.py
_WORK = 320                                                     # contours are extracted at this long side (px), then scaled back


# ============================================================================================ plate data
class PlateData:
    """per-frame plate / subject matte / depth of one shot (or shots joined in time): uint8 arrays [T,H,W(,3)], one fps."""

    def __init__(self, matte=None, depth=None, plate=None, fps=24.0):
        self.matte, self.depth, self.plate, self.fps = matte, depth, plate, float(fps)
        ref = next((a for a in (plate, matte, depth) if a is not None), None)
        if ref is None:
            raise ValueError("PlateData needs at least one of plate / matte / depth")
        self.n = min(len(a) for a in (plate, matte, depth) if a is not None)
        self.size = (int(ref.shape[2]), int(ref.shape[1]))            # (w, h)

    def has(self, kind):
        return {"matte_outline": self.matte is not None, "depth_iso": self.depth is not None,
                "luma_edges": self.plate is not None}[kind]

    def frame(self, kind, f):
        a = {"matte_outline": self.matte, "depth_iso": self.depth, "luma_edges": self.plate}[kind]
        return a[int(min(max(f, 0), self.n - 1))]

    @staticmethod
    def read(path, gray=False, size=None, max_frames=None):
        """video / image directory -> uint8 [T,H,W] (gray) or [T,H,W,3] RGB."""
        out = []
        if os.path.isdir(path):
            names = sorted(os.listdir(path), key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
            srcs = (cv2.imread(os.path.join(path, n), cv2.IMREAD_COLOR) for n in names if n.lower().endswith((".png", ".jpg")))
        else:
            cap = cv2.VideoCapture(path)

            def gen():
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    yield fr
            srcs = gen()
        for fr in srcs:
            if size is not None and (fr.shape[1], fr.shape[0]) != tuple(size):
                fr = cv2.resize(fr, tuple(size), interpolation=cv2.INTER_AREA)
            out.append(fr[..., 0] if gray else fr[..., ::-1])
            if max_frames and len(out) >= max_frames:
                break
        if not out:
            raise ValueError(f"no frames read from {path}")
        return np.ascontiguousarray(np.stack(out))

    @classmethod
    def from_files(cls, plate=None, matte=None, depth=None, fps=24.0, size=None, max_frames=None):
        kw = dict(size=size, max_frames=max_frames)
        return cls(matte=None if matte is None else cls.read(matte, True, **kw),
                   depth=None if depth is None else cls.read(depth, True, **kw),
                   plate=None if plate is None else cls.read(plate, False, **kw), fps=fps)

    @classmethod
    def concat(cls, parts):
        """shots joined in time (every part needs the same channels and size)."""
        def cat(name):
            xs = [getattr(p, name) for p in parts]
            return None if any(x is None for x in xs) else np.concatenate([x[:p.n] for x, p in zip(xs, parts)])
        return cls(matte=cat("matte"), depth=cat("depth"), plate=cat("plate"), fps=parts[0].fps)


def available(piece, shot=None):
    """which contour ports a piece (the P*.json dict) supports: {"ports": [kinds usable on every shot], "shots": {sid: {kind: bool}},
    "matte": [...] }. luma_edges needs only the footage (always); matte_outline needs the shot's Alpha-Gen matte ("matte": true);
    depth_iso needs its Video-Depth-Anything pass ("depth": true). A PlateData answers for its own channels. `shot` limits it to one shot id."""
    if isinstance(piece, PlateData):
        return {"ports": [k for k in KINDS if piece.has(k)], "shots": {"-": {k: piece.has(k) for k in KINDS}}, "matte": list(MATTE_MODES)}
    shots = piece.get("shots") or []
    out = {}
    for s in shots:
        if shot is not None and s.get("id") != shot:
            continue
        out[s.get("id", str(len(out)))] = {"luma_edges": True, "matte_outline": bool(s.get("matte")), "depth_iso": bool(s.get("depth"))}
    common = [k for k in KINDS if out and all(v[k] for v in out.values())]
    return {"ports": common, "shots": out, "matte": list(MATTE_MODES) if any(v["matte_outline"] for v in out.values()) else ["window", "glow", "wipe"]}


# ============================================================================================ extraction (cv2)
def _find(a, t, pad=2):
    """contours of a > t as (n,2) float32 (x, y) px; the frame is replicate-padded so regions cut by the frame edge trace
    along the pad ring (off-canvas), not along the edge."""
    b = (cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_REPLICATE) > t).astype(np.uint8)
    cnts = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    return [c[:, 0, :].astype(np.float32) - pad for c in cnts if len(c) >= 8]


def _arc(p, closed):
    q = np.vstack([p, p[:1]]) if closed else p
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]


def _smooth(p, k, closed):
    if k <= 0 or len(p) < 2 * k + 3:
        return p
    ker = np.ones(2 * k + 1, np.float32) / (2 * k + 1)
    if closed:
        q = np.concatenate([p[-k:], p, p[:k]])
        return np.stack([np.convolve(q[:, i], ker, mode="valid") for i in range(2)], 1)
    q = np.concatenate([np.repeat(p[:1], k, 0), p, np.repeat(p[-1:], k, 0)])
    return np.stack([np.convolve(q[:, i], ker, mode="valid") for i in range(2)], 1)


def _resample(p, n, closed):
    """n points uniform in arc length (closed: the first point is not repeated)."""
    cum = _arc(p, closed)
    q = np.vstack([p, p[:1]]) if closed else p
    if cum[-1] < 1e-6:
        return np.repeat(p[:1], n, 0)
    s = np.linspace(0, cum[-1], n, endpoint=not closed)
    return np.stack([np.interp(s, cum, q[:, 0]), np.interp(s, cum, q[:, 1])], 1).astype(np.float32)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))   # > 0: clockwise on screen (y down)


def _open_runs(cnt, w, h, margin=1.0):
    """split a closed pixel contour into the runs that lie inside the canvas (the iso-line proper, not the pad ring)."""
    ok = (cnt[:, 0] >= margin) & (cnt[:, 0] <= w - 1 - margin) & (cnt[:, 1] >= margin) & (cnt[:, 1] <= h - 1 - margin)
    if ok.all():
        return [(cnt, True)]
    if not ok.any():
        return []
    k = int(np.argmin(ok))                                  # start at an outside point so no run wraps
    c, ok = np.roll(cnt, -k, 0), np.roll(ok, -k)
    runs, cur = [], []
    for p, o in zip(c, ok):
        if o:
            cur.append(p)
        elif cur:
            runs.append((np.array(cur, np.float32), False)); cur = []
    if cur:
        runs.append((np.array(cur, np.float32), False))
    return runs


def _work_scale(h, w):
    return min(1.0, _WORK / float(max(h, w)))


def _prep(img, f, sigma):
    img = np.asarray(img, np.float32)
    if f < 1.0:
        img = cv2.resize(img, (max(8, int(round(img.shape[1] * f))), max(8, int(round(img.shape[0] * f)))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (0, 0), sigma * max(f, 0.5)) if sigma > 0 else img


def matte_outlines(matte_u8, thresh=0.5, max_lines=1, smooth=2.0, min_len=40.0):
    """outlines of the matte > thresh, largest first: [(pts (n,2) in matte px, clockwise on screen)]."""
    h, w = matte_u8.shape[:2]
    f = _work_scale(h, w)
    a = _prep(matte_u8.astype(np.float32) / 255.0, f, smooth)
    out = []
    for c in _find(a, thresh):
        c = _smooth(c, 3, True)
        if _arc(c, True)[-1] < min_len * f:
            continue
        if _area(c) < 0:
            c = c[::-1].copy()
        out.append((abs(_area(c)), c / f))
    out.sort(key=lambda x: -x[0])
    return [c for _, c in out[:max_lines]]


def iso_lines(img, levels, min_len=40.0, smooth=2.0, keep=4):
    """iso-lines of img (float 0..1) at every level -> per level the `keep` longest [(pts in img px, closed)], longest first
    (open lines run left to right, closed ones clockwise on screen)."""
    h, w = img.shape[:2]
    f = _work_scale(h, w)
    a = _prep(img, f, smooth)
    hh, ww = a.shape
    out = []
    for t in levels:
        cands = []
        for cnt in _find(a, t):
            for run, closed in _open_runs(cnt, ww, hh):
                run = _smooth(run, 2, closed)
                L = _arc(run, closed)[-1]
                if L < min_len * f:
                    continue
                if closed:
                    if _area(run) < 0:
                        run = run[::-1].copy()
                elif run[0, 0] > run[-1, 0]:
                    run = run[::-1].copy()
                cands.append((L, run / f, closed))
        cands.sort(key=lambda c: -c[0])
        out.append([(r, c) for _, r, c in cands[:keep]])
    return out


def _pick_run(cands, prev, n):
    """the candidate line that continues `prev` (resampled (n,2), None at the first frame): the longest at first, then the nearest
    one among the long ones (so the line a strand sits on does not jump to another feature)."""
    if not cands:
        return None
    if prev is None:
        return cands[0]
    best, bd = None, 1e30
    for run, closed in cands:
        q = _resample(run, n, closed)
        d = float(np.linalg.norm(q - prev, axis=1).mean())
        if d < bd:
            best, bd = (run, closed), d
    return best


def depth_levels(depth_u8, matte_u8=None, n=5, lo_q=12.0, hi_q=88.0):
    """n iso levels (0..1) between two percentiles of the depth map outside the subject."""
    d = depth_u8.astype(np.float32) / 255.0
    if matte_u8 is not None:
        d = d[matte_u8 < 100]
    d = d.reshape(-1)
    a, b = np.percentile(d, lo_q), np.percentile(d, hi_q)
    return list(np.linspace(a, b, n + 2)[1:-1])


def inpaint_subject(depth_u8, matte_u8, grow=5):
    """depth with the subject (matte, grown) painted over from the surroundings, so floor / back iso-lines run on behind it."""
    h, w = depth_u8.shape[:2]
    f = _work_scale(h, w)
    d = cv2.resize(depth_u8, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else depth_u8
    m = cv2.resize(matte_u8, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA) if f < 1 else matte_u8
    mk = cv2.dilate((m > 60).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(grow) + 1,) * 2))
    out = cv2.inpaint(d, mk, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) if f < 1 else out


# ============================================================================================ tracking
def _sample_line(P, s, closed):
    """P (N,2) line, s (n,) parameter in 0..1 -> (n,2) (closed wraps, open clamps)."""
    N = len(P)
    s = np.asarray(s, float)
    if closed:
        x = (s % 1.0) * N
        i0 = np.floor(x).astype(int) % N
        i1 = (i0 + 1) % N
        fr = (x - np.floor(x))[:, None]
    else:
        x = np.clip(s, 0.0, 1.0) * (N - 1)
        i0 = np.minimum(np.floor(x).astype(int), N - 2)
        i1 = i0 + 1
        fr = (x - i0)[:, None]
    return P[i0] * (1 - fr) + P[i1] * fr


class ContourTrack:
    """every frame's contour lines, in correspondence over time. P (F, L, N, 2) float32 in plate px, valid (F, L), closed (L,)."""

    def __init__(self, kind, size, fps, P, valid, closed, levels=None):
        self.kind, self.size, self.fps = kind, tuple(size), float(fps)
        self.P, self.valid, self.closed, self.levels = P, valid, np.asarray(closed, bool), levels
        self.F, self.L, self.N = P.shape[:3]

    def lines_at(self, f):
        """(L, N, 2) at the (fractional) frame f, linear in time."""
        f = min(max(float(f), 0.0), self.F - 1.0)
        i0 = int(math.floor(f))
        i1 = min(i0 + 1, self.F - 1)
        a = f - i0
        return self.P[i0] * (1 - a) + self.P[i1] * a if a > 1e-6 and i1 != i0 else self.P[i0]

    def sample(self, f, line, s):
        """points (n,2) at frame f of lines `line` (int or (n,) ints) at arc parameters s (n,)."""
        s = np.asarray(s, float)
        L = self.lines_at(f)
        line = np.broadcast_to(np.asarray(line, int), s.shape)
        out = np.zeros((len(s), 2), np.float32)
        for l in np.unique(line):
            m = line == l
            out[m] = _sample_line(L[l], s[m], bool(self.closed[l]))
        return out

    def tangent(self, f, line, s, eps=0.01):
        s = np.asarray(s, float)
        a = self.sample(f, line, s - eps)
        b = self.sample(f, line, s + eps)
        d = b - a
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-6)

    def length(self, f, line):
        L = self.lines_at(f)[line]
        return float(_arc(L, bool(self.closed[line]))[-1])

    def valid_lines(self, f):
        return [l for l in range(self.L) if self.valid[int(min(max(round(f), 0), self.F - 1)), l]]


def _align_closed(prev, cur):
    """roll cur so that point i is the nearest correspondence of prev's point i (min sum of squared distances over the shifts)."""
    N = len(cur)
    best, bs = 1e30, 0
    for s in range(0, N, 1):
        d = ((np.roll(cur, -s, 0) - prev) ** 2).sum()
        if d < best:
            best, bs = d, s
    return np.roll(cur, -bs, 0)


def build_track(data, kind, f0=0, f1=None, n=192, max_lines=1, levels=5, thresh=0.5, smooth=2.0, min_len=40.0, tsmooth=1):
    """ContourTrack of `kind` for frames f0..f1 of PlateData. matte_outline: `max_lines` largest blobs; depth_iso / luma_edges:
    `levels` iso-lines of the depth / luminance map (fixed levels from the first frame, so line l is the same band all clip long)."""
    if kind not in KINDS:
        raise ValueError(f"unknown contour port {kind!r}; one of {KINDS}")
    if not data.has(kind):
        raise ValueError(f"plate has no data for {kind!r}")
    f1 = data.n - 1 if f1 is None else min(f1, data.n - 1)
    fr = list(range(f0, f1 + 1))
    F = len(fr)
    if kind == "matte_outline":
        L, closed = max_lines, [True] * max_lines
    else:
        L, closed = levels, [False] * levels
    P = np.zeros((F, L, n, 2), np.float32)
    valid = np.zeros((F, L), bool)
    lv = None
    if kind == "depth_iso":
        m0 = data.matte[f0] if data.matte is not None else None
        lv = depth_levels(data.depth[f0], m0, levels)
    elif kind == "luma_edges":
        g0 = cv2.cvtColor(data.plate[f0], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        a, b = np.percentile(g0, 14), np.percentile(g0, 92)
        lv = list(np.linspace(a, b, levels + 2)[1:-1])
    prev = [None] * L
    prev_c = [None] * L
    sx = data.size[0] / float(data.matte.shape[2]) if kind == "matte_outline" else 1.0
    for fi, f in enumerate(fr):
        found = [None] * L
        if kind == "matte_outline":
            cs = matte_outlines(data.matte[f], thresh, max_lines, smooth, min_len)
            used = set()
            for l in range(L):
                if not cs:
                    break
                if prev_c[l] is None:
                    j = next((j for j in range(len(cs)) if j not in used), None)
                else:
                    cand = [(np.linalg.norm(cs[j].mean(0) - prev_c[l]), j) for j in range(len(cs)) if j not in used]
                    j = min(cand)[1] if cand else None
                if j is None:
                    continue
                used.add(j)
                found[l] = (cs[j], True)
        else:
            if kind == "depth_iso":
                img = inpaint_subject(data.depth[f], data.matte[f]) if data.matte is not None else data.depth[f].astype(np.float32) / 255.0
            else:
                img = cv2.cvtColor(data.plate[f], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            res = iso_lines(img, lv, min_len, smooth)
            found = [_pick_run(res[l], prev[l], n) or (None, False) for l in range(L)]
        for l in range(L):
            if found[l] is None or found[l][0] is None:
                if prev[l] is not None:
                    P[fi, l] = prev[l]
                continue
            pts, cl = found[l]
            pts = _resample(pts, n, cl)
            if cl:
                if fi == 0 or prev[l] is None:
                    k = int(np.argmax(pts[:, 1]))                    # start at the lowest point: s = 0 bottom, 0.5 top
                    pts = np.roll(pts, -k, 0)
                else:
                    pts = _align_closed(prev[l], pts)
            if prev[l] is not None and not cl and np.linalg.norm(pts[0] - prev[l][0]) > np.linalg.norm(pts[-1] - prev[l][0]) * 2.5:
                pts = pts[::-1].copy()
            P[fi, l] = pts
            valid[fi, l] = True
            prev[l] = pts
            prev_c[l] = pts.mean(0)
    # all contours are in plate px of the *source* array: matte / depth / luma may have other sizes -> rescale to data.size
    src = {"matte_outline": data.matte, "depth_iso": data.depth, "luma_edges": data.plate}[kind]
    k = data.size[0] / float(src.shape[2])
    if abs(k - 1.0) > 1e-6:
        P = P * k
    if tsmooth > 0 and F > 2 * tsmooth + 1:
        P = _tsmooth(P, valid, tsmooth)
    return ContourTrack(kind, data.size, data.fps, P, valid, closed, lv)


def _tsmooth(P, valid, k):
    """moving average over 2k+1 frames (valid frames only) so the contour does not jitter."""
    out = P.copy()
    F = len(P)
    for l in range(P.shape[1]):
        for f in range(F):
            lo, hi = max(0, f - k), min(F, f + k + 1)
            sel = [g for g in range(lo, hi) if valid[g, l]]
            if sel and valid[f, l]:
                out[f, l] = P[sel, l].mean(0)
    return out


# ============================================================================================ context (projection into the type world)
class ContourContext:
    """plate contours + the projection into the type world. `world` = (w, h) of the frame plane (cam3d.set_world), `depth` = distance
    along the camera ray at which the contour lives (the strands' line widths scale with F / depth). `cam_at(t)` = the piece camera
    (a CameraChain / anything with .at(t) -> cam3d.Camera, or a callable): without one the contour is lifted into the home view."""

    def __init__(self, data, world=None, F=1000.0, depth=1000.0, n=192, fps=None, beats=None):
        self.data, self.F, self.depth, self.n = data, float(F), float(depth), int(n)
        self.world = tuple(world) if world else (cam3d.W, cam3d.H)
        self.fps = float(fps or data.fps)
        self.beats = None if beats is None else np.asarray(beats, float)
        self._tracks, self.cam = {}, None
        self.k = self.world[0] / float(data.size[0])                       # plate px -> world frame px

    def bind_camera(self, cam):
        self.cam = cam
        return self

    def camera(self, t):
        if self.cam is None:
            cx, cy = self.world[0] / 2.0, self.world[1] / 2.0
            return Camera((cx, cy, -self.F), (cx, cy, 0.0), 0.0, self.F)
        return self.cam(t) if callable(self.cam) else self.cam.at(t)

    def track(self, kind, **kw):
        key = (kind,) + tuple(sorted(kw.items()))
        if key not in self._tracks:
            self._tracks[key] = build_track(self.data, kind, n=self.n, **kw)
        return self._tracks[key]

    def frame_of(self, t):
        return float(t) * self.fps

    def set_plane(self, z, ref):
        """multiplane plates: contours live ON the card plane z (a world-fixed plane of the type world; `ref` = the plate's reference camera
        distance, see mograph.render.multiplane) instead of being pinned to the picture, so they parallax with the cards under any camera."""
        self.plane = (float(z), float(ref))
        return self

    plane = None

    def lift(self, xy, cam, depth=None):
        """plate px (n,2) -> world (n,3): the point at `depth` along the camera ray of that pixel (or on the card plane, see set_plane)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.plane is not None:
            z, ref = self.plane
            s = (ref + z) / self.F
            cx, cy = self.world[0] / 2.0, self.world[1] / 2.0
            x, y = xy[:, 0] * self.k, xy[:, 1] * self.k
            return np.stack([cx + (x - cx) * s, cy + (y - cy) * s, np.full(len(xy), z)], 1)
        d = self.depth if depth is None else depth
        x = xy[:, 0] * self.k - self.world[0] / 2.0
        y = xy[:, 1] * self.k - self.world[1] / 2.0
        Q = np.stack([d * x / cam.F, d * y / cam.F, np.full(len(xy), float(d))], 1)
        return cam.eye + Q @ cam.R


# ============================================================================================ port spec
def _smooth01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Layout:
    """where the K strands of a ride sit on the contour (parameters 0..1 along their line)."""
    line: np.ndarray       # (K,) int
    s0: np.ndarray         # (K,) strand head at the contour end of the ride's lying pose
    step: np.ndarray       # (K,) +1 / -1 direction of travel along s
    travel: np.ndarray     # (K,) arc (s units) ridden between the lying pose and the leave / join point
    slen: float            # strand length (s units)


class ContourPort:
    """a contour + how strands use it. Build with ContourPort.parse(ctx, spec) from the piece JSON dict:
        {"port": "matte_outline"|"depth_iso"|"luma_edges", "dir": "up"|"along"|"around", "lead": 0.6 (s, ride duration), "travel": 1.0
         (x default arc), "strands": null (= the word's strand count), "strand_len": 0.16, "lines": null (all) | int | [ints],
         "levels": 5 (iso lines), "glow": 0.0..1 (contour line left glowing), "persist": 0.0 (s the glow stays after the strands lie
         down), "pulse": null | bpm | "beat" (glow pulses), "width": 1.5, "colour": null (palette fg)}"""

    def __init__(self, ctx, kind, dir="along", lead=0.6, travel=1.0, strand_len=0.16, lines=None, levels=5, glow=0.0, persist=0.0,
                 pulse=None, width=1.5, colour=None, strands=None, thresh=0.5, max_lines=1, smooth=2.0):
        if kind not in KINDS:
            raise ValueError(f"unknown contour port {kind!r}; one of {KINDS}")
        if dir not in DIRS:
            raise ValueError(f"unknown ride direction {dir!r}; one of {DIRS}")
        self.ctx, self.kind, self.dir = ctx, kind, dir
        self.lead, self.travel, self.slen = float(lead), float(travel), float(strand_len)
        self.sel, self.levels, self.glow, self.persist, self.pulse = lines, int(levels), float(glow), float(persist), pulse
        self.width, self.colour, self.strands = float(width), colour, strands
        self.tkw = dict(thresh=thresh, max_lines=max_lines, smooth=smooth) if kind == "matte_outline" else dict(levels=int(levels), smooth=smooth)

    @classmethod
    def parse(cls, ctx, spec, default_dir="along"):
        if spec is None:
            return None
        if isinstance(spec, ContourPort):
            return spec
        d = {"port": spec} if isinstance(spec, str) else dict(spec)
        kind = d.pop("port")
        return cls(ctx, kind, dir=d.pop("dir", default_dir), **{k: v for k, v in d.items() if k in (
            "lead", "travel", "strand_len", "lines", "levels", "glow", "persist", "pulse", "width", "colour", "strands", "thresh", "max_lines", "smooth")})

    @property
    def track(self):
        return self.ctx.track(self.kind, **self.tkw)

    def lines(self, f):
        tr = self.track
        ok = tr.valid_lines(f) or list(range(tr.L))
        if self.sel is None:
            return ok
        sel = [self.sel] if isinstance(self.sel, int) else list(self.sel)
        return [l for l in sel if l in range(tr.L)] or ok

    # ---------------------------------------------------------------- layout
    def layout(self, K, f):
        """strand placement at frame f. s0 = where a strand's head lies when it is on the contour (the start of an entry / via ride, the
        end of an exit ride). Closed line: s = 0 is the lowest point, 0.5 the top, s grows clockwise on screen (the left side goes up).
        dir 'up': strands ride up both sides to the top; 'along': clockwise (open lines: left to right); 'around': half a turn
        (open lines: serpentine, neighbouring lines run in opposite directions)."""
        tr = self.track
        ls = self.lines(f)
        K = int(K)
        line = np.array([ls[k % len(ls)] for k in range(K)], int)
        rank = np.array([k // len(ls) for k in range(K)], int)
        cnt = np.array([int(np.sum(line == l)) for l in line], int)
        frac = (rank + 0.5) / np.maximum(cnt, 1)
        s0, step, travel = np.zeros(K), np.ones(K), np.zeros(K)
        sl, ts = self.slen, self.travel
        for li, l in enumerate(ls):
            m = line == l
            fr = frac[m]
            if bool(tr.closed[l]):
                if self.dir == "up":
                    span = 0.5 - sl - 0.05
                    left = fr < 0.5
                    s = np.where(left, sl + 0.02 + span * (2 * fr), 1.0 - sl - 0.02 - span * (2 * (fr - 0.5)))
                    st = np.where(left, 1.0, -1.0)
                    d = np.maximum(np.abs(s - 0.5) - 0.03, 0.0)
                elif self.dir == "around":
                    s, st, d = fr, np.ones_like(fr), np.full_like(fr, 0.5)
                else:
                    s, st, d = fr, np.ones_like(fr), np.full_like(fr, 0.3)
            else:
                P = tr.lines_at(f)[l]
                if self.dir == "up":
                    sgn = 1.0 if P[-1, 1] < P[0, 1] else -1.0          # toward the end that is higher on screen
                elif self.dir == "around":
                    sgn = 1.0 if li % 2 == 0 else -1.0
                else:
                    sgn = 1.0
                lo, hi = sl + 0.03, 1.0 - sl - 0.03
                if sgn > 0:
                    s = lo + (0.55 - lo) * fr
                    d = np.minimum(0.45, 1.0 - 0.03 - s)
                else:
                    s = hi - (hi - 0.45) * fr
                    d = np.minimum(0.45, s - 0.03)
                st = np.full_like(fr, sgn)
            s0[m], step[m], travel[m] = s, st, d * ts
        return Layout(line, s0, step, travel, sl)

    # ---------------------------------------------------------------- geometry of the lying strands
    def lying(self, lay, f, cam, t, kp=7, tail_gate=False):
        """world points of the K lying strands (head first) at frame f: (K, kp, 3), the plate-px points (K, kp, 2), their s."""
        K = len(lay.line)
        j = np.linspace(0.0, 1.0, kp)[None, :]
        s = lay.s0[:, None] - lay.step[:, None] * lay.slen * j
        tr = self.track
        px = np.zeros((K, kp, 2))
        for l in np.unique(lay.line):
            m = lay.line == l
            px[m] = tr.sample(f, int(l), s[m].reshape(-1)).reshape(-1, kp, 2)
        W = self.ctx.lift(px.reshape(-1, 2), cam).reshape(K, kp, 3)
        return W, px, s

    def color(self, pal=None):
        if self.colour is not None:
            c = self.colour
            return tuple(int(round(float(x) * (255 if max(c) <= 1.0 else 1))) for x in c)
        return tuple(pal.fg) if pal is not None else (236, 230, 218)


# ============================================================================================ the ride (transition)
def _curve(p0, p1, d0, d1, stiff=1.0):
    dist = float(np.linalg.norm(p1 - p0))
    m = stiff * (0.9 * dist + 60.0)
    n = max(10, int(dist / 6.0))
    return hermite_curve(p0, p1, unit(d0) * m, unit(d1) * m, np.linspace(0.0, 1.0, n))


def _arc_world(ctx, tr, cam, f, line, s_from, s_to):
    """world points of the contour arc s_from -> s_to (about every 5 px) and their parameters."""
    n = max(4, int(abs(s_to - s_from) * tr.length(f, line) / 5.0))
    s = np.linspace(s_from, s_to, n)
    return ctx.lift(tr.sample(f, line, s), cam), s


def _world_at(ctx, tr, cam, f, line, s):
    return ctx.lift(tr.sample(f, line, np.array([s], float)), cam)[0]


class _TrackedRoute:
    """a route whose points moved with the contour. Slides keep their arc lengths from build time; this maps them onto the moved
    polyline (piecewise linear between the old and the new cumulative lengths), so a window keeps covering the same part of the route."""

    def __init__(self, P, old_cum):
        R = Route3.__new__(Route3)
        R.P = P
        R.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        R.L = float(R.cum[-1])
        self.R, self.old, self.L = R, old_cum, float(old_cum[-1])

    def _m(self, s):
        return float(np.interp(s, self.old, self.R.cum))

    def at(self, s):
        return self.R.at(self._m(s))

    def window(self, a, b):
        return self.R.window(self._m(max(a, 0.0)), self._m(min(b, self.L)))

    def tangent_at(self, s):
        return self.R.tangent_at(self._m(s))


class ContourRide(PathTransition):
    """strands travel A gate -> contour -> along it -> B gate (`via`), start lying on the contour and peel off into B (`entry`), or
    travel to the contour and lie down on it (`exit`). The contour parts are re-evaluated every frame from the plate track (so the
    strands stay on the moving outline) and projected through the piece camera of that moment.

    params: ctx (ContourContext), port (ContourPort), mode 'via' | 'entry' | 'exit', t_abs (piece time of the ride's start, s),
    stretch (body lengthening mid-way, 0.15), fade (glow line fades out over the last share of an entry ride, 0.35).
    Not in the engine's pool (it needs a ContourContext): build it with `via_ride` / `entry_ride` / `exit_ride` or `apply_ports`."""
    name = "contour_ride"
    description = ("Strands leave the word, ride along a tracked contour of the footage (subject outline, depth band, luminance edge) "
                   "and form the next word from it; the contour keeps moving with the picture.")
    SUITS = None

    def __init__(self, a_exit=None, b_entry=None, dur=None, **params):
        self.ctx = params.get("ctx")
        self.port = params.get("port")
        self.mode = params.get("mode", "via")
        self.t_abs = float(params.get("t_abs", 0.0))
        self._ck = None
        super().__init__(a_exit, b_entry, dur, **params)

    # ---------------------------------------------------------------- timing
    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        d = float(np.linalg.norm(b_entry.centroid() - a_exit.centroid())) * 1.2
        return float(min(max(1.7 * d / 520.0, 0.8), 2.6))

    def t_snap(self):
        return {"entry": 0.0, "exit": self.dur, "via": 0.5 * self.dur}[self.mode]

    # ---------------------------------------------------------------- pairing
    def pair(self):
        """entry / exit: Hungarian pairing on the world distance between A's heads and B's tails (squared distances: the streams do not
        cross); via: the usual `adapt`. self.slot[k] = index of the contour slot (virtual strand) of pair k for entry / exit."""
        a, b = self.a_exit, self.b_entry
        self.slot = None
        if self.mode == "via":
            from mograph.modules.core import adapt
            pr = adapt(a, b)
            A, B = pr.bundles(a, b)
        else:
            cost = ((a.heads()[:, None, :] - b.tails()[None, :, :]) ** 2).sum(-1)
            ri, ci = linear_sum_assignment(cost)
            A, B = a.subset(ri), b.subset(ci)
            self.slot = ri if self.mode == "entry" else ci
        self.A, self.B, self.K = A, B, A.N
        self.tube = max(A.tube, B.tube)
        return A, B

    # ---------------------------------------------------------------- build
    def build(self):
        ctx, port, mode = self.ctx, self.port, self.mode
        A, B = self.pair()
        K = self.K
        ts = self.t_snap()
        f_snap = ctx.frame_of(self.t_abs + ts)
        cam = ctx.camera(self.t_abs + ts)
        lay = port.layout(K, f_snap)
        tr = port.track
        if mode == "via":                                         # contour slot of strand k: left to right on screen, as the gates are
            W0, _, _ = port.lying(lay, f_snap, cam, ts)
            order_c = np.argsort(W0[:, 0, :] @ cam.r)
            order_g = np.argsort(cam.project(A.heads())[0][:, 0])
            slot = np.zeros(K, int)
            slot[order_g] = order_c
        else:
            slot = self.slot
        self.lay, self.slot_used = lay, slot
        self.routes, self.slides, self.anchors, self.pts = [], [], [], []
        T = self.dur
        sl = lay.slen
        ar = lambda l, s_from, s_to: _arc_world(ctx, tr, cam, f_snap, l, s_from, s_to)
        ramp = lambda n: _smooth01(np.linspace(0, 1, n))
        for k in range(K):
            c = int(slot[k])
            l, st, s_h, trv = int(lay.line[c]), float(lay.step[c]), float(lay.s0[c]), float(lay.travel[c])
            body_a, body_b = A.P[k][::-1], B.P[k][::-1]            # tail -> head
            na, nb = len(body_a), len(body_b)
            nan = lambda n: np.full(n, np.nan)
            if mode == "entry":
                body_s = s_h - st * sl * np.linspace(1.0, 0.0, na)
                Wa, sa_ = ar(l, s_h, s_h + st * trv)
                d_end = unit(_world_at(ctx, tr, cam, f_snap, l, sa_[-1] + 0.02 * st) - Wa[-1])
                conn = _curve(Wa[-1], body_b[0], d_end, B.D[k])
                full = [body_a, Wa[1:], conn[1:-1], body_b]
                S = [body_s, sa_[1:], nan(len(conn) - 2), nan(nb)]
                Wt = [np.ones(na), np.ones(len(Wa) - 1), 1.0 - ramp(len(conn))[1:-1], np.zeros(nb)]
            elif mode == "exit":
                s_tail = s_h - st * sl
                s_join = s_tail - st * trv
                s_join = float(np.clip(s_join, 0.0, 1.0)) if (not tr.closed[l] or port.dir == "up") else s_join
                Wa, sa_ = ar(l, s_join, s_tail)
                d0 = unit(_world_at(ctx, tr, cam, f_snap, l, s_join + 0.02 * st) - Wa[0])
                conn = _curve(body_a[-1], Wa[0], A.D[k], d0)
                body_s = s_h - st * sl * np.linspace(1.0, 0.0, nb)
                full = [body_a, conn[1:-1], Wa, body_b[1:]]
                S = [nan(na), nan(len(conn) - 2), sa_, body_s[1:]]
                Wt = [np.zeros(na), ramp(len(conn))[1:-1], np.ones(len(Wa)), np.ones(nb - 1)]
            else:
                Wa, sa_ = ar(l, s_h, s_h + st * trv)
                d0 = unit(Wa[min(2, len(Wa) - 1)] - Wa[0])
                d1 = unit(Wa[-1] - Wa[max(len(Wa) - 3, 0)])
                c1, c2 = _curve(body_a[-1], Wa[0], A.D[k], d0), _curve(Wa[-1], body_b[0], d1, B.D[k])
                full = [body_a, c1[1:-1], Wa, c2[1:-1], body_b]
                S = [nan(na), nan(len(c1) - 2), sa_, nan(len(c2) - 2), nan(nb)]
                Wt = [np.zeros(na), ramp(len(c1))[1:-1], np.ones(len(Wa)), 1.0 - ramp(len(c2))[1:-1], np.zeros(nb)]
            P, S, Wt = np.vstack(full), np.concatenate(S), np.concatenate(Wt)
            n0 = len(P)
            keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
            P, S, Wt = P[keep], S[keep], Wt[keep]
            ix = np.cumsum(keep) - 1                               # index map old -> kept
            R = Route3(P)
            i_head, i_tail = int(ix[na - 1]), int(ix[n0 - nb])    # A's head, B's tail on the route
            La, Lt = float(R.cum[i_head]), float(R.cum[i_tail])
            sl_ = Slide(R, La, R.L, 0.0, Lt, T, float(A.v[k]), float(B.v[k]), k=1.6)
            sl_.tail_lag = float(self.params.get("stretch", 0.15))
            self.routes.append(R)
            self.slides.append(sl_)
            self.pts.append(P)
            self.anchors.append((np.full(len(P), l, int), _fill_nan_nearest(S, Wt), Wt))
        self.axis = np.concatenate([p[::4] for p in self.pts]) if self.pts else np.zeros((0, 3))
        self._cum0 = [R.cum.copy() for R in self.routes]
        # base anchor positions (what the route was built from): tracking moves anchored points by lift(now) - base
        self._base = [self._anchor_world(f_snap, cam, a) for a in self.anchors]
        self._ck = None

    def _anchor_world(self, f, cam, anc):
        l, S, Wt = anc
        out = np.zeros((len(S), 3))
        m = Wt > 0
        if m.any():
            tr = self.port.track
            px = np.zeros((int(m.sum()), 2))
            idx = np.where(m)[0]
            px = tr.sample(f, l[idx], S[idx])
            out[idx] = self.ctx.lift(px, cam)
        return out

    # ---------------------------------------------------------------- per frame tracking
    def _track(self, t):
        key = int(round(t * 600))
        if self._ck == key:
            return
        ctx = self.ctx
        ta = self.t_abs + t
        f, cam = ctx.frame_of(ta), ctx.camera(ta)
        for k, sl in enumerate(self.slides):
            P0, anc, base = self.pts[k], self.anchors[k], self._base[k]
            cur = self._anchor_world(f, cam, anc)
            w = anc[2][:, None]
            P = P0 + w * (cur - base)
            R = _TrackedRoute(P, self._cum0[k])
            sl.R = R
            self.routes[k] = R
        self._ck = key

    def windows(self, t):
        self._track(t)
        return [sl.window(t) for sl in self.slides]

    # ---------------------------------------------------------------- glow line + layers
    def glow_layers(self, t, env=1.0, pal=None):
        port = self.port
        if port.glow <= 0 or env <= 0.004:
            return []
        ctx = self.ctx
        ta = self.t_abs + t
        f, cam = ctx.frame_of(ta), ctx.camera(ta)
        tr = port.track
        L = tr.lines_at(f)
        ls = port.lines(f)
        polys = []
        for l in ls:
            polys.append(ctx.lift(L[l], cam))
        pul = 1.0
        if port.pulse:
            pul = _pulse(ctx, ta, port.pulse)
        col = port.color(pal)
        a = float(np.clip(port.glow * env * pul, 0, 1))
        closed = [bool(tr.closed[l]) for l in ls]
        out = []
        for cl in (True, False):
            sel = [p for p, c in zip(polys, closed) if c == cl]
            if not sel:
                continue
            out.append(Layer(sel, col, "line", port.width * (3.2 + 1.2 * pul), closed=cl, alpha=a * 0.22, bias=-0.5))
            out.append(Layer(sel, col, "line", port.width * 1.1, closed=cl, alpha=min(1.0, a), bias=-0.4))
        return out

    def env(self, t):
        u = min(max(t / max(self.dur, 1e-6), 0.0), 1.0)
        if self.mode == "exit":
            return float(_smooth01(u / 0.6))
        if self.mode == "entry":
            return float(1.0 - _smooth01((u - 0.3) / 0.5))
        return float(math.sin(math.pi * u) ** 0.7)

    def layers(self, t):
        pal = self.a_word.palette if getattr(self, "a_word", None) is not None else (
            self.b_word.palette if getattr(self, "b_word", None) is not None else None)
        base = super().layers(t)
        return self.glow_layers(t, self.env(t), pal) + base


def _fill_nan_nearest(S, W):
    """connector points carry no parameter of their own: copy the nearest contour parameter along the route."""
    S = S.copy()
    ok = ~np.isnan(S)
    if ok.all():
        return S
    if not ok.any():
        return np.zeros_like(S)
    idx = np.where(ok)[0]
    for i in np.where(~ok)[0]:
        S[i] = S[idx[np.argmin(np.abs(idx - i))]]
    return S


def _pulse(ctx, t, pulse):
    """0.65..1.0 envelope, a hit on every beat (bpm number or 'beat' = ctx.beats)."""
    if isinstance(pulse, str) and ctx.beats is not None and len(ctx.beats):
        b = ctx.beats
        i = int(np.searchsorted(b, t, "right")) - 1
        ph = (t - b[i]) / max(float(b[min(i + 1, len(b) - 1)] - b[i]), 1e-3) if i >= 0 and i + 1 < len(b) else 0.5
    else:
        bpm = float(pulse) if not isinstance(pulse, str) else 120.0
        ph = (t * bpm / 60.0) % 1.0
    return float(0.62 + 0.38 * math.exp(-5.0 * ph))


# ============================================================================================ virtual bundles (strands lying on the contour)
def lying_bundle(port, K, t_abs, role, cam=None, pal=None, v=260.0, tube=0.0, width=None, col=None):
    """Bundle of K strands lying on the contour at piece time t_abs (what an entry ride starts from / an exit ride ends on)."""
    ctx = port.ctx
    f = ctx.frame_of(t_abs)
    cam = cam or ctx.camera(t_abs)
    lay = port.layout(K, f)
    W, px, s = port.lying(lay, f, cam, t_abs)
    tg = port.track.tangent(f, lay.line, lay.s0) * lay.step[:, None]
    D = unit(ctx.lift(px[:, 0, :] + tg * 6.0, cam) - W[:, 0, :])
    c = np.array(port.color(pal), float)
    return Bundle(W, D, col if col is not None else c, port.width if width is None else width, t_abs, v, tube)


def entry_ride(ctx, port, b_entry, t_abs, dur=None, pal=None, **kw):
    """ride from the contour into `b_entry` (a word's entry bundle). The strands lie on the contour at piece time t_abs."""
    dur = port.lead if dur is None else dur
    a = lying_bundle(port, b_entry.N, t_abs, "entry", pal=pal, v=float(max(b_entry.v.mean(), 180.0)), tube=b_entry.tube,
                     width=float(b_entry.w.mean()) if b_entry.tube > 0.01 else None)
    return ContourRide(a, b_entry, dur, ctx=ctx, port=port, mode="entry", t_abs=t_abs, **kw)


def exit_ride(ctx, port, a_exit, t_abs, dur=None, pal=None, **kw):
    """ride from `a_exit` (a word's exit bundle) onto the contour, ending lying on it at piece time t_abs + dur."""
    dur = port.lead if dur is None else dur
    b = lying_bundle(port, a_exit.N, t_abs + dur, "exit", pal=pal, v=float(max(a_exit.v.mean(), 180.0)), tube=a_exit.tube,
                     width=float(a_exit.w.mean()) if a_exit.tube > 0.01 else None)
    return ContourRide(a_exit, b, dur, ctx=ctx, port=port, mode="exit", t_abs=t_abs, **kw)


def via_ride(ctx, port, dur=None, t_abs=0.0, **kw):
    """bare transition (params only): Sequence binds it between two words; strands ride the contour in between."""
    return ContourRide(None, None, dur, ctx=ctx, port=port, mode="via", t_abs=t_abs, **kw)


# ============================================================================================ word wrapper
class PortedWord(WordModule):
    """any word module with a contour `entry` and / or `exit` port. The word keeps its place on the timeline: with carve=True (default)
    the entry ride takes `lead_in` seconds out of the module's form window and the exit ride `lead_out` out of its exit window
    (the module is rebuilt with the shorter timing), so hit / release stay on their beats. carve=False makes the word longer instead."""
    _raw = True

    def __init__(self, inner, ctx, entry=None, exit=None, t_start=0.0, carve=True, min_form=0.25):
        self.ctx = ctx
        self.entry_port = ContourPort.parse(ctx, entry, "up")
        self.exit_port = ContourPort.parse(ctx, exit, "up")
        tm = inner.timing
        li = 0.0 if self.entry_port is None else self.entry_port.lead
        lo = 0.0 if self.exit_port is None else self.exit_port.lead + self.exit_port.persist
        if carve:
            li = min(li, max(tm.form - min_form, 0.0))
            lo = min(lo, max(tm.exit - min_form, 0.0))
            inner2 = copy.copy(inner)
            inner2.timing = Timing(tm.form - li, tm.hold, tm.exit - lo)
            inner2._mats, inner2._shape = {}, None
            inner2.form_in, inner2.form_out = copy.copy(inner.form_in), copy.copy(inner.form_out)
            inner2.build()
            inner2.form_in.attach(inner2, "in")
            inner2.form_out.attach(inner2, "out")
            inner = inner2
        self.inner, self.lead_in, self.lead_out = inner, li, lo
        self.timing = Timing(inner.timing.form + li, inner.timing.hold, inner.timing.exit + lo)
        for k in ("words", "lines", "font", "placement", "entry", "exit", "palette", "params", "form_in", "form_out", "name", "description"):
            setattr(self, k, getattr(inner, k))
        self._plain, self._mats, self._shape = True, {}, None
        from mograph.modules.io import get_style
        self.form_in = self.form_out = get_style("lines")
        self.t_start = float(t_start)
        self._rides()

    def __getattr__(self, name):                       # anything else (lay, eye_hint, STYLE_MECH ...) is the module's
        if name in ("inner",) or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def build(self):
        pass

    # the glyph-outline hand-off of a library transition (sequence.py, 2026-11-01) re-styles and re-times a word's own form / exit; a ported word wraps ITS module and the contour ride is its
    # hand-off, so the sequence leaves it alone (these methods would otherwise shorten self.timing while layers() still plays the wrapped module at its old length)
    def set_style(self, side, spec):
        pass

    def retime(self, form=None, exit=None):
        return 0.0

    def handoff_reset(self):
        pass

    def _rides(self):
        pal = self.palette
        self.ride_in = self.ride_out = None
        if self.entry_port is not None and self.lead_in > 0.05:
            self.ride_in = entry_ride(self.ctx, self.entry_port, self.inner.entry_bundle(), self.t_start, self.lead_in, pal=pal)
        if self.exit_port is not None and self.lead_out > 0.05:
            ride_dur = max(self.lead_out - self.exit_port.persist, 0.1)
            t0 = self.t_start + self.lead_in + self.inner.duration
            self.ride_out = exit_ride(self.ctx, self.exit_port, self.inner.exit_bundle(), t0, ride_dur, pal=pal)
            self.ride_out_dur = ride_dur

    @property
    def duration(self):
        return self.timing.total

    def entry_bundle(self):
        return self.ride_in.a_exit if self.ride_in is not None else self.inner.entry_bundle()

    def exit_bundle(self):
        return self.ride_out.b_entry if self.ride_out is not None else self.inner.exit_bundle()

    def layers(self, t):
        li = self.lead_in
        inner_end = li + self.inner.duration
        if self.ride_in is not None and t < li:
            return self.ride_in.layers(t)
        if t < inner_end or self.ride_out is None:
            return self.inner.layers(max(t - li, 0.0))
        tt = t - inner_end
        lyr = self.ride_out.layers(tt)
        if tt > self.ride_out_dur and self.exit_port.persist > 0:           # strands stay on the contour, the glow fades
            u = (tt - self.ride_out_dur) / max(self.exit_port.persist, 1e-6)
            fade = 1.0 - _smooth01(u)
            for L in lyr:
                L.alpha *= float(max(fade, 0.0))
        return lyr

    def focus(self, t):
        li = self.lead_in
        if self.ride_in is not None and t < li:
            return self.ride_in.focus(t)
        if self.ride_out is not None and t >= li + self.inner.duration:
            return self.ride_out.focus(min(t - li - self.inner.duration, self.ride_out.dur))
        return self.inner.focus(max(t - li, 0.0))

    def hold_pose(self, dist=900.0):
        return self.inner.hold_pose(dist)

    def bounds_points(self):
        return self.inner.bounds_points()


# ============================================================================================ spec glue (planner hooks)
PORT_KEYS = ("entry", "exit", "via", "matte")


def split_port_spec(word):
    """the planner / spec.build_scene helper. A word's `entry` / `exit` are either a module side ('up', 'left', 30 ...) or, as a dict /
    {"port": ..} string of a contour kind, a contour port. Returns (module_entry, module_exit, ports) where ports = {"entry": dict|None,
    "exit": dict|None, "via": dict|None, "matte": str|None}; module_entry / module_exit are what to pass to the word module (a dict falls back to its "side")."""
    ports = {}
    out = {}
    for key, default in (("entry", "up"), ("exit", "down")):
        v = word.get(key, default)
        if isinstance(v, dict) and "port" in v:
            ports[key] = dict(v)
            out[key] = v.get("side", default)
        elif isinstance(v, str) and v in KINDS:
            ports[key] = {"port": v}
            out[key] = default
        else:
            ports[key] = None
            out[key] = v
    v = word.get("via")
    ports["via"] = {"port": v} if isinstance(v, str) else (dict(v) if v else None)
    ports["matte"] = word.get("matte")
    return out["entry"], out["exit"], ports


def apply_ports(items, words, ctx, starts=None, carve=True, dur_via=None):
    """items = [word, transition, word, ...] as spec.build_scene makes them (before the Sequence is built); words = the word spec dicts
    (same order as the word modules). Wraps the words that carry an `entry` / `exit` contour port in `PortedWord` and replaces the
    transition after a word that carries a `via` port by a `ContourRide`. `starts[i]` = planned start (s) of word i (default: from the
    items' durations). Returns the new items list: pass it to Sequence(...), then `attach(seq, ctx)`."""
    mods = [it for it in items if isinstance(it, WordModule)]
    if len(mods) != len(words):
        raise ValueError("apply_ports: items and word specs do not match")
    out, wi, t = [], 0, 0.0
    for it in items:
        if isinstance(it, WordModule):
            _, _, ports = split_port_spec(words[wi])
            t0 = t if starts is None else float(starts[wi])
            new = PortedWord(it, ctx, ports["entry"], ports["exit"], t_start=t0, carve=carve) if (ports["entry"] or ports["exit"]) else it
            out.append(new)
            wi += 1
            t += new.duration
        else:
            via = split_port_spec(words[wi - 1])[2]["via"]
            d = it.dur if getattr(it, "dur", None) else dur_via
            if via:
                out.append(via_ride(ctx, ContourPort.parse(ctx, via, "along"), dur=d, t_abs=t))
            else:
                out.append(it)
            t += d or 0.0
    return out


def attach(seq, ctx):
    """after Sequence(...) is built: give every contour item its absolute start time from the timeline and rebuild what moved.
    Call it again after the camera is bound (the contour is projected through the camera of each moment)."""
    for it, a, b in seq.timeline:
        if isinstance(it, PortedWord):
            it.t_start = a
            it._rides()
        elif isinstance(it, ContourRide):
            if it.mode == "via":
                it.t_abs = a
                it.build()
    return seq
