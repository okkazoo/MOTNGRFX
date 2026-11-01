#!/usr/bin/env python3
"""explore_cuts2 - EXPERIMENTAL round 2 of the transition experiments (2026-10-30; round 1 = explore_cuts.py, rejected by the user: "I dont like any of the volume
changes really with the circles"). Nothing here is a default; no other module is edited.

1. DEPTH LINES FORM THE WORD (user: "create lines at the depth points instead and have them roughly track into the resolve of the planes but they form the words")
   Iso-depth contour lines of the shot's depth (Video Depth Anything archive frame, 4 levels) + the subject matte contour, cut wherever they touch the frame border
   (never a line on the frame edge), drawn as LIVE BRUSH STROKES (thick-thin along the arc, small breaks, a thinner offset sketch stroke, wobble in time, tapered ends,
   crisp, no glow). The strokes drift with their depth plane (near planes move more), then re-target: every stroke is resampled by arc length and moved point by point
   (staggered along its arc) onto a target path; targets are covered greedily by centroid distance, extra strokes land on the nearest target with a small offset
   (the 2-3 sketchy strokes of the line style). The last target set = the glyph outlines (outer contours + counters) of the word in a liked font, centred; the letters
   fill (hard on with a 1-frame flicker) ON THE BEAT and the word holds >= 1 beat.
     depth_lines_word      A's depth lines track onto B's depth lines while the picture resolves A -> B plane by plane (near planes first, crisp front), then morph
                           into B's word over B's playing footage (ink = the palette colour with the best contrast over B under the word).
     depth_lines_window    the same, but the planes resolve to a FLAT GROUND; the strokes land as the word and B's playing footage appears inside the letters.
     depth_lines_w2w       word to word inside ONE shot: word A (filled + outline strokes) un-fills, its outline strokes dissolve onto the shot's depth lines, track
                           with the planes, and re-form as word B, which fills on the beat.
2. CIRCLE GRID MATTES (user: "a grid of circles becoming mattes for the transition - blinking off and into existence")
   A regular grid of circles is the matte between A and B. The space between the circles blinks off first (A becomes a grid of circles on a flat ground), each A circle
   blinks OFF (off / on / off, hard, no fade) and B's circle blinks ON in its place 2 frames later (on / off / on); when every cell is B the space between the circles
   blinks to B: a clean B frame. Order: `beat` (seeded random cells in batches on the half beats), `radial` (outward from A's subject centroid), `sweep` (along `dir`).
   Optional crisp brush outline on the circles while they blink. `circle_grid_word`: the same with B's word revealed as dots of a finer circle grid that blink on
   inside the letters, then the solid word blinks on on the beat and holds.

Both clips play through (A ping-pong past its end, B from its frame 0 at i2 and ping-pong before it); mattes follow the picture (A's matte where A shows, B's where B
shows). Per-pixel compositing = torch on CUDA when >= 6 GB VRAM is free (else CPU); the stroke polylines are rasterised with cv2 at 2x and downsampled.
Hook: `render_cut(style, a, ma, b, mb, da, db, params)` -> explore_cuts.Result (.frames RGB uint8, .matte uint8, .info image2_start) and `window_frames(style, params)`.
CLI (box): python -m tools.fusion.explore_cuts2 job <jobs.json> <name> --out <dir>   (one process per variant, run them in parallel);  sheet <dir> <names..>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import explore_cuts as X  # noqa: E402

STYLES = ("depth_lines_word", "depth_lines_window", "depth_lines_w2w", "circle_grid", "circle_grid_word")
FONT_DIR = os.path.join(ROOT, "mograph", "fonts")
N_PTS = 128
LEVELS = (0.22, 0.42, 0.62, 0.80)
DEFAULTS = {
    "depth_lines_word": dict(word="WORD", font="PassionOne-Black.ttf", axes=None, bpm=120.0, wf=0.80, kmax=16, seed=0, direct=False, line_w=0.0105, phase=0.0, cap=0, ink=None, ground=None, morph=False, stage=None),
    "depth_lines_window": dict(word="WORD", font="PassionOne-Black.ttf", axes=None, bpm=120.0, wf=0.80, kmax=16, seed=0, direct=False, line_w=0.0105, phase=0.0, cap=0, ink=None, ground=None, var=1.25, morph=False, stage=None),
    "depth_lines_w2w": dict(word_a="ONE", word="TWO", font_a="PassionOne-Black.ttf", font="PassionOne-Black.ttf", axes_a=None, axes=None, bpm=120.0, wf=0.80, kmax=16, seed=0, direct=False, line_w=0.0105, phase=0.0, cap=0, ink=None, ground=None, morph=False),
    "circle_grid": dict(order="beat", cols=9, dir="right", outline=True, bpm=120.0, seed=0, phase=0.0, snap=False),
    "circle_grid_word": dict(order="beat", cols=9, dir="right", outline=False, word="WORD", font="PassionOne-Black.ttf", axes=None, word_cols=30, bpm=120.0, wf=0.80, seed=0, phase=0.0, snap=False, ink=None, stage=None),
}


def _dev():
    import torch
    try:
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 6e9:
            return torch.device("cuda")
    except Exception:   # noqa: BLE001
        pass
    return torch.device("cpu")


def resolve(style, params=None):
    if style not in STYLES:
        raise KeyError(f"unknown style {style!r}; one of {STYLES}")
    p = dict(DEFAULTS[style])
    for k, v in dict(params or {}).items():
        if k not in p:
            raise KeyError(f"{style}: unknown param {k!r}")
        p[k] = v
    return p


def beat_frames(p, fps=24):
    return fps * 60.0 / float(p["bpm"])


def _on_beat(f, bf, phase=0.0):
    """first beat frame >= f. phase = how many frames after a beat of the piece grid the window starts (0 = the window starts on a beat): the returned frame is a window-local frame
    whose piece frame (window start + f) is on a beat of the grid."""
    return int(round(math.ceil((f + phase) / bf - 1e-6) * bf - phase))


def timeline(style, p):
    bf = beat_frames(p)
    ph = float(p.get("phase") or 0.0)
    if style.startswith("depth_lines") and p.get("morph"):          # outline morph: the image outlines go straight onto the letter outlines (no free-lines stage): a shorter move
        land = _on_beat(46, bf, ph)
        return dict(n=land + int(math.ceil(max(bf, 14))) + 2, land=land, i2=22, bf=bf)
    if style.startswith("depth_lines"):
        land = _on_beat(54, bf, ph)
        n = land + int(math.ceil(max(bf, 14))) + 4
        return dict(n=n, land=land, i2=22, bf=bf)
    land = _on_beat(46, bf, ph)
    n = land + int(math.ceil(max(bf, 14))) + 2
    return dict(n=n, land=land, i2=24, bf=bf)


def window_frames(style, params=None):
    t = timeline(style, resolve(style, params))
    return t["n"], t["i2"]


# ----------------------------------------------------------------------------------------------- geometry
def resample(pts, n, closed=False):
    pts = np.asarray(pts, np.float32)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tot = max(float(s[-1]), 1e-3)
    u = np.linspace(0.0, tot, n, endpoint=not closed) if closed else np.linspace(0.0, tot, n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1).astype(np.float32)


def arclen(pts):
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def _smooth(pts, k, closed):
    if len(pts) < k + 2:
        return pts
    ker = np.ones(k, np.float32) / k
    if closed:
        ext = np.vstack([pts[-k:], pts, pts[:k]])
        return np.stack([np.convolve(ext[:, i], ker, "same")[k:-k] for i in (0, 1)], 1)
    ext = np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
    return np.stack([np.convolve(ext[:, i], ker, "same")[k:-k] for i in (0, 1)], 1)


def split_border(pts, W, H, min_len, m=3):
    """a closed contour -> open runs that never lie on the frame border (contours only on things in the image, never on the frame)."""
    bad = (pts[:, 0] <= m) | (pts[:, 0] >= W - 1 - m) | (pts[:, 1] <= m) | (pts[:, 1] >= H - 1 - m)
    if not bad.any():
        p = _smooth(pts, 9, True)
        return [(p, True)] if arclen(np.vstack([p, p[:1]])) >= min_len else []
    i0 = int(np.argmax(bad))
    pts, bad = np.roll(pts, -i0, 0), np.roll(bad, -i0)
    out, cur = [], []
    for q, b in zip(pts, bad):
        if b:
            if len(cur) > 4:
                out.append(np.array(cur, np.float32))
            cur = []
        else:
            cur.append(q)
    if len(cur) > 4:
        out.append(np.array(cur, np.float32))
    return [(_smooth(r, 9, False), False) for r in out if arclen(r) >= min_len]


def iso_strokes(dn, W, H, min_len, levels=LEVELS):
    d = cv2.GaussianBlur(dn.astype(np.float32), (0, 0), max(3.0, 0.012 * W))
    res = []
    for li, L in enumerate(levels):
        m = (d > L).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            for p, cl in split_border(c[:, 0, :].astype(np.float32), W, H, min_len):
                res.append(dict(pts=_smooth(p, 9, cl), closed=cl, lvl=float(L), subj=False))
    return res


def matte_strokes(m, W, H, min_len):
    if m is None or float(m.max()) < 0.5:
        return []
    mm = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=len)
    return [dict(pts=p, closed=cl, lvl=0.9, subj=True) for p, cl in split_border(c[:, 0, :].astype(np.float32), W, H, min_len)]


def pick_strokes(strokes, kmax):
    subj = sorted([s for s in strokes if s["subj"]], key=lambda s: -arclen(s["pts"]))[:2]
    rest = sorted([s for s in strokes if not s["subj"]], key=lambda s: -arclen(s["pts"]))[:max(1, kmax - len(subj))]
    return subj + rest


# ----------------------------------------------------------------------------------------------- word
def _font(file, size, axes):
    from mograph import fonts_liked            # the ONE place a font resolves: liked.json only, raises FontNotLiked, no generic fallback (user rule 2026-10-30)
    f = fonts_liked.truetype(file, int(size))
    if axes:
        try:
            names = [a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"]) for a in f.get_variation_axes()]
            tags = {"Width": "wdth", "Weight": "wght"}
            vals = []
            for a, nm in zip(f.get_variation_axes(), names):
                tag = tags.get(nm, nm.lower()[:4])
                vals.append(float(axes.get(tag, a.get("default", a["minimum"]))))
            f.set_variation_by_axes(vals)
        except Exception as e:   # noqa: BLE001
            print("font axes not set:", e)
    return f


def stage_layout(stage, W, H, sc=2):
    """B's word laid out EXACTLY as the type stage lands it (layout_preview.flat_frame: stable_type.fit_box rows / cap height / +-90 roll, the word's liked face + axes, centred): the cut's last
    frame = the type stage's first frame (no jump at the hand-over). `stage` = the word dict (text, font, axes, lines, placement). -> the same triple as word_layout."""
    from tools.fusion import layout_preview as LP
    _, union, lid = LP.flat_frame(stage, size=(W * sc, H * sc), letters=True)
    return union, lid, _contours(union, lid, sc)


def word_layout(text, file, axes, W, H, wf=0.80, hf=0.26, sc=2, stage=None):
    """centred one-line word at 2x: (union mask uint8 (2H,2W), letter-id map int8 (-1 none), glyph contours [(pts 1x, closed, letter)]). With `stage` (the word dict as the type stage holds it)
    the word is laid out the stage's way instead (stage_layout): rows, roll, cap height of its real layout."""
    if stage:
        return stage_layout(stage, W, H, sc)
    from PIL import Image, ImageDraw
    f0 = _font(file, 200, axes)
    tw = f0.getlength(text)
    bb = f0.getbbox(text)
    th = bb[3] - bb[1]
    size = 200 * min(wf * W * sc / max(tw, 1), hf * H * sc / max(th, 1))
    f = _font(file, size, axes)
    tw = f.getlength(text)
    bb = f.getbbox(text)
    x0 = (W * sc - tw) / 2.0
    y0 = (H * sc - (bb[3] - bb[1])) / 2.0 - bb[1]
    union = np.zeros((H * sc, W * sc), np.uint8)
    lid = np.full((H * sc, W * sc), -1, np.int16)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        im = Image.new("L", (W * sc, H * sc), 0)
        ImageDraw.Draw(im).text((x0 + f.getlength(text[:i]), y0), ch, font=f, fill=255)
        a = np.asarray(im)
        lid[(a > 127) & (lid < 0)] = i
        union = np.maximum(union, a)
    return union, lid, _contours(union, lid, sc)


def _contours(union, lid, sc):
    """the glyph contours (outer + counters) of a word mask, each tagged with its letter id: [dict(pts 1x, closed, letter)]."""
    cs, _ = cv2.findContours((union > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    glyphs = []
    for c in cs:
        p = c[:, 0, :].astype(np.float32)
        if arclen(p) < 24 * sc:
            continue
        j = int(lid[int(p[0, 1]), int(p[0, 0])])
        if j < 0:
            yy, xx = int(p[len(p) // 2, 1]), int(p[len(p) // 2, 0])
            j = int(lid[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3].max())
        glyphs.append(dict(pts=_smooth(p / sc, 3, True), closed=True, letter=max(j, 0)))
    return glyphs


# ----------------------------------------------------------------------------------------------- correspondence
def _centroid(p):
    return p.mean(0)


def _align(src, tgt, closed):
    if not closed:
        f = np.abs(tgt - src).sum()
        r = np.abs(tgt[::-1] - src).sum()
        return tgt if f <= r else tgt[::-1].copy()
    best, bt = None, None
    for t in (tgt, tgt[::-1]):
        k = int(np.argmin(np.hypot(*(t - src[0]).T)))
        rt = np.roll(t, -k, 0)
        cost = float(np.hypot(*(rt - src).T).sum())
        if best is None or cost < best:
            best, bt = cost, rt
    return bt.copy()


def assign(cur, targets):
    """cur: list of (N,2) paths; targets: list of dict(pts, closed). -> (cur (maybe split), aligned target paths, closed flags, target index per stroke)."""
    cur = [c for c in cur]
    while len(cur) < len(targets):
        i = int(np.argmax([arclen(c) for c in cur]))
        c = cur.pop(i)
        h = len(c) // 2
        cur += [resample(c[:h + 1], N_PTS), resample(c[h:], N_PTS)]
    tg = [resample(t["pts"], N_PTS, t["closed"]) for t in targets]
    cc = np.array([_centroid(c) for c in cur])
    tc = np.array([_centroid(t) for t in tg])
    D = np.hypot(cc[:, None, 0] - tc[None, :, 0], cc[:, None, 1] - tc[None, :, 1])
    out = [-1] * len(cur)
    covered = set()
    for idx in np.argsort(D, axis=None):
        i, j = divmod(int(idx), len(tg))
        if out[i] >= 0 or j in covered:
            continue
        out[i] = j
        covered.add(j)
        if len(covered) == len(tg):
            break
    dup = {}
    paths, closed = [], []
    for i in range(len(cur)):
        if out[i] < 0:
            out[i] = int(np.argmin(D[i]))
        j = out[i]
        t = _align(cur[i], tg[j], targets[j]["closed"])
        k = dup.get(j, 0)
        dup[j] = k + 1
        if k:                                               # a second / third stroke on the same outline: offset along the normal (sketchy multi-stroke)
            d = np.gradient(t, axis=0)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            t = t + nrm * (1.3 * k * (1 if k % 2 else -1))
        paths.append(t.astype(np.float32))
        closed.append(bool(targets[j]["closed"]))
    return cur, paths, closed, out


def cap_mask(tgt_idx, strokes, targets, cap, tgt_pts=None):
    """keep-flags for strokes assigned to targets (tgt_idx[i] = target of stroke i, -1 = unknown = kept): at most `cap` strokes per target, the ones whose centroid is nearest the target's win.
    tgt_pts = the aligned target path per stroke (stage 2); else the target set's own `pts`."""
    cen = [_centroid(np.asarray(c, np.float32)) for c in strokes]
    keep = [True] * len(tgt_idx)
    by = {}
    for i, j in enumerate(tgt_idx):
        if j is not None and j >= 0 and i < len(cen):
            by.setdefault(int(j), []).append(i)
    for j, idx in by.items():
        if len(idx) <= cap:
            continue
        tc = _centroid(np.asarray(targets[j]["pts"], np.float32)) if tgt_pts is None else None
        d = []
        for i in idx:
            c2 = _centroid(np.asarray(tgt_pts[i], np.float32)) if tgt_pts is not None else tc
            d.append(float(np.hypot(cen[i][0] - c2[0], cen[i][1] - c2[1])))
        for i in [idx[k] for k in np.argsort(d)][cap:]:
            keep[i] = False
    return keep


# ----------------------------------------------------------------------------------------------- brush
def draw_brush(mask2, pts, w, ph, t, prog=1.0, closed=False, sc=2, var=1.0):
    """live brush stroke along a polyline into a uint8 mask at sc x (in place)."""
    n = len(pts)
    if n < 2 or w <= 0:
        return
    P = pts * sc
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / max(s[-1], 1e-3)
    ww = w * sc * (0.62 + 0.28 * var * np.sin(6.283 * 3 * s + ph * 6.283 + 1.7 * t) + 0.14 * var * np.sin(6.283 * 7 * s + ph * 13.0 - 2.3 * t))
    ww = np.where(np.sin(6.283 * 5 * s + ph * 20.0 + 0.9 * t) > 0.95, 0.0, ww)
    if not closed:
        ww = ww * np.clip(np.minimum(s, 1.0 - s) / 0.07, 0.0, 1.0) ** 0.6
    if prog < 1.0:
        ww = ww * np.clip((prog * 1.08 - s) / 0.05, 0.0, 1.0)
    d = np.gradient(P, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
    off = (1.1 + 0.9 * np.sin(6.283 * 4 * s + ph * 9.0 + 1.3 * t))[:, None] * w * sc * 0.45
    Q = P + nrm * off
    sk = np.sin(6.283 * 7 * s + ph * 13.0 - 2.3 * t) > -0.2
    for i in range(len(P) - 1):
        if ww[i] >= 0.6:
            cv2.line(mask2, tuple(np.round(P[i] * 4).astype(int)), tuple(np.round(P[i + 1] * 4).astype(int)), 255, max(1, int(round(ww[i]))), cv2.LINE_AA, 2)
        if sk[i] and ww[i] >= 1.2:
            cv2.line(mask2, tuple(np.round(Q[i] * 4).astype(int)), tuple(np.round(Q[i + 1] * 4).astype(int)), 255, max(1, int(round(ww[i] * 0.35))), cv2.LINE_AA, 2)


def circle_pts(c, r, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- colours
def _lum(c_bgr):
    c = np.clip(np.asarray(c_bgr, np.float64) / 255.0, 0, 1)[::-1]
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def ink_for(frames_bgr, mask2, extra=()):
    """the palette colour (of the footage under the word) with the best contrast against the mean colour under the word box."""
    try:
        from tools.fusion.compose import palette_from_footage
        cent = [np.asarray(c, np.float32) for c in palette_from_footage(list(frames_bgr[::max(1, len(frames_bgr) // 8)]), k=5)]
    except Exception:   # noqa: BLE001
        cent = [np.float32([30, 30, 30]), np.float32([230, 230, 230])]
    H, W = frames_bgr[0].shape[:2]
    m = cv2.resize(mask2, (W, H), interpolation=cv2.INTER_AREA) > 60
    if not m.any():
        m[:] = True
    ys, xs = np.where(m)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    bg = np.mean([f[box].reshape(-1, 3).mean(0) for f in frames_bgr[::max(1, len(frames_bgr) // 6)]], 0)
    cands = cent + [np.asarray(e, np.float32) for e in extra]
    sc = [contrast(c, bg) for c in cands]
    i = int(np.argmax(sc))
    return cands[i], float(sc[i]), bg


def piece_ink(p, frames_bgr, mask2, extra=()):
    """the word's ink: `p["ink"]` (BGR, the colour the piece designed for the word) when it reaches 3:1 over what is behind the word, else the best palette colour (ink_for). The type stage picks
    the word up after the cut with the same designed colour (legible_color keeps a colour that passes), so the cut's colour == the stage's."""
    best, cr, bg = ink_for(frames_bgr, mask2, extra=tuple(extra) + (np.float32([20, 20, 20]), np.float32([245, 245, 245])))      # near-black / near-white: legible_color's last resort, so 3:1 is reachable
    given = p.get("ink")
    if given is not None:
        g = np.asarray(given, np.float32)
        c = contrast(g, bg)
        if c >= 3.0:
            return g, float(c), bg
    return best, cr, bg


# ----------------------------------------------------------------------------------------------- torch helpers
class TT:
    def __init__(self, dev):
        import torch
        self.torch, self.dev = torch, dev

    def img(self, x):
        return self.torch.from_numpy(np.ascontiguousarray(x)).to(self.dev, self.torch.float32)

    def a2(self, mask2):
        """2x uint8 mask -> 1x alpha (H,W,1) by 2x2 area mean on the device."""
        t = self.torch.from_numpy(mask2).to(self.dev, self.torch.float32) / 255.0
        t = self.torch.nn.functional.avg_pool2d(t[None, None], 2)[0, 0]
        return t[..., None]


# ----------------------------------------------------------------------------------------------- depth lines
class DepthLines:
    def __init__(self, style, p, A, mA, dA, B, mB, dB, tt, dA_all=None):
        self.s, self.p, self.tt = style, p, tt
        self.A, self.mA, self.B, self.mB = A, mA, B, mB
        H, W = A[0].shape[:2]
        self.H, self.W = H, W
        self.tl = timeline(style, p)
        minl = 0.10 * min(H, W)
        self.dA, self.dB = dA, dB
        rng = np.random.default_rng(p["seed"])
        self.union, self.lid, glyphs = word_layout(p["word"], p["font"], p["axes"], W, H, p["wf"], stage=p.get("stage"))
        self.nlet = len(p["word"])
        kmax = p["kmax"]
        if style == "depth_lines_w2w":
            self.union_a, self.lid_a, glyph_a = word_layout(p["word_a"], p["font_a"], p["axes_a"], W, H, p["wf"])
            src = [dict(pts=g["pts"], closed=True, lvl=0.5, subj=False) for g in glyph_a]
            mid = pick_strokes(iso_strokes(dA, W, H, minl) + matte_strokes(mA[0], W, H, minl), kmax)
            sets = [src, mid, glyphs]
        elif p.get("morph"):                             # image outlines -> letter outlines, one continuous morph (each contour onto the glyph contour with the nearest centroid)
            s0 = pick_strokes(iso_strokes(dA, W, H, minl) + matte_strokes(mA[0], W, H, minl), kmax)
            gl = [dict(g, lvl=0.5, subj=False) for g in glyphs]
            sets = [s0, gl, gl]
        else:
            s0 = pick_strokes(iso_strokes(dA, W, H, minl) + matte_strokes(mA[0], W, H, minl), kmax)
            s1 = iso_strokes(dB, W, H, minl) + matte_strokes(mB[0], W, H, minl)
            s1 = pick_strokes(s1, kmax) or s0
            sets = [s0, s1, glyphs]
        # chain of keyframes: k0 = set 0 resampled, k1 = assigned onto set 1, k2 = assigned onto the glyphs
        k0 = [resample(s["pts"], N_PTS, s["closed"]) for s in sets[0]]
        self.closed0 = [bool(s["closed"]) for s in sets[0]]
        lv0 = [s["lvl"] for s in sets[0]]
        cur, k1, cl1, j1 = assign(k0, sets[1])
        # strokes split by assign: inherit level / closed from their parent (approximated as open, same level order)
        while len(lv0) < len(cur):
            lv0.append(lv0[-1])
            self.closed0.append(False)
        self.closed0 = [c and len(cur) == len(k0) for c in self.closed0]
        lv1 = [sets[1][j]["lvl"] for j in j1]
        if p.get("morph"):
            cur2, k2, cl2, j2 = list(k1), list(k1), list(cl1), list(j1)      # already on the glyph outlines: no second stage
        else:
            cur2, k2, cl2, j2 = assign(k1, sets[2])
        if len(cur2) > len(cur):                         # glyphs outnumbered the strokes: the extra strokes are split halves of k1; give them k0 / k1 halves too
            k0e, k1e, lv0e, lv1e = list(cur), list(k1), list(lv0), list(lv1)
            while len(k1e) < len(cur2):
                i = int(np.argmax([arclen(c) for c in k1e[:len(k1)]]))
                k0e.append(k0e[i].copy())
                k1e.append(k1e[i].copy())
                lv0e.append(lv0e[i])
                lv1e.append(lv1e[i])
            cur, lv0, lv1 = k0e, lv0e, lv1e
            self.closed0 += [False] * (len(cur2) - len(self.closed0))
        self.k0, self.k1, self.k2 = [np.asarray(c, np.float32) for c in cur], [np.asarray(c, np.float32) for c in cur2], k2
        self.cl1, self.cl2 = cl1 + [False] * (len(cur2) - len(cl1)), cl2
        self.lv0, self.lv1 = lv0, lv1
        self.letter = [sets[2][j]["letter"] for j in j2]
        self.capped = 0
        cap = int(p.get("cap") or 0)
        if cap > 0:                                      # at most `cap` strokes per target (the horse scribble: many strokes collapsed onto one small subject outline / one glyph outline)
            keep = cap_mask([j1[i] if i < len(j1) else -1 for i in range(len(self.k1))], self.k1, sets[1], cap)
            keep = [a and b for a, b in zip(keep, cap_mask(j2, self.k1, sets[2], cap, tgt_pts=k2))]
            if not any(keep):
                keep[0] = True
            self.capped = len(keep) - sum(keep)
            pick = lambda xs: [x for x, kp in zip(xs, keep) if kp]      # noqa: E731
            self.k0, self.k1, self.k2 = pick(self.k0), pick(self.k1), pick(self.k2)
            self.cl1, self.cl2, self.letter = pick(self.cl1), pick(self.cl2), pick(self.letter)
            self.lv0, self.lv1 = pick(self.lv0 + [0.5] * (len(keep) - len(self.lv0))), pick(self.lv1 + [0.5] * (len(keep) - len(self.lv1)))
            self.closed0 = pick(self.closed0 + [False] * (len(keep) - len(self.closed0)))
        self.subj0 = [False] * len(self.k0)
        self.phase = rng.random(len(self.k0))
        self.stag = rng.uniform(0.15, 0.45, len(self.k0))
        self.dirn = rng.choice([-1.0, 1.0], len(self.k0))
        mats = [mA[0]]
        self.cA0 = self._cent(mA[0])
        self.cB0 = self._cent(mB[0])
        # colours: ink with the best contrast over what is behind the word (B footage; the flat ground for the window style), stroke = lively palette colour
        stroke_a, ground_a = X._palette(A)
        stroke_b, ground_b = X._palette(B)
        self.ground = np.float32(np.minimum(ground_a, ground_b) * 0.6) if p.get("ground") is None else np.float32(p["ground"])
        if style == "depth_lines_window":
            self.ink = None
            self.stroke = stroke_b
            self.ink_cr = None
        else:
            self.ink, self.ink_cr, _ = piece_ink(p, B if style != "depth_lines_w2w" else A, self.union, extra=(stroke_b,))
            self.stroke = stroke_b if np.abs(stroke_b - self.ink).sum() > 90 else (np.float32([245, 245, 245]) if _lum(self.ink) < 0.4 else np.float32([20, 20, 20]))
        self.w = p["line_w"] * W
        self.morph = bool(p.get("morph"))
        self.direct = bool(p.get("direct")) or self.morph
        if self.direct:                                  # lines go straight into the word: per-letter wipe reveal driven by that letter's stroke arrival (no outline stage)
            H2, W2 = self.lid.shape
            xn = np.zeros((H2, W2), np.float32)
            for li in range(self.nlet):
                m = self.lid == li
                if not m.any():
                    continue
                xs = np.nonzero(m.any(0))[0]
                x0, x1 = int(xs[0]), int(xs[-1])
                xn[m] = (np.nonzero(m)[1] - x0) / float(max(x1 - x0, 1))
            self.xnT = tt.img(xn)
            self.lidT = tt.torch.from_numpy((self.lid.astype(np.int64) + 1)).to(tt.dev)
            self.uT = tt.img(self.union.astype(np.float32) / 255.0)
        self.dA_all = dA_all if dA_all is not None else [dA]
        self._dn_cache = {}
        self.U = tt.a2(self.union)
        if style == "depth_lines_w2w":
            self.Ua = tt.a2(self.union_a)
        self.lidt = None

    def _cent(self, m):
        if m is None or float(m.sum()) < 10:
            return np.float32([self.W / 2, self.H / 2])
        ys, xs = np.nonzero(m > 0.5)
        return np.float32([xs.mean(), ys.mean()]) if len(xs) else np.float32([self.W / 2, self.H / 2])

    def _drift(self, f, lvl):
        a = 2 * math.pi * f / 96.0
        amp = 0.025 * self.W * (lvl - 0.45) * 2.0
        return np.float32([amp * math.sin(a) * 1.0, 0.4 * amp * math.cos(a)])

    def _times(self):
        L = self.tl["land"]
        if self.s == "depth_lines_w2w":
            return dict(on=(0, 2), m1=(12, 32), m2=(40, L), fill_off=10)
        if self.morph:
            return dict(on=(4, 14), mm=(12, L - 12), res=(14, 36))
        return dict(on=(4, 14), m1=(14, 36), m2=(40, L), res=(14, 36))

    def paths(self, f, ka, kb):
        T = self._times()
        out = []
        trA = (self._cent(self.mA[min(ka, len(self.mA) - 1)]) - self.cA0)
        trB = (self._cent(self.mB[min(kb, len(self.mB) - 1)]) - self.cB0)
        u = np.linspace(0, 1, N_PTS, dtype=np.float32)[:, None]
        e2m = []
        for i in range(len(self.k0)):
            p0 = self.k0[i] + (self._drift(f, self.lv0[i]) if self.s != "depth_lines_w2w" else 0)
            if self.lv0[i] >= 0.9 and self.s != "depth_lines_w2w":
                p0 = p0 + trA
            p1 = self.k1[i] + self._drift(f, self.lv1[i] if i < len(self.lv1) else 0.5) + (trB if (i < len(self.lv1) and self.lv1[i] >= 0.9) else 0)
            if self.s == "depth_lines_w2w" and i < len(self.lv1) and self.lv1[i] >= 0.9:
                p1 = self.k1[i] + self._drift(f, 0.9) + trA
            p2 = self.k2[i]
            uu = u if self.dirn[i] > 0 else 1 - u
            st = self.stag[i]

            def e_of(a, b):
                x = (f - a) / max(b - a, 1)
                return X.sm(x * (1 + st) - st * uu)
            if self.morph:                                   # one continuous morph: the image outline moves point for point (staggered along its arc) onto the glyph outline
                e2 = e_of(*T["mm"])
                p = p0 * (1 - e2) + p2 * e2
                e1 = e2
            else:
                e1 = e_of(*T["m1"])
                e2 = e_of(*T["m2"])
                p = p0 * (1 - e1) + p1 * e1
                p = p * (1 - e2) + p2 * e2
            closed = (self.closed0[i] and float(e1.max()) <= 0.0) or (self.cl2[i] and float(e2.min()) >= 0.995)
            prog = float(X.sm((f - T["on"][0]) / max(T["on"][1] - T["on"][0], 1)))
            out.append((p, closed, prog))
            e2m.append(float(np.mean(e2)))
        self._e2m = e2m
        return out

    def frame(self, f, a_bgr, a_m, b_bgr, b_m, ka, kb):
        tt, torch = self.tt, self.tt.torch
        T = self._times()
        A, Bm = tt.img(a_bgr), tt.img(b_bgr)
        am, bm = tt.img(a_m)[..., None], tt.img(b_m)[..., None]
        if self.s == "depth_lines_w2w":
            pic, mat = A, am
        else:
            r0, r1 = T["res"]
            L = 1.06 - 1.12 * float(X.sm((f - r0) / (r1 - r0)))
            k = min(ka, len(self.dA_all) - 1)
            if k not in self._dn_cache:
                self._dn_cache = {k: tt.img(cv2.GaussianBlur(self.dA_all[k].astype(np.float32), (0, 0), 2.0))[..., None]}
            sel = torch.clamp((self._dn_cache[k] - L) / 0.012 + 0.5, 0, 1)
            if self.s == "depth_lines_window":
                G = torch.tensor(self.ground, device=tt.dev).view(1, 1, 3).expand_as(A)
                pic = A * (1 - sel) + G * sel
                mat = am * (1 - sel)
            else:
                pic = A * (1 - sel) + Bm * sel
                mat = am * (1 - sel) + bm * sel
        land = self.tl["land"]
        # letters: hard on with a 1-frame flicker on the beat (land - 2 on, land - 1 off, land on)
        fill_on = f >= land or f == land - 2
        pths = self.paths(f, ka, kb) if self.direct else None
        if self.direct:
            fill_on = False
            Uf = self.U
            if f >= land:
                Uf = self.U
            else:
                lp = np.full(self.nlet + 1, 0.0, np.float32)
                for li in range(self.nlet):
                    v = [self._e2m[i] for i in range(len(self._e2m)) if self.letter[i] == li]
                    lp[li + 1] = float(np.mean(v)) if v else float(np.mean(self._e2m))
                edge = np.clip((lp - 0.30) / 0.55, 0.0, 1.0) * 1.02
                if self.morph:                           # the outlines are COMPLETE first (strokes on the glyph contours), then the letters fill left to right and the outline thins away
                    h = float(np.clip((f - (land - 10)) / 10.0, 0.0, 1.0))
                    edge = np.zeros(self.nlet + 1, np.float32)
                    for li in range(self.nlet):
                        edge[li + 1] = float(np.clip(h * 1.5 - 0.5 * li / max(self.nlet - 1, 1), 0.0, 1.02))
                rev = (self.xnT < torch.from_numpy(edge).to(tt.dev)[self.lidT]).float() * self.uT
                Uf = torch.nn.functional.avg_pool2d(rev[None, None], 2)[0, 0][..., None]
            if self.s == "depth_lines_window":
                pic = pic * (1 - Uf) + Bm * Uf
                mat = mat * (1 - Uf) + bm * Uf
            else:
                ink = torch.tensor(self.ink, device=tt.dev).view(1, 1, 3)
                pic = pic * (1 - Uf) + ink * Uf
                mat = mat * (1 - Uf)
        if self.s == "depth_lines_w2w":
            if f < T["fill_off"] or f == T["fill_off"] + 1:
                ink = torch.tensor(self.ink, device=tt.dev).view(1, 1, 3)
                pic = pic * (1 - self.Ua) + ink * self.Ua
        if fill_on:
            if self.s == "depth_lines_window":
                pic = pic * (1 - self.U) + Bm * self.U
                mat = mat * (1 - self.U) + bm * self.U
            else:
                ink = torch.tensor(self.ink, device=tt.dev).view(1, 1, 3)
                pic = pic * (1 - self.U) + ink * self.U
                mat = mat * (1 - self.U)
        m2 = np.zeros((self.H * 2, self.W * 2), np.uint8)
        for i, (p, closed, prog) in enumerate(pths if self.direct else self.paths(f, ka, kb)):
            if self.direct:                              # strokes thin and taper away as they land (gone at the beat), thick-thin variation widened
                g = float(np.clip((self._e2m[i] - 0.40) / 0.50, 0.0, 1.0)) if not self.morph else float(np.clip((f - (land - 10)) / 10.0, 0.0, 1.0))
                if f >= land or g >= 1.0:
                    continue
                draw_brush(m2, p, self.w * (1.0 - g), float(self.phase[i]), f / 24.0, prog, closed, var=float(self.p.get("var", 1.25)))
            else:
                draw_brush(m2, p, self.w, float(self.phase[i]), f / 24.0, prog, closed)
        sa = tt.a2(m2)
        col = torch.tensor(self.stroke, device=tt.dev).view(1, 1, 3)
        pic = pic * (1 - sa) + col * sa
        mat = mat * (1 - sa)
        return pic, mat[..., 0]


# ----------------------------------------------------------------------------------------------- circle grid
class CircleGrid:
    def __init__(self, style, p, A, mA, B, mB, tt):
        self.s, self.p, self.tt = style, p, tt
        torch = tt.torch
        H, W = A[0].shape[:2]
        self.H, self.W = H, W
        self.tl = timeline(style, p)
        cols = int(p["cols"])
        cell = W / cols
        rows = int(math.ceil(H / cell))
        oy = (H - rows * cell) / 2.0
        cx = (np.arange(cols) + 0.5) * cell
        cy = oy + (np.arange(rows) + 0.5) * cell
        self.cell, self.r = cell, 0.46 * cell
        gx, gy = np.meshgrid(cx, cy)
        self.cx, self.cy = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
        K = len(self.cx)
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        ci = np.clip(np.floor(xx / cell), 0, cols - 1).astype(np.int64)
        cj = np.clip(np.floor((yy - oy) / cell), 0, rows - 1).astype(np.int64)
        kid = cj * cols + ci
        dist = np.hypot(xx - self.cx[kid], yy - self.cy[kid])
        self.kid = torch.from_numpy(kid).to(tt.dev)
        self.cov = tt.img(np.clip(self.r - dist + 0.5, 0, 1))[..., None]
        rng = np.random.default_rng(p["seed"])
        bf = self.tl["bf"]
        t0, t1 = 10, self.tl["land"] - 12
        self.snap = bool(p.get("snap"))
        hb = bf / 2.0
        self.ph = float(p.get("phase") or 0.0)
        if self.snap:                                    # the blink rhythm sits on the piece's beat grid: the first batch is the first half-beat at/after frame 10 (window-local frame f is on the grid at (f + phase) % half-beat == 0)
            t0 = _on_beat(10, hb, self.ph)
        order = p["order"]
        if order == "beat":
            perm = rng.permutation(K)
            step = bf / 2.0
            nb = max(1, int((t1 - t0) // step))
            grp = np.array_split(perm, nb)
            tk = np.zeros(K)
            for g, idx in enumerate(grp):
                tk[idx] = t0 + g * step
            if self.snap:
                tk = np.round(tk)
        elif order == "radial":
            ys, xs = np.nonzero(mA[0] > 0.5) if mA is not None and float(mA[0].max()) > 0.5 else (np.array([H / 2]), np.array([W / 2]))
            c = np.float32([xs.mean(), ys.mean()])
            d = np.hypot(self.cx - c[0], self.cy - c[1])
            tk = t0 + (t1 - t0) * (d / d.max()) + rng.uniform(-1.5, 1.5, K)
        else:
            v = {"right": self.cx / W, "left": 1 - self.cx / W, "down": self.cy / H, "up": 1 - self.cy / H}[p["dir"]]
            tk = t0 + (t1 - t0) * v + rng.uniform(-2.0, 2.0, K)
        self.tA = np.round(tk).astype(int)
        self.tB = self.tA + 2
        self.t_gap_off = 6
        self.t_gap_on = self.tl["land"] - 6 if style == "circle_grid" else self.tl["land"] - 8
        self.phase = rng.random(K)
        stroke_a, ground_a = X._palette(A)
        stroke_b, ground_b = X._palette(B)
        self.ground = np.float32(np.minimum(ground_a, ground_b) * 0.6) if p.get("ground") is None else np.float32(p["ground"])
        self.stroke = stroke_b
        self.w = 0.0045 * min(W, H)       # 2026-11-01 user "keep the outline weight quite fine" on the circle transitions: draw_brush width 0.62-1.04 x this = 0.0028-0.0047 (mean ~0.0037) of the short side (was 0.0085 x W: ~0.0068 on 9:16, ~0.012 on 16:9)
        if style == "circle_grid_word":
            self.union, _, _ = word_layout(p["word"], p["font"], p["axes"], W, H, p["wf"], stage=p.get("stage"))
            self.ink, self.ink_cr, _ = piece_ink(p, B, self.union, extra=(stroke_b,))
            U = cv2.resize(self.union, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            self.U = tt.img(U)[..., None]
            wc = W / int(p["word_cols"])
            wr = int(math.ceil(H / wc))
            wx, wy = (np.arange(int(p["word_cols"])) + 0.5) * wc, (H - wr * wc) / 2 + (np.arange(wr) + 0.5) * wc
            wkx = np.clip(np.floor(xx / wc), 0, int(p["word_cols"]) - 1).astype(np.int64)
            wky = np.clip(np.floor((yy - (H - wr * wc) / 2) / wc), 0, wr - 1).astype(np.int64)
            wk = wky * int(p["word_cols"]) + wkx
            gxx, gyy = np.meshgrid(wx, wy)
            gxx, gyy = gxx.ravel(), gyy.ravel()
            wd = np.hypot(xx - gxx[wk], yy - gyy[wk])
            self.wcov = tt.img(np.clip(0.44 * wc - wd + 0.5, 0, 1))[..., None]
            self.wk = torch.from_numpy(wk).to(tt.dev)
            inside = U[np.clip(gyy.astype(int), 0, H - 1), np.clip(gxx.astype(int), 0, W - 1)] > 0.5
            won = rng.integers(18, self.tl["land"] - 4, len(gxx))
            if self.snap:                                # dot blinks on the half-beats of the piece grid (the last batch lands a half-beat before the word)
                won = np.array([min(_on_beat(int(w), hb, self.ph), self.tl["land"] - max(int(round(hb)), 2)) for w in won])
            self.won = np.where(inside, won, 10 ** 6)

    @staticmethod
    def _flick(f, t, on_first):
        """state of a hard blink that starts at t: on_first -> (on, off, on, on ...); else (off, on, off, off ...). Before t: the opposite steady state."""
        if f < t:
            return not on_first
        k = f - t
        return (k != 1) if on_first else (k == 1)

    def frame(self, f, a_bgr, a_m, b_bgr, b_m, ka, kb):
        tt, torch = self.tt, self.tt.torch
        A, Bm = tt.img(a_bgr), tt.img(b_bgr)
        am, bm = tt.img(a_m)[..., None], tt.img(b_m)[..., None]
        G = torch.tensor(self.ground, device=tt.dev).view(1, 1, 3)
        K = len(self.cx)
        a_on = np.array([self._flick(f, t, False) for t in self.tA])          # A circle: on until tA, then off / on / off
        b_on = np.array([self._flick(f, t, True) for t in self.tB])           # B circle: off until tB, then on / off / on
        a_on_t = torch.from_numpy(a_on.astype(np.float32)).to(tt.dev)[self.kid][..., None]
        b_on_t = torch.from_numpy(b_on.astype(np.float32)).to(tt.dev)[self.kid][..., None]
        circ = Bm * b_on_t + A * a_on_t * (1 - b_on_t) + G * (1 - a_on_t) * (1 - b_on_t)
        cm = bm * b_on_t + am * a_on_t * (1 - b_on_t)
        gap_a = self._flick(f, self.t_gap_off, False)                         # the space between the circles: A, then blinks to the ground
        gap_b = self._flick(f, self.t_gap_on, True)                           # ... and to B at the end
        gap = Bm if gap_b else (A if gap_a else G.expand_as(A))
        gm = bm if gap_b else (am if gap_a else torch.zeros_like(am))
        if gap_b and f >= self.t_gap_on + 2:
            pic, mat = Bm, bm
        else:
            pic = gap * (1 - self.cov) + circ * self.cov
            mat = gm * (1 - self.cov) + cm * self.cov
        if self.p["outline"] and self.t_gap_off <= f < self.t_gap_on:
            m2 = np.zeros((self.H * 2, self.W * 2), np.uint8)
            vis = np.nonzero(a_on | b_on)[0]
            for k in vis:
                draw_brush(m2, circle_pts((self.cx[k], self.cy[k]), self.r - self.w * 0.4), self.w, float(self.phase[k]), f / 24.0, 1.0, True)
            sa = tt.a2(m2)
            col = torch.tensor(self.stroke, device=tt.dev).view(1, 1, 3)
            pic = pic * (1 - sa) + col * sa
            mat = mat * (1 - sa)
        if self.s == "circle_grid_word":
            land = self.tl["land"]
            ink = torch.tensor(self.ink, device=tt.dev).view(1, 1, 3)
            if f >= land or f == land - 2:
                pic = pic * (1 - self.U) + ink * self.U
                mat = mat * (1 - self.U)
            else:
                on = np.array([self._flick(f, t, True) for t in self.won], np.float32)
                ont = torch.from_numpy(on).to(tt.dev)[self.wk][..., None]
                a = self.wcov * ont * self.U.clamp(0, 1).gt(0.5).float()
                pic = pic * (1 - a) + ink * a
                mat = mat * (1 - a)
        return pic, mat[..., 0]


# ----------------------------------------------------------------------------------------------- render + CLI
def render_cut(style, a, ma, b, mb, da=None, db=None, params=None) -> "X.Result":
    p = resolve(style, params)
    tl = timeline(style, p)
    n, i2 = tl["n"], tl["i2"]
    A = X._as_list_rgb(a)[..., ::-1].copy()
    Bc = X._as_list_rgb(b)
    hw = tuple(A.shape[1:3])
    if Bc.shape[1:3] != hw:
        Bc = np.stack([cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for x in Bc])
    B = Bc[..., ::-1].copy()
    mA, mB = X._mattes(ma, len(A), hw), X._mattes(mb, len(B), hw)
    dev = _dev()
    tt = TT(dev)
    t0 = time.time()
    if style.startswith("depth_lines"):
        dA, srcA = X._depths(da, A, hw)
        dB, srcB = X._depths(db, B, hw)
        eng = DepthLines(style, p, A, mA, dA[0], B, mB, dB[0], tt, dA_all=dA)
    else:
        srcA = srcB = "not used"
        eng = CircleGrid(style, p, A, mA, B, mB, tt)
    t_setup = time.time() - t0
    frames = np.zeros((n,) + hw + (3,), np.uint8)
    matte = np.zeros((n,) + hw, np.uint8)
    for f in range(n):
        ka = X.pingpong(len(A), f)
        kb = X.pingpong(len(B), f - i2)
        pic, mt = eng.frame(f, A[ka], mA[min(ka, len(mA) - 1)], B[kb], mB[min(kb, len(mB) - 1)], ka, kb)
        frames[f] = pic.clamp(0, 255).add(0.5).to(tt.torch.uint8).cpu().numpy()[..., ::-1]
        matte[f] = (mt.clamp(0, 1) * 255 + 0.5).to(tt.torch.uint8).cpu().numpy()
    info = dict(style=style, params=p, image2_start=i2, n_frames=n, land=tl["land"], beat_frames=round(tl["bf"], 2), device=str(dev),
                depth_a=srcA, depth_b=srcB, setup_s=round(t_setup, 2), render_s=round(time.time() - t0, 2))
    if hasattr(eng, "ink") and eng.ink is not None:
        info.update(ink_bgr=[round(float(x)) for x in eng.ink], ink_contrast=round(float(eng.ink_cr), 2))
    if hasattr(eng, "k0"):
        info["strokes"] = len(eng.k0)
    return X.Result(frames, matte, info)


def make_render(style: str, **params):
    """pivot_cut.apply_cut-compatible callable: render(a, ma, b, mb, da, db, _params) -> Result (frames, matte, info image2_start)."""
    def render(a, ma, b, mb, da=None, db=None, _params=None):
        return render_cut(style, a, ma, b, mb, da, db, params)
    return render


def run_job(job, out_dir, pad=6):
    style, prm = job["style"], dict(job.get("params") or {})
    n, i2 = window_frames(style, prm)
    hw = None
    srcs = {}
    for side in ("a", "b"):
        s = job[side]
        clip = X.read_video(s["clip"], hw)
        hw = hw or tuple(clip.shape[1:3])
        mat = X.read_mattes(s["matte"], hw) if s.get("matte") else None
        dep = cv2.imread(s["depth"], cv2.IMREAD_GRAYSCALE) if s.get("depth") else None
        if s.get("depth_npy"):                           # per-frame Video Depth Anything (T,H,W) disparity, high = near
            dep = np.load(s["depth_npy"]).astype(np.float32)
        srcs[side] = (clip, mat, dep)
    ca, ma, da = srcs["a"]
    cb, mb, db = srcs["b"]
    a0 = max(pad, len(ca) - i2) if job["a"]["clip"] != job["b"]["clip"] else pad
    a_slice = np.array([ca[X.pingpong(len(ca), a0 + k)] for k in range(n)])
    ma_slice = None if ma is None else np.array([ma[min(X.pingpong(len(ca), a0 + k), len(ma) - 1)] for k in range(n)])
    if job["a"]["clip"] == job["b"]["clip"]:             # one shot (word to word): B = the same clip continuing
        cb = np.array([ca[X.pingpong(len(ca), a0 + i2 + k)] for k in range(len(ca))])
        mb = None if ma is None else np.array([ma[min(X.pingpong(len(ca), a0 + i2 + k), len(ma) - 1)] for k in range(len(ca))])
    if da is not None and da.ndim == 3:
        da = np.array([da[min(X.pingpong(len(ca), a0 + k), len(da) - 1)] for k in range(n)])
    if job["a"]["clip"] == job["b"]["clip"] and da is not None and da.ndim == 3:
        db = da[i2:] if len(da) > i2 + 1 else da
    t0 = time.time()
    res = render_cut(style, a_slice, ma_slice, cb, mb, da, db, prm)
    wall = time.time() - t0
    pre = ca[a0 - pad:a0]
    film = np.concatenate([pre, res.frames])
    name = job["name"]
    X.write_mp4(os.path.join(out_dir, f"{name}.mp4"), film)
    X.write_mp4(os.path.join(out_dir, f"{name}_matte.mp4"), np.repeat(np.concatenate([np.zeros((pad,) + res.matte.shape[1:], np.uint8), res.matte])[..., None], 3, -1))
    land = res.info.get("land", n - 1)
    idx = list(np.linspace(pad, pad + land - 4, 6).round().astype(int)) + [pad + land, pad + n - 1]
    X.write_strip(os.path.join(out_dir, f"{name}_strip.png"), film, idx, [f"f{int(i - pad):02d}" + (" LAND" if i - pad == land else "") for i in idx])
    info = dict(res.info, name=name, wall_s=round(wall, 1), sec_per_frame=round(wall / n, 3), a=job["a"], b=job["b"], a0=int(a0), mp4_frames=int(len(film)),
                size=f"{hw[1]}x{hw[0]}", strip_frames=[int(i - pad) for i in idx])
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"{name}: {style} n={n} land={land} dev={res.info['device']} wall {wall:.1f}s ({wall / n:.3f} s/frame)", flush=True)
    return info


def contact_sheet(out_dir, names, path, tile_w=150, gut=10):
    rows = []
    for nm in names:
        info = json.load(open(os.path.join(out_dir, f"{nm}.json")))
        cap = cv2.VideoCapture(os.path.join(out_dir, f"{nm}.mp4"))
        fr = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fr.append(im)
        pad = 6
        tiles = []
        for k in info["strip_frames"]:
            im = fr[min(pad + k, len(fr) - 1)]
            s = tile_w / im.shape[1]
            im = cv2.resize(im, (tile_w, int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
            tiles.append(im)
            tiles.append(np.full((im.shape[0], gut, 3), 245, np.uint8))
        row = np.hstack(tiles[:-1])
        lab = np.full((22, row.shape[1], 3), 245, np.uint8)
        cv2.putText(lab, f"{nm}   (frames {', '.join(str(k) for k in info['strip_frames'])}; land {info.get('land')})", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        rows += [lab, row, np.full((gut, row.shape[1], 3), 245, np.uint8)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=245) for r in rows]
    sheet = np.vstack(rows)
    sheet = np.pad(sheet, ((gut, 0), (gut, gut), (0, 0)), constant_values=245)
    cv2.imwrite(path, sheet)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("jobs")
    j.add_argument("name")
    j.add_argument("--out", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("dir")
    s.add_argument("out")
    s.add_argument("names", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "job":
        os.makedirs(args.out, exist_ok=True)
        job = [x for x in json.load(open(args.jobs)) if x["name"] == args.name][0]
        run_job(job, args.out)
    else:
        contact_sheet(args.dir, args.names, args.out)


if __name__ == "__main__":
    main()
