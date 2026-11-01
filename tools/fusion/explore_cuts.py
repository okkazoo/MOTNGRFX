#!/usr/bin/env python3
"""explore_cuts - EXPERIMENTAL 3D shot cuts (2026-10-30 transition experiments; nothing here is a default, nothing in pivot_cut / depth_orbit / depth_smooth is edited).

User: "being in a volume that is only revealed as you start travelling forward into it - or better you are already in a volume and the translation shows you the parallax
of the individual panels - also can we try some of them as circles - or before the transition the squares transition into circles with the hard outlines that merge into
overlapping circles like a boolean join and we then travel through that in either direction".

Two engines, both LIVE (A's texture = A's playing frame at every output frame, B's = B's playing frame; B before its take-over frame i2 plays ping-pong |i - i2|, A past
its end ping-pongs: never a freeze), both render the SUBJECT MATTE through the very same geometry (matte follows its clip), no subject split, no hole fill, no type.

  PANEL VOLUME (styles volume_reveal, volume_parallax_lateral, circle_volume, hex_lateral)
      Each picture is a quadtree of flat panels (base cell ~ short/6, split where the depth inside a cell spreads, down to ~short/20), each panel at its own depth Z in
      [1, ratio] from the clip's depth (Video Depth Anything archive; per-frame depth video when given, EMA-smoothed), sized so that the home camera sees exactly the
      frame: frame 0 IS the plain shot. A pinhole camera (f = H) moves; panels are warped by their own homography, painter-sorted by view depth, near-lens fade.
      Gaps between panels show a flat dark ground taken from the clip's palette (not a blur, not a silhouette).
        volume_reveal       forward: the camera travels INTO A's volume (with a small lateral sway so parallax reads from the first move), A's panels part centre-out
                            and fly past the lens, B's volume waits behind and its panels ASSEMBLE from depth (each starts further back, far ones land first) and the
                            camera arrives at B's home.  backward: the camera pulls back out of A, B's panels come from behind the lens and settle in front.
        volume_parallax_lateral   already in the volume: the camera arcs sideways round a pivot in the volume and back (dir left/right); A's panels recede into
                            depth swept along the move, B's panels come forward from far behind, the two volumes pass THROUGH each other (panels reorder), B lands.
        circle_volume       (variant) volume_reveal forward where A's panels first morph square -> circle with a crisp brush outline before the camera moves; B's
                            panels arrive as outlined circles and morph back to squares (outline gone) as they land.
        hex_lateral         (variant) a straight lateral truck (no arc) through interleaved volumes, panels morph square -> hexagon and B's hexagon -> square, no outline.
  CIRCLE UNION (style circle_union, dir forward | backward)  screen-space SDF field per picture
      A's frame = a grid of square cells; they round into circles (SDF rounded-box lerp, corner radius 0 -> half size) with a crisp brush outline drawing on, then the
      circles GROW by depth (near cells big, far cells small) and merge as a hard boolean UNION (min of the SDFs; the outline follows the union boundary only); the
      far cells stay apart, so the union has holes where the picture is far (sky, distance). The camera then travels THROUGH the hole nearest the centre: A's union
      scales up about it and flies past, B behind it is its own union of circles (its own depth) growing from 32 % to its home, which un-merges into circles and back
      to square cells = B's plain frame.  backward = the same film with the pictures swapped and the state run backwards: A merges and recedes into the distance
      through B's hole, B's union comes in from the lens and settles, un-merges, squares, B.
  Outlines are live brush strokes (variable width with thick-thin, small breaks, a second offset sketch stroke, wobble with time, draw-on along the shape), crisp
  1 px anti-aliasing, NO glow, only on the circle / union / panel-shape boundaries (these are the objects), never on the frame border or on square card edges.

Hook into the real cut path = `make_render(style, **params)` returns a callable with apply_cut's `render=` signature: render(a, ma, b, mb, da, db, params) -> object
with .frames (uint8 RGB n,H,W,3), .matte (uint8 n,H,W), .info {"image2_start": i2, ...}; a = A's frames for output frames 0..n-1 (may be shorter: ping-pong), b = B's frames
for output frames i2..; `window_frames(style, params)` -> (n, i2) without rendering (what pivot_cut.plan_cut needs).

CLI (box, scratch tree): python -m tools.fusion.explore_cuts batch <jobs.json> --out <dir>      (one job = style, dir, A / B clip + matte dir + depth png / depth mp4)
CPU numpy + cv2.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402  (normalize_disparity, estimate_depth: imported, not edited)

STYLES = ("volume_reveal", "volume_parallax_lateral", "circle_union", "circle_volume", "hex_lateral")
DEFAULTS = {
    "volume_reveal": dict(dir="forward", hold1=8, travel=44, hold2=8, ratio=3.0, sway=0.22, i2_q=0.55),
    "volume_parallax_lateral": dict(dir="right", hold1=8, travel=44, hold2=8, ratio=3.0, arc_deg=24.0, pivot=2.0, recede=2.2, i2_q=0.5),
    "circle_volume": dict(dir="forward", hold1=6, travel=48, hold2=8, ratio=3.0, sway=0.18, i2_q=0.6),
    "hex_lateral": dict(dir="left", hold1=8, travel=44, hold2=8, ratio=3.0, truck=0.9, recede=2.0, i2_q=0.5),
    "circle_union": dict(dir="forward", hold1=6, travel=52, hold2=6, cols=6, b_scale0=0.32, i2_q=0.55),
}
NEAR = 0.10          # view z below which a panel is culled
STROKE_W = 0.0095    # brush width as a fraction of the frame width (~3.4 px at 360)


# ----------------------------------------------------------------------------------------------- small helpers
def sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def resolve(style: str, params: dict | None = None) -> dict:
    if style not in STYLES:
        raise KeyError(f"unknown explore style {style!r}; one of {STYLES}")
    p = dict(DEFAULTS[style])
    for k, v in dict(params or {}).items():
        if k not in p and k not in ("seed",):
            raise KeyError(f"{style}: unknown param {k!r}")
        p[k] = v
    p.setdefault("seed", 0)
    return p


def window_frames(style: str, params: dict | None = None) -> tuple[int, int]:
    """(n, i2) of a style: n output frames; B takes over (plays its own frame 0) at output frame i2 = first frame with q >= i2_q."""
    p = resolve(style, params)
    n = int(p["hold1"] + p["travel"] + p["hold2"])
    for i in range(n):
        if _q(i, p) >= p["i2_q"]:
            return n, i
    return n, n - 1


def _q(i, p):
    return float(np.clip((i - p["hold1"]) / max(p["travel"], 1), 0.0, 1.0))


def pingpong(n: int, k: int) -> int:
    if n <= 1:
        return 0
    per = 2 * (n - 1)
    k = abs(int(k)) % per
    return k if k < n else per - k


def _palette(frames_bgr):
    """(stroke colour, ground colour) BGR float from the clip's own k-means palette (compose.palette_from_footage when importable)."""
    try:
        from tools.fusion.compose import palette_from_footage
        cent = np.asarray(palette_from_footage(list(frames_bgr[::max(1, len(frames_bgr) // 8)]), k=5), np.float32)
    except Exception:   # noqa: BLE001
        x = np.concatenate([cv2.resize(f, (48, 86)).reshape(-1, 3) for f in frames_bgr[::max(1, len(frames_bgr) // 8)]]).astype(np.float32)
        _, _, cent = cv2.kmeans(x, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
        cent = cent[np.argsort(cent @ np.array([0.0722, 0.7152, 0.2126]))]
    hsv = cv2.cvtColor(np.clip(cent[None], 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    score = hsv[:, 1] / 255.0 * 0.6 + hsv[:, 2] / 255.0
    c = hsv[int(np.argmax(score))].copy()
    c[1] = min(255.0, c[1] * 1.25 + 30.0)          # the stroke = the palette's liveliest colour, pushed brighter so it reads as a drawn line
    c[2] = max(c[2], 235.0)
    stroke = cv2.cvtColor(c[None, None].astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    ground = cent[0] * 0.45
    return stroke, ground


def _disp(d, hw):
    """depth (H,W) | (H,W,c) uint8/float bright = near -> normalised disparity float (h,w)."""
    a = np.asarray(d)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return dd.normalize_disparity(a)


def disp_to_z(dn, ratio):
    inv = 1.0 / ratio
    return 1.0 / (inv + (1.0 - inv) * np.clip(dn, 0.0, 1.0))


# ----------------------------------------------------------------------------------------------- SDF + brush
def sdf_rbox(px, py, hx, hy, rc):
    qx = np.abs(px) - hx + rc
    qy = np.abs(py) - hy + rc
    return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - rc


def sdf_hex(px, py, r):
    k0, k1, k2 = -0.866025404, 0.5, 0.577350269
    x, y = np.abs(px), np.abs(py)
    dot = np.minimum(k0 * x + k1 * y, 0.0) * 2.0
    x = x - dot * k0
    y = y - dot * k1
    x = x - np.clip(x, -k2 * r, k2 * r)
    y = y - r
    return np.hypot(x, y) * np.sign(y)


def brush(d, th, ph, t, w, prog=1.0):
    """alpha of a live brush stroke on the zero set of the SDF d (px): thick-thin along the angle th, small breaks, a thinner offset sketch stroke, wobble
    with time t, drawn on along the angle up to `prog` (tapered head). ph = per-object phase (array or scalar)."""
    n1 = np.sin(3.0 * th + ph * 6.283 + 1.7 * t)
    n2 = np.sin(7.0 * th + ph * 13.0 - 2.3 * t)
    ww = w * (0.62 + 0.28 * n1 + 0.14 * n2)
    ww = np.where(np.sin(5.0 * th + ph * 20.0 + 0.9 * t) > 0.95, 0.0, ww)
    if prog < 1.0:
        u = np.mod(th / 6.283 + 0.5 + ph, 1.0)
        ww = ww * np.clip((prog * 1.1 - u) / 0.08, 0.0, 1.0)
    a1 = np.clip(ww * 0.5 - np.abs(d) + 0.5, 0.0, 1.0)
    off = 1.1 + 0.9 * np.sin(4.0 * th + ph * 9.0 + 1.3 * t)
    a2 = np.clip(ww * 0.17 - np.abs(d - off * (w / 3.4)) + 0.5, 0.0, 1.0) * (n2 > -0.2)
    return np.maximum(a1, a2 * 0.85)


# ----------------------------------------------------------------------------------------------- panel volume
class Vol:
    """one picture as a quadtree of depth panels; textures + Z per output frame from the clip."""

    def __init__(self, frames, mattes, dn_frames, ratio, seed, base_div=6, min_div=20, thr=0.16):
        self.frames, self.mattes = frames, mattes            # BGR uint8 list (n,H,W,3), float (n,H,W)
        H, W = frames[0].shape[:2]
        self.H, self.W = H, W
        short = min(H, W)
        base, mn = short / base_div, max(8, short / min_div)
        dn0 = dn_frames[0]
        nx, ny = max(1, round(W / base)), max(1, round(H / base))
        xs = np.linspace(0, W, nx + 1).round().astype(int)
        ys = np.linspace(0, H, ny + 1).round().astype(int)
        st = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(nx) for j in range(ny)]
        tiles = []
        while st:
            x0, y0, x1, y1 = st.pop()
            blk = dn0[y0:y1, x0:x1]
            spread = float(np.percentile(blk, 90) - np.percentile(blk, 10))
            if spread > thr and (x1 - x0) >= 2 * mn and (y1 - y0) >= 2 * mn:
                xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
                st += [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]
            else:
                tiles.append((x0, y0, x1, y1))
        self.t = np.array(tiles, np.int32)
        rng = np.random.default_rng(seed)
        self.rand = rng.random(len(tiles))
        self.phase = rng.random(len(tiles))
        cx = (self.t[:, 0] + self.t[:, 2]) * 0.5
        cy = (self.t[:, 1] + self.t[:, 3]) * 0.5
        self.cx, self.cy = cx, cy
        self.rc = np.hypot((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2)) / math.sqrt(2)     # 0 centre .. 1 corner
        self.ratio = ratio
        self.z = self._tile_z(dn_frames, ratio)              # (n_frames, K) home depth per frame
        self.dn_t = np.clip(1.0 / np.maximum(self.z[0], 1e-3), 0, 1)

    def _tile_z(self, dn_frames, ratio):
        out, prev = [], None
        for dn in dn_frames:
            ii = cv2.integral(dn.astype(np.float64))
            x0, y0, x1, y1 = self.t.T
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            m = s / np.maximum((x1 - x0) * (y1 - y0), 1)
            prev = m if prev is None else prev * 0.7 + m * 0.3
            out.append(disp_to_z(prev, ratio))
        return np.array(out, np.float32)

    def z_at(self, k):
        return self.z[min(k, len(self.z) - 1)]


def _view(P, C, yaw):
    d = P - C
    cy, sy = math.cos(yaw), math.sin(yaw)
    xv = cy * d[..., 0] - sy * d[..., 2]
    zv = sy * d[..., 0] + cy * d[..., 2]
    return xv, d[..., 1], zv


def _shape_mask(tw, th, kind, m, stroke=None, t=0.0, ph=0.0, prog=1.0, pad=1):
    """alpha (th+2p, tw+2p) of a panel's shape (morph m 0 square .. 1 target) + optional brush stroke alpha on its edge."""
    ys, xs = np.mgrid[0:th + 2 * pad, 0:tw + 2 * pad].astype(np.float32)
    px, py = xs - pad - (tw - 1) * 0.5, ys - pad - (th - 1) * 0.5
    hx, hy = tw * 0.5, th * 0.5
    dsq = sdf_rbox(px, py, hx, hy, 0.0)
    if kind == "square" or m <= 0.0:
        d = dsq
    else:
        r = min(hx, hy)
        tgt = sdf_rbox(px, py, r * 0.94, r * 0.94, r * 0.94) if kind == "circle" else sdf_hex(px, py, r * 0.92)
        d = dsq * (1.0 - m) + tgt * m
    a = np.clip(0.5 - d, 0.0, 1.0)
    a[:pad, :] = 0
    a[-pad:, :] = 0
    a[:, :pad] = 0
    a[:, -pad:] = 0
    sa = None
    if stroke is not None and stroke > 0:
        th_ = np.arctan2(py, px)
        sa = brush(d + stroke * 0.5, th_, ph, t, stroke, prog) * a
    return a, sa


def render_panels(items, H, W, ground):
    """items: list of dict(tex BGR+matte float (th,tw,4), alpha (th+2,tw+2), corners (4,2) screen, zv) -> (BGR float, matte float)."""
    if np.ndim(ground) == 3:
        canvas = np.array(ground, np.float32)
    else:
        canvas = np.empty((H, W, 4), np.float32)
        canvas[..., :3] = ground
        canvas[..., 3] = 0.0
    for it in sorted(items, key=lambda r: -r["zv"]):
        dst = it["corners"]
        bx0, by0 = np.floor(dst.min(0)).astype(int)
        bx1, by1 = np.ceil(dst.max(0)).astype(int) + 1
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, W), min(by1, H)
        if cx1 - cx0 < 1 or cy1 - cy0 < 1:
            continue
        tex, al = it["tex"], it["alpha"]
        th, tw = tex.shape[:2]
        src = np.float32([[0.5, 0.5], [tw + 0.5, 0.5], [tw + 0.5, th + 0.5], [0.5, th + 0.5]])     # padded tile, pixel centres on integers (cv2)
        M = cv2.getPerspectiveTransform(src, (dst - [cx0 + 0.5, cy0 + 0.5]).astype(np.float32))
        texp = cv2.copyMakeBorder(tex, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        wt = cv2.warpPerspective(texp, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wa = cv2.warpPerspective(al, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None] * it.get("g", 1.0)
        reg = canvas[cy0:cy1, cx0:cx1]
        reg[:] = reg * (1.0 - wa) + wt * wa
    return canvas[..., :3], canvas[..., 3]


class PanelCut:
    """the four panel styles. A / B = Vol; frame lookups by output frame index."""

    def __init__(self, style, p, va, vb, stroke_a, stroke_b, ground):
        self.s, self.p, self.A, self.B = style, p, va, vb
        self.stroke_a, self.stroke_b, self.ground = stroke_a, stroke_b, ground
        self.H, self.W = va.H, va.W
        self.f = float(self.H)
        self.n, self.i2 = window_frames(style, p)
        R = p["ratio"]
        self.D = R + 0.6
        sgn = -1.0 if p["dir"] in ("left",) else 1.0
        self.sgn = sgn
        # stagger keys
        if style in ("volume_parallax_lateral", "hex_lateral"):
            for v in (va, vb):
                xn = (v.cx / v.W) if sgn > 0 else (1.0 - v.cx / v.W)
                v.key = 0.70 * xn + 0.30 * v.rand
        else:
            for v in (va, vb):
                v.key = 0.55 * v.rc + 0.45 * v.rand

    # camera ------------------------------------------------------------------------------------------------
    def cam(self, q):
        p, s = self.p, self.s
        if s in ("volume_reveal", "circle_volume"):
            qc = q if s == "volume_reveal" else float(sm((q - 0.18) / 0.82))
            sw = math.sin(math.pi * qc)
            z = self.D * qc * (1.0 if p["dir"] == "forward" else -1.0)
            return np.array([p["sway"] * sw * self.sgn, -0.35 * p["sway"] * sw, z], np.float32), -0.10 * sw * self.sgn, qc
        if s == "volume_parallax_lateral":
            th = math.radians(p["arc_deg"]) * math.sin(math.pi * q) ** 0.85 * self.sgn
            Zp = p["pivot"]
            return np.array([Zp * math.sin(th), -0.05 * abs(math.sin(th)), Zp - Zp * math.cos(th)], np.float32), -th, q
        # hex_lateral: straight truck by `truck` world units, B's volume is built round the end camera
        return np.array([p["truck"] * float(sm(q)) * self.sgn, 0.0, 0.0], np.float32), 0.0, q

    def end_cam(self):
        C, yaw, _ = self.cam(1.0)
        return C

    # per-panel state ---------------------------------------------------------------------------------------
    def _state(self, side, v, q, qc):
        """-> (z_eff_rel (K,), scale (K,), shape morph m, stroke prog, global alpha (K,), origin (3,))"""
        p, s, K = self.p, self.s, len(v.t)
        one = np.ones(K, np.float32)
        if s in ("volume_reveal", "circle_volume"):
            fwd = p["dir"] == "forward"
            if side == "a":
                tau = (0.22 if fwd else 0.50) + (0.50 if fwd else 0.40) * v.key
                sc = 1.0 - sm((qc - tau) / 0.24)
                m = float(sm(q / 0.20)) if s == "circle_volume" else 0.0
                return 0.0, sc, m, float(sm((q - 0.02) / 0.22)), one, np.zeros(3, np.float32)
            tau = 0.30 + 0.36 * np.clip(v.dn_t, 0, 1)          # far panels land first, near last
            land = sm((qc - tau) / 0.34)
            extra = (1.0 if fwd else -0.45) * (1.0 - land)
            sc = 0.85 + 0.15 * land
            m = float(1.0 - sm((q - 0.80) / 0.16)) if s == "circle_volume" else 0.0
            org = np.array([0, 0, self.D if fwd else -self.D], np.float32)
            return extra, sc, m, 1.0, one, org
        rec = p["recede"]
        kind_m = 0.0
        if side == "a":
            tau = 0.08 + 0.42 * v.key
            go = sm((q - tau) / 0.40)
            if s == "hex_lateral":
                kind_m = float(sm(q / 0.25))
            return rec * go, 1.0 - 0.08 * np.sin(np.pi * go), kind_m, 1.0, one, np.zeros(3, np.float32)
        tau = 0.18 + 0.42 * v.key
        land = sm((q - tau) / 0.40)
        if s == "hex_lateral":
            kind_m = float(1.0 - sm((q - 0.78) / 0.20))
        org = self.end_cam() if s == "hex_lateral" else np.zeros(3, np.float32)
        return rec * (1.0 - land), 0.92 + 0.08 * land, kind_m, 1.0, one, org

    def _items(self, side, v, tex_bgr, tex_m, zrel, q, qc, C, yaw, t_sec):
        extra, sc, m, prog, g, org = self._state(side, v, q, qc)
        K = len(v.t)
        extra = np.broadcast_to(np.asarray(extra, np.float32), (K,))
        sc = np.broadcast_to(np.asarray(sc, np.float32), (K,))
        kind = "square"
        stroke = 0.0
        if self.s == "circle_volume":
            kind = "circle"
            stroke = STROKE_W * self.W * 0.85 * float(sm((m - 0.55) / 0.4))
        elif self.s == "hex_lateral":
            kind = "hex"
        col = self.stroke_a if side == "a" else self.stroke_b
        out = []
        f = self.f
        W, H = self.W, self.H
        for k in range(K):
            if sc[k] < 0.02:
                continue
            x0, y0, x1, y1 = v.t[k]
            tw, th = int(x1 - x0), int(y1 - y0)
            zh = float(zrel[k])
            ze = zh + float(extra[k])
            if ze <= 0.05:
                continue
            bleed = 1.0 + 0.7 / max(min(tw, th), 1)
            hx = (tw * 0.5) * zh / f * float(sc[k]) * bleed
            hy = (th * 0.5) * zh / f * float(sc[k]) * bleed
            Xc = (float(v.cx[k]) - W / 2) * ze / f + org[0]
            Yc = (float(v.cy[k]) - H / 2) * ze / f + org[1]
            Zc = ze + org[2]
            P = np.array([[Xc - hx, Yc - hy, Zc], [Xc + hx, Yc - hy, Zc], [Xc + hx, Yc + hy, Zc], [Xc - hx, Yc + hy, Zc]], np.float32)
            xv, yv, zv = _view(P, C, yaw)
            if float(zv.min()) < NEAR:
                continue
            u = f * xv / zv + W / 2
            vv = f * yv / zv + H / 2
            corners = np.stack([u, vv], 1)
            span = corners.max(0) - corners.min(0)
            if span[0] > 10 * W or span[1] > 10 * H:
                continue
            if corners[:, 0].max() < -2 or corners[:, 0].min() > W + 2 or corners[:, 1].max() < -2 or corners[:, 1].min() > H + 2:
                continue
            zmean = float(zv.mean())
            al, sa = _shape_mask(tw, th, kind, m, stroke=stroke, t=t_sec, ph=float(v.phase[k]), prog=prog)
            fade = float(sm((zmean - NEAR) / 0.30))
            tex = np.dstack([tex_bgr[y0:y1, x0:x1], tex_m[y0:y1, x0:x1]]).astype(np.float32)
            if sa is not None:
                s_in = sa[1:-1, 1:-1, None]
                tex[..., :3] = tex[..., :3] * (1.0 - s_in) + col * s_in
                tex[..., 3:] = tex[..., 3:] * (1.0 - s_in)
            out.append(dict(tex=tex, alpha=(al * fade).astype(np.float32), corners=corners, zv=zmean))
        return out

    def frame(self, i, a_bgr, a_m, b_bgr, b_m, ka, kb):
        q = _q(i, self.p)
        C, yaw, qc = self.cam(q)
        t_sec = i / 24.0
        if q <= 0.0:
            return a_bgr.astype(np.float32), a_m
        if q >= 1.0:
            return b_bgr.astype(np.float32), b_m
        items = self._items("a", self.A, a_bgr, a_m, self.A.z_at(ka), q, qc, C, yaw, t_sec)
        items += self._items("b", self.B, b_bgr, b_m, self.B.z_at(kb), q, qc, C, yaw, t_sec)
        ua = 1.0 - float(sm((q - 0.12) / 0.2) if self.s == "circle_volume" else sm(q / 0.12))     # circle_volume: the circles form over A's own frame first
        ub = float(sm((q - 0.84) / 0.13))
        under = np.empty((self.H, self.W, 4), np.float32)
        under[..., :3] = self.ground * (1.0 - ua - ub) + a_bgr * ua + b_bgr * ub
        under[..., 3] = a_m * ua + b_m * ub
        return render_panels(items, self.H, self.W, under)


# ----------------------------------------------------------------------------------------------- circle union
class CircleField:
    """a picture's grid of cells as rounded boxes / circles in LAYER pixel coords."""

    def __init__(self, dn, cols, seed):
        H, W = dn.shape
        self.H, self.W = H, W
        cw = W / cols
        rows = max(1, round(H / cw))
        ch = H / rows
        self.cw, self.ch = cw, ch
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) * cw, (np.arange(rows) + 0.5) * ch)
        self.cx, self.cy = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
        K = len(self.cx)
        rng = np.random.default_rng(seed)
        self.phase = rng.random(K).astype(np.float32)
        small = cv2.resize(dn, (cols, rows), interpolation=cv2.INTER_AREA).ravel()
        self.dn = small
        r0 = min(cw, ch) * 0.5 * 0.86
        self.r0 = r0
        R = min(cw, ch) * (0.50 + 0.55 * small)              # near cells big (merge), far cells small (stay apart -> holes)
        drop = (small < np.percentile(small, 35)) & (rng.random(K) < 0.45)
        R[drop] = 0.0
        self.R = R.astype(np.float32)
        self.edge = np.minimum(np.minimum(self.cx, W - self.cx), np.minimum(self.cy, H - self.cy)).astype(np.float32) - 1.0
        # the hole to fly through: the far cell nearest the centre gets dropped, T = farthest-from-union point in the centre box
        cen = np.hypot((self.cx - W / 2) / W, (self.cy - H * 0.45) / H)
        far = np.argsort(cen + (small > np.percentile(small, 50)) * 10.0)[:1]
        self.R[far] = 0.0
        u, v = np.meshgrid(np.arange(0, W, 2, dtype=np.float32), np.arange(0, H, 2, dtype=np.float32))
        d, _ = self.sdf(u, v, 0.0, 1.0)
        box = (u > W * 0.2) & (u < W * 0.8) & (v > H * 0.22) & (v < H * 0.72)
        dd_ = np.where(box, d, -1e9)
        j = int(np.argmax(dd_))
        self.T = np.array([u.ravel()[j], v.ravel()[j]], np.float32)

    def params(self, p1, p2):
        cw, ch, r0 = self.cw, self.ch, self.r0
        if p2 <= 0.0:
            sh = float(sm((p1 - 0.5) / 0.5))                 # cells keep touching until the corners are round (no straight gap lines = no card borders)
            hx = cw * 0.5 + (r0 - cw * 0.5) * sh
            hy = ch * 0.5 + (r0 - ch * 0.5) * sh
            rc = r0 * float(sm(p1 / 0.6))
            K = len(self.cx)
            return np.full(K, hx, np.float32), np.full(K, hy, np.float32), np.full(K, rc, np.float32)
        r = np.minimum(r0 + (self.R - r0) * p2, np.maximum(self.edge, r0))
        return r, r, r

    def sdf(self, u, v, p1, p2, sway=None):
        hx, hy, rc = self.params(p1, p2)
        best = np.full(u.shape, 1e9, np.float32)
        arg = np.zeros(u.shape, np.int16)
        cx, cy = self.cx, self.cy
        if sway is not None:
            cx = cx + sway[0] * (self.dn - 0.5)
            cy = cy + sway[1] * (self.dn - 0.5)
        reach = float(max(hx.max(), hy.max())) + 4.0
        for k in range(len(cx)):
            if hx[k] < 0.5:
                continue
            px, py = u - cx[k], v - cy[k]
            near = (np.abs(px) < reach * 3) & (np.abs(py) < reach * 3)
            if not near.any():
                continue
            dk = sdf_rbox(px, py, hx[k], hy[k], rc[k])
            m = dk < best
            best = np.where(m, dk, best)
            arg = np.where(m, k, arg)
        return best, arg


class CircleCut:
    def __init__(self, p, front_frames, back_frames, dn_front, dn_back, stroke_f, stroke_b, ground, seed):
        self.p = p
        H, W = front_frames[0].shape[:2]
        self.H, self.W = H, W
        self.F = CircleField(dn_front, p["cols"], seed)
        self.G = CircleField(dn_back, p["cols"], seed + 1)
        self.sf, self.sb, self.ground = stroke_f, stroke_b, ground
        self.n, self.i2 = window_frames("circle_union", p)
        self.xx, self.yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.w = STROKE_W * W

    def _layer(self, fld, img, mat, S, c_from, c_to, p1, p2, prog, stroke_col, stroke_on, t, sway):
        """screen -> layer: (x - c_to) / S + c_from. -> (rgb, matte, coverage alpha)"""
        u = (self.xx - c_to[0]) / S + c_from[0]
        v = (self.yy - c_to[1]) / S + c_from[1]
        d, arg = fld.sdf(u, v, p1, p2, sway)
        ds = d * S
        bleed = 0.5 * (1.0 - float(sm((p1 - 0.5) / 0.25))) if p2 <= 0 else 0.0     # touching squares never show a seam
        inside = (u >= -0.5) & (u <= self.W - 0.5) & (v >= -0.5) & (v <= self.H - 0.5)
        cov = np.clip(0.5 - ds + bleed, 0.0, 1.0) * inside
        rgb = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mt = cv2.remap(mat, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        st = None
        if stroke_on > 0.003:
            th = np.arctan2(v - fld.cy[arg], u - fld.cx[arg])
            st = brush(ds + self.w * 0.15, th, fld.phase[arg], t, self.w, prog) * stroke_on * inside
        return rgb, mt, cov, st

    def frame(self, i, front_bgr, front_m, back_bgr, back_m):
        p = self.p
        q = _q(i, p)
        t = i / 24.0
        if p["dir"] == "backward":
            q = 1.0 - q
        cen = np.array([self.W / 2, self.H / 2], np.float32)
        p1 = float(sm(q / 0.20))
        p2 = float(sm((q - 0.16) / 0.22))
        e = float(sm((q - 0.36) / 0.46))
        swayF = np.array([10.0, -5.0], np.float32) * math.sin(math.pi * min(1, q / 0.42)) * float(sm((p1 - 0.5) / 0.5))   # depth sway only once the cells have separated (touching squares never slide apart into a line)
        cz = 1.02 * e                                      # the camera travels through the front picture's plane (at distance 1): S = 1 / (1 - cz), gone once passed
        S_F = 1.0 / (1.0 - cz) if cz < 0.985 else 1e9
        T = self.F.T
        out = np.empty((self.H, self.W, 3), np.float32)
        out[:] = self.ground
        mo = np.zeros((self.H, self.W), np.float32)
        # back picture (B in forward): its own union, growing from b_scale0 about the hole to its home
        if q >= 0.12:
            b2 = 1.0 - float(sm((q - 0.80) / 0.12))
            b1 = 1.0 - float(sm((q - 0.88) / 0.12))
            S_B = p["b_scale0"] * (1.0 / p["b_scale0"]) ** e
            c_to = T + (cen - T) * e
            on = float(sm((b1 - 0.55) / 0.35))
            rgb, mt, cov, st = self._layer(self.G, back_bgr, back_m, S_B, cen, c_to, b1, b2, 1.0, self.sb, on, t, None)
            c3 = cov[..., None]
            out = out * (1.0 - c3) + rgb * c3
            mo = mo * (1.0 - cov) + mt * cov
            if st is not None:
                out = out * (1.0 - st[..., None]) + self.sb * st[..., None]
                mo = mo * (1.0 - st)
        if S_F < 1e8:
            on = float(sm((p1 - 0.55) / 0.35)) * float(1.0 - sm((e - 0.75) / 0.25))
            rgb, mt, cov, st = self._layer(self.F, front_bgr, front_m, S_F, T, T, p1, p2, float(sm((q - 0.02) / 0.2)), self.sf, on, t, swayF)
            c3 = cov[..., None]
            out = out * (1.0 - c3) + rgb * c3
            mo = mo * (1.0 - cov) + mt * cov
            if st is not None:
                out = out * (1.0 - st[..., None]) + self.sf * st[..., None]
                mo = mo * (1.0 - st)
        return out, mo


# ----------------------------------------------------------------------------------------------- public render
class Result:
    def __init__(self, frames, matte, info):
        self.frames, self.matte, self.info = frames, matte, info


def _as_list_rgb(x):
    a = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.01 else 1.0) + 0.5, 0, 255).astype(np.uint8)
    return a[..., :3]


def _mattes(m, n, hw):
    if m is None:
        return np.zeros((max(n, 1),) + hw, np.float32)
    a = np.asarray(m.detach().cpu().numpy() if hasattr(m, "detach") else m).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    if a.max() > 1.01:
        a = a / 255.0
    if a.shape[1:3] != hw:
        a = np.stack([cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) for x in a])
    return a


def _depths(d, frames_bgr, hw):
    """normalised disparity per frame (list); None -> Depth-Anything-V2-Small on the first frame (CPU), static."""
    if d is None:
        dep = dd.estimate_depth(frames_bgr[0])
        return [_disp(dep, hw)], "depth-anything-v2-small (estimated, static)"
    a = np.asarray(d)
    if a.ndim == 4:
        a = a[..., 0]
    if a.ndim == 2:
        return [_disp(a, hw)], "given (1 frame)"
    if a.ndim == 3 and a.shape[-1] in (1, 3, 4) and a.shape[0] > 16:
        return [_disp(a, hw)], "given (1 frame)"
    return [_disp(x, hw) for x in a], f"given ({len(a)} frames)"


def make_render(style: str, **params):
    """apply_cut-compatible render callable: render(a, ma, b, mb, da, db, params=None)."""
    def render(a, ma, b, mb, da=None, db=None, _params=None):
        return render_cut(style, a, ma, b, mb, da, db, dict(params, **({} if not isinstance(_params, dict) else {k: v for k, v in _params.items() if k in DEFAULTS[style] or k == "seed"})))
    return render


def render_cut(style, a, ma, b, mb, da=None, db=None, params=None, on_frame=None) -> Result:
    """a: A's frames for output frames 0.. (RGB uint8 or float), b: B's frames for output frames i2.. ; mattes (T,H,W); da / db depth (one frame or a clip, bright = near)."""
    p = resolve(style, params)
    n, i2 = window_frames(style, p)
    A = _as_list_rgb(a)
    Bc = _as_list_rgb(b)
    hw = tuple(A.shape[1:3])
    if Bc.shape[1:3] != hw:
        Bc = np.stack([cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for x in Bc])
    A_bgr = A[..., ::-1].copy()
    B_bgr = Bc[..., ::-1].copy()
    mA, mB = _mattes(ma, len(A), hw), _mattes(mb, len(Bc), hw)
    dA, srcA = _depths(da, A_bgr, hw)
    dB, srcB = _depths(db, B_bgr, hw)
    sa, ga = _palette(A_bgr)
    sb, gb = _palette(B_bgr)
    ground = (ga * 0.5 + gb * 0.5)
    seed = int(p.get("seed", 0))
    t0 = time.time()
    frames = np.zeros((n,) + hw + (3,), np.uint8)
    matte = np.zeros((n,) + hw, np.uint8)
    rev = p.get("dir") == "backward" and style == "circle_union"
    if style == "circle_union":
        fr_d, bk_d = (dB[0], dA[0]) if rev else (dA[0], dB[0])
        eng = CircleCut(p, (B_bgr if rev else A_bgr), (A_bgr if rev else B_bgr), fr_d, bk_d, sb if rev else sa, sa if rev else sb, ground, seed)
    else:
        va = Vol(A_bgr, mA, dA, p["ratio"], seed)
        vb = Vol(B_bgr, mB, dB, p["ratio"], seed + 1)
        eng = PanelCut(style, p, va, vb, sa, sb, ground)
    for i in range(n):
        ka = pingpong(len(A), i)
        kb = pingpong(len(Bc), i - i2)
        a_img, a_m = A_bgr[ka], mA[min(ka, len(mA) - 1)]
        b_img, b_m = B_bgr[kb], mB[min(kb, len(mB) - 1)]
        if style == "circle_union":
            if rev:
                img, mt = eng.frame(i, b_img, b_m, a_img, a_m)
            else:
                img, mt = eng.frame(i, a_img, a_m, b_img, b_m)
        else:
            img, mt = eng.frame(i, a_img, a_m, b_img, b_m, ka, kb)
        frames[i] = np.clip(img[..., ::-1] + 0.5, 0, 255).astype(np.uint8)
        matte[i] = np.clip(mt * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if on_frame:
            on_frame(i, n)
    info = dict(style=style, params=p, image2_start=i2, n_frames=n, depth_a=srcA, depth_b=srcB, render_s=round(time.time() - t0, 2),
                stroke_a=[round(float(x)) for x in sa[::-1]], stroke_b=[round(float(x)) for x in sb[::-1]], ground=[round(float(x)) for x in ground[::-1]])
    if style != "circle_union":
        info["panels"] = [int(len(eng.A.t)), int(len(eng.B.t))]
    return Result(frames, matte, info)


# ----------------------------------------------------------------------------------------------- CLI (box)
def read_video(path, hw=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if hw is not None and f.shape[:2] != hw:
            f = cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        out.append(f[..., ::-1])
    cap.release()
    return np.array(out)


def read_mattes(d, hw):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    return np.array([cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) for f in fs], np.float32) / 255.0


def write_mp4(path, frames_rgb, fps=24):
    h, w = frames_rgb.shape[1:3]
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    pr.stdin.write(np.ascontiguousarray(frames_rgb).tobytes())
    pr.stdin.close()
    pr.wait()


def write_strip(path, frames_rgb, idx, labels, width=200):
    tiles = []
    for i, lab in zip(idx, labels):
        f = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        s = width / f.shape[1]
        f = cv2.resize(f, (width, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(f, (0, 0), (width, 15), (0, 0, 0), -1)
        cv2.putText(f, lab, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    cv2.imwrite(path, np.hstack(tiles))


def run_job(job, out_dir, pad=6):
    """job: name, style, params, a{clip, matte, depth, depth_video}, b{..}. A's frames for the window = the last (i2) frames before its end + onward ping-pong;
    the mp4 = `pad` frames of plain A before the window + the window + `pad` frames of plain B after it."""
    style, prm = job["style"], dict(job.get("params") or {})
    n, i2 = window_frames(style, prm)
    hw = None
    srcs = {}
    for side in ("a", "b"):
        s = job[side]
        clip = read_video(s["clip"], hw)
        hw = hw or tuple(clip.shape[1:3])
        mat = read_mattes(s["matte"], hw) if s.get("matte") else None
        dep = None
        if s.get("depth_video"):
            dep = read_video(s["depth_video"], hw)[..., 0]
        elif s.get("depth"):
            dep = cv2.imread(s["depth"], cv2.IMREAD_GRAYSCALE)
        srcs[side] = (clip, mat, dep)
    ca, ma, da = srcs["a"]
    cb, mb, db = srcs["b"]
    a0 = max(pad, len(ca) - i2)                     # the window starts i2 frames before A's end: B takes over where A would have ended
    a_slice = np.array([ca[pingpong(len(ca), a0 + k)] for k in range(n)])
    ma_slice = None if ma is None else np.array([ma[min(pingpong(len(ca), a0 + k), len(ma) - 1)] for k in range(n)])
    da_slice = da if (da is None or da.ndim == 2) else np.array([da[min(pingpong(len(ca), a0 + k), len(da) - 1)] for k in range(n)])
    t0 = time.time()
    res = render_cut(style, a_slice, ma_slice, cb, mb, da_slice, db, prm)
    wall = time.time() - t0
    pre = ca[a0 - pad:a0]
    b_len = n - i2
    post = np.array([cb[pingpong(len(cb), b_len + k)] for k in range(pad)])
    film = np.concatenate([pre, res.frames, post])
    name = job["name"]
    write_mp4(os.path.join(out_dir, f"{name}.mp4"), film)
    mfilm = np.concatenate([np.zeros((pad,) + res.matte.shape[1:], np.uint8), res.matte, np.zeros((pad,) + res.matte.shape[1:], np.uint8)])
    write_mp4(os.path.join(out_dir, f"{name}_matte.mp4"), np.repeat(mfilm[..., None], 3, -1))
    idx = np.linspace(pad, pad + n - 1, 8).round().astype(int)
    write_strip(os.path.join(out_dir, f"{name}_strip.png"), film, idx, [f"f{int(i - pad):02d}" + (" B" if i - pad >= i2 else "") for i in idx])
    info = dict(res.info, name=name, wall_s=round(wall, 1), sec_per_frame=round(wall / n, 3), a=job["a"], b=job["b"], a0=int(a0), mp4_frames=int(len(film)),
                size=f"{hw[1]}x{hw[0]}")
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"{name}: {style} n={n} i2={i2} wall {wall:.1f}s ({wall / n:.2f} s/frame) {info.get('panels', '')}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch")
    b.add_argument("jobs")
    b.add_argument("--out", required=True)
    b.add_argument("--only", default="")
    args = ap.parse_args(argv)
    jobs = json.load(open(args.jobs))
    os.makedirs(args.out, exist_ok=True)
    keep = set(filter(None, args.only.split(",")))
    for j in jobs:
        if keep and j["name"] not in keep:
            continue
        run_job(j, args.out)


if __name__ == "__main__":
    main()
