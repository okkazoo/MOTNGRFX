#!/usr/bin/env python3
"""Depth-parallax dolly transitions between stills (CPU only, numpy + OpenCV; Depth-Anything-V2-Small for depth).

Ported from comfyui-projects/LTX_seamless/mtrack/local/ (not imported, copied in spirit):
  grow_geometry.py          -> `estimate_depth` (transformers depth-estimation pipeline, device=-1, DA-V2-Small)
  depth_to_mesh_dolly.py    -> unprojection (OpenCV x right, y down, z forward; Z = ZNEAR / clip(disp, floor, 1);
                               f from HFOV), z-buffer splat with footprint-sized blocks, LDI back layer for
                               disocclusions (`build_layer`), hole fill (`_fill_holes`, then cv2.inpaint)
  build_camera_path_editor  -> HOME camera ("flatStart"): frame 0 of the move reproduces the source image exactly.
                               Here home = camera at the origin, identity rotation, focal = the unprojection focal.

Scene: image k is a depth mesh unprojected from a camera at the origin, then translated k spans along the travel
direction D (one span = the extent that pushes the FARTHEST content of a frame out of view). One camera, no
rotation, translates along D with a quintic ease; so at integer positions the camera sits exactly at image k's
home and sees that image's source pixels. Near content moves further per camera unit than far content (parallax),
so the layers separate as image k leaves and image k+1 arrives.

v003 per-depth motion (`Scene.render_t`, `frames`): the camera stays at home and each image's points are translated
span*dist along D by their OWN curve, delayed by farness w (0 near .. 1 far, from 1/Z): the leaving image's far
content starts late (delay lag*w*T, quintic), the forming image is the time-reverse (delay lag*(1-w)*T, near lands
last) with a damped-spring landing (overshoot, settle). lag=0 + spring=False == the rigid one-camera move above
(`render_c`, `timeline`). Loop: image 1 is appended as the 4th image (same Layer), the hold is split across the end /
start. Motion blur sub-frames re-evaluate the per-point curves.

The alpha matte (subject = white) is used to make the subject a separate layer: its depth edge is sharpened (the
DA halo is replaced by the background depth), and the background BEHIND the subject is rebuilt (push-pull fill of
disparity + colour from the outside) as the LDI back layer that shows when the subject slides off the background.

CLI:
  python tools/fusion/depth_dolly.py depth   [--out VERSION_DIR]   # depth_s{1,2,3}a: .npy in local/, .png in review/
  python tools/fusion/depth_dolly.py render  [--out VERSION_DIR] [--dirs up,down,left,right] [--lag .45] [--overshoot .08]
                                              [--settle 18] [--no-loop] [--depth-ratio 3] [--depth-from DIR]
VERSION_DIR defaults to 17_depth-dolly/v001: videos -> out/motion-graphic_depth-dolly_v001_<dir>.mp4,
depth PNGs + dolly_sheet.png -> review/, depth caches + scratch frames -> local/.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

DIRS = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "down": (0.0, 1.0), "up": (0.0, -1.0)}   # camera travel (+y = down)


# ----------------------------------------------------------------------------------------------- params
@dataclass
class Params:
    hfov: float = 55.0          # horizontal FOV of the home camera (monocular depth has no real focal length)
    depth_ratio: float = 3.0    # far Z / near Z (Z in [1, ratio]); bigger = stronger parallax, longer span
    span_margin: float = 1.08   # span = margin * farthest-content extent -> image k is fully off at the end
    push: float = 0.10          # slight forward push in world units (peaks mid-travel, 0 at both homes)
    near_clip: float = 0.2
    # subject / LDI
    core_erode: int = 2         # px the matte is eroded to get trustworthy subject depth
    halo_dilate: int = 3        # px outside the matte whose DA depth is replaced by background depth
    subject_relief: float = 0.45  # 0 = subject is a flat card at its median depth, 1 = raw DA relief
    subject_gap: float = 0.07   # min disparity step subject -> background behind it (sharp occlusion boundary)
    ldi_jump: float = 0.045     # per-px disparity jump that counts as a real depth edge
    ldi_r: int = 14             # px behind a jump taken as occluder for the generic LDI
    ldi_margin: float = 0.03
    depth_blur: float = 2.0     # px, before the guided filter
    guided_r: int = 12          # guided-filter radius for the depth (0 = off)
    guided_eps: float = 2e-3
    # raster
    ease_hold: int = 24         # frames per hold (1 s @ 24)
    ease_travel: int = 29       # frames per travel (~1.2 s)
    fps: int = 24
    max_block: int = 14
    fill_iters: int = 400       # hole growth passes for pads / large voids
    crack_iters: int = 3        # passes that close cracks inside real content before pads show
    shutter: float = 0.55       # motion blur: fraction of a frame the shutter is open
    sub_px: float = 14.0        # one sub-render per this many px of near-content travel per frame
    max_sub: int = 7
    # per-depth motion (v003)
    lag: float = 0.45           # depth delay as a fraction of ease_travel: the travel window is T*(1+lag) frames;
                                # leaving image: far starts late; forming image: near lands last (time-reverse)
    spring: bool = True         # forming image lands with a damped spring (False = quintic like the leaving image)
    overshoot: float = 0.08     # first overshoot of NEAR content past home, as a fraction of the frame width
    settle: int = 18            # frames after the last landing until everything is at rest (inside the hold)
    ring_period: float = 9.0    # frames per oscillation of the settle
    ring_lead: float = 0.2      # the ring starts this fraction of ease_travel before the base ease ends
    tol_px: float = 0.1         # residual of near content when the spring is cut to rest
    zbias: float = 0.08         # forming image sorts this much nearer than the leaving one at equal depth
    loop: bool = True           # 1 -> 2 -> 3 -> 1
    # v004: random patch delays
    delay_mode: str = "depth"   # depth = delay from farness (v003); random = per-patch random delay (lag*T*r)
    cell: float = 28.0          # random mode: patch size in px at 432 wide (jittered-grid Voronoi cells)
    seed: int = 0
    hold_start: int | None = None   # override the first / last hold length (frames); None = ease_hold (or the loop split)
    hold_end: int | None = None


# ----------------------------------------------------------------------------------------------- helpers
def smootherstep(p):
    p = np.clip(p, 0.0, 1.0)
    return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)           # quintic ease-in-out


def lag_frames(p):
    """Largest depth delay in whole frames (lag is a fraction of ease_travel)."""
    return int(round(p.lag * p.ease_travel))


def window(p):
    """Frames of one travel window: every depth moves for ease_travel frames, delays spread over lag_frames."""
    return p.ease_travel + lag_frames(p)


def rest_time(p):
    """First frame (since the window start) at which everything is at rest on the new image."""
    return float(window(p) + (max(p.settle, 1) - 1 if p.spring else 0))


@functools.lru_cache(maxsize=32)
def _spring_cfg(T, S, P, lead, os_c, tol_c):
    """Arrival curve x(u) (u = frames since this depth started; x: 0 -> 1): a quintic ease-in-out plus a damped
    ring that starts `lead*T` before the ease ends, so there is no stall at home. The ring amplitude is solved so
    max(x) - 1 == os_c exactly; its decay is set so the residual is tol_c at u_end = T + S - 1 where x is cut to 1."""
    om = 2.0 * math.pi / P
    ur = T * (1.0 - lead)
    u_end = float(T + S - 1)
    dur = max(u_end - ur, 1.0)
    gam = math.log(max(os_c / tol_c, 2.0)) / (0.65 * dur)         # envelope hits tol well before u_end: 1-2 visible swings
    ron = P / 6.0
    uu = np.linspace(0.0, u_end, 4000)
    r = np.maximum(uu - ur, 0.0)
    base = smootherstep(uu / T) - 1.0
    gg = np.exp(-gam * r) * np.sin(om * r) * smootherstep(r / ron)
    lo, hi = 0.0, 50.0 * os_c
    for _ in range(60):
        A = 0.5 * (lo + hi)
        if (base + A * gg).max() < os_c:
            lo = A
        else:
            hi = A
    return (0.5 * (lo + hi), gam, om, ur, ron, u_end, float(T))


def spring_curve(u, cfg):
    """Damped-spring arrival, 0 -> overshoot -> 1; exactly 1 from u_end on, 0 for u <= 0."""
    A, gam, om, ur, ron, u_end, T = cfg
    u = np.asarray(u, np.float64)
    r = np.maximum(u - ur, 0.0)
    x = smootherstep(u / T) + A * np.exp(-gam * r) * np.sin(om * r) * smootherstep(r / ron)
    return np.where(u >= u_end, 1.0, x)


def out_progress(t, w, p):
    """Leaving image: progress 0 -> 1 of a point of farness w (0 near, 1 far) t frames into the window. Far starts
    later (delay lag*w*T), near leaves first."""
    L = lag_frames(p)
    return smootherstep((np.asarray(t, np.float64) - L * np.asarray(w, np.float64)) / p.ease_travel)


def in_progress(t, w, p, cfg=None):
    """Forming image (time-reverse): far arrives first, near last (delay lag*(1-w)*T); spring landing if cfg."""
    L = lag_frames(p)
    u = np.asarray(t, np.float64) - L * (1.0 - np.asarray(w, np.float64))
    if p.spring and cfg is not None:
        return spring_curve(u, cfg)
    return smootherstep(u / p.ease_travel)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


def unique_path(path):
    """Never overwrite: foo.png -> foo_g02.png -> foo_g03.png ..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists("%s_g%02d%s" % (base, n, ext)):
        n += 1
    return "%s_g%02d%s" % (base, n, ext)


def push_pull(val, known, levels=9):
    """Fill unknown pixels smoothly from known ones (pyramid push-pull). val (H,W) or (H,W,C); known bool (H,W)."""
    v = np.asarray(val, np.float32)
    two = v.ndim == 2
    if two:
        v = v[..., None]
    w = known.astype(np.float32)
    pyr = [(v * w[..., None], w)]
    for _ in range(levels):
        a, b = pyr[-1]
        h, ww = b.shape
        if min(h, ww) < 4:
            break
        a2 = cv2.resize(a, (ww // 2, h // 2), interpolation=cv2.INTER_AREA).reshape(h // 2, ww // 2, -1)
        b2 = cv2.resize(b, (ww // 2, h // 2), interpolation=cv2.INTER_AREA)
        pyr.append((a2, b2))
    a, b = pyr[-1]
    cur = a / np.maximum(b, 1e-6)[..., None]
    for a, b in reversed(pyr[:-1]):
        up = cv2.resize(cur, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(b.shape[0], b.shape[1], -1)
        al = np.clip(b, 0.0, 1.0)[..., None]
        cur = al * (a / np.maximum(b, 1e-6)[..., None]) + (1.0 - al) * up
    return cur[..., 0] if two else cur


def guided_filter(guide, src, r=8, eps=1e-2):
    """Edge-aware smoothing of `src` (float32 0..1) following the edges of `guide` (gray 0..1): lifts the DA patch
    staircase onto the real image edges."""
    k = (2 * r + 1, 2 * r + 1)
    mg, ms = cv2.boxFilter(guide, -1, k), cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mg * ms
    var = cv2.boxFilter(guide * guide, -1, k) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def nearest_fill(val, valid):
    """Each invalid pixel takes its nearest valid pixel's value (Voronoi, as in the reference's gap fill)."""
    inv = (~valid).astype(np.uint8)
    _, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lbl.max()) + 1, np.float32)
    lut[lbl[valid]] = val[valid]
    return np.where(valid, val, lut[lbl]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- depth model
_PIPE = None


def estimate_depth(img_bgr):
    """Depth-Anything-V2-Small on CPU. Returns relative DISPARITY (high = near), float32, size of the input."""
    global _PIPE
    from PIL import Image
    from transformers import pipeline
    import torch
    if _PIPE is None:
        _PIPE = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = _PIPE(Image.fromarray(rgb))
    pd = out["predicted_depth"]
    if pd.ndim == 2:
        pd = pd[None]
    pd = torch.nn.functional.interpolate(pd[None].float(), size=img_bgr.shape[:2], mode="bicubic",
                                         align_corners=False)[0, 0]
    return pd.numpy().astype(np.float32)


def normalize_disparity(d, lo_pct=1.0, hi_pct=99.0):
    lo, hi = np.percentile(d, lo_pct), np.percentile(d, hi_pct)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def disp_to_z(dn, p):
    """Normalised disparity (1 = near) -> world Z in [1, depth_ratio], disparity-linear."""
    inv_r = 1.0 / p.depth_ratio
    return 1.0 / (inv_r + (1.0 - inv_r) * np.clip(dn, 0.0, 1.0))


# ----------------------------------------------------------------------------------------------- random patches
def patch_labels(h, w, cell, seed, subj=None):
    """Voronoi cells of a jittered grid (one seed per `cell` px square), (h,w) int labels. With `subj` (bool matte) a
    cell is split along the matte: label = cell * 2 + subject, so subject and background pieces separate."""
    rng = np.random.default_rng(seed)
    gx, gy = int(math.ceil(w / cell)) + 1, int(math.ceil(h / cell)) + 1
    sx = (np.arange(gx)[None, :] + rng.random((gy, gx))) * cell
    sy = (np.arange(gy)[:, None] + rng.random((gy, gx))) * cell
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ci = np.clip((ys // cell).astype(np.int32), 0, gy - 1)
    cj = np.clip((xs // cell).astype(np.int32), 0, gx - 1)
    best = np.full((h, w), np.inf, np.float32)
    lab = np.zeros((h, w), np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ii = np.clip(ci + dy, 0, gy - 1)
            jj = np.clip(cj + dx, 0, gx - 1)
            d = (sx[ii, jj] - xs) ** 2 + (sy[ii, jj] - ys) ** 2
            better = d < best
            best = np.where(better, d, best)
            lab = np.where(better, ii * gx + jj, lab)
    lab = lab * 2
    if subj is not None:
        lab = lab + subj.astype(np.int32)
    return lab


# ----------------------------------------------------------------------------------------------- layers
@dataclass
class Layer:
    """One image as a depth mesh at mesh resolution (camera at the origin, translated by `offset` in the world)."""
    col: np.ndarray                 # (H,W,3) uint8 BGR, the home view
    disp: np.ndarray                # (H,W) normalised disparity of the front surface (subject edge cleaned)
    bmask: np.ndarray               # (H,W) bool, pixels that also get a BACK point (LDI)
    bdisp: np.ndarray               # (H,W) disparity of the back layer (valid under bmask)
    bcol: np.ndarray                # (H,W,3) colour of the back layer
    subj: np.ndarray | None = None  # (H,W) bool subject matte (for QC)
    params: Params = field(default_factory=Params)
    u0: float = 0.0                 # source-pixel coordinate of grid column 0 / row 0 (pads extend beyond the frame)
    v0: float = 0.0
    sortbias: float = 0.0           # added to Z for ordering only (pads always sit behind real content)
    f_ref: float = 0.0              # focal of the parent image (pads share it); 0 = own width
    cx0: float = 0.0                # principal point of the parent image (pads)
    cy0: float = 0.0
    wrange: tuple | None = None     # (lo, hi) of 1/Z that maps to farness 1 / 0 (pads share the parent's)
    parent: object | None = None    # pads: the Layer they were mirrored from (random patch delays follow it)
    flip_ax: int = 0                # pads: cv2.flip code used to mirror it

    def __post_init__(self):
        self.H, self.W = self.disp.shape
        p = self.params
        self.f = self.f_ref or (self.W / 2.0) / math.tan(math.radians(p.hfov) / 2.0)
        if self.f_ref:
            self.cx, self.cy = self.cx0, self.cy0
        else:
            self.cx, self.cy = self.W / 2.0, self.H / 2.0
        uu, vv = np.meshgrid(np.arange(self.W, dtype=np.float32) + self.u0,
                             np.arange(self.H, dtype=np.float32) + self.v0)
        self.grids = []
        for disp, cont_thr in ((self.disp, p.ldi_jump),
                               (np.where(self.bmask, self.bdisp, self.disp).astype(np.float32), p.ldi_jump)):
            Z = disp_to_z(disp, p).astype(np.float32)
            X = (uu - self.cx) / self.f * Z
            Y = (vv - self.cy) / self.f * Z
            cx_ = np.abs(np.diff(disp, axis=1)) < cont_thr          # (H,W-1) neighbours on one continuous surface
            cy_ = np.abs(np.diff(disp, axis=0)) < cont_thr
            self.grids.append((X, Y, Z, cx_, cy_))
        # farness w in [0,1] per grid point (0 nearest, 1 farthest), from the same Z: linear in 1/Z (= screen
        # parallax), normalised by the 2..98 percentile of the image's own front surface
        invz = [1.0 / g[2] for g in self.grids]
        if self.wrange is None:
            lo, hi = np.percentile(invz[0], (2.0, 98.0))
            self.wrange = (float(lo), float(hi))
        lo, hi = self.wrange
        self.wfar = [np.clip((hi - iz) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32) for iz in invz]

    def project(self, C, offset, which=0):
        """Project grid `which` (0 front, 1 back) for camera position C (3,) and layer offset (3,).
        Returns u, v, zc (H,W each) float32."""
        X, Y, Z, _, _ = self.grids[which]
        if callable(offset):                                   # per-point motion: offset(layer, which) -> (ox, oy, oz)
            offset = offset(self, which)
        xc = X + (offset[0] - C[0])
        yc = Y + (offset[1] - C[1])
        zc = Z + (offset[2] - C[2])
        zs = np.where(zc > self.params.near_clip, zc, 1.0)
        return (self.f * xc / zs + self.cx).astype(np.float32), (self.f * yc / zs + self.cy).astype(np.float32), \
            zc.astype(np.float32)

    def block_sizes(self, u, v, zc, which):
        """Per-point splat size: footprint, or the stretch to the CONTINUOUS neighbours (fills cracks, never
        bridges a real depth edge)."""
        _, _, Z, cx_, cy_ = self.grids[which]
        fp = self.f * (Z / self.f) / np.where(zc > self.params.near_clip, zc, 1.0)     # projected size of 1 src px
        sx = np.zeros_like(u)
        dx = np.abs(np.diff(u, axis=1)) * cx_ + np.abs(np.diff(v, axis=1)) * cx_
        sx[:, :-1] = np.maximum(sx[:, :-1], dx)
        sx[:, 1:] = np.maximum(sx[:, 1:], dx)
        dy = (np.abs(np.diff(v, axis=0)) + np.abs(np.diff(u, axis=0))) * cy_
        sx[:-1] = np.maximum(sx[:-1], dy)
        sx[1:] = np.maximum(sx[1:], dy)
        sz = np.maximum(fp, sx)
        return np.clip(np.ceil(sz - 0.002), 1, self.params.max_block).astype(np.int32)


def build_layer(img_bgr, dn, matte, p: Params):
    """img_bgr (H,W,3) uint8 and dn (H,W) normalised disparity and matte (H,W) float 0..1, all at MESH resolution.
    Cleans the subject depth edge with the matte and builds the LDI back layer."""
    H, W = dn.shape
    k3 = np.ones((3, 3), np.uint8)
    subj = (matte > 0.5)
    su8 = subj.astype(np.uint8)
    dn = dn.astype(np.float32)
    if p.guided_r:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if p.depth_blur > 0:                                  # DA works on 14-px patches: kill the staircase, then
            dn = cv2.GaussianBlur(dn, (0, 0), p.depth_blur)   # let the guide put the edges back where the image has them
        dn = np.clip(guided_filter(g, dn, p.guided_r, p.guided_eps), 0.0, 1.0).astype(np.float32)
    if subj.any():
        core = cv2.erode(su8, k3, iterations=p.core_erode) > 0
        if not core.any():
            core = subj
        sdil = cv2.dilate(su8, k3, iterations=p.halo_dilate) > 0
        # background depth with subject + halo removed (what is BEHIND the person)
        bg = push_pull(dn, ~sdil)
        # subject depth: core only, edge ring takes the nearest core value; flattened toward its median
        sd = nearest_fill(cv2.GaussianBlur(dn, (0, 0), 1.2), core)
        med = float(np.median(dn[core]))
        sd = med + (sd - med) * p.subject_relief
        # always in front of what is behind it, with a SMOOTH floor (a stepped floor would crack the body open)
        bg_s = cv2.GaussianBlur(bg, (0, 0), 12.0)
        sd = np.clip(np.maximum(sd, bg_s + p.subject_gap), 0.0, 1.0)
        front = np.where(subj, sd, np.where(sdil, bg, dn)).astype(np.float32)
        bgmap = np.where(sdil, bg, dn).astype(np.float32)
    else:
        sdil = np.zeros_like(subj)
        bg = dn
        front = dn.astype(np.float32)
        bgmap = front
    # generic LDI (reference): near side of real depth jumps = occluder, back disparity filled from the far side
    edge = np.zeros((H, W), bool)
    edge[:, 1:] |= np.abs(np.diff(bgmap, axis=1)) > p.ldi_jump
    edge[1:, :] |= np.abs(np.diff(bgmap, axis=0)) > p.ldi_jump
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.ldi_r + 1, 2 * p.ldi_r + 1))
    lmin, lmax = cv2.erode(bgmap, kern), cv2.dilate(bgmap, kern)
    band = cv2.dilate(edge.astype(np.uint8), kern) > 0
    rng = lmax - lmin
    fgl = band & (rng > p.ldi_jump) & (bgmap - lmin > 0.5 * rng)
    far = band & (rng > p.ldi_jump) & ~fgl
    fillg = push_pull(bgmap, far) if far.any() else bgmap
    bdisp_g = np.clip(np.minimum(fillg, bgmap - p.ldi_margin), 0.0, 1.0)
    bmask = sdil | fgl
    bdisp = np.where(sdil, np.minimum(bg, front - p.subject_gap * 0.5), bdisp_g)
    bdisp = np.clip(np.where(bmask, np.minimum(bdisp, front - 0.01), front), 0.0, 1.0).astype(np.float32)
    bcol = push_pull(img_bgr.astype(np.float32), ~bmask)
    bcol = np.clip(bcol, 0, 255).astype(np.uint8)
    return Layer(col=img_bgr, disp=front, bmask=bmask, bdisp=bdisp, bcol=bcol, subj=subj, params=p)


def make_pad(layer: Layer, direction, side, p: Params):
    """Outpaint beyond the frame: the BACKGROUND PLATE (subject removed, from the LDI back layer) mirrored across the
    border, one frame long, on side +1 (the travel side) or -1. It shares the parent's camera, always sorts behind
    real content, and only shows where the two images do not cover the screen (the overlap at the seam)."""
    dx, dy = DIRS[direction]
    ax = 1 if dx else 0
    side = side * (dx or dy)                                 # +1 = the travel side (left/up are negative axes)
    plate_col = np.where(layer.bmask[..., None], layer.bcol, layer.col).astype(np.uint8)
    plate_disp = np.where(layer.bmask, layer.bdisp, layer.disp).astype(np.float32)
    col = cv2.flip(plate_col, ax)                            # flipCode 1 = horizontal mirror, 0 = vertical
    disp = cv2.flip(plate_disp, ax)
    col = cv2.GaussianBlur(col, (0, 0), 1.5)
    u0 = v0 = 0.0
    if ax == 1:
        u0 = float(layer.W) if side > 0 else -float(layer.W)
    else:
        v0 = float(layer.H) if side > 0 else -float(layer.H)
    nomask = np.zeros(disp.shape, bool)
    return Layer(col=col, disp=disp, bmask=nomask, bdisp=disp, bcol=col, params=p, u0=u0, v0=v0,
                 sortbias=10.0 * p.depth_ratio, f_ref=layer.f, cx0=layer.cx, cy0=layer.cy,
                 wrange=layer.wrange, parent=layer, flip_ax=ax)


# ----------------------------------------------------------------------------------------------- rasteriser
_BIG = np.float32(3.0e9)


def _fill_holes(R, iters, axis, iters2=30):
    """Fill never-written pixels with the FARTHEST written neighbour (rank = far->near order, so min rank), grown
    one px per pass: a disocclusion reveals the background, not the foreground. Grows along the travel axis first
    (clean streaks, no diagonal diamonds), then 3x3 for what is left."""
    kax = np.ones((1, 3), np.uint8) if axis == 0 else np.ones((3, 1), np.uint8)
    for stage, k3 in enumerate((kax, np.ones((3, 3), np.uint8))):
      for _ in range(iters if stage == 0 else iters2):
        hole = R == 0
        if not hole.any() or hole.all():
            break
        rb = np.where(hole, _BIG, R)
        m = cv2.erode(rb, k3)
        new = hole & (m < _BIG)
        if not new.any():
            break
        R[new] = m[new]
    return R


class Rasteriser:
    def __init__(self, out_w, out_h, params: Params, axis=0):
        self.W, self.H, self.p, self.axis = out_w, out_h, params, axis
        self.M = 24                                              # border margin so big blocks near the edge still land

    def _splat(self, items, C, small):
        """Z-buffer splat of `items`. Returns (rank image (H,W) float32, colours sorted far->near) or None."""
        W, H, M = self.W, self.H, self.M
        zs, us, vs, bs, cs = [], [], [], [], []
        for it in items:
            layer, off = it[0], it[1]
            bias = it[2] if len(it) > 2 else 0.0                 # ordering only: the forming image wins ties
            for which in ((0, 1) if layer.bmask.any() else (0,)):
                u, v, zc = layer.project(C, off, which)
                b = layer.block_sizes(u, v, zc, which)
                col = layer.col if which == 0 else layer.bcol
                m = zc > self.p.near_clip
                if which == 1:
                    m &= layer.bmask
                m &= (u > -M + 1) & (u < W + M - 1) & (v > -M + 1) & (v < H + M - 1)
                if not m.any():
                    continue
                zs.append(zc[m] - np.float32(bias)); us.append(u[m]); vs.append(v[m]); bs.append(b[m]); cs.append(col[m])
        if not zs:
            return None
        z = np.concatenate(zs); u = np.concatenate(us); v = np.concatenate(vs)
        b = np.concatenate(bs); c = np.concatenate(cs)
        order = np.argsort(-z, kind="stable")                    # far -> near
        ui = np.round(u[order]).astype(np.int32) + M
        vi = np.round(v[order]).astype(np.int32) + M
        b = b[order]; c = c[order]
        rank = np.arange(1, len(order) + 1, dtype=np.float32)
        R = np.zeros((H + 2 * M, W + 2 * M), np.float32)
        for bs_ in np.unique(b):
            m = b == bs_
            buf = np.zeros_like(R)
            buf[vi[m], ui[m]] = rank[m]                          # later (nearer) writes win
            if bs_ > 1:
                buf = cv2.dilate(buf, np.ones((int(bs_), int(bs_)), np.uint8))
            np.maximum(R, buf, out=R)
        R = R[M:M + H, M:M + W].copy()
        return _fill_holes(R, small, self.axis, 3 if small else 30), c

    def render(self, items, C):
        """items: [(Layer, offset(3,))]; C camera position (3,). Returns BGR uint8 (H,W,3).
        Real content is splatted first and only SMALL cracks are closed from its own neighbours; what is still empty
        (the overlap at the seam) is taken from the pads (mirrored plates, always behind), then grown/inpainted."""
        W, H = self.W, self.H
        real = [it for it in items if it[0].sortbias == 0]
        pads = [it for it in items if it[0].sortbias != 0]
        got = self._splat(real, C, self.p.crack_iters)
        img = np.zeros((H, W, 3), np.uint8)
        hole = np.ones((H, W), bool)
        if got is not None:
            R, c = got
            hole = R == 0
            img = c[np.maximum(R.astype(np.int64) - 1, 0)]
            img[hole] = 0
        if hole.any() and pads:
            gp = self._splat(pads, C, self.p.fill_iters)
            if gp is not None:
                Rp, cp = gp
                use = hole & (Rp > 0)
                img[use] = cp[Rp[use].astype(np.int64) - 1]
                hole &= ~use
        if hole.any() and not hole.all():
            img = cv2.inpaint(np.ascontiguousarray(img), hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        return img


# ----------------------------------------------------------------------------------------------- scene / camera
def travel_span(layer: Layer, direction, p: Params):
    """World distance between image k and k+1 along D: far content of a frame (Z = depth_ratio) leaves the view."""
    dx, dy = DIRS[direction]
    extent = layer.W if dx else layer.H
    return p.span_margin * extent * p.depth_ratio / layer.f


class Scene:
    """Images 0..n-1 laid one span apart along D; one camera. c in [0, n-1] is the camera position in image units."""

    def __init__(self, layers, direction, params: Params, out_size=None):
        self.layers, self.dir, self.p = layers, direction, params
        self.dvec = np.array([DIRS[direction][0], DIRS[direction][1], 0.0], np.float32)
        self.span = travel_span(layers[0], direction, params)
        H, W = layers[0].H, layers[0].W
        self.W, self.H = (out_size or (W, H))
        self.ras = Rasteriser(self.W, self.H, params, axis=0 if DIRS[direction][0] else 1)
        self.pads = {}                                            # (layer id, side) -> pad Layer, built lazily
        # per-depth motion: spring calibrated in span units (near content: overshoot = fraction of the frame width)
        self.f = layers[0].f
        fs = self.f * self.span                                   # px of NEAR content (Z=1) per span
        self.cfg = None
        if params.spring:
            self.cfg = _spring_cfg(params.ease_travel, params.settle, params.ring_period, params.ring_lead,
                                   params.overshoot * self.W / fs, params.tol_px / fs)
        self.t_rest = rest_time(params)
        self.ZERO = np.zeros(3, np.float32)
        self._rcache = {}

    def pad(self, k, side):
        key = (id(self.layers[k]), side)                          # looped last image shares image 0's layer + pads
        if key not in self.pads:
            self.pads[key] = make_pad(self.layers[k], self.dir, side, self.p)
        return self.pads[key]

    def _rnd(self, layer, role):
        """(front, back) random delay phases r in [0,1] per patch for a non-pad layer; independent draws per image
        (by first position in the scene) and per role (leaving / forming)."""
        key = (id(layer), role)
        if key not in self._rcache:
            ordn = next(i for i, l in enumerate(self.layers) if l is layer)
            seed = self.p.seed + 7919 * ordn + (17 if role == "out" else 91)
            cell = self.p.cell * layer.W / 432.0
            rng = np.random.default_rng(seed + 1)
            labf = patch_labels(layer.H, layer.W, cell, seed, layer.subj)
            labb = patch_labels(layer.H, layer.W, cell, seed, None)    # back layer = the background behind the subject
            tab = rng.random(int(labf.max()) + 2).astype(np.float32)
            self._rcache[key] = (tab[labf], tab[labb])
        return self._rcache[key]

    def phase(self, layer, which, mode):
        """Delay phase w in [0,1] of every point of a grid: depth mode = farness; random mode = patch randomness
        (leaving: delay lag*T*r; forming: lag*T*(1-w) with w = 1-r)."""
        if self.p.delay_mode != "random":
            return layer.wfar[which]
        base = layer.parent or layer
        rf, rb = self._rnd(base, mode)
        if layer.parent is not None:
            r = cv2.flip(rb, layer.flip_ax)
        else:
            r = rf if which == 0 else rb
        return r if mode == "out" else 1.0 - r

    # --- per-point motion: camera fixed at home, image points translated by span * dist along D
    def point_offset(self, mode, t):
        """callable(layer, which) -> (ox, oy, oz) per point. mode 'out': leaving image, dist = -q_out(t, w);
        'in': forming image, dist = 1 - q_in(t, w) (q_in may pass 1: the overshoot). oz = the old camera push."""
        p = self.p

        def f(layer, which):
            w = self.phase(layer, which, mode)
            if mode == "out":
                q = out_progress(t, w, p)
                dist = -q
            else:
                q = in_progress(t, w, p, self.cfg)
                dist = 1.0 - q
            oz = -p.push * np.sin(np.pi * np.clip(q, 0.0, 1.0))
            return ((self.dvec[0] * self.span * dist).astype(np.float32),
                    (self.dvec[1] * self.span * dist).astype(np.float32), oz.astype(np.float32))
        return f

    def home_of(self, fr):
        """Index of the image the frame shows at rest (exact home, no motion), or None while anything moves."""
        if fr.t <= 0:
            return fr.seg
        if fr.t >= self.t_rest:
            return fr.seg + 1
        return None

    def state_items(self, seg, t):
        """Render items of transition seg -> seg+1, t frames after its window start."""
        p = self.p
        if t <= 0:
            return [(self.layers[seg], self.ZERO)]
        if t >= self.t_rest:
            return [(self.layers[seg + 1], self.ZERO)]
        items = []
        if t < window(p):                                          # the leaving image is gone once its far edge is out
            off = self.point_offset("out", t)
            items += [(self.layers[seg], off), (self.pad(seg, +1), off)]
        off = self.point_offset("in", t)
        items.append((self.layers[seg + 1], off, p.zbias))
        q = in_progress(t, np.linspace(0.0, 1.0, 9), p, self.cfg)
        if q.min() < 1.0 - 1e-6:
            items.append((self.pad(seg + 1, -1), off, p.zbias))
        if q.max() > 1.0 + 1e-6:                                   # overshoot opens the other edge
            items.append((self.pad(seg + 1, +1), off, p.zbias))
        return items

    def blur_subs(self, t):
        """Sub-frames for the shutter: from the fastest screen speed (px/frame) over all depths and both images."""
        p = self.p
        ws = np.linspace(0.0, 1.0, 5)
        zw = 1.0 / (1.0 - ws * (1.0 - 1.0 / p.depth_ratio))        # Z at farness w
        if p.delay_mode == "random":
            zw = np.ones_like(ws)                                  # any depth can have any delay: assume near
        d = np.maximum(np.abs(out_progress(t + 0.5, ws, p) - out_progress(t - 0.5, ws, p)),
                       np.abs(in_progress(t + 0.5, ws, p, self.cfg) - in_progress(t - 0.5, ws, p, self.cfg)))
        px = float((d * self.span * self.f / zw).max())
        return int(np.clip(math.ceil(px * p.shutter / p.sub_px), 1, p.max_sub))

    def render_t(self, seg, t, nsub=1, shutter=0.0):
        """Render transition seg at time t (frames); nsub > 1 averages sub-frames over `shutter` frames, each with
        its own per-point curves (blur along every point's own velocity)."""
        ts = [t] if nsub <= 1 else [t + shutter * ((i + 0.5) / nsub - 0.5) for i in range(nsub)]
        acc = None
        for tt in ts:
            fr = self.ras.render(self.state_items(seg, tt), self.ZERO).astype(np.float32)
            acc = fr if acc is None else acc + fr
        return np.clip(acc / len(ts) + 0.5, 0, 255).astype(np.uint8)

    def items(self, c):
        ids = self.active(c)
        items = [(self.layers[i], self.offset(i)) for i in ids]
        if len(ids) == 2:                                         # mid-travel: the outgoing extends ahead, the
            items.append((self.pad(ids[0], +1), self.offset(ids[0])))   # incoming extends behind -> overlap at the seam
            items.append((self.pad(ids[1], -1), self.offset(ids[1])))
        return items

    def offset(self, k):
        return self.dvec * (self.span * k)

    def camera(self, c):
        k = math.floor(c + 1e-9)
        e = c - k
        z = self.p.push * math.sin(math.pi * e)                    # 0 at every home
        return self.dvec * (self.span * c) + np.array([0.0, 0.0, z], np.float32)

    def active(self, c):
        lo = math.floor(c + 1e-9)
        ids = {lo} if abs(c - round(c)) < 1e-9 else {lo, lo + 1}
        return sorted(i for i in ids if 0 <= i < len(self.layers))

    def render_c(self, c, nsub=1, shutter_dc=0.0):
        """Render the camera at position c; nsub > 1 averages sub-frames over [c - shutter_dc/2, c + shutter_dc/2]."""
        if nsub <= 1:
            cs = [c]
        else:
            cs = [c + shutter_dc * ((i + 0.5) / nsub - 0.5) for i in range(nsub)]
        acc = None
        for cc in cs:
            cc = min(max(cc, 0.0), len(self.layers) - 1.0)
            fr = self.ras.render(self.items(cc), self.camera(cc)).astype(np.float32)
            acc = fr if acc is None else acc + fr
        return np.clip(acc / len(cs) + 0.5, 0, 255).astype(np.uint8)


def timeline(params: Params, n_images=3):
    """[(c, kind)] frame list: hold, travel, hold, travel, hold. c = camera position in image units."""
    p = params
    out = [(0.0, "hold")] * p.ease_hold
    for k in range(n_images - 1):
        for i in range(1, p.ease_travel):
            out.append((k + float(smootherstep(i / p.ease_travel)), "travel"))
        out += [(k + 1.0, "hold")] * p.ease_hold
    return out


class Frame(NamedTuple):
    seg: int        # transition seg -> seg+1 whose clock t runs
    t: float        # frames since that transition's window start (0 = image seg at home, >= rest_time = seg+1 at home)
    kind: str       # hold | travel | settle


def frames(params: Params, n_images=3, loop=False):
    """Frame list for per-depth motion. Window = T*(1+lag) frames per transition, then the hold (the spring settles
    inside it). loop: n_images includes the repeated first image; the hold is split across the end (>= settle, so
    the spring is at rest on the last frame) and the start (the rest of ease_hold), so a repeat shows one hold."""
    p = params
    Tw = window(p)
    rest = rest_time(p)
    h_start = h_end = p.ease_hold
    if p.hold_start is not None:
        h_start = p.hold_start
    if p.hold_end is not None:
        h_end = p.hold_end
    if loop:
        h_end = max(p.settle if p.spring else 0, p.ease_hold // 2)
        h_start = max(p.ease_hold - h_end, 0)
    out = [Frame(0, 0.0, "hold")] * h_start
    for k in range(n_images - 1):
        for i in range(1, Tw):
            out.append(Frame(k, float(i), "travel"))
        hold = h_end if ((loop or p.hold_end is not None) and k == n_images - 2) else p.ease_hold
        for h in range(hold):
            t = float(Tw + h)
            out.append(Frame(k, t, "settle" if t < rest else "hold"))
    return out


def render_sequence(scene: Scene, params: Params, on_frame, loop=False):
    p = params
    cache = {}
    for i, fr in enumerate(frames(p, len(scene.layers), loop)):
        if fr.kind == "hold":
            key = scene.home_of(fr)
            if key not in cache:
                cache[key] = scene.render_t(fr.seg, fr.t)
            on_frame(i, cache[key], fr, fr.kind)
            continue
        on_frame(i, scene.render_t(fr.seg, fr.t, scene.blur_subs(fr.t), p.shutter), fr, fr.kind)


def _inv_ease(e):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if smootherstep(mid) < e:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------------------------------- io
def _ffmpeg_writer(path, w, h, fps):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "16", "-preset", "medium", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mesh_size(src_w, src_h, short=432):
    """Mesh/output size from the source: short side `short`, aspect kept, both even."""
    s = short / float(min(src_w, src_h))
    w = int(round(src_w * s / 2.0)) * 2
    h = int(round(src_h * s / 2.0)) * 2
    return w, h


def first_frame(video, out_png):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-frames:v", "1", out_png],
                   check=True)
    return cv2.imread(out_png)


def load_layer_inputs(frame_bgr, matte_png, short=432, dn_full=None, p=None):
    """-> (img_mesh, dn_mesh, matte_mesh). dn_full: normalised disparity at source size (else estimated)."""
    h0, w0 = frame_bgr.shape[:2]
    w, h = mesh_size(w0, h0, short)
    img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    if dn_full is None:
        dn_full = normalize_disparity(estimate_depth(frame_bgr))
    dn = cv2.resize(dn_full, (w, h), interpolation=cv2.INTER_AREA)
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from tools.fusion import pack_alpha                          # packed ingredients/*_matte.mkv when present, else the PNG
    m = pack_alpha.load_matte(matte_png, frames=[0])[0].astype(np.float32) / 255.0   # == cv2.imread(png, IMREAD_GRAYSCALE)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return img, dn, m


# ----------------------------------------------------------------------------------------------- CLI
SHOTS = ("s1a", "s2a", "s3a")
V3 = os.path.join(ROOT, "15_fusion", "v003")


def _paths(shot):
    vid = os.path.join(V3, "out", "motion-graphic_fusion_v003_P06_%s_1_00001_.mp4" % shot)
    alpha = os.path.join(V3, "local", "pieces", "P06", "seq",
                         "motion-graphic_fusion_v003_P06_alpha_%s_1_image_00001_.png" % shot)
    still = os.path.join(V3, "out", "motion-graphic_fusion_v003_P06_still_%s_1_00001_.png" % "ABC"[SHOTS.index(shot)])
    return vid, alpha, still


def version_dirs(vdir):
    """-> (out, review, local) folders of a PROJECTS version dir, created."""
    ds = [os.path.join(vdir, n) for n in ("out", "review", "local")]
    for d in ds:
        os.makedirs(d, exist_ok=True)
    return ds


def cmd_depth(a):
    _, review, out = version_dirs(a.out)
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    p = Params()
    for shot in SHOTS:
        vid, alpha, still = _paths(shot)
        fr = first_frame(vid, os.path.join(work, "frame_%s.png" % shot))
        st = cv2.imread(still)
        print("%s first frame %s vs still: PSNR %.1f dB" % (shot, fr.shape, psnr(fr, st)))
        t0 = time.time()
        raw = estimate_depth(fr)
        dn_full = normalize_disparity(raw)
        t1 = time.time()
        img, dn, m = load_layer_inputs(fr, alpha, a.short, dn_full=dn_full)
        layer = build_layer(img, dn, m, p)
        t2 = time.time()
        np.save(unique_path(os.path.join(out, "depth_%s_raw.npy" % shot)), raw)
        np.save(unique_path(os.path.join(out, "depth_%s.npy" % shot)), layer.disp)
        cv2.imwrite(unique_path(os.path.join(review, "depth_%s.png" % shot)), (layer.disp * 255).astype(np.uint8))
        print("%s depth %.1fs (DA-V2-Small CPU), clean+layers %.1fs, mesh %dx%d, subject px %d, back px %d"
              % (shot, t1 - t0, t2 - t1, layer.W, layer.H, int(layer.subj.sum()), int(layer.bmask.sum())))


def make_layers(out, short, p):
    """out = the local/ cache folder (depth_*_raw.npy are read from / scratch frames go to it)."""
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    layers = []
    for shot in SHOTS:
        vid, alpha, _ = _paths(shot)
        fr = first_frame(vid, os.path.join(work, "frame_%s.png" % shot))
        npy = os.path.join(out, "depth_%s_raw.npy" % shot)
        dn_full = normalize_disparity(np.load(npy)) if os.path.exists(npy) else None
        img, dn, m = load_layer_inputs(fr, alpha, short, dn_full=dn_full)
        layers.append(build_layer(img, dn, m, p))
    return layers


def _sheet(rows, path, tw=170, cols=10):
    """rows: {label: [(frame, text), ...]} -> one row per entry (wrapped every `cols` tiles)."""
    out = []
    wrapped = {}
    for label, tiles in rows.items():
        for j in range(0, len(tiles), cols):
            wrapped[label if j == 0 else "%s (cont)" % label] = tiles[j:j + cols]
    for label, tiles in wrapped.items():
        th = []
        for f, txt in tiles:
            t = cv2.resize(f, (tw, int(round(tw * f.shape[0] / f.shape[1]))), interpolation=cv2.INTER_AREA)
            cv2.putText(t, txt, (4, t.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            th.append(t)
        th += [np.zeros_like(th[0])] * (cols - len(th))
        row = np.hstack([np.pad(t, ((0, 0), (0, 3), (0, 0))) for t in th])
        cv2.putText(row, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.pad(row, ((0, 3), (0, 0), (0, 0))))
    cv2.imwrite(path, np.vstack(out))


def sheet_times(sc: Scene, p: Params, n=10):
    """n integer window times for one transition: evenly through the travel (tearing), the landing, the near-content
    overshoot peak + the undershoot after it, and the first fully-rest frame."""
    Tw = window(p)
    ts = np.arange(0, Tw + int(rest_time(p) - Tw) + 1)
    qn = in_progress(ts.astype(np.float64), 0.0, p, sc.cfg)
    pk = int(ts[Tw - 2 + int(np.argmax(qn[Tw - 2:]))]) if p.spring else Tw
    tr = pk + 1 + int(np.argmin(qn[pk + 1:])) if p.spring and pk + 1 < len(ts) else Tw
    picks = {int(round(x)) for x in np.linspace(4, Tw - 4, n - 4)} | {Tw, pk, tr, int(rest_time(p))}
    for x in np.linspace(5, rest_time(p), 40):
        if len(picks) >= n:
            break
        picks.add(int(round(x)))
    return sorted(picks)[:n]


def cmd_render(a):
    p = Params(depth_ratio=a.depth_ratio, subject_relief=a.relief, ease_travel=a.travel, lag=a.lag,
               overshoot=a.overshoot, settle=a.settle, loop=a.loop, spring=a.spring,
               delay_mode=a.delay_mode, cell=a.cell, seed=a.seed)
    if lag_frames(p) == 0 and not p.spring:
        p.zbias = 0.0                                          # old rigid move: no tie-break bias either
    t0 = time.time()
    outd, review, local = version_dirs(a.out)
    vname = os.path.basename(os.path.normpath(a.out)) + (("_" + a.tag) if a.tag else "")
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    layers = make_layers(a.depth_from or local, a.short, p)    # --depth-from: reuse another version's local/ depth
    n_pick = 10
    if a.first_only:                                           # only image 1 -> 2: hold, travel, settle, hold
        layers, p.loop, n_pick = layers[:2], False, 20
        p.hold_start, p.hold_end = 12, (p.settle if p.spring else 0) + 12
    if p.loop:
        layers = layers + [layers[0]]                          # 1 -> 2 -> 3 -> 1: the 4th image IS the first layer
    print("layers built %.1fs" % (time.time() - t0))
    sheet = {}
    sp = unique_path(os.path.join(review, "dolly_sheet%s.png" % (("_" + a.tag) if a.tag else "")))
    for d in a.dirs.split(","):
        sc = Scene(layers, d, p)
        mp4 = unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, vname, d)))
        proc = _ffmpeg_writer(mp4, sc.W, sc.H, p.fps)
        fl = frames(p, len(layers), p.loop)
        picks = sheet_times(sc, p, n_pick)
        want = {}                                                 # frame index -> (row label, tile text)
        for seg in range(len(layers) - 1):
            for t in picks:
                cand = [(abs(f.t - t), i) for i, f in enumerate(fl) if f.seg == seg and f.t > 0]
                want[min(cand)[1]] = ("%s  image %d -> %d" % (d, seg % 3 + 1, (seg + 1) % 3 + 1), "t=%d" % t)
        keep = {}
        t1 = time.time()

        def on_frame(i, fr, f, kind):
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
            if i in want:
                keep[i] = fr.copy()
            if i % 12 == 0:
                print("  %s frame %d/%d seg %d t=%.0f %s  %.0fs" % (d, i, len(fl), f.seg, f.t, kind, time.time() - t1),
                      flush=True)

        render_sequence(sc, p, on_frame, p.loop)
        proc.stdin.close()
        proc.wait()
        for i in sorted(want):
            sheet.setdefault(want[i][0], []).append((keep[i], want[i][1]))
        print("%s -> %s (%.0fs)" % (d, mp4, time.time() - t1), flush=True)
        _sheet(sheet, sp)                                         # rewritten after each direction (first one is judged first)
        print("sheet ->", sp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["depth", "render"])
    ap.add_argument("--out", default=os.path.join(ROOT, "17_depth-dolly", "v001"))
    ap.add_argument("--short", type=int, default=432)
    ap.add_argument("--dirs", default="up,down,left,right")
    ap.add_argument("--depth-ratio", type=float, default=Params.depth_ratio, help="far Z / near Z: bigger = more parallax")
    ap.add_argument("--relief", type=float, default=Params.subject_relief, help="subject depth relief 0..1")
    ap.add_argument("--travel", type=int, default=Params.ease_travel, help="frames per eased travel")
    ap.add_argument("--depth-from", default=None, help="local/ folder holding depth_*_raw.npy (e.g. v001/local)")
    ap.add_argument("--tag", default="", help="suffix for output names (sweep members)")
    ap.add_argument("--lag", type=float, default=Params.lag, help="depth delay, fraction of --travel (0 = rigid)")
    ap.add_argument("--overshoot", type=float, default=Params.overshoot,
                    help="landing overshoot of near content, fraction of the frame width")
    ap.add_argument("--settle", type=int, default=Params.settle, help="frames for the landing spring to come to rest")
    ap.add_argument("--spring", action=argparse.BooleanOptionalAction, default=True,
                    help="damped-spring landing (--no-spring = quintic ease on the forming image too; with --lag 0 "
                         "--no-loop it reproduces the old rigid move)")
    ap.add_argument("--delay-mode", choices=["depth", "random"], default="depth",
                    help="depth: far leaves late / forms early; random: per-patch random delays")
    ap.add_argument("--cell", type=float, default=Params.cell, help="random mode: patch size in px at 432 wide")
    ap.add_argument("--seed", type=int, default=0, help="random mode: patch layout + delay seed")
    ap.add_argument("--first-only", action=argparse.BooleanOptionalAction, default=False,
                    help="only the first transition (image 1 -> 2): 12-frame hold, travel, settle, 12-frame hold")
    ap.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="1 -> 2 -> 3 -> 1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"depth": cmd_depth, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
