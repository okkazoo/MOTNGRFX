#!/usr/bin/env python3
"""Camera-ORBIT transition between two depth images (CPU, numpy + OpenCV). Reuses tools/fusion/depth_dolly.py
(Params, make_layers, Rasteriser z-buffer splat, ffmpeg writer, sheet helper) WITHOUT editing it.

ONE rigid camera, every point fixed in the world (no per-point delays). Image 1 sits at its home pose (camera at the
origin, identity rotation; frame 0 == image 1). The camera orbits (yaw, horizontal arc, to its right) round the
vertical axis through P1 = the centre of image 1's depth volume on its optical axis, with a growing radius so it
clears the volume, and its look direction LEADS the arc (yaw = -lead * orbit angle, lead > 1) so image 1 swings round
in perspective and slides off the side of the frame. Image 2 is placed in the world at the END camera pose (its
points = T_end applied to its home camera-space points, uniformly scaled by s about the end camera centre so it fits
between the end camera and image 1). It enters turned `enter_angle` (45 deg) to the camera and lands flat: the last
`enter_angle` degrees are an orbit round P2 = the centre of image 2's volume on ITS home axis (relative angle camera
vs image 2: enter_angle -> 0). The two arcs are joined by a cubic Hermite connector (continuous position, heading and
tangent). One ease (quintic) or one spring (overshoot a few degrees along the arc, 1-2 decaying oscillations).

2D convention (x right, z forward, top-down): heading psi -> forward (sin psi, cos psi), right (cos psi, -sin psi).
Camera frame: X_cam = R^T (X_world - C); the existing splat is fed X_cam as if the camera sat at the origin.

CLI:
  python tools/fusion/depth_orbit.py render --out 17_depth-dolly/v005 --depth-from 17_depth-dolly/v001/local \
         [--depth-ratio 6] [--angle 100] [--lead 1.35] [--enter-angle 45] [--travel 36] [--spring|--no-spring] [--tag T]
  python tools/fusion/depth_orbit.py diagram --out 17_depth-dolly/v005 --depth-from ...   # review/orbit_path.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402

rad = math.radians


# ----------------------------------------------------------------------------------------------- params
@dataclass
class OrbitParams:
    depth_ratio: float = 6.0
    short: int = 432
    fps: int = 24
    hold1: int = 12                 # frames of image 1 before the move (its clip keeps playing: set_frames(i, ..))
    travel: int = 36                # frames of the eased move (t = 1..travel; t = travel is the landing)
    hold2: int = 24                 # frames on image 2 (the spring settles inside it)
    angle: float = 110.0            # orbit angle (deg) round P1 at which the arrival arc starts
    lead: float = 1.6               # peak lead: yaw = -(1 + (lead-1) sin^p(pi a/angle)) * orbit angle a (1 at the ends)
    lead_pow: float = 2.0
    radius_end: float = 7.0         # orbit radius at `angle` (starts at the pivot distance so frame 0 is home)
    enter_angle: float = 45.0       # image 2 enters turned this much to the camera, lands flat
    delta: float = 20.0             # deg of the first arc replaced by the Hermite connector
    pivot: str = "median"           # P1 depth: median of image 1's Z ("mid" = (min+max)/2)
    scale: float | None = None      # image 2 scale s about the end camera; None = largest s that keeps `gap`
    s_max: float = 0.9
    cam_margin: float = 0.05        # camera stays this far outside both volumes' bounding boxes
    gap: float = 0.4                # min world distance between the two point clouds
    spring: bool = False
    overshoot_deg: float = 5.0      # spring: peak overshoot along the arrival arc (deg)
    settle: int = 18                # spring: frames after the landing until rest (inside hold2)
    ring_period: float = 13.0       # frames per oscillation
    ring_decay: float = 0.17        # 1/frames
    ring_lead: int = 6              # the ring starts this many frames before the ease ends
    shutter: float = 0.55
    sub_px: float = 10.0
    max_sub: int = 8
    near_clip: float = 0.02         # world units (scene is scaled: image 2 is s * its home depth)
    bg_dark: float = 0.5
    # v006 pivot mode
    mode: str = "orbit"             # orbit (v005) | pivot (v006)
    out_angle: float = 60.0         # pivot: image 1 has turned this much when image 2 starts to enter
    in_angle: float = 60.0          # pivot: image 2 enters turned this much (relative yaw -in_angle -> 0)
    overlap: float | None = None    # pivot: image 2's share of the travel (None = in/(out+in)); overrides out_angle
    edge_soft: float = 1.0          # pivot: sigma (px) of image 2's coverage edge
    fade_in_deg: float = 14.0       # pivot: image 2 fades in over its first degrees (no pop); image 1 fades out 75 -> 90
    splat: str = "stretch"          # pivot: stretch (v007: block stretched to its continuous neighbours) | billboard (v008)
    max_chunks: int = 100           # billboard: HARD cap on cards per image volume (subject + background/LDI-back cards)
    size_range: tuple = (1.5, 4.0)  # billboard: card side = ref * U(lo, hi) (log-uniform), ref = sqrt(W*H/max_chunks)
    seed: int = 0                   # billboard: jitter / size seed
    card_grow: float = 0.5          # billboard: card grows this many home px per side (guaranteed overlap at the seams)
    card_max: float = 1400.0        # billboard: sanity cap on a card's screen side (px)
    blur: str = "auto"              # motion blur: steps (<= max_sub sharp sub-frames averaged, v008) | continuous (each card
    #                                 smeared along its own screen motion over the shutter, billboard only) | off;
    #                                 auto = continuous for billboard, steps for stretch
    subject_cards: int = 1          # billboard: 1 = the subject is ONE cut-out card (v012); 0 = jittered subject cards (v008)
    video: bool = False             # billboard: textures + runner matte come from the live s1a / s2a clips (v013), geometry from frame 0
    bg_chunks: int = 0              # billboard: cap on the BACKGROUND cards (0 = whatever --max-chunks leaves); size ref stays from max_chunks
    type: str = ""                  # pivot: 'P06:WORD ON IMAGE 1,WORD ON IMAGE 2' = the piece's words as flat type planes in the scene (v014)
    type_depth: float | None = None  # type plane depth: behind_subject = fraction of the way from the subject's Z to the background's (default 0.5);
    #                                  front = this x the subject's median Z (default 0.8, < 1: in front of the runner)
    type_layer: str = "behind_subject"  # behind_subject = background cards -> type -> runner card (his matte; compose's `behind` recipe) | front
    type_style: str = "organic"     # organic (v015: carried with the turn, forms after the landing, beside the runner) | v014 (q-retimed, centred)
    type_scale: float = 0.75        # organic: word size relative to v014's (full frame width layout)
    # v021-v023 volume modes (mode = push | pullback | fly): ONE rigid camera path through the card volume, quintic q(t)
    vol_dz: float = 5.0             # camera travel along the optical axis (world units; image 1's volume spans z ~ 1..6)
    hole0: float = 0.20             # push / fly: image 1's cards start to dissolve from the centre outward at this q ...
    hole1: float = 0.85             # ... and are all gone at this q
    hole_soft: float = 0.16
    hole_max: float = 1.25
    near_fade: float = 0.55         # cards fade out over camera distances zc in (near_clip, near_fade)
    zoom_fade: tuple = (2.5, 4.0)   # ... and when the camera has closed in so far that they zoom to ratio 2.5 .. 4 x their home size
    roll_deg: float = 32.0          # fly: peak bank (the envelope puts the actual peak at ~27 deg)
    fly_amp: tuple = (0.9, 0.4)     # fly: lateral / vertical excursion (world units)
    fly_yaw: float = 14.0           # fly: peak heading change (deg)
    fly_pitch: float = 7.0          # fly: peak pitch (deg)
    pull_rise: float = 0.3          # pullback: the camera rises this far (world units) mid-way and comes back
    vol_bg: float = 0.9             # volume modes: brightness of the diffused fill round image 2's shrunk window (pivot modes use bg_dark 0.5)
    # shot-cut styles (tools/fusion/depth_smooth.py): chunks = the card scenes above | smooth = depth-parallax layers (a real camera move, no mosaic) | slide = lateral parallax slide
    style: str = "chunks"
    dir: str = "right"              # slide: the camera travels this way (the picture slides the other way)
    aim: float = 1.0                # smooth push / fly: how far the camera heads for the subject's centre (0 = straight down the axis, 1 = into the subject)


def psi_R(psi):
    """3x3 world rotation of a yaw `psi` (camera basis as columns: right, down, forward)."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)


def fwd(psi):
    return np.array([math.sin(psi), math.cos(psi)])


def right(psi):
    return np.array([math.cos(psi), -math.sin(psi)])


# ----------------------------------------------------------------------------------------------- path
class OrbitPath:
    """pose(tau) -> (x, z, psi), tau = distance along the path (world units), tau in [0, L]; tau > L continues the
    arrival arc past the landing (spring overshoot)."""

    def __init__(self, zp, op: OrbitParams, d2):
        self.zp, self.op, self.d2 = zp, op, d2
        a_c = rad(op.angle)
        self.a_c = a_c
        self.a1 = rad(op.angle - op.delta)
        self.lead = op.lead
        self.rm = op.radius_end
        # arrival arc B: beta (relative angle of the camera to image 2's axis) from beta0 to 0
        self.beta0 = rad(op.enter_angle)
        pB0, psiB0 = self.A(a_c)
        self.psi_B0 = psiB0
        self.psi_e = psiB0 - self.beta0
        self.P2 = pB0 + d2 * fwd(psiB0)
        self.C_e = self.P2 - d2 * fwd(self.psi_e)
        # A table (position arc length)
        al = np.linspace(0.0, self.a1, 800)
        pts = np.array([self.A(x)[0] for x in al])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        self.tabA = (np.concatenate([[0.0], np.cumsum(seg)]), al)
        self.LA = float(self.tabA[0][-1])
        # connector (Hermite): A(a1) -> B(beta0)
        pA1, psiA1 = self.A(self.a1)
        eps = 1e-4
        tA = (self.A(self.a1 + eps)[0] - self.A(self.a1 - eps)[0]) / (2 * eps)
        sp = float(np.hypot(*tA))
        self.tA = tA / sp
        self.hA = (self.A(self.a1 + eps)[1] - self.A(self.a1 - eps)[1]) / (2 * eps) / sp   # d psi / d tau at the end of A
        self.pA1, self.psiA1 = pA1, psiA1
        self.pB0 = pB0
        self.tB = right(psiB0)                                        # motion direction at the start of B (unit)
        self.hB = -1.0 / d2                                           # beta falls at 1/d2 per unit length
        self.Lc = 1.1 * float(np.hypot(*(pB0 - pA1)))
        self.LB = d2 * self.beta0
        self.L = self.LA + self.Lc + self.LB

    def r_of(self, al):
        u = min(max(al / self.a_c, 0.0), 1.0)
        return self.zp + (self.rm - self.zp) * u * u * (3.0 - 2.0 * u)

    def A(self, al):
        r = self.r_of(al)
        u = min(max(al / self.a_c, 0.0), 1.0)
        lead = 1.0 + (self.lead - 1.0) * math.sin(math.pi * u) ** self.op.lead_pow    # peaks mid-arc, 1 at the end
        return np.array([r * math.sin(al), self.zp - r * math.cos(al)]), -lead * al

    def B(self, beta):
        psi = self.psi_e + beta
        return self.P2 - self.d2 * fwd(psi), psi

    def pose(self, tau):
        if tau <= 0.0:
            return 0.0, 0.0, 0.0
        if tau <= self.LA:
            al = float(np.interp(tau, self.tabA[0], self.tabA[1]))
            p, psi = self.A(al)
        elif tau <= self.LA + self.Lc:
            u = (tau - self.LA) / self.Lc
            h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
            h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
            p = h00 * self.pA1 + h10 * self.Lc * self.tA + h01 * self.pB0 + h11 * self.Lc * self.tB
            psi = h00 * self.psiA1 + h10 * self.Lc * self.hA + h01 * self.psi_B0 + h11 * self.Lc * self.hB
        else:
            beta = self.beta0 - (tau - self.LA - self.Lc) / self.d2
            p, psi = self.B(beta)
        return float(p[0]), float(p[1]), float(psi)

    def rel_angle(self, tau):
        """Camera heading relative to image 2's normal, degrees (0 = flat on image 2)."""
        return math.degrees(self.pose(tau)[2] - self.psi_e)

    def end_pose(self):
        return float(self.C_e[0]), float(self.C_e[1]), float(self.psi_e)


# ----------------------------------------------------------------------------------------------- geometry
def _cloud(layer, stride=6):
    X, Y, Z, _, _ = layer.grids[0]
    return np.stack([X[::stride, ::stride].ravel(), Y[::stride, ::stride].ravel(), Z[::stride, ::stride].ravel()], 1)


def place_image2(cloud2, psi_e, C_e, s):
    """World points of image 2: T_end applied to s * home camera-space points (y is untouched by a yaw)."""
    R = psi_R(psi_e)
    c = s * cloud2
    w = c @ R.T
    return w + np.array([C_e[0], 0.0, C_e[1]])


@dataclass
class Geometry:
    path: OrbitPath
    s: float
    zp: float
    zc2: float
    mind: float
    c1: np.ndarray
    c2w: np.ndarray
    aabb1: tuple
    aabb2: tuple


def bbox(pts):
    return pts.min(0), pts.max(0)


def make_geometry(layers, op: OrbitParams) -> Geometry:
    from scipy.spatial import cKDTree
    c1, c2 = _cloud(layers[0]), _cloud(layers[1])
    z1 = c1[:, 2]
    zp = float(np.median(z1)) if op.pivot == "median" else float(0.5 * (z1.min() + z1.max()))
    zc2 = float(np.median(c2[:, 2]))
    # the camera at the entry angle sits d2*(1-cos(enter)) in front of its landing pose: keep that short of the
    # nearest point of image 2 (else it starts inside image 2's bounding box), so the pivot may be nearer than the median
    zc2 = min(zc2, 0.75 * float(c2[:, 2].min()) / max(1.0 - math.cos(rad(op.enter_angle)), 1e-6))
    tree = cKDTree(c1)

    box1 = bbox(c1)

    def build(s):
        path = OrbitPath(zp, op, s * zc2)
        w2 = place_image2(c2, path.psi_e, path.C_e, s)
        return path, w2, float(tree.query(w2)[0].min())

    def cam_clear(path, w2, s):
        box2 = bbox(w2)
        for tau in np.linspace(0.0, path.L * 1.03, 400):
            x, z, _ = path.pose(tau)
            pos = np.array([x, 0.0, z])
            if inside_aabb(pos, box1, op.cam_margin) or inside_aabb(pos, box2, min(op.cam_margin, 0.15 * s)):
                return False
        return True

    if op.scale is not None:
        s = op.scale
    else:
        s = op.s_max
        while s > 0.05:
            path, w2, mind = build(s)
            if mind >= op.gap and cam_clear(path, w2, s):
                break
            s *= 0.97
    path, w2, mind = build(s)
    return Geometry(path, s, zp, zc2, mind, c1, w2, bbox(c1), bbox(w2))


def inside_aabb(pos3, box, margin=0.0):
    lo, hi = box
    return bool(np.all(pos3 >= lo - margin) and np.all(pos3 <= hi + margin))


# ----------------------------------------------------------------------------------------------- timing
def spring_q(t, op: OrbitParams, L, d2):
    """Progress q(t) in path length / L. Quintic ease to 1 at t = travel plus a damped ring (peak overshoot =
    overshoot_deg along the arrival arc) that starts `ring_lead` frames before the ease ends; q = 1 after settle."""
    T = op.travel
    e = float(dd.smootherstep(t / T))
    if not op.spring:
        return e
    t0 = T - op.ring_lead
    t_end = T + op.settle

    def ring(tt):
        if tt <= t0:
            return 0.0
        if tt >= t_end:
            return 0.0
        k, P = op.ring_decay, op.ring_period
        return math.exp(-k * (tt - t0)) * math.sin(2 * math.pi * (tt - t0) / P)

    target = rad(op.overshoot_deg) * d2 / L
    ts = np.linspace(t0, t_end, 400)
    base = np.array([float(dd.smootherstep(x / T)) for x in ts])
    rg = np.array([ring(x) for x in ts])
    lo, hi = 0.0, 1.0
    for _ in range(40):                                           # amplitude so that max(q) - 1 == target
        mid = 0.5 * (lo + hi)
        if (base + mid * rg).max() - 1.0 < target:
            lo = mid
        else:
            hi = mid
    return e + 0.5 * (lo + hi) * ring(t)


def frame_list(op: OrbitParams):
    """[(t, kind)]: hold1 x (0, hold), travel x (1..travel), hold2 x (travel + h, settle|hold)."""
    out = [(0.0, "hold")] * op.hold1
    out += [(float(i), "travel") for i in range(1, op.travel + 1)]
    rest = op.travel + (op.settle if op.spring else 0)
    for h in range(1, op.hold2 + 1):
        t = float(op.travel + h)
        out.append((t, "settle" if (t < rest) else "hold"))
    return out


def q_at(t, op, L, d2):
    if t <= 0:
        return 0.0
    if op.spring and t >= op.travel + op.settle:
        return 1.0
    if t >= op.travel and not op.spring:
        return 1.0
    return spring_q(t, op, L, d2)


# ----------------------------------------------------------------------------------------------- render
class Cam:
    def __init__(self, x, z, psi):
        self.pos = np.array([x, 0.0, z], np.float64)
        self.psi = psi
        self.R = psi_R(psi)


class RigidLayer:
    """A depth_dolly Layer placed rigidly in the world: world = t + R(psi_k) (s * X_home). `project` takes a Cam."""

    def __init__(self, layer, psi=0.0, t=(0.0, 0.0, 0.0), s=1.0):
        self.L = layer
        self.s = s
        self.Rk = psi_R(psi)
        self.t = np.asarray(t, np.float64)
        self.sortbias = 0.0

    def __getattr__(self, name):
        return getattr(self.L, name)

    def project(self, cam, off, which=0):
        X, Y, Z, _, _ = self.L.grids[which]
        M = (cam.R.T @ self.Rk * self.s).astype(np.float32)
        b = (cam.R.T @ (self.t - cam.pos)).astype(np.float32)
        xc = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z + b[0]
        yc = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z + b[1]
        zc = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z + b[2]
        zs = np.where(zc > self.nc, zc, 1.0)
        return (self.L.f * xc / zs + self.L.cx).astype(np.float32), \
            (self.L.f * yc / zs + self.L.cy).astype(np.float32), zc.astype(np.float32)

    def block_sizes(self, u, v, zc, which):
        """Footprint of one source pixel (s * Z / zc) or the stretch to its CONTINUOUS neighbours (never bridges a
        real depth edge, so side-on meshes stretch and tear at the depth jumps)."""
        _, _, Z, cx_, cy_ = self.L.grids[which]
        fp = self.s * Z / np.where(zc > self.nc, zc, 1.0)
        sx = np.zeros_like(u)
        dx = np.abs(np.diff(u, axis=1)) * cx_ + np.abs(np.diff(v, axis=1)) * cx_
        sx[:, :-1] = np.maximum(sx[:, :-1], dx)
        sx[:, 1:] = np.maximum(sx[:, 1:], dx)
        dy = (np.abs(np.diff(v, axis=0)) + np.abs(np.diff(u, axis=0))) * cy_
        sx[:-1] = np.maximum(sx[:-1], dy)
        sx[1:] = np.maximum(sx[1:], dy)
        sz = np.maximum(fp, sx)
        return np.clip(np.ceil(sz - 0.002), 1, self.L.params.max_block).astype(np.int32)

    nc = 0.02


class OrbitRasteriser(dd.Rasteriser):
    def __init__(self, w, h, params, bg_dark=0.5):
        super().__init__(w, h, params, axis=0)
        self.bg_dark = bg_dark
        self.plate = None                                          # (H,W,3) float32 set per frame by the scene

    def render(self, items, cam):
        """Z-buffer splat of the rigid layers; holes (what the turned volumes do not cover) = a blurred, darkened
        version of the nearest rendered content, dark neutral where nothing is near."""
        got = self._splat(items, cam, self.p.crack_iters)
        H, W = self.H, self.W
        img = np.zeros((H, W, 3), np.uint8)
        hole = np.ones((H, W), bool)
        if got is not None:
            R, c = got
            hole = R == 0
            img = c[np.maximum(R.astype(np.int64) - 1, 0)]
            img[hole] = 0
        if hole.any():
            valid = (~hole).astype(np.float32)
            bg = np.zeros((H, W, 3), np.float32)
            have = np.zeros((H, W), bool)
            f32 = img.astype(np.float32)
            for sig in (6.0, 18.0, 50.0, 140.0):
                num = cv2.GaussianBlur(f32 * valid[..., None], (0, 0), sig)
                den = cv2.GaussianBlur(valid, (0, 0), sig)
                ok = (den > 0.02) & ~have
                bg[ok] = num[ok] / den[ok][:, None]
                have |= ok
            bg *= self.bg_dark
            if self.plate is not None:                              # far from any content: the nearest plate, blurred + dark
                bg[~have] = self.plate[~have]
            else:
                bg[~have] = (12.0, 11.0, 10.0)
            bg = cv2.GaussianBlur(bg, (0, 0), 9.0)                  # no contour bands between the blur levels
            img[hole] = np.clip(bg[hole] + 0.5, 0, 255).astype(np.uint8)
        return img


class OrbitScene:
    def __init__(self, layers, geo: Geometry, op: OrbitParams):
        self.op, self.geo, self.path = op, geo, geo.path
        import dataclasses
        p = dataclasses.replace(layers[0].params, near_clip=op.near_clip)
        self.params = p
        self.W, self.H = layers[0].W, layers[0].H
        self.f = layers[0].f
        self.ras = OrbitRasteriser(self.W, self.H, p, op.bg_dark)
        ex, ez, epsi = geo.path.end_pose()
        self.rl1 = RigidLayer(layers[0])
        self.rl2 = RigidLayer(layers[1], psi=epsi, t=(ex, 0.0, ez), s=geo.s)
        for r in (self.rl1, self.rl2):
            r.nc = op.near_clip
        self.L = geo.path.L
        self.d2 = geo.path.d2
        self.plates = []
        for layer in layers[:2]:                                   # blurred, darkened home views = the fill far from content
            pl = cv2.GaussianBlur(layer.col, (0, 0), 22.0).astype(np.float32) * op.bg_dark * 0.8
            self.plates.append(pl)

    def q(self, t):
        return q_at(t, self.op, self.L, self.d2)

    def cam_at_q(self, q):
        return Cam(*self.path.pose(q * self.L))

    def render_q(self, q, both=True):
        cam = self.cam_at_q(q)
        w = float(dd.smootherstep((q - 0.35) / 0.4))              # plate cross-fade image 1 -> 2 mid-move
        self.ras.plate = (1.0 - w) * self.plates[0] + w * self.plates[1]
        if q <= 0.0:
            items = [(self.rl1, None)]
        elif q == 1.0:
            items = [(self.rl2, None)]
        else:
            items = [(self.rl1, None), (self.rl2, None)]
        return self.ras.render(items, cam)

    def render_t(self, t, nsub=1):
        op = self.op
        if nsub <= 1:
            return self.render_q(self.q(t))
        acc = None
        for i in range(nsub):
            tt = t + op.shutter * ((i + 0.5) / nsub - 0.5)
            fr = self.render_q(self.q(tt)).astype(np.float32)
            acc = fr if acc is None else acc + fr
        return np.clip(acc / nsub + 0.5, 0, 255).astype(np.uint8)

    def nsub_at(self, t):
        op = self.op
        a = self.path.pose(self.q(t - 0.5) * self.L)
        b = self.path.pose(self.q(t + 0.5) * self.L)
        move = math.hypot(b[0] - a[0], b[1] - a[1])
        px = self.f * (abs(b[2] - a[2]) + move / 3.0)
        return int(np.clip(math.ceil(px * op.shutter / op.sub_px), 1, op.max_sub))

    def home_q(self, t):
        q = self.q(t)
        return q

    def poses(self, op=None):
        """Camera pose (x, z, psi_deg) and relative angle to image 2 for every frame of the timeline."""
        out = []
        for t, kind in frame_list(self.op):
            q = self.q(t)
            x, z, psi = self.path.pose(q * self.L)
            out.append((t, kind, x, z, math.degrees(psi), math.degrees(psi - self.path.psi_e)))
        return out


# ----------------------------------------------------------------------------------------------- pivot mode (v006)
def fill_bg(img, valid, plate, dark):
    """Background for pixels without content: blurred diffusion of the nearby content (weights `valid`, float 0..1),
    then the blurred dark plate where nothing is near. Returns (H,W,3) float32."""
    H, W = valid.shape
    f32 = img.astype(np.float32)
    bg = np.zeros((H, W, 3), np.float32)
    have = np.zeros((H, W), bool)
    for sig in (6.0, 18.0, 50.0, 140.0):
        num = cv2.GaussianBlur(f32 * valid[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(valid, (0, 0), sig)
        ok = (den > 0.02) & ~have
        bg[ok] = num[ok] / den[ok][:, None]
        have |= ok
    bg *= dark
    bg[~have] = plate[~have] if plate is not None else (12.0, 11.0, 10.0)
    return cv2.GaussianBlur(bg, (0, 0), 9.0)


def cell_points(mask, step, rng):
    """One random True pixel of `mask` per step x step cell (cells with none give nothing). Returns (ys, xs)."""
    H, W = mask.shape
    ny, nx = -(-H // step), -(-W // step)
    sc = np.full((ny * step, nx * step), -1.0, np.float32)
    sc[:H, :W] = np.where(mask, rng.random((H, W), dtype=np.float32), -1.0)
    cell = sc.reshape(ny, step, nx, step).transpose(0, 2, 1, 3).reshape(ny, nx, step * step)
    k = cell.argmax(2)
    cyi, cxi = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ok = cell.max(2) >= 0.0
    return (cyi * step + k // step)[ok], (cxi * step + k % step)[ok]


def _covered(shape, cx, cy, S):
    """Pixels fully inside at least one card (square side S centred on pixel (cx, cy))."""
    H, W = shape
    cov = np.zeros((H, W), bool)
    for x, y, sz in zip(cx, cy, S):
        h = sz / 2.0
        cov[max(int(math.ceil(y - h)), 0):int(math.floor(y + h)) + 1, max(int(math.ceil(x - h)), 0):int(math.floor(x + h)) + 1] = True
    return cov


def enlarge_to_cover(mask, cx, cy, S):
    """Grow the cards (never add one) until every True pixel of `mask` lies inside some card: the card that needs the
    least growth to reach the first uncovered pixel gets just big enough for it. S is modified in place."""
    for _ in range(4000):
        unc = mask & ~_covered(mask.shape, cx, cy, S)
        if not unc.any() or len(S) == 0:
            break
        ys, xs = np.nonzero(unc)
        y, x = ys[0], xs[0]
        d = np.maximum(np.abs(cx - x), np.abs(cy - y))
        k = int(np.argmin(2.0 * d - S))
        S[k] = 2.0 * d[k] + 1.0
    return S


def subject_split(img, a, grow=3):
    """(fg, bg) float32 BGR of one frame split by its soft matte `a` (float 0..1): `bg` = the frame with the subject REMOVED and filled from the pixels around it
    (push-pull of the current frame, hole = the matte grown by `grow` px, soft edge) - what the background cards show behind the subject, so a parallax shift
    never reveals a stale or foreign colour (the cyan / pale silhouette beside the subject in 22_batch-24gb v010 r02 was the frame-0 LDI back colour
    `Layer.bcol` painted where the subject stands); `fg` = the subject colour with the background mix removed at its soft edge, (img - (1-a) bg) / a, so the
    one subject card carries no halo. a.max() < 0.02: no subject, bg = img, fg = img."""
    img = np.asarray(img, np.float32)
    a = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    if float(a.max()) < 0.02:
        return img, img
    hole = (a > 0.03) if int(grow) <= 0 else cv2.dilate((a > 0.03).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(grow)) > 0
    fill = dd.push_pull(img, ~hole) if (~hole).any() else img
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.2)[..., None]
    bg = img * (1.0 - soft) + fill * soft
    a3 = a[..., None]
    fg = np.where(a3 > 0.1, np.clip((img - (1.0 - a3) * bg) / np.maximum(a3, 0.1), 0.0, 255.0), img)
    return fg.astype(np.float32), bg.astype(np.float32)


def make_cards(layer, op, seed, subj_union=None):
    """At most op.max_chunks flat textured cards for one image volume (hard cap). Two kinds: SUBJECT cards (one jittered
    point per cell on the matte, texture = the image, alpha = the matte, drawn in front) and BACKGROUND cards (a jittered
    grid over the whole frame, texture = the image with the subject replaced by the rebuilt LDI back layer, so what is
    behind the runner is real background). Each card is one flat plate at ONE depth = the median depth of the pixels it
    covers (the subject's front depth for subject cards, the background / LDI-back depth for background cards).
    Sizes are random (ref * U(lo, hi), log-uniform) and the smallest cards are enlarged until every pixel is covered."""
    H, W = layer.H, layer.W
    rng = np.random.default_rng(seed)
    N = int(op.max_chunks)
    ref = math.sqrt(W * H / N)
    lo, hi = op.size_range
    subj = layer.subj if layer.subj is not None else np.zeros((H, W), bool)
    Zf = layer.grids[0][2]
    Zw = dd.disp_to_z(np.where(subj, layer.bdisp, layer.disp), layer.params).astype(np.float32)
    ss = max(int(round(ref)), 1)
    sy = sx = np.zeros(0, np.int64)
    one_subj = bool(subj.any()) and int(getattr(op, "subject_cards", 1)) == 1
    if one_subj:                                                  # ONE card: bbox of the matte, alpha = the matte
        yy, xx = np.nonzero(subj if subj_union is None else (subj | subj_union))   # video: the card spans every frame's matte
        x0_, x1_, y0_, y1_ = xx.min(), xx.max(), yy.min(), yy.max()
        sx = np.array([(x0_ + x1_) / 2.0]); sy = np.array([(y0_ + y1_) / 2.0])
        one_S = np.array([max(x1_ - x0_, y1_ - y0_) + 2.0])
    elif subj.any():
        while True:
            sy, sx = cell_points(subj, ss, rng)
            if len(sy) <= max(N // 3, 1):
                break
            ss += 1
    ns = len(sy)
    sw = max(int(round(ref)), 1)
    nb_max = N - ns if not getattr(op, "bg_chunks", 0) else min(N - ns, int(op.bg_chunks))   # v014: --bg-chunks caps the background grid
    while (-(-W // sw)) * (-(-H // sw)) > nb_max:
        sw += 1
    wy, wx = cell_points(np.ones((H, W), bool), sw, rng)
    assert ns + len(wy) <= N

    def sizes(n):
        return ref * np.exp(rng.uniform(math.log(lo), math.log(hi), n))

    sc = {"x": sx.astype(np.float64), "y": sy.astype(np.float64), "S": one_S if one_subj else sizes(ns)}
    wc = {"x": wx.astype(np.float64), "y": wy.astype(np.float64), "S": sizes(len(wy))}
    if not one_subj:
        enlarge_to_cover(subj, sc["x"], sc["y"], sc["S"])
    enlarge_to_cover(np.ones((H, W), bool), wc["x"], wc["y"], wc["S"])

    def depth(c, Z, sel):
        out = np.zeros(len(c["S"]), np.float64)
        for i, (x, y, sz) in enumerate(zip(c["x"], c["y"], c["S"])):
            h = sz / 2.0
            y0, y1 = max(int(math.ceil(y - h)), 0), int(math.floor(y + h)) + 1
            x0, x1 = max(int(math.ceil(x - h)), 0), int(math.floor(x + h)) + 1
            z = Z[y0:y1, x0:x1]
            m = sel[y0:y1, x0:x1]
            out[i] = float(np.median(z[m])) if m.any() else float(Z[int(y), int(x)])
        return out

    sc["Z"] = depth(sc, Zf, subj)
    wc["Z"] = depth(wc, Zw, np.ones((H, W), bool))
    x = np.concatenate([sc["x"], wc["x"]]); y = np.concatenate([sc["y"], wc["y"]])
    Sz = np.concatenate([sc["S"], wc["S"]]); Z = np.concatenate([sc["Z"], wc["Z"]])
    tid = np.concatenate([np.zeros(ns, np.int64), np.ones(len(wc["S"]), np.int64)])
    tex = np.zeros((2, H, W, 4), np.float32)
    fg0, bg0 = subject_split(layer.col, subj.astype(np.float32), grow=0)      # a still has only the hard matte: no grown hole, so the home frame stays pixel exact
    tex[0, ..., :3] = fg0
    tex[0, ..., 3] = subj
    tex[1, ..., :3] = bg0
    tex[1, ..., 3] = 1.0
    return {"x": x, "y": y, "S": Sz, "Z": Z, "tid": tid, "tex": tex.reshape(-1, 4), "n_subject": ns, "n_back": len(wc["S"]),
            "X": (x - layer.cx) / layer.f * Z, "Y": (y - layer.cy) / layer.f * Z,
            "bias": np.where(tid == 0, 1.0e3, 0.0), "ref": ref}


def card_splat(u, v, side, ratio, cx, cy, tid, rank, tex, W, H, gain=None):
    """Painter's-by-depth splat of screen-aligned textured cards. Card k: centre (u,v) px, side `side` px, sampled from
    its source (texture `tid`, flat (n,4) BGR+alpha) at home-pixel coordinate (cx,cy) + (pixel - (u,v)) / ratio, bilinear.
    Fragment alpha = 1-px edge coverage * texture alpha; fragments are composited nearest-first per pixel with the
    'over' operator (rank: 0 = nearest). Returns (rgb float32 (H,W,3), alpha float32 (H,W))."""
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    n = len(u)
    if n == 0:
        return rgb, alpha
    half = 0.5 * side
    xa, xb, ya, yb = u - half, u + half, v - half, v + half
    px0 = np.maximum(np.floor(xa + 0.5).astype(np.int64), 0)
    px1 = np.minimum(np.floor(xb + 0.5).astype(np.int64), W - 1)
    py0 = np.maximum(np.floor(ya + 0.5).astype(np.int64), 0)
    py1 = np.minimum(np.floor(yb + 0.5).astype(np.int64), H - 1)
    nx, ny = px1 - px0 + 1, py1 - py0 + 1
    k = np.nonzero((nx > 0) & (ny > 0))[0]
    if not k.size:
        return rgb, alpha
    cnt = nx[k] * ny[k]
    tot = int(cnt.sum())
    cid = np.repeat(k, cnt)
    loc = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    lx = loc % nx[cid]
    px = px0[cid] + lx
    py = py0[cid] + (loc // nx[cid])
    del loc, lx
    ox = np.clip(np.minimum(xb[cid], px + 0.5) - np.maximum(xa[cid], px - 0.5), 0.0, 1.0)
    oy = np.clip(np.minimum(yb[cid], py + 0.5) - np.maximum(ya[cid], py - 0.5), 0.0, 1.0)
    tx = cx[cid] + (px - u[cid]) / ratio[cid]
    ty = cy[cid] + (py - v[cid]) / ratio[cid]
    inside = (tx >= -0.5) & (tx <= W - 0.5) & (ty >= -0.5) & (ty <= H - 0.5)   # no texture beyond the source frame
    tx = np.clip(tx, 0.0, W - 1.0)
    ty = np.clip(ty, 0.0, H - 1.0)
    x0 = np.minimum(np.floor(tx).astype(np.int64), W - 2) if W > 1 else np.zeros_like(px)
    y0 = np.minimum(np.floor(ty).astype(np.int64), H - 2) if H > 1 else np.zeros_like(py)
    fx = (tx - x0).astype(np.float32)[:, None]
    fy = (ty - y0).astype(np.float32)[:, None]
    i00 = tid[cid] * (H * W) + y0 * W + x0
    top = tex[i00] * (1.0 - fx) + tex[i00 + 1] * fx
    bot = tex[i00 + W] * (1.0 - fx) + tex[i00 + W + 1] * fx
    val = top * (1.0 - fy) + bot * fy
    del top, bot, i00, tx, ty, x0, y0, fx, fy
    a = ox * oy * val[:, 3] * inside
    if gain is not None:                                               # v021-v023: per-card opacity (near-plane fade, dissolve)
        a = a * gain[cid]
    g = a > (1.0 / 512.0)
    pix = (py * W + px)[g]
    col = val[g, :3]
    a = a[g]
    r = rank[cid[g]]
    del cid, px, py, ox, oy, val
    f = np.argsort(pix * n + r, kind="stable")                         # by pixel, nearest card first
    pix, col, a = pix[f], col[f], a[f]
    la = np.log1p(-np.minimum(a, 1.0 - 1e-6).astype(np.float64))
    cs = np.cumsum(la)
    before = cs - la
    first = np.ones(len(pix), bool)
    first[1:] = pix[1:] != pix[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
    w = a * np.exp(before - before[start])                             # transmittance of the nearer cards
    totw = np.bincount(pix, weights=w, minlength=H * W)
    ok = totw > 1e-6
    flat = rgb.reshape(-1, 3)
    for ch in range(3):
        num = np.bincount(pix, weights=w * col[:, ch], minlength=H * W)
        flat[ok, ch] = (num[ok] / totw[ok]).astype(np.float32)
    alpha.reshape(-1)[:] = np.minimum(totw, 1.0).astype(np.float32)
    return rgb, alpha


def card_patch(u, v, side, ratio, cx, cy, tid, tex, W, H, rx0, ry0, rx1, ry1, gain=1.0):
    """ONE card (same maths as card_splat) on the pixel rectangle [rx0,rx1) x [ry0,ry1), which may reach outside the frame.
    Returns (P float32 (ph,pw,4)): premultiplied BGR in 0..255 and alpha 0..1 in the last channel."""
    pw, ph = rx1 - rx0, ry1 - ry0
    P = np.zeros((ph, pw, 4), np.float32)
    half = 0.5 * side
    xa, xb, ya, yb = u - half, u + half, v - half, v + half
    px0, px1 = max(int(math.floor(xa + 0.5)), rx0), min(int(math.floor(xb + 0.5)), rx1 - 1)
    py0, py1 = max(int(math.floor(ya + 0.5)), ry0), min(int(math.floor(yb + 0.5)), ry1 - 1)
    if px1 < px0 or py1 < py0:
        return P
    px = np.arange(px0, px1 + 1); py = np.arange(py0, py1 + 1)
    ox = np.clip(np.minimum(xb, px + 0.5) - np.maximum(xa, px - 0.5), 0.0, 1.0)
    oy = np.clip(np.minimum(yb, py + 0.5) - np.maximum(ya, py - 0.5), 0.0, 1.0)
    tx = cx + (px - u) / ratio
    ty = cy + (py - v) / ratio
    ins = ((tx >= -0.5) & (tx <= W - 0.5))[None, :] & ((ty >= -0.5) & (ty <= H - 0.5))[:, None]
    tx = np.clip(tx, 0.0, W - 1.0); ty = np.clip(ty, 0.0, H - 1.0)
    x0 = np.minimum(np.floor(tx).astype(np.int64), W - 2) if W > 1 else np.zeros(len(px), np.int64)
    y0 = np.minimum(np.floor(ty).astype(np.int64), H - 2) if H > 1 else np.zeros(len(py), np.int64)
    fx = (tx - x0).astype(np.float32)[None, :, None]
    fy = (ty - y0).astype(np.float32)[:, None, None]
    i00 = tid * (H * W) + y0[:, None] * W + x0[None, :]
    top = tex[i00] * (1.0 - fx) + tex[i00 + 1] * fx
    bot = tex[i00 + W] * (1.0 - fx) + tex[i00 + W + 1] * fx
    val = top * (1.0 - fy) + bot * fy
    a = (oy[:, None] * ox[None, :]).astype(np.float32) * val[..., 3] * ins * np.float32(gain)
    a = np.where(a > (1.0 / 512.0), a, 0.0).astype(np.float32)
    P[py0 - ry0:py1 - ry0 + 1, px0 - rx0:px1 - rx0 + 1, :3] = val[..., :3] * a[..., None]
    P[py0 - ry0:py1 - ry0 + 1, px0 - rx0:px1 - rx0 + 1, 3] = a
    return P


def smear_patch(P, dx, dy):
    """Continuous motion blur of one card patch: the exact line integral (box filter of length L = |(dx,dy)|, centred) of
    the premultiplied RGBA patch along its screen motion (dx,dy). The patch is rotated so the motion runs along x, box
    filtered with a cumulative sum (fractional ends are linearly interpolated, so there are no steps and no repeated
    edges whatever the length), and rotated back. L < 1 px: returned unchanged. The caller leaves a margin >= L/2."""
    L = math.hypot(dx, dy)
    if L < 1.0:
        return P
    ph, pw = P.shape[:2]
    c, sn = dx / L, dy / L
    D = int(math.ceil(math.hypot(ph, pw))) + 2
    Rm = np.array([[c, sn], [-sn, c]], np.float64)
    c0 = np.array([(pw - 1) / 2.0, (ph - 1) / 2.0])
    t = np.array([(D - 1) / 2.0] * 2) - Rm @ c0
    M = np.hstack([Rm, t[:, None]])
    Q = cv2.warpAffine(P, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    S = np.zeros((D, D + 1, 4), np.float64)
    np.cumsum(Q, axis=1, dtype=np.float64, out=S[:, 1:])
    xs = np.arange(D, dtype=np.float64)

    def F(p):                                                     # integral of the row from -inf to position p
        k = np.floor(p + 0.5).astype(np.int64)
        fr = (p + 0.5 - k)[None, :, None]
        return S[:, np.clip(k, 0, D)] * (1.0 - fr) + S[:, np.clip(k + 1, 0, D)] * fr
    B = ((F(xs + 0.5 * L) - F(xs - 0.5 * L)) / L).astype(np.float32)
    return cv2.warpAffine(B, cv2.invertAffineTransform(M), (pw, ph), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def load_video(W, H, shots=("s1a", "s2a")):
    """[(frames uint8 (n,H,W,3), mattes float32 (n,H,W))] per shot: the clip decoded and resized exactly like the first-frame
    layer (INTER_AREA), the per-frame alpha PNG (numbering 00001 = clip frame 0) resized INTER_LINEAR. depth_dolly._paths."""
    out = []
    for shot in shots:
        vid, alpha0, _ = dd._paths(shot)
        cap = cv2.VideoCapture(vid)                                # only for the size; frames are decoded by ffmpeg exactly
        w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # like first_frame()
        cap.release()
        pr = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", vid, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                              stdout=subprocess.PIPE)
        frames = []
        nb = w0 * h0 * 3
        while True:
            buf = pr.stdout.read(nb)
            if len(buf) < nb:
                break
            frames.append(cv2.resize(np.frombuffer(buf, np.uint8).reshape(h0, w0, 3), (W, H), interpolation=cv2.INTER_AREA))
        pr.stdout.close()
        pr.wait()
        from tools.fusion import pack_alpha                        # packed ingredients/*_matte.mkv when present, else the PNGs
        try:
            raw = pack_alpha.load_matte(alpha0)                    # uint8 (n,h,w) == cv2.imread(png, IMREAD_GRAYSCALE) per frame
        except FileNotFoundError:
            raw = []
        mattes = []
        for i in range(len(frames)):
            m = raw[i] if i < len(raw) else None
            if m is None:
                mattes.append(mattes[-1] if mattes else np.zeros((H, W), np.float32))
                continue
            mattes.append(cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR))
        out.append((np.stack(frames), np.stack(mattes)))
    return out


class PivotScene:
    """Shared pivot P = (0, 0, zp) on image 1's axis (zp = median Z). The camera orbits P on a circle of constant radius
    zp, always looking at P (yaw psi = -phi, position P + zp (sin phi, -cos phi)); phi = Phi * q(t). Both volumes sit
    fixed in the world (image 2 = T_end applied to its home points, T_end = camera at phi = Phi, so its home point at
    depth zp lands exactly on P). Seen from the camera: image 1 turns by phi, image 2 by phi - Phi (-in_angle -> 0).
    Image 2 is composited OVER image 1 (own coverage, soft edge) regardless of depth."""

    def __init__(self, layers, op: OrbitParams, video=None):
        import dataclasses
        self.op = op
        self.video = video
        self.typ = None                                               # v014: depth_type.TypeLayer (set by build_scene)
        self._q_cur = 0.0
        self.matte_pass = False                                       # v016: render the runner MATTE through the same scene (see render_matte_t)
        self._mtex = (None, None)
        self.t_cur = 0.0                                              # v015: output time (frames) of the frame being rendered (type timing)
        self._vf = [None, None]
        p = dataclasses.replace(layers[0].params, near_clip=op.near_clip)
        self.params = p
        self.W, self.H = layers[0].W, layers[0].H
        self.f = layers[0].f
        z1 = layers[0].grids[0][2]
        self.zp = float(np.median(z1)) if op.pivot == "median" else float(0.5 * (z1.min() + z1.max()))
        a_in = rad(op.in_angle)
        if op.overlap:
            self.Phi = a_in / op.overlap
        else:
            self.Phi = rad(op.out_angle + op.in_angle)
        self.out_angle = math.degrees(self.Phi) - op.in_angle         # image 1's turn when image 2 starts to enter
        self.ras = OrbitRasteriser(self.W, self.H, p, op.bg_dark)
        ex, ez, epsi = self.pose_phi(self.Phi)
        self.rl1 = RigidLayer(layers[0])
        self.rl2 = RigidLayer(layers[1], psi=epsi, t=(ex, 0.0, ez), s=1.0)
        for r in (self.rl1, self.rl2):
            r.nc = op.near_clip
        self.plates = [cv2.GaussianBlur(l.col, (0, 0), 22.0).astype(np.float32) * op.bg_dark * 0.8 for l in layers[:2]]
        self.L = self.Phi                                             # "path length" in rad for the spring overshoot
        self.d2 = 1.0
        self.cards = {}
        if op.splat == "billboard":
            for i, rl in enumerate((self.rl1, self.rl2)):
                un = (video[i][1] > 0.5).any(0) if video is not None else None
                self.cards[id(rl)] = make_cards(rl.L, op, op.seed + i, subj_union=un)
            if video is not None:
                self.set_frames(0, 0)

    # --- video textures (v013): geometry stays the frame-0 cards, only the texture changes per output frame
    def set_frames(self, f1, f2):
        """Textures of image 1 / image 2 = clip frame f1 / f2 (clamped to the clip: a clip that runs out holds its last
        frame). Runner card: texture = the frame (background mix removed at its soft edge), alpha = that frame's SOFT matte. Background cards: the frame
        with the runner filled out of it from the pixels around him (`subject_split`; was the static frame-0 LDI back layer = the ghost silhouette)."""
        H, W = self.H, self.W
        for k, (rl, f) in enumerate(zip((self.rl1, self.rl2), (f1, f2))):
            frames, mattes = self.video[k]
            f = int(min(max(f, 0), len(frames) - 1))
            if self._vf[k] == f:
                continue
            self._vf[k] = f
            col = frames[f].astype(np.float32)
            a = np.where(mattes[f] > 0.02, mattes[f], 0.0).astype(np.float32)      # the soft matte = the runner card's alpha (crisp, no 0.5 threshold, no halo)
            fg, bg = subject_split(col, a)
            tex = np.zeros((2, H, W, 4), np.float32)
            tex[0, ..., :3] = fg
            tex[0, ..., 3] = a
            tex[1, ..., :3] = bg                                  # background cards: this frame with the runner filled out of it (not the frame-0 LDI colour)
            tex[1, ..., 3] = 1.0
            self.cards[id(rl)]["tex"] = tex.reshape(-1, 4)

    def image2_start(self):
        """First output frame index at which image 2 is visible (start of its fade-in)."""
        for i, (t, _) in enumerate(frame_list(self.op)):
            if self.weights(self.q(t))[1] > 0.0:
                return i
        return len(frame_list(self.op))

    # --- pose
    def pose_phi(self, phi):
        return self.zp * math.sin(phi), self.zp * (1.0 - math.cos(phi)), -phi

    def end_pose(self):
        return self.pose_phi(self.Phi)

    def q(self, t):
        return q_at(t, self.op, self.Phi, 1.0)

    def rel_yaw(self, q):
        """(image 1, image 2) yaw as seen from the camera, degrees."""
        phi = q * self.Phi
        return math.degrees(phi), math.degrees(phi - self.Phi)

    def poses(self):
        out = []
        for t, kind in frame_list(self.op):
            q = self.q(t)
            x, z, psi = self.pose_phi(q * self.Phi)
            r1, r2 = self.rel_yaw(q)
            out.append((t, kind, x, z, math.degrees(psi), r1, r2))
        return out

    # --- render
    def _proj_bb(self, rl, cam):
        """Card centres / sides of one rigid layer seen from `cam`: (u, v, side, ratio, zc, ok_z)."""
        c = self.cards[id(rl)]
        L, op = rl.L, self.op
        M = (cam.R.T @ rl.Rk * rl.s).astype(np.float64)
        b = (cam.R.T @ (rl.t - cam.pos)).astype(np.float64)
        X, Y, Z = c["X"], c["Y"], c["Z"]
        xc = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z + b[0]
        yc = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z + b[1]
        zc = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z + b[2]
        ok = zc > rl.nc
        zs = np.where(ok, zc, 1.0)
        fk = float(getattr(cam, "fk", 1.0))                           # v023: zoom-out factor while the camera rolls (the frame is rotated + zoomed back after)
        u = L.f * fk * xc / zs + L.cx
        v = L.f * fk * yc / zs + L.cy
        S = c["S"] + 2.0 * op.card_grow
        ratio = np.minimum(rl.s * Z / zs * fk, op.card_max / S)
        return u, v, ratio * S, ratio, zc, ok

    def _pass_bb(self, rl, cam, sel=None):
        """Textured-card pass of one rigid layer: (img uint8 BGR, alpha float32 0..1). Every card stays screen-aligned;
        its centre is the (flat, one-depth) card point projected by the camera, its side = (S + 2 grow) * s * Z / Z_cam."""
        W, H = self.W, self.H
        c = self.cards[id(rl)]
        u, v, side, ratio, zc, ok = self._proj_bb(rl, cam)
        gain = self._gain(rl, cam, u, v, side, ratio, zc)
        if gain is not None:
            ok &= gain > 0.004
        ok &= (u + side / 2 > -0.5) & (u - side / 2 < W - 0.5) & (v + side / 2 > -0.5) & (v - side / 2 < H - 0.5)
        if sel is not None:                                           # v015: only the subject cards (tid 0) or only the background cards (tid 1)
            ok &= c["tid"] == sel
        k = np.nonzero(ok)[0]
        if not k.size:
            return np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.float32)
        order = np.argsort(zc[k] - c["bias"][k], kind="stable")      # near -> far
        rank = np.empty(len(k), np.int64)
        rank[order] = np.arange(len(k))
        rgb, al = card_splat(u[k], v[k], side[k], ratio[k], c["x"][k], c["y"][k], c["tid"][k], rank, self._tex_src(c), W, H, None if gain is None else gain[k].astype(np.float32))
        return np.clip(rgb + 0.5, 0, 255).astype(np.uint8), al

    def _pass_bb_motion(self, rl, cam, cam_a, cam_b, max_len=400.0, sel=None):
        """As _pass_bb, but every card is rendered once at the frame time and smeared along ITS OWN screen motion between
        the shutter-open camera `cam_a` and shutter-close camera `cam_b` (continuous blur, see smear_patch). Cards are
        composited far -> near with 'over' on premultiplied colour. A card with no motion (< 1 px) is left sharp."""
        W, H = self.W, self.H
        c = self.cards[id(rl)]
        u, v, side, ratio, zc, ok = self._proj_bb(rl, cam)
        gain = self._gain(rl, cam, u, v, side, ratio, zc)
        if gain is not None:
            ok = ok & (gain > 0.004)
        ua, va, _, _, _, oka = self._proj_bb(rl, cam_a)
        ub, vb, _, _, _, okb = self._proj_bb(rl, cam_b)
        good = oka & okb
        if sel is not None:
            ok = ok & (c["tid"] == sel)
        dx = np.where(good, ub - ua, 0.0); dy = np.where(good, vb - va, 0.0)
        Ln = np.hypot(dx, dy)
        sc = np.minimum(1.0, max_len / np.maximum(Ln, 1e-9))
        dx, dy, Ln = dx * sc, dy * sc, Ln * sc
        Cp = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), np.float32)
        far_first = np.argsort(-(zc - c["bias"]), kind="stable")
        for i in far_first:
            if not ok[i]:
                continue
            m = int(math.ceil(Ln[i] / 2.0)) + 2 if Ln[i] >= 1.0 else 0
            half = side[i] / 2.0
            rx0 = max(int(math.floor(u[i] - half + 0.5)) - m, -m); rx1 = min(int(math.floor(u[i] + half + 0.5)) + m + 1, W + m)
            ry0 = max(int(math.floor(v[i] - half + 0.5)) - m, -m); ry1 = min(int(math.floor(v[i] + half + 0.5)) + m + 1, H + m)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            P = card_patch(u[i], v[i], side[i], ratio[i], c["x"][i], c["y"][i], int(c["tid"][i]), self._tex_src(c), W, H, rx0, ry0, rx1, ry1,
                         1.0 if gain is None else float(gain[i]))
            if m:
                P = smear_patch(P, dx[i], dy[i])
            sx0, sx1, sy0, sy1 = max(rx0, 0), min(rx1, W), max(ry0, 0), min(ry1, H)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            sub = P[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0]
            a = sub[..., 3]
            Cp[sy0:sy1, sx0:sx1] = Cp[sy0:sy1, sx0:sx1] * (1.0 - a[..., None]) + sub[..., :3]
            A[sy0:sy1, sx0:sx1] = A[sy0:sy1, sx0:sx1] * (1.0 - a) + a
        rgb = np.where(A[..., None] > 1e-6, Cp / np.maximum(A, 1e-6)[..., None], 0.0)
        return np.clip(rgb + 0.5, 0, 255).astype(np.uint8), np.minimum(A, 1.0)

    def _pass(self, rl, cam):
        if self.op.splat == "billboard":
            img, al = self._pass_bb(rl, cam)
            return img, al > 0.5
        got = self.ras._splat([(rl, None)], cam, self.params.crack_iters)
        H, W = self.H, self.W
        if got is None:
            return np.zeros((H, W, 3), np.uint8), np.zeros((H, W), bool)
        R, c = got
        mask = R > 0
        img = c[np.maximum(R.astype(np.int64) - 1, 0)]
        img[~mask] = 0
        return img, mask

    def weights(self, q):
        """Scalar visibility of image 1 (fades out 75 -> 90 deg, where it turns edge-on) and of image 2 (fades in over
        its first fade_in_deg, where it starts to face the camera)."""
        r1, r2 = self.rel_yaw(q)
        w1 = 1.0 - float(dd.smootherstep((r1 - 75.0) / 15.0))
        w2 = float(dd.smootherstep((r2 + self.op.in_angle) / max(self.op.fade_in_deg, 1e-3)))
        return w1, w2

    def blur_mode(self):
        b = self.op.blur
        if b == "auto":
            b = "continuous" if self.op.splat == "billboard" else "steps"
        if b == "continuous" and self.op.splat != "billboard":
            b = "steps"
        return b

    def render_q(self, q, both=True, blur_t=None):
        cam = Cam(*self.pose_phi(q * self.Phi))
        if q <= 0.0:
            return self._rest(self.rl1, 0, cam, q, blur_t)
        if q == 1.0:
            return self._rest(self.rl2, 1, cam, q, blur_t)
        w1, w2 = self.weights(q)
        bb = self.op.splat == "billboard"
        ctx = None
        if bb and blur_t is not None:                                 # continuous: smear each card over the shutter
            ca = Cam(*self.pose_phi(self.q(blur_t - 0.5 * self.op.shutter) * self.Phi))
            cb = Cam(*self.pose_phi(self.q(blur_t + 0.5 * self.op.shutter) * self.Phi))
            ctx = (ca, cb)
        pas = lambda rl, cm: self._layer_pass(rl, cm, q, blur_t, ctx)  # noqa: E731
        img1, m1 = pas(self.rl1, cam) if w1 > 0 else (None, None)
        img2, m2 = pas(self.rl2, cam) if w2 > 0 else (None, None)
        H, W = self.H, self.W
        v1 = np.zeros((H, W), np.float32) if m1 is None else m1.astype(np.float32) * w1
        if m2 is None:
            a2 = np.zeros((H, W), np.float32)
        elif bb:                                                      # card edges are already antialiased
            a2 = m2.astype(np.float32) * w2
        else:
            m = cv2.erode(m2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
            a2 = np.clip(cv2.GaussianBlur(m, (0, 0), self.op.edge_soft), 0, 1) * w2
        pre = np.zeros((H, W, 3), np.uint8)
        if img1 is not None:
            pre = img1.copy()
        if img2 is not None:
            pre = np.where((a2 > 0.5)[..., None], img2, pre)
        valid = np.maximum(v1, a2)
        wp = float(dd.smootherstep((q - 0.35) / 0.4))
        plate = (1.0 - wp) * self.plates[0] + wp * self.plates[1]
        bg = fill_bg(pre, valid, plate, self.op.bg_dark)
        if self.matte_pass:                                           # matte pass: holes are 'not runner', not diffused colour
            bg = np.zeros_like(bg)
        under = v1[..., None] * (img1.astype(np.float32) if img1 is not None else 0.0) + (1.0 - v1[..., None]) * bg
        front = self.typ is not None and not self._sandwich_on()
        if front:                                                     # image 1's word over image 1 (image 2 still covers it)
            under = self.typ.over(under, 0, q, blur_t)
        out = a2[..., None] * (img2.astype(np.float32) if img2 is not None else 0.0) + (1.0 - a2[..., None]) * under
        if front:                                                     # image 2's word on top, with image 2 (same pose + shutter)
            out = self.typ.over(out, 1, q, blur_t)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _gain(self, rl, cam, u, v, side, ratio, zc):
        """per-card opacity of a pass (None = all 1); the volume modes (v021-v023) override it."""
        return None

    def _tex_src(self, c):
        """the cards' texture; in a matte pass the same texture as a MATTE: runner card white (alpha = his matte, as in the colour pass),
        background cards black and opaque (so they occlude him exactly where they do in colour)."""
        if not self.matte_pass:
            return c["tex"]
        if self._mtex[0] is not c["tex"]:
            t = c["tex"].reshape(2, self.H, self.W, 4).copy()
            t[0, ..., :3] = 255.0
            t[1, ..., :3] = 0.0
            self._mtex = (c["tex"], t.reshape(-1, 4))
        return self._mtex[1]

    def render_matte_t(self, t, nsub=1):
        """float32 (H, W) 0..1: the runner's MATTE as it appears in the output frame at time t. The same scene as `render_t` (same cards,
        camera pose, depth order, per-card continuous blur, fades, image 2 over image 1); the texture is the per-frame matte (video) or the
        frame-0 matte, runner card white / background cards black (opaque), holes 0. No type. Home frames == the source matte."""
        typ, self.matte_pass = self.typ, True
        self.typ = None
        try:
            fr = self.render_t(t, nsub)
        finally:
            self.typ, self.matte_pass = typ, False
        return fr[..., 0].astype(np.float32) / 255.0

    def _typ(self, img, which, q, blur_t=None):
        """uint8 frame with image `which`'s type plane over it (front mode; at a rest pose blur_t only blurs the word's own motion)."""
        if self.typ is None:
            return img
        return np.clip(self.typ.over(img.astype(np.float32), which, q, blur_t) + 0.5, 0, 255).astype(np.uint8)

    def _sandwich_on(self):
        """v015: type between the runner and the background (billboard cards only): background cards -> type plane -> runner card."""
        return self.typ is not None and self.typ.mode == "behind_subject" and self.op.splat == "billboard"

    def _layer_pass(self, rl, cam, q, blur_t=None, ctx=None):
        """(img uint8 BGR, alpha float32) of one image volume: the plain card pass, or the sandwich when the type sits behind the subject."""
        bb = self.op.splat == "billboard"
        if not bb:
            return self._pass(rl, cam)
        if self._sandwich_on():
            return self._sandwich(rl, cam, q, blur_t, ctx)
        return self._pass_bb_motion(rl, cam, ctx[0], ctx[1]) if ctx is not None else self._pass_bb(rl, cam)

    def _sandwich(self, rl, cam, q, blur_t=None, ctx=None):
        """background cards -> the image's type plane -> the runner card with his per-frame matte alpha (he occludes the letters).
        The image's visibility weight is applied by the caller on the whole result, so the word fades with its image."""
        which = 0 if rl is self.rl1 else 1
        H, W = self.H, self.W
        if ctx is not None:
            bg, ab = self._pass_bb_motion(rl, cam, ctx[0], ctx[1], sel=1)
            sj, asub = self._pass_bb_motion(rl, cam, ctx[0], ctx[1], sel=0)
        else:
            bg, ab = self._pass_bb(rl, cam, sel=1)
            sj, asub = self._pass_bb(rl, cam, sel=0)
        C = bg.astype(np.float32) * ab[..., None]
        A = ab.astype(np.float32).copy()
        ty = self.typ.layer(which, q, blur_t, self.t_cur, weight=False)
        if ty is not None:
            pm, ta = ty
            C = C * (1.0 - ta[..., None]) + pm
            A = A * (1.0 - ta) + ta
        C = C * (1.0 - asub[..., None]) + sj.astype(np.float32) * asub[..., None]
        A = A * (1.0 - asub) + asub
        img = np.where(A[..., None] > 1e-6, C / np.maximum(A, 1e-6)[..., None], 0.0)
        return np.clip(img + 0.5, 0, 255).astype(np.uint8), np.minimum(A, 1.0)

    def _rest(self, rl, which, cam, q, blur_t=None):
        """frame at a rest pose (q = 0 or 1): the image's own cards (+ its type plane)."""
        if self._sandwich_on():
            return self._sandwich(rl, cam, q, blur_t, None)[0]
        return self._typ(self._pass(rl, cam)[0], which, q, blur_t)

    def render_t(self, t, nsub=1):
        bm = self.blur_mode()
        self.t_cur = float(t)
        if bm == "off":
            return self.render_q(self.q(t))
        if bm == "continuous":
            q = self.q(t)
            if 0.0 < q < 1.0:
                return self.render_q(q, blur_t=t)
            return self.render_q(q, blur_t=t if self.typ is not None else None)   # rest pose: only the word moves (v015)
        if nsub <= 1:
            return self.render_q(self.q(t))
        acc = None
        for i in range(nsub):
            tt = t + self.op.shutter * ((i + 0.5) / nsub - 0.5)
            fr = self.render_q(self.q(tt)).astype(np.float32)
            acc = fr if acc is None else acc + fr
        return np.clip(acc / nsub + 0.5, 0, 255).astype(np.uint8)

    def nsub_at(self, t):
        d = abs(self.q(t + 0.5) - self.q(t - 0.5)) * self.Phi
        px = self.f * d * 0.7                                          # the pivot stays put; the periphery sweeps
        return int(np.clip(math.ceil(px * self.op.shutter / self.op.sub_px), 1, self.op.max_sub))


# ----------------------------------------------------------------------------------------------- v021 / v022 / v023: camera paths through the card volume
def _sm01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cam_R(yaw, pitch):
    """world -> camera basis, columns (right, down, forward); yaw > 0 turns toward +x, pitch > 0 looks UP (y is down)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    f = np.array([sy * cp, -sp, cy * cp])
    r = np.array([cy, 0.0, -sy])
    return np.stack([r, np.cross(f, r), f], axis=1)


class CamG:
    """general rigid camera: position (3), yaw / pitch (rotation), roll (deg, applied to the finished frame - a pure rotation about the optical
    axis is exactly an image rotation about the principal point) and fk (<= 1: the frame is rendered at a wider field of view and rotated /
    zoomed back, so a rolled frame has no empty corners)."""

    def __init__(self, pos, yaw=0.0, pitch=0.0, roll=0.0, fk=1.0):
        self.pos = np.asarray(pos, np.float64)
        self.yaw, self.pitch, self.roll, self.fk = float(yaw), float(pitch), float(roll), float(fk)
        self.R = cam_R(yaw, pitch)


def volume_pose(mode, q, op, aspect=762.0 / 432.0):
    """-> dict(pos, yaw, pitch, roll (deg), fk) of the camera at path progress q (0..1) for the volume modes. Home at q = 0 (image 1) and q = 1
    (image 2): position, yaw, pitch, roll all exact, fk = 1."""
    q = float(q)
    dz = float(op.vol_dz)
    env = math.sin(math.pi * q) ** 0.6 if 0.0 < q < 1.0 else 0.0
    if mode == "push":
        return dict(pos=(0.0, 0.0, dz * q), yaw=0.0, pitch=0.0, roll=0.0, fk=1.0)
    if mode == "pullback":
        rise = -float(op.pull_rise) * math.sin(math.pi * q) if 0.0 < q < 1.0 else 0.0      # y is down: negative = up
        return dict(pos=(0.0, rise, -dz * q), yaw=0.0, pitch=0.0, roll=0.0, fk=1.0)
    ax, ay = op.fly_amp
    x = ax * math.sin(2.0 * math.pi * q) * env                        # one S-curve through the volume ...
    y = ay * math.sin(4.0 * math.pi * q) * env                        # ... with a rise and a dip
    yaw = math.radians(op.fly_yaw) * math.cos(2.0 * math.pi * q) * env     # heading follows the lateral velocity
    pitch = math.radians(op.fly_pitch) * math.cos(4.0 * math.pi * q) * env
    roll = float(op.roll_deg) * math.sin(2.0 * math.pi * q) * env        # banks into the turn (bank ~ lateral acceleration), flat at both ends
    th = math.radians(abs(roll))
    fk = 1.0 / (math.cos(th) + math.sin(th) * max(aspect, 1.0 / aspect))
    return dict(pos=(x, y, dz * q), yaw=yaw, pitch=pitch, roll=roll, fk=fk)


class VolumeScene(PivotScene):
    """Two depth volumes on ONE optical axis and one rigid camera path through them (pivot machinery reused: cards, video textures, continuous blur, matte
    pass). push / fly: image 2 sits dz BEHIND image 1 (world t = (0, 0, +dz)); the camera travels forward through image 1's cards (cards that reach the lens
    fade; image 1's cards dissolve from the centre outward = the chunks part) and arrives at image 2's home. pullback: image 2 sits dz BEHIND THE CAMERA
    START (t = (0, 0, -dz)); the camera backs away from image 1 and image 2's cards fly past the lens and settle at its home. Both homes are exact (camera at the
    volume's origin, identity rotation). Per frame the front layer's coverage is kept in `front_alpha` (the type is hidden behind it)."""

    def __init__(self, layers, op: OrbitParams, video=None):
        super().__init__(layers, op, video)
        self.kind = op.mode
        sign = -1.0 if op.mode == "pullback" else 1.0
        old = self.rl2
        self.rl2 = RigidLayer(layers[1], psi=0.0, t=(0.0, 0.0, sign * op.vol_dz), s=1.0)
        self.rl2.nc = op.near_clip
        self.cards[id(self.rl2)] = self.cards.pop(id(old))
        self.front = self.rl2 if op.mode == "pullback" else self.rl1
        self.plates = [cv2.GaussianBlur(l.col, (0, 0), 30.0).astype(np.float32) * op.vol_bg for l in layers[:2]]      # a bright diffused surround, not the pivot's dark one
        self.front_alpha = None
        self._vf = [None, None]
        if video is not None:
            self.set_frames(0, 0)
        self.aspect = self.H / float(self.W)

    # --- the path
    def pose(self, q):
        return volume_pose(self.kind, q, self.op, self.aspect)

    def cam_q(self, q):
        p = self.pose(q)
        return CamG(p["pos"], p["yaw"], p["pitch"], p["roll"], p["fk"])

    def poses(self):
        out = []
        for t, kind in frame_list(self.op):
            p = self.pose(self.q(t))
            out.append((t, kind) + tuple(float(v) for v in p["pos"]) + (math.degrees(p["yaw"]), math.degrees(p["pitch"]), p["roll"], p["fk"]))
        return out

    # --- per-card opacity: lens fade, zoom fade, and (push / fly) the dissolve of image 1 from the centre outward
    def _gain(self, rl, cam, u, v, side, ratio, zc):
        op, q = self.op, self._q_cur
        c = self.cards[id(rl)]
        g = _sm01((zc - op.near_clip) / max(op.near_fade - op.near_clip, 1e-6))
        zoom = ratio / max(getattr(cam, "fk", 1.0), 1e-6)
        # background cards (home sides up to ~0.9 of the frame width) fade when they would exceed 1.0 -> 1.3 frame widths: no card becomes a wall;
        # the runner card (a cut-out, bigger box) fades by its zoom relative to home (2.5 -> 4 x)
        g_bg = _sm01((1.3 * self.W - side) / (0.3 * self.W))
        g_sub = _sm01((op.zoom_fade[1] - zoom) / (op.zoom_fade[1] - op.zoom_fade[0]))
        g = g * np.where(c["tid"] == 0, g_sub, g_bg)
        if rl is self.front and self.kind in ("push", "fly"):
            nx = (c["x"] - rl.L.cx) / (self.W / 2.0)
            ny = (c["y"] - rl.L.cy) / (self.H / 2.0)
            r = np.hypot(nx, ny) / math.sqrt(2.0)
            R = -op.hole_soft + (op.hole_max + op.hole_soft) * float(_sm01((q - op.hole0) / max(op.hole1 - op.hole0, 1e-6)))
            g = g * (1.0 - _sm01((R - r) / op.hole_soft + 0.5))
        return g

    # --- frame
    def _roll_warp(self, img, cam, interp=cv2.INTER_LINEAR):
        if abs(cam.roll) < 1e-6 and abs(cam.fk - 1.0) < 1e-9:
            return img
        M = cv2.getRotationMatrix2D((self.rl1.L.cx, self.rl1.L.cy), cam.roll, 1.0 / cam.fk)
        return cv2.warpAffine(img, M, (self.W, self.H), flags=interp, borderMode=cv2.BORDER_REPLICATE)

    def render_q(self, q, both=True, blur_t=None):
        self._q_cur = q
        H, W = self.H, self.W
        cam = self.cam_q(q)
        if q <= 0.0 or q >= 1.0:
            rl = self.rl1 if q <= 0.0 else self.rl2
            img, al = self._pass_bb(rl, cam)
            self.front_alpha = np.ones((H, W), np.float32) if q <= 0.0 else np.zeros((H, W), np.float32)
            return img
        ctx = None
        if blur_t is not None:
            ca = self.cam_q(self.q(blur_t - 0.5 * self.op.shutter))
            cb = self.cam_q(self.q(blur_t + 0.5 * self.op.shutter))
            ctx = (ca, cb)
        pas = (lambda rl: self._pass_bb_motion(rl, cam, ctx[0], ctx[1])) if ctx is not None else (lambda rl: self._pass_bb(rl, cam))
        fr_img, fr_a = pas(self.front)
        bk = self.rl2 if self.front is self.rl1 else self.rl1
        bk_img, bk_a = pas(bk)
        if self.kind == "pullback":                                   # image 1 recedes and fades; the gaps between image 2's arriving cards are filled from image 2
            bk_a = bk_a * float(1.0 - _sm01((q - 0.15) / 0.45))
            wp = float(_sm01((q - 0.1) / 0.35))
        else:
            wp = float(dd.smootherstep((q - 0.35) / 0.4))
        plate = (1.0 - wp) * self.plates[0] + wp * self.plates[1]
        bg = np.asarray(plate, np.float32)                            # a bright blurred backdrop of the arriving image (no dark diffusion halo)
        if self.matte_pass:
            bg = np.zeros_like(bg)
        under = bk_a[..., None] * bk_img.astype(np.float32) + (1.0 - bk_a[..., None]) * bg
        out = fr_a[..., None] * fr_img.astype(np.float32) + (1.0 - fr_a[..., None]) * under
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        out = self._roll_warp(out, cam)
        self.front_alpha = self._roll_warp(np.clip(fr_a, 0.0, 1.0).astype(np.float32), cam)
        return out

    def render_t(self, t, nsub=1):
        self.t_cur = float(t)
        q = self.q(t)
        return self.render_q(q, blur_t=t if 0.0 < q < 1.0 else None)

    def nsub_at(self, t):
        return 1

    def image2_start(self):
        """first output frame at which image 2 is on screen enough to play its clip: push / fly = the dissolve of image 1 has opened (q >= 0.45); pullback = image 2's cards arrive (q >= 0.7)."""
        thr = 0.7 if self.kind == "pullback" else 0.45
        for i, (t, _) in enumerate(frame_list(self.op)):
            if self.q(t) >= thr:
                return i
        return len(frame_list(self.op))



# ----------------------------------------------------------------------------------------------- build
def build_scene(depth_from, op: OrbitParams):
    p = dd.Params(depth_ratio=op.depth_ratio)
    layers = dd.make_layers(depth_from, op.short, p)[:2]
    if op.blur == "continuous" and op.splat != "billboard":
        raise SystemExit("--blur continuous is implemented for --splat billboard only")
    if op.splat == "billboard" and op.mode not in ("pivot", "push", "pullback", "fly"):
        raise SystemExit("--splat billboard is implemented for --mode pivot / push / pullback / fly only")
    if op.video and op.splat != "billboard":
        raise SystemExit("--video needs --splat billboard --mode pivot")
    if op.mode in ("pivot", "push", "pullback", "fly"):
        video = load_video(layers[0].W, layers[0].H) if op.video else None
        sc = (PivotScene if op.mode == "pivot" else VolumeScene)(layers, op, video=video)
        if op.type:                                                   # v014: the piece's words as type planes in the same scene
            from tools.fusion import depth_type as dtp
            sc.typ = dtp.TypeLayer(sc, op.type, depth=op.type_depth, mode=op.type_layer, style=op.type_style, scale=op.type_scale)
        return sc, layers
    geo = make_geometry(layers, op)
    return OrbitScene(layers, geo, op), layers


def run_name(a):
    return os.path.basename(os.path.normpath(a.out)) + (("_" + a.tag) if a.tag else "")


def ops_from_args(a):
    return OrbitParams(depth_ratio=a.depth_ratio, angle=a.angle, lead=a.lead, travel=a.travel, spring=a.spring,
                       enter_angle=a.enter_angle, radius_end=a.radius, scale=a.scale, short=a.short, mode=a.mode,
                       out_angle=a.out_angle, in_angle=a.in_angle, overlap=a.overlap, splat=a.splat,
                       max_chunks=a.max_chunks, size_range=tuple(float(x) for x in a.size_range.split(',')), seed=a.seed,
                       blur=a.blur, subject_cards=a.subject_cards, video=a.video, vol_dz=a.vol_dz, hole0=a.hole0, hole1=a.hole1, roll_deg=a.roll_deg, fly_amp=(a.fly_amp_x, a.fly_amp_y), fly_yaw=a.fly_yaw,
                       fly_pitch=a.fly_pitch, pull_rise=a.pull_rise, bg_chunks=a.bg_chunks, type=a.type, type_depth=a.type_depth,
                       type_layer=a.type_layer, type_style=a.type_style, type_scale=a.type_scale, hold1=a.hold1, hold2=a.hold2)


def sheet_frames(op, n=10):
    """n frame indices spread over the transition (start of the move .. just after the landing)."""
    fl = frame_list(op)
    first = op.hold1
    last = op.hold1 + op.travel + (6 if op.spring else 2)
    idx = np.unique(np.round(np.linspace(first - 1, last, n)).astype(int))
    return [int(i) for i in idx if i < len(fl)]


def cmd_render(a):
    op = ops_from_args(a)
    outd, review, local = dd.version_dirs(a.out)
    t0 = time.time()
    sc, layers = build_scene(a.depth_from or local, op)
    if op.mode in ("push", "pullback", "fly"):
        info = {"mode": op.mode, "vol_dz": op.vol_dz, "hole": [op.hole0, op.hole1], "roll_deg": op.roll_deg}
        print("cards per image:", [(int(c["n_subject"]), int(c["n_back"])) for c in sc.cards.values()], "| mode %s dz %.2f" % (op.mode, op.vol_dz), flush=True)
    elif op.mode == "pivot":
        info = {"zp": sc.zp, "Phi_deg": math.degrees(sc.Phi), "out_angle": sc.out_angle, "in_angle": op.in_angle}
        if sc.cards:
            info["cards"] = [{"subject": int(c["n_subject"]), "background": int(c["n_back"]), "ref_px": float(c["ref"]),
                              "size_px": [float(c["S"].min()), float(c["S"].max())]} for c in sc.cards.values()]
            print("cards per image:", info["cards"], flush=True)
        if sc.typ is not None:
            info["type"] = sc.typ.report
            for r in sc.typ.report:
                print("type plane:", r, flush=True)
        print("layers+pivot %.1fs | zp=%.2f total camera orbit %.0f deg (image 1 turned %.0f when image 2 enters, "
              "image 2 enters at %.0f deg)" % (time.time() - t0, sc.zp, math.degrees(sc.Phi), sc.out_angle, op.in_angle),
              flush=True)
    else:
        g = sc.geo
        info = {"s": g.s, "L": g.path.L, "mind": g.mind}
        print("layers+geometry %.1fs | s=%.3f zp=%.2f zc2=%.2f path L=%.2f (A %.2f conn %.2f B %.2f) psi_end=%.1f deg "
              "min cloud dist %.2f" % (time.time() - t0, g.s, g.zp, g.zc2, g.path.L, g.path.LA, g.path.Lc, g.path.LB,
                                       math.degrees(g.path.psi_e), g.mind), flush=True)
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    tag = a.tag or ("spring" if op.spring else "plain")
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), tag)))
    proc = dd._ffmpeg_writer(mp4, sc.W, sc.H, op.fps)
    mproc, mdir = None, None
    fdir = None
    if op.mode in ("push", "pullback", "fly"):                      # v021-v023: the front volume's coverage per frame (the type hides behind it)
        gm1 = re.search(r"_(g\d\d)\.mp4$", mp4)
        fdir = dd.unique_path(os.path.join(local, "front_%s%s" % (tag, ("_" + gm1.group(1)) if gm1 else "")))
        os.makedirs(fdir, exist_ok=True)
    if getattr(a, "matte_pass", False):                            # v016: the runner matte through the same scene, frame by frame
        gm0 = re.search(r"_(g\d\d)\.mp4$", mp4)
        mdir = dd.unique_path(os.path.join(local, "matte_%s%s" % (tag, ("_" + gm0.group(1)) if gm0 else "")))
        os.makedirs(mdir, exist_ok=True)
        mproc = dd._ffmpeg_writer(mp4[:-4] + "_matte.mp4", sc.W, sc.H, op.fps)
    fl = frame_list(op)
    keep_idx = sheet_frames(op, 12 if op.mode == "pivot" else 10)
    keep, cache = {}, {}
    i2 = sc.image2_start() if getattr(sc, "video", None) is not None else 0
    if getattr(sc, "video", None) is not None:
        n1, n2 = len(sc.video[0][0]), len(sc.video[1][0])
        info["video"] = {"image1_frame": "output i -> clip frame min(i, %d)" % (n1 - 1),
                         "image2_start_output_frame": i2,
                         "image2_frame": "output i -> clip frame min(max(i - %d, 0), %d)" % (i2, n2 - 1),
                         "clip_frames": [n1, n2]}
        print("video mapping:", info["video"], flush=True)
    t1 = time.time()
    for i, (t, kind) in enumerate(fl):
        q = sc.q(t)
        if getattr(sc, "video", None) is not None:
            sc.set_frames(i, i - i2)
        sc.t_cur = float(t)
        if q in (0.0, 1.0) and (getattr(sc, "video", None) is not None or getattr(sc, "typ", None) is not None):
            fr = sc.render_q(q, blur_t=float(t) if getattr(sc, "typ", None) is not None else None)   # video / type move at rest poses too
        elif q in (0.0, 1.0):
            if q not in cache:
                cache[q] = sc.render_q(q)
            fr = cache[q]
        else:
            fr = sc.render_t(t, sc.nsub_at(t))
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        if fdir is not None and getattr(sc, "front_alpha", None) is not None:
            cv2.imwrite(os.path.join(fdir, "%04d.png" % i), np.clip(sc.front_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
        if mproc is not None:
            mm = np.clip(sc.render_matte_t(t, sc.nsub_at(t) if 0.0 < q < 1.0 else 1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(mdir, "%04d.png" % i), mm)
            mproc.stdin.write(np.ascontiguousarray(np.repeat(mm[..., None], 3, -1)).tobytes())
        if i in keep_idx:
            keep[i] = fr.copy()
        if i % 6 == 0:
            print("  %s frame %d/%d t=%.0f q=%.3f %s  %.0fs" % (tag, i, len(fl), t, q, kind, time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    if mproc is not None:
        mproc.stdin.close()
        mproc.wait()
        print("matte pass ->", mdir, flush=True)
    # sheet tiles + poses of this take (never overwritten: the generation suffix comes from the mp4 name);
    # the combined sheet is built with the `sheet` command (--gen gNN)
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    gen = "_" + gm.group(1) if gm else ""
    tiles = [(keep[i], "t=%d" % fl[i][0]) for i in keep_idx if i in keep]
    np.savez_compressed(os.path.join(local, "sheet_%s%s.npz" % (tag, gen)),
                        **{"f%02d" % j: f for j, (f, _) in enumerate(tiles)}, texts=np.array([t for _, t in tiles]))
    json.dump({"tag": tag, "poses": sc.poses(), **info, "params": {k: v for k, v in op.__dict__.items()}},
              open(os.path.join(local, "poses_%s%s.json" % (tag, gen)), "w"))
    print("%s -> %s (%.0fs render, %.0fs total)" % (tag, mp4, time.time() - t1, time.time() - t0), flush=True)


def build_sheet(review, local, gen=""):
    """orbit_sheet<gen>.png from the takes sheet_<tag><gen>.npz (plain row, spring row). Never overwrites (unique_path)."""
    rows = {}
    for tag in ("plain", "spring"):
        f = os.path.join(local, "sheet_%s%s.npz" % (tag, gen))
        if os.path.exists(f):
            z = np.load(f)
            rows["orbit " + tag] = [(z["f%02d" % j], str(z["texts"][j])) for j in range(len(z["texts"]))]
    if rows:
        out = dd.unique_path(os.path.join(review, "orbit_sheet%s.png" % gen))
        dd._sheet(rows, out, tw=140, cols=12)
        print("sheet ->", out)


def cmd_sheet(a):
    outd, review, local = dd.version_dirs(a.out)
    build_sheet(review, local, ("_" + a.gen) if a.gen else "")


# ----------------------------------------------------------------------------------------------- diagram
def cmd_diagram(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    op = ops_from_args(a)
    outd, review, local = dd.version_dirs(a.out)
    sc, layers = build_scene(a.depth_from or local, op)
    g, path = sc.geo, sc.path
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_axes([0.05, 0.30, 0.62, 0.66])
    ax2 = fig.add_axes([0.72, 0.30, 0.25, 0.66])
    axa = fig.add_axes([0.07, 0.05, 0.9, 0.18])
    ax.scatter(g.c1[:, 0], g.c1[:, 2], s=1, c="tab:blue", alpha=0.35, label="image 1 volume (footprint)")
    ax.scatter(g.c2w[:, 0], g.c2w[:, 2], s=1, c="tab:red", alpha=0.35, label="image 2 volume (footprint, s=%.2f)" % g.s)
    for box, col in ((g.aabb1, "tab:blue"), (g.aabb2, "tab:red")):
        lo, hi = box
        ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec=col, ls=":", lw=1))
    taus = np.linspace(0, path.L * 1.01, 500)
    xy = np.array([path.pose(t)[:2] for t in taus])
    ax.plot(xy[:, 0], xy[:, 1], "k-", lw=1.5, label="camera path")
    ax.plot(*path.P2, "r+", ms=10)
    ax.plot(0, g.zp, "b+", ms=10)
    hf = rad(55.0 / 2)
    fr = list(range(0, 8))
    for k, q in enumerate(np.linspace(0, 1, 9)):
        x, z, psi = path.pose(q * path.L)
        ln = 1.4
        for sgn in (-1, 1):
            ang = psi + sgn * hf
            ax.plot([x, x + ln * math.sin(ang)], [z, z + ln * math.cos(ang)], color=plt.cm.viridis(q), lw=1)
        ax.plot(x, z, "o", color=plt.cm.viridis(q), ms=5)
        ax.annotate("%.0f%%" % (100 * q), (x, z), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (world, right)")
    ax.set_ylabel("z (world, forward) - top-down")
    ax.set_title("orbit path: %.0f deg arc, lead %.2f, radius %.2f->%.2f, enter %.0f deg, s=%.2f"
                 % (op.angle, op.lead, g.zp, op.radius_end, op.enter_angle, g.s))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    # zoom: the arrival arc round image 2
    ax2.scatter(g.c1[:, 0], g.c1[:, 2], s=2, c="tab:blue", alpha=0.35)
    ax2.scatter(g.c2w[:, 0], g.c2w[:, 2], s=2, c="tab:red", alpha=0.35)
    ax2.plot(xy[:, 0], xy[:, 1], "k-", lw=1.5)
    for q in np.linspace(path.LA / path.L, 1.0, 6):
        x, z, psi = path.pose(q * path.L)
        for sgn in (-1, 1):
            ang = psi + sgn * hf
            ax2.plot([x, x + 1.2 * math.sin(ang)], [z, z + 1.2 * math.cos(ang)], color=plt.cm.plasma(q), lw=1)
        ax2.plot(x, z, "o", color=plt.cm.plasma(q), ms=4)
    pe = g.c2w
    ax2.set_xlim(pe[:, 0].min() - 2.5, pe[:, 0].max() + 2.5)
    ax2.set_ylim(pe[:, 2].min() - 2.5, pe[:, 2].max() + 2.5)
    ax2.set_aspect("equal")
    ax2.set_title("arrival (zoom)")
    ax2.grid(alpha=0.3)
    # image 2's angle to the camera over time
    t = np.arange(0, op.hold1 + op.travel + op.hold2)
    for spring, col in ((False, "tab:green"), (True, "tab:orange")):
        o2 = OrbitParams(**{**op.__dict__, "spring": spring})
        ang = []
        for ti, kind in frame_list(o2):
            ang.append(math.degrees(path.pose(q_at(ti, o2, path.L, path.d2) * path.L)[2] - path.psi_e))
        axa.plot(np.arange(len(ang)), ang, color=col, label="%s" % ("spring" if spring else "plain"))
    axa.axhline(op.enter_angle, color="gray", ls=":")
    axa.axhline(0, color="gray", ls=":")
    axa.set_ylabel("image 2 angle to camera (deg)", fontsize=8)
    axa.set_xlabel("frame")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3)
    out = dd.unique_path(os.path.join(review, "orbit_path.png"))
    fig.savefig(out, dpi=110)
    print("diagram ->", out)


def cmd_diagram_pivot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    op = ops_from_args(a)
    outd, review, local = dd.version_dirs(a.out)
    sc, layers = build_scene(a.depth_from or local, op)
    c1, c2 = _cloud(layers[0]), _cloud(layers[1])
    ex, ez, epsi = sc.end_pose()
    c2w = place_image2(c2, epsi, (ex, ez), 1.0)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[2.3, 1.2, 1.0], hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(c1[:, 0], c1[:, 2], s=1, c="tab:blue", alpha=0.35, label="image 1 volume (home)")
    ax.scatter(c2w[:, 0], c2w[:, 2], s=1, c="tab:red", alpha=0.35, label="image 2 volume (T_end applied)")
    phis = np.linspace(0, sc.Phi * 1.0, 200)
    ax.plot(sc.zp * np.sin(phis), sc.zp * (1 - np.cos(phis)), "k-", lw=1.5, label="camera path (radius %.2f about P)" % sc.zp)
    ax.plot(0, sc.zp, "k+", ms=14, mew=2)
    ax.annotate("P (pivot)", (0, sc.zp), textcoords="offset points", xytext=(6, -12))
    hf = rad(55.0 / 2)
    for q in np.linspace(0, 1, 7):
        x, z, psi = sc.pose_phi(q * sc.Phi)
        for sgn in (-1, 1):
            ang = psi + sgn * hf
            ax.plot([x, x + 1.6 * math.sin(ang)], [z, z + 1.6 * math.cos(ang)], color=plt.cm.viridis(q), lw=1)
        ax.plot(x, z, "o", color=plt.cm.viridis(q), ms=5)
        ax.annotate("%.0f deg" % math.degrees(q * sc.Phi), (x, z), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (world)")
    ax.set_ylabel("z (world) - top-down")
    ax.set_title("pivot mode: camera orbits P (centre of image 1's volume) at constant radius, always looking at P; "
                 "total %.0f deg; image 2 = T_end applied to its home points" % math.degrees(sc.Phi))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    # camera frame: both volumes as the camera sees them (top-down), at 5 times
    for k, q in enumerate(np.linspace(0, 1, 5)):
        axk = fig.add_subplot(gs[1, k])
        cam = Cam(*sc.pose_phi(q * sc.Phi))
        for cl, col, vis in ((c1, "tab:blue", True), (c2w, "tab:red", True)):
            pts = cl if cl is c1 else c2w
            pw = pts - np.array([cam.pos[0], 0.0, cam.pos[2]])
            pc = pw @ cam.R                                     # = R^T (X - C)
            axk.scatter(pc[:, 0], pc[:, 2], s=0.8, c=col, alpha=0.3)
        for sgn in (-1, 1):
            axk.plot([0, sgn * 5 * math.tan(hf)], [0, 5], color="gray", lw=0.8)
        axk.plot(0, sc.zp, "k+", ms=10, mew=1.5)
        axk.set_xlim(-6, 6)
        axk.set_ylim(-1, 9)
        axk.set_aspect("equal")
        axk.set_title("camera frame, phi=%.0f" % math.degrees(q * sc.Phi), fontsize=9)
        axk.tick_params(labelsize=7)
    axa = fig.add_subplot(gs[2, :])
    for spring, col, ls in ((False, "k", "-"), (True, "tab:orange", "--")):
        o2 = OrbitParams(**{**op.__dict__, "spring": spring})
        qs = [q_at(t, o2, sc.Phi, 1.0) for t, _ in frame_list(o2)]
        axa.plot([math.degrees(q * sc.Phi) for q in qs], color="tab:blue", ls=ls, label="image 1 yaw to camera (%s)" % ("spring" if spring else "plain"))
        axa.plot([math.degrees(q * sc.Phi - sc.Phi) for q in qs], color="tab:red", ls=ls, label="image 2 yaw to camera (%s)" % ("spring" if spring else "plain"))
    axa.axhline(0, color="gray", ls=":")
    axa.axhline(-op.in_angle, color="gray", ls=":")
    axa.set_xlabel("frame")
    axa.set_ylabel("yaw to camera (deg)")
    axa.legend(fontsize=7, ncol=2)
    axa.grid(alpha=0.3)
    out = dd.unique_path(os.path.join(review, "orbit_path.png"))
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print("diagram ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "diagram", "sheet"])
    ap.add_argument("--out", default=os.path.join(ROOT, "17_depth-dolly", "v005"))
    ap.add_argument("--depth-from", default=None)
    ap.add_argument("--short", type=int, default=OrbitParams.short)
    ap.add_argument("--depth-ratio", type=float, default=OrbitParams.depth_ratio)
    ap.add_argument("--angle", type=float, default=OrbitParams.angle, help="orbit angle (deg) at which the arrival arc starts")
    ap.add_argument("--lead", type=float, default=OrbitParams.lead, help="yaw = -lead * orbit angle")
    ap.add_argument("--radius", type=float, default=OrbitParams.radius_end, help="orbit radius at --angle")
    ap.add_argument("--enter-angle", type=float, default=OrbitParams.enter_angle, help="image 2 enters turned this much (deg)")
    ap.add_argument("--scale", type=float, default=None, help="image 2 scale about the end camera (default: auto)")
    ap.add_argument("--travel", type=int, default=OrbitParams.travel)
    ap.add_argument("--mode", choices=["orbit", "pivot", "push", "pullback", "fly"], default="orbit", help="orbit = v005 (arc off + arrival arc); "
                    "pivot = v006 (constant-radius pivot about image 1's centre, image 2 swings in over the top)")
    ap.add_argument("--out-angle", type=float, default=OrbitParams.out_angle, help="pivot: image 1's turn when image 2 enters")
    ap.add_argument("--in-angle", type=float, default=OrbitParams.in_angle, help="pivot: image 2 enters turned this much")
    ap.add_argument("--overlap", type=float, default=None, help="pivot: image 2's share of the travel (overrides --out-angle)")
    ap.add_argument("--splat", choices=["stretch", "billboard"], default="stretch", help="pivot: stretch = blocks stretched to "
                    "their continuous neighbours (v007); billboard = textured camera-facing chunk cards (v008)")
    ap.add_argument("--max-chunks", type=int, default=100, help="billboard: hard cap on cards per image volume")
    ap.add_argument("--size-range", default="1.5,4", help="billboard: card side = U(lo,hi) x sqrt(W*H/max_chunks) (log-uniform)")
    ap.add_argument("--seed", type=int, default=0, help="billboard: jitter / size seed")
    ap.add_argument("--blur", choices=["auto", "steps", "continuous", "off"], default="auto", help="motion blur: steps = average of "
                    "<= 8 sharp sub-frames (v008), continuous = each card smeared along its own screen motion over the shutter "
                    "(billboard only), off; auto = continuous for billboard, steps for stretch")
    ap.add_argument("--subject-cards", type=int, choices=[0, 1], default=1, help="billboard: 1 = the subject is ONE cut-out "
                    "card (full matte as alpha), 0 = jittered subject cards (v008)")
    ap.add_argument("--video", action="store_true", help="billboard pivot: project the live s1a / s2a clips onto the frame-0 "
                    "cards (image 1 plays from output frame 0, image 2 from the frame it first becomes visible, a finished clip "
                    "holds its last frame); runner alpha = that frame's matte")
    ap.add_argument("--bg-chunks", type=int, default=0, help="billboard: cap the background cards (0 = what --max-chunks leaves after the "
                    "subject cards); the card size reference still comes from --max-chunks (v014: 32 = v009's background)")
    ap.add_argument("--type", default="", help="pivot: 'P06:WORD ON IMAGE 1,WORD ON IMAGE 2' - the piece's words as flat type planes "
                    "moved by the same camera (tools/fusion/depth_type.py)")
    ap.add_argument("--type-depth", type=float, default=None, help="type plane depth: behind_subject = fraction of the way from the subject's Z "
                    "to the background's (default 0.5); front = this x the subject's median Z (default 0.8)")
    ap.add_argument("--type-layer", choices=["behind_subject", "front"], default="behind_subject",
                    help="behind_subject = background cards -> type -> runner card by his per-frame matte (compose's `behind` recipe: the "
                         "subject drawn over the type); front = the type over the whole image (compose's front= / v014)")
    ap.add_argument("--type-style", choices=["organic", "v014"], default="organic",
                    help="organic = word 1 carried with the turn, word 2 forms after the landing, beside the runner (v015); v014 = q-retimed, centred")
    ap.add_argument("--vol-dz", type=float, default=5.0, help="push / pullback / fly: camera travel along the axis (image 2 sits this far behind image 1 / behind the start)")
    ap.add_argument("--hole0", type=float, default=0.20, help="push / fly: image 1's cards start to dissolve from the centre at this q")
    ap.add_argument("--hole1", type=float, default=0.85, help="... and are all gone at this q")
    ap.add_argument("--roll-deg", type=float, default=32.0, help="fly: peak bank (deg; actual peak ~27)")
    ap.add_argument("--fly-amp-x", type=float, default=0.9)
    ap.add_argument("--fly-amp-y", type=float, default=0.4)
    ap.add_argument("--fly-yaw", type=float, default=14.0)
    ap.add_argument("--fly-pitch", type=float, default=7.0)
    ap.add_argument("--pull-rise", type=float, default=0.3, help="pullback: the camera rises this much mid-way")
    ap.add_argument("--matte-pass", action="store_true", help="also render the runner's MATTE through the same scene (same cards / camera / "
                    "blur / fades): local/matte_<tag>_<gen>/NNNN.png (8-bit, lossless) + out/..._matte.mp4 (grey preview)")
    ap.add_argument("--type-scale", type=float, default=OrbitParams.type_scale, help="organic: word size relative to the full-width layout")
    ap.add_argument("--hold1", type=int, default=OrbitParams.hold1, help="frames of image 1 before the move")
    ap.add_argument("--hold2", type=int, default=OrbitParams.hold2, help="frames on image 2 after the landing")
    ap.add_argument("--spring", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen", default="", help="sheet: take generation to combine, e.g. g02 (default: the first take)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "diagram" and a.mode == "pivot":
        cmd_diagram_pivot(a)
        return
    {"render": cmd_render, "diagram": cmd_diagram, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
