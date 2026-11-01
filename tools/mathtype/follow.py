#!/usr/bin/env python
"""Math-type test A3 v2: camera follow (10 s, 4:5).

One continuous 2D world. The three phrases sit at different places in it: P1 at the origin, P2 one
frame-width to the right, P3 one frame-height below P2. A tracking camera (power2 in-out, ~0.2 s
behind the particle streams) carries the eye from phrase to phrase: P1 -> P2 exits through the right
edge, P2 -> P3 through the bottom edge.

Particles never settle crisply. Each particle's home is target + a living, bounded offset (a coherent
divergent wave field, so gaps open and close, plus a per-particle Lissajous orbit, ~8% of cap height,
a few outliers to ~25%). The offset shrinks to ~4.5% in each phrase's read window but never stops.

Each phrase's target points (stratified over the letter interiors, as in v1) are split into equal-count
spatial patches. Patches leave and arrive in a seeded order; a patch's flight is a closed-form two-phase
curve: ease-in (constant acceleration) peel, then a damped spring (zeta 0.8) into the hovering home.
Each move has its own path family, added to the straight path with envelope sin(pi*u):
  P1 build    curl-noise flow (curl of a sum-of-sines stream function)
  P1 -> P2    vortex rope: lanes twist twice about the travel axis, radius pinching as exp(-k sin(pi u)),
              depth per patch, so the patches braid as ribbons
  P2 -> P3    damped harmonograph per patch (across 2:1 + 3:1, along 3:1, loops back on itself)
Thick -> hairline -> thick: on departure a bubble thins to a ~0.8 px hairline head (brightness held) and
grows a long trail (20 previous samples of its own path, world space, seen through the camera, fading
along its length); on arrival the trail retracts and the bubble swells back. Pure function of t.

TIMING (96 bpm, beat = 0.625 s; 'land' = last patch arrived = pluck; read window = land -> next peel)
  event                         t (s)              beat
  P1 patches fly in (curl)      -0.7 .. 1.09       (first arrivals ~0.2)
  P1 landed / pluck 1           1.09375            1.75
  P1 read window                1.09375-2.65625    (1.56 s)
  P1 -> P2 (rope) peel          2.65625            4.25   transition 1.875 s
  camera pan right              2.856-4.431        power2 in-out
  P2 landed / pluck 2           4.53125            7.25
  P2 read window                4.53125-6.09375    (1.56 s)
  P2 -> P3 (harmonograph) peel  6.09375            9.75   transition 1.875 s
  camera pan down               6.294-7.869
  P3 landed / pluck 3           7.96875            12.75
  P3 read window                7.96875-10.0       (2.03 s)
  credit swell / sets in        8.4375             13.5 (letters landed by ~8.9)

    python tools/mathtype/follow.py --draft          # 360x450 -> v002/local/A3_follow_draft.mp4
    python tools/mathtype/follow.py --final          # 1080x1350 -> v002/local/A3_follow.mp4
    python tools/mathtype/follow.py --still 1.0 3.2  # PNG stills (draft size) -> v002/review/
    python tools/mathtype/follow.py --hum --mux      # hum_A3.wav, then A3_follow_hum.mp4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from particles import (BG, DH, DW, EDGE, INK, ACCENT, line_w, load_font,  # noqa: E402
                       stratified)

REPO = os.path.dirname(os.path.dirname(HERE))
VDIR = os.path.join(REPO, "11_mathtype", "v002")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

FPS, DUR = 25, 10.0
NFR = int(round(FPS * DUR))
BEAT = 0.625

# ------------------------------------------------------------------------------------------ timing
LAND = [1.75 * BEAT, 7.25 * BEAT, 12.75 * BEAT]   # 1.09375, 4.53125, 7.96875 (plucks)
PEEL = [4.25 * BEAT, 9.75 * BEAT]                  # 2.65625, 6.09375
CREDIT_T = 13.5 * BEAT                             # 8.4375
SPREAD = 0.7                                       # patch start spread per transition (s)
STAG = 0.12                                        # extra stagger inside a patch (s)
SPREAD0, F0 = 0.8, 0.9                             # P1 build: patch spread, flight time (s)
CAM_LAG = 0.2                                      # camera starts this long after the first peel
CAM_DUR = 1.575

# ------------------------------------------------------------------------------------------ world
FONTS = ("Le Murmure", "Blackout")               # tall line over puffy line
PHRASES = [("THERE IS", "GEOMETRY"), ("IN THE", "HUMMING"), ("OF THE", "STRINGS")]
MAX_CAP = {"Le Murmure": 330.0, "Blackout": 300.0}
CREDIT_FONT = "Le Murmure"
TEXT_W = 980.0
CX, CY = 540.0, 650.0
ORIGIN = np.array([[0.0, 0.0], [DW, 0.0], [DW, DH]])   # world offset of each phrase's frame
DIRS = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]   # exit direction of move 1 and move 2
N = 3200
K = 18
SEED = 11

# ------------------------------------------------------------------------------------------ motion
ZETA = 0.8
XA, GA, WN = 0.5, 0.45, 7.0      # peel phase ends at x=0.5 having covered 45%; spring wn in 1/flight
HOVER_BASE, HOVER_READ = 0.08, 0.045     # fraction of the reference cap height
OUTLIER_FRAC = 0.05
CURL_AMP = 130.0                 # P1 build: curl-noise flow displacement (design px)
ROPE_TURNS, ROPE_PINCH = 2, 0.7  # P1->P2: lanes twist twice about the travel axis, radius exp(-k sin)
HARM_AMP = (150.0, 70.0, 120.0)  # P2->P3: harmonograph amplitudes (across 2:1 / 3:1, along 3:1)
TRAIL_M, TRAIL_DT = 20, 0.017    # silk trails: previous samples of the closed-form path
HAIR_PX = 0.8                    # hairline width in output px
TRAIL_FRAC, TRAIL_PEAK = 0.4, 0.75   # share of particles that draw a trail; trail head brightness
NSUB = 5                         # motion-blur subframes
SHUTTER = 0.5                    # fraction of a frame


def ease2(x):
    """power2 in-out."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < 0.5, 2 * x * x, 1 - 0.5 * (2 - 2 * x) ** 2)


def flight(x: np.ndarray) -> np.ndarray:
    """Normalised flight progress: constant-acceleration peel to x=XA, then a damped spring
    (zeta 0.8) from the matching position/velocity. g(0)=0, g(1)~1.0, overshoot ~1%."""
    a = 2 * GA / XA ** 2
    v0, e0 = a * XA, 1 - GA
    wd = WN * np.sqrt(1 - ZETA ** 2)
    tau = np.maximum(x - XA, 0.0)
    e = np.exp(-ZETA * WN * tau) * (e0 * np.cos(wd * tau) + ((-v0 + ZETA * WN * e0) / wd) * np.sin(wd * tau))
    g = np.where(x < XA, 0.5 * a * np.clip(x, 0, None) ** 2, 1 - e)
    return np.where(x > 4.0, 1.0, g)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def phrase_layout(lines):
    """Two-line stack, each line justified to TEXT_W (cap capped per font). Mask + metrics."""
    fonts, caps, tops = [], [], []
    for s, fn in zip(lines, FONTS):
        probe = load_font(fn, 200)
        pb = probe.getbbox("H")
        size = 200 * TEXT_W / line_w(probe, s)
        size = int(min(size, 200 * MAX_CAP[fn] / (pb[3] - pb[1])))
        f = load_font(fn, size)
        fonts.append(f)
        hb = f.getbbox("H")
        caps.append(hb[3] - hb[1])
    gap = 0.16 * caps[1] + 18
    top = CY - (sum(caps) + gap) / 2
    im = Image.new("L", (DW, DH), 0)
    d = ImageDraw.Draw(im)
    y0, rights = top, []
    from particles import TRACK
    for s, f, cap in zip(lines, fonts, caps):
        hb = f.getbbox("H")
        x = CX - line_w(f, s) / 2 - f.getbbox(s[0])[0]
        rights.append(CX + line_w(f, s) / 2)
        for ch in s:
            d.text((x, y0 - hb[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        tops.append(y0)
        y0 += cap + gap
    return np.array(im), dict(caps=caps, ref_cap=float(min(caps)), bottom=y0 - gap, right=max(rights),
                              top=top)


def kd_patches(pts: np.ndarray, k: int) -> np.ndarray:
    """Equal-count spatial patches by recursive median-style splits along the longer side
    (equal counts keep patch-to-patch particle blocks aligned, so streams don't spill sideways)."""
    lab = np.zeros(len(pts), int)

    def split(idx, k, base):
        if k == 1:
            lab[idx] = base
            return
        k1 = k // 2
        ext = np.ptp(pts[idx], axis=0)
        ax = int(ext[1] * 1.6 > ext[0])               # favour vertical cuts (letters side by side)
        o = idx[np.argsort(pts[idx, ax], kind="stable")]
        cut = int(round(len(idx) * k1 / k))
        split(o[:cut], k1, base)
        split(o[cut:], k - k1, base + k1)

    split(np.arange(len(pts)), k, 0)
    return lab


def band_sort(xy: np.ndarray, across: int, nb: int) -> np.ndarray:
    """Indices sorted into nb equal-count bands along axis `across`, then along the other axis."""
    o = np.argsort(xy[:, across], kind="stable")
    out = []
    for b in np.array_split(o, nb):
        out.append(b[np.argsort(xy[b, 1 - across], kind="stable")])
    return np.concatenate(out)


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.size_q = rng.random(N)
        self.order = rng.permutation(N)                       # fixed draw order
        loc, rad, lab, self.met = [], [], [], []
        for lines in PHRASES:
            mask, met = phrase_layout(lines)
            pts, dist = stratified(mask, N, rng)
            if not rad:
                self.base_r = 0.6 * np.sqrt((mask > 127).sum() / N)
            r = self.base_r * (0.55 + 0.9 * self.size_q ** 1.6)
            lb = kd_patches(pts, K)
            loc.append(pts)
            rad.append(np.minimum(r, dist + 2.5))
            lab.append(lb)
            self.met.append(met)

        # ---- patch orders and particle matching -------------------------------------------------
        # P1 arrives (from the left) in a seeded order biased right-first, so later patches never
        # fly through landed ones. Every later phrase arrives near-side first; the departing patches
        # are assigned to arrival slots by centroid position across the travel axis (Hungarian), so
        # streams keep their lanes, and particles are matched block-to-block in that shared order.
        def centroids(k):
            return np.array([loc[k][lab[k] == j].mean(0) for j in range(K)])

        rank = [None, None, None]                              # rank[k][patch] in arrival order
        c0 = centroids(0)
        key = 0.6 * (1 - (c0[:, 0] - c0[:, 0].min()) / np.ptp(c0[:, 0])) + rng.random(K)
        rank[0] = np.argsort(np.argsort(key))
        perm = [np.arange(N)]                                  # perm[k][i]: phrase-k point of particle i
        dep_rank = [None, None]
        for m, dvec in enumerate(DIRS):
            a, b = m, m + 1
            ca, cb = centroids(a), centroids(b)
            along = cb @ dvec
            keyb = 0.6 * (along - along.min()) / np.ptp(along) + rng.random(K)
            rb = np.argsort(np.argsort(keyb))                  # arrival rank of each phrase-b patch
            slot_patch_b = np.argsort(rb)                      # patch of phrase b in slot s
            across = np.array([dvec[1], dvec[0]])
            xa_ = (ca @ across - CX * across.sum()) / 300.0
            xb_ = (cb[slot_patch_b] @ across - CX * across.sum()) / 300.0
            lead = (ca @ dvec) / np.ptp(ca @ dvec)              # front of the departing word
            cost = np.abs(xa_[:, None] - xb_[None, :]) + 0.25 * np.abs(
                (1 - (lead - lead.min()))[:, None] - np.arange(K)[None, :] / (K - 1))
            ri, ci = linear_sum_assignment(cost)
            ra = np.empty(K, int)
            ra[ri] = ci                                        # departure rank of phrase-a patch
            dep_rank[m] = ra
            rank[b] = rb
            # particle order: current phrase-a positions of the particles, by (dep rank, across)
            pa = perm[a]
            la = lab[a][pa]
            ax_a = loc[a][pa] @ across
            oa = np.lexsort((ax_a, ra[la]))                    # particle indices, departure order
            lbb = lab[b]
            ob = np.lexsort((loc[b] @ across, rb[lbb]))        # phrase-b points, arrival order
            pb = np.empty(N, int)
            pb[oa] = ob
            perm.append(pb)

        self.loc = [loc[k][perm[k]] for k in range(3)]         # aligned per particle
        self.rad = [rad[k][perm[k]] for k in range(3)]
        self.world = [self.loc[k] + ORIGIN[k] for k in range(3)]

        # ---- schedules: departure time d, flight duration f per move -----------------------------
        self.moves = []
        stag = rng.random((3, N)) * STAG
        lab0 = lab[0][perm[0]]
        d0 = LAND[0] - F0 - STAG - SPREAD0 + rank[0][lab0] * SPREAD0 / (K - 1) + stag[0]
        src = self.world[0].copy()
        src[:, 0] -= 1250 + 260 * rng.random(N)
        src[:, 1] = CY + (src[:, 1] - CY) * 0.55 + 90 * (rng.random(N) - 0.5)
        self.src = src
        self.moves.append(dict(d=d0, f=np.full(N, F0), dir=np.array([1.0, 0.0])))
        for m in range(2):
            la = lab[m][perm[m]]
            f = LAND[m + 1] - PEEL[m] - STAG - SPREAD
            d = PEEL[m] + dep_rank[m][la] * SPREAD / (K - 1) + stag[m + 1]
            self.moves.append(dict(d=d, f=np.full(N, f), dir=DIRS[m]))

        # ---- path families -------------------------------------------------------------------------
        # P1 build: curl-noise flow. Stream function psi = sum a_j sin(k_j.p + nu_j t + phi_j); the
        # displacement is its curl (dpsi/dy, -dpsi/dx), evaluated along the straight path.
        ang = rng.random(3) * 2 * np.pi
        lam = 260 + 260 * rng.random(3)
        kv = np.stack([np.cos(ang), np.sin(ang)], 1) * (2 * np.pi / lam)[:, None]
        self.curl = dict(k=kv, nu=0.8 + 0.6 * rng.random(3), ph=rng.random(3) * 2 * np.pi)
        # P1 -> P2: the swarm's lanes twist ROPE_TURNS times about the travel axis (a helix seen side-on)
        # while the radius pinches by exp(-ROPE_PINCH sin(pi u)): a log-spiral vortex rope.
        # depth is per patch (plus a little jitter), so each patch twists as one coherent ribbon
        self.rope_z = (rng.random(K) * 2 - 1)[lab0] * 260.0 + 40.0 * (rng.random(N) - 0.5)
        # P2 -> P3: damped harmonograph per patch (ribbons), small per-particle phase spread.
        lab1 = lab[1][perm[1]]
        self.harm_ph = (rng.random((K, 3)) * 2 * np.pi)[lab1] + 0.5 * rng.random((N, 3))
        self.harm_amp = 0.75 + 0.5 * rng.random(N)

        # ---- hover field ---------------------------------------------------------------------------
        self.waves = []
        for k in range(3):
            ang = rng.random(4) * 2 * np.pi
            lam = 110 + 150 * rng.random(4)
            kv = np.stack([np.cos(ang), np.sin(ang)], 1) * (2 * np.pi / lam)[:, None]
            ev = rng.normal(size=(4, 2))
            ev /= np.linalg.norm(ev, axis=1, keepdims=True)
            self.waves.append(dict(k=kv, e=ev, nu=(0.7 + 0.7 * rng.random(4)) * rng.choice([-1, 1], 4),
                                   psi=rng.random(4) * 2 * np.pi))
        self.orb_w = 0.9 + 1.3 * rng.random((N, 2))
        self.orb_ph = rng.random((N, 2)) * 2 * np.pi
        out = rng.random(N) < OUTLIER_FRAC
        self.amp = np.where(out, 2.6 + 0.6 * rng.random(N), 0.75 + 0.5 * rng.random(N))
        self.orb_w[out] *= 0.6
        self.has_trail = rng.random(N) < TRAIL_FRAC

    # ------------------------------------------------------------------------------------ state
    def read_env(self, k: int, t: float) -> float:
        up = smooth((t - LAND[k] + 0.25) / 0.5)
        down = 1.0 if k == 2 else 1 - smooth((t - PEEL[k] + 0.3) / 0.3)
        return float(up * down)

    def hover(self, k: int, t: float) -> np.ndarray:
        A = self.met[k]["ref_cap"] * (HOVER_BASE - (HOVER_BASE - HOVER_READ) * self.read_env(k, t))
        w = self.waves[k]
        ph = self.loc[k] @ w["k"].T + w["nu"] * t + w["psi"]           # (N, 4)
        field = np.sin(ph) @ w["e"] / np.sqrt(2.0)
        orb = np.sin(self.orb_w * t + self.orb_ph)
        return (A * self.amp)[:, None] * (0.7 * field + 0.6 * orb)

    def cam(self, t: float) -> np.ndarray:
        c = ORIGIN[0].copy()
        for m in range(2):
            c = c + (ORIGIN[m + 1] - ORIGIN[m]) * float(ease2((t - PEEL[m] - CAM_LAG) / CAM_DUR))
        return c

    def pos(self, t: float):
        """World positions (N,2), bubble radii, thread factor s (0 bubble .. 1 hairline), shade."""
        homes = [self.world[k] + self.hover(k, t) for k in range(3)]
        p = self.src.copy()
        r = np.full(N, self.base_r * 0.55)
        thread = np.zeros(N)
        shade = np.ones(N)
        prev, prev_r = self.src, r
        for k, mv in enumerate(self.moves):
            x = (t - mv["d"]) / mv["f"]
            g = flight(x)
            p = p + (homes[k] - prev) * g[:, None]
            r = r + (self.rad[k] - prev_r) * g
            thread = np.maximum(thread, smooth(x / 0.16) * (1 - smooth((x - 0.72) / 0.28)))
            gb = np.clip(g, 0, 1)
            env = np.sin(np.pi * gb)
            if k == 0:                                         # curl-noise flow
                w = self.curl
                ph = p @ w["k"].T + w["nu"] * t + w["ph"]
                kn = w["k"] / np.linalg.norm(w["k"], axis=1, keepdims=True)
                cu = np.cos(ph) @ np.stack([kn[:, 1], -kn[:, 0]], 1) / np.sqrt(1.5)
                p = p + (CURL_AMP * env)[:, None] * cu
            elif k == 1:                                       # vortex rope about y = CY
                y = p[:, 1] - CY
                th = 2 * np.pi * ROPE_TURNS * gb
                rho = np.exp(-ROPE_PINCH * env)
                p[:, 1] = CY + rho * (y * np.cos(th) - self.rope_z * np.sin(th))
                depth = (y * np.sin(th) + self.rope_z * np.cos(th)) / 300.0
                shade = shade * (1 - 0.4 * env * (0.5 - 0.5 * np.clip(depth, -1, 1)))
            else:                                              # damped harmonograph
                A0, A1, A2 = HARM_AMP
                ph = self.harm_ph
                dmp = np.exp(-1.1 * gb) * env * self.harm_amp
                ox = A0 * np.sin(4 * np.pi * gb + ph[:, 0]) + A1 * np.sin(6 * np.pi * gb + ph[:, 1])
                oy = A2 * np.sin(6 * np.pi * gb + ph[:, 2])
                p = p + (dmp[:, None] * np.stack([ox, oy], 1))
            prev, prev_r = homes[k], self.rad[k]
        return p, r, thread, shade

    def moving(self, t: float) -> bool:
        for mv in self.moves:
            x = (t - mv["d"]) / mv["f"]
            if np.any((x > -0.05) & (x < 1.4)):
                return True
        return any(0 < (t - PEEL[m] - CAM_LAG) / CAM_DUR < 1 for m in range(2))

    # ------------------------------------------------------------------------------------ drawing
    def trails(self, t: float, W: int, H: int) -> np.ndarray:
        """Supersampled background canvas with the silk trails at time t. Drawn once per frame (the
        trail already is the motion record; blurring it over the shutter only turns threads to fog)."""
        ss = max(1, int(round(2 * DW / W)))
        sc = W / DW * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        c = self.cam(t)
        p, r, th, shade = self.pos(t)
        img = np.empty((Hc, Wc, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        # silk trails: TRAIL_M previous samples of each particle's own path, in world space, seen
        # through the current camera, fading along their length; length scales with the thread factor
        idx = np.nonzero((th > 0.02) & self.has_trail)[0]
        if idx.size:
            samp = [p[idx]] + [self.pos(t - j * TRAIL_DT)[0][idx] for j in range(1, TRAIL_M + 1)]
            q = np.round((np.stack(samp, 0) - c) * sc * Sh).astype(np.int32)   # (M+1, n, 2)
            layer = np.zeros((Hc, Wc), np.uint8)
            thick = max(1, int(round(HAIR_PX * ss)))
            nb = 12
            for j in range(TRAIL_M - 1, -1, -1):
                fade = (1 - j / TRAIL_M) ** 1.3
                a = TRAIL_PEAK * th[idx] * shade[idx] * fade * (j < th[idx] * TRAIL_M)
                lev = np.ceil(a * nb).astype(int)
                seg = np.stack([q[j], q[j + 1]], 1)                            # (n, 2, 2)
                for L in range(1, nb + 1):
                    sel = lev == L
                    if sel.any():
                        cv2.polylines(layer, list(seg[sel]), False, int(255 * L / nb), thick, cv2.LINE_AA, 4)
            lf = layer.astype(np.float32)[..., None] / 255.0
            img = (BG + (INK - BG) * lf).astype(np.uint8)
        return img

    def draw(self, t: float, W: int, H: int, base: np.ndarray) -> np.ndarray:
        """Bubbles / hairline heads at time t over a copy of the trail canvas."""
        ss = max(1, int(round(2 * DW / W)))
        osc = W / DW
        sc = osc * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        c = self.cam(t)
        p, r, th, shade = self.pos(t)
        hair = HAIR_PX / osc                                   # hairline width in design px
        img = base.copy()
        # bubbles: thick at rest, a hairline head while threading
        rd = r + (0.6 * hair - r) * th
        xy = (p - c) * sc
        m = 60 * sc
        vis = (xy[:, 0] > -m) & (xy[:, 0] < Wc + m) & (xy[:, 1] > -m) & (xy[:, 1] < Hc + m)
        cq = np.round(xy * Sh).astype(np.int64)
        lw = max(1.3 * osc, 0.45) * ss
        R = np.maximum(rd * sc, 0.5 * ss)
        Ro = np.round(R * Sh).astype(np.int64)
        Ri = np.round(np.maximum(R - lw, 0.3 * ss) * Sh).astype(np.int64)
        Rf = np.round(R * Sh).astype(np.int64)
        edge = th < 0.3                                        # threads lose the dark bubble outline
        fill = BG + (INK - BG) * shade[:, None]
        for i in self.order:
            if not vis[i]:
                continue
            c_ = (int(cq[i, 0]), int(cq[i, 1]))
            if edge[i]:
                cv2.circle(img, c_, int(Ro[i]), EDGE, -1, cv2.LINE_AA, 4)
                cv2.circle(img, c_, int(Ri[i]), tuple(float(v) for v in fill[i]), -1, cv2.LINE_AA, 4)
            else:
                cv2.circle(img, c_, int(Rf[i]), tuple(float(v) for v in fill[i]), -1, cv2.LINE_AA, 4)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def frame(self, t: float, W: int, H: int) -> np.ndarray:
        base = self.trails(t, W, H)
        if self.moving(t):
            dts = (np.arange(NSUB) / (NSUB - 1) - 0.5) * SHUTTER / FPS
            acc = np.zeros((H, W, 3), np.float32)
            for d in dts:
                acc += self.draw(t + d, W, H, base)
            img = np.clip(acc / NSUB + 0.5, 0, 255).astype(np.uint8)
        else:
            img = self.draw(t, W, H, base)
        if t >= CREDIT_T:
            img = self.credit(img, t, W / DW)
        return img

    def credit(self, img: np.ndarray, t: float, sc: float) -> np.ndarray:
        """'— PYTHAGORAS' in Le Murmure: letters drop in one after another on a zeta-0.8 spring."""
        met = self.met[2]
        c = self.cam(t)
        size = max(int(round(124 * sc)), 10)
        f = load_font(CREDIT_FONT, size)
        text = "— PYTHAGORAS"
        track = 0.12 * size
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + track * (len(text) - 1)
        x = (met["right"] + ORIGIN[2][0] - c[0]) * sc - total
        y = (met["bottom"] + 0.42 * met["caps"][1] + ORIGIN[2][1] - c[1]) * sc
        layer = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        w, wd = 13.0, 13.0 * np.sqrt(1 - ZETA ** 2)
        for i, ch in enumerate(text):
            tau = max(t - CREDIT_T - 0.035 * i, 0.0)
            u = 1 - np.exp(-ZETA * w * tau) * (np.cos(wd * tau) + ZETA * w / wd * np.sin(wd * tau))
            a = smooth(tau / 0.18)
            dy = (1 - u) * -46 * sc
            col = ACCENT if i == 0 else tuple(int(q) for q in INK)
            if a > 0:
                d.text((x, y + dy), ch, font=f, fill=col + (int(255 * a),))
            x += widths[i] + track
        out = Image.fromarray(img).convert("RGBA")
        out.alpha_composite(layer)
        return np.array(out.convert("RGB"))


# ---------------------------------------------------------------------------------------------- output
_SC = None


def _init():
    global _SC
    _SC = Scene()


def _work(args):
    n, W, H = args
    return _SC.frame(n / FPS, W, H).tobytes()


def render(path: str, W: int, H: int, procs: int = 8) -> float:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with mp.Pool(procs, initializer=_init) as pool:
        for buf in pool.imap(_work, [(n, W, H) for n in range(NFR)], chunksize=2):
            proc.stdin.write(buf)
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"{path}  {W}x{H}  {NFR} frames  {dt:.1f}s")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hum", action="store_true")
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    a = ap.parse_args()
    if a.draft:
        render(os.path.join(LOCAL, "A3_follow_draft.mp4"), 360, 450)
    if a.final:
        render(os.path.join(LOCAL, "A3_follow.mp4"), 1080, 1350)
    if a.still:
        os.makedirs(REVIEW, exist_ok=True)
        sc = Scene()
        for t in a.still:
            Image.fromarray(sc.frame(t, 360, 450)).save(os.path.join(REVIEW, f"A3_still_{t:.2f}.png"))
    wav = os.path.join(LOCAL, "hum_A3.wav")
    if a.hum:
        subprocess.run([sys.executable, os.path.join(HERE, "hum.py"), "--duration", str(DUR),
                        "--plucks", *[f"{x:g}" for x in LAND], "--credit", f"{CREDIT_T:g}", "--out", wav],
                       check=True)
    if a.mux:
        if not os.path.exists(wav):
            sys.exit("hum_A3.wav not found (run --hum)")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(LOCAL, "A3_follow.mp4"),
                        "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        os.path.join(LOCAL, "A3_follow_hum.mp4")], check=True)


if __name__ == "__main__":
    main()
