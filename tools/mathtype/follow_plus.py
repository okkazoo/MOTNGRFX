#!/usr/bin/env python
"""Math-type test A3b: camera follow, "+" grid variant (18.2 s, 4:5). Sibling of follow.py (A3).

Same world and camera structure as A3 (P1 at the origin, P2 one frame-width right, P3 one
frame-height below P2; exits right, then down; camera power2 in-out, 0.2 s behind the streams) and
the same silk-thread look, slowed down. Flight paths are cubic Hermite arcs (no wobble): the start
tangent follows a broad, static curl-noise flow (one or two eddies, shared by neighbours), the end
tangent is a gentle approach along the travel direction, progress decelerates into the slot.

Differences from A3:
- Type fills the screen: Bungee (Blackout's tiny counters closed up on the grid), three lines per phrase, each line justified to the frame width
  (4% margins), the block scaled vertically (one factor per phrase, 1.07-1.22) to fill 86% of the height.
- Gathered particles sit on a uniform square grid (pitch PITCH) inside the letter mask, drawn as crisp
  marks, a fixed per-particle mix of 50% "+", 25% "|", 25% "-" (the bars are one + arm long); "+" marks (two hairline bars, 0.65 * pitch) with a small living hover (15% of pitch, 10% in the
  read windows). Counts differ per phrase; the pool holds the largest count and the spare particles
  are invisible "ghosts" that ride with a buddy particle and fade in / out mid-transition.
- Slow resolve: each patch departs in peel order and arrives (reaches 90%) at its own time, spread
  over 1.9 s; the flight time per particle is solved so the soft arrival spring (zeta 0.9, ~1.1 s
  settle) hits that time. On arrival the trail retracts and the + arms open on a spring (~0.5 s).
  Gathering takes ~2.6 s per phrase and overlaps the end of the ~3 s thread transition.

No credit: P3 holds through its read window to the end.

TIMING (96 bpm, beat = 0.625 s). readable = ~70% of + marks open (chord change); gathered = every
mark within 3 px (the hover) of its slot with both arms open; read window = gathered -> next peel.
  event                          t (s)          beat
  P1 patches reach 90% (from L)  0.05-2.05     (+ opens ~0.37 s after each)
  P1 readable / chord 1          2.34375        3.75
  P1 gathered                    ~3.05          5
  P1 read window                 3.05-4.69      (1.64 s)
  P1 -> P2 peel                  4.6875-5.99    7.5    peel spread 1.2 s
  camera pan right               5.94-8.49      power2 in-out, 2.55 s (median swarm leads by ~0.2 s)
  P2 patches reach 90%           6.24-8.24
  P2 readable / chord 2          8.59375        13.75
  P2 gathered                    ~9.24          14.75
  P2 read window                 9.24-10.94     (1.70 s)
  P2 -> P3 peel                  10.9375-12.24  17.5
  camera pan down                12.19-14.74
  P3 patches reach 90%           12.49-14.49
  P3 readable / chord 3          14.84375       23.75
  P3 gathered                    ~15.49         24.75
  P3 read window                 15.49-18.2     (2.71 s)

Outputs are numbered generations (gens.py): v002/local/A3b_follow_gNN_draft.mp4, _final.mp4,
chords_A3b_gNN.wav, _final_chords.mp4; lineage rows in v002/local/GENERATIONS.md; two stills per gen.
    python tools/mathtype/follow_plus.py --draft --trigger "user: ..." --change "..."
    python tools/mathtype/follow_plus.py --final        # same code -> same gNN; + chords.py + mux
    python tools/mathtype/follow_plus.py --still 2 5    # ad-hoc PNG stills (draft size) -> v002/review/
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
PREVIEW = os.environ.get("MT_PREVIEW") == "1"     # --preview: trail samples x0.5, 1 subframe, 1x supersample (env so pool workers inherit)
sys.path.insert(0, HERE)
from particles import BG, DH, DW, INK, load_font  # noqa: E402
from follow import ease2, smooth, kd_patches  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
VDIR = os.path.join(REPO, "11_mathtype", "v002")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

FPS, DUR = 25, 18.2
NFR = int(round(FPS * DUR))
BEAT = 0.625

# ------------------------------------------------------------------------------------------ timing
GATHERED = [5 * BEAT, 14.75 * BEAT, 24.75 * BEAT]     # 3.125, 9.21875, 15.46875: all marks in + open
READABLE = [3.75 * BEAT, 13.75 * BEAT, 23.75 * BEAT]  # 2.34, 8.59, 14.84: ~70% of + open (chords)
PEEL = [7.5 * BEAT, 17.5 * BEAT]                      # 4.6875, 10.9375
ARR0 = [0.05, PEEL[0] + 1.55, PEEL[1] + 1.55]         # first patch reaches 90% (+ opens ~0.37 s later)
ARR_SPREAD = 1.9                                      # patch arrivals spread (gathering ~2.5 s with settle)
SPREAD, STAG = 1.2, 0.1                               # patch peel spread, extra stagger inside a patch
SPREAD0, LEAD0 = 1.0, 1.4                             # P1 build: departure spread, first flight to arrival
CAM_LAG, CAM_DUR = 1.25, 2.55                         # fitted: ~0.2 s behind the swarm's median progress

# ------------------------------------------------------------------------------------------ world
FONT = "Bungee"                                    # Blackout closed its O counters at this pitch
PHRASES = [("THERE", "IS", "GEOMETRY"), ("IN", "THE", "HUMMING"), ("OF", "THE", "STRINGS")]
MARGIN = 0.04                                      # side margins (fraction of width)
BLOCK_H = 0.86                                     # block height (fraction of frame height)
LEAD_GAP = 0.03                                    # gap between lines (fraction of frame height)
TRACK = 0.03                                       # letter spacing (em)
PITCH = 20.0                                       # grid pitch (design px)
PLUS = 0.65                                        # "+" size as a fraction of the pitch
LINE_PX = 1.0                                      # one line width (output px) for trails and + bars
ORIGIN = np.array([[0.0, 0.0], [DW, 0.0], [DW, DH]])
DIRS = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
CY = DH / 2
K = 18
SEED = 23

# ------------------------------------------------------------------------------------------ motion
HOVER_BASE, HOVER_READ = 0.15, 0.10                # fraction of the pitch
TAN0, TAN1 = 1.0, 0.7                              # Hermite tangent lengths (x path length)
TRAIL_M, TRAIL_DT = 20, 0.022
TRAIL_FADE = 2.2                                   # brightness along the trail = (1 - age)^TRAIL_FADE
ARR_W, ARR_Z = 4.4, 0.9                            # arrival spring (rad/s, zeta): settles in ~1.1 s
XA, GA = 0.45, 0.40                                # peel phase: to 45% of the flight time, 40% of the way
OPEN_W, OPEN_Z = 7.0, 0.7                          # arm lengths open on this spring (~0.5 s) at g = 0.9
ROT_W, ROT_Z = 6.0, 0.6                            # arm 1 turns from the approach angle to the nearest axis
OPEN_G = 0.985                                     # flight progress at which the + starts to open
MARK_MIX = (0.5, 0.25, 0.25)                       # share of '+', '|', '-' marks (fixed per particle)
ARM2_DELAY = 0.2                                   # arm 2 grows out of the crossing point this much later
NSUB, SHUTTER = 5, 0.5


def ss_for(W):
    return 1 if PREVIEW else max(1, int(round(2 * DW / W)))


def trail_params():
    """(samples, spacing): preview halves the sample count over the same trail length."""
    return (TRAIL_M // 2, TRAIL_DT * 2) if PREVIEW else (TRAIL_M, TRAIL_DT)


def line_w(f, s: str) -> float:
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def phrase_mask(lines):
    """Lines justified to the frame width (minus margins); block squashed vertically to BLOCK_H."""
    W = (1 - 2 * MARGIN) * DW
    rows = []
    for s in lines:
        probe = load_font(FONT, 200)
        f = load_font(FONT, int(200 * W / line_w(probe, s)))
        hb = f.getbbox("H")
        cap = hb[3] - hb[1]
        im = Image.new("L", (DW, cap + 4), 0)
        d = ImageDraw.Draw(im)
        x = (DW - line_w(f, s)) / 2 - f.getbbox(s[0])[0]
        for ch in s:
            d.text((x, 2 - hb[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        rows.append(np.array(im))
    gap = LEAD_GAP * DH
    Ht = BLOCK_H * DH
    sy = (Ht - gap * (len(rows) - 1)) / sum(r.shape[0] for r in rows)
    mask = np.zeros((DH, DW), np.uint8)
    y = (DH - Ht) / 2
    for r in rows:
        h = int(round(r.shape[0] * sy))
        rr = cv2.resize(r, (DW, h), interpolation=cv2.INTER_AREA)
        y0 = int(round(y))
        mask[y0:y0 + h] = np.maximum(mask[y0:y0 + h], rr)
        y += h + gap
    return mask, dict(sy=sy, bottom=(DH + Ht) / 2, right=DW * (1 - MARGIN))


def grid_points(mask):
    ys, xs = np.mgrid[PITCH / 2:DH:PITCH, PITCH / 2:DW:PITCH]
    sel = mask[ys.astype(int), xs.astype(int)] > 127
    return np.stack([xs[sel], ys[sel]], 1).astype(float)


def flight_soft(x, f):
    """Constant-acceleration peel to x=XA, then a soft spring (zeta ARR_Z, ARR_W rad/s in real time)
    from the matching position and velocity. x = (t - d) / f, f per particle."""
    a = 2 * GA / XA ** 2
    v0, e0 = a * XA, 1 - GA
    wn = ARR_W * f
    wd = wn * np.sqrt(1 - ARR_Z ** 2)
    tau = np.maximum(x - XA, 0.0)
    e = np.exp(-ARR_Z * wn * tau) * (e0 * np.cos(wd * tau) + ((-v0 + ARR_Z * wn * e0) / wd) * np.sin(wd * tau))
    g = np.where(x < XA, 0.5 * a * np.clip(x, 0, None) ** 2, 1 - e)
    return np.where(x > 6.0, 1.0, g)


def _t_table(level):
    """Real time from departure to g = level as a function of the flight scale f (monotonic)."""
    fs = np.linspace(0.5, 7.0, 300)
    x = np.linspace(0, 4, 8001)
    return fs, np.array([x[np.argmax(flight_soft(x, f) >= level)] * f for f in fs])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _rot(v, a):
    c, s_ = np.cos(a), np.sin(a)
    return np.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1]], 1)


def spring(tau, w, z):
    tau = np.maximum(tau, 0.0)
    wd = w * np.sqrt(1 - z * z)
    return 1 - np.exp(-z * w * tau) * (np.cos(wd * tau) + (z * w / wd) * np.sin(wd * tau))


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        pts, lab, self.met = [], [], []
        for lines in PHRASES:
            mask, met = phrase_mask(lines)
            g = grid_points(mask)
            pts.append(g)
            lab.append(kd_patches(g, K))
            self.met.append(met)
        self.counts = [len(g) for g in pts]
        N = self.N = max(self.counts)
        self.order = rng.permutation(N)

        def centroids(k):
            return np.array([pts[k][lab[k] == j].mean(0) for j in range(K)])

        # per phrase k and pool particle i: slot index into pts[k], visibility, patch label
        slot = [np.zeros(N, int) for _ in range(3)]
        vis = [np.zeros(N) for _ in range(3)]
        n0 = self.counts[0]
        slot[0][:n0] = np.arange(n0)
        vis[0][:n0] = 1
        slot[0][n0:] = slot[0][rng.integers(0, n0, N - n0)]
        c0 = centroids(0)
        key = 0.6 * (1 - (c0[:, 0] - c0[:, 0].min()) / np.ptp(c0[:, 0])) + rng.random(K)
        rank0 = np.argsort(np.argsort(key))
        dep_order = []                                         # per move: real particles in departure order
        for m, dvec in enumerate(DIRS):
            a, b = m, m + 1
            ca, cb = centroids(a), centroids(b)
            across = np.array([dvec[1], dvec[0]])
            # arrival order: seeded random. Departing patches pair with the arrival slot whose patch is
            # closest to their own position carried by one frame along the travel axis, so every stream
            # travels about the same distance, in parallel (no leap-frogging spikes).
            rb = np.argsort(np.argsort(rng.random(K)))
            slot_patch_b = np.argsort(rb)
            # (centroids are in phrase-local frame coords, so "carried one frame along" = same local spot)
            cost = np.linalg.norm(cb[slot_patch_b][None, :, :] - ca[:, None, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            ra = np.empty(K, int)
            ra[ri] = ci
            real = np.nonzero(vis[a] > 0)[0]
            la = lab[a][slot[a][real]]
            R = real[np.lexsort((pts[a][slot[a][real]] @ across, ra[la]))]   # departure order
            S = np.lexsort((pts[b] @ across, rb[lab[b]]))                   # phrase-b slots, arrival order
            na, nb = len(R), len(S)
            ghosts = list(np.nonzero(vis[a] == 0)[0])
            if nb >= na:
                pick = np.round(np.arange(na) * (nb - 1) / max(na - 1, 1)).astype(int)
                taken = np.zeros(nb, bool)
                taken[pick] = True
                slot[b][R] = S[pick]
                vis[b][R] = 1
                for j in np.nonzero(~taken)[0]:                  # ghosts spawn from the stream
                    gi = ghosts.pop()
                    buddy = R[int(round(j * (na - 1) / max(nb - 1, 1)))]
                    slot[a][gi] = slot[a][buddy]
                    slot[b][gi] = S[j]
                    vis[b][gi] = 1
                    dep_order.append(None)
            else:
                pick = np.round(np.arange(nb) * (na - 1) / max(nb - 1, 1)).astype(int)
                slot[b][R[pick]] = S
                vis[b][R[pick]] = 1
                left = np.setdiff1d(np.arange(na), pick)
                for q in left:                                   # extras fade into the stream
                    near = pick[np.argmin(np.abs(pick - q))]
                    slot[b][R[q]] = slot[b][R[near]]
            for gi in ghosts:                                    # idle ghosts shadow a random real one
                buddy = R[rng.integers(0, na)]
                slot[a][gi] = slot[a][buddy]
                rbuddy = np.nonzero(vis[b] > 0)[0]
                slot[b][gi] = slot[b][rbuddy[rng.integers(0, len(rbuddy))]]
            dep_order.append((ra, across))
        self.loc = [pts[k][slot[k]] for k in range(3)]
        self.vis = vis
        self.world = [self.loc[k] + ORIGIN[k] for k in range(3)]
        labp = [lab[k][slot[k]] for k in range(3)]               # patch of each particle per phrase
        self.labp = labp

        # ---- schedules: departure d, flight scale f, opening time per particle --------------------
        # Each patch has a departure time (peel order) and an arrival time (same rank, spread over
        # ARR_SPREAD); f is solved so the soft flight reaches g = 0.9 exactly at the arrival time.
        fs, t90 = _t_table(0.9)
        _, t_open = _t_table(OPEN_G)
        stag = rng.random((3, N)) * STAG
        stag_a = rng.random((3, N)) * STAG
        self.moves = []
        arr = ARR0[0] + rank0[labp[0]] * ARR_SPREAD / (K - 1) + stag_a[0]
        d0 = ARR0[0] - LEAD0 + rank0[labp[0]] * SPREAD0 / (K - 1) + stag[0]
        src = self.world[0].copy()
        src[:, 0] -= 1250 + 260 * rng.random(N)
        src[:, 1] = CY + (src[:, 1] - CY) * 0.55 + 90 * (rng.random(N) - 0.5)
        self.src = src
        self.moves.append(dict(d=d0, arr=arr))
        ras = [o for o in dep_order if o is not None]
        for m in range(2):
            ra, _ = ras[m]
            d = PEEL[m] + ra[labp[m]] * SPREAD / (K - 1) + stag[m + 1]
            self.moves.append(dict(d=d, arr=None))
        # make ghosts depart exactly with their buddies (same slot at phrase m -> same patch; match stagger)
        for m in range(2):
            key_ = slot[m] * 1.0
            _, first = np.unique(key_, return_index=True)
            lead_of = {s_: i for s_, i in zip(np.unique(key_), first)}
            real = vis[m] > 0
            owner = {}
            for i in np.nonzero(real)[0]:
                owner.setdefault(slot[m][i], i)
            dd = self.moves[m + 1]["d"]
            for i in np.nonzero(~real)[0]:
                o = owner.get(slot[m][i], lead_of[slot[m][i]])
                dd[i] = dd[o]
                stag_a[m + 1][i] = stag_a[m + 1][o]

        for m in range(2):                                     # arrival rank = departure rank
            ra, _ = ras[m]
            self.moves[m + 1]["arr"] = (ARR0[m + 1] + ra[labp[m]] * ARR_SPREAD / (K - 1)
                                        + stag_a[m + 1])
        for mv in self.moves:
            mv["f"] = np.interp(mv["arr"] - mv["d"], t90, fs)
            # the + opens only once the particle has all but stopped (g = OPEN_G), so the last stretch
            # of its own trail is still there to become arm 1 while the rest retracts into the head
            mv["open"] = mv["d"] + np.interp(mv["f"], fs, t_open)

        # ---- flight paths: cubic Hermite arcs, tangents from a broad curl-noise flow ----------------
        # psi = sum of two sines with wavelengths of 2-3 frames (one or two broad eddies); the start
        # tangent follows the flow at the start (mixed with the travel direction so nothing loops back),
        # the end tangent is a gentle approach along the travel direction. Per-patch spread only.
        ang = rng.random(2) * 2 * np.pi
        lam = np.array([2200.0, 3000.0]) * (0.85 + 0.3 * rng.random(2))
        self.flow = dict(k=np.stack([np.cos(ang), np.sin(ang)], 1) * (2 * np.pi / lam)[:, None],
                         ph=rng.random(2) * 2 * np.pi)
        jit = rng.random((3, K, 3)) * 2 - 1
        starts = [self.src, self.world[0], self.world[1]]
        travel = [np.array([1.0, 0.0])] + DIRS
        self.tan = []
        for k in range(3):
            A_, B_ = starts[k], self.world[k]
            L = np.linalg.norm(B_ - A_, axis=1, keepdims=True)
            jb = jit[k][labp[max(k - 1, 0)]]
            t0 = _rot(_unit(0.55 * travel[k] + 0.45 * self.flow_dir(A_)), 0.35 * jb[:, 0])
            t1 = _rot(_unit(0.75 * travel[k] + 0.25 * self.flow_dir(B_)), 0.30 * jb[:, 1])
            self.tan.append((t0 * L * TAN0 * (1 + 0.15 * jb[:, 2:3]), t1 * L * TAN1))

        # ---- hover ----------------------------------------------------------------------------------
        self.waves = []
        for k in range(3):
            ang = rng.random(4) * 2 * np.pi
            lam = 140 + 160 * rng.random(4)
            kv = np.stack([np.cos(ang), np.sin(ang)], 1) * (2 * np.pi / lam)[:, None]
            ev = rng.normal(size=(4, 2))
            ev /= np.linalg.norm(ev, axis=1, keepdims=True)
            self.waves.append(dict(k=kv, e=ev, nu=(0.6 + 0.6 * rng.random(4)) * rng.choice([-1, 1], 4),
                                   psi=rng.random(4) * 2 * np.pi))
        self.orb_w = 0.8 + 1.0 * rng.random((N, 2))
        self.orb_ph = rng.random((N, 2)) * 2 * np.pi
        # approach / departure angles: end tangent of each move, start tangent of each move
        self.phi_in = [np.arctan2(T1[:, 1], T1[:, 0]) for _, T1 in self.tan]
        self.phi_out = [np.arctan2(T0[:, 1], T0[:, 0]) for T0, _ in self.tan]
        # mark type per particle from its own seeded hash (stable over time): 0 '+', 1 '|', 2 '-'
        hsh = np.random.default_rng(SEED + 101).random(N)
        self.mtype = np.searchsorted(np.cumsum(MARK_MIX)[:-1], hsh, side="right")

    # ------------------------------------------------------------------------------------ state
    def flow_dir(self, q):
        """Unit velocity of the broad, static curl-noise flow at points q (N,2)."""
        w = self.flow
        kn = w["k"] / np.linalg.norm(w["k"], axis=1, keepdims=True)
        v = np.cos(q @ w["k"].T + w["ph"]) @ np.stack([kn[:, 1], -kn[:, 0]], 1)
        return _unit(v)

    def read_env(self, k, t):
        up = smooth((t - GATHERED[k] + 0.25) / 0.5)
        down = 1.0 if k == 2 else 1 - smooth((t - PEEL[k] + 0.3) / 0.3)
        return float(up * down)

    def hover(self, k, t):
        A = PITCH * (HOVER_BASE - (HOVER_BASE - HOVER_READ) * self.read_env(k, t))
        w = self.waves[k]
        ph = self.loc[k] @ w["k"].T + w["nu"] * t + w["psi"]
        field = np.sin(ph) @ w["e"] / np.sqrt(2.0)
        orb = np.sin(self.orb_w * t + self.orb_ph)
        return A * (0.6 * field + 0.6 * orb)

    def cam(self, t):
        c = ORIGIN[0].copy()
        for m in range(2):
            c = c + (ORIGIN[m + 1] - ORIGIN[m]) * float(ease2((t - PEEL[m] - CAM_LAG) / CAM_DUR))
        return c

    def pos(self, t):
        """World pos (N,2), thread factor, shade, + arm factor, visibility."""
        N = self.N
        homes = [self.world[k] + self.hover(k, t) for k in range(3)]
        p = self.src.copy()
        thread = np.zeros(N)
        shade = np.ones(N)
        arm = np.zeros(N)
        vis = self.vis[0].copy()
        prev = self.src
        for k, mv in enumerate(self.moves):
            x = (t - mv["d"]) / mv["f"]
            g = flight_soft(x, mv["f"])
            thread = np.maximum(thread, smooth(x / 0.12) * (1 - smooth((g - 0.78) / 0.14)))
            arm = arm * (1 - smooth(x / 0.1)) + spring(t - mv["arr"], OPEN_W, OPEN_Z)
            gb = np.clip(g, 0, 1)
            if k > 0:
                vis = vis + (self.vis[k] - self.vis[k - 1]) * smooth((gb - 0.3) / 0.4)
            T0, T1 = self.tan[k]
            g2, g3 = g * g, g * g * g
            hp = ((2 * g3 - 3 * g2 + 1)[:, None] * prev + (g3 - 2 * g2 + g)[:, None] * T0
                  + (-2 * g3 + 3 * g2)[:, None] * homes[k] + (g3 - g2)[:, None] * T1)
            p = p + hp - prev
            prev = homes[k]
        return p, thread, shade, np.clip(arm, 0, 1.6), np.clip(vis, 0, 1)

    def moving(self, t):
        for mv in self.moves:
            x = (t - mv["d"]) / mv["f"]
            if np.any((x > -0.05) & (t < mv["open"] + 0.8)):
                return True
        return any(0 < (t - PEEL[m] - CAM_LAG) / CAM_DUR < 1 for m in range(2))

    # ------------------------------------------------------------------------------------ drawing
    def arms(self, t):
        """Arm 1 angle, arm 1 / arm 2 length factors. Arrival: arm 1 extends along the approach
        direction (it is the last stretch of the trail), turns on a spring to the nearest axis, arm 2
        grows out of the crossing. Departure: arm 2 retracts, arm 1 turns to the departure direction
        and shrinks into the head as the trail stretches out behind it."""
        N = self.N
        th1, a1, a2 = np.zeros(N), np.zeros(N), np.zeros(N)
        for k, mv in enumerate(self.moves):
            ta = t - mv["open"]
            phi = self.phi_in[k]
            # '+': arm 1 turns to the nearest axis; '|' to vertical; '-' to horizontal (lines: mod 180)
            to_v = np.mod(np.pi / 2 - phi + np.pi / 2, np.pi) - np.pi / 2
            to_h = np.mod(-phi + np.pi / 2, np.pi) - np.pi / 2
            dl = np.select([self.mtype == 1, self.mtype == 2],
                           [to_v, to_h], np.mod(-phi + np.pi / 4, np.pi / 2) - np.pi / 4)
            arrived = ta >= 0
            th1 = np.where(arrived, phi + dl * spring(ta, ROT_W, ROT_Z), th1)
            a1 = np.where(arrived, spring(ta, OPEN_W, OPEN_Z), a1)
            a2 = np.where(arrived, spring(ta - ARM2_DELAY, OPEN_W, OPEN_Z), a2)
            if k + 1 < len(self.moves):
                td = t - self.moves[k + 1]["d"]
                dep = td >= 0
                ax = phi + dl
                dd = np.mod(self.phi_out[k + 1] - ax + np.pi / 2, np.pi) - np.pi / 2
                th1 = np.where(dep, ax + dd * smooth(td / 0.35), th1)
                a1 = np.where(dep, 1 - smooth((td - 0.15) / 0.45), a1)
                a2 = np.where(dep, 1 - smooth(td / 0.25), a2)
        a2 = np.where(self.mtype == 0, a2, 0.0)                     # single-bar marks have no arm 2
        return th1, np.clip(a1, 0, 1.6), np.clip(a2, 0, 1.6)

    def trails(self, t, W, H):
        ss = ss_for(W)
        sc = W / DW * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        TM, TDT = trail_params()
        c = self.cam(t)
        p, th, shade, _, vis = self.pos(t)
        img = np.empty((Hc, Wc, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        idx = np.nonzero(vis > 0.01)[0]                       # every particle carries its own trail
        if idx.size:
            samp = [p[idx]] + [self.pos(t - j * TDT)[0][idx] for j in range(1, TM + 1)]
            q = np.round((np.stack(samp, 0) - c) * sc * Sh).astype(np.int32)
            layer = np.zeros((Hc, Wc), np.uint8)
            thick = max(1, int(round(LINE_PX * ss)))
            nb = 12
            for j in range(TM - 1, -1, -1):
                fade = (1 - j / TM) ** TRAIL_FADE
                a = shade[idx] * vis[idx] * fade                  # length comes from speed alone
                lev = np.ceil(a * nb).astype(int)
                seg = np.stack([q[j], q[j + 1]], 1)
                for L in range(1, nb + 1):
                    sel = lev == L
                    if sel.any():
                        cv2.polylines(layer, list(seg[sel]), False, int(255 * L / nb), thick, cv2.LINE_AA, 4)
            lf = layer.astype(np.float32)[..., None] / 255.0
            img = (BG + (INK - BG) * lf).astype(np.uint8)
        return img

    def draw(self, t, W, H, base):
        ss = ss_for(W)
        osc = W / DW
        sc = osc * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        c = self.cam(t)
        p, th, shade, _, vis = self.pos(t)
        img = base.copy()
        xy = (p - c) * sc
        m = 40 * sc
        ok = (xy[:, 0] > -m) & (xy[:, 0] < Wc + m) & (xy[:, 1] > -m) & (xy[:, 1] < Hc + m) & (vis > 0.01)
        cq = np.round(xy * Sh).astype(np.int64)
        th1, a1, a2 = self.arms(t)
        h = 0.5 * PLUS * PITCH * sc * Sh
        e1 = np.stack([np.cos(th1), np.sin(th1)], 1)
        e2 = np.stack([-e1[:, 1], e1[:, 0]], 1)
        d1 = np.round(e1 * (h * a1)[:, None]).astype(np.int64)
        d2 = np.round(e2 * (h * a2)[:, None]).astype(np.int64)
        bar = max(1, int(round(LINE_PX * ss)))
        col = BG + (INK - BG) * (shade * vis)[:, None]
        for i in self.order:
            if not ok[i]:
                continue
            x0, y0 = int(cq[i, 0]), int(cq[i, 1])
            cl = tuple(float(v) for v in col[i])
            if a1[i] > 0.02:
                cv2.line(img, (x0 - int(d1[i, 0]), y0 - int(d1[i, 1])), (x0 + int(d1[i, 0]), y0 + int(d1[i, 1])),
                         cl, bar, cv2.LINE_AA, 4)
            if a2[i] > 0.02:
                cv2.line(img, (x0 - int(d2[i, 0]), y0 - int(d2[i, 1])), (x0 + int(d2[i, 0]), y0 + int(d2[i, 1])),
                         cl, bar, cv2.LINE_AA, 4)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def frame(self, t, W, H):
        base = self.trails(t, W, H)
        if self.moving(t) and not PREVIEW:
            dts = (np.arange(NSUB) / (NSUB - 1) - 0.5) * SHUTTER / FPS
            acc = np.zeros((H, W, 3), np.float32)
            for d in dts:
                acc += self.draw(t + d, W, H, base)
            img = np.clip(acc / NSUB + 0.5, 0, 255).astype(np.uint8)
        else:
            img = self.draw(t, W, H, base)
        return img


# ---------------------------------------------------------------------------------------------- output
_SC = None


def _init(cls):
    global _SC
    _SC = cls()


def _work(args):
    n, W, H = args
    return _SC.frame(n / FPS, W, H).tobytes()


def render(path, W, H, cls=None, nfr=None, procs=8):
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")
    cls = cls or Scene
    nfr = nfr or NFR
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with mp.Pool(procs, initializer=_init, initargs=(cls,)) as pool:
        for buf in pool.imap(_work, [(n, W, H) for n in range(nfr)], chunksize=2):
            proc.stdin.write(buf)
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"{path}  {W}x{H}  {nfr} frames  {dt:.1f}s")
    return dt


def cli(test, stem, cls, still_times, readable, dur, code_files, extra_info=None):
    """Shared CLI: numbered generations (gens.py), stills, chords, mux. Never overwrites."""
    import gens
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true", help="final render + chords + mux")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--preview", action="store_true", help="fast CPU test: 360x450, trail samples x0.5, 1 subframe, 1x supersample -> _preview.mp4")
    ap.add_argument("--still", type=float, nargs="*", help="ad-hoc PNG stills (draft size) to review/")
    ap.add_argument("--trigger", default="", help="lineage: user: ... / self-QC: ...")
    ap.add_argument("--change", default="", help="lineage: what changed (<=15 words)")
    ap.add_argument("--parent", default=None, help="lineage: e.g. 'A3b g02' (default: previous gen)")
    a = ap.parse_args()
    global PREVIEW
    if a.preview:
        PREVIEW = True
        os.environ["MT_PREVIEW"] = "1"
    nfr = int(round(FPS * dur))
    if a.info and extra_info:
        extra_info()
    h = gens.code_hash(code_files)
    if a.preview:
        nn = gens.gen_for(LOCAL, test, stem, "preview", h + "-preview", a.parent, a.trigger or "user: fast CPU preview mode",
                          a.change or "--preview: trail samples x0.5, 1 subframe, 1x supersample")
        name = gens.video_name(stem, nn, "preview")
        render(os.path.join(LOCAL, name), 360, 450, cls, nfr)
        gens.add_files(LOCAL, test, nn, [name])
    if a.draft:
        nn = gens.gen_for(LOCAL, test, stem, "draft", h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, "draft")
        render(os.path.join(LOCAL, name), 360, 450, cls, nfr)
        sc = cls()
        stills = []
        for t in still_times:
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            Image.fromarray(sc.frame(t, 540, 675)).save(os.path.join(REVIEW, fn), quality=90)
            stills.append(fn)
        gens.add_files(LOCAL, test, nn, [name], stills)
    if a.final:
        nn = gens.gen_for(LOCAL, test, stem, "final", h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, "final")
        render(os.path.join(LOCAL, name), 1080, 1350, cls, nfr)
        wav = f"chords_{test}_g{nn:02d}.wav"
        files = [name]
        if os.path.exists(os.path.join(HERE, "chords.py")) and not os.path.exists(os.path.join(LOCAL, wav)):
            subprocess.run([sys.executable, os.path.join(HERE, "chords.py"), "--duration", f"{dur:g}",
                            "--changes", ",".join(f"{x:g}" for x in readable),
                            "--out", os.path.join(LOCAL, wav)], check=True)
        if os.path.exists(os.path.join(LOCAL, wav)):
            mux = gens.video_name(stem, nn, "final", chords=True)
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", os.path.join(LOCAL, name),
                            "-i", os.path.join(LOCAL, wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", os.path.join(LOCAL, mux)], check=True)
            files += [wav, mux]
        gens.add_files(LOCAL, test, nn, files)
    if a.still:
        os.makedirs(REVIEW, exist_ok=True)
        sc = cls()
        for t in a.still:
            Image.fromarray(sc.frame(t, 360, 450)).save(os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))


def main():
    def info():
        sc = Scene()
        print("pitch", PITCH, "counts", sc.counts, "pool", sc.N, "squash", [round(m["sy"], 3) for m in sc.met])
    cli("A3b", "A3b_follow", Scene, [PEEL[0] + 2.0, GATHERED[1] + 0.6], READABLE, DUR,
        [os.path.abspath(__file__), os.path.join(HERE, "follow.py"), os.path.join(HERE, "particles.py")],
        info)


if __name__ == "__main__":
    main()
