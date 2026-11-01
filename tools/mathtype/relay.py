#!/usr/bin/env python
"""Math-type test A2: edge relay (18 s, 4:5, 25 fps).

One swarm of N particles relays the three phrases across the frame edges. No cards or planes are
drawn: the relay lives in the paths. P1 leaves through the right edge and the same particles come
back in from the left to fill P2 (horizontal wrap). P2 leaves through the top and comes back from
the bottom to fill P3 (vertical wrap). Each move has its own path family, closed-form in t:

  build  rose curve r = R cos(5/2 th), particles flowing along it as silk, peeling off in patches
  T1     logarithmic-spiral vortex about a centre that travels right (one full turn)
  T2     harmonograph ribbons (decaying sums of sines, phase-coherent across the word) travelling
         up, with the swarm tilting the other way

On departure a bubble thins to a hairline and grows a long fading trail (16 earlier samples of its
own path); on arrival the trail retracts and it swells back. Phrases never form crisply: every
particle hovers around its target in the letter interior (curl + compressive waves + Lissajous),
quieter in each read window. Progress per particle: power2 ease-in travel, then a soft zeta 0.9
damped-spring landing (~1 s settle) entered with matched velocity. Patches (k-means) land in seeded
random order spread over ~2.4 s, so each word gathers slowly (~3 s with the settle).

Timings (96 bpm grid, beat = 0.625 s):

    t (s)    beat   event
    0.00      0     rose silk drifting; P1 patches peel off and land 0.60-3.00 (1 s settle each)
    3.125     5     P1 gathered -> read window 3.125-4.75  (chord change 1)
    4.75            T1 departures (rightmost first, over 0.6 s): spiral vortex exits right
    6.65-9.00       P2 patches land, wrapped in from the left
    9.0625   14.5   P2 gathered -> read window 9.06-10.7  (chord change 2)
   10.70            T2 departures (top first, over 0.6 s): harmonograph ribbons exit top
   12.90-15.25      P3 patches land, wrapped in from the bottom
   15.3125   24.5   P3 gathered -> holds through its read window 15.31-18.0  (chord change 3)
   18.00     28.8   end

Every frame is a pure function of t; nothing is carried between frames.

Outputs follow the naming rule: every generation gets a fresh gNN (glob max + 1, never overwrite).

    python tools/mathtype/relay.py --draft                  # new gNN: A2_relay_gNN_draft.mp4 + 2 stills
    python tools/mathtype/relay.py --final --gen NN         # A2_relay_gNN_final.mp4 (same code as the draft)
    python tools/mathtype/relay.py --chords --gen NN        # chords_A2_relay_gNN.wav + ..._final_chords.mp4
    python tools/mathtype/relay.py --log --gen NN --parent "A2_relay g00" --trigger "..." --change "..."
    python tools/mathtype/relay.py --still 1 3.6            # loose PNG stills (draft size) to v002/review/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import particles as pm  # noqa: E402  (v1: glyph raster, stratified fill, Hilbert order, fonts)

pm.FONT = "Blackout"                  # read by pm.phrase_mask

REPO = pm.REPO
VDIR = os.path.join(REPO, "11_mathtype", "v002")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

DW, DH = pm.DW, pm.DH
FPS, DUR = 25, 18.0
NFR = int(round(FPS * DUR))
BEAT = 0.625
BG, INK, EDGE, ACCENT = pm.BG, pm.INK, pm.EDGE, pm.ACCENT
C0 = np.array([pm.CX, pm.CY])

# ---- timing table (s) ----------------------------------------------------------------------
LAND = [5 * BEAT, 14.5 * BEAT, 24.5 * BEAT]       # 3.125, 9.0625, 15.3125  (chord changes)
ARRIVE = [(0.60, 3.00), (6.65, 9.00), (12.90, 15.25)]  # patch landing windows (+ in-patch stagger)
DEPART = [None, 4.75, 10.70]                       # first departure of T1 / T2 (= read-window end)
DEP_SPREAD = 0.60
READ_END = [DEPART[1], DEPART[2], DUR + 1.0]
FAST = [(-0.1, 3.4), (4.65, 9.4), (10.6, 15.6)]      # windows that get motion blur

PHRASES = [("THERE IS", "GEOMETRY"), ("IN THE", "HUMMING"), ("OF THE", "STRINGS")]
N = 2600
SEED = 11
K_PATCH = 18
STAG = 0.07                       # extra stagger inside a patch (s)
ZETA = 0.9                        # soft landing spring
T_LAND = 1.0                      # landing (settle) phase per particle (s)
F_S = 0.6                         # travel covered before the landing spring takes over
M = 80.0                          # wrap margin (design px) outside the frame
PX, PY = DW + 2 * M, DH + 2 * M   # wrap periods
R_HAIR = 0.9                      # hairline head radius (design px)
TRAIL_K, TRAIL_DT = 16, 0.03      # trail samples / spacing
TRAIL_I = 0.55                    # trail head intensity
NSUB = 4                          # motion-blur subframes
SHUTTER = 0.5 / FPS               # 180 degree shutter
TEST = "A2_relay"
PREVIEW = False                   # --preview: trail samples x0.5, 1 subframe, 1x supersample
GEN_STILLS = (6.0, 10.0)          # one mid-transition (T1) and one read-window (P2) still per generation


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def progress(t, d, a):
    """0 -> 1 along one flight from departure d to arrival a (per particle).

    power2 ease-in travel to F_S, then a soft zeta 0.9 damped spring over the last T_LAND s,
    entered with the matching velocity. Closed form; ~0.98 at a, settling to 1 just after.
    """
    T = np.maximum(a - d, 0.4)
    TL = np.minimum(T_LAND, 0.5 * T)
    T1 = T - TL
    tau = (t - d) / T1
    p1 = F_S * np.clip(tau, 0, 1) ** 2
    w = 4.0 / (ZETA * TL)                           # ~98 % settled at a
    wd = w * np.sqrt(1 - ZETA ** 2)
    v = 2 * F_S / T1 / (1 - F_S)                    # seam velocity, normalised to the remaining travel
    B = (ZETA * w - v) / wd
    x = np.maximum(t - (d + T1), 0)
    p2 = F_S + (1 - F_S) * (1 - np.exp(-ZETA * w * x) * (np.cos(wd * x) + B * np.sin(wd * x)))
    return np.where(tau < 1, p1, p2)


class Phrase:
    def __init__(self, k, rng):
        self.k = k
        mask, met = pm.phrase_mask(PHRASES[k])
        self.met = met
        pts, dist = pm.stratified(mask, N, rng)
        o = pm.hilbert_order(pts)                    # rank i pairs with rank i of the other phrases
        pts, dist = pts[o], dist[o]
        spacing = np.sqrt((mask > 127).sum() / N)
        q = rng.random(N)
        self.r = np.minimum(0.62 * spacing * (0.55 + 0.9 * q ** 1.6), dist + 2.0)
        self.tgt = pts
        self.cap = cap = float(met["cap"])
        # patches: k-means cells over the targets, landing in seeded random order
        _, lab = kmeans2(pts, K_PATCH, seed=int(rng.integers(1 << 30)), minit="++")
        slot = np.empty(K_PATCH)
        slot[rng.permutation(K_PATCH)] = np.linspace(0, 1, K_PATCH)
        a0, a1 = ARRIVE[k]
        pa = a0 + (a1 - a0) * np.clip(slot + rng.uniform(-0.03, 0.03, K_PATCH), 0, 1)
        stag = np.zeros(N)
        idx = np.arange(N)                           # Hilbert rank: ripple along the curve
        for c in range(K_PATCH):
            m = lab == c
            if m.any():
                h = idx[m]
                stag[m] = STAG * (h - h.min()) / max(1, np.ptp(h))
        self.arrive = pa[lab] + stag
        # hover field parameters (bounded, never decays)
        self.kv = rng.uniform(0.012, 0.03, (3, 2)) * rng.choice([-1, 1], (3, 2))
        self.kg = rng.uniform(0.010, 0.024, (2, 2)) * rng.choice([-1, 1], (2, 2))
        self.wv = rng.uniform(0.6, 1.3, 3)
        self.wg = rng.uniform(0.5, 1.0, 2)
        self.pv = rng.uniform(0, 2 * np.pi, 3)
        self.pg = rng.uniform(0, 2 * np.pi, 2)
        out = rng.random(N) < 0.08
        self.la = np.where(out, rng.uniform(0.12, 0.25, N), rng.uniform(0.005, 0.03, N)) * cap
        self.lf = np.where(out[:, None], rng.uniform(0.25, 0.6, (N, 2)), rng.uniform(0.9, 2.2, (N, 2)))
        self.lp = rng.uniform(0, 2 * np.pi, (N, 2))

    def hover(self, t):
        p = self.tgt
        d = np.zeros_like(p)
        for k in range(3):                # curl (divergence-free) swirl, coherent in space
            K = self.kv[k]
            d += np.cos(p @ K + self.wv[k] * t + self.pv[k])[:, None] * (np.array([K[1], -K[0]]) / np.linalg.norm(K))
        d *= 0.075 * self.cap / 3 ** 0.5
        g = np.zeros_like(p)
        for k in range(2):                # compressive waves: gaps open and close
            K = self.kg[k]
            g += np.cos(p @ K + self.wg[k] * t + self.pg[k])[:, None] * (K / np.linalg.norm(K))
        d += g * 0.065 * self.cap / 2 ** 0.5
        d += self.la[:, None] * np.sin(self.lf * t + self.lp)   # Lissajous orbit / outliers
        w = smooth((t - LAND[self.k] + 0.1) / 0.5) * smooth((READ_END[self.k] - t) / 0.3)
        return d * (1 - 0.3 * w)


def polar(v):
    return np.hypot(v[:, 0], v[:, 1]), np.arctan2(v[:, 1], v[:, 0])


class Relay:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.ph = [Phrase(k, rng) for k in range(3)]
        self.metrics = [p.met for p in self.ph]
        self.hash = rng.random(N)
        self.thread = TRAIL_I * (0.3 + 0.7 * rng.random(N) ** 2)   # per-thread brightness
        # build: rose parameter follows the Hilbert rank so patches peel off coherent strands
        self.th0 = 4 * np.pi * np.arange(N) / N + 0.02 * rng.standard_normal(N)
        self.band = 1 + 0.07 * rng.standard_normal(N)
        self.dep0 = self.ph[0].arrive - (1.6 + 0.3 * rng.random(N))
        # T1 departures: rightmost first; T2: topmost first
        x0 = self.ph[0].tgt[:, 0]
        y1 = self.ph[1].tgt[:, 1]
        self.dep1 = DEPART[1] + DEP_SPREAD * (1 - (x0 - x0.min()) / np.ptp(x0)) + 0.03 * rng.random(N)
        self.dep2 = DEPART[2] + DEP_SPREAD * (y1 - y1.min()) / np.ptp(y1) + 0.03 * rng.random(N)
        # harmonograph phase: coherent across the word (ribbons)
        q = self.ph[1].tgt
        self.hphase = 0.0021 * q[:, 0] + 0.0008 * q[:, 1]

    # ---------------------------------------------------------------- closed-form state
    def rose(self, t):
        th = self.th0 + 0.42 * t
        R = 520.0 * (0.7 + 0.3 * smooth(t / 0.9)) * self.band
        rr = R * np.cos(2.5 * th)
        psi = 0.22 * t
        v = np.stack([rr * np.cos(th), 0.92 * rr * np.sin(th)], 1)
        c, s = np.cos(psi), np.sin(psi)
        return C0 + np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)

    def state(self, t):
        """Positions (N,2) in design px, radius (N,), hairness h (N,) in [0, 1]."""
        P0, P1, P2 = self.ph
        e0 = P0.tgt + P0.hover(t)
        e1 = P1.tgt + P1.hover(t)
        e2 = P2.tgt + P2.hover(t)
        # build: rose silk -> P1
        s = progress(t, self.dep0, P0.arrive)
        sc = np.clip(s, 0, 1)
        src = self.rose(t)
        rel = (src - C0) * (1 - s)[:, None] + (e0 - C0) * s[:, None]
        ang = 0.9 * (1 - sc) * sc                  # slight inward swirl as they peel off
        ca, sa = np.cos(ang), np.sin(ang)
        pos = C0 + np.stack([ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]], 1)
        h = smooth((1 - s) / 0.2)
        r = P0.r * (1 - h) + R_HAIR * h
        # T1: log-spiral vortex about a centre travelling right, wrapped in x
        if np.any(t >= self.dep1):
            s1 = progress(t, self.dep1, P1.arrive)
            c1 = np.clip(s1, 0, 1)
            r0, a0 = polar(e0 - C0)
            r1, a1 = polar(e1 - C0)
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi + 2 * np.pi
            lr = (1 - s1) * np.log(np.maximum(r0, 15)) + s1 * np.log(np.maximum(r1, 15))
            rad = np.exp(lr) * (1 + 0.28 * np.sin(np.pi * c1))
            an = a0 + da * s1
            p1 = C0 + np.stack([PX * s1 + rad * np.cos(an), rad * np.sin(an)], 1)
            p1[:, 0] = (p1[:, 0] + M) % PX - M
            h1 = smooth(s1 / 0.1) * smooth((1 - s1) / 0.2)
            rr1 = (P0.r * (1 - c1) + P1.r * c1) * (1 - h1) + R_HAIR * h1
            m = t >= self.dep1
            pos[m], h[m], r[m] = p1[m], h1[m], rr1[m]
        # T2: harmonograph ribbons travelling up, swarm tilting the other way, wrapped in y
        if np.any(t >= self.dep2):
            s2 = progress(t, self.dep2, P2.arrive)
            c2 = np.clip(s2, 0, 1)
            A = e1
            B = e2 - [0.0, PY]
            lin = A + (B - A) * s2[:, None]
            cen = C0 + np.outer(s2, [0.0, -PY])
            rel = lin - cen
            tilt = -0.6 * np.sin(np.pi * c2)
            ct, st = np.cos(tilt), np.sin(tilt)
            rel = np.stack([ct * rel[:, 0] - st * rel[:, 1], st * rel[:, 0] + ct * rel[:, 1]], 1)
            env = np.sin(np.pi * c2)
            ph = self.hphase
            tw = 2 * np.pi * s2
            hx = 240 * np.exp(-0.9 * s2) * np.sin(1.0 * tw + ph) + 60 * np.sin(2.0 * tw + 1.2 * ph + 1.0)
            hy = 150 * np.exp(-0.6 * s2) * np.sin(1.5 * tw + ph + 0.5) + 40 * np.sin(2.5 * tw + 0.8 * ph + 2.0)
            p2 = cen + rel + env[:, None] * np.stack([hx, hy], 1)
            p2[:, 1] = (p2[:, 1] + M) % PY - M
            h2 = smooth(s2 / 0.1) * smooth((1 - s2) / 0.2)
            rr2 = (P1.r * (1 - c2) + P2.r * c2) * (1 - h2) + R_HAIR * h2
            m = t >= self.dep2
            pos[m], h[m], r[m] = p2[m], h2[m], rr2[m]
        r = r * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t + 6.28 * self.hash))
        return pos, r, h

    # ---------------------------------------------------------------- drawing
    def sub(self, t, W, H):
        ss = 1 if PREVIEW else (3 if W < 720 else 2)
        Ws, Hs = W * ss, H * ss
        k = Ws / DW
        S = 16.0
        lw_h = max(1, int(round(ss * W / DW)))      # hairline: 1 output px at 1080, same look in the draft
        TK, TDT = (TRAIL_K // 2, TRAIL_DT * 2) if PREVIEW else (TRAIL_K, TRAIL_DT)
        xy, r, h = self.state(t)
        # trails: earlier samples of each particle's own path, fading along the length,
        # length proportional to hairness (they retract as the particle swells back)
        acc = np.zeros((Hs, Ws), np.float32)
        live = h > 0.02
        if live.any():
            hist = [xy] + [self.state(t - j * TDT)[0] for j in range(1, TK + 1)]
            segs_by_b = {}
            for j in range(TK):
                a = np.clip(TK * h - j, 0, 1) * (1 - j / TK) ** 1.3 * self.thread
                p, q = hist[j], hist[j + 1]
                ok = live & (a > 0.03) & (np.abs(p - q).max(1) < 300)
                if not ok.any():
                    continue
                b = np.minimum((a[ok] * 8).astype(int), 7)
                seg = np.round(np.stack([p[ok], q[ok]], 1) * k * S).astype(np.int32)
                for bb in np.unique(b):
                    segs_by_b.setdefault(bb, []).append(seg[b == bb])
            lay = np.zeros((Hs, Ws), np.uint8)
            for bb, lst in segs_by_b.items():
                lay[:] = 0
                cv2.polylines(lay, list(np.concatenate(lst)), False, 255, lw_h, cv2.LINE_AA, 4)
                acc += lay.astype(np.float32) * ((bb + 0.5) / 8 / 255.0)
            # hairline heads
            hd = live & (h > 0.3)
            lay[:] = 0
            for c in np.round(xy[hd] * k * S).astype(np.int64):
                cv2.circle(lay, (int(c[0]), int(c[1])), int(0.6 * lw_h * S), 255, -1, cv2.LINE_AA, 4)
            acc += lay.astype(np.float32) * (TRAIL_I / 255.0)
        img = (BG + np.clip(acc, 0, 1)[:, :, None] * (INK - BG)).astype(np.uint8)
        # bubbles (v1 look: warm-white fill, thin dark outline)
        lw = max(1.3 * W / DW, 0.45) * ss
        R = r * k
        bub = (h < 0.85) & (R > 1.2 * ss)
        cen = np.round(xy * k * S).astype(np.int64)
        Ro = np.round(R * S).astype(np.int64)
        Ri = np.round(np.maximum(R - lw, 0.3) * S).astype(np.int64)
        fill = tuple(float(v) for v in INK)
        for i in np.nonzero(bub)[0]:
            c = (int(cen[i, 0]), int(cen[i, 1]))
            cv2.circle(img, c, int(Ro[i]), EDGE, -1, cv2.LINE_AA, 4)
            cv2.circle(img, c, int(Ri[i]), fill, -1, cv2.LINE_AA, 4)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

    def frame(self, t, W, H):
        if PREVIEW or not any(a <= t <= b for a, b in FAST):
            return np.clip(self.sub(t, W, H), 0, 255).astype(np.uint8)
        acc = np.zeros((H, W, 3), np.float32)
        for j in range(NSUB):
            acc += self.sub(t + SHUTTER * (j / (NSUB - 1) - 0.5), W, H)
        return np.clip(acc / NSUB + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- output
def render(path, W, H):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc = Relay()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(NFR):
        proc.stdin.write(sc.frame(n / FPS, W, H).tobytes())
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"{path}  {W}x{H}  {NFR} frames  {dt:.1f}s")
    return dt


def gens(out_dir, stem):
    """Generation numbers already used by <stem>_gNN_* files in out_dir."""
    import glob
    import re
    out = set()
    for f in glob.glob(os.path.join(out_dir, f"{stem}_g*")):
        m = re.match(rf"{re.escape(stem)}_g(\d+)_", os.path.basename(f))
        if m:
            out.add(int(m.group(1)))
    return out


def next_gen(out_dir, stem):
    """max + 1 over existing <stem>_gNN_* files (unnumbered legacy files count as g00)."""
    return max(gens(out_dir, stem) | {0}) + 1


def fresh(path):
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}; bump the generation")
    return path


def log_gen(g, parent, trigger, change):
    """Append one lineage row to local/GENERATIONS.md (shared file: append only)."""
    import glob
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Pacific/Auckland"))
    except Exception:
        now = datetime.now(timezone.utc) + timedelta(hours=13)   # NZDT
    tag = f"{TEST}_g{g:02d}"
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(LOCAL, f"*{tag}[_.]*")))
    stills = sorted(os.path.basename(f) for f in glob.glob(os.path.join(REVIEW, f"{tag}_still_*")))
    path = os.path.join(LOCAL, "GENERATIONS.md")
    if not os.path.exists(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("| gen | time NZ | test | parent | trigger | change | files | still |\n"
                     "|---|---|---|---|---|---|---|---|\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"| g{g:02d} | {now:%H:%M} | {TEST} | {parent} | {trigger} | {change} | "
                 f"{', '.join(files)} | {', '.join(stills)} |\n")
    print("logged", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--preview", action="store_true", help="fast CPU test: 360x450, trail samples x0.5, 1 subframe, 1x supersample -> _preview.mp4")
    ap.add_argument("--chords", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--gen", type=int, default=None, help="generation (default for --draft: next free)")
    ap.add_argument("--parent", default="")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--still", type=float, nargs="*")
    a = ap.parse_args()
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    g = a.gen
    if a.preview:
        global PREVIEW
        PREVIEW = True
        parent = max(gens(LOCAL, TEST) | {0})
        pg = next_gen(LOCAL, TEST)
        print(f"generation g{pg:02d} (preview)")
        render(fresh(os.path.join(LOCAL, f"{TEST}_g{pg:02d}_preview.mp4")), 360, 450)
        log_gen(pg, a.parent or f"{TEST} g{parent:02d}", a.trigger or "user: fast CPU preview mode",
                a.change or "--preview: trail samples x0.5, 1 subframe, 1x supersample")
    if a.draft:
        if g is None:
            g = next_gen(LOCAL, TEST)
        elif g in gens(LOCAL, TEST) and os.path.exists(os.path.join(LOCAL, f"{TEST}_g{g:02d}_draft.mp4")):
            sys.exit(f"g{g:02d} draft exists; omit --gen to take the next one")
        print(f"generation g{g:02d}")
        render(fresh(os.path.join(LOCAL, f"{TEST}_g{g:02d}_draft.mp4")), 360, 450)
        sc = Relay()
        for t in GEN_STILLS:
            Image.fromarray(sc.frame(t, 540, 675)).save(
                fresh(os.path.join(REVIEW, f"{TEST}_g{g:02d}_still_{t:.1f}.jpg")), quality=90)
    if (a.final or a.chords or a.log) and g is None:
        sys.exit("--final/--chords/--log need --gen NN (the draft's generation)")
    if a.final:
        render(fresh(os.path.join(LOCAL, f"{TEST}_g{g:02d}_final.mp4")), 1080, 1350)
    if a.chords:
        cw = fresh(os.path.join(LOCAL, f"chords_{TEST}_g{g:02d}.wav"))
        subprocess.run([sys.executable, os.path.join(HERE, "chords.py"), "--duration", str(DUR),
                        "--changes", ",".join(f"{x:.4f}" for x in LAND), "--out", cw], check=True)
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error",
                        "-i", os.path.join(LOCAL, f"{TEST}_g{g:02d}_final.mp4"), "-i", cw,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        fresh(os.path.join(LOCAL, f"{TEST}_g{g:02d}_final_chords.mp4"))], check=True)
    if a.still:
        sc = Relay()
        for t in a.still:
            Image.fromarray(sc.frame(t, 360, 450)).save(os.path.join(REVIEW, f"A2_still_{t:.2f}.png"))
    if a.log:
        log_gen(g, a.parent, a.trigger, a.change)


if __name__ == "__main__":
    main()
