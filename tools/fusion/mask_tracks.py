"""Point tracks for the Motion-Track IC-LoRA, seeded inside a matte (subject) and steered along big smooth lines.

    python tools/fusion/mask_tracks.py --matte-dir <dir of PNG> --style pull_apart --out tracks.json \
        [--analysis analysis.json --start-s 1.3 --snap beat] [--points 40] [--distance 300]

Output = ltx_ic.build("motiontrack", tracks=...) format: a list of tracks, each a list of {"x","y"} per frame, in the
pixel space of the matte (default 576x1024 = the stage-1 canvas; scale_tracks() for other sizes). Every track has one
point per frame (N frames = matte frames), because LTXVDrawTracks has no per-track start offset.

Behaviour of every style: seed points on the subject at frame 0 (contour samples + farthest-point interior samples);
until the start frame they follow the subject (matte-centroid translation, smoothed); from the start frame each point
leaves along its own eased path (staggered starts, smootherstep, ends at the last frame) and the follow offset is frozen.
Earlier probes (15_fusion/FINDINGS.md): the LoRA only follows big, smooth, coherent motion at ~0.5x speed, so the
default distances are large (about 2x what you want to see) and nothing jitters.

Styles
  pull_apart  left / right halves (split at the subject's centre) stream off in opposite, parallel lines
              (`axis` y: top / bottom halves). Per-point distance varies, clamped so a line never leaves the canvas.
  unravel     each point leaves outward along its contour normal (interior points: away from the centroid) while the
              direction turns (spiral, `spin` radians over the move).
  rain        points drop straight down (slight sway), accelerating (s^1.7), starts spread randomly over `stagger_s`.
  rise        rain mirrored upward: points float straight up (slight sway), accelerating then easing out (smootherstep)
              toward the top, starts spread over `stagger_s`; the threads thin out near the top (late starters stop short of the
              top margin). Safe for people (nothing falls on them).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

STYLES = ("pull_apart", "unravel", "rain", "rise")
DEFAULT_SIZE = (576, 1024)


# --------------------------------------------------------------------------------------------- inputs
def load_matte_seq(directory: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Directory of PNGs (sorted by name) -> (F,H,W) float32 0..1 (first channel). `size` = (w,h) resize check."""
    import cv2
    files = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNG in {directory}")
    out = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise IOError(f)
        if size and (im.shape[1], im.shape[0]) != tuple(size):
            im = cv2.resize(im, tuple(size), interpolation=cv2.INTER_LINEAR)
        out.append(im.astype(np.float32) / 255.0)
    return np.stack(out)


def snap_start(start_s: float, analysis: dict | None, fps: float = 24.0, mode: str = "beat", n_frames: int | None = None):
    """Start time -> (frame, seconds, label). mode: 'beat' | 'downbeat' = nearest such time in analysis.json ('beats' /
    'downbeats', seconds); 'none' = no snapping. The snapped frame stays inside [0, n_frames-2]."""
    t, label = float(start_s), "free"
    if analysis and mode in ("beat", "downbeat"):
        times = [float(x) for x in analysis.get("beats" if mode == "beat" else "downbeats", [])]
        if times:
            t = min(times, key=lambda x: abs(x - start_s))
            label = f"{mode}@{t:.3f}s"
    f = int(round(t * fps))
    if n_frames:
        f = max(0, min(f, n_frames - 2))
    return f, f / fps, label


def smoothstep5(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (s * (s * 6 - 15) + 10)


def _gauss1d(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(a) < 2:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, ((r, r),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    return np.stack([np.convolve(p[:, j], k, mode="valid") for j in range(a.shape[1])], 1)


# --------------------------------------------------------------------------------------------- subject
def _largest_component(binary: np.ndarray) -> np.ndarray:
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def centroid_path(matte: np.ndarray, threshold: float = 0.5, smooth: float = 3.0) -> np.ndarray:
    """(F,H,W) -> (F,2) centroid (x,y) of the largest subject blob per frame, gaussian-smoothed over time."""
    c = np.zeros((matte.shape[0], 2), np.float64)
    last = None
    for i, m in enumerate(matte):
        b = _largest_component(m > threshold)
        ys, xs = np.nonzero(b)
        if len(xs):
            last = (xs.mean(), ys.mean())
        if last is None:
            raise ValueError(f"matte frame {i} is empty and no earlier frame has a subject")
        c[i] = last
    return _gauss1d(c, smooth)


def seed_points(matte0: np.ndarray, n: int = 40, contour_frac: float = 0.55, threshold: float = 0.5, seed: int = 1):
    """Points on the subject of one matte frame. -> (pts (n,2) x,y float, normals (n,2) outward unit, is_contour (n,) bool,
    centroid (2,)). Contour points are spaced evenly by arc length; interior points are farthest-point samples of the
    eroded interior (seeded with the contour so they spread between it)."""
    import cv2
    rng = np.random.default_rng(seed)
    b = _largest_component(matte0 > threshold)
    if b.sum() < 50:
        raise ValueError("matte frame has no subject")
    ys, xs = np.nonzero(b)
    cen = np.array([xs.mean(), ys.mean()])
    cnts, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=len)[:, 0, :].astype(np.float64)
    n_con = int(round(n * contour_frac))
    n_int = n - n_con
    seg = np.linalg.norm(np.diff(np.vstack([cnt, cnt[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    targets = (np.arange(n_con) + rng.random()) * arc[-1] / n_con % arc[-1]
    idx = np.clip(np.searchsorted(arc, targets), 0, len(cnt) - 1)
    con = cnt[idx]
    # outward normal from the tangent over +-k contour points; flip if it points into the subject
    k = max(3, len(cnt) // 60)
    tang = cnt[(idx + k) % len(cnt)] - cnt[(idx - k) % len(cnt)]
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    H, W = b.shape
    probe = np.clip((con + 5 * nrm).round().astype(int), 0, [W - 1, H - 1])
    inside = b[probe[:, 1], probe[:, 0]]
    nrm[inside] *= -1
    # interior
    er = cv2.erode(b.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    iy, ix = np.nonzero(er)
    cand = np.stack([ix, iy], 1).astype(np.float64)[::3]
    if len(cand) < max(n_int, 1):
        cand = np.stack([xs, ys], 1).astype(np.float64)[::3]
    d = np.linalg.norm(cand[:, None, :] - con[None, ::max(1, len(con) // 40)], axis=2).min(1) if len(con) else np.full(len(cand), 1e9)
    sel = []
    for _ in range(n_int):
        j = int(d.argmax())
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(cand - cand[j], axis=1))
    inter = cand[sel] if sel else np.zeros((0, 2))
    vec = inter - cen
    ln = np.linalg.norm(vec, axis=1, keepdims=True)
    inorm = np.where(ln > 3, vec / np.maximum(ln, 1e-6), np.array([0.0, -1.0]))
    pts = np.vstack([con, inter])
    normals = np.vstack([nrm, inorm])
    is_con = np.concatenate([np.ones(len(con), bool), np.zeros(len(inter), bool)])
    return pts, normals, is_con, cen


# --------------------------------------------------------------------------------------------- styles
def build_tracks(matte: np.ndarray, style: str = "pull_apart", *, n_points: int = 40, start_frame: int = 32,
                 fps: float = 24.0, seed: int = 1, distance: float = 300.0, stagger_s: float = 0.6, margin: float = 14.0,
                 angle_deg: float = -10.0, axis: str = "x", spin: float = 1.2, follow: str = "centroid",
                 threshold: float = 0.5):
    """Matte (F,H,W) -> (tracks (N,F,2) float64 in matte pixels, meta dict). `distance` in matte pixels (576x1024 canvas)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if follow not in ("centroid", "none"):
        raise ValueError("follow must be 'centroid' or 'none'")
    F, H, W = matte.shape
    f0 = int(max(0, min(start_frame, F - 2)))
    rng = np.random.default_rng(seed)
    pts, nrm, is_con, cen0 = seed_points(matte[0], n_points, threshold=threshold, seed=seed)
    N = len(pts)
    # follow offset: subject translation until f0, frozen afterwards
    if follow == "centroid":
        c = centroid_path(matte, threshold)
        off = c - c[0]
        off = np.vstack([off[:f0 + 1], np.repeat(off[f0:f0 + 1], F - f0 - 1, axis=0)])
    else:
        off = np.zeros((F, 2))
    base = pts[None] + off[:, None, :]                      # (F,N,2) before leaving
    p0 = base[f0]                                            # where each point leaves the subject
    f = np.arange(F, dtype=np.float64)[:, None]
    total_s = (F - 1 - f0) / fps
    u = rng.random(N)
    if style == "pull_apart":   # points next to the split line go first, outer ones follow (+ a little noise)
        ax_ = 0 if axis == "x" else 1
        stag = stagger_s * (0.6 * np.clip(np.abs(p0[:, ax_] - cen0[ax_]) / 120.0, 0, 1) + 0.4 * rng.random(N))
    else:
        stag = stagger_s * rng.random(N)
    stag = np.minimum(stag, 0.45 * total_s)
    s_lin = np.clip((f - f0 - stag[None] * fps) / np.maximum((total_s - stag[None]) * fps, 1.0), 0.0, 1.0)  # (F,N)
    ease = smoothstep5(s_lin)
    a = math.radians(angle_deg)
    pos = base.copy()
    if style == "pull_apart":
        ax = 0 if axis == "x" else 1
        side = np.sign(p0[:, ax] - cen0[ax])
        side[side == 0] = np.where(rng.random((side == 0).sum()) > 0.5, 1.0, -1.0)
        if axis == "x":
            d = np.stack([side * math.cos(a), np.full(N, math.sin(a))], 1)     # left (-cos, sin a), right (+cos, sin a)
        else:
            d = np.stack([np.full(N, math.sin(a)), side * math.cos(a)], 1)
        dist = distance * (0.75 + 0.5 * u) * (0.85 + 0.3 * np.clip(np.abs(p0[:, ax] - cen0[ax]) / 100.0, 0, 1))
        lim = np.full(N, 1e9)
        for j, (dim, lo, hi) in enumerate(((0, margin, W - margin), (1, margin, H - margin))):
            dj = d[:, dim]
            room = np.where(dj > 1e-6, (hi - p0[:, dim]) / np.maximum(dj, 1e-6),
                            np.where(dj < -1e-6, (lo - p0[:, dim]) / np.minimum(dj, -1e-6), 1e9))
            lim = np.minimum(lim, np.maximum(room, 0))
        dist = np.minimum(dist, lim)
        pos[:] = base + (ease[:, :, None] * dist[None, :, None] * d[None])
    elif style == "unravel":
        a0 = np.arctan2(nrm[:, 1], nrm[:, 0])
        R = distance * np.where(is_con, 0.85, 0.65) * (0.75 + 0.5 * u)
        ang = a0[None] + spin * ease                                          # direction turns while moving out
        disp = np.stack([np.cos(ang), np.sin(ang)], 2) * (R[None] * ease)[:, :, None]
        # shrink each point's whole path (not clamp it, which would pile points up on the border) until it fits the canvas
        lam = np.ones(N)
        for dim, lo, hi in ((0, margin, W - margin), (1, margin, H - margin)):
            top = np.maximum(disp[:, :, dim].max(0), 1e-6)
            bot = np.minimum(disp[:, :, dim].min(0), -1e-6)
            lam = np.minimum(lam, np.maximum((hi - p0[:, dim]) / top, 0.0))
            lam = np.minimum(lam, np.maximum((lo - p0[:, dim]) / bot, 0.0))
        pos[:] = base + disp * np.clip(lam, 0.0, 1.0)[None, :, None]
    elif style == "rise":
        # up is -y. Thinning near the top: the higher a thread starts, the shorter its path (it dissolves before the border),
        # and a random third of the threads stop at 60-85 % of their path so fewer threads reach the top.
        room = np.maximum(p0[:, 1] - margin, 0)
        keep = np.where(rng.random(N) < 0.35, rng.uniform(0.6, 0.85, N), 1.0)
        dist = np.minimum(distance * 1.4 * (0.7 + 0.6 * u) * keep, room)
        lift = ease                                                                # accelerates, then eases out toward the top
        sway = 5.0 * np.sin(2 * np.pi * (0.35 * (f - f0) / fps + rng.random(N)[None])) * ease
        pos[:] = base
        pos[:, :, 1] = base[:, :, 1] - dist[None] * lift
        pos[:, :, 0] = base[:, :, 0] + sway
    else:   # rain
        dist = np.minimum(distance * 1.4 * (0.7 + 0.6 * u), np.maximum(H - margin - p0[:, 1], 0))
        fall = s_lin ** 1.7
        sway = 5.0 * np.sin(2 * np.pi * (0.35 * (f - f0) / fps + rng.random(N)[None])) * ease
        pos[:] = base
        pos[:, :, 1] = base[:, :, 1] + dist[None] * fall
        pos[:, :, 0] = base[:, :, 0] + sway
    pos[:, :, 0] = np.clip(pos[:, :, 0], margin, W - margin)
    pos[:, :, 1] = np.clip(pos[:, :, 1], margin, H - margin)
    tracks = np.transpose(pos, (1, 0, 2))                                      # (N,F,2)
    disp = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=1)
    meta = dict(style=style, n_points=int(N), frames=int(F), start_frame=f0, start_s=round(f0 / fps, 3),
                size=[int(W), int(H)], seed=int(seed), distance=float(distance), stagger_s=float(stagger_s),
                angle_deg=float(angle_deg), axis=axis, spin=float(spin), follow=follow,
                contour_points=int(is_con.sum()), subject_centroid0=[round(float(cen0[0]), 1), round(float(cen0[1]), 1)],
                net_displacement_px=dict(min=round(float(disp.min()), 1), median=round(float(np.median(disp)), 1),
                                         max=round(float(disp.max()), 1)))
    return tracks, meta


def corridor_matte(tracks: np.ndarray, size=DEFAULT_SIZE, radius: float = 20.0, trail: int = 16) -> np.ndarray:
    """(N,F,2) tracks in `size` px -> (F,H,W) float32 0/1: at frame f the union of discs of `radius` swept by every track over
    its last `trail` frames. Used as the node's extra_mask so the regenerated area also covers where the threads travel."""
    import cv2
    w, h = size
    N, F, _ = tracks.shape
    out = np.zeros((F, h, w), np.uint8)
    for f in range(F):
        a = max(0, f - trail)
        for i in range(N):
            pl = np.round(tracks[i, a:f + 1]).astype(np.int32).reshape(-1, 1, 2)
            if len(pl) > 1:
                cv2.polylines(out[f], [pl], False, 1, thickness=int(2 * radius), lineType=cv2.LINE_8)
            cv2.circle(out[f], tuple(int(v) for v in pl[-1, 0]), int(radius), 1, -1)
    return out.astype(np.float32)


def scale_tracks(tracks: np.ndarray, from_size, to_size) -> np.ndarray:
    """(N,F,2) in `from_size` (w,h) px -> `to_size` px."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return np.asarray(tracks) * np.array([sx, sy])


def to_json_tracks(tracks: np.ndarray) -> list:
    """(N,F,2) -> [[{"x","y"} per frame] per track] (rounded to 0.1 px)."""
    return [[{"x": round(float(x), 1), "y": round(float(y), 1)} for x, y in t] for t in tracks]


def from_json_tracks(data) -> np.ndarray:
    return np.array([[(float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1])) for p in t]
                     for t in data])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matte-dir", required=True)
    ap.add_argument("--style", choices=STYLES, default="pull_apart")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis", help="analysis.json with beats / downbeats (seconds)")
    ap.add_argument("--start-s", type=float, default=1.3)
    ap.add_argument("--snap", choices=["beat", "downbeat", "none"], default="beat")
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--distance", type=float, default=300.0)
    ap.add_argument("--stagger-s", type=float, default=0.6)
    ap.add_argument("--axis", choices=["x", "y"], default="x")
    ap.add_argument("--angle", type=float, default=-10.0)
    ap.add_argument("--spin", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--size", default="576x1024", help="matte / output canvas WxH")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    m = load_matte_seq(a.matte_dir, (w, h))
    an = json.load(open(a.analysis, encoding="utf-8")) if a.analysis else None
    f0, t0, label = snap_start(a.start_s, an, a.fps, a.snap, len(m))
    tr, meta = build_tracks(m, a.style, n_points=a.points, start_frame=f0, fps=a.fps, seed=a.seed, distance=a.distance,
                            stagger_s=a.stagger_s, angle_deg=a.angle, axis=a.axis, spin=a.spin)
    meta["start_snap"] = label
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(to_json_tracks(tr), fh)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    sys.exit(main())
