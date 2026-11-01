"""Automatic first-pass review of a rendered video, in about a second. Prints flags, writes JSON + a strip.

    rep = qc.review("local/X_g03_draft.mp4", timeline, palette, mask_fn=piece.text_mask, out_prefix="review/X_g03_qc")
    python -m kit.qc VIDEO [--dur 10]          # standalone (no timeline: generic checks only)

Checks (all on a 360 px wide decode, luma 0..1):
- legibility at each read moment (middle of the read window): contrast between the glyphs and their
  surroundings (text mask from piece.text_mask(i, W, H) if given, else an Otsu split of the frame), taking
  the better of fill contrast (inside vs a ring outside) and edge contrast (the outline band vs the
  rest, for line-drawn type); glyph stroke width 2 * area / perimeter in px at 360 wide.
  Flags: contrast < 0.30, glyph stroke < 1.5 px.
- near-empty frames: mean |luma - ground| < 0.012 (ground = palette ground, else the median frame median).
- colour-area overruns: palette.overruns per frame (accent share > the palette's limit, off-palette > 5 %).
- frozen frames: runs where the mean frame difference < 0.0006 lasting > 1.0 s outside the read windows, or
  > 3.0 s anywhere (a dead stretch).
- timing: timeline.check() (read windows vs 0.5 s + 0.3 s/word, <= 20 cps, last >= 2 s).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

TH = dict(contrast=0.30, stroke=1.5, empty=0.012, frozen=0.0006, frozen_s=1.0, dead_s=3.0)


def decode(video, width=360):
    """All frames as uint8 RGB (N, H, W, 3) at `width` px wide, plus fps."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,r_frame_rate", "-of", "csv=p=0", video],
                        capture_output=True, text=True, check=True).stdout.strip().split(",")
    w0, h0 = int(pr[0]), int(pr[1])
    num, den = pr[2].split("/")
    fps = float(num) / float(den)
    H = int(round(h0 * width / w0 / 2)) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "0", "-i", video, "-vf", f"scale={width}:{H}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, width, 3)
    return fr, fps


def luma(fr):
    """Rec.601 luma 0..1 float32 (cv2 per frame: ~50x faster than float64 numpy on 250 frames)."""
    if fr.ndim == 3:
        return cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = np.empty(fr.shape[:3], np.float32)
    for i, f in enumerate(fr):
        out[i] = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    return out * (1.0 / 255.0)


def _stroke(m):
    """Mean stroke width of a binary mask: 2 * area / perimeter (px)."""
    m8 = m.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2.0 * float(m8.sum()) / per if per > 0 else 0.0


def legibility(Y, mask=None, ground=None):
    """dict(contrast, mode, stroke_px, ink_stroke_px, ink_share) for one luma frame (H, W) in 0..1."""
    if mask is None:                                     # Otsu split; ink = the class away from the ground
        u8 = (Y * 255).astype(np.uint8)
        thr, b = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = b > 0
        g = np.median(Y) if ground is None else ground
        if hi.any() and (~hi).any():
            mask = hi if abs(Y[hi].mean() - g) > abs(Y[~hi].mean() - g) else ~hi
        else:
            mask = hi
    mask = mask.astype(bool)
    if mask.sum() < 20:
        return dict(contrast=0.0, mode="none", stroke_px=0.0, ink_stroke_px=0.0, ink_share=0.0)
    k3 = np.ones((3, 3), np.uint8)
    m8 = mask.astype(np.uint8)
    dil = cv2.dilate(m8, k3, iterations=4) > 0
    ero = cv2.erode(m8, k3, iterations=1) > 0
    ring = dil & ~(cv2.dilate(m8, k3, iterations=1) > 0)
    fillc = abs(float(Y[ero].mean()) - float(Y[ring].mean())) if ero.any() and ring.any() else 0.0
    edge = (cv2.dilate(m8, k3, iterations=1) > 0) & ~(cv2.erode(m8, k3, iterations=1) > 0)
    rest = (dil & ~edge) | (cv2.erode(m8, k3, iterations=3) > 0)
    edgec = abs(float(Y[edge].mean()) - float(Y[rest].mean())) if edge.any() and rest.any() else 0.0
    # the ink actually drawn near the glyphs (frame-side stroke): pixels that differ from the ground
    g = float(np.median(Y[ring])) if ring.any() else float(np.median(Y))
    ink = (np.abs(Y - g) > 0.25) & dil
    return dict(contrast=round(max(fillc, edgec), 3), mode="fill" if fillc >= edgec else "edge",
                stroke_px=round(_stroke(mask), 2), ink_stroke_px=round(_stroke(ink), 2),
                ink_share=round(float(ink.mean()), 4))


def _runs(flags):
    out, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def review(video, timeline=None, palette=None, mask_fn=None, out_prefix=None, width=360, quiet=False, th=None):
    """Run every check; returns the report dict (also written to out_prefix.json + out_prefix_strip.jpg)."""
    th = dict(TH, **(th or {}))
    t_start = time.perf_counter()
    fr, fps = decode(video, width)
    N, H, W = fr.shape[:3]
    Y = luma(fr)
    flags = []
    ground = None
    if palette is not None:
        g = palette.ground
        ground = float(g[0] * 0.299 + g[1] * 0.587 + g[2] * 0.114)
    else:
        ground = float(np.median(np.median(Y.reshape(N, -1), axis=1)))
    energy = np.abs(Y - ground).reshape(N, -1).mean(1)
    empty = energy < th["empty"]
    for a, b in _runs(empty):
        flags.append(dict(kind="near_empty", t0=round(a / fps, 2), t1=round(b / fps, 2),
                          energy=round(float(energy[a:b + 1].min()), 4)))
    diff = np.r_[1.0, np.abs(np.diff(Y, axis=0)).reshape(N - 1, -1).mean(1)] if N > 1 else np.ones(1)
    frozen = diff < th["frozen"]
    reading = timeline.read_mask([n / fps for n in range(N)]) if timeline is not None else [False] * N
    for a, b in _runs(frozen):
        dur = (b - a + 1) / fps
        in_read = all(reading[a:b + 1])
        if dur > th["dead_s"] or (dur > th["frozen_s"] and not in_read):
            flags.append(dict(kind="frozen", t0=round(a / fps, 2), t1=round(b / fps, 2), seconds=round(dur, 2),
                              in_read=in_read))
    leg = []
    if timeline is not None:
        for i, t in enumerate(timeline.read_moments()):
            n = min(int(round(t * fps)), N - 1)
            m = mask_fn(i, W, H) if mask_fn is not None else None
            L = legibility(Y[n], m, ground)
            L.update(phrase=i, t=round(t, 2), text=timeline.phrases[i].text, mask="piece" if m is not None else "otsu")
            leg.append(L)
            if L["contrast"] < th["contrast"]:
                flags.append(dict(kind="low_contrast", phrase=i, t=L["t"], contrast=L["contrast"]))
            if m is not None and L["stroke_px"] < th["stroke"]:
                flags.append(dict(kind="thin_stroke", phrase=i, t=L["t"], stroke_px=L["stroke_px"]))
        for f in timeline.check():
            flags.append(dict(f, kind="timing_" + f["kind"]))
    colour = {}
    if palette is not None:
        from .palette import area, overruns
        worst = {}
        over, why = np.zeros(N, bool), {}
        step = max(1, int(round(fps / 5)))                              # 5 samples per second
        for n in range(0, N, step):
            sh = area(fr[n], palette)
            for k, v in sh.items():
                worst[k] = max(worst.get(k, 0.0), v)
            o = overruns(None, palette, shares=sh)
            if o:
                over[n:n + step] = True
                why[n] = o
        colour = {k: round(v, 3) for k, v in worst.items()}
        for a, b in _runs(over):
            hits = [o for n, os_ in why.items() if a <= n <= b for o in os_]
            peak = {}
            for o in hits:
                peak[o["colour"]] = max(peak.get(o["colour"], 0.0), o["share"])
            flags.append(dict(kind="colour_overrun", t0=round(a / fps, 2), t1=round(b / fps, 2), over=peak,
                              limit=palette.limit))
    rep = dict(video=os.path.basename(video), frames=N, fps=fps, size=f"{W}x{H}", ground=round(ground, 3),
               energy=dict(min=round(float(energy.min()), 4), mean=round(float(energy.mean()), 4)),
               motion=dict(min=round(float(diff[1:].min()) if N > 1 else 0, 5), mean=round(float(diff[1:].mean()) if N > 1 else 0, 5)),
               legibility=leg, colour_max_share=colour, flags=flags,
               thresholds=th, seconds=0.0)
    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        times = [L["t"] for L in leg] + ([0.5 * (a + b) for a, b in timeline.trans] if timeline is not None else [])
        times += [f["t0"] for f in flags if "t0" in f][:6]
        if not times:
            times = list(np.linspace(0, (N - 1) / fps, 8))
        bad = {round(f.get("t", f.get("t0", -1)), 2) for f in flags}
        tiles = []
        for t in sorted(set(round(x, 2) for x in times))[:14]:
            n = min(int(round(t * fps)), N - 1)
            im = cv2.resize(fr[n], (120, int(round(120 * H / W))), interpolation=cv2.INTER_AREA)
            im = cv2.copyMakeBorder(im, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                    value=(220, 40, 40) if t in bad else (60, 60, 60))
            tiles.append(im)
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out_prefix + "_strip.jpg", quality=88)
        rep["strip"] = os.path.basename(out_prefix + "_strip.jpg")
    rep["seconds"] = round(time.perf_counter() - t_start, 2)
    if out_prefix:
        with open(out_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    if not quiet:
        print(f"QC {rep['video']}: {len(flags)} flag(s) in {rep['seconds']:.2f}s")
        for L in leg:
            print(f"  read P{L['phrase'] + 1} t={L['t']:.2f} contrast {L['contrast']:.2f} ({L['mode']}, {L['mask']} mask) "
                  f"glyph stroke {L['stroke_px']:.1f}px ink stroke {L['ink_stroke_px']:.1f}px")
        for f in flags:
            print("  FLAG", json.dumps(f))
    return rep


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--palette", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from kit import palette as _pal
    review(a.video, None, _pal.get(a.palette) if a.palette else None, out_prefix=a.out)
