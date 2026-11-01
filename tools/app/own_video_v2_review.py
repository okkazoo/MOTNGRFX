#!/usr/bin/env python
"""Review sheets for the own-video route (runs on the PC: reads files already rendered on the box; no rendering).

    python tools/app/own_video_v2_review.py mattes --arch DIR --out sheet.png     # every 8th frame of every planned shot with its matte as a red overlay (matte follows its picture?)
    python tools/app/own_video_v2_review.py video  --video out.mp4 --out-dir DIR [--cuts 4.625,7.29,...] [--land 1.31,...]   # 3 fps sheets + per-cut strips + landing-frame strips
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np


def _label(img, text, org=(3, 12), scale=0.4, col=(0, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols]
        r += [np.zeros_like(tiles[0])] * (cols - len(r))
        rows.append(np.hstack(r))
    return np.vstack(rows)


def cmd_mattes(a) -> int:
    mp = json.load(open(os.path.join(a.arch, "matte_map.json")))
    tiles, tw, th = [], 108, 192
    for sid in sorted(mp, key=lambda s: int(s[1:])):
        clip = read_frames(os.path.join(a.arch, f"{sid}_clip.mp4"))
        for i in range(0, len(clip), 8):
            m = cv2.imread(os.path.join(a.arch, f"{sid}_matte", f"{i + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
            f = clip[i].copy()
            ov = f.copy()
            ov[m > 127] = (0.35 * ov[m > 127] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            t = cv2.resize(ov, (tw, th), interpolation=cv2.INTER_AREA)
            tl = mp[sid][i]
            _label(t, f"{sid}.{i} t{tl[0] / 24:.2f}", scale=0.33)
            tiles.append(t)
    cv2.imwrite(a.out, sheet(tiles, 12))
    print("wrote", a.out, len(tiles), "tiles")
    return 0


def cmd_video(a) -> int:
    os.makedirs(a.out_dir, exist_ok=True)
    fr = read_frames(a.video)
    n = len(fr)
    fps = float(a.fps)
    tw, th = 135, 240
    tiles = []
    step = int(round(fps / 3.0))          # 3 sheet frames per second of the video's own clock
    for i in range(0, n, step):
        t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
        _label(t, f"{i / fps:.2f}s", scale=0.4)
        tiles.append(t)
    per = 36
    for k in range(0, len(tiles), per):
        cv2.imwrite(os.path.join(a.out_dir, f"sheet_3fps_{k // per}.png"), sheet(tiles[k:k + per], 9))
    print("sheets:", (len(tiles) + per - 1) // per, "frames", n)
    if a.cuts:
        cuts = [float(x) for x in a.cuts.split(",")]
        for j, c in enumerate(cuts):
            f0 = int(round(c * fps))
            idx = [f0 - 6, f0 - 3, f0 - 1, f0, f0 + 1, f0 + 3, f0 + 6, f0 + 12]
            row = []
            for i in idx:
                i = min(max(i, 0), n - 1)
                t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
                _label(t, f"f{i} {i / fps:.2f}", scale=0.36)
                row.append(t)
            cv2.imwrite(os.path.join(a.out_dir, f"cut_{j + 1}_{c:.2f}s.png"), np.hstack(row))
    if a.moments:
        ms = json.load(open(a.moments))
        for j, m in enumerate(ms):
            A, Bm = int(m["a"]), int(m["b"])
            idx = [A - 8, A - 5, A - 2, A, (A + Bm) // 2, Bm - 1, Bm + 2, Bm + 5, Bm + 8]
            row = []
            for i in idx:
                i = min(max(i, 0), n - 1)
                t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
                _label(t, f"f{i} {i / fps:.2f}", scale=0.36)
                row.append(t)
            cv2.imwrite(os.path.join(a.out_dir, f"moment_{j + 1}_{m['mode']}_{m['word'].replace(' ', '')}.png"), np.hstack(row))
    if a.gaps:
        rep = json.load(open(a.gaps))
        ks = float(a.k)
        for g in [int(x) for x in a.gap_idx.split(",")]:
            A_, B_ = rep["words"][g], rep["words"][g + 1]
            t0, t1 = A_["dry_rel"] / ks, B_["dry_hit"] / ks                     # A's release (its exit starts) -> B's hit, real seconds
            idx = [int(round((t0 + (t1 - t0) * q / 5.0) * fps)) for q in range(6)]
            row = []
            for i in idx:
                i = min(max(i, 0), n - 1)
                t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
                _label(t, f"f{i} {i / fps:.2f}", scale=0.36)
                row.append(t)
            kind = (rep.get("transition_gaps") or [{}] * 9)[g].get("kind")
            cv2.imwrite(os.path.join(a.out_dir, f"transition_{g + 1}_{A_['text'].replace(' ', '')}_to_{B_['text'].replace(' ', '')}_{kind}.png"), np.hstack(row))
    if a.land:
        land = [float(x) for x in a.land.split(",")]
        rows = []
        for j, c in enumerate(land):
            f0 = int(round(c * fps))
            idx = [f0 - 12, f0 - 6, f0 - 3, f0 - 1, f0, f0 + 1, f0 + 3, f0 + 12, f0 + 24, f0 + 36]
            row = []
            for i in idx:
                i = min(max(i, 0), n - 1)
                t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
                _label(t, f"f{i} {i / fps:.2f}" + (" LAND" if i == f0 else ""), scale=0.36)
                row.append(t)
            rows.append(np.hstack(row))
            cv2.imwrite(os.path.join(a.out_dir, f"land_{j + 1}_{c:.2f}s.png"), rows[-1])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mattes", "video"])
    ap.add_argument("--arch")
    ap.add_argument("--out")
    ap.add_argument("--video")
    ap.add_argument("--out-dir")
    ap.add_argument("--cuts")
    ap.add_argument("--land")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--moments")
    ap.add_argument("--gaps", help="retime_report.json: a 6-frame strip across the word-to-word transitions listed in --gap-idx")
    ap.add_argument("--gap-idx", default="0,3")
    ap.add_argument("--k", default="1.25", help="stretched-clock factor of the report times")
    a = ap.parse_args()
    return cmd_mattes(a) if a.cmd == "mattes" else cmd_video(a)


if __name__ == "__main__":
    raise SystemExit(main())
