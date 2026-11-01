#!/usr/bin/env python
"""Render a style-matrix plan (08_styles/vNNN/plans/*.json) with the CPU pipeline, sound included.

    python tools/style_drafts.py 08_styles/v001/plans/S3_type_soft.json --fmt both --sheet
    python tools/style_drafts.py PLAN.json --fmt 9x16 --short 360 --sheet-times 0.5,2,4,6,8,9.5

Plan JSON -> validate_plan -> synth music + sfx -> mix -> render_frames -> mp4 (H.264 + AAC).
Default output: <vNNN>/out/motion-graphic_<idea>_<vNNN>_<cell>_<fmt>.mp4, where the plan
"S3_type_soft.json" in "08_styles/v001/plans/" gives idea "styles" and cell "S3_type-soft".
--sheet writes <vNNN>/review/<S3>_sheet.jpg: 6 frames per rendered format, read back from the mp4.

Plan extension read only by this script: audio.drums = false renders the music bed with the drum
kit silenced (pad, bass and arp only); the synth itself has no drum switch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mograph import geometry, mix, synth  # noqa: E402
from mograph.analyze import audio_envelope  # noqa: E402
from mograph.encode import encode_mp4  # noqa: E402
from mograph.plan import validate_plan  # noqa: E402
from mograph.render import render_frames  # noqa: E402

SR = 48000
FMTS = {"9x16": "9:16", "16x9": "16:9"}
ANALYSIS = {"text": "", "text_lines": [], "duration_hint": None, "images": [], "audio": None,
            "mood": {"energy": 0.5, "valence": 0.5, "keywords": []}}


def build_audio(plan: dict, sr: int = SR) -> np.ndarray:
    """Synth music bed + sfx, mixed; (2, N) float32."""
    if (plan.get("audio") or {}).get("drums", True) is False:
        kit = synth._kit

        def silent_kit(sr_, mood, rng):
            return {k: np.zeros_like(v) for k, v in kit(sr_, mood, rng).items()}
        synth._kit = silent_kit
        try:
            music = synth.render_soundtrack(plan, sr)
        finally:
            synth._kit = kit
    else:
        music = synth.render_soundtrack(plan, sr)
    return np.asarray(mix.mix(music, synth.render_sfx(plan, sr), None, sr), np.float32)


def default_names(plan_path: str):
    """-> (version dir, file prefix, cell id, short cell id)."""
    plan_path = os.path.abspath(plan_path)
    pdir = os.path.dirname(plan_path)
    vdir = os.path.dirname(pdir) if os.path.basename(pdir) == "plans" else pdir
    version = os.path.basename(vdir)
    idea = os.path.basename(os.path.dirname(vdir)).split("_", 1)[-1]
    stem = os.path.splitext(os.path.basename(plan_path))[0]
    short, _, rest = stem.partition("_")
    cell = f"{short}_{rest.replace('_', '-')}" if rest else short
    return vdir, f"motion-graphic_{idea}_{version}", cell, short


def contact_sheet(videos: list[tuple[str, str]], times: list[float], out_path: str, row_h: int = 300) -> str:
    """One row per video (label, path): the frames at ``times`` read back from the mp4."""
    rows = []
    for label, path in videos:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round(t * fps)))))
            ok, fr = cap.read()
            if not ok:
                fr = np.zeros((row_h, row_h, 3), np.uint8)
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (max(1, int(round(w * row_h / h))), row_h), interpolation=cv2.INTER_AREA)
            cv2.putText(fr, f"{label} {t:.2f}s", (6, row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(fr, f"{label} {t:.2f}s", (6, row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(fr, ((4, 4), (4, 4), (0, 0)), constant_values=40))
        cap.release()
        rows.append(np.concatenate(tiles, axis=1))
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=40) for r in rows], 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", help="plan JSON path")
    ap.add_argument("--fmt", choices=("9x16", "16x9", "both"), default="both")
    ap.add_argument("--quality", choices=("draft", "full"), default="draft")
    ap.add_argument("--short", type=int, default=540, help="draft short side in px (default 540)")
    ap.add_argument("--out", default=None, help="output dir (default <vNNN>/out)")
    ap.add_argument("--sheet", action="store_true", help="write a 6-frame contact sheet to <vNNN>/review/")
    ap.add_argument("--sheet-times", default="", help="comma list of seconds for the sheet (default: 6 even)")
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args(argv)

    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    errs = validate_plan(plan, ANALYSIS)
    if errs:
        sys.exit("plan invalid:\n  " + "\n  ".join(errs))
    vdir, prefix, cell, short = default_names(a.plan)
    out_dir = a.out or os.path.join(vdir, "out")
    geometry.DRAFT_SHORT_SIDE = int(a.short) // 2 * 2          # frame_size reads this at call time

    t0 = time.time()
    audio = build_audio(plan)
    print(f"audio {time.time() - t0:.1f}s", flush=True)
    fps = float(plan["fps"])
    n = int(round(float(plan["duration"]) * fps))
    env = audio_envelope(audio, SR, fps, n)

    videos = []
    for key in (("9x16", "16x9") if a.fmt == "both" else (a.fmt,)):
        t1 = time.time()
        W, H = geometry.frame_size(FMTS[key], a.quality)
        out = os.path.join(out_dir, f"{prefix}_{cell}_{key}.mp4")
        encode_mp4(render_frames(plan, ANALYSIS, FMTS[key], a.quality, env), fps, W, H, audio, SR, out, a.crf)
        print(f"[{key}] {W}x{H} {time.time() - t1:.1f}s -> {out}", flush=True)
        videos.append((key, out))

    if a.sheet:
        dur = float(plan["duration"])
        times = ([float(s) for s in a.sheet_times.split(",")] if a.sheet_times
                 else [round(dur * (k + 0.5) / 6.0, 2) for k in range(6)])
        p = contact_sheet(videos, times, os.path.join(vdir, "review", f"{short}_sheet.jpg"))
        print(f"sheet -> {p}", flush=True)


if __name__ == "__main__":
    main()
