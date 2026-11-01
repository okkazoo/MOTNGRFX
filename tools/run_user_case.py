#!/usr/bin/env python
"""Render a "real user" case end to end on CPU: a folder of photos + a line of text -> MP4.

    python tools/run_user_case.py 06_real-user/v001/local/inputs/U1 --text "Our cafe's new autumn menu" \
        --fmt both --duration 10 --out 06_real-user/v001/local/renders --name U1 --sheet 06_real-user/v001/review

mograph.analyze.analyze_inputs -> planner_rules.plan_from_analysis -> validate_plan ->
soundtrack.build_soundtrack -> render_frames -> encode_mp4. Writes <out>/<name>_<fmt>.mp4 and
<out>/<name>_plan.json; --sheet DIR also writes DIR/<name>_<fmt>_sheet.jpg (frames read back
from the mp4). Images are loaded in name order (jpg/jpeg/png/webp), RGB, long side capped at
--max-side (default 2048) the way an upload path would.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mograph import geometry  # noqa: E402
from mograph import plan as P  # noqa: E402
from mograph.analyze import analyze_inputs, audio_envelope  # noqa: E402
from mograph.encode import encode_mp4  # noqa: E402
from mograph.planner_rules import plan_from_analysis  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from mograph.soundtrack import build_soundtrack  # noqa: E402

FMTS = {"9x16": "9:16", "16x9": "16:9"}
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_images(folder: str, max_side: int) -> tuple[list[np.ndarray], list[str]]:
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    out = []
    for f in names:
        im = Image.open(os.path.join(folder, f))
        im.load()
        im = im.convert("RGB")
        if max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        out.append(np.asarray(im, np.float32) / 255.0)
    return out, names


def contact_sheet(video: str, times: list[float], out_path: str, row_h: int = 360) -> str:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (round(w * row_h / h), row_h), interpolation=cv2.INTER_AREA)
        cv2.putText(fr, f"{t:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(fr, f"{t:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        raise RuntimeError(f"no frames read from {video}")
    cols = 4 if tiles[0].shape[1] > tiles[0].shape[0] else len(tiles)  # landscape: 4 per row
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    gap = np.full((row_h, 4, 3), 255, np.uint8)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r]
        for t in tiles[r + 1:r + cols]:
            row = np.concatenate([row, gap, t], axis=1)
        rows.append(row)
    hgap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.concatenate([sheet, hgap, row], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", help="folder of images")
    ap.add_argument("--text", default="", help="user text (newlines split beats)")
    ap.add_argument("--fmt", choices=("9x16", "16x9", "both"), default="both")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--quality", choices=("draft", "full"), default="draft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--name", default=None, help="file prefix (default: images folder name)")
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--sheet", default=None, help="dir for 8-frame contact sheets")
    a = ap.parse_args(argv)

    name = a.name or os.path.basename(os.path.normpath(a.images))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    images, names = load_images(a.images, a.max_side)
    analysis = analyze_inputs(a.text.replace("\\n", "\n"), images, None, a.duration)
    plan = plan_from_analysis(analysis, seed=a.seed, duration=a.duration)
    errors = P.validate_plan(plan, analysis)
    if errors:
        raise SystemExit(f"plan invalid: {errors}")
    with open(os.path.join(a.out, f"{name}_plan.json"), "w", encoding="utf-8") as f:
        f.write(P.to_json(plan))
    summary = {"name": name, "files": names, "mood": analysis["mood"], "text_lines": analysis["text_lines"],
               "images": [{k: im[k] for k in ("id", "palette", "accent", "luma_mean", "contrast", "saturation",
                                              "saliency_box", "focus", "empty_regions")}
                          | {"n_contours": len(im["contours"]), "n_lines": len(im["lines"])}
                          for im in analysis["images"]]}
    with open(os.path.join(a.out, f"{name}_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    wave, sr = build_soundtrack(plan, analysis)
    print(f"{name}: analyze+plan+sound {time.perf_counter() - t0:.1f}s, {len(images)} images, "
          f"{len(plan['scenes'])} scenes, mood {analysis['mood']}")

    fps = plan["fps"]
    n_frames = int(round(plan["duration"] * fps))
    env = audio_envelope(wave, sr, fps, n_frames)
    for key in (("9x16", "16x9") if a.fmt == "both" else (a.fmt,)):
        fmt = FMTS[key]
        t = time.perf_counter()
        w, h = geometry.frame_size(fmt, a.quality)
        path = encode_mp4(render_frames(plan, analysis, fmt, a.quality, audio_env=env), fps, w, h, wave, sr,
                          os.path.join(a.out, f"{name}_{key}.mp4"))
        print(f"  {key}: {w}x{h} {n_frames} frames in {time.perf_counter() - t:.1f}s -> {path}")
        if a.sheet:
            d = plan["duration"]
            times = [round(d * (i + 0.5) / 8, 2) for i in range(8)]
            print("  sheet:", contact_sheet(path, times, os.path.join(a.sheet, f"{name}_{key}_sheet.jpg")))


if __name__ == "__main__":
    main()
