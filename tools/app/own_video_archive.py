#!/usr/bin/env python
"""(runs ON THE BOX, stdlib only) assemble the archive dir MographArchiveLoad reads from the own-video segments.

    python own_video_archive.py --seg /workspace/own_video/seg --alpha-glob '/workspace/ComfyUI/output/projects/.../*_alpha_s{k}*_png*' --work /tmp/work --out DIR
sK_clip.mp4 <- seg/sK_clip.mp4; sK_matte/00001.png.. <- the first n frames of the Alpha-Gen PNG sequence of segment K (n = frames of sK_clip);
still.png <- first frame of s1; piece.json + index.json <- the planned piece (tools/app/own_video.py plan) with the shot starts.
"""
import argparse, glob, json, os, shutil, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--seg", required=True)
ap.add_argument("--alpha-glob", required=True, help="glob of the Alpha-Gen PNG frames of segment {k}")
ap.add_argument("--piece", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
idx = json.load(open(a.index))
for k in range(1, 7):
    n = len(glob.glob(a.alpha_glob.replace("{k}", str(k))))      # the clip + matte carry the shot's 8k+1 frames (span + the real frames that follow), like an LTX shot longer than its span
    shutil.copy(f"{a.seg}/../ComfyUI/input/own_video_s{k}.mp4" if os.path.isfile(f"{a.seg}/../ComfyUI/input/own_video_s{k}.mp4") else f"/workspace/ComfyUI/input/own_video_s{k}.mp4", f"{a.out}/s{k}_clip.mp4")
    pngs = sorted(glob.glob(a.alpha_glob.replace("{k}", str(k))))
    md = f"{a.out}/s{k}_matte"
    shutil.rmtree(md, ignore_errors=True)
    os.makedirs(md)
    for i, p in enumerate(pngs[:n], 1):
        shutil.copy(p, f"{md}/{i:05d}.png")
    print(f"s{k}: {n} clip frames, {len(pngs)} matte frames -> {n}")
shutil.copy(f"{a.seg}/s1_first.png", f"{a.out}/still.png")
shutil.copy(a.piece, f"{a.out}/piece.json")
shutil.copy(a.index, f"{a.out}/index.json")
print("archive", a.out)
