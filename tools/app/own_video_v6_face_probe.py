#!/usr/bin/env python
"""(runs ON THE BOX, ComfyUI venv python) how much of the subject's FACE a word's layout box covers over the word's hold.

    python own_video_v6_face_probe.py --piece piece_designed.json --report retime_report.json --work W --alpha-dir D --stem S [--out face.json]

Face region of a frame = the top 25 % of the subject's matte bounding box (rows), the central 50 % of its width. A layout's box = stable_type.fit_box(...)["box"] (frame fractions, centred).
Overlap of a word = over the hold frames (pipeline frame = source frame) that HAVE a subject: the mean share of the face rectangle inside the layout box; `frames_covered` = the share of
the hold's frames where more than `--limit` (default 0.35) of the face rectangle is covered. Hold frames without a subject (inserts) cost nothing.
"""
import argparse
import json
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--piece", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--alpha-dir", required=True)
ap.add_argument("--stem", default="motion-graphic_own-video_v003")
ap.add_argument("--limit", type=float, default=0.35)
ap.add_argument("--out", default="")
ap.add_argument("--root", default="/workspace/ComfyUI/custom_nodes/motion-graphic")
a = ap.parse_args()
sys.path.insert(0, a.root)
from tools.fusion import stable_type as ST     # noqa: E402

piece = json.load(open(a.piece))
rep = json.load(open(a.report))
segs = json.load(open(a.work + "/segments.json"))
W, H = 384, 704


def piece_of(j):
    return next((s for s in segs if s["a"] <= j < s["b"] and s["person"]), None)


def face_rect(j):
    s = piece_of(j)
    if s is None:
        return None
    m = cv2.imread(f"{a.alpha_dir}/{a.stem}_alpha_h{s['job']}_image_{j - s['a'] + 1:05d}_.png", 0)
    if m is None:
        return None
    ys, xs = np.where(m > 127)
    if len(ys) < 200:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    cx = 0.5 * (x0 + x1)
    return (cx - 0.25 * w) / m.shape[1], y0 / m.shape[0], (cx + 0.25 * w) / m.shape[1], (y0 + 0.25 * h) / m.shape[0]


def covered(box, fr):
    ix0, iy0, ix1, iy1 = max(box[0], fr[0]), max(box[1], fr[1]), min(box[2], fr[2]), min(box[3], fr[3])
    area = max(fr[2] - fr[0], 1e-6) * max(fr[3] - fr[1], 1e-6)
    return max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0) / area


out = []
for i, w in enumerate(piece["type"]["words"]):
    rw = rep["words"][i]
    h0, h1 = int(rw["dry_hit"] * 24), int(rw["dry_rel"] * 24)
    frames = list(range(h0, h1, 2))
    faces = [(j, face_rect(j)) for j in frames]
    row = dict(text=w["text"], hold_frames=[h0, h1], frames_with_subject=sum(1 for _, f in faces if f is not None), layouts={})
    for lay in ST.allowed(str(w["text"])):
        bl = ST.fit_box(str(w["text"]), w.get("font"), w.get("axes"), lay, 9.0 / 16.0)
        rows = len(bl["lines"])
        ov = [covered(bl["box"], f) for _, f in faces if f is not None]
        row["layouts"][lay] = dict(rows=rows, lines=bl["lines"], cap=round(bl["cap"], 3), ok=bool(bl["ok"]), face_overlap_mean=round(float(np.mean(ov)), 3) if ov else 0.0,
                                   frames_covered=round(float(np.mean([o > a.limit for o in ov])), 3) if ov else 0.0)
    out.append(row)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out))
