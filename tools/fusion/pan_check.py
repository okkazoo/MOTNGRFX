"""pan_check - local CPU check of the pan transitions at tiny size (no box, no GPU): two words per direction, a real plate frame, the type rendered
through the real camera chain, the plate panned by `plate_pan` with the track of that camera, composited, one contact sheet.

    python -m tools.fusion.pan_check [--out 16_app/core_check/v006_pan_dirs_local.png] [--size 192 320]

Each row = one direction: A holds, A exits / B enters over the travel (5 frames), B holds. Under each frame: the type's on-screen motion of word A
(px, projected through the camera) and the plate's displacement (px) - they are the same number by construction of the track and are asserted
equal in tests/test_pan_transitions.py. The last column prints the frame B's hit lands on and the distance to the nearest downbeat (frames).
Size must be multiples of 64 for spec.check_size: 192 x 320 is the smallest portrait that is both.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import spec as S  # noqa: E402
from tools.fusion import pan_plan as PAN  # noqa: E402
from tools.fusion.plate_pan import plate_pan  # noqa: E402

PLATES = {"down": "15_fusion/v005/out/footage/P16/P16_still_A_g01.png", "up": "15_fusion/v005/out/footage/P17/P17_still_A_g01.png",
          "right": "15_fusion/v005/out/footage/P18/P18_still_A_g01.png", "left": "15_fusion/v005/out/footage/P19/P19_still_A_g01.png"}
WORDS = {"down": ("LOOK", "DOWN"), "up": ("LOOK", "UP"), "right": ("TURN", "RIGHT"), "left": ("TURN", "LEFT")}
INK = np.array([255, 236, 200], np.float32) / 255.0


def synth_plate(size, direction="down"):
    """a plate with a grid and a gradient (used when the footage still is missing): displacement is easy to read on it."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([0.25 + 0.5 * xx / w, 0.3 + 0.4 * yy / h, 0.5 + 0.0 * xx], -1)
    img[(yy.astype(int) % 32) < 2] *= 0.6
    img[(xx.astype(int) % 32) < 2] *= 0.6
    return img


def load_plate(direction, size):
    p = os.path.join(ROOT, PLATES[direction])
    if not os.path.isfile(p):
        return synth_plate(size, direction)
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = im.shape[:2]
    sc = max(size[0] / w, size[1] / h)
    im = cv2.resize(im, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    y0, x0 = (im.shape[0] - size[1]) // 2, (im.shape[1] - size[0]) // 2
    return np.ascontiguousarray(im[y0:y0 + size[1], x0:x0 + size[0]])


def pan_spec(direction, size, beats=2, bpm=96, duration=9.0, words=None):
    a, b = words or WORDS[direction]
    return dict(size=list(size), fps=24, duration=duration, bpm=bpm, bar=4, camera="auto",
                words=[dict(text=a, module="lanes", font="Big Shoulders Display", transition=dict(kind="pan", dir=direction, beats=beats)),
                       dict(text=b, module="lanes", font="Big Shoulders Display")],
                palette=dict(bg=[0, 0, 0], fg=[255, 255, 255], accents=[[255, 255, 255], [255, 236, 200], [200, 220, 255]]))


def build(direction, size=(192, 320), **kw):
    sc = S.build_scene(pan_spec(direction, size, **kw), size)
    tr = PAN.plate_track(sc)
    return sc, tr


def type_alpha(sc, t, size):
    f = sc.seq.frame(t, sc.chain, size=size, S=1).astype(np.float32) / 255.0
    return f.mean(-1)


def frame_at(sc, tr, plate, i, size):
    """plate (panned by the track at frame i, compose size = type size) + the type over it -> float RGB."""
    W, H = tr["world"]
    pl = plate_pan(plate, dx=tr["dx"][i] / W, dy=tr["dy"][i] / H, vx=tr["vx"][i] / W, vy=tr["vy"][i] / H, seam_amt=tr["amt"][i])
    a = type_alpha(sc, i / sc.fps, size)[..., None]
    return pl * (1.0 - a) + INK * a


def sheet(out, size=(192, 320), dirs=("down", "up", "right", "left"), cols=7):
    rows, report = [], []
    for d in dirs:
        sc, tr = build(d, size)
        plate = load_plate(d, size)
        t0, t1, _ = tr["windows"][0]
        fps = sc.fps
        ts = [t0 - 0.12] + [t0 + (t1 - t0) * u for u in (0.12, 0.3, 0.5, 0.7, 0.88)] + [t1 + 0.55]
        idx = [min(int(round(t * fps)), tr["nf"] - 1) for t in ts]
        mv = PAN.type_motion(sc, 0)
        cells = []
        for k, i in enumerate(idx):
            im = (np.clip(frame_at(sc, tr, plate, i, size), 0, 1) * 255).astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im, (0, 0), (size[0] - 1, size[1] - 1), (60, 60, 60), 1)
            lab = f"f{i}"
            cv2.putText(im, lab, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            pad = np.full((34, size[0], 3), 18, np.uint8)
            cv2.putText(pad, f"type {mv[i][0]:+.0f},{mv[i][1]:+.0f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(pad, f"plate {tr['dx'][i]:+.0f},{tr['dy'][i]:+.0f}", (3, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1, cv2.LINE_AA)
            cells.append(np.vstack([im, pad]))
        grid = sc.grid
        hit_t = sc.info["words"][1]["hit"]
        dn = float(grid.snap(hit_t, bars=True))
        info = np.full((size[1] + 34, size[0], 3), 18, np.uint8)
        for j, txt in enumerate([f"CAMERA {d.upper()}", f"travel {t0:.2f}-{t1:.2f}s", f"B hit {hit_t:.3f}s", f"downbeat {dn:.3f}s",
                                 f"off {(hit_t - dn) * fps:+.2f} f", f"peak {np.hypot(tr['vx'], tr['vy']).max():.0f}px/f"]):
            cv2.putText(info, txt, (6, 24 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(cells + [info]))
        report.append(dict(dir=d, travel=(round(t0, 3), round(t1, 3)), hit=round(hit_t, 3), downbeat=round(dn, 3),
                           off_frames=round((hit_t - dn) * fps, 3), peak_px_per_frame=round(float(np.hypot(tr["vx"], tr["vy"]).max()), 1)))
    img = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, img)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "16_app", "core_check", "v006_pan_dirs_local.png"))
    ap.add_argument("--size", type=int, nargs=2, default=[192, 320])
    a = ap.parse_args(argv)
    rep = sheet(a.out, tuple(a.size))
    for r in rep:
        print(r)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
