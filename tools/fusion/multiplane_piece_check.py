"""multiplane_piece_check: run the REAL compose path (piece_runner.Piece.recipe -> compose_parallel) of one shot of a multiplane piece on the box,
with the shot's footage / matte / depth taken from its `media` entry (no GPU graphs): the type passes are rendered by the piece's own type stage.

    python -m tools.fusion.multiplane_piece_check 15_fusion/pieces/P17.json --shot s1 --out-version v005 --out out/v005_ports/P17_mp [--frames N]

Writes <out>/frames/*.png, <out>/<piece>_<shot>_mp.mp4, <out>/strip.png (16 frames), <out>/recipe_ops.json (the ops the planner's recipe turned into)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.fusion import piece_runner as PR  # noqa: E402


def seed(P, sid):
    """state items for the shot from its `media` paths (footage mp4, matte / depth PNG folders or mp4 -> PNG folders next to them)."""
    media = P.shots[sid]["media"]
    shot = ROOT / media["shot"]

    def frames(key):
        p = ROOT / media[key]
        d = p.with_suffix("")
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), str(d / "f_%05d.png")], check=True)
        return [str(f) for f in sorted(d.glob("*.png"))]

    items = P.state["items"]
    rec = lambda **o: dict(out=o, wall_s=0.0, gpu_s=0.0, gpu_exec_s=0.0, t_start=0.0, t_end=0.0, at="seeded")
    sz = [int(v) for v in P.shots[sid].get("size", (384, 704))]
    items[f"shot:{sid}"] = rec(paths=[str(shot)], local=str(shot), size=sz)
    a, d = frames("matte"), frames("depth")
    items[f"alpha:{sid}"] = rec(paths=a, files=a, n=len(a))
    items[f"depth:{sid}"] = rec(paths=d, files=d, n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--shot", default="s1")
    ap.add_argument("--out-version", default="v005")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()
    PR.set_version(a.out_version)
    P = PR.Piece(a.piece, a.procs, exec_mode="local", preview=True)
    PR.TYPE_SIZE = tuple(P.type_size)                              # preview: the local type stage uses the module constant, the box stage self.type_size
    seed(P, a.shot)
    t0 = time.time()
    P.state["items"].pop("type", None)                            # always the piece's own type passes at this size (CPU)
    P.typ()
    print("passes", P.need_passes(), f"type {time.time() - t0:.0f}s", flush=True)
    rec = P.recipe(a.shot)
    print("ops", [(o["op"], o.get("out")) for o in rec["ops"]], flush=True)
    if a.frames:
        rec["frames"] = min(rec["frames"], a.frames)
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "recipe_ops.json").write_text(json.dumps([dict(op=o["op"], out=o.get("out")) for o in rec["ops"]], indent=1))
    t0 = time.time()
    spf = PR.compose_parallel(rec, out / "frames", a.procs)
    wall = time.time() - t0
    mp4 = out / f"{P.id}_{a.shot}_mp.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(rec["fps"]), "-i", str(out / "frames" / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", str(mp4)], check=True)
    n = rec["frames"]
    idx = np.linspace(0, n - 1, 16).round().astype(int)
    tiles = [cv2.imread(str(out / "frames" / f"{i:05d}.png")) for i in idx]
    cv2.imwrite(str(out / "strip.png"), np.vstack([np.hstack(tiles[:8]), np.hstack(tiles[8:])]))
    print(json.dumps(dict(frames=n, wall_s=round(wall, 1), s_per_frame_per_core=round(spf, 2), mp4=str(mp4))))


if __name__ == "__main__":
    main()
