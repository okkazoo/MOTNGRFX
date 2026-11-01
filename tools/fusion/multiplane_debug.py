"""multiplane_debug: every card of the multiplane plate on its own (rgb over magenta + alpha) at one time of the P14 check clip.
    python -m tools.fusion.multiplane_debug --plates DIR --out FILE.png --t 3.58"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

from mograph.render import cam3d, multiplane as MP  # noqa: E402
from tools.fusion import multiplane_check as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=float, default=3.58)
    a = ap.parse_args()
    S = C.build(a.plates, (360, 640), 24, 4, dict(C.CAMERA, chain=[e for e in C.CAMERA["chain"] if e["module"] == "push_through"]))
    cam3d.set_world(*S["scene"].world)
    mp = MP.MultiPlane(S["data"], S["scene"].world, (360, 640), cards=4, z=S["cfg"]["z"], ref=S["cfg"]["ref"])
    cam = S["chain"].at(a.t)
    f = int(round(a.t * 24))
    cards = mp.cards_at(f)
    grids, ts, valid = mp.warp(cam)
    print("last", mp.last)
    tiles = []
    for k in range(mp.n):
        smp = TF.grid_sample(cards[k:k + 1], grids[k:k + 1], mode="bilinear", padding_mode="reflection" if k == 0 else "zeros", align_corners=False)[0]
        rgb = smp[:3].permute(1, 2, 0).cpu().numpy()
        al = (smp[3] * valid[k]).cpu().numpy()
        over = rgb * al[..., None] + np.array([1.0, 0.0, 1.0]) * (1 - al[..., None])
        tiles.append(np.hstack([over[..., ::-1], np.repeat(al[..., None], 3, 2)]))
    cv2.imwrite(a.out, (np.hstack(tiles) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
