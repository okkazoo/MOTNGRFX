#!/usr/bin/env python
"""Patch the LTX-2.5 API graphs in workflows/api/ for one clip and write the result.

    python tools/ltx_graph.py i2v --image mograph/routes/S1_0.png --prompt "..." \
        --seconds 3 --fmt 9x16 --seed 7 --out /tmp/S1_i2v_0.json
    python tools/ltx_graph.py flf --image a.png --last b.png --prompt "..." --seconds 2 --out g.json

--image / --last are paths relative to ComfyUI/input on the box (upload them first).
Sizes: 9x16 -> 512x896, 16x9 -> 896x512 (draft, ~512 short side); --width/--height override.
Frames = seconds * fps + 1 (LTX wants 8k+1: 24 fps x 1..5 s qualifies).
Both graphs come from the official ComfyUI templates video_ltx2_5_i2v / video_ltx2_5_flf2v
(distilled int8 transformer), flattened to API format with the optional prompt enhancer removed.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "workflows", "api")
SIZES = {"9x16": (512, 896), "16x9": (896, 512)}

# node ids of the patchable fields
FIELDS = {
    "i2v": {"prompt": ("398:364", "text"), "width": ("398:372", "value"), "height": ("398:360", "value"),
            "seconds": ("398:362", "value"), "fps": ("398:361", "value"),
            "seed": [("398:339", "noise_seed"), ("398:338", "noise_seed")],
            "image": ("395", "image"), "prefix": ("75", "filename_prefix")},
    "flf": {"prompt": ("251:222", "text"), "width": ("251:215", "value"), "height": ("251:216", "value"),
            "seconds": ("251:198", "value"), "fps": ("251:205", "value"),
            "seed": [("251:196", "noise_seed")],
            "image": ("31", "image"), "last": ("39", "image"), "prefix": ("68", "filename_prefix")},
}


# i2v stage 2 (x2 latent upscale + 3-step refine) and everything only it uses; dropped for single-stage drafts
I2V_STAGE2 = ["398:348", "398:349", "398:371", "398:338", "398:340", "398:396", "398:391", "398:341", "398:368", "398:369"]


def build(kind: str, prompt: str, image: str, last: str | None = None, seconds: int = 3, fps: int = 24,
          fmt: str = "9x16", width: int | None = None, height: int | None = None, seed: int = 1,
          prefix: str | None = None, two_stage: bool = True) -> dict:
    """two_stage (i2v only): width/height are the FINAL size, stage 1 renders at half and stage 2 is the x2 latent upscale.
    two_stage=False: single stage at width x height (a draft: no stage 2, no x2 decode), flf is always two-stage."""
    if not two_stage and kind != "i2v":
        raise SystemExit("single stage is only available for i2v")
    with open(os.path.join(API, f"ltx_{kind}.json"), encoding="utf-8") as f:
        g = json.load(f)
    w, h = SIZES[fmt]
    vals = {"prompt": prompt, "width": int(width or w), "height": int(height or h), "seconds": int(seconds),
            "fps": int(fps), "seed": int(seed), "image": image, "last": last, "prefix": prefix}
    for key, loc in FIELDS[kind].items():
        v = vals.get(key)
        if v is None:
            continue
        for nid, inp in (loc if isinstance(loc, list) else [loc]):
            g[nid]["inputs"][inp] = v
    if kind == "flf" and not last:
        raise SystemExit("flf needs --last")
    if not two_stage:
        for nid in I2V_STAGE2:
            g.pop(nid)
        for nid in ("398:353", "398:355"):                # stage-1 size = the given size (was a/2)
            g[nid]["inputs"]["expression"] = "a"
        g["398:374"]["inputs"]["samples"] = ["398:367", 0]   # decode the stage-1 video latent ...
        g["398:358"]["inputs"]["samples"] = ["398:367", 1]   # ... and its audio
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["i2v", "flf"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--last")
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--fmt", choices=sorted(SIZES), default="9x16")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--single-stage", action="store_true", help="i2v draft: one stage at --width x --height, no x2 upscale")
    ap.add_argument("--prefix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = build(a.kind, a.prompt, a.image, a.last, a.seconds, a.fps, a.fmt, a.width, a.height, a.seed, a.prefix,
                not a.single_stage)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1, ensure_ascii=False)
    print(a.out)


if __name__ == "__main__":
    main()
