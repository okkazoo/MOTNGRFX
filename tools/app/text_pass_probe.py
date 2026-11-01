#!/usr/bin/env python
"""TEXT-PASS PROBE: which V2V start sigma / guide forms legible letters on top of an existing CLEAN LTX clip (three-layer route, 16_app/LAYERS.md).

Runs ONLY the text pass (`t{k}.*` of the layered app graph) on an archived clean clip: no concept / planner / Alpha-Gen / compose. The graph is cut out
of `app/graphs/fusion_design_qwen_auto_layers.json` (layered_shots.convert output), so the text chain, loaders, sampler and IC-guide wiring are the app's own:

    piece.json -> MographPieceFromJSON -> MographTypePasses -> MographShotSlot k   (control pass, prompt, seed, guide: exactly as the app)
    s{k}_clean.mp4 -> LoadVideo -> first N frames -> resize -> VAEEncode            (stands in for the clean chain's latent `l{k}.40`)
    t{k}.* (union-control IC guide, strength = guide) with `lay_sig` = tail of the distilled schedule from `--start`  -> VAEDecodeTiled -> SaveVideo

    python tools/app/text_pass_probe.py build --piece P/piece.json --clip s1_clean.mp4 --out DIR       # writes DIR/<variant>.json for the whole sweep
    python tools/app/text_pass_probe.py build ... --only d0909_g10 --frames 49
    python tools/app/text_pass_probe.py build ... --passes             # 2026-10-29: the same shot under the control passes depth / line / union (+ start 0.9875, 0.909, full-res guide)
    python tools/app/text_pass_probe.py score --render R.mp4 --clean s1_clean.mp4 --matte-dir TYPE/matte --start-frame N --first M   # layout IoU vs the flat type + pixel excess
    python tools/app/text_pass_probe.py strip RENDER.mp4 OUT.png [--n 6]                                # contact strip of n evenly spaced frames

The clip must be on the box in ComfyUI/input/ (LoadVideo): `scp s1_clean.mp4 box:/workspace/ComfyUI/input/`. Submit every graph with
scripts/progress_standin.py (repo rule). The first N frames are used (N = 8k+1); the control pass is cut from frame 0 too, so they line up.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.app import layered_shots as LS  # noqa: E402

LAYERS_GRAPH = os.path.join(ROOT, "app", "graphs", "fusion_design_qwen_auto_layers.json")
ONE_WORD = 'the word "{w}" carved in thick sandstone, extruded 3D letters'

# name -> (v2v start sigma, guide, prompt mode). prompt: "slot" = the slot prompt as the app sends it (all words), "word" = clean scene + ONE word clause.
SWEEP = {
    "s0909_g10": (0.909375, 1.0, "slot"),      # current default: 3 steps
    "s0975_g10": (0.975, 1.0, "slot"),         # 4 steps
    "s0994_g10": (0.99375, 1.0, "slot"),       # 7 steps
    "s1000_g10": (1.0, 1.0, "slot"),           # 8 steps = a full render from noise with the clean first frame
}
# 2026-10-29 (v005 r02 "DRIFT", Codystar dotted: the depth control gave LTX's own stacked letters, the reading failed): the same word under the three control passes the planner can pick
# (owner.control_pass_for: thin / dotted / outline faces -> line = canny, mid -> union = depth with the line drawn over, heavy -> depth), at draft size, V2V start 0.975, guide 1.0
# (the IC guide strength tops out at 1.0), plus the union with more freedom for the clean clip's scene (start 0.9875). `build --passes` writes these; `score` rates each render.
SWEEP_PASSES = {
    "p_depth": (0.975, 1.0, "slot", "depth"),
    "p_line": (0.975, 1.0, "slot", "line"),
    "p_union": (0.975, 1.0, "slot", "union"),
    "p_union_s09875": (0.9875, 1.0, "slot", "union"),
    "p_line_s0909": (0.909375, 1.0, "slot", "line"),       # more of the clean clip kept (3 steps): thin faces may hold their layout better where heavy ones only formed ghost outlines
    "p_line_ds1": (0.975, 1.0, "slot", "line", 1),         # the IC guide at full control resolution (latent_downscale_factor 1): the app uses 2, which turns thin dots into mush (untested: slower, more memory)
}


def with_pass(piece: dict, k: int, pass_name: str) -> str:
    """piece JSON with shot k's control pass set (depth | line | union) and the type passes it needs."""
    q = copy.deepcopy(piece)
    for s_ in q["shots"]:
        if s_["id"] == f"s{k}":
            s_.setdefault("control", {})["pass"] = pass_name
    tp = list(q.get("type_passes") or [])
    for n_ in {"depth": ("depth",), "line": ("line",), "union": ("depth", "line")}[pass_name]:
        if n_ not in tp:
            tp.append(n_)
    q["type_passes"] = tp
    for w in (q.get("type") or {}).get("words") or []:               # an archived piece has the fallback applied (flat, ltx_missing): the probe asks for the word the plan asked for
        if w.get("ltx_missing") and w.get("render_asked"):
            w["render"] = w.pop("render_asked")
            w.pop("ltx_missing", None)
            w.setdefault("ltx_at", 0.5)
            w.setdefault("material", "glowing wire strands, thin")
    return json.dumps(q)


def build(piece: dict, piece_json: str, clip_file: str, *, start: float, guide: float, prompt: str | None, k: int = 1, frames: int = 49,
          name: str = "probe", base: dict | None = None, downscale: int | None = None) -> dict:
    """one API graph: the text pass of shot k on the first `frames` frames of `clip_file` (a file in the box's ComfyUI/input)."""
    if (frames - 1) % 8:
        raise ValueError("frames must be 8n+1")
    g = copy.deepcopy(base if base is not None else json.load(open(LAYERS_GRAPH, encoding="utf-8")))
    t = f"t{k}."
    if f"{t}dec" not in g:
        raise ValueError(f"no text chain {t}* in the layered graph")
    # piece: from the archived piece.json (no concept / planner), music optional (the piece's type block carries beats)
    g["plan"] = {"class_type": "MographPieceFromJSON", "inputs": {"piece_json": piece_json}, "_meta": {"title": "Piece (archived)"}}
    g["type"]["inputs"] = {"piece": ["plan", 0], "procs": g["type"]["inputs"].get("procs", 10)}
    # the clean clip -> first `frames` frames -> the clean latent that `lay_gate{k}` carried
    g["pr_clip"] = {"class_type": "LoadVideo", "inputs": {"file": clip_file}}
    g["pr_comp"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["pr_clip", 0]}}
    g["pr_cut"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["pr_comp", 0], "batch_index": 0, "length": frames}}
    g["first"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["pr_comp", 0], "batch_index": 0, "length": 1}}
    g["pr_rs"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "scale_method": "lanczos", "input": ["pr_cut", 0], "resize_type": "scale dimensions", "resize_type.crop": "center",
        "resize_type.width": ["10", 0], "resize_type.height": ["11", 0]}}
    g[f"lay_gate{k}"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["pr_rs", 0], "vae": ["3", 0]}, "_meta": {"title": "Clean latent (encoded)"}}
    # control pass: only the same first `frames` frames (the slot hands the whole shot)
    g["pr_ctl"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"slot{k}", 3], "batch_index": 0, "length": frames}}
    g[t + "17"]["inputs"]["input"] = ["pr_ctl", 0]
    # the knobs
    g["lay_sig"]["inputs"]["sigmas"] = LS.v2v_sigmas(start)
    g[t + "32"]["inputs"]["strength"] = float(guide)
    if downscale is not None:
        g[t + "32"]["inputs"]["latent_downscale_factor"] = int(downscale)
    if prompt is not None:
        g[t + "6"]["inputs"]["text"] = prompt
    # decode + save (the app decodes with MographDecodeClip -> compose size; here the plain tiled decode at the latent size)
    g["pr_dec"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": [t + "41", 2], "vae": ["3", 0], "tile_size": 512, "overlap": 64,
                                                               "temporal_size": 64, "temporal_overlap": 8}}
    g["pr_vid"] = {"class_type": "CreateVideo", "inputs": {"images": ["pr_dec", 0], "fps": float(piece.get("fps", 24))}}
    g["pr_save"] = {"class_type": "SaveVideo", "inputs": {"video": ["pr_vid", 0], "filename_prefix": f"video/textprobe_{name}", "format": "mp4", "codec": "h264"}}
    # keep only what the SaveVideo reads
    seen, stack = set(), ["pr_save"]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack += [v[0] for v in g[n]["inputs"].values() if LS._is_link(v) and v[0] in g]
    return {n: v for n, v in g.items() if n in seen}


def one_word_prompt(piece: dict, k: int, word: str) -> str:
    """clean scene prompt of shot k (no letter clauses) + ONE word clause."""
    shot = next(s for s in piece["shots"] if s["id"] == f"s{k}")
    return LS.clean_prompt(shot["prompt"]) + ", " + ONE_WORD.format(w=word)


def score(render: str, clean: str, matte_dir: str, start_frame: int = 0, first: int = 0) -> int:
    """layout IoU (tools/fusion/ltx_check.layout_iou_frames) + pixel excess of a text-pass render against the flat type matte: the number that decides whether an LTX word counts."""
    import glob

    import cv2
    import numpy as np

    from tools.fusion import ltx_check as LC

    def frames(path):
        cap, out = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        return out
    r, c = frames(render), frames(clean)
    files = sorted(glob.glob(os.path.join(matte_dir, "*.png")))
    n = min(len(r), len(c))
    idx = [j for j in range(first, n) if start_frame + j < len(files)]
    idx = idx[::max(1, len(idx) // LC.MAX_SAMPLES)] or idx
    trip = [(c[j], cv2.resize(r[j], (c[j].shape[1], c[j].shape[0])), cv2.imread(files[start_frame + j], cv2.IMREAD_GRAYSCALE)) for j in idx]
    li, px = LC.layout_iou_frames(trip), LC.score_frames(trip)
    print(json.dumps(dict(render=os.path.basename(render), frames=len(idx), layout=li, pixel=px, passes_layout=bool(li and li["iou"] >= LC.LAYOUT_IOU_MIN)), indent=1))
    return 0


def strip(video: str, out: str, n: int = 6, height: int = 420, upto: int = 0) -> None:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if upto:
        total = min(total, upto)
    idx = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        w = int(f.shape[1] * height / f.shape[0])
        f = cv2.resize(f, (w, height), interpolation=cv2.INTER_AREA)
        cv2.putText(f, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(f)
    cv2.imwrite(out, np.hstack(tiles))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the sweep graphs")
    b.add_argument("--piece", required=True, help="archived piece.json")
    b.add_argument("--clip", required=True, help="file name of the clean clip in the box's ComfyUI/input")
    b.add_argument("--out", required=True)
    b.add_argument("--shot", type=int, default=1)
    b.add_argument("--frames", type=int, default=49)
    b.add_argument("--only", action="append", help="variant name(s); default the whole SWEEP")
    b.add_argument("--passes", action="store_true", help="the control-pass sweep SWEEP_PASSES (depth / line / union at draft size) instead of the start-sigma SWEEP")
    b.add_argument("--add", action="append", default=[], metavar="NAME=START,GUIDE[,word:W]",
                   help="extra variant, e.g. s0975_g065=0.975,0.65 or s0975_g10_w=0.975,1.0,word:HANDS")
    c = sub.add_parser("score", help="layout IoU of a render against the flat type matte (ltx_check.layout_iou_frames) + the pixel excess")
    c.add_argument("--render", required=True, help="the text-pass render (mp4)")
    c.add_argument("--clean", required=True, help="the clean clip it started from (mp4)")
    c.add_argument("--matte-dir", required=True, help="the type pass 'matte' directory of the piece (flat type, one png per piece frame)")
    c.add_argument("--start-frame", type=int, default=0, help="piece frame of the first frame of the clip (the shot's start)")
    c.add_argument("--first", type=int, default=0, help="first frame to score (after the handoff + 0.25 s)")
    s = sub.add_parser("strip", help="contact strip of a render")
    s.add_argument("video")
    s.add_argument("out")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--upto", type=int, default=0, help="only the first N frames (the clean clip is longer than the probe)")
    a = ap.parse_args(argv)
    if a.cmd == "strip":
        strip(a.video, a.out, a.n, upto=a.upto)
        return 0
    if a.cmd == "score":
        return score(a.render, a.clean, a.matte_dir, a.start_frame, a.first)
    text = open(a.piece, encoding="utf-8").read()
    piece = json.loads(text)
    variants = dict(SWEEP_PASSES if a.passes else SWEEP)
    for spec in a.add:
        nm, rest = spec.split("=", 1)
        parts = rest.split(",")
        variants[nm] = (float(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else "slot") + ((parts[3].split(":", 1)[1],) if len(parts) > 3 and parts[3].startswith("pass:") else ())
    base = json.load(open(LAYERS_GRAPH, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    for nm, (start, guide, pm, *ps) in variants.items():
        if a.only and nm not in a.only:
            continue
        prompt = one_word_prompt(piece, a.shot, pm.split(":", 1)[1]) if pm.startswith("word:") else None
        g = build(piece, with_pass(piece, a.shot, ps[0]) if ps else text, a.clip, start=start, guide=guide, prompt=prompt, k=a.shot, frames=a.frames, name=nm, base=base,
                  downscale=(ps[1] if len(ps) > 1 else None))
        path = os.path.join(a.out, nm + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1)
        print(f"{nm}: start {start} ({LS.steps_of(start)} steps) guide {guide} control {ps[0] if ps else 'as the piece'} prompt {'1 word' if prompt else 'slot'} -> {path} ({len(g)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
