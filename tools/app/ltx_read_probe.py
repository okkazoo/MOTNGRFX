#!/usr/bin/env python
"""LTX READ PROBE: does the design VLM read back the word an LTX text pass was asked to draw? (tools/fusion/ltx_check.py crop_jobs / read_verdict)

Runs the SAME crop + verdict code the compose seam uses (nodes_fusion/layers.py MographLtxRead) on an archived layered shot, with the VLM called through the app's
own MographGGUFVL node, so it works on a box whose ComfyUI started before MographLtxRead existed:

    python tools/app/ltx_read_probe.py crops --piece P/piece.json --textbg P/s1_textbg.mp4 --sid s1 --out DIR [--work WORK]
        type matte of the needed frames only (CPU, cached in WORK/type; the other frames are blank placeholders), crops -> DIR/crop_NN.png + DIR/jobs.json
    python tools/app/ltx_read_probe.py graph --out DIR --arm gemma12b_q6 [--prefix ltxread_a_]
        DIR/graph.json: one LoadImage + one MographGGUFVL per crop (prompt = ltx_check.READ_PROMPT, the arm's settings, model kept loaded between the calls) + PreviewAny
        scp DIR/crop_*.png box:/workspace/ComfyUI/input/ ; submit graph.json with scripts/progress_standin.py
    python tools/app/ltx_read_probe.py verdict --out DIR --replies replies.json [--piece P/piece.json]
        replies.json = {"<node id>": "<reply>"} (or a list in crop order) -> the per-word table (word, readings, matches, pass / fail)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import ltx_check as LC  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402


def read_mp4(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[..., ::-1])
    return np.stack(out).astype(np.uint8)


def prepare_matte(piece: dict, type_dir: str, frames: list[int], nf: int, size=(384, 704)) -> None:
    """type matte of `frames` rendered on the CPU (tools/fusion/type_passes), blank placeholders elsewhere (ltx_check indexes the matte files by frame number)."""
    from tools.fusion import type_passes as TP
    d = os.path.join(type_dir, "matte")
    os.makedirs(d, exist_ok=True)
    blank = np.zeros((8, 8), np.uint8)
    for f in range(nf):
        p = os.path.join(d, f"matte_{f:05d}.png")
        if not os.path.isfile(p):
            cv2.imencode(".png", blank)[1].tofile(p)
    spec = copy.deepcopy(piece["type"])
    spec["fps"], spec["duration"] = int(piece.get("fps") or 24), float(piece.get("duration") or 16.0)
    sub = os.path.join(type_dir, "_r")
    for a, b in _ranges(sorted(set(frames))):
        TP.render_passes(spec, sub, size=size, passes=["matte"], frames=(a, b + 1), procs=4, overwrite=True, mp4=False, verbose=False)
        for f in range(a, b + 1):
            src = os.path.join(sub, "matte", f"matte_{f:05d}.png")
            if os.path.isfile(src):
                cv2.imencode(".png", cv2.imread(src, 0))[1].tofile(os.path.join(d, f"matte_{f:05d}.png"))


def _ranges(fs: list[int]) -> list[tuple[int, int]]:
    out = []
    for f in fs:
        if out and f - out[-1][1] <= 3:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def cmd_crops(a) -> int:
    piece = json.load(open(a.piece, encoding="utf-8"))
    textbg = read_mp4(a.textbg)
    work = a.work or os.path.join(a.out, "work")
    type_dir = os.path.join(work, "type")
    wins = OW.windows(piece)
    from tools.fusion.recipe_eval import piece_layout
    lay, _ = piece_layout(piece)
    L = lay[a.sid]
    need = []
    for w in wins:
        if w["render"] in ("ltx", "handoff"):
            need += LC.word_frames(w, int(piece.get("fps") or 24), L["start"], textbg.shape[0])
    if not need:
        print("no LTX-owned word in this shot")
        return 1
    prepare_matte(piece, type_dir, need, textbg.shape[0] + L["start"])
    jobs = LC.crop_jobs(piece, a.sid, type_dir, textbg, wins=wins)
    os.makedirs(a.out, exist_ok=True)
    meta = []
    for k, j in enumerate(jobs):
        name = f"{a.prefix}crop_{k:02d}.png"
        cv2.imencode(".png", j["image"][..., ::-1])[1].tofile(os.path.join(a.out, name))
        meta.append({x: j[x] for x in ("i", "text", "render", "frame")} | {"file": name})
    json.dump(meta, open(os.path.join(a.out, f"{a.prefix}jobs.json"), "w"), indent=1)
    print(f"{len(jobs)} crops -> {a.out}")
    return 0


def cmd_graph(a) -> int:
    from tools.app import llm_arms as LA
    arm = LA.ARMS[a.arm]
    meta = json.load(open(os.path.join(a.out, f"{a.prefix}jobs.json")))
    g: dict = {"prompt": {"inputs": {"value": LC.READ_PROMPT}, "class_type": "PrimitiveStringMultiline", "_meta": {"title": "IN prompt"}}}
    for k, m in enumerate(meta):
        g[f"img{k}"] = {"inputs": {"image": m["file"]}, "class_type": "LoadImage", "_meta": {"title": f"IN crop {k}"}}
        base = {"inputs": {"custom_prompt": ["prompt", 0], "image": [f"img{k}", 0], "max_tokens": 64, "temperature": 0.0, "top_p": arm["top_p"],
                           "repetition_penalty": arm["repeat_penalty"], "seed": 1}, "_meta": {"title": "x"}}
        n = LA.convert_node(base, arm, title=f"{arm['label']} read {k}")
        n["inputs"]["image"] = [f"img{k}", 0]
        n["inputs"]["keep_model_loaded"] = k < len(meta) - 1           # one load for all crops, freed after the last
        g[f"llm{k}"] = n
        g[f"out{k}"] = {"inputs": {"source": [f"llm{k}", 0]}, "class_type": "PreviewAny", "_meta": {"title": f"OUT read {k}"}}
    json.dump(g, open(os.path.join(a.out, f"{a.prefix}graph.json"), "w"), indent=1)
    print(f"graph with {len(meta)} reads -> {a.out}")
    return 0


def cmd_verdict(a) -> int:
    meta = json.load(open(os.path.join(a.out, f"{a.prefix}jobs.json")))
    rep = json.load(open(a.replies))
    replies = [rep[f"llm{k}"] if isinstance(rep, dict) else rep[k] for k in range(len(meta))]
    piece = json.load(open(a.piece, encoding="utf-8")) if a.piece else {}
    rows = LC.reads_from([dict(i=m["i"], text=m["text"], frame=m["frame"]) for m in meta], replies)
    print(f"{'word':16s} {'readings':48s} {'match':6s} verdict")
    for r in rows:
        v = LC.read_verdict(r["text"], r["readings"], LC.allowed_dist(piece, r["text"]))
        print(f"{r['text']:16s} {str(r['readings']):48s} {v['matches']}/{v['n']}   {'PASS' if v['ok'] else 'FAIL -> flat'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--piece", required=True)
    c.add_argument("--textbg", required=True)
    c.add_argument("--sid", default="s1")
    c.add_argument("--out", required=True)
    c.add_argument("--work")
    c.add_argument("--prefix", default="")
    g = sub.add_parser("graph")
    g.add_argument("--out", required=True)
    g.add_argument("--arm", default="gemma12b_q6")
    g.add_argument("--prefix", default="")
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True)
    v.add_argument("--replies", required=True)
    v.add_argument("--piece")
    v.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    return {"crops": cmd_crops, "graph": cmd_graph, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
