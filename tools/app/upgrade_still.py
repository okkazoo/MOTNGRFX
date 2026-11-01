#!/usr/bin/env python
"""UPGRADE a STILL PREVIEW to the real render: the SAME stills, the SAME design (no model call), real LTX footage. Doc: 16_app/STILL_PREVIEW.md.

    python tools/app/upgrade_still.py --target COMPETITIONS/motion-graphic/app/v012 --part r01_gemma_ltx_still --iid 12345678 --port 8190 [--quality final] [--dry-run]
    python tools/app/run_batch.py --upgrade COMPETITIONS/motion-graphic/app/v012/r01_gemma_ltx_still --iid 12345678 --port 8190      # the same thing

What a still-preview run keeps (MographArchive, nothing extra to save): `<stem>_piece.json` = the DESIGNED piece (words, fonts, modules, compose chains, cuts, shots, stills; its `request` key = the app
inputs of the run, `still_preview` = true), `<stem>_still.png` + `<stem>_still_<k>.png` = the stills of the places (draft size, 576x1024), `<stem>_music.wav` = the song.

The upgrade turns the app graph that made the preview (any arm, e.g. fusion_design_qwen_auto_layers_pivot_gemma31b_qat.json) into the graph of the real render, by `upgrade_graph` (pure):
    quality                  -> the app's Final (still_preview.APP_FINAL_TIER = draft sizes, real LTX: clean i2v, Alpha-Gen over the shot, depth); the piece's `still_preview` flag is off
    the concept chain        -> gone; `request` carries the saved words / scene / music / look / shots / owners as literals (+ length, seed, format)
    the MOSS still chains    -> gone; the stills are LoadImage nodes of the saved PNGs (same pixels), the angle-still chain is off
    plan + design_piece      -> MographPieceFromJSON of the saved designed piece: the shots / prompts / seeds of the preview run, the design as the model made it (no design model call, no repair)
    the song                 -> the saved wav as the user audio (the music chain is skipped; the beat grid is measured on that audio)
Everything else (type passes, LTX clean shots, Alpha-Gen, depth, flow, 3D cuts, compose, archive) is the unchanged app graph. Only possible while APP_FINAL_TIER is draft: the saved stills are the draft-size stills.

NOTHING IS OVERWRITTEN (user rule): the upgrade READS the preview's files and writes under a NEW part label (`r01_gemma_ltx_still` -> `r01_gemma_ltx_up540`, a second upgrade `..._up540b`), so
every ingredient it makes (stills as the new run's `<stem>_still*.png`, clips, mattes, depth, piece.json, wav) has its own stem; the ComfyUI input copies of the stills / song carry the new part too.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import ingredient_names as IN  # noqa: E402
from tools.fusion import places as PL  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402

REQUEST_NODE_FIELDS = ("words", "scene", "music", "look")          # app inputs that are IN nodes (or concept outputs) of the graph
CONCEPT_NODES = ("concept_prompt", "concept_qwen", "concept_parse", "concept_qwen2", "concept_final", "out_concept", "out_concept_reply", "out_concept_check", "previous")
DESIGN_OUTS = ("out_design_reply", "out_design_check", "out_design_final", "out_design_reasoning")
IMAGE_NODE = "LoadImage"
DEFAULT_GRAPH = os.path.join(ROOT, "app", "graphs", "fusion_design_qwen_auto_layers_pivot_gemma31b_qat.json")     # the app's default graph


# ---------------------------------------------------------------------------------------------- reading the preview run's files
def find_files(shots_dir: str) -> dict:
    """the files of a still-preview archive folder (`<stem>_out_shots` / `..._shots`): {stem, piece, stills {k: path}, audio}. Prefixed names first, then the short legacy ones."""
    names = os.listdir(shots_dir)
    stem = None
    for n in names:
        if n.endswith("_piece.json"):
            stem = n[:-len("_piece.json")]
    piece = os.path.join(shots_dir, f"{stem}_piece.json") if stem else os.path.join(shots_dir, "piece.json")
    stills: dict[int, str] = {}
    if stem:
        for k in range(1, PL.MAX_STILLS + 1):
            p = os.path.join(shots_dir, IN.still(stem, k))
            if os.path.isfile(p):
                stills[k] = p
        if 1 not in stills and os.path.isfile(os.path.join(shots_dir, IN.still(stem))):
            stills[1] = os.path.join(shots_dir, IN.still(stem))
    elif os.path.isfile(os.path.join(shots_dir, "still.png")):
        stills[1] = os.path.join(shots_dir, "still.png")
    audio = os.path.join(shots_dir, IN.nm(stem, "music.wav")) if stem else ""
    return {"stem": stem, "piece": piece if os.path.isfile(piece) else None, "stills": stills, "audio": audio if audio and os.path.isfile(audio) else None}


def load_saved(shots_dir: str) -> dict:
    """-> {stem, piece (dict), stills {k: path}, audio path}; raises when the preview run did not keep what the upgrade needs."""
    f = find_files(shots_dir)
    if not f["piece"]:
        raise FileNotFoundError(f"{shots_dir}: no <stem>_piece.json (the designed piece)")
    piece = json.load(open(f["piece"], encoding="utf-8"))
    if not piece.get(SP.FLAG):
        raise ValueError(f"{f['piece']}: not a still preview (piece has no {SP.FLAG}); nothing to upgrade")
    if not piece.get(SP.REQUEST_KEY):
        raise ValueError(f"{f['piece']}: the piece carries no `request` (planned before the still preview existed)")
    if not f["stills"]:
        raise FileNotFoundError(f"{shots_dir}: no stills (<stem>_still.png / <stem>_still_<k>.png)")
    if not f["audio"]:
        raise FileNotFoundError(f"{shots_dir}: no <stem>_music.wav (the song of the preview run)")
    lock = None
    try:                                                                       # "Use this layout" on the Randomise-layout sheet: <stem>_layout_lock.json -> the same draw on the saved design
        from tools.fusion import layout_preview as LPV
        lock = LPV.read_lock(shots_dir, f["stem"]) if f["stem"] else None
        if lock:
            import cv2
            import numpy as np
            first = f["stills"][min(f["stills"])]
            im = cv2.imdecode(np.fromfile(first, np.uint8), cv2.IMREAD_COLOR)
            stl = {k: cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for k, p in f["stills"].items()} if im is not None else None
            piece = LPV.apply_lock(piece, lock, stl)
    except Exception as e:      # noqa: BLE001  a lock that cannot be applied must not stop the upgrade: say so and keep the design as saved
        print(f"[upgrade_still] layout lock not applied: {type(e).__name__}: {e}")
        lock = None
    return {**f, "piece": piece, "layout_lock": lock}


# ---------------------------------------------------------------------------------------------- the pure graph transform
def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def prune_unreachable(g: dict) -> list[str]:
    from tools.app import layered_shots as LS
    return LS.prune_unreachable(g)


def upgrade_graph(g: dict, saved: dict, names: dict, quality: str = "final") -> dict:
    """the app graph `g` (API format, the one that made the preview) -> the graph of the real render at `quality` from the saved run. Pure (g is not modified).
    saved: {piece: the designed piece dict}; names: {stills: {k: ComfyUI input file name}, audio: input file name}."""
    if quality != "final":
        raise ValueError(f"the upgrade renders the app's Final (asked for {quality!r})")
    g = copy.deepcopy(g)
    piece = SP.upgraded_piece(saved["piece"])                                # raises unless SP.APP_FINAL_TIER is the draft tier (the stills are draft-size)
    req = dict(piece.get(SP.REQUEST_KEY) or {})
    piece_json = json.dumps(piece, indent=1, sort_keys=True, ensure_ascii=False)
    for need in ("quality", "request", "plan", "design_piece", "first", "audio"):
        if need not in g:
            raise ValueError(f"the graph has no node {need!r}: not an app design graph")
    # ---- request: the saved app inputs as literals, the concept chain gone
    g["quality"]["inputs"]["value"] = quality                                # the app's "final": MographRequest maps it to SP.APP_FINAL_TIER
    rin = g["request"]["inputs"]
    for f in REQUEST_NODE_FIELDS:
        if f in req:
            src = rin.get(f)
            if _is_link(src) and src[0] in g and g[src[0]]["class_type"].startswith(("Primitive",)):
                g[src[0]]["inputs"]["value"] = req[f]                       # an IN node of the plain graph
            else:
                rin[f] = req[f]                                             # a concept output of the auto graph -> the literal
    for f in ("shots", "owners"):
        if f in rin or f in req:
            rin[f] = req.get(f, "")
    for nid, key in (("length", "length"), ("seed", "seed")):
        if key in req and nid in g:
            g[nid]["inputs"]["value"] = int(req[key])
    if "format" in req and "format" in g:
        g["format"]["inputs"]["value"] = req["format"]
    for nid in CONCEPT_NODES:
        g.pop(nid, None)
    # ---- the song of the preview run as the user audio (the ACE chain is lazy-skipped), the picture inputs untouched
    g["audio"]["inputs"]["audio"] = names["audio"]
    # ---- the stills: LoadImage of the saved PNGs in place of the MOSS chains (same node ids, so every consumer link stays)
    stills = names.get("stills") or {}
    for k, name in sorted(stills.items()):
        nid = "still_dec" if int(k) == 1 else f"still{int(k)}_dec"
        if nid not in g:
            continue
        g[nid] = {"inputs": {"image": name}, "class_type": IMAGE_NODE, "_meta": {"title": f"Saved still of place {int(k)} (preview run)"}}
    if "first_set" in g and "angles" in g["first_set"]["inputs"]:
        g["first_set"]["inputs"]["angles"] = False                         # the saved rows already are the angle stills where a place had one
    # ---- plan + design: the saved designed piece (no design model call, no repair)
    music = g["plan"]["inputs"].get("music") or ["route", 1]
    g["plan"] = {"inputs": {"piece_json": piece_json, "music": music}, "class_type": "MographPieceFromJSON", "_meta": {"title": "Plan = the saved piece"}}
    g["design_piece"] = {"inputs": {"piece_json": piece_json, "music": music}, "class_type": "MographPieceFromJSON", "_meta": {"title": "Piece from design (saved)"}}
    for nid in DESIGN_OUTS:
        g.pop(nid, None)
    if "archive" in g:
        g["archive"]["inputs"].pop("sheet", None)                           # the design sheet belongs to the model call that did not run
    prune_unreachable(g)
    return g


def sidecar_for(side: dict | None, quality: str = "final") -> dict | None:
    """the app sidecar of the upgrade graph = the original with a title that says what it is."""
    if side is None:
        return None
    s = copy.deepcopy(side)
    s["title"] = f"{s.get('title', 'Fusion')} - upgrade of a still preview ({quality})"
    s["note"] = (str(s.get("note", "")) + " Upgrade: the same stills and the same design as the still preview, real LTX footage; run through tools/app/upgrade_still.py.").strip()
    return s


# ---------------------------------------------------------------------------------------------- the box: upload + submit
def upload_input(port: int, path: str, name: str | None = None, opener=None) -> str:
    """POST a file into the ComfyUI input dir through /upload/image (images and the wav); -> the name ComfyUI stored it under. `opener` = urllib opener (tests)."""
    import urllib.request
    import uuid
    name = name or os.path.basename(path)
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in (("type", "input"), ("overwrite", "true")):
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
    body += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with (opener or urllib.request.urlopen)(req, timeout=120) as r:
        return str(json.load(r).get("name") or name)


def input_names(saved: dict, part: str) -> dict:
    """the file names the upgrade uploads to the ComfyUI input dir (unique per part, so two upgrades never overwrite each other's stills)."""
    return {"stills": {k: f"upgrade_{part}_still_{k}.png" for k in saved["stills"]}, "audio": f"upgrade_{part}_music.wav"}


def upgrade_part(part: str, quality: str = "final", compose_h: int = 540, taken=()) -> str:
    """the NEW part label of an upgrade (never the source's, never one that already has output): `r01_gemma_still` -> `r01_gemma_ltx_up540`, `r01_gemma_ltx_still` -> `r01_gemma_ltx_up540`;
    a second upgrade of the same run -> `..._up540b`, `..._up540c` (`taken` = part labels that already exist in the version)."""
    base = part[:-len(SP.SUFFIX)] if part.endswith(SP.SUFFIX) else part
    if not base.endswith("_ltx"):
        base += "_ltx"
    new = f"{base}_up{int(compose_h)}"
    cand, i = new, 0
    while cand in set(taken) or cand == part:
        i += 1
        cand = new + chr(ord("a") + i)                                   # b, c, d ...
    return cand


def out_stem(target: str, part: str) -> str:
    """the run stem the outputs of `part` in version `target` carry (`motion-graphic_app_v012_r01_gemma_ltx_up540`: progress_standin puts project / idea / version / part in front of
    the graph's own `archive/motngrfx` prefix; tools/fusion/ingredient_names.run_stem). Every ingredient (stills, clips, mattes, depth, piece.json, wav) is written under it."""
    t = [x for x in str(target).replace("\\", "/").strip("/").split("/") if x]
    return "_".join(t[1:] + [part]) if len(t) > 1 else "_".join(t + [part])


def existing_parts(vdir: str) -> set[str]:
    """part labels that already have an archive folder in <version>/out (`*_<part>_out_shots`)."""
    import re
    out = set()
    for root, dirs, _ in os.walk(os.path.join(vdir, "out")):
        for d in dirs:
            m = re.search(r"_(r\d+[a-z]?(?:_[a-z0-9]+)*)_out_shots$", d)
            if m:
                out.add(m.group(1))
    return out


def run(a: dict, *, run_graph=None, uploader=upload_input, resolve=None, log=print) -> dict:
    """a: target (PROJECTS version, e.g. COMPETITIONS/motion-graphic/app/v012), part (the still preview's part label), iid, port, graph (path of the app graph), quality, dry_run, timeout."""
    from tools.app import run_batch as RB
    from tools.app import run_timing as RT
    run_graph = run_graph or RT.run_graph
    vdir = (resolve or RB.version_dir)(a["target"])
    if not vdir:
        raise SystemExit(f"cannot resolve {a['target']} (proj.py resolve)")
    f = RB.collect_files(vdir, a["part"])
    if not f.get("shots_dir"):
        raise SystemExit(f"no `*_{a['part']}_*_shots` folder in {vdir}/out: sync the preview run first (dashboard) or pull its ingredients")
    saved = load_saved(f["shots_dir"])
    gpath = os.path.abspath(a.get("graph") or DEFAULT_GRAPH)
    g0 = json.load(open(gpath, encoding="utf-8"))
    side0 = json.load(open(gpath[:-5] + ".app.json", encoding="utf-8")) if os.path.isfile(gpath[:-5] + ".app.json") else None
    sz = (saved["piece"].get("sizes") or {}).get("compose") or [540, 960]
    new_part = upgrade_part(a["part"], a.get("quality", "final"), int(min(sz)), existing_parts(vdir))
    if out_stem(a["target"], new_part) == out_stem(a["target"], a["part"]):
        raise SystemExit("the upgrade would write over the still preview's own files: refusing")
    names = input_names(saved, new_part)
    g = upgrade_graph(g0, saved, names, a.get("quality", "final"))
    out = os.path.join(a.get("out") or os.path.join(ROOT, "16_app", "bench", "upgrade_still"), new_part)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gfile = out + "_graph_src.json"                                         # the graph the run is built from (and its sidecar next to it, for run_timing's field fill)
    json.dump(g, open(gfile, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if side0 is not None:
        json.dump(sidecar_for(side0, a.get("quality", "final")), open(gfile[:-5] + ".app.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"upgrade {a['part']} -> {new_part}: {len(g)} nodes (from {len(g0)}), stills {sorted(saved['stills'])}, graph {gfile}")
    if a.get("dry_run"):
        return {"graph": gfile, "part": new_part, "names": names, "dry_run": True}
    for k, p in saved["stills"].items():
        uploader(a["port"], p, names["stills"][k])
    uploader(a["port"], saved["audio"], names["audio"])
    summ = run_graph(gfile, a["iid"], a["port"], {}, out, project=["--idea", a["target"], "--part", new_part], timeout=a.get("timeout") or 7200, graph_obj=g)
    return {"graph": gfile, "part": new_part, "summary": summ}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="PROJECTS version of the still preview, e.g. COMPETITIONS/motion-graphic/app/v012")
    ap.add_argument("--part", required=True, help="the part label of the still preview run, e.g. r01_gemma_ltx_still")
    ap.add_argument("--iid", required=True)
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--graph", default=DEFAULT_GRAPH, help="the app graph (the one that made the preview)")
    ap.add_argument("--quality", default="final", choices=("final",))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="build and write the upgrade graph, upload and submit nothing")
    a = ap.parse_args(argv)
    run(dict(target=a.target, part=a.part, iid=a.iid, port=a.port, graph=a.graph, quality=a.quality, out=a.out, timeout=a.timeout, dry_run=a.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
