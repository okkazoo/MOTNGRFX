"""multiplane_recipe: make a piece's compose recipe render its plate as depth cards when the piece camera says plate = multiplane.

A multiplane piece (`type.camera = {"chain": .., "plate": "multiplane", "cards": 4, "subject_z": ..}`, words with `slot`) has its type rendered by a camera that
travels through the footage's depth cards. The planner's recipe draws the type flat over the footage (depth_occlude / sandwich / knockout ..), so
`apply` swaps that for the `multiplane` compose op (mograph.render.multiplane.compose_op: cards of the shot's depth + matte warped by the piece camera,
the type pass composited per pixel by depth, the legible guard on card occlusion) and drops the ops that would draw the type a second time (the matte
treatments front / window / fill / wipe: the cards do the behind / in front themselves; glow and contour lines stay). Dead ops are removed.

    ops = multiplane_recipe.apply(piece, ops, sid)       # piece_runner.Piece (needs .j, .id, .typ(), .shots)   -> ops (unchanged for a flat piece)
    multiplane_recipe.extra_passes(piece_json)           # {"beauty", "matte", "depth"} when the piece is multiplane, else set()
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

TYPE_DRAWERS = {"sandwich", "knockout", "depth_occlude", "depth_cover", "legible", "flow_wake"}    # (tint_type stays while the glow lines use it; mask ops go by prefix)
HELPER_PREFIX = ("_fr", "_wi", "_fi", "_leg", "_cov")
TYPE_ALPHAS = ("$wl", "$wm", "$wa")                       # flow_wake'd copies of the type alpha: the plain type matte stands in for them


def is_multiplane(piece_json):
    cam = (piece_json.get("type") or {}).get("camera")
    return isinstance(cam, dict) and cam.get("plate") == "multiplane"


def extra_passes(piece_json):
    return {"beauty", "matte", "depth"} if is_multiplane(piece_json) else set()


def _refs(o, acc):
    if isinstance(o, str) and o.startswith("$"):
        acc.add(o[1:])
    elif isinstance(o, dict):
        for v in o.values():
            _refs(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _refs(v, acc)


def _subst(o, m):
    if isinstance(o, str):
        return m.get(o, o)
    if isinstance(o, dict):
        return {k: _subst(v, m) for k, v in o.items()}
    if isinstance(o, list):
        return [_subst(v, m) for v in o]
    return o


def _dce(ops, output):
    """ops reduced to what the final `output` buffer needs (backward walk by name, honouring re-definitions)."""
    need, keep = {output}, []
    for j in range(len(ops) - 1, -1, -1):
        o = ops[j]
        out = o.get("out", "image")
        if out in need:
            need.discard(out)
            keep.append(j)
            acc = set()
            _refs(o.get("args", {}), acc)
            need |= acc
    return [ops[j] for j in sorted(keep)]


def setup_for_dir(piece_json, type_dir, device="cpu", scene_key=None):
    """the JSON `setup` of the multiplane op: the type scene's spec as the type pass rendered it (manifest), its size / world, the depth
    range of the type depth pass, the palette bg of the beauty pass, the word holds of the legible guard. `scene_key` shares the (slow to
    build) type scene between the shots of one piece."""
    from tools.fusion import recipe_eval, spec as S_
    tdir = Path(type_dir)
    man = json.loads((tdir / "manifest.json").read_text(encoding="utf-8"))
    sp = man["spec"]
    dsp = man.get("depth") or {}
    holds = recipe_eval.piece_word_holds(piece_json, str(tdir))
    out = dict(type_spec=sp, size=list(man["size"]), near=float(dsp.get("near", 500.0)), far=float(dsp.get("far", 1500.0)),
               mode=dsp.get("mode", "linear"), bg=list(S_.make_palette(sp.get("palette")).bg), holds=[[round(a, 4), round(b, 4)] for a, b in holds],
               device=device)
    if scene_key:
        out["scene_key"] = scene_key
    return out


def setup_for(piece, sid):
    """setup_for_dir of a piece_runner.Piece (the local runner: cpu cards)."""
    return setup_for_dir(piece.j, piece.typ()["dir"], "cpu")


def apply(piece, ops, sid):
    """see the module doc. Returns `ops` itself when the piece is flat or the shot lacks the depth map the cards need.
    piece_runner.Piece entry (the local runner); the on-box path (recipe_eval.build_recipe) calls `apply_to` with the same transform."""
    if not is_multiplane(piece.j):
        return ops
    if not piece.shots[sid].get("depth"):
        piece.log(f"multiplane {sid}: the shot has no depth pass (\"depth\": true) - plate stays flat")
        return ops
    sp = Path(piece.dir) / f"mp_setup_{sid}.json"                      # the setup holds the whole type spec: a file, not an op argument
    return apply_to(piece.j, ops, sid, str(piece.typ()["dir"]), str(sp), log=piece.log, state_key=f"{piece.id}:{sid}",
                    shot=piece.shots[sid])


def lc_ops(legible, plate, alpha="$tmat", face="$tface"):
    """the legible-colour ops of a multiplane shot (compose op `legible_color`, tools/fusion/legible_color.py): measure the plate behind the word, recolour the
    beauty face, lay it out like the beauty pass (palette bg outside the letters: the layout the `multiplane` op expects), support painted afterwards by
    `legible_support`. -> (pre ops, beauty-layout buffer, plain recoloured face buffer)."""
    bg = [float(x) / 255.0 if max(legible.get("bg") or [0]) > 1.0 else float(x) for x in (legible.get("bg") or [0.0, 0.0, 0.0])][:3]
    pre = [{"op": "legible_color", "out": "_lcmp", "lc": True, "args": {
               "face": face, "alpha": alpha, "plate": plate, "t": "$t", "frame_idx": "$frame_idx", "u": "$u", "fps": legible["fps"],
               "words": copy.deepcopy(legible["words"]), "state_key": legible["state_key"], "mode": "post",
               **{k: copy.deepcopy(legible[k]) for k in ("outline", "ink_palette", "soft_support", "shot_palette", "piece_key", "shot_t0", "user_palette", "boost", "outline_w", "life") if legible.get(k)}}},
           {"op": "rgba_part", "out": "lcmp_face", "lc": True, "args": {"rgba": "$_lcmp", "part": "rgb"}},
           {"op": "fill", "out": "_lcmp_bg", "lc": True, "args": {"like": face, "color": bg}},
           {"op": "over", "out": "lcmp_beauty", "lc": True, "args": {"base": "$_lcmp_bg", "fg": "$lcmp_face", "alpha": alpha}}]
    return pre, "$lcmp_beauty", "$lcmp_face"


def apply_to(piece_json, ops, sid, type_dir, setup_path, log=print, state_key=None, device="cpu", alpha="$type_matte", scene_key=None,
             shot=None, legible=None):
    """THE multiplane transform of a shot's op list (one place: `apply` and recipe_eval.build_recipe both come here). Returns `ops` itself (the
    same list object) when the piece is flat, the shot has no depth pass, no op draws the type, or the type manifest is missing; else the ops
    with the first type-drawing op replaced by `multiplane` and everything that drew the type a second time dropped.
    alpha: the buffer that stands in for the flow-wake'd type alphas and is the multiplane op's type matte ("$tmat" = the legibility pass's
    robust alpha). legibility_pass interplay: `hold_net` ops are kept and re-pointed at the beauty face ($tface / alpha) over the multiplane
    output (the net's plain version = the face drawn in front of the cards inside the word region); the multiplane op draws the type at full
    opacity (the planner's type_opacity ramps go with the dropped drawers), which is the hold floor.
    legible: dict(words, fps, state_key, bg) = the cards draw the face recoloured by `legible_color` (measured on the flat plate behind the word: the cards
    move with the camera, so this is a proxy) instead of the raw beauty; its support (outline / shadow / scrim) is painted under the type on the output."""
    if not is_multiplane(piece_json):
        return ops
    sh = shot if shot is not None else {x["id"]: x for x in piece_json.get("shots", [])}.get(sid, {})
    if not sh.get("depth"):
        log(f"multiplane {sid}: the shot has no depth pass (\"depth\": true) - plate stays flat")
        return ops
    if not (Path(type_dir) / "manifest.json").is_file():
        log(f"multiplane {sid}: no type manifest in {type_dir} - plate stays flat")
        return ops
    ops = copy.deepcopy(ops)
    i0 = next((k for k, o in enumerate(ops) if o["op"] in ("depth_occlude", "sandwich", "knockout")), None)
    if i0 is None:
        log(f"multiplane {sid}: no op draws the type - plate stays flat")
        return ops
    if legible and alpha == "$tmat":
        ops = [o for o in ops if not o.get("lc")]                   # the flat-draw colour ops go with the dropped drawers; lc_ops below replaces them
        i0 = next(k for k, o in enumerate(ops) if o["op"] in ("depth_occlude", "sandwich", "knockout"))
    comp = piece_json.get("compose", {})
    out = comp.get("shots", {}).get(sid, {}).get("output") or comp.get("default", {}).get("output") or "image"
    dev = device
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                                              # noqa: BLE001
            dev = "cpu"
    Path(setup_path).parent.mkdir(parents=True, exist_ok=True)
    Path(setup_path).write_text(json.dumps(setup_for_dir(piece_json, type_dir, dev, scene_key)), encoding="utf-8")
    plate = "$plate" if any(o["op"] == "layout_shapes" and o.get("out") == "plate" for o in ops) else "$footage"      # hold-layout strips are part of the plate
    lc_pre, lc_rgb, lc_face = lc_ops(legible, plate, alpha) if (legible and alpha == "$tmat") else ([], "$type_beauty", None)
    args = dict(footage=plate, depth=f"${sid}_depth", matte=(f"${sid}_matte" if sh.get("matte") else None), type_rgb=lc_rgb,
                type_a=alpha, type_depth="$type_depth", t="$t", state_key=state_key or f"{piece_json.get('id', 'piece')}:{sid}",
                setup_path=str(setup_path))
    ops = _subst(ops, {t: alpha for t in TYPE_ALPHAS})                # the flow-wake'd type alphas: the plain (robust) type matte stands in for them
    for o in ops:                                                    # the hold net draws the beauty face in front of the cards' output
        if o["op"] == "hold_net":
            a = o.setdefault("args", {})
            if alpha == "$tmat":
                a["face"], a["alpha"] = (lc_face or "$tface"), "$tmat"
            a["plate"] = a.get("image", "$image")
    new, gone = [], set()                                     # gone: buffers of dropped ops (the main buffer itself is re-assigned many times: never "gone")
    for k, o in enumerate(ops):
        name, o_out = o["op"], o.get("out", "image")
        acc = set()
        _refs(o.get("args", {}), acc)
        if k == i0:
            new += lc_pre
            new.append({"op": "multiplane", "out": o_out, "args": args})
            if lc_pre:                                           # the support of the colour stage goes under the type, over the cards
                new.append({"op": "legible_support", "out": o_out, "lc": True, "args": {
                    "image": "$" + o_out, "alpha": alpha, "state_key": legible["state_key"], "frame_idx": "$frame_idx"}})
            continue
        drop = name in TYPE_DRAWERS or str(o_out).startswith(HELPER_PREFIX) or bool(acc & gone)
        if drop:
            if o_out != out and not (name == "legible" and o_out == _name_of(o.get("args", {}).get("matte"))):
                gone.add(o_out)                                  # (a legible guard that re-assigns its own matte buffer leaves the buffer as it was)
            continue
        gone.discard(o_out)
        new.append(o)
    return _dce(new, out)


def _name_of(v):
    return v[1:] if isinstance(v, str) and v.startswith("$") else None
