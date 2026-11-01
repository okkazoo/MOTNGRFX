"""recipe_eval: the compose recipe logic of piece_runner.py (Piece.layout / need_passes / recipe, _refs / _subst / _hold_index,
the `res` closure and the frame loop of _chunk) lifted into pure functions that work on in-memory frames, for the ComfyUI
nodes (nodes_fusion/compose.py). piece_runner.py itself is unchanged.

    u8, stats = run_compose(piece, "s1", type_dir=".../type", footage=u8_frames, matte=None, depth=depth_frames)

- A piece is the P*.json dict (15_fusion/pieces/P02.json) plus optional `layout`, `sizes`, `quality`.
- Frames cross the node boundary as uint8 (footage) or float 0..1 / uint8 (matte, depth) arrays [T,H,W(,C)], numpy or torch (CPU).
- Ops run through `fn.raw` (backend arrays: CUDA tensors on the torch backend), so the whole op chain of a frame stays on the
  GPU; ops without `.raw` (fill_holes) detour to numpy. One download of the finished uint8 frame per frame.
- Works on both compose backends (`backend="numpy" | "torch" | "auto"`); with the numpy backend the output matches
  piece_runner's PNGs.
"""
from __future__ import annotations

import contextlib
import copy
import math
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tools.fusion import compose
from tools.fusion import matte_align as MA
from tools.fusion import owner as OW

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

FPS = 24
BUILTIN = {"frame_idx", "t", "u", "n", "palette", "piece_frame", "footage"}


# ================================================================================================== lifted helpers
def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def refs(obj, acc):
    """collect every "$name" string (without the $) in a nested op-args structure."""
    if isinstance(obj, str) and obj.startswith("$"):
        acc.add(obj[1:])
    elif isinstance(obj, dict):
        for v in obj.values():
            refs(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            refs(v, acc)


def subst(obj, m):
    """replace whole-string values found in `m` (e.g. "$matte" -> "$s1_matte") throughout a nested structure."""
    if isinstance(obj, str):
        return m.get(obj, obj)
    if isinstance(obj, dict):
        return {k: subst(v, m) for k, v in obj.items()}
    if isinstance(obj, list):
        return [subst(v, m) for v in obj]
    return obj


def layout(piece):
    """start frame (piece time) + length of every shot; shot k starts `crossfade` before the previous ends.
    -> ({sid: {start, frames, xf}}, crossfade seconds per boundary)."""
    fps = int(piece.get("fps", FPS))
    comp = piece.get("compose") or piece.get("recipe") or {}
    cf = comp.get("crossfade", 0.5)
    shots = {s["id"]: s for s in piece["shots"]}
    ids = list(shots)
    cfl = cf if isinstance(cf, list) else [cf] * max(len(ids) - 1, 0)
    out, start = {}, 0
    for k, sid in enumerate(ids):
        n = int(shots[sid].get("span") or shots[sid]["frames"])
        c = int(round(cfl[k - 1] * fps)) if k else 0
        if k:
            start = start - c
        out[sid] = dict(start=start, frames=n, xf=c)
        start += n
    return out, cfl


def piece_layout(piece):
    """the piece's own `layout` (planner output) if present, else computed from its shots."""
    lay = piece.get("layout")
    if lay:
        cf = (piece.get("compose") or piece.get("recipe") or {}).get("crossfade", 0.5)
        ids = list(lay)
        fps = int(piece.get("fps", FPS))
        cfl = cf if isinstance(cf, list) else [cf] * max(len(ids) - 1, 0)
        if len(cfl) < len(ids) - 1:      # derive from the layout itself
            cfl = [lay[s]["xf"] / fps for s in ids[1:]]
        return {s: dict(lay[s]) for s in ids}, cfl
    return layout(piece)


def _compose_spec(piece, sid):
    comp = piece.get("compose") or piece.get("recipe") or {}
    return comp, (comp.get("shots", {}).get(sid) or comp.get("default") or {"ops": []})


KNOWN_PASSES = ("beauty", "matte", "depth", "line", "clay", "flow")       # tools/fusion/type_passes.PASSES: the passes the type stage can render
DEFAULT_PASSES = ("matte", "depth", "line", "beauty")


def pass_names(v, default=DEFAULT_PASSES, warn=None):
    """a `type_passes` value (or a control `pass`) -> a list of KNOWN pass names, whatever it is. A string must never reach `set()`: set("all") is {'a', 'l'} and the type stage asserted
    `unknown pass 'a'` (box run 22_batch-24gb v008 r01, a model-written `type_passes`). list / tuple / set: each entry as below; str: split on + , / ; and spaces; `all` / `auto` / `default` /
    empty -> `default`; `union` -> depth + line; an unknown name is dropped (reported through `warn(text)` when given); nothing left -> `default`. -> list in first-seen order."""
    items = list(v) if isinstance(v, (list, tuple, set)) else [v] if v is not None else []
    out, bad = [], []
    for it in items:
        for tok in re.split(r"[+,/;\s]+", str(it).strip().lower()) if it is not None else []:
            if not tok:
                continue
            if tok in ("all", "auto", "default", "any"):
                out += list(default)
            elif tok == "union":
                out += ["depth", "line"]
            elif tok in KNOWN_PASSES:
                out.append(tok)
            else:
                bad.append(tok)
    if bad and warn:
        warn(f"type_passes / control pass {bad} are not type passes (known: {', '.join(KNOWN_PASSES)}): dropped")
    out = list(dict.fromkeys(out))
    return out or list(default)


def norm_control_pass(name, warn=None) -> str:
    """a shot's control pass as ONE name the control reader knows: depth | line | union | matte | ... ('depth+line' -> union, 'all' / unknown -> depth)."""
    names = pass_names(name, default=("depth",), warn=warn)
    if set(names) == {"depth", "line"}:
        return "union"
    return names[0] if len(names) == 1 else ("union" if {"depth", "line"} <= set(names) else names[0])


def _ctl_names(name):
    """the type passes a shot's control pass name stands for: `union` (owner.control_pass_for) = depth + line; a string like 'depth+line' / 'all' is read tolerantly (pass_names)."""
    return tuple(pass_names(name, default=("depth",)))


def need_passes(piece):
    """type passes the piece needs: its `type_passes` + matte + whatever compose ops ($type_*) and control shots reference."""
    ps = set(pass_names(piece.get("type_passes"))) | {"matte"}
    rs = set()
    comp = piece.get("compose") or piece.get("recipe") or {}
    refs(comp.get("shots", {}), rs)
    refs(comp.get("default", {}), rs)
    for r in rs:
        m = re.match(r"type_(\w+?)(?:@-?\d+)?$", r)
        if m:
            ps.add(m.group(1))
    from tools.fusion import multiplane_recipe as MPR
    ps |= MPR.extra_passes(piece)                    # plate = multiplane: the cards composite the type pass (beauty, matte, depth) by depth
    if rs & {"type_matte", "type_line"} and comp.get("beauty_face") is not False:
        ps.add("beauty")                             # legibility_pass draws the type from the beauty pass
    for s in piece.get("shots", []):
        if s.get("kind") == "control":
            ps.update(_ctl_names(s.get("control", {}).get("pass", "depth")))
        if s.get("kind") == "motiontrack":
            ps |= {"flow", "matte"}
    return sorted(ps)


def control_passes(piece):
    """passes the LTX control video / motion tracks are cut from (what the blocking 'control' type job renders)."""
    ps = set()
    for s in piece.get("shots", []):
        if s.get("kind", "control") == "control":
            ps.update(_ctl_names(s.get("control", {}).get("pass", "depth")))
        if s.get("kind") == "motiontrack":
            ps |= {"flow", "matte"}
    return sorted(ps | {"matte"})


def hold_index(recipe, nm, i, po, off):
    """freeze-frames (recipe["holds"] = [[t_s, dur_s], ..] in piece time). Footage-like inputs stop dead for the hold and resume
    where they stopped; type-pass inputs (dir) hold the frame of the hold start and then jump back onto the music grid
    (holds_type, default true). Signoff inputs are untouched."""
    sp = recipe["inputs"][nm]
    if "signoff" in sp:
        return i + off
    fps = recipe["fps"]
    hs = [(int(round(float(a) * fps)), int(round(float(b) * fps))) for a, b in recipe["holds"]]
    p = i + po
    if "dir" in sp:
        if recipe.get("holds_type", True):
            for h0, hn in hs:
                if h0 <= p < h0 + hn:
                    return h0 + (off - po)
        return i + off
    fz = lambda q: sum(min(max(q - h0, 0), hn) for h0, hn in hs)  # noqa: E731
    return i + off - (fz(p) - fz(po))


def resolve(v, ctx, curves, u, pf, fps):
    """op-argument value -> concrete value ($buffer, {anim}, {timeline}, {curve}, lists). pf = piece frame."""
    if isinstance(v, str) and v.startswith("$"):
        return ctx[v[1:]]
    if isinstance(v, dict) and "anim" in v:
        return compose._anim(v, u)
    if isinstance(v, dict) and "timeline" in v:
        val = compose._timeline(v, pf / fps)
        hf = v.get("hold_floor")                     # legibility_pass: never below `value` inside the word holds (ramped)
        if hf and not isinstance(val, (list, tuple)):
            val = max(float(val), float(hf.get("value", 1.0)) * max((compose._leg_weight(pf / fps, a, b, float(hf.get("ramp", 0.25)))
                                                                    for a, b in hf["holds"]), default=0.0))
        return val
    if isinstance(v, dict) and "cam" in v:           # camera track of the footage flow (camera_track), px at compose size
        xy = ctx["__cam__" + str(v.get("ref", "$flow")).lstrip("$")]
        return float(xy[0 if str(v["cam"]).lower() == "x" else 1]) * float(v.get("gain", 1.0))
    if isinstance(v, dict) and "curve" in v:
        c = curves[v["curve"]]
        return float(v.get("lo", 0.0) + (v.get("hi", 1.0) - v.get("lo", 0.0)) * c[min(max(pf, 0), len(c) - 1)])
    if isinstance(v, list):
        return [resolve(x, ctx, curves, u, pf, fps) for x in v]
    return v


def depth_norm_frames(gray_frames, n=10):
    """(lo, hi) = 2nd / 98th percentile of the clip's depth values (VDA output only spans a slice of 0..1 per clip).
    `gray_frames`: sequence of HxW float32 numpy arrays (as piece_runner._depth_norm, but in memory)."""
    idx = np.linspace(0, len(gray_frames) - 1, min(n, len(gray_frames))).round().astype(int)
    v = np.concatenate([np.asarray(gray_frames[i], np.float32)[::4, ::4].ravel() for i in idx])
    lo, hi = np.percentile(v, [2, 98])
    return [float(lo), float(max(hi, lo + 1e-3))]


def camera_track(flow, size, smooth=5):
    """Camera track from the footage flow ([T, h, w, 2], px/frame of the flow grid, x right / y down, flow[t] = frame t -> t+1).
    Per frame the global drift (median flow, subsampled) is scaled to the compose size (W/w, H/h) and summed over the frames:
    track[t] = scene displacement of frame t relative to frame 0 (track[0] = 0), px at compose size. `smooth` = odd moving-average
    window (frames) applied to the per-frame drift to kill jitter (1/0 = off). -> float32 [T, 2] (x, y). Adding track[t] to a
    layer (translate dx, dy) makes it ride the scene: constant +2 px/frame flow at flow res -> +2*(W/w) px/frame to the right."""
    n = int(flow.shape[0])
    fh, fw = int(flow.shape[1]), int(flow.shape[2])
    sx, sy = size[0] / float(fw), size[1] / float(fh)
    v = np.zeros((n, 2), np.float64)
    for t in range(n):
        x = flow[t]
        x = (x.cpu().numpy() if _is_torch(x) else np.asarray(x)).astype(np.float32)
        v[t] = np.median(x[::4, ::4].reshape(-1, 2), axis=0)
    v[:, 0] *= sx
    v[:, 1] *= sy
    k = int(smooth) if smooth else 0
    if k > 1 and n > 2:
        k = min(k, n if n % 2 else n - 1)
        r = k // 2
        pad = np.pad(v, ((r, r), (0, 0)), mode="edge")
        ker = np.ones(k) / k
        v = np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(2)], 1)
    c = np.zeros((n, 2), np.float64)
    c[1:] = np.cumsum(v[:-1], axis=0)          # frame 0 = 0; step t -> t+1 is the flow of frame t
    return c.astype(np.float32)


def cam_ops(layers=("type_a", "type_rgb"), gain=1.0, ref="$flow", prefix="cam_"):
    """`translate` op dicts that make type layers follow the scene (the footage camera track), to prepend to a shot's ops.
    `layers` = buffer names without "$" (e.g. ("type_matte", "type_line", "type_beauty") or earlier op outputs); each becomes
    {"op": "translate", "out": prefix + layer, "args": {"img": "$"+layer, "dx": {"cam": "x", "gain": gain, "ref": ref}, "dy": {...}}},
    so later ops read "$cam_<layer>" instead of "$<layer>" (prefix="" moves a buffer in place: only valid for op outputs, not for
    $type_* inputs). `ref` = flow buffer ("$flow" or "$<sid>_flow"). The shot then needs the footage flow (flow input)."""
    out = []
    for lay in layers:
        out.append({"op": "translate", "out": f"{prefix}{lay}",
                    "args": {"img": f"${lay}", "dx": {"cam": "x", "gain": gain, "ref": ref},
                             "dy": {"cam": "y", "gain": gain, "ref": ref}}})
    return out


def _cam_refs(ops):
    """names of the flow buffers referenced by {"cam": ..} args in an op list."""
    acc = set()

    def walk(o):
        if isinstance(o, dict):
            if "cam" in o:
                acc.add(str(o.get("ref", "$flow")).lstrip("$"))
            else:
                for x in o.values():
                    walk(x)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
    walk([op.get("args", {}) for op in ops])
    return acc


# ================================================================================================== legibility guard
def owner_windows(piece, type_dir=None):
    """per word {start, end, render, ltx_at, ..} (tools/fusion/owner.py) from the type pass manifest (the TRUE timeline), else the words' t0 / t1; None for a piece
    without owner fields (OLD pieces: every word flat, compose unchanged)."""
    if not OW.has_owner((piece.get("type") or {}).get("words")):
        return None
    rows = None
    try:
        if type_dir is not None and (Path(type_dir) / "manifest.json").is_file():
            import json
            rows = OW.manifest_words(json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8"))) or None
    except Exception:
        rows = None
    return OW.windows(piece, rows)


def PC_clip_holds(piece, holds):
    from tools.fusion import pivot_cut as PC
    return PC.clip_holds_to_cuts(piece, holds)


def piece_word_holds(piece, type_dir=None):
    """[(hit_s, release_s)] of every word, piece seconds: the windows where the word must READ as flat type. Words LTX owns are not in it (an `ltx` word is
    dropped, a `handoff` word's release stops at its handoff time), so the legibility guard / hold net / colour stage never try to resurrect type that
    compose deliberately does not draw. See _piece_word_holds for the sources."""
    h = PC_clip_holds(piece, _piece_word_holds(piece, type_dir))             # a word owned by a promoted cut is not redrawn inside the cut's window (pivot_cut.clip_holds_to_cuts)
    wins = owner_windows(piece, type_dir)
    if wins is None or len(wins) != len(h):
        return h
    out = []
    for w, (a, b) in zip(wins, h):
        until = OW.flat_visible_until(w)
        if until == float("inf"):
            out.append((a, b))
        elif until > a + 0.05:
            out.append((a, min(b, until)))
    return out


def _piece_word_holds(piece, type_dir=None):
    """[(hit_s, release_s)] of every word, piece seconds: the windows where the word must read. Source of truth first: what the type
    pass actually rendered (<type_dir>/manifest.json info.words hit / release); else the piece's own plan (plan.flow_wake.holds,
    written by plan_rules.weave_flow); else plan_rules.word_holds from piece["type"]. [] when the piece has no word timing (the guard
    then does nothing)."""
    try:
        if type_dir is not None and (Path(type_dir) / "manifest.json").is_file():
            import json
            ws = ((json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8")).get("info") or {}).get("words")) or []
            h = [(float(w["hit"]), float(w["release"])) for w in ws if "hit" in w and "release" in w]
            if h:
                return h
    except Exception:
        pass
    try:
        h = ((piece.get("plan") or {}).get("flow_wake") or {}).get("holds")
        if h:
            return [(float(a), float(b)) for a, b in h]
        if (piece.get("type") or {}).get("words"):
            from tools.fusion import plan_rules
            return [(float(a), float(b)) for a, b in plan_rules.word_holds(piece, None)]
    except Exception:
        pass
    return []


def guard_ops(piece, ops, type_dir=None):
    """Legibility guard (user rule: a matte never hides a word's letters for the whole hold). Fills `holds` of every explicit
    `legible` op, and inserts one before each `sandwich` (matte -> guarded matte) and `depth_occlude` (depth_cover -> guarded cover)
    that does not already read a legible output. Holds: see piece_word_holds. Opt out with compose.legible = false (piece) or legible = false (op args).
    No word holds in the piece -> ops returned untouched.
    HARD FLOOR (2026-11-01, compose.LEG_FLOOR): every subject-over-type draw gets the word-level floor, also a sandwich with legible = false or a piece
    with compose.legible = false (then the per-letter rule is off, `letters` false) and the `over` pairs of the pure-type route (type drawn by an `over`,
    then the footage put back over it by the subject matte): a word the subject hides by > 25 % for most of its hold comes in front whole. Only
    compose.legible_floor = false turns the floor off."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    floor_on = comp.get("legible_floor") is not False
    if comp.get("legible") is False and not floor_on:
        return ops
    holds = piece_word_holds(piece, type_dir)
    if not holds:
        return ops
    fps = int(piece.get("fps", FPS))
    hl = [[round(a, 4), round(b, 4)] for a, b in holds]
    fl = {"floor": dict(compose.LEG_FLOOR)} if floor_on else {}
    if floor_on and comp.get("behind_words") is not False:         # the behind quota (compose.LEG_BEHIND): about half of the words the subject overlaps stay behind it, in every mode; compose.behind_words = false turns it off
        fl["behind"] = dict(compose.LEG_BEHIND)
    letters_on = comp.get("legible") is not False
    outs = {}                      # buffer name -> the op that made it last (the over-pair test)

    def matte_like(v):
        """a $ref that is the subject matte: the shot's matte input or an upscale / hole fill / guard of it."""
        nm = v[1:] if isinstance(v, str) and v.startswith("$") else None
        for _ in range(4):
            if nm is None:
                return False
            if nm in ("matte", "subject_matte") or nm.endswith("_matte") and nm not in ("type_matte",) and not nm.startswith("type_"):
                return True
            o = outs.get(nm)
            if o is not None and o["op"] == "luminosity_key":     # 2026-11-02: a luminosity key read as the subject (a design's `over(footage, alpha = luminosity_key)`) is guarded like a matte
                return True
            if o is None or o["op"] not in ("upscale_matte", "fill_holes", "legible", "blur"):
                return False
            src = o.get("args", {}).get("matte", o.get("args", {}).get("mask", o.get("args", {}).get("img")))
            nm = src[1:] if isinstance(src, str) and src.startswith("$") else None
        return False

    def mirror(args):
        d = {}
        for k in ("matte_threshold", "matte_soft", "matte_gamma", "invert"):
            v = args.get(k)
            if isinstance(v, (int, float, bool)):
                d[k] = v
        if isinstance(args.get("front"), str) and args["front"].startswith("$"):
            d["front"] = args["front"]
        return d

    out, made = [], set()          # made: buffer names whose latest value came out of a legible op
    for k, op in enumerate(ops):
        args = op.get("args", {})
        name = op["op"]
        if name == "legible":
            if not args.get("holds"):
                args["holds"] = copy.deepcopy(hl)
            args.setdefault("t", "$t")
            args.setdefault("fps", fps)
            for kk, vv in fl.items():
                args.setdefault(kk, copy.deepcopy(vv))
            out.append(op)
            made.add(op.get("out", "image"))
            outs[op.get("out", "image")] = op
            continue
        want = name in ("sandwich", "depth_occlude") and args.get("legible", True) is not False and letters_on
        if name == "sandwich" and (want or fl):
            ta, m = args.get("type_a"), args.get("matte")
            if isinstance(ta, str) and isinstance(m, str) and m.startswith("$") and m[1:] not in made:
                g = f"_leg{k}"
                out.append({"op": "legible", "out": g, "args": dict(type_a=ta, matte=m, holds=copy.deepcopy(hl), t="$t", fps=fps,
                                                                    **mirror(args), **copy.deepcopy(fl), **({} if want else {"letters": False}))})
                args["matte"] = "$" + g
        elif name == "over" and fl and isinstance(args.get("base"), str) and args["base"][1:] in outs and matte_like(args.get("alpha")) \
                and args["alpha"][1:] not in made and isinstance(args.get("fg"), (str, list)):
            prev = outs[args["base"][1:]]          # the pure-type pair: over(type) -> over(subject by its matte) on top of it
            pa = prev.get("args", {})
            if prev["op"] == "over" and isinstance(pa.get("alpha"), str) and pa["alpha"].startswith("$") and not matte_like(pa["alpha"]):
                g = f"_leg{k}"
                out.append({"op": "legible", "out": g, "args": dict(type_a=pa["alpha"], matte=args["alpha"], holds=copy.deepcopy(hl), t="$t", fps=fps,
                                                                    letters=False, **copy.deepcopy(fl))})
                args["alpha"] = "$" + g
        elif name == "window_cut" and fl and args.get("subject", "outside") != "none" and isinstance(args.get("type_a"), str):
            for key in ("fg_matte", "matte_a", "matte_b"):     # the subject stepping out over the letter windows (QC: "HOT WITH" second line lost in the crowd)
                m = args.get(key)
                if isinstance(m, str) and m.startswith("$") and m[1:] not in made:
                    g = f"_leg{k}{key[-2:]}"
                    out.append({"op": "legible", "out": g, "args": dict(type_a=args["type_a"], matte=m, holds=copy.deepcopy(hl), t="$t", fps=fps,
                                                                        letters=False, **copy.deepcopy(fl))})
                    args[key] = "$" + g
        elif name == "depth_occlude" and want:
            ta, dp = args.get("type_a"), args.get("footage_depth")
            if isinstance(ta, str) and isinstance(dp, str) and args.get("cover") is None:
                cv_, g = f"_cov{k}", f"_leg{k}"
                cargs = dict(footage_depth=dp)
                for key in ("plane", "softness"):
                    if key in args:
                        cargs[key] = copy.deepcopy(args[key])
                out.append({"op": "depth_cover", "out": cv_, "args": cargs})
                out.append({"op": "legible", "out": g, "args": dict(type_a=ta, matte="$" + cv_, holds=copy.deepcopy(hl), t="$t", fps=fps, **copy.deepcopy(fl))})
                args["cover"] = "$" + g
        out.append(op)
        made.discard(op.get("out", "image"))
        outs[op.get("out", "image")] = op
    return out


def floor_ops(piece, ops, type_dir=None):
    """READABLE-SIZE FLOOR (2026-11-01, QC: hairline UNLEASH, landscape words under the 9 % cap floor): every `type_face` op of the recipe (the
    legibility pass head, the pure-type route's own head) gets the word holds + compose.type_floor (cap TF_MIN_CAP, stroke TF_MIN_STROKE), so the
    face is scaled / thickened at run time inside the holds whatever wrote the design. Off: compose.legible_floor = false."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    if comp.get("legible_floor") is False or not any(o["op"] == "type_face" for o in ops):
        return ops
    holds = piece_word_holds(piece, type_dir)
    if not holds:
        return ops
    hl = [[round(a, 4), round(b, 4)] for a, b in holds]
    for o in ops:
        if o["op"] == "type_face":
            a = o.setdefault("args", {})
            a.setdefault("holds", copy.deepcopy(hl))
            a.setdefault("t", "$t")
            a.setdefault("min_stroke", compose.TF_MIN_STROKE)
            a.setdefault("min_cap", BOOST_MIN_CAP if user_style(piece)["legible_boost"] else compose.TF_MIN_CAP)
    return ops


def _lit3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)


def style_ops(piece, ops, sid="s1"):
    """2026-11-01 colour review, run time for every design source: (1) SHOT TONES: a `fill` with a literal colour that is later the `base` of the chain (a flat
    ground / plate section: flat_ground_window, contrast moments) becomes `shot_tone` (the ground's lightness, the shot's top hue), and a literal-colour `over`
    drawn on that ground (the inverse type) takes the shot's far hue at >= 4.5:1 against the new ground (legible_color.tone_pair); the user's palette wins.
    (2) the user asked for legible / plain text: every flow_wake is gated off (no wake / echo). Opt-out of (1): compose.shot_colour = false."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    us = user_style(piece)
    ops = copy.deepcopy(ops)
    if us["plain"]:
        for o in ops:
            if o["op"] == "flow_wake":
                o.setdefault("args", {})["gate"] = 0.0
    if comp.get("shot_colour") is False:
        return ops
    used_as_base = {}
    for j, o in enumerate(ops):
        b = (o.get("args") or {}).get("base")
        if isinstance(b, str):
            used_as_base.setdefault(b[1:], []).append(j)
    upal = {"user_palette": [[float(v) / 255.0 for v in c] for c in us["palette"]]} if us["palette"] else {}
    ground_of, out, k = {}, [], 0
    pid = str(piece.get("id", "piece"))
    for j, o in enumerate(ops):
        a = o.get("args") or {}
        nm = o.get("out", "image")
        if o["op"] == "fill" and not o.get("lc") and _lit3(a.get("color")) and any(i > j for i in used_as_base.get(nm, [])):      # lc: the legible-colour stage's own layout fill
            g = [float(x) for x in a["color"]]
            out.append({"op": "shot_tone", "out": nm, "tone": True, "args": {"like": a.get("like", "$footage"), "plate": "$footage", "ground": g, "part": "ground",
                                                                         "state_key": f"{pid}:{sid}:tone", **upal}})
            ground_of[nm] = g
            continue
        if o["op"] == "over" and _lit3(a.get("fg")) and isinstance(a.get("base"), str) and a["base"][1:] in ground_of:
            tn = f"_tone{k}"
            k += 1
            out.append({"op": "shot_tone", "out": tn, "tone": True, "args": {"like": a["base"], "plate": "$footage", "ground": ground_of[a["base"][1:]],
                                                                         "ink": [float(x) for x in a["fg"]], "part": "ink", "state_key": f"{pid}:{sid}:tone", **upal}})
            o = copy.deepcopy(o)
            o["args"]["fg"] = "$" + tn
        elif nm in ground_of and not (isinstance(a.get("base"), str) and a["base"][1:] == nm):
            ground_of.pop(nm, None)                                     # the buffer is re-defined from something else: no longer that ground
        out.append(o)
    return out


def _closure(ops, k, names):
    """ops[:k] reduced to what is needed to compute the buffers `names` (backward walk by name, honouring re-definitions)."""
    need, keep = set(names), []
    for j in range(k - 1, -1, -1):
        o = ops[j]
        if o.get("out", "image") in need:
            need.discard(o.get("out", "image"))
            keep.append(j)
            acc = set()
            refs([o.get("args", {})], acc)
            need |= acc
    return [ops[j] for j in reversed(keep)], need


def _arg_name(v):
    return v[1:] if isinstance(v, str) and v.startswith("$") else None


# ================================================================================================== legibility pass (v005, 2026-10-27)
# USER RULE: the visible type is ALWAYS the beauty pass (alpha = the type matte); treatments only modulate it. The planner's chains draw the
# type with `type_rgb = $tcol` (a flat colour sampled from the plate) and an alpha that may be the LINE pass crossfaded into the matte, which
# turned wireframes / extrusions into flat slabs. `legibility_pass` rewrites a shot's ops at build time (so planner output already baked into
# the piece JSONs gets it too):
#   1 head ops: `type_face` (beauty + matte -> $tface colour, $tmat alpha; a matte that contradicts the beauty is rebuilt from the beauty),
#     every `$type_matte` reference becomes `$tmat`
#   2 the colour layer follows the alpha through the same warps (flow_wake twin with type_rgb, rigid_pose / translate twins, mask ops) -> face_of(alpha)
#   3 sandwich / depth_occlude / `over(fg=$tcol)` of a type alpha draw `face_of(alpha)`; a drawer's alpha never comes from the line pass; the main
#     drawer's `type_opacity` ramp is floored to 1.0 inside the word holds (`hold_floor`, resolved by `resolve`); every `knockout` (window) gets
#     min_contrast + the face (compose.knockout)
#   4 `hold_net` before the closing grade / grain / vignette (compose.hold_net: the word region blends toward the plain beauty when it does not read)
LP_LINEISH = ("$type_line", "$wl")
LP_TAIL = ("grade", "grain", "vignette")
LP_FLOOR_RAMP = 0.25
LP_MIN_CONTRAST = 0.22


def _beauty_bg(piece, type_dir):
    """background colour (0..255 RGB) of the beauty pass: the type scene's palette (manifest), else the piece's own palette, else None."""
    try:
        import json as _json
        mp = Path(type_dir) / "manifest.json" if type_dir is not None else None
        if mp is not None and mp.is_file():
            from tools.fusion import spec as S_
            sp = _json.loads(mp.read_text(encoding="utf-8")).get("spec") or {}
            return [float(x) for x in S_.make_palette(sp.get("palette")).bg]
    except Exception:
        pass
    try:
        bg = ((piece.get("type") or {}).get("palette") or {}).get("bg")
        if bg:
            return [float(x) for x in bg][:3]
    except Exception:
        pass
    return None


def _name(v):
    return v[1:] if isinstance(v, str) and v.startswith("$") else None


def _floor_opacity(v, holds):
    """type_opacity (number | timeline dict) -> the same with a `hold_floor` (1.0 inside the holds, ramped); None = leave it (a gated
    treatment window that returns to 0 at its end)."""
    spec = {"holds": copy.deepcopy(holds), "value": 1.0, "ramp": LP_FLOOR_RAMP}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return None if float(v) >= 0.999 else {"timeline": [[0.0, float(v)], [100000.0, float(v)]], "hold_floor": spec}
    if isinstance(v, dict) and "timeline" in v and "hold_floor" not in v:
        ks = v["timeline"]
        if ks and not isinstance(ks[-1][1], (list, tuple)) and float(ks[-1][1]) > 0.3:
            return dict(v, hold_floor=spec)
    return None


def _norm_text(x):
    return re.sub(r"\s+", " ", str(x or "")).strip().upper()


def color_plan(piece, type_dir=None):
    """(on, words, texts) of the legible-colour stage (tools/fusion/legible_color.py). on = False when the piece opts out (`"color": "fixed"` in
    compose / the piece / piece.type, or compose.legible_color = false). words = [[start, hit, release, end, fixed], ..] piece seconds, one per word:
    from the type pass manifest (info.words), else from piece_word_holds; fixed = 1 when that word has `"color": "fixed"` in piece["type"]["words"]
    (matched by index when the lists are as long as each other, else by text)."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    typ = piece.get("type") or {}
    if comp.get("legible_color") is False or "fixed" in (comp.get("color"), piece.get("color"), typ.get("color")):
        return False, [], []
    rows = []
    try:
        if type_dir is not None and (Path(type_dir) / "manifest.json").is_file():
            import json as _json
            ws = ((_json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8")).get("info") or {}).get("words")) or []
            rows = [(w.get("text"), float(w.get("start", w["hit"])), float(w["hit"]), float(w["release"]), float(w.get("end", w["release"])))
                    for w in ws if "hit" in w and "release" in w]
    except Exception:
        rows = []
    if not rows:
        rows = [(None, a - 0.4, a, b, b + 0.4) for a, b in piece_word_holds(piece, type_dir)]
    pw = [w for w in (typ.get("words") or []) if isinstance(w, dict)]
    fixed = [0] * len(rows)
    if any(w.get("color") == "fixed" for w in pw):
        if len(pw) == len(rows):
            fixed = [1 if w.get("color") == "fixed" else 0 for w in pw]
        else:
            bad = {_norm_text(w.get("text") or " ".join(w.get("lines") or [])) for w in pw if w.get("color") == "fixed"}
            fixed = [1 if _norm_text(r[0]) in bad else 0 for r in rows]
    wins = owner_windows(piece, type_dir)
    if wins is not None and len(wins) == len(rows):          # words LTX owns are not tracked by the colour stage; a handoff word ends at its handoff
        keep = []
        for w, r, f in zip(wins, rows, fixed):
            until = OW.flat_visible_until(w)
            if until == float("inf"):
                keep.append((r, f))
            elif until > r[2] + 0.05:
                keep.append(((r[0], r[1], r[2], min(r[3], until), min(r[4], until)), f))
        rows, fixed = [k[0] for k in keep], [k[1] for k in keep]
    words = [[round(r[1], 4), round(r[2], 4), round(r[3], 4), round(r[4], 4), f] for r, f in zip(rows, fixed)]
    return True, words, [r[0] if r[0] is not None else i for i, r in enumerate(rows)]


def _outline_windows(piece, holds) -> list:
    """windows [[t0, t1], ..] (piece seconds) where the type outline is forced on: the piece system's drawn stand-out outline (random_system.draw_outline: at SOME words or hits), else none.
    `compose.outline` (a spec dict, or a list of windows) overrides it; compose.outline = false switches it off."""
    from tools.fusion import random_system as RSY
    comp = piece.get("compose") or {}
    o = comp.get("outline")
    if o is False:
        return []
    if isinstance(o, list):
        return [[float(a), float(b)] for a, b in o]
    spec = o if isinstance(o, dict) else ((piece.get("plan") or {}).get("piece_system") or {}).get("outline")
    if not spec:
        return []
    seed = ((piece.get("plan") or {}).get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0))
    return RSY.outline_windows(spec, holds, (piece.get("type") or {}).get("bpm") or (piece.get("audio") or {}).get("bpm"), seed)


def _ink_palette(piece) -> list:
    """the palette colours (type.palette fg + accents, plan.still_palette colours) as sRGB 0..1 triples: the inks legible_color tries before it shifts the lightness of the designed hue."""
    typ = piece.get("type") or {}
    pal = typ.get("palette") or {}
    raw = ([pal.get("fg")] + list(pal.get("accents") or []) + list(((piece.get("plan") or {}).get("still_palette") or {}).get("colors") or []))
    out = []
    for c in raw:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                v = [float(x) for x in c[:3]]
            except (TypeError, ValueError):
                continue
            v = [round(x / 255.0 if max(v) > 1.0 else x, 4) for x in v]
            if v not in out:
                out.append(v)
    return out


from tools.fusion.legible_color import LC_ECHO_ALPHA, ECHO_MODULES as LC_ECHO_MODULES        # noqa: E402  the echo opacity behind the front copy; modules that draw echo copies
OUTLINE_W = (0.25, 0.7)       # 2026-11-01 user "vary the outline weightings": text outline width multiplier range, drawn per word; then "these all have quite thick outlines" -> thinner; "they should be finer strokes" -> 0.25-0.7 (was 0.6-2.2)
OUTLINE_W_STEP = 0.15         # consecutive words differ by at least this (was 0.45)
GLINT_AMP = 0.4               # hold glint opacity (legible_color life): subtle
BOOST_MIN_CAP = 0.12          # user asked for legible / bigger text: the readable-size floor's cap height (TF_MIN_CAP is 0.09)


def _seed_of(piece) -> int:
    import zlib
    return zlib.crc32(str(((piece.get("plan") or {}).get("piece_system") or {}).get("seed", piece.get("id", "piece"))).encode())


def outline_weights(piece, n) -> list:
    """per-word text-outline width multipliers (the stroke AROUND the letters, legible_color support / glint; never the subject contour lines), seeded by the
    piece: from thin to bold within OUTLINE_W, consecutive words at least OUTLINE_W_STEP apart. A re-render is exact."""
    import random
    rng = random.Random(_seed_of(piece) ^ 0x5EED)
    out = []
    for _ in range(max(int(n), 0)):
        for _try in range(12):
            w = round(rng.uniform(*OUTLINE_W), 2)
            if not out or abs(w - out[-1]) >= OUTLINE_W_STEP:
                break
        out.append(w)
    return out


def user_style(piece) -> dict:
    """the user's explicit style wishes (tools/fusion/user_style.parse_direction): palette / legible_boost / plain. Cached on nothing: pure."""
    from tools.fusion import user_style as US
    return US.parse_direction(piece)


def _beats_of(piece, type_dir) -> list:
    """the piece's beat times (s): the type manifest's spec beats, else type.beats, else a bpm grid over the cut."""
    try:
        if type_dir is not None and (Path(type_dir) / "manifest.json").is_file():
            import json as _json
            b = (_json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8")).get("spec") or {}).get("beats")
            if b:
                return [round(float(x), 4) for x in b]
    except (OSError, ValueError):
        pass
    b = (piece.get("type") or {}).get("beats")
    if b:
        return [round(float(x), 4) for x in b]
    bpm = float((piece.get("type") or {}).get("bpm") or (piece.get("audio") or {}).get("bpm") or 0)
    cut = float((piece.get("audio") or {}).get("cut") or 0)
    return [round(k * 60.0 / bpm, 4) for k in range(int(cut * bpm / 60.0) + 1)] if bpm > 0 and cut > 0 else []


def legibility_pass(piece, ops, type_dir=None, sid="s1"):
    """see the block comment above. Returns `ops` itself when the piece opts out (compose.beauty_face = false), is a `traced` piece, or no
    op draws the type matte."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    if comp.get("beauty_face") is False or piece.get("technique") == "traced":
        return ops
    if type_dir is None or not (Path(type_dir) / "beauty").is_dir():       # no beauty pass to draw from (synthetic / partial type dirs)
        return ops
    rs = set()
    refs([o.get("args", {}) for o in ops], rs)
    if "type_matte" not in rs and "type_line" not in rs:
        return ops
    if not any(o["op"] in ("sandwich", "depth_occlude") and isinstance(o.get("args", {}).get("type_rgb"), str) for o in ops):
        return ops
    holds = piece_word_holds(piece, type_dir)
    hl = [[round(a, 4), round(b, 4)] for a, b in holds]
    lc_on, lc_words, _lc_texts = color_plan(piece, type_dir)
    fps_ = int(piece.get("fps", FPS))
    outline_w = _outline_windows(piece, holds)
    ink_pal = _ink_palette(piece)
    soft_ok = bool((piece.get("compose") or {}).get("soft_support"))      # soft shadow / scrim behind the type: only when asked for (effect `soft_support`)
    # 2026-11-01 colour review (defaults for every piece; the user's own direction wins): shot-palette inks, complementary support, per-word outline weight,
    # hold glint, crisp front copy over a subdued wake. Opt-outs: compose.shot_colour = false, compose.hold_glint = false, compose.echo_front = false.
    us = user_style(piece)
    shot_on = comp.get("shot_colour") is not False
    try:
        _lay, _ = piece_layout(piece)
        shot_t0 = round(float(_lay[sid]["start"]) / float(piece.get("fps", FPS)), 3) if sid in _lay else None
    except Exception:          # noqa: BLE001  synthetic pieces without a layout: no variety memory
        shot_t0 = None
    ow_list = outline_weights(piece, len(lc_words))
    glint = None if (us["plain"] or comp.get("hold_glint") is False) else {"beats": _beats_of(piece, type_dir), "amp": GLINT_AMP}
    front_on = comp.get("echo_front") is not False
    wake_front = {}                                             # wake-drawn alpha buffer -> (crisp front alpha buffer, its face buffer)
    front_swap = []                                             # (wake buffer, legible_color alpha buffer, index in `new`)
    ops = copy.deepcopy(ops)
    ops = [subst(o, {"$type_matte": "$tmat"}) for o in ops]
    head = [{"op": "type_face", "out": "_tf4", "args": {"beauty": "$type_beauty", "type_a": "$type_matte", "bg": _beauty_bg(piece, type_dir)}},
            {"op": "rgba_part", "out": "tmat", "args": {"rgba": "$_tf4", "part": "a"}},
            {"op": "rgba_part", "out": "tface", "args": {"rgba": "$_tf4", "part": "rgb"}}]
    fm = {"tmat": "tface"}                                      # alpha buffer -> its colour buffer
    new, drawers = [], []
    face_of = lambda nm: fm.get(_name(nm) or "", "tface")      # noqa: E731
    lc_made, ver = {}, {}

    def recolour(a):
        """legible_color in front of a type drawer: the drawn face (and alpha, extended by the support) come from the measured background.
        -> (colour ref, alpha ref) for the net: the recoloured face and the plain alpha."""
        if not lc_on or not isinstance(a.get("type_a"), str) or not isinstance(a.get("type_rgb"), str):
            return a.get("type_rgb"), a.get("type_a")
        j = len(lc_made)
        if front_on and _name(a["type_a"]) in wake_front:              # later drawers of the same wake read the front + subdued echo alpha (cheap, stateless: screen(front, wake x echo))
            new.append({"op": "mask_gate", "out": f"_eb{j}", "args": {"a": a["type_a"], "gate": 0.0 if us["plain"] else LC_ECHO_ALPHA}})
            new.append({"op": "screen", "out": f"_ef{j}", "args": {"a": "$" + wake_front[_name(a["type_a"])][0], "b": f"$_eb{j}"}})
            front_swap.append((_name(a["type_a"]), f"_ef{j}", len(new)))
        plate = a["footage"] if isinstance(a.get("footage"), str) else "$footage"
        sig = (a["type_rgb"], ver.get(_name(a["type_rgb"]), 0))
        if sig in lc_made:           # a later drawer of the same face (the front re-draw over the subject, a window pass): SAME colour as the first, own alpha
            a["type_rgb"] = lc_made[sig][0]
            return lc_made[sig]
        new.append({"op": "legible_color", "out": f"_lc{j}", "lc": True, "args": {
            "face": a["type_rgb"], "alpha": a["type_a"], "plate": plate, "t": "$t", "frame_idx": "$frame_idx", "u": "$u", "fps": fps_,
            "words": copy.deepcopy(lc_words), "state_key": f"{piece.get('id', 'piece')}:{sid}:lc{j}", "mode": "layer", **({"outline": outline_w} if outline_w else {}),
            **({"ink_palette": ink_pal} if ink_pal else {}),
            **({"soft_support": True} if soft_ok else {}),
            **({"shot_palette": True, "piece_key": str(piece.get("id", "piece")), "shot_t0": shot_t0} if shot_on else {}),
            **({"user_palette": [[float(v) / 255.0 for v in c] for c in us["palette"]]} if us["palette"] else {}),
            **({"boost": True} if us["legible_boost"] else {}),
            **({"echo_alpha": 0.0} if us["plain"] else {}),
            **({"outline_w": ow_list} if ow_list else {}),
            **({"life": glint} if glint and glint["beats"] else {}),
            **({"echo_face": True} if front_on and any((w or {}).get("module") in LC_ECHO_MODULES for w in (piece.get("type") or {}).get("words") or []) else {}),
            **({"front": "$" + wake_front[_name(a["type_a"])][0], "front_face": "$" + wake_front[_name(a["type_a"])][1]}
               if front_on and _name(a["type_a"]) in wake_front else {})}})
        new.append({"op": "rgba_part", "out": f"lcrgb{j}", "lc": True, "args": {"rgba": f"$_lc{j}", "part": "rgb"}})
        new.append({"op": "rgba_part", "out": f"lca{j}", "lc": True, "args": {"rgba": f"$_lc{j}", "part": "a"}})
        ref = (f"$lcrgb{j}", a["type_a"])
        lc_made[sig] = ref
        a["type_rgb"], a["type_a"] = f"$lcrgb{j}", f"$lca{j}"
        return ref
    for o in ops:
        name, out, a = o["op"], o.get("out", "image"), o.setdefault("args", {})
        if name == "flow_wake" and _name(a.get("type_a")) in fm and "type_rgb" not in a:
            new.append(o)
            tw = copy.deepcopy(o)                                # twin: same displacement, the colour layer warped with it (own state)
            tw["out"], tw["args"]["type_rgb"] = f"_{out}4", "$" + fm[_name(a["type_a"])]
            tw["args"]["state_key"] = f"{a.get('state_key', 'default')}:face"
            fo = f"{out}_face"
            new += [tw, {"op": "rgba_part", "out": fo, "args": {"rgba": f"$_{out}4", "part": "rgb"}}]
            wake_front[out] = wake_front.get(_name(a["type_a"]), (_name(a["type_a"]), fm[_name(a["type_a"])]))
            fm[out] = fo
            continue
        if name in ("rigid_pose", "translate", "hold_life") and _name(a.get("img")) in fm:        # hold_life (tools/fusion/hold_life.py): the twin on the face reads the same `ref` alpha
            new.append(o)
            src = fm[_name(a["img"])]
            tw = copy.deepcopy(o)
            tw["args"]["img"] = "$" + src
            tw["out"] = src if out == _name(a["img"]) else f"{out}_face"
            new.append(tw)
            fm[out] = tw["out"]
            wf = wake_front.get(_name(a["img"]))
            if wf is not None and front_on:                     # the crisp front copy gets the same pose / hold life (same `ref`), alpha and face
                f_a, f_f = copy.deepcopy(o), copy.deepcopy(o)
                f_a["out"], f_a["args"]["img"] = f"{out}_front", "$" + wf[0]
                f_f["out"], f_f["args"]["img"] = f"{out}_frontface", "$" + wf[1]
                new += [f_a, f_f]
                wake_front[out] = (f"{out}_front", f"{out}_frontface")
            continue
        if name == "crossfade":
            if a.get("a") in LP_LINEISH and _name(a.get("b")) in fm:        # ta = line -> matte: the line pass is never the visible type
                a["a"] = a["b"]
            fa_, fb_ = wake_front.get(_name(a.get("a"))), wake_front.get(_name(a.get("b")))
            if fa_ is not None and (fb_ == fa_ or _name(a.get("b")) == _name(a.get("a"))):
                wake_front[out] = fa_                               # a crossfade between wakes of the same front: the same front
            for k in ("b", "a"):
                if _name(a.get(k)) in fm:
                    fm[out] = fm[_name(a[k])]
                    break
        elif name in ("mask_mul", "mask_gate") and _name(a.get("a")) in fm:
            fm[out] = fm[_name(a["a"])]
        if name in ("sandwich", "depth_occlude"):
            if a.get("type_a") in LP_LINEISH:
                a["type_a"] = "$wm" if "wm" in fm else "$tmat"
            if a.get("type_rgb") == "$tcol":
                a["type_rgb"] = "$" + face_of(a.get("type_a"))
            fl = _floor_opacity(a.get("type_opacity", 1.0), hl) if hl else None
            if fl is not None:
                a["type_opacity"] = fl
            ref = recolour(a)
            drawers.append((a, fl is not None, ref))
        elif name == "over" and a.get("fg") == "$tcol" and _name(a.get("alpha")) in fm:
            a["fg"] = "$" + fm[_name(a["alpha"])]
        elif name == "knockout":
            a.setdefault("min_contrast", LP_MIN_CONTRAST)
            if _name(a.get("type_a")) in fm:
                fb = fm[_name(a["type_a"])]
                a.setdefault("face", lc_made[("$" + fb, ver.get(fb, 0))][0] if ("$" + fb, ver.get(fb, 0)) in lc_made else "$" + fb)   # the solid fallback is the recoloured face
        new.append(o)
        ver[out] = ver.get(out, 0) + 1
    for wb, la_, at in front_swap:                              # R1: the window / knockout passes drew the raw wake (echo at full ink): they draw the front copy + subdued echo too
        for o in new[at:]:
            if o["op"] in ("hold_net", "flow_wake", "hold_life", "legible_color", "rgba_part") or o.get("out") == wb:
                continue
            o["args"] = subst(o.get("args", {}), {"$" + wb: "$" + la_})
    k = len(new)
    while k > 0 and new[k - 1]["op"] in LP_TAIL:
        k -= 1
    net = []
    if hl and drawers:
        main, _fl, (main_rgb, main_alpha) = next((d for d in drawers if d[1]), drawers[0])
        buf = _name(new[k]["args"].get("frame")) if k < len(new) else None
        buf = buf or (new[k - 1].get("out", "image") if k else "image")
        plate = main.get("footage") if isinstance(main.get("footage"), str) and main["footage"] != "$" + buf else "$footage"
        net = [{"op": "hold_net", "out": buf, "args": {"image": "$" + buf, "plate": plate, "face": main_rgb, "alpha": main_alpha,
                                                      "holds": hl, "t": "$t", "frame_idx": "$frame_idx", "state_key": f"{piece.get('id', 'piece')}:{sid}", **({"scrim": True} if soft_ok else {})}}]
    return head + new[:k] + net + new[k:]


def multiplane_ops(piece, ops, type_dir, sid):
    """`camera.plate: multiplane` pieces: the type-drawing ops become the `multiplane` compose op (multiplane_recipe.apply_to, the SAME transform
    piece_runner.Piece.recipe uses): the plate as depth cards under the piece's type camera, the beauty pass composited between / through them per
    pixel by depth. Runs after legibility_pass: the op takes the robust alpha `$tmat` + the beauty pass, the hold_net stays after it (before the
    closing grade) and draws the beauty face in front of the cards inside the word region when the cards hide the word. Flat pieces, shots
    without a depth pass, type dirs without a manifest and `compose.multiplane = false`: ops returned unchanged. The cards run on the GPU when
    there is one (setup.device)."""
    comp = piece.get("compose") or piece.get("recipe") or {}
    if type_dir is None or comp.get("multiplane") is False:
        return ops
    from tools.fusion import multiplane_recipe as MPR
    if not MPR.is_multiplane(piece):
        return ops
    alpha = "$tmat" if any(o["op"] == "type_face" for o in ops) else "$type_matte"
    lc = next((o for o in ops if o["op"] == "legible_color"), None)
    legible = None
    if lc is not None and alpha == "$tmat":                  # the cards draw the RECOLOURED face (legible_color on the beauty face, support painted afterwards)
        la = lc["args"]
        legible = dict(words=copy.deepcopy(la["words"]), fps=la["fps"], state_key=la["state_key"].rsplit(":lc", 1)[0] + ":lcmp",
                       bg=_beauty_bg(piece, type_dir), **{k: copy.deepcopy(la[k]) for k in ("outline", "ink_palette", "soft_support", "shot_palette", "piece_key", "shot_t0", "user_palette", "boost", "outline_w", "life") if la.get(k)})
    return MPR.apply_to(piece, ops, sid, str(type_dir), str(Path(type_dir).parent / f"mp_setup_{sid}.json"),
                        log=lambda m: print(m, flush=True), state_key=f"{piece.get('id', 'piece')}:{sid}", device="auto", alpha=alpha,
                        scene_key=str(piece.get("id", "piece")), legible=legible)


def net_summary(piece, sid, type_dir=None):
    """per-hold report of the hold_net that ran in this process for shot `sid` (compose.net_report): [] when the piece has none."""
    hl = piece_word_holds(piece, type_dir)
    return compose.net_report(f"{piece.get('id', 'piece')}:{sid}", hl, int(piece.get("fps", FPS))) if hl else []


# ================================================================================================== recipe
def subject_matte_wanted(piece) -> bool:
    """three-layer shots: the subject on top of the type is cut with the Alpha-Gen matte (see depth_occlude(subject_matte=)); `piece.layers.subject_matte = false` opts out."""
    return (piece.get("layers") or {}).get("subject_matte") is not False


def shot_needs(piece, sid, layered=False):
    """{'matte': bool, 'depth': bool}: does the shot's op chain read $<sid>_matte / $<sid>_depth? layered (the shot has a clean clip, 16_app/LAYERS.md): a `depth_occlude` shot
    also needs the matte (the subject is cut by it, not only by the depth cover: batch 3060 v007 r01 had no matte and the type sat over the climber)."""
    _, spec = _compose_spec(piece, sid)
    ops = subst(copy.deepcopy(spec["ops"]), {"$matte": f"${sid}_matte", "$depth": f"${sid}_depth"})
    rs = set()
    refs([o.get("args", {}) for o in ops], rs)
    need = {"matte": f"{sid}_matte" in rs, "depth": f"{sid}_depth" in rs}
    if layered and subject_matte_wanted(piece) and any(o["op"] == "depth_occlude" for o in ops):
        need["matte"] = True
    from tools.fusion import multiplane_recipe as MPR
    if MPR.is_multiplane(piece):                          # the cards are cut from the shot's depth (+ matte): multiplane_ops reads both
        sh = next((x for x in piece.get("shots", []) if x["id"] == sid), {})
        need["depth"] = need["depth"] or bool(sh.get("depth"))
        need["matte"] = need["matte"] or bool(sh.get("depth") and sh.get("matte"))
    from tools.fusion import pivot_cut as PC             # a live 3D cut (depth_pivot / orbit:* / depth_dolly:*) reads real depth of BOTH its shots (MographPivotCut depth_k); without it the 16x16 placeholder arrives
    if any(sid in (sp["a"], sp["b"]) and sp["kind"] != "circle_grid" for sp in PC.specs(piece)):       # depth_lines (promoted explore cut) reads the real per-frame VDA depth too; circle_grid uses none
        need["depth"] = True
    if PC.may_get_depth_cuts(piece):                     # the gathers get the PLAN piece (graph: plan -> gather -> VDA -> design_pkg -> design): no compose.cuts exist yet, so EVERY shot gets a depth ref (cheap)
        need["depth"] = True
    return need


def shot_needs_flow(piece, sid):
    """does the shot's op chain read $flow / $<sid>_flow (the footage optical flow, tools/fusion/footage_flow.py)?"""
    _, spec = _compose_spec(piece, sid)
    from tools.fusion import multiplane_recipe as MPR
    if MPR.is_multiplane(piece) and next((x for x in piece.get("shots", []) if x["id"] == sid), {}).get("depth"):
        return False                                   # multiplane_ops drops the flow-wake chain: the cards need no footage flow
    rs = set()
    refs([o.get("args", {}) for o in spec["ops"]], rs)
    return bool(rs & {"flow", f"{sid}_flow", "flow_m", f"{sid}_flow_m"})


def compose_size(piece):
    sz = (piece.get("sizes") or {}).get("compose")
    if sz:
        return tuple(int(x) for x in sz)
    comp = piece.get("compose") or piece.get("recipe") or {}
    if comp.get("size"):
        return tuple(int(x) for x in comp["size"])
    sz = (540, 960) if piece.get("quality") == "draft" else (1080, 1920)
    return sz[::-1] if piece.get("format") == "16:9" else sz


def _other_shot(ref, sid, shots):
    """(target shot id, 'footage' | 'matte') when `ref` is a two-shot input ($next, $prev, $shot:<id>; tools/fusion/window_cut.py), else None."""
    from tools.fusion import window_cut
    return window_cut.other_shot_ref(ref, sid, list(shots))


def build_recipe(piece, sid, *, type_dir=None, cut=None, curves=None, layered=False):
    """lifted Piece.recipe(): ops with $matte / $depth / $shot resolved for shot `sid`, the signoff op appended when the shot is
    on screen at the signoff time, and an `inputs` table describing every buffer. In-memory sources are keyed by `src`
    ("footage" | "matte" | "depth"); type passes by `dir` (type_dir/<pass>).
    layered (three-layer shot route, 16_app/LAYERS.md): `$footage` is the text-rendered BACKGROUND and the subject is taken from the extra input
    `$subject` (the clean clip): tools/fusion/layers_recipe.layerize_ops rewrites the ops that take "the subject". Default False = single clip."""
    fps = int(piece.get("fps", FPS))
    comp, spec = _compose_spec(piece, sid)
    shots = {s["id"]: s for s in piece["shots"]}
    ops = subst(copy.deepcopy(spec["ops"]), {"$matte": f"${sid}_matte", "$depth": f"${sid}_depth", "$shot": f"${sid}"})
    from tools.fusion import letter_fill as _LF                       # 2026-11-01: solid letters by default (footage / plate windows in the letters only when the user asks)
    ops, _n_solid = _LF.solid_ops(piece, ops)
    ops = guard_ops(piece, ops, type_dir)
    ops = legibility_pass(piece, ops, type_dir, sid)
    ops = multiplane_ops(piece, ops, type_dir, sid)
    ops = floor_ops(piece, ops, type_dir)
    ops = style_ops(piece, ops, sid)
    from tools.fusion import contour_sweep as _CS                     # 2026-10-30: compose.contour_sweep = true / effect `contour_sweep` -> the sweep op after the last type drawer
    ops = _CS.recipe_ops(piece, ops, sid, shots)
    from tools.fusion import word_bridge as _WB                       # 2026-11-01: library word-to-word transitions = ONE continuous bridge (letters A -> [contour] -> pattern -> letters B)
    try:
        bridge_ev = _WB.plan(piece, type_dir) if type_dir is not None else []
    except Exception as e:      # noqa: BLE001  never break a compose for the bridge
        print(f"compose {sid}: word_bridge skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
        bridge_ev = []
    ops = _WB.recipe_ops(piece, ops, sid, type_dir, bridge_ev)
    if not any(o.get("op") == _WB.NAME for o in ops):
        bridge_ev = []                                              # this shot draws no bridge: no gate either (the gaps belong to other shots)
    layer_rep = None
    if layered:
        from tools.fusion import layers_recipe as LR
        from tools.fusion import multiplane_recipe as MPR
        if MPR.is_multiplane(piece):                          # the cards are cut from the shot's own footage: layering is not supported there
            layer_rep = {"changed": 0, "clones": 0, "skipped": ["multiplane piece: single-clip compose"]}
        else:
            ops, layer_rep = LR.layerize_ops(ops)
            if subject_matte_wanted(piece):                  # depth_occlude: the Alpha-Gen matte of the clean clip keeps the subject in front of the type (after the legibility guard)
                for o in ops:
                    if o["op"] == "depth_occlude" and o.get("args", {}).get("fg") is not None and o["args"].get("subject_matte") is None:
                        o["args"]["subject_matte"] = f"${sid}_matte"
                        layer_rep["subject_matte"] = layer_rep.get("subject_matte", 0) + 1
    lay, _ = piece_layout(piece)
    L = lay[sid]
    size = compose_size(piece)
    rs = set()
    refs([o.get("args", {}) for o in ops], rs)
    out_name = spec.get("output", "image")
    so = piece.get("signoff")
    if so and so.get("off"):        # planner switch: no signoff text on this piece
        so = None
    if so:
        t0f = int(round(float(so["t0"]) * fps))
        if L["start"] + L["frames"] > t0f:
            ops.append({"op": "over", "out": out_name,
                        "args": {"base": f"${out_name}", "fg": list(c / 255.0 for c in so.get("color", [235, 230, 225])),
                                 "alpha": "$signoff"}})
            rs.add("signoff")
    produced = {o.get("out", "image") for o in ops}
    inputs = {"footage": dict(src="footage", kind="rgb")}
    for r in sorted(rs - produced - BUILTIN):
        if r == sid:
            inputs[r] = dict(src="footage", kind="rgb")
        elif r == "subject":                                 # layered shots: the clean clip (the pixels the matte was cut from)
            inputs[r] = dict(src="subject", kind="rgb")
        elif r == f"{sid}_matte":
            inputs[r] = dict(src="matte", kind="gray", native=True)
        elif r == f"{sid}_depth":
            inputs[r] = dict(src="depth", kind="gray", native=True, invert=bool(shots[sid].get("depth_invert")), norm="auto")
        elif r in ("flow", f"{sid}_flow"):                  # footage optical flow [T,h,w,2] (footage_flow.flow_for), kept native
            inputs[r] = dict(src="flow", kind="flow", native=True)
        elif r in ("flow_m", f"{sid}_flow_m"):              # motion matte derived from that flow (FlowMatteSource), kept native
            inputs[r] = dict(src="flowmatte", kind="gray", native=True)
        elif r.startswith("type_"):
            p, _, lag = r[5:].partition("@")          # optional "@-3" = frame offset (echo: the type 3 frames ago)
            if type_dir is None:
                raise ValueError(f"compose {sid}: ${r} needs the type layers directory")
            inputs[r] = dict(dir=str(Path(type_dir) / p), kind="gray" if p in ("matte", "depth", "line") else "rgb",
                             offset=L["start"] + (int(lag) if lag else 0))
        elif _other_shot(r, sid, shots) is not None:         # window_cut.py: $next / $prev / $shot:<id> (+ _matte): frames of ANOTHER shot, aligned in piece time
            tgt, kd = _other_shot(r, sid, shots)
            inputs[r] = dict(src=kd, shot=tgt, kind="rgb" if kd in ("footage", "subject") else "gray", native=(kd == "matte"),
                             offset=L["start"] - lay[tgt]["start"])
        elif r == "signoff":
            if cut is None:
                cut = max(v["start"] + v["frames"] for v in lay.values()) / fps
            inputs[r] = dict(signoff=dict(text=so["text"], font=so.get("font", "Space Mono"), t0=float(so["t0"]), t1=float(cut),
                                          px=int(round(int(so.get("px", 30)) * min(size) / 1080.0)), fade=float(so.get("fade", 0.35))),
                             offset=L["start"])
        else:
            raise ValueError(f"compose {sid}: unknown reference ${r}")
    owners = owner_windows(piece, type_dir)
    if owners is not None:                       # one owner per word: the flat type is drawn only while compose owns the word (per-frame gain, owner.py)
        gs = OW.gain_spec(owners)
        bg = _beauty_bg(piece, type_dir)
        for r, spx in inputs.items():
            if not r.startswith("type_"):
                continue
            kind = r[5:].partition("@")[0]
            if kind in ("matte", "line"):
                spx["gain"] = gs
            elif kind == "beauty" and bg is not None:        # faded toward the pass background so type_face's colour recovery stays exact
                spx["gain"] = OW.gain_spec(owners, [float(c) / 255.0 for c in bg][:3])
    from tools.fusion import pivot_cut as _PC
    gw = _PC.gate_windows(piece)
    if bridge_ev:                                # word_bridge OWNS its gaps: the type pass fades out after A's release and back in just before B's hit (the bridge draws the lines between)
        gw = list(gw or []) + _WB.gate_windows(bridge_ev)
    if gw:                                       # a promoted cut (circle_grid word form / depth_lines) OWNS B's word inside its window: the type stage draws nothing there (pivot_cut.apply_gate)
        bg_ = _beauty_bg(piece, type_dir)
        for r, spx in inputs.items():
            if r.startswith("type_"):
                spx["gate"] = {"win": gw, **({"bg": [float(c) / 255.0 for c in bg_][:3]} if (r[5:].partition("@")[0] == "beauty" and bg_ is not None) else {})}
    rec = dict(name=sid, size=list(size), fps=fps, ops=ops, output=out_name, inputs=inputs, piece_offset=L["start"],
               frames=L["frames"], crf=comp.get("crf", 16),
               curves={k: v for k, v in (curves or {}).items() if isinstance(v, list)})
    if layer_rep is not None:
        rec["layers"] = layer_rep
    hn = next((o for o in ops if o["op"] == "hold_net"), None)
    if hn is not None:                      # compose_frames reports what the hold net did per hold (stats["hold_net"])
        rec["net"] = dict(key=hn["args"]["state_key"], holds=copy.deepcopy(hn["args"]["holds"]))
    lc_ = next((o for o in ops if o["op"] == "legible_color"), None)
    if lc_ is not None:                     # compose_frames reports what the colour stage measured / chose per word (stats["legible_color"])
        _, _w, _t = color_plan(piece, type_dir)
        rec["lc"] = dict(key=lc_["args"]["state_key"], words=copy.deepcopy(lc_["args"]["words"]), texts=_t, fps=fps)
    from tools.fusion import pivot_cut as PC_
    hold_ok = PC_.holds_outside_cuts(piece, comp.get("holds") or [], fps)       # never a freeze-frame inside a 3D cut's window (ship #10b)
    if hold_ok:
        rec["holds"] = hold_ok
        rec["holds_type"] = bool(comp.get("holds_type", True))
    if comp.get("palette"):
        rec["palette"] = dict(comp["palette"])
    return rec


# ================================================================================================== sources
def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


def to_backend(x):
    """numpy / torch (u8, u16, float) frame -> float32 backend array in [0,1] (u8 /255 as imread; float as is)."""
    B = compose.B
    if _is_torch(x):
        x = x.detach()
        if B.name == "torch":
            t = x.to(B.device)
            if t.dtype == torch.uint8:
                return t.float() / 255.0
            return t.float().contiguous()
        x = x.cpu().numpy()
    if x.dtype == np.uint8:
        f = x.astype(np.float32) / 255.0
    elif x.dtype == np.uint16:
        f = x.astype(np.float32) / 65535.0
    else:
        f = np.asarray(x, np.float32)
    f = np.ascontiguousarray(f)
    return B.asarray(f) if B.name == "torch" else f


def to_gray(f):
    """as compose.imread(kind='gray') / FrameSource: 1 channel -> itself, RGB(A) -> Rec.709 luma."""
    if f.ndim == 2:
        return f
    c = f.shape[-1]
    if c == 1:
        return f[..., 0]
    return f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722


def _shape_kind(f, kind):
    if kind in ("gray", "alpha"):
        f = to_gray(f)
    elif f.ndim == 2:
        f = B_stack3(f)
    elif f.shape[-1] == 1:
        f = B_stack3(f[..., 0])
    else:
        f = f[..., :3]
    return f.contiguous() if _is_torch(f) else np.ascontiguousarray(f)


def B_stack3(g):
    return torch.stack([g, g, g], -1) if _is_torch(g) else np.stack([g, g, g], -1)


class ArraySource:
    """frames held in memory (uint8 or float, [T,H,W,C] / [T,H,W]); one frame at a time goes to the backend device."""

    def __init__(self, frames, kind="rgb", fit=None, invert=False, norm=None):
        self.frames, self.kind = frames, kind
        self.fit = tuple(fit) if fit else None
        self.invert, self.norm = invert, norm
        self.n = int(frames.shape[0])

    def __len__(self):
        return self.n

    def hint(self, js):
        pass

    def host_gray(self, j):
        """numpy float32 HxW of frame j (unfitted), for statistics."""
        x = self.frames[j]
        x = x.cpu().numpy() if _is_torch(x) else np.asarray(x)
        f = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
        return np.ascontiguousarray(to_gray(f))

    def _decode(self, j):
        return self.frames[j]

    def __getitem__(self, j):
        f = _shape_kind(to_backend(self._decode(j)), self.kind)
        if self.fit is not None and tuple(f.shape[:2]) != (self.fit[1], self.fit[0]):
            f = compose.fit_cover.raw(f, self.fit)
        if self.norm:
            f = (f - self.norm[0]) / (self.norm[1] - self.norm[0])
            f = f.clip(0.0, 1.0)
        return 1.0 - f if self.invert else f

    def close(self):
        pass


class RefSource(ArraySource):
    """a matte / depth clip cut from the shot's footage (Alpha-Gen / depth references are `cover_resize` crops of the clip, nodes_fusion/footage.py `_u8_to_ref`): frame j is the reference frame at
    the same fraction of the clip (a clip of another length is not clamped at its last frame) and only the window of it that is the compose frame is handed on (matte_align.ref_to_frame), so
    the later stretch to the compose size lands the subject on the footage's subject. The matte is only ever laid over the picture it was cut from, frame for frame."""

    def __init__(self, frames, kind, fit, invert, norm, footage, size):
        super().__init__(frames, kind, fit, invert, norm)
        self.n_ref = self.n
        self.n = int(footage.shape[0])
        self.fhw = (int(footage.shape[1]), int(footage.shape[2]))
        self.size = (int(size[0]), int(size[1]))

    def _decode(self, j):
        x = self.frames[MA.frame_index(j, self.n_ref, self.n)]
        hw = (int(x.shape[0]), int(x.shape[1]))
        if MA.canvas_hw(hw, self.fhw, self.size) == hw:
            return x
        return MA.ref_to_frame(x.cpu().numpy() if _is_torch(x) else np.asarray(x), self.fhw, self.size)

    def host_gray(self, j):
        x = self._decode(j)
        x = x.cpu().numpy() if _is_torch(x) else np.asarray(x)
        f = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
        return np.ascontiguousarray(to_gray(f))


FM_LO, FM_HI = 0.35, 2.5      # motion matte ramp: residual flow speed (px / frame of the half-res flow grid) -> 0..1


class FlowMatteSource(ArraySource):
    """Motion matte from the footage flow: where the footage moves relative to its own global drift (median flow, i.e. camera
    pan removed), as smoothstep(speed). Stands in for a subject matte, so flow_wake works without Alpha-Gen. Flow frames may be
    numpy or torch, fp16 or fp32; one frame at a time goes through float32 on the host."""

    def __init__(self, flow, lo=FM_LO, hi=FM_HI):
        super().__init__(flow, "gray", None)
        self.lo, self.hi = float(lo), float(hi)

    def _decode(self, j):
        x = self.frames[j]
        f = (x.cpu().numpy() if _is_torch(x) else np.asarray(x)).astype(np.float32)
        med = np.median(f[::8, ::8].reshape(-1, 2), axis=0)
        r = f - med
        mag = np.sqrt(r[..., 0] ** 2 + r[..., 1] ** 2)
        t = np.clip((mag - self.lo) / max(self.hi - self.lo, 1e-6), 0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    def host_gray(self, j):
        return self._decode(j)


class DirSource(ArraySource):
    """PNG directory (type passes). Decoding runs in a small thread pool, `hint()` schedules the frames ahead."""

    def __init__(self, directory, kind="gray", fit=None, pool=None, invert=False):
        files = sorted((str(p) for p in Path(directory).glob("*.png")), key=natural)
        if not files:
            raise FileNotFoundError(f"no PNG frames in {directory}")
        self.files, self.kind, self.fit, self.invert, self.norm = files, kind, tuple(fit) if fit else None, invert, None
        self.n = len(files)
        self._pool = pool
        self._cache = OrderedDict()

    @staticmethod
    def _read(path):
        im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"cannot read image {path}")
        if im.ndim == 3 and im.shape[2] >= 3:
            im = np.ascontiguousarray(im[..., :3][..., ::-1])       # BGR -> RGB
        elif im.ndim == 3:
            im = im[..., 0]
        return im

    def hint(self, js):
        if self._pool is None:
            return
        for j in js:
            j = min(max(int(j), 0), self.n - 1)
            if j not in self._cache:
                self._cache[j] = self._pool.submit(self._read, self.files[j])
        while len(self._cache) > 10:
            self._cache.popitem(last=False)

    def _decode(self, j):
        j = int(j)
        f = self._cache.get(j)
        if f is None:
            return self._read(self.files[j])
        return f.result() if hasattr(f, "result") else f

    def host_gray(self, j):
        raise NotImplementedError


class _BackendWrap:
    """numpy-returning source (signoff) -> backend arrays."""

    def __init__(self, src):
        self.src = src
        self.n = len(src)

    def __len__(self):
        return self.n

    def hint(self, js):
        pass

    def __getitem__(self, j):
        return compose._in(self.src[j])

    def close(self):
        pass


def open_sources(rec, *, footage=None, matte=None, depth=None, pool=None, flow=None, others=None, subject=None):
    """recipe inputs -> {name: source}. Uses the CURRENT compose backend (see use_backend). Sources of the same buffer are shared.
    others = {shot id: {"footage": frames, "matte": frames}} for the inputs that read another shot ($next / $prev / $shot:<id>, window_cut.py)."""
    size = tuple(rec["size"])
    mem = {"footage": footage, "matte": matte, "depth": depth, "flow": flow, "subject": subject}
    shared, srcs = {}, {}
    for nm, sp in rec["inputs"].items():
        fit = None if sp.get("native") else size
        if "signoff" in sp:
            from tools.fusion.piece_runner import _SignoffSource
            srcs[nm] = _BackendWrap(_SignoffSource(sp["signoff"], size, rec["fps"]))
        elif sp.get("src") == "flowmatte":
            if flow is None:
                raise ValueError(f"compose {rec['name']}: input ${nm} needs the footage flow but none was given")
            key = ("flowmatte",)
            if key not in shared:
                shared[key] = FlowMatteSource(flow)
            srcs[nm] = shared[key]
        elif "src" in sp:
            fr = ((others or {}).get(sp["shot"]) or {}).get(sp["src"]) if sp.get("shot") else mem.get(sp["src"])
            if fr is None:
                raise ValueError(f"compose {rec['name']}: input ${nm} needs {sp['src']} frames{' of shot ' + sp['shot'] if sp.get('shot') else ''} but none were given")
            key = (sp["src"], sp.get("kind"), fit, sp.get("invert", False)) + ((sp["shot"],) if sp.get("shot") else ())
            if key not in shared:
                norm = sp.get("norm")
                if norm == "auto":
                    norm = depth_norm_frames(_Lazy(ArraySource(fr, sp.get("kind", "gray"))))
                own = ((others or {}).get(sp["shot"]) or {}).get("footage") if sp.get("shot") else footage
                if sp["src"] in ("matte", "depth") and own is not None and getattr(own, "ndim", 0) == 4:      # cut from that shot's footage: same frame, same geometry
                    shared[key] = RefSource(fr, sp.get("kind", "rgb"), fit, sp.get("invert", False), norm, own, size)
                else:
                    shared[key] = ArraySource(fr, sp.get("kind", "rgb"), fit, sp.get("invert", False), norm)
            srcs[nm] = shared[key]
        elif "dir" in sp:
            srcs[nm] = DirSource(sp["dir"], sp.get("kind", "gray"), fit, pool, sp.get("invert", False))
        else:
            raise ValueError(f"compose {rec['name']}: bad input spec {nm}: {sp}")
    return srcs


class _Lazy:
    """len + getitem over an ArraySource's host gray frames (depth_norm_frames only reads ~10 of them)."""

    def __init__(self, src):
        self.src = src

    def __len__(self):
        return self.src.n

    def __getitem__(self, i):
        return self.src.host_gray(int(i))


# ================================================================================================== ops / backend
@contextlib.contextmanager
def use_backend(name=None):
    """temporarily switch the global compose backend ('numpy' | 'torch' | 'auto'; None keeps the current one)."""
    prev = compose.B
    if name:
        compose.set_backend(name)
    try:
        yield compose.B
    finally:
        compose.B = prev


def _raw_blur(img, sigma=4.0):
    return compose.B.blur(img, sigma)


def _raw_fill(like, color=(0.0, 0.0, 0.0)):
    return compose._color(color, like)


def _raw_translate(img, dx=0.0, dy=0.0):
    B = compose.B
    h, w = img.shape[:2]
    return B.warp(img, B.asarray(np.full((h, w), -float(dx), np.float32)), B.asarray(np.full((h, w), -float(dy), np.float32)))


def _raw_luma(x):
    return compose._luma(x)


_EXTRA_RAW = {"blur": _raw_blur, "fill": _raw_fill, "translate": _raw_translate, "luma": _raw_luma}


def op_table():
    """{op name: (callable on backend arrays, raw: bool)}. raw=False ops detour through numpy (fill_holes)."""
    t = {}
    for name, fn in compose._ops_registry().items():
        raw = getattr(fn, "raw", None) or _EXTRA_RAW.get(name)
        t[name] = (raw, True) if raw is not None else (fn, False)
    return t


def call_op(entry, args):
    fn, is_raw = entry
    if is_raw:
        return fn(**{k: compose._in(v) for k, v in args.items()})
    r = fn(**{k: compose._out(v) for k, v in args.items()})      # numpy detour
    return compose._in(r)


def _step_edges(tracer, footage, low=0.03, high=0.09, temporal_smooth=0.5, blur=1.2):
    return tracer._step(compose._in(footage), low, high, temporal_smooth, blur)     # stays on the backend device


# ================================================================================================== frame loop
def _to_u8_host(img, pin):
    """finished backend frame (HxW or HxWx3 float) -> uint8 HxWx3 numpy/torch on the CPU."""
    B = compose.B
    if B.name == "torch":
        if img.ndim == 2:
            img = img[..., None].expand(-1, -1, 3)
        u8 = torch.clamp(torch.round(img[..., :3] * 255.0), 0, 255).to(torch.uint8)
        if pin is not None and u8.is_cuda:
            pin.copy_(u8)
            return pin
        return u8.cpu()
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return compose._to_u8(img)[..., :3]


def _legible_prepass(ops, k, ctx_at, resolve_args, table, n, po, fps):
    """The per-hold decision of the legible op ops[k] (compose.legible_plan) from the shot's own frames: the op chain that produces its
    type_a / matte arguments is evaluated at the sampled frames of each hold. flow_wake in that chain is bypassed (its type_a input
    is used: the wake only nudges the letters, which the plan's boxes tolerate); a trace_edges in it -> no plan (guard off)."""
    op = ops[k]
    a = op["args"]
    names = {_arg_name(a.get("type_a")), _arg_name(a.get("matte")), _arg_name(a.get("front"))} - {None}
    chain, inputs = _closure(ops, k, names)
    if any(o["op"] == "trace_edges" for o in chain):
        return None
    need = {x for x in inputs if x not in (BUILTIN - {"footage"})}

    def evalf(i):
        ctx, u, pf = ctx_at(i, need)
        for o in chain:
            args = resolve_args(o, ctx, u, pf)
            if o["op"] == "flow_wake":
                ctx[o.get("out", "image")] = args["type_a"]
            else:
                ctx[o.get("out", "image")] = call_op(table[o["op"]], args)
        c2 = dict(ctx)
        ra = resolve_args(op, c2, u, pf)
        fr = compose._out(ra["front"]) if _arg_name(a.get("front")) else None
        return compose._out(ra["type_a"]), compose._out(ra["matte"]), fr

    cache = {}

    def at(f):
        i = f - po
        if i not in cache:
            if len(cache) > 3:
                cache.clear()
            cache[i] = evalf(i)
        return cache[i]

    kw = {key: a[key] for key in ("min_vis", "min_frac", "dilate", "matte_threshold", "matte_soft", "matte_gamma", "invert", "floor", "letters", "behind") if key in a}
    return compose.legible_plan(lambda f: at(f)[0], lambda f: at(f)[1], [tuple(h) for h in a["holds"]], fps=a.get("fps", fps),
                                frame0=po, frame1=po + n, front_at=(lambda f: at(f)[2]) if _arg_name(a.get("front")) else None, **kw)


def compose_frames(rec, srcs, *, i0=0, i1=None, progress=None, palette=None, dump=None):
    """render frames [i0, i1) of a recipe (shot-local indices; trace_edges is warmed up on the 4 frames before i0).
    Uses the current compose backend. -> (uint8 torch tensor [i1-i0, H, W, 3] on the CPU (numpy if torch is missing), stats)."""
    n = int(rec["frames"])
    i1 = n if i1 is None else min(int(i1), n)
    ops = rec["ops"]
    table = op_table()
    for op in ops:
        if op["op"] not in table and op["op"] != "trace_edges":
            raise ValueError(f"unknown op {op['op']!r}; available: {sorted(set(table) | {'trace_edges'})}")
    offs = {nm: int(sp.get("offset", 0)) for nm, sp in rec["inputs"].items()}
    modes = {nm: sp.get("mode", "hold") for nm, sp in rec["inputs"].items()}
    gains = {nm: sp["gain"] for nm, sp in rec["inputs"].items() if sp.get("gain")}
    gates = {nm: sp["gate"] for nm, sp in rec["inputs"].items() if sp.get("gate")}
    tracers = {k: compose.EdgeTracer() for k, o in enumerate(ops) if o["op"] == "trace_edges"}
    po = int(rec.get("piece_offset", 0))
    fps = rec["fps"]
    curves = rec.get("curves", {})
    out_name = rec.get("output", "image")
    warm = 4 if tracers else 0
    if any(o["op"] == "flow_wake" for o in ops):     # displacement / echo state builds up over ~2 decay times
        warm = max(warm, 16)
    if any(o["op"] == "legible_color" for o in ops):  # the colour smoothing (EMA) of a chunk starting mid-shot matches a sequential run
        from tools.fusion import legible_color as _LC
        warm = max(warm, _LC.LC_WARM)
    W, H = rec["size"]
    out = torch.empty((i1 - i0, H, W, 3), dtype=torch.uint8) if torch is not None else np.empty((i1 - i0, H, W, 3), np.uint8)
    pin = torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True) if (
        torch is not None and compose.B.name == "torch" and compose.B.device != "cpu") else None
    pal = compose._in(np.asarray(palette, np.float32)) if palette is not None else None
    cam = {}
    for nm in _cam_refs(ops):                        # camera tracks of the flow buffers the ops ask for ({"cam": "x", "ref": "$flow"})
        if nm not in srcs or not hasattr(srcs[nm], "frames"):
            raise ValueError(f"cam track needs the footage flow input ${nm} (add it to the shot's refs)")
        cam[nm] = camera_track(srcs[nm].frames, rec["size"])
    if rec.get("lc"):                                 # a fresh log / smoothing state for this render
        from tools.fusion import legible_color as _LC
        _LC.reset(rec["lc"]["key"])
    t_acc = 0.0
    ctx_ctx = torch.no_grad() if torch is not None else contextlib.nullcontext()

    def ctx_at(i, names=None):
        """input buffers + built-ins of shot-local frame i (names = only these inputs). -> (ctx, u, piece frame)"""
        ctx, got = {}, {}
        for nm, s in srcs.items():
            if names is not None and nm not in names:
                continue
            j = i + offs[nm]
            if rec.get("holds"):
                j = hold_index(rec, nm, i, po, offs[nm])
            Ls = len(s)
            j = j % Ls if modes[nm] == "loop" else min(max(j, 0), Ls - 1)
            s.hint([j + 1, j + 2, j + 3])
            key = (id(s), j, nm if (nm in gains or nm in gates) else None)
            if key not in got:
                got[key] = OW.apply_gain(s[j], gains[nm], j / float(fps)) if nm in gains else s[j]
                if nm in gates:
                    from tools.fusion import pivot_cut as _PCg
                    got[key] = _PCg.apply_gate(got[key], gates[nm], j / float(fps))
            ctx[nm] = got[key]
            if nm in cam:
                ctx["__cam__" + nm] = cam[nm][j]
        u = i / max(n - 1, 1)
        pf = i + po
        ctx.update(frame_idx=pf, t=pf / fps, u=u, n=n, piece_frame=pf)
        if pal is not None:
            ctx["palette"] = pal
        return ctx, u, pf

    plans = {}
    skipped = set()                                   # indices of ops skipped for an unresolved $reference (logged once each)
    with ctx_ctx:
        for k, op in enumerate(ops):                 # legibility guard: one decision per hold, before the frame loop (see compose.legible)
            if op["op"] == "legible" and op.get("args", {}).get("holds") and op.get("args", {}).get("mode", "front") != "off":
                plans[k] = _legible_prepass(ops, k, ctx_at, resolve_args=lambda o, c, u, pf: {
                    a: resolve(b, c, curves, u, pf, fps) for a, b in o.get("args", {}).items()}, table=table, n=n, po=po, fps=fps)
        for i in range(max(0, i0 - warm), i1):
            t0 = time.perf_counter()
            ctx, u, pf = ctx_at(i)
            for k, op in enumerate(ops):
                try:
                    args = {a: resolve(b, ctx, curves, u, pf, fps) for a, b in op.get("args", {}).items()}
                except KeyError as e:        # an op reading a buffer nothing makes (model-written chain): skip that op, the piece goes on
                    if k not in skipped:
                        skipped.add(k)
                        print(f"compose {rec.get('name', '?')}: op {k} ({op['op']} -> {op.get('out', 'image')}) skipped: unresolved reference {e}", flush=True)
                    continue
                if op["op"] == "trace_edges":
                    fo = args.pop("footage")
                    r = _step_edges(tracers[k], fo, **args)
                else:
                    if op["op"] == "legible":
                        args["plan"] = plans.get(k)
                    r = call_op(table[op["op"]], args)
                ctx[op.get("out", "image")] = r
            if i >= i0:
                if dump is not None:                 # debug / QC hook: dump(shot-local frame, ctx of every buffer after the last op)
                    dump(i, ctx)
                host = _to_u8_host(ctx[out_name] if (out_name in ctx or not skipped) else ctx["footage"], pin)    # every op that makes the output was skipped: the plain footage
                if torch is not None:
                    out[i - i0] = host if _is_torch(host) else torch.from_numpy(np.ascontiguousarray(host))
                else:
                    out[i - i0] = host
                t_acc += time.perf_counter() - t0
                if progress:
                    progress(i - i0 + 1, i1 - i0)
    k = max(i1 - i0, 1)
    st = dict(n=i1 - i0, s_per_frame=t_acc / k, wall_s=t_acc, backend=f"{compose.B.name}:{compose.B.device}")
    if rec.get("net"):
        st["hold_net"] = compose.net_report(rec["net"]["key"], rec["net"]["holds"], fps)
    if rec.get("lc"):
        from tools.fusion import legible_color as _LC
        st["legible_color"] = _LC.lc_report(rec["lc"]["key"], rec["lc"]["words"], fps, rec["lc"].get("texts"))
    if any(plans.values()):
        st["legible"] = {str(k): [dict(a=h["a"], b=h["b"], flagged=len(h["boxes"]), letters=h["n"]) for h in pl["holds"]]
                         for k, pl in plans.items() if pl}
    return out, st


def run_compose(piece, sid, *, type_dir, footage, matte=None, depth=None, curves=None, cut=None, backend="auto",
                frames=None, progress=None, threads=3, flow=None, dump=None, others=None, subject=None):
    """compose shot `sid` of `piece`: footage uint8 frames [T,H,W,3] (+ matte / depth frames when the ops read them) and the type
    layers directory (<type_dir>/{matte,line,depth,beauty}/*.png) -> (uint8 [n,H,W,3], stats). frames=(a, b) renders a sub-range."""
    with use_backend(backend):
        rec = build_recipe(piece, sid, type_dir=type_dir, cut=cut, curves=curves, layered=subject is not None)
        lay_note = None
        if subject is not None and subject is not footage and (rec.get("layers") or {}).get("skipped"):
            # a plate chain the layers cannot clone (flow_wake, plate_pan, hold_net, legible_color ...) keeps its single-clip behaviour: the matte (cut from the CLEAN clip) would be laid over
            # the text-rendered plate = a picture it was not cut from. The clean clip is the only picture then (matte_align.py).
            lay_note = "layers skipped (" + "; ".join(str(x) for x in rec["layers"]["skipped"][:3]) + "): clean clip is the only picture, the text-rendered background is not used"
            footage, subject = subject, None
            rec = build_recipe(piece, sid, type_dir=type_dir, cut=cut, curves=curves, layered=False)
        pool = ThreadPoolExecutor(threads) if threads else None
        srcs = open_sources(rec, footage=footage, matte=matte, depth=depth, pool=pool, flow=flow, others=others, subject=subject)
        try:
            palette = None
            if rec.get("palette"):
                fs = srcs["footage"]
                palette = compose.palette_from_footage(_HostFit(fs), k=int(rec["palette"].get("k", 5)))
            a, b = (0, None) if frames is None else frames
            out_, st_ = compose_frames(rec, srcs, i0=a, i1=b, progress=progress, palette=palette, dump=dump)
            if lay_note:
                st_["layers_fallback"] = lay_note
            return out_, st_
        finally:
            for s in srcs.values():
                s.close()
            if pool:
                pool.shutdown(wait=False)


class _HostFit:
    """numpy float frames of a source (palette_from_footage runs on the host)."""

    def __init__(self, src):
        self.src = src

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return compose._out(self.src[i])


# ================================================================================================== assemble
def assemble_parts(piece):
    """lifted Piece._assemble timing: [{'cut': seconds the shot begins at (start + xf/2)}, ...] in shot order + crossfade list (s)."""
    lay, cfl = piece_layout(piece)
    fps = int(piece.get("fps", FPS))
    parts = []
    for sid, L in lay.items():
        parts.append(dict(sid=sid, cut=(L["start"] + L["xf"] / 2.0) / fps if L["xf"] or L["start"] else 0.0))
    from tools.fusion import window_cut
    owned = window_cut.owned_boundaries(piece)
    if owned:                       # a window cut owns these boundaries: HARD cut, the incoming shot starts at its piece start (no crossfade)
        parts, cfl = window_cut.hard_cut_parts(parts, cfl, owned, lay, fps)
    return parts, list(cfl)


class ClipSource:
    """uint8 clip [T,H,W,3] (torch / numpy) as an assemble() source: float32 RGB frames, cover-fitted to `size` if needed."""

    def __init__(self, u8, size=None):
        self.u8, self.size = u8, size
        self.n = int(u8.shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.u8[int(i)]
        x = x.cpu().numpy() if _is_torch(x) else np.asarray(x)
        f = x[..., :3].astype(np.float32) / 255.0
        if self.size is not None and f.shape[:2] != (self.size[1], self.size[0]):
            f = compose.fit_cover(f, self.size)
        return f

    def close(self):
        pass
