"""compose_blocks: the compose ops as BUILDING BLOCKS an open model may write itself (2026-10-29, 20_open-model-design par round).

The user rejected fixed recipes ("we will lose the creativity of uniqueness"), so the model does not pick a recipe: it writes the op chain of a shot
(`compose.default.ops` / `compose.shots.<id>.ops`, the same {"op", "out", "args"} list the engine runs), from the ops in `ALLOWED`. This module is the
menu text and the validator:

    menu(cut)                                -> prompt text: every allowed op, its arguments (from the real signature) and the first lines of its doc
    validate_ops(ops, cut, buffers, ...)     -> (kept ops, fixes): each op is checked ALONE (name, argument names, argument values, $references); an invalid
                                                argument is clamped / dropped, an op that cannot work is dropped, and an op that read a dropped op's output
                                                goes with it. Nothing is invented; plumbing the engine always needs (upscale_matte for a raw `$matte`) is
                                                added and logged. A chain that draws no type at all returns None (the caller falls back to the rules chain).

Argument grammar (what the engine's `recipe_eval.resolve` accepts): a number / bool / [r,g,b] / "$buffer" / {"timeline": [[sec, v], ...], "step": bool,
"ease": "linear" | "smooth"} (piece seconds, v = number or [r,g,b] colour 0..1) / {"curve": "sub|low|mid|high|rms|onset", "lo": a, "hi": b} (the music's envelope) /
{"anim": [[u, v], ...]} (u = 0..1 position in the shot).
"""
from __future__ import annotations

import copy
import inspect
import re
import threading

# the ops a model may use. Not offered: legible / hold_net / type_face / legible_color / legible_support / rgba_part / depth_cover / multiplane / plate_pan
# (the engine inserts the legibility and colour machinery itself), window_cut (a `compose.cuts` choice), layout_shapes / edge_run_type / rigid_pose
# (blocks of the planner), trace_edges (stateful, needs warm-up).
ALLOWED = ("upscale_matte", "grade", "tint_type", "sandwich", "depth_occlude", "knockout", "over", "add_glow", "halation", "grain", "vignette", "crossfade",
           "blur", "fill", "screen", "multiply", "overlay", "soft_light", "luma", "luminosity_key", "fill_holes", "flow_wake", "contour_lines", "refract",
           "edges_in_type", "type_from_edges", "mask_gate", "mask_mul", "matte_reach", "matte_ring", "split_region", "translate")
DRAWERS = ("sandwich", "depth_occlude", "knockout")           # ops that put the type into the picture
INTERNAL_ARGS = {"state_key", "state", "plan", "legible"}     # arguments the engine owns
STRING_ARGS = {"mode": {"sample", "complement", "gradient_map", "iso", "outline", "scribble"}, "axis": {"x", "y", "diag"}, "side": {"left", "right", "top", "bottom"},
               "part": {"rgb", "a"}}
CURVES = ("sub", "low", "mid", "high", "rms", "onset")
# buffers a chain may read before it has made any (`$matte` / `$depth` are the SHOT's; the type passes are aligned to piece time by the engine)
BUFFERS = ("footage", "matte", "depth", "flow", "flow_m", "type_matte", "type_line", "type_depth", "type_beauty", "frame_idx", "t", "u", "n", "palette")
PASS_FOR = {"type_matte": "matte", "type_line": "line", "type_depth": "depth", "type_beauty": "beauty"}
# (lo, hi) clamps by argument name, applied silently and logged
RANGES = {"grade.gain": (0.2, 2.5), "grade.sat": (0.0, 2.5), "grade.lift": (0.0, 0.2), "grade.gamma": (0.5, 1.8), "grain.amount": (0.0, 0.08), "grain.size": (0.3, 4.0),
          "vignette.strength": (0.0, 0.8), "vignette.radius": (0.1, 1.0), "tint_type.sat": (0.0, 3.0), "tint_type.lum_target": (0.0, 1.0), "tint_type.lum_amount": (0.0, 1.0),
          "tint_type.sample_blur": (0.0, 120.0), "sandwich.type_opacity": (0.0, 1.0), "sandwich.matte_gamma": (0.3, 3.0), "sandwich.matte_gain": (0.0, 2.0),
          "sandwich.matte_soft": (0.0, 0.3), "sandwich.matte_threshold": (0.05, 0.95), "depth_occlude.type_opacity": (0.0, 1.0), "depth_occlude.plane": (0.0, 1.0),
          "depth_occlude.softness": (0.0, 0.5), "knockout.min_contrast": (0.0, 0.5), "add_glow.radius": (2.0, 120.0), "add_glow.intensity": (0.0, 1.5), "add_glow.passes": (1, 4),
          "blur.sigma": (0.0, 80.0), "screen.opacity": (0.0, 1.0), "multiply.opacity": (0.0, 1.0), "overlay.opacity": (0.0, 1.0), "soft_light.opacity": (0.0, 1.0),
          "fill_holes.amount": (0.0, 1.0), "flow_wake.strength": (0.0, 12.0), "flow_wake.radius": (2.0, 150.0), "flow_wake.decay": (0.5, 0.97), "flow_wake.trail": (0.0, 20.0),
          "flow_wake.settle": (0.0, 1.0), "flow_wake.trail_gain": (0.0, 2.0), "flow_wake.max_disp": (0.0, 0.4), "contour_lines.levels": (1, 14), "contour_lines.width": (1, 6),
          "contour_lines.phase": (-8.0, 8.0), "contour_lines.reveal": (0.0, 1.0), "contour_lines.strokes": (1, 4), "contour_lines.max_contours": (1, 8),
          "refract.strength": (0.0, 40.0), "refract.chroma": (0.0, 0.5), "refract.rim": (0.0, 1.0), "refract.blur": (0.0, 30.0), "halation.threshold": (0.0, 1.0),
          "halation.strength": (0.0, 1.0), "halation.radius": (2.0, 80.0), "luminosity_key.low": (0.0, 1.0), "luminosity_key.high": (0.0, 1.0),
          "type_from_edges.grow": (0.0, 40.0), "type_from_edges.softness": (0.0, 12.0), "type_from_edges.density": (0.0, 3.0), "type_from_edges.fill": (0.0, 1.0),
          "edges_in_type.gain": (0.0, 3.0), "mask_mul.floor": (0.0, 1.0), "mask_gate.gate": (0.0, 1.0), "matte_reach.progress": (0.0, 1.0), "matte_ring.progress": (0.0, 1.0),
          "split_region.pos": (0.0, 1.0)}
COLOR_ARGS = {"color", "plate_rgb", "fg", "tint", "rim_color", "lift_c"}
# an op's 'fg' is a colour or buffer; sandwich 'fg' is a buffer. Which args are buffer-valued is read from the usage: any "$name" is a buffer.


# buffer KINDS: rgb (HxWx3) | gray (HxW) | any. An op fed the wrong kind (a flow_wake alpha as type_rgb) fails in the engine, so the chain check catches it.
BUF_KIND = {"footage": "rgb", "matte": "gray", "depth": "gray", "flow_m": "gray", "type_matte": "gray", "type_line": "gray", "type_depth": "gray", "type_beauty": "rgb"}
KIND_IN = {"sandwich": dict(footage="rgb", type_rgb="rgb", type_a="gray", matte="gray", fg="rgb", front="gray"),
           "depth_occlude": dict(footage="rgb", footage_depth="gray", type_rgb="rgb", type_a="gray", cover="gray", fg="rgb", subject_matte="gray"),
           "knockout": dict(plate_rgb="rgb", type_a="gray", footage="rgb", fg_matte="gray", fg="rgb", face="rgb"), "tint_type": dict(type_a="gray", footage="rgb"),
           "over": dict(base="rgb", fg="rgb", alpha="gray"), "add_glow": dict(img="rgb", mask="gray"), "grade": dict(frame="rgb"), "halation": dict(frame="rgb"),
           "grain": dict(frame="rgb"), "vignette": dict(frame="rgb"), "crossfade": dict(t="gray"), "luma": dict(x="rgb"), "luminosity_key": dict(img="rgb"),
           "fill_holes": dict(mask="gray"), "flow_wake": dict(type_a="gray", matte="gray"), "refract": dict(footage="rgb", type_depth_or_matte="gray", mask="gray"),
           "edges_in_type": dict(edges="gray", type_a="gray"), "type_from_edges": dict(edges="gray", type_a="gray"), "mask_gate": dict(a="gray"), "mask_mul": dict(a="gray", b="gray"),
           "matte_reach": dict(matte="gray"), "matte_ring": dict(matte="gray"), "upscale_matte": dict(matte="gray", guide="rgb")}
KIND_OUT = {"sandwich": "rgb", "depth_occlude": "rgb", "knockout": "rgb", "tint_type": "rgb", "add_glow": "rgb", "grade": "rgb", "halation": "rgb", "grain": "rgb", "vignette": "rgb",
            "fill": "rgb", "refract": "rgb", "luma": "gray", "luminosity_key": "gray", "fill_holes": "gray", "flow_wake": "gray", "contour_lines": "gray", "edges_in_type": "gray",
            "type_from_edges": "gray", "mask_gate": "gray", "mask_mul": "gray", "matte_reach": "gray", "matte_ring": "gray", "split_region": "gray", "upscale_matte": "gray"}
SCALAR_OK = {("crossfade", "t")}
BLEND = ("screen", "multiply", "overlay", "soft_light")
PRIMARY = {"grade": "frame", "halation": "frame", "grain": "frame", "vignette": "frame", "add_glow": "img", "blur": "img", "screen": "a", "multiply": "a", "overlay": "a",
           "soft_light": "a", "crossfade": "a", "translate": "img", "over": "base"}       # picture-to-picture filters: when one cannot work it is bypassed, the chain goes on


def _rewrite(obj, alias):
    if isinstance(obj, str) and obj.startswith("$") and obj[1:] in alias:
        return "$" + alias[obj[1:]]
    if isinstance(obj, dict):
        return {k: _rewrite(v, alias) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, alias) for v in obj]
    return obj
KIND_FOLLOWS = {"over": "base", "crossfade": "a", "blur": "img", "screen": "a", "multiply": "a", "overlay": "a", "soft_light": "a", "translate": "img"}


_TL = threading.local()


def _bufs() -> set:
    """the buffers a chain may read before it has made any, minus the ones this call says the shot does not have (validate_ops(unavailable=))."""
    return set(BUFFERS) - set(getattr(_TL, "unavailable", ()))


def _sig(name: str):
    from tools.fusion import compose
    fn = compose._ops_registry().get(name)
    if fn is None:
        return None
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def menu(cut: float | None = None, names=ALLOWED, doc_chars: int = 520) -> str:
    """the op menu for a prompt: name(args with defaults) + the first paragraph of its docstring."""
    from tools.fusion import compose
    reg = compose._ops_registry()
    rows = []
    for n in names:
        fn = reg.get(n)
        if fn is None:
            continue
        sig = _sig(n)
        args = ", ".join(str(p).replace(" ", "") for p in (sig.parameters.values() if sig else []) if p.name not in INTERNAL_ARGS)
        doc = re.sub(r"\s+", " ", (inspect.getdoc(fn) or "").strip())
        rows.append(f"- {n}({args}): {doc[:doc_chars]}")
    return "\n".join(rows)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) < 1e9


def _clamp_num(v, name, where, fixes):
    r = RANGES.get(name)       # name = "op.arg"
    if r is None or not _is_num(v):
        return v
    c = min(max(v, r[0]), r[1])
    if isinstance(r[0], int) and isinstance(r[1], int):
        c = int(round(c))
    if c != v:
        fixes.append(f"{where}: {v} clamped to {c} (allowed {r[0]}..{r[1]})")
    return c


def _color(v, name, where, fixes):
    """a colour written as 0..255 integers becomes 0..1 floats (the engine's colours are 0..1; palettes of the type spec are 0..255)."""
    if name in COLOR_ARGS and isinstance(v, (list, tuple)) and len(v) in (3, 4) and all(_is_num(x) for x in v) and max(v) > 1.0:
        fixes.append(f"{where}: colour {list(v)} written as 0..255, divided by 255")
        return [round(float(x) / 255.0, 4) for x in v]
    return v


def _value(v, name, where, cut, defined, fixes, op=""):
    """validate one argument value. -> (value, ok). A bad part inside a timeline is dropped; a bad whole value -> (None, False)."""
    if isinstance(v, str):
        if v.startswith("$"):
            ref = v[1:]
            if ref in defined or ref in _bufs():
                return v, True
            fixes.append(f"{where}: reference {v} is not a buffer this chain has made")
            return None, False
        if name not in STRING_ARGS and (v in defined or v in _bufs()):        # a buffer written without its $ (the commonest slip): the nearest valid value
            fixes.append(f"{where}: {v!r} is a buffer name, written as ${v}")
            return "$" + v, True
        if name in STRING_ARGS:
            if v in STRING_ARGS[name]:
                return v, True
            fixes.append(f"{where}: {v!r} is not one of {sorted(STRING_ARGS[name])}")
            return None, False
        fixes.append(f"{where}: a text value {v!r} is not an argument value")
        return None, False
    if isinstance(v, bool):
        return v, True
    if _is_num(v):
        return _clamp_num(v, f"{op}.{name}", where, fixes), True
    if isinstance(v, (list, tuple)):
        if not v or not all(_is_num(x) for x in v):
            ok = all(isinstance(x, (str, int, float, list, dict)) for x in v)
            if not ok:
                return None, False
            out = []
            for k, x in enumerate(v):
                y, good = _value(x, name, f"{where}[{k}]", cut, defined, fixes, op)
                if not good:
                    return None, False
                out.append(y)
            return out, True
        return list(_color(v, name, where, fixes)), True
    if isinstance(v, dict):
        if "timeline" in v:
            ks = v.get("timeline")
            if not isinstance(ks, list) or not ks:
                fixes.append(f"{where}: timeline needs a list of [seconds, value]")
                return None, False
            good, last = [], -1.0
            for k in ks:
                if not (isinstance(k, (list, tuple)) and len(k) == 2 and _is_num(k[0])):
                    fixes.append(f"{where}: timeline key {k!r} dropped (needs [seconds, value])")
                    continue
                t, val = float(k[0]), k[1]
                if t < last or t > (cut or 1e9) + 1.0:
                    fixes.append(f"{where}: timeline key at {t:g} s dropped (out of order or past the piece)")
                    continue
                if _is_num(val):
                    val = _clamp_num(val, f"{op}.{name}", where, fixes)
                elif isinstance(val, (list, tuple)) and val and all(_is_num(x) for x in val):
                    val = _color([float(x) for x in val], name, where, fixes)
                else:
                    fixes.append(f"{where}: timeline value {val!r} dropped")
                    continue
                good.append([round(t, 4), val])
                last = t
            if not good:
                return None, False
            ease = v.get("ease", "linear")
            out = {"timeline": good}
            if v.get("step"):
                out["step"] = True
            if ease in ("linear", "smooth") and ease != "linear":
                out["ease"] = ease
            return out, True
        if "curve" in v:
            if v["curve"] not in CURVES:
                fixes.append(f"{where}: curve {v['curve']!r} is not one of {', '.join(CURVES)}")
                return None, False
            lo, hi = v.get("lo", 0.0), v.get("hi", 1.0)
            if not (_is_num(lo) and _is_num(hi)):
                return None, False
            return {"curve": v["curve"], "lo": lo, "hi": hi}, True
        if "anim" in v:
            ks = v.get("anim")
            if isinstance(ks, list) and ks and all(isinstance(k, (list, tuple)) and len(k) == 2 and _is_num(k[0]) for k in ks):
                return {"anim": [[float(k[0]), k[1]] for k in ks], **({"ease": v["ease"]} if v.get("ease") in ("linear", "smooth") else {})}, True
            return None, False
        fixes.append(f"{where}: an object that is not timeline / curve / anim")
        return None, False
    return None, False


def _pre_alias(pre_alias: dict, op: dict, name: str, new_args: dict, args_in: dict, pk) -> None:
    """an op dropped because of one of ITS arguments (a crossfade whose `b` is nonsense): when it is a picture filter and its primary input is a buffer reference, later
    readers of its output read that input instead (the chain goes on without the filter)."""
    out = op.get("out", "image")
    v = new_args.get(pk) if pk else None
    if not (isinstance(v, str) and v.startswith("$")) and pk and isinstance(args_in.get(pk), str):
        raw = args_in[pk]
        v = raw if raw.startswith("$") else "$" + raw
    if pk and isinstance(out, str) and isinstance(v, str) and v.startswith("$") and v != "$@prev":
        pre_alias[out] = v[1:]


def _produces(op: dict) -> str:
    return op.get("out", "image") if isinstance(op.get("out", "image"), str) else "image"


def validate_ops(ops, cut: float, extra_buffers=(), shot: str = "", max_ops: int = 40, unavailable=()):
    """-> (kept ops or None, fixes). Each op is mended or dropped on its own; see the module docstring."""
    _TL.unavailable = tuple(unavailable)
    try:
        return _validate_ops(ops, cut, extra_buffers, shot, max_ops)
    finally:
        _TL.unavailable = ()


def _validate_ops(ops, cut, extra_buffers, shot, max_ops):
    fixes: list[str] = []
    if not isinstance(ops, list) or not ops:
        return None, ["ops: not a list"]
    sigs = {n: _sig(n) for n in ALLOWED}
    pre_alias: dict = {}
    defined = set(extra_buffers)
    kept: list[dict] = []
    for i, op in enumerate(ops[:max_ops]):
        w = f"{shot + '.' if shot else ''}ops[{i}]"
        if not isinstance(op, dict) or not isinstance(op.get("op"), str):
            fixes.append(f"{w}: not an {{op, out, args}} object, dropped")
            continue
        name = op["op"]
        if name not in ALLOWED:
            fixes.append(f"{w}: op {name!r} is not an allowed building block, dropped")
            continue
        sig = sigs.get(name)
        if sig is None:
            fixes.append(f"{w}: op {name!r} is not available in this engine build, dropped")
            continue
        args_in = _rewrite(op.get("args"), pre_alias) if isinstance(op.get("args"), dict) else {}
        params = {p.name: p for p in sig.parameters.values()}
        new_args, bad = {}, False
        for k, v in args_in.items():
            if k in INTERNAL_ARGS or k not in params:
                fixes.append(f"{w}.{name}: argument {k!r} does not exist, dropped")
                continue
            val, ok = _value(v, k, f"{w}.{name}.{k}", cut, defined, fixes, name)
            if ok:
                new_args[k] = val
            else:
                if params[k].default is inspect.Parameter.empty:
                    fixes.append(f"{w}.{name}.{k}: required and unusable -> the op is dropped")
                    bad = True
                    break
                fixes.append(f"{w}.{name}.{k}: unusable, left at its default")
        pk_ = PRIMARY.get(name)
        if bad:
            _pre_alias(pre_alias, op, name, new_args, args_in, pk_)
            continue
        if pk_ and pk_ not in new_args and pk_ in params:         # a picture filter without its input (grade without `frame`): it works on the picture made so far
            new_args[pk_] = "$@prev"
            fixes.append(f"{w}.{name}: `{pk_}` missing; the picture made so far is used")
        miss = [p.name for p in params.values() if p.default is inspect.Parameter.empty and p.name not in new_args and p.name not in INTERNAL_ARGS]
        if miss:
            fixes.append(f"{w}.{name}: required argument(s) {', '.join(miss)} missing, dropped")
            _pre_alias(pre_alias, op, name, new_args, args_in, pk_)
            continue
        if name == "flow_wake" and "type_rgb" in new_args:        # with type_rgb it returns RGBA; the legibility pass makes the colour twin itself
            del new_args["type_rgb"]
            fixes.append(f"{w}.flow_wake: type_rgb dropped (flow_wake then returns alpha only; the engine warps the type colour with the same displacement)")
        for k_, ref_ in (("frame_idx", "$frame_idx"), ("u", "$u")):        # plumbing every op of that kind needs; the model never has to write it
            if k_ in params and k_ not in new_args:
                new_args[k_] = ref_
        out = op.get("out", "image")
        if not isinstance(out, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", out) or out in BUFFERS:
            fixes.append(f"{w}: output name {out!r} not usable, 'image' used" if name not in ("upscale_matte",) else f"{w}: output name {out!r} not usable, 'm' used")
            out = "image" if name != "upscale_matte" else "m"
        kept.append({"op": name, "out": out, "args": new_args})
        defined.add(out)
    if len(ops) > max_ops:
        fixes.append(f"ops: {len(ops) - max_ops} ops past the first {max_ops} dropped")
    # a raw `$matte` read by a drawer: the engine needs it at the footage size -> make `m` with upscale_matte (plumbing, logged)
    if any(isinstance(o["args"].get("matte"), str) and o["args"]["matte"] == "$matte" for o in kept) and not any(o["op"] == "upscale_matte" for o in kept):
        kept.insert(0, {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [1080, 1920], "guide": "$footage"}})
        for o in kept[1:]:
            if o["args"].get("matte") == "$matte":
                o["args"]["matte"] = "$m"
        fixes.append(f"{shot + '.' if shot else ''}ops: a raw $matte was used; upscale_matte -> $m added")
    # drop ops whose inputs are no longer made (a dropped op's reader goes with it)
    final, made = [], set(extra_buffers)
    kinds = dict(BUF_KIND)
    tint_outs: set = set()
    alias: dict = dict(pre_alias)

    def bypass(o, why):
        """drop o; a picture filter is bypassed (its output name now means its primary input) so later ops keep working."""
        pk = PRIMARY.get(o["op"])
        v = o["args"].get(pk) if pk else None
        if isinstance(v, str) and v.startswith("$") and (v[1:] in made or v[1:] in _bufs()):
            alias[o["out"]] = v[1:]
            if kinds.get(v[1:]):
                kinds[o["out"]] = kinds[v[1:]]
            fixes.append(f"{shot + '.' if shot else ''}{o['op']} -> {o['out']}: {why}; bypassed (later ops read its input instead)")
        else:
            fixes.append(f"{shot + '.' if shot else ''}{o['op']} -> {o['out']}: {why}, dropped")

    last_rgb = {"name": "footage"}
    for o in kept:
        o["args"] = _rewrite(o["args"], alias)
        for k_, v_ in list(o["args"].items()):
            if v_ == "$@prev":
                o["args"][k_] = "$" + last_rgb["name"]
        rs = set()
        _collect(o["args"], rs)
        miss = {r for r in rs if r not in made and r not in _bufs()}
        if miss:
            bypass(o, f"reads {sorted(miss)} which no kept op makes")
            continue
        scalar_drop = None
        for k, want in KIND_IN.get(o["op"], {}).items():           # a number / bool / timeline where a picture is needed (matte = 1, fg_matte = true, alpha = {"anim": ..})
            v = o["args"].get(k)
            colour_tl = want == "rgb" and isinstance(v, dict) and isinstance(v.get("timeline"), list) and v["timeline"] and isinstance(v["timeline"][0][1], list)    # a colour over time is a colour
            if (isinstance(v, bool) or _is_num(v) or (isinstance(v, dict) and any(x in v for x in ("timeline", "curve", "anim")))) and (o["op"], k) not in SCALAR_OK and not colour_tl:
                par = _sig(o["op"]).parameters.get(k) if _sig(o["op"]) else None
                if par is not None and par.default is not inspect.Parameter.empty:
                    del o["args"][k]
                    fixes.append(f"{shot + '.' if shot else ''}{o['op']} -> {o['out']}: {k} is a number / timeline but a {want} picture is needed; the argument is left out")
                elif k == "type_a":
                    o["args"][k] = "$type_matte"
                elif k == "matte" and "m" in made:
                    o["args"][k] = "$m"
                else:
                    scalar_drop = f"{k} is a number / timeline but a {want} picture is needed"
        if scalar_drop:
            bypass(o, scalar_drop)
            continue
        wrong = [(k, v, kinds.get(v[1:]), want) for k, want in KIND_IN.get(o["op"], {}).items() for v in [o["args"].get(k)]
                 if isinstance(v, str) and v.startswith("$") and kinds.get(v[1:]) in ("rgb", "gray") and kinds.get(v[1:]) != want]
        drop = False
        for k, v, got, want in wrong:                   # field-level: an optional argument is removed, a drawer's colour / alpha / matte gets the plain default, else the op goes
            where = f"{shot + '.' if shot else ''}{o['op']} -> {o['out']}"
            par = _sig(o["op"]).parameters.get(k) if _sig(o["op"]) else None
            if par is not None and par.default is not inspect.Parameter.empty:
                del o["args"][k]
                fixes.append(f"{where}: {k} = {v} is a {got} picture but {want} is needed; the argument is left out")
            elif k == "type_rgb":
                tint = next((n for n in reversed(list(made)) if kinds.get(n) == "rgb" and n in tint_outs), None)
                o["args"][k] = f"${tint}" if tint else [0.93, 0.93, 0.93]
                fixes.append(f"{where}: type_rgb = {v} is a {got} picture; {o['args'][k]} used instead")
            elif k == "type_a":
                o["args"][k] = "$type_matte"
                fixes.append(f"{where}: type_a = {v} is a {got} picture; $type_matte used instead")
            elif k == "matte" and "m" in made:
                o["args"][k] = "$m"
                fixes.append(f"{where}: matte = {v} is a {got} picture; $m used instead")
            else:
                drop = f"{k} = {v} is a {got} picture but {want} is needed"
                break
        if drop:
            bypass(o, drop)
            continue
        if o["op"] in BLEND:                             # screen / multiply / overlay / soft_light: both pictures of the same kind (or one a number)
            ka = kinds.get(o["args"]["a"][1:]) if isinstance(o["args"].get("a"), str) and o["args"]["a"].startswith("$") else None
            kb = kinds.get(o["args"]["b"][1:]) if isinstance(o["args"].get("b"), str) and o["args"]["b"].startswith("$") else None
            if ka and kb and ka != kb:
                bypass(o, f"a is a {ka} picture and b a {kb} picture; they cannot be blended")
                continue
        if o["op"] == "tint_type":
            tint_outs.add(o["out"])
        final.append(o)
        made.add(o["out"])
        kd = KIND_OUT.get(o["op"])
        if kd is None and o["op"] in KIND_FOLLOWS:
            src = o["args"].get(KIND_FOLLOWS[o["op"]])
            kd = kinds.get(src[1:]) if isinstance(src, str) and src.startswith("$") else ("rgb" if isinstance(src, list) else None)
        if kd:
            kinds[o["out"]] = kd
            if kd == "rgb":
                last_rgb["name"] = o["out"]
        else:
            kinds.pop(o["out"], None)
    final = sweep_unresolved(final, extra_buffers, fixes, shot)            # last guard: no op may read a buffer that nothing before it makes
    if not any(o["op"] in DRAWERS or (o["op"] == "over" and _reads_type(o)) for o in final):
        return None, fixes + [f"{shot + '.' if shot else ''}ops: no op draws the type (sandwich / depth_occlude / knockout / over of a type pass) -> the whole chain is rejected"]
    if final and final[-1]["out"] != "image":
        last = final[-1]
        if kinds.get(last["out"]) == "rgb" and not any(last["out"] in _refs_of(o) for o in final):         # the finished picture was named something else: plumbing
            fixes.append(f"{shot + '.' if shot else ''}ops: the last op wrote {last['out']!r}, not `image`; renamed to `image`")
            last["out"] = "image"
    if not any(o["out"] == "image" for o in final):
        return None, fixes + [f"{shot + '.' if shot else ''}ops: no op writes `image`"]
    return final, fixes


def sweep_unresolved(ops, extra_buffers=(), fixes=None, shot: str = ""):
    """the ops whose every `$name` is a source buffer (BUFFERS minus the unavailable ones, extra_buffers) or the `out` of an EARLIER kept op. An op with an unresolved
    reference is dropped (logged in `fixes`); a later op that read its output is then unresolved too and goes with it (unless an earlier op made the same name)."""
    have = set(extra_buffers) | _bufs()
    kept = []
    for o in ops:
        miss = sorted(_refs_of(o) - have)
        if miss:
            if fixes is not None:
                fixes.append(f"{shot + '.' if shot else ''}{o['op']} -> {o.get('out', 'image')}: reads {miss} which nothing before it makes, dropped")
            continue
        kept.append(o)
        have.add(o.get("out", "image"))
    return kept


def _refs_of(o: dict) -> set:
    rs: set = set()
    _collect(o["args"], rs)
    return rs


def _reads_type(o: dict) -> bool:
    rs = set()
    _collect(o["args"], rs)
    return any(r.startswith("type_") for r in rs)


def _collect(obj, acc):
    if isinstance(obj, str) and obj.startswith("$"):
        acc.add(obj[1:])
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def prompt_text(cut: float) -> str:
    """the COMPOSE BUILDING BLOCKS section of the design prompt: the buffers, the kinds, the timeline grammar (the first two sections of 16_app/PIECE_REFERENCE_syntax.md) and the op menu."""
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / "16_app" / "PIECE_REFERENCE_syntax.md"
    notes = ""
    if f.is_file():
        t = f.read_text(encoding="utf-8")
        i = t.find("References and kinds")
        j = t.find("Type and timing")
        notes = t[i:j].strip() if i >= 0 else ""
    return ("Buffers a chain may read before any op has made them: $footage (the shot), $matte (the subject cut-out of the shot, 0..1; only if the shot has one), $depth, $flow and $flow_m "
            "(footage motion), $type_matte / $type_line / $type_depth / $type_beauty (the type passes, aligned to piece time), $frame_idx, $t, $u. Every op writes its result to `out`; a later op reads it as \"$out\".\n"
            + notes + "\n\nThe blocks (signature and doc from tools/fusion/compose.py):\n" + menu(cut))


def normalize_compose(comp_in, shot_ids, cut: float, unavailable: dict | None = None):
    """the model's `compose` object -> ({"default": {"ops"}, "shots": {sid: {"ops"}}} | {}, fixes). Every chain is validated alone; a chain that does not survive is
    left out (the caller uses the rules' chain for that shot). Unknown shot ids are dropped. unavailable = {shot id: {"matte", "depth", ...}}: passes the generated footage of
    that shot does not have (the app makes the footage before the design step); the default chain may only read what EVERY shot has."""
    fixes: list[str] = []
    out: dict = {}
    if not isinstance(comp_in, dict):
        return out, ["compose: not an object"]
    d = comp_in.get("default")
    if isinstance(d, dict) and d.get("ops") is not None:
        ops, fx = validate_ops(d.get("ops"), cut, shot="default", unavailable=set().union(*[set(v) for v in (unavailable or {}).values()]) if unavailable else ())
        fixes += fx
        if ops:
            out["default"] = {"ops": ops}
    sh = comp_in.get("shots")
    if isinstance(sh, dict):
        for sid, v in sh.items():
            if sid not in shot_ids:
                fixes.append(f"compose.shots.{sid}: no such shot, dropped")
                continue
            ops, fx = validate_ops((v or {}).get("ops") if isinstance(v, dict) else None, cut, shot=sid, unavailable=(unavailable or {}).get(sid, ()))
            fixes += fx
            if ops:
                out.setdefault("shots", {})[sid] = {"ops": ops}
    return out, fixes


def passes_of(ops) -> set[str]:
    """type passes the chain reads ($type_matte -> matte ...)."""
    rs: set = set()
    _collect([o.get("args", {}) for o in ops], rs)
    return {PASS_FOR[r] for r in rs if r in PASS_FOR}


def uses(ops, buf: str) -> bool:
    rs: set = set()
    _collect([o.get("args", {}) for o in ops], rs)
    return buf in rs


def count_timelines(ops) -> int:
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if "timeline" in o or "curve" in o or "anim" in o:
                n += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk([x.get("args", {}) for x in ops])
    return n


def deep(o):
    return copy.deepcopy(o)
