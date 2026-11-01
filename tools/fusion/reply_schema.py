"""reply_schema: JSON Schemas of the two open-model replies (the app's DESIGN reply and the CONCEPT reply), built from the code's own constants, so a
llama.cpp grammar can make a wrong TYPE or VALUE impossible at generation time (user 2026-10-29: "can't you get Qwen to return JSON correctly? It is not
consistent": the generic `json_object` grammar only guarantees valid syntax; a model still wrote `type_passes` as the string "all", `form_in` as "gradient_fade",
a palette as floats, `transitions` as a string of names, a bracket slip in the middle of a long reply ...). Pure functions; NO model call, no network.

    design_schema(words=None, techniques=None, shot_ids=None, ...)   the reply of qwen_piece.schema_text (concept / system / type / compose)
    concept_schema(n_words=None, n_shots=None, ...)                  the reply of concept.concept_prompt
    type_passes_schema()                                             the one place a model may write `type_passes` (a LIST of known pass names)
    schema_for(name, prompt=None, **kw)                              "design" | "concept" | "none" -> dict | None; with a prompt the exact counts are read from it
    design_context(prompt) / concept_context(prompt)                 words, techniques, shot ids / word and shot counts, read from the prompt text
    to_gbnf(schema)                                                  the GBNF llama.cpp will sample with (None when llama_cpp is not importable)
    violations(obj, schema)                                          what the grammar would have forbidden (jsonschema; the measurement of 16_app/REPLY_SCHEMA.md)
    unsupported_keywords(schema)                                     schema keywords outside what llama.cpp's json-schema-to-grammar converter handles ([] = fine)

WHAT THE GRAMMAR ENFORCES (llama_cpp.llama_grammar.json_schema_to_gbnf, JamePeng llama-cpp-python 0.4.5, read 2026-10-29): types, enum / const, anyOf, nested
objects with the declared keys (REQUIRED keys first in schema order, then the optional ones in schema order; an undeclared key is impossible: additionalProperties
is closed everywhere except the free-form `args` of a compose op and the axis / role maps), array item types, minItems / maxItems, INTEGER minimum / maximum (rgb
0..255, bpm), string `pattern` (concept lines: upper case, <= 26 chars). NOT enforced by the converter: minimum / maximum of a `number` (the RANGES the code clamps
anyway are written into the schema for `violations()` and the docs, the grammar ignores them), uniqueItems, minLength on large values (never used: it unrolls).
Nothing here uses $ref, allOf, oneOf, format, exclusiveMinimum, if/then or patternProperties.
"""
from __future__ import annotations

import copy
import json
import re

TYPE_PASSES = ("beauty", "matte", "depth", "line", "clay", "flow")        # tools/fusion/spec.PASSES (checked by tests/test_reply_schema.py)
WORD_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")                   # roles of `system.pattern`
LINE_PATTERN = r"^[A-Z0-9 '!?.,&-]{1,26}$"                               # concept.normalize: _CLEAN allows these characters, MAX_LINE_CHARS 26
SUPPORTED = {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const", "anyOf", "minimum", "maximum", "pattern", "description"}

# ------------------------------------------------------------------------------------------------- tiny builders
S = {"type": "string"}


def _num(lo=None, hi=None) -> dict:
    d: dict = {"type": "number"}
    if lo is not None:
        d["minimum"] = lo
    if hi is not None:
        d["maximum"] = hi
    return d


def _int(lo=None, hi=None) -> dict:
    d: dict = {"type": "integer"}
    if lo is not None:
        d["minimum"] = lo
    if hi is not None:
        d["maximum"] = hi
    return d


def _enum(vals) -> dict:
    return {"enum": list(vals)}


def _arr(item: dict, lo=None, hi=None) -> dict:
    d: dict = {"type": "array", "items": item}
    if lo is not None:
        d["minItems"] = lo
    if hi is not None:
        d["maxItems"] = hi
    return d


def _obj(props: dict, required=(), closed: bool = True) -> dict:
    d: dict = {"type": "object", "properties": props}
    if required:
        d["required"] = list(required)
    if not closed:
        d["additionalProperties"] = True
    return d


def _rng(key: str):
    """(lo, hi) of qwen_piece.RANGES[key]."""
    from tools.fusion import qwen_piece as QP
    return QP.RANGES[key]


def _n(key: str) -> dict:
    lo, hi = _rng(key)
    return _num(lo, hi)


def _rgb255() -> dict:
    return _arr(_int(0, 255), 3, 3)


def type_passes_schema() -> dict:
    """a `type_passes` value (or a control `pass`): a LIST of known pass names; the string "all" / "depth+line" cannot be written."""
    return _arr(_enum(TYPE_PASSES), 1, len(TYPE_PASSES))


# ------------------------------------------------------------------------------------------------- the design reply
def _cut_kinds() -> list[str]:
    from tools.fusion import design_options as DO
    return [k for k in DO.CUT_KINDS if getattr(DO, "DOLLY_ON", False) or not str(k).startswith("depth_dolly")]     # the menu does not offer the dolly while it is off


def _cuts_schema(shot_ids=None, require_why: bool = False, kinds=None) -> dict:
    """compose.cuts. kinds = the cut kinds the piece's system drew (design_context reads them from the prompt): only those (+ hard / crossfade) are in the grammar; None = the whole vocabulary."""
    from tools.fusion import design_options as DO
    from tools.fusion import cut_kinds as CK
    after = _enum(shot_ids[:-1]) if shot_ids and len(shot_ids) > 1 else dict(S)
    exits = sorted({e for v in DO.EXITS.values() for e in v})
    cut_enum = [k for k in (list(kinds) + ["hard", "crossfade"]) if k in CK.ALL] if kinds else _cut_kinds()
    props = {"after": after, "cut": _enum(list(dict.fromkeys(cut_enum))), "why": dict(S), "word": _int(0, None), "exit": _enum(exits), "freeze_s": _num(0.0, 0.5),
             "zoom_s": _num(0.5, 1.8), "letter": _int(0, None), "style": _enum(list(CK.STYLES)), "dir": _enum(list(CK.DOLLY_DIRS))}      # style / dir: the optional look of a 3D cut (cut_kinds.style_of)
    req = ["after", "cut"] + (["why"] if require_why else [])
    return _arr(_obj(props, req), None, max(len(shot_ids) - 1, 0) if shot_ids else None)


def _entry_prop() -> dict:
    """`entry`: the side the strands of a `lines` word come from (tools/fusion/openings.ORDER; offered by qwen_piece.schema_text since 2026-10-29); absent when the code has no such list."""
    from tools.fusion import qwen_piece as QP
    order = getattr(getattr(QP, "OP", None), "ORDER", None)
    return {"entry": _enum(order)} if order else {}


def word_schema(text: str | None = None, fonts=None, require_why: bool = True) -> dict:
    """one entry of type.words (qwen_piece.schema_text). text = the locked word (const) or None (any string)."""
    from tools.fusion import owner as OW
    from tools.fusion import qwen_piece as QP
    fonts = list(fonts) if fonts else [f["file"] for f in QP.liked_fonts()]
    placement = _obj({"yaw": _n("yaw"), "wf": _n("wf"), "hf": _n("hf"), "pitch": _n("pitch"), "roll": _n("roll"), "dx": _n("dx"), "dy": _n("dy")}, ["yaw", "wf", "hf"])
    axis_val = {"anyOf": [_num(), _arr({"anyOf": [_num(), {"const": "beat"}]}, 2, 3)]}
    props = {
        "text": {"const": text} if text is not None else dict(S),
        "why": dict(S),
        "font": _enum(fonts),
        "module": _enum(QP.WORD_MODULES),
        "hit": _num(0.0, None),
        "hold_s": _n("hold_s"), "form_s": _n("form_s"), "exit_s": _n("exit_s"),
        "form_in": _enum(QP.IO_STYLES), "form_out": _enum(QP.IO_STYLES),
        "placement": placement,
        "lines": _arr(dict(S), 0, 6),
        **_entry_prop(),
        "screen_offset": _arr(_num(-0.3, 0.3), 2, 2),
        "axes": {"type": "object", "additionalProperties": axis_val},
        "matte": _enum(QP.MATTES),
        "render": _enum(OW.RENDERS),
        "ltx_at": _num(*OW.LTX_AT_RANGE),
        "material": dict(S),
        "window_source": dict(S),
    }
    req = ["text"] + (["why"] if require_why else []) + ["font", "module", "hit", "hold_s", "form_s", "exit_s", "form_in", "form_out", "placement"]
    return _obj(props, req)


def design_schema(words: list[str] | None = None, techniques: list[str] | None = None, shot_ids: list[str] | None = None, *, require_why: bool = True,
                  require_system: bool = True, lock_text: bool = True, cut_kinds: list[str] | None = None) -> dict:
    """the design reply (qwen_piece.schema_text). words = the piece's words in order (fixes the entry count and, with lock_text, each entry's `text`; None = any count),
    techniques = the compose techniques the footage supports (None = all), shot_ids = ids of the SHOTS list (compose.cuts[].after).
    require_why: the `why` fields are required keys (they are read; the grammar then makes the model write them BEFORE the choice they explain). require_system: `system`
    is required (the prompt calls it optional but strongly wanted; an optional key would be generated LAST, after the words that should follow it)."""
    from tools.fusion import qwen_piece as QP
    techniques = list(techniques) if techniques else list(QP.TECHNIQUES)
    fonts = [f["file"] for f in QP.liked_fonts()]
    font = _enum(fonts)
    if words:
        n = len(words)
        words_arr = _tuple([word_schema(t, fonts, require_why) for t in words]) if lock_text else _arr(word_schema(None, fonts, require_why), n, n)
        gaps = max(n - 1, 0)
        tr_items = {"enum": list(QP.TRANSITIONS) + [None]}
        transitions = {"anyOf": [{"const": "auto"}, _arr(tr_items, gaps, gaps)]}
    else:
        words_arr = _arr(word_schema(None, fonts, require_why), 1, 12)
        transitions = {"anyOf": [{"const": "auto"}, _arr({"enum": list(QP.TRANSITIONS) + [None]}, 0, 11)]}
    system = _obj({
        "rule": dict(S), "roles": {"type": "object", "additionalProperties": font}, "pattern": _arr(_enum(WORD_LETTERS), 1, 12),
        "yaw": _arr(_n("yaw"), 1, 12),
        "axis": _obj({"font": font, "tag": dict(S), "windows": _arr(_arr(_num(), 2, 2), 1, 6)}, ["font", "tag", "windows"]),
        "progression": dict(S), "why": dict(S)}, ["rule", "roles", "pattern"] + (["why"] if require_why else []))
    camera_params = _obj({"read_dps": _n("read_dps"), "intro_deg": _n("intro_deg"), "fast_dps": _n("fast_dps"), "elev": _n("elev"), "bob": _n("bob")})
    typ = _obj({
        "mood": dict(S), "energy": _n("energy"),
        "palette": _obj({"bg": _rgb255(), "fg": _rgb255(), "accents": _arr(_rgb255(), 3, 4)}, ["bg", "fg", "accents"]),
        "camera": _enum(QP.CAMERAS), "camera_params": camera_params, "transitions": transitions, "line": _obj({"px": _n("line.px")}, ["px"]),
        "words": words_arr}, ["mood", "energy", "palette", "camera", "transitions", "line", "words"])
    op = _obj({"op": _enum(_compose_ops()), "out": dict(S), "args": {"type": "object"}}, ["op", "out"])
    chain = _obj({"why": dict(S), "ops": _arr(op, 1, 40)}, ["ops"])
    compose = _obj({
        "technique": _enum(techniques),
        "plate_gain": _n("plate_gain"), "plate_sat": _n("plate_sat"), "glow_lo": _n("glow_lo"), "glow_hi": _n("glow_hi"), "glow_color": _arr(_num(0.0, 1.0), 3, 3),
        "grain": _n("grain"), "vignette": _n("vignette"), "type_opacity": _n("type_opacity"),
        "default": chain,
        "shots": {"type": "object", "additionalProperties": _obj({"why": dict(S), "ops": _arr(op, 1, 40)}, ["ops"])},
        "holds": _arr(_arr(_num(0.0, None), 2, 2), 0, 6),
        "cuts": _cuts_schema(shot_ids, require_why=False, kinds=cut_kinds)},
        ["technique", "plate_gain", "plate_sat", "glow_lo", "glow_hi", "glow_color", "grain", "vignette", "type_opacity"])
    top = _obj({"concept": dict(S), "idea_why": dict(S), "treatment_notes": dict(S), "system": system, "type": typ, "compose": compose},
               ["concept"] + (["idea_why"] if require_why else []) + ["treatment_notes"] + (["system"] if require_system else []) + ["type", "compose"])
    return top


def _tuple(items: list[dict]) -> dict:
    """a fixed-length array with one schema per position: the converter turns `items: [..]` into a tuple rule (exactly len(items) entries, in order); `validator_schema`
    rewrites it to prefixItems for the jsonschema package. Used to lock the text of each word entry and the shape of each shot_plan entry."""
    return {"type": "array", "items": items, "minItems": len(items), "maxItems": len(items)}


def _compose_ops() -> list[str]:
    from tools.fusion import compose_blocks as CB
    return list(CB.ALLOWED)


# ------------------------------------------------------------------------------------------------- the concept reply
def shot_flag_default() -> str:
    """the name of the per-shot place flag the concept prompt asks for TODAY: `same_place` (same place from another camera) or the old `continues`."""
    from tools.fusion import concept as C
    return "same_place" if '"same_place"' in C.concept_prompt(0, 16, [], "cut") else "continues"


# ------------------------------------------------------------------------------------------------- PLATES AND ANGLES (added 2026-10-29, separate block)
# Optional keys of a concept `shot_plan` entry for the two new footage options (16_app/ANGLE_STILLS.md, 16_app/CLEAN_PLATE.md). The concept prompt does not ask for them yet
# (concept.py is another worker's file: the lines to add are in the guide section "Plates and angles"), so they are optional here: a model that is told about them can write
# them, one that is not never does. Values mirror tools/fusion/plate_layers.py (EASES, ranges) and tools/fusion/angle_still.py (FRAMING_CLAUSES).
def plate_angle_keys() -> dict:
    from tools.fusion import explore_effects as EE
    from tools.fusion import angle_still as AS
    from tools.fusion import plate_layers as PLT
    move = _obj({"dx": _num(-0.5, 0.5), "dy": _num(-0.5, 0.5), "scale": _num(0.5, 3.0), "ease": _enum(PLT.EASES), "anchor": _arr(_num(0.0, 1.0), 2, 2), "feather": _num(0.0, 0.02)})
    pan = _obj({"dx": _num(-0.25, 0.25), "dy": _num(-0.25, 0.25), "scale": _num(1.0, 2.0), "ease": _enum(PLT.EASES)})
    return {"same_subject_as": dict(S), "framing": {"type": "string", "description": "close-up | low | profile | top-down | wide | ... (" + ", ".join(AS.FRAMING_CLAUSES) + ")"},
            "plate": _enum(["clean"]), "type_matte": _enum(PLT.TYPE_MATTE_ALL), "subject_move": move, "plate_pan": pan, "plate_subject": dict(S),
            "fx": _enum(list(EE.NAMES))}          # 2026-10-30: one of the hearted explore looks on this shot (tools/fusion/explore_effects.py; guide section 24)


def concept_schema(n_words: int | None = None, n_shots: int | None = None, *, require_why: bool = True, free_counts: bool = False, shot_flag: str | None = None) -> dict:
    """the concept reply (concept.concept_prompt). n_words: the exact number of `words` entries (multi-shot: one lockup per shot; None = a single-shot piece, 5..7 free
    lines); n_shots = the exact number of `shot_plan` entries (default n_words, else 1). free_counts: any number (1..12) of words / shot_plan / owners entries (a prompt whose
    counts are unknown, e.g. an archived reply measured without its prompt); the shot flag is then optional. shot_flag: the key of the boolean place flag of shot_plan
    entries 2.. (default: what concept_prompt asks for today, `same_place`; the old runs' prompts said `continues`)."""
    flag = shot_flag or shot_flag_default()
    from tools.fusion import concept as C
    from tools.fusion import owner as OW
    ns = n_shots if n_shots else (n_words if n_words else 1)
    line = {"type": "string", "pattern": LINE_PATTERN}
    words = _arr(line, 1, 12) if free_counts else _arr(line, n_words, n_words) if n_words else _arr(line, C.MIN_LINES, C.MAX_LINES)
    n_own = n_words if n_words else None
    owner = _obj({"render": _enum(OW.RENDERS), "why": dict(S), "ltx_at": _num(*OW.LTX_AT_RANGE), "material": dict(S)}, ["render"] + (["why"] if require_why else []))
    owners = _arr(owner, 1, 12) if free_counts else _arr(owner, n_own, n_own) if n_own else _arr(owner, C.MIN_LINES, C.MAX_LINES)

    def shot(k: int) -> dict:
        props = {"prompt": dict(S), "camera": dict(S), "why": dict(S), **plate_angle_keys()}      # + the optional angle-still / clean-plate keys (block below)
        req = ["prompt", "camera"] + (["why"] if require_why else [])
        if free_counts:                                           # either name, both optional
            props["same_place"] = props["continues"] = {"type": "boolean"}
        elif k > 0:
            props[flag] = {"type": "boolean"}
            req.append(flag)
        return _obj(props, req)
    shot_plan = _arr(shot(0), 1, 12) if free_counts else _tuple([shot(k) for k in range(ns)]) if ns > 1 else _arr(shot(0), 1, 1)
    props = {"title": dict(S), "words": words, "subject": dict(S), "subject_desc": dict(S), "scene": dict(S), "music": dict(S), "bpm": _int(*C.BPM_RANGE), "mood": dict(S), "look": _enum(C.LOOKS),
             "shots": _enum(C.MODES), "idea": dict(S), "idea_why": dict(S), "shot_plan": shot_plan, "owners": owners}
    req = ["title", "words", "subject", "scene", "music", "bpm", "mood", "look", "shots", "idea"] + (["idea_why"] if require_why else []) + ["shot_plan", "owners"]
    return _obj(props, req)


# ------------------------------------------------------------------------------------------------- the schema for a prompt
def design_context(prompt: str) -> dict:
    """words, techniques and shot ids as the design prompt states them (qwen_piece.build_prompt); a key is missing when the prompt does not show it."""
    ctx: dict = {}
    m = re.search(r"^Words, in order, each shown once: (.+)$", prompt or "", re.M)
    if m:
        ctx["words"] = [w.strip() for w in m.group(1).split(" / ") if w.strip()]
    m = re.search(r'"technique": "<EXACTLY one of: ([^>]+)>"', prompt or "")
    if m:
        ctx["techniques"] = [t.strip() for t in m.group(1).split(",") if t.strip()]
    m = re.search(r'^\s*"cut": "([A-Za-z_:| ]+)",', prompt or "", re.M)                      # the cut kinds of the prompt's cuts schema line (the piece's drawn kinds, or the whole vocabulary)
    if m:
        from tools.fusion import cut_kinds as CK
        ks = [k.strip() for k in m.group(1).split("|") if k.strip()]
        if ks and all(k in CK.ALL for k in ks):
            ctx["cut_kinds"] = ks
    m = re.search(r"^=== SHOTS \(footage, fixed\) ===\n(.*?)(?:\n\n|\nImages attached|\n===)", prompt or "", re.M | re.S)
    if m:
        ids = re.findall(r"^- ([A-Za-z0-9_]+): starts", m.group(1), re.M)
        if ids:
            ctx["shot_ids"] = ids
    return ctx


def concept_context(prompt: str) -> dict:
    """n_words / n_shots as the concept prompt states them (`words` has EXACTLY n entries / `shot_plan` has EXACTLY n entr...; absent = a single-shot piece) and the name of the
    shot place flag (`same_place` | `continues`) the prompt asks for."""
    ctx: dict = {}
    m = re.search(r"`words` has EXACTLY (\d+) entries", prompt or "")
    if m:
        ctx["n_words"] = int(m.group(1))
    m = re.search(r"`shot_plan` has EXACTLY (\d+) entr", prompt or "")
    if m:
        ctx["n_shots"] = int(m.group(1))
    if '"same_place"' in (prompt or ""):
        ctx["shot_flag"] = "same_place"
    elif '"continues"' in (prompt or ""):
        ctx["shot_flag"] = "continues"
    return ctx


NAMES = ("none", "design", "concept", "design_repair")
REPAIR_MARK = "=== YOUR PREVIOUS REPLY (rejected) ==="          # design_step.repair_prompt: the LONG repair prompt (the whole design prompt + the rejected reply + the errors)


def schema_for(name: str | None, prompt: str | None = None, **kw) -> dict | None:
    """'design' | 'concept' -> the schema (exact counts from the prompt text when one is given; keyword arguments override); 'none' / '' / None -> None."""
    n = str(name or "none").strip().lower()
    if n in ("none", "off", ""):
        return None
    if n == "design":
        ctx = design_context(prompt or "")
        ctx.update(kw)
        return design_schema(**ctx)
    if n == "concept":
        ctx = concept_context(prompt) if prompt else {"free_counts": True}
        ctx.update(kw)
        return concept_schema(**ctx)
    if n == "design_repair":
        # ONE graph node serves both repairs and cannot know which prompt it gets (design_step.parse decides at run time): the LONG repair (REPAIR_MARK in the prompt, the whole design
        # prompt + the rejected reply; the reply is a whole design object) gets the design schema, the SHORT repair (design_salvage.repair_prompt; the reply is a {path: value} map) none.
        ctx = design_context(prompt or "")
        if REPAIR_MARK not in (prompt or "") or "words" not in ctx:
            return None
        ctx.update(kw)
        return design_schema(**ctx)
    raise ValueError(f"schema must be one of {NAMES}, got {name!r}")


# ------------------------------------------------------------------------------------------------- grammar + measurement
def converter():
    """llama.cpp's json_schema_to_gbnf from the installed llama-cpp-python, or from the file named by $REPLY_SCHEMA_CONVERTER (a copy of llama_cpp/llama_grammar.py);
    None when neither exists."""
    import os
    try:
        from llama_cpp.llama_grammar import json_schema_to_gbnf
        return json_schema_to_gbnf
    except Exception:       # noqa: BLE001  (not installed: CPU PC)
        pass
    p = os.environ.get("REPLY_SCHEMA_CONVERTER")
    if p and os.path.isfile(p):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_reply_schema_llama_grammar", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.json_schema_to_gbnf
    return None


def to_gbnf(schema: dict) -> str | None:
    fn = converter()
    return fn(json.dumps(schema)) if fn else None


def unsupported_keywords(schema, path: str = "") -> list[str]:
    """every key of the schema (outside `properties` names / `enum` / `const` values) that is not in SUPPORTED."""
    bad: list[str] = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k not in SUPPORTED:
                bad.append(f"{path}/{k}")
            if k == "properties" and isinstance(v, dict):
                for pk, pv in v.items():
                    bad += unsupported_keywords(pv, f"{path}/properties/{pk}")
            elif k in ("items", "additionalProperties") and isinstance(v, (dict, list)):
                for j, x in enumerate(v if isinstance(v, list) else [v]):
                    bad += unsupported_keywords(x, f"{path}/{k}" + (f"[{j}]" if isinstance(v, list) else ""))
            elif k == "anyOf":
                for j, x in enumerate(v):
                    bad += unsupported_keywords(x, f"{path}/anyOf[{j}]")
    return bad


_SOFT = ("minimum", "maximum")


def validator_schema(schema: dict) -> dict:
    """the same schema in Draft 2020-12 for the `jsonschema` package, read the way llama.cpp's converter reads it: the tuple form (`items` = a list) becomes `prefixItems`, and an
    object with `properties` and no `additionalProperties` is closed."""
    s = copy.deepcopy(schema)

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n and "additionalProperties" not in n:
                n["additionalProperties"] = False          # the converter's reading: a declared-keys object is closed (jsonschema's default is open)
            if isinstance(n.get("items"), list):
                n["prefixItems"] = [walk(x) for x in n.pop("items")]
                n["items"] = False
            for k, v in list(n.items()):
                if k != "prefixItems":
                    n[k] = walk(v)
        elif isinstance(n, list):
            return [walk(x) for x in n]
        return n
    return walk(s)


def _norm_path(parts) -> str:
    out = ""
    for p in parts:
        out += "[]" if isinstance(p, int) else ("." + str(p) if out else str(p))
    return out


def violations(obj, schema: dict) -> list[dict]:
    """-> [{path, kind, msg}] for everything in `obj` the schema forbids. path = dotted with [] for list indices (type.words[].form_in). kind: type | enum | const | missing |
    extra | length (minItems / maxItems / tuple size) | pattern | int_range (grammar-enforced) | range (a `number` bound: the code clamps it, the grammar cannot) | other."""
    import jsonschema
    out: list[dict] = []
    v = jsonschema.Draft202012Validator(validator_schema(schema))
    for e in v.iter_errors(obj):
        base = list(e.absolute_path)
        val = e.validator
        msg = e.message
        if val == "anyOf" and e.context:
            best = max(e.context, key=lambda c: (len(c.relative_path), c.validator != "type"))
            if best is not None:
                base += list(best.relative_path)
                val, msg = best.validator, best.message
        if val == "required":
            m = re.match(r"'([^']+)' is a required property", msg)
            out.append(dict(path=_norm_path(base + ([m.group(1)] if m else [])), kind="missing", msg=msg[:160]))
        elif val == "additionalProperties":
            for k in re.findall(r"'([^']+)'", msg.split("(")[-1]) if "(" in msg else []:
                out.append(dict(path=_norm_path(base + [k]), kind="extra", msg=msg[:160]))
            if not out or out[-1]["kind"] != "extra":
                out.append(dict(path=_norm_path(base), kind="extra", msg=msg[:160]))
        elif val in ("minimum", "maximum"):
            integer = isinstance(e.instance, int) and not isinstance(e.instance, bool) and e.schema.get("type") == "integer"
            out.append(dict(path=_norm_path(base), kind="int_range" if integer else "range", msg=msg[:160]))
        elif val == "type":
            out.append(dict(path=_norm_path(base), kind="type", msg=msg[:160]))
        elif val in ("enum", "const"):
            out.append(dict(path=_norm_path(base), kind=val, msg=msg[:160]))
        elif val in ("minItems", "maxItems", "prefixItems", "items") or "expected" in msg:
            out.append(dict(path=_norm_path(base), kind="length", msg=msg[:160]))
        elif val == "pattern":
            out.append(dict(path=_norm_path(base), kind="pattern", msg=msg[:160]))
        else:
            out.append(dict(path=_norm_path(base), kind="other", msg=f"{val}: {msg}"[:160]))
    return out


VALUE = ("type", "enum", "const", "pattern", "int_range")         # a field that is there has the wrong type / value (what the user asked to make impossible)
STRUCT = ("missing", "extra", "length", "other")                 # a key is missing / not declared / a list has the wrong length
HARD = VALUE + STRUCT                                            # everything the grammar makes impossible


# ------------------------------------------------------------------------------------------------- the measurement over the reply archive
def _read(p) -> str | None:
    from pathlib import Path
    q = Path(p)
    return q.read_text(encoding="utf-8", errors="replace") if q.is_file() else None


def _row_inputs(row: dict, root) -> list[tuple[str, str, str | None, None]]:
    """-> [(kind, reply text, prompt text | None, None)] of one replies.jsonl row (kind design | concept)."""
    from pathlib import Path
    root = Path(root)
    d = root / row["dir"]
    out = []
    if row["source"] == "app":
        pre = row["prefix"]
        for kind in ("design", "concept"):
            t = _read(d / f"{pre}_{kind}_reply.txt")
            if t is None:
                continue
            prompts = sorted(d.glob(f"{pre}_{kind}_prompt*.txt"))
            out.append((kind, t, _read(prompts[0]) if prompts else None, None))
    elif row.get("level") == "par_app":
        t = _read(d / "reply.txt")
        if t is not None:
            out.append(("design", t, _read(d / "prompt.txt"), None))
    elif row.get("level") == "par_concept":
        t = _read(d / "reply.txt")
        if t is not None:
            out.append(("concept", t, _read(d / "prompt.txt"), None))
    return out


def measure_archive(root=None, index="20_open-model-design/replies.jsonl") -> dict:
    """check every archived design / concept reply of the app runs and the hosted-API app-prompt cells against the schemas.
    -> {(model, kind): dict(n, syntax_ok, fields_ok_mended, lenient_ok, strict_ok, top=[(field, kind, n replies)], examples)}"""
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path
    root = Path(root or Path(__file__).resolve().parents[2])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.fusion import qwen_piece as QP
    rows = [json.loads(l) for l in (root / index).read_text(encoding="utf-8").splitlines() if l.strip()]
    acc: dict = defaultdict(lambda: dict(n=0, syntax_ok=0, parsed=0, lenient_ok=0, both_ok=0, strict_ok=0, hits=Counter(), struct_hits=Counter(), range_hits=Counter(), range_any=0, per_reply=[]))
    for r in rows:
        for kind, text, prompt, _ in _row_inputs(r, root):
            if not text.strip():
                continue
            a = acc[(r["model"], kind)]
            a["n"] += 1
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            try:
                syn = isinstance(json.loads(t), dict)
            except Exception:       # noqa: BLE001
                syn = False
            a["syntax_ok"] += int(syn)
            obj = QP.extract_json(text)
            if obj is None:
                a["per_reply"].append(dict(dir=r["dir"], prefix=r.get("prefix"), kind=kind, parsed=False))
                continue
            a["parsed"] += 1
            exact = (r.get("prompt_srcs") or {}).get(kind) == "exact"       # a reconstructed prompt carries today's words / counts, not what the run saw
            if kind == "design":
                ctx = design_context(prompt or "") if exact else {"techniques": list(QP.TECHNIQUES) + ["built_in"]}
                lenient = design_schema(require_why=False, require_system=False, **ctx)
                strict = design_schema(**ctx)
            else:
                ctx = concept_context(prompt or "") if exact and prompt else {"free_counts": True}
                lenient, strict = concept_schema(require_why=False, **ctx), concept_schema(**ctx)
            vl = violations(obj, lenient)
            vs = violations(obj, strict)
            hard_l = [v for v in vl if v["kind"] in VALUE]
            struct_l = [v for v in vl if v["kind"] in STRUCT]
            hard_s = [v for v in vs if v["kind"] in HARD]
            a["lenient_ok"] += int(not hard_l)
            a["both_ok"] += int(syn and not hard_l and not struct_l)
            a["strict_ok"] += int(not hard_s)
            for k in {(v["path"], v["kind"]) for v in struct_l}:
                a["struct_hits"][k] += 1
            soft = [v for v in vl if v["kind"] == "range"]
            a["range_any"] += int(bool(soft))
            for k in {(v["path"], v["kind"]) for v in hard_l}:
                a["hits"][k] += 1
            for k in {(v["path"], v["kind"]) for v in soft}:
                a["range_hits"][k] += 1
            a["per_reply"].append(dict(dir=r["dir"], prefix=r.get("prefix"), kind=kind, parsed=True, syntax_ok=syn, hard=[f'{v["path"]} [{v["kind"]}] {v["msg"][:90]}' for v in hard_l + struct_l][:14]))
    return dict(acc)


def measure_table(acc: dict) -> str:
    """a markdown table: model, reply, n, % JSON syntax ok as written, % whose written fields all have the right type / value (parsed with the code's own mender when the syntax
    is broken), % valid AS WRITTEN (syntax + types + values + keys + list lengths), top-5 wrong type / value fields, top-3 key / length violations."""
    lines = ["| model | reply | n | JSON syntax ok | types + values ok | fully valid as written | top-5 wrong type / value (replies) | top-3 keys / lengths (replies) |", "|---|---|---|---|---|---|---|---|"]
    for (model, kind), a in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = a["n"]
        top = ", ".join(f"`{p}` {k} x{c}" for (p, k), c in a["hits"].most_common(5)) or "none"
        st = ", ".join(f"`{p}` {k} x{c}" for (p, k), c in a["struct_hits"].most_common(3)) or "none"
        lines.append(f"| {model} | {kind} | {n} | {100 * a['syntax_ok'] / n:.0f} % | {100 * a['lenient_ok'] / n:.0f} % | {100 * a['both_ok'] / n:.0f} % | {top} | {st} |")
    return chr(10).join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="reply schemas: measure the archive / dump a schema")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="violations of the archived replies (20_open-model-design/replies.jsonl)")
    m.add_argument("--json-out")
    d = sub.add_parser("dump", help="print a schema (and its GBNF size when a converter is available)")
    d.add_argument("name", choices=("design", "concept"))
    d.add_argument("--prompt-file")
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        acc = measure_archive()
        print(measure_table(acc))
        if a.json_out:
            slim = {f"{k[0]}|{k[1]}": {kk: (vv if not hasattr(vv, "most_common") else vv.most_common(30)) for kk, vv in v.items()} for k, v in acc.items()}
            open(a.json_out, "w", encoding="utf-8").write(json.dumps(slim, indent=1))
        return 0
    prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else None
    s = schema_for(a.name, prompt)
    g = to_gbnf(s)
    print(json.dumps(s, indent=1))
    print(f"# schema {len(json.dumps(s))} chars; GBNF " + ("not available (llama_cpp not importable)" if g is None else f"{len(g)} chars, {g.count(chr(10)) + 1} rules"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
