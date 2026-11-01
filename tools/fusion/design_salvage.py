"""design_salvage: FIELD-LEVEL salvage of an open model's design reply (pure functions, no ComfyUI, no network). NO CLAUDE.

Before (3060 batch v001): one invalid field (an io style `gradient_fade`, an axis the font lacks, matte "behind", a window cut after the only shot)
sent the whole type section to the rules planner, so most of a piece's design was the rules', not the model's. Now every word and every compose field
is read on its own:

    flatten(obj, words)              reply object -> {path: value}  (paths: type.mood, type.words[1].form_in, type.words[1].placement.wf, compose.glow_hi ..)
    salvage(raw, rules, ctx)         -> Salvage(clean reply object, status per path, pending list); `rules` = the rules planner's value per path
    account(sv, final, fixes, ...)   after the piece rules (timing fit, font rules, window validator) ran: which fields are still the model's
    report(sv, ...)                  fields_from_model / fields_fixed / fields_from_rules, with the paths, and the % the model kept
    repair_prompt(sv, ...)           the SHORT repair prompt: only the invalid fields (path, bad value, allowed values) + the reply format
    repair_flat(reply_text, sv)      the repair reply (a {path: value} map, or a design-shaped object) -> {path: value} for the pending paths

RULES (one per field, in this order)
  1. A valid field is kept exactly as the model wrote it (status `model`).
  2. An invalid field is replaced ALONE: by the NEAREST ALLOWED value when there is an obvious one (status `fixed`, mapping below), else by the rules
     planner's value for THAT field of THAT word (status `rules`; it goes on the `pending` list = what the repair call is asked for), else, for an optional
     field the rules do not set (axes, matte, lines, screen_offset ..), the field is dropped (status `fixed`, "dropped").
  3. A word or a section is never replaced because one of its fields is bad. Only a whole missing section (`type` / `compose` / `type.words` /
     one word's entry) comes from the rules, and only that section / word.
  4. Piece-level rules (>= 3 liked faces incl. one expressive, consecutive-line contrast, timing fit to the cut, window-cut eligibility) run AFTER the
     salvage, change as few fields as they can (qwen_piece.diversify_fonts, fit_scene, design_options.validate) and every change is logged;
     `account()` marks each field they changed `fixed` ("piece rule: ...").

NEAREST ALLOWED VALUE (the mapping; case, spaces, `-` and `_` are ignored first, then the synonym tables below, then difflib >= 0.8, else no match)
  enum fields   exact -> kept; same word in another spelling (`Wipe`, `gradient-fade`) -> `fixed`; SYN_* tables; difflib closest -> `fixed`; none -> rules
  io styles     SYN_IO: fade / dissolve / gradient_fade -> wipe (a soft mask sweep), shrink / grow / melt / morph -> blobs, glitch / strobe -> flicker,
                dust / sparks / scatter / disintegrate -> blobs (no dot-cloud style since 2026-11-01), draw / trace / outline -> lines
  numbers       clamped into the field's range (RANGES in qwen_piece; hold_s floor 1.2 s) -> `fixed`; "1.6s" / "1.6" strings read as numbers; not a number -> rules
  colours       [r,g,b] 0..255; floats 0..1 (any non-integer, all <= 1) x 255; glow_color 0..1, and 0..255 integers / 255 (the model wrote [10,10,10]: that
                used to clamp to white); clamped; unreadable -> rules
  font          liked-font lookup (key, family or file, ignoring case / extension / axes brackets) -> the exact file name (`fixed` if the spelling differed);
                not a liked font -> rules' font for that word
  axes          tag matched ignoring case, synonyms (weight -> wght, width -> wdth, slant -> slnt, italic -> ital, optical size -> opsz); an axis the font
                does not have -> dropped; values clamped into the font's range
  matte         `behind` / none / plain / off -> dropped (plain behind is the technique); synonyms; else difflib; else dropped
  transitions   a list with the wrong length: cut / padded with null (null = the engine picks that gap); an unknown entry -> SYN_TRANSITION / difflib, else
                null (= the rules' `auto` for that gap); a name used twice (except cut) -> the later one becomes null
  technique     a technique the generated footage cannot support -> the plan's own technique (logged by normalize_design)
  holds         [t, d] or {time|t, freeze_s|freeze|duration|d}; unreadable entries dropped one by one
  lines         a line list that does not spell the word -> the rules' lines for that word, else dropped
"""
from __future__ import annotations

import copy
import difflib
import json
import re

from tools.fusion import owner as OW
from tools.fusion import qwen_piece as QP
from tools.fusion import why_fields as WF

# ------------------------------------------------------------------------------------------------- synonym tables
SYN_IO = {"fade": "wipe", "fade_in": "wipe", "fade_out": "wipe", "dissolve": "wipe", "gradient_fade": "wipe", "gradient": "wipe", "soft": "wipe", "opacity": "wipe",
          "slide": "wipe", "reveal": "wipe", "sweep": "wipe", "mask": "wipe", "curtain": "wipe",
          "shrink": "blobs", "grow": "blobs", "expand": "blobs", "scale": "blobs", "zoom": "blobs", "melt": "blobs", "morph": "blobs", "ink": "blobs", "liquid": "blobs", "blob": "blobs",
          "glitch": "flicker", "strobe": "flicker", "blink": "flicker", "flash": "flicker", "pop": "flicker", "none": "flicker", "cut": "flicker",
          "dust": "blobs", "sparks": "blobs", "scatter": "blobs", "spray": "blobs", "disintegrate": "blobs", "dots": "blobs", "particle": "blobs", "particles": "blobs",
          "line": "lines", "strands": "lines", "draw": "lines", "trace": "lines", "outline": "lines", "stroke": "lines"}
SYN_SIDE = {"top": "up", "above": "up", "from_top": "up", "bottom": "down", "below": "down", "from_bottom": "down", "from_left": "left", "from_right": "right", "centre": "in", "center": "in",
            "middle": "in", "front": "out", "back": "in", "depth": "in", "forward": "out", "toward": "out"}
SYN_MODULE = {"strand": "lanes", "strands": "lanes", "lines": "lanes", "lane": "lanes", "pipes": "pipework", "pipe": "pipework", "tubes": "hollow_tubes",
              "tube": "hollow_tubes", "tower": "topo_tower", "topo": "topo_tower", "hinge": "hinge_unfold", "unfold": "hinge_unfold", "xray": "xray_unfold",
              "candy": "candy_orbit", "orbit": "candy_orbit", "tunnel": "depth_tunnel", "bars": "beat_bars", "echo": "bold_thin_echo", "fade": "gradient_fade",
              "clash": "scale_clash", "hollow": "solid_hollow", "stairs": "staircase", "spray": "tangent_spray", "fly": "fly_into", "flow": "lanes", "outflow": "lanes",       # `outflow` retired 2026-11-02: a reply that names it lands on a kept module
              "voxels": "voxel_orbit", "voxel": "voxel_orbit", "dust": "tangent_spray", "particles": "tangent_spray", "particle": "tangent_spray", "circuits": "circuit", "pcb": "circuit", "echoes": "depth_echo"}
SYN_CAMERA = {"orbit": "beat_orbit", "drift": "hold_drift", "static": "hold_drift", "still": "hold_drift", "push": "push_through", "pull_back": "pullback_reveal",
              "pullback": "pullback_reveal", "spin": "spin_read_spin", "tilt": "tilt_to_plan", "ease": "ease_reversal", "even_orbit": "orbit_even"}
SYN_TRANSITION = {"fade": "ripple", "dissolve": "ripple", "crossfade": "ripple", "wave": "ripple", "hard_cut": "cut", "hardcut": "cut", "none": "cut", "hard": "cut",
                  "contour": "contour_morph", "contour_to_word": "contour_morph", "morph": "contour_morph", "spiral": "spiral_around", "circuit": "circuit_board", "swirl": "twist", "braided": "braid", "auto": None}
SYN_TECHNIQUE = {"behind_the_subject": "behind", "sheer": "behind_sheer", "trace": "traced", "tracing": "traced", "behind_subject": "behind"}
SYN_MATTE = {"foreground": "front", "in_front": "front", "windows": "window", "glowing": "glow", "fills": "fill", "wiped": "wipe"}
SYN_AXIS = {"weight": "wght", "width": "wdth", "slant": "slnt", "slope": "slnt", "italic": "ital", "optical_size": "opsz", "opticalsize": "opsz", "size": "opsz",
            "soft": "SOFT", "wonk": "WONK", "roundness": "ROND", "round": "ROND"}
NO_MATTE = ("behind", "none", "null", "no", "plain", "off", "false", "n/a")
MIN_HOLD = QP.MIN_HOLD_IN
NEED_PENDING = ("font", "module", "hit", "hold_s")      # a field the model left out that the repair call is asked for (the rest has a rules/default value)
UNCOUNTED = ("text", "concept", "treatment_notes")      # locked text / prose: not design fields


def _norm(s) -> str:
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def _num(v):
    """float from a number or a string like '1.6s'; None otherwise (bools, NaN, inf are not numbers)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", "."))
        if not m:
            return None
        x = float(m.group(0))
    else:
        return None
    return x if x == x and abs(x) != float("inf") else None


# ------------------------------------------------------------------------------------------------- validators: value -> (new, how, why)
# how: "ok" (kept as written) | "fix" (nearest allowed value) | None (invalid, no obvious value)
def v_enum(allowed, syn=None):
    allowed = list(allowed)
    norm = {_norm(a): a for a in allowed}

    def check(v):
        if isinstance(v, str) and v in allowed:
            return v, "ok", ""
        if not isinstance(v, (str, int, float)) or isinstance(v, bool):
            return None, None, f"{v!r} is not one of {', '.join(allowed)}"
        n = _norm(v)
        if n in norm:
            return norm[n], "fix", f"{v!r} -> {norm[n]!r} (spelling)"
        if syn and n in syn and syn[n] in allowed:
            return syn[n], "fix", f"{v!r} -> {syn[n]!r} (nearest allowed)"
        if syn and n in syn and syn[n] is None:
            return None, None, f"{v!r} is not one of {', '.join(allowed)}"
        c = difflib.get_close_matches(n, list(norm), n=1, cutoff=0.8)
        if c:
            return norm[c[0]], "fix", f"{v!r} -> {norm[c[0]]!r} (closest allowed)"
        return None, None, f"{v!r} is not one of {', '.join(allowed)}"
    return check


def v_num(key, lo=None, hi=None, floor=None, nd=3):
    """number clamped into RANGES[key] (or lo..hi); `floor` raises the low end (hold_s 1.2)."""
    if lo is None:
        lo, hi = QP.RANGES[key]
    if floor is not None:
        lo = max(lo, floor)

    def check(v):
        x = _num(v)
        if x is None:
            return None, None, f"{v!r} is not a number ({lo:g}..{hi:g})"
        y = min(max(x, lo), hi)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(y - x) < 1e-9:
            return v, "ok", ""
        if abs(y - x) < 1e-9 and not isinstance(v, str):
            return round(y, nd), "ok", ""
        return round(y, nd), "fix", (f"{x:g} clamped to {y:g} ({lo:g}..{hi:g})" if abs(y - x) > 1e-9 else f"{v!r} read as the number {y:g}")
    return check


def v_rgb(scale):
    """[r,g,b]: scale 255 (palette) or 1 (glow). Floats 0..1 given for 255 are x255; integers 0..255 given for 1.0 are /255."""
    def check(v):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            return None, None, f"{v!r} needs [r, g, b]"
        a = [_num(x) for x in v]
        if None in a:
            return None, None, f"{v!r} needs three numbers"
        how, why = "ok", ""
        if scale > 1 and max(a) <= 1.0 and any(abs(x - round(x)) > 1e-9 for x in a):
            a, how, why = [x * 255.0 for x in a], "fix", "floats 0..1 read as 0..255"
        elif scale <= 1 and max(a) > 1.0:
            a, how, why = [x / 255.0 for x in a], "fix", "0..255 read as 0..1"
        out = [min(max(x, 0.0), scale) for x in a]
        if out != a and how == "ok":
            how, why = "fix", f"clamped to 0..{scale:g}"
        res = [int(round(x)) for x in out] if scale > 1 else [round(x, 3) for x in out]
        if how == "ok" and res != [x for x in v] and scale > 1 and any(not isinstance(x, int) for x in v):
            how, why = "fix", "rounded to integers"
        return res, how, why
    return check


def v_text(maxlen=40):
    def check(v):
        s = str(v).strip() if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""
        if not s:
            return None, None, "empty"
        return (s[:maxlen], "ok" if len(s) <= maxlen and s == v else "fix", "" if s == v else "trimmed")
    return check


def v_ltx_at(v):
    """handoff time as a fraction of the word's span: clamped into OW.LTX_AT_RANGE ('50%' / 50 read as 0.5)."""
    x, how = OW.clamp_ltx_at(v)
    if how is None:
        return None, None, f"{v!r} is not a number ({OW.LTX_AT_RANGE[0]:g}..{OW.LTX_AT_RANGE[1]:g})"
    return x, ("ok" if how == "ok" else "fix"), ("" if how == "ok" else f"{v!r} read as {x:g} ({OW.LTX_AT_RANGE[0]:g}..{OW.LTX_AT_RANGE[1]:g})")


def v_material(v):
    """material: free text of at most OW.MATERIAL_MAX_WORDS words (longer: cut; a trailing 'letters' is dropped)."""
    m, how = OW.clean_material(v)
    if how is None:
        return None, None, f"{v!r} is not text"
    return m, ("ok" if how == "ok" else "fix"), ("" if how == "ok" else f"{str(v)[:60]!r} -> {m!r} (at most {OW.MATERIAL_MAX_WORDS} words)")


def v_font(v):
    from mograph import fonts_liked as fl
    if not isinstance(v, str) or not v.strip():
        return None, None, f"{v!r}: not in the liked font list; use a file name from the FONTS list"
    e = fl.lookup(v)
    if e is None:
        return None, None, f"{v!r}: not in the liked font list; use a file name from the FONTS list"
    return e["file"], ("ok" if e["file"] == v else "fix"), ("" if e["file"] == v else f"{v!r} -> {e['file']!r} (spelling)")


# ------------------------------------------------------------------------------------------------- flatten / unflatten
def _w(i: int, k: str) -> str:
    return f"type.words[{i}].{k}"


def align_words(words_in, want: list[str]):
    """model word entries -> one entry (or None) per wanted word: the entry with the same text first, else the same position."""
    out = [None] * len(want)
    if not isinstance(words_in, list):
        return out
    used = set()
    key = lambda s: re.sub(r"\s+", " ", str(s or "").strip().upper())          # noqa: E731
    for i, t in enumerate(want):
        for j, w in enumerate(words_in):
            if j not in used and isinstance(w, dict) and key(w.get("text")) == key(t):
                out[i], _ = w, used.add(j)
                break
    for i in range(len(want)):
        if out[i] is None and i < len(words_in) and i not in used and isinstance(words_in[i], dict):
            out[i] = words_in[i]
            used.add(i)
    return out


def lower_keys(v, axes=False):
    """the same object with every dict key lower-cased (`Matte`, `Technique`), except the keys of an `axes` dict (axis tags are case-sensitive: SOFT, wght)."""
    if isinstance(v, dict):
        return {(k if axes or not isinstance(k, str) else k.strip().lower()): lower_keys(x, axes=isinstance(k, str) and k.strip().lower() == "axes") for k, x in v.items()}
    if isinstance(v, list):
        return [lower_keys(x) for x in v]
    return v


WORD_SCALARS = ("text", "font", "module", "lines", "hit", "hold_s", "form_s", "exit_s", "form_in", "form_out", "entry", "exit", "screen_offset", "matte", "window_source",
                "render", "ltx_at", "material")
COMPOSE_SCALARS = ("technique", "plate_gain", "plate_sat", "glow_lo", "glow_hi", "glow_color", "grain", "vignette", "type_opacity", "holds", "cuts")


def flatten(obj, want: list[str]) -> dict:
    """reply object -> {path: value} (None / non-dict -> {}). Section presence is recorded under `@type` / `@compose` / `@words` (True when it is an object /
    a list). Words are aligned to the wanted texts first (`align_words`)."""
    raw: dict = {}
    if not isinstance(obj, dict):
        return raw
    c0 = obj.get("compose") if isinstance(obj.get("compose"), dict) else (obj.get("type") or {}).get("compose") if isinstance(obj.get("type"), dict) else None
    if isinstance(c0, dict) and (isinstance(c0.get("default"), dict) or isinstance(c0.get("shots"), dict)):
        raw["compose.ops"] = {k: copy.deepcopy(c0[k]) for k in ("default", "shots") if k in c0}      # the model's op chains, case of the shot ids kept (compose_blocks validates them)
    if isinstance(obj.get("system"), dict):
        raw["system"] = copy.deepcopy(obj["system"])
    try:                                    # the model's reasons (par round r6): collected and scored, never judged as a design field; a missing / empty / odd `why` is no error and blocks nothing
        wh = WF.collect(obj)
        if wh:
            s = WF.score(obj)
            raw["@why"] = dict(whys=wh, decisions=s["decisions"], present=s["present"], missing=s["missing"], copied=s["copied"], repeated=s["repeated"], reasons=s["reasons"], flagged=s["flagged"])
    except Exception as e:      # noqa: BLE001
        raw["@why"] = dict(whys={}, error=f"{type(e).__name__}: {e}"[:160])
    obj = lower_keys(obj)
    if isinstance(obj.get("type"), dict) and isinstance(obj["type"].get("compose"), dict) and not isinstance(obj.get("compose"), dict):
        obj = dict(obj, compose=obj["type"]["compose"])                      # `compose` written inside an unclosed `type` object
    for k in ("concept", "treatment_notes"):
        if k in obj:
            raw[k] = obj[k]
    t, c = obj.get("type"), obj.get("compose")
    raw["@type"], raw["@compose"] = isinstance(t, dict), isinstance(c, dict)
    if isinstance(t, dict):
        for k in ("mood", "energy", "camera", "transitions"):
            if k in t:
                raw[f"type.{k}"] = t[k]
        if isinstance(t.get("line"), dict) and "px" in t["line"]:
            raw["type.line.px"] = t["line"]["px"]
        pal = t.get("palette")
        if isinstance(pal, dict):
            for k in ("bg", "fg", "accents"):
                if k in pal:
                    raw[f"type.palette.{k}"] = pal[k]
        cp = t.get("camera_params")
        if isinstance(cp, dict):
            for k, v in cp.items():
                raw[f"type.camera_params.{k}"] = v
        raw["@words"] = isinstance(t.get("words"), list)
        for i, w in enumerate(align_words(t.get("words"), want)):
            if w is None:
                continue
            raw[f"@word{i}"] = True
            for k in WORD_SCALARS:
                if k in w:
                    raw[_w(i, k)] = w[k]
            if isinstance(w.get("placement"), dict):
                for k, v in w["placement"].items():
                    raw[_w(i, f"placement.{k}")] = v
            if isinstance(w.get("axes"), dict):
                for k, v in w["axes"].items():
                    raw[_w(i, f"axes.{k}")] = v
    if isinstance(c, dict):
        for k in COMPOSE_SCALARS:
            if k in c:
                raw[f"compose.{k}"] = c[k]
    if isinstance(obj.get("system"), dict) and isinstance(t, dict):
        apply_system(raw, obj["system"], len(want))
    return raw


def _font_in(text) -> str | None:
    """the liked font file named in a free text ('GeistPixel[ELSH].ttf - hard pixel grid', 'Doto'), or None."""
    from mograph import fonts_liked as FL
    low = str(text or "").lower()
    for e in sorted(FL.LIKED, key=lambda e: -len(e["file"])):
        if e["file"].lower() in low:
            return e["file"]
    for e in sorted(FL.LIKED, key=lambda e: -len(e["key"])):
        if len(e["key"]) >= 4 and e["key"].lower() in low or (len(e["family"]) >= 4 and e["family"].lower() in low):
            return e["file"]
    return None


def apply_system(raw: dict, system: dict, n: int) -> list[str]:
    """the `system` block of a design reply (the ONE rule of the piece as a pattern) fills the word fields the words OMIT; a field the word wrote is never touched.
        roles    {role: font file (or a text that names one)}
        pattern  [role, ...]  cycled over the words (A, B, A, B ...)  -> type.words[i].font
        yaw      [deg, ...]   cycled over the words                    -> placement.yaw
        axis     {"font": file, "tag": "ELSH", "windows": [[lo, hi], ...]}  cycled over the words that use that font -> axes.<tag> = [lo, hi, "beat"]
    -> the paths filled (also recorded under raw["@system"])."""
    filled: list[str] = []
    roles = {str(k).strip().lower(): _font_in(v) for k, v in (system.get("roles") or {}).items()} if isinstance(system.get("roles"), dict) else {}
    pat = [str(x).strip().lower() for x in system.get("pattern") or [] if isinstance(x, (str, int))] if isinstance(system.get("pattern"), list) else []
    for i in range(n):
        if pat and roles.get(pat[i % len(pat)]) and _w(i, "font") not in raw:
            raw[_w(i, "font")] = roles[pat[i % len(pat)]]
            filled.append(_w(i, "font"))
    yaw = [x for x in system.get("yaw") or [] if isinstance(x, (int, float)) and not isinstance(x, bool)] if isinstance(system.get("yaw"), list) else []
    for i in range(n):
        if yaw and _w(i, "placement.yaw") not in raw:
            raw[_w(i, "placement.yaw")] = yaw[i % len(yaw)]
            filled.append(_w(i, "placement.yaw"))
    ax = system.get("axis") if isinstance(system.get("axis"), dict) else None
    if ax and ax.get("tag") and isinstance(ax.get("windows"), list):
        ff = _font_in(ax.get("font"))
        k = 0
        for i in range(n):
            f = raw.get(_w(i, "font"))
            if (ff and _font_in(f) != ff) or any(p.startswith(_w(i, "axes.")) for p in raw):
                continue
            w = ax["windows"][k % len(ax["windows"])] if ax["windows"] else None
            k += 1
            if isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(x, (int, float)) for x in w):
                raw[_w(i, f"axes.{ax['tag']}")] = [w[0], w[1], "beat"]
                filled.append(_w(i, f"axes.{ax['tag']}"))
    raw["@system"] = filled
    return filled


_PATH = re.compile(r"^(type|compose)(\.|\[)")


def pathmap(obj) -> dict | None:
    """a repair reply that is a {path: value} map ('type.words[1].form_in': 'wipe') -> the map; anything else None."""
    if isinstance(obj, dict) and obj and sum(1 for k in obj if isinstance(k, str) and _PATH.match(k)) >= max(1, len(obj) // 2):
        return {k: v for k, v in obj.items() if isinstance(k, str) and _PATH.match(k)}
    if isinstance(obj, dict) and isinstance(obj.get("fixes"), dict):
        return pathmap(obj["fixes"])
    return None


def covers(pending_path: str, p: str) -> bool:
    return p == pending_path or p.startswith(pending_path + ".") or p.startswith(pending_path + "[")


# ------------------------------------------------------------------------------------------------- the salvage
class Salvage:
    """result of salvage(): `clean` (a reply object normalize_design accepts: every field valid), `status` {path: (status, why)}, `pending` [dict(path, bad,
    why, allowed)], `raw` (the flat model values the clean reply was made from; stored in the piece so a repair call can be merged and the whole thing
    re-run), `from_repair` (paths whose value came from the repair call)."""

    def __init__(self, raw: dict, rules: dict, want: list[str], ctx: dict):
        self.raw, self.rules, self.want, self.ctx = raw, rules, want, ctx
        self.status: dict = {}
        self.pending: list = []
        self.from_repair: set = set()
        self.clean: dict = {}
        self.timing_from_rules = False

    # -- one field
    def put(self, path, status, why="", pending=None):
        self.status[path] = (status, why)
        if pending:
            self.pending.append(dict(path=path, bad=self.raw.get(path), why=why, allowed=pending))

    def take(self, path, check, *, required=False, allowed="", optional=False):
        """-> the value to use (or _MISSING when the field is left out). required: a missing field is asked for in the repair call."""
        rv = self.rules.get(path, _MISSING)
        if path in self.raw:
            val, how, why = check(self.raw[path])
            src = "repair" if path in self.from_repair else ""
            if how == "ok":
                self.put(path, "model", src)
                return val
            if how == "fix":
                self.put(path, "fixed", why + (" (repair reply)" if src else ""))
                return val
            if rv is not _MISSING:
                self.put(path, "rules", f"invalid: {why}", pending=allowed or why)
                return rv
            self.put(path, "fixed", f"invalid, dropped: {why}")
            return _MISSING
        if rv is not _MISSING and not optional:
            leaf = path.rsplit(".", 1)[-1]
            self.put(path, "rules", "missing in the reply", pending=(allowed or "a value") if leaf in NEED_PENDING else None)
            return rv
        return _MISSING


_MISSING = object()


def salvage(raw: dict, rules: dict, ctx: dict) -> Salvage:
    """raw = flatten(model reply) (+ repair values), rules = flatten(rules reply) (+ the rules' lines per word), ctx = dict(want=[word texts], techniques=[allowed],
    plan_technique, cut, vary={font file: {axis: (lo, hi)}} ). -> Salvage."""
    want = list(ctx["want"])
    n = len(want)
    sv = Salvage(raw, rules, want, ctx)
    sv.from_repair = set(ctx.get("from_repair") or [])
    T: dict = {}
    C: dict = {}
    has_t, has_c = bool(raw.get("@type")), bool(raw.get("@compose"))
    no_section = lambda sec: f"missing object `{sec}`"                          # noqa: E731
    if not has_t:
        sv.pending.append(dict(path="type", bad=None, why=no_section("type"), allowed="the whole `type` object"))
    if not has_c:
        sv.pending.append(dict(path="compose", bad=None, why=no_section("compose"), allowed="the whole `compose` object"))
    T_raw = lambda path: path in raw and has_t                                   # noqa: E731

    def take(path, check, **kw):
        if (path.startswith("type") and not has_t) or (path.startswith("compose") and not has_c):
            rv = rules.get(path, _MISSING)
            if rv is not _MISSING:
                sv.put(path, "rules", "section missing")
            return rv
        return sv.take(path, check, **kw)

    cut = float(ctx.get("cut") or 0.0)
    # ---- type
    mood = take("type.mood", v_text(40), allowed="one or two words")
    energy = take("type.energy", v_num("energy"), allowed="0..1")
    pal: dict = {}
    for k in ("bg", "fg"):
        v = take(f"type.palette.{k}", v_rgb(255), allowed="[r,g,b] integers 0..255")
        if v is not _MISSING:
            pal[k] = v
    accs = None
    if f"type.palette.accents" in raw and has_t:
        a_in = raw["type.palette.accents"]
        good = []
        for j, a in enumerate(a_in[:4] if isinstance(a_in, list) else []):
            v, how, why = v_rgb(255)(a)
            if how:
                good.append(v)
                if how == "fix":
                    sv.put(f"type.palette.accents[{j}]", "fixed", why)
        if good:
            accs = good
            sv.put("type.palette.accents", "model" if good == a_in[:4] else "fixed", "" if good == a_in[:4] else "entries repaired / dropped")
        elif "type.palette.accents" in rules:
            accs = rules["type.palette.accents"]
            sv.put("type.palette.accents", "rules", "no usable colour", pending="[[r,g,b], ...] integers 0..255")
    elif "type.palette.accents" in rules:
        accs = rules["type.palette.accents"]
        sv.put("type.palette.accents", "rules", "section missing" if not has_t else "missing in the reply")
    if accs is not None:
        pal["accents"] = accs
        if len(accs) < 3 and pal.get("fg") is not None:
            QP.pad_accents(pal)                 # the 3D word modules read accents[2]: a palette with two accents crashed the type scene (par round r3 P11)
    cam = take("type.camera", v_enum(QP.CAMERAS, SYN_CAMERA), allowed=", ".join(QP.CAMERAS))
    cps = {}
    for k in ("read_dps", "intro_deg", "fast_dps", "elev", "bob"):
        v = take(f"type.camera_params.{k}", v_num(k), optional=True)
        if v is not _MISSING:
            cps[k] = v
    px = take("type.line.px", v_num("line.px"), allowed="2..8")
    # transitions
    tr = _MISSING
    if has_t and "type.transitions" in raw:
        tr_in = raw["type.transitions"]
        if tr_in == "auto" or tr_in is None:
            tr = "auto"
            sv.put("type.transitions", "model" if tr_in == "auto" else "fixed", "" if tr_in == "auto" else "null -> auto")
        elif isinstance(tr_in, list):
            ent = list(tr_in[:max(n - 1, 0)])
            ent += [None] * (max(n - 1, 0) - len(ent))
            if len(tr_in) != max(n - 1, 0):
                sv.put("type.transitions", "fixed", f"{len(tr_in)} entries for {n - 1} gaps: " + ("cut" if len(tr_in) > n - 1 else "padded with null (engine's choice)"))
            seen, outl = set(), []
            chk = v_enum(QP.TRANSITIONS, SYN_TRANSITION)
            for j, e in enumerate(ent):
                p = f"type.transitions[{j}]"
                if e is None:
                    outl.append(None)
                    sv.put(p, "rules" if j >= len(tr_in) else "fixed", "null: the engine picks this gap")
                    continue
                v, how, why = chk(e)
                if how is None:
                    outl.append(None)
                    sv.put(p, "rules", f"invalid: {why}", pending=", ".join(QP.TRANSITIONS))
                elif v not in ("cut", "contour_morph") and v in seen:
                    outl.append(None)
                    sv.put(p, "fixed", f"{v!r} already used by an earlier gap; null: the engine picks this gap")
                else:
                    outl.append(v)
                    seen.add(v)
                    sv.put(p, "model" if how == "ok" else "fixed", "" if how == "ok" else why)
            tr = outl
        else:
            tr = "auto"
            sv.put("type.transitions", "rules", "not 'auto' or a list", pending="\"auto\" or a list of one transition per gap")
    elif "type.transitions" in rules:
        tr = rules["type.transitions"]
        sv.put("type.transitions", "rules", "section missing" if not has_t else "missing in the reply")
    # words
    words_out = []
    wsec_ok = bool(raw.get("@words")) and has_t
    if has_t and not wsec_ok:
        sv.pending.append(dict(path="type.words", bad=None, why=f"type.words needs exactly {n} entries in order ({', '.join(want)})", allowed="a list of word objects"))
    for i in range(n):
        has_w = bool(raw.get(f"@word{i}")) and has_t
        if has_t and wsec_ok and not has_w:
            sv.pending.append(dict(path=f"type.words[{i}]", bad=None, why=f"type.words[{i}] ({want[i]!r}) is missing", allowed="a word object"))
        o: dict = {"text": want[i]}
        if has_w and str(raw.get(_w(i, "text"), "")).strip().upper() != want[i].strip().upper():
            sv.put(_w(i, "text"), "fixed", f"text {raw.get(_w(i, 'text'))!r} -> {want[i]!r} (locked)")
        for k, chk, kw in (("font", v_font, dict(allowed="a font file from the FONTS list, exact")),
                           ("module", v_enum(QP.WORD_MODULES, SYN_MODULE), dict(allowed=", ".join(QP.WORD_MODULES))),
                           ("form_in", v_enum(QP.IO_STYLES, SYN_IO), dict(allowed=", ".join(QP.IO_STYLES))),
                           ("form_out", v_enum(QP.IO_STYLES, SYN_IO), dict(allowed=", ".join(QP.IO_STYLES))),
                           ("entry", v_enum(QP.OP.SIDES, SYN_SIDE), dict(allowed=", ".join(QP.OP.SIDES))),
                           ("exit", v_enum(QP.OP.SIDES, SYN_SIDE), dict(allowed=", ".join(QP.OP.SIDES))),
                           ("hit", v_num("hit", 0.0, max(cut, 0.5)), dict(allowed="seconds: a downbeat or beat of the music analysis")),
                           ("hold_s", v_num("hold_s", floor=MIN_HOLD), dict(allowed=f"seconds, {MIN_HOLD:g}..6")),
                           ("form_s", v_num("form_s"), dict(allowed="0.45..2.0")),
                           ("exit_s", v_num("exit_s"), dict(allowed="0.5..1.5"))):
            pth = _w(i, k)
            if has_t and not has_w and wsec_ok:
                rv = rules.get(pth, _MISSING)
                if rv is not _MISSING:
                    sv.put(pth, "rules", "word entry missing")
                v = rv
            else:
                v = take(pth, chk, **kw)
            if v is not _MISSING:
                o[k] = v
        # lines
        pth = _w(i, "lines")
        if pth in raw and has_t and raw[pth] in ([], None, ""):
            pass                                                        # an empty line list = no line breaks asked: not a field
        elif pth in raw and has_t:
            ls = raw[pth]
            ok = isinstance(ls, list) and ls and "".join(str(x) for x in ls).replace(" ", "").upper() == want[i].replace(" ", "").upper()
            if ok and not QP.DO.word_aligned(want[i], ls):
                sv.put(pth, "fixed", f"{ls!r} cuts a word: dropped (lines break only at spaces; a word stays on one line)")      # MIDN/IGHT, WA/KE/UP in the round-3 QC
            elif ok:
                o["lines"] = [str(x) for x in ls]
                sv.put(pth, "model")
            elif _w(i, "lines") in rules:
                o["lines"] = rules[pth]
                sv.put(pth, "rules", f"{ls!r}: joined without spaces they must spell {want[i]!r}", pending=f"line breaks that spell {want[i]!r}")
            else:
                sv.put(pth, "fixed", f"{ls!r} does not spell {want[i]!r}: dropped")
        elif pth in rules and not has_w:
            o["lines"] = rules[pth]
            sv.put(pth, "rules", "word entry missing")
        # placement / screen_offset / axes / matte / window_source
        pl = {}
        for k in ("yaw", "wf", "hf"):
            pth = _w(i, f"placement.{k}")
            v = _MISSING
            if has_w or not (has_t and wsec_ok):
                v = take(pth, v_num(k), optional=False)
            elif pth in rules:
                v = rules[pth]
                sv.put(pth, "rules", "word entry missing")
            if v is not _MISSING:
                pl[k] = v
        for k in ("pitch", "roll", "dx", "dy"):                 # optional placement keys (the lead designer's pieces use dy / pitch); counted only when the model wrote them
            pth = _w(i, f"placement.{k}")
            if has_w and pth in raw:
                v = take(pth, v_num(k), optional=True)
                if v is not _MISSING:
                    pl[k] = v
        if pl:
            o["placement"] = pl
        pth = _w(i, "screen_offset")
        if pth in raw and has_t:
            so = raw[pth]
            vs = [v_num("so")(x) for x in so] if isinstance(so, list) and len(so) == 2 else None
            if vs and all(h for _, h, _ in vs):
                o["screen_offset"] = [v for v, _, _ in vs]
                sv.put(pth, "model" if all(h == "ok" for _, h, _ in vs) else "fixed", "" if all(h == "ok" for _, h, _ in vs) else "clamped")
            else:
                sv.put(pth, "fixed", f"{so!r}: needs [x, y] numbers; dropped")
        font = o.get("font")
        vary = (ctx.get("vary") or {}).get(font) or {}
        ax = {}
        for tag_path in [p for p in raw if p.startswith(f"type.words[{i}].axes.")] if has_t else []:
            tag = tag_path.rsplit(".", 1)[-1]
            tg = tag if tag in vary else next((a for a in vary if a.lower() == tag.lower()), None) or SYN_AXIS.get(_norm(tag))
            if tg not in vary:
                sv.put(tag_path, "fixed", f"axis {tag!r} is not an axis of {font}: dropped")
                continue
            val = raw[tag_path]
            lo, hi = vary[tg]
            try:
                if isinstance(val, (list, tuple)):
                    a0, a1 = float(val[0]), float(val[1])
                    nv = [min(max(a0, lo), hi), min(max(a1, lo), hi), "beat"]
                    same = list(val) == nv
                else:
                    nv = min(max(float(val), lo), hi)
                    same = isinstance(val, (int, float)) and abs(nv - float(val)) < 1e-9
                ax[tg] = nv
                sv.put(tag_path, "model" if same and tg == tag else "fixed", "" if same and tg == tag else f"axis {tag!r} -> {tg!r} / clamped to {lo:g}..{hi:g}")
            except (TypeError, ValueError, IndexError):
                sv.put(tag_path, "fixed", f"axis {tag!r}: unreadable value {val!r}; dropped")
        if ax:
            o["axes"] = ax
        pth = _w(i, "matte")
        if pth in raw and has_t:
            mt = raw[pth]
            if mt is None or (isinstance(mt, str) and _norm(mt) in NO_MATTE):
                sv.put(pth, "fixed", f"{mt!r}: plain behind is the technique, not a treatment; dropped")
            else:
                v, how, why = v_enum(QP.MATTES, SYN_MATTE)(mt)
                if how:
                    o["matte"] = v
                    sv.put(pth, "model" if how == "ok" else "fixed", why)
                else:
                    sv.put(pth, "fixed", f"{why}; dropped")
        pth = _w(i, "window_source")
        if pth in raw and has_t and raw[pth] not in (None, "", "self"):
            o["window_source"] = str(raw[pth])
            sv.put(pth, "model")

        def own(k, chk, **kw):
            """owner field k of word i: the model's value (mended alone), else the rules' value (default handoff / 0.5 / the rules' material)."""
            pth_ = _w(i, k)
            if has_t and not has_w and wsec_ok:
                rv_ = rules.get(pth_, _MISSING)
                if rv_ is not _MISSING:
                    sv.put(pth_, "rules", "word entry missing")
                return rv_
            if has_t and pth_ not in raw:                        # not mentioned in the reply: the default, not counted against the model (older replies know no owner)
                rv_ = rules.get(pth_, _MISSING)
                if rv_ is not _MISSING:
                    sv.put(pth_, "default", "not in the reply: the default")
                return rv_
            return take(pth_, chk, **kw)
        r = own("render", v_enum(OW.RENDERS, OW.SYN_RENDER), allowed=", ".join(OW.RENDERS))
        o["render"] = OW.DEFAULT_RENDER if r is _MISSING else r
        if o["render"] == "handoff":
            v = own("ltx_at", v_ltx_at, allowed=f"{OW.LTX_AT_RANGE[0]:g}..{OW.LTX_AT_RANGE[1]:g}")
            o["ltx_at"] = OW.DEFAULT_LTX_AT if v is _MISSING else v
        if o["render"] != "flat":
            v = own("material", v_material, allowed=f"at most {OW.MATERIAL_MAX_WORDS} words: what the letters are made of in the footage")
            if v is _MISSING:
                v = OW.default_material(mood if mood is not _MISSING else None, pal, ctx.get("scene"))
                sv.put(_w(i, "material"), "default", "not in the reply: the default from the mood / palette / scene")
            o["material"] = v
        words_out.append(o)
    # ---- compose
    techs = list(ctx.get("techniques") or QP.TECHNIQUES)
    plan_t = ctx.get("plan_technique") or techs[0]
    tech = _MISSING
    if has_c and "compose.technique" in raw:
        v, how, why = v_enum(list(dict.fromkeys(list(QP.TECHNIQUES) + techs)), SYN_TECHNIQUE)(raw["compose.technique"])       # the plan's own technique (built_in of a single shot) is no QP.TECHNIQUES entry but is allowed
        if how is None:
            tech = plan_t
            sv.put("compose.technique", "rules", f"invalid: {why}", pending=", ".join(techs))
        elif v not in techs:
            tech = plan_t
            sv.put("compose.technique", "fixed", f"{v!r} needs a matte / depth the footage does not have; {plan_t!r} used instead")
        else:
            tech = v
            sv.put("compose.technique", "model" if how == "ok" else "fixed", why)
    else:
        tech = plan_t
        sv.put("compose.technique", "rules", "missing in the reply" if has_c else "section missing")
    for k in ("plate_gain", "plate_sat", "glow_lo", "glow_hi", "grain", "vignette", "type_opacity"):
        if has_c and f"compose.{k}" in raw:
            v = sv.take(f"compose.{k}", v_num(k, nd=4))
            if v is not _MISSING:
                C[k] = v
    if has_c and "compose.glow_color" in raw:
        v = sv.take("compose.glow_color", v_rgb(1))
        if v is not _MISSING:
            C["glow_color"] = v
    if has_c and "compose.holds" in raw:
        hs, h_in = [], raw["compose.holds"]
        if isinstance(h_in, list):
            for j, h in enumerate(h_in):
                t_, d_ = None, None
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    t_, d_ = _num(h[0]), _num(h[1])
                elif isinstance(h, dict):
                    t_ = next((_num(h[k]) for k in ("time", "t", "at", "start") if k in h and _num(h[k]) is not None), None)
                    d_ = next((_num(h[k]) for k in ("freeze_s", "freeze", "duration", "dur", "d", "len", "length", "hold_s") if k in h and _num(h[k]) is not None), None)
                if t_ is None or d_ is None or j >= 6:
                    sv.put(f"compose.holds[{j}]", "fixed", f"{h!r}: " + ("more than 6 holds" if j >= 6 else "needs [time s, freeze s]") + "; dropped")
                else:
                    hs.append([t_, d_])
                    if not (isinstance(h, (list, tuple))):
                        sv.put(f"compose.holds[{j}]", "fixed", f"{h!r} read as [{t_:g}, {d_:g}]")
                    else:
                        sv.put(f"compose.holds[{j}]", "model")
            if hs:
                C["holds"] = hs
        else:
            sv.put("compose.holds", "fixed", f"{h_in!r}: not a list; dropped")
    if has_c and "compose.cuts" in raw:
        C["cuts"] = raw["compose.cuts"]
        sv.put("compose.cuts", "model")
    C["technique"] = tech
    # ---- the clean reply
    t_out = {"words": words_out}
    for k, v in (("mood", mood), ("energy", energy), ("camera", cam), ("transitions", tr)):
        if v is not _MISSING:
            t_out[k] = v
    t_out["palette"] = pal
    if cps:
        t_out["camera_params"] = cps
    if px is not _MISSING:
        t_out["line"] = {"px": px}
    sv.pending = [p for p in sv.pending if not any(q["path"] != p["path"] and covers(q["path"], p["path"]) for q in sv.pending)]
    sv.clean = {"concept": str(raw.get("concept") or rules.get("concept") or "the rules' own plan")[:300],
                "treatment_notes": str(raw.get("treatment_notes") or rules.get("treatment_notes") or "")[:300], "type": t_out, "compose": C}
    return sv


# ------------------------------------------------------------------------------------------------- accounting after the piece rules
_FIX_PATH = re.compile(r"^(type\.words\[(\d+)\](?:\.[\w\.]+)?|type\.[\w\.]+|compose\.[\w\.]+|words\[(\d+)\]\.(\w+)|cuts\[(\d+)\])")


def _mark(sv: Salvage, path: str, why: str) -> None:
    """a piece rule changed `path` (or everything under it): every tracked path under it that was the model's becomes `fixed`."""
    for p, (st, _) in list(sv.status.items()):
        if st == "model" and covers(path, p):
            sv.status[p] = ("fixed", "piece rule: " + why[:160])


def account(sv: Salvage, fixes: list[str], pre: list[dict] | None, post: list[dict] | None, final_design: dict | None) -> None:
    """mark the fields the piece-level rules (compile_timing, fit_scene, window validator, font rules) changed after the salvage.
    pre / post = the words' timing before / after the fit (dicts with t0, t1, form_s, exit_s, hit_on); final_design = the design the pieces uses."""
    for f in fixes:
        m = re.match(r"^type\.words\[(\d+)\]\.(\w+)(?:\.(\w+))?", f)
        if m:
            i, k, k2 = int(m.group(1)), m.group(2), m.group(3)
            _mark(sv, _w(i, f"{k}.{k2}" if k in ("axes", "placement") and k2 else k), f)
            continue
        if f.startswith("compose.cuts"):
            _mark(sv, "compose.cuts", f)
            continue
        m = re.match(r"^type\.words\[(\d+)\]:", f)
        if m and "not reachable" in f:
            _mark(sv, _w(int(m.group(1)), "hit"), f)
            continue
        m = re.match(r"^words\[(\d+)\]\.(\w+)", f)
        if m:
            _mark(sv, _w(int(m.group(1)), m.group(2)), f)
            continue
        m = re.match(r"^cuts\[(\d+)\]|^cuts (\w+)", f)
        if m:
            _mark(sv, "compose.cuts", f)
            continue
        if f.startswith("all holds shortened"):
            for i in range(len(sv.want)):
                _mark(sv, _w(i, "hold_s"), f)
    if pre:
        for i, (a, w) in enumerate(zip(pre, sv.clean["type"]["words"])):
            if "form_s" in w and abs(float(a["form_s"]) - float(w["form_s"])) > 0.01:
                _mark(sv, _w(i, "form_s"), "the timing solver moved the form time to land the hit")
    if pre and post:
        for i, (a, b) in enumerate(zip(pre, post)):
            if abs(float(a["t0"]) - float(b["t0"])) > 0.02 or a.get("hit_on") != b.get("hit_on"):
                _mark(sv, _w(i, "hit"), "timing fit moved the hit")
            if abs(float(a["form_s"]) - float(b["form_s"])) > 0.01:
                _mark(sv, _w(i, "form_s"), "timing fit")
            if abs(float(a.get("exit_s", 0)) - float(b.get("exit_s", 0))) > 0.01:
                _mark(sv, _w(i, "exit_s"), "timing fit")
            ha = float(a["t1"]) - float(a["t0"]) - float(a["form_s"]) - float(a.get("exit_s", 0))
            hb = float(b["t1"]) - float(b["t0"]) - float(b["form_s"]) - float(b.get("exit_s", 0))
            if abs(ha - hb) > 0.02:
                _mark(sv, _w(i, "hold_s"), "timing fit shortened the hold")
    if final_design:
        cl, fw = sv.clean["type"]["words"], final_design["type"]["words"]
        for i, (a, b) in enumerate(zip(cl, fw)):
            for k in ("font", "module", "form_in", "form_out", "lines", "screen_offset", "matte", "window_source", "render", "ltx_at", "material"):
                if _eff(k, a.get(k)) != _eff(k, b.get(k)):
                    _mark(sv, _w(i, k), f"{k} {a.get(k)!r} -> {b.get(k)!r}")
            for k in ("yaw", "wf", "hf", "pitch", "roll", "dx", "dy"):
                if (a.get("placement") or {}).get(k) != (b.get("placement") or {}).get(k):
                    _mark(sv, _w(i, f"placement.{k}"), f"placement.{k} changed")
            for tag in {*(a.get("axes") or {}), *(b.get("axes") or {})}:
                if (a.get("axes") or {}).get(tag) != (b.get("axes") or {}).get(tag):
                    _mark(sv, _w(i, f"axes.{tag}"), f"axis {tag} changed")
        ca, cb = sv.clean["compose"].get("cuts"), final_design["compose"].get("cuts")
        if ca != cb and ca is not None:
            _mark(sv, "compose.cuts", "cuts after the window validator")


def _eff(k, v):
    """the effective value of a word key: an all-zero screen_offset is no offset (normalize_design drops it; that is not a change of the model's design)."""
    if k == "screen_offset" and isinstance(v, (list, tuple)) and all(abs(float(x)) < 1e-9 for x in v):
        return None
    return v


def counted(sv: Salvage) -> dict:
    return {p: s for p, s in sv.status.items() if p.rsplit(".", 1)[-1] not in UNCOUNTED and not p.endswith(".text")}


def report(sv: Salvage, src: dict | None = None) -> dict:
    """dict(fields_from_model [paths], fields_fixed [{path, why}], fields_from_rules [{path, why}], pct_model, pct_model_or_fixed, n)."""
    st = counted(sv)
    model = sorted(p for p, (s, _) in st.items() if s == "model")
    fixed = [dict(path=p, why=w) for p, (s, w) in sorted(st.items()) if s == "fixed"]
    rules = [dict(path=p, why=w) for p, (s, w) in sorted(st.items()) if s == "rules"]
    n = max(len(model) + len(fixed) + len(rules), 1)
    return dict(fields_from_model=model, fields_fixed=fixed, fields_from_rules=rules, n=n, pct_model=round(100.0 * len(model) / n, 1),
                pct_model_or_fixed=round(100.0 * (len(model) + len(fixed)) / n, 1), pending=[dict(path=p["path"], why=p["why"]) for p in sv.pending],
                sections={"type": "model" if sv.raw.get("@type") else "rules", "compose": "model" if sv.raw.get("@compose") else "rules"},
                why=sv.raw.get("@why") or dict(whys={}, decisions=0, present=0, missing=0, copied=0, repeated=0, reasons=None, flagged=[]))


# ------------------------------------------------------------------------------------------------- repair: the short prompt and the merge
def repair_prompt(sv: Salvage, reply: str, fonts_text: str = "", words: list[str] | None = None) -> str:
    """the SHORT prompt of the repair call: only the fields the code could not mend (path, what was written, what is allowed) + the reply format."""
    lines = []
    for p in sv.pending[:24]:
        bad = "" if p["bad"] is None else f" (you wrote {json.dumps(p['bad'], ensure_ascii=False)[:80]})"
        lines.append(f"- {p['path']}{bad}: {p['why']}. Allowed: {p['allowed']}")
    ex_path = sv.pending[0]["path"] if sv.pending else "type.words[0].form_in"
    out = ["You designed the type of a short music video. A few fields of your JSON reply cannot be used. Correct ONLY these fields.",
           f"Words, in order: {' / '.join(words or sv.want)}.",
           "", "=== FIELDS TO CORRECT ===", *lines, "",
           "Reply with ONE JSON object that maps each field path above to its corrected value, and nothing else (no fence, no commentary). Example:",
           json.dumps({ex_path: "<corrected value>"})]
    if any(re.search(r"\.font$", p["path"]) for p in sv.pending) and fonts_text:
        out += ["", "=== FONTS (write the file name exactly) ===", fonts_text]
    return "\n".join(out)


def repair_flat(obj, sv: Salvage) -> dict:
    """the repair reply object -> {path: value} for the pending paths only (a path map, or a design-shaped object flattened like the first reply).
    Presence flags (`@type`, `@compose`, `@words`, `@word<i>`) come along when the reply supplies a section / word that was missing."""
    if obj is None:
        return {}
    pm = pathmap(obj)
    flat = pm if pm is not None else flatten(obj, sv.want)
    out = {}
    for pth, v in flat.items():
        if pth.startswith("@"):
            fp = {"@type": "type", "@compose": "compose", "@words": "type.words"}.get(pth) or (f"type.words[{pth[5:]}]" if pth.startswith("@word") else None)
            if v and fp and any(covers(pp["path"], fp) for pp in sv.pending):
                out[pth] = v
        elif any(covers(pp["path"], pth) for pp in sv.pending):
            out[pth] = v
    if pm is not None:                                          # a path map names sections / words by their paths
        for pth in pm:
            m = re.match(r"^type\.words\[(\d+)\]", pth)
            if m and any(covers(pp["path"], f"type.words[{m.group(1)}]") for pp in sv.pending):
                out[f"@word{m.group(1)}"] = True
                out["@words"] = True
            if pth.startswith("type") and any(covers(pp["path"], "type") for pp in sv.pending):
                out["@type"] = True
            if pth.startswith("compose") and any(covers(pp["path"], "compose") for pp in sv.pending):
                out["@compose"] = True
    return out
