"""owner: ONE owner per word, so a word is drawn once (pure functions, no ComfyUI, no GPU). NO CLAUDE.

The bug it fixes (3060 batch v001, every piece): every shot is a union-control LTX shot guided by the TYPE control pass, so LTX drew its own (garbled, glassy,
mis-registered) copy of every word into the footage, and compose then drew the flat type passes over it as well: two copies of each word, out of register.

Every word of a piece now has an owner field the design fills (`qwen_piece` schema, `design_salvage`, `16_app/DESIGN_GUIDE_open_model.md` section 12):

    render    "flat"     compose draws the word, LTX never sees it                         (control pass: never; flat alpha: 1 for the span)
              "ltx"      LTX draws the word, compose never draws it                        (control pass: whole span; flat alpha: 0, no glow / echo copies)
              "handoff"  flat type forms in, LTX's letters take over at `ltx_at` and hold  (control pass: ramps in at ltx_at; flat alpha: 1 until ltx_at, then 0 over FLAT_FADE_S)
    ltx_at    0.2..0.8   fraction of the word's span [start, end] where LTX takes over (handoff only)
    material  <= 8 words describing the physical letters for LTX ("thick carved ice, extruded, frosted edges"); needed when render != flat

WHO CHOOSES (round 2, 2026-10-28): the CONCEPT step (tools/fusion/concept.py, before the footage exists) picks render / ltx_at / material per lockup (`norm_owner`,
`remap_owners`), the planner writes them on the words and uses the rules' values only where the concept gave none. A shot keeps at most MAX_LTX_PER_SHOT (1) LTX-owned
word, MAX_LTX_LONG_SHOT (2) when it lasts >= LTX_LONG_SHOT_S (8 s): `cap_per_shot` keeps the word with the most letters and demotes the rest to flat. The prompt
clause names only the kept words. If LTX still formed no letters, compose falls back to flat (tools/fusion/ltx_check.py).

OLD pieces (no word has `render`) and DRAFT pieces (no LTX letters: the draft tier draws every word flat; also the still preview) are all "flat": the control pass, the guide, the prompts and compose stay exactly as they were. Only a piece whose words carry
owner fields uses the owners; a word of such a piece without `render` is "flat" (DEFAULT_RENDER since 2026-10-29; it was "handoff").

The same windows drive the three seams, so they can never disagree:
    control_gain(wins, t)   -> 0..1   the share of the TYPE CONTROL pass LTX gets at time t        (nodes_fusion/footage.py MographShotSlot)
    flat_gain(wins, t)      -> 0..1   the share of the FLAT type compose draws at time t           (tools/fusion/recipe_eval.py build_recipe / compose_frames)
    shot_owned / clause     the shot guide (0 when no word of the shot is LTX's, else OWNED_GUIDE) and the prompt clause (tools/fusion/plan_rules.py, MographShotSlot)
A word's region is [start_k, start_{k+1}): the word AND the transition after it belong to it; u = (t - start) / (end - start) is clamped to 1 in the transition.
"""
from __future__ import annotations

import difflib
import re

import numpy as np

RENDERS = ("flat", "ltx", "handoff")
DEFAULT_RENDER = "flat"                # the default owner is the FLAT graphic (2026-10-29: clear text first). An LTX-rendered word only where the concept / design asks for it AND it passes the rules (<= MAX_LTX_LETTERS letters, one per shot, the reading + layout checks)
LEGACY_RENDER = "flat"
LTX_AT_RANGE = (0.2, 0.8)
DEFAULT_LTX_AT = 0.5
MATERIAL_MAX_WORDS = 8
MIN_FLAT_S = 0.5                       # a handoff word stays flat at least this long after it lands when the tempo is unknown (else 1 beat, handoff_t)
FLAT_FADE_S = 0.25                     # flat -> LTX crossfade length at a handoff
CONTROL_RAMP_S = 0.25                  # the control pass ramps in over this long from ltx_at (the same window as the flat fade: a clean crossfade)
OWNED_GUIDE = 1.0                      # union-control guide of a shot with LTX-owned letters (the planner's recipe guide is 0.4). 1.0 is a constant: the text-pass probe (16_app/LAYERS.md section 8, 21_batch-3060/v006) found 0.65 gives weak / thin / garbled letters at V2V start 0.975; batch 3060 v007 r02 (guides 0.65): no letters formed for 3 words
MAX_CLAUSES = 2                        # a prompt never names more than this many words (cap_per_shot keeps a shot at or below it)
MAX_LTX_PER_SHOT = 1                   # LTX letters are expensive and fragile (batch 3060 v005: 3 words + a material in one prompt formed NO letters): ONE LTX-owned word per shot ...
LTX_LONG_SHOT_S = 8.0                  # ... two when the shot is at least this long (a single 16 s shot of 5 words: the second word comes ~6 s after the first)
MAX_LTX_LONG_SHOT = 2
MAX_LTX_LETTERS = 6                    # an LTX-owned word has at most this many LETTERS (letters only, spaces / punctuation not counted; one short key word preferred): batch 3060 v007 r01 spelled "STORM BREAKER" as "BRAAKER", a 13-letter "STILL STANDING" formed no letters
CLAUSE_TAIL = "extruded 3D lettering"  # closes the prompt clause: tells LTX the letters are solid lettering, not a texture
OFF_GUIDE = 0.0                        # a shot whose words are all flat: plain i2v, no IC guide
SYN_RENDER = {"static": "flat", "2d": "flat", "overlay": "flat", "compose": "flat", "type": "flat", "code": "flat",
              "video": "ltx", "3d": "ltx", "material": "ltx", "generated": "ltx", "model": "ltx", "full": "ltx",
              "hand_off": "handoff", "hand-off": "handoff", "transition": "handoff", "crossfade": "handoff", "mix": "handoff", "both": "handoff",
              "morph": "handoff", "swap": "handoff"}


# ---------------------------------------------------------------------------------------------------------------- what the graphic looks like (2026-10-29)
# The video model never invents its own text: it is given a depth / canny control of THE GRAPHIC and told which word it spells. Two things must match the graphic for that to work:
#   - the prompt names the graphic's look (a dotted font is "dotted dot-matrix capitals", not "glowing wire strands"): FONT_LOOK / ROLE_LOOK, `font_look`
#   - the control pass suits the glyph weight: thin / dotted / outline faces carry no depth worth guiding, so they get the canny LINE pass; solid heavy faces the DEPTH pass; in between the
#     UNION (depth + line, drawn together). `stroke_class` measures the face (PIL: the widest stroke as a fraction of the cap-line height, dotted / fragmented faces count as thin).
FONT_LOOK = {
    "Codystar": "dotted dot-matrix", "BungeeOutline": "outlined hollow", "BungeeInline": "inline-striped block", "BungeeHairline": "hairline thin", "Monofett": "chunky cracked",
    "RubikBrokenFax": "broken fax-scratched", "RubikGlitchPop": "glitch-sliced", "RubikLines": "striped-line", "LondrinaSketch": "hand-sketched outline", "Rock3D": "3D rock-edged outline",
    "MoiraiOne": "thin ornamental serif", "ZenTokyoZoo": "ornamental stencil", "Honk": "wobbly ornamental", "Fascinate": "art-deco block", "Barrio": "rough hand-cut",
    "AlienBlock": "bold alien block", "Doto": "dot-matrix", "GeistPixel": "pixel-grid", "Bitcount": "pixel-grid", "BitcountSingle": "pixel-grid", "Bytesized": "pixel-grid",
    "PaletteMosaic": "mosaic pixel", "Silkscreen": "pixel-grid", "FolditThin": "thin folded-line", "NixieOne": "thin nixie-tube", "Interlope": "thin elegant", "Tourney": "heavy tournament-stencil",
    "Jacquard12": "gothic blackletter pixel", "Jacquard24": "gothic blackletter pixel", "JacquardaBastarda9": "gothic blackletter", "MajorMonoDisplay": "monospace block"}
ROLE_LOOK = {"heavy": "heavy block", "tall": "tall condensed", "fine": "thin hairline", "mono": "monospace", "ornamental": "ornamental display", "pixel": "pixel-grid",
             "blackletter": "gothic blackletter", "decon": "outlined broken", "text": "bold sans"}
THIN_MAX, MID_MAX = 0.15, 0.30       # widest stroke / cap-line height: below THIN_MAX the face is thin (line pass), below MID_MAX it is mid (union), above it is heavy (depth)
FRAGMENTS = 60                       # a face whose "HANDS" breaks into more pieces than this is dotted / scratched: thin whatever its widest stroke
_STROKE: dict = {}


def font_look(font) -> str:
    """how the letters of `font` look, for the prompt ('dotted dot-matrix'); '' for an unknown face."""
    try:
        from mograph import fonts_liked as fl
        e = fl.lookup(font)
    except Exception:       # noqa: BLE001
        e = None
    if e is None:
        return ""
    return FONT_LOOK.get(e["key"]) or ROLE_LOOK.get(e["role"], "")


def stroke_measure(font, axes=None, text: str = "HANDS"):
    """(widest stroke / glyph-box height, connected pieces) of `text` drawn in `font` with PIL at the registry's default axes; None when the face / PIL is not available."""
    try:
        import os

        import cv2
        from PIL import Image, ImageDraw, ImageFont

        from mograph import font_registry as fr
        from mograph import fonts_liked as fl
        e = fl.lookup(font)
        if e is None:
            return None
        reg = next((v for v in fr.REGISTRY.values() if v["file"] == e["file"]), {})
        ax = {k: float(v) for k, v in (reg.get("axes") or {}).items()}
        for k, v in (axes or {}).items():
            v = v[1] if isinstance(v, (list, tuple)) else v
            try:
                ax[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        key = (e["file"], tuple(sorted(ax.items())), text)
        if key in _STROKE:
            return _STROKE[key]
        f = ImageFont.truetype(os.path.join(fr.FONT_DIR, e["file"]), 220)
        try:
            names = f.get_variation_axes()
        except Exception:       # noqa: BLE001
            names = []
        if names:
            tag_of = {str(n).lower(): t for t, n in fr.AXIS_NAMES.items()}
            vals = []
            for a in names:
                nm = a["name"].decode("latin-1").lower() if isinstance(a["name"], bytes) else str(a["name"]).lower()
                tg = next((t for t in ax if t.lower() == tag_of.get(nm, nm)), None)
                vals.append(min(max(ax[tg], a["minimum"]), a["maximum"]) if tg is not None else a["default"])
            f.set_variation_by_axes(vals)
        im = Image.new("L", (1600, 420), 0)
        ImageDraw.Draw(im).text((20, 20), text, font=f, fill=255)
        a = np.array(im) > 127
        ys, xs = np.where(a)
        if not len(ys):
            return None
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        dt = cv2.distanceTransform(a, cv2.DIST_L2, 3)
        r = (2.0 * float(dt.max()) / a.shape[0], int(cv2.connectedComponents(a)[0]) - 1)
        _STROKE[key] = r
        return r
    except Exception:       # noqa: BLE001
        return None


def stroke_class(font, axes=None) -> str:
    """'thin' | 'mid' | 'heavy' for a face. Measured (stroke_measure); an unmeasurable face falls back on its role (fine / decon -> thin, ornamental / pixel / blackletter -> mid, else heavy)."""
    m = stroke_measure(font, axes)
    if m is None:
        try:
            from mograph import fonts_liked as fl
            role = fl.role_of(font)
        except Exception:       # noqa: BLE001
            role = "heavy"
        return "thin" if role in ("fine", "decon") else "mid" if role in ("ornamental", "pixel", "blackletter") else "heavy"
    w, pieces = m
    return "thin" if (w < THIN_MAX or pieces > FRAGMENTS) else "mid" if w < MID_MAX else "heavy"


CONTROL_BY_CLASS = {"thin": "line", "mid": "union", "heavy": "depth"}


def control_pass_for(fonts, axes_list=None) -> str:
    """the type control pass for a shot whose LTX words use these faces: the thinnest face decides (a thin dotted word in a shot with a heavy one still needs the line pass)."""
    order = ("thin", "mid", "heavy")
    cls = [stroke_class(f, (axes_list or [None] * len(fonts))[i]) for i, f in enumerate(fonts) if f]
    return CONTROL_BY_CLASS[min(cls, key=order.index)] if cls else "depth"


# ---------------------------------------------------------------------------------------------------------------- fields
def norm_render(v, default: str | None = None):
    """-> (render, how) with how in ok | fix | None (None: not mendable, `default` returned). Exact names are `ok`; spelling / a synonym / the closest name is `fix`."""
    if isinstance(v, str) and v in RENDERS:
        return v, "ok"
    if isinstance(v, str) and v.strip():
        n = re.sub(r"[\s\-]+", "_", v.strip().lower())
        if n in RENDERS:
            return n, "fix"
        if n in SYN_RENDER:
            return SYN_RENDER[n], "fix"
        c = difflib.get_close_matches(n, list(RENDERS) + list(SYN_RENDER), n=1, cutoff=0.6)
        if c:
            return (c[0] if c[0] in RENDERS else SYN_RENDER[c[0]]), "fix"
    return default, None


def clamp_ltx_at(v):
    """-> (value, how): numbers (also '0.5' / '50%' strings; 2..100 are read as percent) clamped into LTX_AT_RANGE; not a number -> (None, None)."""
    if isinstance(v, bool):
        return None, None
    pct = False
    if isinstance(v, str):
        s = v.strip()
        pct = s.endswith("%")
        try:
            x = float(s.rstrip("%").replace(",", "."))
        except ValueError:
            return None, None
    elif isinstance(v, (int, float)):
        x = float(v)
    else:
        return None, None
    if x != x or abs(x) == float("inf"):
        return None, None
    read = pct or x > 1.0
    if read:
        x /= 100.0
    lo, hi = LTX_AT_RANGE
    y = min(max(x, lo), hi)
    return round(y, 3), ("ok" if (not read and not isinstance(v, str) and abs(y - x) < 1e-9) else "fix")


def clean_material(v, max_words: int = MATERIAL_MAX_WORDS):
    """-> (text, how): free text of at most `max_words` words, no quotes, no trailing 'letters' (the prompt clause adds it). Empty / not text -> (None, None)."""
    if not isinstance(v, (str, int, float)) or isinstance(v, bool):
        return None, None
    s = re.sub(r"[\"“”]", "", str(v)).strip()
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    s = re.sub(r"\b(?:letters|lettering|letterforms|text|typography)$", "", s, flags=re.I).strip(" ,.;:")
    if not s:
        return None, None
    ws = s.split()
    if len(ws) > max_words:
        s = " ".join(ws[:max_words]).rstrip(" ,.;:")
    return s, ("ok" if s == v else "fix")


_COLOURS = [((235, 235, 235), "white"), ((20, 20, 20), "black"), ((130, 130, 130), "grey"), ((200, 40, 40), "red"), ((240, 140, 40), "orange"),
            ((235, 200, 60), "golden"), ((60, 170, 80), "green"), ((60, 200, 200), "turquoise"), ((70, 120, 220), "blue"), ((150, 80, 200), "violet"),
            ((230, 110, 170), "pink"), ((140, 100, 60), "brown"), ((190, 225, 245), "pale blue")]
# (keywords in the scene, material) - the first row with a keyword in the scene wins; then the mood table
_SCENE_MATERIAL = [
    (("ice", "snow", "frozen", "glacier", "frost", "winter", "iceberg"), "thick carved ice, extruded, frosted edges"),
    (("sea", "ocean", "water", "river", "rain", "harbour", "harbor", "wave", "lake", "ferry", "flood"), "clear glass full of water, rounded edges"),
    (("forest", "wood", "pine", "moss", "leaf", "leaves", "jungle", "tree", "garden", "greenhouse"), "carved dark wood, moss in the grooves"),
    (("desert", "sand", "salt", "dune", "canyon", "dust"), "wind-worn sandstone slabs, crumbling edges"),
    (("neon", "market", "arcade", "club", "karaoke"), "glowing neon tubes with a hot core"),
    (("train", "tram", "test clip", "station", "factory", "machine", "engine", "bus", "city", "rooftop", "laundromat"), "brushed steel plates, riveted edges"),
    (("fire", "sunset", "sunrise", "dusk", "lantern", "candle", "ember", "lava", "sun"), "molten glass, glowing orange core"),
    (("fog", "mist", "cloud", "haze", "smoke", "steam"), "dense soft cloud, volumetric, lit from behind"),
    (("stone", "cliff", "mountain", "cathedral", "ruin", "castle", "rock", "lighthouse"), "chiselled granite blocks, mossy edges"),
    (("wheat", "field", "grass", "flower", "meadow", "petal"), "woven wheat stalks and petals"),
    (("space", "star", "galaxy", "moon", "night sky", "orbit"), "polished chrome, star-lit reflections"),
]
_MOOD_MATERIAL = {"dark": "black polished obsidian, sharp edges", "calm": "frosted glass, soft edges", "luxury": "polished brass, thick extruded",
                  "warm": "warm hammered copper, thick extruded", "nostalgic": "worn painted metal, chipped edges", "techno": "glowing neon tubes with a hot core",
                  "playful": "soft inflated rubber, glossy", "hype": "chrome with a sharp reflection, thick extruded"}


def _colour_name(rgb) -> str:
    try:
        a = np.array([float(x) for x in rgb][:3])
        if a.max() <= 1.0:
            a = a * 255.0
        return min(_COLOURS, key=lambda c: float(((np.array(c[0]) - a) ** 2).sum()))[1]
    except Exception:       # noqa: BLE001
        return ""


def default_material(mood: str | None = None, palette: dict | None = None, scene: str | None = None) -> str:
    """the rules' material (<= 8 words) from the scene text first, then the mood; the palette's accent / fg colour name leads when it fits in 8 words."""
    low = " ".join(str(scene or "").lower().replace(",", " ").split())
    words = set(low.split())
    base = next((m for keys, m in _SCENE_MATERIAL if any((k in words) if " " not in k else (k in low) for k in keys)), None)
    if base is None:
        mm = str(mood or "").lower()
        base = _MOOD_MATERIAL.get(mm) or next((v for k, v in _MOOD_MATERIAL.items() if k in mm), "polished glass, thick extruded, soft edges")
    col = ""
    pal = palette or {}
    acc = (pal.get("accents") or [None])[0] or pal.get("fg")
    if acc is not None:
        col = _colour_name(acc)
    tint = any(k in base for k in ("neon", "glass", "brass", "copper", "chrome", "steel", "rubber", "obsidian"))      # a colour only where the palette can tint the material
    if col and tint and len(base.split()) < MATERIAL_MAX_WORDS and col not in base.split():
        base = f"{col} {base}"
    return clean_material(base)[0]


def word_fields(w: dict, mood=None, palette=None, scene=None) -> dict:
    """a word with owner fields filled from the rules: {render, ltx_at, material} (material only when the word is not flat)."""
    r, _ = norm_render(w.get("render"), DEFAULT_RENDER)
    a, _ = clamp_ltx_at(w.get("ltx_at", DEFAULT_LTX_AT))
    d = {"render": r, "ltx_at": a if a is not None else DEFAULT_LTX_AT}
    if r != "flat":
        m, _ = clean_material(w.get("material"))
        d["material"] = m or default_material(mood, palette, scene)
    return d


def norm_owner(v, label: str = "owners[0]") -> tuple[dict, list[str]]:
    """one owner entry of a CONCEPT reply (`{render, ltx_at, material}`; a bare string is the render) -> (clean entry, fixes). Only what the model gave and what could
    be mended is in the entry: a missing key stays missing (the planner then uses the rules' default for that word). ltx_at only for handoff, material only
    when the word is not flat. The same enums / clamps / trims as the design schema (norm_render, clamp_ltx_at, clean_material)."""
    fixes: list[str] = []
    if isinstance(v, str):
        v = {"render": v}
    if not isinstance(v, dict):
        return {}, ([f"{label}: not an object, the rules' owner used"] if v is not None else [])
    out: dict = {}
    if v.get("render") is not None:
        r, how = norm_render(v.get("render"), None)
        if r is None:
            fixes.append(f"{label}.render {v.get('render')!r}: not one of {', '.join(RENDERS)}, the rules' owner used")
        else:
            out["render"] = r
            if how == "fix":
                fixes.append(f"{label}.render {v.get('render')!r} -> {r!r}")
    if out.get("render") == "handoff" and v.get("ltx_at") is not None:
        a, how = clamp_ltx_at(v.get("ltx_at"))
        if a is None:
            fixes.append(f"{label}.ltx_at {v.get('ltx_at')!r}: not a number, {DEFAULT_LTX_AT} used")
        else:
            out["ltx_at"] = a
            if how == "fix":
                fixes.append(f"{label}.ltx_at {v.get('ltx_at')!r} -> {a:g}")
    if out.get("render", "handoff") != "flat" and v.get("material") is not None:
        m, how = clean_material(v.get("material"))
        if m:
            out["material"] = m
            if how == "fix":
                fixes.append(f"{label}.material {str(v.get('material'))[:60]!r} -> {m!r}")
    return out, fixes


def remap_owners(owners, before: list[str], after: list[str]):
    """owner entries per line of `before` -> per line of `after` (the lines were merged / split by plan_rules.fit_lockups; the words and their order never change):
    the owner of an `after` line is the owner of the `before` line that holds its first word. None / empty owners -> None; a missing entry -> {}."""
    if not owners:
        return None
    owners = list(owners)
    if list(before) == list(after):
        return (owners + [{}] * len(after))[:len(after)]
    starts, n = [], 0
    for ln in before:
        starts.append(n)
        n += len(str(ln).split())
    out, off = [], 0
    for ln in after:
        k = max([j for j, s0 in enumerate(starts) if s0 <= off] or [0])
        out.append(owners[k] if k < len(owners) and isinstance(owners[k], dict) else {})
        off += len(str(ln).split())
    return out


def has_owner(words) -> bool:
    """does any word of the piece carry an owner field? (False = an OLD piece: every word is flat, nothing changes)."""
    return any(isinstance(w, dict) and w.get("render") is not None for w in (words or []))


def effective(words) -> list[dict]:
    """per word: {render, ltx_at, material}. Old piece (no `render` anywhere) -> all flat; else a missing / invalid `render` is the default (handoff)."""
    owned = has_owner(words)
    out = []
    for w in words or []:
        if not owned:
            out.append({"render": LEGACY_RENDER, "ltx_at": DEFAULT_LTX_AT, "material": ""})
            continue
        r, _ = norm_render(w.get("render"), DEFAULT_RENDER)
        a, _ = clamp_ltx_at(w.get("ltx_at", DEFAULT_LTX_AT))
        out.append({"render": r, "ltx_at": DEFAULT_LTX_AT if a is None else a, "material": str(w.get("material") or "")})
    return out


# ---------------------------------------------------------------------------------------------------------------- windows
def manifest_words(manifest) -> list[dict]:
    """info.words of a type-pass manifest dict (or None)."""
    return list((((manifest or {}).get("info") or {}).get("words")) or [])


def windows(piece: dict, rows=None) -> list[dict] | None:
    """per type word: {start, end, next, render, ltx_at, material, text}, piece seconds. `rows` = the type pass manifest's info.words (the TRUE timeline the
    passes were rendered with), else the words' own t0 / t1. None for an old piece (no owner field): callers then change nothing."""
    ws = (piece.get("type") or {}).get("words") or []
    if not has_owner(ws) or str(piece.get("quality") or "") == "draft":      # a draft (and a still preview, which is a draft) has no LTX letters: the flat type stays the only copy
        return None
    eff = effective(ws)
    out = []
    try:
        beat_s = 60.0 / float((piece.get("type") or {}).get("bpm") or 0) if float((piece.get("type") or {}).get("bpm") or 0) > 0 else MIN_FLAT_S
    except (TypeError, ValueError):
        beat_s = MIN_FLAT_S
    for i, w in enumerate(ws):
        r = rows[i] if rows and i < len(rows) else None
        if r is not None and "start" in r and "end" in r:
            a, b = float(r["start"]), float(r["end"])
        else:
            a = float(w.get("t0", 0.0))
            b = float(w.get("t1", a + 2.0))
        lt = w.get("ltx_t")
        lt = float(lt) if isinstance(lt, (int, float)) and not isinstance(lt, bool) else None
        hit = r.get("hit") if r is not None else None
        if not isinstance(hit, (int, float)) or isinstance(hit, bool):
            hit = a + float(w.get("form_s") or MIN_FLAT_S)
        out.append(dict(start=a, end=max(b, a + 1e-3), next=None, ltx_t=lt, hit=float(hit), beat=beat_s, text=str(w.get("text") or " ".join(w.get("lines") or [])), look=font_look(w.get("font")),
                        lines=[str(x) for x in (w.get("lines") or [])], **eff[i]))
    for i in range(len(out) - 1):
        out[i]["next"] = out[i + 1]["start"]
    return out


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def handoff_t(w: dict) -> float:
    """time where LTX takes over a word: `ltx_t` (absolute seconds; written by the design step when the footage was already generated with an owner timeline,
    so the flat fade lands where the LTX letters really start) else start + ltx_at x (end - start).
    LEGIBILITY FLOOR (2026-10-30, v026 r01 GLASS faded before its hold): the flat word stays fully drawn for at least 1 beat after it lands (`hit` + `beat`; MIN_FLAT_S without a
    tempo) and an `ltx_t` outside the word's own span (written for an earlier plan timeline: GLASS had 4.44 s for a word that starts at 6.37 s, so its flat copy never showed) is
    ignored. A final handoff time (`final`, the gain spec of a recipe input) is returned as it is."""
    if w.get("final") and w.get("ltx_t") is not None:
        return float(w["ltx_t"])
    a, b = float(w["start"]), float(w["end"])
    lt = w.get("ltx_t")
    if lt is not None and a - 1e-6 <= float(lt) <= b + 1e-6:
        t = float(lt)
    else:
        t = a + float(w.get("ltx_at", DEFAULT_LTX_AT)) * (b - a)
    hit = w.get("hit")
    hit = float(hit) if isinstance(hit, (int, float)) and not isinstance(hit, bool) else a
    beat = w.get("beat")
    beat = float(beat) if isinstance(beat, (int, float)) and not isinstance(beat, bool) and beat > 0 else MIN_FLAT_S
    return max(t, hit + max(beat, MIN_FLAT_S))


def word_at(wins: list[dict], t: float) -> dict | None:
    """the word whose region [start, next start) holds t (before the first start: the first word)."""
    if not wins:
        return None
    k = 0
    for j, w in enumerate(wins):
        if t >= w["start"] - 1e-9:
            k = j
    return wins[k]


def flat_gain_of(w: dict, t: float, fade: float = FLAT_FADE_S) -> float:
    r = w["render"]
    if r == "flat":
        return 1.0
    if r == "ltx":
        return 0.0
    return 1.0 - _smooth((t - handoff_t(w)) / max(fade, 1e-6))


def control_gain_of(w: dict, t: float, ramp: float = CONTROL_RAMP_S) -> float:
    r = w["render"]
    if r == "flat":
        return 0.0
    if r == "ltx":
        return 1.0
    return _smooth((t - handoff_t(w)) / max(ramp, 1e-6))


def flat_gain(wins: list[dict] | None, t: float, fade: float = FLAT_FADE_S) -> float:
    """the share of the flat type compose draws at time t (1.0 when `wins` is None = an old piece)."""
    if wins is None:
        return 1.0
    w = word_at(wins, t)
    return 1.0 if w is None else flat_gain_of(w, t, fade)


def control_gain(wins: list[dict] | None, t: float, ramp: float = CONTROL_RAMP_S) -> float:
    """the share of the type control pass LTX gets at time t (1.0 when `wins` is None = an old piece: the whole pass, as before)."""
    if wins is None:
        return 1.0
    w = word_at(wins, t)
    return 0.0 if w is None else control_gain_of(w, t, ramp)


def control_gains(piece: dict, start: int, frames: int, rows=None, fps: int | None = None) -> np.ndarray | None:
    """float32 [frames] control gain of the piece frames start..start+frames-1 (None for an old piece)."""
    wins = windows(piece, rows)
    if wins is None:
        return None
    fps = int(fps or piece.get("fps") or 24)
    return np.array([control_gain(wins, (start + i) / fps) for i in range(int(frames))], np.float32)


def flat_visible_until(w: dict) -> float:
    """the time the flat word is still fully drawn: +inf (flat), -inf (ltx), the handoff time."""
    if w["render"] == "flat":
        return float("inf")
    if w["render"] == "ltx":
        return float("-inf")
    return handoff_t(w)


def clip_word_rows(piece: dict, rows: list, rows_manifest=None) -> list:
    """rows = [(start, hit, release, end), ...] one per word in order (the legibility machinery's hold windows) -> the same rows clipped to the part of
    the word the FLAT type is drawn: an ltx word is dropped, a handoff word's release / end stop at the handoff time (dropped when the handoff comes
    before its hit). An old piece: rows untouched. Index-aligned input only."""
    wins = windows(piece, rows_manifest)
    if wins is None or len(wins) != len(rows):
        return rows
    out = []
    for w, r in zip(wins, rows):
        until = flat_visible_until(w)
        if until == float("inf"):
            out.append(r)
        elif until <= float(r[1]) + 0.05:
            continue
        else:
            out.append(tuple(r[:2]) + (min(float(r[2]), until), min(float(r[3]), until)) + tuple(r[4:]))
    return out


# ---------------------------------------------------------------------------------------------------------------- shots
def shot_span_s(piece: dict, sid: str):
    """(start s, end s) of shot `sid` in piece time (the layout)."""
    fps = int(piece.get("fps") or 24)
    L = (piece.get("layout") or {}).get(sid)
    if L is None:
        return None
    n = int(L.get("frames") or 0)
    sh = next((s for s in piece.get("shots") or [] if s.get("id") == sid), {})
    n = int(sh.get("span") or n)
    return L["start"] / fps, (L["start"] + n) / fps


def shot_words(wins: list[dict] | None, a: float, b: float, step: float = 1.0 / 24) -> list[dict]:
    """words LTX owns somewhere inside [a, b): those with control gain > 0 at any frame of the window (first-seen order)."""
    if wins is None:
        return []
    seen, out = set(), []
    t = a
    while t < b - 1e-9:
        w = word_at(wins, t)
        if w is not None and control_gain_of(w, t) > 1e-3 and id(w) not in seen:
            seen.add(id(w))
            out.append(w)
        t += step
    return out


def shot_owned(wins, a: float, b: float) -> bool:
    return bool(shot_words(wins, a, b))


def shot_guide(wins, a: float, b: float, base: float) -> float:
    """the control guide of a shot: unchanged for an old piece; OFF_GUIDE (plain i2v) when no word of the shot is LTX's; else max(base, OWNED_GUIDE)."""
    if wins is None:
        return float(base)
    return max(float(base), OWNED_GUIDE) if shot_owned(wins, a, b) else OFF_GUIDE


def clause(text: str, material: str, look: str | None = None, lines=None) -> str:
    """the prompt clause for one word: the word "SPIN" in <look> capitals stacked on 2 lines formed as <material> letters, extruded 3D lettering (a lockup of several words: `the words "A B" ...`).
    `look` = the graphic's own look (font_look: 'dotted dot-matrix') and `lines` its line breaks: the video model is told what the control pass shows, so its letters follow the graphic and it
    never invents a layout of its own. Without them the clause is the plain old form. The material never ends in 'letters' (clean_material drops it), so the clause cannot say 'letters letters'."""
    t = re.sub(r"\s+", " ", str(text).strip().upper())
    m = clean_material(material)[0] or default_material()
    seg = (f" in {look} capitals" if look else "") + (f" stacked on {len(lines)} lines" if lines and len(lines) > 1 else "")
    return f'the {"word" if len(t.split()) <= 1 else "words"} "{t}"{seg} formed as {m} letters, {CLAUSE_TAIL}'


CLAUSE_RE = re.compile(r'(?:,\s*)?the words? "[^"]*"(?: in [a-z0-9\- /]+? capitals)?(?: stacked on \d+ lines)? formed as [^"]*? letters(?:,\s*extruded 3D lettering)?(?=\s*(?:,|$))', re.I)     # also the old `the words "X" formed as M letters`


def strip_clause(prompt: str) -> str:
    """the prompt without the owner clauses (the clean chain of the three-layer route shows the scene only)."""
    s = CLAUSE_RE.sub("", str(prompt or ""))
    return re.sub(r"\s*,\s*,+", ",", s).strip(" ,")


def prompt_with_clause(prompt: str, wins, a: float, b: float) -> str:
    """the shot prompt + one clause per LTX-owned word of the shot (idempotent; an old piece / no owned word: unchanged). The clause goes before the trailing
    'cinematic' tag, after the scene and motion text."""
    ws = shot_words(wins, a, b)[:MAX_CLAUSES]
    if not ws:
        return prompt
    cl = [clause(w["text"], w["material"] or default_material(), w.get("look"), w.get("lines")) for w in ws]
    base = strip_clause(prompt)
    parts = [p.strip() for p in base.split(",")]
    tail = [parts.pop()] if parts and parts[-1].lower() == "cinematic" else []
    return ", ".join([p for p in parts if p] + cl + tail)


def slot_plan(piece: dict, manifest, start: int, frames: int, span: int, prompt: str, base_guide: float, fps: int | None = None):
    """what MographShotSlot does for one shot: -> (control gain float32 [frames] or None, guide, prompt). `manifest` = the control type pass manifest (info.words = the
    true word timeline), start / frames / span in piece frames (frames = generated length, span = the part on screen). An old piece (no owner field): (None,
    base_guide, prompt) = unchanged. Else the control pass holds only what LTX owns (gain 0 where it owns nothing), the guide is OFF_GUIDE for an all-flat
    shot (plain i2v) / max(base, OWNED_GUIDE), and the prompt gets the letter clause of every LTX-owned word."""
    wins = windows(piece, manifest_words(manifest) or None)
    if wins is None:
        return None, float(base_guide), prompt
    fps = int(fps or piece.get("fps") or 24)
    gains = np.array([control_gain(wins, (start + i) / fps) for i in range(int(frames))], np.float32)
    a, b = start / float(fps), (start + int(span)) / float(fps)
    return gains, shot_guide(wins, a, b, base_guide), prompt_with_clause(prompt, wins, a, b)


def _letters_of(w: dict) -> int:
    return len(re.sub(r"[^A-Za-z0-9]", "", str(w.get("text") or "")))


def _alpha_len(w: dict) -> int:
    return len(re.sub(r"[^A-Za-z]", "", str(w.get("text") or "")))


def shot_cap(dur_s: float) -> int:
    """most LTX-owned words one shot may hold: MAX_LTX_PER_SHOT (1), MAX_LTX_LONG_SHOT (2) when the shot lasts at least LTX_LONG_SHOT_S."""
    return MAX_LTX_LONG_SHOT if float(dur_s) >= LTX_LONG_SHOT_S - 1e-9 else MAX_LTX_PER_SHOT


def cap_per_shot(piece: dict) -> list[str]:
    """LTX letters are fragile: an LTX-owned word has at most MAX_LTX_LETTERS letters (longer ones go flat), and a shot keeps at most shot_cap(duration) LTX-owned words (render ltx / handoff, `shot_words`). The key word is the one with the most
    letters (ties: the first); the others are demoted to flat (ltx_at / ltx_t / material dropped) IN PLACE on piece["type"]["words"]. -> one fix text per demotion
    (the caller logs them). An old / draft piece: nothing happens."""
    wins = windows(piece)
    if wins is None:
        return []
    words = (piece.get("type") or {}).get("words") or []
    fixes: list[str] = []
    for i, w in enumerate(wins):                  # length first: LTX can only spell a short word (<= MAX_LTX_LETTERS letters)
        if w["render"] in ("ltx", "handoff") and _alpha_len(w) > MAX_LTX_LETTERS:
            was = w["render"]
            w["render"] = "flat"
            if i < len(words):
                words[i]["render"] = "flat"
                for k in ("ltx_at", "ltx_t", "material"):
                    words[i].pop(k, None)
            fixes.append(f"type.words[{i}] {w['text']!r}: render {was!r} -> 'flat': {_alpha_len(w)} letters; an LTX-owned word has at most {MAX_LTX_LETTERS} letters "
                         f"(LTX misspells / cannot form longer words)")
    pos = {id(w): i for i, w in enumerate(wins)}
    for sh in piece.get("shots") or []:
        sp = shot_span_s(piece, sh["id"])
        if sp is None:
            continue
        a, b = sp
        owned = [pos[id(w)] for w in shot_words(wins, a, b)]
        cap = shot_cap(b - a)
        if len(owned) <= cap:
            continue
        keep = set(sorted(owned, key=lambda i: (-_letters_of(wins[i]), i))[:cap])
        for i in owned:
            if i in keep:
                continue
            was = wins[i]["render"]
            wins[i]["render"] = "flat"
            if i < len(words):
                words[i]["render"] = "flat"
                for k in ("ltx_at", "ltx_t", "material"):
                    words[i].pop(k, None)
            fixes.append(f"type.words[{i}] {wins[i]['text']!r}: render {was!r} -> 'flat': shot {sh['id']} ({b - a:.1f} s) may hold {cap} LTX-owned word(s) "
                         f"(kept {', '.join(repr(wins[j]['text']) for j in sorted(keep))}; LTX letters are fragile, one key word per shot)")
    return fixes


def apply_to_plan(piece: dict) -> dict:
    """plan time (plan_rules._build / the design step): per shot, from the words' planned t0 / t1: the prompt gets the letter clause of every LTX-owned word, the
    control guide is OWNED_GUIDE (an LTX-owned word in the shot) or OFF_GUIDE (all flat). `shots[k].control.guide` is what MographShotSlot reads. An old
    piece is returned untouched. -> the piece (changed in place)."""
    wins = windows(piece)
    if wins is None:
        return piece
    from tools.fusion import letter_fill as _LF  # 2026-11-01: solid letters by default; an LTX material word only when the user's direction asks for one
    flat = _LF.solid_owners(piece)
    wins = windows(piece)                        # after solid_owners: it flattens an LTX word, or restores one a pass before the user's direction was stamped on the piece
    if flat:
        po = piece.setdefault("plan", {}).setdefault("owner", {})
        po.setdefault("fixes", []).extend(flat)
        piece["plan"].setdefault("notes", []).extend(flat)
    capped = cap_per_shot(piece)                 # at most 1 LTX-owned word per shot (2 in a shot >= LTX_LONG_SHOT_S): the others are flat
    if capped:
        wins = windows(piece)
        po = piece.setdefault("plan", {}).setdefault("owner", {})
        po.setdefault("fixes", []).extend(capped)
        piece["plan"].setdefault("notes", []).extend(capped)
    words_ = (piece.get("type") or {}).get("words") or []
    for s in piece.get("shots") or []:
        sp = shot_span_s(piece, s["id"])
        if sp is None:
            continue
        a, b = sp
        s["prompt"] = prompt_with_clause(s.get("prompt", ""), wins, a, b)
        base = float((s.get("control") or {}).get("guide", s.get("guide", 1.0)))
        g = shot_guide(wins, a, b, base)
        s.setdefault("control", {})["guide"] = g
        s["guide"] = g
        owned = shot_words(wins, a, b)
        idx = [i for i, w in enumerate(wins) if any(w is o for o in owned)]
        if idx and (piece.get("layers") or {}).get("control_pass") != "fixed":        # the control pass suits the glyph weight of the LTX word(s) of the shot: thin / dotted / outline -> canny line, mid -> union, heavy -> depth
            cp = control_pass_for([words_[i].get("font") for i in idx if i < len(words_)], [words_[i].get("axes") for i in idx if i < len(words_)])
            if cp != s["control"].get("pass"):
                piece.setdefault("plan", {}).setdefault("owner", {}).setdefault("control_pass", []).append(
                    f"{s['id']}: control pass {s['control'].get('pass')!r} -> {cp!r} for {', '.join(repr(wins[i]['text']) for i in idx)} ({', '.join(stroke_class(words_[i].get('font'), words_[i].get('axes')) for i in idx)} face)")
                s["control"]["pass"] = cp
            need = {"line": ("line",), "depth": ("depth",), "union": ("depth", "line")}.get(cp, ())
            tp = piece.setdefault("type_passes", [])
            for n_ in need:
                if n_ not in tp:
                    tp.append(n_)
    return piece


# ---------------------------------------------------------------------------------------------------------------- cube-lattice words that cannot be read (render R17)
# R17: FINAL / ECHO stacked in Bungee Inline at wf 0.9 on `voxel_roll` rendered as one merged black block (6 cube rows per cap height: the gaps between a fat face's letters and
# between the two lines fall inside one cell), and CRASHING FAITH in MoiraiOne (hairline) voxelised to 3 cubes. The lattice words measure themselves on the cell grid
# (voxel_roll.cells_legible: every letter keeps >= 4 cubes, the cubes fall into >= 0.75 x letters pieces); a word that fails is moved to a plain readable module at plan time.
LATTICE_MODULES = ("voxel_roll",)
LATTICE_FALLBACK = "lanes"                         # reading-first strand module (design_defaults.READING_FIRST)
LATTICE_PARAMS = ("rows", "lean", "run", "grid", "cull", "tail_len")


def _frame_aspect(piece: dict) -> float:
    """frame width / height of a piece: its type render size, else its `format` ('9:16'), else 9:16."""
    try:
        sz = ((piece.get("sizes") or {}).get("type")) or ((piece.get("sizes") or {}).get("shot"))
        if sz and float(sz[1]) > 0:
            return float(sz[0]) / float(sz[1])
        a, b = str(piece.get("format") or "").split(":")
        return float(a) / float(b)
    except (TypeError, ValueError, KeyError, IndexError):
        return 0.5625


def fix_illegible_modules(piece: dict) -> list[str]:
    """plan time, every design source (called from text_guard.fix_plan): a word on a cube-lattice module (`LATTICE_MODULES`) whose cell grid does not keep its letters apart is moved to a
    readable module: one the piece already uses for another word (not a lattice one), else LATTICE_FALLBACK. Its module-only params are dropped. In place; -> one fix text per word."""
    words = [w for w in ((piece.get("type") or {}).get("words") or [])]
    if not any(isinstance(w, dict) and w.get("module") in LATTICE_MODULES for w in words):
        return []
    try:
        from mograph.modules.words.voxel_roll import cells_legible
    except Exception:       # noqa: BLE001  (a bare install without the module package: nothing to measure)
        return []
    asp = _frame_aspect(piece)
    fixes = []
    for i, w in enumerate(words):
        if not isinstance(w, dict) or w.get("module") not in LATTICE_MODULES:
            continue
        lines = w.get("lines") or [t for t in str(w.get("text") or "").split("\n") if t.strip()] or [str(w.get("text") or "")]
        pl = w.get("placement") or {}
        font = w.get("font") or w.get("font_pin")
        if w.get("axes"):
            font = {"name": font, "axes": dict(w["axes"])}
        try:
            r = cells_legible(font, [str(x) for x in lines], wf=float(pl.get("wf") or 0.9), hf=float(pl.get("hf") or 0.3), aspect=asp, roll=float(pl.get("roll") or 0.0),
                              **({"rows": int((w.get("params") or {}).get("rows"))} if (w.get("params") or {}).get("rows") else {}))
        except Exception:       # noqa: BLE001  (an unknown face / an empty word: leave the word to the other guards)
            continue
        if r["ok"]:
            continue
        others = [x.get("module") for j, x in enumerate(words) if j != i and isinstance(x, dict) and x.get("module") and x.get("module") not in LATTICE_MODULES]
        new = max(set(others), key=lambda m: (others.count(m), -others.index(m))) if others else LATTICE_FALLBACK
        was = w["module"]
        w["module"] = new
        if isinstance(w.get("params"), dict):
            for k in LATTICE_PARAMS:
                w["params"].pop(k, None)
        fixes.append(f"type.words[{i}] {w.get('text')!r}: module {was} -> {new}: {r['reason']} (cells {r['cells']}, letters {r['letters']}, pieces {r['pieces']})")
    if fixes:
        piece.setdefault("plan", {}).setdefault("module_legibility", []).extend(fixes)
        piece["plan"].setdefault("notes", []).extend(fixes)
    return fixes


# ---------------------------------------------------------------------------------------------------------------- no speckle letters (user 2026-11-01, R18 / R19)
# R19 RISING SWELL and R18 GLOWING / STILLNESS / ASCEND: every word on the `particles` module, the letters a speckle of white dots (and the piece_system drew that ONE module for every word). User: "i dont like the text
# being filled with texture like this ... if its 3d it can be a type of material like metal or chrome or marble but not this bitsy texture fill", then "throughout any generation": no letter is a dot cloud in the
# form-in, the hold or the exit. A word that names the `particles` module (design_defaults.SPECKLE_MODULES; 2026-11-02 also the retired `outflow`, BANNED_MODULES) or a `particles` form_in / form_out (SPECKLE_IO) is swapped at plan time.
SPECKLE_PARAMS = ("n", "mode", "kinds", "size", "eye", "runway", "spread", "speed")      # the particles module's own parameters (dropped with it)
RETIRED_PARAMS = ("tilt", "rings", "n_arcs", "width", "speed")                           # the outflow module's own parameters (mograph.modules.words._rings; dropped with it)
BANNED_PARAMS = {"particles": SPECKLE_PARAMS, "outflow": RETIRED_PARAMS}


def fix_speckle_forms(piece: dict) -> list[str]:
    """plan time, every design source (called from text_guard.fix_plan): no word keeps the `particles` module (speckle) or the retired `outflow` module (design_defaults.BANNED_MODULES), or a `particles` io style. The module becomes one the piece already uses for another word
    (most used, not a speckle one), else the next of design_defaults.SPECKLE_MODULE_FALLBACK by word index (a piece of three speckle words becomes three different line / slat / ring words); the io style
    becomes the same side's style on another word, else the next of SPECKLE_IO_FALLBACK. The piece system's own module / io lists follow. In place; -> one fix text per change."""
    from tools.fusion import design_defaults as DD
    ty = piece.get("type") or {}
    words = [w for w in (ty.get("words") or [])]
    fixes: list[str] = []
    mods = [w.get("module") for w in words if isinstance(w, dict)]
    if any(m in DD.BANNED_MODULES for m in mods):
        used = [m for m in mods if m and m not in DD.BANNED_MODULES]
        pool = [m for m in DD.SPECKLE_MODULE_FALLBACK if m not in used] or list(DD.SPECKLE_MODULE_FALLBACK)
        k = 0
        for i, w in enumerate(words):
            if not isinstance(w, dict) or w.get("module") not in DD.BANNED_MODULES:
                continue
            others = [x for x in used if x]
            distinct = {m for m in used if m}
            if len(distinct) >= DD.MAX_MODULES - 1 and others:
                new = max(distinct, key=lambda m: (others.count(m), -others.index(m)))      # the piece is at its module cap: reuse
            else:
                new = pool[k % len(pool)]
                k += 1
                used.append(new)
            was = w["module"]
            w["module"] = new
            if isinstance(w.get("params"), dict):
                for p in BANNED_PARAMS.get(was, SPECKLE_PARAMS):
                    w["params"].pop(p, None)
            why = "retired: rings streaming in from one side and out the other are not offered" if was in DD.RETIRED_MODULES else "the letters were a speckle of dots; letters are never a dot cloud"
            fixes.append(f"type.words[{i}] {w.get('text')!r}: module {was} -> {new} ({why})")
    for side in ("form_in", "form_out"):
        have = [w.get(side) for w in words if isinstance(w, dict)]
        if not any(v in DD.SPECKLE_IO for v in have):
            continue
        ok = [v for v in have if v and v not in DD.SPECKLE_IO]
        k = 0
        for i, w in enumerate(words):
            if not isinstance(w, dict) or w.get(side) not in DD.SPECKLE_IO:
                continue
            new = ok[i % len(ok)] if ok else DD.SPECKLE_IO_FALLBACK[(i + k) % len(DD.SPECKLE_IO_FALLBACK)]
            was = w[side]
            w[side] = new
            if side == "form_in" and new != "lines":
                w.pop("entry", None)
            fixes.append(f"type.words[{i}] {w.get('text')!r}: {side} {was} -> {new} (a dot-cloud letter form is never used)")
    ps = (piece.get("plan") or {}).get("piece_system")
    if isinstance(ps, dict):
        ms = [m for m in (ps.get("modules") or []) if m not in DD.BANNED_MODULES]
        wm = [m for m in (ps.get("word_modules") or []) if m not in DD.BANNED_MODULES]
        if ps.get("word_modules") and len(wm) != len(ps["word_modules"]):
            now_w = [w.get("module") for w in words if isinstance(w, dict) and w.get("module")]
            ps["word_modules"] = now_w if len(now_w) == len(words) else (wm or list(DD.SPECKLE_MODULE_FALLBACK[:1]))      # word i takes word_modules[i]: follow the swapped words
            fixes.append(f"plan.piece_system word_modules -> {ps['word_modules']} (no banned module)")
        if ps.get("module") in DD.BANNED_MODULES or len(ms) != len(ps.get("modules") or []):
            now = [w.get("module") for w in words if isinstance(w, dict) and w.get("module")]
            ps["modules"] = list(dict.fromkeys(ms + now)) or list(DD.SPECKLE_MODULE_FALLBACK[:1])
            if ps.get("module") in DD.BANNED_MODULES:
                ps["module"] = ps["modules"][0]
            fixes.append(f"plan.piece_system modules -> {ps['modules']} (no speckle / retired module)")
        io = ps.get("io")
        if isinstance(io, dict):
            for side in ("form_in", "form_out"):
                v = io.get(side)
                if isinstance(v, list) and any(x in DD.SPECKLE_IO for x in v):
                    keep = [x for x in v if x not in DD.SPECKLE_IO]
                    for x in DD.SPECKLE_IO_FALLBACK:
                        if len(keep) >= max(len(v), 2):
                            break
                        if x not in keep:
                            keep.append(x)
                    io[side] = keep
                    fixes.append(f"plan.piece_system.io.{side} -> {keep} (no speckle io style)")
    if fixes:
        piece.setdefault("plan", {}).setdefault("notes", []).extend(fixes)
    return fixes


# ---------------------------------------------------------------------------------------------------------------- ONE text at any moment
HANDOFF_MAX_FADE_S = 0.3        # the flat graphic crossfades into the rendered letters over at most this long, in the same place; the two are never both held


def check_one_text(piece: dict, textbg: dict | None = None, rows=None, step: float = 1.0 / 24) -> list[str]:
    """problems against 'one text at any moment' (user 2026-10-29: "I want the rendered text to match what the graphic text is - not do two different versions on top of each other"):
    1. the flat share and the rendered share of every word sum to 1 at every sample of its span (the crossfade of a handoff is the only moment both show, and then they add up to one text),
    2. the crossfades (FLAT_FADE_S, CONTROL_RAMP_S) are at most HANDOFF_MAX_FADE_S,
    3. `textbg` {shot id: bool} (compose: which shots still have a text-rendered background): a shot WITHOUT one has no rendered letters, so no word of it may hand over to LTX (its flat
       share would fade to nothing). [] = fine. Old / draft pieces (no owner fields) have one owner (flat) and pass."""
    probs: list[str] = []
    wins = windows(piece, rows)
    if wins is None:
        return probs
    if FLAT_FADE_S > HANDOFF_MAX_FADE_S + 1e-9 or CONTROL_RAMP_S > HANDOFF_MAX_FADE_S + 1e-9:
        probs.append(f"the handoff crossfade ({FLAT_FADE_S:g} s flat, {CONTROL_RAMP_S:g} s control) is longer than {HANDOFF_MAX_FADE_S:g} s")
    for i, w in enumerate(wins):
        t = w["start"]
        worst = 0.0
        while t < w["end"]:
            worst = max(worst, abs(flat_gain_of(w, t) + control_gain_of(w, t) - 1.0))
            t += step
        if worst > 1e-6:
            probs.append(f"word {i} {w['text']!r}: the flat and the rendered share do not add up to one text (off by {worst:.2f})")
    if textbg:
        for sid, has in textbg.items():
            sp = shot_span_s(piece, sid)
            if has or sp is None:
                continue
            own = shot_words(wins, sp[0], sp[1])
            if own:
                probs.append(f"shot {sid}: no text-rendered background but {[w['text'] for w in own]} is LTX-owned: its flat graphic would fade out and nothing would replace it")
    return probs


# ---------------------------------------------------------------------------------------------------------------- compose seam
def gain_spec(wins: list[dict] | None, bg=None) -> dict | None:
    """the JSON-able `gain` of a recipe input (type_matte / type_line / type_beauty): the windows + the fade (+ the beauty background colour 0..1)."""
    if wins is None:
        return None
    sp = {"owner": [[round(w["start"], 4), round(w["end"], 4), w["render"], round(handoff_t(w), 4)] for w in wins], "fade": FLAT_FADE_S}
    if bg is not None:
        sp["bg"] = [round(float(c), 5) for c in bg]
    return sp


def gain_at(spec: dict, t: float) -> float:
    wins = [dict(start=a, end=b, render=r, ltx_at=DEFAULT_LTX_AT, ltx_t=th, final=True) for a, b, r, th in spec["owner"]]
    return flat_gain(wins, t, float(spec.get("fade", FLAT_FADE_S)))


def apply_gain(frame, spec: dict, t: float):
    """frame (HxW or HxWx3 float array, numpy or torch) scaled by the owner gain at time t. With spec['bg'] (the beauty pass) the frame is faded toward
    that background colour instead of toward black, so type_face's colour recovery (beauty - bg * (1 - alpha)) stays exact while the alpha fades too."""
    g = gain_at(spec, t)
    if g >= 1.0 - 1e-6:
        return frame
    bg = spec.get("bg")
    if bg is not None and getattr(frame, "ndim", 0) == 3:
        if hasattr(frame, "new_tensor"):
            b = frame.new_tensor(bg)
        else:
            b = np.asarray(bg, np.float32)
        return b + (frame - b) * g
    return frame * g
