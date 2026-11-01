"""design_defaults: the qualities of the draft strips (16_app/core_check/transition_cut_check: clean type, a designed cut) as DEFAULTS that normalize / salvage enforce on every design
(pure functions on the design words / the piece, no ComfyUI, no network, no Claude; every change is one fix string, logged by the caller).

User 2026-10-29: "can we get more of this stuff happening - clear text and transitions?". Generated pieces read worse than the drafts: too many modules and treatments, smears and
echoes on the holds, words too small for a phone. The rules (guide section 20 tells the models the same up front; this repairs what they did not follow):

  1. CLEAR HOLDS   a word's hold has no smear / echo / flicker / particle noise: flow_wake is fully settled inside every hold (`enforce_compose`: the settle timeline of the hold windows
                   replaces whatever the chain had; plan_rules.SETTLE_HOLD = 1.0), the modules that keep echo copies behind the word (HOLD_NOISY) are replaced by the piece's reading module,
                   io styles (form_in / form_out) are the only place for effects: they act in the in / out phases by construction.
  2. ONE SYSTEM    at most MAX_MODULES modules (strand material types), at most MAX_FONTS faces (liked, incl. one expressive) and one matte treatment, at most MAX_FORMS io styles per
                   piece; extras are folded field by field into the kept ones (the reading-first module and the most used face win).
  3. SIZE FLOOR    every word reads at phone size: cap height >= CAP_FLOOR (9 %, the guide's absolute floor) of the frame height, estimated from the real glyph widths of its face
                   (design_options.cap_estimates); a word below it gets its lines stacked / its box widened (the guide asks 25 % for a headline; this is only the floor).
  4. LEGIBILITY GUARD stays on: compose.legible / the ops' `legible` are never False in a generated piece.
"""
from __future__ import annotations

import copy
import re

from tools.fusion import design_options as DO

EXPRESSIVE = ("pixel", "ornamental", "decon", "blackletter")
READING_FIRST = ("lanes", "slats", "gradient_fade", "staircase", "beat_bars")                                        # guide section 5: the word is plainly legible on its hold
# 2026-10-30 (user: "just words flat as is not what we want - we should still be able to see the text effects"; every piece had ended in beat_bars / one line module): the fold keeps
# ONLY the modules whose hold measurably breaks legibility. The effect-rich ones (EFFECT_MODULES) are allowed again: they act in the word's FORM-IN and EXIT, and the hold stays legible by
# the same guards as before (flow_wake fully settled inside every hold, compose.legible, the hold-life legibility guard >= 85 % letters visible, the 9 % size floor).
HOLD_NOISY = ("depth_echo", "bold_thin_echo")                                                                      # echo copies stay BEHIND the word through the whole hold (ghosts, v014/v015 rejects; the 24 GB QC 2026-10-29): still folded
EFFECT_MODULES = ("tangent_spray", "voxel_orbit", "voxel_roll", "spectrum", "fly_into")        # allowed again 2026-10-30: streaks / sprays / rings / cubes / bars / tunnels land into the word and leave it; the hold is the settled word
MULTI_STYLE = ("scale_clash", "solid_hollow")                                                                         # one word in two styles (one part huge / filled, the rest tiny / outlined): "SIL" huge 3D + "ENT" thin outline, 24 GB QC 2026-10-29
EXTRUDED = ("hollow_tubes", "topo_tower", "hinge_unfold", "xray_unfold", "candy_orbit", "depth_tunnel")               # 3D / texture-first slabs: smeared gold slabs, mangled 3D words (round-3 QC 2026-10-29); legible only at one pose
# 2026-11-01 (user, R18/R19: "i dont like the text being filled with texture like this ... not this bitsy texture fill", "throughout any generation"): NO letter is ever a speckle / dot cloud, in the form-in, the hold or
# the exit. The `particles` word module (a 3000-dot cloud) and the `particles` io style (marks that land on the outline) are retired from every choosable set; a piece / reply that still names one is swapped
# at plan time (owner.fix_speckle_forms, called from text_guard.fix_plan). A free-floating accent that is not letter-shaped is not this; a dotted FONT is a font choice and stays.
SPECKLE_MODULES = ("particles",)
# 2026-11-02 (user, review of a 16:9 piece where SKIP used `outflow`: "I'm not a fan of these echos coming in from one side and then going down. I'd remove this type of echo from the library of choices"): the
# `outflow` word module (outline rings stream in from the entry side, flow through the word as a growing echo, leave down the exit side) is retired the same way: the module file and its registration stay (the
# library is intact), but no choosable set offers it and a piece / reply that names it is swapped at plan time (owner.fix_speckle_forms). NOT the liked time echoes (x08 / echo effects / flow_wake echo cap).
RETIRED_MODULES = ("outflow",)
BANNED_MODULES = SPECKLE_MODULES + RETIRED_MODULES      # never drawn, picked or kept
SPECKLE_IO = ("particles", "blobs")                                             # 2026-11-01 R20: a `blobs` exit broke ASCEND into falling dots -> banned too
SPECKLE_MODULE_FALLBACK = ("lanes", "slats", "pipework", "gradient_fade")        # strand / slab modules that draw the word with lines or bars and hold it solid: what a banned (speckle / retired) word becomes
SPECKLE_IO_FALLBACK = ("lines", "wipe", "flicker")                              # strand lines, a wipe, a flicker: none leaves a speckled glyph
AVOID_MODULES = HOLD_NOISY + MULTI_STYLE + EXTRUDED + BANNED_MODULES                                                                   # never kept: folded into the piece's reading module(s)
HARD_READ_FONTS = ("Monofett", "RubikBrokenFax", "RubikGlitchPop", "RubikLines", "LondrinaSketch", "BungeeOutline", "BungeeHairline")     # TURN read "TURO" in Monofett; outline / hairline / broken faces smear on footage
HARD_READ_MIN_LETTERS = 6     # a hard-to-read face only on a word of at least this many letters (a long word gives context), never on a short one
READABLE_EXPRESSIVE = ("BitcountSingle", "Silkscreen", "Barrio", "Fascinate", "Honk", "Bitcount", "AlienBlock")       # characterful and plainly legible: what a hard-to-read face is swapped to
MAIN_HOLD_RISKY = HARD_READ_FONTS + ("Doto", "Codystar", "FolditThin", "Interlope", "NixieOne", "FacadeVF")     # round-3 QC 2026-10-29: dotted (Doto), thin, outline-only / triple, very tall-thin condensed (ESCAPE read "AFE")
RISKY_CAP_MIN = 0.12          # a risky face is kept for a word's main hold only when its cap height is known and at least this share of the frame height (the contrast check is the compose legibility guard)
HEAVY_READERS = ("Anybody", "BadeenDisplay", "EricaOne", "Ultra", "PassionOne-Black", "BBHBartle", "Blackout-Midnight", "Tourney")     # heavy faces read best on a single-word hold (v009 r02: heavy white type)
HEAVY_READER_FILES = ("Anybody[wdth,wght].ttf", "BadeenDisplay-Regular.ttf", "EricaOne-Regular.ttf", "Ultra-Regular.ttf", "PassionOne-Black.ttf", "BBHBartle-Regular.ttf", "Blackout_Midnight.ttf", "Tourney[wdth,wght].ttf")
# 2026-10-30 QC v026 (r02 7.5-9 s + 13-15 s, r03 11-12.5 s: pixel-glyph words with ray extrusions / on a 4-row grid were unreadable). Pixel / dot-matrix / bitmap faces are TEXTURE faces: they read
# only on a short word, big, with a plain module. Jacquard 12 / 24 and Jacquarda Bastarda 9 are bitmap blackletters (role `blackletter` in liked.json) and count as pixel faces.
PIXEL_DOT_FACES = ("Bitcount", "BitcountSingle", "Bytesized", "Doto", "GeistPixel", "PaletteMosaic", "Silkscreen", "Jacquard12", "Jacquard24", "JacquardaBastarda9", "Codystar", "Monofett")
PIXEL_SAFE_MODULES = ("beat_bars", "slats", "lanes")      # flat bars / slats / lanes: the glyphs stay whole on the hold; every other module (gradient_fade, tangent_spray, spectrum, voxel_*, particles, outflow, fly_into, staircase, ...) extrudes, sprays or rays them
PIXEL_MAX_LETTERS = 6         # a pixel / dot face only on a word (phrase) of at most this many letters
PIXEL_CAP_MIN = 0.12          # ... and only when its fitted cap height is at least this share of the frame height (a 4-row grid / a long phrase is far below it)
PIXEL_FALLBACK = ("Barrio", "Fascinate", "Honk", "AlienBlock")      # characterful, plainly legible, not pixel: what the piece's one expressive face becomes
MAX_WF = 0.90                 # the word box never wider than this share of the frame (4-5 % margin each side: AWAKE bled off both edges)
YAW_WF_LOSS = 0.004           # a turned word is wider on its near side: the box shrinks by this much of the frame per degree of |yaw| (SLOW was cropped at the right edge)
EDGE_MARGIN = 0.04            # the word box (centre + dx/screen_offset, width wf) stays this far from every frame edge
MAX_MODULES, MAX_FONTS, MAX_FORMS = 4, 3, 2   # 2026-11-01 user: modules 2 -> 4 (varied word modules per piece)
CAP_FLOOR = 0.09              # cap height / frame height: DESIGN_GUIDE_open_model.md section 1 rule 2 ("9 % is the absolute floor")
CAP_AIM = 0.10                # what a word below the floor is fitted to (a little above it, so rounding cannot put it back under)
FAMILY = {"strand": ("lanes", "pipework", "circuit", "voxel_orbit", "voxel_roll"), "slab": ("beat_bars", "gradient_fade", "scale_clash", "slats", "solid_hollow", "staircase")}


def _hist(vals):
    """{value: count} and the first index of each value."""
    cnt, first = {}, {}
    for i, v in enumerate(vals):
        cnt[v] = cnt.get(v, 0) + 1
        first.setdefault(v, i)
    return cnt, first


# ================================================================================================== 2. one module system (+ clear holds: no echo / spray modules)
def one_module_system(words: list, fixes: list, protected=(), seed: int = 0) -> int:
    """words: design words (dicts with `module`). AVOID_MODULES (echo / spray, two-style, 3D texture-first) are replaced by a kept module; then at most MAX_MODULES distinct modules remain (reading-first first, then the most used,
    then the first seen; with no usable module at all the fallback is a seeded rotation, never `lanes` first). A replaced word takes the kept module of its own family (strand / slab) if there is one, else the first kept. -> number of words changed."""
    mods = [w.get("module") for w in words]
    if not words or any(m is None for m in mods):
        return 0
    cnt, first = _hist(mods)
    clean = [m for m in cnt if m not in AVOID_MODULES]
    rank = sorted(clean, key=lambda m: (-cnt[m], first[m]))               # the model's own choice decides (most used, then first written): READING_FIRST no longer re-ranks lanes first
    if not rank:                                                         # every module the model wrote is on the avoid list: the readable module nearest its family, else a seeded rotation
        from tools.fusion import openings as OP
        fam = next((f for f, ms in FAMILY.items() if any(m in ms for m in cnt)), None)
        rank = [next((m for m in FAMILY.get(fam, ()) if m in READING_FIRST + ("pipework", "circuit")), OP.default_module(0, seed or len(words)))]
    keep = rank[:MAX_MODULES]
    changed = 0
    for i, w in enumerate(words):
        m = w.get("module")
        if m in keep or i in protected:
            continue
        fam = next((f for f, ms in FAMILY.items() if m in ms), None)
        new = next((k for k in keep if fam and k in FAMILY[fam]), keep[0])
        why = ("it draws echo copies / streaks on the hold" if m in HOLD_NOISY else "its letters are a speckle of dots (never a letter fill)" if m in SPECKLE_MODULES else "it is retired (rings streaming in from one side and out the other)" if m in RETIRED_MODULES else "it puts one word in two styles" if m in MULTI_STYLE else "a 3D / texture-first module smears on footage"
               if m in EXTRUDED else f"one type system per piece: at most {MAX_MODULES} modules")
        fixes.append(f"type.words[{i}].module {m} -> {new} ({why}; kept {', '.join(keep)})")
        w["module"] = new
        changed += 1
    return changed


# ================================================================================================== 2. at most three faces
def one_font_system(words: list, fixes: list, protected=()) -> int:
    """at most MAX_FONTS distinct faces in the piece, one of them expressive. keep = the expressive face of the word with the longest hold, the faces of the protected (window)
    words, then the most used faces (distinct roles first). Every other word takes the kept face that differs from its neighbours and is used least; its axes go with the old face."""
    from mograph import fonts_liked as fl
    fonts = [w.get("font") for w in words]
    if not words or any(not f for f in fonts):
        return 0
    cnt, first = _hist(fonts)
    if len(cnt) <= MAX_FONTS:
        return 0
    role = {f: fl.role_of(f) for f in cnt}
    keep: list = []

    def add(f):
        if f is not None and f not in keep and len(keep) < MAX_FONTS:
            keep.append(f)
    for i in sorted(protected):
        if 0 <= i < len(words):
            add(words[i]["font"])
    hero = max(range(len(words)), key=lambda i: (float(words[i].get("hold_s", words[i].get("_h0", 0.0))), -i))
    exp = [f for f in cnt if role[f] in EXPRESSIVE]
    if exp and not any(role[f] in EXPRESSIVE for f in keep):
        add(words[hero]["font"] if role[words[hero]["font"]] in EXPRESSIVE else max(exp, key=lambda f: (cnt[f], -first[f])))
    while len(keep) < MAX_FONTS:
        rest = [f for f in cnt if f not in keep]
        if not rest:
            break
        add(min(rest, key=lambda f: (role[f] in {role[k] for k in keep}, -cnt[f], first[f])))
    use = {k: sum(1 for f in fonts if f == k) for k in keep}
    changed = 0
    for i, w in enumerate(words):
        f = w["font"]
        if f in keep or i in protected:
            continue
        prev = words[i - 1]["font"] if i else None
        nxt = fonts[i + 1] if i + 1 < len(words) else None

        def cost(k):
            return ((2 if k == prev else 0) + (1 if k == nxt else 0) + (0.5 if role[k] in EXPRESSIVE and i != hero else 0), use[k], keep.index(k))
        new = min(keep, key=cost)
        fixes.append(f"type.words[{i}].font {f} -> {new} (one type system: at most {MAX_FONTS} faces per piece, one expressive; kept {', '.join(keep)})")
        if w.pop("axes", None) is not None:
            fixes.append(f"type.words[{i}].axes dropped (they belonged to {f})")
        w["font"] = new
        use[new] += 1
        changed += 1
    return changed


# ================================================================================================== 2. the same treatment family across the words
def one_treatment(words: list, fixes: list, protected=()) -> int:
    """matte treatments (front | window | fill | wipe | glow): one kind per piece (the most used, ties: the first); the others are dropped. io styles form_in / form_out: at most MAX_FORMS
    different ones each; a word with another style takes the most used kept one. Protected (window) words keep theirs. -> number of fields changed."""
    changed = 0
    ms = [w.get("matte") for w in words if w.get("matte")]
    if len(set(ms)) > 1:
        cnt, first = _hist(ms)
        keep = max(cnt, key=lambda m: (cnt[m], -first[m]))
        for i, w in enumerate(words):
            if w.get("matte") and w["matte"] != keep and i not in protected:
                fixes.append(f"type.words[{i}].matte {w['matte']!r} dropped (one treatment family per piece: {keep!r})")
                w.pop("matte")
                changed += 1
    for k in ("form_in", "form_out"):
        vals = [w.get(k) for w in words if w.get(k)]
        cnt, first = _hist(vals)
        if len(cnt) <= MAX_FORMS:
            continue
        keep = sorted(cnt, key=lambda v: (-cnt[v], first[v]))[:MAX_FORMS]
        for i, w in enumerate(words):
            if w.get(k) and w[k] not in keep and i not in protected:
                fixes.append(f"type.words[{i}].{k} {w[k]!r} -> {keep[0]!r} (at most {MAX_FORMS} io styles per piece; effects live in the in / out phases only)")
                w[k] = keep[0]
                changed += 1
    return changed


# ================================================================================================== 3a. legibility screen: hard-to-read faces, safe width
def legibility_screen(words: list, fixes: list, protected=(), keep_font=()) -> int:
    """a hard-to-read face (HARD_READ_FONTS: Monofett read 'TURO' for TURN) is allowed only on a word of at least HARD_READ_MIN_LETTERS letters; a shorter word takes a readable expressive face (one the
    piece already uses first, so the face count does not grow), its axes dropped. The word box is at most MAX_WF of the frame wide (bleeding off both edges is never a style). Protected (window /
    LTX-owned) words are left alone. -> number of fields changed."""
    from mograph import fonts_liked as fl
    changed = 0
    used = [w.get("font") for w in words if w.get("font")]
    for i, w in enumerate(words):
        if i in protected:
            continue
        e = fl.lookup(w.get("font")) if w.get("font") else None
        n_letters = len(re.sub(r"[^A-Za-z0-9]", "", str(w.get("text", ""))))
        cap_i, _ = cap_of(w)
        bad = None
        if e is not None and e["key"] in HARD_READ_FONTS and n_letters < HARD_READ_MIN_LETTERS:
            bad = f"{e['key']} is hard to read on a {n_letters}-letter word (needs >= {HARD_READ_MIN_LETTERS} letters)"
        elif e is not None and e["key"] in MAIN_HOLD_RISKY and ((cap_i is None and e["key"] not in HARD_READ_FONTS) or (cap_i is not None and cap_i < RISKY_CAP_MIN)):      # a hard-to-read face on a long word with no box of its own keeps the old rule (letters only)
            bad = (f"{e['key']} (dotted / thin / outline-only / very tall-thin) needs a cap height of {RISKY_CAP_MIN:.0%} to be read on a main hold" +
                   (f", this word has ~{cap_i:.0%}" if cap_i is not None else ", this word has no box of its own to show it"))
        if bad and i in keep_font:                                             # the user chose this face for this line (font_pins): it stays, the problem is only reported
            fixes.append(f"type.words[{i}].font {w['font']} kept although {bad}: the user's own choice for this line")
            bad = None
        if bad:
            others = [fl.lookup(x.get("font")) for j, x in enumerate(words) if j != i and x.get("font")]
            expr_left = any(o and o["role"] in EXPRESSIVE for o in others)
            used_ok = [f for f in dict.fromkeys(used) if (fl.lookup(f) or {}).get("key") not in MAIN_HOLD_RISKY]
            heavy = [f for f in used_ok if (fl.lookup(f) or {}).get("role") in ("heavy", "tall", "text") and f != w["font"]]
            if e["role"] in EXPRESSIVE and not expr_left:                      # keep the piece's one characterful face, but a readable one
                new = next((f for f in used_ok if (fl.lookup(f) or {}).get("key") in READABLE_EXPRESSIVE and f != w["font"]), None) or                     next((fl.BY_KEY[k]["file"] for k in READABLE_EXPRESSIVE if k in fl.BY_KEY), None)
            else:                                                              # a single-word hold reads best in a heavy face
                new = (heavy[0] if heavy else next((fl.BY_KEY[k]["file"] for k in HEAVY_READERS if k in fl.BY_KEY and fl.BY_KEY[k]["file"] != w["font"]), None))
            faces_ = [x.get("font") for j, x in enumerate(words) if j != i]
            if new and len(set(faces_) | {new}) < min(3, len(words)):          # the swap must not cost the piece its third face: a heavy reader nobody uses yet
                spare = next((fl.BY_KEY[k]["file"] for k in HEAVY_READERS if k in fl.BY_KEY and fl.BY_KEY[k]["file"] not in faces_ and fl.BY_KEY[k]["file"] != w["font"]), None)
                new = spare or new
            if new:
                fixes.append(f"type.words[{i}].font {w['font']} -> {new}: {bad}")
                if w.pop("axes", None) is not None:
                    fixes.append(f"type.words[{i}].axes dropped (they belonged to {e['key']})")
                w["font"] = new
                changed += 1
        pl = w.get("placement")
        if isinstance(pl, dict) and isinstance(pl.get("wf"), (int, float)):
            yaw_ = abs(float(pl["yaw"])) if isinstance(pl.get("yaw"), (int, float)) else 0.0
            cap_wf = round(MAX_WF - YAW_WF_LOSS * yaw_, 3)
            if pl["wf"] > cap_wf + 1e-9:
                fixes.append(f"type.words[{i}].placement.wf {pl['wf']:g} -> {cap_wf:g}: the word stays inside the safe area (a box wider than {MAX_WF:.0%}, less {YAW_WF_LOSS:.1%} per degree of yaw, bleeds off the frame)")
                pl["wf"] = cap_wf
                changed += 1
            room = round(0.5 - EDGE_MARGIN - float(pl["wf"]) / 2.0, 3)         # how far the box centre may move from the frame centre
            for key, holder in (("dx", pl), ("so_x", None)):
                v = pl.get("dx") if key == "dx" else ((w.get("screen_offset") or [None])[0])
                if isinstance(v, (int, float)) and abs(v) > room + 1e-9:
                    nv = round(max(min(v, room), -room), 3)
                    fixes.append(f"type.words[{i}].{'placement.dx' if key == 'dx' else 'screen_offset.x'} {v:g} -> {nv:g}: the word box keeps {EDGE_MARGIN:.0%} of the frame from the edge")
                    if key == "dx":
                        pl["dx"] = nv
                    else:
                        w["screen_offset"] = [nv, w["screen_offset"][1]]
                    changed += 1
    return changed


# ================================================================================================== 3b. pixel / dot faces: short words, plain modules, big
def pixel_problem(w: dict, cap=None) -> str | None:
    """why a pixel / dot / bitmap face cannot carry this word (None = it can): it is drawn by an extrusion / ray / spray module, the text is longer than PIXEL_MAX_LETTERS letters, or the fitted cap
    height (`cap`, frame fraction; None = unknown) is under PIXEL_CAP_MIN. LTX-owned words keep the graphic their footage was made from."""
    from mograph import fonts_liked as fl
    e = fl.lookup(w.get("font")) if w.get("font") else None
    if e is None or e["key"] not in PIXEL_DOT_FACES or w.get("render") in ("ltx", "handoff"):
        return None
    n = len(re.sub(r"[^A-Za-z0-9]", "", str(w.get("text", ""))))
    mod = w.get("module")
    if mod and mod not in PIXEL_SAFE_MODULES:
        return f"{e['key']} (pixel / dot face) is never drawn by the extrusion / ray / spray module {mod} (only {', '.join(PIXEL_SAFE_MODULES)})"
    if n > PIXEL_MAX_LETTERS:
        return f"{e['key']} (pixel / dot face) only on words of at most {PIXEL_MAX_LETTERS} letters, this one has {n}"
    if cap is not None and cap < PIXEL_CAP_MIN:
        return f"{e['key']} (pixel / dot face) needs a cap height of {PIXEL_CAP_MIN:.0%} to read, this word is fitted to ~{cap:.0%}"
    return None


def pixel_swap(words: list, i: int, fixes: list, cap=None, protected=(), allowed=None) -> bool:
    """swap word i's face when pixel_problem says it cannot read: a non-pixel face the piece already uses (a heavy reader first, so the face count does not grow), else the piece's one characterful
    face becomes a legible expressive non-pixel one (PIXEL_FALLBACK), else a heavy reader. Axes of the old face dropped. -> True when the face changed."""
    from mograph import fonts_liked as fl
    w = words[i]
    if i in protected:
        return False
    why = pixel_problem(w, cap)
    if not why:
        return False
    key_of = lambda f: (fl.lookup(f) or {}).get("key")           # noqa: E731
    role_of = lambda f: (fl.lookup(f) or {}).get("role")         # noqa: E731
    bad = PIXEL_DOT_FACES + MAIN_HOLD_RISKY
    used = [x.get("font") for j, x in enumerate(words) if j != i and x.get("font")]
    pool = [f for f in dict.fromkeys(used) if key_of(f) not in bad and f != w.get("font")]
    extra = [f for f in (allowed or []) if f not in pool and f != w.get("font") and key_of(f) not in bad]          # the piece system's shortlist (random_system / piece_system): stays inside it when it can
    expr_left = any(role_of(f) in EXPRESSIVE for f in used if key_of(f) not in PIXEL_DOT_FACES)
    heavy = [f for f in pool + extra if role_of(f) in ("heavy", "tall", "text")]
    expr = [f for f in pool + extra if role_of(f) in EXPRESSIVE]
    nxt = None
    if expr_left:
        nxt = (heavy or pool or extra or [None])[0]
    else:
        nxt = (expr or [None])[0] or next((fl.BY_KEY[k]["file"] for k in PIXEL_FALLBACK if k in fl.BY_KEY), None)
    nxt = nxt or next((fl.BY_KEY[k]["file"] for k in HEAVY_READERS if k in fl.BY_KEY and fl.BY_KEY[k]["file"] != w.get("font")), None)
    if not nxt:
        return False
    fixes.append(f"type.words[{i}].font {w.get('font')} -> {nxt}: {why}")
    if w.pop("axes", None) is not None:
        fixes.append(f"type.words[{i}].axes dropped (they belonged to {key_of(w.get('font'))})")
    w["font"] = nxt
    return True


def pixel_guard(words: list, fixes: list, protected=(), caps: dict | None = None, keep_font=()) -> int:
    """pixel_swap on every word (the design-level pass: caps = {index: fitted cap fraction} when known, else the estimate of the word's own box). -> number of faces swapped."""
    n = 0
    for i, w in enumerate(words):
        if i in keep_font:                                                     # the user chose this face (font_pins): the MODULE gives way instead (a pixel / dot face needs a flat module), the face stays
            why = pixel_problem(w, None)
            if why and w.get("module") not in PIXEL_SAFE_MODULES:
                fixes.append(f"type.words[{i}].module {w.get('module')} -> {PIXEL_SAFE_MODULES[0]}: the user's face {w.get('font')} is a pixel / dot face (flat modules only)")
                w["module"] = PIXEL_SAFE_MODULES[0]
            elif why:
                fixes.append(f"type.words[{i}].font {w.get('font')} kept although {why}: the user's own choice for this line")
            continue
        cap = (caps or {}).get(i)
        if cap is None:
            cap, _ = cap_of(w)
        n += 1 if pixel_swap(words, i, fixes, cap, protected) else 0
    return n


# ================================================================================================== 3. size floor
def cap_of(w: dict, aspect=0.5625):
    """(cap height / frame height, 'estimated') of a design word from its OWN box (placement.wf and hf) and the real glyph widths of its face; (None, None) when the word has no box of its own (the engine's
    default sizing is not estimated here) or a multi-word text without lines is judged by its words stacked (the engine stacks the words of a phrase)."""
    pl = w.get("placement") or {}
    if not isinstance(pl.get("wf"), (int, float)) or not isinstance(pl.get("hf"), (int, float)):
        return None, None
    v = w
    if not w.get("lines") and len(str(w.get("text", "")).split()) > 1:
        v = dict(w, lines=str(w["text"]).split())
    est_h, est_w = DO.cap_estimates(v, aspect)
    return (est_w if est_h is None else min(est_h, est_w)), "estimated"


def size_floor(words: list, fixes: list, caps: dict | None = None, aspect=0.5625, skip=()) -> int:
    """every word has a cap height of at least CAP_FLOOR of the frame. `caps` {word index: measured cap fraction} wins over the estimate. A word below the floor keeps its lines when
    its own box can be widened to the floor (wf up to 1.0, hf taller), else it is stacked into 2-3 lines (the even splits of design_options._splits). `skip` = word indices left alone (window
    words are fitted to 30 %; an LTX-owned word keeps the graphic its control pass was made from). -> number of words changed."""
    changed = 0
    for i, w in enumerate(words):
        if i in skip:
            continue
        cap, _ = (float(caps[i]), "measured") if caps and caps.get(i) is not None else cap_of(w, aspect)
        if cap is None or cap + 1e-9 >= CAP_FLOOR:
            continue
        text = str(w.get("text", ""))
        own = [str(x) for x in (w.get("lines") or [])]
        pl = w.setdefault("placement", {})
        wf0, hf0 = float(pl.get("wf", 0.8)), float(pl.get("hf", 0.5))
        best = None
        for ls in DO._splits(text, own):
            rs = [DO.line_width_per_cap(w.get("font"), w.get("axes"), x) if w.get("font") else None for x in ls]
            rs = [r if r is not None else len(x) * DO.CHAR_W for r, x in zip(rs, ls)]
            ratio = max(rs)
            c = min(0.95 / len(ls), MAX_WF * aspect / ratio)        # the box never goes past the safe width
            if best is None or c > best[0]:
                best = (c, ls, ratio)
            if c >= CAP_AIM:
                best = (c, ls, ratio)
                break
        c, ls, ratio = best
        pl["wf"] = round(min(MAX_WF, max(wf0, (CAP_AIM + 0.005) * ratio / aspect + 0.02)), 3)
        pl["hf"] = round(min(1.0, max(hf0, (CAP_AIM + 0.005) * len(ls) + 0.01)), 3)
        stacked = [x.replace(" ", "") for x in ls] != [x.replace(" ", "") for x in (own or [text])] or (len(ls) > 1 and len(ls) != len(own or [text]))
        if stacked:
            w["lines"] = ls
        fixes.append(f"type.words[{i}].placement cap height ~{cap:.0%} < the {CAP_FLOOR:.0%} floor: " + (f"stacked into {len(ls)} lines {ls}, " if stacked else "") +
                     f"wf {wf0:g} -> {pl['wf']:g}, hf {hf0:g} -> {pl['hf']:g} (~{min(c, CAP_AIM + 0.005):.0%} at best)")
        changed += 1
    return changed


# ================================================================================================== all type rules
def hero_face(words: list, fixes: list, protected=()) -> int:
    """user 2026-10-30 ("always use our favourited fonts ... prefer the expressive liked faces; plain or neutral only as the contrast line, never as the hero word"): the word with the longest hold
    (the hero) wears a characterful liked face (role pixel / ornamental / decon / blackletter) when the piece has one; if it wears a reader face it swaps faces with the expressive word that
    holds shortest. Window words (protected: they need a fat face) are never touched. -> number of swaps."""
    from mograph import fonts_liked as fl
    expr = ("pixel", "ornamental", "decon", "blackletter")
    if len(words) < 2:
        return 0
    hero = max(range(len(words)), key=lambda i: (float(words[i].get("hold_s", words[i].get("_h0", 0.0))), len(str(words[i].get("text", ""))), -i))
    if hero in protected or not words[hero].get("font"):
        return 0
    try:
        if fl.role_of(words[hero]["font"]) in expr:
            return 0
        cand = [i for i, w in enumerate(words) if i != hero and i not in protected and w.get("font") and fl.role_of(w["font"]) in expr]
    except KeyError:
        return 0
    if not cand:
        return 0
    j = min(cand, key=lambda i: (float(words[i].get("hold_s", words[i].get("_h0", 0.0))), i))
    a, b = words[hero], words[j]
    fixes.append(f"type.words[{hero}].font {a['font']} <-> type.words[{j}].font {b['font']}: the hero word wears the characterful face, the reader face is the contrast line")
    a["font"], b["font"] = b["font"], a["font"]
    a["axes"], b["axes"] = b.get("axes"), a.get("axes")
    return 1


def enforce_type(words: list, fixes: list, protected=(), caps: dict | None = None, font_protected=()) -> dict:
    """the type-level defaults on design words (in place): one module system, <= 3 faces, one treatment family, the legibility screen, the size floor. `protected` = indices of window words (their face
    / matte are the window's). -> counts per rule."""
    return dict(modules=one_module_system(words, fixes, protected), fonts=one_font_system(words, fixes, set(protected) | set(font_protected)) + hero_face(words, fixes, set(protected) | set(font_protected)), treatments=one_treatment(words, fixes, protected),
                legibility=legibility_screen(words, fixes, protected, font_protected), pixel=pixel_guard(words, fixes, protected, caps, font_protected), size=size_floor(words, fixes, caps, skip=protected))


# ================================================================================================== 1. + 4. compose level
def enforce_compose(piece: dict, holds) -> list[str]:
    """the compose-level defaults on a piece (in place): every flow_wake op's `settle` is the hold timeline (fully settled inside every hold, free in the in / out phases: a model-written
    chain kept its own settle, or none = the echo and the push ran during the hold); the legibility guard is never switched off. holds = [(hit s, release s)]. -> fix texts."""
    from tools.fusion import plan_rules as PR
    notes: list[str] = []
    comp = piece.get("compose") or {}
    settle = PR.settle_timeline(holds)
    chains = [("default", comp["default"])] if comp.get("default") else []
    chains += list((comp.get("shots") or {}).items())
    for name, ch in chains:
        for o in ch.get("ops", []):
            a = o.get("args") or {}
            if o.get("op") == "flow_wake" and a.get("settle") != settle:
                a["settle"] = copy.deepcopy(settle)
                o["args"] = a
                notes.append(f"compose.{name}: flow_wake {o.get('out', 'w')} settle forced to a full settle ({PR.SETTLE_HOLD:g}) inside every word hold (clear holds: no echo, no push)")
            if o.get("op") in ("sandwich", "depth_occlude") and a.get("legible") is False:
                a["legible"] = True
                notes.append(f"compose.{name}: {o['op']} legible=False -> True (the legibility guard stays on in a generated piece)")
    if not comp.get("type_glow"):                                       # 2026-10-30 (user: "I dont like the use of the glow behind all the text"): no glow behind the type unless asked for (compose.type_glow) or drawn by hold life (glow_breathe)
        for name, ch in chains:
            notes += drop_type_glow(ch.get("ops", []), name)
    if comp.get("legible") is False:
        comp.pop("legible")
        notes.append("compose.legible=False removed (the legibility guard stays on in a generated piece)")
    if notes:
        piece.setdefault("plan", {}).setdefault("clear_text", []).extend(notes)
    return notes


TYPE_MASKS = ("$wm", "$ta", "$tmat", "$type_matte", "$type_a", "$type_line", "$tm")


def _is_type_mask(v) -> bool:
    return isinstance(v, str) and (v in TYPE_MASKS or v.startswith("$type"))


def drop_type_glow(ops: list, name: str = "default") -> list[str]:
    """remove every add_glow whose mask is the TYPE (a model-written chain keeps adding `add_glow(mask=$wm)` behind every word); later references to its output are re-pointed at its input.
    Hold-life glows (`hl`) and glows of a subject outline (matte treatments) stay. In place -> fix texts."""
    notes = []
    for o in list(ops):
        a = o.get("args") or {}
        if o.get("op") != "add_glow" or o.get("hl") or not _is_type_mask(a.get("mask")):
            continue
        src, dst = a.get("img"), o.get("out", "image")
        ops.remove(o)
        if isinstance(src, str) and src.startswith("$") and src != "$" + dst:
            for later in ops[ops.index(ops[0]):]:
                la = later.get("args") or {}
                for k, v in list(la.items()):
                    if v == "$" + dst:
                        la[k] = src
        notes.append(f"compose.{name}: add_glow {dst} (mask {a.get('mask')}) removed (no glow behind the type unless asked for: pure type)")
    return notes


def count_repairs(fixes) -> dict:
    """how many logged fixes belong to each of the rules above (for the par tables)."""
    pats = dict(modules=r"\.module \w+ ->", fonts=r"\.font .* -> .*one type system", legibility=r"hard to read|safe area", treatments=r"\.(matte|form_in|form_out) ", size=r"floor",
                settle=r"settle forced", legible=r"legible", cuts=r"cuts: .* planned|window_other planned")
    return {k: sum(1 for f in fixes if re.search(p, str(f))) for k, p in pats.items()}
