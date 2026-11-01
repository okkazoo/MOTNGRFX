"""effects_menu: the NAMED EFFECTS the per-piece random system draws from and a request can FORCE (user 2026-10-30: "can we ask for specific effects - can this be a module for it to choose from").
Pure functions on the system dict, no ComfyUI / network.

    EFFECTS                                  name -> dict(kind, text): the menu
    norm_effects(v)                          request `effects` (list / "a, b" string) -> ([known names], [unknown names as written])
    draw_effects(seed, earlier, requested)   0-2 CONTRAST sections drawn (+ the requested ones) -> dict(contrast, requested, unknown)
    apply_effects(system, requested)         a copy of the system with the requested effects forced in + effects.active
    place_contrast(shots, system, seed)      the contrast names -> `type_matte` shot keys on the piece's shots (through plate_layers.shot_keys), one section each
    prompt_text(system)                      the sentence of the design / concept prompt that names the required and the drawn effects

The three CONTRAST names are rendered through the shot key `type_matte: <name>` (tools/fusion/plate_layers.py, built by the own-video worker; we only emit the key through
`plate_layers.shot_keys`). Each is a SECTION: one or two word holds (2-4 s), 0-2 per piece. The other names reuse what exists: `flat_ground_window` (word key window_ground plain,
design_options.plan_flat_window), `window_zoom` and any other cut kind (random_system cuts), `outline` (legible_color's outline support at some words / hits).

Request field: `effects` (a list of names or a comma separated string) in the piece request: read by plan_rules.effects_of (request `effects` key, or the same key inside the `owners` extras
object) and by concept.parse_effects (`{"concepts": [...], "effects": [...]}` in the concept step's `previous` string: no new graph input); empty = random as before. No app UI field yet.
"""
from __future__ import annotations

import copy
import random
import re

from tools.fusion import cut_kinds as CK
from tools.fusion import explore_effects as EE
from tools.fusion import random_system as RS

CONTRAST_EFFECTS = ("window_subject_front", "plate_in_subject", "plate_in_type")
EFFECTS = {
    "window_subject_front": dict(kind="type_matte", text="flat ground, the letters are windows onto the footage, the subject cut out in front of the type"),
    "plate_in_subject": dict(kind="type_matte", text="flat ground, the word behind the subject, the subject's silhouette filled with the clean plate"),
    "plate_in_type": dict(kind="type_matte", text="flat ground, the subject behind the type, the letters filled with the clean plate"),
    "flat_ground_window": dict(kind="word", text="a short word whose letters show the footage over a flat ground in a colour of the palette"),
    "window_zoom": dict(kind="cut", text="the camera scales into one letter of the holding word until the next shot fills the frame (a cut kind)"),
    "outline": dict(kind="outline", text="a thin outline in the opposite luminance stands the type out at some words / hits"),
    # EXPLORE LOOKS the user hearted (16_app/core_check/explore_masks x14 / x08 / x10 / x03 / x02), tools/fusion/explore_effects.py: sections of 2-4 s (outline_to_word = one word's form-in), 0-2 per piece
    **{n: dict(kind="explore", text=EE.SPECS[n]["text"]) for n in EE.NAMES},
    "soft_support": dict(kind="legibility", text="allow the soft shadow / local scrim behind the type where the ink colour and the outline cannot reach the contrast floor (OFF by default: pure type)"),
    # TYPE LAYOUTS (tools/fusion/stable_type.py): a request forces them onto at least half of the words (>= 2), the no-orphan rule and the size floor still apply
    "grid": dict(kind="layout", text="a word split MID-WORD into balanced rows of 2-4 letters that fill the frame (DREAMING = DR / EA / MI / NG)"),
    "column": dict(kind="layout", text="a word with one letter per row, stacked vertically (S / T / E / A / M)"),
    # HOLD LIFE (tools/fusion/hold_life.py, 16_app/HOLD_LIFE.md): the landed word stays centred and still but alive; requested names are copied into compose.hold_life by plan_rules._finish
    "letter_flicker": dict(kind="hold_life", text="single letters of the landed word blink off for 1-2 frames on sub-beats (words of 7+ letters)"),
    "glitch_slice": dict(kind="hold_life", text="2-4 frame bursts of horizontal slice offsets with an RGB split on sub-beats"),
    "pulse": dict(kind="hold_life", text="the landed word scales 1.00 to 1.03 and lifts in brightness on each beat"),
    "glow_breathe": dict(kind="hold_life", text="the type glow breathes with the beat envelope and the music's loudness"),
    "outline_shimmer": dict(kind="hold_life", text="the stand-out outline breathes in brightness (needs an outline in the piece)"),
    # CONTOUR SWEEP (tools/fusion/contour_sweep.py, 16_app/CONTOUR_SWEEP.md): the footage's contour lines draw on / off as brush strokes along the forms, swept in the direction of each transition
    "contour_sweep": dict(kind="contour_sweep", text="the subject's contour lines draw off along the form in the direction of every cut / word transition and draw back on from the same side, then settle on the edge before the next word lands"),
}
HOLD_LIFE_EFFECTS = ("letter_flicker", "glitch_slice", "pulse", "glow_breathe", "outline_shimmer")      # == hold_life.EFFECTS keys (tests/test_effects_menu.py)
ALIASES = {**EE.ALIASES, "zoom": "window_zoom", "letter_zoom": "window_zoom", "zoom_through": "window_zoom", "flat_window": "flat_ground_window", "flat_ground": "flat_ground_window",
           "inverse": "flat_ground_window", "inverse_type": "flat_ground_window", "stroke": "outline", "type_outline": "outline", "subject_front": "window_subject_front",
           "window_front": "window_subject_front"}
CONTRAST_COUNT_WEIGHTS = (0.35, 0.40, 0.25)         # 0 / 1 / 2 contrast sections per piece
MAX_FORCED_CUTS = 5
SECTION_MIN_S = 3.0                                 # a contrast section is placed in a shot at least this long (a section is 2-4 s)


def norm_effects(v) -> tuple[list, list]:
    """a request's `effects` (list, comma / semicolon / newline separated string, or None) -> ([known names in order, no duplicates], [unknown names as written]). Cut kinds (window_cut,
    orbit:push ...) are effects too; the spelling is forgiving (case, '-', ' ', aliases)."""
    if v is None:
        return [], []
    if isinstance(v, str):
        v = [x for x in re.split(r"[,;|\n]+", v) if x.strip()]
    names, unknown = [], []
    for x in v if isinstance(v, (list, tuple)) else []:
        raw = str(x).strip()
        k = re.sub(r"\s*:\s*", ":", raw.lower()).replace("-", "_").replace(" ", "_")
        k = ALIASES.get(k, k)
        if k in EFFECTS:
            nm = k
        else:
            ck = CK.normalize(k)
            nm = ck if ck in CK.ALL and ck not in ("crossfade", "hard") else None
        if nm is None:
            unknown.append(raw)
        elif nm not in names:
            names.append(nm)
    return names, unknown


def draw_effects(seed, earlier=(), requested=None) -> dict:
    """the piece's effects: `contrast` = 0-2 named contrast sections drawn (the previous piece's are drawn less) PLUS every requested contrast name (forced: a request for three gives
    three). -> dict(contrast [..], requested [..], unknown [..]). Tone independent (the concept step and the final system draw the same ones)."""
    req, unknown = norm_effects(requested)
    rng = random.Random(f"rsys:{int(seed)}:effects")
    used = {n for s in list(earlier or [])[-1:] if isinstance(s, dict) for n in ((s.get("effects") or {}).get("contrast") or [])}
    k = RS._wpick(rng, [0, 1, 2], CONTRAST_COUNT_WEIGHTS)
    drawn: list = []
    for _ in range(k):
        cand = [n for n in CONTRAST_EFFECTS if n not in drawn]
        drawn.append(RS._wpick(rng, cand, [0.4 if n in used else 1.0 for n in cand]))
    forced = [n for n in req if n in CONTRAST_EFFECTS]
    return dict(contrast=list(dict.fromkeys(forced + drawn)), requested=req, unknown=unknown, explore=EE.draw(seed, earlier, [n for n in req if n in EE.NAMES]))


def apply_effects(system: dict, requested=None) -> dict:
    """a copy of `system` with the requested effects forced in: contrast names -> effects.contrast; flat_ground_window -> inverse.plain_window; outline -> outline mode `words` (when it was
    none); a cut kind (window_zoom, window_cut, orbit:push ...) -> cuts.kinds (a window kind brings a heavy face to the shortlist). Unknown names stay in effects.unknown (reported, never
    fatal). effects.active lists every effect the piece uses (drawn + forced)."""
    s = copy.deepcopy(system)
    req, unknown = norm_effects(requested)
    eff = copy.deepcopy(s.get("effects") or {})
    eff.setdefault("contrast", [])
    eff["requested"] = list(dict.fromkeys(list(eff.get("requested") or []) + req))
    eff["unknown"] = list(dict.fromkeys(list(eff.get("unknown") or []) + unknown))
    for n in req:
        kd = EFFECTS.get(n, {}).get("kind", "cut")
        if kd == "explore":
            if n not in eff.setdefault("explore", []):
                eff["explore"].append(n)
            continue
        if kd in ("hold_life", "layout", "legibility", "contour_sweep"):
            continue                                   # recorded in effects.requested; plan_rules._finish copies it into compose.hold_life
        if kd == "type_matte":
            if n not in eff["contrast"]:
                eff["contrast"].append(n)
        elif n == "flat_ground_window":
            s["inverse"] = dict(s.get("inverse") or {}, plain_window=True)
        elif n == "outline":
            o = dict(s.get("outline") or {})
            if o.get("mode", "none") == "none":
                o = dict(mode="words", share=RS.OUTLINE_SHARE)
            s["outline"] = o
        else:
            c = copy.deepcopy(s.get("cuts") or dict(version=RS.VERSION, kinds=[], styles={}, order=[]))
            if n not in c["kinds"]:
                if len(c["kinds"]) >= MAX_FORCED_CUTS:
                    drop = next((k for k in reversed(c["kinds"]) if k not in req and not RS.robust(k)), None) or next((k for k in reversed(c["kinds"]) if k not in req), None)
                    if drop:
                        c["kinds"].remove(drop)
                        c["order"] = [k for k in c.get("order", []) if k != drop]
                        c.get("styles", {}).pop(drop, None)
                c["kinds"].append(n)
                c["order"] = list(c.get("order") or []) + [n]
                c.setdefault("styles", {})[n] = "chunks" if n.startswith("orbit:") else None
                c["forced"] = list(dict.fromkeys(list(c.get("forced") or []) + [n]))
            s["cuts"] = c
    if s.get("fonts") and any(CK.family_of(k) == "window" for k in (s.get("cuts") or {}).get("kinds", [])):
        from mograph import fonts_liked as fl
        if not any(fl.role_of(f) == "heavy" for f in s["fonts"]):
            from tools.fusion import piece_system as PS
            heavy = [e for e in PS._font_pool() if e["role"] == "heavy" and e["file"] not in s["fonts"]]
            if heavy:
                h = random.Random(f"rsys:{int(s.get('seed', 0))}:heavy").choice(heavy)
                s["fonts"] = list(s["fonts"][:-1]) + [h["file"]]
    active = list(eff["contrast"])
    if (s.get("inverse") or {}).get("plain_window"):
        active.append("flat_ground_window")
    if "window_zoom" in (s.get("cuts") or {}).get("kinds", []):
        active.append("window_zoom")
    if (s.get("outline") or {}).get("mode", "none") != "none":
        active.append("outline")
    active += list(eff.get("explore") or [])
    active += [n for n in eff["requested"] if n in HOLD_LIFE_EFFECTS or EFFECTS.get(n, {}).get("kind") in ("layout", "legibility", "contour_sweep")]
    eff["active"] = list(dict.fromkeys(active))
    s["effects"] = eff
    return s


def place_contrast(shots: list, system: dict | None, seed=0) -> list:
    """the system's contrast effects -> `type_matte: <name>` on the piece's shots (emitted through plate_layers.shot_keys, one section per shot, never on a shot that already has a
    type_matte). A shot qualifies when it is at least SECTION_MIN_S long and has a subject matte (`matte` true; a piece with no matte shot takes any long enough shot). Seeded, shots
    spread out (no two neighbours when avoidable). In place; -> [{name, shot}] placed."""
    from tools.fusion import plate_layers as PLT
    names = [n for n in ((system or {}).get("effects") or {}).get("contrast") or [] if n in CONTRAST_EFFECTS]
    if not names or not shots:
        return []

    def secs(s):
        try:
            return float(s.get("span") or s.get("frames") or 0) / 24.0
        except (TypeError, ValueError):
            return 0.0
    free = [i for i, s in enumerate(shots) if not s.get("type_matte") and secs(s) >= SECTION_MIN_S]
    cand = [i for i in free if shots[i].get("matte")] or free
    rng = random.Random(f"rsys:{int((system or {}).get('seed', seed) or seed)}:place")
    rng.shuffle(cand)
    placed, taken = [], []
    for n in names:
        pick = next((i for i in cand if i not in taken and all(abs(i - j) > 1 for j in taken)), None)
        if pick is None:
            pick = next((i for i in cand if i not in taken), None)
        if pick is None:
            break
        shots[pick].update(PLT.shot_keys({"type_matte": n}))
        taken.append(pick)
        placed.append(dict(name=n, shot=shots[pick].get("id", f"s{pick + 1}")))
    return placed


def prompt_text(system: dict | None) -> str:
    """the prompt sentence: the effects the user REQUIRED and the contrast sections drawn for this piece (empty without any)."""
    e = (system or {}).get("effects") or {}
    req, active = e.get("requested") or [], e.get("active") or []
    if not req and not active:
        return ""
    parts = []
    if req:
        parts.append("REQUIRED by the user (they stay in the piece, you place them on beats): " + "; ".join(f"{n} ({EFFECTS[n]['text']})" if n in EFFECTS else n for n in req))
    other = [n for n in active if n not in req]
    if other:
        parts.append("drawn for this piece: " + ", ".join(other))
    if e.get("explore"):
        parts.append("EXPLORE looks drawn for this piece (the code places each as a 2-4 s section on a shot; you do not write them): " + "; ".join(f"{n} ({EE.SPECS[n]['text']})" for n in e["explore"]))
    if e.get("contrast"):
        parts.append("each contrast look (" + ", ".join(CONTRAST_EFFECTS) + ") is ONE section of one or two word holds (2-4 s), written as `type_matte: <name>` on that shot of the shot plan; the code places it if you do not")
    return "- Named effects: " + ". ".join(parts) + "."
