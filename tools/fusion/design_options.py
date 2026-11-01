"""design_options - the "text as window" design choices for the open-model design step (user 2026-10-28): per-word `window_source` and per-cut
`cut` choices, the menu text for the model, and the deterministic validator / fixer that keeps them inside what reads.

    SCHEMA_WORD / SCHEMA_CUTS / schema_fragment(shot_ids)   text to splice into qwen_piece.schema_text (per-word key, compose.cuts block)
    JSON_SCHEMA, EXAMPLE, normalize_cuts(obj)               machine-readable schema + a structural parser of compose.cuts (no timing rules)
    MENU_TEXT, menu_text(shots, ana)                        when-to-use text for the model + a table of the real boundaries with their nearest beat
    validate(design, ctx, notes=None) -> (design, fixes)    the deterministic validator / fixer (rules below); every change is a fix string
    to_piece(piece, design, type_dir=None) -> [notes]       design -> piece JSON (compose.cuts, words' window_source) + compose ops (window_cut.apply_to_piece)

The choices (tools/fusion/window_cut.py is the compose side; blocks window_other / window_cut / window_swap in blocks.py):
    window_source  (per word)   "self" | "next" | "<shot id>": the letters show that shot during the word's hold, the word's own shot stays outside.
    cut            (per cut)    "crossfade" (default) | "hard" | "window_cut" | "window_swap" at a real shot boundary, with `word` (index), `exit`, `freeze_s`.

Rules (validate): window choices only at a real shot boundary, with a word whose hold spans or ends at the cut; the word's font is a fat / heavy /
blocky LIKED font (else the nearest liked fat font, else downgrade); cap height >= WINDOW_MIN_CAP of the frame height; a swap lands on a
downbeat / chord change / dropout of the music analysis; at most MAX_WINDOW_CUTS window cuts per piece (and MAX_WINDOW_WORDS window_source words);
the two shots differ enough in luma or colour for the window to read (compose.window_contrast). Downgrade = `hard` for a cut, `self` for a word.
"""
from __future__ import annotations

import copy
import json
import re

from tools.fusion import cut_kinds as CK
from tools.fusion import random_system as RS

WINDOW_KINDS = ("window_cut", "window_swap", "window_zoom")         # window_zoom (2026-10-30): the camera scales into ONE letter of the holding word until the next shot fills the frame
SLIDE_ON = False             # 2026-10-29: the `slide` style of the 3D cuts tore into strips with pale ghost legs in the box probe (cut_probe_box): not offered; a `style: slide` a model writes is dropped (logged).
                             # A piece author can still write it in the JSON, `validate` only strips it from model replies; set True (or ctx allow_slide) to let it through
DOLLY_ON = False             # 2026-10-29: the depth dolly rendered as a blotchy dark / gold shard collage for ~1.5 s in the 24 GB round (v006 r02, `qc_r02_cut3.png`) and takes 906 s for 90 frames on a CPU: off. A model that
                             # writes it gets the orbit / pivot the music asks for (logged); `validate(ctx allow_dolly=True)` and the cut_kinds / pivot_cut code still know it
CUT_KINDS = CK.ALL           # crossfade hard | window_cut window_swap window_other | depth_pivot orbit:push|pullback|fly depth_dolly:down|up|left|right | pan:down|up|left|right
SHOT_3D = CK.SHOT_3D
SPECIAL = CK.WINDOW + CK.SHOT_3D + CK.PANS       # the kinds a piece should not repeat (a plain hard cut / crossfade may)
MIN_OUT_S = 0.4              # a pan / word-level compile needs a word whose hold ends within this much of the cut (before it) ...
MAX_PAN_GAP_S = 1.6          # ... and whose successor starts within this much after it
EXITS = {"window_cut": ("grow", "io"), "window_swap": ("io", "fade", "shrink"), "window_zoom": ("zoom",)}
DEFAULT_EXIT = {"window_cut": "grow", "window_swap": "fade", "window_zoom": "zoom"}
WINDOW_MIN_CAP = 0.26        # cap height / frame height for any word that is a window (LEGIBILITY_READOUT: reads at 27-37 %, rule 1 asks >= 25 %); was 0.30 until 2026-10-30 (a 6-letter word read vertically reaches 26 %)
CAP_TOL = 0.92               # "about 30 %": a measured / estimated cap height >= WINDOW_MIN_CAP x CAP_TOL passes (3 stacked letters reach 28.3 %: P10)
CAP_PER_HF = 1.0             # estimate when only the placement is known: cap height = placement.hf / lines (height limit) ...
CHAR_W = 0.55                # ... and wf x aspect / (longest line in letters x CHAR_W) (width limit); the smaller of the two
MAX_WINDOW_CUTS = 2          # per piece (window_cut / window_swap / window_zoom)
MAX_WINDOW_WORDS = 2         # window_source words per piece
ZOOM_S, ZOOM_MIN_S, ZOOM_MAX_S = 0.9, 0.5, 1.8     # window_zoom: the camera move into the letter takes this long (s); at least 0.5 s = 12 frames at 24 fps so it can be SEEN (window_cut.ZOOM_*)
BEAT_TOL = 0.15              # s: a swap must land this close to a downbeat / chord change / dropout start
MIN_P1, MIN_P2 = 0.5, 0.4    # s the word must hold before / after the swap (phase 1 B-in-letters, phase 2 A-in-letters)
MIN_LUMA_GAP, MIN_COLOR_GAP = 0.12, 0.18     # window reads when the shots differ by this much in mean luma (0..1) or in mean RGB distance
FAT_ROLES = ("heavy",)
FAT_IF_WEIGHT = {"BigShouldersDisplay": 800.0}      # liked tall faces that are fat only at a heavy weight (axes.wght)
NEAREST_FAT = {                                     # liked font role -> fat liked fonts to swap to, nearest first
    "tall": ("Anybody", "PassionOne-Black", "EricaOne"), "text": ("Anybody", "Ultra"), "ornamental": ("Ultra", "BadeenDisplay", "BBHBartle"),
    "decon": ("BBHBartle", "Ultra", "Anybody"), "blackletter": ("Ultra", "BadeenDisplay"), "pixel": ("BBHBartle", "Tourney", "Anybody"),
    "fine": ("Anybody", "PassionOne-Black"), "mono": ("Anybody", "Tourney"), "heavy": ("Anybody", "Ultra")}


# ================================================================================================== schema + menu
SCHEMA_WORD = ('"window_source": "<optional: self | next | a shot id from the SHOTS list: the letters of this word show that shot while its own '
               'shot stays outside; heavy face, cap height >= 30 % of the frame>"')
SCHEMA_CUTS = ('"cuts": [                   // optional; at most one entry per boundary between two consecutive shots; at most 2 window cuts; EVERY boundary of a multi-shot piece should get its own entry and the kinds differ\n'
               '   {"after": "<id of the shot before the boundary>",\n'
               '    "cut": "crossfade | hard | window_cut | window_swap | window_other | window_zoom | depth_pivot | orbit:push | orbit:pullback | orbit:fly | pan:down | pan:up | pan:left | pan:right | circle_grid | depth_lines",\n'
               '    "style": "<3D cuts only, optional: chunks | smooth | slide; the default is chunks for depth_pivot AND orbit:push / pullback / fly (smooth = one flat card, only when asked for); slide needs a dir>", "dir": "<slide only: left | right | up | down>",\n'
               '    "word": <index of a word whose hold spans or ends at that boundary; window_* and pan:* only>,\n'
               '    "exit": "<window_cut: grow | io;  window_swap: io | fade | shrink;  window_zoom: zoom>", "freeze_s": <0..0.5, optional: freeze frame on the swap>,\n'
               '    "why": "<one sentence: why this kind and direction at THIS boundary>"}\n'
               '  ]')

def schema_cuts(system=None) -> str:
    """SCHEMA_CUTS for a piece: with a drawn system (random_system.draw_cuts) the `cut` line lists ONLY the drawn kinds (and their styles), else the full vocabulary."""
    kinds = RS.kinds_of(system)
    if not kinds:
        return SCHEMA_CUTS
    full = "crossfade | hard | window_cut | window_swap | window_other | window_zoom | depth_pivot | orbit:push | orbit:pullback | orbit:fly | pan:down | pan:up | pan:left | pan:right | circle_grid | depth_lines"
    one = SCHEMA_CUTS.replace(full, " | ".join(dict.fromkeys(list(kinds) + ["hard"])))
    one = one.replace("chunks | smooth | slide; the default is chunks for depth_pivot AND orbit:push / pullback / fly (smooth = one flat card, only when asked for); slide needs a dir", "the style drawn for the kind (see the system); slide needs a dir")
    return one.replace("at most 2 window cuts;", "at most 2 window cuts; ONLY the kinds of YOUR SYSTEM may be used;")


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "cuts": {"type": "array", "items": {
            "type": "object", "required": ["after", "cut"],
            "properties": {"after": {"type": "string"}, "cut": {"enum": list(CUT_KINDS)}, "why": {"type": "string"}, "word": {"type": "integer", "minimum": 0},
                           "exit": {"enum": sorted({e for v in EXITS.values() for e in v})}, "freeze_s": {"type": "number", "minimum": 0, "maximum": 0.5},
                           "style": {"enum": list(CK.STYLES)}, "dir": {"enum": list(CK.DOLLY_DIRS)}}}},
    },
    "word_properties": {"window_source": {"type": "string", "description": "self | next | <shot id>"}},
}
EXAMPLE = {"cuts": [{"after": "s1", "cut": "window_swap", "word": 0, "exit": "fade", "freeze_s": 0.25}, {"after": "s2", "cut": "depth_dolly:right", "why": "the runner leaves right"},
                    {"after": "s3", "cut": "orbit:pullback", "why": "the song releases here"}]}


def schema_fragment(shot_ids=None):
    """-> dict(word=<line for the words schema>, cuts=<block for the compose schema>, note=<one line of rules>)."""
    ids = ", ".join(shot_ids) if shot_ids else "the SHOTS list"
    return dict(word=SCHEMA_WORD, cuts=SCHEMA_CUTS,
                note=f"cuts / window_source: shot ids are {ids}; window choices need a heavy liked face and a word at least 30 % of the frame tall; "
                     f"the code downgrades a choice that cannot read and says so")


MENU_TEXT = """TEXT AS WINDOW (optional, at most two per piece; the heaviest tools in the menu: use them where the picture can carry them)
Letters can be windows onto a second shot. Like a P10-style window: the letters are cut-outs onto footage, the plate around them is another picture, the
subject of the plate stands out in front of the letters. All three need a HEAVY face (Anybody, Ultra, Erica One, Badeen Display, Passion One, Blackout,
Tourney, BBH Bartle), a word at least 30 % of the frame tall, and two shots whose brightness or colour clearly differ.
- window_source "next" / "<shot id>" (per word): while the word holds, its letters show the NEXT shot (or the named one) and the word's own shot stays
  outside. WHEN: a short word (1-2 words, 3-6 letters) over a shot whose picture is calm behind the word; you want a preview of the next scene, or two
  places at once. Not for a shot that is dark or flat all over, not for two shots of the same scene and light.
- cut "window_cut" (per boundary): over shot A the word is a window onto shot B; the letters then grow until B fills the frame; that is the cut (no
  crossfade). WHEN: the word is the last thing said in shot A and the next shot is a new place or time; the cut sits on a downbeat. exit grow = the
  letters swell open; io = the letters stay and the cut is hard.
- cut "window_swap" (per boundary): B inside the letters with A outside, then ON THE CUT BEAT they swap: B becomes the whole picture and A is inside
  the letters; the letters then leave (fade | shrink | io) and only B remains. WHEN: the word is held across the cut; the cut lands on a downbeat,
  a chord change or a dropout; the strongest choice for a turn in the song. freeze_s (0.2-0.4) freezes both pictures on the swap, best on a dropout.
- cut "hard": an ordinary hard cut on the beat, no window. cut "crossfade" (default): the piece's own soft cut.
- cut "window_other" (per boundary, the TEXT FILLED WITH THE NEXT SHOT): the word before the cut holds with its letters showing the FOLLOWING shot, then the hard
  cut lands on the beat. WHEN: a held, heavy word (cap height >= 30 %) just before a place change, the letters open onto the place you are about to enter.
  Same face and contrast rules as the window cuts; it is the gentlest of the three.
Rules: only at a real boundary between two shots; the word's hold must span or end at the cut (a swap needs 0.5 s of hold before the cut and 0.4 s after);
two window cuts at most; never on two neighbouring boundaries; the word used by a window cut is not also a window_source word.

3D SHOT TRANSITIONS (cut kinds `depth_pivot`, `orbit:<push|pullback|fly>`; the shots become depth scenes for
about 1-3 s around the cut; the SUBJECT stays crisp in front of the type while the background transforms, the type sits between them and is never part of the
transition's pixels, and the word before the cut leaves and the word after it arrives in the direction of the move; the code does the matte and the word sides.
Each cut has a style (`params.style`, optional): chunks = the card mosaic (the pivot's look), smooth = a camera move through depth-parallax layers as one flat card (only when asked for; the default of every 3D cut is chunks, broken-up fragments))
- depth_pivot: the approved look. A pivots away round its volume and B swings in over the top, turned 45 degrees, and lands flat (3 s window: B is visible from
  1.2 s). WHEN: a place change with a subject in both shots (it needs a matte of each); the default 3D choice.
- orbit:push / orbit:pullback / orbit:fly: one rigid camera path through the two depth pictures. push = the camera dives into A's subject and B opens behind it (a build, the
  music rises into the cut), pullback = the camera backs out of A and B flies in past the lens (a release, the music drops after the cut), fly = an S-curve with a
  bank through both (a free, joyful or driving cut).
- pan:<dir>: the type camera travels <dir> from the word before the cut to the word after it and the plate moves with it. WHEN: two words that continue one
  sentence across a cut; the direction follows the footage's own camera move.
Rules: a piece with several shots uses a DIFFERENT transition at every boundary (a repeated kind, or the same kind and direction, is swapped by the code for the next
unused fitting one and logged); place changes (a new still) take a 3D, window or pan kind (never a crossfade or a plain hard cut: the code upgrades those), continuations of the same place (`first: last:<id>`) take `hard`, or a camera-move cut
(`orbit:*`, `pan:*`) when the two shots' cameras differ; the shot before a 3D cut needs >= 1.2 s of footage and the shot after it >= 1.8 s; the cut sits on a downbeat, a chord change or a dropout; keep a
word readable for at least 1.2 s before and after the transition window."""


def music_marks(ana):
    """[(t, kind)] sorted: downbeats, chord changes and dropout starts of a music analysis dict ({downbeats, chord_changes [{t}], curves.rms, fps, cut})."""
    out = []
    cut = float((ana or {}).get("cut") or 1e9)
    for d in (ana or {}).get("downbeats") or []:
        if float(d) < cut:
            out.append((float(d), "downbeat"))
    for c in (ana or {}).get("chord_changes") or []:
        out.append((float(c["t"] if isinstance(c, dict) else c), "chord change"))      # dicts {t, from, to} (analysis tool) or bare seconds (t2m grid)
    out += [(a, "dropout") for a, _ in _dropouts(ana)]
    return sorted(out)


def _dropouts(ana, thr=0.3, min_s=0.3):
    """(start s, length s) where the rms envelope stays under thr x its median for min_s (same rule as qwen_piece.dropouts)."""
    r = ((ana or {}).get("curves") or {}).get("rms") or []
    if not len(r):
        return []
    fps = float((ana or {}).get("fps") or 24)
    v = sorted(float(x) for x in r)
    med = v[len(v) // 2]
    out, i = [], 0
    while i < len(r):
        if float(r[i]) < thr * med:
            j = i
            while j < len(r) and float(r[j]) < thr * med:
                j += 1
            if (j - i) / fps >= min_s:
                out.append((round(i / fps, 2), round((j - i) / fps, 2)))
            i = j
        else:
            i += 1
    return out


def landing(t, ana):
    """(kind, time, delta s) of the music mark nearest to t, or None when the analysis has none."""
    ms = music_marks(ana)
    if not ms:
        return None
    mt, kd = min(ms, key=lambda m: abs(m[0] - t))
    return kd, mt, t - mt


def boundaries(shots):
    """[(shot id before, shot id after, t of the boundary s)] from rows {id, start s, dur s}: the boundary = the incoming shot's piece start."""
    return [(a["id"], b["id"], float(b["start"])) for a, b in zip(shots, shots[1:])]


KIND_TEXT = {
    "window_other": "window_other (per boundary, the TEXT FILLED WITH THE NEXT SHOT): the word before the cut holds with its letters showing the FOLLOWING shot, then the hard cut lands on the beat. "
                    "WHEN: a held, heavy word just before a place change; the gentlest window.",
    "window_cut": "window_cut (per boundary): over shot A the word is a window onto shot B; the letters then grow until B fills the frame; that is the cut. WHEN: the word is the last thing said in A and B is a "
                  "new place or time. exit grow | io.",
    "window_swap": "window_swap (per boundary): B inside the letters with A outside, ON THE CUT BEAT they swap, then the letters leave (fade | shrink | io). WHEN: the cut lands on a downbeat, a chord "
                   "change or a dropout; freeze_s 0.2-0.4 on a dropout.",
    "window_zoom": "window_zoom (per boundary): the word holds as a window onto the next shot, then the camera scales INTO one letter (ground and letter edge scale with it) until the next shot fills the "
                   "frame; that is the cut. WHEN: a short heavy word held up to the cut; a build into a new place.",
    "depth_pivot": "depth_pivot: A pivots away round its volume and B swings in over the top, turned 45 degrees, and lands flat. WHEN: a place change with a subject in both shots (needs a matte of each).",
    "orbit:push": "orbit:push: one rigid camera path through the two depth pictures; the camera dives into A's subject and B opens behind it (a build, the music rises into the cut).",
    "orbit:pullback": "orbit:pullback: the camera backs out of A and B flies in past the lens (a release, the music drops after the cut).",
    "orbit:fly": "orbit:fly: an S-curve with a bank through both shots (a free, joyful or driving cut).",
    "hard": "hard: an ordinary hard cut on the beat.",
}


def drawn_menu_text(system) -> str:
    """the transition menu of a piece that has a drawn system: ONLY its kinds (with the style of each), the rules that apply to them, nothing else. '' without a draw."""
    kinds = RS.kinds_of(system)
    if not kinds:
        return ""
    st = ((system or {}).get("cuts") or {}).get("styles") or {}
    out = ["SHOT TRANSITIONS DRAWN FOR THIS PIECE (the only kinds you may write; the draw is random per piece, the design chooses WHICH of them goes at WHICH boundary)"]
    for k in kinds:
        t = KIND_TEXT.get(k) or (f"{k}: the type camera travels {k.split(':')[1]} from the word before the cut to the word after it and the plate moves with it. WHEN: two words that continue one sentence "
                                 f"across a cut." if k.startswith("pan:") else k)
        out.append(f"- {t}" + (f" Style: {st[k]} (chunks = card mosaic, smooth = depth-parallax camera move, slide = lateral parallax slide)." if st.get(k) else ""))
    out.append("Rules: place changes (a new still) take one of the kinds above, never a crossfade or a plain hard cut (the code upgrades those); with more boundaries than kinds a kind may come back, "
               "not twice in a row; continuations of the same place take `hard`; every transition plays over at least 0.5 s (12 frames) so it can be SEEN, and no word hold covers it (at a 3D or pan cut the "
               "word before the cut leaves as the move starts); keep a word readable for at least 1.2 s before and after the transition window.")
    if any(k in WINDOW_KINDS or k == "window_other" for k in kinds):
        out.append("Window kinds need a HEAVY liked face (Anybody, Ultra, Erica One, Badeen Display, Passion One, Blackout, Tourney, BBH Bartle) and two shots that differ in brightness or colour; the word's "
                   "hold must span or end at the cut; two window cuts at most, never on neighbouring boundaries. A short word (up to 6 letters) or a short phrase fits; the code reads it vertically "
                   "when that is the only way to make the letters big.")
    return "\n".join(out)


def menu_text(shots, ana=None, system=None):
    """MENU_TEXT (or, with a drawn `system`, only the drawn kinds) + the real boundaries of this piece with the music mark nearest to each (swap fit) -> the text the package carries."""
    if len(shots) < 2:
        return ""
    rows = []
    for a, b, t in boundaries(shots):
        ln = landing(t, ana)
        fit = f"nearest {ln[0]} at {ln[1]:g} s ({ln[2]:+.2f} s)" + (", swap ok" if abs(ln[2]) <= BEAT_TOL else ", too far for a swap") if ln else "no music marks"
        rows.append(f"- after {a} (into {b}) at {t:g} s: {fit}")
    return (drawn_menu_text(system) or MENU_TEXT) + "\nBOUNDARIES (a window cut sits on one of these; times are fixed):\n" + "\n".join(rows)


def normalize_cuts(obj):
    """compose.cuts as the model wrote it -> (clean list, fixes). Structure only: unknown kinds / keys, types, ranges. No timing rules (validate)."""
    fixes = []
    if obj is None:
        return [], fixes
    if not isinstance(obj, list):
        return [], ["compose.cuts: not a list, dropped"]
    out = []
    for i, c in enumerate(obj):
        wh = f"compose.cuts[{i}]"
        if not isinstance(c, dict) or not isinstance(c.get("after"), str):
            fixes.append(f"{wh}: needs an object with `after` (a shot id), dropped")
            continue
        raw_kd = c.get("cut", "crossfade")
        kd = CK.normalize(raw_kd)
        if kd is None:
            fixes.append(f"{wh}.cut {raw_kd!r}: allowed {', '.join(CUT_KINDS)}; crossfade used")
            kd = "crossfade"
        elif kd != str(raw_kd).strip():
            if kd not in ("hard", "crossfade") or str(raw_kd).strip().lower() not in ("hard", "crossfade"):
                fixes.append(f"{wh}.cut {raw_kd!r} read as {kd!r}")
        o = {"after": c["after"], "cut": kd}
        if isinstance(c.get("why"), str) and c["why"].strip():
            o["why"] = c["why"].strip()[:300]
        if kd in CK.EXPLORE and isinstance(c.get("params"), dict):                       # promoted cuts: only the knobs the designer may set (pivot_cut.explore_plan fills the rest)
            o["params"] = {k: v for k, v in c["params"].items() if k in ("word", "order", "cols", "outline", "dir", "seed", "kmax")}
        if kd == "circle_grid":                                                        # the dots build B's word by default (the best of round 2); the word-less grid only when `word` is false / 0 in params
            o.setdefault("params", {})
            if o["params"].get("word") is None:
                o["params"]["word"] = "auto"
        if kd in SHOT_3D and isinstance(c.get("params"), dict):
            o["params"] = {k: v for k, v in c["params"].items() if isinstance(k, str)}
            if not SLIDE_ON and str(o["params"].get("style", "")).strip().lower() == "slide":
                o["params"].pop("style")
                fixes.append(f"{wh}.params.style 'slide' is not offered (SLIDE_ON is off): the kind's default style is used")
        if kd in SHOT_3D and c.get("style") is not None:                                  # how the picture changes at a 3D cut (cut_kinds.style_of): chunks | smooth | slide (+ dir)
            st = str(c.get("style")).strip().lower()
            if st in CK.STYLES:
                o["style"] = st
                dr = str(c.get("dir") or "").strip().lower()
                if st == "slide" and dr in CK.DOLLY_DIRS:
                    o["dir"] = dr
            else:
                fixes.append(f"{wh}.style {c.get('style')!r}: allowed {', '.join(CK.STYLES)}; the kind's default is used")
        if kd in CK.WINDOW or kd == "pan" or kd.startswith("pan:"):
            if c.get("word") is not None:
                try:
                    o["word"] = int(c["word"])
                except (TypeError, ValueError):
                    fixes.append(f"{wh}.word {c.get('word')!r}: unreadable, the code picks the word")
        if kd in WINDOW_KINDS:
            try:
                o["word"] = int(c["word"])
            except (KeyError, TypeError, ValueError):
                o["word"] = 0
                fixes.append(f"{wh}.word missing or unreadable: word 0 used")
            if kd == "window_zoom":
                try:
                    zs = float(c.get("zoom_s", ZOOM_S))
                except (TypeError, ValueError):
                    zs = ZOOM_S
                zc = min(max(zs, ZOOM_MIN_S), ZOOM_MAX_S)
                if abs(zc - zs) > 1e-9:
                    fixes.append(f"{wh}.zoom_s {zs:g}: clamped to {zc:g} (the zoom must be long enough to be seen)")
                o["zoom_s"] = round(zc, 3)
                if c.get("letter") is not None:
                    try:
                        o["letter"] = max(0, int(c["letter"]))
                    except (TypeError, ValueError):
                        fixes.append(f"{wh}.letter {c.get('letter')!r}: unreadable, the code picks the letter")
            ex = str(c.get("exit", DEFAULT_EXIT[kd])).strip().lower()
            if ex not in EXITS[kd]:
                fixes.append(f"{wh}.exit {c.get('exit')!r}: {kd} allows {', '.join(EXITS[kd])}; {DEFAULT_EXIT[kd]} used")
                ex = DEFAULT_EXIT[kd]
            o["exit"] = ex
            if c.get("freeze_s") not in (None, 0, 0.0):
                try:
                    f = float(c["freeze_s"])
                    g = min(max(f, 0.0), 0.5)
                    if abs(g - f) > 1e-9:
                        fixes.append(f"{wh}.freeze_s {f:g}: clamped to {g:g}")
                    if g > 0:
                        o["freeze_s"] = round(g, 3)
                except (TypeError, ValueError):
                    fixes.append(f"{wh}.freeze_s: unreadable, dropped")
        out.append(o)
    return out, fixes


# ================================================================================================== shot transitions: direction + fit
_MOVE = (("right", r"\b(to the right|rightwards?|right across|(?:moves?|pans?|runs?|walks?|drifts?|sweeps?|slides?|leans?) (?:to the )?right|towards? the right)\b"),
         ("left", r"\b(to the left|leftwards?|left across|(?:moves?|pans?|runs?|walks?|drifts?|sweeps?|slides?|leans?) (?:to the )?left|towards? the left)\b"),
         ("up", r"\b(rises?|rising|climbs?|ascend\w*|tilts? up|upwards?|lifts?|soars?)\b"),
         ("down", r"\b(descend\w*|falls?|sinks?|tilts? down|downwards?|drops?|pours?)\b"))
_DIR_ORDER = ("right", "left", "down", "up")


def suggest_direction(desc, used=()):
    """direction the outgoing shot's description moves in (subject or camera): the first of right / left / up / down it mentions, else the first
    direction not in `used`. -> (direction, 'motion' | 'default')."""
    for d, rx in _MOVE:
        if re.search(rx, str(desc or "").lower()) and d not in used:
            return d, "motion"
    return next((d for d in _DIR_ORDER if d not in used), _DIR_ORDER[0]), "default"


def music_slope(t, ana, span=1.0):
    """mean rms of the `span` s after t over the `span` s before it (None without an rms curve): > 1.15 = the music builds into the cut, < 0.87 = it releases."""
    r = ((ana or {}).get("curves") or {}).get("rms") or []
    fps = float((ana or {}).get("fps") or 24)
    if not len(r):
        return None
    i, n = int(round(t * fps)), max(1, int(round(span * fps)))
    a, b = r[max(0, i - n):i], r[i:i + n]
    if not len(a) or not len(b):
        return None
    ma = sum(float(x) for x in a) / len(a)
    return (sum(float(x) for x in b) / len(b)) / ma if ma > 1e-9 else None


def least_used_orbit(used=()):
    """the orbit mode used least often in `used` (a list of cut kinds); ties go fly, push, pullback (so an empty piece still starts on fly) -> alternates push / pullback / fly."""
    used = [str(u) for u in (used or [])]
    return min(("fly", "push", "pullback"), key=lambda m: used.count(f"orbit:{m}"))


def suggest_orbit(t, ana, used=()):
    """(mode, why) for an orbit cut at t from the music: build = push in, release = pull back; steady / no rms curve = the orbit mode least used in the piece (`used` = the cut kinds so far; never
    'fly' every time)."""
    r = music_slope(t, ana)
    if r is None:
        return least_used_orbit(used), "no rms curve"
    if r > 1.15:
        return "push", f"music builds into the cut (x{r:.2f})"
    if r < 0.87:
        return "pullback", f"music releases at the cut (x{r:.2f})"
    return least_used_orbit(used), f"music steady (x{r:.2f})"


def _row(shots, sid):
    return next((s for s in shots if s["id"] == sid), None)


def _frames(row, fps=24):
    return int(row.get("frames") or round(float(row.get("dur", 0)) * fps))


def feasible_3d(kind, params, row_a, row_b):
    """(ok, why not): the shot before needs `i2` frames of footage ahead of the cut and the shot after n - i2 (tools/fusion/pivot_cut.window_frames)."""
    try:
        from tools.fusion import pivot_cut as PC
        n, i2 = PC.window_frames(kind, params)
    except KeyError:
        raise
    except Exception as e:      # noqa: BLE001  engine not importable here: the check is skipped, the render side checks again
        return True, f"not checked ({type(e).__name__})"
    if row_a is not None and _frames(row_a) < i2:
        return False, f"{row_a['id']} has {_frames(row_a)} frames, {kind} needs {i2} before the cut"
    if row_b is not None and _frames(row_b) < n - i2:
        return False, f"{row_b['id']} has {_frames(row_b)} frames, {kind} plays {n - i2} of it after the cut"
    return True, ""


def _continuation(row):
    f = str((row or {}).get("first") or "")
    return f.startswith("last:")


def _pan_word(words, t_cut):
    """index of the word whose gap to the next word straddles the cut (its hold ends no later than MIN_OUT_S after it, the next hit is no more than MAX_PAN_GAP_S away)."""
    best = None
    for i in range(len(words) - 1):
        _, _, rel = word_times(words[i])
        _, hit2, _ = word_times(words[i + 1])
        if rel <= t_cut + MIN_OUT_S and 0 <= hit2 - t_cut <= MAX_PAN_GAP_S + 0.5:
            sc = abs(t_cut - rel) + abs(hit2 - t_cut)
            if best is None or sc < best[0]:
                best = (sc, i)
    return None if best is None else best[1]


def to_drawn(c, wh, system, used, fixes):
    """a normalised cut entry whose kind is not one of the piece's drawn kinds -> an entry of a drawn kind of the same family when there is one (a window for a window, a 3D for a 3D / pan), else
    a drawn 3D kind; a drawn orbit kind without a style takes the drawn style. Plain `hard` / `crossfade` stay (the planning rules decide those). -> entry (a copy when changed)."""
    drawn = RS.kinds_of(system)
    if not drawn:
        return c
    kd = c["cut"]
    if kd in ("hard", "crossfade"):
        return c
    if kd not in drawn:
        fam = CK.family_of(kd)
        pool = [k for k in drawn if CK.family_of(k) == fam and k != "hard"] or [k for k in drawn if RS.robust(k)] or [k for k in drawn if k != "hard"]
        new = RS.pick_next(system, used, among=pool, feasible=lambda k: k != (used[-1] if used else None) and list(used).count(k) < 2) if pool else None     # REPEAT RULE: not the kind of the cut before, not a kind used twice
        if new is None:
            fixes.append(f"{wh}: {kd} is not drawn for this piece ({', '.join(drawn)}); hard cut used")
            return {"after": c["after"], "cut": "hard"}
        fixes.append(f"{wh}: {kd} is not one of the kinds drawn for this piece ({', '.join(drawn)}); {new} used")
        keep = {k: v for k, v in c.items() if k in ("after", "why", "word")}
        if new in WINDOW_KINDS:
            keep.setdefault("word", c.get("word", 0))
            keep["exit"] = DEFAULT_EXIT[new]
        c = dict(keep, cut=new)
        kd = new
    st = RS.style_for(system, kd)
    if kd in SHOT_3D and st and not c.get("style") and not (c.get("params") or {}).get("style") and (st != "slide" or SLIDE_ON):
        c = dict(c, style=st)
    return c


def _resolve_special(c, wh, d, words, shots, bnd, ana, caps, probe, fixes, notes, allow_dolly=False, system=None, used=()):
    """a cut entry that is not a window_cut / window_swap: window_other (-> the word's window_source + a hard cut), 3D (bare family -> direction chosen
    from the footage / music; infeasible -> hard), pan (needs a word pair across the cut). -> the entry to keep."""
    kd = c["cut"]
    nxt, t_cut = bnd[c["after"]]
    ra, rb = _row(shots, c["after"]), _row(shots, nxt)
    if kd in ("hard", "crossfade"):
        return c
    if rb is not None and "first" in rb and _continuation(rb):
        if kd.startswith("orbit") or kd.startswith("pan"):
            pass                                                                # a camera-move cut inside one place: the cameras of the two shots differ (shot_camera), the move is the transition
        elif kd in SHOT_3D or kd in ("depth_dolly",):
            mode, why_ = suggest_orbit(t_cut, ana, used)                           # a pivot / dolly swaps PLACES; a continuation takes the camera-move cut the music asks for instead of a plain hard cut
            fixes.append(f"{wh}: {nxt} continues {c['after']} (same place); {kd} swaps places, orbit:{mode} used ({why_})")
            c = dict(c, cut=f"orbit:{mode}")
            c.pop("params", None)
            kd = c["cut"]
        else:
            fixes.append(f"{wh}: {nxt} continues {c['after']} (same place); {kd} is for a place change, hard cut used")
            return {"after": c["after"], "cut": "hard"}
    out = {k: v for k, v in c.items() if k != "params"}
    if kd == "window_other":
        wi = c.get("word", 0)

        def fits(i):
            if not 0 <= i < len(words):
                return False
            _, hit, rel = word_times(words[i])
            return hit <= t_cut - 0.5 and rel >= t_cut - 0.35
        if not fits(wi):
            alt = [i for i in range(len(words)) if fits(i)]
            if not alt:
                fixes.append(f"{wh}: no word holds up to the cut at {t_cut:g} s (hit >= 0.5 s before it, release within 0.35 s of it); hard cut used")
                return {"after": c["after"], "cut": "hard"}
            fixes.append(f"{wh}: word {wi} does not hold up to {t_cut:g} s; word {alt[0]} used")
            wi = alt[0]
        why = _word_ok(d, wi, wh, fixes, caps, fonts=(system or {}).get("fonts"))
        if not why:
            st, hit, rel = word_times(words[wi])
            ok, txt = _reads(probe, wi, "window_other", (hit + rel) / 2.0, notes, wh)
            why = "" if ok else txt
        if why:
            fixes.append(f"{wh}: downgraded to hard: {why}")
            return {"after": c["after"], "cut": "hard"}
        words[wi]["window_source"] = nxt
        out.update(cut="hard", was="window_other", word=wi)
        return out
    if (kd.startswith("depth_dolly") or kd == "depth_dolly") and not allow_dolly:
        alt = next(iter(_alternatives(rb, t_cut, ana, set(), ra, system=system)), "hard")
        fixes.append(f"{wh}: depth_dolly is off (slow, renders as a shard collage); {alt} used instead")
        if alt == "hard":
            return {"after": c["after"], "cut": "hard"}
        out["cut"] = kd = alt
        out.pop("params", None)
    if kd in SHOT_3D or kd in ("orbit", "depth_dolly"):
        params = dict(c.get("params") or {})
        if kd in ("orbit", "depth_dolly"):                                    # the direction is part of the kind: choose it from the footage / music
            if kd == "orbit":
                mode, why = suggest_orbit(t_cut, ana, used)
                kd = f"orbit:{mode}"
            else:
                dr, how = suggest_direction((ra or {}).get("desc", ""))
                kd, why = f"depth_dolly:{dr}", ("the outgoing shot moves that way" if how == "motion" else "no motion named, default order")
            fixes.append(f"{wh}: no direction given; {kd} chosen ({why})")
            out["cut"] = kd
        try:
            ok, txt = feasible_3d(kd, params, ra, rb)
        except KeyError as e:
            fixes.append(f"{wh}: unknown parameters dropped ({str(e)[:80]})")
            params = {}
            ok, txt = feasible_3d(kd, params, ra, rb)
        if not ok:
            fixes.append(f"{wh}: downgraded to hard: {txt}")
            return {"after": c["after"], "cut": "hard"}
        ln = landing(t_cut, ana)
        if ln is not None and abs(ln[2]) > 0.35:
            notes.append(f"{wh}: the cut at {t_cut:g} s is {abs(ln[2]):.2f} s from the nearest {ln[0]} ({ln[1]:g} s)")
        if params:
            out["params"] = params
        return out
    if kd.startswith("pan"):
        wi = c.get("word")
        if kd == "pan":
            dr, how = suggest_direction((ra or {}).get("desc", ""))
            kd = f"pan:{dr}"
            fixes.append(f"{wh}: no direction given; {kd} chosen ({'the outgoing shot moves that way' if how == 'motion' else 'default order'})")
            out["cut"] = kd
        pw = _pan_word(words, t_cut)
        if pw is None:
            fixes.append(f"{wh}: no pair of words has its gap across the cut at {t_cut:g} s; hard cut used")
            return {"after": c["after"], "cut": "hard"}
        if wi is not None and wi != pw:
            fixes.append(f"{wh}: word {wi} is not the one before the cut; word {pw} used")
        out["word"] = pw
        return out
    return out


def _alternatives(row_b, t_cut, ana, used, row_a, a_desc="", system=None):
    """kinds that could replace a repeated one at this boundary, best first. With a DRAWN system (random_system.draw_cuts) only the drawn 3D kinds, in the system's seeded order (the orbit the music
    asks for first when it is drawn); no fixed pivot-first / fly fallback. Without a draw (a request with no system): a place change takes the 3D kinds (pivot first, then the orbit the music
    asks for, then the other orbits; the depth dolly is slow and never a replacement); a continuation takes a hard cut."""
    if _continuation(row_b):
        return ["hard"]
    mode = suggest_orbit(t_cut, ana, list(used))[0]
    if RS.kinds_of(system):
        ks = [k for k in RS.kinds_of(system) if RS.robust(k) and k not in used]
        order = (system.get("cuts") or {}).get("order") or ks
        ks.sort(key=lambda k: (k != f"orbit:{mode}", order.index(k) if k in order else 99))
        return ks
    cand = [CK.PIVOT] + [f"orbit:{m}" for m in dict.fromkeys([mode, "push", "pullback", "fly"])]
    return [k for k in cand if k not in used]


# ================================================================================================== the validator
def _font_entry(name):
    from mograph import fonts_liked as fl
    return fl.lookup(name)


def is_fat(font, axes=None):
    """(ok, role) : is the liked font `font` a fat / heavy / blocky face for a window (role heavy, or a tall face at wght >= 800)."""
    e = _font_entry(font)
    if e is None:
        return False, None
    if e["role"] in FAT_ROLES:
        return True, e["role"]
    lim = FAT_IF_WEIGHT.get(e["key"])
    if lim is not None and isinstance(axes, dict):
        wg = axes.get("wght")
        wg = wg[1] if isinstance(wg, (list, tuple)) else wg
        if wg is not None and float(wg) >= lim:
            return True, e["role"]
    return False, e["role"]


def nearest_fat_font(font, fonts=None):
    """liked.json entry of the nearest fat liked font for `font` (by role), or None for a font that is not liked at all. `fonts` = the piece's drawn shortlist: a fat face on it wins."""
    from mograph import fonts_liked as fl
    e = fl.lookup(font)
    if e is None:
        return None
    for f in (fonts or []):
        fe = fl.lookup(f)
        if fe is not None and fe["role"] in FAT_ROLES:
            return fe
    for k in NEAREST_FAT.get(e["role"], NEAREST_FAT["heavy"]):
        if k in fl.BY_KEY:
            return fl.BY_KEY[k]
    return None


def _fat_axes(entry, axes):
    """axes dict for the swapped font: wght at the heavy end of its range when the registry says the face has one."""
    try:
        from mograph import font_registry as fr
        vary = next((v for v in fr.REGISTRY.values() if v["file"] == entry["file"]), {}).get("vary") or {}
        if "wght" in vary:
            return {"wght": float(min(vary["wght"][1], 900.0))}
    except Exception:       # noqa: BLE001
        pass
    return None


def word_times(w):
    """(start, hit, release) seconds of a design word (hit + hold_s when it has no release)."""
    hit = float(w.get("hit", w.get("t0", 0.0) or 0.0))
    rel = float(w["release"]) if "release" in w else hit + float(w.get("hold_s", 1.6))
    st = float(w["t0"]) if w.get("t0") is not None else hit - float(w.get("form_s", 0.75))
    return st, hit, rel


def caps_from_manifest(type_dir):
    """{word index: cap height / frame height} measured by the type engine (type pass manifest: info.words[].cap_px / world height)."""
    from pathlib import Path
    m = json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8"))
    return {i: float(w["cap_px"]) / float(m["world"][1]) for i, w in enumerate((m.get("info") or {}).get("words") or []) if w.get("cap_px")}


_LINE_CACHE: dict = {}


def line_width_per_cap(font, axes, line):
    """width of `line` in the liked font `font` (file / key) at the given axes, in units of its cap height (PIL metrics of the real font file, the registry's default axes under the word's
    own): the cover fraction of a word from its actual glyphs. The older estimate was len(line) x CHAR_W, which is the width of a condensed face and over-promised for a normal heavy one
    (BadeenDisplay: ~0.91 x len). None when the font file or PIL is not available (callers fall back to CHAR_W)."""
    try:
        e = _font_entry(font)
        if e is None:
            return None
        ax = {}
        reg = None
        from mograph import font_registry as fr
        reg = next((v for v in fr.REGISTRY.values() if v["file"] == e["file"]), {})
        ax.update({k: float(v) for k, v in (reg.get("axes") or {}).items()})
        for k, v in (axes or {}).items():
            v = v[1] if isinstance(v, (list, tuple)) else v
            try:
                ax[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        key = (e["file"], tuple(sorted(ax.items())), str(line))
        if key in _LINE_CACHE:
            return _LINE_CACHE[key]
        import os

        from PIL import ImageFont
        f = ImageFont.truetype(os.path.join(fr.FONT_DIR, e["file"]), 200)
        try:
            names = f.get_variation_axes()
        except Exception:       # noqa: BLE001  not a variable font
            names = []
        if names:
            tag_of = {str(n).lower(): t for t, n in fr.AXIS_NAMES.items()}
            vals = []
            for a in names:
                nm = a["name"].decode("latin-1").lower() if isinstance(a["name"], bytes) else str(a["name"]).lower()
                tg = next((t for t in ax if t.lower() == tag_of.get(nm, nm)), None)
                vals.append(min(max(ax[tg], a["minimum"]), a["maximum"]) if tg is not None else a["default"])
            f.set_variation_by_axes(vals)
        bb = f.getbbox("H")
        cap = max(bb[3] - bb[1], 1)
        r = float(f.getlength(str(line))) / cap
        _LINE_CACHE[key] = r
        return r
    except Exception:       # noqa: BLE001
        return None


def cap_estimates(w, aspect=0.5625):
    """(height limit, width limit) of a word's cap height as fractions of the frame height, from its placement; None where unknown. The width limit uses the glyph widths of the word's
    font (line_width_per_cap) when the font is known, else len(line) x CHAR_W."""
    pl = w.get("placement") or {}
    lines = [str(x) for x in (w.get("lines") or [])] or [str(w.get("text", "")).replace(" ", "")]
    rolled = is_rolled(pl)
    est_h = CAP_PER_HF * float(pl["hf"]) * (aspect if rolled else 1.0) / max(1, len(lines)) if pl.get("hf") is not None else None
    ratio = None
    if w.get("font"):
        rs = [line_width_per_cap(w["font"], w.get("axes"), x) for x in lines]
        ratio = max(rs) if rs and None not in rs else None
    if ratio is None:
        ratio = max(len(x) for x in lines) * CHAR_W
    est_w = float(pl.get("wf", 0.8)) * (1.0 if rolled else aspect) / ratio       # rolled +-90: wf is the long side as a fraction of the frame HEIGHT (spec.build_scene)
    return est_h, est_w


def is_rolled(pl) -> bool:
    """is the placement rolled +-90 degrees (the word reads vertically)? then wf / hf are fractions of the frame height / width (spec.build_scene)."""
    try:
        return abs(abs(float((pl or {}).get("roll") or 0.0)) - 90.0) < 1.0
    except (TypeError, ValueError):
        return False


def cap_fraction(w, i=None, caps=None):
    """(cap height / frame height, 'measured' | 'estimated' | None) of a word: ctx caps first (measured on the type pass), else from placement."""
    if caps is not None and i is not None and caps.get(i) is not None:
        return float(caps[i]), "measured"
    est_h, est_w = cap_estimates(w)
    if est_h is None:
        return None, None
    return min(est_h, est_w), "estimated"


def _reads(probe, i, kind, t, notes, where):
    """(ok, text). probe(kind, word index, t) -> (frame_a, frame_b, type_a) float arrays or None; ok False when the two pictures are too alike."""
    if probe is None:
        notes.append(f"{where}: contrast not measured (no probe)")
        return True, ""
    got = probe(kind, i, t)
    if got is None:
        notes.append(f"{where}: contrast not measured (probe returned nothing)")
        return True, ""
    import numpy as np
    from tools.fusion import compose as C
    fa, fb, ta = got
    lin, lout, n = C.window_contrast(fa, ta, fb)
    la = float(np.mean(C._small_np(fa) @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
    lb = float(np.mean(C._small_np(fb) @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
    dc = float(np.linalg.norm(C._small_np(fa).reshape(-1, 3).mean(0) - C._small_np(fb).reshape(-1, 3).mean(0)))
    gap = max(abs(lin - lout) if n else 0.0, abs(la - lb))
    if gap >= MIN_LUMA_GAP or dc >= MIN_COLOR_GAP:
        return True, ""
    return False, f"the two shots are too alike (luma gap {gap:.2f} < {MIN_LUMA_GAP}, colour gap {dc:.2f} < {MIN_COLOR_GAP})"


def _word_ok(design, i, where, fixes, caps, allow_cap_bump=True, fonts=None):
    """font + cap-height rules for word i as a window. `fonts` = the piece's drawn shortlist (files): a fat face of it is taken before any other. -> '' when it can be a window (maybe after a fix),
    else the reason it cannot."""
    w = design["type"]["words"][i]
    ok, role = is_fat(w.get("font"), w.get("axes"))
    if not ok:
        e = nearest_fat_font(w.get("font"), fonts)
        if e is None:
            return f"font {w.get('font')!r} is not a liked font"
        old = w.get("font")
        w["font"] = e["file"]
        ax = _fat_axes(e, w.get("axes"))
        w["axes"] = ax if ax else {k: v for k, v in (w.get("axes") or {}).items() if k != "wght"} or None
        if w["axes"] is None:
            w.pop("axes")
        fixes.append(f"{where}: font {old!r} ({role}) is not a fat face; word {i} font -> {e['file']!r}")
    cap, how = cap_fraction(w, i, caps)
    if cap is None:
        return ""
    thr = WINDOW_MIN_CAP * CAP_TOL
    if cap + 1e-9 < thr:
        if how == "estimated" and allow_cap_bump:
            est_h, est_w = cap_estimates(w)
            n = max(1, len(w.get("lines") or [1]))
            need = WINDOW_MIN_CAP * n / CAP_PER_HF
            if est_w + 1e-9 >= thr and need <= 0.95:               # the box height is the limit: make the box taller
                hf0 = (w.get("placement") or {}).get("hf")
                w.setdefault("placement", {})["hf"] = round(need + 0.01, 3)
                fixes.append(f"{where}: word {i} cap height ~{cap:.0%} < {WINDOW_MIN_CAP:.0%}; placement.hf {hf0:g} -> {need + 0.01:.3g}")
                return ""
            if est_w + 1e-9 < thr:
                fit = window_fit2(w, fonts=fonts)
                if fit["ok"]:                                   # the code stacks the word into lines / widens the box / takes the narrow heavy face (real glyph widths), instead of giving the window up
                    old_ = (w.get("lines"), w.get("font"), (w.get("placement") or {}).get("wf"))
                    if fit["lines"]:
                        w["lines"] = fit["lines"]
                    if fit["font"]:
                        w["font"] = fit["font"]
                        if fit["axes"]:
                            w["axes"] = fit["axes"]
                        else:
                            w.pop("axes", None)
                    pl_ = w.setdefault("placement", {})
                    pl_["wf"] = fit["wf"]
                    pl_["hf"] = fit["hf"] if fit.get("roll") else max(float(pl_.get("hf", 0.0)), fit["hf"])
                    pl_["roll"] = fit.get("roll", 0.0)
                    fixes.append(f"{where}: word {i} cap height ~{cap:.0%} < {WINDOW_MIN_CAP:.0%}; fitted to ~{fit['cap']:.0%} (lines {fit['lines'] or old_[0] or 'unchanged'}, face {fit['font'] or 'unchanged'}, wf {old_[2]} -> {fit['wf']}{', read vertically (rolled 90)' if fit.get('roll') else ''})")
                    return ""
                return f"word {i} is too long for a cap height of {WINDOW_MIN_CAP:.0%} even stacked (~{est_w:.0%} at this width, {len(re.sub(r'[^A-Za-z0-9]', '', str(w.get('text', ''))))} letters)"
        return f"word {i} cap height {'~' if how == 'estimated' else ''}{cap:.0%} of the frame is under {WINDOW_MIN_CAP:.0%}"
    return ""


def validate(design, ctx, notes=None):
    """Deterministic validator / fixer for design["compose"]["cuts"] and the words' `window_source`. design = the normalised design dict
    (type.words with font / hit / hold_s / placement ..., compose). ctx = dict(
        shots=[{id, start (s), dur (s)}]          rows as qwen_piece.shot_rows
        ana=music analysis dict                    downbeats, chord_changes, curves.rms, fps, cut
        caps={word index: cap height fraction}     optional, measured on the type pass (else estimated from placement.hf)
        probe=callable(kind, word index, t)        optional -> (frame_a, frame_b, type_a) float frames for the contrast rule; None = not measured).
    -> (design copy, fixes [str]). Every change (font swap, hf bump, downgrade, drop) is one fix string; `notes` (list) collects checks that could
    not be run. Order: structure, boundary, word, font + cap, beat (swap), contrast, adjacency, per-piece caps."""
    d = copy.deepcopy(design)
    notes = notes if notes is not None else []
    d.setdefault("compose", {})
    shots = list(ctx.get("shots") or [])
    caps, probe, ana = ctx.get("caps"), ctx.get("probe"), ctx.get("ana")
    ids = [s["id"] for s in shots]
    words = (d.get("type") or {}).get("words") or []
    cuts, fixes = normalize_cuts(d["compose"].get("cuts"))
    bnd = {a: (b, t) for a, b, t in boundaries(shots)}

    # ---- cuts
    seen, kept = set(), []
    system = ctx.get("system")
    fonts_ok = ctx.get("fonts") or (system or {}).get("fonts")
    drawn_used: list = []
    for k, c in enumerate(cuts):
        wh = f"cuts[{k}] {c['cut']} after {c['after']}"
        if c["after"] not in seen and c["after"] in bnd:
            c = to_drawn(c, wh, system, drawn_used, fixes)
            drawn_used.append(c["cut"])
        if c["after"] in seen:
            fixes.append(f"{wh}: second entry for the same boundary, dropped")
            continue
        seen.add(c["after"])
        if c["after"] not in bnd:
            fixes.append(f"{wh}: {'no such shot' if c['after'] not in ids else 'the last shot has no boundary after it'}; entry dropped")
            continue
        if c["cut"] not in WINDOW_KINDS:
            kept.append(_resolve_special(c, wh, d, words, shots, bnd, ana, caps, probe, fixes, notes, allow_dolly=bool(ctx.get("allow_dolly", DOLLY_ON)), system=system, used=[k_["cut"] for k_ in kept]))
            continue
        nxt, t_cut = bnd[c["after"]]
        wi = c.get("word", 0)
        # which word can carry it: the asked one, else the best other one (hold spans / ends at the cut)
        def fits(i, kind=c["cut"]):
            if i < 0 or i >= len(words):
                return False
            _, hit, rel = word_times(words[i])
            return hit <= t_cut - (MIN_P1 if kind == "window_swap" else 0.5) and rel >= t_cut + (MIN_P2 if kind == "window_swap" else -0.35)
        why = ""
        if not fits(wi):
            alt = [i for i in range(len(words)) if fits(i)]
            if alt:
                fixes.append(f"{wh}: word {wi} does not hold across {t_cut:g} s; word {alt[0]} used")
                wi = alt[0]
            else:
                why = f"no word holds across the cut at {t_cut:g} s (needs {'0.5 s before and 0.4 s after' if c['cut'] == 'window_swap' else 'a hold ending within 0.35 s of it'})"
        if not why:
            why = _word_ok(d, wi, wh, fixes, caps, fonts=fonts_ok)
        if not why and c["cut"] == "window_swap":
            ln = landing(t_cut, ana)
            if ln is None:
                notes.append(f"{wh}: no music marks to check the swap against")
            elif abs(ln[2]) > BEAT_TOL:
                why = f"the cut at {t_cut:g} s is {abs(ln[2]):.2f} s from the nearest {ln[0]} ({ln[1]:g} s); a swap must land within {BEAT_TOL:g} s"
        if not why:
            tp = min(max(t_cut - 0.3, word_times(words[wi])[1]), word_times(words[wi])[2])
            ok, txt = _reads(probe, wi, c["cut"], tp, notes, wh)
            why = "" if ok else txt
        if why:
            fixes.append(f"{wh}: downgraded to hard: {why}")
            kept.append({"after": c["after"], "cut": "hard"})
            continue
        c["word"] = wi
        kept.append(c)
    # ---- adjacency + per-piece cap
    out, last_k, n_win, used_w = [], -9, 0, set()
    for c in sorted(kept, key=lambda c: ids.index(c["after"])):
        if c["cut"] in WINDOW_KINDS:
            k = ids.index(c["after"])
            wh = f"cuts {c['cut']} after {c['after']}"
            if k - last_k == 1:
                fixes.append(f"{wh}: neighbouring boundary already has a window cut; downgraded to hard")
                c = {"after": c["after"], "cut": "hard"}
            elif n_win >= MAX_WINDOW_CUTS:
                fixes.append(f"{wh}: more than {MAX_WINDOW_CUTS} window cuts in the piece; downgraded to hard")
                c = {"after": c["after"], "cut": "hard"}
            elif c["word"] in used_w:
                fixes.append(f"{wh}: word {c['word']} already carries a window cut; downgraded to hard")
                c = {"after": c["after"], "cut": "hard"}
            else:
                n_win += 1
                last_k = k
                used_w.add(c["word"])
        out.append(c)
    # ---- no repeat: every special kind (window / 3D / pan, with its direction) at most once per piece; a repeat becomes the next unused fitting kind
    used, uniq = set(), []
    used_l: list = []
    last_key = None
    for c in out:
        key = c.get("was") or c["cut"]
        if key in CK.ALL and key in SPECIAL:
            if key in used:
                nb, t_cut = bnd[c["after"]]
                ra, rb = _row(shots, c["after"]), _row(shots, nb)
                alts = [k for k in _alternatives(rb, t_cut, ana, used, ra, (ra or {}).get("desc", ""), system=system) if k == "hard" or feasible_3d(k, {}, ra, rb)[0]]
                if RS.kinds_of(system) and not alts and key in SHOT_3D and key != last_key and used_l.count(key) < 2:
                    pass                                   # a DRAWN piece has 2-3 kinds and more boundaries than that: a 3D kind may come back once the drawn ones are used (never twice in a row)
                else:
                    new = alts[0] if alts else "hard"
                    fixes.append(f"cuts {key} after {c['after']}: the piece already uses {key}; {new} used instead (different transitions at different cuts)")
                    if c.get("was") == "window_other" and c.get("word") is not None:
                        words[c["word"]].pop("window_source", None)
                    c = {"after": c["after"], "cut": new}
                    key = new
            used.add(key)
            used_l.append(key)
            last_key = key
        uniq.append(c)
    out = uniq
    if ctx.get("fill_3d"):                                # a window choice the rules downgraded at a place change gets a designed 3D cut instead of a plain hard one
        out, n3 = fill_3d(out, shots, ana, keep_hard=model_hard(design["compose"].get("cuts")), system=system)
        fixes += [f"cuts: {x}" if not x.startswith("cuts:") else x for x in n3]
    d["compose"]["cuts"] = out
    if not out:
        d["compose"].pop("cuts", None)

    # ---- per-word window_source
    n_src = 0
    for i, w in enumerate(words):
        src = w.get("window_source")
        ground = w.get("window_ground")
        if src in (None, "", "self") and not ground:
            w.pop("window_source", None)
            w.pop("window_ground", None)
            continue
        if src in (None, "", "self"):
            src = "self"                                             # inverse type on a flat ground: the word's OWN footage shows inside the letters
        wh = f"words[{i}].window_source {src!r}" + (" on a flat ground" if ground else "")
        st, hit, rel = word_times(w)
        under = next((s for s in shots if float(s["start"]) <= (hit + rel) / 2.0 < float(s["start"]) + float(s["dur"])), shots[-1] if shots else None)
        tgt = src
        if src == "next":
            k = ids.index(under["id"]) + 1 if under else len(ids)
            tgt = ids[k] if k < len(ids) else None
        elif src == "self":
            tgt = under["id"] if under else None
        why = ""
        if under is None or (len(shots) < 2 and not ground):
            why = "the piece has a single shot"
        elif tgt not in ids:
            why = f"{'no next shot' if src == 'next' else 'no such shot'}"
        elif tgt == under["id"] and not ground:
            fixes.append(f"{wh}: names the word's own shot; set to self")
            w.pop("window_source")
            continue
        elif any(c["cut"] in WINDOW_KINDS and c.get("word") == i for c in out):
            why = "the word already carries a window cut"
        elif n_src >= MAX_WINDOW_WORDS:
            why = f"more than {MAX_WINDOW_WORDS} window_source words in the piece"
        if not why:
            why = _word_ok(d, i, wh, fixes, caps, fonts=fonts_ok)
        if not why and not ground:
            ok, txt = _reads(probe, i, "window_other", (hit + rel) / 2.0, notes, wh)
            why = "" if ok else txt
        if why:
            fixes.append(f"{wh}: set to self: {why}")
            w.pop("window_source")
            w.pop("window_ground", None)
            for c in d["compose"].get("cuts") or []:
                if c.get("was") == "window_other" and c.get("word") == i:
                    c.pop("was")
                    c.pop("word", None)
            continue
        w["window_source"] = tgt
        n_src += 1
    return d, fixes


# ================================================================================================== planning rule: put a word on a place change (window_other)
WO_MIN_BEFORE = 0.6          # s: the word's hit is at least this long before the cut (the validator asks 0.5 after the timing compile moved it a little)
WO_OVER_S = 0.45             # s: the planned hold ends this long AFTER the cut (the letters are a window onto the next shot until the hard cut; the hold carries on over the new shot)
WO_MAX_STRETCH_S = 2.5       # s: the most a hold may be lengthened to reach the cut
WO_MAX_HIT_GAP_S = 3.4       # s: the word that carries the cut hit no earlier than this before it
HOLD_MAX_S = 6.0             # RANGES["hold_s"][1] of qwen_piece


def model_hard(raw) -> frozenset:
    """the boundaries (`after` ids) where the MODEL wrote an explicit `hard` cut: a deliberate choice the planning rules leave alone (round-3 QC 2026-10-29). Read from the reply's own
    compose.cuts BEFORE any rule or validator has touched it (a hard the validator makes by downgrading a window choice is not in it and is still planned over)."""
    return frozenset(c.get("after") for c in (raw or []) if isinstance(c, dict) and CK.normalize(c.get("cut")) == "hard" and c.get("after"))


def _plain_cut(entry, keep_hard=frozenset()):
    """a cuts[] entry that chose nothing heavy at its boundary (crossfade, or a kind that is not in the vocabulary). An explicit `hard` the model wrote stays hard (round-3 QC rule: the code
    never turns a deliberate hard cut into a 3D kind: `keep_hard` = model_hard(the reply's cuts)); a hard the validator made from a window choice is planned over as before."""
    keep_hard = keep_hard or frozenset()
    kd = CK.normalize(entry.get("cut", "crossfade"))
    return kd in (None, "hard", "crossfade") and not entry.get("was") and not (kd == "hard" and entry.get("after") in keep_hard)


WO_MAX_LETTERS = 6           # a window word has at most this many letters (letters only; was 4 until 2026-10-30): a word is never cut across lines (round-3 QC); a longer one reads vertically (window_fit2 rolled candidate)
WO_MAX_PHRASE_LETTERS = 9    # a short PHRASE (2+ words, lines break only at spaces) may have this many letters in all
WO_LEN_MAX = 0.86            # a rolled window word: its line is at most this share of the frame height
WO_WF_MAX = 1.0              # the word box may use the whole frame width
WO_HF_MAX = 0.95
CONDENSED = ("Anybody", {"wdth": 50.0, "wght": 900.0})      # the narrowest heavy liked face: Anybody at its condensed end (width axis 50)


def word_aligned(text, lines) -> bool:
    """do `lines` break the text only at its spaces (every line a run of whole words)? A line list that cuts a word (MIDN/IGHT, WA/KE/UP) is not aligned."""
    return " ".join(str(x).strip() for x in (lines or [])).split() == str(text or "").split()


def _splits(text: str, lines):
    """candidate line lists for a word: its own lines (only when they break at spaces), the text as one line, one line per token. Lines break ONLY at spaces: a word is never cut into
    letter slices (round-3 QC 2026-10-29: MIDN/IGHT, WA/KE/UP, FLO/WST/ATE, GLA/SS); a single word that is too wide is scaled / squeezed, not stacked."""
    flat = text.replace(" ", "")
    cands = []
    if lines and word_aligned(text, lines):
        cands.append([str(x) for x in lines])
    cands.append([text.strip()] if " " in text.strip() else [flat])
    toks = [t for t in text.split() if t]
    if len(toks) > 1:
        cands.append(toks)
    uniq = []
    for c in cands:
        if c not in uniq and "".join(c).replace(" ", "").upper() == flat.upper():
            uniq.append(c)
    return uniq


def _window_fonts(w, fonts=None):
    """[(font file, axes)] to try for a window word: its own face when fat, the fat faces of the piece's drawn shortlist (`fonts`), the condensed Anybody, then every heavy liked face."""
    from mograph import fonts_liked as fl
    out = []
    if w.get("font") and is_fat(w.get("font"), w.get("axes"))[0]:
        out.append((w["font"], w.get("axes")))
    for f in (fonts or []):
        fe = fl.lookup(f)
        if fe is not None and fe["role"] in FAT_ROLES and all(fe["file"] != o[0] for o in out):
            out.append((fe["file"], {"wdth": 50.0, "wght": 900.0} if fe["key"] == CONDENSED[0] else None))
    e = fl.BY_KEY.get(CONDENSED[0])
    if e:
        out.append((e["file"], dict(CONDENSED[1])))
    out += [(x["file"], None) for x in fl.LIKED if x["role"] in FAT_ROLES and x["key"] != CONDENSED[0]]
    return out


def window_fit2(w, aspect=0.5625, fonts=None):
    """can a design word be a window (cap height >= WINDOW_MIN_CAP x CAP_TOL of the frame height) once the code sets its lines, face and box? Real glyph widths
    (line_width_per_cap), up to WO_MAX_LETTERS letters. Tries the word's own lines and face first, then an even split into 2-3 lines and the heaviest narrow faces (Anybody at width 50,
    then the other heavy liked faces); the best cap height wins. -> dict(ok, cap, lines (list | None = the word's own), font (file | None = unchanged), axes (dict | None), wf, hf)."""
    text = str(w.get("text", ""))
    letters = re.sub(r"[^A-Za-z0-9]", "", text)
    thr = WINDOW_MIN_CAP * CAP_TOL
    wf0 = float((w.get("placement") or {}).get("wf", 0.8)) if isinstance(w.get("placement"), dict) else 0.8
    limit = WO_MAX_PHRASE_LETTERS if len(text.split()) > 1 else WO_MAX_LETTERS
    if not letters or len(letters) > limit:
        return dict(ok=False, cap=0.0, lines=None, font=None, axes=None, wf=wf0, hf=None, roll=0.0)
    best = None
    own_lines = [str(x) for x in (w.get("lines") or [])]
    fonts_ = _window_fonts(w, fonts)
    rolled_ok = aspect < 1.0
    for fi, (ff, fax) in enumerate(fonts_):
        for li, ls in enumerate(_splits(text, own_lines)):
            rs = [line_width_per_cap(ff, fax, x) for x in ls]
            if None in rs:
                rs = [len(x) * CHAR_W for x in ls]
            cap = min(WO_HF_MAX / len(ls), WO_WF_MAX * aspect / max(rs))
            # preference: keep the word's own face and lines (an index penalty), the cap decides among candidates that pass
            score = (cap >= thr - 1e-9, -(fi if cap >= thr else 0) * 0.0 - (li + fi * 3) * 1e-3 if cap >= thr else cap)
            if best is None or score > best[0]:
                best = (score, cap, ls, ff, fax, max(rs), fi, li)
    _, cap, ls, ff, fax, ratio, fi, li = best
    if cap < thr - 1e-9 and rolled_ok:                       # no horizontal layout reaches the size: read it vertically (rolled +-90) along the frame height
        rbest = None
        for fi2, (ff2, fax2) in enumerate(fonts_):
            r2 = line_width_per_cap(ff2, fax2, " ".join(text.split()))
            r2 = r2 if r2 is not None else len(letters) * CHAR_W
            cap2 = min(WO_LEN_MAX / r2, 0.88 * aspect)
            if rbest is None or cap2 > rbest[0] + 1e-9:
                rbest = (cap2, ff2, fax2, r2, fi2)
        if rbest is not None and rbest[0] >= thr - 1e-9 and rbest[0] > cap:
            cap2, ff2, fax2, r2, fi2 = rbest
            own2 = fi2 == 0 and w.get("font") and is_fat(w.get("font"), w.get("axes"))[0]
            c0 = max(cap2 if cap2 < thr + 0.02 else thr + 0.012, thr + 0.012)
            c0 = min(c0, cap2)
            return dict(ok=True, cap=round(cap2, 4), lines=[" ".join(text.split())], font=None if own2 else ff2, axes=None if own2 else (dict(fax2) if fax2 else None),
                        wf=round(min(WO_LEN_MAX + 0.02, c0 * r2 / 0.94 + 0.01), 3), hf=round(min(0.9, c0 / aspect * 1.15 + 0.04), 3), roll=90.0)
    ok = cap >= thr - 1e-9
    own = fi == 0 and w.get("font") and is_fat(w.get("font"), w.get("axes"))[0]
    wf = round(min(WO_WF_MAX, max(wf0, (thr + 0.012) * ratio / aspect + 0.02)), 3)
    hf = round(min(WO_HF_MAX, max(float((w.get("placement") or {}).get("hf", 0.0)), (thr + 0.012) * len(ls) + 0.01)), 3)
    same_lines = ls == (own_lines or [letters]) or (not own_lines and ls == [letters])
    return dict(ok=bool(ok), cap=round(cap, 4), lines=None if same_lines else ls, font=None if own else ff, axes=None if own else (dict(fax) if fax else None), wf=wf, hf=hf, roll=0.0)


def window_fit(w, aspect=0.5625):
    """back-compat view of window_fit2: (ok, lines | None, wf)."""
    r = window_fit2(w, aspect)
    return r["ok"], r["lines"], r["wf"]


def _existing_window_other(words, raw_cuts) -> int:
    n = sum(1 for w in words if isinstance(w, dict) and w.get("window_source") not in (None, "", "self"))
    return max(n, sum(1 for c in raw_cuts if isinstance(c, dict) and CK.normalize(c.get("cut")) in ("window_other", "window_zoom", "window_cut")))


def place_changes(shots) -> int:
    return sum(1 for a, b, _ in boundaries(shots) if not _continuation(_row(shots, b)))


def wo_target(shots) -> int:
    """how many window_other cuts a piece takes: 1 up to three place changes, 2 beyond (never neighbouring; alternating with the 3D cuts)."""
    return 1 if place_changes(shots) <= 3 else 2


def plan_window_other(reply, shots, ana=None, max_n=1, keep_hard=None, kind="window_other", fonts=None):
    """PLANNING / SALVAGE RULE for the window_other cut. The models never put a word's hold across a shot boundary (r7: 0 of 8 replies), so `window_other` (the letters of the
    word before the cut are a window onto the next shot, then the hard cut) is never reached. This puts a word there. `reply` = a design reply dict (type.words with the
    model's own hit / hold_s, compose.cuts); `shots` = rows {id, start, dur, first} (qwen_piece.shot_rows); `ana` = the music analysis. -> (reply copy, notes, pins).
    A boundary takes it when ALL hold: the next shot is a new place (`first` is not `last:<id>`); the design chose nothing heavy there (no entry, `hard` or `crossfade`:
    a 3D, pan or window choice of the model is never overridden); the piece has fewer than `max_n` window_other / window_source words and no window cut or window_other on a
    neighbouring boundary (a kind is used once per piece, never twice in a row). Among those boundaries the one that lands on the music (a downbeat / chord change / dropout within
    BEAT_TOL: the held beat) comes first, then the one whose current word already has a heavy face, then the nearest mark. The current word = the last word whose hit is at least
    WO_MIN_BEFORE before the cut (and the next word does not hit just before it); its hold_s is lengthened (by at most WO_MAX_STRETCH_S, to HOLD_MAX_S) so that it ENDS WO_OVER_S after
    the cut, and the entry `{"after", "cut": "window_other", "word", "why"}` replaces / is added to compose.cuts. A word of up to WO_MAX_LETTERS letters qualifies (window_fit2: real
    glyph widths; the code sets the lines, the heavy narrow face and the box so the cap height reaches 30 % of the frame); only a longer word is skipped. The validator (`validate`)
    then still checks the face, the cap height, the hold and the contrast on the compiled timing and downgrades to `hard` where it cannot read, so this only proposes. No boundary
    fits -> the reply comes back unchanged.
    -> (reply copy, notes, pins {word index: release time s}): the timing compile (qwen_piece._compile) keeps the pinned word's RELEASE there while it shifts hits to fit the
    music (the models overrun the music in practice, so a plain hold_s would be shortened again by the fit)."""
    out = copy.deepcopy(reply) if isinstance(reply, dict) else reply
    notes, pins = [], {}
    if not isinstance(out, dict) or len(shots or []) < 2:
        return out, notes, pins
    t = out.get("type") if isinstance(out.get("type"), dict) else {}
    words = t.get("words") if isinstance(t.get("words"), list) else []
    comp = out.get("compose") if isinstance(out.get("compose"), dict) else {}
    raw_cuts = comp.get("cuts") if isinstance(comp.get("cuts"), list) else []
    if not words:
        return out, notes, pins
    if _existing_window_other(words, raw_cuts) >= max_n:
        return out, notes, pins                                              # the piece already has its window_other(s)
    by_after = {c.get("after"): c for c in raw_cuts if isinstance(c, dict)}
    ids = [r["id"] for r in shots]
    heavy_at = {c.get("after") for c in raw_cuts if isinstance(c, dict) and CK.normalize(c.get("cut")) in WINDOW_KINDS + ("window_other",)}

    def num(w, k, dflt=None):
        try:
            return float(w[k])
        except (KeyError, TypeError, ValueError):
            return dflt
    cands, skipped = [], []
    for a, b, t_cut in boundaries(shots):
        rb = _row(shots, b)
        if _continuation(rb):
            continue
        ent = by_after.get(a)
        if ent is not None and not _plain_cut(ent, keep_hard):
            continue
        k = ids.index(a)
        if any(x in heavy_at for x in (ids[k - 1] if k else None, ids[k + 1] if k + 1 < len(ids) - 1 else None)):
            continue
        cur = None
        for i, w in enumerate(words):
            hit = num(w, "hit") if isinstance(w, dict) else None
            if hit is not None and hit <= t_cut - WO_MIN_BEFORE and t_cut - hit <= WO_MAX_HIT_GAP_S:
                cur = i
        if cur is None:
            continue
        nxt = words[cur + 1] if cur + 1 < len(words) else None
        if isinstance(nxt, dict) and num(nxt, "hit") is not None and num(nxt, "hit") < t_cut - 0.1:
            continue                                                         # another word hits just before the cut: it carries the cut itself
        w = words[cur]
        if not window_fit2(w, fonts=fonts)["ok"]:
            skipped.append(f"word {cur} {str(w.get('text', ''))!r}")
            continue                                                         # more than WO_MAX_LETTERS letters: the validator would turn it into a hard cut
        hold = num(w, "hold_s", 1.6)
        d = max(0.0, t_cut + WO_OVER_S - (num(w, "hit") + hold))
        if d > WO_MAX_STRETCH_S or hold + d > HOLD_MAX_S + 1e-9:
            continue
        ln = landing(t_cut, ana)
        on_beat = ln is not None and abs(ln[2]) <= BEAT_TOL
        fat = bool(is_fat(w.get("font"), w.get("axes"))[0])
        cands.append(((0 if on_beat else 1, 0 if fat else 1, round(d, 3), abs(ln[2]) if ln else 9.0), a, b, t_cut, cur, d, hold, ln, on_beat))
    if not cands:
        if skipped:
            notes.append(f"window_other not planned: the word at the place change ({', '.join(skipped)}) has more than {WO_MAX_LETTERS} letters and cannot be 30 % of the frame tall")
        return out, notes, pins
    _, a, b, t_cut, cur, d, hold, ln, on_beat = min(cands, key=lambda x: x[0])[:9]
    w = words[cur]
    if d > 1e-9:
        w["hold_s"] = round(hold + d, 3)
    fit = window_fit2(w, fonts=fonts)
    if fit["lines"]:
        w["lines"] = fit["lines"]
    if fit["font"]:
        w["font"] = fit["font"]
        if fit["axes"]:
            w["axes"] = fit["axes"]
        else:
            w.pop("axes", None)
    if not isinstance(w.get("placement"), dict):
        w["placement"] = {}
    w["placement"]["wf"] = fit["wf"]
    if fit["hf"]:
        w["placement"]["hf"] = fit["hf"] if fit.get("roll") else max(float(w["placement"].get("hf", 0.0)), fit["hf"])
    w["placement"]["roll"] = fit.get("roll", 0.0)
    pins[cur] = round(t_cut + WO_OVER_S, 3)
    why = (f"planned by the code: word {cur} {w.get('text', '')!r} is the one holding at the {a}->{b} place change ({t_cut:g} s); its letters show {b} and "
           + ("the camera zooms into one letter until it fills the frame at " if kind == "window_zoom" else "the window grows until it fills the frame at " if kind == "window_cut" else "the hard cut lands ")
           + (f"the {ln[0]} at {ln[1]:g} s" if on_beat else "the shot boundary"))
    entry = {"after": a, "cut": kind, "word": cur, "why": why}
    if kind in DEFAULT_EXIT:
        entry["exit"] = DEFAULT_EXIT[kind]
    if a in by_after:
        out["compose"]["cuts"] = [entry if (isinstance(c, dict) and c.get("after") == a) else c for c in raw_cuts]
    else:
        if not isinstance(out.get("compose"), dict):
            out["compose"] = {}
        out["compose"]["cuts"] = list(raw_cuts) + [entry]
    fitted = f"; fitted to {fit['cap']:.0%} cap height ({'/'.join(fit['lines']) + ', ' if fit['lines'] else ''}{fit['font'] or 'own face'})" if (fit["lines"] or fit["font"]) else ""
    notes.append(f"{kind} planned after {a} ({t_cut:g} s): word {cur} {w.get('text', '')!r} hit {num(w, 'hit'):g}, hold_s {hold:g} -> {num(w, 'hold_s'):g} so its hold ends {WO_OVER_S:g} s after the cut"
                 + fitted + (f"; lands on the {ln[0]} at {ln[1]:g} s" if on_beat else "; no music mark within the beat tolerance")
                 + ("" if a not in by_after else f"; replaces the design's {by_after[a].get('cut', 'crossfade')!r}"))
    return out, notes, pins


# ================================================================================================== planning rule: a designed cut at EVERY place change
DEFAULT_3D = (CK.PIVOT,)         # then the orbit modes; depth_dolly is slow (906 s for 90 frames on a CPU): never planned, only kept when the design chose it


def full_3d_kinds(row_a=None):
    """the FULL set of designed 3D kinds a place change can take when the piece's drawn kinds are used up: straight orbit push / pullback, depth_pivot, orbit fly, depth_dolly (only when the
    dolly is allowed: DOLLY_ON), depth_lines, circle_grid. Order = the tie-break of the least-used pick."""
    out = ["orbit:push", "orbit:pullback", CK.PIVOT, "orbit:fly"]
    if DOLLY_ON:
        out.append(f"depth_dolly:{suggest_direction((row_a or {}).get('desc', ''))[0]}")
    return out + ["depth_lines", "circle_grid"]


def fill_3d(raw, shots, ana=None, keep_hard=None, system=None):
    """every place-change boundary of `raw` (a compose.cuts list) whose entry is missing / hard / crossfade gets a 3D kind: depth_pivot first, then the orbit mode the music asks for, then the other
    orbit modes; a kind (with its direction) is used once per piece (window_other / window cuts count as used boundaries, never replaced); a boundary the shot lengths cannot carry
    (feasible_3d) stays plain. Also run AFTER the validator (a window choice it downgraded to hard at a place change gets a 3D cut then). -> (new list, notes)."""
    raw, notes = list(raw), []
    by_after = {c.get("after"): i for i, c in enumerate(raw) if isinstance(c, dict)}
    used = set()
    used_n: list = []
    for c in raw:
        if isinstance(c, dict):
            kd = CK.normalize(c.get("cut"))
            if kd in SPECIAL:
                used.add(kd)
                used_n.append(kd)
    drawn = RS.kinds_of(system)
    prev_kind = None
    bnds = list(boundaries(shots))
    for bi, (a, b, t_cut) in enumerate(bnds):
        rb, ra = _row(shots, b), _row(shots, a)
        if _continuation(rb):
            prev_kind = None
            continue
        i = by_after.get(a)
        if i is not None and not _plain_cut(raw[i], keep_hard if keep_hard is not None else model_hard(raw)):
            prev_kind = CK.normalize(raw[i].get("cut")) if isinstance(raw[i], dict) else None
            continue
        nxt_kind = None                                                # the boundary after this one, when it already carries a cut of the model / the rules
        if bi + 1 < len(bnds):
            j_ = by_after.get(bnds[bi + 1][0])
            if j_ is not None and isinstance(raw[j_], dict) and not _plain_cut(raw[j_], keep_hard if keep_hard is not None else model_hard(raw)):
                nxt_kind = CK.normalize(raw[j_].get("cut"))
        mode = suggest_orbit(t_cut, ana, used_n)

        def _fits(k, ra=ra, rb=rb):
            return RS.placeable(k) and feasible_3d(k, {}, ra, rb)[0]

        def _ok(k, _fits=_fits):                                       # REPEAT RULE (2026-10-30, v028 r03 had orbit:fly on all 4 cuts): never the kind of the cut before / after, never a kind already used twice
            return k != prev_kind and k != nxt_kind and used_n.count(k) < 2
        if drawn:                                                      # the piece's DRAWN kinds first, in the system's seeded order (the orbit the music asks for first)
            pick = (RS.pick_next(system, used_n, feasible=lambda k: _fits(k) and _ok(k), prefer=f"orbit:{mode[0]}")
                    or RS.pick_next(system, used_n, feasible=lambda k: _fits(k) and k != prev_kind and used_n.count(k) < 2, prefer=f"orbit:{mode[0]}"))
            cand = []
        else:
            cand = list(dict.fromkeys([k for k in list(DEFAULT_3D) + [f"orbit:{mode[0]}"] + [f"orbit:{m}" for m in ("push", "pullback", "fly")] if k not in used]))
            pick = next((k for k in cand if feasible_3d(k, {}, ra, rb)[0] and _ok(k)), None)
        if pick is None:                                               # drawn kinds exhausted: the least used kind of the FULL 3D set
            pool = full_3d_kinds(ra)
            pool = [k for k in pool if feasible_3d(k, {}, ra, rb)[0] and used_n.count(k) < 2]
            pool_ok = [k for k in pool if k != prev_kind and k != nxt_kind] or [k for k in pool if k != prev_kind]
            pick = min(pool_ok, key=lambda k: (used_n.count(k), pool_ok.index(k))) if pool_ok else None
        if pick is None:
            notes.append(f"cuts: no 3D kind fits {a}->{b} at {t_cut:g} s (shot lengths or all kinds used); it stays a plain cut")
            continue
        was = raw[i].get("cut", "crossfade") if i is not None else None
        why = (f"planned by the code: {a}->{b} is a place change at {t_cut:g} s; "
               + (mode[1] if pick == f"orbit:{mode[0]}" else "the approved depth pivot" if (pick == CK.PIVOT and not drawn) else f"{pick} (one of the kinds drawn for this piece)" if drawn
                  else f"{pick} (the other kinds are used elsewhere in the piece)"))
        entry = {"after": a, "cut": pick, "why": why}
        if pick == "circle_grid":                                      # the word form: the dots build B's word, which the cut owns (word-less when B has no short flat word)
            entry["params"] = {"word": "auto"}
        st_ = RS.style_for(system, pick) if drawn else None
        if st_ and (st_ != "slide" or SLIDE_ON):
            entry["style"] = st_
        if i is not None:
            raw[i] = entry
        else:
            raw.append(entry)
            by_after[a] = len(raw) - 1
        used.add(pick)
        used_n.append(pick)
        prev_kind = pick
        notes.append(f"cuts: {pick} planned after {a} ({t_cut:g} s): a place change takes a designed cut" + (f" (replaces the design's {was!r})" if was else ""))
    return raw, notes


def plan_flat_window(reply, system=None):
    """PLANNING RULE (2026-10-30, "more inverse imagery inside the texts with a plain background"): when the piece's system drew an inverse window (`system.inverse.plain_window`), ONE short word
    that is not already a window gets `window_source: self` + `window_ground: plain`: its letters show the word's own footage over a flat ground in a colour of the piece's palette
    (window_cut.ground_colour). The word must be a window candidate (up to WO_MAX_LETTERS letters, a fat face fits it: window_fit2). The validator (`validate`) still fits face and box.
    -> (reply copy, notes)."""
    out = copy.deepcopy(reply) if isinstance(reply, dict) else reply
    notes: list = []
    inv = (system or {}).get("inverse") if isinstance(system, dict) else None
    if not isinstance(out, dict) or not (inv or {}).get("plain_window"):
        return out, notes
    t = out.get("type") if isinstance(out.get("type"), dict) else {}
    words = t.get("words") if isinstance(t.get("words"), list) else []
    comp = out.get("compose") if isinstance(out.get("compose"), dict) else {}
    taken = {c.get("word") for c in (comp.get("cuts") or []) if isinstance(c, dict) and isinstance(c.get("word"), int)}
    fonts = (system or {}).get("fonts")
    cand = []
    for i, w in enumerate(words):
        if not isinstance(w, dict) or i in taken or w.get("window_source") not in (None, "", "self") or w.get("window_ground"):
            continue
        if window_fit2(w, fonts=fonts)["ok"]:
            cand.append(i)
    if not cand:
        notes.append("inverse window not planned: no short word fits a window (up to six letters)")
        return out, notes
    import random
    i = random.Random(f"inverse:{(system or {}).get('seed', 0)}").choice(cand)
    words[i]["window_source"], words[i]["window_ground"] = "self", "plain"
    notes.append(f"inverse window planned: word {i} {words[i].get('text', '')!r} shows its own footage inside the letters over a flat palette ground")
    return out, notes


def plan_cuts(reply, shots, ana=None, windows=True, system=None):
    """PLANNING / SALVAGE RULE (2026-10-29, "more clear text and transitions"): every place-change boundary gets a DESIGNED cut. Plain hard cuts stay for continuations
    (`first: last:<id>`) and for shots too short for a 3D move. 1. `window_other` is planned up to wo_target(shots) times (plan_window_other: 1 up to three place changes, else 2, never on
    neighbouring boundaries). 2. Every other place-change boundary whose entry is missing / hard / crossfade gets a 3D kind: depth_pivot first, then the orbit mode the music asks for,
    then the other orbit modes; a kind (with its direction) is used once per piece; a boundary the shot lengths cannot carry (feasible_3d) stays hard. A heavy choice of the model
    (3D, pan, window) is never replaced. windows=False skips step 1 (the retry when a planned window_other hold cannot be fitted to the music). -> (reply copy, notes, pins) like plan_window_other."""
    out, notes, pins = (copy.deepcopy(reply) if isinstance(reply, dict) else reply), [], {}
    if not isinstance(out, dict) or len(shots or []) < 2:
        return out, notes, pins
    target = wo_target(shots)
    keep = model_hard(((out.get("compose") or {}).get("cuts")) if isinstance(out.get("compose"), dict) else None)
    drawn = RS.kinds_of(system)
    plannable = [k for k in ((system or {}).get("cuts", {}).get("order") or drawn) if k in ("window_other", "window_zoom", "window_cut")] if drawn else ["window_other"]
    if drawn:
        target = min(target, len(plannable))                           # a drawn piece plans the window kinds it drew (one boundary each), never one it did not
    fonts = (system or {}).get("fonts") if drawn else None
    for j in range(target if windows else 0):
        out, n_, p_ = plan_window_other(out, shots, ana, max_n=target, keep_hard=keep, kind=plannable[j % len(plannable)], fonts=fonts)
        notes += n_
        pins.update(p_)
        if not p_:
            break
    comp = out.get("compose") if isinstance(out.get("compose"), dict) else None
    raw = list(comp.get("cuts") or []) if comp and isinstance(comp.get("cuts"), list) else []
    raw, n3 = fill_3d(raw, shots, ana, keep_hard=keep, system=system)
    notes += n3
    out, n4 = plan_flat_window(out, system)
    notes += n4
    if raw:
        out.setdefault("compose", {})
        if not isinstance(out["compose"], dict):
            out["compose"] = {}
        out["compose"]["cuts"] = raw
    return out, notes, pins


# ================================================================================================== design -> piece
def to_piece(piece, design, type_dir=None):
    """copy design["compose"]["cuts"] and the words' window_source into the piece JSON (in place), then build the shots' compose ops
    (window_cut.apply_to_piece). -> notes. Pieces without window choices are left untouched."""
    from tools.fusion import window_cut
    cuts = (design.get("compose") or {}).get("cuts")
    if cuts:
        piece.setdefault("compose", {})["cuts"] = copy.deepcopy(cuts)
    for i, w in enumerate(((design.get("type") or {}).get("words")) or []):
        if w.get("window_source") and i < len((piece.get("type") or {}).get("words") or []):
            piece["type"]["words"][i]["window_source"] = w["window_source"]
            if w.get("window_ground"):
                piece["type"]["words"][i]["window_ground"] = w["window_ground"]
        if w.get("font") and i < len((piece.get("type") or {}).get("words") or []):
            piece["type"]["words"][i]["font"] = w["font"]
            if w.get("axes") is not None:
                piece["type"]["words"][i]["axes"] = w["axes"]
    from tools.fusion import pivot_cut as _pivot_cut
    own_notes = _pivot_cut.own_words(piece)                      # promoted circle_grid (word form) / depth_lines cuts own B's word: owned_by_cut + landed by the window's end
    notes = window_cut.apply_to_piece(piece, type_dir)
    return own_notes + notes + shot_cut_words(piece, design)


def shot_cut_words(piece, design):
    """the words around a 3D / pan cut follow its direction (the user-liked look, 20_open-model-design v001: A slides one way, B comes in from the other, the strands
    flow the same way): word A (the last word whose hold ends by the cut) gets `exit`, word B (the next) `entry` = the sides of cut_kinds.sides(kind), unless the
    word already has a contour port; a pan cut becomes the word's `transition` {kind: pan}. In place; -> notes."""
    notes = []
    words = (piece.get("type") or {}).get("words") or []
    dw = ((design.get("type") or {}).get("words")) or words
    cuts = (design.get("compose") or {}).get("cuts") or []
    if not cuts or len(words) < 2:
        return notes
    lay = {}
    try:
        from tools.fusion import recipe_eval as RE
        lay, _ = RE.piece_layout(piece)
    except Exception:       # noqa: BLE001
        return notes
    fps = int(piece.get("fps", 24))
    ids = [s["id"] for s in piece.get("shots", [])]
    for c in cuts:
        kd = c.get("cut")
        if kd not in SHOT_3D and not str(kd).startswith("pan:"):
            continue
        if c.get("after") not in ids or ids.index(c["after"]) + 1 >= len(ids):
            continue
        t_cut = lay[ids[ids.index(c["after"]) + 1]]["start"] / float(fps)
        wi = c.get("word") if str(kd).startswith("pan:") else _pan_word(dw, t_cut)
        if wi is None or wi + 1 >= len(words):
            notes.append(f"{kd} after {c['after']}: no word pair across the cut, words keep their own sides")
            continue
        if str(kd).startswith("pan:"):
            words[wi]["transition"] = {"kind": "pan", "dir": kd.split(":")[1], "beats": 2}
            trs = piece["type"].get("transitions")
            if isinstance(trs, list) and wi < len(trs):
                trs[wi] = None                                             # the pan replaces whatever strand transition that gap had
            notes.append(f"{kd} after {c['after']}: word {wi} -> {wi + 1} pan {kd.split(':')[1]} (needs pan_plan.weave for the plate)")
            continue
        cp = c.get("params") or {}
        sd = CK.sides(kd, c.get("style") or cp.get("style"), c.get("dir") or cp.get("dir"))
        if sd is None:
            continue
        for i, side, key in ((wi, sd[0], "exit"), (wi + 1, sd[1], "entry")):
            if isinstance(words[i].get(key), dict):
                continue
            words[i][key] = side
        notes.append(f"{kd} after {c['after']}: word {wi} exits {sd[0]}, word {wi + 1} enters from {sd[1]}")
    return notes


if __name__ == "__main__":
    print(json.dumps(schema_fragment(), indent=1))
