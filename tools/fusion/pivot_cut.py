"""pivot_cut - the 3D (depth) shot transitions as SHOT CUTS of a piece: the planner / splice side that the block `depth_pivot_chunks` listed as missing.
Image engines: `mograph/modules/transitions/depth_pivot.py` (node MOTNGRFX_DepthPivotTransition; 17_depth-dolly v013 pivot = the approved look, and the
v021-v023 camera paths push / pullback / fly through the card volume) and `tools/fusion/depth_dolly.py` (v004 directional depth-parallax dolly, stills).
This module only decides WHERE their frames go on the piece timeline and writes them into the two shots' plates (and their mattes).

Piece key (same slot as the window cuts, `compose.cuts[]`), kind = `family` or `family:direction`:
    {"after": "<shot before the boundary>", "cut": "depth_pivot" | "orbit:push" | "orbit:pullback" | "orbit:fly" | "depth_dolly:down|up|left|right",
     "params": {"chunks": 32, "travel": 36, "hold1": 12, "hold2": 24, ..}}
`params` = the module's public params (depth_pivot.PARAMS for pivot / orbit; DOLLY_DEFAULTS for the dolly; unknown keys raise); `matte_pass` (default true) also
renders the SUBJECT MATTE through the same scene, so the type sandwich keeps the subject crisp in front while the background transforms (the user-liked
look, 20_open-model-design v001). The boundary is a HARD cut for the assembler
(window_cut.owned_boundaries) because the transition frames already contain the whole change of picture.

Timeline (N = hold1 + travel + hold2 output frames, i2 = the output frame image B first becomes visible = `depth_pivot_render` info["image2_start"]):
    B plays from its own frame 0 at output frame i2, so output frame i2 sits on B's piece start. Output frame 0 sits at piece frame B.start - i2.
    A shot (before): its local frames [a0, a0 + i2) are replaced by output frames [0, i2)   (a0 = B.start - i2 - A.start; A must have >= i2 frames)
    B shot (after):  its local frames [0, N - i2)   are replaced by output frames [i2, N)    (B must have >= N - i2 frames)
    The frames of A / B outside those ranges are untouched, so frame 0 of the transition equals A's clip frame exactly and the last equals B's.
The replaced frames are DEPTH-SCENE frames at `short` px (432 default) scaled back to the clip size. The shots' subject mattes follow them: `matte_pass` (default) renders the same
scene on matte-coloured layers (`depth_orbit.py --matte-pass`) and `apply_cut` returns that as the patch for the replaced frames. Where no transition matte exists for a replaced frame
(matte_pass off, or the engine returned none) the patch is ZERO there: the shot's own matte describes the old picture, never the transformed one (a matte over a picture it was not cut from
shows two people), so inside `window_ranges` the type sandwich sees no subject.

    normalize(kind)                    -> canonical kind or None ("pivot" -> "depth_pivot", "dolly:down" -> "depth_dolly:down", ...)
    specs(piece)                       -> [{k, a, b, kind, params}] the 3D cuts of a piece
    image2_start(params, kind)         -> int, without rendering
    plan_cut(piece, spec)              -> dict(a, b, kind, n, i2, start, a0, b_len, window_a, window_b, problems [..])   (pure arithmetic)
    splice(a_u8, b_u8, frames, plan)   -> (a_new, b_new) uint8 copies
    apply_cut(piece, spec, clips, mattes=None, depths=None, render=None) -> CutResult(a, b, plan, matte_a, matte_b)   renders (CPU) and splices
    patch_matte(matte, patch)          -> the shot's matte with the transition's matte written in (compose_clips does this from clip["matte_patch"])
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph.modules.transitions import depth_pivot as DP      # noqa: E402

from tools.fusion import cut_kinds as CK      # noqa: E402

KIND = CK.PIVOT
ORBIT_MODES, DOLLY_DIRS, KINDS = CK.ORBIT_MODES, CK.DOLLY_DIRS, CK.SHOT_3D
DOLLY_DEFAULTS = dict(depth_ratio=6.0, lag=0.7, overshoot=0.08, settle=18, cell=28.0, seed=0, spring=True, delay_mode="random", subject_relief=0.45,
                      travel=29, hold1=12, hold2=None, short=432)       # v004's settings; hold2 None = settle + 12 (spring at rest before the end)


def normalize(kind) -> str | None:
    """a cut kind (any spelling) -> the canonical 'depth_pivot' | 'orbit:<mode>' | 'depth_dolly:<dir>', or None when it is not a complete 3D shot cut
    (a bare 'orbit' / 'depth_dolly' has no direction yet; pan and window kinds are other families)."""
    k = CK.normalize(kind)
    return k if (k in CK.SHOT_3D or k in CK.EXPLORE) else None


MIN_DEPTH_SIDE = 64      # a depth map smaller than this is the footage nodes' 16x16 zero placeholder (nodes_fusion/footage.py _placeholder), never a real depth reference


def depth_hw(d):
    """(H, W) of a depth input (tensor or array: (H,W) | (H,W,c) | (T,H,W) | (T,H,W,c)), or None when it is None / has no spatial shape."""
    if d is None:
        return None
    s = tuple(int(x) for x in getattr(d, "shape", ()))
    if len(s) == 4:
        return s[1:3]
    if len(s) == 3:
        return s[:2] if s[-1] in (1, 3, 4) else s[1:3]
    return s if len(s) == 2 else None


def is_placeholder_depth(d) -> bool:
    """True when `d` is a depth input but smaller than MIN_DEPTH_SIDE px on a side (the 16x16 zero placeholder): not a depth estimate, treat it as missing."""
    hw = depth_hw(d)
    return hw is not None and min(hw) < MIN_DEPTH_SIDE


def usable_depth(d):
    """`d`, or None when it is a placeholder (depth_pivot's CPU Depth-Anything fallback then runs, depth_pivot._layer)."""
    return None if is_placeholder_depth(d) else d


def label_depth(info: dict, placeholder_a: bool, placeholder_b: bool) -> dict:
    """record where each side's depth came from in the cut info["depth"] = [a, b]: 'given' (a real depth reached the cut; also the default when the engine reports none),
    'fallback' (the input was a placeholder and was passed on as None, so the CPU Depth-Anything estimate ran; every engine path estimates on None), 'placeholder' (it was a
    placeholder yet the engine still reports 'given': the guard was bypassed). Edits and returns `info`."""
    eng = list(info.get("depth") or [])
    out = []
    for i, ph in enumerate((placeholder_a, placeholder_b)):
        e = eng[i] if i < len(eng) else None
        out.append(("placeholder" if e == "given" else "fallback") if ph else (e or "given"))
    info["depth"] = out
    return info


def family(kind: str) -> str:
    if kind in CK.EXPLORE:
        return "explore"
    return "dolly" if kind.startswith("depth_dolly") else "orbit" if kind.startswith("orbit") else "pivot"


def specs(piece: dict) -> list[dict]:
    """the 3D cuts of a piece: compose.cuts[] entries whose `cut` is a pivot / orbit / dolly kind at a real shot boundary."""
    ids = [s["id"] for s in piece.get("shots", [])]
    out = []
    for c in ((piece.get("compose") or {}).get("cuts") or []):
        kind = normalize(c.get("cut"))
        if kind is None or c.get("after") not in ids:
            continue
        k = ids.index(c["after"])
        if k + 1 < len(ids) and all(s["k"] != k for s in out):
            prm = dict(c.get("params") or {})
            for key in ("style", "dir"):                      # `style` / `dir` may sit on the cut itself or in its params (the design validator keeps `params`)
                if c.get(key) and key not in prm:
                    prm[key] = c[key]
            out.append(dict(k=k, a=ids[k], b=ids[k + 1], kind=kind, params=prm, split=bool(c.get("split") or prm.pop("split", False))))
    return out


def may_get_depth_cuts(piece: dict) -> bool:
    """True when the piece is still UNDESIGNED (no `compose.cuts` list yet: the plan piece the footage / layer gathers receive; the design reads the depth THEY produce, so the cuts cannot exist
    yet) and its drawn cut kinds (plan.piece_system.cuts.kinds) allow a depth-reading 3D cut (pivot / orbit / dolly / depth_lines) at one of its shot boundaries; no drawn kinds = the code's
    default pivot may be planned. 2026-10-30: the gather's `shot_needs` saw no cuts and emitted the 16x16 placeholder depth for every shot."""
    if len(piece.get("shots", [])) < 2 or "cuts" in (piece.get("compose") or {}):
        return False
    kinds = (((piece.get("plan") or {}).get("piece_system") or {}).get("cuts") or {}).get("kinds")
    if not kinds:
        return True
    return any(normalize(k) not in (None, "circle_grid") for k in kinds if isinstance(k, str))


TYPE_KEYS = ("type", "type_depth")        # the depth engine can draw words as planes of its own scene; a SHOT CUT never does (below)


def _no_type(raw: dict) -> dict:
    """the cut's params without the engine's own type planes. The words of a piece are a compose layer between the transition's background and its subject (the matte
    through the transition occludes them); if they were drawn into the transition's pixels they would be chunked, blurred and doubled with the background."""
    return {k: v for k, v in raw.items() if k not in TYPE_KEYS}


def _dolly_params(params: dict | None):
    """the public dolly params (DOLLY_DEFAULTS + overrides, unknown keys raise) -> (dict, depth_dolly.Params with the first-only holds)."""
    given = dict(params or {})
    bad = [k for k in given if k not in DOLLY_DEFAULTS and k not in ("matte_pass", "dir", "style")]
    if bad:
        raise KeyError(f"depth_dolly cut: unknown parameter(s) {bad}; known: {sorted(DOLLY_DEFAULTS)}")
    d = {k: given.get(k, v) for k, v in DOLLY_DEFAULTS.items()}
    from tools.fusion import depth_dolly as dd
    p = dd.Params(depth_ratio=float(d["depth_ratio"]), ease_travel=int(d["travel"]), lag=float(d["lag"]), overshoot=float(d["overshoot"]),
                  settle=int(d["settle"]), spring=bool(d["spring"]), loop=False, delay_mode=str(d["delay_mode"]), cell=float(d["cell"]),
                  seed=int(d["seed"]), subject_relief=float(d["subject_relief"]))
    p.hold_start = int(d["hold1"])
    p.hold_end = int(d["hold2"]) if d["hold2"] is not None else (p.settle if p.spring else 0) + 12
    d["hold2"] = p.hold_end
    return d, p


def _dolly_frame_count(p) -> int:
    from tools.fusion import depth_dolly as dd
    return len(dd.frames(p, 2, False))


def _dolly_i2(p) -> int:
    """output frame where B takes over: the middle of the travel window (A's frame is frozen before, B's still lands at the end)."""
    from tools.fusion import depth_dolly as dd
    return int(p.hold_start + dd.window(p) // 2)


def image2_start(params: dict | None = None, kind: str = KIND) -> int:
    """first output frame at which image B takes over, from the engine's own weights / ease (no scene, no render). pivot / orbit: the first frame B is
    visible (= `depth_pivot_render` info["image2_start"]); dolly: the middle of the travel window."""
    kind = normalize(kind) or KIND
    if family(kind) == "explore":
        return explore_window_frames(kind, params)[1]
    if family(kind) == "dolly":
        return _dolly_i2(_dolly_params(params)[1])
    dd, do = DP._engine()
    p = DP.resolve_params(dict(params or {}, **({"mode": kind.split(":")[1]} if family(kind) == "orbit" else {})))
    op = DP._op(do, p["engine"])
    from tools.fusion import depth_smooth as ds
    if ds.uses(op):                                       # smooth / slide: B plays from the q the layers are on screen (ds.I2_Q)
        import dataclasses
        return int(ds.image2_start(dataclasses.replace(op, spring=False)))
    sc = object.__new__(do.PivotScene if op.mode == "pivot" else do.VolumeScene)         # only .op .Phi (.kind) are read by image2_start / q / weights
    sc.op, sc.kind = op, op.mode
    sc.Phi = (math.radians(op.in_angle) / op.overlap) if getattr(op, "overlap", 0) else math.radians(op.out_angle + op.in_angle)
    return int(sc.image2_start())


def window_frames(kind: str, params: dict | None = None) -> tuple[int, int]:
    """(n, i2): the transition's frame count and the output frame B takes over at, for a kind + params (raises KeyError on an unknown param)."""
    kind = normalize(kind) or KIND
    if family(kind) == "explore":
        return explore_window_frames(kind, params)
    raw = _no_type({k: v for k, v in dict(params or {}).items() if k != "matte_pass"})
    if family(kind) == "dolly":
        p, dp = _dolly_params(raw)
        return _dolly_frame_count(dp), _dolly_i2(dp)
    p = DP.resolve_params(dict(raw, **({"mode": kind.split(":")[1]} if family(kind) == "orbit" else {})))
    return int(p["hold1"] + p["travel"] + p["hold2"]), image2_start(p, kind)


def _settle_frame(fam: str, p) -> int:
    """output frame at which the camera move has settled (the end of the eased travel): the subject of a plate shot glitches in from here."""
    if fam == "dolly":
        from tools.fusion import depth_dolly as dd
        return int(p.hold_start + dd.window(p))
    return int(p["hold1"] + p["travel"])



# ================================================================================================== promoted explore cuts: circle_grid (+ word form) and depth_lines (2026-10-30)
# Rendered by tools/fusion/explore_cuts2.py through apply_cut(render=make_render(...)). The window starts so that the word LANDS on a beat of the piece grid, the blink rhythm
# is snapped to the half-beats of that grid, and B's word is OWNED BY THE CUT: the type stage does not draw it inside the window (gate_windows, recipe_eval) and picks it up after.
MAX_WORD_LETTERS = 12            # one line at 80 % of the frame width: longer texts get unreadably small, the cut then runs word-less (circle_grid) or is skipped (depth_lines)
CAP_PER_TARGET = 3               # depth_lines: at most this many strokes per target outline (the horse scribble)
EXPLORE_DEFAULTS = {"depth_lines": dict(direct=True, line_w=0.006, var=1.25, cap=CAP_PER_TARGET, morph=True),   # "r3 direct" look (user-liked) + 2026-10-30 outline morph: the image outlines go straight onto the letter outlines, no free-lines stage
                    "circle_grid": dict(snap=True)}           # circle_grid word form = the user-approved lamp render (cols 9, word_cols 30, beat order, no outline), locked: only `snap` + the beat phase are added


def explore_style(kind: str, params: dict | None = None) -> str:
    """the explore_cuts2 style of a promoted kind: depth_lines -> depth_lines_window; circle_grid -> circle_grid_word when params.word is set, else circle_grid."""
    return "depth_lines_window" if kind == "depth_lines" else ("circle_grid_word" if (params or {}).get("word") else "circle_grid")


def beat_grid(piece: dict):
    """(bpm, first beat s) of the piece: audio.bpm else type.bpm else 120; first beat = type.beats[0] (0 without)."""
    bpm = None
    for src in ((piece.get("audio") or {}).get("bpm"), (piece.get("type") or {}).get("bpm"), (piece.get("plan") or {}).get("bpm")):
        try:
            if src and float(src) > 20:
                bpm = float(src)
                break
        except (TypeError, ValueError):
            pass
    beats = (piece.get("type") or {}).get("beats") or []
    try:
        off = float(beats[0]) if beats else 0.0
    except (TypeError, ValueError, IndexError):
        off = 0.0
    return (bpm or 120.0), off


def b_word(piece: dict, t_cut: float, window_s: float):
    """(index, word) of B's word: the first word that starts at/after the cut window opens (minus a quarter second) and still runs past the cut; else the first word that runs past the cut; (None, None) without one."""
    ws = ((piece.get("type") or {}).get("words")) or []
    cands = [(i, w) for i, w in enumerate(ws) if isinstance(w, dict) and w.get("t0") is not None and w.get("t1") is not None and w.get("text")]
    for i, w in cands:
        if float(w["t0"]) >= window_s - 0.25 and float(w["t1"]) > t_cut:
            return i, w
    for i, w in cands:
        if float(w["t1"]) > t_cut:
            return i, w
    return None, None


def word_ink_bgr(piece: dict, w: dict):
    """the designed colour of a word as BGR floats: word.color (RGB 0-255 / hex) else the type palette's fg (RGB); None without one."""
    c = w.get("color")
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        c = [int(c[k:k + 2], 16) for k in (1, 3, 5)]
    if not (isinstance(c, (list, tuple)) and len(c) >= 3):
        c = ((piece.get("type") or {}).get("palette") or {}).get("fg")
    if not (isinstance(c, (list, tuple)) and len(c) >= 3):
        return None
    v = [float(x) for x in c[:3]]
    if max(v) <= 1.0:
        v = [x * 255.0 for x in v]
    return [v[2], v[1], v[0]]


def word_font(piece: dict, w: dict) -> str:
    """the liked face file of the word (word.font, else the first of the piece's faces, else a liked heavy reader). Only fonts_liked resolves a face."""
    from mograph import fonts_liked as FL
    for f in [w.get("font")] + list(((piece.get("type") or {}).get("fonts")) or []) + list(((piece.get("plan") or {}).get("fonts")) or []):
        if isinstance(f, str) and f and FL.is_liked(f):
            return FL.font_file(f)
    return FL.font_file(FL.families("heavy")[0])


def stage_word(w: dict, text: str, font: str) -> dict:
    """the word as the type stage lands it, for explore_cuts2.stage_layout: text, liked face file, axes, the rows (`lines`) and the roll of its layout (stable_type: line / stack / rot90 / grid / column).
    A text that differs from the designed word (params.word) drops the designed rows."""
    st = dict(text=text, font=font, axes=w.get("axes"), placement=dict(roll=float((w.get("placement") or {}).get("roll") or 0.0)))
    if w.get("lines") and " ".join(str(w.get("text") or "").split()) == " ".join(str(text).split()):
        st["lines"] = list(w["lines"])
    return st


RADIAL_SHARE = 0.5                 # word-less circle_grid with a subject matte on A: this share of the cuts orders the dots radially out from the subject (user-liked r2_cg_radial_horse), else on the beat
RADIAL_PARAMS = dict(order="radial", cols=8, outline=False)       # as the round-2 radial render (r2_cg_radial_horse.json); timing is the same 64 / 24 frames


def radial_order(piece: dict, plan: dict, ma) -> dict:
    """{} or RADIAL_PARAMS: the per-cut seeded draw (piece type seed + the cut's index) of the ORDER VARIANT of the word-less circle_grid. Only when A has a subject matte at the window's start
    (the dots radiate from its centroid) and the spec did not set `order` itself; circle_grid_word (the approved lamp look) is never touched."""
    prm = plan.get("params") or {}
    if plan.get("kind") != "circle_grid" or prm.get("style") != "circle_grid" or "order" in prm or ma is None:
        return {}
    m = _matte_sl(ma, plan["a0"], plan["a0"] + 1)
    if m is None or not len(m) or float(m[0].max()) <= 0.5:
        return {}
    import random
    rnd = random.Random(f"radial:{(piece.get('type') or {}).get('seed', 0)}:{plan.get('k', 0)}")
    return dict(RADIAL_PARAMS) if rnd.random() < RADIAL_SHARE else {}


def explore_plan(piece: dict, spec: dict, lay: dict, params: dict | None = None) -> dict:
    """everything the promoted cut needs, no rendering: dict(style, params (explore_cuts2 params), n, i2, owned (word index | None), word, problems [..], t_lo, t_hi)."""
    from tools.fusion import explore_cuts2 as E
    kind = normalize(spec.get("kind")) or "depth_lines"
    raw = dict(spec.get("params") or {}, **(params or {}))
    for k in ("matte_pass", "style", "dir_"):
        raw.pop(k, None)
    want_word = CK.owns_word(kind, raw)
    wtxt = raw.pop("word", None)
    base = dict(EXPLORE_DEFAULTS.get(kind, {}))
    base.update(raw)
    if kind == "depth_lines":
        base["morph"] = True                                     # LOCKED (user 2026-10-30: "these are great ... this transition should be added"): the outline morph IS the depth_lines cut, the free-lines form is gone
    problems = []
    fps = float(piece.get("fps", 24))
    lb = lay[spec["b"]]
    bpm, off = beat_grid(piece)
    base.setdefault("bpm", bpm * 24.0 / fps)                         # explore_cuts2 counts beats in 24 fps frames
    t_cut = int(lb["start"]) / fps
    style = explore_style(kind, {"word": want_word})
    i2 = E.timeline(style, dict(E.DEFAULTS[style], **{k: v for k, v in base.items() if k in E.DEFAULTS[style]}))["i2"]
    w0 = int(lb["start"]) - i2
    bf = 24.0 * 60.0 / float(base["bpm"])
    base["phase"] = float((w0 - off * fps) % bf)                      # frames after a beat of the piece grid at which the window starts
    owned = None
    w = None
    if want_word:
        widx, w = b_word(piece, t_cut, w0 / fps)
        if w is None:
            problems.append(f"{kind}: no word of B ({spec['b']}) to form")
        else:
            txt = str(w["text"]).strip()
            letters = len(txt.replace(" ", ""))
            if letters > MAX_WORD_LETTERS or "\n" in txt:
                problems.append(f"{kind}: B's word {txt!r} has {letters} letters (> {MAX_WORD_LETTERS}): one line would be too small")
            elif str(w.get("render") or "flat") != "flat":
                problems.append(f"{kind}: B's word {txt!r} is owned by {w.get('render')!r}, the cut can only form a flat word")
            else:
                owned = widx
                if kind == "depth_lines":                              # B plays INSIDE the letters: a thin / fragmented face (Rock3D) leaves hairlines with nothing to show (window forms need a heavy face)
                    from tools.fusion import owner as _OW
                    face = word_font(piece, w)
                    if _OW.stroke_class(face, w.get("axes")) == "thin":
                        owned = None
                        problems.append(f"depth_lines: B's word {txt!r} is set in {face}, too thin to hold the next shot inside its letters")
                base["word"] = str(wtxt) if isinstance(wtxt, str) and wtxt not in ("auto", "B", "b") else txt
                base.setdefault("font", word_font(piece, w))
                if w.get("axes") is not None:
                    base.setdefault("axes", w["axes"])
                base.setdefault("stage", stage_word(w, base["word"], os.path.basename(str(base["font"]))))   # the type stage's own layout of this word (rows, roll, cap, face): no jump at the hand-over
                ink = word_ink_bgr(piece, w)
                if ink is not None and style != "depth_lines_window":
                    base.setdefault("ink", ink)
        if kind == "circle_grid" and owned is None:                  # word form impossible -> the word-less grid
            style = "circle_grid"
            problems = []
            for k in ("word", "font", "axes", "ink", "word_cols", "wf"):
                base.pop(k, None)
            i2 = E.timeline(style, dict(E.DEFAULTS[style]))["i2"]
            w0 = int(lb["start"]) - i2
            base["phase"] = float((w0 - off * fps) % bf)
    if kind == "depth_lines":
        try:
            from tools.fusion import window_cut as WC
            g = WC.ground_colour(piece, "plain")                    # the flat ground the word keeps after the cut (window_ground): the same colour inside the cut
            base.setdefault("ground", [g[2] * 255.0, g[1] * 255.0, g[0] * 255.0])
        except Exception:   # noqa: BLE001
            pass
    ok = {k: v for k, v in base.items() if k in E.DEFAULTS[style]}
    n, i2 = E.window_frames(style, ok)
    return dict(style=style, params=ok, n=int(n), i2=int(i2), owned=owned, word=ok.get("word"), problems=problems,
                t_lo=(int(lb["start"]) - int(i2)) / fps, t_hi=(int(lb["start"]) + (int(n) - int(i2))) / fps)


def owned_cuts(piece: dict) -> list:
    """one dict per promoted cut that owns B's word and can run: spec, plan (explore_plan), t_lo, t_hi, word index."""
    out = []
    try:
        from tools.fusion import recipe_eval as RE
        lay, _ = RE.piece_layout(piece)
    except Exception:   # noqa: BLE001
        return out
    for sp in specs(piece):
        if sp["kind"] not in CK.EXPLORE:
            continue
        try:
            xp = explore_plan(piece, sp, lay)
        except (KeyError, ValueError):
            continue
        la, lb = lay[sp["a"]], lay[sp["b"]]
        a0 = int(lb["start"]) - xp["i2"] - int(la["start"])
        if xp["owned"] is None or xp["problems"] or a0 < 0 or xp["n"] - xp["i2"] > int(lb["frames"]) or a0 + xp["i2"] > int(la["frames"]):
            continue
        out.append(dict(spec=sp, plan=xp, t_lo=xp["t_lo"], t_hi=xp["t_hi"], word=xp["owned"]))
    return out


def gate_windows(piece: dict) -> list:
    """[[lo_s, hi_s]] piece-second windows in which the type stage draws NOTHING because a promoted cut owns B's word (the cut's pixels contain it)."""
    return [[round(g["t_lo"], 4), round(g["t_hi"], 4)] for g in owned_cuts(piece)]


def own_words(piece: dict) -> list:
    """PLAN-TIME hook (design_options.to_piece, before window_cut.apply_to_piece): mark B's word of every owning promoted cut `owned_by_cut` {cut, kind, t0, t1} and make sure it is LANDED when
    the cut hands it back: its form-in starts inside the window (t0 clamped to [t_lo, t_hi - form_s - 2 frames]) and it runs a beat past the window. depth_lines also sets
    window_source self + window_ground (the flat ground of the cut) + window_t from the window's end, so the word keeps its look (B inside the letters on the ground) after the cut. In place; -> notes."""
    notes = []
    words = (piece.get("type") or {}).get("words") or []
    fps = float(piece.get("fps", 24))
    for oc in owned_cuts(piece):
        w = words[oc["word"]]
        sp, lo, hi = oc["spec"], oc["t_lo"], oc["t_hi"]
        fs = float(w.get("form_s") or 0.6)
        w["t0"] = round(min(max(float(w["t0"]), lo), max(lo, hi - fs - 2.0 / fps)), 4)
        beat = 60.0 / beat_grid(piece)[0]
        if float(w["t1"]) < hi + beat:
            w["t1"] = round(hi + beat, 4)
        w["owned_by_cut"] = dict(cut=f"{sp['a']}->{sp['b']}", kind=sp["kind"], t0=round(lo, 4), t1=round(hi, 4))
        if sp["kind"] == "depth_lines":
            w["window_source"], w["window_t"] = "self", [round(hi, 4), float(w["t1"])]
            g = oc["plan"]["params"].get("ground")
            w["window_ground"] = [round(g[2]), round(g[1]), round(g[0])] if g else "plain"
        notes.append(f"{sp['kind']} {sp['a']}->{sp['b']}: word {oc['word']} {w.get('text')!r} owned by the cut {lo:.2f}-{hi:.2f} s, the type stage picks it up at {hi:.2f} s")
    return notes


def clip_holds_to_cuts(piece: dict, holds: list) -> list:
    """word holds [(hit_s, release_s)] (one per word) with a cut-owned word's hit moved to the end of its cut window: the legibility guard / hold net must not redraw it inside the window."""
    ws = (piece.get("type") or {}).get("words") or []
    out = list(holds)
    for i, w in enumerate(ws):
        oc = w.get("owned_by_cut") if isinstance(w, dict) else None
        if oc and i < len(out):
            a, b = out[i]
            out[i] = (max(a, float(oc["t1"])), max(b, float(oc["t1"])))
    return out


def cut_windows_s(piece: dict, fps: float | None = None) -> list:
    """[(lo_s, hi_s)] piece-second windows of every 3D / explore shot cut of the piece (plan_cut: start .. start + n); a cut whose plan cannot be made is skipped."""
    fps = float(fps or piece.get("fps") or 24)
    out = []
    for sp in specs(piece):
        try:
            pl = plan_cut(piece, sp)
        except Exception:    # noqa: BLE001
            continue
        out.append((int(pl["start"]) / fps, (int(pl["start"]) + int(pl["n"])) / fps))
    return out


def holds_outside_cuts(piece: dict, holds: list, fps: float | None = None) -> list:
    """compose.holds [[t_s, dur_s], ..] (freeze-frames) without the ones that touch a 3D shot cut's window. 2026-10-30 (user on ship #10 f175-193: "why does it stop the video ... the transition
    should resolve with the contour lines drawing at the same time"): the designed freeze [7.29 s, 0.75 s] sat inside the s2 -> s3 depth_pivot window (f140-211) and stopped B's footage, the
    contour sweep and the type dead for 18 frames, then snapped back. Never still through a transition: such a hold is dropped, not moved."""
    if not holds:
        return []
    wins = cut_windows_s(piece, fps)
    if not wins:
        return list(holds or [])
    return [h for h in (holds or []) if not any(float(h[0]) < hi and lo < float(h[0]) + float(h[1]) for lo, hi in wins)]


def apply_gate(frame, spec: dict, t: float):
    """a type-pass frame (numpy / torch HxW or HxWx3) with nothing drawn at piece time t inside a cut-owned window: zero (matte / line / depth), the pass background colour for the beauty pass (spec['bg'] 0..1)."""
    k = 1.0                                   # a window [lo, hi, fade_out, fade_in] (tools/fusion/word_bridge.py) ramps: the type fades out from lo, back in up to hi
    for wn in spec["win"]:
        lo, hi = wn[0], wn[1]
        if lo <= t < hi:
            fo, fi = (float(wn[2]), float(wn[3])) if len(wn) >= 4 else (0.0, 0.0)
            g = max(1.0 - (t - lo) / fo if fo > 0 else 0.0, 1.0 - (hi - t) / fi if fi > 0 else 0.0, 0.0)
            k = min(k, g)
    if k >= 1.0:
        return frame
    bg = spec.get("bg")
    if bg is not None and getattr(frame, "ndim", 0) == 3:
        b = frame.new_tensor(bg) if hasattr(frame, "new_tensor") else np.asarray(bg, np.float32)
        return b + (frame - b) * k
    return frame * k


def explore_window_frames(kind: str, params: dict | None = None) -> tuple[int, int]:
    """(n, i2) of a promoted kind from explore_cuts2.window_frames (no piece: phase 0 = the window starts on a beat)."""
    from tools.fusion import explore_cuts2 as E
    k = normalize(kind) or kind
    prm = dict(EXPLORE_DEFAULTS.get(k, {}), **{a: b for a, b in dict(params or {}).items() if a not in ("matte_pass", "style")})
    style = explore_style(k, prm)
    return E.window_frames(style, {a: b for a, b in prm.items() if a in E.DEFAULTS[style]})


def _plan_explore(piece: dict, spec: dict, kind: str, params: dict | None) -> dict:
    """plan_cut of a promoted kind (circle_grid / depth_lines): timing from explore_cuts2 with the piece's beat grid, B's word owned by the cut when the kind forms one."""
    from tools.fusion import recipe_eval as RE
    lay, _ = RE.piece_layout(piece)
    la, lb = lay[spec["a"]], lay[spec["b"]]
    xp = explore_plan(piece, dict(spec, kind=kind), lay, params)
    n, i2 = xp["n"], xp["i2"]
    start = int(lb["start"]) - i2
    a0 = start - int(la["start"])
    b_len = n - i2
    problems = list(xp["problems"])
    if a0 < 0:
        problems.append(f"shot {spec['a']} starts {-a0} frames too late: the transition needs {i2} frames of it before {spec['b']}")
    if a0 + i2 > int(la["frames"]):
        problems.append(f"shot {spec['a']} has {la['frames']} frames, the transition reads up to {a0 + i2}")
    if b_len > int(lb["frames"]):
        problems.append(f"shot {spec['b']} has {lb['frames']} frames, the transition plays {b_len} of it")
    prm = dict(xp["params"], style=xp["style"])
    return dict(a=spec["a"], b=spec["b"], k=spec["k"], kind=kind, n=n, i2=i2, start=start, a0=a0, b_len=b_len, params=prm, matte_pass=True, split=False, settle=i2,
                window_a=[int(la["start"]) + a0, int(la["start"]) + a0 + i2], window_b=[int(lb["start"]), int(lb["start"]) + b_len],
                owned_word=xp["owned"], owned_t=[xp["t_lo"], xp["t_hi"]], problems=problems)


def plan_cut(piece: dict, spec: dict, params: dict | None = None) -> dict:
    """where the transition's frames go. `problems` lists everything that makes the cut impossible (the caller falls back to a plain hard cut)."""
    from tools.fusion import recipe_eval as RE
    kind = normalize(spec.get("kind", KIND)) or KIND
    fam = family(kind)
    if fam == "explore":
        return _plan_explore(piece, spec, kind, params)
    raw = _no_type(dict(spec.get("params") or {}, **(params or {})))
    matte_pass = bool(raw.pop("matte_pass", True))
    if fam == "dolly":
        p, dp = _dolly_params(dict(raw))
        n, i2 = _dolly_frame_count(dp), _dolly_i2(dp)
        settle = _settle_frame(fam, dp)
    else:
        p = DP.resolve_params(dict(raw, **({"mode": kind.split(":")[1]} if fam == "orbit" else {})))
        n, i2 = int(p["hold1"] + p["travel"] + p["hold2"]), image2_start(p, kind)
        settle = _settle_frame(fam, p)
    lay, _ = RE.piece_layout(piece)
    la, lb = lay[spec["a"]], lay[spec["b"]]
    start = int(lb["start"]) - i2
    a0 = start - int(la["start"])
    b_len = n - i2
    problems = []
    if a0 < 0:
        problems.append(f"shot {spec['a']} starts {-a0} frames too late: the transition needs {i2} frames of it before {spec['b']} (hold1 {p['hold1']}, travel {p['travel']})")
    if a0 + i2 > int(la["frames"]):
        problems.append(f"shot {spec['a']} has {la['frames']} frames, the transition reads up to {a0 + i2}")
    if b_len > int(lb["frames"]):
        problems.append(f"shot {spec['b']} has {lb['frames']} frames, the transition plays {b_len} of it (hold2 {p['hold2']} too long)")
    return dict(a=spec["a"], b=spec["b"], k=spec["k"], kind=kind, n=n, i2=i2, start=start, a0=a0, b_len=b_len, params=p, matte_pass=matte_pass, split=bool(spec.get("split")),
                settle=settle,
                window_a=[int(la["start"]) + a0, int(la["start"]) + a0 + i2], window_b=[int(lb["start"]), int(lb["start"]) + b_len],
                problems=problems)


def window_ranges(plan: dict) -> dict:
    """piece-frame ranges [lo, hi) where the shots' own mattes no longer match the picture: {sid: [lo, hi]}. With matte_pass the spliced mattes
    (CutResult.matte_a / matte_b, clip["matte_patch"]) replace them there."""
    return {plan["a"]: list(plan["window_a"]), plan["b"]: list(plan["window_b"])}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _fit(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    if frames.shape[1:3] == (h, w):
        return frames
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if f.shape[0] > h else cv2.INTER_CUBIC) for f in frames])


def splice(a_u8, b_u8, frames, plan: dict):
    """write the transition frames into copies of the two clips (uint8 [T,H,W,3], numpy or torch -> numpy). Raises when the plan has problems."""
    if plan["problems"]:
        raise ValueError(f"{plan['kind']} cut impossible: " + "; ".join(plan["problems"]))
    a, b, fr = _np(a_u8)[..., :3].copy(), _np(b_u8)[..., :3].copy(), np.asarray(frames)
    if len(fr) != plan["n"]:
        raise ValueError(f"{plan['kind']} cut: {len(fr)} frames, the plan has {plan['n']}")
    i2, a0 = plan["i2"], plan["a0"]
    a[a0:a0 + i2] = _fit(fr[:i2], *a.shape[1:3])
    b[:plan["b_len"]] = _fit(fr[i2:], *b.shape[1:3])
    return a, b


def splice_window(base, new, lo: int, hi: int):
    """copy of `base` (uint8 [T,H,W,3], numpy or torch -> numpy) with frames [lo:hi) taken from `new` (resized to base's frame size when they differ). A three-layer shot's
    TEXT clip (u8) keeps every frame outside the cut's window, so an LTX word that ends before the window is not thrown away with the rest of the shot."""
    b = _np(base)[..., :3].copy()
    n = _np(new)[..., :3]
    b[lo:hi] = _fit(n[lo:hi], *b.shape[1:3])
    return b


def type_clearance(piece: dict, plan: dict) -> list[dict]:
    """how each word of the piece meets the cut's two windows (piece seconds). The transition frames never contain type (pivot_cut drops the engine's own type planes;
    the words are a compose layer between the transition's picture and its subject, occluded by the matte through the transition).
      rides        a flat word live in a window: its own layer, only its own motion changes
      exits_in     a flat word whose exit (`exit_s` before `t1`) runs inside either window: its exit animation (particles / wipe / flicker ...) is the only smear left
      conflict     an `ltx` / `handoff` word live in a window: its letters are baked into the shot's text clip, which the window replaces by the clean transition frames
                   -> that word must be `flat` across the cut or end before the window
      outside      not live in either window
      owned_by_cut B's word of a promoted cut that forms it (circle_grid word form, depth_lines): the cut's pixels contain it and the type stage draws nothing inside the cut's window
                   (recipe_eval gates the type passes there) and picks the word up after it; wins over every other state of that word"""
    fps = float(piece.get("fps", 24))
    owned = plan.get("owned_word")
    wa = [x / fps for x in plan["window_a"]]
    wb = [x / fps for x in plan["window_b"]]
    out = []
    for i, w in enumerate(((piece.get("type") or {}).get("words")) or piece.get("words") or []):
        if not isinstance(w, dict) or w.get("t0") is None or w.get("t1") is None:
            continue
        t0, t1 = float(w["t0"]), float(w["t1"])
        ex = float(w.get("exit_s") or 0.0)
        live = [name for name, (lo, hi) in (("a", wa), ("b", wb)) if t0 < hi and t1 > lo]
        owner = str(w.get("render") or "flat")
        if owned is not None and i == owned:
            state = "owned_by_cut"
            live = live or ["b"]
        elif not live:
            state = "outside"
        elif owner in ("ltx", "handoff"):
            state = "conflict"
        elif any((t1 - ex) < hi and t1 > lo for lo, hi in (wa, wb)):          # the exit animation itself runs inside a window
            state = "exits_in"
        else:
            state = "rides"
        out.append(dict(word=i, text=w.get("text"), state=state, live=live, render=owner, t0=t0, t1=t1, exit_from=round(t1 - ex, 3), window_a=wa, window_b=wb))
    return out


def _matte_sl(m, lo, hi):
    """matte clip slice [lo:hi] as float (k,H,W) or None."""
    if m is None:
        return None
    a = _np(m)
    a = a[..., 0] if a.ndim == 4 else a
    a = a[lo:hi].astype(np.float32)
    return a / 255.0 if a.size and a.max() > 1.5 else a


def patch_matte(matte, patch, footage_hw=None):
    """matte (numpy / torch [T,H,W(,C)], 0..1 or uint8) with `patch` = (lo, float32 (k,H,W) 0..1) written at frames lo..lo+k; same type back.
    footage_hw = (H, W) of the footage the matte was cut from: the patch (made in footage geometry) is then put into the matte's own reference geometry (matte_align.frame_to_ref: the
    reference is a cover-resized crop of the clip), so compose, which maps the reference back onto the footage, lands it exactly; without it the patch is stretched to the matte's size."""
    if patch is None or matte is None:
        return matte
    lo, arr = patch
    is_t = hasattr(matte, "detach")
    m = _np(matte).copy()
    scale = 255.0 if m.dtype == np.uint8 else 1.0
    h, w = m.shape[1:3]
    if arr.shape[1:3] != (h, w):
        if footage_hw is not None and tuple(footage_hw) != (h, w):
            from tools.fusion import matte_align as MA
            arr = np.stack([MA.frame_to_ref(x, (h, w), tuple(arr.shape[1:3])) for x in arr])
        else:
            arr = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in arr])
    k = min(len(arr), len(m) - lo)
    if k <= 0:
        return matte
    val = (np.clip(arr[:k], 0.0, 1.0) * scale).astype(m.dtype)
    m[lo:lo + k] = val[..., None] if m.ndim == 4 else val
    if is_t:
        import torch
        return torch.from_numpy(m).to(matte.device)
    return m


def clip_patches(clip: dict) -> list:
    """every matte patch riding on a clip, in cut order: `matte_patches` (one per 3D cut the shot takes part in), else the single `matte_patch`."""
    ps = clip.get("matte_patches")
    if ps:
        return list(ps)
    return [clip["matte_patch"]] if clip.get("matte_patch") is not None else []


def add_matte_patch(clip: dict, patch) -> dict:
    """the clip dict with one more cut's matte patch on it (`matte_patches` keeps them all in cut order, `matte_patch` the latest for older readers)."""
    clip["matte_patches"] = clip_patches(clip) + [patch]
    clip["matte_patch"] = patch
    return clip


def patch_matte_all(matte, patches, footage_hw=None):
    """`matte` with every patch written in, in order (a later cut overwrites the frames it shares with an earlier one)."""
    for pt in patches or []:
        matte = patch_matte(matte, pt, footage_hw=footage_hw)
    return matte


def patched_mattes(clips, mattes: dict | None) -> dict:
    """{sid: matte} with the transition mattes of ALL the cuts of every shot written in. A shot in the middle of the piece is the B of one cut AND the A of the next: the second cut used to
    overwrite the first one's patch (`by[s]["matte_patch"] = mp`), so the frames of the first transition kept the shot's OWN matte over the transformed picture (a half-transparent second
    figure at the cut, draft v014 r02), and the window cuts read the neighbours' unpatched mattes."""
    out = dict(mattes or {})
    for clip in clips or []:
        sid, ps, m = clip.get("sid"), clip_patches(clip), out.get(clip.get("sid"))
        if ps and m is not None:
            out[sid] = patch_matte_all(m, ps, footage_hw=tuple(int(x) for x in clip["u8"].shape[1:3]))
    return out


class CutResult:
    """a, b = the new clips (numpy uint8); plan; matte_a / matte_b = (clip-local first frame, float32 (k,H,W)) patches for the two shots' mattes or None."""

    def __init__(self, a, b, plan, matte_a=None, matte_b=None, info=None):
        self.a, self.b, self.plan, self.matte_a, self.matte_b = a, b, plan, matte_a, matte_b
        self.info = info or {}                           # the engine's info (size, cards, which depth: given | depth-anything-v2-small)

    def __iter__(self):                                  # (a, b, plan): the original 3-tuple unpacking keeps working
        return iter((self.a, self.b, self.plan))


def _dolly_render(ca, cb, ma, mb, da, db, plan):
    """depth_dolly with LIVE textures (2026-10-30, user: the footage must never go still in a transition): the depth / matte geometry comes from A's frame a0 and B's frame 0 (one depth per shot),
    but every output frame re-textures the two layers with the clips' own frames: A plays a0 + i (clamped at the clip's real end), B plays from its first frame (index i - i2 from the frame B
    takes over), mattes follow the same index. ma / mb = float (T,H,W) mattes or None. -> SimpleNamespace(frames uint8 RGB (n,H,W,3), info, matte uint8 (n,H,W) | None)."""
    import dataclasses
    from types import SimpleNamespace
    from tools.fusion import depth_dolly as dd
    d, p = _dolly_params(dict(plan["params"]))
    a0, i2, n = plan["a0"], plan["i2"], plan["n"]
    h0, w0 = ca[a0].shape[:2]
    w, h = dd.mesh_size(w0, h0, int(d["short"]))
    prm = dd.Params(depth_ratio=p.depth_ratio, subject_relief=p.subject_relief)
    direction = plan["kind"].split(":")[1]
    fl = dd.frames(p, 2, False)
    ia_of = lambda i: min(a0 + i, len(ca) - 1)                       # A: its own next frame, held only at the clip's real end   # noqa: E731
    ib_of = lambda i: min(max(i - i2, 0), len(cb) - 1)              # B: from its first frame at the frame it takes over         # noqa: E731
    m_of = lambda m, j: None if m is None else m[min(j, len(m) - 1)]   # noqa: E731
    deps = [DP.as_depth(g) if g is not None else dd.estimate_depth(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)) for im, g in ((ca[a0], da), (cb[0], db))]      # once
    mats0 = [m_of(ma, a0), m_of(mb, 0)]

    def base(k, im, m):
        m = np.zeros((h0, w0), np.float32) if m is None else cv2.resize(np.asarray(m, np.float32), (w0, h0))
        return DP._layer(dd, im, m, deps[k], w, h, prm)[0]
    base_a, base_b = base(0, ca[a0], mats0[0]), base(1, cb[0], mats0[1])

    def live(layer, im_rgb):
        """the layer's geometry with this frame's colours (the back layer keeps its inpainted colours behind the subject)."""
        col = cv2.resize(cv2.cvtColor(np.ascontiguousarray(im_rgb), cv2.COLOR_RGB2BGR), (layer.W, layer.H), interpolation=cv2.INTER_AREA)
        bcol = np.where(layer.bmask[..., None], layer.bcol, col).astype(np.uint8)
        return dataclasses.replace(layer, col=col, bcol=bcol)

    def render(i, fr, a_img, b_img):
        sc = dd.Scene([live(base_a, a_img), live(base_b, b_img)], direction, p)
        return np.ascontiguousarray(sc.render_t(fr.seg, fr.t, sc.blur_subs(fr.t), p.shutter) if fr.kind != "hold" else sc.render_t(fr.seg, fr.t)).copy()
    frames, mattes = [], []
    want_m = plan.get("matte_pass") and (ma is not None or mb is not None)
    for i, fr in enumerate(fl):
        ia, ib = ia_of(i), ib_of(i)
        frames.append(cv2.cvtColor(render(i, fr, ca[ia], cb[ib]), cv2.COLOR_BGR2RGB))
        if want_m:                                                    # the same scene on matte-coloured layers = the matte through the dolly, same frame index
            g = [np.repeat(((np.zeros((h0, w0), np.float32) if m is None else cv2.resize(np.asarray(m, np.float32), (w0, h0))) * 255).astype(np.uint8)[..., None], 3, -1)
                 for m in (m_of(ma, ia), m_of(mb, ib))]
            mattes.append(render(i, fr, g[0], g[1])[..., 0])
    return SimpleNamespace(frames=np.stack(frames), info={"image2_start": plan["i2"], "n_frames": len(frames), "live": True}, matte=np.stack(mattes) if mattes else None)


GLITCH_LEAD_FRAMES = 4           # ship #10b: B's subject starts to glitch in this many frames BEFORE the end of the travel (the eased camera is at rest by then): the resolve overlaps the settle, no pause-then-pop
GLITCH_OUT_FRAMES = 4            # A's subject, at the start of a move on a plate shot: a short flicker out (it cannot ride the camera: there is no card)


def _matte_u8_rgb(m, lo, hi, shape):
    """matte frames [lo:hi) as uint8 RGB frames of `shape` (T,H,W,3); zeros when there is no matte."""
    hh, ww = shape[1:3]
    if m is None:
        return np.zeros((hi - lo, hh, ww, 3), np.uint8)
    mm = np.asarray(m)[lo:hi]
    mm = mm.astype(np.float32) / (255.0 if mm.dtype == np.uint8 else 1.0)
    if mm.shape[1:3] != (hh, ww):
        mm = np.stack([cv2.resize(x, (ww, hh), interpolation=cv2.INTER_LINEAR) for x in mm])
    u = np.clip(mm * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.repeat(u[..., None], 3, -1)


def _matte_f(m, j, hw):
    """matte frame j as float (H,W) at hw (clamped to the clip's last matte frame); zeros without a matte."""
    if m is None:
        return np.zeros(hw, np.float32)
    x = np.asarray(m)[min(j, len(m) - 1)]
    x = x.astype(np.float32) / (255.0 if x.dtype == np.uint8 else 1.0)
    return x if x.shape[:2] == tuple(hw) else cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def _engine_frames(srcA, srcB, da, db, plan):
    """the transition frames of the plan's kind with NO subject split (zero mattes: the whole frame is one layer, no card, no hole fill). srcA / srcB = full clips (uint8)."""
    a0, n, b_len = plan["a0"], plan["n"], plan["b_len"]
    if family(plan["kind"]) == "dolly":
        return _dolly_render(srcA, srcB, None, None, da, db, dict(plan, matte_pass=False)).frames
    return DP.depth_pivot_render(srcA[a0:a0 + n], None, srcB[:b_len], None, da, db, plan["params"], matte_pass=False).frames


def _plain_render(ca, cb, ma, mb, da, db, plan, plates):
    """2026-10-30 (user on own-video v003 base2 s2: "it should not have used a blurred out silhouette ... the movement just with the background and the text coming on ... then the matted foreground element
    glitch into existence once the frame had settled"). A cut never splits the subject out of its frame:
      * a shot WITHOUT a clean plate moves as ONE picture (its own frames, A from a0 + k, B from its first frame): no card, no silhouette, no blurred hole fill; its matte through the transition is the
        engine run once more on matte-coloured frames (the matte follows the very same warp);
      * a shot WITH a clean plate (plates[sid], same frame index) moves as its PLATE; its subject is absent during the move and GLITCHES IN over `glitch_in.GLITCH_FRAMES` frames once the move has settled
        (B; `plan['settle']`), and flickers out at the start of the move (A); the matte through the transition is the glitched matte (0 while the subject is absent).
    -> (frames uint8 (n,H,W,3) at the clips' size, matte_a (i2,H,W) | None, matte_b (b_len,H,W) | None, info)."""
    from tools.fusion import glitch_in as G
    a0, i2, n, b_len, settle = plan["a0"], plan["i2"], plan["n"], plan["b_len"], int(plan.get("settle", plan["i2"]))
    pa, pb = (plates or {}).get(plan["a"]), (plates or {}).get(plan["b"])
    hw = tuple(ca.shape[1:3])
    srcA = pa if pa is not None else ca
    srcB = pb if pb is not None else cb
    frames = _fit(np.asarray(_engine_frames(srcA, srcB, da, db, plan)), *hw).copy()
    want_m = plan["matte_pass"] and (ma is not None or mb is not None)
    mA = mB = None
    if want_m and (pa is None or pb is None):                     # the matte through the warp of the sides that moved with their subject in the picture
        ga = np.zeros_like(ca) if (pa is not None or ma is None) else _matte_u8_rgb(ma, 0, len(ca), ca.shape)
        gb = np.zeros_like(cb) if (pb is not None or mb is None) else _matte_u8_rgb(mb, 0, len(cb), cb.shape)
        pm = _fit(np.asarray(_engine_frames(ga, gb, da, db, plan)), *hw)[..., 0].astype(np.float32) / 255.0
        mA, mB = pm[:i2], pm[i2:]
    prm = plan["params"]
    hold1 = int(prm["hold1"] if isinstance(prm, dict) else prm.hold_start)
    seed = int(plan.get("k", 0)) * 7 + 3
    info = {}
    if pa is not None:                                            # A: the subject stays through the lead-in, flickers out as the move starts, then it is only the plate
        mA = np.zeros((i2,) + hw, np.float32)
        for i in range(i2):
            k = i - hold1
            m_ = _matte_f(ma, a0 + i, hw)
            if k < 0:
                frames[i], mA[i] = ca[a0 + i], m_                  # the clip's own frame, subject and all (the plate is only the background of the move)
            else:
                frames[i], mA[i] = G.composite(frames[i], ca[a0 + i], m_, k, GLITCH_OUT_FRAMES, seed, reverse=True)
        info["subject_exit"] = "flicker_out"
    if pb is not None:                                            # B: plate only while the camera moves, the subject glitches in once it has settled
        mB = np.zeros((b_len,) + hw, np.float32)
        j_s = max(settle - i2 - GLITCH_LEAD_FRAMES, 0)
        for j in range(b_len):
            k = j - j_s
            frames[i2 + j], mB[j] = G.composite(frames[i2 + j], cb[j], _matte_f(mb, j, hw), k, G.GLITCH_FRAMES, seed + 1)
        info.update(subject_entry="glitch_in", glitch_start=int(j_s), glitch_frames=int(G.GLITCH_FRAMES))
    return frames, mA, mB, info


def apply_cut(piece: dict, spec: dict, clips: dict, mattes: dict | None = None, depths: dict | None = None, render=None, plates: dict | None = None) -> CutResult:
    """render the transition from the plan's slices of the two clips and splice it in (frames AND, with matte_pass, the subject matte through the
    transition). clips / mattes / depths = {sid: array}; mattes [T,H,W] float or uint8, depths = a depth frame or clip (the first frame is used; empty =
    estimated on the CPU). `plates` = {sid: clean-plate clip, same frames} (uint8): such a shot moves as its plate and its subject glitches in after the move (`_plain_render`).
    A cut never splits the subject out of its frame unless the cut asks for it (`split: true` on the cut = the old card + hole-fill look). `render` = a replacement for the engine call (tests):
    render(a, ma, b, mb, da, db, params) -> object with .frames / .info (/ .matte). -> CutResult(a, b, plan, matte_a, matte_b); unpacks as (a, b, plan)."""
    plan = plan_cut(piece, spec)
    if plan["problems"]:
        raise ValueError(f"{plan['kind']} cut impossible: " + "; ".join(plan["problems"]))
    ca, cb = _np(clips[plan["a"]])[..., :3], _np(clips[plan["b"]])[..., :3]
    ma, mb = (mattes or {}).get(plan["a"]), (mattes or {}).get(plan["b"])
    da, db = (depths or {}).get(plan["a"]), (depths or {}).get(plan["b"])
    explore = family(plan["kind"]) == "explore"
    da, db = (None if x is None else (_np(x) if explore else _np(x)[:1]) for x in (da, db))       # the promoted explore cuts read the real PER-FRAME depth (Video Depth Anything), the 3D cuts one frame
    a0, i2, n, b_len = plan["a0"], plan["i2"], plan["n"], plan["b_len"]
    want_m = plan["matte_pass"] and (ma is not None or mb is not None)
    if explore:
        from tools.fusion import explore_cuts2 as E
        prm = {k: v for k, v in plan["params"].items() if k != "style"}
        prm.update(radial_order(piece, plan, ma))
        render = render or E.make_render(plan["params"]["style"], **prm)
        sl = lambda d, lo, hi: d if (d is None or d.ndim < 3 or len(d) <= 1) else d[lo:hi]      # noqa: E731
        da, db = sl(da, a0, a0 + n), sl(db, 0, b_len)
    if render is None and not plan["split"]:
        pl_ = {s: _np(v)[..., :3] for s, v in (plates or {}).items() if v is not None and s in (plan["a"], plan["b"])}
        frames, pma, pmb, extra = _plain_render(ca, cb, None if ma is None else _np(ma), None if mb is None else _np(mb), da, db, plan, pl_)
        a_new, b_new = splice(ca, cb, frames, plan)
        mpa = mpb = None
        if want_m and pma is not None:
            mpa = (a0, np.asarray(pma, np.float32))
        if want_m and pmb is not None:
            mpb = (0, np.asarray(pmb, np.float32))
        hw = tuple(ca.shape[1:3])
        if mpa is None and ma is not None:
            mpa = (a0, np.zeros((i2,) + hw, np.float32))
        if mpb is None and mb is not None:
            mpb = (0, np.zeros((n - i2,) + hw, np.float32))
        info = {"image2_start": i2, "n_frames": n, "split": False, "plates": sorted(pl_), **extra}
        return CutResult(a_new, b_new, plan, mpa, mpb, info)
    if family(plan["kind"]) == "dolly" and render is None:
        res = _dolly_render(ca, cb, _matte_sl(ma, 0, None), _matte_sl(mb, 0, None), da, db, plan)
    else:
        render = render or (lambda *a: DP.depth_pivot_render(*a, matte_pass=want_m))
        res = render(ca[a0:a0 + n], None if ma is None else _np(ma)[a0:a0 + n], cb[:b_len], None if mb is None else _np(mb)[:b_len], da, db, plan["params"])
    got = int(res.info.get("image2_start", i2))
    if got != i2:
        raise ValueError(f"{plan['kind']} cut: the engine put image B at output frame {got}, the plan assumed {i2}")
    a_new, b_new = splice(ca, cb, res.frames, plan)
    pm = getattr(res, "matte", None)
    mpa = mpb = None
    if want_m and pm is not None:
        pm = np.asarray(pm).astype(np.float32) / 255.0
        mpa, mpb = (a0, pm[:i2]), (0, pm[i2:])
    hw = tuple(ca.shape[1:3])
    if mpa is None and ma is not None:                           # no matte through the transition: the old matte must not stay on the transformed frames
        mpa = (a0, np.zeros((i2,) + hw, np.float32))
    if mpb is None and mb is not None:
        mpb = (0, np.zeros((n - i2,) + hw, np.float32))
    return CutResult(a_new, b_new, plan, mpa, mpb, dict(getattr(res, 'info', None) or {}))


def cut_record(res: CutResult) -> dict:
    """what MographArchive keeps of one rendered cut (ingredients: `<stem>_cut_s1-s2_<kind>.mp4` + `_matte/` PNGs + .json): the transition frames as two VIEWS of the new
    clips (A's tail, B's head: no copy), the subject matte through the transition as two float (k,H,W) arrays (None when there was none: the zero patches that only
    blank the old matte are not a render), the plan's facts and the engine info."""
    p = res.plan
    a0, i2, b_len = int(p["a0"]), int(p["i2"]), int(p["b_len"])
    matte = None
    if res.matte_a is not None and res.matte_b is not None and (bool(np.any(res.matte_a[1])) or bool(np.any(res.matte_b[1]))):
        matte = [res.matte_a[1], res.matte_b[1]]
    return dict(a=p["a"], b=p["b"], kind=p["kind"], n=int(p["n"]), i2=i2, params=p["params"], window_a=p["window_a"], window_b=p["window_b"],
                info=dict(res.info or {}), frames=[res.a[a0:a0 + i2], res.b[:b_len]], matte=matte)
