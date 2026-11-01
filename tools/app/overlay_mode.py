#!/usr/bin/env python
"""Own-video EDIT MODE (2026-11-01): "overlay" (default) vs "redesign", and everything the overlay route needs that is not a ComfyUI call.

    edit "overlay"   the user's picture is NOT touched. Every output frame is the matching source frame (same size, same fps, same count, same order, the clip's real cuts left as hard
                     cuts) except where our type is composited. Words animate in / out, sit in front of the subject or behind it (the matte), carry fine outlines and shot colours, and
                     land when they are SUNG (tools/fusion/story._find_sung / _land_for) or on a beat of the soundtrack, on the SOURCE frame grid.
    edit "redesign"  the old route: the design's 3D cuts, word-to-word transitions, plate effects. It keeps the source size, fps and frame count too (no 9:16 crop, no tier rescale, no fps
                     halving), but it may change the picture.

Pure helpers (stdlib; numpy / cv2 only inside the pixel functions), unit-tested in tests/test_overlay_mode.py:

    norm_edit(v)                      "overlay" | "redesign" (anything else -> "overlay")
    native_sizes(W, H, quality)       the sizes of one run at the SOURCE size: compose = clip = source (even), matte = Alpha-Gen size (multiple of 32, aspect kept), format 9:16 | 16:9
    quantise_time / word_frame        a time on the source fps frame grid (t = frame / src_fps)
    quantise_slots(slots, fps, n)     word slots {land, hold_until} -> source-frame-exact seconds
    asr_from_lyrics / sung_hits       the lyric route: lines found in the transcription land when they are sung (story._find_sung / _land_for), else on their beat slot
    lock_piece(piece)                 the PICTURE LOCK on a designed piece (what the retime step runs for edit overlay): every footage-touching op / cut / hold / plate key removed
    picture_violations(piece)         [] when the piece cannot change a picture pixel outside the type
    restore_frame / restore_video     box side, after the render: output = the SOURCE frame except where the composed frame differs from the engine's own footage (= type)
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path

EDITS = ("overlay", "redesign")
DEFAULT_EDIT = "overlay"
PIPE_FPS = 24


def norm_edit(v) -> str:
    """the brief's / request's `edit` -> "overlay" (default) | "redesign". Accepts a few spellings ("with transitions", "transitions")."""
    s = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("redesign", "with_transitions", "transitions", "full", "edit"):
        return "redesign"
    return DEFAULT_EDIT


# ============================================================================================================================ sizes
def even(v) -> int:
    """the even size at or below v (what ffmpeg's trunc(iw/2)*2 gives); an even size is unchanged."""
    return max(2, int(float(v)) // 2 * 2)


def ceil64(v) -> int:
    """the multiple of 64 at or above v (at least 64)."""
    return max(64, -(-int(v) // 64) * 64)


TYPE_ALIGN_TOL = 0.001        # a type canvas whose aspect is within 0.1 % of the source's counts as the same aspect (a 540x960 source -> 576x1024 is exact)


def type_canvas(W: int, H: int, tol: float = TYPE_ALIGN_TOL, grow: int = 6) -> tuple[int, int]:
    """the size the TYPE passes are rendered at for a source W x H: both sides a multiple of 64 (the type engine's rule, spec.check_size) and each at or above the source side, so the
    type is never rendered smaller than the picture. Chosen to keep the aspect: among the canvases cw = ceil64(W) + 64 k (k <= grow) with ch the multiple of 64 nearest cw * H / W (and >= H),
    the SMALLEST whose aspect is within `tol` of the source's (540x960 -> 576x1024, 1080x1920 -> 1152x2048, 1920x1080 -> 2048x1152, 576x1024 -> itself); when none is, the closest aspect.
    The picture is not padded or stretched: compose cover-fits every layer to the compose size (compose.fit_cover = scale, then a centred crop of the sliver left by the aspect difference,
    identically for footage, matte and type), so the type lands where it was laid out in the visible frame. The layouts are fitted to this canvas's aspect (`type_aspect`)."""
    W, H = int(W), int(H)
    want = W / float(H)
    cw0, ch0 = ceil64(W), ceil64(H)
    cands = set()
    for k in range(int(grow) + 1):
        cw = cw0 + 64 * k
        ideal = cw / want
        for ch in (int(ideal // 64) * 64, ceil64(ideal)):
            if ch >= H and ch >= 64:
                cands.add((cw, ch))
    cands.add((cw0, ch0))
    err = lambda c: abs((c[0] / float(c[1])) / want - 1.0)      # noqa: E731
    good = [c for c in cands if err(c) <= tol]
    best = min(good, key=lambda c: (c[0] * c[1], err(c))) if good else min(cands, key=lambda c: (round(err(c), 5), c[0] * c[1]))
    return int(best[0]), int(best[1])


def visible_rect(canvas, source) -> tuple[float, float, float, float]:
    """the part of a type canvas (cw, ch) that is the visible source frame (sw, sh) after compose's cover-fit, in canvas pixels (x0, y0, x1, y1). The whole canvas when the aspects agree."""
    cw, ch = float(canvas[0]), float(canvas[1])
    sw, sh = float(source[0]), float(source[1])
    s = max(sw / cw, sh / ch)
    vw, vh = sw / s, sh / s
    return (cw - vw) / 2.0, (ch - vh) / 2.0, (cw + vw) / 2.0, (ch + vh) / 2.0


def matte_size(W: int, H: int, quality: str = "draft") -> tuple[int, int]:
    """the Alpha-Gen size for a source W x H: the long side 704 (preview) / 1024 (draft, final), the short side from the source aspect, both a multiple of 32 (ltx_ic alpha rule), at least 128.
    The matte is the only thing run below the source size; it is upscaled back to the source size by compose's `upscale_matte` (the picture is never scaled)."""
    long_side = 704 if quality in ("preview", "still") else 1024
    s = long_side / float(max(W, H))
    w, h = max(128, int(round(W * s / 32.0)) * 32), max(128, int(round(H * s / 32.0)) * 32)
    return int(w), int(h)


def native_sizes(W: int, H: int, quality: str = "draft") -> dict:
    """sizes of a run at the source's own size. compose / shot / clip = the source size (even sides), control / matte = the Alpha-Gen size, type = `type_canvas` (the type engine renders only
    at multiples of 64; compose cover-fits it back to the source size, so a 540x960 source is typed at 576x1024 and its final is still 540x960).
    `piece` is the dict written to piece['sizes'] (plan_rules.sizes_for keys). format = '16:9' for a landscape source else '9:16' (a square one is called 9:16; the layouts use `type_aspect`,
    the aspect of the canvas the type is drawn on = `aspect` to within 0.1 % unless no multiple-of-64 canvas is that close)."""
    cw, ch = even(W), even(H)
    mw, mh = matte_size(cw, ch, quality)
    tw, th = type_canvas(cw, ch)
    fmt = "16:9" if cw > ch else "9:16"
    return dict(compose=(cw, ch), clip=(cw, ch), matte=(mw, mh), type=(tw, th), shot=(cw, ch), control=(mw, mh), format=fmt, aspect=cw / float(ch), type_aspect=tw / float(th),
                piece=dict(control=[mw, mh], shot=[cw, ch], type=[tw, th], compose=[cw, ch]))


# ============================================================================================================================ time on the source frame grid
def quantise_time(t: float, fps: float) -> float:
    """the time of the source frame nearest to t: round(t * fps) / fps."""
    return round(float(t) * float(fps)) / float(fps)


def word_frame(t: float, fps: float) -> int:
    """the source frame a word lands on: an INTEGER, t = frame / fps."""
    return int(round(float(t) * float(fps)))


def quantise_slots(slots: list[dict], fps: float, nframes: int) -> list[dict]:
    """word slots (real seconds) -> every land / hold_until on a source frame (land in [0, nframes - 1], hold_until after land). Adds land_frame / hold_frame (ints)."""
    out = []
    for s in slots:
        lf = min(max(word_frame(s["land"], fps), 0), max(nframes - 1, 0))
        hf = min(max(word_frame(s.get("hold_until", s["land"]), fps), lf + 1), nframes)
        out.append(dict(s, land=round(lf / float(fps), 6), hold_until=round(hf / float(fps), 6), land_frame=lf, hold_frame=hf))
    return out


def pipeline_frame(t_real: float, fps: float) -> float:
    """a real time -> the frame it is on the pipeline clock (K-stretched 24 fps clock: t_pipe = t * K, frame = t_pipe * 24 = t * fps). Equals the SOURCE frame index."""
    return float(t_real) * float(fps)


def asr_from_lyrics(lyr: list) -> list[tuple]:
    """the READ's lyrics [{text, t, ...}] -> [(word, sung second in the SOURCE file)] exactly as story.norm_req builds it (story._line_ts spreads a line's words over its seconds)."""
    from tools.fusion import story as ST
    rows = [l for l in lyr or [] if isinstance(l, dict) and str(l.get("text") or "").strip()]
    out = []
    for j, l in enumerate(rows):
        out += [(ST._cmp_word(w), round(float(t), 2)) for w, t in zip(str(l["text"]).split(), ST._line_ts(rows, j))]
    return out


def sung_hits(texts: list[str], slots: list[dict], asr: list[tuple], beats: list[float], dur: float, w0: float = 0.0) -> tuple[list[dict], list[str]]:
    """the lyric route, story.py's own logic unchanged: every on-screen line found in the transcription `asr` = [(word, sung second in the SOURCE file)] lands when its FIRST word is sung
    (story._find_sung), snapped forward to the nearest beat within SNAP_S and never before it is sung (story._land_for); hold_until = its last word + MIN_HOLD_S; a line that is not found keeps
    its beat slot from `slots`; landings are MIN_GAP_S apart (story._sung_hits). -> (hits [{text, land, hold_until}] in landing order, notes). Seconds are the window clock (source second - w0)."""
    from tools.fusion import story as ST
    R = dict(asr=[(ST._cmp_word(w), round(float(t), 2)) for w, t in asr], vplan=dict(w0=float(w0), beats=[float(b) for b in beats]), length=float(dur), text_src="pasted", tok_t=None)
    notes: list[str] = []
    hits, _ = ST._sung_hits(R, list(texts), [dict(s) for s in slots], None, notes)
    return hits, notes


# ============================================================================================================================ word lifetime + speckle ban (own-video retime)
EXIT_OVERLAP_S = 0.5       # a word may run past its shot's end cut by at most this (pipeline s; also capped at one beat): the exit / transition handing over to the next shot (user 2026-11-02: "It can overlap a bit if the transition needs time but not that long")
EXIT_BEATS = 0.5           # a word's exit lasts half a beat (~0.42 s on the 72 bpm pipeline clock, spec.OVERLAP_FLOORS exit 0.4): it was a whole beat, which made one word cost 3.3 s hit-to-hit and left no room for a word per shot
TRANSITION_MIN_GAP_S = 1.2  # a word-to-word strand transition needs about this much room between one word's end and the next word's start (box runs: 1.1-1.7 s gaps carried strands with the hits on their beats); a planned gap shorter than this is a plain cut
HOLD_QUANT = 4             # holds are whole quarter beats


def quarter_beats(seconds: float, beat_s: float) -> float:
    """seconds -> beats rounded UP to a quarter beat."""
    return math.ceil(float(seconds) / float(beat_s) * HOLD_QUANT - 1e-9) / HOLD_QUANT


def word_hold_s(hit: float, need: float, exit_s: float, shot_b, cuts_p, beat_s: float, hold_until=None, pad_max: float = EXIT_OVERLAP_S) -> float:
    """the HOLD (pipeline s, a whole quarter beat) of a word that lands (is fully formed) at `hit` in a shot ending at `shot_b`: the hold runs so the exit ends by the shot's end + min(pad_max, one beat),
    never shorter than `need` (the readable minimum, max(spec.min_hold, spec.HOLD_FLOOR)). A shot too short to read the word after it lands extends only as far as the readable minimum (hit + need + exit),
    snapped to a later cut in `cuts_p` that falls within a beat after it; never to the next word or a far cut. shot_b None (no shot info) = the slot's own hold_until - hit, floor `need`."""
    hu = (float(hold_until) - hit) if hold_until else None                  # the slot's own hold: an upper bound while it still fits the shot (a word sung early in a long shot does not stay for all of it)
    if shot_b is None:
        hold = max(need, hu if hu is not None else need)
    else:
        pad = min(pad_max, beat_s)
        cap = float(shot_b) + pad
        if hit + need + exit_s > cap + 1e-6:
            m_end = hit + need + exit_s                                            # the earliest the word can be read and gone
            later = sorted(float(c) for c in (cuts_p or []) if m_end - pad <= float(c) <= m_end + beat_s)
            cap = max(m_end, later[0]) if later else m_end                         # a cut within a beat after that: leave on it; else just the readable minimum (never on to a far cut)
            hu = None                                                        # the short shot decides, not the slot's hold
        hold = max(need, min(cap - hit - exit_s, hu if hu is not None else 1e9))
    return max(quarter_beats(need, beat_s), math.floor(hold / beat_s * HOLD_QUANT + 1e-9) / HOLD_QUANT) * beat_s      # rounded DOWN to a quarter beat (never past the cap), floor the readable minimum rounded up


def expand_designed(designed: list, wanted: list) -> list[tuple[int, str]]:
    """the design made fewer words than the words file holds (one word per planned design shot; a word for every source shot with sung lyrics can be more): -> [(designed index to copy, wanted text)]
    for every wanted word, in order. A designed slot whose text joins several consecutive wanted words ("A. B.") is split back into them; a missing tail reuses the designs in turn."""
    out, i = [], 0
    for d, dt in enumerate(designed):
        k = next((q for q in range(min(len(wanted) - i, 6), 0, -1) if " ".join(str(x) for x in wanted[i:i + q]).upper() == str(dt).upper()), 1)
        out += [(d, str(t)) for t in wanted[i:i + k]]
        i += k
    out += [(q % max(len(designed), 1), str(wanted[q])) for q in range(i, len(wanted))]
    return out[:len(wanted)]


def module_pool(pool) -> tuple:
    """the own-video module pool minus the banned speckle modules (design_defaults.SPECKLE_MODULES: `particles`, banned everywhere since 2026-11-01)."""
    from tools.fusion import design_defaults as DD
    return tuple(m for m in pool if m not in DD.BANNED_MODULES)


def mend_speckle(piece: dict) -> list[str]:
    """the image-mode plan-time swap (text_guard.fix_plan -> owner.fix_speckle_forms) on an own-video piece: no word keeps a speckle module or io style. In place; -> one fix text per change."""
    from tools.fusion import owner as OW
    return OW.fix_speckle_forms(piece)


# ============================================================================================================================ the picture lock
PICTURE_OPS = {"grade": "frame", "halation": "frame", "grain": "frame", "vignette": "frame",       # whole-picture finishing
               "crossfade": "a",                                                                 # plate treatments dim / brighten the plate and fade between versions of it
               "plate_pan": "img", "multiplane": "footage",                                      # the plate travels / is rebuilt as depth cards under a camera
               "hold_net": "image"}                                                              # the hold net's scrim darkens the surround of a word
SHOT_KEYS = ("plate", "type_matte", "subject_move", "plate_pan", "flat", "treatment")
# ops that paint a GROUND or restyle the picture: `window_cut` (letters as windows onto a flat colour / another shot: box run R16 drew a cream ground over the whole picture for 3 s), and the explore
# looks that recolour / posterise / difference the footage. A per-shot chain that has one is dropped (the shot falls back to the default chain); the default chain may not have one.
GROUND_OPS = ("window_cut", "duotone", "halftone", "posterize", "depth_bands", "matte_bands", "invert_in", "difference", "plate_diff_key", "chrono_echoes", "img_over")
WORD_KEYS = ("window_source", "window_ground")          # words[]: letters as a window onto another shot / a flat ground (design_options.py); the engine turns them into the shot's window_cut chain
HARD = "hard"


def _subst(obj, old: str, new):
    if isinstance(obj, dict):
        return {k: _subst(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, old, new) for v in obj]
    return new if obj == old else obj


def chains_of(piece: dict) -> list[dict]:
    comp = piece.get("compose") or {}
    return ([comp["default"]] if comp.get("default") else []) + list((comp.get("shots") or {}).values())


def _primary(op: dict):
    key = PICTURE_OPS.get(op.get("op"))
    return (op.get("args") or {}).get(key) if key else None


def _bypass(ops: list[dict], k: int) -> bool:
    """remove ops[k] (a picture op); everything that read its output reads its picture input instead. -> True when removed."""
    o = ops[k]
    src = _primary(o)
    if not isinstance(src, str) or not src.startswith("$"):
        return False
    out = o.get("out", "image")
    del ops[k]
    for j in range(k, len(ops)):
        ops[j]["args"] = _subst(ops[j].get("args") or {}, "$" + out, src)
    if out == "image":                                                      # the chain's result: the op that made `src` now writes `image`
        name = src[1:]
        p = next((i for i in range(k - 1, -1, -1) if ops[i].get("out", "image") == name), None)
        if p is not None and name != "image":
            ops[p]["out"] = "image"
            for j in range(p + 1, len(ops)):
                ops[j]["args"] = _subst(ops[j].get("args") or {}, src, "$image")
        elif p is None:                                                     # `src` is an engine buffer ($footage, ...): an identity grade carries it into `image`
            ops.insert(k, {"op": "grade", "out": "image", "args": {"frame": src, "lift": 0.0, "gamma": 1.0, "gain": 1.0, "sat": 1.0}, "lock_identity": True})
    return True


def lock_piece(piece: dict, sizes: dict | None = None) -> dict:
    """The PICTURE LOCK (edit "overlay"): edits the designed piece IN PLACE so that nothing but the type can change a picture pixel. -> report.
       compose.cuts            every cut -> hard (no 3D cut, no window / pan / circle_grid, no crossfade); params dropped
       compose.holds           [] (no freeze-frames); compose.crossfade 0 (no dissolve between shots)
       compose.soft_support    False (no soft shadow / scrim behind the type)
       chain ops               grade / halation / grain / vignette / crossfade(plate) / plate_pan / multiplane / hold_net bypassed (readers read the op's picture input);
                               knockout.min_contrast 0 (the surround is not darkened to lift the window)
       shots                   plate / type_matte / subject_move / plate_pan / flat / treatment / camera keys dropped
       sizes                   piece.sizes + compose.size = the source size when `sizes` (native_sizes()) is given
    flow_wake (the 'bg_drag' push) moves the TYPE alpha by the footage flow, never the footage: it stays."""
    rep = dict(cuts_made_hard=[], ops_removed=[], knockouts=0, shot_keys_dropped=[], holds_removed=None, crossfade_was=None, chains_dropped=[], word_keys_dropped=[])
    comp = piece.setdefault("compose", {})
    for c in comp.get("cuts") or []:
        if str(c.get("cut", HARD)).lower() != HARD:
            rep["cuts_made_hard"].append([c.get("after"), c.get("cut")])
        c["cut"] = HARD
        c.pop("params", None)
        c.pop("style", None)
    rep["holds_removed"] = comp.get("holds")
    comp["holds"] = []
    rep["crossfade_was"] = comp.get("crossfade")
    comp["crossfade"] = 0.0
    comp["soft_support"] = False
    for sid, ch in list((comp.get("shots") or {}).items()):         # a shot chain that paints a ground / restyles the picture: dropped, the shot uses the default chain (type over the footage)
        bad = [o.get("op") for o in ch.get("ops") or [] if o.get("op") in GROUND_OPS]
        if bad:
            del comp["shots"][sid]
            rep["chains_dropped"].append([sid, bad])
    for i, wd in enumerate((piece.get("type") or {}).get("words") or []):
        for key in WORD_KEYS:
            if key in wd:
                wd.pop(key)
                rep["word_keys_dropped"].append([i, key])
    for ci, ch in enumerate(chains_of(piece)):
        ops = ch.get("ops") or []
        k = 0
        while k < len(ops):
            o = ops[k]
            name = o.get("op")
            if name in PICTURE_OPS and not o.get("lock_identity"):
                if _bypass(ops, k):
                    rep["ops_removed"].append(name)
                    continue
            elif name == "knockout":
                if float((o.get("args") or {}).get("min_contrast", 0.0) or 0.0) > 0.0:
                    o["args"]["min_contrast"] = 0.0
                    rep["knockouts"] += 1
            k += 1
        ch["ops"] = ops
    for sh in piece.get("shots") or []:
        for key in SHOT_KEYS:
            if key in sh:
                sh.pop(key)
                rep["shot_keys_dropped"].append([sh.get("id"), key])
    if sizes:
        piece["sizes"] = dict(piece.get("sizes") or {}, **sizes["piece"])
        comp["size"] = list(sizes["piece"]["compose"])
        piece["format"] = sizes["format"]
    piece.setdefault("plan", {})["overlay_lock"] = rep
    return rep


def picture_violations(piece: dict) -> list[str]:
    """what in the piece could still change a picture pixel outside the type ([] = locked): non-hard cuts, holds, crossfade, picture ops, hold_net, knockout surround, plate keys."""
    v = []
    comp = piece.get("compose") or {}
    for c in comp.get("cuts") or []:
        if str(c.get("cut", HARD)).lower() != HARD:
            v.append(f"cut after {c.get('after')}: {c.get('cut')}")
    if comp.get("holds"):
        v.append("compose.holds")
    if float(comp.get("crossfade") or 0.0) != 0.0:
        v.append("compose.crossfade")
    if comp.get("soft_support"):
        v.append("compose.soft_support")
    for ch in chains_of(piece):
        for o in ch.get("ops") or []:
            n = o.get("op")
            if n in PICTURE_OPS and not o.get("lock_identity"):                 # (the identity grade the lock inserts to carry the footage into `image` is not a change)
                v.append(f"op {n}")
            elif n == "knockout" and float((o.get("args") or {}).get("min_contrast", 0.0) or 0.0) > 0.0:
                v.append("op knockout min_contrast")
            elif n in GROUND_OPS:
                v.append(f"op {n} (paints a ground / restyles the picture)")
    for sh in piece.get("shots") or []:
        for key in SHOT_KEYS:
            if key in sh:
                v.append(f"shot {sh.get('id')} key {key}")
    for i, wd in enumerate((piece.get("type") or {}).get("words") or []):
        for key in WORD_KEYS:
            if key in wd:
                v.append(f"word {i} key {key}")
    lay = piece.get("layout") or {}
    for sid, L in lay.items():
        if int(L.get("xf") or 0) != 0:
            v.append(f"layout {sid} xf {L.get('xf')}")
    return v


# ============================================================================================================================ type BEHIND the subject (overlay)
# user R16 review e5 / e6 (2026-11-01): "none of the text is being masked between the layers". Overlay draws type over AND behind the footage; the subject's Alpha-Gen matte puts the subject back
# over the letters. Per word: FRONT (the guard of compose.LEG_FLOOR: a word the subject hides by > 25 % comes in front whole) or BEHIND (the relaxed floor below: the subject may cover part of
# the letters, the word stays readable). Entry and exit of a front word stay behind the subject (the guard only clears the matte inside the hold + a 0.25 s ramp).
# 2026-11-02 (user, own-video job 20261101-160830-aec4, frame 406: "why are most of the text neither fully over or fully behind the main subject - we need moments like these to read the text
# correctly?"): a word is either FRONT (whole, the subject never over it for the whole hold) or BEHIND only where it still reads. The old relaxed behind floor (the subject may hide 50 % of the
# word, a letter may drop to 10 %) let a word through with its middle letters gone, and on this job it never even ran (the design's occluder was a luminosity key, apply_behind found no matte: chains 0).
# VIS_RULE = the readable-behind rule, measured over the word's hold on the real letter shapes: every glyph keeps >= `letter` of its area visible, the whole word >= `word`, in all but
# `most` of the hold's frames. `letter` 0.70 / `word` 0.85 come from the job's numbers: the failing word had a glyph 0 % visible, the words that still read had every glyph
# >= 0.7 and the word >= 0.85 visible; an end letter hidden by half (0.5) already costs the read. The same numbers are the run-time floor (BEHIND_FLOOR, compose.LEG_FLOOR).
VIS_RULE = {"letter": 0.70, "word": 0.85, "most": 0.25}
BEHIND_FLOOR = {"word": 0.15, "letter": 0.70, "most": 0.25}      # compose.legible_plan floor: the subject hides > 15 % of the letter pixels, or ANY letter < 70 % visible, in more than 25 % of the hold -> the word comes in FRONT whole
FRONT_FLOOR = {"word": -1.0, "letter": 1.01, "most": 0.0}       # a FRONT word: the floor fires on every frame of the hold (word hidden > -1 always), so every letter of it is in front of the subject for the whole hold
BEHIND_SHARE = 0.5          # about half of the words that are on screen go behind, at least BEHIND_MIN
BEHIND_MIN = 1
OVERLAP_MIN = 0.03          # a frame counts as "the subject is in the word's box" above this share of the box
SWEET_COVER = 0.22          # the mean share of the box the subject covers that reads best behind: part of the letters, not most of them


def word_cover(box, masks) -> dict:
    """box = (x0, y0, x1, y1) of the word block as 0..1 of the frame, masks = subject masks (HxW floats 0..1, any size) of the hold's frames.
    -> dict(cover = mean share of the box the subject covers, overlap = share of the frames with cover > OVERLAP_MIN, frames = n)."""
    import numpy as np
    cov = []
    for m in masks:
        if m is None:
            continue
        m = np.asarray(m, np.float32)
        h, w = m.shape[:2]
        x0, y0, x1, y1 = max(int(box[0] * w), 0), max(int(box[1] * h), 0), min(int(round(box[2] * w)), w), min(int(round(box[3] * h)), h)
        cov.append(float(m[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0)
    if not cov:
        return dict(cover=0.0, overlap=0.0, frames=0)
    return dict(cover=round(float(np.mean(cov)), 4), overlap=round(float(np.mean([c > OVERLAP_MIN for c in cov])), 4), frames=len(cov))


def behind_score(sc: dict | None) -> float:
    """how well a word suits being behind the subject: the subject is in its box for most of the hold (overlap) and covers a part of it (cover near SWEET_COVER)."""
    if not sc or not sc.get("frames"):
        return 0.0
    return float(sc["overlap"]) * max(0.05, 1.0 - abs(float(sc["cover"]) - SWEET_COVER) / SWEET_COVER) if sc["overlap"] > 0 else 0.0


def choose_behind(scores: list, share: float = BEHIND_SHARE, min_n: int = BEHIND_MIN) -> list[int]:
    """indices of the words that go BEHIND the subject: about `share` of the words that have frames on screen (scores[i]['frames'] > 0), at least `min_n`, the best behind_score first.
    2026-11-02: a score carrying `ok` (word_visibility / rule_verdict: the VIS_RULE holds over the whole hold) only counts when ok is True AND the subject really is in the word's box
    (overlap > OVERLAP_MIN): a word the subject would hide too much of is never chosen, so min_n can leave nothing behind (no word is sacrificed to fill a quota). Scores without `ok` keep the old rule."""
    on = [i for i, s in enumerate(scores) if s and s.get("frames")]
    if not on:
        return []
    cand = [i for i in on if "ok" not in scores[i] or (scores[i]["ok"] and float(scores[i].get("overlap", 0.0)) > OVERLAP_MIN)]
    if not cand:
        return []
    k = min(len(cand), max(int(min_n), int(share * len(on) + 0.5)))
    cand.sort(key=lambda i: (-behind_score(scores[i]), -float(scores[i].get("overlap", 0.0)), i))
    return sorted(cand[:k])


# ---- the readable-behind rule on real letter shapes (2026-11-02). numpy / cv2 / PIL; the run-time guard (compose.legible_plan with BEHIND_FLOOR) re-checks the same rule on the type the engine draws.
SOLID_CLOSE = 6            # px (compose size): the subject occluder is closed over gaps of this size and its enclosed holes filled (compose.fill_holes close=)
NUDGE_DY = (0.0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20)      # a word stays near the centre (user: type lands centred): at most a fifth of the frame up / down, 8 % sideways, 10 % smaller
NUDGE_DX = (0.0, -0.04, 0.04, -0.08, 0.08)
NUDGE_SCALE = (1.0, 0.9)
NUDGE_EDGE = 0.04          # a nudged word keeps this much of the frame between its ink and every edge (design_defaults.EDGE_MARGIN)
NUDGE_CAP_FLOOR = 0.085    # a shrunk word keeps a cap height of at least this share of the frame height (design_defaults.CAP_FLOOR)


def solid_mask(m, close: int = 2, thresh: float = 0.5):
    """the subject occluder as the compose draws it after upscale_matte -> fill_holes(close=): numpy HxW 0..1 -> HxW 0..1 with enclosed holes filled and narrow gaps bridged."""
    import numpy as np
    from tools.fusion import compose
    m = np.asarray(m, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    return np.asarray(compose.fill_holes(m, amount=1.0, level=1, thresh=thresh, soften=0.0, close=int(close)), np.float32)


def _pil_font(font, axes):
    """the PIL font (200 px) of a registry face at the word's axes, or None when the file / PIL is not there."""
    try:
        import os

        from PIL import ImageFont

        from mograph import font_registry as fr
        from tools.fusion import design_options as DO
        e = DO._font_entry(font)
        if e is None:
            return None
        F = ImageFont.truetype(os.path.join(fr.FONT_DIR, e["file"]), 200)
        ax = {}
        reg = next((v for v in fr.REGISTRY.values() if v["file"] == e["file"]), {})
        ax.update({k: float(v) for k, v in (reg.get("axes") or {}).items()})
        for k, v in (axes or {}).items():
            v = v[1] if isinstance(v, (list, tuple)) else v
            try:
                ax[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        try:
            names = F.get_variation_axes()
        except Exception:      # noqa: BLE001  not a variable font
            names = []
        if names:
            tag_of = {str(n).lower(): t for t, n in fr.AXIS_NAMES.items()}
            vals = []
            for a_ in names:
                nm = a_["name"].decode("latin-1").lower() if isinstance(a_["name"], bytes) else str(a_["name"]).lower()
                tg = next((t for t in ax if t.lower() == tag_of.get(nm, nm)), None)
                vals.append(min(max(ax[tg], a_["minimum"]), a_["maximum"]) if tg is not None else a_["default"])
            F.set_variation_by_axes(vals)
        return F
    except Exception:          # noqa: BLE001
        return None


def glyph_stack(text, font, axes, lines, cap, size_wh, layout="line", centre=(0.5, 0.5)):
    """one boolean mask per LETTER of a word laid out like the type engine does: `lines` (rows), cap height `cap` (share of the frame height), centred on `centre` (frame fractions), the
    face's real glyph widths (PIL, the registry's axes); equal letter slots when the font file is not available. rot90 / rot-90 layouts: the row is laid horizontally on a square canvas
    and turned. -> (K, h, w) bool."""
    import cv2
    import numpy as np
    w, h = int(size_wh[0]), int(size_wh[1])
    if layout in ("rot90", "rot-90"):
        S = max(w, h)
        inner = glyph_stack(text, font, axes, lines, float(cap) * h / S, (S, S), "line", (0.5, 0.5))
        M = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), -90.0 if layout == "rot90" else 90.0, 1.0)
        rot = [cv2.warpAffine(m.astype(np.uint8), M, (S, S), flags=cv2.INTER_NEAREST) > 0 for m in inner]
        x0, y0 = (S - w) // 2, (S - h) // 2
        out = np.stack([m[y0:y0 + h, x0:x0 + w] for m in rot]) if rot else np.zeros((0, h, w), bool)
        return shift_stack(out, centre[0] - 0.5, centre[1] - 0.5) if tuple(centre) != (0.5, 0.5) and len(out) else out
    rows = [str(r) for r in (lines or [text])]
    cap_px = max(float(cap) * h, 2.0)
    lead = 1.0 if len(rows) == 1 else 1.12
    F = _pil_font(font, axes)
    cx, cy = centre[0] * w, centre[1] * h
    row_h = cap_px * lead
    top = cy - row_h * len(rows) / 2.0 + (row_h - cap_px) / 2.0
    masks = []
    for j, r in enumerate(rows):
        base_y = top + j * row_h + cap_px
        if F is None:
            n = max(len(r), 1)
            cw = cap_px * 0.7
            for i, ch in enumerate(r):
                if ch.isspace():
                    continue
                m = np.zeros((h, w), bool)
                x0, x1 = int(round(cx - cw * n / 2 + i * cw)), int(round(cx - cw * n / 2 + (i + 0.85) * cw))
                y0, y1 = int(round(base_y - cap_px)), int(round(base_y))
                m[max(y0, 0):max(min(y1, h), 0), max(x0, 0):max(min(x1, w), 0)] = True
                masks.append(m)
            continue
        from PIL import Image, ImageDraw
        bb = F.getbbox("H")
        capf = float(max(bb[3] - bb[1], 1))
        k_ = cap_px / capf
        width = int(F.getlength(r)) + 8
        for i, ch in enumerate(r):
            if ch.isspace():
                continue
            im = Image.new("L", (width, 420), 0)
            ImageDraw.Draw(im).text((F.getlength(r[:i]) + 4, 320), ch, font=F, fill=255, anchor="ls")
            a_ = np.asarray(im) > 96
            ys, xs = np.nonzero(a_)
            if not len(xs):
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            tw, th = max(int(round((x1 - x0) * k_)), 1), max(int(round((y1 - y0) * k_)), 1)
            piece = cv2.resize(a_[y0:y1, x0:x1].astype(np.uint8) * 255, (tw, th), interpolation=cv2.INTER_AREA)
            px = int(round(cx - width * k_ / 2.0 + x0 * k_))
            py = int(round(base_y - (320 - y0) * k_))
            m = np.zeros((h, w), np.uint8)
            xa, ya, xb, yb = max(px, 0), max(py, 0), min(px + tw, w), min(py + th, h)
            if xb > xa and yb > ya:
                m[ya:yb, xa:xb] = piece[ya - py:yb - py, xa - px:xb - px]
            masks.append(m > 127)
    return np.stack(masks) if masks else np.zeros((0, h, w), bool)


def stack_bbox(st):
    """(x0, y0, x1, y1) of a glyph stack as frame fractions (None when empty)."""
    import numpy as np
    if st is None or not len(st) or not st.any():
        return None
    ys, xs = np.nonzero(st.any(0))
    h, w = st.shape[1:]
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def shift_stack(st, dx: float, dy: float, scale: float = 1.0, about=(0.5, 0.5)):
    """the glyph stack moved by (dx, dy) frame fractions and scaled about `about` (frame fractions)."""
    import cv2
    import numpy as np
    k, h, w = st.shape
    M = np.float32([[scale, 0, (1 - scale) * about[0] * w + dx * w], [0, scale, (1 - scale) * about[1] * h + dy * h]])
    return np.stack([cv2.warpAffine(m.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0 for m in st]) if k else st


def word_visibility(st, occluders) -> dict:
    """visible share of every glyph in every frame: st = (K, h, w) bool glyph stack, occluders = [HxW 0..1 subject occluders at the stack's size (None = no subject)].
    -> dict(vis = (F, K) array, word = (F,) visible share of the word's letter pixels, n)."""
    import numpy as np
    K = len(st)
    area = np.maximum(st.reshape(K, -1).sum(1).astype(np.float64), 1.0) if K else np.ones(0)
    rows, wv = [], []
    for o in occluders:
        if o is None or not K:
            rows.append(np.ones(K))
            wv.append(1.0)
            continue
        o = np.asarray(o, np.float32)
        hid = (st * o[None]).reshape(K, -1).sum(1)
        rows.append(1.0 - hid / area)
        wv.append(1.0 - float(hid.sum()) / float(area.sum()))
    return dict(vis=np.asarray(rows, np.float64).reshape(len(rows), K), word=np.asarray(wv), n=len(rows))


def rule_verdict(wv: dict, rule: dict | None = None) -> dict:
    """VIS_RULE on a word_visibility result: a frame FAILS when the word is < rule.word visible or any glyph < rule.letter; the word reads behind when at most rule.most of the hold's frames fail.
    -> dict(ok, fail_share, word_min, word_mean, letter_min, letter_worst (glyph index), frames)."""
    import numpy as np
    r = dict(VIS_RULE, **(rule or {}))
    vis, word = wv["vis"], wv["word"]
    if not wv["n"] or vis.shape[1] == 0:
        return dict(ok=False, fail_share=1.0, word_min=0.0, word_mean=0.0, letter_min=0.0, letter_worst=-1, frames=0)
    bad = (word < r["word"]) | (vis.min(1) < r["letter"])
    share = float(bad.mean())
    return dict(ok=bool(share <= r["most"] + 1e-9), fail_share=round(share, 3), word_min=round(float(word.min()), 3), word_mean=round(float(word.mean()), 3),
                letter_min=round(float(vis.min()), 3), letter_worst=int(np.argmin(vis.min(0))), frames=int(wv["n"]))


def _overlap(st, occluders) -> float:
    """share of the hold's frames in which the subject covers more than OVERLAP_MIN of the word's letter pixels (0 when there is no subject)."""
    import numpy as np
    tot = max(float(st.sum()), 1.0)
    n, k = 0, 0
    for o in occluders:
        if o is None:
            continue
        k += 1
        if float((st * np.asarray(o, np.float32)[None]).sum()) / tot > OVERLAP_MIN:
            n += 1
    return round(n / k, 3) if k else 0.0


def plan_word(st, occluders, cap: float, rule: dict | None = None, dys=NUDGE_DY, dxs=NUDGE_DX, scales=NUDGE_SCALE) -> dict:
    """the readable-behind decision for one word. 1 the word where it is: verdict at (0, 0, 1). 2 failing: the cheapest nudge (dy / dx / scale <= 1, the ink inside the frame margin, the cap kept
    >= NUDGE_CAP_FLOOR) that makes it pass while the subject STILL overlaps it somewhere in the hold (a real behind moment; a word the subject no longer touches is plain front type and stays
    centred). 3 otherwise FRONT. -> dict(status 'behind' | 'nudged' | 'front', dx, dy, scale, verdict (at the final placement), base (at the start), overlap)."""
    base = rule_verdict(word_visibility(st, occluders), rule)
    ov0 = _overlap(st, occluders)
    if base["ok"]:
        return dict(status="behind", dx=0.0, dy=0.0, scale=1.0, verdict=base, base=base, overlap=ov0)
    best = None
    full = occluders
    occluders = occluders[::max(1, len(occluders) // 12)]            # the search runs on <= ~12 frames of the hold; the chosen placement is re-measured on all of them
    for s in scales:
        if s < 1.0 and float(cap) * s < NUDGE_CAP_FLOOR - 1e-9:
            continue
        for dy in dys:
            for dx in dxs:
                if dx == 0.0 and dy == 0.0 and s == 1.0:
                    continue
                cand = shift_stack(st, dx, dy, s)
                bb = stack_bbox(cand)
                if bb is None or bb[0] < NUDGE_EDGE or bb[1] < NUDGE_EDGE or bb[2] > 1 - NUDGE_EDGE or bb[3] > 1 - NUDGE_EDGE:
                    continue
                ov = _overlap(cand, occluders)
                if ov <= OVERLAP_MIN:
                    continue
                v = rule_verdict(word_visibility(cand, occluders), rule)
                if not v["ok"]:
                    continue
                cost = abs(dy) + 1.5 * abs(dx) + (1.0 - s)
                if best is None or cost < best[0]:
                    best = (cost, dx, dy, s, v, ov)
    if best is None:
        return dict(status="front", dx=0.0, dy=0.0, scale=1.0, verdict=base, base=base, overlap=ov0)
    fin = shift_stack(st, best[1], best[2], best[3])
    v = rule_verdict(word_visibility(fin, full), rule)
    if not v["ok"]:                                                  # the sampled frames flattered it: front
        return dict(status="front", dx=0.0, dy=0.0, scale=1.0, verdict=base, base=base, overlap=ov0)
    return dict(status="nudged", dx=best[1], dy=best[2], scale=best[3], verdict=v, base=base, overlap=_overlap(fin, full))


def _matte_like(name, outs: dict) -> bool:
    nm = name[1:] if isinstance(name, str) and name.startswith("$") else None
    for _ in range(4):
        if nm is None:
            return False
        if nm in ("matte", "subject_matte") or (nm.endswith("_matte") and not nm.startswith("type_")):
            return True
        o = outs.get(nm)
        if o is None or o.get("op") not in ("upscale_matte", "fill_holes", "legible", "blur"):
            return False
        a = o.get("args") or {}
        src = a.get("matte", a.get("mask", a.get("img")))
        nm = src[1:] if isinstance(src, str) and src.startswith("$") else None
    return False


def apply_behind(piece: dict, holds: list, behind: list[int], fps: int = 24, spans: list | None = None) -> dict:
    """Per-word front / behind on the pure-type chains: the pure-type route draws every word with `over(type)` and then puts the footage back over it by the subject matte
    (`over(base=type layer, fg=footage, alpha=matte)`); recipe_eval.guard_ops would wrap that alpha in ONE `legible` op with the default floor for every word. Here two explicit `legible`
    ops (guard_ops leaves a matte that already comes out of a legible op alone) split the words: the FRONT words get FRONT_FLOOR (the floor fires on every frame: the word is in front of the subject for its
    whole hold, never half over / half behind), the BEHIND words get BEHIND_FLOOR (the VIS_RULE numbers: a word the subject hides too much of is brought in front whole at run time). `holds` = [(hit_s, release_s)]
    per word (pipeline seconds, as plan.flow_wake.holds). `spans` = [(start_s, end_s)] per word (the whole appearance, PR.word_spans): a FRONT word is in front over its whole span, shaped by the FULL type
    alpha (the echo / wake copy as well as the crisp front copy) - the first box render of this rule (job 20261101-170553-6845) had the subject solid over the forming letters, then the whole word
    snapped in front at the hit (frame 619 jump, overlay_check cuts FAIL). -> report {chains, behind, front}."""
    import copy
    from tools.fusion import compose
    hl = [[round(float(a), 4), round(float(b), 4)] for a, b in holds]
    bset = set(int(i) for i in behind)
    h_b = [hl[i] for i in range(len(hl)) if i in bset]
    h_f = [hl[i] for i in range(len(hl)) if i not in bset]
    if spans:                                                       # front words: the whole appearance (forming + hold + exit), not only the hold
        h_f = [[round(float(spans[i][0]), 4), round(float(spans[i][1]), 4)] if i < len(spans) else hl[i] for i in range(len(hl)) if i not in bset]
    rep = dict(chains=0, behind=sorted(bset), front=[i for i in range(len(hl)) if i not in bset], floor=dict(BEHIND_FLOOR), front_floor=dict(FRONT_FLOOR))
    for ch in chains_of(piece):
        ops = ch.get("ops") or []
        outs, k = {}, 0
        while k < len(ops):
            o = ops[k]
            args = o.get("args") or {}
            if o.get("op") == "over" and isinstance(args.get("base"), str) and args["base"][1:] in outs and _matte_like(args.get("alpha"), outs)                     and (outs[args["base"][1:]].get("op") == "over") and not _matte_like((outs[args["base"][1:]].get("args") or {}).get("alpha"), outs)                     and (outs.get(args["alpha"][1:]) or {}).get("op") != "legible":
                ta = outs[args["base"][1:]]["args"]["alpha"]
                ta_full = ta                                         # the full type alpha: the echo / wake copy drawn under the crisp front copy (mask_gate of the word's hold_life)
                b2 = outs[args["base"][1:]]["args"].get("base")
                o2 = outs.get(b2[1:]) if isinstance(b2, str) and b2.startswith("$") else None
                if o2 is not None and o2.get("op") == "over" and isinstance((o2.get("args") or {}).get("alpha"), str):
                    o3 = outs.get(o2["args"]["alpha"][1:])
                    if o3 is not None and o3.get("op") == "mask_gate" and isinstance((o3.get("args") or {}).get("a"), str):
                        ta_full = o3["args"]["a"]
                new = []
                src = args["alpha"]
                if h_f:
                    new.append({"op": "legible", "out": f"_legF{k}", "args": {"type_a": ta_full, "matte": src, "holds": copy.deepcopy(h_f), "t": "$t", "fps": int(fps), "floor": dict(FRONT_FLOOR), "letters": False, "behind": False}})
                    src = f"$_legF{k}"
                if h_b:
                    new.append({"op": "legible", "out": f"_legB{k}", "args": {"type_a": ta, "matte": src, "holds": copy.deepcopy(h_b), "t": "$t", "fps": int(fps), "floor": dict(BEHIND_FLOOR), "letters": False, "behind": False}})
                    src = f"$_legB{k}"
                args["alpha"] = src
                ops[k:k] = new
                for n_ in new:
                    outs[n_["out"]] = n_
                k += len(new)
                rep["chains"] += 1
            outs[o.get("out", "image")] = o
            k += 1
        ch["ops"] = ops
    return rep


def solid_occluder(piece: dict, size=None) -> dict:
    """The occluder that puts the subject back over the type must be the SUBJECT, solid. Found on job 20261101-160830-aec4 (frame 406): the design drew the type with
    `sandwich(matte = luminosity_key(footage, 0.18 .. 0.55))`, i.e. the subject was whatever is BRIGHT: the subject's dark clothing (luma < 0.18) was transparent, so letter fragments showed through
    the body in patches, and a luminosity key is not a matte, so neither apply_behind nor the guard (both read a matte-like alpha) touched the word (report chains: 0).
    Every `luminosity_key` that is read as the alpha of an `over` / the matte of a `sandwich` gets a second path next to it: `upscale_matte(matte = $matte (the shot's Alpha-Gen alpha), guide = the
    footage)` -> `fill_holes(close = SOLID_CLOSE)` (enclosed holes filled, gaps bridged) and the readers are re-pointed at it; the key op itself stays (other readers). `$matte` is the shot's
    own Alpha-Gen matte (an empty clip where no person is). -> dict(chains, replaced)."""
    rep_ = dict(chains=0, replaced=0, size=None)
    comp = piece.get("compose") or {}
    sz = size or comp.get("size") or (piece.get("sizes") or {}).get("compose")
    if not sz:
        return rep_
    sz = [int(sz[0]), int(sz[1])]
    rep_["size"] = sz
    for ch in chains_of(piece):
        ops = ch.get("ops") or []
        keys = {o.get("out", "image"): o for o in ops if o.get("op") == "luminosity_key"}
        if not keys:
            continue
        done = {}
        k = 0
        while k < len(ops):
            o = ops[k]
            a = o.get("args") or {}
            fld = "alpha" if o.get("op") == "over" else ("matte" if o.get("op") == "sandwich" else None)
            ref = a.get(fld) if fld else None
            if isinstance(ref, str) and ref.startswith("$") and ref[1:] in keys:
                nm = ref[1:]
                if nm not in done:
                    kk = next(i for i, x in enumerate(ops) if x is keys[nm])
                    up, solid = f"_occ_{nm}_up", f"_occ_{nm}"
                    guide = (keys[nm].get("args") or {}).get("img", "$footage")
                    ops[kk + 1:kk + 1] = [{"op": "upscale_matte", "out": up, "args": {"guide": guide, "matte": "$matte", "size": list(sz)}},
                                          {"op": "fill_holes", "out": solid, "args": {"mask": "$" + up, "amount": 1.0, "level": 1, "thresh": 0.5, "soften": 1.0, "close": int(SOLID_CLOSE)}}]
                    done[nm] = solid
                    rep_["replaced"] += 1
                    k = next(i for i, x in enumerate(ops) if x is o)
                a[fld] = "$" + done[nm]
            k += 1
        if done:
            rep_["chains"] += 1
    return rep_


def frame_count_preserved(n_src: int, spans: list[int]) -> bool:
    """the planned shot spans cover exactly the window's frames (so compose emits n_src frames)."""
    return int(sum(int(x) for x in spans)) == int(n_src)


# ============================================================================================================================ restore the picture (box side, after the render)
MIN_AREA = 8               # a changed region smaller than this (px) is codec noise; a 1 px line (a strand, a thin outline) is longer than that and stays


def changed_pixels(d, thr: int, min_area: int = MIN_AREA):
    """uint8 0/1 mask of the pixels of difference image `d` (uint8, max over channels) above `thr`, minus connected specks under `min_area` px. A median filter would also erase 1-2 px lines
    (word-bridge strands, fine outlines: box run R16 frames 98 / 152), so noise is removed by region size instead."""
    import cv2
    import numpy as np
    m = (d > thr).astype(np.uint8)
    if m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
        keep[0] = False
        m = keep[lab].astype(np.uint8)
    return m


def type_mask(E, R, thr: int = 20, grow: int = 4, feather: float = 1.5):
    """E = the composed frame, R = the engine's own footage for that frame (same size, uint8 BGR / RGB) -> float mask 0..1 where the type is: |E - R| (max over channels) > thr, specks under
    MIN_AREA px dropped (codec noise), grown by `grow` px (anti-aliased edges, outline halos), feathered."""
    import cv2
    import numpy as np
    d = np.abs(E.astype(np.int16) - R.astype(np.int16)).max(axis=2).astype(np.uint8)
    m = changed_pixels(d, thr)
    if grow > 0 and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    m = m.astype(np.float32)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return np.clip(m * 1.0, 0.0, 1.0)


MAX_BBOX_SHARE = 0.24      # a word's ink box never covers more of the frame than this: over ~0.65 the type is dropped as broken (MAX_TYPE_SHARE), and a word of 0.36 popping solid -> outline in one frame still tripped overlay_check's cut test (box job 20261101-205838-a5d3 frames 75 / 112); the widest word of the passing runs was 0.24
MAX_TYPE_SHARE = 0.45      # type never covers more of a frame than this; a frame whose difference mask is bigger is BROKEN (a ground, a wash, a failed op), not type
STRICT_THR = 64            # the second look at a broken frame: only pixels that changed by more than this count


def restore_frame_ex(S, E, R, thr: int = 20, grow: int = 4, feather: float = 1.5, max_share: float = MAX_TYPE_SHARE, active: bool = True):
    """-> (output, info). output = S (the SOURCE frame) outside the type mask, E (the composed frame) inside it. S, E, R uint8 HxWx3 of one size (E / R are resized to S's when the engine ran on
    another size). The mask is a difference, so a ground painted over the whole picture would look like 'type'. Guards: (1) `active` False (no word / bridge can be on screen) -> the source frame;
    (2) the mask covers more than `max_share` of the frame -> broken: a second, strict look (changes > STRICT_THR, tight 2 px grow); still over the cap -> the source frame, type dropped for that frame.
    info = {share, state: ok | strict | dropped | inactive}."""
    import cv2
    import numpy as np
    h, w = S.shape[:2]
    if not active:
        return S.copy(), dict(share=0.0, state="inactive")
    if E.shape[:2] != (h, w):
        E = cv2.resize(E, (w, h), interpolation=cv2.INTER_LINEAR)
    if R.shape[:2] != (h, w):
        R = cv2.resize(R, (w, h), interpolation=cv2.INTER_LINEAR)
    m = type_mask(E, R, thr, grow, feather)
    share, state = float((m > 0.5).mean()), "ok"
    if share > max_share:
        m = type_mask(E, R, STRICT_THR, 2, feather)
        share, state = float((m > 0.5).mean()), "strict"
        if share > max_share:
            return S.copy(), dict(share=share, state="dropped")
    m = m[..., None]
    # S + m * (E - R): the SOURCE pixels plus what the engine ADDED over its own footage (the type). The footage inside the mask stays the source's, whatever colour matrix the engine assumed.
    return np.clip(S.astype(np.float32) + m * (E.astype(np.float32) - R.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8), dict(share=share, state=state)


def restore_frame(S, E, R, thr: int = 20, grow: int = 4, feather: float = 1.5):
    """restore_frame_ex without the info."""
    return restore_frame_ex(S, E, R, thr, grow, feather)[0]


def probe_video(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,color_space,color_primaries,color_transfer,color_range:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    s = (j.get("streams") or [{}])[0]
    n, d = (s.get("r_frame_rate") or "0/1").split("/")
    return dict(width=int(s.get("width") or 0), height=int(s.get("height") or 0), fps=float(n) / float(d or 1), fps_str=s.get("r_frame_rate"), frames=int(s.get("nb_read_frames") or 0),
                duration=float((j.get("format") or {}).get("duration") or 0.0), color_space=s.get("color_space"), color_primaries=s.get("color_primaries"), color_transfer=s.get("color_transfer"),
                color_range=s.get("color_range"))


def color_args(ps: dict) -> tuple[str, list]:
    """(scale filter, output flags) for the writer so the restored video has the SOURCE's colour description. ffmpeg decodes a clip with its tagged matrix (bt709 in the R16 source) and an untagged one
    as bt601; writing the frames back untagged (bt601) shifted every colour of a bt709 clip and cost ~2 levels of mean error on the whole picture. The matrix used to go RGB -> YUV is the tagged one
    (else bt601, what an untagged source was decoded with) and the tags are copied."""
    known = lambda v: v if v and v not in ("unknown", "unspecified", "reserved") else None      # noqa: E731
    cs = known(ps.get("color_space"))
    mat = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(cs or "", "bt601")
    flt = f"scale=in_range=full:out_range=tv:out_color_matrix={mat},format=yuv420p"
    flags = []
    for flag, v in (("-colorspace", cs), ("-color_primaries", known(ps.get("color_primaries"))), ("-color_trc", known(ps.get("color_transfer"))), ("-color_range", known(ps.get("color_range")))):
        if v:
            flags += [flag, v]
    return flt, flags


def _frames(path, w: int, h: int, limit: int | None = None, bt601: bool = False):
    """BGR uint8 frames of a video via an ffmpeg rawvideo pipe (no fps filter: every decoded frame, in order). Colours follow the clip's own tags (untagged = bt601); bt601=True ignores the tags
    (the engine decodes its inputs without them, so the render and its footage are compared in that one convention)."""
    import numpy as np
    vf = ["-vf", "scale=in_color_matrix=bt601:in_range=tv:out_range=full,format=bgr24"] if bt601 else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-an", *vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)          # passthrough: a shot clip is tagged 30/1 but its pts run on the 24 fps pipeline clock (setpts=N/(24*TB)); the default cfr rawvideo output would duplicate 1 frame in 5 and shift every later frame (K != 1 sources, 2026-11-02)
    n = 0
    try:
        while limit is None or n < limit:
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def restore_video(source, raw, shot_clips: list, spans: list, out, fps_str: str, thr: int = 20, grow: int = 4, feather: float = 1.5, crf: int = 14, active_frames=None, inactive_frames=None) -> dict:
    """Write `out` (video only) = for every frame i: the SOURCE frame i outside the type, the composed frame inside it.
    source      the window clip at the source size and fps (the user's pixels)
    raw         the render's mp4 (the engine's size, the 24 fps pipeline clock; frames map 1:1 to the source's)
    shot_clips  [s1_clip.mp4, ...] the clips the engine read; the first spans[k] frames of each are the engine's own footage R
    out         libx264 crf `crf`, yuv420p, tagged `fps_str` (the source's rate)
    active_frames  optional (first, last) source frames outside which no word / bridge can be on screen: those frames are the source frame
    inactive_frames  optional [(first, last), ...] source frames that are the source frame whatever the render shows (a word's form-in before its own shot)
    -> report {frames, source_size, raw_size, changed_share_mean, changed_share_max, broken_frames [..], dropped_frames [..]}"""
    import numpy as np
    ps, pr = probe_video(source), probe_video(raw)
    W, H = ps["width"], ps["height"]
    n_exp = int(sum(spans))
    if ps["frames"] < n_exp or pr["frames"] < n_exp:
        raise RuntimeError(f"restore: source has {ps['frames']} frames, render {pr['frames']}, the window needs {n_exp}")
    first = probe_video(shot_clips[0])
    cw, chh = first["width"], first["height"]

    def r_frames():
        for clip, n in zip(shot_clips, spans):
            yield from _frames(clip, cw, chh, int(n), bt601=True)
    cflt, cflags = color_args(ps)
    wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps_str, "-i", "-", "-vf", cflt, "-c:v", "libx264", "-preset", "medium",
                           "-crf", str(int(crf)), "-pix_fmt", "yuv420p", *cflags, "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    shares, n, broken, dropped = [], 0, [], []
    try:
        for S, E, R in zip(_frames(source, W, H, n_exp), _frames(raw, pr["width"], pr["height"], n_exp, bt601=True), r_frames()):
            o, info = restore_frame_ex(S, E, R, thr, grow, feather, active=(active_frames is None or active_frames[0] <= n <= active_frames[1]) and not any(a_ <= n <= b_ for a_, b_ in (inactive_frames or ())))
            if info["state"] in ("strict", "dropped"):
                broken.append(n)
            if info["state"] == "dropped":
                dropped.append(n)
            shares.append(float((np.abs(o.astype(np.int16) - S.astype(np.int16)).max(axis=2) > 0).mean()))
            wr.stdin.write(np.ascontiguousarray(o).tobytes())
            n += 1
    finally:
        wr.stdin.close()
        rc = wr.wait()
    if rc or n != n_exp:
        raise RuntimeError(f"restore: wrote {n} of {n_exp} frames (ffmpeg rc {rc})")
    return dict(frames=n, source_size=[W, H], raw_size=[pr["width"], pr["height"]], changed_share_mean=round(float(np.mean(shares)), 5), changed_share_max=round(float(np.max(shares)), 5), broken_frames=broken, dropped_frames=dropped)
