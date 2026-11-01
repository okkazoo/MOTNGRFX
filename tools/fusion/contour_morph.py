"""contour_morph: the DEFAULT word-to-word move whenever the shot has a subject matte (explore round 2, gallery r2b1 `word_to_contour_to_word`, user-approved 2026-10-30:
"make contour_morph the default word-to-word move whenever a subject matte exists"). The word's letters dissolve into the subject's contour (SDF morph), the contour re-forms into the next
word, and that word's letters fill as the strokes land. Built from existing ops only (explore_ops / explore_ops2 / matte_treatments): sdf_morph x2, mask_gate / mask_mul / split_region,
mask_max, knockout (letters = windows onto the shot, the subject in front), brush_edge(guard=False). The effect is gated into the shot's compose chain by `explore_effects.weave_chain`
(its own flag + matte buffer, so it never strips an explore look), exactly like the explore sections.

    plan(piece, holds, music=None, designed=False)   -> [event]: the word pairs it applies to (same shot, shot has a matte, 1.0 s <= gap <= MAX_GAP_S, not under an explore section)
    build_ops(ctx)                                    -> the ops of one event (called by explore_effects.build_ops("contour_morph", ctx))
    weave(piece, music, chk, size, designed=False)    THE HOOK (plan_rules._finish and design_step._finish, like contour_sweep.weave / weave_hold_life). Never raises.

TIMING (gap G = next word's hit - this word's release, piece seconds): morph 1 (word A -> the subject's contour) over [release, release + G/2], morph 2 (contour -> word B) over
[release + G/2, hit - 0.3]; ONE CONTINUOUS LINE (user 2026-10-30: the graphic's outline goes STRAIGHT into the text's outline, image-form outline -> in-between shape -> letter outline,
no stage where loose lines play on their own): the stroke follows the morphing shape the whole time; only once it is B's letter outline do the letters fill (left to right, hit - 0.3 to hit + 0.1)
and the outline thins away. The word shapes come from the type pass itself at a frame offset (`$type_matte@-N` = A's landed hold seen N frames earlier,
`@N` = B's landed hold N frames ahead), so the morph runs on the true landed letters while the engine's own exit / form-in animate underneath.

LINE SCALE (user liked): stroke width LINE_W x frame width (0.006 = brush_edge width_px 2.16 at its 360 px reference), thick-thin variation LINE_VAR 1.25 (`brush_edge(var=)`).
LEGIBILITY: A reads to its release (the section crossfades in 0.2 s before it, both looks read), B reads from its hit for max(1 beat, 0.6 s); letters are knockout windows (min_contrast 0.22).
A specific request `look` (anything but "auto") keeps its family's own transitions unless contour_morph is named for the gap. No matte on the shot, a gap under MIN_GAP_S / over MAX_GAP_S, a hold too short to sample, or a model-chosen explicit transition: nothing changes (the existing transition plays).
"""
from __future__ import annotations

NAME = "contour_morph"
FLAG = "fxcm"                    # ops of this effect carry this flag (explore_effects keeps its own `fx`)
MBUF = "cmm"                     # its matte buffer
MIN_GAP_S = 1.0                  # release -> next hit: at least this (0.5 s per half)
MAX_GAP_S = 2.4                  # above this the word has long gone: the engine's own transition plays
LINE_W = 0.006                   # stroke width, share of the frame width (user: 'i like the line scale')
LINE_VAR = 1.25                  # thick-thin variation of the stroke
REF_W = 360.0                    # brush_edge's width reference (width_px is per 360 px of width)
THIN_END = 0.42                  # the stroke is this share of its width when the word has landed
GATE_S = 0.2
EXPLICIT_AUTO = (None, "", "auto", "contour_morph")


def _tl(*pts):
    return {"timeline": [[round(float(a), 3), round(float(b), 3)] for a, b in pts], "ease": "smooth"}


def _kind(tr):
    if isinstance(tr, dict):
        return tr.get("kind") or tr.get("module")
    return tr


def explicit_transition(piece: dict, i: int):
    """the transition a pair carries, or None when it is the default / auto: word.transition (dict kind = pan ...) first, then type.transitions[i]."""
    words = (piece.get("type") or {}).get("words") or []
    tk = _kind(words[i].get("transition")) if i < len(words) and isinstance(words[i], dict) else None
    if tk in EXPLICIT_AUTO:
        trs = (piece.get("type") or {}).get("transitions")
        tk = _kind(trs[i]) if isinstance(trs, list) and i < len(trs) else None
    return None if tk in EXPLICIT_AUTO else tk


def look_of(piece: dict) -> str:
    """the request `look` (auto | strand | mech | slab | stack | ring | voxel | particles, or a module name; plan.look)."""
    return str(((piece.get("plan") or {}).get("look")) or "auto").strip().lower()


def asked(piece: dict, i: int) -> bool:
    """the model (or the request) NAMED contour_morph for gap i (design_step._morph_names_to_null records it before the engine sees a null)."""
    if i in (((piece.get("plan") or {}).get("contour_morph_asked")) or []):
        return True
    words = (piece.get("type") or {}).get("words") or []
    trs = (piece.get("type") or {}).get("transitions")
    return (i < len(words) and isinstance(words[i], dict) and _kind(words[i].get("transition")) == NAME) or (isinstance(trs, list) and i < len(trs) and _kind(trs[i]) == NAME)


def plan(piece: dict, holds, music: dict | None = None, designed: bool = False, fps: float | None = None) -> list[dict]:
    """the word pairs the morph applies to. `designed` = a model designed the piece: a transition it named is kept (the morph is only the default for null / auto / contour_morph);
    the rules planner's own assignment is a default and the morph replaces it on a matte shot."""
    from tools.fusion import contour_sweep as CS
    from tools.fusion import explore_effects as EE
    comp = piece.get("compose") or {}
    if comp.get("contour_morph") is False or comp.get("technique") == "traced":
        return []
    fps = float(fps or piece.get("fps") or 24)
    beat = CS._beat_s(piece, music)
    try:
        spans = EE._shot_spans(piece)
    except Exception:       # noqa: BLE001  no layout yet: no morph
        return []
    shots = {s.get("id"): s for s in piece.get("shots") or []}
    taken = [(e["t0"] - GATE_S, e["t1"] + GATE_S) for e in ((piece.get("plan") or {}).get("explore") or {}).get("placed") or []]
    out: list[dict] = []
    for i in range(len(holds) - 1):
        (ha, ra), (hb, rb) = holds[i], holds[i + 1]
        gap = hb - ra
        if gap < MIN_GAP_S - 1e-6 or gap > MAX_GAP_S:
            continue
        if i in (((piece.get("plan") or {}).get("contour_morph_skip")) or []):
            continue                                              # WORD RULES (tools/fusion/word_rules.py): a morph on the gap before / a third morph in the piece is dropped
        if look_of(piece) != "auto" and not asked(piece, i):
            continue                                              # a specific text look (strand / mech / ... or a module name) keeps that family's own word-to-word transitions
        tk = explicit_transition(piece, i)
        if tk is not None and (designed or str(tk) == "pan"):
            continue                                              # the model chose another transition (or a pan owns the gap): kept
        sid = next((s for s, (a, b) in spans.items() if a <= ra + 0.05 and hb + 0.3 <= b), None)
        if sid is None or not shots.get(sid, {}).get("matte"):
            continue
        half = gap / 2.0
        d_a = half / 2.0 + 0.1                                    # A's landed hold, seen this far back during the morph
        d_b = min(half, rb - hb - 0.05)                           # B's landed hold, seen this far ahead
        if ra - ha < d_a + 0.05 or d_b < half / 2.0:
            continue                                              # a hold too short to sample the landed letters
        t1 = min(hb + max(beat, 0.6), rb - 0.05)
        if t1 - hb < 0.35:
            continue
        t0 = ra
        if any(t0 - GATE_S < b and a < t1 + GATE_S for a, b in taken):
            continue
        out.append(dict(name=NAME, word=i, shot=sid, t0=round(t0, 3), t1=round(t1, 3),
                        params=dict(mid=round(ra + half, 3), land=round(hb, 3), rel=round(ra, 3), lag_a=int(round(d_a * fps)), lead_b=int(round(d_b * fps)))))
        taken.append((t0 - GATE_S, t1 + GATE_S))
    return out


def build_ops(ctx: dict) -> list[dict]:
    """the ops that make buffer ``fx`` for one event (ctx: colours, seed, plated, params{rel, mid, land, lag_a, lead_b}); the matte placeholder is `$fxm`."""
    c, seed = ctx["colours"], int(ctx.get("seed", 0))
    p = ctx["params"]
    rel, mid, land = float(p["rel"]), float(p["mid"]), float(p["land"])
    ta, tb = f"$type_matte@-{int(p['lag_a'])}", f"$type_matte@{int(p['lead_b'])}"
    m = "$fxm"
    form_t = land - 0.3                                                      # the outline has become B's letter outline here; ONLY THEN do the letters fill (one continuous line, no free strokes)
    sub = "$subject" if ctx.get("plated") else "$footage"
    w0 = LINE_W * REF_W
    return [
        {"op": "sdf_morph", "out": "cms1", "args": {"a": ta, "b": m, "k": _tl((rel - 0.01, 0.0), (rel, 0.0), (mid, 1.0)), "soft": 1.0}},
        {"op": "sdf_morph", "out": "cmsh", "args": {"a": "$cms1", "b": tb, "k": _tl((mid - 0.01, 0.0), (mid, 0.0), (form_t, 1.0), (land + 2.0, 1.0)), "soft": 1.0}},
        {"op": "mask_gate", "out": "cmwa", "args": {"a": ta, "gate": _tl((rel - 0.01, 1.0), (rel + 0.12, 0.0))}},
        {"op": "split_region", "out": "cmwipe", "args": {"like": "$footage", "axis": "x", "pos": _tl((form_t, -0.1), (land + 0.1, 1.15)), "soft": 0.14, "side": "left"}},
        {"op": "mask_mul", "out": "cmwb", "args": {"a": tb, "b": "$cmwipe"}},
        {"op": "mask_max", "out": "cmwin", "args": {"a": "$cmwa", "b": "$cmwb"}},
        {"op": "knockout", "out": "fx", "args": {"plate_rgb": c["black"], "type_a": "$cmwin", "footage": "$footage", "fg_matte": m, "fg": sub, "min_contrast": 0.22}},
        {"op": "brush_edge", "out": "fx", "args": {"img": "$fx", "mask": "$cmsh", "frame_idx": "$frame_idx", "color": c["accent"], "n_strokes": 2,
                                                 "width_px": _tl((rel, w0), (form_t, w0), (land + 0.35, w0 * THIN_END)), "var": LINE_VAR, "seed": seed, "guard": False,
                                                 "opacity": _tl((rel - 0.15, 0.0), (rel + 0.05, 1.0), (land, 1.0), (land + 0.45, 0.0))}},
    ]


def weave(piece: dict, music: dict | None = None, chk: dict | None = None, size=None, designed: bool = False) -> dict:
    """THE HOOK: plan the morphs (kept in plan.contour_morph) and gate them into the shots' chains. No pair = no change (earlier morph ops are removed: idempotent). Never raises."""
    pl = piece.setdefault("plan", {})
    comp = piece.get("compose") or {}
    chains = ([("default", comp["default"])] if comp.get("default") else []) + list((comp.get("shots") or {}).items())
    try:
        from tools.fusion import explore_effects as EE
        from tools.fusion import plan_rules as PR
        for _n, ch in chains:
            ch["ops"] = [o for o in ch.get("ops", []) if not o.get(FLAG)]
        if comp.get("contour_morph") is False or not chains:
            pl.pop("contour_morph", None)
            return piece
        holds = PR.word_holds(piece, music, chk)
        events = plan(piece, holds, music, designed)
        pl["contour_morph"] = dict(words=[e["word"] for e in events], events=events, designed=bool(designed))
        if not events:
            return piece
        sz = list(size or (piece.get("sizes") or {}).get("compose") or (540, 960))
        seed = int((pl.get("piece_system") or {}).get("seed", 0))
        base = dict(size=sz, colours=EE.colours(piece), seed=seed, plated=False,
                    plated_shots={str(s_.get("id")): str(s_.get("plate") or "").lower() == "clean" for s_ in piece.get("shots") or []})
        own = set((comp.get("shots") or {}))
        for nm, ch in chains:
            mine = [e for e in events if (e["shot"] not in own if nm == "default" else e["shot"] == nm)]
            ch["ops"] = EE.weave_chain(ch.get("ops", []), mine, base, flag=FLAG, mbuf=MBUF)
    except Exception as e:      # noqa: BLE001  never break a piece for an effect
        pl.setdefault("notes", []).append(f"contour_morph skipped: {type(e).__name__}: {str(e)[:120]}")
    return piece
