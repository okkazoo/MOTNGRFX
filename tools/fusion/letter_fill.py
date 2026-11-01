"""letter_fill: SOLID LETTERS BY DEFAULT (user 2026-11-01: "I'd like the text not to be filled with that textured pattern").

The inside of a letter is a solid ink from the shot's palette (legible_color's shot hues, picked at compose time) unless the USER's own direction asks for a material /
footage / pattern inside the letters (user_style.wants_textured_letters; piece.user_style.textured or compose.letter_fill = "texture" force it). Every textured mode stays in the
code and is used when asked; nothing is deleted. Applied DOWNSTREAM (every design source: the rules, Gemma, presets, own-video):

    owners      a word owned by LTX (render ltx / handoff with a `material`, e.g. "polished crimson marble") -> flat     (owner.apply_to_plan, plan time)
    plate modes a shot / word type_matte plate_in_type -> NONE (plain type over the footage); window_subject_front | text_as_mask (plate / footage in the letters) -> plate_in_subject (solid word, the subject
                silhouette is the window) (plate_layers.type_matte_of, compose time)
    knockout    letters as windows onto the footage (explore x14 letters_hold_subject, x02 plate_letters_subject_over, outline_to_word, contour_morph's letters): the window is
                filled with a solid ink of the piece palette that contrasts with the knockout ground (the ground, outlines and the subject in front stay)  (recipe time)
    window_cut  mode `other` (a word's letters show another shot while it holds) -> the standard type drawer (sandwich: the shot outside, solid type, the subject in front),
                which the legibility pass then inks from the shot's hues. cut / swap / zoom are SHOT CUTS through the letters and are kept.

    solid(piece) -> bool                     True = solid letters (the default)
    solid_mode(piece, mode) -> mode          the plate-route type-matte mode to use
    solid_ops(piece, ops) -> (ops, n)        the recipe ops with the textured fills replaced (n = ops changed)
    solid_owners(piece) -> [notes]           owners flattened in place
"""
from __future__ import annotations

import copy

# plate_in_type -> None (2026-11-01, render R17): the plate-filled silhouette on a FLAT ground is not a stand-in for "no textured letters" (it turned a whole shot into a grey ground with a ghost subject);
# a shot that asked for letters filled with the plate gets PLAIN type over its footage instead (no plate mode). The other two keep plate_in_subject (it is windowed to the word holds in plate_layers).
TEXTURED_MODES = {"plate_in_type": None, "window_subject_front": "plate_in_subject", "text_as_mask": "plate_in_subject"}


def solid(piece) -> bool:
    try:
        from tools.fusion import user_style as US
        return not US.wants_textured_letters(piece or {})
    except Exception:          # noqa: BLE001
        return True


def solid_mode(piece, mode):
    if mode in TEXTURED_MODES and solid(piece):
        return TEXTURED_MODES[mode]
    return mode


def _rgb01(c):
    try:
        v = [float(x) for x in list(c)[:3]]
    except (TypeError, ValueError):
        return None
    return [x / 255.0 for x in v] if max(v) > 1.0 else v


def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ink_against(piece, ground) -> list:
    """the piece-palette colour (accents / fg / bg) with the strongest luma contrast to `ground` (0..1 rgb); a near-white / near-black when the palette has none."""
    pal = (((piece.get("plan") or {}).get("piece_system") or {}).get("palette")) or (piece.get("type") or {}).get("palette") or {}
    cands = [_rgb01(a) for a in (pal.get("accents") or [])] + [_rgb01(pal.get("fg")), _rgb01(pal.get("bg"))]
    try:
        from tools.fusion import user_style as US
        cands = [_rgb01(c) for c in (US.parse_direction(piece).get("palette") or [])] + cands          # the user's palette first
    except Exception:          # noqa: BLE001
        pass
    cands = [c for c in cands if c]
    g = _rgb01(ground) or [0.05, 0.05, 0.05]
    lg = _lum(g)
    best = max(cands, key=lambda c: abs(_lum(c) - lg) + 0.15 * (max(c) - min(c)), default=None)
    if best is None or abs(_lum(best) - lg) < 0.35:
        best = [0.95, 0.94, 0.9] if lg < 0.5 else [0.06, 0.06, 0.07]
    return [round(float(x), 4) for x in best]


def solid_ops(piece, ops):
    if not solid(piece):
        return ops, 0
    out, n = [], 0
    for o in ops:
        a = o.get("args") or {}
        if o.get("op") == "knockout" and a.get("footage") is not None:
            o = copy.deepcopy(o)
            ground = a.get("plate_rgb") if isinstance(a.get("plate_rgb"), (list, tuple)) else None
            ink = ink_against(piece, ground)
            buf = f"_solid{n}"
            flags = {k: v for k, v in o.items() if k not in ("op", "out", "args")}
            out.append(dict(flags, op="fill", out=buf, args={"like": "$footage", "color": ink}))
            if a.get("fg_matte") is not None and a.get("fg") is None:
                o["args"]["fg"] = a["footage"]                    # the subject in front keeps its own footage
            o["args"]["footage"] = "$" + buf
            o["args"].pop("min_contrast", None)                   # solid ink: the window-legibility guard has nothing to do
            o["solid_fill"] = True
            n += 1
        elif o.get("op") == "window_cut" and str(a.get("mode", "swap")) == "other" and isinstance(a.get("type_a"), str):
            base = a.get("a")
            pre = []
            if not isinstance(base, str):                          # a flat ground colour: make it a buffer
                pre.append({"op": "fill", "out": f"_solidg{n}", "args": {"like": "$footage", "color": _rgb01(base) or [0.05, 0.05, 0.05]}})
                base = f"$_solidg{n}"
            matte = next((a[k] for k in ("fg_matte", "matte_a") if isinstance(a.get(k), str)), None)
            if matte is None:
                pre.append({"op": "mask_gate", "out": f"_solidz{n}", "args": {"a": a["type_a"], "gate": 0.0}})
                matte = f"$_solidz{n}"
            out += pre
            new = {"op": "sandwich", "out": o.get("out", "image"), "solid_fill": True,
                   "args": {"footage": base, "type_rgb": "$type_beauty", "type_a": a["type_a"], "matte": matte, "matte_threshold": 0.5, "matte_soft": 0.08,
                            **({"fg": a["fg"]} if isinstance(a.get("fg"), str) else {})}}
            o = new
            n += 1
        out.append(o)
    return out, n


def solid_owners(piece) -> list:
    """LTX-owned words (render ltx / handoff: the letters are an LTX-rendered material) become flat unless the user asked for textured letters. In place; -> notes."""
    words = ((piece.get("type") or {}).get("words")) or []
    if not solid(piece):                                   # the user asked for textured letters: an owner flattened by an earlier pass (no direction on the piece yet) is restored
        for w in words:
            if isinstance(w, dict) and w.get("render_asked") and str(w.get("render")) == "flat":
                w["render"] = w.pop("render_asked")
        return []
    notes = []
    for i, w in enumerate(words):
        if isinstance(w, dict) and str(w.get("render") or "").lower() in ("ltx", "handoff"):
            notes.append(f"letter_fill: word {i} {w.get('text')!r} render {w.get('render')} material {w.get('material')!r} -> flat (solid letters by default)")
            w["render_asked"], w["render"] = w.get("render"), "flat"
    return notes
