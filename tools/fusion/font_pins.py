"""font_pins: the user's per-line font choice (STORYBOARD v2, 16_app/MEDIA_BOX.md; brief.story.shots[].font) WINS over every font rule of the planner / designer.

One line on screen = one lockup = one entry of ``type.words``. The choice travels as ``pins``: a list aligned with the request's lines, ``[{"text": <line>, "font": <liked file or None>}, ...]``
(None = the user did not choose: the random piece system, the contrast rule and the designer's font rules decide as before). Route of a pin:

    page story.shots[k].font -> concept.parse_brief -> concept.request_extras ("fonts": [file|null, ..] aligned with the words) -> MographRequest.owners extras
    -> plan_rules.normalize_request (``pins_of``) -> plan_rules._build (the word wears it; ``type.font_pins`` rides on the piece)
    -> design_step / qwen_piece.finish_design + merge_locked (re-applied after the model + every font rule) -> spec.build_scene (the last guard: whatever a later step wrote, the type
    passes are drawn with the pinned face) -> compose / LTX control passes read those type passes.

Only liked faces (mograph/fonts/liked.json via mograph.fonts_liked): a pin that is not liked is substituted by the liked face of its role (production rule) or raises in strict mode.
Matching a word to its pin: equal text (alphanumerics, case-insensitive), then one contains the other (lockups that fit_lockups merged), then the same index when both lists have the same length.
"""
from __future__ import annotations

import re

KEY = "font_pins"


def _norm(s) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def _file_of(v) -> str | None:
    """liked.json file of a font choice (file, key, family, {font|file|key|name: ..}); None for an empty choice. Not liked -> substituted (strict mode raises FontNotLiked)."""
    if isinstance(v, dict):
        v = v.get("font") or v.get("file") or v.get("key") or v.get("name")
    if v is None or not str(v).strip():
        return None
    from mograph import fonts_liked as fl
    return fl.require_liked(str(v).strip())["file"]


def pins_of(*vs, lines=None) -> list:
    """the request's pins from the value(s) given: a list of font choices aligned with ``lines`` (``fonts`` of the owners extras object / the request), or already-made pins
    ``[{"text", "font"}]``. -> [] when nothing is chosen (a request without a user font is byte-identical to before), else one entry per line."""
    lines = [str(x) for x in (lines or [])]
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            import json
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            v = v.get("fonts") if v.get("fonts") is not None else v.get(KEY)
        if not isinstance(v, (list, tuple)) or not v:
            continue
        out = []
        for k, x in enumerate(v):
            text = x.get("text") if isinstance(x, dict) and x.get("text") else (lines[k] if k < len(lines) else "")
            out.append({"text": str(text), "font": _file_of(x)})
        if any(p["font"] for p in out):
            return out
    return []


def pin_for(text, i: int, n: int, pins) -> str | None:
    """liked font file chosen for the word ``text`` (index i of n words), or None."""
    if not pins:
        return None
    t = _norm(text)
    for p in pins:                                              # 1. the same line
        if t and _norm(p.get("text")) == t:
            return p.get("font")
    for p in pins:                                              # 2. a lockup that merged (or split) lines
        pt = _norm(p.get("text"))
        if t and pt and p.get("font") and (pt in t or t in pt):
            return p.get("font")
    if len(pins) == n and 0 <= i < n:                           # 3. same count: same position
        return pins[i].get("font")
    return None


def pinned_indices(words, pins) -> set:
    n = len(words or [])
    return {i for i, w in enumerate(words or []) if isinstance(w, dict) and pin_for(w.get("text"), i, n, pins)}


def apply(words, pins, fixes=None) -> int:
    """the pinned faces onto ``words`` (in place). A word that changes face loses its axes (they belonged to the other face). -> number of words changed."""
    if not pins or not words:
        return 0
    from mograph import fonts_liked as fl
    n, changed = len(words), 0
    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        pf = pin_for(w.get("text"), i, n, pins)
        if not pf:
            continue
        cur = fl.lookup(w.get("font")) if w.get("font") else None
        want = fl.lookup(pf)
        if cur is None or want is None or cur["file"] != want["file"]:
            old = w.get("font")
            w["font"] = pf
            w.pop("axes", None)
            changed += 1
            if fixes is not None:
                fixes.append(f"type.words[{i}].font {old} -> {pf}: the user's choice for this line (font pin)")
        w["font_pin"] = pf
    return changed


def applied(words, pins):
    """``words`` with the pins applied on COPIES of the words that change (the input is not touched); the same list when there is nothing to do."""
    if not pins:
        return words
    out = [dict(w) if isinstance(w, dict) else w for w in words]
    apply(out, pins)
    return out


def contrast(words, pins, fixes=None) -> int:
    """after the pins are on: an UNCHOSEN line that wears the same face as a neighbour takes another face the piece already uses that differs from both neighbours (a chosen line is never
    touched; no new face enters the piece). -> number of words changed."""
    if not pins or not words:
        return 0
    from mograph import fonts_liked as fl
    n, changed = len(words), 0
    pinned = pinned_indices(words, pins)

    def key(i):
        e = fl.lookup(words[i].get("font")) if 0 <= i < n and words[i].get("font") else None
        return e["file"] if e else None
    for i, w in enumerate(words):
        if i in pinned or not key(i) or key(i) not in (key(i - 1), key(i + 1)):
            continue
        used = [k for k in dict.fromkeys(key(j) for j in range(n)) if k]
        alt = [k for k in used if k not in (key(i - 1), key(i + 1), key(i))]
        if alt:
            old = w["font"]
            w["font"] = alt[0]
            w.pop("axes", None)
            changed += 1
            if fixes is not None:
                fixes.append(f"type.words[{i}].font {old} -> {alt[0]}: the same face as a neighbouring line (contrast; this line was not chosen by the user)")
    return changed
