"""openings: how a piece OPENS, and variety of the way words come in (user 2026-10-29: "why do all the compositions start with the graphics coming in with a similar transition from the top?").
Every piece of the 24 GB batches (v005 r01 FIRST, r02 WAKE, v006 r01 WILD WIND, r02 DEEP BLUE, v007 r01-r03) opened with word 1 = form_in "lines" + module "lanes" = strands falling in from the top:
`lines` is first in every enum and in every example, and the strands' entry side defaults to `up` (mograph/modules/core.py Port, `contour_ports.split_port_spec` default "up"). Pure functions.

    SIDES / ORDER                      the sides a word's strands can arrive from (`entry`), "up" LAST in the rotation: it is the module default, not a design
    opening_of(piece) / summary(o)     the opening of a finished piece (word 1: form_in, entry, module; the first cut) and its one-line text (piece_summary carries it)
    parse_summary(text)                the opening back out of a `piece:` report line (tools/app/run_batch.py keeps it with the concept of the run)
    prompt_text(earlier)               the design prompt's line "earlier pieces opened with ... open differently"
    enforce(words, fixes, earlier, seed, protected)   (form_in, entry) is unique per word, >= 2 different form_in styles, word 1 does not repeat an earlier opening; a `lines` word that
                                                       names no entry gets one from the rotation (never the silent default "up"); every change is a fix string
    default_style(i, explicit, seed)   the form_in of a word that names none: one of the piece's own explicit styles, else a rotation, never "lines" by default
"""
from __future__ import annotations

import re

SIDES = ("up", "down", "left", "right", "in", "out")
ORDER = ("left", "right", "down", "in", "out", "up")          # rotation order of the entry sides: the default side "up" comes last
STYLE_ORDER = ("lines", "wipe", "flicker")     # a style for a duplicate: strand-carrying ones first (wipe / flicker turn the next gap into a cut). 2026-11-02: no `blobs` / `particles` (design_defaults.SPECKLE_IO, banned 2026-11-01)
SECOND_STYLE = ("lines", "wipe")                              # the second style of a piece that only used one: strands first, so transitions still work
DEFAULT_ENTRY = "up"


def _rot(seq, k: int):
    k = int(k) % max(len(seq), 1)
    return tuple(seq[k:]) + tuple(seq[:k])


def effective_entry(w: dict) -> str:
    e = w.get("entry")
    return e.strip().lower() if isinstance(e, str) and e.strip().lower() in SIDES else DEFAULT_ENTRY


def style_of(w: dict) -> str:
    return str(w.get("form_in") or "lines")


def key_of(w: dict) -> tuple:
    """what the eye sees of an entry: the style, and for `lines` the side the strands come from (the other styles have no side)."""
    st = style_of(w)
    return (st, effective_entry(w) if st == "lines" else None)


READABLE_MODULES = ("lanes", "pipework", "circuit", "slats", "gradient_fade", "staircase", "beat_bars")        # the modules design_defaults keeps; none is "the default"


def default_module(i: int, seed: int = 0) -> str:
    """the module of a word that names none and has no sibling that does: a rotation seeded per piece (the same for every word of the piece: ONE system)."""
    return _rot(READABLE_MODULES, seed)[0]


def default_style(i: int, explicit: list, seed: int = 0) -> str:
    """form_in of word i when the reply names none. The piece's own explicit styles come first (cycled), else a rotation seeded per piece: never `lines` by default."""
    ex = [x for x in explicit if x]
    if ex:
        return ex[i % len(ex)]
    return _rot(("wipe", "lines", "flicker"), seed + i)[0]


# ----------------------------------------------------------------------------------------------------------------------------------------- the opening of a piece
def opening_of(piece: dict) -> dict | None:
    ws = ((piece.get("type") or {}).get("words")) or []
    if not ws:
        return None
    w = ws[0]
    cuts = (piece.get("compose") or {}).get("cuts") or []
    c = cuts[0] if cuts and isinstance(cuts[0], dict) else {}
    return dict(word=str(w.get("text", "")), form_in=style_of(w), entry=effective_entry(w), module=str(w.get("module") or ""), cut=str(c.get("was") or c.get("cut") or "none"))


def summary(o: dict | None) -> str:
    if not o:
        return ""
    return f"opening={o['form_in']}/{o['entry']}/{o['module']} first_cut={o['cut']}"


_SUM_RE = re.compile(r"opening=(\w+)/(\w+)/(\w+) first_cut=(\S+)")


def parse_summary(text: str) -> dict | None:
    m = _SUM_RE.search(str(text or ""))
    return dict(form_in=m.group(1), entry=m.group(2), module=m.group(3), cut=m.group(4)) if m else None


def prompt_text(earlier) -> str:
    """the line of the design prompt about the openings of the earlier pieces of this batch ('' when there are none)."""
    rows = [o for o in (earlier or []) if isinstance(o, dict) and o.get("form_in")]
    if not rows:
        return ""
    ops = "; ".join(f"{i}. {o['form_in']} entering from {o.get('entry', 'up')} on {o.get('module', '?')}, first cut {o.get('cut', '?')}" for i, o in enumerate(rows, 1))
    return (f"EARLIER PIECES OF THIS BATCH opened like this: {ops}. Open THIS piece differently: another form_in or another `entry` side for word 1 (and another module if you can), and never "
            f"the same opening as the piece before it.")


def _opening_key(o: dict) -> tuple:
    return (o.get("form_in"), o.get("entry") if o.get("form_in") == "lines" else None)


# ----------------------------------------------------------------------------------------------------------------------------------------- the rule
def enforce(words: list, fixes: list, earlier=(), seed: int = 0, protected=(), system=None) -> int:
    """(form_in, entry side) is unique per `lines` word; the piece uses at least two different form_in styles; word 1 does not open like an earlier piece; a `lines` word with no `entry` of its own
    gets one (the module's silent default is "up" for every word, which is how every piece came to open with strands from the top). Words in `protected` are left alone. -> number of fields changed.
    A clash is mended in this order: the SIDE first (only `lines` has one), then the STYLE, changed to another style of the piece's own system (`system` = the piece system's io pair, else
    the styles the piece already uses, else STYLE_ORDER) within MAX_FORMS styles; never to `lines` by default. Side-less styles (blobs, particles, wipe, flicker) that repeat across words are the
    piece's own pair and stay: the system alternates two styles over the words, only the opening of the piece before it and the lack of a second style are mended."""
    n = len(words)
    if not n:
        return 0
    changed = 0
    taken = {_opening_key(o) for o in (earlier or []) if isinstance(o, dict) and o.get("form_in")}
    last_prev = _opening_key(list(earlier)[-1]) if earlier else None
    seen: set = set()
    from tools.fusion.design_defaults import MAX_FORMS                              # the piece keeps at most MAX_FORMS io styles (a third one lengthens the forms and the fold would undo it)
    sys_in = [x for x in ((system or {}).get("io") or {}).get("form_in", []) if x]
    sys_sides = [x for x in ((system or {}).get("io") or {}).get("entries", []) if x in SIDES]
    side_order = tuple(dict.fromkeys(sys_sides + list(ORDER)))

    def free_side(i, banned):
        for s_ in _rot(side_order, seed + i):
            if ("lines", s_) not in banned:
                return s_
        return _rot(side_order, seed + i)[0]

    def other_style(i, banned):
        """a style for word i other than its own, from the piece's own system, within MAX_FORMS distinct styles, whose key is not banned; None when there is none."""
        cur = style_of(words[i])
        rest = {style_of(w) for j, w in enumerate(words) if j != i}
        for x in sys_in + [style_of(w) for w in words] + list(STYLE_ORDER):
            if x == cur or x not in ("lines", "blobs", "particles", "wipe", "flicker") or len(rest | {x}) > MAX_FORMS:
                continue
            if x == "lines" or (x, None) not in banned:
                return x
        return None
    for i, w in enumerate(words):
        if i in protected:
            seen.add(key_of(w))
            continue
        banned = ({k for k in seen if k[0] == "lines"}) | ({last_prev} if i == 0 and last_prev else set()) | (taken if i == 0 else set())      # two side-less words with one style are fine; two `lines` from one side are not
        if style_of(w) == "lines" and not isinstance(w.get("entry"), str):          # no side named: the default "up" would be the design
            new = free_side(i, banned)
            fixes.append(f"type.words[{i}].entry omitted: lines from the top is the module default, not a design; {new!r} chosen for this word")
            w["entry"] = new
            changed += 1
        k = key_of(w)
        if k in banned:
            if style_of(w) == "lines":                                            # the side first
                new = free_side(i, banned)
                fixes.append(f"type.words[{i}].entry {effective_entry(w)!r} -> {new!r}: " + ("the piece before opened with the same entry" if i == 0 and k in taken | ({last_prev} if last_prev else set()) else "no two words enter the same way"))
                w["entry"] = new
                changed += 1
            else:                                                                  # a side-less style cannot change side: another style of the piece's own system
                new = other_style(i, banned)
                if new is not None:
                    fixes.append(f"type.words[{i}].form_in {style_of(w)!r} -> {new!r}: the piece before opened with the same style")
                    w["form_in"] = new
                    if new == "lines":
                        w["entry"] = free_side(i, banned)
                    else:
                        w.pop("entry", None)
                    changed += 1
        seen.add(key_of(w))
    styles = {style_of(w) for w in words}
    if n >= 2 and len(styles) < 2:                                                   # one style for the whole piece = one way in: the middle word takes a second one
        j = next((x for x in range(n // 2, n) if x not in protected), None)
        if j is not None:
            cur = style_of(words[j])
            new_style = next((s_ for s_ in sys_in + list(_rot(SECOND_STYLE, seed)) if s_ != cur), "wipe")
            fixes.append(f"type.words[{j}].form_in {cur!r} -> {new_style!r}: a piece uses at least two different io styles")
            words[j]["form_in"] = new_style
            if new_style == "lines":
                words[j]["entry"] = free_side(j, {k for k in seen if k[0] == "lines"})
            else:
                words[j].pop("entry", None)
            changed += 1
    return changed
