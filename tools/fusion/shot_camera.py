"""shot_camera: a different camera in every shot of a piece (user 2026-10-29: "when we do multi shot I'd like to have the camera be quite different on each shot"). Pure functions on the
concept's shot plan, no ComfyUI, no network.

Each shot has a FRAMING (how close / from where) and a MOVE (what the camera does). The rules, enforced here after the model wrote them (and told to the models in guide section 21):
    - no two consecutive shots share their framing, and no two consecutive shots share their move;
    - no (framing, move) pair repeats in a piece;
    - a continuation shot (`continues`) keeps the place but changes the camera like any other (wide -> close-up, or the move changes).
The camera text lands in the shot prompt's `camera:` clause (plan_rules.model_shot_prompt) and the framing in the still prompt of a new place (plan_rules.place_still).

    parse(text)                -> (framing | None, move | None)   read from free camera text ("slow push in on a close-up")
    enforce(entries, n)        -> (entries, [fix text])           the shot_plan entries [{prompt, camera, ..}] with every repeat repaired field by field
    describe(framing, move)    -> the camera text written into the entry
"""
from __future__ import annotations

import re

FRAMINGS = ("wide", "medium", "close-up", "extreme close-up", "low angle", "high angle", "over-the-shoulder", "profile")
MOVES = ("static", "push in", "pull back", "pan left", "pan right", "tilt up", "tilt down", "orbit", "tracking", "crane up", "crane down", "handheld drift")
# the order candidates are tried in (rotated by the shot index): contrast first (wide <-> close), then angle changes
FRAMING_ORDER = ("wide", "close-up", "medium", "low angle", "extreme close-up", "high angle", "profile", "over-the-shoulder")
MOVE_ORDER = ("tracking", "push in", "pull back", "orbit", "pan right", "crane up", "tilt down", "static", "pan left", "handheld drift", "tilt up", "crane down")

_FRAMING_RE = (("extreme close-up", r"\b(extreme close[- ]?up|ecu|macro|detail shot|insert shot)\b"), ("close-up", r"\b(close[- ]?up|close on|tight (?:shot|on|frame)|closer on|face fills)\b"),
               ("over-the-shoulder", r"\bover[- ]the[- ]shoulder\b"), ("low angle", r"\b(low[- ]angle|from below|worm'?s[- ]eye|looking up at)\b"),
               ("high angle", r"\b(high[- ]angle|top[- ]down|overhead|from above|bird'?s[- ]eye|aerial)\b"), ("profile", r"\b(profile|side view|from the side|side-on)\b"),
               ("wide", r"\b(wide|establishing|long shot|far away|vast)\b"), ("medium", r"\b(medium|mid[- ]shot|waist[- ]up|mid-range|mid range)\b"))
_MOVE_RE = (("crane up", r"\b(crane up|boom up|rises? up|ascend\w*|lifts? up)\b"), ("crane down", r"\b(crane down|boom down|descend\w*)\b"),
            ("push in", r"\b(push(?:es|ing)?[- ]?in|dolly(?:ing)? in|zoom(?:s|ing)? in|move(?:s|ing)? closer|creeps? in|slowly closer|pushes? toward)\b"),
            ("pull back", r"\b(pull(?:s|ing)?[- ]?back|dolly(?:ing)? out|backs? away|backing away|retreat\w*|pulls? out|widen\w*)\b"),
            ("pan left", r"\bpan(?:s|ning)? (?:to the )?left\b|\bleft pan\b"), ("pan right", r"\bpan(?:s|ning)? (?:to the )?right\b|\bright pan\b|\bsweep from left to right\b"),
            ("tilt up", r"\btilt(?:s|ing)? up\b"), ("tilt down", r"\btilt(?:s|ing)? down\b"), ("orbit", r"\b(orbit\w*|circl\w+ (?:around|the)|arc(?:s|ing)? around|revolv\w+)\b"),
            ("tracking", r"\b(track\w*|dolly(?:ing)? alongside|follow\w*|alongside|lateral|truck(?:s|ing)?)\b"), ("handheld drift", r"\b(handheld|hand-held|drift\w*|float\w*|sway\w*)\b"),
            ("static", r"\b(static|locked[- ]off|lock(?:ed)?|still camera|tripod|fixed camera|holds? still)\b"))


def parse(text) -> tuple[str | None, str | None]:
    """(framing, move) named in a free camera text; None for what it does not name."""
    t = str(text or "").lower()
    fr = next((k for k, rx in _FRAMING_RE if re.search(rx, t)), None)
    mv = next((k for k, rx in _MOVE_RE if re.search(rx, t)), None)
    return fr, mv


def describe(framing: str, move: str) -> str:
    """the camera text of an entry: 'close-up, slow push in' (the words the video model reads)."""
    word = {"static": "locked-off static camera", "handheld drift": "handheld drift", "tracking": "camera tracks alongside the subject", "orbit": "camera orbits the subject",
            "push in": "slow push in", "pull back": "slow pull back", "pan left": "slow pan left", "pan right": "slow pan right", "tilt up": "tilt up", "tilt down": "tilt down",
            "crane up": "crane up", "crane down": "crane down"}[move]
    return f"{framing}, {word}"


def _strip_move(prompt: str, move: str) -> str:
    """the prompt without the comma-clauses that name camera move `move` (so a repaired camera does not contradict the prompt's own: 'the camera pushes in' next to 'pull back')."""
    rx = next((r for k, r in _MOVE_RE if k == move), None)
    if rx is None:
        return prompt
    parts = [p.strip() for p in re.split(r",\s*", str(prompt)) if p.strip()]
    return ", ".join(p for p in parts if not re.search(rx, p, re.I))


def _pick(order, k, bad):
    n = len(order)
    return next((order[(k + j) % n] for j in range(n) if order[(k + j) % n] not in bad), order[k % n])


def enforce(entries, n: int, scene: str = "") -> tuple[list, list[str]]:
    """the shot-plan entries ([{prompt, camera, continues?}, ...], up to n) -> (entries with distinct cameras, fixes). A missing camera is chosen; a shot that repeats the framing or the move of
    the shot before it, or a (framing, move) pair used earlier, has the offending part replaced by the first option (rotated by the shot index) that breaks no rule. The entry's `camera`
    becomes describe(framing, move) when anything changed. Shots beyond the entries are not invented here (shot_plan_of pads them first)."""
    out, fixes, used = [], [], set()
    prev_fr = prev_mv = None
    for k, e in enumerate(list(entries)[:n]):
        e = dict(e) if isinstance(e, dict) else {}
        if not e:
            out.append(e)
            prev_fr = prev_mv = None
            continue
        fr0, mv0 = parse(e.get("camera"))
        if fr0 is None and mv0 is None:
            fr1, mv1 = parse(e.get("prompt"))
            fr0, mv0 = fr1, mv1
        fr, mv = fr0, mv0
        why = []
        if fr is None or fr == prev_fr:
            why.append("no framing" if fr is None else f"framing {fr!r} repeats the shot before")
            fr = _pick(FRAMING_ORDER, k, {prev_fr, *(f for f, m in used if m == mv)} - {None})
        if mv is None or mv == prev_mv:
            why.append("no move" if mv is None else f"move {mv!r} repeats the shot before")
            mv = _pick(MOVE_ORDER, k, {prev_mv, *(m for f, m in used if f == fr)} - {None})
        if (fr, mv) in used:
            why.append(f"({fr}, {mv}) is already used in the piece")
            mv = _pick(MOVE_ORDER, k + 1, {prev_mv, *(m for f, m in used if f == fr)} - {None})
            if (fr, mv) in used:
                fr = _pick(FRAMING_ORDER, k + 1, {prev_fr, *(f for f, m in used if m == mv)} - {None})
        if why:
            new = describe(fr, mv)
            if mv != mv0 and mv0 and e.get("prompt"):
                e["prompt"] = _strip_move(e["prompt"], mv0) or e["prompt"]
            fixes.append(f"shot_plan[{k}].camera {e.get('camera')!r} -> {new!r}: " + "; ".join(why) + " (a different camera in every shot)")
            e["camera"] = new
        used.add((fr, mv))
        prev_fr, prev_mv = fr, mv
        out.append(e)
    out += list(entries)[n:]
    return out, fixes


def framing_of(entry) -> str | None:
    """the framing of a shot-plan entry (its camera text, else its prompt) for the still prompt of a new place."""
    if not isinstance(entry, dict):
        return None
    return parse(entry.get("camera"))[0] or parse(entry.get("prompt"))[0]


# ----------------------------------------------------------------------------------------------------------------------------------- every shot of the piece, not just the lockups
LEADS = {"wide": "Wide establishing shot", "medium": "Medium shot", "close-up": "Tight close-up", "extreme close-up": "Extreme close-up on a detail", "low angle": "Low-angle shot from the ground",
         "high angle": "High-angle shot looking down", "over-the-shoulder": "Over-the-shoulder shot", "profile": "Side-profile shot"}


def distinct(entries, n: int) -> tuple[list, list[str]]:
    """the shot-plan entries padded to ONE per shot (n) with the camera of every shot DISTINCT in the piece (regression 2026-10-29, draft v014 r02: the concept wrote 4 entries for a 6-shot piece, the
    last two shots had no camera, and the framings of the rest were only checked against the neighbour; all six stills were the same medium shot). Framing: no two shots of the piece share one while
    n <= len(FRAMINGS); move: no two consecutive share one, no (framing, move) pair repeats. An entry that already satisfies this is left alone; a padded entry gets only a `camera`.
    -> (n entries, fixes)."""
    ents = [dict(e) if isinstance(e, dict) else {} for e in list(entries)[:n]]
    ents += [{} for _ in range(max(0, n - len(ents)))]
    fixes: list[str] = []
    unique = n <= len(FRAMINGS)
    used_fr: list[str] = []
    used_pairs: set = set()
    prev_fr = prev_mv = None
    for k, e in enumerate(ents):
        fr0, mv0 = parse(e.get("camera"))
        if fr0 is None and mv0 is None:
            fr0, mv0 = parse(e.get("prompt"))
        fr, mv = fr0, mv0
        why = []
        bad_fr = ({prev_fr} | (set(used_fr) if unique else set())) - {None}
        if fr is None or fr in bad_fr:
            why.append("no framing" if fr is None else f"framing {fr!r} is already used" if fr in used_fr else f"framing {fr!r} repeats the shot before")
            fr = _pick(FRAMING_ORDER, k, bad_fr)
        if mv is None or mv == prev_mv:
            why.append("no move" if mv is None else f"move {mv!r} repeats the shot before")
            mv = _pick(MOVE_ORDER, k, {prev_mv, *(m for f, m in used_pairs if f == fr)} - {None})
        if (fr, mv) in used_pairs:
            why.append(f"({fr}, {mv}) is already used in the piece")
            mv = _pick(MOVE_ORDER, k + 1, {prev_mv, *(m for f, m in used_pairs if f == fr)} - {None})
        if why:
            if mv != mv0 and mv0 and e.get("prompt"):
                e["prompt"] = _strip_move(e["prompt"], mv0) or e["prompt"]
            e["camera"] = describe(fr, mv)
            fixes.append(f"shot {k + 1}: camera -> {e['camera']!r}: " + "; ".join(why))
        used_fr.append(fr)
        used_pairs.add((fr, mv))
        prev_fr, prev_mv = fr, mv
    return ents, fixes


def lead(entry) -> str:
    """the camera text a shot / still prompt LEADS with: 'Tight close-up, slow push in' (framing + move of the entry's camera); '' when the entry names no framing."""
    if not isinstance(entry, dict):
        return ""
    fr, mv = parse(entry.get("camera"))
    if fr is None:
        return ""
    word = describe(fr, mv or "static").split(", ", 1)[1] if mv else ""
    return LEADS[fr] + (f", {word}" if word else "")
