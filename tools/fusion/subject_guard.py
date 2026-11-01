#!/usr/bin/env python
"""SUBJECT FROM THE USER'S PICTURE (bug, render archive mvxblalqa2y, 2026-11-01): image-mode renders changed the uploaded subject. The drummer still (media read `shows`:
"A drummer playing a kit in a flooded gothic cathedral with light beams.") became "a gaunt man, pale skin, wearing a crimson silk tunic and gold rings" in every LTX shot prompt.
LTX conditions only frame 0 (LTXVImgToVideoInplace strength 1.0), so by frame ~48 the clip turned into the prompt's red-robed man; an earlier render made the drummer an owl.
The user's direction ("more legible text, palette: crimson, gold") had leaked into the subject's clothing.

Pure functions (no model, no ComfyUI), used by tools/fusion/concept.py (the post-parse guard on the shots that start from a user image) and tools/app/layered_shots.py (the
scrub of the prompt the clean and the text-pass chains both read):

    subject_text(shows)               the shot's subject text = the image's media-read `shows`, trimmed (replaces Gemma's `subject_desc` for that shot)
    flagged(text, allowed)            words of `text` that introduce a colour / clothing / jewellery / body / person / animal word that `allowed` does not contain
    guard_prompt(prompt, shows, ok)   the shot prompt of a user-image shot: offending clauses removed, loose colour words deleted, the shows subject in front
    strip_framing(prompt)             "A close-up of X" -> "X" (the camera lead of the shot already names the framing; two framings contradict)
    dedupe_clauses(prompt)            identical comma-separated clauses (case / whitespace blind) kept once; run AFTER the framing is stripped (render R17: 'a drummer in a flooded
                                      cathedral ..., a drummer in a flooded cathedral ...' was two different clauses until 'A close-up of' came off one of them)
    palette_names(piece)              the colour names of the user's typed palette (piece.user_style.palette -> names)
    scrub_prompt(prompt, piece, sid)  drops those palette names from the prompt of a shot that starts from a user image, unless the image's `shows` has them
"""
from __future__ import annotations

import re

from tools.fusion import user_style as US

EXTRA_COLOURS = ("burgundy maroon ochre bronze copper charcoal beige tan khaki auburn vermilion sapphire ruby blonde blond ginger platinum").split()
COLOURS = frozenset(set(US.NAMES) | set(EXTRA_COLOURS))
CLOTHING = frozenset(("tunic robe robes cloak coat jacket dress gown suit shirt hoodie hood hat cap helmet crown scarf glove gloves boot boots shoe shoes sandal sandals armor armour "
                      "uniform vest sweater cape mantle apron skirt trousers jeans pants veil mask turban kimono poncho overcoat waistcoat bandana headband "
                      "ring rings necklace earring earrings bracelet jewelry jewellery pendant amulet tiara").split())
BODY = frozenset(("gaunt pale skin skinned tattooed bald hair haired beard bearded wrinkled freckled scarred muscular emaciated").split())
PEOPLE = frozenset(("man men woman women boy girl person people figure stranger elder child children kid teenager monk knight priest priestess warrior king queen soldier sailor fisherman "
                    "wizard witch sorcerer lady gentleman guy").split())
ANIMALS = frozenset(("owl cat dog bird horse wolf fox deer stag lion tiger bear raven crow eagle hawk falcon fish whale dragon rabbit hare snake monkey cow sheep elephant "
                     "pigeon swan duck butterfly spider bat mouse rat frog turtle shark dolphin octopus lizard goat pig").split())
SUBJECT_WORDS = CLOTHING | BODY | PEOPLE | ANIMALS                    # a clause with one of these (absent from `shows`) is dropped whole
FRAMING_WORDS = ("close-up", "close up", "closeup", "wide shot", "establishing shot", "tight", "medium shot")      # what the tests / reports count as a framing phrase

_WORD = re.compile(r"[a-z][a-z']*", re.I)
_FRAMING = re.compile(
    r"\b(?:an?\s+)?(?:extreme\s+)?close[- ]?up(?:\s+(?:shot|frame))?(?:\s+(?:of|on))?\b"
    r"|\b(?:an?\s+)?(?:(?:wide|medium|long|tight|establishing)\s+){1,2}(?:shot|frame|angle)(?:\s+(?:of|on))?\b"
    r"|\b(?:an?\s+)?(?:low|high)[- ]angle\s+(?:shot|view)(?:\s+(?:of|on))?\b"
    r"|\b(?:an?\s+)?over[- ]the[- ]shoulder\s+shot(?:\s+of)?\b"
    r"|\b(?:tight|close)\s+on\b|\btight\s+(?:framing|frame)\b|\bcloser\s+on\b", re.I)
SHOT_MAX_WORDS = 70


def norm(w: str) -> str:
    """lower case, no possessive, plural 's' dropped ('rings' -> 'ring', 'glass' stays)."""
    w = w.lower().strip("-'")
    w = re.sub(r"'s$", "", w)
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text) -> set:
    return {norm(m.group(0)) for m in _WORD.finditer(str(text or ""))}


_FLAG = frozenset(norm(w) for w in (COLOURS | SUBJECT_WORDS))
_COLOUR = frozenset(norm(w) for w in COLOURS)


def flagged(text, allowed) -> list:
    """the colour / clothing / body / person / animal words of `text` whose normal form is not in `allowed` (a set of normal forms, see tokens())."""
    out = []
    for m in _WORD.finditer(str(text or "")):
        n = norm(m.group(0))
        if n in _FLAG and n not in allowed and m.group(0).lower() not in out:
            out.append(m.group(0).lower())
    return out


def subject_text(shows, max_words: int = 24) -> str:
    """the subject text of a shot that starts from a user image: its media-read `shows` (one sentence, no final stop, at most max_words words); '' without one."""
    s = re.sub(r"\s+", " ", str(shows or "")).strip().strip(" ,;.")
    return " ".join(s.split()[:max_words]).strip(" ,;.")


def dedupe_clauses(prompt: str) -> str:
    """the comma-separated clauses of a prompt with every repeat dropped (compared case, whitespace, trailing stop and a leading 'a / an / the' blind); the first one stays, order kept."""
    seen, out = set(), []
    for c in re.split(r"\s*,\s*", str(prompt or "")):
        c = re.sub(r"\s+", " ", c).strip()
        key = re.sub(r"^(?:an?|the)\s+", "", c.strip(" .;").lower())
        if not c or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return ", ".join(out)


def strip_framing(prompt: str) -> str:
    """the framing phrases of a model shot prompt removed ('A close-up of the drummer, wide shot' -> 'the drummer'): the shot's camera lead ('Wide establishing shot, handheld drift')
    is the one framing of the final prompt. Clauses that became identical once the framing came off are kept once (dedupe_clauses)."""
    p = _FRAMING.sub(" ", str(prompt or ""))
    parts = [re.sub(r"\s+", " ", x).strip(" -") for x in re.split(r"\s*,\s*", p)]
    return dedupe_clauses(", ".join(x for x in parts if x).strip(" ,;"))


def has_framing(prompt: str) -> bool:
    return bool(_FRAMING.search(str(prompt or "")))


def _clauses(prompt: str) -> list:
    return [c.strip() for c in re.split(r"\s*[,;]\s*|(?<=[.!?])\s+", str(prompt or "")) if c.strip(" .")]


def _drop_words(clause: str, words) -> str:
    out = clause
    for w in words:
        out = re.sub(rf"\b{re.escape(w)}\b-?\s*", "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" -")


def guard_prompt(prompt: str, shows, allowed_text="") -> str:
    """the shot prompt of a shot that starts from a USER image. Every clause that introduces a clothing / jewellery / body / person / animal word (or a colour word) that neither the image's
    `shows` nor `allowed_text` (the user's own on-screen words) contains is dropped; a clause that only carries a loose colour word keeps its other words; the `shows` subject is put in
    front when the prompt does not already start with it. Without `shows` the prompt is returned unchanged."""
    subj = subject_text(shows)
    if not subj:
        return str(prompt or "")
    allowed = tokens(shows) | tokens(allowed_text)
    kept = []
    for c in _clauses(prompt):
        fl = flagged(c, allowed)
        if not fl:
            kept.append(c)
        elif all(norm(w) in _COLOUR for w in fl):
            c2 = _drop_words(c, fl)
            if len(c2.split()) >= 2:
                kept.append(c2)
        # else: a new subject / clothing / animal clause: dropped, the shows text carries the subject
    low = subj.lower()
    kept = [c for c in kept if c.lower() != low]
    head = subj if not (kept and kept[0].lower().startswith(low)) else ""
    tail = ", ".join(kept)
    room = max(SHOT_MAX_WORDS - len(subj.split()), 10)
    tail = " ".join(tail.split()[:room]).rstrip(" ,;")
    return ", ".join(x for x in (head, tail) if x) if head or tail else subj


def palette_names(piece) -> list:
    """colour names of the user's typed palette: every name of US.NAMES whose rgb is in the palette that user_style.parse_direction(piece) reads (piece.user_style, else the typed
    direction in piece.request, which survives the design step), so 'gold' also gives 'golden'. [] when the user typed none."""
    if not isinstance(piece, dict):
        return []
    st = US.parse_direction(piece)
    rgbs = {tuple(c[:3]) for c in (st.get("palette") or []) if isinstance(c, (list, tuple)) and len(c) >= 3}
    return sorted(n for n, v in US.NAMES.items() if tuple(v) in rgbs)


def shows_of(piece, sid) -> str:
    """the media-read `shows` of the user image a shot starts from (piece.request.shows[k-1] for shot sK; `request` is kept by the design step, tools/fusion/design_step.LOCKED_KEYS);
    '' when that shot has no user image."""
    if not isinstance(piece, dict):
        return ""
    sh = (piece.get("request") or {}).get("shows") if isinstance(piece.get("request"), dict) else None
    if not isinstance(sh, list):
        sh = piece.get("shows")
    m = re.fullmatch(r"s(\d+)", str(sid or ""))
    if not isinstance(sh, list) or not m:
        return ""
    k = int(m.group(1)) - 1
    return str(sh[k] or "").strip() if 0 <= k < len(sh) else ""


def scrub_prompt(prompt: str, piece, sid) -> str:
    """the user's typed palette never reaches the video prompt of a shot that starts from the user's picture: its colour names are deleted from the prompt unless that picture's `shows`
    has them. Other shots and pieces without a typed palette are returned unchanged."""
    shows = shows_of(piece, sid)
    if not shows:
        return prompt
    bad = [n for n in palette_names(piece) if norm(n) not in tokens(shows)]
    if not bad:
        return prompt
    parts = [_drop_words(x, bad) for x in re.split(r"\s*,\s*", str(prompt or ""))]
    return ", ".join(x for x in parts if x)
