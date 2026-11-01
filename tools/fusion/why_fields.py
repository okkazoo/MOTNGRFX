"""why_fields: the model's REASON per decision (par round r6, 2026-10-29) + the thinking-text split. Pure functions, no model call, no engine.

The user wants to see WHY a model chose a route, not only what it chose. Every design reply (brief -> piece JSON of the par harness, the app's design reply, the app's
concept reply) may carry one short `why` per decision, specific to THIS piece (its music / footage / idea), never a restated rule:

    idea_why                      the concept: why this idea for this music / footage
    system.why                    why this one rule organises the type
    shots[i].why                  why this location / camera move / continuation (concept: shot_plan[i].why)
    type.words[i].why             why this font role, module, form in / out, timing (hit), owner (render / material)
    compose.default.why           why this chain (compose.shots.<id>.why for a per-shot chain)

SALVAGE RULES: a missing / empty / unreadable why is NEVER an error (it is a count in `score`), it never blocks a piece; the engine and the salvage ignore the fields
(`strip_why` takes them out before the reply is compiled). A why that COPIES the guide (>= COPY_RATIO of its 4-grams occur verbatim in the guide / reference docs) is flagged
`copied`: the first 24 GB Gemma piece wrote the guide's own "behind" technique sentence as its concept. `reasons` = specific / decisions, where specific = present, at least
MIN_WORDS words, not copied from the guide, not a repeat of another decision's why (the same kind of decision with >= TEMPLATE_RATIO of its 3-grams already
used by an earlier one counts as a repeat: "the hit lands on the Xs downbeat so the word forms ... the exit is fast on the Ys" written for every word is a template, not a reason).

split_think(text) -> (answer, reasoning): thinking models either return the reasoning in a separate field (a hosted API's `reasoning`) or leave a <think>...</think> block (llama.cpp
chat templates with thinking on) in the content; the parser must never see it. The answer is the text with every think block removed (an unclosed or closing-only tag included);
the reasoning is saved next to the reply.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_RATIO = 0.5       # share of a why's 3-grams already used by earlier whys of the same kind above which it counts as the same sentence with other numbers (templated)
COPY_RATIO = 0.6            # share of a why's 4-grams found verbatim in the guide / reference docs above which it counts as copied
NGRAM = 4
MIN_WORDS = 5               # fewer words than this is not a reason ("fits the mood")
GUIDE_FILES = ["16_app/DESIGN_GUIDE_open_model.md", "16_app/MODULES_MENU.md"]
GUIDE_GLOBS = ["16_app/PIECE_REFERENCE_*.md"]
WHY_KEYS = ("why", "idea_why")

# the paragraph the app's design prompt (qwen_piece.schema_text) adds after the schema
APP_RULES = (
    "REASONS: next to each decision write a short `why` (one or two sentences, at most ~35 words), specific to THIS piece: name the music moment (a chord change, a dropout, the tempo), what is in the "
    "footage or the word it serves. Say what the choice does for this piece, not what the rule says; never paste or paraphrase a sentence of this guide. Two decisions never share one reason. "
    "Fields: `idea_why` (top level), `system.why`, `type.words[].why` (font role, module, form in/out, why the hit sits on that music moment, why that owner / material), `compose.default.why` "
    "(and `compose.shots.<id>.why`). A missing `why` never fails the reply, but the reasons are read.")

# ------------------------------------------------------------------------------------------------- thinking split
# Qwen3.x: <think>...</think> (the template may pre-fill the opening tag, so only </think> is in the reply); Gemma 4: `<|channel>thought ... <channel|>` (format from its chat template,
# not yet seen in a llama.cpp log here: unverified, the pattern is harmless when absent).
_THINK = re.compile(r"<(think|thinking|thought|reasoning)>(.*?)</\1>", re.S | re.I)
_GEMMA = re.compile(r"<\|channel\>\s*(?:thought)?\s*(.*?)<channel\|>", re.S)


def split_think(text) -> tuple[str, str]:
    """-> (answer without any thinking block, the thinking text joined by blank lines; '' when none)."""
    t = text if isinstance(text, str) else ""
    parts: list[str] = []

    def grab(m):
        parts.append((m.group(2) if m.re is _THINK else m.group(1)).strip())
        return ""
    t = _THINK.sub(grab, t)
    t = _GEMMA.sub(grab, t)
    m = re.search(r"</(think|thinking|thought|reasoning)>", t, re.I)           # closing tag only: the opening tag was pre-filled by the template
    if m:
        parts.append(t[:m.start()].strip())
        t = t[m.end():]
    m = re.search(r"<(think|thinking|thought|reasoning)>", t, re.I)             # opened and never closed (cut off by max_tokens): the rest is thinking, there is no answer
    if m:
        parts.append(t[m.end():].strip())
        t = t[:m.start()]
    return t.strip(), "\n\n".join(p for p in parts if p)


# ------------------------------------------------------------------------------------------------- the fields
def strip_why(obj):
    """a deep copy of the reply without any `why` / `idea_why` key (what the compilers see)."""
    if isinstance(obj, dict):
        return {k: strip_why(v) for k, v in obj.items() if k not in WHY_KEYS}
    if isinstance(obj, list):
        return [strip_why(v) for v in obj]
    return obj


def _text(v) -> str:
    """a why as one string (a string, or an object / list of strings joined: models sometimes write {"font": "...", "timing": "..."})."""
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip()
    if isinstance(v, dict):
        return _text(" ".join(f"{k}: {_text(x)}" for k, x in v.items() if _text(x)))
    if isinstance(v, (list, tuple)):
        return _text(" ".join(_text(x) for x in v))
    return ""


def decisions(obj, words: list[str] | None = None) -> list[dict]:
    """every decision slot of a reply -> [{key, kind, why}] (why '' = the model wrote none). Slots: idea, system, one per shot (`shots`, or the concept's `shot_plan`), one per
    word (`type.words`), one per compose chain (default + per shot). Does not raise on a malformed reply."""
    out: list[dict] = []
    if not isinstance(obj, dict):
        return out

    def add(key, kind, v):
        out.append(dict(key=key, kind=kind, why=_text(v)))
    if "idea" in obj or "idea_why" in obj:
        add("idea", "idea", obj.get("idea_why"))
    sysb = obj.get("system")
    if isinstance(sysb, dict) or "system" in obj:
        add("system", "system", sysb.get("why") if isinstance(sysb, dict) else None)
    for fld in ("shots", "shot_plan"):
        lst = obj.get(fld)
        if isinstance(lst, list):
            for i, s in enumerate(x for x in lst if isinstance(x, dict)):
                add(f"{fld}[{i}]", "shot", s.get("why"))
            break
    t = obj.get("type") if isinstance(obj.get("type"), dict) else {}
    ow = obj.get("owners")                      # the concept reply: who draws each line (render / ltx_at / material) and why
    for i, o in enumerate(ow if isinstance(ow, list) else []):
        if isinstance(o, dict):
            add(f"owners[{i}]", "owner", o.get("why"))
    ws = t.get("words") if isinstance(t.get("words"), list) else (obj.get("words") if isinstance(obj.get("words"), list) else [])
    for i, w in enumerate(ws):
        if isinstance(w, dict):
            add(f"word[{i}] {str(w.get('text', ''))[:24]}".strip(), "word", w.get("why"))
    c = obj.get("compose") if isinstance(obj.get("compose"), dict) else (t.get("compose") if isinstance(t.get("compose"), dict) else {})
    if isinstance(c.get("default"), dict):
        add("compose.default", "chain", c["default"].get("why"))
    if isinstance(c.get("shots"), dict):
        for sid, v in c["shots"].items():
            if isinstance(v, dict):
                add(f"compose.shots.{sid}", "chain", v.get("why"))
    return out


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _grams(toks: list[str], n: int = NGRAM) -> set:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


_CORPUS: dict = {}


def guide_grams(root: Path = ROOT, extra: list[str] | None = None) -> set:
    """the 4-grams of the guide + reference docs (cached per root; `extra` texts are added uncached)."""
    key = str(root)
    if key not in _CORPUS:
        files = [root / f for f in GUIDE_FILES]
        for g in GUIDE_GLOBS:
            files += sorted(root.glob(g))
        gs: set = set()
        for f in files:
            if f.is_file():
                gs |= _grams(_tokens(f.read_text(encoding="utf-8", errors="replace")))
        _CORPUS[key] = gs
    gs = _CORPUS[key]
    for e in extra or []:
        gs = gs | _grams(_tokens(e))
    return gs


def overlap(why: str, grams: set) -> float:
    """share of the why's 4-grams that occur verbatim in the guide corpus (0 for a why shorter than 4 words)."""
    g = _grams(_tokens(why))
    return round(sum(1 for x in g if x in grams) / len(g), 3) if g else 0.0


def score(obj, grams: set | None = None, anchors: set | None = None, words: list[str] | None = None) -> dict:
    """-> dict(decisions, present, missing, copied, repeated, specific, reasons, anchored, by_kind, flagged, detail).
    reasons = specific / decisions (the 'reasons' column of the rubric / craft tables). anchored = share of the PRESENT whys that name something of the piece: a number, one of
    `anchors` (lower-case tokens of the words, the shot prompts, the music: the caller builds them) - a second view on 'specific to this piece', not part of `reasons`."""
    grams = guide_grams() if grams is None else grams
    ds = decisions(obj, words)
    seen: set = set()
    used3: dict = {}                    # kind -> 3-grams of the earlier whys of that kind
    detail, by_kind = [], {}
    n_present = n_copied = n_rep = n_spec = n_anch = 0
    for d in ds:
        w = d["why"]
        toks = _tokens(w)
        norm = " ".join(toks)
        present = len(toks) > 0
        ov = overlap(w, grams) if present else 0.0
        copied = present and len(toks) >= NGRAM + 2 and ov > COPY_RATIO
        g3 = _grams(toks, 3)
        tmpl = bool(present and len(toks) >= 8 and g3 and len(g3 & used3.get(d["kind"], set())) / len(g3) >= TEMPLATE_RATIO)
        repeated = present and (norm in seen or tmpl)
        if present:
            seen.add(norm)
            used3.setdefault(d["kind"], set()).update(g3)
        spec = present and len(toks) >= MIN_WORDS and not copied and not repeated
        anch = present and (any(c.isdigit() for c in w) or bool(anchors and any(t in anchors for t in toks)))
        n_present += present
        n_copied += copied
        n_rep += repeated
        n_spec += spec
        n_anch += anch
        k = by_kind.setdefault(d["kind"], dict(n=0, specific=0))
        k["n"] += 1
        k["specific"] += int(spec)
        detail.append(dict(key=d["key"], kind=d["kind"], present=present, words=len(toks), overlap=ov, copied=copied, repeated=repeated, templated=tmpl, specific=spec, anchored=anch))
    n = len(ds)
    return dict(decisions=n, present=n_present, missing=n - n_present, copied=n_copied, repeated=n_rep, specific=n_spec, reasons=round(n_spec / n, 3) if n else None,
                anchored=round(n_anch / n_present, 3) if n_present else None, by_kind=by_kind,
                flagged=[f"{d['key']}: copies the guide ({d['overlap']:.0%} of its 4-grams)" for d in detail if d["copied"]] + [f"{d['key']}: same why as an earlier decision (or the same sentence with other numbers)" for d in detail if d["repeated"]],
                detail=detail)


def anchors_of(*texts: str) -> set:
    """lower-case tokens of 4+ letters from the piece's own words / prompts / music text (for `score(anchors=)`)."""
    return {t for s in texts for t in _tokens(s) if len(t) >= 4 and not t.isdigit()}


def collect(obj) -> dict:
    """{decision key: why} of the non-empty whys (what the archive / the WHY_*.md show)."""
    return {d["key"]: d["why"] for d in decisions(obj) if d["why"]}
