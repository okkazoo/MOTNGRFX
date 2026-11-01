"""text_guard: the video / image model never invents its own text (user 2026-10-29: "LTX or whatever renderer should never invent its own text - it should be based off a canny or depth
of the graphic with the words told to the model so it knows what it spells"). Pure functions on prompts and pieces, no ComfyUI, no network, no Claude.

Two rules, enforced in code (the guide only tells the models):

  1. A shot whose prompt names a word MUST carry a type control pass of that word's graphic (guide > 0; the pass is picked per face, owner.control_pass_for) and the prompt must
     spell exactly the word(s) that control pass shows (`the word "DRIFT" ...`, same case as the graphic, owner.clause). `fix_plan` rewrites the clause from the control pass's words.
  2. Every shot WITHOUT a type control pass (guide 0: the clean / plain / Alpha-Gen clips of the layered route, every all-flat shot, still prompts, any other renderer)
     gets every text / letters / word / sign / typography mention stripped from its prompt, and the negative prompt names them (NEG_NO_TEXT). Stills the same (invented signs in the still
     carry into the video).

    strip_text(prompt)            -> (prompt without text mentions, [removed phrases])
    spelled_words(prompt)         -> the words a prompt asks the model to spell (the quoted `the word "X"` clauses)
    plan_problems(piece)          -> [problem text]  the design-check validator: rule 1 or rule 2 broken
    fix_plan(piece)               -> [fix text]      mends the piece in place (prompts, stills, shot negatives), every change logged
    NEG_NO_TEXT / NEG_TEXT_PASS   negative prompts (the clean chain of the layered graph / the text-pass chain)
"""
from __future__ import annotations

import re

from tools.fusion import owner as OW

NEG_NO_TEXT = "text, letters, words, captions, subtitles, signage, signs, typography, lettering, logos, watermark"
NEG_TEXT_PASS = "extra text, other words, misspelled letters, garbled letters, captions, signage, watermark"
STILL_NEG_TOKENS = ("text", "letters", "words", "signage", "typography")
TEXT_RE = re.compile(r"\b(?:text|texts|letters?|lettering|letterforms?|words?|captions?|titles?|subtitles?|typography|typeface|fonts?|signs?|signage|signpost|writing|written|inscriptions?|"
                     r"logos?|billboards?|posters?|banners?|labels?|graffiti|calligraphy|handwriting|neon signs?|storefront signs?|street signs?)\b", re.I)
QUOTED_RE = re.compile(r'the words? "([^"]*)"', re.I)


def strip_text(prompt: str) -> tuple[str, list[str]]:
    """drop every comma-separated phrase of the prompt that mentions text / letters / words / a sign (a phrase = one clause of the comma list). The owner clause is not a phrase to keep:
    it belongs to a shot WITH a control pass. -> (clean prompt, removed phrases)."""
    parts = [p.strip() for p in re.split(r",\s*", str(prompt or "")) if p.strip()]
    keep, removed = [], []
    for p in parts:
        (removed if TEXT_RE.search(p) else keep).append(p)
    return ", ".join(keep), removed


def spelled_words(prompt: str) -> list[str]:
    """the words / lockups the prompt asks the model to spell: the quoted parts of `the word "X"` / `the words "X Y"` clauses, in order."""
    return [m.group(1) for m in QUOTED_RE.finditer(str(prompt or ""))]


def _shot_owned(piece: dict, wins, s: dict):
    sp = OW.shot_span_s(piece, s["id"]) if piece.get("layout") else None
    if wins is None or sp is None:
        return [], None
    return OW.shot_words(wins, sp[0], sp[1]), sp


def _guide(s: dict) -> float:
    return float((s.get("control") or {}).get("guide", s.get("guide", 0.0)) or 0.0)


def _layout(piece: dict) -> dict:
    if piece.get("layout"):
        return piece
    try:
        from tools.fusion import plan_rules as PR
        return dict(piece, layout=PR.layout(piece))
    except Exception:       # noqa: BLE001
        return piece


def plan_problems(piece: dict) -> list[str]:
    """the design-check validator of the two rules above. [] = the piece is clean. A piece without owner fields (an old plan) or a draft has no LTX words: every shot is judged as a
    shot without a control pass."""
    piece = _layout(piece)
    probs: list[str] = []
    wins = OW.windows(piece)
    for s in piece.get("shots") or []:
        if wins is None and _guide(s) > 0:
            continue                                                       # an old plan (no owner fields) / a draft: the control pass shows the words, no per-word judgement
        owned, _ = _shot_owned(piece, wins, s)
        prompt = str(s.get("prompt") or "")
        spelled = spelled_words(prompt)
        if owned:
            if _guide(s) <= 0:
                probs.append(f"{s['id']}: an LTX-owned word {[w['text'] for w in owned]} but the control guide is {_guide(s):g}: the video model would invent its letters")
            want = [re.sub(r"\s+", " ", w["text"].strip().upper()) for w in owned]
            got = [re.sub(r"\s+", " ", x.strip()) for x in spelled]
            if sorted(got) != sorted(want):
                probs.append(f"{s['id']}: the prompt spells {got or 'nothing'} but the control pass shows {want}")
            rest, removed = strip_text(OW.strip_clause(prompt))
            if removed:
                probs.append(f"{s['id']}: the prompt names text outside the control clause: {removed[:2]}")
        else:
            if spelled or TEXT_RE.search(prompt):
                probs.append(f"{s['id']}: no type control pass (guide {_guide(s):g}) but the prompt names text / letters / words: {strip_text(prompt)[1][:2] or spelled[:2]}")
    for st in piece.get("stills") or []:
        if TEXT_RE.search(str(st.get("prompt") or "")):
            probs.append(f"still {st.get('id')}: the prompt names text / letters / a sign: {strip_text(str(st.get('prompt')))[1][:2]}")
        neg = str(st.get("negative") or "").lower()
        if not all(t in neg for t in STILL_NEG_TOKENS):
            probs.append(f"still {st.get('id')}: the negative prompt does not rule out text ({', '.join(t for t in STILL_NEG_TOKENS if t not in neg)} missing)")
    return probs


def fix_plan(piece: dict) -> list[str]:
    """mend the piece in place: prompts of shots without a control pass lose their text mentions and carry `negative` = NEG_NO_TEXT; shots with LTX words get their clause rebuilt from the
    control pass's words (idempotent); still prompts lose text mentions and the negative gets the missing tokens. -> one fix text per change (callers log them)."""
    fixes: list[str] = OW.fix_speckle_forms(piece)             # 2026-11-01 (R18 / R19): no letter is a speckle / dot cloud, so no word keeps the `particles` module or io style (owner.fix_speckle_forms)
    fixes += OW.fix_illegible_modules(piece)          # 2026-11-01 (render R17): a cube-lattice word that merges into a block is moved to a readable module (owner.fix_illegible_modules)
    probe = _layout(piece)
    wins = OW.windows(probe)
    for s in piece.get("shots") or []:
        if wins is None and _guide(s) > 0:
            continue
        owned, sp = _shot_owned(probe, wins, s)
        prompt = str(s.get("prompt") or "")
        if owned:
            clean, removed = strip_text(OW.strip_clause(prompt))               # text named outside the control clause goes; the clause is rebuilt from the control pass's words (idempotent)
            new = OW.prompt_with_clause(clean, wins, sp[0], sp[1])
            if removed:
                fixes.append(f"{s['id']}: text named outside the control clause removed from the shot prompt: {removed[:2]}")
            if new != prompt:
                if not removed:
                    fixes.append(f"{s['id']}: the prompt now spells exactly the control pass's word(s) {[w['text'] for w in owned]}")
                s["prompt"] = new
            s.pop("negative", None)
            continue
        clean, removed = strip_text(prompt)
        if removed:
            s["prompt"] = clean
            fixes.append(f"{s['id']}: no type control pass; text / letters mentions removed from the prompt: {removed[:3]}")
        if s.get("negative") != NEG_NO_TEXT:
            s["negative"] = NEG_NO_TEXT
    for st in piece.get("stills") or []:
        clean, removed = strip_text(str(st.get("prompt") or ""))
        if removed:
            st["prompt"] = clean
            fixes.append(f"still {st.get('id')}: text / sign mentions removed from the still prompt: {removed[:3]}")
        neg = [x.strip() for x in str(st.get("negative") or "").split(",") if x.strip()]
        add = [t for t in STILL_NEG_TOKENS if t not in " ".join(neg).lower()]
        if add:
            st["negative"] = ", ".join(neg + add)
            fixes.append(f"still {st.get('id')}: negative prompt gained {', '.join(add)}")
    if fixes:
        piece.setdefault("plan", {}).setdefault("text_guard", []).extend(fixes)
    return fixes
