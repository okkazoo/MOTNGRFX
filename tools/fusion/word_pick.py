"""word_pick.py: which sung words go on a shot (user 2026-11-01: "in the case of lyrics taken from the song we have a better way of choosing which words to use in the shot and be able to
change them easily"). Pure functions, no repo imports (story.py calls them; the page gets the ranked alternatives as `word_alts` so it can cycle them without a model call).

    prepare(tokens, times, ends=None, lines=None, beats=None) -> P      per-song tables: normalised words, repeat counts, sung durations, line ends, beat distances
    word_scores(P)                  -> [score] per token                  (word_parts(P, i) shows the terms)
    runs_in_span(P, a, b, ...)      -> [(score, [token index, ...])] best first, every run sung inside the span (a, b) (source seconds)
    pick(P, a, b, lo, seed)         -> [token index, ...] the best run, [] when nothing worth showing is sung in the span
    alternatives(P, a, b, seed, k)  -> [{t: [token index, ...], s: score}] k DIFFERENT ideas, best first (what the page's cycle button steps through)

SCORING RULE (one number per word, then per run of 1-3 adjacent words of ONE sung line):
  word  = base  (+2.0 content word, +0.8 light verb / filler like "get" "know" "just", +0.2 vocable like "oh" "yeah", -2.5 function word)
        + hook  (+1.2 * log2(times the word is sung in the whole song), capped at +2.5, content words only: the chorus / hook words)
        + held  (+1.6 for a note held >= 1.2 s, linear from 0.3 s; only when the transcription has real per-word times)
        + beat  (+1.0 when the word starts on a beat of the analysed grid, linear to 0 at min(0.18 s, 0.3 beat) away)
        + end   (+0.9 the last word of its sung line; +0.4 the first word after a pause >= 0.6 s; +0.6 the first word of a sung line)
        + shape (a capitalised word mid-line +0.8; letters: 3-10 fine, +0.2 for 4-8; 2 letters -1.2; 1 letter -3; over 12 letters -0.8 and -0.4 per extra; a fragment like 'cause or a vowel-less stub -1.0..-1.5)
  run   = mean(word) + 0.6 for a natural two-word phrase / +0.3 for three (no function word at either end)
        + hook for a REPEATED phrase (+1.0 * log2(times the same words are sung), capped at +2.0)
  A run never starts or ends on a function word, never crosses a sung line or a pause > 0.9 s, is <= 3 words / <= max_chars letters, and never cuts through a phrase that is sung twice or more
  ("golden hour" is never split into "golden" or "hour" when the pair repeats). Ties are broken by a small deterministic jitter of the seed (same seed = same pick, always)."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

MAX_RUN_WORDS = 3
MAX_RUN_CHARS = 18
MAX_GAP_S = 0.9              # two words further apart than this are not one phrase
PAUSE_S = 0.6                # a pause that makes the next word a phrase onset
HELD_FULL_S = 1.2            # a note this long (or more) gets the full held bonus
LINE_START = 0.6             # bonus for the first word of a sung line (user 2026-11-02: a line opens on its key noun, which should be the word on screen)
PROPER = 0.8                 # bonus for a capitalised word in the middle of a line (a name / place)
MIN_SCORE = 0.5              # a run under this is not worth a shot (nothing good is sung there)
JITTER = 0.05                # seed-dependent tie-break amplitude (never reorders runs further apart than this)

STOP = frozenset("""a an the and or but nor so yet of to in on at for with from by as into onto upon over under about than then if because that this these those
i you he she we they me him her us them my your his our their its it is are was were be been am isn't aren't wasn't weren't will would can could should shall may might do does did have has had
not no don't doesn't didn't can't won't cannot i'm i'll i've i'd you're you'll you've you'd it's that's there's we're we'll we've they're they'll he's she's what when while where which who whom
how why all any some each every both more most very too also just still than then there here up down out off again once only own same such""".split())
# not function words, but nothing to put on a screen alone
LIGHT = frozenset("""get got gets getting go goes going gone gonna wanna gotta know knew like let lets make made say said says see come came take took put way thing things really
cause 'cause till 'til ain't ya yo man baby""".split())
VOCABLE = frozenset("oh ooh oooh ah aah uh um mm mmm hmm yeah yea yeh ay ayy ey hey la na da doo woah whoa ha huh ho hoo".split())
VOWELS = set("aeiouyAEIOUY")


def norm(w) -> str:
    """a token -> its comparison form: lower case, edge punctuation off, inner apostrophes kept."""
    return re.sub(r"^[^\w']+|[^\w']+$", "", str(w or "").lower().replace("’", "'")).strip("-")


def letters(w) -> int:
    return len(re.findall(r"[^\W_]", str(w or "")))


def _fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def is_weak(w) -> bool:
    """a function word: a run never starts or ends on it."""
    n = norm(w)
    return (not n) or n in STOP


def _kind(n: str) -> str:
    return "stop" if (not n or n in STOP) else "vocable" if n in VOCABLE else "light" if n in LIGHT else "content"


def prepare(tokens, times, ends=None, lines=None, beats=None) -> dict:
    """the per-song tables. tokens: the lyric words; times: sung start of each (source seconds; None = unknown); ends: sung end of each when the transcription has it (else the start of the
    next word, capped); lines: [(first token, last token)] of each sung line (inclusive; default: one line); beats: the beat grid in source seconds."""
    n = len(tokens)
    nw = [norm(t) for t in tokens]
    ts = [_fnum(times[i]) if i < len(times or []) else None for i in range(n)]
    timed = n > 0 and all(t is not None for t in ts)
    es = []
    for i in range(n):
        e = _fnum(ends[i]) if ends is not None and i < len(ends) else None
        es.append(e)
    real_ends = timed and ends is not None and all(e is not None for e in es)
    dur = []
    for i in range(n):
        if not timed:
            dur.append(None)
        elif real_ends:
            dur.append(max(0.0, es[i] - ts[i]))
        else:
            nxt = ts[i + 1] if i + 1 < n else None
            dur.append(min(max(0.0, (nxt - ts[i])), 1.2) if nxt is not None else None)           # no end time: the gap to the next word (capped; the last word of a song has none)
    ls = [(int(a), int(b)) for a, b in (lines or [])] or ([(0, n - 1)] if n else [])
    line_of = [0] * n
    for k, (a, b) in enumerate(ls):
        for i in range(max(0, a), min(n, b + 1)):
            line_of[i] = k
    last_of_line = {min(b, n - 1) for _, b in ls}
    first_of_line = {max(a, 0) for a, _ in ls}
    gap = [None] * n                                              # pause before the word (within its line)
    for i in range(1, n):
        if timed and line_of[i] == line_of[i - 1]:
            prev_end = es[i - 1] if real_ends else ts[i - 1]
            gap[i] = max(0.0, ts[i] - prev_end)
    bl = sorted(float(b) for b in (beats or []) if _fnum(b) is not None)
    per = (sum(b - a for a, b in zip(bl, bl[1:])) / (len(bl) - 1)) if len(bl) > 1 else None
    bd = []
    for i in range(n):
        if not bl or not timed:
            bd.append(None)
            continue
        bd.append(min(abs(ts[i] - b) for b in bl))                                       # n x beats: songs are short (a few hundred words, a few hundred beats)
    cnt = Counter(w for w, k in ((w, _kind(w)) for w in nw) if k != "stop")
    big = Counter(tuple(nw[i:i + 2]) for i in range(n - 1) if line_of[i] == line_of[i + 1] and nw[i] and nw[i + 1])
    return dict(tokens=list(tokens), n=n, nw=nw, times=ts, ends=es, dur=dur, timed=timed, real_ends=real_ends, line_of=line_of, last_of_line=last_of_line, first_of_line=first_of_line, gap=gap, beat_d=bd, beat_period=per,
                count=cnt, bigram=big)


def word_parts(P: dict, i: int) -> dict:
    """the terms of word i's score (and the total under 'score')."""
    w = P["nw"][i]
    kind = _kind(w)
    base = {"content": 2.0, "light": 0.8, "vocable": 0.2, "stop": -2.5}[kind]
    c = P["count"].get(w, 0)
    hook = min(2.5, 1.2 * math.log2(c)) if (kind in ("content", "light") and c >= 2) else 0.0
    d = P["dur"][i] if i < len(P["dur"]) else None
    held = 1.6 * min(1.0, max(0.0, (d - 0.3) / (HELD_FULL_S - 0.3))) if (d is not None and kind != "stop") else 0.0
    bd = P["beat_d"][i]
    tol = min(0.18, 0.3 * P["beat_period"]) if P["beat_period"] else 0.18
    beat = max(0.0, 1.0 - bd / tol) if (bd is not None and kind != "stop") else 0.0
    end = 0.0
    if kind != "stop":
        if i in P["last_of_line"]:
            end += 0.9
        g = P["gap"][i]
        if g is not None and g >= PAUSE_S:
            end += 0.4
        if i in P.get("first_of_line", ()):
            end += LINE_START                                                                 # the first word of a sung line (after the breath): the phrase onset
    L = letters(P["tokens"][i])
    shape = 0.0
    if L <= 1:
        shape = -3.0
    elif L == 2:
        shape = -1.2
    elif L > 12:
        shape = -0.8 - 0.4 * (L - 12)
    elif 4 <= L <= 8:
        shape = 0.2
    raw = str(P["tokens"][i])
    core = re.sub(r"^[^\w']+", "", raw)
    if kind == "content" and i not in P.get("first_of_line", ()) and core[:1].isupper() and any(c.islower() for c in core[1:]):
        shape += PROPER                                                                       # a capital in the middle of a line (Bronx, Marble): a name, a concrete noun worth a screen
    if raw.strip().startswith(("'", "’")) and L <= 6:
        shape -= 1.5                                                                       # 'cause, 'em, 'til: a fragment
    elif L <= 3 and w and not (set(w) & VOWELS) and kind == "content":
        shape -= 1.0                                                                       # a vowel-less stub
    parts = dict(base=base, hook=hook, held=held, beat=beat, end=end, shape=shape)
    parts["score"] = sum(parts.values())
    return parts


def word_scores(P: dict) -> list[float]:
    return [word_parts(P, i)["score"] for i in range(P["n"])]


def _cuts_repeated(P: dict, i: int, j: int) -> bool:
    """does the run i..j (inclusive) start or end inside a two-word phrase that is sung at least twice (a 'golden hour' split into 'golden')?"""
    nw, big = P["nw"], P["bigram"]
    for a in (i - 1, j):                                   # the pair (a, a+1) with exactly one word inside the run
        if a < 0 or a + 1 >= P["n"] or P["line_of"][a] != P["line_of"][a + 1]:
            continue
        if (a < i <= a + 1 <= j) or (i <= a <= j < a + 1):
            pair = (nw[a], nw[a + 1])
            if big.get(pair, 0) >= 2 and not is_weak(pair[0]) and not is_weak(pair[1]):
                return True
    return False


def _jit(seed, i: int, j: int) -> float:
    h = hashlib.md5(f"{int(seed)}:{i}:{j}".encode()).hexdigest()
    return JITTER * (int(h[:8], 16) / 0xFFFFFFFF)


def runs_in_span(P: dict, a: float, b: float, lo: int = 0, seed: int = 0, max_words: int = MAX_RUN_WORDS, max_chars: int = MAX_RUN_CHARS, scores: list[float] | None = None) -> list[tuple[float, list[int]]]:
    """every acceptable run whose FIRST word is sung inside [a, b) and starts at token >= lo -> [(score, [indices])] best first. Without sung times (P["timed"] False) nothing qualifies by time:
    the caller must not use this for text without times."""
    if not P["timed"]:
        return []
    ws = scores if scores is not None else word_scores(P)
    n, nw = P["n"], P["nw"]
    out = []
    for i in range(max(0, lo), n):
        t = P["times"][i]
        if t < a - 1e-6:
            continue
        if t >= b - 1e-6:
            break
        if is_weak(P["tokens"][i]) or not nw[i]:
            continue
        for k in range(1, max_words + 1):
            j = i + k - 1
            if j >= n or P["line_of"][j] != P["line_of"][i]:
                break
            if k > 1 and (P["gap"][j] is None or P["gap"][j] > MAX_GAP_S):
                break
            if not nw[j]:
                break
            if is_weak(P["tokens"][j]):
                continue                                                                    # a function word may sit inside a run ("rise and shine"), never end it
            if len(" ".join(str(P["tokens"][x]) for x in range(i, j + 1))) > max_chars and k > 1:
                break
            if _cuts_repeated(P, i, j):
                continue
            idx = list(range(i, j + 1))
            s = sum(ws[x] for x in idx) / k
            if k == 2:
                s += 0.6
            elif k == 3:
                s += 0.3
            if k > 1:
                seq = tuple(nw[x] for x in idx)
                c = sum(1 for x in range(0, n - k + 1) if tuple(nw[x:x + k]) == seq and P["line_of"][x] == P["line_of"][x + k - 1])
                if c >= 2:
                    s += min(2.0, math.log2(c))
            out.append((s + _jit(seed, i, j), idx))
    out.sort(key=lambda r: (-r[0], r[1][0], len(r[1])))
    return out


def pick(P: dict, a: float, b: float, lo: int = 0, seed: int = 0, **kw) -> list[int]:
    """the best run sung inside [a, b), [] when nothing scores MIN_SCORE."""
    r = runs_in_span(P, a, b, lo, seed, **kw)
    return r[0][1] if r and r[0][0] >= MIN_SCORE else []


def alternatives(P: dict, a: float, b: float, seed: int = 0, k: int = 6, **kw) -> list[dict]:
    """k different ideas for the span, best first: a run is skipped when it contains or lies inside one already listed (so the cycle does not walk golden / golden hour / hour)."""
    out, chosen = [], []
    for s, idx in runs_in_span(P, a, b, 0, seed, **kw):
        if s < MIN_SCORE:
            break
        st = set(idx)
        if any(st <= c or c <= st for c in chosen):
            continue
        chosen.append(st)
        out.append(dict(t=idx, s=round(s, 2)))
        if len(out) >= k:
            break
    return out


def model_pick_ok(P: dict, run: list[int], a: float, b: float, slack: float = 0.3) -> bool:
    """guard for a pick the model made: its first word is sung inside the span (slack s of grace before it), and it is not only function words / a fragment."""
    if not run or not P["timed"] or any(not 0 <= i < P["n"] for i in run):
        return False
    t = P["times"][run[0]]
    if not (a - slack <= t < b):
        return False
    return any(not is_weak(P["tokens"][i]) and P["nw"][i] and letters(P["tokens"][i]) >= 2 for i in run)
