"""story: the STORY step of the Add-media box (16_app/MEDIA_BOX.md "STORY step"). Pure functions; the nodes are in nodes_fusion/story.py, the side graph in tools/app/story_graph.py.

    story_prompt(req)        -> the prompt for Gemma (TEXT ONLY: the READ already described the pictures)
    parse(reply, req)        -> STORY JSON v1 (dict) with the code guarantees below; a rule story when the reply is unusable
    fallback(req)            -> the rule story alone (beats from media[].shows)

req = {"read": READ JSON v1 (the page's edited one; dict or its JSON text), "length": seconds, "seed": int, "format": "9:16",
       and optional (STORYBOARD v2): "use": ["I1","I4"] (the images the user wants, shot k = k-th used), "text": the original pasted text,
       "select": [{"i": token, "phrase": k, "shot": n}] | [token, ...] (the user's own word choice, honoured verbatim; v18 SHOT-FOCUSED WORDS: an entry with `shot` puts the token on THAT shot, the shot's line = its
       tokens in token order, no time-based placement; shots without an entry have no words), "fonts": [file, ...] | {"1": file} (per shot, liked fonts only)}.

STORY JSON v2 (v1 fields kept): {"v": 2, "title", "logline", "look", "music",
                "shots": [{"n", "image": "I1" | null, "start", "end", "beat", "line", "camera": <CAMERA_MOVES>, "font": liked font file | null}],
                "candidates": [{"image": "I4", "use": bool, "reason"}]  (one per dropped image),
                "capacity": {"shots": N, "why": "...", "images", "min_length", "pick_length", "per_shot_s", "clip_max_s", "feasible": {"8": bool, "12": bool, ...}},
                "text_select": [{"i": token index, "phrase": k, "shot": n}]  (only when text was pasted: token = text.split(); phrase k -> shot k's line; v18: `shot` = the shot that carries the token),
                "notes",
                "word_alts": {"<shot n>": [{"t": [token index, ...], "s": score}]}  (lyrics only: WORD PICK, tools/fusion/word_pick.py: the best-scoring words sung INSIDE the shot, best first; the page's cycle button steps through them)}
shots[].camera = the move that carries shot k into shot k+1 (the cut after it; the last shot is "hold").

Guarantees (code, not the model; the model only writes title / logline / look / music / beats / free lines):
  - SHOT COUNT = shots_for(length, n_pictures) (STORYBOARD v12): the planner's one lockup per shot (concept.lockup_count(length, "cut") = mograph.fusion_brief.shot_count: 6 s -> 2, 8 -> 2, 12 -> 3, 16 -> 4, 24 -> 6, max 6),
    but with >= 2 pictures never more shots than pictures (24 s + 5 pictures = 5 shots of 4.8 s). 0 pictures (text only) and 1 picture: the planner's count, unchanged. capacity.feasible[L] = every shot fits one clip (CLIP_MAX_S).
    The render honours it: concept.request_extras `story_shots` -> plan_rules.story_shot_count / cut_spans(exact=True).
  - image k -> shot k, in the user's order (IN image = shot 1, IN images = shots 2..); a shot without an image has image null; images beyond the shot count are listed in `notes` (with the Length that would use them).
  - the READ's on_screen lines are the shots' lines VERBATIM and in order (line k in shot k; more lines than shots -> the planner's own plan_rules.fit_lockups merge; fewer -> the later shots have no line).
    With NO on_screen line Gemma's own lines are cleaned by the concept's house style (UPPER CASE, concept._CLEAN, <= LOCKUP_WORDS words, <= MAX_LINE_CHARS letters, no concept.OVERUSED_WORDS, no repeat of the previous line).
  - the READ's concept is the logline and the READ's look is the look when the user gave them; the music string always names the bpm of the READ's music.
  - start / end come from plan_rules.cut_spans on the bar grid (4 beats) of the READ's bpm; without a bpm, even cuts. First start 0, last end = length, shots meet.
  - an unusable reply gives the rule story (beats from media[].shows) and says so in `notes`.

VIDEO MODE (MEDIA_BOX.md "VIDEO MODE STORYBOARD"; read.media has a video): the storyboard is built FROM the video, never invented.
  - length = the request's Length capped to the window / video (video_feasible: options <= the video, or the video itself when < 8 s), not clamped to the planner's 6 s minimum.
  - shots = the video's REAL cuts (READ `video_cuts`, re-fitted to [window start, window start + length] by video_pieces; v18: EVERY real cut is a shot, no 10-piece cap, no 1 s rule: only a piece under
    VIDEO_MIN_FRAMES (6) frames merges into its neighbour (own_video_job.clean_cuts min_piece) and a piece > CLIP_MAX_S is split evenly);
    no cuts in the READ = an even split. Each shot: image "V1", src_start / src_end (s in the video; the page shows the cut's first frame), start / end (s in the piece), camera "as filmed".
    shot.beat is "" (the cuts are not described); v18: no transition / camera choice per shot either (the design step picks the cuts at render time); shot.camera stays "as filmed".
  - hits = [{text, land, hold_until, sung?}] (s in the piece) = the render's own word slots: own_video_job.slots_from_analysis(READ.beat_grid, n_words, length) (imported, not copied) with the words from
    pasted/quoted text > the READ's lines > lyrics (the model picks phrases from the numbered lyrics) > the model's free words (own_video_v7.words_prompt text + parse_words). shot.line = the hit(s) whose `land`
    falls inside the shot's span. capacity gains video_duration, length, feasible_lengths, length_default.
  - text source: pasted text wins; with none and READ.lyrics the tokens are the lyrics (text_candidates {source: "lyrics", tokens: [{i, w, t, tp}]}: t = sung time in the source file, tp = in the piece).
    IMAGE mode with an audio file's lyrics: each chosen phrase goes on the shot whose span holds its sung time.
"""
from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

from mograph import fonts_liked as FL
from tools.fusion import user_style as US
from tools.fusion import concept as CO
from tools.fusion import cut_kinds as CK
from tools.fusion import music_read as MR
from tools.fusion import plan_rules as PR
from tools.fusion import word_pick as WP

FPS = PR.FPS
MAX_IMAGES = CO.BRIEF_MAX_IMAGES
TITLE_WORDS = 5
LOGLINE_CHARS, LOOK_CHARS, MUSIC_CHARS, BEAT_CHARS = 300, 300, 200, 320
BEAT_WORDS = "12-30"
DEFAULT_LENGTH = 12
# the camera moves the pipeline really has (cut_kinds): a 3D cut or a type-camera pan into the next shot, or a hold. depth_dolly is left out: design_options.DOLLY_ON is False (renders as a shard collage).
CAMERA_MOVES = ("hold", "depth_pivot", "orbit:push", "orbit:pullback", "orbit:fly") + tuple(CK.PANS)
CAMERA_ROTATION = ("orbit:push", "depth_pivot", "pan:left", "orbit:pullback", "orbit:fly", "pan:right")      # the default per shot when the model gives none (rotated by the seed)
TEXT_CAP = 400                                                                                             # words of the pasted text the page shows and the model sees
STOP = frozenset("a an the and or but of to in on at for with from by as is are was were be been it its this that these those i you he she we they me my your our their so if not no do does did "
                 "have has had will would can could just than then there here what when where who which how all any some into out up down over under again very more most "
                 "i'm i'll i've i'd you're you'll it's that's don't can't we're they're".split())
SYSTEM_PROMPT = ("You are a screenwriter for very short music-video pieces (kinetic typography over generated footage). You answer with exactly one JSON object and nothing else. "
                 "You never reword the user's on-screen lines.")

# PHRASES + SUNG TIMING (user 2026-11-01: a line that ends on a small word is half a sentence; words must sit on the beat they are sung). A phrase taken from the LYRICS is a sensible standalone line
# (never ends on a small function word, <= PHRASE_WORDS words / ~PHRASE_CHARS letters unless the whole line is complete and a little longer) and lands when its first word is sung.
PHRASE_WORDS, PHRASE_CHARS, PHRASE_SOFT_CHARS = 4, 18, 20
SNAP_S, MIN_HOLD_S, MIN_GAP_S, PUSH_MAX_S = 0.25, 0.7, 2.4, 1.2       # beat snap (forward only: never before it is sung), minimum hold, gap between two landings (the render's slot floor), longest push later
END_BAD = frozenset("a an the and or but nor so yet of to in on at for with from by as into onto upon over under about than then if because that this these those my your his our their its "
                    "is are was were be been am i'm i'll i've i'd you're you'll it's that's don't can't we're they're isn't aren't will would can could should shall may might do does did have has had "
                    "when while where which who what".split())
# ONE WORD PER SHOT (user 2026-11-02: "You can use them for every shot in the users video. Why do we have that rule?" / "If no lyrics are in a shot we don't need to do words there"): in VIDEO mode with timed
# lyrics and no hand-picked words, EVERY shot whose span holds sung lyrics gets the best-scoring word sung in it (no capacity cap; the old n_want slots / MIN_GAP_S 2.4 s landing floor / 5-of-15 no longer apply
# on this path); a shot with nothing sung in its span has no word. The ONE physical limit left is the type engine's minimum word cycle (hold >= spec.HOLD_FLOOR 1.2 s + form 0.35 + exit 0.4 + transition 0.4
# + the word's own form-in = ~3 s on the 24 fps pipeline clock = ~2.5 s real at K 1.25): two hits closer than MIN_WORD_CYCLE_S cannot both play, so the later shot's word is dropped (said in notes), never moved into another shot.
SHOT_LEAD_S = 0.6                    # a word lands at least this long after its shot starts (its form-in plays inside the shot), or half the shot when the shot is shorter
SHOT_ROOM_S = 1.0                    # a run first sung less than this before its shot ends loses to one with room (shots under 2x this keep every run)
SHOT_END_PAD_S = 0.1                 # ... and at least this long before the shot ends
MIN_WORD_CYCLE_S = 2.5               # least gap between two hits the engine can play (MEASURED on the box retime, 30 fps source: hit to hit 2.9-3.3 s on the pipeline clock = 2.3-2.7 s real)
PHRASE_RULE = ("Every phrase must read as a sensible line on its own: a complete short lyric line, or a cut at a natural break (the end of a noun phrase or clause, before small words like on / the / in / of / and / to / with), "
               f"at most {PHRASE_WORDS} words and about {PHRASE_CHARS} letters, and it must NEVER end on such a small word (write NEON RAIN, not NEON RAIN ON).")


def _is_fn(w: str) -> bool:
    """a small function word (article, preposition, conjunction, auxiliary ...): a phrase never ends on it."""
    return str(w or "").lower().strip(".,;:!?&-'\"") in END_BAD


def _cmp_word(w: str) -> str:
    return _clean_word(w).lower().strip(".,;:!?&-'\"")


def _phrase_span(raw_toks, max_words: int = PHRASE_WORDS, max_chars: int = PHRASE_CHARS, soft_chars: int | None = None) -> tuple[int, int]:
    """raw tokens of a line -> (a, b): the sensible standalone phrase is raw_toks[a:b] ((0, 0) = nothing sensible). A complete line of <= max_words words and <= soft_chars letters that does not end on a
    function word stays whole; otherwise the longest prefix (<= max_words words, <= max_chars letters) that ends on a content word, preferring a cut at a comma / full stop;
    a line that starts with function words is tried again without them."""
    soft = soft_chars or (max_chars + 2)
    ix = [i for i, t in enumerate(raw_toks) if _clean_word(t)]
    toks = [_clean_word(raw_toks[i]) for i in ix]
    brk = [bool(re.search(r"[,;:.!?]$", str(raw_toks[i]).strip())) for i in ix]
    m = len(toks)

    def pick(lo: int):
        n = m - lo
        chars = lambda k: len(" ".join(toks[lo:lo + k]))                          # noqa: E731
        ok = lambda k: not _is_fn(toks[lo + k - 1])                              # noqa: E731
        if n <= max_words and chars(n) <= soft and ok(n):
            return lo, m
        valid = [k for k in range(min(n, max_words), 0, -1) if ok(k) and (k == 1 or chars(k) <= max_chars)]       # longest first
        if not valid:
            return None
        at_break = [k for k in valid if k > 1 and lo + k < m and brk[lo + k - 1]]                                    # a comma / full stop ends the phrase
        return lo, lo + (at_break[0] if at_break else valid[0])
    lead = next((i for i, w in enumerate(toks) if not _is_fn(w)), m)                  # the first content word
    for lo in dict.fromkeys((0, lead)):
        if lo < m:
            r = pick(lo)
            if r:
                return ix[r[0]], ix[r[1] - 1] + 1
    return 0, 0


def trim_phrase(text, max_words: int = PHRASE_WORDS, max_chars: int = PHRASE_CHARS, soft_chars: int | None = None) -> str:
    """a lyric line -> the on-screen phrase (UPPER CASE): a complete short line, else a cut at a natural break; never ends on a function word. "" when the line is only function words."""
    raw = str(text or "").split()
    a, b = _phrase_span(raw, max_words, max_chars, soft_chars)
    return " ".join(w for w in (_clean_word(t) for t in raw[a:b]) if w).rstrip(".,;:")


# ----------------------------------------------------------------------------------------------------------- request
def _ws(v, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


def norm_req(req) -> dict:
    """the request -> {"read" (dict), "length" (int, clamped to the planner's range), "seed", "format", "bpm", "images", "video", "audio", "lines", "n"}."""
    if isinstance(req, str):
        try:
            req = json.loads(req)
        except ValueError:
            req = {}
    req = req if isinstance(req, dict) else {}
    read = req.get("read")
    if isinstance(read, str):
        try:
            read = json.loads(read)
        except ValueError:
            read = {}
    read = read if isinstance(read, dict) else {}
    try:
        length = int(round(float(req.get("length", DEFAULT_LENGTH))))
    except (TypeError, ValueError):
        length = DEFAULT_LENGTH
    length = max(PR.MIN_LENGTH, min(PR.MAX_LENGTH, length))
    try:
        seed = int(req.get("seed", 1))
    except (TypeError, ValueError):
        seed = 1
    media = [m for m in read.get("media") or [] if isinstance(m, dict)]
    kind = lambda k: [m for m in media if str(m.get("kind") or "").lower() == k]       # noqa: E731
    images = kind("image")
    for k, m in enumerate(images, 1):
        m["id"] = _ws(m.get("id"), 8) or f"I{k}"
    mu = read.get("music") if isinstance(read.get("music"), dict) else None
    bpm = None
    if mu:
        try:
            b = float(mu.get("bpm"))
            bpm = b if 20 <= b <= 300 else None
        except (TypeError, ValueError):
            bpm = None
    lines = []
    for x in read.get("on_screen") if isinstance(read.get("on_screen"), list) else []:
        t = re.sub(r"\s+", " ", US.words_of(str(x or ""))).strip()          # a directive clause ("palette: ...", "more legible text") is never an on-screen line
        if t and (not lines or lines[-1] != t):
            lines.append(t)
    lines = US.drop_direction_lines(lines, req.get("text"))        # a stale READ line that was copied out of the typed direction is not a word
    video = (kind("video") or [None])[0]
    text = US.words_of(str(req.get("text") or "")[:20000])          # the typed box text minus its direction clauses (user_style.split_typed): only words are candidates
    text_src, tok_t, lyr_lines, tok_e = ("pasted" if text.strip() else None), None, [], []
    lyr = [l for l in read.get("lyrics") or [] if isinstance(l, dict) and str(l.get("text") or "").strip()]
    if not text.strip() and lyr:                                    # no pasted text: the lyrics are the text candidates, each token with its sung time (source seconds)
        text, text_src, tok_t = "\n".join(" ".join(str(l["text"]).split()) for l in lyr)[:20000], "lyrics", []
        for j, l in enumerate(lyr):
            words = str(l["text"]).split()
            t0 = _num(l.get("t")) or 0.0
            ts = _line_ts(lyr, j)
            lyr_lines.append((len(tok_t), len(tok_t) + len(words) - 1, round(t0, 2)))
            tok_t += [round(float(x), 2) for x in ts]
            tok_e += _line_ends(lyr, j, len(words))
    asr = [(_cmp_word(w), round(float(t), 2)) for j, l in enumerate(lyr) for w, t in zip(str(l["text"]).split(), _line_ts(lyr, j))]     # (word, sung second in the source file) of the whole transcription, any text source
    tokens = text.split()
    if tok_t is not None and len(tok_t) != len(tokens):
        tok_t, text_src, lyr_lines, tok_e = None, "pasted", [], []
    sel, sel_notes, sel_shot = _user_select(req.get("select"), len(tokens))
    vp = _video_plan(read, video, req.get("length", DEFAULT_LENGTH), bpm, lines, sel) if (video and not images) else None          # the box never holds a video and pictures together; if a READ does, the picture rules apply
    if vp:
        images, used, overflow, use_req, n = [], [], [], None, len(vp["pieces"])
        length = vp["length"]
    else:
        n_pics = len(_use(req.get("use"), images, MAX_IMAGES)[1])           # the pictures the story is about (the user's choice, STORYBOARD v12): shots never exceed them
        n = shots_for(length, n_pics)
        use_req, used, overflow = _use(req.get("use"), images, n)
    fonts, font_notes = _fonts(req.get("fonts"), n)
    wp = None
    if tok_t:                                                       # WORD PICK (tools/fusion/word_pick.py): the song's tables for scoring the sung words (repeats, held notes, beats, line ends)
        grid_b = (read.get("beat_grid") or {}).get("beats") if isinstance(read.get("beat_grid"), dict) else None
        grid_b = grid_b or ((mu or {}).get("beats") if isinstance((mu or {}).get("beats"), list) else None) or []
        off = vp["w0"] if vp else 0.0                                 # the grid is on the window clock in video mode, the lyrics on the source clock
        wp = WP.prepare(tokens, tok_t, tok_e if len(tok_e) == len(tok_t) else None, [(a_, b_) for a_, b_, _ in lyr_lines], [float(x) + off for x in grid_b if _num(x) is not None])
    return dict(read=read, length=length, seed=seed, format=str(req.get("format") or "9:16"), bpm=bpm, key=_ws((mu or {}).get("key"), 24), images=images, used=used, overflow=overflow,
                use_req=use_req, video=video, audio=(kind("audio") or [None])[0], lines=lines[:CO.BRIEF_MAX_LINES], n=n,
                text=text, tokens=tokens, select=sel, select_shot=sel_shot, fonts=fonts, req_notes=sel_notes + font_notes + (vp["notes"] if vp else []),
                vmode=vp is not None, vplan=vp, slots=(vp["slots"] if vp else []), text_src=text_src, tok_t=tok_t, lyr_lines=lyr_lines, asr=asr, wp=wp, tok_e=tok_e,
                soundtrack=(lambda v: "video" if v == "keep" else (v if v in ("video", "track", "generate") else ""))(str(read.get("soundtrack") or "").strip().lower()))     # v19: which audio the READ analysed


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _line_ts(lyr: list, j: int) -> list[float]:
    """the sung start (source s) of every word of lyric line j: the READ's per-word `ts` (Whisper word timestamps), else spread over the line (never past the next line)."""
    l = lyr[j]
    words = str(l["text"]).split()
    t0 = _num(l.get("t")) or 0.0
    ts = [_num(x) for x in l["ts"]] if isinstance(l.get("ts"), list) and len(l["ts"]) == len(words) else None
    if (ts is None or None in ts) and isinstance(l.get("words"), list) and len(l["words"]) == len(words):          # the READ's per-word detail [{text, t, end, p}]
        ts = [_num(x.get("t")) if isinstance(x, dict) else None for x in l["words"]]
    if ts is None or None in ts:
        nxt = (_num(lyr[j + 1].get("t")) if j + 1 < len(lyr) else None)
        span = max(0.2, min((nxt - t0) if nxt is not None and nxt > t0 else 0.45 * len(words), 0.45 * len(words)))
        ts = [t0 + span * i / len(words) for i in range(len(words))]
    return ts


def _line_ends(lyr: list, j: int, n: int) -> list:
    """the sung END (source s) of every word of lyric line j when the READ carries per-word detail (`words`: [{text, t, end, p}], Whisper word timestamps), else [None] * n."""
    w = lyr[j].get("words")
    if isinstance(w, list) and len(w) == n:
        return [_num(x.get("end")) if isinstance(x, dict) else None for x in w]
    return [None] * n


def _word_alts(R: dict, spans) -> dict:
    """WORD PICK alternatives per shot (key = shot n): [{t: [token index, ...], s: score}] best first, from the lyrics sung inside the shot's span (source seconds). The page's cycle button steps through them."""
    wp = R.get("wp")
    if not wp or not wp["timed"] or R["text_src"] != "lyrics":
        return {}
    out = {}
    for k, (a, b) in enumerate(spans, 1):
        alts = [x for x in WP.alternatives(wp, a, b, R["seed"] + k, 6, max_chars=CO.MAX_LINE_CHARS) if max(x["t"]) < TEXT_CAP]
        if alts:
            out[str(k)] = alts
    return out


def _video_plan(read: dict, video: dict, req_length, bpm, lines: list, sel) -> dict:
    """VIDEO MODE: the Length capped to the window / video, the cut pieces inside it (-> shots), and the render's word slots (own_video_job.slots_from_analysis on the READ's beat grid).
    -> {length, w0, w1, duration, pieces [[src_a, src_b]], times [(start, end) in the piece], slots [{land, hold_until}], notes}."""
    notes = []
    L = _num(req_length) or float(DEFAULT_LENGTH)
    dur = _num(read.get("video_duration"))
    win = video.get("window") if isinstance(video.get("window"), (list, tuple)) and len(video["window"]) == 2 else None
    a, b = (_num(win[0]), _num(win[1])) if win else (None, None)
    w0 = max(0.0, a) if a is not None else 0.0
    w1 = b if (b is not None and a is not None and b > a) else None
    if dur:
        w1 = min(w1, dur) if w1 else dur
    avail = (w1 - w0) if w1 is not None else None
    eff = L
    if avail is not None and avail > 0 and L > avail + 1e-6:
        eff = round(avail, 3)                                                         # the piece keeps the video's own length, to the frame (user 2026-11-03); no note: this is what was asked for
    eff = max(1.0, eff)
    eff = int(eff) if float(eff).is_integer() else eff
    w1 = w0 + eff
    cuts = []
    for c in read.get("video_cuts") or []:
        if isinstance(c, dict) and _num(c.get("start")) is not None and _num(c.get("end")) is not None:
            cuts += [_num(c["start"]), _num(c["end"])]
    has_cuts = bool(read.get("video_cuts"))
    if has_cuts:
        pieces = video_pieces(cuts, w0, w1)
    else:
        k = max(shot_count(max(eff, PR.MIN_LENGTH)), math.ceil(eff / CLIP_MAX_S - 1e-9))
        pieces = [[round(w0 + eff * i / k, 3), round(w0 + eff * (i + 1) / k, 3)] for i in range(k)]
        notes.append("the READ has no video_cuts: the clip is split evenly")
    times = [(round(x - w0, 3), round(y - w0, 3)) for x, y in pieces]
    grid = read.get("beat_grid") if isinstance(read.get("beat_grid"), dict) else {}
    ana = dict(bpm=_num(grid.get("bpm")) or bpm or 90.0, beats=[float(x) for x in (grid.get("beats") or []) if _num(x) is not None],
               downbeats=[float(x) for x in (grid.get("downbeats") or []) if _num(x) is not None])
    if not ana["beats"]:
        notes.append("no beat grid in the READ: the word slots sit on a synthetic grid" + (f" at {ana['bpm']:g} bpm" if (_num(grid.get("bpm")) or bpm) else " (90 bpm default)"))
    beat_s = 60.0 / ana["bpm"]
    beats = [b for b in ana["beats"] if 0.0 <= b <= eff + 1e-6] or [round(k * beat_s, 3) for k in range(int(eff / beat_s) + 1)]       # the grid the sung words snap to (window seconds)
    n_want = len(sel) if sel else (len(lines) if lines else VIDEO_WORDS_DEFAULT)           # own_video_job: n_want = len(texts) or 4
    from tools.app import own_video_job as OJ
    slots = [dict(land=float(x["land"]), hold_until=float(x["hold_until"])) for x in OJ.slots_from_analysis(ana, max(1, min(n_want, CO.BRIEF_MAX_LINES)), float(eff))]
    return dict(length=eff, w0=w0, w1=w1, duration=dur, pieces=pieces, times=times, slots=slots, notes=notes, has_cuts=has_cuts, beats=beats)


def _use(raw, images: list, n: int):
    """req.use (image ids the user wants, e.g. ["I1","I4","I5"]) -> (the valid ids or None, the images used = the first n of them in the user's order, the wanted ones beyond n).
    No / empty / no valid id = the first n images in order."""
    ids = [images[k]["id"] for k in range(len(images))]
    want = None
    if isinstance(raw, (list, tuple)):
        w = {str(x).strip().upper() for x in raw}
        want = [i for i in ids if i.upper() in w] or None
    chosen = [m for m in images if want is None or m["id"] in want]
    return want, chosen[:n], chosen[n:]


def _user_select(raw, ntok: int):
    """req.select -> ({key: [token index, ...]} or None, notes, by_shot). Accepts [{"i", "phrase", "shot"?}] or a plain list of token indices (adjacent indices = one phrase, phrases numbered 1.. in order).
    v18: when any entry carries `shot` (an int >= 1) the selection is BY SHOT: the key is the shot number, the entry's `phrase` is ignored, an entry without `shot` is ignored (said in notes).
    Without `shot` the key is the phrase number (phrase k = shot k / word slot k, as before). Honoured verbatim: the only checks are range and duplicates."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None, [], False
    notes, seen, ph = [], set(), {}
    plain = [x for x in raw if not isinstance(x, dict)]
    by_shot = any(isinstance(x, dict) and _int(x.get("shot")) is not None for x in raw)
    last, k = None, 0
    for x in raw:
        try:
            if isinstance(x, dict):
                i = int(x.get("i"))
                if by_shot:
                    p = _int(x.get("shot"))
                    if p is None or p < 1:
                        notes.append(f"select: token {i} has no shot, ignored")
                        continue
                else:
                    p = int(x.get("phrase") or 1)
            else:
                i = int(x)
                k = k + 1 if (last is None or i != last + 1) else k
                p = k
                last = i
        except (TypeError, ValueError):
            continue
        if not 0 <= i < ntok or i in seen or p < 1:
            notes.append(f"select: token {i} ignored (out of range or repeated)")
            continue
        seen.add(i)
        ph.setdefault(p, []).append(i)
    if plain and len(plain) != len(raw):
        notes.append("select: mixed formats")
    return ({p: sorted(v) for p, v in sorted(ph.items())} or None), notes, by_shot and bool(ph)


def _int(v):
    try:
        return int(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def _fonts(raw, n: int):
    """req.fonts (list by shot, or {"1": file}) -> ({shot k: liked font file}, notes). A name that is not in liked.json is refused (never substituted)."""
    items = list(enumerate(raw, 1)) if isinstance(raw, (list, tuple)) else [(int(k), v) for k, v in raw.items() if str(k).isdigit()] if isinstance(raw, dict) else []
    out, notes = {}, []
    for k, v in items:
        if not v or k > n:
            continue
        e = FL.lookup(v)
        if e is None:
            notes.append(f"shot {k}: font {str(v)[:40]!r} is not a liked font, ignored")
        else:
            out[k] = e["file"]
    return out, notes


def shot_count(length) -> int:
    """the planner's number of shots for a Length (one lockup per shot)."""
    return CO.lockup_count(max(PR.MIN_LENGTH, min(PR.MAX_LENGTH, int(round(float(length))))), "cut")


CLIP_MAX_S = PR.SHOT_FRAMES / PR.FPS          # the longest span one shot may have = one LTX clip (plan_rules.cut_spans' cap, 121 frames = 5.04 s; the contract's "5.0 s": same table for whole-second Lengths)


def shots_for(length, n_images: int) -> int:
    """STORYBOARD v12, the ONE shot-count rule (bind.js mirrors it): shots never exceed pictures. n_images >= 2: min(n_images, shot_count(length)); n_images 0 (text only) and 1: shot_count(length)
    as before (one picture: the later shots have no picture and invent theirs, flagged in capacity.why)."""
    n = shot_count(length)
    return min(n, int(n_images)) if int(n_images) >= 2 else n


def is_feasible(length, n_images: int) -> bool:
    """can a clip of at most CLIP_MAX_S carry every shot of `length` s with n_images pictures (length / shots_for <= the clip max, in frames like the planner)?"""
    s = shots_for(length, n_images)
    return int(round(float(length) * PR.FPS)) / s <= PR.SHOT_FRAMES + 1e-9


def feasible_lengths(n_images: int, options=PR.LENGTHS) -> list[int]:
    """the Length options that are feasible with n_images pictures (STORYBOARD v12): 0 or 1 picture -> all of them (today's rule), n >= 2 -> only where no shot is longer than one clip."""
    return [L for L in options if is_feasible(L, n_images)]


VIDEO_MIN_FRAMES = 6                  # v18: a real cut piece shorter than this many frames (at the clip's fps) merges into its neighbour = own_video_job.clean_cuts' min_piece (no 1 s rule, no piece cap)
VIDEO_FPS_DEFAULT = 30.0              # the clip fps when the caller does not know it (6 frames = 0.2 s)
VIDEO_WORDS_DEFAULT = 4               # own_video_job: `n_want = len(texts) or 4`
CAMERA_FILMED = "as filmed"
# v18: video mode has NO transition field: the cuts between the footage's shots are the design (artist) step's choice at render time, not the storyboard's (user 2026-10-31).


def floor_half(x: float) -> float:
    return math.floor(float(x) * 2 + 1e-9) / 2


def video_feasible(duration, options=PR.LENGTHS) -> list:
    """VIDEO MODE: the Length options a video of `duration` s can carry: the options <= duration; a video shorter than the smallest option -> its own duration rounded down to 0.5 s."""
    d = float(duration)
    ok = [L for L in options if L <= d + 1e-6]
    return ok or [floor_half(d)]


def video_pieces(cuts, lo: float, hi: float, fps: float | None = None, max_s: float = CLIP_MAX_S) -> list[list[float]]:
    """the cut pieces [[start, end], ...] (seconds in the SOURCE) of the span lo..hi from the cut times `cuts` (ffmpeg scene detect) (v18: EVERY real cut is kept): cuts outside the span are ignored;
    a cut that would leave a piece under VIDEO_MIN_FRAMES frames (at `fps`, default 30 = 0.2 s) before or after it is dropped, first to last, exactly like own_video_job.clean_cuts (the render's own rule);
    a piece longer than `max_s` is split into equal parts. Pieces meet; first start = lo, last end = hi."""
    lo, hi = float(lo), float(hi)
    f = float(fps) if fps and float(fps) > 0 else VIDEO_FPS_DEFAULT
    frames = lambda a, b: int(round((b - a) * f + 1e-6))                                                 # noqa: E731
    keep, prev = [], lo
    for c in sorted({round(float(c), 3) for c in cuts if lo + 1e-6 < float(c) < hi - 1e-6}):
        if frames(prev, c) >= VIDEO_MIN_FRAMES and frames(c, hi) >= VIDEO_MIN_FRAMES:
            keep.append(c)
            prev = c
    b = [lo] + keep + [hi]
    out = []
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        k = max(1, math.ceil((e - a) / max_s - 1e-9))
        out += [[round(a + (e - a) * j / k, 3), round(a + (e - a) * (j + 1) / k, 3)] for j in range(k)]
    return out


def length_for(n_images: int, lengths=PR.LENGTHS) -> int | None:
    """the shortest quick-pick Length that uses all n_images images (shot count holds them and every shot fits one clip); None = none does."""
    for L in sorted(lengths):
        if shot_count(L) >= n_images and is_feasible(L, n_images):
            return L
    return None


def shot_times(length: int, n: int, bpm=None) -> list[tuple[float, float]]:
    """[(start_s, end_s)] x n: plan_rules.cut_spans over the bar grid (downbeat every 4 beats from 0) of `bpm`, even cuts without a bpm or when the bars cannot hold n shots."""
    nf = int(round(int(length) * FPS))
    grid = None
    if bpm:
        bar = 4 * 60.0 / float(bpm)
        grid = SimpleNamespace(d=[k * bar for k in range(int(length / bar) + 2)])
    spans = PR.cut_spans(nf, grid, FPS, n=n)
    if len(spans) != n or sum(spans) != nf:                                      # cut_spans may add shots when the bars do not fit: the story keeps the planner's count
        b = [int(round(nf * k / n)) for k in range(n + 1)]
        spans = [b[i + 1] - b[i] for i in range(n)]
    out, t = [], 0
    for s in spans:
        out.append((round(t / FPS, 3), round((t + s) / FPS, 3)))
        t += s
    return out


# ----------------------------------------------------------------------------------------------------------- prompt
def _spec(n: int, fixed: list[str], pick: bool = False) -> str:
    """the reply shape. pick = the words come from the user's numbered text: each shot names the token indices instead of writing a line."""
    shots = []
    for k in range(1, n + 1):
        if pick:
            line = '"pick": [<shot %d: the numbers of 1-%d ADJACENT words of the text, ascending>]' % (k, CO.LOCKUP_WORDS)
        elif fixed:
            line = '"line": "<copy this exactly: %s>"' % json.dumps(fixed[k - 1], ensure_ascii=False) if k <= len(fixed) else '"line": ""'
        else:
            line = f'"line": "<shot {k}: ONE short word, UPPER CASE (two or three words only when they are one fixed phrase)>"'
        cam = "hold (the last shot)" if k == n else "one of: " + " | ".join(CAMERA_MOVES[1:])
        shots.append('{"n": %d, "beat": "<what happens in shot %d: %s words, concrete and photographic>", %s, "camera": "<%s>"}' % (k, k, BEAT_WORDS, line, cam))
    return ('{\n "title": "<2-%d words>",\n "logline": "<ONE sentence: what happens in the whole piece>",\n "look": "<feel / pace / palette / camera in one sentence>",\n'
            ' "music": "<8-14 comma separated style tags for a text-to-music model: genre, groove, real instruments, mood; no artist names>",\n "shots": [\n  ' % TITLE_WORDS
            + ",\n  ".join(shots) + '\n ],\n "notes": "<anything you could not place, else an empty string>"\n}')


def _numbered(R: dict) -> str:
    tk = R["tokens"][:TEXT_CAP]
    if R["text_src"] == "lyrics" and R["lyr_lines"]:                   # the lyrics, one row per sung line with the time it starts (piece seconds)
        off = R["vplan"]["w0"] if R["vmode"] else 0.0
        rows = [f"[{t - off:.1f} s] " + " ".join(f"{i}:{tk[i]}" for i in range(a, min(b + 1, len(tk)))) for a, b, t in R["lyr_lines"] if a < len(tk)]
        return "\n" + "\n".join(rows) + (f"\n... ({len(R['tokens']) - TEXT_CAP} more words not shown)" if len(R["tokens"]) > TEXT_CAP else "")
    return " ".join(f"{i}:{t}" for i, t in enumerate(tk)) + (f"  ... ({len(R['tokens']) - TEXT_CAP} more words not shown)" if len(R["tokens"]) > TEXT_CAP else "")


def _text_mode(R: dict) -> str:
    """how the on-screen words are decided: 'fixed' (the READ's lines / quoted words, or the user's own select), 'pick' (the model picks words of the pasted text), 'free' (the model writes them)."""
    if R["select"] or R["lines"]:
        return "fixed"
    return "pick" if R["tokens"] else "free"


def story_prompt(req) -> str:
    R = norm_req(req)
    if R["vmode"]:
        return _video_prompt(R)
    read, n, length, lines = R["read"], R["n"], R["length"], R["lines"]
    times = shot_times(length, n, R["bpm"])
    mode = _text_mode(R)
    fixed = _fixed_lines(R) if mode == "fixed" else []
    imgs = R["used"]
    L = [f"You write the STORY of one short music-video piece: kinetic typography over generated footage and a generated song. The user dropped pictures into an app; the piece is built from your story, "
         f"so every shot must be filmable and the story must FIT the pictures the user gave, in their order.",
         "",
         f"THE PIECE: {length} seconds, format {R['format']}, exactly {n} shot{'s' if n != 1 else ''} (hard cuts), one short line of type per shot at most."]
    L += ["", "THE SHOTS AND THEIR TIME SPANS:"]
    for k, (a, b) in enumerate(times, 1):
        pic = imgs[k - 1] if k <= len(imgs) else None
        L.append(f"- shot {k}: {a:g}-{b:g} s" + (f", starts from the user's picture {pic['id']}" if pic else (", no picture of the user: you invent it, carrying on from the previous shot" if imgs or R["video"] else "")))
    if imgs:
        L += ["", "THE USER'S PICTURES (already described by a reader; treat them as fixed facts: the subject, place, light and colours must not contradict them):"]
        for k, m in enumerate(imgs[:MAX_IMAGES], 1):
            bits = [_ws(m.get("shows"), 240).rstrip(".") or "(not described)"] + ([f"mood {_ws(m.get('mood'), 60)}"] if m.get("mood") else []) + ([f"palette {', '.join(str(c) for c in m['palette'][:5])}"] if m.get("palette") else [])
            L.append(f"- picture {k} ({m['id']}) = shot {k}" + (" (the first frame)" if k == 1 else "") + ": " + "; ".join(bits))
        if len(R["images"]) > len(imgs):
            L.append(f"- the user gave {len(R['images'])} pictures but only {len(imgs)} are used ({n} shots at {length} s or the user's choice): the others are not used; do not mention them.")
    v = R["video"]
    if v:
        L += ["", f"THE USER'S VIDEO ({v.get('id') or 'V1'}): {_ws(v.get('shows'), 240) or '(not described)'}" + (f"; window {v['window'][0]:g}-{v['window'][1]:g} s" if isinstance(v.get("window"), list) and len(v["window"]) == 2 else "")
              + (f"; mood {_ws(v.get('mood'), 60)}" if v.get("mood") else "")]
    a = R["audio"]
    if a and R["soundtrack"] != "generate":                                  # v19: with "new music" the audio file is not used
        L += ["", f"THE USER'S AUDIO ({a.get('id') or 'A1'}): {_ws(a.get('shows'), 240) or 'the soundtrack'}" + (f"; mood {_ws(a.get('mood'), 60)}" if a.get("mood") else "")]
    if R["bpm"]:
        L += ["", f"MUSIC (measured): {R['bpm']:g} bpm" + (f", key {R['key']}" if R["key"] else "") + ". The user's own song is the soundtrack; the cuts follow its bars, so pace the beats to it."]
    lyr = [l for l in read.get("lyrics") or [] if isinstance(l, dict) and str(l.get("text") or "").strip()]
    if lyr and not (R["text_src"] == "lyrics" and mode == "pick"):
        L += ["", "SUNG LYRICS of the song (seconds in the source file; let the story answer what is sung):"] + [f"- [{float(l.get('t') or 0):.1f} s] {_ws(l['text'], 200)}" for l in lyr]
    L += ["", "THE USER'S DIRECTION:"]
    L.append("- CONCEPT (follow it; the logline restates it): " + _ws(read.get("concept"), 600) if _ws(read.get("concept"), 600) else "- concept: none given, invent the story from the pictures.")
    L.append("- LOOK / FEEL (follow it): " + _ws(read.get("look"), 300) if _ws(read.get("look"), 300) else "- look: none given, take it from the pictures' mood and palette.")
    if mode == "pick":
        lyr_src = R["text_src"] == "lyrics"
        L.append(("- THE SONG'S LYRICS (the audio file's), numbered word by word (index:word), each row with the second it starts in the audio: " if lyr_src else "- THE USER'S TEXT, numbered word by word (index:word): ") + _numbered(R))
        if lyr_src:
            L.append("- The type must land WITH the singing: shot k's phrase comes from the lyrics sung inside shot k's time span (the row times above); if nothing is sung in that span give an empty `pick`.")
        L.append(f"- ON-SCREEN WORDS: you choose them FROM THIS TEXT, never write other words. Each shot k shows ONE phrase = 1-{CO.LOCKUP_WORDS} ADJACENT words of the text (give their numbers in `pick`, ascending), "
                 f"the phrases follow the text's order (shot 2's numbers are higher than shot 1's), no number is used twice. Prefer the strongest, most visual or most emotional words; skip filler words; "
                 f"if the text is a song or poem take its hook lines. " + PHRASE_RULE + " Shots after the end of the useful text may repeat nothing: give an empty `pick`.")
    elif fixed:
        L.append(f"- ON-SCREEN LINES, fixed by the user: {json.dumps(fixed, ensure_ascii=False)}. Line k is the type of shot k (an empty line = that shot has no type); shots after the last line have no type. Copy them letter for letter into `line` "
                 f"(the code restores them anyway; never translate, reword, add, merge or drop one) and write each beat so that its line belongs to it.")
    else:
        L.append(f"- on-screen lines: none given, you write them: one per shot, UPPER CASE, 1-{CO.LOCKUP_WORDS} words, at most {CO.MAX_LINE_CHARS} letters, a feeling or a turn of the story, not a description of the picture; "
                 f"each shot's line differs from the others; never use {', '.join(CO.OVERUSED_WORDS)}. " + PHRASE_RULE)
    L += ["", "RULES:",
          f"1. ONE story with a beginning, a turn and an end across the {n} shot{'s' if n != 1 else ''}; each beat says what happens in THAT shot (what the subject does, where, the light), 1 sentence of {BEAT_WORDS} words, concrete and photographic.",
          "2. Shot k starts from picture k: its beat begins with what that picture shows and moves the story on. The same subject and world run through the shots (one subject, one description); the beats differ from each other.",
          "3. A beat contains no text, letters, captions or logos. Never invent a person or object that contradicts a picture.",
          "4. The music is playable by real instruments (no toy sounds)" + ("; it suits the measured tempo" if R["bpm"] else "") + ".",
          "5. `camera` of shot k = the camera move that carries shot k into shot k+1 (the cut after it): depth_pivot = the picture breaks into cards and turns, orbit:push = the camera dives into the subject "
          "(a build), orbit:pullback = it backs out (a release), orbit:fly = an S-curve through the scene, pan:<dir> = the type camera travels that way. Choose from the list in the reply shape, vary them, "
          "match the story's energy; the last shot is `hold`.",
          "", f"Story seed (vary your choices with it): {R['seed']}.", "", "Reply with ONE JSON object and nothing else, in exactly this shape:", _spec(n, fixed, mode == "pick")]
    return "\n".join(L)


def _video_prompt(R: dict) -> str:
    """VIDEO MODE: title / logline / look / music, the words for the render's own word slots (the cuts are NOT described and have no transitions: shot.beat stays empty). Never invent shots."""
    read, n, vp, v, slots = R["read"], R["n"], R["vplan"], R["video"], R["slots"]
    mode = _text_mode(R)
    fixed = _fixed_lines(R)[: len(slots)] if mode == "fixed" else []
    cut_of = lambda t: next((k for k, (a, b) in enumerate(vp["times"], 1) if a <= t < b), n)      # noqa: E731
    L = ["You write the STORY of one short music-video piece. The user's own VIDEO is re-rendered with big kinetic typography on top of it. The piece IS the user's footage, in its own order, with its own cuts: "
         "you choose the words; you never invent shots, people, places or events that the footage description does not contain.", "",
         f"THE PIECE: {R['length']:g} s taken from the user's video (seconds {vp['w0']:g}-{vp['w1']:g} of a {vp['duration'] or vp['w1']:g} s clip), format {R['format']}, exactly {n} shot{'s' if n != 1 else ''} = the REAL CUTS found in the footage "
         f"(hard cuts; the times are in the piece, the video's own time in brackets):"]
    for k, ((a, b), (x, y)) in enumerate(zip(vp["times"], vp["pieces"]), 1):
        L.append(f"- shot {k}: {a:g}-{b:g} s (video {x:g}-{y:g} s)")
    L += ["", f"THE USER'S VIDEO ({v.get('id') or 'V1'}), as described by a reader that saw it: {_ws(v.get('shows'), 400) or '(not described)'}" + (f"; mood {_ws(v.get('mood'), 60)}" if v.get("mood") else "")
          + ("; palette " + ", ".join(str(c) for c in v["palette"][:5]) if v.get("palette") else "")]
    if R["bpm"]:
        own = "the user's own audio file replaces the video's sound and is the soundtrack" if R["soundtrack"] == "track" else "the video's own audio is the soundtrack"
        L += ["", f"MUSIC (measured, {own}): {R['bpm']:g} bpm" + (f", key {R['key']}" if R["key"] else "") + "."]
    lyr = [l for l in read.get("lyrics") or [] if isinstance(l, dict) and str(l.get("text") or "").strip()]
    if lyr and mode != "pick":
        L += ["", "SUNG LYRICS (seconds in the piece; let the story answer what is sung):"] + [f"- [{float(l.get('t') or 0) - vp['w0']:.1f} s] {_ws(l['text'], 200)}" for l in lyr]
    L += ["", "THE USER'S DIRECTION:"]
    L.append("- CONCEPT (follow it; the logline restates it): " + _ws(read.get("concept"), 600) if _ws(read.get("concept"), 600) else "- concept: none given, take it from the footage and the lyrics.")
    L.append("- LOOK / FEEL (follow it): " + _ws(read.get("look"), 300) if _ws(read.get("look"), 300) else "- look: none given, take it from the footage's mood and palette.")
    L += ["", f"WORD SLOTS (the render lands each word ON a beat at this time and holds it; {len(slots)} slot{'s' if len(slots) != 1 else ''}):"]
    L += [f"  slot {j}: lands at {s['land']:.2f} s (during shot {cut_of(s['land'])})" for j, s in enumerate(slots, 1)]
    if mode == "pick":
        lyr_src = R["text_src"] == "lyrics"
        L.append(("- THE SONG'S LYRICS, numbered word by word (index:word), each row with the second it is sung in the piece: " if lyr_src else "- THE USER'S TEXT, numbered word by word (index:word): ") + _numbered(R))
        L.append(f"- ON-SCREEN WORDS: you choose them FROM THIS TEXT, never write other words. Slot j shows ONE phrase = 1-{CO.LOCKUP_WORDS} ADJACENT words (give their numbers in `pick`, ascending), the phrases follow the text's order "
                 f"(slot 2's numbers are higher than slot 1's), no number is used twice. " + ("Prefer the phrase sung closest to the slot's landing time. " if lyr_src else "")
                 + "Prefer the strongest, most visual or most emotional words; skip filler words; if the text is a song or poem take its hook lines. " + PHRASE_RULE
                 + (" A phrase lands when its first word is sung (the render reads the time from the row times above); the slot times are only a fallback." if lyr_src else ""))
    elif fixed:
        L.append(f"- ON-SCREEN LINES, fixed by the user: {json.dumps(fixed, ensure_ascii=False)}. Line j is the type of slot j. You do not write words; write each beat so that the line landing in its shot belongs to it.")
    else:
        from tools.app import own_video_v7 as V7
        shows = _ws(v.get("shows"), 400)
        theme = " ".join(x for x in (_ws(read.get("concept"), 300), shows, f"Look: {_ws(read.get('look'), 200)}" if _ws(read.get("look"), 200) else "") if x) or "the user's video"
        reading = dict(arc=_ws(read.get("look"), 200), sections=[dict(start=0.0, end=float(R["length"]), label="track", energy="")], hit_moments=[])
        head = V7.words_prompt([dict(s, section="track") for s in slots], reading, theme).split("\nReply with JSON only")[0]
        L += ["- ON-SCREEN WORDS: none given, you choose them (the render's own word chooser says:)", head, "Give your words in `hits` (slot j = slot j above). " + PHRASE_RULE]
    L += ["", "RULES:",
          "1. You do NOT describe the cuts: the user's footage is kept as it is, so write no beat for any shot.",
          "2. You do not choose transitions or camera moves: the render's design step decides how the shots join.",
          "3. title / logline / look come from the footage" + (" and the meaning of the lyrics" if lyr else "") + "; music = the style of the soundtrack in 8-14 comma separated tags ("
          + {"track": "the user's own audio file is the soundtrack: describe it, do not change it", "generate": "new music will be written for the piece: describe the music that fits the footage's mood and look"}.get(R["soundtrack"], "the video's own audio is kept: describe it, do not change it") + ").",
          "", f"Story seed (vary your choices with it): {R['seed']}.", "", "Reply with ONE JSON object and nothing else, in exactly this shape:", _video_spec(R, mode)]
    return "\n".join(L)


def _video_spec(R: dict, mode: str) -> str:
    n, vp = R["n"], R["vplan"]
    hits = ""
    if mode == "pick":
        hits = ',\n "hits": [\n  ' + ",\n  ".join('{"slot": %d, "pick": [<slot %d: the numbers of 1-%d ADJACENT words, ascending>]}' % (j, j, CO.LOCKUP_WORDS) for j in range(1, len(R["slots"]) + 1)) + "\n ]"
    elif mode == "free":
        hits = ',\n "hits": [\n  ' + ",\n  ".join('{"slot": %d, "text": "<slot %d: 1-2 UPPER CASE words>"}' % (j, j) for j in range(1, len(R["slots"]) + 1)) + "\n ]"
    return ('{\n "title": "<2-%d words>",\n "logline": "<ONE sentence: what happens in the whole piece>",\n "look": "<feel / pace / palette / camera in one sentence>",\n'
            ' "music": "<8-14 comma separated style tags>"' % TITLE_WORDS + hits + ',\n "notes": "<anything you could not place, else an empty string>"\n}')


# ----------------------------------------------------------------------------------------------------------- the story
def _first_sentence(s: str, n: int = BEAT_CHARS) -> str:
    s = _ws(s, 2000)
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    return _ws(m.group(1) if m else s, n)


def _gen_line(raw, prev: str) -> tuple[str, str | None]:
    """a line Gemma invented -> (cleaned line or "", problem or None). The concept's character clean-up, <= LOCKUP_WORDS words, <= MAX_LINE_CHARS letters, no overused word, no repeat."""
    t = re.sub(r"\s+", " ", CO._CLEAN.sub("", str(raw or "").upper())).strip()
    if not t:
        return "", None
    w = t.split()
    t = trim_phrase(t, CO.LOCKUP_WORDS, CO.MAX_LINE_CHARS, CO.MAX_LINE_CHARS)       # <= LOCKUP_WORDS words, a sensible phrase: never ends on a function word
    if not t:
        return "", "line is only small words"
    t = t[:CO.MAX_LINE_CHARS].strip()
    if t == prev:
        return "", f"line {t!r} repeats the previous one"
    if any(x.strip(".,!?'&-") in CO.OVERUSED_WORDS for x in t.split()):
        return "", f"line {t!r} uses an overused word"
    return t, None


def _fb_title(R: dict) -> str:
    c = _ws(R["read"].get("concept"), 200)
    return " ".join(c.split()[:4]).strip(" .,;:") or "Untitled story"


def _fb_logline(R: dict) -> str:
    c = _ws(R["read"].get("concept"), LOGLINE_CHARS)
    if c:
        return c
    sh = [_first_sentence(m.get("shows"), 120) for m in R["used"] if m.get("shows")]
    if R["vmode"] and not sh and R["video"] and _ws(R["video"].get("shows"), 240):
        sh = [_first_sentence(R["video"]["shows"], 200)]
    if len(sh) >= 2:
        return _ws(f"From: {sh[0].rstrip('.')}. To: {sh[-1].rstrip('.')}.", LOGLINE_CHARS)
    return sh[0] if sh else ""


def _fb_look(R: dict) -> str:
    if _ws(R["read"].get("look"), LOOK_CHARS):
        return _ws(R["read"].get("look"), LOOK_CHARS)
    moods = []
    for m in R["images"] + [x for x in (R["video"], R["audio"]) if x]:
        t = _ws(m.get("mood"), 60)
        if t and t not in moods:
            moods.append(t)
    return _ws("; ".join(moods[:4]), LOOK_CHARS)


def _fb_beat(R: dict, k: int) -> str:
    if R["vmode"]:
        a, b = R["vplan"]["times"][k - 1]
        sh = _first_sentence((R["video"] or {}).get("shows"))
        return _ws(f"Cut {k} ({a:g}-{b:g} s): {sh}" if sh else f"Cut {k} of the video ({a:g}-{b:g} s).", BEAT_CHARS)
    imgs = R["used"]
    if k <= len(imgs) and _ws(imgs[k - 1].get("shows"), 240):
        return _first_sentence(imgs[k - 1]["shows"])
    prev = next((_first_sentence(m["shows"]) for m in reversed(imgs[:k]) if _ws(m.get("shows"), 240)), "")
    if not prev and R["video"] and _ws(R["video"].get("shows"), 240):
        prev = _first_sentence(R["video"]["shows"])
    return _ws(f"The scene carries on, a new angle and later light: {prev[0].lower() + prev[1:]}" if prev else "", BEAT_CHARS)


def _music_string(raw, R: dict) -> str:
    m = _ws(raw, MUSIC_CHARS)
    if R["bpm"] and "bpm" not in m.lower():
        m = (m + ", " if m else "the user's own song, ") + f"{R['bpm']:g} bpm"
    return m


def _fixed_lines(R: dict) -> list[str]:
    """the lines the user fixed: the user's own word choice (req.select: one phrase per line) or the READ's on_screen (merged by the planner's fit_lockups when there are more than shots)."""
    if R["select"] and R["select_shot"]:                                     # v18: by shot. Image mode: one entry per shot ("" = no words); video mode: the shots' lines in shot order (empty ones left out = the word slots)
        if R["vmode"]:
            return [t for t in (_join_tokens(R, idx) for p, idx in R["select"].items() if 1 <= p <= R["n"]) if t]
        out = [""] * R["n"]
        for p, idx in R["select"].items():
            if 1 <= p <= R["n"]:
                out[p - 1] = _join_tokens(R, idx)
        return out
    if R["select"]:
        return [_join_tokens(R, idx) for p, idx in R["select"].items()][: (len(R["slots"]) if R["vmode"] else R["n"])]
    lines = R["lines"]
    if R["text_src"] == "lyrics":                                            # lines taken from the SUNG lyrics: each a sensible standalone phrase, never ending on a function word (a line not in the lyrics stays verbatim)
        lines = [t for t in (_lyric_phrase(R, x) for x in lines) if t]
    if R["vmode"]:                                                           # the render drops the lines that find no word slot (own_video_job), it does not merge them
        return lines
    return lines if len(lines) <= R["n"] else PR.fit_lockups(lines, R["n"])[0]


def _clean_word(t: str) -> str:
    """a token -> its on-screen form (the concept's character clean-up, UPPER CASE); "" when nothing letter-like is left (a lone comma or dash is not a word)."""
    w = re.sub(r"\s+", " ", CO._CLEAN.sub("", str(t or "").upper())).strip()
    return w if re.search(r"[A-Z0-9]", w) else ""


def _same_word(a: str, b: str) -> bool:
    return a.strip(".,!?'&-") == b.strip(".,!?'&-")


def _join_tokens(R: dict, idx: list[int]) -> str:
    return " ".join(w for w in (_clean_word(R["tokens"][i]) for i in idx if 0 <= i < len(R["tokens"])) if w)


def _lyric_phrase(R: dict, line: str) -> str:
    """a READ on_screen line that is a run of the sung lyrics -> its sensible standalone phrase (trim_phrase); any other line (the user's own words) is returned as it is."""
    words = [w for w in (_clean_word(t) for t in str(line).split()) if w]
    return trim_phrase(line) if words and _find_sung(R, [_cmp_word(w) for w in words], 0, False) is not None else line


def _lines_for(R: dict) -> list[str]:
    lines = _fixed_lines(R)
    return lines + [""] * (R["n"] - len(lines))


def _find_run(R: dict, words: list[str], start: int) -> list[int] | None:
    """token indices (consecutive) from `start` on whose cleaned form equals `words`, or None."""
    toks = [_clean_word(t) for t in R["tokens"]]
    for i in range(start, len(toks) - len(words) + 1):
        if all(_same_word(toks[i + j], words[j]) for j in range(len(words))):
            return list(range(i, i + len(words)))
    return None


def _rule_pick(R: dict, k: int, lo: int, hi_exclusive: int, n: int | None = None) -> list[int]:
    """a rule phrase for shot k: content words (not filler) from `lo` on, spread over what is left of the text: the word and, when the next one is a content word too, its neighbour."""
    toks = [_clean_word(t) for t in R["tokens"]]
    ok = [i for i in range(lo, min(hi_exclusive, len(toks))) if toks[i] and toks[i].lower().strip(".,!?'&-") not in STOP and len(toks[i]) > 1]
    if not ok:
        return []
    n_left = max((n or R["n"]) - k + 1, 1)
    i = ok[min(len(ok) - 1, (len(ok) // n_left) // 2)]
    run = [i]
    if i + 1 < len(toks) and i + 1 in ok and len(toks[i]) + len(toks[i + 1]) + 1 <= CO.MAX_LINE_CHARS:
        run.append(i + 1)
    return run


def _model_pick(R: dict, raw, lo: int) -> list[int]:
    """one shot's `pick` from the reply -> a run of consecutive token indices >= lo, <= LOCKUP_WORDS long, or []."""
    if not isinstance(raw, list):
        return []
    run: list[int] = []
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            break
        if i < lo or i >= min(len(R["tokens"]), TEXT_CAP) or (run and i != run[-1] + 1) or not _clean_word(R["tokens"][i]):
            if run:
                break
            continue
        run.append(i)
        if len(run) >= CO.LOCKUP_WORDS:
            break
    while run and sum(len(_clean_word(R["tokens"][i])) + 1 for i in run) - 1 > CO.MAX_LINE_CHARS:
        run.pop()
    return run


def _tok_range(R: dict, span, lo: int) -> tuple[int, int]:
    """the token index range [i0, i1) of the lyrics sung inside span (a, b) (source seconds), from `lo` on."""
    tt = R["tok_t"]
    return next((i for i, t in enumerate(tt) if t >= span[0] and i >= lo), len(tt)), next((i for i, t in enumerate(tt) if t >= span[1]), len(tt))


def _text_lines(R: dict, picks: list, notes: list[str], n: int | None = None, spans=None) -> tuple[list[str], list[dict]]:
    """text pasted, the model chooses (mode pick): shot k's run of tokens -> (the n lines, text_select). A bad / missing / out-of-order pick is replaced by a rule pick (said in notes).
    n = the number of phrases (shots; the word slots in video mode); spans = per phrase the (a, b) source seconds the sung words should come from (lyrics only: rule picks stay inside it)."""
    n, lo, runs = n or R["n"], 0, []
    wp = R.get("wp") if (spans and R["tok_t"] and R["text_src"] == "lyrics") else None          # WORD PICK: timed lyrics -> the words come from what is sung INSIDE the shot's span, by score
    for k in range(1, n + 1):
        run = _model_pick(R, picks[k - 1] if k <= len(picks) else None, lo)
        if run and R["text_src"] == "lyrics":
            a_, b_ = _phrase_span([R["tokens"][i] for i in run])
            run = run[a_:b_]
        if run and wp and k <= len(spans) and not WP.model_pick_ok(wp, run, spans[k - 1][0], spans[k - 1][1]):
            notes.append(f"shot {k}: the model's words are not sung inside this shot's {spans[k - 1][0]:.1f}-{spans[k - 1][1]:.1f} s: picked by the word score")
            run = []
        if not run and wp and k <= len(spans):
            run = WP.pick(wp, spans[k - 1][0], spans[k - 1][1], lo, R["seed"] + k, max_chars=CO.MAX_LINE_CHARS)
            if run:
                notes.append(f"shot {k}: words picked by the word score (repeats, content, held notes, beats, line ends)")
        elif not run:
            i0, i1 = _tok_range(R, spans[k - 1], lo) if (spans and R["tok_t"] and k <= len(spans)) else (lo, TEXT_CAP)
            run = _rule_pick(R, k, i0, i1, n)
            if run:
                notes.append(f"shot {k}: no usable word choice in the reply, picked by rule")
        if run:
            lo = run[-1] + 1
        runs.append(run)
    return [_join_tokens(R, r) for r in runs], [dict(i=i, phrase=k) for k, r in enumerate(runs, 1) for i in r]


def _text_select_for_lines(R: dict, lines: list[str], notes: list[str]) -> list[dict]:
    """text pasted and the lines are fixed (the READ's quoted / on_screen words): where each shot's line sits in the text (consecutive tokens, in order)."""
    out, pos = [], 0
    for k, ln in enumerate(lines, 1):
        words = [w for w in (_clean_word(t) for t in ln.split()) if w]
        run = _find_run(R, words, pos) if words else None
        if run is None and words:
            run = _find_run(R, words, 0)
        if run:
            out += [dict(i=i, phrase=k) for i in run]
            pos = run[-1] + 1
        elif words:
            notes.append(f"shot {k}: line {ln!r} is not in the pasted text")
    return sorted(out, key=lambda d: d["i"])


def _video_capacity(R: dict) -> dict:
    vp, n = R["vplan"], R["n"]
    dur = vp["duration"]
    opts = video_feasible(dur) if dur else list(PR.LENGTHS)
    feas = {str(int(x)): (x <= dur + 1e-6 if dur else True) for x in PR.LENGTHS}
    if dur and dur < min(PR.LENGTHS):
        feas[f"{floor_half(dur):g}"] = True
    longest = max((b - a for a, b in vp["times"]), default=0.0)
    why = (f"{R['length']:g} s of the video" + (f" (it is {dur:g} s)" if dur else "") + f": {n} shot{'s' if n != 1 else ''} = its real cut{'s' if n != 1 else ''} (longest {longest:.1f} s)"
           if vp["has_cuts"] else f"{R['length']:g} s of the video: no cut information, split into {n} even shots")
    if dur:
        off = [int(x) for x in PR.LENGTHS if x > dur + 1e-6]
        if off:
            why += "; " + ", ".join(f"{x} s" for x in off) + f" {'is' if len(off) == 1 else 'are'} not available: the video is {dur:g} s"
    return dict(shots=n, why=why, images=0, min_length=None, pick_length=None, per_shot_s=round(R["length"] / n, 1), clip_max_s=round(CLIP_MAX_S, 2), feasible=feas,
                video_duration=dur, length=R["length"], feasible_lengths=opts, length_default=max(opts))


def capacity(R: dict) -> dict:
    """how many shots the Length holds, the shortest Length that holds every dropped image, and a plain sentence saying why (the planner's rule: one lockup per shot, mograph.fusion_brief.shot_count)."""
    if R["vmode"]:
        return _video_capacity(R)
    n, ni, L = R["n"], len(R["use_req"] or [m["id"] for m in R["images"]]), R["length"]
    per = round((PR.SHOT_FRAMES - PR.XFADE) / FPS, 1)
    minL = next((x for x in range(PR.MIN_LENGTH, PR.MAX_LENGTH + 1) if shot_count(x) >= ni and is_feasible(x, ni)), None) if ni else None
    why = (f"{L} s fits {n} shot{'s' if n != 1 else ''}: each shot is one clip of up to {PR.SHOT_FRAMES / FPS:.1f} s with a {PR.XFADE / FPS:g} s crossfade, so every line needs about {per:g} s to land and hold")
    pick = length_for(ni) if ni else None
    if ni > n:
        why += (f"; your {ni} pictures need {minL} s or more" + (f" (the {pick} s pick uses all {ni})" if pick and pick != minL else "")) if minL else f"; {ni} pictures do not fit"
    elif ni:
        why += f"; all {ni} picture{'s' if ni != 1 else ''} fit"
    feas = {str(int(x)): is_feasible(x, ni) for x in PR.LENGTHS}
    off = [int(x) for x, ok in feas.items() if not ok]
    if off:
        add = {x: next((k for k in range(ni, MAX_IMAGES + 1) if is_feasible(x, k) and shot_count(x) >= k), None) for x in off}
        why += "; " + ", ".join(f"{x} s is not available with {ni} picture{'s' if ni != 1 else ''}: a shot would run {x / shots_for(x, ni):.1f} s (one clip holds {CLIP_MAX_S:.1f} s)" +
                                (f", add {add[x] - ni} image{'s' if add[x] - ni != 1 else ''} for {x} s" if add[x] and add[x] > ni else "") for x in off)
    if ni == 1:
        why += "; with one picture the later shots have none and invent theirs"
    return dict(shots=n, why=why, images=len(R["images"]), min_length=minL, pick_length=pick, per_shot_s=per, clip_max_s=round(CLIP_MAX_S, 2), feasible=feas)


def _candidates(R: dict) -> list[dict]:
    used = {m["id"]: k for k, m in enumerate(R["used"], 1)}
    cap = {m["id"] for m in R["overflow"]}
    c = capacity(R)
    out = []
    for m in R["images"]:
        i = m["id"]
        if i in used:
            out.append(dict(image=i, use=True, reason=f"shot {used[i]}"))
        elif i in cap:
            out.append(dict(image=i, use=False, reason=f"{R['length']} s fits {R['n']} shots; " + (f"pick {c['pick_length'] or c['min_length']} s to use all {len(R['images'])}" if c["min_length"] else "too many pictures")))
        else:
            out.append(dict(image=i, use=False, reason="left out by you" if R["use_req"] else f"{R['length']} s fits {R['n']} shots"))
    return out


def _extra_note(R: dict) -> str:
    ov = R["overflow"]
    if not ov:
        return ""
    ni = len(R["use_req"] or R["images"])
    best = length_for(ni)
    return f"{ni} images wanted but {R['n']} shots at {R['length']} s: {', '.join(m['id'] for m in ov)} not used" + (f" (Length {best} s uses all {ni})." if best else ".")


def _camera(raw, k: int, n: int, prev: str, seed: int) -> str:
    """shot k's camera move (the cut after it): the model's when valid (cut_kinds.normalize, in CAMERA_MOVES), else the seeded rotation; the last shot holds."""
    if k == n:
        return "hold"
    c = CK.normalize(raw) if str(raw or "").strip().lower() != "hold" else "hold"
    if c in CAMERA_MOVES and c != "hold":
        return c
    for j in range(len(CAMERA_ROTATION)):
        d = CAMERA_ROTATION[(seed + k + j) % len(CAMERA_ROTATION)]
        if d != prev:
            return d
    return "hold"


def _by_sung_time(R: dict, ulines: list[str], sel: list[dict], times: list, notes: list[str]) -> tuple[list[str], list[dict]]:
    """IMAGE mode with lyrics: each chosen phrase goes on the shot whose span holds its sung time (in order; a taken or earlier shot moves it to the next free one; outside the piece = dropped, said in notes).
    -> (the lines per shot, text_select with phrase = the shot that carries it)."""
    n, first, new, mp, last = len(ulines), {}, [""] * len(ulines), {}, -1
    for d in sel:
        first[d["phrase"]] = min(first.get(d["phrase"], 10 ** 9), d["i"])
    for k in sorted(first):
        if not (1 <= k <= n) or not ulines[k - 1]:
            continue
        t = R["tok_t"][first[k]]
        j = next((j for j, (a, b) in enumerate(times) if a <= t < b or (j == len(times) - 1 and a <= t <= b)), None)
        if j is None:
            notes.append(f"{ulines[k - 1]!r} is sung at {t:g} s, outside the {times[-1][1]:g} s piece: dropped")
            continue
        j = max(j, last + 1)
        if j >= n:
            notes.append(f"{ulines[k - 1]!r} (sung at {t:g} s) finds no free shot: dropped")
            continue
        new[j], mp[k], last = ulines[k - 1], j + 1, j
    for k, ln in enumerate(ulines, 1):
        if ln and k not in mp and not new[k - 1]:
            new[k - 1] = ln
    return new, [dict(i=d["i"], phrase=mp[d["phrase"]]) for d in sel if d["phrase"] in mp]


def _text_candidates(R: dict) -> dict | None:
    """the words the storyboard offers: {source: "pasted"|"lyrics", words, tokens [{i, w, t, tp}] (lyrics only: t = sung second in the source file, tp = in the piece)}."""
    if not R["tokens"]:
        return None
    out = dict(source=R["text_src"] or "pasted", words=len(R["tokens"]))
    if R["text_src"] == "lyrics" and R["tok_t"]:
        off = R["vplan"]["w0"] if R["vmode"] else 0.0
        out["tokens"] = [dict(i=i, w=w, t=t, tp=round(t - off, 2)) for i, (w, t) in enumerate(zip(R["tokens"][:TEXT_CAP], R["tok_t"]))]
    return out


def _rule_words(R: dict, k: int, used: set) -> str:
    """a last-resort on-screen word for slot k (the reply gave none): a content word of the footage description / concept, not used yet."""
    src = " ".join([_ws(R["read"].get("concept"), 300), _ws((R["video"] or {}).get("shows"), 400)])
    for w in re.findall(r"[A-Za-z]{3,14}", src):
        u = w.upper()
        if w.lower() not in STOP and u not in used and u not in CO.OVERUSED_WORDS:
            return u
    return next((u for u in ("MOVE", "LIGHT", "NOW", "GO", "AGAIN", "HOLD") if u not in used), "GO")


def _land_for(R: dict, s0: float) -> float:
    """the landing second (piece) of a phrase whose first word is sung at s0: the nearest beat in [s0, s0 + SNAP_S] (never before it is sung), else s0 itself; not before the first word can form (1.4 s)."""
    dur = float(R["length"])
    s = max(s0, min(1.4, 0.3 * dur))
    ahead = [b for b in R["vplan"]["beats"] if s - 1e-6 <= b <= s + SNAP_S + 1e-6]
    return round(min(ahead, key=lambda b: b - s) if ahead else s, 3)


def _find_sung(R: dict, words: list[str], start: int = 0, windowed: bool = True):
    """the consecutive run of transcription words equal to `words` (first at or after `start`, else the first anywhere) whose first word is sung inside the piece -> (first index, last index) or None.
    A line with a word missing from the transcription is placed by its longest run of words that is there (at least half the line)."""
    asr, k = R["asr"], len(words)
    w0, dur = (R["vplan"]["w0"], float(R["length"])) if windowed else (0.0, 1e9)
    if not asr or not k or not all(words):
        return None
    cw = [w for w, _ in asr]
    for part in range(k, 0, -1):                              # the whole line first, then (Whisper drops words it is unsure of, so the READ's lyrics can lack one) the longest run of its words that is there
        if part < k and part < (k + 1) // 2:
            break
        if part == 1 and k > 1 and not any(len(w) >= 4 and not _is_fn(w) for w in words):
            break
        for a in range(0, k - part + 1):                      # the earliest part of the line wins
            sub = words[a:a + part]
            if part == 1 and (len(sub[0]) < 4 or _is_fn(sub[0])):
                continue
            for lo in ((start, 0) if start else (0,)):
                for i in range(lo, len(asr) - part + 1):
                    if cw[i:i + part] == sub and 0.0 <= asr[i][1] - w0 < dur:
                        return i, i + part - 1
    return None


def _sung_hits(R: dict, texts: list, slots: list, sel, notes: list) -> tuple[list[dict], dict]:
    """the word slots' lines -> hits {text, land, hold_until, sung?} and {phrase j: landing second, None when dropped}.
    A line found in the transcription lands when its FIRST word is sung (Whisper word times, piece seconds), snapped forward to a beat within SNAP_S, never before it is sung; hold_until = its LAST
    word sung + MIN_HOLD_S. A line not found keeps its word slot (the render's own beat). Landings are MIN_GAP_S apart: a later one is pushed back by <= PUSH_MAX_S, else dropped (said in notes)."""
    w0, dur = R["vplan"]["w0"], float(R["length"])
    idx = {}
    if R["text_src"] == "lyrics" and R["tok_t"]:
        for d in sel or []:
            idx.setdefault(d["phrase"], []).append(d["i"])
    rows, pos = [], 0
    for j, (t, sl) in enumerate(zip(texts, slots), 1):
        t = " ".join(str(t).upper().split())
        if not t:
            continue
        h = dict(text=t, land=round(sl["land"], 3), hold_until=round(sl["hold_until"], 3), _j=j, _timed=False)
        ii = sorted(i for i in idx.get(j, []) if i < len(R["asr"]))
        run = (ii[0], ii[-1]) if ii else _find_sung(R, [_cmp_word(x) for x in t.split()], pos)
        if run:
            s0, s1 = R["asr"][run[0]][1] - w0, R["asr"][run[1]][1] - w0
            pos = run[1] + 1
            if R["text_src"] == "lyrics" and R["tok_t"]:
                h["sung"] = R["asr"][run[0]][1]
            if 0.0 <= s0 < dur:
                land = _land_for(R, s0)
                h.update(land=land, hold_until=round(max(s1 + MIN_HOLD_S, land + MIN_HOLD_S), 3), _timed=True)
        rows.append(h)
    rows.sort(key=lambda h: h["land"])
    out, land_of = [], {}
    for h in rows:
        if h["_timed"] and h["land"] > dur - 1.0:
            notes.append(f"{h['text']!r} is sung at {h['land']:.1f} s, too near the end of the piece for a hold: dropped")
            land_of[h["_j"]] = None
            continue
        if out and h["land"] < out[-1]["land"] + MIN_GAP_S - 1e-6 and (h["_timed"] or out[-1]["_timed"]):
            need = out[-1]["land"] + MIN_GAP_S
            if need - h["land"] <= PUSH_MAX_S and need <= dur - 1.0:
                h.update(land=round(need, 3), hold_until=round(max(h["hold_until"], need + MIN_HOLD_S), 3))
            else:
                notes.append(f"{h['text']!r} is sung {h['land'] - out[-1]['land']:.1f} s after the line before it (landings need {MIN_GAP_S:g} s): dropped")
                land_of[h["_j"]] = None
                continue
        out.append(h)
        land_of[h["_j"]] = h["land"]
    for a_, b_ in zip(out, out[1:]):
        if a_["_timed"] or b_["_timed"]:
            a_["hold_until"] = round(min(a_["hold_until"], max(a_["land"] + MIN_HOLD_S, b_["land"] - 0.3)), 3)
    for h in out:
        h.pop("_j"), h.pop("_timed")
    return out, land_of


def _auto_shot_ok(R: dict) -> bool:
    """one word per shot applies: VIDEO mode, the words are the timed lyrics, nothing hand-picked (no select, no READ lines)."""
    wp = R.get("wp")
    return bool(R["vmode"] and wp and wp.get("timed") and R["text_src"] == "lyrics" and not R["select"] and not R["lines"])


def _shot_runs(R: dict, k: int, x: float, y: float, top: int = 4) -> list[tuple[list[int], float]]:
    """up to `top` ideas for shot k (source seconds x..y), best first -> [(token run, score)]: the best scored runs whose FIRST word is sung inside the shot (a phrase that runs on past the cut stays on the shot where it
    starts), else the best single sung content word of 2+ letters, else ([], 0): nothing worth a word is sung in the shot (a shot of only function words / vocables has no word)."""
    wp = R["wp"]
    runs = [r for r in WP.runs_in_span(wp, x, y, 0, R["seed"] + k, max_chars=CO.MAX_LINE_CHARS) if r[0] >= WP.MIN_SCORE]
    roomy = [r for r in runs if y - wp["times"][r[1][0]] >= SHOT_ROOM_S or y - x <= 2.0 * SHOT_ROOM_S]       # a word first sung in the last second of a long shot has no room to hold: the best one with room wins
    if roomy or runs:
        out, seen = [], []
        for s, idx in (roomy or runs) + [r for r in runs if r not in (roomy or [])]:
            if any(set(idx) <= set(q) or set(q) <= set(idx) for q in seen):
                continue
            seen.append(idx)
            out.append((idx, s))
            if len(out) >= top:
                break
        return out
    ws = WP.word_scores(wp)
    cand = [i for i in range(wp["n"]) if x - 1e-6 <= wp["times"][i] < y - 1e-6 and not WP.is_weak(wp["tokens"][i]) and WP.letters(wp["tokens"][i]) >= 2]
    return [([i], ws[i]) for i in sorted(cand, key=lambda q: (-ws[q], q))[:top]]


def _shot_hits(R: dict, notes: list[str]) -> tuple[list[dict], list[dict]]:
    """ONE WORD PER SHOT -> (hits [{text, land, hold_until, sung, shot}], text_select [{i, phrase, shot}]). Each shot's candidate = its best run (_shot_run); land = the sung time of the first word snapped
    forward to a beat within SNAP_S (never before it is sung), kept SHOT_LEAD_S inside the shot so the word forms between the cuts; hold_until = the last word sung + MIN_HOLD_S, never past the shot's end.
    A phrase that starts in a shot belongs to that shot. The shots that get a word are the largest set whose landings are MIN_WORD_CYCLE_S apart (ties: the higher total score); the others are named in notes."""
    vp, wp = R["vplan"], R["wp"]
    w0, cand = vp["w0"], []
    for k, ((a, b), (x, y)) in enumerate(zip(vp["times"], vp["pieces"]), 1):
        for run, sc in _shot_runs(R, k, x, y):
            s0, s1 = wp["times"][run[0]] - w0, wp["times"][run[-1]] - w0
            ahead = [t for t in vp["beats"] if s0 - 1e-6 <= t <= s0 + SNAP_S + 1e-6]
            land = min(ahead) if ahead else s0
            lead = min(SHOT_LEAD_S, 0.5 * (b - a))
            land = min(max(land, a + lead), max(b - SHOT_END_PAD_S, a + lead))
            hold = min(b, max(s1 + MIN_HOLD_S, land + MIN_HOLD_S))
            sc -= 1.0 if min(hold, b) - land < 0.5 else 0.0                    # a word that lands in the last half second of its shot has no time in it: a roomier idea of the same shot (or a neighbour) wins
            cand.append(dict(k=k, run=run, score=sc, land=land, hold=max(hold, land + 0.05), sung=wp["times"][run[0]], text=" ".join(_join_tokens(R, run).upper().split())))
    best: list = []                                           # best[j] = (count, score, chain of cand indices ending at j): one idea per shot, landings MIN_WORD_CYCLE_S apart, no word twice
    for j, c in enumerate(cand):
        top = (1, c["score"], [j])
        for i in range(j):
            pi, ci = best[i], cand[i]
            if ci["k"] < c["k"] and c["land"] >= ci["land"] + MIN_WORD_CYCLE_S - 1e-6 and not (set(ci["run"]) & set(c["run"])):
                t = (pi[0] + 1, pi[1] + c["score"], pi[2] + [j])
                if t[:2] > top[:2]:
                    top = t
        best.append(top)
    chain = max(best, key=lambda t: t[:2])[2] if best else []
    hits, sel, got = [], [], set()
    for j in chain:
        c = cand[j]
        got.add(c["k"])
        hits.append(dict(text=c["text"], land=round(c["land"], 3), hold_until=round(c["hold"], 3), sung=round(c["sung"], 3), shot=c["k"]))
        sel += [dict(i=i, phrase=c["k"], shot=c["k"]) for i in c["run"]]
    for k in sorted({c["k"] for c in cand} - got):
        c = next(c for c in cand if c["k"] == k)
        notes.append(f"shot {k}: {c['text']!r} would land too soon after the neighbouring word (the type engine needs {MIN_WORD_CYCLE_S:g} s per word): no word on this shot")
    if hits:
        notes.append(f"one word per shot: {len(hits)} of {len(vp['times'])} shots have words (the best-scoring word sung in each shot; shots with nothing sung have none)")
    return hits, sel


def _assemble_video(R: dict, title, logline, look, music, beats: list[str], gen_lines: list, notes: list[str], picks=()) -> dict:
    n, slots, vp = R["n"], R["slots"], R["vplan"]
    nw = len(slots)
    notes = notes + R["req_notes"]
    mode = _text_mode(R)
    sel, texts = None, []
    shot_text, hits_by_shot, auto_shot = {}, [], False
    if R["select"] and R["select_shot"]:                      # v18 SHOT-FOCUSED WORDS: the user put the words on shots; no time-based placement
        for p, idx in R["select"].items():
            t = " ".join(_join_tokens(R, idx).upper().split())
            if not 1 <= p <= n:
                notes.append(f"shot {p} does not exist ({n} cuts in {R['length']:g} s): those words are ignored")
            elif t:
                shot_text[p] = t
        if any(len(t.split()) > CO.LOCKUP_WORDS for t in shot_text.values()):
            notes.append(f"your word choice has a shot with more than {CO.LOCKUP_WORDS} words: the render may shorten it")
        sel = [dict(i=i, phrase=p, shot=p) for p, idx in R["select"].items() if 1 <= p <= n for i in idx]
        taken = set()
        for k, t in sorted(shot_text.items()):                # the hit lands on the render's own word slot inside the shot when there is one, else just after the shot starts
            a, b = vp["times"][k - 1]
            j = next((j for j, sl in enumerate(slots) if j not in taken and a - 1e-6 <= sl["land"] < b - 1e-6), None)
            if j is not None:
                taken.add(j)
                land, hold = float(slots[j]["land"]), min(b, float(slots[j]["hold_until"]))
            else:
                land = a + min(0.3, 0.1 * (b - a))
                hold = min(b, land + 1.5)
            h = dict(text=t, land=round(land, 3), hold_until=round(max(hold, land), 3))
            if R["text_src"] == "lyrics" and R["tok_t"]:
                ts = [R["tok_t"][i] for i in R["select"][k] if i < len(R["tok_t"])]
                if ts:
                    h["sung"] = min(ts)
                    s0, s1 = min(ts) - vp["w0"], max(ts) - vp["w0"]
                    if a - 1e-6 <= s0 < b - 1e-6 and s0 <= R["length"] - 1.0:      # sung inside this shot: the words land when they are sung
                        land2 = _land_for(R, s0)
                        land2 = land2 if land2 < b else round(s0, 3)
                        h.update(land=land2, hold_until=round(max(s1 + MIN_HOLD_S, land2 + MIN_HOLD_S), 3))
            hits_by_shot.append(h)
    elif _auto_shot_ok(R):                                    # ONE WORD PER SHOT (user 2026-11-02): every shot with sung lyrics, no slot cap
        hits_by_shot, sel = _shot_hits(R, notes)
        auto_shot = True
    elif not nw:
        notes.append("no word slot fits the window: no on-screen words")
    elif mode == "fixed":
        lines = _fixed_lines(R)
        if R["select"]:
            extra = [p for p in R["select"] if p > nw]
            if extra:
                notes.append(f"phrase{'s' if len(extra) > 1 else ''} {', '.join(map(str, extra))} won't fit: {nw} word slots in {R['length']:g} s")
            sel = [dict(i=i, phrase=p) for p, idx in R["select"].items() if p <= nw for i in idx]
        if len(lines) > nw:
            notes.append(f"{len(lines)} on-screen lines but only {nw} word slot{'s' if nw != 1 else ''} fit {R['length']:g} s on the beats: the last {len(lines) - nw} are dropped (the render does the same)")
        texts = lines[:nw]
        if R["tokens"] and sel is None:
            sel = _text_select_for_lines(R, texts, notes)
    elif mode == "pick":
        w0 = vp["w0"]
        spans = [(w0 + s["land"] - 1.5, w0 + s["hold_until"] + 1.5) for s in slots]
        texts, sel = _text_lines(R, picks, notes, nw, spans if R["tok_t"] else None)
    else:
        raw = list(gen_lines)[:nw] + [""] * (nw - len(list(gen_lines)[:nw]))
        try:
            from tools.app import own_video_v7 as V7
            parsed = V7.parse_words(json.dumps({"lines": [dict(slot=j, text=str(t or "")) for j, t in enumerate(raw, 1)]}), [dict(s, section="track") for s in slots])
            raw = [x["text"] for x in parsed]
        except (SystemExit, ValueError, KeyError, TypeError, IndexError):
            pass
        prev, used = "", set()
        for j, t in enumerate(raw, 1):
            w, prob = _gen_line(t, prev)
            if prob:
                notes.append(f"slot {j}: {prob}")
            if not w or w in used:
                w = _rule_words(R, j, used)
                notes.append(f"slot {j}: no usable word in the reply, rule word {w}")
            texts.append(w)
            prev = w
            used.add(w)
    hits = list(hits_by_shot)
    sung_hits, land_of = ([], {}) if auto_shot else _sung_hits(R, texts, slots, sel, notes)
    hits += sung_hits
    shots = []
    shot_at = lambda t: next((k for k, (a, b) in enumerate(vp["times"], 1) if a - 1e-6 <= t < b - 1e-6), n)      # noqa: E731
    if not R["select_shot"] and sel and not auto_shot:       # v18: every text_select entry names the shot that carries it (phrase j = the hit's landing second: a sung phrase lands where it is sung)
        sel = [dict(d, shot=shot_at(land_of[d["phrase"]]) if land_of.get(d["phrase"]) is not None else None) for d in sel]
    for k, ((a, b), (x, y)) in enumerate(zip(vp["times"], vp["pieces"]), 1):
        inside = [shot_text[k]] if R["select_shot"] and k in shot_text else [] if R["select_shot"] else [h["text"] for h in hits if a - 1e-6 <= h["land"] < b - 1e-6 or (k == n and h["land"] >= b - 1e-6)]
        shots.append(dict(n=k, image="V1", src_start=x, src_end=y, start=a, end=b, beat="", line=" ".join(inside), camera=CAMERA_FILMED, font=R["fonts"].get(k)))
    out = dict(v=2, title=title, logline=logline, look=look, music=music, shots=shots, hits=hits, candidates=[], capacity=capacity(R),
               video=dict(duration=vp["duration"], window=[vp["w0"], vp["w1"]], length=R["length"], real_cuts=vp["has_cuts"]))
    tc = _text_candidates(R)
    if tc:
        out["text_candidates"] = tc
        out["text_select"] = sel or []
        alts = _word_alts(R, vp["pieces"])
        if alts:
            out["word_alts"] = alts                                                    # WORD PICK: the ranked other ideas per cut (lyrics sung inside the cut, source seconds)
    out["notes"] = " | ".join(n_ for n_ in notes if n_)
    return out


def _assemble(R: dict, title, logline, look, music, beats: list[str], gen_lines: list[str], notes: list[str], cams=(), picks=()) -> dict:
    if R["vmode"]:
        return _assemble_video(R, title, logline, look, music, beats, gen_lines, notes, picks)
    n, imgs = R["n"], R["used"]
    notes = notes + R["req_notes"]
    mode = _text_mode(R)
    sel = None
    if mode == "fixed":
        ulines = _lines_for(R)
        if R["select"]:
            extra = [p for p in R["select"] if p > n]
            if extra and R["select_shot"]:
                notes.append(f"shot{'s' if len(extra) > 1 else ''} {', '.join(map(str, extra))} {'do' if len(extra) > 1 else 'does'} not exist ({n} shots at {R['length']} s): those words are ignored")
            elif extra:
                notes.append(f"phrase{'s' if len(extra) > 1 else ''} {', '.join(map(str, extra))} won't fit: {n} shots at {R['length']} s")
            if any(len(x.split()) > CO.LOCKUP_WORDS for x in ulines):
                notes.append(f"your word choice has a phrase of more than {CO.LOCKUP_WORDS} words: the planner may shorten it")
            sel = [dict(i=i, phrase=p, shot=(p if p <= n else None)) for p, idx in R["select"].items() if not (R["select_shot"] and p > n) for i in idx]
        elif len(R["lines"]) > n:
            notes.append(f"{len(R['lines'])} on-screen lines for {n} shots: neighbouring lines share a shot (the planner's merge), order unchanged")
        if R["tokens"] and sel is None:
            sel = _text_select_for_lines(R, ulines, notes)
    elif mode == "pick":
        ulines, sel = _text_lines(R, picks, notes, spans=[(a, b) for a, b in shot_times(R["length"], n, R["bpm"])] if R["tok_t"] else None)
    else:
        ulines = None
    times = shot_times(R["length"], n, R["bpm"])
    if R["text_src"] == "lyrics" and R["tok_t"] and sel and ulines is not None and not R["select_shot"]:        # an audio file's lyrics: the phrase sits on the shot where it is sung (not when the user placed it by shot)
        ulines, sel = _by_sung_time(R, ulines, sel, times, notes)
    shots, prev, pcam = [], "", ""
    for k in range(1, n + 1):
        if ulines is not None:
            line = ulines[k - 1]
        else:
            line, prob = _gen_line(gen_lines[k - 1] if k <= len(gen_lines) else "", prev)
            if prob:
                notes.append(f"shot {k}: {prob}, no line")
        prev = line or prev
        cam = _camera(cams[k - 1] if k <= len(cams) else None, k, n, pcam, R["seed"])
        pcam = cam
        shots.append(dict(n=k, image=imgs[k - 1]["id"] if k <= len(imgs) else None, start=times[k - 1][0], end=times[k - 1][1], beat=beats[k - 1], line=line, camera=cam, font=R["fonts"].get(k)))
    x = _extra_note(R)
    if x:
        notes.append(x)
    out = dict(v=2, title=title, logline=logline, look=look, music=music, shots=shots, candidates=_candidates(R), capacity=capacity(R))
    if R["tokens"]:
        out["text_select"] = [dict(d, shot=d["shot"]) if "shot" in d else dict(d, shot=(d["phrase"] if d["phrase"] <= n else None)) for d in (sel or [])]          # v18: every entry names its shot
        out["text_candidates"] = _text_candidates(R)
        alts = _word_alts(R, times)
        if alts:
            out["word_alts"] = alts                                                    # WORD PICK: the ranked other ideas per shot (lyrics sung inside the shot)
    out["notes"] = " | ".join(n_ for n_ in notes if n_)
    return out


def fallback(req, why: str = "") -> dict:
    """the rule story: beats from media[].shows (shot k = picture k, then a carry-on of the last picture), the user's lines, no invented words."""
    R = norm_req(req)
    notes = ([why] if why else []) + ([] if R["images"] or R["video"] else ["no pictures to take beats from"])
    beats = [_fb_beat(R, k) for k in range(1, R["n"] + 1)]
    return _assemble(R, _fb_title(R), _fb_logline(R), _fb_look(R), _music_string("", R), beats, [], notes)


def parse(reply: str, req) -> dict:
    """the model reply + the request -> STORY JSON v1 with the guarantees of the module docstring. Never raises."""
    R = norm_req(req)
    obj, pnotes = (MR.parse_reply(reply) if isinstance(reply, str) and reply.strip() else (None, ["empty reply"]))
    raw = [s for s in (obj or {}).get("shots") or [] if isinstance(s, dict)] if isinstance(obj, dict) and isinstance(obj.get("shots"), list) else []
    if not raw and R["vmode"] and isinstance(obj, dict) and any(obj.get(k) for k in ("title", "logline", "look", "music", "hits")):
        pass                                                                          # v18 video mode: the reply has no per-shot part (no beats, no transitions)
    elif not raw:
        return fallback(req, "MODEL REPLY NOT USABLE (" + "; ".join(pnotes or ["no shots in the reply"]) + "): beats come from the pictures' descriptions")
    n = R["n"]
    ns = [s.get("n") for s in raw]
    if len(raw) >= n and sorted(x for x in ns if isinstance(x, int)) == list(range(1, len(raw) + 1)):
        raw = sorted(raw, key=lambda s: s["n"])                       # the model numbered its shots: that order
    notes = (["reply: " + "; ".join(pnotes)] if pnotes else [])
    if len(raw) != n and not R["vmode"]:
        notes.append(f"the reply had {len(raw)} shots, the piece has {n}" + (": the missing beats come from the pictures" if len(raw) < n else ": the extra ones are dropped"))
    beats, gen, cams, picks = [], [], [], []
    for k in range(1, n + 1):
        s = raw[k - 1] if k <= len(raw) else {}
        b = _ws(s.get("beat"), BEAT_CHARS)
        if not b and R["vmode"]:
            pass                                                                      # video mode: the cuts are not described (the footage is kept as is)
        elif not b:
            b = _fb_beat(R, k)
            if k <= len(raw):
                notes.append(f"shot {k}: no beat in the reply, taken from the picture")
        beats.append(b)
        gen.append(s.get("line"))
        cams.append(s.get("camera"))
        picks.append(s.get("pick"))
    if R["vmode"]:                                                                    # the words are per word slot (`hits`), not per shot
        hs = [h for h in (obj.get("hits") if isinstance(obj.get("hits"), list) else []) if isinstance(h, dict)]
        picks, gen = [], []
        for j in range(1, len(R["slots"]) + 1):
            h = next((x for x in hs if str(x.get("slot")).strip() == str(j)), None) or (hs[j - 1] if j - 1 < len(hs) else {})
            picks.append(h.get("pick"))
            gen.append(h.get("text"))
    title = " ".join(_ws(obj.get("title"), 60).split()[:TITLE_WORDS]).strip(" .,;:") or _fb_title(R)
    read = R["read"]
    logline = _ws(read.get("concept"), LOGLINE_CHARS) or _ws(obj.get("logline"), LOGLINE_CHARS) or _fb_logline(R)      # the user's concept wins
    look = _ws(read.get("look"), LOOK_CHARS) or _ws(obj.get("look"), LOOK_CHARS) or _fb_look(R)                         # the user's look wins
    music = _music_string(obj.get("music"), R)
    if _ws(obj.get("notes"), 200):
        notes.insert(0, _ws(obj["notes"], 200))
    return _assemble(R, title, logline, look, music, beats, gen, notes, cams, picks)
