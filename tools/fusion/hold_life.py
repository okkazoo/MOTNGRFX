"""hold_life: effects that keep a LANDED word alive during its hold while it stays legible and centred (user 2026-10-30: "when the text resolves it shouldnt be static it should have some
glitch or pulse or glow or something to keep it alive - it should be legible for a beat but in that beat still be active or subtly glitching or each letter is glitching on and off").

The word never moves (stable_type rule 1: centred, no camera move): the box centre is fixed, the word lands on the beat and reads for >= 1 beat. Life is added on the DRAWN type layer only
(never to LTX-rendered words, never to the footage), in the hold window [hit, release] of each word. The menu (`EFFECTS`), 1-2 drawn per piece (`draw_effects`, seeded):

    letter_flicker   individual letters blink off / dim for 1-2 frames, a few per beat, on sub-beats (off-beat first)
    glitch_slice     2-4 frame bursts of horizontal slice offsets + a 1-3 px RGB split, on sub-beats
    pulse            scale 1.00 -> 1.03 -> 1.00 + a brightness lift on the beat, eased, about the word's box centre (the centre never moves)
    glow_breathe     the type glow follows the beat envelope (and the music's rms curve when the piece has one)
    outline_shimmer  the stand-out outline (random_system.draw_outline) breathes in brightness (its windows get short dips); needs an outline in the piece

LEGIBILITY GUARD (hard; `GUARD`, enforced when planning AND again in the op, tested in tests/test_hold_life.py):
    * every frame of a hold: >= 85 % of the word's letters fully visible (so at most floor(0.15 n) letters off / dim at once; a word with n < 7 letters has NO letter flicker)
    * a letter is off / dim for at most 2 frames at 24 fps (`max_off_frames(fps)` scales with the fps)
    * never two ADJACENT letters off in the same frame
    * displacements (slice + RGB split, pulse growth) <= 1.5 % of the frame width
    * the first READ_FRAMES (3) frames after the landing are clean (no dropout, no offset, scale 1, no lift), and the dropout class (flicker, slices, split) never touches the frames on
      a beat (+-1 frame); only the pulse sits ON the beat, and it is a scale <= 1.03 about the box centre (every letter stays fully visible).

HOW IT GETS INTO A PIECE: the planner calls ONE function, `weave(piece, music, chk, size)` (see 16_app/HOLD_LIFE.md for the exact line). It writes the plan into `piece["plan"]["hold_life"]`
and puts compose op `hold_life` (registered through OPS in compose._ops_registry) in front of every type drawer (sandwich / depth_occlude / knockout) of every compose chain, plus an
`add_glow` after the last drawer for glow_breathe. recipe_eval.legibility_pass twins the op onto the beauty face (like `translate`): the alpha twin gets the dropouts / slices / scale, the
face twin the same warp + the RGB split + the brightness lift, both from the SAME reference alpha, so face and alpha always agree.

LETTER MASKS: the type passes (type_passes.py) expose NO per-glyph alpha (beauty / matte / depth / line / clay / flow only), so letters are CONNECTED COMPONENTS of the type alpha (`letter_map`):
core = alpha > 0.35, every other drawn pixel takes the nearest component, small marks (dot of an i / j, accents) join the letter they sit over, then they are sorted into reading order (rows
top to bottom, left to right; a word that reads vertically, a rolled or column layout, is sorted along y). The plan names letters by their index in the text (n = non-space characters); when the
component count differs from n (touching letters merge) the index is mapped proportionally and the guard is enforced again on the mapped set (`enforce_guard`).

Pure planning (`plan`, `frame_state`, `check_plan`) has no numpy-array dependency; the op uses the compose backend (torch on CUDA, numpy / cv2 fallback)."""
from __future__ import annotations

import copy
import math
import random

import numpy as np

from tools.fusion import compose as C

EFFECTS = {
    "letter_flicker": "individual letters blink off / dim for 1-2 frames, a few per beat, on sub-beats",
    "glitch_slice": "2-4 frame bursts of horizontal slice offsets + a 1-3 px RGB split on sub-beats",
    "pulse": "scale 1.00 -> 1.03 -> 1.00 and a brightness lift on each beat, about the box centre",
    "glow_breathe": "the type glow follows the beat envelope (and the music's rms curve)",
    "outline_shimmer": "the stand-out outline breathes in brightness (needs an outline in the piece)",
}
WEIGHTS = {"letter_flicker": 3.0, "glitch_slice": 2.5, "pulse": 3.0, "glow_breathe": 0.7, "outline_shimmer": 1.5}
NO_PAIR = {frozenset(("letter_flicker", "glitch_slice"))}          # too much dropout in one word: never drawn together

GUARD = dict(min_visible=0.85,            # share of a word's letters fully visible at every frame of a hold
             max_off_s=2.0 / 24.0,        # a letter is off / dim at most this long (2 frames at 24 fps)
             read_frames=3,               # clean frames after the landing
             beat_clean=1,                # dropouts keep +-this many frames away from a beat frame
             max_disp=0.015,              # displacement <= this x frame width
             pulse_max=0.03,              # scale growth
             min_letters=7)               # floor(0.15 n) >= 1 needs n >= 7
SLICE_DX = (0.003, 0.009)                 # slice offset, share of the frame width (x intensity)
SPLIT_W = (0.0019, 0.0055)                # RGB split per channel, share of the frame width (1-3 px at 540 px wide)
DIM_GAIN = 0.3
FALL_BEATS = 0.4                          # pulse decay (share of a beat)
RISE_S = 2.0 / 24.0                       # pulse attack (ends ON the beat frame)


# ---------------------------------------------------------------------------------------------------------------------------------------------- guard numbers
def max_off_frames(fps) -> int:
    """most frames a letter may stay off / dim in a row: 2 at 24 fps, scaled with the fps (at least 1)."""
    return max(1, int(GUARD["max_off_s"] * float(fps) + 1e-6))


def budget(n) -> int:
    """letters that may be off / dim in the same frame: floor((1 - 0.85) n)."""
    return int((1.0 - GUARD["min_visible"]) * int(n) + 1e-9)


def n_letters(text) -> int:
    return sum(1 for c in str(text) if not c.isspace())


def enforce_guard(off: dict, m: int) -> dict:
    """{reading index: gain < 1} -> the subset that keeps the guard for a word of m letters: at most budget(m), never two adjacent. Deterministic (ascending index)."""
    keep: dict = {}
    for r in sorted(off):
        if len(keep) >= budget(m):
            break
        if (r - 1) in keep or (r + 1) in keep:
            continue
        keep[r] = off[r]
    return keep


def map_letters(gains: dict, n: int, m: int) -> dict:
    """plan letter index (of n) -> reading index of m components, then the guard again on the mapped set."""
    if m <= 0:
        return {}
    out: dict = {}
    for L, g in gains.items():
        r = int(L) if m == n else min(m - 1, int((int(L) + 0.5) * m / max(n, 1)))
        out[r] = min(out.get(r, 1.0), g)
    return enforce_guard(out, m)


# ---------------------------------------------------------------------------------------------------------------------------------------------- the draw
def draw_effects(seed, texts=(), has_outline=False, requested=None) -> tuple:
    """1-2 effects of the menu for the piece, seeded. letter_flicker needs a word of >= 7 letters, outline_shimmer an outline in the piece; letter_flicker + glitch_slice are never paired.
    `requested` (names) forces exactly those in (feasible ones, at most 2). -> (list of names, {skipped name: why})."""
    why: dict = {}
    cand = dict(WEIGHTS)
    if not any(n_letters(t) >= GUARD["min_letters"] for t in texts):
        cand.pop("letter_flicker")
        why["letter_flicker"] = f"no word has {GUARD['min_letters']}+ letters (one off letter of 6 is 83 % < 85 % visible)"
    if not has_outline:
        cand.pop("outline_shimmer")
        why["outline_shimmer"] = "the piece has no outline"
    rng = random.Random(f"holdlife:{int(seed or 0)}:draw")
    req = [r for r in dict.fromkeys(requested or []) if r in EFFECTS]
    picked = [r for r in req if r in cand][:2]
    for r in req:
        if r not in cand:
            why[r] = why.get(r, "not feasible")
    want = 1 if rng.random() < 0.45 else 2
    forced = bool(picked)                                  # a request is honoured as asked: no random extra on top
    while not forced and len(picked) < want and cand:
        pool = [k for k in cand if k not in picked and not any(frozenset((k, p)) in NO_PAIR for p in picked)]
        if not pool:
            break
        tot = sum(cand[k] for k in pool)
        r = rng.random() * tot
        for k in pool:
            r -= cand[k]
            if r <= 0:
                break
        picked.append(k)
    return picked, why


# ---------------------------------------------------------------------------------------------------------------------------------------------- timing helpers
def _smooth(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def beat_times(beats, bpm, hit, release):
    """beat times (s) inside [hit - 1 beat, release + 1 beat]: the piece's beat list when given, else a grid anchored on the hit (a word lands on a beat)."""
    iv = 60.0 / float(bpm) if bpm else 0.5
    if beats:
        b = sorted(float(x) for x in beats)
        out = [x for x in b if hit - iv * 1.05 <= x <= release + iv * 1.05]
        if len(out) >= 2:
            return out
    k0, k1 = -1, int((release - hit) / iv) + 2
    return [hit + k * iv for k in range(k0, k1 + 1)]


def pulse_env(t, beats_t, rise_s=RISE_S, fall_s=0.2):
    """0..1 envelope of the beat pulse at time t: smooth attack that ends ON the beat, smooth decay after it; the max over the beats."""
    e = 0.0
    for tb in beats_t:
        if tb - rise_s <= t <= tb:
            e = max(e, _smooth((t - (tb - rise_s)) / rise_s))
        elif tb < t < tb + fall_s:
            e = max(e, 1.0 - _smooth((t - tb) / fall_s))
    return e


def slice_state(seed, f, amp=1.0):
    """pure function of (burst seed, frame): ([(y0, y1, dx)], split) with y in 0..1 of the word box height, dx and split in shares of the frame width (dx signed, + = right)."""
    r = random.Random(int(seed) * 7919 + int(f) * 104729)
    bands = []
    for _ in range(1 + int(r.random() * 3)):
        y0 = r.random() * 0.85
        h = 0.06 + 0.14 * r.random()
        dx = (SLICE_DX[0] + (SLICE_DX[1] - SLICE_DX[0]) * r.random()) * amp * (1 if r.random() < 0.5 else -1)
        bands.append((round(y0, 4), round(min(y0 + h, 1.0), 4), round(dx, 5)))
    split = (SPLIT_W[0] + (SPLIT_W[1] - SPLIT_W[0]) * r.random()) * amp
    return bands, round(split, 5)


# ---------------------------------------------------------------------------------------------------------------------------------------------- plan
def _clean_frame(f, wp_on, wp_off, beat_fs):
    """a frame where a dropout / slice may be: inside [f_on, f_off), not within beat_clean frames of a beat frame."""
    if f < wp_on or f >= wp_off:
        return False
    return all(abs(f - b) > GUARD["beat_clean"] for b in beat_fs)


def _slots(beats_t, hit, release):
    """sub-beat candidate times: (beat k + 0.5 beat) first, then the 16ths, per beat interval inside the hold."""
    out = []
    for a, b in zip(beats_t[:-1], beats_t[1:]):
        if b <= hit - 1e-6 or a >= release:
            continue
        iv = b - a
        out.append([a + 0.5 * iv, a + 0.25 * iv, a + 0.75 * iv])
    return out


def plan_word(i, text, hit, release, effects, seed, fps, beats_t, intensity=None):
    """the plan of one word: events per effect, frames are PIECE frames. See the module doc for the guard."""
    rng = random.Random(f"holdlife:{int(seed or 0)}:word:{i}:{text}")
    inten = round(0.6 + 0.4 * rng.random(), 3) if intensity is None else float(intensity)
    fps = float(fps)
    hit_f, rel_f = int(round(hit * fps)), int(round(release * fps))
    f_on = hit_f + GUARD["read_frames"]
    f_off = rel_f
    beat_fs = sorted({int(round(b * fps)) for b in beats_t})
    n = n_letters(text)
    wp = dict(i=i, n=n, hit_f=hit_f, release_f=rel_f, f_on=f_on, f_off=f_off, intensity=inten, beat_frames=beat_fs, flicker=[], slices=[], pulse=None, slice_amp=round(0.6 + 0.4 * inten, 3), skipped=[])
    if f_off - f_on < 4:
        wp["skipped"].append("hold too short for hold life")
        return wp
    slots = _slots(beats_t, hit, release)
    mo = max_off_frames(fps)
    if "letter_flicker" in effects:
        bud = budget(n)
        if bud < 1:
            wp["skipped"].append(f"letter_flicker: {n} letters (< {GUARD['min_letters']})")
        else:
            occ: dict = {}
            recent: list = []

            def fits(L, f0, dur):
                for f in range(f0, f0 + dur):
                    if not _clean_frame(f, f_on, f_off, beat_fs):
                        return False
                    s = occ.get(f, set())
                    if len(s) >= bud or (L - 1) in s or (L + 1) in s or L in s:
                        return False
                return True

            def put(L, f0, dur, g):
                for f in range(f0, f0 + dur):
                    occ.setdefault(f, set()).add(L)
                wp["flicker"].append([f0, L, dur, g])

            for cand in slots:
                k = 1 + (1 if rng.random() < 0.55 * inten else 0)
                order = [cand[0]] + rng.sample(cand[1:], len(cand) - 1)
                done = 0
                for tc in order:
                    if done >= k:
                        break
                    f0 = int(round(tc * fps))
                    dur = 1 if rng.random() < 0.55 else min(2, mo)
                    g = 0.0 if rng.random() < 0.65 else DIM_GAIN
                    pool = [L for L in range(n) if L not in recent[-3:]] or list(range(n))
                    rng.shuffle(pool)
                    for L in pool:
                        if fits(L, f0, dur):
                            put(L, f0, dur, g)
                            recent.append(L)
                            done += 1
                            if rng.random() < 0.25 and dur == 1 and fits(L, f0 + 2, 1):       # stutter: off, on, off
                                put(L, f0 + 2, 1, g)
                            break
            wp["flicker"].sort()
    if "glitch_slice" in effects:
        last_end = -99
        for cand in slots:
            if rng.random() > 0.35 + 0.5 * inten:
                continue
            order = [cand[0]] + rng.sample(cand[1:], len(cand) - 1)
            dur = rng.choice((2, 3, 4)) if rng.random() < 0.85 else 2
            for tc in order:
                f0 = int(round(tc * fps))
                if f0 >= last_end + 2 and all(_clean_frame(f, f_on, f_off, beat_fs) for f in range(f0, f0 + dur)):
                    wp["slices"].append([f0, dur, rng.getrandbits(30)])
                    last_end = f0 + dur
                    break
    if "pulse" in effects:
        iv = float(np.median(np.diff(beats_t))) if len(beats_t) > 1 else 0.5
        fall_s = max(FALL_BEATS * iv, 0.15)
        cand_t = [b for b in beats_t if b > hit + 1e-6]
        ok = [b for b in cand_t if int(round((b - RISE_S) * fps)) >= f_on and int(round((b + fall_s) * fps)) <= f_off]
        if not ok:
            half = hit + 0.5 * iv                                              # a one-beat hold: pulse on the off-beat
            if int(round((half - RISE_S) * fps)) >= f_on and int(round((half + fall_s) * fps)) <= f_off:
                ok = [half]
        if ok:
            wp["pulse"] = dict(beats=[round(b, 4) for b in ok], amp=round(0.012 + 0.018 * inten, 4), lift=round(0.05 + 0.10 * inten, 4), rise=round(RISE_S, 4), fall=round(fall_s, 4))
        else:
            wp["skipped"].append("pulse: no beat fits the hold")
    return wp


PER_WORD = ("letter_flicker", "glitch_slice", "pulse")        # the hold-life effects that play inside one word's hold (glow_breathe / outline_shimmer are piece-wide)


def word_kinds(seed, effects, n) -> list:
    """the per-word effect set (the word's hold-life KIND): when the piece drew 2+ per-word effects, consecutive words never carry the same set (VARIETY RULE 2026-10-31, like the cuts); a
    set is any non-empty subset of the drawn per-word effects, the fuller ones likelier. One drawn effect = nothing to vary (every word carries it)."""
    we = [e for e in effects if e in PER_WORD]
    if len(we) < 2:
        return [tuple(we) for _ in range(n)]
    opts = [tuple(e for j, e in enumerate(we) if m >> j & 1) for m in range(1, 1 << len(we))]
    out: list = []
    for i in range(n):
        rng = random.Random(f"holdlife:{int(seed or 0)}:kinds:{i}")
        cand = [o for o in opts if not out or o != out[-1]]
        wts = [float(len(o)) for o in cand]
        r = rng.random() * sum(wts)
        pick = cand[-1]
        for o, x in zip(cand, wts):
            r -= x
            if r <= 0:
                pick = o
                break
        out.append(pick)
    return out


def plan(words, piece, seed, fps=None, beats=None, bpm=None, effects=None, requested=None, curves=None, outline=None, size=None) -> dict:
    """the hold-life plan of a piece (JSON-able). words = [{text, hit, release}] (seconds; `rel` accepted) of the words that compose draws FLAT. The piece supplies fps / bpm / beats /
    outline windows unless given. -> dict(v, seed, fps, effects, skipped, words, glow, outline)."""
    t = (piece or {}).get("type") or {}
    fps = int(fps or (piece or {}).get("fps") or 24)
    bpm = bpm or t.get("bpm") or ((piece or {}).get("audio") or {}).get("bpm")
    beats = beats if beats is not None else t.get("beats")
    texts = [str(w.get("text", "")) for w in words]
    out_w = [list(o) for o in (outline or [])]
    if effects is None:
        effects, why = draw_effects(seed, texts, bool(out_w), requested)
    else:
        effects, why = [e for e in effects if e in EFFECTS], {}
    ws = []
    kinds = word_kinds(seed, effects, len(words))
    for i, w in enumerate(words):
        hit = float(w["hit"])
        rel = float(w.get("release", w.get("rel", hit)))
        bt = beat_times(beats, bpm, hit, rel)
        wp_ = plan_word(i, w.get("text", ""), hit, rel, [e for e in effects if e not in PER_WORD or e in kinds[i]], seed, fps, bt)
        wp_["kinds"] = list(kinds[i])
        ws.append(wp_)
    p = dict(v=1, seed=int(seed or 0), fps=fps, bpm=bpm, effects=list(effects), skipped=why, words=ws, glow=None, outline=None)
    if "glow_breathe" in effects:
        p["glow"] = glow_plan(ws, words, fps, bpm, beats, curves, size)
    if "outline_shimmer" in effects:
        p["outline"] = shimmer_windows(out_w, bpm, seed)
    return p


def glow_plan(wps, words, fps, bpm, beats, curves=None, size=None) -> dict:
    """the glow envelope: timeline keys [[t s, intensity]] (0 outside the holds, ramps 0.2 s), radius in px (follows the music's rms curve when the piece has one)."""
    W = float((size or (540, 960))[0])
    keys = []
    for wp, w in zip(wps, words):
        a, b = wp["f_on"] / fps, wp["f_off"] / fps
        if b - a < 0.2:
            continue
        hit = float(w["hit"])
        bt = beat_times(beats, bpm, hit, float(w.get("release", w.get("rel", hit))))
        iv = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
        lo, hi = 0.10 * wp["intensity"] + 0.04, 0.34 * wp["intensity"] + 0.08
        keys.append([round(a - 0.2, 4), 0.0])
        n = int(round((b - a) * fps))
        for k in range(n + 1):
            t = a + k / fps
            e = 0.0
            for tb in bt:
                if tb - 0.1 <= t <= tb:
                    e = max(e, _smooth((t - (tb - 0.1)) / 0.1))
                elif t > tb:
                    e = max(e, math.exp(-(t - tb) / (0.45 * iv)))
            ramp = max(0.0, min(1.0, (t - a) / 0.2, (b - t) / 0.2))
            keys.append([round(t, 4), round((lo + (hi - lo) * e) * ramp, 4)])
        keys.append([round(b + 0.2, 4), 0.0])
    keys = _mono(keys)
    music = None
    for c in ("rms", "onset"):
        if curves and isinstance(curves.get(c), list):
            music = c
            break
    return dict(keys=keys, radius=[round(0.012 * W, 2), round(0.030 * W, 2)], curve=music)


def _mono(keys):
    """timeline keys sorted by time with duplicates collapsed (the later key wins)."""
    out: dict = {}
    for t, v in keys:
        out[round(float(t), 4)] = float(v)
    return [[t, out[t]] for t in sorted(out)]


def shimmer_windows(windows, bpm, seed) -> list:
    """outline windows [[t0, t1]] -> the same windows with short dips (2-3 frames). legible_color eases the outline in and out (LC_TAU_SUP 0.15 s), so a dip is a brightness breath to
    ~40-55 %, not a blink. Dips sit on off-beats, never in the first 0.3 s or the last 0.12 s of a window (a 2-beat hold at 120 bpm gets one dip, at 0.75 s). Width stays legible_color's."""
    rng = random.Random(f"holdlife:{int(seed or 0)}:shimmer")
    iv = 60.0 / float(bpm) if bpm else 0.5
    out = []
    for a, b in windows:
        a, b = float(a), float(b)
        dips, feas, j = [], [], 0
        while True:
            tc = a + (j + 0.5) * iv
            gap = (2.0 + rng.random()) / 24.0
            if tc + gap > b - 0.12:
                break
            if tc >= a + 0.3:
                feas.append((tc, tc + gap))
                if rng.random() < 0.75:
                    dips.append((tc, tc + gap))
            j += 1
        if not dips and feas:                              # a window long enough for a dip always gets one
            dips.append(feas[0])
        cur = a
        for d0, d1 in dips:
            out.append([round(cur, 4), round(d0, 4)])
            cur = d1
        out.append([round(cur, 4), round(b, 4)])
    return [w for w in out if w[1] > w[0] + 1e-6]


# ---------------------------------------------------------------------------------------------------------------------------------------------- state of one frame
def word_index(plan_, f):
    """index + plan of the word whose hold-life window contains piece frame f, else (None, None)."""
    for k, wp in enumerate((plan_ or {}).get("words") or []):
        if wp["f_on"] <= f < wp["f_off"]:
            return k, wp
    return None, None


def frame_state(wp, f, fps) -> dict:
    """what is done to the word at piece frame f: gains {plan letter: gain}, bands [(y0, y1, dx)], split, scale, lift. Pure; the op and the tests both read it. Outside [f_on, f_off) (the read
    frames after the landing included) the state is the identity."""
    st = dict(gains={}, bands=[], split=0.0, scale=1.0, lift=0.0)
    if wp is None or not (wp["f_on"] <= f < wp["f_off"]):
        return st
    for f0, L, dur, g in wp.get("flicker") or []:
        if f0 <= f < f0 + dur:
            st["gains"][int(L)] = min(st["gains"].get(int(L), 1.0), float(g))
    for f0, dur, sd in wp.get("slices") or []:
        if f0 <= f < f0 + dur:
            st["bands"], st["split"] = slice_state(sd, f, wp.get("slice_amp", 1.0))
    pu = wp.get("pulse")
    if pu:
        e = pulse_env(f / float(fps), pu["beats"], pu.get("rise", RISE_S), pu["fall"])
        st["scale"] = 1.0 + min(float(pu["amp"]), GUARD["pulse_max"]) * e
        st["lift"] = float(pu["lift"]) * e
    return st


def is_identity(st) -> bool:
    return not st["gains"] and not st["bands"] and st["split"] == 0.0 and abs(st["scale"] - 1.0) < 1e-5 and st["lift"] < 1e-5


def displacement(st, half_w=0.5) -> float:
    """largest displacement of the frame state as a share of the frame width: slice offset + RGB split, or the pulse growth at `half_w` (the word's half width as a share of the frame width)."""
    sl = max([abs(b[2]) for b in st["bands"]] or [0.0]) + st["split"]
    return max(sl, (st["scale"] - 1.0) * half_w)


def check_plan(plan_) -> dict:
    """the guard numbers of a plan, frame by frame. -> dict(min_visible, max_off_run, adjacent, read_dirty, beat_dirty, max_disp, over_budget) (each violation count 0 = clean)."""
    fps = float(plan_["fps"])
    res = dict(min_visible=1.0, max_off_run=0, adjacent=0, read_dirty=0, beat_dirty=0, max_disp=0.0, over_budget=0, max_scale=1.0)
    for wp in plan_["words"]:
        n = wp["n"]
        run: dict = {}
        for f in range(wp["hit_f"], wp["release_f"] + 1):
            st = frame_state(wp, f, fps)
            offs = {L for L, g in st["gains"].items() if g < 1.0}
            if n:
                res["min_visible"] = min(res["min_visible"], 1.0 - len(offs) / n)
            if len(offs) > budget(n):
                res["over_budget"] += 1
            res["adjacent"] += sum(1 for L in offs if (L + 1) in offs)
            for L in list(run):
                if L not in offs:
                    run.pop(L)
            for L in offs:
                run[L] = run.get(L, 0) + 1
                res["max_off_run"] = max(res["max_off_run"], run[L])
            if wp["hit_f"] <= f < wp["hit_f"] + GUARD["read_frames"] and not is_identity(st):
                res["read_dirty"] += 1
            res["max_disp"] = max(res["max_disp"], displacement(st))
            res["max_scale"] = max(res["max_scale"], st["scale"])
    res["beat_dirty"] = beat_dirty(plan_)
    return res


def beat_dirty(plan_) -> int:
    """frames on a beat (+-1) of a hold that carry a dropout / slice / split: must be 0."""
    fps = float(plan_["fps"])
    bad = 0
    for wp in plan_["words"]:
        for bf in wp.get("beat_frames") or []:
            for f in range(bf - GUARD["beat_clean"], bf + GUARD["beat_clean"] + 1):
                if wp["f_on"] <= f < wp["f_off"]:
                    st = frame_state(wp, f, fps)
                    if st["gains"] or st["bands"] or st["split"]:
                        bad += 1
    return bad


# ---------------------------------------------------------------------------------------------------------------------------------------------- letter masks
def letter_map(alpha, thr=0.35, tiny=0.03, mark_area=0.45, mark_overlap=0.6) -> dict:
    """letters of a type alpha (HxW float, numpy) = connected components, in reading order. -> dict(rmap int16 HxW (reading index, m = no letter), m, boxes [(x0, y0, x1, y1)], bbox, vertical).
    Small marks (extent across the reading axis < 45 % of the neighbour's: the dot of i / j, accents) are merged into the letter they overlap along the reading axis."""
    import cv2
    a = np.asarray(alpha, np.float32)
    h, w = a.shape[:2]
    core = (a > thr).astype(np.uint8)
    n, cc, stats, cent = cv2.connectedComponentsWithStats(core, connectivity=8)
    none = dict(rmap=np.zeros((h, w), np.int16), m=0, boxes=[], bbox=None, vertical=False)
    if n <= 1:
        return none
    areas = stats[1:, 4].astype(np.float64)
    ids = [k + 1 for k in range(n - 1) if areas[k] >= tiny * areas.max()]
    ys, xs = np.nonzero(a > 0.1)
    if not len(xs):
        return none
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cxs = np.array([cent[k][0] for k in ids])
    cys = np.array([cent[k][1] for k in ids])
    vertical = len(ids) > 2 and float(cys.std()) > 1.5 * float(cxs.std())
    box = {k: (int(stats[k][0]), int(stats[k][1]), int(stats[k][0] + stats[k][2]), int(stats[k][1] + stats[k][3])) for k in ids}
    area = {k: float(stats[k][4]) for k in ids}
    parent = {k: k for k in ids}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    ax = (1, 3) if vertical else (0, 2)           # reading axis of the box: along y when the word reads vertically
    ox = (0, 2) if vertical else (1, 3)
    for p in ids:
        for q in ids:
            if p == q or (box[p][ox[1]] - box[p][ox[0]]) >= mark_area * (box[q][ox[1]] - box[q][ox[0]]) or area[p] >= area[q]:
                continue
            lo, hi = max(box[p][ax[0]], box[q][ax[0]]), min(box[p][ax[1]], box[q][ax[1]])
            small = box[p][ax[1]] - box[p][ax[0]]
            gap = max(box[p][ox[0]], box[q][ox[0]]) - min(box[p][ox[1]], box[q][ox[1]])
            if hi - lo >= mark_overlap * max(small, 1) and gap < 0.6 * (box[q][ox[1]] - box[q][ox[0]]):
                parent[find(p)] = find(q)
    roots = sorted({find(k) for k in ids})
    members = {r: [k for k in ids if find(k) == r] for r in roots}
    gbox = {}
    for r, ks in members.items():
        gbox[r] = (min(box[k][0] for k in ks), min(box[k][1] for k in ks), max(box[k][2] for k in ks), max(box[k][3] for k in ks))
    big = {r: max(members[r], key=lambda k: area[k]) for r in roots}
    gc = {r: (cent[big[r]][0], cent[big[r]][1]) for r in roots}
    if vertical:
        order = sorted(roots, key=lambda r: gc[r][1])
    else:
        hs = sorted(gbox[r][3] - gbox[r][1] for r in roots)
        med = hs[len(hs) // 2]
        by_y = sorted(roots, key=lambda r: gc[r][1])
        rows, cur = [], [by_y[0]]
        for r in by_y[1:]:
            if gc[r][1] - gc[cur[-1]][1] > 0.6 * med:
                rows.append(cur)
                cur = [r]
            else:
                cur.append(r)
        rows.append(cur)
        order = [r for row in rows for r in sorted(row, key=lambda r: gc[r][0])]
    m = len(order)
    seeds = np.full((h, w), 255, np.uint8)
    for k in ids:
        seeds[cc == k] = 0
    _, lbl = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
    lut = np.full(int(lbl.max()) + 1, m, np.int16)
    for ridx, r in enumerate(order):
        for k in members[r]:
            lut[np.unique(lbl[cc == k])] = ridx
    rmap = lut[lbl]
    rmap[a <= 0.004] = m
    return dict(rmap=rmap.astype(np.int16), m=m, boxes=[gbox[r] for r in order], bbox=bbox, vertical=bool(vertical))


_CACHE: dict = {}


def reset(key=None):
    """forget the cached letter maps (of one state key, or all)."""
    if key is None:
        _CACHE.clear()
    else:
        for k in [k for k in _CACHE if k[0] == key]:
            _CACHE.pop(k)


def _word_cache(state_key, k, f, a_np):
    key = (state_key, k)
    c = _CACHE.get(key)
    if c is None or f < c["last"] or c["shape"] != a_np.shape:
        c = letter_map(a_np)
        c["shape"] = a_np.shape
        _CACHE[key] = c
    c["last"] = f
    return c


# ---------------------------------------------------------------------------------------------------------------------------------------------- the compose op
@C.api
def hold_life(img, ref, t=0.0, frame_idx=None, plan=None, state_key="hl"):
    """compose op: the hold-life transform of the frame for the type layer `img` (an alpha HxW, or the face HxWx3) whose letters are read from `ref` (the type alpha, HxW). `plan` = the piece's
    hold-life plan (`plan()`); `frame_idx` = piece frame. Outside a word's hold-life window (landing frames, exits, other words) the image comes back untouched. The alpha twin returns the alpha
    after dropouts / slices / split (max of the three channel alphas) / pulse scale; the face twin returns the face after the same warp, the RGB split and the brightness lift."""
    B = C.B
    if not plan:
        return img
    fps = float(plan.get("fps", 24))
    f = int(frame_idx) if frame_idx is not None else int(round(float(t) * fps))
    k, wp = word_index(plan, f)
    if wp is None:
        return img
    st = frame_state(wp, f, fps)
    if is_identity(st):
        return img
    face = img.ndim == 3 and img.shape[-1] >= 3
    a_ref = C._gray(ref)
    h, w = int(a_ref.shape[0]), int(a_ref.shape[1])
    a_np = np.asarray(B.tonumpy(a_ref), np.float32)
    lm = _word_cache(state_key, k, f, a_np)
    if lm["m"] == 0:
        return img
    x0, y0, x1, y1 = lm["bbox"]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bh = float(y1 - y0)
    gm = None
    if st["gains"]:
        g = map_letters(st["gains"], wp["n"], lm["m"])
        if g:
            vec = np.ones(lm["m"] + 1, np.float32)
            for r, gv in g.items():
                vec[r] = gv
            gm = B.asarray(vec[lm["rmap"]])
    scale, lift = st["scale"], st["lift"]
    shift_rows, sp = None, 0.0
    if st["bands"]:
        lim = GUARD["max_disp"] * w
        shift_rows = np.zeros(h, np.float32)
        for ya, yb, dx in st["bands"]:
            shift_rows[max(0, int(y0 + ya * bh)):max(0, int(y0 + yb * bh))] = float(np.clip(dx * w, -lim, lim))
        sp = float(min(max(1.0, st["split"] * w), lim))
        sp = min(sp, max(lim - float(np.abs(shift_rows).max()), 1.0))
    moves = abs(scale - 1.0) > 1e-5 or shift_rows is not None
    if moves:
        xs, ys = C._grid_xy(h, w, B)
        kk = 1.0 / scale - 1.0
        dxm, dym = (xs - cx) * kk, (ys - cy) * kk
        if shift_rows is not None:
            dxm = dxm - B.asarray(np.broadcast_to(shift_rows[:, None], (h, w)).copy())

    def alphas(a):
        """the alpha(s) after gains + warp: one, or three (R, G, B) with a split."""
        a = a * gm if gm is not None else a
        if not moves:
            return [a]
        if sp > 0:
            return [B.warp(a, dxm - sp, dym), B.warp(a, dxm, dym), B.warp(a, dxm + sp, dym)]
        return [B.warp(a, dxm, dym)]

    def union(als):
        o = als[0]
        for x in als[1:]:
            o = B.maximum(o, x)
        return o

    if not face:
        a_in = C._gray(img)
        a_in = C._fit(a_in, h, w) if tuple(a_in.shape[:2]) != (h, w) else a_in
        out = union(alphas(a_in)).clip(0.0, 1.0)
        return out if img.ndim == 2 else out[..., None]
    rgb = C._fit(img[..., :3], h, w) if tuple(img.shape[:2]) != (h, w) else img[..., :3]
    base = B.warp(rgb, dxm, dym) if moves else rgb
    als = alphas(a_ref)
    if len(als) == 3:
        top = B.maximum(B.maximum(als[0], als[1]), als[2])
        den = B.maximum(top, 1e-4)
        base = B.stack([als[0] * base[..., 0] / den, als[1] * base[..., 1] / den, als[2] * base[..., 2] / den], axis=-1)
    if lift > 1e-5:
        base = base * (1.0 + lift)
    return base.clip(0.0, 1.0)


OPS = {"hold_life": hold_life}


# ---------------------------------------------------------------------------------------------------------------------------------------------- weave (the planner hook)
DRAWERS = ("sandwich", "depth_occlude", "knockout")
LINEISH = ("$type_line", "$wl")


def unweave_ops(ops: list) -> list:
    """drop the ops this module added and give the drawers their original `type_a` back (idempotent weave)."""
    back = {o["out"]: o["args"]["ref"] for o in ops if o.get("hl") and o.get("op") == "hold_life"}
    out = []
    for o in ops:
        if o.get("hl"):
            continue
        a = o.get("args") or {}
        if o.get("op") in DRAWERS and a.get("type_a") in {"$" + k for k in back}:
            o = dict(o, args=dict(a, type_a=back[a["type_a"][1:]]))
        out.append(o)
    return out


def weave_ops(ops: list, p: dict, state_key: str, tint_ref=None) -> list:
    """the chain with hold_life in front of every type drawer and add_glow after the last one. A drawer whose `type_a` is a line pass (the old planner chains) or not a buffer is left alone."""
    ops = unweave_ops(copy.deepcopy(ops))
    made: dict = {}
    new = []
    last = None
    per_word = bool({"letter_flicker", "glitch_slice", "pulse"} & set(p.get("effects") or []))      # glow / outline alone need no op on the type layer
    wm_made = any(o.get("out") == "wm" and (o.get("args") or {}).get("type_a") == "$type_matte" for o in ops)
    if not wm_made and any(o.get("op") in DRAWERS and (o.get("args") or {}).get("type_a") in LINEISH for o in ops):
        # the chain wakes only the LINE pass ($wl): give the matte its own wake (what plan_rules.weave_flow makes for every alpha a chain reads) so the type keeps reacting to the footage flow
        # once its drawer is remapped to the matte buffer below
        for j, o in enumerate(ops):
            if o.get("op") == "flow_wake" and (o.get("args") or {}).get("type_a") == "$type_line":
                tw = copy.deepcopy(o)
                tw["out"] = "wm"
                tw["args"]["type_a"] = "$type_matte"
                tw["args"]["state_key"] = f"{(o.get('args') or {}).get('state_key', 'default')}:wm"
                tw["hl"] = False
                ops.insert(j + 1, tw)
                wm_made = True
                break
    for o in ops:
        a = o.get("args") or {}
        if o.get("op") in DRAWERS and a.get("type_a") in LINEISH:
            # 2026-10-30: a Gemma-written chain draws through the LINE pass ($wl); the compose legibility pass (recipe_eval.legibility_pass) redraws the visible type from the beauty pass / matte there anyway
            # ($wm when the chain has the matte flow_wake, else the plain type matte), so the same buffer is the one hold life lives on: r14-r16 pieces had no hold life at all on such chains.
            a["type_a"] = "$wm" if wm_made else "$type_matte"
        if o.get("op") in DRAWERS and isinstance(a.get("type_a"), str) and a["type_a"].startswith("$") and a["type_a"] not in LINEISH:
            ref = a["type_a"]
            if not per_word:
                last = o
            elif ref not in made:
                nm = f"hl{len(made)}"
                made[ref] = nm
                new.append({"op": "hold_life", "out": nm, "hl": True,
                            "args": {"img": ref, "ref": ref, "t": "$t", "frame_idx": "$frame_idx", "plan": p, "state_key": f"{state_key}:{nm}"}})
            if per_word:
                a["type_a"] = "$" + made[ref]
                last = o
        new.append(o)
    if last is not None and p.get("glow") and p["glow"].get("keys"):
        g = p["glow"]
        dst = last.get("out", "image")
        rad = {"curve": g["curve"], "lo": g["radius"][0], "hi": g["radius"][1]} if g.get("curve") else g["radius"][0]
        args = {"img": "$" + dst, "mask": last["args"]["type_a"], "radius": rad, "intensity": {"timeline": g["keys"]}}
        if tint_ref:
            args["color"] = tint_ref
        i = max(j for j, o in enumerate(new) if o is last)
        new.insert(i + 1, {"op": "add_glow", "out": dst, "hl": True, "args": args})
    return new


def requested_effects(piece) -> list:
    """effect names asked for: compose.hold_life (list or comma string), else piece.effects / the request's `effects` when they name hold-life effects."""
    comp = piece.get("compose") or {}
    v = comp.get("hold_life")
    if v is None:
        v = piece.get("effects")
    if isinstance(v, str):
        v = [x.strip().lower().replace("-", "_").replace(" ", "_") for x in v.replace(";", ",").split(",")]
    return [x for x in (v if isinstance(v, (list, tuple)) else []) if x in EFFECTS]


def weave(piece: dict, music: dict | None = None, chk: dict | None = None, size=None, holds=None) -> dict:
    """THE HOOK. Plan the hold life of the piece and weave it into every compose chain (in place; returns the piece; idempotent). Opt out: compose.hold_life = false. Words LTX owns
    (owner render 'ltx') get nothing; a handoff word's window stops at its handoff time. Holds: `holds` [(hit, release)], else plan_rules.word_holds(piece, music, chk)."""
    comp = piece.setdefault("compose", {})
    chains = ([comp["default"]] if comp.get("default") else []) + list((comp.get("shots") or {}).values())
    pl = piece.setdefault("plan", {})
    if comp.get("hold_life") is False:
        for ch in chains:
            ch["ops"] = unweave_ops(ch.get("ops", []))
        pl.pop("hold_life", None)
        return piece
    words = (piece.get("type") or {}).get("words") or []
    if holds is None:
        from tools.fusion import plan_rules as PR
        try:
            holds = PR.word_holds(piece, music, chk)
        except Exception as e:      # noqa: BLE001  the hook must never break a piece: no word timing, no hold life
            pl["hold_life"] = dict(effects=[], error=f"no word holds: {e}")
            return piece
    from tools.fusion import owner as OW
    wins = OW.windows(piece) if OW.has_owner(words) else None
    rows = []
    for i, (hit, rel) in enumerate(holds):
        w = words[i] if i < len(words) else {}
        text = str(w.get("text") or " ".join(w.get("lines") or []))
        if wins is not None and i < len(wins):
            until = OW.flat_visible_until(wins[i])
            if until != float("inf"):
                rel = min(rel, until)
                if rel <= hit + 0.2:
                    continue
        rows.append(dict(text=text, hit=float(hit), release=float(rel), src=i))
    seed = ((pl.get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0)))
    sizes = size or (piece.get("sizes") or {}).get("compose") or (540, 960)
    outline = _outline_of(piece, [(r["hit"], r["release"]) for r in rows])
    curves = (music or {}).get("curves") or piece.get("curves") or {}
    t = piece.get("type") or {}
    p = plan(rows, piece, seed, beats=(music or {}).get("beats") or t.get("beats"), bpm=(music or {}).get("bpm") or t.get("bpm"), requested=requested_effects(piece),
             curves=curves, outline=outline, size=sizes)
    for wp, r in zip(p["words"], rows):
        wp["src"] = r["src"]
    n = 0
    named = ([("default", comp["default"])] if comp.get("default") else []) + list((comp.get("shots") or {}).items())
    for name, ch in named:
        outs = {o.get("out") for o in ch.get("ops", [])}
        ch["ops"] = weave_ops(ch.get("ops", []), p, f"{piece.get('id', 'piece')}:{name}", "$tcol" if "tcol" in outs else None)
        n += 1
    if p.get("outline") is not None:
        comp["outline"] = p["outline"]                 # legible_color reads compose.outline (a list of windows) as the forced outline
    pl["hold_life"] = dict(effects=p["effects"], skipped=p["skipped"], chains=n, seed=p["seed"],
                           words=[dict(i=wp["src"], kinds=list(wp.get("kinds") or []), n=wp["n"], intensity=wp["intensity"], flicker=len(wp["flicker"]), slices=len(wp["slices"]), pulses=len((wp["pulse"] or {}).get("beats", [])),
                                       skipped=wp["skipped"]) for wp in p["words"]])
    return piece


def _outline_of(piece, holds) -> list:
    """the outline windows of the piece (the same ones recipe_eval._outline_windows would use), [] when there is none."""
    comp = piece.get("compose") or {}
    o = comp.get("outline")
    if o is False:
        return []
    if isinstance(o, list):
        return [[float(a), float(b)] for a, b in o]
    pl = piece.get("plan") or {}
    spec = o if isinstance(o, dict) else (pl.get("piece_system") or {}).get("outline")
    if not spec or not holds:
        return []
    from tools.fusion import random_system as RSY
    seed = (pl.get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0))
    return RSY.outline_windows(spec, holds, (piece.get("type") or {}).get("bpm") or (piece.get("audio") or {}).get("bpm"), seed)
