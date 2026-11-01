"""stable_type: type that LANDS STABLE and CENTRED (user 2026-10-30: "I want the text to land pretty stable for a beat - centre frame for now, as they work the best - the words can be stacked or
rotated 90 or -90 degrees to read vertically - but we should clearly see the transitions happening"; and: "words FORM ON THE BEAT").

Rules this module enforces on the design words (pure functions on dicts, no ComfyUI / network; every change is one fix string):

  1. CENTRED + STILL   every word: placement dx = dy = 0, yaw = pitch = 0, no screen offset, no hold layout that moves the box; the type camera is `beat_orbit` held still
                       (read_dps 0, bob 0, elev 0, intro_deg 0): the landed word does not move for its whole hold; motion lives only in the word's own form-in and exit.
  2. LAYOUT PER WORD   drawn per word from: `line` (one line centred), `stack` (the words of a phrase stacked, breaks only at spaces, centred), `rot90` / `rot-90` (rolled +-90 degrees, reads
                       vertically along the frame height, centred), `grid` (a word split MID-WORD into rows of 2-4 letters that fill the frame: DREAMING = DR / EA / MI / NG, SEARCHING = SEA / RCH /
                       ING, STEAM = STE / AM; the letters are all one size, the rows centred) and `column` (every letter on its own row: S / T / E / A / M, 3-8 letters). The only mid-word
                       splits are grid and column, and the NO-ORPHAN RULE holds (`orphan_free`): a split word never leaves a row with ONE letter unless every row is one letter (S-T / E / A-M and
                       STEA / M are invalid; S/T/E/A/M and STE / AM are valid). Every layout is fitted inside the safe area (fit_box) and a layout that cannot carry the word at the
                       size floor (design_defaults.CAP_FLOOR) falls back to the one with the biggest cap height. A rolled box is written in spec.build_scene's rolled convention:
                       wf = the long side as a fraction of the frame HEIGHT, hf = the short side as a fraction of the frame WIDTH.
  3. BEAT              the hold lasts at least one full beat (`min_hold_s(bpm)`); the form-in ends on a beat because spec.plan_timing snaps every hit to the beat grid (`landed_on_beat` checks it).
  4. TRANSITIONS READ  `window_ok(kind, frames)`: a drawn cut plays over >= MIN_TRANSITION_FRAMES frames (12 = 0.5 s at 24 fps) and the holds around it leave the window free.
"""
from __future__ import annotations

import random

from tools.fusion import design_defaults as DD
from tools.fusion import design_options as DO

LAYOUTS = ("line", "stack", "rot90", "rot-90", "grid", "column")
ROLL = {"line": 0.0, "stack": 0.0, "rot90": 90.0, "rot-90": -90.0, "grid": 0.0, "column": 0.0}
SPLIT_LAYOUTS = ("grid", "column")        # the layouts that break a word MID-WORD (the others break only at spaces)
GRID_ROW_MIN, GRID_ROW_MAX, GRID_ROWS = 2, 4, (2, 3, 4)        # a grid row has 2-4 letters, a word has 2-4 rows
GRID_MAX_TOTAL_ROWS = 4                                         # a grid WORD (a phrase counts all its words' rows) has at most this many rows: QC v026 r02 'FORGETS TURN' came out as a 5-6 row grid
FACE_LIMIT = 0.35                                               # a layout box may cover at most this share of the subject's face rectangle for the hold (own_video_v6_face_probe's limit)
FACE_TOP_SHARE, FACE_WIDTH_SHARE = 0.25, 0.5                    # the face rectangle = the top 25 % of the subject's matte box, its central 50 % of the width
COLUMN_LETTERS = (3, 8)                                         # a column is for words of 3-8 letters
LEAD = 1.1                                                      # row pitch of a split block (x cap height)
SAFE_H = 0.92                                                   # a split block is at most this share of the frame height (a single word aims at 80-90 %)
LAYOUT_WEIGHTS = {"line": 3.0, "stack": 2.0, "rot90": 2.0, "rot-90": 2.0, "grid": 4.0, "column": 1.0}        # 2026-10-30 (user: "I dont see any of the grid typography"): grid ~1 in 3 words of 4+ letters, column ~1 in 12
SAFE_W = 0.88                 # a horizontal line is at most this share of the frame width (the safe area keeps 6 % per side)
SAFE_LEN = 0.86               # a rolled line is at most this share of the frame height
MIN_BEATS_HOLD = 1.0          # the landed word holds still for at least this many beats
MIN_TRANSITION_FRAMES = 12    # a cut plays over at least this many frames (0.5 s at 24 fps)
STATIC_CAMERA = dict(read_dps=0.0, bob=0.0, elev=0.0, intro_deg=0.0)       # beat_orbit that never moves (cameras.BeatOrbit: no creep, no bob, no intro spin)


def min_hold_s(bpm, beats: float = MIN_BEATS_HOLD) -> float:
    """seconds of one full beat x beats at this tempo (the minimum hold of a landed word)."""
    try:
        return round(60.0 / float(bpm) * beats, 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.6


def static_camera_params(old: dict | None = None) -> dict:
    """camera_params of the held camera; params a caller had for the other beat_orbit settings (fast_dps ...) stay, the motion ones are zero."""
    out = {k: v for k, v in (old or {}).items() if k in ("fast_dps", "fill", "radius_scale", "lead")}
    out.update(STATIC_CAMERA)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- geometry of a layout
def _ratio(font, axes, line) -> float:
    r = DO.line_width_per_cap(font, axes, line) if font else None
    return r if r is not None else max(len(str(line).replace(" ", "")), 1) * DO.CHAR_W


# ---------------------------------------------------------------------------------------------------------------------------------------------- mid-word splits (grid / column)
def _letters(tok: str) -> str:
    return "".join(ch for ch in str(tok) if ch.isalnum())


def grid_splits(n: int) -> list:
    """the row-size tuples (longer rows first) that split a word of n letters into 2-4 rows of 2-4 letters, balanced (sizes differ by at most one): n=8 -> (4,4) (3,3,2) (2,2,2,2);
    n=9 -> (3,3,3) (3,2,2,2); n=5 -> (3,2); n=4 -> (2,2); n=3 -> [] (no grid: it would orphan a letter). Never a row of one letter."""
    out = []
    for r in GRID_ROWS:
        base, extra = divmod(int(n), r)
        sizes = tuple([base + 1] * extra + [base] * (r - extra))
        if sizes and min(sizes) >= GRID_ROW_MIN and max(sizes) <= GRID_ROW_MAX:
            out.append(sizes)
    return out


def rows_by_token(lines, text):
    """the rows of `lines` grouped by the words of `text` ([[rows of word 1], [rows of word 2] ...]); None when a row mixes two words or the letters do not spell the text."""
    toks = [_letters(t) for t in str(text).split() if _letters(t)]
    rows = [_letters(x) for x in (lines or []) if _letters(x)]
    out, k = [], 0
    for tok in toks:
        got, acc = [], ""
        while k < len(rows) and len(acc) < len(tok):
            got.append(rows[k])
            acc += rows[k]
            k += 1
        if acc.upper() != tok.upper():
            return None
        out.append(got)
    return out if k == len(rows) else None


def orphan_free(lines, text) -> bool:
    """THE NO-ORPHAN RULE: a word that is split over several rows never leaves a row with a single letter unless EVERY row of that word is a single letter (= column). S-T / E / A-M and
    STEA / M are invalid; S / T / E / A / M and STE / AM are valid; a word on ONE row (A, GO) is not split. False when the rows do not spell the text or mix two words."""
    g = rows_by_token(lines, text)
    if g is None:
        return False
    for rows in g:
        if len(rows) > 1 and any(len(r) == 1 for r in rows) and not all(len(r) == 1 for r in rows):
            return False
    return True


def grid_possible(text: str) -> bool:
    """a grid exists for the text: a word of 4+ letters, every word fits 4 rows of 4, and the rows of ALL its words together are at most GRID_MAX_TOTAL_ROWS (a short word is 1 row, a longer one its fewest balanced rows)."""
    toks = [_letters(t) for t in str(text).split() if _letters(t)]
    if not toks or not any(len(t) >= 4 for t in toks) or any(len(t) > GRID_ROW_MAX * max(GRID_ROWS) for t in toks):
        return False
    need = 0
    for t in toks:
        if len(t) < 4:
            need += 1
            continue
        sp = grid_splits(len(t))
        if not sp:
            return False
        need += min(len(x) for x in sp)                           # the fewest rows this word can be split into (REMEMBERS = 3 rows of 3)
    return need <= GRID_MAX_TOTAL_ROWS


def column_possible(text: str) -> bool:
    toks = [_letters(t) for t in str(text).split() if _letters(t)]
    return len(toks) == 1 and COLUMN_LETTERS[0] <= len(toks[0]) <= COLUMN_LETTERS[1]


def _split_rows(text: str, layout: str, font, axes, aspect: float):
    """the rows of a split layout: grid = per word the balanced split of the best cap height (equal rows preferred), a word under 4 letters stays one row; column = one letter per row.
    -> (rows [str], cap) or (None, 0)."""
    import itertools
    toks = [_letters(t) for t in str(text).split() if _letters(t)]
    if layout == "column":
        if not column_possible(text):
            return None, 0.0
        rows = list(toks[0])
        r = max(_ratio(font, axes, ch) for ch in rows)
        return rows, min(SAFE_W * aspect / r, SAFE_H / (len(rows) * LEAD))
    opts = []
    for t in toks:
        sp = grid_splits(len(t)) if len(t) >= 4 else []
        opts.append(sp if sp else [(len(t),)])
    best = None
    for combo in itertools.product(*opts):
        rows = []
        for t, sizes in zip(toks, combo):
            k = 0
            for z in sizes:
                rows.append(t[k:k + z])
                k += z
        if len(rows) > GRID_MAX_TOTAL_ROWS:                        # never more than 4 rows (the letters would drop under the size floor and the block fills the frame)
            continue
        r = max(_ratio(font, axes, row) for row in rows)
        cap = min(SAFE_W * aspect / r, SAFE_H / (len(rows) * LEAD))
        equal = all(len(set(c)) == 1 for c in combo)
        score = cap * (1.0 if equal else 0.9)
        if best is None or score > best[0] + 1e-9:
            best = (score, rows, cap)
    return (best[1], best[2]) if best else (None, 0.0)


def fit_box(text: str, font, axes, layout: str, aspect: float = 0.5625) -> dict:
    """the fitted box of a word in a layout. -> dict(layout, lines, roll, wf, hf, cap, ok, box [x0, y0, x1, y1] frame fractions of the INK box, centred on (0.5, 0.5)).
    cap = cap height / frame height from the real glyph widths (DO.line_width_per_cap); ok = cap >= CAP_FLOOR and the ink box is inside the safe area."""
    text = " ".join(str(text).split())
    toks = text.split()
    if layout in SPLIT_LAYOUTS:
        rows, cap = _split_rows(text, layout, font, axes, aspect)
        if not rows:
            return dict(layout=layout, lines=[text], roll=0.0, wf=SAFE_W, hf=0.3, cap=0.0, box=[0.0, 0.0, 0.0, 0.0], ok=False)
        r = max(_ratio(font, axes, row) for row in rows)
        width, height = r * cap / aspect, cap * len(rows) * LEAD
        wf = round(min(SAFE_W + 0.02, width / 0.94 + 0.02), 3)
        hf = round(min(0.95, height / 0.9 + 0.02), 3)
        box = [0.5 - width / 2, 0.5 - height / 2, 0.5 + width / 2, 0.5 + height / 2]
        return dict(layout=layout, lines=rows, roll=0.0, wf=wf, hf=hf, cap=round(cap, 4), box=[round(v, 4) for v in box],
                    ok=cap + 1e-9 >= DD.CAP_FLOOR and width <= SAFE_W + 1e-9 and height <= SAFE_H + 1e-9 and orphan_free(rows, text), split=True)
    if layout in ("rot90", "rot-90"):
        lines = [text]
        r = _ratio(font, axes, text)
        cap = min(SAFE_LEN / r, 0.88 * aspect)               # along the frame height, and the cap (horizontal extent) inside the width
        length = r * cap
        wf = round(min(SAFE_LEN + 0.02, length / 0.94 + 0.01), 3)
        hf = round(min(0.9, cap / aspect * 1.15 + 0.04), 3)
        box = [0.5 - cap / aspect / 2, 0.5 - length / 2, 0.5 + cap / aspect / 2, 0.5 + length / 2]
        return dict(layout=layout, lines=lines, roll=ROLL[layout], wf=wf, hf=hf, cap=round(cap, 4), box=[round(v, 4) for v in box], ok=cap + 1e-9 >= DD.CAP_FLOOR and length <= SAFE_LEN + 1e-9)
    lines = toks if layout == "stack" and len(toks) > 1 else [text]
    if layout == "stack" and len(lines) > 3:
        lines = [" ".join(toks[:2])] + toks[2:4] if len(toks) == 4 else lines[:3]
    rs = [_ratio(font, axes, ln) for ln in lines]
    cap = min(SAFE_W * aspect / max(rs), 0.9 / len(lines))
    width = max(rs) * cap / aspect                             # fraction of the frame width
    height = cap * len(lines) * (1.0 if len(lines) == 1 else 1.12)
    wf = round(min(SAFE_W + 0.02, width / 0.94 + 0.01), 3)
    hf = round(min(0.95, height / 0.9 + 0.02), 3)
    box = [0.5 - width / 2, 0.5 - height / 2, 0.5 + width / 2, 0.5 + height / 2]
    return dict(layout=layout, lines=lines, roll=0.0, wf=wf, hf=hf, cap=round(cap, 4), box=[round(v, 4) for v in box], ok=cap + 1e-9 >= DD.CAP_FLOOR and width <= SAFE_W + 1e-9)


def allowed(text: str, fmt: str = "9:16") -> list:
    """the layouts that exist for this word: `stack` needs a phrase (breaks only at spaces); a landscape frame has no rolled layouts."""
    toks = str(text).split()
    out = ["line"]
    if len(toks) > 1:
        out.append("stack")
    if fmt != "16:9":
        out += ["rot90", "rot-90"]
    if grid_possible(text):
        out.append("grid")
    if column_possible(text):
        out.append("column")
    return out


def draw_layouts(seed, texts, fmt: str = "9:16") -> list:
    """one layout per word, seeded: weighted, never the same layout as the word before, never two rolled words in a row with the same sign (alternating signs read as a rhythm)."""
    out: list = []
    for i, t in enumerate(texts):
        rng = random.Random(f"layout:{int(seed or 0)}:{i}:{t}")
        cand = allowed(t, fmt)
        w = []
        for c in cand:
            x = LAYOUT_WEIGHTS[c]
            if out and c == out[-1]:
                x *= 0.0 if len(cand) > 1 else 0.15                   # 2026-10-31 (like the cuts): never the same layout twice in a row when the word has another
            w.append(x)
        tot = sum(w)
        r = rng.random() * tot
        pick = cand[-1]
        for c, x in zip(cand, w):
            r -= x
            if r <= 0:
                pick = c
                break
        out.append(pick)
    return out


def force_layouts(drawn: list, texts: list, effects: dict | None, seed=0, fmt: str = "9:16", protected=()) -> list:
    """a request's `effects` names `grid` / `column` (effects_menu): at least half of the words (and >= 2 when there are 2+) take one of the requested layouts, longest words first (seeded
    ties), only where the layout exists for the word (`allowed`: the no-orphan rule); words already drawn grid / column count. best_layout still applies the size floor afterwards."""
    req = [n for n in ((effects or {}).get("requested") or []) if n in SPLIT_LAYOUTS]
    if not req or not texts:
        return drawn
    out = list(drawn)
    want = min(len(texts), max(2, -(-len(texts) // 2)))
    have = sum(1 for d in out if d in req)
    rng = random.Random(f"layout-force:{int(seed or 0)}:{len(texts)}")
    order = sorted((i for i in range(len(texts)) if i not in protected and out[i] not in req), key=lambda i: (-len(_letters(texts[i])), rng.random()))
    k = 0
    for i in order:
        if have >= want:
            break
        opts = [r for r in req[k % len(req):] + req[:k % len(req)] if r in allowed(texts[i], fmt)]
        if opts:
            out[i] = opts[0]
            have += 1
            k += 1
    for r in req:                                                    # 2026-10-30 (QC v026 r02: `column` was requested and every word was already a grid, so none became a column -> HANDS came out rotated): EACH requested layout
        if r in out:                                                 # is used by at least one eligible word, whatever the others took
            continue
        elig = [i for i in range(len(texts)) if i not in protected and r in allowed(texts[i], fmt)]
        if not elig:
            continue
        held = {q: sum(1 for d in out if d == q) for q in req}        # never take the only holder of another requested layout
        elig = [i for i in elig if out[i] not in req or held[out[i]] > 1] or []
        if not elig:
            continue
        want_n = 5 if r == "column" else 6                           # a column reads best at ~5 letters (cap above the floor), a grid at ~6
        elig.sort(key=lambda i: (abs(len(_letters(texts[i])) - want_n), random.Random(f"layout-each:{int(seed or 0)}:{r}:{i}").random()))
        out[elig[0]] = r
    return out


def face_rect(matte, top: float = FACE_TOP_SHARE, width: float = FACE_WIDTH_SHARE, min_px: int = 200):
    """the subject's FACE rectangle (x0, y0, x1, y1 as frame fractions) from one matte frame (HxW, 0..1 or 0..255): the top `top` of the matte's bounding box, the central `width` of its width.
    None when there is no subject (fewer than min_px matte pixels). The rule behind it (user 2026-10-30): a letter grid never sits over the subject's face for its hold."""
    import numpy as np
    m = np.asarray(matte)
    if m.ndim == 3:
        m = m[..., 0]
    thr = 127 if m.max() > 1.5 else 0.5
    ys, xs = np.where(m > thr)
    if len(ys) < min_px:
        return None
    hh, ww = m.shape[:2]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cx, h, w = 0.5 * (x0 + x1), float(y1 - y0), float(x1 - x0)
    return ((cx - 0.5 * width * w) / ww, y0 / hh, (cx + 0.5 * width * w) / ww, (y0 + top * h) / hh)


def face_cover(box, rect) -> float:
    """the share of the face rectangle `rect` that a layout box [x0, y0, x1, y1] (frame fractions) covers (0 = clear, 1 = all of it)."""
    if not rect or not box:
        return 0.0
    ix0, iy0, ix1, iy1 = max(box[0], rect[0]), max(box[1], rect[1]), min(box[2], rect[2]), min(box[3], rect[3])
    area = max(rect[2] - rect[0], 1e-6) * max(rect[3] - rect[1], 1e-6)
    return max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0) / area


def best_layout(text: str, font, axes, drawn: str, fmt: str = "9:16", aspect: float = 0.5625, face=None, avoid=None) -> dict:
    """_best_layout + `repeat_forced` = True when the layout is `avoid` (the layout of the word before) because no other layout carries the word at the size floor (legibility beats variety)."""
    r = _best_layout(text, font, axes, drawn, fmt, aspect, face, avoid)
    return dict(r, repeat_forced=True) if avoid is not None and r["layout"] == avoid else r


def _best_layout(text: str, font, axes, drawn: str, fmt: str = "9:16", aspect: float = 0.5625, face=None, avoid=None) -> dict:
    """the drawn layout when it passes (cap >= floor, inside the safe area), else the allowed layout with the biggest cap height. `face` = the subject's face rectangle over the word's hold (face_rect;
    None = no matte / no subject): a split layout (grid / column) whose box covers more than FACE_LIMIT of it does not pass; the others are tried (stack / line / rolled), the one that covers the
    face least wins when none is clear."""
    cand = allowed(text, fmt)
    boxes = {c: fit_box(text, font, axes, c, aspect) for c in cand}
    if face:
        for c, b in boxes.items():
            b["face_cover"] = round(face_cover(b["box"], face), 3)
            if b.get("split") and b["face_cover"] > FACE_LIMIT:
                b["ok"] = False
                b["why_not"] = f"covers {b['face_cover']:.0%} of the subject's face"
    d = boxes.get(drawn) or boxes[cand[0]]
    if avoid is not None and d["layout"] == avoid:                    # VARIETY RULE (2026-10-31, like the cuts): never the layout of the word before when another one carries the word at the size floor
        alt = [b for c, b in boxes.items() if c != avoid and b["ok"]]
        if alt:
            return max(alt, key=lambda b: b["cap"])
    if d["ok"]:
        return d
    ok = [b for b in boxes.values() if b["ok"] and (avoid is None or b["layout"] != avoid)] or [b for b in boxes.values() if b["ok"]]
    if not ok and face:                                                # nothing passes: the least face-covering layout that still reads
        reads = [b for b in boxes.values() if b["cap"] >= DD.CAP_FLOOR] or list(boxes.values())
        return min(reads, key=lambda b: (b.get("face_cover", 0.0), -b["cap"]))
    return max(ok or boxes.values(), key=lambda b: b["cap"])


# ---------------------------------------------------------------------------------------------------------------------------------------------- enforce on a design
def enforce(t: dict, system: dict | None, bpm, fixes: list, protected=(), seed=0, fmt: str = "9:16", keep_font=None) -> int:
    """in place on a design `type` object (words, camera ...): centre + still + layout + minimum hold + held camera. `protected` = indices of window words: they keep the face, the lines and the
    box of the window fit (their cap is 30 % of the frame) but are centred and still too. -> number of fields changed."""
    n = 0
    aspect = 16.0 / 9.0 if fmt == "16:9" else 9.0 / 16.0
    words = t.get("words") or []
    seed = int((system or {}).get("seed", seed) or 0)
    drawn = draw_layouts(seed, [str(w.get("text", "")) for w in words], fmt)
    pin = (system or {}).get("layouts")                                   # a piece author can pin the layouts (a list, one per word)
    if isinstance(pin, (list, tuple)):
        drawn = [pin[i] if i < len(pin) and pin[i] in LAYOUTS else d for i, d in enumerate(drawn)]
    drawn = force_layouts(drawn, [str(w.get("text", "")) for w in words], (system or {}).get("effects"), seed, fmt, protected)
    hold_min = min_hold_s(bpm)
    if keep_font is None:                                                 # the user's per-line fonts (tools/fusion/font_pins.py): never swapped for legibility, the layout is fitted around them
        from tools.fusion import font_pins as FP
        keep_font = FP.pinned_indices(words, t.get("font_pins"))
    keep_font_ = set(keep_font)
    fz = (system or {}).get("face_rects")                                 # {word index | list position: (x0, y0, x1, y1)} the subject's face over each word's hold (from a matte: face_rect); None = no matte
    faces = [fz.get(i) if isinstance(fz, dict) else (fz[i] if isinstance(fz, (list, tuple)) and i < len(fz) else None) for i in range(len(words))]
    chosen: dict = {}
    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        pl = w.get("placement") if isinstance(w.get("placement"), dict) else {}
        w["placement"] = pl
        was = {k: pl.get(k) for k in ("dx", "dy", "yaw", "pitch")}
        if i not in protected:
            fr_ = faces[i] if i < len(faces) else None
            bl = best_layout(str(w.get("text", "")), w.get("font"), w.get("axes"), drawn[i], fmt, aspect, fr_, avoid=chosen.get(i - 1))
            if i not in keep_font_ and DD.pixel_swap(words, i, fixes, bl.get("cap"), protected, (system or {}).get("fonts")):                 # a pixel / dot face on a long word, a ray / extrusion module or a small fitted cap does not read (QC v026): another face, refitted
                bl = best_layout(str(w.get("text", "")), w.get("font"), w.get("axes"), drawn[i], fmt, aspect, fr_, avoid=chosen.get(i - 1))
                n += 1
            chosen[i] = bl["layout"]
            old = (w.get("lines"), pl.get("roll"), pl.get("wf"), pl.get("hf"))
            if len(str(w.get("text", "")).split()) > 1 or w.get("lines") or bl.get("split"):          # a single word needs no `lines`; a phrase gets the layout's lines; a grid / column word its rows (mid-word)
                w["lines"] = list(bl["lines"])
            if bl.get("split"):
                w["ar_min"] = 0.85                                         # the rows keep their letters' shape (never squeezed by more than 15 %, hold_layouts.AR_MIN)
            pl["roll"], pl["wf"], pl["hf"] = bl["roll"], bl["wf"], bl["hf"]
            if old != (w.get("lines"), pl.get("roll"), pl.get("wf"), pl.get("hf")):
                fixes.append(f"type.words[{i}] layout {bl['layout']}" + (f" (the draw was {drawn[i]}; it did not carry the word at the size floor)" if bl["layout"] != drawn[i] else "")
                             + f": {'/'.join(bl['lines'])}, roll {bl['roll']:g}, cap ~{bl['cap']:.0%} of the frame height, centred (user rule: stable, centred, stacked or rotated)")
                n += 1
        pl["dx"] = pl["dy"] = pl["yaw"] = pl["pitch"] = 0.0
        if any(v not in (None, 0, 0.0) for v in was.values()):
            fixes.append(f"type.words[{i}].placement dx/dy/yaw/pitch {was} -> 0: the word lands centred and still")
            n += 1
        if w.pop("screen_offset", None) is not None:
            fixes.append(f"type.words[{i}].screen_offset dropped (centred)")
            n += 1
        hl = w.get("hold_layout")
        if isinstance(hl, dict) and (hl.get("box") or hl.get("offset")):
            w.pop("hold_layout")
            fixes.append(f"type.words[{i}].hold_layout {hl.get('name')!r} dropped (the word is centred)")
            n += 1
        try:
            hs = float(w.get("hold_s", 0.0))
        except (TypeError, ValueError):
            hs = 0.0
        if "hold_s" in w and hs < hold_min - 1e-9:
            fixes.append(f"type.words[{i}].hold_s {hs:g} -> {hold_min:g} (a landed word holds still for at least one beat)")
            w["hold_s"] = hold_min
            n += 1
    for r in [q for q in (((system or {}).get("effects") or {}).get("requested") or []) if q in SPLIT_LAYOUTS]:       # a requested layout nobody got: say why (never a silent fallback)
        if r in chosen.values():
            continue
        elig = [i for i, w in enumerate(words) if isinstance(w, dict) and i not in protected and r in allowed(str(w.get("text", "")), fmt)]
        if not elig:
            why = "no eligible word (" + ("column: one word of 3-8 letters" if r == "column" else f"grid: a word of 4+ letters whose rows total at most {GRID_MAX_TOTAL_ROWS}") + ")"
        else:
            why = "the eligible words fell back: " + "; ".join(f"{words[i].get('text')!r} -> {chosen.get(i)}" for i in elig)
        fixes.append(f"effects {r} requested but no word uses it: {why}")
    cp = static_camera_params(t.get("camera_params") if isinstance(t.get("camera_params"), dict) else None)
    if t.get("camera") != "beat_orbit" or t.get("camera_params") != cp:
        fixes.append(f"type.camera {t.get('camera')!r} -> 'beat_orbit' held still (read_dps 0, bob 0, elev 0): the landed words do not move")
        t["camera"], t["camera_params"] = "beat_orbit", cp
        t.pop("camera_override", None)
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------------------------------- checks used by the tests and the report
def landed_on_beat(hits, beat_s: float, fps: int = 24, t0: float = 0.0, tol_frames: float = 1.0) -> list:
    """[(hit s, distance to the nearest beat in frames)] for hits not within tol_frames of a beat (the empty list = every word lands on a beat)."""
    bad = []
    for h in hits:
        k = round((float(h) - t0) / beat_s)
        d = abs((float(h) - t0) - k * beat_s) * fps
        if d > tol_frames + 1e-9:
            bad.append((float(h), round(d, 3)))
    return bad


def window_ok(frames: int, fps: int = 24, min_frames: int = MIN_TRANSITION_FRAMES) -> bool:
    """does a transition of `frames` output frames read (>= 0.5 s)?"""
    return int(frames) >= int(min_frames)


def transition_windows(cuts, shots, fps: int = 24) -> list:
    """[(kind, after id, start s, end s)] of the 3D cuts of a design: the move plays from t_cut - i2 / fps to t_cut + (n - i2) / fps (pivot_cut.window_frames); a window / hard / plain cut has none."""
    from tools.fusion import cut_kinds as CK
    from tools.fusion import pivot_cut as PC
    rows = {s["id"]: s for s in shots or []}
    ids = [s["id"] for s in shots or []]
    out = []
    for c in cuts or []:
        kd = CK.normalize(c.get("cut")) if isinstance(c, dict) else None
        if kd not in CK.SHOT_3D or c.get("after") not in ids or ids.index(c["after"]) + 1 >= len(ids):
            continue
        prm = dict(c.get("params") or {})
        if c.get("style") and "style" not in prm:
            prm["style"] = c["style"]
        n, i2 = PC.window_frames(kd, {k: v for k, v in prm.items() if k != "dir"})
        t_cut = float(rows[ids[ids.index(c["after"]) + 1]]["start"])
        out.append((kd, c["after"], round(t_cut - i2 / fps, 3), round(t_cut + (n - i2) / fps, 3)))
    return out


def transition_overlaps(words, windows, tol_s: float = 0.3) -> list:
    """[(word index, kind, seconds)] of the words whose HOLD (hit .. release, seconds on the word) still runs `tol_s` or more into a 3D cut's move window: the word hides the transition. A word
    whose release is at the start of the window (it exits as the move starts) does not count."""
    out = []
    for i, w in enumerate(words or []):
        try:
            hit, rel = float(w["hit"]), float(w["release"])
        except (KeyError, TypeError, ValueError):
            continue
        for kd, _after, a, b in windows:
            ov = min(rel, b) - max(hit, a + tol_s)
            if ov > 1e-6 and hit < b and rel > a + tol_s:
                out.append((i, kd, round(ov, 2)))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- coverage: no long stretch without a word
COVER_MIN_S = 0.75            # the empty stretch between a word's release and the next word's form-in start is at most max(one beat, this)
COVER_EXIT_FLOOR = 0.4        # the exit is shortened down to this (spec.OVERLAP_FLOORS["exit"]) before the next word is pulled in
COVER_GAP_FLOOR = 0.35        # spec.plan_timing reads an explicit gap (next t0 - this t1) only from 0.35 s up; below it the pace default applies


def max_blank_s(bpm) -> float:
    """the longest stretch allowed between one word's release and the next word's form-in start: max(1 beat, 0.75 s)."""
    try:
        return max(60.0 / float(bpm), COVER_MIN_S)
    except (TypeError, ValueError, ZeroDivisionError):
        return COVER_MIN_S


def blank_gaps(chk_words, bpm, windows=(), pan_after=(), tol: float = 0.02) -> list:
    """[(word index i, blank s, limit s)]: i's release -> i+1's form-in start is longer than max_blank_s, and the stretch is NOT a designed transition: a 3D cut's move window
    (`windows` from transition_windows: that move is on screen then) or a pan after word i (`pan_after` = indices: the camera travel is the transition). chk_words = [{start, hit, release, end}]."""
    lim = max_blank_s(bpm)
    pans = set(pan_after)
    out = []
    for i in range(len(chk_words) - 1):
        rel, nxt = float(chk_words[i]["release"]), float(chk_words[i + 1]["start"])
        blank = nxt - rel
        if blank <= lim + tol or i in pans:
            continue
        if any(float(a) < nxt and float(b) > rel for _k, _after, a, b in windows):
            continue
        out.append((i, round(blank, 3), round(lim, 3)))
    return out


def _shot_index(t: float, starts) -> int:
    k = 0
    for j, s0 in enumerate(starts or []):
        if t + 1e-6 >= float(s0):
            k = j
    return k


def _cover_problems(cw, starts) -> int:
    """words whose hold runs across a shot boundary (the hold must end before the cut into the next shot)."""
    if not starts:
        return 0
    return sum(1 for w in cw if _shot_index(float(w["hit"]), starts) != _shot_index(float(w["release"]) - 0.02, starts))


def enforce_coverage(t: dict, run, windows=(), pan_after=(), starts=None, cut=None, fixes=None, tries: int = 4) -> list:
    """in place on a type spec `t` (words with t0 / t1 / exit_s, bpm, duration): shorten the long empty stretches between words (QC v026: 1.0-1.25 s of nothing after every word, a ~1.6 s travel, r01 / r02 / r03).
    `run(t) -> [{start, hit, release, end}]` = the timeline of a spec (plan_rules.dry_run words, or spec.plan_timing). Per offending gap the exit of the word is shortened (down to COVER_EXIT_FLOOR: the
    hold gets longer, the word still ends when it did), then the next word is pulled in on its explicit gap (down to COVER_GAP_FLOOR; later words shift with it), and when the beat-snapped mode cannot get
    there (tempo above 90 bpm) the overlap mode is switched on (`overlap: on`: only the hits stay on the grid). A change is kept only when it shrinks the total excess, adds no new problem
    (no hold across a cut, the last word still ends 0.3 s before the end) and leaves every hit in its own shot; else it is undone. -> the final timeline."""
    import copy
    fixes = fixes if fixes is not None else []
    bpm = t.get("bpm")
    cw = run(t)

    def excess(c):
        return sum(b - lim for _i, b, lim in blank_gaps(c, bpm, windows, pan_after))

    def ok_vs(c, old):
        if _cover_problems(c, starts) > _cover_problems(old, starts):
            return False
        if cut is not None and max(float(w["end"]) for w in c) > max(float(cut) - 0.3, max(float(w["end"]) for w in old)) + 1e-6:
            return False
        if starts and any(_shot_index(float(a["hit"]), starts) != _shot_index(float(b["hit"]), starts) for a, b in zip(c, old)):
            return False
        locked = [i for i, w in enumerate(t.get("words") or []) if isinstance(w, dict) and w.get("render") not in (None, "flat")]      # an LTX-owned word is drawn by the footage at a fixed time (ltx_t): it does not move
        if any(abs(float(c[i]["hit"]) - float(old[i]["hit"])) > 0.05 for i in locked if i < len(c) and i < len(old)):
            return False
        return True

    for _ in range(tries):
        gaps = blank_gaps(cw, bpm, windows, pan_after)
        if not gaps:
            break
        i, blank, lim = gaps[0]
        words = t.get("words") or []
        if i + 1 >= len(words):
            break
        base_ex = excess(cw)
        tried = []
        for step in ("exit", "gap", "overlap"):
            cand = copy.deepcopy(t)
            cws = cand["words"]
            note = None
            x_cur = float(cw[i]["end"]) - float(cw[i]["release"])
            ex_left = blank - lim
            new_x = max(COVER_EXIT_FLOOR, x_cur - ex_left)
            if new_x < x_cur - 0.01:
                cws[i]["exit_s"] = round(new_x, 3)
                ex_left -= x_cur - new_x
                note = f"exit {x_cur:.2f} -> {new_x:.2f} s"
            if step in ("gap", "overlap") and ex_left > 0.01:
                gap_cur = float(cw[i + 1]["start"]) - float(cw[i]["end"])
                new_gap = max(COVER_GAP_FLOOR, gap_cur - ex_left)
                if new_gap < gap_cur - 0.01 and "t1" in cws[i] and "t0" in cws[i + 1]:
                    d = float(cws[i]["t1"]) + new_gap - float(cws[i + 1]["t0"])
                    for w2 in cws[i + 1:]:
                        for k in ("t0", "t1"):
                            if k in w2:
                                w2[k] = round(float(w2[k]) + d, 3)
                    note = (note + "; " if note else "") + f"gap {gap_cur:.2f} -> {new_gap:.2f} s (next word pulled in {abs(d):.2f} s)"
            if step == "overlap":
                if cand.get("overlap") in (True, "on"):
                    continue
                cand["overlap"] = "on"
                note = (note + "; " if note else "") + "overlap mode on (only the hits stay on the beat grid)"
            if note is None:
                continue
            try:
                c2 = run(cand)
            except Exception:       # noqa: BLE001  a spec the engine rejects is never kept
                continue
            tried.append(step)
            if excess(c2) < base_ex - 0.01 and ok_vs(c2, cw):
                t.clear()
                t.update(cand)
                cw = c2
                fixes.append(f"type.words[{i}] -> [{i + 1}]: {blank:.2f} s with no word on screen (limit {lim:.2f} s = max(1 beat, 0.75 s)): {note}")
                break
        else:
            fixes.append(f"type.words[{i}] -> [{i + 1}]: {blank:.2f} s with no word on screen (limit {lim:.2f} s) could not be shortened without breaking a hold or a cut ({', '.join(tried) or 'nothing to change'})")
            break
    return cw
