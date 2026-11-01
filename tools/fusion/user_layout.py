"""user_layout: the LAYOUT half of the user's "more legible text" override (2026-11-01, real run R12b: direction "more legible text, palette: crimson, gold" reached the piece, the
colour stage and the readable-size floor, but the words were still laid out by the design rules: FINAL ECHO rolled 90 deg at a cap of ~10 %).

The layout (line / stack / rolled / grid, its cap height) is chosen by stable_type / plan_rules BEFORE the type pass, from the words and the safe area only; they never read
`user_style`. This module is the one place that re-fits a design's words when `user_style.parse_direction(piece)["legible_boost"]` is set, and it only calls stable_type.fit_box
(the same geometry the rules use), so every result is a layout the rules would accept:

    apply(piece, fixes=None) -> int      in place on piece["type"]["words"]; 0 changes without a legibility wish
    boost_layout(t, fmt, fixes)          the same on a design `type` object
    apply_json(piece_json) -> (piece_json, [fix strings])      the string form the design nodes pass around

Rules per word:
  0. (2026-11-01, R17b "SAC/RED/NOI/SE" in BitcountSingle, "TOTAL LOSS" in JacquardaBastarda9) a MID-WORD split (grid / column rows that break inside a word) is re-laid as whole words:
     one line or a stack of whole words, the one with the bigger cap that passes the size floor (an upright layout first, a rolled one only when no upright one passes);
     and a RISKY face the user did NOT pick (a word with `font_pin` / a `type.font_pins` entry keeps its face: user_pinned) (design_defaults.MAIN_HOLD_RISKY, which is also tools/app/fonts_route.py's `risky` flag, plus every pixel / dot-matrix / bitmap-blackletter face:
     design_defaults.PIXEL_DOT_FACES, role pixel / blackletter, names Bitcount* / Silkscreen / Jacquard* / Pixel* / Doto / Bytesized) is swapped for a readable LIKED face
     (never risky): a heavy reader first, then readable ornamental, then tall; a face the piece does not use yet when there is one, and
     the first whose whole-word layout passes the size floor at >= MIN_CAP (a wide face cannot carry a long phrase). A non-risky face is swapped the same way only when a
     mid-word split cannot be re-laid in it at the size floor. The swap drops the old face's axes and re-fits the word's layout to the new glyph widths
     (it never touches a user pin; the pin rewrite below is only a safety net).
  1. a ROLLED word (roll +-90) becomes upright (line / stack, the one with the bigger cap) when an upright layout reads at the size floor (design_defaults.CAP_FLOOR) and is not
     smaller than 95 % of the rolled cap: sideways text is the least legible layout;
  2. an upright word whose cap is under MIN_CAP (0.12 of the frame height) takes the allowed line / stack / rolled layout with the biggest cap that passes the size floor, when it is
     at least 2 % bigger.
  RARE MID-WORD SPLIT (user 2026-11-01, render R19 "SURFER" as SU/RF/ER, every piece, image and video mode): a split word stays only as a RARE option. relay_midword(piece) is the
general pass apply() runs for every piece WITHOUT the legibility wish: at most ONE word per piece keeps its mid-word split, and only in about 1 piece in 4 (seeded per piece:
MIDWORD_ONE_IN, keep_index()); every other split word is re-laid as whole words by the rule-0 geometry above (a font pin stays, only the layout changes). legible_boost pieces keep none.
Modules, timing and everything else stay; only font (risky faces) and lines / roll / wf / hf of the placement change (what stable_type.enforce writes). Every change is also appended to
`piece["user_style_log"]` by apply(). Without the legibility wish only the rare-split pass runs (nothing else is touched). Pure python (stable_type is pure)."""
from __future__ import annotations

import json
import re

MIN_CAP = 0.12            # same number as recipe_eval.BOOST_MIN_CAP (the compose floor): cap height / frame height
GAIN = 1.02               # a re-fit must be at least this much bigger to be worth the change
UPRIGHT_KEEP = 0.95       # rolled -> upright: the upright cap may be at most this much smaller
MIDWORD_ONE_IN = 4        # a piece keeps one mid-word split in about 1 piece in this many (user 2026-11-01); the other pieces keep none


PIXEL_NAME = re.compile(r"^(Bitcount|Silkscreen|Jacquard|Pixel|Doto|Bytesized|GeistPixel|PaletteMosaic|Codystar)", re.I)      # pixel / blackletter-bitmap / dot-matrix families by name
SWAP_ROLES = ("heavy", "tall", "ornamental")                                                                             # the display roles a risky face is swapped to (expressive but readable)


def _norm(s) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def _mid_split(w: dict, toks: list) -> bool:
    """the word's rows break INSIDE a word (grid / column: 'SAC/RED/NOI/SE'): the tokens of the rows are not the word's own tokens."""
    lines = w.get("lines") if isinstance(w.get("lines"), list) and w.get("lines") else None
    if not lines:
        return False
    return [_norm(x) for x in " ".join(str(x) for x in lines).split()] != [_norm(x) for x in toks]


def user_pinned(t: dict, w: dict) -> bool:
    """the USER chose this word's face in the app: the storyboard font control (web_app/app.js storyExtras: only fonts the user edited / picked go out as `fonts`) -> request.font_pins
    -> plan_rules `type.font_pins` + the word's `font_pin` (tools/fusion/font_pins.py). The random piece system / design rules never write either, so a pin = the user's choice and wins."""
    if w.get("font_pin"):
        return True
    return any(isinstance(pn, dict) and pn.get("font") and _norm(pn.get("text")) == _norm(w.get("text")) for pn in ((t or {}).get("font_pins") or []))


def is_risky(font) -> bool:
    """a face that is hard to read at this size: design_defaults.MAIN_HOLD_RISKY (as tools/app/fonts_route.py's `risky` flag; family name matched too), design_defaults.PIXEL_DOT_FACES,
    role pixel / blackletter in liked.json, or a pixel / bitmap / dot-matrix family name (Bitcount*, Silkscreen, Jacquard*, Pixel*, ...)."""
    from mograph import fonts_liked as fl
    from tools.fusion import design_defaults as DD
    if not font:
        return False
    e = fl.lookup(font)
    stem = re.sub(r"[\[\-_.].*$", "", str(font))
    key = (e or {}).get("key") or stem
    fam = re.sub(r"[^A-Za-z0-9]", "", str((e or {}).get("family") or ""))
    bad = tuple(DD.MAIN_HOLD_RISKY) + tuple(DD.PIXEL_DOT_FACES)
    return bool(key in bad or (fam and fam in bad) or (e or {}).get("role") in ("pixel", "blackletter") or PIXEL_NAME.match(key) or PIXEL_NAME.match(stem))


def swap_pool() -> list:
    """the liked files a risky face can become, best first: heavy readers (design_defaults.HEAVY_READERS order), readable ornamental (READABLE_EXPRESSIVE), then the other liked
    heavy / tall faces (an ornamental face only from the vetted READABLE_EXPRESSIVE list). Never a risky one."""
    from mograph import fonts_liked as fl
    from tools.fusion import design_defaults as DD
    order = [k for k in DD.HEAVY_READERS] + [k for k in DD.READABLE_EXPRESSIVE] + [e["key"] for e in fl.LIKED if e.get("role") in ("heavy", "tall")]
    out = []
    for k in order:
        e = fl.BY_KEY.get(k)
        if e and e.get("role") in SWAP_ROLES and not is_risky(e["file"]) and e["file"] not in out:
            out.append(e["file"])
    return out


def _best_upright(text: str, font, axes, fmt: str, aspect: float):
    """the upright whole-word layout (line / stack) with the biggest cap for this face: (box, ok) with ok = it passes the size floor; (None, False) when the word has none."""
    from tools.fusion import stable_type as ST
    bs = [ST.fit_box(text, font, axes, c, aspect) for c in ST.allowed(text, fmt) if c in ("line", "stack")]
    good = [b for b in bs if b["ok"]]
    pool = good or bs
    return (max(pool, key=lambda b: b["cap"]), bool(good)) if pool else (None, False)


def _swap_font(t: dict, i: int, w: dict, fixes: list | None, why: str = "risky", fmt: str = "9:16", min_cap: float = MIN_CAP) -> bool:
    """swap word i's face for a readable liked one. Candidates: the swap pool, a face no other word of the piece wears first, rotated by the word index so two words get different
    faces; the first whose whole-word layout passes the size floor at >= min_cap wins (a wide face on a long phrase cannot), else the one with the biggest passing cap."""
    from mograph import fonts_liked as fl
    old = w.get("font") or w.get("font_pin")
    if not old:
        return False
    text = " ".join(str(w.get("text", "")).split())
    aspect = 16.0 / 9.0 if fmt == "16:9" else 9.0 / 16.0
    words = [x for x in (t.get("words") or []) if isinstance(x, dict)]
    used = {(fl.lookup(x.get("font")) or {}).get("file") for x in words if x is not w and x.get("font") and not is_risky(x.get("font"))}
    pool = [f for f in swap_pool() if f != old]
    if not pool:
        return False
    fresh = [f for f in pool if f not in used] or pool
    k = i % len(fresh)
    cands = fresh[k:] + fresh[:k]
    scored = [(f, *_best_upright(text, f, None, fmt, aspect)) for f in cands]
    new = next((f for f, b, ok in scored if b and ok and b["cap"] >= min_cap), None)
    if new is None:
        new = max(scored, key=lambda r: (r[2], r[1]["cap"] if r[1] else 0.0))[0]
    w["font"] = new
    if w.get("font_pin"):
        w["font_pin"] = new
    for pn in (t.get("font_pins") or []):
        if isinstance(pn, dict) and pn.get("font") and _norm(pn.get("text")) == _norm(w.get("text")):
            pn["font"] = new
    if w.pop("axes", None) is not None and fixes is not None:
        fixes.append(f"type.words[{i}].axes dropped (they belonged to {(fl.lookup(old) or {}).get('key') or old})")
    if fixes is not None:
        ro, rn = (fl.lookup(old) or {}), (fl.lookup(new) or {})
        fixes.append(f"type.words[{i}].font {ro.get('key') or old} ({ro.get('role', '?')}, {why}) -> {rn.get('key') or new} ({rn.get('role', '?')}) "
                     f"(the user asked for more legible text)")
    return True


def boost_layout(t: dict, fmt: str = "9:16", fixes: list | None = None, min_cap: float = MIN_CAP, split_only: bool = False, keep=()) -> int:
    """see the module doc. In place on a design `type` object. -> number of words changed (a swapped face and / or a re-laid layout).
    split_only: touch ONLY words split mid-word (no risky-face swap, no rolled / small re-fit); keep: word indexes whose mid-word split stays (the rare option)."""
    from tools.fusion import stable_type as ST
    aspect = 16.0 / 9.0 if fmt == "16:9" else 9.0 / 16.0
    n = 0
    for i, w in enumerate((t or {}).get("words") or []):
        if not isinstance(w, dict):
            continue
        text = " ".join(str(w.get("text", "")).split())
        toks = text.split()
        if not toks:
            continue
        split = _mid_split(w, toks)
        if split_only and (not split or i in keep):
            continue
        swapped = False
        if user_pinned(t, w):
            pass                                                           # the user's own face wins over the legibility swap (the layout is still made of whole words)
        elif is_risky(w.get("font") or w.get("font_pin")) and not split_only:
            swapped = _swap_font(t, i, w, fixes, "risky", fmt, min_cap)
        elif split and not _best_upright(text, w.get("font"), w.get("axes"), fmt, aspect)[1]:
            swapped = _swap_font(t, i, w, fixes, "whole words are under the size floor in this face", fmt, min_cap)      # no whole-word layout reads at the floor: a face that does
        pl = w.get("placement") if isinstance(w.get("placement"), dict) else {}
        roll = float(pl.get("roll") or 0.0)
        rolled = abs(roll) >= 45.0
        cur = "grid" if split else (("rot90" if roll > 0 else "rot-90") if rolled else ("stack" if len(w.get("lines") or []) > 1 else "line"))
        boxes = {c: ST.fit_box(text, w.get("font"), w.get("axes"), c, aspect) for c in ST.allowed(text, fmt) if c not in ST.SPLIT_LAYOUTS}
        ok = [b for b in boxes.values() if b["ok"]]
        upright = [b for b in ok if not b["roll"]]
        pick = None
        if split:
            cur_cap = 0.0
            for c in ST.SPLIT_LAYOUTS:
                if c in ST.allowed(text, fmt):
                    cur_cap = float(ST.fit_box(text, w.get("font"), w.get("axes"), c, aspect)["cap"])
                    break
            flat = [b for b in boxes.values() if not b["roll"]]
            pool_ = upright or ok or flat
            pick = max(pool_, key=lambda b: b["cap"]) if pool_ else None
        else:
            if cur not in boxes:
                n += 1 if swapped else 0
                continue
            cur_cap = float(boxes[cur]["cap"])
            if rolled and upright:
                best = max(upright, key=lambda b: b["cap"])
                if best["cap"] >= cur_cap * UPRIGHT_KEEP:
                    pick = best
            elif cur_cap + 1e-9 < min_cap and ok:
                best = max(ok, key=lambda b: b["cap"])
                if best["layout"] != cur and best["cap"] > cur_cap * GAIN:
                    pick = best
            if pick is None and swapped:
                pick = boxes[cur]                                          # same layout, re-fitted to the new face's glyph widths
        if pick is None:
            n += 1 if swapped else 0
            continue
        w["placement"] = pl
        if len(toks) > 1 or w.get("lines"):
            w["lines"] = list(pick["lines"])
        pl["roll"], pl["wf"], pl["hf"] = pick["roll"], pick["wf"], pick["hf"]
        if fixes is not None and (split or pick["layout"] != cur):
            why = ("a mid-word split is rare (at most one word in 1 piece in %d): re-laid as whole words" % MIDWORD_ONE_IN if split_only else "a mid-word split is re-laid as whole words") if split else "the user asked for more legible text"
            fixes.append(f"type.words[{i}] layout {cur} -> {pick['layout']} ({why}): {'/'.join(pick['lines'])}, roll {pick['roll']:g}, "
                         f"cap {cur_cap:.1%} -> {pick['cap']:.1%} of the frame height")
        n += 1
    return n


def split_words(t: dict) -> list:
    """the indexes of the words whose rows break inside a word."""
    out = []
    for i, w in enumerate((t or {}).get("words") or []):
        toks = " ".join(str((w or {}).get("text", "")).split()).split() if isinstance(w, dict) else []
        if toks and _mid_split(w, toks):
            out.append(i)
    return out


def _seed_key(piece: dict, seed=None) -> str:
    """what identifies the piece for the rare-split draw: the explicit seed (own-video route: --color-seed), else plan.seed / request.seed / seed / id, else the words."""
    if seed is not None:
        return f"s{seed}"
    plan, req = piece.get("plan"), piece.get("request")
    for v in ((plan or {}).get("seed") if isinstance(plan, dict) else None, (req or {}).get("seed") if isinstance(req, dict) else None, piece.get("seed"), piece.get("id")):
        if v is not None and v != "":
            return f"s{v}"
    return "w" + "|".join(str(w.get("text", "")) for w in ((piece.get("type") or {}).get("words") or []) if isinstance(w, dict))


def keep_index(piece: dict, n_split: int, seed=None):
    """which of the piece's n_split split words keeps its split, or None. A pure function of the piece's seed: about 1 piece in MIDWORD_ONE_IN keeps exactly one."""
    import hashlib
    if n_split <= 0:
        return None
    h = int(hashlib.md5(("midword|" + _seed_key(piece, seed)).encode("utf-8")).hexdigest()[:8], 16)
    return (h // MIDWORD_ONE_IN) % n_split if h % MIDWORD_ONE_IN == 0 else None


def relay_midword(piece: dict, fixes: list | None = None, seed=None) -> int:
    """the general pass, every piece without the legibility wish (a boost piece keeps none, see boost_layout): at most ONE mid-word split stays, in about 1 piece in 4 (seeded);
    every other split word is re-laid as whole words (the bigger cap that passes the size floor). A user-pinned font stays. -> words re-laid."""
    t = piece.get("type") if isinstance(piece, dict) else None
    if not isinstance(t, dict):
        return 0
    sw = split_words(t)
    if not sw:
        return 0
    k = keep_index(piece, len(sw), seed)
    keep = {sw[k]} if k is not None else set()
    from tools.fusion import design_defaults as DD
    return boost_layout(t, str(piece.get("format") or "9:16"), fixes, float(DD.CAP_FLOOR), split_only=True, keep=keep)


def apply(piece: dict, fixes: list | None = None, seed=None) -> int:
    """EVERY piece: with the user's wish for legible text (piece.user_style / direction / request.direction) boost_layout (no mid-word split at all); without it relay_midword
    (at most one rare mid-word split, about 1 piece in 4). Changes also land in piece["user_style_log"] and `fixes`. -> words changed."""
    from tools.fusion import user_style as US
    if not isinstance(piece, dict) or not isinstance(piece.get("type"), dict):
        return 0
    log: list = []
    if US.parse_direction(piece)["legible_boost"]:
        n = boost_layout(piece["type"], str(piece.get("format") or "9:16"), log)
    else:
        n = relay_midword(piece, log, seed)
    if log:
        if not isinstance(piece.get("user_style_log"), list):
            piece["user_style_log"] = []
        piece["user_style_log"].extend(log)
        if fixes is not None:
            fixes.extend(log)
    return n


def apply_json(piece_json):
    """the design nodes' string form -> (piece_json, fixes). Unreadable / nothing to change = the input string, []."""
    try:
        p = json.loads(piece_json)
    except (TypeError, ValueError):
        return piece_json, []
    fixes: list = []
    if apply(p, fixes) and fixes:
        return json.dumps(p, ensure_ascii=False), fixes
    return piece_json, []
