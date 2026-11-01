"""random_system: the seeded RANDOM SYSTEM drawn at the start of every piece (user 2026-10-30: "can we set it as random? it chooses transitions and a range of fonts at the start with a
limited palette discerned from the imagery"). Pure functions, no ComfyUI / network / Claude. piece_system.choose() calls these and stores the result in the system dict, so a piece
carries it in `plan.piece_system` and a re-render with the same seed is exact.

    draw_cuts(seed, tone, earlier)     -> {kinds [2-3 cut kinds], styles {kind: style | None}, order [seeded order of the kinds], recent_excluded [..]}
                                          kinds + styles come from the FULL menu (menu()); the kinds used by the previous RECENT_PIECES systems of the batch are excluded
    draw_fonts(seed, tone, earlier, texts) -> 3-4 liked font files (Anybody in the pool at its normal weight, never forced); fonts that fail the legibility screen at the size the
                                          words need are skipped; the shortlist keeps one reader, one characterful face and contrast between neighbours
    drawn_kinds(system)                -> the cut kinds a piece may use ([] = no draw: the old, fixed vocabulary)
    pick_next(system, used, ...)       -> the next drawn kind (seeded order, least used first) for a fallback / no-repeat swap; never a fixed depth_pivot-first or fly

The Gemma / Qwen design stays free INSIDE the draw: which words, timing, layout and which of the drawn transitions go at which boundary.
"""
from __future__ import annotations

import random

from tools.fusion import cut_kinds as CK

VERSION = 1
RECENT_PIECES = 2             # the kinds of the last two pieces of the batch are not drawn again
N_KINDS = (2, 3)              # a piece draws 2 or 3 cut kinds
FONTS_MIN, FONTS_MAX, FONTS_DEFAULT_LEN = 2, 6, 16.0      # the number of faces scales with the piece length (user 2026-10-30): n = clamp(1 + ceil(length_s / 8), 2, 6)


def n_fonts(length_s=None) -> int:
    """faces drawn for a piece of `length_s` seconds: 6 s -> 2, 12 s -> 3, 16 s -> 3, 18-24 s -> 4, 30 s -> 5, 48 s and more -> 6 (none known: a 16 s piece)."""
    import math
    try:
        L = float(length_s)
    except (TypeError, ValueError):
        L = FONTS_DEFAULT_LEN
    if not L > 0:
        L = FONTS_DEFAULT_LEN
    return int(min(max(1 + math.ceil(L / 8.0 - 1e-9), FONTS_MIN), FONTS_MAX))

# kind -> (weight, family, styles). The 3D orbit kinds have three looks (cut_kinds.STYLES); the pivot only its chunks; windows / pans / hard have none.
# Windows are weighted below the 3D kinds (they need a heavy word and a held beat) but they ARE drawn: 1 piece of 12 reached a window on the screen before the draw.
_BASE = {
    "orbit:push": (2.2, "orbit", ("chunks",)),            # 2026-10-30: `smooth` (one whole-frame card) is no longer drawn: broken-up chunks only; user: "a simple straight push in or pull out" -> the plain straight moves
    "orbit:pullback": (2.2, "orbit", ("chunks",)),        # (zero orbit / yaw / roll, no mid-way rise), chunk mosaic only (a lateral `slide` is not straight), weighted up; fly keeps chunks + slide
    "orbit:fly": (1.0, "orbit", ("chunks", "slide")),
    "depth_pivot": (1.3, "pivot", ("chunks",)),
    "pan:left": (0.7, "pan", ()), "pan:right": (0.7, "pan", ()), "pan:up": (0.5, "pan", ()), "pan:down": (0.5, "pan", ()),
    "window_other": (1.2, "window", ()), "window_cut": (0.9, "window", ()), "window_swap": (0.8, "window", ()), "window_zoom": (1.1, "window", ()),
    "circle_grid": (4.0, "explore", ()), "depth_lines": (4.0, "explore", ()),    # 2026-10-30 promoted explore cuts (word forms: B's word is owned by the cut, pivot_cut.EXPLORE_*)
    "hard": (0.15, "plain", ()),
}
READER_ROLES = ("heavy", "tall", "text")      # roles that carry a reading word cleanly (the liked-font rule: legible first)
FAMILY_REPEAT = 0.35          # weight multiplier of a kind whose family is already drawn
STYLE_WEIGHT = {"smooth": 1.5, "chunks": 1.0, "slide": 1.0}


def menu(slide_on=None, dolly_on=None) -> dict:
    """{kind: (weight, family, styles)} the full draw menu now: push / pullback / fly x smooth / chunks / slide (slide only while design_options.SLIDE_ON), depth_pivot, pan:*, depth_dolly:* (only
    while DOLLY_ON), window_other / window_cut / window_swap / window_zoom, hard."""
    from tools.fusion import design_options as DO
    slide_on = DO.SLIDE_ON if slide_on is None else slide_on
    dolly_on = DO.DOLLY_ON if dolly_on is None else dolly_on
    out = {}
    for k, (w, fam, styles) in _BASE.items():
        if k not in CK.ALL:                                       # window_zoom exists once cut_kinds knows it
            continue
        st = tuple(s for s in styles if s != "slide" or slide_on)
        out[k] = (w, fam, st)
    if dolly_on:
        for d in CK.DOLLY_DIRS:
            out[f"depth_dolly:{d}"] = (0.5, "dolly", ())
    return out


def _rng(seed, field: str, tone: str = "") -> random.Random:
    return random.Random(f"rsys:{int(seed)}:{tone}:{field}")


def _wpick(rng, items, weights):
    items, weights = list(items), [max(float(w), 0.0) for w in weights]
    tot = sum(weights)
    if not items:
        return None
    if tot <= 0:
        return items[rng.randrange(len(items))]
    x = rng.random() * tot
    for it, w in zip(items, weights):
        x -= w
        if x <= 0:
            return it
    return items[-1]


def kinds_of(system) -> list:
    """the cut kinds a (system dict | None) draws; [] when it has none."""
    c = (system or {}).get("cuts") if isinstance(system, dict) else None
    return [k for k in (c or {}).get("kinds", []) if isinstance(k, str)] if isinstance(c, dict) else []


def _recent(earlier, n=RECENT_PIECES) -> list:
    out = []
    for s in [e for e in (earlier or []) if isinstance(e, dict)][-n:]:
        out += kinds_of(s)
    return out


def robust(kind: str) -> bool:
    """a kind that always renders at a place change if the shot lengths allow it (3D orbit / pivot); window and pan kinds need a fitting word and can fail."""
    return CK.family_of(kind) == "3d"


def placeable(kind: str) -> bool:
    """a kind the planning rules may put at a place change by themselves: the 3D kinds and the promoted explore kinds (circle_grid, depth_lines: they need a short flat word of B, else the cut runs
    word-less / falls back to a plain cut at render time). `robust` stays the 3D kinds only (they never fail)."""
    return CK.family_of(kind) in ("3d", "explore")


def draw_cuts(seed, tone: str = "", earlier=(), n: int | None = None, slide_on=None, dolly_on=None) -> dict:
    """2-3 cut kinds and a style for each, from the FULL menu. Exclusions: the kinds of the previous RECENT_PIECES systems in `earlier` (relaxed one piece at a time if the menu would run
    short). At least one drawn kind is a 3D kind (a window or pan kind can fail on a word; the fallback is then another DRAWN kind). Deterministic in (seed, tone, earlier)."""
    mn = menu(slide_on, dolly_on)
    rng = _rng(seed, "cuts", tone)
    want = n if n is not None else rng.randint(*N_KINDS)
    ex = list(dict.fromkeys(_recent(earlier)))
    pool = [k for k in mn if k not in ex]
    if len(pool) < want + 1:                                      # relax: only the newest piece's kinds stay excluded, then none
        for keep in (1, 0):
            ex = list(dict.fromkeys(_recent(earlier, keep)))
            pool = [k for k in mn if k not in ex]
            if len(pool) >= want + 1:
                break
    kinds: list = []
    while len(kinds) < want and len(kinds) < len(pool):
        cand = [k for k in pool if k not in kinds]
        fams = {mn[k][1] for k in kinds}
        k = _wpick(rng, cand, [mn[c][0] * (FAMILY_REPEAT if mn[c][1] in fams else 1.0) for c in cand])
        kinds.append(k)
    if kinds and not any(robust(k) for k in kinds):               # guarantee one kind that always renders
        r3 = [k for k in pool if robust(k) and k not in kinds] or [k for k in mn if robust(k) and k not in kinds]
        if r3:
            kinds[-1] = _wpick(rng, r3, [mn[c][0] for c in r3])
    styles = {}
    used_styles = []
    for k in kinds:
        sts = mn[k][2]
        if not sts:
            styles[k] = None
            continue
        st = _wpick(rng, sts, [STYLE_WEIGHT.get(s, 1.0) * (0.5 if s in used_styles else 1.0) for s in sts])
        styles[k] = st
        used_styles.append(st)
    order = list(kinds)
    rng.shuffle(order)
    return dict(version=VERSION, kinds=kinds, styles=styles, order=order, recent_excluded=ex)


def style_for(system, kind):
    """the drawn style of a kind (None when the kind has none / is not drawn)."""
    c = (system or {}).get("cuts") if isinstance(system, dict) else None
    return ((c or {}).get("styles") or {}).get(kind) if isinstance(c, dict) else None


def pick_next(system, used=(), feasible=None, prefer=None, among=None) -> str | None:
    """the next drawn kind for a boundary: kinds not in `used` first (in the system's seeded `order`), then the least used one; `feasible(kind)` filters (a window kind that cannot read,
    a 3D kind the shot lengths cannot carry); `prefer` = a kind to put first when it is drawn and free (e.g. the orbit mode the music asks for); `among` restricts the drawn kinds. None when nothing is drawn / fits."""
    ks = kinds_of(system)
    if among is not None:
        ks = [k for k in ks if k in among]
    if not ks:
        return None
    c = (system or {}).get("cuts") or {}
    order = [k for k in (c.get("order") or ks) if k in ks] or ks
    if prefer in order:
        order = [prefer] + [k for k in order if k != prefer]
    used = list(used)
    ok = [k for k in order if feasible is None or feasible(k)]
    if not ok:
        return None
    free = [k for k in ok if k not in used]
    if free:
        return free[0]
    return min(ok, key=lambda k: (used.count(k), order.index(k)))


# ---------------------------------------------------------------------------------------------------------------------------------------------- fonts
def _legible_at(entry, texts, aspect=0.5625) -> bool:
    """can the face carry its longest word at the legibility floor? One line at the safe width, or the same word read vertically (rotated: the whole frame height is the line)."""
    if not texts:
        return True
    try:
        from tools.fusion import design_defaults as DD
        from tools.fusion import design_options as DO
    except Exception:       # noqa: BLE001
        return True
    best = None
    for t in texts:
        for line in str(t).split():
            letters = "".join(ch for ch in line if ch.isalnum())
            if not letters:
                continue
            r = DO.line_width_per_cap(entry["file"], None, letters)
            if r is None:
                return True
            best = r if best is None else max(best, r)
    if best is None:
        return True
    cap_h = DD.MAX_WF * aspect / best                      # one horizontal line at the safe width
    cap_v = 0.84 / best                                    # read vertically along the frame height (cap measured in frame heights)
    return max(cap_h, min(cap_v, 0.84 * aspect)) + 1e-9 >= DD.CAP_FLOOR


EXPRESSIVE_ROLES = ("pixel", "ornamental", "decon", "blackletter")
EXPRESSIVE_SHARE = 0.6         # share of a piece's faces that are characterful (role pixel / ornamental / decon / blackletter); the rest are readers
PLAIN_FACES = ("SpaceGrotesk-Bold.ttf", "MajorMonoDisplay-Regular.ttf")        # liked but plain: drawn a quarter as often (one is fine as a quiet counterpoint)
PLAIN_DAMP = 0.25


def draw_fonts(seed, tone: str = "", earlier=(), texts=(), pool=None, expr_weights=None, reader_weights=None, need_heavy: bool = False, length_s=None) -> list[str]:
    """n_fonts(length_s) (2-6, by the piece length) font FILES from mograph/fonts/liked.json (the legibility-safe pool of piece_system: no hard-to-read faces; Anybody is in it at its normal weight, drawn like any other). One
    characterful face (role pixel / ornamental / decon / blackletter, by tone), one reader (heavy / tall / text), then faces whose role style CONTRASTS with the previous pick
    (liked-font rule: neighbouring lines must contrast). Faces used by the previous pieces are drawn less (REPEAT_PENALTY). Every face must carry the longest word at the size floor."""
    from mograph import fonts_liked as fl
    from tools.fusion import piece_system as PS
    pool = list(pool if pool is not None else PS._font_pool())
    pool = [e for e in pool if _legible_at(e, texts)] or list(pool)
    used = set(PS._used(earlier, "fonts"))
    rng = _rng(seed, "fonts", tone)
    n = n_fonts(length_s)

    def w(e):
        return PS.REPEAT_PENALTY if e["file"] in used else 1.0
    expr_w = expr_weights or PS.EXPR_ROLE_WEIGHTS[tone or "bright"]
    read_w = {r: x for r, x in (reader_weights or PS.READER_ROLE_WEIGHTS[tone or "bright"]).items() if r in READER_ROLES}           # the reader of the shortlist is a plainly legible face
    if need_heavy:                                                      # a window cut / inverse word needs a fat face (design_options.FAT_ROLES): the reader is one
        read_w = {"heavy": 1.0}
    chosen: list = []
    cand = [e for e in pool if e["role"] in expr_w]
    if cand:
        chosen.append(_wpick(rng, cand, [expr_w[e["role"]] * w(e) for e in cand]))
    cand = [e for e in pool if e["role"] in read_w and e not in chosen]
    if cand:
        chosen.append(_wpick(rng, cand, [read_w[e["role"]] * w(e) * (PLAIN_DAMP if e["file"] in PLAIN_FACES else 1.0) for e in cand]))
    elif need_heavy:
        cand = [e for e in PS._font_pool() if e["role"] == "heavy"]
        chosen.append(_wpick(rng, cand, [1.0] * len(cand)))
    want_expr = max(1, int(round(n * EXPRESSIVE_SHARE)))               # 2026-10-30 (user on v027 r02: "some of the fonts used look a little plain and not part of the favourites"): most of a piece's faces are the characterful ones
    while len(chosen) < n:
        prev = chosen[-1]
        rest = [e for e in pool if e not in chosen]
        if not rest:
            break
        have_expr = sum(1 for e in chosen if e["role"] in EXPRESSIVE_ROLES)
        if have_expr < want_expr:
            rest_e = [e for e in rest if e["role"] in EXPRESSIVE_ROLES]
            rest = rest_e or rest
        contrast = [e for e in rest if fl.contrast_axes(prev["role"], e["role"]) and e["role"] != prev["role"]]
        cand = contrast or [e for e in rest if e["role"] != prev["role"]] or rest
        chosen.append(_wpick(rng, cand, [w(e) * (PLAIN_DAMP if e["file"] in PLAIN_FACES else 1.0) for e in cand]))
    files = [e["file"] for e in chosen]
    rng.shuffle(files)
    # neighbours contrast: reorder so that no two consecutive faces share a role style (a stable reading order for the A B A C pattern)
    return _contrast_order(files)


def _contrast_order(files: list) -> list:
    from mograph import fonts_liked as fl
    best, best_bad = list(files), 10 ** 6
    import itertools
    for perm in itertools.permutations(files):
        bad = sum(1 for a, b in zip(perm, perm[1:]) if not fl.contrast_axes(fl.role_of(a), fl.role_of(b)))
        if bad < best_bad:
            best, best_bad = list(perm), bad
            if bad == 0:
                break
    return best


# ---------------------------------------------------------------------------------------------------------------------------------------------- outline as a stand-out option
# User 2026-10-30: the outline that helped the graphics stand out "at points" in 26_own-video/v002 (redesign_r1) is legible_color's thin OUTLINE SUPPORT (an outline in the opposite luminance of the
# type, eased in and out, tools/fusion/legible_color.py). That render got it only where the plate failed the contrast target. Here it is a drawn option of the piece: at SOME words or hits the
# same outline is forced on (never on every word), by the same op (`legible_color` arg `outline` = windows in piece seconds); nothing new is drawn.
OUTLINE_MODES = {"none": 0.5, "hits": 0.5}      # 2026-10-30 (user: "we dont need to outline everything"): the random draw is none | hits only; `words` (a subset for the whole hold) stays reachable through the `effects` field (effects_menu.apply_effects)
OUTLINE_SHARE = 0.4           # share of the words that carry the outline in `words` mode (at least one, never all when there are two or more)
HITS_SHARE = 0.34            # share of the words that get the one-beat outline pulse in `hits` mode: at most ceil(n / 3) words (never every word)
OUTLINE_PULSE_BEATS = 1.0     # length of the outline on a hit (beats) in `hits` mode


def draw_outline(seed, tone: str = "") -> dict:
    """{mode: none | words | hits, share} of the piece: `words` = the outline on a seeded subset of the words for their whole hold; `hits` = a one-beat outline pulse as a word lands (on about 60 % of
    the words); `none` = only the adaptive outline legible_color adds where the plate fails the contrast target."""
    rng = _rng(seed, "outline", tone)
    mode = _wpick(rng, list(OUTLINE_MODES), list(OUTLINE_MODES.values()))
    return dict(mode=mode, share=OUTLINE_SHARE if mode == "words" else HITS_SHARE if mode == "hits" else 0.0)


def outline_windows(spec, holds, bpm=None, seed=0) -> list:
    """[[t0, t1], ...] piece seconds where the outline is forced on. spec = draw_outline(); holds = [(hit s, release s)] per word. One word: `words` mode becomes a pulse (an outline over
    the only word's whole hold would be every word). Deterministic in (seed, spec)."""
    mode = (spec or {}).get("mode", "none")
    n = len(holds or [])
    if mode == "none" or n == 0:
        return []
    rng = random.Random(f"outline:{int(seed)}:{n}")
    beat = 60.0 / float(bpm) if bpm else 0.6
    if mode == "words" and n >= 2:
        k = min(max(1, int(round(float((spec or {}).get("share", OUTLINE_SHARE)) * n))), n - 1)
        pick = sorted(rng.sample(range(n), k))
        return [[round(float(holds[i][0]), 3), round(float(holds[i][1]), 3)] for i in pick]
    share = float((spec or {}).get("share", HITS_SHARE)) if mode == "hits" else HITS_SHARE
    k = min(max(1, int(round(share * n))), -(-n // 3))                  # at most ceil(n / 3) words carry the outline pulse
    pick = sorted(rng.sample(range(n), k))
    out = []
    for i in pick:
        hit, rel = float(holds[i][0]), float(holds[i][1])
        out.append([round(hit, 3), round(min(hit + OUTLINE_PULSE_BEATS * beat, rel), 3)])
    return [w for w in out if w[1] > w[0]]
