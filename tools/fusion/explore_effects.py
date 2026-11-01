"""explore_effects: the user's HEARTED looks (16_app/core_check/explore_masks x14 / x08 / x10 / x03 / x02) as named effects the designer (Gemma + the random system) can choose, emitted as compose
recipes (2026-10-30; user: "turn the hearted looks into things the DESIGNER can choose, with instructions"). The ops themselves are tools/fusion/explore_ops.py (not edited here); the artist's round-2 ops
(explore_ops2.py) plug in through ``register`` / the OPS merge in compose._ops_registry.

    NAMES / SPECS      name -> dict(kind section | form_in, needs (matte | plate), length (min s, max s), weight, text)
    draw(seed, earlier, requested)      the names a piece draws (0-2, favouring these; a request forces its names in)
    place(piece, names, holds, seed)    -> [{name, shot, t0, t1, word}]: a section on a shot that has what it needs, inside one or two word holds (2-4 s), clear of the shot edges
    weave(piece, music, chk, size)      the hook (plan_rules._finish and design_step._finish): builds each placed effect's ops and gates them into the shot's compose chain
    OPS                                  extra compose ops (``chrono_echoes``: the subject's matte + footage a few frames earlier, shifted back)

LEGIBILITY: the word of a section reads from its hit: letters-as-windows go through knockout(min_contrast) (the window must out-shine the surround, else the letters turn solid), the word under the
echoes is drawn ABOVE them (an echo never covers a read frame), the live subject is the only thing in front of a word. A section never touches the first / last 0.8 s of its shot (the 3D cuts).
"""
from __future__ import annotations

import copy
import random

import numpy as np

SECTION_MARGIN_S = 0.8        # a section stays this far from both ends of its shot (cuts / transitions live there)
GATE_S = 0.2                  # fade of the section in and out of the shot's own chain
MAX_PER_PIECE = 2

SPECS = {
    "letters_hold_subject": dict(kind="section", needs=("matte",), length=(2.0, 4.0), weight=3.0,
                                 text="plain pale ground; the whole shot (subject included) exists only INSIDE the letters, outside them the subject carries on as a live outline (x14)"),
    "chronophoto_echoes": dict(kind="section", needs=("matte",), length=(2.0, 4.0), weight=3.0,
                               text="plain ground; the moving subject leaves a trail of time-offset echoes behind the word, the live subject steps out in front of it (x08)"),
    "echo_trail": dict(kind="section", needs=("matte",), length=(2.0, 4.0), weight=3.0,
                       text="plain ground; half-beat time echoes of the subject (filled with the CLEAN PLATE when the shot has one, else its own footage) trail BEHIND a solid word, the live subject steps out in front of it; echoes are never in front of the word (r2a2)"),
    "outline_to_word": dict(kind="form_in", needs=("matte",), length=(1.6, 2.6), weight=3.0,
                            text="a word's FORM-IN: the subject's contour lines morph into the outline of the word, then the letters open as windows onto the shot (x10)"),
    "outline_over_word": dict(kind="section", needs=("matte",), length=(2.0, 4.0), weight=2.0,
                              text="dark ground, the word solid behind; the subject's silhouette is a graphic window with an animated outline running round it (x03)"),
    "plate_letters_subject_over": dict(kind="section", needs=("matte", "plate"), length=(2.0, 4.0), weight=2.0,
                                       text="plain ground; the letters are windows onto the CLEAN PLATE and the subject (with its dust) is matted over the top (x02)"),
}
NAMES = tuple(SPECS)
ALIASES = {"x14": "letters_hold_subject", "subject_escapes_letters": "letters_hold_subject", "x08": "chronophoto_echoes", "chronophoto": "chronophoto_echoes", "echoes": "chronophoto_echoes",
           "r2a2": "echo_trail", "echo_plate_cutouts_behind_word": "echo_trail", "echo_behind": "echo_trail", "echoes_behind": "echo_trail", "echo_trail_behind": "echo_trail",
           "x10": "outline_to_word", "outline_becomes_word": "outline_to_word", "x03": "outline_over_word", "subject_holds_other_plate": "outline_over_word",
           "x02": "plate_letters_subject_over", "cross_shot_letters": "plate_letters_subject_over"}


def norm(name):
    k = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    k = ALIASES.get(k, k)
    return k if k in SPECS else None


def draw(seed, earlier=(), requested=()) -> list[str]:
    """0-2 names drawn for a piece (weights favour these looks; a name an earlier piece used is drawn a third as often) + every requested name (forced). Deterministic in the seed."""
    rng = random.Random(f"explore:{int(seed or 0)}")
    used = {n for s in list(earlier or [])[-2:] if isinstance(s, dict) for n in ((s.get("effects") or {}).get("explore") or [])}
    k = rng.choices([0, 1, 2], weights=[0.25, 0.45, 0.30])[0]
    pool = [n for n in NAMES if n != "plate_letters_subject_over"]            # the plate look needs a plated shot: only when asked for / when the piece has plates (place() drops it otherwise)
    drawn: list = []
    for _ in range(k):
        cand = [n for n in pool if n not in drawn]
        if not cand:
            break
        w = [SPECS[n]["weight"] * (0.33 if n in used else 1.0) for n in cand]
        drawn.append(rng.choices(cand, weights=w)[0])
    forced = [n for n in (norm(x) for x in requested or []) if n]
    return list(dict.fromkeys(forced + drawn))


# ------------------------------------------------------------------------------------------------------------------------ colours
def _rgb01(c, default):
    try:
        return tuple(float(x) / (255.0 if max(c) > 1.0 else 1.0) for x in c[:3])
    except (TypeError, ValueError):
        return default


def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def colours(piece: dict) -> dict:
    """ground / ink / accent of the piece's palette (0..1): a pale or dark ground by the piece's own bg, ink = the opposite, accent = the most chromatic accent."""
    pal = (((piece.get("plan") or {}).get("piece_system") or {}).get("palette")) or (piece.get("type") or {}).get("palette") or {}
    bg = _rgb01(pal.get("bg") or (10, 10, 12), (0.04, 0.04, 0.05))
    fg = _rgb01(pal.get("fg") or (235, 235, 230), (0.92, 0.92, 0.9))
    acc = [_rgb01(a, (0.9, 0.5, 0.2)) for a in (pal.get("accents") or [])] or [(0.9, 0.5, 0.2)]
    acc = max(acc, key=lambda c: max(c) - min(c))
    dark = bg if _lum(bg) < _lum(fg) else fg
    light = fg if _lum(fg) > _lum(bg) else bg
    pale = tuple(min(1.0, 0.82 + 0.1 * v) for v in light)           # a pale paper ground (never pure white)
    black = tuple(0.035 + 0.1 * v for v in dark)
    return dict(pale=pale, black=black, ink=tuple(0.5 * v for v in dark), light=light, accent=acc)


# ------------------------------------------------------------------------------------------------------------------------ the recipes
def _tl(*pts):
    return {"timeline": [[round(float(a), 3), round(float(b), 3)] for a, b in pts], "ease": "smooth"}


def build_ops(name: str, ctx: dict) -> list[dict]:
    """the ops that make buffer ``fx`` (the effect's whole picture) from the chain's inputs ($footage, $matte via ``fxm``, $type_matte). ctx: colours, t0, t1, seed, plated, size."""
    c, seed = ctx["colours"], int(ctx.get("seed", 0))
    t0, t1 = float(ctx["t0"]), float(ctx["t1"])
    m = "$fxm"
    ops: list[dict] = []
    if name == "contour_morph":                                       # tools/fusion/contour_morph.py (the default word-to-word move; not an explore SECTION)
        from tools.fusion import contour_morph as CM
        return CM.build_ops(ctx)
    if name == "letters_hold_subject":
        ops = [{"op": "knockout", "out": "fx", "args": {"plate_rgb": c["pale"], "type_a": "$type_matte", "footage": "$footage", "min_contrast": 0.22}},
               {"op": "brush_edge", "out": "fx", "args": {"img": "$fx", "mask": m, "frame_idx": "$frame_idx", "color": c["ink"], "n_strokes": 3, "width_px": 2.4, "seed": seed, "avoid": "$type_matte"}}]
    elif name in ("chronophoto_echoes", "echo_trail"):
        # ORDER IS THE RULE (r2a3 hurt legibility): ground -> echoes -> the WORD (solid, above every echo) -> the live subject in front. Echo spacing = half a beat of the piece's tempo.
        delays, shifts, alphas = echo_plan(ctx.get("half_beat_f"), 4 if name == "echo_trail" else 3)
        plated = bool(ctx.get("plated")) and name == "echo_trail"
        args = {"base": "$fx", "footage": "$footage", "matte": m, "frame_idx": "$frame_idx", "state_key": f"echo:{seed}", "delays": delays, "shifts": shifts, "alphas": alphas}
        if plated:
            args["fill"] = "$footage"                                  # plated shots: $footage IS the clean plate (plate_layers), the subject layer is $subject
        ops = [{"op": "fill", "out": "fx", "args": {"like": "$footage", "color": c["pale"]}},
               {"op": "chrono_echoes", "out": "fx", "args": args},
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": c["ink"], "alpha": "$type_matte"}},                       # the word is drawn ABOVE the echoes: no echo covers a read frame
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": "$subject" if plated else "$footage", "alpha": m}}]      # the live subject steps out in front
    elif name == "outline_to_word":
        sc = min(1.0, max(0.5, (t1 - t0) / 2.2))                         # a section shortened by the relaxed placement runs the same morph faster
        a, b, d = t0, t0 + 1.1 * sc, t0 + 1.2 * sc
        ops = [{"op": "sdf_morph", "out": "fxsh", "args": {"a": m, "b": "$type_matte", "k": _tl((a - 0.01, 0.0), (a + 0.25, 0.0), (b + 0.25, 1.0), (t1 + 1.0, 1.0)), "soft": 1.0}},
               {"op": "mask_gate", "out": "fxwin", "args": {"a": "$type_matte", "gate": _tl((d - 0.01, 0.0), (d + 0.4, 1.0), (t1 + 1.0, 1.0))}},
               {"op": "fill", "out": "fx", "args": {"like": "$footage", "color": c["black"]}},
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": "$footage", "alpha": "$fxwin"}},
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": "$footage", "alpha": m}},
               {"op": "brush_edge", "out": "fx", "args": {"img": "$fx", "mask": "$fxsh", "frame_idx": "$frame_idx", "color": c["accent"], "n_strokes": 2, "width_px": 2.2, "seed": seed, "guard": False}}]
    elif name == "outline_over_word":
        ops = [{"op": "fill", "out": "fx", "args": {"like": "$footage", "color": c["black"]}},
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": c["accent"], "alpha": "$type_matte"}},
               {"op": "duotone", "out": "fxduo", "args": {"img": "$footage", "dark": c["black"], "light": c["light"], "contrast": 1.2}},
               {"op": "over", "out": "fx", "args": {"base": "$fx", "fg": "$fxduo", "alpha": m}},
               {"op": "brush_edge", "out": "fx", "args": {"img": "$fx", "mask": m, "frame_idx": "$frame_idx", "sample": "$footage", "n_strokes": 2, "seed": seed}}]
    elif name == "plate_letters_subject_over":
        sub = "$subject" if ctx.get("plated") else "$footage"
        ops = [{"op": "knockout", "out": "fx", "args": {"plate_rgb": c["black"], "type_a": "$type_matte", "footage": "$footage", "fg_matte": m, "fg": sub, "min_contrast": 0.22}}]
    return ops


# ------------------------------------------------------------------------------------------------------------------------ op: time-offset echoes
_HIST: dict = {}


def chrono_echoes(base, footage, matte, frame_idx=0, state_key="echo", delays=(24, 16, 8), shifts=(-0.27, -0.18, -0.09), alphas=(0.45, 0.6, 0.8), fill=None):
    """lay the subject (its matte + footage `delay` frames EARLIER, shifted by `shift` x the frame width) over `base`, oldest first = a Muybridge trail. History is a ring of the frames this op saw
    (the compose loop renders a shot's frames in order); before a delay is reached the oldest frame seen stands in, so a trail grows out of the subject instead of popping in.
    fill = an image (the CLEAN PLATE, playing) that fills every echo instead of the subject's own delayed footage; only the echo MASK is delayed + shifted (r2a2). Never flat colour."""
    b = np.asarray(base, np.float32)
    H, W = b.shape[:2]
    f = np.asarray(footage, np.float32)
    m = np.asarray(matte, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (H, W):
        import cv2
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    if f.shape[:2] != (H, W):
        import cv2
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
    fl_img = None
    if fill is not None:
        fl_img = np.asarray(fill, np.float32)
        if fl_img.shape[:2] != (H, W):
            import cv2
            fl_img = cv2.resize(fl_img, (W, H), interpolation=cv2.INTER_LINEAR)
    h = _HIST.setdefault(state_key, {})
    i = int(frame_idx)
    h[i] = (f.copy(), np.clip(m, 0.0, 1.0).copy())
    for old in [k for k in h if k < i - max(delays) - 2]:
        del h[old]
    out = b
    for d, sh, al in zip(delays, shifts, alphas):
        j = i - int(d)
        keys = sorted(k for k in h if k <= j)
        src = h[keys[-1]] if keys else h[min(h)]
        if not keys and i < int(d):                                     # the trail has not grown that far back yet
            continue
        import cv2
        M = np.float32([[1, 0, float(sh) * W], [0, 1, 0]])
        ef = cv2.warpAffine(src[0], M, (W, H), borderMode=cv2.BORDER_REPLICATE) if fl_img is None else fl_img
        em = cv2.warpAffine(src[1], M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0) * float(al)
        out = out * (1.0 - em[..., None]) + ef * em[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


ECHO_MAX_F = 30                  # the oldest echo is at most this many frames back (1.25 s): a slow tempo gets fewer echoes, not a smear across the whole shot
SHIFT_PER_F = -0.0111            # echo shift (share of the frame width) per frame of delay (x08 / r2a2: -4 px per frame at 360 wide)


def echo_plan(half_beat_f, n=4):
    """(delays, shifts, alphas) of n echoes at 1..n half-beats (`half_beat_f` frames; None = 6, 120 bpm), OLDEST first, dropping the oldest while it is further back than ECHO_MAX_F."""
    hb = max(2, int(round(float(half_beat_f)))) if half_beat_f else 6
    ks = [k for k in range(1, int(n) + 1) if k * hb <= ECHO_MAX_F] or [1]
    delays = [k * hb for k in reversed(ks)]
    shifts = [round(SHIFT_PER_F * d, 4) for d in delays]
    alphas = [round(0.55 + 0.12 * i, 3) for i in range(len(delays))]
    return delays, shifts, alphas


OPS = {"chrono_echoes": chrono_echoes}


# ------------------------------------------------------------------------------------------------------------------------ placement + weave
def _shot_spans(piece: dict) -> dict:
    from tools.fusion import recipe_eval as RE
    lay, _ = RE.piece_layout(piece)
    fps = float(piece.get("fps", 24))
    return {sid: (int(v["start"]) / fps, (int(v["start"]) + int(v["frames"])) / fps) for sid, v in lay.items()}


def _cands_for(name, spec, sid, a, b, holds, margin, minlen):
    """candidate sections of one effect on one shot [a, b] at one strictness (margin from the shot ends, shortest accepted section)."""
    lo, hi = spec["length"]
    u0, u1 = a + margin, b - margin
    out = []
    for wi, (h0, h1) in enumerate(holds):
        if spec["kind"] == "form_in":
            if h0 < u0 - (0.0 if margin >= SECTION_MARGIN_S else 0.3) or h0 > u1:
                continue
            t0 = max(u0, h0 - 1.2)
            t1 = min(u1, h0 + 1.4)
        else:
            if h1 <= u0 or h0 >= u1:
                continue                                              # this word's hold does not touch the usable part of the shot
            t0 = max(u0, h0 - 0.1)
            t1 = min(u1, max(h1 + 0.2, t0 + lo), t0 + hi)
            nxt = holds[wi + 1] if wi + 1 < len(holds) else None
            if t1 - t0 < lo and nxt and nxt[0] < u1:                  # one hold is short: take the next word too
                t1 = min(u1, nxt[1] + 0.2, t0 + hi)
        if minlen <= t1 - t0 <= hi + 0.05:
            out.append(dict(name=name, shot=sid, t0=round(t0, 3), t1=round(t1, 3), word=wi))
    return out


RELAX = ((SECTION_MARGIN_S, None), (0.45, 0.6), (0.25, 0.45))     # (margin from the shot ends, shortest section as a share of the look's minimum length): strict, then relaxed (2026-10-30, v030 r02 dropped all three requested looks on 2.7 s shots)


def place(piece: dict, names, holds, seed=0, prefer=None, why=None) -> list[dict]:
    """each name -> {name, shot, t0, t1, word} on a shot that has what it needs, inside one or two word holds (length 2-4 s; outline_to_word: the 1.6-2.6 s that end 1.2 s after a hit).
    STRICT first (clear of the shot's first / last SECTION_MARGIN_S s, section >= length - 0.35 s, a different shot for each when avoidable); a name that fits nowhere is RELAXED, never silently
    dropped: the margin shrinks to 0.45 then 0.25 s, the section may be down to 60 % / 45 % of the look's minimum length (>= 0.8 s), any word hold that touches the usable part of a shot
    counts, and a shot may take a second look. Only what is physically impossible is dropped (no shot with a matte; a plate look with no plated shot; no hold touches any shot): the reason is
    written into ``why[name]``. The random 0-2 cap never applies to requested names (draw() adds them on top)."""
    spans = _shot_spans(piece)
    shots = {s["id"]: s for s in piece.get("shots") or []}
    rng = random.Random(f"explore-place:{int(seed or 0)}")
    out, taken = [], set()
    why = why if why is not None else {}
    for name in sorted(names, key=lambda n: ("plate" not in SPECS[n]["needs"], names.index(n))):          # the most constrained look first (the plate one)
        spec = SPECS[name]
        lo, hi = spec["length"]
        sids = [sid for sid in spans if not (prefer and prefer.get(name) and sid != prefer[name])]
        if prefer and prefer.get(name) and not sids:
            sids = list(spans)                                                    # the preferred shot does not exist any more: any shot
        ok_sids = []
        for sid in sids:
            if "plate" in spec["needs"] and str(shots.get(sid, {}).get("plate") or "").lower() != "clean":
                continue
            if shots.get(sid, {}).get("matte") is False:
                continue
            ok_sids.append(sid)
        if not ok_sids:
            why[name] = ("needs a shot with plate: clean" if "plate" in spec["needs"] else "no shot with a subject matte") + (f" (the concept put it on {prefer[name]})" if prefer and prefer.get(name) else "")
            continue
        pick, tier_used = None, 0
        for ti, (margin, share) in enumerate(RELAX):
            minlen = lo - 0.35 if share is None else max(0.8, share * lo)
            cands = []
            for sid in ok_sids:
                if sid in taken and len(taken) < len(spans) and ti == 0 and not (prefer and prefer.get(name)):
                    continue
                a, b = spans[sid]
                cands += _cands_for(name, spec, sid, a, b, holds, margin, minlen)
            if cands:
                pick, tier_used = rng.choice(cands), ti
                break
        if pick is None:
            best = max((spans[s][1] - spans[s][0] - 2 * RELAX[-1][0] for s in ok_sids), default=0.0)
            why[name] = f"no word hold touches the usable part of a shot even relaxed (longest usable window {max(best, 0.0):.2f} s, shortest section 0.8 s)"
            continue
        if tier_used:
            pick["relaxed"] = tier_used
        out.append(pick)
        taken.add(pick["shot"])
    return out


TAIL = ("grade", "grain", "vignette", "halation", "tint")


def _gate(t0: float, t1: float):
    return _tl((t0 - GATE_S, 0.0), (t0, 1.0), (t1, 1.0), (t1 + GATE_S, 0.0))


def weave_chain(ops: list, placed: list[dict], ctx_base: dict, flag: str = "fx", mbuf: str = "fxm") -> list:
    """the chain with the placed effects of ITS shot gated in just before the closing tail ops (idempotent: ops flagged ``flag`` are dropped first). Another effect family
    (tools/fusion/contour_morph.py) passes its own ``flag`` + ``mbuf`` (matte buffer) so the two never strip each other; its ops come from ``build_ops`` (name ``contour_morph``)."""
    ops = [o for o in copy.deepcopy(ops) if not o.get(flag)]
    if not placed:
        return ops
    k = len(ops)
    while k > 0 and ops[k - 1].get("op") in TAIL:
        k -= 1
    if k == 0:
        return ops
    base = ops[k - 1].get("out", "image")
    have_m = any(o.get("out") == "m" and o.get("op") == "upscale_matte" for o in ops)
    new: list = []
    new.append({"op": "upscale_matte", "out": mbuf, flag: True, "args": {"matte": "$matte", "size": ctx_base["size"], "guide": "$footage"}}) if not have_m else None
    mname = "m" if have_m else mbuf
    cur = "$" + base
    for e in placed:
        ctx = dict(ctx_base, t0=e["t0"], t1=e["t1"], seed=int(ctx_base.get("seed", 0)) + 7 * len(new))
        if e["shot"] in (ctx_base.get("plated_shots") or {}):
            ctx["plated"] = bool(ctx_base["plated_shots"][e["shot"]])          # this effect's own shot decides: the plate fill needs a plated shot
        if e.get("params"):
            ctx["params"] = copy.deepcopy(e["params"])
        built = build_ops(e["name"], ctx)
        for o in built:
            o = copy.deepcopy(o)
            o[flag] = True
            _swap(o, "$fxm", "$" + mname)
            new.append(o)
        new.append({"op": "crossfade", "out": base, flag: True, "args": {"a": cur, "b": "$fx", "t": _gate(e["t0"], e["t1"])}})
        cur = "$" + base
    return ops[:k] + new + ops[k:]


def _swap(o: dict, a: str, b: str) -> None:
    for k, v in list((o.get("args") or {}).items()):
        if v == a:
            o["args"][k] = b


def weave(piece: dict, music: dict | None = None, chk: dict | None = None, size=None) -> dict:
    """THE HOOK. Place (once; kept in plan.explore) and weave the effects the piece's system drew / the request forced. No names = no change. Never raises (a failure is a plan note)."""
    pl = piece.setdefault("plan", {})
    names = list(((pl.get("piece_system") or {}).get("effects") or {}).get("explore") or [])
    names += [s["fx"] for s in piece.get("shots") or [] if s.get("fx") in SPECS and s["fx"] not in names]
    comp = piece.get("compose") or {}
    chains = ([("default", comp["default"])] if comp.get("default") else []) + list((comp.get("shots") or {}).items())
    if not names and not (pl.get("explore") or {}).get("placed"):
        for _n, ch in chains:
            ch["ops"] = [o for o in ch.get("ops", []) if not o.get("fx")]
        return piece
    try:
        from tools.fusion import plan_rules as PR
        holds = PR.word_holds(piece, music, chk)
        prev = (pl.get("explore") or {}).get("placed")
        pref = {s["fx"]: s["id"] for s in piece.get("shots") or [] if s.get("fx") in SPECS}          # looks the concept named on a shot
        names = list(dict.fromkeys(list(pref) + names))
        why: dict = {}
        placed = prev if prev else place(piece, [n for n in names if n in SPECS], holds, int((pl.get("piece_system") or {}).get("seed", 0)), pref, why)
        dropped = [n for n in names if n not in {e["name"] for e in placed}]
        sz = list(size or (piece.get("sizes") or {}).get("compose") or (540, 960))
        col = colours(piece)
        plated = any(str(s.get("plate") or "").lower() == "clean" for s in piece.get("shots") or [])
        from tools.fusion import contour_sweep as _CS
        hb_f = 0.5 * _CS._beat_s(piece, music) * float(piece.get("fps") or 24)           # half a beat in frames (echo spacing)
        for nm, ch in chains:
            mine = [e for e in placed if nm == "default" or e["shot"] == nm]
            if nm == "default":                                         # the default chain serves every shot without its own: only the effects on those shots
                own = set((comp.get("shots") or {}))
                mine = [e for e in placed if e["shot"] not in own]
            ch["ops"] = weave_chain(ch.get("ops", []), mine, dict(size=sz, colours=col, seed=int((pl.get("piece_system") or {}).get("seed", 0)), plated=plated, half_beat_f=hb_f,
                                                                     plated_shots={str(s_.get("id")): str(s_.get("plate") or "").lower() == "clean" for s_ in piece.get("shots") or []}))
        pl["explore"] = dict(placed=placed, requested=list(names), dropped=dropped, dropped_why={n: why.get(n, "not placed") for n in dropped})
    except Exception as e:      # noqa: BLE001  never break a piece for an effect
        pl.setdefault("notes", []).append(f"explore effects skipped: {type(e).__name__}: {str(e)[:120]}")
    return piece
