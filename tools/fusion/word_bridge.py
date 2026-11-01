"""word_bridge: the CONTOUR LEG of a library word-to-word transition (compose side). Never draws a pattern of its own.

History: 2026-11-01 v1 drew its own spiral / circuit / braid / fan / ring / helix patterns in compose; the user rejected them (R16b e3 / e7, R17:
"did I confuse you by saying mathematical patterns? I was referring to the library transitions that we created"). That path (`pattern()`,
`flight()`) is DELETED. The library transition is now connected in the type engine: mograph/modules/io/outline.py (word A's glyph outlines peel
off into the module's own strands, which land on word B's glyph outlines; mograph/modules/sequence.outline_handoff).

What stays here is the leg the user liked (R16b e6, f168: "the transition from the type to the outline is good"): in a library gap, seeded per
gap (about half the time when the shot has a subject matte), compose takes over between A's release and a hand-over time Th inside the
library transition (its start + HANDOVER x its length):
    A's letter outlines (type matte at A's landed frame)  ->  the subject's LIVE contour over the untouched footage (the arcs travel along it)
    ->  exactly where the library module's strands are at Th (read from the type matte of frame Th: the hand-over target),
while the type pass is gated out of [release, Th] (pivot_cut.apply_gate ramps); at Th the gate opens, the library strands carry on in their
own pattern and land on B (no compose drawing there). A subject too small to carry lines (< 1.5 % of the frame / < 25 % of its height):
straight from A's outlines to the library strands. No matte / the roll says no: no compose op at all, the type pass alone does the
connected library transition.

Colour: A's own ink (sampled under A's letters at the release; else a shot-hue ink). Every piece (image and video mode): ON by default; compose.word_bridge = false switches the leg off, compose.word_bridge_contour = "subject"
forces it on a gap, false forbids it. Fine strokes: 0.0028 x the short side on letters, 0.0035 on the contour.

    plan(piece, type_dir)                    -> [event]   deterministic in the piece seed (every shot's chain makes the same plan)
    gate_windows(events)                     -> [[lo, hi, fade_out_s, fade_in_s]] for pivot_cut.apply_gate
    recipe_ops(piece, ops, sid, type_dir)    -> ops with one `word_bridge` op before the closing grade / grain (recipe_eval.build_recipe)
    OPS = {"word_bridge": word_bridge}       the compose op (numpy / cv2 geometry, backend composite)
"""
from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import cv2
import numpy as np

from tools.fusion import compose as C

NAME = "word_bridge"
LIB = ("spiral_around", "circuit_board", "braid", "fan", "ripple", "twist")
FADE_A_S = 0.20            # A's fill fades out under its outline over this long after the release
FADE_IN_S = 0.18           # the type pass (the library strands) fades back in over this long, ending at the hand-over Th
OUT_S = 0.15               # the compose lines fade out over this long after Th (the library strands carry on)
START_S = 0.06             # the lines start moving this long after the release (they sit on A's outline first)
HANDOVER = 0.4             # Th = the library transition's start + this share of its length (its own pattern is formed by then)
MIN_LEG_S = 0.8            # release -> Th at least this long, else no leg (the type pass alone)
LETTER_W = 0.0028          # stroke width on letters, share of the frame short side (user: "finer strokes")
LINE_W = 0.0035            # on the subject contour (R16b f168 look)
LINE_VAR = 1.25            # thick-thin variation
N_PTS = 64                 # points per strand
K_MIN, K_MAX = 28, 72      # strands
STAGGER = 0.35             # left-to-right delay of a morph across the strands (share of a phase)
P_SUBJECT = 0.5            # chance of the leg when the shot has a subject matte
SS = 2                     # supersampling of the stroke mask
LEG = (("a", "c", 0.38), ("c", "c", 0.27), ("c", "p", 0.35))      # A outline -> contour -> hand-over to the library strands




# ================================================================================================================ plan
def _kind(tr):
    if isinstance(tr, dict):
        return tr.get("kind") or tr.get("module")
    return tr


def _manifest(type_dir):
    try:
        import json
        p = Path(type_dir) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if type_dir and p.is_file() else {}
    except Exception:          # noqa: BLE001
        return {}


def gap_kinds(piece: dict, type_dir=None) -> list:
    """the transition of every gap: the one the design named (word.transition / type.transitions[i]), else the one the type engine chose (manifest info.transitions[i].module)."""
    words = (piece.get("type") or {}).get("words") or []
    trs = (piece.get("type") or {}).get("transitions")
    man = ((_manifest(type_dir).get("info") or {}).get("transitions")) or []
    out = []
    for i in range(max(len(words) - 1, 0)):
        k = _kind(words[i].get("transition")) if isinstance(words[i], dict) else None
        if k in (None, "", "auto") and isinstance(trs, list) and i < len(trs):
            k = _kind(trs[i])
        if k in (None, "", "auto") and i < len(man) and isinstance(man[i], dict):
            k = man[i].get("module")
        out.append(str(k).strip().lower() if k not in (None, "") else None)
    return out


def _seed(piece: dict) -> int:
    pl = piece.get("plan") or {}
    return int((pl.get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0)) or 0)


def _beats(piece: dict, type_dir=None) -> list:
    sp = _manifest(type_dir).get("spec") or {}
    b = sp.get("beats") or ((piece.get("audio") or {}).get("beats")) or []
    try:
        return [float(x) for x in b]
    except (TypeError, ValueError):
        return []


def _blocked(piece: dict) -> list:
    """piece-second windows a bridge must not touch: promoted cuts that own B's word (circle_grid / depth_lines), explore sections. A plain 3D cut (pivot / orbit / dolly)
    only moves the footage under the type: the bridge runs over it like the type does."""
    out = []
    try:
        from tools.fusion import pivot_cut as PC
        out += [tuple(w) for w in PC.gate_windows(piece)]
    except Exception:          # noqa: BLE001
        pass
    for e in (((piece.get("plan") or {}).get("explore") or {}).get("placed")) or []:
        try:
            out.append((float(e["t0"]) - 0.2, float(e["t1"]) + 0.2))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def _shot_at(lay: dict, fps: float, t0: float, t1: float):
    """the shot whose span holds [t0, t1] (piece seconds), else None."""
    for sid, L in lay.items():
        a, b = L["start"] / fps, (L["start"] + L["frames"]) / fps
        if a <= t0 and t1 <= b:
            return sid
    return None


def phase_times(s0: float, s1: float) -> list:
    """[(key_from, key_to, t_start, t_end)] of the leg over [s0, s1]."""
    d, t, out = s1 - s0, s0, []
    for a, b, f in LEG:
        out.append((a, b, t, t + f * d))
        t += f * d
    return [tuple(p[:3]) + (s1 if i == len(out) - 1 else p[3],) for i, p in enumerate(out)]


def plan(piece: dict, type_dir=None) -> list:
    """the gaps that take the contour leg (see the module doc). Deterministic: every shot's recipe gets the same events."""
    comp = piece.get("compose") or {}
    force = comp.get("word_bridge_contour")
    if comp.get("word_bridge") is False or force is False or piece.get("technique") == "traced":
        return []
    man = _manifest(type_dir)
    info = man.get("info") or {}
    ws, trs = info.get("words") or [], info.get("transitions") or []
    if len(ws) < 2:
        return []
    fps = float(piece.get("fps") or man.get("fps") or 24)
    kinds = gap_kinds(piece, type_dir)
    morph = set((((piece.get("plan") or {}).get("contour_morph")) or {}).get("words") or [])
    blocked = _blocked(piece)
    try:
        from tools.fusion import recipe_eval as RE
        lay, _ = RE.piece_layout(piece)
    except Exception:          # noqa: BLE001
        return []
    seed = _seed(piece)
    out = []
    for i in range(min(len(ws) - 1, len(kinds), len(trs))):
        kind = kinds[i]
        if kind not in LIB or i in morph or not isinstance(trs[i], dict):
            continue
        try:
            rel, hit, hit_a = float(ws[i]["release"]), float(ws[i + 1]["hit"]), float(ws[i]["hit"])
            th = float(trs[i]["start"]) + HANDOVER * float(trs[i]["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        if th - rel < MIN_LEG_S or th >= hit:
            continue
        if any(rel - 0.05 < b and a < th + 0.05 for a, b in blocked):
            continue
        s0 = rel + START_S
        ph = phase_times(s0, th)
        mid_c = 0.5 * (ph[1][2] + ph[1][3])
        sid_c = _shot_at(lay, fps, mid_c, mid_c)                             # the shot under the contour hold (a cut inside the leg: the lines follow the new shot's subject, like the picture)
        if sid_c is None:
            continue
        try:
            has_m = bool(RE.shot_needs(piece, sid_c).get("matte"))
        except Exception:      # noqa: BLE001
            has_m = False
        rng = random.Random(f"word_bridge:{seed}:{i}")
        roll = rng.random()
        if not has_m or (force != "subject" and roll >= P_SUBJECT):
            continue
        out.append(dict(name=NAME, word=i, kind=kind, rel=round(rel, 4), hit=round(hit, 4), th=round(th, 4), s0=round(s0, 4),
                        fa=max(int(round(hit_a * fps)), int(round(rel * fps)) - 2), fth=int(round(th * fps)),
                        end=round(th + OUT_S, 4), contour=dict(src="subject", shot=sid_c), fps=fps,
                        phases=[list(p) for p in ph], seed=rng.randrange(1 << 30)))
    return out


def gate_windows(events: list) -> list:
    """[[lo, hi, fade_out, fade_in]]: the type pass fades out after A's release and back in up to the hand-over."""
    return [[e["rel"], e["th"], FADE_A_S, FADE_IN_S] for e in events]


TAIL = ("grade", "grain", "vignette", "halation", "tint")


def recipe_ops(piece: dict, ops: list, sid: str, type_dir=None, events=None) -> list:
    """`ops` of shot `sid` with ONE word_bridge op before the closing tail ops, carrying the events that overlap the shot. Idempotent; no event = unchanged."""
    if any(o.get("op") == NAME for o in ops):
        return ops
    events = plan(piece, type_dir) if events is None else events
    if not events or type_dir is None:
        return ops
    try:
        from tools.fusion import recipe_eval as RE
        lay, _ = RE.piece_layout(piece)
        L = lay[sid]
    except Exception:          # noqa: BLE001
        return ops
    fps = float(piece.get("fps") or 24)
    a, b = L["start"] / fps, (L["start"] + L["frames"]) / fps
    mine = [e for e in events if e["rel"] - 0.1 < b and a < e["end"] + 0.1]
    if not mine:
        return ops
    ops = copy.deepcopy(ops)
    k = len(ops)
    while k > 0 and ops[k - 1].get("op") in TAIL:
        k -= 1
    if k == 0:
        return ops
    out = ops[k - 1].get("out", "image")
    matte = None
    for o in ops[:k]:                                      # the chain's own subject matte buffer (always resolvable), else the shot's when the chain reads it
        for key in ("matte", "subject_matte", "fg_matte"):
            v = (o.get("args") or {}).get(key)
            if isinstance(v, str) and v.startswith("$") and "type" not in v:
                matte = v
    if matte is None:
        rs = set()
        RE.refs([o.get("args", {}) for o in ops], rs)
        if f"{sid}_matte" in rs:
            matte = f"${sid}_matte"
    pal = None
    try:
        from tools.fusion import user_style as US
        pal = US.parse_direction(piece).get("palette")
    except Exception:          # noqa: BLE001
        pal = None
    args = {"img": f"${out}", "footage": "$footage", "frame_idx": "$frame_idx", "events": mine, "sid": sid,
            "type_dir": str(Path(type_dir) / "matte"), "user_palette": pal}
    if matte:
        args["matte"] = matte
    ops.insert(k, {"op": NAME, "out": out, "wb": True, "args": args})
    return ops


# ================================================================================================================ geometry
_SHAPES: dict = {}
_KEYS: dict = {}
_COL: dict = {}
_LIST: dict = {}
_SRC: dict = {}
SUBJECT_MIN_AREA = 0.015   # a subject matte smaller than this share of the frame ...
SUBJECT_MIN_H = 0.25       # ... or shorter than this share of the frame height does not carry the contour phase (the footage edges do)


def _matte_frame(type_dir: str, idx: int, w: int, h: int):
    key = (type_dir, idx, w, h)
    if key in _SHAPES:
        return _SHAPES[key]
    if type_dir not in _LIST:
        from tools.fusion import recipe_eval as RE
        _LIST[type_dir] = sorted((str(p) for p in Path(type_dir).glob("*.png")), key=RE.natural)
    fs = _LIST[type_dir]
    m = None
    if fs:
        im = cv2.imread(fs[min(max(int(idx), 0), len(fs) - 1)], cv2.IMREAD_GRAYSCALE)
        if im is not None:
            m = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if len(_SHAPES) > 64:
        _SHAPES.clear()
    _SHAPES[key] = m
    return m


def _contours(mask, min_len=12.0, top=None):
    """closed outlines of a mask (HxW 0..1) -> [(n,2) float] longest first."""
    if mask is None:
        return []
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = [c[:, 0, :].astype(np.float32) for c in cs if len(c) >= 6]
    cs = [c for c in cs if cv2.arcLength(c.reshape(-1, 1, 2), True) >= min_len]
    cs.sort(key=lambda c: -cv2.arcLength(c.reshape(-1, 1, 2), True))
    return cs[:top] if top else cs


def _resample(pts, n, closed=True):
    p = np.asarray(pts, np.float32)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-6:
        return np.repeat(p[:1], n, 0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1).astype(np.float32)


def outline_strands(contours, K, n=N_PTS, shift=0.0, open_=False) -> np.ndarray | None:
    """K strands of n points that together trace the given closed outlines: every outline gets >= 1 strand (largest remainder by length; the smallest outlines are dropped when
    there are more outlines than K). shift (0..1) rotates every outline's start (the arcs travel along it). -> (K, n, 2) or None."""
    cs = [c for c in contours if len(c) >= 3]
    if not cs:
        return None
    cs = cs[:K]
    lens = np.array([cv2.arcLength(c.reshape(-1, 1, 2), not open_) for c in cs], np.float64)
    alloc = np.ones(len(cs), int)
    rest = K - len(cs)
    if rest > 0:
        share = lens / lens.sum() * K - 1.0
        share = np.maximum(share, 0.0)
        add = np.floor(share / max(share.sum(), 1e-9) * rest).astype(int)
        alloc += add
        left = K - alloc.sum()
        for j in np.argsort(-(share - add))[:left]:
            alloc[j] += 1
    out = []
    for c, m in zip(cs, alloc):
        M = max(m * (n - 1) + 1, 8)
        r = _resample(c, M, closed=not open_)
        if shift and not open_:
            r = np.roll(r[:-1], -int(round(shift * (M - 1))) % (M - 1), 0)
            r = np.vstack([r, r[:1]])
        for j in range(m):
            seg = r[j * (n - 1):(j + 1) * (n - 1) + 1]
            out.append(_resample(seg, n, closed=False) if len(seg) != n else seg)
    return np.stack(out[:K]).astype(np.float32)


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def match_index(src: np.ndarray, dst: np.ndarray):
    """(order, flip): dst[order] (strands with flip True reversed) is the assignment that moves strand k of src onto strand k of dst with the least travel."""
    ca, cb = src.mean(1), dst.mean(1)
    cost = np.linalg.norm(ca[:, None] - cb[None], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        _r, col = linear_sum_assignment(cost)
    except Exception:          # noqa: BLE001  greedy fallback
        col, used = np.zeros(len(src), int), set()
        for i in np.argsort(cost.min(1)):
            j = min((j for j in range(len(dst)) if j not in used), key=lambda j: cost[i, j])
            col[i] = j
            used.add(j)
    flip = np.zeros(len(src), bool)
    for i, j in enumerate(col):
        d0 = np.linalg.norm(src[i, 0] - dst[j, 0]) + np.linalg.norm(src[i, -1] - dst[j, -1])
        d1 = np.linalg.norm(src[i, 0] - dst[j, -1]) + np.linalg.norm(src[i, -1] - dst[j, 0])
        flip[i] = d1 < d0
    return np.asarray(col), flip


def apply_index(dst: np.ndarray, idx) -> np.ndarray:
    col, flip = idx
    out = dst[col].copy()
    out[flip] = out[flip][:, ::-1]
    return out


def match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """dst strands reordered (and reversed where shorter) so strand k of src moves onto strand k of the result with the least travel."""
    return apply_index(dst, match_index(src, dst))





_IDX: dict = {}


def keys_for(ev: dict, type_dir: str, w: int, h: int):
    """(K, A strands on A's letter outlines, P strands where the library strands are at Th, rank) of an event, cached; None when either is empty."""
    key = (ev["word"], ev["fa"], ev["fth"], type_dir, w, h)
    if key in _KEYS:
        return _KEYS[key]
    ca = _contours(_matte_frame(type_dir, ev["fa"], w, h))
    cp = _contours(_matte_frame(type_dir, ev["fth"], w, h), min_len=6.0)
    if not ca or not cp:
        _KEYS[key] = None
        return None
    K = int(min(K_MAX, max(K_MIN, len(ca))))
    A = outline_strands(ca, K)
    P = outline_strands(cp, K)
    if A is None or P is None or len(A) != K or len(P) != K:
        _KEYS[key] = None
        return None
    A = A[np.argsort(A.mean(1)[:, 0], kind="stable")]                    # strand rank left -> right (the morph travels with the reading direction)
    rank = np.arange(K) / max(K - 1, 1)
    if len(_KEYS) > 32:
        _KEYS.clear()
    _KEYS[key] = (K, A, P, rank)
    return _KEYS[key]


def state_at(ev: dict, t: float, A, P, rank, contour_strands=None):
    """the K strands at piece time t + taper (0..1). Before s0: A; from Th: P (the library strands). The assignment between keys is made once per event."""
    has_c = contour_strands is not None
    ik = (ev["word"], ev["seed"], len(A), has_c, id(A))
    if ik not in _IDX:
        if has_c:
            ic = match_index(A, contour_strands)
            ip = match_index(apply_index(contour_strands, ic), P)
        else:
            ic, ip = None, match_index(A, P)
        if len(_IDX) > 64:
            _IDX.clear()
        _IDX[ik] = (ic, ip)
    ic, ip = _IDX[ik]
    seq = {"a": A, "p": apply_index(P, ip)}
    if has_c:
        seq["c"] = apply_index(contour_strands, ic)
        phases = ev["phases"]
    else:                                                                    # no usable contour this frame: straight from A's outlines to the library strands
        phases = [("a", "p", ev["s0"], ev["th"])]
    if t <= ev["s0"]:
        return A, 0.0
    if t >= ev["th"]:
        return seq["p"], 0.0
    for a, b, t0, t1 in phases:
        if t0 <= t < t1 or t >= t1 and (a, b) == tuple(phases[-1][:2]):
            u = (t - t0) / max(t1 - t0, 1e-6)
            if a == b:
                return seq[a], 1.0
            uk = _ss(u * (1.0 + STAGGER) - STAGGER * rank)[:, None, None]
            taper = float(min(1.0, u * 2.0)) if a == "a" else float(min(1.0, (1.0 - u) * 2.0))
            return seq[a] + (seq[b] - seq[a]) * uk, taper
    return seq["p"], 0.0


def stroke_mask(strands, w, h, width, taper=1.0, var=LINE_VAR, seed=0):
    """alpha HxW of the strands drawn as brush lines: width px, thick-thin `var`, ends tapered by `taper` (0 = an even outline)."""
    m = np.zeros((h * SS, w * SS), np.uint8)
    K, n = strands.shape[:2]
    s = np.linspace(0.0, 1.0, n - 1)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 1, K)
    for k in range(K):
        p = (strands[k] * SS).astype(np.float32)
        wv = 1.0 + (var - 1.0) * np.sin(2 * math.pi * (1.7 * s + ph[k]))
        tp = (1.0 - taper) + taper * (0.25 + 0.75 * np.sin(math.pi * s) ** 0.6)
        ww = np.maximum(1, np.round(width * SS * wv * tp)).astype(int)
        q = np.round(p * 4).astype(np.int32)
        for j in range(n - 1):
            cv2.line(m, tuple(q[j]), tuple(q[j + 1]), 255, int(ww[j]), cv2.LINE_AA, 2)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def line_colour(footage, user_palette=None):
    """an ink of one of the shot's own hues (the user's palette first), ordered away from the strongest shot hue, >= 3:1 (4.5 preferred) against the frame. -> sRGB 0..1 (3,)."""
    from tools.fusion import legible_color as LC
    im = np.asarray(footage, np.float32)[::6, ::6, :3]
    lin = LC.srgb_to_lin(np.clip(im, 0, 1))
    yb = float(np.median(LC.rel_lum(lin.reshape(-1, 3))))
    if user_palette:
        hues = LC.hues_of_colors(user_palette)
    else:
        hues = LC.shot_hues(lin)
        if len(hues) > 1:
            top = hues[0]["h"]
            hues = sorted(hues[1:], key=lambda q: -LC.hue_dist(q["h"], top)) + hues[:1]
    for need in (4.5, 3.0):
        for hu in hues:
            Ls, cand, ce = LC._hue_inks(hu)
            r = LC.contrast_ratio(LC.rel_lum(cand), yb)
            ok = r >= need
            if ok.any():
                i = int(np.argmax(np.where(ok, ce - 0.02 * np.abs(Ls - float(hu.get("L", 0.6))), -9.0)))
                return np.clip(LC.lin_to_srgb(cand[i]), 0, 1).astype(np.float32)
    return np.array([0.96, 0.95, 0.92] if yb < 0.35 else [0.06, 0.06, 0.07], np.float32)


@C.api


@C.api
def word_bridge(img, footage=None, matte=None, frame_idx=0, events=None, sid=None, type_dir=None, user_palette=None, fps=None):
    """compose op: draw the active contour leg (if any) over `img`. Outside every leg the image comes back untouched."""
    from tools.fusion.contour_sweep import _tonp
    B = C.B
    if not events or type_dir is None:
        return img
    ev, t = None, 0.0
    for e in events:
        t = float(frame_idx) / float(fps or e.get("fps") or 24)
        if e["rel"] - 0.1 <= t <= e["end"]:
            ev = e
            break
    if ev is None:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    ks = keys_for(ev, type_dir, w, h)
    if ks is None:
        return img
    K, A, P, rank = ks
    cs = None
    if ev["phases"][0][2] - 0.05 <= t <= ev["phases"][2][3] + 0.05 and matte is not None:
        m = _tonp(B, C._gray(C._in(matte)))
        m = cv2.resize(m, (w, h)) if m.shape[:2] != (h, w) else m
        dk = (type_dir, ev["word"], ev["seed"], sid)
        big = _contours(m, min_len=0.05 * min(w, h), top=3)
        if dk not in _SRC:                                  # decided once per event and shot: a subject too small to carry the lines -> straight to the strands
            ok = False
            if big:
                pts = np.concatenate(big)
                area = sum(cv2.contourArea(c.reshape(-1, 1, 2)) for c in big)
                ok = area >= SUBJECT_MIN_AREA * w * h and (pts[:, 1].max() - pts[:, 1].min()) >= SUBJECT_MIN_H * h
            _SRC[dk] = ok
        if _SRC[dk] and big:
            trav = 0.22 * max(0.0, t - ev["phases"][0][2]) / max(ev["phases"][2][3] - ev["phases"][0][2], 1e-6)
            cs = outline_strands(big, K, shift=trav)
            if cs is not None and len(cs) != K:
                cs = None
    pts, taper = state_at(ev, t, A, P, rank, cs)
    S = min(w, h)
    width = (LETTER_W + (LINE_W - LETTER_W) * float(taper)) * S
    op = float(_ss((t - (ev["rel"] - 0.1)) / 0.12))
    if t > ev["th"]:
        op *= 1.0 - float(_ss((t - ev["th"]) / OUT_S))
    if op <= 0.0:
        return img
    a_np = stroke_mask(pts, w, h, width, taper, seed=ev["seed"]) * op
    ck = (type_dir, ev["word"], ev["seed"], ev["rel"])                      # ONE colour per leg (the first shot that draws it picks it)
    if ck not in _COL:
        if len(_COL) > 64:
            _COL.clear()
        col = None
        if t <= ev["s0"] + 0.05:                            # the lines are A's own ink: sampled from the composed picture under A's letters at the release
            ma = _matte_frame(type_dir, ev["fa"], w, h)
            if ma is not None and int((ma > 0.6).sum()) >= 50:
                im = _tonp(B, img)
                col = np.median(im[ma > 0.6][:, :3], axis=0).astype(np.float32)
        _COL[ck] = col if col is not None else line_colour(_tonp(B, footage) if footage is not None else _tonp(B, img), user_palette)
    a = B.asarray(a_np)
    colv = B.asarray(np.broadcast_to(_COL[ck], (h, w, 3)).copy())
    return (img * (1.0 - a[..., None]) + colv * a[..., None]).clip(0.0, 1.0)


OPS = {NAME: word_bridge}
