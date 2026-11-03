"""THE LINES BETWEEN THE FRAMES' WORDS (2026-11-04 user: "I wanted those previews in the GUI as basic layouts but the artist engine still needs to transform to them through graphic mathematical lines").
The GUI layout stays the exact resting pose of every word. Between two words the APPROVED library carries the strands (mograph/modules/io/outline.py + mograph/modules/transitions/*, kinds = sequence.OUTLINE_KINDS):
A's fill fades into its own glyph outline cut into 10-18 arcs, the library transition (braid / circuit_board / fan / spiral_around / twist / ripple, built with unravel, no ghost / pads / web) is BOUND to A's and B's arcs and run
unchanged: its own layers(t) polylines are projected by a fixed front camera (cam3d.Camera.home, world = frame px x k, k = 960 / long side, as the portrait 540x960 renders) and drawn as flat hairlines in the word's own fill / ring colours.
They land as arcs just off B's outline, ease on while B's fill comes in, then thin away; after the form-in B is the plain resting word. First word: strands form it. Last word: its arcs fade out in place.
Timing: B's land frame never moves; A holds still >= HOLD_S, then releases; travel <= T_MAX; a gap too short stays a plain cut. numpy / cv2 only."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import numpy as np

EXIT_S, FORM_S, HOLD_S, T_MAX, T_MIN = 0.35, 0.55, 0.5, 1.2, 0.25
HAIR_W = 1.1                  # the library's unravel hairline (outline.HAIR), world px
MATTE_LOOKS = ("behind", "maskm", "maskb")


def _ss(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


@contextmanager
def world(fw: int, fh: int):
    """the cam3d world plane for this frame size (restored after): modules read cam3d.W / H while they build and draw."""
    from mograph.render import cam3d
    k = 960.0 / max(fw, fh)
    prev = cam3d.set_world(int(round(fw * k)), int(round(fh * k)))
    try:
        yield k, cam3d
    finally:
        cam3d.set_world(*prev)


class Arcs:
    """a word's outline cut into arcs, in world px: base on the glyph outline, nrm the outward unit normals, off the hand-off offset."""
    def __init__(self, base, nrm, off):
        self.base, self.nrm, self.off = base, nrm, off

    def at(self, k: float) -> np.ndarray:
        return self.base + self.nrm * (self.off * k)

    def moved(self, v) -> "Arcs":
        return Arcs(self.base + np.asarray(v, float), self.nrm, self.off)


def arcs_of(word, fw: int, fh: int, k: float):
    """Arcs from the word's REAL glyph outline in its GUI pose (FlatType mask contours), cut by the library's own outline_arcs (K by perimeter / cap as outline.Outline.arcs); None if blank."""
    if getattr(word, "_fl_arcs", 0) != 0:
        return word._fl_arcs
    import cv2
    from mograph.modules.io.outline import outline_arcs, PTS, K_RANGE
    m = word.flat.mask(fw, fh)
    cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cons = [c[:, 0, :].astype(float) * k for c in cs if len(c) >= 12]
    word._fl_arcs = None
    if cons:
        cap = 0.72 * word.lay["fs"] * word.lay["s"] * k
        per = sum(float(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum()) for c in cons)
        K = int(np.clip(round(per / max(1.6 * cap, 1.0)), *K_RANGE))
        base = outline_arcs(cons, K, PTS)
        g = cv2.GaussianBlur(m, (0, 0), max(1.5, 0.05 * cap / k))
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        ix = np.clip(np.round(base[..., 0] / k).astype(int), 0, fw - 1)
        iy = np.clip(np.round(base[..., 1] / k).astype(int), 0, fh - 1)
        v = -np.stack([gx[iy, ix], gy[iy, ix]], -1)                              # outward = down the mask's gradient
        word._fl_arcs = Arcs(base, v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9), 0.06 * cap)
    return word._fl_arcs


def bundle(arcs: Arcs, side: str, d, v: float):
    """the hand-off bundle outline.Outline.material builds (exit: head first along the outline; entry: tail first), flat hairlines."""
    from mograph.modules.core import Bundle, unit
    P = np.concatenate([arcs.at(1.0), np.zeros(arcs.base.shape[:2] + (1,))], 2)
    d = np.asarray(d, float)
    if side == "out":
        D = unit(0.75 * d[None] + 0.25 * unit(P[:, 0] - P[:, 1]))
    else:
        P = P[:, ::-1].copy()
        D = unit(0.75 * d[None] + 0.25 * unit(P[:, -2] - P[:, -1]))
    K = len(P)
    return Bundle(P, D, np.full((K, 3), 255.0), np.full(K, HAIR_W), 0.0, np.full(K, v), 0.0)


class Gap:
    """one word-to-word (or entry / exit) window: kind, frames [ws, we), exit frames ex, travel (ws+ex .. L), B's land frame L, form frames."""
    def __init__(self, kind, a=None, b=None, ws=0, we=0, ex=0, L=0, form=0, why="", size=(540, 960), fps=25.0):
        self.kind, self.a, self.b, self.ws, self.we, self.ex, self.L, self.form, self.why = kind, a, b, ws, we, ex, L, form, why
        self.size, self.fps, self.k, self.tr, self.pre = size, fps, 960.0 / max(size), None, None

    def info(self) -> dict:
        return dict(kind=self.kind, a=self.a.w["n"] if self.a else None, b=self.b.w["n"] if self.b else None, window=[self.ws, self.we], exit=self.ex, travel=self.L - self.ws - self.ex if self.b else 0, land=self.L, form=self.form, why=self.why)


def _build(g: Gap, kind: str, size, seed: int, rng) -> None:
    """bind the library transition `kind` to A's and B's arcs (A missing / gone: B's own outline shifted just off the frame, so the strands come in from a side)."""
    from mograph.modules.core import get_module
    fw, fh = size
    with world(fw, fh) as (k, cam3d):
        B = arcs_of(g.b, fw, fh, k)
        A = arcs_of(g.a, fw, fh, k) if g.ex else None
        assert B is not None and (A is not None or not g.ex)
        cb = B.base.mean((0, 1))
        if A is None:
            d = np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)][int(rng.integers(4))])
            lo, hi, Wk, Hk = B.base.min((0, 1)), B.base.max((0, 1)), cam3d.W, cam3d.H
            ext = {(1, 0): hi[0], (-1, 0): Wk - lo[0], (0, 1): hi[1], (0, -1): Hk - lo[1]}[(int(d[0]), int(d[1]))] + 8.0
            A = B.moved(-d * ext)
        else:
            d = cb - A.base.mean((0, 1))
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0])
        T = (g.L - g.ws - g.ex) / g.fps
        v = max(float(np.linalg.norm(cb - A.base.mean((0, 1)))), 200.0) / T
        tr = get_module(kind)(dur=T, seed=int(seed), unravel=True, ghost=False, pads=False, web=False)
        tr.bind(bundle(A, "out", np.r_[d, 0.0], v), bundle(B, "in", np.r_[d, 0.0], v), T)
        g.tr = tr


def plan(objs, fps: float, size, cuts=None, seed=None, log=None) -> list[Gap]:
    """windows for every gap between the words (+ the first word's entry, the last word's exit); sets o.gin / o.gout. cuts = {cut frame: {h}} of the picture moves (windows never run inside one)."""
    for o in objs:
        o.gin = o.gout = None
    if not objs:
        return []
    from mograph.modules.sequence import OUTLINE_KINDS
    seed = int(hashlib.sha1(json.dumps([[o.w["text"], o.w["start"], o.w["end"]] for o in objs]).encode()).hexdigest()[:8], 16) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    ex_f, form0, hold_f, tmax, tmin = (max(1, int(round(x * fps))) for x in (EXIT_S, FORM_S, HOLD_S, T_MAX, T_MIN))
    moves = [(c - m["h"], c + m["h"]) for c, m in (cuts or {}).items()]
    gaps, prev = [], None

    def hit(ws, we):
        return any(lo < we and ws < hi for lo, hi in moves)

    def hold_from(o):
        return o.gin.we if o.gin is not None and o.gin.kind != "cut" else o.w["start"] + 3

    def finish(g, a, b):
        nonlocal prev
        ok = g.kind != "cut" and (g.b is None or arcs_of(g.b, *size, g.k) is not None) and (not (g.ex or g.kind == "exit") or arcs_of(g.a, *size, g.k) is not None)
        if g.kind != "cut" and not ok:
            g.kind, g.why = "cut", "blank outline"
        if g.kind in ("full", "enter") and g.L - g.ws - g.ex <= 0:
            g.kind = "form"                                   # first word with no room before it: the strands only ease on
        if g.kind in ("full", "enter"):
            ks = [x for x in OUTLINE_KINDS if x != prev]
            first, err = int(rng.integers(len(ks))), None
            for j in range(len(ks)):
                kd = ks[(first + j) % len(ks)]
                try:
                    _build(g, kd, size, seed + len(gaps), rng)
                    g.kind, err = kd, None
                    break
                except Exception as e:                        # a kind the library cannot drive here: the next one, never a made-up pattern
                    err = f"{kd}: {type(e).__name__} {e}"
            if g.tr is None:
                g.kind, g.why = "cut", err or "no kind"
            else:
                prev = g.kind
        if g.kind == "cut":
            g.ws = g.we = g.L
        gaps.append(g)
        if b is not None:
            b.gin = g
        if a is not None:
            a.gout = g

    for i, b in enumerate(objs):
        a = objs[i - 1] if i else None
        L = b.w["start"]
        form = min(form0, max(0, b.w["end"] - L))
        g = Gap("cut", a, b, L, L + form, 0, L, form, size=size, fps=fps)
        if getattr(b, "fm_in", None) is not None:             # frame_modules builds this word (its construction window ends on L): no strands here
            g.why = "word module window"
            finish(g, a, b)
            continue
        if a is None:
            g.ws, g.kind = max(0, L - tmax), "enter"
        else:
            Ae, R = a.w["end"], max(hold_from(a) + hold_f, L - ex_f - tmax)
            if R <= Ae and L - R - ex_f >= tmin:
                g.ws, g.ex, g.kind = R, ex_f, "full"
            elif L - max(Ae, L - tmax) >= tmin:
                g.ws, g.kind = max(Ae, L - tmax), "enter"
            else:
                g.why = "gap too short for a transition: plain cut"
        if g.kind != "cut" and hit(g.ws, g.we):
            if g.kind == "enter":                             # a move that ended before B lands: start after it
                g.ws = max([g.ws] + [hi for lo, hi in moves if hi <= L])
                if L - g.ws < tmin:
                    g.ws = L if a is None else g.ws
            if (g.kind == "enter" and a is not None and L - g.ws < tmin) or hit(g.ws, g.we):
                g.kind, g.why = "cut", "collides with a picture move"
        finish(g, a, b)
    o = objs[-1]                                              # the last word: its arcs fade out in place
    Ae = o.w["end"]
    R = max(hold_from(o) + hold_f, Ae - int(round(0.8 * fps)))
    if Ae - R >= int(round(0.3 * fps)) and not hit(R, Ae) and getattr(o, "fm_out", None) is None:
        finish(Gap("exit", o, None, R, Ae, int(min(ex_f, 0.6 * (Ae - R))), Ae, 0, size=size, fps=fps), o, None)
    return gaps


def gaps_of(objs) -> list[Gap]:
    g = [o.gin for o in objs if getattr(o, "gin", None) is not None]
    last = getattr(objs[-1], "gout", None) if objs else None
    return g + ([last] if last is not None and last.kind == "exit" else [])


def index(objs) -> dict:
    """{frame: Gap} for every frame inside a window."""
    return {f: g for g in gaps_of(objs) if g.kind != "cut" for f in range(g.ws, g.we)}


def describe(objs) -> list[dict]:
    return [g.info() for g in gaps_of(objs)]


# ------------------------------------------------------------------------------------------------ one frame of a window
def _mix(a, b, t):
    return tuple(float(x) * (1 - t) + float(y) * t for x, y in zip(a, b))


def _polys(g: Gap, f: int):
    """[(frame-px points, alpha, closed)] of the strands at frame f, from outline.py's easing (hand-off) and the library transition's own layers(t)."""
    out, (fw, fh), k = [], g.size, g.k
    use_a = g.ex > 0 or g.kind == "exit"
    A = arcs_of(g.a, fw, fh, k) if use_a else None
    B = arcs_of(g.b, fw, fh, k) if g.b is not None and f >= g.L else None

    def put(P, al):
        out.extend((p / k, al, False) for p in P)
    if g.kind == "exit":
        p = (f - g.ws) / max(g.ex, 1)
        if p < 1.0:
            put(A.at(_ss((p - 0.45) / 0.55)), _ss(p / 0.35))
        else:
            put(A.at(1.0), 1.0 - _ss((f - g.ws - g.ex) / max(g.we - g.ws - g.ex, 1)))
    elif f >= g.L:                                            # B's form window: the arcs ease onto the outline, then thin away
        p = (f - g.L) / max(g.form, 1)
        put(B.at(1.0 - _ss(p / 0.55)), 1.0 - _ss((p - 0.6) / 0.4))
    elif f < g.ws + g.ex:                                     # A's exit: fill -> outline, the arcs ease off the letters
        p = (f - g.ws) / g.ex
        put(A.at(_ss((p - 0.45) / 0.55)), _ss(p / 0.35))
    else:                                                     # the library transition's own strands
        with world(*g.size) as (k2, cam3d):
            from mograph.render.cam3d import Camera
            cam, cen = Camera.home(), np.array([cam3d.W / 2.0, cam3d.H / 2.0])
            for L in g.tr.layers((f - g.ws - g.ex) / g.fps):
                if L.kind != "line" or L.pts.shape[0] == 0:
                    continue
                Q = (L.pts - cam.eye) @ cam.R.T
                z = np.maximum(Q[:, 2], 6.0)
                xy = (cen + cam.F * Q[:, :2] / z[:, None]) / k2
                for i in range(len(L.polys)):
                    o0, o1 = L.offs[i], L.offs[i + 1]
                    if o1 - o0 > 1 and Q[o0:o1, 2].min() > 6.0:
                        out.append((xy[o0:o1], float(L.alpha), bool(L.closed)))
    return out


def frame(g: Gap, im: np.ndarray, f: int, mattes) -> np.ndarray:
    """the composed frame f inside a window: A / B fills blended with their own resting looks, then the strands over them (flat hairlines in the words' fill + ring colours; the subject matte over them for behind / matte looks)."""
    import cv2
    a, b, out = g.a, g.b, im

    def blend(w, fi, al):
        full = w.over(im, f, mattes, fi)
        return full if al >= 1.0 else cv2.addWeighted(im, 1.0 - al, full, al, 0.0)
    if a is not None and g.ex:
        a.colours(im)
        if f < g.ws + g.ex:
            al = 1.0 - _ss((f - g.ws) / g.ex / 0.6)
            if al > 0.0:
                out = blend(a, min(f, a.w["end"] - 1), al)
    if b is not None and f >= g.L:
        b.colours(im)
        al = _ss(((f - g.L) / max(g.form, 1) - 0.15) / 0.6)
        if al > 0.0:
            out = blend(b, None, al)
    polys = _polys(g, f)
    if not polys:
        return out
    if g.pre is None:                                         # before B has colours: A's, else this shot's own picked pair
        from tools.app.frame_render import pick_colors
        g.pre = (a.fill, a.ring) if a is not None and g.ex else pick_colors(im)[:2]
    t = 0.0 if (b is None or f < g.L or b.fill is None) else _ss((f - g.L) / max(g.form, 1) / 0.5)
    fill, ring = (_mix(g.pre[0], b.fill, t), _mix(g.pre[1], b.ring, t)) if t > 0 else g.pre
    h, w = im.shape[:2]
    hair, edge = max(2, int(round(0.0016 * max(w, h)))), max(1, int(round(0.0007 * max(w, h))))        # the fill colour carries the line, the ring is a thin edge
    mf, mr = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for pts, al, closed in sorted(polys, key=lambda p: p[1]):
        q = (np.clip(pts, -1e4, 1e4) * 16).astype(np.int32)[None]
        v = int(round(255 * min(1.0, max(0.0, al))))
        cv2.polylines(mr, q, closed, v, hair + 2 * edge, cv2.LINE_AA, 4)
        cv2.polylines(mf, q, closed, v, hair, cv2.LINE_AA, 4)
    ys, xs = np.nonzero(mr)
    if not len(ys):
        return out
    k = 0.0
    if mattes is not None:                                    # the outgoing look before the travel's end, the incoming after (a behind / matte look keeps the subject over the strands)
        ma = (g.ex > 0 or g.kind == "exit") and a.look in MATTE_LOOKS
        mb = b is not None and b.look in MATTE_LOOKS
        u = 0.0 if b is None else (_ss((f - g.ws) / max(g.L - g.ws, 1)) if g.ex else 1.0)
        k = (1 - u) * ma + u * mb
    res = out.copy()
    px = res[ys, xs].astype(np.float32)
    ar, af = mr[ys, xs, None] / 255.0, mf[ys, xs, None] / 255.0
    if k > 0:
        keep = (1.0 - k * mattes.get(f)[ys, xs])[:, None]
        ar, af = ar * keep, af * keep
    px = px * (1 - ar) + np.float32(ring) * ar
    res[ys, xs] = (px * (1 - af) + np.float32(fill) * af).clip(0, 255).astype(np.uint8)
    return res
