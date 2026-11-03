"""THE WORD MODULES BUILD AND TAKE APART THE FRAMES' WORDS (2026-11-04 user: "I want the GUI just to inform where I want the word to be when constructed, it can choose how it constructs into and out of that word";
"not the exact text just that we are getting the construction / deconstruction and back into the text renders").

The GUI layout is law only for the FORMED word (position, rotation, scale, font, look, land frame): that stays the plain GUI fill (frame_render.Word.over). Everything before it and after it belongs to the engine:
every word is drawn by a REAL library word module (mograph/modules/words/*, reused, nothing reimplemented), placed in the frame's world plane on the word's GUI pose and fitted so its formed silhouette matches the GUI word (bbox fit +
IoU screen; a module whose formed word does not match the GUI letters, or that fails to build, is dropped for the next of the pool, else the word stays flat). Per gap between two words (seeded, neighbours differ):
  A's module deconstructs (its own strands, or the glyph-outline hand-off) -> an approved library transition (sequence.OUTLINE_KINDS) carries the strands -> B's module constructs the word, landing exactly on B's land frame.
The construction / exit frames are NOT confined to the GUI box: strands start and end anywhere (entry / exit ports are seeded or point along the A -> B axis). The first word constructs from a frame edge, the last deconstructs to one.
Rendering: the module's cam3d layers through the same raster the type pass uses (sequence._draw, Camera.home(), world = frame px x k, k = 960 / long side, as frame_lines) on black and on white -> premultiplied RGBA; no cache of frames (each is used once),
the per-word fit is cached to work/modules. Module colours = the shot's own colours (Word.colours -> Palette). numpy / cv2 only.
A gap that cannot hold a build (< D_MIN s between A's hold and B's land, a picture move, a metal word) stays on frame_lines / a plain cut."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.app import frame_lines as FL

POOL = ("lanes", "slats", "staircase", "gradient_fade", "beat_bars", "circuit", "spectrum", "voxel_roll")      # the references' word modules that read as text
HANDOFFS = ("outline", "gate")                    # outline = strands land on B's glyph outlines; gate = each module's own strand exit / entry
D_FULL = (0.7, 0.9, 1.0)                          # seconds: A's exit, the transition's travel, B's form
D_MIN = 1.1                                       # a build needs at least this between A's release and B's land (else a plain cut)
HOLD_S = 0.5                                      # A holds still at least this long after its land
XF = 3                                            # frames: the flat GUI word cross-fades with the module's formed word
IOU_MIN = 0.5                                     # the module's formed silhouette vs the GUI word's
ENTRY_MIN_S = 0.45


def _ss(x):
    return FL._ss(x)


# ------------------------------------------------------------------------------------------------ the module's own raster
def rgba(layers, ww: int, wh: int):
    """cam3d layers -> (premultiplied BGR float32, alpha float32) at the world size: sequence._draw on black and on white (the type pass's raster, Camera.home())."""
    from mograph.modules.sequence import _draw
    from mograph.render.cam3d import Camera
    cam = Camera.home()
    b = _draw(layers, cam, (0, 0, 0), (ww, wh), 2).astype(np.float32)
    w = _draw(layers, cam, (255, 255, 255), (ww, wh), 2).astype(np.float32)
    a = np.clip(1.0 - (w - b).mean(2) / 255.0, 0.0, 1.0)
    return np.ascontiguousarray(b[:, :, ::-1]), a


def rows_of(text: str, tl: dict) -> list[str]:
    """the lines the page lays out (frame_render.layout): grid words are cut into rows of `cols` letters."""
    txt = text.upper()
    cols = int((tl or {}).get("cols") or 0)
    if cols:
        ch = [c for c in txt if not c.isspace()]
        return ["".join(ch[k:k + cols]) for k in range(0, len(ch), cols)] or [""]
    return [l.strip() for l in txt.split("\n") if l.strip()] or [""]


def palette_of(o):
    """the shot's colours as a module Palette (fg = the word's fill, accents = its ring / solid / a mix, lightened so they read on footage)."""
    from mograph.modules.core import Palette, mixc
    rgb = lambda c: (int(c[2]), int(c[1]), int(c[0]))      # noqa: E731   BGR -> RGB
    fill, ring, solid = rgb(o.fill), rgb(o.ring), rgb(o.solid)
    wht = (255, 255, 255)
    return Palette(bg=(12, 12, 11), fg=fill, accents=(mixc(ring, wht, 0.35), mixc(solid, wht, 0.30), mixc(fill, ring, 0.5)))


def build_mod(o, name: str, k: float, fit, rows, timing, entry="left", exit_="right", form_in="lines", form_out="lines"):
    """the module `name` for word o on its GUI pose in the world plane (call inside FL.world)."""
    from mograph.modules.core import Placement, get_module
    L = o.lay
    W, H = L["box"]
    c, ox, oy = fit
    pl = Placement.at((L["cx"] * k + ox, L["cy"] * k + oy, 0.0), roll=L["rot"], w=max(8.0, W * L["s"] * k * c), h=max(8.0, H * L["s"] * k * c))
    return get_module(name)(words="\n".join(rows), font=o.w["font"], placement=pl, entry=entry, exit=exit_, timing=timing, palette=o.fm_pal, form_in=form_in, form_out=form_out)


def _bbox(m):
    ys, xs = np.nonzero(m)
    return None if not len(xs) else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def calibrate(o, name: str, fw: int, fh: int, rows):
    """fit the module's FORMED word onto the GUI word: (c, ox, oy) = scale of its box and origin shift in world px, and the IoU of the two silhouettes. None when it cannot build / draw."""
    import cv2
    from mograph.modules.core import Timing
    with FL.world(fw, fh) as (k, cam3d):
        ww, wh = cam3d.W, cam3d.H
        flat = cv2.resize(o.flat.mask(fw, fh), (ww, wh), interpolation=cv2.INTER_AREA) > 0.5
        bf = _bbox(flat)
        if bf is None:
            return None
        cf = np.array([(bf[0] + bf[2]) / 2.0, (bf[1] + bf[3]) / 2.0])
        fit, iou = (1.0, 0.0, 0.0), 0.0
        for it in range(3):
            m = build_mod(o, name, k, fit, rows, Timing(0.6, 0.4, 0.4))
            t = m.timing.form + 0.5 * m.timing.hold
            _, a = rgba(m.layers(t), ww, wh)
            mm = a > 0.5
            bm = _bbox(mm)
            if bm is None:
                return None
            iou = float((mm & flat).sum()) / max(float((mm | flat).sum()), 1.0)
            if it == 2:
                break
            cm = np.array([(bm[0] + bm[2]) / 2.0, (bm[1] + bm[3]) / 2.0])
            c_new = float(np.clip(min((bf[2] - bf[0]) / max(bm[2] - bm[0], 1), (bf[3] - bf[1]) / max(bm[3] - bm[1], 1)), 0.5, 2.0))
            org = np.array([o.lay["cx"] * k + fit[1], o.lay["cy"] * k + fit[2]])
            cm2 = org + (cm - org) * c_new                          # the bbox centre after the box is scaled about its origin
            sh = cf - cm2
            fit = (fit[0] * c_new, fit[1] + float(sh[0]), fit[2] + float(sh[1]))
        return dict(fit=list(fit), iou=round(iou, 3))


def grab(video, ps, wanted: set[int]) -> dict:
    """{frame: BGR} of the wanted frames in one decode pass."""
    from tools.app import overlay_mode as OM
    out, last = {}, max(wanted) if wanted else -1
    for f, im in enumerate(OM._frames(video, ps["width"], ps["height"])):
        if f in wanted:
            out[f] = im
        if f >= last:
            break
    return out


# ------------------------------------------------------------------------------------------------ the windows
class MGap:
    """one window of module frames. kind: pair (A deconstructs, library transition, B constructs: frames [ws, L)), enter (the first word constructs), exit (the last word deconstructs: frames [ws, we))."""

    def __init__(self, kind, a=None, b=None, ws=0, L=0, we=0, why="", size=(540, 960), fps=25.0):
        self.kind, self.a, self.b, self.ws, self.L, self.we, self.why = kind, a, b, ws, L, we, why
        self.size, self.fps = size, fps
        self.seq = self.mod = None
        self.t0 = self.D = 0.0
        self.handoff = self.tr = ""

    def info(self) -> dict:
        return dict(kind=self.kind, a=self.a.w["n"] if self.a else None, b=self.b.w["n"] if self.b else None, window=[self.ws, self.we], land=self.L,
                    a_module=getattr(self.a, "module", None) if self.a else None, b_module=getattr(self.b, "module", None) if self.b else None, handoff=self.handoff, transition=self.tr, why=self.why)

    def t_at(self, f: int) -> float:
        if self.kind == "pair":
            return max(0.0, self.t0 + self.D + (f - self.L) / self.fps)
        if self.kind == "enter":
            return max(0.0, self.mod.timing.form + (f - self.L) / self.fps)
        return self.mod.timing.form + self.mod.timing.hold + (f - self.ws) / self.fps

    def layers(self, f: int):
        with FL.world(*self.size):
            return (self.seq if self.kind == "pair" else self.mod).layers(self.t_at(f))


def _port(v):
    return np.array([float(v[0]), float(v[1]), 0.0])


def _edge(rng, o, size):
    """a frame edge direction for the first / last word's strands (seeded, toward the side with the most room)."""
    fw, fh = size
    cx, cy = o.lay["cx"], o.lay["cy"]
    room = np.array([fw - cx, cx, fh - cy, cy])
    d = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    p = room / room.sum()
    return _port(d[int(rng.choice(4, p=p))])


def choose(objs, seed: int) -> list:
    """the pool order per word: tl.module (the user's override) first, then a seeded rotation of POOL with the previous word's module left out."""
    rng = np.random.default_rng(seed)
    order = [POOL[j] for j in rng.permutation(len(POOL))]
    out, prev = [], None
    for i, o in enumerate(objs):
        if o.metal:
            out.append([])
            prev = None
            continue
        ov = (o.w["tl"].get("module") or "").strip() if isinstance(o.w["tl"].get("module"), str) else ""
        rot = order[(2 * i) % len(order):] + order[:(2 * i) % len(order)]
        cands = ([ov] if ov else []) + [m for m in rot if m != prev and m != ov]
        out.append(cands)
        prev = cands[0]
    return out


def plan(objs, video, ps, cuts=None, seed=None, work=None, log=print) -> list[MGap]:
    """module windows for every gap whose two words both got a module (+ the first word's construction, the last word's exit); sets o.module / o.fm_in / o.fm_out. Words are drawn flat where no module fits."""
    from mograph.modules.core import Timing, get_module
    from mograph.modules.sequence import OUTLINE_KINDS, Sequence
    fps, size = float(ps["fps"]), (ps["width"], ps["height"])
    fw, fh = size
    for o in objs:
        o.module, o.fm_in, o.fm_out, o.fm = None, None, None, None
    if not objs:
        return []
    seed = int(hashlib.sha1(json.dumps([[o.w["text"], o.w["start"], o.w["end"]] for o in objs]).encode()).hexdigest()[:8], 16) if seed is None else int(seed)
    rng = np.random.default_rng(seed + 11)
    frames = grab(video, ps, {o.w["start"] for o in objs})
    for o in objs:
        if o.w["start"] in frames:
            o.colours(frames[o.w["start"]])
    cands = choose(objs, seed)
    wd = Path(work) / "modules" if work else None
    taken = None
    for i, o in enumerate(objs):                        # the module per word: the first of its candidates whose formed silhouette matches the GUI word
        if o.fill is None or not cands[i]:
            continue
        o.fm_pal, rows = palette_of(o), rows_of(o.w["text"], o.w["tl"])
        for name in [m for m in cands[i] if m != taken][:3] or cands[i][:3]:
            key = hashlib.sha1(json.dumps([name, o.w["text"], o.w["font"], o.w["tl"], o.lay["cx"], o.lay["cy"], o.lay["s"], size, o.fill, o.ring, o.solid], sort_keys=True, default=str).encode()).hexdigest()[:14]
            cf = wd / f"{key}.json" if wd else None
            try:
                cal = json.loads(cf.read_text()) if cf and cf.is_file() else calibrate(o, name, fw, fh, rows)
                if cal and cf and not cf.is_file():
                    cf.parent.mkdir(parents=True, exist_ok=True)
                    cf.write_text(json.dumps(cal))
            except Exception as e:
                log(f"module {name} for {o.w['text']!r}: {type(e).__name__} {e}")
                continue
            if cal and cal["iou"] >= IOU_MIN:
                o.module, o.fm = name, dict(fit=tuple(cal["fit"]), iou=cal["iou"], rows=rows)
                taken = name
                break
            log(f"module {name} for {o.w['text']!r}: formed word does not match the GUI letters (IoU {cal['iou'] if cal else None}): next")
    ex_f, hold_f = int(round(D_FULL[0] * fps)), int(round(HOLD_S * fps))
    moves = [(c - m["h"], c + m["h"]) for c, m in (cuts or {}).items()]
    hit = lambda ws, we: any(lo < we and ws < hi for lo, hi in moves)      # noqa: E731
    gaps: list[MGap] = []

    def mk(o, timing, entry, exit_, form_in="lines", form_out="lines"):
        return build_mod(o, o.module, 960.0 / max(size), o.fm["fit"], o.fm["rows"], timing, entry, exit_, form_in, form_out)

    prev_kind = None
    for i, b in enumerate(objs):
        if b.module is None:
            continue
        a = objs[i - 1] if i else None
        L = b.w["start"]
        try:
            if a is None:
                f = min(D_FULL[2], L / fps)
                if f < ENTRY_MIN_S or hit(L - int(round(f * fps)), L):
                    continue
                g = MGap("enter", None, b, L - int(round(f * fps)), L, L, "first word constructs", size, fps)
                with FL.world(fw, fh):
                    g.mod = mk(b, Timing(f, 0.5, 0.4), _edge(rng, b, size), "right")
                gaps.append(g)
                b.fm_in = g
                continue
            if a.module is None:
                continue
            Rmin = a.w["start"] + hold_f
            D = min(sum(D_FULL), (L - Rmin) / fps)
            if D < D_MIN:
                continue
            q = D / sum(D_FULL)
            x, T, f = (v * q for v in D_FULL)
            d = np.array([b.lay["cx"] - a.lay["cx"], b.lay["cy"] - a.lay["cy"]])
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0])
            ks = [kd for kd in OUTLINE_KINDS if kd != prev_kind]
            mode = HANDOFFS[int(rng.integers(len(HANDOFFS)))]
            err, g = None, None
            for j in range(len(ks)):
                kd = ks[(int(rng.integers(len(ks))) + j) % len(ks)]
                try:
                    with FL.world(fw, fh):
                        A = mk(a, Timing(0.3, 0.3, x), "left", _port(d))
                        B = mk(b, Timing(f, 0.5, 0.5), _port(-d), "right")
                        tr = get_module(kd)(dur=T, seed=seed + i, unravel=True, ghost=False, pads=False, web=False, handoff=mode)
                        seq = Sequence([A, tr, B])
                        tB = seq.timeline[2][1]
                        g = MGap("pair", a, b, 0, L, L, "", size, fps)
                        g.seq, g.t0 = seq, A.timing.form + A.timing.hold
                        g.D = tB + B.timing.form - g.t0
                    g.handoff, g.tr = mode, kd
                    break
                except Exception as e:
                    err, g = f"{kd}: {type(e).__name__} {e}", None
            if g is None:
                log(f"gap {a.w['text']} -> {b.w['text']}: {err}")
                continue
            g.ws = L - int(round(g.D * fps))
            if g.ws < Rmin or g.ws >= a.w["end"] or hit(g.ws, L):
                continue
            prev_kind = g.tr
            g.we = L
            gaps.append(g)
            b.fm_in, a.fm_out = g, g
        except Exception as e:
            log(f"gap into {b.w['text']}: {type(e).__name__} {e}")
    o = objs[-1]                                                     # the last word deconstructs to a frame edge
    if o.module is not None and o.fm_out is None:
        end = o.w["end"]
        x = D_FULL[0]
        R = max(o.w["start"] + hold_f, end - int(round(x * fps)))
        if end - R >= int(round(0.3 * fps)) and not hit(R, end):
            try:
                g = MGap("exit", o, None, R, end, end, "last word deconstructs", size, fps)
                with FL.world(fw, fh):
                    g.mod = mk(o, Timing(0.3, 0.3, max(0.3, (end - R) / fps)), "left", _edge(rng, o, size))
                gaps.append(g)
                o.fm_out = g
            except Exception as e:
                log(f"last word exit: {type(e).__name__} {e}")
    return gaps


def gaps_of(objs) -> list[MGap]:
    seen, out = set(), []
    for o in objs:
        for g in (getattr(o, "fm_in", None), getattr(o, "fm_out", None)):
            if g is not None and id(g) not in seen:
                seen.add(id(g))
                out.append(g)
    return sorted(out, key=lambda g: g.ws)


def index(objs) -> dict:
    return {f: g for g in gaps_of(objs) for f in range(g.ws, g.we)}


def describe(objs) -> list[dict]:
    return [g.info() for g in gaps_of(objs)]


# ------------------------------------------------------------------------------------------------ one frame of a window
def frame(g: MGap, im: np.ndarray, f: int, mattes) -> np.ndarray:
    """the composed frame f: the flat GUI words cross-fade (XF frames) with their module's formed word, the module's own frames in between (strands anywhere; the subject matte over them for behind / matte looks)."""
    import cv2
    a, b = g.a, g.b
    out = im.astype(np.float32)
    wa = wb = 0.0
    if a is not None and f < a.w["end"]:
        wa = 1.0 - _ss((f - g.ws) / XF)
        if wa > 0.0:
            a.colours(im)
            out = out * (1 - wa) + a.over(im, f, mattes, min(f, a.w["end"] - 1)).astype(np.float32) * wa
    if b is not None and g.kind in ("pair", "enter"):
        wb = _ss((f - (g.L - XF)) / XF)
        if wb > 0.0:
            b.colours(im)
            out = out * (1 - wb) + b.over(im, f, mattes, g.L).astype(np.float32) * wb
    wm = (1.0 - wa if g.kind in ("pair", "exit") else 1.0) * (1.0 - wb)
    if wm > 0.0:
        with FL.world(*g.size) as (k, cam3d):
            ww, wh = cam3d.W, cam3d.H
            pm, al = rgba(g.layers(f), ww, wh)
        h, w = im.shape[:2]
        if (ww, wh) != (w, h):
            pm, al = cv2.resize(pm, (w, h), interpolation=cv2.INTER_CUBIC), cv2.resize(al, (w, h), interpolation=cv2.INTER_LINEAR)
        keep = 1.0
        if mattes is not None:
            ma = a is not None and a.look in FL.MATTE_LOOKS
            mb = b is not None and b.look in FL.MATTE_LOOKS
            u = _ss((f - g.ws) / max(g.L - g.ws, 1)) if g.kind == "pair" else (1.0 if g.kind == "enter" else 0.0)
            kk = (1 - u) * ma + u * mb
            if kk > 0:
                keep = 1.0 - kk * mattes.get(f)
        keep = keep * wm
        out = out * (1.0 - (al * keep)[:, :, None]) + pm * (keep[:, :, None] if isinstance(keep, np.ndarray) else keep)
    return out.clip(0, 255).astype(np.uint8)
