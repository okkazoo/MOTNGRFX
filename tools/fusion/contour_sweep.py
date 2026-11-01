"""contour_sweep: the contour lines of the footage's forms draw ON and OFF along the forms, swept in the direction of the transition they belong to
(user 2026-10-30: "I want the contour lines to animate over the forms of the elements it is based on and move on and off in the direction of the transform - they could be then
input and output of the transforms"; line style, same day: "I dont want dashed lines around them like they are a photoshop selection - I want the edge to give a vector and movement
for an animated line of differing width and agility and messyness to surround it").

WHAT IS DRAWN. Contours of the shot's Alpha-Gen subject matte (iso 0.5, `mograph.modules.contour_ports.matte_outlines`; optional depth iso-lines / luma iso-lines, `sources`), re-traced on
every frame, so the lines follow the forms (the matte follows its clip: in a 3D cut the chain's matte is the clip's patched one, so the lines take the clip's transform). Each contour
is drawn as `n_strokes` BRUSH STROKES, never dashes:
    * variable width along the stroke: base * (0.3 .. 1.6) from smooth noise over the arc length (drifting with time), tapered ends (pressure-like), a filled ribbon built from the
      offset curves (anti-aliased, crisp, no glow, no blur);
    * movement: the visible run travels ALONG the contour and grows / shrinks with the sweep, its lead end runs ahead of the tail (different easings), a short thin overshoot past the front;
    * messiness: 2-3 strokes per contour, offset from each other, differently timed (stroke j starts later), a low-frequency wobble off the edge (`wobble` x subject size, smooth over arc
      AND time, never per-frame jitter), occasional small breaks. All seeded: the same (piece seed, event, contour, stroke) draws the same stroke.

SHOT CUTS ARE OUT (2026-10-30, user on `16_app/core_check/contour_sweep/cut_s1-s2_sweep.mp4`: "don't use this - just do a simple straight push in or pull out"): `plan` / `weave` make NO event for a
3D / pan / window shot cut (`SHOT_CUT_EVENTS = False`); the sweep stays on WORD-TO-WORD transitions inside a shot (the ship-#8 `contour_morph` owns those gaps when the word has a matte, the sweep covers the rest).
A shot cut that wants a plain move uses `orbit:push` / `orbit:pullback` (straight along the axis, no orbit, no rotation, no lines). The text below about cut directions describes the disabled path.

HOW IT SWEEPS (`phase`): the transition has a direction `f` (unit screen vector, or radial out / in from the centre). A point of a contour has s' = its position along f, scaled to the
subject's own extent (0 = the entry side of the form, 1 = the exit side), and u2 = its arc distance from the contour's entry point (0..1, the two arms of a closed contour grow towards each
other). With the front F(t) and the arc head H(t):
    OFF  (word / shot A, first half)   a point is drawn while s' > F and u2 > G   = the lines are consumed from the entry side, in the direction f, along the forms
    ON   (word / shot B, second half)  a point is drawn while s' <= F + tip and u2 <= H = the lines grow from the same side, in the direction f
    RESOLVE                             the full contour, the extra strokes fade first, the wobble goes to 0 (the stroke sits ON the subject edge), then the alpha goes to 0 BEFORE
                                        the landed word of B (`END_GUARD_S`).
DIRECTIONS (`flow_from_sides`): f = from A's exit side to B's entry side (g04 depth_slide_type: A leaves left, B comes in from the right, the strands flow left -> right), which for a
`pan:<d>` / `depth_dolly:<d>` is the named direction d (the camera travel), for `depth_pivot` = right (cut_kinds.sides), for an orbit in the `slide` style its `dir`; orbit:push = radial
OUT from the centre, orbit:pullback = radial IN, orbit:fly = its `dir` (default right); window_zoom = radial out; window_cut / swap / other = a seeded side. Word-to-word transitions take
the words' `exit` / `entry` sides when the piece names them, else a seeded side per transition (two neighbours never alike).

LEGIBILITY (the guard, tested): the lines are zero in `clean_windows` = from 2 frames before each word's landing to landing + max(1 beat, 0.5 s) (capped at the release), ramp in / out over
`RAMP_S`; B's lines are gone `END_GUARD_S` before B's word lands; `avoid` (the type alpha) is cut out of the lines, so a stroke never crosses a letter. A transition whose B word lands
within `MIN_B_S` of the cut gets only A's draw-off.

PLAN: `plan(piece, seed) -> dict` (pure, JSON, no arrays) with `events`, `clean`, `params`, `color`; `frame_state(plan, t, sid)` -> the active phases; the compose op `contour_sweep` draws
them (geometry with cv2 like `contour_lines`, composite on the compose backend: torch on CUDA, numpy fallback); `recipe_ops(piece, ops, sid)` puts the op into a compose chain when
`compose.contour_sweep` is true (or the request named the `contour_sweep` effect, or `plan.contour_sweep` exists), called by recipe_eval.build_recipe and piece_runner.Piece.recipe: no planner
edit needed. `edge_alpha(mask, frame, ...)` is the idle (no transition) form: a live brush outline for plate_layers' contrast-moment edge line (mode `sweep`).
"""
from __future__ import annotations

import copy
import math
import random
import zlib

import numpy as np

import cv2

from tools.fusion import compose as C

VERSION = 1
SIDES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}
OPP = {"left": "right", "right": "left", "up": "down", "down": "up"}
RADIAL = {"out": 1, "in": -1}
SHOT_CUT_EVENTS = False     # 2026-10-30 (user on cut_s1-s2_sweep.mp4: "don't use this - just do a simple straight push in or pull out"): NO contour sweep on SHOT cuts (3D / pan / window); word-to-word transitions only
END_GUARD_S = 0.10          # B's lines are gone this long before B's word lands
RAMP_S = 0.12               # alpha ramp next to a clean window
MIN_B_S = 0.30              # a B phase shorter than this is not drawn
CLEAN_PRE_FRAMES = 2        # clean from this many frames before the landing
CLEAN_MIN_S = 0.5           # ... to landing + max(1 beat, this)
MIN_GAP_S = 0.45            # shortest word-to-word transition that gets a sweep

PARAMS = dict(
    width=2.0 / 360.0,       # base stroke width, share of the frame width (2.0 px at 360 wide; 1.5-3 px band after the 0.3..1.6 multiplier)
    width_range=(0.3, 1.6),  # multiplier band along the stroke (pressure-like)
    n_strokes=2,             # strokes per contour (1-3)
    wobble=0.018,            # offset off the edge, share of the subject size (1-3 %), smooth over arc and time
    speed=1.0,               # flow speed of the width noise / wobble / idle runs
    gaps=1.0,                # gap rate multiplier (0 = no breaks)
    max_contours=4,
    sources=("matte",),      # matte | depth | luma
    taper=0.16,              # share of a run's length that tapers at each end
    stagger=0.14,            # stroke j starts j * stagger of the phase later
    overshoot=0.035,         # the lead end runs this far (in s') past the front
    min_px=0.7,              # thinnest drawn width (px)
    border=0.02,             # no stroke within this share of the frame border (a matte cut by the frame never gets a stroke along it)
    straight=0.25,           # a near-straight run longer than this share of the frame side is a frame / card edge, not a form: not drawn (0 = off)
    straight_dev=0.005,      # 'near-straight' = every point within this share of the frame side of the run's chord
    guard=True,              # False = draw every contour point (letters: plate_layers text_as_mask)
)


# ================================================================================================================ small pure helpers
def _seed(*parts) -> int:
    return zlib.crc32(":".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def _ease_in(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 2.2


def unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else None


def flow_from_sides(a_exit, b_entry):
    """unit vector from A's exit side to B's entry side (a side = left / right / up / down); None for equal / unknown sides."""
    if a_exit in SIDES and b_entry in SIDES:
        va, vb = SIDES[a_exit], SIDES[b_entry]
        return unit((vb[0] - va[0], vb[1] - va[1]))
    return None


def cut_flow(kind, style=None, dir=None, rng=None):
    """the sweep direction of a SHOT cut kind -> dict(f=(fx, fy) | None, radial=0 | +1 | -1, sides=(exit A, entry B) | None, how=str); None when the kind has no transform to ride
    (crossfade / hard / unknown). rng: seeded choice for the kinds without a direction (window cut / swap / other)."""
    from tools.fusion import cut_kinds as CK
    k = CK.normalize(kind) or str(kind)
    fam = CK.family_of(k)
    if k in CK.PLAIN:
        return None
    if k == "orbit:push":
        return dict(f=None, radial=1, sides=None, how="orbit:push = radial out from the centre")
    if k == "orbit:pullback":
        return dict(f=None, radial=-1, sides=None, how="orbit:pullback = radial in to the centre")
    if k == "window_zoom":
        return dict(f=None, radial=1, sides=None, how="window_zoom = radial out from the centre (the camera scales into a letter)")
    if k == "orbit:fly":
        d = dir if dir in OPP else "right"
        s = (OPP[d], d)
        return dict(f=SIDES[d], radial=0, sides=s, how=f"orbit:fly = its dir {d}")
    sd = CK.sides(k, style, dir)
    if sd is not None:
        f = flow_from_sides(*sd)
        return dict(f=f, radial=0, sides=tuple(sd), how=f"{k}: A leaves {sd[0]}, B enters from {sd[1]}")
    if fam == "window":
        r = rng or random.Random(0)
        d = r.choice(list(SIDES))
        s = (OPP[d], d)
        return dict(f=SIDES[d], radial=0, sides=s, how=f"{k}: no direction of its own, seeded side {d}")
    if fam in ("3d", "pan"):                               # a bare family (direction chosen later): seeded
        r = rng or random.Random(0)
        d = r.choice(list(SIDES))
        s = (OPP[d], d)
        return dict(f=SIDES[d], radial=0, sides=s, how=f"{k}: bare family, seeded side {d}")
    return None


def _side_of(w, key):
    v = w.get(key)
    if isinstance(v, dict):
        v = v.get("side") or v.get("dir")
    return v if v in SIDES or v in ("in", "out") else None


def word_flow(wa, wb, rng, prev=None):
    """sweep direction of a word-to-word transition: from A's exit side to B's entry side (the words' `exit` / `entry` keys when set, else a seeded side; never the same as the previous
    transition's). -> dict like cut_flow."""
    ea, eb = _side_of(wa, "exit"), _side_of(wb, "entry")
    if ea in RADIAL or eb in RADIAL:
        out = ea == "out" or eb == "in"
        return dict(f=None, radial=1 if out else -1, sides=None, how=f"word transition: A exit {ea}, B entry {eb} = radial {'out' if out else 'in'}")
    f = flow_from_sides(ea, eb) if ea and eb else None
    if f is None and ea in SIDES and eb is None:
        f = SIDES[OPP[ea]]
    if f is None:
        sides = [s for s in SIDES if s != prev]
        d = rng.choice(sides)
        return dict(f=SIDES[d], radial=0, sides=(OPP[d], d), how=f"word transition: no sides named, seeded {d}")
    return dict(f=f, radial=0, sides=(ea, eb), how=f"word transition: A exit {ea} -> B entry {eb}")


# ================================================================================================================ the plan
def _holds(piece, music=None, chk=None, holds=None):
    """(hit, release) per word, piece seconds: the planner's readout, else spec timing, else the words' t0 / t1 with form / exit times."""
    if holds:
        return [(float(a), float(b)) for a, b in holds]
    try:
        from tools.fusion import plan_rules as PR
        return [(float(a), float(b)) for a, b in PR.word_holds(piece, music, chk)]
    except Exception:                                      # noqa: BLE001 never break a piece for the timing
        out = []
        for w in (piece.get("type") or {}).get("words") or []:
            t0, t1 = float(w.get("t0", 0.0)), float(w.get("t1", 0.0))
            out.append((t0 + float(w.get("form_s", 0.45)), max(t0 + float(w.get("form_s", 0.45)) + 0.2, t1 - float(w.get("exit_s", 0.5)))))
        return out


def _beat_s(piece, music):
    bpm = ((music or {}).get("bpm") or (piece.get("type") or {}).get("bpm") or (piece.get("audio") or {}).get("bpm") or 90.0)
    return 60.0 / max(float(bpm), 30.0)


def clean_windows(holds, beat_s, fps):
    """windows where no contour line may be drawn: [hit - CLEAN_PRE_FRAMES frames, hit + max(1 beat, CLEAN_MIN_S)] capped at the release (>= 0.3 s)."""
    out = []
    for hit, rel in holds:
        end = min(hit + max(beat_s, CLEAN_MIN_S), max(rel, hit + 0.3))
        out.append([round(hit - CLEAN_PRE_FRAMES / fps, 4), round(end, 4)])
    return out


def _palette_color(piece):
    pal = (piece.get("type") or {}).get("palette") or {}
    cands = [c for c in (pal.get("accents") or []) if isinstance(c, (list, tuple)) and len(c) >= 3]

    def chroma(c):
        return max(c[:3]) - min(c[:3])
    if cands:
        c = max(cands, key=chroma)
    else:
        c = pal.get("fg") or (240, 236, 228)
    return [round(float(v) / 255.0, 4) for v in c[:3]]


def _cut_window(piece, cut, k, tb, fps):
    """(a0, tc, b1, land) seconds of one shot cut: A's window starts a0, the picture changes at tc, B's window ends b1, B's word lands at `land` (None = unknown)."""
    from tools.fusion import cut_kinds as CK
    kind = CK.normalize(cut.get("cut") or cut.get("kind")) or str(cut.get("cut"))
    fam = CK.family_of(kind)
    if fam == "pan":
        bs = ((piece.get("plan") or {}).get("pan") or {}).get("boundaries") or []
        best = min(bs, key=lambda b: abs(float(b.get("hit", b.get("t1", 0.0))) - tb), default=None)
        if best is not None and abs(float(best.get("hit", best.get("t1", 0.0))) - tb) < 1.0:
            t0, t1 = float(best["t0"]), float(best["t1"])
            return t0, t1, t1, float(best.get("hit", t1))                 # the travel is A's draw-off; B lands at the end of it (no room for a B phase)
        return tb - 1.0, tb - 0.1, tb - 0.1, tb
    if fam == "3d":
        n, i2 = 72, 27
        try:
            from tools.fusion import pivot_cut as PC
            n, i2 = PC.window_frames(kind, cut.get("params"))
        except Exception:                                  # noqa: BLE001 engine not importable here: the default window
            pass
        return tb - i2 / fps, tb, tb + (n - i2) / fps, None
    if kind == "window_zoom":
        return tb - 0.5, tb, tb + 0.9, None
    return tb - 0.6, tb, tb + 0.6, None


def plan(piece, seed=None, music=None, chk=None, holds=None, params=None, fps=None) -> dict:
    """the contour sweep plan of a piece: one event per shot cut with a transform (3D / pan / window kinds) and per word-to-word transition, each with its direction and its
    A-off / B-on / resolve times, plus the clean windows. Pure (no arrays), JSON-able, deterministic in (piece, seed)."""
    from tools.fusion import cut_kinds as CK
    comp = piece.get("compose") or {}
    fps = float(fps or piece.get("fps") or 24)
    if seed is None:
        seed = ((piece.get("plan") or {}).get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0))
    seed = int(seed)
    cfg = comp.get("contour_sweep")
    pr = dict(PARAMS)
    pr.update({k: v for k, v in (cfg if isinstance(cfg, dict) else {}).items() if k in PARAMS})
    pr.update({k: v for k, v in (params or {}).items() if k in PARAMS})
    words = (piece.get("type") or {}).get("words") or []
    hs = _holds(piece, music, chk, holds)
    beat = _beat_s(piece, music)
    clean = clean_windows(hs, beat, fps)
    lay = piece.get("layout") or {}
    ids = [s.get("id") for s in piece.get("shots", [])]
    events = []

    def land_after(t):
        c = [h for h, _ in hs if h >= t - 0.05]
        return min(c) if c else None

    for k, cut in enumerate((comp.get("cuts") or []) if SHOT_CUT_EVENTS else []):          # shot cuts get NO sweep (SHOT_CUT_EVENTS); the loop stays for a deliberate re-enable
        kind = CK.normalize(cut.get("cut") or cut.get("kind")) or str(cut.get("cut") or "")
        a = cut.get("after")
        if a not in ids or ids.index(a) + 1 >= len(ids):
            continue
        b = ids[ids.index(a) + 1]
        if b not in lay:
            continue
        tb = float(lay[b]["start"]) / fps
        rng = random.Random(_seed(seed, "cut", k))
        fl = cut_flow(kind, cut.get("style") or (cut.get("params") or {}).get("style"), cut.get("dir") or (cut.get("params") or {}).get("dir"), rng)
        if fl is None:
            continue
        a0, tc, b1, land = _cut_window(piece, cut, k, tb, fps)
        land = land if land is not None else land_after(tc)
        events.append(_event(len(events), "cut", kind, a, b, a0, tc, b1, land, fl, cut.get("word"), k))
    covered = {e["word"] for e in events if e.get("word") is not None}
    covered |= {int(w) for w in (((piece.get("plan") or {}).get("contour_morph") or {}).get("words") or [])}      # ship #8: the contour_morph owns those word-to-word gaps
    cut_b_s = []
    try:
        from tools.fusion import pivot_cut as PC
        cut_b_s = sorted(float(lay[sp["b"]]["start"]) / fps for sp in PC.specs(piece) if sp["b"] in lay)
    except Exception:                          # noqa: BLE001  (engine not importable here: no alignment)
        pass
    prev = None
    for i in range(len(words) - 1):
        wa, wb = words[i], words[i + 1]
        tr = wa.get("transition")
        tk = (tr.get("kind") or tr.get("module")) if isinstance(tr, dict) else tr
        trs = (piece.get("type") or {}).get("transitions") or []
        if not tk and i < len(trs):
            tt = trs[i]
            tk = (tt.get("kind") or tt.get("module")) if isinstance(tt, dict) else tt
        if tk in (None, "", "cut", "none", "hard", "crossfade") or i in covered or i + 1 >= len(hs):
            continue
        a0 = hs[i][1]
        land = hs[i + 1][0]
        if land - a0 < MIN_GAP_S:
            continue
        rng = random.Random(_seed(seed, "word", i))
        if tk == "pan" and isinstance(tr, dict) and tr.get("dir") in SIDES:
            fl = cut_flow("pan:" + tr["dir"])
        else:
            fl = word_flow(wa, wb, rng, prev)
        if fl is None:
            continue
        if fl.get("f"):
            prev = OPP.get(next((n for n, v in SIDES.items() if v == fl["f"]), ""), None)
        D = land - END_GUARD_S - a0
        tc = a0 + 0.4 * D
        for tb in cut_b_s:                      # a 3D shot cut inside the span: B's lines start drawing the moment B takes over (the pivot's settle), no empty gap after A's lines are off
            if a0 + 0.1 < tb < land - END_GUARD_S - MIN_B_S:
                tc = tb
                break
        events.append(_event(len(events), "word", str(tk), None, None, a0, tc, land - END_GUARD_S, land, fl, i, i))
    # a word event that overlaps a cut event is dropped (the cut owns the span)
    keep = []
    for e in events:
        if e["type"] == "word" and any(o["type"] == "cut" and o["t_a0"] < e["t_e"] and e["t_a0"] < o["t_e"] for o in events):
            continue
        keep.append(e)
    for j, e in enumerate(keep):
        e["i"] = j
    req = list(((piece.get("plan") or {}).get("effects") or {}).get("requested") or [])
    return dict(version=VERSION, seed=seed, fps=fps, params=pr, color=_palette_color(piece), events=keep, clean=clean, beat_s=round(beat, 4),
                forced=bool(comp.get("contour_sweep")) or "contour_sweep" in req)


def _event(i, typ, kind, shot_a, shot_b, a0, tc, b1, land, fl, word, k):
    """one transition: A's draw-off [a0, tc], B's draw-on [tc, tm], resolve [tm, te] with te <= landing - END_GUARD_S; a B phase under MIN_B_S is dropped (A only)."""
    te = b1 if land is None else min(b1, land - END_GUARD_S)
    has_b = (te - tc) >= MIN_B_S
    if has_b:
        tm = tc + 0.55 * (te - tc)
    else:
        tm = te = tc
    return dict(i=i, type=typ, kind=kind, shot_a=shot_a, shot_b=shot_b, t_a0=round(a0, 4), t_c=round(tc, 4), t_m=round(tm, 4), t_e=round(te, 4), land=None if land is None else round(land, 4),
                f=None if fl.get("f") is None else [round(fl["f"][0], 4), round(fl["f"][1], 4)], radial=int(fl.get("radial", 0)), sides=list(fl["sides"]) if fl.get("sides") else None,
                b=bool(has_b), how=fl.get("how"), word=word, k=k)


def clean_gate(plan_, t) -> float:
    """0 inside a clean window, ramping to 1 over RAMP_S on both sides, else 1."""
    g = 1.0
    for a, b in plan_.get("clean") or []:
        if a <= t <= b:
            return 0.0
        d = (a - t) if t < a else (t - b)
        if d < RAMP_S:
            g = min(g, _ease_io(d / RAMP_S))
    return g


def frame_state(plan_, t, sid=None) -> list:
    """the active phases at time t (piece seconds) for shot `sid` (None = any): [dict(role off|on|resolve, tau, rho, f, radial, ev)]. A's OFF only on shot A, B's ON / RESOLVE only on
    shot B (word transitions: any shot)."""
    out = []
    for e in plan_.get("events") or []:
        if t < e["t_a0"] or t > e["t_e"]:
            continue
        if t < e["t_c"]:
            if e["shot_a"] is not None and sid is not None and sid != e["shot_a"]:
                continue
            out.append(dict(role="off", tau=(t - e["t_a0"]) / max(e["t_c"] - e["t_a0"], 1e-6), rho=0.0, f=e["f"], radial=e["radial"], ev=e["i"]))
        elif e["b"]:
            if e["shot_b"] is not None and sid is not None and sid != e["shot_b"]:
                continue
            if t < e["t_m"]:
                out.append(dict(role="on", tau=(t - e["t_c"]) / max(e["t_m"] - e["t_c"], 1e-6), rho=0.0, f=e["f"], radial=e["radial"], ev=e["i"]))
            else:
                out.append(dict(role="resolve", tau=1.0, rho=(t - e["t_m"]) / max(e["t_e"] - e["t_m"], 1e-6), f=e["f"], radial=e["radial"], ev=e["i"]))
    return out


def front_axis(f, radial):
    """('dir', (fx, fy)) | ('radial', +1 / -1): how s' is measured."""
    return ("radial", int(radial)) if radial else ("dir", tuple(f or (1.0, 0.0)))


# ================================================================================================================ strokes
class Noise:
    """smooth periodic noise in [-1, 1] over the arc fraction x (integer frequencies: closes seamlessly on a ring) and slowly over time t."""

    def __init__(self, seed, n=4, fmax=5, tspeed=0.35, std=0.5):
        r = np.random.default_rng(seed)
        self.f = r.integers(1, fmax + 1, n).astype(np.float32)
        self.g = r.uniform(-1.0, 1.0, n).astype(np.float32) * tspeed
        self.p = r.uniform(0, 2 * np.pi, n).astype(np.float32)
        a = 1.0 / np.sqrt(self.f)
        self.a = a / np.sqrt(np.sum(a * a) * 0.5) * std

    def __call__(self, x, t=0.0):
        x = np.asarray(x, np.float32)[:, None]
        v = np.sin(2 * np.pi * (self.f[None] * x + self.g[None] * t) + self.p[None]) * self.a[None]
        return np.clip(v.sum(1), -1.0, 1.0)


def _resample_closed(p, step):
    q = np.concatenate([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    if L < 6.0:
        return None, L
    m = int(max(24, L / step))
    s = np.linspace(0.0, L, m, endpoint=False)
    return np.stack([np.interp(s, cum, q[:, 0]), np.interp(s, cum, q[:, 1])], 1).astype(np.float32), L


def _smooth_closed(p, k):
    if k <= 0 or len(p) < 2 * k + 3:
        return p
    q = np.concatenate([p[-k:], p, p[:k]])
    ker = np.ones(2 * k + 1, np.float32) / (2 * k + 1)
    return np.stack([np.convolve(q[:, i], ker, mode="valid") for i in range(2)], 1).astype(np.float32)


def extract_contours(matte=None, depth=None, luma=None, sources=("matte",), max_contours=4, size=None):
    """closed contours (list of (n, 2) float32 px arrays in the SIZE grid, clockwise on screen, entry NOT yet chosen) of the matte (iso 0.5), depth iso-lines and luma iso-lines. Uses
    mograph.modules.contour_ports when importable (the extraction the ports use), else a local cv2 version."""
    out = []
    h, w = (matte if matte is not None else (depth if depth is not None else luma)).shape[:2]
    ow, oh = (int(size[0]), int(size[1])) if size else (w, h)
    sc = np.array([ow / float(w), oh / float(h)], np.float32)
    ml = max(h, w) * 0.07
    if "matte" in sources and matte is not None:
        m8 = np.clip(np.asarray(matte, np.float32) * 255.0, 0, 255).astype(np.uint8)
        try:
            from mograph.modules.contour_ports import matte_outlines
            cs = matte_outlines(m8, 0.5, max_lines=2 * max_contours, smooth=2.0, min_len=ml)
        except Exception:                                  # noqa: BLE001
            cs = _local_outlines(m8, ml, 2 * max_contours)
        out += [c * sc for c in cs]
    if "depth" in sources and depth is not None:
        try:
            from mograph.modules.contour_ports import depth_levels, iso_lines
            d8 = np.clip(np.asarray(depth, np.float32) * 255.0, 0, 255).astype(np.uint8)
            m8 = np.clip(np.asarray(matte, np.float32) * 255.0, 0, 255).astype(np.uint8) if matte is not None else None
            lv = depth_levels(d8, m8, n=3)
            for lines in iso_lines(np.asarray(depth, np.float32), [float(x) for x in lv], min_len=ml, keep=1):
                for run, closed in lines:
                    if closed:
                        out.append(np.asarray(run, np.float32) * sc)
        except Exception:                                  # noqa: BLE001
            pass
    if "luma" in sources and luma is not None:
        try:
            lu = np.asarray(luma, np.float32)
            for q in (35, 65):
                t = float(np.percentile(lu, q))
                cs = _local_outlines(np.clip((lu > t) * 255, 0, 255).astype(np.uint8), ml, 1, thresh=0.5)
                out += [c * sc for c in cs]
        except Exception:                                  # noqa: BLE001
            pass
    return out[:2 * max_contours + 2]                      # the largest are kept; stroke_alpha drops the frame / card ones and then keeps `max_contours`


def _local_outlines(m8, min_len, n, thresh=0.5):
    a = cv2.GaussianBlur(m8.astype(np.float32) / 255.0, (0, 0), 2.0)
    b = cv2.copyMakeBorder((a > thresh).astype(np.uint8), 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    cnts = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    res = []
    for c in cnts:
        p = c[:, 0, :].astype(np.float32) - 2
        if len(p) < 8:
            continue
        p = _smooth_closed(p, 3)
        L = float(np.linalg.norm(np.diff(np.concatenate([p, p[:1]]), axis=0), axis=1).sum())
        if L >= min_len:
            area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
            res.append((abs(area), p if area >= 0 else p[::-1].copy()))
    res.sort(key=lambda x: -x[0])
    return [c for _, c in res[:n]]


def _projection(P, axis, size):
    """s of every point: position along the sweep direction (dir) or the distance from the centre (radial), in px-free units (-1..1 across the frame)."""
    W, H = size
    if axis[0] == "dir":
        fx, fy = axis[1]
        return ((P[:, 0] - W / 2.0) * fx + (P[:, 1] - H / 2.0) * fy) / (0.5 * (abs(fx) * W + abs(fy) * H))
    r = np.hypot(P[:, 0] - W / 2.0, P[:, 1] - H / 2.0) / (0.5 * math.hypot(W, H))
    return r if axis[1] > 0 else -r


def _runs_circ(vis):
    """runs of True of a ring: None when all True (a full ring), else a list of index arrays (a run may wrap past the end)."""
    n = len(vis)
    if vis.all():
        return None
    k = int(np.argmin(vis))
    idx = (np.arange(n) + k) % n
    return [idx[a:b + 1] for a, b in _runs(vis[idx])]


def _runs(vis):
    """runs of True in a bool array -> [(i0, i1)] (inclusive indices); a ring that is True everywhere -> [(0, n - 1, ring=True)] handled by the caller."""
    n = len(vis)
    out, i = [], 0
    while i < n:
        if vis[i]:
            j = i
            while j + 1 < n and vis[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _dilate_ring(bad, e):
    if e <= 0 or not bad.any():
        return bad
    q = np.concatenate([bad[-e:], bad, bad[:e]]).astype(np.int32)
    return np.convolve(q, np.ones(2 * e + 1, np.int32), mode="valid") > 0


def _straight_flags(P, k, tol, step):
    """per point: True when the window of +-k points around it is near-straight (every point within `tol` px of the chord of the window)."""
    M = len(P)
    A, Bp = np.roll(P, k, 0), np.roll(P, -k, 0)
    ch = Bp - A
    L = np.maximum(np.linalg.norm(ch, axis=1), 1e-6)
    dev = np.zeros(M, np.float32)
    for j in range(-k + 1, k):
        d = np.roll(P, -j, 0) - A
        dev = np.maximum(dev, np.abs(ch[:, 0] * d[:, 1] - ch[:, 1] * d[:, 0]) / L)
    return (dev < tol) & (L > 0.8 * 2 * k * step)           # (and the window is not a closed-up knot: its ends are far apart)


def reject_mask(P, size, params=None, quads=None):
    """points of a resampled closed contour P (n, 2 px, ~1.8 px apart) that must NOT be drawn (user 2026-10-30: "I am more interested in the contours around the things in the image never
    around the frame"): (1) within `border` (+ a margin for the wobble) of the frame border, which is where a matte cut by the frame runs; (2) on a near-straight run longer than `straight` of the
    frame side (the edge of a 3D-cut card, a frame line, a horizon rule), the run dilated by 3 % of the side so no stub is left at its ends; (3) within 3 px of an explicit card quad edge in `quads`
    (list of (4, 2) px corner arrays, when the cut engine's projection is known). -> bool (n,)."""
    pr = dict(PARAMS)
    pr.update(params or {})
    W, H = float(size[0]), float(size[1])
    M = len(P)
    bad = np.zeros(M, bool)
    if not pr.get("guard", True) or M < 8:
        return bad
    side = min(W, H)
    step = max(float(np.linalg.norm(np.diff(np.concatenate([P, P[:1]]), axis=0), axis=1).mean()), 0.5)
    b = float(pr.get("border", 0.0))
    if b > 0:
        mx, my = (b + 0.012) * W, (b + 0.012) * H
        bad |= (P[:, 0] < mx) | (P[:, 0] > W - 1 - mx) | (P[:, 1] < my) | (P[:, 1] > H - 1 - my)
    sf = float(pr.get("straight", 0.0))
    if sf > 0:
        k = int(max(3, round(0.5 * sf * side / step)))
        if M > 2 * k + 3:
            fl = _straight_flags(P, k, float(pr["straight_dev"]) * side, step)
            if fl.any():
                bad |= _dilate_ring(fl, k)
    for q in quads or []:
        q = np.asarray(q, np.float32)
        for a, c in zip(q, np.roll(q, -1, 0)):
            d = c - a
            ln2 = max(float(d @ d), 1e-6)
            tt = np.clip(((P - a) @ d) / ln2, 0.0, 1.0)
            bad |= np.linalg.norm(P - (a + tt[:, None] * d), axis=1) < 3.0
    bad = _dilate_ring(bad, int(round(0.03 * side / step)))
    if sf > 0 and bad.any() and not bad.all():                    # a short stub left between two rejected runs is the frame / card corner, not a form
        k0 = int(np.argmax(bad))
        idx = (np.arange(M) + k0) % M
        keep = ~bad[idx]
        i = 0
        while i < M:
            if keep[i]:
                j = i
                while j + 1 < M and keep[j + 1]:
                    j += 1
                if (j - i + 1) * step < sf * side:
                    bad[idx[i:j + 1]] = True
                i = j + 1
            else:
                i += 1
    return bad


def stroke_alpha(contours, size, t, states, plan_=None, params=None, seed=0, idle=False, ev_index=0, quads=None):
    """the line alpha (float32 H x W, 0..1) of `contours` at time t under the active phase `states` (frame_state output; ignored when idle). size = (W, H)."""
    pr = dict(PARAMS)
    pr.update(params or {})
    W, H = int(size[0]), int(size[1])
    canvas = np.zeros((H, W), np.uint8)
    if not contours:
        return canvas.astype(np.float32)
    base = max(pr["min_px"], float(pr["width"]) * W)
    lo, hi = pr["width_range"]
    nst = int(max(1, min(3, pr["n_strokes"])))
    prep = []                                                                           # (resampled contour, rejected points) per contour
    for P0 in contours:
        P, _L = _resample_closed(_smooth_closed(P0, 2), 1.8)
        if P is not None:
            prep.append((P, reject_mask(P, (W, H), pr, quads)))
    prep = [(P, bd) for P, bd in prep if (~bd).any()][:int(pr["max_contours"])]
    keep_pts = [P[~bd] for P, bd in prep]
    if not keep_pts:
        return canvas.astype(np.float32)
    allp = np.concatenate(keep_pts, 0)                                                   # the subject's extent = the points that may be drawn (not the frame border)
    ext = np.array([allp[:, 0].max() - allp[:, 0].min(), allp[:, 1].max() - allp[:, 1].min()])
    subj = float(max(ext.mean(), 8.0))
    st = states[0] if states else None
    axis = front_axis(st["f"], st["radial"]) if st else ("dir", (1.0, 0.0))
    for ci, (P, bad_pts) in enumerate(prep):
        if bad_pts.all():
            continue
        M = len(P)
        s = _projection(P, axis, (W, H))
        s0 = _projection(allp, axis, (W, H))
        smin, smax = float(s0.min()), float(s0.max())
        sn = (s - smin) / max(smax - smin, 1e-3)                                      # 0 = entry side of the subject, 1 = exit side
        i0 = int(np.argmin(sn))
        P, sn, bad_pts = np.roll(P, -i0, 0), np.roll(sn, -i0), np.roll(bad_pts, -i0)
        u = np.arange(M, dtype=np.float32) / M
        u2 = np.minimum(u, 1.0 - u) * 2.0
        tan = np.gradient(np.concatenate([P[-2:], P, P[:2]], 0), axis=0)[2:-2]
        tn = np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
        nrm = np.stack([tan[:, 1], -tan[:, 0]], 1) / tn
        for j in range(nst):
            sd = _seed(seed, "stroke", ev_index, ci, j)
            nw, no, ng = Noise(sd + 1, 4, 5, 0.30 * pr["speed"], 0.62), Noise(sd + 2, 3, 3, 0.22 * pr["speed"], 0.7), Noise(sd + 3, 3, 7, 0.45 * pr["speed"], 0.6)
            d_j = j * float(pr["stagger"])
            alpha_j, amp_k, ring = 1.0, 1.0, False
            if idle:
                vis, ring, alpha_j = _idle_vis(u, t, sd, j, pr), False, 1.0
            else:
                vis = np.ones(M, bool)
                for ph in states:
                    v, a_j, amp = _phase_vis(ph, sn, u2, d_j, j, pr)
                    vis &= v
                    alpha_j *= a_j
                    amp_k = min(amp_k, amp)
                if alpha_j <= 0.003:
                    continue
            gthr = 1.05 + 0.25 * (j == 0)
            vis &= ~bad_pts                                                              # never along the frame border / a card edge / a straight frame line
            if pr["gaps"] > 0:
                vis &= ~(ng(u, t) > gthr / max(pr["gaps"], 0.2))
            if not vis.any():
                continue
            ring = bool(vis.all())
            amp = float(pr["wobble"]) * subj * amp_k
            bias = (j - 0.5 * (nst - 1)) * 0.55
            off = amp * (bias + no(u, t))
            C = P + nrm * off[:, None]
            wm = lo + (hi - lo) * (0.5 + 0.5 * nw(u, t))
            if not idle and st and st["role"] == "resolve":
                wm = wm + (1.0 - wm) * _ease_io(st["rho"])
            runs = [np.concatenate([np.arange(M), np.arange(3)]) % M] if ring else _runs_circ(vis)
            tmp = canvas if alpha_j >= 0.995 else np.zeros_like(canvas)
            for idx in runs:
                n = len(idx)
                if n < 4:
                    continue
                x = np.linspace(0.0, 1.0, n)
                tp = np.ones(n, np.float32) if ring else _taper(x, pr["taper"], n * 1.8)
                w = np.maximum(base * wm[idx] * tp, 0.35)
                Ci, Ni = C[idx], nrm[idx]
                poly = np.concatenate([Ci + Ni * (w[:, None] * 0.5), (Ci - Ni * (w[:, None] * 0.5))[::-1]], 0)
                cv2.fillPoly(tmp, [np.round(poly * 16.0).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_AA, 4)
            if tmp is not canvas:
                np.maximum(canvas, (tmp.astype(np.float32) * alpha_j).astype(np.uint8), out=canvas)
    b = float(pr.get("border", 0.0)) if pr.get("guard", True) else 0.0
    if b > 0:                                                                            # hard guarantee: not one stroke pixel inside the border band
        by, bx = int(round(b * H)), int(round(b * W))
        canvas[:by, :] = 0
        canvas[H - by:, :] = 0
        canvas[:, :bx] = 0
        canvas[:, W - bx:] = 0
    return canvas.astype(np.float32) / 255.0


def _taper(x, share, length_px):
    """pressure-like width factor along a run: ramps 0.12 -> 1 over `share` of the run at both ends (a run shorter than ~12 px is a dot)."""
    s = max(share, 0.04)
    a = np.clip(x / s, 0.0, 1.0)
    b = np.clip((1.0 - x) / s, 0.0, 1.0)
    return (0.12 + 0.88 * (a * a * (3 - 2 * a)) * (b * b * (3 - 2 * b))).astype(np.float32)


def _phase_vis(ph, sn, u2, d_j, j, pr):
    """(visible bool per point, alpha factor of the stroke, wobble factor) of one stroke j under one phase."""
    role = ph["role"]
    ov = float(pr["overshoot"])
    n = len(sn)
    if role == "off":
        x = min(max((ph["tau"] - d_j) / max(1.0 - d_j, 1e-3), 0.0), 1.0)
        F = _ease_io(x) * 1.08                        # the front, entry side -> exit side
        G = _ease_in(x) * 1.10                        # the arc tail: slower at first than the front, then catches up
        vis = (sn > F - ov * x) & (u2 > G - 0.02)
        return vis, _ease_io(ph["tau"] / 0.12), 1.0
    if role == "on":
        x = min(max((ph["tau"] - d_j) / max(1.0 - d_j, 1e-3), 0.0), 1.0)
        F = _ease_io(x) * 1.08
        H = _ease_out(x) * 1.12                       # the lead end along the arc runs ahead of the front
        vis = (sn <= F + ov * (1.0 - x)) & (u2 <= H)
        return vis, 1.0, 1.0
    rho = ph["rho"]                                   # resolve: full contour, extra strokes first, wobble to 0, then everything fades
    a = 1.0 if j == 0 else 1.0 - _ease_io(rho / 0.55)
    a *= 1.0 - _ease_io((rho - 0.35) / 0.65)
    return np.ones(n, bool), a, 1.0 - _ease_io(rho)


def _idle_vis(u, t, sd, j, pr):
    """idle (no transition) visibility: a run that flows along the contour, its lead end and tail on different easings so it grows / shrinks; two runs per stroke."""
    r = np.random.default_rng(sd + 11)
    sp = float(pr["speed"])
    vis = np.zeros(len(u), bool)
    for k in range(2):
        v = (0.07 + 0.05 * r.random()) * sp * (1 if (j + k) % 2 == 0 else -1)
        a0 = r.random()
        wob = 0.12 * r.random() + 0.06
        w1, w2 = 0.31 + 0.2 * r.random(), 0.23 + 0.2 * r.random()
        head = a0 + v * t + wob * math.sin(2 * math.pi * (w1 * t + r.random()))
        tail = a0 + v * t - 0.34 - wob * math.sin(2 * math.pi * (w2 * t + r.random()))
        ln = (head - tail) % 1.0 if (head - tail) < 1.0 else 1.0
        d = (u - tail) % 1.0
        vis |= d <= max(min(head - tail, 0.97), 0.18)
    return vis


# ================================================================================================================ the compose op
def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def pick_color(color, bg_lum):
    """the palette colour, swapped for the light / dark extreme of it when it would sit within 0.28 luminance of the picture under the line."""
    c = [float(v) for v in (color or (0.95, 0.92, 0.86))]
    if abs(_lum(c) - bg_lum) >= 0.28:
        return c
    return [0.97, 0.96, 0.93] if bg_lum < 0.5 else [0.06, 0.06, 0.07]


def _tonp(B, x):
    return None if x is None else np.asarray(B.tonumpy(x) if hasattr(B, "tonumpy") else x, np.float32)


@C.api
def contour_sweep(img, matte=None, avoid=None, depth=None, t=0.0, frame_idx=None, plan=None, sid=None, color=None, state_key="cs", quads=None):
    """compose op: draw the contour sweep of the active transition (if any) over `img` (HxWx3 float, the composed picture). matte: the shot's subject matte for this frame (HxW; the
    chain's own, so it moves with the clip); avoid: the type alpha (lines are cut out where letters are); plan: contour_sweep.plan(); sid: the shot id; t / frame_idx: piece time.
    Outside every phase and inside the clean windows the image comes back untouched (same object)."""
    B = C.B
    if not plan:
        return img
    fps = float(plan.get("fps", 24))
    tt = (float(frame_idx) / fps) if frame_idx is not None else float(t)
    states = frame_state(plan, tt, sid)
    if not states:
        return img
    g = clean_gate(plan, tt)
    if g <= 0.0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    srcs = tuple(plan["params"].get("sources") or ("matte",))
    m_np = _tonp(B, C._gray(C._in(matte))) if matte is not None else None
    d_np = _tonp(B, C._gray(C._in(depth))) if depth is not None else None
    lum_np = None
    if m_np is None or "luma" in srcs:
        im = _tonp(B, img)
        lum_np = im[..., 0] * 0.2126 + im[..., 1] * 0.7152 + im[..., 2] * 0.0722
        if m_np is None:
            srcs = ("luma",)
    cs = extract_contours(m_np, d_np, lum_np, srcs, int(plan["params"].get("max_contours", 4)), size=(w, h))
    a_np = stroke_alpha(cs, (w, h), tt, states, plan, plan["params"], plan.get("seed", 0), ev_index=states[0]["ev"], quads=quads)
    if not a_np.any():
        return img
    a = C._clip(B.asarray(a_np) * g)
    if avoid is not None:
        av = C._mask(C._in(avoid), h, w)
        a = a * (1.0 - C._clip(av * 1.6))
    sample = _tonp(B, img[::8, ::8])
    cen = float((sample[..., 0] * 0.2126 + sample[..., 1] * 0.7152 + sample[..., 2] * 0.0722).mean())
    col = pick_color(color if color is not None else plan.get("color"), cen)
    colv = B.asarray(np.broadcast_to(np.asarray(col, np.float32), (h, w, 3)).copy())
    return (img * (1.0 - a[..., None]) + colv * a[..., None]).clip(0.0, 1.0)


OPS = {"contour_sweep": contour_sweep}


# ================================================================================================================ recipe hook (no planner edit)
DRAWERS = ("sandwich", "depth_occlude", "knockout", "multiplane", "hold_net")


def enabled(piece) -> bool:
    comp = piece.get("compose") or {}
    if comp.get("contour_sweep") is False:
        return False
    if comp.get("contour_sweep"):
        return True
    pl = piece.get("plan") or {}
    return "contour_sweep" in (((pl.get("effects") or {}).get("requested")) or []) or bool(pl.get("contour_sweep"))


def recipe_ops(piece, ops, sid, shots=None, music=None):
    """the compose chain `ops` of shot `sid` with the `contour_sweep` op after the last type drawer (before the closing grade / vignette). No-op unless `enabled(piece)`, the plan has an
    event for the shot, or the chain already has it. The matte is the drawer's own matte buffer, else `$<sid>_matte` when the shot has a matte, else none (luma contours). Idempotent."""
    if not enabled(piece) or any(o.get("op") == "contour_sweep" for o in ops):
        return ops
    pl = piece.setdefault("plan", {})
    p = pl.get("contour_sweep_plan")
    if not isinstance(p, dict) or p.get("version") != VERSION:
        p = plan(piece, music=music)
        pl["contour_sweep_plan"] = p
    if not any((e["shot_a"] in (None, sid) or e["shot_b"] in (None, sid)) for e in p["events"]):
        return ops
    ops = copy.deepcopy(ops)
    last = None
    for i, o in enumerate(ops):
        if o.get("op") in DRAWERS:
            last = i
    if last is None:
        return ops
    d = ops[last]
    da = d.get("args") or {}
    out = d.get("out", "image")
    matte = None
    for k in ("matte", "subject_matte"):
        v = da.get(k)
        if isinstance(v, str) and v.startswith("$"):
            matte = v
            break
    if matte is None:
        shots = shots if shots is not None else {s.get("id"): s for s in piece.get("shots", [])}
        if (shots.get(sid) or {}).get("matte"):
            matte = f"${sid}_matte"
    avoid = da.get("type_a") if isinstance(da.get("type_a"), str) and da["type_a"].startswith("$") and da["type_a"] not in ("$type_line", "$wl") else None
    args = {"img": f"${out}", "plan": p, "sid": sid, "t": "$t", "frame_idx": "$frame_idx"}
    if matte:
        args["matte"] = matte
    if avoid:
        args["avoid"] = avoid
    ops.insert(last + 1, {"op": "contour_sweep", "out": out, "cs": True, "args": args})
    return ops


DRAW_P = 0.4              # chance the per-piece draw turns the sweep on when nobody asked (the planner hook; a request / compose.contour_sweep = true forces it, = false forbids it)


def drawn(piece, seed=None, p=None) -> bool:
    """the seeded per-piece draw (the random system's `contour_sweep` effect): True with probability DRAW_P, the same for the same piece seed."""
    if seed is None:
        seed = ((piece.get("plan") or {}).get("piece_system") or {}).get("seed", (piece.get("type") or {}).get("seed", 0))
    return random.Random(_seed(int(seed), "cs_draw")).random() < (DRAW_P if p is None else p)


def weave(piece, music=None, chk=None, size=None, holds=None, draw=True):
    """THE PLANNER HOOK (one call, never raises). Requested (`effects` names contour_sweep / compose.contour_sweep true) -> on; compose.contour_sweep false -> off; else the seeded draw (`draw=False`
    = only when requested). On: compose.contour_sweep = True, the plan summary in piece.plan.contour_sweep (the full plan is built where the compose chain is made, `recipe_ops`). In place; returns the piece."""
    comp = piece.setdefault("compose", {})
    pl = piece.setdefault("plan", {})
    try:
        if comp.get("contour_sweep") is False:
            pl.pop("contour_sweep", None)
            return piece
        asked = bool(comp.get("contour_sweep")) or "contour_sweep" in (((pl.get("effects") or {}).get("requested")) or [])
        if not asked and not (draw and drawn(piece)):
            return piece
        if not comp.get("contour_sweep"):
            comp["contour_sweep"] = True
        p = plan(piece, music=music, chk=chk, holds=holds)
        pl["contour_sweep"] = dict(seed=p["seed"], requested=asked, events=[dict(kind=e["kind"], type=e["type"], f=e["f"], radial=e["radial"], t=[e["t_a0"], e["t_c"], e["t_e"]], b=e["b"]) for e in p["events"]])
    except Exception as e:                                  # noqa: BLE001 the hook must never break a piece
        pl["contour_sweep"] = dict(events=[], error=f"{type(e).__name__}: {e}")
        comp["contour_sweep"] = False
    return piece


# ================================================================================================================ idle edge (plate_layers contrast moments)
def edge_alpha(mask, frame, seed=0, width_px=None, params=None, section=None, guard=True) -> np.ndarray:
    """the animated brush outline of `mask` (HxW float, the subject's / the letters' matte) at `frame`: the contour of mask > 0.5 as flowing brush strokes (variable width, tapered
    ends, 2 strokes, wobble, small breaks), no dashes, no glow. section = (f0, f1) frames of the contrast moment: the line draws on from the left over the first 12 frames and off
    over the last 12 (the same front mechanics as a transition). -> float32 HxW."""
    m = np.asarray(mask, np.float32)
    H, W = m.shape[:2]
    cs = extract_contours(m, None, None, ("matte",), 4, size=(W, H))
    pr = dict(PARAMS)
    pr.update(params or {})
    pr["guard"] = bool(guard)
    if width_px:
        pr["width"] = float(width_px) / W
    t = float(frame) / 24.0
    states = None
    idle = True
    if section:
        f0, f1 = section
        ramp = float(min(12.0, max(1.0, 0.25 * (f1 - f0))))
        if frame < f0 + ramp:
            states, idle = [dict(role="on", tau=min(max((frame - f0) / ramp, 0.0), 1.0), rho=0.0, f=(1.0, 0.0), radial=0, ev=0)], False
        elif frame > f1 - ramp:
            states, idle = [dict(role="off", tau=min(max((frame - (f1 - ramp)) / ramp, 0.0), 1.0), rho=0.0, f=(1.0, 0.0), radial=0, ev=0)], False
    return stroke_alpha(cs, (W, H), t, states or [], None, pr, seed, idle=idle, ev_index=0)
