"""mograph.modules.engine: the rule-based transition engine (phase 4).

    gap = analyse(word_a, word_b)                      # what lies between A's exit gate and B's entry gate
    tcls, params, dur, cam = choose(gap, history, seed, energy)
    seq = Sequence.auto([w0, w1, w2, ...], seed=3, energy=0.4, bpm=96)       # engine-chosen transitions (explicit ones stay)

`analyse(a, b)` -> GapInfo (plain numbers, so rules can be tested on synthetic gaps):
    dist (gate to gate, px), direction (unit A->B), heading (deg between A's exit travel vector and B's entry travel vector),
    turn_out / turn_in (deg: A's exit vs the gap, the gap vs B's entry; detour = their mean: how far the strands must bend),
    plane_angle (deg between the placements' normals) -> plane 'same' (< 15) | 'fold' | 'opposite' (> 150), depth_offset,
    ortho (0..1: gap direction and heading sit on the 45 degree lattice of A's frame, the grid a circuit board routes on),
    a_kind / b_kind and a_n / b_n (material kind and native count at each end), a_strands / b_strands (strand count the
    transition sees), a_size / b_size (placement w, h), size_ratio (B's linear size / A's), obstacle (px of clearance between the
    cable's axis and any other word's box; 0 = it passes through one; None = unknown), avail (seconds, a budget), beat (s),
    energy 0..1, mood {name: weight}, speed (hand-off speed), dur_auto (the duration core.auto_dur would give).

Vocabulary: every transition class declares `SUITS`, what it is good at, as {feature: (shape, weight)}. A shape is a trapezoid
(a, b, c, d) (0 below a, 1 on b..c, 0 above d), or {category: fit} for `plane`. Keys: dist, heading, detour, plane, plane_angle,
ortho, size_ratio, energy, obstacle, count, mood, base (constant fit), `a|b` (the better of two fits: each a shape). Extras:
`must` (features whose fit multiplies the score: a hard-ish requirement), `moods` (name -> 0..1), `dur` (min, max, factor),
`brief` (plain words). The score of a transition = weighted mean of its fits x must-fits. `rank(gap)` lists them.

choose(): rank -> multiply by a variety factor (repeat of the previous choice x0.35, two back x0.75, each use before x0.93) ->
seeded weighted pick (score^3) among the candidates within 0.65 of the best (max 3) -> params from the gap -> duration (the
duration that keeps the strand speed continuous, scaled by energy and the transition's factor, clamped to its range and to `avail`,
snapped to the beat grid) -> camera hint (a plain dict). A side with material `none` forces `cut`.

Rules table (what each is good at; the numbers are in each transition's SUITS):
    twist          medium distance, any heading, any plane (the generalist; base fit 0.55 at weight 2.5, so it tops out near 0.76)
    circuit_board  same or folded plane, moves on the 45 degree lattice (ortho), techy / tense, mid-high energy
    spiral_around  big heading change or detour, or the route passes a word / object (obstacle), folded planes
    braid          long runs (dist > 500) that stay straight and in one plane, calm, low energy
    fan            B wider / larger than A (size_ratio > 1.2), modest heading change
    cut            a side carries `none`; also a tiny gap at high energy
Params: twist turns = dist x 1.25 / (380 - 100 energy) (1..4, at most dur / 0.55), handed alternates; circuit lanes = strands / 9 (2..5);
spiral around = word when the route passes a word, sphere for a U-turn, else cylinder; turns = 0.9 + heading / 110 (half turns);
radius = 0.45 sqrt(w_B h_B); braid cycles = dur / 1.05 (half cycles, 1..3); fan angle = 46 + 34 log2(size_ratio) (46..88)."""
import math
from dataclasses import dataclass, field, fields
import numpy as np

from mograph.modules.core import (WordModule, TransitionModule, get_module, list_modules, unit)

# ------------------------------------------------------------------ helpers
FAR = 1.0e4                       # "no obstacle" clearance
SAME_DEG, OPPOSITE_DEG = 15.0, 150.0


def trap(x, a, b, c, d):
    """trapezoid fit 0..1: 0 below a, rising to 1 at b, 1 until c, falling to 0 at d (a == b or c == d make a step)."""
    if x < a or x > d:
        return 0.0
    if x < b:
        return (x - a) / max(b - a, 1e-9)
    if x <= c:
        return 1.0
    return (d - x) / max(d - c, 1e-9)


def ang(a, b):
    """angle in degrees between two vectors."""
    a, b = unit(np.asarray(a, float)), unit(np.asarray(b, float))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b))))))


def _lattice(frame):
    """9 unsigned directions of a frame (u, v, n): 3 axes + 6 planar diagonals."""
    u, v, n = frame
    out = [u, v, n]
    for p, q in ((u, v), (u, n), (v, n)):
        out += [unit(p + q), unit(p - q)]
    return out


def ortho_score(direction, heading, frame):
    """0..1: how well the gap direction sits on the 45 degree lattice of `frame` and the heading on a multiple of 45."""
    m = min(ang(direction, l) if ang(direction, l) < 90 else 180 - ang(direction, l) for l in _lattice(frame))
    d = max(0.0, 1.0 - m / 22.5)
    h = max(0.0, 1.0 - min(abs(heading - k) for k in (0, 45, 90, 135, 180)) / 22.5)
    return 0.6 * d + 0.4 * h


def plane_class(angle):
    return "same" if angle < SAME_DEG else ("opposite" if angle > OPPOSITE_DEG else "fold")


MOODS = ("calm", "mysterious", "melancholy", "nostalgic", "playful", "joyful", "tense", "hype")


def coerce_mood(m):
    if m is None:
        return None
    if isinstance(m, str):
        return {m: 1.0}
    return {str(k): float(v) for k, v in dict(m).items()}


# ------------------------------------------------------------------ gap
@dataclass
class GapInfo:
    """everything the rules know about one gap. All fields have neutral defaults, so a synthetic gap is `GapInfo(dist=900, ...)`."""
    index: int = 0
    a_name: str = ""
    b_name: str = ""
    dist: float = 400.0
    heading: float = 0.0
    turn_out: float = 0.0
    turn_in: float = 0.0
    plane_angle: float = 0.0
    depth_offset: float = 0.0
    ortho: float = 0.0
    a_kind: str = "strands"
    b_kind: str = "strands"
    a_n: int = 36
    b_n: int = 36
    a_strands: int = 36
    b_strands: int = 36
    a_size: tuple = (380.0, 150.0)
    b_size: tuple = (380.0, 150.0)
    size_ratio: float = 1.0
    obstacle: float = None
    avail: float = None
    beat: float = None
    energy: float = 0.5
    mood: dict = None
    speed: float = 700.0
    dur_auto: float = 2.0
    centre_dist: float = 0.0
    a_gate: tuple = (0.0, 0.0, 0.0)
    b_gate: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (1.0, 0.0, 0.0)
    a_dir: tuple = (1.0, 0.0, 0.0)
    b_dir: tuple = (1.0, 0.0, 0.0)
    warn: list = field(default_factory=list)          # module quirks seen while reading the gates (e.g. D opposing head - tail)

    def __post_init__(self):
        self.mood = coerce_mood(self.mood)

    @property
    def detour(self):
        return 0.5 * (self.turn_out + self.turn_in)

    @property
    def plane(self):
        return plane_class(self.plane_angle)

    @property
    def has_none(self):
        return self.a_kind == "none" or self.b_kind == "none"

    @property
    def strands(self):
        return max(self.a_strands, self.b_strands)

    def summary(self):
        ob = "-" if self.obstacle is None or self.obstacle >= FAR else ("through" if self.obstacle <= 0 else f"{self.obstacle:.0f}px")
        return (f"{self.a_name}>{self.b_name}: dist {self.dist:.0f}px, heading {self.heading:.0f}deg (detour {self.detour:.0f}), "
                f"plane {self.plane} {self.plane_angle:.0f}deg, ortho {self.ortho:.2f}, size x{self.size_ratio:.2f}, "
                f"{self.a_kind}{self.a_n}>{self.b_kind}{self.b_n}, obstacle {ob}")

    def to_dict(self):
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = [round(float(x), 2) for x in v] if isinstance(v, (tuple, np.ndarray)) else v
        return out


def _box_clearance(P, pl, thick=80.0):
    """distance from points P (n,3) to the thick box of a Placement (0 inside)."""
    r = P - pl.origin
    x, y, z = r @ pl.u, r @ pl.v, r @ pl.n
    d = np.stack([np.maximum(np.abs(x) - pl.w / 2, 0), np.maximum(np.abs(y) - pl.h / 2, 0), np.maximum(np.abs(z) - thick / 2, 0)], 1)
    return np.linalg.norm(d, axis=1)


def _clearance(a_gate, b_gate, da, db, dist, others):
    """px between the cable axis (the same Hermite curve twist uses) and the nearest OTHER word box (not A's or B's own: a route
    that doubles back over its own words shows up as `detour`)."""
    from mograph.modules.transitions._path import hermite_curve
    m = 0.9 * dist + 80.0
    u = np.linspace(0, 1, 48)
    C = hermite_curve(a_gate, b_gate, da * m, db * m, u)
    best = FAR
    for pl in others:
        best = min(best, float(_box_clearance(C, pl).min()))
    return best


def analyse(a, b, others=(), index=0, energy=0.5, mood=None, avail=None, bpm=None, beat=None):
    """GapInfo for the gap between word module `a` (exit) and `b` (entry). others: the other word modules of the chain (for the
    obstacle test). avail: seconds the gap may take; beat (s) or bpm: the beat grid."""
    ma, mb = a.exit_material(), b.entry_material()
    ea, eb = a.exit_bundle() if ma.kind != "none" else None, b.entry_bundle() if mb.kind != "none" else None
    ga, gb = np.asarray(ma.anchor, float), np.asarray(mb.anchor, float)
    da, db = unit(ma.dir), unit(mb.dir)
    gap = gb - ga
    dist = float(np.linalg.norm(gap))
    direction = gap / dist if dist > 1e-6 else da
    pa, pb = a.placement, b.placement
    pang = ang(pa.n, pb.n)
    heading = ang(da, db)
    size = lambda p: math.sqrt(p.w * p.h)
    n_of = lambda m: 0 if m.kind == "none" else (m.bundle.N if m.kind == "strands" else len(m.P))
    sa, sb = (ea.N if ea is not None else 0), (eb.N if eb is not None else 0)
    if ea is not None and eb is not None:
        dur_auto = float(TransitionModule.auto_dur(ea, eb))
        speed = 0.5 * (float(ea.v.mean()) + float(eb.v.mean()))
    else:
        dur_auto, speed = 0.4, 0.0
    obstacle = None
    if ea is not None and eb is not None:
        pls = [w.placement for w in others if w is not a and w is not b]
        obstacle = _clearance(ga, gb, da, db, dist, pls)
    if beat is None and bpm:
        beat = 60.0 / float(bpm)
    warn = []
    for tag, bun, d in (("exit of " + (a.name or "A"), ea, da), ("entry of " + (b.name or "B"), eb, db)):
        if bun is not None:
            v = (bun.heads() - bun.tails()).mean(0)
            if np.linalg.norm(v) > 1e-6 and float(unit(v) @ d) < -0.3:
                warn.append(f"{tag}: strand direction D opposes head - tail (module bug?)")
    return GapInfo(index=index, a_name=a.name or type(a).__name__, b_name=b.name or type(b).__name__, dist=dist, heading=heading,
                   turn_out=ang(da, direction), turn_in=ang(direction, db), plane_angle=pang,
                   depth_offset=float(abs((pb.origin - pa.origin) @ pa.n)),
                   ortho=ortho_score(direction, heading, (pa.u, pa.v, pa.n)), a_kind=ma.kind, b_kind=mb.kind,
                   a_n=n_of(ma), b_n=n_of(mb), a_strands=sa, b_strands=sb, a_size=(pa.w, pa.h), b_size=(pb.w, pb.h),
                   size_ratio=size(pb) / max(size(pa), 1e-6), obstacle=obstacle, avail=avail, beat=beat,
                   energy=float(energy), mood=coerce_mood(mood), speed=speed, dur_auto=dur_auto,
                   centre_dist=float(np.linalg.norm(pb.origin - pa.origin)), a_gate=tuple(ga), b_gate=tuple(gb),
                   direction=tuple(direction), a_dir=tuple(da), b_dir=tuple(db), warn=warn)


# ------------------------------------------------------------------ vocabulary / scoring
DEFAULT_SUITS = dict(brief="no declared vocabulary", base=(0.3, 1.0), dur=(1.2, 3.4, 1.0))


def suits_of(name):
    return getattr(get_module(name), "SUITS", None) or DEFAULT_SUITS


def vocabulary():
    """{name: SUITS} for every registered transition."""
    return {n: suits_of(n) for n in list_modules("transition")}


def _value(gap, feat):
    if feat == "obstacle":
        return FAR if gap.obstacle is None else gap.obstacle
    if feat == "count":
        return gap.strands
    return getattr(gap, feat)


def _fit(gap, feat, shape):
    if feat == "plane":
        return float(shape.get(gap.plane, 0.0))
    if feat == "mood":
        return 0.0
    return trap(_value(gap, feat), *shape)


def feature_fits(gap, suits, energy=None):
    """[(feature, fit, weight)] of one transition's SUITS on a gap (features it does not declare are absent; `mood` only when
    the gap carries one)."""
    out = []
    if energy is not None and energy != gap.energy:
        gap = _with(gap, energy=energy)
    for feat, spec in suits.items():
        if feat in ("brief", "tags", "moods", "dur", "must"):
            continue
        shape, w = spec
        if feat == "base":
            out.append((feat, float(shape), float(w)))
        elif feat == "mood":
            continue
        elif "|" in feat:
            fs = [_fit(gap, f, s) for f, s in zip(feat.split("|"), shape)]
            out.append((feat, max(fs), float(w)))
        else:
            out.append((feat, _fit(gap, feat, shape), float(w)))
    if gap.mood and suits.get("moods"):
        mw = suits["moods"]
        tot = sum(gap.mood.values()) or 1.0
        out.append(("mood", sum(v * mw.get(k, 0.0) for k, v in gap.mood.items()) / tot, float(suits.get("mood_weight", 2.0))))
    return out


def _with(gap, **kw):
    d = {f.name: getattr(gap, f.name) for f in fields(gap)}
    d.update(kw)
    return GapInfo(**d)


def score(gap, name, energy=None):
    """(score 0..1, parts) of transition `name` on `gap`. parts = [(feature, fit, weight)]."""
    suits = suits_of(name)
    parts = feature_fits(gap, suits, energy)
    wsum = sum(w for _, _, w in parts) or 1.0
    s = sum(f * w for _, f, w in parts) / wsum
    for feat in suits.get("must", ()):
        f = next((f for k, f, _ in parts if k == feat), 1.0)
        s *= 0.2 + 0.8 * f
    return float(s), parts


def rank(gap, energy=None, names=None):
    """[(name, score, parts)] best first, rules only (no variety). A side with material `none` can only be cut."""
    names = names or list_modules("transition")
    if gap.has_none:
        names = ["cut"]
    out = [(n, *score(gap, n, energy)) for n in names]
    out.sort(key=lambda r: -r[1])
    return out


# ------------------------------------------------------------------ parameters / duration / camera
def _hist_name(h):
    return h if isinstance(h, str) else getattr(h, "name", None)


def _hist_params(h):
    return {} if isinstance(h, str) else (getattr(h, "params", None) or {})


def next_handed(history, rng):
    """alternate the screw direction: the opposite of the last handed transition in the history (seeded start)."""
    for h in reversed(list(history)):
        p = _hist_params(h)
        if "handed" in p:
            return -1 if p["handed"] >= 0 else 1
    return 1 if rng.random() < 0.5 else -1


def _half(x, lo, hi):
    return float(min(hi, max(lo, round(x * 2) / 2.0)))


def p_twist(g, dur, hist, rng, e):
    t = g.dist * 1.25 / (380.0 - 100.0 * e) + rng.uniform(-0.45, 0.45)           # the seed nudges the count a little
    return dict(turns=int(min(4, max(1, round(t)), max(1, math.floor(dur / 0.55)))), handed=next_handed(hist, rng))


def p_circuit(g, dur, hist, rng, e):
    return dict(lanes=int(min(5, max(2, round(max(g.strands, 1) / 9.0)))))


def p_spiral(g, dur, hist, rng, e):
    if g.obstacle is not None and g.obstacle < 60.0:
        around = "word"
    elif g.heading > 120.0 or (g.heading > 60.0 and rng.random() < 0.4):
        around = "sphere"
    else:
        around = "cylinder"
    r = 0.45 * math.sqrt(g.b_size[0] * g.b_size[1])
    turns = _half(0.9 + g.heading / 110.0 + rng.uniform(-0.3, 0.3), 1.0, 2.5)
    turns = min(turns, max(1.0, math.floor(dur / 0.85 * 2) / 2.0))                       # never faster than ~0.85 s per turn
    return dict(around=around, turns=turns, handed=next_handed(hist, rng),
                radius=float(round(min(190.0, max(70.0, r * rng.uniform(0.9, 1.1))))))


def p_braid(g, dur, hist, rng, e):
    return dict(cycles=_half(dur / 1.05, 1.0, 3.0), handed=next_handed(hist, rng))


def p_fan(g, dur, hist, rng, e):
    a = 46.0 + 34.0 * math.log2(max(g.size_ratio, 1.0)) + rng.uniform(-5.0, 5.0)
    return dict(angle=float(round(min(88.0, max(46.0, a)))))


def p_cut(g, dur, hist, rng, e):
    return dict(lines=3)


PARAMS = dict(twist=p_twist, circuit_board=p_circuit, spiral_around=p_spiral, braid=p_braid, fan=p_fan, cut=p_cut)


def snap_beat(dur, beat, lo=0.0):
    """round dur to the beat grid (a half beat when the beat is slower than 0.8 s), at least lo."""
    if not beat:
        return round(dur * 10.0) / 10.0
    grid = beat if beat <= 0.8 else beat / 2.0
    n = max(1, round(dur / grid))
    while n * grid < lo - 1e-9:
        n += 1
    return n * grid


def pick_duration(name, g, e):
    """seconds: the duration that keeps strand speed continuous (core.auto_dur), scaled 1.15 -> 0.85 by energy and by the
    transition's factor, kept in its range, inside `avail`, snapped to the beat grid."""
    if name == "cut":
        d = float(min(max(0.22 + g.dist / 2600.0, 0.24), 0.7))
        if g.beat:
            half = g.beat / 2.0 if g.beat > 0.4 else g.beat
            d = max(half * max(1, round(d / half)), 0.24)
        return float(d)
    lo, hi, fac = suits_of(name).get("dur", (1.0, 3.6, 1.0))
    d = g.dur_auto * fac * (1.15 - 0.3 * e)
    d = min(max(d, lo), hi)
    if g.avail is not None:
        d = min(d, max(g.avail, 0.8))
    d = snap_beat(d, g.beat, lo=min(lo, g.avail) if g.avail else lo)
    return float(d)


def _hint(follow, move="follow", **kw):
    return dict(move=move, follow=follow, **kw)


def hint_for(name, g, params, dur):
    """camera hint as a plain dict (3b camera modules are named in `move`; 'follow' is the default and always safe):
        move    suggested camera module for this transition window ('follow', 'orbit_even', 'graze', 'pullback_reveal', ...)
        follow  what to look at: 'heads' (strand heads), 'axis' (midpoint of the route), 'object' (the thing being wrapped)
        hold    where to settle after: 'b' (face B's formed word) | None
        lead    seconds the eye may run ahead of the heads
        zoom    'out' when B is wider (give it room) | 'in' | None
        window  ('trans', gap index)"""
    base = dict(window=("trans", g.index), hold="b", lead=0.15 if g.energy < 0.6 else 0.05, zoom=None)
    if name == "twist":
        h = _hint("heads", note="ride behind the cable's head")
    elif name == "circuit_board":
        h = _hint("heads", move="graze" if g.dist > 450 else "follow", note="stay near the plane of the bus")
    elif name == "spiral_around":
        h = _hint("object", move="orbit_even" if params.get("turns", 1) >= 1.5 else "follow", note="keep the wrapped object framed",
                  lead=0.0)
    elif name == "braid":
        h = _hint("axis", note="long calm run: sit back, let the plaits weave", lead=0.25)
    elif name == "fan":
        h = _hint("axis", move="pullback_reveal" if g.size_ratio >= 1.8 else "follow", zoom="out",
                  note="the blade opens toward a larger word")
    else:
        h = _hint("axis", note="a cut: no travel, hold the frame", hold="b", lead=0.0)
    base.update(h)
    return base


# ------------------------------------------------------------------ choose
class Choice(tuple):
    """(TransitionClass, params, dur, camera_hint) with extra attributes: name, score, ranking [(name, adjusted score)], parts,
    forced (bool), why (str), gap."""

    def __new__(cls, tcls, params, dur, hint, **info):
        self = super().__new__(cls, (tcls, params, dur, hint))
        self.__dict__.update(info)
        return self

    def __reduce__(self):                                   # a Sequence is pickled for the render pool
        return (_rebuild_choice, (tuple(self), dict(self.__dict__)))

    @property
    def tcls(self):
        return self[0]

    @property
    def params(self):
        return self[1]

    @property
    def dur(self):
        return self[2]

    @property
    def camera_hint(self):
        return self[3]

    def short(self):
        p = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.name} {p} dur {self.dur:.2f}s"


def _rebuild_choice(t, info):
    return Choice(*t, **info)


VARIETY = (0.35, 0.75)           # factor for a repeat of the previous / of two back
USED_PENALTY = 0.93               # per earlier use
TOP_K, TOP_FRAC, PICK_POW = 3, 0.65, 3.0


def variety_factor(name, history):
    names = [_hist_name(h) for h in history]
    f = 1.0
    if names and names[-1] == name:
        f *= VARIETY[0]
    elif len(names) > 1 and names[-2] == name:
        f *= VARIETY[1]
    return f * USED_PENALTY ** min(names.count(name), 4)


def choose(gap, history=(), seed=0, energy=None, names=None):
    """-> Choice (unpacks as (TransitionClass, params, dur, camera_hint)). Deterministic per (seed, gap.index, len(history)).
    A name already in `history` is excluded while an unused one remains in `names`; once all are used only the soft penalty applies."""
    e = gap.energy if energy is None else float(energy)
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, int(gap.index), len(history)])
    ranked = rank(gap, e, names)
    forced = gap.has_none
    if forced:
        pick = ranked[0]
        adj = [(pick[0], pick[1])]
    else:
        used = {_hist_name(h) for h in history} - {None}
        fresh = [r for r in ranked if r[0] not in used]       # no repeats inside a chain while an unused transition remains;
        adj = sorted(((n, s * variety_factor(n, history)) for n, s, _ in (fresh or ranked)), key=lambda r: -r[1])   # all used: soft penalty
        top = [r for r in adj[:TOP_K] if r[1] >= TOP_FRAC * adj[0][1]]
        w = np.array([max(r[1], 1e-6) ** PICK_POW for r in top])
        k = int(rng.choice(len(top), p=w / w.sum()))
        pick = next(r for r in ranked if r[0] == top[k][0])
    name = pick[0]
    fn = PARAMS.get(name)
    dur = pick_duration(name, gap, e)
    params = fn(gap, dur, history, rng, e) if fn else {}
    parts = pick[2]
    best = sorted(parts, key=lambda p: -p[1] * p[2])[:3]
    why = ("side carries no material -> cut" if forced else
           "; ".join(f"{k} {f:.2f}" for k, f, _ in best) + f" (score {pick[1]:.2f}, " + suits_of(name).get("brief", "") + ")")
    return Choice(get_module(name), params, dur, hint_for(name, gap, params, dur), name=name, score=pick[1], ranking=adj, parts=parts,
                  forced=forced, why=why, gap=gap)


# ------------------------------------------------------------------ sequence
def _is_trans(x):
    return isinstance(x, TransitionModule) or (isinstance(x, type) and issubclass(x, TransitionModule))


def build_sequence(items, seed=0, energy=0.5, bpm=None, mood=None, duration=None, bg=None, tail=0.0, names=None, **kw):
    """Sequence with an engine-chosen transition in every gap between two word modules that has none. `items`: word modules
    (instances), optionally with explicit transitions between them (kept). energy: float or one per gap. duration: target
    total seconds -> `avail` per gap = (duration - words) / gaps. bpm: beat grid for the transitions (build the words' timing with
    Timing.snap(60 / bpm) to keep the whole chain on it). The Sequence gets .engine = [Choice | None per gap] and .gaps."""
    from mograph.modules.sequence import Sequence
    items = list(items)
    words = [x for x in items if isinstance(x, WordModule)]
    if len(words) < 1 or not isinstance(items[0], WordModule):
        raise ValueError("a chain starts with a word module (and needs word modules, not classes)")
    beat = 60.0 / float(bpm) if bpm else None
    free = sum(1 for i, x in enumerate(items[:-1]) if isinstance(x, WordModule) and isinstance(items[i + 1], WordModule))
    avail = None
    if duration is not None and free:
        explicit = sum(x.dur or 0.0 for x in items if isinstance(x, TransitionModule))
        avail = max((float(duration) - sum(w.duration for w in words) - explicit) / free, 0.8)
    out, history, gaps, choices, gi = [], [], [], [], 0
    for i, x in enumerate(items):
        out.append(x)
        if isinstance(x, WordModule) and i + 1 < len(items):
            nxt = items[i + 1]
            if isinstance(nxt, WordModule):
                e = energy[gi] if isinstance(energy, (list, tuple, np.ndarray)) else energy
                gap = analyse(x, nxt, others=words, index=gi, energy=e, mood=mood, avail=avail, beat=beat)
                ch = choose(gap, history, seed, e, names)
                tr = ch.tcls(dur=ch.dur, **ch.params)
                tr.camera_hint, tr.engine = ch.camera_hint, ch
                out.append(tr)
                history.append(ch)
                gaps.append(gap)
                choices.append(ch)
                gi += 1
            elif _is_trans(nxt):
                history.append(getattr(nxt, "name", None) or type(nxt).__name__)
    seq = Sequence(out, bg=bg, tail=tail, **kw)
    seq.gaps, seq.engine = gaps, choices
    seq.engine_args = dict(seed=seed, energy=energy, bpm=bpm, mood=mood, duration=duration)
    return seq


def describe(seq):
    """plain-text lines: per gap the GapInfo summary and the chosen transition."""
    out = []
    for g, c in zip(getattr(seq, "gaps", []), getattr(seq, "engine", [])):
        out.append(g.summary())
        out.append("    -> " + c.short() + "   [" + c.why + "]")
    return out
