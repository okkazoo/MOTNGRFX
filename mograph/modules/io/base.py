"""Shared machinery of the I/O styles: the style base class + registry, layer tools (half-space clip, fade, consolidate),
the staging ribbon (where a style's carried material waits at the gate) and `Flight` (a mark / blob travelling along a
bezier route with a Hermite speed profile that matches the neighbour's hand-off speed)."""
import copy, math
import numpy as np

from mograph.modules.core import Layer, Route3, herm, clamp_slope, unit, bezier
from mograph.render.cam3d import clip_poly, clip_line
from mograph.modules.words._strand import lane_frame


def ss(x):
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


def eo(x, p=2.4):
    x = min(1.0, max(0.0, float(x)))
    return 1.0 - (1.0 - x) ** p


def seg(p, a, b):
    return min(1.0, max(0.0, (p - a) / max(b - a, 1e-9)))


def hash01(*args):
    """deterministic uniform [0,1) per element: integer-valued arrays / scalars in, array out (splitmix-style mixing)."""
    h = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for a in args:
            x = np.asarray(a).astype(np.int64).astype(np.uint64)
            h = h ^ (x + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2)))
            h = h * np.uint64(0xBF58476D1CE4E5B9)
            h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# ------------------------------------------------------------------ style registry
STYLES = {}


def register_style(name):
    def deco(cls):
        cls.name = name
        STYLES[name] = cls
        return cls
    return deco


def list_styles():
    return sorted(STYLES)


class IOStyle:
    """a form-in / form-out style. One instance serves one side of one module (`attach(body, 'in'|'out')`).
    layers(t, p): t = module time, p = progress in the window (form: 0..1; exit: 0..1 and beyond = still coasting away).
    material(): the Material carried at the window's outer end (entry: at t = 0; exit: at t = duration).
    Contract: layers(0, 0) draws exactly entry material, layers(duration, 1) exactly exit material."""
    name = ""
    description = ""
    is_lines = False
    material_kind = "none"

    def __init__(self, **opts):
        self.opts = dict(opts)
        self.body, self.side = None, None

    def opt(self, k, default):
        return self.opts.get(k, default)

    def attach(self, body, side):
        self.body, self.side = body, side

    def layers(self, t, p):
        raise NotImplementedError

    def material(self):
        raise NotImplementedError

    def focus(self, t, p):
        return self.body.placement.origin.copy()

    def __repr__(self):
        return f"{type(self).__name__}({self.opts})" if self.opts else type(self).__name__


def get_style(spec):
    """None | 'name' | IOStyle (copied) | ('name', {options}) | {'name': ..., **options} -> a fresh IOStyle."""
    if spec is None:
        spec = "lines"
    if isinstance(spec, IOStyle):
        st = copy.copy(spec)
        st.opts = dict(spec.opts)
        st.body = st.side = None
        return st
    if isinstance(spec, str):
        if spec not in STYLES:
            raise KeyError(f"unknown I/O style {spec!r}; styles: {list_styles()}")
        return STYLES[spec]()
    if isinstance(spec, dict):
        d = dict(spec)
        return STYLES[d.pop("name")](**d)
    name, opts = spec
    return STYLES[name](**opts)


# ------------------------------------------------------------------ layer tools
def clip_layer(L, n, c):
    """the part of Layer L with (X - c) . n >= 0 (None when nothing is left; L itself when everything is)."""
    if not len(L.pts):
        return None
    d_all = (L.pts - c) @ n
    if d_all.min() >= 0:
        return L
    if d_all.max() < 0:
        return None
    polys = []
    for P, a, b in zip(L.polys, L.offs[:-1], L.offs[1:]):
        d = d_all[a:b]
        if L.kind == "fill":
            q = clip_poly(P, d)
            if q is not None and len(q) >= 3:
                polys.append(q)
        else:
            polys.extend(clip_line(P, d, L.closed))
    if not polys:
        return None
    return Layer(polys, L.color, L.kind, L.width, alpha=L.alpha, closed=(L.closed if L.kind == "fill" else False),
                 split=L.split, bias=L.bias, name=L.name)


def clip_layers(layers, n, c):
    out = []
    for L in layers:
        q = clip_layer(L, n, c)
        if q is not None:
            out.append(q)
    return out


def slab_layers(layers, n, c0, c1):
    """parts with c0 <= (X . n) < c1 (c0, c1 given as scalars along n)."""
    n = np.asarray(n, float)
    out = []
    for L in layers:
        q = clip_layer(L, n, n * c0)
        if q is None:
            continue
        q = clip_layer(q, -n, n * c1)
        if q is not None:
            out.append(q)
    return out


def shifted(L, vec):
    L2 = copy.copy(L)
    L2.polys = [P + vec for P in L.polys]
    L2.pts = L.pts + vec
    return L2


def fade(layers, a):
    """layers at alpha a (multiplied into each layer's own alpha)."""
    if a >= 0.998:
        return list(layers)
    if a <= 0.002:
        return []
    out = []
    for L in layers:
        L2 = copy.copy(L)
        L2.alpha = L.alpha * a
        out.append(L2)
    return out


def style_key(L):
    return (L.kind, L.color, round(L.width, 3), round(L.alpha, 3), L.closed, L.split, round(L.bias, 3))


def consolidate(items):
    """items: list of (style key, polys) -> Layers, one per key (first-appearance order). Keys are Layer.style_key tuples."""
    groups = {}
    for key, polys in items:
        groups.setdefault(key, []).extend(polys)
    out = []
    for (kind, color, width, alpha, closed, split, bias), polys in groups.items():
        out.append(Layer(polys, color, kind, width, alpha=alpha, closed=closed, split=split, bias=bias))
    return out


# ------------------------------------------------------------------ gate geometry
def side_frame(body, side):
    """(d, e1, e2, gate): travel direction d (entry: toward the word; exit: away from it), lane frame, and the distance from
    the placement centre to the gate where carried material waits (just outside the placement, like the strand modules)."""
    pl = body.placement
    d = -body.entry.side(pl) if side == "in" else body.exit.side(pl)
    e1, e2 = lane_frame(d, pl)
    gate = pl.extent_along(d) + 0.16 * max(pl.w, pl.h) + 12.0
    return d, e1, e2, gate


def stage_ribbon(body, side, lat, spacing, row_pitch=None):
    """waiting positions for len(lat) carried items whose targets have lateral coordinates `lat` (along e1): they line up at
    the gate in lateral order (so the streams never cross), in as many rows (stacked back along the lane) as the placement is
    wide enough for. -> dict(S (M,3), d, e1, e2, order (M,) rank of every item)."""
    d, e1, e2, gate = side_frame(body, side)
    pl = body.placement
    M = len(lat)
    half = pl.extent_along(e1)
    rows = max(1, int(math.ceil(M * spacing / (1.7 * max(half, 1.0)))))
    ncol = int(math.ceil(M / rows))
    order = np.empty(M, int)
    order[np.argsort(lat, kind="stable")] = np.arange(M)
    col, row = order // rows, order % rows
    lam = (col - (ncol - 1) / 2.0) * spacing
    rp = row_pitch if row_pitch is not None else 2.2 * spacing
    sgn = -1.0 if side == "in" else 1.0
    S = pl.origin + e1[None] * lam[:, None] + d[None] * (sgn * (gate + row * rp))[:, None]
    return dict(S=S, d=d, e1=e1, e2=e2, order=order, gate=gate, rows=rows)


class Flight:
    """a mark or blob travelling a bezier route with a Hermite speed profile.
    entering: from the gate (heading d, speed v0) to a target (speed 0) in T seconds;
    leaving:  from the word (speed 0) to the gate (heading d, speed v1) in T seconds, then coasting on along d at v1."""

    def __init__(self, A, B, d, v, T, entering, e1=None, wob=0.0, coast=2000.0, k=0.32):
        A, B, d = np.asarray(A, float), np.asarray(B, float), unit(d)
        L0 = float(np.linalg.norm(B - A)) + 1e-6
        e1 = np.zeros(3) if e1 is None else e1
        if entering:
            toward = unit(B - A)
            p1 = A + d * (k * L0)
            p2 = B - unit(0.5 * d + 0.5 * toward) * (k * L0) + e1 * wob * L0
        else:
            p1 = A + unit(0.5 * d + 0.5 * unit(B - A)) * (k * L0) + e1 * wob * L0
            p2 = B - d * (k * L0)
        bez = bezier(A, p1, p2, B, step=5.0)
        self.Lb = Route3(bez).L
        self.R = Route3(bez if entering else np.vstack([bez, B + d * coast]))
        self.entering, self.T = entering, max(float(T), 1e-6)
        self.v = float(v)
        self.m = clamp_slope(self.v * self.T, self.Lb, 2.9)

    def s_of(self, u):
        """arc length travelled at flight fraction u (>1 only for leaving: coasting at the end speed)."""
        if self.entering:
            return float(herm(0.0, self.Lb, self.m, 0.0, min(max(u, 0.0), 1.0)))
        if u >= 1.0:
            return self.Lb + self.m * (u - 1.0)
        return float(herm(0.0, self.Lb, 0.0, self.m, max(u, 0.0)))

    def at(self, u):
        s = self.s_of(u)
        return self.R.at(s), self.R.tangent_at(s)

    def speed(self, u):
        e = 1e-3
        return abs(self.s_of(u + e) - self.s_of(u - e)) / (2 * e * self.T) if u > e else abs(self.s_of(e) - self.s_of(0.0)) / (e * self.T)


def fit_entering(S, B, d, v, Tw, start, arrive, e1, wob):
    """Flight of an item that waits in the moving stream (position S + d v t) until it peels off at window fraction `start` and
    arrives at B at fraction `arrive` (Tw = window seconds). The flight is shortened (later start) when v * T would be too much for
    the Hermite profile to leave at speed v, so the item really leaves the stream at the stream speed. -> (start, Flight)."""
    for _ in range(4):
        fl = Flight(S + d * v * start * Tw, B, d, v, (arrive - start) * Tw, True, e1, wob)
        Tmax = 2.8 * fl.Lb / max(v, 1e-6)
        if fl.T <= Tmax:
            return start, fl
        start = arrive - Tmax / Tw
    fl = Flight(S + d * v * start * Tw, B, d, v, (arrive - start) * Tw, True, e1, wob)
    return start, fl


def fit_leaving(A, G, d, v, Tw, start, e1, wob):
    """Flight from A (speed 0) at window fraction `start` to the gate point G at fraction 1, leaving at speed v (started later
    when the whole window would be too long for that). -> (start, Flight)."""
    fl = Flight(A, G, d, v, (1.0 - start) * Tw, False, e1, wob)
    Tmax = 2.8 * fl.Lb / max(v, 1e-6)
    if fl.T > Tmax:
        start = 1.0 - Tmax / Tw
        fl = Flight(A, G, d, v, Tmax, False, e1, wob)
    return start, fl
