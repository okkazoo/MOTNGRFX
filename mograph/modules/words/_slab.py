"""Shared helpers for the flat / slab word modules (batch A: slats, beat_bars, solid_hollow, gradient_fade, scale_clash,
staircase, bold_thin_echo). Not a module itself.

SlabWord: a WordModule that lays the words out in the placement box and knows the travel frame of its ports:
  d_in   unit 3D vector the strands travel along when they ARRIVE (= minus the entry side)
  d_out  unit 3D vector they travel along when they LEAVE (= the exit side)
  reach_in / reach_out  distance from the placement centre to the gate (just outside the box) along -d_in / +d_out
Modules build their own geometry from these and draw cam3d Layers; the Bundles handed over at the ends are always built from
the same state function the frames use (state(t=0) / state(t=duration)), so the contract A.layers(A.duration) == exit_bundle()
holds by construction."""
import math
import numpy as np
from mograph.modules.core import (WordModule, Bundle, Layer, layout_word, unit, clamp, mixc, ol)


def ssm(t, a, b):
    """smoothstep of t between a and b."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    x = min(1.0, max(0.0, (t - a) / (b - a)))
    return x * x * (3 - 2 * x)


def eo(u, p=2.0):
    """ease out: 1 - (1-u)^p (starts at speed p, lands softly)."""
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


def ei(u, p=2.0):
    u = min(1.0, max(0.0, u))
    return u ** p


def eio(u, p=2.0):
    u = min(1.0, max(0.0, u))
    a = u ** p
    return a / (a + (1.0 - u) ** p)


def hull2(P):
    """convex hull of (n,2) points (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, np.asarray(P, float)))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def simplify(c, tol=0.6, corner_deg=30.0):
    """thin a closed contour (n,2): drop points that lie within tol of the chord of their neighbours (Douglas-Peucker on the
    closed loop, split at the two farthest points). Keeps corners."""
    c = np.asarray(c, float)
    if len(c) < 8:
        return c
    i0 = int(np.argmin(c[:, 0]))
    d = np.linalg.norm(c - c[i0], axis=1)
    i1 = int(np.argmax(d))
    a, b = sorted((i0, i1))
    seg1 = c[a:b + 1]
    seg2 = np.vstack([c[b:], c[:a + 1]])

    def dp(P):
        if len(P) < 3:
            return P
        s, e = P[0], P[-1]
        v = e - s
        n = np.linalg.norm(v)
        if n < 1e-9:
            dist = np.linalg.norm(P - s, axis=1)
        else:
            dist = np.abs(v[0] * (P[:, 1] - s[1]) - v[1] * (P[:, 0] - s[0])) / n
        k = int(np.argmax(dist))
        if dist[k] <= tol:
            return np.array([s, e])
        return np.vstack([dp(P[:k + 1])[:-1], dp(P[k:])])
    out = np.vstack([dp(seg1)[:-1], dp(seg2)[:-1]])
    return out if len(out) >= 3 else c


def corner_points(cons, kmax=6, min_deg=34.0):
    """a few well spread corner vertices of a letter's contours (local 2D): the bar edges handed over as strands."""
    cand = []
    for c in cons:
        n = len(c)
        d1 = c - np.roll(c, 1, 0)
        d2 = np.roll(c, -1, 0) - c
        a = np.arctan2(d1[:, 1], d1[:, 0])
        b = np.arctan2(d2[:, 1], d2[:, 0])
        turn = np.degrees(np.abs((b - a + np.pi) % (2 * np.pi) - np.pi))
        idx = np.where(turn > min_deg)[0]
        if len(idx) == 0:
            idx = np.linspace(0, n, 4, endpoint=False).astype(int)
        cand += [c[i] for i in idx]
    cand = np.array(cand)
    if len(cand) <= kmax:
        return cand
    return cand[np.linspace(0, len(cand) - 1, kmax).astype(int)]



def nz_groups(cons):
    """split a letter's contours into 'outer + its holes' groups, so overlapping outers (variable fonts overlap their
    contours) can be filled as separate Layers (a union) while counters stay open inside their own group. -> list of
    lists of indices into `cons`."""
    import cv2
    ar = np.array([0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])) for c in cons])
    if len(cons) == 0:
        return []
    sg = 1.0 if ar[int(np.argmax(np.abs(ar)))] >= 0 else -1.0
    outers = [i for i in range(len(cons)) if ar[i] * sg > 0]
    groups = {i: [i] for i in outers}
    for h in range(len(cons)):
        if ar[h] * sg > 0:
            continue
        pt = tuple(float(x) for x in cons[h][0])
        best = None
        for o in sorted(outers, key=lambda i: abs(ar[i])):
            if cv2.pointPolygonTest(cons[o].astype(np.float32), pt, False) >= 0:
                best = o
                break
        if best is None and outers:
            best = outers[0]
        if best is not None:
            groups[best].append(h)
    return [groups[i] for i in outers]


class SlabWord(WordModule):
    GAP = 40.0
    BOX = (0.94, 0.9)           # word box as a fraction of the placement

    # ------------------------------------------------------------ frame
    def _reach(self, d):
        """distance from the placement centre to the gate along d: the box half extent along d (at least a fraction of the
        box for travel along the viewing axis, where the extent is 0), plus a gap."""
        pl = self.placement
        ext = pl.extent_along(d)
        ext = max(ext, 0.9 * max(pl.w, pl.h) * abs(float(d @ pl.n)) + 0.0)
        return float(ext + self.GAP)

    def build(self):
        pl = self.placement
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.n3 = pl.n
        self.reach_in = self._reach(self.d_in)
        self.reach_out = self._reach(self.d_out)
        self.lay = layout_word(self.font, self.lines, pl.w * self.BOX[0], pl.h * self.BOX[1])
        self.cap = self.lay["cap"]
        self.letters = self.lay["letters"]
        self.cons = [c for l in self.letters for c in l["cons"]]
        self.setup()

    def setup(self):
        raise NotImplementedError

    # ------------------------------------------------------------ phases
    def phase(self, t):
        tm = self.timing
        if t < tm.form:
            return "form", t
        if t < tm.form + tm.hold:
            return "hold", t - tm.form
        return "exit", t - tm.form - tm.hold

    # ------------------------------------------------------------ geometry helpers
    def lift(self, P2, z=0.0):
        """local 2D (placement frame, y down) -> world; z along the placement normal (+ away from the viewer)."""
        pl = self.placement
        return pl.lift(P2) + pl.n * z

    def fill_sets(self):
        """per letter: list of groups (an outer contour + its holes, local 2D), computed once."""
        if not hasattr(self, "_fsets"):
            self._fsets = [[[l["cons"][i] for i in g] for g in nz_groups(l["cons"])] for l in self.letters]
        return self._fsets

    def fill_layer(self, col, alpha=1.0, bias=2.0, off=None):
        """solid fill of every letter at the placement plane (overlap-safe: one Layer per outer contour + holes)."""
        pl = self.placement
        out = []
        for gs in self.fill_sets():
            for g in gs:
                if off is None:
                    out.append(Layer(g, col, "fill", plane=pl.plane, alpha=alpha, bias=bias))
        return out

    # ------------------------------------------------------------ bundles / camera hints
    def entry_bundle(self):
        return self.bundle_at(0.0)

    def exit_bundle(self):
        return self.bundle_at(self.duration)

    def bundle_at(self, t):
        raise NotImplementedError

    def focus(self, t):
        pl = self.placement
        tm = self.timing
        t = min(max(t, 0.0), self.duration)
        o = pl.origin.copy()
        if t < tm.form:
            u = ssm(t, 0.0, 0.8 * tm.form)
            return o - self.d_in * (self.reach_in * 0.7) * (1 - u)
        if t < tm.form + tm.hold:
            return o
        u = ssm(t - tm.form - tm.hold, 0.2 * tm.exit, tm.exit)
        return o + self.d_out * (self.reach_out * 0.7) * u


# ------------------------------------------------------------------ prism walls (shared by the extruded modules)
def contour_sign(cons):
    """+1 / -1: orientation of the largest contour of a letter (so wall normals point out of the solid for every contour)."""
    best, bs = 0.0, 1.0
    for c in cons:
        a = 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
        if abs(a) > abs(best):
            best, bs = a, (1.0 if a >= 0 else -1.0)
    return bs


def wall_quads(pl, cons, sg, off, ax, D, Ls, light=(-0.4, -0.9), cull=True, band=False):
    """walls of the prism of one letter: face contours `cons` (local 2D, simplified) lifted to the placement plane + world
    offset `off`, extruded from the face along unit vector `ax` by length D, cut into slabs of length Ls. -> dict
    slab k -> dict shade level (0..2) -> list of (4,3) quads, only for walls facing the extrusion side (back-face cull).
    The slabs run from the face (k = 0) to the back."""
    out = {}
    if D < 0.5:
        return out
    u, v = pl.u, pl.v
    ax2 = np.array([ax @ u, ax @ v])
    nax2 = np.linalg.norm(ax2)
    lt = np.array(light, float)
    lt /= np.linalg.norm(lt)
    nsl = int(math.ceil(D / Ls - 1e-9))
    edges = []
    for c in cons:
        p0, p1 = c, np.roll(c, -1, 0)
        t = p1 - p0
        nl = np.linalg.norm(t, axis=1) + 1e-9
        no = sg * np.stack([t[:, 1], -t[:, 0]], 1) / nl[:, None]
        if cull and nax2 > 0.12:
            vis = (no @ ax2) > 0.0
        else:
            vis = np.ones(len(c), bool)
        if not vis.any():
            continue
        mid = 0.5 * (p0 + p1)[vis]
        sh = np.abs(no[vis] @ lt)
        lv = np.minimum((sh * 3).astype(int), 2)
        if band:                               # level 0 = walls whose flat edge is near-horizontal, 1 = the rest
            lv = np.where(np.abs(t[vis][:, 1]) / nl[vis] < 0.5, 0, 1)
        a3 = pl.lift(p0[vis]) + off
        b3 = pl.lift(p1[vis]) + off
        key = (mid @ ax2) if nax2 > 0.12 else np.zeros(len(mid))
        edges.append((a3, b3, lv, key))
    if not edges:
        return out
    A = np.concatenate([e[0] for e in edges])
    B = np.concatenate([e[1] for e in edges])
    LV = np.concatenate([e[2] for e in edges])
    KY = np.concatenate([e[3] for e in edges])
    o = np.argsort(KY)
    A, B, LV = A[o], B[o], LV[o]
    for k in range(nsl):
        s0, s1 = k * Ls, min((k + 1) * Ls, D)
        Q = np.stack([A + ax * s0, B + ax * s0, B + ax * s1, A + ax * s1], 1)
        d = {}
        for lvl in (0, 1, 2):
            m = LV == lvl
            if m.any():
                d[lvl] = list(Q[m])
        out[k] = d
    return out


def wall_layers(slabs_list, Ls, Dref, near, far, hfac=(1.0, 0.86, 0.72), alpha=1.0, bias0=3.0, gamma=1.0, level_cols=None):
    """slabs_list: list of wall_quads() dicts (one per letter). Merge per slab, back slab first, one Layer per (slab, shade).
    Colour of slab k = mix(near, far, ((k+0.5) Ls / Dref)^gamma) times the shade factor."""
    kmax = max((max(s) for s in slabs_list if s), default=-1)
    L = []
    for k in range(kmax, -1, -1):
        tone = min(1.0, ((k + 0.5) * Ls / max(Dref, 1e-6))) ** gamma
        base = np.array(mixc(near, far, tone), float)
        for lvl in (0, 1, 2):
            quads = []
            for s in slabs_list:
                if k in s and lvl in s[k]:
                    quads += s[k][lvl]
            if quads:
                col = tuple(int(x) for x in np.clip(base * hfac[lvl], 0, 255)) if level_cols is None else tuple(level_cols[lvl])
                L.append(Layer(quads, col, "fill", split=True, alpha=alpha, bias=bias0 + 0.05 * (kmax - k)))
    return L


class LetterFlow:
    """per-letter travel in / out along the module's d_in / d_out (letters or any units as rigid pieces).
    pts_world: list of (n,3) world points per unit (rest pose). Timings follow the module timing: unit j lands at Ti[j] (far
    ones first, ease-out from the entry speed), leaves after delay dx[j] (nearest the exit first, ease-in, ends together at the
    end of the exit phase and keeps coasting). `tail` = strand length handed over at the ends (the whole piece, plus this tail,
    starts / ends outside the gate)."""

    def __init__(self, mod, pts_world, tail, rng, land=(0.30, 0.20), lead=0.38, p_in=2.4, p_out=2.2):
        self.m = mod
        o = mod.placement.origin
        n = len(pts_world)
        self.n, self.p_in, self.p_out = n, p_in, p_out
        rin = np.array([float(((P - o) @ mod.d_in).mean()) for P in pts_world])
        rout = np.array([float(((P - o) @ mod.d_out).mean()) for P in pts_world])
        hi = np.array([float(((P - o) @ mod.d_in).max()) for P in pts_world])
        lo = np.array([float(((P - o) @ mod.d_out).min()) for P in pts_world])
        self.fin = 1.0 - (rin - rin.min()) / max(np.ptp(rin), 1e-6)
        self.fout = 1.0 - (rout - rout.min()) / max(np.ptp(rout), 1e-6)
        tm = mod.timing
        ext = max(mod.placement.w, mod.placement.h)
        self.Ti = (land[0] + land[1] * self.fin) * tm.form + 0.04 * rng.rand(n)
        self.Di = hi + mod.reach_in + 0.08 * ext * rng.rand(n)
        self.dx = lead * tm.exit * (0.9 * self.fout + 0.1 * rng.rand(n))
        self.Tx = tm.exit - self.dx
        self.Dx = mod.reach_out - lo + tail + 0.06 * ext * rng.rand(n)

    def at(self, j, t):
        """-> (offset (3,), speed, tau, entering). tau: progress 0..1 (> 1: coasting out)."""
        m, tm = self.m, self.m.timing
        if t < tm.form + tm.hold:
            tau = t / self.Ti[j]
            d = self.Di[j] * (1 - eo(tau, self.p_in)) if tau < 1 else 0.0
            return -m.d_in * d, self.p_in * self.Di[j] / self.Ti[j], tau, True
        te = t - tm.form - tm.hold
        tau = (te - self.dx[j]) / self.Tx[j]
        if tau <= 0:
            d = 0.0
        elif tau >= 1:
            d = self.Dx[j] * (1 + self.p_out * (tau - 1))
        else:
            d = self.Dx[j] * ei(tau, self.p_out)
        return m.d_out * d, self.p_out * self.Dx[j] / self.Tx[j], tau, False


# ------------------------------------------------------------------ graduated prism (translation + one-point perspective)
class PrismItem:
    """one graduated extrusion: face contours `cons` (local 2D, sign sg), world offset `off`, translation d3 (3D, over the whole
    depth), perspective vanishing point vp2 (local 2D) and back scale sb (1 = none), visibility s (0 = same colour as the
    ground, 1 = full)."""

    def __init__(self, cons, sg, off, d3, vp2, sb, s=1.0):
        self.cons, self.sg, self.off = cons, sg, np.asarray(off, float)
        self.d3, self.vp2, self.sb, self.s = np.asarray(d3, float), np.asarray(vp2, float), float(sb), float(s)


def prism_point(pl, it, P, u):
    """world position of local point(s) P (n,2) on slab depth u of item `it` (u = 0 the face)."""
    d2 = np.array([it.d3 @ pl.u, it.d3 @ pl.v])
    dn = float(it.d3 @ pl.n)
    Q = it.vp2 + (P + d2 * u - it.vp2) * (1.0 - (1.0 - it.sb) * u)
    return pl.lift(Q) + it.off + pl.n * (dn * u)


def grad_prisms(pl, items, n, near, far, ground, gamma=1.0, shade=0.45, light=(-0.3, -0.95), bias0=3.0):
    """Layers (back slab first) for graduated solid extrusions. Slab at depth u in [0,1] of an item: local P -> vp2 + (P + d2 u -
    vp2) * (1 - (1 - sb) u), lifted to the placement plane (+ off, + n * (d3.n) u). Walls cut into n slabs, colour = mix(near, far,
    u^gamma) darkened by shade * (1 - |normal . light|), then mixed toward `ground` by (1 - s). Only walls facing the sweep
    direction are made. Slabs of all items are merged into one Layer per (slab, shade, visibility)."""
    n_ = pl.n
    lt = np.array(light, float)
    lt /= np.linalg.norm(lt)
    us = np.linspace(0.0, 1.0, n + 1)
    groups = {}
    for it in items:
        if it.s < 0.004:
            continue
        d2 = np.array([it.d3 @ pl.u, it.d3 @ pl.v])
        dn = float(it.d3 @ n_)
        A, B, LV, KY = [], [], [], []
        far_pt = 0.0
        for c in it.cons:
            p0, p1 = c, np.roll(c, -1, 0)
            t = p1 - p0
            nl = np.linalg.norm(t, axis=1) + 1e-9
            no = it.sg * np.stack([t[:, 1], -t[:, 0]], 1) / nl[:, None]
            mid = 0.5 * (p0 + p1)
            tow = d2[None, :] + (it.vp2[None, :] - mid) * (1.0 - it.sb)
            vis = np.sum(no * tow, axis=1) > 0
            if not vis.any():
                continue
            sh = np.abs(no[vis] @ lt)
            A.append(p0[vis])
            B.append(p1[vis])
            LV.append(np.minimum((sh * 3).astype(int), 2))
            dv = np.linalg.norm(it.vp2[None, :] - mid[vis], axis=1)
            persp = (1.0 - it.sb) * dv > np.linalg.norm(d2)
            e = d2 / (np.linalg.norm(d2) + 1e-9)
            KY.append(np.where(persp, -dv, -(mid[vis] @ e)))
            far_pt = max(far_pt, float(dv.max()))
        if not A:
            continue
        if np.linalg.norm(d2) + abs(dn) + (1.0 - it.sb) * far_pt < 0.8:
            continue
        A, B, LV, KY = np.concatenate(A), np.concatenate(B), np.concatenate(LV), np.concatenate(KY)
        o = np.argsort(KY)[::-1]
        A, B, LV = A[o], B[o], LV[o]
        pts = [(prism_point(pl, it, A, u), prism_point(pl, it, B, u)) for u in us]
        sk = round(it.s, 2)
        for k in range(n):
            Q = np.stack([pts[k][0], pts[k][1], pts[k + 1][1], pts[k + 1][0]], 1)
            for lvl in (0, 1, 2):
                m = LV == lvl
                if m.any():
                    groups.setdefault((k, lvl, sk), []).extend(list(Q[m]))
    L = []
    hf = (1.0, 1.0 - 0.5 * shade, 1.0 - shade)
    for (k, lvl, sk) in sorted(groups, key=lambda g: (-g[0], g[1], g[2])):
        tone = (0.5 * (us[k] + us[k + 1])) ** gamma
        base = np.array(mixc(near, far, tone), float)
        col = np.clip(base * hf[lvl], 0, 255)
        col = mixc(ground, tuple(col), sk)
        L.append(Layer(groups[(k, lvl, sk)], col, "fill", split=True, bias=bias0 + 0.05 * (n - k)))
    return L
