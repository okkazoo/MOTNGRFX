"""twist: carries A's exit strands to B's entry strands as a cable. The strands are paired by optimal assignment
(core.adapt), follow a smooth axis curve from A's exit gate to B's entry gate, squeeze into a tight round bundle in the
middle and wind around the bundle axis `turns` times (handed = +1 / -1 picks the screw direction) before opening back out
into B's entry arrangement. The route is a fixed 3D cable; the strands are windows sliding through it."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from mograph.modules.core import TransitionModule, register, adapt, Route3, Slide, strand_layers, unit, flock
from mograph.render.cam3d import smoother


def _hermite_curve(p0, p1, m0, m1, u):
    u = u[:, None]
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


@register("twist")
class Twist(TransitionModule):
    """params: turns (int winds, None = auto from length / pitch), handed (+1 / -1), radius (bundle radius, None = auto),
    pitch (px of axis per wind when turns is auto, 380), core (0..1, fraction of the hand-off cross-section the cable
    keeps mid-way, 0.55; radius is only the floor), stiff (axis curve stiffness, 1.0), squeeze (0..1, how tightly
    the strands gather in the middle, 1.0), stretch (0..0.4, how far the strand bodies lengthen mid-way so the cable
    reads as long, 0.3), piece (px, depth-sort piece length for tubes, 70)."""
    DEMO_PARAMS = dict(turns=2, stretch=0.3)
    # what the transition engine (mograph.modules.engine) may use it for: {feature: (shape, weight)}, see engine.py
    SUITS = dict(
        brief="medium distance, any heading, any plane (the generalist)",
        dist=((60, 220, 800, 1500), 1.5), plane=({"same": 1.0, "fold": 0.9, "opposite": 0.7}, 0.4),
        energy=((0.0, 0.25, 1.0, 1.0), 0.5), base=(0.55, 2.5), must=("dist",), dur=(1.2, 3.4, 1.0),
        moods={"playful": 0.7, "joyful": 0.7, "mysterious": 0.5, "hype": 0.6, "calm": 0.6, "tense": 0.5, "nostalgic": 0.5,
               "melancholy": 0.4})
    description = ("Pairs the departing strands of one word with the arriving strands of the next, gathers them into a "
                   "tight cable and winds them around the cable axis in a helix before they fan back out. Tune it with "
                   "turns and handed.")

    def build(self):
        P = self.params
        pr = adapt(self.a_exit, self.b_entry)
        self.pairing = pr
        A, B = pr.bundles(self.a_exit, self.b_entry)
        self.A, self.B, K = A, B, pr.K
        Ga, Tb = A.P[:, 0], B.P[:, -1]
        ca, cb = Ga.mean(0), Tb.mean(0)
        da, db = unit(A.D.mean(0)), unit(B.D.mean(0))
        dist = float(np.linalg.norm(cb - ca))
        m = float(P.get("stiff", 1.0)) * (0.9 * dist + 80.0)
        n = max(80, int(dist * 1.6 / 5.0))
        u = np.linspace(0, 1, n)
        C = _hermite_curve(ca, cb, da * m, db * m, u)
        Tg = unit(np.gradient(C, axis=0))
        base = np.stack([_hermite_curve(Ga[k], Tb[k], A.D[k] * m, B.D[k] * m, u) for k in range(K)], 1)     # (n,K,3)
        rel = base - C[:, None, :]
        # parallel-transport frame along the axis
        e1 = np.zeros((n, 3)); e2 = np.zeros((n, 3))
        ref = np.array([0.0, 1.0, 0.0]) if abs(Tg[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        a = unit(ref - (ref @ Tg[0]) * Tg[0])
        for i in range(n):
            a = unit(a - (a @ Tg[i]) * Tg[i])
            e1[i], e2[i] = a, np.cross(Tg[i], a)
        al = np.einsum("nkc,nc->nk", rel, e1)
        be = np.einsum("nkc,nc->nk", rel, e2)
        ga = np.einsum("nkc,nc->nk", rel, Tg)
        tube = max(A.tube, B.tube)
        wm = float(0.5 * (A.w.mean() + B.w.mean()))
        rho = float(P.get("radius") or (1.5 * math.sqrt(K) * wm * (1 + 0.6 * tube) + 6.0))
        # cross-section: the cable keeps `core` x the shape of A's exit cross-section and morphs it into B's entry cross-section
        # (B's rotated in the cable plane to the best fit), so a word-high ribbon winds as a ribbon instead of a tiny knot.
        c0, c1 = np.stack([al[0], be[0]], 1), np.stack([al[-1], be[-1]], 1)
        c0, c1 = c0 - c0.mean(0), c1 - c1.mean(0)
        th0 = math.atan2(float((c1[:, 0] * c0[:, 1] - c1[:, 1] * c0[:, 0]).sum()), float((c1 * c0).sum()) + 1e-9)
        R2 = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])
        c1r = c1 @ R2.T
        core = float(P.get("core", 0.55))
        mm = smoother(u)[:, None, None]
        shape = core * ((1 - mm) * c0[None] + mm * c1r[None])                       # (n,K,2)
        rms = np.sqrt((shape ** 2).sum(-1).mean(1))                                  # (n,)
        floor = np.maximum(rho / np.maximum(rms, 1e-6), 1.0)[:, None, None] if rho > 0 else 1.0
        shape = shape * np.minimum(floor, 6.0)                                       # never thinner than the tight-cable radius
        L_axis = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
        turns = P.get("turns")
        turns = max(1, int(round(L_axis / float(P.get("pitch", 380.0))))) if turns is None else int(turns)
        self.turns = turns
        hand = 1.0 if float(P.get("handed", 1)) >= 0 else -1.0
        th = hand * 2 * math.pi * turns * smoother(u)
        sq = float(P.get("squeeze", 1.0))
        kap = sq * smoother(np.minimum(1, u / 0.3)) * smoother(np.minimum(1, (1 - u) / 0.3))
        x = (1 - kap)[:, None] * al + kap[:, None] * shape[:, :, 0]
        y = (1 - kap)[:, None] * be + kap[:, None] * shape[:, :, 1]
        cs, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        xr, yr = x * cs - y * sn, x * sn + y * cs
        gr = ga * (1 - kap)[:, None]
        pts = C[:, None, :] + xr[:, :, None] * e1[:, None, :] + yr[:, :, None] * e2[:, None, :] + gr[:, :, None] * Tg[:, None, :]
        pts[0] = Ga
        pts[-1] = Tb
        self.routes, self.slides = [], []
        T = self.dur
        unr = bool(P.get("unravel"))
        mids = [pts[:, k] for k in range(K)]
        if unr:                                                  # flock (core.flock): staggered peel-off / arrival, each strand bent a little off the cable
            dl, Tk, mids = flock(K, T, int(P.get("seed", 0)), mids, amp=0.04 * max(dist, 200.0))
        for k in range(K):
            body_a, body_b = A.P[k][::-1], B.P[k][::-1]
            R = Route3(np.vstack([body_a, mids[k][1:-1], body_b]))
            La = Route3(body_a).L
            Lc = Route3(np.vstack([body_a[-1:], mids[k][1:-1], body_b[:1]])).L
            self.routes.append(R)
            if unr:
                sl = Slide(R, La, R.L, 0.0, La + Lc, float(Tk[k]), float(A.v[k]), float(B.v[k]), delay=float(dl[k]), k=1.6)
            else:
                sl = Slide(R, La, R.L, 0.0, La + Lc, T, float(A.v[k]), float(B.v[k]), k=1.6)
            sl.tail_lag = float(P.get("stretch", 0.3))
            self.slides.append(sl)
        self.tube = tube
        self.axis = C

    def _mix(self, t):
        f = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
        f = f * f * (3 - 2 * f)
        A, B = self.A, self.B
        return (A.col * (1 - f) + B.col * f, A.w * (1 - f) + B.w * f, A.rim * (1 - f) + B.rim * f,
                A.tube * (1 - f) + B.tube * f)

    def windows(self, t):
        return [sl.window(t) for sl in self.slides]

    def layers(self, t):
        col, w, rim, tube = self._mix(t)
        polys, cols, ws, rims = [], [], [], []
        piece = float(self.params.get("piece", 70.0))
        for k, (win, h, tl) in enumerate(self.windows(t)):
            if win is None:
                continue
            R = self.routes[k]
            if tube > 0.01:
                a = math.floor(tl / piece) * piece
                while a < h:
                    seg = R.window(max(a, tl), min(a + piece, h))
                    if seg is not None:
                        polys.append(seg); cols.append(col[k]); ws.append(w[k]); rims.append(rim[k])
                    a += piece
            else:
                polys.append(win); cols.append(col[k]); ws.append(w[k]); rims.append(rim[k])
        return strand_layers(polys, cols, ws, tube, np.array(rims) if rims else None, hi_off=np.array([-0.8, -0.8, 0.0]))

    def focus(self, t):
        return np.mean([self.routes[k].at(h) for k, (win, h, tl) in enumerate(self.windows(t))], 0)

    def bounds_points(self):
        return np.concatenate([self.axis, self.A.P.reshape(-1, 3), self.B.P.reshape(-1, 3)])
