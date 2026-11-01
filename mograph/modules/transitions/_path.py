"""Shared engine for path transitions (circuit_board, spiral_around, braid, fan): pair the strands of A's exit bundle with
B's entry bundle (core.adapt), let the subclass lay one 3D path per strand from A's hand-off gate to B's, and slide each
strand's window along its path (core.Slide) exactly like twist does.

Subclass: implement `paths(A, B)` -> list of K (n_k, 3) arrays, path k running from A.P[k][0] (A's head, the exit gate) to
B.P[k][-1] (B's tail, the entry gate), endpoints exact. Optional: `extra_layers(t)` (decoration), `DEMO_PARAMS`, `DEMO_DUR`.

Cross-section helpers: `axis_frame(A, B)` (smooth axis curve + parallel-transport frame + every strand's offsets from the
axis), `cross_extent(A, B)` (width / depth of the hand-off ribbons) and `rebuild(...)`."""
import math
import numpy as np
from mograph.modules.core import (TransitionModule, adapt, Route3, Slide, strand_layers, unit, Layer, flock)
from mograph.render.cam3d import smoother


def hermite_curve(p0, p1, m0, m1, u):
    u = np.asarray(u, float)[:, None]
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def ribbon_axis(P):
    """principal direction (unit 3-vector) of the points P (n,3) about their mean; None when they are one point."""
    X = P - P.mean(0)
    if np.abs(X).max() < 1e-6:
        return None
    return np.linalg.svd(X, full_matrices=False)[2][0]


class PathTransition(TransitionModule):
    DEMO_PARAMS = {}
    DEMO_DUR = 3.0

    # ------------------------------------------------------------ subclass API
    def paths(self, A, B):
        raise NotImplementedError

    def extra_layers(self, t):
        return []

    # ------------------------------------------------------------ build
    def pair(self):
        pr = adapt(self.a_exit, self.b_entry)
        self.pairing = pr
        A, B = pr.bundles(self.a_exit, self.b_entry)
        self.A, self.B, self.K = A, B, pr.K
        self.tube = max(A.tube, B.tube)
        return A, B

    def build(self):
        A, B = self.pair()
        pts = self.paths(A, B)
        self.pts = pts
        self.routes, self.slides = [], []
        T = self.dur
        unr = bool(self.params.get("unravel"))
        if unr:                                                  # flock (core.flock): staggered peel-off / arrival, each strand bent a little off the shared movement
            dist = float(np.linalg.norm(B.P[:, -1].mean(0) - A.P[:, 0].mean(0)))
            dl, Tk, pts = flock(self.K, T, int(self.params.get("seed", 0)), pts, amp=0.05 * max(dist, 200.0))
        for k in range(self.K):
            body_a, body_b = A.P[k][::-1], B.P[k][::-1]
            mid = np.asarray(pts[k], float)
            R = Route3(np.vstack([body_a, mid[1:-1], body_b]))
            La = Route3(body_a).L
            Lc = Route3(np.vstack([body_a[-1:], mid[1:-1], body_b[:1]])).L
            self.routes.append(R)
            if unr:
                sl = Slide(R, La, R.L, 0.0, La + Lc, float(Tk[k]), float(A.v[k]), float(B.v[k]), delay=float(dl[k]), k=1.6)
            else:
                sl = Slide(R, La, R.L, 0.0, La + Lc, T, float(A.v[k]), float(B.v[k]), k=1.6)
            sl.tail_lag = float(self.params.get("stretch", 0.3))
            self.slides.append(sl)
        self.axis = np.concatenate([np.asarray(p, float)[::4] for p in pts]) if pts else np.zeros((0, 3))

    # ------------------------------------------------------------ cross-section tools
    @staticmethod
    def cross_extent(A, B):
        """(width, depth) of the hand-off cross-sections: the spread of the strand heads of A / tails of B along their own
        principal axis (width) and across it (depth), averaged."""
        ws, ds = [], []
        for X in (A.P[:, 0], B.P[:, -1]):
            c = X - X.mean(0)
            if len(X) < 2 or np.abs(c).max() < 1e-6:
                ws.append(0.0); ds.append(0.0)
                continue
            sv = np.linalg.svd(c, full_matrices=False)
            ax = sv[2][0]
            t = c @ ax
            ws.append(float(t.max() - t.min()))
            r = c - np.outer(t, ax)
            ds.append(float(np.linalg.norm(r, axis=1).max() * 2))
        return 0.5 * (ws[0] + ws[1]), 0.5 * (ds[0] + ds[1])

    @staticmethod
    def min_pitch(A, B, tube=None):
        """centre-to-centre spacing below which two strands visibly touch."""
        tube = max(A.tube, B.tube) if tube is None else tube
        wm = float(0.5 * (A.w.mean() + B.w.mean()))
        return wm * (1.0 + 0.6 * tube) * 1.15 + 1.0

    def axis_frame(self, A, B, n=None, stiff=1.0):
        """-> dict(u, C (n,3), T (n,3), e1, e2 (n,3), al, be, ga (n,K): each strand's offset from the axis curve in the
        parallel-transport frame, Ga, Tb) where the strands' own Hermite curves give the offsets."""
        Ga, Tb = A.P[:, 0], B.P[:, -1]
        ca, cb = Ga.mean(0), Tb.mean(0)
        da, db = unit(A.D.mean(0)), unit(B.D.mean(0))
        dist = float(np.linalg.norm(cb - ca))
        m = float(stiff) * (0.9 * dist + 80.0)
        n = n or max(80, int(dist * 1.6 / 5.0))
        u = np.linspace(0, 1, n)
        C = hermite_curve(ca, cb, da * m, db * m, u)
        Tg = unit(np.gradient(C, axis=0))
        base = np.stack([hermite_curve(Ga[k], Tb[k], A.D[k] * m, B.D[k] * m, u) for k in range(len(Ga))], 1)
        rel = base - C[:, None, :]
        e1 = np.zeros((n, 3)); e2 = np.zeros((n, 3))
        ax = ribbon_axis(Ga)
        ref = ax if ax is not None else (np.array([0.0, 1.0, 0.0]) if abs(Tg[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
        a = unit(ref - (ref @ Tg[0]) * Tg[0])
        if np.linalg.norm(ref - (ref @ Tg[0]) * Tg[0]) < 1e-6:
            a = unit(np.cross(Tg[0], [0.0, 1.0, 0.0]))
        for i in range(n):
            a = unit(a - (a @ Tg[i]) * Tg[i])
            e1[i], e2[i] = a, np.cross(Tg[i], a)
        return dict(u=u, C=C, T=Tg, e1=e1, e2=e2, Ga=Ga, Tb=Tb, ca=ca, cb=cb,
                    al=np.einsum("nkc,nc->nk", rel, e1), be=np.einsum("nkc,nc->nk", rel, e2),
                    ga=np.einsum("nkc,nc->nk", rel, Tg), dist=dist)

    @staticmethod
    def rebuild(F, x, y, gr=None):
        """3D points (n,K,3) from the axis frame F and per-strand frame coordinates x (e1), y (e2), gr (axial); the ends are
        pinned to A's gate and B's gate."""
        pts = F["C"][:, None, :] + x[:, :, None] * F["e1"][:, None, :] + y[:, :, None] * F["e2"][:, None, :]
        if gr is not None:
            pts = pts + gr[:, :, None] * F["T"][:, None, :]
        pts[0] = F["Ga"]
        pts[-1] = F["Tb"]
        return pts

    # ------------------------------------------------------------ frame
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
        out = self.extra_layers(t)
        return out + strand_layers(polys, cols, ws, tube, np.array(rims) if rims else None, hi_off=np.array([-0.8, -0.8, 0.0]))

    def focus(self, t):
        return np.mean([self.routes[k].at(h) for k, (win, h, tl) in enumerate(self.windows(t))], 0)

    def bounds_points(self):
        return np.concatenate([self.axis, self.A.P.reshape(-1, 3), self.B.P.reshape(-1, 3)])
