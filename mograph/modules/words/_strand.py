"""Shared engine for strand-built word modules (lanes, pipework): the type outline is cut into N open arcs; every arc is
one strand. Entering, a strand runs down its lane (parallel lanes packed into a ribbon on the entry side), peels off
and draws its arc; leaving, it detaches from the arc, peels into an exit lane and runs away. Subclasses choose how a
lane joins an arc (`connect`) and how strands are drawn (`draw`)."""
import math
import numpy as np
from mograph.modules.core import (WordModule, Bundle, Route3, Slide, Layer, layout_word, unit, ol, strand_layers, clamp)


def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def make_arcs(cons, N, step=3.0):
    """cut closed contours (local 2D) into exactly max(N, len(cons)) open arcs; longest arcs are split first."""
    Q, L = [], []
    for c in cons:
        c = np.vstack([c, c[:1]])
        cum = ol.arclen(c)
        if cum[-1] < 1.0:
            continue
        n = max(8, int(round(cum[-1] / step)))
        s = np.linspace(0, cum[-1], n + 1)
        Q.append(np.stack([np.interp(s, cum, c[:, 0]), np.interp(s, cum, c[:, 1])], 1))
        L.append(cum[-1])
    L = np.array(L)
    N = max(N, len(Q))
    cnt = np.ones(len(Q), int)
    for _ in range(N - len(Q)):
        cnt[int(np.argmax(L / cnt))] += 1
    arcs = []
    for q, k in zip(Q, cnt):
        cum = ol.arclen(q)
        for i in range(k):
            P = ol.sub(q, cum, cum[-1] * i / k, cum[-1] * (i + 1) / k)
            if P is not None and len(P) > 1:
                arcs.append(P)
    return arcs


def lane_frame(d, pl):
    """(e1, e2): e1 lateral across the word (in the placement plane when d is not along its normal), e2 = d x e1."""
    e1 = np.cross(pl.n, d)
    if np.linalg.norm(e1) < 0.3:
        e1 = pl.u - (pl.u @ d) * d
    e1 = unit(e1 - (e1 @ d) * d)
    return e1, np.cross(d, e1)


def manhattan(p0, p1, d, e1, e2, f, ch):
    D = p1 - p0
    dd, d1, d2 = float(D @ d), float(D @ e1), float(D @ e2)
    pts = [p0, p0 + d * (f * dd)]
    pts.append(pts[-1] + e1 * d1)
    pts.append(pts[-1] + e2 * d2)
    pts.append(p1.copy())
    from mograph.modules.core import chamfer
    return chamfer(pts, ch)


class StrandWord(WordModule):
    TUBE = 0.0              # 0 hairline strands, 1 shaded tubes
    ARC_PER_CAP = 0.42      # target arc length / cap height (sets N when n_strands is not given)
    FILL = "solid"          # 'solid' (fg fill fades in at the end of form) | 'panel' (faint inner fill)
    LINE_W = 1.5
    CHAMFER = 0.0           # >0: Manhattan joins with this chamfer (px), 0: bezier joins
    RIBBON = 1.0            # cross-section of the entry / exit bundle = RIBBON x the word's own extent across the lanes
    FAN = 0.5               # default port spread: lanes fan out by this fraction toward the far end of the bundle
    TAIL = 0.6              # bundle (strand body) length / max(placement w, h)

    # ------------------------------------------------------------ build
    def build(self):
        pl, P = self.placement, self.params
        self.lay = layout_word(self.font, self.lines, pl.w * 0.94, pl.h * 0.9)
        cap = self.lay["cap"]
        self.cap = cap
        cons = [c for l in self.lay["letters"] for c in l["cons"]]
        perim = sum(float(ol.arclen(np.vstack([c, c[:1]]))[-1]) for c in cons)
        N = int(P.get("n_strands") or int(np.clip(round(perim / (self.ARC_PER_CAP * cap)), 12, 140)))
        arcs2 = make_arcs(cons, N)
        self.N = N = len(arcs2)
        arcs3 = [pl.lift(a) for a in arcs2]
        self.w = float(P.get("width") or (self.LINE_W * max(0.8, cap / 200.0) if not self.TUBE else float(np.clip(0.045 * cap, 4.5, 9.0))))
        self.Lb = float(P.get("tail_len") or self.TAIL * max(pl.w, pl.h))
        self.run = float(P.get("run") or 0.1 * max(pl.w, pl.h))
        self.coast_len = 2.5 * max(pl.w, pl.h)
        self.pgap = 0.06 * max(pl.w, pl.h) + 12.0
        self.cols = self._colours(N)
        tm = self.timing
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        fan = float(P.get("fan", self.FAN))
        self._in = self._side(arcs3, self.d_in, True, max(self.entry.spread, fan))
        self._out = self._side(arcs3, self.d_out, False, max(self.exit.spread, fan))
        self._routes(tm)
        self._bundles()

    def _colours(self, N):
        pal = self.palette
        if not self.TUBE:
            return np.tile(np.array(pal.fg, float), (N, 1))
        return np.array([pal.accents[i % len(pal.accents)] for i in range(N)], float)

    def connect(self, p0, t0, p1, t1, ctx):
        """polyline from p0 (heading t0) to p1 (arriving heading t1). default: cubic bezier."""
        k = 0.4 * float(np.linalg.norm(p1 - p0)) + 1e-3
        from mograph.modules.core import bezier
        return bezier(p0, p0 + t0 * k, p1 - t1 * k, p1)

    def _side(self, arcs3, d, entering, spread):
        pl = self.placement
        c = pl.origin
        N = len(arcs3)
        e1, e2 = lane_frame(d, pl)
        arcs, ts, te = [], [], []
        for a in arcs3:
            k = min(5, len(a) - 1)
            t_s, t_e = unit(a[k] - a[0]), unit(a[-1] - a[-1 - k])
            flip = ((-t_e) @ d > t_s @ d) if entering else ((-t_s) @ d > t_e @ d)
            if flip:
                a = a[::-1]
                t_s, t_e = -t_e, -t_s
            arcs.append(a); ts.append(t_s); te.append(t_e)
        attach = np.array([a[0] if entering else a[-1] for a in arcs])
        rel = attach - c
        a1, a2, sd = rel @ e1, rel @ e2, rel @ d                 # lateral coords of the attach points, along-d coordinate
        order = np.argsort(a1)
        rank = np.empty(N, int)
        rank[order] = np.arange(N)
        tunnel = abs(d @ pl.n) > 0.7                             # strands fly straight down the view axis into their start points
        if tunnel:
            lam, mu = a1.copy(), a2.copy()
        else:
            minw = 1.5 * self.w if self.TUBE else 3.0
            lo, hi = a1.min(), a1.max()
            span = float(self.params.get("ribbon", self.RIBBON)) * (hi - lo)
            rows = 1
            if self.TUBE and N * minw > 1.25 * max(span, 1.0):       # too many tubes for one row: pack two layers (hex), the
                rows = 2                                              # second layer sits behind the first
            gap = max(minw / rows, span / max(N - 1, 1))
            lam = 0.5 * (lo + hi) + (rank - (N - 1) / 2.0) * gap
            mu = ((rank % rows) - (rows - 1) / 2.0) * (1.05 * minw) if rows > 1 else np.zeros(N)
        lateral = e1 * lam[:, None] + e2 * mu[:, None]
        pls = (14.0 + 0.6 * np.hypot(a1 - lam, a2 - mu)) if not tunnel else np.full(N, 0.08 * max(pl.w, pl.h) + 14.0)
        ext = pl.extent_along(d)
        if entering:
            peel = c + lateral + d[None, :] * (sd - pls)[:, None]
            far = c + lateral * (1.0 + spread) - d * (ext + self.pgap + self.run * (3.5 if tunnel else 1.0))
            Dl = unit(peel - far)
        else:
            peel = c + lateral + d[None, :] * (sd + pls)[:, None]
            far = c + lateral * (1.0 + spread) + d * (ext + self.pgap + max(self.run, self.Lb + 30.0))
            Dl = unit(far - peel)
        frac = (sd - sd.min()) / max(sd.max() - sd.min(), 1e-6)
        return dict(d=d, e1=e1, e2=e2, arcs=arcs, ts=np.array(ts), te=np.array(te), rank=rank, lam=lam, mu=mu, peel=peel,
                    far=far, D=Dl, frac=frac)

    def _jog(self, rank, N, shift):
        f = rank / max(N - 1, 1.0)
        return 0.2 + 0.55 * ((1 - f) if shift > 0 else f)

    def _routes(self, tm):
        N, Lb = self.N, self.Lb
        I, O = self._in, self._out
        self.in_slide, self.out_slide, self.arc_len = [], [], []
        for i in range(N):
            arc = I["arcs"][i]
            # entry route: tail S -> gate J -> peel point -> join -> arc
            J, Pp, D = I["far"][i], I["peel"][i], I["D"][i]
            S = J - D * Lb
            ctx = dict(side=I, i=i, N=N)
            con = self.connect(Pp, D, arc[0], I["ts"][i], dict(ctx, entering=True))
            R = Route3(np.vstack([S, J, Pp, con[1:], arc[1:]]))
            Larc = Route3(arc).L
            self.arc_len.append(Larc)
            I.setdefault("route", []).append(R)
            I.setdefault("Larc", []).append(Larc)
            # exit route: arc -> join -> peel point -> gate -> coast
            arc2 = O["arcs"][i]
            J2, Q, D2 = O["far"][i], O["peel"][i], O["D"][i]
            con2 = self.connect(arc2[-1], O["te"][i], Q, D2, dict(side=O, i=i, N=N, entering=False))
            lane_end = J2 + D2 * self.coast_len
            R2 = Route3(np.vstack([arc2, con2[1:], J2, lane_end]))
            L2 = Route3(arc2).L
            Lbase = Route3(np.vstack([arc2, con2[1:], J2])).L
            O.setdefault("route", []).append(R2)
            O.setdefault("Lbase", []).append(Lbase)
            O.setdefault("Larc", []).append(L2)
        # kinematics
        form, ex = tm.form, tm.exit
        Tin = form * (0.50 + 0.32 * I["frac"])
        Dh = np.array([I["route"][i].L - Lb for i in range(N)])
        V_in = float(self.params.get("speed") or 1.35 * np.mean(Dh / Tin))
        lag = ex * (0.05 + 0.30 * (1.0 - O["frac"]))
        Tout = ex - lag
        Do = np.array([O["Lbase"][i] - O["Larc"][i] for i in range(N)])
        V_out = float(self.params.get("speed") or 1.35 * np.mean(Do / np.maximum(Tout, 1e-3)))
        self.V_in, self.V_out = V_in, V_out
        self.Tin, self.lag, self.Tout = Tin, lag, Tout
        for i in range(N):
            R = I["route"][i]
            self.in_slide.append(Slide(R, Lb, R.L, 0.0, R.L - I["Larc"][i], Tin[i], V_in, 0.0))
            R2 = O["route"][i]
            self.out_slide.append(Slide(R2, O["Larc"][i], O["Lbase"][i], 0.0, O["Lbase"][i] - Lb, Tout[i], 0.0, V_out,
                                        delay=lag[i]))
            self.out_slide[-1].coast = True

    # ------------------------------------------------------------ bundles
    def _bundles(self):
        N, Lb, K = self.N, self.Lb, 8
        I, O = self._in, self._out
        s = np.linspace(0, 1, K)[None, :, None]
        Pin = I["far"][:, None, :] - I["D"][:, None, :] * (Lb * s)
        vin = [min(self.V_in * self.Tin[i], 2.9 * (I["route"][i].L - Lb)) / self.Tin[i] for i in range(N)]
        rim = self._rims()
        self._entry = Bundle(Pin, I["D"], self.cols, self.w, 0.0, vin, self.TUBE, rim)
        Pout = O["far"][:, None, :] - O["D"][:, None, :] * (Lb * s)
        vout = [min(self.V_out * self.Tout[i], 2.9 * (O["Lbase"][i] - O["Larc"][i])) / self.Tout[i] for i in range(N)]
        self._exit = Bundle(Pout, O["D"], self.cols, self.w, self.duration, vout, self.TUBE, rim)

    def _rims(self):
        return self.cols * 0.3 if self.TUBE else None

    def entry_bundle(self):
        return self._entry

    def exit_bundle(self):
        return self._exit

    # ------------------------------------------------------------ frame
    def windows(self, t):
        """per strand polyline (3D, tail -> head) at time t."""
        tm = self.timing
        out = []
        if t < tm.form + tm.hold:
            for sl in self.in_slide:
                out.append(sl.window(t)[0])
        else:
            te = t - tm.form - tm.hold
            for sl in self.out_slide:
                out.append(sl.window(te)[0])
        return out

    def fill_alpha(self, t):
        tm = self.timing
        if t < tm.form:
            a = _ss((t - 0.70 * tm.form) / (0.28 * tm.form))
        elif t < tm.form + tm.hold:
            a = 1.0
        else:
            a = 1.0 - _ss((t - tm.form - tm.hold) / max(0.22 * tm.exit, 1e-6))
        return a

    def fill_layers(self, t):
        a = self.fill_alpha(t) * (0.55 if self.FILL == "panel" else 1.0)
        if a <= 0.004:
            return []
        col = self.palette.fg if self.FILL == "solid" else self.palette.panel
        pl = self.placement
        return [Layer(l["cons"], col, "fill", plane=pl.plane, alpha=a) for l in self.lay["letters"]]

    def draw(self, wins, t):
        return strand_layers(wins, self.cols, [self.w] * self.N, self.TUBE, self._rims(),
                             hi_off=np.array([-0.8, -0.8, 0.0]))

    def layers(self, t):
        wins = self.windows(t)
        return self.fill_layers(t) + self.draw(wins, t)

    def focus(self, t):
        wins = [w for w in self.windows(min(max(t, 0.0), self.duration)) if w is not None]
        if not wins:
            return self.placement.origin.copy()
        return np.mean([w.mean(0) for w in wins], 0)
