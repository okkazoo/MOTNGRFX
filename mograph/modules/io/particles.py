"""particles: the word assembles from, and dissolves into, small marks (+ - | and dots).
Reveal: marks stream in along the entry lane as speed streaks, land on the outline of the type and settle into their marks
(a dash along a horizontal edge, a bar along a vertical one, a plus, a dot) while more marks condense on the outline around
them; then the marks shrink away as the body fades in over them. Conceal: the body fades to its marks, the marks pick up speed
and run out along the exit lane (the ones left behind dissolve). Material: `points` (a transition sees each as a short dash
strand). Options: pitch (px between marks on the outline, auto), carriers (how many marks travel through the gate, auto =
the module's strand count in 24..48), mark (mark half-length px, auto), accent (share of marks in the first accent colour,
0.14), seed (1)."""
import math
import numpy as np

from mograph.modules.core import Layer
from mograph.modules.material import Material, MARK_DOT, MARK_DASH, MARK_BAR, MARK_PLUS
from mograph.modules.io.base import (IOStyle, register_style, side_frame, stage_ribbon, Flight, hash01, ss, consolidate, fade,
                                     fit_entering, fit_leaving)


@register_style("particles")
class Particles(IOStyle):
    description = ("The word assembles out of small marks: + - | and dots stream in as speed streaks, land on the outline and "
                   "settle, then give way to the solid type. On exit the type dissolves into marks that run out along the "
                   "exit direction.")
    material_kind = "points"

    def attach(self, body, side):
        super().attach(body, side)
        self._prep = None

    # ------------------------------------------------------------ preparation
    def prep(self):
        if self._prep is not None:
            return self._prep
        b, pl, tm = self.body, self.body.placement, self.body.timing
        side = self.side
        d, e1, e2, gate = side_frame(b, side)
        sh = b.hold_shape()
        cap = sh.cap
        seed = int(self.opt("seed", 1))
        pitch = float(self.opt("pitch", 0) or max(sh.perimeter() / 300.0, 0.045 * cap))
        P2, T2, let = sh.outline(pitch)
        M = len(P2)
        M3 = pl.lift(P2)
        ar = np.arange(M)
        r = hash01(seed, ar)
        hor, ver = np.abs(T2[:, 0]) > 0.8, np.abs(T2[:, 1]) > 0.8
        mark = np.where(hor, np.where(r < 0.6, MARK_DASH, np.where(r < 0.85, MARK_DOT, MARK_PLUS)),
                        np.where(ver, np.where(r < 0.6, MARK_BAR, np.where(r < 0.85, MARK_DOT, MARK_PLUS)),
                                 np.where(r < 0.5, MARK_DOT, np.where(r < 0.78, MARK_PLUS, np.where(r < 0.9, MARK_DASH, MARK_BAR)))))
        ms = float(self.opt("mark", 0) or np.clip(0.7 * pitch, 3.4, 11.0))
        wm = max(1.6, 1.9 * max(0.8, cap / 200.0) * 0.8)
        col = np.tile(np.array(b.palette.fg, float), (M, 1))
        acc = hash01(seed + 3, ar) < float(self.opt("accent", 0.14))
        col[acc] = b.palette.accents[0]
        raw = b._rawcall("entry_bundle" if side == "in" else "exit_bundle")
        nc = int(self.opt("carriers", 0) or np.clip(raw.N, 24, 48))
        rel = M3 - pl.origin
        lat, along = rel @ e1, rel @ d
        order = np.argsort(lat, kind="stable")
        car = np.sort(order[np.unique(np.round(np.linspace(0, M - 1, min(nc, M))).astype(int))])
        nc = len(car)
        is_car = np.zeros(M, bool)
        is_car[car] = True
        frac = (along - along.min()) / max(along.max() - along.min(), 1e-6)
        spacing = max(4.0, 2.4 * wm)
        st = stage_ribbon(b, side, lat[car], spacing, row_pitch=2.5 * spacing)
        v = max(float(raw.v.mean()), 150.0)
        Ls0 = float(np.clip(0.06 * cap, 12.0, 26.0))
        T = b.style_window(side)[1]
        e1_ = e1
        wob = (hash01(seed + 5, ar) - 0.5) * 0.16
        flights = []
        if side == "in":
            dl = 0.10 * hash01(seed + 6, np.arange(nc))
            arr = 0.30 + 0.30 * frac[car] + 0.05 * hash01(seed + 7, np.arange(nc))
            for jj, i in enumerate(car):
                dl[jj], fl = fit_entering(st["S"][jj], M3[i], d, v, T, dl[jj], arr[jj], e1_, wob[i])
                flights.append(fl)
            self.dl, self.arr = dl, arr
            self.spawn = 0.28 + 0.50 * hash01(seed + 8, ar)
        else:
            dep_p = 0.34 + 0.30 * (1.0 - frac[car]) + 0.04 * hash01(seed + 7, np.arange(nc))
            for jj, i in enumerate(car):
                dep_p[jj], fl = fit_leaving(M3[i], st["S"][jj], d, v, T, dep_p[jj], e1_, wob[i])
                flights.append(fl)
            self.dep_p = dep_p
            self.vanish = 0.30 + 0.45 * hash01(seed + 8, ar)
        self._prep = dict(M=M, M3=M3, mark=mark, ms=ms, wm=wm, col=col, car=car, nc=nc, is_car=is_car, st=st, flights=flights,
                          v=v, ws=max(1.3, 0.62 * wm), Ls0=Ls0, k=Ls0 / v, d=d, T=T, u=pl.u, vv=pl.v, jj_of={int(i): jj for jj, i in enumerate(car)})
        return self._prep

    # ------------------------------------------------------------ marks
    def mark_items(self, i, pos, sc):
        P = self.prep()
        if sc <= 0.02:
            return []
        c = tuple(int(x) for x in P["col"][i])
        s, wm = P["ms"] * sc, P["wm"]
        u, v = P["u"], P["vv"]
        k = P["mark"][i]
        key = lambda w: ("line", c, round(w * 2) / 2, 1.0, False, False, 0.25)
        if k == MARK_DASH:
            return [(key(wm), [np.array([pos - u * s, pos + u * s])])]
        if k == MARK_BAR:
            return [(key(wm), [np.array([pos - v * s, pos + v * s])])]
        if k == MARK_PLUS:
            return [(key(wm), [np.array([pos - u * s, pos + u * s]), np.array([pos - v * s, pos + v * s])])]
        return [(key(2.0 * wm * min(1.0, sc * 1.3)), [np.array([pos - u * 0.35, pos + u * 0.35])])]

    def streak(self, jj, pos, tang, speed):
        P = self.prep()
        i = int(P["car"][jj])
        c = tuple(int(x) for x in P["col"][i])
        L = max(P["k"] * speed, 0.8)
        return (("line", c, round(P["ws"] * 2) / 2, 1.0, False, False, 0.25), [np.array([pos - tang * L, pos])])

    # ------------------------------------------------------------ frame
    def layers(self, t, p):
        return self.reveal(t, p) if self.side == "in" else self.conceal(t, p)

    def reveal(self, t, p):
        P = self.prep()
        T, d = P["T"], P["d"]
        p = max(p, 0.0)
        items = []
        fm = 1.0 - ss((p - 0.84) / 0.14)                         # marks shrink away as the body arrives
        for i in range(P["M"]):
            jj = P["jj_of"].get(i)
            if jj is None:
                sc = ss((p - self.spawn[i]) / 0.14) * fm
                items += self.mark_items(i, P["M3"][i], sc)
            else:
                fl = P["flights"][jj]
                if p < self.arr[jj]:
                    if p < self.dl[jj]:
                        pos, tang, sp = P["st"]["S"][jj] + d * P["v"] * p * T, d, P["v"]
                    else:
                        u = (p - self.dl[jj]) * T / fl.T
                        pos, tang = fl.at(u)
                        sp = fl.speed(u)
                    items.append(self.streak(jj, pos, tang, sp))
                else:
                    items += self.mark_items(i, P["M3"][i], ss((p - self.arr[jj]) / 0.10) * fm)
        out = consolidate(items)
        h = ss((p - 0.80) / 0.14)
        if h > 0.0:
            out += fade(self.body.reveal_layers(0.0, t), h)
        return out

    def conceal(self, t, p):
        P = self.prep()
        items = []
        appear = ss((p - 0.04) / 0.14)
        for i in range(P["M"]):
            jj = P["jj_of"].get(i)
            if jj is None:
                sc = appear * (1.0 - ss((p - self.vanish[i]) / 0.14))
                items += self.mark_items(i, P["M3"][i], sc)
            else:
                fl = P["flights"][jj]
                if p < self.dep_p[jj]:
                    items += self.mark_items(i, P["M3"][i], appear)
                else:
                    u = (p - self.dep_p[jj]) * P["T"] / fl.T
                    pos, tang = fl.at(u)
                    items.append(self.streak(jj, pos, tang, fl.speed(u) if u < 1.0 else fl.v))
        out = consolidate(items)
        h = 1.0 - ss(p / 0.16)
        if h > 0.0:
            out = fade(self.body.conceal_layers(1.0, t), h) + out
        return out

    def focus(self, t, p):
        P = self.prep()
        if self.side == "in" and p < 0.6:
            T = P["T"]
            pts = []
            for jj, fl in enumerate(P["flights"]):
                if p < self.dl[jj]:
                    pts.append(P["st"]["S"][jj] + P["d"] * P["v"] * max(p, 0) * T)
                else:
                    pts.append(fl.at(min(1.0, (p - self.dl[jj]) * T / fl.T))[0])
            return np.mean(pts, 0)
        return self.body.placement.origin.copy()

    # ------------------------------------------------------------ material
    def material(self):
        P = self.prep()
        nc = P["nc"]
        S = P["st"]["S"] if self.side == "in" else np.array([fl.at(1.0)[0] for fl in P["flights"]])
        col = P["col"][P["car"]]
        return Material.points(S, np.tile(P["d"], (nc, 1)), col, P["ws"], P["v"], MARK_DASH, length=P["Ls0"],
                               t=0.0 if self.side == "in" else self.body.duration)
