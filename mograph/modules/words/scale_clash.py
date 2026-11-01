"""scale_clash: scale clash on one shared oblique axis. The first line is HUGE, the next lines (or a tiny copy of the word when
there is only one line) sit small in the corner; every letter throws a graduated grey-to-ground solid extrusion along the SAME
axis, so the huge word's walls run off across the frame while the tiny word's are a long thin wedge. Forming: the letters arrive
along the entry direction as comets (a face with a hairline streak), land, and the streak swings round to the shared axis and
grows into the extrusion (the huge word first, the tiny word a beat later). Leaving: the extrusions pull back into streaks and
the letters run out along the exit direction. (mechanism: 11_mathtype b1 'scale clash + shading', rewritten for the Bundle
interface)

Params: axis (shared extrusion axis: Port name / angle in degrees / 3-vector; default up-right 30 degrees), depth (lean away
from the viewer, default 0.15), reach (huge-word extrusion length in word widths, default 1.4)."""
import math
import numpy as np
from mograph.modules.core import register, Bundle, Layer, Port, unit, mixc, clamp, layout_word
from mograph.modules.words._slab import (SlabWord, LetterFlow, PrismItem, prism_point, grad_prisms, ssm, eo, ei, eio,
                                         simplify, contour_sign, corner_points)


@register("scale_clash")
class ScaleClash(SlabWord):
    description = ("A huge word and a tiny word share one oblique axis: letters arrive as comets, land, and their streaks "
                   "swing round and grow into graduated grey-to-ground extrusions, the huge word's running off across the "
                   "frame, the tiny word's a long thin wedge. It leaves by pulling the extrusions back into streaks.")
    DEMO_WORDS = "HERE\nTHERE"
    DEFAULT_FONT = ("bigsh", {"wght": 900})
    P_IN, P_OUT = 2.4, 2.2

    def build(self):
        pl, P = self.placement, self.params
        lines = list(self.lines)
        if len(lines) == 1:
            lines = [lines[0], lines[0]]
        self.wlines = lines
        letters = []
        w, h = pl.w, pl.h
        self.cap = 0.0
        for i, ln in enumerate(lines):
            if i == 0:
                bw, bh, cx, cy = 0.60 * w, 0.50 * h, -0.5 * w * 0.94 + 0.30 * w, 0.20 * h
            else:
                k = i - 1
                sc = 0.82 ** k
                bw, bh = 0.17 * w * sc, 0.10 * h * sc
                cx, cy = 0.5 * w * 0.94 - 0.5 * bw, 0.42 * h - 0.5 * bh - k * 0.13 * h
            lay = layout_word(self.font, [ln], bw, bh)
            if i == 0:
                self.cap = lay["cap"]
            xs = np.concatenate([np.concatenate(l["cons"]) for l in lay["letters"]])
            wid = float(np.ptp(xs[:, 0]))
            for l in lay["letters"]:
                letters.append(dict(ch=l["ch"], cons=[c + np.array([cx, cy]) for c in l["cons"]], word=i, wid=wid))
        self.letters = letters
        self.cons = [c for l in letters for c in l["cons"]]
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.n3 = pl.n
        self.reach_in = self._reach(self.d_in)
        self.reach_out = self._reach(self.d_out)
        self.setup()

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 4)))
        ext = max(pl.w, pl.h)
        self.Lb0 = float(P.get("tail_len") or 0.22 * ext)
        ax = P.get("axis")
        a = (0.87 * pl.u - 0.5 * pl.v) if ax is None else Port.coerce(ax, "up").side(pl)
        self.ax_rest = unit(unit(a) + float(P.get("depth", 0.15)) * pl.n)
        reach = float(P.get("reach", 1.4))
        n = len(self.letters)
        self.n = n
        self.Dlen = np.array([(reach if l["word"] == 0 else 2.0 * reach / 1.4) * l["wid"] for l in self.letters])
        self.Lb = np.array([self.Lb0 if l["word"] == 0 else 0.45 * self.Lb0 for l in self.letters])
        self.cons_s = [[simplify(c, 0.45 if l["word"] else 0.55) for c in l["cons"]] for l in self.letters]
        self.sg = [contour_sign(l["cons"]) for l in self.letters]
        self.edge_pts = [corner_points(cs, kmax=4 if l["word"] == 0 else 3) for cs, l in zip(self.cons_s, self.letters)]
        self.N = int(sum(len(e) for e in self.edge_pts))
        pts = [np.concatenate([pl.lift(c) for c in l["cons"]]) for l in self.letters]
        self.flow = LetterFlow(self, pts, float(self.Lb.max()), rng, p_in=self.P_IN, p_out=self.P_OUT)
        self.near = mixc(self.palette.fg, self.palette.bg, 0.16)
        self.far = mixc(self.palette.fg, self.palette.bg, 0.94)
        self.nsl = int(P.get("slabs", 26))

    # ------------------------------------------------------------ state
    def state(self, j, t):
        tm, fl = self.timing, self.flow
        word = self.letters[j]["word"]
        off, v, tau, entering = fl.at(j, t)
        if entering:
            Ti = fl.Ti[j]
            w = ssm(t, 0.65 * Ti, Ti + 0.25)
            g0, g1 = (0.30, 0.95) if word == 0 else (0.50, 1.0)
            g = ssm(t, g0 * tm.form, g1 * tm.form) if t < tm.form else 1.0
            s = ssm(t, 0.0, 0.55 * Ti)
            trail = -self.d_in * self.Lb[j]
        else:
            tau_ = max(tau, 0.0)
            w = 1.0 - ssm(tau_, 0.30, 0.80)
            g = 1.0 - ssm(tau_, 0.0, 0.60) if word == 0 else 1.0 - ssm(tau_, 0.05, 0.50)
            s = 1.0 - ssm(tau_, 0.70, 1.0)
            trail = -self.d_out * self.Lb[j]
        d3 = (1 - w) * trail + w * self.ax_rest * self.Dlen[j] * eio(g, 2.2)
        return dict(off=off, v=v, w=w, d3=d3, s=s, entering=entering)

    # ------------------------------------------------------------ bundle / frames
    def bundle_at(self, t):
        pl = self.placement
        K = 8
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        v = np.zeros(self.N)
        us = np.linspace(0, 1, K)
        i = 0
        for j in range(self.n):
            st = self.state(j, t)
            it = PrismItem([], 1.0, st["off"], st["d3"], np.zeros(2), 1.0)
            for p in self.edge_pts[j]:
                Pm[i] = np.array([prism_point(pl, it, p[None], u)[0] for u in us])
                Dv[i] = self.d_in if st["entering"] else self.d_out
                v[i] = st["v"]
                i += 1
        col = np.tile(np.array(self.palette.fg, float), (self.N, 1))
        return Bundle(Pm, Dv, col, 1.2, t, v, 0.0)

    def layers(self, t):
        pl, pal = self.placement, self.palette
        items, faces, lines = [], [], {}
        for j in range(self.n):
            st = self.state(j, t)
            it = PrismItem(self.cons_s[j], self.sg[j], st["off"], st["d3"], np.zeros(2), 1.0, st["s"])
            items.append(it)
            faces.append(([pl.lift(c) + st["off"] for c in self.letters[j]["cons"]], st["s"]))
            if st["w"] < 0.98:
                for p in self.edge_pts[j]:
                    lines.setdefault(round(1.0 - st["w"], 2), []).append(
                        np.array([prism_point(pl, it, p[None], u)[0] for u in np.linspace(0, 1, 4)]))
        L = grad_prisms(pl, items, self.nsl, self.near, self.far, pal.bg, gamma=0.55, bias0=3.0)
        for k, pols in lines.items():
            L.append(Layer(pols, pal.fg, "line", 1.2 * k, closed=False, bias=5.0))
        for polys, s in faces:
            if s > 0.004:
                L.append(Layer(polys, mixc(pal.bg, pal.fg, s), "fill", bias=6.0))
        return L
