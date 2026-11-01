"""solid_hollow: super-bold against hairline. Lines alternate: a SOLID line (heavy face) and a HOLLOW line (hairline face, x-ray
wire only). Every letter forms as a wire prism: its corner edges arrive as hairline strands along the entry direction, the
letter lands, a wire outline rises off it along the extrusion axis (front and back outline joined by depth edges), and on the
solid lines the walls then FILL the wire (banded, flat colour) and the cap fills solid; both sink back to a shallow slab.
Leaving, the solid rises again, hollows out to wire, and the wire recedes into hairline strands that run out along the exit
direction. (mechanism: 11_mathtype a9R / 13_camera strips Solid, rewritten for the Bundle interface)

Params: hollow_font (default bungeehairline), kinds (per line 'solid'|'hollow'), axis (rest extrusion axis: Port name / angle /
3-vector; default up-left, leaning away from the viewer), depth (lean away from the viewer, default 0.3), band (wall colour)."""
import math
import numpy as np
from mograph.modules.core import register, Bundle, Layer, Port, unit, mixc, clamp, layout_word, VERM
from mograph.modules.words._slab import (SlabWord, LetterFlow, ssm, eo, ei, eio, simplify, contour_sign, wall_quads,
                                         wall_layers, corner_points)


@register("solid_hollow")
class SolidHollow(SlabWord):
    description = ("Each letter forms as a wire prism: hairline edges arrive, the letter lands, a wire outline rises off it "
                   "and the walls fill in solid while the cap goes super-bold; a hairline line sits against it as x-ray "
                   "wire. It leaves the same way in reverse, hollowing out to wire and receding into hairline strands.")
    DEMO_WORDS = "SOLID\nHOLLOW"
    DEFAULT_FONT = "passion"
    P_IN, P_OUT = 2.4, 2.2

    def build(self):
        # fonts differ per line, so the layout is done here
        pl, P = self.placement, self.params
        n = len(self.lines)
        self.kinds = list(P.get("kinds") or ["solid" if i % 2 == 0 else "hollow" for i in range(n)])
        hf = P.get("hollow_font", "bungeehairline")
        wts = np.array([1.0 if k == "solid" else 0.55 for k in self.kinds])
        H = pl.h * 0.92
        gap = 0.07 * H if n > 1 else 0.0
        hs = (H - gap * (n - 1)) * wts / wts.sum()
        y = -0.5 * H
        letters = []
        self.cap = 0.0
        for i, ln in enumerate(self.lines):
            font = self.font if self.kinds[i] == "solid" else hf
            lay = layout_word(font, [ln], pl.w * 0.94, hs[i])
            self.cap = max(self.cap, lay["cap"])
            yc = y + hs[i] / 2
            for l in lay["letters"]:
                letters.append(dict(ch=l["ch"], cons=[c + np.array([0.0, yc]) for c in l["cons"]], kind=self.kinds[i], line=i))
            y += hs[i] + gap
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
        rng = np.random.RandomState(int(P.get("seed", 5)))
        ext = max(pl.w, pl.h)
        self.Lb = float(P.get("tail_len") or 0.2 * ext)
        self.DMAX = {"solid": 0.34 * ext, "hollow": 0.28 * ext}
        self.DEND = {"solid": 0.09 * ext, "hollow": 0.045 * ext}
        self.lw = 1.1
        tz = float(P.get("depth", 0.3))
        ax = P.get("axis")
        a = (-0.87 * pl.u - 0.5 * pl.v) if ax is None else Port.coerce(ax, "up").side(pl)
        self.ax_rest = unit(unit(a) + tz * pl.n)
        self.band = tuple(P.get("band") or VERM)
        self.ground = self.palette.bg
        n = len(self.letters)
        self.n = n
        self.cons_s = [[simplify(c, 0.5) for c in l["cons"]] for l in self.letters]
        self.sg = [contour_sign(l["cons"]) for l in self.letters]
        self.edge_pts = [corner_points(cs, kmax=8) for cs in self.cons_s]
        self.N = int(sum(len(e) for e in self.edge_pts))
        pts = [np.concatenate([pl.lift(c) for c in l["cons"]]) for l in self.letters]
        self.flow = LetterFlow(self, pts, self.Lb, rng, land=(0.22, 0.14), p_in=self.P_IN, p_out=self.P_OUT)

    # ------------------------------------------------------------ state
    def state(self, j, t):
        tm, fl = self.timing, self.flow
        kind = self.letters[j]["kind"]
        DM, DE = self.DMAX[kind], self.DEND[kind]
        off, v, tau, entering = fl.at(j, t)
        if entering:
            Ti = fl.Ti[j]
            w = ssm(t, 0.7 * Ti, Ti + 0.2)
            ax = unit((1 - w) * (-self.d_in) + w * self.ax_rest)
            r = eo(clamp((t - Ti * 0.8) / 0.45), 2.0)
            dw = self.Lb + (DM - self.Lb) * r
            ds = DM * eo(clamp((t - Ti * 0.8 - 0.30) / 0.46), 2.0) if kind == "solid" else 0.0
            sink = 1.0 if t >= tm.form else eio(clamp((t / tm.form - 0.80) / 0.17), 2.0)
            dw = dw + (DE - dw) * sink
            if kind == "solid":
                ds = ds + (DE - ds) * sink
            ow = ssm(t, 0.6 * Ti, Ti)
        else:
            te = t - tm.form - tm.hold
            tau_ = max(tau, 0.0)
            w = 1.0 - ssm(tau_, 0.4, 0.8)
            ax = unit((1 - w) * (-self.d_out) + w * self.ax_rest)
            up = eio(clamp(tau_ / 0.35), 2.0)
            dw_up = DE + (DM - DE) * up
            rec = eio(clamp((tau_ - 0.55) / 0.45), 2.0)
            dw = dw_up + (self.Lb - dw_up) * rec
            if kind == "solid":
                ds = (DE + (DM - DE) * up) * (1 - eio(clamp((tau_ - 0.30) / 0.25), 2.0))
            else:
                ds = 0.0
            ow = 1.0 - ssm(tau_, 0.75, 1.0)
        return dict(off=off, v=v, ax=ax, dw=dw, ds=ds, ow=ow, kind=kind)

    # ------------------------------------------------------------ bundle / frames
    def bundle_at(self, t):
        pl = self.placement
        K = 8
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        v = np.zeros(self.N)
        f = np.linspace(0, 1, K)[:, None]
        entry = t < self.timing.form + self.timing.hold
        i = 0
        for j in range(self.n):
            st = self.state(j, t)
            for p in self.edge_pts[j]:
                head = pl.lift(p[None])[0] + st["off"]
                Pm[i] = head + st["ax"] * st["dw"] * f
                Dv[i] = self.d_in if entry else self.d_out
                v[i] = st["v"]
                i += 1
        col = np.tile(np.array(self.palette.fg, float), (self.N, 1))
        return Bundle(Pm, Dv, col, self.lw, t, v, 0.0)

    def layers(self, t):
        pl, pal = self.placement, self.palette
        L = []
        walls, caps, edges, cap_out, back_out = [], [], [], [], []
        ow_groups = {}
        for j in range(self.n):
            st = self.state(j, t)
            off, ax = st["off"], st["ax"]
            dw, ds = st["dw"], st["ds"]
            letter = self.letters[j]
            if ds > 0.5:
                walls.append(wall_quads(pl, self.cons_s[j], self.sg[j], off, ax, ds, ds + 1.0, band=True))
            fa = ssm(ds, 0.0, 14.0)
            if fa > 0.003:
                caps.append(([pl.lift(c) + off for c in letter["cons"]], fa))
            for p in self.edge_pts[j]:
                h = pl.lift(p[None])[0] + off
                edges.append(np.array([h, h + ax * dw]))
            if st["ow"] > 0.02:
                g = ow_groups.setdefault(round(st["ow"], 2), ([], []))
                for c in self.cons_s[j]:
                    P0 = pl.lift(c) + off
                    g[0].append(P0)
                    if dw > 0.6:
                        g[1].append(P0 + ax * dw)
        for k, (front, back) in ow_groups.items():
            if front:
                L.append(Layer(front, pal.fg, "line", self.lw * k, closed=True, bias=6.0))
            if back:
                L.append(Layer(back, pal.fg, "line", self.lw * k, closed=True, bias=6.0))
        if walls:
            L += wall_layers(walls, 1e9, 1.0, self.band, self.ground, bias0=3.0, level_cols=(self.band, self.ground))
        for polys, a in caps:
            L.append(Layer(polys, pal.fg, "fill", alpha=a if a < 0.998 else 1.0, bias=5.0))
        if edges:
            L.append(Layer(edges, pal.fg, "line", self.lw, closed=False, bias=6.5))
        return L
