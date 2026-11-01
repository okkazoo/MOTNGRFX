"""Shared engine of the stacked-contour strand words (spectrum, tangent_spray): the word is a stack of NL copies of its outline
receding behind it along an oblique (each copy at its own depth). Every copy is cut into arcs; every arc is one strand. Strands
arrive down the entry lanes, run in along the outline's own (exact or iso-snapped) tangent and draw their arc; copies arrive
back-first, the stack fills (front copy solid, back copies ground-coloured so they hide each other) and stands. To leave, the
fill melts and the arcs run off along their own tangents into the exit lanes, back copies first.
Built on StrandWord (_strand.py): build() is a copy of its build() with the stack added; everything else (lanes, routes,
bundles, windows) is inherited."""
import math
import numpy as np
from mograph.modules.core import (Palette, Layer, layout_word, unit, ol, strand_layers, bezier, mixc, Route3)
from mograph.modules.words._strand import StrandWord, make_arcs, _ss

ISO6_2D = np.array([[0, -1.0], [0, 1.0], [math.cos(math.radians(30)), -0.5], [-math.cos(math.radians(30)), 0.5],
                    [-math.cos(math.radians(30)), -0.5], [math.cos(math.radians(30)), 0.5]])


def ramp(cols, u):
    u = min(1.0, max(0.0, u)) * (len(cols) - 1)
    i = min(int(u), len(cols) - 2)
    return mixc(cols[i], cols[i + 1], u - i)


class StackWord(StrandWord):
    NL = 8                      # outline copies
    SP = 7.0                    # px between copies (in the placement plane)
    DIR2 = (0.94, 0.34)         # direction the copies recede in the plane (x right, y down)
    DZ = 4.0                    # px between copies in depth
    SPRAY = 0.0                 # length (px, fraction of cap below 3) of the straight run along the tangent before the lane bend
    SNAP = False                # snap the tangent run to the 6 iso directions
    ARC_PER_CAP = 0.9
    LINE_W = 1.1
    DEFAULT_PALETTE = None
    FILL = "stack"

    def __init__(self, words=None, font=None, placement=None, entry="left", exit="right", timing=None, palette=None, **params):
        if palette is None and self.DEFAULT_PALETTE is not None:
            palette = self.DEFAULT_PALETTE
        super().__init__(words, font, placement, entry, exit, timing, palette, **params)

    # ---- per-layer colours / fills (subclass)
    def layer_col(self, l):
        return self.palette.fg

    def layer_fill(self, l):
        return self.palette.fg if l == 0 else self.palette.bg

    def _colours(self, N):
        return np.array([self.layer_col(l) for l in self.layer_of], float)

    # ---- build (copy of StrandWord.build with the stack)
    def build(self):
        pl, P = self.placement, self.params
        NL = int(P.get("layers", self.NL))
        self.NLd = NL
        step = np.array(self.DIR2, float)
        step = step / np.linalg.norm(step) * float(P.get("spacing", self.SP))
        self.step2 = step
        tot = step * (NL - 1)
        self.lay = layout_word(self.font, self.lines, pl.w * 0.94 - abs(tot[0]), pl.h * 0.9 - abs(tot[1]))
        cap = self.lay["cap"]
        self.cap = cap
        self.shift2 = -0.5 * tot
        cons = [c for l in self.lay["letters"] for c in l["cons"]]
        self.cons2 = cons
        perim = sum(float(ol.arclen(np.vstack([c, c[:1]]))[-1]) for c in cons)
        Nl = int(P.get("n_strands") or 0) // NL or int(np.clip(round(perim / (self.ARC_PER_CAP * cap)), 8, 34))
        arcs2 = make_arcs(cons, Nl)
        dz = float(P.get("depth_step", self.DZ))
        arcs3, layer_of = [], []
        for l in range(NL):
            for a in arcs2:
                arcs3.append(pl.lift(a + self.shift2 + l * step) + l * dz * pl.n)
                layer_of.append(l)
        self.layer_of = np.array(layer_of)
        self.dz = dz
        self.N = N = len(arcs3)
        self.w = float(P.get("width") or (self.LINE_W * max(0.8, cap / 200.0)))
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

    def _side(self, arcs3, d, entering, spread):
        out = super()._side(arcs3, d, entering, spread)
        l = self.layer_of / max(self.NLd - 1, 1)
        if entering:
            out["frac"] = 0.72 * (1.0 - l) + 0.28 * out["frac"]          # back copies arrive first
        else:
            out["frac"] = 0.72 * l + 0.28 * out["frac"]                  # back copies leave first
        return out

    # ---- the run along the outline's own tangent
    def _tangent_dir(self, t):
        t = unit(t)
        if not self.SNAP:
            return t
        pl = self.placement
        a = np.array([t @ pl.u, t @ pl.v])
        s = ISO6_2D[int(np.argmax(ISO6_2D @ a))]
        return unit(s[0] * pl.u + s[1] * pl.v)

    def connect(self, p0, t0, p1, t1, ctx):
        Ls = float(self.params.get("spray", self.SPRAY)) * self.cap
        if Ls < 1.0:
            return super().connect(p0, t0, p1, t1, ctx)
        r = 0.12 * self.cap
        if ctx.get("entering", True):
            T = self._tangent_dir(t1)
            q = p1 - T * Ls
            k = 0.4 * float(np.linalg.norm(q - p0)) + 1e-3
            lead = bezier(p0, p0 + t0 * k, q - T * k, q)
            q2 = p1 - T * r
            tail = bezier(q2, q2 + T * r * 0.5, p1 - t1 * r * 0.5, p1, step=3.0)
            return np.vstack([lead, tail[0:1], tail[1:]])
        T = self._tangent_dir(t0)
        q = p0 + T * Ls
        r1 = p0 + T * r
        head = bezier(p0, p0 + t0 * r * 0.5, r1 - T * r * 0.5, r1, step=3.0)
        k = 0.4 * float(np.linalg.norm(p1 - q)) + 1e-3
        lead = bezier(q, q + T * k, p1 - t1 * k, p1)
        return np.vstack([head, lead[1:]])

    # ---- frame
    def layer_polys(self, l):
        pl = self.placement
        return [pl.lift(c + self.shift2 + l * self.step2) + l * self.dz * pl.n for c in self.cons2]

    def lw(self, l):
        return self.w * (1.35 if l == 0 else 1.0)

    def fill_layers(self, t):
        a = self.fill_alpha(t)
        if a <= 0.004:
            return []
        out = []
        for l in range(self.NLd - 1, -1, -1):
            polys = self.layer_polys(l)
            out.append(Layer(polys, self.layer_fill(l), "fill", alpha=a))
            out.append(Layer(polys, self.layer_col(l), "line", self.lw(l), closed=True, alpha=a, bias=0.4))
        return out

    def draw(self, wins, t):
        fa = self.fill_alpha(t)
        if fa >= 0.995:
            return []
        out = strand_layers(wins, self.cols, [self.lw(l) for l in self.layer_of], 0.0, bias0=2.0)
        if fa > 0.004:
            for L in out:
                L.alpha = 1.0 - fa
        return out
