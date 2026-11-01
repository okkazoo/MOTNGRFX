"""circuit: strands arrive down the entry lanes as PCB traces (straight runs with 45 degree chamfered bends, a bright pulse
riding each head) and turn into the outline of the type, which they draw as amber wire; the wire grows a hidden-line depth
box (cabinet projection) and then inverts into a solid: bone caps on shaded amber walls. To leave, the solid melts back to
wire and every edge detaches and runs off as a trace down the exit lanes, a pulse on each head.
(mechanism: 11_mathtype a6 CIRCUIT shading inversion + 13_camera v005 module_journey SIGNAL, rewritten for the Bundle
interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, unit, ol, strand_layers, mixc
from mograph.modules.words._strand import StrandWord, manhattan, _ss


@register("circuit")
class Circuit(StrandWord):
    description = ("PCB-style traces with 45 degree bends and a pulse on each head arrive along the entry lanes and draw the "
                   "outline as amber wire; the wire grows a depth box and inverts into a solid word. To leave, the solid "
                   "melts to wire and every edge runs off down the exit lanes as a trace.")
    DEMO_WORDS = "SIGNAL"
    DEFAULT_FONT = "jetbrainsmono"
    TUBE = 0.0
    FILL = "circuit"
    ARC_PER_CAP = 0.5
    LINE_W = 1.1

    def connect(self, p0, t0, p1, t1, ctx):
        side, i, N = ctx["side"], ctx["i"], ctx["N"]
        d, e1, e2 = side["d"], side["e1"], side["e2"]
        shift = float((p1 - p0) @ e1)
        f = self._jog(side["rank"][i], N, shift)
        ch = (4.0 + 0.25 * abs(side["rank"][i] - N / 2.0)) * max(0.8, self.cap / 200.0)
        return manhattan(p0, p1, d, e1, e2, f, ch)

    # ---- colours
    def _colours(self, N):
        return np.tile(np.array(self.tones()[0], float), (N, 1))

    def tones(self):
        pal = self.palette
        amber = pal.accents[0]
        wire = amber
        trace = mixc(amber, pal.bg, 0.22)
        pulse = mixc(amber, (255, 255, 255), 0.7)
        walls = [mixc(amber, pal.bg, r) for r in (0.50, 0.70, 0.86)]
        return trace, wire, pulse, walls

    # ---- the depth box
    def depth_vec(self):
        pl, cap = self.placement, self.cap
        return (0.11 * cap) * pl.u + (0.10 * cap) * pl.v + (0.31 * cap) * pl.n

    def solid_amounts(self, t):
        tm = self.timing
        f, h, ex = tm.form, tm.hold, tm.exit
        if t < f + h:
            wb = _ss((t - 0.58 * f) / (0.30 * f))
            sol = _ss((t - 0.84 * f) / (0.16 * f))
        else:
            te = t - f - h
            sol = 1.0 - _ss(te / (0.22 * ex))
            wb = 1.0 - _ss((te - 0.12 * ex) / (0.22 * ex))
        return wb, sol

    def fill_layers(self, t):
        wb, sol = self.solid_amounts(t)
        pl, E = self.placement, self.depth_vec()
        trace, wire, pulse, walls = self.tones()
        out = []
        if wb > 0.003:
            e2d = np.array([E @ pl.u, E @ pl.v])
            e2d = e2d / (np.linalg.norm(e2d) + 1e-9)
            edges, backs = [], []
            for c2 in self._cons2():
                c = pl.lift(c2)
                backs.append(c + E)
                for i in ol.feature_idx(c2, e2d, 35):
                    edges.append(np.array([c[i], c[i] + E]))
            out.append(Layer(backs, wire, "line", 1.0, closed=True, alpha=wb * 0.9, bias=0.0))
            if edges:
                out.append(Layer(edges, wire, "line", 1.0, closed=False, alpha=wb * 0.9, bias=0.0))
        if sol > 0.003:
            lt = np.array([-0.6, -0.8])
            lt /= np.linalg.norm(lt)
            quads = [[], [], []]
            caps = []
            for c2 in self._cons2():
                c = pl.lift(c2)
                tv = np.roll(c2, -1, 0) - c2
                nn = np.stack([tv[:, 1], -tv[:, 0]], 1)
                nn /= (np.linalg.norm(nn, axis=1)[:, None] + 1e-9)
                lv = np.clip((np.abs(nn @ lt) * 3).astype(int), 0, 2)
                c1 = np.roll(c, -1, 0)
                for i in range(len(c)):
                    quads[lv[i]].append(np.array([c[i], c1[i], c1[i] + E, c[i] + E]))
                caps.append(c)
            for lv in range(3):
                if quads[lv]:
                    out.append(Layer(quads[lv], walls[lv], "fill", split=True, alpha=sol, bias=0.0))
            out.append(Layer(caps, self.palette.fg, "fill", alpha=sol, bias=0.2))
        return out

    def _cons2(self):
        return [c for l in self.lay["letters"] for c in l["cons"]]

    # ---- strands: trace on the lanes, wire on the arc, a pulse on each moving head
    def layers(self, t):
        tm = self.timing
        trace_c, wire_c, pulse_c, _ = self.tones()
        I, O = self._in, self._out
        tr, wi, pu = [], [], []
        entering = t < tm.form + tm.hold
        if entering:
            sls, side = self.in_slide, I
        else:
            sls, side = self.out_slide, O
        tt = t if entering else t - tm.form - tm.hold
        for i, sl in enumerate(sls):
            poly, h, tl = sl.window(tt)
            if poly is None:
                continue
            R = sl.R
            if entering:
                split = R.L - side["Larc"][i]
                pieces = [(tl, min(h, split), tr), (max(tl, split), h, wi)]
                moving = h < R.L - 0.5 and h > sl.h0 + 0.5
            else:
                split = side["Larc"][i]
                pieces = [(tl, min(h, split), wi), (max(tl, split), h, tr)]
                moving = h > side["Larc"][i] + 0.5 and h < side["Lbase"][i] - 0.5
            for a, b, dst in pieces:
                if b - a > 0.5:
                    w = R.window(a, b)
                    if w is not None:
                        dst.append(w)
            if moving:
                pw = R.window(max(h - 16.0, 0.0), h)
                if pw is not None:
                    pu.append(pw)
        out = self.fill_layers(t)
        if tr:
            out.append(Layer(tr, trace_c, "line", self.w, closed=False, bias=0.3))
        if wi:
            out.append(Layer(wi, wire_c, "line", 1.35, closed=False, bias=0.5))
        if pu:
            out.append(Layer(pu, pulse_c, "line", 3.2, closed=False, bias=1.0))
        return out
