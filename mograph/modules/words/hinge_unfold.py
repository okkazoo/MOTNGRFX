"""hinge_unfold: domino rhythm on true 3D hinges. Strands draw the outline, then every letter in turn tips back about its own
baseline onto the floor, rises as a solid block, sinks while it swings back upright and pushes its depth out toward the
viewer; Lambert-shaded walls under a bone face. On exit the sequence runs backward and the strands leave.
(mechanism: 11_mathtype c7 hinge; 13_camera layout_tour R2 hinge blocks, rewritten for the Bundle interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc, unit
from mograph.modules.words._depth import MechWord, ss, eio, seg, letter_edges, quant_layers


@register("hinge_unfold")
class HingeUnfold(MechWord):
    description = ("Strands draw the outline, then each letter in turn tips back about its own baseline onto the floor, rises "
                   "as a solid block, swings upright while sinking and pushes its depth out toward the viewer, shaded as "
                   "solid walls under a bone face. On exit the domino runs backward and the strands leave.")
    DEMO_WORDS = "HINGE"
    DEFAULT_FONT = "alienblock"
    DEFAULT_PALETTE = "ink"
    DEFAULT_TIMING = (2.8, 0.9, 2.4)
    HIDE = True
    LIB_CAMERA = dict(yaw=(8.0, -24.0, -20.0), pitch=(10.0, 30.0, 26.0), dist=(1180.0, 1160.0, 1200.0), cy=40.0)

    def mech_build(self):
        pal = self.palette
        self.nl = len(self.lay["letters"])
        self.st = min(0.13, 0.75 / max(self.nl - 1, 1))
        self.S_tot = 2.6 + self.st * (self.nl - 1)
        self.yb_l = self.row_baselines()
        E = letter_edges(self.lay["letters"])
        self.E = E
        self.idx = [np.where(E["L"] == i)[0] for i in range(self.nl)]
        self.cap = self.lay["cap"]
        pl = self.placement
        self.LT = unit(-0.45 * pl.u - 0.75 * pl.v - 0.48 * pl.n)
        self.wr = [mixc(pal.accents[2], pal.bg, 0.78), mixc(pal.accents[2], pal.bg, 0.35), pal.accents[2], pal.accents[0]]

    def ramp(self, u):
        x = min(max(u, 0.0), 1.0) * (len(self.wr) - 1)
        i = min(int(x), len(self.wr) - 2)
        return mixc(self.wr[i], self.wr[i + 1], x - i)

    def letter_state(self, i, s):
        tl = s - i * self.st
        tip = eio(seg(tl, 0.0, 0.7), 2.6)
        rise = eio(seg(tl, 0.35, 1.1), 2.4)
        up = eio(seg(tl, 1.2, 2.0), 2.6)
        sink = eio(seg(tl, 1.1, 1.8), 2.4)
        push = eio(seg(tl, 1.8, 2.6), 2.6)
        th = 0.5 * math.pi * (tip - up)
        h = self.cap * (0.75 * rise * (1 - sink) + 0.12 * sink * rise + 0.5 * push)
        return th, h

    def mech_layers(self, t):
        pal, pl, E = self.palette, self.placement, self.E
        p = self.progress(t)
        s = p * self.S_tot
        eye = self.eye_at(t)
        polys, vals = [], []
        out, caps_top, caps_base = [], [], []
        fa = min(ss((t - self.t_swap) / 0.25), ss((self.t_c - t) / 0.25))
        for i in range(self.nl):
            th, h = self.letter_state(i, s)
            c, sn = math.cos(th), math.sin(th)
            yb = self.yb_l[i]
            dd = c * pl.n + sn * pl.v                         # depth direction behind the face
            e = -h * dd
            ix = self.idx[i]
            if h > 0.5 and len(ix):
                A3, B3 = self.pose(E["A"][ix], yb, th), self.pose(E["B"][ix], yb, th)
                N2 = E["N"][ix]
                n3 = N2[:, 0:1] * pl.u + N2[:, 1:2] * (c * pl.v - sn * pl.n)
                mid = 0.5 * (A3 + B3) + 0.5 * e
                vis = ((eye - mid) * n3).sum(1) > 0
                for a, b, n_ in zip(A3[vis], B3[vis], n3[vis]):
                    polys.append(np.array([a, b, b + e, a + e]))
                    vals.append(0.12 + 0.88 * max(0.0, float(n_ @ self.LT)))
            cons = self.lay["letters"][i]["cons"]
            top = [self.pose(cn, yb, th, -h) for cn in cons]
            base = [self.pose(cn, yb, th, 0.0) for cn in cons]
            tm = np.concatenate(top).mean(0)
            if float((-dd) @ (eye - tm)) > 0:
                caps_top.append(top)
            else:
                caps_base.append(base)
        out += quant_layers(polys, vals, self.ramp, levels=20, bias=0.0)
        capcol = pal.fg
        for cp in caps_top:
            out.append(Layer(cp, capcol, "fill", alpha=fa, bias=20.0))
            out.append(Layer(cp, mixc(pal.fg, pal.bg, 0.8), "line", 1.3, closed=True, alpha=fa, bias=21.0))
        for cp in caps_base:
            out.append(Layer(cp, mixc(pal.bg, pal.accents[2], 0.3), "fill", alpha=fa, bias=20.0))
        return out
