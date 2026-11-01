"""candy_orbit: strands draw the outline, then a solid striped rod grows out of the face: 42 filled copies of the word,
banded vermilion / ink three at a time, receding into the page toward a vanishing point that ORBITS the word like a camera
circling it. On exit the rod retracts into the face (still orbiting), then the strands leave.
(mechanism: 11_mathtype b8 candy orbit; 13_camera module_journey INTO, rewritten for the Bundle interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc
from mograph.modules.words._depth import MechWord, ss, eio, eo, seg


@register("candy_orbit")
class CandyOrbit(MechWord):
    description = ("Strands draw the outline, then a striped rod of 42 filled copies grows out of the face, banded vermilion and "
                   "ink, receding toward a vanishing point that orbits the word like a camera going round. On exit the rod "
                   "retracts into the face while still orbiting, and the strands run out.")
    DEMO_WORDS = "ORBIT"
    DEFAULT_FONT = "fascinate"
    DEFAULT_PALETTE = "bone"
    DEFAULT_TIMING = (2.0, 0.8, 2.0)
    MS = 0.22
    LIB_CAMERA = None                    # the orbit is in the geometry: the plain demo camera shows it

    def mech_build(self):
        P, pl = self.params, self.placement
        self.NC = int(P.get("copies", 42))
        self.BAND = int(P.get("band", 3))
        self.cons3 = self.cons2()
        self.turn = float(P.get("turn", 270.0))              # degrees the vanishing point travels round the word
        self.rod = (0.72 * pl.w, 0.44 * pl.w, 1.1 * pl.w)     # (x, y, depth) reach of the rod tip in the placement frame
        self.cols = np.array(self.cols)

    def theta(self, t):
        a, b = self.t_ms, self.t_c
        u = (t - a) / max(b - a, 1e-6)
        return math.radians(135.0 - self.turn * eio(u, 1.7))

    def rod_vec(self, t):
        pl = self.placement
        th = self.theta(t)
        rx, ry, rz = self.rod
        return rx * math.cos(th) * pl.u + ry * math.sin(th) * pl.v + rz * pl.n

    def mech_layers(self, t):
        pal = self.palette
        p = self.progress(t)
        out = []
        n = self.NC * eo(seg(p, 0.0, 1.0), 2.2)
        E = self.rod_vec(t)
        ca, cb = pal.accents[0], pal.fg
        for j in range(int(math.ceil(n)), 0, -1):
            q = min(j, n) / self.NC
            col = (ca, cb)[((j - 1) // self.BAND) % 2]
            out.append(Layer([self.lift(c) + q * E for c in self.cons3], col, "fill", bias=0.0))
        if p > 1e-4:                                          # the face (bg fill + outline) takes over from the strand arcs
            face = [self.lift(c) for c in self.cons3]
            out.append(Layer(face, pal.bg, "fill", bias=1.0))
            out.append(Layer(face, pal.fg, "line", self.w * (1.0 + 0.4 * ss(p / 0.2)), closed=True, bias=2.0))
        return out
