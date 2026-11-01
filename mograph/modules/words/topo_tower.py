"""topo_tower: a topographic model of the word. Strands draw the outline, the word tips back to lie on the floor, and its
contour copies stack straight up out of it, layer after layer, into a contour-line tower (paper-filled layers, grey at the
foot to ink at the top cap, one accent contour half way up). On exit the tower sinks back into the floor layer by layer, the
word stands up again, and the strands leave. (mechanism: 11_mathtype a2 topo tower, rewritten for the Bundle interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc
from mograph.modules.words._depth import MechWord, ss, eio, seg


@register("topo_tower")
class TopoTower(MechWord):
    description = ("Strands draw the outline, the word tips back to lie on the floor and its contour copies stack straight up "
                   "into a topographic tower of paper-filled layers, grey at the foot to ink at the top. On exit the tower "
                   "sinks back layer by layer, the word stands up and the strands run out.")
    DEMO_WORDS = "TALL"
    DEFAULT_FONT = "ultra"
    DEFAULT_PALETTE = "bone"
    DEFAULT_TIMING = (2.6, 0.9, 2.3)
    HIDE = True
    LIB_CAMERA = dict(yaw=(14.0, -14.0, -20.0), pitch=(22.0, 42.0, 40.0), dist=(1250.0, 1180.0, 1250.0), cz=120.0, cy=60.0)

    def mech_build(self):
        P = self.params
        self.NL = int(P.get("layers", 18))
        self.SP = float(P.get("spacing") or 0.075 * self.lay["cap"])
        self.yb = 0.5 * self.lay["size"][1]
        self.cons3 = self.cons2()

    def hold_pose(self, dist=900.0):
        pl = self.placement
        tgt = pl.origin + pl.n * 120.0 + pl.v * self.yb * 0.6
        return tgt - pl.n * dist * 0.7 - pl.v * dist * 0.7, tgt

    def layer_col(self, k, p):
        pal = self.palette
        u = k / max(self.NL - 1, 1)
        tone = mixc(mixc(pal.fg, pal.bg, 0.55), pal.fg, u ** 0.7)
        if self.NL > 6 and k == self.NL // 2:
            tone = pal.accents[0]
        if k == 0:
            tone = mixc(pal.fg, mixc(pal.fg, pal.bg, 0.3), ss(p / 0.2))
        return tone

    def mech_layers(self, t):
        pal = self.palette
        p = self.progress(t)
        tip = eio(seg(p, 0.0, 0.28), 2.4)
        dn = eio(seg(p, 0.24, 0.95), 2.0) * (self.NL - 1)
        th = 0.5 * math.pi * tip
        out = []
        for k in range(self.NL):
            hgt = min(k, dn) * self.SP
            if k > 0 and hgt < 0.5:
                continue
            polys = [self.pose(c, self.yb, th, -hgt) for c in self.cons3]
            top = (k == self.NL - 1)
            ta = ss(dn - (self.NL - 2)) if top else 0.0           # the cap inks in as the tower reaches full height
            col = self.layer_col(k, p)
            out.append(Layer(polys, pal.bg, "fill", bias=0.0))
            if ta > 0.004:
                out.append(Layer(polys, pal.fg, "fill", alpha=ta, bias=0.1))
            out.append(Layer(polys, col, "line", 1.0 + (0.7 if (top or k == self.NL // 2) else 0.0) * ss(p / 0.3) + 0.4 * (k == 0), closed=True))
        return out
