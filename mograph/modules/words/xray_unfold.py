"""xray_unfold: line-only material. Strands draw the outline, then the whole word hinges back about its baseline onto the
floor, its depth rises out as 14 see-through contour slices fading from ink to pale, and it swings back upright as one body
with the slab standing behind it and the corner edges running front to back; the letter's own face keeps a heavier accent
outline. On exit it folds back the same way and the strands leave. (mechanism: 11_mathtype c9 unfold x-ray; 13_camera
module_journey LIGHT, rewritten for the Bundle interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc, ol
from mograph.modules.words._depth import MechWord, ss, eio, eo, seg


@register("xray_unfold")
class XrayUnfold(MechWord):
    description = ("Strands draw the outline, then the word hinges back onto the floor about its baseline, its depth unfolds as "
                   "14 see-through contour slices and it swings upright with the slab standing behind it, the face keeping a "
                   "heavier accent outline. On exit it folds back down and up again, and the strands leave.")
    DEMO_WORDS = "LIGHT"
    DEFAULT_FONT = ("tourney", {"wght": 800, "wdth": 100})
    DEFAULT_PALETTE = "ink"
    DEFAULT_TIMING = (2.6, 0.9, 2.3)
    HIDE = True
    LIB_CAMERA = dict(yaw=(18.0, -12.0, -20.0), pitch=(26.0, 14.0, 20.0), dist=(1250.0, 1180.0, 1250.0), cz=120.0, cy=70.0)

    def mech_build(self):
        P = self.params
        self.NS = int(P.get("slices", 14))
        self.depth = float(P.get("depth", 1.3)) * self.lay["cap"]
        self.yb = 0.5 * self.lay["size"][1]
        self.cons3 = self.cons2()
        self.feat = []
        for c in self.cons3:
            self.feat.append(np.asarray(ol.feature_idx(np.asarray(c), np.array([0.0, 1.0]), 40), int))

    def state(self, p):
        tip = eio(seg(p, 0.0, 0.32), 2.4)
        rise = eio(seg(p, 0.20, 0.62), 2.2)
        up = eio(seg(p, 0.58, 0.97), 2.4)
        return 0.5 * math.pi * (tip - up), self.depth * rise

    def mech_layers(self, t):
        pal = self.palette
        p = self.progress(t)
        th, h = self.state(p)
        yb = self.yb
        out = []
        if h > 2.0:
            for j in range(self.NS, 0, -1):
                u = j / self.NS
                col = mixc(pal.fg, pal.bg, 0.12 + 0.78 * u)
                out.append(Layer([self.pose(c, yb, th, h * u) for c in self.cons3], col, "line", 1.0, closed=True, bias=-0.5 * j))
            edges = []
            for c, fi in zip(self.cons3, self.feat):
                for i in fi:
                    e = np.stack([self.pose(c[i:i + 1], yb, th, 0.0)[0], self.pose(c[i:i + 1], yb, th, h)[0]])
                    edges.append(e)
            if edges:
                out.append(Layer(edges, mixc(pal.fg, pal.bg, 0.45), "line", 0.8, closed=False, bias=-0.2))
        k = ss(p / 0.1)
        out.append(Layer([self.pose(c, yb, th, 0.0) for c in self.cons3], mixc(pal.fg, pal.accents[2], k), "line",
                         self.w * (1.0 + 0.8 * k), closed=True, bias=2.0))
        return out
