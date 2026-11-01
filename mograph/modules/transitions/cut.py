"""cut: the hand-off when one side carries nothing (a flicker or wipe style has material `none`). Sequence puts it in place of
whatever transition sat between the two words. It is a match-cut with a flash: no strands travel; for about a quarter of a
second hairlines join the exit gate of A to the entry gate of B on a few frame-exact frames (a thin line, a thick one, a thin
one, with a dark frame between), so the eye is led across the gap. Duration grows a little with the distance between the words."""
import math
import numpy as np
from mograph.modules.core import TransitionModule, register, Layer, unit

FPS = 30.0


@register("cut")
class Cut(TransitionModule):
    """params: flash (frame pattern, 1 = line on / 0 = dark; default (0, 1, 2, 0, 1)), lines (hairlines across the gap, 3),
    note (name of the transition this replaced, informational)."""
    DEMO_DUR = 0.4
    SUITS = dict(
        brief="a side carries no material (flicker / wipe); also a tiny gap at high energy",
        dist=((-1, 0, 50, 130), 1.0), energy=((0.75, 0.95, 1.0, 1.01), 1.0), must=("dist", "energy"), dur=(0.24, 0.7, 1.0),
        moods={"hype": 0.9, "tense": 0.8, "playful": 0.3})
    description = ("Hand-off between words when one side carries no material (flicker / wipe styles): no strands travel; "
                   "frame-exact hairlines flash across the gap between the exit gate and the entry gate, then the next "
                   "word starts.")

    def __init__(self, a_exit=None, b_entry=None, dur=None, **params):
        self.replaced = params.pop("replaced", None)
        super().__init__(a_exit, b_entry, dur, **params)

    def _anchors(self):
        aw, bw = self.a_word, self.b_word
        if aw is not None and bw is not None:
            return aw.exit_material().anchor, bw.entry_material().anchor, aw.exit_material().dir, bw.entry_material().dir
        pa, pb = self.a_exit.centroid(), self.b_entry.centroid()
        return pa, pb, unit(self.a_exit.D.mean(0)), unit(self.b_entry.D.mean(0))

    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        d = float(np.linalg.norm(b_entry.centroid() - a_exit.centroid()))
        return float(min(max(0.22 + d / 2600.0, 0.24), 0.7))

    def build(self):
        self.pa, self.pb, self.da, self.db = self._anchors()
        n = max(2, int(self.params.get("lines", 3)))
        d = unit(self.pb - self.pa)
        side = np.cross(d, [0.0, 0.0, 1.0])
        side = unit(side) if np.linalg.norm(side) > 1e-6 else np.array([0.0, 1.0, 0.0])
        self.offs = [side * (k - (n - 1) / 2.0) * 9.0 for k in range(n)]

    def layers(self, t):
        pat = self.params.get("flash", (0, 1, 2, 0, 1))
        f = int(round(t * FPS))
        if f < 0 or f >= len(pat) or pat[f] == 0:
            return []
        w = 1.4 if pat[f] == 1 else 4.5
        pal = self.a_word.palette if self.a_word is not None else None
        col = pal.fg if pal is not None else (236, 230, 218)
        polys = [np.array([self.pa + o, self.pb + o]) for o in self.offs]
        return [Layer(polys, col, "line", w, closed=False, bias=0.5)]

    def focus(self, t):
        u = min(max(t / max(self.dur, 1e-6), 0.0), 1.0)
        return self.pa + (self.pb - self.pa) * u

    def bounds_points(self):
        return np.array([self.pa, self.pb])
