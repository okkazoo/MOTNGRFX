"""push_through: dive through a counter (the hole of a letter) into what is behind it. The eye leaves the word's hold pose, lines
up on the biggest counter along the word's plane normal, accelerates through the hole (the type rushes past and grows out of
frame) and, once through, pans on to the strands (the look-at blends from straight ahead to the strand focus). With to_next
(default) the route goes on, beyond the dived-through plane, to the composed hold pose of the NEXT word and arrives there at
rest, so the module hands over to a hold with no U-turn (place the next word on the far side of the plane, along the way the
strands leave). Moves on a FlowPath (C2 spline, arc-length speed): it starts from rest at the hold pose."""
import math
import numpy as np
from mograph.modules.camera.hold_drift import hold_curves
from mograph.modules.camera.base import CameraModule, register_camera, smoother, smooth_series, lerp, unit, PAD, GRID
from mograph.render.cam3d import FlowPath


@register_camera("push_through")
class PushThrough(CameraModule):
    description = ("Dives through a counter: the eye lines up on a letter's hole, accelerates through it so the type rushes past "
                   "and grows out of frame, then pans on to the strands and the next word behind.")
    max_relax = 1.2
    DEFAULTS = dict(word=None, counter=0, cross=0.5, approach=0.5, beyond=0.35, fill=0.7, ease_in=0.55, pan=1.7, to_next=True)

    def _geom(self, hints, t0, t1):
        w = hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(t0 + 0.25 * (t1 - t0))
        D = w.fit(self.p("fill"))
        cs = w.counters()
        ctr = cs[min(self.p("counter"), len(cs) - 1)][0] if cs else w.origin
        n = w.n
        e0, _ = w.pose(D)
        p1 = ctr - n * (self.p("approach") * D)
        p2 = ctr.copy()
        p3 = ctr + n * (self.p("beyond") * D)
        p4 = p3 + n * (0.25 * D)
        pts = [e0, p1, p2, p3, p4]
        nxt = None
        if self.p("to_next") and w.idx + 1 < len(hints.words):
            nxt = hints.words[w.idx + 1]
            e_n, l_n = hold_curves(nxt, np.array([t1]), t1, nxt.t_hold1, fill=self.p("fill"), side=1.0 if nxt.idx % 2 == 0 else -1.0)
            pts.append(e_n[0])
            self._arrive = (nxt, l_n[0])
        else:
            self._arrive = None
        return w, D, ctr, n, np.array(pts)

    def curves(self, hints, ts, t0, t1):
        w, D, ctr, n, pts = self._geom(hints, t0, t1)
        tc = t0 + self.p("cross") * (t1 - t0)
        last = t1 >= hints.T - 1e-6
        arrive = self._arrive is not None
        fp = FlowPath(pts, t0, t1, ease_in=self.p("ease_in"), ease_out=0.7 if arrive else 0.0, v_start=0.0,
                      v_end=0.0 if arrive else 0.3, look=lambda t: ctr, times=[None, None, tc] + [None] * (len(pts) - 3),
                      drift_px=0.0)
        eye = np.array([fp.eye_at(float(min(max(t, t0), t1))) for t in ts])
        L_ahead = ctr + n * (0.8 * D)
        fs = smooth_series(lambda tt: hints.focus(tt - 0.2), ts, 0.45)
        a_in = smoother((ts - t0) / 0.7)
        a_out = smoother((ts - (tc - 0.6)) / self.p("pan"))
        look0 = lerp(w.origin, L_ahead, a_in)
        look = lerp(look0, fs, 0.55 * a_out)
        if arrive:                                                 # the last second settles the view on the next word's hold
            look = lerp(look, np.tile(self._arrive[1], (len(ts), 1)), smoother((ts - (t1 - 1.6)) / 1.3))
        return dict(eye=eye, look=look)

    def noclear(self, hints, t0, t1):
        tc = t0 + self.p("cross") * (t1 - t0)
        return [(tc - 0.9, tc + 1.8)]

    def declare(self, hints, t0, t1):
        tc = t0 + self.p("cross") * (t1 - t0)
        return [(t0 - 1.0, t0 + self.p("ease_in"))], []
