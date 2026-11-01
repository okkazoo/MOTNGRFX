"""ease_reversal: a run across the word that eases to rest at the end and eases out the other way (the strips g04 fix). The
eye travels along the word's reading direction at constant speed, its velocity follows 2*smootherstep - 1 around the reversal
time, so it decelerates to a full stop over `ease` s, turns and accelerates back to the same speed the other way: no bump, no
cusp. The look-at is mostly on the word (the pan leads a `pan` fraction of the travel). The rest is declared."""
import math
import numpy as np
from mograph.modules.camera.base import CameraModule, register_camera, smoother, smooth_series, lerp, cumtrapz, GRID


@register_camera("ease_reversal")
class EaseReversal(CameraModule):
    description = ("A run across the word that eases to rest at the end and eases out the other way: speed follows a smootherstep "
                   "through zero at the turn, so there is no bump or cusp at the reversal.")
    max_relax = 1.6
    DEFAULTS = dict(word=None, direction=1, extent=0.5, back=0.3, end=0.0, ease=0.55, pan=0.3, depth_scale=1.0, fill=0.7, lead_out=1.0)

    def _timing(self, w, t0, t1):
        """speed V and reversal time tr: the run starts `back` x width before the word's centre, turns `extent` x width past it,
        and returns to `end` x width, at one constant speed with an `ease`-second smootherstep stop at the turn."""
        tau = self.p("ease")
        A, A0, E = self.p("extent") * w.w, self.p("back") * w.w, self.p("end") * w.w
        V = (A + A0 + (A - E)) / max(t1 - t0 - tau, 0.5)
        tr = t0 + (A + A0) / V + tau / 2.0
        return A, V, tr, tau

    def _x(self, ts, t0, t1, w):
        d = float(self.p("direction"))
        A, V, tr, tau = self._timing(w, t0, t1)
        v = V * (2.0 * smoother(0.5 + (tr - ts) / (2.0 * tau)) - 1.0)
        x = cumtrapz(v, 1.0 / GRID)
        x = x - np.interp(tr, ts, x) + A                       # x(tr) = A (the far end of the run)
        return d * x, tr, tau

    def curves(self, hints, ts, t0, t1):
        w = hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))
        x, tr, tau = self._x(ts, t0, t1, w)
        R = w.fit(self.p("fill")) * self.p("depth_scale")
        eye = w.origin[None, :] + w.u[None, :] * x[:, None] - w.vn[None, :] * R
        look = w.origin[None, :] + w.u[None, :] * (self.p("pan") * x)[:, None]
        # the last stretch the view goes on to the strands (the word is deconstructing and leaving): no empty frames after the run
        fs = smooth_series(lambda tt: hints.focus(tt - 0.15), ts, 0.25)
        look = lerp(look, fs, smoother((ts - (t1 - self.p("lead_out"))) / self.p("lead_out")))
        return dict(eye=eye, look=look)

    def declare(self, hints, t0, t1):
        tr = self._timing(hints.word_at(0.5 * (t0 + t1)) if self.p("word") is None else hints.words[self.p("word")], t0, t1)[2]
        return [(tr - 0.6 * self.p("ease"), tr + 0.6 * self.p("ease"))], []
