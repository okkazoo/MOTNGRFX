"""hold_drift: a composed still. The eye rests face-on in front of the word at its fit distance, but off-centre (the word sits a
tenth of the frame off the middle, the side alternating word to word), pushes in a few percent over the hold and carries a
sub-pixel, low-frequency drift, so the hold is alive without ever reading as a move. Declares the whole window as a rest."""
import math
import numpy as np
from mograph.modules.camera.base import CameraModule, register_camera, smoother, unit
from mograph.render import cam3d

_FR = np.array([[0.113, 0.171], [0.217, 0.089], [0.067, 0.243]])
_PH = np.array([[0.4, 2.1], [1.7, 0.3], [3.9, 5.2]])


def hold_curves(w, ts, ta, tb, dolly=0.06, comp=(0.09, 0.05), drift_px=1.6, fill=0.7, side=1.0, fit_scale=1.0, push_from=None):
    """(eye, look) arrays of the composed hold of word w on times ts; the push-in runs over [ta, tb]."""
    D0 = w.fit(fill) * fit_scale
    s = smoother((ts - ta) / max(tb - ta, 1e-6))
    D = D0 * (1.0 - dolly * s)
    ppx = D / w.F                                            # world units per frame px at the word
    off = (w.u[None, :] * (-comp[0] * side * cam3d.W * ppx)[:, None] + w.v[None, :] * (comp[1] * cam3d.H * ppx)[:, None])
    nz = np.stack([sum(np.sin(2 * math.pi * _FR[k, c] * ts + _PH[k, c]) for k in range(3)) / 3.0 for c in range(2)], 1)
    dr = (w.u[None, :] * nz[:, :1] + w.v[None, :] * nz[:, 1:]) * (drift_px * ppx)[:, None]
    look = w.origin[None, :] + off + dr
    eye = look - w.vn[None, :] * D[:, None]
    return eye, look


@register_camera("hold_drift")
class HoldDrift(CameraModule):
    description = ("A composed still: the eye rests face-on at the word's fit distance with the word a tenth of a frame off "
                   "centre, a slow push-in of a few percent and a sub-pixel drift. The hold never reads as a static centred shot.")
    DEFAULTS = dict(word=None, dolly=0.06, comp=(0.09, 0.05), drift_px=1.6, fill=0.7)

    def _word(self, hints, t0, t1):
        return hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))

    def curves(self, hints, ts, t0, t1):
        w = self._word(hints, t0, t1)
        eye, look = hold_curves(w, ts, t0, t1, self.p("dolly"), self.p("comp"), self.p("drift_px"), self.p("fill"),
                                1.0 if w.idx % 2 == 0 else -1.0)
        return dict(eye=eye, look=look)

    def declare(self, hints, t0, t1):
        return [(t0, t1)], []
