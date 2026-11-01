"""pullback_reveal: one continuous pull-back that reveals the whole. The eye starts tight on the build front (the strand heads at
`from_scale` x the fit distance) and backs away along the view axis, easing out (quintic) to rest on a framing that holds the
whole word (to='word': `word_scale` x the fit distance, centred on it) or the whole layout (to='scene': every placement and
transition fits, front-on, but never so far out that the word of the window drops below `min_word` of the frame width). The look-at slides from the build front to the centre of what is revealed."""
import math
import numpy as np
from mograph.modules.camera.base import (CameraModule, register_camera, smoother, smooth_series, lerp, slerp_dir)
from mograph.render import cam3d


@register_camera("pullback_reveal")
class PullbackReveal(CameraModule):
    description = ("One continuous pull-back from tight on the build front to a framing of the whole word (or the whole layout), "
                   "easing out to rest as the reveal lands.")
    DEFAULTS = dict(word=None, to="word", from_scale=0.5, word_scale=1.45, margin=1.12, fill=0.7, min_word=0.3)

    def curves(self, hints, ts, t0, t1):
        w = hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))
        D = w.fit(self.p("fill"))
        u = np.clip((ts - t0) / (t1 - t0), 0.0, 1.0)
        s = smoother(u)
        if self.p("to") == "scene":
            P = hints.scene_points()
            lo, hi = P.min(0), P.max(0)
            cen = 0.5 * (lo + hi)
            d1 = max(hints.F, hints.F * (hi - lo)[0] * self.p("margin") / cam3d.W, hints.F * (hi - lo)[1] * self.p("margin") / cam3d.H)
            # never so far out that the word of the window is no longer readable (>= `min_word` of the frame width): a layout
            # whose other words are gone would otherwise end on a tiny speck
            d1 = min(d1, hints.F * w.w / (self.p("min_word") * cam3d.W))
            nd = slerp_dir(w.vn, np.array([0.0, 0.0, 1.0]), s)
        else:
            cen, d1 = w.origin, D * self.p("word_scale")
            nd = np.tile(w.vn, (len(ts), 1))
        d0 = D * self.p("from_scale")
        dist = d0 + (d1 - d0) * s
        fs = smooth_series(lambda tt: hints.focus(tt - 0.15), ts, 0.2)
        c = lerp(fs, cen, smoother(np.clip((u - 0.1) / 0.8, 0, 1)))
        eye = c - nd * dist[:, None]
        return dict(eye=eye, look=c)

    def declare(self, hints, t0, t1):
        return [(t1 - 0.3, t1)], []
