"""graze: a low, close traverse along a surface. The eye sits a short way off the word's plane (`gap` x the type height), a little
below its centre line, and travels along the reading direction at constant speed, looking ahead along the surface (the view is
about atan(1 / lead) off the plane), so the letters and strands stream past with strong parallax and the type is only read in
passing. Follows the creation lines of a word that builds left to right."""
import math
import numpy as np
from mograph.modules.camera.base import CameraModule, register_camera, smoother, smooth_series, lerp


@register_camera("graze")
class Graze(CameraModule):
    description = ("A low, close traverse along the word's surface: the eye skims along the reading direction a short way off "
                   "the plane, looking ahead along it, so the letters and strands stream past with strong parallax.")
    DEFAULTS = dict(word=None, direction=1, reach=1.25, end=0.2, gap=0.75, below=0.3, lead=2.0, fill=0.7, back_off=1.9)

    def curves(self, hints, ts, t0, t1):
        w = hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))
        d = float(self.p("direction"))
        L = self.p("reach") * w.w
        spd = L / (t1 - t0)
        x = d * (self.p("end") * w.w + spd * (ts - t1))            # runs from before the word's start to `end` x width past its centre
        gap0 = self.p("gap") * w.h
        bo = smoother((ts - (t1 - self.p("back_off"))) / self.p("back_off"))       # rises away from the surface at the end
        gap = gap0 + (w.fit(self.p("fill")) * 0.8 - gap0) * bo
        y = self.p("below") * w.h * (1 - bo) + 0.04 * w.h * np.sin(2 * math.pi * 0.19 * ts)
        eye = w.origin[None, :] + w.u[None, :] * x[:, None] - w.vn[None, :] * gap[:, None] + w.v[None, :] * y[:, None]
        lead = self.p("lead") * gap
        look = w.origin[None, :] + w.u[None, :] * (x + d * lead)[:, None] + w.v[None, :] * (0.15 * y)[:, None]
        fs = smooth_series(lambda tt: hints.focus(tt - 0.2), ts, 0.25)       # backing off: the look slides on to the strands
        look = lerp(look, fs, bo)
        return dict(eye=eye, look=look)
