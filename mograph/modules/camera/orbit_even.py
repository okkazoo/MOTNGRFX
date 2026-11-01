"""orbit_even: constant radius around a pivot, look locked on it, constant angular rate (even: no ease inside the move, so the
speed is steady and the parallax reads as one smooth swing). The pivot is the word of the window (its placement centre) or
`pivot`. The orbit is a fixed arc of `sweep` degrees (default 100: a clearly readable swing, not a nudge) centred on the face-on
view of the word (at the middle of the window the camera is square on to the word), run at one rate over the whole window; it
never goes past +-62 deg off the word's normal (never edge-on). Direction: direction = +1 / -1, else alternating from one
orbit to the next. The arc starts at `entry_az`, which the follow before it arrives at (so the join needs no reversal)."""
import math
import numpy as np
from mograph.modules.camera.base import CameraModule, register_camera, yaw_pitch, vdir


@register_camera("orbit_even")
class OrbitEven(CameraModule):
    description = ("Constant-radius orbit around the word, look locked on the pivot, constant angular rate: an even swing of "
                   "about 100 degrees that passes square-on to the word at the middle, the parallax of the strands against the "
                   "type doing the work. The follow before it arrives at the start of the arc.")
    max_relax = 2.0
    DEFAULTS = dict(word=None, sweep=100.0, direction=0, radius_scale=0.95, bob=1.0, fill=0.7, pivot=None, centre=0.0, max_off=62.0)

    def _word(self, hints, t0, t1):
        return hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))

    def _dir(self, nth):
        """+1 / -1: forced by `direction`, else alternating from one orbit to the next."""
        return float(self.p("direction")) if self.p("direction") else (1.0 if nth % 2 == 0 else -1.0)

    def _sweep(self):
        return math.radians(min(self.p("sweep"), 2.0 * self.p("max_off")))

    def entry_az(self, hints, t0, t1, nth):
        """(word index, azimuth in rad) the arc starts at, so the module before it (follow) can arrive there (no reversal)."""
        w = self._word(hints, t0, t1)
        return w.idx, math.radians(self.p("centre")) - self._dir(nth) * 0.5 * self._sweep()

    def curves(self, hints, ts, t0, t1):
        w = self._word(hints, t0, t1)
        piv = np.asarray(self.p("pivot"), float) if self.p("pivot") is not None else w.origin
        R = w.fit(self.p("fill")) * self.p("radius_scale")
        yaw0, pit0 = yaw_pitch(w.vn)
        d = self._dir(hints.nth.get(self.name, 0))
        dur = t1 - t0
        sweep = self._sweep()
        rate = sweep / dur
        az = math.radians(self.p("centre")) + d * rate * (ts - 0.5 * (t0 + t1))
        pitch = pit0 + math.radians(self.p("bob")) * np.sin(2 * math.pi * 0.17 * ts)
        eye = piv[None, :] - R * vdir(yaw0 + az, pitch)
        return dict(eye=eye, look=np.tile(piv, (len(ts), 1)))
