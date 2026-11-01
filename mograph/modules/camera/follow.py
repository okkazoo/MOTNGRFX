"""follow: rides the creation lines. The look-at point is the strand heads / build front (Sequence.focus) trailing it by `lag`
s; the eye sits behind them along the plane normal of the active word (slerped from word A's normal to word B's through a
transition), close while the strands build the type (`close` x the word's fit distance) and a little further out between
words. When a word has formed, the eye settles into that word's composed hold (hold_drift pose) and leaves it again as the
type deconstructs. The eye is a heavier-smoothed version of the track than the look, so it flows round the strands instead of
translating with them (no translate-reframing)."""
import math
import numpy as np
from mograph.modules.camera.base import (CameraModule, register_camera, smoother, smooth_series, slerp_dir, lerp, unit, yaw_pitch, vdir)
from mograph.modules.camera.hold_drift import hold_curves


def anchor_path(hints, ts):
    """slow path through the word origins (smoothstep across every transition)."""
    a = np.tile(hints.words[0].origin, (len(ts), 1))
    for tr in hints.trans:
        s = smoother((ts - tr.a) / (tr.b - tr.a))
        a = np.where((ts >= tr.a)[:, None], lerp(tr.wa.origin, tr.wb.origin, s), a)
    return a


@register_camera("follow")
class Follow(CameraModule):
    description = ("Rides the creation lines: the eye trails the strand heads along the build front (look-at lag 0.2 s), close "
                   "while the type is being drawn, a little further out between words, and settles into a composed hold on each "
                   "formed word before leaving with the strands again.")
    DEFAULTS = dict(lag=0.2, close=0.82, between=1.5, gain=0.3, look_gain=0.85, eye_smooth=1.1, look_smooth=0.22, settle=0.5, fill=0.7)

    def _entry(self, hints):
        """(word index, azimuth) the NEXT module wants to start from (spin_read_spin), if it says so."""
        nm = hints.next_module
        if nm is not None and hasattr(nm, "entry_az") and hints.next_window is not None:
            return nm.entry_az(hints, hints.next_window[0], hints.next_window[1], hints.next_nth)
        return None

    def _track(self, hints, ts, entry=True):
        """(eye, look) of pure tracking and the hold weight per word."""
        n = len(ts)
        nrm = np.zeros((n, 3))
        dist = np.zeros(n)
        hw = np.zeros(n)
        hold_eye, hold_look = np.zeros((n, 3)), np.zeros((n, 3))
        fit = lambda w: w.fit(self.p("fill"))
        ent = self._entry(hints) if entry else None
        # per-time item lookup, vectorised by item windows
        for w in hints.words:
            m = (ts >= w.a) & (ts < w.b)
            nrm[m] = w.vn
            dist[m] = fit(w) * self.p("close")
        for tr in hints.trans:
            m = (ts >= tr.a) & (ts < tr.b)
            if m.any():
                u = (ts[m] - tr.a) / (tr.b - tr.a)
                s = smoother(u)
                nb = tr.wb.vn
                if ent is not None and ent[0] == tr.wb.idx:
                    yb, pb = yaw_pitch(nb)
                    nb = vdir(np.array([yb + ent[1]]), np.array([pb]))[0]
                nrm[m] = slerp_dir(tr.wa.vn, nb, s)
                base = lerp(fit(tr.wa), fit(tr.wb), s) * self.p("close")
                dist[m] = base * (1.0 + (self.p("between") - 1.0) * np.sin(np.pi * u))
        m = ts < hints.words[0].a
        nrm[m], dist[m] = hints.words[0].vn, fit(hints.words[0]) * self.p("close")
        m = ts >= hints.words[-1].b
        nrm[m], dist[m] = hints.words[-1].vn, fit(hints.words[-1]) * self.p("close")
        # the word nearest in time provides the hold pose; weight = settle in during the late form, out during the early exit
        for w in hints.words:
            up = smoother((ts - (w.t_hold0 - self.p("settle"))) / self.p("settle"))
            dn = 1.0 - smoother((ts - w.t_hold1) / (0.8 * self.p("settle")))
            wt = up * dn
            e, l = hold_curves(w, ts, w.t_hold0, w.t_hold1, fill=self.p("fill"), side=1.0 if w.idx % 2 == 0 else -1.0)
            m = wt > hw
            hold_eye[m], hold_look[m], hw[m] = e[m], l[m], wt[m]
        return nrm, dist, hw, hold_eye, hold_look

    def _anchor(self, hints, ts):
        return anchor_path(hints, ts)

    def curves(self, hints, ts, t0, t1):
        lag, g_e, g_l = self.p("lag"), self.p("gain"), self.p("look_gain")

        def eye_track(tt):
            nrm, dist, hw, he, hl = self._track(hints, tt)
            f = lerp(self._anchor(hints, tt), hints.focus(tt), g_e)
            return f - nrm * dist[:, None]

        def look_track(tt):
            return lerp(self._anchor(hints, tt - lag), hints.focus(tt - lag), g_l)

        eye = smooth_series(eye_track, ts, self.p("eye_smooth"))
        look = smooth_series(look_track, ts, self.p("look_smooth"))
        nrm, dist, hw, he, hl = self._track(hints, ts)
        return dict(eye=lerp(eye, he, hw), look=lerp(look, hl, hw))

    def declare(self, hints, t0, t1):
        rests = []
        for w in hints.words:
            a, b = w.t_hold0 - 0.3, w.t_hold1 + 0.3
            a, b = max(a, t0), min(b, t1)
            if b > a + 0.2:
                rests.append((a, b))
        return rests, []
