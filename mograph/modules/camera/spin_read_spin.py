"""spin_read_spin: fast spin to the word, slow to read it, spin on. The eye orbits the word (pivot-locked, constant radius) with a
rate profile: a fast spin-in that ends face-on exactly as the type has formed, a slow drift (a few degrees per second) through
the hold, a fast spin-out that starts as it deconstructs. direction 0 alternates +1 / -1 from one spin_read_spin to the next
(the breath-echo note: different spin directions). Spin windows are declared `fast` (the heading / view turn gates are measured
separately there)."""
import math
import numpy as np
from mograph.modules.camera.base import CameraModule, register_camera, smoother, smooth_series, lerp, yaw_pitch, vdir, _runs


@register_camera("spin_read_spin")
class SpinReadSpin(CameraModule):
    description = ("Fast spin round the word so that it lands face-on as it forms, a slow drift while it is read, then a fast "
                   "spin on as it deconstructs. Pivot-locked at constant radius; every next spin runs the other way.")
    max_relax = 1.6
    DEFAULTS = dict(word=None, sweep=75.0, read_dps=8.0, direction=0, radius_scale=1.0, elev=5.0, fast_dps=90.0, fill=0.7, start_cap=26.0, lead_out=0.85)

    def _words(self, hints, t0, t1):
        return hints.words[self.p("word")] if self.p("word") is not None else hints.word_at(0.5 * (t0 + t1))

    def entry_az(self, hints, t0, t1, nth):
        """(word index, azimuth in rad) the spin starts at: lets the module before it (follow) arrive there."""
        w = self._words(hints, t0, t1)
        d = self.p("direction") or (1.0 if nth % 2 == 0 else -1.0)
        yw = math.degrees(yaw_pitch(w.vn)[0])
        S_in = math.radians(max(15.0, min(self.p("sweep"), 62.0 + d * yw)))
        S_out = math.radians(max(15.0, min(self.p("sweep"), 75.0 - d * yw)))
        S_in, S_out = self._side_cap(w, d, S_in, S_out)
        if t0 <= 1e-6:
            S_in = min(S_in, math.radians(self.p("start_cap")))
        return w.idx, -d * S_in

    def _side_cap(self, w, d, S_in, S_out, cap_deg=28.0):
        """the eye never swings far to the side the strands arrive from / leave to (it would sit in the lanes): on such a side
        the spin is capped at `cap_deg`."""
        yaw0 = yaw_pitch(w.vn)[0]
        R = 1.0
        for k, (S, port) in enumerate(((S_in, w.mod.entry), (S_out, w.mod.exit))):
            az = -d * S if k == 0 else d * S
            off = -R * vdir(np.array([yaw0 + az]), np.array([0.0]))[0] + R * vdir(np.array([yaw0]), np.array([0.0]))[0]
            if float(off @ port.side(w.pl)) > 0.25 * R * math.sin(math.radians(cap_deg)):
                if k == 0:
                    S_in = min(S_in, math.radians(cap_deg))
                else:
                    S_out = min(S_out, math.radians(cap_deg))
        return S_in, S_out

    def _az(self, hints, ts, t0, t1):
        w = self._words(hints, t0, t1)
        d = self.p("direction") or (1.0 if hints.nth.get(self.name, 0) % 2 == 0 else -1.0)
        S = math.radians(self.p("sweep"))
        yw = math.degrees(yaw_pitch(w.vn)[0])                       # keep the eye within +-75 deg of the page (never edge-on / behind)
        S_in = math.radians(max(15.0, min(self.p("sweep"), 62.0 + d * yw)))
        S_out = math.radians(max(15.0, min(self.p("sweep"), 75.0 - d * yw)))
        S_in, S_out = self._side_cap(w, d, S_in, S_out)
        if t0 <= 1e-6:                                            # the chain starts here: a clean entry, never a slanted narrow view
            S_in = min(S_in, math.radians(self.p("start_cap")))
        a0 = max(t0, w.t_hold0 - 0.0)                              # spin-in ends when the word has formed
        a1 = min(t1, w.t_hold1)                                    # spin-out starts as it deconstructs
        tin, tout = min(max(a0 - t0, 0.5), 1.0), min(max(t1 - a1, 0.5), 1.0)
        tc = 0.5 * (a0 + a1)
        read = math.radians(self.p("read_dps")) * (ts - tc)
        spin_in = -S_in * (1.0 - smoother((ts - (a0 - tin)) / tin))
        spin_out = S_out * smoother((ts - a1) / tout)
        return w, d * (spin_in + read + spin_out)

    def curves(self, hints, ts, t0, t1):
        w, az = self._az(hints, ts, t0, t1)
        yaw0, pit0 = yaw_pitch(w.vn)
        pitch = pit0 + math.radians(1.5) * np.sin(2 * math.pi * 0.17 * ts)
        R = w.fit(self.p("fill")) * self.p("radius_scale")
        eye = w.origin[None, :] - R * vdir(yaw0 + az, pitch)
        look = np.tile(w.origin, (len(ts), 1))
        a1 = min(t1, w.t_hold1)                                  # while the type deconstructs the view slides on to the strands leaving
        fs = smooth_series(lambda tt: hints.focus(tt - 0.15), ts, 0.2)
        look = lerp(look, fs, self.p("lead_out") * smoother((ts - (a1 + 0.2)) / max(t1 - a1 - 0.2, 0.3)))
        return dict(eye=eye, look=look)

    def declare(self, hints, t0, t1):
        ts = np.arange(int(round((t0 - 0.5) * 240)), int(round((t1 + 0.5) * 240)) + 1) / 240.0
        w, az = self._az(hints, ts, t0, t1)
        rate = np.abs(np.gradient(az, ts)) * 180.0 / math.pi
        fast = rate > self.p("fast_dps")
        out, i = [], 0
        while i < len(ts):
            if fast[i]:
                j = i
                while j + 1 < len(ts) and fast[j + 1]:
                    j += 1
                out.append((ts[i], ts[j]))
                i = j + 1
            else:
                i += 1
        # the eye is stationary before the spin-in starts and after the spin-out ends (declared rests)
        w, az = self._az(hints, ts, t0, t1)
        rate = np.abs(np.gradient(az, ts)) * 180.0 / math.pi
        still = rate < 1.5
        pre = [(a, b) for a, b in _runs(ts, still & (ts < w.t_hold0)) if b - a > 0.15]
        post = [(a, b) for a, b in _runs(ts, still & (ts > w.t_hold1)) if b - a > 0.15]
        return [r for r in pre + post], out
