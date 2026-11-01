"""beat_orbit: one pivot-locked, even-radius orbit over the WHOLE sequence with a spin-read-spin speed profile locked to the word
hits (13_camera v006 breath_echo, as a CameraModule so it runs in a CameraChain).

Every word is read face-on (the camera azimuth equals the word's placement yaw) from its hit (end of the form, on a downbeat) to its
release, with a slow creep that carries on in the direction of the next spin. Between a release and the next hit the camera spins on
to the next word: a quintic Hermite from the creep state of one reading to the creep state of the next, so it arrives exactly at the
hit and leaves exactly at the release (both on beats when the words are beat-snapped). Spin directions follow the sign of the yaw
change (give the words alternating yaws for breath-echo's alternating spins). The first reading is preceded by an intro spin from
`intro_deg` off face-on. Spin windows are declared fast (the heading gate is measured separately there)."""
import math
import numpy as np

from mograph.modules.camera.base import CameraModule, register_camera, vdir


def _h5(u):
    return (1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5, u - 6 * u ** 3 + 8 * u ** 4 - 3 * u ** 5,
            0.5 * u ** 2 - 1.5 * u ** 3 + 1.5 * u ** 4 - 0.5 * u ** 5)


@register_camera("beat_orbit")
class BeatOrbit(CameraModule):
    description = ("One even-radius orbit around the pivot for the whole piece: each word is read face-on between its hit and its "
                   "release (slow creep), and the camera spins to the next word between the release and the next hit.")
    max_relax = 1.0
    DEFAULTS = dict(fill=0.7, radius_scale=1.0, elev=4.0, bob=0.8, read_dps=7.0, intro_deg=42.0, fast_dps=35.0, lead=0.0)

    def _profile(self, hints):
        """-> list of (t_a, t_b, fn(t)) pieces covering the timeline and the declared fast windows."""
        W = hints.words
        face = [w.yaw_deg for w in W]
        hit = [w.t_hold0 for w in W]
        rel = [max(w.t_hold1, w.t_hold0) for w in W]
        n = len(W)
        sgn = []
        for i in range(n):
            if i + 1 < n:
                d = face[i + 1] - face[i]
                sgn.append(1.0 if d >= 0 else -1.0)
            else:
                sgn.append(sgn[-1] if sgn else 1.0)
        rd = self.p("read_dps")
        mid = [0.5 * (hit[i] + rel[i]) for i in range(n)]
        c = [rd * sgn[i] for i in range(n)]
        read = lambda i, t: face[i] + c[i] * (t - mid[i])
        pieces, fast = [], []
        # intro: rest -> reading 0
        p1, v1 = read(0, hit[0]), c[0]
        p0 = face[0] - sgn[0] * self.p("intro_deg") + c[0] * (hit[0] - mid[0])
        T = max(hit[0], 0.3)
        pieces.append((-1e9, hit[0], ("herm", 0.0, T, p0, 0.0, p1, v1)))
        for i in range(n):
            pieces.append((hit[i], rel[i], ("read", i)))
            if i + 1 < n:
                ta, tb = rel[i], hit[i + 1]
                pieces.append((ta, tb, ("herm", ta, tb - ta, read(i, ta), c[i], read(i + 1, tb), c[i + 1])))
        pieces.append((rel[-1], 1e9, ("read", n - 1)))
        return pieces, read

    def _theta(self, hints, ts):
        pieces, read = self._profile(hints)
        th = np.zeros(len(ts))
        for a, b, spec in pieces:
            m = (ts >= a) & (ts < b)
            if not m.any():
                continue
            if spec[0] == "read":
                i = spec[1]
                th[m] = read(i, ts[m])
            else:
                _, t0, T, p0, v0, p1, v1 = spec
                u = np.clip((ts[m] - t0) / max(T, 1e-6), 0, 1)
                H0, H1, H2 = _h5(u)
                G0, G1, G2 = _h5(1 - u)
                th[m] = H0 * p0 + H1 * T * v0 + G0 * p1 - G1 * T * v1
        return th

    def curves(self, hints, ts, t0, t1):
        W = hints.words
        th = np.radians(self._theta(hints, ts))
        R = max(w.fit(self.p("fill")) for w in W) * self.p("radius_scale")
        pitch = math.radians(self.p("elev")) + math.radians(self.p("bob")) * np.sin(2 * math.pi * 0.17 * ts)
        # pivot: the word's centre while it is read, blended over the spin to the next
        piv = np.tile(W[0].origin, (len(ts), 1))
        for i in range(1, len(W)):
            if np.allclose(W[i].origin, W[i - 1].origin):
                continue
            a, b = W[i - 1].t_hold1, W[i].t_hold0
            u = np.clip((ts - a) / max(b - a, 1e-6), 0, 1)
            u = u * u * u * (u * (u * 6 - 15) + 10)
            piv = piv + (W[i].origin - W[i - 1].origin)[None, :] * u[:, None]
        eye = piv - R * vdir(th, pitch)
        return dict(eye=eye, look=piv)

    def declare(self, hints, t0, t1):
        ts = np.arange(int(round((t0 - 0.5) * 240)), int(round((t1 + 0.5) * 240)) + 1) / 240.0
        th = self._theta(hints, ts)
        rate = np.abs(np.gradient(th, ts))
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
        return [], out
