"""Beat grid for the fusion renderer: a bpm or an explicit list of beat times (seconds); downbeats every `bar` beats.
Used to snap word hits (downbeats), phase boundaries (beats) and to drive animated font axes ("beat")."""
import numpy as np


class Beats:
    def __init__(self, bpm=None, beats=None, bar=4, offset=0.0, duration=60.0, downbeats=None):
        self.bar = int(bar)
        if beats:
            t = np.array(sorted(float(b) for b in beats), float)
            iv = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
            ext = np.arange(1, int(max(0.0, duration + 2.0 - t[-1]) / iv) + 2) * iv + t[-1]
            self.t = np.r_[t, ext] if t[-1] < duration + 2.0 else t
        elif bpm:
            iv = 60.0 / float(bpm)
            self.t = offset + np.arange(0, int((duration + 2.0 - offset) / iv) + 2) * iv
        else:
            self.t = None
            return
        self.iv = float(np.median(np.diff(self.t)))
        if downbeats:
            self.d = np.array(sorted(float(x) for x in downbeats), float)
        else:
            self.d = self.t[::self.bar]

    def __bool__(self):
        return self.t is not None

    def snap(self, x, bars=False, mode="nearest", minimum=None):
        """nearest beat (downbeat if bars) to x; mode 'floor' | 'ceil' | 'nearest'; the result is >= minimum when given."""
        if self.t is None:
            return x if minimum is None else max(x, minimum)
        g = self.d if bars else self.t
        i = int(np.searchsorted(g, x))
        cand = [g[max(i - 1, 0)], g[min(i, len(g) - 1)]]
        if mode == "floor":
            v = g[max(np.searchsorted(g, x + 1e-9, "right") - 1, 0)]
        elif mode == "ceil":
            v = g[min(np.searchsorted(g, x - 1e-9, "left"), len(g) - 1)]
        else:
            v = min(cand, key=lambda c: abs(c - x))
        if minimum is not None and v < minimum - 1e-9:
            j = int(np.searchsorted(g, minimum - 1e-9, "left"))
            v = g[min(j, len(g) - 1)]
        return float(v)

    def prev(self, t, bars=False):
        """(time of the last beat <= t, interval to the next one)."""
        g = self.d if bars else self.t
        i = max(int(np.searchsorted(g, t + 1e-9, "right")) - 1, 0)
        nxt = g[min(i + 1, len(g) - 1)]
        return float(g[i]), float(max(nxt - g[i], 1e-3))
