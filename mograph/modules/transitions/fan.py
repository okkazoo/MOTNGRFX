"""fan: the bundle spreads out into a flat fan (a hand-fan sweep): every strand leaves A's ribbon on its own ray, the outermost
ones sweep `angle` degrees apart, a faint web fills the blade, and the rays gather again into B's entry ribbon. Params:
angle (degrees, total sweep, 54), web (draw the blade, True), stiff, stretch, piece."""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc
from mograph.modules.transitions._path import PathTransition


@register("fan")
class Fan(PathTransition):
    DEMO_PARAMS = dict(angle=62)
    DEMO_DUR = 3.0
    SUITS = dict(
        brief="spreading toward a wider / larger target",
        size_ratio=((1.0, 1.35, 4.0, 6.0), 2.5), heading=((-1, 0, 60, 130), 0.8), detour=((-1, 0, 70, 130), 0.6),
        dist=((80, 200, 800, 1300), 0.8), plane=({"same": 1.0, "fold": 0.6, "opposite": 0.2}, 0.6),
        energy=((0.1, 0.45, 1.0, 1.0), 0.5), must=("size_ratio",), dur=(1.2, 3.2, 0.95),
        moods={"joyful": 0.9, "hype": 0.8, "playful": 0.7, "calm": 0.4, "mysterious": 0.4, "nostalgic": 0.3, "tense": 0.3,
               "melancholy": 0.2})
    description = ("Spreads the bundle into a flat fan: each strand takes its own ray, the outer strands sweep wide apart "
                   "with a faint web between them, and the rays gather back into the next word's entry.")

    def paths(self, A, B):
        P = self.params
        K = self.K
        F = self.axis_frame(A, B, stiff=float(P.get("stiff", 1.0)))
        self.F = F
        u = F["u"]
        n = len(u)
        Lax = float(np.linalg.norm(np.diff(F["C"], axis=0), axis=1).sum())
        ang = math.radians(float(P.get("angle", 54)))
        Hw = 0.5 * Lax * math.tan(ang / 2.0)
        W, _ = self.cross_extent(A, B)
        Hw = max(Hw, 0.7 * W)
        order = np.argsort(F["al"][0], kind="stable")
        rank = np.empty(K, int); rank[order] = np.arange(K)
        fk = (rank - (K - 1) / 2.0) / max((K - 1) / 2.0, 1.0)
        t = 1.0 - np.abs(2 * u - 1.0)                       # triangular: straight rays out and back
        m = max(5, n // 12) | 1
        kern = np.hanning(m + 2)[1:-1]
        kern /= kern.sum()
        pad = np.r_[np.zeros(m // 2), t, np.zeros(m // 2)]
        bell = np.convolve(pad, kern, "valid")
        bell = bell / max(bell.max(), 1e-9)
        x = F["al"] * (1 - bell[:, None]) + (fk * Hw)[None, :] * bell[:, None]
        y = F["be"] * (1 - bell[:, None])
        pts = self.rebuild(F, x, y, F["ga"] * (1 - bell[:, None]))
        self.rank = rank
        return [pts[:, k] for k in range(K)]

    def extra_layers(self, t):
        if not self.params.get("web", True):
            return []
        wins = self.windows(t)
        lo, hi = int(np.argmin(self.rank)), int(np.argmax(self.rank))
        wl, wh = wins[lo][0], wins[hi][0]
        if wl is None or wh is None:
            return []
        f = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
        g = float(np.clip(f / 0.2, 0, 1) * np.clip((1 - f) / 0.2, 0, 1))
        if g < 0.02:
            return []
        poly = np.vstack([wl, wh[::-1]])
        col = mixc(tuple(int(x) for x in self.A.col.mean(0)), (255, 255, 255), 0.4)
        return [Layer([poly], col, "fill", alpha=0.2 * g, bias=-50.0)]
