"""spiral_around: the strands roll up from A's flat hand-off ribbon into a ring, spiral around an object, and unroll into B's
entry. `around` picks the object: 'cylinder' (straight drum, constant radius), 'sphere' (the winding bulges to the middle like
latitudes of a globe) or 'word' (a flat slab, the bounding form of a word block, strands wind round its rounded-rectangle
section). The object is drawn as a faint wireframe while the strands wrap it. Params: around, turns (1.5), handed (+1 / -1),
radius (None = auto from the ribbon), ghost (draw the object, True), stiff, stretch, piece."""
import math
import numpy as np
from mograph.modules.core import register, unit, Layer, GREY
from mograph.modules.transitions._path import PathTransition


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@register("spiral_around")
class SpiralAround(PathTransition):
    DEMO_PARAMS = dict(around="cylinder", turns=1.5)
    DEMO_DUR = 3.2
    SUITS = dict(
        brief="big heading change or detour, or passing an object / word",
        **{"heading|detour|obstacle": (((45, 110, 180, 181), (60, 120, 180, 181), (-1, -1, 40, 220)), 3.0)},
        dist=((100, 260, 1000, 1600), 0.8), plane=({"same": 0.6, "fold": 1.0, "opposite": 0.9}, 0.5),
        energy=((0.0, 0.3, 0.9, 1.0), 0.4), must=("heading|detour|obstacle",), dur=(2.0, 4.2, 1.2),
        moods={"mysterious": 0.9, "joyful": 0.8, "playful": 0.8, "hype": 0.5, "calm": 0.5, "melancholy": 0.5, "nostalgic": 0.6,
               "tense": 0.4})
    description = ("Rolls the strands up into a ring and spirals them around an object, a cylinder, a sphere or the flat slab "
                   "of a word block, drawn as a faint wireframe, before unrolling them into the next word's entry. Pick the "
                   "object with around and the winding with turns and handed.")

    def _profile(self, u, phi):
        """(x, y) frame coordinates of the object surface at axis parameter u and angle phi (arrays broadcast)."""
        R, kind = self.R0, self.around
        if kind == "sphere":
            Rs = 1.2 * R
            z = (u - 0.5) * self.Lax
            rho = np.sqrt(np.maximum(Rs * Rs - np.minimum(np.abs(z), 0.8 * Rs) ** 2, 1e-6))
            return rho * np.cos(phi), rho * np.sin(phi)
        if kind == "word":
            a, b, n = self.slab
            c, s = np.cos(phi), np.sin(phi)
            return a * np.sign(c) * np.abs(c) ** (2.0 / n), b * np.sign(s) * np.abs(s) ** (2.0 / n)
        return R * np.cos(phi) + 0 * u, R * np.sin(phi) + 0 * u

    def paths(self, A, B):
        P = self.params
        K = self.K
        self.around = str(P.get("around", "cylinder"))
        F = self.axis_frame(A, B, stiff=float(P.get("stiff", 1.0)))
        self.F = F
        u = F["u"]
        n = len(u)
        W, Dp = self.cross_extent(A, B)
        pitch = self.min_pitch(A, B)
        self.R0 = float(P.get("radius") or max(0.55 * W, K * pitch * 1.1 / (2 * math.pi), 28.0))
        self.Lax = float(np.linalg.norm(np.diff(F["C"], axis=0), axis=1).sum())
        aw = getattr(self, "a_word", None)
        if aw is not None and hasattr(aw, "placement"):
            ha = 0.5 * float(aw.placement.h)
        else:
            ha = 1.15 * self.R0
        self.slab = (max(ha, self.R0), 0.4 * max(ha, self.R0), 4.0)
        order = np.argsort(F["al"][0], kind="stable")
        rank = np.empty(K, int); rank[order] = np.arange(K)
        phi0 = math.pi * (1.0 - 2.0 * (rank + 0.5) / K)
        hand = 1.0 if float(P.get("handed", 1)) >= 0 else -1.0
        turns = float(P.get("turns", 1.5))
        psi = hand * 2 * math.pi * turns * _ss((u - 0.12) / 0.76)
        phi = phi0[None, :] + psi[:, None]
        kap = (_ss(u / 0.3) * _ss((1 - u) / 0.3))[:, None]
        xs, ys = self._profile(u[:, None], phi)
        x = (1 - kap) * F["al"] + kap * xs
        y = (1 - kap) * F["be"] + kap * ys
        gr = F["ga"] * (1 - kap)
        pts = self.rebuild(F, x, y, gr)
        return [pts[:, k] for k in range(K)]

    # ------------------------------------------------------------ ghost object
    def extra_layers(self, t):
        if not self.params.get("ghost", True):
            return []
        f = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
        g = float(_ss(f / 0.22) * _ss((1 - f) / 0.22))
        if g < 0.02:
            return []
        F = self.F
        n = len(F["u"])
        col = GREY
        out = []
        us = np.linspace(0.22, 0.78, 8)
        th = np.linspace(0, 2 * math.pi, 40)
        for uu in us:
            i = int(round(uu * (n - 1)))
            xs, ys = self._profile(np.full_like(th, F["u"][i]), th)
            ring = F["C"][i] + xs[:, None] * F["e1"][i] + ys[:, None] * F["e2"][i]
            out.append(Layer([ring], col, "line", 1.2, closed=True, alpha=0.5 * g))
        ths = np.linspace(0, 2 * math.pi, 9)[:-1]
        uu = np.linspace(0.22, 0.78, 24)
        for a in ths:
            idx = np.round(uu * (n - 1)).astype(int)
            xs, ys = self._profile(F["u"][idx], np.full(len(idx), a))
            line = F["C"][idx] + xs[:, None] * F["e1"][idx] + ys[:, None] * F["e2"][idx]
            out.append(Layer([line], col, "line", 1.0, closed=False, alpha=0.32 * g))
        return out

    def bounds_points(self):
        return np.concatenate([self.F["C"], np.concatenate(self.pts)[::3], self.A.P.reshape(-1, 3), self.B.P.reshape(-1, 3)])
