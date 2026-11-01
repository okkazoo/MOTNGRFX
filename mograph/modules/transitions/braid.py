"""braid: the strands split into three plaits that weave over and under each other like a three-strand braid (each plait is a
small ribbon of neighbouring strands; the plait centres follow three sine waves a third of a cycle apart and swing in depth so
they pass over / under), then settle into B's entry. Params: cycles (1.5), plaits (3), amp (None = auto lateral swing),
depth (None = auto), handed (+1 / -1), stiff, stretch, piece."""
import math
import numpy as np
from mograph.modules.core import register
from mograph.modules.transitions._path import PathTransition


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@register("braid")
class Braid(PathTransition):
    DEMO_PARAMS = dict(cycles=1.5)
    DEMO_DUR = 3.2
    SUITS = dict(
        brief="long straight same-plane runs, calm",
        dist=((300, 550, 1500, 2400), 2.0), heading=((-1, 0, 35, 80), 1.2), detour=((-1, 0, 30, 70), 1.0),
        plane=({"same": 1.0, "fold": 0.3, "opposite": 0.05}, 1.2), energy=((-1, 0.0, 0.4, 0.75), 1.5),
        must=("dist", "plane", "energy"), dur=(2.2, 4.4, 1.2),
        moods={"calm": 1.0, "nostalgic": 0.8, "melancholy": 0.7, "mysterious": 0.4, "joyful": 0.3, "playful": 0.3, "tense": 0.1,
               "hype": 0.05})
    description = ("Splits the strands into three plaits and weaves them over and under each other like a braid before they "
                   "settle into the next word's entry. Tune it with cycles and handed.")

    def paths(self, A, B):
        if self.params.get("unravel"):
            return self._unravel_paths(A, B)
        P = self.params
        K = self.K
        F = self.axis_frame(A, B, stiff=float(P.get("stiff", 1.0)))
        self.F = F
        u = F["u"]
        W, Dp = self.cross_extent(A, B)
        pitch = self.min_pitch(A, B)
        G = int(max(2, min(int(P.get("plaits", 3)), K)))
        order = np.argsort(F["al"][0], kind="stable")
        rank = np.empty(K, int); rank[order] = np.arange(K)
        grp = np.minimum(rank * G // K, G - 1)
        cnt = np.array([(grp == j).sum() for j in range(G)])
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        within = (rank - start[grp]) - (cnt[grp] - 1) / 2.0
        gw = np.maximum(cnt * pitch, 1.0)                            # plait widths
        W_ = max(W, float(gw.max()) * G * 0.8)
        amp = float(P.get("amp") or 0.34 * W_ + 0.2 * float(gw.max()))
        dep = float(P.get("depth") or max(0.5 * amp, 2.5 * pitch))
        hand = 1.0 if float(P.get("handed", 1)) >= 0 else -1.0
        cyc = float(P.get("cycles", 1.5))
        om = hand * 2 * math.pi * cyc * _ss((u - 0.08) / 0.84)
        ph = om[:, None] + 2 * math.pi * grp[None, :] / G
        kap = (_ss(u / 0.26) * _ss((1 - u) / 0.26))[:, None]
        sx = amp * np.sin(ph) + within[None, :] * pitch
        sy = dep * np.cos(ph)
        x = (1 - kap) * F["al"] + kap * sx
        y = (1 - kap) * F["be"] + kap * sy
        pts = self.rebuild(F, x, y, F["ga"] * (1 - kap))
        return [pts[:, k] for k in range(K)]

    def _unravel_paths(self, A, B):
        """the UNRAVEL braid (2026-11-01, user on the dense knot: "more complex and separated and not so dense - want to see strands braiding ...
        it should flow through"): 3-5 separate groups of a few fine strands, clear gaps between them, plaiting over and under along most of the path
        (the braid spans 80 % of it, wide: 0.55 x the hand-off width, at least 0.18 x the A-B distance), groups opening apart before they converge on B.
        The strands slide along these fixed routes, so they keep flowing through the turns. Params: plaits (None = 3-5 by strand count), cycles (2.5)."""
        P = self.params
        K = self.K
        F = self.axis_frame(A, B, stiff=float(P.get("stiff", 1.0)))
        self.F = F
        u = F["u"]
        W, _ = self.cross_extent(A, B)
        G = int(P.get("plaits") or int(np.clip(round(K / 4.0), 3, 5)))
        G = max(2, min(G, K))
        order = np.argsort(F["al"][0], kind="stable")
        rank = np.empty(K, int); rank[order] = np.arange(K)
        grp = np.minimum(rank * G // K, G - 1)
        cnt = np.array([max((grp == j).sum(), 1) for j in range(G)])
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        within = (rank - start[grp]) - (cnt[grp] - 1) / 2.0
        from mograph.render import cam3d
        amp = float(P.get("amp") or min(max(0.55 * W, 0.18 * F["dist"], 60.0), 0.16 * min(cam3d.W, cam3d.H)))     # never a frame-filling flower (R16: a word-high hand-off)
        lane = 0.11 * amp                                              # the strands of a group keep this far apart: separate lines
        dep = 0.45 * amp
        hand = 1.0 if float(P.get("handed", 1)) >= 0 else -1.0
        cyc = float(P.get("cycles", 2.5))
        om = hand * 2 * math.pi * cyc * _ss((u - 0.06) / 0.88)
        ph = om[:, None] + 2 * math.pi * grp[None, :] / G
        kap = (_ss(u / 0.12) * _ss((1 - u) / 0.12))[:, None]
        sx = amp * np.sin(ph) + within[None, :] * lane
        sy = dep * np.cos(ph)
        x = (1 - kap) * F["al"] + kap * sx
        y = (1 - kap) * F["be"] + kap * sy
        pts = self.rebuild(F, x, y, F["ga"] * (1 - kap))
        return [pts[:, k] for k in range(K)]
