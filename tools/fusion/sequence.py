"""FusionSequence: a mograph Sequence whose word layers follow animated font axes (axes.AxesWarp)."""
from mograph.modules.sequence import Sequence


class FusionSequence(Sequence):
    warps = ()                       # per timeline item: an AxesWarp or None (set by spec.build_scene)

    def layers(self, t):
        k = 0
        for k, (it, a, b) in enumerate(self.timeline):
            if t < b:
                break
        it, a, b = self.timeline[k]
        L = it.layers(max(t - a, 0.0))
        wp = self.warps[k] if k < len(self.warps) else None
        return wp.apply(L, t) if wp is not None else L
