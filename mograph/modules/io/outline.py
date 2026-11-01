"""outline: the word's OWN GLYPH OUTLINES are the hand-off (2026-11-01, user: the library transitions must not sit in their own disjointed corner;
their strands should deform from word A's letters and resolve into word B's letters).

Exit (form_out): over the exit window the fill of the formed word fades while its outlines draw in, then the outline arcs ease a little off the
letters (outward along the glyph's distance field): the exit material is those arcs, K strands (K = the module's own exit-bundle count), head
first, heading along the outline. A library transition (braid / circuit_board / fan / spiral_around / twist via _path.py, ripple) then slides
them off the letters, along its own unchanged route and pattern, to the next word.
Entry (form_in): the transition delivers its strands as arcs lying just off B's outlines (the entry material); over the form window they ease
onto the outlines while the fill comes in under them, then thin away into the formed word.
Colour and speed are the module's own strand values (its raw exit / entry bundle); width / tube are the unravel hairline (below), so the
module keeps its route and character while its strands are fine and separate. Options: off (px the arcs sit off the letters at the hand-off, default 0.06 x cap), pts (points per arc, 24).
UNRAVEL (2026-11-01, user: "strands unravelling through a transition then reforming into the next ... not so dense"; "lose the 3d tubular wireframe
text"): the outline is cut into FEW separate strands (one per ~1.6 cap of outline, 10-18), drawn as fine FLAT hairlines (width HAIR px, no tube)."""
import math
import numpy as np

from mograph.modules.core import Bundle, strand_layers, unit
from mograph.modules.io.base import IOStyle, register_style, fade, ss, side_frame
from mograph.modules.material import Material

PTS = 24
HAIR = 1.1                 # px at home depth: the unravelled strands are fine flat hairlines (never tubes)
K_RANGE = (10, 18)         # strands the outline unravels into


def _resample(c, n):
    c = np.asarray(c, float)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return np.repeat(c[:1], n, 0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, c[:, 0]), np.interp(q, s, c[:, 1])], 1)


def outline_arcs(cons, K, n=PTS):
    """K open arcs (each n points, plane coords) that together cover the closed contours `cons`: every contour gets >= 1 arc (largest remainder by
    length; the smallest contours are dropped when there are more contours than K). -> (K, n, 2)."""
    cs = [np.asarray(c, float) for c in cons if len(c) >= 3]
    lens = np.array([np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum() for c in cs])
    order = np.argsort(-lens, kind="stable")[:K]
    cs, lens = [cs[i] for i in order], lens[order]
    alloc = np.ones(len(cs), int)
    rest = K - len(cs)
    if rest > 0:
        share = np.maximum(lens / lens.sum() * K - 1.0, 0.0)
        add = np.floor(share / max(share.sum(), 1e-9) * rest).astype(int)
        alloc += add
        for j in np.argsort(-(share - add))[:K - alloc.sum()]:
            alloc[j] += 1
    out = []
    for c, m in zip(cs, alloc):
        M = m * (n - 1) + 1
        r = _resample(np.vstack([c, c[:1]]), M)
        for j in range(m):
            out.append(_resample(r[j * (n - 1):(j + 1) * (n - 1) + 1], n))
    return np.stack(out[:K])


def _outward(shape, P2):
    """unit outward normals (plane coords) at points P2 (..., 2) from the gradient of the glyph signed distance field."""
    g = shape.grid()
    sdf = g["sdf"]
    gy, gx = np.gradient(sdf, g["step"])
    ix = np.clip(np.round((P2[..., 0] - g["x0"]) / g["step"]).astype(int), 0, sdf.shape[1] - 1)
    iy = np.clip(np.round((P2[..., 1] - g["y0"]) / g["step"]).astype(int), 0, sdf.shape[0] - 1)
    v = np.stack([gx[iy, ix], gy[iy, ix]], -1)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


@register_style("outline")
class Outline(IOStyle):
    description = ("The word's own glyph outlines are the hand-off: on exit the fill fades into its outline and the outline arcs peel off "
                   "into the transition; on entry the arriving strands lie on the next word's outline and its fill comes in under them.")
    material_kind = "strands"

    def attach(self, body, side):
        super().attach(body, side)
        self._mat = None

    def _raw_bundle(self):
        return self.body._rawcall("entry_bundle" if self.side == "in" else "exit_bundle")

    def arcs(self, off_k=1.0):
        """(K, n, 3) world arcs on the outlines, eased `off_k` x the hand-off offset off the letters (outward)."""
        sh = self.body.hold_shape()
        raw = self._raw_bundle()
        K = int(np.clip(round(sh.perimeter() / max(1.6 * sh.cap, 1.0)), *K_RANGE))
        A2 = outline_arcs(sh.cons, K, int(self.opt("pts", PTS)))
        off = float(self.opt("off", 0.06 * sh.cap)) * float(off_k)
        if off:
            A2 = A2 + _outward(sh, A2) * off
        return self.body.placement.lift(A2), raw, K

    def material(self):
        if self._mat is None:
            P, raw, K = self.arcs(1.0)
            d = side_frame(self.body, self.side)[0]                          # the word's own exit / entry travel (the module's route keeps its shape and travel)
            if self.side == "out":
                D = unit(0.75 * d[None] + 0.25 * unit(P[:, 0] - P[:, 1]))       # head first, peeling off along the outline toward the exit side
            else:
                P = P[:, ::-1].copy()                                         # entry: the tail is where the strand arrives; it slides on toward the head
                D = unit(0.75 * d[None] + 0.25 * unit(P[:, -2] - P[:, -1]))
            idx = np.arange(K) % raw.N
            b = Bundle(P, D, raw.col[idx], np.full(K, HAIR), raw.t, raw.v[idx], 0.0, raw.rim[idx])
            self._mat = Material.from_bundle(b)
        return self._mat

    def _arc_layers(self, off_k, alpha):
        P, raw, K = self.arcs(off_k)
        idx = np.arange(K) % raw.N
        ls = strand_layers(list(P), list(raw.col[idx]), [HAIR] * K, 0.0, None, hi_off=np.array([-0.8, -0.8, 0.0]))
        return fade(ls, alpha)

    def layers(self, t, p):
        p = max(0.0, float(p))
        if self.side == "out":                                                # formed word -> its outline -> the arcs ease off the letters
            if p >= 1.0:
                return self._exact()
            body = fade(self.body.conceal_layers(0.0, t), 1.0 - ss(p / 0.6))
            return body + self._arc_layers(ss((p - 0.45) / 0.55), ss(p / 0.35))
        if p <= 0.0:                                                          # the arcs the transition delivered, just off B's letters ...
            return self._exact()
        if p >= 1.0:
            return self.body.reveal_layers(0.0, t)
        body = fade(self.body.reveal_layers(0.0, t), ss((p - 0.15) / 0.6))   # ... ease onto the outline while the fill comes in, then thin away
        return body + self._arc_layers(1.0 - ss(p / 0.55), 1.0 - ss((p - 0.6) / 0.4))

    def _exact(self):
        b = self.material().bundle
        return strand_layers(list(b.P), list(b.col), list(b.w), b.tube, b.rim if b.tube > 0.01 else None, hi_off=np.array([-0.8, -0.8, 0.0]))

    def focus(self, t, p):
        return self.body.placement.origin.copy()
