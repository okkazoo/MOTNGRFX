"""flicker: the word glitches into or out of existence. Every glyph and every ~44 px fragment of the drawing has its own
birth time and blinks on and off from frame to frame (frame-exact: the state is a function of the 30 fps frame number, so
motion blur keeps each frame crisp) with a small positional jitter, horizontal slices of the picture slip sideways on some
frames, and a couple of frame-exact strobes (one frame fully on, the next fully dark) punctuate it. It locks solid at the end
of the reveal. Nothing is carried, so the material is `none` (a transition next to it becomes a cut).
Options: fps (30), frag (fragment length px, 44), jitter (fraction of the placement width, 0.010), slices (True),
strobes (window fractions, (0.30, 0.62)), beat (seconds: strobes and glyph births snap to the beat grid; default None), seed (1)."""
import math
import numpy as np

from mograph.modules.material import Material
from mograph.modules.io.base import (IOStyle, register_style, side_frame, hash01, consolidate, style_key, slab_layers, shifted,
                                     ss)

LOCK = 0.965


@register_style("flicker")
class Flicker(IOStyle):
    description = ("The word stutters into or out of existence: glyphs and fragments blink on and off frame by frame with "
                   "jitter, slices of the picture slip sideways, and frame-exact strobes punctuate it. Snaps to a beat grid "
                   "on request.")
    material_kind = "none"

    def attach(self, body, side):
        super().attach(body, side)
        self.d, self.e1, self.e2, self.gate = side_frame(body, side)
        self._tab = None

    # ------------------------------------------------------------ fragment table
    def base(self, t):
        b = self.body
        return b.reveal_layers(0.0, t) if self.side == "in" else b.conceal_layers(1.0, t)

    def table(self):
        if self._tab is not None:
            return self._tab
        b, pl = self.body, self.body.placement
        base = self.base(None)
        frag_len = float(self.opt("frag", 44.0))
        keys, polys, cen = [], [], []
        for L in base:
            key = style_key(L)
            if L.kind == "fill":
                groups = [[P] for P in L.polys] if L.split else [list(L.polys)]
                for g in groups:
                    keys.append(key); polys.append(g); cen.append(np.concatenate(g).mean(0))
            else:
                for P in L.polys:
                    if L.closed:
                        P = np.vstack([P, P[:1]])
                    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                    n = max(1, int(round(cum[-1] / frag_len)))
                    cuts = np.searchsorted(cum, np.linspace(0, cum[-1], n + 1))
                    for a, c in zip(cuts[:-1], cuts[1:]):
                        piece = P[a:c + 1] if c > a else P[a:a + 2]
                        if len(piece) < 2:
                            continue
                        keys.append(key); polys.append([piece]); cen.append(piece.mean(0))
        cen = np.array(cen) if cen else np.zeros((0, 3))
        rel = cen - pl.origin
        cu, cv = rel @ pl.u, rel @ pl.v
        # glyph = nearest letter box of the formed word
        sh = b.hold_shape()
        boxes = np.array([[np.concatenate(l)[:, 0].min(), np.concatenate(l)[:, 1].min(), np.concatenate(l)[:, 0].max(),
                           np.concatenate(l)[:, 1].max()] for l in sh.letters])
        dx = np.maximum(np.maximum(boxes[None, :, 0] - cu[:, None], cu[:, None] - boxes[None, :, 2]), 0)
        dy = np.maximum(np.maximum(boxes[None, :, 1] - cv[:, None], cv[:, None] - boxes[None, :, 3]), 0)
        gid = np.argmin(np.hypot(dx, dy), axis=1) if len(cu) else np.zeros(0, int)
        cid = np.floor(cu / 24.0).astype(np.int64) * 7919 + np.floor(cv / 24.0).astype(np.int64) * 104729
        self._tab = dict(keys=keys, polys=polys, cu=cu, cv=cv, gid=gid, cid=cid)
        return self._tab

    # ------------------------------------------------------------ frame
    def snap(self, x, step):
        return x if not step else round(x / step) * step

    def layers(self, t, p):
        b, pl, tm = self.body, self.body.placement, self.body.timing
        reveal = self.side == "in"
        if (reveal and p >= 1.0) or (not reveal and p >= 1.0):
            return [] if not reveal else self.base(t)
        if reveal and p <= 0.0:
            return []
        fps = float(self.opt("fps", 30.0))
        seed = int(self.opt("seed", 1))
        beat = self.opt("beat", None)
        t0, win = b.style_window(self.side)
        f = int(round(t * fps))
        q = p if reveal else 1.0 - p                                   # how built the word is (1 = formed)
        # frame-exact strobes: frame fs fully on, the next one dark
        for sp in self.opt("strobes", (0.30, 0.62)):
            ts = self.snap(t0 + sp * win, beat)
            fs = int(round(ts * fps))
            if f == fs:
                return self.base(t)
            if f == fs + 1:
                return []
        if q >= LOCK:
            return self.base(t)
        tab = self.table()
        n = len(tab["keys"])
        if n == 0:
            return []
        gid, cid = tab["gid"], tab["cid"]
        qstep = (beat / win) / 2.0 if beat else 0.0
        gb = np.array([self.snap(0.04 + 0.5 * float(hash01(seed, g)), qstep) for g in range(int(gid.max()) + 1)])[gid]
        fb = gb + 0.22 * hash01(seed + 1, cid)
        lq = np.clip((q - fb) / 0.28, 0.0, 1.0)
        rf = hash01(seed + 2, cid, f)
        rg = hash01(seed + 3, gid, f // 2)
        prob = 0.3 + 0.62 * lq
        on = (lq > 0) & ((lq >= 1.0) | (rf < prob))
        on &= (rg < 0.35 + 0.65 * np.clip((q - gb) / 0.35, 0, 1))
        on &= ~((lq >= 1.0) & (hash01(seed + 4, cid, f) < 0.07))
        idx = np.nonzero(on)[0]
        amp = float(self.opt("jitter", 0.010)) * pl.w
        k = 1.0 - 0.7 * lq
        tear = np.where(hash01(seed + 7, cid, f) < 0.06, 3.0, 1.0)
        ju = (hash01(seed + 5, cid, f) - 0.5) * 2 * amp * k * tear
        jv = (hash01(seed + 6, cid, f) - 0.5) * 2 * amp * 0.35 * k
        moving = (lq < 1.0) | (q < 0.9)
        items = []
        u, v = pl.u, pl.v
        for i in idx:
            if moving[i]:
                off = u * ju[i] + v * jv[i]
                items.append((tab["keys"][i], [P + off for P in tab["polys"][i]]))
            else:
                items.append((tab["keys"][i], tab["polys"][i]))
        lay = consolidate(items)
        if self.opt("slices", True) and 0.12 < q < 0.94 and hash01(seed + 8, f) < 0.42:
            lay = self.sliced(lay, f, seed)
        return lay

    def sliced(self, lay, f, seed):
        pl = self.body.placement
        H, W = pl.h, pl.w
        nb = 1 + int(hash01(seed + 9, f) * 3)
        bands = []
        for k in range(nb):
            y0 = (float(hash01(seed + 10, f, k)) - 0.5) * H * 0.9
            h = (0.025 + 0.10 * float(hash01(seed + 11, f, k))) * H
            dx = (0.03 + 0.11 * float(hash01(seed + 13, f, k))) * W * (1 if hash01(seed + 12, f, k) < 0.5 else -1)
            bands.append((y0, y0 + h, dx))
        bands.sort()
        edges, shifts = [-1e7], [0.0]
        cur = -1e7
        for y0, y1, dx in bands:
            if y0 <= cur:
                continue
            edges += [y0, y1]
            shifts += [dx, 0.0]
            cur = y1
        edges.append(1e7)
        vo = float(pl.origin @ pl.v)
        out = []
        for a, bnd, dx in zip(edges[:-1], edges[1:], shifts):
            part = slab_layers(lay, pl.v, vo + a, vo + bnd) if (a > -1e6 or bnd < 1e6) else lay
            if dx:
                part = [shifted(L, pl.u * dx) for L in part]
            out += part
        return out

    def material(self):
        pl = self.body.placement
        sgn = -1.0 if self.side == "in" else 1.0
        return Material.none(pl.origin + self.d * sgn * self.gate, self.d)
