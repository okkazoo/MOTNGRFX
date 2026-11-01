"""Shape: the formed word as plane geometry (what `WordModule.hold_shape()` returns), in placement coordinates (x along u,
y along v = down). Gives the styles what they need without knowing the module: contours per letter, a signed distance field
on a regular grid (blobs), interior seed discs (blobs) and outline samples (particles)."""
import math
import numpy as np
import cv2


class Shape:
    def __init__(self, pl, letters, cap):
        self.pl = pl
        self.letters = [[np.asarray(c, float) for c in l if len(c) > 2] for l in letters]
        self.letters = [l for l in self.letters if l]
        self.cons = [c for l in self.letters for c in l]
        self.cap = float(cap)
        P = np.concatenate(self.cons)
        self.bbox = (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))
        self._grid = None

    # ------------------------------------------------------------ construction
    @classmethod
    def from_body(cls, body):
        pl = body.placement
        lay = getattr(body, "lay", None)
        if lay is not None:
            return cls(pl, [l["cons"] for l in lay["letters"]], lay["cap"])
        # fallback: project the closed polygons the body draws at the start of its hold onto the placement plane
        cons = []
        for L in body.body_layers(body.body_time_in()):
            if L.kind != "fill" and not L.closed:
                continue
            for P in L.polys:
                rel = P - pl.origin
                cons.append(np.stack([rel @ pl.u, rel @ pl.v], 1))
        if not cons:
            cons = [np.array([[-pl.w / 2, -pl.h / 2], [pl.w / 2, -pl.h / 2], [pl.w / 2, pl.h / 2], [-pl.w / 2, pl.h / 2]])]
        return cls(pl, [cons], 0.4 * pl.h)

    # ------------------------------------------------------------ signed distance field
    def grid(self, step=1.5, pad=34.0, sub=3):
        """dict(xs (nx,), ys (ny,), sdf (ny,nx)): signed distance to the glyph outline in px (negative inside) at node spacing
        `step`, covering the word's bbox + pad. Letters are filled even-odd per letter and OR-ed."""
        if self._grid is not None and self._grid["step"] == step:
            return self._grid
        x0, y0 = self.bbox[0] - pad, self.bbox[1] - pad
        nx = int(math.ceil((self.bbox[2] + pad - x0) / step)) + 1
        ny = int(math.ceil((self.bbox[3] + pad - y0) / step)) + 1
        fs = step / sub
        W, H = (nx - 1) * sub + 1, (ny - 1) * sub + 1
        mask = np.zeros((H, W), np.uint8)
        for l in self.letters:
            m = np.zeros((H, W), np.uint8)
            for c in l:
                tmp = np.zeros((H, W), np.uint8)
                pts = np.round(((c - [x0, y0]) / fs) * 16.0).astype(np.int32)
                cv2.fillPoly(tmp, [pts.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=4)
                m ^= tmp
            mask |= m
        d_out = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
        d_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        sdf = np.where(mask > 0, -(d_in - 0.5), d_out - 0.5) * fs
        sdf = sdf[::sub, ::sub].astype(np.float32)
        self._grid = dict(step=step, x0=x0, y0=y0, xs=x0 + np.arange(sdf.shape[1]) * step, ys=y0 + np.arange(sdf.shape[0]) * step,
                          sdf=sdf)
        return self._grid

    def seeds(self, max_n=80, min_depth=3.5, cover=0.72):
        """interior discs that tile the strokes: greedy by depth, each new seed lies outside `cover` x the radius of every
        accepted one. -> (C (n,2), depth (n,)) ; depth = distance from the seed to the outline (the largest inscribed radius)."""
        g = self.grid()
        sdf = g["sdf"]
        dep = np.maximum(-sdf, 0.0)
        ys, xs = np.nonzero(dep >= min_depth)
        if len(ys) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        d = dep[ys, xs]
        order = np.argsort(-d, kind="stable")
        X = g["xs"][xs[order]]
        Y = g["ys"][ys[order]]
        D = d[order]
        for cv in (cover, cover + 0.12, cover + 0.25, cover + 0.45, cover + 0.8):
            Cs, Ds = [], []
            covered = np.zeros(len(D), bool)
            i = 0
            n = len(D)
            while i < n:
                if covered[i]:
                    i += 1
                    continue
                Cs.append((X[i], Y[i])); Ds.append(D[i])
                covered |= (X - X[i]) ** 2 + (Y - Y[i]) ** 2 < (cv * D[i]) ** 2
                i += 1
            if len(Cs) <= max_n:
                break
        Cs, Ds = np.array(Cs), np.array(Ds)
        if len(Cs) > max_n:                                   # still too many (hairline fonts): keep the deepest
            keep = np.argsort(-Ds, kind="stable")[:max_n]
            Cs, Ds = Cs[keep], Ds[keep]
        return Cs, Ds

    # ------------------------------------------------------------ outline samples
    def outline(self, pitch):
        """points every `pitch` px along every contour. -> (P (M,2), T (M,2) unit tangent, letter (M,) index)."""
        P, T, Lt = [], [], []
        for li, l in enumerate(self.letters):
            for c in l:
                c2 = np.vstack([c, c[:1]])
                seg = np.linalg.norm(np.diff(c2, axis=0), axis=1)
                cum = np.r_[0.0, np.cumsum(seg)]
                n = max(2, int(round(cum[-1] / pitch)))
                s = (np.arange(n) + 0.5) * cum[-1] / n
                x = np.interp(s, cum, c2[:, 0]); y = np.interp(s, cum, c2[:, 1])
                x2 = np.interp((s + 1.0) % cum[-1], cum, c2[:, 0]); y2 = np.interp((s + 1.0) % cum[-1], cum, c2[:, 1])
                t = np.stack([x2 - x, y2 - y], 1)
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
                P.append(np.stack([x, y], 1)); T.append(t); Lt.append(np.full(n, li))
        return np.concatenate(P), np.concatenate(T), np.concatenate(Lt)

    def perimeter(self):
        return float(sum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum() for c in self.cons))
