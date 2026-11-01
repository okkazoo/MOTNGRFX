"""G-buffer rasteriser for cam3d Layers: the same transform / near clip / painter order / anti-aliasing as
cam3d.render_layers (so the passes line up with the beauty to the pixel), but it keeps what the beauty throws away:

  A  (h, w)     coverage 0..1 of the type (over-composited, alpha of the layers included)       -> matte
  D  (h, w)     premultiplied camera-space depth (D / A = depth where A > 0)                      -> depth, flow
  N  (h, w, 3)  premultiplied camera-space surface normal (pointing at the viewer)                -> clay
  E  (h, w)     thin outline of the type, hidden lines removed (item-id edges + hairline layers) -> line

Fill layers are planar: depth per pixel comes from the plane through the polygon (exact under perspective). Hairlines / tubes get
the depth of their polyline (a polyline is one depth). Overlap handling is the beauty's: painter order by mean depth."""
import math
import numpy as np
import cv2

from mograph.render import cam3d
from mograph.render.cam3d import clip_poly, clip_line

ol = cam3d.ol
SHIFT, SF = ol.SHIFT, ol._SF


class Item:
    __slots__ = ("key", "L", "polys", "idx")

    def __init__(self, key, L, polys):
        self.key, self.L, self.polys, self.idx = key, L, polys, 0


def collect(layers, cam, size, near=6.0, guard=3000.0):
    """cam3d.render_layers' first half: -> items sorted far -> near. polys: [(xy (n,2) px, Qc (n,3) camera space, closed)]."""
    w, h = size
    k = w / float(cam3d.W)
    F = cam.F * k
    cen = np.array([w / 2.0, h / 2.0])
    R, eye = cam.R, cam.eye
    items = []
    for L in layers:
        if L.pts.shape[0] == 0:
            continue
        Q = (L.pts - eye) @ R.T
        zc = Q[:, 2]
        if zc.max() < near:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xy_all = cen + F * Q[:, :2] / zc[:, None]
        npoly = len(L.polys)
        groups = [(i, i + 1) for i in range(npoly)] if L.split else [(0, npoly)]
        for g0, g1 in groups:
            a, b = L.offs[g0], L.offs[g1]
            key = float(zc[a:b].mean()) - L.bias
            polys = []
            for pi in range(g0, g1):
                o0, o1 = L.offs[pi], L.offs[pi + 1]
                z = zc[o0:o1]
                if z.min() >= near:
                    pieces = [(xy_all[o0:o1], Q[o0:o1], L.closed or L.kind == "fill")]
                else:
                    Qp = Q[o0:o1]
                    pieces = []
                    if L.kind == "fill":
                        c = clip_poly(Qp, Qp[:, 2] - near)
                        if c is not None:
                            pieces = [(cen + F * c[:, :2] / c[:, 2:3], c, True)]
                    else:
                        for c in clip_line(Qp, Qp[:, 2] - near, L.closed):
                            pieces.append((cen + F * c[:, :2] / c[:, 2:3], c, False))
                for xy, Qc, cl in pieces:
                    lo, hi = xy.min(0), xy.max(0)
                    if hi[0] < 0 or hi[1] < 0 or lo[0] > w or lo[1] > h:
                        continue
                    if L.kind == "fill" and (lo.min() < -guard or hi[0] > w + guard or hi[1] > h + guard):
                        xyq = np.hstack([xy, Qc])                          # clip the screen xy AND the camera-space points together (same vertices)
                        for ax, sg, lim in ((0, 1, -guard), (0, -1, w + guard), (1, 1, -guard), (1, -1, h + guard)):
                            if xyq is None:
                                break
                            xyq = clip_poly(xyq, sg * (xyq[:, ax] - lim))
                        if xyq is None:
                            continue
                        xy, Qc = xyq[:, :2], xyq[:, 2:5]
                    polys.append((xy, Qc, cl))
            if polys:
                items.append(Item(key, L, polys))
    order = sorted(range(len(items)), key=lambda i: -round(items[i].key, 2))
    out = [items[i] for i in order]
    for n, it in enumerate(out):
        it.idx = n + 1
    return out


def _pts(xy, org, S):
    return np.round((xy - org) * S * SF).astype(np.int32)


def _plane(polys):
    """camera-space plane (n, d) with n facing the camera, through the item's points; None when not planar enough."""
    Q = np.concatenate([p[1] for p in polys])
    if len(Q) < 3:
        return None
    c = Q.mean(0)
    X = Q - c
    try:
        _, sv, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    n = vt[2]
    ext = max(float(sv[0]), 1e-6)
    if np.abs(X @ n).max() > 0.01 * ext + 0.3 or sv[1] < 1e-3 * ext:
        return None
    if n @ c > 0:
        n = -n
    return n, float(n @ c)


class GBuffer:
    def __init__(self, w, h, S, want_lines):
        self.w, self.h, self.S = w, h, S
        self.A = np.zeros((h, w), np.float32)
        self.D = np.zeros((h, w), np.float32)
        self.N = np.zeros((h, w, 3), np.float32)
        self.ids = np.zeros((h * S, w * S), np.int32) if want_lines else None
        self.lm = np.zeros((h * S, w * S), np.uint8) if want_lines else None
        self.E = None
        self.n_items = 0

    def over(self, x0, y0, x1, y1, cov, depth, nrm):
        sl = (slice(y0, y1), slice(x0, x1))
        a = cov
        ia = 1.0 - a
        self.D[sl] = depth * a + self.D[sl] * ia
        self.N[sl] = nrm * a[..., None] + self.N[sl] * ia[..., None]
        self.A[sl] = a + self.A[sl] * ia

    def depth(self):
        return np.where(self.A > 1e-4, self.D / np.maximum(self.A, 1e-4), 0.0)

    def normal(self):
        n = self.N / np.maximum(self.A, 1e-4)[..., None]
        return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)


def _bbox(polys, w, h, pad=1):
    lo = np.min([p[0].min(0) for p in polys], 0)
    hi = np.max([p[0].max(0) for p in polys], 0)
    x0, y0 = max(int(math.floor(lo[0])) - pad, 0), max(int(math.floor(lo[1])) - pad, 0)
    x1, y1 = min(int(math.ceil(hi[0])) + pad, w), min(int(math.ceil(hi[1])) + pad, h)
    return x0, y0, x1, y1


def rasterize(layers, cam, size, S=2, near=6.0, lw_clamp=(0.35, 4.0), want_lines=False, line_px=1.5, items=None):
    """-> GBuffer. layers: cam3d Layers (e.g. seq.layers(t)); cam: cam3d.Camera in WORLD focal units (scaled like Sequence._draw)."""
    w, h = size
    k = w / float(cam3d.W)
    F = cam.F * k
    G = GBuffer(w, h, S, want_lines)
    items = collect(layers, cam, size, near) if items is None else items
    G.n_items = len(items)
    cx, cy = w / 2.0, h / 2.0
    ylin = (np.arange(h, dtype=np.float32) + 0.5 - cy) / F
    xlin = (np.arange(w, dtype=np.float32) + 0.5 - cx) / F
    th_line = max(1, int(round(line_px * S)))
    for it in items:
        L = it.L
        alpha = L.alpha
        if alpha <= 0.002:
            continue
        if L.kind == "fill":
            polys = it.polys
            x0, y0, x1, y1 = _bbox(polys, w, h)
            if x1 <= x0 or y1 <= y0:
                continue
            org = np.array([x0, y0], float)
            m = np.zeros(((y1 - y0) * S, (x1 - x0) * S), np.uint8)
            if L.union and len(polys) > 1:
                for p in polys:
                    if len(p[0]) > 2:
                        cv2.fillPoly(m, [_pts(p[0], org, S)], 255, cv2.LINE_AA, SHIFT)
            else:
                cv2.fillPoly(m, [_pts(p[0], org, S) for p in polys if len(p[0]) > 2], 255, cv2.LINE_AA, SHIFT)
            cov = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) * (alpha / 255.0)
            if G.ids is not None and alpha >= 0.25:
                hs = (slice(y0 * S, y1 * S), slice(x0 * S, x1 * S))
                mb = m > 127
                G.ids[hs][mb] = it.idx
                G.lm[hs][mb] = 0
            pl = _plane(polys)
            if pl is None:
                zm = float(np.concatenate([p[1][:, 2] for p in polys]).mean())
                depth = np.float32(zm)
                nrm = np.array([0.0, 0.0, -1.0], np.float32)
            else:
                n, d = pl
                den = n[0] * xlin[None, x0:x1] + n[1] * ylin[y0:y1, None] + n[2]
                den = np.where(np.abs(den) < 1e-4, -1e-4, den)
                depth = np.clip(d / den, near, 1e6).astype(np.float32)
                nrm = n.astype(np.float32)
            G.over(x0, y0, x1, y1, cov, depth, nrm)
        else:
            pls = it.polys
            ordered = sorted(pls, key=lambda p: -float(p[1][:, 2].mean()))
            if len(ordered) > 3:                               # many hairlines: depth buckets (12), far -> near
                step = int(math.ceil(len(ordered) / 12.0))
                chunks = [ordered[i:i + step] for i in range(0, len(ordered), step)]
            else:
                chunks = [[p] for p in ordered]
            for ch in chunks:
                zm = float(np.mean([p[1][:, 2].mean() for p in ch]))
                x0, y0, x1, y1 = _bbox(ch, w, h, pad=int(math.ceil(L.width * lw_clamp[1])) + 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                org = np.array([x0, y0], float)
                sc = min(max(F / max(zm, 1e-6), lw_clamp[0]), lw_clamp[1])
                th = max(1, int(round(L.width * sc * S)))
                m = np.zeros(((y1 - y0) * S, (x1 - x0) * S), np.uint8)
                for cl in (True, False):
                    sub = [_pts(p[0], org, S) for p in ch if p[2] == cl and len(p[0]) > 1]
                    if sub:
                        cv2.polylines(m, sub, cl, 255, th, cv2.LINE_AA, SHIFT)
                cov = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) * (alpha / 255.0)
                G.over(x0, y0, x1, y1, cov, np.float32(zm), np.array([0.0, 0.0, -1.0], np.float32))
                if G.lm is not None and alpha >= 0.25:
                    m2 = np.zeros_like(m)
                    for cl in (True, False):
                        sub = [_pts(p[0], org, S) for p in ch if p[2] == cl and len(p[0]) > 1]
                        if sub:
                            cv2.polylines(m2, sub, cl, 255, th_line, cv2.LINE_AA, SHIFT)
                    hs = (slice(y0 * S, y1 * S), slice(x0 * S, x1 * S))
                    G.lm[hs] = np.maximum(G.lm[hs], m2)
    if G.ids is not None:
        G.E = _edges(G, line_px)
    return G


def _edges(G, line_px):
    S = G.S
    ids = G.ids
    e = np.zeros(ids.shape, np.uint8)
    e[:, :-1] |= (ids[:, :-1] != ids[:, 1:]).astype(np.uint8)
    e[:-1, :] |= (ids[:-1, :] != ids[1:, :]).astype(np.uint8)
    e *= 255
    r = int(round((line_px * S - 1.0) / 2.0))
    if r > 0:
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    e = np.maximum(e, G.lm)
    return cv2.resize(e, (G.w, G.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
