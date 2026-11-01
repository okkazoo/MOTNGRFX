"""blobs: the word grows out of soft merging blobs, and melts away into blobs on the way out.
Reveal: small round blobs stream in along the entry lane (stretched by their speed like drops), settle on the strokes of the
type, swell until they merge (a smooth union of discs: log-sum-exp metaballs on the placement plane), the merged mass then
sharpens into the glyphs (its signed distance field is blended into the glyph's own) and hands over to the body.
Conceal: the glyphs swell into a blobby mass, the mass breaks into discs that shrink and stream out along the exit lane.
Material: `blobs` (centres, radii; a transition sees each as a capsule strand). Options:
  max_blobs (64)  r0 (staging radius px, 2.0)  soft (merge softness, fraction of the mean radius, 0.28)  seed (1)
  solid (None = auto: the body has an opaque ink fill, so the mass stays opaque while the body fades in over it)"""
import math
import numpy as np
import contourpy

from mograph.modules.core import Layer
from mograph.modules.material import Material
from mograph.modules.io.base import (IOStyle, register_style, side_frame, stage_ribbon, Flight, hash01, ss, consolidate,
                                     fade, seg, fit_entering, fit_leaving)

BIG = 60.0


@register_style("blobs")
class Blobs(IOStyle):
    description = ("The word grows out of soft blobs: round drops stream in, settle on the strokes, swell and merge into one "
                   "mass that sharpens into the letters. On exit the letters swell into blobs that break apart, shrink and "
                   "stream away.")
    material_kind = "blobs"

    def attach(self, body, side):
        super().attach(body, side)
        self._prep = None

    # ------------------------------------------------------------ preparation
    def prep(self):
        if self._prep is not None:
            return self._prep
        b, pl, tm = self.body, self.body.placement, self.body.timing
        side = self.side
        d, e1, e2, gate = side_frame(b, side)
        sh = b.hold_shape()
        C2, dep = sh.seeds(int(self.opt("max_blobs", 64)))
        if len(C2) == 0:                                        # nothing deep enough: one blob in the middle
            C2, dep = np.array([[0.0, 0.0]]), np.array([0.2 * pl.h])
        Rf = dep * 1.12
        C3 = pl.lift(C2)
        n = len(C2)
        seed = int(self.opt("seed", 1))
        r0 = float(self.opt("r0", 2.0))
        spacing = max(2 * r0 + 8.0, 15.0)
        rel = C3 - pl.origin
        lat, along = rel @ e1, rel @ d
        frac = (along - along.min()) / max(along.max() - along.min(), 1e-6)
        st = stage_ribbon(b, side, lat, spacing)
        raw = b._rawcall("entry_bundle" if side == "in" else "exit_bundle")
        v = max(float(raw.v.mean()), 150.0)
        Lc0 = 2.6 * r0 + 6.0
        T = b.style_window(side)[1]
        hr = hash01(seed, np.arange(n))
        wob = (hash01(seed + 5, np.arange(n)) - 0.5) * 0.18
        flights, t0s, A = [], [], []
        if side == "in":
            dl = 0.08 * hr                                       # departure delay (fraction of the window)
            arr = 0.30 + 0.27 * frac                             # arrival at the seed
            for j in range(n):
                dl[j], fl = fit_entering(st["S"][j], C3[j], d, v, T, dl[j], arr[j], e1, wob[j])
                flights.append(fl)
            self.dl, self.arr = dl, arr
        else:
            dep_p = 0.40 + 0.24 * (1.0 - frac)                   # leave: front of the mass first
            for j in range(n):
                dep_p[j], fl = fit_leaving(C3[j], st["S"][j], d, v, T, dep_p[j], e1, wob[j])
                flights.append(fl)
            self.dep_p = dep_p
        base = b.reveal_layers(0.0) if side == "in" else b.conceal_layers(1.0)
        fg = np.array(b.palette.fg, float)
        solid = self.opt("solid", None)
        if solid is None:                      # the body draws an opaque fill in the ink colour: the mass never needs to fade
            solid = any(L.kind == "fill" and L.alpha >= 0.9 and np.abs(np.array(L.color) - fg).max() < 40 for L in base)
        self._prep = dict(solid=bool(solid), C2=C2, C3=C3, Rf=Rf, n=n, st=st, flights=flights, v=v, r0=r0, Lc0=Lc0, d=d, e1=e1, T=T, k=Lc0 / v,
                          frac=frac, k_soft=max(3.0, float(self.opt("soft", 0.28)) * float(Rf.mean())))
        return self._prep

    # ------------------------------------------------------------ pieces
    def radius_in(self, P, j, u):
        return P["r0"] + (0.45 * P["Rf"][j] - P["r0"]) * ss(u)

    def capsule(self, P, pos, tang, speed, r):
        L = max(P["k"] * speed, 0.8)
        return (("line", self._col(), round(2 * r * 2) / 2, 1.0, False, False, 0.25), [np.array([pos - tang * L, pos])])

    def _col(self):
        return tuple(int(c) for c in self.body.palette.fg)

    def field_layers(self, centres2, radii, m, alpha):
        """smooth union of the discs blended into the glyph SDF by m -> one even-odd fill Layer."""
        P = self.prep()
        g = self.body.hold_shape().grid()
        xs, ys, sdf = g["xs"], g["ys"], g["sdf"]
        nx, ny, step = len(xs), len(ys), g["step"]
        k = P["k_soft"]
        E = np.zeros((ny, nx), np.float32)
        for (cx, cy), r in zip(centres2, radii):
            rr = r + 9.0 * k
            i0, i1 = max(0, int((cx - rr - xs[0]) / step)), min(nx, int((cx + rr - xs[0]) / step) + 2)
            j0, j1 = max(0, int((cy - rr - ys[0]) / step)), min(ny, int((cy + rr - ys[0]) / step) + 2)
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(xs[i0:i1] - cx, ys[j0:j1] - cy)
            E[j0:j1, i0:i1] += np.exp(-(np.hypot(X, Y) - r) / k)
        F = np.where(E > 1e-26, -k * np.log(np.maximum(E, 1e-26)), BIG * k).astype(np.float32)
        if m > 0:
            F = (1.0 - m) * F + m * sdf
        F[0, :] = F[-1, :] = F[:, 0] = F[:, -1] = 1.0
        lines = contourpy.contour_generator(x=xs, y=ys, z=F).lines(0.0)
        polys = [np.asarray(l)[:-1] for l in lines if len(l) > 4]
        if not polys:
            return []
        return [Layer(polys, self._col(), "fill", plane=self.body.placement.plane, alpha=alpha, bias=0.0)]

    # ------------------------------------------------------------ frame
    def layers(self, t, p):
        return self.reveal(t, p) if self.side == "in" else self.conceal(t, p)

    def positions(self, p):
        """(pos (n,3), arrived mask) of every blob at window progress p (reveal)."""
        P = self.prep()
        T = P["T"]
        pos = np.zeros((P["n"], 3))
        for j, fl in enumerate(P["flights"]):
            if p < self.dl[j]:
                pos[j] = P["st"]["S"][j] + P["d"] * P["v"] * p * T
            else:
                pos[j] = fl.at(min(1.0, (p - self.dl[j]) * T / fl.T))[0]
        return pos

    def reveal(self, t, p):
        P = self.prep()
        T, n, d = P["T"], P["n"], P["d"]
        p = max(p, 0.0)
        out, items = [], []
        field_c, field_r = [], []
        for j, fl in enumerate(P["flights"]):
            if p < self.arr[j]:
                if p < self.dl[j]:
                    pos, tang, sp, r = P["st"]["S"][j] + d * P["v"] * p * T, d, P["v"], P["r0"]
                else:
                    u = (p - self.dl[j]) * T / fl.T
                    pos, tang = fl.at(u)
                    sp, r = fl.speed(u), self.radius_in(P, j, u)
                items.append(self.capsule(P, pos, tang, sp, r))
            else:
                r_arr = 0.45 * P["Rf"][j]
                gj = ss((p - self.arr[j]) / max(0.82 - self.arr[j], 1e-3))
                field_c.append(P["C2"][j]); field_r.append(r_arr + (P["Rf"][j] - r_arr) * gj)
        m = ss((p - 0.62) / 0.26)
        h = ss((p - 0.86) / 0.14)
        if field_c:
            out += self.field_layers(field_c, field_r, m, 1.0 if P["solid"] else 1.0 - h)
        out += consolidate(items)
        if h > 0.0:
            out += fade(self.body.reveal_layers(0.0, t), h)
        return out

    def conceal(self, t, p):
        P = self.prep()
        n, T = P["n"], P["T"]
        out, items = [], []
        field_c, field_r = [], []
        for j, fl in enumerate(P["flights"]):
            if p < self.dep_p[j]:
                field_c.append(P["C2"][j]); field_r.append(P["Rf"][j])
            else:
                u = (p - self.dep_p[j]) * T / fl.T
                pos, tang = fl.at(u)
                e = ss(min(u, 1.0))
                r = P["Rf"][j] * (1 - e) + P["r0"] * e
                items.append(self.capsule(P, pos, tang, fl.speed(u) if u < 1.0 else fl.v, r))
        h = 1.0 - ss(p / 0.12)                                   # body -> solid glyph
        m = 1.0 - ss((p - 0.08) / 0.28)                          # glyph -> blobby mass
        if field_c:
            out += self.field_layers(field_c, field_r, m, 1.0 if P["solid"] else (1.0 - h))
        out += consolidate(items)
        if h > 0.0:
            out += fade(self.body.conceal_layers(1.0, t), h)
        return out

    def focus(self, t, p):
        P = self.prep()
        if self.side == "in" and p < 0.6:
            return self.positions(max(p, 0.0)).mean(0)
        return self.body.placement.origin.copy()

    # ------------------------------------------------------------ material
    def material(self):
        P = self.prep()
        n = P["n"]
        if self.side == "in":
            S = P["st"]["S"]
        else:
            S = np.array([fl.at(1.0)[0] for fl in P["flights"]])
        return Material.blobs(S, np.full(n, P["r0"]), np.tile(P["d"], (n, 1)), np.tile(self.body.palette.fg, (n, 1)), P["v"],
                              length=P["Lc0"], t=0.0 if self.side == "in" else self.body.duration)
