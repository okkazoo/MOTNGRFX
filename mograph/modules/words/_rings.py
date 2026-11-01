"""Shared engine of the ring-stream word modules (outflow, fly_into). The word is a stream of outline rings flowing along a
shaft axis: a train of NR copies of the word's outline (each cut into arcs = strands) travels at constant speed from the
entry gate, along the shaft axis, through the word's place (the lip) and out toward the exit side, bending from the entry
axis to the exit axis there. The Bundle hand-off is the train itself: at t = 0 the rings are lined up at the entry gate
(the arcs of the first rings are the entry strands), at t = duration the last rings are at the exit gate. While the train
passes the lip, the word stands (a lip outline fades in under the first ring and fades out after the last) and the stream
never stops. Everything is real 3D: the shaft axis is the entry side tilted toward the depth (`tilt`), so a camera sees the
rings recede and grow by perspective; entry 'in' / exit 'out' give the straight tunnel of the original pieces.
Params: rings (NR), tilt, depth (not used), speed (px/s), tail_len."""
import math
import numpy as np
from mograph.modules.core import (WordModule, Bundle, Layer, register, layout_word, unit, ol, mixc, Route3)
from mograph.modules.words._strand import make_arcs, _ss
from mograph.modules.words._cb import ss

K_ARC = 22


def resample_arc(a, K):
    cum = ol.arclen(a)
    s = np.linspace(0, cum[-1], K)
    return np.stack([np.interp(s, cum, a[:, 0]), np.interp(s, cum, a[:, 1])], 1)


class RingStream(WordModule):
    NR = 16
    TILT = 0.75               # depth share of the shaft axis for in-plane entry / exit sides
    ARC_PER_CAP = 0.9
    LINE_W = 1.1
    LIP_W = 1.5

    def build(self):
        P, pl = self.params, self.placement
        self.lay = layout_word(self.font, self.lines, pl.w * 0.94, pl.h * 0.9)
        cap = self.lay["cap"]
        self.cap = cap
        cons = [c for l in self.lay["letters"] for c in l["cons"]]
        self.cons2 = cons
        self.cons3 = [pl.lift(c) - pl.origin for c in cons]           # relative to the placement origin
        perim = sum(float(ol.arclen(np.vstack([c, c[:1]]))[-1]) for c in cons)
        Na = int(P.get("n_arcs") or np.clip(round(perim / (self.ARC_PER_CAP * cap)), 10, 44))
        arcs2 = [resample_arc(a, K_ARC) for a in make_arcs(cons, Na)]
        self.Na = len(arcs2)
        A2 = np.array(arcs2)
        self.A3 = A2[..., 0:1] * pl.u + A2[..., 1:2] * pl.v           # (Na, K, 3) relative to the origin
        self.NR = NR = int(P.get("rings", self.NR))
        self.N = NR * self.Na
        self.w = float(P.get("width") or self.LINE_W * max(0.8, cap / 200.0))
        # shaft axes
        tilt = float(P.get("tilt", self.TILT))
        s_in, s_out = self.entry.side(pl), self.exit.side(pl)
        self.a_in = unit(s_in + tilt * pl.n * (1.0 - abs(s_in @ pl.n)))
        self.a_out = unit(s_out - tilt * pl.n * (1.0 - abs(s_out @ pl.n)))
        self.d_in, self.d_out = -self.a_in, self.a_out                  # directions of travel
        gap = 0.06 * max(pl.w, pl.h) + 12.0
        ext = lambda a: 0.5 * (abs(a @ pl.u) * pl.w + abs(a @ pl.v) * pl.h) + 0.0 * abs(a @ pl.n)
        self.Lt = ext(self.a_in) + gap + 0.25 * cap
        self.Lx = ext(self.a_out) + gap + 0.25 * cap
        tm = self.timing
        T = tm.total
        t_first = 0.55 * tm.form
        self.V = float(P.get("speed") or self.Lt / t_first)
        t_first = self.Lt / self.V
        t_last = T - self.Lx / self.V
        if t_last - t_first < 0.3:
            t_last = t_first + 0.3
        self.t_first, self.t_last = t_first, t_last
        self.G = self.V * (t_last - t_first) / max(NR - 1, 1)
        self.sig0 = -self.Lt - np.arange(NR) * self.G
        self._route_table()
        self.ring_style_init()
        self._bundles()

    # ---- the path of the ring centres (offsets from the origin)
    def _route_table(self):
        tin, tout = self.d_in, self.d_out
        Lb = max(0.5 * self.Lx, 60.0)
        ds = 4.0
        smax = self.sig0[0] + self.V * (self.timing.total + 0.6) + 50.0
        n = int(max(smax, self.Lx + Lb) / ds) + 2
        sg = np.arange(n) * ds
        u = np.clip(sg / Lb, 0, 1)
        w = u * u * (3 - 2 * u)
        dirs = unit((1 - w)[:, None] * tin[None] + w[:, None] * tout[None])
        pos = np.vstack([[0, 0, 0], np.cumsum(dirs[:-1] * ds, axis=0)])
        self.tab_s, self.tab_p = sg, pos

    def offsets(self, sig):
        """3D offset of the ring centre for route parameter sig (array): behind the lip along a_in, ahead along the bend."""
        sig = np.asarray(sig, float)
        beh = (-sig)[:, None] * self.a_in[None]
        ahead = np.stack([np.interp(sig, self.tab_s, self.tab_p[:, k]) for k in range(3)], 1)
        # past the table: straight on
        over = np.maximum(sig - self.tab_s[-1], 0.0)[:, None] * self.d_out[None]
        return np.where((sig < 0)[:, None], beh, ahead + over)

    def sig(self, t):
        return self.sig0 + self.V * t

    def ring_style_init(self):
        pass

    def ring_colours(self, sig):
        """(NR,3) colours per ring"""
        raise NotImplementedError

    # ---- bundles (the ring arcs at t = 0 / t = duration)
    def strands(self, t, orient):
        off = self.offsets(self.sig(t))                                 # (NR,3)
        A = self.A3
        P = self.placement.origin + A[None] + off[:, None, None, :]     # (NR, Na, K, 3)
        P = P.reshape(-1, K_ARC, 3)
        d = orient
        flip = (P[:, 0] - P[:, -1]) @ d < 0
        P = np.where(flip[:, None, None], P[:, ::-1], P)
        return P

    def _bundles(self):
        for name, t, d in (("_entry", 0.0, self.d_in), ("_exit", self.duration, self.d_out)):
            P = self.strands(t, d)
            col = np.repeat(self.ring_colours(self.sig(t)), self.Na, axis=0)
            setattr(self, name, Bundle(P, np.tile(d, (self.N, 1)), col, self.w, t, self.V, 0.0))

    def entry_bundle(self):
        return self._entry

    def exit_bundle(self):
        return self._exit

    # ---- frame
    def lip_alpha(self, t):
        return ss((t - (self.t_first - 0.12)) / 0.2) * (1.0 - ss((t - self.t_last) / 0.25))

    def fill_layers(self, t, sig, off):
        return []

    def layers(self, t):
        t = max(t, 0.0)
        pl = self.placement
        sig = self.sig(t)
        off = self.offsets(sig)
        out = self.fill_layers(t, sig, off)
        cols = self.ring_colours(sig)
        P = pl.origin + self.A3[None] + off[:, None, None, :]
        for r in range(self.NR):
            c = tuple(int(x) for x in cols[r])
            out.append(Layer(list(P[r]), c, "line", self.w, closed=False, bias=0.3 + 0.001 * r))
        a = self.lip_alpha(t)
        if a > 0.004:
            out.append(Layer([c3 + pl.origin for c3 in self.cons3], self.lip_colour(), "line", self.LIP_W * max(0.8, self.cap / 200.0),
                             closed=True, alpha=a, bias=2.0))
        return out

    def lip_colour(self):
        return self.palette.fg

    def focus(self, t):
        t = min(max(t, 0.0), self.duration)
        sig = self.sig(t)
        m = (sig > -self.Lt) & (sig < self.Lx)
        if not m.any():
            return self.placement.origin.copy()
        return (self.placement.origin + self.offsets(sig[m])).mean(0)

    def camera_hint(self, t, dist=900.0):
        """(eye, target) of a camera that flies down the shaft into the word (eye slides along the entry axis from dist to
        0.15 dist while the word forms, then rests; pulls back out along the exit axis)."""
        pl = self.placement
        tm = self.timing
        u = ss(t / tm.form) if t < tm.form else 1.0 - ss((t - tm.form - tm.hold) / tm.exit)
        k = 1.0 - 0.8 * u
        return pl.origin - pl.n * dist * k, pl.origin.copy()

    def hold_pose(self, dist=900.0):
        pl = self.placement
        return pl.origin - pl.n * dist, pl.origin.copy()
