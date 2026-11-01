"""slats: the word is built from thin bars (slats) of varying thickness. Every slat shoots in along the entry direction as a
speed line (leading edge first, the tail stretching behind), lands in its row, and the slats thicken until they close up
into the solid word; the type fills solid. To leave, the slats open back out of the solid, thin down and slide away along
the exit direction, the ones nearest the exit first. (mechanism: 11_mathtype q2_slatsR2 / 13_camera strips, rewritten for
the Bundle interface: each slat IS a strand of the hand-off bundle, its thickness the strand width)

The slat axis follows the entry direction (rows of slats run along the direction of travel, projected into the placement
plane), so an entry from above gives vertical slats falling in, an entry at 35 degrees gives slats at 35 degrees."""
import math
import numpy as np
from mograph.modules.core import register, Bundle, Layer, unit, mixc
from mograph.modules.words._slab import SlabWord, ssm, eo, ei, hull2


def row_intervals(cons, y):
    """nonzero-winding spans of the glyph outlines on scanline y (variable fonts overlap their contours)."""
    xs = []
    for c in cons:
        a = c
        b = np.roll(c, -1, 0)
        m = (a[:, 1] - y) * (b[:, 1] - y) < 0
        if m.any():
            aa, bb = a[m], b[m]
            x = aa[:, 0] + (y - aa[:, 1]) * (bb[:, 0] - aa[:, 0]) / (bb[:, 1] - aa[:, 1])
            xs += list(zip(x, np.sign(bb[:, 1] - aa[:, 1])))
    xs.sort()
    iv, w, start = [], 0, None
    for x, s in xs:
        w0 = w
        w += s
        if w0 == 0 and w != 0:
            start = x
        elif w0 != 0 and w == 0:
            iv.append((start, x))
    return [v for v in iv if v[1] - v[0] > 1.0]


def _match(iv, ref):
    """the interval of iv that overlaps ref the most (else ref itself)."""
    best, bo = None, 0.0
    for a, b in iv:
        o = min(b, ref[1]) - max(a, ref[0])
        if o > bo:
            best, bo = (a, b), o
    return best if best is not None else ref


@register("slats")
class Slats(SlabWord):
    description = ("The word is built from thin bars of varying thickness that shoot in along the entry direction as speed "
                   "lines, land row by row and thicken until they close up into solid type. To leave, the slats open back "
                   "out of the solid, thin down and slide away along the exit direction.")
    DEMO_WORDS = "SLATS"
    DEFAULT_FONT = "anybody"
    P_IN, P_OUT = 2.5, 2.3
    ROWS_PER_CAP = 30.0

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 3)))
        d = self.d_in
        pin = d - (d @ pl.n) * pl.n
        a3 = unit(pin) if np.linalg.norm(pin) > 0.35 else pl.u
        a2 = np.array([a3 @ pl.u, a3 @ pl.v])
        b2 = np.array([-a2[1], a2[0]])
        self.a2, self.b2 = a2, b2
        R = np.stack([a2, b2])
        rc = [c @ R.T for c in self.cons]
        eta = np.concatenate([c[:, 1] for c in rc])
        e0, e1 = float(eta.min()), float(eta.max())
        rows = max(6, int(round((e1 - e0) / max(3.5, self.cap / float(P.get("rows", self.ROWS_PER_CAP))))))
        pitch = (e1 - e0) / rows
        self.pitch, self.hfull = pitch, 1.04 * pitch
        o = self.origin = pl.origin
        a3w = a2[0] * pl.u + a2[1] * pl.v
        b3w = b2[0] * pl.u + b2[1] * pl.v
        self.head_in = "B" if (a3w @ self.d_in) >= -1e-9 else "A"
        self.head_out = "B" if (a3w @ self.d_out) >= -1e-9 else "A"
        eps = 0.03 * pitch
        sl = []
        for r in range(rows):
            y = e0 + pitch * (r + 0.5)
            s = 0.5 + 0.5 * math.sin(2 * math.pi * r / 11.0 + 0.7)
            hthin = pitch * (0.16 + 0.42 * s * s * (3 - 2 * s))
            for (a, b) in row_intervals(rc, y):
                ivt = row_intervals(rc, y - pitch / 2 + eps)
                ivb = row_intervals(rc, y + pitch / 2 - eps)
                ta, tb = _match(ivt, (a, b))
                ba, bb = _match(ivb, (a, b))
                sl.append(dict(eta=y, Ac=0.5 * (ta + ba), Ad=0.5 * (ba - ta), Bc=0.5 * (tb + bb), Bd=0.5 * (bb - tb),
                               hthin=hthin, ca=a, cb=b, lag=0.12 + 0.14 * rng.rand(), jit=rng.rand(), jit2=rng.rand()))
        self.sl = sl
        self.N = len(sl)
        # rest positions along d_in / d_out
        rin, rout = [], []
        for s in sl:
            pa = pl.lift(s["ca"] * a2 + s["eta"] * b2)
            pb = pl.lift(s["cb"] * a2 + s["eta"] * b2)
            s["pin"] = ((pa - o) @ self.d_in, (pb - o) @ self.d_in)
            s["pout"] = ((pa - o) @ self.d_out, (pb - o) @ self.d_out)
            rin.append(0.5 * sum(s["pin"]))
            rout.append(0.5 * sum(s["pout"]))
        rin, rout = np.array(rin), np.array(rout)
        fin = 1.0 - (rin - rin.min()) / max(np.ptp(rin), 1e-6)         # 0 = lands first (farthest along d_in)
        fout = 1.0 - (rout - rout.min()) / max(np.ptp(rout), 1e-6)     # 0 = leaves first (nearest the exit)
        tm = self.timing
        ext = 0.5 * max(pl.w, pl.h)
        for i, s in enumerate(sl):
            s["Ti"] = max(0.35, (tm.form - 0.38)) * (0.55 + 0.40 * fin[i] + 0.05 * s["jit"])
            s["Di"] = max(s["pin"]) + self.reach_in + 0.10 * ext * s["jit2"]
            s["dx"] = 0.42 * tm.exit * (0.9 * fout[i] + 0.1 * s["jit"])
            s["Tx"] = tm.exit - s["dx"]
            s["Dx"] = self.reach_out - min(s["pout"]) + 0.10 * ext * s["jit"]
        self.ex, self.ey = -0.012 * self.cap, 0.020 * self.cap
        self.side_col = mixc(self.palette.bg, self.palette.fg, 0.38)
        self.b2w = b3w

    # ------------------------------------------------------------ state of one slat
    def state(self, s, t):
        tm, pl = self.timing, self.placement
        if t < tm.form + tm.hold:
            tau = t / s["Ti"]
            if tau >= 1.0:
                dh = dt = 0.0
            else:
                lag = s["lag"]
                e1 = eo(tau, self.P_IN)
                e2 = eo(tau - lag * math.sin(math.pi * tau) ** 2, self.P_IN)
                dh, dt = s["Di"] * (1 - e1), s["Di"] * (1 - e2)
            vec, head = -self.d_in, self.head_in
            th = s["hthin"] * (1 - 0.45 * math.sin(math.pi * min(tau, 1.0))) + (self.hfull - s["hthin"]) * ssm(t, s["Ti"], s["Ti"] + 0.28)
            sf = ssm(tau, 0.0, 0.3)
        else:
            te = t - tm.form - tm.hold
            tau = (te - s["dx"]) / s["Tx"]
            if tau <= 0:
                dh = dt = 0.0
            elif tau >= 1.0:
                dh = dt = s["Dx"] * (1 + self.P_OUT * (tau - 1.0))
            else:
                lag = s["lag"]
                dh = s["Dx"] * ei(tau, self.P_OUT)
                dt = s["Dx"] * ei(tau - lag * math.sin(math.pi * tau) ** 2, self.P_OUT)
            vec, head = self.d_out, self.head_out
            fly = math.sin(math.pi * min(max(tau, 0.0), 1.0))
            th = (s["hthin"] + (self.hfull - s["hthin"]) * (1 - ssm(te, s["dx"], s["dx"] + 0.28))) * (1 - 0.45 * fly)
            sf = 1.0 - ssm(tau, 0.7, 1.0) if tau > 0 else 1.0
        dA, dB = (dh, dt) if head == "A" else (dt, dh)
        return dict(vec=vec, dA=dA, dB=dB, th=th, sf=sf, tau=tau)

    def _pt(self, xi, eta, vec, dist):
        return self.placement.lift(xi * self.a2 + eta * self.b2) + vec * dist

    def polys(self, s, st):
        pl = self.placement
        hh = 0.5 * st["th"]
        sq = hh / (0.5 * self.pitch)
        vec = st["vec"]
        At = self._pt(s["Ac"] - sq * s["Ad"], s["eta"] - hh, vec, st["dA"])
        Ab = self._pt(s["Ac"] + sq * s["Ad"], s["eta"] + hh, vec, st["dA"])
        Bt = self._pt(s["Bc"] - sq * s["Bd"], s["eta"] - hh, vec, st["dB"])
        Bb = self._pt(s["Bc"] + sq * s["Bd"], s["eta"] + hh, vec, st["dB"])
        face = np.array([At, Bt, Bb, Ab])
        o3 = (self.ex * pl.u + self.ey * pl.v) * st["sf"]
        if (self.ex * self.b2[0] + self.ey * self.b2[1]) >= 0:
            side = np.array([Ab, Bb, Bb + o3, Ab + o3])
        else:
            side = np.array([At, Bt, Bt + o3, At + o3])
        return face, side

    # ------------------------------------------------------------ bundle / frames
    def bundle_at(self, t):
        K = 8
        N = self.N
        Pm = np.zeros((N, K, 3))
        D = np.zeros((N, 3))
        v = np.zeros(N)
        w = np.zeros(N)
        pl = self.placement
        f = np.linspace(0, 1, K)[:, None]
        entry = t < self.timing.form + self.timing.hold
        for i, s in enumerate(self.sl):
            st = self.state(s, t)
            pa = self._pt(s["ca"], s["eta"], st["vec"], st["dA"])
            pb = self._pt(s["cb"], s["eta"], st["vec"], st["dB"])
            head, tail = (pb, pa) if (self.head_in if entry else self.head_out) == "B" else (pa, pb)
            Pm[i] = head + (tail - head) * f
            D[i] = self.d_in if entry else self.d_out   # travel direction (st["vec"] is the entry *offset*, -d_in)
            w[i] = st["th"]
            v[i] = (self.P_IN * s["Di"] / s["Ti"]) if entry else (self.P_OUT * s["Dx"] / s["Tx"])
        col = np.tile(np.array(self.palette.fg, float), (N, 1))
        return Bundle(Pm, D, col, w, t, v, 0.0)

    def layers(self, t):
        tm = self.timing
        ph, lt = self.phase(t)
        L = []
        if ph == "form":
            a = ssm(t, tm.form - 0.45, tm.form - 0.08)
        elif ph == "hold":
            a = 1.0
        else:
            a = 1.0 - ssm(lt, 0.0, 0.3)
        if a > 0.003:
            L += self.fill_layer(self.palette.fg, a, 2.0)
        if ph != "hold":
            faces, sides = [], []
            for s in self.sl:
                st = self.state(s, t)
                if st["th"] < 0.05:
                    continue
                f, sd = self.polys(s, st)
                faces.append(f)
                sides.append(sd)
            if sides:
                L.append(Layer(sides, self.side_col, "fill", split=True, bias=3.0))
                L.append(Layer(faces, self.palette.fg, "fill", split=True, bias=4.0))
        return L
