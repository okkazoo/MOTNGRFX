"""particles: the word is a cloud of a couple of thousand particles (dots, plus marks, dashes along the strokes). They fly in
paired from the entry lanes as streaks, land on their sites in the glyphs and settle into the word, which then pulses on the
beat (size, brightness, spacing of the word and of each letter, and a kick toward the viewer); to leave, the particles are
released and fly off down the exit lanes. The beat is a parameter: bpm (default 120) or beat (seconds), pulse=False turns
it off. Beats are counted from the moment the word lands, the first one is the downbeat. (mechanism: 13_camera v005
pulse_particles / pp_geo / pp_scene, rewritten for the Bundle interface)

Particle depth is anamorphic: every particle sits on a ray through an eye `eye` px in front of the placement, so a camera
at that distance sees the word exactly whatever the depths; any other viewpoint sees a volume.
SOLID HOLD (user 2026-11-01, R18 / R19: "i dont like the text being filled with texture like this ... not this bitsy texture fill"): the dots are only the entrance and the exit. As the particles
land they melt into a SOLID fill of the glyphs (palette ink, one fine outline in the first accent); by the hit the dots are gone and the word stands solid for the whole hold, pulsing on the beat as before;
at the exit the solid lets go and the particles stream away. (design: the planner no longer picks this module at all, design_defaults.SPECKLE_MODULES; this keeps a hand-built word honest.)
Params: n (particles), mode 'fill' (stipple inside the glyphs) | 'edge' (along the contours), kinds (p_dot, p_plus, p_dash),
size (dot radius px), depth (px), bpm / beat, pulse, speed (px/s of the streams), spread (width of the lanes), seed."""
import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, gaussian_filter
from mograph.modules.core import (WordModule, Bundle, Layer, ConList, register, layout_word, unit, mixc, ol)
from mograph.modules.words._cb import Chain, ss, ssa
from mograph.modules.words._strand import lane_frame

KW, KL, KICK, SZ = 0.028, 0.070, 150.0, 0.50
ATT, DEC, WEAK = 0.040, 0.085, 0.55
NCOL = 6
SOLID_FROM = 0.78          # the solid fill starts to melt in at this share of the form window and is complete at the hit (the end of the window)
SOLID_OUT_S = 0.12         # seconds (at least) the solid takes to let go at the start of the exit


def pulse_curve(t, form, beat, strong_every=4):
    """beat pulse 0..1 at time t (scalar): beats at form + k*beat, k integer (k % 4 == 0 is the strong one, 1.0; others 0.55);
    crisp attack ATT before the beat, exponential decay DEC after."""
    j = math.floor((t - form) / beat + 1e-9)
    out = 0.0
    for jj in (j - 1, j, j + 1):
        tau = t - (form + jj * beat)
        shape = math.exp(-tau / DEC) if tau >= 0 else ss(1.0 + tau / ATT)
        out = max(out, (1.0 if jj % strong_every == 0 else WEAK) * shape)
    return out


def _hex_points(wd, ht, s, ang, jit, rng):
    n = int(max(wd, ht) * 1.5 / s) + 4
    a, b = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    X = (a + 0.5 * (b % 2)) * s
    Y = b * s * math.sqrt(3) / 2
    P = np.stack([X.ravel(), Y.ravel()], 1)
    c, sn = math.cos(ang), math.sin(ang)
    P = P @ np.array([[c, -sn], [sn, c]]).T
    return P + rng.uniform(-jit, jit, P.shape) * s


def fill_sites(lay, N, rng, M=2.0):
    """jittered hex stipple inside the glyphs (spacing tuned to N). -> P (n,2) local, angle (n,) along the stroke, letter id."""
    wd, ht = lay["size"]
    pad = 8
    Wm, Hm = int((wd + 2 * pad) * M) + 1, int((ht + 2 * pad) * M) + 1
    x0, y0 = -wd / 2 - pad, -ht / 2 - pad
    ids = np.full((Hm, Wm), -1, np.int16)
    for li, l in enumerate(lay["letters"]):
        m = np.zeros((Hm, Wm), np.uint8)
        cv2.fillPoly(m, [np.round((q - [x0, y0]) * M * 16).astype(np.int32) for q in l["cons"] if len(q) > 2], 1, cv2.LINE_8, 4)
        ids[m > 0] = li
    mask = ids >= 0
    area = mask.sum() / (M * M)
    s = math.sqrt(2.0 * area / (math.sqrt(3) * N))
    ang = rng.uniform(0, math.pi)
    P = None
    for _ in range(7):
        C = _hex_points(wd, ht, s, ang, 0.32, rng)
        ix = ((C[:, 0] - x0) * M).astype(int)
        iy = ((C[:, 1] - y0) * M).astype(int)
        ok = (ix >= 0) & (ix < Wm) & (iy >= 0) & (iy < Hm)
        ok[ok] = mask[iy[ok], ix[ok]]
        P = C[ok]
        if abs(len(P) - N) <= 0.015 * N:
            break
        s *= math.sqrt(len(P) / float(N))
    if len(P) > N:
        P = P[rng.permutation(len(P))[:N]]
    elif len(P) < N:
        ys, xs = np.nonzero(mask)
        k = rng.integers(0, len(xs), N - len(P))
        P = np.vstack([P, np.stack([xs[k] + rng.random(len(k)), ys[k] + rng.random(len(k))], 1) / M + [x0, y0]])
    dist = gaussian_filter(distance_transform_edt(mask), 1.5)
    gy, gx = np.gradient(dist)
    ix = np.clip(((P[:, 0] - x0) * M).astype(int), 0, Wm - 1)
    iy = np.clip(((P[:, 1] - y0) * M).astype(int), 0, Hm - 1)
    ang = np.arctan2(gy[iy, ix], gx[iy, ix]) + math.pi / 2
    lid = ids[iy, ix].astype(int)
    lid[lid < 0] = 0
    return P, ang, lid


def edge_sites(lay, N, rng):
    segs, tot = [], 0.0
    for li, l in enumerate(lay["letters"]):
        for c in l["cons"]:
            cc = np.vstack([c, c[:1]])
            d = np.linalg.norm(np.diff(cc, axis=0), axis=1)
            segs.append((li, cc, np.r_[0.0, np.cumsum(d)]))
            tot += d.sum()
    P, A, L = [], [], []
    for li, c, cum in segs:
        n = max(1, int(round(N * cum[-1] / tot)))
        s = (np.arange(n) + rng.uniform(0.2, 0.8, n)) / n * cum[-1]
        x, y = np.interp(s, cum, c[:, 0]), np.interp(s, cum, c[:, 1])
        i = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(c) - 2)
        t = c[i + 1] - c[i]
        P.append(np.stack([x, y], 1)); A.append(np.arctan2(t[:, 1], t[:, 0])); L.append(np.full(n, li))
    P, A, L = np.concatenate(P), np.concatenate(A), np.concatenate(L)
    if len(P) > N:
        k = rng.permutation(len(P))[:N]
        P, A, L = P[k], A[k], L[k]
    while len(P) < N:
        k = rng.integers(0, len(P))
        P, A, L = np.vstack([P, P[k] + rng.normal(size=2) * 0.4]), np.r_[A, A[k]], np.r_[L, L[k]]
    return P, A, L.astype(int)


def herm3(A, B, m0, m1, e):
    """vector cubic Hermite for e in [0,1]; linear extensions outside: A + m0 e (e < 0), B + m1 (e - 1) (e > 1)."""
    ec = np.clip(e, 0.0, 1.0)[:, None]
    e2, e3 = ec * ec, ec * ec * ec
    p = (2 * e3 - 3 * e2 + 1) * A + (e3 - 2 * e2 + ec) * m0 + (-2 * e3 + 3 * e2) * B + (e3 - e2) * m1
    lo = (e < 0)[:, None]
    hi = (e > 1)[:, None]
    p = np.where(lo, A + m0 * e[:, None], p)
    p = np.where(hi, B + m1 * (e[:, None] - 1.0), p)
    return p


def pair_sorted(a, b):
    """pairing of two point sets (N,2) (lateral coordinates across the lane direction): equal-count strips along the first
    coordinate, sorted along the second inside each strip, matched in order. O(N log N); streams stay in order (no crossing in
    the first coordinate). -> perm with lane i <-> b[perm[i]]"""
    N = len(a)
    sa, sb = np.ptp(a, axis=0) + 1e-6, np.ptp(b, axis=0) + 1e-6
    ns = int(np.clip(round(math.sqrt(N * max(sb[0], 1e-3) / max(sb[1], 1e-3))), 1, N))
    def order(P):
        o = np.argsort(P[:, 0])
        out = []
        for ch in np.array_split(o, ns):
            out.append(ch[np.argsort(P[ch, 1])])
        return np.concatenate(out)
    oa, ob = order(a), order(b)
    perm = np.empty(N, int)
    perm[oa] = ob
    return perm


def quant_groups(P2, cols, wids, bias, wq=0.4):
    """Layers for 2-point (or n-point) polylines P2 (n,k,3) grouped by (colour, quantised width)."""
    if len(P2) == 0:
        return []
    ci = cols.astype(np.int32)
    wi = np.round(wids / wq).astype(np.int32)
    key = (ci[:, 0] * 65536 + ci[:, 1] * 256 + ci[:, 2]) * 64 + np.clip(wi, 0, 63)
    out = []
    ok = wi > 0
    for kv in np.unique(key[ok]):
        sel = np.nonzero(ok & (key == kv))[0]
        c = tuple(int(x) for x in ci[sel[0]])
        out.append(Layer(list(P2[sel]), c, "line", float(wi[sel[0]] * wq), closed=False, bias=bias))
    return out


@register("particles")
class Particles(WordModule):
    description = ("Two thousand particles fly in paired along the entry lanes as streaks, land on their sites in the glyphs "
                   "and settle into the word, which pulses on the beat; to leave they are released and stream away down the "
                   "exit lanes. bpm and pulse are parameters.")
    DEMO_WORDS = "PULSE"
    DEFAULT_FONT = "ultra"
    MODE = "fill"
    KINDS = (0.70, 0.00, 0.30)
    N = 3000
    TAU = 0.032
    K = 5

    def build(self):
        P, pl, pal = self.params, self.placement, self.palette
        rng = np.random.default_rng(int(P.get("seed", 3)))
        self.lay = layout_word(self.font, self.lines, pl.w * 0.94, pl.h * 0.9)
        n_req = int(P.get("n", self.N))
        mode = P.get("mode", self.MODE)
        if mode == "edge":
            S2, ang, lid = edge_sites(self.lay, n_req, rng)
        else:
            S2, ang, lid = fill_sites(self.lay, n_req, rng)
        self.N = N = len(S2)
        self.mode = mode
        lc = np.array([np.concatenate(l["cons"]).mean(0) if l["cons"] else (0, 0) for l in self.lay["letters"]])
        lc = np.array([(np.concatenate(l["cons"]).min(0) + np.concatenate(l["cons"]).max(0)) / 2 if l["cons"] else (0, 0) for l in self.lay["letters"]])
        self.lc_letter = lc
        self.site2, self.lc2, self.ang = S2, lc[lid], ang
        D = float(P.get("eye", 1000.0))
        self.D = D
        dep = float(P.get("depth", 0.11 * max(pl.w, pl.h)))
        self.dz = dep * (0.45 * np.sin(lid * 1.7) + 0.55 * rng.uniform(-1, 1, N))
        self.kick = rng.uniform(0.35, 1.0, N) * KICK
        self.E0 = pl.origin - pl.n * D
        self.rest = self._place(S2[:, 0], S2[:, 1], self.dz)
        # per-particle look
        sz = float(P.get("size", 1.7))
        self.r0 = sz * rng.uniform(0.62, 1.05, N) * np.where(rng.random(N) < 0.10, 1.35, 1.0)
        self.spark = rng.random(N) < 0.05
        pk = np.array(P.get("kinds", self.KINDS), float)
        pk = pk / pk.sum()
        u = rng.random(N)
        self.kind = np.where(u < pk[0], 0, np.where(u < pk[0] + pk[1], 1, 2))
        self.bright = np.round(rng.uniform(0.62, 1.0, N) * (NCOL - 1)) / (NCOL - 1) * 0.38 + 0.62
        # lanes
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.beat = float(P.get("beat") or 60.0 / float(P.get("bpm", 120.0)))
        self.do_pulse = bool(P.get("pulse", True))
        self.speed = float(P.get("speed", 1000.0))
        self._build_entry(rng)
        self._build_exit(rng)
        self._bundles()

    # ------------------------------------------------------------------ geometry
    def _place(self, x, y, dz, kick=None):
        pl = self.placement
        Z = self.D + dz - (0.0 if kick is None else kick)
        return (self.E0[None, :] + pl.n[None, :] * Z[:, None] + (x[:, None] * pl.u + y[:, None] * pl.v) * (Z / self.D)[:, None])

    def _lanes(self, d, rng, ext):
        """N lane points in a ribbon cross-section around the placement centre at distance `ext` along d (the gate)."""
        pl = self.placement
        e1, e2 = lane_frame(d, pl)
        spread = float(self.params.get("spread", 1.0))
        a = rng.normal(size=self.N)
        b = rng.normal(size=self.N)
        a, b = np.clip(a, -1.9, 1.9) / 1.9, np.clip(b, -1.9, 1.9) / 1.9
        L1 = 2.0 * pl.extent_along(e1)
        L2 = 2.0 * pl.extent_along(e2)
        lat = e1[None] * (a * 0.46 * L1 * spread)[:, None] + e2[None] * (b * 0.22 * max(L2, 0.45 * L1) * spread)[:, None]
        return pl.origin + lat + d * ext

    def _rank(self, d):
        al = (self.rest - self.placement.origin) @ d
        return np.argsort(np.argsort(al)) / max(self.N - 1.0, 1.0)

    def _build_entry(self, rng):
        tm, N, pl = self.timing, self.N, self.placement
        d = self.d_in
        ext = pl.extent_along(d) + 0.06 * max(pl.w, pl.h) + 12.0
        G = self._lanes(d, rng, -ext)
        G = G - d[None] * (rng.random(N) ** 0.8 * float(self.params.get("runway", 0.55 * max(pl.w, pl.h))))[:, None]
        e1, e2 = lane_frame(d, pl)
        perm = pair_sorted(np.stack([G @ e1, G @ e2], 1), np.stack([self.rest @ e1, self.rest @ e2], 1))   # lane i -> site perm[i]
        self.perm_in = perm
        A = G                                                  # lane i starts at A[i]; the particle that owns lane i is `perm[i]`
        # reorder every per-particle array so that index i = lane i
        for k in ("site2", "lc2", "ang", "dz", "kick", "rest", "r0", "spark", "kind", "bright"):
            setattr(self, k, getattr(self, k)[perm])
        B = self.rest
        Lc = np.linalg.norm(B - A, axis=1)
        self.A_in, self.B_in, self.Lc_in = A, B, Lc
        V = 0.0
        field = self._bow_field(A, rng, Lc.mean())
        self.m0_in = d[None] * Lc[:, None]
        self.m1_in = (B - A) * 0.8 + field * 0.4
        r = self._rank(d)
        T = tm.form * (0.50 + 0.38 * r + 0.10 * rng.random(N))
        T = np.minimum(T, tm.form)
        v0 = np.minimum(self.speed / Lc, 2.8 / T)
        self.ch_in = Chain(np.stack([np.zeros(N), T], 1), np.stack([np.zeros(N), np.ones(N)], 1), np.stack([v0, np.zeros(N)], 1), coast=False)
        self.T_in = T

    def _bow_field(self, A, rng, md):
        km = rng.normal(size=(4, 3))
        km = unit(km) * (2 * math.pi / rng.uniform(1100, 1900, (4, 1)))
        ph = rng.uniform(0, 2 * math.pi, 4)
        av = unit(rng.normal(size=(4, 3)))
        fld = (np.sin(A @ km.T + ph)[:, :, None] * av[None]).sum(1) / 2.0
        return fld * 0.35 * md

    def _build_exit(self, rng):
        tm, N, pl = self.timing, self.N, self.placement
        d = self.d_out
        ext = pl.extent_along(d) + 0.06 * max(pl.w, pl.h) + 12.0
        G = self._lanes(d, rng, ext)
        G = G + d[None] * (rng.random(N) ** 0.8 * float(self.params.get("runway", 0.55 * max(pl.w, pl.h))))[:, None]
        e1, e2 = lane_frame(d, pl)
        lane_of = pair_sorted(np.stack([self.rest @ e1, self.rest @ e2], 1), np.stack([G @ e1, G @ e2], 1))   # particle -> lane
        Bx = G[lane_of]
        A = self.rest
        Lc = np.linalg.norm(Bx - A, axis=1)
        field = self._bow_field(A, rng, Lc.mean())
        self.A_out, self.B_out, self.Lc_out = A, Bx, Lc
        self.m0_out = (Bx - A) * 0.5 + field * 0.6 - self.placement.n[None] * (0.18 * Lc)[:, None]
        self.m1_out = d[None] * Lc[:, None]
        r = self._rank(d)
        te = tm.exit * (0.02 + 0.40 * (1.0 - r) + 0.06 * rng.random(N))
        self.te = te
        ex = tm.exit
        v1 = np.minimum(self.speed / Lc, 2.8 / np.maximum(ex - te, 0.2))
        self.ch_out = Chain(np.stack([te, np.full(N, ex)], 1), np.stack([np.zeros(N), np.ones(N)], 1), np.stack([np.zeros(N), v1], 1), coast=True)

    # ------------------------------------------------------------------ state
    def pulse_at(self, t):
        if not self.do_pulse:
            return 0.0
        return pulse_curve(t, self.timing.form, self.beat)

    def state(self, t, with_pulse=True):
        """positions (N,3), settle weight w (N,), pulse p at time t"""
        tm = self.timing
        if t < tm.form + tm.hold:
            e = self.ch_in.pos(t)
            base = herm3(self.A_in, self.B_in, self.m0_in, self.m1_in, e)
            w = ssa((e - 0.85) / 0.15)
        else:
            e = self.ch_out.pos(t - tm.form - tm.hold)
            base = herm3(self.A_out, self.B_out, self.m0_out, self.m1_out, e)
            w = 1.0 - ssa(e / 0.15)
        p = self.pulse_at(t) if with_pulse else 0.0
        if p > 1e-4:
            x, y = self.site2[:, 0], self.site2[:, 1]
            xl = x * (1 + KW * p) + (x - self.lc2[:, 0]) * KL * p
            yl = y * (1 + KW * p) + (y - self.lc2[:, 1]) * KL * p
            moved = self._place(xl, yl, self.dz, self.kick * p)
            base = base + w[:, None] * (moved - self.rest)
        return base, w, p

    def solid_at(self, t):
        """0..1 share of the SOLID glyph fill at time t: melts in over the last part of the form window, 1 for the whole hold, lets go at the start of the exit."""
        tm = self.timing
        if t < tm.form + tm.hold:
            return float(ssa((t - SOLID_FROM * tm.form) / max((1.0 - SOLID_FROM) * tm.form, 1e-6)))
        return float(1.0 - ssa((t - tm.form - tm.hold) / max(SOLID_OUT_S, 0.10 * tm.exit)))

    def _solid_layers(self, s, p):
        """the glyphs as solid fills (even-odd per letter, counters stay open) in the palette ink with one fine accent outline, following the beat pulse. Flat on the placement plane (the dots' depth spread is not kept:
        a solid face has none), under the dots (bias < 0) with the outline over them."""
        if s <= 0.002:
            return []
        pl = self.placement
        kick = float(self.kick.mean()) * p
        Z = self.D - kick
        k = Z / self.D
        alpha = s if s < 0.998 else 1.0
        cap = float(self.lay["cap"])
        lw = float(min(max(0.006 * cap, 1.0), 2.0))
        ink, edge = tuple(self.palette.fg), tuple(self.palette.accents[0])
        out = []
        for li, l in enumerate(self.lay["letters"]):
            polys = ConList()
            polys.union = bool(getattr(l["cons"], "union", False))
            for c in l["cons"]:
                if len(c) < 3:
                    continue
                x, y = c[:, 0], c[:, 1]
                if p > 1e-4:
                    x = x * (1 + KW * p) + (x - self.lc_letter[li, 0]) * KL * p
                    y = y * (1 + KW * p) + (y - self.lc_letter[li, 1]) * KL * p
                polys.append(self.E0[None, :] + pl.n[None, :] * Z + (x[:, None] * pl.u + y[:, None] * pl.v) * k)
            if not polys:
                continue
            out.append(Layer(polys, ink, "fill", alpha=alpha, bias=-60.0))
            out.append(Layer(list(polys), edge, "line", lw, closed=True, alpha=alpha, bias=60.0))
        return out

    def streak(self, t):
        K, tau = self.K, self.TAU
        pts = []
        for k in range(K):
            pts.append(self.state(t - k * tau / (K - 1), False)[0])
        return np.stack(pts, 1)                                # (N, K, 3), head first

    def _bundles(self):
        N = self.N
        fg = np.array(self.palette.fg, float)
        acc = np.array(self.palette.accents[2], float)
        self.col_streak = np.where(self.spark[:, None], acc[None] * 0.8, fg[None] * (self.bright[:, None] * 0.55))
        self.col_streak = np.round(self.col_streak)
        self.w_streak = 0.6
        for name, t in (("_entry", 0.0), ("_exit", self.duration)):
            Pk = self.streak(t)
            v = np.linalg.norm(Pk[:, 0] - Pk[:, 1], axis=1) / (self.TAU / (self.K - 1))
            D = np.tile(self.d_in if name == "_entry" else self.d_out, (N, 1))
            setattr(self, name, Bundle(Pk, D, self.col_streak, self.w_streak, t, v, 0.0))

    def entry_bundle(self):
        return self._entry

    def exit_bundle(self):
        return self._exit

    # ------------------------------------------------------------------ frame
    def layers(self, t):
        t = max(t, 0.0)
        pl, N = self.placement, self.N
        Pk = self.streak(t)
        v = np.linalg.norm(Pk[:, 0] - Pk[:, 1], axis=1) / (self.TAU / (self.K - 1))
        mv = ssa((v - 40.0) / 220.0)
        head, w, p = self.state(t)
        wp = w * p
        s = self.solid_at(t)
        vis = 1.0 - s * w                                      # how much of each settled particle is still a dot: none once the solid fill is in
        fg = np.array(self.palette.fg, float)
        acc = np.array(self.palette.accents[2], float)
        out = self._solid_layers(s, p)
        # streaks (thin, dim): exactly the bundle strands while the particle moves; collapse to dots at rest
        out += quant_groups(Pk, self.col_streak, self.w_streak * np.where(mv > 0.02, 1.0, 0.0), 0.0, wq=0.1)
        rest = 1.0 - mv
        r = self.r0 * (1.0 + SZ * wp)
        base = np.where(self.spark[:, None], acc[None], fg[None]) * (self.bright[:, None] * (0.8 + 0.2 * wp[:, None]))
        base = base + (255.0 - base) * (0.65 * wp[:, None]) * (~self.spark[:, None])
        base = np.clip(np.round(base / 8.0) * 8.0, 0, 255)
        U, V = pl.u, pl.v
        m = np.cos(self.ang)[:, None] * U[None] + np.sin(self.ang)[:, None] * V[None]
        wd = 2.0 * r * rest * vis
        k0 = self.kind == 0
        k1 = self.kind == 1
        k2 = self.kind == 2
        if k0.any():
            h = head[k0]
            out += quant_groups(np.stack([h, h], 1), base[k0], wd[k0], 1.0)
        if k2.any():
            hh = 2.1 * r[k2][:, None] * m[k2]
            out += quant_groups(np.stack([head[k2] - hh, head[k2] + hh], 1), base[k2], np.maximum(0.7, 0.5 * r[k2]) * 2.0 * rest[k2] * vis[k2], 1.0)
        if k1.any():
            hh = 1.8 * r[k1][:, None] * m[k1]
            pr = np.cross(m[k1], pl.n[None]) * 1.8 * r[k1][:, None]
            lw = np.maximum(0.7, 0.5 * r[k1]) * 2.0 * rest[k1] * vis[k1]
            out += quant_groups(np.stack([head[k1] - hh, head[k1] + hh], 1), base[k1], lw, 1.0)
            out += quant_groups(np.stack([head[k1] - pr, head[k1] + pr], 1), base[k1], lw, 1.0)
        return out

    def focus(self, t):
        t = min(max(t, 0.0), self.duration)
        return self.state(t)[0].mean(0)

    def hold_pose(self, dist=1000.0):
        pl = self.placement
        return pl.origin - pl.n * dist, pl.origin.copy()
