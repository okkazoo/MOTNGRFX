"""Variable-font axes for the fusion renderer: static axes dicts, animated axes (function of t / keyframes / beat-driven shapes)
and the warp that makes a running word module follow them.

Glyph outlines in the module engine come from fontTools (`ol.word` -> `getGlyphSet(location=axes)` -> FlatPen), not from Pillow, so
variable axes are applied there: `core.layout_word(font={'key':..,'axes':{..}})` is the single entry (cached per axes dict). Animated
axes are QUANTISED (`quant` steps per axis range) so only a few dozen distinct layouts exist per clip.

A word module builds its strands / fill once at its nominal axes (the middle of each animated range). AxesWarp then moves every
point a module emits near the type outline by the displacement between the nominal outline and the outline at the axes of time t:
contours are matched index-for-index (variable fonts are point-compatible) and resampled by arc length; a layer point takes the
displacement of its nearest nominal outline sample, faded out with distance (so lanes far from the type stay put)."""
import copy
import math
import numpy as np
from scipy.spatial import cKDTree
from fontTools.ttLib import TTFont

from mograph.modules import core

_INFO = {}


def axis_info(font):
    """{tag: (min, default, max)} of the font's fvar axes ({} for a static font). `font`: anything core.resolve_font takes."""
    key, _ = core.resolve_font(font)
    if key not in _INFO:
        f = TTFont(core.ol.font_path(key))
        _INFO[key] = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes} if "fvar" in f else {}
        f.close()
    return _INFO[key]


def canon_axes(font, axes, quant=None):
    """axes dict with case-insensitive tags mapped to the font's own, unknown tags dropped, values clamped (and quantised to
    `quant` steps over the axis range when given)."""
    info = axis_info(font)
    low = {t.lower(): t for t in info}
    out = {}
    for k, v in (axes or {}).items():
        tag = low.get(str(k).lower())
        if tag is None:
            continue
        lo, _, hi = info[tag]
        v = min(max(float(v), lo), hi)
        if quant and hi > lo:
            st = (hi - lo) / quant
            v = lo + round((v - lo) / st) * st
        out[tag] = round(v, 4)
    return out


# ------------------------------------------------------------------ tracks (value of one axis over time)
def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class AxisTrack:
    """one animated axis. spec forms: number | [lo, hi] (slow sine) | [lo, hi, shape, {opts}] with shape 'beat' (pulse on every
    beat: a 0.12-beat attack to hi, exponential decay `decay`=3.0 back to lo) | 'bar' (the same on downbeats) | 'sine' (period
    `period` s, default 4 beats or 3 s; `phase` 0..1) | 'ramp' (lo -> hi over the word) | 'pingpong' | {'keys': [[t, v], ...],
    'ease': 'smooth'|'linear'} (global seconds) | a callable t -> value."""

    def __init__(self, spec, beats=None, span=(0.0, 1.0)):
        self.beats, self.span, self.fn, self.kind = beats, span, None, "const"
        self.opts = {}
        self.lo = self.hi = None
        if callable(spec):
            self.fn, self.kind = spec, "fn"
        elif isinstance(spec, dict):
            ks = sorted((float(a), float(b)) for a, b in spec["keys"])
            self.keys, self.ease, self.kind = np.array(ks), spec.get("ease", "smooth"), "keys"
            self.lo, self.hi = float(self.keys[:, 1].min()), float(self.keys[:, 1].max())
        elif isinstance(spec, (list, tuple)):
            self.lo, self.hi = float(spec[0]), float(spec[1])
            self.kind = str(spec[2]) if len(spec) > 2 else "sine"
            self.opts = dict(spec[3]) if len(spec) > 3 else {}
        else:
            self.lo = self.hi = float(spec)
            self.kind = "const"

    @property
    def animated(self):
        return self.kind != "const" and (self.lo is None or self.hi != self.lo)

    def nominal(self):
        return 0.5 * (self.lo + self.hi) if self.lo is not None else float(self.fn(0.5 * (self.span[0] + self.span[1])))

    def __call__(self, t):
        k = self.kind
        if k == "const":
            return self.lo
        if k == "fn":
            return float(self.fn(t))
        if k == "keys":
            tt, vv = self.keys[:, 0], self.keys[:, 1]
            if self.ease == "linear" or len(tt) < 2:
                return float(np.interp(t, tt, vv))
            if t <= tt[0]:
                return float(vv[0])
            if t >= tt[-1]:
                return float(vv[-1])
            i = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
            u = _ss((t - tt[i]) / max(tt[i + 1] - tt[i], 1e-9))
            return float(vv[i] + (vv[i + 1] - vv[i]) * u)
        lo, hi = self.lo, self.hi
        if k in ("beat", "bar"):
            if not self.beats:
                p = (t * 2.0) % 1.0                          # no grid: 120 bpm
            else:
                t0, iv = self.beats.prev(t, bars=(k == "bar"))
                p = (t - t0) / iv
            dec = float(self.opts.get("decay", 3.0))
            a = 0.12
            env = _ss(p / a) * math.exp(-dec * max(p - a, 0.0))
            return lo + (hi - lo) * env
        if k == "sine":
            per = float(self.opts.get("period", (4.0 * self.beats.iv) if self.beats else 3.0))
            ph = float(self.opts.get("phase", 0.0))
            return lo + (hi - lo) * (0.5 - 0.5 * math.cos(2.0 * math.pi * (t / per + ph)))
        if k == "ramp":
            u = (t - self.span[0]) / max(self.span[1] - self.span[0], 1e-9)
            return lo + (hi - lo) * _ss(u)
        if k == "pingpong":
            per = float(self.opts.get("period", 4.0))
            u = (t / per) % 1.0
            return lo + (hi - lo) * (1.0 - abs(2.0 * u - 1.0))
        raise ValueError(f"unknown axis shape {k!r}")


class AxesAnim:
    """the axes of one word over time: static values + animated tracks. `nominal` is what the module is built at."""

    def __init__(self, font, axes, beats=None, span=(0.0, 1.0), quant=32, base=None):
        self.font, self.quant = font, quant
        self.static, self.tracks = {}, {}
        low = {t.lower(): t for t in axis_info(font)}
        for k, v in (base or {}).items():
            tag = low.get(str(k).lower())
            if tag:
                self.static[tag] = float(v)
        for k, v in (axes or {}).items():
            tag = low.get(str(k).lower())
            if tag is None:
                continue
            tr = AxisTrack(v, beats, span)
            if tr.animated:
                self.tracks[tag] = tr
            else:
                self.static[tag] = tr.nominal()
        self.nominal = canon_axes(font, {**self.static, **{t: tr.nominal() for t, tr in self.tracks.items()}}, quant)

    @property
    def animated(self):
        return bool(self.tracks)

    def at(self, t):
        return canon_axes(self.font, {**self.static, **{tag: tr(t) for tag, tr in self.tracks.items()}}, self.quant)

    def key(self, t):
        return tuple(sorted(self.at(t).items()))


# ------------------------------------------------------------------ warp
def _resample_closed(c, n):
    c2 = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(c2, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] < 1e-9:
        return np.repeat(c[:1], n, 0)
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(s, cum, c2[:, 0]), np.interp(s, cum, c2[:, 1])], 1)


class AxesWarp:
    """moves what a word module emits to follow animated font axes (see the module doc). Picklable."""

    def __init__(self, module, anim, lines, fontkey, ar, box, sigma_k=0.10, spacing=2.0, t_start=0.0):
        self.anim, self.lines, self.fontkey, self.ar, self.box = anim, list(lines), fontkey, ar, box
        self.pl, self.t_start = module.placement, float(t_start)
        self.spacing = spacing
        nom = self._cons(anim.nominal)
        self.sigma = sigma_k * nom["cap"]
        flat = [c for l in nom["letters"] for c in l["cons"]]
        self.n = [int(np.clip(round(float(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum()) / spacing), 12, 400))
                  for c in flat]
        self.nom_samples = np.concatenate([_resample_closed(c, n) for c, n in zip(flat, self.n)])
        self.sig = tuple(len(l["cons"]) for l in nom["letters"])
        self.tree = cKDTree(self.nom_samples)
        self.tables = {}
        self.missing = 0

    def _cons(self, axes):
        font = dict(key=self.fontkey, axes=dict(axes), ar=self.ar)
        return core.layout_word(font, self.lines, self.box[0], self.box[1])

    def table(self, axes_key):
        """per nominal outline sample displacement (N, 2) at the axes `axes_key` (a tuple of items); zeros if the face is not
        point-compatible with the nominal one."""
        if axes_key not in self.tables:
            lay = self._cons(dict(axes_key))
            sig = tuple(len(l["cons"]) for l in lay["letters"])
            if sig != self.sig:
                self.tables[axes_key] = np.zeros_like(self.nom_samples)
                self.missing += 1
            else:
                flat = [c for l in lay["letters"] for c in l["cons"]]
                s = np.concatenate([_resample_closed(c, n) for c, n in zip(flat, self.n)])
                self.tables[axes_key] = s - self.nom_samples
        return self.tables[axes_key]

    def precompute(self, times):
        for t in times:
            self.table(self.anim.key(t))

    def apply(self, layers, t_global):
        D = self.table(self.anim.key(t_global))
        if not np.any(D):
            return layers
        o, u, v = self.pl.origin, self.pl.u, self.pl.v
        out = []
        for L in layers:
            if L.pts.shape[0] == 0:
                out.append(L)
                continue
            rel = L.pts - o
            q = np.stack([rel @ u, rel @ v], 1)
            d, idx = self.tree.query(q)
            wgt = np.exp(-(d / self.sigma) ** 2)
            if wgt.max() < 1e-3:
                out.append(L)
                continue
            disp = D[idx] * wgt[:, None]
            pts = L.pts + disp[:, 0:1] * u + disp[:, 1:2] * v
            L2 = copy.copy(L)
            L2.pts = pts
            L2.polys = [pts[a:b] for a, b in zip(L.offs[:-1], L.offs[1:])]
            out.append(L2)
        return out
