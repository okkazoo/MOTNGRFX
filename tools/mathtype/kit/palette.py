"""Named palettes from the mathtype pieces, plus colour-area measurement (minority rule: accents <= 10-15 %
of the frame).

    pal = palette.get("mono")          # Palette(ground, ink, accents={...})
    pal.ground, pal.ink, pal.accent    # 0..1 RGB float32 (pal.accent = the first accent)
    palette.area(img_u8, pal)          # {"ground": .71, "ink": .26, "accent": .03, "other": 0}
    palette.overruns(img_u8, pal)      # [] or [{"colour": "accent", "share": .19, "limit": .15}]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


@dataclass
class Palette:
    name: str
    ground: np.ndarray
    ink: np.ndarray
    accents: dict = field(default_factory=dict)      # name -> rgb
    tones: dict = field(default_factory=dict)        # extra non-accent tones (e.g. a second ground)
    limit: float = 0.15                               # max share of any one accent

    @property
    def accent(self):
        return next(iter(self.accents.values())) if self.accents else self.ink

    def colours(self):
        d = {"ground": self.ground, "ink": self.ink}
        d.update(self.tones)
        d.update(self.accents)
        return d

    def hex(self):
        return {k: "#" + "".join(f"{int(round(c * 255)):02x}" for c in v) for k, v in self.colours().items()}


def _p(name, ground, ink, accents=None, tones=None, limit=0.15):
    return Palette(name, rgb(ground), rgb(ink), {k: rgb(v) for k, v in (accents or {}).items()},
                   {k: rgb(v) for k, v in (tones or {}).items()}, limit)


PALETTES = {
    # common brief: near-black ground, warm white marks, one optional accent used sparingly
    "mono": _p("mono", "#050505", "#f3eadb", {"accent": "#e2582a"}, limit=0.10),
    # E1 contours: ink / bone with ochre and teal bands, vermilion disc
    "e1": _p("e1", "#111111", "#ECE6DA", {"ochre": "#D9A441", "teal": "#1F5E63", "vermilion": "#D2462A"}),
    "ink": _p("ink", "#111111", "#ECE6DA"),
    # E2 outline paths: night with dawn lights
    "dawn": _p("dawn", "#0A0B10", "#F6E7B8", {"amber": "#F2A65A", "coral": "#E86F68"}),
    # E3 tracegeo: chalk on graphite with leaf / terra / sky
    "chalk": _p("chalk", "#2B2B2B", "#F1EDE4", {"leaf": "#5B8C51", "terra": "#C8553D", "sky": "#7AA6C2"}),
    # E4 moire: greyscale with a cobalt lock
    "e4": _p("e4", "#000000", "#FFFFFF", {"cobalt": "#2D4BD8"}),
    # E5 growth: ink / bone, moss plate, lichen, seed orange
    "nature": _p("nature", "#EEE8DC", "#121212", {"lichen": "#B7C48A", "seed": "#E07B39"}, {"moss": "#3F6B3A"}),
    # E6 video trace: navy sea, foam, sand, coral
    "sea": _p("sea", "#0B1F33", "#CFE3E0", {"coral": "#FF7A59"}, {"sand": "#E8D3A9"}),
}


def get(name: str) -> Palette:
    return PALETTES[name]


def area(img, pal: Palette, tol: float = 0.12, small: int = 96):
    """Share of the frame per palette colour: each pixel goes to the nearest palette colour within `tol`
    (RGB distance, 0..1 units), else 'other'. Neutral greys between ground and ink count as ink/ground
    (anti-aliasing), so only chromatic accents are measured strictly. img: uint8 (H, W, 3) or float."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
    if max(a.shape[:2]) > small:                            # measure on a small copy (fast, area-preserving)
        import cv2
        s = small / max(a.shape[:2])
        a = cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
    px = a.reshape(-1, 3)
    cols = pal.colours()
    names = list(cols)
    C = np.stack([cols[k] for k in names])
    # a pixel on the segment between two palette colours is an anti-aliased / tonal blend of the two: it counts
    # for the nearer end (ground-ink greys go to ground or ink). Only pixels far from every such segment are 'other'.
    d = np.linalg.norm(px[:, None, :] - C[None], axis=2)
    lab_i = d.argmin(1)
    best = d[np.arange(len(px)), lab_i]
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            seg = C[j] - C[i]
            u = np.clip(((px - C[i]) @ seg) / max(float(seg @ seg), 1e-9), 0, 1)
            sd = np.linalg.norm(px - (C[i] + u[:, None] * seg), axis=1)
            take = sd < np.minimum(best, tol)
            lab_i = np.where(take, np.where(u > 0.5, j, i), lab_i)
            best = np.where(take, sd, best)
    lab = np.array(names, dtype=object)[lab_i]
    lab = np.where(best > tol, "other", lab)
    out = {nm: float(np.mean(lab == nm)) for nm in names + ["other"]}
    return out


def overruns(img, pal: Palette, limit: float | None = None, shares: dict | None = None):
    """Accents above the minority limit (default pal.limit), and 'other' (off-palette) above 5 %.
    shares: a precomputed area(img, pal) (img is then unused)."""
    sh = shares if shares is not None else area(img, pal)
    lim = pal.limit if limit is None else limit
    flags = [dict(colour=k, share=round(sh[k], 3), limit=lim) for k in pal.accents if sh[k] > lim]
    acc = sum(sh[k] for k in pal.accents)
    if len(pal.accents) > 1 and acc > lim:
        flags.append(dict(colour="all accents", share=round(acc, 3), limit=lim))
    if sh["other"] > 0.05:
        flags.append(dict(colour="other", share=round(sh["other"], 3), limit=0.05))
    return flags
