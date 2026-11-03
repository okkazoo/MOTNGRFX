"""hold_layouts.py - named hold compositions from the user's layout notes (16_app/LAYOUT_NOTES.md; plates 11_mathtype/v015/local/layouts_g*).
The Type Layout Studies page is used for COMPOSITION ONLY: fonts always come from mograph/fonts/liked.json.

A layout says where and how big ONE held line sits in the frame and may add flat backing shapes. It is a record (JSON-able, stored on
the word as ``hold_layout`` and in ``piece["plan"]["layouts"]``):

    {"name", "plate", "variant", "box": {"cx","cy","sw","sh"} | None, "roll", "ar_min", "shapes": [...], "anchored", "offset", "fill", "area"}

box     the on-screen box of the held text, fractions of the frame (cx, cy = centre, y down; sw, sh = size; > 1 = cropped by the frame).
        None = the full-frame hold (what every piece did before; nothing is changed).
offset  (cx - 0.5, cy - 0.5): the screen shift of the word. spec.build_scene moves the CAMERA by it (ScreenOffsetCamera), so every type
        pass (beauty / matte / depth / line, i.e. the LTX control video too) agrees with the composition.
fill    the camera `fill` that makes the framed box come out at (sw, sh) of the frame (see camera_fill).
shapes  flat backing shapes in frame fractions, drawn by the compose op ``layout_shapes`` (tools/fusion/compose.py) in the word's window.
        rect {cx, cy, w, h, angle} (angle: degrees clockwise) | poly {pts} | circle {cx, cy, r} (r = fraction of the shorter side);
        every shape has a colour ROLE (dark | ink2 | accent) resolved against the piece palette by plan_rules.shape_color: dark,
        low-luminance flats, because the type is tinted bright (tint_type lum_target 0.9).
anchored  the layout carries shapes in screen space, so the camera on that line is held (hold_drift, no dolly): a moving camera would slide
        the type off its strip.

Rules from the notes (checked by ``validate``): one idea per frame, no extra text of any kind, simple shapes (<= MAX_SHAPES), never squeeze
the type (ar_min), small in-plane rolls only (|roll| <= 12 deg), type stays together.
Known approximations (one word module = one held box, the Sequence has no two simultaneous words): edge_L is one line along the bottom /
top edge with an L-shaped solid, no separate vertical arm and junction word; counters_window frames a huge word for the letter-as-window
modules, the small words inside the counters (P4) are not placed; type_circle is a disc behind a horizontal line, not text on a ring.
"""
from __future__ import annotations

import math
import random
import re
import zlib

FORMATS = ("9:16", "16:9")
FILL = {"9:16": (1.35, 0.95), "16:9": (1.0, 0.8)}          # cam3d FIT_FILLK, FIT_HK of a portrait / landscape world plane
MAX_SHAPES = 8
SHAPE_KINDS = ("rect", "poly", "circle")
COLOR_ROLES = ("dark", "ink2", "accent")
MIN_CAP_FRAC = 0.04                                        # a small line's cap height stays >= 4 % of the frame height (26 px at 640); ONLY edge_run lines (see eligible)
HOLD_CAP_FRAC = 0.07                                       # every other hold: letter (cap) height >= 7 % of the frame height (45 px at 640), v005 round 2
FRAME_PX = {"9:16": (360, 640), "16:9": (640, 360)}        # preview frame, only the ratios matter
RESERVED = ("small_line",)                                 # layouts for the edge_run_small_type block only (the planner picks them for a line only when a block asks)
AR_MIN = 0.85                                              # never squeeze a layout line by more than 15 %
MAX_ROLL = 12.0
BIG_BOX_MODULES = ("solid_hollow", "scale_clash")          # slab modules whose extrusions run far outside the letters: they need a layout box >= BIG_BOX_W wide (v005 P21 THE CITY in a quadrant)
BIG_BOX_W = 0.8
WINDOW_MODULES = ("fly_into", "depth_tunnel")              # letter-as-window modules: counters_window needs them
_COUNTER_RE = re.compile(r"[ABDGOPQR0469@]")               # letters that have a counter (a hole)


def _rng(seed, tag) -> random.Random:
    return random.Random(zlib.crc32(f"{int(seed)}:{tag}".encode("utf-8")))


def _asp(fmt: str) -> float:
    return 16.0 / 9.0 if fmt == "16:9" else 9.0 / 16.0


def _r(x, nd=4):
    return float(round(float(x), nd))


def camera_fill(sw: float, sh: float, fmt: str = "9:16") -> float:
    """camera `fill` that frames a box of world size (sw*W, sh*H) so it comes out at (sw, sh) of the frame:
    cam3d: dist = F * max(w / (f W), h / (f HK H)) with f = fill * FILLK  ->  f = max(sw, sh / HK)."""
    fillk, hk = FILL[fmt]
    return _r(max(float(sw), float(sh) / hk) / fillk, 4)


def _letters(text: str) -> int:
    return len(re.sub(r"[\W_]+", "", text or ""))


# ------------------------------------------------------------------------------------------------ the layouts
def _rect(cx, cy, w, h, color, angle=0.0):
    return {"kind": "rect", "cx": _r(cx), "cy": _r(cy), "w": _r(w), "h": _r(h), "angle": _r(angle, 2), "color": color}


def _poly(pts, color):
    return {"kind": "poly", "pts": [[_r(x), _r(y)] for x, y in pts], "color": color}


def _full(rng, fmt):
    return dict(variant="full", box=None, roll=0.0, shapes=[])


def _quadrant_stack(rng, fmt):
    c = rng.choice(["tl", "tr", "bl", "br"])
    cx = 0.27 if c[1] == "l" else 0.73
    cy = 0.27 if c[0] == "t" else 0.73
    return dict(variant=c, box=(cx, cy, 0.46, 0.40 if fmt == "9:16" else 0.44), roll=0.0, shapes=[])


def _edge_l(rng, fmt):
    c = rng.choice(["bl", "br", "tl", "tr"])
    if fmt == "9:16":
        pts, box = [(0, 0.46), (0.065, 0.46), (0.065, 0.70), (1, 0.70), (1, 1), (0, 1)], (0.53, 0.85, 0.88, 0.22)
    else:
        pts, box = [(0, 0.34), (0.04, 0.34), (0.04, 0.62), (1, 0.62), (1, 1), (0, 1)], (0.52, 0.81, 0.90, 0.30)
    fx, fy = c[1] == "r", c[0] == "t"
    pts = [((1 - x) if fx else x, (1 - y) if fy else y) for x, y in pts]
    cx, cy, sw, sh = box
    box = ((1 - cx) if fx else cx, (1 - cy) if fy else cy, sw, sh)
    return dict(variant=c, box=box, roll=0.0, shapes=[_poly(pts, "dark")])


def _spine(rng, fmt):
    side = rng.choice(["left", "right"])
    return dict(variant=side, box=(0.19 if side == "left" else 0.81, 0.5, 0.30, 0.92), roll=0.0, shapes=[])


def _hung_from_top(rng, fmt):
    return dict(variant="top", box=(0.5, 0.30, 0.92, 0.52), roll=0.0, shapes=[])


def _massive_crop(rng, fmt):
    cy = 0.5 + rng.choice([-0.08, 0.0, 0.08])
    return dict(variant="crop", box=(0.5, cy, 1.34, 0.62), roll=0.0, shapes=[])


def _solid_strips(rng, fmt):
    kind = rng.choice(["flat", "angled", "wonky"])
    sign = rng.choice([-1.0, 1.0])
    ang = {"flat": 0.0, "angled": sign * rng.uniform(7.0, 10.0), "wonky": sign * rng.uniform(1.8, 3.2)}[kind]
    cy = rng.choice([0.32, 0.5, 0.68])
    t = 0.30 if fmt == "9:16" else 0.40                     # band thickness, fraction of the height
    sw = 0.90 if fmt == "9:16" else 0.84
    shapes = [_rect(0.5, cy, 1.9, t, "ink2" if kind == "wonky" else "dark", ang)]
    # a thin accent bar, parallel, on the lower edge of the band (Q6): rotate its offset with the band
    off = t / 2.0 + 0.012
    a = math.radians(ang)
    asp = _asp(fmt)
    dx, dy = -math.sin(a) * off * asp, math.cos(a) * off                # offset along the band normal, in frame fractions
    shapes.append(_rect(0.5 + dx, cy + dy, 1.9, 0.016, "accent", ang))
    return dict(variant=kind, box=(0.5, cy, sw, 0.62 * t), roll=(ang if kind == "angled" else 0.0), shapes=shapes)


def _split_sides(rng, fmt):
    side = rng.choice(["a", "b"])                              # a: the type sits in the first block (top / left), b: in the second
    teeth = rng.randint(5, 7)
    base = 0.55 if side == "a" else 0.45                      # where the stepped edge runs
    steps = [min(0.7, max(0.3, base + rng.uniform(-0.05, 0.05))) for _ in range(teeth)]
    mn, mx = min(steps), max(steps)
    ks = [i / teeth for i in range(teeth + 1)]
    cols = ("dark", "ink2") if side == "a" else ("ink2", "dark")          # the type block is the dark one
    if fmt == "9:16":                                         # top / bottom blocks; the stepped edge runs across the width
        edge = [pt for i in range(teeth) for pt in ((ks[i], steps[i]), (ks[i + 1], steps[i]))]
        a = _poly([(0, 0), (1, 0)] + edge[::-1] + [], cols[0])
        b = _poly(edge + [(1, 1), (0, 1)], cols[1])
        box = (0.5, mn / 2.0, 0.88, min(0.42, mn * 0.8)) if side == "a" else             (0.5, mx + (1 - mx) / 2.0, 0.88, min(0.42, (1 - mx) * 0.8))
    else:                                                     # left / right blocks; the stepped edge runs down the height
        edge = [pt for i in range(teeth) for pt in ((steps[i], ks[i]), (steps[i], ks[i + 1]))]
        a = _poly([(0, 0)] + edge + [(0, 1)], cols[0])
        b = _poly([(1, 0)] + edge + [(1, 1)], cols[1])
        box = (mn / 2.0, 0.5, min(0.50, mn * 0.9), 0.62) if side == "a" else             (mx + (1 - mx) / 2.0, 0.5, min(0.50, (1 - mx) * 0.9), 0.62)
    return dict(variant="type_first" if side == "a" else "type_second", box=box, roll=0.0, shapes=[a, b])


def _small_line(rng, fmt):
    cy = rng.choice([0.30, 0.42, 0.58, 0.70])
    return dict(variant="line", box=(0.5, cy, 0.88 if fmt == "9:16" else 0.70, 0.075 if fmt == "9:16" else 0.12), roll=0.0, shapes=[])


def _counters_window(rng, fmt):
    return dict(variant="window", box=(0.5, 0.5, 0.96, 0.86), roll=0.0, shapes=[])


def _type_circle(rng, fmt):
    asp = _asp(fmt)
    if fmt == "9:16":
        r, sw, sh = 0.46, 0.62, 0.46 * asp                    # r of the shorter side (the width); inscribed box 0.62 W x 0.46 W
    else:
        r, sw, sh = 0.44, 0.60 / asp, 0.40
    return dict(variant="disc", box=(0.5, 0.5, sw, sh), roll=0.0,
                shapes=[{"kind": "circle", "cx": 0.5, "cy": 0.5, "r": r, "color": "dark"}])


def _construction_triangle(rng, fmt):
    flip = rng.choice([False, True])                         # apex up (False) / down
    pts = [(0.5, 0.12), (0.93, 0.88), (0.07, 0.88)]
    box = (0.5, 0.72, 0.50, 0.20)
    if flip:
        pts = [(x, 1.0 - y) for x, y in pts]
        box = (0.5, 1.0 - 0.72, 0.50, 0.20)
    return dict(variant="apex_down" if flip else "apex_up", box=box, roll=0.0, shapes=[_poly(pts, "dark")])


def _cells(rng, fmt):
    cols, rows = (2, 2) if fmt == "9:16" else (3, 2)
    x0, x1, y0, y1 = (0.07, 0.93, 0.24, 0.76) if fmt == "9:16" else (0.10, 0.90, 0.20, 0.80)
    g = 0.015
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    shapes = []
    flip = rng.choice([0, 1])
    for j in range(rows):
        for i in range(cols):
            shapes.append(_rect(x0 + (i + 0.5) * cw, y0 + (j + 0.5) * ch, cw - g, ch - g * (16 / 9 if fmt == "9:16" else 1.0),
                                "dark" if (i + j + flip) % 2 == 0 else "ink2"))
    return dict(variant=f"{cols}x{rows}", box=(0.5, 0.5, (x1 - x0) * 0.92, (y1 - y0) * 0.78), roll=0.0, shapes=shapes)


# name -> (plate ids of the notes, formats, anchored, min letters, max letters, max words, maker, note)
LAYOUTS: dict[str, dict] = {
    "full": dict(plate="-", fmts=FORMATS, min_letters=1, max_letters=99, max_words=99, make=_full,
                 note="the full-frame hold every piece had before; no change"),
    "quadrant_stack": dict(plate="N1 R2", fmts=FORMATS, min_letters=5, max_letters=40, max_words=6, make=_quadrant_stack,
                           note="justified stack scaled to the full width of one quadrant (top-right N1, bottom-right R2)"),
    "edge_L": dict(plate="R1", fmts=FORMATS, min_letters=4, max_letters=30, max_words=5, make=_edge_l,
                   note="pushed to the edge, an L-shaped solid in the corner (approximation: one line, no junction word)"),
    "spine": dict(plate="N5", fmts=("9:16",), min_letters=6, max_letters=30, max_words=4, make=_spine,
                  note="book spine: tall narrow stack hard against one edge"),
    "hung_from_top": dict(plate="N8", fmts=FORMATS, min_letters=5, max_letters=40, max_words=6, make=_hung_from_top,
                          note="stack hung from the top edge, empty below"),
    "massive_crop": dict(plate="M2a P1 L1b", fmts=FORMATS, min_letters=3, max_letters=12, max_words=2, make=_massive_crop,
                         note="massive type cropped by the frame at the sides (about 10 % of the word each side)"),
    "solid_strips": dict(plate="Q4 Q6 Q7 Q9 Q12 S8", fmts=FORMATS, min_letters=3, max_letters=24, max_words=4, make=_solid_strips,
                         note="type on a solid flat strip: flat, angled (7-10 deg, text rolls with it) or slightly wonky (2-3 deg, text straight)"),
    "split_sides": dict(plate="S1 S11", fmts=FORMATS, min_letters=4, max_letters=28, max_words=5, make=_split_sides,
                        note="blocky contrasted halves with a stepped edge (liked S1); the type sits on one block"),
    "small_line": dict(plate="v005 P20", fmts=FORMATS, min_letters=4, max_letters=16, max_words=3, make=_small_line, max_lines=1,
                       note="one precise small line (cap height >= 4 % of the frame height, MIN_CAP_FRAC) set where the footage's edges run"),
    "counters_window": dict(plate="P4", fmts=FORMATS, min_letters=3, max_letters=14, max_words=2, make=_counters_window,
                            modules=WINDOW_MODULES, needs_counter=True,
                            note="huge word, letters are windows (fly_into / depth_tunnel); words inside the counters not placed"),
    "type_circle": dict(plate="M5a", fmts=FORMATS, min_letters=3, max_letters=18, max_words=3, make=_type_circle,
                        note="a flat disc with the line inside (approximation of the circle of type)"),
    "construction_triangle": dict(plate="M3b", fmts=FORMATS, min_letters=3, max_letters=16, max_words=3, make=_construction_triangle,
                                  note="a flat construction triangle, the line sits in its wide part"),
    "cells": dict(plate="P9", fmts=FORMATS, min_letters=4, max_letters=24, max_words=4, make=_cells,
                  note="the line over a grid of flat contrasting cells"),
}
NAMES = tuple(LAYOUTS)
DEFAULT = "full"


def names(fmt: str | None = None) -> list[str]:
    return [n for n in NAMES if fmt is None or fmt in LAYOUTS[n]["fmts"]]


# ------------------------------------------------------------------------------------------------ letter size of a layout (v005 round 2)
_REF_FONT: list = []


def _fontd(font):
    from mograph.modules import core
    key, loc = core.resolve_font(font)
    return dict(key=key, axes=loc)


def reference_font() -> str:
    """the liked font the planner uses to judge a layout before the line's font is chosen: the 80th percentile of the liked display faces by
    width per cap height (a wider face than most, so the estimate errs towards a bigger layout)."""
    if not _REF_FONT:
        from mograph import fonts_liked as fl
        from mograph.modules import core
        rows = []
        for e in fl.LIKED:
            try:
                lay = core.layout_word(_fontd(e["file"]), ["ABCDEFGH"], 1000.0, 1000.0)
                rows.append((lay["size"][0] / lay["cap"], e["file"]))
            except Exception:                                     # noqa: BLE001  (a face that cannot be laid out is not a reference)
                continue
        rows.sort()
        _REF_FONT.append(rows[int(0.8 * (len(rows) - 1))][1] if rows else "Ultra-Regular.ttf")
    return _REF_FONT[0]


def hold_box_frac(rec: dict | None, fmt: str = "9:16", fill: float = 0.7) -> tuple[float, float]:
    """(w, h) of the box the held line is fitted into, as fractions of the frame: the layout's box (fit_lines uses 94 % x 90 % of it), or for the
    full-frame hold what the hold_drift camera frames at `fill` (cam3d FIT_FILLK / FIT_HK)."""
    if rec and rec.get("box"):
        return 0.94 * rec["box"]["sw"], 0.90 * rec["box"]["sh"]
    fillk, hk = FILL[fmt]
    return fill * fillk, fill * hk


_CAP_CACHE: dict = {}


def cap_frac(text: str, rec: dict | None, font=None, fmt: str = "9:16", lines=None, fill: float = 0.7) -> float:
    """cap (letter) height of the held line as a fraction of the frame height: the planner's own line fit (spec.fit_lines: best line breaks,
    squeeze ar_min of the layout) in the layout box. Face-on, flat letters (extrusion and yaw not counted)."""
    b = (rec or {}).get("box") or {}
    key = (text, str(font), fmt, tuple(lines) if lines else None, round(float(fill), 3), tuple(sorted(b.items())), (rec or {}).get("ar_min"), (rec or {}).get("max_lines"))
    if key not in _CAP_CACHE:
        _CAP_CACHE[key] = _cap_frac(text, rec, font, fmt, lines, fill)
    return _CAP_CACHE[key]


def _cap_frac(text, rec, font, fmt, lines, fill) -> float:
    from tools.fusion import spec as sp
    fw, fh = FRAME_PX[fmt]
    bw, bh = hold_box_frac(rec, fmt, fill)
    ar_min = rec.get("ar_min") if rec else None
    ml = (rec.get("max_lines") if rec else None) or 5
    try:
        _, _, cap = sp.fit_lines(_fontd(font or reference_font()), (text or "").upper(), bw * fw, bh * fh, ar_min, ml, lines)
    except Exception:                                             # noqa: BLE001  (no font engine: a plain width estimate, 0.95 cap per letter)
        n = max(_letters(w) for w in (text or "A").split()) if text else 1
        cap = min(bw * fw / (0.95 * max(n, 1)), bh * fh)
    return float(cap) / fh


def min_cap(name: str) -> float:
    """the least cap height (fraction of the frame height) a hold of this layout may have: 7 %, a small_line 4 %."""
    return MIN_CAP_FRAC if name in RESERVED else HOLD_CAP_FRAC


def cap_ok(rec: dict, text: str, font=None, fmt: str = "9:16", lines=None, fill: float = 0.7, tol: float = 0.0) -> bool:
    """does this layout record give the line a big enough letter for its hold?"""
    return cap_frac(text, rec, font, fmt, lines, fill) >= min_cap(rec["name"]) - tol


def eligible(name: str, text: str = "", fmt: str = "9:16", module: str | None = None, font=None, lines=None, size_check: bool = True) -> bool:
    """can ``text`` (a hold line) take this layout in this format, with this word module? size_check (default on): the layout must give the line a
    letter height >= HOLD_CAP_FRAC of the frame (4 % for small_line, which the planner hands to the edge_run block only: batch_plan)."""
    d = LAYOUTS[name]
    n = _letters(text) if text else None
    if fmt not in d["fmts"]:
        return False
    if n is not None and not (d["min_letters"] <= n <= d["max_letters"] and len(text.split()) <= d["max_words"]):
        return False
    if d.get("modules") and module is not None and module not in d["modules"]:
        return False
    if d.get("needs_counter") and text and not _COUNTER_RE.search(text.upper()):
        return False
    if module in BIG_BOX_MODULES and name != DEFAULT:
        b = build(name, 0, "big", fmt).get("box")
        if b and b["sw"] < BIG_BOX_W:
            return False
    if size_check and text and name != DEFAULT:
        rec = build(name, 0, "cap", fmt)
        if not cap_ok(rec, text, font, fmt, lines):
            return False
    return True


def build(name: str, seed=0, tag: str = "", fmt: str = "9:16") -> dict:
    """the layout record of ``name`` (deterministic in seed + tag: the variant, corner, angle, step pattern)."""
    d = LAYOUTS[name]
    if fmt not in d["fmts"]:
        raise ValueError(f"layout {name!r} is not defined for {fmt}")
    m = d["make"](_rng(seed, f"{name}:{tag}"), fmt)
    box = m["box"]
    rec = {"name": name, "plate": d["plate"], "variant": m["variant"], "roll": _r(m["roll"], 2), "shapes": m["shapes"],
           "anchored": bool(m["shapes"]), "ar_min": AR_MIN if box else None, "note": d["note"], "max_lines": d.get("max_lines")}
    if box:
        cx, cy, sw, sh = (float(v) for v in box)
        rec["box"] = {"cx": _r(cx), "cy": _r(cy), "sw": _r(sw), "sh": _r(sh)}
        rec["offset"] = [_r(cx - 0.5), _r(cy - 0.5)]
        rec["fill"] = camera_fill(sw, sh, fmt)
        rec["area"] = _r(min(1.2, sw * sh))
    else:
        rec.update(box=None, offset=None, fill=None, area=1.0)
    return rec


def placement(rec: dict) -> dict:
    """spec word `placement` keys of a layout: wf / hf = box size as fractions of the world frame, roll; {} for the full-frame hold.
    Anchored layouts are flat (yaw 0) so the type sits square on its shape."""
    if not rec.get("box"):
        return {}
    out = {"wf": rec["box"]["sw"], "hf": rec["box"]["sh"], "roll": rec.get("roll", 0.0)}
    if rec.get("anchored"):
        out["yaw"] = 0.0
    return out


# Module params that make the hold pose of a word module read in a piece (spec.build_scene applies them under the word's own `params`). The gallery
# modules (24, approved) are untouched: these are the pose a held line settles to, found on the type pass of the v005 pieces (2026-10-27 round 2):
#   fly_into / outflow   the ring train streams past the lip: a tilt of 0.9 leaves ghost copies of the word above and below it; 0.4 and fewer rings keep them close
#   xray_unfold          14 slices at 1.3 cap depth are a wire cage: 9 slices at 0.8 cap keep the face outline readable
#   voxel_roll           the grid floor swallowed the cubes (letters lost in the grid); lean 6 = nearly a wall of flat cube faces (a pixel face)
#   scale_clash          the huge word's extrusion ran 1.4 widths across the frame, 26 slabs: 0.35 widths, 12 slabs
#   solid_hollow         extrusion axis leaned 0.3 into the page, the walls hid the cap: 1.2 turns the extrusion toward the viewer
#   candy_orbit          42 banded copies of a thin face drown the face: 24 copies, bands of 2
#   voxel_orbit          random depth scatter of +-14 % cap breaks the lattice seen from a hold camera that is not at 1000 px: 20 px
MODULE_HOLD_PARAMS: dict[str, dict] = {
    "fly_into": dict(tilt=0.4, rings=12), "outflow": dict(tilt=0.3, rings=6), "xray_unfold": dict(depth=0.8, slices=9),
    "voxel_roll": dict(grid=False, lean=6.0), "scale_clash": dict(reach=0.35, slabs=12), "solid_hollow": dict(depth=1.2),
    "candy_orbit": dict(copies=24, band=2), "voxel_orbit": dict(depth=20.0),
}


def hold_params(module: str | None) -> dict:
    """default module params for a held line of ``module`` (a copy; the word's own params override them)."""
    return dict(MODULE_HOLD_PARAMS.get(module or "", {}))


TAME_MODULES = ("fly_into", "outflow")          # ring-train modules: ghost copies of the word above / below at a steep tilt or a long train
TAME_MAX = {"fly_into": dict(tilt=0.4, rings=12), "outflow": dict(tilt=0.3, rings=6)}      # ceilings of a held word (= their MODULE_HOLD_PARAMS)


def tame_params(module: str | None, params: dict | None = None) -> dict:
    """params of a held word of ``module``: hold_params under the word's own ``params``, then the ring modules (TAME_MODULES) CLAMPED to TAME_MAX, so a word's own tilt 0.9 / rings 20 can no longer
    undo the tame (2026-11-01: GOLDEN HOUR / ASCEND held as a dense hatch of ring copies). Other modules: hold_params + own params unchanged."""
    out = {**hold_params(module), **(params or {})}
    for k, cap in TAME_MAX.get(module or "", {}).items():
        v = out.get(k)
        out[k] = min(v, cap) if isinstance(v, (int, float)) and not isinstance(v, bool) else cap
    return out


PULLBACK_HOLD = dict(from_scale=0.9, word_scale=1.2)     # pullback_reveal runs over the whole word item: its defaults (tight at 0.5 x fit, rest at 1.45 x) crop the word at the hold (v005 P21 THE CITY)


def camera_params(rec: dict, camera: str) -> dict:
    """params of the camera move held on this line: the framing fill, and for hold_drift no off-centre comp (the layout places the
    word), no dolly when the line carries shapes (the type must stay on its strip)."""
    if not rec.get("box"):
        return dict(PULLBACK_HOLD) if camera == "pullback_reveal" else {}
    p = {"fill": rec["fill"]}
    if camera == "pullback_reveal":
        p.update(PULLBACK_HOLD)
    if camera == "hold_drift":
        p["comp"] = [0.0, 0.0]
        if rec.get("anchored"):
            p.update(dolly=0.0, drift_px=0.8)
    return p


def validate(rec: dict, fmt: str = "9:16") -> list[str]:
    """problems of a layout record against the notes' rules ([] = valid)."""
    bad = []
    name = rec.get("name")
    if name not in LAYOUTS:
        return [f"unknown layout {name!r}"]
    if fmt not in LAYOUTS[name]["fmts"]:
        bad.append(f"{name} not defined for {fmt}")
    box = rec.get("box")
    if box:
        if not (0.0 <= box["cx"] <= 1.0 and 0.0 <= box["cy"] <= 1.0):
            bad.append("box centre outside the frame")
        if not (0.15 <= box["sw"] <= 1.4 and 0.06 <= box["sh"] <= 1.0):
            bad.append(f"box size {box['sw']}x{box['sh']} out of range")
        if name != "massive_crop" and (box["cx"] - box["sw"] / 2 < -1e-6 or box["cx"] + box["sw"] / 2 > 1 + 1e-6):
            bad.append("only massive_crop may be cropped by the frame")
        if rec.get("ar_min") is None or rec["ar_min"] < AR_MIN - 1e-9:
            bad.append("type may be squeezed too far (ar_min)")
        if abs(rec.get("roll", 0.0)) > MAX_ROLL:
            bad.append("roll too steep to read")
    elif rec.get("shapes"):
        bad.append("shapes without a text box")
    shapes = rec.get("shapes") or []
    if len(shapes) > MAX_SHAPES:
        bad.append(f"{len(shapes)} shapes: keep shapes simple (<= {MAX_SHAPES})")
    for s in shapes:
        if s.get("kind") not in SHAPE_KINDS:
            bad.append(f"shape kind {s.get('kind')!r}")
        if s.get("color") not in COLOR_ROLES:
            bad.append(f"shape colour role {s.get('color')!r}")
        if "text" in s or "label" in s:
            bad.append("shapes carry no text (one idea per frame)")
        if s.get("kind") == "poly" and len(s.get("pts") or []) < 3:
            bad.append("polygon with < 3 points")
    if bool(shapes) != bool(rec.get("anchored")):
        bad.append("anchored must match the presence of shapes")
    return bad
