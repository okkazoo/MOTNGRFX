"""Shapes-and-sizes text options: rotation (static, keyframes, in/out spin), text on a path (arc / circle / wave /
line), off-frame boxes + overscale, hairline backdrops (dim / blur / band, automatic for "fine" faces over images)
and per-letter sizes (letter_scale)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import plan as P  # noqa: E402
from mograph.geometry import frame_size  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from mograph.render.common import Ctx  # noqa: E402
from mograph.render.text import TextLayer, rotation_at  # noqa: E402
from render_helpers import base_plan, text_layer  # noqa: E402

FPS = 20
NOIMG = {"images": []}
BIG = {"9:16": [0.05, 0.3, 0.9, 0.3], "16:9": [0.05, 0.25, 0.9, 0.4]}


def _plan(layers, duration=1.0, display="Big Shoulders Display", bg=None):
    p = base_plan(duration, FPS)
    p["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
    p["style"]["fonts"] = {"display": display, "body": "Space Grotesk"}
    p["scenes"] = [{"id": "s0", "t0": 0.0, "t1": duration,
                    "background": bg or {"type": "color", "color": "palette:0"},
                    "layers": layers, "transition_out": {"type": "cut", "dur": 0.0}}]
    return p


def _frames(plan, analysis=NOIMG, fmt="9:16"):
    return [f.astype(np.int32) for f in render_frames(plan, analysis, fmt)]


def _lit(f):
    return f.mean(axis=2) > 128


def _ink(f):
    return float(_lit(f).mean())


def _bbox(f):
    ys, xs = np.nonzero(_lit(f))
    return xs.min(), ys.min(), xs.max(), ys.max()


def _layer(lay, plan, fmt="9:16", analysis=NOIMG, on_image=False):
    W, H = frame_size(fmt, "draft")
    ctx = Ctx(plan, analysis, fmt, W, H)
    return TextLayer(lay, ctx, 0, on_image), ctx


def _grey_image(noise=False):
    px = np.full((400, 300, 3), 0.6, np.float32)
    if noise:
        px = np.random.default_rng(1).random((400, 300, 3)).astype(np.float32)
    return {"images": [{"id": "img0", "pixels": px, "width": 300, "height": 400, "focus": [0.5, 0.5],
                        "contours": [], "lines": []}]}


IMG_BG = {"type": "image", "src": "img0", "focus": [0.5, 0.5], "motion": {"verb": "static"},
          "treatment": {"duotone": None, "blur": 0.0, "darken": 0.0}}


# ----------------------------------------------------------------------------- rotation

def test_rotate_static_turns_the_block():
    flat = _frames(_plan([text_layer("WIDETEXT", 0.0, 1.0, "cut", "cut", box=BIG, fit="none")]))[5]
    lay = text_layer("WIDETEXT", 0.0, 1.0, "cut", "cut", box=BIG, fit="none", rotate=90)
    p = _plan([lay])
    assert P.validate_plan(p, NOIMG) == []
    fr = _frames(p)
    x0, y0, x1, y1 = _bbox(flat)
    assert x1 - x0 > 2.5 * (y1 - y0)                        # flat: wide
    X0, Y0, X1, Y1 = _bbox(fr[5])
    assert Y1 - Y0 > 2.5 * (X1 - X0)                        # rotated 90: tall
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2                   # about the block centre
    assert abs((X0 + X1) / 2 - cx) < 6 and abs((Y0 + Y1) / 2 - cy) < 6
    assert abs(_ink(fr[5]) - _ink(flat)) < 0.25 * _ink(flat)    # same letters, turned
    assert all(np.array_equal(a, b) for a, b in zip(fr, _frames(p)))     # deterministic


def test_rotation_canvas_brings_off_frame_type_in():
    # block centred on the left frame edge: half the word is off-frame; turned 180 about its centre that half
    # swings into the frame, so it must have been drawn (expanded canvas), not clipped
    box = {"9:16": [-0.6, 0.45, 1.2, 0.1], "16:9": [-0.6, 0.45, 1.2, 0.1]}
    p0 = _plan([text_layer("LEFTRIGHT", 0.0, 1.0, "cut", "cut", box=box, fit="none")])
    p1 = _plan([text_layer("LEFTRIGHT", 0.0, 1.0, "cut", "cut", box=box, fit="none", rotate=180)])
    assert P.validate_plan(p1, NOIMG) == []
    a, b = _frames(p0)[5], _frames(p1)[5]
    assert _ink(a) > 0 and _ink(b) > 0.6 * _ink(a)
    assert not np.array_equal(a, b)


def test_rotation_keyframes_and_spin():
    assert rotation_at(None, 1.0) == 0.0 and rotation_at(-8, 3.0) == -8.0
    keys = [[0.0, 0.0], [1.0, 90.0]]
    assert rotation_at(keys, -1.0) == 0.0 and rotation_at(keys, 2.0) == 90.0
    assert abs(rotation_at(keys, 0.5) - 45.0) < 1e-6
    lay = text_layer("SPIN", 0.0, 2.0, "fade", "fade", box=BIG, rotate=-8,
                     **{"in": {"verb": "fade", "dur": 0.6, "spin": 180}, "out": {"verb": "fade", "dur": 0.4, "spin": 30}})
    p = _plan([lay], 2.0)
    assert P.validate_plan(p, NOIMG) == []
    L, _ = _layer(lay, p)
    assert abs(L.rotation(0.0) - 172.0) < 1e-6           # -8 + 180 at the start of the in-phase
    assert abs(L.rotation(1.0) + 8.0) < 1e-6              # settled
    assert abs(L.rotation(1.999) - 22.0) < 1.0            # out-phase turns +30
    fr = _frames(p)
    assert np.abs(fr[2] - fr[5]).mean() > 0.2             # moving while it spins in


# ----------------------------------------------------------------------------- off-frame / overscale

def test_overscale_runs_past_the_frame():
    box = {"9:16": [-0.2, 0.35, 1.4, 0.3], "16:9": [-0.2, 0.2, 1.4, 0.6]}
    lay = text_layer("HEART", 0.0, 1.0, "cut", "cut", box=box, overscale=1.6)
    p = _plan([lay], display="Rock 3D")
    assert P.validate_plan(p, NOIMG) == []
    base, _ = _layer(dict(lay, overscale=1.0), p)
    big, _ = _layer(lay, p)
    assert abs(big.size / base.size - 1.6) < 0.05
    for fmt in ("9:16", "16:9"):
        f = _frames(p, fmt=fmt)[5]
        x0, _, x1, _ = _bbox(f)
        assert x0 == 0 and x1 == f.shape[1] - 1           # cropped by both frame edges


# ----------------------------------------------------------------------------- text on a path

def test_arc_glyphs_sit_on_the_circle_and_follow_the_tangent():
    lay = text_layer("around the heart", 0.0, 1.0, "cut", "cut", box=BIG, font="Nixie One",
                     path={"type": "arc", "radius": 0.9, "angle": 0, "span": 160})
    p = _plan([lay])
    assert P.validate_plan(p, NOIMG) == []
    L, _ = _layer(lay, p)
    P_ = L.tpath
    x, y, th, sc, adv = L._path_place(0.0)
    r = np.hypot(x - P_["c"][0], y - P_["c"][1])
    assert np.allclose(r, P_["r"], atol=1e-6)
    ang = np.arctan2(x - P_["c"][0], -(y - P_["c"][1]))   # clockwise from the top
    assert np.allclose(np.unwrap(ang), np.unwrap(th), atol=1e-6)
    assert ang.min() < -0.5 and ang.max() > 0.5 and abs(ang.mean()) < 0.1     # centred on the top
    span = (ang.max() - ang.min()) * P_["r"] + adv[0] / 2 + adv[-1] / 2
    assert span <= math.radians(160) * P_["r"] * 1.01   # fitted into the span
    fr = _frames(p)
    ys, xs = np.nonzero(_lit(fr[5]))
    d = np.hypot(xs + 0.5 - P_["c"][0], ys + 0.5 - P_["c"][1])
    assert d.min() > P_["r"] - 0.3 * L.psize and d.max() < P_["r"] + 1.2 * L.psize


def test_circle_travel_moves_and_is_deterministic():
    lay = text_layer("round and round we go", 0.0, 1.0, "cut", "cut", box=BIG, font="Interlope",
                     path={"type": "circle", "travel": 0.25})
    p = _plan([lay])
    L, _ = _layer(lay, p)
    a = L._path_place(0.0)
    b = L._path_place(1.0)
    assert np.allclose(np.hypot(b[0] - L.tpath["c"][0], b[1] - L.tpath["c"][1]), L.tpath["r"])
    assert np.allclose((b[2] - a[2]) % (2 * math.pi), math.pi / 2)     # a quarter turn per second
    fr = _frames(p)
    assert np.abs(fr[2] - fr[12]).mean() > 0.1
    assert all(np.array_equal(u, v) for u, v in zip(fr, _frames(p)))


def test_flipped_arc_reads_along_the_bottom():
    lay = text_layer("underneath", 0.0, 1.0, "cut", "cut", box=BIG, path={"type": "arc", "flip": True})
    L, _ = _layer(lay, _plan([lay]))
    x, y, th, _, _ = L._path_place(0.0)
    assert np.all(np.diff(x) > 0)                         # left to right
    assert np.all(y > L.tpath["c"][1])                     # below the centre
    assert np.allclose(np.cos(th[len(th) // 2]), 1.0, atol=0.05)    # upright at the bottom


def test_wave_phase_steps_on_beats():
    lay = text_layer("riding the wave", 0.0, 2.0, "cut", "cut", box=BIG, font="Interlope",
                     path={"type": "wave", "amp": 0.3, "cycles": 1.0, "on": "beats", "step": 0.25, "offset": 0})
    p = _plan([lay], 2.0)
    assert P.validate_plan(p, NOIMG) == []
    L, _ = _layer(lay, p)
    assert L._wave_phase(0.45) == L._wave_phase(0.4) == 0.25        # beat at 0 has settled
    assert abs(L._wave_phase(0.9) - 0.5) < 1e-3                       # next beat at 0.5 s
    y1, y2 = L._path_place(0.4)[1], L._path_place(0.9)[1]
    assert np.abs(y1 - y2).max() > 2.0
    # glyph tangent matches the curve slope
    x, y, th, _, _ = L._path_place(0.4)
    P_ = L.tpath
    arg = P_["k"] * (x - P_["x0"]) + 2 * math.pi * 0.25
    assert np.allclose(np.tan(th), P_["A"] * P_["k"] * np.cos(arg), atol=1e-6)


def test_line_path_runs_along_the_diagonal_with_letter_in():
    lay = text_layer("DIAGONAL", 0.0, 1.0, "word_stagger", "cut", box=BIG, path={"type": "line"},
                     **{"in": {"verb": "word_stagger", "dur": 0.3}})
    p = _plan([lay])
    L, ctx = _layer(lay, p)
    x, y, th, _, _ = L._path_place(1.0)
    assert np.allclose(th, th[0]) and th[0] < 0           # rising left to right
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) < 0)
    fr = _frames(p)
    assert 0 < _ink(fr[3]) < _ink(fr[15])                 # letters arrive one by one


# ----------------------------------------------------------------------------- letter_scale

def test_letter_scale_random_seeded_and_refits():
    def mk(seed):
        return text_layer("SHAPES", 0.0, 1.0, "cut", "cut", box=BIG, letter_scale={"type": "random", "seed": seed})
    p = _plan([mk(3)], display="Fascinate")
    assert P.validate_plan(p, NOIMG) == []
    a, _ = _layer(mk(3), p)
    b, _ = _layer(mk(3), p)
    c, _ = _layer(mk(4), p)
    assert np.array_equal(a._ls_static, b._ls_static) and not np.array_equal(a._ls_static, c._ls_static)
    assert abs(a._ls_static.max() - 1.0) < 1e-9 and a._ls_static.min() >= 0.45 - 1e-9
    plain, _ = _layer(text_layer("SHAPES", 0.0, 1.0, "cut", "cut", box=BIG), p)
    assert a.size >= plain.size                            # smaller letters leave room: refitted bigger
    f = _frames(p)[5]
    x0, _, x1, _ = _bbox(f)
    W = f.shape[1]
    assert x0 >= 0.05 * W - 2 and x1 <= 0.95 * W + 2       # still inside the box
    assert (x1 - x0) > 0.75 * 0.9 * W


def test_letter_scale_list_wave_and_bounce():
    lay = text_layer("ABC", 0.0, 1.0, "cut", "cut", box=BIG, letter_scale=[1, 2, 4])
    L, _ = _layer(lay, _plan([lay]))
    assert np.allclose(L._ls_static, [0.25, 0.5, 1.0])
    wav = text_layer("SIZES", 0.0, 2.0, "cut", "cut", box=BIG, letter_scale="wave")
    p = _plan([wav], 2.0)
    assert P.validate_plan(p, NOIMG) == []
    W_, _ = _layer(wav, p)
    assert not np.allclose(W_._lscales(0.0), W_._lscales(0.5)) and W_._lscales(0.3).max() <= 1.0 + 1e-9
    fr = _frames(p)
    assert np.abs(fr[3] - fr[9]).mean() > 0.1
    bounce = text_layer("BOUNCE", 0.0, 1.5, "kinetic_split", "cut", box=BIG, letter_scale="random",
                        **{"in": {"verb": "kinetic_split", "dur": 0.5, "stagger": 0.08}})
    fb = _frames(_plan([bounce], 1.5))
    assert _ink(fb[0]) < _ink(fb[3]) < _ink(fb[22]) or _ink(fb[3]) < _ink(fb[22])


# ----------------------------------------------------------------------------- backdrop

def _region_mean(f, box, fmt="9:16"):
    H, W = f.shape[:2]
    x, y, w, h = box[fmt]
    return float(f[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)].mean())


def test_backdrop_auto_for_fine_faces_over_images():
    ana = _grey_image()
    fine = text_layer("hairline", 0.0, 1.0, "cut", "cut", box=BIG, font="Nixie One", color="palette:4")
    p = _plan([fine], bg=IMG_BG)
    assert P.validate_plan(p, ana) == []
    L, _ = _layer(fine, p, analysis=ana, on_image=True)
    assert L.backdrop is not None and L.backdrop["type"] == "dim"
    off = dict(fine, backdrop="none")
    f_on = _frames(p, ana)[5]
    f_off = _frames(_plan([off], bg=IMG_BG), ana)[5]
    far = {"9:16": [0.0, 0.0, 1.0, 0.1]}
    assert _region_mean(f_on, far) == _region_mean(f_off, far)          # local only
    assert _region_mean(f_on, BIG) < _region_mean(f_off, BIG) - 10      # darker behind the text
    # puffy face / colour background: no automatic backdrop
    L2, _ = _layer(text_layer("FAT", 0.0, 1.0, "cut", "cut", box=BIG, font="Rock 3D"), p, analysis=ana, on_image=True)
    L3, _ = _layer(fine, p, analysis=ana, on_image=False)
    assert L2.backdrop is None and L3.backdrop is None


def test_backdrop_blur_and_band():
    ana = _grey_image(noise=True)
    small = {"9:16": [0.05, 0.42, 0.9, 0.08], "16:9": [0.05, 0.42, 0.9, 0.08]}
    plain = text_layer("x", 0.0, 1.0, "cut", "cut", box=small, color="palette:0", backdrop="none")
    blur = dict(plain, backdrop={"type": "blur", "amount": 1.0})
    band = dict(plain, backdrop={"type": "band", "amount": 1.0, "color": "#000000"})
    for lay in (blur, band):
        assert P.validate_plan(_plan([lay], bg=IMG_BG), ana) == []
    a = _frames(_plan([plain], bg=IMG_BG), ana)[5].astype(np.float32)
    b = _frames(_plan([blur], bg=IMG_BG), ana)[5].astype(np.float32)
    c = _frames(_plan([band], bg=IMG_BG), ana)[5].astype(np.float32)
    H, W = a.shape[:2]
    patch = (slice(int(0.395 * H), int(0.415 * H)), slice(int(0.45 * W), int(0.55 * W)))   # above the glyph
    assert b[patch].std() < 0.5 * a[patch].std()          # smoothed behind the text
    rows = slice(int(0.43 * H), int(0.49 * H))
    assert c[rows, :8].mean() < 0.2 * a[rows, :8].mean()  # the band runs to the frame edge
    assert np.array_equal(c[:int(0.3 * H)], a[:int(0.3 * H)])


# ----------------------------------------------------------------------------- validation

def test_validation_rejects_bad_shape_options():
    def errs(**opts):
        return P.validate_plan(_plan([text_layer("X", 0.0, 1.0, "cut", "cut", box=BIG, **opts)]), NOIMG)
    assert errs(rotate=[[0, 1], [1, 2]]) == [] and errs(rotate=-8) == []
    assert errs(rotate="up") and errs(rotate=[[0, 1, 2]])
    assert errs(path={"type": "spiral"}) and errs(path={"type": "arc", "span": 720}) and errs(path="arc")
    assert errs(path={"type": "wave", "on": "bars"}) and errs(path={"type": "line", "p0": [0]})
    assert errs(backdrop={"type": "glow"}) and errs(backdrop={"type": "dim", "amount": 3})
    assert errs(backdrop=False) == [] and errs(backdrop="none") == [] and errs(backdrop="blur") == []
    assert errs(letter_scale=[1, -1]) and errs(letter_scale="huge") and errs(letter_scale={"type": "wave", "valign": "top"})
    assert errs(overscale=50) and errs(overscale=2.5) == []
    assert errs(**{"in": {"verb": "fade", "dur": 0.3, "spin": "lots"}})
