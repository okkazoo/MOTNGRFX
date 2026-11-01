"""tools/fusion: portrait world plane, variable-font axes, and the synchronised passes (small CPU render).
Run: python -m pytest tests/test_fusion_passes.py -q"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
from mograph.render import cam3d  # noqa: E402
from mograph.modules import core, Placement  # noqa: E402
from tools.fusion import axes, spec as fspec, type_passes  # noqa: E402
from tools.fusion.beats import Beats  # noqa: E402

FONTS = ["Bytesized-Regular.ttf", "Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf", "Rock3D-Regular.ttf", "Doto[ROND,wght].ttf",
         "PaletteMosaic-Regular.ttf", "Foldit[wght].ttf", "MoiraiOne-Regular.ttf", "MajorMonoDisplay-Regular.ttf",
         "Tourney[wdth,wght].ttf", "BadeenDisplay-Regular.ttf", "GeistPixel[ELSH].ttf"]


def test_world_default_and_context():
    assert (cam3d.W, cam3d.H) == (960, 540)
    with cam3d.world(540, 960):
        assert (cam3d.W, cam3d.H) == (540, 960)
        assert np.allclose(cam3d.CEN, [270, 480])
        pl = Placement.centred()
        assert np.allclose(pl.origin, [270, 480, 0])
        assert cam3d.FIT_FILLK > 1.0
    assert (cam3d.W, cam3d.H) == (960, 540) and np.allclose(cam3d.CEN, [480, 270]) and cam3d.FIT_FILLK == 1.0


def test_all_requested_fonts_resolve_and_lay_out():
    for f in FONTS:
        key, loc = core.resolve_font(f)
        lay = core.layout_word(dict(key=key, axes=loc or {}), ["AFTER"], 480, 300)
        assert len(lay["letters"]) == 5, f
    assert core.resolve_font("Rock 3D")[0] == "rock3d"


def test_tall_faces_and_union_flag():
    tall = core.resolve_font("BigShouldersDisplay[wght].ttf")[0]      # liked tall face (Anton is not liked)
    a = core.layout_word(dict(key=tall), ["AFTER"], 500, 900, ar=1.0)
    b = core.layout_word(dict(key=tall, ar=0.5), ["AFTER"], 500, 900)
    assert b["size"][0] < 0.6 * a["size"][0] or b["scale"] > a["scale"]
    bit = core.layout_word(dict(key="bitcount", axes={"wght": 700}), ["AF"], 400, 300)
    assert any(getattr(l["cons"], "union", False) for l in bit["letters"])


def test_axes_tracks_and_quantise():
    g = Beats(bpm=120, duration=10)
    tr = axes.AxisTrack([300, 800, "beat"], g)
    assert tr(0.0) < 400 and tr(0.06) > 700 and tr(0.49) < 450          # attack on the beat, decay to the next one
    assert axes.canon_axes("Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf", {"WGHT": 5000})["wght"] == 900.0
    an = axes.AxesAnim("Doto[ROND,wght].ttf", {"wght": [300, 900, "beat"], "ROND": 100}, g, quant=16)
    assert an.animated and len({an.key(t) for t in np.linspace(0, 2, 60)}) <= 17


def _spec(size):
    return {"size": list(size), "fps": 12, "duration": 1.6, "bpm": 120,
            "words": [{"text": "AB", "module": "lanes", "font": "Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf",
                       "axes": {"wght": [300, 700, "beat"]}, "form_in": "flicker", "form_out": "flicker", "t0": 0, "t1": 1.0,
                       "form_s": 0.3, "exit_s": 0.3},
                      {"text": "CD", "module": "lanes", "font": "Tourney[wdth,wght].ttf", "form_in": "flicker", "form_out": "lines",
                       "t0": 1.0, "t1": 1.6, "form_s": 0.3, "exit_s": 0.3}],
            "camera": "beat_orbit"}


def test_passes_are_aligned(tmp_path):
    out = str(tmp_path / "p")
    man = type_passes.render_passes(_spec((192, 320)), out, frames=(2, 5), procs=1, mp4=False, verbose=False)
    w, h = 192, 320
    for i in (2, 3, 4):
        beauty = cv2.imread(f"{out}/beauty/beauty_{i:05d}.png")
        matte = cv2.imread(f"{out}/matte/matte_{i:05d}.png", cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(f"{out}/depth/depth_{i:05d}.png", cv2.IMREAD_UNCHANGED)
        line = cv2.imread(f"{out}/line/line_{i:05d}.png", cv2.IMREAD_UNCHANGED)
        clay = cv2.imread(f"{out}/clay/clay_{i:05d}.png", cv2.IMREAD_UNCHANGED)
        fl = np.load(f"{out}/flow/flow_{i:05d}.npy")
        assert beauty.shape == (h, w, 3) and matte.shape == (h, w) and depth.shape == (h, w) and line.shape == (h, w)
        assert clay.shape == (h, w, 3) and np.array_equal(clay[..., 0], clay[..., 1]) and np.array_equal(clay[..., 1], clay[..., 2])
        assert fl.shape == (h // 2, w // 2, 2) and fl.dtype == np.float16
        bg = np.array(man["spec"].get("palette", {}).get("bg", (12, 12, 11)), np.float32)[::-1]
        pred = np.abs(beauty.astype(np.float32) - bg).max(2) > 40
        if pred.sum() > 50:
            inside = (matte > 12)
            assert (pred & inside).sum() / max(pred.sum(), 1) > 0.9           # matte covers the beauty's type
            assert (depth[matte < 8] == 0).mean() > 0.98                       # depth only where the type is
            assert depth[matte > 200].min() >= 32 or (depth[matte > 200] > 0).mean() > 0.95
    t = man["timing"]["standalone_s_per_frame_per_pass"]
    assert set(t) >= {"beauty", "matte", "depth", "line", "clay", "flow"}
    assert type_passes.make_sheet(out, str(tmp_path / "s.png"), n=3, idx=[2, 3, 4])


def test_size_must_be_multiple_of_64():
    import pytest
    with pytest.raises(ValueError):
        fspec.check_size((576, 1000))
