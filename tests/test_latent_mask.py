"""Latent-mask maths (tools/fusion/latent_mask.py): shapes, timing, dilation, combine with an existing mask.
Pure numpy; the node wrapper test at the end needs torch + comfy_api and skips otherwise."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import latent_mask as lm  # noqa: E402

H, W, F = 1024, 576, 121
T, HL, WL = 16, 32, 18


def blob(frames=F, y=(300, 332), x=(320, 352), t=None):
    m = np.zeros((frames, H, W), np.float32)
    if t is None:
        m[:, y[0]:y[1], x[0]:x[1]] = 1.0
    else:
        m[t, y[0]:y[1], x[0]:x[1]] = 1.0
    return m


def test_frame_counts():
    assert lm.latent_frames(121) == 16
    assert lm.latent_frames(9) == 2
    assert lm.latent_frames(1) == 1
    assert lm.pixel_frames(16) == 121
    assert [int(v) for v in lm.frame_to_latent(np.array([0, 1, 8, 9, 16, 17, 120]))] == [0, 1, 1, 2, 2, 3, 15]
    assert lm.group_slice(0) == (0, 1) and lm.group_slice(1) == (1, 9) and lm.group_slice(15) == (113, 121)


def test_grid_shape_576x1024():
    fp = lm.make_footprint(blob(), T, HL, WL, dilate_cells=0)
    assert fp.shape == (16, 32, 18)           # 121 frames -> 16 latent frames; 1024/32 x 576/32 cells
    assert fp.dtype == np.float32
    nm = lm.noise_mask_from_footprint(fp, 1.0)
    assert nm.shape == (1, 1, 16, 32, 18)


def test_block_aligned_cell():
    # pixels y 320..352, x 320..352 = exactly cell (row 10, col 10)
    fp = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=0)
    assert fp[5].sum() == 1 and fp[5, 10, 10] == 1
    # one pixel into the neighbour cell lights it up too (any-pixel rule) ...
    fp2 = lm.make_footprint(blob(y=(320, 353), x=(320, 352)), T, HL, WL, dilate_cells=0)
    assert fp2[5].sum() == 2 and fp2[5, 11, 10] == 1
    # ... unless cell_threshold asks for real coverage
    fp3 = lm.make_footprint(blob(y=(320, 353), x=(320, 352)), T, HL, WL, dilate_cells=0, cell_threshold=0.5)
    assert fp3[5].sum() == 1


def test_dilation():
    base = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=0)
    d1 = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=1)
    d2 = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=2)
    assert (base[3].sum(), d1[3].sum(), d2[3].sum()) == (1, 9, 25)
    assert d1[3, 9:12, 9:12].all() and not d1[3, 8, 10]
    # dilation clips at the grid border instead of wrapping
    corner = lm.make_footprint(blob(y=(0, 32), x=(0, 32)), T, HL, WL, dilate_cells=1)
    assert corner[2].sum() == 4


def test_temporal_max_pooling():
    # subject only on pixel frame 9 -> first frame of latent group 2 (frames 9..16)
    fp = lm.make_footprint(blob(t=9), T, HL, WL, dilate_cells=0)
    assert [k for k in range(T) if fp[k].any()] == [2]
    # pixel frame 8 -> last frame of group 1 (1..8)
    fp = lm.make_footprint(blob(t=8), T, HL, WL, dilate_cells=0)
    assert [k for k in range(T) if fp[k].any()] == [1]
    # pixel frame 0 -> latent frame 0 alone
    fp = lm.make_footprint(blob(t=0), T, HL, WL, dilate_cells=0)
    assert [k for k in range(T) if fp[k].any()] == [0]
    # a blob that moves across the group unions its positions
    m = np.zeros((F, H, W), np.float32)
    m[10, 0:32, 0:32] = 1
    m[15, 0:32, 64:96] = 1
    fp = lm.make_footprint(m, T, HL, WL, dilate_cells=0)
    assert fp[2, 0].tolist()[:4] == [1, 0, 1, 0]


def test_short_matte_is_padded_with_last_frame():
    m = blob(frames=60)
    fp = lm.make_footprint(m, T, HL, WL, dilate_cells=0)
    assert fp.shape[0] == 16 and fp[15].any()


def test_keep_first_frame_and_strength():
    fp = lm.make_footprint(blob(), T, HL, WL, dilate_cells=0, keep_first_frame=True)
    assert fp[0].max() == 0 and fp[1].max() == 1
    nm = lm.noise_mask_from_footprint(fp, 0.6)
    assert nm.max() == pytest.approx(0.6)


def test_feather_keeps_inside_and_falls_off():
    fp0 = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=1)
    fp = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=1, feather=1.0)
    assert (fp[3][fp0[3] > 0] == 1).all()               # footprint cells stay at 1
    assert fp[3, 8, 10] > 0 and fp[3, 8, 10] < 1        # ring outside is soft
    assert fp[3, 5, 10] < fp[3, 8, 10] or fp[3, 5, 10] == 0
    assert fp.max() <= 1.0 and fp.min() >= 0.0


def test_combine_with_first_frame_mask():
    new = lm.noise_mask_from_footprint(lm.make_footprint(blob(), T, HL, WL, dilate_cells=0), 1.0)
    ex = np.ones((1, 1, T, 1, 1), np.float32)
    ex[:, :, 0] = 0.0                                    # LTXVImgToVideoInplace: frame 0 conditioned (mask 0)
    out = lm.combine_noise_masks(ex, new)
    assert out.shape == (1, 1, 16, 32, 18)
    assert out[0, 0, 0].max() == 0                       # kept frame stays kept
    assert out[0, 0, 4, 10, 10] == 1 and out[0, 0, 4, 0, 0] == 0
    # strength-style partial existing mask multiplies
    ex2 = np.full((1, 1, T, 1, 1), 0.5, np.float32)
    assert lm.combine_noise_masks(ex2, new).max() == pytest.approx(0.5)
    assert lm.combine_noise_masks(None, new) is not None
    with pytest.raises(ValueError):
        lm.combine_noise_masks(np.ones((1, 1, 9, 1, 1), np.float32), new)


def test_region_pixels():
    fp = lm.make_footprint(blob(y=(320, 352), x=(320, 352)), T, HL, WL, dilate_cells=0)
    r = lm.region_pixels(fp, F, H, W, 0.0)
    assert r.shape == (121, 1024, 576)
    assert r[50, 335, 335] == 1 and r[50, 100, 100] == 0
    assert set(np.unique(r)) <= {0.0, 1.0}
    rf = lm.region_pixels(fp, F, H, W, 6.0)
    assert 0 < rf[50, 318, 335] < 1 and rf[50, 335, 335] == 1
    fp[0] = 0
    assert lm.region_pixels(fp, F, H, W, 0.0)[0].max() == 0     # pixel frame 0 follows latent frame 0 only


def test_main_subject_only_drops_specks():
    m = blob(y=(300, 460), x=(300, 400))                       # subject
    m[:, 900:930, 40:80] = 1.0                                 # foam speck far away (1200 px vs 16000 px)
    keep_all = lm.make_footprint(m, T, HL, WL, dilate_cells=0)
    main = lm.make_footprint(m, T, HL, WL, dilate_cells=0, main_subject_only=True)
    assert keep_all[5, 28, 1] == 1 and main[5, 28, 1] == 0     # speck cell only without the filter
    assert main[5, 10, 10] == 1                                # subject stays
    # a second blob of comparable size is kept
    m2 = blob(y=(300, 460), x=(300, 400))
    m2[:, 600:760, 100:200] = 1.0
    assert lm.make_footprint(m2, T, HL, WL, dilate_cells=0, main_subject_only=True)[5, 20, 4] == 1


def test_extra_matte_is_unioned():
    base = blob(y=(320, 352), x=(320, 352))
    extra = blob(y=(640, 672), x=(64, 96))
    fp = lm.make_footprint(base, T, HL, WL, dilate_cells=0, main_subject_only=True, extra=extra)
    assert fp[4, 10, 10] == 1 and fp[4, 20, 2] == 1 and fp[4].sum() == 2     # extra is not blob-filtered away


def test_two_stage_grid():
    fp = lm.make_footprint(blob(), T, 64, 36, dilate_cells=0)   # 1152x2048 final -> 36x64 latent at stage 2
    assert fp.shape == (16, 64, 36)


def test_node_wrapper():
    torch = pytest.importorskip("torch")
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "..", "..", "ComfyUI")))
        sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "..")))
        import comfy_api.latest  # noqa: F401
        from nodes_fusion import latent_mask as node
    except Exception as e:   # no ComfyUI checkout / missing deps
        pytest.skip(f"comfy_api / nodes_fusion not importable: {type(e).__name__}: {e}")
    x = torch.zeros(1, 128, 16, 32, 18)
    ex = torch.ones(1, 1, 16, 1, 1)
    ex[:, :, 0] = 0
    out = node.MographVideoLatentMask.execute({"samples": x, "noise_mask": ex}, mask=torch.from_numpy(blob()),
                                              dilate_cells=1, feather=0.0, strength=1.0, keep_first_frame=False)
    lat, region = out.args
    assert lat["noise_mask"].shape == (1, 1, 16, 32, 18) and lat["noise_mask"].dtype == torch.float32
    assert lat["noise_mask"][0, 0, 0].max() == 0 and lat["noise_mask"][0, 0, 5].max() == 1
    assert region.shape == (121, 1024, 576)
    assert lat["samples"].shape == x.shape
