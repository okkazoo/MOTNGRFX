"""Strict liked-font mode for every test: a non-liked face name in code RAISES mograph.fonts_liked.FontNotLiked (production substitutes the closest liked face of the same role instead).
Tests of the substitution itself pass strict=False explicitly."""
import os

os.environ["MOGRAPH_FONT_STRICT"] = "1"

import sys

import pytest

_CAM3D_DEFAULTS = (960, 540, 0.8, 1.0)         # W, H, FIT_HK, FIT_FILLK as cam3d.py defines them


@pytest.fixture(autouse=True)
def _restore_cam3d_world():
    """cam3d keeps the world plane in module globals (W, H, CEN, FIT_HK, FIT_FILLK); spec.build_scene / sequence call set_world() for a portrait
    piece and nothing puts it back, so a later test saw 540x960 instead of the 960x540 default. Save before each test, restore after."""
    cam3d = sys.modules.get("mograph.render.cam3d")
    if cam3d is not None:
        saved = (cam3d.W, cam3d.H, cam3d.FIT_HK, cam3d.FIT_FILLK)
    else:
        saved = _CAM3D_DEFAULTS
    yield
    cam3d = sys.modules.get("mograph.render.cam3d")
    if cam3d is not None:
        cam3d.W, cam3d.H, cam3d.FIT_HK, cam3d.FIT_FILLK = saved
        cam3d.CEN[0], cam3d.CEN[1] = saved[0] / 2.0, saved[1] / 2.0


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test (a GPU box runs it; some also skip without CUDA)")
