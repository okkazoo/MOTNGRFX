"""footage_flow + compose.flow_wake: synthetic checks (no assets needed).

    python -m pytest tests/test_footage_flow.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from tools.fusion import compose, footage_flow as ff, recipe_eval as RE  # noqa: E402

torch = compose.torch
H, W = 240, 136                      # portrait, flow grid 120 x 68; sim grid = 270/136 -> scale 1.0 (136 wide)
BACKENDS = ["numpy"] + (["torch"] if torch is not None else [])


@pytest.fixture(autouse=True)
def _restore_backend():
    prev = compose.B
    yield
    compose.B = prev


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def scene(n=8, flow_px=(6.0, 2.0)):
    """type alpha (letter-ish bars, fixed), subject matte (blob), flow field (uniform inside the blob, small elsewhere)."""
    ta = np.zeros((n, H, W), np.float32)
    for t in range(n):
        ta[t, 90:150, 20:50] = 1
        ta[t, 90:110, 20:110] = 1
        ta[t, 130:200, 80:100] = 1
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (60, 130), 40, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    fl = np.zeros((H // 2, W // 2, 2), np.float32)
    fl[..., 0], fl[..., 1] = 0.3, 0.0
    mm = cv2.resize(m, (W // 2, H // 2)) > 0.5
    fl[mm] = flow_px
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1.0, 0.5, 0.2
    return ta, m, fl, rgb


def run_wake(be, n=8, **kw):
    compose.set_backend(be)
    ta, m, fl, rgb = scene(n)
    st = compose.FlowWake()
    return [st.step(ta[i], fl, m, **kw) for i in range(n)], ta


# ---------------------------------------------------------------------------------------------------------------- flow_for
def test_flow_for_shape_dtype_and_direction():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((260, 160, 3)).astype(np.float32), (0, 0), 2.0)
    base = (base - base.min()) / (base.max() - base.min())
    frames = np.stack([np.roll(base, 4 * t, axis=1)[10:250, 10:150] for t in range(5)])   # +4 px/frame to the right (full res)
    u8 = (frames * 255).astype(np.uint8)
    fl = ff.flow_for(u8, device="cpu", verbose=False)
    assert fl.shape == (5, 120, 70, 2) and fl.dtype == np.float16
    med = np.median(fl[:3].astype(np.float32).reshape(-1, 2), axis=0)
    assert 1.0 < med[0] < 3.2 and abs(med[1]) < 0.7, med           # +4 px full res = +2 px at half res
    assert np.array_equal(fl[-1], fl[-2])                           # last frame repeats


def test_wheel_preview_shape():
    fl = np.zeros((3, 20, 30, 2), np.float16)
    fl[..., 0] = 2
    w = ff.wheel(fl)
    assert w.shape == (3, 20, 30, 3) and w.dtype == np.float32


# ---------------------------------------------------------------------------------------------------------------- flow_wake
@pytest.mark.parametrize("be", BACKENDS)
def test_shape_dtype_and_effect(be):
    outs, ta = run_wake(be)
    for o in outs:
        assert o.shape == (H, W) and o.dtype == np.float32
        assert np.isfinite(o).all() and o.min() >= 0 and o.max() <= 1.0 + 1e-5
    assert np.abs(outs[-1] - ta[-1]).max() > 0.2                    # the runner pushed the letters


@pytest.mark.parametrize("be", BACKENDS)
def test_rgba_variant(be):
    outs, _ = run_wake(be, type_rgb=scene()[3])
    o = outs[-1]
    assert o.shape == (H, W, 4) and o.dtype == np.float32
    a = compose.rgba_part(o, "a")
    c = compose.rgba_part(o, "rgb")
    assert a.shape == (H, W) and c.shape == (H, W, 3)
    solid = a > 0.9
    assert solid.any() and np.allclose(c[solid], [1.0, 0.5, 0.2], atol=0.05)


@pytest.mark.parametrize("be", BACKENDS)
def test_zero_flow_is_identity(be):
    compose.set_backend(be)
    ta, m, fl, rgb = scene()
    st = compose.FlowWake()
    z = np.zeros_like(fl)
    for i in range(6):
        out = st.step(ta[i], z, m)
        assert np.array_equal(out, ta[i])
    st = compose.FlowWake()
    out = st.step(ta[0], z, m, type_rgb=rgb)
    assert np.array_equal(out[..., 3], ta[0]) and np.array_equal(out[..., :3], rgb)


@pytest.mark.parametrize("be", BACKENDS)
def test_gate_zero_is_identity(be):
    outs, ta = run_wake(be, gate=0.0)
    for o, t in zip(outs, ta):
        assert np.array_equal(o, t)


def test_displacement_follows_flow_and_settles():
    compose.set_backend("numpy")
    ta, m, fl, _ = scene(40)
    st = compose.FlowWake()
    for i in range(8):
        o = st.step(ta[i], fl, m, trail=0)
    # letters moved right (flow +x): the mass centre of the displaced alpha is to the right of the original
    ys, xs = np.mgrid[0:H, 0:W]
    cx = lambda a: float((a * xs).sum() / max(a.sum(), 1e-6))     # noqa: E731
    assert cx(o) > cx(ta[0]) + 0.5
    z = np.zeros_like(fl)
    for i in range(8, 40):                                                   # motion stops: decays back
        o = st.step(ta[i], z, m, trail=0)
    assert np.abs(o - ta[0]).max() < 0.3 and np.abs(o - ta[0]).mean() < 0.003      # sub-pixel residual only


def test_torch_matches_numpy_psnr():
    if torch is None:
        pytest.skip("no torch")
    a, _ = run_wake("numpy", n=10)
    b, _ = run_wake("torch", n=10)
    p = min(psnr(x, y) for x, y in zip(a, b))
    assert p > 35.0, p


def test_torch_matches_numpy_psnr_rgba():
    if torch is None:
        pytest.skip("no torch")
    rgb = scene()[3]
    a, _ = run_wake("numpy", n=6, type_rgb=rgb)
    b, _ = run_wake("torch", n=6, type_rgb=rgb)
    assert min(psnr(x, y) for x, y in zip(a, b)) > 35.0


def test_registry_and_implicit_state():
    reg = compose._ops_registry()
    assert "flow_wake" in reg and "rgba_part" in reg and hasattr(reg["flow_wake"], "raw")
    compose.set_backend("numpy")
    ta, m, fl, _ = scene(6)
    compose._WAKE.clear()
    kw = dict(strength=2.0, state_key="t")
    outs = [reg["flow_wake"](ta[i], fl, m, frame_idx=10 + i, u=(i + 1) / 10, **kw) for i in range(5)]
    assert np.abs(outs[-1] - ta[-1]).mean() > np.abs(outs[0] - ta[0]).mean()              # state accumulates over consecutive frame_idx
    again = reg["flow_wake"](ta[0], fl, m, frame_idx=15, u=0.0, **kw)                    # u = 0 -> new shot -> cold start
    assert np.allclose(again, outs[0], atol=1e-5)
    jump = reg["flow_wake"](ta[0], fl, m, frame_idx=99, u=0.5, **kw)                     # discontinuity -> cold start
    assert np.allclose(jump, outs[0], atol=1e-5)


# ---------------------------------------------------------------------------------------------------------------- recipe plumbing
def test_p06_v003_recipe_resolves_flow_input():
    pj = ROOT / "15_fusion" / "pieces" / "P06_v003.json"
    if not pj.exists():
        pytest.skip("P06_v003.json missing")
    piece = json.loads(pj.read_text(encoding="utf-8"))
    assert RE.shot_needs_flow(piece, "s1a")
    rec = RE.build_recipe(piece, "s1a", type_dir=ROOT / "nonexistent")
    assert rec["inputs"]["flow"] == dict(src="flow", kind="flow", native=True)
    assert [o["op"] for o in rec["ops"]][:2] == ["upscale_matte", "flow_wake"]
    # the displaced alpha ($wa) replaced $type_matte everywhere after the op
    tail = json.dumps(rec["ops"][2:])
    assert '"$type_matte"' not in tail and '"$wa"' in tail
    # everything else of the piece is untouched vs v002
    v2 = json.loads((ROOT / "15_fusion" / "pieces" / "P06.json").read_text(encoding="utf-8"))
    for k in ("stills", "audio", "type", "signoff", "id", "slug", "fps"):
        assert piece[k] == v2[k]
    assert [{k: v for k, v in s.items() if k not in ("flow", "masktrack")} for s in piece["shots"]] == v2["shots"]
