"""Depth pivot-chunk transition as a reusable module + ComfyUI node (mograph/modules/transitions/depth_pivot.py, nodes_fusion/depth_transition.py).
CPU only, tiny synthetic inputs (64x112), a fake depth map so the depth model is never loaded (the one test that checks the model hook stubs it).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_depth_transition_module.py -q"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph.modules import core  # noqa: E402
from mograph.modules.transitions import depth_pivot as dp  # noqa: E402
from tools.fusion import depth_dolly as dd  # noqa: E402

W, H = 64, 112
SMALL = dict(short=64, travel=6, hold1=3, hold2=3)          # 12 frames
N_SMALL = 12
SLOW = os.environ.get("MOGRAPH_SKIP_SLOW") != "1"


def _img(seed, shift=0):
    base = np.random.default_rng(seed).integers(0, 200, (H // 8, W // 8, 3), dtype=np.uint8)
    return np.clip(np.kron(base, np.ones((8, 8, 1), np.uint8)).astype(int) + shift, 0, 255).astype(np.uint8)


def _matte():
    m = np.zeros((H, W), np.float32)
    m[40:80, 20:44] = 1.0
    return m


def _depth():
    d = np.full((H, W), 0.1, np.float32)
    d[:, :16] = 0.5
    d[40:80, 20:44] = 0.9
    return d


def _clip(seed, n=16):
    return np.stack([_img(seed, 8 * k) for k in range(n)])


def _run(a, ma, b, mb, **params):
    return dp.depth_pivot_render(a, ma, b, mb, _depth(), _depth(), dict(SMALL, **params))


# ----------------------------------------------------------------------------------------------- registry + params
def test_registered_as_a_hidden_image_transition():
    assert core.get_module("depth_pivot").kind == "transition"
    assert "depth_pivot" not in core.list_modules("transition")          # the type engine's pool / batch coverage are unchanged
    assert core.get_module("depth_pivot").PARAMS is dp.PARAMS


def test_defaults_are_the_approved_look():
    p = dp.resolve_params()
    assert (p["chunks"], p["size_lo"], p["size_hi"], p["blur"], p["subject_card"], p["video"]) == (32, 1.5, 4.0, "continuous", 1, "auto")
    assert (p["in_angle"], p["out_angle"], p["travel"], p["hold1"], p["hold2"], p["depth_ratio"], p["spring"], p["short"]) == \
        (45.0, 40.0, 36, 12, 24, 6.0, False, 432)
    assert p["locked"] == {"mode": "pivot", "splat": "billboard", "ease": "quintic"}
    e = p["engine"]
    assert (e["max_chunks"], e["bg_chunks"], e["mode"], e["splat"]) == (50, 32, "pivot", "billboard")      # v009's density: 50 ref, 32 bg cards
    assert dp.n_frames() == 72


def test_resolve_clamps_roundtrips_and_rejects_unknown_keys():
    p = dp.resolve_params({"chunks": 9999, "travel": 1, "in_angle": 500.0, "seed": 7})
    assert p["chunks"] == 400 and p["travel"] == 4 and p["in_angle"] == 80.0 and p["seed"] == 7
    assert dp.resolve_params(p) == p                                                   # a resolved dict is accepted back unchanged
    assert dp.resolve_params(json.loads(json.dumps(p))) == p                           # ... also after JSON
    with pytest.raises(KeyError):
        dp.resolve_params({"chunkz": 3})
    with pytest.raises(ValueError):
        dp.resolve_params({"video": "maybe"})
    assert dp.resolve_params({"size_lo": 4.0, "size_hi": 1.5})["size_lo"] == 1.5      # swapped, not an error


def test_chunks_scales_the_card_scale_with_it():
    a, b = dp.engine_args(dp.resolve_params({"chunks": 32})), dp.engine_args(dp.resolve_params({"chunks": 100}))
    assert (a["max_chunks"], a["bg_chunks"]) == (50, 32)
    assert (b["max_chunks"], b["bg_chunks"]) == (156, 100)
    assert (dp.resolve_params({"chunks": 10})["engine"]["max_chunks"] >= 10)


# ----------------------------------------------------------------------------------------------- render
def test_frames_have_the_right_count_size_and_dtype():
    r = _run(_img(1), _matte(), _img(2), _matte())
    assert r.frames.shape == (N_SMALL, H, W, 3) and r.frames.dtype == np.uint8
    assert r.info["n_frames"] == N_SMALL and r.info["size"] == [W, H] and r.info["video"] is False
    assert r.params["travel"] == 6 and r.params["engine"]["travel"] == 6
    assert dp.depth_pivot(_img(1), _matte(), _img(2), _matte(), _depth(), _depth(), SMALL).shape == r.frames.shape
    assert not np.array_equal(r.frames[0], r.frames[7])                               # the move happens


def test_first_frame_is_a_and_settled_last_frame_is_b_with_video_off():
    a, b = _img(1), _img(2)
    r = _run(a, _matte(), b, _matte(), video="off")
    assert dd.psnr(r.frames[0], a) > 40.0 and dd.psnr(r.frames[-1], b) > 40.0
    assert all(np.array_equal(r.frames[0], r.frames[i]) for i in range(SMALL["hold1"]))     # A holds
    assert all(np.array_equal(r.frames[-1], r.frames[-1 - i]) for i in range(SMALL["hold2"]))   # B holds
    c = _run(_clip(1), _matte(), _clip(2), _matte(), video="off")                      # a clip with video forced off: frame 0 of each
    assert dd.psnr(c.frames[0], _clip(1)[0]) > 40.0 and dd.psnr(c.frames[-1], _clip(2)[0]) > 40.0


def test_same_seed_identical_frames_and_params_remake_the_result():
    a, b = _img(1), _img(2)
    r1, r2 = _run(a, _matte(), b, _matte(), seed=5), _run(a, _matte(), b, _matte(), seed=5)
    assert np.array_equal(r1.frames, r2.frames) and r1.params == r2.params
    r3 = dp.depth_pivot_render(a, _matte(), b, _matte(), _depth(), _depth(), json.loads(json.dumps(r1.params)))     # remade from the stored dict
    assert np.array_equal(r1.frames, r3.frames) and r3.params == r1.params
    r4 = _run(a, _matte(), b, _matte(), seed=6)
    assert not np.array_equal(r1.frames, r4.frames)                                    # another seed = other card jitter / sizes


def test_live_video_is_the_default_for_a_multi_frame_batch_and_a_single_image_is_static():
    clip_a, clip_b = _clip(1), _clip(2)
    mats = np.stack([_matte()] * 16)
    v = _run(clip_a, mats, clip_b, mats)                                               # defaults: video = auto
    assert v.info["video"] is True and v.params["video"] == "auto"
    assert not np.array_equal(v.frames[0], v.frames[SMALL["hold1"] - 1])               # A's textures play during its hold (frame 0 -> 2)
    assert not np.array_equal(v.frames[-2], v.frames[-1])                              # B's clip still plays after the landing
    s = _run(_img(1), _matte(), _img(2), _matte())                                     # single images: a still
    assert s.info["video"] is False
    assert np.array_equal(s.frames[0], s.frames[SMALL["hold1"] - 1]) and np.array_equal(s.frames[-2], s.frames[-1])
    off = _run(clip_a, mats, clip_b, mats, video="off")                                # forced off: static even with a clip
    assert off.info["video"] is False and np.array_equal(off.frames[0], off.frames[SMALL["hold1"] - 1])
    on = _run(_img(1), _matte(), _img(2), _matte(), video="on")                        # forced on with stills: nothing to play, same as still
    assert on.info["video"] is True and np.array_equal(on.frames[0], on.frames[SMALL["hold1"] - 1])
    assert dd.psnr(v.frames[0], clip_a[0]) > 40.0                                      # frame 0 is clip frame 0 either way


def test_depth_model_runs_only_when_no_depth_is_given(monkeypatch):
    calls = []

    def fake(bgr):
        calls.append(bgr.shape)
        return _depth()
    monkeypatch.setattr(dd, "estimate_depth", fake)
    r = dp.depth_pivot_render(_img(1), _matte(), _img(2), _matte(), None, _depth(), dict(SMALL, travel=12, hold1=0, hold2=0))
    assert len(calls) == 1 and r.info["depth"] == ["depth-anything-v2-small", "given"]
    calls.clear()
    dp.depth_pivot_render(_img(1), _matte(), _img(2), _matte(), _depth(), _depth(), dict(SMALL, travel=12, hold1=0, hold2=0))
    assert calls == []


@pytest.mark.skipif(not SLOW, reason="MOGRAPH_SKIP_SLOW=1 (builds two 432x762 scenes, a few seconds)")
def test_chunks_is_the_density_control_32_default_and_smaller_cards_for_more():
    big_w, big_h = 432, 762
    def im(s):
        b = np.random.default_rng(s).integers(0, 200, (big_h // 18 + 1, big_w // 18 + 1, 3), dtype=np.uint8)
        return np.ascontiguousarray(np.kron(b, np.ones((18, 18, 1), np.uint8))[:big_h, :big_w])
    m = np.zeros((big_h, big_w), np.float32)
    m[300:600, 150:290] = 1.0
    d = np.full((big_h, big_w), 0.1, np.float32)
    d[:, :100] = 0.5
    d[300:600, 150:290] = 0.9
    sc, _, p, info = dp.build_scene(im(1), m, im(2), m, d, d, {})
    assert info["size"] == [big_w, big_h]
    for c in info["cards"]:
        assert (c["subject"], c["background"]) == (1, 32)                              # default: 32 background cards + the runner card
    ref0 = info["cards"][0]["ref_px"]
    assert ref0 == pytest.approx(np.sqrt(big_w * big_h / 50.0))                        # v009's card scale
    assert info["cards"][0]["size_px"][1] >= ref0 * 1.5                                # sizes are ref x U(1.5, 4) (enlarged to cover)
    sc2, _, p2, info2 = dp.build_scene(im(1), m, im(2), m, d, d, {"chunks": 100})
    for c, c0 in zip(info2["cards"], info["cards"]):
        assert c["background"] <= 100 and c["subject"] == 1
        assert c["background"] > c0["background"]                                      # more cards ...
        assert c["ref_px"] < c0["ref_px"]                                              # ... on a smaller card scale
    assert info2["cards"][0]["ref_px"] == pytest.approx(np.sqrt(big_w * big_h / 156.0))


# ----------------------------------------------------------------------------------------------- node
def _node_module():
    spec = importlib.util.spec_from_file_location("depth_transition_under_test", os.path.join(ROOT, "nodes_fusion", "depth_transition.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["depth_transition_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _torch():
    return pytest.importorskip("torch")


def test_node_tensor_conversion_both_ways():
    torch = _torch()
    n = _node_module()
    a = _clip(1, 3)
    t = torch.from_numpy(a.astype(np.float32) / 255.0)                                 # ComfyUI IMAGE [B,H,W,C] float 0..1
    assert np.array_equal(n.image_to_numpy(t), a)
    assert n.image_to_numpy(t[0]).shape == (1, H, W, 3)                                # a bare [H,W,C] is one image
    m = torch.from_numpy(np.stack([_matte()] * 3))                                     # MASK [B,H,W]
    mm = n.mask_to_numpy(m)
    assert mm.shape == (3, H, W) and mm.dtype == np.float32 and np.array_equal(mm[0], _matte())
    assert n.mask_to_numpy(m[:, None]).shape == (3, H, W)
    assert n.image_to_numpy(None) is None and n.mask_to_numpy(None) is None
    out = n.frames_to_image(a)
    assert out.dtype == torch.float32 and tuple(out.shape) == a.shape and float(out.max()) <= 1.0
    assert np.array_equal(n.image_to_numpy(out), a)                                    # exact round trip


def test_node_run_returns_frames_tensor_and_params_json_and_reproduces():
    torch = _torch()
    n = _node_module()
    ti = lambda x: torch.from_numpy(x.astype(np.float32) / 255.0)    # noqa: E731
    d = torch.from_numpy(np.repeat(_depth()[None, ..., None], 3, -1))
    mk = torch.from_numpy(_matte()[None])
    kw = dict(chunks=32, video="auto", in_angle=45.0, out_angle=40.0, travel=6, hold1=3, hold2=3, depth_ratio=6.0, blur="continuous",
              subject_card=True, short=64, seed=4, size_lo=1.5, size_hi=4.0, spring=False)
    frames, pj = n.run(ti(_img(1)[None]), mk, ti(_img(2)[None]), mk, d, d, "", **kw)
    assert tuple(frames.shape) == (N_SMALL, H, W, 3) and frames.dtype == torch.float32
    prm = json.loads(pj)
    assert prm["seed"] == 4 and prm["chunks"] == 32 and prm["module"] == "depth_pivot" and prm["engine"]["bg_chunks"] == 32
    again, pj2 = n.run(ti(_img(1)[None]), mk, ti(_img(2)[None]), mk, d, d, pj, **dict(kw, seed=99, chunks=100))   # params_json overrides the widgets
    assert torch.equal(frames, again) and json.loads(pj2) == prm
    clip = n.run(ti(_clip(1)), torch.from_numpy(np.stack([_matte()] * 16)), ti(_clip(2)), torch.from_numpy(np.stack([_matte()] * 16)), d, d, "", **kw)[0]
    assert not torch.equal(clip[0], clip[SMALL["hold1"] - 1])                           # a batch plays live through the node
    with pytest.raises(ValueError):
        n.run(None, None, ti(_img(2)[None]), None)


def test_node_schema_widgets_default_to_the_approved_look():
    n = _node_module()
    if not n.NODES:
        pytest.skip("comfy_api not importable here: the node class is only defined with ComfyUI (helpers tested above)")
    node = n.NODES[0]
    assert node.__name__ == "MOTNGRFX_DepthPivotTransition"
    schema = node.define_schema()
    assert schema.node_id == "MOTNGRFX_DepthPivotTransition"
    ins = {i.id if hasattr(i, "id") else i.name: i for i in schema.inputs}
    assert set(n.WIDGETS) <= set(ins) and {"image_a", "mask_a", "image_b", "mask_b", "depth_a", "depth_b", "params_json"} <= set(ins)
    assert ins["chunks"].default == 32 and ins["video"].default == "auto" and ins["in_angle"].default == 45.0
