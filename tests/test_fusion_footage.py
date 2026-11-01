"""WP-C pieces that need no running server: look families, the stand-in removal, type-pass slicing, graph builder.
Run: python -m pytest tests/test_fusion_footage.py -q   (footage helpers skip when comfy_api is not importable)"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _routes():
    spec = importlib.util.spec_from_file_location("routes_under_test", os.path.join(ROOT, "nodes_fusion", "routes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_look_families_match_the_module_registry():
    from mograph import modules as m
    # most specific family first: mech modules also inherit StrandWord
    order = [("MechWord", "mech"), ("StackWord", "stack"), ("SlabWord", "slab"), ("RingStream", "ring"),
             ("StrandWord", "strand")]
    want: dict[str, set] = {}
    for name in m.list_modules():
        cls = m.get_module(name)
        cls = cls if isinstance(cls, type) else type(cls)
        bases = {b.__name__ for b in cls.__mro__}
        if "WordModule" not in bases:
            continue                                            # transitions are not looks
        fam = next((f for b, f in order if b in bases), "voxel" if name.startswith("voxel") else name)
        want.setdefault(fam, set()).add(name)
    got = {k: set(v) for k, v in _routes().LOOK_FAMILIES.items()}
    assert got == want


def test_looks_payload_starts_with_auto():
    p = _routes().looks_payload()
    assert p["looks"][0]["value"] == "auto"
    assert {l["value"] for l in p["looks"][1:]} == set(p["families"])


@pytest.fixture(scope="module")
def footage():
    for cand in (os.environ.get("MOGRAPH_COMFYUI_PATH"), os.path.abspath(os.path.join(ROOT, "..", "..")),
                 os.path.abspath(os.path.join(ROOT, "..", "ComfyUI"))):
        if cand and os.path.isfile(os.path.join(cand, "comfy_api", "latest", "__init__.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import comfy_api.latest  # noqa: F401
        import folder_paths  # noqa: F401
    except Exception as e:
        pytest.skip(f"comfy_api not importable: {e}")
    import nodes_fusion.footage as f
    return f


def test_the_stand_in_footage_is_gone(footage):
    """no cached clip, no Ken Burns move, no luma matte / depth: the footage is LTX clips or (quality `still`) the shots' own stills."""
    for name in ("ken_burns", "luma_matte", "luma_depth", "cache_path", "cache_fits", "decode_video", "_is_standin", "_is_draft", "CACHE_DIR", "PLATE_DIM",
                 "MographDraftMatte", "MographDraftDepth"):
        assert not hasattr(footage, name), name
    assert [n.__name__ for n in footage.NODES] == ["MographShotSlot", "MographDecodeClip", "MographFootageGather", "MographFootageFlow"]


def test_read_pass_frames_clamps_to_last_frame(footage, tmp_path):
    import cv2
    d = tmp_path / "depth"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"depth_{i:05d}.png"), np.full((20, 10), 40 * (i + 1), np.uint8))
    a = footage.read_pass_frames(str(tmp_path), "depth", 1, 4, (5, 10))
    assert a.shape == (4, 10, 5, 3)
    assert np.allclose(a[:, 0, 0, 0], [80 / 255, 120 / 255, 120 / 255, 120 / 255], atol=1e-3)


def test_build_graphs_check_passes():
    from tools.app import build_graphs as bg
    for draft in (False,):
        g, side = bg.make(draft)
        problems, _ = bg.check(g, side)
        assert problems == []
