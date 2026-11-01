"""Final route = whole-video Real-ESRGAN upscale to 1080x1920 (tools/app/final_upscale.py). Pure graph / size tests, no ComfyUI, no GPU.
Run: python -m pytest tests/test_final_upscale.py -q"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.app import final_upscale as FU  # noqa: E402


def base_graph():
    return {"assemble": {"class_type": "MographAssemble", "inputs": {"crf": 20}},
            "out_final": {"class_type": "SaveVideo", "inputs": {"video": ["assemble", 0], "filename_prefix": "p/x_video", "format": "auto", "codec": "auto"}}}


def test_final_size_per_format():
    assert FU.final_size("9:16") == (1080, 1920)
    assert FU.final_size("16:9") == (1920, 1080)


@pytest.mark.parametrize("quality,fmt,src,model", [
    ("draft", "9:16", (540, 960), FU.MODEL_X2),
    ("preview", "9:16", (360, 640), FU.MODEL_X4),
    ("draft", "16:9", (960, 540), FU.MODEL_X2),
    ("preview", "16:9", (640, 360), FU.MODEL_X4),
])
def test_source_size_and_model_per_tier(quality, fmt, src, model):
    assert FU.source_size(quality, fmt) == src
    assert FU.pick_model(src, FU.final_size(fmt)) == model


def test_final_tier_is_left_alone():
    assert FU.pick_model((1080, 1920), (1080, 1920)) is None
    g = base_graph()
    before = copy.deepcopy(g)
    assert FU.add_final_upscale(g, quality="final") == before


def test_fragment_shape_and_wiring():
    f = FU.fragment(["assemble", 0], (540, 960), (1080, 1920))
    assert [f[k]["class_type"] for k in ("up_comp", "up_model", "up_run", "up_fit", "up_vid")] == \
        ["GetVideoComponents", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "CreateVideo"]
    assert f["up_comp"]["inputs"]["video"] == ["assemble", 0]
    assert f["up_model"]["inputs"]["model_name"] == "RealESRGAN_x2plus.pth"
    assert f["up_run"]["inputs"] == {"upscale_model": ["up_model", 0], "image": ["up_comp", 0]}
    fit = f["up_fit"]["inputs"]
    assert (fit["width"], fit["height"], fit["upscale_method"], fit["crop"]) == (1080, 1920, "lanczos", "disabled")
    assert f["up_vid"]["inputs"]["fps"] == ["up_comp", 2] and f["up_vid"]["inputs"]["audio"] == ["up_comp", 1]     # GetVideoComponents: images, audio, fps


def test_fragment_refuses_a_source_that_already_reaches_the_target():
    with pytest.raises(ValueError):
        FU.fragment(["a", 0], (1080, 1920), (1080, 1920))


def test_add_final_upscale_repoints_the_save_node():
    g = FU.add_final_upscale(base_graph(), quality="preview")
    assert g["out_final"]["inputs"]["video"] == ["up_vid", 0]
    assert g["up_comp"]["inputs"]["video"] == ["assemble", 0]
    assert g["up_model"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    assert g["out_final"]["inputs"]["filename_prefix"] == "p/x_video"          # name, format, codec of the save node are untouched
    assert "out_final_draft" not in g


def test_keep_draft_adds_a_second_save():
    g = FU.add_final_upscale(base_graph(), quality="draft", keep_draft=True)
    assert g["out_final_draft"]["inputs"]["video"] == ["assemble", 0]
    assert g["out_final_draft"]["inputs"]["filename_prefix"] == "p/x_video_draft"
    assert g["out_final"]["inputs"]["video"] == ["up_vid", 0]


def test_missing_nodes_raise():
    with pytest.raises(KeyError):
        FU.add_final_upscale({"out_final": {"inputs": {"video": ["x", 0]}}}, quality="draft")


def test_standalone_graph_loads_an_output_file():
    g = FU.standalone_graph("projects/a/b.mp4 [output]", (360, 640), prefix="projects/a/b_up1080")
    assert g["load"]["inputs"]["file"] == "projects/a/b.mp4 [output]"
    assert g["up_model"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    assert g["out_final"]["class_type"] == "SaveVideo" and g["out_final"]["inputs"]["video"] == ["up_vid", 0]
    ids = set(g)
    for n in g.values():                                                        # every link points at a node of the graph
        for v in n["inputs"].values():
            if isinstance(v, list):
                assert v[0] in ids
