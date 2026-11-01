"""The app's Final = the 540x960 render AND the Real-ESRGAN 1080x1920 upscale (tools/app/final_upscale.py); the upscale hangs on a lazy save node so a Preview never runs it."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.app import final_upscale as FU  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402


def test_wants_upscale_only_for_a_vertical_final():
    wants_upscale = FU.wants_upscale
    assert wants_upscale({"quality": SP.APP_FINAL_TIER, "format": "9:16"})
    assert not wants_upscale({"quality": SP.APP_FINAL_TIER, "format": "9:16", SP.FLAG: True})          # Preview (still)
    assert not wants_upscale({"quality": "preview", "format": "9:16"})                                  # internal 360x640 tier: upscaled by hand, not by the graph
    assert not wants_upscale({"quality": SP.APP_FINAL_TIER, "format": "16:9"})                          # the chain's lanczos target is 1080x1920
    assert not wants_upscale(None)


def test_app_graphs_carry_the_upscale_chain_and_both_outputs():
    g = json.load(open(os.path.join(ROOT, "app", "graphs", "fusion_design_qwen_auto_layers_pivot_gemma31b_qat.json"), encoding="utf-8"))
    assert g["out_final"]["inputs"]["video"] == ["assemble", 0]                                         # the 540x960 render keeps its own save
    assert g["up_run"]["class_type"] == "ImageUpscaleWithModel" and g["up_model"]["inputs"]["model_name"] == FU.MODEL_X2
    assert g["up_fit"]["inputs"]["width"] == 1080 and g["up_fit"]["inputs"]["height"] == 1920
    sv = g["out_final_1080"]
    assert sv["class_type"] == "MographSaveFinalUpscaled" and sv["inputs"]["video"] == ["up_vid", 0]
