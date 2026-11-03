"""v20 PREVIEW + FINAL: the `_final_upscale` side graph (tools/app/final_upscale_graph.py) and the Preview quality that skips the upscale. Pure tests, no ComfyUI, no GPU.
Run: python -m pytest tests/test_final_upscale_graph.py -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.app import final_upscale as FU  # noqa: E402
from tools.app import final_upscale_graph as FG  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402

CORE = {"LoadVideo", "GetVideoComponents", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "CreateVideo", "SaveVideo"}


def by_title(g, title):
    return next(n for n in g.values() if n["_meta"]["title"] == title)


def test_graph_nodes_are_core_only_and_wired_load_to_save():
    g = FG.build("motngrfx_app/video/x_00001_.mp4 [output]", (540, 960))
    assert {n["class_type"] for n in g.values()} == CORE and len(g) == 7
    load = by_title(g, FG.IN_TITLE)
    assert load["inputs"]["file"] == "motngrfx_app/video/x_00001_.mp4 [output]"
    save = by_title(g, FG.OUT_TITLE)
    assert save["class_type"] == "SaveVideo" and save["inputs"]["video"] == ["up_vid", 0]
    assert g["up_comp"]["inputs"]["video"] == ["load", 0]
    assert g["up_run"]["inputs"]["image"] == ["up_comp", 0] and g["up_fit"]["inputs"]["image"] == ["up_run", 0]
    assert g["up_vid"]["inputs"] == {"images": ["up_fit", 0], "fps": ["up_comp", 2], "audio": ["up_comp", 1]}      # fps + audio carried through


@pytest.mark.parametrize("src,fmt,model,target", [
    ((540, 960), "9:16", FU.MODEL_X2, (1080, 1920)),
    ((360, 640), "9:16", FU.MODEL_X4, (1080, 1920)),
    ((960, 540), "16:9", FU.MODEL_X2, (1920, 1080)),
    ((640, 360), "16:9", FU.MODEL_X4, (1920, 1080)),
])
def test_model_and_target_by_source_size(src, fmt, model, target):
    p = FG.plan(src, fmt)
    assert p["model"] == model and tuple(p["target"]) == target
    g = FG.build("a.mp4 [output]", src, fmt)
    assert g["up_model"]["inputs"]["model_name"] == model
    assert (g["up_fit"]["inputs"]["width"], g["up_fit"]["inputs"]["height"]) == target


def test_already_1080_is_a_noop_plan():
    assert FG.plan((1080, 1920), "9:16")["model"] is None


def test_template_on_disk_is_the_540_form_and_not_a_template_dropdown_entry():
    t = json.loads((ROOT / "app" / "graphs" / "_final_upscale.json").read_text(encoding="utf-8"))
    assert t == json.loads(json.dumps(FG.build("", FG.DEFAULT_SRC, "9:16")))             # regenerate with --write-template after a change
    assert (ROOT / "app" / "graphs" / "_final_upscale.json").name.startswith("_")
    assert by_title(t, FG.IN_TITLE)["inputs"]["file"] == ""


def test_new_file_next_to_the_original_never_over_it():
    # the page names the new file <stem>_up1080 (web_app/bind.js buildFinalUpscale); the suffix is the one constant both sides share
    assert FG.SUFFIX == "_up1080"
    g = FG.build("x [output]", (540, 960), prefix="motngrfx_app/video/r1_motngrfx" + FG.SUFFIX)
    assert by_title(g, FG.OUT_TITLE)["inputs"]["filename_prefix"].endswith("_up1080")


# ---- Preview (quality draft540) = the full render at the draft size, WITHOUT the 1080 upscale
def test_preview_quality_is_the_draft_tier_without_stills_and_flags_no_upscale():
    assert SP.split_quality(SP.PREVIEW_FULL) == ("draft", False)
    assert SP.is_preview_full("draft540") and SP.is_preview_full(" Draft540 ") and not SP.is_preview_full("final") and not SP.is_preview_full("draft")
    assert SP.split_quality("final") == ("draft", False)                                    # unchanged: Final = the same tier, plus the upscale


def test_lazy_save_runs_the_upscale_for_final_but_not_for_the_v20_preview():
    final_piece = {"quality": SP.APP_FINAL_TIER, "format": "9:16", "request": {"words": "A"}}
    preview_piece = {"quality": SP.APP_FINAL_TIER, "format": "9:16", "request": {"words": "A", SP.NO_UPSCALE: True}}
    assert FU.wants_upscale(final_piece)
    assert not FU.wants_upscale(preview_piece)
    # the flag rides in piece.request, which design_step.LOCKED_KEYS keeps on the designed piece
    assert SP.NO_UPSCALE in SP.REQUEST_FIELDS
    rec = SP.request_record({"words": "A", SP.NO_UPSCALE: True, "seed": 3})
    assert rec[SP.NO_UPSCALE] is True and SP.request_record({"words": "A", SP.NO_UPSCALE: None}).get(SP.NO_UPSCALE) is None
    p = SP.prepare_piece({"quality": "draft"}, {"words": "A", SP.NO_UPSCALE: True}, False)
    assert p["request"][SP.NO_UPSCALE] is True and not FU.wants_upscale({**p, "format": "9:16"})


def test_design_step_keeps_the_request_key_that_carries_the_flag():
    from tools.fusion import design_step
    assert "request" in design_step.LOCKED_KEYS


# ---- 2026-11-03 image board edited like a video: quality draft540plates = the Preview render, plain footage (no type, no compose ops), no upscale
def test_plates_only_quality_is_the_preview_tier_and_flags_the_request():
    assert SP.split_quality(SP.PLATES_ONLY_Q) == ("draft", False)
    assert SP.is_preview_full(SP.PLATES_ONLY_Q) and SP.is_plates_only(" Draft540Plates ") and not SP.is_plates_only("draft540")
    assert SP.PLATES_ONLY in SP.REQUEST_FIELDS
    p = SP.prepare_piece({"quality": "draft"}, {"words": "A", SP.NO_UPSCALE: True, SP.PLATES_ONLY: True}, False)
    assert p["request"][SP.PLATES_ONLY] is True and not FU.wants_upscale({**p, "format": "9:16"})
    assert SP.PLATES_ONLY not in SP.request_record({"words": "A", SP.PLATES_ONLY: None})


def test_compose_returns_the_plain_footage_for_a_plates_only_piece():
    pytest.importorskip("comfy_api")
    import torch
    from nodes_fusion import compose as C
    piece = {"request": {SP.PLATES_ONLY: True}}
    assert C.plates_only(piece) and not C.plates_only({"request": {}}) and not C.plates_only(None)
    piece["layout"] = {"s1": {"start": 0, "frames": 4}, "s2": {"start": 4, "frames": 4}}
    from tools.fusion import recipe_eval as RE
    try:
        RE.piece_layout(piece)
    except Exception:   # noqa: BLE001
        pytest.skip("piece_layout needs a fully planned piece")
    a, b = torch.zeros(4, 8, 8, 3, dtype=torch.uint8), torch.ones(4, 8, 8, 3, dtype=torch.uint8)
    out, st = C.plate_clips(piece, [{"sid": "s2", "u8": b}, {"sid": "s1", "u8": a}])
    assert [c["sid"] for c in out] == ["s1", "s2"] and out[0]["u8"] is a and out[1]["start"] == 4 and st[0]["backend"] == "plates"
