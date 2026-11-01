"""Tests for mograph.llm_plan (tolerant extraction + repair of LLM planner output)."""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mograph import plan as P  # noqa: E402
from mograph.llm_plan import PlanExtractError, extract_plan, loads_tolerant, repair_plan  # noqa: E402

NO_IMAGES = {"text": "x", "text_lines": ["x"], "images": [], "audio": None}


def good_plan() -> dict:
    beats = [i * 0.5 for i in range(21)]
    return {
        "version": 1, "seed": 3, "duration": 10.0, "fps": 30,
        "style": {"palette": ["#101418", "#f4f1ea", "#ff5a36"], "background": "palette:0",
                  "fonts": {"display": "Big Shoulders Display", "body": "Space Grotesk"}, "motion": "snappy", "energy": 0.7},
        "audio": {"source": "synth", "bpm": 120, "key": "A", "scale": "minor", "mood": "driving",
                  "beats": beats, "downbeats": beats[::4], "sections": [{"t": 0.0, "label": "intro"}],
                  "risers": [], "sfx": [{"t": 4.0, "type": "impact", "gain": 0.8}]},
        "voice": None,
        "scenes": [
            {"id": "s0", "t0": 0.0, "t1": 4.0, "background": {"type": "color", "color": "palette:0"},
             "layers": [{"type": "text", "text": "Hello", "role": "headline", "font": "display", "case": "upper",
                         "color": "palette:1", "highlight": None, "align": "center",
                         "box": {"9:16": [0.1, 0.4, 0.8, 0.2], "16:9": [0.1, 0.4, 0.8, 0.2]},
                         "t0": 0.5, "t1": 4.0, "in": {"verb": "fade", "dur": 0.3},
                         "out": {"verb": "fade", "dur": 0.2}, "emphasis": []}],
             "transition_out": {"type": "crossfade", "dur": 0.5}},
            {"id": "s1", "t0": 4.0, "t1": 10.0,
             "background": {"type": "gradient", "colors": ["palette:0", "palette:2"], "angle": 90},
             "layers": [], "transition_out": {"type": "cut", "dur": 0}}],
    }


def test_good_plan_is_valid_and_untouched():
    p = good_plan()
    assert P.validate_plan(p, NO_IMAGES) == []
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    assert out == p
    assert log == []


def test_fences_prose_and_think_are_stripped():
    text = "<think>plan it</think>Sure! Here is the plan:\n```json\n" + json.dumps(good_plan()) + "\n```\nEnjoy."
    log = []
    out = extract_plan(text, NO_IMAGES, log)
    assert P.validate_plan(out, NO_IMAGES) == []
    assert any("fence" in s for s in log) and any("think" in s for s in log)


def test_trailing_commas_comments_ellipsis_python_literals():
    s = json.dumps(good_plan(), indent=1)
    s = s.replace('"risers": [],', '"risers": [], // none')
    s = s.replace("null", "None").replace('"body": "Space Grotesk"', '"body": "Space Grotesk",')
    s = s.replace('"beats": [', '"beats": [0.0, 0.5, ...,', 1)
    log = []
    obj = loads_tolerant(s, log)
    assert obj["fps"] == 30 and obj["voice"] is None
    assert any(x.startswith("json:") for x in log)


def test_truncated_output_is_closed():
    s = json.dumps(good_plan())
    cut = s[: s.index('"transition_out": {"type": "cut"')]   # model hit max_tokens mid-scene
    log = []
    obj = loads_tolerant(cut, log)
    assert isinstance(obj, dict) and obj["scenes"]
    assert any("truncated" in x for x in log)
    out = extract_plan(cut, NO_IMAGES, [])
    assert P.validate_plan(out, NO_IMAGES) == []


def test_bad_enums_colours_fonts_are_mapped():
    p = good_plan()
    p["style"]["fonts"] = {"display": "Playfair", "body": "Helvetica Neue"}
    p["style"]["palette"] = ["#111", "#f4f1ea", "ff5a36", "#3a7bff", "#00ff00", "#ffffff", "#000000"]
    p["style"]["motion"] = "slow"
    p["audio"]["mood"] = "nostalgic"
    p["audio"]["key"] = "Bbm"
    p["audio"]["sfx"].append({"t": 5.0, "type": "sonar_ping"})
    p["scenes"][0]["transition_out"]["type"] = "dissolve"
    L = p["scenes"][0]["layers"][0]
    L["in"]["verb"] = "typewriter"
    L["color"] = "palette:9"
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    assert P.validate_plan(out, NO_IMAGES) == []
    assert out["style"]["fonts"] == {"display": "Fascinate", "body": "Anybody"}        # aliases map to liked faces (2026-10-30: favourite fonts only)
    assert len(out["style"]["palette"]) == 6 and out["style"]["palette"][0] == "#111111"
    assert out["style"]["motion"] == "smooth" and out["audio"]["mood"] == "calm"
    assert out["audio"]["key"] == "A#"
    assert out["scenes"][0]["transition_out"]["type"] == "crossfade"
    assert out["scenes"][0]["layers"][0]["in"]["verb"] == "type_on"
    assert out["audio"]["sfx"][-1]["type"] == "click"


def test_scene_timing_made_contiguous_and_layers_follow():
    p = good_plan()
    p["scenes"][0]["t1"] = 3.5           # gap to 4.0
    p["scenes"][1]["t1"] = 12.0          # overruns duration
    p["scenes"][0]["transition_out"]["dur"] = 8.0
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    sc = out["scenes"]
    assert sc[0]["t0"] == 0 and sc[-1]["t1"] == out["duration"]
    assert sc[0]["t1"] == sc[1]["t0"]
    assert any(s.startswith("scenes:") for s in log) and any(s.startswith("transition:") for s in log)


def test_missing_images_become_gradients_and_rings():
    p = good_plan()
    p["scenes"][0]["background"] = {"type": "image", "src": "img0", "focus": [0.5, 0.5],
                                    "treatment": {"duotone": None, "darken": 0.3}}
    p["scenes"][0]["layers"].append({"type": "shape", "source": "contour", "ref": "img0:c0", "stroke": "palette:2",
                                     "t0": 0.0, "t1": 4.0, "in": {"verb": "stroke_draw", "dur": 1.0},
                                     "out": {"verb": "fade", "dur": 0.2}})
    p["scenes"][0]["layers"].append({"type": "image", "src": "img1", "box": [0.1, 0.1, 0.5, 0.5],
                                     "t0": 0.0, "t1": 4.0})
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    assert out["scenes"][0]["background"]["type"] == "gradient"
    kinds = [L["type"] for L in out["scenes"][0]["layers"]]
    assert kinds == ["text", "shape"]
    assert out["scenes"][0]["layers"][1]["primitive"] == "ring"


def test_beats_regenerated_and_audio_defaults():
    p = good_plan()
    p["audio"] = {"bpm": "90 bpm", "mood": "chill", "beats": [0, 0.5, "..."]}
    out = extract_plan(json.dumps(p), NO_IMAGES, [])
    a = out["audio"]
    assert a["bpm"] == 90 and a["source"] == "synth" and a["key"] in P.KEYS
    assert a["beats"][1] == pytest.approx(60 / 90, abs=1e-3)
    assert a["downbeats"] == a["beats"][::4]


def test_boxes_single_list_pixels_and_missing_format():
    p = good_plan()
    L = p["scenes"][0]["layers"][0]
    L["box"] = [108, 768, 864, 384]                       # pixels in a 1080x1920 frame
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    b = out["scenes"][0]["layers"][0]["box"]
    assert set(b) == set(P.FORMATS)
    assert all(0 <= v <= 1 for v in b["9:16"])


def test_treatments_cleaned():
    p = good_plan()
    p["style"]["treatment"] = [{"type": "film", "preset": "Warm", "halation": 5}, {"type": "sparkles"},
                               "grain"]
    out = extract_plan(json.dumps(p), NO_IMAGES, [])
    tr = out["style"]["treatment"]
    assert [t["type"] for t in tr] == ["film_grade", "grain"]
    assert tr[0]["halation"] == 2.0 and tr[0]["preset"] == "warm"


def test_unrecoverable_raises():
    with pytest.raises(PlanExtractError):
        extract_plan("I cannot help with that.", NO_IMAGES)


def test_repair_does_not_mutate_input():
    p = good_plan()
    p["style"]["motion"] = "slow"
    before = copy.deepcopy(p)
    repair_plan(p, NO_IMAGES, [])
    assert p == before


def test_per_format_text_dict_is_kept():
    # a valid per-format dict passes untouched
    p = good_plan()
    p["scenes"][0]["layers"][0]["text"] = {"9:16": "NEON\nHEART", "16:9": "NEON HEART"}
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    assert out["scenes"][0]["layers"][0]["text"] == {"9:16": "NEON\nHEART", "16:9": "NEON HEART"}
    assert log == []
    # repair path (another fix forces repair): the dict survives, odd keys and a missing format are fixed
    p["style"]["motion"] = "slow"
    p["scenes"][0]["layers"][0]["text"] = {"9x16": "NEON\nHEART", "extra": "x"}
    log = []
    out = extract_plan(json.dumps(p), NO_IMAGES, log)
    assert out["scenes"][0]["layers"][0]["text"] == {"9:16": "NEON\nHEART", "16:9": "NEON\nHEART"}
    assert P.validate_plan(out, NO_IMAGES) == []
    assert any("text[16:9] missing" in s for s in log)
    # an empty dict is still dropped
    p["scenes"][0]["layers"][0]["text"] = {"9:16": "", "16:9": " "}
    out = extract_plan(json.dumps(p), NO_IMAGES, [])
    assert not [L for L in out["scenes"][0]["layers"] if L["type"] == "text"]
