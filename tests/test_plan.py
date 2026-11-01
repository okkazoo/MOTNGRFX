"""Tests for mograph.plan (schema, validation, helpers) and mograph.planner_rules.

Uses a FAKE analysis dict (SPEC Interface 1) so it does not depend on mograph.analyze.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if importlib.util.find_spec("mograph.geometry") is None:   # W1 not delivered yet: use the local stub
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _geometry_stub  # type: ignore  # noqa: E402
    sys.modules["mograph.geometry"] = _geometry_stub

from mograph import plan as P  # noqa: E402
from mograph.planner_rules import MOOD_TABLE, plan_from_analysis  # noqa: E402


# --------------------------------------------------------------------------
# fake analysis
# --------------------------------------------------------------------------

def fake_image(idx: int, w: int = 1024, h: int = 768, accent: str = "#ff5522") -> dict:
    iid = f"img{idx}"
    return {
        "id": iid, "pixels": np.zeros((8, 8, 3), np.float32), "w": w, "h": h,
        "palette": [{"hex": "#14202e", "weight": np.float32(0.4)}, {"hex": "#1e6470", "weight": 0.25},
                    {"hex": accent, "weight": 0.15}, {"hex": "#f0e8d8", "weight": 0.1},
                    {"hex": "#ffd23f", "weight": 0.06}, {"hex": "#556677", "weight": 0.04}],
        "accent": accent, "luma_mean": 0.35, "contrast": 0.5, "saturation": 0.45,
        "saliency_box": [0.4, 0.3, 0.45, 0.5], "focus": [0.62, 0.58],
        "empty_regions": [[0.0, 0.0, 1.0, 0.25], [0.0, 0.8, 0.4, 0.2]],
        "contours": [
            {"id": f"{iid}:c0", "points": [[0.45, 0.3], [0.8, 0.35], [0.8, 0.8], [0.45, 0.8]], "closed": True,
             "area": 0.2, "bbox": [0.45, 0.3, 0.35, 0.5]},
            {"id": f"{iid}:c1", "points": [[0.15, 0.65], [0.25, 0.65], [0.25, 0.8], [0.15, 0.8]], "closed": True,
             "area": 0.05, "bbox": [0.15, 0.65, 0.1, 0.15]},
            {"id": f"{iid}:c2", "points": [[0.1, 0.1], [0.12, 0.1], [0.12, 0.12]], "closed": True,
             "area": 0.001, "bbox": [0.1, 0.1, 0.02, 0.02]},
        ],
        "lines": [{"id": f"{iid}:l0", "p0": [0.1, 0.9], "p1": [0.9, 0.85], "strength": 0.8},
                  {"id": f"{iid}:l1", "p0": [0.2, 0.2], "p1": [0.3, 0.5], "strength": 0.3}],
        "meaning": None,
    }


def fake_audio(duration: float = 6.0, bpm: float = 120.0) -> dict:
    beat = 60.0 / bpm
    beats = list(np.arange(0.0, duration, beat))
    sr = 48000
    return {"waveform": np.zeros((1, int(duration * sr)), np.float32), "sample_rate": sr, "duration": duration,
            "bpm": np.float64(bpm), "beats": beats, "downbeats": beats[::4], "onsets": beats,
            "sections": [{"t": 0.0, "energy": 0.3}, {"t": 4.0, "energy": 0.9}]}


def fake_analysis(text: str = "", n_images: int = 0, audio: bool = False, duration=None,
                  energy: float = 0.7, valence: float = 0.6, keywords=None) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {
        "text": text.strip(), "text_lines": lines, "duration_hint": duration,
        "images": [fake_image(i, *( (1024, 768) if i % 2 == 0 else (768, 1024) ),
                              accent=["#ff5522", "#2e6bff", "#22cc88"][i % 3]) for i in range(n_images)],
        "audio": fake_audio() if audio else None,
        "mood": {"energy": energy, "valence": valence, "keywords": keywords or []},
    }


TEXT = "Make it move\nShapes from your pictures\nOn the beat"

CASES = {
    "text_only": dict(text=TEXT),
    "images_only": dict(n_images=2),
    "text_images": dict(text=TEXT, n_images=2),
    "audio_only": dict(audio=True),
    "text_images_audio": dict(text=TEXT, n_images=2, audio=True),
}


def all_verbs_used(plan):
    """Every in/out verb used per layer type is in the VERBS list (belt and braces)."""
    for sc in plan["scenes"]:
        for L in sc["layers"]:
            kind = {"text": "text", "shape": "shape", "image": "image"}[L["type"]]
            assert L["in"]["verb"] in P.VERBS[f"{kind}_in"]
            assert L["out"]["verb"] in P.VERBS[f"{kind}_out"]


# --------------------------------------------------------------------------
# a hand-built plan exercising every layer type
# --------------------------------------------------------------------------

def manual_plan() -> dict:
    return {
        "version": 1, "seed": 1, "duration": 4.0, "fps": 30,
        "style": {"palette": ["#111111", "#f2f2f2", "#ff4d2e", "#2e6bff"], "background": "palette:0",
                  "fonts": {"display": "Big Shoulders Display", "body": "Space Grotesk"}, "motion": "snappy", "energy": 0.7,
                  "grain": 0.1, "vignette": 0.2, "beat_flash": 0.0},
        "audio": {"source": "synth", "bpm": 120, "key": "A", "scale": "minor", "mood": "driving",
                  "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], "downbeats": [0.0, 2.0],
                  "sections": [{"t": 0.0, "label": "intro"}, {"t": 2.0, "label": "drop"}],
                  "risers": [{"t0": 1.0, "t1": 2.0}], "sfx": [{"t": 2.0, "type": "impact", "gain": 0.8}]},
        "voice": None,
        "scenes": [
            {"id": "s0", "t0": 0.0, "t1": 2.0,
             "background": {"type": "image", "src": "img0", "focus": [0.5, 0.4],
                            "motion": {"verb": "ken_burns", "zoom": [1.0, 1.1], "pan": [[0, 0], [0.02, 0]]},
                            "treatment": {"duotone": ["palette:0", "palette:2"], "blur": 0.0, "darken": 0.2}},
             "layers": [
                 {"type": "shape", "source": "contour", "ref": "img0:c0", "primitive": None, "box": None,
                  "stroke": "palette:2", "stroke_width": 0.006, "fill": None, "fill_opacity": 0.0,
                  "offset": [0, 0], "scale": 1.0, "t0": 0.0, "t1": 2.0,
                  "in": {"verb": "stroke_draw", "dur": 0.8}, "out": {"verb": "fade", "dur": 0.2},
                  "emphasis": [{"verb": "pulse", "on": "beats", "amount": 0.08}],
                  "echo": {"count": 2, "spacing": 0.015, "decay": 0.5}},
                 {"type": "shape", "source": "line", "ref": "img0:l0", "primitive": None, "box": None,
                  "stroke": "#ffffff", "stroke_width": 0.008, "fill": None, "fill_opacity": 0.0,
                  "offset": [0, 0], "scale": 1.0, "t0": 0.5, "t1": 2.0,
                  "in": {"verb": "stroke_draw", "dur": 0.4}, "out": {"verb": "cut", "dur": 0.0},
                  "emphasis": [], "echo": None},
                 {"type": "shape", "source": "primitive", "ref": None, "primitive": "ring",
                  "box": {"9:16": [0.2, 0.3, 0.6, 0.3375], "16:9": [0.35, 0.2, 0.3, 0.6]},
                  "stroke": "palette:3", "stroke_width": 0.006, "fill": None, "fill_opacity": 0.0,
                  "offset": [0, 0], "scale": 1.0, "t0": 0.0, "t1": 2.0,
                  "in": {"verb": "scale_pop", "dur": 0.3}, "out": {"verb": "fade", "dur": 0.2},
                  "emphasis": [], "echo": None},
                 {"type": "text", "text": "MAKE IT MOVE", "role": "headline", "font": "display", "case": "upper",
                  "color": "palette:1", "highlight": "palette:2", "align": "center",
                  "box": {"9:16": [0.08, 0.62, 0.84, 0.18], "16:9": [0.06, 0.55, 0.5, 0.25]},
                  "t0": 0.25, "t1": 2.15,   # extends into the transition (2.0 + 0.3/2)
                  "in": {"verb": "word_stagger", "dur": 0.5, "stagger_beats": 0.5},
                  "out": {"verb": "slide_up", "dur": 0.25},
                  "emphasis": [{"verb": "shake", "on": [0.5, 1.0], "amount": 0.02}]},
             ],
             "transition_out": {"type": "push", "dir": "left", "dur": 0.3}},
            {"id": "s1", "t0": 2.0, "t1": 4.0,
             "background": {"type": "gradient", "colors": ["palette:0", "#334455"], "angle": 45,
                            "motion": {"verb": "drift"}},
             "layers": [
                 {"type": "image", "src": "img1", "mask": "contour:img1:c0",
                  "box": {"9:16": [0.1, 0.1, 0.8, 0.45], "16:9": [0.55, 0.15, 0.4, 0.7]},
                  "t0": 2.0, "t1": 4.0, "in": {"verb": "scale_pop", "dur": 0.35},
                  "out": {"verb": "fade", "dur": 0.2}, "emphasis": [],
                  "border": {"color": "palette:1", "width": 0.004}},
             ],
             "transition_out": {"type": "cut", "dur": 0.0}},
        ],
    }


@pytest.fixture
def ana():
    return fake_analysis(TEXT, n_images=2)


@pytest.fixture
def good(ana):
    p = manual_plan()
    assert P.validate_plan(p, ana) == []
    return p


def errs_with(plan, ana, needle):
    errs = P.validate_plan(plan, ana)
    assert errs, "expected validation errors"
    assert any(needle in e for e in errs), errs
    return errs


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

def test_constants():
    assert set(P.VERBS) == {"text_in", "text_out", "emphasis", "shape_in", "shape_out", "image_in", "image_out",
                            "background_motion", "primitives", "masks", "sfx", "moods", "motion", "treatments"}
    assert P.FORMATS == {"9:16": (1080, 1920), "16:9": (1920, 1080)}
    assert len(P.TRANSITIONS) == 9 and "shape_reveal" in P.TRANSITIONS
    assert P.BUNDLED_FONTS[:6] == ["Big Shoulders Display", "Facade", "Space Grotesk", "Interlope", "Major Mono Display", "Anybody"]       # the legacy slots are liked faces now
    assert len(P.BUNDLED_FONTS) == len(set(P.BUNDLED_FONTS)) >= 20
    assert len(MOOD_TABLE) >= 8


# --------------------------------------------------------------------------
# validate_plan: one test per rule
# --------------------------------------------------------------------------

def test_rule_scenes_gap(good, ana):
    good["scenes"][1]["t0"] = 2.1
    errs_with(good, ana, "gap/overlap")


def test_rule_scenes_overlap_unsorted(good, ana):
    good["scenes"].reverse()
    errs = P.validate_plan(good, ana)
    assert any("scenes[0].t0" in e for e in errs) and any("gap/overlap" in e for e in errs)


def test_rule_scenes_start_at_zero(good, ana):
    good["scenes"][0]["t0"] = 0.5
    errs_with(good, ana, "scenes[0].t0")


def test_rule_scenes_end_at_duration(good, ana):
    good["duration"] = 5.0
    errs_with(good, ana, "must equal duration")


def test_rule_layer_time_before_scene(good, ana):
    good["scenes"][1]["layers"][0]["t0"] = 1.5
    errs_with(good, ana, "outside allowed")


def test_rule_layer_time_past_transition(good, ana):
    good["scenes"][0]["layers"][3]["t1"] = 2.2      # > 2.0 + 0.3/2
    errs_with(good, ana, "outside allowed")


def test_rule_layer_time_last_scene_ignores_transition(good, ana):
    good["scenes"][1]["transition_out"] = {"type": "crossfade", "dur": 1.0}
    good["scenes"][1]["layers"][0]["t1"] = 4.3
    errs_with(good, ana, "outside allowed")


def test_rule_layer_t1_after_t0(good, ana):
    good["scenes"][0]["layers"][0]["t1"] = 0.0
    errs_with(good, ana, "must be > t0")


def test_rule_ref_background_image(good, ana):
    good["scenes"][0]["background"]["src"] = "img9"
    errs_with(good, ana, "image id 'img9'")


def test_rule_ref_image_layer(good, ana):
    good["scenes"][1]["layers"][0]["src"] = "nope"
    errs_with(good, ana, "image id 'nope'")


def test_rule_ref_contour(good, ana):
    good["scenes"][0]["layers"][0]["ref"] = "img0:c99"
    errs_with(good, ana, "contour id")


def test_rule_ref_contour_mask(good, ana):
    good["scenes"][1]["layers"][0]["mask"] = "contour:img1:c42"
    errs_with(good, ana, "contour id")


def test_rule_ref_line(good, ana):
    good["scenes"][0]["layers"][1]["ref"] = "img0:l9"
    errs_with(good, ana, "line id")


def test_rule_line_id_fallback(good, ana):
    for img in ana["images"]:
        for ln in img["lines"]:
            del ln["id"]
    assert P.validate_plan(good, ana) == []


def test_rule_refs_skipped_without_analysis(good):
    good["scenes"][0]["background"]["src"] = "img9"
    assert P.validate_plan(good, None) == []


def test_rule_palette_index(good, ana):
    good["scenes"][0]["layers"][3]["color"] = "palette:7"
    errs_with(good, ana, "out of range")


def test_rule_palette_index_nested(good, ana):
    good["scenes"][0]["background"]["treatment"]["duotone"] = ["palette:0", "palette:4"]
    errs_with(good, ana, "out of range")


def test_rule_bad_colour_string(good, ana):
    good["scenes"][0]["layers"][0]["stroke"] = "red"
    errs_with(good, ana, "not '#rrggbb'")


def test_rule_palette_size(good, ana):
    good["style"]["palette"] = ["#000000", "#ffffff"]
    errs_with(good, ana, "3..6")


def test_rule_font_style(good, ana):
    good["style"]["fonts"]["display"] = "Comic Sans"
    errs_with(good, ana, "bundled font")


def test_rule_font_layer(good, ana):
    good["scenes"][0]["layers"][3]["font"] = "Helvetica"
    errs_with(good, ana, "bundled font")
    good["scenes"][0]["layers"][3]["font"] = "Facade"     # direct bundled (liked) name is fine
    assert P.validate_plan(good, ana) == []


@pytest.mark.parametrize("path,value,needle", [
    (("scenes", 0, "layers", 3, "in", "verb"), "explode", "text in verb"),
    (("scenes", 0, "layers", 3, "out", "verb"), "type_on", "text out verb"),
    (("scenes", 0, "layers", 3, "emphasis", 0, "verb"), "wobble", "emphasis verb"),
    (("scenes", 0, "layers", 3, "emphasis", 0, "on"), "bars", "emphasis trigger"),
    (("scenes", 0, "layers", 0, "in", "verb"), "word_stagger", "shape in verb"),
    (("scenes", 0, "layers", 0, "out", "verb"), "stroke_draw", "shape out verb"),
    (("scenes", 0, "layers", 2, "primitive"), "hexagon", "primitive"),
    (("scenes", 0, "layers", 0, "source"), "svg", "shape source"),
    (("scenes", 1, "layers", 0, "in", "verb"), "kinetic_split", "image in verb"),
    (("scenes", 1, "layers", 0, "mask"), "star", "mask"),
    (("scenes", 0, "background", "motion", "verb"), "spin", "background motion verb"),
    (("scenes", 0, "background", "type"), "video", "background type"),
    (("scenes", 0, "transition_out", "type"), "star_wipe", "transition"),
    (("scenes", 0, "transition_out", "dir"), "diagonal", "direction"),
    (("audio", "sfx", 0, "type"), "boing", "sfx type"),
    (("audio", "mood"), "sad", "music mood"),
    (("audio", "key"), "H", "key"),
    (("style", "motion"), "wobbly", "motion style"),
])
def test_rule_verbs(good, ana, path, value, needle):
    d = good
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    errs_with(good, ana, needle)


@pytest.mark.parametrize("layer_idx,scene_idx", [(3, 0), (2, 0), (0, 1)])   # text, primitive, image
def test_rule_box_both_formats(good, ana, layer_idx, scene_idx):
    del good["scenes"][scene_idx]["layers"][layer_idx]["box"]["16:9"]
    errs_with(good, ana, "missing format '16:9'")


def test_rule_box_required(good, ana):
    good["scenes"][0]["layers"][3]["box"] = [0.1, 0.1, 0.5, 0.2]
    errs_with(good, ana, "per-format dict")


def test_rule_box_inside_frame(good, ana):
    good["scenes"][0]["layers"][2]["box"]["9:16"] = [0.5, 0.9, 0.6, 0.2]      # primitive: 0..1
    errs_with(good, ana, "not inside 0..1")


def test_rule_text_box_may_run_off_frame(good, ana):
    good["scenes"][0]["layers"][3]["box"]["9:16"] = [-0.3, 0.9, 1.6, 0.5]     # giant type, cropped on purpose
    assert P.validate_plan(good, ana) == []
    good["scenes"][0]["layers"][3]["box"]["9:16"] = [0.5, 1.5, 0.6, 0.8]
    errs_with(good, ana, "not inside -1..2")


def test_rule_contour_shape_needs_no_box(good, ana):
    assert good["scenes"][0]["layers"][0]["box"] is None
    assert P.validate_plan(good, ana) == []


def test_rule_transition_fits(good, ana):
    good["scenes"][0]["transition_out"]["dur"] = 5.0
    errs_with(good, ana, "too long")


def test_rule_audio_times(good, ana):
    good["audio"]["sfx"][0]["t"] = 9.0
    errs_with(good, ana, "> 4.0")
    good = manual_plan()
    good["audio"]["risers"][0] = {"t0": 2.0, "t1": 1.0}
    errs_with(good, ana, "must be > t0")


def test_rule_version(good, ana):
    good["version"] = 2
    errs_with(good, ana, "version")


def test_numpy_values_validate(good, ana):
    good["duration"] = np.float64(4.0)
    good["scenes"][0]["t1"] = np.float32(2.0)
    good["scenes"][1]["t0"] = np.float32(2.0)
    assert P.validate_plan(good, ana) == []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_resolve_color(good):
    assert P.resolve_color(good, "#ff0000") == (1.0, 0.0, 0.0)
    r, g, b = P.resolve_color(good, "palette:2")
    assert abs(r - 1.0) < 1e-9 and abs(g - 0x4d / 255) < 1e-9 and abs(b - 0x2e / 255) < 1e-9
    assert P.resolve_color(good, "#fff") == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        P.resolve_color(good, "palette:9")
    with pytest.raises(ValueError):
        P.resolve_color(good, "blue")


def test_beat_snap():
    beats = [0.0, 0.5, 1.0, 1.5]
    assert P.beat_snap(0.74, beats) == 0.5
    assert P.beat_snap(0.76, beats) == 1.0
    assert P.beat_snap(0.75, beats) == 0.5          # tie -> earlier
    assert P.beat_snap(9.0, np.array(beats)) == 1.5
    assert P.beat_snap(0.3, []) == 0.3


def test_json_roundtrip_numpy(good):
    good["seed"] = np.int64(42)
    good["duration"] = np.float64(4.0)
    good["style"]["energy"] = np.float32(0.5)
    good["audio"]["beats"] = np.arange(0, 4, 0.5)
    good["scenes"][0]["background"]["focus"] = np.array([0.5, 0.4], np.float32)
    good["scenes"][0]["layers"][0]["echo"]["count"] = np.int32(2)
    s = P.to_json(good)
    back = P.from_json(s)
    assert back["seed"] == 42 and isinstance(back["seed"], int)
    assert back["audio"]["beats"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert back["scenes"][0]["background"]["focus"] == pytest.approx([0.5, 0.4])
    assert P.from_json(P.to_json(back)) == back
    assert P.validate_plan(back, fake_analysis(TEXT, n_images=2)) == []


def test_from_json_fenced_and_errors():
    assert P.from_json('```json\n{"version": 1}\n```') == {"version": 1}
    assert P.from_json(b'{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        P.from_json("[1, 2]")


def test_schema_prompt():
    s = P.plan_schema_prompt()
    assert len(s.split()) <= 1500
    for kind in ("text_in", "text_out", "emphasis", "shape_in", "background_motion", "primitives", "sfx", "moods"):
        for verb in P.VERBS[kind]:
            assert verb in s, (kind, verb)
    for t in P.TRANSITIONS:
        assert t in s
    for f in P.BUNDLED_FONTS:
        assert f in s


def test_schema_prompt_example_is_valid():
    ex = copy.deepcopy(P._EXAMPLE_PLAN)
    ex["audio"]["beats"] = [x * 0.5 for x in range(12)]
    ana = {"images": [{"id": "img0", "contours": [{"id": "img0:c0"}], "lines": []}]}
    assert P.validate_plan(ex, ana) == []


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", list(CASES))
def test_planner_valid(case):
    ana = fake_analysis(**CASES[case])
    for seed in (0, 1, 7, 123):
        plan = plan_from_analysis(ana, seed=seed)
        assert P.validate_plan(plan, ana) == [], case
        assert 5.0 <= plan["duration"] <= 15.0
        json.loads(P.to_json(plan))     # serialisable
        all_verbs_used(plan)


def test_planner_text_only():
    ana = fake_analysis(TEXT)
    plan = plan_from_analysis(ana, seed=3)
    texts = [L["text"] for sc in plan["scenes"] for L in sc["layers"] if L["type"] == "text"]
    assert texts == TEXT.splitlines()
    assert all(sc["background"]["type"] in ("gradient", "shapes") for sc in plan["scenes"])
    assert plan["audio"]["source"] == "synth"
    assert 3 <= len(plan["style"]["palette"]) <= 6


def test_planner_keyword_palette():
    ana = fake_analysis("Calm mornings\nbreathe slowly", energy=0.3, keywords=["calm"])
    plan = plan_from_analysis(ana, seed=0)
    assert plan["style"]["palette"] == MOOD_TABLE["calm"]["palette"]
    assert plan["audio"]["mood"] == "calm" and plan["audio"]["scale"] == "major"
    assert plan["style"]["motion"] == "smooth"
    ana = fake_analysis("Future of AI code", energy=0.8, keywords=["tech"])
    plan = plan_from_analysis(ana, seed=0)
    from mograph.font_registry import REGISTRY
    d, b = plan["style"]["fonts"]["display"], plan["style"]["fonts"]["body"]
    assert "techno" in REGISTRY[d]["mood"] and "display" in REGISTRY[d]["role"]
    from mograph import fonts_liked as FL
    assert "text" in REGISTRY[b]["role"] and FL.is_liked(d) and FL.is_liked(b)        # liked faces only; the companion is a liked text face (pairs list non-liked faces)


def test_planner_images_shapes_and_palette():
    ana = fake_analysis(TEXT, n_images=2, energy=0.8)
    plan = plan_from_analysis(ana, seed=0)
    sc0 = plan["scenes"][0]
    assert sc0["background"]["type"] == "image" and sc0["background"]["src"] == "img0"
    contour_layers = [L for L in sc0["layers"] if L["type"] == "shape" and L["source"] == "contour"]
    assert 1 <= len(contour_layers) <= 3
    assert contour_layers[0]["ref"] == "img0:c0" and contour_layers[0]["in"]["verb"] == "stroke_draw"
    assert contour_layers[0]["echo"] is not None
    assert all(L["ref"] != "img0:c2" for L in contour_layers)     # tiny contour skipped
    line_layers = [L for L in sc0["layers"] if L["type"] == "shape" and L["source"] == "line"]
    # img0:l0 runs through the 16:9 headline box and l1 is short and diagonal: neither is drawn
    assert line_layers == []
    pal = plan["style"]["palette"]
    assert "#ff5522" in pal or "#ff5522".upper() in pal      # image accent carried into the palette
    # headline sits in an empty region as mapped per format: 9:16 keeps the top band in frame;
    # in 16:9 the top band is cropped away, so the bottom-left region is used instead
    head = next(L for L in sc0["layers"] if L["type"] == "text")
    assert head["box"]["9:16"][1] < 0.3
    x, y, w, h = head["box"]["16:9"]
    assert y > 0.7 and x + w < 0.45


def test_planner_cycles_images_for_more_lines():
    text = "\n".join(f"line {i}" for i in range(4))
    ana = fake_analysis(text, n_images=2, energy=0.9)
    plan = plan_from_analysis(ana, seed=0, duration=12)
    srcs = [sc["background"]["src"] for sc in plan["scenes"]]
    assert len(srcs) >= 3 and srcs[:3] == ["img0", "img1", "img0"]


def test_planner_scene_bounds_on_downbeats():
    ana = fake_analysis(TEXT, n_images=2, energy=0.7)
    plan = plan_from_analysis(ana, seed=0)
    downs = set(round(d, 4) for d in plan["audio"]["downbeats"])
    beat = 60.0 / plan["audio"]["bpm"]
    for sc in plan["scenes"][1:]:
        assert round(sc["t0"], 4) in downs
    for sc in plan["scenes"]:
        assert sc["t1"] - sc["t0"] >= 4 * beat - 1e-6 or len(plan["scenes"]) == 1


def test_planner_audio_events():
    ana = fake_analysis(TEXT, n_images=2, energy=0.8)
    plan = plan_from_analysis(ana, seed=0)
    bounds = [sc["t1"] for sc in plan["scenes"][:-1]]
    impacts = [s["t"] for s in plan["audio"]["sfx"] if s["type"] == "impact"]
    assert impacts == bounds
    assert [r["t1"] for r in plan["audio"]["risers"]] == bounds
    assert plan["scenes"][-2]["transition_out"]["type"] == "cut"      # hard cut onto the end card
    for sc in plan["scenes"][:-2]:
        assert sc["transition_out"]["type"] in ("zoom_blur", "glitch", "push", "flash")
        if sc["transition_out"]["type"] == "push":
            assert any(s["type"] == "whoosh" for s in plan["audio"]["sfx"])
    low = plan_from_analysis(fake_analysis(TEXT, n_images=2, energy=0.2, valence=0.5), seed=0)
    assert low["audio"]["risers"] == []
    for sc in low["scenes"][:-2]:
        assert sc["transition_out"]["type"] in ("crossfade", "luma_wipe", "shape_reveal")


def test_planner_user_audio():
    ana = fake_analysis(TEXT, n_images=2, audio=True)
    plan = plan_from_analysis(ana, seed=0)
    assert plan["audio"]["source"] == "user"
    assert plan["duration"] == 6.0                       # audio length
    assert plan["audio"]["bpm"] == 120.0
    assert plan["audio"]["beats"][:3] == [0.0, 0.5, 1.0]
    labels = [s["label"] for s in plan["audio"]["sections"]]
    assert labels[0] == "intro" and "drop" in labels


def test_planner_audio_only():
    plan = plan_from_analysis(fake_analysis(audio=True), seed=0)
    for sc in plan["scenes"]:
        assert sc["background"]["type"] == "shapes"
        assert any(L.get("primitive") == "wave" for L in sc["layers"])


def test_planner_duration_rules():
    ana = fake_analysis(TEXT)
    assert plan_from_analysis(ana, duration=7.5)["duration"] == 7.5
    assert plan_from_analysis(ana, duration=40)["duration"] == 15.0
    assert plan_from_analysis(ana, duration=1)["duration"] == 5.0
    assert plan_from_analysis(ana)["duration"] == 10.0
    assert plan_from_analysis(fake_analysis(TEXT, duration=8.0))["duration"] == 8.0
    assert plan_from_analysis(fake_analysis(TEXT, duration=8.0), duration=12)["duration"] == 12.0


def test_planner_tempo_from_energy():
    neutral = "Quarterly report\nThree findings"      # no mood keywords -> energy used as-is
    assert plan_from_analysis(fake_analysis(neutral, energy=0.0, valence=0.5))["audio"]["bpm"] == 84
    assert plan_from_analysis(fake_analysis(neutral, energy=1.0, valence=0.5))["audio"]["bpm"] == 132
    # a matched mood keyword nudges energy (and so tempo) toward that mood's energy
    calm = plan_from_analysis(fake_analysis("so calm", energy=0.5, valence=0.5))
    assert calm["audio"]["bpm"] < 84 + 0.5 * 48


def test_planner_edge_inputs():
    for ana in ({}, fake_analysis(), fake_analysis("x"),
                fake_analysis("\n".join(f"word {i}" for i in range(20)), duration=5),
                fake_analysis(n_images=4, duration=5, energy=0.0)):
        plan = plan_from_analysis(ana, seed=5)
        assert P.validate_plan(plan, ana) == []


def test_planner_many_lines_all_used():
    lines = [f"w{i}" for i in range(12)]
    ana = fake_analysis("\n".join(lines), duration=6)
    plan = plan_from_analysis(ana, seed=0)
    joined = " ".join(L["text"] for sc in plan["scenes"] for L in sc["layers"] if L["type"] == "text")
    assert joined.split() == lines


def test_planner_deterministic():
    ana = fake_analysis(TEXT, n_images=2, audio=False)
    a = P.to_json(plan_from_analysis(ana, seed=11))
    b = P.to_json(plan_from_analysis(copy.deepcopy(ana), seed=11))
    assert a == b
    c = P.to_json(plan_from_analysis(ana, seed=12))
    assert a != c
