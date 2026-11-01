"""Rule-planner fixes from the 06_real-user review (every image used, end card, mood from words,
subject-only outlines, accent plates away from the subject)."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mograph import analyze as A  # noqa: E402
from mograph import plan as P  # noqa: E402
from mograph import planner_rules as R  # noqa: E402
from test_plan import fake_analysis, fake_image  # noqa: E402


def _srcs(plan):
    out = set()
    for sc in plan["scenes"]:
        if sc["background"]["type"] == "image":
            out.add(sc["background"]["src"])
        out |= {L["src"] for L in sc["layers"] if L["type"] == "image"}
    return out


def _texts(sc):
    return [L["text"] for L in sc["layers"] if L["type"] == "text"]


# ---------------------------------------------------------------- 1. every image appears

@pytest.mark.parametrize("energy", [0.2, 0.5, 0.9])
def test_five_images_ten_seconds_all_used(energy):
    ana = fake_analysis("Our new menu", n_images=5, energy=energy)
    plan = R.plan_from_analysis(ana, seed=0, duration=10)
    assert P.validate_plan(plan, ana) == []
    body = [sc for sc in plan["scenes"] if sc["background"]["type"] == "image"][:5]
    assert [sc["background"]["src"] for sc in body] == [f"img{i}" for i in range(5)]
    beat = 60.0 / plan["audio"]["bpm"]
    beats = set(round(b, 4) for b in plan["audio"]["beats"])
    for sc in plan["scenes"][1:]:
        assert round(sc["t0"], 4) in beats                         # cuts stay on the beat grid
    for sc in plan["scenes"]:
        assert sc["t1"] - sc["t0"] >= beat - 1e-3


def test_more_images_than_beats_go_in_as_insets():
    ana = fake_analysis(n_images=12, duration=5.0, energy=0.0)
    plan = R.plan_from_analysis(ana, seed=1)
    assert P.validate_plan(plan, ana) == []
    assert _srcs(plan) == {f"img{i}" for i in range(12)}


# ---------------------------------------------------------------- 2. end card, headline reprise

def test_end_card_holds_key_phrase_on_final_hit():
    ana = fake_analysis("Jazz night Friday 9pm", n_images=5, energy=0.5, keywords=["night", "dark"])
    plan = R.plan_from_analysis(ana, seed=0, duration=10)
    assert P.validate_plan(plan, ana) == []
    first, last = plan["scenes"][0], plan["scenes"][-1]
    assert _texts(first) == ["Jazz night", "Friday 9pm"]           # name as headline, time on its own line
    assert _texts(last)[0] == "Friday 9pm" and "Jazz night" in _texts(last)
    # the end card starts where the synth plays its final hit (last downbeat leaving >= 1 s)
    dbs = [d for d in plan["audio"]["downbeats"] if d <= plan["duration"] - 1.0]
    assert last["t0"] == pytest.approx(dbs[-1])
    assert 1.0 <= plan["duration"] - last["t0"] <= 2.3
    assert plan["scenes"][-2]["transition_out"]["type"] == "cut"
    # held still: no pulse, no exit motion, static background; and not a dead tail
    for L in last["layers"]:
        assert L["emphasis"] == [] and L["out"]["verb"] == "cut"
    assert last["background"]["motion"]["verb"] == "static"
    assert any(L["t0"] > last["t0"] + 1e-6 for L in last["layers"])
    assert any(s["t"] == pytest.approx(last["t0"]) and s["type"] == "impact" for s in plan["audio"]["sfx"])


def test_end_card_reprises_plain_headline():
    ana = fake_analysis("City at night", n_images=3, energy=0.5)
    plan = R.plan_from_analysis(ana, seed=2, duration=10)
    assert _texts(plan["scenes"][0]) == ["City at night"]
    assert _texts(plan["scenes"][-1]) == ["City at night"]


def test_key_phrase():
    assert R._key_phrase("Jazz night Friday 9pm") == ("Friday 9pm", "Jazz night")
    assert R._key_phrase("Open mic, Sat 14th March 8:30pm") == ("Sat 14th March 8:30pm", "Open mic")
    assert R._key_phrase("5 tips for sleep") == ("5 tips for sleep", "")
    assert R._key_phrase("City at night") == ("City at night", "")


def test_text_only_has_no_end_card():
    ana = fake_analysis("Make it move\nOn the beat")
    plan = R.plan_from_analysis(ana, seed=0)
    assert [t for sc in plan["scenes"] for t in _texts(sc)] == ["Make it move", "On the beat"]


# ---------------------------------------------------------------- 3. mood from the user's words

@pytest.mark.parametrize("text, derived, want", [
    ("Our café's new autumn menu", ["new", "bright"], "warm"),
    ("Vintage bike restoration", ["dark"], "warm"),
    ("Jazz night Friday 9pm", ["night", "dark"], "lounge"),
    ("Weekend hike, Southern Alps", ["dark"], "nature"),
    ("Grow your own tomatoes", ["calm"], "nature"),
    ("City at night", ["night", "dark"], "dark"),
])
def test_mood_from_text_not_brightness(text, derived, want):
    ana = fake_analysis(text, n_images=2, energy=0.5, valence=0.4, keywords=derived)
    assert R._pick_mood(ana, 0.5, 0.4)[0] == want


def test_mood_outcomes_for_real_cases():
    cafe = R.plan_from_analysis(fake_analysis("Our café's new autumn menu", n_images=2, energy=0.56,
                                              keywords=["new", "bright"]), seed=0)
    assert cafe["audio"]["mood"] not in ("driving", "dark") and cafe["audio"]["scale"] == "major"
    assert cafe["style"]["fonts"]["display"] != "Anton"
    bike = R.plan_from_analysis(fake_analysis("Vintage bike restoration", n_images=2, energy=0.43,
                                              keywords=["dark"]), seed=0)
    assert bike["audio"]["mood"] != "dark"
    jazz = R.plan_from_analysis(fake_analysis("Jazz night Friday 9pm", n_images=2, energy=0.49,
                                              keywords=["night", "dark"]), seed=0)
    assert jazz["audio"]["scale"] == "minor" and jazz["audio"]["mood"] in ("playful", "dark", "driving")


def test_derived_label_is_only_a_tiebreaker():
    # one typed word beats any number of analyzer-derived labels
    ana = fake_analysis("Fresh coffee", keywords=["dark", "dark", "dark"])
    assert R._pick_mood(ana, 0.5, 0.5)[0] == "warm"
    # with no text hit the derived label still decides
    assert R._pick_mood(fake_analysis("Quarterly report", keywords=["calm"]), 0.5, 0.5)[0] == "calm"
    # ties go to the earliest word in the text, not to table order
    assert R._pick_mood(fake_analysis("garden party"), 0.5, 0.5)[0] == "nature"
    assert R._pick_mood(fake_analysis("party garden"), 0.5, 0.5)[0] == "playful"


# ---------------------------------------------------------------- 4. outlines and lines

def test_analyze_contours_skip_background_texture():
    rng = np.random.default_rng(3)
    img = np.full((480, 640, 3), 0.45, np.float32)
    img += (rng.random((480, 640, 1)) * 0.5 - 0.25).astype(np.float32)    # busy grey texture
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    cv2.circle(img, (400, 240), 110, (0.9, 0.3, 0.1), -1)               # one clear subject
    res = A.analyze_image(np.clip(img, 0, 1), "img0")
    cs = res["contours"]
    assert cs, "the subject outline is kept"
    for c in cs:
        assert 0.0 <= c["salience"] <= 1.0 and 0.0 <= c["edge_support"] <= 1.0
    best = [c for c in cs if c["salience"] >= 0.6 and c["edge_support"] >= 0.5]
    assert best
    x, y, w, h = best[0]["bbox"]
    assert abs(x + w / 2 - 400 / 640) < 0.06 and abs(y + h / 2 - 0.5) < 0.06
    # smoothed: no sawtooth (outline length close to its convex hull's)
    pts = (np.array(best[0]["points"]) * [640, 480]).astype(np.float32)
    assert cv2.arcLength(pts, True) < 1.15 * cv2.arcLength(cv2.convexHull(pts), True)


def test_planner_uses_only_salient_contours():
    ana = fake_analysis("Make it move", n_images=1, energy=0.9)
    img = ana["images"][0]
    img["contours"][0]["salience"] = 0.2                 # background texture
    img["contours"][1]["salience"] = 0.9
    img["contours"][1]["area"] = 0.05
    plan = R.plan_from_analysis(ana, seed=0)
    refs = [L["ref"] for sc in plan["scenes"] for L in sc["layers"]
            if L["type"] == "shape" and L["source"] == "contour"]
    assert "img0:c0" not in refs
    assert all(r == "img0:c1" for r in refs)


def test_lines_must_clear_subject_and_text():
    img = fake_image(0)                                  # 1024x768, saliency box x .4-.85, y .3-.8
    assert R._good_line(img, {"p0": [0.05, 0.12], "p1": [0.9, 0.13]})        # long, horizontal, clear
    assert not R._good_line(img, {"p0": [0.05, 0.5], "p1": [0.95, 0.5]})     # through the subject
    assert not R._good_line(img, {"p0": [0.1, 0.1], "p1": [0.5, 0.6]})       # diagonal
    assert not R._good_line(img, {"p0": [0.1, 0.1], "p1": [0.2, 0.1]})       # short


# ---------------------------------------------------------------- 5. accent plate away from the subject

def _no_empty(i, sal):
    im = fake_image(i)
    im["empty_regions"] = []
    im["saliency_box"] = sal
    im["focus"] = [sal[0] + sal[2] / 2, sal[1] + sal[3] / 2]
    return im


@pytest.mark.parametrize("sal, below", [([0.2, 0.05, 0.6, 0.45], True), ([0.2, 0.5, 0.6, 0.45], False)])
def test_plate_avoids_subject_and_uses_accent(sal, below):
    ana = fake_analysis("Vintage bike restoration", energy=0.4)
    ana["images"] = [_no_empty(0, sal)]
    plan = R.plan_from_analysis(ana, seed=0)
    head = next(L for L in plan["scenes"][0]["layers"] if L["type"] == "text")
    assert head["highlight"] is not None and head["highlight"] != "palette:0"
    pal = plan["style"]["palette"]
    k = int(head["highlight"].split(":")[1])
    assert k >= 2 and np.linalg.norm(R._rgb(pal[k]) - R._rgb("#ff5522") * 0.85 / 1.0) < 0.45
    assert head["color"] in ("palette:0", "palette:1")
    for fmt in P.FORMATS:
        y = head["box"][fmt][1]
        assert (y > 0.5) == below, (fmt, head["box"][fmt])
    assert head["in"]["verb"] in R.PLATE_IN_VERBS                    # plate grows with its words


def test_plate_never_uses_word_stagger():
    for seed in range(12):
        ana = fake_analysis("Make it move now", energy=0.9)
        ana["images"] = [_no_empty(0, [0.2, 0.05, 0.6, 0.45])]
        plan = R.plan_from_analysis(ana, seed=seed)
        for sc in plan["scenes"]:
            for L in sc["layers"]:
                if L["type"] == "text" and L["highlight"]:
                    assert L["in"]["verb"] in R.PLATE_IN_VERBS, (seed, L["in"])
