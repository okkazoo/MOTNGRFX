"""Print / film treatment tests: shape, range, determinism, speed, colour fidelity, dispatcher,
plan validation and plan-level rendering."""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import plan as P  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from mograph.render import treatments as T  # noqa: E402
from mograph.render.common import resolve_color  # noqa: E402
from render_helpers import base_plan, box, make_analysis, text_layer  # noqa: E402

PALETTE = ["#111111", "#f2f2f2", "#ff4d2e", "#2e6bff", "#ffd23f"]
FIX = os.path.join(HERE, "fixtures")

SPECS = {
    "halftone": {"type": "halftone", "cell": 12},
    "screenprint": {"type": "screenprint", "inks": ["palette:0", "palette:2", "#e8dcc3"], "misregister": 3, "grit": 0.4},
    "xerox": {"type": "xerox"},
    "duotone": {"type": "duotone", "colors": ["palette:0", "palette:3"]},
    "tritone": {"type": "tritone"},
    "film_grade": {"type": "film_grade", "preset": "teal_orange", "halation": 0.5},
    "posterize_flat": {"type": "posterize_flat"},
    "paper": {"type": "paper"},
    "grain": {"type": "grain", "amount": 0.3},
    "vignette": {"type": "vignette", "amount": 0.4},
    "light_leak": {"type": "light_leak", "strength": 0.6},
    "dust_scratches": {"type": "dust_scratches", "amount": 0.8},
}
EXTRA = [
    {"type": "halftone", "shape": "line", "angle": 15},
    {"type": "screenprint", "inks": ["palette:0", "palette:3", "palette:2", "palette:4"], "dither": "ordered"},
    {"type": "screenprint", "inks": ["palette:0"], "dither": "diffusion", "boil": 1.0},
    {"type": "screenprint", "dither": "none", "grit": 0.0, "misregister": 0},
    {"type": "film_grade", "preset": "mono"}, {"type": "film_grade", "preset": "warm"},
    {"type": "film_grade", "preset": "cool"}, {"type": "xerox", "boil": 1.0, "toner_grit": 1.0},
]


def _fixture(name, size=None):
    img = np.asarray(Image.open(os.path.join(FIX, name)).convert("RGB"), np.float32) / 255.0
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


@pytest.fixture(scope="module")
def small():
    return _fixture("subject_texture.png", (180, 240))


def test_names_match_plan_verbs():
    assert list(T.TREATMENTS) == P.VERBS["treatments"]
    assert set(SPECS) == set(T.TREATMENTS)
    assert list(T.FILM_PRESETS) == P.FILM_PRESETS


@pytest.mark.parametrize("spec", list(SPECS.values()) + EXTRA, ids=lambda s: s["type"] + str(len(s)))
def test_shape_range_deterministic(small, spec):
    before = small.copy()
    a = T.apply_treatment(small, spec, PALETTE, frame=5, seed=11)
    b = T.apply_treatment(small, spec, PALETTE, frame=5, seed=11)
    assert np.array_equal(small, before)                 # input untouched
    assert a.shape == small.shape and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)                          # deterministic
    assert np.abs(a - small).max() > 0.02                # it did something


def test_time_varying_looks_change_per_frame(small):
    for spec in (SPECS["light_leak"], SPECS["dust_scratches"], SPECS["grain"],
                 {"type": "screenprint", "boil": 2.0}, {"type": "xerox", "boil": 2.0}):
        a = T.apply_treatment(small, spec, PALETTE, frame=0, seed=3)
        diffs = [np.abs(a - T.apply_treatment(small, spec, PALETTE, frame=k, seed=3)).mean() for k in (2, 4, 9)]
        assert sum(d > 1e-4 for d in diffs) >= 2, (spec["type"], diffs)   # a bank pick may repeat once
    # without boil, print looks are static across frames
    s = {"type": "screenprint"}
    assert np.array_equal(T.apply_treatment(small, s, PALETTE, frame=0, seed=3),
                          T.apply_treatment(small, s, PALETTE, frame=9, seed=3))


def test_unknown_and_empty_specs_pass_through(small):
    assert T.apply_treatment(small, None) is small
    assert T.apply_treatment(small, {}) is small
    assert T.apply_treatment(small, {"type": "oil_paint"}) is small
    out = T.apply_treatment(small, {"type": "halftone", "cell": "big", "wobble": 3}, PALETTE)
    assert out.shape == small.shape


# ----------------------------------------------------------------------------- speed

BUDGET_MS = {(640, 360): 8.0, (1920, 1080): 60.0}
MARGIN = {(640, 360): 3.0, (1920, 1080): 2.0}       # generous for shared / noisy CPUs


@pytest.mark.parametrize("size", [(640, 360), (1920, 1080)], ids=["draft", "full"])
def test_time_budget(size):
    img = _fixture("subject_texture.png", size)
    slow = []
    for name, spec in SPECS.items():
        T.apply_treatment(img, spec, PALETTE, frame=0, seed=1)          # build per-size caches
        ts = []
        for k in range(5):
            t = time.perf_counter()
            T.apply_treatment(img, spec, PALETTE, frame=k, seed=1)
            ts.append((time.perf_counter() - t) * 1000.0)
        best = min(ts)
        if best > BUDGET_MS[size] * MARGIN[size]:
            slow.append((name, round(best, 2)))
    assert not slow, f"over {BUDGET_MS[size]} ms x{MARGIN[size]} at {size}: {slow}"


# ----------------------------------------------------------------------------- fidelity

def test_screenprint_uses_only_inks_and_paper():
    img = _fixture("blob_gradient.png", (540, 720))
    inks = ["#111111", "#ff4d2e", "#2e6bff"]
    paper = "#f1ebdd"
    for dither in ("noise", "ordered", "diffusion"):
        out = T.screenprint(img, inks, paper=paper, misregister=4, grit=0.5, dither=dither, texture=0.0,
                            seed=2, scale=1.0)
        tones = np.stack([resolve_color(None, c) for c in inks + [paper]])
        d = np.abs(out.reshape(-1, 1, 3) - tones[None]).max(-1).min(-1)
        assert (d <= 3.0 / 255.0).mean() > 0.95, dither      # rest = 1 px anti-aliased ink edges
        used = (np.abs(out.reshape(-1, 1, 3) - tones[None]).max(-1) <= 3.0 / 255.0).sum(0)
        assert (used > 0.01 * d.size).sum() >= 3, (dither, used)   # several inks really print


def test_screenprint_misregistration_shows_lighter_ink_at_edges():
    img = np.ones((200, 200, 3), np.float32)
    img[60:140, 60:140] = 0.0                                          # black square on white
    kw = dict(inks=["#000000", "#ff0000"], paper="#ffffff", grit=0.0, dither="none", texture=0.0, seed=4,
              scale=1.0, smooth=0.0)
    reg = T.screenprint(img, misregister=0, **kw)
    mis = T.screenprint(img, misregister=12, **kw)
    red = lambda o: ((o[..., 0] > 0.8) & (o[..., 1] < 0.2)).sum()     # noqa: E731
    assert red(mis) > red(reg) + 100


def test_halftone_coverage_increases_with_darkness():
    cov = []
    for level in (0.95, 0.8, 0.6, 0.4, 0.2, 0.05):
        img = np.full((240, 240, 3), level, np.float32)
        out = T.halftone(img, cell=14, ink="#000000", paper="#ffffff", auto_levels=False, texture=0.0, scale=1.0)
        c = 1.0 - float(out.mean())
        cov.append(c)
        assert abs(c - (1.0 - level)) < 0.08, (level, c)
    assert all(b > a for a, b in zip(cov, cov[1:])), cov
    # dots are round: a light tone gives isolated blobs of similar size
    img = np.full((240, 240, 3), 0.8, np.float32)
    out = T.halftone(img, cell=16, ink="#000000", paper="#ffffff", auto_levels=False, texture=0.0, scale=1.0)
    n, _, stats, _ = cv2.connectedComponentsWithStats((out[..., 0] < 0.5).astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    inner = areas[areas > 0.5 * np.median(areas)]
    assert n > 100 and inner.std() / inner.mean() < 0.25


def test_duotone_exact_matches_legacy_and_fast_is_close(small):
    cd, cl = resolve_color(None, "#111111"), resolve_color(None, "#2e6bff")
    L = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)       # legacy background.py maths
    lo, hi = np.percentile(L, [2, 98])
    L = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)[..., None]
    legacy = (cd[None, None] * (1 - L) + cl[None, None] * L).astype(np.float32)
    assert np.array_equal(T.duotone(small, ["#111111", "#2e6bff"], exact=True), legacy)
    fast = T.duotone(small, ["#111111", "#2e6bff"])
    assert np.abs(fast - legacy).mean() < 4.0 / 255.0


def test_posterize_flat_snaps_to_palette():
    img = _fixture("blob_gradient.png", (360, 480))
    pal = ["#1d2b3a", "#f4efe4", "#ff4d2e", "#ffd23f"]
    out = T.posterize_flat(img, pal, smooth=0.5, scale=0.5)
    tones = np.stack([resolve_color(None, c) for c in pal])
    d = np.abs(out.reshape(-1, 1, 3) - tones[None]).max(-1).min(-1)
    assert (d <= 1.5 / 255.0).mean() > 0.97


def test_apply_treatment_resolves_palette_refs():
    ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[None, :, None], 3, 2)
    ramp = np.ascontiguousarray(np.repeat(ramp, 32, 0))
    out = T.apply_treatment(ramp, {"type": "duotone", "colors": ["palette:2", "palette:3"]}, PALETTE)
    assert np.allclose(out[:, 0], resolve_color(None, PALETTE[2]), atol=2 / 255)
    assert np.allclose(out[:, -1], resolve_color(None, PALETTE[3]), atol=2 / 255)
    # rgb-array palettes work too, and refs inside ink lists resolve
    pal_rgb = [resolve_color(None, c) for c in PALETTE]
    sp = T.apply_treatment(ramp, {"type": "screenprint", "inks": ["palette:3"], "paper": "palette:4", "grit": 0,
                                  "misregister": 0, "dither": "none", "texture": 0}, pal_rgb, scale=1.0)
    assert np.allclose(sp[:, 0], resolve_color(None, PALETTE[3]), atol=2 / 255)
    assert np.allclose(sp[:, -1], resolve_color(None, PALETTE[4]), atol=2 / 255)
    # defaults come from the palette: halftone ink = palette:0
    ht = T.apply_treatment(np.zeros((64, 64, 3), np.float32), {"type": "halftone", "texture": 0}, ["#0000ff"] + PALETTE[1:])
    assert ht[..., 2].mean() > 0.9 and ht[..., 0].mean() < 0.1


def test_paper_texture():
    p = T.paper((300, 200), "#efe6d2", fibre=0.8, speckle=0.5, seed=1)
    assert p.shape == (300, 200, 3) and p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1
    assert np.array_equal(p, T.paper((300, 200), "#efe6d2", fibre=0.8, speckle=0.5, seed=1))
    assert p.std() > 0.005
    assert np.abs(p.reshape(-1, 3).mean(0) - resolve_color(None, "#efe6d2")).max() < 0.05


# ----------------------------------------------------------------------------- plan + render

@pytest.fixture(scope="module")
def analysis():
    return make_analysis(scale=0.5)


def _treated_plan(duration=1.0, fps=10):
    p = base_plan(duration, fps)
    p["style"]["treatment"] = [{"type": "film_grade", "preset": "teal_orange"},
                               {"type": "light_leak", "strength": 0.4}]
    p["scenes"] = [{
        "id": "s0", "t0": 0.0, "t1": duration,
        "background": {"type": "image", "src": "img0", "focus": [0.45, 0.5],
                       "motion": {"verb": "ken_burns", "zoom": [1.0, 1.1], "pan": [[0, 0], [0.02, 0]]},
                       "treatment": {"type": "screenprint", "inks": ["palette:0", "palette:2"], "misregister": 3,
                                     "duotone": None, "blur": 0.0, "darken": 0.1}},
        "layers": [
            {"type": "image", "src": "img2", "mask": "circle", "box": box([0.2, 0.1, 0.6, 0.3], [0.6, 0.1, 0.3, 0.5]),
             "t0": 0.0, "t1": duration, "in": {"verb": "cut"}, "out": {"verb": "cut"}, "emphasis": [],
             "border": None, "treatment": {"type": "halftone", "cell": 14, "ink": "palette:3"}},
            text_layer("Printed", 0.0, duration, "cut", "cut"),
        ],
        "transition_out": {"type": "cut", "dur": 0.0}}]
    return p


def test_plan_validation_accepts_and_rejects_treatments(analysis):
    plan = _treated_plan()
    assert [e for e in P.validate_plan(plan, analysis) if "treatment" in e] == []
    bad = _treated_plan()
    bad["style"]["treatment"] = {"type": "oil_paint"}
    bad["scenes"][0]["background"]["treatment"]["inks"] = ["palette:9"]
    bad["scenes"][0]["layers"][0]["treatment"] = {"type": "film_grade", "preset": "sepia", "halation": 5}
    errs = " | ".join(P.validate_plan(bad, analysis))
    for needle in ("style.treatment.type", "treatment.inks[0]", "treatment.preset", "treatment.halation"):
        assert needle in errs, (needle, errs)
    legacy = _treated_plan()
    legacy["style"].pop("treatment")
    legacy["scenes"][0]["background"]["treatment"] = {"duotone": ["palette:0", "palette:4"], "blur": 0.2, "darken": 0.1}
    legacy["scenes"][0]["layers"][0].pop("treatment")
    assert [e for e in P.validate_plan(legacy, analysis) if "treatment" in e] == []


@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
def test_plan_with_treatments_renders(analysis, fmt):
    plan = _treated_plan()
    frames = list(render_frames(plan, analysis, fmt, "draft"))
    assert len(frames) == 10 and frames[0].dtype == np.uint8
    plain = _treated_plan()
    plain["style"].pop("treatment")
    plain["scenes"][0]["background"]["treatment"] = {"duotone": None, "blur": 0.0, "darken": 0.1}
    plain["scenes"][0]["layers"][0].pop("treatment")
    ref = list(render_frames(plain, analysis, fmt, "draft"))
    assert np.abs(frames[3].astype(int) - ref[3].astype(int)).mean() > 5
    again = list(render_frames(plan, analysis, fmt, "draft"))
    assert all(np.array_equal(a, b) for a, b in zip(frames, again))


def test_style_grain_treatment_matches_global_grain(analysis):
    """A {"type": "grain"} style treatment reproduces the compositor's global grain exactly."""
    a = _treated_plan(2.0, 10)
    a["style"].pop("treatment")
    a["scenes"][0]["background"]["treatment"] = {"duotone": None, "blur": 0.0, "darken": 0.1}
    a["scenes"][0]["layers"][0].pop("treatment")
    a["style"].update(grain=0.3, vignette=0.0, beat_flash=0.0)
    b = {**a, "style": dict(a["style"], grain=0.0, treatment={"type": "grain", "amount": 0.3})}
    fa = list(render_frames(a, analysis, "9:16", "draft"))
    fb = list(render_frames(b, analysis, "9:16", "draft"))
    for k in range(0, 16):          # before the final fade
        assert np.array_equal(fa[k], fb[k]), k


def test_background_types_accept_typed_treatments(analysis):
    for bg in ({"type": "color", "color": "palette:0", "treatment": {"type": "paper", "tone": "#d8d0c0"}},
               {"type": "gradient", "colors": ["palette:0", "palette:3"], "motion": {"verb": "drift"},
                "treatment": [{"type": "halftone", "cell": 20}]},
               {"type": "shapes", "color": "palette:0", "treatment": {"type": "xerox", "boil": 1.0}}):
        def plan_with(b):
            p = base_plan(1.0, 10)
            p["scenes"] = [{"id": "s0", "t0": 0.0, "t1": 1.0, "background": b, "layers": [],
                            "transition_out": {"type": "cut", "dur": 0.0}}]
            return p
        p = plan_with(bg)
        assert [e for e in P.validate_plan(p, analysis) if "treatment" in e] == []
        frames = list(render_frames(p, analysis, "9:16", "draft"))
        ref = list(render_frames(plan_with({k: v for k, v in bg.items() if k != "treatment"}), analysis, "9:16"))
        assert len(frames) == 10
        assert np.abs(frames[2].astype(int) - ref[2].astype(int)).mean() > 1.0, bg["type"]
