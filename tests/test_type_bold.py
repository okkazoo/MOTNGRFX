"""Font registry, registry-driven font picks in the planners, and the bold-type text options
(axes / vary, outline + swap, window fill / knockout)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import font_registry as FR  # noqa: E402
from mograph import plan as P  # noqa: E402
from mograph import planner_rules as R  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from mograph.render.text import get_font  # noqa: E402
from render_helpers import base_plan, text_layer  # noqa: E402
from test_plan import fake_analysis  # noqa: E402

FPS = 20
NOIMG = {"images": []}


# ----------------------------------------------------------------------------- registry

def test_registry_tags_and_files():
    roles = {"display", "text", "accent"}
    for name, f in FR.REGISTRY.items():
        assert os.path.isfile(FR.font_path(name)), name
        assert os.path.isfile(os.path.join(FR.FONT_DIR, f["licence"])), name
        lic = "APACHE LICENSE" if f.get("licence_id") == "Apache-2.0" else "SIL OPEN FONT LICENSE"
        assert lic in open(os.path.join(FR.FONT_DIR, f["licence"]), encoding="utf-8").read().upper(), name
        assert set(f["mood"]) <= set(FR.MOOD_TAGS) and f["mood"], name
        assert set(f["role"]) <= roles and f["role"], name
        assert f["width"] in ("condensed", "normal", "wide", "extended", "variable")
        assert f["contrast"] in ("low", "mid", "high")
        assert all(p in FR.REGISTRY for p in f["pairs"]), name
        lo, hi = f["energy"]
        assert 0.0 <= lo < hi <= 1.0
    # every mood has at least three display faces to choose from
    for m in FR.MOOD_TAGS:
        assert sum(1 for f in FR.REGISTRY.values() if m in f["mood"] and "display" in f["role"]) >= 3, m
    assert {"Big Shoulders Display", "Space Grotesk", "Anybody", "Interlope"} <= set(P.BUNDLED_FONTS)        # plan fonts are liked faces only (2026-10-30)
    assert {"Anton", "Inter", "Fraunces", "Climate Crisis", "Instrument Sans"} <= set(FR.REGISTRY) and not {"Anton", "Inter"} & set(P.BUNDLED_FONTS)
    # rare faces (independent libre foundries): provenance recorded, a shape tag, foils that exist
    rare = [n for n in FR.REGISTRY if FR.is_rare(n)]
    assert len(rare) >= 12 and {"Pilowlava", "Cantique", "Sinistre", "Sprat Extended"} <= set(rare)
    for name in rare:
        f = FR.REGISTRY[name]
        assert f["source"].startswith("https://") and f["author"] and f["foundry"], name
        assert f["licence_id"] == "OFL-1.1" and f["shape"] in ("puffy", "fine", "tall", "crazy"), name
        assert "display" in f["role"] and FR.foils(name) and len(FR.foils(name)) == len(f["foil"]), name
        assert all("text" in FR.REGISTRY[c]["role"] for c in f["pairs"]), name
    assert {FR.REGISTRY[n]["shape"] for n in rare} >= {"puffy", "fine"}


def test_variable_axes_change_glyphs():
    path = FR.font_path("Anybody")         # was Syne (not liked): a liked variable face whose width grows with weight
    thin = get_font(FR.font_key(path, {"wght": 100, "wdth": 100}), 80).getlength("HELLO")
    heavy = get_font(FR.font_key(path, {"wght": 900, "wdth": 100}), 80).getlength("HELLO")
    assert heavy > thin * 1.1
    p2 = FR.font_path("Anybody")
    narrow = get_font(FR.font_key(p2, {"wdth": 50, "wght": 900}), 80).getlength("HELLO")
    wide = get_font(FR.font_key(p2, {"wdth": 150, "wght": 900}), 80).getlength("HELLO")
    assert wide > narrow * 1.6
    assert FR.split_key(FR.font_key(path, {"wght": 800})) == (path, {"wght": 800.0})


def test_pick_fonts_deterministic_and_varied():
    for moods in (["hype"], ["calm"], ["warm", "nostalgic"], ["techno"], ["luxury"], ["playful"], ["dark"]):
        picks = [FR.pick_fonts(moods, 0.6, s, "x") for s in range(30)]
        assert picks == [FR.pick_fonts(moods, 0.6, s, "x") for s in range(30)]
        from mograph import fonts_liked as FL
        pool = [n for n, f in FR.REGISTRY.items() if FL.is_liked(n) and "display" in f["role"] and any(m in f["mood"] for m in moods)]
        assert len({d for d, _ in picks}) >= (3 if len(pool) >= 3 else 1), moods        # the default pool is the liked faces only (2026-10-30)
        assert all(FL.is_liked(d) and FL.is_liked(b) for d, b in picks), moods
        for d, b in picks:
            assert moods[0] in FR.REGISTRY[d]["mood"] or any(m in FR.REGISTRY[d]["mood"] for m in moods)
            assert "display" in FR.REGISTRY[d]["role"] and "text" in FR.REGISTRY[b]["role"]


def test_pick_fonts_prefers_rare_display():
    for moods in (["hype"], ["calm"], ["warm", "nostalgic"], ["techno"], ["luxury"], ["playful"], ["dark"]):
        ALL = set(FR.REGISTRY)          # explicit pool: this tests the rarity WEIGHTING; the default pool is the liked faces (no rare face is liked)
        picks = [FR.pick_fonts(moods, 0.6, s, "r", allowed=ALL) for s in range(60)]
        assert picks == [FR.pick_fonts(moods, 0.6, s, "r", allowed=ALL) for s in range(60)]       # deterministic per seed
        rare = sum(FR.is_rare(d) for d, _ in picks) / len(picks)
        common = [FR.pick_fonts(moods, 0.6, s, "r", prefer_rare=False, allowed=ALL)[0] for s in range(60)]
        assert rare >= 0.65 and rare > sum(map(FR.is_rare, common)) / len(common), moods
        assert len({d for d, _ in picks if FR.is_rare(d)}) >= 3, moods
        # companions stay quiet common text faces
        assert all("text" in FR.REGISTRY[b]["role"] and not FR.is_rare(b) for _, b in picks)
    # common faces remain reachable as display fallbacks
    assert any(not FR.is_rare(FR.pick_fonts(["hype"], 0.6, s, "r")[0]) for s in range(60))


def test_rule_planner_picks_registry_fonts():
    seen = set()
    for seed in range(12):
        ana = fake_analysis("Our cafe's autumn menu\nFresh bread daily", energy=0.55, keywords=["cafe"])
        assert R._pick_mood(ana, 0.55, 0.6)[0] == "warm"
        plan = R.plan_from_analysis(ana, seed=seed)
        assert plan == R.plan_from_analysis(ana, seed=seed)
        d, b = plan["style"]["fonts"]["display"], plan["style"]["fonts"]["body"]
        assert d in P.BUNDLED_FONTS and b in P.BUNDLED_FONTS
        assert set(FR.REGISTRY[d]["mood"]) & set(R.FONT_MOODS["warm"])
        heads = [L for sc in plan["scenes"] for L in sc["layers"] if L["type"] == "text" and L["role"] == "headline"]
        want = FR.preferred_case(d) if FR.preferred_case(d) in ("as_is", "lower") else "upper"
        assert all(L["case"] == want for L in heads)
        seen.add(d)
    assert len(seen) >= 3


def test_schema_prompt_lists_font_guide_and_options():
    s = P.plan_schema_prompt()
    assert "Rock 3D" in s and "Climate Crisis" not in s and "Anton (" not in s and "animates" in s and "knockout" in s and '"vary"' in s
    assert len(s.split()) <= 1500


# ----------------------------------------------------------------------------- render helpers

def _plan(layers, duration=1.0, display="Big Shoulders Display", bg=None):
    p = base_plan(duration, FPS)
    p["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
    p["style"]["fonts"] = {"display": display, "body": "Space Grotesk"}
    p["scenes"] = [{"id": "s0", "t0": 0.0, "t1": duration,
                    "background": bg or {"type": "color", "color": "palette:0"},
                    "layers": layers, "transition_out": {"type": "cut", "dur": 0.0}}]
    return p


def _frames(plan, analysis=NOIMG, fmt="9:16"):
    return [f.astype(np.int32) for f in render_frames(plan, analysis, fmt)]


def _ink(f):
    return float((f.mean(axis=2) > 128).mean())


BIG = {"9:16": [0.05, 0.3, 0.9, 0.3], "16:9": [0.05, 0.25, 0.9, 0.4]}


# ----------------------------------------------------------------------------- axes / vary

def test_static_axes_weight():
    thin = _frames(_plan([text_layer("BOLD", 0.0, 1.0, "cut", "cut", box=BIG, axes={"wght": 100})], display="Big Shoulders Display"))
    fat = _frames(_plan([text_layer("BOLD", 0.0, 1.0, "cut", "cut", box=BIG, axes={"wght": 900})], display="Big Shoulders Display"))
    assert _ink(fat[5]) > 1.8 * _ink(thin[5]) > 0


def test_vary_weight_wave_moves_and_validates():
    lay = text_layer("WAVE FORM", 0.0, 2.0, "cut", "cut", box=BIG,
                     vary={"axis": "wght", "range": [100, 900], "period": 1.0, "phase": 0.1})
    p = _plan([lay], duration=2.0, display="Anybody")
    assert P.validate_plan(p, NOIMG) == []
    fr = _frames(p)
    inks = [_ink(f) for f in fr[:-7]]                # before the clip's final fade
    assert max(inks) > 1.25 * min(inks) > 0           # the weight breathes
    assert np.abs(fr[3] - fr[8]).mean() > 0.5
    assert all(np.array_equal(a, b) for a, b in zip(fr, _frames(p)))   # deterministic
    # glyphs stay inside the frame horizontally at the widest point of the wave
    for f in fr[:-7]:
        lit = np.nonzero(f.mean(axis=2) > 128)[1]
        assert lit.min() > 0 and lit.max() < f.shape[1] - 1


def test_vary_scale_on_static_font_and_beat_pulse():
    lay = text_layer("PULSE", 0.0, 2.0, "cut", "cut", box=BIG, vary={"scale": 0.4, "on": "beats", "phase": 0.03})
    fr = _frames(_plan([lay], duration=2.0, display="Passion One"))
    inks = [_ink(f) for f in fr[:-7]]
    assert max(inks) > 1.2 * min(inks) > 0


# ----------------------------------------------------------------------------- outline

def test_outline_is_hollow_and_swaps_on_beats():
    fill = _frames(_plan([text_layer("HOLLOW", 0.0, 2.0, "cut", "cut", box=BIG)], 2.0))
    ol = _frames(_plan([text_layer("HOLLOW", 0.0, 2.0, "cut", "cut", box=BIG, outline={"width": 0.03})], 2.0))
    assert 0 < _ink(ol[3]) < 0.6 * _ink(fill[3])
    # 120 bpm: beats at 0.5 s steps -> outline in (0, 0.5], fill in (0.5, 1.0], outline again ...
    sw = _frames(_plan([text_layer("HOLLOW", 0.0, 2.0, "cut", "cut", box=BIG,
                                   outline={"width": 0.03, "swap": "beats"})], 2.0))
    assert np.array_equal(sw[3], ol[3])
    assert np.array_equal(sw[13], fill[13])
    assert np.array_equal(sw[23], ol[23])


def test_outline_alt_rows_in_row_slide():
    lay = text_layer("SEA", 0.0, 1.0, "row_slide", "cut", box=BIG,
                     outline={"width": 0.04, "rows": "alt"}, **{"in": {"verb": "row_slide", "dur": 0.3, "rows": 3}})
    p = _plan([lay])
    assert P.validate_plan(p, NOIMG) == []
    a = _frames(p)[15]
    b = _frames(_plan([text_layer("SEA", 0.0, 1.0, "row_slide", "cut", box=BIG,
                                  **{"in": {"verb": "row_slide", "dur": 0.3, "rows": 3}})]))[15]
    assert 0 < _ink(a) < 0.85 * _ink(b)


# ----------------------------------------------------------------------------- window

def _red_image():
    px = np.zeros((400, 300, 3), np.float32)
    px[..., 0] = 1.0                                   # pure red (RGB)
    return {"images": [{"id": "img0", "pixels": px, "width": 300, "height": 400, "focus": [0.5, 0.5],
                        "contours": [], "lines": []}]}


def test_window_fill_shows_the_image_inside_letters():
    ana = _red_image()
    lay = text_layer("RED", 0.0, 1.0, "cut", "cut", box=BIG, window={"mode": "fill", "src": "img0", "zoom": 1.0})
    p = _plan([lay])
    assert P.validate_plan(p, ana) == []
    f = _frames(p, ana)[5]
    red = (f[..., 0] > 200) & (f[..., 1] < 60) & (f[..., 2] < 60)
    assert red.mean() > 0.02
    assert not ((f[..., 1] > 200) & (f[..., 2] > 200)).any()   # no white text drawn
    bad = text_layer("RED", 0.0, 1.0, "cut", "cut", box=BIG, window={"mode": "fill", "src": "nope"})
    assert any("window.src" in e for e in P.validate_plan(_plan([bad]), ana))


def test_window_knockout_cuts_letters_out_of_a_plate():
    bg = {"type": "color", "color": "#00ff00"}
    lay = text_layer("CUT", 0.0, 1.0, "cut", "cut", box=BIG, window={"mode": "knockout", "plate": "palette:0"})
    f = _frames(_plan([lay], bg=bg))[5]
    green = (f[..., 1] > 200) & (f[..., 0] < 60)
    dark = f.mean(axis=2) < 40
    assert 0.02 < green.mean() < 0.4                   # only the letters show the green ground
    assert dark.mean() > 0.5                           # the plate covers the rest
    # padded plate: only a band around the text is covered
    lay2 = text_layer("CUT", 0.0, 1.0, "cut", "cut", box=BIG, window={"mode": "knockout", "plate": "palette:0", "pad": 0.2})
    f2 = _frames(_plan([lay2], bg=bg))[5]
    assert ((f2[..., 1] > 200) & (f2[..., 0] < 60)).mean() > 0.5


def test_bad_option_values_rejected():
    bad = [text_layer("X", 0.0, 1.0, outline={"swap": "bars"}),
           text_layer("X", 0.0, 1.0, window={"mode": "stencil"}),
           text_layer("X", 0.0, 1.0, vary={"range": [1]}),
           text_layer("X", 0.0, 1.0, axes={"wght": "heavy"})]
    for lay in bad:
        assert P.validate_plan(_plan([lay]), NOIMG), lay
