"""Planner / timing / layout rules from the artifact-017 review (16_app/PLAN.md "Visual decisions"): line breaks, no caption, pace from
the tempo, readable holds, beats, hard cuts, transition variety, settle, mattes, trails.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_fusion_v004_plan.py -q   (CPU only)"""
from __future__ import annotations

import copy
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from mograph.modules import core  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import spec  # noqa: E402
from tools.fusion.beats import Beats  # noqa: E402
from test_fusion_core import synth_music, _req  # noqa: E402


# ----------------------------------------------------------------------------------------------- 1. line breaks
@pytest.mark.parametrize("word", ["AND", "HOURS", "BLOOM", "GOING", "DREAMING", "AFTER", "I", "OK"])
@pytest.mark.parametrize("landscape", [False, True])
def test_no_lone_letter_line_unless_all_lines_are_single_letters(word, landscape):
    for ls in spec._candidates(word, 5, landscape):
        assert "".join(ls) == word
        if len(ls) > 1 and len(ls) != len(word):
            assert min(len(x) for x in ls) >= (3 if landscape else 2), (word, ls)
    assert [word] in spec._candidates(word, 5, landscape)


def test_the_review_cases_have_no_bad_break():
    bad = {"AND": ["AN", "D"], "HOURS": ["HO", "U", "RS"], "BLOOM": ["BL", "O", "OM"], "GOING": ["GO", "I", "NG"]}
    for w, b in bad.items():
        assert b not in spec._candidates(w, 5)
        assert b not in spec._candidates(w, 5, True)
    assert spec._candidates("DREAMING", 5, True) == [["DREAMING"]] and spec._candidates("DREAMING", 5) == [["DREAMING"]]      # a word is never cut across lines (round-3 QC 2026-10-29)


def _fit(word, size):
    key, _ = core.resolve_font("Big Shoulders Display")
    W, H = spec.world_for(size)
    return spec.fit_lines(dict(key=key, axes=None), word, W * 0.94, 1.08 * H * 0.9, None, 5)[0]


@pytest.mark.parametrize("word", ["AND", "HOURS", "BLOOM", "GOING", "DREAMING"])
def test_fit_lines_never_leaves_a_single_letter(word):
    for size in ((576, 1024), (1024, 576)):
        ls = _fit(word, size)
        assert "".join(ls) == word
        assert len(ls) == 1 or len(ls) == len(word) or min(len(x) for x in ls) >= 2, (word, size, ls)
    assert _fit("DREAMING", (1024, 576)) == ["DREAMING"]                                   # landscape: fits, stays whole
    assert _fit("HOURS", (1024, 576)) == ["HOURS"]


# ----------------------------------------------------------------------------------------------- 3. pace from the tempo
def _gaps(bpm, tier=None, mood=None):
    grid = Beats(bpm, None, 4, 0.0, 60.0)
    words = [dict(text="ONE", form_in="lines", form_out="lines"), dict(text="TWO", form_in="lines", form_out="lines")]
    sp = {k: v for k, v in dict(tier=tier, mood=mood).items() if v}
    return spec.plan_timing(words, sp, grid)[0]["gap"], grid


def test_transition_gap_follows_tempo_and_tier():
    calm, g = _gaps(72, "calm")
    assert 1.2 - 1e-6 <= calm <= 3.2 + g.iv and calm >= 3.0                     # 4 beats at 72 bpm = 3.33 s, capped 3.2 (+ beat snap)
    mid, _ = _gaps(100, "mid")
    kin, _ = _gaps(124, "kinetic")
    assert 1.5 <= mid <= 2.2 + 0.6                                              # 3 beats at 100 bpm = 1.8 s
    assert 0.8 <= kin <= 1.4 + 0.5                                              # 2 beats at 124 bpm = 0.97 s
    assert calm > mid > kin
    old = lambda iv: min(max(2.0 * iv, 0.6), 1.4)                               # the previous rule: <= 1.4 s whatever the mood
    assert mid > old(0.6) + 0.3 and calm > 2 * old(0.833)
    slow_kin, _ = _gaps(80, "kinetic")                                           # a slow tempo is calm whatever the tier says
    assert slow_kin >= 3.0
    assert spec.pace_of({"tier": "kinetic"}, Beats(88, None, 4, 0, 60)) == "calm"
    assert spec.pace_of({"mood": "hype"}, Beats(130, None, 4, 0, 60)) == "kinetic"
    assert spec.pace_of({"mood": "calm"}, Beats(130, None, 4, 0, 60)) == "calm"
    assert spec.pace_of({}, Beats(120, None, 4, 0, 60)) == "mid"


def test_calm_form_and_exit_go_longer():
    grid = Beats(72, None, 4, 0.0, 60.0)
    w = [dict(text="A WORD HERE", t0=0.0, t1=9.0, form_in="lines", form_out="lines")]
    t = spec.plan_timing(w, {"tier": "calm"}, grid)[0]
    k = spec.plan_timing(w, {"tier": "kinetic"}, Beats(130, None, 4, 0.0, 60.0))[0]
    assert t["f"] > 2.0 and t["x"] > 1.6                                         # calm form up to 2.6 s, exit up to 2.2 s
    assert k["f"] <= 1.8 + 0.5 and k["x"] <= 1.4 + 0.5


def test_hold_is_readable_when_the_span_allows():
    grid = Beats(120, None, 4, 0.0, 60.0)
    text = "THE CITY KEEPS DREAMING"                                             # 20 letters -> 2.0 s
    assert spec.min_hold(dict(text=text)) == pytest.approx(0.6 + 0.07 * 20)
    assert spec.min_hold(dict(text="HI")) == 1.0
    w = [dict(text=text, t0=0.0, t1=4.0, hit_on="beat", form_in="lines", form_out="lines")]
    t = spec.plan_timing(w, {"tier": "mid"}, grid)[0]
    assert t["h"] >= 2.0 - 0.5 * grid.iv, t                                        # form / exit were shrunk (default values), the hold kept
    plain = [dict(text=text, t0=0.0, t1=4.0, hit_on="beat", form_s=1.4, exit_s=1.0, form_in="lines", form_out="lines")]
    t2 = spec.plan_timing(plain, {"tier": "mid"}, grid)[0]                          # explicit form / exit are the caller's: untouched
    assert t2["f"] == pytest.approx(1.5, abs=0.3) and t2["h"] < t["h"]


def test_planner_gives_each_line_a_readable_hold():
    for bpm, length in ((100, 12), (122, 16), (110, 12), (72, 24)):
        m = synth_music(bpm, length)
        piece = pr.plan_piece(_req(), m)
        d = pr.dry_run(piece)
        assert d["notes"] == []
        for w in d["words"]:
            need = spec.min_hold(dict(text=w["text"]))
            assert w["release"] - w["hit"] >= need - 1.5 * 60.0 / bpm, (bpm, length, w["text"], w["hit"], w["release"], need)


def test_calm_plan_values():
    assert pr._lead_beats("calm") == 4
    m = synth_music(72, 24)
    p = pr.plan_piece(_req(length=24, shots="single"), m, dry=False)        # `single`: the beat-grid word scheduler (a multi-shot piece times one lockup per shot, tests/test_owner.py)
    assert p["plan"]["tier"] == "calm"
    assert all(w["exit_s"] == 0.7 for w in p["type"]["words"])          # 72 bpm = slow: overlap mode (exit 0.7 s, was 1.4 s with beat-snapped phases)
    assert p["type"]["tier"] == "calm" and p["type"]["mood"]


# ----------------------------------------------------------------------------------------------- 2. no caption
def test_signoff_is_off_by_default_and_keeps_its_timing():
    m = synth_music(100, 12)
    p = pr.plan_piece(_req(), m)
    assert p["signoff"]["off"] is True and p["signoff"]["t0"] > 0 and p["signoff"]["text"]
    q = pr.plan_piece(_req(signoff=True), m)
    assert q["signoff"]["off"] is False and q["signoff"]["t0"] == p["signoff"]["t0"]
    assert pr.normalize_request({"words": "A"})["signoff"] is False


def test_runner_skips_a_signoff_that_is_off():
    from tools.fusion import piece_runner as PR
    src = open(PR.__file__, encoding="utf-8").read()
    assert 'not self.j["signoff"].get("off")' in src


# ----------------------------------------------------------------------------------------------- 4. on the beat
def test_timeline_keys_snap_to_beats_in_order():
    grid = Beats(100, None, 4, 0.2, 30.0)
    obj = {"a": [{"args": {"t": {"timeline": [[0.0, 0.0], [0.62, 0.5], [0.66, 0.7], [3.1, 1.0]]}}}]}
    out = pr._snap_timelines(obj, grid)["a"][0]["args"]["t"]["timeline"]
    ts = [k[0] for k in out]
    assert ts[0] == 0.0 and all(b > a for a, b in zip(ts, ts[1:]))
    assert all(min(abs(t - b) for b in grid.t) < 2e-3 for t in ts[1:])
    assert [k[1] for k in out] == [0.0, 0.5, 0.7, 1.0]


@pytest.mark.parametrize("tech", ["behind", "behind_sheer", "traced", "built_in"])
def test_recipe_pulses_land_on_beats(tech):
    m = synth_music(100, 12)
    m["beats"] = [b + 0.013 for b in m["beats"]]                               # off the multiples of the interval: snapping must move keys
    grid = pr.make_grid(m, m["cut"])
    pal = ["#101418", "#eef3f5", "#37c6d4", "#f2a65a", "#aa5522"]
    comp = pr.apply_recipe(pr.load_recipe(tech), m["cut"], pal, [540, 960], grid)
    plain = pr.apply_recipe(pr.load_recipe(tech), m["cut"], pal, [540, 960])
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "timeline":
                    found.append(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(comp)
    n_plain = []
    walk_plain = found[:]
    found.clear()
    walk(plain)
    n_plain = list(found)
    found[:] = walk_plain
    for keys in found:
        ts = [k[0] for k in keys]
        assert all(b > a for a, b in zip(ts, ts[1:])), keys
        assert all(t == 0 or min(abs(t - b) for b in m["beats"]) < 2e-3 for t in ts), keys
    if found:
        assert len(found) == len(n_plain)
        assert found != n_plain                                                  # the plain scaling was not on the beats


def test_cuts_land_on_downbeats():
    for bpm, length in ((72, 12), (100, 12), (122, 16), (90, 24)):
        m = synth_music(bpm, length)
        spans = pr.cut_spans(m["nframes"], Beats(m["bpm"], m["beats"], 4, 0.0, m["cut"] + 4, m["downbeats"]))
        assert sum(spans) == m["nframes"] and all(24 <= s <= 193 for s in spans)
        c, down = 0, {int(round(d * 24)) for d in m["downbeats"]}
        for s in spans[:-1]:
            c += s
            assert c in down, (bpm, length, c)


# ----------------------------------------------------------------------------------------------- 5. hard cuts, no chaining
def test_default_is_independent_shots_with_hard_cuts():
    m = synth_music(100, 16)
    p = pr.plan_piece(_req(length=16), m, dry=False)
    assert p["plan"]["shots"] == "cut" and p["compose"]["crossfade"] == 0
    assert len(p["shots"]) >= 4
    assert [s["first"] for s in p["shots"]] == list("ABCDEF")[:len(p["shots"])]
    assert len({st["prompt"] for st in p["stills"]}) == len(p["stills"]) == len(p["shots"])
    assert len({st["seed"] for st in p["stills"]}) == len(p["stills"])
    assert all(v["xf"] == 0 for v in p["layout"].values())
    assert sum(v["frames"] for v in p["layout"].values()) == m["nframes"]
    for s in p["shots"]:
        assert s["frames"] % 8 == 1 and s["frames"] >= (s.get("span") or s["frames"])


def test_one_camera_move_per_shot_in_cut_mode():
    m = synth_music(100, 16)
    for mode in ("cut", "single"):
        p = pr.plan_piece(_req(length=16, shots=mode, legacy_chain=True), m, dry=False)
        for s in p["shots"]:
            assert "one " in s["prompt"] and "continuous" in s["prompt"] and s["prompt"].count("camera") == 1, s["prompt"]
    chain = pr.plan_piece(_req(length=16, shots="chain", legacy_chain=True), m, dry=False)
    assert [s["first"] for s in chain["shots"]] == ["A", "last:s1", "last:s2", "last:s3"][:len(chain["shots"])]
    assert chain["compose"]["crossfade"] == 0               # hard cuts on downbeats, no blended generations


def test_single_shot_mode():
    short = synth_music(120, 8, first=0.0)                             # cut at exactly 8.0 s = 192 frames
    p = pr.plan_piece(_req(length=8, shots="single"), short, dry=False)
    assert len(p["shots"]) == 1 and p["shots"][0]["frames"] == pr.snap8(short["nframes"]) <= 193
    assert "long" not in p["shots"][0] and p["layout"]["s1"] == {"start": 0, "frames": short["nframes"], "xf": 0}
    longer = synth_music(120, 16)
    q = pr.plan_piece(_req(length=16, shots="single"), longer, dry=False)
    assert len(q["shots"]) == 1 and q["shots"][0]["long"] is True and q["shots"][0]["frames"] == pr.snap8(longer["nframes"]) > 193
    with pytest.raises(ValueError):
        pr.normalize_request({"words": "A", "shots": "morph"})


# ----------------------------------------------------------------------------------------------- 6. transition variety
def test_transition_pool_is_every_registered_non_cut_transition():
    names = pr.transition_names()
    assert "cut" not in names and names == sorted(names)
    assert set(names) == set(core.list_modules("transition")) - {"cut"} and len(names) >= 5
    p = pr.plan_piece(_req(), synth_music(100, 12), dry=False)
    assert p["type"]["transition_names"] == names


def test_a_planned_piece_does_not_repeat_a_transition():
    from mograph.modules.engine import describe  # noqa: F401
    m = synth_music(124, 24)
    p = pr.plan_piece(_req(words="ONE / TWO / THREE / FOUR / FIVE / SIX", length=24, look="strand"), m, dry=False)
    scene = spec.build_scene(copy.deepcopy(dict(p["type"], camera=None)), size=(576, 1024))
    names = [t["module"] for t in scene.info["transitions"]]
    non_cut = [n for n in names if n != "cut"]
    assert len(non_cut) == len(set(non_cut)), names


# ----------------------------------------------------------------------------------------------- 8. mattes / 9. trails
def test_matte_words():
    f = pr.choose_technique
    assert f({}, "silk drifting in the haze") == "behind_sheer" and f({}, "a lone runner on the beach") == "behind"
    assert f({}, "lanterns over a crowd") == "behind" and f({}, "a neon street") == "built_in"
    assert f({}, "behind curtain veil lace spray") == "behind_sheer" and f({}, "rolling landscape contours") == "traced"
    assert f({"technique": "traced"}, "mist") == "traced"


def test_masktrack_rise_replaces_rain_on_people():
    from tools.fusion import piece_runner as PR
    assert "rise" in PR.MASK_STYLE_TEXT and "up" in PR.MASK_STYLE_TEXT["rise"]
    assert "down" in PR.MASK_STYLE_TEXT["rain"]
    assert PR.PERSON_RE.search("the runner dissolves") and PR.PERSON_RE.search("a woman in mist")
    assert not PR.PERSON_RE.search("a mountain valley at dawn") and not PR.PERSON_RE.search("a mansion")
    src = open(PR.__file__, encoding="utf-8").read()
    assert 'spec.get("style", "rise")' in src



def test_preview_tier_sizes():
    for fmt, sw in (("9:16", lambda v: v), ("16:9", lambda v: v[::-1])):
        s = pr.sizes_for("preview", fmt)
        assert s["control"] == s["shot"] == s["type"] == sw([384, 704]) and s["compose"] == sw([360, 640]), s
        assert all(v % 64 == 0 for v in s["control"] + s["shot"] + s["type"])         # IC ref downscale 2 needs multiples of 64
        assert pr.orient(pr.PREVIEW_I2V, fmt) == sw([384, 672]) and all(v % 32 == 0 for v in pr.PREVIEW_I2V)
    # every layer is cover-fitted to 360x640: x0.9375 (360x660) + a centred 10 px crop top and bottom, same for footage and type
    w, h = pr.sizes_for("preview")["shot"]
    cw, ch = pr.sizes_for("preview")["compose"]
    sc = max(cw / w, ch / h)
    assert sc == 0.9375 and round(w * sc) == cw and (round(h * sc) - ch) % 2 == 0
    assert pr.normalize_request({"words": "A", "quality": "preview"})["quality"] == "preview"
    assert pr.normalize_request({"words": "A", "quality": "bogus"})["quality"] == "final"
    assert pr.sizes_for("draft")["shot"] == [576, 1024] and pr.sizes_for("final")["compose"] == [1080, 1920]      # others untouched
    p = pr.plan_piece(pr.normalize_request(_req(quality="preview")), synth_music(122, 12))
    assert p["quality"] == "preview" and p["compose"]["size"] == [360, 640] and p["shots"][0]["size"] == [384, 704]
    assert all(st["size"] == [384, 704] for st in p["stills"])


def test_ltx_ic_plan_size_preview_is_single_stage():
    from tools.fusion import ltx_ic
    for kind in ("control", "motiontrack"):
        assert ltx_ic.plan_size(kind, (384, 704), False, True) == ((384, 704), False)
        assert ltx_ic.plan_size(kind, (704, 384), False, True) == ((704, 384), False)
        assert ltx_ic.plan_size(kind, (384, 672), False, True) == ((384, 704), False)      # 672 is not a multiple of 64
        assert ltx_ic.plan_size(kind, (576, 1024), True) == ((576, 1024), False)           # draft unchanged
        assert ltx_ic.plan_size(kind, (1152, 2048)) == ((1152, 2048), True)                # final unchanged
    from tools.fusion import piece_runner as PR
    with pytest.raises(ValueError):
        PR.Piece.__init__(object.__new__(PR.Piece), "x.json", draft=True, preview=True)


# ----------------------------------------------------------------------------------------------- slow tempo: overlap mode, kept lines, length, chain
def _ov_report(piece):
    music_grid = Beats(piece["type"]["bpm"], piece["type"]["beats"], 4, 0.0, piece["type"]["duration"] + 4.0, piece["type"]["downbeats"])
    return spec.plan_timing(copy.deepcopy(piece["type"]["words"]), piece["type"], music_grid), music_grid


@pytest.mark.parametrize("bpm,length", [(72, 14), (69, 14), (73, 16), (85, 16)])
def test_slow_tempo_keeps_the_requested_lines_with_readable_holds(bpm, length):
    """the user's line breaks survive at slow tempos: four requested lines stay four, every hit on the beat grid, every hold >= 1.2 s."""
    m = synth_music(bpm, length)
    p = pr.plan_piece(_req(length=length, words="the city / wakes / slowly / into gold", shots="single"), m, dry=False)       # `single`: the beat-grid scheduler keeps the lines (multi-shot: one lockup per shot)
    words = p["type"]["words"]
    assert [w["text"].lower() for w in words] == ["the city", "wakes", "slowly", "into gold"], [w["text"] for w in words]
    tm, grid = _ov_report(p)
    assert spec.overlap_on(p["type"], grid)
    for w, t in zip(words, tm):
        assert t["rel"] - t["hit"] >= spec.HOLD_FLOOR - 0.02, (w["text"], t)
        assert abs(t["hit"] - grid.snap(t["hit"])) <= 1.0 / 24, (w["text"], t)          # lands on a beat (the half-bar hits of slow tempos)
    assert tm[-1]["end"] <= p["audio"]["cut"] + 0.05


def test_overlap_mode_only_at_slow_tempo_and_never_below_the_readable_hold():
    slow, fast = Beats(73.0, None, 4, 0.0, 30.0), Beats(120.0, None, 4, 0.0, 30.0)
    assert spec.overlap_on({}, slow) and not spec.overlap_on({}, fast)
    assert spec.overlap_on({"overlap": True}, fast) and not spec.overlap_on({"overlap": False}, slow)
    words = [dict(text="ONE", t0=0.0, t1=2.2, form_s=0.6, exit_s=0.5, hit_on="beat"),
             dict(text="TWO", t0=2.6, t1=5.0, form_s=0.6, exit_s=0.5, hit_on="beat")]
    tm = spec.plan_timing(words, {"bpm": 73.0}, slow)
    assert all(t["rel"] - t["hit"] >= spec.HOLD_FLOOR - 1e-6 for t in tm)


def test_overlap_gap_is_stated_not_the_calm_default():
    m = synth_music(72, 16)
    p = pr.plan_piece(_req(length=16, words="one two / three four / five six"), m, dry=False)
    ws = p["type"]["words"]
    for a, b in zip(ws, ws[1:]):
        assert b["t0"] - a["t1"] < 1.0, (a, b)                       # not the 4-beat (3.2 s) calm gap


def test_requested_length_is_honoured_not_snapped_to_8_12_16_24():
    assert pr.normalize_request(_req(length=14))["length"] == 14
    assert pr.normalize_request(_req(length=10))["length"] == 10
    assert pr.normalize_request(_req(length=3))["length"] == pr.MIN_LENGTH
    assert pr.normalize_request(_req(length=500))["length"] == pr.MAX_LENGTH


@pytest.mark.parametrize("bpm", [72, 122])
def test_chain_is_hard_cuts_on_downbeats_never_a_crossfade(bpm):
    m = synth_music(bpm, 16)
    p = pr.plan_piece(_req(length=16, shots="chain", legacy_chain=True), m, dry=False)
    assert p["plan"]["shots"] == "chain" and p["compose"]["crossfade"] == 0
    assert [s["first"] for s in p["shots"]] == ["A"] + [f"last:s{k}" for k in range(1, len(p["shots"]))]      # continuity: starts from the last frame
    down = {int(round(d * 24)) for d in p["type"]["downbeats"]}
    for sid, L in p["layout"].items():
        assert L["xf"] == 0
        if L["start"]:
            assert L["start"] in down, (sid, L)
