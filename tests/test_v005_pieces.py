"""v005: contour ports / matte treatments / multiplane wired into spec + planner, and the five hand-assembled pieces P16-P20 (15_fusion/pieces/_gen_P16..20.py,
built from 15_fusion/v005/DESIGNS.md + MEDIA.md with the same machinery the planner uses).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_v005_pieces.py -q   (CPU only; MOGRAPH_FAST unset also builds the type scenes, ~1 min each)"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from mograph import fonts_liked as fl  # noqa: E402
from mograph.modules import contour_ports as CP  # noqa: E402
from tools.fusion import batch_plan as bp  # noqa: E402
from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import compose  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import spec  # noqa: E402
from tools.fusion.beats import Beats  # noqa: E402
from test_fusion_core import synth_music, _req  # noqa: E402

PIECES = ["P16", "P17", "P18", "P19", "P20"]


def _needs(rel):
    """skip when a dev fixture is absent (15_fusion/v005/ is not shipped in the public export)."""
    return pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, rel)), reason=f"dev fixture not shipped in the public export: {rel}")


def _grid_params():
    return [pytest.param(pid, marks=_needs(f"15_fusion/v005/audio/{pid}_grid.json")) for pid in PIECES]


def _plate_params():
    return [pytest.param(pid, marks=_needs(f"15_fusion/v005/out/footage/{pid}/{pid}_s1_g01.mp4")) for pid in PIECES]


FAST = os.environ.get("MOGRAPH_FAST") == "1"


def _piece(pid):
    return json.load(open(os.path.join(ROOT, "15_fusion", "pieces", f"{pid}.json"), encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- wiring: spec
def _plates(n=96):
    from test_contour_ports import synth
    return synth(n)


def _two_words(**extra):
    music = synth_music(100, 14)
    base = dict(font="Big Shoulders Display", module="lanes", t0=0.0, t1=3.0, hit_on="beat", form_in="lines", form_out="lines")
    w0 = dict(base, text="ONE", entry={"port": "matte_outline", "dir": "up"}, exit={"port": "luma_edges", "dir": "up"}, via={"port": "depth_iso", "dir": "along"})
    w1 = dict(base, text="TWO")
    kw = dict(words=[w0, w1], duration=14.0, size=[384, 704], bpm=100, beats=music["beats"], downbeats=music["downbeats"], tier="mid",
              transition_names=["twist"], camera=None)
    kw.update(extra)
    return kw


def test_dict_ports_without_plate_data_fall_back_to_module_sides():
    sc = spec.build_scene(_two_words(), size=(384, 704))                  # no `plates`: the ports are not applied, nothing crashes
    assert [w["text"] for w in sc.info["words"]] == ["ONE", "TWO"] and sc.info["notes"] == []


def test_ports_with_plate_data_wrap_the_words_and_ride_the_transition():
    sc = spec.build_scene(_two_words(plates=_plates()), size=(384, 704))
    assert any(isinstance(it, CP.PortedWord) for it, a, b in sc.seq.timeline)
    assert [t["module"] for t in sc.info["transitions"]] == ["contour_ride"], sc.info["transitions"]
    assert len(sc.info["words"]) == 2
    assert [n for n in sc.info["notes"] if "BOUND" not in n] == [] and any("contour ports BOUND" in n for n in sc.info["notes"])     # the log proof


def test_a_port_that_cannot_be_built_fails_loudly_when_plates_are_declared(monkeypatch):
    """plates declared + ports declared: a port that cannot be built is an error, never a silent fall back to module sides."""
    def boom(*a, **k):
        raise RuntimeError("no contour here")
    monkeypatch.setattr(CP, "apply_ports", boom)
    with pytest.raises(RuntimeError, match="contour ports could not be applied"):
        spec.build_scene(_two_words(plates=_plates()), size=(384, 704))


def test_missing_plate_files_fail_loudly_when_ports_are_declared(tmp_path):
    plates = [{"plate": "15_fusion/does/not/exist.mp4", "matte": str(tmp_path / "m.mp4"), "depth": "nope_depth.mp4", "frames": 24}]
    with pytest.raises(spec.PlatesMissing, match="exist.mp4"):
        spec.build_scene(_two_words(plates=plates), size=(384, 704))
    ts = _two_words(plates=plates)
    for w in ts["words"]:
        for k in ("entry", "exit", "via"):
            w.pop(k, None)
    assert len(spec.build_scene(ts, size=(384, 704)).info["words"]) == 2          # no ports declared: the plates are not needed


def test_multiplane_camera_dict_and_slots():
    ts = _two_words(camera={"chain": [{"module": "hold_drift", "word": 0}, {"module": "ease_reversal", "word": 1}], "plate": "multiplane", "cards": 4})
    ts["words"][1]["slot"] = 2
    ts["words"][0].pop("entry"), ts["words"][0].pop("exit"), ts["words"][0].pop("via")
    sc = spec.build_scene(ts, size=(384, 704))
    assert sc.chain is not None and "ease_reversal" in sc.chain.plan_text()
    z = [it.placement.origin[2] for it, a, b in sc.seq.timeline if hasattr(it, "placement")]
    assert z[0] == 0.0 and z[1] != 0.0                                                 # the slot put word 1 between two cards


# ----------------------------------------------------------------------------------------------- wiring: planner
def _bare_chain(piece, tech):
    """the plain recipe chain of ``tech`` on a planned piece (no matte treatments woven yet)."""
    q = copy.deepcopy(piece)
    t = q["type"]
    comp = pr.apply_recipe(pr.load_recipe(tech), t["duration"], ["#101418", "#eef3f5", "#37c6d4", "#f2a65a", "#aa5522"], [360, 640])
    q["compose"] = {"size": [360, 640], "crossfade": 0, **comp}
    for w in t["words"]:
        w.pop("matte", None)
    return q


def test_matte_keys_become_compose_ops_only_where_the_chain_can_run_them():
    m = synth_music(100, 12)
    p = pr.plan_piece(_req(scene="a lone figure standing in tall wheat", length=12), m, dry=False)
    assert p["plan"]["matte_treatments"]["words"] >= 1 or p["plan"]["contour"]["subject"]            # the planner's own contour keys already carry treatments
    for tech in ("behind", "built_in"):
        q = _bare_chain(p, tech)
        q["type"]["words"][0]["matte"], q["type"]["words"][-1]["matte"] = "window", "wipe"
        pr.weave_matte(q, m, None)
        ops = [o["op"] for o in q["compose"]["default"]["ops"]]
        assert set(ops) <= set(compose._ops_registry()) | {"trace_edges"}
        if tech == "behind":
            assert "matte_reach" in ops and "knockout" in ops and q["plan"]["matte_treatments"]["chains"] == 1
            assert ops[-3:] == ["grade", "grain", "vignette"] and ops.index("matte_reach") < len(ops) - 3      # before the closing grade / grain / vignette
        else:
            assert "matte_reach" not in ops and "knockout" in ops                                          # wipe needs a matte, window does not
        again = [o["op"] for o in pr.weave_matte(q, m, None)["compose"]["default"]["ops"]]
        assert again == ops                                                                                # idempotent


def test_blocks_ask_for_matte_treatments_and_ports_follow_the_piece():
    assert bk.matte_for_lines(["window_ending"], 5) == {4: "window"}
    assert bk.matte_for_lines(["beat_glow"], 3) == {0: "glow", 1: "glow", 2: "glow"}
    assert bk.matte_for_lines(["fill_silhouette_grow", "window_ending"], 4, ["window"]) == {3: "window"}      # `fill` not available: dropped
    p = pr.plan_piece(_req(scene="a lone figure standing in tall wheat"), synth_music(100, 12), dry=False)
    a = bp.assign_batch([{"id": "x", "lines": [w["text"] for w in p["type"]["words"]], "blocks": ["window_ending"], "ports": ["luma_edges"], "mattes": ["window"]}], seed=1)
    ln = a["pieces"][0]["lines"]
    assert all(l["entry"]["port"] == "luma_edges" for l in ln) and all(l["matte"] == "window" for l in ln)
    assert bp.contour_ports_available() and set(bp.contour_port_names()) == set(CP.KINDS)


def test_planner_pieces_with_ports_still_build_a_scene_without_plates():
    m = synth_music(100, 12)
    pieces, asg = bp.plan_batch([_req(scene="a lone figure standing in tall wheat")], [m], seed=3, dry=False)
    p = pieces[0]
    assert p["plan"]["contour"]["available"] and any(isinstance(w.get("entry"), dict) for w in p["type"]["words"])
    d = pr.dry_run(p)                                                                   # dict ports are stripped for the timeline check
    assert len(d["words"]) == len(p["type"]["words"])


# ----------------------------------------------------------------------------------------------- P16 - P20
@pytest.mark.parametrize("pid", _grid_params())
def test_piece_is_a_complete_v005_piece(pid):
    p = _piece(pid)
    cut = p["audio"]["cut"]
    assert p["id"] == pid and p["version"] == "v005" and p["quality"] == "preview" and p["format"] == "9:16"
    assert p["sizes"]["compose"] == [360, 640] and p["compose"]["size"] == [360, 640]
    grid = json.load(open(os.path.join(ROOT, "15_fusion", "v005", "audio", f"{pid}_grid.json"), encoding="utf-8"))
    assert cut == round(grid["cut"], 4) and p["type"]["beats"][0] == round(grid["beats"][0], 4)          # the measured grid, not the design grid
    assert sum(s["span"] for s in p["shots"]) == grid["nframes"]                                       # the shots tile the piece exactly
    for s in p["shots"]:
        assert s["frames"] >= s["span"] and (s["frames"] - 1) % 8 == 0 and s["size"] == [384, 672]
        assert all(os.path.exists(os.path.join(ROOT, v)) for v in s["media"].values()), s["media"]
    assert {s["first"] for s in p["shots"]} == {st["id"] for st in p["stills"]}
    assert all(st["size"] == [1088, 1920] and os.path.exists(os.path.join(ROOT, st["media"])) for st in p["stills"])


@pytest.mark.parametrize("pid", PIECES)
def test_piece_words_follow_the_rules(pid):
    p = _piece(pid)
    words = p["type"]["words"]
    assert 4 <= len(words) <= 6 and all(w["text"].strip() for w in words)
    mods = [w["module"] for w in words]
    assert len(mods) == len(set(mods))                                                                  # no module twice in a piece
    from mograph.modules.core import list_modules
    assert set(mods) <= set(list_modules("word"))
    for w in words:
        assert fl.is_liked(w["font"]), w["font"]                                                        # liked fonts only
        assert hl_valid(w["hold_layout"])
        for key in ("entry", "exit", "via"):
            if w.get(key):
                assert w[key]["port"] in CP.KINDS and w[key].get("dir", "up") in CP.DIRS
        if w.get("matte"):
            assert w["matte"] in CP.MATTE_MODES
    vis = [t for t in p["type"]["transitions"] if t and t != "cut"]
    assert len(vis) == len(set(vis)) and set(vis) <= set(pr.transition_names())                         # no transition repeats, all registered
    same = 0
    for a, b in zip(words, words[1:]):                                                                  # neighbours contrast in font; module family at most once
        same += bp.family_of(a["module"]) == bp.family_of(b["module"])                                   # (the designs themselves put two strand / slab modules side by side)
        assert a["font"] != b["font"]
    assert same <= 1
    t = p["plan"]["timing"]
    assert [x["text"] for x in t] == [w["text"] for w in words]
    hits = [x["hit"] for x in t]
    assert hits == sorted(hits) and hits[-1] < p["audio"]["cut"]
    assert all(x["end"] <= p["audio"]["cut"] + 0.1 for x in t[:-1])
    for b in p["plan"]["blocks"]:
        assert b["name"] in bk.BLOCKS and b["status"] == bk.BLOCKS[b["name"]]["status"]
    assert p["plan"]["deferred"] is not None


def hl_valid(rec):
    from tools.fusion import hold_layouts as hl
    return hl.validate(rec, "9:16") == []


@pytest.mark.parametrize("pid", PIECES)
def test_piece_compose_resolves_and_has_flow_matte_and_shapes_where_designed(pid):
    p = _piece(pid)
    reg = set(compose._ops_registry())
    for sid in [s["id"] for s in p["shots"]]:
        rec = RE.build_recipe(p, sid, type_dir=os.path.join(ROOT, "nonexistent"), curves={})
        names = [o["op"] for o in rec["ops"]]
        assert set(names) <= reg | {"trace_edges"}, set(names) - reg
        assert "flow_wake" in names and "upscale_matte" in names and rec["inputs"]["flow"]["kind"] == "flow"
    names = [o["op"] for o in p["compose"]["default"]["ops"]]
    treatments = {w["matte"] for w in p["type"]["words"] if w.get("matte")}
    assert p["plan"]["matte_treatments"]["words"] == sum(1 for w in p["type"]["words"] if w.get("matte"))
    if "window" in treatments:
        assert "knockout" in names
    if "wipe" in treatments:
        assert "matte_reach" in names and "matte_ring" in names
    if any(w["hold_layout"]["shapes"] for w in p["type"]["words"]):
        assert "layout_shapes" in names and p["plan"]["layout_shapes"]["shapes"] > 0
    assert p["type"]["camera"]["plate"] == "multiplane" and p["type"]["camera"]["chain"]
    assert [pl["frames"] for pl in p["type"]["plates"]] == [s["span"] for s in p["shots"]]


def test_the_five_pieces_are_built_on_different_dominant_blocks_and_cover_the_range():
    ps = [_piece(i) for i in PIECES]
    doms = [p["plan"]["blocks"][0]["name"] for p in ps]
    assert len(doms) == len(set(doms)), doms
    mods = [w["module"] for p in ps for w in p["type"]["words"]]
    assert len(set(mods)) >= 20                                                                          # the 24 modules are spread over the five
    tr = {t for p in ps for t in p["type"]["transitions"] if t}
    assert {"twist", "braid", "fan", "spiral_around", "circuit_board", "ripple", "cut"} <= tr
    assert {p["plan"]["direction"]["pace"] for p in ps} == {"slow", "mid", "driving", "building", "flowing"}
    assert {p["type"]["tier"] for p in ps} == {"calm", "mid", "kinetic"}
    cams = {c["module"] for p in ps for c in p["type"]["camera"]["chain"]}
    assert {"ease_reversal", "push_through", "follow", "graze", "spin_read_spin", "orbit_even", "tilt_to_plan", "pullback_reveal", "hold_drift"} <= cams
    ios = {s for p in ps for w in p["type"]["words"] for s in (w["form_in"], w["form_out"])}
    assert ios == {"lines", "blobs", "flicker", "particles", "wipe"}


@pytest.mark.skipif(FAST, reason="MOGRAPH_FAST=1 (type scene with contour rides, ~1 min per piece)")
@pytest.mark.parametrize("pid", _plate_params())
def test_piece_type_scene_builds_on_the_local_media(pid):
    p = _piece(pid)
    sc = spec.build_scene(dict(copy.deepcopy(p["type"]), camera=None), size=(384, 704))
    notes = [n for n in sc.info["notes"] if "BOUND" not in n]                                            # the log proof of the plates (contour ports BOUND: ...) is expected, not a problem
    assert notes in ([], ["timeline (16.04s) is longer than the clip (16.00s): the end is cut"])
    if any(w.get(k) for w in p["type"]["words"] for k in ("entry", "exit", "via")):
        assert any("contour ports BOUND" in n for n in sc.info["notes"]), sc.info["notes"]
    assert [w["text"] for w in sc.info["words"]] == [w["text"] for w in p["type"]["words"]]
    assert any(t["module"] == "contour_ride" for t in sc.info["transitions"])
    for w, x in zip(sc.info["words"], p["plan"]["timing"]):
        assert abs(w["hit"] - x["hit"]) < 0.01                                                           # the planned timing is the scene's


def test_chain_through_cards_block_gives_the_planner_a_multiplane_camera():
    m = synth_music(100, 12)
    base = pr.plan_piece(_req(scene="a lone figure standing in tall wheat"), m, dry=False)
    a = bp.assign_batch([{"id": "c", "lines": [w["text"] for w in base["type"]["words"]], "blocks": ["chain_through_cards"]}], seed=2)["pieces"][0]
    p = pr.plan_piece(_req(scene="a lone figure standing in tall wheat"), m, dry=False, assignment=a)
    cam = p["type"]["camera"]
    assert isinstance(cam, dict) and cam["plate"] == "multiplane" and 3 <= cam["cards"] <= 5 and "camera_override" not in p["type"]
    assert [c["word"] for c in cam["chain"]] == list(range(len(p["type"]["words"]))) and all(w["slot"] == 1 for w in p["type"]["words"])
    assert p["plan"]["blocks"][0]["name"] == "chain_through_cards" and p["plan"]["blocks"][0]["status"] == "partial"


# ----------------------------------------------------------------------------------------------- slow-tempo timing: every line lands on its beat
PIECES_V5 = ["P16", "P17", "P18", "P19", "P20"]


@pytest.mark.parametrize("pid", PIECES_V5)
def test_every_line_lands_on_its_design_beat_with_a_readable_hold(pid):
    """P16-P20: each word's hit is on the song's beat the design names (+-1 frame), on the DOWNBEAT where the design names a downbeat,
    and holds >= 1.2 s (overlap mode: exit / transition / form compress instead of pushing the next word a beat late)."""
    p = json.load(open(os.path.join(ROOT, "15_fusion", "pieces", f"{pid}.json"), encoding="utf-8"))
    rows = spec.landing_report(p)
    assert len(rows) == len(p["type"]["words"]) >= 4
    for r in rows:
        assert abs(r["d_wanted_f"]) <= 1.0, (pid, r)                                         # lands on the design beat
        assert abs(r["d_beat_f"]) <= 1.0, (pid, r)                                           # ... which is a beat of the measured grid
        if r["wanted_is_downbeat"]:
            assert abs(r["d_down_f"]) <= 1.0, (pid, r)                                       # a downbeat design line lands ON the downbeat
        assert r["hold"] >= spec.HOLD_FLOOR - 0.02, (pid, r)
    bad = [m for m in p["plan"]["deferred"] if m.startswith("timing:")]
    assert bad == [], bad                                                                    # nothing late, nothing short, nothing dropped


def test_p16_and_p20_are_the_slow_overlap_pieces():
    for pid, bpm in (("P16", 74.8), ("P20", 89.2)):
        p = json.load(open(os.path.join(ROOT, "15_fusion", "pieces", f"{pid}.json"), encoding="utf-8"))
        t = p["type"]
        grid = Beats(t["bpm"], t["beats"], 4, 0.0, t["duration"] + 4.0, t["downbeats"])
        assert t["bpm"] == pytest.approx(bpm, abs=0.1) and spec.overlap_on(t, grid)
        for r in spec.landing_report(p)[1:]:
            assert r["wanted_is_downbeat"] and abs(r["d_down_f"]) <= 1.0, (pid, r)          # every line after the opening one is on its downbeat


@_needs("15_fusion/v005/audio/P16_grid.json")
def test_overlap_solver_drops_a_line_instead_of_a_short_hold():
    """an impossible cycle (two hits 1 beat apart) is merged / dropped and recorded, never played with a hold < 1.2 s."""
    import importlib.util
    spc = importlib.util.spec_from_file_location("_v5lib", os.path.join(ROOT, "15_fusion", "pieces", "_v005_lib.py"))
    lib = importlib.util.module_from_spec(spc)
    spc.loader.exec_module(lib)
    g = lib.load_grid("P16")
    beats = Beats(g["bpm"], g["beats"], 4, 0.0, g["cut"] + 4.0, g["downbeats"])
    hit = lambda i: beats.t[i]
    cfgs = [dict(text="ONE", hit=hit(1)), dict(text="TWO", hit=hit(2)), dict(text="THREE", hit=hit(8))]        # TWO is one beat after ONE
    with pytest.raises(lib.Infeasible):
        lib.solve_overlap(cfgs, g, lambda t: t, 1.0, "slow")
