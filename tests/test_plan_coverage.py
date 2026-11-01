"""Batch coverage + hold layouts (user 2026-10-26: "only 2 of the 24 word modules are used in any piece, we should be using it all"; "use the layouts
that we worked on"): a batch of pieces together uses every word module / transition / io style / camera move / hold layout, with the rules of the
notes inside each piece; layouts are valid, their backing shapes are a compose op; contour keys are planned.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_plan_coverage.py -q   (CPU only, no frames)"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from mograph import fonts_liked as fl  # noqa: E402
from tools.fusion import batch_plan as bp  # noqa: E402
from tools.fusion import compose  # noqa: E402
from tools.fusion import hold_layouts as hl  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import spec  # noqa: E402
from test_fusion_core import synth_music, _req  # noqa: E402

SIZES = [6, 5, 7, 6, 6]                                   # 5 pieces of 5-7 lines
TEXTS = [["AFTER HOURS", "THE CITY", "KEEPS", "DREAMING", "IN SILVER LIGHT", "UNTIL MORNING"],
         ["SALT", "AND SILVER", "THE TIDE", "REMEMBERS", "EVERY NAME"],
         ["WILD", "FIRE", "RUNS THROUGH", "THE NIGHT", "SKY", "BURNS", "OUT"],
         ["SLOW DOWN", "BREATHE", "THE ROOM", "IS WARM", "STAY", "HERE"],
         ["NEON", "RAIN", "ON EMPTY", "STREETS", "TONIGHT", "ALONE"]]


@pytest.fixture(scope="module")
def batch():
    return bp.assign_batch(SIZES, seed=3)


def _all_lines(a):
    return [(p, ln) for p in a["pieces"] for ln in p["lines"]]


# ----------------------------------------------------------------------------------------------- coverage
def test_batch_of_five_covers_every_module_transition_io_camera_layout(batch):
    c = batch["coverage"]
    for k in ("modules", "transitions", "io_styles", "cameras", "layouts"):
        assert c[k]["missing"] == [], (k, c[k]["missing"])
    assert c["modules"]["universe"] == 22 and c["modules"]["used"] == 22                      # 2026-11-02: 24 registered, `particles` (2026-11-01) and `outflow` (2026-11-02) retired (design_defaults.BANNED_MODULES)
    assert c["transitions"]["universe"] == 7 and c["io_styles"]["universe"] == 3 and c["cameras"]["universe"] == 9      # io: lines / flicker / wipe (`particles` + `blobs` banned, `outline` is the transition hand-off)
    assert c["complete"] and c["ok"], bp.format_report(c)
    assert set(c["matte_modes"]["counts"]) == set(bp.matte_modes())


@pytest.mark.parametrize("seed", range(8))
def test_coverage_holds_for_other_seeds_and_sizes(seed):
    rnd = random.Random(seed)
    a = bp.assign_batch([rnd.randint(5, 7) for _ in range(5)], seed=seed)
    assert a["coverage"]["ok"], bp.format_report(a["coverage"])


def test_registered_universe_is_what_the_user_listed():
    from mograph.modules.core import list_modules
    assert len(list_modules("word")) == 24 and len(bp.word_modules()) == 22 and set(bp.word_modules()) == set(list_modules("word")) - {"particles", "outflow"}      # the registry still has the 24; the batch draws 22 (no dot-cloud letter, no outflow)
    assert set(pr.transition_names()) == {"twist", "braid", "fan", "spiral_around", "circuit_board", "ripple"}
    assert set(bp.cameras()) == {"follow", "pullback_reveal", "hold_drift", "orbit_even", "spin_read_spin", "push_through", "tilt_to_plan",
                                 "graze", "ease_reversal"}
    assert set(bp.io_styles()) == {"lines", "flicker", "wipe"}                                  # 2026-11-01: `particles` + `blobs` banned (design_defaults.SPECKLE_IO); the registry also holds `outline` (library-transition hand-off, never named by a word)
    from mograph.modules import io as _io
    assert set(_io.list_styles()) >= {"lines", "blobs", "flicker", "particles", "wipe", "outline"}


def test_deterministic_and_seed_dependent():
    a, b, c = bp.assign_batch(SIZES, seed=3), bp.assign_batch(SIZES, seed=3), bp.assign_batch(SIZES, seed=4)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert [ln["module"] for _, ln in _all_lines(a)] != [ln["module"] for _, ln in _all_lines(c)]
    assert c["coverage"]["ok"]


def test_input_forms():
    a = bp.assign_batch([3, ["ONE", "TWO", "THREE"], {"id": "X", "lines": ["A WORD", "ANOTHER"], "format": "16:9"}], seed=1)
    assert [p["n_lines"] for p in a["pieces"]] == [3, 3, 2] and a["pieces"][2]["id"] == "X" and a["pieces"][2]["format"] == "16:9"
    with pytest.raises(ValueError):
        bp.assign_batch([0])


# ----------------------------------------------------------------------------------------------- rules inside a piece
def test_no_repeated_module_or_transition_in_a_piece(batch):
    for p in batch["pieces"]:
        mods = [ln["module"] for ln in p["lines"]]
        assert len(mods) == len(set(mods)), (p["id"], mods)
        vis = [t for t in p["transitions"] if t != "cut"]
        assert len(vis) == len(set(vis)), (p["id"], p["transitions"])
        assert len(p["transitions"]) == p["n_lines"] - 1


def test_consecutive_lines_contrast(batch):
    for p in batch["pieces"]:
        for a, b in zip(p["lines"], p["lines"][1:]):
            assert a["family"] != b["family"], (p["id"], a["module"], b["module"])
            assert a["font"] != b["font"]
            ratio = max(a["layout_record"]["area"], b["layout_record"]["area"]) / min(a["layout_record"]["area"], b["layout_record"]["area"])
            assert fl.contrast_axes(a["role"], b["role"]) or ratio >= 1.25, (a["font"], b["font"])


def test_fonts_are_all_in_liked_json(batch):
    liked = json.load(open(os.path.join(ROOT, "mograph", "fonts", "liked.json"), encoding="utf-8"))
    files = {e["file"] for e in liked["fonts"]}
    for _, ln in _all_lines(batch):
        assert ln["font_file"] in files, ln["font"]
        assert fl.is_liked(ln["font"]) and fl.is_liked(ln["font_file"])


def test_transition_matches_the_io_styles(batch):
    nm = set(bp.no_material())
    for p in batch["pieces"]:
        for a, b in zip(p["lines"], p["lines"][1:]):
            carries = a["io_out"] not in nm and b["io_in"] not in nm
            assert (a["transition"] == "cut") == (not carries), (p["id"], a["io_out"], b["io_in"], a["transition"])
        assert p["lines"][-1]["transition"] is None
        assert sum(t == "cut" for t in p["transitions"]) <= 2


def test_cameras_and_layouts_are_paired(batch):
    for p in batch["pieces"]:
        free = 0
        for ln in p["lines"]:
            rec = ln["layout_record"]
            if rec["anchored"]:
                assert ln["camera"] == "hold_drift" and ln["camera_params"]["dolly"] == 0.0     # shapes are screen space: a held camera
            if ln["camera"] in bp.FREE_CAMERAS:
                free += 1
                assert rec["box"] is None                                                       # leaves the framed box: full-frame holds only
            if ln["layout"] == "counters_window":
                assert ln["module"] in hl.WINDOW_MODULES
        assert free <= 1
        assert not any(a["layout_record"]["anchored"] and b["layout_record"]["anchored"] for a, b in zip(p["lines"], p["lines"][1:]))


def test_layouts_follow_the_text_of_each_line():
    a = bp.assign_batch(TEXTS, seed=2)
    for p, texts in zip(a["pieces"], TEXTS):
        for ln, t in zip(p["lines"], texts):
            assert ln["text"] == t and hl.eligible(ln["layout"], t, "9:16", ln["module"]), (t, ln["layout"])
    assert a["coverage"]["layout_problems"] == []
    short = bp.assign_batch([["SKY", "OUT", "SEA", "ONE", "RUN"]] * 2, seed=1)                  # three-letter lines never get small-only layouts
    for _, ln in _all_lines(short):
        assert ln["layout"] in {n for n in hl.NAMES if hl.LAYOUTS[n]["min_letters"] <= 3}


def test_landscape_pieces_get_only_landscape_layouts():
    a = bp.assign_batch([{"lines": ["ONE LINE", "TWO LINES", "THREE", "FOUR WORDS", "FIVE"], "format": "16:9"}] * 2, seed=5)
    for _, ln in _all_lines(a):
        assert "16:9" in hl.LAYOUTS[ln["layout"]]["fmts"] and ln["layout"] != "spine"
        assert hl.validate(ln["layout_record"], "16:9") == []


# ----------------------------------------------------------------------------------------------- hold layouts
@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
def test_every_layout_is_valid_and_simple(fmt):
    seen = set()
    for name in hl.names(fmt):
        for seed in range(12):
            r = hl.build(name, seed, "t", fmt)
            assert hl.validate(r, fmt) == [], (name, seed, hl.validate(r, fmt))
            assert len(r["shapes"]) <= hl.MAX_SHAPES and not any("text" in s for s in r["shapes"])
            assert json.loads(json.dumps(r)) == r                                              # JSON-able (stored on the word)
            seen.add((name, r["variant"]))
    assert {n for n, _ in seen} == set(hl.names(fmt))
    assert {v for n, v in seen if n == "solid_strips"} == {"flat", "angled", "wonky"}            # angled and slightly wonky strips both exist


def test_the_notes_layouts_exist():
    for n in ("quadrant_stack", "edge_L", "spine", "hung_from_top", "massive_crop", "solid_strips", "split_sides", "counters_window",
              "type_circle", "construction_triangle", "cells", "small_line", "full"):
        assert n in hl.LAYOUTS
    r = hl.build("massive_crop", 1, "t")
    assert r["box"]["sw"] > 1.0                                                                  # cropped by the frame
    assert hl.build("solid_strips", 0, "x")["anchored"] and not hl.build("quadrant_stack", 0, "x")["anchored"]
    ang = [hl.build("solid_strips", s, "t") for s in range(30)]
    assert any(abs(r["roll"]) >= 7 for r in ang) and any(r["variant"] == "wonky" and r["roll"] == 0 and 1.5 < abs(r["shapes"][0]["angle"]) < 4 for r in ang)


def test_camera_fill_frames_the_box_at_its_size():
    from mograph.render import cam3d
    for fmt, world in (("9:16", (540, 960)), ("16:9", (960, 540))):
        prev = cam3d.set_world(*world)
        try:
            W, H = world
            for sw, sh in ((0.46, 0.4), (0.9, 0.186), (1.34, 0.62), (0.3, 0.92)):
                fill = hl.camera_fill(sw, sh, fmt)
                dist = 1000.0 * max(sw * W / (fill * cam3d.FIT_FILLK * W), sh * H / (fill * cam3d.FIT_FILLK * cam3d.FIT_HK * H))
                assert sw * W * 1000.0 / dist / W == pytest.approx(sw, rel=1e-3) or sh * H * 1000.0 / dist / H == pytest.approx(sh, rel=1e-3)
        finally:
            cam3d.set_world(*prev)


def test_word_placed_by_a_layout_lands_on_its_box():
    """the type stage: placement wf / hf + screen_offset + hold_drift(comp 0) put the box centre and size where the layout says."""
    music = synth_music(100, 14)
    rec = hl.build("quadrant_stack", 3, "0", "9:16")
    w = dict(text="THE EDGE OF", font="Big Shoulders Display", module="lanes", t0=0.0, t1=5.0, hit_on="bar", form_in="lines", form_out="lines",
             placement=dict(yaw=0.0, **hl.placement(rec)), screen_offset=rec["offset"], ar_min=0.85)
    ts = dict(words=[w], duration=8.0, size=[576, 1024], bpm=100, beats=music["beats"], downbeats=music["downbeats"], tier="mid", camera="auto",
              camera_override={"word:0": dict(module="hold_drift", **hl.camera_params(rec, "hold_drift"))})
    sc = spec.build_scene(ts, size=(576, 1024))
    mod = next(it for it, a, b in sc.seq.timeline if hasattr(it, "placement"))
    info = sc.info["words"][0]
    cam = sc.chain.at(0.5 * (info["hit"] + info["release"]))
    pl, c = mod.placement, mod.placement.origin
    xy, _ = cam.project(np.array([c, c + pl.u * pl.w / 2, c + pl.v * pl.h / 2]))
    W, H = sc.world
    assert xy[0, 0] / W == pytest.approx(rec["box"]["cx"], abs=0.02) and xy[0, 1] / H == pytest.approx(rec["box"]["cy"], abs=0.02)
    assert 2 * (xy[1, 0] - xy[0, 0]) / W == pytest.approx(rec["box"]["sw"], rel=0.12)


# ----------------------------------------------------------------------------------------------- the compose op for backing shapes
def _plate(h=96, w=54, v=0.5):
    return np.full((h, w, 3), v, np.float32)


def test_layout_shapes_op_is_registered_and_draws_in_its_window():
    assert "layout_shapes" in compose._ops_registry()
    sh = [{"kind": "rect", "cx": 0.5, "cy": 0.5, "w": 2.0, "h": 0.25, "angle": 0.0, "color": [0.1, 0.0, 0.0], "t0": 1.0, "t1": 3.0}]
    f = compose.layout_shapes
    img = _plate()
    assert np.array_equal(f(img, sh, t=0.5), img) and np.array_equal(f(img, sh, t=3.5), img)      # outside the window: untouched
    mid = f(img, sh, t=2.0)
    assert np.allclose(mid[48, 27], [0.1, 0.0, 0.0], atol=1e-4) and np.allclose(mid[5, 27], 0.5)  # strip in the middle only
    half = f(img, sh, t=1.125, fade=0.25)
    assert 0.1 < half[48, 27, 0] < 0.5                                                           # fading in
    kept = f(img, sh, t=2.0, matte=np.where(np.arange(54)[None, :] < 27, 1.0, 0.0).repeat(96, 0).astype(np.float32))
    assert np.allclose(kept[48, 5], 0.5) and np.allclose(kept[48, 45], [0.1, 0.0, 0.0], atol=1e-4)   # the subject stays in front


def test_layout_shapes_angle_is_clockwise_and_poly_circle_work():
    f = compose.layout_shapes
    rect = {"kind": "rect", "cx": 0.5, "cy": 0.5, "w": 1.6, "h": 0.1, "angle": 20.0, "color": [1, 1, 1], "t0": 0, "t1": 9}
    out = f(np.zeros((100, 100, 3), np.float32), [rect], t=4.0)
    assert out[65, 90, 0] > 0.9 and out[35, 90, 0] < 0.1 and out[35, 10, 0] > 0.9                # right end lower, left end higher: clockwise
    poly = {"kind": "poly", "pts": [[0, 0], [1, 0], [0, 1]], "color": [0, 1, 0], "t0": 0, "t1": 9}
    circ = {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.3, "color": [0, 0, 1], "t0": 0, "t1": 9}
    o2 = f(np.zeros((80, 80, 3), np.float32), [poly], t=4.0)
    assert o2[5, 5, 1] > 0.9 and o2[75, 75, 1] < 0.1
    o3 = f(np.zeros((80, 80, 3), np.float32), [circ], t=4.0)
    assert o3[40, 40, 2] > 0.9 and o3[2, 2, 2] < 0.1 and abs(float((o3[..., 2] > 0.5).sum()) - np.pi * 24 ** 2) < 130


def test_shape_colours_stay_dark_so_the_bright_type_reads():
    for pal in ({"bg": [16, 20, 24], "fg": [238, 243, 245], "accents": [[55, 198, 212], [242, 166, 90], [170, 85, 34]]},
                {"bg": [240, 240, 240], "fg": [10, 10, 10], "accents": [[250, 250, 80]]}):
        cols = {r: pr.shape_color(r, pal) for r in hl.COLOR_ROLES}
        for r, c in cols.items():
            assert pr._lum(c) <= pr.SHAPE_LUM[r] + 1e-3 and all(0 <= v <= 1 for v in c), (r, c)
        assert max(abs(a - b) for a, b in zip(cols["dark"], cols["ink2"])) >= 0.05


# ----------------------------------------------------------------------------------------------- applied to pieces
@pytest.fixture(scope="module")
def planned():
    reqs = [_req(words="AFTER HOURS / THE CITY / KEEPS DREAMING / IN SILVER LIGHT / UNTIL MORNING / COMES HOME", length=24, seed=11,
                 scene="a lone figure standing in tall wheat"),
            _req(words="SALT / AND / SILVER / THE TIDE / REMEMBERS / EVERY NAME", length=24, seed=12, scene="a quiet misty coastline at dawn")]
    ms = [synth_music(100, 24), synth_music(96, 24)]
    pieces, asg = bp.plan_batch(reqs, ms, seed=7, dry=False)
    return pieces, asg, ms


def test_plan_piece_applies_the_assignment(planned):
    pieces, asg, _ = planned
    for p, a in zip(pieces, asg["pieces"]):
        words = p["type"]["words"]
        assert [w["text"] for w in words] == [ln["text"] for ln in a["lines"]] or len(words) != a["n_lines"]
        for i, (w, ln) in enumerate(zip(words, a["lines"])):
            moved = any(n.startswith(f"type.words[{i}] ") and f"module {ln['module']} -> {w['module']}" in n for n in p["plan"].get("module_legibility", []))      # 2026-11-01: a cube-lattice word that would merge into a block is moved to a readable module (owner.fix_illegible_modules)
            assert (w["module"] == ln["module"] or moved) and w["font"] == ln["font_file"] and (w["form_in"], w["form_out"]) == (ln["io_in"], ln["io_out"])
            assert (w["hold_layout"]["name"] == ln["layout"] or w["hold_layout"]["name"] == "full") and fl.is_liked(w["font"])         # a layout whose line fit needed a word cut across lines is not eligible any more (round-3 QC): the plan falls back to the full layout
            if w["hold_layout"]["box"]:
                assert w["placement"]["wf"] == w["hold_layout"]["box"]["sw"] and w["screen_offset"] == w["hold_layout"]["offset"]
        ts = p["type"]
        assert ts["transitions"][:len(words) - 1] == [ln["transition"] for ln in a["lines"][:len(words) - 1]]
        assert ts["camera"] == "auto" and set(ts["camera_override"]) == {f"word:{i}" for i in range(len(words))}
        assert p["plan"]["layouts"] and p["plan"]["assignment"][0]["module"] == a["lines"][0]["module"]
        json.dumps(p)                                                                            # the whole piece stays JSON


def test_planned_type_spec_builds_without_notes(planned):
    pieces, _, _ = planned
    d = pr.dry_run(pieces[0])
    assert len(d["words"]) == len(pieces[0]["type"]["words"])


def test_shapes_are_a_compose_op_in_front_of_the_plate_users(planned):
    pieces, _, _ = planned
    n_shapes = 0
    for p in pieces:
        info = p["plan"]["layout_shapes"]
        chains = [p["compose"]["default"]] + list((p["compose"].get("shots") or {}).values())
        for ch in chains:
            names = [o["op"] for o in ch["ops"]]
            if not info.get("shapes"):
                assert "layout_shapes" not in names
                continue
            assert set(names) <= set(compose._ops_registry()) | {"trace_edges"}
            k = names.index("layout_shapes")
            op = ch["ops"][k]
            assert op["out"] == "plate" and op["args"]["t"] == "$t" and op["args"]["shapes"]
            later = json.dumps([o["args"] for o in ch["ops"][k + 1:]])
            assert '"$plate"' in later                                                           # something reads the shape-painted plate
            if p["technique"] in pr._BEHINDS:
                assert op["args"]["matte"] == "$m" and names[k - 1] == "upscale_matte"
            for sh in op["args"]["shapes"]:
                assert sh["t1"] > sh["t0"] and all(0.0 <= v <= 1.0 for v in sh["color"]) and pr._lum(sh["color"]) <= 0.43
                n_shapes += 1
        if info.get("shapes"):                                                                   # the recipe still resolves for the runner / nodes
            rec = RE.build_recipe(p, "s1", type_dir=os.path.join(ROOT, "nonexistent"), curves=synth_music(100, 12)["curves"])
            assert "layout_shapes" in [o["op"] for o in rec["ops"]] and "flow_wake" in [o["op"] for o in rec["ops"]]
    assert n_shapes > 0


def test_flow_wake_still_reads_the_raw_type_with_shapes_woven(planned):
    pieces, _, _ = planned
    for p in pieces:
        for ch in [p["compose"]["default"]] + list((p["compose"].get("shots") or {}).values()):
            names = [o["op"] for o in ch["ops"]]
            assert "flow_wake" in names                                                         # flow in every generation, shapes or not
            wake = [o for o in ch["ops"] if o["op"] == "flow_wake"]
            assert all(o["args"]["type_a"] in ("$type_matte", "$type_line") for o in wake)       # the wake reads the raw type, shapes come later


def test_contour_keys_are_planned_for_pieces_with_a_subject(planned):
    pieces, asg, _ = planned
    for p, a in zip(pieces, asg["pieces"]):
        c = p["plan"]["contour"]
        assert c["available"] is bp.contour_ports_available()
        if not p["plan"]["contour"]["subject"]:
            assert c["lines"] == []
            continue
        assert len(c["lines"]) == len(p["type"]["words"])
        ports, mattes = bp.piece_ports(p)
        for k, ln in enumerate(c["lines"]):
            assert ln["entry"]["port"] in ports and ln["exit"]["port"] in ports and ln["matte"] in mattes and ln["entry"]["dir"] in bp.contour_dirs()
            if k:
                assert ln["entry"]["port"] != c["lines"][k - 1]["entry"]["port"] or len(ports) == 1                 # varied line to line
        if c["available"]:
            assert p["type"]["words"][0]["entry"] == c["lines"][0]["entry"]
    assert any(p["plan"]["contour"]["subject"] for p in pieces)


def test_plain_plan_piece_is_unchanged_without_an_assignment():
    p = pr.plan_piece(_req(), synth_music(100, 12), dry=False)
    assert "camera_override" not in p["type"] and "assignment" not in p["plan"] and "layout_shapes" in p["plan"]
    assert all("hold_layout" not in w and "screen_offset" not in w for w in p["type"]["words"])
    assert p["type"]["camera"] == "beat_orbit" and p["type"]["transitions"] == "auto"


def test_dry_run_ignores_contour_dict_keys():
    p = pr.plan_piece(_req(), synth_music(100, 12), dry=False)
    q = copy.deepcopy(p)
    for w in q["type"]["words"]:
        w["entry"], w["exit"], w["via"], w["matte"] = {"port": "matte_outline", "dir": "up"}, {"port": "depth_iso", "dir": "up"}, {"port": "depth_iso"}, "front"
    assert pr.dry_run(q)["notes"] == pr.dry_run(p)["notes"]


@pytest.mark.skipif(os.environ.get("MOGRAPH_FAST") == "1", reason="MOGRAPH_FAST=1 (full camera plan of a 24-module-style scene, ~1 min)")
def test_assigned_cameras_build_a_scene():
    music = synth_music(100, 16)
    a = bp.assign_batch([{"lines": ["ONE LINE", "TWO MORE", "THREE", "FOUR WORDS"], "id": "Z"}], seed=9)["pieces"][0]
    piece = pr.plan_piece(_req(length=16), music, dry=False, assignment=a)
    ts = copy.deepcopy(piece["type"])
    ts["words"] = ts["words"][:2]
    ts["camera_override"] = {k: v for k, v in ts["camera_override"].items() if k in ("word:0", "word:1")}
    ts["transitions"] = ts["transitions"][:1]
    sc = spec.build_scene(ts, size=(576, 1024))
    assert sc.chain is not None and sc.info["words"]
