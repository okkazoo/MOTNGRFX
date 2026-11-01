"""Lego blocks (tools/fusion/blocks.py): the reusable passages of the v005 designs as declarative specs the planner combines freely with modules,
chains, layouts and cameras; the Qwen brief can name up to 2 of them + a direction; P21 plans from request text only.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_plan_blocks.py -q   (CPU only, no frames)"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from mograph import fusion_brief as fb  # noqa: E402
from tools.fusion import batch_plan as bp  # noqa: E402
from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import hold_layouts as hl  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from test_fusion_core import synth_music, _req  # noqa: E402

NAMED = ("strips_persist_across_cuts", "chain_through_cards", "type_from_sparks", "window_ending", "fill_silhouette_grow",
         "edge_run_small_type", "fly_through_3d_type")


# ----------------------------------------------------------------------------------------------- registry
def test_the_named_passages_are_blocks_and_more():
    assert set(NAMED) <= set(bk.NAMES) and len(bk.NAMES) >= 15
    for n in bk.NAMES:
        assert bk.validate_block(n) == [], n
        assert bk.BLOCKS[n]["status"] in bk.STATUS


def test_status_is_honest_about_what_is_missing():
    assert bk.BLOCKS["chain_through_cards"]["status"] == "partial" and any("MultiPlane" in m for m in bk.BLOCKS["chain_through_cards"]["missing"])
    assert bk.BLOCKS["cut_mid_transition"]["status"] == "spec" and bk.BLOCKS["matte_edge_reveal_wipe"]["status"] == "live"
    assert bk.BLOCKS["fly_through_3d_type"]["status"] == "live" and not bk.BLOCKS["fly_through_3d_type"].get("missing")
    assert {b["status"] for b in bk.BLOCKS.values()} == {"live", "partial", "spec"}


def test_slot_constraints_name_registered_things():
    known = bk.known_sets()
    for n, b in bk.BLOCKS.items():
        sl = b.get("slots") or {}
        ls = sl.get("lines") or {}
        for key, universe in (("module", "module"), ("layout", "layout"), ("camera", "camera"), ("io", "io")):
            for v in ls.get(key) or []:
                assert v in known[universe], (n, key, v)
        for v in sl.get("camera") or []:
            assert v in known["camera"], (n, v)
        for v in sl.get("transitions") or []:
            assert v in known["transitions"], (n, v)
        for k, v in (b.get("params") or {}).items():
            if v["kind"] == "choice" and k == "chain":
                assert set(v["choices"]) == set(bk.GALLERY_CHAINS)
    assert len(bk.GALLERY_CHAINS) == 13


def test_block_params_stay_inside_their_ranges():
    for n in bk.NAMES:
        d = bk.resolve_params(n)
        for k, spec_ in bk.BLOCKS[n].get("params", {}).items():
            assert d[k] == spec_["default"]
        for tag in ("a", "b", "c"):
            r = bk.resolve_params(n, 5, tag)
            for k, spec_ in bk.BLOCKS[n].get("params", {}).items():
                if spec_["kind"] == "choice":
                    assert r[k] in spec_["choices"]
                else:
                    assert spec_["lo"] <= r[k] <= spec_["hi"], (n, k, r[k])
                    assert isinstance(r[k], int) == (spec_["kind"] == "int")
        assert bk.resolve_params(n, 5, "a") == bk.resolve_params(n, 5, "a")
    over = bk.resolve_params("edge_run_small_type", 0, "", {"cap_frac": 5.0})
    assert over["cap_frac"] == 0.07                                                 # clipped into the range


def test_small_type_block_keeps_the_cap_height_floor():
    assert hl.MIN_CAP_FRAC == 0.04 and bk.BLOCKS["edge_run_small_type"]["params"]["cap_frac"]["lo"] >= hl.MIN_CAP_FRAC
    assert bk.BLOCKS["edge_run_small_type"]["params"]["min_px_at_640"]["lo"] == 26
    from tools.fusion import spec
    rec = hl.build("small_line", 2, "a")
    w = dict(text="SLOW DOWN", font="Big Shoulders Display", module="lanes", t0=0, t1=4, placement=dict(yaw=0, **hl.placement(rec)), ar_min=0.85,
             max_lines=rec["max_lines"])
    sc = spec.build_scene(dict(words=[w], duration=5.0, size=[576, 1024], bpm=100, camera=None), size=(576, 1024))
    assert sc.info["words"][0]["cap_px"] / sc.world[1] >= hl.MIN_CAP_FRAC


# ----------------------------------------------------------------------------------------------- eligibility and choice
def test_eligibility_follows_the_needs():
    cut = bk.piece_ctx("cut", "behind", n_lines=6)
    one = bk.piece_ctx("single", "built_in", n_lines=2, min_hold_s=1.0)
    assert bk.eligible("strips_persist_across_cuts", cut)[0] and not bk.eligible("strips_persist_across_cuts", one)[0]
    assert bk.eligible("fill_silhouette_grow", cut)[0] and not bk.eligible("fill_silhouette_grow", bk.piece_ctx("cut", "built_in"))[0]
    ok, why = bk.eligible("chain_through_cards", bk.piece_ctx("cut", "traced", n_lines=6))
    assert not ok and any("depth" in w for w in why) and bk.eligible("chain_through_cards", cut)[0]      # the cards need a depth map
    assert "matte" in cut["sources"] and "matte" not in bk.piece_ctx("cut", "built_in")["sources"] and "depth" in bk.piece_ctx("cut", "built_in")["sources"]


def test_blocks_combine_when_compatible_not_otherwise():
    assert bk.compatible("beat_glow", "type_from_sparks") and bk.compatible("window_ending", "contour_hand_off")
    assert not bk.compatible("locked_plate_move", "strips_persist_across_cuts")   # a locked-off single shot vs cuts + their cameras
    ctx = bk.piece_ctx("cut", "behind", n_lines=6)
    got = bk.choose_blocks(ctx, 3)
    assert 1 <= len(got) <= 2 and len(set(got)) == len(got)
    if len(got) == 2:
        assert bk.BLOCKS[got[0]]["role"] != bk.BLOCKS[got[1]]["role"] or bk.BLOCKS[got[0]]["role"] == "any"
        assert bk.compatible(got[0], got[1], 6)


def test_a_batch_never_shares_a_dominant_block():
    ctxs = [bk.piece_ctx("cut", t, n_lines=n, fmt="9:16") for t, n in (("behind", 6), ("built_in", 5), ("traced", 6), ("behind_sheer", 7), ("built_in", 6))]
    a = bk.batch_blocks(ctxs, 11)
    doms = [x[0] for x in a]
    assert len(doms) == len(set(doms)) == 5
    assert a == bk.batch_blocks(ctxs, 11) and a != bk.batch_blocks(ctxs, 12)
    for names, ctx in zip(a, ctxs):
        assert all(bk.eligible(n, ctx)[0] for n in names)


def test_brief_blocks_and_direction_steer_the_choice():
    ctx = bk.piece_ctx("cut", "behind", n_lines=6, min_hold_s=2.4)
    assert bk.choose_blocks(ctx, 1, {"blocks": ["window_ending"]})[0] == "window_ending"
    flat = bk.piece_ctx("cut", "built_in", n_lines=6, min_hold_s=2.4)
    assert bk.choose_blocks(flat, 1, {"blocks": ["fill_silhouette_grow"]})[0] != "fill_silhouette_grow"   # no subject matte: the brief cannot force it
    got = bk.choose_blocks(ctx, 1, {"direction": {"type_role": "small_precise"}})
    assert got[0] == "edge_run_small_type"
    assert bk.choose_blocks(ctx, 1, {"direction": {"type_role": "flat_strips"}})[0] == "strips_persist_across_cuts"


def test_line_constraints_are_what_a_block_allows():
    c = bk.line_constraints(["window_ending"], 5)
    assert c[4]["layout"] == ["counters_window", "massive_crop", "full"] and c[0]["layout"] is None and c[4]["io_side"] == "out"
    c = bk.line_constraints(["edge_run_small_type", "fly_through_3d_type"], 6)
    assert c[1]["layout"] == []                                                    # disjoint on the middle lines: compatible() keeps them apart
    assert not bk.compatible("edge_run_small_type", "fly_through_3d_type")


def test_batch_with_blocks_obeys_the_block_slots():
    a = bp.assign_batch([{"n_lines": 6, "blocks": ["edge_run_small_type", "beat_glow"]}, {"n_lines": 6, "blocks": ["window_ending"]},
                         {"n_lines": 6, "blocks": ["fly_through_3d_type"]}, {"n_lines": 6, "blocks": ["strips_persist_across_cuts"]},
                         {"n_lines": 6, "blocks": ["type_from_sparks"]}], seed=2)
    c = a["coverage"]
    assert c["blocks"]["repeated_dominant"] == [] and not [m for m in c["pairing_problems"] if "block constraint" in m], c["pairing_problems"]
    p0, p1, p2, p3, p4 = a["pieces"]
    assert [p0["lines"][i]["layout"] for i in (1, 3)] == ["small_line", "small_line"] and p0["lines"][2]["layout"] != "small_line"
    assert p1["lines"][-1]["layout"] in ("counters_window", "massive_crop", "full")
    assert all(p2["lines"][i]["camera"] in ("push_through", "follow") for i in (1, 3))
    assert all(p3["lines"][i]["layout_record"]["anchored"] and p3["lines"][i]["camera"] == "hold_drift" for i in (0, 2, 4))
    assert all(p4["lines"][i]["module"] in bk.BLOCKS["type_from_sparks"]["slots"]["lines"]["module"] for i in (1, 3))


# ----------------------------------------------------------------------------------------------- the Qwen brief
def test_direction_enumerations_match_the_registries():
    assert set(fb.DIRECTION["type_role"]) == set(bk.ROLES) - {"any"} and set(fb.DIRECTION["pace"]) == set(bk.PACES)
    assert set(fb.DIRECTION["camera"]) == set(bp.cameras()) and set(fb.DIRECTION["shots"]) == set(pr.SHOT_MODES)
    assert set(fb.PACE_TIER) == set(fb.DIRECTION["pace"]) and set(fb.PACE_TIER.values()) <= {"calm", "mid", "kinetic"}
    assert set(fb.DIRECTION_PALETTES) == set(fb.DIRECTION["palette"])
    for cols in fb.DIRECTION_PALETTES.values():
        assert len(cols) == 5 and all(len(c) == 7 and c[0] == "#" for c in cols)
    assert set(fb.block_names()) == set(bk.NAMES)


def test_prompt_lists_blocks_and_direction_keys_without_example_values():
    t = fb.brief_prompt({"words": "A / B", "length": 16}, 3)
    for n in bk.NAMES:
        assert n in t
    for k in ("blocks:", "direction:", "pace", "palette", "type_role", "camera", "shots", "music_genre", "music_lead", "bpm"):
        assert k in t
    assert fb.brief_prompt({"words": "A / B"}, 3) != fb.brief_prompt({"words": "A / B"}, 4)       # enumeration order shuffled per seed


def test_parse_brief_blocks_and_direction():
    log = []
    b = fb.parse_brief({"blocks": ["Window Ending", "strips persist across cuts", "edge_run_small_type", "nonsense"],
                        "direction": {"pace": "Rapid", "palette": "silver_bw", "type_role": "small precise", "camera": "graze", "shots": "one",
                                      "music_genre": "neo classical chamber", "music_lead": "felt piano", "bpm": 300}}, None, log)
    assert b["blocks"] == ["window_ending", "strips_persist_across_cuts"]                         # max 2, normalised
    assert any("nonsense" in m for m in log) and any("first 2" in m for m in log)
    assert b["direction"] == {"pace": "driving", "palette": "silver_bw", "type_role": "small_precise", "camera": "graze", "shots": "single",
                              "music_genre": "neo classical chamber", "music_lead": "felt piano", "bpm": 150}
    assert b["music"] == "neo classical chamber, felt piano, instrumental" and b["bpm"] == 150    # direction fills empty music / bpm
    keep = fb.parse_brief({"music": "my tags", "bpm": 70, "direction": {"bpm": 120, "music_genre": "x y"}})
    assert keep["music"] == "my tags" and keep["bpm"] == 70
    assert fb.parse_brief({"blocks": "beat glow, rise_trail", "direction": "slow"})["blocks"] == ["beat_glow", "rise_trail"]


def test_broken_or_empty_brief_is_the_rules_only_plan():
    for bad in (None, "", "no json here", "[]", {"blocks": 5, "direction": [1]}, '{"blocks": null}'):
        b = fb.parse_brief(bad)
        assert fb.brief_is_empty(b) and b["blocks"] == [] and b["direction"] == {}
    a = pr.plan_piece(_req(), synth_music(100, 12), None, dry=False)
    b = pr.plan_piece(_req(), synth_music(100, 12), "garbage", dry=False)
    assert pr.piece_hash({k: v for k, v in a.items() if k != "plan"}) == pr.piece_hash({k: v for k, v in b.items() if k != "plan"})


def test_direction_changes_the_plan_but_only_where_it_speaks():
    m = synth_music(124, 16)
    base = pr.plan_piece(_req(length=16), m, None, dry=False)
    slow = pr.plan_piece(_req(length=16), m, {"direction": {"pace": "slow", "palette": "silver_bw", "shots": "single"}}, dry=False)
    assert base["plan"]["tier"] != "calm" and slow["plan"]["tier"] == "calm"
    assert slow["plan"]["shots"] == "single" and len(slow["shots"]) == 1 and base["plan"]["shots"] == "cut"
    assert slow["type"]["palette"]["bg"] == [5, 5, 5] and base["type"]["palette"]["bg"] != [5, 5, 5]
    forced = pr.plan_piece(_req(length=16, shots="chain"), m, {"direction": {"shots": "single"}}, dry=False)
    assert forced["plan"]["shots"] == "chain"                                                  # the user's own request wins over the brief


# ----------------------------------------------------------------------------------------------- planner + P21
needs_p21 = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "15_fusion", "v005", "plan_p21.py")),
                               reason="dev fixture not shipped in the public export: 15_fusion/v005/plan_p21.py")


@pytest.fixture(scope="module")
def p21():
    spec_ = importlib.util.spec_from_file_location("plan_p21", os.path.join(ROOT, "15_fusion", "v005", "plan_p21.py"))
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


REQUEST = ("Words: AFTER HOURS / THE CITY / KEEPS DREAMING / IN SILVER LIGHT / UNTIL MORNING / COMES HOME / SLOWLY\nLength: 24\n"
           "Scene: a lone dancer under a streetlight on a wet plaza at night\nShots: cut")
BRIEF = {"scene": "a lone dancer under a streetlight on a wet empty plaza at night, mist rising from the stones, deep shadows",
         "motion": ["the dancer turns slowly, one slow camera glide forward"] * 2, "music": "slow cinematic piano, strings", "bpm": 100, "technique": "behind",
         "mood": "calm", "blocks": ["window_ending", "beat_glow"], "direction": {"pace": "mid", "type_role": "window", "camera": "orbit_even", "shots": "cut"}}


@needs_p21
def test_p21_request_text_to_prompt_to_piece(p21, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "OWNERS", False)           # the beat-grid schedule: one-lockup-per-shot makes 1-bar shots (holds ~1.2 s < window_ending's 2.0 s)
    req = p21.request_from_text(REQUEST)
    assert req["length"] == 24 and req["shots"] == "cut" and req["scene"].startswith("a lone dancer") and req["words"].count("/") == 6
    assert p21.request_from_text("SLOW DOWN / BREATHE")["words"] == "SLOW DOWN / BREATHE"
    prompt = fb.brief_prompt(req)
    assert "window_ending" in prompt and "USER WORDS" in prompt
    piece, info = p21.plan(REQUEST, json.dumps(BRIEF), music_path=None, bpm=100, dry=False)
    assert piece["id"] == "P21" and info["provisional_grid"] and info["brief_blocks"] == ["window_ending", "beat_glow"]
    pl = piece["plan"]
    assert pl["blocks"][0]["name"] == "window_ending" and pl["tier"] == "mid" and pl["direction"]["camera"] == "orbit_even"
    words = piece["type"]["words"]
    assert words[-1]["hold_layout"]["name"] in ("counters_window", "massive_crop", "full")        # the block's last-line slot
    assert [w["text"] for w in words] and all(w["font"].endswith((".ttf", ".otf")) for w in words)
    out = tmp_path / "P21.json"
    out.write_text(pr.piece_json(piece), encoding="utf-8")
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["plan"]["blocks"] == json.loads(json.dumps(pl["blocks"]))
    assert "window_ending" in p21.summary(again) and "MISSING" in p21.summary(again)
    (tmp_path / "b.json").write_text(json.dumps(BRIEF), encoding="utf-8")
    assert p21.main(["prompt", REQUEST]) == 0 and p21.main(["music", REQUEST, "--brief", str(tmp_path / "b.json")]) == 0
    assert p21.main(["show", str(out)]) == 0


@needs_p21
def test_p21_without_a_brief_still_gets_blocks_from_the_rules(p21):
    piece, info = p21.plan(REQUEST, None, bpm=100, dry=False)
    assert info["brief_blocks"] == [] and piece["plan"]["blocks"] and piece["plan"]["blocks"][0]["dominant"]
    assert all(bk.eligible(b["name"], pr.block_ctx(piece))[0] for b in piece["plan"]["blocks"])


def test_strips_persist_block_extends_the_strip_windows():
    reqs = [_req(words="AFTER HOURS / THE CITY / KEEPS DREAMING / IN SILVER LIGHT / UNTIL MORNING / COMES HOME", length=24, seed=3,
                 scene="a lone figure standing in tall wheat")]
    m = synth_music(120, 24)
    base = pr.plan_piece(reqs[0], m, dry=False)
    n = len(base["type"]["words"])
    a = bp.assign_batch([{"id": "S", "lines": [w["text"] for w in base["type"]["words"]], "blocks": ["strips_persist_across_cuts"]}], seed=4)["pieces"][0]
    withp = pr.plan_piece(reqs[0], m, dry=False, assignment=a)
    nob = dict(a, blocks=[])
    without = pr.plan_piece(reqs[0], m, dry=False, assignment=nob)
    if n >= 3 and withp["plan"]["layout_shapes"].get("shapes"):
        def t1s(p):
            return [sh["t1"] for ch in [p["compose"]["default"]] for o in ch["ops"] if o["op"] == "layout_shapes" for sh in o["args"]["shapes"]]
        assert max(t1s(withp)) >= max(t1s(without))
    assert withp["plan"]["blocks"][0]["name"] == "strips_persist_across_cuts" and withp["plan"]["blocks"][0]["status"] == "live"


# ----------------------------------------------------------------------------------------------- depth_pivot_chunks (image transition block)
def test_depth_pivot_chunks_block_is_requestable_by_name_and_partial():
    b = bk.BLOCKS["depth_pivot_chunks"]
    assert bk.validate_block("depth_pivot_chunks") == [] and b["status"] == "partial" and b["missing"]
    assert b["needs"]["opt_in"] and set(b["needs"]["shots"]) == {"cut", "chain"} and b["needs"]["min_shots"] == 2 and "matte" in b["needs"]["sources"]
    assert any("video" in x for x in b["needs"]["prefer"]) and any("matte" in x for x in b["needs"]["prefer"])      # the clip + per-frame mattes preferred
    assert b["slots"]["image_transition"]["module"] == "depth_pivot" and b["slots"]["image_transition"]["video"] == "auto"
    ctx = bk.piece_ctx("cut", "behind", n_lines=4)
    assert not bk.eligible("depth_pivot_chunks", ctx)[0]                                  # opt-in: never chosen on its own
    assert bk.eligible("depth_pivot_chunks", dict(ctx, opt_in=("depth_pivot_chunks",)))[0]
    assert not bk.eligible("depth_pivot_chunks", dict(bk.piece_ctx("single", "behind"), opt_in=("depth_pivot_chunks",)))[0]
    got = bk.choose_blocks(ctx, 1, {"blocks": ["depth_pivot_chunks"]})
    assert got[0] == "depth_pivot_chunks"                                                 # named by the brief / request -> planned
    d = bk.describe(["depth_pivot_chunks"], 0, 4)[0]
    assert d["status"] == "partial" and d["image_transition"]["node"] == "MOTNGRFX_DepthPivotTransition" and 10 <= d["params"]["chunks"] <= 400   # describe() seeds a point in the range; resolve_params(name) gives the default 32
    assert bk.resolve_params("depth_pivot_chunks")["chunks"] == 32


def test_depth_pivot_chunks_params_match_the_module():
    from mograph.modules.transitions import depth_pivot as dp
    for k, spec_ in bk.BLOCKS["depth_pivot_chunks"]["params"].items():
        m = dp.PARAMS[k]
        assert (spec_["default"], spec_["lo"], spec_["hi"], spec_["kind"]) == (m["default"], m["lo"], m["hi"], m["kind"]), k
    assert bk.BLOCKS["depth_pivot_chunks"]["params"]["chunks"]["lo"] == 10 and bk.BLOCKS["depth_pivot_chunks"]["params"]["chunks"]["hi"] == 400
