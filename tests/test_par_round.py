"""20_open-model-design/v003 par round: the compose building blocks (tools/fusion/compose_blocks.py), the design schema's new fields (system block, camera fast_dps, placement
pitch / dx / dy, compose.default / compose.shots op chains), the concept's shot plan -> plan_rules wiring, the bracket mender of extract_json, and the level-B piece check
(tools/fusion/piece_brief.py). Fake replies only: no model, no network, no GPU.
Run: python -m pytest tests/test_par_round.py -q"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

from test_fusion_core import synth_music  # noqa: E402
from tools.fusion import compose_blocks as CB  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

CHAIN = [{"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [1080, 1920], "guide": "$footage"}},
         {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.9, "sat": 0.0}},
         {"op": "tint_type", "out": "tcol", "args": {"type_a": "$type_matte", "footage": "$footage", "mode": "sample"}},
         {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_rgb": "$tcol", "type_a": "$type_matte", "matte": "$m", "matte_threshold": 0.5,
                                                      "matte_gain": {"timeline": [[0, 1], [3.0, 0], [6.0, 1]], "step": True}}},
         {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$type_matte", "radius": 20,
                                                      "color": {"timeline": [[0, [0.3, 0.6, 1.0]], [6.0, [1.0, 0.7, 0.3]]], "step": True}, "intensity": {"curve": "onset", "lo": 0.1, "hi": 0.5}}},
         {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.3}}]


def ops(chain=None):
    return copy.deepcopy(chain or CHAIN)


# ------------------------------------------------------------------------------------------------ compose building blocks
def test_a_valid_chain_is_kept_exactly():
    kept, fx = CB.validate_ops(ops(), 12.0)
    assert [o["op"] for o in kept] == [o["op"] for o in CHAIN]
    assert kept[3]["args"]["matte_gain"] == CHAIN[3]["args"]["matte_gain"] and fx == []


def test_buffer_names_without_dollar_and_0_255_colours_are_mended():
    c = ops()
    c[3]["args"]["type_rgb"] = "tcol"
    c[4]["args"]["color"] = [255, 180, 80]
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept[3]["args"]["type_rgb"] == "$tcol" and kept[4]["args"]["color"] == [1.0, 0.7059, 0.3137]
    assert any("written as $tcol" in f for f in fx) and any("divided by 255" in f for f in fx)


def test_unknown_op_and_unknown_argument_are_dropped_alone():
    c = ops()
    c.insert(2, {"op": "sparkle", "out": "x", "args": {}})
    c[1]["args"]["bogus"] = 3
    kept, fx = CB.validate_ops(c, 12.0)
    assert "sparkle" not in [o["op"] for o in kept] and "bogus" not in kept[1]["args"] and len(kept) == len(CHAIN)
    assert any("not an allowed building block" in f for f in fx) and any("'bogus' does not exist" in f for f in fx)


def test_a_picture_filter_that_cannot_work_is_bypassed_and_the_chain_goes_on():
    c = ops()
    c.insert(5, {"op": "screen", "out": "mix", "args": {"a": "$image", "b": "$type_matte"}})     # rgb screen gray: cannot be blended
    c[-1]["args"]["frame"] = "$mix"
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept[-1]["op"] == "vignette" and kept[-1]["args"]["frame"] == "$image"
    assert any("bypassed" in f for f in fx)


def test_kind_mismatch_in_a_drawer_gets_the_plain_default():
    c = ops()
    c.insert(3, {"op": "flow_wake", "out": "wake", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$m", "type_rgb": "$tcol"}})
    c[4]["args"]["type_rgb"] = "$wake"                                  # the alpha of flow_wake used as the colour
    kept, fx = CB.validate_ops(c, 12.0)
    sand = next(o for o in kept if o["op"] == "sandwich")
    assert sand["args"]["type_rgb"] == "$tcol" and "type_rgb" not in next(o for o in kept if o["op"] == "flow_wake")["args"]


def test_a_filter_without_its_input_works_on_the_picture_made_so_far():
    c = ops()
    del c[1]["args"]["frame"]
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept[1]["args"]["frame"] == "$footage" and any("picture made so far" in f for f in fx)


def test_a_number_where_a_mask_is_needed_is_not_a_picture():
    c = ops()
    c[3]["args"]["front"] = 1
    kept, fx = CB.validate_ops(c, 12.0)
    assert "front" not in kept[3]["args"] and any("number / timeline" in f for f in fx)


def test_a_chain_that_draws_no_type_is_rejected_and_a_last_op_is_renamed_to_image():
    kept, fx = CB.validate_ops([o for o in ops() if o["op"] not in ("sandwich",)], 12.0)
    assert kept is None and any("no op draws the type" in f for f in fx)
    c = ops()
    c[-1]["out"] = "final"
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept[-1]["out"] == "image" and any("renamed to `image`" in f for f in fx)


def test_raw_matte_gets_its_upscale_and_values_are_clamped():
    c = ops()[1:]                                               # no upscale_matte
    c[2]["args"]["matte"] = "$matte"
    c[0]["args"]["gain"] = 9.0
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept[0]["op"] == "upscale_matte" and kept[3]["args"]["matte"] == "$m" and kept[1]["args"]["gain"] == 2.5


def test_unavailable_passes_make_a_chain_fall_back():
    comp, fx = CB.normalize_compose({"default": {"ops": ops()}, "shots": {"s2": {"ops": ops()}}}, ["s1", "s2"], 12.0, {"s1": {"matte"}, "s2": set()})
    assert "default" not in comp                                 # s1 has no matte: the default chain (shared) may not read it
    assert "s2" in comp["shots"]


def test_the_lead_designers_chains_survive_untouched():
    """P06 / P10 / P11 / P14 chains (15_fusion/pieces) pass the validator op for op: it does not forbid what the lead designer writes."""
    for pid, cut in (("P06", 18.0), ("P10", 15.25), ("P11", 16.0), ("P14", 12.0)):
        f = ROOT / "15_fusion" / "pieces" / f"{pid}.json"
        if not f.is_file():
            pytest.skip("15_fusion pieces not present")
        comp = json.loads(f.read_text(encoding="utf-8"))["compose"]
        for name, ch in [("default", comp["default"])] + list((comp.get("shots") or {}).items()):
            kept, fx = CB.validate_ops(ch["ops"], cut, shot=name)
            assert kept is not None and len(kept) == len(ch["ops"]), (pid, name, fx)


def test_the_menu_lists_every_allowed_op_with_its_real_signature():
    t = CB.menu(12.0)
    for name in CB.ALLOWED:
        assert f"- {name}(" in t
    assert "sandwich(footage, type_rgb, type_a, matte" in t.replace(" ", " ")


# ------------------------------------------------------------------------------------------------ the design schema: system block, motion fields, op chains
REQ = {"words": "STILL\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING", "scene": "a lone runner in a concrete tunnel", "music": "kinetic and emotional, 160 bpm",
       "length": 12, "look": "auto", "seed": 6, "quality": "preview", "format": "9:16"}


@pytest.fixture(scope="module")
def music():
    return synth_music(160, 12)


@pytest.fixture(scope="module")
def plan(music):
    return pr.plan_piece(REQ, music)


def parse(plan, music, reply):
    return DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "PROMPT " * 200)


def design_of(plan, music):
    return DS.reply_from_piece(plan, music)


def test_system_block_fills_only_the_fields_the_words_omit():
    obj = {"type": {"words": [{"text": "A"}, {"text": "B", "font": "Anybody[wdth,wght].ttf", "placement": {"yaw": 3}}, {"text": "C"}, {"text": "D"}]}, "compose": {},
           "system": {"roles": {"A": "GeistPixel[ELSH].ttf - pixel", "B": "Doto"}, "pattern": ["A", "B"], "yaw": [-16, 16],
                      "axis": {"font": "GeistPixel[ELSH].ttf", "tag": "ELSH", "windows": [[0, 40], [30, 70]]}}}
    raw = SV.flatten(obj, ["A", "B", "C", "D"])
    assert raw["type.words[0].font"] == "GeistPixel[ELSH].ttf" and raw["type.words[2].font"] == "GeistPixel[ELSH].ttf" and raw["type.words[3].font"] == "Doto[ROND,wght].ttf"
    assert raw["type.words[1].font"] == "Anybody[wdth,wght].ttf" and raw["type.words[1].placement.yaw"] == 3          # what the word wrote is never touched
    assert raw["type.words[0].placement.yaw"] == -16 and raw["type.words[2].placement.yaw"] == -16
    assert raw["type.words[0].axes.ELSH"] == [0, 40, "beat"] and raw["type.words[2].axes.ELSH"] == [30, 70, "beat"] and "type.words[3].axes.ELSH" not in raw
    assert "type.words[1].font" not in raw["@system"] and len(raw["@system"]) >= 7


def test_motion_fields_and_op_chains_survive_the_whole_design_step(plan, music):
    r = design_of(plan, music)
    r["type"]["camera"] = "beat_orbit"
    r["type"]["camera_params"] = {"read_dps": 6, "intro_deg": 44, "fast_dps": 52, "elev": 4, "bob": 1.1}
    for i, w in enumerate(r["type"]["words"]):
        w["placement"] = {"yaw": -16 if i % 2 == 0 else 16, "dy": -0.1, "pitch": 6}
    r["system"] = {"rule": "alternate", "roles": {"A": "Anybody[wdth,wght].ttf"}, "pattern": ["A"]}
    r["compose"]["default"] = {"ops": ops()}
    out = parse(plan, music, r)
    p = json.loads(out["piece_json"])
    assert p["type"]["camera_params"]["fast_dps"] == 52.0
    assert p["type"]["words"][0]["placement"]["yaw"] == 0 and p["type"]["words"][0]["placement"]["dy"] == 0 and p["type"]["words"][1]["placement"]["pitch"] == 0       # 2026-10-30 (user): every word lands centred and still (stable_type.enforce); the model's yaw / dy / pitch are not kept
    assert p["type"]["camera_params"]["read_dps"] == 0.0 and p["type"]["camera_params"]["bob"] == 0.0
    assert p["plan"]["compose_ops"]["source"] == "model" and p["plan"]["system"]["rule"] == "alternate"
    default = [o["op"] for o in p["compose"]["default"]["ops"]]
    assert "sandwich" in default and "add_glow" in default
    tl = next(o for o in p["compose"]["default"]["ops"] if o["op"] == "add_glow")["args"]["color"]
    assert tl == "$tcol"                    # 2026-10-30: hold_life.weave_ops puts the model's glow on the type colour ($tcol) - glow exists only as hold life; the model's own colour timeline no longer survives (stale test updated, not a bug)
    assert out["fields"]["pct_model"] >= 62                      # measured 64.7 on 2026-10-30; was 80, relaxed to 60, tightened here: the stable-type rule (2026-10-30) overrides yaw / dy / pitch of every word on purpose, and those count as code fixes


def test_invalid_op_chain_leaves_the_rules_chain_in_place(plan, music):
    r = design_of(plan, music)
    r["compose"]["default"] = {"ops": [{"op": "grade", "out": "image", "args": {"frame": "$footage"}}]}       # draws no type
    p = json.loads(parse(plan, music, r)["piece_json"])
    assert p["plan"]["compose_ops"]["source"] == "rules"
    assert any(o["op"] in ("sandwich", "depth_occlude", "knockout") for o in p["compose"]["default"]["ops"])


def test_a_chain_cannot_read_a_pass_the_footage_does_not_have(plan, music):
    base = json.loads(json.dumps(plan))
    for s in base["shots"]:
        s.pop("matte", None)
    base["compose"] = pr.apply_recipe(pr.load_recipe("traced"), 12.0, ["#101010", "#f0f0f0", "#ff8800"], size=base["compose"]["size"]) | {"size": base["compose"]["size"], "crossfade": 0, "crf": 20}
    base["technique"] = "traced"
    r = DS.reply_from_piece(base, music)
    r["compose"]["default"] = {"ops": ops()}                      # reads $matte; no shot has one
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(base)), base, music, "P " * 100)
    assert json.loads(out["piece_json"])["plan"]["compose_ops"]["source"] == "rules"


def test_schema_text_documents_the_new_fields():
    t = QP.schema_text(["A", "B", "C"], 12.0)
    for needle in ('"system"', '"fast_dps"', '"pitch"', '"default": {"why"', '"ops": [{"op"', "pattern"):
        assert needle in t
    assert "yaw\": [-16" not in t                                      # placeholders, not example values (the models copy those)


# ------------------------------------------------------------------------------------------------ concept shot plan -> plan_rules
def test_concept_prompt_asks_for_idea_and_one_shot_per_lockup():
    p = CO.concept_prompt(7, 16, [])
    assert '"idea"' in p and '"shot_plan"' in p and "EXACTLY" in p


def test_concept_parse_keeps_good_shots_and_flags_bad_ones():
    good = {"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "old fisherman", "scene": "an old fisherman mending a net on a grey harbour wall at dawn, soft light, wide shot, calm water behind him",
            "music": "folk, guitar, fiddle, warm, gentle, light percussion, tender, acoustic", "bpm": 90, "mood": "tender", "look": "auto", "shots": "cut",
            "idea": "A fisherman at work; the camera backs away from him as the light grows.",
            "shot_plan": [{"prompt": "The fisherman mends a net on the harbour wall, grey light, the camera slowly backs away keeping him in frame", "camera": "backs away"},
                          {"prompt": "x"}, {"prompt": "Large text letters spelling HOME fill the sky above the boats at dawn"}]}
    c, errors, fixes = CO.normalize(good, 3, [], 16)
    assert not errors and c["idea"].startswith("A fisherman")
    from tools.fusion import shot_camera as SC
    assert SC.parse(c["shot_plan"][0]["camera"])[1] == "pull back" and "backs away" not in c["shot_plan"][0]["camera"] and len(c["shot_plan"]) == 4      # the model's move is kept, a framing was added               # one entry per shot: the bad / missing ones are PADDED from the reply's own scene text, no template prompt
    assert sum("padded from the scene text" in f for f in fixes) == 3 and not any("template is used" in f for f in fixes)      # shot 2 too short, shot 3 names text, shot 4 missing
    assert all("fisherman" in e["prompt"] for e in c["shot_plan"])
    cams = [(e.get("camera") or "") for e in c["shot_plan"]]
    assert len(set(cams)) == 4 and all(cams)                                                   # every shot has its own camera (shot_camera.enforce); the padded ones were given one
    sp, fx = CO.shot_plan_of(good["shot_plan"] * 3, 2, pad_from=good["scene"])
    assert len(sp) == 2 and any("kept" in f and "9 entries" in f for f in fx)                  # more entries than shots: trimmed, logged
    sp2, fx2 = CO.shot_plan_of(good["shot_plan"], 4)                                           # without a scene to pad from the old behaviour stays (template, trailing empties cut)
    assert len(sp2) == 1 and sum("template is used" in f for f in fx2) == 3


def test_plan_uses_the_models_shot_prompts_and_templates_for_the_rest(music):
    sp = [{"prompt": "A runner sprints down a dark tunnel, cold blue light streaking overhead, tracked alongside", "camera": "tracks alongside the runner"}, {}]
    req = dict(REQ, owners=json.dumps({"owners": [], "shot_plan": sp, "idea": "one runner, three lights"}), shots="cut")
    plan_ = pr.plan_piece(req, music)
    base = pr.plan_piece(dict(REQ, shots="cut"), music)
    assert "A runner sprints down a dark tunnel" in plan_["shots"][0]["prompt"][:120]            # the shot's framing + move LEAD (shot_camera.lead), then the model's prompt
    assert "tracks alongside the runner" in plan_["shots"][0]["prompt"] or "tracked alongside" in plan_["shots"][0]["prompt"]
    if len(plan_["shots"]) > 1:
        assert plan_["shots"][1]["prompt"].endswith(base["shots"][1]["prompt"])                # no entry: the template, behind its own (different) camera lead
        assert plan_["shots"][1]["prompt"].split(",")[0] != plan_["shots"][0]["prompt"].split(",")[0]
    assert plan_["plan"]["footage"]["idea"] == "one runner, three lights" and plan_["plan"]["footage"]["shot_prompts"]["model"] == 1


def test_without_a_shot_plan_the_plan_is_unchanged(music):
    a = pr.plan_piece(dict(REQ, shots="cut"), music)
    b = pr.plan_piece(dict(REQ, shots="cut", owners="[]"), music)
    assert [s["prompt"] for s in a["shots"]] == [s["prompt"] for s in b["shots"]] and "footage" not in a["plan"]


def test_request_extras_is_the_old_owner_list_unless_there_is_more():
    assert json.loads(CO.request_extras({"owners": [{"render": "flat"}]})) == [{"render": "flat"}]
    x = json.loads(CO.request_extras({"owners": [], "shot_plan": [{"prompt": "p"}], "idea": "i"}))
    assert x == {"owners": [], "shot_plan": [{"prompt": "p"}], "idea": "i"}
    assert pr._owners_of(json.dumps(x)) is None and pr.shot_plan_of(json.dumps(x)) == [{"prompt": "p"}]


def test_design_prompt_carries_the_footage_idea(plan, music):
    p2 = copy.deepcopy(plan)
    p2.setdefault("plan", {})["footage"] = {"idea": "one runner, three lights"}
    pk = DS.package(REQ, music, p2, [], None, None, 3)
    assert "THE FOOTAGE IDEA" in pk["prompt"] and "one runner, three lights" in pk["prompt"] and "COMPOSE BUILDING BLOCKS" in pk["prompt"]


# ------------------------------------------------------------------------------------------------ extract_json bracket mender
def test_extract_json_mends_a_typed_bracket_in_a_long_reply():
    t = '{"a": {"timeline": [[0, 1], [2, 3]]], "w": 2}, "b": {"c": [1, 2]}}'        # `]` typed for `}`
    assert QP.extract_json(t) == {"a": {"timeline": [[0, 1], [2, 3]], "w": 2}, "b": {"c": [1, 2]}}
    assert QP.extract_json('{"a": [1, 2]], "b": {"c": 1}}') == {"a": [1, 2], "b": {"c": 1}}
    assert QP.extract_json('```json\n{"a": {"b": [1, 2}, "c": 3}\n```')["a"]["b"] == [1, 2]            # a missing `]`: the data survives (where "c" lands is a guess)


# ------------------------------------------------------------------------------------------------ level B (brief -> piece)
@pytest.mark.skipif(not (ROOT / "15_fusion" / "PIECES.md").exists(), reason="dev fixture not shipped in the public export: 15_fusion/PIECES.md")
def test_piece_check_runs_the_lead_designers_p10_through_the_engine():
    from tools.fusion import piece_brief as PB
    if not (ROOT / "15_fusion" / "pieces" / "P10.json").is_file():
        pytest.skip("15_fusion pieces not present")
    b = PB.load_brief("P10")
    rep = PB.check_piece(b["claude_piece"], b["ana"])
    assert rep["type"]["ok"] and all(v["build"] and v["exec"] for v in rep["shots"].values())
    g = PB.groups(b["claude_piece"], b["ana"])
    assert g["compose"]["drawers"] >= 1 and PB.rubric(g, b["ana"])["total"] >= 5


def test_reply_to_piece_builds_a_checked_piece_from_a_small_reply():
    from tools.fusion import piece_brief as PB
    if not (ROOT / "15_fusion" / "PIECES.md").is_file():
        pytest.skip("15_fusion briefs not present")
    b = PB.load_brief("P10")
    words = [{"text": "WHO", "font": "Rock3D-Regular.ttf", "module": "lanes", "hit": 1.76, "hold_s": 1.8, "form_s": 1.0, "exit_s": 0.8, "placement": {"yaw": -14}},
             {"text": "WAS", "font": "BadeenDisplay-Regular.ttf", "module": "lanes", "hit": 7.07, "hold_s": 1.6, "form_s": 0.6, "exit_s": 0.8, "placement": {"yaw": 14}},
             {"text": "SHE", "font": "Anybody[wdth,wght].ttf", "module": "lanes", "hit": 12.38, "hold_s": 1.4, "form_s": 0.6, "exit_s": 0.8, "placement": {"yaw": -14}}]
    reply = {"idea": "a woman walks to the camera", "system": {"rule": "x"}, "stills": [{"id": "A", "prompt": "a woman in a trench coat under a lamp, film noir"}],
             "shots": [{"id": "s1", "first": "A", "until": 7.0, "prompt": "she walks toward the camera, the camera backs away"},
                       {"id": "s2", "first": "last:s1", "until": 15.25, "prompt": "she stops and looks into the lens"}],
             "type": {"mood": "noir", "energy": 0.6, "palette": {"bg": [8, 8, 8], "fg": [225, 224, 220], "accents": [[120, 120, 118]]}, "camera": "beat_orbit",
                      "camera_params": {"intro_deg": 30, "fast_dps": 40}, "transitions": ["ripple", "ripple"], "words": words},
             "compose": {"crossfade": 0, "default": {"ops": ops()}}}
    piece, fixes, errors = PB.reply_to_piece(reply, b)
    assert piece is not None and not errors, errors
    assert any("type.transitions[1]" in f for f in fixes)                       # a repeated transition becomes null
    assert piece["plan"]["compose_source"] == "model" and piece["type"]["camera_params"]["fast_dps"] == 40.0
    assert [s["id"] for s in piece["shots"]] == ["s1", "s2"] and piece["shots"][0].get("matte") is True
    rep = PB.check_piece(piece, b["ana"], exec_ops=True)
    assert rep["type"]["ok"] and all(v["exec"] for v in rep["shots"].values()), rep


def test_extract_json_closes_an_unclosed_palette_before_the_next_type_key():
    """Gemma 4: `palette` never closed, so camera / words landed inside it (the design then came from the rules for every field after the palette)."""
    t = '{"type": {"mood": "x", "palette": {"bg": [1, 2, 3], "accents": [[1, 2, 3],\n [4, 5, 6]\n ],\n "camera": "beat_orbit", "words": [{"text": "A"}]}, "compose": {"technique": "behind"}}'
    o = QP.extract_json(t)
    assert o["type"]["camera"] == "beat_orbit" and o["type"]["words"] == [{"text": "A"}] and set(o["type"]["palette"]) == {"bg", "accents"}


def test_a_dropped_picture_filter_does_not_take_the_rest_of_the_chain_with_it():
    """par round r3 P11: a crossfade whose `b` was nonsense was dropped and every later op read its output, so the whole chain was rejected."""
    c = ops()
    c.insert(2, {"op": "crossfade", "out": "bd", "args": {"a": "$foot", "b": {"fill": [0.0, 0.0, 0.0]}, "t": 0.3}})
    c[4]["args"]["footage"] = "$bd"
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept is not None and "crossfade" not in [o["op"] for o in kept]
    assert next(o for o in kept if o["op"] == "sandwich")["args"]["footage"] == "$foot"


def test_two_accents_are_padded_to_three_for_the_3d_modules():
    pal = {"bg": [10, 10, 10], "fg": [240, 240, 240], "accents": [[200, 0, 0], [0, 200, 0]]}
    QP.pad_accents(pal)
    assert len(pal["accents"]) == 3 and pal["accents"][:2] == [[200, 0, 0], [0, 200, 0]] and pal["accents"][2] == [240, 240, 240]


# ------------------------------------------------------------------------------------------------ shot structure: places of two shots, continuation, no collapse to single
def test_shot_plan_continues_flags_make_places_of_two_shots(music):
    sp = [{"prompt": "A runner sprints along a dark tunnel toward the light, tracked alongside"},
          {"prompt": "The runner keeps sprinting along the tunnel, the camera tracks alongside him", "continues": True},
          {"prompt": "The runner bursts onto a dawn street, the camera tracks alongside him", "continues": False},
          {"prompt": "The runner keeps running down the street, tracked alongside", "continues": True}]
    req = dict(REQ, length=16, shots="cut", owners={"owners": [], "shot_plan": sp, "idea": "one runner, two places"})
    plan_ = pr.plan_piece(req, synth_music(160, 16))
    firsts = [s["first"] for s in plan_["shots"]]
    # 2026-10-29: `continues` = the same place from another camera: its own ANGLE still (of the place's first still), never the previous shot's last frame
    assert firsts[:4] == ["A", "B", "C", "D"] and [s["id"] for s in plan_["stills"]][:4] == ["A", "B", "C", "D"]
    assert [st.get("angle_of") for st in plan_["stills"][:4]] == [None, "A", None, "C"] and plan_["plan"]["structure"]["angle"] == {"B": "A", "D": "C"}
    assert plan_["plan"]["structure"]["explicit"] is True and plan_["plan"]["structure"]["last_frame_chaining"] is False
    legacy = pr.plan_piece(dict(req, legacy_chain=True), synth_music(160, 16))
    assert [s["first"] for s in legacy["shots"]][:4] == ["A", "last:s1", "B", "last:s3"] and legacy["plan"]["structure"]["places"] == 2


def test_pure_modes_are_exactly_as_before(music):
    cut = pr.plan_piece(dict(REQ, shots="cut"), music)
    chain = pr.plan_piece(dict(REQ, shots="chain"), music)
    n = len(cut["shots"])
    assert [s["first"] for s in cut["shots"]] == [chr(65 + k) for k in range(n)] and len(cut["stills"]) == n
    firsts = [s["first"] for s in chain["shots"]]                           # 2026-10-29: NO shot starts from another's last frame: a chain = one place from another camera in every shot, each with its own (angle) still
    assert firsts == [chr(65 + k) for k in range(len(firsts))] and len(chain["stills"]) == len(firsts) and not any(f.startswith("last:") for f in firsts)
    assert chain["stills"][0].get("angle_of") is None and all(st["angle_of"] == "A" and st["framing"] and st["angle_fallback"] for st in chain["stills"][1:])
    assert chain["plan"]["structure"]["last_frame_chaining"] is False and chain["plan"]["structure"]["angle"] == {st["id"]: "A" for st in chain["stills"][1:]}
    legacy = pr.plan_piece(dict(REQ, shots="chain", legacy_chain=True), music)             # the old behaviour stays available for the tests of the old structure
    assert [s["first"] for s in legacy["shots"]] == ["A"] + [f"last:s{k}" for k in range(1, len(legacy["shots"]))] and len(legacy["stills"]) == 1


def test_concept_keeps_the_continues_flags_and_never_collapses_a_multi_shot_reply():
    plan_ = [{"prompt": "The fisherman mends a net on the harbour wall, grey light, the camera backs away"}, {"prompt": "He pulls the net in, the camera keeps backing away from him", "continues": "true"},
             {"prompt": "A gull lands on the pier at dusk, the camera follows its glide", "continues": False}, {"prompt": "The gull lifts off over the black water, camera tilts up", "continues": True}]
    sp, fixes = CO.shot_plan_of(plan_, 4)
    assert [e.get("continues") for e in sp] == [None, True, False, True] and fixes == []
    good = {"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "old fisherman", "scene": "an old fisherman mending a net on a grey harbour wall at dawn, soft light, wide shot, calm water behind him",
            "music": "folk, guitar, fiddle, warm, gentle, light percussion, tender, acoustic", "bpm": 90, "mood": "tender", "look": "auto", "shots": "chain", "shot_plan": plan_}
    prev = [{"shots": "chain"}, {"shots": "chain"}, {"shots": "cut"}]          # the series has used chain more: the model's chain is NOT replaced
    c, errors, fixes = CO.normalize(good, 3, prev, 16)
    assert c["shots"] == "chain" and not errors and len(c["shot_plan"]) == 4


# ------------------------------------------------------------------------------------------------ unresolved `$image` (24 GB box, Qwen3.8-27B, batch_24gb_qwen run_02)
def _unresolved(ops, have=()):
    """every `$name` an op reads must be a source buffer or the `out` of an earlier op."""
    from tools.fusion import recipe_eval as RE
    have = set(have) | set(CB.BUFFERS) | {"m", "plate", "prev", "next", "prev_matte", "next_matte", "flow", "flow_m"}
    bad = []
    for o in ops:
        rs = set()
        RE.refs([o.get("args", {})], rs)
        bad += [(o["op"], o.get("out"), r) for r in sorted(rs - have)]
        have.add(o.get("out", "image"))
    return bad


def _qwen_reply():
    return json.loads((HERE / "fixtures" / "qwen27b_24gb_run02_design_reply.json").read_text(encoding="utf-8"))


def test_the_real_qwen_reply_chains_survive_with_every_reference_resolved():
    comp, fx = CB.normalize_compose(_qwen_reply()["compose"], ["s1", "s2", "s3", "s4", "s5"], 16.0)
    assert comp["default"]["ops"] and set(comp["shots"]) == {"s2", "s3"}
    for ch in [comp["default"]] + list(comp["shots"].values()):
        assert _unresolved(ch["ops"]) == [] and ch["ops"][-1]["out"] == "image"


def test_matte_treatments_woven_into_a_named_step_chain_read_the_picture_made_so_far():
    """the crash: weave_matte painted on `$image`, but a model-written chain names its steps and writes `image` only as its last op -> KeyError 'image' in compose."""
    comp, _ = CB.normalize_compose(_qwen_reply()["compose"], ["s1", "s2", "s3"], 16.0)
    words = [{"text": t, "matte": "window"} for t in ("WASH", "WAIT", "SPIN", "DRY")]
    piece = {"type": {"words": words}, "compose": copy.deepcopy(comp), "shots": []}
    pr.weave_matte(piece, None, {"words": [{"hit": 2.0 + 3 * i, "release": 3.5 + 3 * i} for i in range(4)]})
    chains = [piece["compose"]["default"]] + list(piece["compose"]["shots"].values())
    assert piece["plan"]["matte_treatments"]["chains"] == 3
    for ch in chains:
        assert _unresolved(ch["ops"]) == [], _unresolved(ch["ops"])
        assert not any(o["out"] == "image" for o in ch["ops"][:-1])          # nothing wrote `image` before the closing vignette: the base is the last picture (glow)
        assert any(o["op"] == "crossfade" and o["out"] == "glow" for o in ch["ops"])


def test_weave_matte_on_a_rules_chain_still_paints_on_image():
    base = [{"op": "sandwich", "out": "image", "args": {"footage": "$footage", "type_rgb": [1, 1, 1], "type_a": "$type_matte", "matte": "$m"}},
            {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.3}}]
    piece = {"type": {"words": [{"text": "A", "matte": "window"}]}, "compose": {"default": {"ops": copy.deepcopy(base)}}}
    pr.weave_matte(piece, None, {"words": [{"hit": 1.0, "release": 2.0}]})
    assert any(o["op"] == "crossfade" and o["out"] == "image" and o["args"]["a"] == "$image" for o in piece["compose"]["default"]["ops"])


def test_an_unresolved_reference_drops_that_op_and_its_readers_not_the_chain():
    c = ops()
    c.insert(3, {"op": "add_glow", "out": "ghost", "args": {"img": "$nowhere", "mask": "$type_matte", "radius": 20}})      # reads a buffer nothing makes
    c.insert(4, {"op": "halation", "out": "halo", "args": {"frame": "$ghost"}})                                             # reads the dropped op's output
    kept, fx = CB.validate_ops(c, 12.0)
    assert kept is not None and "ghost" not in [o["out"] for o in kept] and "halo" not in [o["out"] for o in kept]
    assert _unresolved(kept) == [] and any("nowhere" in f or "ghost" in f for f in fx)
    assert [o["op"] for o in kept if o["op"] in CB.DRAWERS] == ["sandwich"]


def test_sweep_drops_a_reader_of_a_buffer_made_only_later_and_a_chain_left_without_a_drawer_is_rejected():
    ch = [{"op": "grade", "out": "g", "args": {"frame": "$later"}}, {"op": "grade", "out": "later", "args": {"frame": "$footage"}}]
    fx = []
    assert [o["out"] for o in CB.sweep_unresolved(ch, (), fx)] == ["later"] and len(fx) == 1
    bad = [{"op": "sandwich", "out": "image", "args": {"footage": "$ghost", "type_rgb": "$tcol", "type_a": "$type_matte", "matte": "$m"}}]
    kept, fx = CB.validate_ops(bad, 12.0)
    assert kept is None and fx


def test_compose_frames_skips_an_op_with_an_unresolved_reference_instead_of_crashing_the_piece(capsys):
    import numpy as np
    from tools.fusion import recipe_eval as RE
    rec = dict(name="t", size=[64, 48], fps=24, frames=3, output="image", inputs={"footage": dict(src="footage", kind="rgb")},
               ops=[{"op": "grade", "out": "image", "args": {"frame": "$footage", "gain": 0.5}},
                    {"op": "grade", "out": "image", "args": {"frame": "$ghost", "gain": 2.0}}])
    foot = np.full((3, 48, 64, 3), 200, np.uint8)
    with RE.use_backend("numpy"):
        srcs = RE.open_sources(rec, footage=foot)
        out, st = RE.compose_frames(rec, srcs)
    assert tuple(out.shape) == (3, 48, 64, 3) and int(out[0, 0, 0, 0]) < 200           # the first grade ran, the unresolved one did not
    assert capsys.readouterr().out.count("unresolved reference") == 1               # logged once, not once per frame
