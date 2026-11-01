"""Field-level salvage of the open model's design reply (tools/fusion/design_salvage.py + design_step.py), the run-2 IndexError regression, the schema's
allowed values, and the category-based concept uniqueness. Fake replies only: no model, no network, no GPU.
Run: python -m pytest tests/test_design_salvage.py -q"""
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
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

REQ = {"words": "STILL\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING", "scene": "a lone runner in a concrete tunnel", "music": "kinetic and emotional, 160 bpm",
       "length": 12, "look": "auto", "seed": 6, "quality": "preview", "format": "9:16"}


@pytest.fixture(scope="module")
def music():
    return synth_music(160, 12)


@pytest.fixture(scope="module")
def plan(music):
    return pr.plan_piece(REQ, music)


@pytest.fixture(scope="module")
def good(plan, music):
    """a model reply that agrees with the rules: every field valid, so every field is the model's."""
    return DS.reply_from_piece(plan, music)


def parse(plan, music, reply, **kw):
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return DS.parse(text, json.dumps(DS.make_locked(plan)), plan, music, "THE DESIGN PROMPT " * 200, **kw)


def fields(out):
    return out["fields"]


def paths(lst):
    return {x["path"] if isinstance(x, dict) else x for x in lst}


def word(out, i):
    return json.loads(out["piece_json"])["type"]["words"][i]


# ------------------------------------------------------------------------------------------------ the run-2 crash
def test_run2_reply_no_longer_crashes_and_keeps_the_models_design():
    """batch v001 run 2: a window_cut `after s1` in a SINGLE-shot piece raised IndexError in window_cut.apply_to_piece, on parse AND repair, so type and compose
    both came from the rules. The exact reply + the exact piece of that run."""
    piece = json.loads((HERE / "fixtures" / "run02_piece.json").read_text(encoding="utf-8"))
    reply = (HERE / "fixtures" / "run02_design_reply.txt").read_text(encoding="utf-8")
    ana = DS.ana_from_piece(piece)
    lk = json.dumps(DS.make_locked(piece))
    out = DS.parse(reply, lk, piece, ana, "P")
    assert "design step failed" not in out["report"] and "IndexError" not in out["report"]
    assert out["ok"] is True and "type = model, compose = model" in out["report"]
    f = fields(out)
    assert f["pct_model"] > 55 and not f["fields_from_rules"]          # 60 before the pixel-face rule (QC v026): a model word on a pixel face + an extruding module now takes a readable face (font + axes counted as rule fields)
    p = out["piece"]
    assert p["design"]["src"] == {"type": "model", "compose": "model"}
    # the model's fields survived: fonts, modules, io styles of every word
    want = json.loads(reply)["type"]["words"]
    for i, w in enumerate(p["type"]["words"]):
        # a module / io style is the model's unless the one-type-system rules (<= 2 modules, <= 2 io styles per piece, design_defaults.py) folded it, and then the log says so
        assert w["module"] == want[i]["module"] or want[i]["module"] not in QP.WORD_MODULES or any(f"type.words[{i}].module" in x for x in out["report"].splitlines())
        assert (w["form_in"] == want[i]["form_in"] and w["form_out"] == want[i]["form_out"]) or any(f"type.words[{i}].form_" in x for x in out["report"].splitlines())
    assert len({w["module"] for w in p["type"]["words"]}) <= 2 and len({w["font"] for w in p["type"]["words"]}) <= 3
    # the window cut after the only shot and the window_source "next" are dropped, and the log says why
    assert any("last shot has no boundary" in x for x in out["report"].splitlines())
    assert not (p["compose"].get("cuts"))
    assert all(not w.get("window_source") for w in p["type"]["words"])
    # the other silent damage of that reply is mended: glow_color 0..255 -> 0..1, holds written as objects are read
    assert "compose.glow_color" in paths(f["fields_fixed"]) and any(x.startswith("compose.holds[") for x in paths(f["fields_fixed"]))
    # and the repair step on the same state passes
    r = DS.repair(out["piece_json"], True, None, piece, ana, lk)
    assert "not needed" in r["report"]


def test_window_cut_apply_to_piece_skips_a_cut_after_the_last_shot():
    from tools.fusion import window_cut
    piece = json.loads((HERE / "fixtures" / "run02_piece.json").read_text(encoding="utf-8"))
    piece["compose"]["cuts"] = [{"after": "s1", "cut": "window_cut", "word": 1, "exit": "grow"}, {"after": "s9", "cut": "window_swap", "word": 0}]
    piece["type"]["words"][0]["window_source"] = "next"
    notes = window_cut.apply_to_piece(piece)             # used to raise IndexError: list index out of range
    assert any("skipped" in n for n in notes) and len(notes) >= 3


# ------------------------------------------------------------------------------------------------ one bad field never costs a word
def test_a_valid_reply_is_all_model(plan, music, good):
    """the rules' own plan as a reply: nothing from the rules; the only changes are the piece rules' (the hero word gets an expressive face, a planned window_other fits its word, the 2026-10-29
    defaults: echo modules folded, the size floor) and the exit_s floor."""
    out = parse(plan, music, good)
    f = fields(out)
    assert out["ok"] and not f["fields_from_rules"] and f["pct_model"] >= 67.0         # (measured 69.2 on 2026-10-30; was 70 before the stable-type rule, lowered to 60 and tightened again here so a real drop shows: the stable-type rule 2026-10-30 centres every word, those placement fields are code fixes)  # (a word before a 3D cut ends 1.2 s earlier than the model wrote it: its hold_s is a fix)
    assert all(x["path"].endswith(".exit_s") or x["why"].startswith("piece rule") for x in f["fields_fixed"]), f["fields_fixed"]
    assert any(x.startswith("type.words[2].font") for x in f["fields_from_model"])


def test_io_style_gradient_fade_is_mapped_alone(plan, music, good):
    """gradient_fade / fade / shrink are not io styles: each is replaced by the nearest allowed style, every other field of the word stays the model's."""
    bad = copy.deepcopy(good)
    ws = bad["type"]["words"]
    ws[1]["form_in"], ws[1]["form_out"], ws[2]["form_out"] = "gradient_fade", "fade", "shrink"
    ws[1]["hit"], ws[1]["module"] = good["type"]["words"][1]["hit"], good["type"]["words"][1]["module"]
    out = parse(plan, music, bad)
    f = fields(out)
    assert out["ok"] is True and out["repair_prompt"] == ""                     # an obvious nearest value needs no repair call
    assert {"type.words[1].form_in", "type.words[1].form_out", "type.words[2].form_out"} <= paths(f["fields_fixed"])
    assert not f["fields_from_rules"]
    assert word(out, 1)["form_in"] == "wipe" and word(out, 1)["form_out"] == "wipe"
    assert word(out, 2)["form_out"] in ("blobs", "flicker")                     # 'blobs' unless the one-system rule (<= 2 io styles per piece) folded it into the piece's most used style
    assert word(out, 1)["font"] in (good["type"]["words"][1]["font"], "Anybody[wdth,wght].ttf") and word(out, 1)["module"] in QP.WORD_MODULES      # a planned window_other / the one-system rules may change them (logged)
    assert out["piece"]["design"]["src"] == {"type": "model", "compose": "model"}


def test_unmappable_module_takes_the_rules_value_for_that_field_only(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"][2]["module"] = "zzz_unknown_thing"
    bad["type"]["mood"] = "tender"
    out = parse(plan, music, bad)
    f = fields(out)
    assert out["ok"] is False and paths(f["fields_from_rules"]) == {"type.words[2].module"}
    assert word(out, 2)["module"] in QP.WORD_MODULES        # the piece's own module (openings.default_module / the one-system fold keeps the model's choice; no "lanes" bias)
    from tools.fusion import design_defaults as DD
    from mograph import fonts_liked as fl
    f2 = good["type"]["words"][2]["font"]
    assert word(out, 2)["font"] == f2 or fl.lookup(f2)["key"] in DD.MAIN_HOLD_RISKY       # the rest of the word is the model's (a dotted / thin / outline face with no big box is swapped for a readable one: round-3 QC)
    assert json.loads(out["piece_json"])["type"]["mood"] == "tender"
    assert out["piece"]["design"]["src"]["type"] == "model fields + rules fields"
    # the repair prompt is SHORT: only that field, with the allowed values; not the design prompt
    rp = out["repair_prompt"]
    assert "type.words[2].module" in rp and "lanes" in rp and "THE DESIGN PROMPT" not in rp and len(rp) < 3000 and "type.words[0]" not in rp.split("Example")[0]


def test_font_the_model_made_up_is_replaced_alone(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"][0]["font"] = "Comic Sans.ttf"
    out = parse(plan, music, bad)
    f = fields(out)
    assert "type.words[0].font" in paths(f["fields_from_rules"]) and "not in the liked font list" in out["repair_prompt"]
    assert "FONTS" in out["repair_prompt"]                                                  # the font list is only in a prompt that asks for a font
    assert word(out, 1)["font"] in (good["type"]["words"][1]["font"], "Anybody[wdth,wght].ttf")


def test_axis_the_font_lacks_and_matte_behind_are_dropped_alone(plan, music, good):
    bad = copy.deepcopy(good)
    w = bad["type"]["words"][1]
    w["axes"] = {"ROND": 3, "wght": 700}
    w["matte"] = "behind"
    out = parse(plan, music, bad)
    f = fields(out)
    assert out["ok"] is True
    fx = paths(f["fields_fixed"])
    assert "type.words[1].matte" in fx
    assert word(out, 1).get("matte") is None
    assert word(out, 1)["t0"] is not None and word(out, 1)["font"] in (w["font"], "Anybody[wdth,wght].ttf")
    from mograph import font_registry as fr
    if word(out, 1)["font"] != w["font"]:                                                  # a planned window_other gave the word the heavy narrow face: its axes were the old font's (logged)
        return
    vary = next((v.get("vary") or {} for v in fr.REGISTRY.values() if v["file"] == w["font"]), {})
    for tag in ("ROND", "wght"):
        assert (f"type.words[1].axes.{tag}" in paths(f["fields_fixed"])) == (tag not in vary), tag          # an axis the font lacks is dropped; one it has is kept


def test_axis_synonym_and_clamp(plan, music, good):
    bad = copy.deepcopy(good)
    w = bad["type"]["words"][0]
    vary = DS._vary()
    fnt = next(f for f, v in vary.items() if "wght" in v and f in {e["file"] for e in __import__("mograph.fonts_liked", fromlist=["LIKED"]).LIKED})
    w["font"] = fnt
    lo, hi = vary[fnt]["wght"]
    w["axes"] = {"weight": hi + 5000}
    out = parse(plan, music, bad)
    ax = word(out, 0).get("axes") or {}
    if word(out, 0)["font"] == fnt:                       # the font rules may swap it; then the axes went with it (logged)
        assert ax.get("wght") == hi


def test_hold_below_the_floor_is_clamped_not_an_error(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"][0]["hold_s"] = 0.5
    out = parse(plan, music, bad)
    assert out["ok"] is True and "type.words[0].hold_s" in paths(fields(out)["fields_fixed"])


def test_glow_color_in_0_255_is_scaled_and_holds_objects_are_read(plan, music, good):
    bad = copy.deepcopy(good)
    bad["compose"]["glow_color"] = [10, 10, 10]
    bad["compose"]["holds"] = [{"time": 4.0, "freeze_s": 0.4}, [6.0, 0.3], "oops"]
    out = parse(plan, music, bad)
    f = fields(out)
    assert out["ok"] is True
    assert {"compose.glow_color", "compose.holds[0]", "compose.holds[2]"} <= paths(f["fields_fixed"])
    glows = [o for o in out["piece"]["compose"]["default"]["ops"] if o["op"] == "add_glow" and isinstance(o["args"].get("color"), (list, tuple))]
    assert all("0.039" in json.dumps(o["args"]["color"]) for o in glows)                                            # 10/255, not clamped to white (2026-10-30: no default glow behind the type, so the colour only shows where a glow op exists)


def test_transitions_wrong_length_and_unknown_entry_become_null_gaps(plan, music, good):
    bad = copy.deepcopy(good)
    n = len(plan["type"]["words"])
    assert n == 3
    bad["type"]["transitions"] = ["twist", "zigzag", "twist"]                  # 3 entries for 2 gaps, the 2nd unknown
    out = parse(plan, music, bad)
    f = fields(out)
    tr = json.loads(out["piece_json"])["type"]["transitions"]
    assert isinstance(tr, list) and len(tr) == n - 1 and tr[0] == "twist" and tr[1] is None      # unknown -> null: the engine picks that gap
    assert "type.transitions" in paths(f["fields_fixed"]) and "type.transitions[1]" in paths(f["fields_from_rules"])
    assert "type.transitions[0]" in f["fields_from_model"]
    dup = copy.deepcopy(good)
    dup["type"]["transitions"] = ["twist", "twist"]
    tr2 = json.loads(parse(plan, music, dup)["piece_json"])["type"]["transitions"]
    assert tr2 == ["twist", None]                                                 # a name twice (except cut) -> the later one is the engine's choice


def test_one_missing_word_entry_costs_only_that_word(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"].pop(1)
    out = parse(plan, music, bad)
    f = fields(out)
    assert out["ok"] is False and out["piece"]["design"]["src"]["type"] == "model fields + rules fields"
    rs = paths(f["fields_from_rules"])
    assert rs and all(p.startswith("type.words[1]") for p in rs)
    assert any(p.startswith("type.words[0].") for p in f["fields_from_model"]) and any(p.startswith("type.words[2].") for p in f["fields_from_model"])
    assert [w["text"] for w in out["piece"]["type"]["words"]] == [w["text"] for w in plan["type"]["words"]]
    assert "type.words[1]" in out["repair_prompt"]


def test_missing_sections_come_from_the_rules_one_section_at_a_time(plan, music, good):
    bad = copy.deepcopy(good)
    bad.pop("compose")
    out = parse(plan, music, bad)
    assert out["piece"]["design"]["src"] == {"type": "model", "compose": "rules"} and not out["ok"]
    assert fields(out)["sections"]["compose"] == "rules"
    bad2 = copy.deepcopy(good)
    bad2.pop("type")
    out2 = parse(plan, music, bad2)
    assert out2["piece"]["design"]["src"] == {"type": "rules", "compose": "model"} and "missing object `type`" in out2["repair_prompt"]


# ------------------------------------------------------------------------------------------------ the repair call: invalid fields only, merged field by field
def test_repair_reply_path_map_is_merged_field_by_field(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"][2]["module"] = "zzz_unknown_thing"
    bad["type"]["words"][1]["form_in"] = "###"
    lk = json.dumps(DS.make_locked(plan))
    first = parse(plan, music, bad)
    assert paths(fields(first)["fields_from_rules"]) == {"type.words[2].module", "type.words[1].form_in"}
    from tools.fusion import design_defaults as DD
    target = next(m for m in QP.WORD_MODULES if m != plan["type"]["words"][2]["module"] and m not in DD.AVOID_MODULES and m in {w["module"] for w in good["type"]["words"]})         # (an echo / two-style / 3D module is folded into lanes by the one-system rule)
    reply2 = json.dumps({"type.words[2].module": target, "type.words[1].form_in": "also not valid", "type.words[0].font": "Comic Sans.ttf"})
    fixed = DS.repair(first["piece_json"], False, reply2, plan, music, lk)
    f = fixed["fields"]
    p = fixed["piece"]
    assert p["type"]["words"][2]["module"] == target                               # the repaired field is the model's now
    assert "type.words[2].module" in f["fields_from_model"]
    assert paths(f["fields_from_rules"]) == {"type.words[1].form_in"}              # the wrong repair value stays with the rules
    assert p["type"]["words"][0]["font"] == first["piece"]["type"]["words"][0]["font"]      # a field that was not pending cannot be changed by the repair reply
    from tools.fusion import design_defaults as DD
    assert p["type"]["words"][1]["module"] in (good["type"]["words"][1]["module"], "lanes") and (p["type"]["words"][1]["module"] == good["type"]["words"][1]["module"] or good["type"]["words"][1]["module"] in DD.AVOID_MODULES)


def test_repair_reply_that_is_a_design_shaped_object_also_works(plan, music, good):
    bad = copy.deepcopy(good)
    bad["type"]["words"][1]["module"] = "nope"
    first = parse(plan, music, bad)
    lk = json.dumps(DS.make_locked(plan))
    fixed = DS.repair(first["piece_json"], False, json.dumps(good), plan, music, lk)
    from tools.fusion import design_defaults as DD
    gm = good["type"]["words"][1]["module"]
    assert fixed["piece"]["type"]["words"][1]["module"] in QP.WORD_MODULES and fixed["piece"]["type"]["words"][1]["module"] not in DD.AVOID_MODULES and not fixed["fields"]["fields_from_rules"]


# ------------------------------------------------------------------------------------------------ piece-level rules change the minimum and log it
def test_plain_fonts_get_the_fewest_swaps_and_the_report_marks_them(plan, music, good):
    bad = copy.deepcopy(good)
    from mograph import fonts_liked as fl
    plain = next(e["file"] for e in fl.LIKED if e["role"] == "text" and e["key"] not in QP.NOT_PICKED)
    for w in bad["type"]["words"]:
        w["font"] = plain
        w.pop("axes", None)
    out = parse(plan, music, bad)
    f = fields(out)
    changed = [x for x in f["fields_fixed"] if x["path"].endswith(".font")]
    assert changed and all(x["why"].startswith("piece rule") for x in changed)
    n_swapped = sum(1 for w in out["piece"]["type"]["words"] if w["font"] != plain)
    assert len(changed) == n_swapped and n_swapped < len(bad["type"]["words"])       # not every word: the model's font stays wherever the rules allow it
    assert any("swapped" in x for x in out["report"].splitlines())


def test_overlong_last_hold_shortens_only_the_last_word(plan, music, good):
    bad = copy.deepcopy(good)
    last = len(bad["type"]["words"]) - 1
    bad["type"]["words"][last]["hold_s"] = 6.0
    out = parse(plan, music, bad)
    f = fields(out)
    fx = paths(f["fields_fixed"])
    hold_changes = {p for p in fx if p.endswith(".hold_s")}
    assert out["ok"] is True
    if hold_changes:                                       # the timing fit ran: the fewest fields (A0 = the last word's hold only)
        assert hold_changes - {"type.words[0].hold_s", "type.words[1].hold_s"} == {f"type.words[{last}].hold_s"}, hold_changes          # (+ the words before a 3D cut: their exit ends before the cut's window)


# ------------------------------------------------------------------------------------------------ the schema lists every allowed value
def test_schema_lists_the_allowed_values_of_every_enum_field(plan, music):
    pk = DS.package(REQ, music, plan, [], None, "DESIGN_GUIDE_open_model", 6)
    pr_ = pk["prompt"]
    tail = pr_[pr_.index("=== OUTPUT SCHEMA ==="):]
    for group in (QP.IO_STYLES, QP.WORD_MODULES, QP.MATTES, QP.TRANSITIONS, QP.CAMERAS, pk["techniques"]):
        assert all(g in tail for g in group)                                       # the enumerations are rotated per design seed (openings): every value is listed, the order differs
    assert "axes: wght" in pr_ and "axis tags in that font's line" in tail                 # axes per chosen font: the FONTS list carries them
    assert "never write behind or none" in tail and "NOT 0..255" in tail
    # fonts: expressive first, shuffled by the seed (unchanged)
    fb1, fb2 = QP.fonts_block(6), QP.fonts_block(7)
    assert fb1 != fb2 and fb1.splitlines()[0].count("role") == 1
    first_roles = [ln.split("| role ")[1].split(" |")[0] for ln in fb1.splitlines()]
    assert first_roles[0] in QP.EXPRESSIVE


def test_nearest_value_tables_are_documented_in_the_module():
    doc = SV.__doc__
    for key in ("NEAREST ALLOWED VALUE", "gradient_fade", "wipe", "blobs", "hold_s floor 1.2", "glow_color", "synonyms"):
        assert key in doc
    assert SV.SYN_IO["gradient_fade"] == "wipe" and SV.SYN_IO["shrink"] == "blobs"
    for k, v in SV.SYN_IO.items():
        assert v in QP.IO_STYLES
    for table, allowed in ((SV.SYN_MODULE, QP.WORD_MODULES), (SV.SYN_CAMERA, QP.CAMERAS), (SV.SYN_MATTE, QP.MATTES), (SV.SYN_TECHNIQUE, QP.TECHNIQUES)):
        assert all(v in allowed for v in table.values())
    assert all(v in QP.TRANSITIONS or v is None for v in SV.SYN_TRANSITION.values())


# ------------------------------------------------------------------------------------------------ concept uniqueness by category
CHILDREN = ["ice skater child", "child with helium balloon", "young skater girl", "a little boy", "teenager on a bike", "kid with a kite", "young person"]


@pytest.mark.parametrize("subject", CHILDREN)
def test_child_words_are_one_category(subject):
    assert CO.subject_category(subject) == "child"


def test_other_categories():
    assert CO.subject_category("fog-bound pine sapling") == "landscape"
    assert CO.subject_category("young pine sapling") == "landscape"
    assert CO.subject_category("wild white horse") == "animal"
    assert CO.subject_category("young fox") == "animal"
    assert CO.subject_category("old fisherman") == "adult"
    assert CO.subject_category("paper boat") == "vehicle"
    assert CO.subject_category("little robot") == "object"
    assert CO.subject_category("crowd at a concert") == "crowd"
    assert CO.subject_category("neon noodle stall") == "object"
    assert CO.subject_category("", "a boy runs along the shore") == "child"                    # no subject words: the scene decides


def _c(subject, scene, mood="tender", words=None, music="a, b, c"):
    return dict(subject=subject, scene=scene, mood=mood, words=words or ["X", "Y", "Z", "W", "V"], music=music, shots="cut")


def test_a_second_child_is_rejected_until_four_categories_are_used():
    prev = [_c("ice skater child", "a child skating on a frozen lake at dawn", "playful", ["A", "B", "C", "D", "E"], "lo-fi, piano, warm"),
            _c("fog-bound pine sapling", "a pine sapling in a misty forest", "tense", ["F", "G", "H", "I", "J"], "drone, cello, cold")]
    c = _c("child with a lantern", "a child holding a lantern in a dark barn", "hopeful", ["K", "L", "M", "N", "O"], "folk, banjo, bright")
    cl = CO.clashes(c, prev)
    assert any("same kind of subject" in x and "child" in x for x in cl)
    ok = _c("old fisherman", "a fisherman mending nets in a stone shed", "solemn", ["P", "Q", "R", "S", "T"], "blues, slide guitar, slow")
    assert not any("same kind of subject" in x for x in CO.clashes(ok, prev))
    # once 4 different categories were used, a repeat of one is fine
    prev += [_c("wild horse", "a horse on a salt flat", "free", ["a", "b", "c", "d", "e"], "rock, fuzz, drums"),
             _c("paper boat", "a paper boat in a rain gutter on a street", "playful2", ["f", "g", "h", "i", "j"], "folk, guitar, light")]
    assert len(CO.used_categories(prev)["subject"]) >= CO.MIN_CATEGORIES
    assert not any("same kind of subject" in x for x in CO.clashes(c, prev))


def test_same_scene_setting_is_rejected_too():
    prev = [_c("old fisherman", "a fisherman on a pier above the sea at dawn", "solemn")]
    c = _c("wild horse", "a horse galloping along the beach by the sea", "free", ["a", "b", "c", "d", "e"], "rock, fuzz, drums")
    assert any("same scene setting (water)" in x for x in CO.clashes(c, prev))


def test_the_concept_prompt_lists_the_used_categories():
    prev = [_c("ice skater child", "a child skating on a frozen lake at dawn", "playful")]
    p = CO.concept_prompt(3, 16, prev)
    assert "Subject categories already used: child / young person" in p and "pick one" in p
    assert "Subject categories already used" not in CO.concept_prompt(3, 16, [])


def test_the_seven_batch_concepts_would_have_been_stopped():
    """the 3060 batch v001 state: six of seven concepts were 'child'."""
    st = json.loads((ROOT / "16_app" / "bench" / "batch_3060" / "state.json").read_text(encoding="utf-8")) if (ROOT / "16_app" / "bench" / "batch_3060" / "state.json").is_file() else None
    if not st:
        pytest.skip("batch state not present")
    prev, rejected = [], 0
    for r in st["runs"]:
        c = r.get("concept")
        if not c:
            continue
        if CO.category_clashes(c, prev):
            rejected += 1
        prev.append(c)
    assert rejected >= 4


# ------------------------------------------------------------------------------------------------ the redesign graphs (model A/B)
def _bg():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_graphs_redesign_t", ROOT / "tools" / "app" / "build_graphs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_graphs_redesign_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def _redesign_graph_param(arm, name):
    return pytest.param(arm, marks=pytest.mark.skipif(not (ROOT / "app" / "graphs" / f"{name}.json").exists(),
                                                      reason=f"dev fixture not shipped in the public export: app/graphs/{name}.json"))


@pytest.mark.parametrize("arm", [_redesign_graph_param("q4", "fusion_redesign"), _redesign_graph_param("q8", "fusion_redesign_q8"),
                                 _redesign_graph_param("gemma", "fusion_redesign_gemma"), _redesign_graph_param("replay", "fusion_redesign_replay")])
def test_redesign_graph_is_the_design_step_only(arm):
    bg = _bg()
    g, side = bg.build_redesign(arm), bg.build_redesign_sidecar(arm)
    problems, _ = bg.check(g, side)
    assert problems == [], problems
    classes = {n["class_type"] for n in g.values()}
    # no music model, no still model, no LTX, no Alpha-Gen, no depth: only the design step, the type passes, compose, assemble, archive
    for banned in ("KSampler", "UNETLoader", "DualCLIPLoader", "SamplerCustomAdvanced", "LTXAddVideoICLoRAGuide", "VideoDepthAnythingProcess", "MographMusicT2MCut"):
        assert banned not in classes
    for need in ("MographArchiveLoad", "MographDesignPackage", "MographDesignParse", "MographDesignRepair", "MographPieceFromJSON", "MographTypePasses",
                 "MographCompose", "MographAssemble", "MographArchive", "SaveVideo"):
        assert need in classes
    assert bg.REDESIGN_ARMS[arm][0] in {"fusion_redesign", "fusion_redesign_q8", "fusion_redesign_gemma", "fusion_redesign_replay"}
    assert g["archive"]["inputs"]["light"] is True
    # the repair call only runs when parse says ok = false (lazy), and the design seed is the graph's IN seed in every arm
    assert g["design_fix"]["inputs"]["repair_reply"] == ["design_qwen2", 0] and g["design_parse"]["inputs"]["model"] == bg.REDESIGN_ARMS[arm][1]
    assert not any("critique" in json.dumps(n["inputs"]) for n in g.values())        # Refine is not built
    assert (ROOT / "app" / "graphs" / (bg.REDESIGN_ARMS[arm][0] + ".json")).read_text(encoding="utf-8") == bg.dump(g)
    assert (ROOT / "app" / "graphs" / (bg.REDESIGN_ARMS[arm][0] + ".app.json")).read_text(encoding="utf-8") == bg.dump(side)


def test_redesign_arms_differ_only_in_the_model_nodes():
    bg = _bg()
    q4, q8, gm = bg.build_redesign("q4"), bg.build_redesign("q8"), bg.build_redesign("gemma")
    assert q4["design_qwen"]["inputs"]["quantization"] == "4-bit (VRAM-friendly)" and q8["design_qwen"]["inputs"]["quantization"] == "8-bit (Balanced)"
    assert q8["design_qwen2"]["inputs"]["quantization"] == "8-bit (Balanced)"
    same = lambda a, b: {k: v for k, v in a.items() if k not in ("design_qwen", "design_qwen2", "design_parse", "design_fix")} == \
                        {k: v for k, v in b.items() if k not in ("design_qwen", "design_qwen2", "design_parse", "design_fix")}       # noqa: E731
    assert same(q4, q8)
    assert gm["design_qwen"]["class_type"] == "TextGenerate" and gm["design_clip"]["class_type"] == "CLIPLoader"
    assert gm["design_clip"]["inputs"]["clip_name"].startswith("gemma4_12b") and gm["design_qwen"]["inputs"]["thinking"] is False
    assert gm["design_qwen"]["inputs"]["max_length"] >= 2048                                        # the template default (256) cuts the JSON off
    assert gm["design_qwen"]["inputs"]["image"] == ["design_pkg", 3]                                 # ONE image: the combined sheet
    assert gm["design_qwen2"]["inputs"].get("image") is None
    assert q4["design_qwen"]["inputs"]["image"] == ["design_pkg", 3] and q4["design_qwen"]["inputs"]["seed"] == ["seed", 0]


def test_archive_loader_reads_an_archive(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    for c in (str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if (Path(c) / "comfy_api" / "latest" / "__init__.py").is_file():
            sys.path.insert(0, c)
            break
    else:
        pytest.skip("no ComfyUI checkout")
    try:
        from nodes_fusion import redesign as R
    except Exception as e:       # noqa: BLE001
        pytest.skip(f"nodes_fusion not importable here: {e}")
    piece = json.loads((HERE / "fixtures" / "run02_piece.json").read_text(encoding="utf-8"))
    n = piece["layout"]["s1"]["frames"]
    arch = tmp_path / "x_out_shots"
    (arch / "s1_matte").mkdir(parents=True)
    (arch / "piece.json").write_text(json.dumps(piece), encoding="utf-8")
    vw = cv2.VideoWriter(str(arch / "s1_clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 112))
    for i in range(n):
        vw.write(np.full((112, 64, 3), (i * 3) % 255, np.uint8))
    vw.release()
    for i in range(1, n + 1):
        cv2.imwrite(str(arch / "s1_matte" / f"{i:05d}.png"), np.full((112, 64, 3), 255, np.uint8))
    cv2.imwrite(str(arch / "still.png"), np.zeros((112, 64, 3), np.uint8))
    out = R.MographArchiveLoad.execute(str(arch), "", json.dumps({"words": ["A", "B"], "scene": "s", "music": "m", "look": "auto", "shots": "single", "length": 16, "seed": 2})).args
    request, music, loaded, clips, first, audio = out[:6]
    assert [c["sid"] for c in clips] == ["s1"] and clips[0]["u8"].shape[0] == n and tuple(out[6].shape[1:]) == (112, 64, 3) and out[7].shape[1] == 16
    assert loaded["id"] == piece["id"] and music["cut"] == piece["audio"]["cut"] and music["downbeats"] == piece["type"]["downbeats"]
    assert request["seed"] == 2 and tuple(first.shape) == (1, 112, 64, 3)


# ------------------------------------------------------------------------------------------------ malformed JSON of the small models
def test_unclosed_type_object_and_capitalised_keys_lose_nothing(plan, music):
    """Gemma 4 12B (redesign A/B, r01b): `type` was never closed (compose written inside it, the root `}` missing) and keys were `Matte` / `Technique`:
    the old extract_json found only an inner object ('missing object `type`') and the whole type came from the rules."""
    text = (HERE / "fixtures" / "gemma4_12b_r01b_reply.txt").read_text(encoding="utf-8")
    obj = QP.extract_json(text)
    assert isinstance(obj, dict) and "type" in obj and "compose" in obj["type"]
    out = parse(plan, music, text)
    f = fields(out)
    assert out["piece"]["design"]["src"]["type"] != "rules" and "missing object" not in out["report"]
    assert any(p.startswith("type.words[0].font") for p in f["fields_from_model"] + [x["path"] for x in f["fields_fixed"]])
    assert any(p.startswith("compose.") for p in f["fields_from_model"])                      # the nested compose was hoisted
    assert not any(x["path"] == "compose.technique" for x in f["fields_from_rules"])         # `Technique` is read as `technique`


def test_extract_json_closes_missing_brackets_only_as_a_last_resort():
    assert QP.extract_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1, 2]}
    assert QP.extract_json('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert QP.extract_json('{"a": {"b": 2}, "c": {"d": [1, {"e": 3}]') == {"a": {"b": 2}, "c": {"d": [1, {"e": 3}]}}
    assert QP.extract_json("no json here") is None
    assert QP.extract_json('{"a": }') is None


def test_axis_tags_keep_their_case_when_keys_are_lowered():
    o = SV.lower_keys({"Type": {"Words": [{"Font": "x", "Axes": {"SOFT": 1, "wght": 400}}]}})
    assert o == {"type": {"words": [{"font": "x", "axes": {"SOFT": 1, "wght": 400}}]}}


# ------------------------------------------------------------------------------------------------ a timing the music cannot hold: the fewest fields come from the rules
def test_timing_that_runs_out_of_music_takes_the_fewest_fields_from_the_rules(music):
    """redesign r07 (batch v001 r07 too): three words of 2 s form + 1.2 s hold + 1.5 s exit cannot all fit in the music, the engine rejects the model's timing. Before
    this change ALL twelve timing fields went to the rules; now the last word's hit alone does (the word that runs out of music), the rest stays the model's.
    A `single`-shot piece: with one shot per word the shot fit (qwen_piece.fit_to_shots) already moves such timing inside the shots (tests/test_owner.py)."""
    plan = pr.plan_piece(dict(REQ, shots="single"), music)
    good = DS.reply_from_piece(plan, music)
    bad = copy.deepcopy(good)
    for w, h in zip(bad["type"]["words"], (2.1, 7.0, 11.0)):
        w.update(form_s=2.0, exit_s=1.5, hold_s=1.2, hit=h)
    out = parse(plan, music, bad)
    f = fields(out)
    last = len(bad["type"]["words"]) - 1
    # 2026-10-28 (single-shot design replies were thrown away): a chain that does not fit even with every hold at the floor is no longer a hard error, `fit_scene`
    # (tighter form / exit, hits moved to the earliest beats) fits it deterministically: NOTHING comes from the rules, the changed fields are `fixed` and logged
    assert not f["fields_from_rules"], f["fields_from_rules"]
    assert "type.words[0].hold_s" in f["fields_from_model"] + [x["path"] for x in f["fields_fixed"]] and f"type.words[{last}].hit" in [x["path"] for x in f["fields_fixed"]]
    assert out["piece"]["design"]["src"]["type"] == "model" and ("TIMING FIT" in out["report"] or "do not fit" in out["report"])


def test_empty_lines_list_is_not_a_field(plan, music, good):
    bad = copy.deepcopy(good)
    for w in bad["type"]["words"]:
        w["lines"] = []
    out = parse(plan, music, bad)
    assert not any(p.endswith(".lines") for p in [x["path"] for x in fields(out)["fields_fixed"]] + [x["path"] for x in fields(out)["fields_from_rules"]])


def test_a_designed_piece_gets_hold_life_woven_into_the_models_compose_chains(plan, music, good):
    """the plan stage's hold-life weave is replaced when the model's compose is merged: design_step._finish weaves it again (r14 / r15 pieces had no hold_life op at all)."""
    out = parse(plan, music, good)
    p = out["piece"]
    chains = ([p["compose"]["default"]] if p["compose"].get("default") else []) + list((p["compose"].get("shots") or {}).values())
    ops = [o["op"] for c in chains for o in c["ops"] if o.get("hl")]
    hl = (p.get("plan") or {}).get("hold_life") or {}
    assert hl.get("chains", 0) >= 1 and (not hl.get("effects") or ops), (hl, ops)             # an effect was drawn -> its ops are in the designed chains
    forced = copy.deepcopy(good)
    forced["compose"]["hold_life"] = ["pulse"]
    p2 = parse(plan, music, forced)["piece"]
    ops2 = [o["op"] for c in ([p2["compose"]["default"]] if p2["compose"].get("default") else []) for o in c["ops"] if o.get("hl")]
    assert "hold_life" in ops2 or (p2["plan"].get("hold_life") or {}).get("skipped"), ops2
