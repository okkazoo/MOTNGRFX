"""The `reply_schema` arm flag of tools/app/llm_arms.py (default OFF) and the `design_repair` schema name (tools/fusion/reply_schema.py; 16_app/REPLY_SCHEMA.md hand-off).
No model, no GPU: the converted graphs are static JSON. Run: python -m pytest tests/test_reply_schema_flag.py -q"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

from tools.app import llm_arms as LA  # noqa: E402
from tools.app import llm_gguf as LG  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import reply_schema as RS  # noqa: E402

GRAPHS = ROOT / "app" / "graphs"
BASE = "fusion_design_qwen_auto_layers_pivot"
ARMS = ("gemma31b_qat", "gemma31b_qat")
needs_base = pytest.mark.skipif(not (GRAPHS / f"{BASE}.json").exists(), reason=f"dev fixture not shipped in the public export: app/graphs/{BASE}.json")


def load(name):
    return json.load(open(GRAPHS / f"{name}.json", encoding="utf-8")), json.load(open(GRAPHS / f"{name}.app.json", encoding="utf-8"))


@pytest.fixture(scope="module")
def base():
    return load(BASE)


def schemas(g):
    return {nid: n["inputs"].get("schema") for nid, n in g.items() if n["class_type"] == LA.NEW}


@needs_base
def test_the_flag_is_off_by_default_and_the_plain_graph_has_no_schema_input(base):
    assert not any("reply_schema" in a for a in LA.ARMS.values())
    for arm in ARMS:
        g, s = LA.convert(*base, arm)
        assert schemas(g) == {"concept_qwen": None, "concept_qwen2": None, "design_qwen": None, "design_qwen2": None}
        assert all("schema" not in n["inputs"] for n in g.values() if n["class_type"] == LA.NEW) and "reply_schema" not in s
        assert LA.check(g, arm, base[0]) == []


@needs_base
def test_flag_on_sets_the_schema_by_role(base):
    for arm in ARMS:
        g, s = LA.convert(*base, arm, reply_schema=True)
        assert schemas(g) == {"concept_qwen": "concept", "concept_qwen2": None, "design_qwen": "design", "design_qwen2": "design_repair"}
        assert LA.check(g, arm, base[0], reply_schema=True) == []
        assert s["reply_schema"]["design_repair"] == "design_repair" and s["title"].endswith("reply schema grammar")
        # nothing else changed against the plain twin
        plain, _ = LA.convert(*base, arm)
        diff = [n for n in g if g[n] != plain[n]]
        assert sorted(diff) == ["concept_qwen", "design_qwen", "design_qwen2"]
        assert {k for n in diff for k in set(g[n]["inputs"]) ^ set(plain[n]["inputs"])} == {"schema"}
        assert g["design_qwen"]["inputs"]["image"] == ["design_pkg", 3]
        assert g["concept_qwen"]["inputs"].get("json_mode") == plain["concept_qwen"]["inputs"].get("json_mode")      # json_mode stays as it was (the schema implies the JSON grammar)


@needs_base
def test_the_arm_flag_alone_turns_it_on(base, monkeypatch):
    arm = dict(LA.ARMS["gemma31b_qat"], reply_schema=True)
    monkeypatch.setitem(LA.ARMS, "gemma31b_qat", arm)
    g, _ = LA.convert(*base, "gemma31b_qat")
    assert schemas(g)["design_qwen"] == "design"
    assert LA.check(g, "gemma31b_qat", base[0]) == []
    g2, _ = LA.convert(*base, "gemma31b_qat", reply_schema=False)                    # an explicit False wins over the arm flag
    assert all(v is None for v in schemas(g2).values())


@needs_base
def test_a_thinking_arm_keeps_the_design_call_without_a_grammar(base, monkeypatch):
    arm = dict(LA.ARMS["gemma31b_qat"], thinking=True)
    monkeypatch.setitem(LA.ARMS, "gemma31b_qat", arm)
    g, _ = LA.convert(*base, "gemma31b_qat", reply_schema=True)
    assert schemas(g)["design_qwen"] is None and g["design_qwen"]["inputs"]["thinking"] is True and schemas(g)["concept_qwen"] == "concept"
    probs = LA.check(g, "gemma31b_qat", base[0], reply_schema=True)
    assert all("out_design_reasoning" in p for p in probs)             # only the known slot-2 (reasoning) note of the class table, nothing about the schema


@needs_base
def test_check_catches_a_schema_where_none_is_wanted_and_a_missing_one(base):
    g, _ = LA.convert(*base, "gemma31b_qat")
    g["design_qwen"]["inputs"]["schema"] = "design"
    assert any("design_qwen" in p and "schema" in p for p in LA.check(g, "gemma31b_qat", base[0]))
    g2, _ = LA.convert(*base, "gemma31b_qat", reply_schema=True)
    del g2["concept_qwen"]["inputs"]["schema"]
    assert any("concept_qwen" in p for p in LA.check(g2, "gemma31b_qat", base[0], reply_schema=True))


@pytest.mark.parametrize("arm", [a for a in ARMS if (GRAPHS / f"{BASE}_{a}.json").is_file()])        # the Qwen graphs are not in the public export (Gemma only)
@needs_base
def test_the_alternate_graph_files_exist_and_are_current(base, arm):
    plain, ps = load(f"{BASE}_{arm}")
    twin, ts = load(f"{BASE}_{arm}_schema")
    cg, cs = LA.convert(*base, arm, reply_schema=True)
    assert twin == cg and ts == cs                                                       # `llm_arms.py --reply-schema --base ... --arm ...` is up to date
    assert plain == LA.convert(*base, arm)[0]
    assert "schema" not in json.dumps({k: v for k, v in plain.items() if v["class_type"] == LA.NEW})
    assert schemas(twin) == {"concept_qwen": "concept", "concept_qwen2": None, "design_qwen": "design", "design_qwen2": "design_repair"}


# ============================================================================================ design_repair: the long repair gets the design schema, the short one none
DESIGN_PROMPT = ("=== THE PIECE ===\nWords, in order, each shown once: STILL / SEARCHING\n\n=== SHOTS (footage, fixed) ===\n- s1: starts 0 s, lasts 4 s (96 f); matte: yes; x\n"
                 "- s2: starts 4 s, lasts 4 s (96 f); matte: yes; y\n\nImages attached, in order: sheets\n")


def test_design_repair_schema_follows_the_prompt():
    long_p = DS.repair_prompt(DESIGN_PROMPT, '{"type": {"words": "x"}', ["type.words: a list is needed"])
    assert RS.REPAIR_MARK in long_p
    s = RS.schema_for("design_repair", long_p)
    assert s is not None and s == RS.schema_for("design", long_p)                         # the long repair = the design schema with the counts of the prompt
    short_p = SV.repair_prompt(SimpleNamespace(pending=[{"path": "type.words[0].form_in", "bad": "fade", "why": "not an io style", "allowed": "lines, ..."}], want=["STILL", "SEARCHING"]),
                               "{}", "", ["STILL", "SEARCHING"])
    assert RS.schema_for("design_repair", short_p) is None                                # the short repair's reply is a {path: value} map
    assert RS.schema_for("design_repair", "no words here") is None and RS.schema_for("design_repair", None) is None


def test_response_format_for_design_repair():
    long_p = DS.repair_prompt(DESIGN_PROMPT, "{}", ["e"])
    rf = LG.response_format_for(True, "design_repair", long_p)
    assert rf["type"] == "json_object" and rf["schema"]["required"]
    short_p = "You designed the type. Correct ONLY these fields.\nWords, in order: A / B.\n=== FIELDS TO CORRECT ===\n- x"
    assert LG.response_format_for(True, "design_repair", short_p) == {"type": "json_object"}       # json_mode only, as today
    assert LG.response_format_for(False, "design_repair", short_p) is None


def test_the_node_offers_every_schema_name():
    assert RS.NAMES == ("none", "design", "concept", "design_repair")
    with pytest.raises(ValueError):
        RS.schema_for("nope")


def test_concept_schema_knows_the_plate_and_angle_shot_keys_but_does_not_require_them():
    s = RS.concept_schema(3, 3)
    items = s["properties"]["shot_plan"]["prefixItems"] if "prefixItems" in s["properties"]["shot_plan"] else s["properties"]["shot_plan"]["items"]
    first = items[0] if isinstance(items, list) else items
    props = first["properties"]
    for k in ("same_subject_as", "framing", "plate", "type_matte", "subject_move", "plate_pan", "plate_subject"):
        assert k in props and k not in first["required"]
    assert props["type_matte"]["enum"][:3] == ["reveal_subject", "reveal_plate", "subject_through"] and {"window_other", "window_subject_front", "plate_in_subject", "plate_in_type"} <= set(props["type_matte"]["enum"])      # 2026-10-30: the contrast looks joined the enum
    assert RS.unsupported_keywords(s) == []
