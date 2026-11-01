"""Openings (user 2026-10-29: "why do all the compositions start with the graphics coming in with a similar transition from the top?"): every piece of the 24 GB batches opened with
word 1 = form_in lines + module lanes = strands falling in from the top. tools/fusion/openings.py + its plumbing (schema, salvage, concept request, run_batch, design prompt). CPU, no network.
Run: python -m pytest tests/test_openings.py -q"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import openings as OP  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto",
       "seed": 5, "quality": "preview", "format": "9:16"}


def W(i, form_in="lines", entry=None, module="lanes"):
    w = {"text": f"W{i}", "form_in": form_in, "form_out": "lines", "module": module}
    if entry:
        w["entry"] = entry
    return w


# ------------------------------------------------------------------------------------------------ the rule
def test_every_word_of_the_batches_opened_the_same_way_and_the_rule_ends_it():
    ws = [W(i) for i in range(6)]                                           # what the models wrote: lines, no entry named (= the engine's silent "up"), lanes
    fx = []
    n = OP.enforce(ws, fx, seed=3)
    keys = [OP.key_of(w) for w in ws]
    assert len(set(keys)) == 6 and n >= 6 and all(isinstance(w.get("entry"), str) for w in ws if OP.style_of(w) == "lines")
    assert sum(1 for w in ws if OP.style_of(w) == "lines" and OP.effective_entry(w) == "up") <= 1                  # "up" is the last in the rotation, not the design
    assert fx and all(f.startswith("type.words[") for f in fx)
    assert OP.enforce(ws, [], seed=3) == 0                                      # idempotent


def test_a_named_entry_is_kept_duplicates_and_a_single_style_are_mended():
    ws = [W(0, entry="left"), W(1, entry="left"), W(2, "blobs"), W(3, "blobs")]
    fx = []
    OP.enforce(ws, fx, seed=0)
    assert ws[0]["entry"] == "left" and OP.key_of(ws[1]) != OP.key_of(ws[0])                              # the second 'lines from the left' changed
    assert len({OP.key_of(w) for w in ws if OP.style_of(w) == "lines"}) == 2                             # the two `lines` words differ in side
    assert OP.style_of(ws[2]) == OP.style_of(ws[3]) == "blobs"                                           # a repeated side-less style is the piece's own pair: it is NOT turned into `lines` (the lines-from-the-top bias)
    one = [W(i, "lines", entry=s) for i, s in enumerate(("left", "right", "down"))]
    OP.enforce(one, [], seed=0)
    assert len({OP.style_of(w) for w in one}) >= 2                                                       # a piece never uses one io style only
    keep = [W(0, "wipe"), W(1, "lines", "left")]
    assert OP.enforce(keep, [], seed=0) == 0


def test_word_one_does_not_open_like_the_piece_before_it():
    earlier = [dict(form_in="lines", entry="left", module="lanes", cut="depth_pivot")]
    ws = [W(0, entry="left"), W(1, "blobs"), W(2, entry="down")]
    fx = []
    OP.enforce(ws, fx, earlier=earlier, seed=0)
    assert OP.key_of(ws[0]) != ("lines", "left") and any("the piece before" in f for f in fx)
    ws2 = [W(0, "blobs"), W(1, entry="right")]
    assert OP.enforce(ws2, [], earlier=earlier, seed=0) == 0                                              # a different opening is left alone
    prot = [W(0, entry="left"), W(1, entry="left")]
    OP.enforce(prot, [], earlier=earlier, seed=0, protected={0})
    assert prot[0]["entry"] == "left"


def test_opening_summary_roundtrip_and_prompt_text():
    piece = {"type": {"words": [dict(text="A", form_in="lines", entry="left", module="pipework")]}, "compose": {"cuts": [{"after": "s1", "cut": "hard", "was": "window_other"}]}}
    o = OP.opening_of(piece)
    assert o == dict(word="A", form_in="lines", entry="left", module="pipework", cut="window_other")
    line = "piece: mode=cut shots=3 " + OP.summary(o)
    assert OP.parse_summary(line) == dict(form_in="lines", entry="left", module="pipework", cut="window_other") and OP.parse_summary("piece: nothing") is None
    t = OP.prompt_text([OP.parse_summary(line)])
    assert "lines entering from left on pipework" in t and "Open THIS piece differently" in t and OP.prompt_text([]) == ""
    assert OP.opening_of({"type": {"words": [dict(text="A", form_in="blobs")]}})["entry"] == "up"          # the engine's default side is what an unnamed entry means


# ------------------------------------------------------------------------------------------------ the bias is out of the schema, the defaults and the fold
def test_the_schema_and_menu_rotate_their_enumerations_and_offer_entry():
    a, b = QP.schema_text(["A", "B"], 10.0, None, 0), QP.schema_text(["A", "B"], 10.0, None, 2)
    assert a != b and '"entry"' in a and "no two words share a form_in and an entry side" in a
    i0 = a.index("form_in\": \"<EXACTLY one of: ") + len("form_in\": \"<EXACTLY one of: ")
    firsts = {QP.schema_text(["A", "B"], 10.0, None, s)[i0:i0 + 5] for s in range(5)}
    assert len(firsts) >= 4                                                                          # not `lines` first in every prompt
    assert QP.modules_block(0).splitlines()[0] != QP.modules_block(1).splitlines()[0] and "`entry`" in QP.modules_block(0)


def test_a_missing_form_in_or_module_is_not_lines_or_lanes(tmp_path):
    from test_design_salvage import REQ as DREQ
    music = synth_music(160, 12)
    plan = PR.plan_piece(DREQ, music)
    reply = DS.reply_from_piece(plan, music)
    for w in reply["type"]["words"]:
        w.pop("form_in", None)
        w.pop("form_out", None)
        w.pop("module", None)
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        d, fixes, errors = QP.normalize_design(reply, name, plan_windows=True)
    ws = d["type"]["words"]
    assert len({OP.key_of(w) for w in ws}) >= 2                                                       # nothing opened with one way in for every word (side-less styles may repeat: the piece's own pair)
    assert not all(w["form_in"] == "lines" and OP.effective_entry(w) == "up" for w in ws) and len({w["module"] for w in ws}) == 1       # ONE system per piece
    assert not any(e.startswith("type.words[") and "form_in" in e for e in errors)


def test_entry_and_exit_sides_are_read_and_mended():
    music = synth_music(160, 12)
    from test_design_salvage import REQ as DREQ
    plan = PR.plan_piece(DREQ, music)
    reply = DS.reply_from_piece(plan, music)
    ws = reply["type"]["words"]
    for w in ws[:3]:
        w["form_in"], w["module"] = "lines", "lanes"          # the random system now draws side-less opening styles (flicker / particles) for this request; an entry side only exists for `lines`
    ws[0]["entry"], ws[1]["entry"], ws[2]["entry"] = "left", "from the top", "sideways"
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "P" * 500)
    got = json.loads(out["piece_json"])["type"]["words"]
    assert got[0]["entry"] == "left"
    assert (got[1].get("entry") in OP.SIDES or got[1]["form_in"] != "lines") and got[1].get("entry") != "sideways"                    # "from the top" -> up (the synonym), then the uniqueness rule may move it
    assert got[2].get("entry") != "sideways"


def test_the_fold_keeps_the_models_own_module_choice_not_lanes_first():
    ws = [dict(text="A", font="Ultra-Regular.ttf", module="pipework"), dict(text="B", font="Ultra-Regular.ttf", module="pipework"), dict(text="C", font="Ultra-Regular.ttf", module="circuit"),
          dict(text="D", font="Ultra-Regular.ttf", module="lanes"), dict(text="E", font="Ultra-Regular.ttf", module="topo_tower")]
    DD.one_module_system(ws, [])
    assert {w["module"] for w in ws} == {"pipework", "circuit", "lanes"} and ws[4]["module"] in ("pipework", "circuit", "lanes")      # 2026-11-01: up to MAX_MODULES (4) distinct modules stay: lanes is no longer folded away, topo_tower (3D, avoided) is
    many = [dict(text=t, font="Ultra-Regular.ttf", module=m) for t, m in zip("ABCDEF", ("pipework", "pipework", "circuit", "lanes", "slats", "beat_bars"))]
    DD.one_module_system(many, [])
    assert len({w["module"] for w in many}) == DD.MAX_MODULES == 4 and many[5]["module"] == "slats" and [w["module"] for w in many[:5]] == ["pipework", "pipework", "circuit", "lanes", "slats"]      # the cap holds: the 5th distinct module goes to its family's kept one
    allbad = [dict(text="A", font="Ultra-Regular.ttf", module="hollow_tubes"), dict(text="B", font="Ultra-Regular.ttf", module="voxel_roll")]
    DD.one_module_system(allbad, [])
    assert allbad[0]["module"] == allbad[1]["module"] and allbad[0]["module"] not in DD.AVOID_MODULES


# ------------------------------------------------------------------------------------------------ across the pieces of a batch
def test_earlier_openings_ride_concept_request_plan_and_design_prompt_and_a_repeat_is_swapped():
    prev = [dict(words=["ONE"], subject="x", scene="y", mood="calm", music="m", shots="cut", opening=dict(form_in="lines", entry="left", module="lanes", cut="depth_pivot"))]
    c = dict(owners=[], shot_plan=[], idea="", earlier_openings=prev[0:1] and [prev[0]["opening"]])
    extras = CO.request_extras(c)
    assert json.loads(extras)["earlier_openings"][0]["entry"] == "left"
    req = PR.normalize_request(dict(REQ, owners=extras))
    assert req["earlier_openings"] == [prev[0]["opening"]]
    music = synth_music(120, 16)
    plan = PR.plan_piece(dict(REQ, owners=extras), music)
    assert plan["plan"]["earlier_openings"] == [prev[0]["opening"]]
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        prompt = QP.build_prompt(name, 1)
    assert "EARLIER PIECES OF THIS BATCH opened like this" in prompt and "lines entering from left on lanes" in prompt
    reply = DS.reply_from_piece(plan, music)
    for w in reply["type"]["words"]:
        w["form_in"], w["entry"] = "lines", "left"                                                    # the model repeats the earlier opening for every word
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "P" * 500)
    got = json.loads(out["piece_json"])["type"]["words"]
    assert OP.key_of(got[0]) != ("lines", "left") and len({OP.key_of(w) for w in got}) >= 3       # (a 3D cut sets the sides of the words around it; side-less styles may repeat)
    assert "opening=" in out["report"] and OP.parse_summary(out["report"])["form_in"] == OP.style_of(got[0])
    from tools.app import run_batch as RB
    r = RB.record_run({"i": 1, "concept": {"words": ["A"]}}, {"status": "done", "stages": []}, {"OUT design final check": [out["report"]], "OUT concept": [json.dumps({"words": ["A"]})]})
    assert r["concept"]["opening"] == OP.parse_summary(out["report"]) and r["opening"]["form_in"] == OP.style_of(got[0])
    assert RB.previous_of({"runs": [dict(r, i=1), {"i": 2}]}, 2)[0]["opening"] == r["opening"]       # the next run is handed it
