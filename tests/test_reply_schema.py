"""reply_schema (tools/fusion/reply_schema.py, 16_app/REPLY_SCHEMA.md): the JSON Schemas of the design and concept replies, built from the code's own constants; saved good replies
validate, known-bad samples fail, the schemas stay inside what llama.cpp's json-schema-to-grammar converter handles, and the runner passes the schema as response_format.
No GPU, no model, no network. The GBNF test runs only where llama_cpp is importable (or $REPLY_SCHEMA_CONVERTER names a copy of llama_cpp/llama_grammar.py).
Run: python -m pytest tests/test_reply_schema.py -q"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import llm_gguf as LG  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402
from tools.fusion import reply_schema as RS  # noqa: E402

jsonschema = pytest.importorskip("jsonschema")
WORDS = ["HELLO", "BIG WORLD"]
APP0 = os.path.join(ROOT, "20_open-model-design", "v003", "local", "app0")


def _word(text, **kw):
    w = {"text": text, "why": "the hit lands on the second downbeat because the bass drops there", "font": QP.liked_fonts()[0]["file"], "module": "lanes", "hit": 1.5,
         "hold_s": 1.6, "form_s": 0.6, "exit_s": 0.6, "form_in": "lines", "form_out": "wipe", "placement": {"yaw": 14, "wf": 0.7, "hf": 0.3}}
    w.update(kw)
    return w


def good_design(words=WORDS) -> dict:
    f = [x["file"] for x in QP.liked_fonts()]
    return {
        "concept": "type that follows the runner", "idea_why": "the footage runs left to right", "treatment_notes": "behind the subject",
        "system": {"rule": "two faces alternate", "roles": {"A": f[0], "B": f[1]}, "pattern": ["A", "B"], "yaw": [18, -18], "why": "contrast between neighbours"},
        "type": {"mood": "tense", "energy": 0.6, "palette": {"bg": [10, 10, 12], "fg": [240, 235, 220], "accents": [[255, 120, 20], [20, 180, 255], [200, 200, 40]]},
                 "camera": "hold_drift", "transitions": ["ripple"] if len(words) == 2 else "auto", "line": {"px": 3},
                 "words": [_word(w, **({"font": f[1], "axes": {"wght": [200, 700, "beat"]}, "render": "flat", "matte": "front", "lines": [w]} if i else {})) for i, w in enumerate(words)]},
        "compose": {"technique": "behind", "plate_gain": 0.9, "plate_sat": 1.0, "glow_lo": 0.05, "glow_hi": 0.3, "glow_color": [0.9, 0.7, 0.4], "grain": 0.02, "vignette": 0.2,
                    "type_opacity": 0.95, "holds": [[3.2, 0.4]], "cuts": [{"after": "s1", "cut": "window_other", "word": 0, "why": "the letters open onto the next place"}]},
    }


FLAG = RS.shot_flag_default()                  # same_place today (continues in the older prompts)


def good_concept(n=3) -> dict:
    shots = [{"prompt": "a runner crosses a salt flat at dawn while the camera tracks beside", "camera": "tracking shot", "why": "the runner anchors the frame"}]
    shots += [{"prompt": "the same runner in a flooded street, rain on the lens", "camera": "slow push", "why": "a new place", FLAG: bool(k % 2)} for k in range(1, n)]
    return {"title": "SALT AND RAIN", "words": ["KEEP GOING", "RAIN", "HOME"][:n], "subject": "lone runner", "scene": "a runner on a salt flat at dawn, low sun, long shadows, wide shot, nothing else in view",
            "music": "cinematic, driving drums, cello, warm, tense", "bpm": 96, "mood": "restless", "look": "strand", "shots": "cut", "idea": "a runner moves through places and the words ride with him",
            "idea_why": "the run gives the words a direction", "shot_plan": shots,
            "owners": [{"render": "flat", "why": "short and clear"}, {"render": "handoff", "ltx_at": 0.4, "material": "wet stone", "why": "one key word"}, {"render": "flat", "why": "plain"}][:n]}


def kinds(vs):
    return {(v["path"], v["kind"]) for v in vs}


# ------------------------------------------------------------------------------------------------- the schemas build
def test_schemas_build_and_are_json():
    for s in (RS.design_schema(), RS.design_schema(WORDS, ["behind", "traced"], ["s1", "s2"]), RS.design_schema(WORDS, lock_text=False), RS.concept_schema(),
              RS.concept_schema(3, 3), RS.concept_schema(free_counts=True), RS.type_passes_schema()):
        json.dumps(s)
        jsonschema.Draft202012Validator.check_schema(RS.validator_schema(s))
        assert RS.unsupported_keywords(s) == []


def test_enums_come_from_the_code():
    d = RS.design_schema(WORDS)
    w = d["properties"]["type"]["properties"]["words"]["items"][0]["properties"]
    assert w["module"]["enum"] == QP.WORD_MODULES and w["form_in"]["enum"] == QP.IO_STYLES and w["matte"]["enum"] == QP.MATTES and w["render"]["enum"] == list(OW.RENDERS)
    assert w["font"]["enum"] == [f["file"] for f in QP.liked_fonts()] and len(w["font"]["enum"]) == 42
    assert d["properties"]["type"]["properties"]["camera"]["enum"] == QP.CAMERAS
    assert d["properties"]["compose"]["properties"]["technique"]["enum"] == QP.TECHNIQUES
    from tools.fusion import spec as SP
    assert tuple(RS.TYPE_PASSES) == tuple(SP.PASSES)
    from tools.fusion import cut_kinds as CK
    cuts = d["properties"]["compose"]["properties"]["cuts"]["items"]["properties"]["cut"]["enum"]
    assert set(cuts) <= set(CK.ALL) and "window_other" in cuts and "pan:left" in cuts and not any(k.startswith("depth_dolly") for k in cuts)
    c = RS.concept_schema(3, 3)["properties"]
    from tools.fusion import concept as C
    assert c["look"]["enum"] == C.LOOKS and c["shots"]["enum"] == C.MODES and (c["bpm"]["minimum"], c["bpm"]["maximum"]) == C.BPM_RANGE


def test_ranges_are_the_code_clamps():
    p = RS.design_schema(WORDS)["properties"]["type"]["properties"]["words"]["items"][0]["properties"]
    assert (p["hold_s"]["minimum"], p["hold_s"]["maximum"]) == QP.RANGES["hold_s"]
    assert (p["placement"]["properties"]["yaw"]["minimum"], p["placement"]["properties"]["yaw"]["maximum"]) == QP.RANGES["yaw"]
    assert (p["ltx_at"]["minimum"], p["ltx_at"]["maximum"]) == OW.LTX_AT_RANGE


# ------------------------------------------------------------------------------------------------- good replies validate, known-bad ones fail
def test_good_replies_validate():
    assert RS.violations(good_design(), RS.design_schema(WORDS, shot_ids=["s1", "s2"])) == []
    assert RS.violations(good_design(), RS.design_schema()) == []
    assert RS.violations(good_concept(), RS.concept_schema(3, 3)) == []
    assert RS.violations(good_concept(1)["words"] and good_concept(3), RS.concept_schema(free_counts=True)) == []
    assert RS.violations(["depth", "line"], RS.type_passes_schema()) == []


@pytest.mark.parametrize("name", ["new_gemma_s1", "new_qwen_s1", "old_gemma_s1", "old_qwen_s1"])
def test_saved_design_replies_have_no_wrong_types(name):
    d = os.path.join(APP0, name)
    if not os.path.isfile(os.path.join(d, "reply.txt")):
        pytest.skip("archive not on this checkout")
    reply = QP.extract_json(open(os.path.join(d, "reply.txt"), encoding="utf-8").read())
    prompt = open(os.path.join(d, "prompt.txt"), encoding="utf-8").read()
    ctx = RS.design_context(prompt)
    assert ctx["words"] and ctx["techniques"] and ctx["shot_ids"]
    vs = RS.violations(reply, RS.design_schema(require_why=False, require_system=False, **ctx))
    assert [v for v in vs if v["kind"] in RS.VALUE and "'particles' is not one of" not in v["msg"]] == [], vs      # an archived reply that still names the retired `particles` (2026-11-01: no speckle letters) is swapped at plan time, not a type error


def test_saved_concept_replies_have_no_wrong_types():
    for name in ("concept_gemma_s1", "concept_qwen_s1"):
        d = os.path.join(APP0, name)
        if not os.path.isfile(os.path.join(d, "reply.txt")):
            pytest.skip("archive not on this checkout")
        reply = QP.extract_json(open(os.path.join(d, "reply.txt"), encoding="utf-8").read())
        ctx = RS.concept_context(open(os.path.join(d, "prompt.txt"), encoding="utf-8").read())
        assert ctx.get("n_words")
        vs = RS.violations(reply, RS.concept_schema(require_why=False, **ctx))
        assert [v for v in vs if v["kind"] in RS.VALUE] == [], vs


def test_type_passes_string_fails():
    s = RS.type_passes_schema()
    for bad in ("all", "depth+line", ["all"], ["depth", "bogus"], []):
        assert RS.violations(bad, s), bad


def _bad(mut, words=WORDS):
    r = good_design(words)
    mut(r)
    return kinds(RS.violations(r, RS.design_schema(words, ["behind", "traced"], ["s1", "s2"])))


def test_known_bad_design_samples_fail():
    assert ("type.words[].form_in", "enum") in _bad(lambda r: r["type"]["words"][0].update(form_in="gradient_fade"))
    assert ("type.words[].matte", "enum") in _bad(lambda r: r["type"]["words"][0].update(matte="behind"))
    assert ("type.words[].font", "enum") in _bad(lambda r: r["type"]["words"][0].update(font="Big ShouldersDisplay[wght].ttf"))
    assert ("type.words[].hold_s", "type") in _bad(lambda r: r["type"]["words"][0].update(hold_s="1.6s"))
    assert ("type.words[].text", "const") in _bad(lambda r: r["type"]["words"][0].update(text="HELLO!"))
    assert ("type.words[].why_owner", "extra") in _bad(lambda r: r["type"]["words"][0].update(why_owner="x"))
    assert {p for p, _ in _bad(lambda r: r["type"].update(transitions="braid, twist"))} == {"type.transitions"}                    # a string of names is not a list
    assert _bad(lambda r: r["type"].update(transitions=["ripple", "twist"]))                        # 2 entries for 1 gap
    assert ("type.palette.bg[]", "type") in _bad(lambda r: r["type"]["palette"].update(bg=[0.1, 0.2, 0.3]))
    assert ("type.palette.bg[]", "int_range") in _bad(lambda r: r["type"]["palette"].update(bg=[300, 0, 0]))
    assert ("compose.technique", "enum") in _bad(lambda r: r["compose"].update(technique="built_in"))
    assert ("compose.holds[]", "type") in _bad(lambda r: r["compose"].update(holds=[{"time": 1, "freeze_s": 0.3}]))
    assert ("compose.cuts[].cut", "enum") in _bad(lambda r: r["compose"]["cuts"][0].update(cut="depth_dolly:right"))
    assert ("compose.cuts[].after", "enum") in _bad(lambda r: r["compose"]["cuts"][0].update(after="s9"))
    assert ("type.words", "length") in _bad(lambda r: r["type"]["words"].pop())
    assert ("compose", "missing") not in _bad(lambda r: None) and ("compose.technique", "missing") in _bad(lambda r: r["compose"].pop("technique"))
    assert ("type_passes", "extra") in _bad(lambda r: r.update(type_passes="all"))                    # the key cannot be written at all in a design reply


def test_range_violations_are_soft():
    vs = RS.violations({**good_design(), "type": {**good_design()["type"], "energy": 3.0}}, RS.design_schema(WORDS))
    assert [(v["path"], v["kind"]) for v in vs] == [("type.energy", "range")] and "range" not in RS.HARD


def test_known_bad_concept_samples_fail():
    def bad(mut, n=3):
        r = good_concept(n)
        mut(r)
        return kinds(RS.violations(r, RS.concept_schema(n, n)))
    assert ("bpm", "type") in bad(lambda r: r.update(bpm="96"))
    assert ("bpm", "int_range") in bad(lambda r: r.update(bpm=240))
    assert ("look", "enum") in bad(lambda r: r.update(look="neon"))
    assert ("shots", "enum") in bad(lambda r: r.update(shots="montage"))
    assert ("words", "length") in bad(lambda r: r["words"].pop())
    assert ("words[]", "pattern") in bad(lambda r: r["words"].__setitem__(0, "keep going"))
    assert ("words[]", "pattern") in bad(lambda r: r["words"].__setitem__(0, "A / B / C"))
    assert ("owners[].render", "enum") in bad(lambda r: r["owners"][0].update(render="3d"))
    assert (f"shot_plan[].{FLAG}", "type") in bad(lambda r: r["shot_plan"][1].update({FLAG: "yes"}))
    assert (f"shot_plan[].{FLAG}", "missing") in bad(lambda r: r["shot_plan"][1].pop(FLAG))
    assert ("shot_plan[].continues", "extra") in bad(lambda r: r["shot_plan"][1].update(continues=True)) or FLAG == "continues"
    assert ("owners", "type") in bad(lambda r: r.update(owners={"render": "flat"}))


def _prop_names(n, acc=None):
    acc = set() if acc is None else acc
    if isinstance(n, dict):
        for k, v in n.items():
            if k == "properties":
                acc.update(v)
            _prop_names(v, acc)
    elif isinstance(n, list):
        for x in n:
            _prop_names(x, acc)
    return acc


def test_every_key_the_prompts_ask_for_is_in_the_schema():
    """DRIFT GUARD: the schemas are closed (an undeclared key cannot be written), so a key added to qwen_piece.schema_text / design_options / concept.concept_prompt without
    reply_schema would become IMPOSSIBLE for the model to write. This fails then: add the key to tools/fusion/reply_schema.py."""
    import re
    d = RS.design_schema(["HELLO", "WORLD"])
    text = QP.schema_text(["HELLO", "WORLD"], 10.0)
    ask = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', text)) - {"A", "B"}                 # A / B: the roles of system.roles (a map)
    assert ask - _prop_names(d) == set()
    from tools.fusion import concept as C
    ask_c = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', C.concept_prompt(1, 16, [], "cut")))
    assert ask_c - _prop_names(RS.concept_schema(3, 3)) == set()
    assert FLAG in _prop_names(RS.concept_schema(3, 3))


def test_schema_for_reads_counts_from_the_prompt():
    p = "Words, in order, each shown once: ONE / TWO WORDS / THREE\n"
    s = RS.schema_for("design", p)
    assert len(s["properties"]["type"]["properties"]["words"]["items"]) == 3
    assert s["properties"]["type"]["properties"]["words"]["items"][1]["properties"]["text"] == {"const": "TWO WORDS"}
    cp = "`words` has EXACTLY 4 entries: x. `shot_plan` has EXACTLY 4 entries, one per shot"
    assert RS.concept_context(cp) == {"n_words": 4, "n_shots": 4}
    assert RS.schema_for("concept", cp)["properties"]["words"]["maxItems"] == 4
    assert RS.schema_for("none") is None and RS.schema_for(None) is None
    with pytest.raises(ValueError):
        RS.schema_for("bogus")


# ------------------------------------------------------------------------------------------------- the grammar
def test_gbnf_conversion_succeeds():
    conv = RS.converter()
    if conv is None:
        pytest.skip("llama_cpp is not importable and $REPLY_SCHEMA_CONVERTER is not set")
    for s in (RS.design_schema(WORDS, ["behind", "traced"], ["s1", "s2"]), RS.design_schema(), RS.concept_schema(3, 3), RS.concept_schema(), RS.type_passes_schema()):
        g = conv(json.dumps(s))
        assert g.startswith(tuple("abcdefghijklmnopqrstuvwxyz")) or "root ::=" in g
        assert "root ::=" in g
    g = RS.to_gbnf(RS.design_schema(WORDS))
    rule = next(l for l in g.splitlines() if l.startswith("type-words-tuple-0-form-in ::="))
    assert sorted(re.findall(r'\\"(\w+)\\"', rule.split("::=")[1])) == sorted(QP.IO_STYLES)      # the grammar names the allowed io styles, nothing else
    assert "gradient_fade" not in rule and "behind" not in next(l for l in g.splitlines() if l.startswith("type-words-tuple-0-matte ::="))
    assert g.count("type-words-tuple-") > 20 and "type-words-tuple-2" not in g                                                   # exactly two word entries


# ------------------------------------------------------------------------------------------------- the runner and the node
class _FakeLlm:
    n_tokens = 10

    def __init__(self):
        self.kw = None

    def n_ctx(self):
        return 4096

    def create_chat_completion(self, **kw):
        self.kw = kw
        yield {"choices": [{"delta": {"content": "{}"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _chat(**kw):
    r = LG.Runner()
    r.llm = _FakeLlm()
    r.chat("Words, in order, each shown once: A / B\n", **kw)
    return r.llm.kw.get("response_format")


def test_runner_passes_the_schema_as_response_format():
    assert _chat() is None
    assert _chat(json_mode=True) == {"type": "json_object"}
    assert _chat(schema="none") is None
    rf = _chat(schema="design")
    assert rf["type"] == "json_object" and [w["properties"]["text"]["const"] for w in rf["schema"]["properties"]["type"]["properties"]["words"]["items"]] == ["A", "B"]
    assert _chat(json_mode=True, schema="concept")["schema"]["required"][0] == "title"
    d = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert _chat(schema=d) == {"type": "json_object", "schema": d}
    assert LG.response_format_for(False, "off") is None


def test_node_exposes_the_schema_option():
    src = open(os.path.join(ROOT, "nodes_fusion", "llm_gguf.py"), encoding="utf-8").read()
    assert 'io.Combo.Input("schema", options=list(RS.NAMES), default="none", optional=True' in src
    assert "schema=schema" in src and "think=bool(thinking) and not json_mode and schema == \"none\"" in src and RS.NAMES == ("none", "design", "concept", "design_repair")


def test_archive_measurement_runs():
    if not os.path.isfile(os.path.join(ROOT, "20_open-model-design", "replies.jsonl")):
        pytest.skip("archive not on this checkout")
    acc = RS.measure_archive()
    assert acc and all(a["n"] > 0 for a in acc.values())
    assert "| model | reply |" in RS.measure_table(acc)
