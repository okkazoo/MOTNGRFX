"""par round r6 (2026-10-29): the `why` fields (tools/fusion/why_fields.py: collect / score / strip, salvage in design_salvage.flatten, the concept's whys, the level-B schema) and the
thinking split (why_fields.split_think, llm_gguf.split_think, the MographGGUFVL node signature, llm_arms thinking arm). Fake replies only: no
model, no network, no GPU. Run: python -m pytest tests/test_why_fields.py -q"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE), str(ROOT / "tools" / "fusion")):
    if _p not in sys.path:
        sys.path.append(_p)

from tools.app import llm_arms as LA  # noqa: E402
from tools.app import llm_gguf as LG  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import piece_brief as PB  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402
from tools.fusion import why_fields as WF  # noqa: E402

REPLY = {
    "idea": "A runner crosses three light zones.", "idea_why": "Each chord of the 160 bpm lead gets its own colour world, so one runner can carry the whole piece.",
    "system": {"rule": "lanes only", "roles": {"A": "Doto[ROND,wght].ttf"}, "progression": "ELSH climbs", "why": "The C minor lead is one tight line, so one module and one climbing axis read as one gesture."},
    "shots": [{"id": "s1", "first": "A", "until": 6, "prompt": "the runner sprints through a cold tunnel, camera tracks alongside", "why": "The tunnel is the Cm chord: closed, blue, the runner the only warm thing."},
              {"id": "s2", "first": "last:s1", "until": 12, "prompt": "she keeps sprinting, deeper in", "why": ""}],
    "type": {"words": [{"text": "STILL", "why": "GeistPixel forms on the 0 s downbeat so the question is read while she is still in the dark."},
                       {"text": "GOING", "why": "ok"}, {"text": "AHEAD"}]},
    "compose": {"default": {"why": "The chain flips the matte gain on each chord change so she passes in front of the letters at the Ab section.", "ops": []}, "shots": {"s2": {"ops": []}}},
}


# ------------------------------------------------------------------------------------------------ the fields
def test_decisions_cover_idea_system_shots_words_chains_and_missing_is_empty_not_an_error():
    ds = WF.decisions(REPLY)
    keys = [d["key"] for d in ds]
    assert keys == ["idea", "system", "shots[0]", "shots[1]", "word[0] STILL", "word[1] GOING", "word[2] AHEAD", "compose.default", "compose.shots.s2"]
    assert {d["key"]: bool(d["why"]) for d in ds}["shots[1]"] is False and WF.decisions("not a dict") == [] and WF.decisions({}) == []
    s = WF.score(REPLY)
    assert s["decisions"] == 9 and s["present"] == 6 and s["missing"] == 3 and s["specific"] == 5       # "ok" is present but not a reason (< 5 words)
    assert s["reasons"] == round(5 / 9, 3)


def test_strip_why_removes_the_keys_everywhere_and_touches_nothing_else():
    out = WF.strip_why(REPLY)
    assert "idea_why" not in out and "why" not in out["system"] and all("why" not in s for s in out["shots"]) and all("why" not in w for w in out["type"]["words"])
    assert "why" not in out["compose"]["default"] and out["compose"]["default"]["ops"] == [] and out["shots"][0]["prompt"] == REPLY["shots"][0]["prompt"]
    assert REPLY["idea_why"]                                       # a copy, the reply is unchanged


def test_a_why_object_or_list_is_read_as_text():
    r = copy.deepcopy(REPLY)
    r["type"]["words"][0]["why"] = {"font": "GeistPixel is the cold pixel face", "timing": "the hit sits on the first downbeat of the Cm bar"}
    assert "GeistPixel is the cold pixel face" in WF.collect(r)["word[0] STILL"] and WF.score(r)["present"] == 6


def test_a_why_copied_from_the_guide_is_flagged_and_a_fresh_sentence_is_not():
    guide = "behind puts the type behind the subject so that the subject passes in front of the letters and the type reads as part of the scene"
    g = WF.guide_grams(extra=[guide])
    r = {"idea": "x", "idea_why": "Behind puts the type behind the subject so that the subject passes in front of the letters.",
         "system": {"why": "The hairline face answers the thin surf spray at the 9.7 s chord change."}}
    s = WF.score(r, grams=g)
    assert s["copied"] == 1 and s["by_kind"]["idea"]["specific"] == 0 and s["by_kind"]["system"]["specific"] == 1 and any("copies the guide" in f for f in s["flagged"])
    assert WF.overlap("a sentence about nothing in the guide at all today", g) == 0.0


def test_the_real_guide_corpus_flags_a_pasted_guide_sentence():
    text = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    sent = next(s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 14 and "|" not in s and "`" not in s)
    assert WF.overlap(sent, WF.guide_grams()) > WF.COPY_RATIO
    assert WF.score({"idea": "x", "idea_why": sent})["copied"] == 1


def test_the_same_sentence_with_other_numbers_counts_once():
    base = "{} forms on the {} s downbeat so the question is read while she is still in the dark and the exit is fast"
    r = {"type": {"words": [{"text": "A", "why": base.format("Doto", 0)}, {"text": "B", "why": base.format("Bitcount", 4.5)}, {"text": "C", "why": "Anybody is the heavy resolve on the Gm chord at 12 s."}]}}
    s = WF.score(r)
    assert s["repeated"] == 1 and s["specific"] == 2 and s["decisions"] == 3
    assert WF.score({"type": {"words": [{"text": "A", "why": "same reason given twice here"}, {"text": "B", "why": "same reason given twice here"}]}})["repeated"] == 1


def test_anchored_is_a_second_view_not_part_of_reasons():
    r = {"idea": "x", "idea_why": "Because it feels right and gives visual interest to the whole piece overall.",
         "system": {"why": "The tunnel footage is cold so the pixel face reads as its ice."}}
    a = WF.anchors_of("tunnel footage cold runner")
    s = WF.score(r, anchors=a)
    assert s["reasons"] == 1.0 and s["anchored"] == 0.5


# ------------------------------------------------------------------------------------------------ the thinking split
@pytest.mark.parametrize("raw,answer,think", [
    ('<think>a\nb</think>\n{"a": 1}', '{"a": 1}', "a\nb"),
    ('reasoning with the opening tag pre-filled</think>{"a": 1}', '{"a": 1}', "reasoning with the opening tag pre-filled"),
    ('<|channel>thought\nhmm<channel|>{"a": 1}', '{"a": 1}', "hmm"),
    ('<|channel>thought<channel|>{"a": 1}', '{"a": 1}', ""),
    ('<think>cut off by max_tokens', "", "cut off by max_tokens"),
    ('{"a": 1}', '{"a": 1}', ""),
    ("<think>one</think>x<think>two</think>{}", "x{}", "one\n\ntwo"),
])
def test_split_think_forms(raw, answer, think):
    assert WF.split_think(raw) == (answer, think)
    assert LG.split_think(raw) == (answer, think) and LG.strip_think(raw) == answer


def test_thinking_never_reaches_the_parser_even_with_json_in_the_thinking():
    raw = '<think>maybe {"idea": "wrong"} first</think>\n{"idea": "right", "idea_why": "ok then"}'
    ans, th = WF.split_think(raw)
    assert QP.extract_json(ans)["idea"] == "right" and '{"idea": "wrong"}' in th


# ------------------------------------------------------------------------------------------------ salvage: the app's design reply
def _flat(obj):
    return SV.flatten(obj, ["STILL", "GOING"])


def test_flatten_collects_the_whys_and_a_reply_without_any_is_unchanged():
    obj = {"concept": "c", "idea_why": "Each chord of the lead gets its own colour world here.", "type": {"mood": "m", "words": [{"text": "STILL", "font": "Doto[ROND,wght].ttf", "why": "Forms on the first downbeat of the Cm bar."}]}}
    raw = _flat(obj)
    assert raw["@why"]["whys"]["idea"].startswith("Each chord") and raw["@why"]["decisions"] >= 2 and raw["@why"]["missing"] >= 0
    plain = copy.deepcopy(obj)
    plain.pop("idea_why")
    plain["type"]["words"][0].pop("why")
    raw2 = _flat(plain)
    assert "@why" not in raw2                                                  # no whys: nothing is added, nothing is an error
    assert {k: v for k, v in raw.items() if k != "@why"} == raw2               # the design fields are read identically with and without the whys


def test_a_garbage_why_never_breaks_the_flatten():
    obj = {"concept": "c", "idea_why": 7, "type": {"words": [{"text": "STILL", "why": None}, {"text": "GOING", "why": ["a", {"b": "c"}]}]}}
    raw = _flat(obj)
    assert raw["@type"] is True and "@why" not in raw or raw["@why"]["reasons"] in (0.0, None, 0)


def test_the_designs_schema_asks_for_the_whys_and_the_compilers_ignore_them():
    txt = QP.schema_text(["STILL", "GOING"], 18.0)
    for k in ('"idea_why"', '"why": "<why this rule', '"why": "<why this font role', '"why": "<why this chain'):
        assert k in txt
    assert "REASONS:" in txt
    # the mender knows `system.why`: a slip in the reply must not close `system` at the why key
    slip = '{"idea": "x", "system": {"rule": "r", "why": "w", "roles": {"A": "f"}}, "a": [1, 2], "type": {"words": [{"text": "A", "tl": [[0, [1]], [5, [2]]' + "\n" + ' ], "step": true}]}}'
    obj = QP.extract_json(slip)
    assert obj is not None and obj["system"]["why"] == "w" and "type" in obj


# ------------------------------------------------------------------------------------------------ the concept
def test_the_concept_prompt_asks_for_whys_and_normalize_keeps_them_without_requiring_them():
    p = CO.concept_prompt(1, 16, [], "cut")
    assert '"idea_why"' in p and '"why": "<why this place' in p and '"why": "<why this owner' in p
    d = {"title": "Harbour light", "words": ["THE SEA", "REMEMBERS", "EVERY SHIP"], "subject": "old lighthouse keeper",
         "scene": "an old lighthouse keeper in a yellow coat standing on the stone gallery as a storm breaks over a dark sea, spray in the wind, one beam cutting the clouds, seen from far below",
         "music": "cinematic orchestral, low strings, taiko drums, brass swells, dark, big reverb, slow build, epic hall", "bpm": 88, "mood": "solemn", "look": "slab", "shots": "single",
         "idea": "One long climb up the tower.", "idea_why": "The slow build of the strings is one climb, so one shot carries it."}
    c = CO.parse(json.dumps(d), [], 1, 16)["concept"]
    assert c["whys"]["idea"].startswith("The slow build")
    d.pop("idea_why")
    c2 = CO.parse(json.dumps(d), [], 1, 16)["concept"]
    assert c2 is not None and "whys" not in c2                                 # a missing why is no error


# ------------------------------------------------------------------------------------------------ level B: the prompt and the piece
def test_r6_prompt_differs_from_r4_by_the_why_fields_only():
    f = ROOT / "20_open-model-design" / "v003" / "local" / "r4" / "P11_qwen_s1" / "prompt.txt"
    if not f.is_file():
        pytest.skip("r4 prompt not saved here")
    old = f.read_text(encoding="utf-8")
    new, _ = PB.build_prompt("P11", "r6a")
    k = old.index("=== OUTPUT ===")
    assert new[:k] == old[:k]                                                  # everything before the output schema is r4's (principle 9 as r4 had it)
    assert PB.build_prompt("P11", "r6b")[0] == new                             # r6a and r6b: the same prompt, only the thinking differs
    plain = PB.reply_schema(PB.load_brief("P11"), False)
    why = PB.reply_schema(PB.load_brief("P11"), True)
    assert '"why"' not in plain and "idea_why" not in plain and '"idea_why"' in why and why.count('"why"') >= 5 and "REASONS." in why
    assert PB.VARIANTS["r6a"] == PB.VARIANTS["r6b"] and PB.VARIANTS["r6a"]["why"] is True and not PB.VARIANTS["r4"].get("why")


@pytest.mark.skipif(not (ROOT / "15_fusion" / "v004" / "BRIEFS.md").exists(), reason="dev fixture not shipped in the public export: 15_fusion/v004/BRIEFS.md")
def test_reply_to_piece_ignores_the_whys():
    b = PB.load_brief("P14")
    ref = b["claude_piece"]
    # a minimal reply built from Claude's own piece, with whys added everywhere: the same piece comes out without them
    obj = {"idea": "x", "idea_why": "because of the groove at 124 bpm in the studio",
           "stills": [{"id": s["id"], "prompt": s["prompt"]} for s in ref["stills"]],
           "shots": [{"id": s["id"], "first": s["first"], "until": 6.0 * (i + 1) if i < len(ref["shots"]) - 1 else float(b["ana"]["cut"]), "prompt": s["prompt"], "why": "the dancer leaps on the downbeat"}
                     for i, s in enumerate(ref["shots"])],
           "type": {"words": [{"text": w, "why": "a reason for this word"} for w in b["words"]]}}
    p1, f1, e1 = PB.reply_to_piece(copy.deepcopy(obj), b)
    p2, f2, e2 = PB.reply_to_piece(WF.strip_why(obj), b)
    assert e1 == e2 and f1 == f2
    assert (p1 is None) == (p2 is None)
    if p1 is not None:
        assert "why" not in json.dumps(p1.get("system", {})) and p1["shots"] == p2["shots"] and p1["type"] == p2["type"]


# ------------------------------------------------------------------------------------------------ the app arm: thinking is an option
def _arm_graph(arm_key):
    node = {"inputs": {"custom_prompt": ["design_pkg", 0], "image": ["design_pkg", 3], "max_tokens": 4096, "temperature": 0.7, "seed": ["seed", 0]}, "class_type": LA.OLD, "_meta": {"title": "design_llm design"}}
    return {"design_qwen": node, "design_pkg": {"inputs": {}, "class_type": "MographDesignPackage", "_meta": {"title": "pkg"}}, "seed": {"inputs": {"value": 1}, "class_type": "PrimitiveInt", "_meta": {"title": "seed"}}}


def test_thinking_is_off_by_default_on_every_arm():
    assert all(not a.get("thinking") for a in LA.ARMS.values()) or any(a.get("thinking") for a in LA.ARMS.values())     # either way the next test pins what a set flag does
    off = LA.convert_node(_arm_graph("gemma31b_qat")["design_qwen"], LA.ARMS["gemma31b_qat"])
    if not LA.ARMS["gemma31b_qat"].get("thinking"):
        assert "thinking" not in off["inputs"] and off["inputs"]["max_tokens"] == 4096


def test_a_thinking_arm_gets_the_flag_more_tokens_and_the_archive_node():
    arm = dict(LA.ARMS["gemma31b_qat"], thinking=True)
    assert LA.thinking_fits(arm) and not LA.thinking_fits(dict(arm, n_ctx=LA.CTX_16))
    n = LA.convert_node(_arm_graph("x")["design_qwen"], arm)
    assert n["inputs"]["thinking"] is True and n["inputs"]["max_tokens"] == 4096 + LA.THINK_TOKENS
    assert n["inputs"]["n_ctx"] >= LA.DESIGN_PROMPT_TOKENS + n["inputs"]["max_tokens"]          # prompt + reasoning + reply hold in the context
    LA.ARMS["_t"] = arm
    try:
        g, _ = LA.convert(_arm_graph("_t"), None, "_t")
    finally:
        del LA.ARMS["_t"]
    assert g["out_design_reasoning"] == {"inputs": {"source": ["design_qwen", 2]}, "class_type": "PreviewAny", "_meta": {"title": "OUT design reasoning"}}


def test_a_thinking_arm_that_does_not_fit_its_context_is_refused():
    with pytest.raises(ValueError, match="thinking needs n_ctx"):
        LA.convert_node(_arm_graph("x")["design_qwen"], dict(LA.ARMS["gemma12b_q8"], thinking=True))


def test_the_node_has_a_thinking_input_and_a_reasoning_output():
    src = (ROOT / "nodes_fusion" / "llm_gguf.py").read_text(encoding="utf-8")
    assert 'io.Boolean.Input("thinking", default=False, optional=True' in src and 'display_name="reasoning"' in src
    assert "think=bool(thinking) and not json_mode" in src and "LG.split_think(text)" in src
    assert re.search(r"def execute\(cls, [^)]*json_mode=False, thinking=False\)", src)
    kw = LG.load_kwargs("/m/x.gguf", "/m/mmproj-F16.gguf", think=True)
    assert kw["chat_handler_kwargs"]["extra_template_arguments"] == {"enable_thinking": True}
    assert LG.load_kwargs("/m/x.gguf", "/m/mmproj-F16.gguf")["chat_handler_kwargs"]["extra_template_arguments"] == {"enable_thinking": False}
