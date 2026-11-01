"""Concept generation by design_llm_8b (NO CLAUDE): the model invents the REQUEST of a piece; thin wrappers over ``tools/fusion/concept.py``.

    MographConceptPrompt  seed, length, previous (JSON list of earlier concepts), brief   -> prompt
    MographConceptParse   reply, previous, seed, length, prompt, brief                    -> ok, repair_prompt, report
    MographConceptFinal   reply, ok, repair_reply (LAZY, only when ok is false), previous, seed, length, brief
                                                                                        -> words, scene, music, look, shots, concept_json, report, owners

Graph (fusion_design_qwen_auto): ConceptPrompt -> DesignLLM_Advanced (text only) -> ConceptParse -> [DesignLLM_Advanced repair, lazy]
-> ConceptFinal -> MographRequest -> the whole fusion_design_qwen pipeline. `brief` (optional, READ JSON v1, 16_app/MEDIA_BOX.md): the user's on-screen lines become the concept's words
verbatim (code-enforced after the parse), their concept / look are kept in `idea`, the media notes steer the model; empty = today's behaviour. The length is an input of the graph (the batch driver fixes it), the
number of shots stays code-decided. A clash with `previous` (same words / scene subject / overlapping music tags / same mood) goes back to the model
once; if that fails the seeded rules variation is used, so the output is always a unique, valid request.
"""
from __future__ import annotations

import json
import time

from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY

ensure_root_on_path()
from tools.fusion import concept as CO      # noqa: E402


def _wired(cls, name: str) -> bool:
    hid = getattr(cls, "hidden", None)
    prompt = getattr(hid, "prompt", None)
    if not prompt:
        return True
    return isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)


class MographConceptPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographConceptPrompt", display_name="MOTNGRFX Concept Prompt", category=CATEGORY,
            description="The instruction for design_llm to invent the request of one piece (words, scene, music tags, look, shots mode, mood), "
                        "with the earlier concepts to differ from and seeded starting cues. Wire into DesignLLM custom_prompt (no image).",
            inputs=[io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF),
                    io.Int.Input("length", default=16, min=6, max=60, tooltip="Seconds; set by the caller, the model does not choose it."),
                    io.String.Input("previous", multiline=True, default="[]", tooltip="JSON list of the earlier concepts (the batch driver fills it)."),
                    io.String.Input("brief", multiline=True, default="", optional=True, tooltip="READ JSON v1 (16_app/MEDIA_BOX.md): the user's on-screen lines (used verbatim), concept, look and what their media show. Empty = the model invents everything.")],
            outputs=[io.String.Output(display_name="prompt")])

    @classmethod
    def execute(cls, seed, length, previous="[]", brief="") -> io.NodeOutput:
        return io.NodeOutput(CO.concept_prompt(int(seed), int(length), CO.parse_previous(previous), effects=CO.parse_effects(previous), brief=brief or ""))


class MographConceptParse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographConceptParse", display_name="MOTNGRFX Concept Parse", category=CATEGORY,
            description="Validates the model's concept (5-7 UPPER CASE lines, one subject, scene, 8-14 music tags, mood, look, shots mode) and forces uniqueness "
                        "against `previous`: same words, same scene subject, overlapping music tags or the same mood is a clash. ok = clean and unique; "
                        "repair_prompt = the prompt + the problem list for one more model call.",
            inputs=[io.String.Input("reply", multiline=True, default=""),
                    io.String.Input("previous", multiline=True, default="[]"),
                    io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF), io.Int.Input("length", default=16, min=6, max=60),
                    io.String.Input("prompt", multiline=True, default="", optional=True),
                    io.String.Input("brief", multiline=True, default="", optional=True, tooltip="READ JSON v1 (16_app/MEDIA_BOX.md): the user's on-screen lines (used verbatim), concept, look and what their media show. Empty = the model invents everything.")],
            outputs=[io.Boolean.Output(display_name="ok"), io.String.Output(display_name="repair_prompt"), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, reply, previous, seed, length, prompt="", brief="") -> io.NodeOutput:
        prev = CO.parse_previous(previous)
        r = CO.parse(reply, prev, int(seed), int(length), CO.parse_effects(previous), brief or "")
        problems = r["errors"] + r["clashes"]
        rp = "" if r["ok"] else CO.repair_prompt(prompt or CO.concept_prompt(int(seed), int(length), prev, effects=CO.parse_effects(previous), brief=brief or ""), reply, problems)
        rep = f"concept parse: ok={r['ok']}" + "".join(f"\n  problem: {p}" for p in problems) + "".join(f"\n  fix: {f}" for f in r["fixes"])
        print(f"[MographConceptParse] {rep}")
        return io.NodeOutput(bool(r["ok"]), rp, rep)


class MographConceptFinal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographConceptFinal", display_name="MOTNGRFX Concept Final", category=CATEGORY,
            description="The request fields of the piece. ok = true: the first reply as parsed (repair_reply is never requested, the second model call never runs). "
                        "ok = false: the repair reply if it is clean and unique, else the seeded rules variation. Always valid and unique against `previous`.",
            inputs=[io.String.Input("reply", multiline=True, default=""),
                    io.Boolean.Input("ok", default=True),
                    io.String.Input("repair_reply", multiline=True, default="", optional=True, lazy=True),
                    io.String.Input("previous", multiline=True, default="[]"),
                    io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF), io.Int.Input("length", default=16, min=6, max=60),
                    io.String.Input("brief", multiline=True, default="", optional=True, tooltip="READ JSON v1 (16_app/MEDIA_BOX.md): the user's on-screen lines (used verbatim), concept, look and what their media show. Empty = the model invents everything.")],
            outputs=[io.String.Output(display_name="words"), io.String.Output(display_name="scene"), io.String.Output(display_name="music"),
                     io.String.Output(display_name="look"), io.String.Output(display_name="shots"), io.String.Output(display_name="concept_json"),
                     io.String.Output(display_name="report"), io.String.Output(display_name="owners")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id])

    @classmethod
    def check_lazy_status(cls, reply=None, ok=None, repair_reply=None, previous=None, seed=None, length=None, brief=None) -> list[str]:
        if ok is not None and not ok and repair_reply is None and _wired(cls, "repair_reply"):
            return ["repair_reply"]
        return []

    @classmethod
    def execute(cls, reply, ok, previous, seed, length, repair_reply=None, brief="") -> io.NodeOutput:
        t0 = time.time()
        prev = CO.parse_previous(previous)
        eff = CO.parse_effects(previous)
        first = CO.parse(reply, prev, int(seed), int(length), eff, brief or "")
        c, rep = CO.final(None if ok else repair_reply, first, prev, int(seed), int(length), eff, brief or "")
        print(f"[MographConceptFinal] {time.time() - t0:.2f}s {rep}\n  words: {' / '.join(c['words'])}\n  scene: {c['scene']}\n  music: {c['music']}\n  shots: {c['shots']}\n  owners: {c.get('owners') or 'none (the rules decide)'}")
        return io.NodeOutput("\n".join(c["words"]), c["scene"], c["music"], c["look"], c["shots"], json.dumps(c, ensure_ascii=False), rep,
                              CO.request_extras(c))


NODES = [MographConceptPrompt, MographConceptParse, MographConceptFinal]
