"""MographStoryPrompt / MographStoryParse: the STORY step of the "Add media" box (16_app/MEDIA_BOX.md "STORY step"). Logic: tools/fusion/story.py (pure); graph: tools/app/story_graph.py.

    story_in ({"read": READ JSON, "length", "seed", "format"})  -> MographStoryPrompt -> prompt STRING
    prompt -> MographGGUFVL (Gemma 4 31B QAT, TEXT ONLY, json_mode)                    -> reply
    reply + story_in                                             -> MographStoryParse  -> story STRING (STORY JSON v1)

No model call in either node.
"""
from __future__ import annotations

import json

from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY

ensure_root_on_path()
from tools.fusion import story as ST   # noqa: E402

_TIP = '{"read": <READ JSON v1>, "length": 12, "seed": 1, "format": "9:16"}'


class MographStoryPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographStoryPrompt",
            display_name="MOTNGRFX Story Prompt",
            category=CATEGORY,
            description="The page's story request (the edited READ JSON + Length + seed + format) -> the prompt that makes Gemma write a short story that fits the user's pictures in order, "
                        "one beat and one on-screen line per shot (shot count = the planner's for the Length).",
            inputs=[io.String.Input("story_in", multiline=True, default="{}", tooltip=_TIP)],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def fingerprint_inputs(cls, story_in="{}", **kw):
        return story_in

    @classmethod
    def execute(cls, story_in: str = "{}") -> io.NodeOutput:
        return io.NodeOutput(ST.story_prompt(story_in))


class MographStoryParse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographStoryParse",
            display_name="MOTNGRFX Story Parse (STORY JSON v1)",
            category=CATEGORY,
            description="Gemma's reply + the story request -> STORY JSON v1 with the code guarantees: the planner's shot count, image k in shot k, the user's on-screen lines verbatim and in order, "
                        "the READ's concept / look winning, times on the music's bar grid, a rule story (beats from the pictures' descriptions) when the reply is unusable.",
            inputs=[io.String.Input("reply", multiline=True, default=""), io.String.Input("story_in", multiline=True, default="{}", tooltip=_TIP)],
            outputs=[io.String.Output(display_name="story")],
        )

    @classmethod
    def execute(cls, reply: str = "", story_in: str = "{}") -> io.NodeOutput:
        return io.NodeOutput(json.dumps(ST.parse(reply, story_in), indent=1, ensure_ascii=False))


NODES = [MographStoryPrompt, MographStoryParse]
