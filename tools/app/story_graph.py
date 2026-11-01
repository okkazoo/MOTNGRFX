#!/usr/bin/env python
"""API graph of the STORY step of the "Add media" box (16_app/MEDIA_BOX.md "STORY step"): the side graph `_story` the page queues after each completed read (and on a Length change).

    IN story (PrimitiveStringMultiline, {"read": READ JSON, "length", "seed", "format"}) -> MographStoryPrompt
        -> MographGGUFVL (Gemma 4 31B QAT, TEXT ONLY: mmproj "none" like the concept call, json_mode) -> MographStoryParse (also gets story_in) -> PreviewAny "OUT story"

The model settings are the `gemma31b_qat` arm of tools/app/llm_arms.py (model, KV q8_0, flash attention, all layers on the GPU, top_k 64 / top_p 0.95 / repeat_penalty 1.0);
n_ctx = the shared resident 12288 (llm_arms.RESIDENT_N_CTX = max of the read budget 12288 and this graph's STORY_BUDGET need), mmproj ON like the read graph, keep_model_loaded TRUE
(Gemma resident, tools/app/llm_resident.py), temperature 0.7, max_tokens 1500, json_mode on.
The page sets `IN story`.inputs.value and reads outputs["out_story"].text[0] (title `OUT story`) from the ws `executed` message / history.

    python tools/app/story_graph.py --write-template        # app/graphs/_story.json (a leading `_` keeps it out of the template dropdown)
    python tools/app/story_graph.py --story '{"read": ...}' --out g.json
    python <progress_standin.py> submit g.json --iid <id> --port <port>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
TEMPLATE = os.path.join(ROOT, "app", "graphs", "_story.json")

from tools.app import llm_arms as LA  # noqa: E402
from tools.fusion import story as ST  # noqa: E402

# n_ctx of THIS graph only (not the gemma31b_qat arm: 32768 is sized for the design prompt of 18-20 k tokens). Rule: size n_ctx to the prompt, never trim the prompt for fit.
# Worst-case budget in tokens (3 chars/token, a safe over-estimate; the real counts are logged in the node's stats `prompt_tokens` on the box run): instructions + system ~1100 (the fixed text of
# story.story_prompt incl. the reply shape for 6 shots), the 5 picture entries (shows 240 + mood + palette ~ 420 chars each) ~700, Whisper lyrics of a 120 s audio ~1700 (as the READ budget),
# the user's direction (concept 600 + look 300 + 12 on-screen lines of 160 chars ~ 2800 chars) ~930, reply max_tokens 1500 (6 beats of <= 320 chars + lines + header, indented JSON)
# -> 7.8k (v2, text pasted) + 1.5 k headroom = 9.3 k -> the next size up of 12288, which is also the read graph's: ONE shared config (llm_arms.RESIDENT_N_CTX).
STORY_BUDGET = dict(instructions=1300, pictures=700, lyrics=1700, direction=930, text=1700, reply=1500)      # v2: camera rule + reply shape, the numbered pasted text (400 words x ~4 tokens)
STORY_N_CTX = LA.RESIDENT_N_CTX                      # 12288 = the shared resident config (llm_arms), >= the read's AND this graph's own budget (STORY_BUDGET + 1.5 k headroom needs >= 9.4 k)
MAX_TOKENS = STORY_BUDGET["reply"]
TEMPERATURE = 0.7


def llm_inputs(seed: int = 1, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> dict:
    """the MographGGUFVL inputs of `_media_read` (llm_arms.resident_inputs): the SAME load config, so a story after a read reuses the loaded Gemma. The projector is loaded but unused (no image is linked)."""
    return LA.resident_inputs(seed, max_tokens, temperature)


def build(story: str = "{}", seed: int = 1) -> dict:
    """-> API-format graph (dict). `story` = the story request JSON string (the page overwrites IN story.inputs.value)."""
    llm = llm_inputs(seed)
    llm.update(prompt=["prompt", 0], system_prompt=ST.SYSTEM_PROMPT, json_mode=True)
    return {
        "story": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": story}, "_meta": {"title": "IN story"}},
        "prompt": {"class_type": "MographStoryPrompt", "inputs": {"story_in": ["story", 0]}, "_meta": {"title": "Story prompt"}},
        "story_llm": {"class_type": "MographGGUFVL", "inputs": llm, "_meta": {"title": "Gemma 4 31B QAT Q4_K_XL story (text only)"}},
        "parse": {"class_type": "MographStoryParse", "inputs": {"reply": ["story_llm", 0], "story_in": ["story", 0]}, "_meta": {"title": "STORY JSON v1"}},
        "out_story": {"class_type": "PreviewAny", "inputs": {"source": ["parse", 0]}, "_meta": {"title": "OUT story"}},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--story", default="{}")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-template", action="store_true")
    a = ap.parse_args(argv)
    out = TEMPLATE if a.write_template else a.out
    if not out:
        ap.error("give --out or --write-template")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(a.story, a.seed), f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
