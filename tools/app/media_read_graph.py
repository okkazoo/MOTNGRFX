#!/usr/bin/env python
"""API graph of the READ step of the "Add media" box (16_app/MEDIA_BOX.md): the side graph `_media_read` the page queues before Make it.

    IN media (PrimitiveStringMultiline, the MEDIA JSON) -> MographMediaPack (contact sheet + prompt + info; ffmpeg / PIL / DSP, CPU)
        -> MographGGUFVL (Gemma 4 31B QAT, vision, json_mode) -> MographMediaReadParse (also gets info + media) -> PreviewAny "OUT read"

The model settings are the `gemma31b_qat` arm of tools/app/llm_arms.py (model, mmproj, KV q8_0, flash attention, all layers on the GPU, top_k 64 / top_p 0.95 /
repeat_penalty 1.0, projector ON: this call needs the vision); n_ctx 12288 (= READ_N_CTX, the larger of the read and story budgets: the SAME config as `_story` so the story reuses the loaded runner), temperature 0.3, max_tokens 1200, json_mode on,
keep_model_loaded TRUE (Gemma resident: unloaded after 10 min idle or when any other prompt starts, tools/app/llm_resident.py; the app graphs keep it false).
The page sets `IN media`.inputs.value and reads outputs["out_read"].text[0] (title `OUT read`) from the ws `executed` message / history.

    python tools/app/media_read_graph.py --write-template        # app/graphs/_media_read.json (a leading `_` keeps it out of the template dropdown)
    python tools/app/media_read_graph.py --media '{"video": ...}' --out g.json
    python <progress_standin.py> submit g.json --iid <id> --port <port>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
TEMPLATE = os.path.join(ROOT, "app", "graphs", "_media_read.json")

from tools.app import llm_arms as LA  # noqa: E402

# n_ctx of THIS graph only (not the gemma31b_qat arm: 32768 is sized for the design prompt of 18-20 k tokens). Rule: size n_ctx to the prompt, never trim the prompt for fit.
# Worst-case budget in tokens (3 chars/token, a safe over-estimate; the real counts are logged in the node's stats `prompt_tokens` on the box run): instructions + system ~850 (2.5 k chars),
# ONE contact sheet image <= 1200 (Gemma's largest image budget 1120 + margin), Whisper lyrics of a 120 s audio cap at 5 words/s ~1700, the user's pasted text up to 8000 chars ~2700,
# reply max_tokens 1200 = ~7.7 k -> the next size up of 8192 / 12288 / 16384 with headroom is 12288.
READ_BUDGET = dict(instructions=850, image=1200, lyrics=1700, user_text=2700, reply=1200)
READ_N_CTX = LA.RESIDENT_N_CTX                      # = 12288 = max(READ budget, story budget): ONE config for the read and the story graph (see llm_arms.RESIDENT_N_CTX)
SYSTEM_PROMPT = ("You are a careful reader of a user's media and pasted text. You answer with exactly one JSON object and nothing else. "
                 "You never reword or add words the user did not write.")


def llm_inputs(seed: int = 1, max_tokens: int = 1200, temperature: float = 0.3) -> dict:
    """the MographGGUFVL inputs both side graphs share (llm_arms.resident_inputs: gemma31b_qat arm, vision on, n_ctx RESIDENT_N_CTX, keep_model_loaded true = Gemma stays resident for the story)."""
    return LA.resident_inputs(seed, max_tokens, temperature)


def build(media: str = "{}", seed: int = 1) -> dict:
    """-> API-format graph (dict). `media` = the MEDIA JSON string (the page overwrites IN media.inputs.value)."""
    llm = llm_inputs(seed)
    llm.update(prompt=["pack", 1], image=["pack", 0], system_prompt=SYSTEM_PROMPT, json_mode=True)
    return {
        "media": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": media}, "_meta": {"title": "IN media"}},
        "pack": {"class_type": "MographMediaPack", "inputs": {"media": ["media", 0]}, "_meta": {"title": "Media contact sheet + prompt"}},
        "read_llm": {"class_type": "MographGGUFVL", "inputs": llm, "_meta": {"title": "Gemma 4 31B QAT Q4_K_XL media read"}},
        "parse": {"class_type": "MographMediaReadParse", "inputs": {"reply": ["read_llm", 0], "info": ["pack", 2], "media": ["media", 0]}, "_meta": {"title": "READ JSON v1"}},
        "out_read": {"class_type": "PreviewAny", "inputs": {"source": ["parse", 0]}, "_meta": {"title": "OUT read"}},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--media", default="{}")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-template", action="store_true")
    a = ap.parse_args(argv)
    out = TEMPLATE if a.write_template else a.out
    if not out:
        ap.error("give --out or --write-template")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(a.media, a.seed), f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
