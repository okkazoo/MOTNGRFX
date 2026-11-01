#!/usr/bin/env python
"""API graph of the "Gemma music reading" step: LoadImage (the timeline PNG) + the prompt text -> MographGGUFVL (Gemma 4 31B QAT, vision) -> PreviewAny x2
(reply, stats). The model settings are the ones of the shipped default graph's `design_qwen` node (app/graphs/fusion_design_qwen_auto_layers_pivot_gemma31b_qat.json):
31B QAT UD-Q4_K_XL + mmproj-F16, all layers on the GPU, n_ctx 32768, KV q8_0, flash attention; only prompt / image / sampling differ here
(temperature 0.5 for a reading, json_mode on: the llama.cpp JSON grammar, so the reply is one JSON object).

    python tools/app/music_read_graph.py --image music_read_timeline_v001.png --prompt-file v001/prompt.txt --out v001/graphs/music_read.json [--seed 1]
    python ~/Documents/Dev/comfyui-projects/tools/progress_standin.py submit v001/graphs/music_read.json --iid <id> --port 9000 --idea <P/I/vNNN> --why "..."

The PNG must already be in the box's ComfyUI/input/ (scp). Reply text = history outputs of node `out_reply` (the stats json of `out_stats`); fetch_outputs() reads them.
No Claude / Qwen / API anywhere in this graph.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEMMA_MODEL = "gemma-4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
GEMMA_MMPROJ = "gemma-4-31b-qat/mmproj-F16.gguf"
SYSTEM_PROMPT = ("You are a careful music analyst and film editor. You answer with exactly one JSON object and nothing else. "
                 "Every time you write comes from the numbers you are given; you never invent a time that is not in the beat grid.")


def build(image: str, prompt_text: str, seed: int = 1, max_tokens: int = 6000, temperature: float = 0.5, n_ctx: int = 32768, json_mode: bool = True,
          keep_model_loaded: bool = False) -> dict:
    """-> API-format graph (dict). `image` = a file name in the box's ComfyUI/input/."""
    return {
        "image": {"class_type": "LoadImage", "inputs": {"image": image}, "_meta": {"title": "IN timeline image"}},
        "music_read": {
            "class_type": "MographGGUFVL",
            "_meta": {"title": "Gemma 4 31B QAT Q4_K_XL music reading"},
            "inputs": {
                "prompt": prompt_text, "model": GEMMA_MODEL, "mmproj": GEMMA_MMPROJ, "n_gpu_layers": -1, "n_ctx": int(n_ctx), "max_tokens": int(max_tokens),
                "temperature": float(temperature), "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.0, "seed": int(seed), "kv_cache": "q8_0", "flash_attn": True,
                "mmproj_on_gpu": True, "n_cpu_moe": 0, "handler": "auto", "image_max_tokens": -1, "strip_thinking": True, "keep_model_loaded": bool(keep_model_loaded),
                "verbose": False, "system_prompt": SYSTEM_PROMPT, "json_mode": bool(json_mode), "image": ["image", 0],
            },
        },
        "out_reply": {"class_type": "PreviewAny", "inputs": {"source": ["music_read", 0]}, "_meta": {"title": "OUT music reading reply"}},
        "out_stats": {"class_type": "PreviewAny", "inputs": {"source": ["music_read", 1]}, "_meta": {"title": "OUT music reading stats"}},
    }


def fetch_outputs(prompt_id: str, port: int = 9000, host: str = "127.0.0.1") -> dict:
    """-> {'reply': str, 'stats': dict|str} from /history/<prompt_id> of the (tunnelled) ComfyUI. Read-only."""
    with urllib.request.urlopen(f"http://{host}:{port}/history/{prompt_id}", timeout=30) as r:
        h = json.loads(r.read().decode("utf-8"))
    outs = h[prompt_id]["outputs"]

    def text(node):
        t = outs.get(node, {}).get("text")
        return "".join(t) if isinstance(t, list) else (t or "")

    st = text("out_stats")
    try:
        st = json.loads(st)
    except ValueError:
        pass
    return {"reply": text("out_reply"), "stats": st, "status": h[prompt_id].get("status", {}).get("status_str")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, help="file name in the box's ComfyUI/input/")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.5)
    a = ap.parse_args(argv)
    g = build(a.image, open(a.prompt_file, encoding="utf-8").read(), a.seed, a.max_tokens, a.temperature)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
