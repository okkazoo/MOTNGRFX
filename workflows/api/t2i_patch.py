"""Patch the t2i API graphs in this folder: prompt, seed, size, LoRA.

    python t2i_patch.py moss_t2i.json out.json --prompt "..." --seed 7 --size 832x1216
    python t2i_patch.py moss_t2i_lora.json out.json --prompt "..." --lora moss_cndamiph_candid_iphone_v1.safetensors --strength 0.8 --trigger

Fields are found by class_type (and the PROMPT / NEGATIVE titles), so node ids
may change. --lora none removes the LoRA node and wires its consumers straight
to the UNet. --trigger prepends a LoRA's trigger words (LORA_TRIGGERS) to the prompt.
"""
import argparse
import json
from pathlib import Path

LORA_TRIGGERS = {
    "moss_cndamiph_candid_iphone_v1": "cndamiph",
}


def _nodes(g, cls):
    return [k for k, v in g.items() if v["class_type"] == cls]


def _titled(g, cls, title):
    return [k for k in _nodes(g, cls) if g[k].get("_meta", {}).get("title") == title]


def patch(g, prompt=None, seed=None, width=None, height=None, lora=None,
          strength=None, negative=None, ref=None, trigger=False):
    g = json.loads(json.dumps(g))
    if lora == "none":
        for k in _nodes(g, "LoraLoaderModelOnly"):
            src = g[k]["inputs"]["model"]
            del g[k]
            for v in g.values():
                for name, val in v["inputs"].items():
                    if val == [k, 0]:
                        v["inputs"][name] = src
    elif lora:
        short = lora.replace(".safetensors", "")
        name = lora if lora.endswith(".safetensors") else f"{short}.safetensors"
        for k in _nodes(g, "LoraLoaderModelOnly"):
            g[k]["inputs"]["lora_name"] = name
        trig = LORA_TRIGGERS.get(short)
        if trigger and prompt is not None and trig and trig not in prompt:
            prompt = f"{trig}, {prompt}"
    if strength is not None:
        for k in _nodes(g, "LoraLoaderModelOnly"):
            g[k]["inputs"]["strength_model"] = strength
    if prompt is not None:
        for k in _titled(g, "CLIPTextEncode", "PROMPT"):
            g[k]["inputs"]["text"] = prompt
        for k in _nodes(g, "TextEncodeQwenImageEditPlus"):
            g[k]["inputs"]["prompt"] = prompt
    if negative is not None:
        for k in _titled(g, "CLIPTextEncode", "NEGATIVE"):
            g[k]["inputs"]["text"] = negative
    if seed is not None:
        for k in _nodes(g, "KSampler"):
            g[k]["inputs"]["seed"] = seed
        for k in _nodes(g, "RandomNoise"):
            g[k]["inputs"]["noise_seed"] = seed
    for cls in ("EmptyLatentImage", "EmptyFlux2LatentImage", "ModelSamplingFlux"):
        for k in _nodes(g, cls):
            if width:
                g[k]["inputs"]["width"] = width
            if height:
                g[k]["inputs"]["height"] = height
    if ref:
        for k in _nodes(g, "LoadImage"):
            g[k]["inputs"]["image"] = ref
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("out")
    ap.add_argument("--prompt")
    ap.add_argument("--negative")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--size", help="WxH, multiples of 16")
    ap.add_argument("--lora", help="short name (rainywindow), file name, or none")
    ap.add_argument("--strength", type=float)
    ap.add_argument("--trigger", action="store_true", help="prepend the LoRA trigger words")
    ap.add_argument("--ref", help="reference image name in the box's input/ (styleref graph)")
    a = ap.parse_args()
    w = h = None
    if a.size:
        w, h = (int(x) for x in a.size.lower().split("x"))
    g = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    g = patch(g, a.prompt, a.seed, w, h, a.lora, a.strength, a.negative, a.ref, a.trigger)
    Path(a.out).write_text(json.dumps(g, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
