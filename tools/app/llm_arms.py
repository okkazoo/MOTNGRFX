#!/usr/bin/env python
"""LLM arms for the app graph: swap the design_llm_8b 4-bit concept / design / repair nodes (`DesignLLM_Advanced`, transformers + bitsandbytes, outside
ComfyUI's memory manager, 4/8-bit only) for a bigger or higher-bit GGUF model run by llama.cpp (`MographGGUFVL`, `nodes_fusion/llm_gguf.py`), sized to sit at
the edge of the 12 GB card. Docs, probe results and the recommendation: tools/app/llm_arms.py. Open weights, local, NO CLAUDE, no API.

A pure function of the built app graph (+ its `.app.json` sidecar), node ids and titles' stage groups kept so
run_timing / run_batch / the dashboard read the converted graph unchanged; `build_graphs.py` is not edited.

    python tools/app/llm_arms.py                      # write app/graphs/<base>_<arm>.json (+ .app.json) for every arm x base graph on disk
    python tools/app/llm_arms.py --check              # build in memory, validate, compare with the files on disk, write nothing
    python tools/app/llm_arms.py --arm gemma12b_q6    # one arm
    python tools/app/llm_arms.py list                 # arms: model file, size, licence, layers
    python tools/app/llm_arms.py probe-graph --arm X --dir 16_app/bench/llm_arms/run03 --out g.json   # the one-model probe graph (sheet + prompt in, reply out)
    python tools/app/llm_arms.py fetch --arm X        # shell lines that download the arm's files (aria2c -x4, one file at a time)

Conversion of every DesignLLM_Advanced node (concept, concept repair, design, design repair):
    custom_prompt -> prompt        image -> image (the design node only: the ONE combined sheet of MographDesignPackage)
    seed, temperature, max_tokens -> same values;  top_p / top_k / repeat_penalty -> the arm's (the values the probes were measured with);  quantization / attention_mode / device / frame_count ... dropped
    n_gpu_layers, n_ctx (arm; concept nodes get CONCEPT_CTX), kv_cache, mmproj_on_gpu, n_cpu_moe, handler, image_max_tokens, top_k -> from the arm
    keep_model_loaded stays false (LTX / ACE / Alpha-Gen load in between, and llama.cpp is outside ComfyUI's Dynamic VRAM).
Output slot 0 (response) stays slot 0, so every consumer link is untouched; slot 1 (stats json) is new.

REPLY SCHEMA GRAMMAR (arm flag `reply_schema`, default OFF; 16_app/REPLY_SCHEMA.md): with it on, the converted nodes also get the `schema` input of MographGGUFVL so llama.cpp can only
sample replies that keep the TYPE and VALUE of every field inside the schema (tools/fusion/reply_schema.py). By node role (read from where the prompt comes from):
    concept call (prompt <- concept_prompt)          schema "concept"
    main design call (prompt <- design_pkg, image)   schema "design"      (an arm with `thinking` keeps "none": the grammar forbids the reasoning)
    design repair (prompt <- design_parse)           schema "design_repair"  = "design" when the prompt is the LONG repair (whole design prompt + rejected reply), none for the SHORT
                                                     repair (its reply is a {path: value} map; ONE graph node serves both, decided by the prompt text at run time)
    concept repair                                   no schema (json_mode as before)
An A/B pair is built with `--reply-schema`: the same graph, `_schema` appended to its name (`<base>_<arm>_schema.json`); the plain graph is untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import ltx_ic  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402

GRAPHS = os.path.join(ROOT, "app", "graphs")
FILE_BASES = ("fusion_design_qwen", "fusion_design_qwen_auto")                    # the graphs written to app/graphs/<base>_<arm>.json by default
BASES = FILE_BASES + ("fusion_redesign",)                                         # convertible too (the archive redesign graph of the A/B worker): --base fusion_redesign
OLD = "DesignLLM_Advanced"
NEW = "MographGGUFVL"
READ = "MographLtxRead"                    # the VLM legibility check node (nodes_fusion/layers.py): the same llama.cpp settings as the design node, no prompt / image input
JSON_MODE_TOKENS = 1024                    # concept / repair replies of a json_mode arm: no reasoning text in front, but room for the whole object
CONCEPT_CTX = 16384                       # image-less nodes: concept prompts are short, but the design REPAIR (also no image) carries the whole guide (~10k tokens; 8192 crashed batch-3060 v004 run 1)
CTX_16 = 24576                            # 16 GB tier (TIERS.md section 9): the app's design call is 14-18 k prompt tokens + a 1.5-5 k reply, 16384 no longer holds it; KV q8_0 at 24576 adds ~0.3 GiB (est.)
CTX_24 = 32768                            # 24 GB tier: also holds the full brief-level par prompt (19.7 k + up to 9 k reply); KV q8_0 +0.6 GiB over 16384 (est.), projector stays on the GPU (21.8 of 23.99 GiB est.)
# THINKING (par round r6, PAR.md section 9): an arm with `thinking: True` renders the chat template with enable_thinking for the DESIGN call (the one with the image; the image-less
# concept / repair calls keep the JSON grammar, which forbids reasoning) and gets THINK_TOKENS more max_tokens. The reasoning comes out of output slot 2 (the response stays slot 0, so the parser
# never sees it) and is archived through an `out_design_reasoning` PreviewAny. Off unless the arm sets it: `thinking_fits` is the n_ctx check (prompt + reasoning + reply must fit).
THINK_TOKENS = 8192                       # extra max_tokens for the reasoning (r6b measured: see PAR.md section 9)
DESIGN_PROMPT_TOKENS = 18000              # the app's design prompt incl. the sheet: 14-18 k (PAR.md section 7)
DESIGN_REPLY_TOKENS = 5000                # a 6-word piece with why fields and chains: 4-5 k
GGUF_DIR = "LLM/GGUF"                     # under ComfyUI/models
ltx_ic.CLASSES.setdefault(NEW, ("ComfyUI-Mograph", 2))
ltx_ic.CLASSES.setdefault(OLD, ("the design-LLM node pack", 1))
for _k, _v in {"LoadImage": ("ComfyUI core", 2), "PrimitiveStringMultiline": ("ComfyUI core", 1), "PreviewAny": ("ComfyUI core", 0)}.items():
    ltx_ic.CLASSES.setdefault(_k, _v)
try:                                                        # the app classes (Mograph*, ...) register themselves in build_graphs (EXTRA_CLASSES)
    from tools.app import build_graphs as _bg           # noqa: F401
    APP_CLASSES_KNOWN = True
except Exception:                                           # noqa: BLE001  (another worker may be editing it)
    APP_CLASSES_KNOWN = False
try:                                                        # the three-layer graphs' classes (MographLayerPrompt / Clean / Gate / Gather) register themselves in layered_shots
    from tools.app import layered_shots as _ls          # noqa: F401
except Exception:                                           # noqa: BLE001
    pass

# ---------------------------------------------------------------------------------------------- the arms
# `fits` / `n_gpu_layers` / `kv_cache` / `mmproj_on_gpu` are the MEASURED settings (tools/app/llm_arms.py, probe on the RTX 3060 12 GB); `repo` + `file` + `mmproj_file`
# are where the files come from (huggingface.co/<repo>/resolve/main/<file>); `dir` = folder under ComfyUI/models/LLM/GGUF.
ARMS: dict[str, dict] = {
    "gemma12b_q6": dict(
        label="Gemma 4 12B Q6_K", family="gemma", dir="gemma-4-12b", repo="unsloth/gemma-4-12b-it-GGUF", licence="Apache-2.0",
        file="gemma-4-12b-it-Q6_K.gguf", mmproj_file="mmproj-F16.gguf", gib=9.11, mmproj_gib=0.16,
        n_gpu_layers=-1, kv_cache="q8_0", mmproj_on_gpu=True, n_cpu_moe=0, handler="auto", top_k=64, top_p=0.95, repeat_penalty=1.0, n_ctx=16384, image_max_tokens=-1,
        why="mid-size dense model at 6 bits"),
}
# ---- tier arms (2026-10-28): the design model picked per VRAM tier, see 16_app/TIERS.md. NOT MEASURED: `measured=False`, `vram_est_gib` is an ESTIMATE from the 3060 probes
# (weights minus the ~0.77 GiB token-embedding table llama.cpp keeps on the CPU, + ~2.2 GiB for KV q8_0 at n_ctx 16384 + compute buffer + CUDA context; + the projector
# and ~0.5 GiB of image-encode buffer where it runs on the GPU). `tier` ties the arm to TIERS; `bases` = the base graphs the arm is built for by default.
TIER_BASES = ("fusion_design_qwen_auto_layers",)
ARMS.update({
    "gemma12b_q8": dict(
        label="Gemma 4 12B Q8_0", family="gemma", dir="gemma-4-12b", repo="unsloth/gemma-4-12b-it-GGUF", licence="Apache-2.0",
        file="gemma-4-12b-it-Q8_0.gguf", mmproj_file="mmproj-F16.gguf", gib=11.80, mmproj_gib=0.16,
        n_gpu_layers=-1, kv_cache="q8_0", mmproj_on_gpu=True, n_cpu_moe=0, handler="auto", top_k=64, top_p=0.95, repeat_penalty=1.0, n_ctx=CTX_16, image_max_tokens=-1,
        tier=16, measured=False, vram_est_gib=13.8, bases=TIER_BASES,
        why="16 GB speed alternative: the Gemma 12B that measured 54 s / 93 % at Q6_K, now 8-bit, everything on the GPU with ~2.4 GiB spare"),
    "gemma31b_qat": dict(
        label="Gemma 4 31B QAT Q4_K_XL", family="gemma", dir="gemma-4-31b-qat", repo="unsloth/gemma-4-31B-it-qat-GGUF", licence="Apache-2.0",
        file="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", mmproj_file="mmproj-F16.gguf", gib=16.10, mmproj_gib=1.12,
        n_gpu_layers=-1, kv_cache="q8_0", mmproj_on_gpu=True, n_cpu_moe=0, handler="auto", top_k=64, top_p=0.95, repeat_penalty=1.0, n_ctx=CTX_24, image_max_tokens=-1,
        tier=24, measured=False, vram_est_gib=20.4, bases=TIER_BASES,
        why="24 GB alternative: the largest Gemma 4 dense at its quantization-aware 4-bit build (a Q5/Q6 31B does not fit with the projector); other family than the pick"),
})
# tier -> what a tier graph changes besides the design model (applied by convert() to arms with `tier`; node ids are those of the app graph, absent ids are skipped).
# `quality`: default of the page's Quality field = the app's "final", which renders at still_preview.APP_FINAL_TIER (draft 576x1024, single stage: the design graphs are SINGLE-STAGE).
TIERS: dict[int, dict] = {
    16: dict(vram_gb=16, quality="final", n_ctx=CTX_16, flags=["--mmap-torch-files", "--cache-ram", "16", "--reserve-vram", "1.5", "--use-sage-attention"],
             loaders={"still_unet": {"unet_name": "moss_image_sft_bf16.safetensors"}, "still_clip": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors"}},
             why="16 GB: draft 576x1024 single-stage, LTX int8 + gemma int8 streamed by Dynamic VRAM (20 GB > 16), MOSS bf16 + int8-convrot encoder fully resident one at a time"),
    24: dict(vram_gb=24, quality="final", n_ctx=CTX_24, flags=["--mmap-torch-files", "--cache-ram", "16", "--reserve-vram", "1.0", "--use-sage-attention"],
             loaders={"still_unet": {"unet_name": "moss_image_sft_bf16.safetensors"}, "still_clip": {"clip_name": "qwen3.5_9b_bf16.safetensors"}},
             # switches of the lazy stages that need files only the 24 GB card has: the angle-still edit LoRA (moss_image_edit_chinoll, tested on bf16 only) and the clean-plate LoRA
             nodes={"first_set": {"angles": True}, "gather": {"plates": True}},
             why="24 GB: draft 576x1024 single-stage, LTX int8 (20 GB) resident at draft size, gemma int8 resident, MOSS bf16 + bf16 encoder resident one at a time"),
}
# THE SHIPPED NAMES (2026-10-30): the default graphs also written under neutral `motngrfx_*` names (the long `fusion_design_qwen_*` names say "qwen" although the entry is Gemma only).
# Same graph JSON as the arm graph (byte for byte); the sidecar differs in `title` (the dropdown label), `order` (0 = the page opens it) and `alias_of`. The old filenames stay (tests, box tooling).
# alias -> (base graph, arm, sidecar title, sidecar order)
DEFAULT_ALIASES: dict[str, tuple[str, str, str, int]] = {
    "motngrfx_default": ("fusion_design_qwen_auto_layers_pivot", "gemma31b_qat", "MOTNGRFX — 24 GB (Gemma 4 31B)", 0),
    "motngrfx_16gb": ("fusion_design_qwen_auto_layers", "gemma12b_q8", "MOTNGRFX — 16 GB (Gemma 4 12B)", 1),
    "motngrfx_3060": ("fusion_design_qwen_auto_layers", "gemma12b_q6", "MOTNGRFX — 12 GB, RTX 3060 (Gemma 4 12B)", 2),
}
TIER_SWITCHES = "nodes"                    # TIERS[t]["nodes"] = {node id: {input: value}}: set by convert() where the node exists, ignored by the "non-LLM node unchanged" check
FIELDS = ("label", "family", "dir", "repo", "licence", "file", "mmproj_file", "n_gpu_layers", "kv_cache", "mmproj_on_gpu", "n_cpu_moe", "handler", "top_k",
          "top_p", "repeat_penalty", "n_ctx", "image_max_tokens")


def ctx_for(arm: dict, has_image: bool = True) -> int:
    """n_ctx of an arm's node: the arm's for the design call (image) and the legibility read; an image-less call (concept, repairs) is capped at CONCEPT_CTX, except on the tier
    arms (TIERS: n_ctx per tier), whose repair carries the whole guide and so gets the tier's context too."""
    n = int(arm["n_ctx"])
    return n if has_image or arm.get("tier") else min(n, CONCEPT_CTX)


def thinking_fits(arm: dict) -> bool:
    """does the arm's n_ctx hold the largest design prompt + a full reasoning budget + a full reply (32768: 18000 + 8192 + 5000 = 31192 yes; 24576 no)."""
    return int(arm["n_ctx"]) >= DESIGN_PROMPT_TOKENS + THINK_TOKENS + DESIGN_REPLY_TOKENS


def arm_files(arm: dict) -> list[tuple[str, str]]:
    """[(repo-relative file, path under ComfyUI/models/LLM/GGUF)] of an arm: the model and its projector."""
    d = arm["dir"]
    out = [(arm["file"], f"{d}/{arm['file']}")]
    if arm.get("mmproj_file"):
        out.append((arm["mmproj_file"], f"{d}/{arm['mmproj_file']}"))
    return out


def model_path(arm: dict) -> str:
    """the value of the node's `model` combo: relative to models/LLM/GGUF."""
    return f"{arm['dir']}/{arm['file']}"


# ---------------------------------------------------------------------------------------------- conversion
def llm_nodes(g: dict) -> list[str]:
    return [nid for nid, n in g.items() if n.get("class_type") == OLD]


PROMPT_ROLES = {"concept_prompt": "concept", "design_pkg": "design", "design_parse": "design_repair"}     # where a node's prompt comes from -> its schema name (concept_parse = the concept repair: none)


def schema_role(ins: dict) -> str:
    """the reply schema a node's role asks for ('concept' | 'design' | 'design_repair' | 'none'): read from the node that supplies its prompt."""
    pr = ins.get("custom_prompt", ins.get("prompt"))
    return PROMPT_ROLES.get(pr[0], "none") if isinstance(pr, list) and pr else "none"


def convert_node(node: dict, arm: dict, *, title: str | None = None, reply_schema: bool = False) -> dict:
    """one DesignLLM_Advanced node -> one MographGGUFVL node (pure). reply_schema: also write the node's `schema` input (see the module docstring)."""
    ins = node["inputs"]
    has_image = isinstance(ins.get("image"), list)
    new = dict(prompt=ins["custom_prompt"], model=model_path(arm), mmproj=f"{arm['dir']}/{arm['mmproj_file']}" if arm.get("mmproj_file") else "none",
               n_gpu_layers=int(arm["n_gpu_layers"]), n_ctx=ctx_for(arm, has_image),
               max_tokens=int(ins["max_tokens"]), temperature=ins["temperature"], top_p=float(arm["top_p"]),
               top_k=int(arm["top_k"]), repeat_penalty=float(arm["repeat_penalty"]), seed=ins["seed"],
               kv_cache=arm["kv_cache"], flash_attn=True, mmproj_on_gpu=bool(arm["mmproj_on_gpu"]), n_cpu_moe=int(arm["n_cpu_moe"]),
               handler=arm["handler"], image_max_tokens=int(arm["image_max_tokens"]), strip_thinking=True, keep_model_loaded=False, verbose=False)
    if has_image:
        new["image"] = ins["image"]
        if arm.get("thinking"):
            if not thinking_fits(arm):
                raise ValueError(f"arm {arm['label']}: thinking needs n_ctx >= {DESIGN_PROMPT_TOKENS + THINK_TOKENS + DESIGN_REPLY_TOKENS} (prompt + reasoning + reply), the arm has {arm['n_ctx']}")
            new["thinking"] = True
            new["max_tokens"] = int(new["max_tokens"]) + THINK_TOKENS
    else:
        new["mmproj"] = "none"                    # text-only call (concept, repairs): no projector to load
        if arm.get("json_mode"):                  # thinking models (Qwen3.x): the reasoning leaked as plain text and never reached the JSON (batch 3060 v005 concept replies)
            new["json_mode"] = True
            new["max_tokens"] = max(new["max_tokens"], JSON_MODE_TOKENS)
    if reply_schema:
        role = schema_role(ins)
        if role != "none" and not (role == "design" and new.get("thinking")):         # a thinking design call keeps no grammar (the grammar forbids the reasoning)
            new["schema"] = role
    meta = dict(node.get("_meta") or {})
    meta["title"] = title or meta.get("title", "")
    return {"inputs": new, "class_type": NEW, "_meta": meta}


def short_name(arm: dict) -> str:
    """'Gemma 4 31B QAT Q4_K_XL' -> 'Gemma 4 31B', 'Qwen3.8-27B Q5_K_M' -> 'Qwen3.8-27B' (the model without its quantization, for the sidecar texts)."""
    out = []
    for tok in str(arm["label"]).split():
        if tok.startswith("QAT") or re.match(r"^I?Q\d", tok):
            break
        out.append(tok)
    return " ".join(out) or str(arm["label"])


def _rename(text: str, arm: dict) -> str:
    """a base-graph sidecar text (written for the design_llm_8b 4-bit graph) with the arm's real design model named: 'design_llm_8b (4-bit)' / 'design_llm_8b' / 'design_llm' / a bare 'Qwen'."""
    n = short_name(arm)
    text = text.replace("design_llm_8b (4-bit)", n).replace("design_llm_8b runs 4-bit", f"{n} runs as a GGUF").replace("design_llm_8b", n)
    text = re.sub(r"design_llm(?!-)", n, text)
    return re.sub(r"\bQwen\b(?![\d.-])", n, text)


def _retitle(old_title: str, arm: dict) -> str:
    """'design_llm design' -> 'Gemma 4 12B Q6_K design' (the dashboard / timing label)."""
    t = re.sub(r"^design_llm", arm["label"], old_title)
    return t if t != old_title else f"{arm['label']} {old_title}"


def convert(g: dict, side: dict | None, arm_key: str, *, reply_schema: bool | None = None) -> tuple[dict, dict | None]:
    """app graph (+ sidecar) -> the arm's twin. Pure function: the inputs are not modified. reply_schema: None = the arm's flag (`reply_schema`, default False)."""
    if arm_key not in ARMS:
        raise ValueError(f"arm {arm_key!r}: one of {sorted(ARMS)}")
    arm = ARMS[arm_key]
    rs = bool(arm.get("reply_schema", False)) if reply_schema is None else bool(reply_schema)
    nodes = llm_nodes(g)
    if not nodes:
        raise ValueError(f"no {OLD} node in the graph (already converted?)")
    g2 = copy.deepcopy(g)
    titles = {}
    for nid in nodes:
        t = _retitle(g[nid].get("_meta", {}).get("title", nid), arm)
        titles[g[nid].get("_meta", {}).get("title", nid)] = t
        g2[nid] = convert_node(g[nid], arm, title=t, reply_schema=rs)
    for nid in nodes:                                                   # thinking arm: the reasoning of the design call is archived (model_replies keeps every out_* text node)
        if g2[nid]["inputs"].get("thinking"):
            oid = "out_design_reasoning" if nid == "design_qwen" else f"out_{nid}_reasoning"
            g2[oid] = {"inputs": {"source": [nid, 2]}, "class_type": "PreviewAny", "_meta": {"title": "OUT design reasoning"}}
    for nid, n in g2.items():                                           # the VLM legibility check reads with the design model (same arm)
        if n.get("class_type") == READ:
            n["inputs"].update(model=model_path(arm), mmproj=f"{arm['dir']}/{arm['mmproj_file']}" if arm.get("mmproj_file") else "none", n_gpu_layers=int(arm["n_gpu_layers"]),
                               n_ctx=int(arm["n_ctx"]), kv_cache=arm["kv_cache"], flash_attn=True, mmproj_on_gpu=bool(arm["mmproj_on_gpu"]), n_cpu_moe=int(arm["n_cpu_moe"]),
                               handler=arm["handler"], image_max_tokens=int(arm["image_max_tokens"]), top_k=int(arm["top_k"]), top_p=float(arm["top_p"]),
                               repeat_penalty=float(arm["repeat_penalty"]))
    for nid, n in g2.items():                                           # the redesign graph names the arm in plan.design.model through these two nodes
        if n.get("class_type") in LABELLED:                             # ALWAYS set: the generated graphs carry no `model` input on these nodes, so the node default 'design_llm_8b' was written into every GGUF arm's piece / archive
            n["inputs"]["model"] = arm["label"]
    tier = TIERS.get(arm.get("tier"))
    if tier:                                                            # tier arm: stills loader files of the tier (see TIERS), nothing else in the graph
        for nid, ins in {**tier["loaders"], **tier.get(TIER_SWITCHES, {})}.items():
            if nid in g2:
                g2[nid]["inputs"].update({k: v for k, v in ins.items() if k != "angles" or "angles" in g2[nid]["inputs"]})     # `angles`: only a graph built with the angle chain (build_graphs --angles) has the switch
    if side is None:
        return g2, None
    s = copy.deepcopy(side)
    s["title"] = f"{_rename(s.get('title', 'Fusion'), arm)} - {arm['label']}"
    s["order"] = int(s.get("order", 20)) + 3
    if isinstance(s.get("server_flags"), dict) and s["server_flags"].get("why"):
        s["server_flags"]["why"] = _rename(s["server_flags"]["why"], arm)
    s["note"] = (_rename(str(s.get("note", "")), arm) + f" LLM arm: {arm['label']} ({arm['licence']}, llama.cpp GGUF, n_gpu_layers {arm['n_gpu_layers']}, "
                 f"KV {arm['kv_cache']}) instead of design_llm_8b 4-bit; see tools/app/llm_arms.py.").strip()
    s["llm"] = {k: arm[k] for k in FIELDS if k in arm}
    if rs:
        s["title"] = f"{s['title']} - reply schema grammar"
        s["order"] = int(s["order"]) + 1
        s["note"] += " Reply schema grammar ON (A/B twin of the plain graph): concept -> schema concept, main design call -> design, long design repair -> design (short repair: none); 16_app/REPLY_SCHEMA.md."
        s["reply_schema"] = {"concept": "concept", "design": "design", "design_repair": "design_repair", "doc": "16_app/REPLY_SCHEMA.md"}
    labels = s.setdefault("stage_labels", {})
    for old_t, new_t in titles.items():
        if old_t in labels:
            labels[new_t] = _rename(labels[old_t], arm)
    for old_t in list(labels):                                          # the labels of the base titles (no node carries them any more) name the arm's model too
        labels[old_t] = _rename(labels[old_t], arm)
    s.setdefault("llm_models", [model_path(arm), f"{arm['dir']}/{arm['mmproj_file']}"] if arm.get("mmproj_file") else [model_path(arm)])
    if tier:
        s["tier"] = {"vram_gb": tier["vram_gb"], "arm": arm_key, "measured": bool(arm.get("measured", True)), "vram_est_gib": arm.get("vram_est_gib"),
                     "loaders": tier["loaders"], "why": tier["why"], "doc": "16_app/TIERS.md", **({"nodes": tier[TIER_SWITCHES]} if tier.get(TIER_SWITCHES) else {})}
        s["title"] = s.get("title", "Fusion").replace("(RTX 3060 preview)", f"({tier['vram_gb']} GB tier)")
        s["note"] += (f" {tier['vram_gb']} GB tier (NOT MEASURED), Quality defaults to Final = the {SP.APP_FINAL_TIER} tier (576x1024 single-stage); Preview = the held stills; n_ctx {tier['n_ctx']} on every "
                      f"LLM call: {tier['why']}.")
        for f in s.get("fields", []):
            if f.get("id") == "quality":
                f["default"] = tier["quality"]
        s["server_flags"] = {"comfyui": list(tier["flags"]), "why": tier["why"]}
    return g2, s


# GEMMA RESIDENT (16_app/MEDIA_BOX.md "STORYBOARD v2"): the two side graphs `_media_read` (node id read_llm) and `_story` (story_llm) share ONE llama.cpp config so a story after a read reuses the loaded
# runner (nodes_fusion/llm_gguf.py + tools/app/llm_resident.py unload it after 10 min idle or when any other prompt starts). Same model / projector (ON for both) / n_gpu_layers / KV / flash / handler /
# image_max_tokens, and the SAME n_ctx = the larger of the two own budgets (READ 12288, STORY 8192). keep_model_loaded is true on those two nodes ONLY; the app / batch graphs keep it false (check()).
RESIDENT_NODE_IDS = ("read_llm", "story_llm")
RESIDENT_N_CTX = 12288


def resident_inputs(seed: int = 1, max_tokens: int = 1200, temperature: float = 0.3) -> dict:
    """the MographGGUFVL inputs both side graphs share (gemma31b_qat arm, projector ON, n_ctx RESIDENT_N_CTX, keep_model_loaded true); each graph sets its own prompt / system prompt / json_mode / sampling."""
    arm = ARMS["gemma31b_qat"]
    return dict(model=model_path(arm), mmproj=f"{arm['dir']}/{arm['mmproj_file']}", n_gpu_layers=int(arm["n_gpu_layers"]), n_ctx=RESIDENT_N_CTX,
                max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(arm["top_p"]), top_k=int(arm["top_k"]), repeat_penalty=float(arm["repeat_penalty"]),
                seed=int(seed), kv_cache=arm["kv_cache"], flash_attn=True, mmproj_on_gpu=bool(arm["mmproj_on_gpu"]), n_cpu_moe=int(arm["n_cpu_moe"]), handler=arm["handler"],
                image_max_tokens=int(arm["image_max_tokens"]), strip_thinking=True, keep_model_loaded=True, verbose=False)


def _resident_ok(nid: str, ins: dict) -> bool:
    """keep_model_loaded true is allowed on the two side-graph nodes only, and only with the shared config (projector on, n_ctx RESIDENT_N_CTX)."""
    return nid in RESIDENT_NODE_IDS and ins.get("mmproj") not in (None, "", "none") and int(ins.get("n_ctx", 0)) == RESIDENT_N_CTX


READ_FIELDS = ("model", "mmproj", "n_gpu_layers", "n_ctx", "kv_cache", "flash_attn", "mmproj_on_gpu", "n_cpu_moe", "handler", "image_max_tokens", "top_k", "top_p", "repeat_penalty")
LABELLED = ("MographDesignParse", "MographDesignRepair")           # nodes whose `model` input is the arm's label (plan.design.model)


def _tier_inputs(tier: dict | None, nid: str) -> dict | None:
    """the inputs the tier sets on node `nid` (loader files + stage switches), None when it sets none."""
    if not tier:
        return None
    d = {**tier.get("loaders", {}).get(nid, {}), **tier.get(TIER_SWITCHES, {}).get(nid, {})}
    return d or None


def _strip_label(node: dict, loaders: dict | None = None) -> dict:
    if loaders:                                                 # a tier arm's loader files / stage switches (TIERS) are not a change of the base graph
        n = copy.deepcopy(node)
        for k in loaders:
            n["inputs"].pop(k, None)
        return _strip_label(n)
    if node.get("class_type") in LABELLED:
        n = copy.deepcopy(node)
        n["inputs"].pop("model", None)
        return n
    if node.get("class_type") == READ:                    # filled from the arm by convert(): not a change of the base graph
        n = copy.deepcopy(node)
        for k in READ_FIELDS:
            n["inputs"].pop(k, None)
        return n
    return node


# ---------------------------------------------------------------------------------------------- checks
def check(g: dict, arm_key: str | None = None, base: dict | None = None, reply_schema: bool | None = None) -> list[str]:
    """structural problems of a converted graph: no DesignLLM node left, every GGUF node's links / inputs, the design node still reads the sheet, consumers read
    slot 0; with `base` (the unconverted graph): the same node ids, and nothing else changed."""
    bad = list(ltx_ic.check_graph(g))
    if not APP_CLASSES_KNOWN:                                # build_graphs did not import: only the classes known here can be judged
        bad = [b for b in bad if "not in CLASSES" not in b]
    if llm_nodes(g):
        bad.append(f"{OLD} node(s) left: {llm_nodes(g)}")
    gg = [n for n, v in g.items() if v.get("class_type") == NEW]
    if not gg:
        bad.append(f"no {NEW} node")
    arm = ARMS.get(arm_key) if arm_key else None
    for nid in gg:
        ins = g[nid]["inputs"]
        if not isinstance(ins.get("prompt"), list):
            bad.append(f"{nid}: prompt is not linked")
        if arm and ins.get("model") != model_path(arm):
            bad.append(f"{nid}: model {ins.get('model')!r} is not the arm's {model_path(arm)!r}")
        if ins.get("keep_model_loaded") and not _resident_ok(nid, ins):
            bad.append(f"{nid}: keep_model_loaded must be false (LTX / ACE load between the calls); only the read / story side graphs ({', '.join(RESIDENT_NODE_IDS)}) keep Gemma resident")
        if int(ins.get("max_tokens", 0)) < 64 or int(ins.get("n_ctx", 0)) < int(ins.get("max_tokens", 0)) + 1024:
            bad.append(f"{nid}: n_ctx {ins.get('n_ctx')} leaves no room for max_tokens {ins.get('max_tokens')} + a prompt")
        if arm and arm.get("tier") and int(ins.get("n_ctx", 0)) != TIERS[arm["tier"]]["n_ctx"]:
            bad.append(f"{nid}: n_ctx {ins.get('n_ctx')} is not the {arm['tier']} GB tier's {TIERS[arm['tier']]['n_ctx']}")
        if "image" in ins and not (isinstance(ins["image"], list) and ins["image"][1] == 3):
            bad.append(f"{nid}: image must be the Design Package `sheet` (slot 3), got {ins['image']}")
        want_rs = bool(arm.get("reply_schema", False)) if reply_schema is None and arm else bool(reply_schema)
        role = schema_role(dict(ins, custom_prompt=ins.get("prompt")))
        exp = role if (want_rs and role != "none" and not (role == "design" and ins.get("thinking"))) else None
        if ins.get("schema") != exp:
            bad.append(f"{nid}: schema {ins.get('schema')!r}, expected {exp!r} (role {role}, reply_schema {want_rs})")
    for nid, n in g.items():
        if arm and arm.get("tier") and n.get("class_type") == READ and int(n["inputs"].get("n_ctx", 0)) != TIERS[arm["tier"]]["n_ctx"]:
            bad.append(f"{nid}: n_ctx {n['inputs'].get('n_ctx')} is not the {arm['tier']} GB tier's {TIERS[arm['tier']]['n_ctx']}")
        for name, v in n.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and v[0] in gg and v[1] != 0 and n.get("class_type") not in ("PreviewAny",):
                bad.append(f"{nid}.{name} reads {v[0]}[{v[1]}]: only the response (slot 0) is consumed")
    if base is not None:
        extra = {n for n in set(g) - set(base) if n.startswith("out_") and n.endswith("_reasoning")}         # the archive node of a thinking arm
        if set(base) != set(g) - extra:
            bad.append(f"node ids differ from the base: -{sorted(set(base) - set(g))} +{sorted(set(g) - set(base) - extra)}")
        for nid in base:
            ld = _tier_inputs(TIERS.get((arm or {}).get("tier")), nid)
            if nid in g and base[nid].get("class_type") != OLD and _strip_label(base[nid], ld) != _strip_label(g[nid], ld):
                bad.append(f"{nid}: a node that is not an LLM node was changed")
    return bad


# ---------------------------------------------------------------------------------------------- probe graph (one model, the benchmark prompt)
def probe_graph(arm_key: str, prompt: str, sheet_name: str = "llm_probe_sheet.png", *, n_gpu_layers: int | None = None, max_tokens: int = 4096,
                temperature: float = 0.7, seed: int = 3, n_ctx: int | None = None, verbose: bool = False, **over) -> dict:
    """LoadImage(sheet) + the prompt -> MographGGUFVL -> PreviewAny (reply) + PreviewAny (stats). The reply is read from the ComfyUI history."""
    arm = ARMS[arm_key]
    g: dict = {}
    g["prompt"] = {"inputs": {"value": prompt}, "class_type": "PrimitiveStringMultiline", "_meta": {"title": "IN prompt"}}
    g["sheet"] = {"inputs": {"image": sheet_name}, "class_type": "LoadImage", "_meta": {"title": "IN sheet"}}
    base = {"inputs": {"custom_prompt": ["prompt", 0], "image": ["sheet", 0], "max_tokens": max_tokens, "temperature": temperature, "top_p": arm["top_p"],
                       "repetition_penalty": arm["repeat_penalty"], "seed": seed}, "_meta": {"title": "x"}}
    n = convert_node(base, arm, title=f"{arm['label']} design")
    n["inputs"]["image"] = ["sheet", 0]
    if n_gpu_layers is not None:
        n["inputs"]["n_gpu_layers"] = int(n_gpu_layers)
    if n_ctx is not None:
        n["inputs"]["n_ctx"] = int(n_ctx)
    n["inputs"]["verbose"] = bool(verbose)
    n["inputs"].update({k: v for k, v in over.items() if k in n["inputs"]})       # e.g. kv_cache, mmproj_on_gpu, n_cpu_moe for the tuning runs
    g["llm"] = n
    g["out_reply"] = {"inputs": {"source": ["llm", 0]}, "class_type": "PreviewAny", "_meta": {"title": "OUT reply"}}
    g["out_stats"] = {"inputs": {"source": ["llm", 1]}, "class_type": "PreviewAny", "_meta": {"title": "OUT stats"}}
    return g


def baseline_probe_graph(prompt: str, sheet_name: str = "llm_probe_sheet.png", *, max_tokens: int = 4096, temperature: float = 0.7, seed: int = 3) -> dict:
    """the same probe with the BASELINE model: DesignLLM_Advanced, design_llm_8b 4-bit nf4, exactly the design node's settings in build_graphs.qwen_node."""
    g: dict = {}
    g["prompt"] = {"inputs": {"value": prompt}, "class_type": "PrimitiveStringMultiline", "_meta": {"title": "IN prompt"}}
    g["sheet"] = {"inputs": {"image": sheet_name}, "class_type": "LoadImage", "_meta": {"title": "IN sheet"}}
    g["llm"] = _qwen_baseline(["prompt", 0], ["sheet", 0], max_tokens, temperature, seed)
    g["out_reply"] = {"inputs": {"source": ["llm", 0]}, "class_type": "PreviewAny", "_meta": {"title": "OUT reply"}}
    return g


def _qwen_baseline(prompt_link, image_link, max_tokens, temperature, seed) -> dict:
    return {"inputs": {"model_name": "design_llm_8b", "quantization": "4-bit (VRAM-friendly)", "attention_mode": "auto", "use_torch_compile": False,
                       "device": "auto", "preset_prompt": "🖼️ Detailed Description", "custom_prompt": prompt_link, "max_tokens": max_tokens,
                       "temperature": temperature, "top_p": 0.9, "num_beams": 1, "repetition_penalty": 1.05, "frame_count": 16, "video_frame_size": "auto",
                       "keep_model_loaded": False, "seed": seed, "image": image_link},
            "class_type": OLD, "_meta": {"title": "design_llm_8b 4-bit design (baseline)"}}


# ---------------------------------------------------------------------------------------------- IO / CLI
def _read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def graph_name(base: str, arm_key: str) -> str:
    return f"{base}_{arm_key}"


def fetch_lines(arm_key: str) -> list[str]:
    arm = ARMS[arm_key]
    base = "/workspace/ComfyUI/models/LLM/GGUF"
    return [f'mkdir -p {base}/{arm["dir"]} && aria2c -x4 -s4 -k8M --file-allocation=none -c -d {base}/{arm["dir"]} -o {os.path.basename(rel)} '
            f'"https://huggingface.co/{arm["repo"]}/resolve/main/{f}"' for f, rel in arm_files(arm)]


def alias_files(alias: str, graphs_dir: str = GRAPHS) -> tuple[dict, dict] | None:
    """(graph, sidecar) of a DEFAULT_ALIASES name, built from the base graph on disk with the same convert() call as the arm graph; None when the base graph is not on disk."""
    base, arm_key, title, order = DEFAULT_ALIASES[alias]
    bp = os.path.join(graphs_dir, base + ".json")
    if not os.path.isfile(bp):
        return None
    g, s = convert(_read(bp), _read(os.path.join(graphs_dir, base + ".app.json")), arm_key)
    s["title"], s["order"], s["alias_of"] = title, order, graph_name(base, arm_key)
    return g, s


def emit_aliases(graphs_dir: str, out_dir: str, check_only: bool, only_arms=None, only_bases=None) -> int:
    """write (or, with check_only, compare) the motngrfx_* default graphs; returns the number of problems."""
    bad = 0
    for alias, (base, arm_key, _t, _o) in DEFAULT_ALIASES.items():
        if (only_arms and arm_key not in only_arms) or (only_bases and base not in only_bases):
            continue
        built = alias_files(alias, graphs_dir)
        if built is None:
            print(f"{alias}: base graph {base} not on disk, skipped")
            continue
        g, s = built
        problems = check(g, arm_key, _read(os.path.join(graphs_dir, base + ".json")))
        print(f"{alias}: {len(g)} nodes = {graph_name(base, arm_key)} ({ARMS[arm_key]['label']})")
        for suf, obj in ((".json", g), (".app.json", s)):
            path = os.path.join(out_dir, alias + suf)
            if check_only:
                if not os.path.isfile(path):
                    problems.append(f"{alias}{suf} not written yet")
                elif _read(path) != obj:
                    problems.append(f"{alias}{suf} on disk is stale (run tools/app/llm_arms.py)")
            else:
                os.makedirs(out_dir, exist_ok=True)
                _write(path, obj)
        for p in problems:
            print(f"  PROBLEM: {p}")
        bad += len(problems)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["build", "list", "probe-graph", "fetch"], default="build")
    ap.add_argument("--baseline", action="store_true", help="probe-graph: the baseline (DesignLLM_Advanced, design_llm_8b 4-bit) instead of an arm")
    ap.add_argument("--check", action="store_true", help="validate and compare with the files on disk; write nothing")
    ap.add_argument("--arm", action="append", help=f"arm key(s), default all of {sorted(ARMS)}")
    ap.add_argument("--base", action="append", help=f"base graph stem(s), default {FILE_BASES} (those on disk); also {BASES[-1]}")
    ap.add_argument("--graphs-dir", default=GRAPHS)
    ap.add_argument("--out-dir", default=GRAPHS)
    ap.add_argument("--dir", help="probe-graph: package dir with prompt.txt")
    ap.add_argument("--out", help="probe-graph: output file")
    ap.add_argument("--reply-schema", action="store_true", help="the reply-schema A/B twin: schema grammar on the concept / design / long-repair calls; written as <base>_<arm>_schema (16_app/REPLY_SCHEMA.md)")
    ap.add_argument("--n-gpu-layers", type=int)
    ap.add_argument("--n-ctx", type=int)
    a = ap.parse_args(argv)
    arms = a.arm or sorted(ARMS)
    if a.cmd == "list":
        for k in arms:
            r = ARMS[k]
            print(f"{k:16s} {r['label']:26s} {r['file']} {r['gib']:.2f}+{r.get('mmproj_gib', 0):.2f} GiB  {r['licence']}  ngl {r['n_gpu_layers']} kv {r['kv_cache']}  {r['repo']}")
        return 0
    if a.cmd == "fetch":
        for k in arms:
            print("\n".join(fetch_lines(k)))
        return 0
    if a.cmd == "probe-graph":
        prompt = open(os.path.join(a.dir, "prompt.txt"), encoding="utf-8").read()
        g = baseline_probe_graph(prompt) if a.baseline else probe_graph(arms[0], prompt, n_gpu_layers=a.n_gpu_layers, n_ctx=a.n_ctx)
        _write(a.out, g)
        print(f"wrote {a.out}")
        return 0
    bases_arg = a.base
    bases = a.base or [b for b in FILE_BASES if os.path.isfile(os.path.join(a.graphs_dir, b + ".json"))]
    if not bases:
        print(f"no base graph in {a.graphs_dir} (run tools/app/build_graphs.py first)")
        return 1
    bad_total = 0
    for b in bases:
        g0, s0 = _read(os.path.join(a.graphs_dir, b + ".json")), _read(os.path.join(a.graphs_dir, b + ".app.json"))
        for k in arms:
            if not bases_arg and b not in ARMS[k].get("bases", FILE_BASES):          # tier arms are built for their own base graph(s) unless --base is given
                continue
            # variants: the plain graph, and the reply-schema A/B twin (built with --reply-schema; an existing twin is checked / rewritten with its plain graph so it cannot go stale)
            twin_on_disk = os.path.isfile(os.path.join(a.out_dir, graph_name(b, k) + "_schema.json"))
            for rs in ([True] if a.reply_schema else [None] + ([True] if twin_on_disk else [])):
                g, s = convert(g0, s0, k, reply_schema=rs)
                name = graph_name(b, k) + ("_schema" if rs else "")
                problems = check(g, k, g0, reply_schema=rs)
                print(f"{name}: {len(g)} nodes, {len(llm_nodes(g0))} LLM nodes -> {NEW} ({ARMS[k]['label']})" + (" [reply schema grammar]" if rs else ""))
                for suf, obj in ((".json", g), (".app.json", s)):
                    path = os.path.join(a.out_dir, name + suf)
                    if a.check:
                        if not os.path.isfile(path):
                            problems.append(f"{name}{suf} not written yet")
                        elif _read(path) != obj:
                            problems.append(f"{name}{suf} on disk is stale (run tools/app/llm_arms.py{' --reply-schema' if rs else ''})")
                    else:
                        os.makedirs(a.out_dir, exist_ok=True)
                        _write(path, obj)
                for p in problems:
                    print(f"  PROBLEM: {p}")
                bad_total += len(problems)
    if not a.reply_schema:
        bad_total += emit_aliases(a.graphs_dir, a.out_dir, a.check, a.arm, bases_arg)
    print("OK" if not bad_total else f"{bad_total} problem(s)")
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
