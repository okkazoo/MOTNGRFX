"""MographGGUFVL: a local llama.cpp (GGUF) vision-language model as the concept / design / repair step. NO CLAUDE, no API, open weights.

    prompt, image? -> response, stats

One node for Qwen-VL AND Gemma 4 (the GGUF's own chat template, via tools/app/llm_gguf.py), with n_gpu_layers / context / KV-cache type / projector
placement exposed, so a model that is just at the edge of the card's VRAM (or past it, with a few layers on the CPU) can be used. llama.cpp is outside
ComfyUI's memory manager (no Dynamic VRAM), so the node frees ComfyUI's models first and unloads the model after the call unless keep_model_loaded.
`stats` = json: load_s, prompt_tokens / prompt_s / prompt_tok_s, gen_tokens / gen_s / gen_tok_s, total_s, vram_base_mib / vram_peak_mib (whole GPU),
rss_peak_gib. Swapped in for DesignLLM_Advanced by tools/app/llm_arms.py. Needs the CUDA wheel of JamePeng/llama-cpp-python (tools/app/llm_arms.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY

ensure_root_on_path()
from tools.app import llm_gguf as LG        # noqa: E402
from tools.app import llm_resident as LR    # noqa: E402
from tools.fusion import reply_schema as RS  # noqa: E402  (pure: the schemas are built lazily, on the first call that names one)

_RUNNER = LG.Runner()                        # kept between calls only when keep_model_loaded (the `_media_read` + `_story` side graphs: ONE shared config, tools/app/llm_arms.RESIDENT_*)
_RES = LR.Resident(_RUNNER)                  # idle 10 min -> unload; any other prompt starting -> unload (hook below)


def _install_hook() -> None:
    try:
        import execution
        LR.install_executor_hook(execution, _RES)
    except Exception as e:   # noqa: BLE001   outside ComfyUI (tests) there is no execution module
        print(f"[MographGGUFVL] resident hook not installed: {type(e).__name__}: {e}")


_install_hook()


def gguf_root() -> str:
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "LLM", "GGUF")
    except Exception:       # noqa: BLE001
        return LG.DEFAULT_ROOT


def free_vram() -> None:
    """ComfyUI's models out of VRAM before llama.cpp allocates (it is not managed by ComfyUI). Harmless outside ComfyUI."""
    try:
        import gc
        import comfy.model_management as mm
        import torch
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:   # noqa: BLE001
        print(f"[MographGGUFVL] free_vram skipped: {type(e).__name__}: {e}")


def _first_image_b64(image) -> str | None:
    if image is None:
        return None
    a = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if a.ndim == 4:
        a = a[0]
    a = np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)[..., :3]
    return LG.png_b64(a)


class MographGGUFVL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = gguf_root()
        models = LG.list_ggufs(root) or ["(no GGUF model under models/LLM/GGUF)"]
        return io.Schema(
            node_id="MographGGUFVL",
            display_name="MOTNGRFX GGUF VL (llama.cpp)",
            category=CATEGORY,
            description="A GGUF vision-language model (Gemma 4) run by llama.cpp with partial or full GPU offload. Frees ComfyUI's "
                        "models first, unloads after the call. prompt (+ one image) -> reply text and a json of timings / VRAM / RAM.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("model", default=models[0], tooltip="GGUF path relative to ComfyUI/models/LLM/GGUF (a string, not a list: a model downloaded "
                                                                    "after ComfyUI started needs no restart). Found now: " + ", ".join(models)[:300]),
                io.String.Input("mmproj", default="auto", tooltip="vision projector path relative to the same folder; auto = the mmproj*.gguf beside the model, none = text only"),
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=300, tooltip="-1 = all layers on the GPU; fewer = the rest on the CPU (slower)"),
                io.Int.Input("n_ctx", default=16384, min=1024, max=131072, step=512, tooltip="prompt + image + reply tokens"),
                io.Int.Input("max_tokens", default=4096, min=64, max=16384),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.05),
                io.Int.Input("top_k", default=40, min=0, max=200),
                io.Float.Input("repeat_penalty", default=1.05, min=1.0, max=2.0, step=0.01),
                io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF),
                io.Combo.Input("kv_cache", options=list(LG.KV_TYPES), default="q8_0", tooltip="KV cache type; q8_0 halves it (needs flash attention)"),
                io.Boolean.Input("flash_attn", default=True),
                io.Boolean.Input("mmproj_on_gpu", default=True, tooltip="false = the projector runs on the CPU (~1 GB less VRAM, slower image encode)"),
                io.Int.Input("n_cpu_moe", default=0, min=0, max=200, tooltip="MoE models: expert tensors of the first N layers stay on the CPU"),
                io.Combo.Input("handler", options=list(LG.HANDLERS), default="auto", tooltip="auto = the GGUF's own chat template"),
                io.Int.Input("image_max_tokens", default=-1, min=-1, max=16384, tooltip="-1 = the model's default image token budget"),
                io.Boolean.Input("strip_thinking", default=True, tooltip="drop <think> blocks from the reply"),
                io.Boolean.Input("keep_model_loaded", default=False),
                io.Boolean.Input("verbose", default=False, tooltip="llama.cpp load log into the ComfyUI log: layers offloaded, buffer sizes (for tuning n_gpu_layers)"),
                io.String.Input("system_prompt", multiline=True, default="", optional=True),
                io.Image.Input("image", optional=True, tooltip="ONE image (the combined sheet); a batch uses its first frame"),
                io.Boolean.Input("json_mode", default=False, optional=True, tooltip="llama.cpp JSON grammar: the reply is ONE JSON object, no reasoning text before it (Qwen3.x concept / repair calls)"),
                io.Boolean.Input("thinking", default=False, optional=True, tooltip="thinking ON (par round r6): the chat template is rendered with enable_thinking, the model reasons first. The "
                                                                                    "reasoning is split off the reply (output `reasoning`, never in `response`) and costs tokens: max_tokens and n_ctx must "
                                                                                    "hold prompt + reasoning + reply. Not with json_mode or a schema (the grammar forbids the reasoning)."),
                io.Combo.Input("schema", options=list(RS.NAMES), default="none", optional=True,
                               tooltip="reply schema grammar (tools/fusion/reply_schema.py, 16_app/REPLY_SCHEMA.md): design = the app's design reply, concept = the concept reply; none = off. "
                                       "Like json_mode but the grammar also fixes the TYPE and VALUE of every field (enums, list item types, word count read from the prompt); implies json_mode, "
                                       "so no thinking. Not for the short repair call (its reply is a {path: value} map)."),
            ],
            outputs=[io.String.Output(display_name="response"), io.String.Output(display_name="stats"), io.String.Output(display_name="reasoning")],
        )

    @classmethod
    def execute(cls, prompt, model, mmproj, n_gpu_layers, n_ctx, max_tokens, temperature, top_p, top_k, repeat_penalty, seed, kv_cache, flash_attn,
                mmproj_on_gpu, n_cpu_moe, handler, image_max_tokens, strip_thinking, keep_model_loaded, verbose=False, system_prompt="", image=None, schema="none", json_mode=False, thinking=False) -> io.NodeOutput:
        schema = str(schema or "none")
        root = gguf_root()
        mpath = os.path.join(root, model)
        if not os.path.isfile(mpath):
            raise FileNotFoundError(f"[MographGGUFVL] model not found: {mpath}")
        mm = None if mmproj == "none" else (LG.mmproj_for(mpath) if mmproj == "auto" else os.path.join(root, mmproj))
        img = _first_image_b64(image) if mm else None
        if image is not None and mm is None:
            print("[MographGGUFVL] an image was given but the model has no projector: text-only")
        free_vram()
        smp = LG.Sampler()
        smp.start()
        global _RUNNER
        _RES.begin()                      # holds the lock, stops the idle timer; end() re-arms it (keep) or closes the model
        try:
            ls = _RUNNER.load(mpath, mm, handler=handler, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, kv=kv_cache, flash_attn=flash_attn, n_cpu_moe=n_cpu_moe,
                              mmproj_gpu=mmproj_on_gpu, image_max_tokens=image_max_tokens, verbose=bool(verbose), think=bool(thinking) and not json_mode and schema == "none")
            text, st = _RUNNER.chat(prompt, img, system=system_prompt or "", max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                                    repeat_penalty=repeat_penalty, seed=seed, json_mode=bool(json_mode), schema=schema)
        finally:
            mem = smp.stop()
            _RES.end(bool(keep_model_loaded))
        answer, think = LG.split_think(text)
        stats = dict(model=model, mmproj=os.path.basename(mm) if mm else None, n_gpu_layers=n_gpu_layers, kv_cache=kv_cache, **ls, **st, **mem)             # st carries n_ctx
        stats.update(thinking=bool(thinking) and not json_mode and schema == "none", schema=schema,        # any grammar forbids text before the object: no thinking
                      think_chars=len(think), think_tokens_est=int(round(st.get("gen_tokens", 0) * len(think) / max(len(text), 1))),
                     n_ctx_left=int(st.get("n_ctx", n_ctx)) - int(st.get("prompt_tokens", 0)) - int(st.get("gen_tokens", 0)))      # 0 / negative = the context was full
        print("[MographGGUFVL] " + json.dumps(stats))
        if keep_model_loaded:
            print(f"[MographGGUFVL] gemma resident: {'reused (no load)' if ls.get('reused') else 'loaded in %.1f s' % float(ls.get('load_s', 0))}; kept loaded, unloads after {LR.IDLE_S:g} s idle or when another prompt starts")
        return io.NodeOutput(answer if strip_thinking else text.strip(), json.dumps(stats), think)


def _gguf_extra(args: dict) -> dict:
    """what the key needs besides the inputs: the model file (size + mtime) and the projector's."""
    from . import node_cache as _NC
    mp = os.path.join(gguf_root(), str(args.get("model")))
    mm = args.get("mmproj")
    return dict(model=_NC.file_stamp(mp), mmproj=_NC.file_stamp(LG.mmproj_for(mp) if mm == "auto" else os.path.join(gguf_root(), str(mm))) if mm and mm != "none" else None)


def _gguf_hit(vals: list) -> list:
    """a reused reply: the stats JSON says so (run_timing / the page read it as before)."""
    try:
        st = json.loads(vals[1])
        st["cache_hit"] = True
        vals[1] = json.dumps(st)
    except (ValueError, IndexError, TypeError):
        pass
    return vals


try:      # STAGE CACHE (2026-11-01): the same prompt + image + model + seed + sampling returns the earlier reply (also after a ComfyUI restart); performance-only inputs are not in the key
    from . import node_cache as _NC
    MographGGUFVL.execute = classmethod(_NC.cached_text_outputs(MographGGUFVL.execute.__func__, "gemma", ignore=("keep_model_loaded", "verbose", "n_gpu_layers", "mmproj_on_gpu", "n_cpu_moe", "flash_attn"),
                                                                 extra=_gguf_extra, make=lambda v: io.NodeOutput(*v), on_hit=_gguf_hit))
except Exception as _e:   # noqa: BLE001
    print(f"[MographGGUFVL] reply cache not installed: {type(_e).__name__}: {_e}")


NODES = [MographGGUFVL]
