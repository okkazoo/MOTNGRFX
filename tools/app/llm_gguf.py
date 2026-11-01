#!/usr/bin/env python
"""llama.cpp (JamePeng llama-cpp-python, CUDA) runner for the LLM arms of the app graph: ONE code path for Qwen-VL and Gemma 4 GGUF models,
used by the ComfyUI node `MographGGUFVL` (`nodes_fusion/llm_gguf.py`) and by the CLI below (a fresh process = a clean OOM boundary for tuning
`n_gpu_layers`). Open weights, local, no API. Docs: tools/app/llm_arms.py.

Why not the design-LLM node pack's own GGUF node: it hard-codes `design_llmChatHandler` + Qwen stop tokens (no Gemma 4), does not free ComfyUI's models first and
caps nothing we need to measure. This runner takes the chat template from the GGUF itself (`GenericMTMDChatHandler`), or a named handler.

    python tools/app/llm_gguf.py probe --model <file.gguf> [--mmproj <file>|auto|none] --prompt-file p.txt [--image sheet.png] \
        [--n-gpu-layers -1] [--ctx 16384] [--kv q8_0] [--max-tokens 4096] [--out reply.txt] [--stats stats.json]
    python tools/app/llm_gguf.py list [--root /workspace/ComfyUI/models/LLM/GGUF]

The probe prints ONE json line `STATS {...}`: load_s, prompt_tokens, prompt_s (time to first token = prompt + image encode), prompt_tok_s, gen_tokens, gen_s,
gen_tok_s, total_s, vram_peak_mib / vram_base_mib (whole GPU, torch.cuda.mem_get_info, polled 5x/s), rss_peak_gib, n_gpu_layers, layers_offloaded (parsed from the
llama.cpp load log when it was captured).
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

DEFAULT_ROOT = "/workspace/ComfyUI/models/LLM/GGUF"
KV_TYPES = {"f16": 1, "q8_0": 8, "q4_0": 2}                   # ggml_type ints (GGML_TYPE_F16 / Q8_0 / Q4_0)
HANDLERS = ("auto", "generic", "gemma4", "design_llm", "qwen35", "qwen25vl")
IMAGE_SUFFIX = (".png", ".jpg", ".jpeg", ".webp")


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def list_ggufs(root: str | os.PathLike | None = None) -> list[str]:
    """model GGUF files under root as posix paths relative to it (mmproj / imatrix / mtp helper files excluded)."""
    r = Path(root or DEFAULT_ROOT)
    if not r.is_dir():
        return []
    out = []
    for p in sorted(r.rglob("*.gguf")):
        n = p.name.lower()
        if "mmproj" in n or n.startswith("imatrix") or n.startswith("mtp-") or "/mtp/" in p.as_posix().lower():
            continue
        out.append(p.relative_to(r).as_posix())
    return out


def mmproj_for(model_path: str | os.PathLike) -> str | None:
    """the projector beside a model: F16 / BF16 preferred over Q8_0 / F32, None when the folder has none."""
    d = Path(model_path).parent
    cands = sorted(d.glob("*mmproj*.gguf"))
    if not cands:
        return None
    def rank(p: Path) -> int:
        n = p.name.lower()
        return 0 if "f16" in n and "bf16" not in n else 1 if "bf16" in n else 2 if "q8" in n else 3
    return str(sorted(cands, key=rank)[0])


def kv_type(name: str | None) -> int | None:
    """'f16' -> None (llama.cpp default), 'q8_0' -> 8, 'q4_0' -> 2; anything else is an error."""
    n = (name or "f16").lower()
    if n not in KV_TYPES:
        raise ValueError(f"kv cache type must be one of {sorted(KV_TYPES)}, got {name!r}")
    return None if n == "f16" else KV_TYPES[n]


def load_kwargs(model_path: str, mmproj: str | None, *, n_gpu_layers: int = -1, n_ctx: int = 16384, n_batch: int = 512, n_ubatch: int | None = None,
                kv: str = "f16", flash_attn: bool = True, n_cpu_moe: int = 0, n_threads: int | None = None, swa_full: bool = False,
                handler: str = "auto", mmproj_gpu: bool = True, image_min_tokens: int = -1, image_max_tokens: int = -1, verbose: bool = False, think: bool = False) -> dict:
    """arguments of Llama(...) (pure: no llama_cpp import). The projector goes to `mmproj_path` + `chat_handler_kwargs` (auto / generic = the GGUF's own
    chat template); a named handler is built by the caller (see make_handler)."""
    if handler not in HANDLERS:
        raise ValueError(f"handler must be one of {HANDLERS}, got {handler!r}")
    kw: dict = dict(model_path=str(model_path), n_gpu_layers=int(n_gpu_layers), n_ctx=int(n_ctx), n_batch=int(n_batch),
                    n_threads=int(n_threads or max(1, (os.cpu_count() or 8) // 2)), swa_full=bool(swa_full), verbose=bool(verbose),
                    flash_attn_type=1 if flash_attn else 0)
    if n_ubatch:
        kw["n_ubatch"] = int(n_ubatch)
    if n_cpu_moe:
        kw["n_cpu_moe"] = int(n_cpu_moe)
    k = kv_type(kv)
    if k is not None:
        kw["type_k"] = k
        kw["type_v"] = k                         # a quantised V cache needs flash attention
        if not flash_attn:
            raise ValueError("a quantised KV cache (q8_0 / q4_0) needs flash_attn")
    if mmproj:
        if handler in ("auto", "generic"):
            kw["mmproj_path"] = str(mmproj)
            kw["chat_handler_kwargs"] = dict(use_gpu=bool(mmproj_gpu), image_min_tokens=int(image_min_tokens), image_max_tokens=int(image_max_tokens),
                                             verbose=False, extra_template_arguments={"enable_thinking": bool(think)})
    return kw


def handler_kwargs(mmproj: str, *, mmproj_gpu: bool = True, image_min_tokens: int = -1, image_max_tokens: int = -1) -> dict:
    return dict(mmproj_path=str(mmproj), use_gpu=bool(mmproj_gpu), image_min_tokens=int(image_min_tokens), image_max_tokens=int(image_max_tokens), verbose=False)
    # (named handlers take enable_thinking / force_reasoning themselves; the generic one gets it as a template argument, see load_kwargs)


def make_handler(name: str, mmproj: str, **kw):
    """a named chat handler of the fork (design_llm / qwen35 / qwen25vl / gemma4); 'auto' / 'generic' return None (Llama builds the generic one)."""
    if name in ("auto", "generic"):
        return None
    from llama_cpp import llama_chat_format as F
    cls = dict(gemma4=F.Gemma4ChatHandler, qwen35=F.Qwen35ChatHandler, qwen25vl=F.Qwen25VLChatHandler)[name]
    return cls(**handler_kwargs(mmproj, **kw))


def response_format_for(json_mode: bool = False, schema=None, prompt: str = "") -> dict | None:
    """the `response_format` of a chat call. schema = None | "none" -> `json_object` (llama.cpp's generic JSON grammar) when json_mode, else no grammar; a NAME ("design" | "concept",
    tools/fusion/reply_schema.py: the exact counts are read from `prompt`) or a JSON-schema dict -> `{"type": "json_object", "schema": <schema>}`: the grammar then also fixes the
    TYPE and VALUE of every field (enums, list item types, integer ranges, the declared keys), not only the syntax. A schema implies the JSON grammar (json_mode is not needed)."""
    if isinstance(schema, str) and schema.strip().lower() in ("", "none", "off"):
        schema = None
    if schema is None:
        return {"type": "json_object"} if json_mode else None
    if isinstance(schema, str):
        name = schema
        try:
            from tools.fusion import reply_schema as RS
            schema = RS.schema_for(name, prompt)
        except Exception as e:      # noqa: BLE001  a schema that cannot be built must not lose the call: the generic JSON grammar (or none) instead, loudly
            print(f"[llm_gguf] reply schema {name!r} could not be built ({type(e).__name__}: {e}); falling back to {'json_object' if json_mode else 'no grammar'}")
            schema = None
        if schema is None:
            return {"type": "json_object"} if json_mode else None
    return {"type": "json_object", "schema": schema}


def messages_for(prompt: str, image_b64: str | None, system: str = "") -> list[dict]:
    """OpenAI-style chat messages; the image goes first in the user turn (Qwen / Gemma both expect the image before its question)."""
    msgs: list[dict] = []
    if system.strip():
        msgs.append({"role": "system", "content": system})
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}, {"type": "text", "text": prompt}]
    else:
        content = prompt
    msgs.append({"role": "user", "content": content})
    return msgs


def png_b64(arr) -> str:
    """uint8 [H,W,3] numpy -> base64 PNG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_OFF = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.I)


def parse_offload(log: str) -> tuple[int, int] | None:
    """('offloaded 41/49 layers to GPU' in llama.cpp's load log) -> (41, 49)."""
    m = _OFF.search(log or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def split_think(text: str) -> tuple[str, str]:
    """(answer, thinking): a <think>..</think> block / Gemma channel thoughts / a closing-only tag (the template pre-filled the opening one) / an unclosed block are cut out of
    the answer and returned separately (tools/fusion/why_fields.split_think, par round r6): the parser never sees the thinking, the archive keeps it."""
    try:
        from tools.fusion.why_fields import split_think as _split
    except ImportError:         # the module is not on the path (a bare box checkout): the two common forms
        t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
        t = re.sub(r"<\|channel>.*?<channel\|>", "", t, flags=re.S)
        return t.strip(), ""
    return _split(text)


def strip_think(text: str) -> str:
    """drop the thinking, keep the answer."""
    return split_think(text)[0]


class Sampler(threading.Thread):
    """polls whole-GPU used memory (torch.cuda.mem_get_info) and this process' RSS at 5 Hz; .peak() -> dict."""

    def __init__(self, period: float = 0.2):
        super().__init__(daemon=True)
        self.period, self._stop_evt = period, threading.Event()
        self.vram_base = self.vram_peak = 0.0
        self.rss_peak = 0.0

    def _vram_used_mib(self) -> float:
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            return (total - free) / 2**20
        except Exception:       # noqa: BLE001
            return 0.0

    def run(self):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:       # noqa: BLE001
            proc = None
        self.vram_base = self._vram_used_mib()
        while not self._stop_evt.is_set():
            self.vram_peak = max(self.vram_peak, self._vram_used_mib())
            if proc is not None:
                self.rss_peak = max(self.rss_peak, proc.memory_info().rss / 2**30)
            time.sleep(self.period)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=2)
        return dict(vram_base_mib=round(self.vram_base), vram_peak_mib=round(self.vram_peak), rss_peak_gib=round(self.rss_peak, 2))


# ------------------------------------------------------------------------------------------------ runner
class Runner:
    """one loaded model. load() -> load stats; chat() -> (text, stats); close() frees VRAM."""

    def __init__(self):
        self.llm = None
        self.sig = None
        self.load_s = 0.0

    def load(self, model_path: str, mmproj: str | None, handler: str = "auto", **kw) -> dict:
        sig = (str(model_path), str(mmproj), handler, json.dumps(kw, sort_keys=True))
        if self.llm is not None and sig == self.sig:
            return dict(load_s=0.0, reused=True)
        self.close()
        from llama_cpp import Llama
        lkw = load_kwargs(model_path, mmproj, handler=handler, **kw)
        if mmproj and handler not in ("auto", "generic"):
            lkw["chat_handler"] = make_handler(handler, mmproj, mmproj_gpu=kw.get("mmproj_gpu", True), image_min_tokens=kw.get("image_min_tokens", -1),
                                               image_max_tokens=kw.get("image_max_tokens", -1))
        t0 = time.perf_counter()
        self.llm = Llama(**lkw)
        self.load_s = time.perf_counter() - t0
        self.sig = sig
        return dict(load_s=round(self.load_s, 2), reused=False)

    def chat(self, prompt: str, image_b64: str | None = None, *, system: str = "", max_tokens: int = 4096, temperature: float = 0.7, top_p: float = 0.9,
             top_k: int = 40, repeat_penalty: float = 1.05, seed: int = 1, stop: list[str] | None = None, json_mode: bool = False,
             schema=None) -> tuple[str, dict]:
        """json_mode: llama.cpp's JSON grammar (response_format json_object): the reply can only be ONE JSON object, so a thinking model (Qwen3.x: its chat template opens a
        <think> block, the reasoning then leaked as plain text and ate max_tokens, batch 3060 v005 concept replies) cannot reason out loud before the object.
        schema: "design" | "concept" | a JSON-schema dict | None (see response_format_for): the same grammar, but also with the types / enums / list lengths of the reply
        (tools/fusion/reply_schema.py, 16_app/REPLY_SCHEMA.md). Off by default; thinking must be off (the grammar forbids text before the object)."""
        assert self.llm is not None, "load() first"
        rf = response_format_for(json_mode, schema, prompt)
        msgs = messages_for(prompt, image_b64, system)
        n0 = getattr(self.llm, "n_tokens", 0)
        t0 = time.perf_counter()
        first, parts, n = None, [], 0
        finish = None
        for ch in self.llm.create_chat_completion(messages=msgs, max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p),
                                                  top_k=int(top_k), repeat_penalty=float(repeat_penalty), seed=int(seed), stop=stop or None, stream=True,
                                                  **({"response_format": rf} if rf else {})):
            c = (ch.get("choices") or [{}])[0]
            d = (c.get("delta") or {}).get("content")
            if d:
                if first is None:
                    first = time.perf_counter()
                parts.append(d)
                n += 1
            finish = c.get("finish_reason") or finish
        t1 = time.perf_counter()
        first = first or t1
        total_toks = int(getattr(self.llm, "n_tokens", 0))
        ptoks = max(total_toks - n, 0)
        prompt_s, gen_s = first - t0, max(t1 - first, 1e-6)
        stats = dict(prompt_tokens=ptoks, prompt_s=round(prompt_s, 2), prompt_tok_s=round(ptoks / max(prompt_s, 1e-6), 1), gen_tokens=n, gen_s=round(gen_s, 2),
                     gen_tok_s=round(max(n - 1, 0) / gen_s, 2), total_s=round(t1 - t0, 2), finish=finish, n_ctx=int(self.llm.n_ctx()),
                     grammar=("schema" if rf and "schema" in rf else "json" if rf else "none"))
        return "".join(parts), stats

    def close(self) -> None:
        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:       # noqa: BLE001
                pass
        self.llm, self.sig = None, None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:       # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------ CLI
def _probe(a) -> int:
    import numpy as np
    from PIL import Image
    model = Path(a.model)
    if not model.is_absolute() and not model.exists():
        model = Path(a.root) / a.model
    mm = None if a.mmproj == "none" else (mmproj_for(model) if a.mmproj == "auto" else a.mmproj)
    prompt = Path(a.prompt_file).read_text(encoding="utf-8")
    img = png_b64(np.asarray(Image.open(a.image).convert("RGB"))) if a.image and mm else None
    smp = Sampler()
    smp.start()
    r = Runner()
    t_start = time.perf_counter()
    log_fd = None
    if a.log:                          # capture llama.cpp's own (C-level, stderr) load log: layers offloaded, buffer sizes
        sys.stderr.flush()
        log_fd = os.dup(2)
        os.dup2(os.open(a.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), 2)
    try:
        ls = r.load(str(model), mm, handler=a.handler, n_gpu_layers=a.n_gpu_layers, n_ctx=a.ctx, n_batch=a.n_batch, n_ubatch=a.n_ubatch, kv=a.kv,
                    flash_attn=not a.no_flash, n_cpu_moe=a.n_cpu_moe, n_threads=a.threads, swa_full=a.swa_full, mmproj_gpu=not a.mmproj_cpu,
                    image_max_tokens=a.image_max_tokens, verbose=bool(a.log))
        vram_loaded = smp._vram_used_mib()
        text, st = r.chat(prompt, img, system=a.system, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p, seed=a.seed)
    finally:
        mem = smp.stop()
        r.close()
        if log_fd is not None:
            os.dup2(log_fd, 2)
    stats = dict(model=model.name, mmproj=Path(mm).name if mm else None, n_gpu_layers=a.n_gpu_layers, ctx=a.ctx, kv=a.kv, mmproj_gpu=not a.mmproj_cpu,
                 n_cpu_moe=a.n_cpu_moe, vram_after_load_mib=round(vram_loaded), wall_s=round(time.perf_counter() - t_start, 1), **ls, **st, **mem)
    if a.log:
        txt = Path(a.log).read_text(encoding="utf-8", errors="replace")
        off = parse_offload(txt)
        if off:
            stats["layers_offloaded"], stats["layers_total"] = off
        stats["buffers"] = list(dict.fromkeys(re.sub(r"^\w+:\s+", "", l.strip()) for l in txt.splitlines()
                                              if re.search(r"(model buffer size|KV buffer size|compute buffer size|RS buffer size)", l) and "CPU_Mapped" not in l))[:16]
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.stats:
        Path(a.stats).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("STATS " + json.dumps(stats))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--model", required=True)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--mmproj", default="auto", help="auto | none | path")
    p.add_argument("--handler", default="auto", choices=HANDLERS)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--image")
    p.add_argument("--system", default="")
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--n-cpu-moe", type=int, default=0)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--n-batch", type=int, default=512)
    p.add_argument("--n-ubatch", type=int, default=None)
    p.add_argument("--kv", default="f16", choices=sorted(KV_TYPES))
    p.add_argument("--no-flash", action="store_true")
    p.add_argument("--swa-full", action="store_true")
    p.add_argument("--mmproj-cpu", action="store_true", help="run the vision projector on the CPU (saves ~1 GB VRAM, slower image encode)")
    p.add_argument("--image-max-tokens", type=int, default=-1)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--stats")
    p.add_argument("--log")
    l = sub.add_parser("list")
    l.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for m in list_ggufs(a.root):
            print(m, "| mmproj:", Path(mmproj_for(Path(a.root) / m) or "-").name)
        return 0
    return _probe(a)


if __name__ == "__main__":
    sys.exit(main())
