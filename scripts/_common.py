"""Shared helpers for scripts/download_models.py and scripts/check_install.py (stdlib only).

``MODELS`` is the single table of every model file the app graphs load: ComfyUI sub-folder, size, download URL and whether
the Hugging Face repo is gated. URLs and sizes come from the project's model card (HEAD-checked 2026-10-22 / 2026-10-26).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = ROOT / "app" / "graphs"
PACKS_JSON = ROOT / "web_app" / "node_packs.json"      # also served to the page at /mograph/static/node_packs.json

MODEL_EXTS = (".safetensors", ".pth", ".pt", ".ckpt", ".gguf", ".bin")

_HF = "https://huggingface.co"
_ACE = f"{_HF}/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/split_files"
_MOSS = f"{_HF}/Comfy-Org/MOSS-Image/resolve/main"
_LTX = f"{_HF}/Lightricks/LTX-2.5/resolve/main"

# filename -> {folder (under ComfyUI/models), size (bytes), url, gated, what}
MODELS: dict[str, dict] = {
    "acestep_v1.5_xl_base_bf16.safetensors": dict(folder="diffusion_models", size=9974719930, what="ACE-Step 1.5 XL base (music, text-to-music)",
                                                  url=f"{_ACE}/diffusion_models/acestep_v1.5_xl_base_bf16.safetensors"),
    "qwen_0.6b_ace15.safetensors": dict(folder="text_encoders", size=1191588248, what="ACE-Step text encoder A",
                                        url=f"{_ACE}/text_encoders/qwen_0.6b_ace15.safetensors"),
    "qwen_4b_ace15.safetensors": dict(folder="text_encoders", size=8379154232, what="ACE-Step text encoder B (XL)",
                                      url=f"{_ACE}/text_encoders/qwen_4b_ace15.safetensors"),
    "ace_1.5_vae.safetensors": dict(folder="vae", size=337431732, what="ACE-Step VAE",
                                    url=f"{_ACE}/vae/ace_1.5_vae.safetensors"),
    "moss_image_sft_int8_convrot.safetensors": dict(folder="diffusion_models", size=5413698958, what="MOSS-Image SFT int8 (first-frame still)",
                                                    url=f"{_MOSS}/diffusion_models/moss_image_sft_int8_convrot.safetensors"),
    "qwen3.5_9b_w4a8_gptq.safetensors": dict(folder="text_encoders", size=7343591088, what="MOSS-Image text encoder (Qwen3.5 9B w4a8)",
                                             url=f"{_MOSS}/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"),
    "flux2-vae_small.safetensors": dict(folder="vae", size=249521340, what="MOSS-Image VAE",
                                        url=f"{_MOSS}/vae/flux2-vae_small.safetensors"),
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": dict(
        folder="diffusion_models", size=21504034224, gated=True, what="LTX 2.5 22B distilled int8 (footage)",
        url=f"{_LTX}/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors": dict(
        folder="text_encoders", size=15372969374, gated=True, what="LTX 2.5 text encoder (Gemma 4 12B int8)",
        url=f"{_LTX}/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    "ltx-2.5-video-vae-bf16.safetensors": dict(folder="vae", size=1472223346, gated=True, what="LTX 2.5 video VAE",
                                               url=f"{_LTX}/vae/ltx-2.5-video-vae-bf16.safetensors"),
    "ltx-2.5-audio-vae-bf16.safetensors": dict(folder="vae", size=364866540, gated=True, what="LTX 2.5 audio VAE",
                                               url=f"{_LTX}/vae/ltx-2.5-audio-vae-bf16.safetensors"),
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": dict(
        folder="latent_upscale_models", size=995778752, gated=True, what="LTX 2.5 spatial upscaler x2",
        url=f"{_LTX}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    "RealESRGAN_x2plus.pth": dict(folder="upscale_models", size=67061725, gated=False, what="Real-ESRGAN x2plus (Final upscale 540x960 -> 1080x1920, BSD-3)",
                                  url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    "RealESRGAN_x4plus.pth": dict(folder="upscale_models", size=67040989, gated=False, what="Real-ESRGAN x4plus (Final upscale from 360x640, BSD-3)",
                                  url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors": dict(
        folder="loras", size=1308787472, gated=True, what="LTX 2.5 IC-LoRA Alpha-Gen (type matte, gated repo)",
        url=f"{_HF}/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"),
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors": dict(
        folder="loras", size=654465352, what="LTX IC-LoRA Union-Control (depth guide)",
        url=f"{_HF}/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
    "gemma-4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf": dict(
        folder="LLM/GGUF", size=17287670048, what="Gemma 4 31B QAT Q4_K_XL (design model, default graph)",
        url=f"{_HF}/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"),
    "gemma-4-31b-qat/mmproj-F16.gguf": dict(
        folder="LLM/GGUF", size=1198957024, what="Gemma 4 31B vision projector",
        url=f"{_HF}/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/mmproj-F16.gguf"),
    "gemma-4-12b/gemma-4-12b-it-Q6_K.gguf": dict(
        folder="LLM/GGUF", size=9786022720, what="Gemma 4 12B Q6_K (design model arm)",
        url=f"{_HF}/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q6_K.gguf"),
    "gemma-4-12b/gemma-4-12b-it-Q8_0.gguf": dict(
        folder="LLM/GGUF", size=12669647680, what="Gemma 4 12B Q8_0 (16 GB tier design model)",
        url=f"{_HF}/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q8_0.gguf"),
    "gemma-4-12b/mmproj-F16.gguf": dict(
        folder="LLM/GGUF", size=175115840, what="Gemma 4 12B vision projector",
        url=f"{_HF}/unsloth/gemma-4-12b-it-GGUF/resolve/main/mmproj-F16.gguf"),
    "moss_image_sft_bf16.safetensors": dict(folder="diffusion_models", size=10164655808, what="MOSS-Image SFT bf16 (first-frame still, 24 GB tier)",
                                            url=f"{_MOSS}/diffusion_models/moss_image_sft_bf16.safetensors"),
    "qwen3.5_9b_bf16.safetensors": dict(folder="text_encoders", size=19306312328, what="MOSS-Image text encoder (Qwen3.5 9B bf16, 24 GB tier)",
                                        url=f"{_MOSS}/text_encoders/qwen3.5_9b_bf16.safetensors"),
    "qwen3.5_9b_int8_convrot.safetensors": dict(folder="text_encoders", size=9748121532, what="MOSS-Image text encoder (Qwen3.5 9B int8, 16 GB tier)",
                                                url=f"{_MOSS}/text_encoders/qwen3.5_9b_int8_convrot.safetensors"),
    "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors": dict(
        folder="loras", size=327318439, gated=True, what="LTX 2.5 IC-LoRA Clean-Plate (subject-free background plate, gated repo: needs HF_TOKEN)",
        url=f"{_HF}/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate/resolve/main/ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"),
    "video_depth_anything_vits.pth": dict(
        folder="videodepthanything", size=116440756, what="Video Depth Anything (small)",
        url=f"{_HF}/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"),
}

# Optional extras, downloaded ONLY with ``download_models.py --extras <key>``; never part of the default set.
# name -> {folder, size, url, dest (the file name the graphs use), what, licence}. The repo file name differs from ``dest``.
EXTRAS: dict[str, dict] = {
    "angles": dict(
        folder="loras", dest="moss_image_edit_chinoll.safetensors", size=297382216,
        url=f"{_HF}/chinoll/Moss-Image-Edit/resolve/main/adapter_model.safetensors",
        what="MOSS edit LoRA (same-subject new-angle stills; only for graphs built with `build_graphs.py --angles`)",
        licence="UNDECLARED: the repo has no licence. Not part of the default install; do not ship or redistribute it."),
}

# Graphs the downloader skips: no public source recorded for their files (an archived A/B graph). Listed explicitly so they never look like an oversight.
NO_DOWNLOAD_GRAPHS = {"fusion_redesign_gemma"}


def graph_files(patterns: list[str] | None = None, downloadable: bool = False) -> list[Path]:
    """API-format graphs (not sidecars). Default: the GPU app graphs ``fusion*.json``. ``downloadable=True`` drops NO_DOWNLOAD_GRAPHS."""
    pats = patterns or ["fusion*.json"]
    out: list[Path] = []
    for pat in pats:
        p = Path(pat)
        hits = [p] if p.is_file() else [Path(x) for x in sorted(glob.glob(str(GRAPHS_DIR / pat)))]
        if not hits and not pat.endswith(".json"):
            hits = [Path(x) for x in sorted(glob.glob(str(GRAPHS_DIR / (pat + ".json"))))]
        out += [h for h in hits if not h.name.endswith(".app.json")]
    if downloadable:
        out = [h for h in out if h.stem not in NO_DOWNLOAD_GRAPHS]
    return sorted(dict.fromkeys(out))


def load_graph(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def graph_classes(graph: dict) -> dict[str, list[str]]:
    """class_type -> [node titles] of an API-format graph."""
    out: dict[str, list[str]] = {}
    for nid, n in graph.items():
        if isinstance(n, dict) and "class_type" in n:
            out.setdefault(n["class_type"], []).append((n.get("_meta") or {}).get("title") or nid)
    return out


def graph_model_refs(graph: dict) -> list[tuple[str, str, str, str]]:
    """(class_type, input name, filename, node title) for every literal model-file input in an API-format graph."""
    refs = []
    for nid, n in graph.items():
        if not isinstance(n, dict):
            continue
        for k, v in (n.get("inputs") or {}).items():
            if isinstance(v, str) and v.lower().endswith(MODEL_EXTS):
                refs.append((n.get("class_type", "?"), k, v, (n.get("_meta") or {}).get("title") or nid))
    return refs


def model_files(graphs: list[Path]) -> dict[str, list[str]]:
    """filename -> [graph names that use it]."""
    out: dict[str, list[str]] = {}
    for g in graphs:
        for _c, _k, f, _t in graph_model_refs(load_graph(g)):
            out.setdefault(f, [])
            if g.stem not in out[f]:
                out[f].append(g.stem)
    return out


def load_packs() -> dict:
    return json.loads(PACKS_JSON.read_text(encoding="utf-8"))


def pack_for_class(class_type: str, packs: dict | None = None) -> dict | None:
    """The custom node pack entry (node_packs.json) that provides ``class_type``, else None."""
    import re
    packs = packs or load_packs()
    for p in packs["packs"]:
        if class_type in p.get("classes", []) or any(re.search(rx, class_type) for rx in p.get("patterns", [])):
            return p
    return None


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):          # 1024-based, so labelled as such
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n} B"


def find_comfy(cli_value: str | None) -> Path | None:
    """ComfyUI folder: --comfy, $COMFYUI_PATH, the pack's grand-parent (custom_nodes/<pack>), or ./ComfyUI."""
    cands = [cli_value, os.environ.get("COMFYUI_PATH"), os.environ.get("MOGRAPH_COMFYUI_PATH"),
             str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.cwd() / "ComfyUI"), str(Path.cwd())]
    for c in cands:
        if c and (Path(c) / "main.py").is_file() and (Path(c) / "comfy").is_dir():
            return Path(c).resolve()
    return None
