"""Glue nodes of the THREE-LAYER shot route (tools/app/layered_shots.py builds the graph, 16_app/LAYERS.md explains it).

    MographLayerPrompt   piece, prompt -> prompt without the clauses that ask for letters / strands in the footage (the CLEAN clip must show the scene only)
    MographLayerClean    piece, clean_1..6 (lazy) -> MOGRAPH_CLIPS: barrier, every clean (plain image-to-video) clip of the active shots
    MographLayerGate     latent, after (MOGRAPH_CLIPS) -> latent unchanged: makes a text pass wait for the barrier, so all clean shots run before the
                         first union-control pass, and all text passes before Alpha-Gen (model patches change at most twice)
    MographPlatePrompt   piece, index -> the empty-scene prompt of the Clean Plate LoRA for shot `index` (tools/fusion/plate_layers.empty_scene_prompt)
    MographPlateRef      piece, clean, index, text_1..6 (lazy) -> IMAGE: the clean clip of shot `index` at the Alpha-Gen reference size = the input of the clean-plate V2V chain
                         `p{k}.*`; it waits for every text pass, so the plate stage runs after the union-control stage and before Alpha-Gen
    MographLtxRead       piece, type, clips + the design VLM's llama.cpp settings -> clips (+ `ltx_reads` per layered clip) and the readings (JSON): the VLM
                         reading check of LTX-owned words (tools/fusion/ltx_check.py), between the text pass and compose. model "" = off (pass-through)
    MographLayerGather   piece, request, first?, clean, plates?, text_1..6 + plate_1..6 (lazy) -> MOGRAPH_CLIPS, alpha_ref_1..6, depth_ref_1..6: the same outputs as
                         MographFootageGather, so the flow / Alpha-Gen / depth / compose wiring does not change. STILL PREVIEW (piece.still_preview): the clean barrier asks for no clip and
                         the gather holds each shot's own still (footage.still_outputs): no clean, text or plate chain runs. Per shot the compose clip is
                         u8 = the BACKGROUND (text-rendered clip when the shot has a text pass, else the clean clip), `subject` = the clean u8,
                         `textbg` = bool. The Alpha-Gen and depth references are made from the CLEAN clip.
                         A shot with `plate: "clean"` (and the `plates` switch on: the 24 GB tier) has no text pass: its compose clip is the CLEAN PLATE (the clean clip with the
                         subject taken off, `plate_k`), `subject` = the clean clip, `plate` = True; MographCompose bakes the independent motion (tools/fusion/plate_layers.py).

Compose reads `clip["subject"]` (nodes_fusion/compose.py -> recipe_eval.run_compose(subject=)); the archive saves `sK_clean.mp4` / `sK_textbg.mp4`.
Lazy gating as in MographFootageGather: a text clip is requested only for shots with a text pass (guide > 0) that are linked in this node's own
prompt entry, so a shot with guide 0 never runs a V2V pass.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY, MAX_SHOTS, MographClip, MographClips, MographPiece, MographRequest, MographType, active_sids
from .footage import (ALPHA_REF_SIZE, _is_still, _placeholder, _u8_to_ref, _wired, size_of, still_outputs)

ensure_root_on_path()
from tools.app import layered_shots as LS          # noqa: E402
from tools.fusion import plate_layers as PLT       # noqa: E402
from tools.fusion import plan_rules as PR          # noqa: E402
from tools.fusion import recipe_eval as RE         # noqa: E402
from tools.app import llm_gguf as LG               # noqa: E402


class MographLayerPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLayerPrompt",
            display_name="MOTNGRFX Layer Prompt (clean clip)",
            category=CATEGORY,
            description="The shot prompt for the CLEAN clip: the planner's clauses that ask for letters or strands in the footage ('glowing letters built "
                        "into the scene', 'strands draw out and trace the lines') are removed, everything else (scene, subject, motion, camera) is kept.",
            inputs=[io.String.Input("prompt", multiline=True, default="")],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(LS.clean_prompt(prompt))


class MographLayerClean(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLayerClean",
            display_name="MOTNGRFX Layer Clean Clips",
            category=CATEGORY,
            description="Barrier: waits for the CLEAN clips (plain image-to-video, no type guide) of the active shots only.",
            inputs=[MographPiece.Input("piece"), *[MographClip.Input(f"clean_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)]],
            outputs=[MographClips.Output(display_name="clips")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, **clips) -> list[str]:
        if piece is None or _is_still(piece):
            return []                                            # still preview: no clean (image-to-video) clip is made
        sids = set(active_sids(piece))
        return [n for n, v in clips.items() if v is None and n.replace("clean_", "s") in sids and _wired(cls, n)]

    @classmethod
    def execute(cls, piece: dict, **clips) -> io.NodeOutput:
        return io.NodeOutput([c for _, c in sorted(clips.items()) if c is not None])


class MographLayerGate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLayerGate",
            display_name="MOTNGRFX Layer Gate",
            category=CATEGORY,
            description="Passes the latent through unchanged once `after` (the clean-clip barrier) is done: a text pass starts from the clean clip's "
                        "latent but must not run before every clean shot has run.",
            inputs=[io.Latent.Input("latent"), MographClips.Input("after")],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, after) -> io.NodeOutput:
        return io.NodeOutput(latent)


class MographLayerGather(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLayerGather",
            display_name="MOTNGRFX Layer Gather",
            category=CATEGORY,
            description="Barrier + layer split. Waits for the text-rendered clips (only shots with a text pass), then hands on per shot: the BACKGROUND "
                        "(text clip, else the clean clip) as the compose clip with the clean clip riding along as the subject layer, and the "
                        "Alpha-Gen / depth references cut from the CLEAN clip. Outputs match MOTNGRFX Footage Gather.",
            inputs=[MographPiece.Input("piece"), MographRequest.Input("request"),
                    io.Image.Input("first", optional=True), MographClips.Input("clean", optional=True),
                    io.Boolean.Input("plates", default=False, optional=True,
                                     tooltip="true = a shot with plate clean takes the clean-plate route (plate_k instead of a text pass; 24 GB tier, needs the clean-plate LoRA); false = ordinary layered shots."),
                    *[MographClip.Input(f"text_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
                    *[MographClip.Input(f"plate_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)]],
            outputs=[MographClips.Output(display_name="clips"),
                     *[io.Image.Output(display_name=f"alpha_ref_{k}") for k in range(1, MAX_SHOTS + 1)],
                     *[io.Image.Output(display_name=f"depth_ref_{k}") for k in range(1, MAX_SHOTS + 1)]],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, request=None, first=None, clean=None, plates=False, **texts) -> list[str]:
        if piece is None or _is_still(piece, request):
            return []                                            # still preview: no text pass, no clean plate
        want = {f"text_{s[1:]}" for s in active_sids(piece) if LS.wants_text(piece, s, bool(plates))}
        want |= {f"plate_{s[1:]}" for s in active_sids(piece) if LS.wants_plate(piece, s, bool(plates))}
        return [n for n, v in texts.items() if v is None and n in want and _wired(cls, n)]

    @classmethod
    def execute(cls, piece: dict, request: dict, first=None, clean=None, plates=False, **texts) -> io.NodeOutput:
        if _is_still(piece, request):                                      # still preview: each shot's own still held for the shot length, no layers (graphic type only, no text-rendered background)
            out_clips, alpha, depth = still_outputs(piece, first)
            return io.NodeOutput(out_clips, *alpha, *depth)
        if not clean:
            raise ValueError("MographLayerGather: no clean clips arrived (there is no stand-in footage; the still preview is the quality `still`)")
        alpha_size = size_of(piece, "control", ALPHA_REF_SIZE)
        depth_size = tuple(PR.orient((540, 960), PR.format_of(piece)))
        lay = piece.get("layout") or {}
        by = {c["sid"]: c for c in clean}
        out_clips, alpha, depth = [], [], []
        for k in range(1, MAX_SHOTS + 1):
            sid = f"s{k}"
            if sid not in lay:
                alpha.append(_placeholder())
                depth.append(_placeholder())
                continue
            if sid not in by:
                raise ValueError(f"MographLayerGather {sid}: no clean clip arrived")
            plated = LS.wants_plate(piece, sid, bool(plates))
            if plated and texts.get(f"plate_{k}") is None:
                print(f"[MographLayerGather] {sid} asks for a clean plate but none arrived: ordinary layered shot")
                plated = False
            c = LS.layer_clip(by[sid], None if plated else texts.get(f"text_{k}"), texts.get(f"plate_{k}") if plated else None)
            out_clips.append(c)
            need = RE.shot_needs(piece, sid, layered=True)         # references only for the chains the shot's recipe reads, cut from the CLEAN clip
            if c.get("plate"):
                need = dict(need, matte=True)                      # the subject layer is cut by its own matte: the plate route always needs it
            a = c["subject"].numpy()
            alpha.append(_u8_to_ref(a, alpha_size) if need["matte"] else _placeholder())
            depth.append(_u8_to_ref(a, depth_size) if need["depth"] else _placeholder())
        return io.NodeOutput(out_clips, *alpha, *depth)


class MographPlatePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPlatePrompt",
            display_name="MOTNGRFX Plate Prompt (empty scene)",
            category=CATEGORY,
            description="The Clean Plate LoRA prompt of shot `index`: 'An empty clean plate of the exact same location ... No <subject> ...' from the shot prompt "
                        "(shot keys plate_prompt / plate_subject override it). Only runs for a shot that asks for a plate.",
            inputs=[MographPiece.Input("piece"), io.Int.Input("index", default=1, min=1, max=MAX_SHOTS)],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, piece: dict, index: int) -> io.NodeOutput:
        return io.NodeOutput(PLT.empty_scene_prompt(piece, f"s{int(index)}"))


class MographPlateRef(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPlateRef",
            display_name="MOTNGRFX Plate Reference (clean clip)",
            category=CATEGORY,
            description="The CLEAN clip of shot `index` at the Alpha-Gen reference size: the video the Clean Plate IC-LoRA takes the subject off. Waits for every text pass "
                        "(the plate stage is its own stage: after the union-control stage, before Alpha-Gen). Only evaluated for a shot that asks for a plate.",
            inputs=[MographPiece.Input("piece"), MographClips.Input("clean"), io.Int.Input("index", default=1, min=1, max=MAX_SHOTS),
                    *[MographClip.Input(f"text_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)]],
            outputs=[io.Image.Output(display_name="reference")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, clean=None, index=1, **texts) -> list[str]:
        if piece is None:
            return []
        want = {f"text_{s[1:]}" for s in active_sids(piece) if LS.wants_text(piece, s, True)}
        return [n for n, v in texts.items() if v is None and n in want and _wired(cls, n)]

    @classmethod
    def execute(cls, piece: dict, clean, index: int, **texts) -> io.NodeOutput:
        sid = f"s{int(index)}"
        by = {c["sid"]: c for c in clean or []}
        if sid not in by:
            return io.NodeOutput(_placeholder())
        return io.NodeOutput(_u8_to_ref(by[sid]["u8"].numpy(), size_of(piece, "control", ALPHA_REF_SIZE)))


class MographLtxRead(io.ComfyNode):
    """The VLM legibility check of LTX-owned words (replaces the pixel test as the DECIDER, tools/fusion/ltx_check.py): per layered shot with a text-rendered clip, 2-3
    crops of the clip at each ltx / handoff word's matte box, the design VLM reads them, the readings ride on the clip (`ltx_reads`) and MographCompose lets
    `ltx_check.apply_fallback` turn the words it could not read to flat. llama.cpp is outside ComfyUI's memory manager: ComfyUI's models are freed first (LTX is done by
    now), the GGUF is loaded once for all crops and unloaded after. `model` empty = off: the clips pass through, the pixel test stays the decider. tools/app/llm_arms.py
    fills the settings from the arm (the same model as the design node)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLtxRead",
            display_name="MOTNGRFX LTX Read (VLM legibility check)",
            category=CATEGORY,
            description="Reads back the text-rendered clip at each LTX-owned word with the design VLM; a word it cannot read exactly is drawn flat by compose. "
                        "model empty = off. Frees ComfyUI's models before the llama.cpp model loads.",
            inputs=[
                MographPiece.Input("piece"), MographType.Input("type"), MographClips.Input("clips"),
                io.String.Input("model", default="", tooltip="GGUF path relative to ComfyUI/models/LLM/GGUF; empty = off (llm_arms fills it with the design model)"),
                io.String.Input("mmproj", default="auto"),
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=300),
                io.Int.Input("n_ctx", default=16384, min=1024, max=131072, step=512),
                io.Combo.Input("kv_cache", options=list(LG.KV_TYPES), default="q8_0"),
                io.Boolean.Input("flash_attn", default=True),
                io.Boolean.Input("mmproj_on_gpu", default=True),
                io.Int.Input("n_cpu_moe", default=0, min=0, max=200),
                io.Combo.Input("handler", options=list(LG.HANDLERS), default="auto"),
                io.Int.Input("image_max_tokens", default=-1, min=-1, max=16384),
                io.Int.Input("top_k", default=40, min=0, max=200),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.05),
                io.Float.Input("repeat_penalty", default=1.0, min=1.0, max=2.0, step=0.01),
                io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF),
            ],
            outputs=[MographClips.Output(display_name="clips"), io.String.Output(display_name="reads")],
        )

    @classmethod
    def execute(cls, piece, type, clips, model="", mmproj="auto", n_gpu_layers=-1, n_ctx=16384, kv_cache="q8_0", flash_attn=True, mmproj_on_gpu=True, n_cpu_moe=0,
                handler="auto", image_max_tokens=-1, top_k=40, top_p=0.9, repeat_penalty=1.0, seed=1) -> io.NodeOutput:
        import json
        import os
        from .llm_gguf import free_vram, gguf_root
        from .type import wait_type
        from tools.fusion import ltx_check as LC
        from tools.app import llm_gguf as LG_
        if not model or ((piece.get("layers") or {}).get("ltx_read") is False):
            return io.NodeOutput(clips, "{}")
        wait_type(type)
        jobs_by = {}
        for c in clips:
            if c.get("subject") is not None and c.get("textbg"):
                jobs = LC.crop_jobs(piece, c["sid"], type["dir_layers"], c["u8"])
                if jobs:
                    jobs_by[c["sid"]] = jobs
        if not jobs_by:
            return io.NodeOutput(clips, "{}")
        root = gguf_root()
        mpath = os.path.join(root, model)
        if not os.path.isfile(mpath):
            raise FileNotFoundError(f"[MographLtxRead] model not found: {mpath}")
        mm = None if mmproj == "none" else (LG_.mmproj_for(mpath) if mmproj == "auto" else os.path.join(root, mmproj))
        if mm is None:
            print("[MographLtxRead] the model has no projector: no reading, the pixel test decides")
            return io.NodeOutput(clips, "{}")
        free_vram()
        runner = LG_.Runner()
        reads = {}
        try:
            runner.load(mpath, mm, handler=handler, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, kv=kv_cache, flash_attn=flash_attn, n_cpu_moe=n_cpu_moe,
                        mmproj_gpu=mmproj_on_gpu, image_max_tokens=image_max_tokens, verbose=False)
            for sid, jobs in jobs_by.items():
                replies = []
                for j in jobs:
                    text, _ = runner.chat(LC.READ_PROMPT, LG_.png_b64(j["image"]), max_tokens=64, temperature=0.0, top_p=top_p, top_k=top_k,
                                          repeat_penalty=repeat_penalty, seed=seed)
                    replies.append(LG_.strip_think(text).strip())
                reads[sid] = LC.reads_from(jobs, replies)
                for r in reads[sid]:
                    print(f"[MographLtxRead] {sid} {r['text']!r}: read {r['readings']}")
        finally:
            runner.close()
        out = [dict(c, ltx_reads=reads[c["sid"]]) if c["sid"] in reads else c for c in clips]
        return io.NodeOutput(out, json.dumps(reads))


NODES = [MographLayerPrompt, MographLayerClean, MographLayerGate, MographLayerGather, MographPlatePrompt, MographPlateRef, MographLtxRead]
