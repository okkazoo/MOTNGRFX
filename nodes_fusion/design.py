"""The open-model design step (NO CLAUDE: design_llm_8b in DesignLLM_Advanced + code). Thin wrappers over ``tools/fusion/design_step.py``.

    MographDesignPackage  request, music, piece, clips, guide, seed, matte_k? / depth_k? (lazy) -> prompt, images, locked, sheet
    MographDesignParse    reply, locked, piece, music?, prompt?  -> piece_json, ok, repair_prompt, report
    MographDesignRepair   piece_json, ok, repair_reply (LAZY: only when ok is false), piece, music?, locked? -> piece_json, report

Graph (tools/app/build_graphs.py ``fusion_design_qwen``):
    ... footage flow -> Package -> DesignLLM_Advanced (design) -> Parse -> [DesignLLM_Advanced (repair), only if Parse says ok = false]
        -> Repair -> MographPieceFromJSON -> type passes -> compose -> assemble

Parse reads every field of the reply on its own (tools/fusion/design_salvage.py): a valid field stays as the model wrote it, an invalid one is mended alone
(nearest allowed value, else the rules' value for that field); only a field with no obvious value goes to the repair call, which gets just those fields.

Package waits for the mattes / depth the PLAN's recipe reads (lazy inputs, same rule as MographCompose), so the model sees the matte sheet and
the alpha / depth chains have run before the model loads. Parse and Repair never raise on a bad reply: piece_json is always a valid piece
(see design_step.py for how the sections fall back to the rules' plan).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY, MAX_SHOTS, MographClips, MographMusic, MographPiece, MographRequest

ensure_root_on_path()
from tools.fusion import design_step as DS       # noqa: E402
from tools.fusion import recipe_eval as RE       # noqa: E402


def _wired(cls, name: str) -> bool:
    """is input ``name`` linked in this node's prompt entry? (requesting an unlinked lazy input raises in the executor)"""
    hid = getattr(cls, "hidden", None)
    prompt = getattr(hid, "prompt", None)
    if not prompt:
        return True
    return isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)


def free_vram() -> None:
    """ComfyUI's models out of VRAM, caches emptied: DesignLLM loads its model with transformers (outside ComfyUI's model manager), so on a
    12 GB card the LTX / depth / music models must be gone before it starts. Harmless without CUDA / outside ComfyUI."""
    try:
        import gc
        import comfy.model_management as mm
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:   # noqa: BLE001
        print(f"[MographDesign] free_vram skipped: {type(e).__name__}: {e}")


def _img(a: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] or [N,H,W,3] -> float32 IMAGE [N,H,W,3]."""
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[None]
    return torch.from_numpy(a.astype(np.float32) / 255.0)


class MographDesignPackage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographDesignPackage",
            display_name="MOTNGRFX Design Package",
            category=CATEGORY,
            description="Everything the open model needs to design the type and the compose look: the prompt (design guide, words, music analysis "
                        "as text, shot list, module menu, liked fonts, JSON schema) and the images (a 6-frame contact sheet per generated shot + "
                        "a matte sheet). `images` = the sheets as a padded batch; `sheet` = the same sheets in ONE image (wire this one into "
                        "DesignLLM_Advanced: it takes a single image). `locked` = JSON of the fields the model may not change.",
            inputs=[
                MographRequest.Input("request"),
                MographMusic.Input("music"),
                MographPiece.Input("piece", tooltip="The rules' plan: the fallback of every section the model gets wrong."),
                MographClips.Input("clips", tooltip="The generated footage (after the footage flow)."),
                io.Combo.Input("guide", options=DS.guide_options(), default=DS.DEFAULT_GUIDE,
                               tooltip="16_app/DESIGN_GUIDE*.md pasted at the top of the prompt."),
                io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF, tooltip="Design seed shown to the model (the guide picks the concept from it)."),
                *[io.Image.Input(f"matte_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
                *[io.Image.Input(f"depth_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
            ],
            outputs=[io.String.Output(display_name="prompt"), io.Image.Output(display_name="images"),
                     io.String.Output(display_name="locked"), io.Image.Output(display_name="sheet")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, request=None, music=None, piece=None, clips=None, guide=None, seed=None, **imgs) -> list[str]:
        if piece is None or not clips:
            return []
        need = []
        for clip in clips:
            sid = clip["sid"]
            n = RE.shot_needs(piece, sid)
            for kind in ("matte", "depth"):
                name = f"{kind}_{sid[1:]}"
                if n[kind] and imgs.get(name) is None and _wired(cls, name):
                    need.append(name)
        return need

    @classmethod
    def execute(cls, request, music, piece, clips, guide=DS.DEFAULT_GUIDE, seed=1, **imgs) -> io.NodeOutput:
        t0 = time.time()
        free_vram()                                                  # the next node is the 4-bit Qwen
        mattes = {c["sid"]: imgs.get(f"matte_{c['sid'][1:]}") for c in clips}
        pk = DS.package(request, music, piece, clips, mattes, guide, int(seed))
        print(f"[MographDesignPackage] prompt {len(pk['prompt'])} chars, {len(pk['images'])} sheets {pk['sheet'].shape[1]}x{pk['sheet'].shape[0]} "
              f"in {time.time() - t0:.1f}s; techniques allowed: {', '.join(pk['techniques'])}")
        return io.NodeOutput(pk["prompt"], _img(np.stack(pk["images"])), pk["locked"], _img(pk["sheet"]))


class MographDesignParse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographDesignParse",
            display_name="MOTNGRFX Design Parse",
            category=CATEGORY,
            description="The model's reply -> a valid piece. Extracts the JSON, validates it with the repo's validator (liked fonts, modules, "
                        "ranges, the type engine's own dry run), applies the fixes (clamps, timing compiler, deterministic timing fit) and writes "
                        "the locked fields back. ok = the design needed nothing from the rules; when it did, piece_json already holds the rules' "
                        "sections for what failed and repair_prompt (prompt + the errors) is for a second model call.",
            inputs=[
                io.String.Input("reply", multiline=True, default=""),
                io.String.Input("locked", multiline=True, default=""),
                MographPiece.Input("piece", tooltip="The rules' plan (fallback)."),
                MographMusic.Input("music", optional=True, tooltip="Without it the beat grid is rebuilt from the piece."),
                io.String.Input("prompt", multiline=True, default="", optional=True, tooltip="The design prompt; used for the long repair prompt (no usable JSON / a missing section)."),
                io.String.Input("model", default=DS.MODEL, optional=True, tooltip="Name of the model arm, written into plan.design.model."),
            ],
            outputs=[io.String.Output(display_name="piece_json"), io.Boolean.Output(display_name="ok"),
                     io.String.Output(display_name="repair_prompt"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, reply, locked, piece, music=None, prompt="", model=DS.MODEL) -> io.NodeOutput:
        t0 = time.time()
        r = DS.parse(reply, locked, piece, music, prompt or "", model or DS.MODEL)
        print(f"[MographDesignParse] ok={r['ok']} in {time.time() - t0:.1f}s\n{r['report']}")
        return io.NodeOutput(r["piece_json"], bool(r["ok"]), r["repair_prompt"], r["report"])


class MographDesignRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographDesignRepair",
            display_name="MOTNGRFX Design Repair",
            category=CATEGORY,
            description="Final design check. ok = true: piece_json passes through and `repair_reply` is never requested (the second model call "
                        "never runs). ok = false: the repair reply is validated the same way and replaces whatever sections it gets right; "
                        "whatever is still invalid (a whole missing `type` or `compose` included) is filled from the rules' plan "
                        "(plan_rules). The output is always a valid piece.",
            inputs=[
                io.String.Input("piece_json", multiline=True, default=""),
                io.Boolean.Input("ok", default=True),
                io.String.Input("repair_reply", multiline=True, default="", optional=True, lazy=True),
                MographPiece.Input("piece", tooltip="The rules' plan (last-resort fill)."),
                MographMusic.Input("music", optional=True),
                io.String.Input("locked", multiline=True, default="", optional=True),
                io.String.Input("model", default=DS.MODEL, optional=True),
            ],
            outputs=[io.String.Output(display_name="piece_json"), io.String.Output(display_name="report")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece_json=None, ok=None, repair_reply=None, piece=None, music=None, locked=None, model=None) -> list[str]:
        if ok is not None and not ok and repair_reply is None and _wired(cls, "repair_reply"):
            return ["repair_reply"]
        return []

    @classmethod
    def execute(cls, piece_json, ok, piece, repair_reply=None, music=None, locked="", model=DS.MODEL) -> io.NodeOutput:
        t0 = time.time()
        r = DS.repair(piece_json, bool(ok), repair_reply, piece, music, locked, model or DS.MODEL)
        from tools.fusion import user_layout as UL            # every piece: "more legible text" re-fits the designed layout; else at most ONE rare mid-word split (stable_type / plan_rules never read it)
        r["piece_json"], ul_fixes = UL.apply_json(r["piece_json"])
        if ul_fixes:
            r["report"] = f"{r['report']}\n" + "\n".join("  fix: " + x for x in ul_fixes)
        print(f"[MographDesignRepair] {time.time() - t0:.1f}s\n{r['report']}")
        return io.NodeOutput(r["piece_json"], r["report"])


NODES = [MographDesignPackage, MographDesignParse, MographDesignRepair]
