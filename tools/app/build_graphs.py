#!/usr/bin/env python
"""Generate the fusion app graphs (API format) from the box-proven LTX IC-LoRA graphs.

    python tools/app/build_graphs.py                 # write app/graphs/fusion.json (+ .app.json sidecar)
    python tools/app/build_graphs.py --check         # build in memory, validate, compare with the files on disk
    python tools/app/build_graphs.py --check --server http://127.0.0.1:8190
                                                     # also validate classes / inputs / link types against /object_info
    python tools/app/build_graphs.py --fill --server http://127.0.0.1:8191
                                                     # write, filling missing required widget inputs with server defaults

Design: 16_app/DESIGN.md sections 1-3, 5. The control (union IC-LoRA, two-stage 1152x2048) and alpha (576x1024) graphs
come from ``tools.fusion.ltx_ic.build``; their ids get the prefixes ``c{k}.`` / ``a{k}.`` per shot slot k (loaders and
other constants stay shared), the file-based nodes are replaced by slot / gather outputs, and ``ltx_ic.check_graph`` runs
on the result. The page binds fields and outputs by ``_meta.title``:

    IN words, IN scene, IN image, IN audio, IN music, IN length, IN look, IN seed, IN quality, IN format
    OUT still, OUT type, OUT final
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import ltx_ic  # noqa: E402
from tools.fusion import music_graph  # noqa: E402
from tools.fusion import angle_still  # noqa: E402
from tools.fusion import places  # noqa: E402
from tools.fusion import still_preview  # noqa: E402


def _load_optional_nodes():
    spec = importlib.util.spec_from_file_location("mograph_optional_nodes", os.path.join(ROOT, "tools", "app", "optional_nodes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


optional_nodes = _load_optional_nodes()

GRAPHS = os.path.join(ROOT, "app", "graphs")
SLOTS = 6
SIZE_CONTROL = (1152, 2048)
SIZE_ALPHA = (576, 1024)
SIZE_PREVIEW = (384, 704)                          # preview tier IC control, ONE stage (plan_rules.sizes_for("preview")["control"])
SLOT_FRAMES = 121                                  # placeholder; node 14 is driven by the slot

IN_TITLES = ["IN words", "IN scene", "IN image", "IN images", "IN audio", "IN music", "IN length", "IN look", "IN seed", "IN quality",
             "IN format"]
# the "Add media" box (16_app/MEDIA_BOX.md): ONE sidecar field replaces `image` + `audio`; it maps to these IN nodes (IN brief only exists in the auto graphs: the concept step reads it)
MEDIA_FIELD = {"id": "media", "kind": "media", "label": "Add media", "limits": {"video": 1, "image": 5, "audio": 1},
               "image_node": "IN image", "images_node": "IN images", "audio_node": "IN audio", "brief_node": "IN brief",
               "read_graph": "_media_read", "own_video_route": "/mograph/own_video"}
OUT_TITLES = ["OUT still", "OUT type", "OUT final", "OUT final 1080"]

STILL_MODELS = ("moss",)                           # MOSS-Image (Apache 2.0, ungated)

# model files (names as on the box; see 15_fusion v002 graphs)
# music: ACE-Step 1.5 XL base text-to-music, models and settings live in tools/fusion/music_graph.py (T2M_UNET, T2M_CLIPS, T2M_STEPS)
MOSS_UNET = "moss_image_sft_int8_convrot.safetensors"          # as workflows/api/moss_t2i.json (box-proven)
MOSS_CLIP = "qwen3.5_9b_w4a8_gptq.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
VDA_MODEL = "video_depth_anything_vits.pth"         # Small only: the Base / Large weights are CC-BY-NC (asserted by check())

# output counts of the non-LTX classes used here, so ltx_ic.check_graph also range-checks their links
EXTRA_CLASSES = {
    "PrimitiveStringMultiline": ("ComfyUI core", 1), "PrimitiveString": ("ComfyUI core", 1),
    "DualCLIPLoader": ("ComfyUI core", 1), "ModelSamplingAuraFlow": ("ComfyUI core", 1),
    "ConditioningZeroOut": ("ComfyUI core", 1), "EmptyAceStep1.5LatentAudio": ("ComfyUI core", 1),
    "KSampler": ("ComfyUI core", 1), "VAEDecodeAudio": ("ComfyUI core", 1), "EmptyLatentImage": ("ComfyUI core", 1),
    "VAEDecode": ("ComfyUI core", 1), "PreviewImage": ("ComfyUI core", 0), "EmptyFlux2LatentImage": ("ComfyUI core", 1),
    "DesignLLM_Advanced": ("the design-LLM node pack", 1),
    "LoadVideoDepthAnythingModel": ("ComfyUI-VideoDepthAnything", 1),
    "VideoDepthAnythingProcess": ("ComfyUI-VideoDepthAnything", 1),
    "VideoDepthAnythingOutput": ("ComfyUI-VideoDepthAnything", 1),
    "MographOptionalImage": ("ComfyUI-Mograph", 1), "MographOptionalAudio": ("ComfyUI-Mograph", 1), "MographUserImages": ("ComfyUI-Mograph", 1),
    "MographRequest": ("ComfyUI-Mograph", 1), "MographFusionBriefPrompt": ("ComfyUI-Mograph", 2),
    "MographMusicEncode": ("ComfyUI-Mograph", 3), "MographMusicCut": ("ComfyUI-Mograph", 2),
    "MographMaster": ("ComfyUI-Mograph", 1), "MographPiecePlan": ("ComfyUI-Mograph", 4),
    "MographFirstFrame": ("ComfyUI-Mograph", 1), "MographTypePasses": ("ComfyUI-Mograph", 2),
    "MographStillPrompt": ("ComfyUI-Mograph", 2), "MographFirstFrames": ("ComfyUI-Mograph", 1),
    "MographAngleRef": ("ComfyUI-Mograph", 1), "LoraLoaderModelOnly": ("ComfyUI core", 1), "TextEncodeMOSSImageEdit": ("ComfyUI core", 1),
    "MographShotSlot": ("ComfyUI-Mograph", 7), "MographDecodeClip": ("ComfyUI-Mograph", 2),
    "MographFootageGather": ("ComfyUI-Mograph", 1 + 2 * SLOTS), "MographCompose": ("ComfyUI-Mograph", 2),
    "MographAssemble": ("ComfyUI-Mograph", 1), "MographPieceSizes": ("ComfyUI-Mograph", 8),
    "MographFootageFlow": ("ComfyUI-Mograph", 1), "MographPivotCut": ("ComfyUI-Mograph", 1),
    "MographScoreSkeleton": ("ComfyUI-Mograph", 3), "MographMusicStyleEncode": ("ComfyUI-Mograph", 1),     # deprecated route, still registered
    "MographMusicFit": ("ComfyUI-Mograph", 2), "MographMusicRoute": ("ComfyUI-Mograph", 2),
    "MographMusicT2MEncode": ("ComfyUI-Mograph", 3), "MographMusicT2MCut": ("ComfyUI-Mograph", 2),
    "MographTameHighs": ("ComfyUI-Mograph", 1),
    "MographLiveWaveformPreview": ("ComfyUI-Mograph", 1),                                        # live per-step waveform of the music sampler (web page)
    "VAEEncodeAudio": ("ComfyUI core", 1), "PrimitiveFloat": ("ComfyUI core", 1),
    "MographPieceFromJSON": ("ComfyUI-Mograph", 1), "MographDesignPackage": ("ComfyUI-Mograph", 4),
    "MographDesignParse": ("ComfyUI-Mograph", 4), "MographDesignRepair": ("ComfyUI-Mograph", 2),
    "PreviewAny": ("ComfyUI core", 0),
    "MographArchive": ("ComfyUI-Mograph", 0), "MographArchiveLoad": ("ComfyUI-Mograph", 6 + SLOTS), "TextGenerate": ("ComfyUI core", 1),
    "UpscaleModelLoader": ("ComfyUI core", 1), "ImageUpscaleWithModel": ("ComfyUI core", 1), "ImageScale": ("ComfyUI core", 1),
    "MographSaveFinalUpscaled": ("ComfyUI-Mograph", 0),
    "MographConceptPrompt": ("ComfyUI-Mograph", 1), "MographConceptParse": ("ComfyUI-Mograph", 3), "MographConceptFinal": ("ComfyUI-Mograph", 8),
}
ltx_ic.CLASSES.update(EXTRA_CLASSES)

TPL: dict = {}                                     # LTX templates (control, alpha) of the graph being built

# ---- fusion_design_qwen: the design step by design_llm_8b (NO Claude), one prompt request -> final mp4, PREVIEW tier, RTX 3060 (12 GB)
DESIGN_NAME = "fusion_design_qwen"
AUTO_NAME = "fusion_design_qwen_auto"              # the same pipeline behind a design_llm concept step: the model invents the request
AUTO_LENGTH = 16                                   # the batch driver (tools/app/run_batch.py) fixes the length so the timings compare
QWEN_CONCEPT_TOKENS, QWEN_CONCEPT_TEMP = 700, 0.9
QWEN_MODEL = "design_llm_8b"
QWEN_QUANT = "4-bit (VRAM-friendly)"               # the design-LLM node pack option (bitsandbytes); "None (FP16)" does not fit 12 GB next to the rest
QWEN_DESIGN_TOKENS = 4096                          # the design JSON of 3-6 words is ~2.5k tokens
QWEN_DESIGN_TEMP = 0.7
# the request of the test run: P06's words (15_fusion/pieces/P06_v003.json type.words) and scene (stills[0].prompt), verbatim; music = P06's
# mood brief (qwen_piece.PIECES["P06"]["mood_brief"]) + its bpm (P06_v003.json audio.bpm); see 16_app/RUN_3060.md
TEST_REQUEST = dict(
    words="STILL\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING",
    scene=("a lone runner in dark athletic clothes sprinting along a long concrete tunnel, seen from the side in mid stride at the left of frame, "
           "cold blue and cyan light from strip lights in the ceiling, wet floor reflections, deep perspective, strong rim light on the runner, "
           "cinematic 35mm photograph, tall vertical frame, the runner fills about a third of the frame height"),
    music="kinetic and emotional, 160 bpm", length=12, seed=6)

# shared (non per-slot) ids of the two LTX templates
CTL_SHARED = {"1", "2", "3", "4", "5", "7", "9", "10", "11", "12", "13", "21", "22", "27", "28", "37", "38", "55", "56"}
CTL_DROP = {"15", "16", "23", "60", "61", "62"}
ALPHA_LOADERS = {"1", "2", "3", "4", "12", "13"}                      # same nodes as the control graph's
ALPHA_SHARED = {"5", "6", "7", "8", "10", "11", "14", "33", "35", "36"}
ALPHA_DROP = {"15", "16", "41", "42", "43"} | {"40"}                  # 40 (VAEDecodeTiled) -> MographDecodeClip


# --------------------------------------------------------------------------------------------- helpers
def link(node_id: str, slot: int = 0) -> list:
    return [node_id, slot]


def add(g: dict, nid: str, cls: str, title: str, **inputs) -> str:
    assert nid not in g, f"duplicate node id {nid}"
    g[nid] = {"inputs": inputs, "class_type": cls, "_meta": {"title": title}}
    return nid


def _remap_links(inputs: dict, mapping) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
            out[k] = [mapping(v[0]), v[1]]
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------------------------- the graph
def build_graph(draft: bool = False, still: str = "moss", design: bool = False, auto: bool = False, angles: bool = True) -> dict:
    """`draft` is the argument of the REMOVED CPU draft graph (stand-in footage: cached clips / Ken Burns moves, luma mattes): True raises. The fast first look is the quality
    `still` ("Still preview", tools/fusion/still_preview.py) of the same graph. design=True: the fusion_design_qwen graph (single-stage preview LTX, design_llm designs the type, see the module docstring of
    nodes_fusion/design.py); everything else is the fusion graph, unchanged. auto=True (with design): words / scene / music / look / shots come from
    a design_llm concept step (nodes_fusion/concept.py) instead of the IN nodes; `IN previous` = the earlier concepts the new one must differ from."""
    if draft:
        raise ValueError("the CPU draft graph (stand-in footage) was removed; use the quality `still` (Still preview) of the app graph")
    g: dict = {}
    q_default = "preview" if design else "final"
    req0 = TEST_REQUEST if design else dict(words="AFTER HOURS\nTHE CITY KEEPS DREAMING", scene="", music="", length=12, seed=1)

    # ---- app inputs
    if not auto:
        add(g, "words", "PrimitiveStringMultiline", "IN words", value=req0["words"])
        add(g, "scene", "PrimitiveString", "IN scene", value=req0["scene"])
        add(g, "music", "PrimitiveString", "IN music", value=req0["music"])
        add(g, "look", "PrimitiveString", "IN look", value="auto")
    else:
        add(g, "previous", "PrimitiveStringMultiline", "IN previous", value="[]")
    add(g, "quality", "PrimitiveString", "IN quality", value=q_default)
    add(g, "length", "PrimitiveInt", "IN length", value=AUTO_LENGTH if auto else req0["length"])
    add(g, "seed", "PrimitiveInt", "IN seed", value=req0["seed"])
    add(g, "image", "MographOptionalImage", "IN image", image="")
    add(g, "audio", "MographOptionalAudio", "IN audio", audio="")
    add(g, "format", "PrimitiveString", "IN format", value="9:16")
    add(g, "images", "PrimitiveString", "IN images", value="")                                  # the user's images 2..5, comma-separated input names (image 1 = IN image)
    add(g, "user_images", "MographUserImages", "User images (2..5)", names=link("images"), format=link("format"))
    if auto:
        add(g, "brief_in", "PrimitiveStringMultiline", "IN brief", value="")                    # READ JSON v1 (tools/fusion/concept.parse_brief): "" = the model invents everything
        add_concept_step(g)
    src = (lambda k, i: link("concept_final", i)) if auto else (lambda k, i: link(k))        # words / scene / music / look from the concept step
    req_in = dict(words=src("words", 0), scene=src("scene", 1), music=src("music", 2), length=link("length"), look=src("look", 3), seed=link("seed"),
                  quality=link("quality"), format=link("format"), image=link("image"), audio=link("audio"))
    if auto:
        req_in["shots"] = link("concept_final", 4)
        req_in["owners"] = link("concept_final", 7)          # the model's per-lockup owners (flat | ltx | handoff + ltx_at + material), JSON; "[]" = the rules decide
    add(g, "request", "MographRequest", "Request", **req_in)

    # ---- brief (design_llm_8b) - model chains only on the full graph
    brief = None
    if not design:                                   # the design graph has no brief: the design step decides everything after the footage exists
        add(g, "brief_prompt", "MographFusionBriefPrompt", "Brief prompt", request=link("request"))
        add(g, "qwen", "DesignLLM_Advanced", "design_llm creative brief", model_name="design_llm_8b",
            quantization="None (FP16)", attention_mode="auto", use_torch_compile=False, device="auto",
            preset_prompt="\U0001F5BC️ Detailed Description", custom_prompt=link("brief_prompt", 0),
            max_tokens=400, temperature=0.6, top_p=0.9, num_beams=1, repetition_penalty=1.05, frame_count=16,
            video_frame_size="auto", keep_model_loaded=False, seed=link("brief_prompt", 1), image=link("image"))
        brief = link("qwen")

    # ---- music FIRST, plan after (BUILT.md "Music"): ACE-Step 1.5 XL BASE text-to-music (no LM, 50 steps, cfg 6, length + 4 s)
    #      -> exact-length cut + the beat grid measured on the song -> tame_highs. The plan below is built from THAT grid.
    #      The user's audio takes the MusicCut branch; MographMusicRoute picks one (lazy: the other branch never runs).
    #      Replaces the rejected score skeleton + audio-to-audio chain ("circus / tinker toy"); never the turbo model.
    g.update(music_graph.t2m_chain(link("request"), link("seed"), brief, p="t2m_"))
    score = (link("t2m_tame", 0), link("t2m_cut", 1))
    add(g, "cut", "MographMusicCut", "Music cut", request=link("request"), music_bpm=100, user_audio=link("audio"))
    add(g, "route", "MographMusicRoute", "Music route", request=link("request"), user_audio=link("cut", 1),
        user_music=link("cut", 0), score_audio=score[0], score_music=score[1])
    add(g, "master", "MographMaster", "Master", audio=link("route", 0), music=link("route", 1), preset="auto",
        target_lufs=-14.0, width=1.25, mono_safe=True)
    plan = {"request": link("request"), "music": link("route", 1)}
    if brief:
        plan["brief"] = brief
    add(g, "plan", "MographPiecePlan", "Plan the piece", **plan)
    add(g, "sizes", "MographPieceSizes", "Format sizes", piece=link("plan", 0))      # format sizes -> LTX / still size inputs (9:16 | 16:9)

    # ---- still -> first frame (MOSS-Image)
    first_in = {"piece": link("plan", 0), "request": link("request")}
    if still == "moss":
        add(g, "still_unet", "UNETLoader", "MOSS-Image SFT int8", unet_name=MOSS_UNET, weight_dtype="default")
        add(g, "still_clip", "CLIPLoader", "Qwen3.5 9B text encoder", clip_name=MOSS_CLIP, type="moss_image", device="default")
        add(g, "still_vae", "VAELoader", "Still VAE", vae_name=MOSS_VAE)
        add(g, "still_txt", "CLIPTextEncode", "Still prompt", text=link("plan", 2), clip=link("still_clip"))
        add(g, "still_neg", "CLIPTextEncode", "Still negative", text=link("plan", 3), clip=link("still_clip"))
        add(g, "still_lat", "EmptyFlux2LatentImage", "Still size", width=link("sizes", 6), height=link("sizes", 7),
            batch_size=1)
        add(g, "still_ks", "KSampler", "Still sampler", model=link("still_unet"), positive=link("still_txt"),
            negative=link("still_neg"), latent_image=link("still_lat"), seed=link("seed"), steps=25, cfg=2.5,
            sampler_name="er_sde", scheduler="simple", denoise=1.0)
        add(g, "still_dec", "VAEDecode", "Still decode", samples=link("still_ks"), vae=link("still_vae"))
        first_in["generated"] = link("still_dec")
    add(g, "first", "MographFirstFrame", "First frame", **first_in)
    add(g, "out_still", "PreviewImage", "OUT still", images=link("first"))
    add_place_stills(g, still, angles)               # one still per PLACE: stills 2..6 are lazy chains, evaluated only when a shot of the piece starts from them

    # ---- type passes
    add(g, "type", "MographTypePasses", "Type passes", piece=link("plan", 0), music=link("route", 1), procs=10)
    add(g, "out_type", "PreviewImage", "OUT type", images=link("type_final" if design else "type", 1))

    # ---- footage
    # `first` of the gather = the stills, one row per place (first_set): the footage of the quality `still` (Still preview) holds each shot's own still; the LTX clips take over otherwise
    gather = {"piece": link("plan", 0), "request": link("request"), "first": link("first_set" if "first_set" in g else "first")}
    add_ltx_shared(g, single=design)
    for k in range(1, SLOTS + 1):
        add_control_slot(g, k, single=design)
        gather[f"clip_{k}"] = link(f"c{k}.dec", 0)
    add(g, "gather", "MographFootageGather", "Footage gather", **gather)
    # footage motion -> the letters (flow_wake) in EVERY generation: RAFT on CUDA, DIS on the CPU (a still preview's held pictures get zero flow)
    add(g, "flow", "MographFootageFlow", "Footage flow", piece=link("plan", 0), clips=link("gather", 0), method="auto",
        model="raft_large")

    compose = {"piece": link("plan", 0), "type": link("type", 0), "clips": link("flow", 0), "backend": "auto",
               "music": link("route", 1)}
    if design:                                       # the designed piece and ITS type passes (the first passes were the LTX control video)
        compose["piece"] = link("design_piece")
        compose["type"] = link("type_final", 0)
    add(g, "vda", "LoadVideoDepthAnythingModel", "Video Depth Anything", model=VDA_MODEL)
    for k in range(1, SLOTS + 1):
        add_alpha_slot(g, k)
        add_depth_slot(g, k)
        compose[f"matte_{k}"] = link(f"a{k}.dec", 1)
        compose[f"depth_{k}"] = link(f"d{k}.out")
    if design:
        add_design_step(g)
    add(g, "compose", "MographCompose", "Compose", **compose)
    add(g, "assemble", "MographAssemble", "Assemble", composed=link("compose", 0), piece=link("design_piece" if design else "plan", 0),
        audio=link("master"), crf=20 if design else 16, encoder="auto")
    add(g, "out_final", "SaveVideo", "OUT final", video=link("assemble"), filename_prefix="video/motngrfx",
        format="auto", codec="auto")
    if True:      # FINAL = the 540x960 render (OUT final) AND the Real-ESRGAN 1080x1920 upscale (tools/app/final_upscale.py); the save node's lazy `video` keeps it from running in a Preview
        from tools.app import final_upscale as FU
        _src = FU.source_size(still_preview.APP_FINAL_TIER, "9:16")
        g.update(FU.fragment(["assemble", 0], _src, FU.final_size("9:16"), "up"))
        add(g, "out_final_1080", "MographSaveFinalUpscaled", "OUT final 1080", piece=link("design_piece" if design else "plan", 0), video=link("up_vid"),
            filename_prefix="video/motngrfx_1080")
    if design:                                       # keeps the still, the shot clips and the Alpha-Gen matte PNGs next to the final (16_app/RUN_3060.md)
        arch = {"filename_prefix": "archive/motngrfx", "piece": link("design_piece", 0), "plan_piece": link("plan", 0),
                "clips": link("flow", 0), "first": link("first"), "video": link("assemble")}
        if "first_set" in g:                         # ingredients (16_app/INGREDIENTS.md): one still per place, the type passes compose read, depth previews
            arch["places"] = link("first_set")
        if "type_final" in g:
            arch["type"] = link("type_final", 0)
        arch["composed"] = link("compose", 0)         # every composition's parts (16_app/INGREDIENTS.md): the composited shots ...
        if "design_pkg" in g:
            arch["sheet"] = link("design_pkg", 3)     # ... and the sheet image the vision model saw (full_archive on by default)
        for k in range(1, SLOTS + 1):
            arch[f"matte_{k}"] = link(f"a{k}.dec", 1)
            arch[f"depth_{k}"] = link(f"d{k}.out")
        add(g, "archive", "MographArchive", "Archive (still, shots, mattes)", **arch)
    return g


def add_place_stills(g: dict, still: str = "moss", angles: bool = True) -> None:
    """One still per place (`piece.stills`, A..F; tools/fusion/places.py): chains for the stills 2..MAX_STILLS next to the first one, the same loaders / sampler settings, own
    prompt + seed from the plan (MographStillPrompt). MographFirstFrames gathers them into ONE image batch (row i = still i; stills 2.. are lazy inputs, so a chain only
    runs when a shot starts from that still; a one-place piece runs none and gets a batch of one row = the first frame) and the shot slots read that batch. The loaders are
    shared (no extra model load: the encoder / DiT stay loaded between the stills); the negative prompt is the first still's (every still has the same one)."""
    gather = {"piece": link("plan", 0), "request": link("request"), "first": link("first")}
    if "user_images" in g:                           # the user's images 2..5 replace the generated stills of shots 2..5 (places.user_image_rows); their chains then never run
        gather["user_images"] = link("user_images")
    angles = angles and still == "moss"              # angles=False (the shipped default, CLI without --angles): NO LoRA loader and no angle chain, the chain needs moss_image_edit_chinoll.safetensors whose licence is undeclared
    if angles:                                       # same-subject new-angle stills (tools/fusion/angle_still.py): lazy, off unless the tier switches `angles` on (llm_arms.TIERS)
        add(g, "ang_lora", "LoraLoaderModelOnly", "MOSS edit LoRA (angle stills)", model=link("still_unet"), lora_name=angle_still.LORA, strength_model=1.0)
        gather["angles"] = False
    for k in range(2, places.MAX_STILLS + 1):
        tag = f" [place {k}]"
        add(g, f"still{k}_prompt", "MographStillPrompt", "Still text" + tag, index=k, piece=link("plan", 0))
        if angles:
            add_angle_chain(g, k)
            gather[f"angle_{k}"] = link(f"ang{k}_dec")
        if still == "moss":
            add(g, f"still{k}_txt", "CLIPTextEncode", "Still prompt" + tag, text=link(f"still{k}_prompt", 0), clip=link("still_clip"))
            add(g, f"still{k}_lat", "EmptyFlux2LatentImage", "Still size" + tag, width=link("sizes", 6), height=link("sizes", 7), batch_size=1)
            add(g, f"still{k}_ks", "KSampler", "Still sampler" + tag, model=link("still_unet"), positive=link(f"still{k}_txt"), negative=link("still_neg"),
                latent_image=link(f"still{k}_lat"), seed=link(f"still{k}_prompt", 1), steps=25, cfg=2.5, sampler_name="er_sde", scheduler="simple", denoise=1.0)
            add(g, f"still{k}_dec", "VAEDecode", "Still decode" + tag, samples=link(f"still{k}_ks"), vae=link("still_vae"))
        gather[f"still_{k}"] = link(f"still{k}_dec")
    add(g, "first_set", "MographFirstFrames", "First frames (one per place)", **gather)
    add(g, "out_stills", "PreviewImage", "OUT stills", images=link("first_set"))


def add_angle_chain(g: dict, k: int) -> None:
    """The ANGLE-STILL chain of place k (2..6): a MOSS-Image EDIT of the source place's ORIGINAL still (`angle_of`; never an edit of an edit) from a new camera, next to the
    place's text-to-image chain; `MographFirstFrames.angle_k` (lazy) asks for it only when the graph's `angles` switch is on and the place has `angle_of`
    (tools/fusion/places.angle_source). Shape = tools/fusion/angle_still.angle_graph (probe 23_moss-angles/v003): LoRA on the still DiT, `TextEncodeMOSSImageEdit` x2
    (edit_format Moss-Image-Edit, `images.image_0` = the reference, vae connected, the SAME node type for the negative), KSampler 30 steps cfg 4 er_sde / simple.
    One seed per place (sweep 1); the sweep + colour pick of angle_still is not in the graph. Loaders and the negative prompt are the first still's (no extra model load)."""
    tag = f" [angle place {k}]"
    add(g, f"ang{k}_prompt", "MographStillPrompt", "Angle text" + tag, index=k, piece=link("plan", 0), angle=True)
    add(g, f"ang{k}_ref", "MographAngleRef", "Angle reference" + tag, index=k, piece=link("plan", 0), first=link("first"), width=link("sizes", 6), height=link("sizes", 7),
        **{f"still_{j}": link(f"still{j}_dec") for j in range(2, places.MAX_STILLS + 1) if j != k})
    for role, text in (("pos", link(f"ang{k}_prompt", 0)), ("neg", link("plan", 3))):
        g_in = {"clip": link("still_clip"), "prompt": text, "edit_format": angle_still.EDIT_FORMAT, "image_strength": 1.0, "vae": link("still_vae"),
                "images.image_0": link(f"ang{k}_ref")}
        add(g, f"ang{k}_{role}", "TextEncodeMOSSImageEdit", ("Angle prompt" if role == "pos" else "Angle negative") + tag, **g_in)
    add(g, f"ang{k}_lat", "EmptyFlux2LatentImage", "Angle size" + tag, width=link("sizes", 6), height=link("sizes", 7), batch_size=1)
    add(g, f"ang{k}_ks", "KSampler", "Angle sampler" + tag, model=link("ang_lora"), positive=link(f"ang{k}_pos"), negative=link(f"ang{k}_neg"), latent_image=link(f"ang{k}_lat"),
        seed=link(f"ang{k}_prompt", 1), steps=angle_still.STEPS, cfg=angle_still.CFG, sampler_name=angle_still.SAMPLER, scheduler=angle_still.SCHEDULER, denoise=1.0)
    add(g, f"ang{k}_dec", "VAEDecode", "Angle decode" + tag, samples=link(f"ang{k}_ks"), vae=link("still_vae"))


def add_ltx_shared(g: dict, single: bool = False) -> None:
    """Loaders + constants of the control template and the shared alpha nodes, once. single=True: ONE stage at the preview size
    (no stage 2 / x2 upscale: the preview tier, `plan_rules.sizes_for("preview")`), nodes only the dropped stage 2 used are not added."""
    ctl = ltx_ic.build("control", ref_video="r.mp4", first_frame="f.png", prompt="", size=SIZE_PREVIEW if single else SIZE_CONTROL,
                       frames=SLOT_FRAMES, two_stage=not single)
    alpha = ltx_ic.build("alpha", ref_video="r.mp4", size=SIZE_ALPHA, frames=145, png=False)
    TPL["ctl"], TPL["alpha"] = ctl, alpha
    for oid in ctl:                                              # template order (sets iterate in hash order)
        if oid in CTL_SHARED:
            g[oid] = copy.deepcopy(ctl[oid])
    for oid, o in (("10", 0), ("11", 1), ("21", 2), ("22", 3), ("27", 4), ("28", 5)):   # sizes follow the format (9:16 | 16:9)
        g[oid]["inputs"]["value"] = link("sizes", o - 4 if single and o >= 4 else o)     # single stage: the first frame is fitted to the one stage
    # alpha: own LoRA + empty prompt/conditioning + constants; its loaders are the control graph's nodes
    for oid in alpha:
        if oid not in ALPHA_SHARED:
            continue
        n = copy.deepcopy(alpha[oid])
        n["inputs"] = _remap_links(n["inputs"], _alpha_map(None))
        n["_meta"] = {"title": f"{n['_meta']['title']} (alpha)"}
        g[f"a.{oid}"] = n
    g["a.10"]["inputs"]["value"] = link("sizes", 0)              # alpha runs at the stage-1 size
    g["a.11"]["inputs"]["value"] = link("sizes", 1)


def _alpha_map(k):
    def m(oid: str) -> str:
        if oid in ALPHA_LOADERS:
            return oid
        if oid in ALPHA_SHARED:
            return f"a.{oid}"
        return f"a{k}.{oid}"
    return m


def add_control_slot(g: dict, k: int, single: bool = False) -> None:
    ctl = TPL["ctl"]
    pre = f"c{k}."
    sid = f"s{k}"
    slot = f"slot{k}"
    slot_in = {"index": k, "piece": link("plan", 0), "type": link("type", 0), "first": link("first_set" if "first_set" in g else "first")}   # first_set: one row per place
    if k > 1:
        slot_in["prev"] = link(f"c{k - 1}.dec", 0)
    add(g, slot, "MographShotSlot", f"Shot slot [{sid}]", **slot_in)

    def m(oid: str) -> str:
        return oid if oid in CTL_SHARED else pre + oid
    for oid, n in ctl.items():
        if oid in CTL_DROP or oid in CTL_SHARED:
            continue
        g[m(oid)] = {"inputs": _remap_links(n["inputs"], m), "class_type": n["class_type"],
                     "_meta": {"title": f"{n['_meta']['title']} [{sid}]"}}
    c = lambda oid: g[pre + oid]["inputs"]                                  # noqa: E731
    c("17")["input"] = link(slot, 3)                                        # control slice
    c("24")["input"] = link(slot, 4)                                        # first frame
    c("26")["value"] = link(slot, 5)                                        # bypass first frame
    c("14")["value"] = link(slot, 2)                                        # frames
    c("6")["text"] = link(slot, 0)                                          # prompt
    c("35")["noise_seed"] = link(slot, 1)
    if not single:                                                          # stage 2 sampler noise (absent in the single-stage template)
        c("53")["noise_seed"] = link(slot, 1)
    c("32")["strength"] = link(slot, 6)                                     # control guide strength
    add(g, f"{pre}dec", "MographDecodeClip", f"Decode clip [{sid}]", vae=link("3"),
        samples=link(pre + "41", 2) if single else link(pre + "58"),
        piece=link("plan", 0), index=k, gray=False)


def add_alpha_slot(g: dict, k: int) -> None:
    alpha = TPL["alpha"]
    sid = f"s{k}"
    m = _alpha_map(k)
    for oid, n in alpha.items():
        if oid in ALPHA_DROP or oid in ALPHA_SHARED or oid in ALPHA_LOADERS:
            continue
        g[m(oid)] = {"inputs": _remap_links(n["inputs"], m), "class_type": n["class_type"],
                     "_meta": {"title": f"{n['_meta']['title']} [alpha {sid}]"}}
    g[f"a{k}.17"]["inputs"]["input"] = link("gather", k)                    # alpha_ref_k (replaces GetVideoComponents)
    add(g, f"a{k}.dec", "MographDecodeClip", f"Decode matte [{sid}]", vae=link("3"), samples=link(f"a{k}.39", 2),
        piece=link("plan", 0), index=k, gray=True)


def add_depth_slot(g: dict, k: int) -> None:
    sid = f"s{k}"
    add(g, f"d{k}.proc", "VideoDepthAnythingProcess", f"Depth process [{sid}]", vda_model=link("vda"),
        images=link("gather", SLOTS + k), input_size=518, max_res=1280, precision="fp16")
    add(g, f"d{k}.out", "VideoDepthAnythingOutput", f"Depth gray [{sid}]", depths=link(f"d{k}.proc"), colormap="gray")


def qwen_node(g: dict, nid: str, title: str, prompt_link: list, image_link: list | None, max_tokens: int = QWEN_DESIGN_TOKENS,
              temperature: float = QWEN_DESIGN_TEMP, quant: str = QWEN_QUANT) -> None:
    """DesignLLM_Advanced for the design step: design_llm_8b, 4-bit (8-bit for the redesign A/B arm), unloaded after the call (keep_model_loaded False)."""
    ins = dict(model_name=QWEN_MODEL, quantization=quant, attention_mode="auto", use_torch_compile=False, device="auto",
               preset_prompt="🖼️ Detailed Description", custom_prompt=prompt_link, max_tokens=max_tokens,
               temperature=temperature, top_p=0.9, num_beams=1, repetition_penalty=1.05, frame_count=16, video_frame_size="auto",
               keep_model_loaded=False, seed=link("seed"))
    if image_link:
        ins["image"] = image_link
    add(g, nid, "DesignLLM_Advanced", title, **ins)


def add_concept_step(g: dict) -> None:
    """design_llm (text only) invents the request: prompt -> model -> parse (validate + uniqueness vs `IN previous`) -> [model repair, lazy: only on a
    clash / invalid reply] -> final (repair reply, else the seeded rules variation) -> MographRequest. Length is the graph's input, not the model's."""
    bf = {"brief": link("brief_in")} if "brief_in" in g else {}      # the user's READ JSON (words verbatim, concept, look, media notes); absent = fully automatic
    add(g, "concept_prompt", "MographConceptPrompt", "Concept prompt", seed=link("seed"), length=link("length"), previous=link("previous"), **bf)
    qwen_node(g, "concept_qwen", "design_llm concept", link("concept_prompt", 0), None, QWEN_CONCEPT_TOKENS, QWEN_CONCEPT_TEMP)
    add(g, "concept_parse", "MographConceptParse", "Concept parse", reply=link("concept_qwen"), previous=link("previous"), seed=link("seed"),
        length=link("length"), prompt=link("concept_prompt", 0), **bf)
    qwen_node(g, "concept_qwen2", "design_llm concept repair", link("concept_parse", 1), None, QWEN_CONCEPT_TOKENS, QWEN_CONCEPT_TEMP)
    add(g, "concept_final", "MographConceptFinal", "Concept final", reply=link("concept_qwen"), ok=link("concept_parse", 0),
        repair_reply=link("concept_qwen2"), previous=link("previous"), seed=link("seed"), length=link("length"), **bf)
    add(g, "out_concept", "PreviewAny", "OUT concept", source=link("concept_final", 5))
    add(g, "out_concept_reply", "PreviewAny", "OUT concept reply", source=link("concept_qwen"))
    add(g, "out_concept_check", "PreviewAny", "OUT concept check", source=link("concept_final", 6))


def add_design_step(g: dict) -> None:
    """footage (+ mattes, depth, flow) -> package -> design_llm design -> parse -> [Qwen repair, lazy: only when parse says ok = false] ->
    repair/fill from the rules -> piece -> the type passes of the DESIGNED piece. The first `type` node (rules' plan) only made the LTX control
    video. Replies and checks are PreviewAny outputs (ComfyUI history). The repair model is NOT an input of an output node: it is only
    upstream of the lazy `repair_reply`, so it never runs when the first reply passed."""
    pkg = dict(request=link("request"), music=link("route", 1), piece=link("plan", 0), clips=link("flow", 0), guide="DESIGN_GUIDE_open_model",
               seed=link("seed"))
    for k in range(1, SLOTS + 1):
        pkg[f"matte_{k}"] = link(f"a{k}.dec", 1)
        pkg[f"depth_{k}"] = link(f"d{k}.out")
    add(g, "design_pkg", "MographDesignPackage", "Design package", **pkg)
    qwen_node(g, "design_qwen", "design_llm design", link("design_pkg", 0), link("design_pkg", 3))
    add(g, "design_parse", "MographDesignParse", "Design parse", reply=link("design_qwen"), locked=link("design_pkg", 2),
        piece=link("plan", 0), music=link("route", 1), prompt=link("design_pkg", 0))
    qwen_node(g, "design_qwen2", "design_llm design repair", link("design_parse", 2), None)
    add(g, "design_fix", "MographDesignRepair", "Design repair", piece_json=link("design_parse", 0), ok=link("design_parse", 1),
        repair_reply=link("design_qwen2"), piece=link("plan", 0), music=link("route", 1), locked=link("design_pkg", 2))
    add(g, "design_piece", "MographPieceFromJSON", "Piece from design", piece_json=link("design_fix", 0), music=link("route", 1))
    add(g, "type_final", "MographTypePasses", "Type passes (design)", piece=link("design_piece"), music=link("route", 1), procs=10)
    add(g, "out_design_reply", "PreviewAny", "OUT design reply", source=link("design_qwen"))
    add(g, "out_design_check", "PreviewAny", "OUT design check", source=link("design_parse", 3))
    add(g, "out_design_final", "PreviewAny", "OUT design final check", source=link("design_fix", 1))


# --------------------------------------------------------------------------------------------- fusion_redesign*: the design step again, over an archive
# The model A/B graphs (16_app/RUN_3060.md section 11): ONE finished generation's archive (MographArchive folder: still, shot clips, matte PNGs, piece.json)
# + its final mp4 (the song) -> design package -> open model -> parse (field-level salvage) -> lazy repair -> piece -> type passes -> compose -> assemble ->
# archive (light). No music model, no still model, no LTX, no Alpha-Gen. One graph file per model arm; everything else is identical, so the arms compare.
REDESIGN_ARMS = {   # arm -> (graph file name, label written into plan.design.model, sidecar title)
    "q4": ("fusion_redesign", "design_llm_8b-4bit", "Redesign - design_llm_8b 4-bit (the batch model)"),
    "q8": ("fusion_redesign_q8", "design_llm_8b-8bit", "Redesign - design_llm_8b 8-bit"),
    "gemma": ("fusion_redesign_gemma", "gemma-4-12b-int8", "Redesign - Gemma 4 12B (native ComfyUI TextGenerate)"),
    "replay": ("fusion_redesign_replay", "replayed-reply", "Redesign - a saved model reply (no model call): the parse / salvage / render half only"),
}
QWEN_QUANT_8 = "8-bit (Balanced)"                  # the design-LLM node pack option
GEMMA_CLIP = "gemma4_12b_int8_convrot.safetensors"  # Comfy-Org/gemma-4 text_encoders (11.2 GiB); the repo has no 12B fp8 file (fp8 exists for E4B only)
GEMMA_DESIGN_TOKENS = 4096                         # TextGenerate's default is 256: the design JSON of 3-6 words is ~2.5k tokens
REPAIR_TOKENS = 1024                               # the repair reply is a few field paths with values
REDESIGN_IN_TITLES = ["IN archive", "IN source video", "IN concept", "IN seed"]
REDESIGN_OUT_TITLES = ["OUT final"]


def gemma_node(g: dict, nid: str, title: str, prompt_link: list, image_link: list | None, max_length: int) -> None:
    """TextGenerate (ComfyUI core) with the Gemma 4 encoder: thinking off, sampling on (the same temperature as the Qwen arms), seed = IN seed."""
    ins = {"clip": link("design_clip"), "prompt": prompt_link, "max_length": max_length, "sampling_mode": "on",
           "sampling_mode.temperature": QWEN_DESIGN_TEMP, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
           "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": link("seed"), "thinking": False, "use_default_template": True, "mtp": "off"}
    if image_link:
        ins["image"] = image_link
    add(g, nid, "TextGenerate", title, **ins)


def build_redesign(arm: str = "q4") -> dict:
    name, label, _ = REDESIGN_ARMS[arm]
    g: dict = {}
    add(g, "archive_in", "PrimitiveString", "IN archive", value="redesign/src/r03_out_shots")
    add(g, "source_in", "PrimitiveString", "IN source video", value="redesign/src/r03_final.mp4")
    add(g, "concept_in", "PrimitiveStringMultiline", "IN concept", value="")
    add(g, "seed", "PrimitiveInt", "IN seed", value=6)
    add(g, "load", "MographArchiveLoad", "Archive load", archive=link("archive_in"), source_video=link("source_in"), concept=link("concept_in"))
    # the rules' plan of the same request + beat grid = the fallback of every invalid field and the locked fields (it reproduces the archive's shots exactly)
    add(g, "plan", "MographPiecePlan", "Re-plan (rules fallback)", request=link("load", 0), music=link("load", 1))
    add(g, "flow", "MographFootageFlow", "Footage flow", piece=link("plan", 0), clips=link("load", 3), method="auto", model="raft_large")
    pkg = dict(request=link("load", 0), music=link("load", 1), piece=link("plan", 0), clips=link("flow", 0), guide="DESIGN_GUIDE_open_model", seed=link("seed"))
    for k in range(1, SLOTS + 1):
        pkg[f"matte_{k}"] = link("load", 5 + k)
    add(g, "design_pkg", "MographDesignPackage", "Design package", **pkg)
    if arm == "replay":                              # the model's reply is an input: everything after the model call runs for real
        add(g, "design_qwen", "PrimitiveStringMultiline", "IN reply", value="")
    elif arm == "gemma":
        add(g, "design_clip", "CLIPLoader", "Gemma 4 12B text encoder", clip_name=GEMMA_CLIP, type="stable_diffusion", device="default")
        gemma_node(g, "design_qwen", "Gemma 4 design", link("design_pkg", 0), link("design_pkg", 3), GEMMA_DESIGN_TOKENS)
    else:
        qwen_node(g, "design_qwen", "design_llm design", link("design_pkg", 0), link("design_pkg", 3), quant=QWEN_QUANT_8 if arm == "q8" else QWEN_QUANT)
    add(g, "design_parse", "MographDesignParse", "Design parse", reply=link("design_qwen"), locked=link("design_pkg", 2), piece=link("plan", 0),
        music=link("load", 1), prompt=link("design_pkg", 0), model=label)
    if arm == "replay":
        add(g, "design_qwen2", "PrimitiveStringMultiline", "IN repair reply", value="")
    elif arm == "gemma":
        gemma_node(g, "design_qwen2", "Gemma 4 design repair", link("design_parse", 2), None, REPAIR_TOKENS)
    else:
        qwen_node(g, "design_qwen2", "design_llm design repair", link("design_parse", 2), None, REPAIR_TOKENS, quant=QWEN_QUANT_8 if arm == "q8" else QWEN_QUANT)
    add(g, "design_fix", "MographDesignRepair", "Design repair", piece_json=link("design_parse", 0), ok=link("design_parse", 1), repair_reply=link("design_qwen2"),
        piece=link("plan", 0), music=link("load", 1), locked=link("design_pkg", 2), model=label)
    add(g, "design_piece", "MographPieceFromJSON", "Piece from design", piece_json=link("design_fix", 0), music=link("load", 1))
    add(g, "type_final", "MographTypePasses", "Type passes (design)", piece=link("design_piece"), music=link("load", 1), procs=10)
    comp = {"piece": link("design_piece"), "type": link("type_final", 0), "clips": link("flow", 0), "backend": "auto", "music": link("load", 1)}
    for k in range(1, SLOTS + 1):
        comp[f"matte_{k}"] = link("load", 5 + k)
    add(g, "compose", "MographCompose", "Compose", **comp)
    add(g, "assemble", "MographAssemble", "Assemble", composed=link("compose", 0), piece=link("design_piece"), audio=link("load", 5), crf=20, encoder="auto")
    add(g, "out_final", "SaveVideo", "OUT final", video=link("assemble"), filename_prefix="redesign/out", format="auto", codec="auto")
    arch = {"filename_prefix": "redesign/out", "piece": link("design_piece", 0), "plan_piece": link("plan", 0), "clips": link("flow", 0), "first": link("load", 4),
            "video": link("assemble"), "light": True, "composed": link("compose", 0), "sheet": link("design_pkg", 3)}      # light: piece.json + this run's composed shots + the design sheet
    add(g, "archive", "MographArchive", "Archive (piece.json)", **arch)
    add(g, "out_design_reply", "PreviewAny", "OUT design reply", source=link("design_qwen"))
    add(g, "out_design_check", "PreviewAny", "OUT design check", source=link("design_parse", 3))
    add(g, "out_design_final", "PreviewAny", "OUT design final check", source=link("design_fix", 1))
    return g


def build_redesign_sidecar(arm: str = "q4") -> dict:
    name, label, title = REDESIGN_ARMS[arm]
    side = {
        "title": title, "order": 30, "verb": "Redesign it", "running_verb": "Redesigning",
        "cost_line": "Runs on the GPU box (no music, still, video or matte generation)",
        "note": ("The design step again over a finished generation's archive: the open model sees the footage contact sheets and the music, designs the type and "
                 "the compose look; the code mends invalid fields one by one (design_salvage.py); only the type and the composite are rendered again. "
                 f"Arm: {label}."),
        "fields": [
            {"id": "archive", "label": "Archive folder", "kind": "text", "node_title": "IN archive", "input_name": "value", "optional": False, "default": "",
             "placeholder": "redesign/src/r03_out_shots", "hint": "A MographArchive folder under the ComfyUI output dir (still, sK_clip.mp4, sK_matte/, piece.json)."},
            {"id": "source", "label": "Source video", "kind": "text", "node_title": "IN source video", "input_name": "value", "optional": True, "default": "",
             "hint": "The generation's final mp4: its audio is the song."},
            {"id": "concept", "label": "Request (JSON)", "kind": "textarea", "node_title": "IN concept", "input_name": "value", "optional": True, "default": "", "rows": 4,
             "hint": "words / scene / music / look / shots / length / seed of the generation; empty = rebuilt from the piece."},
            {"id": "seed", "label": "Design seed", "kind": "seed", "node_title": "IN seed", "input_name": "value", "optional": True, "placeholder": "Seed"},
        ],
        "missing": {"archive": "Name an archive folder"},
        "outputs": [{"node_title": "OUT final", "kind": "video", "label": "Redesigned video", "final": True}],
        "required_titles": REDESIGN_IN_TITLES + REDESIGN_OUT_TITLES,
        "stage_labels": {"Archive load": "Read the archive", "Re-plan (rules fallback)": "Re-plan the rules' fallback", "Footage flow": "Read the footage motion",
                         "Design package": "Prepare the design brief", "design_llm design": "The model designs the type", "Gemma 4 design": "The model designs the type",
                         "Design parse": "Check the design field by field", "design_llm design repair": "The model corrects the invalid fields",
                         "Gemma 4 design repair": "The model corrects the invalid fields", "Design repair": "Final design check",
                         "Piece from design": "Load the designed piece", "Type passes (design)": "Render the designed 3D type", "Compose": "Compose type into footage",
                         "Assemble": "Encode the video", "OUT final": "Save video"},
        "server_flags": {"comfyui": ["--mmap-torch-files", "--cache-ram", "16", "--use-sage-attention"],
                         "why": "12 GB card: the design model loads alone and is unloaded after each call"},
    }
    if arm == "replay":
        side["fields"].append({"id": "reply", "label": "Model reply (JSON)", "kind": "textarea", "node_title": "IN reply", "input_name": "value", "optional": False,
                               "default": "", "rows": 6, "hint": "A saved design reply; the parse, salvage, type and compose run for real."})
        side["required_titles"] = REDESIGN_IN_TITLES + ["IN reply"] + REDESIGN_OUT_TITLES
    return side


# --------------------------------------------------------------------------------------------- sidecar
def _load_routes():
    spec = importlib.util.spec_from_file_location("mograph_routes", os.path.join(ROOT, "nodes_fusion", "routes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sidecar(draft: bool = False, still: str = "moss", design: bool = False, auto: bool = False) -> dict:
    if draft:
        raise ValueError("the CPU draft graph (stand-in footage) was removed; use the quality `still` (Still preview) of the app graph")
    looks = _load_routes().looks_payload()["looks"]
    side = {
        "title": "Fusion (GPU box)",
        "order": 10,                                 # app_routes sorts by it; the page opens the first template
        "verb": "Make it",
        "running_verb": "Making it",
        "cost_line": "Runs on the GPU box",
        "note": ("Words become 3D type woven into generated footage. Image sets the first frame; audio sets the cut. "
                 "The music is made first (ACE-Step XL text-to-music) and the plan follows its beats."),
        "fields": [
            {"id": "words", "label": "Words", "kind": "textarea", "node_title": "IN words", "input_name": "value",
             "optional": False, "default": "", "rows": 4,
             "placeholder": "One line per beat, e.g.\nAFTER\nHOURS\nTHE CITY KEEPS DREAMING",
             "hint": "One line per beat of the story."},
            {"id": "scene", "label": "Scene prompt", "kind": "textarea", "node_title": "IN scene", "input_name": "value",
             "optional": True, "default": "", "rows": 2, "placeholder": "Empty = written from your words"},
            {k: v for k, v in MEDIA_FIELD.items() if auto or k != "brief_node"},
            {"id": "music", "label": "Music prompt", "kind": "text", "node_title": "IN music", "input_name": "value",
             "optional": True, "default": "", "placeholder": "Optional: describe the music"},
            {"id": "length", "label": "Length", "kind": "select", "node_title": "IN length", "input_name": "value",
             "default": 12, "options": [{"value": v, "label": f"{v} s"} for v in (8, 12, 16, 24)]},
            {"id": "look", "label": "Look", "kind": "select", "node_title": "IN look", "input_name": "value",
             "default": "auto", "options": [{"value": o["value"], "label": o["label"]} for o in looks]},
            {"id": "seed", "label": "Seed", "kind": "seed", "node_title": "IN seed", "input_name": "value",
             "optional": True, "placeholder": "Random seed"},
            {"id": "format", "label": "Format", "kind": "select", "node_title": "IN format", "input_name": "value",
             "default": "9:16", "options": [{"value": "9:16", "label": "Vertical 9:16"},
                                           {"value": "16:9", "label": "Horizontal 16:9"}]},
            {"id": "quality", "label": "Quality", "kind": "select", "node_title": "IN quality", "input_name": "value",
             "default": "final",
             "hint": "Preview (still): each shot's own still held instead of LTX video (type, 3D cuts, music and compose are real) - the quick look at the design; Final: the LTX render "
                     "(still_preview.APP_FINAL_TIER) plus the Real-ESRGAN 1080 upscale; hidden value draft540 (what the page sends since v20) = the same LTX render WITHOUT the upscale; "
                     "tools/app/upgrade_still.py renders a Preview as Final with the same stills and design.",
             "options": [{"value": v, "label": lab} for v, lab in still_preview.APP_QUALITIES]},
        ],
        "missing": {"words": "Add words to start"},
        "outputs": [
            {"node_title": "OUT still", "kind": "image", "label": "First frame"},
            {"node_title": "OUT type", "kind": "image", "label": "Type passes"},
            {"node_title": "OUT final", "kind": "video", "label": "Final video", "final": True},
            {"node_title": "OUT final 1080", "kind": "video", "label": "Final video 1080x1920 (upscaled, Final only)", "optional": True},
        ],
        "stage_labels": {
            "IN words": "Read the words", "IN image": "Load image", "IN audio": "Load audio", "User images (2..5)": "Load your other images",
            "design_llm creative brief": "Write the brief", "Music T2M encode": "Describe the music",
            "Footage flow": "Read the footage motion", "ACE sampler (text-to-music)": "Make the music (ACE-Step XL)",
            "Music T2M cut": "Cut to length, find the beats", "Tame highs": "Soften the highs", "Music cut": "Cut your audio",
            "Music route": "Pick the music", "Master": "Master the audio",
            "Plan the piece": "Plan the piece", "Still sampler": "Generate the still", "First frame": "First frame",
            **{f"Still sampler [place {k}]": f"Generate the still of place {k}" for k in range(2, places.MAX_STILLS + 1)},
            **{f"Angle sampler [angle place {k}]": f"Edit the angle still of place {k}" for k in range(2, places.MAX_STILLS + 1)},
            "First frames (one per place)": "Gather the places' first frames",
            "Type passes": "Render the 3D type", "Footage gather": "Gather footage", "Compose": "Compose type into footage",
            "Assemble": "Encode the video", "OUT final": "Save video",
            "UP Real-ESRGAN x2": "Upscale to 1080x1920 (Final)", "OUT final 1080": "Save the 1080x1920 video",
        },
    }
    if design:
        side["title"] = "Fusion - design_llm designs the type (RTX 3060 preview)"
        side["order"] = 20
        side["note"] = ("One prompt in, one preview mp4 out, no review pass, no Claude: the music is made first, the footage is generated "
                        "(single-stage LTX, 384x704), then design_llm_8b (4-bit) sees the footage and the music and designs the type and the "
                        "compose look; the code validates, clamps and fits the timing and falls back to the rules for anything the model gets wrong.")
        for f in side["fields"]:
            if f["id"] == "quality":
                f["default"] = still_preview.QUALITY
            if f["id"] == "length":
                f["default"] = TEST_REQUEST["length"]
        side["stage_labels"].update({
            "Design package": "Prepare the design brief", "design_llm design": "Qwen designs the type", "Design parse": "Check the design",
            "design_llm design repair": "Qwen repairs the design", "Design repair": "Final design check",
            "Piece from design": "Load the designed piece", "Type passes (design)": "Render the designed 3D type"})
        side["server_flags"] = {"comfyui": ["--mmap-torch-files", "--cache-ram", "16", "--use-sage-attention"],
                                "why": "12 GB card: models load in turn; design_llm_8b runs 4-bit and is unloaded after each call"}
        if auto:
            side["title"] = "Fusion - design_llm invents the piece and designs it (RTX 3060 preview)"
            side["order"] = 21
            side["note"] = ("No request needed: design_llm_8b invents the words, scene, music style, look and shots mode (unique against the earlier pieces in "
                            "`previous`), then the whole fusion_design_qwen pipeline makes the piece. Length and seed are the only inputs.")
            drop = {"words", "scene", "music", "look"}
            side["fields"] = [f for f in side["fields"] if f["id"] not in drop]
            side["fields"].insert(0, {"id": "previous", "label": "Earlier concepts (JSON list)", "kind": "textarea", "node_title": "IN previous",
                                      "input_name": "value", "optional": True, "default": "[]", "rows": 3, "hidden": True,
                                      "hint": "Concepts of the earlier runs; the new one must differ from every one (the batch driver fills this)."})
            side["missing"] = {}
            side["outputs"] = side["outputs"] + [{"node_title": "OUT concept", "kind": "text", "label": "Concept"}]
            side["required_titles"] = [t for t in IN_TITLES if t.split(" ", 1)[1] not in drop] + ["IN previous", "IN brief"] + OUT_TITLES
            side["stage_labels"].update({"Concept prompt": "Prepare the concept brief", "design_llm concept": "Qwen invents the piece",
                                         "Concept parse": "Check the concept is unique", "design_llm concept repair": "Qwen rewrites the clash",
                                         "Concept final": "Fix the request"})
        return side
    # ONE graph with or without the Qwen brief: the page route drops these nodes (and the links to them) when the class is
    # not installed (app_routes.load_graph -> tools/app/optional_nodes.py). The consumers' brief inputs are optional.
    side["optional_nodes"] = [{"class_type": "DesignLLM_Advanced",
                               "titles": ["design_llm creative brief", "Brief prompt"],
                               "note": "the design-LLM node pack is not installed: the rules write the brief"}]
    # box setup (README / install notes): models load in turn, not all at once (62 GB box)
    side["server_flags"] = {"comfyui": ["--mmap-torch-files", "--cache-ram", "16", "--use-sage-attention"],
                            "why": "ACE / MOSS / LTX / Gemma were all held in RAM and ComfyUI was OOM-killed at the LTX load on "
                                   "62 GB without the first two flags (box test 2026-10-26, 16_app/PLAN.md)"}
    return side


# --------------------------------------------------------------------------------------------- checks
def check_titles(g: dict, sidecar: dict) -> list[str]:
    bad = []
    titles: dict[str, list[str]] = {}
    for nid, n in g.items():
        titles.setdefault((n.get("_meta") or {}).get("title", ""), []).append(nid)
    for t in sidecar.get("required_titles") or IN_TITLES + OUT_TITLES:
        if len(titles.get(t, [])) != 1:
            bad.append(f"title {t!r}: expected exactly one node, found {titles.get(t, [])}")
    for f in sidecar["fields"]:
        if f.get("kind") == "media":                  # the Add-media box maps to several IN nodes (MEDIA_FIELD); each named node must exist exactly once (a graph without the concept step has no brief_node)
            for key in ("image_node", "images_node", "audio_node", "brief_node"):
                if f.get(key) and len(titles.get(f[key], [])) != 1:
                    bad.append(f"field {f['id']}: {key} {f[key]!r}: expected exactly one node, found {titles.get(f[key], [])}")
            continue
        ids = titles.get(f["node_title"], [])
        if len(ids) == 1 and f["input_name"] not in g[ids[0]]["inputs"]:
            bad.append(f"field {f['id']}: node {ids[0]} has no input {f['input_name']!r}")
    for o in sidecar["outputs"]:
        if o["node_title"] not in titles:
            bad.append(f"output {o['node_title']!r} has no node")
    return bad


def check_reachability(g: dict) -> list[str]:
    """Every node must feed an output node (PreviewImage / SaveVideo)."""
    outs = [n for n, v in g.items() if v["class_type"] in ("PreviewImage", "SaveVideo", "SaveImage", "PreviewAny", "MographArchive", "MographSaveFinalUpscaled")]
    seen, stack = set(), list(outs)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for v in g[n]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in g:
                stack.append(v[0])
    return [f"node {n} ({g[n]['class_type']}) feeds no output" for n in g if n not in seen]


def fetch_object_info(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=60) as r:
        return json.load(r)


def _input_spec(info: dict, name: str):
    inp = info.get("input", {})
    for sect in ("required", "optional"):
        if name in inp.get(sect, {}):
            return sect, inp[sect][name]
    return None, None


def check_object_info(g: dict, oi: dict, fill: bool = False) -> tuple[list[str], list[str]]:
    """-> (problems, notes). Classes missing on that server are notes (pack not installed there), not problems.
    Checks unknown inputs, missing required inputs (optionally filled with defaults), link output index and type."""
    bad, notes, missing_cls = [], [], {}
    for nid, n in g.items():
        info = oi.get(n["class_type"])
        if info is None:
            missing_cls.setdefault(n["class_type"], []).append(nid)
            continue
        ins = n["inputs"]
        for name in list(ins):
            sect, spec = _input_spec(info, name)
            if sect is None and "." not in name:
                bad.append(f"{nid} ({n['class_type']}): unknown input {name!r}")
        for name, spec in info.get("input", {}).get("required", {}).items():
            if name in ins:
                continue
            typ, opts = spec[0], (spec[1] if len(spec) > 1 else {})
            if isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if fill:
                    ins[name] = (opts.get("default") if "default" in opts else (typ[0] if isinstance(typ, list) else
                                 {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[typ]))
                else:
                    bad.append(f"{nid} ({n['class_type']}): required widget input {name!r} missing")
            elif not (n["class_type"] == "ComfyMathExpression"):
                bad.append(f"{nid} ({n['class_type']}): required link input {name!r} missing")
        for name, v in ins.items():
            if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)):
                continue
            src = g.get(v[0])
            sinfo = oi.get(src["class_type"]) if src else None
            if sinfo is None:
                continue
            outs = sinfo.get("output", [])
            if not 0 <= v[1] < len(outs):
                bad.append(f"{nid}.{name} -> {v[0]}[{v[1]}]: {src['class_type']} has {len(outs)} outputs")
                continue
            _, spec = _input_spec(info, name)
            want = spec[0] if spec else None
            have = outs[v[1]]
            # ComfyUI's validate_node_input (non-strict) accepts any overlap of the comma-split type sets,
            # e.g. INT -> "FLOAT,INT"
            overlap = (isinstance(want, str) and isinstance(have, str)
                       and bool({t.strip() for t in want.split(",")} & {t.strip() for t in have.split(",")}))
            if isinstance(want, str) and isinstance(have, str) and want != have and not overlap \
                    and "*" not in (want, have) \
                    and not want.startswith("COMFY_") and not have.startswith("COMFY_") and name != "values":
                bad.append(f"{nid}.{name}: {src['class_type']}[{v[1]}] is {have}, input wants {want}")
    for cls, ids in sorted(missing_cls.items()):
        notes.append(f"class {cls} not on that server ({len(ids)} node(s)); not validated")
    return bad, notes


def check_licences(g: dict) -> list[str]:
    """Video Depth Anything must be the Small (vits) model: the Base / Large weights are CC-BY-NC."""
    bad = []
    for nid, n in g.items():
        if n["class_type"] == "LoadVideoDepthAnythingModel":
            name = str(n["inputs"].get("model", ""))
            if "vits" not in name or "vitb" in name or "vitl" in name:
                bad.append(f"{nid}: Video Depth Anything model {name!r} must be the Small (vits) weights "
                           f"(Base / Large are CC-BY-NC)")
    return bad


def check_optional(g: dict, sidecar: dict, oi: dict | None) -> list[str]:
    """every ``optional_nodes`` entry names nodes that exist, and the graph without them still validates."""
    bad = []
    for spec in sidecar.get("optional_nodes", []):
        titles = {(n.get("_meta") or {}).get("title") for n in g.values()}
        for t in spec.get("titles", []):
            if t not in titles:
                bad.append(f"optional_nodes {spec['class_type']}: no node titled {t!r}")
        pruned, rep = optional_nodes.prune_optional(g, sidecar, {n["class_type"] for n in g.values()} - {spec["class_type"]})
        if not rep:
            bad.append(f"optional_nodes {spec['class_type']}: pruning removed nothing")
            continue
        bad += [f"without {spec['class_type']}: {p}" for p in check_reachability(pruned)]
        bad += [f"without {spec['class_type']}: {p}" for p in check_titles(pruned, sidecar)
                if not any(t in p for t in spec.get("titles", []))]
        if oi:
            b, _ = check_object_info(pruned, oi)
            bad += [f"without {spec['class_type']}: {p}" for p in b]
    return bad


def check(g: dict, sidecar: dict, oi: dict | None = None) -> tuple[list[str], list[str]]:
    bad = list(ltx_ic.check_graph(g))
    bad += check_titles(g, sidecar)
    bad += check_reachability(g)
    bad += check_licences(g)
    bad += check_optional(g, sidecar, oi)
    notes = []
    if oi:
        b, notes = check_object_info(g, oi)
        bad += b
    return bad, notes


# --------------------------------------------------------------------------------------------- 3D shot cuts
def insert_pivot_cut(g: dict) -> dict:
    """a copy of an app graph with the MographPivotCut node (nodes_fusion/depth_transition.py, tools/fusion/pivot_cut.py) between whatever feeds
    MographCompose.clips and the compose node: piece.compose.cuts[] entries of kind depth_pivot / orbit:<mode> / depth_dolly:<dir> are rendered into the
    two shots' clips (and the matte through the transition rides on the clips); a piece without such a cut passes through unchanged. The matte_k / depth_k
    inputs are the compose node's own links (lazy: only pulled when a cut names the shot). Opt-in (`--pivot-cut`): the generated graph files do not have it."""
    g = copy.deepcopy(g)
    comp = g.get("compose")
    if not comp or comp["class_type"] != "MographCompose":
        raise ValueError("no MographCompose node 'compose' in the graph")
    if "pivot" in g:
        raise ValueError("the graph already has a 'pivot' node")
    ins = comp["inputs"]
    inputs = {"piece": ins["piece"], "clips": ins["clips"]}
    for kind in ("matte", "depth"):
        for k in range(1, SLOTS + 1):
            if f"{kind}_{k}" in ins:
                inputs[f"{kind}_{k}"] = ins[f"{kind}_{k}"]
    add(g, "pivot", "MographPivotCut", "3D shot cuts (pivot / orbit / dolly)", **inputs)
    ins["clips"] = link("pivot", 0)
    if g.get("archive", {}).get("class_type") == "MographArchive":      # the rendered cuts ride on the pivot node's clips -> <stem>_cut_<a>-<b>_<kind>.mp4 + matte PNGs
        g["archive"]["inputs"]["cuts"] = link("pivot", 0)
    return g


# --------------------------------------------------------------------------------------------- main
def make(draft: bool = False, fill_from: dict | None = None, still: str = "moss", design: bool = False, auto: bool = False, angles: bool = False) -> tuple[dict, dict]:
    g = build_graph(draft, still, design, auto, angles)
    side = build_sidecar(draft, still, design, auto)
    if fill_from:
        check_object_info(g, fill_from, fill=True)
    return g, side


def emit_redesign(arm: str, out_dir: str, check_only: bool, oi: dict | None, fill: bool = False) -> int:
    """build, check and write one redesign arm graph (+ .app.json). -> number of problems."""
    name = REDESIGN_ARMS[arm][0]
    g, side = build_redesign(arm), build_redesign_sidecar(arm)
    if fill and oi:
        check_object_info(g, oi, fill=True)
    text, side_text = dump(g), dump(side)
    problems, notes = check(g, side, oi)
    print(f"{name}: {len(g)} nodes, {len({v['class_type'] for v in g.values()})} classes, {len(side['fields'])} fields, {len(side['outputs'])} outputs")
    if check_only:
        for suffix, want in ((".json", text), (".app.json", side_text)):
            path = os.path.join(out_dir, name + suffix)
            if os.path.isfile(path) and open(path, encoding="utf-8").read() != want:
                problems.append(f"{name}{suffix} on disk is stale (run build_graphs.py to regenerate)")
            elif not os.path.isfile(path):
                problems.append(f"{name}{suffix} not written yet")
    for n in notes:
        print(f"  note: {n}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"  {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    if not check_only:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(out_dir, name + ".app.json"), "w", encoding="utf-8") as f:
            f.write(side_text)
        print(f"  wrote {os.path.join(out_dir, name)}.json (+ .app.json)")
    return len(problems)


def dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="only the full graph (fusion.json)")
    ap.add_argument("--check", action="store_true", help="validate (and compare with the files on disk); writes nothing")
    ap.add_argument("--server", help="ComfyUI URL: validate classes / inputs / link types against /object_info")
    ap.add_argument("--fill", action="store_true", help="with --server: fill missing required widget inputs with defaults")
    ap.add_argument("--out-dir", default=GRAPHS)
    ap.add_argument("--still", choices=STILL_MODELS, default="moss",
                    help="still model of the full graph: moss (Apache 2.0)")
    ap.add_argument("--pivot-cut", action="store_true", help="insert MographPivotCut before compose (3D shot cuts: depth_pivot / orbit:* / depth_dolly:*); writes <name>_pivot.json")
    ap.add_argument("--layers", type=int, choices=(1, 3), default=1, help="3 = three-layer shots, tools/app/layered_shots.py; writes <name>_layers.json")
    ap.add_argument("--angles", action="store_true", help="opt-in: add the angle-still chain (MOSS edit LoRA moss_image_edit_chinoll, licence undeclared; 24 GB tier); writes <name>_angles.json. Without it the graphs have NO node that loads the LoRA")
    a = ap.parse_args()
    oi = fetch_object_info(a.server) if a.server else None
    kinds = [False] + ([DESIGN_NAME, AUTO_NAME] if a.still == "moss" else [])         # False = fusion, then the Qwen design graph and its auto variant
    bad_total = 0
    for kind in kinds:
        auto = kind == AUTO_NAME
        design = kind in (DESIGN_NAME, AUTO_NAME)
        name = kind if design else ("fusion" if a.still == "moss" else "fusion_" + a.still)
        g, side = make(False, oi if a.fill else None, a.still, design, auto, a.angles)
        if a.layers == 3 and design:
            from tools.app import layered_shots
            g, side, suffix = layered_shots.build_graphs_hook(g, side, design, auto)
            name += suffix
        if a.pivot_cut:
            g = insert_pivot_cut(g)
            name += "_pivot"
        if a.angles:
            name += "_angles"
        text, side_text = dump(g), dump(side)
        problems, notes = check(g, side, oi)
        n_cls = len({v["class_type"] for v in g.values()})
        print(f"{name}: {len(g)} nodes, {n_cls} classes, {len(side['fields'])} fields, {len(side['outputs'])} outputs")
        if a.check:
            for suffix, want in ((".json", text), (".app.json", side_text)):
                path = os.path.join(a.out_dir, name + suffix)
                if os.path.isfile(path) and open(path, encoding="utf-8").read() != want:
                    problems.append(f"{name}{suffix} on disk is stale (run build_graphs.py to regenerate)")
                elif not os.path.isfile(path):
                    problems.append(f"{name}{suffix} not written yet")
        for n in notes:
            print(f"  note: {n}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        print(f"  {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
        bad_total += len(problems)
        if not a.check:
            os.makedirs(a.out_dir, exist_ok=True)
            with open(os.path.join(a.out_dir, name + ".json"), "w", encoding="utf-8") as f:
                f.write(text)
            with open(os.path.join(a.out_dir, name + ".app.json"), "w", encoding="utf-8") as f:
                f.write(side_text)
            print(f"  wrote {os.path.join(a.out_dir, name)}.json (+ .app.json)")
    if a.still == "moss":
        for arm in REDESIGN_ARMS:
            bad_total += emit_redesign(arm, a.out_dir, a.check, oi, a.fill)
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
