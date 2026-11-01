#!/usr/bin/env python
"""THREE-LAYER shot route for the app graphs: clean footage, subject matte, text-rendered background (open models only, NO Claude). Docs: 16_app/LAYERS.md.

The bug it fixes: in the app graph every shot is ONE union-control LTX shot (type control pass -> LTXAddVideoICLoRAGuide) and the Alpha-Gen matte was made
from THAT clip. The clip shows giant stone letters and often no subject, the "subject matte" is the letters, and compose cannot put type between the
subject and the background. User requirement: "They need to stay separate, to be able to put text between the subject and the background."

What it does: takes the app graph that `tools/app/build_graphs.py` writes (`fusion_design_qwen[_auto]`) and swaps the per-shot union-control chains `c{k}.*`:

    l{k}.*   CLEAN shot: plain LTX image-to-video from the slot's first frame (MOSS still / previous CLEAN clip) with the shot prompt minus the letter
             clauses (MographLayerPrompt). No IC guide, no type, base model (no LoRA).
    a{k}.*   SUBJECT MATTE: the existing Alpha-Gen chain, now fed from the CLEAN clip (MographLayerGather builds its reference from the clean clip).
    t{k}.*   TEXT-RENDERED BACKGROUND, only for shots with a text pass (control guide > 0; guide == 0 never executes it): the same union-control chain
             as before, but started from the CLEAN clip's latent (video-to-video: the sampler runs the tail of the distilled schedule, `v2v_start`),
             guided by the type control pass (LTXAddVideoICLoRAGuide, strength = slot guide) and the first frame, so the letters form in the same scene
             with the same camera. MographLayerGate makes it wait for all clean shots.
    gather   MographLayerGather replaces MographFootageGather (same outputs): the compose clip is the BACKGROUND (text clip, else the clean clip) and the
             clean clip rides along as the subject layer; compose takes the subject from it (tools/fusion/layers_recipe.py); the archive saves
             sK_clean.mp4 / sK_matte/ / sK_textbg.mp4.

    python tools/app/layered_shots.py                    # write app/graphs/<base>_layers.json (+ .app.json) for every base graph on disk
    python tools/app/layered_shots.py --check            # build in memory, validate, compare with the files on disk, write nothing
    python tools/app/layered_shots.py --v2v-start 0.909375  # the refine-pass tail (3 steps): scene holds best but the letters stay ghost outlines

The hook for build_graphs.py (`--layers 3`) is `build_graphs_hook()` below; the exact lines are in 16_app/LAYERS.md.
Order of model use (data dependencies, no manual scheduling): all clean shots (base model) -> all text passes (union-control LoRA) -> Alpha-Gen
(alpha LoRA) -> depth -> flow -> design -> compose.
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
from tools.fusion import plate_layers as PLT  # noqa: E402
from tools.fusion import text_guard as TG  # noqa: E402

GRAPHS = os.path.join(ROOT, "app", "graphs")
BASES = ("fusion_design_qwen", "fusion_design_qwen_auto")      # graphs that get a `_layers` twin when their file exists
SUFFIX = "_layers"
DISTILLED = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]    # the shared 8-step distilled schedule (node 38)
V2V_START = 0.975              # 4 steps. Probe 2026-10-28 (21_batch-3060/v006, tools/app/text_pass_probe.py): 0.909375 (3 steps, = the refine pass) forms only ghost outlines; 0.975 forms solid letters and keeps subject + camera; 0.99375 / 1.0 lose the subject.

LAYER_CLASSES = {"MographLayerPrompt": ("ComfyUI-Mograph", 1), "MographLayerClean": ("ComfyUI-Mograph", 1),
                 "MographLayerGate": ("ComfyUI-Mograph", 1), "MographLayerGather": ("ComfyUI-Mograph", 13), "MographLtxRead": ("ComfyUI-Mograph", 2),
                 "MographPlatePrompt": ("ComfyUI-Mograph", 1), "MographPlateRef": ("ComfyUI-Mograph", 1)}


def _register_classes() -> bool:
    for k, v in LAYER_CLASSES.items():
        ltx_ic.CLASSES.setdefault(k, v)
    try:
        from tools.app import build_graphs  # noqa: F401  (side effect: ltx_ic.CLASSES.update(EXTRA_CLASSES))
        return True
    except Exception:   # noqa: BLE001  (another worker may be editing it)
        return False


APP_CLASSES_KNOWN = _register_classes()


# ---------------------------------------------------------------------------------------------- pure helpers (also used by the nodes)
def shot_guide(piece: dict, sid: str) -> float:
    """control guide of a shot exactly as MographShotSlot reads it (control.guide, else control.strength, else 1.0)."""
    shot = next((s for s in (piece or {}).get("shots") or [] if s.get("id") == sid), {})
    ctl = shot.get("control") or {}
    return float(ctl.get("guide", ctl.get("strength", 1.0)))


def wants_plate(piece: dict, sid: str, plates: bool = True) -> bool:
    """does the shot take the CLEAN PLATE route (`plate: "clean"`, tools/fusion/plate_layers.py)? `plates` is the graph's switch (MographLayerGather.plates: the 24 GB tier);
    off = the shot is an ordinary layered shot and its plate chain never runs."""
    return bool(plates) and PLT.wants_plate(piece, sid)


def wants_text(piece: dict, sid: str, plates: bool = False) -> bool:
    """does the shot get a text-rendered background (union-control V2V)? guide > 0, unless the piece opts out with layers.text = false. A plate shot (plates on) has
    no text-rendered background: the type is drawn flat between the plate and the subject."""
    if ((piece or {}).get("layers") or {}).get("text") is False:
        return False
    if wants_plate(piece, sid, plates):
        return False
    return shot_guide(piece, sid) > 0.0


def _letter_clauses() -> list[str]:
    from tools.fusion import plan_rules as PR
    return [PR._LETTER_PHRASE[k] for k in ("built_in", "traced") if k in PR._LETTER_PHRASE]


def scrub_prompt(prompt: str, piece: dict | None, sid: str) -> str:
    """SUBJECT FROM THE USER'S PICTURE: the user's typed direction / palette never goes into a video prompt of a shot that starts from the user's picture. The palette colour names of
    `piece.user_style` (else the typed direction in `piece.request`) are deleted from the prompt unless that picture's `shows` (`piece.request.shows`) contains them. The slot prompt feeds
    BOTH the clean chain (through clean_prompt) and the text-pass chain, so this is applied where the slot prompt is made (nodes_fusion/footage.py MographShotSlot). Other shots: unchanged."""
    from tools.fusion import subject_guard as SG
    return SG.scrub_prompt(prompt, piece, sid)


def clean_prompt(prompt: str, piece: dict | None = None, sid: str | None = None) -> str:
    """the planner's shot prompt without the clauses that ask for letters / strands in the footage (the clean clip shows the scene only): the built_in / traced
    letter phrases and the owner clause `the words "X" formed as <material> letters`."""
    from tools.fusion import owner as OW
    bad = [c.strip().lower() for c in _letter_clauses()]
    if piece is not None and sid:
        prompt = scrub_prompt(prompt, piece, sid)    # idempotent: the slot prompt was already scrubbed
    prompt = OW.strip_clause(prompt)             # the owner clause: the words "X" formed as <material> letters (tools/fusion/owner.py)
    parts = [p.strip() for p in str(prompt or "").split(",")]
    keep = [p for p in parts if p and p.lower() not in bad]
    from tools.fusion import text_guard as TG
    return TG.strip_text(", ".join(keep))[0]          # the clean clip has NO control pass: it never names text / letters / signs (the renderer must not invent its own)


def layer_clip(clean: dict, text: dict | None = None, plate: dict | None = None) -> dict:
    """MOGRAPH_CLIP of a layered shot: u8 = the background (text clip if any, else the clean clip), subject = the clean u8, textbg = bool. A plate shot (`plate` = the Clean
    Plate V2V clip): u8 = the plate, textbg False, `plate` True; MographCompose bakes the independent motion (plate_layers.bake_clip) before the recipe runs."""
    c = dict(clean)
    c["subject"] = clean["u8"]
    c["textbg"] = False
    if plate is not None:
        p = plate["u8"]
        n = int(clean["u8"].shape[0])
        if int(p.shape[0]) != n:
            import torch
            p = p[:n] if p.shape[0] > n else torch.cat([p, p[-1:].expand(n - p.shape[0], -1, -1, -1)], 0)
        c["u8"], c["plate"] = p, True
        return c
    if text is not None:
        t = text["u8"]
        n = int(clean["u8"].shape[0])
        if int(t.shape[0]) != n:                                  # both come from c.19 (8k+1); be safe anyway
            import torch
            t = t[:n] if t.shape[0] > n else torch.cat([t, t[-1:].expand(n - t.shape[0], -1, -1, -1)], 0)
        c["u8"], c["textbg"] = t, True
    return c


def v2v_sigmas(start: float = V2V_START) -> str:
    """tail of the distilled schedule from `start` (one of its values) -> the ManualSigmas string."""
    s = [x for x in DISTILLED if x <= start + 1e-9]
    if start not in DISTILLED:
        raise ValueError(f"v2v_start {start}: one of {DISTILLED[:-1]}")
    return ", ".join(repr(x) for x in s)


def steps_of(start: float = V2V_START) -> int:
    return len(v2v_sigmas(start).split(",")) - 1


# ---------------------------------------------------------------------------------------------- graph helpers
def link(node_id: str, slot: int = 0) -> list:
    return [node_id, slot]


def _add(g: dict, nid: str, cls: str, title: str, **inputs) -> str:
    assert nid not in g, f"duplicate node id {nid}"
    g[nid] = {"inputs": inputs, "class_type": cls, "_meta": {"title": title}}
    return nid


def _slots(g: dict) -> list[int]:
    out = []
    for nid, n in g.items():
        m = re.fullmatch(r"slot(\d+)", nid)
        if m and n["class_type"] == "MographShotSlot":
            out.append(int(m.group(1)))
    return sorted(out)


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def _copy_chain(g: dict, k: int, new_prefix: str, tag: str) -> list[str]:
    """copy the per-shot chain c{k}.* to {new_prefix}.*: links inside the chain follow, links out of it (shared loaders, slot) stay. -> new ids."""
    old = f"c{k}."
    ids = [n for n in g if n.startswith(old)]
    new = []
    for nid in ids:
        n = copy.deepcopy(g[nid])
        for name, v in n["inputs"].items():
            if _is_link(v) and v[0].startswith(old):
                n["inputs"][name] = [new_prefix + v[0][len(old):], v[1]]
        title = (n.get("_meta") or {}).get("title", "")
        n["_meta"] = {"title": re.sub(r"\[s%d\]" % k, f"[s{k} {tag}]", title) if f"[s{k}]" in title else f"{title} [s{k} {tag}]"}
        g[new_prefix + nid[len(old):]] = n
        new.append(new_prefix + nid[len(old):])
    return new


def _bypass(g: dict, nid: str, out_to_in: dict) -> None:
    """remove node `nid` and rewire every consumer of its outputs to the node's own inputs (out_to_in: output index -> input name)."""
    node = g.pop(nid)
    for n in g.values():
        for name, v in n["inputs"].items():
            if _is_link(v) and v[0] == nid:
                n["inputs"][name] = copy.deepcopy(node["inputs"][out_to_in[v[1]]])


def prune_unreachable(g: dict) -> list[str]:
    """drop the nodes that feed no output node (PreviewImage / PreviewAny / SaveVideo / SaveImage / MographArchive) -> the dropped ids."""
    outs = [n for n, v in g.items() if v["class_type"] in ("PreviewImage", "SaveVideo", "SaveImage", "PreviewAny", "MographArchive", "MographSaveFinalUpscaled")]
    seen, stack = set(), list(outs)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for v in g[n]["inputs"].values():
            if _is_link(v) and v[0] in g:
                stack.append(v[0])
    dropped = [n for n in g if n not in seen]
    for n in dropped:
        del g[n]
    return dropped


# ---------------------------------------------------------------------------------------------- the clean-plate chain
def add_plate_chain(g: dict, ks: list[int], piece: list, clean: list) -> list[str]:
    """The CLEAN PLATE stage (LTX-2.5 Clean Plate IC-LoRA, V2V of the CLEAN clip, 16_app/CLEAN_PLATE.md; probe 24_clean-plate/v001): per slot k a copy `p{k}.*` of the Alpha-Gen
    chain `a{k}.*` with the clean-plate LoRA (`pl.5`, one shared loader node) and the empty-scene prompt of the shot (`MographPlatePrompt` -> `pl{k}.6` -> `pl{k}.8`), reading
    the clean clip at the Alpha-Gen reference size (`MographPlateRef`, which also waits for every text pass, so the plate stage runs after the union-control stage and before
    Alpha-Gen). `p{k}.dec` is a COLOUR clip (gray False) that goes to `MographLayerGather.plate_k` (lazy: requested only for a shot with `plate: "clean"` while the graph's
    `plates` switch is on, so no other shot ever runs it). The plate regenerates the whole frame and peaks near the card's 24 GB, so it is its own stage by data dependency:
    nothing downstream of it (Alpha-Gen, depth, the design LLM) can start before the plate clip exists, and the design LLM / depth model free ComfyUI's models first.
    -> the ids of the nodes that were added."""
    before = set(g)
    pl5 = copy.deepcopy(g["a.5"])
    pl5["inputs"]["lora_name"] = PLT.LORA
    pl5["_meta"] = {"title": "IC-LoRA clean plate (model only)"}
    g["pl.5"] = pl5
    for k in ks:
        _add(g, f"pl{k}.prompt", "MographPlatePrompt", f"Plate prompt [s{k}]", piece=copy.deepcopy(piece), index=k)
        _add(g, f"pl{k}.6", "CLIPTextEncode", f"Plate prompt (positive) [s{k}]", text=link(f"pl{k}.prompt"), clip=copy.deepcopy(g["a.6"]["inputs"]["clip"]))
        c8 = copy.deepcopy(g["a.8"])
        c8["inputs"]["positive"] = link(f"pl{k}.6")
        c8["_meta"] = {"title": f"LTXVConditioning (plate) [s{k}]"}
        g[f"pl{k}.8"] = c8
        _add(g, f"pl{k}.ref", "MographPlateRef", f"Plate reference (clean clip) [s{k}]", piece=copy.deepcopy(piece), clean=copy.deepcopy(clean), index=k,
             **{f"text_{j}": link(f"t{j}.dec") for j in ks if f"t{j}.dec" in g})
        for nid in [n for n in g if n.startswith(f"a{k}.") and n != f"a{k}.dec"]:
            n = copy.deepcopy(g[nid])
            for name, v in n["inputs"].items():
                if _is_link(v):
                    if v[0].startswith(f"a{k}."):
                        n["inputs"][name] = [f"p{k}." + v[0][len(f"a{k}."):], v[1]]
                    elif v[0] == "a.8":
                        n["inputs"][name] = [f"pl{k}.8", v[1]]
                    elif v[0] == "a.5":
                        n["inputs"][name] = ["pl.5", v[1]]
            n["_meta"] = {"title": re.sub(r"\[alpha s%d\]" % k, f"[plate s{k}]", (n.get("_meta") or {}).get("title", "")) or f"plate s{k}"}
            g[f"p{k}." + nid[len(f"a{k}."):]] = n
        g[f"p{k}.17"]["inputs"]["input"] = link(f"pl{k}.ref")                       # the clean clip at the reference size (not the gather: the gather waits for this chain)
        _add(g, f"p{k}.dec", "MographDecodeClip", f"Decode plate clip [s{k}]", vae=link("3"), samples=link(f"p{k}.39", 2), piece=copy.deepcopy(piece), index=k, gray=False)
    return [n for n in g if n not in before]


# ---------------------------------------------------------------------------------------------- the conversion
def convert(g: dict, side: dict | None = None, *, v2v_start: float = V2V_START, plate: bool = True) -> tuple[dict, dict | None]:
    """app graph (+ sidecar) with union-control shot chains `c{k}.*` -> the three-layer twin. Pure function: the inputs are not modified.
    plate=True: also the clean-plate chains `p{k}.*` (lazy; `add_plate_chain`; needs the Alpha-Gen chain `a{k}.*` of the full graph to copy)."""
    g = copy.deepcopy(g)
    ks = _slots(g)
    if not ks:
        raise ValueError("no MographShotSlot nodes (slot1..) in the graph: not an app footage graph")
    gather = g.get("gather")
    if not gather or gather["class_type"] != "MographFootageGather":
        raise ValueError("no MographFootageGather node 'gather' (already converted?)")
    for k in ks:
        if f"c{k}.dec" not in g or f"c{k}.32" not in g:
            raise ValueError(f"slot{k}: no union-control chain c{k}.* (already converted?)")
    two_stage = f"c{ks[0]}.58" in g
    piece, request, first = gather["inputs"]["piece"], gather["inputs"]["request"], gather["inputs"].get("first")
    _add(g, "lay_sig", "ManualSigmas", f"V2V sigmas (text pass: the last {steps_of(v2v_start)} of the 8 distilled steps)", sigmas=v2v_sigmas(v2v_start))
    for k in ks:
        _copy_chain(g, k, f"l{k}.", "clean")
        _copy_chain(g, k, f"t{k}.", "text")
    _add(g, "lay_clean", "MographLayerClean", "Layer clean clips (barrier)", piece=piece,
         **{f"clean_{k}": link(f"l{k}.dec") for k in ks})
    for k in ks:
        slot = f"slot{k}"
        # ---- CLEAN chain: no IC guide, no crop-guides, base model, prompt without the letter clauses
        p = f"l{k}."
        _add(g, f"lay_prompt{k}", "MographLayerPrompt", f"Clean prompt [s{k}]", prompt=link(slot, 0))
        g[p + "6"]["inputs"]["text"] = link(f"lay_prompt{k}")
        _add(g, f"lay_neg{k}", "CLIPTextEncode", f"Clean negative: no text [s{k}]", text=TG.NEG_NO_TEXT, clip=copy.deepcopy(g[p + "6"]["inputs"]["clip"]))       # no control pass = no text, ever
        g[p + "8"]["inputs"]["negative"] = link(f"lay_neg{k}")
        _bypass(g, p + "32", {0: "positive", 1: "negative", 2: "latent"})
        _bypass(g, p + "41", {0: "positive", 1: "negative", 2: "latent"})
        for nid in [n for n in g if n.startswith(p)]:
            n = g[nid]
            if n["class_type"] == "CFGGuider" and n["inputs"].get("model") == ["5", 0]:
                n["inputs"]["model"] = link("1")                    # the base distilled model: no union-control LoRA on the clean shot
        # ---- TEXT chain: starts from the clean latent, tail of the distilled schedule, same union-control guide as the old shot
        q = f"t{k}."
        if k == ks[0]:
            _add(g, "lay_negt", "CLIPTextEncode", "Text-pass negative: only the control's word", text=TG.NEG_TEXT_PASS, clip=copy.deepcopy(g[q + "6"]["inputs"]["clip"]))
        g[q + "8"]["inputs"]["negative"] = link("lay_negt")
        _add(g, f"lay_gate{k}", "MographLayerGate", f"Clean latent -> text pass [s{k}]", latent=link(p + "40", 0), after=link("lay_clean"))
        g[q + "31"]["inputs"]["latent"] = link(f"lay_gate{k}")
        g[q + "39"]["inputs"]["sigmas"] = link("lay_sig")
        g[q + "dec"]["_meta"]["title"] = f"Decode text clip [s{k}]"
        g[p + "dec"]["_meta"]["title"] = f"Decode clean clip [s{k}]"
        # ---- drop the old chain, chain the NEXT clean shot from this clean clip
        for nid in [n for n in g if n.startswith(f"c{k}.")]:
            del g[nid]
        if k > 1:
            g[slot]["inputs"]["prev"] = link(f"l{k - 1}.dec")
    plated = bool(plate) and "a.5" in g and all(f"a{k}.39" in g for k in ks)
    if plated:
        add_plate_chain(g, ks, piece, link("lay_clean"))
    g["gather"] = {"inputs": {"piece": piece, "request": request, **({"first": first} if first else {}), "clean": link("lay_clean"),
                              **{f"text_{k}": link(f"t{k}.dec") for k in ks},
                              **({"plates": False, **{f"plate_{k}": link(f"p{k}.dec") for k in ks}} if plated else {})},
                   "class_type": "MographLayerGather", "_meta": gather.get("_meta", {"title": "Footage gather"})}
    comp = g.get("compose")
    if comp and comp["class_type"] == "MographCompose" and g.get("type_final") and g.get("design_piece"):       # the VLM reading check of LTX-owned words, between the text pass and compose
        _add(g, "ltx_read", "MographLtxRead", "Read the LTX letters (VLM legibility check)", piece=link("design_piece"), type=link("type_final"), clips=comp["inputs"]["clips"],
             model="", mmproj="auto", n_gpu_layers=-1, n_ctx=16384, kv_cache="q8_0", flash_attn=True, mmproj_on_gpu=True, n_cpu_moe=0, handler="auto",
             image_max_tokens=-1, top_k=40, top_p=0.9, repeat_penalty=1.0, seed=1)          # model "" = off in the base graph; tools/app/llm_arms.py fills the arm's model
        comp["inputs"]["clips"] = link("ltx_read")
    prune_unreachable(g)
    if side is not None:
        side = copy.deepcopy(side)
        side["title"] = str(side.get("title", "Fusion")) + " - three-layer shots"
        side["order"] = int(side.get("order", 20)) + 2
        side["note"] = (str(side.get("note", "")) + " Footage route: three layers per shot - a CLEAN plain image-to-video clip (the subject, no type), the "
                        "Alpha-Gen matte cut from it, and a text-rendered background (LTX union-control video-to-video of the clean clip, only for shots "
                        "with a type control guide > 0); compose puts the type between the background and the clean subject.").strip()
        side["layers"] = dict(version=1, clean_prefix="l", text_prefix="t", v2v_start=v2v_start, v2v_steps=steps_of(v2v_start), two_stage=two_stage,
                              doc="16_app/LAYERS.md", **({"plate_prefix": "p", "plate_lora": PLT.LORA, "plate_doc": "16_app/CLEAN_PLATE.md"} if plated else {}))
        side.setdefault("stage_labels", {}).update({
            "Layer clean clips (barrier)": "Finish the clean shots", "Footage gather": "Gather the layers",
        })
        if plated:
            side["stage_labels"].update({f"Plate prompt [s{k}]": f"Describe the empty scene of shot {k}" for k in ks})
            side["stage_labels"].update({f"Plate reference (clean clip) [s{k}]": f"Prepare the clean clip of shot {k} for the plate" for k in ks})
            side["stage_labels"].update({f"Decode plate clip [s{k}]": f"Clean plate of shot {k} (subject off the background)" for k in ks})
            side["stage_labels"].update({f"SamplerCustomAdvanced [plate s{k}]": f"Remove the subject from shot {k} (clean plate)" for k in ks})
    return g, side


def build_graphs_hook(g: dict, side: dict, design: bool = True, auto: bool = False, **opts) -> tuple[dict, dict, str]:
    """what build_graphs.main() calls for `--layers 3`: (graph, sidecar) of the union-control build -> (layered graph, sidecar, file stem suffix)."""
    g2, s2 = convert(g, side, **opts)
    return g2, s2, SUFFIX


# ---------------------------------------------------------------------------------------------- checks
def check(g: dict) -> list[str]:
    """structural problems of a layered graph: ids / output indices / classes (ltx_ic.check_graph), every node feeds an output, per slot: a clean chain
    without IC guide and without LoRA, a text chain with the union-control guide started from the gated clean latent, the gather wired to both, the
    Alpha-Gen chain fed by the gather."""
    bad = list(ltx_ic.check_graph(g))
    if not APP_CLASSES_KNOWN:
        bad = [b for b in bad if "not in CLASSES" not in b]
    outs = [n for n, v in g.items() if v["class_type"] in ("PreviewImage", "SaveVideo", "SaveImage", "PreviewAny", "MographArchive", "MographSaveFinalUpscaled")]
    seen, stack = set(), list(outs)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack += [v[0] for v in g[n]["inputs"].values() if _is_link(v) and v[0] in g]
    bad += [f"node {n} ({g[n]['class_type']}) feeds no output" for n in g if n not in seen]
    ga = g.get("gather")
    if not ga or ga["class_type"] != "MographLayerGather":
        bad.append("gather is not a MographLayerGather")
        return bad
    for k in _slots(g):
        if any(n.startswith(f"c{k}.") for n in g):
            bad.append(f"slot{k}: old union-control chain c{k}.* still present")
        cl = [n for n in g if n.startswith(f"l{k}.")]
        tx = [n for n in g if n.startswith(f"t{k}.")]
        if not cl or f"l{k}.dec" not in g:
            bad.append(f"slot{k}: no clean chain")
        if not tx or f"t{k}.dec" not in g:
            bad.append(f"slot{k}: no text chain")
            continue
        for n in cl:
            if g[n]["class_type"] in ("LTXAddVideoICLoRAGuide", "LTXVCropGuides"):
                bad.append(f"{n}: IC guide node in the clean chain")
            if g[n]["class_type"] == "CFGGuider" and g[n]["inputs"].get("model") != ["1", 0]:
                bad.append(f"{n}: clean chain must use the base model (node 1), has {g[n]['inputs'].get('model')}")
        prompt = g[f"l{k}.6"]["inputs"].get("text")
        if prompt != [f"lay_prompt{k}", 0]:
            bad.append(f"l{k}.6.text is not the clean prompt")
        guide = g.get(f"t{k}.32")
        if not guide or guide["class_type"] != "LTXAddVideoICLoRAGuide":
            bad.append(f"t{k}.32: no IC-LoRA guide in the text chain")
        else:
            if guide["inputs"].get("strength") != [f"slot{k}", 6]:
                bad.append(f"t{k}.32.strength is not slot{k}[6] (the control guide)")
            if g[f"t{k}.31"]["inputs"].get("latent") != [f"lay_gate{k}", 0]:
                bad.append(f"t{k}.31 does not start from the gated clean latent")
        if g[f"t{k}.39"]["inputs"].get("sigmas") != ["lay_sig", 0]:
            bad.append(f"t{k}.39 does not use the V2V sigmas")
        if g.get(f"lay_gate{k}", {}).get("inputs", {}).get("latent") != [f"l{k}.40", 0]:
            bad.append(f"lay_gate{k}: latent is not l{k}.40[0]")
        if ga["inputs"].get(f"text_{k}") != [f"t{k}.dec", 0]:
            bad.append(f"gather.text_{k} is not t{k}.dec")
        if k > 1 and g[f"slot{k}"]["inputs"].get("prev") != [f"l{k - 1}.dec", 0]:
            bad.append(f"slot{k}.prev is not the clean clip of shot {k - 1}")
        if f"a{k}.17" in g and g[f"a{k}.17"]["inputs"].get("input") != ["gather", k]:
            bad.append(f"a{k}.17 (Alpha-Gen reference) is not gather[{k}]")
    if ga["inputs"].get("clean") != ["lay_clean", 0]:
        bad.append("gather.clean is not the clean barrier")
    if "pl.5" in g or any(n.startswith("p1.") for n in g):          # the clean-plate stage (add_plate_chain)
        if g.get("pl.5", {}).get("inputs", {}).get("lora_name") != PLT.LORA:
            bad.append(f"pl.5 is not the clean-plate LoRA {PLT.LORA}")
        if ga["inputs"].get("plates") is not False and ga["inputs"].get("plates") is not True:
            bad.append("gather.plates (the plate switch) is missing")
        for k in _slots(g):
            if g.get(f"p{k}.34", {}).get("inputs", {}).get("model") != ["pl.5", 0]:
                bad.append(f"p{k}.34 (plate sampler guider) does not use the clean-plate LoRA")
            if g.get(f"p{k}.17", {}).get("inputs", {}).get("input") != [f"pl{k}.ref", 0]:
                bad.append(f"p{k}.17 (plate reference) is not pl{k}.ref (the gather would wait for its own plate)")
            for nm in ("29", "31"):
                if g.get(f"p{k}.{nm}", {}).get("inputs", {}).get("positive") != [f"pl{k}.8", 0]:
                    bad.append(f"p{k}.{nm}.positive is not the plate conditioning pl{k}.8")
            if g.get(f"pl{k}.6", {}).get("inputs", {}).get("text") != [f"pl{k}.prompt", 0]:
                bad.append(f"pl{k}.6.text is not the plate prompt node")
            if g.get(f"p{k}.dec", {}).get("inputs", {}).get("gray") is not False:
                bad.append(f"p{k}.dec must be a colour decode (gray false)")
            if ga["inputs"].get(f"plate_{k}") != [f"p{k}.dec", 0]:
                bad.append(f"gather.plate_{k} is not p{k}.dec")
            if g.get(f"a{k}.34", {}).get("inputs", {}).get("model") != ["a.5", 0]:
                bad.append(f"a{k}.34 (Alpha-Gen) lost its own LoRA")
    if "ltx_read" in g and g.get("compose", {}).get("inputs", {}).get("clips") != ["ltx_read", 0]:
        bad.append("compose.clips is not the LTX read node")
    return bad


def _write(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def _read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate and compare with the files on disk; write nothing")
    ap.add_argument("--base", action="append", help=f"base graph stem(s) in --graphs-dir (default {BASES}, those that exist)")
    ap.add_argument("--graphs-dir", default=GRAPHS)
    ap.add_argument("--out-dir", default=GRAPHS)
    ap.add_argument("--v2v-start", type=float, default=V2V_START, help=f"first sigma of the text pass, one of {DISTILLED[1:-1]} (default {V2V_START})")
    a = ap.parse_args(argv)
    bases = a.base or [b for b in BASES if os.path.isfile(os.path.join(a.graphs_dir, b + ".json"))]
    if not bases:
        print(f"no base graph in {a.graphs_dir} (run tools/app/build_graphs.py first)")
        return 1
    bad_total = 0
    for b in bases:
        g0, s0 = _read(os.path.join(a.graphs_dir, b + ".json")), _read(os.path.join(a.graphs_dir, b + ".app.json"))
        g, s = convert(g0, s0, v2v_start=a.v2v_start)
        name = b + SUFFIX
        problems = check(g)
        print(f"{name}: {len(g)} nodes (from {len(g0)}), text pass {steps_of(a.v2v_start)} steps from sigma {a.v2v_start}")
        if a.check:
            for suf, obj in ((".json", g), (".app.json", s)):
                path = os.path.join(a.out_dir, name + suf)
                if not os.path.isfile(path):
                    problems.append(f"{name}{suf} not written yet")
                elif _read(path) != obj:
                    problems.append(f"{name}{suf} on disk is stale (run tools/app/layered_shots.py)")
        else:
            os.makedirs(a.out_dir, exist_ok=True)
            _write(os.path.join(a.out_dir, name + ".json"), g)
            _write(os.path.join(a.out_dir, name + ".app.json"), s)
            print(f"  wrote {os.path.join(a.out_dir, name)}.json (+ .app.json)")
        for p in problems:
            print(f"  PROBLEM: {p}")
        print(f"  {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
        bad_total += len(problems)
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
