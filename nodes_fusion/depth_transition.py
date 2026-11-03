"""MOTNGRFX_DepthPivotTransition: the depth "pivot chunk" image transition as a ComfyUI node (17_depth-dolly v013, approved look).

    image_a, mask_a, image_b, mask_b (+ optional depth_a, depth_b)  ->  frames IMAGE, params_json STRING

A thin wrapper around ``mograph/modules/transitions/depth_pivot.py`` (the engine is ``tools/fusion/depth_orbit.py`` / ``depth_dolly.py``); the
only work done here is torch <-> numpy. Tensors: IMAGE [B,H,W,C] float 0..1, MASK [B,H,W] float 0..1 (straight from the Alpha-Gen matte output),
output IMAGE [N,H,W,3] float 0..1 with N = hold1 + travel + hold2 (72 by default).

LIVE VIDEO IS THE DEFAULT: an input batch with more than one frame is a clip and its frames are the live textures of the cards (image_a plays from
output frame 0, image_b from the frame it first becomes visible), the mask batch is the per-frame subject matte; a single image is a still
(``video`` = auto; on / off force it). ``params_json`` out is the FULL resolved parameter dict: paste it into the optional ``params_json`` in to
re-make the same result (it then overrides the widgets). The seed makes the cards reproducible. Density: ``chunks`` = background cards per image
(default 32, v009's density; the card size scales with it).

The conversion / run helpers below do not need ComfyUI, so they can be unit-tested without a checkout; the node class is only defined when
``comfy_api`` imports.
"""
from __future__ import annotations

import json

import numpy as np
import torch

try:
    from . import ensure_root_on_path
except ImportError:           # loaded outside the package (tests)
    def ensure_root_on_path():
        import os
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.append(root)
        return root

ensure_root_on_path()
from mograph.modules.transitions import depth_pivot as DP      # noqa: E402

CATEGORY = "MOTNGRFX/Fusion"
NODE_ID = "MOTNGRFX_DepthPivotTransition"
WIDGETS = ("chunks", "video", "in_angle", "out_angle", "travel", "hold1", "hold2", "depth_ratio", "blur", "subject_card", "short", "seed",
           "size_lo", "size_hi", "spring")      # the public params that are node widgets (type / type_depth / fps stay API-only)


# --------------------------------------------------------------------------------------------- conversions
def image_to_numpy(t) -> np.ndarray | None:
    """IMAGE [B,H,W,C] (or [H,W,C]) float 0..1 -> uint8 RGB (B,H,W,3). None -> None."""
    if t is None:
        return None
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    return DP.as_frames(a, "image")


def mask_to_numpy(t) -> np.ndarray | None:
    """MASK [B,H,W] (or [H,W], [B,1,H,W]) float 0..1 -> float32 (B,H,W). None -> None."""
    if t is None:
        return None
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    if a.ndim == 4:
        a = a[:, 0] if a.shape[1] == 1 else a[..., 0]
    return a[None] if a.ndim == 2 else a.astype(np.float32)


def frames_to_image(frames: np.ndarray) -> torch.Tensor:
    """uint8 RGB (N,H,W,3) -> IMAGE [N,H,W,3] float32 0..1 (exact: x / 255)."""
    return torch.from_numpy(np.ascontiguousarray(frames.astype(np.float32) / 255.0))


def widget_params(**kw) -> dict:
    return {k: kw[k] for k in WIDGETS if k in kw and kw[k] is not None}


def run(image_a, mask_a, image_b, mask_b, depth_a=None, depth_b=None, params_json: str = "", on_frame=None, **widgets):
    """the node body. -> (IMAGE tensor [N,H,W,3], resolved params JSON string). A non-empty `params_json` (a resolved dict from an earlier run)
    overrides the widgets so a result can be re-made exactly."""
    params = widget_params(**widgets)
    if params_json and params_json.strip():
        params = json.loads(params_json)
    if image_a is None or image_b is None:
        raise ValueError("MOTNGRFX_DepthPivotTransition: connect image_a and image_b")
    res = DP.depth_pivot_render(image_to_numpy(image_a), mask_to_numpy(mask_a), image_to_numpy(image_b), mask_to_numpy(mask_b),
                                image_to_numpy(depth_a), image_to_numpy(depth_b), params, on_frame=on_frame)
    return frames_to_image(res.frames), json.dumps(res.params, indent=1, sort_keys=True)


# --------------------------------------------------------------------------------------------- node
try:
    from comfy_api.latest import io
except ImportError:           # no ComfyUI here: helpers only
    io = None

NODES: list = []

if io is not None:
    class MOTNGRFX_DepthPivotTransition(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            P = DP.PARAMS
            return io.Schema(
                node_id=NODE_ID,
                display_name="MOTNGRFX Depth Pivot Transition (chunk cards)",
                category=CATEGORY,
                description=("Depth pivot-chunk transition between two shots: both are depth scenes of chunky camera-facing cards (the subject one "
                             "cut-out card), one camera pivots round A's volume and B swings in over the top and lands flat. A multi-frame "
                             "IMAGE batch is a clip and plays live on the cards (a single image is a still); masks = the per-frame subject "
                             "matte. Depth is estimated on the CPU from the first frame unless depth_a / depth_b are connected. "
                             "params_json out = the full resolved parameters (paste back in to reproduce)."),
                inputs=[
                    io.Image.Input("image_a", tooltip="Shot A: one image = still, a batch = the live clip."),
                    io.Mask.Input("mask_a", optional=True, tooltip="Subject matte(s) of A [B,H,W] (Alpha-Gen output). Empty = no subject card."),
                    io.Image.Input("image_b", tooltip="Shot B: one image = still, a batch = the live clip."),
                    io.Mask.Input("mask_b", optional=True, tooltip="Subject matte(s) of B [B,H,W]."),
                    io.Image.Input("depth_a", optional=True, tooltip="Depth of A (bright = near). Empty = Depth-Anything-V2-Small on the CPU."),
                    io.Image.Input("depth_b", optional=True, tooltip="Depth of B (bright = near). Empty = Depth-Anything-V2-Small on the CPU."),
                    io.Int.Input("chunks", default=P["chunks"]["default"], min=P["chunks"]["lo"], max=P["chunks"]["hi"],
                                 tooltip="Background cards per image (+ the subject as one card). 32 = the approved look; more = smaller cards."),
                    io.Combo.Input("video", options=P["video"]["choices"], default=P["video"]["default"],
                                   tooltip="auto = live clip textures whenever an input batch has more than one frame."),
                    io.Float.Input("in_angle", default=P["in_angle"]["default"], min=P["in_angle"]["lo"], max=P["in_angle"]["hi"], step=1.0,
                                   tooltip="Degrees B is turned when it starts to enter (it lands flat)."),
                    io.Float.Input("out_angle", default=P["out_angle"]["default"], min=P["out_angle"]["lo"], max=P["out_angle"]["hi"], step=1.0,
                                   tooltip="Degrees A has turned when B starts to enter."),
                    io.Int.Input("travel", default=P["travel"]["default"], min=P["travel"]["lo"], max=P["travel"]["hi"], tooltip="Frames of the eased move."),
                    io.Int.Input("hold1", default=P["hold1"]["default"], min=P["hold1"]["lo"], max=P["hold1"]["hi"], tooltip="Frames on A before the move."),
                    io.Int.Input("hold2", default=P["hold2"]["default"], min=P["hold2"]["lo"], max=P["hold2"]["hi"], tooltip="Frames on B after the landing."),
                    io.Float.Input("depth_ratio", default=P["depth_ratio"]["default"], min=P["depth_ratio"]["lo"], max=P["depth_ratio"]["hi"],
                                   step=0.5, tooltip="Far / near depth ratio (more = more parallax)."),
                    io.Combo.Input("blur", options=P["blur"]["choices"], default=P["blur"]["default"], tooltip="Motion blur."),
                    io.Boolean.Input("subject_card", default=bool(P["subject_card"]["default"]), tooltip="The subject is ONE cut-out card."),
                    io.Int.Input("short", default=P["short"]["default"], min=P["short"]["lo"], max=P["short"]["hi"],
                                 tooltip="Short side of the output in px (432 = the approved look; the aspect of image_a is kept)."),
                    io.Int.Input("seed", default=P["seed"]["default"], min=0, max=2 ** 31 - 1, tooltip="Card jitter / size seed."),
                    io.Float.Input("size_lo", default=P["size_lo"]["default"], min=P["size_lo"]["lo"], max=P["size_lo"]["hi"], step=0.1,
                                   tooltip="Advanced: smallest card side as a multiple of the card scale."),
                    io.Float.Input("size_hi", default=P["size_hi"]["default"], min=P["size_hi"]["lo"], max=P["size_hi"]["hi"], step=0.1,
                                   tooltip="Advanced: largest card side as a multiple of the card scale."),
                    io.Boolean.Input("spring", default=bool(P["spring"]["default"]), tooltip="Advanced: damped-spring landing (the approved look has none)."),
                    io.String.Input("params_json", default="", multiline=True, optional=True,
                                    tooltip="Resolved params of an earlier run: overrides the widgets above (reproduce a result exactly)."),
                ],
                outputs=[io.Image.Output(display_name="frames"), io.String.Output(display_name="params_json")],
            )

        @classmethod
        def execute(cls, image_a, image_b, mask_a=None, mask_b=None, depth_a=None, depth_b=None, params_json="", **widgets) -> io.NodeOutput:
            pbar = None
            try:
                import comfy.utils
                n = DP.n_frames(json.loads(params_json) if params_json and params_json.strip() else widget_params(**widgets))
                pbar = comfy.utils.ProgressBar(n)
            except Exception:   # noqa: BLE001  (no comfy / bad json: the run below reports it)
                pass
            frames, pj = run(image_a, mask_a, image_b, mask_b, depth_a, depth_b, params_json,
                             on_frame=(lambda i, n: pbar.update_absolute(i, n)) if pbar else None, **widgets)
            return io.NodeOutput(frames, pj)

    NODES = [MOTNGRFX_DepthPivotTransition]

    try:
        from ._types import MAX_SHOTS, MographClips, MographPiece
    except ImportError:           # loaded outside the package (tests): the shot-cut node needs the pack's types
        MographClips = None

    if MographClips is not None:
        from tools.fusion import pivot_cut as PC          # noqa: E402  (ensure_root_on_path() ran at the top of the file)

        def _wired(cls, name: str) -> bool:
            hid = getattr(cls, "hidden", None)
            prompt = getattr(hid, "prompt", None)
            return True if not prompt else isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)

        class MographPivotCut(io.ComfyNode):
            """The 3D depth transitions as a SHOT CUT (tools/fusion/pivot_cut.py): piece.compose.cuts[] entries with cut == 'depth_pivot' | 'orbit:push|pullback|fly' |
            'depth_dolly:down|up|left|right' {"after": sid, "params": {..}} get their frames spliced into the two shots' clips (A's tail, B's head) and, with matte_pass
            (default), the subject matte through the transition rides on the clips (clip['matte_patch']); no such cut = clips pass through.
            Place it between the footage (flow) node and MographCompose; the assembler already makes the boundary a hard cut (window_cut.owned_boundaries).
            CPU render (BUILT.md B2: ~170 s per 72 frames at 432 px on a laptop CPU; measured numbers in 16_app/TRANSITIONS.md)."""

            @classmethod
            def define_schema(cls):
                return io.Schema(
                    node_id="MographPivotCut", display_name="MOTNGRFX Pivot Cut (depth chunks)", category=CATEGORY,
                    description=("Splices the 3D depth transition (pivot chunks, orbit push / pullback / fly, depth dolly down / up / left / right) into the clips at "
                                 "the piece's compose.cuts[] entries of those kinds, plus the subject matte through it. matte_k / depth_k of the two shots are only "
                                 "requested when such a cut names them. No such cut = pass-through."),
                    inputs=[MographPiece.Input("piece"), MographClips.Input("clips"),
                            *[io.Image.Input(f"matte_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
                            *[io.Image.Input(f"depth_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)]],
                    outputs=[MographClips.Output(display_name="clips")],
                    hidden=[io.Hidden.prompt, io.Hidden.unique_id])

            @classmethod
            def check_lazy_status(cls, piece=None, clips=None, **imgs) -> list[str]:
                need = []
                for sp in (PC.specs(piece) if piece and not (piece.get("request") or {}).get("plates_only") else []):
                    for sid in (sp["a"], sp["b"]):
                        for kind in ("matte", "depth"):
                            name = f"{kind}_{sid[1:]}"
                            if imgs.get(name) is None and _wired(cls, name) and name not in need:
                                need.append(name)
                return need

            @classmethod
            def execute(cls, piece, clips, **imgs) -> io.NodeOutput:
                sps = PC.specs(piece)
                if not sps or ((piece or {}).get("request") or {}).get("plates_only"):      # plates only (image board edited like a video): the plain footage; a 3D cut carries the design's type into it
                    return io.NodeOutput(clips)
                by = {c["sid"]: dict(c) for c in clips}
                for sp in sps:
                    if sp["a"] not in by or sp["b"] not in by:
                        continue
                    pl = PC.plan_cut(piece, sp)
                    if pl["problems"]:
                        print(f"[MographPivotCut] {sp['a']}->{sp['b']}: skipped, plain hard cut ({'; '.join(pl['problems'])})")
                        continue
                    arr = {s: by[s].get("subject", by[s]["u8"]) for s in (sp["a"], sp["b"])}      # three-layer shots: the CLEAN clip is the source, and gets the frames too
                    from tools.fusion import matte_align as MA         # the mattes / depth are cover-resized crops of the clips (footage._u8_to_ref): put them on the clip's own frame first
                    fhw = {s: tuple(int(x) for x in arr[s].shape[1:3]) for s in arr}

                    def on_clip(t, s, depth=False):
                        if t is None:
                            return None
                        a_ = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)
                        a_ = a_[..., 0] if (a_.ndim == 4 and not depth) else a_
                        return np.stack([MA.ref_to_full(f_, fhw[s]) for f_ in a_]).astype(np.float32)
                    mt = {s: on_clip(imgs.get(f"matte_{s[1:]}"), s) for s in arr}
                    for s in arr:                                      # an earlier cut already replaced frames of this shot: its matte follows them, so this cut's matte pass starts from the matte of the picture it really gets
                        if mt[s] is not None:
                            mt[s] = PC.patch_matte_all(mt[s], PC.clip_patches(by[s]))
                    ph = {s: PC.is_placeholder_depth(imgs.get(f"depth_{s[1:]}")) for s in arr}      # the footage nodes' 16x16 zero placeholder (no real depth reached this cut): count it as None so the CPU Depth-Anything fallback runs
                    if any(ph.values()):
                        print(f"[MographPivotCut] WARNING {sp['a']}->{sp['b']}: depth of {[s for s in arr if ph[s]]} is a placeholder (< {PC.MIN_DEPTH_SIDE} px), using the CPU depth fallback")
                    dp = {s: (None if ph[s] else on_clip(imgs.get(f"depth_{s[1:]}"), s, depth=True)) for s in arr}
                    plates = {s: by[s]["u8"] for s in arr if by[s].get("plate") is True}      # a shot with a clean plate moves as its plate, its subject glitches in after the move (pivot_cut._plain_render)
                    res = PC.apply_cut(piece, sp, arr, mattes=mt, depths=dp, plates=plates)
                    PC.label_depth(res.info, ph[sp["a"]], ph[sp["b"]])        # cut JSON: "given" | "fallback" (placeholder in, CPU estimate ran) | "placeholder"
                    wins = {sp["a"]: (pl["a0"], pl["a0"] + pl["i2"]), sp["b"]: (0, pl["b_len"])}
                    for s, new, mp in ((sp["a"], res.a, res.matte_a), (sp["b"], res.b, res.matte_b)):
                        t = torch.from_numpy(np.ascontiguousarray(new))
                        t = t.to(arr[s].device) if hasattr(arr[s], "device") else t
                        if "subject" in by[s]:            # three-layer shot: the TEXT clip keeps every frame outside the window (an LTX word that ends before the cut survives); inside it both
                            lo, hi = wins[s]               # layers are the clean transition frames, so no letters are ever baked into the cut's pixels
                            tx = torch.from_numpy(np.ascontiguousarray(PC.splice_window(by[s]["u8"], new, lo, hi)))
                            by[s]["u8"] = tx.to(arr[s].device) if hasattr(arr[s], "device") else tx
                            by[s]["subject"] = t
                        else:
                            by[s]["u8"] = t
                        if mp is not None:                       # the subject matte THROUGH the transition rides on the clip; MographCompose writes it into the shot's matte
                            PC.add_matte_patch(by[s], mp)        # one per cut: a shot between two cuts is the B of one and the A of the next, neither may overwrite the other
                    # the rendered cut for MographArchive (ingredients: <stem>_cut_<a>-<b>_<kind>.mp4 + matte PNGs): rides on A's clip, not re-rendered or copied
                    by[sp["a"]]["cut_renders"] = list(by[sp["a"]].get("cut_renders") or []) + [PC.cut_record(res)]
                    print(f"[MographPivotCut] {pl['kind']} ({pl['params'].get('style', 'shards')}) {sp['a']}->{sp['b']}: {pl['n']} frames, B from {pl['i2']}, "
                          f"window A {pl['window_a']} B {pl['window_b']}, matte {'through the transition' if res.matte_a is not None else 'unchanged'}")
                    for tc in PC.type_clearance(piece, pl):
                        if tc["state"] in ("conflict", "exits_in"):
                            print(f"[MographPivotCut]   word {tc['word']} {tc['text']!r}: {tc['state']} ({tc['render']}, live {tc['live']}, its exit starts at {tc['exit_from']} s)")
                return io.NodeOutput([by[c["sid"]] for c in clips])

        NODES.append(MographPivotCut)
