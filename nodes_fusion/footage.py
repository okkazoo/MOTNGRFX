"""Shot slots, clip decode and the footage gather barrier (DESIGN.md sections 1, 3, 5).

    MographShotSlot (index k)   piece, type, first (one image, or one row per place), prev (lazy) -> prompt, seed, frames, control, first, no_first, guide
    MographDecodeClip           vae, samples, piece, index, gray -> MOGRAPH_CLIP (+ float IMAGE for gray/alpha chains)
    MographFootageGather        piece, request, first?, clip_1..6 (lazy) -> MOGRAPH_CLIPS, alpha_ref_1..6, depth_ref_1..6
    MographFootageFlow          piece, clips -> MOGRAPH_CLIPS with the footage optical flow attached (fp16, half-res) for flow_wake

STILL PREVIEW (piece.still_preview, tools/fusion/still_preview.py): no clip is requested; MographFootageGather holds each shot's OWN still (a row of ``first``) for the shot length, the Alpha-Gen
reference is a 9-frame clip of that still, the depth reference is the still (1 frame). There is no other stand-in footage: a shot whose clip did not arrive is an error.

Lazy gating (spike results in 16_app/DESIGN.md): slots / clips of shots that are not in ``piece['layout']`` are never
requested, so their LTX chains never execute. ``check_lazy_status`` only ever requests inputs that are linked in this
node's own prompt entry (requesting an unlinked lazy input raises in ComfyUI's executor).

Piece fields read here (the P*.json schema): ``slug``, ``layout{sid:{start,frames,xf}}``,
``shots[{id, prompt, seed, frames, first, control{pass, guide}}]``, ``sizes.compose``, ``quality``.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import (CATEGORY, FPS, MAX_SHOTS, Clip, MographClip, MographClips, MographPiece, MographRequest,
                     MographType, active_sids, new_clip, sid_of)
from ._util import cover_resize

ensure_root_on_path()
from tools.fusion import footage_flow as FF      # noqa: E402
from tools.fusion import owner as OW             # noqa: E402
from tools.fusion import places as PL            # noqa: E402
from tools.fusion import plan_rules as PR        # noqa: E402
from tools.fusion import recipe_eval as RE       # noqa: E402
from tools.fusion import still_preview as SP     # noqa: E402

# Sizes are (w, h) PORTRAIT defaults; the piece's format (9:16 | 16:9) swaps the sides (see ``size_of``).
CONTROL_SIZE = (288, 512)        # control reference handed to the IC-LoRA graph (= sizes.control / 2)
DECODE_SIZE = (1080, 1920)       # uint8 clips after DecodeClip (= sizes.compose; x0.9375 of the 1152x2048 render)
ALPHA_REF_SIZE = (576, 1024)     # = sizes.control
DEPTH_REF_SIZE = (540, 960)
DRAFT_COMPOSE = (540, 960)


# --------------------------------------------------------------------------------------------- helpers
def size_of(piece: dict, key: str, default: tuple[int, int], div: int = 1) -> tuple[int, int]:
    """(w, h) of ``piece['sizes'][key]`` (divided by ``div``), else ``default`` turned into the piece's format."""
    sz = (piece.get("sizes") or {}).get(key)
    w, h = (int(sz[0]), int(sz[1])) if sz else PR.orient(default, PR.format_of(piece))
    return (w // div, h // div)


def shot_of(piece: dict, sid: str) -> dict:
    for s in piece.get("shots") or []:
        if s.get("id") == sid:
            return s
    raise KeyError(f"piece has no shot {sid!r}")


def _wired(cls, name: str) -> bool:
    """Is input ``name`` linked in this node's own prompt entry? (no prompt available -> assume yes)."""
    hid = getattr(cls, "hidden", None)
    prompt = getattr(hid, "prompt", None)
    if not prompt:
        return True
    return isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)


def _is_still(piece: dict, request: dict | None = None) -> bool:
    """STILL PREVIEW (app quality `still`, tools/fusion/still_preview.py): the footage is each shot's own still, no LTX clip is requested. Quality `draft` / `preview` / `final` never is."""
    return SP.is_still(piece, request)


def _imread_gray_or_rgb(path: str, size: tuple[int, int]) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"cannot read {path}")
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    im = im.astype(np.float32) / scale
    im = np.repeat(im[..., None], 3, -1) if im.ndim == 2 else im[..., :3][..., ::-1]
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA)


def read_pass_frames(pass_dir: str, name: str, start: int, n: int, size: tuple[int, int]) -> np.ndarray:
    """``pass_dir/<name>/<name>_%05d.png`` frames start..start+n-1 (clamped to the last frame, like the runner's
    ``tpad=stop_mode=clone``) -> float32 [n,h,w,3] resized to ``size`` (w, h)."""
    d = os.path.join(pass_dir, name)
    avail = sorted(f for f in os.listdir(d) if f.lower().endswith(".png")) if os.path.isdir(d) else []
    if not avail:
        raise FileNotFoundError(f"type pass {name!r} has no frames in {d}")
    paths = [os.path.join(d, f"{name}_{min(start + i, len(avail) - 1):05d}.png") for i in range(n)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        frames = list(ex.map(lambda p: _imread_gray_or_rgb(p, size), paths))
    return np.stack(frames)


def read_control_frames(pass_dir: str, name: str, start: int, n: int, size: tuple[int, int]) -> np.ndarray:
    """the control frames of a shot: a type pass by name, or `union` = the depth pass with the canny line pass drawn over it (per-pixel max). Thin / dotted / outline faces carry no depth
    worth guiding, so the planner picks `line` for them, `union` for mid-weight faces and `depth` for heavy ones (owner.control_pass_for). Falls back on the one pass that exists."""
    if name != "union":
        return read_pass_frames(pass_dir, name, start, n, size)
    have = [p for p in ("depth", "line") if os.path.isdir(os.path.join(pass_dir, p))]
    if len(have) < 2:
        return read_pass_frames(pass_dir, have[0] if have else "depth", start, n, size)
    return np.maximum(read_pass_frames(pass_dir, "depth", start, n, size), read_pass_frames(pass_dir, "line", start, n, size))


def _u8_to_ref(u8: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float32 torch IMAGE [T,h,w,3] cover-fit to ``size`` (w, h)."""
    w, h = size
    out = np.empty((u8.shape[0], h, w, 3), np.float32)
    for i in range(u8.shape[0]):
        fr = u8[i] if u8[i].shape[:2] == (h, w) else cover_resize(u8[i], w, h)
        out[i] = fr.astype(np.float32) / 255.0
    return torch.from_numpy(out)


def decode_tiles(piece: dict) -> tuple[int, int, int, int]:
    """(tile_size, overlap, temporal_size, temporal_overlap) of the tiled VAE decode. The 24 GB / 5090 values (512, 64, 128, 32) decode 128 frames at
    once; the preview tier runs on a 12 GB card, so it decodes 64-frame chunks (8 overlap): the numbers of the standalone IC graph's own
    VAEDecodeTiled node (tools/fusion/ltx_ic.py, single stage). Half the frames per decoder pass, so about half the activation peak (expected,
    NOT measured on a 12 GB card yet); the 8-frame overlap is blended by ComfyUI."""
    if (piece or {}).get("quality") == "preview":
        return 512, 64, 64, 8
    return 512, 64, 128, 32


def hold_first(images: torch.Tensor, n: int) -> torch.Tensor:
    """frame 0 of an IMAGE batch repeated to `n` frames as a view (stride 0: no copy)."""
    return images[:1].expand(max(int(n), 1), -1, -1, -1)


def _placeholder() -> torch.Tensor:
    return torch.zeros((1, 16, 16, 3), dtype=torch.float32)


# --------------------------------------------------------------------------------------------- nodes
class MographShotSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographShotSlot",
            display_name="MOTNGRFX Shot Slot",
            category=CATEGORY,
            description="Everything one IC-LoRA shot needs, taken from the piece: prompt, seed, length, the control "
                        "pass slice (288x512) and the first frame (the still for shot 1, else the previous clip's last "
                        "frame minus the crossfade). Only runs when its shot is in the piece layout.",
            inputs=[
                io.Int.Input("index", default=1, min=1, max=MAX_SHOTS, tooltip="Shot number 1..6 (s1..s6)."),
                MographPiece.Input("piece"),
                MographType.Input("type"),
                io.Image.Input("first", tooltip="First frame of the piece (MographFirstFrame), or the batch of one first frame per place (MographFirstFrames): "
                                                "a shot that starts a new place takes the row of its still (`first` = A, B ...)."),
                MographClip.Input("prev", optional=True, lazy=True,
                                  tooltip="Previous shot's clip; only evaluated when this shot starts from 'last:sK'."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"), io.Int.Output(display_name="seed"),
                io.Int.Output(display_name="frames"), io.Image.Output(display_name="control"),
                io.Image.Output(display_name="first"), io.Boolean.Output(display_name="no_first"),
                io.Float.Output(display_name="guide"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _first_ref(cls, piece: dict, index: int) -> str | None:
        return shot_of(piece, sid_of(index)).get("first")

    @classmethod
    def check_lazy_status(cls, index=None, piece=None, type=None, first=None, prev=None) -> list[str]:
        if prev is not None or piece is None or index is None or index < 2:
            return []
        try:
            ref = cls._first_ref(piece, int(index))
        except KeyError:
            return []
        if isinstance(ref, str) and ref.startswith("last:") and _wired(cls, "prev"):
            return ["prev"]
        return []

    @classmethod
    def execute(cls, index: int, piece: dict, type: dict, first, prev=None) -> io.NodeOutput:
        sid = sid_of(index)
        lay = (piece.get("layout") or {}).get(sid)
        if lay is None:
            raise ValueError(f"MographShotSlot {sid}: not in the piece layout ({', '.join(active_sids(piece))}); "
                             f"this slot should not have been requested")
        shot = shot_of(piece, sid)
        frames = int(shot.get("frames") or lay["frames"])
        ctl = shot.get("control") or {}
        from tools.fusion import recipe_eval as _RE                      # a control pass written as "depth+line" / "all" is read tolerantly (never a letter-split crash)
        control = read_control_frames(type["dir_control"], _RE.norm_control_pass(ctl.get("pass", "depth")), int(lay["start"]), frames,
                                   size_of(piece, "control", (576, 1024), 2))
        guide = float(ctl.get("guide", ctl.get("strength", 1.0)))
        prompt = str(shot.get("prompt", ""))
        from tools.app import layered_shots as _LS                       # the user's typed palette never reaches the prompt of a shot that starts from the user's picture (clean AND text-pass chains read this prompt)
        prompt = _LS.scrub_prompt(prompt, piece, sid)
        # ONE OWNER PER WORD (tools/fusion/owner.py): the control pass holds only the words LTX owns (render ltx = whole span, handoff = from ltx_at, flat = never),
        # a shot with no LTX-owned word gets guide 0 (plain i2v) and an LTX-owned word is named in the prompt (the words "X" formed as <material> letters).
        # A piece without owner fields (old pieces) is untouched.
        gains, guide, prompt = OW.slot_plan(piece, (type or {}).get("manifest"), int(lay["start"]), frames, int(shot.get("span") or lay["frames"]), prompt, guide)
        if gains is not None:
            control = control * gains[:, None, None, None]

        ref = shot.get("first")
        no_first = ref is None
        first_out = first
        if first is not None and int(first.shape[0]) > 1 and not (isinstance(ref, str) and ref.startswith("last:")):
            row = PL.pick_row(piece, ref, int(first.shape[0]))        # one first frame per place: this shot's own still (a new place), else row 0
            first_out = first[row:row + 1]
        if isinstance(ref, str) and ref.startswith("last:"):
            if prev is None:
                raise ValueError(f"MographShotSlot {sid}: first frame is {ref!r} but no previous clip arrived")
            u8 = prev["u8"]
            idx = max(int(u8.shape[0]) - 1 - int(lay.get("xf", 0)), 0)     # frame on screen when the crossfade starts
            first_out = (u8[idx].float() / 255.0)[None]
        return io.NodeOutput(prompt, int(shot.get("seed", 1)), frames,
                             torch.from_numpy(control), first_out, bool(no_first), guide)


class MographDecodeClip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographDecodeClip",
            display_name="MOTNGRFX Decode Clip",
            category=CATEGORY,
            description="Tiled VAE decode of a shot, resized to 1080x1920 and kept as uint8 on the CPU (clip). With "
                        "gray (alpha chain) the frames keep their native size and also come out as a float IMAGE.",
            inputs=[io.Vae.Input("vae"), io.Latent.Input("samples"), MographPiece.Input("piece"),
                    io.Int.Input("index", default=1, min=1, max=MAX_SHOTS),
                    io.Boolean.Input("gray", default=False, tooltip="Alpha/matte chain: native size, also output as IMAGE.")],
            outputs=[MographClip.Output(display_name="clip"), io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, vae, samples, piece: dict, index: int, gray: bool) -> io.NodeOutput:
        import nodes as comfy_nodes                      # ComfyUI core (already in sys.modules inside the server)
        import comfy.model_management as mm
        sid = sid_of(index)
        start = int(((piece.get("layout") or {}).get(sid) or {}).get("start", 0))
        dw, dh = size_of(piece, "compose", DECODE_SIZE) if not gray else DECODE_SIZE
        images = comfy_nodes.VAEDecodeTiled().decode(vae, samples, *decode_tiles(piece))[0]    # float [T,H,W,3] 0..1
        dev = mm.get_torch_device()
        u8 = torch.empty((images.shape[0], dh, dw, 3) if not gray else tuple(images.shape),
                         dtype=torch.uint8)
        for i in range(0, images.shape[0], 16):
            chunk = images[i:i + 16].to(dev).float()
            if not gray:
                chunk = F.interpolate(chunk.permute(0, 3, 1, 2), size=(dh, dw),
                                      mode="bilinear", antialias=True, align_corners=False).permute(0, 2, 3, 1)
            u8[i:i + 16] = chunk.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        if gray and _is_still(piece):                        # still preview: the 9-frame Alpha-Gen run of ONE picture -> its first frame repeated over the shot (no copy: a view)
            images = hold_first(images, SP.hold_len(piece, sid, int(images.shape[0])))
        return io.NodeOutput(new_clip(u8, sid, start, FPS), images if gray else _placeholder())


def still_outputs(piece: dict, first) -> tuple[list, list, list]:
    """STILL PREVIEW footage: per shot of the layout the shot's OWN still (row of the first-frame batch `first` that its `first` names, tools/fusion/places.pick_row) held for the shot length
    at the compose size, plus the references of the matte and depth chains cut from that same still: the Alpha-Gen reference is ``SP.ALPHA_FRAMES`` (9) copies of it at the control size (a
    9-frame Alpha-Gen run instead of one over the whole shot), the depth reference ONE frame at the depth size. -> (clips, alpha refs, depth refs), one entry per slot like the gather outputs."""
    if first is None or int(first.shape[0]) < 1:
        raise ValueError("still preview: the stills (MographFirstFrames / MographFirstFrame) did not arrive")
    size = size_of(piece, "compose", DRAFT_COMPOSE)
    alpha_size = size_of(piece, "control", ALPHA_REF_SIZE)
    depth_size = tuple(PR.orient(DEPTH_REF_SIZE, PR.format_of(piece)))
    lay = piece.get("layout") or {}
    clips, alpha, depth = [], [], []
    for k in range(1, MAX_SHOTS + 1):
        sid = sid_of(k)
        if sid not in lay:
            alpha.append(_placeholder())
            depth.append(_placeholder())
            continue
        row = SP.shot_row(piece, sid, int(first.shape[0]))
        img = first[row].detach().cpu().float().numpy()[..., :3]
        u8 = SP.hold_u8(img, int(lay[sid]["frames"]), size)
        c = new_clip(torch.from_numpy(u8), sid, int(lay[sid]["start"]), FPS)
        c["still"] = True                                         # a held picture: MographFootageFlow gives it zero flow
        clips.append(c)
        need = RE.shot_needs(piece, sid)
        one = u8[:1]
        alpha.append(_u8_to_ref(np.repeat(one, SP.ALPHA_FRAMES, axis=0), alpha_size) if need["matte"] else _placeholder())
        depth.append(_u8_to_ref(one, depth_size) if need["depth"] else _placeholder())
    return clips, alpha, depth


class MographFootageGather(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographFootageGather",
            display_name="MOTNGRFX Footage Gather",
            category=CATEGORY,
            description="Barrier: waits for the clips of the ACTIVE shots only, then hands them on and prepares the "
                        "alpha (stage-1 size) and depth (half compose size) reference videos, only for shots whose recipe "
                        "reads a matte / depth (the others get a 16x16 placeholder). Still preview: no clip is requested; each "
                        "shot's own still (a row of `first`) is held for the shot length and the references are cut from it "
                        "(9 frames for Alpha-Gen, 1 for depth).",
            inputs=[MographPiece.Input("piece"), MographRequest.Input("request"),
                    io.Image.Input("first", optional=True, tooltip="The stills, one row per place (MographFirstFrames): the footage of a still preview."),
                    *[MographClip.Input(f"clip_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)]],
            outputs=[MographClips.Output(display_name="clips"),
                     *[io.Image.Output(display_name=f"alpha_ref_{k}") for k in range(1, MAX_SHOTS + 1)],
                     *[io.Image.Output(display_name=f"depth_ref_{k}") for k in range(1, MAX_SHOTS + 1)]],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, request=None, first=None, **clips) -> list[str]:
        if piece is None or _is_still(piece, request):
            return []                                            # still preview: the LTX chains stay idle
        sids = set(active_sids(piece))
        return [name for name, v in clips.items()
                if v is None and name.replace("clip_", "s") in sids and _wired(cls, name)]

    @classmethod
    def execute(cls, piece: dict, request: dict, first=None, **clips) -> io.NodeOutput:
        if _is_still(piece, request):
            out_clips, alpha, depth = still_outputs(piece, first)
            return io.NodeOutput(out_clips, *alpha, *depth)
        alpha_size = size_of(piece, "control", ALPHA_REF_SIZE)
        depth_size = tuple(PR.orient(DEPTH_REF_SIZE, PR.format_of(piece)))
        lay = piece.get("layout") or {}
        out_clips: list[Clip] = []
        alpha, depth = [], []
        for k in range(1, MAX_SHOTS + 1):
            sid = sid_of(k)
            if sid not in lay:
                alpha.append(_placeholder())
                depth.append(_placeholder())
                continue
            c = clips.get(f"clip_{k}")
            if c is None:
                raise ValueError(f"MographFootageGather {sid}: no clip arrived (there is no stand-in footage; the still preview is the quality `still`)")
            out_clips.append(c)
            need = RE.shot_needs(piece, sid)             # float32 refs only for the chains the shot's recipe reads
            a = c["u8"].numpy()
            alpha.append(_u8_to_ref(a, alpha_size) if need["matte"] else _placeholder())
            depth.append(_u8_to_ref(a, depth_size) if need["depth"] else _placeholder())
        return io.NodeOutput(out_clips, *alpha, *depth)


def clip_flow(u8, method: str = "auto", model: str = "raft_large", batch: int = 4) -> np.ndarray:
    """uint8 clip [T,H,W,3] -> float16 numpy flow [T,H/2,W/2,2] (px/frame of the half-res grid). RAFT on CUDA, DIS on the CPU; a
    failing RAFT (no weights, out of memory) falls back to DIS. Frames stay uint8, the result fp16: no float32 full-res stack."""
    try:
        return FF.flow_for(u8, method=method, model=model, batch=batch, verbose=True)
    except Exception as e:   # noqa: BLE001
        if method == "dis":
            raise
        print(f"[MographFootageFlow] RAFT failed ({type(e).__name__}: {e}); falling back to DIS on the CPU")
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:   # noqa: BLE001
            pass
        return FF.flow_for(u8, device="cpu", method="dis", verbose=True)


class MographFootageFlow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographFootageFlow",
            display_name="MOTNGRFX Footage Flow",
            category=CATEGORY,
            description="Optical flow of every clip: the motion in the footage that pushes, drags and streaks the type "
                        "(flow_wake in MOTNGRFX Compose; part of every generation). RAFT on the GPU, OpenCV DIS on the CPU "
                        "(draft / local). Frames stay uint8, the flow is float16 at half resolution and rides on the clip.",
            inputs=[
                MographPiece.Input("piece"),
                MographClips.Input("clips"),
                io.Combo.Input("method", options=["auto", "raft", "dis"], default="auto",
                               tooltip="auto = RAFT on CUDA, DIS otherwise."),
                io.Combo.Input("model", options=["raft_large", "raft_small"], default="raft_large"),
            ],
            outputs=[MographClips.Output(display_name="clips")],
        )

    @classmethod
    def execute(cls, piece: dict, clips: list, method: str = "auto", model: str = "raft_large") -> io.NodeOutput:
        out = []
        t0 = time.time()
        for c in clips:
            if RE.shot_needs_flow(piece, c["sid"]) and c.get("flow") is None:
                if c.get("still"):                                   # a held still does not move: zero flow, no RAFT / DIS over hundreds of identical frames
                    c = dict(c, flow=torch.zeros(SP.zero_flow_shape(tuple(c["u8"].shape)), dtype=torch.float16))
                else:
                    fl = clip_flow(c["u8"], method, model)
                    c = dict(c, flow=torch.from_numpy(fl))
            out.append(c)
        n = sum(1 for c in out if c.get("flow") is not None)
        print(f"[MographFootageFlow] {n}/{len(out)} clips in {time.time() - t0:.1f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(out)


NODES = [MographShotSlot, MographDecodeClip, MographFootageGather, MographFootageFlow]
