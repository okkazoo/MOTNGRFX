"""MographCompose and MographAssemble (DESIGN.md sections 1, 4, 5).

    MographCompose   piece, type, clips, music?, matte_1..6 / depth_1..6 (lazy) -> MOGRAPH_CLIPS (composed), hero IMAGE
    MographAssemble  composed, piece, audio?, crf, encoder auto|nvenc|x264      -> VIDEO

Compose runs ``tools/fusion/recipe_eval.run_compose``: the shot's op chain through the ops' ``.raw`` variants on the compose torch
backend (tensors stay on CUDA; numpy backend on CPU-only machines), one uint8 download per frame. Assemble reuses
``compose.assemble`` (centred crossfades, cut = start + xf/2) over in-memory clips and ``compose.encode_mp4`` with libx264 or
h264_nvenc. Draft (``piece['quality'] == 'draft'``): compose size comes from ``piece['sizes']['compose']`` (540x960), x264 veryfast crf 20.
"""
from __future__ import annotations

import functools
import subprocess
import time
import uuid
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import InputImpl, io

try:
    import folder_paths
except Exception:   # pragma: no cover
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY, FPS, MAX_SHOTS, MographClip, MographClips, MographMusic, MographPiece, MographType, new_clip
from .type import wait_type

ensure_root_on_path()
from tools.fusion import compose as fc            # noqa: E402
from tools.fusion import ltx_check as LC          # noqa: E402
from tools.fusion import recipe_eval as RE        # noqa: E402
from tools.fusion import window_cut               # noqa: E402

BACKENDS = ["auto", "torch", "numpy"]
ENCODERS = ["auto", "nvenc", "x264"]


# --------------------------------------------------------------------------------------------- helpers
def _wired(cls, name: str) -> bool:
    """is input ``name`` linked in this node's prompt entry? (requesting an unlinked lazy input raises in the executor)"""
    hid = getattr(cls, "hidden", None)
    prompt = getattr(hid, "prompt", None)
    if not prompt:
        return True
    return isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)


def _out_dir() -> Path:
    base = Path(folder_paths.get_temp_directory()) if folder_paths is not None else Path(__import__("tempfile").gettempdir())
    d = base / "mograph"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cut_seconds(piece: dict, music: dict | None) -> float | None:
    if music and music.get("cut"):
        return float(music["cut"])
    cut = (piece.get("audio") or {}).get("cut")
    return float(cut) if cut else None


def _progress(total: int):
    try:
        from comfy.utils import ProgressBar
        pb = ProgressBar(total)
        state = {"done": 0}

        def step(k, n):          # k of n frames of the current shot
            pb.update_absolute(min(state["done"] + k, total), total)
            if k == n:
                state["done"] += n
        return step
    except Exception:   # noqa: BLE001
        return None


def pick_hero(piece: dict, type_dir: str, composed: list[dict], samples: int = 16) -> np.ndarray:
    """the composed frame (HxWx3 float32) where the type matte is biggest, sampled over the piece; mid-piece without a matte."""
    lay = {c["sid"]: c for c in composed}
    total = max(int(c["start"]) + int(c["u8"].shape[0]) for c in composed)
    best, best_f = -1.0, total // 2
    mdir = Path(type_dir) / "matte" if type_dir else None
    if mdir is not None and mdir.is_dir():
        for f in np.unique(np.linspace(0, total - 1, samples).round().astype(int)):
            im = cv2.imdecode(np.fromfile(str(mdir / f"matte_{int(f):05d}.png"), np.uint8), cv2.IMREAD_GRAYSCALE) \
                if (mdir / f"matte_{int(f):05d}.png").exists() else None
            if im is not None and float(im.mean()) > best:
                best, best_f = float(im.mean()), int(f)
    for c in sorted(lay.values(), key=lambda c: c["start"]):
        j = best_f - int(c["start"])
        if 0 <= j < c["u8"].shape[0]:
            return c["u8"][j].cpu().numpy().astype(np.float32) / 255.0
    c = composed[0]
    return c["u8"][c["u8"].shape[0] // 2].cpu().numpy().astype(np.float32) / 255.0


def prepare_backgrounds(piece: dict, type_t: dict, clips: list[dict], mattes: dict | None = None) -> list[dict]:
    """three-layer shots with a text-rendered clip: (1) did LTX really form the owned words' letters (VLM reading / pixel test)? no -> the word stays flat for its whole span;
    (2) tools/fusion/ltx_check.choose_background: a shot with no LTX letters left, or whose text-rendered subject sits off the clean clip the matte was cut from, gets the CLEAN clip as
    its background (one picture for the type, the subject and the matte). Changes `piece` in place (logged in plan.owner). -> the clips to compose."""
    out = LC.prepare_backgrounds(piece, type_t["dir_layers"], clips, mattes)         # tools/fusion/ltx_check.py: pass 1 judges every word, pass 2 picks the picture under the type
    from tools.fusion import owner as OW
    for m_ in OW.check_one_text(piece, {c["sid"]: bool(c.get("textbg")) for c in out if c.get("subject") is not None}):      # ONE text at any moment (owner.check_one_text)
        print(f"[MographCompose] WARNING one text: {m_}")
    return out


def bake_plate_clips(piece: dict, clips: list[dict], mattes: dict | None, depths: dict | None, type_dir: str | None = None) -> tuple[list[dict], dict, dict, dict]:
    """CLEAN PLATE route (tools/fusion/plate_layers.py, 16_app/CLEAN_PLATE.md): a clip with `plate` set (MographLayerGather: u8 = the clean plate, subject = the clean clip) gets
    its independent motion BAKED before the recipe runs: u8 = the plate moved by `plate_pan`, subject = the clean clip colour-matched to the plate and moved by `subject_move`,
    and its matte (the Alpha-Gen matte of the clean clip, now at the clip's geometry, feathered) moved with the SAME transform as the subject. The recipes then draw the type
    between the two layers as for any layered shot. -> (clips, mattes, depths, {sid: info}); every other clip and matte comes back as it was."""
    mattes, depths, info, out = dict(mattes or {}), dict(depths or {}), {}, []
    for clip in clips:
        if not clip.get("plate") or clip.get("subject") is None or clip.get("plate_baked"):
            out.append(clip)
            continue
        from tools.fusion import plate_layers as PLT
        sid = clip["sid"]
        c2, m2, d2, inf = PLT.bake_clip(piece, clip, mattes.get(sid), depths.get(sid), type_dir)
        if inf.get("skipped"):
            print(f"[MographCompose] {sid} plate route skipped: {inf['skipped']}")
        else:
            print(f"[MographCompose] {sid} plate route: subject {inf['first'][:3]} -> {inf['last'][:3]} (dx px, dy px, scale), type_matte {inf['type_matte']}, plate {inf['first'][3:]} -> {inf['last'][3:]}, "
                  f"colour gain {inf['color']['gain']} offset {inf['color']['offset']} ({'fitted' if inf['color']['ok'] else 'identity'})")
            mattes[sid], depths[sid] = m2, d2
            info[sid] = inf
        out.append(c2)
    return out, mattes, depths, info


def compose_clips(piece: dict, type_t: dict, clips: list[dict], *, music: dict | None = None, mattes: dict | None = None,
                  depths: dict | None = None, backend: str = "auto", progress=None) -> tuple[list[dict], list[dict]]:
    """compose every clip of the active shots. -> (composed MOGRAPH_CLIPS, per-shot stats). mattes / depths: {sid: frames}."""
    wait_type(type_t)                                        # the async layers job
    curves = (music or {}).get("curves") or piece.get("curves") or {}
    cut = _cut_seconds(piece, music)
    lay, _ = RE.piece_layout(piece)
    clips = prepare_backgrounds(piece, type_t, clips, mattes)       # BEFORE any shot is composed: window cuts read the neighbours' final backgrounds
    clips, mattes, depths, plate_info = bake_plate_clips(piece, clips, mattes, depths, type_t["dir_layers"])       # plate shots: the moved plate / subject / matte (the neighbours' window cuts read the moved pictures)
    from tools.fusion import pivot_cut as _pivot_cut                 # 3D shot cuts (depth_transition.py MographPivotCut): the matte through EVERY transition of a shot, for the shot itself and for the neighbours' window cuts
    mattes = _pivot_cut.patched_mattes(clips, mattes)            # (the patch is in clip geometry, the matte in its reference geometry: patch_matte maps it)
    by = {c["sid"]: c for c in clips}
    out, stats = [], []
    for clip in sorted(clips, key=lambda c: lay[c["sid"]]["start"]):
        sid = clip["sid"]
        flow = clip.get("flow")                              # MographFootageFlow: fp16 [T,H/2,W/2,2] riding on the clip
        if flow is None and RE.shot_needs_flow(piece, sid):  # graph without the flow node (PieceFromJSON, tests): do it here
            from .footage import clip_flow
            flow = clip_flow(clip["u8"])
        matte_k = (mattes or {}).get(sid)
        u8, st = RE.run_compose(piece, sid, type_dir=type_t["dir_layers"], footage=clip["u8"], matte=matte_k,
                                depth=(depths or {}).get(sid), curves=curves, cut=cut, backend=backend, progress=progress,
                                flow=flow,           # flow_wake warms its displacement state up over 16 frames (compose_frames)
                                others=window_cut.collect_others(piece, sid, lambda s_: by[s_]["u8"], lambda s_: (mattes or {}).get(s_),
                                                                 subject_of=(lambda s_: by[s_].get("subject", by[s_]["u8"])) if clip.get("subject") is not None else None),
                                subject=clip.get("subject"))     # three-layer shots (nodes_fusion/layers.py): the clean clip = the subject pixels; None = single clip
        out.append(new_clip(u8, sid, lay[sid]["start"], clip.get("fps", FPS)))
        if clip.get("plate_layers") is not None:             # the plate route's components ride on the composed clip for MographArchive (plate_layers.bake_clip)
            out[-1]["plate_layers"] = clip["plate_layers"]
        stats.append(dict(st, sid=sid, **({"plate": plate_info[sid]} if sid in plate_info else {})))
    return out, stats


# --------------------------------------------------------------------------------------------- compose node
class MographCompose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographCompose",
            display_name="MOTNGRFX Compose",
            category=CATEGORY,
            description="Weaves the type layers into each shot's footage with the compose op chain of the piece "
                        "(depth occlusion, tint, glow, grade, grain, vignette, signoff). Ops run on the GPU via their "
                        "raw variants; the footage flow (MOTNGRFX Footage Flow, rides on the clips) drives flow_wake, the "
                        "footage motion that pushes the letters; matte_k / depth_k are only requested when the shot's recipe reads them. "
                        "Recipes with {curve} arguments need the music input.",
            inputs=[
                MographPiece.Input("piece"),
                MographType.Input("type"),
                MographClips.Input("clips"),
                io.Combo.Input("backend", options=BACKENDS, default="auto",
                               tooltip="auto = torch on CUDA, numpy otherwise."),
                MographMusic.Input("music", optional=True),
                *[io.Image.Input(f"matte_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
                *[io.Image.Input(f"depth_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
            ],
            outputs=[MographClips.Output(display_name="composed"), io.Image.Output(display_name="hero")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, type=None, clips=None, backend="auto", music=None, **imgs) -> list[str]:
        if piece is None or not clips:
            return []
        need = []
        for clip in clips:
            sid = clip["sid"]
            k = sid[1:]
            n = RE.shot_needs(piece, sid, layered=clip.get("subject") is not None)
            if clip.get("plate"):
                n = dict(n, matte=True)                      # the plate route cuts the subject layer by its own matte (and moves it with the subject)
            for kind in ("matte", "depth"):
                name = f"{kind}_{k}"
                if n[kind] and imgs.get(name) is None and _wired(cls, name):
                    need.append(name)
            for _ref, tgt, kd in window_cut.other_refs(piece, sid):      # a window cut reads the NEIGHBOUR shot's matte too ($next_matte ...)
                name = f"matte_{tgt[1:]}"
                if kd == "matte" and imgs.get(name) is None and _wired(cls, name) and name not in need:
                    need.append(name)
        return need

    @classmethod
    def execute(cls, piece: dict, type: dict, clips: list, backend: str = "auto", music: dict | None = None,
                **imgs) -> io.NodeOutput:
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:   # noqa: BLE001
            pass
        mattes = {clip["sid"]: imgs.get(f"matte_{clip['sid'][1:]}") for clip in clips}
        depths = {clip["sid"]: imgs.get(f"depth_{clip['sid'][1:]}") for clip in clips}
        total = sum(int(RE.piece_layout(piece)[0][c["sid"]]["frames"]) for c in clips)
        t0 = time.time()
        composed, stats = compose_clips(piece, type, clips, music=music, mattes=mattes, depths=depths, backend=backend,
                                        progress=_progress(total))
        per = ", ".join("%s %.3fs/f" % (s["sid"], s["s_per_frame"]) for s in stats)
        print(f"[MographCompose] {len(composed)} shots, {sum(s['n'] for s in stats)} frames in {time.time() - t0:.1f}s "
              f"({per}) on {stats[0]['backend'] if stats else '-'}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hero = torch.from_numpy(pick_hero(piece, type.get("dir_layers", ""), composed))[None]
        return io.NodeOutput(composed, hero)


# --------------------------------------------------------------------------------------------- encoders
@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """h264_nvenc is listed by ``ffmpeg -encoders`` AND a 1-frame encode works (driver / GPU present)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
        if "h264_nvenc" not in r.stdout:
            return False
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=24:d=0.1", "-frames:v", "1",
                            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True, timeout=60)
        return p.returncode == 0
    except Exception:   # noqa: BLE001
        return False


def encoder_args(encoder: str, crf: int, draft: bool) -> dict:
    """-> kwargs for compose.assemble / encode_mp4 (vcodec, vopts, preset) + 'name'."""
    use_nvenc = encoder == "nvenc" or (encoder == "auto" and nvenc_available())
    if use_nvenc and not nvenc_available():
        print("[MographAssemble] h264_nvenc requested but not usable here; falling back to libx264")
        use_nvenc = False
    if use_nvenc:
        cq = int(crf) + 2 + (2 if draft else 0)            # crf 16 -> cq 18 (DESIGN.md), draft a little softer
        return dict(name="h264_nvenc", vcodec="h264_nvenc", vopts=["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"])
    if draft:
        return dict(name="libx264", vcodec="libx264", vopts=["-preset", "veryfast", "-crf", str(max(int(crf), 20))])
    return dict(name="libx264", vcodec="libx264", vopts=["-preset", "medium", "-crf", str(int(crf))])


def write_wav(audio: dict, path: Path) -> Path:
    """AUDIO {"waveform": [B,C,N], "sample_rate"} -> 16-bit PCM wav (first batch item)."""
    w = audio["waveform"]
    w = w.detach().cpu().float().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, np.float32)
    if w.ndim == 3:
        w = w[0]
    elif w.ndim == 1:
        w = w[None]
    pcm = (np.clip(w.T, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes(np.ascontiguousarray(pcm).tobytes())
    return path


def assemble_video(piece: dict, composed: list[dict], audio_path: str | None, out_path: str, *, crf: int = 16,
                   encoder: str = "auto") -> dict:
    """composed MOGRAPH_CLIPS -> mp4 (crossfades centred on the cuts, audio muxed). -> compose.assemble's dict + 'encoder', 'wall_s'."""
    parts, cfl = RE.assemble_parts(piece)
    by_sid = {c["sid"]: c for c in composed}
    parts = [p for p in parts if p["sid"] in by_sid]
    if not parts:
        raise ValueError("MographAssemble: no composed clips")
    size = RE.compose_size(piece)
    draft = piece.get("quality") in ("draft", "preview")
    enc = encoder_args(encoder, crf, draft)
    name = enc.pop("name")
    fps = int(piece.get("fps", FPS))
    specs = [dict(cut=p["cut"]) for p in parts]
    sources = [RE.ClipSource(by_sid[p["sid"]]["u8"], size) for p in parts]
    t0 = time.time()
    r = fc.assemble(specs, audio_path, str(out_path), crossfades=list(cfl)[:len(parts) - 1], fps=fps, size=size, crf=int(crf),
                    sources=sources, **enc)
    return dict(r, encoder=name, wall_s=time.time() - t0)


class MographAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographAssemble",
            display_name="MOTNGRFX Assemble",
            category=CATEGORY,
            description="Joins the composed shots (centred crossfades), muxes the audio (AAC) and encodes H.264. "
                        "encoder auto = NVENC when ffmpeg lists h264_nvenc and a 1-frame probe works, else libx264 "
                        "(draft: veryfast crf 20).",
            inputs=[
                MographClips.Input("composed"),
                MographPiece.Input("piece"),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("crf", default=16, min=0, max=40),
                io.Combo.Input("encoder", options=ENCODERS, default="auto"),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, composed: list, piece: dict, crf: int = 16, encoder: str = "auto", audio: dict | None = None) -> io.NodeOutput:
        d = _out_dir()
        tag = uuid.uuid4().hex[:10]
        wav = write_wav(audio, d / f"audio_{tag}.wav") if audio is not None and audio.get("waveform") is not None else None
        out = d / f"{piece.get('slug') or piece.get('id') or 'mograph'}_{tag}.mp4"
        r = assemble_video(piece, composed, str(wav) if wav else None, str(out), crf=crf, encoder=encoder)
        print(f"[MographAssemble] {r['frames']} frames, {r['seconds']:.2f}s, {r['encoder']}, {r['wall_s']:.1f}s -> {out.name}")
        return io.NodeOutput(InputImpl.VideoFromFile(str(out)))


NODES = [MographCompose, MographAssemble]
