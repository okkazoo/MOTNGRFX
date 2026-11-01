"""MographArchiveLoad: a finished generation's archive (MographArchive output) as the inputs of the REDESIGN graph (app/graphs/fusion_redesign*.json).

    MographArchiveLoad  archive, source_video, concept -> request, music, piece, clips, first, audio, matte_1..6

The redesign graph runs ONLY the design step again over footage that already exists: design package -> open model -> parse (field salvage) -> lazy repair ->
PieceFromJSON -> type passes -> compose -> assemble -> archive. No music model, no still model, no LTX, no Alpha-Gen. This node supplies what those stages made:

    archive        folder under the ComfyUI output dir (or an absolute path) with still.png, sK_clip.mp4, sK_matte/NNNNN.png, piece.json, index.json
                   (written by MographArchive). ANY archived generation works; the piece.json is the generation's designed piece = the footage layout
                   (shots, frames, sizes, beat grid) the redesign keeps.
    source_video   the generation's final mp4 (its audio track is the song; empty = silence, no curves). Relative to the output dir, or absolute.
    concept        JSON of the generation's request (words / scene / music / look / shots / length / seed); empty = rebuilt from the piece.

music = the piece's own beat grid (beats, downbeats, bpm, cut) + the per-frame band curves measured on the audio (rms, low, ..: dropouts, glow envelope).
Chord changes are not archived and stay empty (the design prompt then names no harmony). Everything is read, nothing is generated: no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import io

try:
    import folder_paths
except Exception:   # pragma: no cover
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY, FPS, MAX_SHOTS, MographClips, MographMusic, MographPiece, MographRequest, new_clip

ensure_root_on_path()
from tools.fusion import design_step as DS        # noqa: E402
from tools.fusion import plan_rules               # noqa: E402


def _resolve(p: str) -> Path:
    q = Path(str(p).strip())
    if q.is_absolute():
        return q
    base = Path(folder_paths.get_output_directory() if folder_paths is not None else ".")
    return base / q


def read_clip(path: Path) -> np.ndarray:
    """mp4 -> uint8 [T,H,W,3] RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"MographArchiveLoad: no frames in {path}")
    return np.stack(frames)


def read_mattes(folder: Path) -> np.ndarray | None:
    """sK_matte/NNNNN.png -> float32 [T,H,W,3] 0..1 (IMAGE), or None."""
    files = sorted(folder.glob("*.png"))
    if not files:
        return None
    out = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    return np.stack(out).astype(np.float32) / 255.0


def request_from(concept: dict, piece: dict) -> dict:
    """the request of the generation: the concept JSON when given, else rebuilt from the piece (words joined, the first still prompt, length)."""
    c = dict(concept or {})
    words = c.get("words")
    if isinstance(words, list):
        words = "\n".join(str(w) for w in words)
    cut = float((piece.get("audio") or {}).get("cut") or 16.0)
    req = {
        "words": words or "\n".join(w["text"] for w in (piece.get("type") or {}).get("words") or []),
        "scene": c.get("scene") or ((piece.get("stills") or [{}])[0].get("prompt") or ""),
        "music": c.get("music") or "", "length": int(c.get("length") or round(cut)), "look": c.get("look") or "auto",
        "seed": int(c.get("seed") or 1), "quality": piece.get("quality") or "preview", "format": piece.get("format") or "9:16",
        "shots": c.get("shots") or ""}
    for k in ("idea", "owners", "shot_plan", "system", "effects", "earlier_openings", "layout_seed"):   # gap 5 (OWN_VIDEO.md): the design direction survives into the request
        if c.get(k) not in (None, "", [], {}):
            req[k] = c[k]
    return plan_rules.normalize_request(req)


def music_from(piece: dict, wave: np.ndarray | None, sr: int | None) -> dict:
    """the beat grid stored in the piece + the band curves measured on the generation's audio."""
    ana = DS.ana_from_piece(piece)
    ana["fade"], ana["src"] = float((piece.get("audio") or {}).get("fade", 0.5)), (piece.get("audio") or {}).get("src", "")     # the plan re-writes piece.audio from these
    if wave is not None and sr:
        from mograph import score_skeleton as sk
        w = wave[:, :int(ana["cut"] * sr)] if wave.ndim == 2 else wave[None, :int(ana["cut"] * sr)]
        ana["curves"] = sk.curves_of(w.astype(np.float32), sr, FPS, int(ana["nframes"]))
    return ana


class MographArchiveLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographArchiveLoad",
            display_name="MOTNGRFX Archive Load",
            category=CATEGORY,
            description="A finished generation's archive folder (still, shot clips, matte PNGs, piece.json) + its final mp4 (the song) as the inputs of the "
                        "redesign graph. Reads files only.",
            inputs=[
                io.String.Input("archive", default="", tooltip="Folder with still.png, sK_clip.mp4, sK_matte/, piece.json: under the ComfyUI output dir, or absolute."),
                io.String.Input("source_video", default="", tooltip="The generation's final mp4: its audio is the song. Empty = no audio."),
                io.String.Input("concept", multiline=True, default="", tooltip="JSON of the generation's request (words, scene, music, look, shots, length, seed). Empty = rebuilt from the piece."),
            ],
            outputs=[MographRequest.Output(display_name="request"), MographMusic.Output(display_name="music"), MographPiece.Output(display_name="piece"),
                     MographClips.Output(display_name="clips"), io.Image.Output(display_name="first"), io.Audio.Output(display_name="audio"),
                     *[io.Image.Output(display_name=f"matte_{k}") for k in range(1, MAX_SHOTS + 1)]],
        )

    @classmethod
    def fingerprint_inputs(cls, archive: str, source_video: str = "", concept: str = ""):
        p = _resolve(archive) / "piece.json"
        return f"{archive}|{source_video}|{p.stat().st_mtime if p.is_file() else 0}|{hash(concept)}"

    @classmethod
    def execute(cls, archive: str, source_video: str = "", concept: str = "") -> io.NodeOutput:
        root = _resolve(archive)
        if not (root / "piece.json").is_file():
            raise ValueError(f"MographArchiveLoad: {root} has no piece.json (is `archive` a MographArchive folder?)")
        piece = json.loads((root / "piece.json").read_text(encoding="utf-8"))
        index = json.loads((root / "index.json").read_text(encoding="utf-8")) if (root / "index.json").is_file() else {}
        try:
            cj = json.loads(concept) if str(concept or "").strip() else {}
        except ValueError:
            cj = {}
        # the song: the final mp4's audio track
        wave, sr, audio = None, None, {"waveform": torch.zeros(1, 2, int(float((piece.get("audio") or {}).get("cut") or 16.0) * 48000)), "sample_rate": 48000}
        if str(source_video or "").strip():
            sv = _resolve(source_video)
            if not sv.is_file():
                raise ValueError(f"MographArchiveLoad: source_video {sv} not found")
            from comfy_extras.nodes_audio import load
            w, sr = load(str(sv))
            wave, sr = w.detach().cpu().float().numpy(), int(sr)
            audio = {"waveform": w.unsqueeze(0), "sample_rate": int(sr)}
        music = music_from(piece, wave, sr)
        request = request_from(cj, piece)
        piece = plan_rules.finalize_piece(piece, music)
        # footage
        lay = piece.get("layout") or {}
        clips, mattes = [], {}
        for sid in sorted(lay, key=lambda s: int(str(s).lstrip("s") or 0)):
            f = root / f"{sid}_clip.mp4"
            if not f.is_file():
                f = root / f"{sid}_clean.mp4"
            u8 = read_clip(f)
            start = int((index.get("shots") or {}).get(sid, {}).get("start", lay[sid]["start"]))
            clips.append(new_clip(torch.from_numpy(u8), sid, start, FPS))
            m = read_mattes(root / f"{sid}_matte")
            if m is not None:
                mattes[sid] = torch.from_numpy(m)
        still = cv2.cvtColor(cv2.imread(str(root / "still.png")), cv2.COLOR_BGR2RGB) if (root / "still.png").is_file() else clips[0]["u8"][0].numpy()
        first = torch.from_numpy(np.ascontiguousarray(still).astype(np.float32) / 255.0)[None]
        ph = torch.zeros(1, 16, 16, 3)
        mt = [mattes.get(f"s{k}", ph) for k in range(1, MAX_SHOTS + 1)]
        print(f"[MographArchiveLoad] {root.name}: {len(clips)} shots, {len(mattes)} mattes, audio {'from ' + str(source_video) if wave is not None else 'silent'}, "
              f"{piece['id']} {len(piece['type']['words'])} words")
        return io.NodeOutput(request, music, piece, clips, first, audio, *mt)


NODES = [MographArchiveLoad]
