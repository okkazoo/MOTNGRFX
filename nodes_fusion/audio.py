"""Audio nodes of the fusion pipeline (16_app/DESIGN.md section 1): MusicEncode, MusicCut, Master.

    MographMusicEncode  ACE-Step 1.5 conditioning (same call as core ``TextEncodeAceStepAudio1.5``) from the request + brief
    MographMusicCut     music analysis + cut on a downbeat below the end of the music (ACE ends early, then digital silence)
    MographMaster       EQ -> glue -> M/S width -> mono below 120 Hz -> -14 LUFS / -1.0 dBTP (the v002 songs were "tiny")

The numpy work is in ``tools/fusion/audio_cut.py`` and ``mograph/mastering.py``; torch <-> numpy conversion happens here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted (ComfyUI's own ``nodes`` module must stay first)
    sys.path.append(_ROOT)

from tools.fusion import audio_cut, plan_rules  # noqa: E402
from mograph import fusion_brief  # noqa: E402

from ._types import CATEGORY, FPS  # noqa: E402
from ._types import MographMusic as MusicT, MographRequest as RequestT  # noqa: E402

MISSING = object()


def audio_to_numpy(audio) -> tuple[np.ndarray, int] | None:
    """AUDIO {"waveform": [B, C, N], "sample_rate"} -> ((C, N) float32, sr), first batch item; None for no / empty audio."""
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None
    w = audio["waveform"]
    w = w.detach().cpu().float().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, np.float32)
    if w.ndim == 3:
        w = w[0]
    elif w.ndim == 1:
        w = w[None]
    if w.size == 0:
        return None
    return np.ascontiguousarray(w, dtype=np.float32), int(audio["sample_rate"])


def numpy_to_audio(wave: np.ndarray, sr: int) -> dict:
    w = np.asarray(wave, np.float32)
    if w.ndim == 1:
        w = w[None]
    return {"waveform": torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0), "sample_rate": int(sr)}


class MographMusicEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicEncode",
            display_name="Mograph Music Encode",
            category=CATEGORY,
            description="ACE-Step 1.5 text encode for the app: tags from your music prompt (else the brief, else the mood), "
                        "instrumental, tempo and key, duration 1.7x the length + 2 s (ACE ends early; the cut trims it).",
            inputs=[
                io.Clip.Input("clip"),
                RequestT.Input("request"),
                io.String.Input("brief", multiline=True, default="", optional=True,
                                tooltip="Raw Qwen brief: its music tags / bpm are used when the request has none."),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning"), io.Float.Output(display_name="seconds"),
                     io.Int.Output(display_name="bpm")],
        )

    @classmethod
    def execute(cls, clip, request, brief="") -> io.NodeOutput:
        req = plan_rules.normalize_request(request)
        b = fusion_brief.parse_brief(brief or None, req)
        mp = plan_rules.music_plan(req, b)
        tokens = clip.tokenize(mp["tags"], lyrics=mp["lyrics"], bpm=mp["bpm"], duration=mp["seconds"],
                               timesignature=int(mp["timesignature"]), language="en", keyscale=mp["keyscale"],
                               seed=mp["seed"], generate_audio_codes=True, cfg_scale=2.0, temperature=0.85, top_p=0.9,
                               top_k=0, min_p=0.0)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning, float(mp["seconds"]), int(mp["bpm"]))


class MographMusicCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicCut",
            display_name="Mograph Music Cut",
            category=CATEGORY,
            description="Analyse the song (tempo, beats, downbeats, key, chord changes, band curves) and cut it to "
                        "exactly the requested length (one-beat fade out; a shorter track is looped at a downbeat with a short crossfade). Your own audio "
                        "replaces the ACE audio (which is then never generated).",
            inputs=[
                RequestT.Input("request"),
                io.Int.Input("music_bpm", default=100, min=10, max=300, tooltip="Tempo asked from ACE (hint for the beat tracker)."),
                io.Audio.Input("ace_audio", lazy=True, optional=True),
                io.Audio.Input("user_audio", optional=True),
            ],
            outputs=[MusicT.Output(display_name="music"), io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def check_lazy_status(cls, request, music_bpm, ace_audio=MISSING, user_audio=None):
        # None = connected but not evaluated yet; MISSING = not connected
        if audio_to_numpy(user_audio) is not None:
            return []
        if ace_audio is None:
            return ["ace_audio"]
        return []

    @classmethod
    def execute(cls, request, music_bpm, ace_audio=None, user_audio=None) -> io.NodeOutput:
        req = plan_rules.normalize_request(request)
        user = audio_to_numpy(user_audio)
        src, hint = "user", plan_rules.parse_bpm(req["music"])      # the tracker alone can return 2/3 of the tempo
        a = user
        if a is None:
            a = audio_to_numpy(ace_audio)
            src, hint = "ace", float(music_bpm)
        if a is None:
            raise ValueError("MographMusicCut needs ace_audio (the music model output) or user_audio")
        wave, sr = a
        ana = audio_cut.analyse_wave(wave, sr, hint, FPS)
        # EXACTLY the chosen Length (the old nearest-downbeat snap made an 8 s piece 9.07 s with a 106 bpm track and a 12 s piece
        # 12.85 s at 75 bpm): the fade is one beat (0.5..1 s) at the end; a track shorter than Length is looped (downbeat to last bar, crossfaded joins).
        length = float(req["length"])
        fade = min(1.0, max(0.5, 60.0 / max(float(ana["bpm"]), 1.0)))
        cut, music = audio_cut.cut_to_downbeat(wave, sr, ana, cut=length, fps=FPS, fade=fade, pad=True)
        music["src"] = src
        return io.NodeOutput(music, numpy_to_audio(cut, 48000))


class MographMaster(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMaster",
            display_name="Mograph Master",
            category=CATEGORY,
            description="Make the song bigger: EQ (low end, less mud, presence, air), glue compressor, mid/side width, mono "
                        "below 120 Hz, then loudness to the target LUFS with a true-peak limiter at -1.0 dBTP.",
            inputs=[
                io.Audio.Input("audio"),
                MusicT.Input("music"),
                io.Combo.Input("preset", options=["auto", *audio_cut.MASTER_PRESETS], default="auto",
                               tooltip="auto: 'warm' for calm / epic music, 'club' otherwise."),
                io.Float.Input("target_lufs", default=-14.0, min=-24.0, max=-8.0, step=0.5),
                io.Float.Input("width", default=1.25, min=0.5, max=2.0, step=0.05, tooltip="Side level: 1.0 unchanged, 0 mono."),
                io.Boolean.Input("mono_safe", default=False, optional=True,
                                 tooltip="True mono below 200 Hz before the master; if the mono fold-down loses more than 1.5 dB the "
                                         "width goes to 1.0 and the side channel is reduced (tools/fusion/t2m_music.safe_master)."),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audio, music, preset, target_lufs, width, mono_safe=False) -> io.NodeOutput:
        a = audio_to_numpy(audio)
        if a is None:
            raise ValueError("MographMaster: the audio input is empty")
        wave, sr = a
        if preset == "auto":      # the text-to-music route names its preset (warm unless a dance mood); else by the grid's mood
            preset = (music or {}).get("master_preset") or audio_cut.preset_for_mood(
                ((music or {}).get("grid") or {}).get("mood") or (music or {}).get("mood"))
        if mono_safe:
            from tools.fusion import t2m_music
            st, sr = audio_cut.as_stereo(wave, sr)
            cut = (music or {}).get("cut")
            y, rep = t2m_music.safe_master(st.T.astype(np.float64), sr, preset, float((music or {}).get("bpm") or 120.0),
                                           None if cut is None else float(cut), width=float(width))
            print(f"[Mograph Master] mono-safe: width {rep['width']:.2f}, side x{rep['side_k']:.2f}, "
                  f"fold-down {rep['mono_after']['folddown_db']:+.2f} dB")
        else:
            y, rep = audio_cut.master_audio(wave, sr, preset=preset, target_lufs=float(target_lufs), width=float(width),
                                            bpm=(music or {}).get("bpm"), cut=(music or {}).get("cut"))
        print(f"[Mograph Master] {preset}: {rep['before']['lufs']:.1f} LUFS / {rep['before']['true_peak_dbtp']:+.1f} dBTP -> "
              f"{rep['lufs']:.1f} LUFS / {rep['true_peak_dbtp']:+.1f} dBTP")
        return io.NodeOutput(numpy_to_audio(y, sr))


NODES = [MographMusicEncode, MographMusicCut, MographMaster]
