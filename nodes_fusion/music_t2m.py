"""The approved music route as nodes (BUILT.md "Music"): ACE-Step 1.5 XL BASE text-to-music, no LM, music first, plan after.

    MographMusicT2MEncode  request (+ brief) -> ACE conditioning (no audio codes), seconds = length + 4, plan_json (bpm, key, mood ...)
    MographMusicT2MCut     raw ACE song -> EXACT piece length (one-beat cosine fade) + the beat grid MEASURED on the song -> MOGRAPH_MUSIC
    MographTameHighs       notch sustained 2-8 kHz tones (whistle / bell tones ACE leaves) + gentle high shelf

Graph (core ACE nodes as ``15_fusion/v004/make_music.py`` t2m): UNETLoader(xl base) -> ModelSamplingAuraFlow(3) -> KSampler(50 steps,
cfg 6, euler / simple, negative = ConditioningZeroOut) over EmptyAceStep1.5LatentAudio(seconds) -> VAEDecodeAudio -> T2MCut -> TameHighs
-> MusicRoute -> MographMaster(mono_safe). The numpy work is ``tools/fusion/t2m_music.py`` (shared with the runner script).
Replaces the rejected MographScoreSkeleton + audio-to-audio chain (``music_score.py``, kept registered, unused).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from comfy_api.latest import io

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted (ComfyUI's own ``nodes`` module must stay first)
    sys.path.append(_ROOT)

from mograph import fusion_brief  # noqa: E402
from mograph import score_skeleton as sk  # noqa: E402
from tools.fusion import audio_cut, plan_rules, t2m_music  # noqa: E402

from ._types import CATEGORY, FPS  # noqa: E402
from ._types import MographMusic as MusicT, MographRequest as RequestT  # noqa: E402
from .audio import audio_to_numpy, numpy_to_audio  # noqa: E402

SR = 48000


class MographMusicT2MEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicT2MEncode",
            display_name="Mograph Music T2M Encode",
            category=CATEGORY,
            description="ACE-Step 1.5 XL base text-to-music encode (no LM, no audio codes): a prose caption from your music prompt "
                        "(else the brief, else the mood) with a warm / mellow / no-high-pitched-lead clause, the tempo nudged to the "
                        "nearest whole-bar value for the length, instrumental. seconds = length + 4 (ACE ends early; the cut trims it).",
            inputs=[
                io.Clip.Input("clip"),
                RequestT.Input("request"),
                io.String.Input("brief", multiline=True, default="", optional=True,
                                tooltip="Raw Qwen brief: its music tags / bpm are used when the request has none."),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning"), io.Float.Output(display_name="seconds"),
                     io.String.Output(display_name="plan_json")],
        )

    @classmethod
    def execute(cls, clip, request, brief="") -> io.NodeOutput:
        req = plan_rules.normalize_request(request)
        mp = plan_rules.music_plan(req, fusion_brief.parse_brief(brief or None, req))
        length = float(req["length"])
        bpm = t2m_music.fit_bpm(mp["bpm"], length)
        text = t2m_music.caption(mp["tags"], bpm, mp["keyscale"], soften=not req["music"])     # the user's own prompt is kept as written
        seconds = length + t2m_music.EXTRA_S
        tokens = clip.tokenize(text, lyrics=mp["lyrics"], bpm=bpm, duration=seconds, timesignature=4, language="en",
                               keyscale=mp["keyscale"], seed=mp["seed"], generate_audio_codes=False, cfg_scale=2.0,
                               temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        plan = dict(bpm=bpm, bpm_asked=int(mp["bpm"]), keyscale=mp["keyscale"], mood=mp["mood"], caption=text, music_prompt=mp["tags"],
                    length=length, seconds=seconds, seed=int(mp["seed"]), preset=t2m_music.preset_for(mp["mood"]),
                    energy=float(plan_rules._MOOD_ENERGY.get(mp["mood"], 0.5)))
        print(f"[Mograph Music T2M Encode] {bpm} bpm (asked {mp['bpm']}), {mp['keyscale']}, {mp['mood']}, {seconds:.0f} s: {text[:120]}")
        return io.NodeOutput(conditioning, float(seconds), json.dumps(plan))


class MographMusicT2MCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicT2MCut",
            display_name="Mograph Music T2M Cut",
            category=CATEGORY,
            description="Raw ACE-Step text-to-music take -> exactly the piece length (one-beat cosine fade; shorter takes are padded), and "
                        "the beat grid MEASURED on the song (tempo, phase, downbeats, bars, chord changes, band curves) as MOGRAPH_MUSIC, "
                        "so the plan is built from the music that was actually made. Falls back to the asked tempo if tracking fails.",
            inputs=[
                io.Audio.Input("audio"),
                RequestT.Input("request"),
                io.String.Input("plan_json", multiline=True, default="{}", tooltip="From Music T2M Encode: tempo asked, key, mood."),
            ],
            outputs=[io.Audio.Output(display_name="audio"), MusicT.Output(display_name="music")],
        )

    @classmethod
    def execute(cls, audio, request, plan_json="{}") -> io.NodeOutput:
        req = plan_rules.normalize_request(request)
        length = float(req["length"])
        plan = json.loads(plan_json or "{}")
        a = audio_to_numpy(audio)
        if a is None:
            raise ValueError("MographMusicT2MCut: the audio input is empty")
        st, sr = audio_cut.as_stereo(a[0], a[1], SR)
        x = st.T.astype(np.float64)
        if float(np.abs(x).max()) < 1e-4:
            raise ValueError("MographMusicT2MCut: ACE-Step returned silence")
        asked = float(plan.get("bpm") or 100)
        try:
            fit = t2m_music.track_beats(x.mean(axis=1), sr, asked)
        except Exception as e:     # noqa: BLE001 - keep the run alive on the asked tempo
            print(f"[Mograph Music T2M Cut] beat tracking failed ({type(e).__name__}: {e}); using the asked {asked:.0f} bpm grid")
            fit = dict(bpm=asked, beat_s=60.0 / asked, t0=0.0, downbeat_phase=0, downbeat_conf=1.0, tracked=False)
        gate = t2m_music.end_gate(x, sr, length)
        y = t2m_music.cut_and_fade(x, sr, length, fit["beat_s"])
        music = t2m_music.t2m_grid(length, fit, y.T, sr, t2m_music.FADE_BEATS * fit["beat_s"], key=plan.get("keyscale", "A minor"),
                                   mood=plan.get("mood", "calm"), energy=float(plan.get("energy", 0.5)), preset=plan.get("preset", "warm"),
                                   music_prompt=plan.get("music_prompt", ""), bpm_hint=asked, seed=int(plan.get("seed", 1)))
        music["curves"] = sk.curves_of(y.T.astype(np.float32), sr, FPS, int(music["nframes"]))
        music["t2m_end_gate"] = gate
        print(f"[Mograph Music T2M Cut] tracked {fit['bpm']:.2f} bpm (asked {asked:.0f}), {music['n_bars']} bars, first downbeat "
              f"{music['downbeats'][0]:.2f} s, {y.shape[0]} samples = {length:.2f} s, music until ~{gate['content_end_s']:.1f} s"
              + ("" if gate["passed"] else "  [END GATE: the song went quiet before the end]"))
        return io.NodeOutput(numpy_to_audio(y.T.astype(np.float32), sr), music)


class MographTameHighs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographTameHighs",
            display_name="Mograph Tame Highs",
            category=CATEGORY,
            description="Notch every sustained narrow tone between 2 and 8 kHz (the whistle / bell tones ACE-Step leaves in text-to-music; "
                        "Q 8, cut = the tone's excess, at most 9 dB) and add a gentle high shelf (-2.5 dB at 5 kHz). Length unchanged.",
            inputs=[io.Audio.Input("audio")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        a = audio_to_numpy(audio)
        if a is None:
            raise ValueError("MographTameHighs: the audio input is empty")
        st, sr = audio_cut.as_stereo(a[0], a[1])
        y, peaks = t2m_music.tame_highs(st.T.astype(np.float64), sr)
        print(f"[Mograph Tame Highs] {len(peaks)} sustained tone(s) notched: "
              + ", ".join(f"{p['hz']:.0f} Hz -{min(9.0, p['excess_db']):.1f} dB" for p in peaks))
        return io.NodeOutput(numpy_to_audio(y.T.astype(np.float32), sr))


NODES = [MographMusicT2MEncode, MographMusicT2MCut, MographTameHighs]
