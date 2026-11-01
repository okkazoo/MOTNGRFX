"""DEPRECATED 2026-10-27 (user: "circus / tinker toy"): the app graph no longer uses these nodes; its music is the ACE-Step XL base
text-to-music route in ``music_t2m.py`` (BUILT.md "Music"). Kept registered for old graphs (``music_exact_test.json``) and tests.
MographMusicRoute is still used by the app graph.

Exact-length, beat-locked music for the fusion pipeline (16_app/MUSIC.md): ScoreSkeleton -> ACE-Step audio-to-audio -> MusicFit.

    MographScoreSkeleton   composed CPU stem of EXACTLY the piece length (SoundFont, mograph/score_skeleton.py) + the known beat grid
    MographMusicStyleEncode  ACE-Step 1.5 text encode for audio-to-audio: tags from the music prompt, bpm / key / duration from the grid
    MographMusicFit        force the restyled audio to the exact sample count, keep the skeleton grid as the analysis
    MographMusicRoute      user audio -> MusicCut branch, else the score branch (lazy: only the chosen branch runs)

Graph (core nodes as in workflows/api/acestep_a2a.json): ScoreSkeleton.audio -> VAEEncodeAudio(ace_1.5_vae) -> KSampler(latent_image,
denoise 0.3-0.7, positive = StyleEncode, negative = ConditioningZeroOut) -> VAEDecodeAudio -> MusicFit(audio, music, skeleton) ->
MographMaster -> MographPiecePlan. MusicFit's MOGRAPH_MUSIC has the keys MographMusicCut makes, so PiecePlan consumes it unchanged.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from comfy_api.latest import io

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted (ComfyUI's own ``nodes`` module must stay first)
    sys.path.append(_ROOT)

from mograph import score_skeleton as sk  # noqa: E402
from tools.fusion import plan_rules  # noqa: E402

from ._types import CATEGORY, FPS  # noqa: E402
from ._types import MographMusic as MusicT, MographRequest as RequestT  # noqa: E402
from .audio import audio_to_numpy, numpy_to_audio  # noqa: E402

MOOD_OPTIONS = ["auto"] + list(sk.MOODS)
# style tags for the audio-to-audio text when there is no music prompt and no request (the skeleton mood)
_MOOD_TAGS = {
    "driving": "driving electronic, punchy four-on-the-floor kick, rolling bass, bright synth stabs, energetic",
    "calm": "ambient cinematic, soft felt piano, warm string pad, cello, slow swelling pads, airy harp, spacious reverb, floating, wistful",
    "dark": "dark deep house, round sub bass, soft kick, brooding analog pad, late-night, hypnotic",
    "bright": "bright melodic house, bouncy plucked synth, round bass, crisp claps, uplifting",
    "playful": "playful groove, bouncy bass, marimba and vibraphone, light percussion, cheerful",
    "epic": "epic cinematic film score, orchestral, sweeping strings, brass, timpani, deep drums, harp, choir pad, building, awe",
}


# landscape imagery gets the orchestral score (strings, brass, timpani) unless the text asks for dance music
_LANDSCAPE = re.compile(r"\b(mountains?|ridges?|ranges?|valleys?|vistas?|highlands?|peaks?|canyons?|glaciers?|horizons?)\b", re.I)
_DANCE = re.compile(r"\b(house|techno|dance|edm|club)\b", re.I)


def _auto_mood(req: dict | None, music_text: str) -> str:
    text = " ".join([music_text or "", (req or {}).get("scene", ""), (req or {}).get("words", "")])
    if _LANDSCAPE.search(text) and not _DANCE.search(text):
        return "epic"
    for m in plan_rules.moods_from_text(text):
        m = sk.MOOD_ALIAS.get(m, m)
        if m in sk.MOODS:
            return m
    return "driving" if re.search(r"\b(house|techno|dance|edm|club)\b", text, re.I) else "calm"


class MographScoreSkeleton(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographScoreSkeleton",
            display_name="Mograph Score Skeleton",
            category=CATEGORY,
            description="Compose a CPU timing stem of exactly the piece length with a known beat grid (bars x meter x 60 / bpm = "
                        "length; the tempo is nudged to the nearest value that fits, the audio is never trimmed). Intro / build / "
                        "drop / outro, drums + bass + chords + pad + arp (SoundFont; synth fallback). Feed it to ACE-Step "
                        "audio-to-audio at denoise 0.3-0.7. Hits (video cut times) snap to the nearest beat.",
            inputs=[
                RequestT.Input("request", optional=True,
                               tooltip="Takes length, seed, and from the music prompt the bpm / key / mood; overrides the fields below."),
                io.Float.Input("length_s", default=12.0, min=1.0, max=120.0, step=0.5,
                               tooltip="Piece length when no request is linked (rounded to whole frames)."),
                io.String.Input("music_prompt", default="", tooltip="Music prompt when no request is linked: its 'NNN bpm', key and mood words are read."),
                io.Int.Input("bpm", default=0, min=0, max=200, tooltip="Tempo hint; 0 = from the music prompt, else the mood's."),
                io.Combo.Input("mood", options=MOOD_OPTIONS, default="auto"),
                io.Float.Input("energy", default=-1.0, min=-1.0, max=1.0, step=0.05, tooltip="-1 = the mood's."),
                io.String.Input("key", default="", tooltip="e.g. 'A minor', 'F# dorian'. Empty = from the music prompt, else mood + seed."),
                io.Int.Input("meter", default=4, min=2, max=7, tooltip="Beats per bar."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffff, tooltip="Used when no request is linked."),
                io.String.Input("hits", default="", tooltip="Video cut times in seconds, e.g. '3.5, 8, 12'; snapped to beats."),
                io.String.Input("sections", default="", tooltip="e.g. 'intro:0-3, build:3-6, drop:6-10, outro:10-12' (snapped to bars). Empty = default arc."),
                io.String.Input("brief", multiline=True, default="", optional=True,
                                tooltip="Raw Qwen brief (with a request): its bpm / mood are used when the music prompt has none."),
            ],
            outputs=[io.Audio.Output(display_name="audio"), MusicT.Output(display_name="music"),
                     io.String.Output(display_name="grid_json")],
        )

    @classmethod
    def execute(cls, request=None, length_s=12.0, music_prompt="", bpm=0, mood="auto", energy=-1.0, key="", meter=4, seed=0,
                hits="", sections="", brief="") -> io.NodeOutput:
        req = plan_rules.normalize_request(request) if request else None
        text = req["music"] if req else " ".join(str(music_prompt or "").split())
        length = float(req["length"]) if req else float(length_s)
        use_seed = plan_rules.sub_seed(req["seed"], "score") if req else int(seed)
        use_bpm = int(bpm) or plan_rules.parse_bpm(text) or None
        km = re.search(r"\b([A-G][b#]?)\s*(major|minor|dorian|phrygian|lydian|mixolydian)\b", text)
        use_key = (key or "").strip() or (f"{km.group(1)} {km.group(2)}" if km else None)
        use_mood = _auto_mood(req, text) if mood == "auto" else mood
        if req:        # same priority as MographMusicEncode: music prompt, then the brief, then rules from the words
            from mograph import fusion_brief
            mp = plan_rules.music_plan(req, fusion_brief.parse_brief(brief or None, req))
            use_bpm = use_bpm or int(mp["bpm"])
            use_key = use_key or mp["keyscale"]
            if mood == "auto":
                use_mood = sk.MOOD_ALIAS.get(mp["mood"], mp["mood"])
                use_mood = use_mood if use_mood in sk.MOODS else "driving"
        wave, grid = sk.compose(length_s=length, bpm=use_bpm, meter=int(meter), key=use_key, mood=use_mood,
                                energy=None if float(energy) < 0 else float(energy), sections=sk.parse_sections(sections),
                                hits=sk.parse_hits(hits), seed=use_seed, fps=FPS, sr=48000)
        music = sk.music_dict(grid, wave, 48000, FPS, src="score")
        slim = {k: grid[k] for k in ("bpm", "bpm_hint", "n_bars", "meter", "bar_len", "length_s", "n_frames", "n_samples", "key",
                                     "key_ace", "mood", "energy", "palette", "backend", "sections", "hits", "hits_requested")}
        print(f"[Mograph Score Skeleton] {grid['length_s']:.3f} s = {grid['n_bars']} bars x {grid['meter']} at {grid['bpm']:.3f} bpm "
              f"(hint {grid['bpm_hint']:.0f}), {grid['key']}, {grid['mood']}/{grid['palette']} via {grid['backend']}, hits {grid['hits']}")
        return io.NodeOutput(numpy_to_audio(wave, 48000), music, json.dumps(slim))


class MographMusicStyleEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicStyleEncode",
            display_name="Mograph Music Style Encode",
            category=CATEGORY,
            description="ACE-Step 1.5 text encode for audio-to-audio over the score skeleton: style tags from the music prompt "
                        "(tags field, else the request, else the skeleton mood), instrumental; bpm, key, time signature and "
                        "duration come from the skeleton grid so the text agrees with the audio. No audio codes (audio reference).",
            inputs=[
                io.Clip.Input("clip"),
                MusicT.Input("music"),
                io.String.Input("tags", multiline=True, default="", tooltip="Style / music prompt. Empty = from the request, else the mood."),
                RequestT.Input("request", optional=True),
                io.String.Input("brief", multiline=True, default="", optional=True,
                                tooltip="Raw Qwen brief: its music tags are used when there is no music prompt."),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, clip, music, tags="", request=None, brief="") -> io.NodeOutput:
        grid = (music or {}).get("grid") or {}
        bpm = float((music or {}).get("bpm") or grid.get("bpm") or 120.0)
        meter = int((music or {}).get("meter") or grid.get("meter") or 4)
        dur = float((music or {}).get("cut") or grid.get("length_s") or 12.0)
        text = " ".join(str(tags or "").split())
        if not text and request:
            from mograph import fusion_brief
            req = plan_rules.normalize_request(request)
            text = plan_rules.music_plan(req, fusion_brief.parse_brief(brief or None, req))["tags"]
        if not text:
            text = _MOOD_TAGS.get(grid.get("mood", "driving"), _MOOD_TAGS["driving"])
        text = re.sub(r"[,\s]*\b\d{2,3}(\.\d+)?\s*bpm\b", "", text, flags=re.I).rstrip(" ,.")     # the grid's tempo, not the prompt's
        if not re.search(r"\bno vocals\b", text, re.I):
            text += ", " + plan_rules.NO_VOX
        text += f", {int(round(bpm))} BPM"
        seed = int(grid.get("seed", 0)) if grid else 0
        tokens = clip.tokenize(text, lyrics="[Instrumental]", bpm=int(round(bpm)), duration=dur,
                               timesignature=meter if meter in (2, 3, 4, 6) else 4, language="en",
                               keyscale=grid.get("key_ace", "A minor"), seed=seed, generate_audio_codes=False, cfg_scale=2.0,
                               temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class MographMusicFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicFit",
            display_name="Mograph Music Fit",
            category=CATEGORY,
            description="Force the restyled (ACE-Step audio-to-audio) audio to the skeleton's exact sample count at 48 kHz: cut + "
                        "short fade when longer; when shorter or silent before the end, the skeleton's tail takes over through an "
                        "equal-power crossfade. The skeleton grid stays as the analysis (bpm, beats, downbeats, chord changes); "
                        "only the per-frame curves are re-measured on the new audio. Optional loudness match to the skeleton.",
            inputs=[
                io.Audio.Input("audio"),
                MusicT.Input("music"),
                io.Audio.Input("skeleton", optional=True, tooltip="The skeleton audio (tail fill + loudness reference)."),
                io.Int.Input("xfade_ms", default=80, min=0, max=1000, tooltip="Crossfade into the skeleton tail."),
                io.Int.Input("tail_fade_ms", default=30, min=0, max=2000, tooltip="Fade-out at the very end (click guard)."),
                io.Boolean.Input("match_loudness", default=False, tooltip="Match the integrated loudness to the skeleton."),
            ],
            outputs=[io.Audio.Output(display_name="audio"), MusicT.Output(display_name="music")],
        )

    @classmethod
    def execute(cls, audio, music, skeleton=None, xfade_ms=80, tail_fade_ms=30, match_loudness=False) -> io.NodeOutput:
        a = audio_to_numpy(audio)
        if a is None:
            raise ValueError("MographMusicFit: the audio input is empty")
        wave, sr = a
        sr_out = int((music or {}).get("sr") or 48000)
        n = int((music or {}).get("n_samples") or round(float(music["cut"]) * sr_out))
        s = audio_to_numpy(skeleton)
        sk_wave = None
        if s is not None:
            sk_wave = sk._stereo(s[0], s[1], sr_out)
        y, rep = sk.fit_exact(wave, sr, n, out_sr=sr_out, skeleton=sk_wave, xfade_s=xfade_ms / 1000.0,
                              tail_fade_s=tail_fade_ms / 1000.0, loudness=bool(match_loudness))
        out = dict(music)
        out["curves"] = sk.curves_of(y, sr_out, int(out.get("fps") or FPS), int(out["nframes"]))
        out["src"] = "score+ace"
        out["fit"] = rep
        print(f"[Mograph Music Fit] {rep['in_samples']} -> {n} samples ({rep['mode']}, live end {rep['live_samples'] / sr_out:.2f} s)"
              + (f", loudness {rep['gain_db']:+.1f} dB" if match_loudness else ""))
        return io.NodeOutput(numpy_to_audio(y, sr_out), out)


def _wired(cls, name: str) -> bool:
    """Is input ``name`` linked in this node's own prompt entry? (no prompt available -> assume yes). Requesting an unlinked
    lazy input raises in ComfyUI's executor (16_app/DESIGN.md "Spike results")."""
    hid = getattr(cls, "hidden", None)
    prompt = getattr(hid, "prompt", None)
    if not prompt:
        return True
    return isinstance((prompt.get(str(hid.unique_id), {}).get("inputs") or {}).get(name), list)


class MographMusicRoute(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMusicRoute",
            display_name="Mograph Music Route",
            category=CATEGORY,
            description="Pick the music branch: the user's own audio (MographMusicCut) when the request carries audio, else the "
                        "score branch (skeleton -> ACE audio-to-audio -> MusicFit). Only the chosen branch is evaluated, so with "
                        "user audio the skeleton and ACE never run, and without it MusicCut never runs.",
            inputs=[
                RequestT.Input("request"),
                io.Audio.Input("user_audio", lazy=True, optional=True),
                MusicT.Input("user_music", lazy=True, optional=True),
                io.Audio.Input("score_audio", lazy=True, optional=True),
                MusicT.Input("score_music", lazy=True, optional=True),
            ],
            outputs=[io.Audio.Output(display_name="audio"), MusicT.Output(display_name="music")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @staticmethod
    def _has_user(request) -> bool:
        return audio_to_numpy((request or {}).get("audio")) is not None

    @classmethod
    def check_lazy_status(cls, request=None, user_audio=None, user_music=None, score_audio=None, score_music=None) -> list[str]:
        if request is None:
            return []
        side = ("user_audio", "user_music") if cls._has_user(request) else ("score_audio", "score_music")
        have = dict(user_audio=user_audio, user_music=user_music, score_audio=score_audio, score_music=score_music)
        return [n for n in side if have[n] is None and _wired(cls, n)]

    @classmethod
    def execute(cls, request, user_audio=None, user_music=None, score_audio=None, score_music=None) -> io.NodeOutput:
        if cls._has_user(request):
            if user_audio is None or user_music is None:
                raise ValueError("MographMusicRoute: the request has audio but user_audio / user_music are not connected")
            return io.NodeOutput(user_audio, user_music)
        if score_audio is None or score_music is None:
            raise ValueError("MographMusicRoute: no user audio in the request and score_audio / score_music are not connected")
        return io.NodeOutput(score_audio, score_music)


NODES = [MographScoreSkeleton, MographMusicStyleEncode, MographMusicFit, MographMusicRoute]
