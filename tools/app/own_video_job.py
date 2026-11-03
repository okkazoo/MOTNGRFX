#!/usr/bin/env python
"""Own-video JOB RUNNER (box side, ComfyUI venv python): one user clip + the READ brief -> a composed mp4 with our type over it.

    python tools/app/own_video_job.py --job ID --video /workspace/ComfyUI/input/clip.mp4 --brief-file brief.json --length 12 --seed 1 --quality final \
        --soundtrack video --out-dir /workspace/ComfyUI/output/motngrfx_app/own/ID --comfy-dir /workspace/ComfyUI --port 18188 [--client-id WS_ID]

Started by the route `/mograph/own_video` (tools/app/own_video_route.py) as a subprocess; the state (stage, log, prompt ids, outputs, error) is `<out-dir>/job.json`, rewritten at every
step, which the route's GET serves. This is the own-video route of 16_app/OWN_VIDEO.md (v003..v007: stretched clock, per-cut Alpha-Gen mattes inside the window only, ingest, pure type, route
defaults of `own_video_v2_retime_box.py`) as ONE runner. The scripts it extends (not copies): own_video.py / _v2 / _v3 / _v7 (spans, segments, cut script, Alpha-Gen graphs, redesign graph
builders, words prompt), own_video_v2_archive.py, own_video_v2_retime_box.py, ingest.py, tools/fusion/music_analysis.py, music_graph.t2m_chain (the approved ACE XL route).

Stages (job.json `stage`): queued, ingest, audio, music (soundtrack=generate only), analyse, words, prep, cut, matte, archive, design, parse, retime, render, finish, done (or error).
  ingest   the window of the clip is cut out (frame accurate), then `ingest.py` plans + renders it at the tier (source fps kept; integer-ratio fps policy; 9:16 crop / fit; real cuts detected)
  audio    the window's audio is the soundtrack (soundtrack=video, the old "keep"): wav for the analysis, the clip's own audio stream for the final mux;
           soundtrack=track (v19): the user's audio file (--audio) replaces the clip's sound: its first window-length seconds are the analysis wav and, with a 0.3 s fade-out, the final mux
  music    soundtrack=generate: ACE-Step XL base text-to-music (music_graph.t2m_chain + Master, the app's graph chain) for the window length from the brief's look / mood; it REPLACES the clip audio
  analyse  DSP beat grid of the soundtrack (music_analysis.analyse); words land on its beats
  words    brief.on_screen verbatim (upper case) on beat-snapped slots; no on_screen = Gemma picks the words from the concept (own_video_v7.words_prompt)
  prep     spans (plan_rules.cut_spans), segments per real cut, cut script, Alpha-Gen graphs, concept / direction (the brief's concept + look go into the design prompt), piece stub
  cut      ffmpeg: shot clips + per-piece clips (source frames, no fps filter)
  matte    Alpha-Gen per piece (cuts + mattes inside the window only), queued to ComfyUI
  archive  own_video_v2_archive.py -> the archive folder MographArchiveLoad reads
  design / parse   the redesign graph up to the model reply (Gemma 4 31B QAT), then the parse
  retime   own_video_v2_retime_box.py route defaults (hits on the beats, pure type, facefix layouts, transitions kept, outline on the busiest words, ink colours from the picture)
  render   the rest of the redesign graph (type passes, 3D cuts, compose, assemble) on the stretched 24 fps clock
  finish   relabel the frames to the source fps and mux the soundtrack (original audio stream or the generated track) -> final.mp4

EDIT MODE (2026-11-01, tools/app/overlay_mode.py): brief.edit / --edit "overlay" (default) | "redesign".
  overlay    the user's picture is NOT touched: every output frame is the matching source frame (source size, fps, frame count, order, real cuts as hard cuts) except where type is composited.
             The retime step runs the PICTURE LOCK (all cuts hard, no holds / crossfade / grade / grain / vignette / plate effects), the render runs at the source size, and `finish` writes
             output = the source frame outside the type, the composed frame inside it (overlay_mode.restore_video). Words land when they are sung (story._find_sung / _land_for on the
             READ's lyrics) else on a beat of the soundtrack, on the source frame grid (t = frame / src_fps).
  redesign   the old route (3D cuts, transitions, plate effects). It also keeps the source size, fps and frame count (no ingest tier crop, no fps halving).
  --size-mode tier  the historical tier route (ingest.py: 9:16 crop, 360x640 / 540x960, fps policy) for both edits.
  --scale full|half (2026-11-03, page Format "Original" | "Half scale"): half = compose, type and the final at half the source size (even sides; overlay_mode.scaled_size), same fps / frame count / edit /
             audio. The window clip and the Alpha-Gen input clips stay at the FULL source / matte size, so the matte keys (clip content + graph) are the same for both scales: a half job reuses the mattes a
             full job made and the other way round; only the shot clips, archive, retime, design and render keys follow the size. The tier route ignores it.
overlay_report.json in the out dir carries what tools/app/overlay_check.py needs.

STAGE CACHE (2026-11-01, "cancel + restart with reuse", tools/app/stage_cache.py): a stage whose inputs have not changed is not run again: its files come back from
<ComfyUI root>/motngrfx_cache/<stage>/<key>/ (beside output/, env MOTNGRFX_CACHE_DIR) as NEW files in this job folder (nothing of an earlier job is touched), `job.json` gets `cached: {stage: true}` and runner.log a
"cache hit <stage> <key>" line. Keys (content hashes; the page's words / fonts / seed are in a key only where the stage really reads them):
  ingest   source clip content + window + ffmpeg cut settings            audio    window id (clip) | the user's track content | the generated-music request graph
  analyse  soundtrack wav content + music_analysis code                  words    (Gemma branch only) the words graph; the rest of the stage is plain python
  cut      window id + soundtrack wav + the cut script                   matte    ONLY the matte clip's content + the Alpha-Gen graph (model, size, frames): no words, fonts, seed
  archive  seg clips + matte keys + segments / spans / piece / index     design   the design graph (prompt inputs, model, seed) + archive + wav + code
  parse    the parse graph (reply, seed) + archive + wav + code          retime   designed piece + words + segments + clip + matte keys + argv + code
  render   the render graph (piece_final, seed) + archive + wav + code   (finish always runs: it is the cheap restore / mux)
MOTNGRFX_NO_CACHE=1 turns it off.

Not in this runner (v003..v006 extras): clean-plate contrast moments / text-as-mask sections (Clean-Plate IC-LoRA jobs + the moments splice); facefix is run with --max-sections 0, so a grid that would
cover the face falls back to another layout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.app import overlay_mode as OM      # noqa: E402
from tools.app import stage_cache as SCH      # noqa: E402

STAGES = ["queued", "ingest", "audio", "music", "analyse", "words", "prep", "cut", "matte", "archive", "design", "parse", "retime", "render", "finish", "done"]
TIER_OF = {"still": "preview", "preview": "preview", "draft": "draft", "final": "draft"}      # Preview = 360x640 (Alpha-Gen 384x704); Final = the 540x960 render (Alpha-Gen 576x1024)
PIPE_FPS = 24
NATIVE_CUT_THR = 0.2      # ffmpeg scene score for a real cut in the source-size route; ingest's 0.35 missed the two soft cuts (scores 0.31 / 0.25) of a 3-shot LTX clip (box run R16), and the next-highest score there was 0.05
MUSIC_LEN = (8, 24)                                                                           # MographRequest.length range
CODE_GLOBS = ("nodes_fusion/*.py", "tools/fusion/**/*.py", "mograph/**/*.py", "mograph/fonts/*.json", "tools/app/own_video*.py", "tools/app/overlay_mode.py", "tools/app/llm_arms.py",
              "tools/app/build_graphs.py", "app/graphs/fusion_redesign.json")       # the code a design / parse / retime / render result depends on (part of those keys)


def matte_key(sc, clip, graph) -> str:
    """cache key of ONE Alpha-Gen matte: the content of the clip it is cut from + the Alpha-Gen graph (model, size, frames, the job's own file names as tokens). Nothing the page changes
    after the read (words, fonts, seed, direction) can enter it, so a word / font edit never redoes a matte."""
    return sc.key("matte", files=[clip], params=dict(graph=graph))


# ============================================================================================================================ pure helpers (unit-tested)
def read_brief(brief) -> dict:
    """READ JSON (object / string / None) -> {on_screen [str], concept, look, media [dict], music, notes}."""
    if isinstance(brief, str):
        try:
            brief = json.loads(brief) if brief.strip() else {}
        except ValueError:
            brief = {}
    b = dict(brief or {})
    on = b.get("on_screen") or []
    if isinstance(on, str):
        on = [x for x in on.splitlines()]
    lyr = [l for l in (b.get("lyrics") or []) if isinstance(l, dict) and str(l.get("text") or "").strip()] if isinstance(b.get("lyrics"), list) else []
    return dict(edit=OM.norm_edit(b.get("edit")), lyrics=lyr, on_screen=[" ".join(str(x).split()) for x in on if str(x).strip()], concept=" ".join(str(b.get("concept") or "").split()),
                look=" ".join(str(b.get("look") or "").split()), media=[m for m in (b.get("media") or []) if isinstance(m, dict)],
                music=b.get("music") if isinstance(b.get("music"), dict) else None, notes=str(b.get("notes") or ""),
                story=b.get("story") if isinstance(b.get("story"), dict) else None,
                direction=b["direction"].replace("\r\n", "\n").replace("\r", "\n").strip()[:2000] if isinstance(b.get("direction"), str) else "")            # the page's storyboard (its hits are the word slots, stage `words`)


def stamp_direction_json(piece_json: str, direction: str) -> str:
    """VIDEO MODE: the ONE place the user's typed "Words or direction" text lands on the piece (piece.direction + piece.user_style, tools/fusion/user_style.stamp); the retime box and the
    render read it from the piece. No typed text (or an unreadable piece) = piece_json unchanged. Lyrics / the storyboard's words are never passed here."""
    if not direction or not str(direction).strip():
        return piece_json
    from tools.fusion import user_style as US
    try:
        p = json.loads(piece_json)
    except (TypeError, ValueError):
        return piece_json
    return json.dumps(US.stamp(p, direction), ensure_ascii=False) if isinstance(p, dict) else piece_json


def video_entry(brief: dict) -> dict | None:
    for m in brief["media"]:
        if m.get("id") == "V1" or m.get("kind") == "video":
            return m
    return None


def window_of(brief: dict, clip_dur: float, length: float) -> tuple[float, float]:
    """(start_s, duration_s): media[V1].window clamped into the clip and to `length`; fallback = the first `length` seconds. Duration at least 3 s when the clip has it."""
    v = video_entry(brief)
    w = (v or {}).get("window")
    s, e = 0.0, float(length)
    try:
        if isinstance(w, (list, tuple)) and len(w) == 2 and float(w[1]) > float(w[0]) >= 0:
            s, e = float(w[0]), float(w[1])
    except (TypeError, ValueError):
        pass
    s = min(max(0.0, s), max(0.0, clip_dur - 3.0)) if clip_dur > 3.0 else 0.0
    dur = min(e - s, float(length), clip_dur - s)
    if dur < 3.0 and clip_dur - s >= 3.0:
        dur = min(float(length), clip_dur - s)
    return round(s, 3), round(max(dur, 0.0), 3)


SOUNDTRACKS = ("video", "track", "generate")
TRACK_FADE_S = 0.3                                                                            # the user's track ends with a short fade-out at the window's end


def norm_soundtrack(v) -> str:
    """"video" | "track" | "generate"; the old "keep" is "video"; anything else -> "video"."""
    s = str(v or "").strip().lower()
    s = "video" if s == "keep" else s
    return s if s in SOUNDTRACKS else "video"


def choose_soundtrack(requested: str, clip_has_audio: bool, track_ok: bool) -> tuple[str, str | None]:
    """the soundtrack the job really uses + a log note when the request could not be honoured.
    video: the clip's own audio (none -> new music); track: the user's audio file (none -> the clip's audio, else new music); generate: new music."""
    r = norm_soundtrack(requested)
    if r == "generate":
        return "generate", None
    if r == "track":
        if track_ok:
            return "track", None
        if clip_has_audio:
            return "video", "soundtrack track requested but no usable audio file: the clip's own audio is used"
        return "generate", "soundtrack track requested but no usable audio file, and the clip has no audio stream: soundtrack falls back to generate"
    if clip_has_audio:
        return "video", None
    return "generate", "the clip has no audio stream: soundtrack falls back to generate"


def track_wav_cmd(src, out, dur: float) -> list:
    """the user's audio file -> the analysis wav: PCM 48 kHz stereo, from the track's start, trimmed to `dur` seconds (a shorter track stays shorter: the mux pads it)."""
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-t", f"{dur:.3f}", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(out)]


def track_audio_filter(dur: float, fade: float = TRACK_FADE_S) -> str:
    """the mux filter for the user's track: pad a short track with silence up to the window length, fade out over the last `fade` s."""
    f = min(fade, max(0.0, dur))
    return f"apad=whole_dur={dur:.3f},afade=t=out:st={max(0.0, dur - f):.3f}:d={f:.3f}"


def track_mux_cmd(raw, track_wav, final, vf: list, dur: float) -> list:
    """final mux of the soundtrack=track job: the render's video + the user's track (already trimmed to the window) with the pad + fade-out filter, AAC 256k, exactly `dur` long."""
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(track_wav), "-map", "0:v:0", "-map", "1:a:0", *vf, "-af", track_audio_filter(dur),
            "-c:a", "aac", "-b:a", "256k", "-t", f"{dur:.4f}", "-movflags", "+faststart", str(final)]


def shortened_note(dur: float, length: float, tol: float = 0.25) -> str | None:
    """the clip / window is shorter than the requested Length: the output is as long as the footage (nothing is looped or frozen)."""
    if dur < float(length) - tol:
        return f"the video window is {dur:.1f} s, shorter than the requested length {float(length):g} s: the output is {dur:.1f} s (the footage is not looped or frozen)"
    return None


def slots_from_analysis(ana: dict, n: int, dur: float, first_min: float = 1.4, tail: float = 2.0, gaps=(3.2, 2.9, 2.6, 2.4)) -> list[dict]:
    """beat-snapped word slots from the DSP grid: [{land, hold_until}] (real seconds), at most n, spread over the window, each >= `gap` s after the last (form-in + 2-beat hold + exit),
    downbeats preferred; the gap shrinks (3.2 -> 2.4 s) before a word is dropped."""
    first_min, tail = min(first_min, 0.3 * dur), min(tail, 0.35 * dur)             # a short clip: the margins shrink with it
    bpm = float(ana.get("bpm") or 90.0)
    beat = 60.0 / bpm
    beats = [float(b) for b in ana.get("beats") or [] if first_min <= float(b) <= dur - tail]
    if not beats:
        t = first_min
        while t <= dur - tail:
            beats.append(round(t, 4))
            t += beat
    downs = {round(float(d), 3) for d in ana.get("downbeats") or []}

    def place(k: int, gap: float) -> list[float]:
        lo, hi = max(first_min, 1.4), max(first_min, dur - tail - 0.5)
        ideals = [0.38 * dur] if k == 1 else [lo + i * (hi - lo) / (k - 1) for i in range(k)]
        out: list[float] = []
        for ideal in ideals:
            cand = [b for b in beats if not out or b >= out[-1] + gap]
            if not cand:
                return out
            out.append(min(cand, key=lambda b: abs(b - ideal) - (0.35 * beat if round(b, 3) in downs else 0.0)))
        return out
    best: list[float] = []
    for gap in gaps:
        kmax = max(1, min(n, 1 + int((dur - tail - first_min) // gap)))
        got = place(kmax, gap)
        if len(got) > len(best):
            best = got
        if len(got) >= n:
            break
    out = []
    for i, t in enumerate(best):
        nxt = best[i + 1] - beat if i + 1 < len(best) else dur - 0.5
        out.append(dict(land=round(t, 4), hold_until=round(max(t + beat, min(t + 2 * beat, nxt)), 4)))
    return out


def clean_cuts(cuts: list[int], nframes: int, min_piece: int = 6) -> list[int]:
    """the real cuts that leave every piece at least `min_piece` frames (a detector hit on frame 1 or 2 would make a one-frame Alpha-Gen job)."""
    out: list[int] = []
    for c in sorted(set(int(x) for x in cuts)):
        if c - (out[-1] if out else 0) >= min_piece and nframes - c >= min_piece:
            out.append(c)
    return out


def engine_text(text) -> str:
    """a storyboard hit's text as ONE word of the type engine (<= 4 words / 26 letters, concept.MAX_LINE_WORDS / MAX_LINE_CHARS). A longer line (an older storyboard, a long hand-typed line) would be split by the
    design into word-per-hold fragments ("NEON / RAIN / ON"): it is cut to a sensible phrase first (story.trim_phrase: never ends on a function word)."""
    t = " ".join(str(text or "").upper().split())
    if len(t.split()) <= 4 and len(t) <= 26:
        return t
    from tools.fusion import story as ST
    return ST.trim_phrase(t, 4, 18, 26) or " ".join(t.split()[:4])


def shot_edges(cuts: list[int], nframes: int) -> list[int]:
    """the shot boundaries in source frames: [0, *real cuts, nframes]."""
    return [0] + [int(c) for c in sorted(set(cuts)) if 0 < int(c) < int(nframes)] + [int(nframes)]


def shot_of_frame(edges: list[int], frame: float) -> int:
    """the 0-based shot a source frame is in (the last shot takes frames up to and past the end)."""
    k = 0
    for i in range(len(edges) - 1):
        if edges[i] <= frame + 1e-6:
            k = i
    return k


def words_for(texts: list[str], slots: list[dict], K: float, cuts: list[int] | None = None, fps: float | None = None, nframes: int | None = None) -> list[dict]:
    """on-screen lines + slots -> the words.json of the route (pipeline-clock `land` / `hold_until` = real x K, `land_real`). With the real cuts (source frames), fps and frame count each word also
    carries the shot it lands in: `shot` (1-based), `shot_a` / `shot_b` (pipeline s = source frame / 24) and `cuts_p` (every cut, pipeline s): the retime box keeps the word inside that shot
    (own_video_v2_retime_box.py, WORD LIFETIME). The shot comes from the landing frame."""
    edges = shot_edges(cuts or [], nframes) if (nframes and fps) else None
    out = []
    for t, s in zip(texts, slots):
        d = dict(text=" ".join(str(t).upper().split()), land=round(s["land"] * K, 4), hold_until=round(s["hold_until"] * K, 4), land_real=round(s["land"], 4),
                 **({"land_frame": int(s["land_frame"])} if "land_frame" in s else {}))
        if edges:
            k = shot_of_frame(edges, s["land_frame"] if "land_frame" in s else float(s["land"]) * float(fps))
            d.update(shot=k + 1, shot_a=round(edges[k] / 24.0, 4), shot_b=round(edges[k + 1] / 24.0, 4), cuts_p=[round(c / 24.0, 4) for c in edges[1:-1]])
        out.append(d)
    return out


def music_request(brief: dict, words: list[str], length: int, seed: int) -> dict:
    """the MographRequest inputs of the generate-music graph: tags from the brief (look + the video's mood), scene = concept, words = the on-screen lines."""
    v = video_entry(brief) or {}
    parts = [p for p in (str(v.get("mood") or "").strip(), brief["look"]) if p]
    tags = ", ".join(parts)[:240]
    return dict(words="\n".join(words) or "MOTNGRFX", scene=(brief["concept"] or str(v.get("shows") or ""))[:300], music=tags, length=max(MUSIC_LEN[0], min(MUSIC_LEN[1], int(round(length)))),
                look="auto", seed=int(seed), quality="preview", format="9:16")


def music_graph_for(mreq: dict, prefix: str, seed: int) -> dict:
    """the generate-soundtrack graph: MographRequest -> the app's ACE XL base t2m chain (music_graph.t2m_chain) -> Master -> SaveAudioAdvanced (flac)."""
    from tools.fusion import music_graph
    g = {"request": {"class_type": "MographRequest", "inputs": dict(mreq), "_meta": {"title": "Request"}}}
    g.update(music_graph.t2m_chain(["request", 0], int(seed), None, p="t2m_"))
    g["master"] = {"class_type": "MographMaster", "inputs": {"audio": ["t2m_tame", 0], "music": ["t2m_cut", 1], "preset": "auto", "target_lufs": -14.0, "width": 1.25, "mono_safe": True},
                   "_meta": {"title": "Master"}}
    g["out_music"] = {"class_type": "SaveAudioAdvanced", "inputs": {"audio": ["master", 0], "filename_prefix": prefix, "format": "flac"}, "_meta": {"title": "OUT music"}}
    return g


OVERLAY_DIRECTION = ("NO 3D cut and no picture effect: the user's picture and edit stay exactly as they are, the type is only drawn over (and, where it helps, behind) the footage")


def direction_text(direction4: str, test_clip: str, brief: dict, shows: str, hits: str, n_shots: int, soundtrack: str, edit: str = "redesign") -> str:
    """V3.DIRECTION4 with the test clip context swapped for this clip + the user's brief (braces in user text cannot break str.format)."""
    ctx = f"The clip is the user's OWN footage ({shows or 'the user video'}), {n_shots} shots, set to " + {"video": "its own soundtrack", "keep": "its own soundtrack", "track": "the user's own audio track as its soundtrack"}.get(soundtrack, "a generated instrumental soundtrack") + ". "
    if brief["concept"]:
        ctx += f"USER BRIEF, concept: {brief['concept']} "
    if brief["look"]:
        ctx += f"Look and feel to honour: {brief['look']}. "
    if brief["on_screen"]:
        ctx += "The words of the type are the user's own and fixed: they are given as the piece's words, use them exactly as written. "
    tmpl = direction4.replace(test_clip, "{ctx}") if test_clip in direction4 else "{ctx}" + direction4
    txt = tmpl.format(ctx=ctx, hits=hits)
    if OM.norm_edit(edit) == "overlay":
        txt = txt.replace("at most one designed 3D cut", OVERLAY_DIRECTION)
    return txt


def retime_argv(python, root, w, out_dir, stem, bpm_pipe: float, seed: int, edit: str, nat: dict | None, matte: tuple) -> list:
    """the command of the retime step (tools/app/own_video_v2_retime_box.py). `--edit overlay` runs the PICTURE LOCK (no 3D cut, no footage-touching op); a source-size job also hands it the sizes and the
    frame aspect. NO `--cut3d` is passed in either mode: overlay makes every cut hard in the lock, redesign leaves the design's cuts."""
    w, root = Path(w), Path(root)
    glob_clip = w / "seg" / "global_clip.mp4"
    return [python, root / "tools/app/own_video_v2_retime_box.py", "--piece", w / "piece_designed.json", "--words", w / "words.json", "--out", w / "piece_final.json",
            "--report", w / "retime_report.json", "--bpm", f"{bpm_pipe}", "--transitions", "keep", "--pure-type", "1", "--layouts", "facefix", "--max-sections", "0", "--glow", "off",
            "--outline", "3", "--outline-share", "0.34", "--busy-clip", glob_clip, "--colors", glob_clip, "--still-palette", root / "tools/fusion/still_palette.py",
            "--stable-type", root / "tools/fusion/stable_type.py", "--work-dir", w, "--alpha-dir", out_dir, "--alpha-stem", stem, "--color-seed", str(seed),
            "--edit", OM.norm_edit(edit), *(["--native-size", f"{nat['compose'][0]}x{nat['compose'][1]}", "--matte-size", f"{matte[0]}x{matte[1]}", "--aspect", f"{nat.get('type_aspect', nat['aspect']):.6f}"] if nat else [])]


def track_needs_loop(track_dur: float, dur: float, tol: float = 0.25) -> bool:
    """the user's track is shorter than the clip window: it is looped at a downbeat (tools/fusion/audio_cut.cut_to_downbeat pad=True) instead of ending early."""
    return float(track_dur) < float(dur) - tol


def native_plan(pr: dict, trimmed_frames: int, cuts_s: list, IG) -> dict:
    """the ingest-shaped plan of the SOURCE-size route: size = the display size (even), fps = the stream's own rate (no halving, no VFR conversion), frames = what the window clip has,
    shots from the cuts. No crop, no fit, no tier."""
    vfr = bool(pr.get("vfr"))
    rate = IG.snap_rate(pr["avg_frame_rate"] if vfr else pr["r_frame_rate"])
    fo = float(rate)
    f0s = [0] + [int(round(float(c) * fo)) for c in cuts_s if 0 < float(c) * fo < trimmed_frames]
    shots = [dict(index=i, f0=f, mode="native") for i, f in enumerate(sorted(set(f0s)))]
    cp = (pr.get("audio") or {}).get("codec") in IG.MP4_AUDIO
    return dict(fps=dict(src=IG.fmt_rate(rate), fps=IG.fmt_rate(rate), fps_float=round(fo, 5), policy="keep", step=1, vfr=vfr),
                frames=int(trimmed_frames), shots=shots, size=[OM.even(pr["display_w"]), OM.even(pr["display_h"])], audio=dict(copy=cp) if pr.get("audio") else None)


# ============================================================================================================================ ComfyUI client
PREVIEW_METHOD = os.environ.get("MOTNGRFX_PREVIEW_METHOD", "auto")   # "none" turns the live previews off; extra_data.preview_method: ComfyUI streams sampler preview frames (binary ws messages) to the page's client id, so the web page's Build view shows each shot refining step by step

class Comfy:
    def __init__(self, port: int, client_id: str, job: "Job"):
        self.base, self.cid, self.job = f"http://127.0.0.1:{int(port)}", client_id or uuid.uuid4().hex, job

    def _get(self, path: str, timeout: float = 30):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def submit(self, graph: dict, label: str) -> str:
        req = urllib.request.Request(self.base + "/prompt", data=json.dumps({"prompt": graph, "client_id": self.cid, "extra_data": {"preview_method": PREVIEW_METHOD, **({"motngrfx_strip_preview": True} if PREVIEW_METHOD != "none" else {})}}).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                pid = json.loads(r.read().decode("utf-8"))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI (port {self.base.rsplit(':', 1)[-1]}) rejected the {label} graph: HTTP {e.code} {e.reason}: {e.read().decode('utf-8', 'replace')[:900]}") from None
        self.job.state["prompt_ids"].append(pid)
        self.job.log(f"queued {label}: {pid}")
        return pid

    def wait(self, pid: str, timeout: float, label: str) -> dict:
        t_end, gone = time.time() + timeout, 0
        while time.time() < t_end:
            try:
                h = self._get(f"/history/{pid}").get(pid)
            except (urllib.error.URLError, OSError):
                h = None
            if h:
                st = (h.get("status") or {})
                if st.get("status_str") == "error" or st.get("completed") is False:
                    msg = ""
                    for m in st.get("messages") or []:
                        if m and m[0] == "execution_error":
                            d = m[1]
                            msg = f"{d.get('node_type')} ({d.get('node_id')}): {str(d.get('exception_message'))[:600]}"
                    raise RuntimeError(f"{label} failed in ComfyUI: {msg or st}")
                if st.get("status_str") == "success" or st.get("completed"):
                    return h
            else:
                try:
                    q = self._get("/queue")
                    live = any(pid in json.dumps(x) for x in (q.get("queue_running") or []) + (q.get("queue_pending") or []))
                    if any(pid in json.dumps(x) for x in q.get("queue_pending") or []):
                        t_end = time.time() + timeout                                   # waiting behind other work does not count: the clock starts when it runs
                except (urllib.error.URLError, OSError, ValueError):
                    live = True
                gone = 0 if live else gone + 1
                if gone > 15:
                    raise RuntimeError(f"{label}: prompt {pid} left the queue without a result")
            time.sleep(2.0)
        raise RuntimeError(f"{label}: no result after {timeout:.0f} s")

    @staticmethod
    def texts(hist: dict) -> dict:
        out = {}
        for nid, o in (hist.get("outputs") or {}).items():
            t = o.get("text")
            if t is not None:
                out[nid] = "".join(t) if isinstance(t, list) else str(t)
        return out

    @staticmethod
    def compute_s(hist: dict) -> float | None:
        ts = {m[0]: m[1].get("timestamp") for m in (hist.get("status") or {}).get("messages") or [] if m and isinstance(m[1], dict)}
        if ts.get("execution_start") and ts.get("execution_success"):
            return round((ts["execution_success"] - ts["execution_start"]) / 1000.0, 1)
        return None


# ============================================================================================================================ job state
class StopAfter(Exception):
    """--stop-after STAGE (a smoke test of the early stages): the run ends cleanly when that stage is finished."""


class Job:
    def __init__(self, a):
        self.a = a
        self.dir = Path(a.out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.work = self.dir / "work"
        self.work.mkdir(exist_ok=True)
        old = {}
        try:
            old = json.loads((self.dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.state = dict(job=a.job, stage="queued", stage_n=0, stages=STAGES, log=[], prompt_ids=[], outputs=[], error=None, done=False, times={}, cached={}, pid=os.getpid(),
                          started=old.get("started") or time.time(), params=dict(quality=a.quality, tier=TIER_OF[a.quality], soundtrack=norm_soundtrack(a.soundtrack), length=a.length, seed=a.seed, edit=OM.norm_edit(getattr(a, "edit", "")),
                                                                                                                size_mode=getattr(a, "size_mode", "native")))
        self._t, self._stage = time.time(), "queued"
        self.save()

    def save(self) -> None:
        p = self.dir / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.state["log"] = (self.state["log"] + [line])[-300:]
        self.save()

    def mark_cached(self, stage: str) -> None:
        """the page's build view shows a small "reused" tag on this stage."""
        self.state.setdefault("cached", {})[stage] = True
        self.save()

    def stage(self, name: str, note: str = "") -> None:
        if getattr(self.a, "stop_after", "") and self._stage == self.a.stop_after:
            raise StopAfter(self._stage)
        now = time.time()
        self.state["times"][self._stage] = round(self.state["times"].get(self._stage, 0.0) + now - self._t, 1)
        self._t, self._stage = now, name
        self.state["stage"], self.state["stage_n"] = name, STAGES.index(name) if name in STAGES else self.state["stage_n"]
        self.log(f"== {name} {note}".rstrip())

    def sh(self, argv: list[str], what: str, env: dict | None = None, timeout: float = 3600) -> str:
        p = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
        if p.returncode != 0:
            raise RuntimeError(f"{what} failed ({p.returncode}): {(p.stderr or p.stdout)[-900:]}")
        return p.stdout


# ============================================================================================================================ the runner
def ffprobe(path, *entries, stream: str = "v:0") -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=" + ",".join(entries) + ":format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    try:
        j = json.loads(out)
    except ValueError:
        return {}
    s = (j.get("streams") or [{}])
    return dict(s[0] if s else {}, duration=float((j.get("format") or {}).get("duration") or 0.0))


def count_frames(path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return int(out.split()[0]) if out else 0


def run(a) -> int:
    job = Job(a)
    try:
        return _run(job, a)
    except StopAfter as e:
        job.state["done"] = True
        job.log(f"stopped after stage {e} (--stop-after)")
        return 0
    except Exception as e:   # noqa: BLE001 - the state file carries the error to the page
        import traceback
        traceback.print_exc()
        job.state["error"], job.state["stage"] = f"{type(e).__name__}: {e}"[:1500], "error"
        job.log(f"ERROR in stage {job._stage}: {e}")
        return 1


def _run(job: Job, a) -> int:
    from tools.app import own_video as V1, own_video_v2 as V2, own_video_v3 as V3, own_video_v7 as V7
    comfy_dir = Path(a.comfy_dir)
    out_root = comfy_dir / "output"
    rel = Path(a.out_dir).resolve().relative_to(out_root.resolve()).as_posix()           # ComfyUI-relative prefix of this job's files
    input_dir = comfy_dir / "input"
    brief = read_brief(json.loads(Path(a.brief_file).read_text(encoding="utf-8")) if a.brief_file else {})
    tier = TIER_OF[a.quality]
    size = V3.SIZES[tier]
    edit = OM.norm_edit(a.edit) if getattr(a, "edit", "") else brief["edit"]            # --edit wins, else the brief's, else overlay
    native = getattr(a, "size_mode", "native") == "native"
    scale = OM.norm_scale(getattr(a, "scale", "full")) if native else "full"          # "half" = compose, type and final at half the source size (even sides); the tier route has no scale
    job.state["params"].update(edit=edit, size_mode="native" if native else "tier", scale=scale)
    job.log(f"edit mode: {edit}; size mode: {'source size / source fps' if native else 'tier (ingest crop)'}; scale: {scale}")
    if not native and OM.norm_scale(getattr(a, "scale", "full")) == "half":
        job.log("WARNING: scale half is ignored in the tier size mode (the tier sets the size)")
    cc = Comfy(a.port, a.client_id, job)
    try:
        cc._get("/queue", timeout=10)
    except Exception as e:   # noqa: BLE001
        raise RuntimeError(f"ComfyUI is not reachable on port {a.port} ({type(e).__name__}: {e})") from None
    w = job.work
    STEM = f"motngrfx_own_{a.job.replace('-', '')}"
    seed = int(a.seed)
    SC = SCH.JobCache(SCH.default_cache_root(comfy_dir), Path(a.out_dir), input_dir, a.job, STEM, rel, log=job.log, mark=job.mark_cached)

    def CF(*extra):                                                # fingerprint of the code a result depends on (only computed while the cache is on)
        return SCH.code_fingerprint(ROOT, CODE_GLOBS + tuple(extra)) if SC.on else ""

    # ------------------------------------------------------------------------------------------------ ingest
    job.stage("ingest", f"window of {a.video}, tier {tier}")
    src = Path(a.video)
    pr = ffprobe(src, "width", "height", "r_frame_rate")
    clip_dur = pr.get("duration") or 0.0
    if clip_dur <= 0:
        raise RuntimeError(f"{src.name}: ffprobe cannot read the clip")
    start, dur = window_of(brief, clip_dur, a.length)
    if dur < 1.5:
        raise RuntimeError(f"window too short ({dur:.1f} s of a {clip_dur:.1f} s clip)")
    job.log(f"clip {clip_dur:.2f} s, window {start:.2f}-{start + dur:.2f} s ({dur:.2f} s), length {a.length}")
    note = shortened_note(dur, a.length)
    if note:
        job.log("NOTE: " + note)
    job.state["info"] = dict(requested_length=a.length, length=round(dur, 3), shortened=bool(note), note=note)
    trimmed = w / "window.mp4"
    nat = None
    if native:
        # SOURCE-SIZE route (2026-11-01, user: "use the video's resolution and framerate"): no ingest tier, no 9:16 crop / fit, no fps halving. The window is cut frame-exact and (near) lossless at the
        # source's own size and rate; the engine runs at that size, only the Alpha-Gen matte is made smaller and upscaled back.
        from tools.app import ingest as IG
        pr0 = IG.probe(str(src))
        est_px = pr0["display_w"] * pr0["display_h"] * dur * max(float(IG.snap_rate(pr0["r_frame_rate"])), 1.0)
        crf = "0" if est_px <= 1920 * 1080 * 30 * 30 else "8"                      # lossless up to 30 s of 1080p30, else visually lossless (disk)
        vfpre = ",".join(IG._pre_filters(pr0) + (["scale=trunc(iw/2)*2:trunc(ih/2)*2"] if (pr0["display_w"] % 2 or pr0["display_h"] % 2) else [])) or "null"
        aud = ["-c:a", "copy"] if (pr0.get("audio") or {}).get("codec") in IG.MP4_AUDIO else ["-c:a", "aac", "-b:a", "256k"]
        k_win = SC.key("ingest", ids=[SCH.file_hash(src) if SC.on else ""], params=dict(start=start, dur=dur, vf=vfpre, crf=crf, aud=aud, preset="fast"))
        if SC.restore("ingest", k_win) is None:
            job.sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", src, "-vf", vfpre, "-c:v", "libx264", "-crf", crf, "-preset", "fast", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    *aud, trimmed], "window cut")
            SC.store("ingest", k_win, [trimmed])
        cuts_s = IG.detect_cuts(str(trimmed), NATIVE_CUT_THR)
        nfr = count_frames(trimmed)
        ing = native_plan(pr0, nfr, cuts_s, IG)
        nat = OM.native_sizes(ing["size"][0], ing["size"][1], tier, scale)          # matte size from the FULL source (a half job cuts the same Alpha-Gen clips as a full one); compose / type / final from the scale
        size = nat["matte"]
        clip = trimmed
        pw = ffprobe(trimmed, "width", "height")
        if (int(pw.get("width") or 0), int(pw.get("height") or 0)) != tuple(nat["source"]):          # the window clip is always cut at the full source size; only the shot clips / compose are scaled
            raise RuntimeError(f"window clip is {pw.get('width')}x{pw.get('height')}, expected the source size {nat['source']}")
        fps = float(ing["fps"]["fps_float"])
        nframes = nfr
        cuts = clean_cuts([int(s["f0"]) for s in ing["shots"][1:]], nframes)
        job.log(f"source size {nat['source'][0]}x{nat['source'][1]} at {ing['fps']['fps']} fps kept ({nframes} frames), matte size {size[0]}x{size[1]} (resampled to the output size), format {nat['format']}")
        job.log(f"scale {scale}: output size {nat['compose'][0]}x{nat['compose'][1]} (type canvas {nat['type'][0]}x{nat['type'][1]})" + (" - half of the source, same fps and frame count, the matte clips are the full job's" if scale == "half" else ""))
    else:
        plan_p = w / "ingest.json"
        job.sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", src, "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                "-c:a", "aac", "-b:a", "256k", trimmed], "window cut")
        job.sh([sys.executable, ROOT / "tools/app/ingest.py", "plan", trimmed, "--cuts", "detect", "--tier", tier, "-o", plan_p], "ingest plan")
        ing = json.loads(plan_p.read_text(encoding="utf-8"))
        ing_dir = w / "ingested"
        ing_dir.mkdir(exist_ok=True)
        job.sh([sys.executable, ROOT / "tools/app/ingest.py", "render", plan_p, "--tier", tier, "-o", ing_dir], "ingest render")
        clip = next(iter(sorted(ing_dir.glob("*_ingest_*.mp4"))), None)
        if clip is None:
            raise RuntimeError("ingest produced no clip")
        fps = float(ing["fps"]["fps_float"])
        nframes = count_frames(clip) or int(ing["frames"])
        cuts = clean_cuts([int(s["f0"]) for s in ing["shots"][1:]], nframes)
    clip_id = k_win if native else (SCH.file_hash(clip) if SC.on else "")          # the identity of the window clip: later keys name it, they never re-hash it
    K = V3.K_of(fps)
    real_dur = nframes / fps
    job.log(f"ingested {clip.name}: {nframes} frames at {ing['fps']['fps']} fps ({ing['fps']['policy']}), {len(cuts)} real cuts {cuts}, K {K:.4f}, " + ("matte size" if native else "shot size") + f" {size[0]}x{size[1]}")

    # ------------------------------------------------------------------------------------------------ audio / music
    wav = w / "music.wav"
    has_audio = bool(ffprobe(clip, "codec_name", stream="a:0").get("codec_name"))
    track_src = Path(a.audio) if getattr(a, "audio", "") else None
    track_ok = bool(track_src and track_src.is_file() and ffprobe(track_src, "codec_name", stream="a:0").get("codec_name"))
    soundtrack, why = choose_soundtrack(a.soundtrack, has_audio, track_ok)
    if why:
        job.log(why)
    gen_audio = None
    if soundtrack == "video":
        job.stage("audio", "the clip's own audio is the soundtrack")
        k_aud = SC.key("audio", ids=[clip_id], params=dict(kind="video"))
        if SC.restore("audio", k_aud) is None:
            job.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", wav], "audio extract")
            SC.store("audio", k_aud, [wav])
    elif soundtrack == "track":
        job.stage("audio", f"the user's audio file {track_src.name} replaces the clip's sound (first {real_dur:.1f} s)")
        tdur = ffprobe(track_src, "codec_name", stream="a:0").get("duration") or 0.0
        k_aud = SC.key("audio", files=[track_src] if SC.on else [], params=dict(kind="track", dur=round(real_dur, 4), loop=track_needs_loop(tdur, real_dur), code=CF()))
        if SC.restore("audio", k_aud) is not None:
            pass
        elif track_needs_loop(tdur, real_dur):
            from tools.fusion import audio_cut as AC
            import soundfile as sf
            y, sr_, mus = AC.cut_file(track_src, cut=real_dur, pad=True)                 # the track is shorter than the clip: looped at a downbeat (equal-power crossfade), exactly the window long
            sf.write(str(wav), y.T, sr_)
            job.log(f"the track is {tdur:.1f} s, the window {real_dur:.1f} s: looped ({(mus.get('loop') or {}).get('passes', 1)} passes, downbeat grid {bool((mus.get('loop') or {}).get('grid'))})")
            SC.store("audio", k_aud, [wav])
        else:
            job.sh(track_wav_cmd(track_src, wav, real_dur), "track extract")
            SC.store("audio", k_aud, [wav])
    else:
        job.stage("music", "generated soundtrack replaces the clip audio (ACE-Step XL base, music_graph.t2m_chain)")
        mreq = music_request(brief, brief["on_screen"], real_dur, seed)
        g = music_graph_for(mreq, f"{rel}/music", seed)
        job.log(f"music request: {json.dumps(mreq, ensure_ascii=False)}")
        k_mus = SC.key("music", params=dict(graph=music_graph_for(mreq, "music", seed), dur=round(real_dur, 3)))
        got_m = SC.restore("music", k_mus)
        if got_m is not None:                                      # the same request + seed + model: the soundtrack made before comes back as a new file
            job.state["outputs"].append(dict(kind="audio", file=got_m["meta"]["audio_name"]))
        else:
            h = cc.wait(cc.submit(g, "music"), 1800, "music")
            files = (h.get("outputs") or {}).get("out_music", {}).get("audio") or []
            if not files:
                raise RuntimeError("the music graph saved no audio")
            gen_audio = out_root / files[0].get("subfolder", "") / files[0]["filename"]
            shutil.copy(gen_audio, w / "music.flac")
            job.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", w / "music.flac", "-t", f"{real_dur:.3f}", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", wav], "music to wav")
            job.state["outputs"].append(dict(kind="audio", file=Path(gen_audio).name))
            job.state["times"]["music_compute"] = cc.compute_s(h)
            job.log(f"generated {gen_audio.name} ({cc.compute_s(h)} s on the GPU)")
            SC.store("music", k_mus, [gen_audio, w / "music.flac", wav], meta=dict(audio_name=Path(gen_audio).name))

    # ------------------------------------------------------------------------------------------------ analyse
    job.stage("analyse", "DSP beat grid of the soundtrack")
    from tools.fusion import music_analysis
    wav_id = SCH.file_hash(wav) if SC.on else ""                   # the soundtrack's content: the id every later key that reads the music names
    k_ana = SC.key("analyse", ids=[wav_id], params=dict(fps=PIPE_FPS, n=4, code=SCH.code_fingerprint(ROOT, ("tools/fusion/music_analysis.py",)) if SC.on else ""))
    if SC.restore("analyse", k_ana) is not None:
        ana = json.loads((w / "analysis.json").read_text(encoding="utf-8"))          # the file is the analysis without its curves; nothing after this step reads the curves
    else:
        ana = music_analysis.analyse(str(wav), None, 4, PIPE_FPS)
        (w / "analysis.json").write_text(json.dumps({k: v for k, v in ana.items() if k != "curves"}), encoding="utf-8")
        SC.store("analyse", k_ana, [w / "analysis.json"])
    bpm = float(ana["bpm"])
    key = ana["key"]["estimate"] if isinstance(ana.get("key"), dict) else ana.get("key", "")
    job.log(f"soundtrack: {bpm:.1f} bpm, {key}, {len(ana['beats'])} beats, downbeats {ana['downbeats'][:6]}")

    # ------------------------------------------------------------------------------------------------ words
    job.stage("words", f"{len(brief['on_screen'])} on-screen lines")
    texts = list(brief["on_screen"])
    n_want = len(texts) or 4
    hits = [h for h in ((brief.get("story") or {}).get("hits") or []) if str(h.get("text") or "").strip() and 0.0 <= float(h.get("land", -1)) < real_dur - 0.3]
    if hits:                                                       # the storyboard placed the words (story.hits, real s): what the cards showed is what renders
        texts = [engine_text(h["text"]) for h in hits]
        if texts != [" ".join(str(h["text"]).upper().split()) for h in hits]:
            job.log(f"words cut to a sensible phrase for the type engine: {[(h['text'], t) for h, t in zip(hits, texts) if t != ' '.join(str(h['text']).upper().split())]}")
        slots = [dict(land=float(h["land"]), hold_until=min(real_dur - 0.2, max(float(h.get("hold_until") or 0), float(h["land"]) + 0.7))) for h in hits]
        job.log(f"words from the storyboard hits: {[(x['text'], x['land']) for x in hits]}")
    else:
        slots = slots_from_analysis(ana, n_want, real_dur)
    if not slots:
        raise RuntimeError("no word slot fits the window")
    if len(slots) < len(texts):
        job.log(f"WARNING: only {len(slots)} of {len(texts)} on-screen lines fit the window ({real_dur:.1f} s, bpm {bpm:.0f}); dropped: {texts[len(slots):]}")
        texts = texts[:len(slots)]
    asr = OM.asr_from_lyrics(brief["lyrics"]) if (brief["lyrics"] and soundtrack != "generate" and not hits) else []
    if asr and texts:                                              # the audio has vocals that match: each line lands when its first word is SUNG (story._find_sung / _land_for), else on its beat slot
        sung, notes_s = OM.sung_hits(texts, slots[:len(texts)], asr, ana["beats"], real_dur, start if soundtrack == "video" else 0.0)
        for n_ in notes_s:
            job.log("lyrics: " + n_)
        if sung:
            texts, slots = [h["text"] for h in sung], [dict(land=h["land"], hold_until=h["hold_until"]) for h in sung]
            job.log("words landed on the sung time where the lyrics match: " + ", ".join(f"{t}@{s['land']:.2f}" for t, s in zip(texts, slots)))
    if not texts:                                                  # no on_screen: Gemma picks the words from the concept (own_video_v7.words_prompt)
        shows = (video_entry(brief) or {}).get("shows") or ""
        reading = dict(arc=brief["look"], sections=[dict(start=0.0, end=real_dur, label="track", energy="")], hit_moments=[])
        theme = " ".join(x for x in (brief["concept"], shows, f"Look: {brief['look']}" if brief["look"] else "") if x) or "the user's video"
        for s in slots:
            s["section"] = "track"
        g = V1.words_graph(V7.words_prompt(slots, reading, theme), seed)
        k_wd = SC.key("words", params=dict(graph=g))
        if SC.restore("words", k_wd) is not None:
            words_reply = (w / "words_reply.txt").read_text(encoding="utf-8")
        else:
            h = cc.wait(cc.submit(g, "words"), 900, "words")
            words_reply = cc.texts(h)["out_reply"]
            (w / "words_reply.txt").write_text(words_reply, encoding="utf-8")
            SC.store("words", k_wd, [w / "words_reply.txt"])
        lines = V7.parse_words(words_reply, slots)
        texts = [x["text"] for x in lines]
        job.log(f"Gemma words: {texts}")
    if texts:
        slots = OM.quantise_slots(slots, fps, nframes)             # every landing on a SOURCE frame: t = frame / src_fps (frame f on the pipeline clock is f / 24 s = f / fps real s x K)
    words = words_for(texts, slots, K, cuts, fps, nframes)
    job.log("words (real s): " + ", ".join(f"{x['text']}@{x['land_real']:.2f}" for x in words))

    # ------------------------------------------------------------------------------------------------ prep
    job.stage("prep", "spans, segments, cut script, Alpha-Gen graphs")
    audio_x = w / "audio_stretched.m4a"
    V1.BOX_IN = str(input_dir)
    V1.SRC = str(wav)                                              # the stretched-audio step of the cut script reads this
    V3.SIZE, V3.QUALITY = size, tier
    V3.CLIP_SIZE, V3.GLOBAL_SIZE, V3.MATTE_FPS, V3.FORMAT = None, None, None, "9:16"          # module state is reset per job (the tier route is the historical one)
    if native:                                                     # shot clips = the source frames; Alpha-Gen clips stretched to the matte size, tagged with the SOURCE rate (LTX frame_rate); analysis clip small
        long_ = 640.0 / max(nat["source"])                         # the analysis clip is sized from the SOURCE, so it is the same for both scales
        V3.CLIP_SIZE, V3.MATTE_FPS, V3.FORMAT = tuple(nat["compose"]), ing["fps"]["fps"], nat["format"]
        V3.GLOBAL_SIZE = (OM.even(nat["source"][0] * long_), OM.even(nat["source"][1] * long_)) if long_ < 1.0 else tuple(nat["source"])
    V3.BOX = V3.BOX_DIR = str(w)
    V3.SRC_BOX = str(clip)
    V3.H_NAME = f"{STEM}_h"
    V3.VERSION, V3.STEM, V3.ARCHIVE = "own", STEM, f"{rel}/archive"
    V3.AUDIO_X = str(audio_x) if abs(K - 1.0) > 1e-6 else str(wav)
    V3.JOIN_REAL_CUTS = True
    V3.patch_v2()
    if soundtrack == "generate":
        V2.CHARACTER_FIND = "NO-MATCH-KEEP"                        # the generated track IS "generated instrumental music": the design prompt's line is true
    V2.SEG_LABELS = [("subject", True)] * (len(cuts) + 1)
    from tools.fusion import plan_rules as PR
    music = V3.music_x(ana, K, nframes)
    spans = PR.cut_spans(nframes, PR.make_grid(music, music["cut"]), V3.PIPE_FPS)
    segs = V2.segments(cuts, nframes, spans)
    V2._write(str(w / "spans.json"), dict(spans=spans, starts=[sum(spans[:i]) for i in range(len(spans))], src_fps=fps, K=K))
    V2._write(str(w / "segments.json"), segs)
    V2._write(str(w / "box_cut_v3.sh"), V3.box_script(spans, segs, fps), js=False)
    alpha = {}
    for s in segs:
        if s["person"]:
            g = V3.alpha_graph(s["job"], s["na"])
            for nid in [k for k, v in g.items() if v["class_type"] == "SaveVideo"]:
                del g[nid]                                         # the PNG stack is all the archive needs
            for v in g.values():
                if v["class_type"] == "SaveImage":
                    v["inputs"]["filename_prefix"] = f"{rel}/{STEM}_alpha_h{s['job']}_image"
            alpha[s["job"]] = g
    shows = (video_entry(brief) or {}).get("shows") or brief["concept"] or "the user's video"
    c = V3.concept(words)
    c["length"] = int(round(nframes / V3.PIPE_FPS))
    c["scene"] = " ".join(x for x in (shows, brief["concept"]) if x)[:400]
    c["seed"] = seed
    if native:
        c["format"] = nat["format"]
    c["idea"] = " ".join(x for x in (brief["concept"], f"Look: {brief['look']}" if brief["look"] else "") if x)[:300]
    c["music"] = (f"the clip's own audio track ({bpm:.0f} bpm, {key}), already recorded, not generated" if soundtrack == "video"
                  else f"the user's own audio track ({bpm:.0f} bpm, {key}), already recorded, not generated" if soundtrack == "track"
                  else f"generated instrumental music ({bpm:.0f} bpm, {key}): {music_request(brief, texts, real_dur, seed)['music'] or 'by mood'}")
    lyr = ("Lyrics: not transcribed (the track is the clip's own audio); the on-screen words are the user's own." if soundtrack == "video"
           else "Lyrics: not transcribed (the track is the user's own audio file); the on-screen words are the user's own." if soundtrack == "track" else "Lyrics: none (instrumental).")
    test_clip = ("The clip is the user's own music video: a rapper in a New York test clip station and at a red-shuttered corner, cut with inserts "
              "(train, graffiti, street, a gold chain close-up, a crowd). ")
    hits = ", ".join(f"{x['text']} {x['land']:.2f}" for x in words)
    direction = direction_text(V3.DIRECTION4, test_clip, brief, shows, hits, len(spans), soundtrack, edit)
    piece, index = V3.piece_stub(ana, music, "\n".join(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT"][:len(spans)]), nframes)
    V2._write(str(w / "concept.json"), c)
    V2._write(str(w / "words.json"), words)
    V2._write(str(w / "lyrics_line.txt"), lyr, js=False)
    V2._write(str(w / "direction.txt"), direction, js=False)
    V2._write(str(w / "piece.json"), piece)
    V2._write(str(w / "index.json"), index)
    job.log(f"spans {spans} ({len(spans)} shots), {len(alpha)} Alpha-Gen pieces, pipeline clock x{K:.3f}")

    # ------------------------------------------------------------------------------------------------ cut
    job.stage("cut", "ffmpeg shot clips")
    k_cut = SC.key("cut", ids=[clip_id, wav_id], texts=[(w / "box_cut_v3.sh").read_text(encoding="utf-8")] if SC.on else [])
    if SC.restore("cut", k_cut) is None:
        job.sh(["bash", w / "box_cut_v3.sh"], "cut script")
        SC.store("cut", k_cut, [w / "seg", w / "audio_stretched.m4a", *sorted(input_dir.glob(f"{V3.H_NAME}*.mp4"))])

    # ------------------------------------------------------------------------------------------------ matte
    job.stage("matte", f"Alpha-Gen x{len(alpha)} (window only)")
    matte_ids, todo = {}, {}
    for j, g in alpha.items():                                     # each piece is reused on its own: only the matte clips whose picture changed are cut out again
        matte_ids[j] = matte_key(SC, input_dir / f"{V3.H_NAME}{j}.mp4", g)
        if SC.restore("matte", matte_ids[j], mark=False) is None:
            todo[j] = g
    if alpha and not todo and SC.on:
        job.mark_cached("matte")
    pids = {j: cc.submit(g, f"alpha h{j}") for j, g in todo.items()}
    for k, (j, pid) in enumerate(pids.items(), 1):
        h = cc.wait(pid, 1800, f"alpha h{j}")
        job.log(f"matte {k}/{len(pids)} (piece h{j}, {cc.compute_s(h)} s)")
        SC.store("matte", matte_ids[j], sorted(Path(a.out_dir).glob(f"{STEM}_alpha_h{j}_image_*")))

    # ------------------------------------------------------------------------------------------------ the frames as designed on the page (2026-11-03): no design model, what the cards show is what renders
    from tools.app import frame_render as FR
    if FR.has_frame_design(brief.get("story")):
        job.stage("render", "the frames as designed: type, mattes, LTX metal" + (", moves on the cuts" if edit == "redesign" else ""))
        shots = brief["story"]["shots"]
        ps, objs, plan, mattes = FR.prepare(shots, segs, a.out_dir, STEM, clip, w, hits=brief["story"].get("hits"), moves=edit == "redesign")
        rep = FR.report(objs, mattes, plan)
        (w / "frame_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        FR.render_metals(objs, ps, comfy_dir / "motngrfx_cache" / "frame_metal", cc, input_dir, out_root, seed, job.log)      # beside the stage cache: the straight and the transitions version share a word's render
        final = Path(a.out_dir) / "final.mp4"
        res = FR.write_video(objs, clip, ps, mattes, plan, final, edit == "redesign", clip, log=job.log)
        job.log(f"frames as designed: {res}")
        job.state["outputs"].insert(0, dict(kind="video", file="final.mp4"))
        job.stage("done")
        job.state["done"] = True
        job.log("done")
        return 0

    # ------------------------------------------------------------------------------------------------ archive
    job.stage("archive", "shot clips + mattes -> archive folder")
    arch = Path(a.out_dir) / "archive"
    k_arch = SC.key("archive", files=[w / "seg" / f"s{k}_clip.mp4" for k in range(1, len(spans) + 1)] if SC.on else [], ids=[matte_ids[j] for j in sorted(matte_ids)],
                    texts=[(w / n).read_text(encoding="utf-8") for n in ("segments.json", "spans.json", "piece.json", "index.json")] if SC.on else [], params=dict(size=list(size)))
    if SC.restore("archive", k_arch) is None:
        job.sh([sys.executable, ROOT / "tools/app/own_video_v2_archive.py", "--work", w, "--alpha-dir", a.out_dir, "--seg", w / "seg", "--stem", STEM, "--size", f"{size[0]}x{size[1]}", "--out", arch],
               "archive")
        SC.store("archive", k_arch, [arch])

    # ------------------------------------------------------------------------------------------------ design / parse
    from tools.app import build_graphs as BG, llm_arms as LA
    base_g, _ = LA.convert(json.loads((ROOT / "app" / "graphs" / "fusion_redesign.json").read_text(encoding="utf-8")), None, "gemma31b_qat")
    base_g = BG.insert_pivot_cut(base_g)

    def seeded(g):
        g["seed"]["inputs"]["value"] = seed
        return g
    job.stage("design", "Gemma 4 31B QAT design call")
    dg = seeded(V2.graph_design(base_g, c, lyr, direction))
    k_des = SC.key("design", ids=[k_arch, wav_id], params=dict(graph=dg, code=CF()))          # fonts are not here: only what the design graph carries (a user-pinned font would be in `c` / the direction)
    if SC.restore("design", k_des) is not None:
        reply = (w / "design_reply.txt").read_text(encoding="utf-8")
    else:
        h = cc.wait(cc.submit(dg, "design"), 3600, "design")
        reply = cc.texts(h).get("out_reply", "")
        (w / "design_reply.txt").write_text(reply, encoding="utf-8")
        job.state["times"]["design_compute"] = cc.compute_s(h)
        SC.store("design", k_des, [w / "design_reply.txt"])
    job.stage("parse")
    pg = seeded(V2.graph_parse(base_g, c, lyr, direction, reply))
    k_par = SC.key("parse", ids=[k_arch, wav_id], params=dict(graph=pg, code=CF()))
    if SC.restore("parse", k_par) is not None:
        t = json.loads((w / "parse_texts.json").read_text(encoding="utf-8"))
    else:
        h = cc.wait(cc.submit(pg, "parse"), 1200, "parse")
        t = cc.texts(h)
        (w / "parse_texts.json").write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
        SC.store("parse", k_par, [w / "parse_texts.json"])
    (w / "piece_designed.json").write_text(stamp_direction_json(t["out_piece_json"], brief["direction"]), encoding="utf-8")
    (w / "parse_report.txt").write_text(t.get("out_parse_report", "") + "\n" + t.get("out_fix_report", ""), encoding="utf-8")

    # ------------------------------------------------------------------------------------------------ retime (route defaults)
    job.stage("retime", "hits on the beats, pure type, layouts, colours from the picture")
    glob_clip = w / "seg" / "global_clip.mp4"
    r_argv = retime_argv(sys.executable, ROOT, w, a.out_dir, STEM, bpm / K, seed, edit, nat, size)
    k_ret = SC.key("retime", files=[glob_clip] if SC.on else [], ids=[matte_ids[j] for j in sorted(matte_ids)],
                   texts=[(w / n).read_text(encoding="utf-8") for n in ("piece_designed.json", "words.json", "segments.json", "spans.json", "piece.json", "index.json", "concept.json") if (w / n).is_file()] if SC.on else [],
                   params=dict(argv=[str(x).replace(str(ROOT), "@ROOT@") for x in r_argv[1:]], code=CF()))
    if SC.restore("retime", k_ret) is None:
        job.sh(r_argv, "retime", env={"MOGRAPH_ROOT": str(ROOT), "CUDA_VISIBLE_DEVICES": ""})
        SC.store("retime", k_ret, [w / "piece_final.json", w / "retime_report.json"])
    rrep = json.loads((w / "retime_report.json").read_text(encoding="utf-8"))
    if edit == "overlay" and rrep.get("overlay_violations"):
        raise RuntimeError("overlay lock incomplete: " + "; ".join(rrep["overlay_violations"]))

    # ------------------------------------------------------------------------------------------------ render
    job.stage("render", "type passes, 3D cuts, compose, assemble (24 fps pipeline clock)")
    g = seeded(V2.graph_full(base_g, c, lyr, direction, (w / "piece_final.json").read_text(encoding="utf-8"), "r1"))
    g.pop("archive", None)                                         # the full ingredient archive is not needed by the app (saves ~1 min)
    g["out_final"]["inputs"]["filename_prefix"] = f"{rel}/raw"
    k_ren = SC.key("render", ids=[k_arch, wav_id, k_ret], params=dict(graph=g, code=CF()))
    got_r = SC.restore("render", k_ren)
    if got_r is not None:                                          # same piece, same pictures, same mattes, same music: the type render comes back as a new raw_*.mp4 in this job
        job.state["times"]["render_compute"] = (got_r.get("meta") or {}).get("compute_s")
        raw = next(iter(sorted(Path(a.out_dir).glob("raw_*.mp4"))), None)
    else:
        h = cc.wait(cc.submit(g, "render"), 7200, "render")
        job.state["times"]["render_compute"] = cc.compute_s(h)
        vids = (h.get("outputs") or {}).get("out_final", {}).get("images") or (h.get("outputs") or {}).get("out_final", {}).get("videos") or []
        raw = out_root / vids[0].get("subfolder", "") / vids[0]["filename"] if vids else next(iter(sorted(Path(a.out_dir).glob("raw_*.mp4"))), None)
    if raw is None or not Path(raw).is_file():
        raise RuntimeError("the render saved no video")
    if got_r is None:
        SC.store("render", k_ren, [raw], meta=dict(compute_s=job.state["times"].get("render_compute")))

    # ------------------------------------------------------------------------------------------------ finish
    job.stage("finish", "source fps + soundtrack")
    final = Path(a.out_dir) / "final.mp4"
    fps_s = ing["fps"]["fps"]
    restore_rep = None
    if edit == "overlay":                                          # output = the SOURCE frame outside the type, the composed frame inside it; 1:1 frame map, source fps tag
        pic = w / "picture.mp4"
        pic_src = clip
        if scale == "half":                                        # the source frames the output keeps are the half-scale ones: the same scale filter the shot clips were cut with (lossless, so they equal the engine's footage)
            pic_src = w / "window_half.mp4"
            job.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-an", "-vf", f"scale={nat['compose'][0]}:{nat['compose'][1]}:flags=lanczos,setsar=1", "-c:v", "libx264", "-crf", "0", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", pic_src], "half-scale source picture")
        wfr = rrep.get("word_frames") or []
        act = (max(0, min(x["start_frame"] for x in wfr) - 6), max(x["end_frame"] for x in wfr) + 6) if wfr else None          # no word and no bridge strand can be on screen outside the first start .. last end
        sup = [tuple(x["suppress_before"]) for x in wfr if x.get("suppress_before")]          # frames of a word's form-in before its own shot starts: the source frame (a word lives inside its shot)
        restore_rep = OM.restore_video(pic_src, raw, [w / "seg" / f"s{k}_clip.mp4" for k in range(1, len(spans) + 1)], spans, pic, fps_s, active_frames=act, inactive_frames=sup)
        if restore_rep.get("broken_frames"):
            job.log(f"WARNING: {len(restore_rep['broken_frames'])} frames had a type mask over {int(OM.MAX_TYPE_SHARE * 100)} % of the picture (a ground / wash, not type); {len(restore_rep['dropped_frames'])} of them show the source frame without type: {restore_rep['broken_frames'][:12]}")
        job.log(f"picture restored from the source: {restore_rep}")
        raw = pic
    vf = ["-c:v", "copy"] if (abs(K - 1.0) < 1e-6 or edit == "overlay") else ["-vf", f"setpts=N/(({fps_s})*TB)", "-r", fps_s, "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p"]
    if soundtrack == "track":
        job.sh(track_mux_cmd(raw, wav, final, vf, real_dur), "final mux (the user's track, 0.3 s fade-out)")
    else:
        if soundtrack == "video":
            aud_in, aud = clip, ["-c:a", "copy"] if (ing.get("audio") or {}).get("copy", True) else ["-c:a", "aac", "-b:a", "256k"]
        else:
            aud_in, aud = w / "music.flac", ["-c:a", "aac", "-b:a", "256k"]
        job.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-i", aud_in, "-map", "0:v:0", "-map", "1:a:0", *vf, *aud, "-t", f"{real_dur:.4f}", "-movflags", "+faststart", final], "final mux")
    pv, pa = ffprobe(final, "width", "height", "r_frame_rate", "nb_frames"), ffprobe(final, "codec_name", stream="a:0")
    job.log(f"final.mp4: {pv.get('width')}x{pv.get('height')} {pv.get('r_frame_rate')} {pv.get('nb_frames')} frames, {pv.get('duration', 0):.3f} s, audio {pa.get('codec_name')} {pa.get('duration', 0):.3f} s "
            f"(expected {nframes} frames, {real_dur:.3f} s" + (f", {nat['compose'][0]}x{nat['compose'][1]}, scale {scale}" if nat else "") + ")")
    if nat and (int(pv.get("width") or 0), int(pv.get("height") or 0)) != tuple(nat["compose"]):
        job.log(f"WARNING: final.mp4 is {pv.get('width')}x{pv.get('height')}, expected {nat['compose'][0]}x{nat['compose'][1]} (scale {scale})")
    job.state["outputs"].insert(0, dict(kind="video", file="final.mp4"))
    (Path(a.out_dir) / "overlay_report.json").write_text(json.dumps(dict(
        edit=edit, size_mode="native" if native else "tier", source=str(src), window=[start, round(start + dur, 3)], fps=fps_s, fps_float=fps, frames=nframes, size=nat["compose"] if nat else None, source_size=nat["source"] if nat else None, scale=scale, real_cuts=cuts,
        word_frames=[dict(wf, **{"fps": fps_s}) for wf in (rrep.get("word_frames") or [])], hits=[dict(text=x["text"], land_real=x["land_real"], land_frame=x.get("land_frame")) for x in words],
        soundtrack=soundtrack, restore=restore_rep, final=str(final), window_clip=str(clip)), ensure_ascii=False, indent=1), encoding="utf-8")
    job.state["info"] = dict(job.state.get("info") or {}, length=round(real_dur, 3), frames=nframes, fps=fps_s, seconds=round(real_dur, 3), size=[pv.get("width"), pv.get("height")], window=[start, round(start + dur, 3)], words=[x["text"] for x in words],
                             soundtrack=soundtrack, **({"track": track_src.name} if soundtrack == "track" else {}), tier=tier, spans=spans, bpm=bpm, key=key, edit=edit, native=native, scale=scale)
    job.stage("done")
    job.state["done"] = True
    job.log("done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True)
    ap.add_argument("--video", required=True, help="absolute path of the user clip (ComfyUI input folder)")
    ap.add_argument("--brief-file", default="")
    ap.add_argument("--length", type=float, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--format", default="9:16", help="legacy (tier route only); the source-size route takes the clip's own aspect")
    ap.add_argument("--edit", default="", choices=["", "overlay", "redesign"], help='"overlay" (the default when neither this nor brief.edit is set): the picture is untouched, only type is composited; "redesign": the old route with 3D cuts / transitions / effects')
    ap.add_argument("--size-mode", default="native", choices=["native", "tier"], help='"native" (default): the source size, fps and frame count are kept (no 9:16 crop, no tier, no fps halving); "tier": the historical ingest tiers')
    ap.add_argument("--scale", default="full", choices=["full", "half"], help='"full" (default, "Original" on the page): the output is the source size; "half": compose, type and the final are half the source size (even sides), same fps / frame count / edit / audio; the Alpha-Gen mattes are those of the full job (same cache keys)')
    ap.add_argument("--quality", default="final", choices=sorted(TIER_OF))
    ap.add_argument("--soundtrack", default="video", choices=["video", "keep", "track", "generate"], help='"keep" is the old name of "video"')
    ap.add_argument("--audio", default="", help="soundtrack=track: absolute path of the user's audio file (ComfyUI input folder)")
    ap.add_argument("--out-dir", required=True, help="<ComfyUI>/output/motngrfx_app/own/<job>")
    ap.add_argument("--comfy-dir", default=os.environ.get("COMFY_DIR", "/workspace/ComfyUI"))
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--client-id", default="")
    ap.add_argument("--stop-after", default="", choices=[""] + STAGES, help="smoke test: end cleanly after this stage")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
