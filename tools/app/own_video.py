#!/usr/bin/env python
"""Own-video route (TEST): a user-supplied clip + its audio as footage + music; our type composed over it by the REDESIGN graph.

    python tools/app/own_video.py plan  --analysis A.json --work DIR [--quality draft]     # spans, box cut script, Alpha-Gen graphs, piece.json/index.json, words graph
    python tools/app/own_video.py archive --root /workspace/ComfyUI/output/own_video/arch --matte-dir ...   # (box) assemble the archive dir

The redesign graph (nodes_fusion/redesign.py MographArchiveLoad) rebuilds the plan from request + music, so the shot spans the plan makes
(plan_rules.cut_spans over the beat grid) must equal the frames of our shot clips: `plan` asks cut_spans itself and cuts the clip to those spans.
Clip = resampled to 24 fps (pipeline FPS), scaled to the draft shot size 576x1024; audio is read from the original mp4 (source_video).
No repo file is edited; read-only use of tools/fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

FPS = 24
SIZE = (576, 1024)        # draft shot size == Alpha-Gen size (multiple of 32), 9:16
BOX_IN = "/workspace/ComfyUI/input"
SRC = "/workspace/own_video/your_clip.mp4"


def music_from(analysis: dict) -> dict:
    return dict(cut=24.0, nframes=24 * FPS, bpm=float(analysis["bpm"]), bar_len=float(analysis["bar_len"]), key=analysis.get("key", ""),
                beats=[round(b, 4) for b in analysis["beats"]], downbeats=[round(d, 4) for d in analysis["downbeats"]], chord_changes=[],
                curves={k: v for k, v in (analysis.get("curves") or {}).items()}, fps=FPS)


def spans_of(music: dict) -> list[int]:
    from tools.fusion import plan_rules as PR
    return PR.cut_spans(music["nframes"], PR.make_grid(music, music["cut"]), FPS)


def alpha_frames(n: int) -> int:
    return ((int(n) - 1 + 7) // 8) * 8 + 1


def box_script(spans: list[int]) -> str:
    """ffmpeg on the box: 24 fps, 576x1024, one mp4 per span (crf 10, yuv420p) into ComfyUI/input (own_video_sK.mp4) + first-frame PNGs."""
    L = ["set -e", f"mkdir -p /workspace/own_video/seg", f"cd /workspace/own_video"]
    start = 0
    for k, n in enumerate(spans, 1):
        vf = f"fps={FPS},scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,trim=start_frame={start}:end_frame={start + n},setpts=PTS-STARTPTS"
        L.append(f"ffmpeg -y -loglevel error -i {SRC} -an -vf \"{vf}\" -c:v libx264 -crf 10 -preset medium -pix_fmt yuv420p seg/s{k}_clip.mp4")
        na = alpha_frames(n)          # Alpha-Gen reads 8k+1 frames: feed it the real following frames too, the archive keeps the first n matte frames
        vfa = f"fps={FPS},scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,trim=start_frame={start}:end_frame={start + na},setpts=PTS-STARTPTS"
        L.append(f"ffmpeg -y -loglevel error -i {SRC} -an -vf \"{vfa}\" -c:v libx264 -crf 10 -preset medium -pix_fmt yuv420p {BOX_IN}/own_video_s{k}.mp4")
        L.append(f"ffmpeg -y -loglevel error -i seg/s{k}_clip.mp4 -frames:v 1 seg/s{k}_first.png")
        start += n
    L.append("for f in seg/s*_clip.mp4; do echo $f $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $f); done")
    return "\n".join(L) + "\n"


def alpha_graph(k: int, frames: int, seed: int = 1) -> dict:
    from tools.fusion import ltx_ic
    return ltx_ic.build("alpha", ref_video=f"own_video_s{k}.mp4", prompt="", size=SIZE, frames=frames, seed=seed, out_prefix=f"own_video/s{k}_alpha", crop="disabled", png=True)


def piece_stub(analysis: dict, music: dict, words: str, quality: str) -> tuple[dict, dict]:
    """planned piece (rules only, no dry run, no rendering) -> piece.json + index.json of the archive."""
    from tools.fusion import plan_rules as PR
    req = dict(words=words, scene="", length=24, quality=quality, format="9:16", seed=1, shots="cut", look="auto")
    piece = PR.plan_piece(PR.normalize_request(req), music, None, dry=False)
    piece = PR.finalize_piece(piece, music)
    index = dict(piece=piece.get("id"), fps=FPS, shots={sid: dict(frames=v["frames"], start=v["start"], clip_hw=[SIZE[1], SIZE[0]], matte_frames=v["frames"], matte_hw=[SIZE[1], SIZE[0]])
                                                         for sid, v in piece["layout"].items()}, still=[SIZE[1], SIZE[0]])
    return piece, index


def words_prompt(lyrics: dict, music: dict, spans: list[int]) -> str:
    segs = []
    start = 0
    for k, n in enumerate(spans, 1):
        segs.append(f"shot {k}: {start / FPS:.2f}-{(start + n) / FPS:.2f} s")
        start += n
    lw = []
    for s in lyrics["segments"]:
        lw += [f"{w['w']}@{w['s']:.2f}" for w in s["words"]]
    beats = " ".join(f"{b:.2f}" for b in music["beats"])
    return (
        "You choose the words that a motion-graphics editor will set as big type over a 24-second music-video clip (a rap verse over a sub-bass beat). "
        "The type shows ONE short lockup per shot; each lockup appears ON A BEAT while that word is sung.\n\n"
        f"LYRICS with the second each word starts (Whisper): {' '.join(lw)}\n\n"
        f"SHOTS (exactly {len(spans)}): {'; '.join(segs)}\n"
        f"BEAT TIMES in seconds (90 bpm): {beats}\n\n"
        f"Pick exactly {len(spans)} lockups, one per shot, in lyric order. Each lockup is 1 or 2 consecutive lyric words (UPPERCASE) that carry the picture or the punch of the line "
        "(concrete nouns and strong verbs; never articles, 'on', 'the', 'i'm'); each must be sung inside or within 0.5 s of its shot's time range. "
        "Give the lyric start second of its first word and the beat time (from the list) it should land on.\n"
        'Reply with JSON only: {"lines":[{"shot":1,"text":"WORD","lyric_t":0.0,"beat_t":0.0,"why":"..."}]}')


def words_graph(prompt: str, seed: int = 1) -> dict:
    return {
        "words": {"class_type": "MographGGUFVL", "inputs": dict(
            prompt=prompt, model="gemma-4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", mmproj="none", n_gpu_layers=-1, n_ctx=8192, max_tokens=1200, temperature=0.5, top_p=0.95,
            top_k=64, repeat_penalty=1.0, seed=seed, kv_cache="q8_0", flash_attn=True, mmproj_on_gpu=True, n_cpu_moe=0, handler="auto", image_max_tokens=-1,
            strip_thinking=True, keep_model_loaded=False, verbose=False, system_prompt="You are a careful typographer and film editor. You answer with exact JSON only.",
            json_mode=True), "_meta": {"title": "Gemma words"}},
        "out_reply": {"class_type": "PreviewAny", "inputs": {"source": ["words", 0]}, "_meta": {"title": "OUT words reply"}},
        "out_stats": {"class_type": "PreviewAny", "inputs": {"source": ["words", 1]}, "_meta": {"title": "OUT words stats"}},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--quality", default="draft")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    ana = json.load(open(a.analysis, encoding="utf-8"))
    music = music_from(ana)
    spans = spans_of(music)
    json.dump(dict(spans=spans, starts=[sum(spans[:i]) for i in range(len(spans))], music={k: v for k, v in music.items() if k != "curves"}), open(os.path.join(a.work, "spans.json"), "w"), indent=1)
    open(os.path.join(a.work, "box_cut.sh"), "w", newline="\n").write(box_script(spans))
    for k, n in enumerate(spans, 1):
        json.dump(alpha_graph(k, alpha_frames(n)), open(os.path.join(a.work, f"alpha_s{k}.json"), "w"), indent=1)
    lyrics = json.load(open(a.lyrics, encoding="utf-8"))
    json.dump(words_graph(words_prompt(lyrics, music, spans)), open(os.path.join(a.work, "words_graph.json"), "w"), indent=1)
    piece, index = piece_stub(ana, music, "\n".join(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"][:len(spans)]), a.quality)
    json.dump(piece, open(os.path.join(a.work, "piece.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(index, open(os.path.join(a.work, "index.json"), "w"), indent=1)
    print("spans", spans, "layout", {k: (v["start"], v["frames"]) for k, v in piece["layout"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
