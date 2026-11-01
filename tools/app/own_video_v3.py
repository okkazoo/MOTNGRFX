#!/usr/bin/env python
"""Own-video route v003 (TEST): NATIVE frame rate end to end + clean-plate contrast moments, preview size.

    python tools/app/own_video_v3.py prep   --work DIR [--src-fps 30]    # cut script, per-piece Alpha-Gen + Clean-Plate graphs, request, words (pipeline time), archive stub
    python tools/app/own_video_v3.py design --work DIR                   # phase A graph (Gemma 31B reply)                         (graphs from own_video_v2, constants patched)
    python tools/app/own_video_v3.py parse  --work DIR --reply R --label L
    python tools/app/own_video_v3.py full   --work DIR --reply PIECE_JSON --label L

NO FPS RESAMPLING (user 2026-10-30: the 30 -> 24 fps drop of v001 / v002 stuttered): every source frame is kept (the clips are cut with trim only, `-fps_mode passthrough`, never `fps=24`).
The nodes of the redesign graph have the 24 fps constant built in (plan_rules.FPS, nodes_fusion/_types.FPS; ComfyUI cannot be restarted), so the pipeline runs on a STRETCHED CLOCK: the
720 source frames are the 720 frames of a 30 s piece at the nominal 24 fps, and every time that reaches the pipeline is the real time x K (K = source fps / 24 = 1.25): the beat grid, the
reading's landing times, the lyrics, the audio (atempo 1/K, same pitch). Nothing is resampled; the last step relabels the frames: the assembled 720 frames are encoded at the SOURCE fps
(setpts x 1/K) with the ORIGINAL audio, so the output has the source's fps, frame count = 24 s x fps and the words land on the same beats as in the music reading.
The source fps is read with ffprobe on the box (--src-fps), it is a parameter, not a constant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.app import own_video as V1          # noqa: E402
from tools.app import own_video_v2 as V2       # noqa: E402

PIPE_FPS = 24                                   # the nodes' fixed clock
SIZES = {"preview": (384, 704), "draft": (576, 1024)}   # shot / type / control size per quality tier (plan_rules.sizes_for); compose 360x640 / 540x960
SIZE = SIZES["preview"]
QUALITY = "preview"
CLIP_SIZE = None        # (w, h): the engine's shot clips are written at this size (the SOURCE size, scale only; overlay_mode.native_sizes) and SIZE is then only the Alpha-Gen / matte size; None = the tier crop
GLOBAL_SIZE = None      # (w, h) of seg/global_clip.mp4 (colour / busyness / face analysis only); None = the clip size
MATTE_FPS = None        # rate the Alpha-Gen input clips are tagged with ("30", "30000/1001"): LTX reads frame_rate from the tag; None = 24 (the historical stretched-clock tag). Frames are never dropped or repeated.
FORMAT = "9:16"         # the request / concept format (9:16 | 16:9)
BOX = "/workspace/own_video"
BOX_DIR = f"{BOX}/v3"
SRC_BOX = f"{BOX}/your_clip.mp4"       # the source clip on the box (the demo route sets its own)
H_NAME = "own_video_v3_h"                           # input / output name stem of the per-piece Alpha-Gen / plate jobs (ComfyUI/input)
VERSION = "v003"
STEM = f"motion-graphic_own-video_{VERSION}"
ARCHIVE = "own_video/arch_v003"
AUDIO_X = f"{BOX_DIR}/audio_stretched.m4a"      # the song at 1/K speed: MographArchiveLoad reads the music grid / curves from it
PLATE_LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
# the 16 cuts of the source at its native fps (ffmpeg scene detection on the 30 fps stream, 25_music-read/v001/cuts.json + 12.033): first frame after each cut
CUTS_NATIVE_30 = [9, 40, 61, 121, 126, 141, 240, 301, 321, 361, 381, 480, 520, 561, 642, 662]
SCENES = {
    "man": "a test clip station platform at night under exposed pipes and fluorescent lights, a graffiti-covered wall on the right, steam drifting",
    "man walking": "a test clip platform at night with a parked train on the right, tiled wall, yellow tactile edge, fluorescent lights in the ceiling",
    "man tunnel": "a test clip platform at night with a parked train on the right, tiled wall, yellow tactile edge, fluorescent lights in the ceiling",
    "man corner": "a red-shuttered corner storefront at night, crates of fruit and hanging produce bags, green-lit shop interior",
}
PLATE_TEMPLATE = ("An empty clean plate of the exact same location, as in the source video: {scene}. No man, no people, no figures, no body parts anywhere in the frame. "
                  "No reflection or cast shadow of a person. Static photorealistic footage, natural light, high detail.")


def K_of(src_fps: float) -> float:
    return float(src_fps) / PIPE_FPS


def music_x(analysis: dict, K: float, nframes: int) -> dict:
    """the analysis on the stretched clock (every time x K, bpm / K)."""
    return dict(cut=nframes / float(PIPE_FPS), nframes=nframes, bpm=float(analysis["bpm"]) / K, bar_len=float(analysis["bar_len"]) * K, key=analysis.get("key", ""),
                beats=[round(b * K, 4) for b in analysis["beats"]], downbeats=[round(d * K, 4) for d in analysis["downbeats"]], chord_changes=[], curves={}, fps=PIPE_FPS)


def box_script(spans: list[int], pieces: list[dict], src_fps: float) -> str:
    """ffmpeg on the box: trim by NATIVE frame index, cover-fit 384x704, no fps filter, frames passed through; clips are tagged 24 fps (the IC-LoRA graphs read frames, not time)."""
    q = "-c:v libx264 -crf 10 -preset medium -pix_fmt yuv420p -fps_mode passthrough"
    fit = f"scale=-2:{SIZE[1]}:flags=lanczos,crop={SIZE[0]}:{SIZE[1]}"
    mfit = fit                                                       # the Alpha-Gen input clips
    gfit = fit                                                       # the analysis clip
    if CLIP_SIZE:                                                    # source-size route: shot clips are the source frames (scale only, no crop); the matte clips are STRETCHED to the matte size (the matte is upscaled back, never the picture)
        fit = f"scale={CLIP_SIZE[0]}:{CLIP_SIZE[1]}:flags=lanczos,setsar=1"
        mfit = f"scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,setsar=1"
        gfit = f"scale={GLOBAL_SIZE[0]}:{GLOBAL_SIZE[1]}:flags=lanczos,setsar=1" if GLOBAL_SIZE else fit
    mrate = str(MATTE_FPS) if MATTE_FPS else "24"
    src = SRC_BOX
    L = ["set -e", f"mkdir -p {BOX_DIR}/seg", f"cd {BOX}"]

    def cut(a, b, pad=0, fit_=None, rate="24"):
        tp = f",tpad=stop_mode=clone:stop={pad}" if pad > 0 else ""
        return f"trim=start_frame={a}:end_frame={b},setpts=N/({rate}*TB){tp},{fit_ or fit}"
    start = 0
    for k, n in enumerate(spans, 1):
        na = V1.alpha_frames(n)
        b = min(start + na, int(sum(spans)))
        L.append(f"ffmpeg -y -loglevel error -i {src} -an -vf \"{cut(start, b, na - (b - start))}\" {q} {BOX_DIR}/seg/s{k}_clip.mp4")
        L.append(f"ffmpeg -y -loglevel error -i {BOX_DIR}/seg/s{k}_clip.mp4 -frames:v 1 {BOX_DIR}/seg/s{k}_first.png")
        start += n
    L.append(f"ffmpeg -y -loglevel error -i {src} -an -vf \"trim=start_frame=0:end_frame={int(sum(spans))},setpts=N/(24*TB),{gfit}\" {q} {BOX_DIR}/seg/global_clip.mp4")
    for s in pieces:
        if not s["person"]:
            continue
        end = min(s["a"] + s["na"], s["seg_b"])
        pad = s["na"] - (end - s["a"])
        L.append(f"ffmpeg -y -loglevel error -i {src} -an -vf \"{cut(s['a'], end, pad, mfit, mrate)}\" {q} {V1.BOX_IN}/{H_NAME}{s['job']}.mp4")
    if abs(K_of(src_fps) - 1.0) > 1e-6:                  # a source at 24 fps needs no stretched clock: its own audio is the pipeline's
        L.append(f"ffmpeg -y -loglevel error -i {V1.SRC} -vn -af atempo={1.0 / K_of(src_fps):.6f} -c:a aac -b:a 192k {AUDIO_X}")
    L.append("for f in " + BOX_DIR + "/seg/*.mp4 " + V1.BOX_IN + "/" + H_NAME + "*.mp4; do echo $f $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $f); done")
    if abs(K_of(src_fps) - 1.0) > 1e-6:
        L.append(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {AUDIO_X}")
    return "\n".join(L) + "\n"


def alpha_graph(job: int, frames: int) -> dict:
    from tools.fusion import ltx_ic
    return ltx_ic.build("alpha", ref_video=f"{H_NAME}{job}.mp4", prompt="", size=SIZE, frames=frames, seed=1, out_prefix=f"own_video/h{job}_alpha", crop="disabled", png=True)


def plate_graph(job: int, frames: int, scene: str) -> dict:
    from tools.fusion import ltx_ic
    return ltx_ic.build("alpha", ref_video=f"{H_NAME}{job}.mp4", prompt=PLATE_TEMPLATE.format(scene=scene), size=SIZE, frames=frames, seed=1, out_prefix=f"own_video/p{job}_plate",
                        crop="disabled", png=True, lora=PLATE_LORA)


def words_x(reading: dict, K: float) -> list[dict]:
    out = []
    for w in V2.picked_words(reading):
        out.append(dict(w, land_real=w["land"], land=round(w["land"] * K, 4), hold_until=round(w["hold_until"] * K, 4)))
    return out


def concept(words) -> dict:
    c = V2.concept(words)
    c["length"] = 30
    c["quality"] = QUALITY
    c["music"] = "the clip's own rap track: a performer's verse over a slow sub-bass beat, already recorded (not generated), with sung vocals and lyrics"
    return c


def lyrics_line_x(lyrics: dict, K: float) -> str:
    lw = []
    for s in lyrics["segments"]:
        lw += [f"{w['w']}@{w['s'] * K:.2f}" for w in s["words"]]
    return "Lyrics: the performer's sung words with the second each starts (Whisper): " + " ".join(lw) + "."


def piece_stub(analysis: dict, music: dict, words: str, nframes: int) -> tuple[dict, dict]:
    from tools.fusion import plan_rules as PR
    req = dict(words=words, scene="", length=int(round(nframes / PIPE_FPS)), quality=QUALITY, format=FORMAT, seed=1, shots="cut", look="auto")
    piece = PR.finalize_piece(PR.plan_piece(PR.normalize_request(req), music, None, dry=False), music)
    index = dict(piece=piece.get("id"), fps=PIPE_FPS, shots={sid: dict(frames=v["frames"], start=v["start"], clip_hw=[(CLIP_SIZE or SIZE)[1], (CLIP_SIZE or SIZE)[0]], matte_frames=v["frames"], matte_hw=[SIZE[1], SIZE[0]])
                                                              for sid, v in piece["layout"].items()}, still=[SIZE[1], SIZE[0]])
    return piece, index


def patch_v2() -> None:
    """own_video_v2's graph builders read module constants: point them at v003."""
    V2.ARCHIVE, V2.SRC, V2.VERSION, V2.STEM = ARCHIVE, AUDIO_X, VERSION, STEM


DIRECTION4 = (
    "DESIGN DIRECTION (from the user, follow it). The clip is the user's own music video: a performer in a test clip station and at a red-shuttered corner, cut with inserts "
    "(train, graffiti, street, a gold chain close-up, a crowd). The type must keep its EFFECTS visible and MOVE from word to word over the same clip, not sit flat. "
    "(1) Build the words from LINES: use ONE line module for the whole piece (circuit, lanes or pipework), with the line io styles `lines` (and `blobs` / `particles`) on every word, "
    "never `flicker` or `wipe`, so the strands of a word form and release visibly. (2) BETWEEN every two consecutive words the type itself travels: the strands of word A leave from its exit "
    "side, cross the frame and form word B at its entry side, one designed word-to-word transition per gap, at least 0.5 s long and clearly visible, from circuit_board, braid, fan, ripple, "
    "spiral_around, twist (no cut between words); the neighbouring gaps differ. (3) Every word lands CENTRED at its hit, on a beat, and holds perfectly still (no yaw, no drift) for at least "
    "two beats before its exit; a long word may be stacked on two lines. The hit times below are fixed, write them as `hit`. Footage cuts are hard; at most one designed 3D cut. "
    "Fixed hits (seconds): {hits}.")


def cmd_prep(a) -> int:
    os.makedirs(a.work, exist_ok=True)
    K = K_of(a.src_fps)
    ana = json.load(open(a.analysis, encoding="utf-8"))
    nframes = int(a.frames)
    music = music_x(ana, K, nframes)
    from tools.fusion import plan_rules as PR
    spans = PR.cut_spans(nframes, PR.make_grid(music, music["cut"]), PIPE_FPS)
    segs = V2.segments(CUTS_NATIVE_30 if round(a.src_fps) == 30 else json.load(open(a.cuts, encoding="utf-8")), nframes, spans)
    reading = json.load(open(a.reading, encoding="utf-8"))
    words = words_x(reading, K)
    V2._write(os.path.join(a.work, "spans.json"), dict(spans=spans, starts=[sum(spans[:i]) for i in range(len(spans))], src_fps=a.src_fps, K=K))
    V2._write(os.path.join(a.work, "segments.json"), segs)
    if not a.no_graphs:
        V2._write(os.path.join(a.work, "box_cut_v3.sh"), box_script(spans, segs, a.src_fps), js=False)
    for s in segs:
        if s["person"] and not a.no_graphs:
            V2._write(os.path.join(a.work, f"alpha_h{s['job']}.json"), alpha_graph(s["job"], s["na"]))
            if a.plates == "all":          # RULE (user 2026-10-30): clean plates ONLY where a contrast section lands (`plates` command, after `plan`); "all" is for tests only (13 plates = ~21 min)
                scene = SCENES.get(s["label"], SCENES["man"])
                V2._write(os.path.join(a.work, f"plate_p{s['job']}.json"), plate_graph(s["job"], s["na"], scene))
    V2._write(os.path.join(a.work, "concept.json"), concept(words))
    V2._write(os.path.join(a.work, "words.json"), words)
    lyrics = json.load(open(a.lyrics, encoding="utf-8"))
    V2._write(os.path.join(a.work, "lyrics_line.txt"), lyrics_line_x(lyrics, K), js=False)
    V2._write(os.path.join(a.work, "direction.txt"),
              (DIRECTION4.format(hits=", ".join(f"{w['text']} {w['land']:.2f}" for w in words)) if a.direction == "v4" else V2.direction_text(words)), js=False)
    piece, index = piece_stub(ana, music, "\n".join(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"][:len(spans)]), nframes)
    V2._write(os.path.join(a.work, "piece.json"), piece)
    V2._write(os.path.join(a.work, "index.json"), index)
    print("src fps", a.src_fps, "K", K, "frames", nframes, "spans", spans, "bpm (pipeline clock)", round(music["bpm"], 2))
    for s in segs:
        print(f"  seg {s['i']:2d} shot s{s['shot']} frames {s['a']:3d}-{s['b']:3d} ({s['n']:3d}) {s['label']:18s} " + (f"alpha h{s['job']} ({s['na']} frames)" if s["person"] else "empty"))
    return 0


def person_runs(segs: list[dict]) -> list[tuple[int, int]]:
    """contiguous frame runs where a matte AND a clean plate exist (person pieces, adjacent pieces of the same real shot joined)."""
    runs: list[list[int]] = []
    for s in segs:
        if not s["person"]:
            continue
        if runs and runs[-1][1] == s["a"] and (JOIN_REAL_CUTS or s["i"] == last_i[0]):
            runs[-1][1] = s["b"]
        else:
            runs.append([s["a"], s["b"]])
        last_i[0] = s["i"]
    return [tuple(r) for r in runs]


last_i = [None]
JOIN_REAL_CUTS = False        # True when every piece has a subject (the own-footage demo): a moment may run across a real cut
IRIS = 8                  # frames of the iris that opens before / closes after a moment (the visible transition)
MIN_MOMENT = 16           # shortest fully-contrast stretch (frames)


SECTION_MIN, SECTION_MAX = 48, 96     # window_subject_front is a SECTION: 2-4 s at 24 frames/s of the pipeline clock (= source frames), on one word hold


def plan_moments(segs: list[dict], hits: list[dict], seed: int, must_section: bool = False) -> list[dict]:
    """DRAWN contrast plan (code, no hand picking): every word whose hold [hit, release) has a stretch of >= MIN_MOMENT frames inside ONE run of person pieces, with IRIS frames of room on
    both sides, is a candidate (the moment sits on the word's own hit and release, shrunk to the run). seed -> 1..3 candidates, a mode for each from plate_in_subject | plate_in_type |
    window_subject_front (the section mode needs >= SECTION_MIN frames and is cut to SECTION_MAX); must_section: the first drawn candidate that can carry a section gets window_subject_front.
    hits: [{text, hit_f, rel_f}] in source frames (pipeline clock frame = source frame)."""
    import random
    last_i[0] = None
    runs = person_runs(segs)
    cands = []
    for k, h in enumerate(hits):
        for a, b in runs:
            A, B = max(h["hit_f"], a + IRIS), min(h["rel_f"], b - IRIS)
            if B - A >= MIN_MOMENT:
                cands.append(dict(word=h["text"], k=k, a=int(A), b=int(B)))
                break
    rng = random.Random(int(seed))
    n = min(len(cands), rng.choice([1, 2, 2, 3]))
    pick = sorted(rng.sample(range(len(cands)), n)) if cands else []
    sec_ok = [i for i in pick if cands[i]["b"] - cands[i]["a"] >= SECTION_MIN]
    sec_first = rng.choice(sec_ok) if (must_section and sec_ok) else None
    out = []
    for i in pick:
        d = dict(cands[i])
        long_enough = d["b"] - d["a"] >= SECTION_MIN
        modes = ["plate_in_subject", "plate_in_type"] + (["window_subject_front"] if long_enough else [])
        d["mode"] = "window_subject_front" if i == sec_first else rng.choice(modes)
        if d["mode"] == "window_subject_front":
            d["b"] = min(d["b"], d["a"] + SECTION_MAX)
        out.append(d)
    return out


def plan_sections(segs: list[dict], report: dict, seed: int, text_mask: int = 0) -> list[dict]:
    """v006: the masking SECTIONS come from the layout rule, not from a free draw. report = retime_report.json (own_video_v2_retime_box --layouts facefix): `layouts.flat_sections` = the
    words whose grid would sit over the subject's face; each becomes a `window_subject_front` section (flat ground, letters = windows, the subject IN FRONT of the letters, the occlusion
    guard waived: the grid is already the bigger layout) on the frames of its hold that lie inside one run of person pieces (IRIS frames of room each side). `text_mask` of them (seeded) are
    drawn as `text_as_mask` instead (the letters' interiors are the mask, the subject is NOT drawn on top). Every section carries the animated edge line (matte contour / letter contour)."""
    import random
    last_i[0] = None
    runs = person_runs(segs)
    words = report["words"]
    out = []
    for fs in (report.get("layouts") or {}).get("flat_sections") or []:
        k = int(fs["k"])
        h0, h1 = int(words[k]["dry_hit"] * PIPE_FPS), int(words[k]["dry_rel"] * PIPE_FPS)
        for a_, b_ in runs:
            A, B = max(h0, a_ + IRIS, int(fs["a"])), min(h1, b_ - IRIS, int(fs["b"]))
            if B - A >= MIN_MOMENT:
                out.append(dict(word=words[k]["text"], k=k, a=int(A), b=int(B), mode="window_subject_front", guard=False, edge_line=True, fill="footage", why="grid over the face -> flat-ground section"))
                break
    rng = random.Random(int(seed))
    for i in sorted(rng.sample(range(len(out)), min(int(text_mask), len(out)))):
        out[i] = dict(out[i], mode="text_as_mask", why="text-as-mask section: the letters' interiors are the mask")
    return out


def cmd_plan6(a) -> int:
    segs = json.load(open(os.path.join(a.work, "segments.json"), encoding="utf-8"))
    rep_ = json.load(open(a.report_json, encoding="utf-8"))
    plan = plan_sections(segs, rep_, a.seed, a.text_mask)
    V2._write(os.path.join(a.work, f"moments_{a.label}.json"), plan)
    for m in plan:
        print(f"{a.label}: {m['mode']:20s} on {m['word']:15s} frames {m['a']}-{m['b']} = {m['a'] / a.src_fps:.2f}-{m['b'] / a.src_fps:.2f} s, edge line {m['edge_line']}, guard {'on' if m['guard'] else 'waived'}")
    return 0


def cmd_plan(a) -> int:
    segs = json.load(open(os.path.join(a.work, "segments.json"), encoding="utf-8"))
    rep = json.load(open(a.report_json, encoding="utf-8"))
    hits = [dict(text=w["text"], hit_f=int(round(w["dry_hit"] * PIPE_FPS)), rel_f=int(round(w["dry_rel"] * PIPE_FPS))) for w in rep["words"]]
    plan = plan_moments(segs, hits, a.seed, must_section=a.must_section)
    V2._write(os.path.join(a.work, f"moments_seed{a.seed}.json"), plan)
    K = K_of(a.src_fps)
    for m in plan:
        print(f"seed {a.seed}: {m['mode']:16s} on {m['word']:16s} frames {m['a']}-{m['b']} = {m['a'] / a.src_fps:.2f}-{m['b'] / a.src_fps:.2f} s (iris {IRIS} f each side)")
    return 0


CUT_PLATE_PRE_S, CUT_PLATE_POST_S = 0.75, 1.25      # 3D cut window around the boundary (s): the move into the cut + the move out and the settle (provisional, from the pivot / orbit move lengths)


def cut_plate_windows(piece: dict, spans: list[int], fps: float = PIPE_FPS) -> list[dict]:
    """(2026-10-30) CLEAN PLATES FOR 3D CUTS: every compose cut that SPLITS the subject (`cut_kinds.SHOT_3D`: orbit push / pullback / fly, depth_pivot, depth_dolly) moves the camera through the
    shot's depth cards; with the clean plate as the background card and the matted subject glitching in after the settle there is no blurred-silhouette hole. So the shots on BOTH sides of such a
    cut get a clean plate (not only the pieces under the contrast moments), over the cut window = [boundary - CUT_PLATE_PRE_S, boundary + CUT_PLATE_POST_S] clipped to the two shots.
    -> [{mode: "cut_plate", cut, after, a, b, shots: [sid_before, sid_after], why}] in pipeline frames (= source frames); `pieces_for_moments` turns them into pieces (a piece under the window needs the plate)."""
    from tools.fusion import cut_kinds as CK
    starts = [sum(spans[:i]) for i in range(len(spans))]
    ends = [sum(spans[:i + 1]) for i in range(len(spans))]
    ids = [f"s{i + 1}" for i in range(len(spans))]
    out = []
    for c in (piece.get("compose") or {}).get("cuts") or []:
        kind = CK.normalize(c.get("cut"))
        if kind not in CK.SHOT_3D or c.get("after") not in ids:
            continue
        i = ids.index(c["after"])
        if i + 1 >= len(spans):
            continue
        bnd = ends[i]
        a = max(starts[i], int(round(bnd - CUT_PLATE_PRE_S * fps)))
        b = min(ends[i + 1], int(round(bnd + CUT_PLATE_POST_S * fps)))
        out.append(dict(mode="cut_plate", cut=kind, after=c["after"], a=int(a), b=int(b), shots=[ids[i], ids[i + 1]], boundary=int(bnd),
                        why=f"{kind} splits the subject: the clean plate is the background of the move, the matted subject glitches in after the settle"))
    return out


def pieces_for_moments(segs: list[dict], moments: list[dict]) -> list[dict]:
    """the person pieces any frame of [a - IRIS, b + IRIS) of a moment lies in: the ONLY pieces that need a clean plate."""
    out = {}
    for m in moments:
        for s in segs:
            if s["person"] and s["a"] < m["b"] + IRIS and s["b"] > m["a"] - IRIS:
                out[s["job"]] = s
    return [out[k] for k in sorted(out)]


def cmd_plates(a) -> int:
    """clean-plate graphs (ONLY for the pieces under the drawn moments of --moments, a moments_seedN.json) into work/."""
    segs = json.load(open(os.path.join(a.work, "segments.json"), encoding="utf-8"))
    moments = json.load(open(a.moments, encoding="utf-8"))
    if getattr(a, "cut_plates", None):                                  # + the shots either side of every 3D cut that splits the subject
        pf = json.load(open(a.cut_plates, encoding="utf-8"))
        spans = json.load(open(os.path.join(a.work, "spans.json"), encoding="utf-8"))["spans"]
        cpw = cut_plate_windows(pf, spans)
        V2._write(os.path.join(a.work, "cut_plate_windows.json"), cpw)
        for w in cpw:
            print(f"cut plate: {w['cut']} after {w['after']} frames {w['a']}-{w['b']} shots {w['shots']}")
        moments = list(moments) + cpw
    ps = pieces_for_moments(segs, moments)
    for s in ps:
        V2._write(os.path.join(a.work, f"plate_p{s['job']}.json"), plate_graph(s["job"], s["na"], SCENES.get(s["label"], SCENES["man"])))
    print("clean plates needed for pieces", [s["job"] for s in ps], "of", sum(1 for s in segs if s["person"]), "person pieces")
    return 0


def cmd_snapshot(a) -> int:
    """freeze everything a variant needs for the one-command upgrade into work/<variant>/: the EXACT designed piece json that the graph ran, the model reply, the drawn moment plan, the
    spans / segments / words / concept / direction, the seeds and the ingest decisions."""
    import shutil
    d = os.path.join(a.work, a.label)
    os.makedirs(d, exist_ok=True)
    for f in ("spans.json", "segments.json", "words.json", "concept.json", "direction.txt", "lyrics_line.txt", "piece.json", "index.json"):
        shutil.copy(os.path.join(a.work, f), os.path.join(d, f))
    shutil.copy(a.reply, os.path.join(d, "piece_final.json"))
    if a.report_json:
        shutil.copy(a.report_json, os.path.join(d, "retime_report.json"))
    mom = os.path.join(a.work, f"moments_seed{a.seed}.json")
    if os.path.isfile(mom):
        shutil.copy(mom, os.path.join(d, "moments.json"))
    for f in ("out_out_reply.txt",):
        if os.path.isfile(os.path.join(a.work, f)):
            shutil.copy(os.path.join(a.work, f), os.path.join(d, "model_reply.txt"))
    V2._write(os.path.join(d, "variant.json"), dict(variant=a.label, moment_seed=a.seed, quality=QUALITY, size=list(SIZE), src_fps=a.src_fps, K=K_of(a.src_fps), frames=a.frames,
        pipeline_fps=PIPE_FPS, design_seed=7, concept_seed=7, alpha_seed=1, plate_seed=1,
        ingest=dict(source="your_clip.mp4", fit=f"scale=-2:{SIZE[1]} lanczos + centred crop {SIZE[0]}x{SIZE[1]}; compose cover-fit x0.9375 + 10 px crop top/bottom (preview) ", no_fps_resample=True),
        upgrade=f"python tools/app/own_video_v3.py upgrade --work {a.work} --label {a.label} --quality draft"))
    print("snapshot ->", d)
    return 0


def cmd_upgrade(a) -> int:
    """ONE COMMAND to the 540x960 version of a preview variant. It reuses the variant's EXACT piece json (only quality / sizes change: no new Gemma call, same seeds, same spans, same
    contrast-moment plan, same word timings) and writes, into work/<label>_<quality>/, the cut script, the per-piece Alpha-Gen + Clean-Plate graphs at the bigger size over the same spans and the
    graph with the piece literal. It does NOT run anything (run order printed). Matte / plate / clip are regenerated at the higher size; the moments script is the same."""
    global SIZE, QUALITY
    QUALITY, SIZE = a.quality, SIZES[a.quality]
    src = os.path.join(a.work, a.label)
    out = os.path.join(a.work, f"{a.label}_{a.quality}")
    os.makedirs(out, exist_ok=True)
    from tools.fusion import plan_rules as PR
    var = json.load(open(os.path.join(src, "variant.json"), encoding="utf-8"))
    spans = json.load(open(os.path.join(src, "spans.json"), encoding="utf-8"))
    segs = json.load(open(os.path.join(src, "segments.json"), encoding="utf-8"))
    piece = json.load(open(os.path.join(src, "piece_final.json"), encoding="utf-8"))
    piece["quality"] = a.quality
    piece["sizes"] = PR.sizes_for(a.quality, "9:16")
    piece.setdefault("compose", {})["size"] = piece["sizes"]["compose"]
    V2._write(os.path.join(out, "piece_final.json"), piece)
    for f in ("spans.json", "segments.json", "words.json", "direction.txt", "lyrics_line.txt", "moments.json", "variant.json"):
        if os.path.isfile(os.path.join(src, f)):
            import shutil
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    c = json.load(open(os.path.join(src, "concept.json"), encoding="utf-8"))
    c["quality"] = a.quality
    V2._write(os.path.join(out, "concept.json"), c)
    V2._write(os.path.join(out, "box_cut_v3.sh"), box_script(spans["spans"], segs, var["src_fps"]), js=False)
    moments = json.load(open(os.path.join(src, "moments.json"), encoding="utf-8")) if os.path.isfile(os.path.join(src, "moments.json")) else []
    need = {s["job"] for s in pieces_for_moments(segs, moments)}
    for s in segs:
        if s["person"]:
            V2._write(os.path.join(out, f"alpha_h{s['job']}.json"), alpha_graph(s["job"], s["na"]))
            if s["job"] in need:                    # clean plates ONLY under the moments
                V2._write(os.path.join(out, f"plate_p{s['job']}.json"), plate_graph(s["job"], s["na"], SCENES.get(s["label"], SCENES["man"])))
    patch_v2()
    base = V2._load_base()
    lyr = open(os.path.join(src, "lyrics_line.txt"), encoding="utf-8").read()
    d = open(os.path.join(src, "direction.txt"), encoding="utf-8").read()
    V2._write(os.path.join(out, f"graph_full_{a.label}_{a.quality}.json"), V2.graph_full(base, c, lyr, d, json.dumps(piece, ensure_ascii=False), f"{a.label}_{a.quality}"))
    print(f"upgrade of {a.label} to {a.quality} {SIZE[0]}x{SIZE[1]} written to {out}; run: box_cut_v3.sh -> alpha_h*.json -> plate_p*.json (progress_standin submit each) -> own_video_v2_archive.py "
          f"--size {SIZE[0]}x{SIZE[1]} -> graph_full_{a.label}_{a.quality}.json -> own_video_v3_moments.py with moments.json (frames are source frames: unchanged)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "design", "parse", "full", "plan", "plan6", "plates", "snapshot", "upgrade"])
    ap.add_argument("--analysis", default=os.path.join(ROOT, "25_music-read", "v001", "work", "your_clip_analysis.json"))
    ap.add_argument("--reading", default=os.path.join(ROOT, "25_music-read", "v001", "reading.json"))
    ap.add_argument("--lyrics", default=os.path.join(ROOT, "25_music-read", "v001", "lyrics_24s.json"))
    ap.add_argument("--cuts")
    ap.add_argument("--src-fps", type=float, default=30.0, help="ffprobe r_frame_rate of the source")
    ap.add_argument("--frames", type=int, default=720, help="frames of the output window (24 s x source fps)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--reply")
    ap.add_argument("--label", default="r1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--must-section", action="store_true", help="plan: the first drawn candidate able to carry a 2-4 s section gets window_subject_front")
    ap.add_argument("--quality", default="preview", choices=["preview", "draft"])
    ap.add_argument("--report-json")
    ap.add_argument("--moments")
    ap.add_argument("--cut-plates", default="", help="plates: piece_final.json whose 3D cuts (orbit push / pullback / fly, pivot, dolly) also get clean plates on both sides (cut_plate_windows)")
    ap.add_argument("--text-mask", type=int, default=0, help="plan6: how many of the sections are drawn as text_as_mask")
    ap.add_argument("--version", default="v003")
    ap.add_argument("--keep-music-line", action="store_true")
    ap.add_argument("--join-cuts", action="store_true", help="plan: a moment may run across real cuts (every piece has the subject)")
    ap.add_argument("--audio-src", default="", help="box path of the graph's source_video (default: the stretched audio of the test clip clip)")
    ap.add_argument("--archive", default="own_video/arch_v003", help="archive folder under the box output dir (v004 reuses the v003 clips + mattes: identical spans, size and fps)")
    ap.add_argument("--direction", default="v3", choices=["v3", "v4"])
    ap.add_argument("--no-graphs", action="store_true", help="prep: skip the cut script and the Alpha-Gen / Clean-Plate graphs (a version that reuses an earlier version's ingredients)")
    ap.add_argument("--plates", default="none", choices=["none", "all"])
    a = ap.parse_args()
    global SIZE, QUALITY, VERSION, STEM, ARCHIVE
    QUALITY, SIZE = a.quality, SIZES[a.quality]
    VERSION, STEM, ARCHIVE = a.version, f"motion-graphic_own-video_{a.version}", a.archive
    global AUDIO_X
    if a.audio_src:
        AUDIO_X = a.audio_src                    # the graph's source_video (its audio track is the song): the ingested clip of an own-footage demo
    patch_v2()
    global JOIN_REAL_CUTS
    JOIN_REAL_CUTS = bool(a.join_cuts)
    if a.keep_music_line:
        V2.CHARACTER_FIND = "NO-MATCH-KEEP"       # our own generated ACE track: the prompt's "generated instrumental music from the prompt" line is TRUE, keep it
    if a.cmd == "plan6":
        return cmd_plan6(a)
    if a.cmd == "plates":
        return cmd_plates(a)
    if a.cmd == "snapshot":
        return cmd_snapshot(a)
    if a.cmd == "upgrade":
        return cmd_upgrade(a)
    if a.cmd == "prep":
        return cmd_prep(a)
    if a.cmd == "plan":
        return cmd_plan(a)
    return dict(design=V2.cmd_design, parse=V2.cmd_parse, full=V2.cmd_full)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
