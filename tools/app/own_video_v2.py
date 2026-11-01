#!/usr/bin/env python
"""Own-video route v002 (TEST): cut-aware mattes, words timed to the music reading, behind-the-subject words, a design-direction block.

    python tools/app/own_video_v2.py prep   --cuts24 cuts24.json --work DIR     # segments, box cut script, one Alpha-Gen graph per PERSON segment, request, direction
    python tools/app/own_video_v2.py design --work DIR                          # phase A graph: archive load -> plan -> design package -> (direction injected) -> Gemma 31B reply
    python tools/app/own_video_v2.py retime --work DIR --reply reply.txt        # reply with the reading's landing times + the behind-the-man op chain -> reply_v2.txt
    python tools/app/own_video_v2.py full   --work DIR --reply reply_v2.txt     # phase B graph: the same redesign graph, the (retimed) reply as a literal, compose + assemble

What v001 could not do and v002 does (16_app/OWN_VIDEO.md):
  * one matte per REAL cut of the clip (not per planned shot): `segments` splits the 24 fps timeline at the video's own cuts, the Alpha-Gen job runs per segment that shows the
    man, inserts (train / graffiti / street / necklace close-up / crowd) get an empty matte, and the archive step (own_video_v2_archive.py, on the box) lays the segment mattes back
    onto the planned shots frame for frame, so a matte always sits on its own picture.
  * words = the music reading's prominent words, landing at their beat-snapped `land`: the model's reply fields `hit / hold_s / form_s / exit_s` are the planner's timing input
    (qwen_piece.compile_timing solves t0 / t1 with spec.plan_timing); `retime` writes the reading's times into them.
  * a design-direction block and a truthful music / lyrics line go into the design prompt with core StringReplace nodes in the graph (MographArchiveLoad.request_from has no field for them).
No repo file is edited; read-only use of tools/fusion and tools/app/own_video.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.app import own_video as V1          # noqa: E402

FPS = V1.FPS
SIZE = V1.SIZE
BOX_IN = V1.BOX_IN
SRC = V1.SRC
BOX_DIR = "/workspace/own_video/v2"
VERSION = "v002"
ARCHIVE = "own_video/arch_v002"
STEM = f"motion-graphic_own-video_{VERSION}"

# What each segment between two real cuts shows (read from the source sheet 26_own-video/v002/work/cuts_check.png, 2026-10-30). person=True: the man (Alpha-Gen runs);
# False: an insert with nobody to cut out (empty matte). 16 cuts -> 17 segments.
SEG_LABELS = [
    ("man", True), ("man", True), ("train", False), ("man", True), ("graffiti", False), ("graffiti", False), ("man", True), ("man walking", True),
    ("street", False), ("man walking", True), ("necklace close-up", False), ("man tunnel", True), ("man corner", True), ("crowd street", False),
    ("man corner", True), ("graffiti", False), ("man corner", True)]

# the reading's prominent words (25_music-read/v001/reading.json) that become the six lockups: one per planned shot; two words were left out because each sat
# 1.33 s before the next word, which leaves no room for form + hold + exit + gap (hold >= 2 beats needs >= ~2.7 s between landings).
PICK = ["FIRST WORD", "SECOND WORD", "THIRD WORD", "FOURTH WORD", "FIFTH WORD", "SIXTH WORD"]
TEXT = {"FIFTH WORD": "FIFTH WORD"}      # the article is dropped (v001 rule: no articles in the type)

DIRECTION = (
    "DESIGN DIRECTION (from the user, follow it). The clip is the user's own music video: a performer in a test clip station and at a red-shuttered corner, cut with inserts "
    "(train, graffiti, street, a gold chain close-up, a crowd). Every word forms ON a beat and lands CENTRED on the frame at its hit, then holds perfectly still (no yaw, no drift, no "
    "wobble, no zoom) for at least two beats and until shortly before the next word, then exits; the hit times below are fixed, write them as `hit`. A long word may be stacked on two "
    "lines. Footage cuts are hard; at most one designed 3D cut; transitions clearly visible. At least ONE word sits BEHIND the man: the type is "
    "drawn between the footage and the man's matte with the man in front of it (the code writes that op chain itself, you only choose which words and keep them big and wide so part of "
    "every letter run stays visible). If the cuts menu offers a text-as-window cut, use it once. Fixed hits (seconds): {hits}.")

LYRICS_FIND = "Lyrics: none (instrumental)."
CHARACTER_FIND = "generated instrumental music from the prompt: "


# ----------------------------------------------------------------------------------------------------------------------------------- segments
def segments(cuts24: list[int], nframes: int, spans: list[int]) -> list[dict]:
    """the clip's own cuts (24 fps frame of the first picture after each cut) x the planned shot spans -> pieces [{i, shot, a, b, n, seg_b, label, person, job, na}].
    RULE (user, 2026-10-30): mattes / control nets are made only over the output window (0..nframes) and PER SHOT: every real-cut segment is cut at the shot boundaries first, each
    piece is trimmed to its own span and Alpha-Gen runs on that trimmed clip. The 8k+1 pad (`na`) takes real frames only while the same real shot continues (never past the window end);
    the rest is the last frame repeated."""
    b = [0] + [int(c) for c in cuts24] + [int(nframes)]
    if len(b) - 1 != len(SEG_LABELS):
        raise SystemExit(f"{len(b) - 1} segments but {len(SEG_LABELS)} labels: re-read the source sheet")
    if sum(spans) != nframes:
        raise SystemExit(f"shot spans sum to {sum(spans)}, the window is {nframes}")
    sb = [0]
    for n_ in spans:
        sb.append(sb[-1] + n_)
    out, job = [], 0
    for i in range(len(b) - 1):
        lab, person = SEG_LABELS[i]
        cuts = sorted({b[i], b[i + 1]} | {x for x in sb if b[i] < x < b[i + 1]})
        for a_, e_ in zip(cuts[:-1], cuts[1:]):
            shot = max(k for k, x in enumerate(sb[:-1]) if x <= a_) + 1
            d = dict(i=i, shot=shot, a=a_, b=e_, n=e_ - a_, seg_b=b[i + 1], label=lab, person=person, job=None, na=None)
            if person:
                job += 1
                d["job"], d["na"] = job, V1.alpha_frames(max(e_ - a_, 9))
            out.append(d)
    return out


def box_script(spans: list[int], segs: list[dict]) -> str:
    """ffmpeg on the box, the v001 filter chain (fps 24 -> 576x1024 -> trim) for the shot clips AND for every person segment, so a matte frame and a clip frame are the same picture."""
    q = "-c:v libx264 -crf 10 -preset medium -pix_fmt yuv420p"
    vf0 = f"fps={FPS},scale={SIZE[0]}:{SIZE[1]}:flags=lanczos"
    L = ["set -e", f"mkdir -p {BOX_DIR}/seg", "cd /workspace/own_video"]
    start = 0
    for k, n in enumerate(spans, 1):
        na = V1.alpha_frames(n)
        L.append(f"ffmpeg -y -loglevel error -i {SRC} -an -vf \"{vf0},trim=start_frame={start}:end_frame={start + na},setpts=PTS-STARTPTS\" {q} {BOX_DIR}/seg/s{k}_clip.mp4")
        L.append(f"ffmpeg -y -loglevel error -i {BOX_DIR}/seg/s{k}_clip.mp4 -frames:v 1 {BOX_DIR}/seg/s{k}_first.png")
        start += n
    for s in segs:
        if not s["person"]:
            continue
        end = min(s["a"] + s["na"], s["seg_b"])          # real frames only while the same real shot goes on (never past the window: the last segment ends at it)
        pad = s["na"] - (end - s["a"])
        tp = f",tpad=stop_mode=clone:stop={pad}" if pad > 0 else ""
        L.append(f"ffmpeg -y -loglevel error -i {SRC} -an -vf \"{vf0},trim=start_frame={s['a']}:end_frame={end},setpts=PTS-STARTPTS{tp}\" {q} {BOX_IN}/own_video_v2_h{s['job']}.mp4")
    L.append("for f in " + BOX_DIR + "/seg/s*_clip.mp4 " + BOX_IN + "/own_video_v2_h*.mp4; do echo $f $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $f); done")
    return "\n".join(L) + "\n"


def alpha_graph_seg(job: int, frames: int, seed: int = 1) -> dict:
    from tools.fusion import ltx_ic
    return ltx_ic.build("alpha", ref_video=f"own_video_v2_h{job}.mp4", prompt="", size=SIZE, frames=frames, seed=seed, out_prefix=f"own_video/h{job}_alpha", crop="disabled", png=True)


# ----------------------------------------------------------------------------------------------------------------------------------- words
def picked_words(reading: dict) -> list[dict]:
    pw = {w["text"]: w for w in reading["prominent_words"]}
    out = []
    for t in PICK:
        w = pw[t]
        out.append(dict(text=TEXT.get(t, t.upper()), land=float(w["land"]), hold_until=float(w["hold_until"]), sung_at=float(w["sung_at"]), beat=int(w["land_beat_index"])))
    return out


def concept(words: list[dict]) -> dict:
    return {"words": "\n".join(w["text"] for w in words),
            "scene": "A performer speaks to camera in a station, then outside a shop at night.",
            "music": "the clip's own rap track: a performer's verse over a slow 90 bpm sub-bass beat, already recorded (not generated), with sung vocals and lyrics",
            "length": 24, "look": "auto", "seed": 7, "quality": "draft", "format": "9:16", "shots": ""}


def lyrics_line(lyrics: dict) -> str:
    lw = []
    for s in lyrics["segments"]:
        lw += [f"{w['w']}@{w['s']:.2f}" for w in s["words"]]
    return "Lyrics: the performer's sung words with the second each starts (Whisper): " + " ".join(lw) + "."


def direction_text(words: list[dict]) -> str:
    return DIRECTION.format(hits=", ".join(f"{w['text']} {w['land']:.2f}" for w in words))


# ----------------------------------------------------------------------------------------------------------------------------------- graphs
def _load_base() -> dict:
    return json.load(open(os.path.join(ROOT, "26_own-video", "v001", "work", "redesign_gemma31b_pivot.json"), encoding="utf-8"))


def _inject_prompt_edits(g: dict, lyrics: str, direction: str) -> str:
    """StringReplace nodes between the design package and the model: the music character line and the lyrics line made truthful, the direction block before the output schema.
    -> id of the last node (its output 0 is the prompt for the model)."""
    steps = [("p_char", CHARACTER_FIND, ""), ("p_lyr", LYRICS_FIND, lyrics), ("p_dir", "=== OUTPUT SCHEMA ===", direction + "\n\n=== OUTPUT SCHEMA ===")]
    prev = ["design_pkg", 0]
    for nid, find, rep in steps:
        g[nid] = {"class_type": "StringReplace", "inputs": {"string": prev, "find": find, "replace": rep}, "_meta": {"title": f"prompt edit {nid}"}}
        prev = [nid, 0]
    return "p_dir"


def graph_common(g: dict, concept_json: dict, lyrics: str, direction: str) -> dict:
    g = copy.deepcopy(g)
    g["archive_in"]["inputs"]["value"] = ARCHIVE
    g["source_in"]["inputs"]["value"] = SRC          # the user's own mp4: its audio track is the song
    g["concept_in"]["inputs"]["value"] = json.dumps(concept_json, ensure_ascii=False)
    g["seed"]["inputs"]["value"] = 7
    last = _inject_prompt_edits(g, lyrics, direction)
    g["design_qwen"]["inputs"]["prompt"] = [last, 0]
    return g


def graph_design(base: dict, concept_json: dict, lyrics: str, direction: str) -> dict:
    """phase A: everything up to the model's reply (the parse, type, compose and assemble nodes are left out)."""
    g = graph_common(base, concept_json, lyrics, direction)
    keep = {"archive_in", "source_in", "concept_in", "seed", "load", "plan", "flow", "design_pkg", "p_char", "p_lyr", "p_dir", "design_qwen"}
    g = {k: v for k, v in g.items() if k in keep}
    g["out_reply"] = {"class_type": "PreviewAny", "inputs": {"source": ["design_qwen", 0]}, "_meta": {"title": "OUT model reply"}}
    g["out_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["p_dir", 0]}, "_meta": {"title": "OUT design prompt"}}
    g["out_locked"] = {"class_type": "PreviewAny", "inputs": {"source": ["design_pkg", 2]}, "_meta": {"title": "OUT locked"}}
    g["out_plan"] = {"class_type": "PreviewAny", "inputs": {"source": ["plan", 1]}, "_meta": {"title": "OUT plan json"}}
    return g


def _preview(g: dict, nid: str, src: list, title: str) -> None:
    g[nid] = {"class_type": "PreviewAny", "inputs": {"source": src}, "_meta": {"title": title}}


def graph_parse(base: dict, concept_json: dict, lyrics: str, direction: str, reply: str) -> dict:
    """phase B1: the (retimed) reply as a literal -> design_parse -> design_fix; no type / compose. Outputs: the designed piece json + the parse reports."""
    g = graph_common(base, concept_json, lyrics, direction)
    g["reply_in"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": reply}, "_meta": {"title": "model reply (Gemma 31B, retimed by own_video_v2.py retime)"}}
    g["design_parse"]["inputs"]["reply"] = ["reply_in", 0]
    g["design_parse"]["inputs"]["prompt"] = ["p_dir", 0]
    drop = {"design_qwen", "type_final", "compose", "assemble", "out_final", "archive", "pivot", "design_piece", "out_design_reply", "out_design_check", "out_design_final"}
    g = {k: v for k, v in g.items() if k not in drop}
    _preview(g, "out_piece_json", ["design_fix", 0], "OUT designed piece json")
    _preview(g, "out_parse_report", ["design_parse", 3], "OUT parse report")
    _preview(g, "out_fix_report", ["design_fix", 1], "OUT fix report")
    return g


def graph_full(base: dict, concept_json: dict, lyrics: str, direction: str, piece_json: str, label: str) -> dict:
    """phase B2: the whole redesign graph from the designed piece on; the piece json is a literal (the parse step's output, plus the compose edits of `behind`):
    type passes -> 3D cuts -> compose -> assemble -> archive. design_pkg stays only for the archive's design sheet."""
    g = graph_common(base, concept_json, lyrics, direction)
    g["piece_in"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": piece_json}, "_meta": {"title": "designed piece json (Gemma 31B design, retimed + behind edit)"}}
    g["design_piece"]["inputs"]["piece_json"] = ["piece_in", 0]
    drop = {"design_qwen", "design_parse", "design_qwen2", "design_fix", "out_design_reply", "out_design_check", "out_design_final"}
    g = {k: v for k, v in g.items() if k not in drop}
    g["out_final"]["inputs"]["filename_prefix"] = f"projects/COMPETITIONS/motion-graphic/own-video/{VERSION}/{STEM}_{label}_video"
    g["archive"]["inputs"]["filename_prefix"] = f"projects/COMPETITIONS/motion-graphic/own-video/{VERSION}/{STEM}_{label}_out"
    return g


# ----------------------------------------------------------------------------------------------------------------------------------- main
def _write(path: str, obj, js=True):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False) if js else obj)


def cmd_prep(a) -> int:
    os.makedirs(a.work, exist_ok=True)
    ana = json.load(open(a.analysis, encoding="utf-8"))
    music = V1.music_from(ana)
    spans = V1.spans_of(music)
    cuts24 = json.load(open(a.cuts24, encoding="utf-8"))
    segs = segments(cuts24, music["nframes"], spans)
    reading = json.load(open(a.reading, encoding="utf-8"))
    words = picked_words(reading)
    _write(os.path.join(a.work, "spans.json"), dict(spans=spans, starts=[sum(spans[:i]) for i in range(len(spans))]))
    _write(os.path.join(a.work, "segments.json"), segs)
    _write(os.path.join(a.work, "box_cut_v2.sh"), box_script(spans, segs), js=False)
    for s in segs:
        if s["person"]:
            _write(os.path.join(a.work, f"alpha_h{s['job']}.json"), alpha_graph_seg(s["job"], s["na"]))
    c = concept(words)
    lyrics = json.load(open(a.lyrics, encoding="utf-8"))
    _write(os.path.join(a.work, "concept.json"), c)
    _write(os.path.join(a.work, "words.json"), words)
    _write(os.path.join(a.work, "lyrics_line.txt"), lyrics_line(lyrics), js=False)
    _write(os.path.join(a.work, "direction.txt"), direction_text(words), js=False)
    piece, index = V1.piece_stub(ana, music, "\n".join(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"][:len(spans)]), "draft")
    _write(os.path.join(a.work, "piece.json"), piece)
    _write(os.path.join(a.work, "index.json"), index)
    print("spans", spans)
    for s in segs:
        print(f"  seg {s['i']:2d} shot s{s['shot']} frames {s['a']:3d}-{s['b']:3d} ({s['n']:2d}) {s['label']:18s} " + (f"alpha job h{s['job']} ({s['na']} frames)" if s["person"] else "empty matte"))
    return 0


def extract_json(text: str) -> dict:
    """the model's reply (possibly fenced) -> dict."""
    t = text.strip()
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


def retime_reply(obj: dict, words: list[dict], bpm: float = 90.0) -> list[str]:
    """write the reading's landing times into the model's reply (the planner's own timing input: type.words[i].hit / hold_s / form_s / exit_s; qwen_piece.compile_timing solves t0 / t1
    from them with spec.plan_timing). Every word: hit = the reading's beat-snapped `land`, form-in 1 beat (the first word 2: it starts at 0 s), exit 1 beat, hold = the beats to the
    next word minus exit, the next word's form-in and a one-beat gap (>= 2 beats). Also: centred and still placement, a camera that does not drift (read_dps at its floor 2), at most one 3D cut."""
    b = 60.0 / bpm
    ws = obj["type"]["words"]
    if len(ws) != len(words):
        raise SystemExit(f"reply has {len(ws)} words, the reading gives {len(words)}")
    notes = []
    for i, (w, lw) in enumerate(zip(ws, words)):
        if " ".join(str(w.get("text", "")).split()) != lw["text"]:
            raise SystemExit(f"word {i}: reply {w.get('text')!r} != {lw['text']!r}")
        nb = round((words[i + 1]["land"] - lw["land"]) / b) if i + 1 < len(ws) else 6
        old = {k: w.get(k) for k in ("hit", "hold_s", "form_s", "exit_s")}
        w["hit"] = round(lw["land"], 3)
        w["form_s"] = round((2 if i == 0 else 1) * b, 2)
        w["exit_s"] = round(b, 2)
        w["hold_s"] = round(max(2, nb - 3) * b, 2)
        pl = w.setdefault("placement", {})
        pl.update(yaw=0, pitch=0, roll=0, dx=0, dy=0)
        if w.get("render") not in (None, "flat"):                       # no LTX text pass in this route: every word is drawn by the type engine
            notes.append(f"{lw['text']}: render {w.get('render')} -> flat")
        w["render"] = "flat"
        for k in ("ltx_at", "ltx_t", "material"):
            w.pop(k, None)
        notes.append(f"{lw['text']}: {old} -> hit {w['hit']} form {w['form_s']} hold {w['hold_s']} exit {w['exit_s']}")
    cp = obj["type"].setdefault("camera_params", {})
    cp.update(read_dps=2.0, intro_deg=0.0, elev=0.0, bob=0.0)
    seen3d = False
    for c in (obj.get("compose") or {}).get("cuts") or []:
        kind = str(c.get("cut", "hard")).lower()
        if kind not in ("hard", "cut") and "window" not in kind:       # a text-as-window cut is not a 3D cut: it stays
            if seen3d:
                notes.append(f"cut after {c.get('after')}: {kind} -> hard (at most one designed cut)")
                c["cut"] = "hard"
            seen3d = True
    return notes


def cmd_retime(a) -> int:
    words = json.load(open(os.path.join(a.work, "words.json"), encoding="utf-8"))
    obj = extract_json(open(a.reply, encoding="utf-8").read())
    notes = retime_reply(obj, words)
    out = os.path.join(a.work, f"reply_{a.label}.json")
    _write(out, obj)
    print(chr(10).join(notes))
    print("wrote", out)
    return 0


def cmd_parse(a) -> int:
    base = _load_base()
    c = json.load(open(os.path.join(a.work, "concept.json"), encoding="utf-8"))
    lyr = open(os.path.join(a.work, "lyrics_line.txt"), encoding="utf-8").read()
    d = open(os.path.join(a.work, "direction.txt"), encoding="utf-8").read()
    reply = open(a.reply, encoding="utf-8").read()
    _write(os.path.join(a.work, f"graph_parse_{a.label}.json"), graph_parse(base, c, lyr, d, reply))
    print("wrote", f"graph_parse_{a.label}.json")
    return 0


def cmd_full(a) -> int:
    base = _load_base()
    c = json.load(open(os.path.join(a.work, "concept.json"), encoding="utf-8"))
    lyr = open(os.path.join(a.work, "lyrics_line.txt"), encoding="utf-8").read()
    d = open(os.path.join(a.work, "direction.txt"), encoding="utf-8").read()
    pj = open(a.reply, encoding="utf-8").read()
    _write(os.path.join(a.work, f"graph_full_{a.label}.json"), graph_full(base, c, lyr, d, pj, a.label))
    print("wrote", f"graph_full_{a.label}.json")
    return 0


def cmd_design(a) -> int:
    base = _load_base()
    c = json.load(open(os.path.join(a.work, "concept.json"), encoding="utf-8"))
    lyr = open(os.path.join(a.work, "lyrics_line.txt"), encoding="utf-8").read()
    d = open(os.path.join(a.work, "direction.txt"), encoding="utf-8").read()
    _write(os.path.join(a.work, "graph_design.json"), graph_design(base, c, lyr, d))
    print("wrote graph_design.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "design", "retime", "parse", "full"])
    ap.add_argument("--analysis", default=os.path.join(ROOT, "25_music-read", "v001", "work", "your_clip_analysis.json"))
    ap.add_argument("--reading", default=os.path.join(ROOT, "25_music-read", "v001", "reading.json"))
    ap.add_argument("--lyrics", default=os.path.join(ROOT, "25_music-read", "v001", "lyrics_24s.json"))
    ap.add_argument("--cuts24")
    ap.add_argument("--work", required=True)
    ap.add_argument("--reply")
    ap.add_argument("--label", default="r1")
    a = ap.parse_args()
    if a.cmd == "prep":
        return cmd_prep(a)
    if a.cmd == "design":
        return cmd_design(a)
    return dict(full=cmd_full, retime=cmd_retime, parse=cmd_parse)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
