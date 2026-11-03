#!/usr/bin/env python
"""Own-video route, OWN-FOOTAGE DEMO (v007): a dropped video that is OUR footage (clean LTX shot clips of a finished piece + its own ACE instrumental), no lyrics.

    python tools/app/own_video_v7.py slots --reading R/reading.json --spans-from analysis.json --frames 384 --n 4 --out slots.json
    python tools/app/own_video_v7.py wordsgraph --slots slots.json --reading R/reading.json --theme "..." --out graph_words.json
    python tools/app/own_video_v7.py wordsparse --reply reply.txt --slots slots.json --out words.json
    python tools/app/own_video_v7.py prep --ingest ingest.json --analysis A.json --reading R.json --words words.json --work DIR --box-src /workspace/own_video/demo/ingested/X.mp4

What is different from the v003 route (the same functions are reused, `own_video_v3.py`):
  * INGEST (tools/app/ingest.py) is the first step: the box runs `ingest.py plan` + `render`: fps policy (integer-ratio only, here 24 -> 24 kept), the aspect plan (native 9:16: no crop) and the
    normalised clip at the compose tier; `prep` reads fps / frame count / shot cuts from ingest.json. That REPLACES the route's own `--src-fps` argument, its hard-coded cut list (a scene-detect
    list written by hand for the test_clip clip) and the clip's own fps handling; the route's remaining crop (scale + centre crop to the 384x704 shot size) stays, it is not an ingest tier.
  * the stretched clock is 1.0 (a 24 fps source needs none): music grid, landing times and audio are used as they are.
  * NO LYRICS: `slots` picks one beat-snapped word slot per word from the music reading (>= 3.2 s apart) and `wordsgraph` / `wordsparse` let Gemma 31B choose the WORDS from a short theme line
    + the reading's sections / hit moments (the fallback when Whisper finds < 3 words or the track is instrumental); the design prompt keeps its truthful "no lyrics" line.
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
from tools.app import own_video_v3 as V3       # noqa: E402


def pick_slots(reading: dict, n: int, frames: int, fps: float = 24.0, min_gap: float = 3.2, first_min: float = 1.4, tail: float = 2.0) -> list[dict]:
    """n word slots from the reading's `word_slots` (beat-snapped, held >= 1 beat): at least `min_gap` s apart (form-in + 2-beat hold + exit + the transition), the last one `tail` s before the
    end; picked greedily from the LATEST backwards so the pieces end on a word, then sorted. -> [{land, hold_until, section}]."""
    dur = frames / fps
    cands = [s for s in reading["word_slots"] if first_min <= s["land"] <= dur - tail]
    chosen: list[dict] = []
    for s in sorted(cands, key=lambda x: -x["land"]):
        if all(abs(s["land"] - c["land"]) >= min_gap for c in chosen):
            chosen.append(s)
        if len(chosen) == n:
            break
    chosen.sort(key=lambda x: x["land"])
    secs = reading["sections"]
    out = []
    for s in chosen:
        sec = next((x["label"] for x in secs if x["start"] <= s["land"] < x["end"]), "")
        out.append(dict(land=float(s["land"]), hold_until=float(s["hold_until"]), section=sec))
    return out


def words_prompt(slots: list[dict], reading: dict, theme: str) -> str:
    hits = "; ".join(f"{h['t']:.2f} s ({h.get('why', '')})" for h in reading.get("hit_moments", []))
    secs = "; ".join(f"{s['start']:.1f}-{s['end']:.1f} s {s['label']} (energy {s.get('energy', '')})" for s in reading["sections"])
    sl = "\n".join(f"  slot {i + 1}: lands at {s['land']:.2f} s, in the {s['section']} section" for i, s in enumerate(slots))
    return (
        "You choose the words that a motion-graphics editor sets as big type over a short music video. The track is INSTRUMENTAL (no lyrics), so the words come from the picture's theme and "
        "from how the music moves.\n\n"
        f"THEME (what the footage shows): {theme}\n"
        f"MUSIC READING: {reading.get('arc', '')} Sections: {secs}. Strongest moments: {hits}.\n\n"
        f"WORD SLOTS (each word lands ON a beat at its slot time and holds there):\n{sl}\n\n"
        f"Pick exactly {len(slots)} lockups, one per slot, in slot order. Each lockup is 1 or 2 UPPERCASE words, concrete nouns or strong verbs from the theme (never articles, 'on', 'the'), "
        "at most 14 letters in all; the later slots may carry the stronger words (the music builds). All different words.\n"
        'Reply with JSON only: {"lines":[{"slot":1,"text":"WORD","why":"..."}]}')


def parse_words(reply: str, slots: list[dict]) -> list[dict]:
    t = reply.strip()
    obj = json.loads(t[t.find("{"):t.rfind("}") + 1])
    lines = sorted(obj["lines"], key=lambda x: int(x["slot"]))
    if len(lines) != len(slots):
        raise SystemExit(f"{len(lines)} lines for {len(slots)} slots")
    out = []
    for ln, s in zip(lines, slots):
        text = " ".join(str(ln["text"]).upper().split())
        out.append(dict(text=text, land=round(s["land"], 4), hold_until=round(s["hold_until"], 4), land_real=round(s["land"], 4), why=ln.get("why", "")))
    return out


def cmd_prep(a) -> int:
    ing = json.load(open(a.ingest, encoding="utf-8"))
    fps = float(ing["fps"]["fps_float"])
    K = V3.K_of(fps)
    nframes = min(int(ing["frames"]), int(round(a.window * fps))) if a.window else int(ing["frames"])
    cuts = [int(s["f0"]) for s in ing["shots"][1:] if int(s["f0"]) < nframes]
    ana = json.load(open(a.analysis, encoding="utf-8"))
    words = json.load(open(a.words, encoding="utf-8"))
    os.makedirs(a.work, exist_ok=True)
    # the v003 module's globals point at this version / this clip
    V3.VERSION, V3.STEM, V3.ARCHIVE = a.version, f"motion-graphic_own-video_{a.version}", f"own_video/arch_{a.version}"
    V3.BOX_DIR = f"{V3.BOX}/{a.box_dir}"
    V3.SRC_BOX = a.box_src
    V3.AUDIO_X = a.box_src                                   # the ingested clip carries the track: it is the graph's source_video
    V3.H_NAME = f"own_video_{a.box_dir}_h"
    V3.SCENES.setdefault("subject", a.scene)
    V3.patch_v2()
    V2.SEG_LABELS = [("subject", True)] * (len(cuts) + 1)
    from tools.fusion import plan_rules as PR
    music = V3.music_x(ana, K, nframes)
    spans = PR.cut_spans(nframes, PR.make_grid(music, music["cut"]), V3.PIPE_FPS)
    segs = V2.segments(cuts, nframes, spans)
    V2._write(os.path.join(a.work, "spans.json"), dict(spans=spans, starts=[sum(spans[:i]) for i in range(len(spans))], src_fps=fps, K=K))
    V2._write(os.path.join(a.work, "segments.json"), segs)
    V2._write(os.path.join(a.work, "box_cut_v3.sh"), V3.box_script(spans, segs, fps), js=False)
    for s in segs:
        if s["person"]:
            V2._write(os.path.join(a.work, f"alpha_h{s['job']}.json"), V3.alpha_graph(s["job"], s["na"]))
    c = V3.concept(words)
    c["length"] = int(round(nframes / V3.PIPE_FPS))
    c["scene"] = a.scene
    c["music"] = f"our own instrumental ACE-Step track for this picture ({ana['bpm']:.0f} bpm, {ana.get('key', {}).get('estimate', '') if isinstance(ana.get('key'), dict) else ana.get('key', '')}), no vocals"
    V2._write(os.path.join(a.work, "concept.json"), c)
    V2._write(os.path.join(a.work, "words.json"), words)
    V2._write(os.path.join(a.work, "lyrics_line.txt"), "Lyrics: none (instrumental).", js=False)
    test_clip = ("The clip is the user's own music video: a rapper in a New York test clip station and at a red-shuttered corner, cut with inserts "
              "(train, graffiti, street, a gold chain close-up, a crowd). ")
    assert test_clip in V3.DIRECTION4
    ctx = f"The clip is OUR OWN footage: {a.scene}; {len(spans)} shots of the same subject, set to our own instrumental track (no vocals, no lyrics). "
    V2._write(os.path.join(a.work, "direction.txt"), V3.DIRECTION4.replace(test_clip, ctx).format(hits=", ".join(f"{w['text']} {w['land']:.2f}" for w in words)), js=False)
    piece, index = V3.piece_stub(ana, music, "\n".join(["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"][:len(spans)]), nframes)
    V2._write(os.path.join(a.work, "piece.json"), piece)
    V2._write(os.path.join(a.work, "index.json"), index)
    print("ingest fps", fps, "policy", ing["fps"]["policy"], "K", K, "frames", nframes, "spans", spans, "real cuts (ingest shots)", cuts)
    for s in segs:
        print(f"  seg {s['i']} shot s{s['shot']} frames {s['a']}-{s['b']} ({s['n']}) alpha h{s['job']} ({s['na']} frames)")
    return 0


def main() -> int:
    V3.JOIN_REAL_CUTS = True
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["slots", "wordsgraph", "wordsparse", "prep", "plates"])
    ap.add_argument("--reading")
    ap.add_argument("--frames", type=int, default=384)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--slots")
    ap.add_argument("--theme", default="")
    ap.add_argument("--reply")
    ap.add_argument("--ingest")
    ap.add_argument("--analysis")
    ap.add_argument("--words")
    ap.add_argument("--work")
    ap.add_argument("--window", type=float, default=0.0, help="seconds of the clip the piece uses (0 = all)")
    ap.add_argument("--version", default="v007")
    ap.add_argument("--box-dir", default="v7")
    ap.add_argument("--box-src")
    ap.add_argument("--moments-file")
    ap.add_argument("--cut-plates", default="", help="plates: piece_final.json; also make clean plates for the shots either side of its 3D cuts")
    ap.add_argument("--scene", default="a sunlit pottery studio with shelves of clay pots, dust in the light beams")
    a = ap.parse_args()
    if a.cmd == "slots":
        rd = json.load(open(a.reading, encoding="utf-8"))
        json.dump(pick_slots(rd, a.n, a.frames), open(a.out, "w"), indent=1)
        print(open(a.out).read())
        return 0
    if a.cmd == "wordsgraph":
        rd = json.load(open(a.reading, encoding="utf-8"))
        sl = json.load(open(a.slots, encoding="utf-8"))
        json.dump(V1.words_graph(words_prompt(sl, rd, a.theme)), open(a.out, "w"), indent=1)
        return 0
    if a.cmd == "wordsparse":
        sl = json.load(open(a.slots, encoding="utf-8"))
        w = parse_words(open(a.reply, encoding="utf-8").read(), sl)
        json.dump(w, open(a.out, "w"), indent=1)
        print(json.dumps(w))
        return 0
    if a.cmd == "plates":
        V3.VERSION, V3.STEM = a.version, f"motion-graphic_own-video_{a.version}"
        V3.H_NAME = f"own_video_{a.box_dir}_h"
        V3.SCENES["subject"] = a.scene
        a.moments = a.moments_file
        a.cut_plates = a.cut_plates or None
        return V3.cmd_plates(a)
    return cmd_prep(a)


if __name__ == "__main__":
    raise SystemExit(main())
