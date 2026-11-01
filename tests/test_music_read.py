"""music_read: video_cuts on a synthetic 2-colour clip, snapping to the beat grid, JSON parsing / validation, the prompt, the graph. No model, no GPU."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.fusion import music_read as MR  # noqa: E402

BPM = 90.0
PER = 60.0 / BPM


def fake_analysis(dur=24.0, first=1.0):
    beats = [round(first + i * PER, 4) for i in range(int((dur - first) / PER) - 1)]       # stops ~1 beat early on purpose (the tracker does)
    downs = [beats[i] for i in range(0, len(beats), 4)]
    n = int(dur * 24)
    t = np.arange(n) / 24
    rms = (0.4 + 0.4 * np.sin(t / 3)).tolist()
    curves = dict(fps=24, n=n, rms=rms, sub=rms, low=rms, mid=rms, high=rms, onset=[0.5] * n)
    bars = [dict(n=i + 1, start=downs[i], end=round(downs[i] + 4 * PER, 4), rms_db=-20.0, chord="Cm7") for i in range(len(downs) - 1)]
    return dict(duration=dur, bpm=BPM, bpm_candidates=[45, 90, 180], time_signature="4/4", bar_len=4 * PER, first_bar_start=first, beats=beats, downbeats=downs,
                key=dict(estimate="C minor"), bars=bars, onsets=[b + 0.01 for b in beats], curves=curves)


# ------------------------------------------------------------------------------------------------ video_cuts
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_video_cuts_synthetic_two_colour_clip(tmp_path):
    mp4 = tmp_path / "two.mp4"
    # 1 s red, 1 s blue, 1 s red, 30 fps: hard cuts at 1.0 and 2.0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "color=c=red:s=160x90:r=30:d=1", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=30:d=1", "-f", "lavfi", "-i", "color=c=red:s=160x90:r=30:d=1",
           "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", str(mp4)]
    subprocess.run(cmd, check=True)
    cuts = MR.video_cuts(mp4)
    assert len(cuts) == 2, cuts
    assert cuts[0] == pytest.approx(1.0, abs=0.05) and cuts[1] == pytest.approx(2.0, abs=0.05)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_video_cuts_none_in_a_single_colour_clip(tmp_path):
    mp4 = tmp_path / "one.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=green:s=160x90:r=30:d=2", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    assert MR.video_cuts(mp4) == []


# ------------------------------------------------------------------------------------------------ snapping
def test_snap_nearest_beat():
    g = np.array([1.0, 1.667, 2.333, 3.0])
    assert MR.snap(1.7, g) == (1, 1.667)
    assert MR.snap(2.9, g) == (3, 3.0)
    assert MR.snap(-5, g) == (0, 1.0)
    assert MR.snap(99, g) == (3, 3.0)


def test_grid_is_extended_to_the_end_of_the_clip():
    a = fake_analysis()
    b = MR._beats(a)
    assert len(b) > len(a["beats"])
    assert b[-1] + PER >= 24.0 - 0.2 - 1e-6
    assert np.allclose(np.diff(b), PER, atol=0.01)
    d = MR._downbeats(a)
    assert len(d) >= len(a["downbeats"]) and np.allclose(np.diff(d), 4 * PER, atol=0.02)


# ------------------------------------------------------------------------------------------------ parsing
def test_parse_plain_fenced_and_trailing_commas():
    assert MR.parse_reply('{"a": 1}')[0] == {"a": 1}
    assert MR.parse_reply('Sure!\n```json\n{"a": [1, 2,],}\n```')[0] == {"a": [1, 2]}
    assert MR.parse_reply('<think>hmm {x}</think>{"a": 2}')[0] == {"a": 2}
    assert MR.parse_reply("no json here")[0] is None
    assert MR.parse_reply("")[0] is None


def test_parse_truncated_reply_is_closed():
    o, notes = MR.parse_reply('{"sections": [{"start": 0, "label": "intro"}, {"start": 5, "label": "ver')
    assert o is not None and o["sections"][0] == {"start": 0, "label": "intro"} and notes
    rd, rep = MR.validate(o, fake_analysis())                                      # the cut-off item has no label: dropped, not invented
    assert [x["label"] for x in rd["sections"]] == ["intro"] and any("no label" in d for d in rep["dropped"])


# ------------------------------------------------------------------------------------------------ validation
def reply(**kw):
    base = dict(sections=[dict(start=0, end=10, label="intro", energy=0.3, why="x"), dict(start=11.0, end=24, label="drop", energy=1.4, why="y")], arc="up",
                hit_moments=[dict(t=3.1, beat_index=3, why="a")], word_slots=[dict(start=3.0, land=3.1, hold_until=3.2, suggest_len_words=9)],
                cut_suggestions=[dict(t=5.5, kind_hint="Hard Cut", why="c")], dropouts=[8.2], notes="n")
    base.update(kw)
    return base


def test_validate_snaps_everything_to_the_grid():
    a = fake_analysis()
    cuts = [5.0]
    rd, rep = MR.validate(reply(), a, cuts)
    assert MR.check_reading(rd, a) == []
    beats = MR._beats(a)
    for h in rd["hit_moments"]:
        assert min(abs(beats - h["t"])) < 1e-3 and h["t"] == pytest.approx(beats[h["beat_index"]], abs=1e-3)
    assert rd["sections"][0]["start"] == 0.0 and rd["sections"][-1]["end"] == 24.0
    assert rd["sections"][0]["end"] == rd["sections"][1]["start"]                  # contiguous
    assert rd["sections"][1]["energy"] == 1.0                                      # clamped
    assert rd["word_slots"][0]["hold_beats"] >= 1 and rd["word_slots"][0]["suggest_len_words"] == 4
    assert rd["cut_suggestions"][0]["kind_hint"] == "hard_cut"
    assert rep["repairs"]


def test_word_slot_held_less_than_one_beat_is_extended_and_overlaps_cut():
    a = fake_analysis()
    beats = MR._beats(a)
    r = reply(word_slots=[dict(start=beats[3], land=beats[3], hold_until=beats[3] + 0.1), dict(start=beats[4], land=beats[4], hold_until=beats[9]),
                          dict(start=beats[4], land=beats[4], hold_until=beats[9]), dict(start=beats[6], land=beats[6] + 0.01, hold_until=beats[7])])
    rd, rep = MR.validate(r, a)
    s = rd["word_slots"]
    assert [w["land_beat_index"] for w in s] == [3, 4, 6]                          # the duplicate land is dropped
    assert all(w["hold_beats"] >= 1 for w in s)
    assert s[1]["hold_until"] <= s[2]["land"] + 1e-6                               # one word at a time
    assert MR.check_reading(rd, a) == []
    assert any("already" in d or "< 1 beat after" in d for d in rep["dropped"])


def test_bad_labels_and_hits_off_by_a_lot():
    a = fake_analysis()
    r = reply(sections=[dict(start=0, label="Chorus", energy="0.7"), dict(start=12.0, label="???", energy=None)],
              hit_moments=[dict(t=3.1, beat_index=20), dict(t=None, beat_index=5), dict(t="x")])
    rd, rep = MR.validate(r, a)
    assert [s["label"] for s in rd["sections"]] == ["hook", "verse"]
    assert [h["beat_index"] for h in rd["hit_moments"]] == [3, 5]                  # t wins when it is within half a beat of its snap; beat_index alone is used when t is missing
    assert len(rd["hit_moments"]) == 2 and any("hit_moments[2]" in d for d in rep["dropped"])
    assert MR.check_reading(rd, a) == []


def test_reject_when_nothing_usable():
    a = fake_analysis()
    with pytest.raises(MR.ReadingRejected):
        MR.validate({"notes": "hi"}, a)
    with pytest.raises(MR.ReadingRejected):
        MR.validate([1, 2], a)
    with pytest.raises(MR.ReadingRejected):
        MR.validate({"sections": [{"label": "intro"}], "word_slots": [{"hold_until": 3}]}, a)


LYR = {"segments": [{"words": [{"w": "steam", "s": 0.0, "e": 0.3, "p": 0.9}, {"w": "Glow", "s": 3.0, "e": 3.4, "p": 0.9}, {"w": "train,", "s": 6.3, "e": 6.7, "p": 0.9},
                               {"w": "late", "s": 9.9, "e": 10.2, "p": 0.9}]}]}


def test_prominent_words_checked_against_the_lyrics_and_get_slots():
    a = fake_analysis()
    beats = MR._beats(a)
    r = reply(word_slots=[dict(start=beats[2], land=beats[2], hold_until=beats[4])],
              prominent_words=[dict(text="glow", sung_at=3.0, land=3.02, hold_until=3.7, why="image"),          # sung at 3.0 -> beat at 3.0 (the grid has 1.0 + k * 0.667)
                               dict(text="train", sung_at=6.3, land=40.0, hold_until=None, why="far land: pulled to the sung beat"),
                               dict(text="NOT IN LYRICS", sung_at=8.0, land=8.0, hold_until=9.0, why="x"),
                               dict(text="late", sung_at=9.0, land=9.3, hold_until=9.31, why="sung_at corrected to the lyric time")])
    rd, rep = MR.validate(r, a, lyrics=LYR)
    pw = rd["prominent_words"]
    assert [p["text"] for p in pw] == ["glow", "train", "late"]
    assert any("NOT IN LYRICS" in d for d in rep["dropped"])
    assert pw[2]["sung_at"] == pytest.approx(9.9)
    assert MR.check_reading(rd, a) == []
    lands = {w["land"] for w in rd["word_slots"]}
    assert all(p["land"] in lands for p in pw)                                    # each prominent word owns a slot
    assert all(p["hold_until"] - p["land"] >= PER * 0.98 for p in pw)
    assert all(abs(p["land"] - p["sung_at"]) <= 1.5 * PER for p in pw)


def test_prominent_word_replaces_a_model_slot_it_collides_with():
    a = fake_analysis()
    beats = MR._beats(a)
    r = reply(word_slots=[dict(start=beats[3], land=beats[3], hold_until=beats[5])],
              prominent_words=[dict(text="late", sung_at=float(beats[3]) + 0.0, land=float(beats[3]), hold_until=float(beats[6]), why="x")])
    lyr = {"segments": [{"words": [{"w": "late", "s": float(beats[3]), "e": float(beats[3]) + 0.3}]}]}
    rd, rep = MR.validate(r, a, lyrics=lyr)
    assert len(rd["word_slots"]) == 1 and rd["word_slots"][0].get("word") == "late"
    assert rd["word_slots"][0]["hold_beats"] == 3


def test_prompt_has_the_numbers_and_the_schema():
    a = fake_analysis()
    txt = MR.prompt(a, [3.0, 7.0], LYR)
    for needle in ("90.00 bpm", "C minor", "bar  1", "EXISTING CUTS", "3.00, 7.00", "LYRICS, timed", "glow", "prominent_words", "word_slots", "hit_moments", "cut_suggestions", "dropouts"):
        assert needle.lower() in txt.lower(), needle
    assert "continue the tracked grid" in txt                                     # the extended beats are labelled
    assert "prominent_words: []" in MR.prompt(a, [])


def test_timeline_png_size(tmp_path):
    from PIL import Image
    a = fake_analysis()
    p = MR.timeline_png(a, [3.0, 7.0], tmp_path / "t.png")
    assert Image.open(p).size == (1600, 900)


def test_graph_matches_the_default_gemma_node():
    from tools.app import music_read_graph as G
    g = G.build("x.png", "hello", seed=3)
    n = g["music_read"]["inputs"]
    assert g["music_read"]["class_type"] == "MographGGUFVL" and g["image"]["class_type"] == "LoadImage"
    assert n["image"] == ["image", 0] and n["prompt"] == "hello" and n["json_mode"] is True
    assert g["out_reply"]["inputs"]["source"] == ["music_read", 0]
    default = ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers_pivot_gemma31b_qat.json"
    if default.is_file():
        d = json.loads(default.read_text(encoding="utf-8")).get("design_qwen")
        if d:
            for k in ("model", "mmproj", "n_gpu_layers", "n_ctx", "kv_cache", "flash_attn", "mmproj_on_gpu", "handler", "top_p", "top_k"):
                assert n[k] == d["inputs"][k], k


def test_reading_md_renders():
    a = fake_analysis()
    rd, rep = MR.validate(reply(), a, [5.0])
    md = MR.reading_md(rd, a, rep, dict(total_s=10, gen_tokens=5, gen_tok_s=1.0), 3.0)
    assert "Sections: Gemma vs the DSP curves" in md and "| intro |" in md and "Timings: DSP 3.0 s" in md
