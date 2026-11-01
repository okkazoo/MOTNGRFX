"""media_read: the READ step of the Add-media box. Prompt / sheet build on a synthetic clip + images, parse guarantees (quoted text verbatim, no invented on_screen lines,
window clamp, palette from pixels, music from the DSP numbers, fallback split), the graph. No model, no GPU. Needs ffmpeg for the pack tests (skipped without)."""
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

from tools.fusion import media_read as MR  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
TEXT = '"HOLD ON" "LET GO"\ngrainy warm film, slow cuts, dusk light'


def info_for(dur=24.0, music=None):
    return dict(v=1, length=12, items=[
        dict(id="V1", kind="video", file="motngrfx_clip.mp4", duration=dur, w=540, h=960, tiles=[1.0, 2.0], palette_px=["#112233", "#445566", "#778899"]),
        dict(id="I1", kind="image", file="motngrfx_a.png", w=100, h=100, palette_px=["#ff0000", "#00ff00"])], music=music, music_detail=None, audio_from=None)


def media(text=TEXT, length=12):
    return json.dumps(dict(video="motngrfx_clip.mp4", images=["motngrfx_a.png"], audio=None, text=text, length=length, format="9:16"))


# ---------------------------------------------------------------------------------------- parse
def test_quoted_words_forced_and_verbatim():
    reply = json.dumps(dict(on_screen=["hold on", "Grainy warm film"], concept="a farewell", look="grainy warm film, slow cuts, dusk light",
                            media=[dict(id="V1", shows="a test clip", window=[3, 15], mood="calm", palette=["#aabbcc", "#112233"])]))
    r = MR.parse(reply, info_for(), media())
    assert r["on_screen"][0] == "HOLD ON" and r["on_screen"][1] == "LET GO"        # forced, in text order, verbatim case (the model wrote 'hold on')
    assert r["on_screen"].count("HOLD ON") == 1 and "hold on" not in r["on_screen"]
    assert r["v"] == 1 and set(MR.KEYS) <= set(r)


def test_model_cannot_invent_on_screen_lines():
    reply = json.dumps(dict(on_screen=["HOLD ON", "NEVER GIVE UP", "slow cuts"], concept="x", look="y", media=[]))
    r = MR.parse(reply, info_for(), media())
    assert "NEVER GIVE UP" not in r["on_screen"]
    assert "slow cuts" in r["on_screen"]                  # it IS in the user's text: kept as the verbatim slice
    assert "not in the user's text" in r["notes"]


def test_no_text_means_empty_fields():
    reply = json.dumps(dict(on_screen=["INVENTED"], concept="a story", look="moody", media=[]))
    r = MR.parse(reply, info_for(), media(text="  "))
    assert r["on_screen"] == [] and r["concept"] == "" and r["look"] == ""


def test_window_clamped_and_length_enforced():
    for w, want in (([20, 32], [12.0, 24.0]), ([-5, 7], [0.0, 12.0]), ([3.3, 4.0], [3.3, 15.3]), ("junk", [0.0, 12.0]), (None, [0.0, 12.0])):
        r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=w)])), info_for(24.0), media())
        assert r["media"][0]["window"] == want, (w, r["media"][0]["window"])
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[0, 8])])), info_for(8.0), media(length=12))     # clip shorter than the request: the whole clip
    assert r["media"][0]["window"] == [0.0, 8.0]
    assert r["media"][1]["window"] is None                 # images have no window


def test_palette_from_pixels_when_model_missing_or_invalid():
    reply = json.dumps(dict(media=[dict(id="V1", palette=["red", "#zzz"]), dict(id="I1", palette=["#ABC", "#102030"])]))
    r = MR.parse(reply, info_for(), media())
    assert r["media"][0]["palette"] == ["#112233", "#445566", "#778899"]
    assert r["media"][1]["palette"] == ["#aabbcc", "#102030"]


def test_music_from_dsp_never_model():
    r = MR.parse(json.dumps(dict(music={"bpm": 200, "key": "C major"})), info_for(music={"bpm": 90.0, "key": "D# minor"}), media())
    assert r["music"] == {"bpm": 90.0, "key": "D# minor"}
    assert MR.parse("{}", info_for(), media())["music"] is None


def test_unparseable_reply_falls_back_to_rule_split():
    r = MR.parse("sorry, I cannot do that", info_for(), media())
    assert r["on_screen"] == ["HOLD ON", "LET GO"]
    assert "grainy warm film" in r["concept"] and '"' not in r["concept"]
    assert "NOT USABLE" in r["notes"]
    assert [m["id"] for m in r["media"]] == ["V1", "I1"] and r["media"][0]["window"] == [0.0, 12.0]


def test_fallback_split_lyric_lines_and_instructions():
    on, concept = MR.fallback_split("hold on\nlet it go\nwe were young\nmake it grainy and warm, slow cuts please")
    assert on == ["hold on", "let it go", "we were young"] and concept.startswith("make it grainy")
    on, concept = MR.fallback_split("a quiet film about a train at dusk")
    assert on == [] and concept


def test_reply_forms_fence_and_truncated():
    good = dict(on_screen=["LET GO"], concept="c", look="l", media=[])
    r = MR.parse("```json\n" + json.dumps(good) + "\n```", info_for(), media())
    assert r["concept"] == "c" and "LET GO" in r["on_screen"]
    cut = json.dumps(good)[:-12] + ', "notes": "unfini'
    r = MR.parse(cut, info_for(), media())
    assert r["v"] == 1 and "on_screen" in r


def test_curly_quotes_and_multiline_quote():
    assert MR.quoted_spans("say “HELLO THERE” ok")[0][2] == "HELLO THERE"
    on, n = MR.merge_on_screen('"one\ntwo" and more', [])
    assert on == ["one", "two"]


def test_on_screen_model_line_matches_despite_punctuation():
    on, _ = MR.merge_on_screen("hold on,\nlet go!", ["Hold on", "let go"])
    assert on == ["hold on", "let go"]


# ---------------------------------------------------------------------------------------- pack
def test_palette_kmeans_two_colours():
    a = np.zeros((40, 40, 3), np.uint8)
    a[:, :20] = (200, 30, 30)
    a[:, 20:] = (20, 20, 220)
    pal = MR.kmeans_palette([a], 3)
    assert len(pal) == 2 and "#c81e1e" in pal and "#1414dc" in pal


def test_sheet_fits():
    v = [(f"V1 {i}.0s", np.full((360, 202, 3), 40 + i * 10, np.uint8)) for i in range(16)]
    im = [(f"I{k}", np.full((300, 400, 3), 90, np.uint8)) for k in range(1, 6)]
    s = MR.contact_sheet(v, im)
    assert max(s.shape[:2]) <= 1024 + 8 and s.shape[2] == 3
    assert MR.contact_sheet([], []).shape[0] > 0
    assert max(MR.contact_sheet(v[:1], []).shape[:2]) <= 1024


def test_tile_times():
    assert len(MR.tile_times(24.0)) == 16 and len(MR.tile_times(5.0)) == 5 and len(MR.tile_times(0.4)) == 1
    t = MR.tile_times(10.0)
    assert t[0] == 0.5 and t[-1] == 9.5


def test_resolve_refuses_escape(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    assert MR.resolve("a.png", tmp_path).name == "a.png"
    with pytest.raises(ValueError):
        MR.resolve("../x.png", tmp_path)
    with pytest.raises(FileNotFoundError):
        MR.resolve("nope.png", tmp_path)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pack_video_image_text(tmp_path):
    from PIL import Image
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=12:duration=6", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                    "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "clip.mp4")], check=True)
    Image.fromarray(np.full((200, 300, 3), (180, 60, 60), np.uint8)).save(tmp_path / "a.png")
    m = json.dumps(dict(video="clip.mp4", images=["a.png"], audio=None, text=TEXT, length=4, format="9:16"))
    sheet, prompt, info = MR.pack(m, tmp_path, run_dsp=False)
    assert max(sheet.shape[:2]) <= 1032
    assert [i["id"] for i in info["items"]] == ["V1", "I1"] and len(info["items"][0]["tiles"]) == 6
    assert abs(info["items"][0]["duration"] - 6.0) < 0.3 and info["items"][0]["has_audio"]
    assert info["items"][1]["palette_px"][0] == "#b43c3c"
    assert TEXT in prompt and "exactly 4 s long" in prompt
    r = MR.parse(json.dumps(dict(on_screen=["HOLD ON"], media=[dict(id="V1", window=[4.5, 9])])), info, m)
    w = r["media"][0]["window"]
    assert 0 <= w[0] and w[1] <= info["items"][0]["duration"] + 0.1 and abs((w[1] - w[0]) - 4) < 0.15
    json.dumps(r)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pack_audio_only_dsp(tmp_path):
    pytest.importorskip("soundfile")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110:duration=8", str(tmp_path / "s.wav")], check=True)
    m = json.dumps(dict(video=None, images=[], audio="s.wav", text="", length=8, format="9:16"))
    sheet, prompt, info = MR.pack(m, tmp_path)
    assert sheet.shape[0] > 0 and "no picture" in prompt and "(the user pasted no text)" in prompt
    assert info["items"][0]["id"] == "A1"
    r = MR.parse("{}", info, m)
    assert r["media"][0]["kind"] == "audio" and r["media"][0]["file"] == "s.wav"
    if info["music"]:
        assert r["music"] == info["music"] and "bpm" in r["media"][0]["shows"]


# ---------------------------------------------------------------------------------------- graph
def test_graph_shape():
    from tools.app import media_read_graph as G
    g = G.build()
    titles = {n["_meta"]["title"]: k for k, n in g.items()}
    assert "IN media" in titles and "OUT read" in titles
    assert g[titles["IN media"]]["class_type"] == "PrimitiveStringMultiline" and g[titles["OUT read"]]["class_type"] == "PreviewAny"
    llm = g["read_llm"]["inputs"]
    assert llm["json_mode"] is True and llm["keep_model_loaded"] is True and llm["mmproj"].endswith("mmproj-F16.gguf") and llm["n_ctx"] == G.READ_N_CTX == 12288
    assert g["parse"]["inputs"]["reply"] == ["read_llm", 0] and g["read_llm"]["inputs"]["image"] == ["pack", 0]
    assert all(n["class_type"] != "SaveImage" or str(n["inputs"].get("filename_prefix", "")).startswith("motngrfx_app/") for n in g.values())       # App label (MEDIA_BOX.md): none today, a save node would need it
    on_disk = json.loads((ROOT / "app" / "graphs" / "_media_read.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(G.build()))


def test_read_n_ctx_sized_to_the_prompt():
    from tools.app import media_read_graph as G
    from tools.app import llm_arms as LA
    assert G.READ_N_CTX in (8192, 12288, 16384) and G.READ_N_CTX < LA.ARMS["gemma31b_qat"]["n_ctx"]      # own value; the arm is untouched
    assert LA.ARMS["gemma31b_qat"]["n_ctx"] == 32768
    need = sum(G.READ_BUDGET.values())
    assert need + 1500 <= G.READ_N_CTX                      # >= 1.5 k tokens headroom over the worst case
    assert G.READ_N_CTX / 1.5 < need * 1.6 + 1                # ... and not oversized: the next size down would not hold it
    # the real built prompt (instructions part, a 3-media case) is inside the instruction budget at 3 chars per token
    info = info_for()
    p = MR.build_prompt(MR.parse_media(media(text="")), info["items"], None, None)
    assert len(p) / 3 <= G.READ_BUDGET["instructions"] + 50
    assert G.build()["read_llm"]["inputs"]["max_tokens"] == G.READ_BUDGET["reply"]


# ---------------------------------------------------------------------------------------- lyrics (Whisper, CPU, optional)
def asr_json(words_per_seg=((("hold", 0.9), ("on", 0.9), ("tight", 0.8)), (("let", 0.9), ("go", 0.2), ("now", 0.9), ("please", 0.9)))):
    segs, t = [], 1.0
    for ws in words_per_seg:
        words = []
        for w, p in ws:
            words.append(dict(w=w, s=round(t, 2), e=round(t + 0.4, 2), p=p))
            t += 0.5
        segs.append(dict(start=words[0]["s"], end=words[-1]["e"], text=" ".join(w for w, _ in ws), words=words))
    return dict(segments=segs)


def test_lyrics_from_asr_drops_low_prob_words_and_no_vocals():
    lines, note = MR.lyrics_from_asr(asr_json())
    assert note is None and [l["text"] for l in lines] == ["hold on tight", "let now please"]     # 'go' (p 0.2) dropped
    assert lines[0]["t"] == 1.0
    lines, note = MR.lyrics_from_asr(asr_json(((("la", 0.9),),)))
    assert lines == [] and "no vocals" in note
    lines, note = MR.lyrics_from_asr(asr_json(((("a", 0.35), ("b", 0.4), ("c", 0.4), ("d", 0.35), ("e", 0.4)),)))
    assert lines == [] and "no clear vocals" in note


def test_transcribe_skips_without_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(MR, "ASR_PYTHON_DEFAULT", str(tmp_path / "nope"))
    monkeypatch.delenv("MOTNGRFX_ASR_PYTHON", raising=False)
    lines, note = MR.transcribe(tmp_path / "x.wav")
    assert lines == [] and "Whisper venv not found" in note


def test_transcribe_runs_in_cpu_only_subprocess(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        Path(cmd[-1]).write_text(json.dumps(asr_json()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    py = tmp_path / "py"
    py.write_text("x")
    monkeypatch.setenv("MOTNGRFX_ASR_PYTHON", str(py))
    monkeypatch.setattr(MR.subprocess, "run", fake_run)
    lines, note = MR.transcribe(tmp_path / "x.wav")
    assert note is None and len(lines) == 2 and seen["env"]["CUDA_VISIBLE_DEVICES"] == "" and seen["cmd"][0] == str(py) and seen["cmd"][1].endswith("asr_whisper.py")
    monkeypatch.setattr(MR.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "boom"))
    lines, note = MR.transcribe(tmp_path / "x.wav")
    assert lines == [] and "Whisper failed" in note


LYR = [dict(t=1.0, end=3.0, text="hold on tight"), dict(t=14.0, end=16.0, text="let me go now"), dict(t=20.0, end=22.0, text="never again")]


def test_on_screen_from_lyrics_only_when_no_quotes():
    ltxt = MR.lyrics_text(LYR)
    on, n = MR.merge_on_screen("a moody film at dusk", ["let me go now", "INVENTED WORD", "hold on"], ltxt)
    assert on == ["hold on", "let me go now"] or on == ["let me go now", "hold on"] or set(on) == {"hold on", "let me go now"}
    assert [x.lower() for x in on] == ["hold on", "let me go now"]                       # in the order sung
    assert "INVENTED WORD" not in on and "lyrics" in " ".join(n)
    on, n = MR.merge_on_screen('"HOLD ON" and some look text', ["never again", "HOLD ON"], ltxt)          # quoted words: the lyrics are not used
    assert on == ["HOLD ON"]
    on, _ = MR.merge_on_screen("", ["never again", "made up"], ltxt)                       # no text at all: lyric lines still allowed, nothing invented
    assert on == ["never again"]
    on, _ = MR.merge_on_screen("", ["never again"], "")
    assert on == []


def test_on_screen_lyric_lines_capped_and_deduped():
    lyr = [dict(t=float(i), end=i + 0.5, text=f"word{i} here") for i in range(12)]
    on, _ = MR.merge_on_screen("", [f"word{i} here" for i in range(12)] + ["word1 here"], MR.lyrics_text(lyr))
    assert len(on) == MR.MAX_LYRIC_LINES and len(set(on)) == len(on)


def test_parse_lyrics_key_window_filter_and_notes():
    inf = info_for(24.0)
    inf.update(lyrics=LYR, audio_from="V1", notes=["lyrics skipped: test note"])
    reply = json.dumps(dict(on_screen=["let me go now"], media=[dict(id="V1", window=[12, 24])]))
    r = MR.parse(reply, inf, media(text="", length=12))
    assert [l["text"] for l in r["lyrics"]] == ["let me go now", "never again"] and all(set(l) <= {"t", "end", "text", "ts", "words"} and {"t", "text"} <= set(l) for l in r["lyrics"])    # 'hold on tight' (t=1) is outside 12-24
    assert r["on_screen"] == ["let me go now"] and "lyrics skipped: test note" in r["notes"]
    assert MR.parse("{}", info_for(), media())["lyrics"] == []                                     # key always present
    assert set(MR.KEYS) <= set(r)


def test_prompt_carries_the_lyrics_and_the_lyrics_rule():
    info = info_for()
    p = MR.build_prompt(MR.parse_media(media(text="")), info["items"], None, "V1", LYR)
    assert "SUNG LYRICS" in p and "[14.0s] let me go now" in p and "prominent sung words" in p
    assert "NEVER ends on a small word" in p and "at most 4 words" in p                                  # the phrase rule is in the READ prompt too
    p0 = MR.build_prompt(MR.parse_media(media(text="")), info["items"], None, None, [])
    assert "SUNG LYRICS" not in p0 and "prominent sung words" not in p0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pack_puts_lyrics_in_prompt_and_info(tmp_path, monkeypatch):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=12:duration=6", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                    "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "clip.mp4")], check=True)
    monkeypatch.setattr(MR, "transcribe", lambda wav, timeout=600.0: (LYR[:2], None))
    m = json.dumps(dict(video="clip.mp4", images=[], audio=None, text="", length=4, format="9:16"))
    sheet, prompt, info = MR.pack(m, tmp_path, run_dsp=False)
    assert info["lyrics"] == LYR[:2] and info["audio_from"] == "V1" and "[1.0s] hold on tight" in prompt
    monkeypatch.setattr(MR, "transcribe", lambda wav, timeout=600.0: ([], "lyrics skipped: Whisper venv not found"))
    sheet, prompt, info = MR.pack(m, tmp_path, run_dsp=False)
    assert info["lyrics"] == [] and "SUNG LYRICS" not in prompt and any("Whisper" in n for n in info["notes"])


# ---------------------------------------------------------------------------------------- VIDEO MODE: video_duration / video_cuts / beat_grid (code, never the model)
def test_lyrics_from_asr_keeps_the_sung_start_of_every_word():
    lines, _ = MR.lyrics_from_asr(asr_json())
    assert lines[0]["ts"] == [1.0, 1.5, 2.0] and len(lines[1]["ts"]) == len(lines[1]["text"].split()) == 3          # 'go' dropped: ts follow the kept words


def test_parse_video_cuts_cleaned_inside_the_window():
    inf = info_for(24.0)
    inf["items"][0]["cuts"] = [0.3, 1.333, 2.033, 4.033, 4.2, 4.7, 8.0, 10.033, 10.7, 16.0, 17.333, 18.7, 21.4, 22.067]          # the test_clip clip's raw ffmpeg hits
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[0, 12])])), inf, media(text="", length=12))
    assert r["video_duration"] == 24.0 and r["media"][0]["window"] == [0.0, 12.0]
    cuts = r["video_cuts"]
    assert cuts[0]["start"] == 0.0 and cuts[-1]["end"] == 12.0 and all(abs(a["end"] - b["start"]) < 1e-9 for a, b in zip(cuts, cuts[1:]))
    assert all(c["end"] - c["start"] >= 0.2 - 1e-2 for c in cuts) and len(cuts) == 9               # v18: every real cut stays (no 1 s rule, no cap); only pieces under 6 frames merge (4.2 s: 5 frames after 4.033)
    assert [c["start"] for c in cuts] == [0.0, 0.3, 1.33, 2.03, 4.03, 4.7, 8.0, 10.03, 10.7]
    from tools.fusion import story as ST
    assert all(c["end"] - c["start"] <= ST.CLIP_MAX_S + 1e-6 for c in cuts)                       # pieces > CLIP_MAX_S split
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[12, 24])])), inf, media(text="", length=12))
    assert r["video_cuts"][0]["start"] == 12.0 and r["video_cuts"][-1]["end"] == 24.0              # source seconds
    inf["items"][0]["cuts"] = []                                                                    # no cuts: 12 s, one piece split evenly (<= 5.04 s)
    r = MR.parse("{}", inf, media(text="", length=12))
    assert len(r["video_cuts"]) == 3 and r["video_cuts"][1] == dict(start=4.0, end=8.0)


def test_parse_no_video_has_no_cuts_and_keys_are_always_there():
    inf = info_for(24.0)
    inf["items"] = [i for i in inf["items"] if i["kind"] != "video"]
    r = MR.parse("{}", inf, json.dumps(dict(video=None, images=["motngrfx_a.png"], audio=None, text="", length=12)))
    assert r["video_duration"] is None and r["video_cuts"] == [] and r["beat_grid"] is None
    assert set(MR.KEYS) <= set(r)


def test_parse_beat_grid_is_window_relative_and_only_for_the_videos_audio():
    inf = info_for(24.0)
    inf.update(audio_from="V1", music_detail=dict(bpm=120.0, key="C", beats=[round(0.5 * k, 2) for k in range(1, 40)], downbeats_all=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
               music=dict(bpm=120.0, key="C"))
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[4, 16])])), inf, media(text="", length=12))
    g = r["beat_grid"]
    assert g["bpm"] == 120.0 and g["t0"] == 4.0 and g["beats"][0] == 0.0 and g["beats"][-1] == 12.0 and g["downbeats"] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    inf["audio_from"] = "A1"                      # v19: "my track" over a video: the grid is the TRACK's first window-length seconds, not shifted by the window start
    g = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[4, 16])])), inf, media(text="", length=12))["beat_grid"]
    assert g["t0"] == 0.0 and g["beats"][0] == 0.5 and g["beats"][-1] == 12.0 and g["downbeats"] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    inf["audio_from"] = None                      # new music / nothing read: no grid
    assert MR.parse("{}", inf, media(text="", length=12))["beat_grid"] is None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pack_detects_the_real_cut_of_a_clip(tmp_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=12:duration=3", "-f", "lavfi", "-i", "smptebars=size=270x480:rate=12:duration=3",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(tmp_path / "cut.mp4")], check=True)
    m = json.dumps(dict(video="cut.mp4", images=[], audio=None, text="", length=6, format="9:16"))
    sheet, prompt, info = MR.pack(m, tmp_path, run_dsp=False, run_asr=False)
    cuts = info["items"][0]["cuts"]
    assert len(cuts) == 1 and abs(cuts[0] - 3.0) < 0.2
    r = MR.parse("{}", info, m)
    assert r["video_duration"] and abs(r["video_duration"] - 6.0) < 0.3 and len(r["video_cuts"]) == 2 and abs(r["video_cuts"][0]["end"] - 3.0) < 0.2
    json.dumps(r)


# ---------------------------------------------------------------------------------------- v19 AUDIO TRACK + SOUNDTRACK
def test_norm_soundtrack_maps_keep_to_video():
    assert [MR.norm_soundtrack(x) for x in ("keep", "video", "TRACK", "generate", "", None, "x")] == ["video", "video", "track", "generate", "", "", ""]
    assert MR.parse_media(json.dumps(dict(video="a.mp4", soundtrack="keep")))["soundtrack"] == "video"
    assert MR.parse_media("{}")["soundtrack"] == ""


@pytest.mark.parametrize("st,has_v,v_audio,has_a,want", [
    ("video", True, True, True, "V1"), ("video", True, True, False, "V1"), ("video", True, False, True, None),          # a silent clip: nothing read (the audio file is not the pick)
    ("track", True, True, True, "A1"), ("track", True, False, True, "A1"), ("track", True, True, False, "V1"), ("track", False, False, True, "A1"),
    ("track", True, False, False, None), ("generate", True, True, True, None), ("generate", False, False, True, None),
    ("", True, True, True, "V1"), ("", True, False, True, "A1"), ("", False, False, True, "A1"), ("", False, False, False, None),
    ("video", False, False, True, "A1")])                                                                               # images + audio: "video" has no meaning, the file is the only audio
def test_choose_audio_table(st, has_v, v_audio, has_a, want):
    got, notes = MR.choose_audio(st, has_v, v_audio, has_a)
    assert got == want
    if st == "track" and got == "A1":
        assert any("audio file" in n for n in notes)
    if st == "video" and got == "V1" and has_a:
        assert any("audio file is not read" in n for n in notes)               # says which audio was read
    if st == "generate" and (v_audio or has_a):
        assert any("generated" in n for n in notes)


def test_parse_track_lyrics_are_put_on_the_videos_timeline():
    inf = info_for(24.0)
    ly = [dict(t=1.0, text="hold on tight", ts=[1.0, 1.4, 1.8]), dict(t=11.0, text="inside the window"), dict(t=15.0, text="past the window")]
    inf.update(lyrics=ly, audio_from="A1", soundtrack="track")
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[2, 14])])), inf, media(text="", length=12))
    # window 2..14 = 12 s of video; the track plays from its start: track seconds 0..12 are heard, shifted by the window start (2 s) onto the video's timeline
    assert [(l["t"], l["text"]) for l in r["lyrics"]] == [(3.0, "hold on tight"), (13.0, "inside the window")]
    assert r["lyrics"][0]["ts"] == [3.0, 3.4, 3.8] and r["soundtrack"] == "track"
    inf.update(audio_from="V1", soundtrack="video")                          # the video's own audio: seconds in the SOURCE, window filter as before
    r = MR.parse(json.dumps(dict(media=[dict(id="V1", window=[2, 14])])), inf, media(text="", length=12))
    assert [l["t"] for l in r["lyrics"]] == [11.0] and r["soundtrack"] == "video"
    inf.update(audio_from=None, soundtrack="generate")
    r = MR.parse("{}", dict(inf, lyrics=[]), media(text="", length=12))
    assert r["lyrics"] == [] and r["beat_grid"] is None and r["soundtrack"] == "generate"


def test_parse_media_entry_carries_has_audio():
    inf = info_for(24.0)
    inf["items"][0]["has_audio"] = False
    assert MR.parse("{}", inf, media())["media"][0]["has_audio"] is False
    assert "has_audio" not in MR.parse("{}", info_for(24.0), media())["media"][0]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pack_reads_the_audio_the_soundtrack_picks(tmp_path, monkeypatch):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=12:duration=6", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                    "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "clip.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110:duration=8", str(tmp_path / "song.wav")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=12:duration=3", "-pix_fmt", "yuv420p", str(tmp_path / "silent.mp4")], check=True)
    seen = []

    def fake(wav, timeout=600.0):
        seen.append(Path(wav).name)
        return ([dict(t=1.0, text="from " + Path(wav).name)], None)
    monkeypatch.setattr(MR, "transcribe", fake)

    def pack(video, st, audio="song.wav"):
        seen.clear()
        m = json.dumps(dict(video=video, images=[], audio=audio, text="", length=4, format="9:16", **({"soundtrack": st} if st else {})))
        return MR.pack(m, tmp_path, run_dsp=False)[2], list(seen)

    info, got = pack("clip.mp4", "video")
    assert got == ["v1_16k.wav"] and info["audio_from"] == "V1" and info["lyrics"][0]["text"] == "from v1_16k.wav" and any("audio file is not read" in n for n in info["notes"])
    info, got = pack("clip.mp4", "track")
    assert got == ["a1_16k.wav"] and info["audio_from"] == "A1" and info["lyrics"][0]["text"] == "from a1_16k.wav" and info["soundtrack"] == "track"
    assert any("my track" in n and "video's own sound is not read" in n for n in info["notes"])
    info, got = pack("clip.mp4", "generate")
    assert got == [] and info["audio_from"] is None and info["lyrics"] == [] and info["music"] is None and any("generated" in n for n in info["notes"])
    info, got = pack("clip.mp4", "")                                          # not given: the old rule (video audio wins)
    assert got == ["v1_16k.wav"] and info["audio_from"] == "V1"
    info, got = pack("silent.mp4", "video")                                   # a clip with no audio stream: nothing to read, said in the notes
    assert got == [] and info["audio_from"] is None and any("no audio stream" in n for n in info["notes"])
    assert info["items"][0]["has_audio"] is False
    r = MR.parse("{}", info, json.dumps(dict(video="silent.mp4", audio="song.wav", length=4, soundtrack="video")))
    assert r["media"][0]["has_audio"] is False
    info, got = pack(None, "track")                                           # images mode: only the audio file
    assert got == ["a1_16k.wav"]
    info, got = pack(None, "generate")
    assert got == [] and info["audio_from"] is None
