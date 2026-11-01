"""Own-video route (tools/app/own_video_route.py) + the pure helpers of the job runner (tools/app/own_video_job.py). The subprocess is mocked; no ComfyUI, no GPU."""
import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import own_video_job as J      # noqa: E402
from tools.app import own_video_route as R    # noqa: E402


class FakeProc:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


class Spawner:
    def __init__(self):
        self.calls = []
        self.proc = FakeProc(None)

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return self.proc


@pytest.fixture
def mgr(tmp_path):
    out, inp = tmp_path / "output", tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    (inp / "motngrfx_clip.mp4").write_bytes(b"x")
    sp = Spawner()
    m = R.JobManager(out, inp, tmp_path, port=18188, popen=sp, python="py", script="runner.py")
    m.spawner = sp
    return m


BODY = {"video": "motngrfx_clip.mp4", "brief": {"v": 1, "on_screen": ["HOLD ON", "LET GO"], "media": [{"id": "V1", "kind": "video", "window": [2, 14]}]}, "length": 12, "seed": 5,
        "format": "9:16", "quality": "final"}


# ------------------------------------------------------------------------------------------------------------ route logic
def test_start_writes_job_and_spawns_runner(mgr):
    job = mgr.start(dict(BODY, client_id="abc123", soundtrack="generate"))
    assert R.JOB_RE.match(job)
    argv, kw = mgr.spawner.calls[0]
    assert argv[:3] == ["py", "runner.py", "--job"] and job in argv
    i = argv.index
    assert argv[i("--length") + 1] == "12" and argv[i("--seed") + 1] == "5" and argv[i("--quality") + 1] == "final"
    assert argv[i("--soundtrack") + 1] == "generate" and argv[i("--client-id") + 1] == "abc123"
    assert argv[i("--video") + 1].endswith("motngrfx_clip.mp4")
    d = mgr.job_dir(job)
    assert json.loads((d / "brief.json").read_text())["on_screen"] == ["HOLD ON", "LET GO"]
    assert str(d).replace("\\", "/").endswith(f"output/motngrfx_app/own/{job}")
    st = mgr.status(job)
    assert st["stage"] == "queued" and st["done"] is False and st["error"] is None and st["outputs"] == []


def test_brief_null_is_allowed(mgr):
    job = mgr.start({"video": "motngrfx_clip.mp4", "brief": None, "soundtrack": "keep"})
    assert json.loads((mgr.job_dir(job) / "brief.json").read_text()) == {"edit": "overlay"}                  # the edit mode rides in the brief (default overlay)


def test_brief_as_json_string_and_defaults(mgr):
    job = mgr.start({"video": "motngrfx_clip.mp4", "brief": json.dumps({"on_screen": ["A"]})})
    argv = mgr.spawner.calls[0][0]
    assert argv[argv.index("--soundtrack") + 1] == "video" and "--audio" not in argv and argv[argv.index("--quality") + 1] == "final" and argv[argv.index("--length") + 1] == "12"
    assert json.loads((mgr.job_dir(job) / "brief.json").read_text())["on_screen"] == ["A"]


@pytest.mark.parametrize("patch,frag", [
    ({"video": ""}, "video"), ({"video": "../x.mp4"}, "video"), ({"video": "nope.mp4"}, "not in the input"), ({"brief": "{bad"}, "brief"), ({"brief": [1]}, "brief"),
    ({"length": "abc"}, "length"), ({"quality": "ultra"}, "quality"), ({"soundtrack": "remix"}, "soundtrack"), ({"format": "16:9"}, "format"),
])
def test_bad_requests(mgr, patch, frag):
    with pytest.raises(R.BadRequest) as e:
        mgr.start(dict(BODY, **patch))
    assert frag in str(e.value)
    assert not mgr.spawner.calls


def test_length_is_clamped(mgr):
    mgr.start(dict(BODY, length=99))
    argv = mgr.spawner.calls[0][0]
    assert argv[argv.index("--length") + 1] == "24"


def test_one_job_at_a_time_409(mgr):
    first = mgr.start(BODY)
    with pytest.raises(R.Busy) as e:
        mgr.start(BODY)
    assert str(e.value) == first
    mgr.spawner.proc.code = 0                              # the first runner has exited
    state = mgr._read_state(first)
    state.update(done=True, stage="done")
    mgr.write_state(first, state)
    time.sleep(1.1)                                        # job ids carry seconds
    assert mgr.start(BODY) != first


def test_busy_survives_a_server_restart_via_the_state_pid(mgr, tmp_path):
    first = mgr.start(BODY)
    st = mgr._read_state(first)
    st["pid"] = os.getpid()                                # a live pid in job.json, no child handle
    mgr.write_state(first, st)
    mgr.procs.clear()
    assert mgr.running() == first
    st["pid"] = 2 ** 22 + 12345                            # a dead pid
    mgr.write_state(first, st)
    assert mgr.running() is None


def test_status_reports_stage_log_outputs_and_urls(mgr):
    job = mgr.start(BODY)
    st = mgr._read_state(job)
    st.update(stage="matte", stage_n=8, log=[f"line {i}" for i in range(50)], prompt_ids=["p1", "p2"], outputs=[{"kind": "video", "file": "final.mp4"}, {"kind": "audio", "file": "music_00001_.flac"}])
    mgr.write_state(job, st)
    s = mgr.status(job)
    assert s["stage"] == "matte" and s["prompt_ids"] == ["p1", "p2"] and len(s["log"]) == 20 and s["log"][-1] == "line 49"
    assert s["outputs"][0]["kind"] == "video" and s["outputs"][0]["final"] is True and s["outputs"][0]["label"] == "Final video"
    assert s["outputs"][1]["final"] is False and s["outputs"][1]["label"] == "Generated soundtrack"
    assert s["outputs"][0]["url"] == f"/view?filename=final.mp4&subfolder=motngrfx_app%2Fown%2F{job}&type=output"
    assert s["outputs"][1]["kind"] == "audio"


def test_runner_that_died_is_an_error(mgr):
    job = mgr.start(BODY)
    mgr.spawner.proc.code = 3
    s = mgr.status(job)
    assert "exited (code 3)" in s["error"] and s["stage"] == "error"
    assert mgr.running() is None


def test_status_unknown_or_malformed_job(mgr):
    assert mgr.status("nope") is None and mgr.status("../../etc") is None and mgr.status("20260101-000000-abcd") is None


def test_log_falls_back_to_runner_log_tail(mgr):
    job = mgr.start(BODY)
    (mgr.job_dir(job) / "runner.log").write_text("\n".join(f"r{i}" for i in range(40)), encoding="utf-8")
    assert mgr.status(job)["log"][-1] == "r39"


def test_stages_match_between_route_and_runner():
    assert R.STAGES == J.STAGES and "music" in R.STAGES


# ------------------------------------------------------------------------------------------------------------ runner helpers
def test_read_brief_normalises():
    b = J.read_brief({"on_screen": ["  hold   on ", "", "LET GO"], "concept": "a  b", "look": " dusk ", "media": [{"id": "V1", "kind": "video"}, "x"], "music": {"bpm": 90}})
    assert b["on_screen"] == ["hold on", "LET GO"] and b["concept"] == "a b" and b["look"] == "dusk" and len(b["media"]) == 1
    assert J.read_brief("{bad")["on_screen"] == [] and J.read_brief(None)["media"] == [] and J.read_brief("")["concept"] == ""
    assert J.read_brief({"story": {"hits": [{"text": "GO", "land": 1.2}]}})["story"]["hits"][0]["land"] == 1.2 and J.read_brief({})["story"] is None   # the storyboard rides through


@pytest.mark.parametrize("window,clip,length,expect", [
    ([2, 14], 24.0, 12, (2.0, 12.0)),
    ([2, 30], 24.0, 12, (2.0, 12.0)),            # clamped to length
    ([20, 40], 24.0, 12, (20.0, 4.0)),           # clamped to the clip end
    ([5, 6], 24.0, 12, (5.0, 12.0)),              # under 3 s: widened to length
    (None, 24.0, 12, (0.0, 12.0)),               # fallback: the first `length` s
    ("bad", 8.0, 12, (0.0, 8.0)),
    ([-3, 5], 24.0, 12, (0.0, 12.0)),
])
def test_window_of(window, clip, length, expect):
    media = [{"id": "V1", "kind": "video", "window": window}] if window is not None else []
    assert J.window_of(J.read_brief({"media": media}), clip, length) == expect


ANA = {"bpm": 90.0, "beats": [round(0.3 + i * 0.6667, 4) for i in range(40)], "downbeats": [round(0.3 + i * 2.6667, 4) for i in range(10)]}


def test_slots_two_words_in_twelve_seconds():
    s = J.slots_from_analysis(ANA, 2, 12.0)
    assert len(s) == 2 and s[1]["land"] - s[0]["land"] >= 3.2 and s[0]["land"] >= 1.4 and s[1]["land"] <= 10.0
    assert all(any(abs(x["land"] - b) < 1e-6 for b in ANA["beats"]) for x in s)        # landed ON a beat
    assert s[0]["hold_until"] > s[0]["land"]


def test_slots_drop_words_that_do_not_fit():
    s = J.slots_from_analysis(ANA, 9, 12.0)
    assert 2 <= len(s) < 9
    assert all(b["land"] - a["land"] >= 2.4 - 1e-6 for a, b in zip(s, s[1:]))


def test_slots_without_beats_fall_back_to_the_tempo():
    s = J.slots_from_analysis({"bpm": 120.0, "beats": [], "downbeats": []}, 2, 12.0)
    assert len(s) == 2


def test_clip_shorter_than_length_gives_a_short_output_not_a_failure():
    b = J.read_brief({"on_screen": ["HOLD ON", "LET GO"], "media": [{"id": "V1", "kind": "video", "window": [0, 4.7]}]})
    assert J.window_of(b, 4.7, 12) == (0.0, 4.7)                                  # output length = min(length, window, clip)
    assert J.window_of(J.read_brief({}), 4.7, 12) == (0.0, 4.7)                   # no window in the brief: the whole short clip
    assert J.window_of(J.read_brief({"media": [{"id": "V1", "kind": "video", "window": [0, 30]}]}), 4.7, 12) == (0.0, 4.7)
    note = J.shortened_note(4.7, 12)
    assert "4.7 s" in note and "12 s" in note and "not looped or frozen" in note
    assert J.shortened_note(12.0, 12) is None and J.shortened_note(11.9, 12) is None
    s = J.slots_from_analysis(ANA, 2, 4.7)                                        # margins shrink: at least one word still lands, on a beat, inside the clip
    assert 1 <= len(s) and all(0.5 <= x["land"] < 4.7 for x in s)
    s2 = J.slots_from_analysis(ANA, 2, 2.0)
    assert len(s2) >= 1


def test_clean_cuts_drops_tiny_pieces():
    assert J.clean_cuts([1, 61, 66, 81, 180, 241, 261, 358], 360) == [61, 81, 180, 241, 261]
    assert J.clean_cuts([], 360) == [] and J.clean_cuts([0, 3], 360) == []


def test_words_are_upper_case_and_on_the_stretched_clock():
    slots = [{"land": 2.0, "hold_until": 3.4}, {"land": 6.0, "hold_until": 7.4}]
    w = J.words_for(["hold on", "LET GO"], slots, 1.25)
    assert [x["text"] for x in w] == ["HOLD ON", "LET GO"] and w[0]["land"] == 2.5 and w[0]["land_real"] == 2.0 and w[1]["hold_until"] == 9.25


def test_music_request_uses_look_mood_and_concept():
    b = J.read_brief({"on_screen": ["HOLD ON"], "concept": "a quiet farewell", "look": "slow, warm, grainy", "media": [{"id": "V1", "kind": "video", "mood": "wistful"}]})
    r = J.music_request(b, b["on_screen"], 12.2, 3)
    assert r["music"] == "wistful, slow, warm, grainy" and r["scene"] == "a quiet farewell" and r["length"] == 12 and r["words"] == "HOLD ON" and r["seed"] == 3
    r2 = J.music_request(J.read_brief({}), [], 5.0, 1)
    assert r2["length"] == 8 and r2["words"] == "MOTNGRFX" and r2["music"] == ""                # MographRequest.length starts at 8
    assert J.music_request(b, [], 99.0, 1)["length"] == 24


def test_generate_music_graph_links_resolve_and_uses_the_approved_chain():
    b = J.read_brief({"on_screen": ["HOLD ON"], "look": "slow, warm"})
    g = J.music_graph_for(J.music_request(b, b["on_screen"], 12.0, 4), "motngrfx_app/own/J/music", 4)
    for nid, n in g.items():
        for k, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, (nid, k, v)
    assert g["t2m_unet"]["inputs"]["unet_name"].startswith("acestep_v1.5_xl_base") and g["t2m_ks"]["inputs"]["steps"] == 50 and g["t2m_ks"]["inputs"]["seed"] == 4
    assert g["request"]["inputs"]["length"] == 12 and g["request"]["inputs"]["music"] == "slow, warm"
    assert g["out_music"]["inputs"]["filename_prefix"] == "motngrfx_app/own/J/music" and g["out_music"]["inputs"]["audio"] == ["master", 0]


def test_direction_text_survives_braces_and_keeps_the_hits():
    test_clip = "test clip CONTEXT. "
    tmpl = "DIRECTION. " + test_clip + "Fixed hits (seconds): {hits}."
    b = J.read_brief({"concept": "a {curly} idea", "look": "dusk", "on_screen": ["A"]})
    t = J.direction_text(tmpl, test_clip, b, "a street", "A 2.50", 4, "keep")
    assert "a {curly} idea" in t and "A 2.50" in t and "test clip" not in t and "its own soundtrack" in t and "dusk" in t
    assert "generated instrumental soundtrack" in J.direction_text(tmpl, test_clip, b, "", "A 1.00", 3, "generate")


def test_runner_cli_has_the_flags_the_route_passes():
    import argparse
    src = open(os.path.join(ROOT, "tools", "app", "own_video_job.py"), encoding="utf-8").read()
    for flag in ("--job", "--video", "--brief-file", "--length", "--seed", "--format", "--quality", "--soundtrack", "--audio", "--out-dir", "--comfy-dir", "--port", "--client-id"):
        assert f'"{flag}"' in src
    assert argparse is not None


# ------------------------------------------------------------------------------------------------------------ v19 AUDIO TRACK + SOUNDTRACK
def test_route_soundtrack_values_keep_is_video_and_track_needs_the_file(mgr):
    mgr.start(dict(BODY, soundtrack="keep"))
    argv = mgr.spawner.calls[0][0]
    assert argv[argv.index("--soundtrack") + 1] == "video" and "--audio" not in argv


def test_route_track_passes_the_audio_file(mgr, tmp_path):
    (tmp_path / "input" / "motngrfx_song.wav").write_bytes(b"x")
    job = mgr.start(dict(BODY, soundtrack="track", audio="motngrfx_song.wav"))
    argv = mgr.spawner.calls[0][0]
    assert argv[argv.index("--soundtrack") + 1] == "track" and argv[argv.index("--audio") + 1].endswith("motngrfx_song.wav")
    assert json.loads((mgr.job_dir(job) / "request.json").read_text())["soundtrack"] == "track"


@pytest.mark.parametrize("patch,frag", [({"soundtrack": "track"}, "needs the audio file"), ({"soundtrack": "track", "audio": "nope.wav"}, "not in the input"),
                                        ({"soundtrack": "track", "audio": "../x.wav"}, "audio"), ({"soundtrack": "video", "audio": "nope.wav"}, "not in the input")])
def test_route_bad_track_requests(mgr, patch, frag):
    with pytest.raises(R.BadRequest) as e:
        mgr.start(dict(BODY, **patch))
    assert frag in str(e.value) and not mgr.spawner.calls


def test_route_audio_is_not_passed_unless_track(mgr, tmp_path):
    (tmp_path / "input" / "motngrfx_song.wav").write_bytes(b"x")
    mgr.start(dict(BODY, soundtrack="generate", audio="motngrfx_song.wav"))
    assert "--audio" not in mgr.spawner.calls[0][0]


def test_job_norm_and_choose_soundtrack():
    assert [J.norm_soundtrack(x) for x in ("keep", "video", "track", "generate", "", None, "zz")] == ["video", "video", "track", "generate", "video", "video", "video"]
    assert J.choose_soundtrack("video", True, True) == ("video", None)
    assert J.choose_soundtrack("keep", True, False) == ("video", None)
    assert J.choose_soundtrack("track", True, True) == ("track", None)                           # a track wins over the clip's own sound
    assert J.choose_soundtrack("track", False, True) == ("track", None)                          # ... also when the clip is silent
    t, why = J.choose_soundtrack("track", True, False)
    assert t == "video" and "no usable audio file" in why
    t, why = J.choose_soundtrack("track", False, False)
    assert t == "generate" and "falls back to generate" in why
    t, why = J.choose_soundtrack("video", False, True)
    assert t == "generate" and "no audio stream" in why                                           # the clip has no sound: new music (the old keep rule)
    assert J.choose_soundtrack("generate", True, True) == ("generate", None)


def test_track_ffmpeg_commands_and_direction_text():
    c = J.track_wav_cmd("/in/song.wav", "/w/music.wav", 12.0)
    assert c[0] == "ffmpeg" and c[c.index("-i") + 1] == "/in/song.wav" and c[c.index("-t") + 1] == "12.000" and "-ss" not in c and c[-1] == "/w/music.wav"      # from the track's start
    assert J.track_audio_filter(12.0) == "apad=whole_dur=12.000,afade=t=out:st=11.700:d=0.300"
    assert J.track_audio_filter(0.2) == "apad=whole_dur=0.200,afade=t=out:st=0.000:d=0.200"      # a tiny window: the fade never starts before 0
    m = J.track_mux_cmd("/w/raw.mp4", "/w/music.wav", "/o/final.mp4", ["-c:v", "copy"], 12.0)
    assert m[m.index("-i") + 1] == "/w/raw.mp4" and m.count("-i") == 2 and "-c:v" in m and m[m.index("-af") + 1].endswith("d=0.300") and m[m.index("-t") + 1] == "12.0000"
    assert m[m.index("-map") + 1] == "0:v:0" and m[m.index("-c:a") + 1] == "aac" and m[-1] == "/o/final.mp4"
    tmpl = "DIRECTION. test clip CONTEXT. Fixed hits (seconds): {hits}."
    t = J.direction_text(tmpl, "test clip CONTEXT. ", J.read_brief({}), "a street", "A 2.50", 4, "track")
    assert "the user's own audio track as its soundtrack" in t
    assert "its own soundtrack" in J.direction_text(tmpl, "test clip CONTEXT. ", J.read_brief({}), "a street", "A 2.50", 4, "video")


def test_track_commands_really_trim_pad_and_fade_with_ffmpeg(tmp_path):
    import shutil
    import subprocess
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg not installed")

    def dur(p, sel):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "stream=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())

    def rms(p, a, b):
        r = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-ss", str(a), "-t", str(b - a), "-i", str(p), "-vn", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
        return float([x for x in r.stderr.splitlines() if "mean_volume" in x][0].split(":")[1].split()[0])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=12:duration=3", "-pix_fmt", "yuv420p", str(tmp_path / "raw.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=8", str(tmp_path / "long.wav")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=1", str(tmp_path / "short.wav")], check=True)
    for track, name in (("long.wav", "f_long.mp4"), ("short.wav", "f_short.mp4")):                  # a track longer than the window is trimmed, a shorter one is padded
        subprocess.run(J.track_wav_cmd(tmp_path / track, tmp_path / "m.wav", 3.0), check=True)
        subprocess.run(J.track_mux_cmd(tmp_path / "raw.mp4", tmp_path / "m.wav", tmp_path / name, ["-c:v", "copy"], 3.0), check=True)
        assert abs(dur(tmp_path / name, "a:0") - 3.0) < 0.1, track
        assert abs(dur(tmp_path / name, "v:0") - 3.0) < 0.1, track
    assert rms(tmp_path / "f_long.mp4", 0.5, 1.5) > -25 and rms(tmp_path / "f_long.mp4", 2.75, 3.0) < rms(tmp_path / "f_long.mp4", 0.5, 1.5) - 6          # fade-out in the last 0.3 s
    assert rms(tmp_path / "f_short.mp4", 0.2, 0.8) > -25 and rms(tmp_path / "f_short.mp4", 2.0, 2.9) < -60                                           # padded with silence


def test_engine_text_cuts_a_long_storyboard_line_to_a_sensible_phrase_for_the_type_engine():
    assert J.engine_text("NEON RAIN ON THE ROOFTOPS") == "NEON RAIN"                    # 5 words: the engine would split it into fragments ending on ON
    assert J.engine_text("marble clouds over the harbor") == "MARBLE CLOUDS"
    assert J.engine_text("HOLD ON") == "HOLD ON" and J.engine_text("quiet engines hum") == "QUIET ENGINES HUM"      # short lines stay as the user / storyboard wrote them
    for t in ("A B C D E F", "LANTERNS OVER THE HARBOR AND THE SEA", "FIREFLIES ON THE OLD ROOFTOP TONIGHT"):
        out = J.engine_text(t)
        assert out and len(out.split()) <= 4 and out.split()[-1].lower() not in ("on", "the", "and", "of", "in", "to", "with", "a")
