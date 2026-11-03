"""Overlay-only own-video route (tools/app/overlay_mode.py + own_video_job / route / retime args + overlay_check). No ComfyUI, no GPU; ffmpeg only for the tiny synthetic-video tests (skipped without it)."""
import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import overlay_mode as OM          # noqa: E402
from tools.app import own_video_job as J          # noqa: E402
from tools.app import own_video_route as R        # noqa: E402
from tools.fusion import cut_kinds as CK          # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ------------------------------------------------------------------------------------------------------------ the edit choice
def test_edit_defaults_to_overlay_and_redesign_is_explicit():
    assert OM.norm_edit(None) == "overlay" and OM.norm_edit("") == "overlay" and OM.norm_edit("nonsense") == "overlay"
    assert OM.norm_edit("redesign") == "redesign" and OM.norm_edit("With transitions") == "redesign" and OM.norm_edit("with-transitions") == "redesign"
    assert OM.DEFAULT_EDIT == "overlay" and R.norm_edit(None) == "overlay" and R.norm_edit("redesign") == "redesign"


def test_brief_edit_reaches_the_runner_and_defaults_to_overlay(tmp_path):
    assert J.read_brief({})["edit"] == "overlay" and J.read_brief(None)["edit"] == "overlay" and J.read_brief("")["edit"] == "overlay"
    assert J.read_brief({"edit": "redesign"})["edit"] == "redesign" and J.read_brief(json.dumps({"edit": "overlay"}))["edit"] == "overlay"
    assert J.read_brief({"lyrics": [{"text": "a b", "t": 1.0}, {"nope": 1}]})["lyrics"] == [{"text": "a b", "t": 1.0}]


def _mgr(tmp_path):
    from tests.test_own_video_route import Spawner         # noqa: PLC0415
    out, inp = tmp_path / "output", tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    (inp / "clip.mp4").write_bytes(b"x")
    sp = Spawner()
    m = R.JobManager(out, inp, tmp_path, port=18188, popen=sp, python="py", script="runner.py")
    m.spawner = sp
    return m


def test_route_passes_edit_to_the_runner_and_the_brief(tmp_path):
    m = _mgr(tmp_path)
    j = m.start({"video": "clip.mp4", "brief": {"on_screen": ["A"]}})                         # nothing chosen -> overlay
    argv = m.spawner.calls[0][0]
    assert argv[argv.index("--edit") + 1] == "overlay"
    assert json.loads((m.job_dir(j) / "brief.json").read_text())["edit"] == "overlay"
    m.spawner.proc.code = 0
    st = m._read_state(j)
    st.update(done=True, stage="done")
    m.write_state(j, st)
    import time
    time.sleep(1.1)
    j2 = m.start({"video": "clip.mp4", "brief": {"edit": "redesign"}})                         # the brief's own choice
    argv = m.spawner.calls[1][0]
    assert argv[argv.index("--edit") + 1] == "redesign" and json.loads((m.job_dir(j2) / "brief.json").read_text())["edit"] == "redesign"
    m.spawner.proc.code = 0
    st = m._read_state(j2)
    st.update(done=True, stage="done")
    m.write_state(j2, st)
    time.sleep(1.1)
    m.start({"video": "clip.mp4", "brief": {"edit": "redesign"}, "edit": "overlay"})            # the request field wins
    argv = m.spawner.calls[2][0]
    assert argv[argv.index("--edit") + 1] == "overlay"


def test_runner_cli_takes_edit_and_size_mode():
    import argparse
    src = open(os.path.join(ROOT, "tools/app/own_video_job.py"), encoding="utf-8").read()
    assert '"--edit"' in src and '"--size-mode"' in src and 'choices=["native", "tier"]' in src
    assert isinstance(J.main, type(lambda: 0)) and argparse


# ------------------------------------------------------------------------------------------------------------ route args of the retime step: no 3D cut flag, overlay lock on
def test_retime_argv_overlay_locks_the_picture_and_never_asks_for_a_3d_cut():
    nat = OM.native_sizes(1920, 1080, "draft")
    argv = [str(x) for x in J.retime_argv("py", "/r", "/w", "/o", "STEM", 72.0, 3, "overlay", nat, nat["matte"])]
    assert argv[argv.index("--edit") + 1] == "overlay"
    assert "--cut3d" not in argv                                                              # the lock makes every cut hard; no 3D cut is requested anywhere
    assert argv[argv.index("--native-size") + 1] == "1920x1080" and argv[argv.index("--matte-size") + 1] == "1024x576"
    assert abs(float(argv[argv.index("--aspect") + 1]) - 16 / 9) < 1e-4
    assert argv[argv.index("--transitions") + 1] == "keep"                                    # word-to-word TYPE transitions are fine (drawn over the untouched picture)
    assert argv[argv.index("--glow") + 1] == "off" and argv[argv.index("--pure-type") + 1] == "1"
    red = [str(x) for x in J.retime_argv("py", "/r", "/w", "/o", "STEM", 72.0, 3, "redesign", None, (576, 1024))]
    assert red[red.index("--edit") + 1] == "redesign" and "--native-size" not in red


def test_direction_text_forbids_3d_cuts_in_overlay_only():
    d4 = "DESIGN. {ctx}Footage cuts are hard; at most one designed 3D cut. Fixed hits (seconds): {hits}."
    br = J.read_brief({})
    ov = J.direction_text(d4, "zzz", br, "a clip", "A 1.00", 3, "video", "overlay")
    rd = J.direction_text(d4, "zzz", br, "a clip", "A 1.00", 3, "video", "redesign")
    assert "at most one designed 3D cut" in rd and "at most one designed 3D cut" not in ov and "NO 3D cut" in ov


# ------------------------------------------------------------------------------------------------------------ the picture lock on the compose chains
def _chain():
    return {"ops": [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.8, "sat": 0.7}},
        {"op": "halation", "out": "foot2", "args": {"frame": "$foot", "strength": 0.3}},
        {"op": "type_face", "out": "tf", "args": {"beauty": "$type_beauty", "type_a": "$type_matte"}},
        {"op": "sandwich", "out": "img", "args": {"footage": "$foot2", "type_rgb": "$tf", "type_a": "$type_matte", "matte": "$matte", "legible": False}},
        {"op": "knockout", "out": "ko", "args": {"plate_rgb": [0, 0, 0], "type_a": "$type_matte", "footage": "$foot2", "min_contrast": 0.22}},
        {"op": "plate_pan", "out": "pp", "args": {"img": "$img", "dx": 0.1}},
        {"op": "crossfade", "out": "cf", "args": {"a": "$pp", "b": "$ko", "t": "$t"}},
        {"op": "flow_wake", "out": "wk", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$matte", "bg_drag": 0.12}},
        {"op": "hold_net", "out": "net", "args": {"image": "$cf", "plate": "$foot2", "face": "$tf", "alpha": "$wk"}},
        {"op": "grain", "out": "g", "args": {"frame": "$net", "amount": 0.03}},
        {"op": "vignette", "out": "image", "args": {"frame": "$g", "strength": 0.4}},
    ]}


def _piece():
    return {"fps": 24, "format": "9:16", "sizes": {"control": [576, 1024], "shot": [576, 1024], "type": [576, 1024], "compose": [540, 960]},
            "shots": [{"id": "s1", "span": 80, "plate": "clean", "type_matte": "reveal_subject", "flat": "light"}, {"id": "s2", "span": 80, "subject_move": "x", "treatment": "dim"}],
            "layout": {"s1": {"start": 0, "frames": 80, "xf": 0}, "s2": {"start": 80, "frames": 80, "xf": 0}},
            "compose": {"size": [540, 960], "crossfade": 0.5, "holds": [[7.5, 0.46]], "soft_support": True,
                        "cuts": [{"after": "s1", "cut": "orbit:push", "params": {"style": "smooth"}, "style": "slide"}, {"after": "s2", "cut": "window_other"}, {"after": "s3", "cut": "crossfade"}],
                        "default": _chain(), "shots": {"s2": _chain()}},
            "type": {"words": [{"text": "A"}]}}


def _defined_before_use(ops):
    """every $buffer an op reads is a shell buffer or produced by an earlier op."""
    shell = {"footage", "matte", "depth", "flow", "flow_m", "type_matte", "type_line", "type_depth", "type_beauty", "frame_idx", "t", "u", "n", "palette"}
    seen, bad = set(shell), []

    def refs(x):
        if isinstance(x, str) and x.startswith("$"):
            yield x[1:]
        elif isinstance(x, dict):
            for v in x.values():
                yield from refs(v)
        elif isinstance(x, list):
            for v in x:
                yield from refs(v)
    for o in ops:
        bad += [r for r in refs(o.get("args")) if r not in seen]
        seen.add(o.get("out", "image"))
    return bad


def test_lock_leaves_no_3d_cut_and_no_footage_touching_op():
    p = _piece()
    assert OM.picture_violations(p)                                                          # the design as written touches the picture
    nat = OM.native_sizes(1920, 1080, "draft")
    rep = OM.lock_piece(p, nat)
    assert OM.picture_violations(p) == []
    for c in p["compose"]["cuts"]:
        assert c["cut"] == "hard" and "params" not in c and CK.family_of(c["cut"]) == "plain" and c["cut"] not in CK.SHOT_3D
    assert p["compose"]["holds"] == [] and p["compose"]["crossfade"] == 0.0 and not p["compose"]["soft_support"]
    assert {"orbit:push", "window_other", "crossfade"} <= {c[1] for c in rep["cuts_made_hard"]}
    for ch in OM.chains_of(p):
        names = [o["op"] for o in ch["ops"]]
        assert not set(names) & set(OM.PICTURE_OPS), names
        assert "flow_wake" in names and "sandwich" in names                                  # the type machinery stays (flow_wake moves the TYPE by the footage flow, never the footage)
        assert all(o["args"].get("min_contrast", 0) == 0 for o in ch["ops"] if o["op"] == "knockout")
        assert sum(1 for o in ch["ops"] if o.get("out", "image") == "image") == 1                # the chain still defines its result `image` (renamed onto the op that made the picture)
        assert _defined_before_use(ch["ops"]) == [], _defined_before_use(ch["ops"])
        sw = next(o for o in ch["ops"] if o["op"] == "sandwich")
        assert sw["args"]["footage"] == "$footage"                                            # the plate grade was bypassed: the sandwich reads the footage itself
    assert all(k not in s for s in p["shots"] for k in OM.SHOT_KEYS)
    assert p["sizes"]["compose"] == [1920, 1080] and p["sizes"]["type"] == list(nat["type"]) == [2048, 1152] and p["sizes"]["control"] == [1024, 576] and p["compose"]["size"] == [1920, 1080] and p["format"] == "16:9"


def test_lock_with_a_tail_that_reads_the_footage_directly_keeps_the_chain_valid():
    p = {"compose": {"cuts": [], "default": {"ops": [{"op": "grade", "out": "image", "args": {"frame": "$footage", "gain": 0.9}}]}}, "shots": []}
    OM.lock_piece(p)
    ops = p["compose"]["default"]["ops"]
    assert len(ops) == 1 and ops[0]["out"] == "image" and ops[0]["args"]["frame"] == "$footage"
    assert ops[0]["args"]["gain"] == 1.0 and ops[0]["args"]["gamma"] == 1.0 and ops[0]["args"]["sat"] == 1.0 and ops[0]["args"]["lift"] == 0.0     # an identity carries the footage into `image`
    q = copy.deepcopy(_piece())
    OM.lock_piece(q)
    assert OM.lock_piece(q)["ops_removed"] == []                                              # idempotent


def test_violations_name_every_kind_of_picture_change():
    p = _piece()
    v = " | ".join(OM.picture_violations(p))
    for frag in ("cut after s1: orbit:push", "compose.holds", "compose.crossfade", "op grade", "op grain", "op vignette", "op halation", "op plate_pan", "op crossfade", "op hold_net", "op knockout", "shot s1 key plate"):
        assert frag in v, frag
    p["layout"]["s2"]["xf"] = 12
    assert any("xf" in x for x in OM.picture_violations(p))


# ------------------------------------------------------------------------------------------------------------ source size
@pytest.mark.parametrize("W,H", [(1920, 1080), (1080, 1920), (3840, 2160), (720, 1280), (1000, 1000), (853, 480), (1279, 719)])
def test_native_sizes_keep_the_source_size(W, H):
    n = OM.native_sizes(W, H, "draft")
    assert n["compose"] == (OM.even(W), OM.even(H)) == n["clip"] == n["shot"]                                # the picture is never scaled
    assert n["type"][0] % 64 == 0 and n["type"][1] % 64 == 0 and n["type"][0] >= n["compose"][0] and n["type"][1] >= n["compose"][1] and n["piece"]["type"] == list(n["type"])    # the type canvas is a multiple of 64
    assert abs(n["compose"][0] - W) <= 1 and abs(n["compose"][1] - H) <= 1
    mw, mh = n["matte"]
    assert mw % 32 == 0 and mh % 32 == 0 and max(mw, mh) <= 1024 + 32 and n["piece"]["control"] == [mw, mh] and n["piece"]["compose"] == list(n["compose"])
    assert abs(mw / mh - W / H) < 0.12 or min(mw, mh) == 128                                          # the matte keeps the aspect (to a 32 px step) and is upscaled back
    assert n["format"] == ("16:9" if W > H else "9:16") and abs(n["aspect"] - OM.even(W) / OM.even(H)) < 1e-9


def test_matte_tier_follows_quality():
    assert max(OM.native_sizes(1920, 1080, "preview")["matte"]) == 704 and max(OM.native_sizes(1920, 1080, "draft")["matte"]) == 1024


def test_native_plan_keeps_fps_and_frames():
    from tools.app import ingest as IG
    for r, f in (("30/1", 30.0), ("30000/1001", 29.97003), ("24/1", 24.0), ("60/1", 60.0), ("25/1", 25.0)):
        pr = dict(display_w=1281, display_h=720, r_frame_rate=r, avg_frame_rate=r, vfr=False, audio={"codec": "aac"})
        nat = J.native_plan(pr, 720, [0.5, 1.2], IG)
        assert nat["fps"]["policy"] == "keep" and nat["fps"]["step"] == 1 and abs(nat["fps"]["fps_float"] - f) < 1e-3     # 60 fps is NOT halved (ingest's tier policy would)
        assert nat["frames"] == 720 and nat["size"] == [1280, 720]
        assert [s["f0"] for s in nat["shots"]][0] == 0 and all(0 <= s["f0"] < 720 for s in nat["shots"])
        assert nat["audio"] == {"copy": True}


# ------------------------------------------------------------------------------------------------------------ time on the source frame grid
@pytest.mark.parametrize("fps", [24.0, 25.0, 29.97003, 30.0, 50.0, 59.94006, 60.0, 23.976])
def test_word_frames_are_integers_on_the_source_fps(fps):
    n = int(fps * 12)
    slots = [dict(land=1.31, hold_until=3.3), dict(land=5.3109, hold_until=7.3), dict(land=9.3099, hold_until=10.9), dict(land=11.9, hold_until=12.6)]
    q = OM.quantise_slots(slots, fps, n)
    K = fps / 24.0
    for s, o in zip(slots, q):
        assert isinstance(o["land_frame"], int) and 0 <= o["land_frame"] < n and o["hold_frame"] > o["land_frame"] and o["hold_frame"] <= n
        assert abs(o["land"] * fps - o["land_frame"]) < 1e-4                                       # t = frame / src_fps
        assert abs(o["land"] - s["land"]) <= 0.5 / fps + 1e-6                                       # moved by at most half a source frame
        assert abs(OM.quantise_time(s["land"], fps) - o["land"]) < 1e-5
        w = J.words_for(["X"], [o], K)[0]
        assert abs(w["land"] * 24.0 - o["land_frame"]) < 0.01                                      # on the stretched pipeline clock the word sits on pipeline frame == source frame
        assert w["land_frame"] == o["land_frame"] and abs(OM.pipeline_frame(o["land"], fps) - o["land_frame"]) < 1e-3


@pytest.mark.parametrize("fps,n", [(24.0, 576), (30.0, 720), (29.97003, 719), (59.94006, 1439), (25.0, 300)])
def test_frame_count_is_preserved_on_the_stretched_clock(fps, n):
    from tools.fusion import plan_rules as PR
    K = fps / 24.0
    music = dict(cut=n / 24.0, nframes=n, bpm=90.0 / K, bar_len=2.667 * K, key="", beats=[round(i * 0.667 * K, 4) for i in range(60)], downbeats=[round(i * 2.667 * K, 4) for i in range(15)], chord_changes=[], curves={}, fps=24)
    spans = PR.cut_spans(n, PR.make_grid(music, music["cut"]), 24)
    assert OM.frame_count_preserved(n, spans) and sum(spans) == n                                    # the planned shots cover exactly the source frames
    assert abs(n / 24.0 - (n / fps) * K) < 1e-9                                                       # the relabel back to the source fps maps pipeline frame i to source frame i (1:1, no drop / repeat)


# ------------------------------------------------------------------------------------------------------------ the lyric route
def test_sung_hits_land_on_the_sung_second_snapped_to_a_beat_else_the_beat_slot():
    lyr = [{"text": "FIRST WORD on the platform tonight", "t": 13.1}, {"text": "bass hit hard", "t": 17.0}]
    asr = OM.asr_from_lyrics(lyr)
    beats = [round(0.6667 * i, 4) for i in range(60)]
    slots = [dict(land=2.0, hold_until=4.0), dict(land=6.0, hold_until=8.0), dict(land=10.5, hold_until=11.5)]
    hits, notes = OM.sung_hits(["FIRST WORD", "BASS HIT", "NOT IN SONG"], slots, asr, beats, 12.0, w0=10.0)       # the window starts at 10 s of the source
    assert [h["text"] for h in hits] == ["FIRST WORD", "BASS HIT", "NOT IN SONG"]
    assert 3.1 <= hits[0]["land"] <= 3.1 + 0.6667 and any(abs(b - hits[0]["land"]) < 1e-3 for b in beats)                           # sung at 13.1 - 10 = 3.1, snapped FORWARD to a beat, never before it is sung
    assert 7.0 <= hits[1]["land"] <= 7.0 + 0.6667                                                       # on the nearest beat inside story.SNAP_S, else the sung second itself (never before it)
    assert hits[2]["land"] == 10.5                                                                    # not found: keeps its beat slot
    early, _ = OM.sung_hits(["FIRST WORD"], slots[:1], OM.asr_from_lyrics([{"text": "FIRST WORD", "t": 10.2}]), beats, 12.0, w0=10.0)
    assert early[0]["land"] >= 1.4 - 1e-9                                                              # story._land_for: not before the first word can form (1.4 s)


# ------------------------------------------------------------------------------------------------------------ restore the picture (tiny synthetic frames)
def test_restore_frame_returns_the_source_outside_the_type():
    import numpy as np
    rng = np.random.default_rng(1)
    S = rng.integers(0, 255, (96, 64, 3), dtype=np.uint8)
    R = np.clip(S.astype(np.int16) + rng.integers(-2, 3, S.shape), 0, 255).astype(np.uint8)         # the engine's own footage: codec-noisy copy
    E = R.copy()
    E[30:50, 20:44] = (255, 255, 255)                                                                  # the type
    out = OM.restore_frame(S, E, R)
    assert (out[34:46, 24:40] >= 250).all()                                                                  # the type is there (S + what the engine added: within the footage noise of white)
    keep = np.ones(S.shape[:2], bool)
    keep[22:58, 12:52] = False                                                                         # the type box + its grown margin
    assert (out[keep] == S[keep]).all()                                                                # everywhere else the output IS the source frame, bit for bit
    assert (OM.restore_frame(S, R, R) == S).all()                                                      # no type: the whole frame is the source


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_restore_video_and_overlay_check_on_a_synthetic_clip(tmp_path):
    import numpy as np
    from tools.app import overlay_check as OC
    W, H, N = 160, 96, 12

    def write(path, frames, fps="30"):
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        assert p.wait() == 0
    rng = np.random.default_rng(3)
    base = [np.full((H, W, 3), 40 + 5 * i, np.uint8) for i in range(N)]
    for i, f in enumerate(base):
        f[:, :, 1] = np.linspace(60, 160, W).astype(np.uint8)[None, :] + i                            # a smooth gradient picture (no hard chroma edge: yuv420 round trips are not exact on those)
    cut_at = 6
    for i in range(cut_at, N):
        base[i] = np.clip(255 - base[i].astype(np.int16), 0, 255).astype(np.uint8)                    # a hard cut at frame 6
    src, raw, pic = tmp_path / "src.mp4", tmp_path / "raw.mp4", tmp_path / "picture.mp4"
    write(src, base)
    typed = []
    for i, f in enumerate(base):
        g = f.copy()
        if 3 <= i <= 8:
            g[30:60, 40:120] = (0, 220, 255)                                                          # our type
        typed.append(g)
    write(raw, typed, "24")                                                                           # the render is on the 24 fps pipeline clock
    clips = []
    for k, (a, b) in enumerate([(0, 6), (6, 12)], 1):
        c = tmp_path / f"s{k}_clip.mp4"
        write(c, base[a:b] + [base[b - 1]] * 3, "24")                                                 # shot clips carry span + pad frames
        clips.append(c)
    rep = OM.restore_video(src, raw, clips, [6, 6], pic, "30")
    assert rep["frames"] == N and rep["source_size"] == [W, H]
    ps, pp = OM.probe_video(src), OM.probe_video(pic)
    assert (pp["width"], pp["height"], pp["frames"], pp["fps_str"]) == (ps["width"], ps["height"], ps["frames"], ps["fps_str"])
    wf = {"word_frames": [dict(text="X", start_frame=3, end_frame=8)]}
    res = OC.compare(src, pic, 0.0, N, wf)
    assert res["ok"], res
    assert res["checks"]["size"]["ok"] and res["checks"]["fps"]["ok"] and res["checks"]["frames"]["ok"] and res["checks"]["cuts"]["source"] == res["checks"]["cuts"]["final"] == [cut_at]
    assert res["checks"]["type_share"]["max_changed_share"] > 0.05                                    # the type is there and the checker saw it
    # a picture that changed (a 6 level global grade) must FAIL; so must another size / frame count
    bad = [np.clip(f.astype(np.int16) + 14, 0, 255).astype(np.uint8) for f in base]
    write(tmp_path / "bad.mp4", bad)
    assert not OC.compare(src, tmp_path / "bad.mp4", 0.0, N, wf)["ok"]
    write(tmp_path / "short.mp4", base[:-2])
    r2 = OC.compare(src, tmp_path / "short.mp4", 0.0, N, wf)
    assert not r2["ok"] and not r2["checks"]["frames"]["ok"]
    assert OC.main([str(src), str(pic), "--frames", str(N)]) == 1 or OC.main([str(src), str(pic), "--frames", str(N), "--max-changed", "0.6"]) == 0
    assert OC.main([str(src), str(tmp_path / "bad.mp4"), "--frames", str(N)]) == 1


def test_overlay_check_cli_has_the_documented_flags():
    from tools.app import overlay_check as OC
    src = open(OC.__file__, encoding="utf-8").read()
    for flag in ("--start", "--frames", "--report", "--mask-video", "--mad", "--psnr", "--max-changed", "--json"):
        assert flag in src


# ------------------------------------------------------------------------------------------------------------ the web page
def test_web_app_edit_choice_defaults_to_overlay_and_rides_in_the_request():
    app = open(os.path.join(ROOT, "web_app/app.js"), encoding="utf-8").read()
    assert 'edit: "overlay"' in app                                                          # media state default
    assert "B.EDIT_OPTIONS" in app and "const edit = B.normEdit(S.media.edit);" in app
    assert "const body = { edit, video: video.uploaded, brief: brief ? Object.assign(JSON.parse(brief), { edit }) : { edit }," in app      # both the request and the brief carry it
    bind = open(os.path.join(ROOT, "web_app/bind.js"), encoding="utf-8").read()
    assert "Overlay only" in bind and "With transitions" in bind
    node = shutil.which("node")
    if node:
        out = subprocess.run([node, "-e", "const B=require('./web_app/bind.js');console.log(JSON.stringify([B.normEdit(undefined),B.normEdit('With transitions'),B.normEdit('redesign'),B.normEdit('x'),B.EDIT_OPTIONS.map(o=>o.val)]))"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
        assert json.loads(out) == ["overlay", "redesign", "redesign", "overlay", ["overlay", "redesign"]]


# ------------------------------------------------------------------------------------------------------------ box run R16: a cream ground painted over the picture (letters-as-window chain)
def _window_piece():
    p = _piece()
    p["type"] = {"words": [{"text": "A"}, {"text": "RIDE FAST", "window_source": "s3", "window_ground": [0.94, 0.92, 0.87]}]}
    p["compose"]["shots"] = {"s3": {"ops": [
        {"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m"}},
        {"op": "window_cut", "out": "image", "args": {"a": [0.9373, 0.9216, 0.8745], "b": "$footage", "mode": "other", "type_a": "$wm", "under": "$footage", "t": "$t"}}]}}
    return p


def test_lock_drops_the_letters_as_window_ground_chain_and_word_keys():
    p = _window_piece()
    v = " | ".join(OM.picture_violations(p))
    assert "op window_cut" in v and "word 1 key window_source" in v and "word 1 key window_ground" in v
    rep = OM.lock_piece(p)
    assert OM.picture_violations(p) == []
    assert "s3" not in p["compose"]["shots"] and rep["chains_dropped"] == [["s3", ["window_cut"]]]            # s3 now uses the default chain: type over the footage
    assert not any(k in w for w in p["type"]["words"] for k in OM.WORD_KEYS) and len(rep["word_keys_dropped"]) == 2
    assert "default" in p["compose"] and p["compose"]["default"]["ops"]


def test_a_default_chain_with_a_ground_op_is_a_violation():
    p = _piece()
    p["compose"]["default"]["ops"].insert(0, {"op": "duotone", "out": "d", "args": {"frame": "$footage"}})
    assert any("duotone" in x for x in OM.picture_violations(OM.lock_piece(p) and p))


def test_restore_never_lets_a_ground_through():
    import numpy as np
    rng = np.random.default_rng(5)
    S = rng.integers(20, 200, (80, 48, 3), dtype=np.uint8)
    R = S.copy()
    cream = np.full_like(S, (222, 235, 240))                                                          # the flat cream ground over the whole frame
    out, info = OM.restore_frame_ex(S, cream, R)
    assert (out == S).all() and info["state"] == "dropped" and info["share"] > OM.MAX_TYPE_SHARE
    wash = np.clip(R.astype(np.int16) + 35, 0, 255).astype(np.uint8)                                  # a weak full-frame wash with a strong small mark on it
    wash[10:22, 8:30] = (255, 255, 255)
    out, info = OM.restore_frame_ex(S, wash, R)
    assert info["state"] == "strict" and (out[12:20, 10:28] >= 245).all()                              # the strong mark is type and stays
    keep = np.ones(S.shape[:2], bool)
    keep[4:30, 2:36] = False
    assert (out[keep] == S[keep]).all()                                                                # the wash is gone: the source frame everywhere else
    text = R.copy()
    text[30:50, 10:40] = (0, 220, 255)
    out, info = OM.restore_frame_ex(S, text, R)
    assert info["state"] == "ok" and info["share"] < 0.45 and (out[34:46, 14:36] == (0, 220, 255)).all()   # normal type is untouched by the guard
    out, info = OM.restore_frame_ex(S, text, R, active=False)
    assert (out == S).all() and info["state"] == "inactive"                                           # no word can be on screen: the source frame


def test_native_cut_threshold_catches_the_soft_cuts_of_a_generated_clip():
    assert 0.1 <= J.NATIVE_CUT_THR < 0.25          # R16: the cuts of a 3-shot LTX clip scored 0.31 / 0.25, everything else <= 0.055; ingest's 0.35 found none


def test_thin_type_lines_survive_the_restore_and_noise_specks_do_not():
    import numpy as np
    rng = np.random.default_rng(9)
    S = rng.integers(40, 200, (96, 64, 3), dtype=np.uint8)
    R = S.copy()
    E = R.copy()
    E[48, 6:58] = (255, 255, 255)                                                                      # a 1 px strand (a median filter erased these: box run R16 frames 98 / 152)
    E[20, 10] = (255, 255, 255)                                                                        # a single noisy pixel is not type
    out = OM.restore_frame(S, E, R)
    assert (out[48, 10:54] >= 200).all()
    assert (out[20, 10] == S[20, 10]).all()


def test_restore_keeps_the_source_colour_description(tmp_path):
    assert OM.color_args(dict(color_space="bt709", color_primaries="bt709", color_transfer="iec61966-2-1", color_range="tv"))[0].endswith("out_color_matrix=bt709,format=yuv420p")
    flt, flags = OM.color_args(dict(color_space="bt709", color_primaries="bt709", color_transfer="iec61966-2-1", color_range="tv"))
    assert flags == ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv"]
    flt, flags = OM.color_args(dict(color_space="unknown", color_primaries=None, color_transfer="unknown", color_range=None))
    assert "out_color_matrix=bt601" in flt and flags == []                                               # untagged source: decoded as bt601, written as bt601, no tags invented


def test_word_windows_merge_across_the_transition_gaps():
    from tools.app import overlay_check as OC
    rep = {"word_frames": [dict(start_frame=2, end_frame=94), dict(start_frame=100, end_frame=170), dict(start_frame=171, end_frame=296), dict(start_frame=330, end_frame=400)]}
    assert OC.word_windows(rep, 4) == [(0, 404)]                       # R16: frame 98 was a one-frame gap between two spans and was compared whole-frame
    assert OC.word_windows(rep, 0, merge_gap=3) == [(2, 94), (100, 170), (171, 296), (330, 400)][:1] + [(100, 296), (330, 400)]


# ------------------------------------------------------------------------------------------------------------ type behind the subject (R16 review e5 / e6)
def _pure_chain():
    """the chain the retime step's pure-type pass writes: the type over the footage, then the footage put back over it by the subject matte."""
    return {"ops": [
        {"op": "type_face", "out": "_tf4", "args": {"beauty": "$type_beauty", "type_a": "$type_matte"}},
        {"op": "rgba_part", "out": "tmat", "args": {"rgba": "$_tf4", "part": "a"}},
        {"op": "rgba_part", "out": "tface", "args": {"rgba": "$_tf4", "part": "rgb"}},
        {"op": "hold_life", "out": "hl0_front", "args": {"img": "$tmat", "ref": "$tmat"}},
        {"op": "upscale_matte", "out": "m_dep", "args": {"guide": "$footage", "matte": "$matte", "size": [576, 1024]}},
        {"op": "over", "out": "image_t", "args": {"base": "$footage", "fg": "$tface", "alpha": "$hl0_front"}},
        {"op": "over", "out": "image", "args": {"base": "$image_t", "fg": "$footage", "alpha": "$m_dep"}},
    ]}


def test_choose_behind_takes_about_half_and_at_least_one():
    import numpy as np
    box = (0.1, 0.4, 0.9, 0.6)
    on_subject = np.zeros((100, 60), np.float32)
    on_subject[40:60, 5:15] = 1.0                                         # the subject covers the left part of the word's box
    empty = np.zeros((100, 60), np.float32)
    sc = [OM.word_cover(box, [on_subject] * 6), OM.word_cover(box, [empty] * 6), OM.word_cover(box, [None]), OM.word_cover(box, [empty] * 3)]
    assert sc[0]["overlap"] == 1.0 and 0.1 < sc[0]["cover"] < 0.35 and sc[1]["overlap"] == 0.0 and sc[2]["frames"] == 0
    got = OM.choose_behind(sc)
    assert got[0] == 0 and len(got) == 2 and 2 not in got              # 3 words on screen -> 2 behind, the one with the subject in its box first; a word with no frames is never chosen
    assert OM.choose_behind([sc[0]]) == [0]                            # one word: at least one behind
    assert len(OM.choose_behind([sc[1]] * 4)) == 2                     # even with the subject nowhere near a box: half of them
    assert OM.choose_behind([dict(frames=0)] * 3) == []                # no matte frames at all: nothing to choose


def test_apply_behind_splits_front_and_behind_holds_and_keeps_the_picture_lock():
    piece = {"fps": 24, "compose": {"holds": [], "crossfade": 0.0, "soft_support": False, "cuts": [], "default": _pure_chain()}, "type": {"words": [{"text": "A"}] * 4}}
    holds = [(1.0, 2.5), (4.0, 5.5), (7.0, 8.5), (10.0, 11.5)]
    rep = OM.apply_behind(piece, holds, [1, 3])
    assert rep["chains"] == 1 and rep["behind"] == [1, 3] and rep["front"] == [0, 2]
    ops = piece["compose"]["default"]["ops"]
    lf, lb = [o for o in ops if o["op"] == "legible"]
    from tools.fusion import compose
    assert lf["args"]["holds"] == [[1.0, 2.5], [7.0, 8.5]] and lf["args"]["floor"] == OM.FRONT_FLOOR and lf["args"]["letters"] is False
    assert lb["args"]["holds"] == [[4.0, 5.5], [10.0, 11.5]] and lb["args"]["floor"] == OM.BEHIND_FLOOR and lb["args"]["matte"] == "$" + lf["out"]
    assert lb["args"]["behind"] is False and lf["args"]["behind"] is False and lb["args"]["floor"] == dict(compose.LEG_FLOOR) and lf["args"]["floor"]["word"] < 0      # 2026-11-02: front = the floor fires on every frame (whole hold in front); behind = the readable rule, the same numbers as compose.LEG_FLOOR
    assert ops[-1]["args"]["alpha"] == "$" + lb["out"] and ops[-1]["args"]["fg"] == "$footage"          # the subject goes back over the type, from the picture
    assert lf["args"]["type_a"] == lb["args"]["type_a"] == "$hl0_front"
    assert not _defined_before_use(ops)
    assert OM.picture_violations(piece) == []                                                              # no scrim / hold_net / soft_support / ground came with it
    assert not any(o["op"] in ("hold_net", "vignette", "grade", "knockout") for o in ops) and piece["compose"]["soft_support"] is False
    n0 = len(ops)
    OM.apply_behind(piece, holds, [1, 3])                                                                  # idempotent: the alpha already comes out of a legible op
    assert len(piece["compose"]["default"]["ops"]) == n0


def test_guard_ops_leaves_the_split_alone():
    from tools.fusion import recipe_eval as RE
    piece = {"fps": 24, "plan": {"flow_wake": {"holds": [[1.0, 2.5], [4.0, 5.5]]}}, "compose": {"holds": [], "default": _pure_chain()}, "type": {"words": [{"text": "A"}, {"text": "B"}]}}
    OM.apply_behind(piece, [(1.0, 2.5), (4.0, 5.5)], [1])
    ops = RE.guard_ops(piece, copy.deepcopy(piece["compose"]["default"]["ops"]))
    leg = [o for o in ops if o["op"] == "legible"]
    assert len(leg) == 2 and leg[0]["args"]["holds"] == [[1.0, 2.5]] and leg[1]["args"]["holds"] == [[4.0, 5.5]]       # the guard added no third, all-words legible op
    assert leg[1]["args"]["floor"] == OM.BEHIND_FLOOR and leg[1]["args"]["letters"] is False


def test_restore_keeps_the_subject_over_a_word_that_is_behind_it():
    import numpy as np
    rng = np.random.default_rng(5)
    S = rng.integers(0, 255, (96, 64, 3), dtype=np.uint8)
    R = np.clip(S.astype(np.int16) + rng.integers(-2, 3, S.shape), 0, 255).astype(np.uint8)
    E = R.copy()
    E[30:50, 10:50] = (255, 255, 255)                                  # the word
    E[30:60, 28:40] = R[30:60, 28:40]                                  # the subject stands in front of its middle: the composed frame has the engine's footage there
    out = OM.restore_frame(S, E, R)
    assert (out[32:48, 14:24] >= 250).all() and (out[32:48, 44:48] >= 250).all()                       # the letters on both sides are there
    assert (out[34:58, 31:37] == S[34:58, 31:37]).all()                                                # the subject's pixels are the source's: no type painted over him


# ------------------------------------------------------------------------------------------------------------ a source that is not a multiple of 64 (540x960 failed: spec.check_size)
@pytest.mark.parametrize("W,H,canvas", [(540, 960, (576, 1024)), (576, 1024, (576, 1024)), (1080, 1920, (1152, 2048)), (1920, 1080, (2048, 1152)), (1000, 1000, (1024, 1024)), (150, 200, (192, 256)), (1080, 1350, (1280, 1600))])
def test_type_canvas_is_a_multiple_of_64_at_the_source_aspect(W, H, canvas):
    assert OM.type_canvas(W, H) == canvas
    cw, ch = canvas
    assert cw % 64 == 0 and ch % 64 == 0 and cw >= W and ch >= H
    assert abs(cw / ch / (W / H) - 1.0) <= OM.TYPE_ALIGN_TOL
    assert OM.visible_rect(canvas, (W, H)) == (0.0, 0.0, float(cw), float(ch))                            # the aspects agree: the whole canvas is the visible frame, nothing lost to a margin


@pytest.mark.parametrize("W,H", [(852, 480), (1278, 718), (1000, 1100), (333, 777), (4, 4), (1, 1)])
def test_type_canvas_when_no_exact_aspect_exists_crops_only_a_sliver(W, H):
    cw, ch = OM.type_canvas(W, H)
    assert cw % 64 == 0 and ch % 64 == 0 and cw >= W and ch >= H
    x0, y0, x1, y1 = OM.visible_rect((cw, ch), (W, H))
    lost = 1.0 - (x1 - x0) * (y1 - y0) / (cw * ch)                                                         # canvas share cover-fit crops away (compose centre-crops, it never stretches)
    assert 0.0 <= x0 and 0.0 <= y0 and x1 <= cw + 1e-6 and y1 <= ch + 1e-6
    if min(W, H) >= 300:
        assert lost < 0.01, (W, H, cw, ch, lost)


def test_540x960_is_typed_at_576x1024_and_still_composed_at_540x960():
    n = OM.native_sizes(540, 960, "draft")
    assert n["compose"] == n["clip"] == n["shot"] == (540, 960) and n["type"] == (576, 1024) and n["piece"]["compose"] == [540, 960] and n["piece"]["type"] == [576, 1024]
    assert abs(n["type_aspect"] - 540 / 960) < 1e-9
    from tools.fusion import spec as fspec
    with pytest.raises(ValueError):
        fspec.check_size((540, 960))                                                                       # the old failure: the type engine takes multiples of 64 only
    assert fspec.check_size(n["type"]) == (576, 1024)
    argv = [str(x) for x in J.retime_argv("py", "/r", "/w", "/o", "STEM", 72.0, 3, "overlay", n, n["matte"])]
    assert argv[argv.index("--native-size") + 1] == "540x960" and abs(float(argv[argv.index("--aspect") + 1]) - 0.5625) < 1e-6


def test_the_canvas_is_fitted_back_to_the_source_without_moving_the_type():
    import numpy as np
    from tools.fusion import compose
    W, H = 540, 960
    cw, ch = OM.type_canvas(W, H)
    lay = np.zeros((ch, cw, 3), np.float32)
    bx0, by0, bx1, by1 = int(cw * 0.25), int(ch * 0.40), int(cw * 0.75), int(ch * 0.55)                  # a word box at a known fraction of the frame
    lay[by0:by1, bx0:bx1] = 1.0
    out = compose.fit_cover(lay, (W, H))
    assert out.shape[:2] == (H, W)
    ys, xs = np.nonzero(out[..., 0] > 0.5)
    assert abs(xs.min() / W - 0.25) < 0.01 and abs(xs.max() / W - 0.75) < 0.01 and abs(ys.min() / H - 0.40) < 0.01 and abs(ys.max() / H - 0.55) < 0.01
    # a canvas wider than the frame: cover-fit centre-crops the margin, the words laid out inside the visible rect all survive
    c2 = OM.type_canvas(852, 480)
    vx0, vy0, vx1, vy1 = OM.visible_rect(c2, (852, 480))
    lay2 = np.zeros((c2[1], c2[0], 3), np.float32)
    lay2[int(vy0) + 10:int(vy1) - 10, int(vx0) + 10:int(vx1) - 10] = 1.0
    o2 = compose.fit_cover(lay2, (852, 480))
    assert o2.shape[:2] == (480, 852) and (o2[..., 0] > 0.5).sum() >= 0.95 * (852 - 24) * (480 - 24)


def test_type_engine_renders_a_source_that_is_not_a_multiple_of_64_and_the_overlay_stays_exact(tmp_path):
    """150x200 source (neither side a multiple of 64): the type engine refuses it (check_size), renders at the type canvas, compose cover-fits the layer to the source size,
    and the restored overlay video keeps the source size / fps / frame count and passes overlay_check (picture unchanged outside the type)."""
    if not HAVE_FFMPEG:
        pytest.skip("needs ffmpeg")
    import numpy as np
    import cv2
    from tools.app import overlay_check as OC
    from tools.fusion import compose, type_passes
    W, H, N = 150, 200, 8
    nat = OM.native_sizes(W, H, "draft")
    assert nat["compose"] == (W, H) and nat["type"] == (192, 256)
    tspec = {"size": [W, H], "fps": 12, "duration": 1.0, "bpm": 120,
             "words": [{"text": "AB", "module": "lanes", "font": "Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf", "axes": {"wght": [300, 700, "beat"]}, "form_in": "flicker", "form_out": "flicker",
                        "t0": 0, "t1": 1.0, "form_s": 0.2, "exit_s": 0.2}], "camera": "beat_orbit"}
    with pytest.raises(ValueError):
        type_passes.render_passes(tspec, str(tmp_path / "bad"), size=(W, H), frames=(0, 1), procs=1, mp4=False, verbose=False)
    out = str(tmp_path / "type")
    type_passes.render_passes(tspec, out, size=nat["type"], frames=(2, 2 + N), procs=1, mp4=False, verbose=False)

    def write(path, frames, fps):
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        assert p.wait() == 0
    base = []
    for i in range(N):
        f = np.zeros((H, W, 3), np.uint8)
        f[:, :, 1] = np.linspace(60, 160, W).astype(np.uint8)[None, :] + i
        f[:, :, 2] = np.linspace(40, 90, H).astype(np.uint8)[:, None]
        base.append(f)
    typed, seen = [], 0
    for i in range(N):
        lay = cv2.imread(f"{out}/beauty/beauty_{2 + i:05d}.png")
        mat = cv2.imread(f"{out}/matte/matte_{2 + i:05d}.png", cv2.IMREAD_UNCHANGED)
        assert lay.shape[:2] == (256, 192) and mat.shape[:2] == (256, 192)
        lay = compose.fit_cover(lay.astype(np.float32) / 255.0, (W, H))
        mat = compose.fit_cover(mat.astype(np.float32)[..., None] / 255.0, (W, H))
        mat = mat if mat.ndim == 2 else mat[..., 0]
        assert lay.shape[:2] == (H, W) and mat.shape == (H, W)
        a = (mat > 0.2)[..., None]
        seen += int(a.sum() > 0)
        typed.append(np.where(a, (lay * 255).astype(np.uint8), base[i]))
    assert seen >= 2                                                                                         # the type is on screen in some frames
    src, raw, pic = tmp_path / "src.mp4", tmp_path / "raw.mp4", tmp_path / "picture.mp4"
    write(src, base, "12")
    write(raw, typed, "12")
    clip = tmp_path / "s1_clip.mp4"
    write(clip, base + [base[-1]] * 3, "12")
    rep = OM.restore_video(src, raw, [clip], [N], pic, "12")
    assert rep["frames"] == N and rep["source_size"] == [W, H]
    ps, pp = OM.probe_video(src), OM.probe_video(pic)
    assert (pp["width"], pp["height"], pp["frames"], pp["fps_str"]) == (W, H, N, ps["fps_str"])
    res = OC.compare(src, pic, 0.0, N, {"word_frames": [dict(text="AB", start_frame=0, end_frame=N)]})
    assert res["ok"], res
    assert res["checks"]["size"]["ok"] and res["checks"]["fps"]["ok"] and res["checks"]["frames"]["ok"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_frames_reads_every_decoded_frame_even_when_the_pts_run_on_the_24fps_clock(tmp_path):
    """a shot clip of a 30 fps source is tagged 30/1 but written with setpts=N/(24*TB) (box_cut_v3.sh): the reader must not duplicate frames to hold 30 fps (that shifted the footage against the render
    and made the restore see ground / wash everywhere on every K != 1 source)."""
    import numpy as np
    W, H, N = 64, 48, 20
    raw = tmp_path / "raw.rgb"
    raw.write_bytes(b"".join(np.full((H, W, 3), 10 + 10 * i, np.uint8).tobytes() for i in range(N)))
    src, clip = tmp_path / "src.mp4", tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", str(raw), "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(src)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", "trim=start_frame=0:end_frame=20,setpts=N/(24*TB),setsar=1", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-fps_mode", "passthrough", str(clip)], check=True)
    got = [int(f[..., 0].mean()) for f in OM._frames(clip, W, H, bt601=False)]
    assert len(got) == N, len(got)
    assert got == sorted(set(got)) and got[0] < got[-1], got                                               # every frame once, in order (a cfr conversion repeats 1 in 5)
    assert [int(f[..., 0].mean()) for f in OM._frames(clip, W, H, 7, bt601=False)] == got[:7]


def test_word_windows_merge_a_wide_gap_where_the_strands_run_across_it():
    from tools.app import overlay_check as OC
    rep = {"word_frames": [dict(start_frame=389, end_frame=556), dict(start_frame=606, end_frame=677), dict(start_frame=706, end_frame=799)]}      # 540x960 30 fps run: 50-frame gap between two words
    assert OC.word_windows(rep, 4) == [(385, 803)]
    assert OC.word_windows({"word_frames": [dict(start_frame=10, end_frame=50), dict(start_frame=400, end_frame=450)]}, 4) == [(6, 54), (396, 454)]          # a genuinely empty stretch stays outside every window
