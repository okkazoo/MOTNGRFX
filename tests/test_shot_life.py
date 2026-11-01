"""Own-video word lifetime + one word per shot + the speckle ban on the own-video route (user 2026-11-02). Synthetic fixtures only; no ComfyUI, no GPU.

  * a word lives inside the shot it is assigned to (form-in, hold, exit end by the shot's end cut + the short exit pad), a shot too short to read it extends to the first later cut, never to the next word
  * every shot whose span holds sung lyrics gets a word (no capacity cap); a shot with nothing sung has none; words stay inside their shot
  * `particles` never reaches an own-video piece (module pool + the plan-time swap)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import overlay_mode as OM          # noqa: E402
from tools.app import own_video_job as J          # noqa: E402
from tools.fusion import story as ST              # noqa: E402
from tools.fusion import word_pick as WP          # noqa: E402

B = 60.0 / 72.0          # pipeline beat of a 90 bpm song on the 1.25 stretched clock
NEED = 1.3               # a readable hold (spec.min_hold of a 10 letter word)
EXIT = B


def _end(hit, hold):
    return hit + hold + EXIT


# ------------------------------------------------------------------------------------------------------------ word lifetime
def test_long_shot_word_ends_by_the_cut_plus_the_short_pad():
    hit, shot_b = 3.0, 9.0                                       # lands at 3 s in a shot that ends at 9 s; the next word is 12 s away
    hold = OM.word_hold_s(hit, NEED, EXIT, shot_b, [9.0, 14.0], B, hold_until=20.0)
    assert hold >= NEED - 1e-9
    assert _end(hit, hold) <= shot_b + min(OM.EXIT_OVERLAP_S, B) + 1e-6          # never past the cut + the exit pad
    assert _end(hit, hold) >= shot_b - B                                          # and it does use the shot (not a 1.3 s blink)


def test_word_never_fills_the_gap_to_the_next_word():
    # the old rule held a word until the next word's landing - 3 beats; the new one ignores the next word entirely
    near = OM.word_hold_s(2.0, NEED, EXIT, 6.0, [6.0], B)
    far = OM.word_hold_s(2.0, NEED, EXIT, 6.0, [6.0], B, hold_until=60.0)
    assert near == far
    assert _end(2.0, near) <= 6.0 + OM.EXIT_OVERLAP_S + 1e-6


def test_short_shot_extends_only_as_far_as_the_readable_minimum():
    hit = 10.0                                                    # a 0.7 s shot (9.6-10.3): the word cannot be read before its end cut
    cuts = [10.3, 11.0, 13.5, 16.0]
    hold = OM.word_hold_s(hit, NEED, EXIT, 10.3, cuts, B)
    assert hold >= NEED - 1e-9
    assert _end(hit, hold) <= hit + NEED + EXIT + 0.25 * B + 1e-6                 # the readable minimum, not on to the cut at 13.5 / 16
    assert _end(hit, hold) < 13.5


def test_short_shot_leaves_on_a_cut_that_falls_within_a_beat_of_the_minimum():
    hit = 10.0
    m_end = hit + NEED + EXIT                                                      # 12.13
    cuts = [10.3, round(m_end + 0.4, 2), 16.0]
    hold = OM.word_hold_s(hit, NEED, EXIT, 10.3, cuts, B)
    assert abs(_end(hit, hold) - cuts[1]) <= 0.25 * B + 1e-6


def test_short_shot_with_no_later_cut_still_gets_its_readable_hold():
    hold = OM.word_hold_s(23.0, NEED, EXIT, 23.4, [], B)
    assert hold >= NEED - 1e-9


def test_no_shot_info_uses_the_slots_own_hold():
    hold = OM.word_hold_s(4.0, NEED, EXIT, None, [], B, hold_until=5.5)
    assert abs(hold - 1.5) <= 0.25 * B + 1e-6 and hold >= NEED - 1e-9
    assert OM.word_hold_s(4.0, NEED, EXIT, None, [], B, hold_until=4.2) >= NEED - 1e-9


def test_words_json_carries_the_shot_the_word_lands_in():
    cuts = [30, 120, 150]                                         # source frames of a 30 fps, 6 s window (pipeline clock = frame / 24)
    slots = [dict(land=2.0, hold_until=3.0), dict(land=4.1, hold_until=5.0)]
    ws = J.words_for(["ONE", "TWO"], slots, 1.25, cuts, 30.0, 180)
    assert (ws[0]["shot"], ws[0]["shot_a"], ws[0]["shot_b"]) == (2, round(30 / 24.0, 4), round(120 / 24.0, 4))
    assert (ws[1]["shot"], ws[1]["shot_a"], ws[1]["shot_b"]) == (3, round(120 / 24.0, 4), round(150 / 24.0, 4))
    assert ws[0]["cuts_p"] == [round(c / 24.0, 4) for c in cuts]
    plain = J.words_for(["ONE"], slots[:1], 1.25)[0]              # no cut info: the old shape, nothing added
    assert "shot_a" not in plain and plain["land"] == 2.5


def test_retime_box_uses_the_lifetime_and_has_no_next_word_hold():
    src = open(os.path.join(ROOT, "tools", "app", "own_video_v2_retime_box.py"), encoding="utf-8").read()
    assert "OVM.word_hold_s" in src and "nb - 3" not in src and "round((want[i + 1][\"land\"] - want[i][\"land\"]) / b)" not in src


# ------------------------------------------------------------------------------------------------------------ one word per shot
def _read(cuts, lyrics, dur=30.0):
    """a synthetic READ: cuts = [(start, end)], lyrics = [(line text, [sung seconds per word])]."""
    lyr = [dict(t=ts[0], text=text, ts=ts, end=ts[-1] + 0.3) for text, ts in lyrics]
    return dict(v=1, on_screen=[], concept="a test clip", look="plain", media=[dict(id="V1", kind="video", file="x.mp4", shows="people", window=[0, dur])], music=dict(bpm=90, key="C"),
                beat_grid=dict(bpm=90, t0=0, beats=[round(0.667 * k, 3) for k in range(1, int(dur / 0.667))], downbeats=[]), video_duration=dur,
                video_cuts=[dict(start=a, end=b) for a, b in cuts], lyrics=lyr, soundtrack="video")


def _story(read, dur=30.0, seed=7):
    return ST.fallback(dict(read=read, length=dur, seed=seed, format="9:16"))


CUTS12 = [(i * 3.0, (i + 1) * 3.0) for i in range(10)]            # ten 3 s shots (a word lands 0.6 s into its shot, so the landings are 3 s apart: the engine can play them all)
LINES12 = [(f"alpha{k} marble{k} garden{k}", [k * 3.0 + 0.5, k * 3.0 + 1.0, k * 3.0 + 1.5]) for k in range(10)]


def test_every_shot_with_sung_lyrics_gets_a_word_past_the_old_cap():
    out = _story(_read(CUTS12, LINES12))
    shots = {s["n"]: s for s in out["shots"]}
    assert len(out["hits"]) == 10 and len(out["hits"]) > ST.VIDEO_WORDS_DEFAULT + 4           # not the old 4-5 slots
    for h in out["hits"]:
        s = shots[h["shot"]]
        assert s["start"] - 1e-6 <= h["land"] < s["end"] - 1e-6, (h, s)                       # lands inside its own shot
        assert h["land"] < h["hold_until"] <= s["end"] + 1e-6                                  # holds inside it
        assert s["line"]                                                                      # and the shot says which words it carries
    assert sorted(h["shot"] for h in out["hits"]) == list(range(1, 11))
    assert {d["shot"] for d in out["text_select"]} == set(range(1, 11))


def test_shots_with_nothing_sung_have_no_word():
    lyrics = [LINES12[k] for k in (0, 1, 4, 5, 8)]                                            # shots 3, 4, 7, 8, 10 are instrumental
    out = _story(_read(CUTS12, lyrics))
    assert sorted(h["shot"] for h in out["hits"]) == [1, 2, 5, 6, 9]
    assert not any(s["line"] for s in out["shots"] if s["n"] in (3, 4, 7, 8, 10))


def test_a_long_shot_gets_a_word_where_the_old_pick_left_it_empty():
    # a 3.5 s shot with a sung line in it, a short one after it: the long shot has a content word, never "nothing"
    read = _read([(0, 3.5), (3.5, 4.2), (4.2, 9)], [("quiet harbour lights drift", [1.0, 1.6, 2.2, 2.8]), ("the", [6.0])], dur=9.0)
    out = _story(read, dur=9.0)
    hit = next(h for h in out["hits"] if h["shot"] == 1)
    assert hit["text"].split()[0] in {"QUIET", "HARBOUR", "LIGHTS", "DRIFT"} and 0.0 <= hit["land"] < 3.5
    assert not any(h["shot"] == 2 for h in out["hits"])                                      # 0.7 s shot, nothing sung in it
    assert not any(h["shot"] == 3 for h in out["hits"])                                      # only "the" is sung in shot 3: a function word is no word


def test_a_slot_hold_shorter_than_the_shot_is_kept_while_it_fits():
    hold = OM.word_hold_s(3.0, NEED, EXIT, 20.0, [20.0], B, hold_until=5.5)
    assert abs(hold - 2.5) <= 0.25 * B + 1e-6                                                # the slot's own 2.5 s, not 17 s
    short = OM.word_hold_s(3.0, NEED, EXIT, 3.4, [3.4, 20.0], B, hold_until=3.1)
    assert _end(3.0, short) <= 3.0 + NEED + EXIT + 0.25 * B + 1e-6                                          # a too-short shot is decided by the cut, not by a tiny slot hold


def test_a_phrase_that_starts_in_one_shot_stays_on_it():
    read = _read([(0, 4), (4, 8)], [("golden river runs", [3.2, 3.8, 4.4])], dur=8.0)        # "river"/"runs" are sung past the cut
    out = _story(read, dur=8.0)
    assert [h["shot"] for h in out["hits"]] == [1]
    assert out["hits"][0]["land"] < 4.0 and out["hits"][0]["hold_until"] <= 4.0 + 1e-6


def test_hits_closer_than_the_engine_can_play_are_named_not_moved():
    cuts = [(0, 1.0), (1.0, 2.0), (2.0, 6.0)]
    lyrics = [("stone", [0.5]), ("water", [1.5]), ("window", [3.0])]
    out = _story(_read(cuts, lyrics, dur=6.0), dur=6.0)
    lands = [h["land"] for h in out["hits"]]
    assert all(b - a >= ST.MIN_WORD_CYCLE_S - 1e-6 for a, b in zip(lands, lands[1:]))        # the engine's minimum word cycle
    assert "no word on this shot" in out["notes"]
    for h in out["hits"]:
        s = out["shots"][h["shot"] - 1]
        assert s["start"] <= h["land"] < s["end"]


def test_word_score_prefers_a_line_opener_and_a_name_in_the_line():
    toks = "Harbour light buzzing , Cat on the Harbour".replace(" ,", "").split()
    times = [1.0, 1.4, 1.8, 2.2, 2.4, 2.6, 2.8]
    P = WP.prepare(toks, times, lines=[(0, 2), (3, 6)], beats=[])
    sc = WP.word_scores(P)
    first, mid = WP.word_parts(P, 0), WP.word_parts(P, 6)
    assert first["end"] >= WP.LINE_START and sc[0] > sc[1]                                  # "Harbour" opens its line, "light" does not
    assert mid["shape"] >= WP.PROPER - 1e-9                                                  # "Harbour" is capitalised mid-line
    assert WP.word_parts(P, 3)["shape"] < WP.PROPER                                          # "Cat" opens the 2nd line: no name bonus


# ------------------------------------------------------------------------------------------------------------ speckle ban
def test_module_pool_has_no_speckle_module():
    pool = OM.module_pool(("slats", "gradient_fade", "staircase", "beat_bars", "particles", "outflow"))
    assert "particles" not in pool and "outflow" not in pool and set(pool) == {"slats", "gradient_fade", "staircase", "beat_bars"}      # `outflow` retired 2026-11-02
    src = open(os.path.join(ROOT, "tools", "app", "own_video_v2_retime_box.py"), encoding="utf-8").read()
    assert "OVM.module_pool(" in src and "OVM.mend_speckle(piece)" in src


def test_speckle_swap_on_an_own_video_piece():
    from tools.fusion import design_defaults as DD
    words = [dict(text="ONE", module="particles", form_in="particles", form_out="lines"), dict(text="TWO", module="slats", form_in="lines", form_out="blobs"),
             dict(text="THREE", module="particles", form_in="wipe", form_out="flicker")]
    piece = dict(type=dict(words=words))
    fixes = OM.mend_speckle(piece)
    assert fixes
    for w in piece["type"]["words"]:
        assert w["module"] not in DD.SPECKLE_MODULES and w["form_in"] not in DD.SPECKLE_IO and w["form_out"] not in DD.SPECKLE_IO
    assert piece["type"]["words"][0]["module"] != piece["type"]["words"][2]["module"]       # a piece of two speckle words becomes two different modules
    assert OM.mend_speckle(piece) == []                                                      # idempotent


def test_designed_slots_are_split_back_into_the_wanted_words():
    designed = ["ONE", "TWO THREE", "FOUR FIVE", "SIX"]
    wanted = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"]
    got = OM.expand_designed(designed, wanted)
    assert [t for _, t in got] == wanted and [d for d, _ in got] == [0, 1, 1, 2, 2, 3]       # each copy keeps its slot's design
    short = OM.expand_designed(["A", "B"], ["A", "B", "C", "D"])                             # a missing tail reuses the designs in turn
    assert [t for _, t in short] == ["A", "B", "C", "D"] and [d for d, _ in short] == [0, 1, 0, 1]
    assert OM.expand_designed(["X Y"], ["X", "Y"]) == [(0, "X"), (0, "Y")]


def test_restore_video_hides_a_words_form_in_before_its_shot():
    import shutil
    import subprocess
    import tempfile
    import numpy as np
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return
    W, H, N = 64, 48, 10
    with tempfile.TemporaryDirectory() as td:
        def write(path, frames):
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(path)], stdin=subprocess.PIPE)
            for f in frames:
                p.stdin.write(f.tobytes())
            p.stdin.close()
            assert p.wait() == 0
        base = [np.full((H, W, 3), 60, np.uint8) for _ in range(N)]
        typed = [f.copy() for f in base]
        for i in range(2, 9):
            typed[i][10:30, 10:50] = (0, 220, 255)                                                   # a word on frames 2..8 (its form-in is frames 2-4, its shot starts at frame 5)
        src, raw, clip, pic = (os.path.join(td, n) for n in ("src.mp4", "raw.mp4", "s1.mp4", "pic.mp4"))
        write(src, base)
        write(raw, typed)
        write(clip, base + [base[-1]] * 3)
        OM.restore_video(src, raw, [clip], [N], pic, "24", inactive_frames=[(2, 4)])
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", pic, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        fr = np.frombuffer(p.stdout, np.uint8).reshape(N, H, W, 3)
        painted = [bool((np.abs(f.astype(int) - 60) > 40).any()) for f in fr]
        assert painted == [False, False, False, False, False, True, True, True, True, False]         # frames 2-4 are the source, 5-8 keep the type
