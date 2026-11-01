"""Transition rubric columns for the open-model par tables (tools/fusion/transition_rubric.py). Pure functions on piece dicts.
Run: python -m pytest tests/test_transition_rubric.py -q"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import transition_rubric as TR  # noqa: E402


def piece(tr="auto", n_words=5, cf=0, shots=3, cuts=None, words=None, camera=None):
    ws = words or [{"text": f"W{i}"} for i in range(n_words)]
    t = {"words": ws, "transitions": tr}
    if camera:
        t["camera"] = camera
    comp = {"crossfade": cf}
    if cuts:
        comp["cuts"] = cuts
    return {"type": t, "shots": [{"id": f"s{i + 1}"} for i in range(shots)], "compose": comp}


def test_auto_writes_nothing_and_scores_by_what_is_written():
    c = TR.columns(piece("auto"))
    assert (c["text_used"], c["text_distinct"], c["text"]) == (0, 0, 0) and c["text_auto"]
    c = TR.columns(piece(["twist", "ripple", "braid", "cut"]))
    assert (c["text_used"], c["text_distinct"], c["text"], c["repeats"]) == (4, 3, 2, 0)


def test_cut_does_not_count_as_distinct_and_repeats_are_counted():
    s = TR.stats(piece(["twist", "twist", "cut", "cut"]))
    assert s["text"]["distinct"] == 1 and s["text"]["cuts"] == 2 and s["text"]["repeats"] == 1 and s["repeats"] == 1
    c = TR.columns(piece(["twist", "twist", "cut", "cut"]))
    assert c["text"] == 1 and c["repeats_score"] == 1
    assert TR.columns(piece(["twist", "twist", "twist", "twist"]))["repeats_score"] == 0


def test_dict_entries_null_and_short_lists():
    s = TR.stats(piece([{"module": "ripple", "span": 0.9}, None]))
    assert s["text"]["names"] == ["ripple", None, None, None] and s["text"]["used"] == 1


def test_video_kinds_per_boundary_and_beyond():
    p = piece(cf=[0, 0.4], cuts=[{"after": "s1", "cut": "window_cut"}])
    s = TR.stats(p)
    assert s["video"]["kinds"] == ["window_cut", "crossfade"] and s["video"]["beyond"] == 1
    c = TR.columns(p)
    assert c["video"] == 2 and c["video_beyond"] == 1 and c["linked"] == 1 and c["linked_score"] == 2
    assert TR.columns(piece(cf=0))["video"] == 0                                # all plain hard cuts
    assert TR.columns(piece(cf=[0, 0.3]))["video"] == 1                         # hard + crossfade mixed on purpose
    two = piece(cf=0, cuts=[{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "depth_pivot"}])
    assert TR.stats(two)["video"]["repeats"] == 1 and TR.stats(two)["repeats"] == 1
    assert TR.columns(piece(shots=1))["video"] is None                           # a single shot has no boundary


def test_linked_pan_window_words_and_ports():
    w = [{"text": "A", "transition": {"kind": "pan", "dir": "down"}}, {"text": "B", "window_source": "next"}, {"text": "C", "via": {"port": "depth_iso"}}, {"text": "D"}]
    s = TR.stats(piece(["twist", "twist", "cut"], words=w))
    assert s["text"]["pans"] == 1 and s["text"]["names"][0] == "pan"
    assert s["linked"] == dict(window_cuts=0, window_words=1, pans=1, ports=1, via=1)
    assert TR.columns(piece(words=w, tr="auto"))["linked_score"] == 2
    only_src = TR.columns(piece(words=[{"text": "A", "window_source": "s2"}, {"text": "B"}], tr=[None]))
    assert only_src["linked_score"] == 1 and only_src["linked"] == 1
    assert TR.columns(piece("auto", camera={"move": "pan", "dir": "down"}))["video"] == 2


def test_io_styles_make_a_boundary_score_one_and_one_word_has_no_text_gap():
    w = [{"text": "A", "form_out": "wipe"}, {"text": "B", "form_in": ["wipe", {"soft": 0.1}]}]
    assert TR.stats(piece(words=w))["io"] == ["wipe"] and TR.columns(piece(words=w, cf=0))["video"] == 1
    assert TR.columns(piece(words=[{"text": "A"}]))["text"] is None


def test_claude_pieces_if_present():
    path = os.path.join(ROOT, "15_fusion", "pieces", "P14.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        c = TR.columns(json.load(f))
    assert c["text_distinct"] == 3 and c["text"] == 2 and c["video"] is not None


def test_shot_transition_columns_distinct_3d_windows_front_and_direction():
    ops = {"ops": [{"op": "upscale_matte"}, {"op": "sandwich"}, {"op": "grain"}]}
    p = piece(["twist", "ripple", "braid", "cut"], shots=4, cf=0, words=[{"text": "A", "exit": "left"}, {"text": "B", "entry": "right"}, {"text": "C"}, {"text": "D"}, {"text": "E"}],
              cuts=[{"after": "s1", "cut": "depth_dolly:right"}, {"after": "s2", "cut": "hard", "was": "window_other", "word": 2}, {"after": "s3", "cut": "orbit:push"}])
    p["compose"]["default"] = ops
    c = TR.columns(p)
    assert c["shot_kinds"] == ["depth_dolly:right", "window_other", "orbit:push"] and c["shot_distinct"] == 3 and c["three_d"] == 2 and c["windows"] == 1
    assert c["subject_front"] == 1.0 and c["shared_direction"] == 1.0 and c["video"] == 2 and c["repeats"] == 0
    q = json.loads(json.dumps(p))
    q["compose"]["cuts"][0]["params"] = {"matte_pass": False}
    q["compose"]["default"] = {"ops": [{"op": "grain"}]}
    cq = TR.columns(q)
    assert cq["subject_front"] == 0.0                                          # matte pass off and no sandwich: the subject is not held in front
    q["type"]["words"][0].pop("exit")
    assert TR.columns(q)["shared_direction"] == 0.0
    plain = TR.columns(piece(["twist"], shots=3, cf=0))
    assert plain["subject_front"] is None and plain["shared_direction"] is None and plain["three_d"] == 0 and plain["shot_distinct"] == 1
    dup = TR.columns(piece("auto", shots=3, cuts=[{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "depth_pivot"}]))
    assert dup["repeats"] == 1 and dup["shot_distinct"] == 1
    pan = TR.columns(piece("auto", shots=3, cuts=[{"after": "s1", "cut": "pan:left", "word": 0}], words=[{"text": "A", "transition": {"kind": "pan", "dir": "left"}}, {"text": "B"}]))
    assert pan["video_beyond"] == 1                                              # one pan, not counted twice (cut kind + word transition)
