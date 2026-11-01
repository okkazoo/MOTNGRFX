"""2026-10-30 (user on cut_s1-s2_sweep.mp4: "don't use this - just do a simple straight push in or pull out"): no contour sweep on SHOT cuts; the straight push / pull kinds have zero rotation.
Run one file at a time; numpy / cv2 only (the push / pull pose test reads the depth_orbit pose function, no render)."""
import os

os.environ.setdefault("FUSION_BACKEND", "numpy")

import math

import pytest

from tools.fusion import contour_sweep as CS
from tools.fusion import cut_kinds as CK
from tools.fusion import random_system as RS


def piece(cuts, transitions=True):
    words = [{"text": "ONE", "t0": 0.2, "t1": 2.6, "exit_s": 0.4, "transition": "lines" if transitions else None},
             {"text": "TWO", "t0": 4.0, "t1": 6.4, "exit_s": 0.4}, {"text": "THREE", "t0": 8.0, "t1": 11.0, "exit_s": 0.4}]
    return {"id": "t", "fps": 24, "shots": [{"id": "s1", "frames": 96}, {"id": "s2", "frames": 96}, {"id": "s3", "frames": 96}],
            "layout": {"s1": {"start": 0, "frames": 96}, "s2": {"start": 96, "frames": 96}, "s3": {"start": 192, "frames": 96}},
            "type": {"bpm": 90.0, "duration": 12.0, "palette": {"fg": [250, 240, 230], "bg": [10, 12, 18]}, "words": words},
            "compose": {"cuts": cuts}}


HOLDS = [(0.5, 2.6), (4.4, 6.4), (8.4, 11.0)]
CUTS = [{"after": "s1", "cut": "orbit:push"}, {"after": "s2", "cut": "orbit:pullback"}]


@pytest.mark.parametrize("kind", ["orbit:push", "orbit:pullback", "orbit:fly", "depth_pivot", "pan:left", "window_cut", "window_other", "window_zoom", "depth_lines", "circle_grid"])
def test_no_sweep_event_on_any_shot_cut(kind):
    assert CS.SHOT_CUT_EVENTS is False
    p = CS.plan(piece([{"after": "s1", "cut": kind}, {"after": "s2", "cut": kind}]), seed=3, holds=HOLDS)
    assert not [e for e in p["events"] if e["type"] == "cut"], p["events"]


def test_the_sweep_stays_on_word_to_word_transitions():
    p = CS.plan(piece([]), seed=3, holds=HOLDS)
    assert [e for e in p["events"] if e["type"] == "word"]


def test_a_designed_piece_weaves_no_cut_sweep_ops():
    pc = piece(CUTS)
    pc["compose"]["contour_sweep"] = True
    CS.weave(pc, holds=HOLDS)
    ev = ((pc.get("plan") or {}).get("contour_sweep") or {}).get("events") or []
    assert not [e for e in ev if e.get("type") == "cut"], ev


def test_straight_push_and_pull_have_zero_rotation_and_no_sideways_or_rise():
    from mograph.modules.transitions import depth_pivot as DP
    dd, do = DP._engine()
    for mode in ("push", "pullback"):
        p = DP.resolve_params({"mode": mode})
        op = DP._op(do, DP.engine_args(p))
        xs = []
        for i in range(41):
            q = i / 40.0
            pose = do.volume_pose(mode, q, op)
            assert pose["yaw"] == 0.0 and pose["pitch"] == 0.0 and pose["roll"] == 0.0 and pose["fk"] == 1.0, (mode, q, pose)
            assert pose["pos"][0] == 0.0 and pose["pos"][1] == 0.0, (mode, q, pose)           # on the axis: no sideways drift, no mid-way rise
            xs.append(pose["pos"][2])
        d = [b - a for a, b in zip(xs, xs[1:])]
        assert all(v >= 0 for v in d) if mode == "push" else all(v <= 0 for v in d)             # monotonic: straight in / straight out
        assert math.isclose(abs(xs[-1]), float(op.vol_dz))


def test_push_and_pull_are_the_plain_default_chunked_moves_and_weighted_up():
    assert CK.style_of("orbit:push") == "chunks" and CK.style_of("orbit:pullback") == "chunks"
    m = RS.menu()
    assert m["orbit:push"][0] >= 2.0 and m["orbit:pullback"][0] >= 2.0 and m["orbit:push"][2] == ("chunks",) == m["orbit:pullback"][2]      # no lateral `slide` on the straight moves
    assert m["orbit:push"][0] > m["orbit:fly"][0]
    from tools.fusion import recipe_eval as RE
    q = piece(CUTS)
    q["shots"] = [{"id": s["id"], "frames": s["frames"]} for s in q["shots"]]
    assert RE.shot_needs(q, "s1")["depth"] and RE.shot_needs(q, "s2")["depth"]                  # real depth applies to both shots of a push / pull


def test_b_lines_start_when_a_3d_cut_hands_over_not_after_a_gap():
    """ship #10b (user on ship #10 f175-193: "the transition should resolve with the contour lines drawing at the same time"): a word event spanning a 3D cut switches from A's draw-off
    to B's draw-on at the cut's B take-over (s2 starts at 96 / 24 = 4.0 s), not at 40 % of the word gap."""
    pc = piece([{"after": "s1", "cut": "depth_pivot"}])
    pc["type"]["words"][0].update(t1=2.6, transition="lines")
    # words ONE 0.5-2.6, TWO 4.4-6.4; shots are 96 frames = 4.0 s, so s2 starts at 4.0 s inside the ONE -> TWO gap
    p = CS.plan(pc, seed=3, holds=HOLDS)
    ev = [e for e in p["events"] if e["type"] == "word" and e["t_a0"] < 4.0 < e["t_e"]]
    assert ev and abs(ev[0]["t_c"] - 4.0) < 1e-6, [(e["t_a0"], e["t_c"], e["t_e"]) for e in p["events"]]
    q = CS.plan(piece([]), seed=3, holds=HOLDS)
    e0 = [e for e in q["events"] if e["type"] == "word" and e["t_a0"] < 4.0 < e["t_e"]]
    assert e0 and abs(e0[0]["t_c"] - 4.0) > 1e-3                              # without the cut the split stays at its default
