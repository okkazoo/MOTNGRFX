"""User 2026-10-30 (horse piece): "we need the videos to continue animating through these transitions ... it never should go still". During ANY 3D cut A keeps playing its own frames (a0 + k) and B plays
from its first frame, so no two consecutive frames of a transition window may be identical when the source clips change every frame. Tiny synthetic clips whose every frame differs (a brightness ramp
on a checker), CPU, 64 x 112 mesh. Run: python -m pytest tests/test_transition_never_still.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_cut_styles as T  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402

H, W = T.H, T.WD


def ramp(n, base, step):
    yy, xx = np.mgrid[0:H, 0:W]
    chk = (((xx // 8 + yy // 8) % 2) * 10)[..., None]
    fr = np.stack([np.clip(np.zeros((H, W, 3), np.float32) + base + step * k + chk, 0, 255).astype(np.uint8) for k in range(n)])
    return fr, np.zeros((n, H, W), np.float32)


def cut(kind, style=None):
    a, ma = ramp(60, 20, 3)
    b, mb = ramp(60, 120, 2)
    p = T.piece(kind, dict(T.SMALL), n=60, split=False)
    if style:
        p["compose"]["cuts"][0]["style"] = style
    dm = np.full((H, W), 0.3, np.float32)
    res = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, {"s1": ma, "s2": mb}, {"s1": dm, "s2": dm})
    pl = res.plan
    return list(res.a[pl["a0"]:pl["a0"] + pl["i2"]]) + list(res.b[:pl["b_len"]]), pl, (a, b)


@pytest.mark.parametrize("kind,style", [("depth_pivot", None), ("orbit:push", "smooth"), ("orbit:pullback", "smooth"), ("orbit:fly", "smooth"), ("orbit:push", "slide"),
                                        ("depth_dolly:right", None), ("depth_dolly:left", None)])
def test_no_two_consecutive_frames_of_a_transition_are_identical(kind, style):
    seq, pl, _ = cut(kind, style)
    assert len(seq) == pl["i2"] + pl["b_len"]
    same = [i for i in range(len(seq) - 1) if np.array_equal(seq[i], seq[i + 1])]
    assert not same, (kind, "frozen at output frames", same)


@pytest.mark.parametrize("kind", ["depth_pivot", "orbit:push", "depth_dolly:right"])
def test_a_plays_its_own_next_frame_through_the_move_and_b_from_its_first(kind):
    seq, pl, (a, b) = cut(kind)
    ma = [float(f.mean()) for f in seq[:pl["i2"]]]
    src = [float(f.mean()) for f in a[pl["a0"]:pl["a0"] + pl["i2"]]]
    if kind.startswith("depth_dolly"):
        assert ma[:pl["params"]["hold1"] + 1] == pytest.approx(src[:pl["params"]["hold1"] + 1], abs=1.5)         # A is the clip's own frame a0 + k while it sits (the dolly then slides it away)
    else:
        assert ma == pytest.approx(src, abs=1.5)                                                                  # A's texture index advances every frame, the whole window
    assert float(seq[-1].mean()) == pytest.approx(float(b[pl["b_len"] - 1].mean()), abs=1.5)                      # B lands on its own frame b_len - 1: it played from frame 0 (no still)


def test_the_pre_move_hold_is_playing_footage_not_a_frozen_frame():
    seq, pl, (a, b) = cut("depth_pivot")
    h1 = pl["params"]["hold1"]
    assert h1 >= 1 and all(not np.array_equal(seq[i], seq[i + 1]) for i in range(h1 + 1))


# ----------------------------------------------------------------------------------------------- ship #10b: no freeze-frame inside a cut (user on ship #10 f175-193)
import cv2  # noqa: E402

from tools.fusion import recipe_eval as RE  # noqa: E402


def _shown(seq, start, holds, fps=24):
    """what the piece shows over the cut window: the cut's frames (footage input) with a moving synthetic type bar (type-pass input) on top, both routed through the compose freeze-frames
    (recipe_eval.hold_index), the way compose_frames reads them. -> list of 90x160 float frames."""
    rec = {"inputs": {"f": {"src": "footage"}, "t": {"dir": "x"}}, "holds": holds, "fps": fps, "holds_type": True}
    out = []
    for p in range(start, start + len(seq)):
        f = seq[RE.hold_index(rec, "f", p, 0, 0) - start].astype(np.float32)
        x = (RE.hold_index(rec, "t", p, 0, 0) * 2) % f.shape[1]
        f[:, x:x + 6] = 255.0                                                   # the "type": a bar that moves every piece frame
        out.append(cv2.resize(f, (90, 160), interpolation=cv2.INTER_AREA))
    return out


def _runs_below(fr, thr=1.0):
    d = [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
    best = run_ = 0
    for x in d:
        run_ = run_ + 1 if x < thr else 0
        best = max(best, run_)
    return best, d


def test_a_designed_freeze_inside_a_3d_cut_is_dropped_and_the_shown_output_never_stalls():
    seq, pl, _ = cut("depth_pivot")
    p = T.piece("depth_pivot", dict(T.SMALL), n=60, split=False)
    pl = PC.plan_cut(p, PC.specs(p)[0])
    start, n, i2 = int(pl["start"]), int(pl["n"]), int(pl["i2"])
    seq = seq[:n]
    inside = [(start + i2 + 2) / 24.0, 0.6]                                     # a Gemma 'dropout' freeze right after B takes over (ship #10: [7.29 s, 0.75 s] inside the pivot window)
    outside = [0.0, 0.1]
    kept = PC.holds_outside_cuts(p, [inside, outside])
    assert kept == [outside]
    run_with, _ = _runs_below(_shown(seq, start, [inside]))
    assert run_with > 3                                                         # the check is meaningful: the raw freeze stalls the picture
    run_ok, d = _runs_below(_shown(seq, start, kept))
    assert run_ok <= 3, (run_ok, [round(x, 1) for x in d])
    assert not any(d[i] > 3 * max(d[i - 1], 1.0) and d[i - 1] < 1.0 for i in range(1, len(d)))      # no stall-then-snap (ship #10: 0.7 -> 17.9)


def test_hold_dropping_reaches_the_recipe():
    p = T.piece("depth_pivot", dict(T.SMALL), n=60, split=False)
    pl = PC.plan_cut(p, PC.specs(p)[0])
    t_in = (int(pl["start"]) + int(pl["i2"]) + 2) / 24.0
    p["compose"]["holds"] = [[t_in, 0.6]]
    assert PC.holds_outside_cuts(p, p["compose"]["holds"]) == []
