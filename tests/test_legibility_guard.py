"""Legibility guard: a matte must never hide a word's letters for the whole hold (compose.legible / legible_plan, recipe_eval hook).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_legibility_guard.py -q      (CPU, synthetic data)"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

FPS = 24
W, H = 96, 48
N = 96                                   # 4 s
HOLD = (1.5, 3.0)                        # (hit, release) seconds -> frames 36..72
LETTERS = [(8, 12, 30, 36), (38, 12, 60, 36), (68, 12, 90, 36)]      # x0, y0, x1, y1: three letters side by side
RAMP = 0.25


@pytest.fixture(autouse=True)
def _restore_backend():
    prev = C.B
    yield
    C.B = prev


def type_alpha():
    ta = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in LETTERS:
        ta[y0:y1, x0:x1] = 1.0
    return ta


def bar_matte(frames, box=LETTERS[1]):
    """matte of N frames: 1 over the letter-2 box (a bar a little larger than the letter) on `frames`, else 0."""
    x0, y0, x1, y1 = box
    out = np.zeros((N, H, W), np.float32)
    for f in frames:
        out[f, y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 1.0
    return out


def visible(ta, m, box):
    x0, y0, x1, y1 = box
    t = ta[y0:y1, x0:x1]
    return float((t * (1.0 - m[y0:y1, x0:x1])).sum() / t.sum())


def run_guard(mattes, holds=(HOLD,), **kw):
    """plan once, then apply the op at every frame -> adjusted mattes [N,H,W]."""
    ta = type_alpha()
    plan = C.legible_plan(lambda f: ta, lambda f: mattes[min(f, N - 1)], list(holds), fps=FPS, **kw)
    outs = [C.legible(ta, mattes[f], holds=list(holds), t=f / FPS, plan=plan, ramp_s=RAMP) for f in range(N)]
    return plan, np.stack(outs)


f0, f1 = int(HOLD[0] * FPS), int(HOLD[1] * FPS)
ramp_f = int(round(RAMP * FPS))


def test_permanent_bar_over_letter_two_is_cleared_for_the_hold():
    mattes = bar_matte(range(N))
    plan, out = run_guard(mattes)
    h = plan["holds"][0]
    assert h["n"] == 3 and len(h["boxes"]) == 1                       # only letter 2 flagged
    assert h["hidden"][0] == 0 and h["hidden"][1] == 1.0 and h["hidden"][2] == 0
    ta = type_alpha()
    vis = np.array([visible(ta, out[f], LETTERS[1]) for f in range(f0, f1)])
    assert (vis >= 0.8).mean() >= 0.8
    assert vis.min() >= 0.95                                          # in fact the whole hold reads
    for k in (0, 2):                                                  # other letters untouched
        assert visible(ta, out[f0 + 5], LETTERS[k]) == pytest.approx(1.0)


def test_outside_the_holds_the_matte_is_untouched_and_the_ramp_is_gradual():
    mattes = bar_matte(range(N))
    _, out = run_guard(mattes)
    ta = type_alpha()
    for f in list(range(0, f0 - ramp_f - 1)) + list(range(f1 + ramp_f + 2, N)):
        assert np.array_equal(out[f], mattes[f]), f                   # exits / forming: letters fully occluded, unchanged
        assert visible(ta, out[f], LETTERS[1]) == pytest.approx(0.0)
    ramp = [visible(ta, out[f], LETTERS[1]) for f in range(f0 - ramp_f, f0 + 1)]
    assert all(b >= a - 1e-6 for a, b in zip(ramp, ramp[1:]))         # monotone rise into the hit, no pop
    assert 0.0 < ramp[len(ramp) // 2] < 1.0 and max(np.abs(np.diff(ramp))) < 0.45
    fall = [visible(ta, out[f], LETTERS[1]) for f in range(f1, f1 + ramp_f + 1)]
    assert all(b <= a + 1e-6 for a, b in zip(fall, fall[1:]))


def test_brief_pass_over_20_percent_of_the_hold_is_left_alone():
    n_hold = f1 - f0
    brief = range(f0 + 10, f0 + 10 + int(0.2 * n_hold))              # 20% of the hold
    mattes = bar_matte(brief)
    plan, out = run_guard(mattes)
    assert plan["holds"][0]["boxes"] == []
    assert np.array_equal(out, mattes)


def test_a_pass_over_more_than_20_percent_is_cleared():
    n_hold = f1 - f0
    mattes = bar_matte(range(f0 + 4, f0 + 4 + int(0.4 * n_hold)))
    plan, out = run_guard(mattes)
    assert len(plan["holds"][0]["boxes"]) == 1
    ta = type_alpha()
    assert all(visible(ta, out[f], LETTERS[1]) >= 0.8 for f in range(f0, f1))


def test_partial_cover_below_min_vis_counts_and_above_does_not():
    ta = type_alpha()
    x0, y0, x1, y1 = LETTERS[1]
    m = np.zeros((N, H, W), np.float32)
    m[:, y0:y1, x0:x0 + 10] = 1.0                                     # 10 of 22 columns hidden: 55% visible
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS)
    assert len(plan["holds"][0]["boxes"]) == 1
    m[:] = 0.0
    m[:, y0:y1, x0:x0 + 3] = 1.0                                      # 86% visible
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS)
    assert plan["holds"][0]["boxes"] == []


def test_two_holds_second_one_is_clear():
    h2 = (3.3, 3.9)
    mattes = bar_matte(range(0, 84))                                  # covers hold 1 fully, not hold 2 (frames 79..93)
    mattes[80:] = 0
    plan, out = run_guard(mattes, holds=(HOLD, h2))
    assert len(plan["holds"][0]["boxes"]) == 1 and plan["holds"][1]["boxes"] == []


def test_mode_off_no_holds_no_plan_are_noops():
    ta = type_alpha()
    m = bar_matte(range(N))[40]
    plan = C.legible_plan(lambda f: ta, lambda f: m, [HOLD], fps=FPS)
    t = 2.0
    assert np.array_equal(C.legible(ta, m, holds=[HOLD], t=t, plan=plan, mode="off"), m)
    assert np.array_equal(C.legible(ta, m, holds=None, t=t, plan=plan), m)
    assert np.array_equal(C.legible(ta, m, holds=[HOLD], t=t, plan=None), m)
    with pytest.raises(ValueError):
        C.legible(ta, m, holds=[HOLD], t=t, plan=plan, mode="nope")
    assert not np.array_equal(C.legible(ta, m, holds=[HOLD], t=t, plan=plan), m)


def test_invert_clears_toward_one():
    ta = type_alpha()
    x0, y0, x1, y1 = LETTERS[1]
    m = np.ones((N, H, W), np.float32)                                # inverted: occluder = 1 - m = 0 outside the bar
    m[:, y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 0.0                          # -> occluder 1 over letter 2
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, invert=True)
    assert len(plan["holds"][0]["boxes"]) == 1
    out = C.legible(ta, m[50], holds=[HOLD], t=2.0, plan=plan, invert=True)
    assert visible(ta, 1.0 - out, LETTERS[1]) >= 0.95


def test_solid_threshold_means_grey_matte_does_not_trigger():
    ta = type_alpha()
    m = np.full((N, H, W), 0.3, np.float32)                           # grey haze: below the sandwich threshold -> shows fully
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, matte_threshold=0.5, matte_soft=0.08)
    assert plan["holds"][0]["boxes"] == []
    m[:] = 0.9
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, matte_threshold=0.5, matte_soft=0.08)
    assert len(plan["holds"][0]["boxes"]) == 3


# ============================================================================================ recipe_eval hook
def make_piece(ops, holds=(HOLD,), legible=None):
    p = {"fps": FPS, "shots": [{"id": "s1", "frames": N}], "sizes": {"compose": [W, H]}, "type_passes": ["matte"],
         "compose": {"default": {"ops": ops}}, "plan": {"flow_wake": {"holds": [list(h) for h in holds]}}}
    if legible is not None:
        p["compose"]["legible"] = legible
    return p


def sandwich_ops(**extra):
    a = dict(footage="$footage", type_rgb=[1.0, 1.0, 1.0], type_a="$type_matte", matte="$matte", matte_threshold=0.5)
    a.update(extra)
    return [{"op": "sandwich", "out": "image", "args": a}]


def type_dir(tmp):
    d = Path(tmp) / "matte"
    d.mkdir(parents=True)
    ta = (type_alpha() * 255).astype(np.uint8)
    for i in range(N):
        cv2.imwrite(str(d / f"{i:05d}.png"), ta)
    return Path(tmp)


def letter2_luma(frame):
    x0, y0, x1, y1 = LETTERS[1]
    return float(np.asarray(frame, np.float32)[y0:y1, x0:x1].mean() / 255.0)


@pytest.mark.parametrize("be", ["numpy"] + (["torch"] if C.torch is not None else []))
def test_build_recipe_inserts_guard_and_compose_clears_letter(be):
    mattes = bar_matte(range(N))
    footage = np.zeros((N, H, W, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp, RE.use_backend(be):
        td = type_dir(tmp)
        piece = make_piece(sandwich_ops())
        rec = RE.build_recipe(piece, "s1", type_dir=td)
        names = [o["op"] for o in rec["ops"]]
        assert names == ["legible", "sandwich"], names
        leg = rec["ops"][0]["args"]
        assert leg["holds"] == [[1.5, 3.0]] and leg["matte_threshold"] == 0.5
        assert rec["ops"][1]["args"]["matte"] == "$" + rec["ops"][0]["out"]
        out, st = RE.run_compose(piece, "s1", type_dir=td, footage=footage, matte=mattes, backend=be)
        out = out.numpy() if hasattr(out, "numpy") else out
        assert st["legible"]
        inside = [letter2_luma(out[f]) for f in range(f0, f1)]
        assert min(inside) > 0.9                                      # letter 2 reads for the whole hold
        for f in (2, 10, 85, 94):                                     # forming / exiting: occluded as the matte says
            assert letter2_luma(out[f]) < 0.05
        # opt out: compose.legible = false + compose.legible_floor = false keeps the letter hidden (legible = false alone keeps the 2026-11-01 hard floor)
        off = make_piece(sandwich_ops(), legible=False)
        off["compose"]["legible_floor"] = False
        assert [o["op"] for o in RE.build_recipe(off, "s1", type_dir=td)["ops"]] == ["sandwich"]
        o2, _ = RE.run_compose(off, "s1", type_dir=td, footage=footage, matte=mattes, backend=be)
        o2 = o2.numpy() if hasattr(o2, "numpy") else o2
        assert max(letter2_luma(o2[f]) for f in range(f0, f1)) < 0.05


def test_build_recipe_without_holds_or_with_op_optout_is_untouched():
    p = make_piece(sandwich_ops(), holds=())
    p["plan"] = {}
    assert [o["op"] for o in RE.build_recipe(p, "s1", type_dir=".")["ops"]] == ["sandwich"]
    p2 = make_piece(sandwich_ops(legible=False))
    p2["compose"]["legible_floor"] = False
    assert [o["op"] for o in RE.build_recipe(p2, "s1", type_dir=".")["ops"]] == ["sandwich"]
    p3 = make_piece(sandwich_ops(legible=False))                       # op opt-out alone: only the word-level floor (letters off)
    ops = RE.build_recipe(p3, "s1", type_dir=".")["ops"]
    assert [o["op"] for o in ops] == ["legible", "sandwich"] and ops[0]["args"]["letters"] is False and ops[0]["args"]["floor"]


def test_explicit_legible_op_gets_holds_and_is_not_doubled():
    ops = [{"op": "legible", "out": "m", "args": {"type_a": "$type_matte", "matte": "$matte", "t": "$t"}}]
    ops += sandwich_ops(matte="$m")
    rec = RE.build_recipe(make_piece(ops), "s1", type_dir=".")
    assert [o["op"] for o in rec["ops"]] == ["legible", "sandwich"]
    assert rec["ops"][0]["args"]["holds"] == [[1.5, 3.0]]


def test_depth_occlude_gets_depth_cover_and_guard():
    ops = [{"op": "depth_occlude", "out": "image", "args": dict(footage="$footage", footage_depth="$depth", type_rgb=[1.0, 1.0, 1.0],
                                                                 type_a="$type_matte", plane=0.5, softness=0.1)}]
    rec = RE.build_recipe(make_piece(ops), "s1", type_dir=".")
    assert [o["op"] for o in rec["ops"]] == ["depth_cover", "legible", "depth_occlude"]
    assert rec["ops"][2]["args"]["cover"] == "$" + rec["ops"][1]["out"]
    footage = np.zeros((N, H, W, 3), np.uint8)
    depth = np.zeros((N, H, W), np.float32)
    x0, y0, x1, y1 = LETTERS[1]
    depth[:, y0 - 2:y1 + 2, x0 - 2:x1 + 2] = 1.0                      # near object permanently in front of letter 2
    with tempfile.TemporaryDirectory() as tmp:
        td = type_dir(tmp)
        out, _ = RE.run_compose(make_piece(ops), "s1", type_dir=td, footage=footage, depth=depth, backend="numpy")
        out = out.numpy() if hasattr(out, "numpy") else out
        assert min(letter2_luma(out[f]) for f in range(f0, f1)) > 0.9
        assert letter2_luma(out[3]) < 0.05


def test_depth_occlude_cover_default_unchanged():
    rng = np.random.default_rng(3)
    foot = rng.random((H, W, 3)).astype(np.float32)
    dep = rng.random((H, W)).astype(np.float32)
    ta = type_alpha()
    a = C.depth_occlude(foot, dep, (1, 1, 1), ta, plane=0.5, softness=0.1)
    cov = C.depth_cover(dep, 0.5, 0.1)
    b = C.depth_occlude(foot, dep, (1, 1, 1), ta, plane=0.5, softness=0.1, cover=cov)
    assert np.allclose(a, b, atol=1e-6)


def test_recipes_carry_the_guard():
    import json
    for name in ("behind", "behind_sheer"):
        r = json.loads((ROOT / "tools" / "fusion" / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        ops = r["compose"]["default"]["ops"]
        k = [o["op"] for o in ops]
        assert k.index("legible") < k.index("sandwich")
        assert ops[k.index("legible")]["out"] == "m" and ops[k.index("sandwich")]["args"]["matte"] == "$m"


def test_front_region_already_in_front_is_not_flagged():
    ta = type_alpha()
    m = bar_matte(range(N))
    front = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = LETTERS[1]
    front[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = 1.0                         # sandwich's `front`: the type is in front here already
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, front_at=lambda f: front)
    assert plan["holds"][0]["boxes"] == []
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS)
    assert len(plan["holds"][0]["boxes"]) == 1


def test_build_recipe_mirrors_front_into_the_guard():
    ops = sandwich_ops(front="$type_matte")
    rec = RE.build_recipe(make_piece(ops), "s1", type_dir=".")
    assert rec["ops"][0]["op"] == "legible" and rec["ops"][0]["args"]["front"] == "$type_matte"


# ---- the behind quota (compose.LEG_BEHIND, user 2026-11-01 "none of the text is being masked": a rule for every video, image mode included)
def _half_bar(frames):
    """matte over the left 25 % of letter 2 only (word occlusion ~0.05, that letter 75 % visible: it reads behind - letter >= 70 %, word >= 85 % visible, 2026-11-02 rule - but the per-letter
    rule (min_vis 0.8) still brings that one letter in front when there is no quota)."""
    x0, y0, x1, y1 = LETTERS[1]
    out = np.zeros((N, H, W), np.float32)
    for f in frames:
        out[f, y0:y1, x0:x0 + int(0.25 * (x1 - x0))] = 1.0
    return out


def test_strict_guard_fronts_a_partly_covered_word_and_the_quota_keeps_one_behind():
    ta = type_alpha()
    m = _half_bar(range(N))
    strict = C.legible_plan(lambda f: ta, lambda f: m[min(f, N - 1)], [HOLD], fps=FPS, floor=True)
    assert strict["holds"][0]["boxes"], "without the quota the covered letters come in front"
    q = C.legible_plan(lambda f: ta, lambda f: m[min(f, N - 1)], [HOLD], fps=FPS, floor=True, behind=True)
    assert q["holds"][0]["boxes"] == [] and q["holds"][0]["behind"] is True and q["holds"][0]["floor"] is False      # one word overlapped -> at least one stays behind


def test_quota_takes_about_half_and_never_a_word_the_subject_hides():
    ta = type_alpha()
    holds = [(0.2, 1.0), (1.5, 2.3), (2.8, 3.6)]
    m = np.zeros((N, H, W), np.float32)
    m[:] = _half_bar(range(N))                                           # all three holds are overlapped the same way
    p = C.legible_plan(lambda f: ta, lambda f: m[min(f, N - 1)], holds, fps=FPS, floor=True, behind=True)
    assert sum(1 for h in p["holds"] if h.get("behind")) == 2             # round(0.5 * 3) = 2 of the 3 overlapped words stay behind
    assert sum(1 for h in p["holds"] if h["boxes"]) == 1                  # the third comes in front whole
    full = bar_matte(range(N))                                           # letter 2 fully covered: not readable behind (letter < 10 % visible) -> never chosen
    pf = C.legible_plan(lambda f: ta, lambda f: full[min(f, N - 1)], [HOLD], fps=FPS, floor=True, behind=True)
    assert pf["holds"][0]["boxes"] and not pf["holds"][0].get("behind")
    none = C.legible_plan(lambda f: ta, lambda f: np.zeros((H, W), np.float32), [HOLD], fps=FPS, floor=True, behind=True)
    assert not none["holds"][0].get("behind") and none["holds"][0]["boxes"] == []        # no subject near the word: nothing to choose


def test_guard_ops_puts_the_quota_on_every_subject_over_type_draw_and_it_can_be_turned_off():
    import copy as _c
    ops = [{"op": "sandwich", "out": "image", "args": {"footage": "$footage", "type_rgb": "$tc", "type_a": "$ta", "matte": "$matte"}}]
    piece = {"fps": 24, "plan": {"flow_wake": {"holds": [[1.0, 2.0]]}}, "compose": {}, "type": {"words": [{"text": "A"}]}}
    leg = [o for o in RE.guard_ops(piece, _c.deepcopy(ops)) if o["op"] == "legible"]
    assert leg and leg[0]["args"]["behind"] == C.LEG_BEHIND
    piece["compose"]["behind_words"] = False
    leg = [o for o in RE.guard_ops(piece, _c.deepcopy(ops)) if o["op"] == "legible"]
    assert leg and "behind" not in leg[0]["args"]
