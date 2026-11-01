"""The depth pivot-chunk transition as a shot cut (tools/fusion/pivot_cut.py): where its frames go on the piece timeline, the splice into the two
shots' plates, the hard-cut ownership. CPU only, tiny synthetic clips (64x112); one real tiny render checks the engine's frame alignment.
Run: python -m pytest tests/test_pivot_cut.py -q"""
from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import pivot_cut as PC  # noqa: E402
from tools.fusion import window_cut as W  # noqa: E402

H, WD = 112, 64
SMALL = dict(short=64, travel=6, hold1=3, hold2=3)          # 12 frames, image B visible from output frame 5


def piece(n_a=72, n_b=72, cuts=None, cf=0):
    return {"fps": 24, "shots": [{"id": "s1", "frames": n_a}, {"id": "s2", "frames": n_b}, {"id": "s3", "frames": 40}],
            "compose": {"crossfade": cf, "cuts": [{"after": "s1", "cut": "depth_pivot"}] if cuts is None else cuts}}


def clip(seed, n):
    base = np.random.default_rng(seed).integers(0, 200, (H // 8, WD // 8, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8, 1), np.uint8)).astype(int)
    return np.stack([np.clip(img + 4 * k, 0, 255).astype(np.uint8) for k in range(n)])


def matte(n):
    m = np.zeros((n, H, WD), np.float32)
    m[:, 40:80, 20:44] = 1.0
    return m


def depth():
    d = np.full((H, WD), 0.1, np.float32)
    d[:, :16] = 0.5
    d[40:80, 20:44] = 0.9
    return d


def test_specs_and_hard_cut_ownership():
    p = piece(cuts=[{"after": "s1", "cut": "depth_pivot", "params": {"chunks": 50}}, {"after": "s9", "cut": "depth_pivot"},
                    {"after": "s3", "cut": "depth_pivot"}, {"after": "s2", "cut": "crossfade"}])
    sp = PC.specs(p)
    assert [(s["a"], s["b"], s["params"]) for s in sp] == [("s1", "s2", {"chunks": 50})]       # unknown shot / last shot / crossfade are not cuts
    assert W.owned_boundaries(p) == [0]                                                         # s3 is the last shot: no boundary after it; the crossfade stays soft
    assert PC.specs(piece(cuts=[])) == []


def test_image2_start_is_the_v013_29_by_default_and_follows_the_angles():
    assert PC.image2_start() == 29                                    # 17_depth-dolly v013: B plays from output frame 29
    assert PC.image2_start(dict(in_angle=60.0)) != 29
    assert PC.image2_start(SMALL) == 5


def test_plan_arithmetic_default_params():
    pl = PC.plan_cut(piece(), PC.specs(piece())[0])
    assert (pl["n"], pl["i2"], pl["b_len"]) == (72, 29, 43)
    assert pl["start"] == 72 - 29 and pl["a0"] == 72 - 29                       # output frame 0 sits i2 frames before B's piece start; A starts at 0
    assert pl["window_a"] == [43, 72] and pl["window_b"] == [72, 115]
    assert pl["problems"] == []
    assert PC.window_ranges(pl) == {"s1": [43, 72], "s2": [72, 115]}


def test_plan_follows_the_crossfade_layout_and_reports_problems():
    p = piece(cf=0.5)                                                           # B starts 12 frames before A ends: A's window is 12 earlier in A's own frames
    pl = PC.plan_cut(p, PC.specs(p)[0])
    assert pl["window_a"][1] == 72 - 12 and pl["problems"] == []
    short = PC.plan_cut(piece(n_a=20), PC.specs(piece(n_a=20))[0])
    assert short["problems"] and "needs 29 frames" in short["problems"][0]
    short_b = PC.plan_cut(piece(n_b=30), PC.specs(piece(n_b=30))[0])
    assert any("hold2" in x for x in short_b["problems"])
    assert PC.plan_cut(piece(n_b=30), PC.specs(piece(n_b=30))[0], params=dict(hold2=4))["problems"] == []


def test_splice_replaces_exactly_the_planned_ranges_and_rescales():
    pl = PC.plan_cut(piece(), PC.specs(piece())[0])
    a, b = clip(1, 72), clip(2, 72)
    fr = np.stack([np.full((56, 32, 3), 10 + k, np.uint8) for k in range(pl["n"])])         # transition at half size
    na, nb = PC.splice(a, b, fr, pl)
    assert na.shape == a.shape and nb.shape == b.shape
    assert np.array_equal(na[:43], a[:43]) and np.array_equal(nb[43:], b[43:])           # nothing else touched
    assert abs(int(na[43].mean()) - 10) <= 1 and abs(int(na[71].mean()) - (10 + 28)) <= 1   # output frames 0..28 -> A[43:72]
    assert abs(int(nb[0].mean()) - (10 + 29)) <= 1 and abs(int(nb[42].mean()) - (10 + 71)) <= 1   # output frames 29..71 -> B[0:43]
    assert np.array_equal(a, clip(1, 72))                                                  # inputs are not modified
    with pytest.raises(ValueError):
        PC.splice(a, b, fr[:10], pl)
    with pytest.raises(ValueError):
        PC.splice(a, b, fr, PC.plan_cut(piece(n_a=20), PC.specs(piece(n_a=20))[0]))


def test_apply_cut_passes_the_planned_slices_and_checks_the_engine_alignment():
    seen = {}

    def fake(a, ma, b, mb, da, db, params):
        seen.update(a=len(a), b=len(b), ma=None if ma is None else len(ma), mb=None if mb is None else len(mb), da=None if da is None else len(da), p=params)
        return SimpleNamespace(frames=np.zeros((params["hold1"] + params["travel"] + params["hold2"], 56, 32, 3), np.uint8), info={"image2_start": 29})

    p = piece()
    clips = {"s1": clip(1, 72), "s2": clip(2, 72)}
    na, nb, pl = PC.apply_cut(p, PC.specs(p)[0], clips, mattes={"s1": matte(72), "s2": matte(72)}, depths={"s1": depth()[None]}, render=fake)
    assert seen["a"] == 29 and seen["b"] == 43 and seen["ma"] == 29 and seen["mb"] == 43 and seen["da"] == 1     # A from its frame 43 (to the clip end), B its first 43
    assert na.shape == clips["s1"].shape
    bad = lambda *a: SimpleNamespace(frames=np.zeros((72, 56, 32, 3), np.uint8), info={"image2_start": 30})        # noqa: E731
    with pytest.raises(ValueError, match="assumed 29"):
        PC.apply_cut(p, PC.specs(p)[0], clips, render=bad)


def test_real_tiny_render_lands_on_both_clips():
    p = piece(n_a=16, n_b=16, cuts=[{"after": "s1", "cut": "depth_pivot", "params": SMALL}])
    a, b = clip(1, 16), clip(2, 16)
    from mograph.modules.transitions import depth_pivot as DP
    real = lambda ca, ma, cb, mb, da, db, prm: DP.depth_pivot_render(ca, ma, cb, mb, depth(), depth(), prm)      # noqa: E731
    na, nb, pl = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, mattes={"s1": matte(16), "s2": matte(16)}, render=real)
    assert (pl["i2"], pl["a0"], pl["b_len"]) == (5, 11, 7) and pl["problems"] == []
    assert np.array_equal(na[:11], a[:11]) and np.array_equal(nb[7:], b[7:])
    d0 = np.abs(na[11].astype(int) - a[11].astype(int)).mean()                    # the transition starts on A's own frame (home pose)
    d1 = np.abs(nb[6].astype(int) - b[6].astype(int)).mean()                      # and ends on B's own frame (settled)
    assert d0 < 2.0 and d1 < 2.0, (d0, d1)
    assert np.abs(na[14].astype(int) - a[14].astype(int)).mean() > 3.0            # mid-transition frames really differ


# ----------------------------------------------------------------------------------------------- kinds with a direction, matte pass
def test_normalize_kinds_and_directions():
    assert [PC.normalize(k) for k in ["pivot", "depth_pivot", "orbit:push", "push", "Orbit:PullBack", "fly_through", "dolly:down", "depth_dolly:left"]] == \
        ["depth_pivot", "depth_pivot", "orbit:push", "orbit:push", "orbit:pullback", "orbit:fly", "depth_dolly:down", "depth_dolly:left"]
    assert PC.normalize("orbit") is None and PC.normalize("depth_dolly") is None and PC.normalize("depth_dolly:sideways") is None
    assert PC.normalize("pan:left") is None and PC.normalize("window_cut") is None and PC.normalize(None) is None       # pan / window cuts are other families
    assert set(PC.KINDS) == {"depth_pivot", "orbit:push", "orbit:pullback", "orbit:fly", "depth_dolly:down", "depth_dolly:up", "depth_dolly:left", "depth_dolly:right"}
    p = piece(cuts=[{"after": "s1", "cut": "dolly:up"}])
    assert PC.specs(p)[0]["kind"] == "depth_dolly:up" and W.owned_boundaries(p) == [0]


DOLLY_SMALL = dict(short=64, hold1=3, travel=6, settle=4, hold2=4, lag=0.5)


@pytest.mark.parametrize("kind,par", [("depth_pivot", SMALL), ("orbit:push", SMALL), ("orbit:pullback", SMALL), ("orbit:fly", SMALL),
                                      ("depth_dolly:down", DOLLY_SMALL), ("depth_dolly:right", DOLLY_SMALL)])
def test_every_3d_kind_renders_splices_and_writes_the_matte_through_it(kind, par):
    p = piece(n_a=40, n_b=40, cuts=[{"after": "s1", "cut": kind, "params": par}])
    a, b = clip(1, 40), clip(2, 40)
    res = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, mattes={"s1": matte(40), "s2": matte(40)}, depths={"s1": depth()[None], "s2": depth()[None]})
    pl = res.plan
    assert pl["kind"] == kind and pl["problems"] == []
    assert np.array_equal(res.a[:pl["a0"]], a[:pl["a0"]]) and np.array_equal(res.b[pl["b_len"]:], b[pl["b_len"]:])
    assert res.matte_a[0] == pl["a0"] and len(res.matte_a[1]) == pl["i2"] and res.matte_b[0] == 0 and len(res.matte_b[1]) == pl["b_len"]
    assert 0.0 <= res.matte_a[1].min() and res.matte_b[1].max() <= 1.0
    d0 = np.abs(res.a[pl["a0"]].astype(int) - a[pl["a0"]].astype(int)).mean()                  # starts on A's own frame
    d1 = np.abs(res.b[pl["b_len"] - 1].astype(int) - b[pl["b_len"] - 1].astype(int)).mean()      # lands on B's own frame (dolly: B's still = this frame)
    assert d0 < 3.0 and d1 < 3.0, (kind, d0, d1)
    assert np.abs(res.matte_a[1][0] - matte(40)[pl["a0"]]).mean() < 0.05                         # the matte pass starts on A's matte


def test_matte_pass_off_zeroes_the_old_matte_on_the_transformed_frames_and_no_mattes_give_no_patches():
    p = piece(n_a=40, n_b=40, cuts=[{"after": "s1", "cut": "orbit:push", "params": dict(SMALL, matte_pass=False)}])
    r = PC.apply_cut(p, PC.specs(p)[0], {"s1": clip(1, 40), "s2": clip(2, 40)}, mattes={"s1": matte(40), "s2": matte(40)}, depths={"s1": depth()[None], "s2": depth()[None]})
    pl = r.plan                                                # a matte over a picture it was not cut from shows two people: zero on the replaced frames
    assert r.matte_a[0] == pl["a0"] and len(r.matte_a[1]) == pl["i2"] and r.matte_a[1].max() == 0.0
    assert r.matte_b[0] == 0 and len(r.matte_b[1]) == pl["b_len"] and r.matte_b[1].max() == 0.0
    p2 = piece(n_a=40, n_b=40, cuts=[{"after": "s1", "cut": "orbit:push", "params": SMALL}])
    r2 = PC.apply_cut(p2, PC.specs(p2)[0], {"s1": clip(1, 40), "s2": clip(2, 40)}, depths={"s1": depth()[None], "s2": depth()[None]})
    assert r2.matte_a is None


def test_patch_matte_keeps_type_and_resizes():
    m = np.zeros((10, 20, 12, 3), np.float32)
    out = PC.patch_matte(m, (4, np.ones((3, 40, 24), np.float32)))
    assert out.shape == m.shape and out[4:7].min() == 1.0 and out[3].max() == 0.0 and out[7].max() == 0.0 and m.max() == 0.0
    u = PC.patch_matte(np.zeros((6, 8, 8), np.uint8), (1, np.full((2, 8, 8), 0.5, np.float32)))
    assert u.dtype == np.uint8 and u[1].min() == 127 and u[0].max() == 0
    assert PC.patch_matte(None, (0, np.ones((1, 2, 2)))) is None and PC.patch_matte(m, None) is m


def test_unknown_dolly_param_raises():
    p = piece(cuts=[{"after": "s1", "cut": "depth_dolly:down", "params": {"chunks": 5}}])
    with pytest.raises(KeyError):
        PC.plan_cut(p, PC.specs(p)[0])


# ----------------------------------------------------------------------------------------------- graph insertion (opt-in)
def test_insert_pivot_cut_sits_between_the_footage_and_compose_in_every_graph_variant():
    from tools.app import build_graphs as BG
    for draft, design, auto in ((False, True, True), (False, False, False)):
        g = BG.build_graph(draft, "moss", design, auto)
        src = g["compose"]["inputs"]["clips"]
        g2 = BG.insert_pivot_cut(g)
        assert g["compose"]["inputs"]["clips"] == src and "pivot" not in g                  # the input graph is untouched
        pv = g2["pivot"]
        assert pv["class_type"] == "MographPivotCut" and pv["inputs"]["clips"] == src and pv["inputs"]["piece"] == g["compose"]["inputs"]["piece"]
        assert g2["compose"]["inputs"]["clips"] == ["pivot", 0]
        for k in range(1, BG.SLOTS + 1):
            for kind in ("matte", "depth"):
                assert pv["inputs"].get(f"{kind}_{k}") == g["compose"]["inputs"].get(f"{kind}_{k}")
        assert not BG.check_reachability(g2)
        with pytest.raises(ValueError):
            BG.insert_pivot_cut(g2)
    layered = None
    try:
        from tools.app import layered_shots as LS
        g = BG.build_graph(False, "moss", True, True)
        layered, _ = LS.convert(g, None)
    except Exception:       # noqa: BLE001  the layered twin needs the full LTX template files; the plain graphs above are the structural test
        layered = None
    if layered is not None:
        g3 = BG.insert_pivot_cut(layered)
        assert g3["pivot"]["inputs"]["clips"] == layered["compose"]["inputs"]["clips"]       # after the VLM reading check (compose's own source)


# --------------------------------------------------------------------------------------------- depth reaches the 3D cuts (2026-10-30: they were fed the 16x16 placeholder)
def test_shot_needs_depth_for_both_sides_of_a_3d_cut():
    from tools.fusion import recipe_eval as RE
    for kind in ("depth_pivot", "orbit:pullback", "depth_dolly:left"):
        p = piece(cuts=[{"after": "s1", "cut": kind}])
        assert RE.shot_needs(p, "s1")["depth"] is True and RE.shot_needs(p, "s2")["depth"] is True, kind      # A and B
        assert RE.shot_needs(p, "s3")["depth"] is False, kind                                                  # a shot no cut names
    hard = piece(cuts=[{"after": "s1", "cut": "hard"}])
    assert all(RE.shot_needs(hard, s)["depth"] is False for s in ("s1", "s2", "s3"))


def test_placeholder_depth_guard_and_label():
    z16 = np.zeros((1, 16, 16, 3), np.float32)                         # footage._placeholder()
    assert PC.is_placeholder_depth(z16) and PC.usable_depth(z16) is None
    assert PC.usable_depth(np.zeros((16, 16), np.float32)) is None and PC.usable_depth(np.zeros((8, 16, 16), np.float32)) is None
    real = np.zeros((1, 64, 64, 3), np.float32)
    assert PC.usable_depth(real) is real and PC.usable_depth(None) is None and not PC.is_placeholder_depth(None)
    assert PC.label_depth({"depth": ["depth-anything-v2-small", "given"]}, True, False)["depth"] == ["fallback", "given"]
    assert PC.label_depth({}, True, False)["depth"] == ["fallback", "given"]
    assert PC.label_depth({"depth": ["given", "given"]}, True, True)["depth"] == ["placeholder", "placeholder"]


# --------------------------------------------------------------------------------------------- the gathers get the PLAN piece (no compose.cuts yet) - ship #7 box renders kept the 16x16 depth
R02_CUTS = [{"after": "s1", "cut": "depth_pivot", "style": "chunks"}, {"after": "s2", "cut": "pan:left", "word": 1}, {"after": "s3", "cut": "depth_pivot", "style": "chunks"},
            {"after": "s4", "cut": "depth_pivot", "style": "chunks"}, {"after": "s5", "cut": "hard"}]


def r02_piece(designed=True, kinds=("pan:left", "depth_pivot")):
    """the shape of 16_app/bench/r19_explore_box2/pull/r02 piece.json: 6 shots; `designed` = with the model's compose.cuts, else the plan piece the gathers receive (no `cuts` key)."""
    comp = {"crossfade": 0}
    if designed:
        comp["cuts"] = copy.deepcopy(R02_CUTS)
    return {"fps": 24, "shots": [{"id": f"s{k}", "frames": 72} for k in range(1, 7)], "compose": comp,
            "plan": {"piece_system": {"cuts": {"kinds": list(kinds)}}}}


def test_gather_sees_the_plan_piece_depth_for_every_shot_when_3d_cuts_may_come():
    from tools.fusion import recipe_eval as RE
    d = r02_piece(designed=True)
    assert {s: RE.shot_needs(d, s)["depth"] for s in ("s1", "s2", "s3", "s4", "s5", "s6")} == dict(s1=True, s2=True, s3=True, s4=True, s5=True, s6=False)      # designed: only the shots next to a 3D cut
    plan = r02_piece(designed=False)
    assert "cuts" not in plan["compose"] and PC.specs(plan) == []                    # what the gather saw: no cuts, so ship #7's specs()-only rule gave no depth
    assert all(RE.shot_needs(plan, f"s{k}")["depth"] is True for k in range(1, 7))   # every shot of a piece whose drawn kinds allow a 3D cut
    assert all(RE.shot_needs(r02_piece(False, kinds=()), f"s{k}")["depth"] for k in (1, 6))          # no drawn kinds = the default pivot may be planned
    assert all(RE.shot_needs(r02_piece(False, kinds=("depth_lines",)), f"s{k}")["depth"] for k in (1, 6))
    for kinds in (("pan:left", "hard"), ("circle_grid",), ("window_cut",)):
        assert not any(RE.shot_needs(r02_piece(False, kinds=kinds), f"s{k}")["depth"] for k in range(1, 7)), kinds        # no depth-reading cut drawn: placeholder stays
    one = r02_piece(False)
    one["shots"] = one["shots"][:1]
    assert RE.shot_needs(one, "s1")["depth"] is False                                  # one shot has no boundary
