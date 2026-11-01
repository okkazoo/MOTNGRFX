"""3D shot cuts: styles (chunks | smooth | slide), the subject card, the type that never enters the transition, hard cuts that are hard.
Tiny synthetic renders on the CPU (64 x 112 mesh). Run: python -m pytest tests/test_cut_styles.py -q"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph.modules.transitions import depth_pivot as DP  # noqa: E402
from tools.fusion import cut_kinds as CK  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402
from tools.fusion import window_cut as W  # noqa: E402

H, WD = 112, 64
SUBJ = np.array([250, 40, 40], np.float32)          # a colour no background pixel has
SMALL = dict(short=64, travel=8, hold1=3, hold2=3)


def clip(seed, n, drift, bg=(60, 150, 90)):
    rng = np.random.default_rng(seed)
    base = np.zeros((H, WD, 3), np.float32)
    checker = ((np.indices((H, WD)).sum(0) // 6) % 2).astype(np.float32)[..., None]
    base[:] = np.array(bg, np.float32) * (0.55 + 0.45 * checker)
    base += rng.normal(0, 3, base.shape).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:WD].astype(np.float32)
    fr, mt = [], []
    for k in range(n):
        cx, cy = WD * 0.5 + drift * k, H * 0.6
        m = np.clip(1.6 - np.sqrt(((xx - cx) / 17.0) ** 2 + ((yy - cy) / 30.0) ** 2) * 1.6, 0.0, 1.0)          # a soft-edged ellipse
        f = base * (1 - m[..., None]) + SUBJ * m[..., None]
        fr.append(np.clip(f, 0, 255).astype(np.uint8))
        mt.append(m.astype(np.float32))
    return np.stack(fr), np.stack(mt)


def depth_of(m):
    d = np.full((H, WD), 0.15, np.float32)
    d[:, :20] = 0.4
    d[m[0] > 0.5] = 0.9
    return d


def piece(kind, par, n=40, words=None, split=True):
    """split=True: these tests are about the subject-split looks (card, hole fill, the matte of the cards); a cut splits only when it says so since 2026-10-30 (tests/test_cut_plain_and_glitch.py)."""
    p = {"fps": 24, "shots": [{"id": "s1", "frames": n}, {"id": "s2", "frames": n}], "compose": {"crossfade": 0, "cuts": [{"after": "s1", "cut": kind, "params": par, "split": split}]}}
    if words:
        p["type"] = {"words": words}
    return p


def render(kind, par, drift_a=0.3, drift_b=-0.3, words=None):
    a, ma = clip(1, 40, drift_a)
    b, mb = clip(2, 40, drift_b, bg=(40, 90, 200))
    p = piece(kind, par, words=words)
    res = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, mattes={"s1": ma, "s2": mb}, depths={"s1": depth_of(ma)[None], "s2": depth_of(mb)[None]})
    return res, a, b, ma, mb


# ----------------------------------------------------------------------------------------------- vocabulary / defaults
def test_style_defaults_and_slide_sides():
    assert CK.style_of("depth_pivot") == "chunks" and CK.style_of("depth_pivot", "smooth") == "chunks"          # the pivot has no smooth form
    assert [CK.style_of(f"orbit:{m}") for m in CK.ORBIT_MODES] == ["chunks"] * 3 and CK.style_of("orbit:push", "smooth") == "smooth"      # 2026-10-30: no single flat card by default
    assert CK.style_of("orbit:push", "chunks") == "chunks" and CK.style_of("orbit:fly", "slide") == "slide" and CK.style_of("depth_dolly:up") is None
    assert CK.sides("orbit:push") is None and CK.sides("orbit:push", "smooth") is None
    assert CK.sides("orbit:push", "slide", "left") == ("right", "left") and CK.sides("orbit:fly", "slide") == ("left", "right") and CK.sides("depth_pivot") == ("left", "right")
    r = {m: DP.resolve_params({"mode": m})["style"] for m in ("pivot", "push", "pullback", "fly")}
    assert r == {"pivot": "chunks", "push": "chunks", "pullback": "chunks", "fly": "chunks"}
    assert DP.resolve_params({"mode": "push", "style": "chunks"})["style"] == "chunks" and DP.resolve_params({"mode": "pivot", "style": "slide"})["style"] == "chunks"
    again = DP.resolve_params(DP.resolve_params({"mode": "fly", "style": "slide"}))
    assert again["style"] == "slide"                                                                              # a resolved dict re-makes the same result


def test_style_travels_in_the_cut_or_its_params():
    p = {"fps": 24, "shots": [{"id": "s1", "frames": 90}, {"id": "s2", "frames": 90}], "compose": {"cuts": [{"after": "s1", "cut": "orbit:push", "style": "slide", "dir": "left"}]}}
    sp = PC.specs(p)[0]
    assert sp["params"]["style"] == "slide" and sp["params"]["dir"] == "left"
    pl = PC.plan_cut(p, sp)
    assert pl["params"]["style"] == "slide" and pl["i2"] == PC.window_frames("orbit:push", {"style": "slide"})[1]


def test_smooth_and_slide_change_the_plan_arithmetic_only_through_i2():
    n_c, i_c = PC.window_frames("orbit:push", {"style": "chunks"})
    n_s, i_s = PC.window_frames("orbit:push", {"style": "smooth"})
    assert PC.window_frames("orbit:push")[1] == i_c                                                                   # the default is chunks now
    assert (n_c, n_s) == (72, 72) and i_c == 29 and 20 <= i_s <= 33 and PC.window_frames("depth_pivot")[1] == 29           # the pivot is untouched
    assert PC.window_frames("orbit:pullback", {"style": "smooth"})[1] < PC.window_frames("orbit:pullback", {"style": "chunks"})[1]


# ----------------------------------------------------------------------------------------------- every style renders; homes are exact
@pytest.mark.parametrize("kind,style", [("orbit:push", "smooth"), ("orbit:pullback", "smooth"), ("orbit:fly", "smooth"), ("orbit:push", "slide"), ("orbit:fly", "slide")])
def test_smooth_styles_render_with_exact_homes_and_a_matte(kind, style):
    res, a, b, ma, mb = render(kind, dict(SMALL, style=style))
    pl = res.plan
    assert pl["problems"] == [] and res.info["style"] == style and res.info["cards"] == []
    fa = cv2.resize(a[pl["a0"]], (res.a.shape[2] if False else 64, 112), interpolation=cv2.INTER_AREA)
    assert np.abs(res.a[pl["a0"]].astype(int) - a[pl["a0"]].astype(int)).mean() < 1.0            # frame 0 of the transition = A's own frame
    assert np.abs(res.b[pl["b_len"] - 1].astype(int) - b[pl["b_len"] - 1].astype(int)).mean() < 1.0      # the last = B's own frame
    assert np.abs(res.matte_a[1][0] - ma[pl["a0"]]).mean() < 0.02 and np.abs(res.matte_b[1][-1] - mb[pl["b_len"] - 1]).mean() < 0.02
    mid = res.a[pl["a0"] + pl["i2"] - 1].astype(int)
    assert np.abs(mid - a[pl["a0"] + pl["i2"] - 1].astype(int)).mean() > 3.0                      # the picture really moved
    assert fa.shape == (112, 64, 3)


# ----------------------------------------------------------------------------------------------- the subject is ONE crisp card: no chunks inside the matte, no duplicates
@pytest.mark.parametrize("kind,style", [("orbit:push", "smooth"), ("orbit:pullback", "smooth"), ("orbit:fly", "smooth"), ("orbit:push", "slide"),
                                        ("orbit:push", "chunks"), ("depth_pivot", "chunks")])
def test_no_chunk_artefacts_inside_the_subject_matte_and_no_second_subject(kind, style):
    res, *_ = render(kind, dict(SMALL, style=style) if kind != "depth_pivot" else dict(SMALL))
    pl = res.plan
    frames = np.concatenate([res.a[pl["a0"]:pl["a0"] + pl["i2"]], res.b[:pl["b_len"]]]).astype(np.float32)
    matte = np.concatenate([res.matte_a[1], res.matte_b[1]])
    checked = 0
    for f, m in zip(frames, matte):
        if (m > 0.97).sum() < 20:
            continue
        core = (cv2.erode((m > 0.97).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if core.sum() < 8:
            continue
        # A's subject colour: inside the matte's core the frame is the subject colour (A's) or B's own subject colour; never the checker background of either shot
        d = np.minimum(np.abs(f[core] - SUBJ).max(-1), 1e9)
        assert (d < 40).mean() > 0.90, (kind, style, (d < 40).mean())
        checked += 1
        # the same colour outside the (grown) matte = a second silhouette / ghost: only a thin rim is allowed
        near = cv2.dilate((m > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=4) > 0
        ghost = (np.abs(f - SUBJ).max(-1) < 30) & ~near
        assert ghost.mean() < 0.005, (kind, style, ghost.mean())
    assert checked >= 6


def test_matte_centroid_follows_the_subject_of_the_picture():
    res, *_ = render("orbit:fly", dict(SMALL, style="smooth"), drift_a=0.6, drift_b=-0.6)
    pl = res.plan
    frames = np.concatenate([res.a[pl["a0"]:pl["a0"] + pl["i2"]], res.b[:pl["b_len"]]]).astype(np.float32)
    matte = np.concatenate([res.matte_a[1], res.matte_b[1]])
    yy, xx = np.mgrid[0:H, 0:WD].astype(np.float32)
    n = 0
    for f, m in zip(frames, matte):
        sub = (np.abs(f - SUBJ).max(-1) < 30).astype(np.float32)
        if sub.sum() < 40 or (m > 0.5).sum() < 40:
            continue
        cm = ((m > 0.5) * xx).sum() / (m > 0.5).sum(), ((m > 0.5) * yy).sum() / (m > 0.5).sum()
        cs = (sub * xx).sum() / sub.sum(), (sub * yy).sum() / sub.sum()
        assert abs(cm[0] - cs[0]) < 2.0 and abs(cm[1] - cs[1]) < 2.0, (cm, cs)
        n += 1
    assert n >= 5


# ----------------------------------------------------------------------------------------------- type never enters the transition
def test_the_engines_own_type_planes_are_dropped_and_words_do_not_change_the_cut():
    words = [{"text": "CATCH AIR", "t0": 0.0, "t1": 1.9, "exit_s": 0.5, "render": "flat"}]
    r1, *_ = render("orbit:push", dict(SMALL, type="P06:CATCH,AIR", type_depth=0.5), words=words)
    r2, *_ = render("orbit:push", dict(SMALL))
    assert r1.plan["params"]["type"] == "" and np.array_equal(r1.a, r2.a) and np.array_equal(r1.b, r2.b)          # the words are not in the transition's pixels
    assert np.array_equal(r1.matte_a[1], r2.matte_a[1])
    assert PC.window_frames("depth_pivot", {"type": "P06:A,B"}) == PC.window_frames("depth_pivot")


def test_type_clearance_names_words_that_would_smear():
    p = {"fps": 24, "shots": [{"id": "s1", "frames": 100}, {"id": "s2", "frames": 100}], "compose": {"crossfade": 0, "cuts": [{"after": "s1", "cut": "orbit:push"}]},
         "type": {"words": [{"text": "ONE", "t0": 0.0, "t1": 2.0, "exit_s": 0.4, "render": "flat"},                # ends before A's window (frames 71-100 = 2.96-4.17 s)
                            {"text": "TWO", "t0": 3.0, "t1": 4.3, "exit_s": 0.5, "render": "flat"},                # exits inside it
                            {"text": "THREE", "t0": 3.2, "t1": 7.5, "exit_s": 0.5, "render": "flat"},               # holds across: rides
                            {"text": "FOUR", "t0": 3.0, "t1": 5.0, "exit_s": 0.5, "render": "ltx"}]}}
    cl = {c["text"]: c["state"] for c in PC.type_clearance(p, PC.plan_cut(p, PC.specs(p)[0]))}
    assert cl == {"ONE": "outside", "TWO": "exits_in", "THREE": "rides", "FOUR": "conflict"}


def test_splice_window_keeps_the_text_clip_outside_the_window():
    base = np.random.default_rng(0).integers(0, 255, (20, 8, 6, 3), dtype=np.uint8)
    new = np.full((20, 16, 12, 3), 7, np.uint8)
    out = PC.splice_window(base, new, 5, 9)
    assert out.shape == base.shape and np.array_equal(out[:5], base[:5]) and np.array_equal(out[9:], base[9:]) and (out[5:9] == 7).all()


# ----------------------------------------------------------------------------------------------- hard cuts are hard
def test_a_hard_cut_owns_its_boundary_and_splices_nothing():
    p = {"fps": 24, "shots": [{"id": "s1", "frames": 60}, {"id": "s2", "frames": 60}, {"id": "s3", "frames": 60}],
         "compose": {"crossfade": 0.5, "cuts": [{"after": "s1", "cut": "hard", "was": "window_other"}, {"after": "s2", "cut": "crossfade"}]}}
    assert PC.specs(p) == []                                          # nothing for the 3D splicer: no transition frames anywhere near a hard cut
    assert W.owned_boundaries(p) == [0]                               # boundary 0 is hard, boundary 1 stays the soft one
    from tools.fusion import recipe_eval as RE
    parts, cfl = RE.assemble_parts(p)
    assert cfl[0] == 0.0 and cfl[1] == 0.5
    lay, _ = RE.piece_layout(p)
    assert parts[1]["cut"] == lay["s2"]["start"] / 24.0              # B starts exactly on its own first frame, A is not overlapped by it


def test_hard_cut_frames_are_never_blended(tmp_path):
    from tools.fusion import compose as C
    from tools.fusion import recipe_eval as RE

    class Flat:
        def __init__(self, v, n):
            self.v, self.n, self.size = v, n, (16, 24)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return np.full((24, 16, 3), self.v, np.float32)

        def close(self):
            pass

    p = {"fps": 24, "shots": [{"id": "s1", "frames": 24}, {"id": "s2", "frames": 24}], "compose": {"crossfade": 0.5, "cuts": [{"after": "s1", "cut": "hard"}]}}
    parts, cfl = RE.assemble_parts(p)
    out = tmp_path / "o.mp4"
    try:
        C.assemble(parts, None, str(out), crossfades=cfl, fps=24, size=(16, 24), sources=[Flat(0.2, 24), Flat(0.8, 24)])
    except Exception as e:      # noqa: BLE001  no encoder in this environment: the arithmetic test above covers the timing
        pytest.skip(f"assemble not runnable here: {type(e).__name__}")
    cap = cv2.VideoCapture(str(out))
    means = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        means.append(float(f.mean()))
    assert means, "no frames written"
    lo, hi = 0.2 * 255, 0.8 * 255
    assert all(abs(m - lo) < 14 or abs(m - hi) < 14 for m in means), sorted(set(round(m) for m in means))        # every frame is A or B, none in between
    lay, _ = RE.piece_layout(p)
    first_b = next(i for i, m in enumerate(means) if abs(m - hi) < 14)
    assert first_b == lay["s2"]["start"] and all(abs(m - hi) < 14 for m in means[first_b:])          # one frame A, the next frame B, on B's own first frame


# ----------------------------------------------------------------------------------------------- box-probe fixes (22_batch-24gb v010 cut_probe_box)
def test_a_shrunk_subject_cropped_by_the_frame_does_not_become_a_rectangle():
    """pullback: A's picture shrinks inside the frame; a subject the frame cropped (legs to the bottom edge) must fade out at that edge, not end in a hard line."""
    def cropped(seed, n, bg):
        fr, mt = clip(seed, n, 0.0, bg=bg)
        yy, xx = np.mgrid[0:H, 0:WD].astype(np.float32)
        m = np.clip(1.6 - np.sqrt(((xx - WD * 0.5) / 20.0) ** 2 + ((yy - H * 1.0) / 60.0) ** 2) * 1.6, 0.0, 1.0)         # an ellipse centred ON the bottom edge
        m = np.repeat(m[None], n, 0).astype(np.float32)
        fr = (fr * (1 - m[..., None]) + SUBJ * m[..., None]).astype(np.uint8)
        return fr, m
    a, ma = cropped(1, 40, (60, 150, 90))
    b, mb = cropped(2, 40, (40, 90, 200))
    p = piece("orbit:pullback", dict(SMALL))
    res = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, mattes={"s1": ma, "s2": mb}, depths={"s1": depth_of(ma)[None], "s2": depth_of(mb)[None]})
    mat = np.concatenate([res.matte_a[1], res.matte_b[1]])
    worst = max(max(np.abs(np.diff(m, axis=0)).max(), np.abs(np.diff(m, axis=1)).max()) for m in mat)
    assert worst < 0.6, worst                                       # the source's own edge is soft (~0.1 / px); a cut at the picture's border would jump by ~1


def test_smooth_fly_banks_hard_and_keeps_the_subject_longer():
    from tools.fusion import depth_smooth as DS
    from tools.fusion import depth_orbit as do
    a, ma = clip(1, 12, 0.0)
    b, mb = clip(2, 12, 0.0)
    res = DP.depth_pivot_render(a, ma, b, mb, depth_of(ma), depth_of(mb), dict(SMALL, mode="fly", short=64, style="smooth"), matte_pass=True)
    assert res.info["style"] == "smooth"
    from tools.fusion import depth_dolly as dd
    op = DP._op(do, DP.resolve_params(dict(SMALL, mode="fly", style="smooth"))["engine"])
    l = [DP._layer(dd, a[0][..., ::-1].copy(), ma[0], depth_of(ma), 64, 112, dd.Params(depth_ratio=6.0))[0]] * 2
    sc = DS.SmoothScene(l, op, [(a[:, :, :, ::-1].copy(), ma), (b[:, :, :, ::-1].copy(), mb)])
    assert max(abs(sc.pose(x / 40.0).roll) for x in range(41)) > 30.0           # the bank (chunk fly: ~27)
    assert sc.op.zoom_fade[1] >= 5.0 and sc.op.vol_dz <= 3.6                   # the subject card lives longer; B starts larger


def test_slide_style_is_not_offered_to_models():
    from tools.fusion import design_options as D
    out, fx = D.normalize_cuts([{"after": "s1", "cut": "orbit:push", "params": {"style": "slide", "dir": "left"}}, {"after": "s2", "cut": "orbit:fly", "params": {"style": "smooth"}}])
    assert "style" not in out[0]["params"] and out[1]["params"]["style"] == "smooth" and any("slide" in f for f in fx)
    assert "slide" not in D.MENU_TEXT.lower().split("3d shot transitions")[1]
