"""Text-as-window design choices (tools/fusion/window_cut.py compose op + two-shot refs, tools/fusion/design_options.py schema / menu / validator,
blocks window_other / window_cut / window_swap). CPU, synthetic frames only (one optional real-frame check of P10's knockout when its footage exists).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_window_cut.py -q"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import compose as C  # noqa: E402
from tools.fusion import design_options as D  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import window_cut as W  # noqa: E402

H, Wd = 96, 64
RED = np.array([0.8, 0.2, 0.2], np.float32)
CYAN = np.array([0.2, 0.8, 0.9], np.float32)
yy, xx = np.mgrid[0:H, 0:Wd]
LETTERS = ((abs(yy - 48) < 10) & (abs(xx - 32) < 20)).astype(np.float32)
SUBJ = ((abs(yy - 20) < 8) & (abs(xx - 16) < 6)).astype(np.float32)          # a subject blob away from the letters
CENTRE, CORNER, SUBJ_PX = (48, 32), (90, 4), (20, 16)


@pytest.fixture(autouse=True)
def _restore_backend():
    prev = C.B
    yield
    C.B = prev


def flat(c):
    return np.broadcast_to(c, (H, Wd, 3)).astype(np.float32).copy()


A, B = flat(RED), flat(CYAN)
T = dict(t_open=0.5, t_cut=2.0, t_end=3.0)


def px(img, p):
    return np.asarray(img[p[0], p[1]])


def is_(img, p, c, tol=1e-4):
    return bool(np.allclose(px(img, p), c, atol=tol))


# ----------------------------------------------------------------------------------------------- phases
def test_swap_phases():
    f = lambda t, **k: W.window_cut_phase(t, "swap", exit="fade", exit_s=0.5, **T, **k)      # noqa: E731
    assert f(0.2)["name"] == "pre" and f(0.2)["inside"] is None and f(0.2)["bg"] == "a"
    p1 = f(1.99)
    assert (p1["name"], p1["bg"], p1["inside"]) == ("w1", "a", "b")                         # phase 1: B in the letters, A outside
    p2 = f(2.0)
    assert (p2["name"], p2["bg"], p2["inside"]) == ("w2", "b", "a")                         # on the cut beat: swap
    assert f(3.25)["name"] == "exit" and 0 < f(3.25)["exit_k"] < 1 and f(3.25)["bg"] == "b"
    assert f(3.6)["name"] == "post" and f(3.6)["inside"] is None and f(3.6)["bg"] == "b"     # phase 3: the letters left, B remains


def test_cut_and_other_phases():
    c = lambda t: W.window_cut_phase(t, "cut", t_open=0.5, t_cut=2.0, grow_s=0.7, exit="grow")      # noqa: E731
    assert c(1.0)["name"] == "window" and c(1.0)["grow"] == 0.0 and c(1.0)["inside"] == "b"
    g = [c(t)["grow"] for t in (1.3, 1.5, 1.7, 1.9, 1.999)]
    assert g[0] < 1e-3 and all(x < y for x, y in zip(g, g[1:])) and g[-1] > 0.99
    assert c(2.0)["name"] == "post" and c(2.0)["bg"] == "b" and c(2.0)["inside"] is None
    assert W.window_cut_phase(1.9, "cut", t_open=0.5, t_cut=2.0, exit="io")["grow"] == 0.0          # io: letters stay, hard cut
    o = lambda t: W.window_cut_phase(t, "other", t_open=1.0, t_end=3.0, exit="fade", exit_s=0.4)      # noqa: E731
    assert o(0.5)["inside"] is None and o(2.0)["inside"] == "b" and o(2.0)["bg"] == "a" and o(3.5)["inside"] is None
    with pytest.raises(ValueError):
        W.window_cut_phase(1.0, "cut", t_open=0.0)


# ----------------------------------------------------------------------------------------------- the op on synthetic frames
@pytest.mark.parametrize("be", ["numpy"] + (["torch"] if C.torch is not None else []))
def test_swap_inside_outside_flip_exactly_on_the_beat_no_crossfade(be):
    C.set_backend(be)
    kw = dict(mode="swap", exit="fade", exit_s=0.5, **T)
    before = W.window_cut(LETTERS, A, B, t=2.0 - 1 / 24, **kw)
    after = W.window_cut(LETTERS, A, B, t=2.0, **kw)
    assert is_(before, CENTRE, CYAN) and is_(before, CORNER, RED)           # phase 1: B inside, A outside
    assert is_(after, CENTRE, RED) and is_(after, CORNER, CYAN)             # swapped
    for img in (before, after):                                             # only the two pure pictures: nothing blended in between
        flat_px = img.reshape(-1, 3)
        near = [np.all(np.abs(flat_px - c) < 1e-3, axis=1) for c in (RED, CYAN)]
        assert (near[0] | near[1]).mean() > 0.97                            # only the antialiased letter edge is mixed
    late = W.window_cut(LETTERS, A, B, t=3.6, **kw)
    assert np.allclose(late, B, atol=1e-5)                                  # phase 3: the letters left, B only
    pre = W.window_cut(LETTERS, A, B, t=0.1, **kw)
    assert np.allclose(pre, A, atol=1e-5)


def test_subject_of_the_outside_shot_stands_over_both():
    mA, mB = SUBJ, np.zeros_like(SUBJ)
    kw = dict(mode="swap", exit="io", matte_a=mA, matte_b=mB, **T)
    p1 = W.window_cut(LETTERS, A, B, t=1.0, **kw)
    assert is_(p1, SUBJ_PX, RED)                                            # A is the plate: its subject (red) steps out over the picture
    kw2 = dict(mode="swap", exit="io", matte_a=np.zeros_like(SUBJ), matte_b=SUBJ, **T)
    p2 = W.window_cut(LETTERS, A, B, t=2.5, **kw2)
    assert is_(p2, SUBJ_PX, CYAN)                                           # after the swap B is the plate: its subject is on top
    none = W.window_cut(LETTERS, A, B, t=1.0, subject="none", **{**kw, "mode": "swap"})
    assert is_(none, SUBJ_PX, RED)                                          # (the blob lies on plain A either way; the matte matters where it overlaps B)
    over = np.zeros_like(SUBJ)
    over[40:56, 24:40] = 1.0                                                # a matte over the letters region
    q = W.window_cut(LETTERS, A, B, t=1.0, mode="swap", exit="io", matte_a=over, matte_b=None, **T)
    assert is_(q, CENTRE, RED)                                              # the subject covers the window with A's picture
    q2 = W.window_cut(LETTERS, A, B, t=1.0, mode="swap", exit="io", matte_a=over, subject="none", **T)
    assert is_(q2, CENTRE, CYAN)


@pytest.mark.parametrize("be", ["numpy"] + (["torch"] if C.torch is not None else []))
def test_window_cut_grows_until_b_fills_the_frame(be):
    C.set_backend(be)
    kw = dict(mode="cut", t_open=0.5, t_cut=2.0, grow_s=0.7, exit="grow", matte_a=SUBJ)
    share = []
    for t in (1.0, 1.4, 1.6, 1.8, 1.95, 1.9999):
        o = W.window_cut(LETTERS, A, B, t=t, **kw)
        share.append(float((np.abs(o - B).sum(-1) < 0.02).mean()))
    assert abs(share[0] - LETTERS.mean()) < 0.02 and all(a <= b + 1e-6 for a, b in zip(share, share[1:]))
    assert share[-1] > 0.999                                                # the last frame before the cut is B (the subject faded out with the window)
    cut = W.window_cut(LETTERS, A, B, t=2.0, **kw)
    assert np.allclose(cut, B, atol=1e-5)                                   # the cut itself: B alone, no crossfade


def test_exit_fade_and_shrink_remove_the_letters():
    kw = dict(mode="swap", **T)
    mid = W.window_cut(LETTERS, A, B, t=3.25, exit="fade", exit_s=0.5, **{k: v for k, v in kw.items() if k != "mode"}, mode="swap")
    assert not is_(mid, CENTRE, RED, 0.05) and not is_(mid, CENTRE, CYAN, 0.05)       # half way: neither
    sh = [float((np.abs(W.window_cut(LETTERS, A, B, t=t, mode="swap", exit="shrink", exit_s=0.5, **T) - RED).sum(-1) < 0.02).mean()) for t in (3.0, 3.2, 3.4, 3.6)]
    assert sh[0] > 0.1 and sh[-1] == 0.0 and all(a >= b for a, b in zip(sh, sh[1:]))   # the letters erode away


def test_flat_ground_flip_is_the_degenerate_case_of_p10():
    """P10: knockout(plate_rgb = flat ground that flips on a step timeline, footage in the letters, fg_matte = the matte)."""
    rng = np.random.default_rng(1)
    foot = rng.random((H, Wd, 3)).astype(np.float32)
    ground = [0.88, 0.88, 0.87]
    ref = C.knockout(ground, LETTERS, foot, fg_matte=SUBJ)
    got = W.window_cut(LETTERS, ground, foot, t=1.0, mode="other", t_open=0.0, fg_matte=SUBJ)
    assert np.array_equal(ref, got)
    ref2 = C.knockout([0.03, 0.03, 0.028], LETTERS, foot, fg_matte=SUBJ, min_contrast=0.22)
    got2 = W.window_cut(LETTERS, [0.03, 0.03, 0.028], foot, t=1.0, mode="other", t_open=0.0, fg_matte=SUBJ, min_contrast=0.22)
    assert np.array_equal(ref2, got2)


P10 = ROOT / "15_fusion" / "v002" / "ingredients" / "P10"


@pytest.mark.skipif(not P10.is_dir(), reason="P10 footage not on this machine")
def test_p10_flip_rebuilt_through_the_new_op_gives_the_same_frames():
    mp4 = next(P10.glob("*_s2_1_*.mp4"))
    cap = cv2.VideoCapture(str(mp4))
    for i in range(30):
        ok, fr = cap.read()
    cap.release()
    foot = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (270, 480)).astype(np.float32) / 255.0
    foot = C.grade(foot, gain=1.08, sat=0.0)
    m = np.zeros((480, 270), np.float32)
    m[120:420, 90:200] = 1.0
    t_a = np.zeros_like(m)
    t_a[200:300, 40:230] = 1.0
    for g in ([0.88, 0.88, 0.87], [0.03, 0.03, 0.028]):
        ref = C.knockout(g, t_a, foot, fg_matte=m)
        got = W.window_cut(t_a, g, foot, t=9.0, mode="other", t_open=0.0, fg_matte=m)
        assert np.array_equal(ref, got)


# ----------------------------------------------------------------------------------------------- two-shot refs, boundary ownership, assembler
def test_other_shot_refs():
    ids = ["s1", "s2", "s3"]
    assert W.other_shot_ref("next", "s1", ids) == ("s2", "footage") and W.other_shot_ref("prev_matte", "s2", ids) == ("s1", "matte")
    assert W.other_shot_ref("shot:s3", "s1", ids) == ("s3", "footage") and W.other_shot_ref("shot:s3:matte", "s1", ids) == ("s3", "matte")
    assert W.other_shot_ref("type_matte", "s1", ids) is None and W.other_shot_ref("s1_matte", "s1", ids) is None
    with pytest.raises(ValueError):
        W.other_shot_ref("prev", "s1", ids)
    with pytest.raises(ValueError):
        W.other_shot_ref("shot:zz", "s1", ids)


def piece(cuts=None, words=None, xf=0.0833, n=24):
    p = {"id": "T", "fps": 24, "sizes": {"compose": [Wd, H]}, "type_passes": ["matte", "line"],
         "shots": [{"id": "s1", "frames": n}, {"id": "s2", "frames": n}, {"id": "s3", "frames": n}],
         "compose": {"crossfade": [xf, xf], "default": {"ops": [{"op": "grade", "out": "image", "args": {"frame": "$footage", "gain": 1.0}}]}},
         "type": {"words": words or [{"text": "AB", "t0": 0.0, "t1": 3.0, "hit": 1.0, "release": 2.5}]}}
    if cuts:
        p["compose"]["cuts"] = cuts
    return p


def test_owned_boundary_is_a_hard_cut_in_the_assembler_inputs():
    p = piece([{"cut": "window_swap", "after": "s1", "word": 0}])
    assert W.owned_boundaries(p) == [0] and W.owned_boundaries(piece()) == []
    parts, cfl = RE.assemble_parts(p)
    lay, _ = RE.piece_layout(p)
    assert cfl[0] == 0.0 and cfl[1] == pytest.approx(0.0833)                         # only the owned boundary lost its crossfade
    assert parts[1]["cut"] == pytest.approx(lay["s2"]["start"] / 24.0)               # B starts exactly at its piece start, not at start + xf/2
    plain_parts, plain_cfl = RE.assemble_parts(piece())
    assert plain_cfl[0] == pytest.approx(0.0833) and plain_parts[1]["cut"] == pytest.approx((lay["s2"]["start"] + lay["s2"]["xf"] / 2.0) / 24.0)   # untouched
    assert W.owned_boundaries(piece([{"cut": "hard", "after": "s2"}])) == [1] and W.owned_boundaries(piece([{"cut": "crossfade", "after": "s1"}])) == []


class MemSource:
    def __init__(self, frames):
        self.f = frames

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.f[i]

    def close(self):
        pass


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_assembled_cut_has_no_blended_frame(tmp_path):
    a = [flat(RED)] * 30
    b = [flat(CYAN)] * 30
    p = piece([{"cut": "window_swap", "after": "s1", "word": 0}], n=30)
    lay, _ = RE.piece_layout(p)
    parts, cfl = RE.assemble_parts(p)
    plain = [{"src": "x", "cut": parts[0]["cut"]}, {"src": "y", "cut": parts[1]["cut"]}]
    out = tmp_path / "o.mp4"
    C.assemble(plain, None, str(out), crossfades=cfl[:1], fps=24, size=(Wd, H), sources=[MemSource(a), MemSource(b)], crf=10)
    cap = cv2.VideoCapture(str(out))
    cols = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cols.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[48, 32].astype(float) / 255.0)
    cap.release()
    k = lay["s2"]["start"]
    d_red = [np.abs(c - RED).max() for c in cols]
    d_cyan = [np.abs(c - CYAN).max() for c in cols]
    assert d_red[k - 1] < 0.06 and d_cyan[k] < 0.06                                  # the frame before is A, the frame at the cut is B
    assert all(min(r, c) < 0.06 for r, c in zip(d_red, d_cyan))                      # no frame is a mix


def test_recipe_reads_the_neighbour_shot_and_renders_both_sides(tmp_path):
    n = 24
    p = piece([{"cut": "window_swap", "after": "s1", "word": 0, "exit": "io"}], n=n)
    tdir = tmp_path / "type"
    for nm in ("matte", "line"):
        (tdir / nm).mkdir(parents=True)
        for i in range(80):
            cv2.imwrite(str(tdir / nm / f"{i:05d}.png"), (LETTERS * 255).astype(np.uint8))
    notes = W.apply_to_piece(p)
    assert notes and "window_swap s1->s2" in notes[0]
    t_cut = RE.piece_layout(p)[0]["s2"]["start"] / 24.0
    ops1 = p["compose"]["shots"]["s1"]["ops"]
    ops2 = p["compose"]["shots"]["s2"]["ops"]
    assert ops1[0]["args"]["b"] == "$next" and ops1[0]["args"]["a"] == "$footage"
    assert ops2[0]["args"]["a"] == "$prev" and ops2[0]["args"]["b"] == "$footage"
    assert ops1[0]["args"]["t_cut"] == pytest.approx(t_cut, abs=1e-3) == pytest.approx(ops2[0]["args"]["t_cut"], abs=1e-3)
    ops1[0]["args"].pop("line", None)
    ops2[0]["args"].pop("line", None)
    for o in (ops1[0], ops2[0]):
        o["args"]["t_open"], o["args"]["min_contrast"] = 0.0, None
        del o["args"]["min_contrast"]
    rec = RE.build_recipe(p, "s1", type_dir=tdir)
    assert rec["inputs"]["next"]["shot"] == "s2" and rec["inputs"]["next"]["offset"] == -RE.piece_layout(p)[0]["s2"]["start"]
    fa = np.stack([np.round(flat(RED) * 255)] * n).astype(np.uint8)
    fb = np.stack([np.round(flat(CYAN) * 255)] * n).astype(np.uint8)
    ma = np.stack([np.round(SUBJ * 255)] * n).astype(np.uint8)
    zeros = np.zeros_like(ma)
    kw = dict(type_dir=tdir, backend="numpy", threads=0)
    out1, _ = RE.run_compose(p, "s1", footage=fa, matte=ma, others={"s2": {"footage": fb, "matte": zeros}}, **kw)
    out2, _ = RE.run_compose(p, "s2", footage=fb, matte=zeros, others={"s1": {"footage": fa, "matte": ma}}, **kw)
    out1, out2 = np.asarray(out1), np.asarray(out2)
    ca, cb = np.round(RED * 255), np.round(CYAN * 255)
    k_cut = RE.piece_layout(p)[0]["s2"]["start"]
    last_a = out1[k_cut - 1]                                                         # last frame of A before the cut (the assembler drops A's overlap tail): B in the letters
    first_b = out2[0]                                                                # shot B's first frame = the cut: A in the letters, B around
    assert np.allclose(last_a[48, 32], cb, atol=1) and np.allclose(last_a[90, 4], ca, atol=1)
    assert np.allclose(first_b[48, 32], ca, atol=1) and np.allclose(first_b[90, 4], cb, atol=1)
    with pytest.raises(ValueError):
        RE.run_compose(p, "s1", footage=fa, matte=ma, **kw)                          # the neighbour's frames must be supplied


# ----------------------------------------------------------------------------------------------- schema + validator
SHOTS = [{"id": "s1", "start": 0.0, "dur": 5.0}, {"id": "s2", "start": 5.0, "dur": 5.0}, {"id": "s3", "start": 10.0, "dur": 5.0}, {"id": "s4", "start": 15.0, "dur": 5.0}]
ANA = {"downbeats": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0], "chord_changes": [{"t": 10.0, "from": "A", "to": "B"}], "cut": 20.0, "fps": 24,
       "curves": {"rms": [1.0] * 480}}


def design(font="BadeenDisplay-Regular.ttf", hf=0.95, hit=3.0, hold=3.0, cuts=None, extra_words=None):
    words = [{"text": "WAS", "font": font, "lines": ["W", "A", "S"], "hit": hit, "hold_s": hold, "placement": {"hf": hf}, "form_s": 0.7}] + (extra_words or [])
    return {"type": {"words": words}, "compose": {"cuts": cuts if cuts is not None else [{"after": "s1", "cut": "window_swap", "word": 0, "exit": "fade"}]}}


def probe_for(gap):
    a = np.full((H, Wd, 3), 0.2, np.float32)
    b = np.full((H, Wd, 3), 0.2 + gap, np.float32)
    return lambda kind, i, t: (a, b, LETTERS)


def ctx(**k):
    return dict(shots=SHOTS, ana=ANA, **k)


def test_schema_fragment_parses():
    json.loads(json.dumps(D.JSON_SCHEMA))
    cuts, fixes = D.normalize_cuts(D.EXAMPLE["cuts"])
    assert fixes == [] and [c["cut"] for c in cuts] == ["window_swap", "depth_dolly:right", "orbit:pullback"] and cuts[0]["freeze_s"] == 0.25 and cuts[1]["why"]
    fr = D.schema_fragment(["s1", "s2"])
    assert "window_source" in fr["word"] and all(k in fr["cuts"] for k in D.CUT_KINDS if D.DOLLY_ON or not k.startswith("depth_dolly")) and "s1, s2" in fr["note"]
    c, f = D.normalize_cuts([{"after": "s1", "cut": "Swap", "word": "x", "exit": "wipe", "freeze_s": 3}, {"cut": "hard"}, "bad"])
    assert c[0]["cut"] == "window_swap" and c[0]["exit"] == "fade" and c[0]["freeze_s"] == 0.5 and len(c) == 1 and len(f) >= 4
    txt = D.menu_text(SHOTS, ANA)
    assert "window_swap" in txt and "WHEN" in txt and "after s1 (into s2) at 5 s" in txt and "swap ok" in txt
    assert "too far" in D.menu_text([{"id": "a", "start": 0, "dur": 3}, {"id": "b", "start": 3.7, "dur": 3}], ANA)
    assert D.menu_text(SHOTS[:1], ANA) == ""


def test_good_swap_passes_untouched():
    d, fixes = D.validate(design(), ctx(probe=probe_for(0.4)))
    assert fixes == [] and d["compose"]["cuts"] == [{"after": "s1", "cut": "window_swap", "word": 0, "exit": "fade"}]


def test_thin_font_is_swapped_to_a_fat_liked_font():
    d, fixes = D.validate(design(font="Interlope-Regular.ttf"), ctx(probe=probe_for(0.4)))
    assert any("not a fat face" in f for f in fixes) and d["compose"]["cuts"][0]["cut"] == "window_swap"
    assert D.is_fat(d["type"]["words"][0]["font"], d["type"]["words"][0].get("axes"))[0]
    d2, f2 = D.validate(design(font="NotALikedFont.ttf"), ctx())
    assert d2["compose"]["cuts"][0]["cut"] == "hard" and any("not a liked font" in f for f in f2)


def test_too_small_cap_height_is_bumped_or_downgraded():
    d, fixes = D.validate(design(hf=0.3), ctx(probe=probe_for(0.4)))                  # 3 stacked letters in a 0.3 box = 10 % caps -> the box is made taller
    assert any("placement.hf" in f for f in fixes) and d["type"]["words"][0]["placement"]["hf"] >= round(D.WINDOW_MIN_CAP * 3, 2)
    d2, f2 = D.validate(design(), ctx(caps={0: 0.2}, probe=probe_for(0.4)))             # measured 20 %: cannot be predicted away
    assert d2["compose"]["cuts"] == [{"after": "s1", "cut": "hard"}] and any("cap height" in f and "downgraded to hard" in f for f in f2)
    one_line = design(extra_words=[])
    one_line["type"]["words"][0].pop("lines")
    one_line["type"]["words"][0]["text"] = "STILL"
    d3, f3 = D.validate(one_line, ctx(probe=probe_for(0.4)))                          # five letters on one line cannot reach 30 % caps in 9:16: the code stacks it (real glyph widths, the narrow heavy face) instead of giving the window up
    w3 = d3["type"]["words"][0]
    assert d3["compose"]["cuts"][0]["cut"] == "window_swap" and D.cap_fraction(w3)[0] >= D.WINDOW_MIN_CAP * D.CAP_TOL     # 2026-10-30: STILL is never cut across lines; the narrow heavy face reaches the relaxed window size (WINDOW_MIN_CAP 0.26)
    short = design(extra_words=[])
    short["type"]["words"][0].pop("lines")
    short["type"]["words"][0]["text"] = "RISE"
    d4, f4 = D.validate(short, ctx(probe=probe_for(0.4)))                              # a short word on one line, in the narrow heavy face, reaches the window's cap height
    w4 = d4["type"]["words"][0]
    assert d4["compose"]["cuts"][0]["cut"] == "window_swap" and "Anybody" in w4["font"] and w4["axes"]["wdth"] == 50.0 and D.cap_fraction(w4)[0] >= D.WINDOW_MIN_CAP * D.CAP_TOL
    long_ = design(extra_words=[])
    long_["type"]["words"][0].pop("lines")
    long_["type"]["words"][0]["text"] = "SEARCHING"                                      # 9 letters: not even stacked
    d4, f4 = D.validate(long_, ctx(probe=probe_for(0.4)))
    assert d4["compose"]["cuts"][0]["cut"] == "hard" and any("even stacked" in f for f in f4)
    assert D.cap_estimates({"text": "WAS", "lines": ["W", "A", "S"], "placement": {"hf": 0.95}})[0] > 0.3 * D.CAP_TOL      # P10's stacked WAS passes


def test_no_boundary_or_no_word_across_it_downgrades():
    d, f = D.validate(design(cuts=[{"after": "s4", "cut": "window_swap", "word": 0}]), ctx())
    assert "cuts" not in d["compose"] and any("no boundary" in x for x in f)
    d, f = D.validate(design(cuts=[{"after": "s9", "cut": "window_cut", "word": 0}]), ctx())
    assert "cuts" not in d["compose"] and any("no such shot" in x for x in f)
    d, f = D.validate(design(hit=0.5, hold=1.6), ctx())                                 # the hold ends long before 5 s
    assert d["compose"]["cuts"][0]["cut"] == "hard" and any("no word holds across the cut" in x for x in f)
    d, f = D.validate(design(hit=0.5, hold=1.6, extra_words=[{"text": "X", "font": "Ultra-Regular.ttf", "hit": 3.5, "hold_s": 2.5, "placement": {"hf": 0.9}}]),
                      ctx(probe=probe_for(0.4)))
    assert d["compose"]["cuts"][0]["word"] == 1 and any("word 1 used" in x for x in f)       # another word holds across the cut


def test_swap_must_land_on_the_music():
    shots = [{"id": "s1", "start": 0.0, "dur": 4.3}, {"id": "s2", "start": 4.3, "dur": 5.0}]
    d, f = D.validate(design(hit=2.0, hold=3.5), dict(shots=shots, ana=ANA, probe=probe_for(0.4)))
    assert d["compose"]["cuts"][0]["cut"] == "hard" and any("downbeat" in x and "swap must land" in x for x in f)
    d, f = D.validate(design(hit=2.0, hold=3.5, cuts=[{"after": "s1", "cut": "window_cut", "word": 0, "exit": "grow"}]),
                      dict(shots=shots, ana=ANA, probe=probe_for(0.4)))                       # only the swap needs the beat
    assert d["compose"]["cuts"][0]["cut"] == "window_cut" and f == []
    shots2 = [{"id": "s1", "start": 0.0, "dur": 9.0}, {"id": "s2", "start": 9.9, "dur": 5.0}]
    ana2 = dict(ANA, downbeats=[], chord_changes=[{"t": 9.95, "from": "A", "to": "B"}])
    d, f = D.validate(design(hit=7.0, hold=3.5), dict(shots=shots2, ana=ana2, probe=probe_for(0.4)))   # a chord change within 0.15 s counts
    assert d["compose"]["cuts"][0]["cut"] == "window_swap"


def test_low_contrast_between_the_shots_downgrades():
    d, f = D.validate(design(), ctx(probe=probe_for(0.02)))
    assert d["compose"]["cuts"][0]["cut"] == "hard" and any("too alike" in x for x in f)
    notes = []
    d, f = D.validate(design(), ctx(), notes)
    assert f == [] and any("contrast not measured" in n for n in notes)


def test_more_than_two_window_cuts_and_neighbours_are_dropped():
    words = [{"text": t, "font": "Ultra-Regular.ttf", "hit": h, "hold_s": 3.0, "placement": {"hf": 0.9}} for t, h in (("A", 8.0), ("B", 13.0), ("C", 18.0))]
    dd = design(hit=3.0, hold=3.0, extra_words=words, cuts=[{"after": "s1", "cut": "window_cut", "word": 0}, {"after": "s2", "cut": "window_cut", "word": 1},
                                                           {"after": "s3", "cut": "window_swap", "word": 2}])
    d, f = D.validate(dd, dict(shots=SHOTS + [{"id": "s5", "start": 20.0, "dur": 5.0}], ana=dict(ANA, downbeats=ANA["downbeats"] + [20.0]), probe=probe_for(0.4)))
    kinds = [c["cut"] for c in d["compose"]["cuts"]]
    assert kinds.count("window_cut") + kinds.count("window_swap") <= D.MAX_WINDOW_CUTS
    assert any("neighbouring boundary" in x or "more than 2 window cuts" in x for x in f)
    assert kinds == ["window_cut", "hard", "window_swap"] or kinds == ["window_cut", "hard", "hard"]


def test_window_source_words_follow_the_same_rules():
    w2 = [{"text": "B", "font": "Interlope-Regular.ttf", "hit": 7.0, "hold_s": 2.0, "placement": {"hf": 0.9}, "window_source": "next"}]
    dd = design(cuts=[], extra_words=w2)
    d, f = D.validate(dd, ctx(probe=probe_for(0.4)))
    assert d["type"]["words"][1]["window_source"] == "s3" and any("not a fat face" in x for x in f)
    d, f = D.validate(dd, ctx(probe=probe_for(0.01)))
    assert "window_source" not in d["type"]["words"][1] and any("too alike" in x for x in f)
    own = design(cuts=[], extra_words=[dict(w2[0], window_source="s2")])
    d, f = D.validate(own, ctx())
    assert "window_source" not in d["type"]["words"][1] and any("own shot" in x for x in f)
    last = [dict(w2[0], hit=16.0, window_source="next")]
    d, f = D.validate(design(cuts=[], extra_words=last), ctx())
    assert "window_source" not in d["type"]["words"][1] and any("no next shot" in x for x in f)
    three = [dict(w2[0], hit=h, font="Ultra-Regular.ttf", window_source="next") for h in (1.0, 6.0, 11.0)]
    d, f = D.validate(design(cuts=[], extra_words=three), ctx(probe=probe_for(0.4)))
    assert sum(1 for w in d["type"]["words"] if w.get("window_source")) <= D.MAX_WINDOW_WORDS and any("more than 2 window_source" in x for x in f)


def test_validator_is_deterministic_and_leaves_the_input_alone():
    dd = design(font="Interlope-Regular.ttf", hf=0.3)
    snap = copy.deepcopy(dd)
    a = D.validate(dd, ctx(probe=probe_for(0.4)))
    b = D.validate(dd, ctx(probe=probe_for(0.4)))
    assert a == b and dd == snap


# ----------------------------------------------------------------------------------------------- blocks + design -> piece
def test_blocks_are_registered_opt_in_window_role():
    for n in ("window_other", "window_cut", "window_swap"):
        assert n in bk.NAMES and bk.validate_block(n) == [] and bk.BLOCKS[n]["role"] == "window" and bk.BLOCKS[n]["needs"]["opt_in"]
        ctx_ = bk.piece_ctx("cut", "behind", n_lines=4)
        assert not bk.eligible(n, ctx_)[0]                                              # never picked on its own
        ctx_["opt_in"] = [n]
        assert bk.eligible(n, ctx_)[0]
        assert not bk.eligible(n, dict(ctx_, shots="single", n_shots=1))[0]
    assert bk.WINDOW_MODULES == ("depth_tunnel", "fly_into")                            # untouched


def test_design_to_piece_builds_ops_and_freeze():
    p = piece()
    d = {"type": {"words": [{"text": "AB", "font": "Ultra-Regular.ttf", "window_source": "s3"}]},
         "compose": {"cuts": [{"after": "s1", "cut": "window_swap", "word": 0, "exit": "shrink", "freeze_s": 0.25}]}}
    notes = D.to_piece(p, d)
    assert len(notes) == 2 and p["compose"]["holds"] == [[pytest.approx(RE.piece_layout(p)[0]["s2"]["start"] / 24.0, abs=1e-3), 0.25]]
    assert p["compose"]["cuts"][0]["cut"] == "window_swap" and p["type"]["words"][0]["font"] == "Ultra-Regular.ttf"
    assert p["compose"]["shots"]["s1"]["ops"][0]["args"]["exit"] == "shrink"
    assert p["compose"]["shots"]["s1"]["ops"][-1]["op"] == "grade"                         # the shot's closing ops are kept after the window op
    assert D.to_piece(piece(), {"type": {"words": []}, "compose": {}}) == []


@pytest.mark.skipif(not (P10.parent.parent / "local" / "pieces" / "P10" / "type" / "manifest.json").is_file(), reason="P10 type pass not on this machine")
def test_caps_from_a_real_type_manifest():
    caps = D.caps_from_manifest(P10.parent.parent / "local" / "pieces" / "P10" / "type")
    assert set(caps) == {0, 1, 2} and all(0.27 < v < 0.30 for v in caps.values())          # three stacked letters, Badeen Display: 28.3 %
    d = design(extra_words=[])
    d2, f = D.validate(d, ctx(caps=caps, probe=probe_for(0.4)))
    assert f == [] and d2["compose"]["cuts"][0]["cut"] == "window_swap"                    # measured 28.3 % passes the "about 30 %" rule


def test_collect_others_lists_the_frames_a_shot_reads():
    p = piece([{"cut": "window_swap", "after": "s1", "word": 0}])
    W.apply_to_piece(p)
    got = W.collect_others(p, "s1", lambda s: f"foot:{s}", lambda s: f"matte:{s}")
    assert got == {"s2": {"footage": "foot:s2", "matte": "matte:s2"}}
    assert W.collect_others(p, "s3", lambda s: s) is None                                 # a shot without window ops reads nothing else
    assert sorted(r for r, _t, _k in W.other_refs(p, "s2")) == ["prev", "prev_matte"]


def test_box_stages_passes_the_neighbour_shot_to_a_window_cut():
    """tools/fusion/box_stages.py cmd_compose: others_for decodes the neighbour shots a window cut reads from the job's files (it passed none before)."""
    from tools.fusion import box_stages as BS
    p = piece([{"cut": "window_swap", "after": "s1", "word": 0}])
    W.apply_to_piece(p)
    shots = {"s1": {"sid": "s1", "footage": "f1", "matte": "m1"}, "s2": {"sid": "s2", "footage": "f2", "matte": "m2"}, "s3": {"sid": "s3", "footage": "f3"}}
    got = BS.others_for(p, "s1", shots, decode=lambda f: f"dec:{f}", read_seq=lambda f: f"seq:{f}")
    assert got == {"s2": {"footage": "dec:f2", "matte": "seq:m2"}}
    assert BS.others_for(p, "s3", shots, decode=lambda f: f, read_seq=lambda f: f) is None
    with pytest.raises(KeyError):
        BS.others_for(p, "s1", {"s1": shots["s1"]}, decode=lambda f: f, read_seq=lambda f: f)


# ------------------------------------------------------------------------------------------------ QC v026 r02: blank grey frames around a flat-ground window
def test_flat_ground_window_never_leaves_a_bare_ground_frame():
    """a window_other word on a flat ground showed the bare ground colour from the shot start to t_open and from t_end + exit_s to the shot end (grey frames, r02 9.5-10.4 s / 12.0-12.75 s).
    With `under` (the shot's own picture) the frame is that picture outside the window, the ground only while the window lives, and the ground fades with the letters."""
    ground = [0.8157, 0.8431, 0.8784]
    foot = np.broadcast_to(RED, (H, Wd, 3)).astype(np.float32).copy()
    kw = dict(mode="other", t_open=1.0, t_end=2.0, exit="fade", exit_s=0.5, min_contrast=None)
    bare = W.window_cut(LETTERS, ground, foot, t=0.5, **kw)                         # old behaviour: the bare ground
    assert np.allclose(bare[0, 0], ground, atol=1e-3)
    for t in (0.0, 0.5, 0.9, 2.6, 3.4):                                              # before the window opens, after it has gone
        out = W.window_cut(LETTERS, ground, foot, t=t, under=foot, **kw)
        assert np.allclose(out, foot, atol=1e-3), t                                  # the shot's own picture, not the ground
    mid = W.window_cut(LETTERS, ground, foot, t=1.6, under=foot, **kw)               # window held: ground outside the letters, picture inside
    assert np.allclose(mid[0, 0], ground, atol=1e-3)
    assert np.allclose(mid[48, 32], RED, atol=0.05)
    ramp = W.window_cut(LETTERS, ground, foot, t=1.1, under=foot, **kw)              # the ground fades in over GROUND_RAMP_S as the window opens
    d_pic, d_gr = np.abs(ramp[0, 0] - foot[0, 0]).max(), np.abs(ramp[0, 0] - np.array(ground)).max()
    assert d_pic > 0.02 and d_gr > 0.02
    fade = W.window_cut(LETTERS, ground, foot, t=2.25, under=foot, **kw)             # the ground leaves with the letters' exit
    assert np.abs(fade[0, 0] - foot[0, 0]).max() < np.abs(mid[0, 0] - foot[0, 0]).max()


def test_flat_ground_shot_ops_carry_the_shots_own_picture():
    spec = dict(kind="window_other", to="self", t_open=1.0, t_end=2.0, ground=[0.8, 0.8, 0.8], exit="fade", exit_s=0.5)
    ops = W.shot_ops(spec, "from")
    assert ops[0]["args"]["under"] == "$footage"
    plain = W.shot_ops(dict(spec, ground=None), "from")                              # the normal window_other keeps A = the shot, no `under`
    assert "under" not in plain[0]["args"]
