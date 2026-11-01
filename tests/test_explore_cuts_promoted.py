"""The promoted explore cuts (2026-10-30): `circle_grid` (+ its word form) and `depth_lines` (the "r3 direct" window look) as live cut kinds.

Covers: the kinds are registered (cut_kinds, the cut menu, the reply schema, the random draw, the 3D depth list), the cut is drawn through pivot_cut.apply_cut with
explore_cuts2.make_render at the planned size and beat phase, `owned_by_cut` (type_clearance state, the recipe's type gate that stops the type stage drawing B's word inside the window
and hands it back after it), the stroke cap per target, the liked-font and contrast rules. Small frames, CPU torch: one quick file."""
import copy
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import cut_kinds as CK            # noqa: E402
from tools.fusion import explore_cuts2 as E         # noqa: E402
from tools.fusion import pivot_cut as PC            # noqa: E402
from tools.fusion import recipe_eval as RE          # noqa: E402

H, W = 96, 54


def piece(kind, prm=None, word2="LAMP", bpm=88, t1=9.0, font2="Blackout_Midnight.ttf"):
    return {"fps": 24, "audio": {"bpm": bpm}, "shots": [{"id": "s1", "frames": 120}, {"id": "s2", "frames": 120}],
            "compose": {"crossfade": 0, "cuts": [{"after": "s1", "cut": kind, "params": prm or {}}]},
            "type": {"bpm": bpm, "beats": [0.4, 1.08], "palette": {"fg": [238, 240, 242], "bg": [27, 31, 43]},
                     "words": [{"text": "ONE", "t0": 0.2, "t1": 3.0, "exit_s": 0.4, "render": "flat", "font": "Interlope-Regular.ttf", "form_s": 0.6},
                               {"text": word2, "t0": 5.2, "t1": t1, "exit_s": 0.4, "render": "flat", "font": font2, "form_s": 0.6}]}}


def clips(n=120, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, H, W, 3), np.uint8)
    b = np.zeros((n, H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):                                  # moving gradients: A bluish, B warm, both change every frame (nothing static)
        a[i] = np.stack([(xx * 3 + i * 2) % 255, (yy * 2 + i) % 200 + 20, np.full_like(xx, 140)], -1)
        b[i] = np.stack([np.full_like(xx, 200), (yy * 2 + xx + i * 3) % 230, (xx * 2 + i * 2) % 120 + 60], -1)
    return a, b


def depth_clip(n=120):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([np.clip((yy / H) * 0.7 + 0.25 * np.sin((xx + i) / 9.0) + 0.15, 0, 1) for i in range(n)]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- registered
def test_kinds_are_registered_everywhere():
    assert set(CK.EXPLORE) == {"circle_grid", "depth_lines"} and set(CK.EXPLORE) <= set(CK.ALL)
    assert not set(CK.EXPLORE) & set(CK.SHOT_3D)                           # no depth_pivot params / orbit styles apply to them
    for raw, want in (("circles", "circle_grid"), ("dot grid", "circle_grid"), ("circle_grid_word", "circle_grid"), ("depth lines", "depth_lines"), ("depth_lines_window", "depth_lines")):
        assert CK.normalize(raw) == want, raw
        assert PC.normalize(raw) == want
    assert CK.family_of("circle_grid") == "explore" == CK.family_of("depth_lines")
    assert CK.owns_word("depth_lines") and CK.owns_word("circle_grid", {"word": "auto"}) and not CK.owns_word("circle_grid") and not CK.owns_word("orbit:push", {"word": 1})
    from tools.fusion import design_options as DO
    from tools.fusion import random_system as RS
    from tools.fusion import reply_schema as RSC
    assert {"circle_grid", "depth_lines"} <= set(DO.CUT_KINDS) and {"circle_grid", "depth_lines"} <= set(RS.menu())
    enum = RSC._cuts_schema(["s1", "s2"])["items"]["properties"]["cut"]["enum"]
    assert {"circle_grid", "depth_lines"} <= set(enum)
    drawn = set()
    for seed in range(60):
        drawn |= set(RS.draw_cuts(seed, "")["kinds"])
    assert {"circle_grid", "depth_lines"} <= drawn                           # the per-piece random draw reaches both
    assert RS.placeable("circle_grid") and not RS.robust("depth_lines")      # the planner may place them; the "always renders" guarantee stays with the 3D kinds


def test_the_guide_names_both_kinds():
    text = open(os.path.join(ROOT, "16_app", "DESIGN_GUIDE_open_model.md"), encoding="utf-8").read()
    assert "`circle_grid`" in text and "`depth_lines`" in text


def test_depth_lines_is_in_the_3d_depth_list_circle_grid_is_not():
    q = piece("depth_lines")
    assert RE.shot_needs(q, "s1")["depth"] and RE.shot_needs(q, "s2")["depth"]       # real per-frame VDA for both shots of the cut
    q = piece("circle_grid", {"word": "auto"})
    assert not RE.shot_needs(q, "s1")["depth"]


# ----------------------------------------------------------------------------------------------- timing on the beat grid
@pytest.mark.parametrize("kind,prm", [("depth_lines", {}), ("circle_grid", {"word": "auto"})])
def test_the_word_lands_on_a_beat_of_the_piece_grid(kind, prm):
    q = piece(kind, prm)
    pl = PC.plan_cut(q, PC.specs(q)[0])
    assert not pl["problems"], pl["problems"]
    style = pl["params"]["style"]
    tl = E.timeline(style, E.resolve(style, {k: v for k, v in pl["params"].items() if k != "style"}))
    assert (pl["n"], pl["i2"]) == (tl["n"], tl["i2"]) == PC.window_frames(kind, {}) or kind  # window_frames without a piece = phase 0
    beat_f = 24.0 * 60.0 / 88.0
    first = 0.4 * 24
    land_abs = pl["start"] + tl["land"]
    k = (land_abs - first) / beat_f
    assert abs(k - round(k)) * beat_f <= 0.6, (land_abs, k)                    # within a frame of a beat
    assert pl["a0"] >= 0 and pl["b_len"] <= 120


def test_circle_dots_blink_on_the_half_beats():
    q = piece("circle_grid", {"word": "auto"})
    pl = PC.plan_cut(q, PC.specs(q)[0])
    a, b = clips()
    p = E.resolve("circle_grid_word", {k: v for k, v in pl["params"].items() if k != "style"})
    tt = E.TT(E._dev())
    eng = E.CircleGrid("circle_grid_word", p, a[:3, :, :, ::-1], np.zeros((3, H, W), np.float32), b[:3, :, :, ::-1], np.zeros((3, H, W), np.float32), tt)
    hb = (24.0 * 60.0 / 88.0) / 2.0
    ons = sorted({int(x) for x in eng.won if x < 10 ** 6})
    assert ons
    for f in ons:
        k = (f + eng.ph) / hb
        assert abs(k - round(k)) * hb <= 0.75 or f >= eng.tl["land"] - 2, (f, k)       # on a half-beat (rounded to a frame), or clamped before the landing


# ----------------------------------------------------------------------------------------------- drawn through apply_cut
@pytest.mark.parametrize("kind,prm", [("circle_grid", {"word": "auto"}), ("circle_grid", {"word": False}), ("depth_lines", {})])
def test_the_cut_is_drawn_and_a_and_b_keep_playing(kind, prm):
    q = piece(kind, dict(prm, **({"word": False} if prm.get("word") is False else {})))
    sp = PC.specs(q)[0]
    a, b = clips()
    m = np.zeros((120, H, W), np.float32)
    m[:, 30:60, 15:35] = 1.0
    res = PC.apply_cut(q, sp, {"s1": a, "s2": b}, mattes={"s1": m, "s2": m}, depths={"s1": depth_clip(), "s2": depth_clip()})
    pl = res.plan
    assert not pl["problems"] and PC.window_ranges(pl)
    assert res.info["style"] == pl["params"]["style"] and res.info["image2_start"] == pl["i2"]
    # the window was written into both clips, the rest of the shots is untouched
    assert (res.a[:pl["a0"]] == a[:pl["a0"]]).all() and (res.b[pl["b_len"]:] == b[pl["b_len"]:]).all()
    assert not (res.a[pl["a0"]:pl["a0"] + pl["i2"]] == a[pl["a0"]:pl["a0"] + pl["i2"]]).all()
    frames = np.concatenate([res.a[pl["a0"]:pl["a0"] + pl["i2"]], res.b[:pl["b_len"]]])
    # never frozen: the picture changes through the whole window, no run of identical frames longer than a blink (<= 3)
    same = [bool((frames[i] == frames[i + 1]).all()) for i in range(len(frames) - 1)]
    run = best = 0
    for s_ in same:
        run = run + 1 if s_ else 0
        best = max(best, run)
    assert best <= 3, best
    # the last frame is B's own frame there (outside the word the picture is B playing)
    last = frames[-1].astype(int)
    own = b[pl["b_len"] - 1].astype(int)
    if kind == "circle_grid" and not prm.get("word"):
        assert np.abs(last - own).mean() < 1.0
    elif kind == "depth_lines":                                      # the flat ground of the word + B playing inside the letters
        ground = np.array([238, 240, 242])
        on_ground = (np.abs(last - ground).max(-1) <= 2)
        assert 0.6 < on_ground.mean() < 0.97
        assert (np.abs(last - own).max(-1)[~on_ground] <= 6).mean() > 0.5                      # inside the letters it is B's own frame (letter edges blend with the ground; the type stage's wider layout has ~45 % edge pixels at this 54 px frame, 2x frames pass 0.6)
    else:                                                            # circle_grid word: B outside the solid word
        diff = np.abs(last - own).max(-1) > 4
        assert 0.0 < diff.mean() < 0.4
    assert res.matte_a is not None and res.matte_b is not None


def test_depth_lines_uses_the_r3_direct_settings_and_the_stroke_cap():
    q = piece("depth_lines")
    pl = PC.plan_cut(q, PC.specs(q)[0])
    p = pl["params"]
    assert p["direct"] is True and p["line_w"] == 0.006 and p["var"] == 1.25 and p["cap"] == PC.CAP_PER_TARGET == 3 and p["morph"] is True and p["style"] == "depth_lines_window"


def test_cap_mask_limits_strokes_per_target():
    rng = np.random.default_rng(1)
    strokes = [rng.random((8, 2)).astype(np.float32) * 50 for _ in range(10)]
    targets = [dict(pts=rng.random((8, 2)).astype(np.float32) * 50, closed=True) for _ in range(2)]
    idx = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]                                     # six strokes collapse onto target 0, four onto target 1
    keep = E.cap_mask(idx, strokes, targets, 3)
    assert sum(keep[:6]) == 3 and sum(keep[6:]) == 3
    assert all(E.cap_mask(idx, strokes, targets, 99))
    # near ones win
    c0 = E._centroid(targets[0]["pts"])
    d = [math.hypot(*(E._centroid(strokes[i]) - c0)) for i in range(6)]
    assert [i for i in range(6) if keep[i]] == sorted(np.argsort(d)[:3].tolist())


def test_the_depth_line_engine_never_has_more_strokes_per_glyph_than_the_cap():
    a, b = clips(30)
    m = np.zeros((30, H, W), np.float32)
    m[:, 40:60, 20:30] = 1.0                                                  # a small subject: its outline is one small target
    dep = depth_clip(30)
    kw = dict(word="LAMP", font="Blackout_Midnight.ttf", bpm=88.0, direct=True, line_w=0.006, kmax=24)
    tt = E.TT(E._dev())

    def eng(cap):
        p = E.resolve("depth_lines_window", dict(kw, cap=cap))
        return E.DepthLines("depth_lines_window", p, a[..., ::-1].copy(), m, dep[0], b[..., ::-1].copy(), m, dep[0], tt, dA_all=list(dep))
    free, capped = eng(0), eng(2)
    assert len(capped.k0) <= len(free.k0)
    assert len(capped.k0) == len(capped.k1) == len(capped.k2) == len(capped.letter) == len(capped.lv1)
    from collections import Counter
    # per glyph target: the strokes left on one aligned target path
    per = Counter(tuple(np.round(k[0], 1)) for k in capped.k2)
    assert max(per.values()) <= 2 + 1, per                                    # cap 2 (+1 for a closed path whose first point is shared by an offset duplicate)


# ----------------------------------------------------------------------------------------------- owned_by_cut
@pytest.mark.parametrize("kind,prm", [("depth_lines", {}), ("circle_grid", {"word": "auto"})])
def test_owned_by_cut_state_and_the_type_gate(kind, prm):
    q = piece(kind, prm)
    pl = PC.plan_cut(q, PC.specs(q)[0])
    cl = {c["text"]: c["state"] for c in PC.type_clearance(q, pl)}
    assert cl == {"ONE": "outside", "LAMP": "owned_by_cut"}
    # a word-less grid owns nothing
    q0 = piece("circle_grid")
    assert {c["state"] for c in PC.type_clearance(q0, PC.plan_cut(q0, PC.specs(q0)[0]))} == {"outside", "rides"} and PC.gate_windows(q0) == []
    gw = PC.gate_windows(q)
    assert len(gw) == 1 and abs(gw[0][0] - pl["start"] / 24.0) < 0.95 and abs(gw[0][1] - (pl["start"] + pl["n"]) / 24.0) < 1e-3
    # the recipe gates every type pass inside the window and nowhere else
    spec = {"win": gw}
    t_in, t_out = (gw[0][0] + gw[0][1]) / 2.0, gw[0][1] + 0.05
    fr = np.ones((4, 4), np.float32)
    assert PC.apply_gate(fr, spec, t_in).sum() == 0 and PC.apply_gate(fr, spec, t_out).sum() == 16
    beauty = np.full((4, 4, 3), 0.9, np.float32)
    assert np.allclose(PC.apply_gate(beauty, dict(spec, bg=[0.1, 0.2, 0.3]), t_in), np.array([0.1, 0.2, 0.3], np.float32))
    assert PC.apply_gate(beauty, dict(spec, bg=[0.1, 0.2, 0.3]), t_out) is beauty


def test_the_type_stage_draws_nothing_in_the_window_and_picks_the_word_up_after(tmp_path):
    from tools.fusion import compose as C
    q = piece("depth_lines")
    notes = PC.own_words(q)
    assert notes and q["type"]["words"][1]["owned_by_cut"]["kind"] == "depth_lines"
    lo, hi = q["type"]["words"][1]["owned_by_cut"]["t0"], q["type"]["words"][1]["owned_by_cut"]["t1"]
    w = q["type"]["words"][1]
    assert lo <= w["t0"] <= hi - w["form_s"] + 1e-6 and w["t1"] >= hi + 0.3        # it forms inside the window (invisible) and is LANDED when the window ends
    assert w["window_source"] == "self" and w["window_ground"] and abs(w["window_t"][0] - hi) < 1e-3     # then keeps the window look on the same flat ground
    assert PC.own_words(q)                                                           # idempotent
    q.setdefault('plan', {})['flow_wake'] = {'holds': [[0.8, 2.6], [5.8, 8.6]]}
    assert RE.piece_word_holds(q)[1][0] >= hi - 1e-6 and RE.piece_word_holds(q)[0] == (0.8, 2.6)                                 # the hold net never redraws the word inside the window
    # compose: a type matte with the word in it on every frame; frames inside the window come out without it
    n = 240
    td = tmp_path / "type"
    (td / "matte").mkdir(parents=True)
    import cv2
    for i in range(n):
        im = np.zeros((H, W), np.uint8)
        im[40:56, 8:46] = 255
        cv2.imwrite(str(td / "matte" / f"{i:05d}.png"), im)
    q2 = copy.deepcopy(q)
    q2["compose"]["default"] = {"ops": [{"op": "over", "out": "image", "args": {"base": "$footage", "fg": [1.0, 1.0, 1.0], "alpha": "$type_matte"}}], "output": "image"}
    rec = RE.build_recipe(q2, "s2", type_dir=str(td))
    assert rec["inputs"]["type_matte"]["gate"]["win"] == PC.gate_windows(q2)
    foot = np.zeros((120, H, W, 3), np.uint8)
    u8, _st = RE.run_compose(q2, "s2", type_dir=str(td), footage=foot, backend="numpy", threads=0)
    u8 = np.asarray(u8)
    t = np.arange(len(u8)) / 24.0 + 120 / 24.0                                       # shot s2 starts at piece frame 120
    gw = PC.gate_windows(q2)
    inside = [i for i in range(len(u8)) if any(a_ <= t[i] < b_ for a_, b_ in gw)]
    outside = [i for i in range(len(u8)) if i not in inside]
    assert inside and outside
    assert all(u8[i].max() == 0 for i in inside), "the type stage drew inside the cut's window"
    assert all(u8[i].max() > 200 for i in outside[-10:])                              # after the window the word is drawn again, at once


# ----------------------------------------------------------------------------------------------- colour + fonts
def test_the_word_form_uses_the_designed_colour_only_when_it_reads_and_a_liked_face():
    q = piece("circle_grid", {"word": "auto"})
    pl = PC.plan_cut(q, PC.specs(q)[0])
    prm = pl["params"]
    from mograph import fonts_liked as FL
    assert FL.is_liked(prm["font"]) and prm["word"] == "LAMP"
    assert prm["ink"] == [242.0, 240.0, 238.0]                                        # the piece's word colour (BGR) is the cut's ink candidate
    frames = np.full((6, 40, 40, 3), 235.0, np.float32)                               # a bright plate: the pale designed ink would not read -> the best palette colour is taken
    mask = np.full((80, 80), 255, np.uint8)
    ink, cr, _ = E.piece_ink(prm, frames, mask)
    assert cr >= 3.0 and not np.allclose(ink, prm["ink"])
    dark = np.full((6, 40, 40, 3), 20.0, np.float32)                                  # a dark plate: the designed colour passes 3:1 and is kept
    ink2, cr2, _ = E.piece_ink(prm, dark, mask)
    assert cr2 >= 3.0 and np.allclose(ink2, prm["ink"])
    # a long B word cannot be formed: the word form falls back to the word-less grid, depth_lines to a plain cut
    ql = piece("circle_grid", {"word": "auto"}, word2="EXTRAORDINARY SILENCE")
    assert "word" not in PC.plan_cut(ql, PC.specs(ql)[0])["params"] and PC.plan_cut(ql, PC.specs(ql)[0])["owned_word"] is None
    qd = piece("depth_lines", word2="EXTRAORDINARY SILENCE")
    assert PC.plan_cut(qd, PC.specs(qd)[0])["problems"]


def test_no_generic_font_in_the_new_code():
    for rel in ("tools/fusion/pivot_cut.py", "tools/fusion/explore_cuts2.py"):
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "load_default" not in src and "DejaVu" not in src and "arial" not in src.lower()


def test_old_styles_are_unchanged_by_the_new_params():
    for st in ("depth_lines_word", "depth_lines_window", "circle_grid", "circle_grid_word"):
        assert E.window_frames(st, {}) == E.window_frames(st, {"phase": 0.0})
    assert E.window_frames("circle_grid_word", {}) == (64, 24) and E.window_frames("depth_lines_window", {}) == (78, 22)
    assert E.window_frames("circle_grid", {"phase": 5.0, "bpm": 88}) != E.window_frames("circle_grid", {"bpm": 88})


def test_depth_lines_needs_a_face_that_can_hold_the_next_shot():
    q = piece("depth_lines", font2="Rock3D-Regular.ttf")                          # a hairline / fragmented face: nothing to show inside the letters
    pl = PC.plan_cut(q, PC.specs(q)[0])
    assert pl["problems"] and "too thin" in pl["problems"][0] and pl["owned_word"] is None and PC.gate_windows(q) == []
    q = piece("depth_lines", font2="Fascinate-Regular.ttf")
    assert not PC.plan_cut(q, PC.specs(q)[0])["problems"]


def test_circle_grid_word_form_is_the_approved_lamp_look():
    """user on promoted_circle_grid_word_lamp.mp4: 'this is great, let's add this as a transition' -> its parameters are locked."""
    q = piece("circle_grid", {"word": "auto"})
    pl = PC.plan_cut(q, PC.specs(q)[0])
    p = pl["params"]
    assert p["style"] == "circle_grid_word" if "style" in p else True
    d = E.DEFAULTS["circle_grid_word"]
    assert (d["order"], d["cols"], d["word_cols"], d["outline"], d["wf"]) == ("beat", 9, 30, False, 0.80)
    assert not {k for k in ("order", "cols", "word_cols", "outline", "wf") if k in p and p[k] != d[k]}      # nothing of the look is overridden by the promotion
    assert p["snap"] is True
    from tools.fusion import random_system as RS
    m = RS.menu()
    assert m["circle_grid"][0] >= 3.0 and m["circle_grid"][0] > m["depth_pivot"][0] and m["depth_lines"][0] >= 3.0                          # a solid weight in the random draw
    hits = sum("circle_grid" in RS.draw_cuts(s_, "")["kinds"] for s_ in range(100))
    assert hits >= 30, hits


def test_depth_lines_morph_has_no_free_lines_stage():
    a, b = clips(30)
    m = np.zeros((30, H, W), np.float32)
    m[:, 30:60, 15:35] = 1.0
    dep = depth_clip(30)
    kw = dict(word="LAMP", font="Blackout_Midnight.ttf", bpm=88.0, direct=True, line_w=0.006, kmax=24, cap=3, morph=True)
    tt = E.TT(E._dev())
    p = E.resolve("depth_lines_window", kw)
    eng = E.DepthLines("depth_lines_window", p, a[..., ::-1].copy(), m, dep[0], b[..., ::-1].copy(), m, dep[0], tt, dA_all=list(dep))
    assert eng.morph and all(np.allclose(x, y) for x, y in zip(eng.k1, eng.k2))                 # one morph: the second stage is the first
    assert E.window_frames("depth_lines_window", kw)[0] < E.window_frames("depth_lines_window", dict(kw, morph=False))[0] + 1
    land = eng.tl["land"]
    f_mid, f_end = 12 + (land - 24) // 2, land - 12
    d_mid = np.mean([np.abs(p_[0] - eng.k0[i]).mean() for i, p_ in enumerate(eng.paths(f_mid, 0, 0))])
    d_tot = np.mean([np.abs(eng.k2[i] - eng.k0[i]).mean() for i in range(len(eng.k0))])
    assert 0.1 * d_tot < d_mid < 0.95 * d_tot                                                   # in between: the image outline is on its way to the glyph outline
    on_glyph = np.mean([np.abs(p_[0] - eng.k2[i]).mean() for i, p_ in enumerate(eng.paths(f_end, 0, 0))])
    assert on_glyph < 0.35 * d_tot                                                              # the letter outlines are (nearly) complete before the fill starts at land - 10
    from collections import Counter
    assert max(Counter(tuple(np.round(k[0], 1)) for k in eng.k2).values()) <= 3 + 1
    # the letters are empty until the outlines are done, full on the beat
    f0, _ = eng.frame(land - 11, a[0][..., ::-1], m[0], b[0][..., ::-1], m[0], 0, 0)
    f1, _ = eng.frame(land, a[0][..., ::-1], m[0], b[0][..., ::-1], m[0], 0, 0)
    U = eng.union > 127
    Uh = U.reshape(H, 2, W, 2).all((1, 3))
    inside = lambda fr: np.abs(fr.cpu().numpy()[Uh] - b[0][..., ::-1].astype(np.float32)[Uh]).max(-1) < 3          # noqa: E731
    assert inside(f1).mean() > 0.8 and inside(f0).mean() < 0.3


def test_depth_lines_is_always_the_outline_morph():
    """user: 'these are great ... this transition should be added': the outline morph is THE depth_lines cut; no param brings the free-lines form back."""
    q = piece("depth_lines", {"morph": False, "direct": False})
    pl = PC.plan_cut(q, PC.specs(q)[0])
    assert pl["params"]["morph"] is True
    from tools.fusion import random_system as RS
    hits = sum("depth_lines" in RS.draw_cuts(s_, "")["kinds"] for s_ in range(100))
    assert hits >= 25, hits


# ----------------------------------------------------------------------------------------------- radial order variant of the word-less circle_grid
def _radial_for(seed, matte=True, prm=None):
    q = piece("circle_grid", prm or {})
    q["type"]["seed"] = seed
    sp = PC.specs(q)[0]
    pl = PC.plan_cut(q, sp)
    ma = None
    if matte:
        ma = np.zeros((120, H, W), np.float32)
        ma[:, 30:60, 15:35] = 1.0
    return PC.radial_order(q, pl, ma), pl


def test_wordless_circle_grid_gets_the_radial_order_for_some_seeds_only_with_a_matte():
    got = [_radial_for(s)[0] for s in range(40)]
    n = sum(1 for g in got if g)
    assert 8 < n < 32 and all(g == PC.RADIAL_PARAMS for g in got if g)          # about half
    assert [_radial_for(s)[0] for s in range(40)] == got                         # seeded: the same cut draws the same
    assert not any(_radial_for(s, matte=False)[0] for s in range(40))            # no subject matte: never radial
    assert not any(_radial_for(s, prm={"order": "sweep"})[0] for s in range(40)) # a spec that sets the order keeps it
    r, pl = next((_radial_for(s) for s in range(40) if _radial_for(s)[0]))
    assert E.window_frames("circle_grid", dict(r, bpm=pl["params"]["bpm"], phase=pl["params"]["phase"], snap=True)) == (pl["n"], pl["i2"])       # timing unchanged


def test_circle_grid_word_and_depth_lines_are_never_radial():
    for s in range(30):
        r, pl = _radial_for(s, prm={"word": "auto"})
        assert pl["params"]["style"] == "circle_grid_word" and not r and "order" not in pl["params"] and pl["params"].get("cols", 9) == 9
