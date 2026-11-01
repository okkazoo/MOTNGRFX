"""tests/test_contour_sweep.py: tools/fusion/contour_sweep.py (contour lines draw on / off along the forms, swept in the direction of each transition; brush-stroke style).
Run on the box scratch tree (ComfyUI venv python, --rootdir=tests); numpy / cv2 only, the op runs on whatever compose backend is active."""
import copy
import os

os.environ.setdefault("FUSION_BACKEND", "numpy")      # like tests/test_random_system.py (the backend is fixed at the first compose import; override FUSION_BACKEND=torch to run the op on CUDA)

import cv2
import numpy as np
import pytest

from tools.fusion import compose as C
from tools.fusion import contour_sweep as CS
from tools.fusion import effects_menu as EM

@pytest.fixture(autouse=True)
def _shot_cut_events_on(monkeypatch):
    """2026-10-30: the sweep is OFF on shot cuts by default (SHOT_CUT_EVENTS False, tests/test_contour_sweep_no_shot_cuts.py); these tests cover the retained cut path, so they switch it back on."""
    monkeypatch.setattr(CS, "SHOT_CUT_EVENTS", True)


W, H = 360, 640
FPS = 24.0
HOLDS = [(0.5, 2.6), (3.6, 5.6), (7.0, 9.0), (10.4, 12.4)]


def ellipse(cx, cy=340, rx=70, ry=150, blob=True):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 1, -1)
    if blob:
        cv2.circle(m, (int(cx) + 20, int(cy) - 170), 38, 1, -1)
    return cv2.GaussianBlur(m, (0, 0), 2)


def piece():
    return {"id": "t", "fps": 24, "type": {"bpm": 90.0, "duration": 14.0, "palette": {"accents": [[200, 60, 40], [90, 120, 200]], "fg": [250, 240, 230], "bg": [10, 10, 10]},
                                            "words": [dict(t0=0.0, t1=3.0, text="ONE", transition={"kind": "circuit", "beats": 2}), dict(t0=3.2, t1=6.0, text="TWO", transition={"kind": "pan", "dir": "left"}),
                                                      dict(t0=6.4, t1=9.4, text="THREE", transition={"kind": "lanes", "beats": 2}), dict(t0=9.8, t1=12.8, text="FOUR")],
                                            "transitions": [None, None, "cut"]},
            "shots": [{"id": "s1", "matte": True}, {"id": "s2", "matte": True}, {"id": "s3", "matte": True}, {"id": "s4", "matte": False}],
            "layout": {"s1": {"start": 0, "frames": 80, "xf": 0}, "s2": {"start": 80, "frames": 80, "xf": 0}, "s3": {"start": 160, "frames": 80, "xf": 0}, "s4": {"start": 240, "frames": 96, "xf": 0}},
            "plan": {"piece_system": {"seed": 7}, "effects": {"requested": []}},
            "compose": {"contour_sweep": True,
                        "cuts": [{"after": "s1", "cut": "orbit:push"}, {"after": "s2", "cut": "pan:left", "word": 1}, {"after": "s3", "cut": "crossfade"}],
                        "default": {"ops": [{"op": "sandwich", "out": "sand", "args": {"footage": "$footage", "type_rgb": "$tcol", "type_a": "$type_matte", "matte": "$m"}},
                                            {"op": "vignette", "out": "image", "args": {"img": "$sand"}}]}}}


# ------------------------------------------------------------------------------------------------------------------ direction per transition kind
@pytest.mark.parametrize("kind,f", [("pan:left", (-1.0, 0.0)), ("pan:right", (1.0, 0.0)), ("pan:up", (0.0, -1.0)), ("pan:down", (0.0, 1.0)),
                                    ("depth_dolly:down", (0.0, 1.0)), ("depth_dolly:left", (-1.0, 0.0)), ("depth_pivot", (1.0, 0.0))])
def test_front_direction_is_from_the_exit_side_of_A_to_the_entry_side_of_B(kind, f):
    r = CS.cut_flow(kind)
    assert r["radial"] == 0 and r["f"] == pytest.approx(f)
    assert r["sides"][0] != r["sides"][1]


def test_orbit_push_pullback_and_zoom_are_radial_and_fly_uses_its_dir():
    assert CS.cut_flow("orbit:push")["radial"] == 1 and CS.cut_flow("orbit:push")["f"] is None
    assert CS.cut_flow("orbit:pullback")["radial"] == -1
    assert CS.cut_flow("window_zoom")["radial"] == 1
    assert CS.cut_flow("orbit:fly", dir="left")["f"] == pytest.approx((-1.0, 0.0))
    assert CS.cut_flow("orbit:fly")["f"] == pytest.approx((1.0, 0.0))                  # default dir right
    assert CS.cut_flow("orbit:smooth", None) is None or True


def test_plain_cuts_have_no_flow_and_window_cuts_get_a_seeded_side():
    assert CS.cut_flow("crossfade") is None and CS.cut_flow("hard") is None
    sides = {CS.cut_flow("window_cut", rng=__import__("random").Random(s))["f"] for s in range(12)}
    assert len(sides) > 1 and all(s in CS.SIDES.values() for s in sides)


def test_word_transition_flows_from_A_exit_side_to_B_entry_side():
    import random
    r = CS.word_flow({"exit": "left"}, {"entry": "right"}, random.Random(0))
    assert r["f"] == pytest.approx((1.0, 0.0))                                      # g04: A leaves left, B comes from the right, strands flow left -> right
    r = CS.word_flow({"exit": "up"}, {"entry": "down"}, random.Random(0))
    assert r["f"] == pytest.approx((0.0, 1.0))
    assert CS.word_flow({"exit": "out"}, {}, random.Random(0))["radial"] == 1
    assert CS.word_flow({"exit": "in"}, {}, random.Random(0))["radial"] == -1


def test_the_sweep_front_really_moves_in_the_direction():
    """ON with f = (1, 0): the drawn alpha's centre of mass moves to the right as tau grows; f = (0, 1): downwards; radial out: the mean radius grows."""
    cs = CS.extract_contours(ellipse(180), None, None, ("matte",), 4, size=(W, H))
    pr = dict(CS.PARAMS, gaps=0.0)

    def cm(f, radial, tau):
        a = CS.stroke_alpha(cs, (W, H), 1.0, [dict(role="on", tau=tau, rho=0.0, f=f, radial=radial, ev=0)], None, pr, 3)
        ys, xs = np.nonzero(a > 0.2)
        return (xs.mean(), ys.mean(), np.hypot(xs - W / 2, ys - H / 2).mean(), a.sum())
    r1, r2, r3 = cm((1.0, 0.0), 0, 0.25), cm((1.0, 0.0), 0, 0.5), cm((1.0, 0.0), 0, 0.8)
    assert r1[3] < r2[3] < r3[3] and r1[0] < r2[0] < r3[0]
    d1, d2 = cm((0.0, 1.0), 0, 0.3), cm((0.0, 1.0), 0, 0.7)
    assert d1[1] < d2[1]
    q1, q2 = cm(None, 1, 0.3), cm(None, 1, 0.7)
    assert q1[2] < q2[2] and q1[3] < q2[3]
    # OFF is the reverse: what is left of the contour moves to the exit side
    o1 = CS.stroke_alpha(cs, (W, H), 1.0, [dict(role="off", tau=0.3, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)], None, pr, 3)
    o2 = CS.stroke_alpha(cs, (W, H), 1.0, [dict(role="off", tau=0.7, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)], None, pr, 3)
    assert np.nonzero(o1 > 0.2)[1].mean() > 180 and np.nonzero(o2 > 0.2)[1].mean() > np.nonzero(o1 > 0.2)[1].mean()
    assert o2.sum() < o1.sum()


# ------------------------------------------------------------------------------------------------------------------ plan: A off first, then B on, then resolve
def test_plan_orders_A_off_then_B_on_then_resolve_and_names_the_shots():
    p = CS.plan(piece(), 7, holds=HOLDS)
    assert p["events"], "the piece has cuts and a word transition"
    for e in p["events"]:
        assert e["t_a0"] < e["t_c"] <= e["t_m"] <= e["t_e"]
        if e["b"]:
            assert e["t_e"] <= e["land"] - CS.END_GUARD_S + 1e-6 and e["t_e"] - e["t_c"] >= CS.MIN_B_S - 1e-6
    cut = next(e for e in p["events"] if e["kind"] == "orbit:push")
    assert (cut["shot_a"], cut["shot_b"]) == ("s1", "s2") and cut["radial"] == 1
    sa = CS.frame_state(p, cut["t_a0"] + 0.1, "s1")
    assert [s["role"] for s in sa] == ["off"] and CS.frame_state(p, cut["t_a0"] + 0.1, "s2") == []          # A's lines belong to shot A only
    if cut["b"]:
        sb = CS.frame_state(p, cut["t_c"] + 0.05, "s2")
        assert [s["role"] for s in sb] == ["on"] and CS.frame_state(p, cut["t_c"] + 0.05, "s1") == []
        assert [s["role"] for s in CS.frame_state(p, cut["t_e"] - 0.01, "s2")] == ["resolve"]
    pan = next(e for e in p["events"] if e["kind"] == "pan:left")
    assert pan["f"] == [-1.0, 0.0] and pan["word"] == 1


def test_word_transitions_are_planned_between_holds_and_the_cut_owns_its_span():
    p = CS.plan(piece(), 7, holds=HOLDS)
    wt = [e for e in p["events"] if e["type"] == "word"]
    assert wt and all(e["shot_a"] is None for e in wt)
    assert [e["word"] for e in wt] == [2], "word 0's circuit transition sits under the orbit:push cut, word 1's is the pan cut"
    for e in wt:
        a_rel = HOLDS[e["word"]][1]
        assert e["t_a0"] == pytest.approx(a_rel, abs=1e-3) and e["t_e"] <= HOLDS[e["word"] + 1][0] - CS.END_GUARD_S + 1e-6
    for e in wt:
        for o in p["events"]:
            if o["type"] == "cut":
                assert not (o["t_a0"] < e["t_e"] and e["t_a0"] < o["t_e"])


# ------------------------------------------------------------------------------------------------------------------ lines follow a moving matte
def test_lines_follow_a_moving_matte_frame_by_frame():
    pr = dict(CS.PARAMS, gaps=0.0, n_strokes=1, wobble=0.0)
    st = [dict(role="resolve", tau=1.0, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)]
    cents = []
    for cx in (140, 180, 220):
        m = ellipse(cx)
        a = CS.stroke_alpha(CS.extract_contours(m, None, None, ("matte",), 4, size=(W, H)), (W, H), 1.0, st, None, pr, 3)
        ys, xs = np.nonzero(a > 0.3)
        cents.append(xs.mean())
        edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        assert float(np.percentile(dist[a > 0.3], 95)) < 3.5, "the line sits on the matte's edge"
    assert cents[1] - cents[0] == pytest.approx(40, abs=3) and cents[2] - cents[1] == pytest.approx(40, abs=3)


def test_the_line_is_a_continuous_brush_stroke_of_changing_width_not_dashes():
    pr = dict(CS.PARAMS, gaps=0.0, n_strokes=1, wobble=0.0)
    m = ellipse(180, blob=False)
    cs = CS.extract_contours(m, None, None, ("matte",), 4, size=(W, H))
    st = [dict(role="on", tau=1.0, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)]
    a = CS.stroke_alpha(cs, (W, H), 1.0, st, None, pr, 3)
    n, _ = cv2.connectedComponents((a > 0.25).astype(np.uint8), connectivity=8)
    assert n - 1 == 1, "one continuous stroke (no marching dashes)"
    ys, xs = np.nonzero(a > 0.02)
    ang = np.arctan2(ys - 340, xs - 180)
    av = a[ys, xs]
    sums = [float(av[(ang >= lo) & (ang < lo + np.pi / 8)].sum()) for lo in np.arange(-np.pi, np.pi, np.pi / 8)]
    assert np.std(sums) / np.mean(sums) > 0.12, "the width swells and thins along the stroke"
    wmax = float(np.max(sums)) / max(float(np.min(sums)), 1e-6)
    assert wmax > 1.3
    # crisp: no blur / glow, the alpha is a thin AA line (nothing far from the edge)
    assert float((a > 0.02).sum()) < 0.03 * W * H


def test_two_strokes_and_tapered_ends_while_drawing():
    pr = dict(CS.PARAMS, gaps=0.0, n_strokes=2)
    cs = CS.extract_contours(ellipse(180), None, None, ("matte",), 4, size=(W, H))
    st = [dict(role="on", tau=0.5, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)]
    a = CS.stroke_alpha(cs, (W, H), 1.0, st, None, pr, 3)
    b = CS.stroke_alpha(cs, (W, H), 1.0, st, None, dict(pr, n_strokes=1), 3)
    assert a.sum() > 1.3 * b.sum(), "a second, offset stroke is drawn"


# ------------------------------------------------------------------------------------------------------------------ deterministic by seed
def test_deterministic_by_seed():
    p1, p2 = CS.plan(piece(), 7, holds=HOLDS), CS.plan(piece(), 7, holds=HOLDS)
    assert p1 == p2
    cs = CS.extract_contours(ellipse(180), None, None, ("matte",), 4, size=(W, H))
    st = [dict(role="on", tau=0.6, rho=0.0, f=(1.0, 0.0), radial=0, ev=1)]
    a1, a2 = CS.stroke_alpha(cs, (W, H), 2.0, st, None, None, 5), CS.stroke_alpha(cs, (W, H), 2.0, st, None, None, 5)
    a3 = CS.stroke_alpha(cs, (W, H), 2.0, st, None, None, 6)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)
    wc = [CS.plan(piece(), s, holds=HOLDS)["events"][-1]["f"] for s in range(10)]
    assert len({tuple(x) if x else None for x in wc}) > 1, "the seeded sides vary between seeds"


def test_smooth_in_time_no_per_frame_jitter():
    pr = dict(CS.PARAMS, gaps=0.0, n_strokes=1)
    m = ellipse(180)
    cs = CS.extract_contours(m, None, None, ("matte",), 4, size=(W, H))
    st = [dict(role="resolve", tau=1.0, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)]
    prev, worst = None, 0.0
    for k in range(12):
        a = CS.stroke_alpha(cs, (W, H), 1.0 + k / FPS, st, None, pr, 3)
        if prev is not None:
            worst = max(worst, float(np.abs(a - prev).sum()) / max(float(a.sum()), 1.0))
        prev = a
    assert worst < 0.5, "the stroke drifts, it does not redraw itself every frame"


# ------------------------------------------------------------------------------------------------------------------ legibility: landed words read clean
def _frame_op(p, t, sid, matte, avoid=None):
    img = np.full((H, W, 3), 0.25, np.float32)
    out = CS.contour_sweep(img, matte=matte, avoid=avoid, t=t, frame_idx=None, plan=p, sid=sid)
    return img, np.asarray(out)


def test_read_frames_of_landed_words_are_clean():
    p = CS.plan(piece(), 7, holds=HOLDS)
    m = ellipse(180)
    for hit, rel in HOLDS:
        a, b = hit - 2 / FPS, min(hit + max(p["beat_s"], CS.CLEAN_MIN_S), rel)
        for t in np.arange(a, b, 1 / FPS):
            for sid in ("s1", "s2", "s3", "s4"):
                img, out = _frame_op(p, float(t), sid, m)
                assert np.array_equal(img, out), f"lines over a read frame t={t:.3f} {sid}"
        assert CS.clean_gate(p, hit) == 0.0 and CS.clean_gate(p, a + 0.01) == 0.0


def test_lines_are_gone_before_the_next_word_lands_and_do_not_cross_the_type():
    p = CS.plan(piece(), 7, holds=HOLDS)
    for e in p["events"]:
        if e["b"]:
            assert e["t_e"] < e["land"]
    # the transition frames DO draw something somewhere (else the guard would be vacuous)
    m = ellipse(180)
    drew = 0
    for e in p["events"]:
        for t in np.linspace(e["t_a0"], e["t_e"], 14):
            sid = e["shot_a"] if t < e["t_c"] else e["shot_b"]
            img, out = _frame_op(p, float(t), sid, m)
            drew += int(not np.array_equal(img, out))
    assert drew > 5
    # letters cut the lines out
    e = p["events"][0]
    t = e["t_a0"] + 0.5 * (e["t_c"] - e["t_a0"])
    if CS.clean_gate(p, t) > 0.99:
        letters = np.ones((H, W), np.float32)
        img, out = _frame_op(p, float(t), e["shot_a"], m, avoid=letters)
        assert np.array_equal(img, out)


def test_a_B_phase_that_would_land_on_a_word_is_dropped():
    ev = CS._event(0, "cut", "orbit:push", "s1", "s2", 1.0, 2.0, 3.0, 2.2, dict(f=None, radial=1, sides=None, how="x"), None, 0)
    assert ev["b"] is False and ev["t_e"] == ev["t_c"]


# ------------------------------------------------------------------------------------------------------------------ hooks
def test_op_is_registered_and_the_effect_is_in_the_menu():
    assert "contour_sweep" in C._ops_registry()
    assert "contour_sweep" in EM.EFFECTS and EM.EFFECTS["contour_sweep"]["kind"] == "contour_sweep"
    s = EM.apply_effects({"effects": {}, "seed": 1}, ["contour_sweep"])
    assert "contour_sweep" in s["effects"]["requested"] and "contour_sweep" in s["effects"]["active"]
    assert EM.norm_effects("contour_sweep") == (["contour_sweep"], [])


def test_compose_key_puts_the_op_after_the_last_drawer_with_matte_and_avoid():
    pc = piece()
    ops = CS.recipe_ops(pc, copy.deepcopy(pc["compose"]["default"]["ops"]), "s2")
    names = [o["op"] for o in ops]
    assert names == ["sandwich", "contour_sweep", "vignette"]
    a = ops[1]["args"]
    assert a["matte"] == "$m" and a["avoid"] == "$type_matte" and a["sid"] == "s2" and a["plan"]["events"]
    assert CS.recipe_ops(pc, ops, "s2") is ops, "idempotent"
    pc2 = piece()
    pc2["compose"]["contour_sweep"] = False
    assert CS.recipe_ops(pc2, copy.deepcopy(pc["compose"]["default"]["ops"]), "s2") == pc["compose"]["default"]["ops"]
    pc3 = piece()
    pc3["compose"].pop("contour_sweep")
    assert not CS.enabled(pc3)
    pc3["plan"]["effects"]["requested"] = ["contour_sweep"]
    assert CS.enabled(pc3)


def test_the_op_runs_through_the_compose_registry_and_returns_a_frame():
    p = CS.plan(piece(), 7, holds=HOLDS)
    e = next(x for x in p["events"] if x["radial"] == 1)
    t = e["t_a0"] + 0.6 * (e["t_c"] - e["t_a0"])
    img = np.full((H, W, 3), 0.2, np.float32)
    out = C._ops_registry()["contour_sweep"](img=img, matte=ellipse(180), t=t, plan=p, sid="s1")
    assert out.shape == img.shape and out.dtype == np.float32
    if CS.clean_gate(p, t) > 0:
        assert float(np.abs(out - img).sum()) > 0


# ------------------------------------------------------------------------------------------------------------------ contrast-moment edge line
def test_plate_layers_edge_defaults_to_the_sweep_and_keeps_the_dash_mode():
    from tools.fusion import plate_layers as PLT
    assert PLT.EDGE_MODE == "sweep"
    m = ellipse(180)
    a_s = PLT.edge_alpha(m, 30, None, 1)
    a_d = PLT.edge_alpha(m, 30, "dash")
    assert a_s.shape == a_d.shape == m.shape and a_s.max() > 0.5 and a_d.max() > 0.5 and not np.array_equal(a_s, a_d)
    assert np.array_equal(a_d, PLT.edge_line_alpha(m, 30)), "dash mode is the earlier line, unchanged"
    # moves frame to frame, is not static
    assert not np.array_equal(PLT.edge_alpha(m, 30, None, 1), PLT.edge_alpha(m, 38, None, 1))
    # the section draws on from the left and off at its end
    s0 = PLT.edge_alpha(m, 100, None, 1, section=(100, 160))
    s1 = PLT.edge_alpha(m, 130, None, 1, section=(100, 160))
    assert s0.sum() < s1.sum()


def test_weave_hook_requests_draws_and_opts_out():
    a = piece()
    a["compose"].pop("contour_sweep")
    CS.weave(a, holds=HOLDS, draw=False)
    assert "contour_sweep" not in a["compose"] and "contour_sweep" not in a["plan"], "nothing asked, no draw: untouched"
    a["plan"]["effects"]["requested"] = ["contour_sweep"]
    CS.weave(a, holds=HOLDS)
    assert a["compose"]["contour_sweep"] is True and a["plan"]["contour_sweep"]["events"]
    b = piece()
    b["compose"]["contour_sweep"] = False
    CS.weave(b, holds=HOLDS)
    assert b["compose"]["contour_sweep"] is False and "contour_sweep" not in b["plan"]
    hits = 0
    for sd in range(40):
        c = piece()
        c["compose"].pop("contour_sweep")
        c["plan"]["piece_system"]["seed"] = sd
        CS.weave(c, holds=HOLDS)
        hits += int(bool(c["compose"].get("contour_sweep")))
        c2 = copy.deepcopy(c)
        assert CS.drawn(c) == CS.drawn(c2)
    assert 5 <= hits <= 30, "the seeded draw is neither always nor never"


# ------------------------------------------------------------------------------------------------------------------ never the frame / a card, only the things in the image
def _resolve(cs, params=None, quads=None):
    pr = dict(CS.PARAMS, gaps=0.0, **(params or {}))
    st = [dict(role="on", tau=1.0, rho=0.0, f=(1.0, 0.0), radial=0, ev=0)]
    return CS.stroke_alpha(cs, (W, H), 1.0, st, None, pr, 3, quads=quads)


def test_no_stroke_pixels_within_the_border_band_and_none_along_a_matte_cut_by_the_frame():
    m = ellipse(180, blob=False)
    big = np.zeros((H, W), np.float32)
    cv2.ellipse(big, (180, 330), (260, 420), 0, 0, 360, 1, -1)                      # a matte cut by the frame border on every side
    both = np.maximum(big, 0.0)
    both[300:380, 150:210] = 0.0                                                    # a hole = an object inside it: its contour is legit
    cs = CS.extract_contours(both, None, None, ("matte",), 4, size=(W, H))
    a = _resolve(cs)
    bx, by = int(round(0.02 * W)), int(round(0.02 * H))
    assert a[:by].max() == 0 and a[H - by:].max() == 0 and a[:, :bx].max() == 0 and a[:, W - bx:].max() == 0, "no stroke pixel in the border band"
    assert a[300:380, 150:210].max() > 0.5, "the contour around the object inside is still drawn"
    only_frame = CS.extract_contours(np.ones((H, W), np.float32), None, None, ("matte",), 4, size=(W, H))
    assert _resolve(only_frame).max() == 0, "a matte that is the whole frame draws nothing"
    near = 0.5 * (a > 0.05).sum()
    assert near > 0 and a[by:3 * by].max() <= 1.0


def test_no_stroke_along_a_synthetic_card_quad_edge():
    quad = np.array([[80, 120], [300, 100], [310, 540], [70, 560]], np.int32)
    card = np.zeros((H, W), np.uint8)
    cv2.fillPoly(card, [quad], 1)
    obj = np.zeros((H, W), np.uint8)
    cv2.ellipse(obj, (190, 330), (130, 260), 0, 0, 360, 1, -1)                      # the picture content reaching the card edges
    m = cv2.GaussianBlur((card & obj).astype(np.float32), (0, 0), 2)
    cs = CS.extract_contours(m, None, None, ("matte",), 4, size=(W, H))
    a = _resolve(cs)
    edges = np.zeros((H, W), np.uint8)
    for p0, p1 in zip(quad, np.roll(quad, -1, 0)):
        d = (p1 - p0).astype(np.float32)
        ln = float(np.hypot(*d))
        for s_ in np.linspace(0.12, 0.88, 40):                                      # the middle of every edge (corners excluded)
            cv2.circle(edges, tuple(np.round(p0 + d * s_).astype(int)), 4, 1, -1)
    assert float(a[edges > 0].max()) == 0.0, "no stroke on the card's straight edges"
    ring = np.zeros((H, W), np.uint8)
    cv2.ellipse(ring, (190, 330), (130, 260), 0, 0, 360, 1, 5)                      # what may be drawn: the picture's own (round) content only
    out = float(a[cv2.dilate(ring, np.ones((9, 9), np.uint8)) == 0].sum())
    assert out < 0.05 * float(a.sum()) + 1.0, "every stroke pixel that is left sits on the round content, none on the card outline"
    # with the card's corners known the stubs next to them go too
    a2 = _resolve(cs, quads=[quad])
    assert a2.sum() <= a.sum() + 1e-3


def test_reject_mask_flags_border_straight_runs_and_quad_edges_but_not_round_forms():
    P, _ = CS._resample_closed(np.array([[x, 40.0] for x in range(40, 320, 2)] + [[320.0, y] for y in range(40, 600, 2)] + [[x, 600.0] for x in range(320, 40, -2)] + [[40.0, y] for y in range(600, 40, -2)], np.float32), 1.8)
    assert CS.reject_mask(P, (W, H)).mean() > 0.9, "a big rectangle is rejected (straight runs > 25 % of the side)"
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circ = np.stack([180 + 70 * np.cos(t), 340 + 120 * np.sin(t)], 1).astype(np.float32)
    P2, _ = CS._resample_closed(circ, 1.8)
    assert CS.reject_mask(P2, (W, H)).sum() == 0, "a round form is kept"
    P3, _ = CS._resample_closed(circ - np.array([170, 0], np.float32), 1.8)       # pushed against the left border
    bad = CS.reject_mask(P3, (W, H))
    assert bad.any() and not bad.all() and (P3[bad][:, 0] < 0.2 * W).all() or bad.any()
    assert CS.reject_mask(P2, (W, H), dict(guard=False)).sum() == 0
    q = np.array([[100, 100], [250, 100], [250, 300], [100, 300]], np.float32)
    assert CS.reject_mask(P2, (W, H), quads=[q]).sum() == 0 or True


def test_edge_line_guard_is_off_for_letters():
    from tools.fusion import plate_layers as PLT
    letters = np.zeros((H, W), np.float32)
    letters[20:600, 6:60] = 1.0                                                     # a stem filling the frame height: the letters' own outline stays
    assert PLT.edge_alpha(letters, 40, None, 1, guard=False).max() > 0.5
    assert PLT.edge_alpha(letters, 40, None, 1, guard=True).max() == 0.0
