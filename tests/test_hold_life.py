"""hold life (2026-10-30): effects that keep a landed word alive in its hold while it stays legible and centred (tools/fusion/hold_life.py).
Synthetic words drawn with cv2 (no font files), run through the real compose op (`hold_life`, registered in compose._ops_registry) on whatever backend FUSION_BACKEND selects
(numpy by default; on the box: FUSION_BACKEND=torch). Covers the LEGIBILITY GUARD (fraction visible, max off frames, no adjacent off, displacement), the read frames, the centre under
the pulse, determinism by seed, and the wiring (registry, weave, legibility_pass twin)."""
import copy
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import hold_life as H  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

W, HH = 360, 640
FPS = 24
ALL = ["letter_flicker", "glitch_slice", "pulse", "glow_breathe", "outline_shimmer"]


def word_alpha(text="DREAMING", w=W, h=HH, frac=0.8):
    img = np.zeros((h, w), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 3)
    s = frac * w / tw
    cv2.putText(img, text, (int(0.5 * (w - tw * s)), int(h / 2 + th * s / 2)), font, s, 255, max(2, int(3 * s)), cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def mkplan(text="DREAMING", effects=("letter_flicker",), seed=3, bpm=120, fps=FPS, hit=0.5, release=4.5, outline=None):
    words = [dict(text=text, hit=hit, release=release)]
    return H.plan(words, {"fps": fps, "type": {"bpm": bpm}}, seed, effects=list(effects), outline=outline)


def coverage(clean, out, lm):
    cov = []
    for r in range(lm["m"]):
        msk = lm["rmap"] == r
        cov.append(float(out[msk].sum() / max(clean[msk].sum(), 1e-6)))
    return np.array(cov)


def bbox_centre(a, thr=0.1):
    ys, xs = np.nonzero(a > thr)
    return 0.5 * (xs.min() + xs.max() + 1), 0.5 * (ys.min() + ys.max() + 1), xs.min(), xs.max() + 1


def row_lags(a, o, maxlag=10):
    """per row with ink: the horizontal lag (px) at which the row of `a` best matches the row of `o`."""
    lags = np.arange(-maxlag, maxlag + 1)
    corr = np.stack([(np.roll(a, int(L), axis=1) * o).sum(1) for L in lags])
    best = lags[corr.argmax(0)].astype(np.float32)
    best[a.sum(1) < 1e-3] = 0.0
    return best


def run_hold(p, alpha, key, frames=None):
    """alpha twin output for every frame of the first word's hold."""
    wp = p["words"][0]
    frames = frames or range(wp["hit_f"] - 2, wp["release_f"] + 2)
    return {f: np.asarray(H.hold_life(alpha, alpha, frame_idx=f, plan=p, state_key=key)) for f in frames}


# ------------------------------------------------------------------------------------------------------------- registry / menu
def test_op_is_registered_in_compose_and_recipe_eval():
    assert "hold_life" in C._ops_registry()
    assert "hold_life" in RE.op_table()
    assert callable(H.weave) and set(H.EFFECTS) == set(ALL)


def test_menu_draws_one_or_two_and_never_flicker_with_slice():
    texts = ["DREAMING", "UNBREAKABLE"]
    seen = set()
    for seed in range(120):
        eff, _ = H.draw_effects(seed, texts, True)
        assert 1 <= len(eff) <= 2 and len(set(eff)) == len(eff)
        assert not {"letter_flicker", "glitch_slice"} <= set(eff)
        seen |= set(eff)
    assert seen == set(ALL)
    eff, why = H.draw_effects(1, ["NEON", "SEA"], False)
    assert "letter_flicker" not in eff and "outline_shimmer" not in eff and "letter_flicker" in why
    eff, _ = H.draw_effects(1, ["DREAMING"], True, requested=["pulse", "nonsense"])
    assert eff == ["pulse"]                                             # a request is honoured as asked


# ------------------------------------------------------------------------------------------------------------- legibility guard (the plan)
@pytest.mark.parametrize("fps", [24, 30])
def test_guard_numbers_over_many_seeds(fps):
    for text in ("DREAMING", "UNBREAKABLE", "STAND UP", "NEON", "AMBER ROOM LIGHT"):
        for bpm in (90, 120, 140, 174):
            for seed in range(14):
                p = mkplan(text, ALL, seed, bpm, fps, hit=60.0 / bpm * 2, release=60.0 / bpm * 2 + 6 * 60.0 / bpm, outline=[[1.0, 4.0]])
                g = H.check_plan(p)
                assert g["min_visible"] >= H.GUARD["min_visible"] - 1e-9, (text, bpm, seed, g)
                assert g["max_off_run"] <= H.max_off_frames(fps), (text, bpm, seed, g)
                assert g["adjacent"] == 0 and g["over_budget"] == 0, (text, bpm, seed, g)
                assert g["read_dirty"] == 0 and g["beat_dirty"] == 0, (text, bpm, seed, g)
                assert g["max_disp"] <= H.GUARD["max_disp"] + 1e-9, (text, bpm, seed, g)
                assert g["max_scale"] <= 1.0 + H.GUARD["pulse_max"] + 1e-9


def test_letter_flicker_is_active_on_long_words_and_absent_on_short_ones():
    long_ = mkplan("DREAMING", ["letter_flicker"], seed=2)["words"][0]
    assert len(long_["flicker"]) >= 4 and all(1 <= d <= 2 for _, _, d, _ in long_["flicker"])
    short = mkplan("NEON", ["letter_flicker"], seed=2)["words"][0]
    assert short["flicker"] == [] and short["skipped"]
    assert H.budget(6) == 0 and H.budget(7) == 1 and H.budget(14) == 2


def test_max_off_frames_scales_with_fps():
    assert H.max_off_frames(24) == 2 and H.max_off_frames(48) == 4 and H.max_off_frames(12) == 1


def test_enforce_guard_drops_adjacent_and_over_budget():
    assert H.enforce_guard({2: 0.0, 3: 0.0}, 8) == {2: 0.0}
    assert H.enforce_guard({1: 0.0, 5: 0.0}, 8) == {1: 0.0}            # budget of 8 letters is 1
    assert H.enforce_guard({1: 0.0, 5: 0.0}, 14) == {1: 0.0, 5: 0.0}
    assert H.enforce_guard({0: 0.0}, 6) == {}


def test_read_frames_and_hold_edges_are_identity():
    p = mkplan("DREAMING", ALL, seed=5, outline=[[0.5, 4.0]])
    wp = p["words"][0]
    for f in range(wp["hit_f"] - 3, wp["hit_f"] + H.GUARD["read_frames"]):
        assert H.is_identity(H.frame_state(wp, f, FPS)), f
    for f in (wp["release_f"], wp["release_f"] + 1):
        assert H.is_identity(H.frame_state(wp, f, FPS)), f


def test_deterministic_by_seed():
    a = json.dumps(mkplan("DREAMING", ALL, seed=11, outline=[[1.0, 4.0]]), sort_keys=True)
    b = json.dumps(mkplan("DREAMING", ALL, seed=11, outline=[[1.0, 4.0]]), sort_keys=True)
    c = json.dumps(mkplan("DREAMING", ALL, seed=12, outline=[[1.0, 4.0]]), sort_keys=True)
    assert a == b and a != c
    assert H.draw_effects(7, ["DREAMING"], True) == H.draw_effects(7, ["DREAMING"], True)
    s = json.dumps(H.slice_state(99, 40, 0.8))
    assert s == json.dumps(H.slice_state(99, 40, 0.8))


# ------------------------------------------------------------------------------------------------------------- the op on real pixels
def test_letter_masks_are_the_letters_and_marks_join_their_letter():
    lm = H.letter_map(word_alpha("DREAMING"))
    assert lm["m"] == 8 and not lm["vertical"]
    xs = [0.5 * (b[0] + b[2]) for b in lm["boxes"]]
    assert xs == sorted(xs)                                              # reading order, left to right
    lm2 = H.letter_map(word_alpha("mind"))                               # the dot of the i is part of the i
    assert lm2["m"] == 4
    lm3 = H.letter_map(np.rot90(word_alpha("DREAMING", frac=0.7)).copy())  # a word that reads vertically is sorted along y
    assert lm3["vertical"] and lm3["m"] == 8
    ys = [0.5 * (b[1] + b[3]) for b in lm3["boxes"]]
    assert ys == sorted(ys)


def test_op_letter_flicker_guard_on_pixels():
    a = word_alpha("DREAMING")
    p = mkplan("DREAMING", ["letter_flicker"], seed=4)
    lm = H.letter_map(a)
    assert lm["m"] == p["words"][0]["n"] == 8
    outs = run_hold(p, a, "t_flick")
    runs, total_off, minvis = {}, 0, 1.0
    for f, o in outs.items():
        cov = coverage(a, o, lm)
        off = set(np.nonzero(cov < 0.97)[0].tolist())
        total_off += len(off)
        minvis = min(minvis, 1.0 - len(off) / lm["m"])
        assert not any((r + 1) in off for r in off), f"adjacent letters off at {f}"
        for r in list(runs):
            if r not in off:
                runs.pop(r)
        for r in off:
            runs[r] = runs.get(r, 0) + 1
            assert runs[r] <= H.max_off_frames(FPS), f"letter {r} off {runs[r]} frames"
    assert total_off >= 4, "the effect must actually blink letters"
    assert minvis >= 0.85 - 1e-9


def test_op_read_frames_and_beat_frames_are_untouched_by_dropouts_and_slices():
    a = word_alpha("DREAMING")
    p = mkplan("DREAMING", ["letter_flicker"], seed=6)
    p2 = mkplan("DREAMING", ["glitch_slice"], seed=6)
    for plan_, key in ((p, "t_rd1"), (p2, "t_rd2")):
        wp = plan_["words"][0]
        beat_fs = [b for b in wp["beat_frames"] if wp["f_on"] <= b < wp["f_off"]]
        frames = list(range(wp["hit_f"], wp["hit_f"] + H.GUARD["read_frames"])) + [b + d for b in beat_fs for d in (-1, 0, 1)]
        for f, o in run_hold(plan_, a, key, frames=frames).items():
            assert np.array_equal(o, a), (key, f)


def test_op_pulse_centre_never_moves_and_growth_is_bounded():
    a = word_alpha("DREAMING")
    p = mkplan("DREAMING", ["pulse"], seed=3, release=5.5)
    cx0, cy0, x0, x1 = bbox_centre(a)
    outs = run_hold(p, a, "t_pulse")
    widest = 0.0
    changed = 0
    for f, o in outs.items():
        cx, cy, ox0, ox1 = bbox_centre(o)
        assert abs(cx - cx0) <= 0.75 and abs(cy - cy0) <= 0.75, (f, cx - cx0, cy - cy0)      # sub-pixel, from the integer bbox of an antialiased edge
        widest = max(widest, (ox1 - ox0) / (x1 - x0))
        changed += int(not np.array_equal(o, a))
        assert max(abs(ox0 - x0), abs(ox1 - x1)) <= H.GUARD["max_disp"] * W + 1.0
    assert 1.01 <= widest <= 1.03 + 0.004, widest
    assert changed >= 6, "the pulse must be visible on several frames"


def test_op_slices_stay_within_one_and_a_half_percent_and_split_the_face():
    a = word_alpha("DREAMING")
    p = mkplan("DREAMING", ["glitch_slice"], seed=8, release=6.5)
    wp = p["words"][0]
    face = np.stack([np.full_like(a, 0.9), np.full_like(a, 0.8), np.full_like(a, 0.2)], -1)
    lim = H.GUARD["max_disp"] * W
    hit_any = 0
    for f, o in run_hold(p, a, "t_slice").items():
        if np.array_equal(o, a):
            continue
        hit_any += 1
        assert float(np.abs(row_lags(a, o)).max()) <= lim + 0.5, f          # best horizontal lag of every row (the split's union copies sit +-split around it)
        fo = np.asarray(H.hold_life(face, a, frame_idx=f, plan=p, state_key="t_slice"))
        assert fo.shape == face.shape and fo.min() >= 0.0 and fo.max() <= 1.0
    assert hit_any >= 3
    # the split: on a burst frame the red and blue channel alphas are shifted against each other by 1-3 px
    f = wp["slices"][0][0]
    st = H.frame_state(wp, f, FPS)
    assert 0 < st["split"] * W <= 3.3
    fo = np.asarray(H.hold_life(face, a, frame_idx=f, plan=p, state_key="t_slice"))
    rr, bb = fo[..., 0] / 0.9, fo[..., 2] / 0.2
    assert float(np.abs(rr - bb).max()) > 0.3                           # fringes: red-only and blue-only pixels exist


def test_op_is_identity_outside_holds_and_without_a_plan():
    a = word_alpha("DREAMING")
    p = mkplan("DREAMING", ALL, seed=1)
    assert H.hold_life(a, a, frame_idx=0, plan=p) is not None
    assert np.array_equal(np.asarray(H.hold_life(a, a, frame_idx=0, plan=p, state_key="t_id")), a)
    assert np.array_equal(np.asarray(H.hold_life(a, a, frame_idx=40, plan=None)), a)


# ------------------------------------------------------------------------------------------------------------- weave
def _chain():
    return [
        {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [96, 160], "guide": "$footage"}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "crossfade", "out": "ta", "args": {"a": "$type_line", "b": "$type_matte", "t": {"timeline": [[0.0, 0.0], [5.0, 1.0]]}}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "matte": "$m", "type_a": "$ta", "type_rgb": "$tcol", "matte_threshold": 0.5}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0}},
    ]


def _piece(words=("DREAMING", "NEON"), n=96, extra=None):
    t = {"bpm": 120, "beats": [round(0.5 * k, 3) for k in range(40)], "words": [dict(text=w) for w in words]}
    pc = {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [96, 160]}, "type": t,
          "shots": [{"id": "s1", "frames": n, "matte": True}],
          "compose": {"crossfade": 0.0, "default": {"ops": _chain()}, "color": "fixed"}, "plan": {}}
    pc.update(extra or {})
    return pc


HOLDS = [(0.5, 2.5), (3.0, 5.0)]


def test_weave_puts_the_op_in_front_of_the_drawer_and_is_idempotent():
    pc = _piece()
    pc["compose"]["hold_life"] = ["letter_flicker", "pulse"]
    H.weave(pc, holds=HOLDS)
    ops = pc["compose"]["default"]["ops"]
    names = [o["op"] for o in ops]
    assert names.index("hold_life") < names.index("sandwich")
    sw = next(o for o in ops if o["op"] == "sandwich")
    hl = next(o for o in ops if o["op"] == "hold_life")
    assert sw["args"]["type_a"] == "$" + hl["out"] and hl["args"]["ref"] == "$ta"
    json.dumps(pc)                                                       # JSON-able
    again = copy.deepcopy(pc)
    H.weave(again, holds=HOLDS)
    assert [o["op"] for o in again["compose"]["default"]["ops"]] == names
    assert pc["plan"]["hold_life"]["effects"] == ["letter_flicker", "pulse"]
    pc["compose"]["hold_life"] = False
    H.weave(pc, holds=HOLDS)
    assert [o["op"] for o in pc["compose"]["default"]["ops"]] == [o["op"] for o in _chain()]
    assert next(o for o in pc["compose"]["default"]["ops"] if o["op"] == "sandwich")["args"]["type_a"] == "$ta"


def test_weave_glow_breathe_adds_a_beat_driven_glow_after_the_drawer():
    pc = _piece()
    pc["compose"]["hold_life"] = ["glow_breathe"]
    H.weave(pc, music={"curves": {"rms": [0.2] * 200, "fps": 24}}, holds=HOLDS)
    ops = pc["compose"]["default"]["ops"]
    names = [o["op"] for o in ops]
    g = ops[names.index("add_glow")]
    assert names.index("sandwich") < names.index("add_glow") < names.index("grade")
    assert g["args"]["mask"] == "$ta" and g["args"]["radius"]["curve"] == "rms"                  # glow alone puts no op on the type layer
    assert not any(o["op"] == "hold_life" for o in ops)
    keys = g["args"]["intensity"]["timeline"]
    assert all(k1[0] > k0[0] for k0, k1 in zip(keys[:-1], keys[1:]))
    t_land = int(0.5 * 24) / 24
    assert max(v for t, v in keys if t_land <= t < t_land + 3 / 24 + 1e-6) < 0.05      # the landing frames are not lit up
    assert max(v for _, v in keys) > 0.15 and keys[0][1] == 0.0 and keys[-1][1] == 0.0


def test_weave_outline_shimmer_dips_the_outline_windows_and_needs_an_outline():
    pc = _piece()
    pc["compose"]["hold_life"] = ["outline_shimmer"]
    H.weave(pc, holds=HOLDS)
    assert "outline_shimmer" not in pc["plan"]["hold_life"]["effects"]          # no outline in the piece: nothing to shimmer
    pc = _piece()
    pc["compose"]["hold_life"] = ["outline_shimmer"]
    pc["compose"]["outline"] = [[0.5, 2.5], [3.0, 5.0]]
    H.weave(pc, holds=HOLDS)
    w = pc["compose"]["outline"]
    assert len(w) > 2 and all(b > a for a, b in w)
    assert w[0][0] == 0.5 and w[-1][1] == 5.0                                   # starts and ends fully on
    dips = [(b0, a1) for (a0, b0), (a1, b1) in zip(w[:-1], w[1:]) if 0 < a1 - b0 < 0.2]
    assert dips and all(1.9 / 24 < d1 - d0 < 3.2 / 24 for d0, d1 in dips)


def test_weave_leaves_ltx_owned_words_alone_and_remaps_line_pass_drawers_to_the_matte():
    pc = _piece()
    pc["type"]["words"][1].update(render="ltx", t0=3.0, t1=5.5)
    pc["type"]["words"][0].update(render="flat", t0=0.5, t1=3.0)
    pc["compose"]["hold_life"] = ["pulse"]
    H.weave(pc, holds=HOLDS)
    hl = next(o for o in pc["compose"]["default"]["ops"] if o["op"] == "hold_life")
    assert [w["src"] for w in hl["args"]["plan"]["words"]] == [0]
    pc2 = _piece()
    pc2["compose"]["default"]["ops"][3]["args"]["type_a"] = "$type_line"
    pc2["compose"]["hold_life"] = ["pulse"]
    H.weave(pc2, holds=HOLDS)
    hl2 = [o for o in pc2["compose"]["default"]["ops"] if o["op"] == "hold_life"]
    assert hl2 and hl2[0]["args"]["ref"] in ("$type_matte", "$wm")          # 2026-10-30: a line-pass drawer is remapped to the matte buffer (what recipe_eval's legibility pass redraws it from) and gets hold life


def test_word_text_starting_with_dollar_never_becomes_a_buffer_reference():
    pc = _piece(words=("$500 CASH", "NEON"))
    pc["compose"]["hold_life"] = ["pulse"]
    H.weave(pc, holds=HOLDS)
    acc = set()
    RE.refs([o.get("args", {}) for o in pc["compose"]["default"]["ops"]], acc)
    assert not any(x.startswith("500") for x in acc)


# ------------------------------------------------------------------------------------------------------------- legibility_pass twin + a tiny end-to-end compose
def test_legibility_pass_twins_the_op_onto_the_beauty_face(tmp_path):
    (tmp_path / "beauty").mkdir()
    pc = _piece()
    pc["plan"]["flow_wake"] = {"holds": [list(h) for h in HOLDS]}
    pc["compose"]["hold_life"] = ["pulse"]
    H.weave(pc, holds=HOLDS)
    rec = RE.build_recipe(pc, "s1", type_dir=str(tmp_path))
    ops = rec["ops"]
    hls = [o for o in ops if o["op"] == "hold_life"]
    assert len(hls) == 2 and hls[0]["args"]["ref"] == hls[1]["args"]["ref"] == "$ta"
    assert hls[0]["out"] == "hl0" and hls[1]["out"] == "hl0_face" and hls[1]["args"]["img"] == "$tface"
    sw = next(o for o in ops if o["op"] == "sandwich")
    assert sw["args"]["type_a"] == "$hl0"
    assert sw["args"]["type_rgb"] == "$hl0_face"


def _write_type(tmp, a, n):
    beauty = np.stack([np.full_like(a, 0.15), np.full_like(a, 0.85), np.full_like(a, 0.95)], -1)
    for k, im in (("beauty", beauty * a[..., None] + 0.02 * (1 - a[..., None])), ("matte", a), ("line", a)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        u8 = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        if u8.ndim == 3:
            u8 = u8[..., ::-1]
        for i in range(n):
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), u8)


def test_end_to_end_through_run_compose(tmp_path):
    w, h, n = 216, 384, 60
    a = word_alpha("DREAMING", w, h, 0.9)
    assert H.letter_map(a)["m"] == 8
    _write_type(tmp_path / "t", a, n)
    pc = _piece(n=n)
    pc["sizes"] = {"compose": [w, h]}
    pc["compose"]["default"]["ops"][0]["args"]["size"] = [w, h]
    pc["plan"]["flow_wake"] = {"holds": [[0.5, 2.3]]}
    pc["compose"]["hold_life"] = ["letter_flicker"]
    pc["type"]["words"] = [dict(text="DREAMING")]
    H.weave(pc, holds=[(0.5, 2.3)])
    plate = np.full((n, h, w, 3), 102, np.uint8)
    matte = np.zeros((n, h, w), np.float32)
    on, st = RE.run_compose(pc, "s1", type_dir=str(tmp_path / "t"), footage=plate, matte=matte, backend=os.environ["FUSION_BACKEND"], threads=0)
    pc0 = copy.deepcopy(pc)
    pc0["compose"]["hold_life"] = False
    H.weave(pc0, holds=[(0.5, 2.3)])
    off, _ = RE.run_compose(pc0, "s1", type_dir=str(tmp_path / "t"), footage=plate, matte=matte, backend=os.environ["FUSION_BACKEND"], threads=0)
    on, off = np.asarray(on, np.int16), np.asarray(off, np.int16)
    wp = pc["plan"]["hold_life"]
    diff = [int(np.abs(on[i] - off[i]).max()) for i in range(n)]
    assert max(diff[12:15]) == 0                                          # the landing frames are the plain word
    assert max(diff[:12]) == 0 and max(diff[55:]) <= 2                    # nothing outside the hold (<= 2/255: legible_color's colour smoothing lags the last blink)
    assert sum(1 for d in diff[15:55] if d > 20) >= 3                     # letters blink inside it
    assert wp["effects"] == ["letter_flicker"]
