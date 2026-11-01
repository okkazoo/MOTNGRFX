"""Run-time legibility floors (2026-11-01, QC of public renders): (1) occlusion floor: a word the subject hides by > 25 % (or one letter < 50 %
visible) for most of its hold comes in front whole, on every subject-over-type draw incl. the pure-type `over` pairs and opted-out sandwiches;
(2) readable-size floor: type_face scales a word under the cap floor and thickens hairline strokes inside the holds; (3) contrast floor:
legible_color re-chooses the ink against the ring around the letters when it fails there.
Run: python -m pytest --rootdir=tests -q -p no:cacheprovider tests/test_legibility_floor.py     (CPU, synthetic data)"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import legible_color as LC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

FPS = 24
W, H = 120, 48
N = 96
HOLD = (1.5, 3.0)
LETTERS = [(6, 12, 22, 36), (30, 12, 46, 36), (54, 12, 70, 36), (78, 12, 94, 36), (100, 12, 116, 36)]     # five letters: "WRONG"


def five():
    ta = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in LETTERS:
        ta[y0:y1, x0:x1] = 1.0
    return ta


def cover(boxes, frames=range(N)):
    m = np.zeros((N, H, W), np.float32)
    for f in frames:
        for x0, y0, x1, y1 in boxes:
            m[f, y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 1.0
    return m


# ------------------------------------------------------------------------------------------------ 1 occlusion floor
def test_one_hidden_letter_brings_the_whole_word_in_front_with_letters_off():
    ta, m = five(), cover([LETTERS[2]])                   # "WR NG": the O hidden all the hold (20 % of the word)
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, floor=True, letters=False)
    h = plan["holds"][0]
    assert h["floor"] and len(h["boxes"]) == 5             # whole word, not just the O
    assert 0.15 < h["occ"] < 0.3
    plan0 = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, letters=False)       # no floor + letters off = nothing
    assert plan0["holds"][0]["boxes"] == [] and not plan0["holds"][0]["floor"]


def test_word_share_over_25_percent_fires_but_a_brief_or_light_cover_does_not():
    ta = five()
    half = np.zeros((N, H, W), np.float32)
    half[:, 12:24, :] = 1.0                                # top half of every letter covered: 50 % of the word, no letter fully gone
    plan = C.legible_plan(lambda f: ta, lambda f: half[f], [HOLD], fps=FPS, floor=True, letters=False)
    assert plan["holds"][0]["floor"] and len(plan["holds"][0]["boxes"]) == 5
    thin = np.zeros((N, H, W), np.float32)
    thin[:, 12:14, :] = 1.0                                # a sliver (~8 %, under the 15 % word floor of 2026-11-02): reads, no floor
    assert not C.legible_plan(lambda f: ta, lambda f: thin[f], [HOLD], fps=FPS, floor=True, letters=False)["holds"][0]["floor"]
    brief = cover(LETTERS, frames=range(36, 44))           # whole word hidden for 8 of 36 hold frames (22 %): within the 25 % the rule tolerates
    assert not C.legible_plan(lambda f: ta, lambda f: brief[f], [HOLD], fps=FPS, floor=True, letters=False)["holds"][0]["floor"]


def test_legible_op_with_floor_clears_the_whole_word():
    ta, m = five(), cover([LETTERS[2]])
    plan = C.legible_plan(lambda f: ta, lambda f: m[f], [HOLD], fps=FPS, floor=True, letters=False)
    full = cover(LETTERS)[50]
    out = np.asarray(C.legible(ta, full, holds=[HOLD], t=50 / FPS, plan=plan))
    assert float((out * ta).sum()) / ta.sum() < 0.05       # every letter's occluder cleared mid-hold


def _piece(ops, comp=None):
    p = {"fps": FPS, "shots": [{"id": "s1", "frames": N}], "sizes": {"compose": [W, H]}, "type_passes": ["matte"],
         "compose": dict({"default": {"ops": ops}}, **(comp or {})), "plan": {"flow_wake": {"holds": [list(HOLD)]}}}
    return p


def pure_type_ops():
    """the own-video pure-type chain (tools/app/own_video_v2_retime_box.py --pure-type): type drawn by `over`, the subject put back by its matte."""
    return [{"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "guide": "$footage", "size": [W, H]}},
            {"op": "over", "out": "sand_t", "args": {"base": "$footage", "fg": [1.0, 1.0, 1.0], "alpha": "$type_matte"}},
            {"op": "over", "out": "image", "args": {"base": "$sand_t", "fg": "$footage", "alpha": "$m"}}]


def test_guard_ops_wraps_the_pure_type_over_pair():
    ops = RE.guard_ops(_piece(pure_type_ops()), RE.subst(pure_type_ops(), {"$matte": "$s1_matte"}))
    names = [o["op"] for o in ops]
    assert names == ["upscale_matte", "over", "legible", "over"], names
    leg = ops[2]["args"]
    assert leg["type_a"] == "$type_matte" and leg["matte"] == "$m" and leg["letters"] is False and leg["floor"]
    assert ops[3]["args"]["alpha"] == "$" + ops[2]["out"]
    off = RE.guard_ops(_piece(pure_type_ops(), {"legible_floor": False}), pure_type_ops())
    assert [o["op"] for o in off] == ["upscale_matte", "over", "over"]


def test_guard_ops_guards_the_window_cut_subject_matte():
    ops = [{"op": "window_cut", "out": "image", "args": {"a": [0.9, 0.9, 0.9], "b": "$footage", "type_a": "$type_matte", "matte_b": "$s1_matte",
                                                         "mode": "other", "subject": "inside", "t": "$t"}}]
    out = RE.guard_ops(_piece(ops), ops)
    assert [o["op"] for o in out] == ["legible", "window_cut"] and out[1]["args"]["matte_b"] == "$" + out[0]["out"]
    assert out[0]["args"]["letters"] is False and out[0]["args"]["type_a"] == "$type_matte"
    ops[0]["args"]["subject"] = "none"
    ops[0]["args"]["matte_b"] = "$s1_matte"
    assert [o["op"] for o in RE.guard_ops(_piece(ops), ops)] == ["window_cut"]


def test_type_over_type_is_not_a_matte_pair():
    ops = [{"op": "over", "out": "a", "args": {"base": "$footage", "fg": [1, 1, 1], "alpha": "$type_matte"}},
           {"op": "over", "out": "b", "args": {"base": "$a", "fg": [0, 0, 0], "alpha": "$type_line"}}]
    assert [o["op"] for o in RE.guard_ops(_piece(ops), ops)] == ["over", "over"]


def test_pure_type_compose_shows_the_hidden_word():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "matte"
        d.mkdir()
        for i in range(N):
            cv2.imwrite(str(d / f"{i:05d}.png"), (five() * 255).astype(np.uint8))
        foot = np.zeros((N, H, W, 3), np.uint8)
        matte = (cover([LETTERS[2]]) * 255).astype(np.uint8)
        out, st = RE.run_compose(_piece(pure_type_ops()), "s1", type_dir=tmp, footage=foot, matte=matte, backend="numpy")
        out = out.numpy() if hasattr(out, "numpy") else out
        x0, y0, x1, y1 = LETTERS[2]
        assert out[54, y0 + 2:y1 - 2, x0 + 2:x1 - 2].mean() > 230        # the O reads mid-hold
        assert out[5, y0 + 2:y1 - 2, x0 + 2:x1 - 2].mean() < 10          # before the hold the matte still hides it
        o2, _ = RE.run_compose(_piece(pure_type_ops(), {"legible_floor": False}), "s1", type_dir=tmp, footage=foot, matte=matte, backend="numpy")
        o2 = o2.numpy() if hasattr(o2, "numpy") else o2
        assert o2[54, y0 + 2:y1 - 2, x0 + 2:x1 - 2].mean() < 10


# ------------------------------------------------------------------------------------------------ 2 readable-size floor
def _word(h=540, w=960, cap=0.05, stroke=2, n=7, dots=False):
    a = np.zeros((h, w), np.float32)
    ch = int(round(cap * h))
    x = w // 2 - n * ch // 2
    y0 = h // 2 - ch // 2
    for _ in range(n):
        if dots:
            for yy in range(y0, y0 + ch, 6):
                for xx in range(x, x + ch // 2, 6):
                    cv2.circle(a, (xx, yy), 2, 1.0, -1)
        else:
            cv2.rectangle(a, (x, y0), (x + ch // 2, y0 + ch), 1.0, stroke)
        x += ch
    return a


def test_type_floor_scales_a_small_word_to_the_cap_floor():
    a = _word(cap=0.05, stroke=-1)                         # filled letters: the cap is exactly 5 %
    col = np.ones(a.shape + (3,), np.float32)
    _c, a2, info = C.type_floor(col, a, 1.0, min_stroke=None, min_cap=0.09)
    assert 1.6 < info["scale"] < 1.9
    _c, _a, info2 = C.type_floor(col, a2, 1.0, min_stroke=None, min_cap=0.09)
    assert info2["cap"] >= 0.085 and "scale" not in info2
    _c, a0, _ = C.type_floor(col, a, 0.0, min_stroke=None, min_cap=0.09)          # outside the hold: untouched
    assert np.array_equal(a0, a)


def test_type_floor_never_pushes_a_long_word_out_of_the_frame():
    a = _word(cap=0.05, stroke=-1, n=17)                   # 17 letters at 5 % cap fill ~80 % of the width already
    _c, a2, info = C.type_floor(np.ones(a.shape + (3,), np.float32), a, 1.0, min_stroke=None, min_cap=0.09)
    xs = np.nonzero(a2.max(0) > 0.5)[0]
    assert xs.max() - xs.min() <= C.TF_MAX_BOX * a.shape[1] + 2 and xs.min() > 0 and xs.max() < a.shape[1] - 1


def test_type_floor_thickens_a_hairline_and_leaves_bold_type_alone():
    col = np.ones((540, 960, 3), np.float32)
    thin = _word(cap=0.11, stroke=1)
    _c, a2, info = C.type_floor(col, thin, 1.0, min_stroke=C.TF_MIN_STROKE, min_cap=None)
    assert info["stroke_px"] < 3.0 and info["grow_px"] > 0.5
    assert C._stroke_px(a2 > 0.5) >= 0.85 * C.TF_MIN_STROKE * 540
    bold = _word(cap=0.11, stroke=14)
    _c, b2, info = C.type_floor(col, bold, 1.0, min_stroke=C.TF_MIN_STROKE, min_cap=None)
    assert "grow_px" not in info and np.array_equal(b2, bold)


def test_type_floor_reads_a_dot_face_as_letters_not_dots():
    a = _word(cap=0.12, dots=True)
    _c, _a, info = C.type_floor(np.ones(a.shape + (3,), np.float32), a, 1.0, min_stroke=None, min_cap=0.09)
    assert info["cap"] > 0.09 and "scale" not in info


def test_floor_ops_sets_the_type_face_floor():
    ops = [{"op": "type_face", "out": "_tf4", "args": {"beauty": "$type_beauty", "type_a": "$type_matte"}}]
    out = RE.floor_ops(_piece(ops), ops)
    a = out[0]["args"]
    assert a["holds"] == [list(HOLD)] and a["t"] == "$t" and a["min_cap"] == C.TF_MIN_CAP and a["min_stroke"] == C.TF_MIN_STROKE
    ops2 = [{"op": "type_face", "out": "_tf4", "args": {"beauty": "$type_beauty", "type_a": "$type_matte"}}]
    assert "holds" not in RE.floor_ops(_piece(ops2, {"legible_floor": False}), ops2)[0]["args"]


def test_type_face_applies_the_floor_only_in_the_hold():
    a = _word(cap=0.11, stroke=1)
    beauty = np.zeros(a.shape + (3,), np.float32) + a[..., None]
    out_in = C.type_face(beauty, a, bg=[0, 0, 0], holds=[HOLD], t=2.0, min_stroke=C.TF_MIN_STROKE, min_cap=C.TF_MIN_CAP)
    out_out = C.type_face(beauty, a, bg=[0, 0, 0], holds=[HOLD], t=0.5, min_stroke=C.TF_MIN_STROKE, min_cap=C.TF_MIN_CAP)
    assert C._stroke_px(out_in[..., 3] > 0.5) >= 0.85 * C.TF_MIN_STROKE * 540 > C._stroke_px(a > 0.5)
    assert np.allclose(out_out[..., 3], a)


# ------------------------------------------------------------------------------------------------ 3 contrast floor
def test_ring_floor_rechooses_dark_ink_on_a_dark_surround():
    """R5 ASC/END: light window behind the letters (counted double), navy flat ground around them: the combined measure picks dark ink."""
    yl = np.full((60, 120), 0.012, np.float32)            # navy ground
    core = np.zeros((60, 120), bool)
    core[20:40, 30:90] = True
    yl[core] = 0.45                                       # light plate seen through the letters
    cl = {"core": core}
    dark = LC.srgb_to_lin(np.full(3, 0.06, np.float32))
    ch = dict(kind="kept", lin=dark, cover=0.9, support=None, resid=0.1)     # what the combined measure kept (QC: "how": "kept", #100d04)
    ring = LC._ring_hist(cl, yl)
    ch2 = LC.ring_floor(ch, ring, dark, LC.srgb_to_lin(np.full(3, 0.4, np.float32)), 3.0)
    y2 = float(LC.rel_lum(ch2["lin"]))
    assert ch2.get("ring_floor") and LC.contrast_ratio(y2, 0.012) >= 3.0
    ok = dict(kind="design", lin=LC.srgb_to_lin(np.full(3, 0.95, np.float32)), cover=1.0, support=None, resid=0.0)
    assert LC.ring_floor(ok, ring, dark, dark, 3.0) is ok       # light ink on navy passes: untouched


def test_legible_color_op_reads_against_the_surround():
    h, w = 120, 240
    plate = np.full((h, w, 3), 0.1, np.float32)          # dark navy-ish ground
    a = np.zeros((h, w), np.float32)
    a[40:80, 60:180] = 1.0
    plate[a > 0.5] = 0.85                                 # light window behind the letters
    face = np.full((h, w, 3), 0.05, np.float32)           # designed near-black ink
    LC.reset("t_ring")
    out = None
    for f in range(30):
        out = np.asarray(LC.legible_color(face, a, plate, t=f / 24.0, frame_idx=f, state_key="t_ring", smooth=0.0))
    ink = out[60, 120, :3]
    y_ink = float(LC.rel_lum(LC.srgb_to_lin(ink)))
    y_bg = float(LC.rel_lum(LC.srgb_to_lin(np.full(3, 0.1, np.float32))))
    assert LC.contrast_ratio(y_ink, y_bg) >= 3.0
