"""The two lego blocks that went live on 2026-10-27: strips_persist_across_cuts (P18) and edge_run_small_type (P20).
  * strips_persist_across_cuts: layout_shapes shapes carry a `pose` (keyframes eased 0.25 s from each cut downbeat, per-cut targets from the
    shot's first frames), the type follows through `rigid_pose` (tools/fusion/strip_resnap.py)
  * edge_run_small_type: `edge_run_type` sets small type along a per-frame tracked footage edge, glyph baseline on the polyline, cap height
    >= 26 px at 640 (tools/fusion/edge_run.py)
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_blocks_live.py -q   (CPU only, synthetic frames, ~20 s)"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import compose  # noqa: E402
from tools.fusion import edge_run as ER  # noqa: E402
from tools.fusion import hold_layouts as hl  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import strip_resnap as SR  # noqa: E402

TWO = ("strips_persist_across_cuts", "edge_run_small_type")


# ----------------------------------------------------------------------------------------------- registry
@pytest.mark.parametrize("name", TWO)
def test_both_blocks_are_live_and_well_formed(name):
    b = bk.BLOCKS[name]
    assert b["status"] == "live" and not b.get("missing")
    assert bk.validate_block(name) == []
    reg = compose._ops_registry()
    assert all(c["op"] in reg for c in b["slots"]["compose"]), "every compose template of a live block is a registered op"


def test_ops_are_registered_and_the_old_shape_api_still_works():
    reg = compose._ops_registry()
    assert {"layout_shapes", "rigid_pose", "edge_run_type"} <= set(reg)
    m = compose.shape_mask({"kind": "rect", "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.2, "angle": 0.0}, 64, 32)      # (sh, h, w) without t: as before
    assert m.shape == (64, 32) and 0.1 < m.mean() < 0.2


def test_the_strips_block_has_a_persist_cuts_param():
    p = bk.resolve_params("strips_persist_across_cuts", 3, "x")
    assert 0 <= p["persist_cuts"] <= 2 and 0.0 <= p["angle_deg"] <= 10.0


# ----------------------------------------------------------------------------------------------- pose maths
def test_pose_at_eases_between_keys_and_clamps_outside():
    ks = [[1.0, 0, 0, 0], [1.25, 0, 0.1, 8.0]]
    assert SR.pose_at(ks, 0.0) == (0.0, 0.0, 0.0) and SR.pose_at(ks, 9.0) == (0.0, 0.1, 8.0)
    dx, dy, da = SR.pose_at(ks, 1.125)
    assert abs(dy - 0.05) < 1e-9 and abs(da - 4.0) < 1e-9                       # smoothstep(0.5) = 0.5
    assert SR.pose_at(ks, 1.0625)[2] < 2.0 < SR.pose_at(ks, 1.1875)[2]


def test_move_xy_rotates_clockwise_on_screen():
    p = SR.move_xy([[110.0, 100.0]], 0, 0, 90.0, 100.0, 100.0)[0]               # the point right of the pivot goes DOWN (y down) when rotated clockwise
    assert abs(p[0] - 100.0) < 1e-6 and abs(p[1] - 110.0) < 1e-6


def _strip(pose=None, t0=0.0, t1=10.0):
    sh = {"kind": "rect", "cx": 0.5, "cy": 0.5, "w": 1.9, "h": 0.2, "angle": 0.0, "color": [1.0, 0.0, 0.0], "t0": t0, "t1": t1}
    if pose:
        sh["pose"] = pose
    return sh


def _rows(img, col):
    c = np.where(img[:, col, 0] > 0.5)[0]
    return float(c.mean())


def test_layout_shapes_moves_and_tilts_the_strip_on_its_pose_keys():
    base = np.zeros((200, 100, 3), np.float32)
    pose = {"pivot": [0.5, 0.5], "ease": "smooth", "keys": [[1.0, 0, 0, 0], [1.25, 0, -0.2, 10.0]]}
    shapes = [_strip(pose)]
    before = compose.layout_shapes(base, shapes, t=0.9, fade=0.01)
    after = compose.layout_shapes(base, shapes, t=1.5, fade=0.01)
    plain = compose.layout_shapes(base, [_strip()], t=0.9, fade=0.01)
    assert np.allclose(before, plain), "before the cut the strip is where its layout put it"
    assert abs(_rows(before, 50) - 100) < 1.0
    assert _rows(after, 50) < 100 - 25, "dy -0.2 of the height moves the strip up"
    assert _rows(after, 90) > _rows(after, 10) + 8, "10 degrees clockwise: the right end goes down"
    mid = compose.layout_shapes(base, shapes, t=1.125, fade=0.01)
    assert _rows(after, 50) < _rows(mid, 50) < _rows(before, 50), "eased in over the 0.25 s"


def test_rigid_pose_op_follows_the_same_move():
    m = np.zeros((100, 200), np.float32)
    m[45:55, 90:110] = 1.0
    assert np.allclose(SR.rigid_pose(m), m)
    r = SR.rigid_pose(m, dx=0.25, dy=0.1, angle=0.0)
    ys, xs = np.nonzero(r > 0.5)
    assert abs(xs.mean() - (99.5 + 50)) < 1.5 and abs(ys.mean() - (49.5 + 10)) < 1.5
    rot = SR.rigid_pose(m, angle=90.0)
    ys, xs = np.nonzero(rot > 0.5)
    assert ys.max() - ys.min() > 15 and xs.max() - xs.min() < 15, "a flat bar turned 90 degrees stands up"
    assert compose._ops_registry()["rigid_pose"] is SR.rigid_pose


# ----------------------------------------------------------------------------------------------- the edge analysis
def _edge_img(angle, y=0.6, h=320, w=180):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    line = y * h + (xx - w / 2) * math.tan(math.radians(angle))
    return (1.0 / (1.0 + np.exp(np.clip(-(yy - line) / 2.0, -30, 30)))).astype(np.float32)


@pytest.mark.parametrize("angle", [7.0, -5.0, 0.0])
def test_dominant_edge_angle_and_row_follow_the_picture(angle):
    t = SR.shot_target([_edge_img(angle)], None, angle_max=10.0, strip_h=0.3)
    assert abs(t["angle"] - angle) < 1.0 and t["strength"] > 0.5
    assert abs(t["edge_y"] - 0.6) < 0.03
    assert abs(t["cy"] - (0.6 - 0.15)) < 0.03                                   # the strip's lower boundary sits on the edge


def test_the_target_angle_is_clamped_to_the_block_param():
    assert SR.shot_target([_edge_img(25.0)], None, angle_max=6.0)["angle"] == pytest.approx(6.0, abs=0.01)
    assert SR.shot_target([np.full((64, 64), 0.5, np.float32)], None)["angle"] == 0.0            # no edges: the strip goes flat


def test_the_matte_outline_counts_as_an_edge():
    flat = np.full((320, 180), 0.5, np.float32)
    matte = _edge_img(6.0)
    t = SR.shot_target([flat], [matte], angle_max=10.0)
    assert abs(t["angle"] - 6.0) < 1.5


def test_pose_keys_start_on_the_shot_pose_and_ease_from_each_cut():
    base = {"cy": 0.32, "angle": -2.0}
    tg = {"s1": dict(cy=0.4, angle=1.0), "s2": dict(cy=0.3, angle=5.0), "s3": dict(cy=0.5, angle=-1.0)}
    cuts = [(2.0, "s2"), (4.0, "s3")]
    k = SR.pose_keys(base, 2.3, 4.6, cuts, tg, first="s1")
    assert k[0][0] == 2.3 and k[0][1:] == [0.0, pytest.approx(-0.02, abs=1e-4), 7.0], "appears in s2: stands on s2's pose, no ease"
    assert [4.0, 0.0, pytest.approx(-0.02, abs=1e-4), 7.0] in [[a, b, c, d] for a, b, c, d in k]
    assert any(abs(x[0] - 4.25) < 1e-9 and abs(x[2] - 0.18) < 1e-3 and abs(x[3] - 1.0) < 1e-6 for x in k), "0.25 s ease to s3's pose"
    assert k[-1][0] == 4.6
    assert [x[0] for x in k] == sorted(x[0] for x in k)


# ----------------------------------------------------------------------------------------------- weave on a synthetic piece
def _word(text, i, strips, a, b):
    rec = hl.build("solid_strips" if strips else "small_line", 3, f"t{i}", "9:16")
    w = {"text": text, "lines": [text], "font": "Silkscreen-Regular.ttf", "module": "lanes", "t0": a, "t1": b, "form_s": 0.5, "exit_s": 0.5,
         "hold_layout": rec, "placement": {}}
    return w, rec


def _piece(edge=False):
    w0, r0 = _word("CONCRETE", 0, True, 1.0, 6.0)
    w1, r1 = _word("BENDS", 1, not True, 7.0, 9.0)
    shapes = []
    for sh in r0["shapes"]:
        d = copy.deepcopy(sh)
        d["color"] = [0.2, 0.3, 0.8]
        d["t0"], d["t1"] = 1.0, 5.6
        shapes.append(d)
    ops = [{"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m"}},
           {"op": "flow_wake", "out": "wl", "args": {"type_a": "$type_line", "flow": "$flow", "matte": "$flow_m"}},
           {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [90, 160], "guide": "$footage"}},
           {"op": "layout_shapes", "out": "plate", "args": {"img": "$footage", "shapes": shapes, "t": "$t", "fade": 0.3, "matte": "$m"}},
           {"op": "tint_type", "out": "tcol", "args": {"footage": "$footage", "type_a": "$wl", "mode": "sample"}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$plate", "matte": "$m", "type_a": "$wm", "type_rgb": "$tcol"}},
           {"op": "grade", "out": "image", "args": {"frame": "$image", "gain": 1.0}},
           {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.3}}]
    blocks = [{"name": "strips_persist_across_cuts", "params": {"angle_deg": 8.0, "persist_s": 0.6, "persist_cuts": 1}, "lines": [0], "status": "live"},
              {"name": "edge_run_small_type", "params": {"cap_frac": 0.05, "min_px_at_640": 28}, "lines": [1], "status": "live"}]
    piece = {"id": "T1", "fps": 24, "audio": {"cut": 12.0}, "shots": [{"id": "s1", "span": 72}, {"id": "s2", "span": 72}, {"id": "s3", "span": 72},
                                                                    {"id": "s4", "span": 72}],
             "compose": {"crossfade": 0, "size": [90, 160], "default": {"ops": ops}},
             "type": {"words": [w0, w1], "plates": []},
             "plan": {"blocks": blocks, "timing": [{"text": "CONCRETE", "hit": 1.8, "release": 4.5, "end": 5.0},
                                                   {"text": "BENDS", "hit": 7.5, "release": 8.6, "end": 9.0}]}}
    return piece


TG = {"s1": dict(cy=0.30, angle=2.0), "s2": dict(cy=0.26, angle=5.0), "s3": dict(cy=0.40, angle=-1.0), "s4": dict(cy=0.35, angle=0.0)}


def test_weave_poses_the_strip_and_makes_the_type_follow():
    p = _piece()
    SR.weave(p, None, None, targets=TG)
    info = p["plan"]["strip_resnap"]
    assert [round(c[0], 3) for c in info["cuts"]] == [3.0, 6.0, 9.0] and info["groups"][0]["moved"]
    ops = p["compose"]["default"]["ops"]
    names = [o["op"] for o in ops]
    assert names[:4] == ["flow_wake", "flow_wake", "rigid_pose", "rigid_pose"], "the type follows right after the flow_wake outputs"
    assert {o["out"] for o in ops[2:4]} == {"wm", "wl"}
    ls = next(o for o in ops if o["op"] == "layout_shapes")
    assert all("pose" in s for s in ls["args"]["shapes"] if s["kind"] == "rect")
    s0 = ls["args"]["shapes"][0]
    assert s0["pose"]["keys"][0][0] == 1.0 and s0["t1"] > 6.0, "the strip lives on past the cut at 6.0 (persist_cuts 1)"
    keys = s0["pose"]["keys"]
    assert any(abs(k[0] - 3.25) < 1e-6 for k in keys) and any(abs(k[0] - 6.25) < 1e-6 for k in keys), "eased in 0.25 s after the cuts at 3.0 and 6.0"
    # the same keyframes drive the type, only while the word is on (back to the identity after its end)
    tl = ops[2]["args"]["angle"]["timeline"]
    assert tl[-1][1] == 0.0 and tl[-1][0] <= 5.1
    again = copy.deepcopy(p)
    SR.weave(again, None, None, targets=TG)
    assert again["compose"] == p["compose"], "idempotent"


def test_weave_evaluates_in_the_recipe_machinery():
    p = _piece()
    SR.weave(p, None, None, targets=TG)
    ops = p["compose"]["default"]["ops"]
    fps = 24
    ctx = {}
    pf = int(round(3.5 * fps))                                            # 0.5 s after the cut at 3.0
    a = {k: RE.resolve(v, ctx, {}, 0.0, pf, fps) for k, v in ops[2]["args"].items() if k != "img"}
    ls = ops[[o["op"] for o in ops].index("layout_shapes")]
    band = ls["args"]["shapes"][0]
    assert a["angle"] == pytest.approx(TG["s2"]["angle"] - band.get("angle", 0.0), abs=1e-6), "the type turns by the strip's own delta"
    assert a["dy"] == pytest.approx(TG["s2"]["cy"] - band["cy"], abs=1e-6)
    img = np.zeros((160, 90, 3), np.float32)
    out = compose.layout_shapes(img, ls["args"]["shapes"], t=3.5, fade=0.01)
    assert out.max() > 0.1, "the posed strip is on screen after the cut"


def test_weave_wraps_the_type_inputs_when_the_chain_has_no_flow_wake():
    p = _piece()
    p["compose"]["default"]["ops"] = [o for o in p["compose"]["default"]["ops"] if o["op"] != "flow_wake"]
    p["compose"]["default"]["ops"][1]["args"]["type_a"] = "$type_matte"
    p["compose"]["default"]["ops"].insert(0, {"op": "fill", "out": "wm", "args": {"like": "$type_matte"}})
    SR.weave(p, None, None, targets=TG)
    ops = p["compose"]["default"]["ops"]
    assert ops[0]["op"] == "rigid_pose" and ops[0]["args"]["img"] == "$type_matte"
    assert any(o["op"] == "fill" and o["args"]["like"] == "$_sr_type_matte" for o in ops), "later ops read the moved copies"


def test_weave_without_the_block_or_without_cuts_changes_nothing():
    p = _piece()
    p["plan"]["blocks"] = [b for b in p["plan"]["blocks"] if b["name"] != "strips_persist_across_cuts"]
    q = copy.deepcopy(p)
    SR.weave(q, None, None, targets=TG)
    assert q["compose"] == p["compose"]
    one = _piece()
    one["shots"] = one["shots"][:1]
    SR.weave(one, None, None, targets=TG)
    assert "skipped" in one["plan"]["strip_resnap"] and not any(o["op"] == "rigid_pose" for o in one["compose"]["default"]["ops"])


# ----------------------------------------------------------------------------------------------- edge_run_type
def _river(h=640, w=360, shift=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = 300 + shift + 0.12 * (xx - w / 2) + 14 * np.sin(xx / 80.0)
    g = 0.5 + 0.3 * np.tanh((yy - edge) / 4.0)
    return np.repeat(cv2.GaussianBlur(g, (0, 0), 1.0)[..., None], 3, -1).astype(np.float32), edge


def test_cap_height_floor_is_26_px_at_640():
    assert ER.cap_px_for(640, 0.04, 26) >= 26 and ER.cap_px_for(640, 0.05, 26) == pytest.approx(32.0)
    assert ER.cap_px_for(640, 0.02, 26) == pytest.approx(26.0)                  # a smaller cap_frac never goes under the floor
    assert ER.cap_px_for(320, 0.02, 26) == pytest.approx(13.0)
    assert hl.MIN_CAP_FRAC * 640 >= 25.6


def test_edge_run_sets_legible_type_along_the_tracked_edge():
    img, edge = _river()
    kw = dict(text="BENDS", font="GeistPixel[ELSH].ttf", t0=0.0, t1=5.0, fade=0.2, reveal_s=0.2, cap_frac=0.04, min_px_at_640=26, state_key="t_er1",
              near=(0.5, 0.5))
    a0, info = ER.edge_run_type(img, t=1.0, frame_idx=10, return_info=True, **kw)
    assert a0.shape == (640, 360) and a0.dtype == np.float32 and a0.max() > 0.9
    assert info["cap_px"] >= 26.0
    ys, xs = np.nonzero(a0 > 0.5)
    assert len(xs) > 300
    # the ink stands on the edge: every ink pixel is within two cap heights above it (letters sit on the baseline = the edge)
    rel = edge[ys, xs] - ys
    assert float(np.median(rel)) > 4.0 and float(rel.max()) < 3.2 * info["cap_px"] and float(rel.min()) > -2 * info["cap_px"]
    # tilt of the word follows the edge (slope ~ 0.12 + the sine): the glyph row centres are not all level
    w = info["window"]
    assert w.shape == (ER.M_PTS, 2) and float(w[:, 0].max() - w[:, 0].min()) > 0.8 * info["length"] * 0.9
    # next frame, the edge has moved 14 px down: the tracked window follows it, same stretch (it did not jump to another feature)
    img2, edge2 = _river(shift=14.0)
    a1, info2 = ER.edge_run_type(img2, t=1.0 + 1 / 24, frame_idx=11, return_info=True, **kw)
    d = info2["window"].mean(0) - w.mean(0)
    assert 3.0 < d[1] < 20.0 and abs(d[0]) < 25.0
    assert a1.max() > 0.9


def test_edge_run_letters_are_not_clipped_and_stand_cap_px_tall():
    h, w = 640, 360
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    g = cv2.GaussianBlur(0.5 + 0.3 * np.tanh((yy - 330.0) / 4.0), (0, 0), 1.0)           # a level edge: letters stand upright
    img = np.repeat(g[..., None], 3, -1).astype(np.float32)
    a, info = ER.edge_run_type(img, "HEH", "Silkscreen-Regular.ttf", t=1.0, t0=0.0, t1=4.0, reveal_s=0.1, cap_frac=0.05, min_px_at_640=26,
                               near=(0.5, 0.5), frame_idx=1, state_key="t_er5", return_info=True)
    ys, xs = np.nonzero(a > 0.5)
    assert ys.max() - ys.min() + 1 >= 0.9 * info["cap_px"] and ys.max() - ys.min() + 1 <= 1.2 * info["cap_px"]


def test_edge_run_outside_its_window_is_empty_and_drops_the_track():
    img, _ = _river()
    kw = dict(text="BENDS", font="GeistPixel[ELSH].ttf", t0=2.0, t1=3.0, fade=0.2, state_key="t_er2")
    assert ER.edge_run_type(img, t=1.0, frame_idx=1, **kw).max() == 0.0
    assert ER.edge_run_type(img, t=3.5, frame_idx=2, **kw).max() == 0.0
    assert "t_er2" not in ER._STATE


def test_edge_run_letters_are_written_on_one_after_another():
    img, _ = _river()
    kw = dict(text="BENDS", font="GeistPixel[ELSH].ttf", t0=1.0, t1=5.0, fade=0.2, reveal_s=1.0, state_key="t_er3")
    early = ER.edge_run_type(img, t=1.05, frame_idx=1, **kw)
    full = ER.edge_run_type(img, t=2.5, frame_idx=2, **kw)
    assert 0 < early.sum() < 0.6 * full.sum()


def test_edge_run_weave_adds_the_ops_before_the_closing_grade_and_hands_the_letters_over():
    p = _piece()
    ER.weave(p, None, None)
    ops = p["compose"]["default"]["ops"]
    names = [o["op"] for o in ops]
    i = names.index("edge_run_type")
    assert names[i - 1] == "sandwich" and names[i + 5:] == ["grade", "vignette"], "before the closing grade / vignette"
    a = ops[i]["args"]
    assert a["text"] == "BENDS" and a["min_px_at_640"] >= 26 and a["t0"] < 7.6 < a["t1"] and a["t"] == "$t" and a["frame_idx"] == "$frame_idx"
    assert [o["op"] for o in ops[i + 1:i + 5]] == ["blur", "mask_gate", "over", "over"]
    assert ops[i + 4]["args"]["fg"] == "$tcol" and ops[i + 3]["args"]["fg"][0] < 0.1, "dark halo under the ink in the type colour"
    gates = [o for o in ops if o["op"] == "mask_gate" and o["out"] in ("wm", "wl")]
    assert len(gates) == 2 and names.index("mask_gate") < i, "the module letters step aside right after the flow_wake outputs"
    again = copy.deepcopy(p)
    ER.weave(again, None, None)
    assert again["compose"] == p["compose"]
    assert p["plan"]["edge_run"]["lines"][0]["cap_px_at_640"] >= 26


def test_edge_run_weave_leaves_a_piece_without_small_lines_alone():
    p = _piece()
    p["plan"]["blocks"] = [b for b in p["plan"]["blocks"] if b["name"] != "edge_run_small_type"]
    q = copy.deepcopy(p)
    ER.weave(q, None, None)
    assert q["compose"] == p["compose"]


# ----------------------------------------------------------------------------------------------- the generated pieces
@pytest.mark.parametrize("pid,name", [("P18", "strips_persist_across_cuts"), ("P20", "edge_run_small_type")])
def test_generated_pieces_carry_the_live_blocks(pid, name):
    path = os.path.join(ROOT, "15_fusion", "pieces", f"{pid}.json")
    if not os.path.isfile(path):
        pytest.skip(f"{path} is local-only (gitignored)")
    p = json.load(open(path, encoding="utf-8"))
    blk = next(b for b in p["plan"]["blocks"] if b["name"] == name)
    assert blk["status"] == "live" and not blk["missing"]
    ops = p["compose"]["default"]["ops"]
    if pid == "P18":
        ls = next(o for o in ops if o["op"] == "layout_shapes")
        assert any("pose" in s for s in ls["args"]["shapes"]) and sum(1 for o in ops if o["op"] == "rigid_pose") >= 2
        assert p["plan"]["strip_resnap"]["groups"][0]["snaps"] >= 1
    else:
        er = [o for o in ops if o["op"] == "edge_run_type"]
        assert er and er[0]["args"]["min_px_at_640"] >= 26 and er[0]["args"]["cap_frac"] >= hl.MIN_CAP_FRAC
        assert all(o["op"] in compose._ops_registry() for o in ops)
