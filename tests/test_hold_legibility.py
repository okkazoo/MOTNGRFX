"""v005 hold legibility (2026-10-27): the visible type is ALWAYS the beauty pass, treatments only modulate it.
Tiny synthetic frames (96x160), numpy backend, no GPU. One test class per failure class of the v005 previews:
1 flat hull (final painted the type silhouette flat instead of the beauty), 2 window into a dark plate, 3 inverted type matte,
4 type crushed by the plate darkening, 5 the hold net (score + blend + log)."""
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
from tools.fusion import matte_treatments as MT  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W, N = 160, 96, 24
BG = (0.02, 0.03, 0.05)
FACE = np.array([0.15, 0.85, 0.95], np.float32)


def _letters():
    """a 'wireframe' word: three bars, each drawn as thin 1 px outlines. -> (matte of the wire lines, filled hull, beauty rgb)."""
    hull = np.zeros((H, W), np.float32)
    for x0 in (10, 38, 66):
        cv2.rectangle(hull, (x0, 50), (x0 + 20, 110), 1.0, -1)
    wires = np.zeros((H, W), np.float32)
    for x0 in (10, 38, 66):
        for y in range(50, 111, 6):
            cv2.line(wires, (x0, y), (x0 + 20, y), 1.0, 1)
        cv2.rectangle(wires, (x0, 50), (x0 + 20, 110), 1.0, 1)
    beauty = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    beauty = beauty * (1 - wires[..., None]) + FACE * wires[..., None]
    return wires, hull, beauty


def _write_type(tmp, wires, hull, beauty, matte=None, n=N):
    for k, im in (("beauty", beauty), ("matte", wires if matte is None else matte), ("line", hull)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        u8 = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        if u8.ndim == 3:
            u8 = u8[..., ::-1]
        for i in range(n):
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), u8)


def _piece(ops, holds=((0.3, 0.9),)):
    return {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]},
            "shots": [{"id": "s1", "frames": N, "matte": True}],
            "compose": {"crossfade": 0.0, "default": {"ops": ops}, "color": "fixed"},   # these tests are about the face wiring; the colour stage has tests/test_legible_color.py
            "plan": {"flow_wake": {"holds": [list(h) for h in holds]}}}


def _plate(v=0.4):
    f = np.full((N, H, W, 3), v, np.float32)
    return np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)


def _run(tmp, ops, wires, hull, beauty, plate_v=0.4, holds=((0.3, 0.9),), matte_type=None, frames=None):
    _write_type(tmp, wires, hull, beauty, matte=matte_type)
    piece = _piece(ops, holds)
    matte = np.zeros((N, H, W), np.float32)
    u8, st = RE.run_compose(piece, "s1", type_dir=str(tmp), footage=_plate(plate_v), matte=matte, backend="numpy", threads=0, frames=frames)
    return np.asarray(u8, np.uint8).astype(np.float32) / 255.0, st


# the planner's old chain: tcol (flat colour sampled from the plate) x ta = line pass -> matte crossfade, behind the subject matte
def _old_ops():
    return [
        {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 1.0}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "crossfade", "out": "ta", "args": {"a": "$type_line", "b": "$type_matte", "t": {"timeline": [[0.0, 0.0], [5.0, 1.0]]}}},
        {"op": "sandwich", "out": "image", "args": {
            "footage": "$foot", "matte": "$m", "type_a": "$ta", "type_rgb": "$tcol", "matte_threshold": 0.5,
            "type_opacity": {"timeline": [[0.0, 0.0], [0.4, 0.15], [1.0, 0.95]]}}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0}},
    ]


def _mid(holds=((0.3, 0.9),)):
    return int(round((holds[0][0] + holds[0][1]) / 2 * 24))


# ----------------------------------------------------------------------------------------------------------- class 1: flat hull
def test_type_is_the_beauty_not_a_flat_hull(tmp_path):
    wires, hull, beauty = _letters()
    mid = _mid()
    img, st = _run(tmp_path / "t", _old_ops(), wires, hull, beauty, frames=(mid, mid + 1))
    f = img[0]
    plate = 0.4
    gap = (hull > 0.5) & (cv2.erode(wires, np.ones((3, 3), np.uint8)) < 0.5) & (cv2.dilate(wires, np.ones((5, 5), np.uint8)) < 0.5)
    line = wires > 0.9
    assert gap.sum() > 50 and line.sum() > 50
    assert float(np.abs(f[gap] - plate).mean()) < 0.06, "hull gaps must show the plate, not a flat slab"
    assert float(np.abs(f[line] - FACE).mean()) < 0.12, "the lines carry the beauty colour (hold floor lifts the opacity to 1)"


def test_pass_is_off_with_beauty_face_false_and_changes_nothing_else(tmp_path):
    (tmp_path / "beauty").mkdir()
    piece = _piece(_old_ops())
    piece["compose"]["beauty_face"] = False
    rec = RE.build_recipe(piece, "s1", type_dir=str(tmp_path))
    assert not any(o["op"] in ("type_face", "hold_net") for o in rec["ops"])
    assert "type_beauty" not in rec["inputs"]
    rec2 = RE.build_recipe(_piece(_old_ops()), "s1", type_dir=str(tmp_path))
    assert rec2["ops"][0]["op"] == "type_face" and any(o["op"] == "hold_net" for o in rec2["ops"])
    sw = next(o for o in rec2["ops"] if o["op"] == "sandwich")
    assert sw["args"]["type_rgb"] == "$tface" and sw["args"]["type_a"] == "$ta"
    ta = next(o for o in rec2["ops"] if o.get("out") == "ta")
    assert ta["args"]["a"] == ta["args"]["b"] == "$tmat", "the line pass is never the visible type"
    assert sw["args"]["type_opacity"]["hold_floor"]["value"] == 1.0
    assert "beauty" in RE.need_passes(_piece(_old_ops()))


def test_hold_floor_lifts_opacity_inside_the_hold_only():
    v = {"timeline": [[0.0, 0.0], [0.4, 0.15], [1.0, 0.95]], "hold_floor": {"holds": [[0.5, 0.9]], "value": 1.0, "ramp": 0.25}}
    r = lambda t: RE.resolve(v, {}, {}, 0.0, int(round(t * 24)), 24)   # noqa: E731
    assert r(0.7) == pytest.approx(1.0)
    assert r(0.1) == pytest.approx(0.03, abs=0.02)
    assert 0.2 < r(0.35) < 0.95      # ramp


def test_flow_wake_and_rigid_pose_carry_the_colour_layer(tmp_path):
    ops = [
        {"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m", "frame_idx": "$frame_idx", "u": "$u",
                                                  "state_key": "T:m"}},
        {"op": "rigid_pose", "out": "wm", "args": {"img": "$wm", "angle": 3.0, "dx": 0.0, "dy": 0.0}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.9, 0.9, 0.9]}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "matte": "$matte", "type_a": "$wm", "type_rgb": "$tcol"}},
    ]
    (tmp_path / "beauty").mkdir()
    rec = RE.build_recipe(_piece(ops), "s1", type_dir=str(tmp_path))
    outs = [o.get("out") for o in rec["ops"]]
    assert "wm_face" in outs and outs.count("wm_face") >= 2, outs        # rgba_part of the twin + the rigid_pose twin
    sw = next(o for o in rec["ops"] if o["op"] == "sandwich")
    assert sw["args"]["type_rgb"] == "$wm_face"


# ----------------------------------------------------------------------------------------------------------- class 2: window into dark
def test_window_contrast_is_enforced_by_darkening_the_surround():
    h, w = 64, 48
    a = np.zeros((h, w), np.float32)
    a[20:44, 8:40] = 1.0
    plate = np.full((h, w, 3), 0.7, np.float32)            # bright surround
    foot = np.full((h, w, 3), 0.85, np.float32)            # window only a little brighter
    off = C.knockout(plate, a, foot)
    on = C.knockout(plate, a, foot, min_contrast=0.25)
    lin_off, lout_off = off[a > 0.5].mean(), off[a < 0.01].mean()
    lin_on, lout_on = on[a > 0.5].mean(), on[a < 0.01].mean()
    assert lin_off - lout_off < 0.2 and lin_on - lout_on >= 0.24


def test_window_into_a_dark_plate_falls_back_to_solid_type():
    h, w = 64, 48
    a = np.zeros((h, w), np.float32)
    a[20:44, 8:40] = 1.0
    plate = np.full((h, w, 3), 0.08, np.float32)
    foot = np.full((h, w, 3), 0.06, np.float32)            # the footage itself is dark: no window can show anything
    face = np.zeros((h, w, 3), np.float32) + np.array([0.9, 0.8, 0.5], np.float32)
    off = C.knockout(plate, a, foot)
    on = C.knockout(plate, a, foot, min_contrast=0.22, face=face)
    assert off[a > 0.5].mean() < 0.1
    assert on[a > 0.5].mean() > 0.6 and on[a < 0.01].mean() < 0.1


def test_window_treatment_ops_carry_min_contrast():
    ops = MT.ops_for([{"matte": "window", "t0": 0.3, "t1": 0.9}], plate="$foot")
    kn = next(o for o in ops if o["op"] == "knockout")
    assert kn["args"]["min_contrast"] == pytest.approx(0.22)


# ----------------------------------------------------------------------------------------------------------- class 3: inverted matte
def test_inverted_type_matte_is_rebuilt_from_the_beauty():
    wires, hull, beauty = _letters()
    solid = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    solid = solid * (1 - hull[..., None]) + np.array([0.9, 0.4, 0.2], np.float32) * hull[..., None]
    inverted = 1.0 - hull                                  # white everywhere except the letters
    r = C.type_face(solid, inverted, bg=[5, 8, 13])
    a = r[..., 3]
    assert abs(float(a.mean()) - float(hull.mean())) < 0.02
    assert float(np.abs(a - hull).mean()) < 0.03
    assert np.abs(r[..., :3][hull > 0.5] - np.array([0.9, 0.4, 0.2])).mean() < 0.05
    ok = C.type_face(solid, hull, bg=[5, 8, 13])
    assert np.allclose(ok[..., 3], hull, atol=1e-5), "a consistent matte is never touched"


def test_inverted_matte_end_to_end_final_shows_the_letters(tmp_path):
    wires, hull, beauty = _letters()
    solid = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    solid = solid * (1 - hull[..., None]) + np.array([0.9, 0.4, 0.2], np.float32) * hull[..., None]
    mid = _mid()
    img, _ = _run(tmp_path / "t", _old_ops(), 1.0 - hull, hull, solid, matte_type=1.0 - hull, frames=(mid, mid + 1))
    f = img[0]
    assert abs(float(f[hull > 0.5].mean()) - float(f[hull < 0.5].mean())) > 0.08   # (was 0.12 with the hold net scrim darkening the surround; 2026-10-30 pure type: no scrim, the ink colour stage carries the contrast; here color is fixed)
    assert float(np.abs(f[hull < 0.5] - 0.4).mean()) < 0.05, "the background stays the plate (not the inverted matte)"


# ----------------------------------------------------------------------------------------------------------- class 4: crushed type
def test_window_treatment_never_crushes_the_type(tmp_path):
    wires, hull, beauty = _letters()
    ops = _old_ops()[:-1]
    ops += MT.ops_for([{"matte": "window", "t0": 0.3, "t1": 0.9, "subject_out": False}], base="image", plate="$foot", matte="$m",
                      type_a="$type_matte", type_rgb="$tcol")
    ops += [{"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0}}]
    mid = _mid()
    img, _ = _run(tmp_path / "t", ops, wires, hull, beauty, plate_v=0.07, frames=(mid, mid + 1))      # a night plate
    f = img[0]
    line = wires > 0.9
    assert float(f[line].mean()) > 0.35, "type must stay readable on a dark plate under the window dimming"


# ----------------------------------------------------------------------------------------------------------- class 5: the net
def test_hold_score_tells_a_slab_from_the_word():
    wires, hull, beauty = _letters()
    plate = np.full((H, W, 3), 0.4, np.float32)
    good = plate * (1 - wires[..., None]) + FACE * wires[..., None]
    slab = plate * (1 - hull[..., None]) + np.array([0.7, 0.7, 0.7], np.float32) * hull[..., None]
    dark = plate * (1 - wires[..., None] * 0.9)             # lines barely different from the plate
    dark = np.clip(plate + (dark - plate) * 0.1, 0, 1)
    assert C.hold_score(good, wires, plain=good)["score"] > 0.85
    assert C.hold_score(slab, wires, plain=good)["score"] < 0.6
    assert C.hold_score(dark, wires, plain=good)["score"] < 0.45
    assert C.hold_score(plate, np.zeros((H, W), np.float32))["score"] == 1.0          # no type: nothing to measure


def test_hold_net_blends_inside_the_hold_only_and_logs():
    wires, hull, beauty = _letters()
    plate = np.full((H, W, 3), 0.4, np.float32)
    slab = plate * (1 - hull[..., None]) + np.array([0.7, 0.7, 0.7], np.float32) * hull[..., None]
    C.NET_LOG.clear()
    holds = [[0.5, 1.5]]
    inside = C.hold_net(slab, plate, FACE, wires, holds=holds, t=1.0, frame_idx=24, state_key="x")
    outside = C.hold_net(slab, plate, FACE, wires, holds=holds, t=2.5, frame_idx=60, state_key="x")
    ramp = C.hold_net(slab, plate, FACE, wires, holds=holds, t=0.4, frame_idx=10, state_key="x")
    assert np.array_equal(outside, slab), "outside the hold (+ ramp) the image is untouched"
    line = wires > 0.9
    assert np.abs(inside[line] - FACE).mean() < 0.2 and np.abs(slab[line] - FACE).mean() > 0.3
    assert 0.0 < float(np.abs(ramp - slab).max()) < float(np.abs(inside - slab).max())          # 0.25 s ramp
    rep = C.net_report("x", holds)
    assert rep[0]["fired"] and rep[0]["mid_score"] is not None and rep[0]["mid_score"] < 0.6 and rep[0]["frames"] == 1 and rep[0]["max_k"] > 0.5
    good = plate * (1 - wires[..., None]) + FACE * wires[..., None]
    assert np.allclose(C.hold_net(good, plate, FACE, wires, holds=holds, t=1.0, frame_idx=25, state_key="y"), good, atol=1e-6)
    assert not C.net_report("y", holds)[0]["fired"]


def test_hold_net_scrim_for_cream_on_pale():
    wires, hull, beauty = _letters()
    pale = np.full((H, W, 3), 0.85, np.float32)
    cream = np.array([0.93, 0.9, 0.82], np.float32)
    img = pale.copy()
    out = C.hold_net(img, pale, cream, hull, holds=[[0.5, 1.5]], t=1.0, frame_idx=24, state_key="z", scrim=True)
    assert abs(float(out[hull > 0.5].mean()) - float(out[(hull < 0.5)][:200].mean())) > 0.25 or float(out[hull > 0.5].mean()) > 0.85
    ring = (cv2.dilate(hull, np.ones((9, 9), np.uint8)) > 0) & (hull < 0.5)
    assert float(out[ring].mean()) < 0.75, "the surround is darkened so the cream face reads"


def test_net_runs_in_a_real_compose_and_reports(tmp_path):
    wires, hull, beauty = _letters()
    C.NET_LOG.clear()
    img, st = _run(tmp_path / "t", _old_ops(), wires, hull, beauty)
    assert st.get("hold_net") and st["hold_net"][0]["hit"] == pytest.approx(0.3)
    mid = _mid()
    f = img[mid]
    line = wires > 0.9
    assert float(np.abs(f[line] - FACE).mean()) < 0.15


# ----------------------------------------------------------------------------------------------------------- multiplane on the box path
def _mp_piece(plate="multiplane"):
    ops = [
        {"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m", "frame_idx": "$frame_idx", "u": "$u",
                                                  "state_key": "T:m"}},
        {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.7}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "legible", "out": "m", "args": {"type_a": "$wm", "matte": "$m"}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "matte": "$m", "type_a": "$wm", "type_rgb": "$tcol", "matte_threshold": 0.5,
                                                    "type_opacity": {"timeline": [[0.0, 0.1], [1.0, 0.95]]}}},
        {"op": "contour_lines", "out": "_gl0c", "args": {"src": "$m", "mode": "outline", "frame_idx": "$frame_idx"}},
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$tcol", "alpha": "$_gl0c"}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx"}},
    ]
    p = _piece(ops)
    p["shots"] = [{"id": "s1", "frames": N, "matte": True, "depth": True, "flow": True}]
    p["type"] = {"camera": {"chain": "auto", "plate": plate, "cards": 4} if plate else None, "words": [], "palette": {"bg": [5, 9, 13]}}
    if not plate:
        p["type"].pop("camera")
    return p


def _mp_type_dir(tmp_path):
    td = tmp_path / "type"
    (td / "beauty").mkdir(parents=True)
    (td / "manifest.json").write_text(json.dumps({"spec": {"palette": {"bg": [5, 9, 13]}}, "size": [W, H], "depth": {"near": 500, "far": 1500}}))
    return td


def test_multiplane_piece_recipe_has_cards_face_and_net_in_order(tmp_path):
    td = _mp_type_dir(tmp_path)
    rec = RE.build_recipe(_mp_piece(), "s1", type_dir=str(td))
    names = [o["op"] for o in rec["ops"]]
    assert names.count("multiplane") == 1 and names.count("type_face") == 1 and names.count("hold_net") == 1
    assert names.index("type_face") < names.index("multiplane") < names.index("hold_net") < names.index("grade", names.index("hold_net"))
    assert names.index("hold_net") < names.index("grain"), "the net runs before the closing grade / grain"
    mp = next(o for o in rec["ops"] if o["op"] == "multiplane")
    assert mp["args"]["type_rgb"] == "$type_beauty" and mp["args"]["type_a"] == "$tmat", "the cards draw the beauty with the robust alpha, not $tcol"
    net = next(o for o in rec["ops"] if o["op"] == "hold_net")
    assert net["args"]["face"] == "$tface" and net["args"]["alpha"] == "$tmat" and net["args"]["plate"] == net["args"]["image"] == "$image"
    assert not any(o["op"] in ("sandwich", "knockout", "flow_wake", "legible") for o in rec["ops"]), "nothing draws the type a second time"
    assert any(o["op"] == "contour_lines" for o in rec["ops"]) and any(o["op"] == "over" for o in rec["ops"]), "glow / contour lines stay (m survives the dropped guard)"
    assert {"s1_depth", "s1_matte", "type_beauty", "type_depth", "type_matte"} <= set(rec["inputs"]) and "flow" not in rec["inputs"]
    assert (td.parent / "mp_setup_s1.json").is_file()
    setup = json.loads((td.parent / "mp_setup_s1.json").read_text())
    assert setup["scene_key"] == "T" and "holds" in setup
    p = _mp_piece()
    assert RE.shot_needs(p, "s1") == {"matte": True, "depth": True} and not RE.shot_needs_flow(p, "s1")
    assert {"beauty", "matte", "depth"} <= set(RE.need_passes(p))


def test_flat_piece_recipe_is_unchanged_by_the_multiplane_hook(tmp_path):
    td = _mp_type_dir(tmp_path)
    flat = _mp_piece(plate=None)
    rec = RE.build_recipe(flat, "s1", type_dir=str(td))
    names = [o["op"] for o in rec["ops"]]
    assert "multiplane" not in names and "sandwich" in names and "flow_wake" in names and "hold_net" in names
    assert not (td.parent / "mp_setup_s1.json").exists()
    assert RE.shot_needs_flow(flat, "s1")
    mp_off = _mp_piece()
    mp_off["compose"]["multiplane"] = False
    assert "multiplane" not in [o["op"] for o in RE.build_recipe(mp_off, "s1", type_dir=str(td))["ops"]]
    assert RE.multiplane_ops(flat, [{"op": "sandwich", "args": {}}], str(td), "s1") == [{"op": "sandwich", "args": {}}]


def test_multiplane_hook_skips_a_shot_without_depth_or_manifest(tmp_path):
    td = _mp_type_dir(tmp_path)
    p = _mp_piece()
    p["shots"][0]["depth"] = False
    assert "multiplane" not in [o["op"] for o in RE.build_recipe(p, "s1", type_dir=str(td))["ops"]]
    (td / "manifest.json").unlink()
    assert "multiplane" not in [o["op"] for o in RE.build_recipe(_mp_piece(), "s1", type_dir=str(td))["ops"]]


def test_local_runner_entry_and_box_entry_share_one_transform(tmp_path):
    from tools.fusion import multiplane_recipe as MPR
    td = _mp_type_dir(tmp_path)
    ops = _mp_piece()["compose"]["default"]["ops"]
    out = MPR.apply_to(_mp_piece(), copy.deepcopy(ops), "s1", str(td), str(tmp_path / "s.json"), log=lambda m: None)
    names = [o["op"] for o in out]
    assert names.count("multiplane") == 1 and "sandwich" not in names and "flow_wake" not in names
    assert next(o for o in out if o["op"] == "multiplane")["args"]["type_a"] == "$type_matte"      # the local runner has no legibility pass


def test_multiplane_cards_are_cut_from_the_plate_with_the_hold_layout_strips(tmp_path):
    td = _mp_type_dir(tmp_path)
    p = _mp_piece()
    ops = p["compose"]["default"]["ops"]
    ops.insert(2, {"op": "layout_shapes", "out": "plate", "args": {"img": "$footage", "shapes": [], "t": "$t", "matte": "$m"}})
    for o in ops:
        if o["op"] == "grade" and o.get("out") == "foot":
            o["args"]["frame"] = "$plate"
    rec = RE.build_recipe(p, "s1", type_dir=str(td))
    assert next(o for o in rec["ops"] if o["op"] == "multiplane")["args"]["footage"] == "$plate"
    names = [o["op"] for o in rec["ops"]]
    assert names.index("layout_shapes") < names.index("multiplane")
