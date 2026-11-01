"""v006 directional pan transitions: the camera travel IS the transition (down / up / right / left). Type and footage travel together.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_pan_transitions.py -q     (CPU, tiny frames, ~1 min)

What is asserted
  * geometry: B sits one frame-span further along the travel in the SAME plane; the pan camera is a pure translation with quintic ease and rests on
    both words
  * timing: travel starts after A's hold, ENDS on B's hit, B lands on a downbeat (several tempi / beat counts / runs of three words)
  * type and plate move together: per frame the displacement of the REAL rendered type (A's geometry through the actual camera chain) equals the
    plate_pan displacement (the plate track of that camera), within 1 px, in all 4 directions; at B's hold the plate is exactly the shot's frame again
  * motion blur grows with speed; the op is registered; the transition composes with any word module and the io styles
  * planner: spec keys, block camera_pan_transition (opt-in), batch coverage of the 4 directions, LTX camera phrases, direction from footage flow
"""
from __future__ import annotations

import copy
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

from mograph.modules import core  # noqa: E402
from mograph.modules.camera import pan_dir  # noqa: E402
from mograph.modules.sequence import _draw  # noqa: E402
from mograph.render import cam3d  # noqa: E402
from tools.fusion import batch_plan as bp  # noqa: E402
from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import compose  # noqa: E402
from tools.fusion import pan_check as PC  # noqa: E402
from tools.fusion import pan_plan as PAN  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import spec as S  # noqa: E402
from tools.fusion.plate_pan import plate_pan  # noqa: E402

SIZE = (192, 320)
DIRS = ("down", "up", "right", "left")


@pytest.fixture(autouse=True)
def _world():
    w = (cam3d.W, cam3d.H)
    yield
    cam3d.set_world(*w)


@pytest.fixture(scope="module")
def scenes():
    out = {}
    for d in DIRS:
        sc, tr = PC.build(d, SIZE)
        out[d] = (sc, tr)
    cam3d.set_world(960, 540)
    return out


def _use(sc):
    cam3d.set_world(*sc.world)
    return sc


def _words(sc):
    return [it for it, a, b in sc.seq.timeline if isinstance(it, core.WordModule)]


def _pan(sc):
    return next((it, a, b) for it, a, b in sc.seq.timeline if getattr(it, "name", "") == "pan")


# ----------------------------------------------------------------------------------------------- geometry
@pytest.mark.parametrize("d", DIRS)
def test_b_sits_one_frame_span_further_along_in_the_same_plane(scenes, d):
    sc, _ = scenes[d]
    _use(sc)
    a, b = _words(sc)
    D = pan_dir.fit_dist(a.placement.w, a.placement.h, 1000.0, 0.7)
    span = pan_dir.frame_span(d, D, 1000.0, sc.world)
    vx, vy = pan_dir.DIR_VEC[d]
    off = b.placement.origin - a.placement.origin
    assert off[2] == 0.0 and np.allclose(off[:2], [vx * span, vy * span], atol=1e-6), (d, off, span)
    assert np.allclose(a.placement.n, b.placement.n) and np.allclose(a.placement.n, [0, 0, 1])      # same face-on plane
    hints = sc.seq.camera_hints(1000.0)
    assert abs(hints.words[0].fit(0.7) - D) < 1e-6                                                     # the planner's rule IS WordInfo.fit


@pytest.mark.parametrize("d", DIRS)
def test_camera_is_a_pure_translation_with_rest_on_both_words(scenes, d):
    sc, tr = scenes[d]
    _use(sc)
    _, a_, b_ = _pan(sc)
    ts = np.linspace(a_ - 0.5, b_ + 0.5, 120)
    cams = [sc.chain.at(t) for t in ts]
    f0 = cams[0].f
    assert all(np.allclose(c.f, f0, atol=1e-6) for c in cams), "view direction changed: not a pure pan"
    assert max(abs(c.roll) for c in cams) < 1e-9
    eye = np.array([c.eye for c in cams])
    along = eye - eye[0]
    vx, vy = pan_dir.DIR_VEC[d]
    ax = np.array([vx, vy, 0.0])
    assert np.abs(along @ np.cross(ax, [0, 0, 1])).max() < 1e-6 and np.abs(along[:, 2]).max() < 1e-6      # moves along the direction only
    s = along @ ax
    assert s[-1] > 0 and np.all(np.diff(s) >= -1e-6), "travel is not monotone"
    v = np.gradient(s, ts)
    i_mid = int(np.argmax(v))
    assert abs(ts[i_mid] - 0.5 * (a_ + b_)) < 0.12 * (b_ - a_) and v[0] < 1.0 and v[-1] < 1.0          # ease in / out: peak mid-travel, rest at both ends
    assert abs(s[-1] - pan_dir.frame_span(d, pan_dir.fit_dist(*[_words(sc)[0].placement.w, _words(sc)[0].placement.h], 1000.0, 0.7), 1000.0, sc.world)) < 1e-3


# ----------------------------------------------------------------------------------------------- timing
@pytest.mark.parametrize("bpm,beats,land", [(73, 2, "downbeat"), (96, 2, "downbeat"), (96, 4, "downbeat"), (120, 3, "downbeat"), (89, 2, "beat")])
def test_travel_starts_after_the_hold_and_ends_on_the_hit_which_lands_on_a_beat(bpm, beats, land):
    sp = PC.pan_spec("down", SIZE, beats=beats, bpm=bpm, duration=16.0)
    sp["words"][0]["transition"]["land"] = land
    sp["camera"] = None
    sc = S.build_scene(sp, SIZE)
    w0, w1 = sc.info["words"]
    _, ta, tb = _pan(sc)
    grid = sc.grid
    fps = sc.fps
    assert abs((tb - ta) - beats * grid.iv) < 1e-6                                   # the travel is `beats` beats long
    assert abs(tb + PAN.EDGE_S - w1["hit"]) < 2e-3                                    # ... and B's hit is right after it (info times are rounded to ms)
    assert w0["release"] - w0["hit"] >= S.min_hold({"text": "LOOK"}) - 2e-3          # A's hold is kept (readable)
    assert abs(ta - (w0["release"] + PAN.EDGE_S)) < 2e-3                              # travel starts right after A's hold
    near = grid.snap(w1["hit"], bars=(land != "beat"))
    assert abs(w1["hit"] - near) * fps < 0.5, (w1["hit"], near)                       # B lands on the downbeat (the beat in 'beat' mode)


def test_run_of_three_words_every_hit_on_a_downbeat():
    sp = PC.pan_spec("right", SIZE, duration=24.0)
    sp["words"][1]["transition"] = dict(kind="pan", dir="down", beats=3)
    sp["words"].append(dict(text="THREE", module="lanes", font="Big Shoulders Display"))
    sp["camera"] = None
    sc = S.build_scene(sp, SIZE)
    ws = sc.info["words"]
    assert len(sc.info["pans"]) == 2 and [p["dir"] for p in sc.info["pans"]] == ["right", "down"]
    for w in ws[1:]:
        assert abs(w["hit"] - sc.grid.snap(w["hit"], bars=True)) * sc.fps < 0.5
    o = [m.placement.origin for m in _words(sc)]
    assert o[1][1] == o[0][1] and o[1][0] > o[0][0] and o[2][0] == o[1][0] and o[2][1] > o[1][1]     # right, then down: an L in the world plane


# ----------------------------------------------------------------------------------------------- type and plate move together
def _shift_search(img0, img1, guess, rad=1.5, step=0.25):
    """the (dx, dy) within `rad` px of `guess` that best maps img0 onto img1 (mean abs error over the pixels img0 still covers) -> (err, dx, dy)."""
    h, w = img0.shape[:2]
    best = None
    for ddx in np.arange(-rad, rad + 1e-9, step):
        for ddy in np.arange(-rad, rad + 1e-9, step):
            sx, sy = guess[0] + ddx, guess[1] + ddy
            w0 = cv2.warpAffine(img0, np.float32([[1, 0, sx], [0, 1, sy]]), (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
            err = float(np.abs(w0 - img1)[w0 >= 0].mean())
            if best is None or err < best[0] - 1e-9:
                best = (err, sx, sy)
    return best


@pytest.mark.parametrize("d", DIRS)
def test_type_and_plate_displacement_match_per_frame(scenes, d):
    sc, tr = scenes[d]
    _use(sc)
    a, b = _words(sc)
    t0, t1, _ = tr["windows"][0]
    fps = sc.fps
    k = SIZE[0] / float(sc.world[0])                       # world frame px -> render px
    layers_a = a.layers_uncarried(a.timing.form + a.timing.hold)
    cam3d.set_world(*sc.world)
    img0 = _draw(layers_a, sc.chain.at(t0), (0, 0, 0), SIZE, 1).astype(np.float32).mean(-1) / 255.0
    rng = np.random.RandomState(1)
    plate = cv2.GaussianBlur(rng.rand(SIZE[1], SIZE[0], 3).astype(np.float32), (0, 0), 3.0)         # a textured plate with no tile structure
    plate = (plate - plate.min()) / (plate.max() - plate.min())
    checked = 0
    for u in np.linspace(0.06, 0.72, 14):
        i = int(round((t0 + u * (t1 - t0)) * fps))
        dx_w, dy_w = tr["dx"][i], tr["dy"][i]
        want = (dx_w * k, dy_w * k)
        # the real type: A's own geometry through the actual camera chain at frame i
        img = _draw(layers_a, sc.chain.at(i / fps), (0, 0, 0), SIZE, 1).astype(np.float32).mean(-1) / 255.0
        if img.sum() < 0.45 * img0.sum():                   # the type is mostly out of frame: its displacement is no longer measurable (flat error)
            continue
        err, sx, sy = _shift_search(img0, img, want, rad=1.5, step=0.25)
        assert abs(sx - want[0]) <= 1.0 and abs(sy - want[1]) <= 1.0, (d, i, (sx, sy), want)       # type displacement == track (<= 1 px)
        # the plate: the op with the same arguments the recipe bakes
        W, H = tr["world"]
        out = plate_pan(plate, dx=dx_w / W, dy=dy_w / H, seam_amt=0.0)
        g0, g1 = plate.mean(-1), out.mean(-1)
        errp, px, py = _shift_search(g0, g1, want, rad=1.5, step=0.25)
        assert abs(px - want[0]) <= 0.5 and abs(py - want[1]) <= 0.5, (d, i, (px, py), want)         # plate displacement == track
        assert abs(px - sx) <= 1.0 and abs(py - sy) <= 1.0                                           # ... and type == plate, per frame
        checked += 1
    assert checked >= 4
    i1 = int(round(t1 * fps)) + 2                                                      # B holds: one frame-span later the plate is the shot's frame again
    W, H = tr["world"]
    after = plate_pan(plate, dx=tr["dx"][i1] / W, dy=tr["dy"][i1] / H)
    assert np.abs(after - plate).max() < 1e-4 and abs(abs(tr["dx"][i1]) / W + abs(tr["dy"][i1]) / H - 1.0) < 1e-3


@pytest.mark.parametrize("d", DIRS)
def test_directions_are_the_camera_direction(scenes, d):
    sc, tr = scenes[d]
    _use(sc)
    vx, vy = pan_dir.DIR_VEC[d]
    W, H = tr["world"]
    # the camera travels one frame-span along D, so the picture slides one frame the OTHER way (camera down -> type up, dy = -H)
    assert np.allclose([tr["dx"][-1], tr["dy"][-1]], [-vx * W, -vy * H], atol=0.01), (d, tr["dx"][-1], tr["dy"][-1])
    mv = PAN.type_motion(sc, 0)
    n = int(round(sc.info["words"][1]["release"] * sc.fps))                  # up to B's release (the pan run's own window; later the chain moves on)
    assert np.allclose(mv[:n, 0], tr["dx"][:n], atol=1e-3) and np.allclose(mv[:n, 1], tr["dy"][:n], atol=1e-3)     # word A's origin moves exactly like the plate


def test_no_independent_slide_the_type_only_moves_with_the_camera(scenes):
    """A and B are static in the world: only the camera changes between frames, word layers are identical at every travel frame."""
    sc, _ = scenes["down"]
    _use(sc)
    tr_, ta, tb = _pan(sc)
    l0 = [np.concatenate([np.atleast_2d(p).ravel() for p in L.polys]) for L in tr_.layers(0.0)]
    l1 = [np.concatenate([np.atleast_2d(p).ravel() for p in L.polys]) for L in tr_.layers(0.6 * tr_.duration)]
    l2 = [np.concatenate([np.atleast_2d(p).ravel() for p in L.polys]) for L in tr_.layers(tr_.duration)]
    assert len(l0) == len(l1) == len(l2) and all(np.array_equal(x, y) for x, y in zip(l0, l1)) and all(np.array_equal(x, y) for x, y in zip(l0, l2))


# ----------------------------------------------------------------------------------------------- the plate op
def test_motion_blur_is_proportional_to_speed_and_zero_at_rest():
    img = np.zeros((200, 100, 3), np.float32)
    img[:, 48:52] = 1.0                                    # a vertical bar
    img[98:102, :] = 1.0                                   # a horizontal bar
    def spread(v):
        o = plate_pan(img, dy=0.0, vy=v / 200.0, shutter=0.5)[:, :, 0]
        col = o[:, 10]                                      # the horizontal bar crosses this column
        return float((col > 0.05).sum())
    s0, s1, s2 = spread(0.0), spread(20.0), spread(40.0)
    assert s0 <= 4 and abs((s1 - 4) - 10.0) <= 2.5 and abs((s2 - 4) - 20.0) <= 3.5, (s0, s1, s2)           # blur length = speed x shutter (px)
    # blur is along the motion only: a vertical move does not smear the vertical bar sideways
    o = plate_pan(img, dy=0.0, vy=0.2, shutter=0.5)[:, :, 0]
    assert (o[50, 40:60] > 0.05).sum() <= 6


def test_op_is_registered_in_compose_and_wrap_is_exact_at_whole_periods():
    assert compose._ops_registry()["plate_pan"] is plate_pan
    rng = np.random.RandomState(3)
    img = rng.rand(64, 48, 3).astype(np.float32)
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1)):
        assert np.array_equal(plate_pan(img, dx=dx, dy=dy), img)
    assert not np.array_equal(plate_pan(img, dy=0.5), img)
    # parallax: near pixels (depth 1) move more than far ones (0): on a vertical ramp pushed down, the near half ends up darker at the same row
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32)[:, None, None], (1, 48, 3))
    depth = np.zeros((64, 48), np.float32)
    depth[:, :24] = 1.0
    out = plate_pan(ramp, dy=0.2, depth=depth, parallax=1.0)
    assert out[32, 5, 0] < out[32, 40, 0] - 0.02
    assert np.allclose(plate_pan(ramp, dy=0.2, depth=depth, parallax=0.0), plate_pan(ramp, dy=0.2))


# ----------------------------------------------------------------------------------------------- lego: any word module, io styles
MODULES = ["lanes", "pipework", "voxel_roll", "depth_tunnel", "particles", "solid_hollow"]


@pytest.mark.parametrize("m", MODULES)
def test_pan_composes_with_any_word_module(m):
    sp = PC.pan_spec("down", SIZE, duration=9.0)
    for w in sp["words"]:
        w["module"] = m
    sp["camera"] = None
    sc = S.build_scene(sp, SIZE)
    tr_, ta, tb = _pan(sc)
    a, b = _words(sc)
    la = a.layers_uncarried(a.timing.form + a.timing.hold)
    n_mid = len(tr_.layers(0.5 * tr_.duration))
    assert n_mid == len(la) + len(b.layers_uncarried(b.timing.form)) and n_mid > 0
    assert a.carry == {"out": True} and b.carry == {"in": True}
    # the carried windows stay frozen on the hold: the sequence's own sampling of A / B around the travel is the hold frame
    ha = a.layers(a.timing.form + a.timing.hold + 0.5 * a.timing.exit)
    assert [np.asarray(L.polys[0]).shape for L in ha] == [np.asarray(L.polys[0]).shape for L in la]


@pytest.mark.parametrize("io", ["blobs", "particles", "lines"])
def test_io_styles_at_the_frame_edge(io):
    sp = PC.pan_spec("right", SIZE, duration=9.0)
    sp["words"][0].update(form_out=io)
    sp["words"][1].update(form_in=io)
    sp["words"][0]["transition"].update(exit_io=True, entry_io=True)
    sp["camera"] = None
    sc = S.build_scene(sp, SIZE)
    tr_, _, _ = _pan(sc)
    n0, n1 = len(tr_.layers(0.0)), len(tr_.layers(tr_.duration))
    assert n0 > 0 and n1 > 0
    a, b = _words(sc)
    assert len(tr_.layers(0.0)) >= len(a.layers_uncarried(a.timing.form + a.timing.hold))
    # flicker / wipe carry nothing: they are dropped on a pan side (a cut would replace the travel)
    sp2 = PC.pan_spec("right", SIZE, duration=9.0)
    sp2["words"][0]["form_out"] = "flicker"
    sp2["words"][1]["form_in"] = "wipe"
    sp2["camera"] = None
    sc2 = S.build_scene(sp2, SIZE)
    assert any(getattr(it, "name", "") == "pan" for it, _, _ in sc2.seq.timeline) and any("flicker -> lines" in n for n in sc2.info["notes"])


def test_pan_is_hidden_from_the_engine_and_the_coverage_universe():
    assert "pan" not in core.list_modules("transition") and core.get_module("pan").kind == "transition"
    from mograph.modules.camera import list_cameras
    assert "pan" in list_cameras() and "pan" not in pr.transition_names()


# ----------------------------------------------------------------------------------------------- spec keys
def test_spec_keys_word_transition_list_and_camera_move():
    w = [dict(text="A"), dict(text="B"), dict(text="C"), dict(text="D")]
    sp = dict(words=copy.deepcopy(w))
    sp["words"][0]["transition"] = dict(kind="pan", dir="up", beats=3)
    sp["transitions"] = [None, dict(module="pan", dir="left"), "twist"]
    b = PAN.boundaries(sp)
    assert sorted(b) == [0, 1] and b[0]["dir"] == "up" and b[0]["beats"] == 3 and b[1]["dir"] == "left" and b[1]["beats"] == 2
    sp2 = dict(words=copy.deepcopy(w), camera={"move": "pan", "dir": ["down", "right"], "beats": 4}, transitions=[None, "twist", None])
    b2 = PAN.boundaries(sp2)
    assert sorted(b2) == [0, 2] and [b2[0]["dir"], b2[2]["dir"]] == ["down", "right"] and b2[0]["beats"] == 4        # an explicit transition keeps its gap
    assert PAN.boundaries(dict(words=copy.deepcopy(w))) == {} and PAN.runs(b2, 4) == [(0, 1), (2, 3)]
    with pytest.raises(ValueError):
        PAN.boundaries(dict(words=copy.deepcopy(w), transitions=[dict(module="pan", dir="sideways")]))


def test_camera_move_pan_builds_the_whole_piece_as_pans():
    sp = PC.pan_spec("down", SIZE, duration=14.0)
    sp["words"][0].pop("transition")
    sp["words"].append(dict(text="NEXT", module="lanes", font="Big Shoulders Display"))
    sp["camera"] = {"move": "pan", "dir": ["down", "left"], "beats": 2}
    sc = S.build_scene(sp, SIZE)
    assert [p["dir"] for p in sc.info["pans"]] == ["down", "left"]
    assert sc.chain.plan[0][0] == "pan" and sc.chain.plan[0][2] == 0.0 and abs(sc.chain.plan[0][3] - sc.info["words"][-1]["end"]) < 2e-3
    _use(sc)
    tr = PAN.plate_track(sc)
    assert len(tr["windows"]) == 2 and abs(tr["dy"][-1] + sc.world[1]) < 1e-3 and abs(tr["dx"][-1] - sc.world[0]) < 1e-3


# ----------------------------------------------------------------------------------------------- plan: weave, LTX phrase, footage direction, block, batch
def _tiny_piece(sc, tr, d="down"):
    ops = [{"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.8}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_rgb": "$tcol", "type_a": "$type_matte", "matte": "$matte"}}]
    sp = PC.pan_spec(d, SIZE)
    return {"id": "T", "fps": 24, "type": dict(sp, downbeats=list(sc.grid.d[:8]), beats=list(sc.grid.t[:30]), bpm=96, duration=9.0),
            "plan": {"tier": "mid"}, "compose": {"default": {"ops": ops}, "shots": {}},
            "shots": [{"id": "s1", "kind": "control", "first": "A", "frames": 216, "prompt": "a room, one slow continuous camera sweep from left to right, cinematic"}]}


def test_weave_adds_plate_pan_ops_for_footage_and_matte_and_is_idempotent(scenes):
    sc, tr = scenes["down"]
    p = _tiny_piece(sc, tr)
    base = {"timeline": [[0.0, 0.0], [0.6, 0.75], [3.6, 0.75], [5.2, 0.1], [6.0, 0.75]]}
    p["compose"]["default"]["ops"].insert(0, {"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "settle": copy.deepcopy(base)}})
    PAN.weave(p, track=tr)
    ops = p["compose"]["default"]["ops"]
    wake = next(o for o in ops if o["op"] == "flow_wake")
    st = wake["args"]["settle"]["timeline"]
    t0_, t1_ = tr["windows"][0][:2]
    assert all(v == 1.0 for t, v in st if t0_ <= t <= t1_) and any(t == round(t0_, 4) for t, _ in st) and st == sorted(st)     # the wake is settled while the camera carries the type
    assert [k for k in st if k[0] < t0_ - 0.13] == [k for k in base["timeline"] if k[0] < t0_ - 0.13]                       # ... and untouched elsewhere
    assert [o["op"] for o in ops[:2]] == ["plate_pan", "plate_pan"] and {o["out"] for o in ops[:2]} == {"footage_p", "matte_p"}
    grade = next(o for o in ops if o["op"] == "grade")
    assert grade["args"]["frame"] == "$footage_p" and next(o for o in ops if o["op"] == "sandwich")["args"]["matte"] == "$matte_p"
    keys = ops[0]["args"]["dy"]["timeline"]
    assert len(keys) < 80 and abs(keys[-1][1] + 1.0) < 1e-4                                 # sparse keys; a whole frame-span up at the end
    assert p["plan"]["pan"]["boundaries"][0]["dir"] == "down" and abs(p["plan"]["pan"]["boundaries"][0]["d_down_f"]) < 0.5
    first = copy.deepcopy(p["compose"])
    PAN.weave(p, track=tr)
    assert p["compose"] == first                                                           # idempotent (the settle bump is rebuilt from the stored original)
    # the baked arguments reproduce the track at every frame the recipe resolves them
    from tools.fusion import recipe_eval as RE
    w, h = tr["world"]
    for i in (60, 90, 100, 108, 140):
        v = RE.resolve(ops[0]["args"]["dy"], {}, {}, 0.0, i, 24)
        assert abs(v * h - tr["dy"][i]) < 0.6
    assert next(o for o in PAN.unweave_ops(ops) if o["op"] == "grade")["args"]["frame"] == "$footage"


def test_ltx_camera_phrases_and_shot_prompts(scenes):
    assert PAN.ltx_phrase("down", "calm") == "one slow continuous camera tilt down"
    assert PAN.ltx_phrase("left", "kinetic").endswith("camera pan left") and PAN.ltx_phrase("up").endswith("tilt up") and PAN.ltx_phrase("right").endswith("pan right")
    sc, tr = scenes["left"]
    p = _tiny_piece(sc, tr, "left")
    PAN.weave(p, track=tr)
    pr_ = p["shots"][0]["prompt"]
    assert "camera pan left" in pr_ and "left to right" not in pr_ and pr_.endswith("cinematic")
    assert p["plan"]["pan"]["ltx"] == {"s1": "left"}


def test_direction_from_the_footages_own_camera_motion():
    T = 30
    def flow(mx, my):
        f = np.zeros((T, 16, 9, 2), np.float32)
        f[..., 0], f[..., 1] = mx, my
        return f
    # content slides UP -> the camera moved DOWN; content slides left -> the camera panned right
    assert PAN.dir_from_flow(flow(0, -3.0), 24, 0.0, 1.0)[0] == "down"
    assert PAN.dir_from_flow(flow(0, 2.0), 24, 0.0, 1.0)[0] == "up"
    assert PAN.dir_from_flow(flow(-4.0, 0), 24, 0.0, 1.0)[0] == "right"
    assert PAN.dir_from_flow(flow(2.5, 0.2), 24, 0.0, 1.0)[0] == "left"
    d, px, conf = PAN.dir_from_flow(flow(-4.0, 1.0), 24, 0.0, 1.0)
    assert d == "right" and conf > 0.75 and abs(px - 4.0) < 1e-3
    assert PAN.dir_from_flow(flow(0.01, 0.0), 24, 0.0, 1.0)[0] is None           # a static clip has no direction


def test_retarget_takes_the_direction_from_the_footage(scenes):
    sc, tr = scenes["down"]
    p = _tiny_piece(sc, tr)
    T = 216
    f = np.zeros((T, 8, 4, 2), np.float32)
    f[..., 0] = -3.0                                       # the scene slides left: the camera panned right
    p["shots"][0]["frames"] = T
    p["type"]["words"][0]["transition"]["dir"] = "down"
    track2 = {}
    import tools.fusion.pan_plan as PP
    orig = PP.build_camera_scene
    try:
        PP.build_camera_scene = lambda piece, size=None: S.build_scene(dict(piece["type"], size=list(SIZE)), SIZE)
        ch = PAN.retarget(p, {"s1": f}, size=SIZE)
    finally:
        PP.build_camera_scene = orig
    assert ch and ch[0][1] == "down" and ch[0][2] == "right" and p["type"]["words"][0]["transition"]["dir"] == "right"
    assert p["plan"]["pan"]["boundaries"][0]["dir"] == "right" and "camera pan right" in p["shots"][0]["prompt"]


def test_block_is_registered_opt_in_and_never_chosen_on_its_own():
    assert bk.PAN_BLOCK in bk.BLOCKS and bk.validate_block(bk.PAN_BLOCK) == []
    ctx = bk.piece_ctx(n_lines=5, min_hold_s=2.0)
    ok, why = bk.eligible(bk.PAN_BLOCK, ctx)
    assert not ok and "opt-in" in why[0]
    assert bk.eligible(bk.PAN_BLOCK, dict(ctx, opt_in=(bk.PAN_BLOCK,)))[0]
    for seed in range(6):
        assert bk.PAN_BLOCK not in bk.choose_blocks(ctx, seed)
    assert bk.choose_blocks(ctx, 1, brief={"blocks": [bk.PAN_BLOCK]})[0] == bk.PAN_BLOCK


def test_batch_covers_all_four_directions_and_keeps_the_universes():
    pcs = [dict(lines=["SALT", "AND SILVER", "THE TIDE", "REMEMBERS"], blocks=[bk.PAN_BLOCK], pan=2),
           dict(lines=["WILD", "FIRE", "RUNS THROUGH", "THE NIGHT"], blocks=[bk.PAN_BLOCK], pan=2),
           dict(lines=["ONE", "TWO", "THREE"], blocks=[bk.PAN_BLOCK], pan=["left", "up"])]
    a = bp.assign_batch(pcs, seed=4)
    rep = a["coverage"]
    assert rep["pan_dirs"]["universe"] == 4 and rep["pan_dirs"]["missing"] == [], rep["pan_dirs"]
    assert rep["transitions"]["universe"] == 7 and rep["cameras"]["universe"] == 9 and not rep["pairing_problems"]
    for p in a["pieces"]:
        pans = [(ln["index"], ln["pan"]) for ln in p["lines"] if ln.get("pan")]
        assert len(pans) == 2 and all(p["lines"][i]["transition"] == "pan" for i, _ in pans)
        for i, _ in pans:                                                       # screen-free layouts on both sides of a pan, never a cut
            for j in (i, i + 1):
                assert not p["lines"][j]["layout_record"]["anchored"]
            assert p["lines"][i]["io_out"] not in bp.no_material() and p["lines"][i + 1]["io_in"] not in bp.no_material()
        ds = [x["dir"] for _, x in pans]
        assert all(a_ != b_ for a_, b_ in zip(ds, ds[1:]))
    # a request names the directions
    assert [ln["pan"]["dir"] for ln in a["pieces"][2]["lines"] if ln.get("pan")] == ["left", "up"]


@pytest.mark.skipif(os.environ.get("MOGRAPH_FAST") == "1", reason="MOGRAPH_FAST=1: the full-camera piece plan takes ~1 min")
def test_planned_piece_with_the_block_has_the_pan_and_the_weave(monkeypatch):
    from test_fusion_core import synth_music, _req
    monkeypatch.setattr(pr, "OWNERS", False)           # the beat-grid schedule: one-lockup-per-shot makes 1-bar shots (holds ~1.25 s < the block's 1.4 s)
    piece = pr.plan_with_blocks(_req(words="Salt\nAnd silver\nThe tide\nRemembers", length=16, seed=3), synth_music(96, 16), {"blocks": [bk.PAN_BLOCK]}, seed=3, dry=False)
    ws = piece["type"]["words"]
    assert any(isinstance(w.get("transition"), dict) and w["transition"]["kind"] == "pan" for w in ws)
    assert piece["plan"]["pan"]["boundaries"] and all(abs(b["d_down_f"]) < 0.5 for b in piece["plan"]["pan"]["boundaries"])
    assert any(o["op"] == "plate_pan" for o in piece["compose"]["default"]["ops"])
    assert not [n for n in piece["plan"]["notes"] if "needs not met" in n] and any("camera pan" in s["prompt"] or "camera tilt" in s["prompt"] for s in piece["shots"])


# ----------------------------------------------------------------------------------------------- a pan the DESIGN chose (design_options.to_piece -> design_step._finish)
def _design_pan_piece(sc, tr):
    """what the design step holds after to_piece wrote `pan:down`: the word carries the transition, the compose chain has no plate_pan yet."""
    p = _tiny_piece(sc, tr)
    return p


def test_design_step_weaves_a_pan_the_design_chose(scenes, monkeypatch):
    from tools.fusion import design_step as DS
    sc, tr = scenes["down"]
    monkeypatch.setattr(PAN, "build_camera_scene", lambda piece, size=None: sc)
    probe = _design_pan_piece(sc, tr)
    out = copy.deepcopy(probe)
    assert not any(o["op"] == "plate_pan" for o in probe["compose"]["default"]["ops"])
    assert DS.weave_pan(probe, out, {}, None) == []
    ops = probe["compose"]["default"]["ops"]
    assert {o["out"] for o in ops if o["op"] == "plate_pan"} == {"footage_p", "matte_p"}                        # the plate travels with the type camera
    assert probe["plan"]["pan"]["boundaries"][0]["dir"] == "down" and "camera tilt down" in probe["shots"][0]["prompt"]   # ... and the LTX prompt names the move


def test_design_step_without_a_pan_does_nothing(scenes, monkeypatch):
    from tools.fusion import design_step as DS
    sc, tr = scenes["down"]
    monkeypatch.setattr(PAN, "build_camera_scene", lambda *a, **k: pytest.fail("the camera scene must not be built without a pan"))
    probe = _design_pan_piece(sc, tr)
    for w in probe["type"]["words"]:
        w.pop("transition", None)
    before = copy.deepcopy(probe)
    assert DS.weave_pan(probe, copy.deepcopy(probe), {}, None) == [] and probe == before


def test_design_step_drops_a_pan_whose_weave_fails_and_says_so(scenes, monkeypatch):
    from tools.fusion import design_step as DS
    sc, tr = scenes["down"]

    def boom(*a, **k):
        raise RuntimeError("no camera")
    monkeypatch.setattr(PAN, "build_camera_scene", boom)
    probe = _design_pan_piece(sc, tr)
    out = copy.deepcopy(probe)
    before = copy.deepcopy(probe["compose"])
    notes = DS.weave_pan(probe, out, {}, None)
    assert notes and "pan dropped" in notes[0] and probe["compose"] == before                                    # the probe's chains are untouched
    assert not any(PAN._is_pan(w.get("transition")) for w in out["type"]["words"]) and notes[0] in probe["plan"]["window_notes"]


def test_design_step_finish_calls_the_pan_weave(scenes, monkeypatch):
    """the wiring: _finish runs weave_pan on the probe after the matte / flow weavers and carries its compose / shots / plan into the piece."""
    from tools.fusion import design_step as DS
    sc, tr = scenes["down"]
    monkeypatch.setattr(PAN, "build_camera_scene", lambda piece, size=None: sc)
    for fn in ("fill_type_timing", "weave_matte", "weave_flow"):
        monkeypatch.setattr(DS.PR, fn, lambda *a, **k: None)
    monkeypatch.setattr(DS.PR, "dry_run", lambda *a, **k: {})
    monkeypatch.setattr(DS.PR, "finalize_piece", lambda p, ana=None: p)
    monkeypatch.setattr(DS.RE, "need_passes", lambda p: None)
    out = _design_pan_piece(sc, tr)
    out["design"] = {"src": {}}
    fin = DS._finish(out, {})
    assert any(o["op"] == "plate_pan" for o in fin["compose"]["default"]["ops"]) and fin["plan"]["pan"]["boundaries"][0]["dir"] == "down"
    assert "camera tilt down" in fin["shots"][0]["prompt"]
