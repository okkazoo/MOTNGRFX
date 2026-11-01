"""CLEAN PLATE + INDEPENDENT SUBJECT LAYER + TYPE AS A MATTE (tools/fusion/plate_layers.py, nodes_fusion/compose.py bake_plate_clips, tools/app/layered_shots.py plate chain,
16_app/CLEAN_PLATE.md). No model, no GPU, no ComfyUI: synthetic frames through `bake` and through the real compose engine (numpy backend), and the static graph.
Run: python -m pytest tests/test_plate_layers.py -q"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.app import build_graphs as BG  # noqa: E402
from tools.app import layered_shots as LS  # noqa: E402
from tools.fusion import plate_layers as PLT  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W, N = 96, 64, 5
GREEN = np.array([0.2, 0.7, 0.2], np.float32)          # the clean plate
RED = np.array([0.9, 0.1, 0.1], np.float32)            # the subject
SUBJ = np.zeros((H, W), np.float32)
SUBJ[30:62, 8:28] = 1.0                                 # the subject's matte at t=0 (rows 30..61, cols 8..27)
LET = np.zeros((H, W), np.float32)
LET[40:70, 20:56] = 1.0                                 # the letters (type matte)
MOVE = {"dx": 0.25, "dy": 0.0, "scale": 1.0, "ease": "linear", "anchor": [0.5, 0.5], "feather": 0.0}      # +16 px at the last frame
PAN0 = {"dx": 0.0, "dy": 0.0, "scale": 1.0}


def u8(c, n=N):
    return np.broadcast_to(np.clip(np.rint(np.asarray(c, np.float32) * 255), 0, 255).astype(np.uint8), (n, H, W, 3)).copy()


def clean_clip():
    c = u8(GREEN)
    c[:, SUBJ > 0.5] = np.clip(np.rint(RED * 255), 0, 255).astype(np.uint8)
    return c


def matte_ref():
    return np.broadcast_to(SUBJ, (N, H, W)).copy()


def run_bake(mode=None, move=MOVE, pan=PAN0, color=False):
    return PLT.bake(u8(GREEN), clean_clip(), matte_ref(), move=move, pan=pan, color=color, mode=mode, letters=(lambda j: LET) if mode else None)


def red(img):
    """mask of the subject's pixels in a baked subject / picture (the clean background is green)"""
    f = np.asarray(img).astype(np.int32)
    return (f[..., 0] > 128) & (f[..., 1] < 64)


def centroid(m):
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


# ============================================================================================ the shot keys
def test_defaults_are_the_probe_composite_numbers():
    assert PLT.norm_move(None)["dx"] == 0.08 and PLT.norm_move({})["scale"] == 1.10 and PLT.norm_move(None)["anchor"] == [0.5, 0.75]
    p = PLT.norm_pan(None)
    assert p["dx"] == 0.035 and p["scale"] == 1.12


def test_plate_pan_scale_always_covers_the_pan():
    p = PLT.norm_pan({"dx": 0.2, "dy": 0.0, "scale": 1.05})
    assert p["scale"] >= 1.0 + 2 * 0.2                       # scale >= 1 + 2 x pan (CLEAN_PLATE.md section 5.5)
    assert PLT.norm_pan({"dx": 0.0, "scale": 1.0})["scale"] == 1.0
    m = PLT.norm_move({"dx": 9, "dy": "x", "scale": 99, "ease": "bounce"})
    assert m["dx"] == 0.5 and m["dy"] == 0.0 and m["scale"] == 3.0 and m["ease"] == "linear"


def test_move_curves_start_at_identity_and_end_at_the_given_values():
    m = PLT.norm_move({"dx": 0.1, "dy": -0.05, "scale": 1.2, "ease": "ease_in_out"})
    assert PLT.move_at(m, 0.0) == (0.0, 0.0, 1.0)
    dx, dy, s = PLT.move_at(m, 1.0)
    assert (round(dx, 6), round(dy, 6), round(s, 6)) == (0.1, -0.05, 1.2)
    assert 0.0 < PLT.move_at(m, 0.5)[0] < 0.1 and abs(PLT.move_at(m, 0.5)[0] - 0.05) < 1e-9          # smoothstep is symmetric
    pan = PLT.norm_pan({"dx": 0.03, "scale": 1.1})
    assert PLT.pan_at(pan, 0.0)[0] == 0.03 and PLT.pan_at(pan, 1.0)[0] == -0.03                   # the plate pans from +dx to -dx


def test_wants_plate_and_text():
    p = {"shots": [{"id": "s1", "plate": "clean"}, {"id": "s2"}, {"id": "s3", "type_matte": "reveal_plate"}, {"id": "s4", "type_matte": "window_other"}]}
    assert [PLT.wants_plate(p, f"s{k}") for k in (1, 2, 3, 4)] == [True, False, True, False]        # a baked type-matte mode implies the plate; window_other does not
    assert PLT.plate_sids(p) == ["s1", "s3"]
    assert not PLT.wants_plate(dict(p, layers={"plate": False}), "s1")                              # the piece opts out
    assert LS.wants_text(p, "s1") is True and LS.wants_text(p, "s1", plates=True) is False        # a plate shot has no text-rendered background while the graph's switch is on
    assert LS.wants_plate(p, "s1", plates=False) is False and LS.wants_plate(p, "s1", plates=True) is True
    assert PLT.norm_type_matte("Reveal-Subject") == "reveal_subject" and PLT.norm_type_matte("nope") is None


def test_a_word_can_choose_the_type_matte_of_its_shot():
    p = {"fps": 24, "layout": {"s1": {"start": 0, "frames": 48}, "s2": {"start": 48, "frames": 48}},
         "shots": [{"id": "s1"}, {"id": "s2"}], "type": {"words": [{"text": "A", "hit": 1.0}, {"text": "B", "hit": 3.0, "type_matte": "subject_through"}]}}
    assert PLT.type_matte_of(p, "s1") is None and PLT.type_matte_of(p, "s2") == "subject_through"
    assert PLT.wants_plate(p, "s2")


def test_layer_clip_plate():
    torch = pytest.importorskip("torch")
    clean = {"u8": torch.full((9, 4, 4, 3), 10, dtype=torch.uint8), "sid": "s2", "start": 5, "fps": 24}
    plate = {"u8": torch.full((9, 4, 4, 3), 99, dtype=torch.uint8), "sid": "s2", "start": 5, "fps": 24}
    c = LS.layer_clip(clean, None, plate)
    assert int(c["u8"][0, 0, 0, 0]) == 99 and int(c["subject"][0, 0, 0, 0]) == 10 and c["textbg"] is False and c["plate"] is True
    assert "plate" not in LS.layer_clip(clean, None)


# ============================================================================================ geometry, matte, colour
def test_ref_to_clip_maps_a_cover_cropped_matte_back():
    clip_hw = (120, 100)                                         # the clip is wider than the 9:16 reference: the reference shows its centred window
    ref = np.zeros((160, 90), np.float32)                       # 160x90 reference (9:16): scale = max(90/100, 160/120) = 1.333, window = 120 x 67.5 clip px
    ref[:, 45:] = 1.0                                           # right half of the window
    m = PLT.ref_to_clip(ref, clip_hw)
    assert m.shape == clip_hw
    assert m[:, :16].max() == 0.0 and m[:, 84:].min() == 1.0    # outside the window the matte's own border is repeated (left 0, right 1)
    assert m[60, 44] == 0.0 and m[60, 56] == 1.0               # the window's centre column (the matte's own centre) is the clip's centre column 50
    same = PLT.ref_to_clip(np.ones((96, 64), np.float32), (96, 64))
    assert same.shape == (96, 64) and same.min() == 1.0


def test_soften_erodes_and_feathers_but_keeps_the_body():
    m = np.zeros((200, 200), np.float32)
    m[60:140, 60:140] = 1.0
    s = PLT.soften(m, 0.01)
    assert s[100, 100] == 1.0 and 0.0 < s[60, 100] < 1.0 and s[40, 100] == 0.0
    assert s.sum() < m.sum() * 1.02                              # eroded + blurred: no growth (no halo of the clean clip's background)
    assert np.array_equal(PLT.soften(m, 0.0), m)


def test_color_fit_recovers_the_plate_drift_on_the_background():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.8, (N, H, W, 3)).astype(np.float32)
    clean = np.clip(np.rint(base * 255), 0, 255).astype(np.uint8)
    drift = np.clip(base * 1.04 + 0.03, 0, 1)                    # the plate: gain 1.04, offset +0.03 (the LoRA regenerates the whole frame)
    plate = np.clip(np.rint(drift * 255), 0, 255).astype(np.uint8)
    f = PLT.fit_color(clean, plate, matte_ref())
    assert f["ok"] and all(abs(g - 1.04) < 0.02 for g in f["gain"]) and all(abs(o - 0.03) < 0.02 for o in f["offset"])
    none = PLT.fit_color(clean, plate, np.ones((N, H, W), np.float32))        # a matte covering everything: no background to fit on -> identity
    assert not none["ok"] and none["gain"] == [1.0] * 3


def test_bake_colour_matches_the_subject_to_the_plate():
    plate = u8(np.clip(GREEN + 0.04, 0, 1))                      # the plate drifted by +10/255
    r = PLT.bake(plate, clean_clip(), matte_ref(), move={"dx": 0, "dy": 0, "scale": 1.0, "ease": "linear", "anchor": [0.5, 0.5], "feather": 0.0}, pan=PAN0, color=True)
    assert r["info"]["color"]["ok"]
    got = r["subject"][0, 45, 18].astype(np.float32) / 255.0      # inside the subject at t=0 (no move)
    assert np.allclose(got, np.clip(RED + 0.04, 0, 1), atol=0.02), got


# ============================================================================================ the matte follows its picture
def test_matte_follows_the_subject():
    r = run_bake()
    x0, y0 = centroid(SUBJ > 0.5)
    for j in range(N):
        t = j / (N - 1)
        sx, sy = centroid(red(r["subject"][j]))           # where the subject's pixels are (everything else is 0)
        mx, my = centroid(r["matte"][j, ..., 0] > 127)
        assert abs(sx - mx) < 0.6 and abs(sy - my) < 0.6, (j, sx, mx, sy, my)    # SAME transform: picture and matte coincide
        assert abs(mx - (x0 + 0.25 * W * t)) < 0.8                                 # and it is where the move says (dx 25 % of the width at the last frame)
    assert (r["matte"][0, ..., 0] > 127).sum() == (SUBJ > 0.5).sum()               # frame 0 = identity


def test_matte_follows_the_subject_with_scale_and_other_frame_counts():
    mv = PLT.norm_move({"dx": -0.1, "dy": 0.1, "scale": 1.25, "ease": "ease_in_out", "anchor": [0.3, 0.8], "feather": 0.0})
    short_matte = matte_ref()[:3]                                                  # a matte clip of another length is indexed by the same fraction of the clip
    r = PLT.bake(u8(GREEN), clean_clip(), short_matte, move=mv, pan=PAN0, color=False)
    for j in range(N):
        sx, sy = centroid(red(r["subject"][j]))
        mx, my = centroid(r["matte"][j, ..., 0] > 127)
        assert abs(sx - mx) < 0.8 and abs(sy - my) < 0.8


def test_the_plate_moves_on_its_own_and_the_subject_does_not_follow_it():
    pan = PLT.norm_pan({"dx": 0.1, "scale": 1.3})
    r = PLT.bake(u8(GREEN), clean_clip(), matte_ref(), move={"dx": 0, "dy": 0, "scale": 1.0, "ease": "linear", "anchor": [0.5, 0.5], "feather": 0.0}, pan=pan, color=False)
    assert r["info"]["first"][3] > 0 > r["info"]["last"][3] and r["info"]["first"][0] == r["info"]["last"][0] == 0.0     # plate +dx px -> -dx px, subject fixed
    assert red(r["subject"][N - 1]).sum() == (SUBJ > 0.5).sum()


def test_depth_is_warped_with_the_plate_only():
    d = np.zeros((N, 24, 16, 3), np.float32)
    d[:, :, 6:10] = 1.0
    r = PLT.bake(u8(GREEN), clean_clip(), matte_ref(), move=PLT.norm_move({"dx": 0.5}), pan={"dx": 0.1, "dy": 0.0, "scale": 1.3, "ease": "linear"}, depth=d, color=False)
    assert r["depth"].shape == d.shape and not np.array_equal(r["depth"][0], r["depth"][N - 1]) and r["depth"].dtype == np.float32


# ============================================================================================ type as a matte
def _moved_subject_mask():
    c = PLT.bake(u8(GREEN), clean_clip(), matte_ref(), move=MOVE, pan=PAN0, color=False)
    return c["matte"][N - 1, ..., 0] > 200


def _is(img, col, tol=0.05):
    return np.abs(img.astype(np.float32) / 255.0 - col).max(-1) < tol


def test_type_matte_reveal_subject_subject_only_inside_the_letters():
    r = run_bake("reveal_subject")
    j = N - 1
    ms = _moved_subject_mask()
    inner = cv2.erode(ms.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let_in = cv2.erode((LET > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let_out = cv2.erode((LET < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    pic = r["footage"][j]
    assert _is(pic, RED)[inner & let_in].all()                    # subject AND letters: the subject shows (the full clip inside the word)
    assert _is(pic, GREEN)[inner & let_out].all()                 # subject outside the letters: gone (the clean plate)
    assert r["matte"][..., 0].max() == 0                           # nothing stands in front of the type
    assert np.array_equal(r["subject"], r["footage"])
    assert not np.array_equal(r["plate_moved"], r["footage"])      # the moved layers are kept for the archive


def test_type_matte_reveal_plate_the_word_erases_the_subject():
    r = run_bake("reveal_plate")
    j = N - 1
    ms = _moved_subject_mask()
    inner = cv2.erode(ms.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let_in = cv2.erode((LET > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let_out = cv2.erode((LET < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    pic = r["footage"][j]
    assert _is(pic, GREEN)[inner & let_in].all()                  # the letters show the clean plate where the subject would be
    assert _is(pic, RED)[inner & let_out].all()                   # outside the letters the subject is there
    assert r["matte"][..., 0].max() == 0


def test_type_matte_subject_through_matte_is_subject_times_letters():
    r = run_bake("subject_through")
    j = N - 1
    ms = _moved_subject_mask()
    plain = PLT.bake(u8(GREEN), clean_clip(), matte_ref(), move=MOVE, pan=PAN0, color=False)
    m = r["matte"][j, ..., 0].astype(np.float32) / 255.0
    pm = plain["matte"][j, ..., 0].astype(np.float32) / 255.0
    assert np.allclose(m, pm * LET, atol=1.01 / 255)               # the subject layer in front of the type only where it overlaps the letters
    assert m[LET < 0.5].max() == 0 and m[(~ms)].max() < 0.2
    assert _is(r["footage"][j], GREEN).all()                       # the picture under the type stays the plate


def test_type_matte_needs_the_letters():
    with pytest.raises(ValueError):
        PLT.bake(u8(GREEN), clean_clip(), matte_ref(), mode="reveal_plate", letters=None)


def test_letter_reader_reads_the_type_matte_pass_frame_exact():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "matte"
        d.mkdir()
        for i in range(10):
            a = np.zeros((H, W), np.uint8)
            a[i * 5:i * 5 + 5, :] = 255                            # a band that moves down 5 px per piece frame
            cv2.imwrite(str(d / f"matte_{i:05d}.png"), a)
        rd = PLT.LetterReader(tmp, offset=3, hw=(H, W), soft=0.0)
        for j in (0, 2):
            m = rd(j)
            assert m.shape == (H, W) and m[(3 + j) * 5 + 2, 0] == 1.0 and m[(3 + j) * 5 - 3, 0] == 0.0     # shot frame j <-> piece frame offset + j
        assert PLT.LetterReader(tmp, 99, (H, W), 0.0)(0)[9 * 5 + 2, 0] == 1.0                              # past the end: the last frame
        assert PLT.LetterReader(tmp + "/none", 0, (H, W))(0).max() == 0.0 and not PLT.LetterReader(tmp + "/none", 0, (H, W)).files


# ============================================================================================ the compose layer order
SANDWICH = [
    {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "type_rgb": [1.0, 1.0, 1.0], "type_a": "$type_matte", "matte": "$matte",
                                                "matte_threshold": 0.5, "matte_soft": 0.08}},
]


def _piece():
    return {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]}, "shots": [{"id": "s1", "frames": N, "matte": True}],
            "compose": {"crossfade": 0.0, "default": {"ops": SANDWICH}}}


def _compose(r):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "matte"
        d.mkdir()
        for i in range(N):
            cv2.imwrite(str(d / f"matte_{i:05d}.png"), (LET * 255).astype(np.uint8))
        out, _ = RE.run_compose(_piece(), "s1", type_dir=tmp, footage=r["footage"], matte=r["matte"], backend="numpy", threads=0, subject=r["subject"])
    return np.asarray(out, np.uint8)[N - 1].astype(np.float32) / 255.0


def test_compose_layer_order_plate_then_type_then_the_moved_subject():
    r = run_bake()
    out = _compose(r)
    ms = cv2.erode(_moved_subject_mask().astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    old = cv2.erode((SUBJ > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let = cv2.erode((LET > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.abs(out[ms & let] - RED).max() < 0.06                 # the moved subject stands in FRONT of the type where they overlap
    assert np.abs(out[let & ~ms & ~old & (np.arange(W)[None, :] > 47)] - 1.0).max() < 0.06      # the type over the plate where the subject is not
    assert np.abs(out[old & (cv2.dilate(_moved_subject_mask().astype(np.uint8), np.ones((5, 5), np.uint8)) == 0) & (cv2.dilate(LET, np.ones((5, 5), np.uint8)) == 0)] - GREEN).max() < 0.06       # where the subject WAS: the clean plate (the subject really left)
    assert np.abs(out[:8, :8] - GREEN).max() < 0.06


def test_compose_subject_through_makes_the_word_a_window_onto_the_subject():
    out = _compose(run_bake("subject_through"))
    ms = cv2.erode(_moved_subject_mask().astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    let = cv2.erode((LET > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.abs(out[ms & let] - RED).max() < 0.06                 # subject visible through the letters
    assert np.abs(out[ms & ~let & ~(LET > 0.0)] - GREEN).max() < 0.06     # outside the letters the subject is not drawn: the plate


# ============================================================================================ the prompt
def test_empty_scene_prompt_names_the_subject_and_keeps_the_place():
    piece = {"shots": [{"id": "s1", "prompt": "A woman in a red vest paddles a kayak across a calm lake, warm golden light, mist on the water, slow dolly in, cinematic"}]}
    p = PLT.empty_scene_prompt(piece, "s1")
    assert p.startswith("An empty clean plate of the exact same location") and "woman in a red vest" in p and "kayak" in p
    assert "calm lake" in p and "golden light" in p and "dolly" not in p and "cinematic" not in p
    assert PLT.empty_scene_prompt({"shots": [{"id": "s1", "prompt": "x", "plate_prompt": "EMPTY"}]}, "s1") == "EMPTY"
    assert "No dancer" in PLT.empty_scene_prompt({"shots": [{"id": "s1", "prompt": "A runner sprints through a tunnel", "plate_subject": "dancer"}]}, "s1")
    runner = PLT.empty_scene_prompt({"shots": [{"id": "s1", "prompt": "A runner sprints toward the camera through a wet cyan tunnel, LED strips streaking past overhead"}]}, "s1")
    assert "No runner" in runner and "tunnel" in runner and "toward the camera" not in runner


# ============================================================================================ the clip dict (compose node, archive)
def test_bake_clip_moves_the_clip_and_exposes_every_component():
    torch = pytest.importorskip("torch")
    piece = {"fps": 24, "layout": {"s1": {"start": 0, "frames": N, "xf": 0}}, "shots": [{"id": "s1", "plate": "clean", "subject_move": MOVE, "plate_pan": PAN0}]}
    clip = {"u8": torch.from_numpy(u8(GREEN)), "subject": torch.from_numpy(clean_clip()), "sid": "s1", "start": 0, "fps": 24, "plate": True, "textbg": False}
    c2, m2, d2, info = PLT.bake_clip(piece, clip, torch.from_numpy(matte_ref()), None)
    assert c2["plate_baked"] and c2["textbg"] is False and m2.shape == (N, H, W, 1) and m2.dtype == torch.uint8 and d2 is None
    pl = c2["plate_layers"]
    assert set(pl) >= {"plate_raw", "clean", "plate_moved", "subject_moved", "matte_moved", "type_matte", "move", "pan", "color"}
    assert pl["plate_raw"] is clip["u8"] and pl["clean"] is clip["subject"] and pl["matte_moved"] is m2
    # skipped without a matte / not a plate shot: the clip comes back as it was
    assert PLT.bake_clip(piece, clip, None, None)[0] is clip and PLT.bake_clip(piece, dict(clip, plate=False), m2, None)[0]["plate"] is False
    # a type-matte mode without letters on disk is skipped, not crashed
    p2 = dict(piece, shots=[dict(piece["shots"][0], type_matte="reveal_plate")])
    assert "skipped" in PLT.bake_clip(p2, clip, torch.from_numpy(matte_ref()), None, "/no/such/dir")[3]


# ============================================================================================ the graph
@pytest.fixture(scope="module")
def layered():
    g = BG.build_graph(False, "moss", True, True)
    side = BG.build_sidecar(False, "moss", True, True)
    return LS.convert(g, side)


def test_plate_chain_is_in_the_layered_graph_and_checks_clean(layered):
    g, side = layered
    assert LS.check(g) == [] and side["layers"]["plate_lora"] == PLT.LORA
    assert g["pl.5"]["class_type"] == "LTXICLoRALoaderModelOnly" and g["pl.5"]["inputs"]["lora_name"] == "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
    assert g["a.5"]["inputs"]["lora_name"] != g["pl.5"]["inputs"]["lora_name"]                       # Alpha-Gen keeps its own LoRA
    for k in range(1, 7):
        assert g[f"p{k}.34"]["inputs"]["model"] == ["pl.5", 0] and g[f"a{k}.34"]["inputs"]["model"] == ["a.5", 0]
        assert g[f"p{k}.17"]["inputs"]["input"] == [f"pl{k}.ref", 0]                                    # the plate reads the CLEAN clip, not the gather
        assert g[f"pl{k}.ref"]["class_type"] == "MographPlateRef" and g[f"pl{k}.ref"]["inputs"]["clean"] == ["lay_clean", 0]
        assert g[f"pl{k}.prompt"]["class_type"] == "MographPlatePrompt" and g[f"pl{k}.6"]["inputs"]["text"] == [f"pl{k}.prompt", 0]
        assert g[f"p{k}.dec"]["inputs"]["gray"] is False and g["gather"]["inputs"][f"plate_{k}"] == [f"p{k}.dec", 0]
        assert g[f"pl{k}.ref"]["inputs"][f"text_{k}"] == [f"t{k}.dec", 0]                                # the plate stage waits for the text passes
    assert g["gather"]["inputs"]["plates"] is False                                                     # off until a tier switches it on


def test_only_the_gather_pulls_the_plate_chain_so_it_is_lazy(layered):
    g, _ = layered
    chain = {n for n in g if n.startswith(("p1.", "p2.", "p3.", "p4.", "p5.", "p6.", "pl")) and not n.startswith("plan")}
    assert chain
    users = {}
    for nid, n in g.items():
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and v[0] in chain and nid not in chain:
                users.setdefault(v[0], set()).add(nid)
    assert users == {f"p{k}.dec": {"gather"} for k in range(1, 7)}


def test_plate_false_leaves_the_layered_graph_as_it_was():
    g = BG.build_graph(False, "moss", True, True)
    g2, _ = LS.convert(g, None, plate=False)
    assert not [n for n in g2 if n.startswith("pl") and not n.startswith("plan")] and "plates" not in g2["gather"]["inputs"] and LS.check(g2) == []


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers_pivot.json").exists(),
                    reason="dev fixture not shipped in the public export: app/graphs/fusion_design_qwen_auto_layers_pivot.json")
def test_tier_24_switches_plates_on_and_the_others_do_not_angles_stay_out_of_the_default_graphs():
    import json
    from tools.app import llm_arms as LA
    base = json.load(open(ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers_pivot.json", encoding="utf-8"))
    side = json.load(open(ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers_pivot.app.json", encoding="utf-8"))
    assert base["gather"]["inputs"]["plates"] is False and "angles" not in base["first_set"]["inputs"]      # angles: opt-in (build_graphs --angles), the default graphs need no edit LoRA
    for arm, on in (("gemma31b_qat", True), ("gemma31b_qat", True), ("gemma12b_q8", False), ("gemma12b_q8", False)):
        g, _ = LA.convert(base, side, arm)
        assert g["gather"]["inputs"]["plates"] is on and "angles" not in g["first_set"]["inputs"], arm
        assert LA.check(g, arm, base) == []


# ============================================================================================ the nodes (need a ComfyUI checkout)
def _node_modules():
    try:
        from test_layered_shots import _comfy
        _comfy()
        from nodes_fusion import compose as NC
        from nodes_fusion import layers as NL
    except pytest.skip.Exception:
        raise
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"nodes_fusion does not import here: {type(e).__name__}: {e}")
    return NL, NC


def _out(no):
    return no.args if hasattr(no, "args") else tuple(no)


def test_gather_hands_the_plate_on_only_for_plate_shots_and_only_with_the_switch():
    torch = pytest.importorskip("torch")
    NL, NC = _node_modules()
    piece = dict(_piece(), layout={"s1": {"start": 0, "frames": N, "xf": 0}, "s2": {"start": N, "frames": N, "xf": 0}},
                 shots=[{"id": "s1", "frames": N, "matte": True, "plate": "clean"}, {"id": "s2", "frames": N, "matte": True}])
    mk = lambda v, sid: {"u8": torch.full((N, H, W, 3), v, dtype=torch.uint8), "sid": sid, "start": int(piece["layout"][sid]["start"]), "fps": 24}   # noqa: E731
    clean = [mk(40, "s1"), mk(40, "s2")]
    none = {f"text_{k}": None for k in range(1, 7)} | {f"plate_{k}": None for k in range(1, 7)}
    on = NL.MographLayerGather.check_lazy_status(piece=piece, request={}, clean=clean, plates=True, **none)
    off = NL.MographLayerGather.check_lazy_status(piece=piece, request={}, clean=clean, plates=False, **none)
    assert set(on) == {"plate_1", "text_2"}                           # the plate shot asks for its plate and no text pass; the ordinary shot for its text pass
    assert set(off) == {"text_1", "text_2"}                           # switch off: ordinary layered shots, no plate chain is ever requested
    texts = {"text_2": mk(220, "s2"), "plate_1": mk(99, "s1")}
    out = _out(NL.MographLayerGather.execute(piece, {}, None, clean, plates=True, **texts))
    by = {c["sid"]: c for c in out[0]}
    assert by["s1"]["plate"] is True and int(by["s1"]["u8"][0, 0, 0, 0]) == 99 and int(by["s1"]["subject"][0, 0, 0, 0]) == 40 and by["s1"]["textbg"] is False
    assert "plate" not in by["s2"] and int(by["s2"]["u8"][0, 0, 0, 0]) == 220 and by["s2"]["textbg"] is True
    assert out[1].shape[0] == N and abs(float(out[1].mean()) - 40 / 255.0) < 0.01          # the Alpha-Gen reference of the plate shot is the CLEAN clip, not the plate
    ref = _out(NL.MographPlateRef.execute(piece, clean, 1))[0]
    assert ref.shape[0] == N and abs(float(ref.mean()) - 40 / 255.0) < 0.01
    assert NL.MographPlateRef.check_lazy_status(piece=piece, clean=clean, index=1, **{f"text_{k}": None for k in range(1, 7)}) == ["text_2"]      # the plate waits for the text passes
    assert "empty clean plate" in _out(NL.MographPlatePrompt.execute({"shots": [{"id": "s1", "prompt": "A runner sprints through a tunnel, cold light"}]}, 1))[0]


def test_compose_node_bakes_the_plate_shots_and_leaves_the_others():
    torch = pytest.importorskip("torch")
    NL, NC = _node_modules()
    piece = dict(_piece(), layout={"s1": {"start": 0, "frames": N, "xf": 0}}, shots=[{"id": "s1", "frames": N, "matte": True, "plate": "clean", "subject_move": MOVE, "plate_pan": PAN0}])
    clip = {"u8": torch.from_numpy(u8(GREEN)), "subject": torch.from_numpy(clean_clip()), "sid": "s1", "start": 0, "fps": 24, "plate": True, "textbg": False}
    other = {"u8": torch.from_numpy(u8(GREEN)), "subject": torch.from_numpy(u8(GREEN)), "sid": "s2", "start": N, "fps": 24, "textbg": False}
    clips, mattes, depths, info = NC.bake_plate_clips(piece, [clip, other], {"s1": torch.from_numpy(matte_ref())}, {})
    assert clips[1] is other and set(info) == {"s1"} and mattes["s1"].shape == (N, H, W, 1) and clips[0]["plate_baked"]
    assert NC.bake_plate_clips(piece, clips, mattes, depths)[0][0] is clips[0]            # baked once
    assert "plate_layers" in clips[0]
    src = __import__("inspect").getsource(NC.compose_clips)
    assert "bake_plate_clips(" in src and 'out[-1]["plate_layers"]' in src                # the components ride on the composed clip for the archive


# ============================================================================================ how a shot asks (concept entry / probe override)
def test_shot_keys_of_a_concept_entry_are_normalised():
    k = PLT.shot_keys({"prompt": "x", "plate": "clean", "subject_move": {"dx": 9, "scale": 1.2}, "plate_pan": {"dx": 0.1}, "type_matte": "Reveal-Plate", "plate_subject": "the skater", "framing": "wide"})
    assert k["plate"] == "clean" and k["type_matte"] == "reveal_plate" and k["subject_move"]["dx"] == 0.5 and k["plate_pan"]["scale"] >= 1.2 and k["plate_subject"] == "the skater"
    assert "prompt" not in k and "framing" not in k
    assert PLT.shot_keys({"prompt": "x"}) == {} and PLT.shot_keys(None) == {}
    assert PLT.shot_keys({"type_matte": "window_other"}) == {"type_matte": "window_other"}               # a cut option: nothing else is carried
    assert PLT.shot_keys({"subject_move": {"dx": 0.1}}) == {}                                          # a move without a plate asks for nothing


def test_probe_override_file_merges_keys_over_the_piece(tmp_path, monkeypatch):
    piece = {"slug": "abc", "shots": [{"id": "s1"}, {"id": "s2"}]}
    f = tmp_path / "keys.json"
    f.write_text(json.dumps({"*": {"s2": {"plate": "clean", "type_matte": "subject_through", "prompt": "ignored"}}, "abc": {"s1": {"plate": "clean"}}}), encoding="utf-8")
    assert not PLT.wants_plate(piece, "s2")                                                             # no variable: no effect
    monkeypatch.setenv(PLT.OVERRIDE_ENV, str(f))
    assert PLT.wants_plate(piece, "s1") and PLT.wants_plate(piece, "s2") and PLT.type_matte_of(piece, "s2") == "subject_through"
    assert "prompt" not in PLT.shot_of(piece, "s2") and not PLT.wants_plate(dict(piece, slug="other"), "s1")      # only the shot keys, and the slug scoped entry only for its piece
    f.write_text("{not json", encoding="utf-8")
    assert not PLT.wants_plate(piece, "s1")                                                             # a broken file is ignored


def test_static_validation_of_the_plate_nodes_against_graphs_the_server_ran(layered):
    from tools.app import graph_signatures as GS
    sigs = GS.proven_signatures()
    if "LTXICLoRALoaderModelOnly" not in sigs:
        pytest.skip("24_clean-plate/v001/graphs not present")
    g, _ = layered
    assert GS.check(g, sigs, prefixes=("pl.", "pl1", "pl6", "p1.", "p6.")) == []
    # the clean-plate graph the box ran uses the same classes as the chain copied from Alpha-Gen
    assert {n["class_type"] for k, n in g.items() if k.startswith("p3.")} <= set(sigs) | {"MographDecodeClip"}
# ============================================================================================ render R17: flat-ground sections, ground colour, plate_in_type
# R17 (drummer stills, "more legible text, palette: crimson, gold"): type_matte plate_in_subject ran over the WHOLE 4.5 s shot, flat mid-grey (98,100,96) ground, ghost drummer.
CRIMSON, GOLD = [190, 20, 50], [230, 180, 40]
DARK = np.array([0.27, 0.27, 0.27], np.float32)            # a dark photo: mean about 70 / 255
T45 = 108                                                   # 4.5 s at 24 fps


def r17_piece(mode="plate_in_subject", hit=1.5, release=3.0, palette=True, frames=T45):
    p = {"fps": 24, "layout": {"s1": {"start": 0, "frames": frames, "xf": 0}},
         "shots": [{"id": "s1", "plate": "clean", "type_matte": mode, "subject_move": MOVE, "plate_pan": PAN0}],
         "type": {"duration": frames / 24.0, "words": [{"text": "FINAL", "hit": hit, "release": release}],
                  "palette": {"fg": [232, 229, 223], "bg": [19, 19, 16], "accents": [[98, 100, 96], [62, 61, 55], [146, 144, 138]]}}}
    if palette:
        p["user_style"] = {"palette": [CRIMSON, GOLD]}
    return p


def r17_clip(n=T45):
    torch = pytest.importorskip("torch")
    plate = np.broadcast_to(np.clip(np.rint(DARK * 255), 0, 255).astype(np.uint8), (n, H, W, 3)).copy()
    clean = plate.copy()
    clean[:, SUBJ > 0.5] = np.clip(np.rint((DARK + 0.02) * 255), 0, 255).astype(np.uint8)
    return {"u8": torch.from_numpy(plate), "subject": torch.from_numpy(clean), "sid": "s1", "start": 0, "fps": 24, "plate": True, "textbg": False}, plate


def r17_bake(piece, n=T45):
    torch = pytest.importorskip("torch")
    clip, plate = r17_clip(n)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "matte"
        d.mkdir()
        cv2.imwrite(str(d / "matte_00000.png"), (LET * 255).astype(np.uint8))        # one frame: the reader clamps to the last file
        c2, m2, _, info = PLT.bake_clip(piece, clip, torch.from_numpy(np.broadcast_to(SUBJ, (N, H, W)).copy()), None, tmp)
    return c2, m2, info, plate


def test_flat_ground_only_inside_the_words_window_of_a_long_shot():
    piece = r17_piece()
    assert PLT.contrast_sections(piece) == {"s1": [(1.5, 3.0)]}
    c2, m2, info, plate = r17_bake(piece)
    pic = c2["u8"].numpy()
    ground = np.array(piece["plan"]["flat_sections"]["s1"]["ground"]["rgb"], np.uint8)
    far = (2, 2)                                                                       # outside the letters and the subject
    inside = [j for j in range(T45) if 1.5 <= (j + 0.5) / T45 * 4.5 <= 3.0]
    outside = [j for j in range(T45) if (j + 0.5) / T45 * 4.5 < 1.5 - PLT.FLAT_EASE_S - 0.05 or (j + 0.5) / T45 * 4.5 > 3.0 + PLT.FLAT_EASE_S + 0.05]
    assert len(inside) >= 30 and len(outside) >= 40
    assert all(np.abs(pic[j][far].astype(int) - ground.astype(int)).max() <= 2 for j in inside), "the flat ground fills the hold"
    assert all(np.array_equal(pic[j], plate[j]) for j in outside), "outside the section the shot shows its real footage"
    assert not any(np.array_equal(pic[j][far], plate[j][far]) for j in inside)
    assert int(m2[outside[0]].sum()) > 0 and int(m2[inside[len(inside) // 2]].sum()) == 0          # the subject matte stays on the real footage, is empty in the one-picture hold
    assert c2["subject"] is not c2["u8"] and info["window_frames"] < T45                  # the frames outside keep their own subject layer
    mid = next(j for j in range(T45) if (j + 0.5) / T45 * 4.5 >= 1.4)                 # half way up the ease in (1.3 s .. 1.5 s)
    assert 0 < PLT.window_weights([(1.5, 3.0)], 0.0, 4.5, T45)[mid] < 1 and 0 < abs(int(pic[mid][far][0]) - int(plate[mid][far][0])) < abs(int(ground[0]) - int(plate[mid][far][0]))    # dissolves, no jump


def test_flat_sections_stay_under_40_percent_of_the_piece():
    piece = r17_piece(hit=0.3, release=4.4)                                            # the hold would cover almost the whole 4.5 s shot
    secs = PLT.contrast_sections(piece)["s1"]
    assert sum(b - a for a, b in secs) <= PLT.FLAT_SHARE_MAX * 4.5 + 0.01 and secs[0][0] == 0.3
    two = r17_piece()
    two["layout"]["s2"] = {"start": T45, "frames": T45, "xf": 0}
    two["shots"].append({"id": "s2", "plate": "clean", "type_matte": "plate_in_subject"})
    two["type"]["words"].append({"text": "ECHO", "hit": 6.0, "release": 8.5})
    tot = sum(b - a for v in PLT.contrast_sections(two).values() for a, b in v)
    assert tot <= PLT.FLAT_SHARE_MAX * 9.0 + 0.01, tot                                  # 1.5 + a trimmed second section, within 0.4 x 9 s
    assert PLT.contrast_sections(r17_piece(mode=None)) == {}


def test_the_ground_comes_from_the_user_palette_and_contrasts():
    piece = r17_piece()
    c2, _, info, _ = r17_bake(piece)
    g = piece["plan"]["flat_sections"]["s1"]["ground"]
    assert g["source"] == "user_palette" and g["rgb"] != [98, 100, 96]
    fill = [(DARK[0] + 0.02) * 255] * 3
    assert PLT.contrast_ratio(g["rgb"], fill) >= PLT.GROUND_CR_MIN and abs(PLT.oklab_l(g["rgb"]) - PLT.oklab_l(fill)) >= PLT.GROUND_DL_MIN
    assert PLT.contrast_ratio(g["rgb"], g["ink"]) >= PLT.INK_CR_MIN                    # the word ink (and its contour lines) read on the ground
    no_user = r17_piece(palette=False)
    r17_bake(no_user)
    g2 = no_user["plan"]["flat_sections"]["s1"]["ground"]
    assert g2["source"] == "shot" and PLT.contrast_ratio(g2["rgb"], fill) >= PLT.GROUND_CR_MIN       # no palette typed: the shot's own colours, still a contrasting ground


def test_pick_ground_r17_numbers_never_a_close_mid_grey():
    fill = [73, 73, 73]                                                                # the R17 photo (subject mean about 70-73)
    grey = [98, 100, 96]
    assert PLT.contrast_ratio(grey, fill) < PLT.GROUND_CR_MIN                          # the R17 ground really was a close shade
    r = PLT.pick_ground([CRIMSON, GOLD, grey], [232, 229, 223], fill)
    assert r["rgb"] not in ([98, 100, 96],) and r["cr_fill"] >= PLT.GROUND_CR_MIN and r["dl_fill"] >= PLT.GROUND_DL_MIN
    assert r["cr_fill"] >= PLT.GROUND_CR_GOOD, r                                       # gold is clearly lighter than the photo
    assert r["cr_ink"] >= PLT.INK_CR_MIN and r["ink"] == CRIMSON, r                    # cream ink on gold would not read (1.6:1): the palette's crimson instead (3.3:1)
    only_crimson = PLT.pick_ground([CRIMSON], [232, 229, 223], fill)                    # no palette colour reaches it: crimson is pushed to the far end, hue kept
    assert only_crimson["pushed"] and only_crimson["cr_fill"] >= PLT.GROUND_CR_MIN and only_crimson["dl_fill"] >= PLT.GROUND_DL_MIN
    assert only_crimson["rgb"][0] > only_crimson["rgb"][1] + 20 and only_crimson["cr_ink"] >= PLT.INK_CR_MIN
    light = PLT.pick_ground([GOLD, [200, 190, 120]], [20, 20, 20], [225, 215, 190])      # a light photo: a deep ground
    assert PLT.contrast_ratio(light["rgb"], [225, 215, 190]) >= PLT.GROUND_CR_MIN and PLT.oklab_l(light["rgb"]) < PLT.oklab_l([225, 215, 190]) - PLT.GROUND_DL_MIN


def test_plate_in_type_makes_no_flat_ground_without_a_texture_wish():
    piece = r17_piece(mode="plate_in_type")
    c2, m2, info, plate = r17_bake(piece)
    assert info["type_matte"] is None and c2["type_matte"] is None and "flat_sections" not in (piece.get("plan") or {})
    assert all(np.array_equal(c2["u8"].numpy()[j], plate[j]) for j in range(0, T45, 9))      # plain plate over the whole shot: footage, no ground
    ask = r17_piece(mode="plate_in_type")
    ask["direction"] = "put the footage inside the letters"                              # the user DID ask for textured letters: the mode stays
    assert r17_bake(ask)[2]["type_matte"] == "plate_in_type"
