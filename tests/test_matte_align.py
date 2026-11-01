"""A matte is only ever laid over the exact picture it was cut from, frame for frame (user rule 2026-10-29, tools/fusion/matte_align.py). Synthetic disc subjects: after the geometry
the compose side applies, the matte's centroid must match the picture's subject centroid.
Run: python -m pytest tests/test_matte_align.py -q"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import ltx_check as LC  # noqa: E402
from tools.fusion import matte_align as MA  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402


def cover_resize(img, w, h):
    """what nodes_fusion/footage.py cover_resize does: scale to cover (w, h), centre-crop."""
    s = max(w / img.shape[1], h / img.shape[0])
    nw, nh = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def disc(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255


def centroid(m):
    m = np.asarray(m, np.float32)
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    s = float(m.sum())
    return float((xs * m).sum() / s), float((ys * m).sum() / s)


def footage_with_disc(n, h, w, cx_of, cy=None, r=None):
    cy = h // 2 if cy is None else cy
    r = r or w // 7
    out = np.zeros((n, h, w, 3), np.uint8)
    for j in range(n):
        out[j][disc(h, w, cx_of(j), cy, r) > 0] = 255
    return out


def matte_from_reference(footage, ref_wh):
    """the Alpha-Gen path: the reference IS the clip cover-resized to the reference size, the matte is the subject in that reference."""
    return np.stack([(cover_resize(f, ref_wh[0], ref_wh[1])[..., 0] > 127).astype(np.uint8) * 255 for f in footage])


def composed_matte(footage, matte, j, size, aligned=True):
    with RE.use_backend("numpy"):
        if aligned:
            src = RE.RefSource(matte, "gray", None, False, None, footage, size)
        else:
            src = RE.ArraySource(matte, "gray", None, False, None)
        m = src[j]
        up = C.upscale_matte(m, size)
        return np.asarray(C._out(up) if hasattr(C, "_out") else up, np.float32)


def fitted_footage_mask(footage, j, size):
    with RE.use_backend("numpy"):
        f = np.asarray(C._out(C.fit_cover(footage[j].astype(np.float32) / 255.0, size)), np.float32)
    return (f[..., 0] > 0.5).astype(np.float32)


@pytest.mark.parametrize("size", [(360, 640), (540, 960)])
def test_matte_cut_from_a_cover_cropped_reference_lands_on_the_subject(size):
    # a 360x640 clip, the reference 384x704 (footage._u8_to_ref cover-resizes it: 6 px of the width are cropped each side): the subject sits near the left edge, where a plain stretch is off most
    foot = footage_with_disc(5, 640, 360, lambda j: 55 + 3 * j)
    matte = matte_from_reference(foot, (384, 704))
    for j in (0, 4):
        want = centroid(fitted_footage_mask(foot, j, size))
        got = centroid(composed_matte(foot, matte, j, size, aligned=True))
        old = centroid(composed_matte(foot, matte, j, size, aligned=False))
        sc = size[0] / 360.0
        assert abs(got[0] - want[0]) < 0.8 * sc and abs(got[1] - want[1]) < 0.8 * sc, (got, want)
        assert abs(old[0] - want[0]) > abs(got[0] - want[0]) + 1.5 * sc, (old, got, want)        # the old stretch was visibly off here


def test_same_aspect_reference_is_untouched():
    foot = footage_with_disc(3, 640, 360, lambda j: 180)
    matte = matte_from_reference(foot, (180, 320))
    assert MA.canvas_hw((320, 180), (640, 360), (360, 640)) == (320, 180)
    a = composed_matte(foot, matte, 1, (360, 640), aligned=True)
    b = composed_matte(foot, matte, 1, (360, 640), aligned=False)
    assert np.allclose(a, b)


def test_a_matte_clip_of_another_length_is_indexed_by_time_not_clamped():
    n_f, n_m = 30, 40
    foot = footage_with_disc(n_f, 40, 100, lambda j: 8 + 2 * j, r=6)
    mfoot = footage_with_disc(n_m, 40, 100, lambda k: 8 + 2 * k * (n_f - 1) / (n_m - 1), r=6)      # the same motion sampled at 40 frames
    matte = np.stack([f[..., 0] for f in mfoot])
    with RE.use_backend("numpy"):
        src = RE.RefSource(matte, "gray", None, False, None, foot, (100, 40))
        assert len(src) == n_f
        for j in (0, 10, 29):
            want = centroid(foot[j][..., 0] / 255.0)
            got = centroid(np.asarray(C._out(src[j]), np.float32))
            assert abs(got[0] - want[0]) < 1.5, (j, got, want)
    # the old clamp: frame 29 of the footage read frame 29 of a 40-frame matte = 10 matte frames behind
    old = RE.ArraySource(matte, "gray", None, False, None)
    with RE.use_backend("numpy"):
        assert abs(centroid(np.asarray(C._out(old[29]), np.float32))[0] - centroid(foot[29][..., 0] / 255.0)[0]) > 8.0


def test_frame_index_and_canvas_pad_replicates_the_border():
    assert [MA.frame_index(j, 33, 30) for j in (0, 15, 29)] == [0, 17, 32] and MA.frame_index(5, 30, 30) == 5 and MA.frame_index(99, 10, 10) == 9
    m = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = MA.ref_to_frame(m, (30, 60), (80, 30))              # the window of the reference that is the compose frame: cropped in height, padded in width
    assert out.shape == MA.canvas_hw((3, 4), (30, 60), (80, 30)) and out.shape[0] < 3 < out.shape[1] and (out[:, 0] == out[:, 1]).all()
    wide = MA.ref_to_frame(np.ones((6, 4), np.uint8), (60, 60), (60, 60))      # a narrow reference, square frame: pad left/right with the edge
    assert wide.shape == (6, 6) and wide.min() == 1


# ------------------------------------------------------------------------------------------------- the two pictures of a layered shot
def textured_subject_clip(n, h, w, shift=0, seed=3):
    """a textured disc on a flat ground; `shift` px to the right in every frame."""
    rng = np.random.default_rng(seed)
    tex = np.roll(cv2.GaussianBlur(rng.integers(40, 255, (h, w), dtype=np.uint8), (0, 0), 1.2), shift, axis=1)       # the texture travels with the subject
    out = np.full((n, h, w, 3), 30, np.uint8)
    for j in range(n):
        m = disc(h, w, w // 2 + shift, h // 2, w // 5) > 0
        out[j][m] = tex[m][:, None]
    return out


def test_subject_drift_flags_a_shifted_background_subject():
    clean = textured_subject_clip(8, 192, 108)
    matte = np.stack([disc(192, 108, 54, 96, 21)] * 8)
    same = MA.subject_drift(clean, clean.copy(), matte)
    assert same["ok"] is True and same["px"] < 0.5
    near = MA.subject_drift(clean, textured_subject_clip(8, 192, 108, shift=1), matte)
    assert near["ok"] is True, near
    far = MA.subject_drift(clean, textured_subject_clip(8, 192, 108, shift=8), matte)
    assert far["ok"] is False and 6.5 < far["px"] < 9.5 and far["rel"] > MA.DRIFT_MAX, far
    assert MA.subject_drift(clean, clean, np.zeros((8, 192, 108), np.uint8))["ok"] is None            # no subject, no measurement


# ------------------------------------------------------------------------------------------------- the background of a layered shot
def _layered_piece():
    from test_ltx_read import _piece
    return _piece(("HANDS", "CLAY"), 6.0)


def test_clean_clip_is_the_background_when_no_ltx_word_is_left(tmp_path):
    p = _layered_piece()
    for w in p["type"]["words"]:
        w["render"] = "flat"
    clean = textured_subject_clip(4, 96, 54)
    text = textured_subject_clip(4, 96, 54, shift=2)
    clip = dict(sid="s1", u8=text, subject=clean, textbg=True)
    out, notes = LC.choose_background(p, "s1", clip, np.stack([disc(96, 54, 27, 48, 11)] * 4), tmp_path)
    assert out["u8"] is clean and out["textbg"] is False and any("no LTX-owned word is left" in n for n in notes)
    assert p["plan"]["owner"]["background"][0]["kept_textbg"] is False


def test_clean_clip_is_the_background_when_the_text_clip_drifts_and_the_ltx_words_go_flat(tmp_path):
    p = _layered_piece()
    clean = textured_subject_clip(6, 192, 108)
    matte = np.stack([disc(192, 108, 54, 96, 21)] * 6)
    drifted = dict(sid="s1", u8=textured_subject_clip(6, 192, 108, shift=9), subject=clean, textbg=True)
    q = copy.deepcopy(p)
    out, notes = LC.choose_background(q, "s1", drifted, matte, tmp_path)
    assert out["u8"] is clean and out["textbg"] is False and any("two people" in n for n in notes)
    assert [w["render"] for w in q["type"]["words"]] == ["flat", "flat"] and all(w["ltx_missing"] for w in q["type"]["words"])
    steady = dict(sid="s1", u8=textured_subject_clip(6, 192, 108, shift=0), subject=clean, textbg=True)
    r = copy.deepcopy(p)
    out2, notes2 = LC.choose_background(r, "s1", steady, matte, tmp_path)
    assert out2 is steady and notes2 == [] and [w["render"] for w in r["type"]["words"]] == ["handoff", "handoff"]
    assert r["plan"]["owner"]["background"][0]["kept_textbg"] is True


def test_non_layered_clip_passes_through():
    p = _layered_piece()
    clip = dict(sid="s1", u8=textured_subject_clip(2, 32, 18))
    assert LC.choose_background(p, "s1", clip) == (clip, [])


# ------------------------------------------------------------------------------------------------- a stateful plate chain cannot be layered: one picture
def test_run_compose_uses_the_clean_clip_when_the_layers_are_skipped(monkeypatch):
    seen = {}

    def fake_build(piece, sid, **kw):
        seen.setdefault("layered", []).append(kw.get("layered"))
        return {"layers": {"skipped": ["plate chain through plate_pan"]}} if kw.get("layered") else {"size": [8, 8]}

    def fake_open(rec, **kw):
        seen["footage"], seen["subject"] = kw["footage"], kw["subject"]
        raise RuntimeError("stop")

    monkeypatch.setattr(RE, "build_recipe", fake_build)
    monkeypatch.setattr(RE, "open_sources", fake_open)
    clean, text = np.zeros((2, 8, 8, 3), np.uint8), np.ones((2, 8, 8, 3), np.uint8)
    with pytest.raises(RuntimeError):
        RE.run_compose({}, "s1", type_dir=None, footage=text, subject=clean, backend="numpy")
    assert seen["layered"] == [True, False] and seen["footage"] is clean and seen["subject"] is None


# ------------------------------------------------------------------------------------------------- a transformed picture keeps its own transformed matte
def test_transform_applied_to_picture_and_matte_alike_keeps_them_aligned():
    """the rule for every site that moves a picture (pan, 3D cut, crossfade-free hard cut): picture and matte get the SAME transform. plate_pan is the op that moves a plate."""
    from tools.fusion.plate_pan import plate_pan
    foot = footage_with_disc(1, 160, 90, lambda j: 45, r=14)[0].astype(np.float32) / 255.0
    matte = (foot[..., 0] > 0.5).astype(np.float32)
    for dx, dy in ((0.2, 0.0), (-0.15, 0.1), (0.3, -0.25)):
        pf = plate_pan(foot, dx=dx, dy=dy, mode="wrap")
        pm = plate_pan(matte, dx=dx, dy=dy, mode="wrap")
        a, b = centroid(pf[..., 0] > 0.5), centroid(pm > 0.5)
        assert abs(a[0] - b[0]) < 0.6 and abs(a[1] - b[1]) < 0.6
    # the old pivot/orbit bug shape: a matte left at the old position is off by the whole move
    moved = plate_pan(foot, dx=0.3, dy=0.0, mode="wrap")
    assert abs(centroid(moved[..., 0] > 0.5)[0] - centroid(matte)[0]) > 20


# ------------------------------------------------------------------------------------------------- 3D cuts and window cuts keep the geometry too
def test_ref_to_full_and_back_roundtrip_and_land_on_the_footage_subject():
    foot = footage_with_disc(1, 640, 360, lambda j: 55)
    ref = matte_from_reference(foot, (384, 704))[0]
    full = MA.ref_to_full(ref, (640, 360))
    assert full.shape == (640, 360)
    want, got, old = centroid(foot[0][..., 0] / 255.0), centroid(full / 255.0), centroid(cv2.resize(ref, (360, 640)) / 255.0)
    assert abs(got[0] - want[0]) < 0.8 and abs(old[0] - want[0]) > 2.0                    # the plain resize (what the pivot node did) was off at the edge
    back = MA.frame_to_ref(full, ref.shape, (640, 360))
    assert back.shape == ref.shape and np.abs(back.astype(np.float32) - ref.astype(np.float32)).mean() < 6.0


def test_a_3d_cut_patch_written_in_clip_geometry_lands_on_the_clip_after_compose_maps_the_reference_back():
    n = 6
    foot = footage_with_disc(n, 640, 360, lambda j: 55)
    ref_m = matte_from_reference(foot, (384, 704))                                        # the shot's matte, in reference geometry
    patch_frames = np.stack([disc(640, 360, 70, 300, 40).astype(np.float32) / 255.0] * 3)      # the matte through the transition, made on the clip (frames 2..4)
    patched = PC.patch_matte(ref_m, (2, patch_frames), footage_hw=(640, 360))
    assert patched.shape == ref_m.shape and np.array_equal(patched[0], ref_m[0])            # frames outside the patch are untouched
    want = centroid(patch_frames[0])
    got = centroid(composed_matte(foot, patched, 3, (360, 640), aligned=True))
    assert abs(got[0] - want[0]) < 1.5 and abs(got[1] - want[1]) < 1.5, (got, want)
    plain = PC.patch_matte(ref_m, (2, patch_frames))                                       # without footage_hw: the old resize (kept for same-geometry callers)
    assert plain.shape == ref_m.shape


def test_window_cut_neighbour_matte_is_mapped_with_the_neighbours_own_footage():
    a = footage_with_disc(4, 640, 360, lambda j: 180)
    b = footage_with_disc(4, 640, 360, lambda j: 50)                                      # shot B's subject sits near the left edge
    mb = matte_from_reference(b, (384, 704))
    ma = matte_from_reference(a, (384, 704))
    rec = {"size": [360, 640], "inputs": {"footage": dict(src="footage", kind="rgb"), "s1_matte": dict(src="matte", kind="gray", native=True),
                                         "next": dict(src="footage", shot="s2", kind="rgb"), "next_matte": dict(src="matte", shot="s2", kind="gray", native=True)}}
    with RE.use_backend("numpy"):
        srcs = RE.open_sources(rec, footage=a, matte=ma, others={"s2": {"footage": b, "matte": mb}})
        assert isinstance(srcs["next_matte"], RE.RefSource) and isinstance(srcs["s1_matte"], RE.RefSource)
        m = C.upscale_matte(srcs["next_matte"][0], (360, 640))
        want = centroid(fitted_footage_mask(b, 0, (360, 640)))
        got = centroid(np.asarray(C._out(m), np.float32))
    assert abs(got[0] - want[0]) < 0.8                                                     # B's matte lands on B's subject (A's matte would be 130 px away)


def test_freeze_frames_hold_the_picture_and_its_matte_on_the_same_frame():
    rec = {"fps": 24, "holds": [[1.0, 0.5]], "holds_type": True, "inputs": {"footage": dict(src="footage"), "matte": dict(src="matte"), "subject": dict(src="subject"), "depth": dict(src="depth")}}
    for i in (10, 24, 30, 36, 50):
        idx = {nm: RE.hold_index(rec, nm, i, 0, 0) for nm in rec["inputs"]}
        assert len(set(idx.values())) == 1, (i, idx)


def test_a_crossfade_blends_finished_shots_each_composed_with_its_own_matte():
    """compose.assemble only blends composed frames: no matte is applied after the blend, so a shot's matte can never sit over the blended picture."""
    import inspect
    src = inspect.getsource(C.assemble)
    assert "matte" not in src.lower()


# ------------------------------------------------------------------------------------------------- the real doubled-subject cases of the 24 GB box batch (QC 2026-10-29)
BATCH = ROOT / "22_batch-24gb"


def _real(ver, run, sid):
    import glob
    import re
    d = BATCH / ver / "ingredients" / f"motion-graphic_batch-24gb_{ver}_{run}"
    tb, cl = d / f"motion-graphic_batch-24gb_{ver}_{run}_{sid}_textbg.mp4", d / f"motion-graphic_batch-24gb_{ver}_{run}_{sid}_clean.mp4"
    md = d / f"motion-graphic_batch-24gb_{ver}_{run}_{sid}_matte"
    if not (tb.is_file() and cl.is_file() and md.is_dir()):
        pytest.skip("the batch ingredients are not present")

    def load(p):
        cap, out = cv2.VideoCapture(str(p)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return np.stack(out)
    c, t = load(cl), load(tb)
    files = sorted(glob.glob(str(md / "*.png")), key=lambda s: int(re.findall(r"(\d+)\.png", s)[0]))
    m = np.stack([cv2.imread(f, 0) for f in files])
    hw = c.shape[1:3]
    ma = np.stack([cv2.resize(MA.ref_to_frame(x, hw, (hw[1], hw[0])), (hw[1], hw[0])) for x in m])
    return c, t, ma


@pytest.mark.parametrize("ver,run,sid,ok", [("v005", "r01", "s5", False), ("v005", "r01", "s3", False), ("v006", "r01", "s4", False), ("v005", "r01", "s6", True), ("v005", "r02", "s2", True)])
def test_the_text_pass_clips_that_showed_two_people_are_flagged_and_the_steady_ones_pass(ver, run, sid, ok):
    """v005 r01 s5: the skater is under 1 % of the frame and the text pass moved it by ~16 px of 360 (two skaters side by side); v006 r01 s4: 11 px (ghost skirt under the type)."""
    c, t, ma = _real(ver, run, sid)
    d = MA.subject_drift(c, t, ma)
    assert d["ok"] is ok, d
    if not ok:
        assert d["px"] > 5.0


def test_choose_background_trusts_the_text_clip_only_when_the_subject_was_verified(tmp_path):
    from test_ltx_read import _piece
    p = _piece(("HANDS", "CLAY"), 6.0)
    clean = textured_subject_clip(6, 192, 108)
    steady = dict(sid="s1", u8=textured_subject_clip(6, 192, 108), subject=clean, textbg=True)
    matte = np.stack([disc(192, 108, 54, 96, 21)] * 6)
    q = copy.deepcopy(p)
    out, notes = LC.choose_background(q, "s1", steady, None, tmp_path)                   # no matte: nothing to verify against
    assert out["u8"] is clean and any("could not be checked" in n for n in notes) and all(w["render"] == "flat" for w in q["type"]["words"])
    r = copy.deepcopy(p)
    out2, notes2 = LC.choose_background(r, "s1", steady, matte, tmp_path)               # verified: kept
    assert out2 is steady and notes2 == []
    z = copy.deepcopy(p)
    out3, _ = LC.choose_background(z, "s1", steady, np.zeros((6, 192, 108), np.uint8), tmp_path)      # a matte with no subject: nothing verified either
    assert out3["u8"] is clean
