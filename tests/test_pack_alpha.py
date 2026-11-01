"""Alpha-Gen matte packing (tools/fusion/pack_alpha.py): synthetic sequence -> FFV1 mkv + ProRes 4444 mov, bit-exact round
trip, load_matte parity (mkv vs PNGs vs auto-resolved ingredients), depth_orbit.load_video parity. Needs ffmpeg on PATH.
Run: python -m pytest tests/test_pack_alpha.py -q"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402
from tools.fusion import pack_alpha as pa  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg missing")

SHOT, N, MW, MH = "s9z", 6, 48, 80          # matte 48x80, clip 96x160
PREFIX = "motion-graphic_fusion_v003_P06"


def _make(tmp_path):
    vdir = tmp_path / "15_fusion" / "v003"
    seq = vdir / "local" / "pieces" / "P06" / "seq"
    seq.mkdir(parents=True)
    rng = np.random.default_rng(3)
    files = []
    for i in range(N):
        g = (np.linspace(0, 255, MW)[None, :] * np.ones((MH, 1)) * (0.4 + 0.1 * i)).clip(0, 255)
        g = g + rng.integers(0, 40, (MH, MW))
        im = np.stack([g, g + rng.integers(0, 9, (MH, MW)), g - rng.integers(0, 9, (MH, MW))], -1).clip(0, 255).astype(np.uint8)
        f = str(seq / ("%s_alpha_%s_1_image_%05d_.png" % (PREFIX, SHOT, i + 1)))
        assert cv2.imwrite(f, im)
        files.append(f)
    out = vdir / "out"
    out.mkdir()
    clip = str(out / ("%s_%s_1_00001_.mp4" % (PREFIX, SHOT)))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=96x160:rate=24:duration=0.25", "-f", "lavfi", "-i", "sine=d=0.25", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-frames:v", str(N), clip], check=True)
    return vdir, files, clip


@pytest.fixture()
def packed(tmp_path):
    vdir, files, clip = _make(tmp_path)
    ing = str(vdir / "ingredients" / "P06")
    rep = pa.pack_shot(files, clip, pa.stem_of(SHOT), ing, log=lambda *a: None)
    return vdir, files, clip, ing, rep


def test_matte_mkv_is_bit_exact_and_rgba_has_alpha(packed):
    vdir, files, clip, ing, rep = packed
    mkv = os.path.join(ing, pa.stem_of(SHOT) + "_matte.mkv")
    mov = os.path.join(ing, pa.stem_of(SHOT) + "_rgba.mov")
    dec = pa._decode_bgr(mkv)
    assert dec.shape == (N, MH, MW, 3)
    for i, f in enumerate(files):
        assert np.array_equal(dec[i], cv2.imread(f, cv2.IMREAD_COLOR))          # all three channels, every pixel
    p = pa._probe(mov)
    assert p["pix_fmt"].startswith("yuva") and p["frames"] == N and (p["w"], p["h"]) == (96, 160) and p["fps"] == 24
    assert p["audio"] == ["aac"]
    assert rep["rgba"]["alpha_corr"] > 0.999 and rep["matte_exact_frames"] == N
    assert not [f for f in os.listdir(ing) if f.endswith(".part")]


def test_load_matte_matches_imread_grayscale_from_mkv_and_pngs(packed):
    vdir, files, clip, ing, rep = packed
    ref = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    mkv = os.path.join(ing, pa.stem_of(SHOT) + "_matte.mkv")
    a = pa.load_matte(mkv)
    assert a.dtype == np.uint8 and a.shape == ref.shape and np.array_equal(a, ref)
    assert np.array_equal(pa.load_matte(os.path.join(os.path.dirname(files[0]), "*.png")), ref)
    assert np.array_equal(pa.load_matte(os.path.dirname(files[0])), ref)
    assert np.array_equal(pa.load_matte(files[0].replace("00001_", "%05d_")), ref)
    assert np.array_equal(pa.load_matte(mkv, frames=[2, 4]), ref[[2, 4]])
    # first-frame PNG path resolves to the packed file (and equals the PNGs); NO_PACKED forces the PNGs
    assert pa.packed_for(files[0]) == mkv.replace("\\", "/") or os.path.samefile(pa.packed_for(files[0]), mkv)
    assert np.array_equal(pa.load_matte(files[0]), ref)
    assert np.array_equal(pa.load_matte(files[0], frames=[0])[0], ref[0])


def test_no_packed_env_forces_pngs(packed, monkeypatch):
    vdir, files, clip, ing, rep = packed
    ref = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    os.remove(files[3])                                                   # PNG sequence now has a hole; mkv still complete
    assert len(pa.load_matte(files[0])) == N                              # packed wins
    monkeypatch.setenv("MOTNGRFX_NO_PACKED", "1")
    assert len(pa.load_matte(files[0])) == 3                              # PNG path: consecutive until the first missing
    assert np.array_equal(pa.load_matte(files[0]), ref[:3])


def test_depth_orbit_load_video_identical_either_way(packed, monkeypatch):
    vdir, files, clip, ing, rep = packed
    monkeypatch.setattr(dd, "_paths", lambda shot: (clip, files[0], None))
    W, H = 64, 112
    monkeypatch.setenv("MOTNGRFX_NO_PACKED", "1")
    (f0, m0), = do.load_video(W, H, shots=(SHOT,))
    monkeypatch.delenv("MOTNGRFX_NO_PACKED")
    assert pa.packed_for(files[0])
    (f1, m1), = do.load_video(W, H, shots=(SHOT,))
    assert m0.dtype == np.float32 and m0.shape == (N, H, W)
    assert np.array_equal(f0, f1) and np.array_equal(m0, m1)


def test_depth_dolly_first_matte_identical(packed, monkeypatch):
    vdir, files, clip, ing, rep = packed
    frame = cv2.imread(files[0])
    frame = cv2.resize(frame, (96, 160))
    dn = np.full((160, 96), 0.5, np.float32)
    monkeypatch.setenv("MOTNGRFX_NO_PACKED", "1")
    _, _, m0 = dd.load_layer_inputs(frame, files[0], short=96, dn_full=dn)
    monkeypatch.delenv("MOTNGRFX_NO_PACKED")
    _, _, m1 = dd.load_layer_inputs(frame, files[0], short=96, dn_full=dn)
    assert np.array_equal(m0, m1)


def test_existing_outputs_are_never_overwritten_and_corruption_fails_loudly(packed):
    vdir, files, clip, ing, rep = packed
    with pytest.raises(FileExistsError):
        pa.pack_shot(files, clip, pa.stem_of(SHOT), ing, log=lambda *a: None)
    mkv = os.path.join(ing, pa.stem_of(SHOT) + "_matte.mkv")
    bad = cv2.imread(files[2])
    bad[10, 10, 1] ^= 1                                                   # one channel of one pixel
    cv2.imwrite(files[2], bad)
    with pytest.raises(AssertionError, match="NOT bit-exact"):
        pa.verify_matte(files, mkv)
    with pytest.raises(AssertionError, match="frame count"):
        pa.verify_matte(files[:5], mkv)
