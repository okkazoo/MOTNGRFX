"""CPU tests for tools/fusion/compose.py on synthetic data. Run: python -m pytest tools/fusion/test_compose.py -q"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import compose as C  # noqa: E402

H, W = 128, 72


@pytest.fixture(autouse=True)
def _numpy_backend():
    C.set_backend("numpy")
    yield
    C.set_backend("numpy")


def grad_footage(shift=0.0):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([xs / W, ys / H, np.full_like(xs, 0.3 + shift)], -1).astype(np.float32)


def disc(cx, cy, r, soft=1.5):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip((r - np.hypot(xs - cx, ys - cy)) / soft + 0.5, 0, 1).astype(np.float32)


RED = (1.0, 0.0, 0.0)


def test_sandwich_semitransparent_matte_keeps_type_at_half():
    f = grad_footage()
    ta = np.ones((H, W), np.float32)
    m = np.full((H, W), 0.5, np.float32)
    out = C.sandwich(f, RED, ta, m)
    expect = 0.5 * np.array(RED) + 0.5 * f  # type fully opaque, footage covers it by 50%
    assert np.allclose(out, expect, atol=1e-5)
    # visible type share is exactly the uncovered 50%
    only_footage = C.sandwich(f, RED, np.zeros((H, W), np.float32), m)
    assert np.allclose(only_footage, f, atol=1e-6)


def test_sandwich_no_threshold_and_gamma_gain_invert():
    f = grad_footage()
    ta = np.ones((H, W), np.float32)
    ramp = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    out = C.sandwich(f, RED, ta, ramp)
    # type red channel weight is (1-matte): strictly graded, no steps
    assert np.allclose(out[..., 0], (1 - ramp) * 1.0 + ramp * f[..., 0], atol=1e-5)
    black = C.sandwich(np.zeros_like(f), RED, ta, ramp)  # over black the red channel is exactly 1 - matte
    assert np.allclose(black[..., 0], 1 - ramp, atol=1e-5)
    assert len(np.unique(np.round(black[0, :, 0], 4))) > W * 0.95  # graded, no steps
    g = C.sandwich(f, RED, ta, ramp, matte_gamma=2.0)
    assert np.allclose(g[..., 0], (1 - ramp ** 2) + ramp ** 2 * f[..., 0], atol=1e-5)
    inv = C.sandwich(f, RED, ta, ramp, invert=True)
    assert np.allclose(inv[..., 0], ramp + (1 - ramp) * f[..., 0], atol=1e-5)
    fg = np.zeros_like(f)
    o = C.sandwich(f, RED, ta, np.ones((H, W), np.float32), fg=fg)
    assert np.allclose(o, 0, atol=1e-6)


def test_depth_occlude_hides_type_where_footage_nearer():
    f = grad_footage()
    depth = np.full((H, W), 0.2, np.float32)
    depth[:, W // 2:] = 0.9  # right half is near
    ta = np.ones((H, W), np.float32)
    out = C.depth_occlude(f, depth, RED, ta, plane=0.5, softness=0.1)
    assert np.allclose(out[:, : W // 2], np.array(RED), atol=1e-5)  # far: type shows
    assert np.allclose(out[:, W // 2:], f[:, W // 2:], atol=1e-5)  # near: footage covers type
    # soft ramp: depth exactly on the plane -> 50%
    d2 = np.full((H, W), 0.5, np.float32)
    o2 = C.depth_occlude(f, d2, RED, ta, plane=0.5, softness=0.2)
    assert np.allclose(o2, 0.5 * np.array(RED) + 0.5 * f, atol=1e-5)
    # plane as a depth map
    o3 = C.depth_occlude(f, depth, RED, ta, plane=np.full((H, W), 0.95, np.float32), softness=0.05)
    assert np.allclose(o3, np.array(RED), atol=1e-5)


def test_knockout_footage_only_inside_letters():
    f = grad_footage()
    plate = (0.1, 0.1, 0.1)
    ta = disc(W / 2, H / 2, 20, soft=0.01)
    out = C.knockout(plate, ta, f)
    assert np.allclose(out[ta == 0], 0.1, atol=1e-6)
    assert np.allclose(out[ta == 1], f[ta == 1], atol=1e-6)
    m = disc(10, 10, 8, soft=0.01)  # subject steps out over the plate (outside the letters)
    out2 = C.knockout(plate, ta, f, fg_matte=m)
    assert np.allclose(out2[m == 1], f[m == 1], atol=1e-6)
    assert np.allclose(out2[(ta == 0) & (m == 0)], 0.1, atol=1e-6)


def test_refract_unchanged_where_flat_and_displaces_edges():
    f = grad_footage()
    flat = np.full((H, W), 0.7, np.float32)
    assert np.allclose(C.refract(f, flat, strength=20), f, atol=1e-6)
    ta = disc(W / 2, H / 2, 18, soft=2)
    out = C.refract(f, ta, strength=10, blur=3, rim=0.0)
    far = np.hypot(*np.mgrid[0:H, 0:W] - np.array([H / 2, W / 2])[:, None, None]) > 40
    assert np.allclose(out[far], f[far], atol=1e-4)  # untouched away from the letter
    assert np.abs(out - f).max() > 0.02  # displaced at the edge
    rim = C.refract(f, ta, strength=0, rim=1.0)
    assert (rim - f).max() > 0.1  # specular rim only


def test_refract_mask_limits_effect():
    f = grad_footage()
    ta = disc(W / 2, H / 2, 18, soft=2)
    zero = np.zeros((H, W), np.float32)
    assert np.allclose(C.refract(f, ta, strength=10, mask=zero), f, atol=1e-6)


def test_tint_modes_and_palette():
    f = np.zeros((H, W, 3), np.float32)
    f[:, : W // 2] = (0.9, 0.1, 0.1)
    f[:, W // 2:] = (0.1, 0.2, 0.9)
    pal = C.palette_from_footage([f], k=2)
    assert pal.shape == (2, 3)
    assert np.allclose(sorted(pal[:, 1]), sorted([0.1, 0.2]), atol=0.02)  # the two flat colours recovered
    assert C.luma(pal[None])[0, 0] <= C.luma(pal[None])[0, 1]  # dark -> light
    ta = np.ones((H, W), np.float32)
    s = C.tint_type(ta, f, "sample", sample_blur=1)
    assert s.shape == (H, W, 3) and 0 <= s.min() and s.max() <= 1
    assert s[H // 2, 4, 0] > s[H // 2, 4, 2]  # left side stays red-ish
    comp = C.tint_type(ta, f, "complement", sample_blur=1)
    assert comp[H // 2, 4, 2] + comp[H // 2, 4, 1] > comp[H // 2, 4, 0]  # complement of red is cyan-ish
    gm = C.tint_type(ta, f, "gradient_map", palette=np.array([[0, 0, 0.2], [1, 0.8, 0.3]], np.float32), sample_blur=1)
    assert gm.shape == (H, W, 3)
    with pytest.raises(ValueError):
        C.tint_type(ta, f, "nope")


def test_edges_and_traced_type():
    f = np.zeros((H, W, 3), np.float32)
    f[:, W // 2:] = 1.0  # one vertical step edge
    e = C.trace_edges(f)
    assert e.shape == (H, W)
    assert e[H // 2, W // 2 - 2: W // 2 + 2].max() > 0.9 and e[H // 2, 5] < 0.01
    # temporal: a sequence returns a same-length generator / array, and a flat clip stays empty
    seq = C.trace_edges(np.stack([f, f, f]))
    assert seq.shape == (3, H, W)
    gen = list(C.trace_edges(iter([f, f])))
    assert len(gen) == 2 and np.allclose(gen[0], gen[1], atol=1e-5)
    ta = np.zeros((H, W), np.float32)
    ta[:, : W // 2 + 10] = 1.0
    inside = C.edges_in_type(e, ta)
    assert inside.max() > 0.9
    ta2 = np.zeros((H, W), np.float32)
    ta2[:, :10] = 1.0
    assert C.edges_in_type(e, ta2).max() < 0.01  # edge outside the letters is hidden
    tfe = C.type_from_edges(e, ta2, grow=0, line_width=0)
    assert tfe.max() < 0.01
    tfe2 = C.type_from_edges(e, ta2, grow=W // 2, line_width=2, softness=1)
    assert tfe2.max() > 0.5  # dilated type reaches the contour


def test_blend_helpers_basic_identities():
    a = grad_footage()
    black, white = np.zeros_like(a), np.ones_like(a)
    assert np.allclose(C.screen(a, black), a, atol=1e-6)
    assert np.allclose(C.multiply(a, white), a, atol=1e-6)
    assert np.allclose(C.multiply(a, black), 0)
    grey = np.full_like(a, 0.5)
    assert np.allclose(C.overlay(a, grey), a, atol=1e-5)
    assert np.allclose(C.soft_light(a, grey), a, atol=1e-5)
    assert C.soft_light(a, white).mean() > a.mean()
    k = C.luminosity_key(a, 0.2, 0.8)
    assert k.shape == (H, W) and 0 <= k.min() and k.max() <= 1
    m = disc(W / 2, H / 2, 6)
    g = C.add_glow(np.zeros_like(a), m, radius=6, intensity=1.0)
    assert g[H // 2, W // 2, 0] > 0.3 and g[0, 0, 0] < g[H // 2, W // 2, 0]


def test_finishing():
    f = grad_footage()
    assert np.allclose(C.grade(f), f, atol=1e-5)
    assert C.grade(f, gamma=2.0).mean() > f.mean()
    assert C.grade(f, lift=0.2).min() >= 0.2 - 1e-6
    g0 = C.grade(f, sat=0.0)
    assert np.allclose(g0[..., 0], g0[..., 1], atol=1e-5)
    a, b, a2 = C.grain(f, 3, seed=1), C.grain(f, 4, seed=1), C.grain(f, 3, seed=1)
    assert np.array_equal(a, a2) and not np.allclose(a, b)  # deterministic, varies in time
    assert abs(float((a - f).std()) - 0.03 * 0.4) < 0.02
    v = C.vignette(f, strength=0.5)
    assert v[0, 0, 0] <= f[0, 0, 0] and np.allclose(v[H // 2, W // 2], f[H // 2, W // 2], atol=1e-3)
    h = C.halation(np.where(disc(W / 2, H / 2, 6)[..., None] > 0.5, 1.0, 0.0).astype(np.float32).repeat(3, -1), 0.5, 8, 0.5)
    assert h[H // 2 + 12, W // 2, 0] > h[H // 2 + 12, W // 2, 2]  # warm tint


def test_utilities():
    img = np.random.default_rng(0).random((100, 200, 3)).astype(np.float32)
    c = C.fit_cover(img, (54, 96))
    assert c.shape == (96, 54, 3)
    c2 = C.fit_cover(img[:90, :50], (100, 180))
    assert c2.shape == (180, 100, 3)
    a, b = np.zeros((4, 4, 3), np.float32), np.ones((4, 4, 3), np.float32)
    assert np.allclose(C.crossfade(a, b, 0.25), 0.25)
    seq = [np.full((2, 2, 3), v, np.float32) for v in (0.0, 1.0, 2.0)]
    vals = lambda mode, n: [float(x[0, 0, 0]) for x in C.match_length(seq, n, mode)]
    assert vals("hold", 5) == [0, 1, 2, 2, 2]
    assert vals("loop", 5) == [0, 1, 2, 0, 1]
    assert vals("pingpong", 6) == [0, 1, 2, 1, 0, 1]
    assert np.allclose(vals("retime", 5), [0, 0.5, 1, 1.5, 2], atol=1e-6)
    assert vals("hold", 2) == [0, 1]


def test_upscale_matte_guided_is_sharper_than_cubic():
    ys, xs = np.mgrid[0:256, 0:144].astype(np.float32)
    truth = (np.hypot(xs - 72, ys - 128) < 40).astype(np.float32)
    guide = np.stack([truth * 0.8 + 0.1] * 3, -1)
    small = cv2.resize(truth, (36, 64), interpolation=cv2.INTER_AREA)
    cub = C.upscale_matte(small, (144, 256))
    gui = C.upscale_matte(small, (144, 256), guide=guide)
    assert cub.shape == gui.shape == (256, 144)
    width = lambda m: ((m > 0.1) & (m < 0.9)).sum()
    assert width(gui) < 0.6 * width(cub)  # transition band much narrower, edge snapped to the guide
    assert ((gui > 0.5) != (truth > 0.5)).sum() < 0.2 * ((cub > 0.5) != (truth > 0.5)).sum() + 1  # 0.5 contour on the guide edge
    assert 0 <= gui.min() and gui.max() <= 1
    # soft values are kept (no thresholding) where the guide is flat
    flat_soft = C.upscale_matte(np.full((64, 36), 0.4, np.float32), (144, 256), guide=np.full((256, 144, 3), 0.5, np.float32))
    assert np.allclose(flat_soft, 0.4, atol=1e-3)


def test_numpy_and_torch_cpu_backends_agree():
    torch = pytest.importorskip("torch")
    f = grad_footage()
    ta = disc(W / 2, H / 2, 20, soft=2)
    m = disc(W / 3, H / 2, 30, soft=12) * 0.8
    depth = disc(W / 2, H / 2, 25, soft=3)

    def run():
        return {
            "sandwich": C.sandwich(f, RED, ta, m, matte_gamma=0.8),
            "depth": C.depth_occlude(f, depth, RED, ta, 0.5, 0.2),
            "knock": C.knockout((0.1, 0.1, 0.1), ta, f, m),
            "refract": C.refract(f, ta, strength=8, blur=3),
            "tint": C.tint_type(ta, f, "sample", sample_blur=3),
            "edges": C.trace_edges(f),
            "grade": C.grade(f, 0.05, 1.2, 1.1, 1.3),
            "vig": C.vignette(f),
            "grain": C.grain(f, 5, seed=2),
            "up": C.upscale_matte(m[::2, ::2], (W, H), guide=f),
            "fit": C.fit_cover(f, (50, 90)),
            "glow": C.add_glow(f, ta, 4),
            "tfe": C.type_from_edges(C.trace_edges(f), ta),
        }

    C.set_backend("numpy")
    a = run()
    assert C.backend_name().startswith("numpy")
    C.set_backend("torch")
    assert C.backend_name().startswith("torch")
    b = run()
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype == np.float32, k
        assert np.abs(a[k] - b[k]).max() < (2e-2 if k == 'glow' else 5e-3), (k, np.abs(a[k] - b[k]).max())


# ------------------------------------------------------------------------------------------------ IO


def ffprobe_duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          str(p)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ffprobe_streams(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate",
                          "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def make_audio(path, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-ar", "44100", str(path)], check=True)


def synth_frames(n, w=64, h=96):
    for i in range(n):
        f = np.zeros((h, w, 3), np.float32)
        f[..., 0] = i / max(n - 1, 1)
        f[..., 1] = np.linspace(0, 1, w)[None, :]
        yield f


def test_png_sequence_roundtrip_and_lazy_reading(tmp_path):
    frames = list(synth_frames(5))
    n = C.write_png_sequence(iter(frames), tmp_path / "png")
    assert n == 5
    src = C.FrameSource(tmp_path / "png")
    assert len(src) == 5
    assert np.abs(src[3] - frames[3]).max() < 1 / 255 + 1e-6
    assert np.abs(src[0] - frames[0]).max() < 1 / 255 + 1e-6
    g = C.FrameSource(tmp_path / "png", "gray")[2]
    assert g.ndim == 2
    out = list(C.read_frames(tmp_path / "png", size=(32, 48), start=1, stop=4))
    assert len(out) == 3 and out[0].shape == (48, 32, 3)
    # 16 bit
    C.imwrite(tmp_path / "w16.png", frames[2], bit16=True)
    assert np.abs(C.imread(tmp_path / "w16.png") - frames[2]).max() < 1e-4


def test_mp4_encode_with_audio_has_right_duration(tmp_path):
    make_audio(tmp_path / "a.wav", 2.0)
    r = C.encode_mp4(synth_frames(30), tmp_path / "o.mp4", fps=24, audio=tmp_path / "a.wav")
    d = ffprobe_duration(tmp_path / "o.mp4")
    assert abs(d - 2.0) < 0.06, d  # 30 frames (1.25 s) held to the 2.0 s audio
    assert r["frames"] == 48
    streams = {s["codec_name"]: s for s in ffprobe_streams(tmp_path / "o.mp4")}
    assert streams["h264"]["pix_fmt"] == "yuv420p" and streams["h264"]["r_frame_rate"] == "24/1"
    assert "aac" in streams
    # longer video trimmed to the audio
    C.encode_mp4(synth_frames(120), tmp_path / "o2.mp4", fps=24, audio=tmp_path / "a.wav")
    assert abs(ffprobe_duration(tmp_path / "o2.mp4") - 2.0) < 0.06
    # odd sizes are cropped to even; no audio -> frames/fps
    C.encode_mp4(synth_frames(24, w=63, h=95), tmp_path / "o3.mp4", fps=24)
    st = ffprobe_streams(tmp_path / "o3.mp4")[0]
    assert (st["width"], st["height"]) == (62, 94)
    assert abs(ffprobe_duration(tmp_path / "o3.mp4") - 1.0) < 0.06


def test_mp4_read_back_is_lazy_sequential_and_random_access(tmp_path):
    C.encode_mp4(synth_frames(24), tmp_path / "v.mp4", fps=24, crf=10)
    src = C.FrameSource(tmp_path / "v.mp4")
    assert len(src) == 24
    ref = list(synth_frames(24))
    for i in (0, 1, 2, 3):
        assert np.abs(src[i] - ref[i]).mean() < 0.03
    assert np.abs(src[20] - ref[20]).mean() < 0.03  # random access (re-seek)
    assert np.abs(src[10] - ref[10]).mean() < 0.03  # backwards
    assert len(list(C.read_frames(tmp_path / "v.mp4"))) == 24
    src.close()


def test_compose_shot_recipe_and_assemble(tmp_path):
    f = list(synth_frames(6))
    C.write_png_sequence(iter(f), tmp_path / "in" / "footage")
    ys, xs = np.mgrid[0:96, 0:64].astype(np.float32)
    ta = [np.clip((20 - np.hypot(xs - 32, ys - 48)) / 1.0 + 0.5, 0, 1).astype(np.float32)] * 6
    C.write_png_sequence(iter(ta), tmp_path / "in" / "type_a")
    C.write_png_sequence(iter([np.full((96, 64), 0.5, np.float32)] * 2), tmp_path / "in" / "matte")  # shorter: held
    make_audio(tmp_path / "a.wav", 0.5)
    recipe = {
        "name": "t", "size": [64, 96], "fps": 24, "frames": 12, "length_mode": "loop", "palette": {"k": 3},
        "inputs": {"footage": "in/footage", "type_a": "in/type_a", "matte": {"path": "in/matte", "mode": "hold"},
                   "type_rgb": [1, 0, 0]},
        "ops": [
            {"op": "tint_type", "out": "tint", "args": {"type_a": "$type_a", "footage": "$footage",
                                                         "mode": "gradient_map", "palette": "$palette"}},
            {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "type_rgb": "$tint", "type_a": "$type_a",
                                                         "matte": "$matte", "matte_gamma": {"anim": [[0, 1], [1, 2]]}}},
            {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "amount": 0.01}},
        ],
    }
    r = C.compose_shot(recipe, tmp_path / "shot", base_dir=tmp_path)
    assert r["n"] == 12 and len(list((tmp_path / "shot" / "frames").glob("*.png"))) == 12
    assert abs(ffprobe_duration(r["mp4"]) - 0.5) < 0.06
    # inside the disc the type is visible at ~50% (matte 0.5 at gamma 1.0 -> 0.5; gamma 2 -> 0.25 later)
    im0 = C.imread(tmp_path / "shot" / "frames" / "00000.png")
    assert im0[48, 32].sum() != pytest.approx(f[0][48, 32].sum(), abs=0.02)
    with pytest.raises(ValueError):
        C.compose_shot({"size": [64, 96], "inputs": {"footage": "in/footage"},
                        "ops": [{"op": "nope", "args": {}}]}, tmp_path / "bad", base_dir=tmp_path)
    # assemble two shots with a crossfade, audio-length output
    r2 = C.assemble([{"src": r["mp4"], "cut": 0}, {"src": str(tmp_path / "in" / "footage"), "cut": 0.25}],
                    tmp_path / "a.wav", tmp_path / "final.mp4", crossfades=[0.1], base_dir=tmp_path)
    assert abs(ffprobe_duration(tmp_path / "final.mp4") - 0.5) < 0.06 and r2["frames"] == 12


def test_cli_info():
    out = subprocess.run([sys.executable, str(Path(C.__file__)), "info"], capture_output=True, text=True,
                         env={**os.environ, "FUSION_BACKEND": "numpy"})
    assert out.returncode == 0 and out.stdout.strip().startswith("numpy")
