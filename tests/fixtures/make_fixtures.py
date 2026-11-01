"""Generate the synthetic test fixtures (deterministic). Run from anywhere:

    python tests/fixtures/make_fixtures.py

Writes next to this file:
  blob_gradient.png   1024x768  large circle-ish blob on a vertical gradient, quiet top band
  stripes.png         768x1024  bold diagonal stripes (strong straight lines) with a plain lower band
  subject_texture.png 1024x768  photo-like noise texture with a bright subject
  beat_120.wav        6 s, 120 bpm click + bass, 48 kHz mono 16-bit
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _save_png(arr: np.ndarray, name: str) -> str:
    path = os.path.join(HERE, name)
    img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    img.save(path, optimize=True)
    return path


def blob_gradient(w: int = 1024, h: int = 768) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (yy / (h - 1))[..., None]
    top = np.array([0.06, 0.10, 0.22], np.float32)
    bot = np.array([0.10, 0.45, 0.50], np.float32)
    img = top * (1 - t) + bot * t
    cx, cy, r0 = 0.62 * w, 0.58 * h, 0.27 * h
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(xx - cx, yy - cy)
    r = r0 * (1 + 0.12 * np.sin(3 * ang + 0.4) + 0.06 * np.sin(5 * ang + 1.3))
    edge = np.clip((r - rad) / 2.0 + 0.5, 0, 1)[..., None]       # 2 px anti-aliased edge
    blob = np.array([1.0, 0.45, 0.12], np.float32) * (0.85 + 0.15 * (1 - rad / (r0 * 1.2)))[..., None]
    img = img * (1 - edge) + np.clip(blob, 0, 1) * edge
    # a small secondary blob for a second contour
    rad2 = np.hypot(xx - 0.2 * w, yy - 0.72 * h)
    e2 = np.clip((0.09 * h - rad2) / 2.0 + 0.5, 0, 1)[..., None]
    img = img * (1 - e2) + np.array([1.0, 0.85, 0.2], np.float32) * e2
    return img


def stripes(w: int = 768, h: int = 1024) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cols = np.array([[0.95, 0.93, 0.88], [0.10, 0.12, 0.16], [0.18, 0.42, 1.0]], np.float32)
    u = (xx + 0.6 * yy) / 90.0
    idx = np.floor(u).astype(np.int64) % 3
    img = cols[idx]
    # plain band at the bottom (empty region for text)
    band = (yy >= 0.74 * h)[..., None]
    img = np.where(band, np.array([0.08, 0.09, 0.12], np.float32), img)
    return img


def subject_texture(w: int = 1024, h: int = 768, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # multi-octave value noise (upsampled low-res noise), muted earthy colours
    acc = np.zeros((h, w, 3), np.float32)
    for k, amp in ((8, 0.5), (24, 0.3), (64, 0.2)):
        small = rng.random((max(2, h * k // w), k, 3)).astype(np.float32)
        up = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
        acc += amp * up
    base = np.array([0.22, 0.20, 0.18], np.float32) + 0.25 * (acc - 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # bright subject: soft-edged rounded star at left-centre
    cx, cy = 0.36 * w, 0.48 * h
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(xx - cx, yy - cy)
    r = 0.24 * h * (1 + 0.22 * np.cos(5 * ang))
    m = np.clip((r - rad) / 3.0 + 0.5, 0, 1)[..., None]
    subj = np.array([0.98, 0.92, 0.70], np.float32) - 0.15 * (rad / (0.3 * h))[..., None]
    img = base * (1 - m) + np.clip(subj, 0, 1) * m
    img = np.clip(img, 0, 1)
    # quantise lightly so the PNG stays small while still looking photographic
    img = np.round(img * 63) / 63
    pil = Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(pil, np.float32) / 255.0


def beat_wav(seconds: float = 6.0, bpm: float = 120.0, sr: int = 48000) -> np.ndarray:
    n = int(round(seconds * sr))
    out = np.zeros(n, np.float32)
    beat = 60.0 / bpm
    rng = np.random.default_rng(1)
    t_click = np.arange(int(0.03 * sr)) / sr
    t_bass = np.arange(int(0.4 * sr)) / sr
    for i in range(int(seconds / beat)):
        s = int(round(i * beat * sr))
        down = i % 4 == 0
        click = (np.sin(2 * np.pi * (2000 if down else 1500) * t_click) + 0.3 * rng.standard_normal(t_click.size))
        click *= np.exp(-t_click * 180) * (0.6 if down else 0.4)
        e = min(n, s + click.size)
        out[s:e] += click[: e - s]
        f = 55.0 if (i // 4) % 2 == 0 else 41.2
        bass = np.sin(2 * np.pi * f * t_bass) * np.exp(-t_bass * 6) * (0.55 if down else 0.4)
        bass *= np.clip(t_bass / 0.005, 0, 1)
        e = min(n, s + bass.size)
        out[s:e] += bass[: e - s]
    out /= max(1e-6, np.abs(out).max()) / 0.8
    return out


def main() -> None:
    print(_save_png(blob_gradient(), "blob_gradient.png"))
    print(_save_png(stripes(), "stripes.png"))
    print(_save_png(subject_texture(), "subject_texture.png"))
    path = os.path.join(HERE, "beat_120.wav")
    wavfile.write(path, 48000, (beat_wav() * 32767).astype(np.int16))
    print(path)


if __name__ == "__main__":
    main()
