"""Frames + audio -> MP4.

Primary backend: PyAV (``import av``, the library ComfyUI itself uses for video).
Fallback (only when ``import av`` fails): the ``ffmpeg`` binary on PATH via a
rawvideo stdin pipe. Both stream frames one at a time and never hold the clip.

``BACKEND`` can be set to "pyav" or "ffmpeg" to force a backend (tests use this);
``None`` means auto. The env var ``MOGRAPH_ENCODER`` does the same.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from typing import Iterable

import numpy as np

BACKEND: str | None = None

VIDEO_CODECS = ("libx264", "h264", "libopenh264", "mpeg4")


# ----------------------------------------------------------------------------- helpers

def _audio_2d(audio) -> np.ndarray:
    """-> (channels, samples) float32 clipped to -1..1."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio must be (channels, samples) or (samples,), got shape {a.shape}")
    if a.shape[0] > 8 and a.shape[1] <= 8:      # accept (samples, channels)
        a = a.T
    if a.shape[0] > 2:                            # downmix to stereo
        a = np.stack([a[0::2].mean(0), a[1::2].mean(0)])
    return np.clip(np.nan_to_num(a), -1.0, 1.0).astype(np.float32)


def _check_frame(fr, n: int, width: int, height: int) -> np.ndarray:
    fr = np.asarray(fr)
    if fr.shape != (height, width, 3):
        raise ValueError(f"frame {n} has shape {fr.shape}, expected {(height, width, 3)}")
    if fr.dtype != np.uint8:
        fr = np.clip(fr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(fr)


def _fps_fraction(fps: float) -> Fraction:
    return Fraction(float(fps)).limit_denominator(1001)


def _choose_backend() -> str:
    forced = BACKEND or os.environ.get("MOGRAPH_ENCODER") or None
    if forced:
        if forced not in ("pyav", "ffmpeg"):
            raise ValueError(f"unknown encoder backend {forced!r}")
        return forced
    try:
        import av  # noqa: F401
        return "pyav"
    except Exception:
        return "ffmpeg"


# ----------------------------------------------------------------------------- PyAV

def _pick_video_codec(av) -> str:
    for name in VIDEO_CODECS:
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError("PyAV build has no usable H.264/MPEG-4 encoder")


def _encode_pyav(frames, fps, width, height, audio, sample_rate, out_path, crf) -> str:
    import av

    rate = _fps_fraction(fps)
    pad_w, pad_h = width + (width % 2), height + (height % 2)
    codec = _pick_video_codec(av)

    container = av.open(out_path, mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        vs = container.add_stream(codec, rate=rate)
        vs.width, vs.height = pad_w, pad_h
        vs.pix_fmt = "yuv420p"
        if codec == "libx264":
            vs.options = {"crf": str(int(crf)), "preset": "fast"}
        else:
            vs.bit_rate = int(pad_w * pad_h * float(fps) * 0.15)
        vs.codec_context.time_base = Fraction(1) / rate

        a = None
        aus = None
        if audio is not None and np.asarray(audio).size > 0:
            a = _audio_2d(audio)
            layout = "stereo" if a.shape[0] == 2 else "mono"
            aus = container.add_stream("aac", rate=int(sample_rate))
            aus.bit_rate = 192000
            aus.layout = layout
            aus.codec_context.time_base = Fraction(1, int(sample_rate))
        apos = 0

        def push_audio(upto: int):
            nonlocal apos
            if upto <= apos:
                return
            chunk = np.zeros((a.shape[0], upto - apos), np.float32)
            have = a[:, apos:min(upto, a.shape[1])]
            chunk[:, :have.shape[1]] = have
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp",
                                            layout="stereo" if a.shape[0] == 2 else "mono")
            af.sample_rate = int(sample_rate)
            af.pts = apos
            af.time_base = Fraction(1, int(sample_rate))
            for pkt in aus.encode(af):
                container.mux(pkt)
            apos = upto

        n = 0
        for fr in frames:
            fr = _check_frame(fr, n, width, height)
            if (pad_w, pad_h) != (width, height):
                fr = np.pad(fr, ((0, pad_h - height), (0, pad_w - width), (0, 0)), mode="edge")
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf.pts = n
            vf.time_base = Fraction(1) / rate
            for pkt in vs.encode(vf):
                container.mux(pkt)
            n += 1
            if aus is not None:
                push_audio(int(round(n * int(sample_rate) / float(fps))))
        if n == 0:
            raise ValueError("encode_mp4 received no frames")
        for pkt in vs.encode(None):
            container.mux(pkt)
        if aus is not None:
            for pkt in aus.encode(None):
                container.mux(pkt)
    finally:
        container.close()
    return out_path


# ----------------------------------------------------------------------------- ffmpeg binary

def _encode_ffmpeg(frames, fps, width, height, audio, sample_rate, out_path, crf) -> str:
    from scipy.io import wavfile

    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("neither PyAV nor an ffmpeg binary on PATH is available")
    wav_path = None
    if audio is not None and np.asarray(audio).size > 0:
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mograph_")
        os.close(fd)
        pcm = (_audio_2d(audio).T * 32767.0).round().astype(np.int16)
        wavfile.write(wav_path, int(sample_rate), np.ascontiguousarray(pcm))

    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(_fps_fraction(fps)), "-i", "-"]
    if wav_path:
        cmd += ["-i", wav_path]
    if width % 2 or height % 2:
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", "fast", "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p"]
    if wav_path:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest"]
    cmd += ["-movflags", "+faststart", out_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    n = 0
    try:
        for fr in frames:
            proc.stdin.write(_check_frame(fr, n, width, height).tobytes())
            n += 1
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        rc = proc.wait()
    except BrokenPipeError:
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
        raise RuntimeError(f"ffmpeg exited early after {n} frames:\n{err}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (exit {rc}):\n{err}")
    if n == 0:
        raise ValueError("encode_mp4 received no frames")
    return out_path


# ----------------------------------------------------------------------------- public

def encode_mp4(frames: Iterable[np.ndarray], fps: float, width: int, height: int,
               audio: np.ndarray | None, sample_rate: int, out_path: str, crf: int = 18) -> str:
    """Encode HxWx3 uint8 RGB frames (+ optional (channels, samples) float audio) to an MP4.

    H.264 yuv420p (CRF ``crf``), AAC 192k, faststart. Frames are consumed one at a time.
    Audio is padded with silence or cut so the clip length is set by the frame count.
    Returns the absolute output path.
    """
    width, height = int(width), int(height)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if _choose_backend() == "pyav":
        return _encode_pyav(frames, fps, width, height, audio, sample_rate, out_path, crf)
    return _encode_ffmpeg(frames, fps, width, height, audio, sample_rate, out_path, crf)
