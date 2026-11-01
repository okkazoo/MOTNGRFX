"""footage_flow: dense forward optical flow of a footage clip (the motion the type interacts with; see compose.flow_wake).

    flow = flow_for(frames_u8)              # [T,H,W,3] uint8 -> [T, H/2, W/2, 2] float16, px/frame at half-res, x right / y down
    flow, path = flow_cached("shot.mp4")    # same, cached as <shot>_flow.npy next to the clip + <shot>_flow.mp4 colour-wheel preview

flow[t] is the motion from frame t to t+1 (the last frame repeats the one before). Units are pixels of the HALF-resolution
frame (a 1088x1920 clip -> 544x960 flow), so the same field is valid for any compose size: scale by out_w / flow_w.

Backends
- CUDA: torchvision RAFT (raft_large default, or raft_small; default weights = open weights, BSD-3, downloaded by torchvision on
  first use) at half resolution (padded / resized to a multiple of 8, flow rescaled back), batches of frame pairs, fp16 autocast
  with an automatic fp32 retry if a batch comes back non-finite.
- CPU fallback: OpenCV DIS (preset MEDIUM) at quarter resolution, flow upsampled x2 (and its values x2) to the half-res grid.

CLI:  python tools/fusion/footage_flow.py clip.mp4 [--out DIR] [--method auto|raft|dis] [--model raft_large|raft_small]
                                          [--frames N] [--no-preview] [--force] [--name s1a]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

_RAFT_CACHE = {}


# ================================================================================================== video IO
def probe(path):
    """(w, h, n_frames, fps) of an mp4."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
           "stream=width,height,nb_read_packets,r_frame_rate", "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    return w, h, int(out[3]), float(num) / float(den or 1)


def read_u8(path, size=None, max_frames=None, start=0):
    """mp4 -> uint8 [T,H,W,3] via an ffmpeg pipe. size=(w,h) scales (area) while decoding; start / max_frames select a range."""
    w, h, n, _ = probe(path)
    ow, oh = (w, h) if size is None else (int(size[0]), int(size[1]))
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)]
    vf = []
    if start:
        vf.append(f"select=gte(n\\,{int(start)})")
    if (ow, oh) != (w, h):
        vf.append(f"scale={ow}:{oh}:flags=area")
    if vf:
        cmd += ["-vf", ",".join(vf), "-vsync", "0"]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fs = ow * oh * 3
    return np.frombuffer(raw, np.uint8)[: (len(raw) // fs) * fs].reshape(-1, oh, ow, 3).copy()


# ================================================================================================== CPU: DIS
def _dis_flow(frames, scale=0.25, preset="medium", progress=None):
    """frames uint8 [T,H,W,3] -> float32 [T, H/2, W/2, 2] (half-res px), DIS at `scale` of the full frame."""
    T, H, W = frames.shape[:3]
    hh, ww = max(1, H // 2), max(1, W // 2)
    qh, qw = max(8, int(round(H * scale))), max(8, int(round(W * scale)))
    p = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
         "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset]
    dis = cv2.DISOpticalFlow_create(p)
    gray = [cv2.cvtColor(cv2.resize(f, (qw, qh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
    k = (ww / qw, hh / qh)                                  # quarter-res px -> half-res px
    out = np.zeros((T, hh, ww, 2), np.float16)               # fp16 out: no float32 full-res stack
    for t in range(T - 1):
        fl = dis.calc(gray[t], gray[t + 1], None)
        fl = cv2.resize(fl, (ww, hh), interpolation=cv2.INTER_LINEAR)
        out[t, ..., 0], out[t, ..., 1] = fl[..., 0] * k[0], fl[..., 1] * k[1]
        if progress:
            progress(t + 1, T - 1)
    if T > 1:
        out[T - 1] = out[T - 2]
    return out


# ================================================================================================== CUDA: RAFT
def _raft(model, device):
    key = (model, str(device))
    if key not in _RAFT_CACHE:
        from torchvision.models import optical_flow as of
        if model == "raft_small":
            net = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT, progress=True)
        else:
            net = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT, progress=True)
        _RAFT_CACHE[key] = net.to(device).eval()
    return _RAFT_CACHE[key]


def _raft_flow(frames, device, model="raft_large", batch=8, fp16=True, scale=0.5, progress=None):
    T, H, W = frames.shape[:3]
    hh, ww = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    h8, w8 = max(8, int(round(hh / 8.0)) * 8), max(8, int(round(ww / 8.0)) * 8)
    net = _raft(model, device)
    out = np.zeros((T, hh, ww, 2), np.float16)               # fp16 out: no float32 full-res stack

    def prep(a):                                            # uint8 [b,H,W,3] -> [-1,1] [b,3,h8,w8]
        x = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(h8, w8), mode="area" if scale < 1 else "bilinear", **({} if scale < 1 else {"align_corners": False}))
        return x * 2.0 - 1.0

    def run(a, b, half):
        with torch.no_grad():
            if half and device != "cpu":
                with torch.autocast("cuda", dtype=torch.float16):
                    f = net(a, b)[-1]
            else:
                f = net(a, b)[-1]
        return f.float()

    sx, sy = ww / w8, hh / h8                               # RAFT-grid px -> half-res px
    for s in range(0, T - 1, batch):
        e = min(s + batch, T - 1)
        a, b = prep(frames[s:e]), prep(frames[s + 1:e + 1])
        f = run(a, b, fp16)
        if fp16 and not bool(torch.isfinite(f).all()):
            f = run(a, b, False)
        if (h8, w8) != (hh, ww):
            f = F.interpolate(f, size=(hh, ww), mode="bilinear", align_corners=False)
        f[:, 0] *= sx
        f[:, 1] *= sy
        out[s:e] = f.permute(0, 2, 3, 1).cpu().numpy()
        if progress:
            progress(e, T - 1)
    if T > 1:
        out[T - 1] = out[T - 2]
    return out


# ================================================================================================== public
def _have_raft():
    if torch is None:
        return False
    try:
        import torchvision.models.optical_flow  # noqa: F401
        return True
    except Exception:
        return False


def flow_for(frames_u8, device="auto", method="auto", model="raft_large", batch=8, fp16=True, dis_preset="medium",
             progress=None, verbose=True):
    """Dense forward flow of a clip. frames_u8: uint8 [T,H,W,3] (numpy or torch). -> float16 numpy [T, H//2, W//2, 2]
    (px/frame at half resolution, x right / y down; the last frame repeats the one before).
    device: 'auto' (CUDA if there is one) | 'cuda' | 'cpu'. method: 'auto' (RAFT on CUDA, DIS on CPU) | 'raft' | 'dis'.
    Prints seconds per frame."""
    if torch is not None and isinstance(frames_u8, torch.Tensor):
        frames_u8 = frames_u8.cpu().numpy()
    fr = np.ascontiguousarray(frames_u8)
    if fr.dtype != np.uint8:
        fr = np.clip(np.rint(fr * 255.0), 0, 255).astype(np.uint8)
    if fr.ndim != 4 or fr.shape[-1] != 3:
        raise ValueError(f"frames_u8 must be [T,H,W,3], got {fr.shape}")
    cuda = torch is not None and torch.cuda.is_available()
    dev = ("cuda" if cuda else "cpu") if device == "auto" else device
    m = method
    if m == "auto":
        m = "raft" if (dev == "cuda" and _have_raft()) else "dis"
    t0 = time.perf_counter()
    if m == "raft":
        if not _have_raft():
            raise RuntimeError("method='raft' needs torch + torchvision (torchvision.models.optical_flow)")
        fl = _raft_flow(fr, dev, model=model, batch=batch, fp16=fp16, progress=progress)
        label = f"RAFT({model},{dev}{',fp16' if fp16 and dev != 'cpu' else ''})"
    else:
        fl = _dis_flow(fr, preset=dis_preset, progress=progress)
        label = f"DIS({dis_preset},quarter-res,cpu)"
    dt = time.perf_counter() - t0
    if verbose:
        print(f"footage_flow: {label} {fr.shape[0]} frames {fr.shape[2]}x{fr.shape[1]} -> flow {fl.shape[2]}x{fl.shape[1]}: "
              f"{dt / max(fr.shape[0] - 1, 1):.3f} s/frame ({dt:.1f}s)", file=sys.stderr)
    return fl.astype(np.float16, copy=False)


def wheel(flow, vmax=None):
    """flow [..,H,W,2] -> RGB float32 [..,H,W,3] colour wheel (hue = direction, value = speed / vmax; vmax defaults to the 99th percentile)."""
    f = np.asarray(flow, np.float32)
    mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    if vmax is None:
        vmax = max(float(np.percentile(mag[..., ::2, ::2], 99)), 1e-3)
    ang = (np.arctan2(f[..., 1], f[..., 0]) + np.pi) / (2 * np.pi) * 179.0
    hsv = np.stack([ang, np.full_like(ang, 255.0), np.clip(mag / vmax, 0, 1) * 255.0], -1).astype(np.uint8)
    flat = hsv.reshape(-1, *hsv.shape[-2:], 3) if hsv.ndim > 3 else hsv[None]
    rgb = np.stack([cv2.cvtColor(h, cv2.COLOR_HSV2RGB) for h in flat])
    return (rgb.reshape(hsv.shape) if hsv.ndim > 3 else rgb[0]).astype(np.float32) / 255.0


def write_preview(flow, out_mp4, fps=24, vmax=None):
    from tools.fusion import compose
    if vmax is None:
        mag = np.sqrt(flow[::4, ::4, ::4, 0].astype(np.float32) ** 2 + flow[::4, ::4, ::4, 1].astype(np.float32) ** 2)
        vmax = max(float(np.percentile(mag, 99)), 1e-3)
    compose.encode_mp4((wheel(f, vmax) for f in flow), out_mp4, fps=fps, crf=20, preset="veryfast")
    return str(out_mp4)


def flow_path(video, out_dir=None, name=None):
    video = Path(video)
    return Path(out_dir or video.parent) / f"{name or video.stem}_flow.npy"


def flow_cached(video, out_dir=None, force=False, preview=True, max_frames=None, name=None, **kw):
    """flow_for() of an mp4 with a cache: <out_dir or clip dir>/<clip>_flow.npy (float16, T x H/2 x W/2 x 2) + <clip>_flow.mp4.
    -> (flow, npy path). `kw` goes to flow_for. Frames are decoded at full size (DIS works at 1/4, RAFT at 1/2 of that)."""
    p = flow_path(video, out_dir, name)
    if p.exists() and not force:
        return np.load(p, mmap_mode="r"), p
    frames = read_u8(video, max_frames=max_frames)
    fl = flow_for(frames, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, fl)
    if preview:
        write_preview(fl, p.with_suffix(".mp4"))
    return fl, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="output dir (default: next to the clip)")
    ap.add_argument("--method", default="auto", choices=["auto", "raft", "dis"])
    ap.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frames", type=int, help="only the first N frames")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", help="cache name: <name>_flow.npy (default: the clip's file stem)")
    a = ap.parse_args(argv)
    fl, p = flow_cached(a.video, a.out, force=a.force, preview=not a.no_preview, max_frames=a.frames, name=a.name,
                        device=a.device, method=a.method, model=a.model)
    mag = np.sqrt(np.asarray(fl[..., 0], np.float32) ** 2 + np.asarray(fl[..., 1], np.float32) ** 2)
    print(f"{p}  shape={fl.shape} dtype={fl.dtype}  |flow| mean {mag.mean():.2f} p99 {np.percentile(mag, 99):.2f} px/frame (half-res)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
