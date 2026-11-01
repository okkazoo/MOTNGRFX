#!/usr/bin/env python
"""Cut video clips (or mograph-animated stills) together to a CPU soundtrack, with film treatments,
light-leak bursts, exposure strobe and small held captions. Driven by an edit-list JSON.

    python tools/assemble_clips.py edit.json [--sheet REVIEW_DIR] [--out OUT.mp4]

Edit list (paths are relative to the edit file unless absolute):
{
  "out": "renders/routeB_S1_10s_9x16.mp4",
  "size": [512, 896], "fps": 24, "duration": 10.0, "seed": 7,
  "palette": ["#0e1a24", "#f3eadb", "#e2582a"],
  "audio": {"bpm": 72, "mood": "calm", "key": "D", "scale": "major", "energy": 0.25,
            "phase": 0.0, "sections": [], "risers": [], "sfx": []},     # -> mograph.soundtrack
  "fit": true,            # scale the edit lengths so they sum to "duration"
  "snap": "beats",        # snap cut points to beats | downbeats | none
  # route B: video clips. Each clip owns the edit span between two cut points ("len" seconds or
  # "beats"); "trans" is centred on its outgoing cut, so both clips play under it.
  "clips": [ {"src": "clips/S1_i2v_0.mp4", "in": 0.0, "out": null, "len": 2.5, "speed": null,
              "reverse": false, "stutter": 1, "zoom": [1.0, 1.0],
              "trans": {"type": "crossfade", "dur": 0.6}} ],
      # speed null = play the whole in..out range across the visible span; a number = rate (1 = real time)
      # trans types: cut | crossfade | dip | flash | leak (crossfade under a light-leak burst) | whip (left|right)
  "inserts": [ {"src": "clips/S2_i2v_1.mp4", "at": 1.2, "t": 3.8, "dur": 0.2, "zoom": [1.1, 1.1]} ],
      # flash-frame inserts: replace the flow for "dur" seconds with real-time footage from "at"
  # route A instead of clips: stills animated by the mograph CPU renderer
  "mograph": {"images": ["inputs/a.png"], "scenes": [ {"src": 0, "len": 2.5, "focus": [0.5, 0.5],
              "motion": {"verb": "ken_burns", "zoom": [1.0, 1.1], "pan": [[0, 0], [0.02, 0]]},
              "transition_out": {"type": "crossfade", "dur": 0.6}} ]},
  "treatment": [ {"type": "film_grade", "preset": "warm"}, {"type": "grain", "amount": 0.4} ],
  "leaks": [ {"t": 2.5, "dur": 1.6, "strength": 0.9} ],          # extra light-leak bursts (centred on t)
  "leak_on_cuts": 0.0,                                           # burst strength added on every cut
  "strobe": {"on": "beats", "dip": 4, "spike": 3, "amount": 0.45, "t0": 0, "t1": 99},
  "weave": 0.0,                                                  # gate weave, px at 1080 short side
  "captions": [ {"text": "Somewhere between", "t0": 1.0, "t1": 4.0, "font": "Interlope-Regular.ttf",
                 "size": 0.035, "pos": [0.5, 0.86], "color": "#f3eadb", "fade": 0.4} ],
  "fade_in": 0.3, "fade_out": 0.6,
  "dark_tail": 0.0          # last N seconds become plain palette[0] ground (under an end title)
}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mograph.encode import encode_mp4  # noqa: E402
from mograph.render.easing import clamp01, cubic_in_out  # noqa: E402
from mograph.render.treatments import apply_treatment, light_leak  # noqa: E402
from mograph.soundtrack import build_soundtrack  # noqa: E402

FONT_DIR = os.path.join(ROOT, "mograph", "fonts")


# ----------------------------------------------------------------------------- helpers

def _path(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def _hex(c, default=(1.0, 1.0, 1.0)):
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return default


def beat_grid(audio: dict, duration: float):
    bpm = float(audio.get("bpm") or 120.0)
    ph = float(audio.get("phase", 0.0))
    bl = 60.0 / bpm
    beats = np.arange(ph, duration + bl, bl)
    beats = beats[(beats >= -1e-6) & (beats < duration)]
    return beats, beats[::4]


def cover(img: np.ndarray, W: int, H: int, zoom: float = 1.0) -> np.ndarray:
    """Centre cover-crop + resize to WxH (zoom >= 1 crops further)."""
    h, w = img.shape[:2]
    s = max(W / w, H / h) * max(zoom, 1.0)
    cw, ch = W / s, H / s
    x0, y0 = (w - cw) / 2.0, (h - ch) / 2.0
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


class ClipCache:
    def __init__(self, keep: int = 3):
        self.keep, self.d, self.order = keep, {}, []

    def get(self, path: str):
        if path in self.d:
            return self.d[path]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise SystemExit(f"no frames in {path}")
        self.d[path] = (np.stack(frames), float(fps))
        self.order.append(path)
        while len(self.order) > self.keep:
            self.d.pop(self.order.pop(0), None)
        return self.d[path]


# ----------------------------------------------------------------------------- timeline

def cut_points(items: list, edit: dict, beats, downbeats) -> list[float]:
    """Boundaries b_0 = 0 < b_1 < ... < b_n = duration from item lengths (s or beats)."""
    dur = float(edit["duration"])
    bl = 60.0 / float(edit.get("audio", {}).get("bpm") or 120.0)
    lens = np.array([float(it["beats"]) * bl if "beats" in it else float(it.get("len", 2.0)) for it in items])
    if edit.get("fit", True):
        lens = lens * dur / max(lens.sum(), 1e-6)
    b = np.concatenate([[0.0], np.cumsum(lens)])
    snap = edit.get("snap", "beats")
    grid = beats if snap == "beats" else downbeats if snap == "downbeats" else None
    if grid is not None and len(grid):
        for i in range(1, len(b) - 1):
            j = int(np.argmin(np.abs(grid - b[i])))
            b[i] = max(grid[j], b[i - 1] + 0.2)
    b[-1] = dur
    return [float(x) for x in b]


def _tdur(it) -> float:
    tr = it.get("trans") or it.get("transition_out") or {}
    return 0.0 if tr.get("type", "cut") == "cut" else float(tr.get("dur", 0.5))


def clip_frame(clip: dict, cache: ClipCache, base: str, t: float, span: tuple, W: int, H: int, fps: float):
    """Frame of a clip at timeline time t; span = (visible start, visible end)."""
    arr, cfps = cache.get(_path(base, clip["src"]))
    n = len(arr)
    s_in = float(clip.get("in", 0.0))
    s_out = clip.get("out")
    s_out = (n - 1) / cfps if s_out is None else min(float(s_out), (n - 1) / cfps)
    vis = max(span[1] - span[0], 1e-3)
    u = t - span[0]
    st = int(clip.get("stutter", 1) or 1)
    if st > 1:                                     # frame-rate stutter: hold every st frames
        u = math.floor(u * fps / st) * st / fps
    speed = clip.get("speed")
    if speed is None:
        src_t = s_in + (s_out - s_in) * clamp01(u / vis)
    else:
        src_t = min(s_in + u * float(speed), s_out)
    if clip.get("reverse"):
        src_t = s_out - (src_t - s_in)
    x = float(np.clip(src_t * cfps, 0, n - 1))
    i0 = int(math.floor(x))
    i1 = min(i0 + 1, n - 1)
    a = x - i0
    if a < 1e-3 or i1 == i0:
        f = arr[i0].astype(np.float32) / 255.0
    else:                                          # frame blend for slowed clips
        f = (arr[i0].astype(np.float32) * (1 - a) + arr[i1].astype(np.float32) * a) / 255.0
    z = clip.get("zoom") or [1.0, 1.0]
    zoom = float(z[0]) + (float(z[1]) - float(z[0])) * cubic_in_out(clamp01(u / vis))
    return cover(f, W, H, zoom)


def whip(A, B, p, direction="left"):
    W = A.shape[1]
    e = cubic_in_out(p)
    off = int(round(e * W))
    sgn = -1 if direction == "left" else 1
    out = np.empty_like(A)
    MA = np.float32([[1, 0, sgn * off], [0, 1, 0]])
    MB = np.float32([[1, 0, sgn * off - sgn * W], [0, 1, 0]])
    a = cv2.warpAffine(A, MA, (W, A.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    b = cv2.warpAffine(B, MB, (W, A.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    out = a + b
    v = math.sin(math.pi * p)                      # blur peaks mid-whip
    k = int(round(v * 0.25 * W)) | 1
    if k > 2:
        out = cv2.blur(out, (k, 1))
    return out


def blend(kind: str, A, B, p: float, tr: dict, t: float, seed: int):
    if kind == "crossfade":
        e = cubic_in_out(p)
        return A * (1 - e) + B * e
    if kind == "dip":
        return A * clamp01(1 - 2 * p) if p < 0.5 else B * clamp01(2 * p - 1)
    if kind == "flash":
        w = np.float32(1.0)
        if p < 0.5:
            e = clamp01(2 * p) ** 2
            return A * (1 - e) + w * e
        e = clamp01(2 * p - 1)
        return w * (1 - e) + B * e
    if kind == "leak":
        e = cubic_in_out(p)
        f = A * (1 - e) + B * e
        s = float(tr.get("strength", 1.0)) * math.sin(math.pi * p) ** 0.7
        return light_leak(np.clip(f, 0, 1), t, seed=seed + 31, strength=1.4 * s, speed=2.0)
    if kind == "whip":
        return whip(A, B, p, tr.get("dir", "left"))
    return A if p < 0.5 else B


# ----------------------------------------------------------------------------- base frames

def clip_frames(edit: dict, base: str, W: int, H: int, fps: float, beats, downbeats):
    clips = edit["clips"]
    b = cut_points(clips, edit, beats, downbeats)
    cache = ClipCache()
    d = [_tdur(c) for c in clips]
    spans = []
    for i in range(len(clips)):
        s0 = b[i] - (d[i - 1] / 2 if i > 0 else 0.0)
        s1 = b[i + 1] + (d[i] / 2 if i < len(clips) - 1 else 0.0)
        spans.append((s0, s1))
    edit["_cuts"] = b
    n = int(round(edit["duration"] * fps))
    seed = int(edit.get("seed", 0))
    for fi in range(n):
        t = fi / fps
        i = max(0, min(len(clips) - 1, int(np.searchsorted(b, t, side="right")) - 1))
        f = None
        for j in (i - 1, i):                       # inside a transition centred on b[j+1]?
            if 0 <= j < len(clips) - 1 and d[j] > 0 and abs(t - b[j + 1]) < d[j] / 2:
                A = clip_frame(clips[j], cache, base, t, spans[j], W, H, fps)
                B = clip_frame(clips[j + 1], cache, base, t, spans[j + 1], W, H, fps)
                p = (t - (b[j + 1] - d[j] / 2)) / d[j]
                tr = clips[j]["trans"]
                f = blend(tr.get("type", "cut"), A, B, p, tr, t, seed + j)
                break
        if f is None:
            f = clip_frame(clips[i], cache, base, t, spans[i], W, H, fps)
        for ins in edit.get("inserts") or []:      # flash-frame inserts laid over the flow
            t0 = float(ins["t"])
            if t0 <= t < t0 + float(ins.get("dur", 0.2)):
                f = clip_frame({**ins, "speed": 1.0, "in": float(ins.get("at", 0.0))}, cache, base, t,
                               (t0, t0 + float(ins.get("dur", 0.2))), W, H, fps)
        yield np.clip(f, 0, 1).astype(np.float32)


def mograph_frames(edit: dict, base: str, W: int, H: int, fps: float, beats, downbeats):
    """Route A: stills animated by the mograph CPU renderer (ken_burns / drift / parallax_split,
    mograph transitions), no text, no grain/vignette (the shared post adds those)."""
    from mograph.analyze import analyze_inputs
    from mograph.plan import validate_plan
    from mograph.render import compositor

    mg = edit["mograph"]
    imgs = []
    for p in mg["images"]:
        im = Image.open(_path(base, p)).convert("RGB")
        im.thumbnail((1280, 1280))
        imgs.append(np.asarray(im, np.float32) / 255.0)
    analysis = analyze_inputs("", imgs, None, edit["duration"])
    scenes_in = mg["scenes"]
    b = cut_points(scenes_in, edit, beats, downbeats)
    edit["_cuts"] = b
    a = edit.get("audio", {})
    pal = edit.get("palette") or ["#111111", "#f2f2f2", "#ff4d2e"]
    scenes = []
    for i, s in enumerate(scenes_in):
        bg = {"type": "image", "src": f"img{int(s['src'])}", "motion": s.get("motion", {"verb": "ken_burns"}),
              "treatment": {"duotone": None, "blur": 0.0, "darken": 0.0}}
        if s.get("focus"):
            bg["focus"] = s["focus"]
        sc = {"id": f"s{i}", "t0": b[i], "t1": b[i + 1], "background": bg, "layers": [],
              "transition_out": s.get("transition_out", {"type": "cut", "dur": 0.0})}
        scenes.append(sc)
    plan = {"version": 1, "seed": int(edit.get("seed", 0)), "duration": float(edit["duration"]), "fps": fps,
            "style": {"palette": pal, "background": "palette:0", "fonts": {"display": "Big Shoulders Display", "body": "Space Grotesk"},
                      "motion": mg.get("motion", "smooth"), "energy": float(a.get("energy", 0.5)),
                      "grain": 0.0, "vignette": 0.0, "beat_flash": 0.0, "treatment": None},
            "audio": {"source": "synth", "bpm": float(a.get("bpm", 120)), "key": a.get("key", "A"),
                      "scale": a.get("scale", "minor"), "mood": a.get("mood", "calm"),
                      "beats": [float(x) for x in beats], "downbeats": [float(x) for x in downbeats],
                      "sections": [], "risers": [], "sfx": []},
            "voice": None, "scenes": scenes}
    errs = validate_plan(plan, analysis)
    if errs:
        print("plan warnings:", errs[:8])
    fmt = "9:16" if H >= W else "16:9"
    orig = compositor.frame_size
    compositor.frame_size = lambda f, q: (W, H)     # render at the edit's size, not 360/1080
    try:
        r = compositor.Renderer(plan, analysis, fmt, "draft")
    finally:
        compositor.frame_size = orig
    for fi in range(r.ctx.n_frames):
        yield r.frame(fi).astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- post

class Captions:
    def __init__(self, caps: list, W: int, H: int):
        self.items = []
        S = min(W, H)
        for c in caps or []:
            font = ImageFont.truetype(os.path.join(FONT_DIR, c.get("font", "SpaceGrotesk-Bold.ttf")),
                                      max(8, int(round(float(c.get("size", 0.035)) * S))))
            # render once at 2x for clean edges, then downsample
            txt = str(c["text"])
            big = ImageFont.truetype(font.path, font.size * 2)
            l, tp, r, bt = big.getbbox(txt)
            tw, th = r - l + 8, bt - tp + 8
            im = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(im).text((4 - l, 4 - tp), txt, font=big, fill=255)
            im = im.resize((max(1, tw // 2), max(1, th // 2)), Image.LANCZOS)
            m = np.asarray(im, np.float32) / 255.0
            sh = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.004 * S)) * 0.55
            px, py = c.get("pos", [0.5, 0.86])
            x0 = int(round(px * W - m.shape[1] / 2))
            y0 = int(round(py * H - m.shape[0] / 2))
            self.items.append((float(c["t0"]), float(c["t1"]), float(c.get("fade", 0.3)), m, sh, x0, y0,
                               np.array(_hex(c.get("color", "#ffffff")), np.float32)))

    def draw(self, f: np.ndarray, t: float):
        H, W = f.shape[:2]
        for t0, t1, fd, m, sh, x0, y0, col in self.items:
            if not (t0 <= t < t1):
                continue
            a = 1.0 if fd <= 0 else min(clamp01((t - t0) / fd), clamp01((t1 - t) / fd))
            h, w = m.shape
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if xb <= xa or yb <= ya:
                continue
            mm = m[ya - y0:yb - y0, xa - x0:xb - x0, None] * a
            ss = sh[ya - y0:yb - y0, xa - x0:xb - x0, None] * a
            roi = f[ya:yb, xa:xb]
            roi *= 1 - ss
            roi[:] = roi * (1 - mm) + col * mm
        return f


def post(frames, edit: dict, W: int, H: int, fps: float, beats, downbeats):
    seed = int(edit.get("seed", 0))
    dur = float(edit["duration"])
    pal = edit.get("palette")
    treat = edit.get("treatment")
    leaks = list(edit.get("leaks") or [])
    loc = float(edit.get("leak_on_cuts", 0.0) or 0.0)
    strobe = edit.get("strobe")
    weave = float(edit.get("weave", 0.0) or 0.0) * min(W, H) / 1080.0
    caps = Captions(edit.get("captions"), W, H)
    fin, fout = float(edit.get("fade_in", 0.3)), float(edit.get("fade_out", 0.6))
    dark_tail = float(edit.get("dark_tail", 0.0) or 0.0)
    tail_col = np.array(_hex((pal or ["#000000"])[0], (0.0, 0.0, 0.0)), np.float32)
    if dark_tail > 0:
        fout = 0.0                                  # the end title holds to the last frame
    st_times = None
    if strobe:
        on = strobe.get("on", "beats")
        st_times = np.array(beats if on == "beats" else downbeats if on == "downbeats" else on, np.float64)
        st_times = st_times[(st_times >= float(strobe.get("t0", 0))) & (st_times <= float(strobe.get("t1", 1e9)))]
    rng = np.random.default_rng(seed + 77)
    for fi, f in enumerate(frames):
        t = fi / fps
        if loc > 0 and fi == 0:
            leaks += [{"t": c, "dur": 0.5, "strength": loc} for c in edit.get("_cuts", [])[1:-1]]
        if treat:
            f = apply_treatment(f, treat, pal, fi, seed, fps)
        for lk in leaks:
            ld = float(lk.get("dur", 1.0))
            u = (t - float(lk["t"])) / ld + 0.5
            if 0 < u < 1:
                s = float(lk.get("strength", 0.8)) * math.sin(math.pi * u) ** 0.8
                f = light_leak(f, t, seed=seed + int(lk["t"] * 100), strength=1.3 * s, speed=1.5)
        if st_times is not None and len(st_times):
            dip, spk = int(strobe.get("dip", 4)), int(strobe.get("spike", 3))
            amt = float(strobe.get("amount", 0.45))
            k = (t - st_times) * fps                # frames relative to each beat
            g = 1.0
            pre = k[(k > -dip) & (k < 0)]
            if len(pre):
                g *= 1.0 - amt * (1 + float(pre.max()) / dip)      # dip ramps down into the beat
            post_ = k[(k >= 0) & (k < spk)]
            if len(post_):
                lift = amt * 1.3 * (1 - float(post_.min()) / spk)
                f = f + (1 - f) * np.float32(lift)
            f = f * np.float32(g)
        if dark_tail > 0 and t >= dur - dark_tail:  # plain ground under an end title
            f = np.empty_like(f)
            f[:] = tail_col
        if weave > 0:
            dx, dy = rng.normal(0, weave * 0.6), rng.normal(0, weave)
            f = cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        f = caps.draw(np.ascontiguousarray(f, np.float32), t)
        g = min(clamp01(t / fin) if fin > 0 else 1.0, clamp01((dur - t) / fout) if fout > 0 else 1.0)
        if g < 1:
            f = f * np.float32(g)
        yield (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def soundtrack(edit: dict, beats, downbeats):
    a = dict(edit.get("audio") or {})
    sfx = list(a.get("sfx") or [])
    if a.get("auto_sfx", True) and "clips" in edit:
        for c, t in zip(edit["clips"][:-1], edit.get("_cuts", [])[1:-1]):
            k = (c.get("trans") or {}).get("type")
            if k == "whip":
                sfx.append({"t": t, "type": "whoosh", "gain": 0.5, "dur": 0.4})
            elif k == "flash":
                sfx.append({"t": t, "type": "impact", "gain": 0.5})
    plan = {"duration": float(edit["duration"]), "seed": int(edit.get("seed", 0)),
            "style": {"energy": float(a.get("energy", 0.5))},
            "audio": {"source": "synth", "bpm": float(a.get("bpm", 120)), "key": a.get("key", "A"),
                      "scale": a.get("scale", "minor"), "mood": a.get("mood", "calm"),
                      "beats": [float(x) for x in beats], "downbeats": [float(x) for x in downbeats],
                      "sections": a.get("sections") or [], "risers": a.get("risers") or [], "sfx": sfx}}
    wav, sr = build_soundtrack(plan, {})
    n = wav.shape[-1]
    fo = int(0.5 * sr)
    wav[:, n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return wav, sr


def contact_sheet(path: str, sheet_dir: str, n: int = 12, cols: int = 6):
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tiles = []
    for k in range(n):
        idx = int(round((k + 0.5) * N / n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = 240 / max(h, w)
        fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.putText(fr, f"{idx / fps:.2f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * (th + 4) + 4, cols * (tw + 4) + 4, 3), 20, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[4 + r * (th + 4):4 + r * (th + 4) + t.shape[0], 4 + c * (tw + 4):4 + c * (tw + 4) + t.shape[1]] = t
    os.makedirs(sheet_dir, exist_ok=True)
    out = os.path.join(sheet_dir, os.path.splitext(os.path.basename(path))[0] + "_sheet.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def assemble(edit_path: str, out: str | None = None, sheet: str | None = None) -> str:
    with open(edit_path, encoding="utf-8") as f:
        edit = json.load(f)
    base = os.path.dirname(os.path.abspath(edit_path))
    W, H = (int(v) for v in edit.get("size", [512, 896]))
    fps = float(edit.get("fps", 24))
    beats, downbeats = beat_grid(edit.get("audio") or {}, float(edit["duration"]))
    gen = (mograph_frames if "mograph" in edit else clip_frames)(edit, base, W, H, fps, beats, downbeats)
    first = next(gen)                               # sets edit["_cuts"] before the soundtrack is built

    def chain():
        yield first
        yield from gen
    wav, sr = soundtrack(edit, beats, downbeats)
    out = out or _path(base, edit["out"])
    t0 = time.time()
    encode_mp4(post(chain(), edit, W, H, fps, beats, downbeats), fps, W, H, wav, sr, out, crf=18)
    print(f"{out}  cuts={[round(c, 2) for c in edit.get('_cuts', [])]}  {time.time() - t0:.1f}s")
    if sheet:
        print(contact_sheet(out, sheet))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edit")
    ap.add_argument("--out")
    ap.add_argument("--sheet", help="also write a 12-frame contact sheet into this folder")
    a = ap.parse_args()
    assemble(a.edit, a.out, a.sheet)


if __name__ == "__main__":
    main()
