"""THE FRAMES AS DESIGNED GO TO THE RENDER (2026-11-03 user: "I would like these matte decisions on the frames to come through on the render").

The page keeps each shot's type as shots[].line + .font + .tl (rot, align, anch, cols, x, y, s, layer, bg). This module renders exactly that over the source video with the job's full-length mattes:
no design model, no retime, nothing re-chosen. Layout follows the page's fitVidType, the looks follow its CSS (.tmk / .tbh / .sb-to):
  front    type over the picture
  behind   type between the background and the subject (matte)
  mask     solid colour, the picture through the letters, a ring around them
  maskm    the same + the subject over the colour, the ring over the subject
  maskb    the same + the subject over the colour, the ring under the subject
tl.metal = gold | chrome (front / behind only): LTX 2.5 renders the graphic (union-control on the word's depth, on black, the word + material in the prompt); the letters go over the source through the graphic's own mask, so the picture and the word's shape stay.
transitions = False keeps every cut hard; True turns cuts into 3D / motion moves, each ending ON the source's cut frame, so the frame count and the rhythm stay. Words stop before a move starts in either version.
lines = True (default, both edit modes, tools/app/frame_lines.py): each word keeps its GUI layout as its resting pose, and the approved library strands (outline hand-off + braid / circuit_board / fan / spiral_around / twist / ripple) carry it into and out of that pose: the sung frame never moves, B's fill is the plain word once the form-in ends. lines = False: hard cuts between the words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.app import overlay_mode as OM  # noqa: E402

LINE_H = 1.02                     # .sb-ov line-height
GRID_GAP = 0.06                   # .gr gap, em
LOOKS = ("front", "behind", "mask", "maskm", "maskb")
MATTE_LOOKS = ("behind", "maskm", "maskb")
METALS = ("gold", "chrome")
LONG_SHOT = 30                    # frames: a shot this long takes a full move; shorter ones a quick one on every third cut
KINDS_LONG = ("flip", "whip", "cube", "zoom", "roll")
KINDS_FAST = ("whip", "zoom", "roll")


# ------------------------------------------------------------------------------------------------ the design
def has_frame_design(story) -> bool:
    """True when the page's story carries type the user laid out on the frames."""
    shots = (story or {}).get("shots") if isinstance(story, dict) else None
    return bool(shots) and any(str(s.get("line") or "").strip() and (isinstance(s.get("tl"), dict) or s.get("src_start") is not None) for s in shots)     # a storyboard shot of the clip with words: an untouched one renders as the frame shows it too


def layer_of(t: dict) -> str:
    """the page's layerOf."""
    l = (t or {}).get("layer") or ("behind" if (t or {}).get("behind") else "front")
    l = "mask" if l == "inverse" else l
    return l if l in LOOKS else "front"


def shot_frames(shots: list[dict], fps: float, nframes: int) -> list[tuple[int, int]]:
    """each shot's [a, b) in source frames; contiguous, last one closes at the clip's end."""
    a = [max(0, min(nframes, int(round(float(s["src_start"]) * fps)))) for s in shots]
    out = []
    for i, s in enumerate(shots):
        b = a[i + 1] if i + 1 < len(shots) else nframes
        out.append((a[i], b))
    return out


def cut_plan(spans: list[tuple[int, int]], merged_cuts: set[int] | None = None) -> dict[int, dict]:
    """{cut frame: {kind, h}} for every cut that gets a move: h frames before and after it. Long shots always; the fast section every third cut. A cut inside a held word keeps hard."""
    merged_cuts = merged_cuts or set()
    plan, n_long, n_fast, n_seen = {}, 0, 0, 0
    for (a0, b0), (a1, b1) in zip(spans[:-1], spans[1:]):
        room = min(b0 - a0, b1 - a1)
        if b0 in merged_cuts:
            continue
        if room >= LONG_SHOT:
            plan[b0] = dict(kind=KINDS_LONG[n_long % len(KINDS_LONG)], h=7)
            n_long += 1
        else:
            n_seen += 1
            if n_seen % 3 == 0 and room >= 8:
                plan[b0] = dict(kind=KINDS_FAST[n_fast % len(KINDS_FAST)], h=3)
                n_fast += 1
    return plan


def plan_words(shots: list[dict], fps: float, nframes: int, hits=None, moves: bool = True) -> tuple[list[dict], dict[int, dict]]:
    """the words as frame windows: neighbouring shots with the same words and the same layout hold as ONE word across their cuts; a word starts after a move ends and stops before the next starts.
    hits = the page's story.hits ({land} on the piece clock): a word sung inside its shot lands when it is sung and holds to the end of the shot."""
    spans = shot_frames(shots, fps, nframes)
    groups: list[dict] = []
    for s, sp in zip(shots, spans):
        line = str(s.get("line") or "").strip()
        if not line or sp[1] <= sp[0]:
            continue
        t = dict(s.get("tl") or {})
        sig = json.dumps([line, s.get("font"), {k: v for k, v in t.items() if k not in ("color", "outline", "colorA", "colorBusy")}], sort_keys=True)      # the sampled colour differs shot to shot: a held word is still one word, in its first shot's colour
        if groups and groups[-1]["sig"] == sig and groups[-1]["b"] == sp[0]:
            groups[-1]["b"] = sp[1]
            groups[-1]["inner"].append(sp[0])
        else:
            off = float(s["src_start"]) - float(s["start"]) if s.get("start") is not None and s.get("src_start") is not None else 0.0      # piece clock -> clip clock
            groups.append(dict(sig=sig, n=s.get("n"), text=line, font=s.get("font"), tl=t, a=sp[0], b=sp[1], inner=[], off=off))
    merged = {c for g in groups for c in g["inner"]}
    plan = cut_plan(spans, merged)
    words = []
    for g in groups:
        h0 = plan.get(g["a"], {}).get("h", 0) if moves else 0      # moves = False (overlay only): no transition needs the room, the word has its whole shot
        h1 = plan.get(g["b"], {}).get("h", 0) if moves else 0
        start, end = g["a"] + h0 + (1 if h0 else 0), g["b"] - h1 - (1 if h1 else 0)
        if end - start < 4:
            start, end = g["a"], g["b"]
        lands = sorted(f for f in (int(round((float(h["land"]) + g["off"]) * fps)) for h in (hits or []) if h.get("land") is not None) if g["a"] <= f < g["b"])
        if lands and start < lands[0] and end - lands[0] >= 4:
            start = lands[0]
        words.append(dict(n=g["n"], text=g["text"], font=g["font"], tl=g["tl"], start=start, end=end))
    return words, plan


def font_path(name) -> Path:
    """a liked font's file; anything else is refused (HARD rule, mograph/fonts/liked.json)."""
    liked = json.loads((ROOT / "mograph" / "fonts" / "liked.json").read_text(encoding="utf-8"))["fonts"]
    if str(name) not in {f["file"] for f in liked}:
        raise ValueError(f"font not in the liked list: {name}")
    return ROOT / "mograph" / "fonts" / str(name)


# ------------------------------------------------------------------------------------------------ layout: the page's fitVidType
def layout(text: str, font: Path, fw: int, fh: int, t: dict) -> dict:
    """letters [{ch, x, y (baseline), w}] about the block's centre at the fitted size (before t.s), + fs, cx, cy, rot, s, box."""
    from PIL import ImageFont
    f100 = ImageFont.truetype(str(font), 100)
    asc, desc = f100.getmetrics()
    t = t or {}
    deg, al, s, cols = int(t.get("rot") or 0) % 360, t.get("align") or "c", float(t.get("s") or 1), int(t.get("cols") or 0)
    rot = deg % 180 != 0
    span = fh if rot else fw
    txt = text.upper()
    if cols:
        ch = [c for c in txt if not c.isspace()]
        rows = [ch[k:k + cols] for k in range(0, len(ch), cols)]
        cell = max(f100.getlength(c) for c in ch) / 100
        row_w = lambda r: len(r) * cell + (len(r) - 1) * GRID_GAP      # noqa: E731
        w100 = max(row_w(r) for r in rows) * 100
        h100 = len(rows) * LINE_H * 100
    else:
        rows = [l.strip() for l in txt.split("\n") if l.strip()] or [""]
        cell = 0.0
        w100 = max(f100.getlength(l) for l in rows)
        h100 = len(rows) * LINE_H * 100
    fs = min(fw * 0.26, 100 * min(0.88 * fw / (h100 if rot else w100), 0.84 * fh / (w100 if rot else h100)))
    font_fs = ImageFont.truetype(str(font), max(4, int(round(fs))))
    k = fs / 100
    base = (LINE_H * fs - (asc + desc) * k) / 2 + asc * k
    W = w100 * k
    if al == "j" and not cols:
        W = max(W, 0.88 * span / s)
    H = len(rows) * LINE_H * fs
    letters = []
    for i, row in enumerate(rows):
        y = -H / 2 + i * LINE_H * fs + base
        if cols:
            n, c_w = len(row), cell * fs
            lw = n * c_w + (n - 1) * GRID_GAP * fs
            gap = GRID_GAP * fs
            if al == "j" and n > 1:
                gap, x0 = (W - n * c_w) / (n - 1), -W / 2
            else:
                x0 = {"l": -W / 2, "r": W / 2 - lw}.get(al, -lw / 2)
            for j, c in enumerate(row):
                g = f100.getlength(c) * k
                letters.append(dict(ch=c, x=x0 + j * (c_w + gap) + (c_w - g) / 2, y=y, w=g))
        else:
            lw = f100.getlength(row) * k
            n = len(row)
            ls = (W - lw) / (n - 1) if al == "j" and n > 1 else 0.0
            x0 = -W / 2 if al in ("l", "j") else (W / 2 - lw if al == "r" else -lw / 2)
            for j, c in enumerate(row):
                if c.isspace():
                    continue
                letters.append(dict(ch=c, x=x0 + f100.getlength(row[:j]) * k + j * ls, y=y, w=f100.getlength(c) * k))
    cx, cy = (0.5 if t.get("x") is None else float(t["x"])) * fw, (0.5 if t.get("y") is None else float(t["y"])) * fh
    if t.get("anch") and not cols:
        L, m, flip = w100 * k * s, fw * 0.06, deg in (180, 270)
        pos = span / 2 if al in ("c", "j") else (m + L / 2 if (al == "l") != flip else span - m - L / 2)
        if rot:
            cy = pos
        else:
            cx = pos
    return dict(letters=letters, fs=fs, cx=cx, cy=cy, rot=deg, s=s, box=(W, H), font=str(font), font_px=int(font_fs.size))


def to_frame(lay: dict, x: float, y: float) -> tuple[float, float]:
    """block point -> frame pixel: scale, then rotate clockwise, then the block's centre (the page's CSS transform)."""
    a = math.radians(lay["rot"])
    sx, sy = x * lay["s"], y * lay["s"]
    return lay["cx"] + sx * math.cos(a) - sy * math.sin(a), lay["cy"] + sx * math.sin(a) + sy * math.cos(a)


class FlatType:
    """the letters as a mask, drawn once at 2x and warped per frame."""
    SS = 2

    def __init__(self, lay: dict):
        from PIL import Image, ImageDraw, ImageFont
        self.lay = lay
        W, H = lay["box"]
        self.pad = int(lay["fs"] * 0.6)
        self.w, self.h = int(W * self.SS) + 2 * self.pad, int(H * self.SS) + 2 * self.pad
        font = ImageFont.truetype(lay["font"], max(4, int(round(lay["fs"] * self.SS))))
        self.each = []
        all_ = Image.new("L", (self.w, self.h), 0)
        for l in lay["letters"]:
            one = Image.new("L", (self.w, self.h), 0)
            ImageDraw.Draw(one).text((self.w / 2 + l["x"] * self.SS, self.h / 2 + l["y"] * self.SS), l["ch"], fill=255, font=font, anchor="ls")
            self.each.append(one)
            all_.paste(255, (0, 0), one)
        self.all = np.asarray(all_, np.uint8)

    def matrix(self, zoom: float = 1.0):
        L = self.lay
        a = math.radians(L["rot"])
        k = L["s"] * zoom / self.SS
        c, s = math.cos(a) * k, math.sin(a) * k
        return np.array([[c, -s, L["cx"] - c * self.w / 2 + s * self.h / 2], [s, c, L["cy"] - s * self.w / 2 - c * self.h / 2]], np.float32)

    def mask(self, fw: int, fh: int, zoom: float = 1.0, src=None) -> np.ndarray:
        import cv2
        return cv2.warpAffine(self.all if src is None else src, self.matrix(zoom), (fw, fh), flags=cv2.INTER_AREA if self.lay["s"] * zoom < 1 else cv2.INTER_LINEAR).astype(np.float32) / 255.0


def pick_colors(frame: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """(fill, ring, solid) in BGR from the shot's own picture: fill = its strongest bright colour, ring = a dark version of the opposite hue, solid = its deepest saturated colour."""
    import cv2
    hsv = cv2.cvtColor(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    s, v = hsv[:, 1] / 255, hsv[:, 2] / 255
    hue = hsv[int(np.argmax(s * v)), 0]
    deep = hsv[int(np.argmax(s * (1 - np.abs(v - 0.35))))]
    to_bgr = lambda h, s_, v_: tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h % 180, s_, v_]]]), cv2.COLOR_HSV2BGR)[0, 0])      # noqa: E731
    return to_bgr(hue, 170, 255), to_bgr(hue + 90, 230, 70), to_bgr(deep[0], max(150, deep[1]), 55)


# ------------------------------------------------------------------------------------------------ mattes: the job's Alpha-Gen stacks
class Mattes:
    """source frame -> the full-length matte (segments.json a, b, job + <stem>_alpha_h<job>_image_<k>_.png)."""

    def __init__(self, segments: list[dict], alpha_dir, stem: str, size: tuple[int, int]):
        self.segs, self.dir, self.stem, self.size = segments, Path(alpha_dir), stem, size
        self.last = None

    def path(self, f: int) -> Path | None:
        for s in self.segs:
            if s["a"] <= f < s["b"] and s.get("job"):
                return self.dir / f"{self.stem}_alpha_h{s['job']}_image_{f - s['a'] + 1:05d}_.png"
        return None

    def get(self, f: int) -> np.ndarray:
        import cv2
        if self.last and self.last[0] == f:
            return self.last[1]
        p = self.path(f)
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p and p.is_file() else None
        if im is None:
            raise FileNotFoundError(f"no matte for frame {f}: {p}")
        m = cv2.resize(im, self.size, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        m = np.clip(m, 0, 1)
        self.last = (f, m)
        return m


# ------------------------------------------------------------------------------------------------ metal: LTX 2.5 renders the graphic (union-control follows the word's depth; the prompt names the word and the material)
MATERIALS = {"gold": "polished liquid gold, mirror-bright, warm yellow gold", "chrome": "polished mirror chrome, liquid silver, high contrast reflections"}
LTX_LAND = 5                      # frames: the control word settles from 1.14x
LTX_DRIFT = 0.018                 # the word grows this much over its hold, so the light moves over the metal


def metal_prompt(text: str, metal: str) -> str:
    words = " ".join(text.upper().split())
    return (f'The word "{words}" as thick extruded 3D letters made of {MATERIALS[metal]}, bevelled edges, glossy specular highlights, a bright streak of light slowly sweeping across the metal, '
            "exact letter shapes, every letter sharp and complete, studio product shot, the letters alone on a pure black background, nothing else in the frame, locked-off camera")


def control_zoom(i: int, n: int) -> float:
    """the word's scale at its frame i: lands, then a slow growth."""
    if i < LTX_LAND:
        return 1.0 + 0.14 * (1 - (i + 1) / LTX_LAND) ** 2
    return 1.0 + LTX_DRIFT * (i - LTX_LAND) / max(1, n - LTX_LAND)


def height_map(flat: "FlatType") -> np.ndarray:
    """the letters as a pillowed depth picture (distance from each edge): what the control pass reads."""
    import cv2
    r = max(3.0, flat.lay["fs"] * flat.SS * 0.17)
    d = cv2.distanceTransform((flat.all > 127).astype(np.uint8), cv2.DIST_L2, 5)
    h = np.sqrt(np.clip(1 - (1 - np.clip(d / r, 0, 1)) ** 2, 0, 1))
    return (((0.45 + 0.55 * h) * (flat.all > 127)) * 255).astype(np.uint8)


def ltx_size(fw: int, fh: int) -> tuple[int, int]:
    """the two-stage control size of this aspect (sides multiples of 128): 2048 long side, then resized to the frame."""
    return (2048, 1152) if fw > fh else ((1152, 2048) if fh > fw else (1536, 1536))


def metal_key(word: dict, lay: dict, size) -> str:
    return hashlib.sha1(json.dumps([word["text"], word["tl"], word["start"], word["end"], lay["letters"], lay["cx"], lay["cy"], list(size), metal_prompt(word["text"], word["tl"]["metal"]), LTX_LAND, LTX_DRIFT], sort_keys=True).encode()).hexdigest()[:14]


def write_control(flat: "FlatType", n: int, size: tuple[int, int], out: Path, fps_str: str) -> None:
    w, h = size
    src = height_map(flat)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", fps_str, "-i", "-", "-c:v", "libx264", "-crf", "8", "-preset", "fast", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        m = flat.mask(w, h, control_zoom(i, n), src)
        enc.stdin.write((m * 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("the control clip failed")


def render_metal(word: dict, lay: dict, size: tuple[int, int], work: Path, comfy, input_dir: Path, output_dir: Path, fps_str: str, seed: int = 1, log=print) -> Path:
    """the word's clip from LTX (kept and reused while nothing about the word changes): a depth control of the graphic -> union-control, the word and the material in the prompt."""
    import shutil
    from tools.fusion import ltx_ic
    d = work / "metal" / metal_key(word, lay, size)
    clip = d / "ltx.mp4"
    if clip.is_file():
        return clip
    d.mkdir(parents=True, exist_ok=True)
    n = word["end"] - word["start"]
    frames = ((max(n, 9) - 1 + 7) // 8) * 8 + 1                               # the next 8k+1 at or over the word's length
    name = f"frame_ctrl_{d.name}.mp4"
    write_control(FlatType(lay), frames, size, input_dir / name, fps_str)
    wh = ltx_size(*size)
    g = ltx_ic.build("control", ref_video=name, prompt=metal_prompt(word["text"], word["tl"]["metal"]), size=wh, frames=frames, seed=seed + int(word.get("n") or 1), two_stage=True, out_prefix=f"frame_metal/{d.name}", crop="disabled")
    h = comfy.wait(comfy.submit(g, f"metal {word['text']}"), 1800, f"metal {word['text']}")
    vids = (h.get("outputs") or {}).get("62", {})
    vids = vids.get("images") or vids.get("videos") or []
    if not vids:
        raise RuntimeError(f"LTX saved no video for {word['text']!r}")
    shutil.copyfile(output_dir / vids[0].get("subfolder", "") / vids[0]["filename"], clip)
    log(f"metal {word['tl'].get('metal')} {word['text']!r}: {n} frames, LTX {wh[0]}x{wh[1]}, {comfy.compute_s(h)} s")
    return clip


# ------------------------------------------------------------------------------------------------ one word over one frame
def pop(i: int, n: int, land: int = 3) -> float:
    """flat type lands with a short overshoot, then holds still."""
    return 1.0 + 0.09 * (1 - i / land) ** 2 if (i < land and n > land + 3) else 1.0


HIDDEN_BELOW = 0.85      # a matte look with a letter less visible than this gets a clear reading moment first
CLEAR_SHARE = 0.55       # the word reads in front of the subject over this share of its hold ...
CLEAR_RAMP = 10          # ... then the subject comes over it across this many frames
EDGE_AMP = 0.006        # the live edge: how far the letter edges drift, share of the type's size
EDGE_CELL = 0.07         # the noise's grain, share of the type's size
EDGE_STEP = 6            # frames between two noise fields


def clear_gain(i: int, n: int) -> float:
    """how much the subject covers a hidden word at its frame i: 0 while it reads clearly, then up to 1. Short words stay readable throughout."""
    if n < 2 * CLEAR_RAMP:
        return 0.0
    u = (i - CLEAR_SHARE * n) / CLEAR_RAMP
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


class Word:
    def __init__(self, w: dict, lay: dict, size: tuple[int, int]):
        self._hidden, self._noise = None, {}
        self.w, self.lay, self.size, self.look = w, lay, size, layer_of(w["tl"])
        self.metal = w["tl"].get("metal") if w["tl"].get("metal") in METALS and self.look in ("front", "behind") else None
        self.flat = FlatType(lay)
        self.fill = self.ring = self.solid = None
        self.metal_clip = None
        self._metal_frames = None

    def colours(self, frame: np.ndarray) -> None:
        if self.fill is None:
            self.fill, self.ring, self.solid = pick_colors(frame)
            t = self.w["tl"]
            hexbgr = lambda h: (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))      # noqa: E731
            if isinstance(t.get("bg"), str) and len(t["bg"]) == 7:
                self.solid = hexbgr(t["bg"])
            if isinstance(t.get("color"), str) and len(t["color"]) == 7:
                self.fill = hexbgr(t["color"])
            if isinstance(t.get("outline"), str) and len(t["outline"]) == 7:
                self.ring = hexbgr(t["outline"])

    def metal_frame(self, i: int) -> np.ndarray:
        """frame i of the word's LTX clip at the frame's size. Frames arrive in order."""
        import cv2
        if self._metal_frames is None:
            ps = OM.probe_video(self.metal_clip)
            self._metal_frames = (OM._frames(self.metal_clip, ps["width"], ps["height"]), -1, None)
        it, k, im = self._metal_frames
        while k < i:
            im, k = next(it), k + 1
        self._metal_frames = (it, k, im)
        return cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)

    def ring_of(self, m: np.ndarray) -> np.ndarray:
        import cv2
        r = max(2, int(round(self.size[0] * 0.0028)))
        return np.clip(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) - m, 0, 1)

    def _field(self, key: int) -> np.ndarray:
        """one smooth noise field (dx, dy), -1..1, the same for this word on every run."""
        import cv2
        if key not in self._noise:
            fw, fh = self.size
            cell = max(6.0, self.lay["fs"] * self.lay["s"] * EDGE_CELL)
            seed = int(hashlib.sha1(self.w["text"].encode()).hexdigest()[:8], 16)
            g = np.random.default_rng([seed, key]).uniform(-1, 1, (max(2, int(fh / cell)), max(2, int(fw / cell)), 2)).astype(np.float32)
            self._noise = {k: v for k, v in self._noise.items() if k >= key - 1}
            self._noise[key] = cv2.resize(g, (fw, fh), interpolation=cv2.INTER_CUBIC)
        return self._noise[key]

    def live_edge(self, m: np.ndarray, f: int) -> np.ndarray:
        """the letters' mask with its edges drifting a little: subtle slow noise, so the held word is never dead still and stays fully readable."""
        import cv2
        ys, xs = np.where(m > 0.004)
        if not len(xs):
            return m
        amp = max(1.0, self.lay["fs"] * self.lay["s"] * EDGE_AMP)
        pad = int(amp) + 3
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1), max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
        k, u = f // EDGE_STEP, (f % EDGE_STEP) / EDGE_STEP
        d = (self._field(k)[y0:y1, x0:x1] * (1 - u) + self._field(k + 1)[y0:y1, x0:x1] * u) * amp
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        out = m.copy()
        out[y0:y1, x0:x1] = cv2.remap(m, gx + d[:, :, 0], gy + d[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    def over(self, frame: np.ndarray, f: int, mattes: Mattes | None, fi: int | None = None) -> np.ndarray:
        """fi = the frame whose pose is drawn (frame_lines holds a releasing word still past its end); the picture and the matte are always frame f's."""
        import cv2
        i, n = (f if fi is None else fi) - self.w["start"], self.w["end"] - self.w["start"]
        self.colours(frame)
        F = frame.astype(np.float32)
        M = None
        if self.look in MATTE_LOOKS and mattes is not None:
            M = mattes.get(f)[:, :, None]
            if self._hidden is None:
                self._hidden = min(self.visibility(f, mattes) or [1.0]) < HIDDEN_BELOW
            if self._hidden:
                M = M * clear_gain(i, n)
        fw, fh = self.size
        if self.metal:
            rgb = self.metal_frame(i)
            core = self.flat.mask(fw, fh, control_zoom(i, n))
            r = max(2, int(round(self.lay["fs"] * self.lay["s"] * 0.035)))
            near = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            key = np.clip((rgb.max(axis=2).astype(np.float32) / 255.0 - 0.07) / 0.12, 0, 1)         # the extrusion and bevel LTX adds just outside the glyph, never its black ground
            a = np.maximum(core, cv2.GaussianBlur(near * key, (0, 0), 1.0))[:, :, None]
            if M is not None:
                a = a * (1 - M)
            return (F * (1 - a) + rgb.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
        m = self.live_edge(self.flat.mask(fw, fh, pop(i, n)), f)
        if self.look in ("front", "behind"):
            ring = self.ring_of(m)[:, :, None]
            out = F * (1 - ring) + np.float32(self.ring) * ring
            a = m[:, :, None]
            out = out * (1 - a) + np.float32(self.fill) * a
            if M is not None:
                out = out * (1 - M) + F * M
            return out.clip(0, 255).astype(np.uint8)
        a = m[:, :, None]
        out = np.float32(self.solid) * (1 - a) + F * a                      # the picture through the letters on the solid colour
        ring = self.ring_of(m)[:, :, None]
        rc = np.float32((245, 245, 245)) if sum(self.solid) < 380 else np.float32((16, 16, 16))
        if self.look == "maskb":
            out = out * (1 - ring) + rc * ring
        if M is not None:
            out = out * (1 - M) + F * M
        if self.look != "maskb":
            out = out * (1 - ring) + rc * ring
        return out.clip(0, 255).astype(np.uint8)

    def visibility(self, f: int, mattes: Mattes) -> list[float]:
        """each letter's share left visible with this frame's matte over it (a behind look must keep every letter readable)."""
        fw, fh = self.size
        M = mattes.get(f)
        out = []
        for one in self.flat.each:
            m = self.flat.mask(fw, fh, 1.0, np.asarray(one, np.uint8))
            tot = float(m.sum())
            out.append(round(float((m * (1 - M)).sum()) / tot, 3) if tot else 1.0)
        return out


# ------------------------------------------------------------------------------------------------ the moves on a cut
def _ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def _plane(img: np.ndarray, deg: float, hinge: float = 0.0) -> np.ndarray:
    """img on a plane turned deg about the vertical axis (hinge = the plane's distance in front of the axis, in widths), perspective, on black."""
    import cv2
    h, w = img.shape[:2]
    a, d = math.radians(deg), 1.6 * w
    pts = []
    for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        z0 = hinge * w
        X, Z = x * math.cos(a) + z0 * math.sin(a), -x * math.sin(a) + z0 * math.cos(a) - hinge * w
        k = d / (d - Z)
        pts.append((w / 2 + X * k, h / 2 + y * k))
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(pts))
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _avg(frames: list[np.ndarray]) -> np.ndarray:
    return (sum(f.astype(np.float32) for f in frames) / len(frames)).astype(np.uint8)


def move_frame(kind: str, u: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """one frame of a move. u = 0..1 over the move, the source's cut at 0.5; A is the outgoing picture, B the incoming one."""
    import cv2
    h, w = A.shape[:2]
    e = _ease(u)
    if kind == "flip":
        deg = e * 180
        return _plane(A, deg) if deg < 90 else _plane(B, deg - 180)
    if kind == "cube":
        deg = e * 90
        a_face, b_face = _plane(A, -deg, 0.5), _plane(B, 90 - deg, 0.5)
        return np.where(b_face.sum(2, keepdims=True) > 0, b_face, a_face) if deg > 45 else np.where(a_face.sum(2, keepdims=True) > 0, a_face, b_face)
    if kind == "whip":
        strip = np.concatenate([A, B], axis=1)
        speed = 4 * u * (1 - u)
        out = []
        for k in range(9):
            x = int(_ease(min(1.0, max(0.0, u + (k - 4) * 0.012 * speed))) * w)
            out.append(strip[:, x:x + w])
        return _avg(out)
    src, sgn = (A, 1) if u < 0.5 else (B, -1)
    if kind == "zoom":
        v = (u / 0.5) if u < 0.5 else ((1 - u) / 0.5)
        out = []
        for k in range(7):
            z = 1 + 2.4 * (v * (1 + k * 0.05)) ** 2.4
            T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, z)
            out.append(cv2.warpAffine(src, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        return _avg(out)
    v = (u / 0.5) if u < 0.5 else ((1 - u) / 0.5)                           # roll
    out = []
    for k in range(7):
        T = cv2.getRotationMatrix2D((w / 2, h / 2), sgn * 70 * (v * (1 + k * 0.05)) ** 2, 1 + 0.5 * v)
        out.append(cv2.warpAffine(src, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return _avg(out)


# ------------------------------------------------------------------------------------------------ the whole video
def prepare(shots, segments, alpha_dir, stem, video, work, log=print, hits=None, moves=True, lines=True, lines_seed=None):
    """lines = the library's strands between the words (frame_lines; both edit modes, never inside a picture move); False = every word cuts hard as before."""
    ps = OM.probe_video(video)
    fw, fh, fps, n = ps["width"], ps["height"], ps["fps"], ps["frames"]
    words, plan = plan_words(shots, fps, n, hits, moves)
    size = (fw, fh)
    objs = [Word(w, layout(w["text"], font_path(w["font"]), fw, fh, w["tl"]), size) for w in words]
    if lines and objs:
        from tools.app import frame_lines as FL
        FL.plan(objs, fps, size, plan if moves else {}, lines_seed)
    mattes = Mattes(segments, alpha_dir, stem, size)
    return ps, objs, plan, mattes


def render_metals(objs, ps, work, comfy, input_dir, output_dir, seed: int = 1, log=print) -> None:
    """every metal word's LTX clip, one at a time (reused when the word has not changed)."""
    for o in objs:
        if o.metal:
            o.metal_clip = render_metal(o.w, o.lay, (ps["width"], ps["height"]), Path(work), comfy, Path(input_dir), Path(output_dir), ps["fps_str"], seed, log)


def compose_frames(objs, video, ps, mattes, frames=None):
    """(f, composed BGR frame) in order. frames = a set: only those are composed, the others yield None (their pictures are skipped cheaply)."""
    by = {}
    for o in objs:
        for f in range(o.w["start"], o.w["end"]):
            by[f] = o
    from tools.app import frame_lines as FL
    win = FL.index(objs)                                                      # {frame: window} of the strands between the words (empty when lines are off)
    for f, im in enumerate(OM._frames(video, ps["width"], ps["height"])):
        o = by.get(f)
        if frames is not None and f not in frames:
            if o is not None and o.metal:                                     # its clip is read in order
                o.metal_frame(f - o.w["start"])
            yield f, None
            continue
        if f in win:
            yield f, FL.frame(win[f], im, f, mattes)
            continue
        yield f, (o.over(im, f, mattes) if o else im)


def stills(objs, video, ps, mattes, plan, frames: list[int], out_dir: Path, transitions: bool = False) -> list[Path]:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for f, im in compose_frames(objs, video, ps, mattes, set(frames)):
        if im is not None:
            p = out_dir / f"f{f:05d}.jpg"
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            got.append(p)
        if frames and f >= max(frames):
            break
    return got


def write_video(objs, video, ps, mattes, plan, out: Path, transitions: bool, audio_from, dur=None, log=print) -> dict:
    """composed frames -> out (source size, rate and frame count; the source audio copied)."""
    flt, flags = OM.color_args(ps)
    w, h = ps["width"], ps["height"]
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", ps["fps_str"], "-i", "-", "-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?",
                            "-vf", flt, "-c:v", "libx264", "-crf", "15", "-preset", "medium", *flags, "-c:a", "copy", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    cuts = plan if transitions else {}
    look = max((m["h"] for m in cuts.values()), default=0)
    buf: deque = deque()
    src = compose_frames(objs, video, ps, mattes)
    done = False
    a_last, n_out = None, 0
    moved = 0

    def active(f):
        for c, m in cuts.items():
            if c - m["h"] <= f < c + m["h"]:
                return c, m
        return None, None

    while True:
        while not done and len(buf) <= look:
            try:
                buf.append(next(src))
            except StopIteration:
                done = True
        if not buf:
            break
        f, im = buf.popleft()
        c, m = active(f)
        if c is None:
            a_last = im
            enc.stdin.write(im.tobytes())
        else:
            u = (f - (c - m["h"]) + 0.5) / (2 * m["h"])
            if f < c:
                b = next((x[1] for x in buf if x[0] == c), im)
                frame = move_frame(m["kind"], u, im, b)
                a_last = im
            else:
                frame = move_frame(m["kind"], u, a_last if a_last is not None else im, im)
            enc.stdin.write(frame.tobytes())
            moved += 1
        n_out += 1
        if n_out % 100 == 0:
            log(f"{n_out} / {ps['frames']} frames")
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("the encoder failed")
    return dict(frames=n_out, moved_frames=moved, transitions=transitions)


def report(objs, mattes, plan) -> dict:
    words = []
    for o in objs:
        d = dict(n=o.w["n"], text=o.w["text"], font=o.w["font"], look=o.look, metal=o.metal, start=o.w["start"], end=o.w["end"], fs=round(o.lay["fs"], 1), rot=o.lay["rot"], s=o.lay["s"], cx=round(o.lay["cx"]), cy=round(o.lay["cy"]))
        if o.look == "behind":
            fr = [o.w["start"] + 1, (o.w["start"] + o.w["end"]) // 2, o.w["end"] - 1]
            vis = [o.visibility(f, mattes) for f in fr]
            d["letters_visible_min"] = [min(col) for col in zip(*vis)]
        words.append(d)
    from tools.app import frame_lines as FL
    return dict(words=words, moves={str(k): v for k, v in sorted(plan.items())}, lines=FL.describe(objs) if objs else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--shots", required=True, help="json: the page's story.shots (line, font, src_start, src_end, tl)")
    ap.add_argument("--segments", required=True)
    ap.add_argument("--alpha-dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="", help="write the video here")
    ap.add_argument("--transitions", type=int, default=0)
    ap.add_argument("--lines", type=int, default=1, help="1 = the library's strands carry each word to the next (frame_lines); 0 = hard cuts between the words")
    ap.add_argument("--stills", default="", help="comma list of frames: write only these into <work>/stills")
    ap.add_argument("--no-metal", type=int, default=0, help="stills: draw metal words flat (fast layout check)")
    ap.add_argument("--comfy-dir", default="/workspace/ComfyUI")
    ap.add_argument("--port", type=int, default=18188)
    ap.add_argument("--client-id", default="")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    shots = json.loads(Path(a.shots).read_text(encoding="utf-8"))
    shots = shots.get("shots", shots) if isinstance(shots, dict) else shots
    segs = json.loads(Path(a.segments).read_text(encoding="utf-8"))
    ps, objs, plan, mattes = prepare(shots, segs, a.alpha_dir, a.stem, a.video, a.work, lines=bool(a.lines), lines_seed=a.seed)
    if a.no_metal:
        for o in objs:
            o.metal = None
    rep = report(objs, mattes, plan)
    (Path(a.work)).mkdir(parents=True, exist_ok=True)
    (Path(a.work) / "frame_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["words"], ensure_ascii=False))
    if any(o.metal for o in objs):
        from tools.app.own_video_job import Comfy
        cd = Path(a.comfy_dir)
        class _J:
            log = staticmethod(print)
            state = dict(times={}, prompt_ids=[])
            def save(self): pass
            def check_cancel(self): pass
        render_metals(objs, ps, a.work, Comfy(a.port, a.client_id or "frame_render", _J()), cd / "input", cd / "output", a.seed)
    if a.stills:
        print(stills(objs, a.video, ps, mattes, plan, [int(x) for x in a.stills.split(",")], Path(a.work) / "stills"))
    if a.out:
        print(write_video(objs, a.video, ps, mattes, plan, Path(a.out), bool(a.transitions), a.video))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
