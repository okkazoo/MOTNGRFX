"""media_read.py - the READ step of the "Add media" box (16_app/MEDIA_BOX.md): the dropped media + the pasted text -> ONE contact sheet + a prompt for Gemma ->
READ JSON v1 (what to put on screen, the concept, the look, what each file shows, a window of the video, the music numbers).

    pack(media_json, input_dir)             -> (sheet RGB uint8, prompt, info dict)    the ffmpeg / PIL / DSP part, no model call
    parse(reply, info, media_json)          -> read dict (READ JSON v1)                the code guarantees; the model only proposes
    on_screen_from_text(text)               quoted words (always on screen) + `fallback_split` (no usable model reply)

Pure functions (numpy / PIL / ffmpeg / the DSP music analysis); the ComfyUI nodes are nodes_fusion/media_read.py, the graph is tools/app/media_read_graph.py.
Code guarantees of parse(): valid JSON with every key; text in "double quotes" is on_screen VERBATIM (never rewritten); every other on_screen line must occur in the
user's text (the verbatim slice of it is used, so the model cannot invent words); no user text -> on_screen [] and empty concept / look; the video window is clamped to the
video duration and has the requested length (or the whole clip when it is shorter); palette = Gemma's valid hexes else k-means on that item's own tile pixels; `music` comes
from the DSP numbers, never from the model; `video_duration` / `video_cuts` / `beat_grid` (VIDEO MODE STORYBOARD, MEDIA_BOX.md) come from code only (ffmpeg scene detect via tools/app/ingest.detect_cuts, the DSP beat grid): the cuts are cleaned inside the window (v18: every real cut is kept; only pieces under 6 frames at the clip fps merge, pieces > story.CLIP_MAX_S split evenly),
`lyrics` = Whisper's timed lines (CPU, optional, only inside the video window), and with no quoted words the model may take on_screen lines from them (never a word that is in neither the lyrics nor the text); an unparseable reply -> a rule split of the text, said so in `notes`.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np

from tools.fusion import user_style as US

MAX_VIDEO_TILES = 16
MAX_IMAGES = 5
SHEET_MAX = 1024                  # long side of the contact sheet
MAX_TILE_H = 480
GAP = 4
AUDIO_CAP_S = 120.0               # the DSP looks at the first two minutes of an audio file
KEYS = ("v", "on_screen", "concept", "look", "media", "music", "lyrics", "video_duration", "video_cuts", "beat_grid", "notes", "soundtrack")
SOUNDTRACKS = ("video", "track", "generate")      # v19 AUDIO TRACK + SOUNDTRACK: the video's own sound | the user's audio file | new music (nothing is read)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")


# ------------------------------------------------------------------------------------------------- input
def norm_soundtrack(v) -> str:
    """"video" | "track" | "generate" | "" (not given / unknown); the old "keep" is "video"."""
    s = str(v or "").strip().lower()
    return "video" if s == "keep" else (s if s in SOUNDTRACKS else "")


def parse_media(media) -> dict:
    """the page's MEDIA JSON (string or dict) -> a clean dict; limits enforced here too (1 video, <= 5 images, 1 audio). `soundtrack` (v19) picks which audio the READ analyses."""
    if isinstance(media, str):
        try:
            media = json.loads(media) if media.strip() else {}
        except ValueError:
            media = {}
    if not isinstance(media, dict):
        media = {}

    def name(v):
        v = str(v or "").strip()
        return v or None

    imgs = media.get("images") or []
    if isinstance(imgs, str):
        imgs = [s for s in imgs.split(",")]
    imgs = [n for n in (name(i) for i in imgs) if n][:MAX_IMAGES]
    try:
        length = float(media.get("length") or 12)
    except (TypeError, ValueError):
        length = 12.0
    words, direction = US.split_typed(media.get("text"))          # the typed "Words or direction" text: directive clauses (palette / legibility / plain / instructions) are the DIRECTION, never on-screen words
    return dict(video=name(media.get("video")), images=imgs, audio=name(media.get("audio")), text=words, direction=direction,
                length=max(1.0, length), format=str(media.get("format") or "9:16"), soundtrack=norm_soundtrack(media.get("soundtrack")))


def choose_audio(soundtrack: str, has_video: bool, video_has_audio: bool, has_audio_file: bool) -> tuple[str | None, list[str]]:
    """v19: which audio the READ analyses (lyrics via Whisper + beat grid / bpm / key) -> ("V1" | "A1" | None, notes).
    soundtrack "video" = the video's own audio, "track" = the audio file, "generate" = none (the music is described from look / mood); "" (not given) = the old rule: the video's audio, else the audio file.
    A pick that cannot be honoured falls back to the other source and says so in the notes; the box never fails because of it."""
    notes: list[str] = []
    st = norm_soundtrack(soundtrack)
    vid = has_video and video_has_audio
    if st == "generate":
        if vid or has_audio_file:
            notes.append("soundtrack: new music will be generated: neither the video's sound nor the audio file is read (no lyrics, no beat grid)")
        return None, notes
    if st == "track":
        if has_audio_file:
            notes.append("soundtrack: my track: lyrics and beat grid come from the audio file" + ("; the video's own sound is not read" if vid else ""))
            return "A1", notes
        notes.append("soundtrack: my track chosen but no audio file is in the box: " + ("reading the video's own audio" if vid else "no audio to read"))
        return ("V1" if vid else None), notes
    if st == "video":
        if vid:
            notes.append("soundtrack: video sound: lyrics and beat grid come from the video's own audio" + ("; the audio file is not read" if has_audio_file else ""))
            return "V1", notes
        if has_video:
            notes.append("soundtrack: the video has no audio stream: no lyrics, no beat grid")
            return None, notes
        return ("A1" if has_audio_file else None), notes          # images: there is no video sound, the audio file is the only one
    if vid:
        if has_audio_file:
            notes.append("audio file ignored for the music numbers: the video's own audio is the soundtrack")
        return "V1", notes
    return ("A1" if has_audio_file else None), notes


def resolve(name: str, input_dir) -> Path:
    """a file name from the page -> a file inside ComfyUI's input dir (sub folders ok); anything that leaves the folder is refused."""
    base = Path(input_dir).resolve()
    p = (base / name).resolve()
    if base != p and base not in p.parents:
        raise ValueError(f"media file outside the input folder: {name}")
    if not p.is_file():
        raise FileNotFoundError(f"media file not found in the input folder: {name}")
    return p


# ------------------------------------------------------------------------------------------------- ffmpeg / PIL
def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def probe_video(path, ffprobe: str = "ffprobe") -> dict:
    r = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr[-300:]}")
    j = json.loads(r.stdout)
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise ValueError(f"no video stream in {path}")
    v = vs[0]
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    try:
        rot = int((v.get("tags") or {}).get("rotate", rot))
    except ValueError:
        pass
    if abs(rot) % 180 == 90:
        w, h = h, w
    dur = None
    for src in (v.get("duration"), (j.get("format") or {}).get("duration")):
        try:
            dur = float(src)
            break
        except (TypeError, ValueError):
            continue
    if not dur or dur <= 0:
        raise ValueError(f"cannot read the duration of {path}")
    fps = 0.0
    try:
        a, b = str(v.get("avg_frame_rate", "0/1")).split("/")
        fps = float(a) / float(b) if float(b) else 0.0
    except ValueError:
        pass
    has_audio = any(s.get("codec_type") == "audio" for s in j.get("streams", []))
    return dict(duration=round(dur, 3), width=w, height=h, fps=round(fps, 3), has_audio=has_audio)


def tile_times(duration: float) -> list[float]:
    """~1 tile per second, at most 16: the centres of n equal slices of the clip."""
    n = int(min(MAX_VIDEO_TILES, max(1, round(duration))))
    return [round((i + 0.5) * duration / n, 3) for i in range(n)]


def video_frame(path, t: float, height: int = 360, ffmpeg: str = "ffmpeg") -> np.ndarray:
    from PIL import Image
    r = _run([ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale=-2:{int(height)}", "-f", "image2pipe", "-vcodec", "png", "-"])
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg could not read a frame at {t:.2f}s of {path}: {r.stderr[-300:]!r}")
    return np.asarray(Image.open(BytesIO(r.stdout)).convert("RGB"))


def load_image(path, height: int = 360) -> np.ndarray:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if im.height > height:
        im = im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)
    return np.asarray(im)


def extract_audio(src, out_wav, max_s: float = AUDIO_CAP_S, ffmpeg: str = "ffmpeg", ac: int = 2, ar: int = 44100) -> Path:
    r = _run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-vn", "-t", f"{max_s:.1f}", "-ac", str(ac), "-ar", str(ar), str(out_wav)])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not extract audio from {src}: {r.stderr[-300:]!r}")
    return Path(out_wav)


def detect_video_cuts(path, ffmpeg: str = "ffmpeg") -> list[float]:
    """the real cuts of a video (s): ffmpeg scene detect, tools/app/ingest.detect_cuts (the same detector the own-video route uses). Cheap on a phone-length clip."""
    from tools.app import ingest
    return ingest.detect_cuts(str(path), ffmpeg=ffmpeg)


def dsp(wav) -> dict:
    """the DSP part of the music reading (tools/fusion/music_analysis.analyse: beats, downbeats, key, sections); NO model call."""
    from tools.fusion import music_analysis as MA
    a = MA.analyse(str(wav))
    secs = []
    for s in a.get("sections") or []:
        if isinstance(s, dict):
            st, en = s.get("start", s.get("t0")), s.get("end", s.get("t1"))
        else:
            st, en = (list(s) + [None, None])[:2]
        try:
            secs.append([round(float(st), 1), round(float(en), 1)])
        except (TypeError, ValueError):
            continue
    key = (a.get("key") or {}).get("estimate")
    return dict(bpm=round(float(a["bpm"]), 1), key=str(key) if key else None, duration=round(float(a["duration"]), 2),
                bar_len=a.get("bar_len"), downbeats=[round(float(x), 2) for x in (a.get("downbeats") or [])[:8]], sections=secs,
                beats=[round(float(x), 2) for x in (a.get("beats") or [])], downbeats_all=[round(float(x), 2) for x in (a.get("downbeats") or [])])


# ------------------------------------------------------------------------------------------------- lyrics (Whisper, CPU only)
ASR_PYTHON_DEFAULT = "/workspace/asr_venv/bin/python"          # the separate venv with faster-whisper (BUILT.md Speeds: Whisper large-v3 lyrics)
ASR_SCRIPT = Path(__file__).resolve().parent / "asr_whisper.py"
MIN_LYRIC_WORDS = 4
MIN_WORD_P = 0.3                  # words below this probability are dropped (Whisper hallucinates words over music)
MIN_MEAN_P = 0.45                 # a lower mean = no real vocals


def asr_python() -> str | None:
    """the python of the faster-whisper venv: env MOTNGRFX_ASR_PYTHON, else /workspace/asr_venv/bin/python; None when neither exists (-> no lyrics, said in the notes)."""
    for c in (os.environ.get("MOTNGRFX_ASR_PYTHON"), ASR_PYTHON_DEFAULT):
        if c and os.path.isfile(c):
            return c
    return None


def lyrics_from_asr(j: dict) -> tuple[list[dict], str | None]:
    """the json of asr_whisper.py ({segments:[{start,end,text,words:[{w,s,e,p}]}]}) -> ([{t, end, text}] one line per Whisper segment, note). Words under MIN_WORD_P are dropped;
    fewer than MIN_LYRIC_WORDS words left or a mean probability under MIN_MEAN_P = no vocals (-> [], note)."""
    lines, probs = [], []
    for sg in (j or {}).get("segments") or []:
        ws = []
        for w in sg.get("words") or []:
            pr = _num(w.get("p"))
            pr = 1.0 if pr is None else pr
            if str(w.get("w", "")).strip() and pr >= MIN_WORD_P:
                ws.append((w, pr))
        if not ws:
            continue
        probs += [pr for _, pr in ws]
        text = " ".join(str(w["w"]).strip() for w, _ in ws)
        ln = dict(t=round(float(ws[0][0].get("s", sg.get("start", 0.0))), 2), end=round(float(ws[-1][0].get("e", sg.get("end", 0.0))), 2), text=text)
        if len(text.split()) == len(ws):             # the sung start of every word (the storyboard's text tokens carry it)
            ln["ts"] = [round(float(w.get("s", ln["t"])), 2) for w, _ in ws]
            ln["words"] = [dict(text=str(w["w"]).strip(), t=round(float(w.get("s", ln["t"])), 2), end=round(float(w.get("e", w.get("s", ln["t"]))), 2), p=round(pr, 2)) for w, pr in ws]       # WORD PICK: start / end / probability of every word (the page and story.word_pick use them)
        lines.append(ln)
    n = len(probs)
    if n < MIN_LYRIC_WORDS:
        return [], f"no vocals found (Whisper kept {n} words): no lyrics"
    mean_p = sum(probs) / n
    if mean_p < MIN_MEAN_P:
        return [], f"no clear vocals (Whisper mean word probability {mean_p:.2f}): no lyrics"
    return lines, None


def transcribe(wav, timeout: float = 600.0) -> tuple[list[dict], str | None]:
    """timed lyrics of a wav: Whisper large-v3 int8 on the CPU in the separate venv (tools/fusion/asr_whisper.py, CUDA hidden so it never takes VRAM from Gemma) -> (lines, note).
    Any problem (no venv, error, timeout, no vocals) returns ([], note): the READ step goes on without lyrics."""
    py = asr_python()
    if py is None:
        return [], "lyrics skipped: Whisper venv not found (MOTNGRFX_ASR_PYTHON / /workspace/asr_venv)"
    out = str(Path(wav).with_suffix(".asr.json"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    try:
        r = subprocess.run([py, str(ASR_SCRIPT), str(wav), out], capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0:
            tail = [ln.strip() for ln in (r.stderr or r.stdout or "").splitlines() if ln.strip()]
            return [], f"lyrics skipped: Whisper failed ({(tail[-1] if tail else 'no output')[:200]})"
        return lyrics_from_asr(json.loads(Path(out).read_text(encoding="utf-8")))
    except subprocess.TimeoutExpired:
        return [], f"lyrics skipped: Whisper timed out after {timeout:.0f} s"
    except Exception as e:       # noqa: BLE001
        return [], f"lyrics skipped: {type(e).__name__}: {e}"


def lyric_out(l: dict, shift: float = 0.0) -> dict:
    """one lyric line of the Whisper pass -> the READ's line, put on the video's timeline (+ shift): t, text and, when the line has them, the per-word starts `ts`, the line `end` and the per-word
    detail `words` [{text, t, end, p}] (WORD PICK: held notes and exact sung times)."""
    n = len(str(l["text"]).split())
    d = dict(t=float(l["t"]) + shift, text=str(l["text"]))
    if l.get("ts") and len(l["ts"]) == n:
        d["ts"] = [float(x) + shift for x in l["ts"]]
    if _num(l.get("end")) is not None:
        d["end"] = float(l["end"]) + shift
    ws = l.get("words")
    if isinstance(ws, list) and len(ws) == n and all(isinstance(w, dict) and _num(w.get("t")) is not None for w in ws):
        d["words"] = [dict(text=str(w.get("text", "")), t=float(w["t"]) + shift, end=(float(w["end"]) + shift) if _num(w.get("end")) is not None else float(w["t"]) + shift,
                           p=(float(w["p"]) if _num(w.get("p")) is not None else 1.0)) for w in ws]
    return d


def lyrics_text(lyrics: list[dict]) -> str:
    return "\n".join(str(l.get("text", "")) for l in lyrics or [])


def fmt_lyrics(lyrics: list[dict]) -> str:
    return "\n".join(f"[{float(l['t']):.1f}s] {l['text']}" for l in lyrics)


# ------------------------------------------------------------------------------------------------- palette
def kmeans_palette(rgb_images, k: int = 4) -> list[str]:
    """dominant colours of a set of RGB uint8 images: k-means (deterministic: farthest-point init) on <= ~2.5k pixels; most common first, near-duplicates merged."""
    px = []
    for im in rgb_images:
        a = np.asarray(im)
        if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] < 1:
            continue
        sy, sx = max(1, a.shape[0] // 24), max(1, a.shape[1] // 24)
        px.append(a[::sy, ::sx, :3].reshape(-1, 3))
    if not px:
        return []
    x = np.concatenate(px).astype(np.float64)
    if len(x) > 2500:
        x = x[np.linspace(0, len(x) - 1, 2500).astype(int)]
    k = max(1, min(k, len(np.unique(x, axis=0))))
    c = [x.mean(0)]
    while len(c) < k:
        d = np.min([((x - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(x[int(d.argmax())])
    c = np.array(c)
    for _ in range(14):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        nc = np.array([x[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(nc, c, atol=0.5):
            c = nc
            break
        c = nc
    lab = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
    order = np.argsort(-np.bincount(lab, minlength=k))
    out: list[np.ndarray] = []
    for i in order:
        if (lab == i).sum() == 0:
            continue
        if all(np.linalg.norm(c[i] - o) > 28 for o in out):
            out.append(c[i])
    return [rgb_hex(o) for o in out]


def rgb_hex(c) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in c[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


_HEX = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def clean_palette(p) -> list[str]:
    """Gemma's palette -> '#rrggbb' list (3-digit hex expanded, dupes dropped); anything else is dropped; max 6."""
    if isinstance(p, str):
        p = re.split(r"[,\s]+", p.strip())
    out = []
    for v in p if isinstance(p, list) else []:
        m = _HEX.match(str(v).strip())
        if not m:
            continue
        h = m.group(1).lower()
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        h = "#" + h
        if h not in out:
            out.append(h)
    return out[:6]


# ------------------------------------------------------------------------------------------------- the sheet
def _font(size: int):
    from PIL import ImageFont
    try:
        from tools.fusion import music_read as MR
        fp = MR._font_path()
        if fp:
            return ImageFont.truetype(fp, size)
    except Exception:       # noqa: BLE001
        pass
    raise RuntimeError("sheet label font missing: SpaceGrotesk-Bold (liked.json); liked fonts only, no generic fallback")


def layout(nv: int, av: float, ni: int, ai: float, limit: int = SHEET_MAX):
    """cols of the video block / the image block that give the largest tiles inside limit x limit -> (cols_v, cols_i, cell_h)."""
    best = None
    for cv in (range(1, nv + 1) if nv else [0]):
        for ci in (range(1, ni + 1) if ni else [0]):
            rows = (math.ceil(nv / cv) if nv else 0) + (math.ceil(ni / ci) if ni else 0)
            wv = cv * av + (cv - 1) * GAP / 100.0 if cv else 0
            wi = ci * ai + (ci - 1) * GAP / 100.0 if ci else 0
            w, h = max(wv, wi), rows
            if w <= 0 or h <= 0:
                continue
            s = min((limit - GAP * (max(cv, ci) - 1)) / w, (limit - GAP * (rows - 1)) / h, MAX_TILE_H)
            if best is None or s > best[0]:
                best = (s, cv, ci)
    s, cv, ci = best
    return cv, ci, max(24, int(s))


def _tile(img: np.ndarray, cw: int, ch: int, label: str, font) -> np.ndarray:
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    sc = min(cw / w, ch / h)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    im = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
    cell = Image.new("RGB", (cw, ch), (20, 20, 20))
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    cell.paste(im, (ox, oy))
    d = ImageDraw.Draw(cell)
    tb = d.textbbox((0, 0), label, font=font)
    pad = 3
    x0, y0 = ox + 2, oy + 2
    d.rectangle([x0, y0, x0 + tb[2] - tb[0] + 2 * pad, y0 + tb[3] - tb[1] + 2 * pad], fill=(0, 0, 0))
    d.text((x0 + pad - tb[0], y0 + pad - tb[1]), label, fill=(255, 235, 60), font=font)
    return np.asarray(cell)


def contact_sheet(video_tiles: list[tuple[str, np.ndarray]], image_tiles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """ONE image: the video tiles in a grid (reading order, labelled 'V1 3.0s'), the images in a row block under it (labelled 'I1'..); long side <= 1024."""
    if not video_tiles and not image_tiles:
        return placeholder()
    av = video_tiles[0][1].shape[1] / video_tiles[0][1].shape[0] if video_tiles else 1.0
    ai = float(np.clip(np.mean([t[1].shape[1] / t[1].shape[0] for t in image_tiles]), 0.5, 1.8)) if image_tiles else 1.0
    cv, ci, ch = layout(len(video_tiles), av, len(image_tiles), ai)
    font = _font(max(11, ch // 11))
    blocks = []
    for tiles, cols, asp in ((video_tiles, cv, av), (image_tiles, ci, ai)):
        if not tiles:
            continue
        cw = max(8, int(round(ch * asp)))
        rows = math.ceil(len(tiles) / cols)
        W, H = cols * cw + (cols - 1) * GAP, rows * ch + (rows - 1) * GAP
        blk = np.full((H, W, 3), 60, np.uint8)
        for i, (lab, im) in enumerate(tiles):
            r, c = divmod(i, cols)
            blk[r * (ch + GAP):r * (ch + GAP) + ch, c * (cw + GAP):c * (cw + GAP) + cw] = _tile(im, cw, ch, lab, font)
        blocks.append(blk)
    W = max(b.shape[1] for b in blocks)
    H = sum(b.shape[0] for b in blocks) + GAP * (len(blocks) - 1)
    sheet = np.full((H, W, 3), 60, np.uint8)
    y = 0
    for b in blocks:
        x = (W - b.shape[1]) // 2
        sheet[y:y + b.shape[0], x:x + b.shape[1]] = b
        y += b.shape[0] + GAP
    return sheet


def placeholder() -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (320, 120), (20, 20, 20))
    ImageDraw.Draw(im).text((16, 50), "NO VISUAL MEDIA", fill=(255, 235, 60), font=_font(22))
    return np.asarray(im)


# ------------------------------------------------------------------------------------------------- prompt
READ_SPEC = """{
  "on_screen": ["text lines meant to APPEAR on screen, in order, each copied letter for letter from the user's text"],
  "concept": "one or two sentences: what the piece is about, taken from the user's text (empty string if the user gave none)",
  "look": "the user's instructions about feel / look / pace / palette / camera (empty string if none)",
  "media": [{"id": "V1", "shows": "what this file shows, one sentence", "window": [start_s, end_s], "mood": "two or three words", "palette": ["#rrggbb", "#rrggbb", "#rrggbb"]}],
  "notes": "anything you could not place, else an empty string"
}"""


def fmt_music(m: dict | None) -> str:
    if not m:
        return ""
    secs = m.get("sections") or []
    s = f"{m.get('duration', 0):.1f} s, about {m['bpm']:.0f} bpm, key {m.get('key') or 'unknown'}"
    if secs:
        s += f", {len(secs)} section{'s' if len(secs) != 1 else ''} (" + ", ".join(f"{a:g}-{b:g} s" for a, b in secs[:8]) + ")"
    return s


def build_prompt(med: dict, items: list[dict], music: dict | None, audio_from: str | None, lyrics: list[dict] | None = None) -> str:
    L = ["You read the media a user dropped into a motion-graphic app BEFORE the piece is designed, and you split their pasted text into what is shown on screen and what is direction.",
         "You see ONE contact sheet image of the media (no other pictures)."]
    L.append("")
    L.append("THE CONTACT SHEET")
    for it in items:
        if it["kind"] == "video":
            L.append(f"- Tiles labelled V1 with a time (e.g. 'V1 3.0s') are frames of the VIDEO '{it['file']}' ({it['duration']:.1f} s long, {it['w']}x{it['h']}), "
                     f"{len(it['tiles'])} frames in time order, one about every {it['duration'] / max(1, len(it['tiles'])):.1f} s; the label is the second the frame comes from.")
        elif it["kind"] == "image":
            L.append(f"- The tile labelled {it['id']} is the IMAGE '{it['file']}' ({it['w']}x{it['h']}).")
    if not any(i["kind"] in ("video", "image") for i in items):
        L.append("- There is no picture: the sheet is a blank placeholder. The media is audio only.")
    for it in items:
        if it["kind"] == "audio":
            L.append(f"- {it['id']} is the AUDIO file '{it['file']}' ({it['duration']:.1f} s); it is not on the sheet.")
    if music:
        L.append("")
        src = "the video's own audio" if audio_from == "V1" else "the audio file"
        L.append(f"MUSIC (measured by software from {src}, not by you; you do not write any music field): {fmt_music(music)}.")
    if lyrics:
        src = "the video's own audio" if audio_from == "V1" else "the audio file"
        L += ["", f"SUNG LYRICS (transcribed by Whisper from {src}; times are seconds in the source file; the transcription may have small errors):", fmt_lyrics(lyrics)]
    L += ["", f"REQUESTED PIECE: {med['length']:g} seconds long, format {med['format']}.", "", "THE USER'S PASTED TEXT, exactly as typed, between the markers:", "<<<TEXT"]
    L.append(med["text"] if med["text"].strip() else "(the user pasted no text)")
    L += [">>>TEXT", ""]
    if (med.get("direction") or "").strip():
        L += ["THE USER'S DIRECTION, split from the text by software (instructions about palette / legibility / feel, NOT words to show): it is never on_screen and is never copied into on_screen or concept; use it only for `look`.",
              "<<<DIRECTION", med["direction"], ">>>DIRECTION", ""]
    L += ["YOUR JOB",
          "1. Split the user's text. on_screen = the words / lyric lines the user wants SEEN as type in the piece; copy each letter for letter from the text above, in order, never reword, never add a word the user did not write. "
          "Text inside \"double quotes\" is ALWAYS on_screen. concept = what the piece is about. look = instructions about the feel, look, pace, colour, camera. "
          "If the text has no words to display, on_screen is []" + (", unless there are sung lyrics above and the user quoted no words: then choose 3 to 8 prominent sung words or short lines from the lyrics (copied letter for letter, in the order they are sung) as on_screen; each one a sensible standalone phrase of at most 4 words (a complete short line, or cut at a natural break) that NEVER ends on a small word such as on / the / in / of / and / to / with, and never starts mid-sentence" if lyrics else "")
          + ". If the user gave no concept or no look, those are empty strings; do not invent them.",
          "2. media: one entry per file (" + ", ".join(i["id"] for i in items if i["kind"] != "audio") + (", " if any(i["kind"] == "audio" for i in items) and any(i["kind"] != "audio" for i in items) else "")
          + ", ".join(i["id"] for i in items if i["kind"] == "audio") + "): shows = what is in it, mood, palette = 3 to 5 dominant colours as #rrggbb read from the picture.",
          ]
    vid = next((i for i in items if i["kind"] == "video"), None)
    if vid:
        ln = min(med["length"], vid["duration"])
        L.append(f"3. For V1 give window = [start_s, end_s] in seconds inside 0..{vid['duration']:.1f}: the most promising stretch of the video for the piece, exactly {ln:g} s long "
                 f"(end_s - start_s = {ln:g}); use the tile times to choose, and prefer a stretch that fits the user's look / concept text.")
    L += ["", "Reply with ONE JSON object and nothing else, in exactly this shape:", READ_SPEC]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- pack
def _audio_pass(src, wav_path: Path, ffmpeg: str, run_dsp: bool, run_asr: bool, notes: list[str], what: str):
    """extract the audio once -> (music_detail | None, lyrics lines); every failure is a note, never an exception."""
    music_detail, lyrics = None, []
    try:
        wav = extract_audio(src, wav_path, ffmpeg=ffmpeg)
    except Exception as e:       # noqa: BLE001
        notes.append(f"{what}: audio extraction failed: {type(e).__name__}: {e}")
        return None, []
    if run_dsp:
        try:
            music_detail = dsp(wav)
        except Exception as e:       # noqa: BLE001
            notes.append(f"music analysis of {what} failed: {type(e).__name__}: {e}")
    if run_asr:
        try:
            wav16 = extract_audio(src, Path(wav_path).with_name(Path(wav_path).stem + "_16k.wav"), ffmpeg=ffmpeg, ac=1, ar=16000)       # Whisper's own format: 16 kHz mono
        except Exception as e:       # noqa: BLE001
            notes.append(f"lyrics skipped: 16 kHz extraction of {what} failed ({type(e).__name__})")
            return music_detail, []
        lyrics, note = transcribe(wav16)
        if note:
            notes.append(note)
    return music_detail, lyrics


def pack(media, input_dir, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", run_dsp: bool = True, run_asr: bool = True) -> tuple[np.ndarray, str, dict]:
    """MEDIA JSON + the ComfyUI input dir -> (sheet RGB uint8, prompt, info). info carries what parse() needs (durations, ids, files, pixel palettes, the music numbers)."""
    t0 = time.time()
    med = parse_media(media)
    items: list[dict] = []
    notes: list[str] = []
    vtiles: list[tuple[str, np.ndarray]] = []
    itiles: list[tuple[str, np.ndarray]] = []
    music, audio_from, music_detail, lyrics = None, None, None, []
    v_audio = False
    tmp = Path(tempfile.mkdtemp(prefix="mediaread_"))
    try:
        if med["video"]:
            p = resolve(med["video"], input_dir)
            pr = probe_video(p, ffprobe)
            times = tile_times(pr["duration"])
            frames = [video_frame(p, t, ffmpeg=ffmpeg) for t in times]
            vtiles = [(f"V1 {t:.1f}s", f) for t, f in zip(times, frames)]
            items.append(dict(id="V1", kind="video", file=med["video"], duration=pr["duration"], w=pr["width"], h=pr["height"], fps=pr["fps"], has_audio=pr["has_audio"],
                              tiles=times, palette_px=kmeans_palette(frames)))
            try:
                items[-1]["cuts"] = [float(c) for c in detect_video_cuts(p, ffmpeg)]
            except Exception as e:       # noqa: BLE001
                notes.append(f"video cuts: scene detect failed ({type(e).__name__}: {e}); the storyboard splits the clip evenly")
            v_audio = pr["has_audio"]
        for k, name in enumerate(med["images"], 1):
            img = load_image(resolve(name, input_dir))
            itiles.append((f"I{k}", img))
            items.append(dict(id=f"I{k}", kind="image", file=name, w=int(img.shape[1]), h=int(img.shape[0]), palette_px=kmeans_palette([img])))
        if med["audio"]:
            p = resolve(med["audio"], input_dir)
            dur = None
            r = _run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], text=True)
            try:
                dur = float(r.stdout.strip())
            except ValueError:
                notes.append("audio duration unreadable")
            items.append(dict(id="A1", kind="audio", file=med["audio"], duration=round(dur or 0.0, 2)))
        audio_from, pick_notes = choose_audio(med["soundtrack"], bool(med["video"]), v_audio, bool(med["audio"]))       # v19: the soundtrack picks which audio is read
        notes += pick_notes
        if audio_from and not (run_dsp or run_asr):
            audio_from = None
        elif audio_from == "V1":
            music_detail, lyrics = _audio_pass(resolve(med["video"], input_dir), tmp / "v1.wav", ffmpeg, run_dsp, run_asr, notes, "the video's audio")
        elif audio_from == "A1":
            music_detail, lyrics = _audio_pass(resolve(med["audio"], input_dir), tmp / "a1.wav", ffmpeg, run_dsp, run_asr, notes, "the audio file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if music_detail:
        music = {"bpm": music_detail["bpm"], "key": music_detail["key"]}
    sheet = contact_sheet(vtiles, itiles)
    prompt = build_prompt(med, items, music_detail, audio_from, lyrics)
    info = dict(v=1, length=med["length"], format=med["format"], text=med["text"], items=items, music=music, music_detail=music_detail, audio_from=audio_from, lyrics=lyrics, notes=notes, soundtrack=med["soundtrack"],
                sheet=dict(w=int(sheet.shape[1]), h=int(sheet.shape[0])), pack_s=round(time.time() - t0, 2))
    return sheet, prompt, info


# ------------------------------------------------------------------------------------------------- text split
_QUOTE = re.compile(r'"([^"]{1,600}?)"|“([^”]{1,600}?)”')


def quoted_spans(text: str) -> list[tuple[int, int, str]]:
    """[(start, end, inner text)] of every "double quoted" run of the text, in order (straight and curly quotes)."""
    out = []
    for m in _QUOTE.finditer(text or ""):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if inner.strip():
            out.append((m.start(), m.end(), inner))
    return out


def _quoted_lines(text: str) -> list[str]:
    out = []
    for _, _, inner in quoted_spans(text):
        out += [ln.strip() for ln in inner.splitlines() if ln.strip()]
    return out


def _norm_map(text: str):
    """lower-case alphanumerics of the text and, per character, its index in the original."""
    chars, idx = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            idx.append(i)
    return "".join(chars), idx


def _find_verbatim(line: str, text: str, nmap) -> tuple[int, int] | None:
    nt, idx = nmap
    nl = "".join(ch.lower() for ch in line if ch.isalnum())
    if not nl:
        return None
    j = nt.find(nl)
    if j < 0:
        return None
    return idx[j], idx[j + len(nl) - 1] + 1


MAX_LYRIC_LINES = 8


def _slice_lines(src: str, sp: tuple[int, int]) -> list[tuple[int, str]]:
    seg = src[sp[0]:sp[1]]
    if "\n" in seg:
        return [(sp[0] + seg.find(p.strip()), p.strip()) for p in seg.splitlines() if p.strip()]
    return [(sp[0], seg.strip())]


def merge_on_screen(text: str, model_lines, lyrics: str = "") -> tuple[list[str], list[str]]:
    """-> (on_screen, notes). Quoted runs verbatim + the model's lines that are found in the user's text (the verbatim slice is used), in the order of the text; a model line that
    overlaps a quoted run is dropped (the quoted run covers it). Lines found only in the sung `lyrics` (plain text, one line per row) are allowed when the user quoted no words (<= 8,
    in the order sung, after the text lines); everything else the model wrote is dropped, so nothing is invented."""
    notes: list[str] = []
    text = text or ""
    ml = model_lines if isinstance(model_lines, list) else ([model_lines] if isinstance(model_lines, str) and model_lines.strip() else [])
    q = quoted_spans(text)
    cands: list[tuple[int, int, str]] = []         # (start, end, text) in the user's text
    lcands: list[tuple[int, int, str]] = []        # in the lyrics
    for s, e, inner in q:
        pos = s + 1
        for ln in inner.splitlines():
            if ln.strip():
                k = text.find(ln.strip(), pos)
                cands.append((k, k + len(ln.strip()), ln.strip()))
                pos = max(pos, k + len(ln.strip()))
    tmap = _norm_map(text) if text.strip() else None
    lmap = _norm_map(lyrics) if (lyrics or "").strip() and not q else None
    dropped = 0
    for raw in ml:
        raw = str(raw).strip()
        if not raw:
            continue
        sp = _find_verbatim(raw, text, tmap) if tmap else None
        if sp is not None:
            if any(sp[0] < qe and sp[1] > qs for qs, qe, _ in q):
                continue
            for a, p in _slice_lines(text, sp):
                cands.append((a, a + len(p), p))
            continue
        lp = _find_verbatim(raw, lyrics, lmap) if lmap else None
        if lp is not None:
            for a, p in _slice_lines(lyrics, lp):
                lcands.append((a, a + len(p), p))
            continue
        dropped += 1
    if dropped:
        notes.append(f"{dropped} on_screen line(s) of the model were not in the user's text" + (" or the lyrics" if (lyrics or "").strip() else "") + " and were dropped")
    out, seen = [], set()
    for group, cap in ((sorted(cands, key=lambda c: c[0]), None), (sorted(lcands, key=lambda c: c[0]), MAX_LYRIC_LINES)):
        n = 0
        for _, _, p in group:
            key = "".join(ch.lower() for ch in p if ch.isalnum())
            if key and key not in seen and (cap is None or n < cap):
                seen.add(key)
                out.append(p)
                n += cap is not None
    if lcands and out:
        notes.append("on_screen words taken from the sung lyrics (the user gave no display text)")
    return out, notes


def fallback_split(text: str) -> tuple[list[str], str]:
    """no usable model reply -> (on_screen, concept). Quoted runs are on_screen, the rest of the text is the concept. With no quotes: runs of >= 2 consecutive short lines
    (<= 6 words, no comma / full stop inside) are lyric-like and go to on_screen, every other line goes to the concept."""
    text = text or ""
    q = quoted_spans(text)
    if q:
        rest = text
        for s, e, _ in reversed(q):
            rest = rest[:s] + " " + rest[e:]
        rest = re.sub(r"\s+", " ", rest).strip()
        return _quoted_lines(text), rest
    lines = text.splitlines()
    short = [bool(ln.strip()) and len(ln.split()) <= 6 and not re.search(r"[,.;:]", ln.strip().rstrip(".!?")) for ln in lines]
    on, rest = [], []
    i = 0
    while i < len(lines):
        if short[i]:
            j = i
            while j < len(lines) and (short[j] or not lines[j].strip()):
                j += 1
            run = [ln.strip() for ln, s in zip(lines[i:j], short[i:j]) if s]
            if len(run) >= 2:
                on += run
            else:
                rest += run
            i = j
        else:
            if lines[i].strip():
                rest.append(lines[i].strip())
            i += 1
    return on, " ".join(rest)


# ------------------------------------------------------------------------------------------------- parse
def _str(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def clamp_window(w, duration: float, length: float) -> tuple[list[float], str | None]:
    """the model's window -> [start, end] inside the clip with length min(length, duration): the start is kept (moved down when it would run past the end)."""
    ln = min(float(length), float(duration))
    note = None
    a = None
    if isinstance(w, (list, tuple)) and len(w) >= 2:
        a, b = _num(w[0]), _num(w[1])
        if a is None or b is None:
            a = None
    if a is None:
        a, note = 0.0, "no usable window from the model: starts at 0"
    start = min(max(0.0, a), max(0.0, duration - ln))
    start = round(start, 1)
    return [start, round(min(duration, start + ln), 1)], note


def parse(reply: str, info: dict, media) -> dict:
    """the model reply + info from pack() + the MEDIA JSON -> READ JSON v1 (dict with every key). See the module docstring for the guarantees."""
    med = parse_media(media)
    text, direction = med["text"], med.get("direction") or ""
    obj, pnotes = (None, [])
    if isinstance(reply, str) and reply.strip():
        from tools.fusion import music_read as MR
        obj, pnotes = MR.parse_reply(reply)
    else:
        pnotes = ["empty reply"]
    notes: list[str] = []
    if obj is None:
        on, concept = fallback_split(text)
        notes.append("MODEL REPLY NOT USABLE (" + "; ".join(pnotes) + "): on_screen / concept come from a rule split of the text (quoted runs, short line-broken lines), look and the per-file notes are empty")
        look, model_media = direction, []
        obj_notes = ""
    else:
        if pnotes:
            notes.append("reply: " + "; ".join(pnotes))
        on, n2 = merge_on_screen(text, obj.get("on_screen"), lyrics_text((info or {}).get("lyrics")))
        notes += n2
        concept, look = _str(obj.get("concept")), _str(obj.get("look"))
        model_media = obj.get("media") if isinstance(obj.get("media"), list) else []
        obj_notes = _str(obj.get("notes"))
    if not text.strip():
        concept = ""
        look = look if direction.strip() else ""            # direction only: the model's look (read from the direction) stays, and there is no concept
    if direction.strip() and not look:
        look = " ".join(direction.split())[:300]
    # on_screen stays what the lyrics gave (merge_on_screen); with no text and no lyrics it is []
    mm = [m for m in model_media if isinstance(m, dict)]
    by_id = {str(m.get("id", "")).strip().upper(): m for m in mm}   # Gemma sometimes uses the file name as the id (dev read 2026-10-30)
    by_id.update({str(m.get(k, "")).strip().upper(): m for m in mm for k in ("file", "id") if str(m.get(k, "")).strip()})
    media_out = []
    for n, it in enumerate((info or {}).get("items", [])):
        m = by_id.get(it["id"]) or by_id.get(str(it["file"]).upper()) or (mm[n] if n < len(mm) and len(mm) == len(info["items"]) else {})
        e = dict(id=it["id"], kind=it["kind"], file=it["file"], shows=_str(m.get("shows")), window=None, mood=_str(m.get("mood")), palette=[])
        if it["kind"] == "video":
            e["window"], wn = clamp_window(m.get("window"), float(it["duration"]), float(med["length"]))
            if wn:
                notes.append(f"V1: {wn}")
            if "has_audio" in it:
                e["has_audio"] = bool(it["has_audio"])          # v19: the page greys "Video sound" out for a clip with no audio stream
        pal = clean_palette(m.get("palette"))
        if len(pal) < 2:
            if pal:
                notes.append(f"{it['id']}: palette of the model too short")
            pal = it.get("palette_px") or []
            if pal and it["kind"] != "audio":
                notes.append(f"{it['id']}: palette from the pixels")
        e["palette"] = pal if it["kind"] != "audio" else []
        if it["kind"] == "audio":
            md = (info or {}).get("music_detail") if (info or {}).get("audio_from") == "A1" else None
            e["shows"] = e["shows"] or ("audio, " + fmt_music(md) if md else f"audio, {it.get('duration', 0):.1f} s")
        media_out.append(e)
    afrom = (info or {}).get("audio_from")
    vwin = next((m["window"] for m in media_out if m["kind"] == "video" and m["window"]), None)
    shift = 0.0
    if afrom == "V1":
        win = vwin
    elif afrom == "A1" and vwin:
        # v19: the user's audio file replaces the clip's sound and plays from ITS start over the window: the lyrics inside 0..window length are kept and put on the video's timeline
        # (track second + window start) so that the storyboard, which compares sung time with the video's cuts, lands them where they are heard
        win, shift = [0.0, float(vwin[1]) - float(vwin[0])], float(vwin[0])
    else:
        win = None
    lyr = [lyric_out(l, shift) for l in (info or {}).get("lyrics") or [] if win is None or win[0] - 0.01 <= float(l["t"]) <= win[1]]
    vid = next((m for m in media_out if m["kind"] == "video"), None)
    vit = next((i for i in (info or {}).get("items", []) if i.get("kind") == "video"), None)
    video_duration, video_cuts, grid = None, [], None
    if vid and vit:
        video_duration = round(float(vit["duration"]), 2)
        w0, w1 = vid["window"] if vid.get("window") else (0.0, video_duration)
        from tools.fusion import story as ST
        video_cuts = [dict(start=round(a, 2), end=round(b, 2)) for a, b in ST.video_pieces(vit.get("cuts") or [], w0, w1, fps=vit.get("fps"))]
        md = (info or {}).get("music_detail") if afrom in ("V1", "A1") else None
        if md and md.get("beats"):                 # window-relative (the render analyses the window's own audio from 0): what slots_from_analysis reads
            if afrom == "V1":
                rel = lambda xs: [round(float(x) - w0, 2) for x in xs if w0 - 0.01 <= float(x) <= w1 + 0.01]       # noqa: E731
                grid = dict(bpm=md.get("bpm"), t0=round(w0, 2), beats=rel(md["beats"]), downbeats=rel(md.get("downbeats_all") or []))
            else:                                  # v19 "my track": the render lays the track over the window from the track's own start, so the grid is the track's first (w1 - w0) seconds, unshifted
                span = w1 - w0
                keep = lambda xs: [round(float(x), 2) for x in xs if -0.01 <= float(x) <= span + 0.01]       # noqa: E731
                grid = dict(bpm=md.get("bpm"), t0=0.0, beats=keep(md["beats"]), downbeats=keep(md.get("downbeats_all") or []))
    notes = list((info or {}).get("notes") or []) + notes          # the pack step's notes (Whisper skipped / no vocals / ignored audio file ...)
    if obj_notes:
        notes.insert(0, obj_notes)
    return dict(v=1, on_screen=on, concept=concept, look=look, media=media_out, music=(info or {}).get("music"), lyrics=lyr, video_duration=video_duration, video_cuts=video_cuts, beat_grid=grid, notes=" | ".join(notes),
                soundtrack=(info or {}).get("soundtrack") or {"V1": "video", "A1": "track"}.get(afrom, ""))
