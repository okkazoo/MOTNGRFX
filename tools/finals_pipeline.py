#!/usr/bin/env python
"""Finished-piece pipeline: plan builder -> CPU music stem -> ACE-Step a2a -> onset snap -> SFX -> mix -> render.

    python tools/finals_pipeline.py stem   PIECE.py --work DIR            # CPU music stem (nominal beat grid)
    python tools/finals_pipeline.py ace    PIECE.py --work DIR --idea P/I/vNNN   # scp stem, a2a via progress_standin
    python tools/finals_pipeline.py render PIECE.py --work DIR --out DIR --short 540 [--fmt both] [--sheet DIR]
    python tools/finals_pipeline.py preview IN.mp4 OUT.mp4                # 720 short side, <= 8 MB

A PIECE.py module defines:
    ID = "F8"; NAME = "F8_S4-type-hectic"; DUR = 10.0; BPM = 140.0; KEY = "E"; SCALE = "minor"; MOOD = "driving"
    ENERGY = 0.95; DRUMS = True; SEED = 8
    ACE = {"tags": "...", "keyscale": "E minor", "denoise": 0.5}
    IMAGES = {"img0": "path relative to repo or absolute"}; CUTOUTS = {"img0"}   (optional)
    def build(beats: list[float]) -> plan dict   # every time is placed via the beat list (see Grid)
    TAPE_STOP = None | seconds | fn(Grid)        # optional: music pitches down to a stop from this time
    MUSIC_ENV = None | [(t, dB)] | fn(Grid)      # optional: gain automation on the music (rests, swells)

Sound follows _info/decisions.md: the music stem only goes through ACE-Step a2a (denoise by energy); the plan is
then rebuilt on the beat grid snapped to the ACE output's onsets, so cuts and hits land on the final audio; the
CPU SFX (plan.audio.sfx + risers, rendered from the rebuilt plan) are mixed on top and the mix is normalised to
-14 LUFS. Steps and their wall / ComfyUI execution seconds are appended to DIR/<ID>_timings.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workflows", "api"))

from mograph import geometry, mix, synth  # noqa: E402
from mograph.analyze import audio_envelope  # noqa: E402
from mograph.encode import encode_mp4  # noqa: E402
from mograph.plan import validate_plan  # noqa: E402
from mograph.render import render_frames  # noqa: E402

SR = 48000
FMTS = {"9x16": "9:16", "16x9": "16:9"}
STANDIN = os.environ.get("PROGRESS_STANDIN") or os.path.join(os.path.expanduser("~"), "Documents/Dev/comfyui-projects/tools/progress_standin.py")
BOX = {"host": os.environ.get("MOGRAPH_BOX_HOST", "root@<box-ip>"),
       "port": os.environ.get("MOGRAPH_BOX_PORT", "15314"),
       "dir": "/workspace/ComfyUI/input/mograph/finals/"}
IID = os.environ.get("MOGRAPH_IID", "57178352")
PORT = int(os.environ.get("MOGRAPH_PORT", "9000"))


# ----------------------------------------------------------------------------- piece + grid
def load_piece(path: str):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def nominal_beats(bpm: float, dur: float, phase: float = 0.0) -> list[float]:
    B = 60.0 / bpm
    n = int((dur - phase) / B + 1e-6) + 1
    return [round(phase + k * B, 4) for k in range(n)]


class Grid:
    """Beat-index -> seconds over a (possibly snapped) beat list; fractional and out-of-range indices
    interpolate / extrapolate at the local beat length."""

    def __init__(self, beats, dur):
        self.t = np.asarray(beats, np.float64)
        self.dur = float(dur)
        self.B = float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.5

    def __call__(self, k: float) -> float:
        t, n = self.t, len(self.t)
        if k <= 0:
            v = t[0] + k * (t[1] - t[0] if n > 1 else self.B)
        elif k >= n - 1:
            v = t[-1] + (k - (n - 1)) * (t[-1] - t[-2] if n > 1 else self.B)
        else:
            i = int(np.floor(k))
            v = t[i] + (k - i) * (t[i + 1] - t[i])
        return round(float(min(max(v, 0.0), self.dur)), 4)


# ----------------------------------------------------------------------------- audio helpers
def write_wav(path: str, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    wavfile.write(path, SR, pcm.T.copy())


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def music_stem(plan: dict, drums: bool = True) -> np.ndarray:
    if not drums:
        kit = synth._kit

        def silent_kit(sr_, mood, rng):
            return {k: np.zeros_like(v) for k, v in kit(sr_, mood, rng).items()}
        synth._kit = silent_kit
        try:
            return synth.render_soundtrack(plan, SR)
        finally:
            synth._kit = kit
    return synth.render_soundtrack(plan, SR)


def onset_env(x: np.ndarray, sr: int = SR, hop: int = 256):
    """Spectral-flux onset strength (normalised 0..1) and its frame times."""
    mono = x.mean(axis=0)
    nfft = 1024
    nfr = 1 + max(0, (len(mono) - nfft) // hop)
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(mono, (nfr, nfft), (mono.strides[0] * hop, mono.strides[0]))
    mag = np.log1p(10.0 * np.abs(np.fft.rfft(frames * win, axis=1)))
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux = flux - np.convolve(flux, np.ones(9) / 9.0, mode="same")      # local-mean removal
    flux = np.maximum(flux, 0.0)
    flux /= (np.percentile(flux, 99.5) + 1e-9)
    times = (np.arange(nfr) * hop + nfft / 2) / sr
    return np.clip(flux, 0.0, 1.5), times


def snap_beats(x: np.ndarray, beats: list[float], dur: float) -> tuple[list[float], dict]:
    """Snap the nominal grid to the audio: one global offset (median over strong hits), then each beat to the
    strongest onset within +-45 ms of its shifted position when that onset is strong; else shifted only."""
    env, ft = onset_env(x)
    B = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    W = min(0.12, 0.3 * B)
    offs, strong = [], []
    for b in beats:
        m = (ft >= b - W) & (ft <= b + W)
        if not m.any():
            continue
        i = np.argmax(env[m])
        if env[m][i] > 0.35:
            offs.append(ft[m][i] - b)
    off = float(np.median(offs)) if len(offs) >= 3 else 0.0
    out = []
    moved = 0
    for b in beats:
        c = b + off
        m = (ft >= c - 0.045) & (ft <= c + 0.045)
        t = c
        if m.any():
            i = np.argmax(env[m])
            if env[m][i] > 0.4:
                t = float(ft[m][i])
                moved += 1
        out.append(t)
    out = np.clip(np.maximum.accumulate(np.asarray(out)), 0.0, dur)
    if beats and beats[0] == 0.0:
        out[0] = 0.0                        # the piece still starts at 0
    info = {"global_offset_ms": round(1000 * off, 1), "beats_snapped_to_onsets": moved, "n_beats": len(beats),
            "strong_hits": len(offs)}
    return [round(float(v), 4) for v in out], info


def tape_stop(x: np.ndarray, t0: float, length: float = 0.6) -> np.ndarray:
    """Music slows to a halt from t0 over `length` s (pitch and speed fall together), then silence."""
    n0 = int(t0 * SR)
    if n0 >= x.shape[1]:
        return x
    L = int(length * SR)
    rate = np.linspace(1.0, 0.0, L) ** 1.3
    pos = n0 + np.cumsum(rate)
    out = x.copy()
    for c in range(2):
        seg = np.interp(pos, np.arange(x.shape[1]), x[c])
        seg *= np.linspace(1.0, 0.0, L) ** 0.5
        end = min(x.shape[1], n0 + L)
        out[c, n0:end] = seg[:end - n0]
        out[c, end:] = 0.0
    return out


# ----------------------------------------------------------------------------- timings
def log_step(work: str, pid: str, step: dict, inputs=None) -> None:
    p = os.path.join(work, f"{pid}_timings.json")
    d = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {"piece": pid, "steps": [], "inputs": []}
    d["steps"] = [s for s in d["steps"] if s["name"] != step["name"]] + [step]
    for i in inputs or []:
        if i not in d["inputs"]:
            d["inputs"].append(i)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


# ----------------------------------------------------------------------------- images + cut-outs
def cutout_mask(px: np.ndarray, close: float | None = None) -> np.ndarray:
    """Subject mask of a still shot isolated on a flat ground: colour distance from the border median,
    closed, holes filled, largest component (+ components > 15 % of it), then GrabCut-refined.
    close (fraction of the short side): a large elliptical closing instead of GrabCut, for subjects with parts the
    same colour as the ground (a white sole on white) that a dark outline encloses."""
    h, w = px.shape[:2]
    u8 = (np.clip(px, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bw = max(4, min(h, w) // 40)
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    ref = np.median(border, 0)
    spread = np.percentile(np.linalg.norm(border - ref, axis=1), 95)
    d = np.linalg.norm(lab - ref, axis=2)
    m = (d > max(14.0, 2.2 * spread)).astype(np.uint8)
    k = max(3, min(h, w) // 150) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    if close:
        kk = int(close * min(h, w)) | 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        ff = m.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        m = m | (1 - ff)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        return (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if n > 1 else m
    if m.sum() < 0.05 * h * w:                      # subject close to the ground colour: rect GrabCut instead
        return _grabcut_rect(u8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * areas.max()]
    m = np.isin(lbl, keep).astype(np.uint8)
    # fill holes
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    m = m | (1 - ff)
    # grabcut refine on a downscaled copy
    s = 640.0 / max(h, w)
    if s < 1:
        small = cv2.resize(u8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        small, ms = u8, m
    gc = np.where(ms > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    er = cv2.erode(ms, np.ones((9, 9), np.uint8))
    gc[er > 0] = cv2.GC_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        r = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else r
        if r.sum() > 0.5 * m.sum():
            m = r
    except cv2.error:
        pass
    return m


def _grabcut_rect(u8: np.ndarray) -> np.ndarray:
    h, w = u8.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    sm = cv2.resize(u8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    gc = np.zeros((sh, sw), np.uint8)
    rect = (int(0.06 * sw), int(0.04 * sh), int(0.9 * sw), int(0.96 * sh) - 1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR), gc, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def mask_contour(m: np.ndarray, eps: float = 0.0012) -> dict:
    h, w = m.shape
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float64)
    pts = np.stack([c[:, 0] / w, c[:, 1] / h], 1)
    x, y, bw, bh = cv2.boundingRect(c.astype(np.int32))
    return {"points": pts.round(5).tolist(), "closed": True, "area": float(cv2.contourArea(c.astype(np.float32)) / (w * h)),
            "bbox": [x / w, y / h, bw / w, bh / h]}


def load_images(piece) -> tuple[list[dict], list[str]]:
    out, used = [], []
    cut = set(getattr(piece, "CUTOUTS", set()) or set())
    cache = os.path.join(ROOT, "10_finals", ".cutout_cache")
    for iid, rel in (getattr(piece, "IMAGES", {}) or {}).items():
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        im = Image.open(path)
        im.load()
        im = im.convert("RGB")
        px = np.asarray(im, np.float32) / 255.0
        d = {"id": iid, "pixels": px, "w": im.width, "h": im.height, "contours": [], "lines": [], "focus": [0.5, 0.5]}
        if iid in cut:
            os.makedirs(cache, exist_ok=True)
            close = (getattr(piece, "CUTOUT_CLOSE", {}) or {}).get(iid)
            cp = os.path.join(cache, os.path.basename(path) + (f".close{close}" if close else "") + ".json")
            if os.path.isfile(cp):
                c = json.load(open(cp))
            else:
                c = mask_contour(cutout_mask(px, close))
                json.dump(c, open(cp, "w"))
            c["id"] = f"{iid}:c0"
            d["contours"] = [c]
            bx, by, bw, bh = c["bbox"]
            d["focus"] = [bx + bw / 2, by + bh / 2]
            d["cut_bbox"] = c["bbox"]
        out.append(d)
        used.append(os.path.relpath(path, ROOT).replace("\\", "/"))
    return out, used


def analysis_for(images):
    return {"text": "", "text_lines": [], "duration_hint": None, "images": images, "audio": None,
            "mood": {"energy": 0.5, "valence": 0.5, "keywords": []}}


# ----------------------------------------------------------------------------- contact sheet
def contact_sheet(videos, times, out_path, row_h=360):
    """Frames at `times` read back from each mp4; rows wrap at 6 portrait / 4 landscape tiles."""
    rows = []
    for label, path in videos:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round(t * fps)))))
            ok, fr = cap.read()
            if not ok:
                fr = np.zeros((row_h, row_h, 3), np.uint8)
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (max(1, int(round(w * row_h / h))), row_h), interpolation=cv2.INTER_AREA)
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(fr, f"{label} {t:.2f}s", (6, row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
            tiles.append(np.pad(fr, ((3, 3), (3, 3), (0, 0)), constant_values=40))
        cap.release()
        per = 6 if tiles[0].shape[1] < tiles[0].shape[0] else 4      # wrap long rows
        for i in range(0, len(tiles), per):
            rows.append(np.concatenate(tiles[i:i + per], axis=1))
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=40) for r in rows], 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return out_path


# ----------------------------------------------------------------------------- commands
def nominal_plan(piece):
    beats = nominal_beats(piece.BPM, piece.DUR, getattr(piece, "PHASE", 0.0))
    return piece.build(beats), beats


def cmd_stem(a):
    piece = load_piece(a.piece)
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    plan, _ = nominal_plan(piece)
    x = music_stem(plan, getattr(piece, "DRUMS", True))
    x = fit_len(np.asarray(x, np.float32), int(round(piece.DUR * SR)))
    x = x / (np.abs(x).max() + 1e-9) * 0.89
    out = os.path.join(a.work, f"{piece.ID}_stem.wav")
    write_wav(out, x)
    log_step(a.work, piece.ID, {"name": "music_stem", "kind": "cpu", "model": "mograph.synth (CPU)",
                                "wall_s": round(time.time() - t0, 2)})
    print(out)


def _history(pid: str) -> dict:
    h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/history/{pid}", timeout=30).read())
    return h.get(pid, {})


def cmd_ace(a):
    from acestep_patch import patch
    piece = load_piece(a.piece)
    work = a.work
    stem = os.path.join(work, f"{piece.ID}_stem.wav")
    if not os.path.isfile(stem):
        sys.exit(f"missing {stem}; run stem first")
    t0 = time.time()
    remote = f"mograph/finals/{os.path.basename(stem)}"
    subprocess.run(["ssh", "-p", BOX["port"], BOX["host"], f"mkdir -p {BOX['dir']}"], check=True,
                   capture_output=True)
    subprocess.run(["scp", "-q", "-P", BOX["port"], stem, f"{BOX['host']}:{BOX['dir']}"], check=True)
    t_up = time.time() - t0
    ace = piece.ACE
    g = json.load(open(os.path.join(ROOT, "workflows", "api", "acestep_a2a.json"), encoding="utf-8"))
    g = patch(g, tags=ace["tags"], seed=ace.get("seed", 1), bpm=int(round(piece.BPM)), key=ace["keyscale"],
              seconds=piece.DUR, audio=remote, denoise=ace["denoise"])
    gp = os.path.join(work, f"{piece.ID}_ace_graph.json")
    json.dump(g, open(gp, "w"), indent=1)
    part = f"{piece.ID}_ace"
    pidf = os.path.join(work, f"{part}.pid")
    cmd = [sys.executable, STANDIN, "submit", gp, "--iid", IID, "--port", str(PORT), "--idea", a.idea,
           "--part", part, "--prompt-id-file", pidf, "--no-mem"]
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t_sub = time.time() - t1
    open(os.path.join(work, f"{part}.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"standin failed rc={r.returncode}: {(r.stdout + r.stderr)[-800:]}")
    pid = open(pidf).read().strip()
    h = _history(pid)
    msgs = {m[0]: m[1] for m in h.get("status", {}).get("messages", [])}
    e0 = msgs.get("execution_start", {}).get("timestamp")
    e1 = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
    exec_s = round((e1 - e0) / 1000.0, 2) if e0 and e1 else None
    files = [f for o in h.get("outputs", {}).values() for f in (o.get("audio") or [])]
    if not files:
        sys.exit(f"no audio output in history for {pid}")
    f0 = files[0]
    name = f0["filename"]
    dst = os.path.join(work, f"{piece.ID}_ace{os.path.splitext(name)[1]}")
    synced = os.path.join(a.synced, name) if a.synced else None
    got = "view"
    for _ in range(24):
        if synced and os.path.isfile(synced) and os.path.getsize(synced) > 0:
            got = "sync"
            break
        time.sleep(5)
    if got == "sync":
        with open(synced, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
    else:
        q = urllib.parse.urlencode({"filename": name, "subfolder": f0.get("subfolder", ""), "type": f0.get("type", "output")})
        with open(dst, "wb") as fo:
            fo.write(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/view?{q}", timeout=120).read())
    log_step(work, piece.ID, {"name": "ace_step_a2a", "kind": "gpu", "model": "acestep_v1.5_turbo (a2a, 8 steps)",
                              "denoise": ace["denoise"], "wall_s": round(time.time() - t0, 2),
                              "upload_s": round(t_up, 2), "submit_to_done_s": round(t_sub, 2),
                              "comfy_exec_s": exec_s, "prompt_id": pid, "output": name, "fetched_via": got})
    print(f"{piece.ID}: ace exec {exec_s}s, wall {time.time() - t0:.1f}s -> {dst} ({got})")


def final_audio(piece, work, use: str):
    """-> (mix (2,N), snapped beats, info, plan)."""
    N = int(round(piece.DUR * SR))
    src = None
    if use in ("auto", "ace"):
        for ext in (".flac", ".wav", ".mp3"):
            p = os.path.join(work, f"{piece.ID}_ace{ext}")
            if os.path.isfile(p):
                src = p
                break
        if src is None and use == "ace":
            sys.exit("no ACE output yet")
    if src is None:
        src = os.path.join(work, f"{piece.ID}_stem.wav")
    music = fit_len(load_audio(src), N)
    nb = nominal_beats(piece.BPM, piece.DUR, getattr(piece, "PHASE", 0.0))
    beats, info = snap_beats(music, nb, piece.DUR) if src.endswith(("ace.flac", "ace.wav", "ace.mp3")) else (nb, {})
    info["music_source"] = os.path.basename(src)
    plan = piece.build(beats)
    ts = getattr(piece, "TAPE_STOP", None)
    if callable(ts):
        ts = ts(Grid(beats, piece.DUR))
    if ts:
        music = tape_stop(music, float(ts))
    env = getattr(piece, "MUSIC_ENV", None)        # [(t, dB), ...] gain automation on the music, e.g. a rest
    if callable(env):
        env = env(Grid(beats, piece.DUR))
    if env:
        tt = np.array([p[0] for p in env], np.float64)
        gd = np.array([p[1] for p in env], np.float64)
        g = 10.0 ** (np.interp(np.arange(N) / SR, tt, gd) / 20.0)
        music = music * g[None, :].astype(np.float32)
    sfx = synth.render_sfx(plan, SR)
    out = mix.mix(music * 0.9, sfx, None, SR, target_lufs=-14.0)
    info["lufs"] = round(mix.loudness_lufs(out, SR), 2)
    return out, beats, info, plan


def cmd_render(a):
    piece = load_piece(a.piece)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    images, used = load_images(piece)
    analysis = analysis_for(images)
    t_img = time.time() - t0
    t1 = time.time()
    audio, beats, info, plan = final_audio(piece, a.work, a.audio)
    errs = validate_plan(plan, analysis)
    if errs:
        sys.exit("plan invalid:\n  " + "\n  ".join(errs[:30]))
    with open(os.path.join(a.work, f"{piece.ID}_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1)
    with open(os.path.join(a.work, f"{piece.ID}_snap.json"), "w", encoding="utf-8") as f:
        json.dump(dict(info, beats=beats), f, indent=1)
    wav = os.path.join(a.work, f"{piece.ID}_mix.wav")
    write_wav(wav, audio)
    t_aud = time.time() - t1
    print(f"{piece.ID}: audio {t_aud:.1f}s {info}", flush=True)
    log_step(a.work, piece.ID, {"name": "snap_sfx_mix", "kind": "cpu", "model": "onset snap + mograph.synth SFX + mograph.mix (-14 LUFS)",
                                "wall_s": round(t_aud, 2), **{k: v for k, v in info.items()}}, used)
    geometry.DRAFT_SHORT_SIDE = int(a.short) // 2 * 2
    quality = "full" if a.short >= 1080 else "draft"
    fps = float(plan["fps"])
    n = int(round(float(plan["duration"]) * fps))
    env = audio_envelope(audio, SR, fps, n)
    vids = []
    tag = f"{int(round(piece.DUR))}s"
    for key in (("9x16", "16x9") if a.fmt == "both" else (a.fmt,)):
        t2 = time.time()
        W, H = geometry.frame_size(FMTS[key], quality)
        out = os.path.join(a.out, f"motion-graphic_finals_v002_{piece.NAME}_{tag}_{key}.mp4")
        encode_mp4(render_frames(plan, analysis, FMTS[key], quality, env), fps, W, H, audio, SR, out, a.crf)
        dt = time.time() - t2
        print(f"[{key}] {W}x{H} {dt:.1f}s -> {out}", flush=True)
        log_step(a.work, piece.ID, {"name": f"render_{key}_{min(W, H)}", "kind": "cpu",
                                    "model": "mograph CPU compositor + ffmpeg x264", "wall_s": round(dt, 2),
                                    "output": os.path.relpath(out, ROOT).replace("\\", "/")})
        vids.append((key, out))
    if a.sheet:
        dur = float(plan["duration"])
        times = ([float(s) for s in a.sheet_times.split(",")] if a.sheet_times
                 else [round(dur * (k + 0.5) / 8.0, 2) for k in range(8)])
        p = contact_sheet(vids, times, os.path.join(a.sheet, f"{piece.ID}_{a.short}_sheet.jpg"),
                          row_h=300)
        print("sheet", p)


def cmd_preview(a):
    for crf in (23, 26, 29, 32):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-vf",
                        "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags",
                        "+faststart", a.dst], check=True)
        if os.path.getsize(a.dst) <= 8 * 1024 * 1024:
            break
    print(f"{a.dst} {os.path.getsize(a.dst) / 1e6:.2f} MB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stem")
    s.add_argument("piece")
    s.add_argument("--work", required=True)
    c = sub.add_parser("ace")
    c.add_argument("piece")
    c.add_argument("--work", required=True)
    c.add_argument("--idea", required=True)
    c.add_argument("--synced", default=None, help="dir the dashboard syncs outputs into (else /view fetch)")
    r = sub.add_parser("render")
    r.add_argument("piece")
    r.add_argument("--work", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--short", type=int, default=540)
    r.add_argument("--fmt", choices=("9x16", "16x9", "both"), default="both")
    r.add_argument("--audio", choices=("auto", "ace", "stem"), default="auto")
    r.add_argument("--sheet", default=None)
    r.add_argument("--sheet-times", default="")
    r.add_argument("--crf", type=int, default=18)
    p = sub.add_parser("preview")
    p.add_argument("src")
    p.add_argument("dst")
    a = ap.parse_args(argv)
    {"stem": cmd_stem, "ace": cmd_ace, "render": cmd_render, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
