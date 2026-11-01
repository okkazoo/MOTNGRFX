#!/usr/bin/env python3
"""CPU synth draft soundtracks for the test briefs (B1-B5, S1, S2), for ACE-Step a2a tests.

  python tools/sound_drafts.py --out 03_sound/v001/local            # render all drafts
  python tools/sound_drafts.py --out ... --only B3 B3_30s --upload  # + scp to the box input/mograph/
  python tools/sound_drafts.py compare --drafts DIR --outputs DIR --json result.json

render: builds a minimal plan per brief (plan.audio: bpm, beats, downbeats, sections, risers,
sfx; style.energy), renders mograph.synth music + sfx, mixes with mograph.mix and writes
<id>_draft.wav (48 kHz 16-bit stereo) + <id>_draft.json (plan, ACE tags, key) + drafts.json.
--upload copies the WAVs to /workspace/ComfyUI/input/mograph/ on the box over scp.

compare: for every <id>_draft.wav and every output file whose name contains "<id>_" (flac/wav/mp3),
runs mograph.analyze.analyze_audio and reports bpm, beat-grid offset and how many of the
draft's onsets reappear in the output (within 50 ms).

mixvox: python tools/sound_drafts.py mixvox --music B1.flac --voice vox.flac --at 2.5 --out mix.wav
lays a voice line over music with mograph.mix (voice sidechain ducking, -14 LUFS, limiter).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mograph import analyze, mix, synth  # noqa: E402

SR = 48000
BOX = {"host": os.environ.get("MOGRAPH_BOX_HOST", "root@<box-ip>"),
       "port": os.environ.get("MOGRAPH_BOX_PORT", "15314"), "dir": "/workspace/ComfyUI/input/mograph/"}

# id: bpm, mood (synth), key, scale, energy, ACE keyscale, sections [(fraction, label)], sfx recipe, ACE tags
BRIEFS = {
    "B1": dict(bpm=96, mood="calm", key="D", scale="dorian", energy=0.55, keyscale="D minor",
               sections=[(0.0, "intro"), (0.3, "drop"), (0.7, "chorus")], sfx="b1",
               tags="lo-fi hip-hop, warm Rhodes electric piano, vinyl crackle, soft kick, mellow, "
                    "rain, nocturnal, cosy, instrumental"),
    "B2": dict(bpm=60, mood="calm", key="D", scale="major", energy=0.15, keyscale="D major",
               sections=[(0.0, "intro"), (0.4, "main")], sfx="b2",
               tags="ambient, soft synth pad, slow swells, breathing, airy, calm, meditation, "
                    "no drums, instrumental"),
    "B3": dict(bpm=128, mood="driving", key="A", scale="minor", energy=0.9, keyscale="A minor",
               sections=[(0.0, "intro"), (0.3, "drop"), (0.7, "chorus")], sfx="b3",
               tags="driving electronic, heavy kick, four on the floor, risers, impacts, aggressive, "
                    "hype, energetic, sports promo, instrumental"),
    "B4": dict(bpm=84, mood="playful", key="F", scale="major", energy=0.35, keyscale="F major",
               sections=[(0.0, "intro"), (0.3, "main"), (0.7, "chorus")], sfx="b4",
               tags="mid-century library music, warm groove, brushed drums, upright bass, vibraphone, "
                    "nostalgic, tender, instrumental"),
    "B5": dict(bpm=70, mood="dark", key="C#", scale="minor", energy=0.3, keyscale="C# minor",
               sections=[(0.0, "intro"), (0.35, "main"), (0.75, "climax")], sfx="b5",
               tags="dark ambient, deep drones, sonar pings, slow swell, mysterious, underwater, "
                    "cinematic, instrumental"),
    "S1": dict(bpm=70, mood="calm", key="G", scale="major", energy=0.25, keyscale="G major",
               sections=[(0.0, "intro"), (0.4, "main")], sfx="s1",
               tags="ambient guitar, clean electric guitar, reverb, tape hiss, warm, soft, film score, "
                    "instrumental"),
    "S2": dict(bpm=140, mood="playful", key="E", scale="minor", energy=0.95, keyscale="E minor",
               sections=[(0.0, "intro"), (0.3, "drop"), (0.7, "chorus")], sfx="s2",
               tags="breakbeat, fast chopped drums, risers, impacts, energetic, hectic, electronic, "
                    "cinematic, instrumental"),
}
# variants: (id, brief, duration, seed)
DRAFTS = [(b, b, 10.0, 7) for b in BRIEFS] + [("B3_30s", "B3", 30.0, 7)]


def _sfx(kind: str, beats: list, dbs: list, secs: dict, beat: float, dur: float):
    """SFX + risers per brief recipe. secs maps label -> time."""
    sfx, risers = [], []
    drop = secs.get("drop") or secs.get("main")
    top = secs.get("chorus") or secs.get("climax")
    later_dbs = [d for d in dbs if 0 < d < dur - 0.5]
    if kind == "b1":  # chopstick tap on the reveal, soft swell into the chorus
        if drop:
            sfx.append({"t": drop, "type": "click", "gain": 0.6})
        if top:
            sfx.append({"t": top, "type": "swell", "gain": 0.3})
    elif kind == "b2":  # soft whoosh + swell on each inhale (4 beats in, 4 out = every 2nd bar)
        for d in [0.0] + later_dbs[1::2]:
            sfx.append({"t": d, "type": "swell", "gain": 0.35})
            sfx.append({"t": d, "type": "whoosh", "gain": 0.2})
    elif kind == "b3":  # whoosh on every whip (into each downbeat), impact on the title, risers
        for d in later_dbs:
            sfx.append({"t": max(0.0, d - 0.5 * beat), "type": "whoosh", "gain": 0.45})
        for t in (drop, top):
            if t:
                sfx += [{"t": t, "type": "impact", "gain": 0.85}, {"t": t, "type": "hit", "gain": 0.5}]
                risers.append({"t0": max(0.0, t - 4 * beat), "t1": t})
    elif kind == "b4":  # paper slaps on downbeats, scissor snips on the chorus offbeats
        for d in later_dbs:
            sfx.append({"t": d, "type": "pop", "gain": 0.5})
        if top:
            for b in beats:
                if top <= b < dur - 0.5:
                    sfx.append({"t": b + 0.5 * beat, "type": "click", "gain": 0.35})
    elif kind == "b5":  # sonar ping on each cut (downbeat), swell to the reveal
        for d in [0.0] + later_dbs:
            sfx.append({"t": d, "type": "pop", "gain": 0.4})
        if top:
            sfx += [{"t": max(0.0, top - 4 * beat), "type": "swell", "gain": 0.5},
                    {"t": top, "type": "impact", "gain": 0.45}]
    elif kind == "s1":  # slow swells on the section changes
        for t in [0.0] + [secs[k] for k in secs if secs[k] > 0]:
            sfx.append({"t": t, "type": "swell", "gain": 0.3})
    elif kind == "s2":  # breakbeat: risers, impacts, glitches, hits on downbeats
        for d in later_dbs:
            sfx.append({"t": d, "type": "hit", "gain": 0.4})
        for t in (drop, top):
            if t:
                sfx += [{"t": t, "type": "impact", "gain": 0.8},
                        {"t": max(0.0, t - 0.5 * beat), "type": "glitch", "gain": 0.4}]
                risers.append({"t0": max(0.0, t - 4 * beat), "t1": t})
    return sorted(sfx, key=lambda s: s["t"]), risers


def make_plan(brief_id: str, duration: float = 10.0, seed: int = 7) -> dict:
    b = BRIEFS[brief_id]
    bpm = float(b["bpm"])
    beat = 60.0 / bpm
    beats = [round(float(t), 6) for t in np.arange(0.0, duration - 1e-9, beat)]
    dbs = beats[::4]
    sections, used = [], set()
    for frac, label in b["sections"]:
        t = min(dbs, key=lambda d: abs(d - frac * duration))
        if t in used or (t > 0 and t > duration - 1.0):
            continue
        used.add(t)
        sections.append({"t": t, "label": label})
    secs = {s["label"]: s["t"] for s in sections}
    sfx, risers = _sfx(b["sfx"], beats, dbs, secs, beat, duration)
    return {
        "version": 1, "seed": seed, "duration": duration, "fps": 30,
        "style": {"energy": b["energy"]},
        "audio": {"source": "synth", "bpm": bpm, "key": b["key"], "scale": b["scale"], "mood": b["mood"],
                  "beats": beats, "downbeats": dbs, "sections": sections, "risers": risers, "sfx": sfx},
        "voice": None, "scenes": [],
    }


def render(plan: dict) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    music = synth.render_soundtrack(plan, SR)
    sfx = synth.render_sfx(plan, SR)
    out = mix.mix(music, sfx, None, SR)
    return out, {"render_s": round(time.perf_counter() - t0, 2), "lufs": round(mix.loudness_lufs(out, SR), 2),
                 "peak": round(float(np.abs(out).max()), 3)}


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    wavfile.write(str(path), SR, pcm.T.copy())


def cmd_render(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for did, bid, dur, seed in DRAFTS:
        if args.only and did not in args.only:
            continue
        plan = make_plan(bid, dur, seed)
        x, stats = render(plan)
        wav = out / f"{did}_draft.wav"
        write_wav(wav, x)
        b = BRIEFS[bid]
        meta = {"id": did, "brief": bid, "duration": dur, "bpm": b["bpm"], "keyscale": b["keyscale"],
                "tags": b["tags"], "wav": wav.name, **stats, "plan": plan}
        (out / f"{did}_draft.json").write_text(json.dumps(meta, indent=1))
        manifest[did] = {k: v for k, v in meta.items() if k != "plan"}
        print(f"{did:<8} {dur:5.1f}s {b['bpm']:4d} bpm  {stats['render_s']:5.2f}s render  "
              f"{stats['lufs']:6.2f} LUFS  peak {stats['peak']:.3f}  -> {wav}")
    mpath = out / "drafts.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    old.update(manifest)
    mpath.write_text(json.dumps(old, indent=1))
    if args.upload:
        files = [str(out / f"{d}_draft.wav") for d in manifest]
        subprocess.run(["ssh", "-p", BOX["port"], BOX["host"], f"mkdir -p {BOX['dir']}"], check=True)
        subprocess.run(["scp", "-q", "-P", BOX["port"], *files, f"{BOX['host']}:{BOX['dir']}"], check=True)
        print(f"uploaded {len(files)} files to {BOX['dir']}")


# ----------------------------------------------------------------------------- compare


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """(channels, n) float32 + sr; wav via scipy, anything else via ffmpeg."""
    if path.suffix.lower() == ".wav":
        sr, x = wavfile.read(str(path))
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
        return (x.T if x.ndim == 2 else x[None, :]), sr
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy(), SR


def grid_offset(onsets: np.ndarray, period: float) -> float:
    """Circular mean phase (s) of onsets on a grid of the given period."""
    if len(onsets) == 0:
        return float("nan")
    ph = np.angle(np.mean(np.exp(2j * np.pi * onsets / period)))
    return float(ph / (2 * np.pi) * period)


def compare_one(ref: dict, path: Path, plan_bpm: float) -> dict:
    x, sr = load_audio(path)
    a = analyze.analyze_audio(x, sr)
    on = np.array(a["onsets"])
    ref_on = np.array(ref["onsets"])
    hit = float(np.mean([np.min(np.abs(on - o)) <= 0.05 for o in ref_on])) if len(on) and len(ref_on) else 0.0
    prec = float(np.mean([np.min(np.abs(ref_on - o)) <= 0.05 for o in on])) if len(on) and len(ref_on) else 0.0
    period = 60.0 / plan_bpm
    grid8 = np.arange(0.0, a["duration"] + period, period / 2)  # 8th-note grid of the plan
    on_grid = float(np.mean([np.min(np.abs(grid8 - o)) <= 0.04 for o in on])) if len(on) else 0.0
    # analyze_audio reports 70..180 bpm; fold octave errors onto the plan tempo
    folded = a["bpm"] * 2.0 ** round(np.log2(plan_bpm / a["bpm"])) if a["bpm"] > 0 else 0.0
    # tracked beats vs the plan beat grid (at the tracker's octave)
    bt = np.array(a["beats"])
    gp = period * 2.0 ** round(np.log2((60.0 / a["bpm"]) / period)) if a["bpm"] > 0 else period
    gridb = np.arange(0.0, a["duration"] + gp, gp)
    beats_on = float(np.mean([np.min(np.abs(gridb - b)) <= 0.05 for b in bt])) if len(bt) else 0.0
    return {"file": path.name, "bpm": a["bpm"], "bpm_folded": round(folded, 2),
            "tempo_ok": bool(abs(folded - plan_bpm) / plan_bpm <= 0.03),
            "n_onsets": len(on), "draft_hits_kept": round(hit, 3),
            "onsets_matching_draft": round(prec, 3), "onsets_on_plan_grid": round(on_grid, 3),
            "beats_on_plan_grid": round(beats_on, 3),
            "beat_phase_ms": round(1000 * grid_offset(bt, gp), 1), "duration": round(a["duration"], 2)}


def cmd_compare(args) -> None:
    drafts = Path(args.drafts)
    outs = [Path(p) for d in args.outputs for p in Path(d).rglob("*")
            if p.suffix.lower() in (".flac", ".wav", ".mp3") and not any(s.startswith(".") for s in p.parts[-3:-1])]
    result = {}
    for meta_p in sorted(drafts.glob("*_draft.json")):
        meta = json.loads(meta_p.read_text())
        did = meta["id"]
        if args.only and did not in args.only:
            continue
        x, sr = load_audio(drafts / meta["wav"])
        ref = analyze.analyze_audio(x, sr)
        rows = [compare_one(ref, drafts / meta["wav"], meta["bpm"])]
        rows[0]["file"] = "draft:" + meta["wav"]
        pat = re.compile(rf"_{re.escape(did)}_(?!30s)")
        for p in sorted(outs):
            if pat.search(p.name) and "_draft" not in p.name:
                rows.append(compare_one(ref, p, meta["bpm"]))
        result[did] = {"plan_bpm": meta["bpm"], "rows": rows}
        print(f"\n{did}  (plan {meta['bpm']} bpm)")
        print(f"  {'file':<52}{'bpm':>8}{'fold':>8}{'onsets':>7}{'kept':>7}{'match':>7}{'on8th':>7}"
              f"{'beatsOK':>8}{'phase ms':>9}")
        for r in rows:
            print(f"  {r['file'][-52:]:<52}{r['bpm']:8.1f}{r['bpm_folded']:8.1f}{r['n_onsets']:7d}{r['draft_hits_kept']:7.2f}"
                  f"{r['onsets_matching_draft']:7.2f}{r['onsets_on_plan_grid']:7.2f}{r['beats_on_plan_grid']:8.2f}"
                  f"{r['beat_phase_ms']:9.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


def cmd_mixvox(args) -> None:
    """Voice over music with mograph.mix (sidechain duck), voice starting at --at seconds."""
    from scipy.signal import resample_poly

    m, msr = load_audio(Path(args.music))
    v, vsr = load_audio(Path(args.voice))
    if msr != SR:
        m = resample_poly(m, SR, msr, axis=1).astype(np.float32)
    if vsr != SR:
        v = resample_poly(v, SR, vsr, axis=1).astype(np.float32)
    N = m.shape[1]
    voice = np.zeros((2, N), np.float32)
    s = int(round(args.at * SR))
    seg = (v if v.shape[0] == 2 else np.repeat(v[:1], 2, 0))[:, : max(0, N - s)]
    voice[:, s:s + seg.shape[1]] = seg
    # match voice level to the music before the mixer's ducking/normalisation
    voice *= 10 ** ((mix.loudness_lufs(m, SR) - mix.loudness_lufs(seg, SR) + args.voice_db) / 20.0)
    out = mix.mix(m, None, voice, SR, duck_db=args.duck_db)
    write_wav(Path(args.out), out)
    print(f"wrote {args.out}  voice {seg.shape[1] / SR:.2f}s at {args.at}s, duck {args.duck_db} dB")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("render")
    c = sub.add_parser("compare")
    v = sub.add_parser("mixvox")
    v.add_argument("--music", required=True)
    v.add_argument("--voice", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--at", type=float, default=2.5)
    v.add_argument("--duck-db", type=float, default=-9.0)
    v.add_argument("--voice-db", type=float, default=3.0, help="voice level vs music loudness (LU)")
    for p in (ap, r):
        p.add_argument("--out", default=str(REPO / "03_sound" / "v001" / "local"))
        p.add_argument("--only", nargs="*")
        p.add_argument("--upload", action="store_true")
    c.add_argument("--drafts", required=True)
    c.add_argument("--outputs", nargs="+", required=True)
    c.add_argument("--only", nargs="*")
    c.add_argument("--json")
    args = ap.parse_args(argv)
    {"compare": cmd_compare, "mixvox": cmd_mixvox}.get(args.cmd, cmd_render)(args)


if __name__ == "__main__":
    main()
