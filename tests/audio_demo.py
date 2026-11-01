"""Render 10 s demo soundtracks for every mood at energy 0.3 and 0.8.

    python tests/audio_demo.py

Writes tests/out/audio_<mood>_<energy>.wav (music + sfx, mixed), prints render
timings and numeric sanity checks (peak, DC, LUFS, spectral centroid, onset
regularity vs the beat grid, longest near-silent gap), and writes a
spectrogram PNG of the driving/0.8 demo to tests/out/audio_spectrogram.png.
"""

import os
import sys
import time

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mograph import analyze, mix, synth  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out")
SR = 48000
KEYS = {"driving": ("A", "minor"), "calm": ("D", "major"), "dark": ("C#", "minor"),
        "bright": ("G", "major"), "playful": ("F", "major"), "epic": ("D", "minor")}


def demo_plan(mood: str, energy: float, seed: int = 7, duration: float = 10.0) -> dict:
    bpm = 92.0 if energy < 0.5 else 128.0
    beat = 60.0 / bpm
    beats = [round(float(t), 6) for t in np.arange(0.0, duration - 1e-9, beat)]
    dbs = beats[::4]
    bar = 4 * beat
    t_a, t_b = dbs[1] if bpm < 100 else dbs[2], dbs[3] if bpm < 100 else dbs[4]
    key, scale = KEYS[mood]
    sfx = [{"t": t_a, "type": "impact", "gain": 0.8}, {"t": t_a, "type": "whoosh", "gain": 0.6},
           {"t": t_b, "type": "impact", "gain": 0.7}, {"t": t_b - 0.5 * beat, "type": "glitch", "gain": 0.4},
           {"t": t_b + beat, "type": "pop", "gain": 0.5}, {"t": t_b + 2 * beat, "type": "click", "gain": 0.5},
           {"t": t_a + 2 * beat, "type": "hit", "gain": 0.5}, {"t": t_b, "type": "swell", "gain": 0.5}]
    risers = [{"t0": t_a - bar, "t1": t_a}] if energy >= 0.5 else [{"t0": t_a - bar / 2, "t1": t_a}]
    return {
        "version": 1, "seed": seed, "duration": duration, "fps": 30,
        "style": {"energy": energy},
        "audio": {"source": "synth", "bpm": bpm, "key": key, "scale": scale, "mood": mood,
                  "beats": beats, "downbeats": dbs,
                  "sections": [{"t": 0.0, "label": "intro"}, {"t": t_a, "label": "drop"},
                               {"t": t_b, "label": "chorus"}],
                  "risers": risers, "sfx": sfx},
        "voice": None, "scenes": [],
    }


def metrics(x: np.ndarray, plan: dict) -> dict:
    mono = x.mean(axis=0).astype(np.float64)
    f, _, Z = signal.stft(mono, SR, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    cen = float((f[:, None] * P).sum() / (P.sum() + 1e-12))
    blk = int(0.05 * SR)
    lv = np.sqrt(np.mean(mono[: len(mono) // blk * blk].reshape(-1, blk) ** 2, axis=1))
    lv_db = 20 * np.log10(lv + 1e-12)
    quiet = lv_db < -50
    gap, run = 0, 0
    for q in quiet[:-8]:  # ignore the final fade
        run = run + 1 if q else 0
        gap = max(gap, run)
    a = analyze.analyze_audio(x, SR)
    beats = np.array(plan["audio"]["beats"])
    on = np.array(a["onsets"])
    near = np.mean([np.min(np.abs(beats - o)) < 0.03 for o in on]) if len(on) else 0.0
    return {"peak": float(np.abs(x).max()), "dc": float(mono.mean()), "lufs": mix.loudness_lufs(x, SR),
            "centroid_hz": cen, "gap_s": gap * 0.05, "est_bpm": a["bpm"], "onsets_on_grid": float(near)}


def spectrogram_png(x: np.ndarray, path: str) -> None:
    from PIL import Image

    mono = x.mean(axis=0)
    f, t, Z = signal.stft(mono, SR, nperseg=2048, noverlap=2048 - 480)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    # log-frequency axis 30 Hz .. 20 kHz, 400 rows
    rows = np.geomspace(30, 20000, 400)
    idx = np.clip(np.searchsorted(f, rows), 0, len(f) - 1)
    img = S[idx][::-1]
    img = np.clip((img - (img.max() - 90)) / 90.0, 0, 1)
    rgb = np.stack([img ** 0.6, img ** 1.2, 0.35 + 0.65 * img ** 3], -1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(rgb).resize((1000, 400))
    # beat ticks along the top
    arr = np.array(im)
    dur = len(mono) / SR
    for b in np.arange(0, dur, 60.0 / 128.0):
        c = int(b / dur * 999)
        arr[:8, c] = (255, 255, 255)
    Image.fromarray(arr).save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'demo':<16}{'music s':>8}{'sfx s':>7}{'mix s':>7}{'peak':>7}{'LUFS':>7}{'DC':>9}"
          f"{'cent Hz':>9}{'gap s':>7}{'bpm':>8}{'on-grid':>8}")
    for mood in synth.MOODS:
        for energy in (0.3, 0.8):
            p = demo_plan(mood, energy)
            t0 = time.perf_counter()
            music = synth.render_soundtrack(p, SR)
            t1 = time.perf_counter()
            sfx = synth.render_sfx(p, SR)
            t2 = time.perf_counter()
            out = mix.mix(music, sfx, None, SR)
            t3 = time.perf_counter()
            name = f"audio_{mood}_{energy}"
            wavfile.write(os.path.join(OUT, name + ".wav"), SR, out.T.copy())
            m = metrics(out, p)
            print(f"{name:<16}{t1 - t0:8.2f}{t2 - t1:7.2f}{t3 - t2:7.2f}{m['peak']:7.3f}{m['lufs']:7.2f}"
                  f"{m['dc']:9.5f}{m['centroid_hz']:9.0f}{m['gap_s']:7.2f}{m['est_bpm']:8.1f}{m['onsets_on_grid']:8.2f}"
                  f"   (plan bpm {p['audio']['bpm']:.0f})")
            if mood == "driving" and energy == 0.8:
                spectrogram_png(out, os.path.join(OUT, "audio_spectrogram.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
