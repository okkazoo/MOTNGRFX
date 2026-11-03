"""asr_whisper.py: Whisper large-v3 (faster-whisper int8, CPU) timed words of a wav -> json. Runs in the SEPARATE venv (/workspace/asr_venv/bin/python, BUILT.md Speeds: 24 s clip in 11.8 s),
standalone (no repo imports). CPU only on purpose: it must not take VRAM from Gemma.   python asr_whisper.py in.wav out.json
Output {model, secs, language, language_probability, segments:[{start, end, text, no_speech_prob, words:[{w, s, e, p}]}]} (the shape tools/fusion/music_read.lyric_words reads)."""
import json
import os
import sys
import time

from faster_whisper import WhisperModel

import wave

import numpy as np

wav, out = sys.argv[1], sys.argv[2]
w_ = wave.open(wav)                      # the wav is read here (16 kHz mono int16, what media_read.extract_audio writes): faster-whisper's own file decoder (PyAV) failed in the box venv (`open() got an unexpected keyword argument 'metadata_errors'`)
if w_.getframerate() != 16000 or w_.getnchannels() != 1 or w_.getsampwidth() != 2:
    sys.exit(f"asr_whisper: need a 16 kHz mono 16-bit wav, got {w_.getframerate()} Hz / {w_.getnchannels()} ch / {8 * w_.getsampwidth()} bit")
audio = np.frombuffer(w_.readframes(w_.getnframes()), dtype=np.int16).astype(np.float32) / 32768
t0 = time.time()
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=min(32, os.cpu_count() or 8))
segs, info = m.transcribe(audio, word_timestamps=True, language=None, beam_size=5, vad_filter=True)       # the VAD drops silence / music-only stretches (fewer hallucinated words)
segs = list(segs)
vad = True
if not segs:                             # 2026-11-03: the VAD can drop singing over a loud mix (a 30 s clip with one sung line came back empty), so an empty result is tried again without it; the stock phrases Whisper invents over music are then dropped
    vad = False
    FALSE_LINES = {"thank you", "thanks", "thanks for watching", "thank you for watching", "you", "bye", "okay"}
    found, pos, sr = [], 0.0, 16000
    while len(audio) / sr - pos > 2.0:   # word timing raises IndexError on a stretch with no words in it (a long instrumental start): keep what came, go on from just past it
        end = pos
        try:
            it, info = m.transcribe(audio[int(pos * sr):], word_timestamps=True, language=None, beam_size=5, vad_filter=False)
            for s in it:
                found.append((pos, s))
                end = pos + s.end
            break
        except IndexError:
            pos = max(end, pos + 5.0)
    segs = [(o, s) for o, s in found if s.no_speech_prob < 0.8 and s.text.strip().lower().strip(" .,!?") not in FALSE_LINES]
    segs = [q for k, q in enumerate(segs) if not (k and q[1].text.strip().lower() == segs[k - 1][1].text.strip().lower())]      # without the VAD Whisper repeats a line over the music after it
segs = [q if isinstance(q, tuple) else (0.0, q) for q in segs]
res = [{"start": round(o + s.start, 2), "end": round(o + s.end, 2), "text": s.text.strip(), "no_speech_prob": round(float(s.no_speech_prob), 3),
        "words": [{"w": w.word.strip(), "s": round(o + w.start, 2), "e": round(o + w.end, 2), "p": round(w.probability, 2)} for w in (s.words or [])]} for o, s in segs]
json.dump({"model": "whisper large-v3 (faster-whisper int8, CPU)" + ("" if vad else ", no VAD"), "secs": round(time.time() - t0, 1), "language": info.language, "language_probability": round(float(info.language_probability), 3),
           "segments": res}, open(out, "w"), indent=1)
print("secs", round(time.time() - t0, 1), "segments", len(res))
