"""Patch the ACE-Step 1.5 API graphs in this folder: tags, lyrics, seed, bpm, key, duration, input, denoise.

    python acestep_patch.py acestep_t2m.json out.json --tags "lo-fi hip-hop, ..." --seconds 10 --bpm 96 --key "D minor" --seed 1
    python acestep_patch.py acestep_a2a.json out.json --tags "..." --audio mograph/B3_draft.wav --denoise 0.5 --seconds 10 --bpm 128
    python acestep_patch.py audio_passthrough.json out.json --audio mograph/B3_draft.wav

Settings follow the official template audio_ace_step_1_5_split (turbo: 8 steps, cfg 1, euler/simple,
AuraFlow shift 3, LM cfg 2.0 / temp 0.85 / top_p 0.9). a2a turns generate_audio_codes off (the node's
tooltip: off when an audio reference is given) and starts the sampler from the VAE-encoded input at
denoise < 1. Fields are found by class_type, so node ids may change.
"""
import argparse
import json
from pathlib import Path


def _nodes(g, cls):
    return [k for k, v in g.items() if v["class_type"] == cls]


def patch(g, tags=None, lyrics=None, seed=None, bpm=None, key=None, seconds=None, audio=None,
          denoise=None, steps=None, prefix=None):
    g = json.loads(json.dumps(g))
    for k in _nodes(g, "TextEncodeAceStepAudio1.5"):
        i = g[k]["inputs"]
        for name, val in (("tags", tags), ("lyrics", lyrics), ("seed", seed), ("bpm", bpm),
                          ("keyscale", key), ("duration", seconds)):
            if val is not None:
                i[name] = int(val) if name in ("bpm", "seed") else val
    for k in _nodes(g, "EmptyAceStep1.5LatentAudio"):
        if seconds is not None:
            g[k]["inputs"]["seconds"] = float(seconds)
    for k in _nodes(g, "KSampler"):
        i = g[k]["inputs"]
        if seed is not None:
            i["seed"] = int(seed)
        if denoise is not None:
            i["denoise"] = float(denoise)
        if steps is not None:
            i["steps"] = int(steps)
    for k in _nodes(g, "LoadAudio"):
        if audio is not None:
            g[k]["inputs"]["audio"] = audio
    for k in _nodes(g, "SaveAudioAdvanced"):
        if prefix is not None:
            g[k]["inputs"]["filename_prefix"] = prefix
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("out")
    ap.add_argument("--tags")
    ap.add_argument("--lyrics")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--key")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--audio")
    ap.add_argument("--denoise", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--prefix")
    a = ap.parse_args()
    g = json.loads(Path(a.graph).read_text())
    g = patch(g, a.tags, a.lyrics, a.seed, a.bpm, a.key, a.seconds, a.audio, a.denoise, a.steps, a.prefix)
    Path(a.out).write_text(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
