"""songs.py - the 9 ACE-Step 1.5 songs for the 15_fusion pieces (P06 uses the user's Song audio, not listed here).

    python tools/fusion/songs.py build [--seeds 2]   write 15_fusion/songs.json + 15_fusion/v001/graphs_pending/<piece>_s<seed>.json
    python tools/fusion/songs.py check               validate every pending graph (loads, loader filenames, fields)
    python tools/fusion/songs.py table               print piece / bpm / key / duration / prompt start

Graphs are API-format, built from workflows/api/acestep_t2m.json (ACE-Step 1.5 TURBO: acestep_v1.5_turbo.safetensors,
qwen_0.6b_ace15 + qwen_1.7b_ace15, ace_1.5_vae - the filenames the motion-graphic card installs) and patched with
workflows/api/acestep_patch.py. Submit later with scripts/progress_standin.py submit <graph> --iid <id> --port <port>.

Duration = PLAN.md length + 1.5 s tail so the cut can end on a downbeat. Text-to-music cannot hit an exact planned
tempo (03_sound/v001), which is fine: the visuals follow the song, so read bpm/key/chords back with
tools/fusion/music_analysis.py after generation. Prompts: one continuous arrangement per song (no stitched sections);
`pace` records the intended shape (slow-build / steady / hectic) so the set varies.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FUSION = ROOT / "15_fusion"
PENDING = FUSION / "v001" / "graphs_pending"
BASE_GRAPH = ROOT / "workflows" / "api" / "acestep_t2m.json"
TAIL_S = 1.5
EXPECT = {"UNETLoader.unet_name": "acestep_v1.5_turbo.safetensors",
          "DualCLIPLoader.clip_name1": "qwen_0.6b_ace15.safetensors",
          "DualCLIPLoader.clip_name2": "qwen_1.7b_ace15.safetensors",
          "VAELoader.vae_name": "ace_1.5_vae.safetensors"}

NO_VOX = "instrumental, no vocals, no singing, no choir"

# piece, slug, plan length (s), bpm, key, pace, text (text-only pieces), tags
SONGS = [
    dict(piece="P01", slug="salt-silver", len_s=15, bpm=72, key="D minor", pace="slow-build",
         text="SALT / AND / SILVER / THE SEA REMEMBERS",
         tags="cinematic film score, melancholic and vast, solo felt piano with sparse low notes, slow bowed cellos "
              "and violas, high airy violins like sea mist, soft sustained string pad, distant harp harmonics, "
              "deep sub drone, very sparse opening that grows in one slow continuous crescendo, long swelling "
              "strings, gentle rolling piano arpeggio, spacious cathedral reverb, wide stereo, emotional, "
              "cold silver light, one continuous piece, 72 BPM, " + NO_VOX),
    dict(piece="P02", slug="after-hours", len_s=12, bpm=122, key="F minor", pace="steady-groove",
         text="AFTER / HOURS / THE CITY / KEEPS / DREAMING",
         tags="deep house, late-night city, warm Rhodes electric piano chords, round sub bass, soft four-on-the-floor "
              "kick, shuffled closed hi-hats, rim shot, dusty vinyl texture, analog pad, filtered intro that gradually "
              "opens up into the full groove, one continuous rolling groove, hypnotic, smooth, rainy neon, "
              "sidechained pads, warm and intimate, 122 BPM, " + NO_VOX),
    dict(piece="P03", slug="tide-line", len_s=10, bpm=84, key="Bb major", pace="slow-build",
         text="WHERE / THE TIDE / LETS GO",
         tags="ambient cinematic, soft felt piano with long decay, slow swelling synth and string pads like waves "
              "rising and ebbing, sparse opening, gradual crescendo, gentle tape-warbled textures, airy "
              "shimmering high pad, soft sub swell, no drums, floating, wistful, spacious reverb, "
              "continuous and flowing, 84 BPM, " + NO_VOX),
    dict(piece="P04", slug="hold-nothing", len_s=20, bpm=100, key="G minor", pace="evolving",
         text="HOLD / ON TO / NOTHING / LET IT / MOVE",
         tags="organic deep house, warm round bass guitar, soft kick, hand percussion and shakers, congas, "
              "kalimba and marimba melody, plucked muted guitar, breathy airy pad, wooden textures, "
              "starts light and layers in gradually, one continuous evolving groove, flowing, release, "
              "sunlit and earthy, 100 BPM, " + NO_VOX),
    dict(piece="P05", slug="wild-bloom", len_s=8, bpm=140, key="E minor", pace="hectic",
         text="WILD / BLOOM / RUN",
         tags="cinematic breakbeat, driving, relentless, fast chopped amen-style breakbeat drums, punchy kick and "
              "snare, rolling distorted bass, urgent staccato strings ostinato, pulsing arpeggiated synth, "
              "riser sweeps, energetic and wild, no breakdown, constant forward momentum from the first beat, "
              "aggressive yet bright, wide stereo, 140 BPM, " + NO_VOX),
    dict(piece="P07", slug="desert-signal", len_s=24, bpm=124, key="F# minor", pace="hypnotic",
         text=None,
         tags="hypnotic deep house, desert at dusk, locked four-on-the-floor kick, deep rolling bassline, "
              "mid-eastern flavoured plucked synth motif with slow filter movement, arpeggiated analog sequencer, "
              "dry claps, shaker, spacious dub delays, evolving tiny details over a steady pulse, trance-like, "
              "mysterious, one continuous groove with no breaks, 124 BPM, " + NO_VOX),
    dict(piece="P08", slug="long-light", len_s=30, bpm=60, key="G major", pace="slow-build",
         text=None,
         tags="slow cinematic film score, vast open landscape at sunrise, sparse opening with a single warm "
              "cello line and low sustained strings, one long gradual crescendo as full orchestral strings "
              "swell and rise, soft french horn, harp, gentle piano pulse, shimmering high violins, "
              "deep timpani roll near the peak, hopeful, awe, expansive, wide stereo reverb, "
              "one continuous slow build, 60 BPM, " + NO_VOX),
    dict(piece="P09", slug="slow-gold", len_s=6, bpm=118, key="A major", pace="steady-groove",
         text=None,
         tags="melodic deep house, golden hour, warm bright plucked synth hook, glowing supersaw pad, "
              "round bass, crisp four-on-the-floor kick, open hi-hats, light percussion, uplifting and "
              "bittersweet, immediately in the groove, steady and continuous, polished, 118 BPM, " + NO_VOX),
    dict(piece="P10", slug="noir-step", len_s=15, bpm=90, key="B minor", pace="steady-tense",
         text=None,
         tags="trip-hop, film noir, dusty lazy breakbeat with a heavy swung kick and snare, vinyl crackle, "
              "deep upright bass walking line, muted trumpet, smoky Rhodes chords, spy-theme tremolo guitar, "
              "cool and tense, nocturnal, steady lurching groove, one continuous arrangement, 90 BPM, " + NO_VOX),
]


def _load_patcher():
    spec = importlib.util.spec_from_file_location("acestep_patch", ROOT / "workflows" / "api" / "acestep_patch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def entry(s, n_seeds=2):
    n = int(s["piece"][1:])
    dur = round(s["len_s"] + TAIL_S, 2)
    bar = 4 * 60.0 / s["bpm"]
    return {
        "piece": s["piece"], "slug": s["slug"], "text": s["text"], "len_s": s["len_s"],
        "duration": dur, "bpm": s["bpm"], "key": s["key"], "time_signature": "4",
        "instrumental": True, "lyrics": "[Instrumental]", "pace": s["pace"], "tags": s["tags"],
        "bar_s": round(bar, 3), "bars_in_len": round(s["len_s"] / bar, 2),
        "seeds": [n * 100 + 1 + i for i in range(n_seeds)],
        "graphs": [f"v001/graphs_pending/{s['piece']}_s{n * 100 + 1 + i}.json" for i in range(n_seeds)],
        "prefix": "projects/COMPETITIONS/motion-graphic/fusion/v001/motion-graphic_fusion_v001_{piece}_s{seed}",
    }


def make_graph(e, seed):
    P = _load_patcher()
    g = json.loads(BASE_GRAPH.read_text())
    g = P.patch(g, tags=e["tags"], lyrics=e["lyrics"], seed=seed, bpm=e["bpm"], key=e["key"], seconds=e["duration"],
                prefix=e["prefix"].format(piece=e["piece"], seed=seed))
    for k, v in g.items():
        if v["class_type"] == "TextEncodeAceStepAudio1.5":
            v["inputs"]["timesignature"] = e["time_signature"]
    return g


def build(n_seeds=2):
    PENDING.mkdir(parents=True, exist_ok=True)
    entries = [entry(s, n_seeds) for s in SONGS]
    for e in entries:
        for seed in e["seeds"]:
            (PENDING / f"{e['piece']}_s{seed}.json").write_text(json.dumps(make_graph(e, seed), indent=1))
    out = {"note": "ACE-Step 1.5 turbo songs for 15_fusion; P06 = user Song audio (see v001/local/audio/P06_notes.md). "
                   "Generated by tools/fusion/songs.py build.",
           "tail_s": TAIL_S, "models": EXPECT, "songs": entries}
    (FUSION / "songs.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {FUSION / 'songs.json'} and {sum(len(e['seeds']) for e in entries)} graphs in {PENDING}")


def check():
    data = json.loads((FUSION / "songs.json").read_text(encoding="utf-8"))
    bad = 0
    n = 0
    for e in data["songs"]:
        for seed, rel in zip(e["seeds"], e["graphs"]):
            p = FUSION / rel
            n += 1
            try:
                g = json.loads(p.read_text())
            except Exception as ex:
                print("FAIL load", p.name, ex)
                bad += 1
                continue
            probs = []
            have = {f"{v['class_type']}.{k}": val for v in g.values() for k, val in v["inputs"].items()}
            for k, want in EXPECT.items():
                if have.get(k) != want:
                    probs.append(f"{k}={have.get(k)!r} want {want!r}")
            te = [v["inputs"] for v in g.values() if v["class_type"] == "TextEncodeAceStepAudio1.5"][0]
            ks = [v["inputs"] for v in g.values() if v["class_type"] == "KSampler"][0]
            lat = [v["inputs"] for v in g.values() if v["class_type"] == "EmptyAceStep1.5LatentAudio"][0]
            sv = [v["inputs"] for v in g.values() if v["class_type"] == "SaveAudioAdvanced"][0]
            for name, a, b in (("bpm", te["bpm"], e["bpm"]), ("keyscale", te["keyscale"], e["key"]),
                               ("duration", te["duration"], e["duration"]), ("latent seconds", lat["seconds"], e["duration"]),
                               ("te seed", te["seed"], seed), ("ks seed", ks["seed"], seed),
                               ("tags", te["tags"], e["tags"]), ("lyrics", te["lyrics"], "[Instrumental]")):
                if a != b:
                    probs.append(f"{name}={a!r} want {b!r}")
            if not sv["filename_prefix"].endswith(f"{e['piece']}_s{seed}"):
                probs.append("prefix " + sv["filename_prefix"])
            for nid, v in g.items():   # every link points at an existing node
                for val in v["inputs"].values():
                    if isinstance(val, list) and len(val) == 2 and str(val[0]) not in g:
                        probs.append(f"dangling link {nid}->{val[0]}")
            if probs:
                bad += 1
                print("FAIL", p.name, "; ".join(probs))
    print(f"checked {n} graphs: {'OK' if not bad else str(bad) + ' failed'}")
    return bad


def table():
    for s in SONGS:
        e = entry(s)
        print(f"{e['piece']}  {e['bpm']:>3} bpm  {e['key']:<9} {e['duration']:>5} s  {e['pace']:<14} "
              + " ".join(e["tags"].split()[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "table"])
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.seeds)
    elif a.cmd == "check":
        sys.exit(1 if check() else 0)
    else:
        table()
