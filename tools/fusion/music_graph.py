"""music_graph.py - the exact-length music graph (skeleton -> ACE-Step audio-to-audio -> fit -> master) as API-format JSON.

    python tools/fusion/music_graph.py write        write app/graphs/music_exact_test.json + .app.json (overwrites these two)
    python tools/fusion/music_graph.py check        validate the files on disk (links, unique titles, sidecar bindings)

``a2a_chain(...)`` is the reusable part: it returns the node dict that replaces the text-to-music chain of the fusion graph
(see 16_app/MUSIC.md "Core graph change"). Node ids are strings, links are [id, output index], as in workflows/api/*.json.
Submit the test graph on the box with scripts/progress_standin.py (never a raw POST /prompt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "app" / "graphs"
NAME = "music_exact_test"
PREFIX = "projects/COMPETITIONS/motion-graphic/music_exact/mograph_music_exact"


def _n(cls, title, **inputs):
    return {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}


def ace_loaders(p: str = "ace_") -> dict:
    """the ACE-Step 1.5 loaders exactly as workflows/api/acestep_a2a.json (and fusion.json)."""
    return {
        f"{p}unet": _n("UNETLoader", "ACE-Step UNET", unet_name="acestep_v1.5_turbo.safetensors", weight_dtype="default"),
        f"{p}shift": _n("ModelSamplingAuraFlow", "ACE sampling shift", model=[f"{p}unet", 0], shift=3),
        f"{p}clip": _n("DualCLIPLoader", "ACE text encoders", clip_name1="qwen_0.6b_ace15.safetensors",
                       clip_name2="qwen_1.7b_ace15.safetensors", type="ace", device="default"),
        f"{p}vae": _n("VAELoader", "ACE VAE", vae_name="ace_1.5_vae.safetensors"),
    }


def a2a_chain(skel: str, seed_link, denoise_link, tags_link=None, p: str = "ace_", steps: int = 8) -> dict:
    """ScoreSkeleton node id ``skel`` (outputs: 0 audio, 1 music) -> StyleEncode -> VAEEncodeAudio -> KSampler -> VAEDecodeAudio
    -> MusicFit. Returns the nodes; the fitted audio / music are ``[f"{p}fit", 0]`` / ``[f"{p}fit", 1]``.
    ``seed_link`` / ``denoise_link``: a value or a link. ``tags_link``: a link to a STRING (music prompt) or None (tags field empty)."""
    enc_in = {"clip": [f"{p}clip", 0], "music": [skel, 1], "tags": tags_link if tags_link is not None else ""}
    return {
        f"{p}enc": _n("MographMusicStyleEncode", "Music style encode (a2a)", **enc_in),
        f"{p}zero": _n("ConditioningZeroOut", "ACE negative", conditioning=[f"{p}enc", 0]),
        f"{p}venc": _n("VAEEncodeAudio", "Skeleton to latent", audio=[skel, 0], vae=[f"{p}vae", 0]),
        f"{p}ks": _n("KSampler", "ACE sampler (a2a)", model=[f"{p}shift", 0], positive=[f"{p}enc", 0],
                     negative=[f"{p}zero", 0], latent_image=[f"{p}venc", 0], seed=seed_link, steps=steps, cfg=1.0,
                     sampler_name="euler", scheduler="simple", denoise=denoise_link),
        f"{p}dec": _n("VAEDecodeAudio", "ACE decode", samples=[f"{p}ks", 0], vae=[f"{p}vae", 0]),
        f"{p}fit": _n("MographMusicFit", "Music fit", audio=[f"{p}dec", 0], music=[skel, 1], skeleton=[skel, 0],
                      xfade_ms=80, tail_fade_ms=30, match_loudness=False),
    }


# ACE-Step 1.5 XL BASE text-to-music: the APPROVED route (BUILT.md "Music"; 15_fusion/v004/make_music.py base_nolm_g2). Never turbo here.
T2M_UNET = "acestep_v1.5_xl_base_bf16.safetensors"
T2M_CLIPS = ("qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors")
T2M_STEPS, T2M_CFG = 50, 6.0


def t2m_chain(request_link, seed_link, brief_link=None, p: str = "t2m_", live_preview: bool = True) -> dict:
    """request -> ACE XL base text-to-music (no LM, no audio codes, 50 steps, cfg 6, euler / simple, shift 3) at length + 4 s ->
    T2MCut (exact length + the beat grid measured on the song) -> TameHighs. The cut audio / music are ``[f"{p}tame", 0]`` /
    ``[f"{p}cut", 1]``; the music-first plan is built from that grid.
    ``live_preview`` (default on): a MographLiveWaveformPreview model patch between the shift node and the sampler (``{p}live``): the web page
    draws the song's waveform forming step by step (websocket event motngrfx.waveform); the render is bit-identical with it on or off."""
    enc = {"clip": [f"{p}clip", 0], "request": request_link}
    if brief_link:
        enc["brief"] = brief_link
    ks_model = [f"{p}live", 0] if live_preview else [f"{p}shift", 0]
    g = {
        f"{p}unet": _n("UNETLoader", "ACE-Step XL base", unet_name=T2M_UNET, weight_dtype="default"),
        f"{p}shift": _n("ModelSamplingAuraFlow", "ACE sampling shift", model=[f"{p}unet", 0], shift=3),
        f"{p}clip": _n("DualCLIPLoader", "ACE text encoders", clip_name1=T2M_CLIPS[0], clip_name2=T2M_CLIPS[1], type="ace", device="default"),
        f"{p}vae": _n("VAELoader", "ACE VAE", vae_name="ace_1.5_vae.safetensors"),
        f"{p}enc": _n("MographMusicT2MEncode", "Music T2M encode", **enc),
        f"{p}zero": _n("ConditioningZeroOut", "ACE negative", conditioning=[f"{p}enc", 0]),
        f"{p}lat": _n("EmptyAceStep1.5LatentAudio", "Empty latent (length + 4 s)", seconds=[f"{p}enc", 1], batch_size=1),
        f"{p}ks": _n("KSampler", "ACE sampler (text-to-music)", model=ks_model, positive=[f"{p}enc", 0], negative=[f"{p}zero", 0],
                     latent_image=[f"{p}lat", 0], seed=seed_link, steps=T2M_STEPS, cfg=T2M_CFG, sampler_name="euler", scheduler="simple",
                     denoise=1.0),
        f"{p}dec": _n("VAEDecodeAudio", "ACE decode", samples=[f"{p}ks", 0], vae=[f"{p}vae", 0]),
        f"{p}cut": _n("MographMusicT2MCut", "Music T2M cut", audio=[f"{p}dec", 0], request=request_link, plan_json=[f"{p}enc", 2]),
        f"{p}tame": _n("MographTameHighs", "Tame highs", audio=[f"{p}cut", 0]),
    }
    if live_preview:
        g[f"{p}live"] = _n("MographLiveWaveformPreview", "Live waveform preview (per step)", model=[f"{p}shift", 0], vae=[f"{p}vae", 0],
                           enabled=True, bins=800)
    return g


def build(length=12, bpm=0, music="", denoise=0.5, seed=1) -> dict:
    g = {
        "length": _n("PrimitiveFloat", "IN length", value=float(length)),
        "bpm": _n("PrimitiveInt", "IN bpm", value=int(bpm)),
        "music": _n("PrimitiveStringMultiline", "IN music", value=music),
        "denoise": _n("PrimitiveFloat", "IN denoise", value=float(denoise)),
        "seed": _n("PrimitiveInt", "IN seed", value=int(seed)),
    }
    g.update(ace_loaders())
    g["skel"] = _n("MographScoreSkeleton", "Score skeleton", length_s=["length", 0], music_prompt=["music", 0], bpm=["bpm", 0],
                   mood="auto", energy=-1.0, key="", meter=4, seed=["seed", 0], hits="", sections="")
    g.update(a2a_chain("skel", ["seed", 0], ["denoise", 0], tags_link=["music", 0]))
    g["master"] = _n("MographMaster", "Master", audio=["ace_fit", 0], music=["ace_fit", 1], preset="club", target_lufs=-14.0,
                     width=1.25)
    g["out_skeleton"] = _n("SaveAudioAdvanced", "OUT skeleton", audio=["skel", 0], filename_prefix=PREFIX + "_skeleton", format="flac")
    g["out_raw"] = _n("SaveAudioAdvanced", "OUT restyled raw", audio=["ace_dec", 0], filename_prefix=PREFIX + "_raw", format="flac")
    g["out_final"] = _n("SaveAudioAdvanced", "OUT music", audio=["master", 0], filename_prefix=PREFIX, format="flac")
    return g


def sidecar() -> dict:
    return {
        "title": "Exact-length music test (GPU box)",
        "order": 60,
        "verb": "Make music",
        "running_verb": "Making music",
        "cost_line": "Runs on the GPU box (ACE-Step 1.5)",
        "note": "A composed skeleton of exactly the chosen length is restyled by ACE-Step audio-to-audio. Denoise 0.3 keeps the "
                "skeleton's timing tight, 0.7 gives more ACE character. Outputs: the skeleton, ACE's raw audio, the fitted + mastered music.",
        "fields": [
            {"id": "length", "label": "Length", "kind": "select", "node_title": "IN length", "input_name": "value", "default": 12,
             "options": [{"value": 8, "label": "8 s"}, {"value": 12, "label": "12 s"}, {"value": 16, "label": "16 s"},
                         {"value": 24, "label": "24 s"}]},
            {"id": "bpm", "label": "Tempo hint (bpm)", "kind": "number", "node_title": "IN bpm", "input_name": "value",
             "optional": True, "default": 0, "min": 0, "max": 200,
             "hint": "0 = from the music prompt, else the mood. The tempo used is the nearest one that fits the length in whole bars."},
            {"id": "music", "label": "Music prompt", "kind": "textarea", "node_title": "IN music", "input_name": "value",
             "optional": True, "default": "", "rows": 3,
             "placeholder": "e.g. deep house, warm Rhodes chords, soft kick, 122 bpm, A minor"},
            {"id": "denoise", "label": "Denoise (0.3 tight - 0.7 free)", "kind": "number", "node_title": "IN denoise",
             "input_name": "value", "optional": True, "default": 0.5, "min": 0.2, "max": 0.9},
            {"id": "seed", "label": "Seed", "kind": "seed", "node_title": "IN seed", "input_name": "value", "optional": True},
        ],
        "missing": {},
        "outputs": [
            {"node_title": "OUT skeleton", "kind": "audio", "label": "Skeleton (before ACE)"},
            {"node_title": "OUT restyled raw", "kind": "audio", "label": "ACE output (unfitted)"},
            {"node_title": "OUT music", "kind": "audio", "label": "Fitted + mastered music", "final": True},
        ],
        "stage_labels": {"IN length": "Read the request", "Score skeleton": "Compose the skeleton",
                         "Music style encode (a2a)": "Music prompt", "Skeleton to latent": "Encode the skeleton",
                         "ACE sampler (a2a)": "Restyle with ACE-Step", "ACE decode": "Decode", "Music fit": "Fit to exact length",
                         "Master": "Master the audio", "OUT music": "Save audio"},
    }


def check(graph: dict | None = None, side: dict | None = None) -> list[str]:
    g = graph if graph is not None else json.loads((GRAPHS / f"{NAME}.json").read_text(encoding="utf-8"))
    s = side if side is not None else json.loads((GRAPHS / f"{NAME}.app.json").read_text(encoding="utf-8"))
    bad = []
    titles = [v.get("_meta", {}).get("title") for v in g.values()]
    for t in set(titles):
        if titles.count(t) > 1:
            bad.append(f"duplicate title {t!r}")
    for nid, v in g.items():
        for k, val in v["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int) and str(val[0]) not in g:
                bad.append(f"dangling link {nid}.{k} -> {val[0]}")
    for f in s["fields"]:
        if f["node_title"] not in titles:
            bad.append(f"field {f['id']}: no node titled {f['node_title']!r}")
        else:
            node = next(v for v in g.values() if v.get("_meta", {}).get("title") == f["node_title"])
            if f["input_name"] not in node["inputs"]:
                bad.append(f"field {f['id']}: {f['node_title']} has no input {f['input_name']!r}")
    for o in s["outputs"]:
        if o["node_title"] not in titles:
            bad.append(f"output {o['node_title']!r} missing")
    return bad


def write() -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    (GRAPHS / f"{NAME}.json").write_text(json.dumps(build(), indent=1), encoding="utf-8")
    (GRAPHS / f"{NAME}.app.json").write_text(json.dumps(sidecar(), indent=1), encoding="utf-8")
    print(f"wrote {GRAPHS / (NAME + '.json')} and {NAME}.app.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "write":
        write()
    problems = check()
    print("OK" if not problems else "\n".join(problems))
    sys.exit(1 if problems else 0)
