"""Score skeleton: exact-length composed stems with a known beat grid, the fit step, and the three music_score nodes (CPU).
Run: python -m pytest tests/test_score_skeleton.py -q"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph import score_skeleton as sk  # noqa: E402

FPS = 24
LENGTHS = (8, 12, 16, 24)
HINT = {8: 100, 12: 80, 16: 122, 24: 124}          # tempo hints per length
MUSIC_KEYS = {"music_end", "cut", "nframes", "fade", "bpm", "bar_len", "key", "beats", "downbeats", "chord_changes", "curves",
              "fps", "src"}                         # what MographMusicCut produces


@pytest.fixture(scope="module")
def pieces():
    out = {}
    for L in LENGTHS:
        mood = "driving" if HINT[L] > 110 else "calm"
        hits = [L * 0.5 + 0.13, 2.9]               # off the beat on purpose: must be snapped
        out[L] = sk.compose(n_frames=L * FPS, bpm=HINT[L], mood=mood, seed=7, hits=hits)
    return out


def _bar_end(g):
    return g["bars"][-1]["end"]


# ----------------------------------------------------------------------------------------------- exact length
@pytest.mark.parametrize("L", LENGTHS)
def test_exact_sample_count(pieces, L):
    wave, g = pieces[L]
    assert wave.shape == (2, L * 48000) and wave.dtype == np.float32
    assert g["n_samples"] == L * 48000 and g["n_frames"] == L * FPS and g["length_s"] == L
    assert np.isfinite(wave).all() and np.abs(wave).max() <= 1.0
    assert np.abs(wave).max() > 0.05                                    # not silence
    assert np.abs(wave[:, -48:]).max() < 0.02                          # the last ms has faded out (no cut-off click)


def test_other_rates_and_odd_lengths():
    w, g = sk.compose(n_frames=288, sr=44100, backend="synth", seed=1)
    assert w.shape == (2, round(12 * 44100)) and g["n_samples"] == w.shape[1]
    w, g = sk.compose(length_s=10.04, backend="synth", seed=1)         # not whole frames: rounded to 241 frames
    assert g["n_frames"] == 241 and w.shape[1] == round(241 / 24 * 48000)


# ----------------------------------------------------------------------------------------------- grid
@pytest.mark.parametrize("L", LENGTHS)
def test_grid_consistent_with_bpm(pieces, L):
    _, g = pieces[L]
    beat = 60.0 / g["bpm"]
    assert g["beat_s"] == pytest.approx(beat, rel=1e-12)
    assert len(g["beats"]) == g["n_bars"] * g["meter"]
    assert np.allclose(np.diff(g["beats"]), beat, atol=1e-9)
    assert np.allclose(np.diff(g["downbeats"]), g["meter"] * beat, atol=1e-9)
    assert len(g["downbeats"]) == g["n_bars"] and g["bar_len"] == pytest.approx(g["meter"] * beat)
    assert set(g["downbeats"]) <= set(g["beats"])
    assert g["beats"][0] == 0.0 and g["beats"][-1] + beat == pytest.approx(L)
    assert 40 <= g["bpm"] <= 200


@pytest.mark.parametrize("L", LENGTHS)
def test_ends_on_a_bar_line(pieces, L):
    _, g = pieces[L]
    assert g["n_bars"] * g["bar_len"] == pytest.approx(L, abs=1e-9)
    assert _bar_end(g) == pytest.approx(L, abs=1e-9)
    assert g["n_bars"] * g["meter"] * 60.0 / g["bpm"] == pytest.approx(L, abs=1e-9)
    assert g["sections"][-1]["end"] == pytest.approx(L) and g["sections"][0]["start"] == 0.0
    for a, b in zip(g["sections"], g["sections"][1:]):                  # contiguous, on bar lines
        assert a["end"] == pytest.approx(b["start"])
    for s in g["sections"]:
        assert s["start"] / g["bar_len"] == pytest.approx(round(s["start"] / g["bar_len"]), abs=1e-9)


def test_tempo_nudged_within_window_not_trimmed():
    for L, hint, want in [(12, 80, 80.0), (16, 122, 120.0), (24, 124, 120.0), (12, 100, 100.0), (8, 100, 90.0)]:
        p = sk.plan_grid(n_frames=L * FPS, bpm=hint)
        assert p["bpm"] == pytest.approx(want), (L, hint, p["bpm"])
        assert p["n_bars"] * p["meter"] * 60 / p["bpm"] == pytest.approx(L)
        assert abs(p["bpm"] / hint - 1) <= 0.15 and p["in_window"]
    p = sk.plan_grid(n_frames=7 * FPS, bpm=200, meter=3)              # 3/4: still a whole number of bars
    assert p["n_bars"] * 3 * 60 / p["bpm"] == pytest.approx(7)


@pytest.mark.parametrize("L", LENGTHS)
def test_hits_land_on_beats(pieces, L):
    _, g = pieces[L]
    assert g["hits"] and len(g["hits_requested"]) == 2
    for h in g["hits"]:
        assert min(abs(h - b) for b in g["beats"]) < 1e-9
        assert 0.0 <= h < L
    for r in g["hits_requested"]:                                       # snapped to the NEAREST beat
        nearest = min(g["beats"], key=lambda b: abs(b - r))
        assert nearest in g["hits"]


def test_hits_out_of_range_dropped_and_deduped():
    _, g = sk.compose(n_frames=288, hits=[-1.0, 3.0, 3.01, 99.0], backend="synth", seed=2)
    assert len(g["hits"]) == 1 and g["hits_requested"] == [-1.0, 3.0, 3.01, 99.0]


def test_sections_snap_and_validate():
    _, g = sk.compose(n_frames=288, bpm=120, sections=[(0, 2.2, "intro"), (2.2, 6.1, "build"), (6.1, 12, "main")], backend="synth", seed=1)
    assert [(s["role"], s["bar0"], s["bar1"]) for s in g["sections"]] == [("intro", 0, 1), ("build", 1, 3), ("drop", 3, 6)]
    with pytest.raises(ValueError):
        sk.compose(n_frames=288, sections=[(0, 4, "chorus?")], backend="synth")
    assert sk.parse_sections("intro:0-3, drop:3-12") == [(0.0, 3.0, "intro"), (3.0, 12.0, "drop")]
    assert sk.parse_sections("") is None and sk.parse_hits("1.5, 3 4;5") == [1.5, 3.0, 4.0, 5.0]


def test_key_modes_and_ace_keyscale():
    pc, mode, name = sk.parse_key("F# dorian")
    assert (pc, mode, name) == (6, "dorian", "F#") and sk.ace_keyscale(name, mode) == "F# minor"
    assert sk.ace_keyscale("Bb", "lydian") == "Bb major"
    with pytest.raises(ValueError):
        sk.parse_key("H minor")
    _, g = sk.compose(n_frames=288, key="D minor", mood="calm", backend="synth", seed=1)
    assert g["key"] == "D minor" and g["key_ace"] == "D minor"


# ----------------------------------------------------------------------------------------------- determinism
def test_deterministic_per_seed():
    a, ga = sk.compose(n_frames=192, bpm=120, mood="dark", seed=11, hits=[4.0])
    b, gb = sk.compose(n_frames=192, bpm=120, mood="dark", seed=11, hits=[4.0])
    c, gc = sk.compose(n_frames=192, bpm=120, mood="dark", seed=12, hits=[4.0])
    assert np.array_equal(a, b) and json.dumps(ga, sort_keys=True) == json.dumps(gb, sort_keys=True)
    assert not np.array_equal(a, c)
    assert ga["bpm"] == gc["bpm"]                                       # the grid does not depend on the seed
    s1, _ = sk.compose(n_frames=192, bpm=120, mood="dark", seed=11, backend="synth")
    s2, _ = sk.compose(n_frames=192, bpm=120, mood="dark", seed=11, backend="synth")
    assert np.array_equal(s1, s2) and s1.shape == a.shape


def test_every_mood_renders_exact(pieces):
    for mood in sk.MOODS:
        w, g = sk.compose(n_frames=8 * FPS, mood=mood, seed=3, backend="synth")
        assert w.shape == (2, 8 * 48000) and g["mood"] == mood
    w, g = sk.compose(n_frames=8 * FPS, mood="hype", seed=3, backend="synth")           # app mood word
    assert g["mood"] == "driving"
    with pytest.raises(ValueError):
        sk.compose(n_frames=8 * FPS, mood="nope")


def test_warm_palettes_are_full():
    """calm / epic: a pad in EVERY bar, strings under the piano (soft), a low sustain on the root, timpani on the downbeat (filmic)."""
    from mograph import score_skeleton as S
    for mood, palette in (("calm", "soft"), ("epic", "filmic")):
        plan = S.plan_grid(n_frames=12 * FPS, mood=mood)
        g = dict(plan, sections=S._norm_sections(None, plan["n_bars"], plan["bar_s"], mood, 0.5), hits=[])
        tonic, mode, _ = S.parse_key(None, mood, 3)
        chords = S._chords(plan["n_bars"], mood, tonic, mode, 3)
        P, _, _ = S._arrange(g, mood, palette, 0.5, tonic, mode, chords, 3)
        bars = plan["n_bars"]
        pad_bars = {round(e.t / plan["bar_s"]) for e in P["pad"]}
        assert pad_bars == set(range(bars)), (mood, pad_bars)                   # was every second bar
        assert {round(e.t / plan["bar_s"]) for e in P["low"]} >= set(range(bars - 1))
        assert all(e.midi < 66 for e in P["low"])
        if palette == "soft":
            assert {round(e.t / plan["bar_s"]) for e in P["strings"]} == set(range(bars))
            assert max(e.midi for e in P["strings"]) < 70                       # low voicing, under the piano
        else:
            assert len(P["timp"]) >= bars // 2 and all(abs(e.t / plan["bar_s"] - round(e.t / plan["bar_s"])) < 1e-6 for e in P["timp"])
    P, _, _ = S._arrange(dict(plan, sections=g["sections"], hits=[]), "driving", "electronic", 0.8, 9, "minor",
                         S._chords(plan["n_bars"], "driving", 9, "minor", 1), 1)
    assert not P["strings"] and not P["low"] and not P["timp"]                  # electronic palette is unchanged


def test_warm_pad_target_is_louder():
    assert S_TARGET("pad", "soft") == -19.0 and S_TARGET("pad", "filmic") == -19.0 and S_TARGET("pad", "electronic") == -24.0


def S_TARGET(part, palette):
    return (sk._TARGET if palette == "electronic" else sk._TARGET_WARM)[part]


def test_skeleton_tempo_is_recoverable_by_the_tracker(pieces):
    """sanity of the 'grid is ground truth' claim: the existing beat tracker, given the hint, finds the grid tempo."""
    from tools.fusion import audio_cut
    wave, g = pieces[16]
    ana = audio_cut.analyse_wave(wave, 48000, g["bpm"], FPS)
    assert ana["bpm"] == pytest.approx(g["bpm"], rel=0.02)


# ----------------------------------------------------------------------------------------------- music dict
def test_music_dict_has_musiccut_keys(pieces):
    wave, g = pieces[12]
    m = sk.music_dict(g, wave, 48000, FPS)
    assert MUSIC_KEYS <= set(m)
    assert m["nframes"] == 12 * FPS and m["cut"] == 12.0 and m["bpm"] == g["bpm"] and m["bar_len"] == g["bar_len"]
    assert m["beats"][0] == 0.0 and m["beats"][-1] >= 12 + 3.9 and m["downbeats"][-1] >= 12 + 3.9    # extended like MusicCut's
    assert 12.0 in [round(d, 6) for d in m["downbeats"]]                                   # the cut is a downbeat
    assert set(m["curves"]) == {"rms", "sub", "low", "mid", "high", "onset"}
    assert all(len(v) == m["nframes"] + 1 and 0.0 <= min(v) and max(v) <= 1.0 for v in m["curves"].values())
    json.dumps(m)                                                                          # plain data only


# ----------------------------------------------------------------------------------------------- fit
def test_fit_exact_all_cases(pieces):
    wave, g = pieces[12]
    n = g["n_samples"]
    # longer -> cut + fade
    y, rep = sk.fit_exact(np.concatenate([wave, wave[:, :9000]], axis=1), 48000, n)
    assert y.shape == (2, n) and rep["mode"] == "cut" and np.abs(y[:, -8:]).max() < 1e-3
    # a little shorter (ACE decode of 1920-sample latent frames) with the skeleton: tail comes from the skeleton
    y, rep = sk.fit_exact(wave[:, : n - 1500], 48000, n, skeleton=wave)
    assert y.shape == (2, n) and rep["mode"] in ("cut", "crossfade")
    # silent tail (ACE pads short renders with zeros) -> crossfade into the skeleton, not silence
    x = wave.copy()
    x[:, 300000:] = 0.0
    y, rep = sk.fit_exact(x, 48000, n, skeleton=wave)
    assert y.shape == (2, n) and rep["mode"] == "crossfade" and 290000 < rep["join"] <= 300000
    assert np.abs(y[:, 400000:500000]).max() > 0.05 and np.abs(y[:, :200000] - x[:, :200000]).max() < 1e-6
    assert np.abs(y[:, 296000:304000]).max() <= 1.0
    # without a skeleton: fade out + silence, still exact
    y, rep = sk.fit_exact(x, 48000, n)
    assert y.shape == (2, n) and rep["mode"] == "pad" and np.abs(y[:, 310000:]).max() == 0.0
    # other sample rate, mono
    y, rep = sk.fit_exact(wave[0, ::2][None], 24000, n, out_sr=48000)
    assert y.shape == (2, n)
    # loudness match to the skeleton
    y, rep = sk.fit_exact(wave * 0.1, 48000, n, skeleton=wave, loudness=True)
    assert rep["gain_db"] > 6.0 and np.abs(y).max() <= 10 ** (-1.0 / 20) + 1e-6
    with pytest.raises(ValueError):
        sk.fit_exact(np.zeros((2, 0), np.float32), 48000, n)


# ----------------------------------------------------------------------------------------------- the graph
def test_music_graph_on_disk():
    sys.path.insert(0, ROOT)
    from tools.fusion import music_graph as mg
    assert mg.check() == []
    g = json.load(open(os.path.join(ROOT, "app", "graphs", "music_exact_test.json"), encoding="utf-8"))
    cls = {v["class_type"] for v in g.values()}
    for need in ("MographScoreSkeleton", "MographMusicStyleEncode", "MographMusicFit", "MographMaster", "VAEEncodeAudio", "KSampler",
                 "VAEDecodeAudio", "UNETLoader", "DualCLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "ConditioningZeroOut"):
        assert need in cls
    ks = next(v for v in g.values() if v["class_type"] == "KSampler")["inputs"]
    assert ks["steps"] == 8 and ks["cfg"] == 1.0 and ks["sampler_name"] == "euler" and ks["scheduler"] == "simple"
    assert ks["denoise"] == ["denoise", 0] and g["denoise"]["inputs"]["value"] == 0.5
    side = json.load(open(os.path.join(ROOT, "app", "graphs", "music_exact_test.app.json"), encoding="utf-8"))
    assert {f["id"] for f in side["fields"]} == {"length", "bpm", "music", "denoise", "seed"}


# ----------------------------------------------------------------------------------------------- nodes
def _import_comfy():
    cands = [os.environ.get("MOGRAPH_COMFYUI_PATH"), os.path.abspath(os.path.join(ROOT, "..", "ComfyUI")),
             os.path.abspath(os.path.join(ROOT, "..", "..", "..", "ComfyUI")), os.path.abspath(os.path.join(ROOT, "..", ".."))]
    for c in filter(None, cands):
        if os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            try:
                import comfy_api.latest  # noqa: F401
                return c
            except Exception as e:      # noqa: BLE001
                pytest.skip(f"comfy_api import failed: {e}")
    pytest.skip("no ComfyUI checkout found (set MOGRAPH_COMFYUI_PATH)")


def test_node_schemas_load():
    _import_comfy()
    from nodes_fusion import music_score as nodes
    ids = [c.define_schema().node_id for c in nodes.NODES]
    assert ids == ["MographScoreSkeleton", "MographMusicStyleEncode", "MographMusicFit", "MographMusicRoute"]
    for cls in nodes.NODES:
        cls.GET_SCHEMA()
    import nodes_fusion
    assert "music_score" in nodes_fusion.SUBMODULES
    assert {c.__name__ for c in nodes.NODES} <= {c.__name__ for c in nodes_fusion.NODES}, nodes_fusion.FAILED


def test_landscape_words_pick_the_orchestral_mood():
    _import_comfy()
    from nodes_fusion import music_score as nodes
    for text in ("a mountain ridge at dawn", "valley mist, glacier light", "the horizon", "canyon vista"):
        assert nodes._auto_mood({"scene": text, "words": ""}, "") == "epic", text
    assert nodes._auto_mood({"scene": "mountain club night", "words": ""}, "") != "epic"      # dance words keep the electronic pick
    assert "orchestral" in nodes._MOOD_TAGS["epic"] and "timpani" in nodes._MOOD_TAGS["epic"] and "brass" in nodes._MOOD_TAGS["epic"]
    assert "warm string pad" in nodes._MOOD_TAGS["calm"] and "cello" in nodes._MOOD_TAGS["calm"]


def test_nodes_skeleton_fit_plan_chain():
    """ScoreSkeleton -> (stand-in for ACE: the skeleton, 1500 samples short) -> MusicFit -> Master -> PiecePlan, all CPU."""
    _import_comfy()
    import torch
    from nodes_fusion import music_score as nodes
    from nodes_fusion import audio as audio_nodes
    from nodes_fusion import plan as plan_nodes
    req = {"words": "AFTER HOURS / THE CITY KEEPS DREAMING", "scene": "", "music": "deep house, warm Rhodes, 122 bpm, A minor",
           "length": 12, "look": "auto", "seed": 4, "quality": "draft"}
    audio, music, grid_json = nodes.MographScoreSkeleton.execute(req).args
    assert audio["waveform"].shape == (1, 2, 12 * 48000) and audio["sample_rate"] == 48000
    gj = json.loads(grid_json)
    assert gj["n_samples"] == 12 * 48000 and gj["key"] == "A minor" and gj["bpm"] == pytest.approx(120.0)      # 122 -> 120 (5 bars... 6 bars of 4)
    assert MUSIC_KEYS <= set(music) and music["src"] == "score" and music["cut"] == 12.0
    # explicit fields, no request
    a2, m2, _ = nodes.MographScoreSkeleton.execute(None, 16.0, "calm piano 70 bpm", 0, "auto", -1.0, "", 4, 3, "4.1", "").args
    assert a2["waveform"].shape == (1, 2, 16 * 48000) and m2["grid"]["mood"] == "calm" and m2["grid"]["hits"] == [4.0 if abs(4.1 - 4.0) < abs(4.1 - 4.5) else 4.5]
    # restyled stand-in: a little short, with a silent tail
    w = audio["waveform"].clone()[..., : 12 * 48000 - 1500]
    w[..., 400000:] = 0.0
    fitted, music2 = nodes.MographMusicFit.execute({"waveform": w, "sample_rate": 48000}, music, audio, 80, 30, False).args
    assert fitted["waveform"].shape == (1, 2, 12 * 48000) and fitted["sample_rate"] == 48000
    assert music2["fit"]["mode"] == "crossfade" and music2["src"] == "score+ace"
    for k in ("bpm", "beats", "downbeats", "bar_len", "key", "chord_changes", "cut", "nframes"):
        assert music2[k] == music[k]                                                       # the skeleton grid is kept
    assert len(music2["curves"]["onset"]) == music2["nframes"] + 1
    (mastered,) = audio_nodes.MographMaster.execute(fitted, music2, "club", -14.0, 1.25).args
    assert mastered["waveform"].shape[-1] == 12 * 48000                                    # Master trims to cut: still exact
    # PiecePlan consumes it unchanged
    piece, piece_json, *_ = plan_nodes.MographPiecePlan.execute(req, music2, "").args
    assert piece["layout"] and sum(1 for _ in piece["layout"]) >= 2
    with pytest.raises(ValueError):
        nodes.MographMusicFit.execute({"waveform": torch.zeros(1, 2, 0), "sample_rate": 48000}, music, None, 80, 30, False)


def test_style_encode_text_and_args():
    _import_comfy()
    from nodes_fusion import music_score as nodes

    class FakeClip:
        def tokenize(self, text, **kw):
            self.text, self.kw = text, kw
            return "tokens"

        def encode_from_tokens_scheduled(self, tokens):
            assert tokens == "tokens"
            return [["cond", {}]]

    _, g = sk.compose(n_frames=288, bpm=122, key="F minor", mood="driving", seed=9, backend="synth")
    music = sk.music_dict(g)
    clip = FakeClip()
    nodes.MographMusicStyleEncode.execute(clip, music, "deep house, warm Rhodes, 122 bpm, no vocals").args
    kw = clip.kw
    assert clip.text.count("BPM") == 1 and "120 BPM" in clip.text and "122" not in clip.text      # the grid's tempo, not the prompt's
    assert (kw["bpm"], kw["duration"], kw["keyscale"], kw["timesignature"]) == (120, 12.0, "F minor", 4)
    assert kw["generate_audio_codes"] is False and kw["lyrics"] == "[Instrumental]" and kw["seed"] == g["seed"]
    nodes.MographMusicStyleEncode.execute(clip, music, "").args
    assert "no vocals" in clip.text and "BPM" in clip.text and "driving electronic" in clip.text      # mood fallback


def test_route_picks_one_branch_lazily():
    _import_comfy()
    import torch
    from nodes_fusion import music_score as nodes
    user = {"waveform": torch.zeros(1, 2, 4800), "sample_rate": 48000}
    req_user = {"words": "A", "length": 12, "audio": user}
    req_none = {"words": "A", "length": 12, "audio": None}
    chk = nodes.MographMusicRoute.check_lazy_status
    assert chk(req_user) == ["user_audio", "user_music"]                              # only the user branch is requested ...
    assert chk(req_none) == ["score_audio", "score_music"]                            # ... or only the score branch
    assert chk(req_none, score_audio={"x": 1}) == ["score_music"] and chk(req_none, None, None, "a", "m") == []
    assert chk(req_user, "ua", "um") == []
    a, m = nodes.MographMusicRoute.execute(req_none, None, None, "sa", "sm").args
    assert (a, m) == ("sa", "sm")
    a, m = nodes.MographMusicRoute.execute(req_user, "ua", "um", "sa", "sm").args
    assert (a, m) == ("ua", "um")
    with pytest.raises(ValueError):
        nodes.MographMusicRoute.execute(req_none, "ua", "um", None, None)
    with pytest.raises(ValueError):
        nodes.MographMusicRoute.execute(req_user, None, None, "sa", "sm")


def test_skeleton_node_uses_the_music_prompt_and_brief():
    _import_comfy()
    from nodes_fusion import music_score as nodes
    base = {"words": "AFTER HOURS", "scene": "", "length": 8, "look": "auto", "seed": 2, "quality": "draft"}
    _, m, _ = nodes.MographScoreSkeleton.execute(dict(base, music="slow ambient piano, 72 bpm, D minor")).args
    assert m["grid"]["key"] == "D minor" and m["grid"]["mood"] == "calm" and m["bpm"] in (60.0, 75.0, 90.0)   # 8 s: 72 -> nearest fit
    _, m, _ = nodes.MographScoreSkeleton.execute(dict(base, music="")).args                                   # no prompt: rules from the words
    assert m["nframes"] == 8 * 24 and m["grid"]["mood"] in sk.MOODS
