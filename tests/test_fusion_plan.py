"""Fusion planner: tools/fusion/plan_rules.py, mograph/fusion_brief.py, recipes and the plan nodes (CPU only).
Run: python -m pytest tests/test_fusion_plan.py -q      (MOGRAPH_FAST=1 skips the 35 s full-camera scene build)"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph import fusion_brief as fb  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402

P02_ANALYSIS = os.path.join(ROOT, "15_fusion", "v002", "local", "pieces", "P02", "analysis.json")
WORDS = "AFTER HOURS / THE CITY KEEPS DREAMING"
REQ = {"words": WORDS, "scene": "", "music": "", "length": 12, "look": "auto", "seed": 5, "quality": "final"}


@pytest.fixture(scope="module")
def p02_music():
    if not os.path.isfile(P02_ANALYSIS):
        pytest.skip("P02 analysis.json not present")
    return json.load(open(P02_ANALYSIS, encoding="utf-8"))


@pytest.fixture(scope="module")
def planned(p02_music):
    return pr.plan_piece(REQ, p02_music)


def synth_music(bpm: float, length: float, first: float = 0.2) -> dict:
    """a MOGRAPH_MUSIC dict on a regular grid whose cut is the downbeat nearest ``length`` seconds."""
    iv = 60.0 / bpm
    bar = 4 * iv
    beats = [first + i * iv for i in range(int(length / iv) + 40)]
    down = beats[::4]
    cut = min([d for d in down if d <= length + bar / 2], key=lambda d: abs(d - length))
    nf = math.ceil(cut * 24 - 1e-6)
    return dict(cut=round(cut, 4), nframes=nf, bpm=bpm, bar_len=bar, key="F minor", beats=beats, downbeats=down,
                chord_changes=[], curves={"onset": [0.15] * (nf + 1)}, music_end=cut + 1, fade=0.5, fps=24)


# ----------------------------------------------------------------------------------------------- determinism, timeline
def test_plan_is_deterministic(p02_music, planned):
    again = pr.plan_piece(REQ, p02_music)
    assert pr.piece_hash(again) == pr.piece_hash(planned)
    assert pr.piece_json(again) == pr.piece_json(planned)
    other = pr.plan_piece(dict(REQ, seed=6), p02_music, dry=False)
    assert pr.piece_hash(other) != pr.piece_hash(planned)


def test_words_become_lines_in_order(planned):
    texts = [w["text"] for w in planned["type"]["words"]]
    assert " ".join(texts).split() == WORDS.replace("/", " ").split()
    assert len(planned["type"]["words"]) <= 6
    assert len({w["font"] for w in planned["type"]["words"]}) <= 3


def test_hits_are_on_downbeats_and_fit_before_the_signoff(planned):
    d = pr.dry_run(planned)
    assert d["notes"] == []
    cut = planned["audio"]["cut"]
    assert d["last_end"] <= cut - 0.3
    beats = planned["type"]["beats"]       # one lockup per shot: a hit is on a downbeat when the shot has room, else on a beat (tests/test_owner.py)
    for w in d["words"]:
        assert min(abs(w["hit"] - x) for x in beats) <= 1 / 24 + 1e-6, w
    assert planned["signoff"]["t0"] >= d["last_end"] - 0.6      # the signoff fades in as the last word leaves (no dead air)


@pytest.mark.skipif(os.environ.get("MOGRAPH_FAST") == "1", reason="MOGRAPH_FAST=1")
def test_build_scene_dry_run_has_no_timeline_note(planned):
    """the real thing: spec.build_scene on the planned type spec at 576x1024, camera plan included."""
    from tools.fusion import spec
    scene = spec.build_scene(copy.deepcopy(planned["type"]), size=(576, 1024))
    assert not [n for n in scene.info["notes"] if "timeline" in n and "longer" in n], scene.info["notes"]
    assert scene.info["notes"] == []


def test_dry_run_without_camera_has_no_notes_for_all_looks(p02_music):
    for look in ("auto", "strand", "mech", "slab", "stack", "ring", "voxel", "particles", "hollow_tubes"):
        piece = pr.plan_piece(dict(REQ, look=look), p02_music)
        assert pr.dry_run(piece)["notes"] == [], look


@pytest.mark.parametrize("bpm,length", [(60, 8), (60, 24), (72, 12), (150, 8), (150, 24)])
def test_timeline_fits_at_any_tempo(bpm, length):
    m = synth_music(bpm, length)
    piece = pr.plan_piece(dict(REQ, length=length), m)
    d = pr.dry_run(piece)
    assert d["notes"] == [] and d["last_end"] <= m["cut"] - 0.3
    assert piece["plan"]["tries"] == 1                       # the estimate is exact, the dry run only confirms it


def test_a_dry_run_note_shrinks_the_windows(monkeypatch, p02_music):
    real, bars = pr.dry_run, []

    def fake(piece, size=pr.DRY_SIZE, camera=False):
        r = real(piece, size, camera)
        bars.append(piece["plan"]["bars"])
        if len(bars) == 1:
            r = dict(r, notes=["timeline (13.00s) is longer than the clip (11.75s): the end is cut"])
        return r

    monkeypatch.setattr(pr, "dry_run", fake)
    p = pr.plan_piece(dict(REQ, shots="single"), p02_music)             # `single`: the bar-based scheduler that the retry shrinks (a multi-shot piece times one lockup per shot)
    assert bars[0] - bars[-1] == 1 and p["plan"]["tries"] == 2        # dry_run is also called by pull_in_lags / enforce_coverage inside an attempt now, so compare the first attempt with the last
    assert any("shrunk by 1 bar" in n for n in p["plan"]["notes"]) and any("dry run 1" in n for n in p["plan"]["notes"])


# ----------------------------------------------------------------------------------------------- shots / layout
def _check_cut_shots(piece, nframes):
    """hard cuts (default): no crossfade, no chaining, every cut on a downbeat frame, LTX lengths 8k+1 >= the span they fill."""
    lay, fps = piece["layout"], piece["fps"]
    down = {int(round(d * fps)) for d in piece["type"]["downbeats"]}
    end = 0
    for k, s in enumerate(piece["shots"]):
        L = lay[s["id"]]
        span = s.get("span") or s["frames"]
        assert s["frames"] % 8 == 1 and 9 <= s["frames"] <= 193 and s["frames"] >= span and s["frames"] - span < 8, s
        assert L == {"start": end, "frames": span, "xf": 0}
        assert s["first"] == chr(ord("A") + k) and not s["first"].startswith("last:")        # its own still, no continuity
        if k:
            assert L["start"] in down, (L, sorted(down)[:5])                                   # the cut is a downbeat frame
        end += span
    assert end == nframes
    assert piece["compose"]["crossfade"] == 0
    assert [st["id"] for st in piece["stills"]] == [s["first"] for s in piece["shots"]]


def _check_shots(piece, nframes):
    if piece["plan"]["shots"] != "chain":
        return _check_cut_shots(piece, nframes)
    # chain = each shot starts from the previous shot's last on-screen frame, joined by a HARD CUT on a downbeat (never a crossfade between generations)
    lay, fps = piece["layout"], piece["fps"]
    down = {int(round(d * fps)) for d in piece["type"]["downbeats"]}
    sids = [s["id"] for s in piece["shots"]]
    assert sids == [f"s{k + 1}" for k in range(len(sids))] and list(lay) == sids
    assert piece["compose"]["crossfade"] == 0
    end = 0
    for k, s in enumerate(piece["shots"]):
        span = s.get("span") or s["frames"]
        assert s["frames"] % 8 == 1 and 9 <= s["frames"] <= 193 and s["frames"] >= span and s["frames"] - span < 8, s
        L = lay[s["id"]]
        assert L == {"start": end, "frames": span, "xf": 0}
        if k:
            assert L["start"] in down, (L, sorted(down)[:5])
        assert s["first"] == ("A" if k == 0 else f"last:s{k}")
        end += span
    assert end == nframes


def test_shots_of_the_p02_plan(p02_music, planned):
    _check_shots(planned, p02_music["nframes"])
    assert planned["plan"]["shots"] == "cut" and len(planned["shots"]) >= 3


@pytest.mark.parametrize("length,n", [(8, 2), (12, 3), (16, 4), (24, 6)])
@pytest.mark.parametrize("bpm", [72, 122])
def test_length_gives_shot_count(length, n, bpm):
    m = synth_music(bpm, length)
    piece = pr.plan_piece(dict(REQ, length=length, shots="chain", legacy_chain=True), m, dry=False)
    assert len(piece["shots"]) >= n                                                      # chain = hard cuts on downbeats now: as many as cut mode
    _check_shots(piece, m["nframes"])
    assert piece["audio"]["cut"] == pytest.approx(m["cut"])
    cut = pr.plan_piece(dict(REQ, length=length), m, dry=False)                          # hard cuts: at least as many shots, on downbeats
    assert len(cut["shots"]) >= n
    _check_shots(cut, m["nframes"])


def test_more_than_six_shots_stay_six_slots():
    m = synth_music(122, 40)
    for mode in ("chain", "cut"):
        piece = pr.plan_piece(dict(REQ, shots=mode, legacy_chain=True), m, dry=False)
        assert len(piece["shots"]) == 6
        assert piece["layout"]["s6"]["start"] + piece["layout"]["s6"]["frames"] >= m["nframes"]


# ----------------------------------------------------------------------------------------------- brief
BAD_BRIEFS = ["", "   \n", "not json at all", "{", "[1, 2, 3]", "```json\n```", '{"scene": 5, "motion": {"a": 1}, "bpm": []}',
              '{"technique": "sideways", "mood": "purple", "palette": ["x"]}', "\x00\x01 garbage }{ [[", None]


@pytest.mark.parametrize("text", BAD_BRIEFS)
def test_bad_brief_falls_back_to_rules(text, p02_music):
    base = pr.plan_piece(REQ, p02_music, None, dry=False)
    got = pr.plan_piece(REQ, p02_music, text, dry=False)
    a, b = copy.deepcopy(base), copy.deepcopy(got)
    a.pop("plan"), b.pop("plan")
    assert a == b                                   # nothing usable in the brief = no brief
    assert fb.brief_is_empty(fb.parse_brief(text))


def test_brief_supplies_creative_text_only(p02_music):
    text = ('Sure! ```json\n{"scene": "a lone swimmer in a glass-walled pool at night, steam and soft light", '
            '"still": "a swimmer seen through foggy glass, blue light", "motion": ["steam drifts across the glass", '
            '"the swimmer glides slowly", "ripples spread, camera pushes in"], "music": "dream pop, soft synth pads", '
            '"bpm": 98, "technique": "Behind", "mood": "dreamy", "palette": ["#0a0f14", "#eef3f5", "#37c6d4", "#f2a65a"], '
            '"signoff": "Night Swim"}\n```')
    base = pr.plan_piece(REQ, p02_music, None, dry=False)
    p = pr.plan_piece(REQ, p02_music, text, dry=False)
    assert p["technique"] == "behind" and p["shots"][0]["guide"] == pytest.approx(0.0) and p["shots"][0].get("matte")      # the recipe guide 0.4; "AFTER HOURS" has 10 letters > OW.MAX_LTX_LETTERS: no LTX-owned word, guide OFF (OWNED_GUIDE 1.0 applies only to a short LTX word)
    assert "swimmer" in p["stills"][0]["prompt"] and "steam drifts" in p["shots"][0]["prompt"]
    assert p["signoff"]["text"] == "night swim"
    assert p["type"]["palette"]["bg"] == [10, 15, 20]
    assert [w["text"] for w in p["type"]["words"]] == [w["text"] for w in base["type"]["words"]]   # words stay the user's
    assert p["layout"] == base["layout"]                                                            # timing is the rules'
    assert p["brief"]["technique"] == "behind"
    assert pr.piece_hash(pr.plan_piece(REQ, p02_music, text, dry=False)) == pr.piece_hash(p)


def test_brief_prompt_is_shuffled_per_seed_and_asks_for_n_shots():
    p1 = fb.brief_prompt(dict(REQ, seed=1))
    assert p1 == fb.brief_prompt(dict(REQ, seed=1))
    assert p1 != fb.brief_prompt(dict(REQ, seed=2))
    assert "exactly 3 short phrases" in p1 and "AFTER HOURS\nTHE CITY KEEPS DREAMING" in p1
    assert "exactly 6 short phrases" in fb.brief_prompt(dict(REQ, length=24))


# ----------------------------------------------------------------------------------------------- rules
def test_technique_from_scene_keywords(p02_music):
    for scene, tech in (("a woman in drifting mist", "behind_sheer"), ("a runner behind a glass wall", "behind_sheer"),
                        ("a dancer in a field of wheat", "behind"), ("rocks and tall grass at dusk", "behind"),
                        ("rolling landscape contours", "traced"), ("a neon street", "built_in")):
        p = pr.plan_piece(dict(REQ, scene=scene), p02_music, dry=False)
        assert p["technique"] == tech
        assert p["shots"][0]["control"]["pass"] == ("line" if tech == "traced" else "depth")
        assert bool(p["shots"][0].get("matte")) == tech.startswith("behind")


def test_recipe_timelines_rescale_to_the_cut(p02_music):
    m = dict(p02_music)
    p = pr.plan_piece(dict(REQ, scene="a woman in mist"), m, dry=False)       # behind = P01 recipe (ref cut 14.4)
    f = m["cut"] / 14.4
    ops = p["compose"]["default"]["ops"]
    tl = [o["args"]["t"]["timeline"] for o in ops if o["op"] == "crossfade" and "timeline" in o["args"]["t"] and not str(o["out"]).startswith("_") and not any(str(v).startswith("$_") for v in o["args"].values())]   # recipe pulses (matte-treatment gates use _xx buffers)
    beats = m["beats"]
    assert abs(tl[0][0][0] - 9.9 * f) <= 60.0 / m["bpm"] / 2 + 0.002                      # rescaled, then snapped to the nearest beat
    for keys in tl:
        assert all(min(abs(k[0] - b) for b in beats) < 0.002 for k in keys if k[0] > 0), keys      # every pulse key is on a beat
        assert all(b[0] > a[0] for a, b in zip(keys, keys[1:]))
    assert p["compose"]["size"] == [1080, 1920] and p["compose"]["crossfade"] == 0
    chain = pr.plan_piece(dict(REQ, scene="a woman in mist", shots="chain", legacy_chain=True), m, dry=False)
    assert chain["compose"]["crossfade"] == 0                  # chain = hard cuts, generations are never blended
    glow = pr.plan_piece(REQ, m, dry=False)["compose"]["default"]["ops"]
    assert all(o["args"]["color"] != "@accent0" for o in glow if o["op"] == "add_glow")          # 2026-10-30: no default glow (it exists only as hold life), so there may be no add_glow op at all


def test_draft_sizes(p02_music):
    p = pr.plan_piece(dict(REQ, quality="draft"), p02_music, dry=False)
    assert p["quality"] == "draft" and p["sizes"]["type"] == [576, 1024] and p["sizes"]["compose"] == [540, 960]
    assert p["sizes"]["shot"] == [576, 1024] and all(s["size"] == [576, 1024] for s in p["shots"])      # draft: single LTX stage


def test_request_validation():
    assert pr.normalize_request(dict(REQ, length=13))["length"] == 13                      # the requested length is honoured (no snap to 8/12/16/24)
    assert pr.normalize_request(dict(REQ, length=14))["length"] == 14
    assert pr.normalize_request(dict(REQ, length=2))["length"] == pr.MIN_LENGTH and pr.normalize_request(dict(REQ, length=999))["length"] == pr.MAX_LENGTH
    assert pr.normalize_request(dict(REQ, look="LANES"))["look"] == "lanes"
    with pytest.raises(ValueError):
        pr.normalize_request(dict(REQ, words="  \n "))
    with pytest.raises(ValueError):
        pr.normalize_request(dict(REQ, words="Style: calm"))
    with pytest.raises(ValueError):
        pr.normalize_request(dict(REQ, look="nonsense"))


def test_music_plan(p02_music):
    mp = pr.music_plan(REQ)
    assert "no vocals" in mp["tags"] and mp["seconds"] == pytest.approx(22.4) and mp["lyrics"] == "[Instrumental]"
    mp2 = pr.music_plan(dict(REQ, music="lofi hip hop, 84 bpm, A minor"))
    assert mp2["bpm"] == 84 and mp2["keyscale"] == "A minor" and mp2["tags"].startswith("lofi hip hop")
    mp3 = pr.music_plan(REQ, dict(fb.empty_brief(), music="slow strings", bpm=70))
    assert mp3["tags"].startswith("slow strings") and mp3["bpm"] == 70


# ----------------------------------------------------------------------------------------------- hand-written pieces
def test_finalize_p02_json(p02_music):
    d = json.load(open(os.path.join(ROOT, "15_fusion", "pieces", "P02.json"), encoding="utf-8"))
    p = pr.finalize_piece(d, p02_music)
    assert p["layout"] == {"s1": {"start": 0, "frames": 121, "xf": 0}, "s2": {"start": 109, "frames": 121, "xf": 12},
                           "s3": {"start": 218, "frames": 105, "xf": 12}}
    assert p["technique"] == "built_in" and p["quality"] == "final" and p["sizes"]["compose"] == [1080, 1920]
    assert p["type"]["bpm"] == pytest.approx(p02_music["bpm"], abs=0.001) and p["type"]["downbeats"]
    assert pr.finalize_piece(json.loads(pr.piece_json(p)), p02_music) == p      # idempotent
    d1 = json.load(open(os.path.join(ROOT, "15_fusion", "pieces", "P01.json"), encoding="utf-8"))
    assert pr.finalize_piece(d1)["technique"] == "behind"
    d4 = json.load(open(os.path.join(ROOT, "15_fusion", "pieces", "P04.json"), encoding="utf-8"))
    assert pr.finalize_piece(d4)["technique"] == "traced"


# ----------------------------------------------------------------------------------------------- ComfyUI nodes
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


def test_plan_nodes(p02_music):
    _import_comfy()
    from nodes_fusion import plan as nodes
    ids = []
    for cls in nodes.NODES:
        s = cls.define_schema()
        cls.GET_SCHEMA()
        ids.append(s.node_id)
    assert ids == ["MographRequest", "MographFusionBriefPrompt", "MographPiecePlan", "MographPieceFromJSON", "MographPieceSizes"]
    req = nodes.MographRequest.execute(WORDS, "", "", 12, "auto", 5, "final").args[0]
    assert req["length"] == 12 and req["seed"] == 5
    prompt, seed = nodes.MographFusionBriefPrompt.execute(req).args
    assert "USER WORDS" in prompt and isinstance(seed, int)
    piece, pj, still, neg = nodes.MographPiecePlan.execute(req, p02_music, "not json").args
    assert json.loads(pj) == piece and still and neg
    direct = pr.plan_piece(req, p02_music, "not json")
    from tools.fusion import still_preview as SP
    SP.prepare_piece(direct, req.get("raw"), bool(req.get(SP.FLAG)))        # the node also stamps the app inputs on the piece (still preview flag + raw request); the planner output underneath is identical
    assert pr.piece_hash(piece) == pr.piece_hash(direct)
    again = nodes.MographPieceFromJSON.execute(pj, p02_music).args[0]
    assert again["layout"] == piece["layout"] and again["type"] == piece["type"]      # replays exactly
    src = open(os.path.join(ROOT, "15_fusion", "pieces", "P02.json"), encoding="utf-8").read()
    p02 = nodes.MographPieceFromJSON.execute(src).args[0]                              # analysis found via the piece id
    assert p02["layout"]["s2"]["start"] == 109 and p02["type"]["downbeats"]
    with pytest.raises(ValueError):
        nodes.MographPieceFromJSON.execute("{nope")
