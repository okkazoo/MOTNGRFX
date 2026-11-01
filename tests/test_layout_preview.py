"""RANDOMISE LAYOUT PREVIEW (user 2026-10-30: "an option on the preview to create a random seed so we can see a preview of the text layout on the single frames and hit a randomise button to
quickly create a new one"): tools/fusion/layout_preview.py, nodes_fusion/layout_preview.py, tools/app/layout_preview_graph.py, the request `layout_seed`, the page route. CPU only, no model, no network.
Run (on the box, ComfyUI venv python): python -m pytest tests/test_layout_preview.py -q --rootdir=tests"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import layout_preview as LP  # noqa: E402
from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import stable_type as ST  # noqa: E402

FIXTURE = HERE / "fixtures" / "layout_preview_piece.json"


def piece():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def stills(n=5):
    """synthetic stills (a vertical gradient with a bright blob, one per place), RGB uint8 576x1024."""
    out = {}
    yy, xx = np.mgrid[0:1024, 0:576].astype(np.float32)
    for k in range(1, n + 1):
        r = (xx / 576.0) * 200 + 20 * k
        g = (yy / 1024.0) * 160 + 10
        b = 90 + 60 * np.sin(yy / 90.0 + k)
        im = np.stack([r, g, b], -1)
        im += 90 * np.exp(-(((xx - 288) / 120.0) ** 2 + ((yy - 400) / 200.0) ** 2))[..., None]
        out[k] = np.clip(im, 0, 255).astype(np.uint8)
    return out


def layouts(p):
    return [LP.layout_of(w) for w in p["type"]["words"]]


# ------------------------------------------------------------------------------------------------------------------------ the draw
def test_same_seed_same_draw():
    a, ia = LP.redraw_piece(piece(), 4242, stills())
    b, ib = LP.redraw_piece(piece(), 4242, stills())
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert ia["summary"] == ib["summary"] and ia["holds"] == ib["holds"] and ia["fixes"] == ib["fixes"]
    assert a["plan"]["layout_seed"] == 4242 and a["plan"]["piece_system"]["seed"] == 4242


def test_different_seed_different_layouts_fonts_and_cuts():
    seen = {}
    for s in range(1, 13):
        p, info = LP.redraw_piece(piece(), s, stills())
        seen[s] = (tuple(layouts(p)), tuple(w["font"] for w in p["type"]["words"]), tuple(info["summary"]["cuts"]), tuple(info["summary"]["fonts"]), info["summary"]["outline"],
                   tuple(info["summary"]["hold_life"]))
    assert len({v[0] for v in seen.values()}) >= 4, "the layout vector varies with the seed"
    assert len({v[1] for v in seen.values()}) >= 6, "the faces vary with the seed"
    assert len({v[2] for v in seen.values()}) >= 6, "the cut kinds vary with the seed"
    assert len({v[4] for v in seen.values()}) >= 2 and len({v[5] for v in seen.values()}) >= 3, "outline and hold-life choices vary"
    for s in (1, 2, 3):                                            # any two neighbouring seeds differ somewhere
        assert seen[s] != seen[s + 1]
    kinds = {x for v in seen.values() for x in v[0]}
    assert {"line", "grid"} <= kinds and kinds & {"rot +90", "rot -90"}, kinds


def test_words_timings_and_shots_stay_the_same():
    p0 = piece()
    p, info = LP.redraw_piece(p0, 77, stills())
    for k in ("t0", "t1", "form_s", "exit_s", "hit_on", "text"):
        assert [w.get(k) for w in p["type"]["words"]] == [w.get(k) for w in p0["type"]["words"]], k
    assert p["layout"] == p0["layout"] and p["shots"] == p0["shots"] and p["stills"] == p0["stills"] and p["audio"] == p0["audio"]
    assert len(info["holds"]) == len(p0["type"]["words"]) and all(b > a for a, b in info["holds"])


def test_no_orphan_rule_holds_for_every_seed():
    texts = ["DREAMING", "STEAM", "GO", "RISE", "FIRST WORD", "WIND", "A"]
    base = piece()
    words0 = base["type"]["words"]
    base["type"]["words"] = [dict(copy.deepcopy(words0[i % len(words0)]), text=t) for i, t in enumerate(texts)]
    n_split = 0
    for s in range(1, 31):
        p, _ = LP.redraw_piece(base, s, stills())
        for w in p["type"]["words"]:
            lines = w.get("lines")
            if lines and len(lines) > 1:
                assert ST.orphan_free(lines, w["text"]), (s, w["text"], lines)
            if LP.layout_of(w) in ("grid", "column"):
                n_split += 1
                assert w["lines"] and "".join(ST._letters(x) for x in w["lines"]).upper() == "".join(ST._letters(w["text"])).upper()
    assert n_split >= 10, "grid / column layouts do get drawn"


def test_effects_force_the_requested_layouts_and_hold_life():
    p, info = LP.redraw_piece(piece(), 9, stills(), "grid, pulse")
    assert "grid" in p["plan"]["effects"]["requested"] and "pulse" in info["summary"]["hold_life"]
    assert sum(1 for x in layouts(p) if x == "grid") >= 2
    assert p["compose"]["hold_life"] == ["pulse"]


def test_palette_is_the_stills_own_and_changes_with_the_seed():
    cols = set()
    for s in range(1, 9):
        p, _ = LP.redraw_piece(piece(), s, stills())
        pal = p["plan"]["piece_system"]["palette"]
        assert pal["source"] == "still" and 3 <= len(pal["colors"]) <= 5
        tp = p["type"]["palette"]
        allc = [list(c) for c in pal["colors"]]
        assert list(tp["bg"]) in allc and list(tp["fg"]) in allc and all(list(a) in allc for a in tp["accents"])
        cols.add((tuple(tp["bg"]), tuple(tp["fg"]), len(pal["colors"])))
    assert len(cols) >= 3


def test_cuts_are_remapped_onto_the_drawn_kinds():
    from tools.fusion import cut_kinds as CK
    for s in (3, 4, 5):
        p, info = LP.redraw_piece(piece(), s, stills())
        kinds = set(p["plan"]["piece_system"]["cuts"]["kinds"])
        for c in p["compose"]["cuts"]:
            assert CK.normalize(c["cut"]) in kinds, (s, c)


# ------------------------------------------------------------------------------------------------------------------------ the sheet
def test_one_tile_per_word_unique_filenames_and_the_header(tmp_path):
    p0 = piece()
    r = LP.preview(p0, stills(), 31, out_dir=tmp_path, stem="demo", procs=2, work=str(tmp_path / "w1"))
    n = len(p0["type"]["words"])
    assert len(r["tiles"]) == n and all(t.shape == (LP.TILE_H, LP.TILE_W, 3) for t in r["tiles"])
    assert r["timing"]["color_errors"] == [], "legible_color ran on every tile (no silent fallback to the plain face)"
    assert r["sheet"].ndim == 3 and r["sheet"].shape[0] > LP.TILE_H and r["sheet"].dtype == np.uint8
    assert Path(r["files"]["png"]).name == "demo_layout_s31.png" and Path(r["files"]["json"]).name == "demo_layout_s31.json"
    first = Path(r["files"]["png"]).read_bytes()
    j = json.loads(Path(r["files"]["json"]).read_text(encoding="utf-8"))
    assert j["seed"] == 31 and len(j["summary"]["words"]) == n and j["piece"]["plan"]["layout_seed"] == 31
    r2 = LP.preview(p0, stills(), 31, out_dir=tmp_path, stem="demo", procs=2, work=str(tmp_path / "w2"))      # the same seed again: a NEW name, the first stays
    assert Path(r2["files"]["png"]).name == "demo_layout_s31_b.png" and Path(r["files"]["png"]).read_bytes() == first
    assert np.array_equal(r["sheet"], r2["sheet"]), "same seed, same pixels"
    r3 = LP.preview(p0, stills(), 32, out_dir=tmp_path, stem="demo", procs=2, work=str(tmp_path / "w3"))
    assert Path(r3["files"]["png"]).name == "demo_layout_s32.png" and not np.array_equal(r["sheet"], r3["sheet"])
    assert len(list(tmp_path.glob("demo_layout_s*.png"))) == 3


def test_the_tile_shows_the_word_over_its_own_shots_still(tmp_path):
    p0 = piece()
    st = stills()
    r = LP.preview(p0, st, 5, procs=0, work=str(tmp_path / "w"))
    # the word's time falls in one shot -> that shot's place still is the plate; compare a corner (no type there) with the cover-scaled still
    for i, tile in enumerate(r["tiles"]):
        hit, rel = r["info"]["holds"][i]
        k, img = LP.still_for_time(r["piece"], (hit + rel) / 2.0, st)
        plate = LP._cover(img, LP.TILE_W, LP.TILE_H)
        d = np.abs(tile[:24, :24].astype(int) - plate[:24, :24].astype(int)).mean()
        assert d < 40, (i, k, d)


def test_still_for_time_maps_a_word_to_its_shot():
    p0 = piece()
    st = stills()
    rows = sorted(((v["start"], sid) for sid, v in p0["layout"].items()))
    seen = []
    for start, sid in rows:
        k, _ = LP.still_for_time(p0, start / 24.0 + 0.2, st)
        seen.append(k)
    assert seen == [1, 2, 3, 4, 5], seen


# ------------------------------------------------------------------------------------------------------------------------ the lock: "Use this layout"
def test_locked_seed_reproduces_the_draw(tmp_path):
    p0 = piece()
    seed = 8123
    drawn, _ = LP.redraw_piece(p0, seed, stills(), "outline")
    LP.write_lock(tmp_path, "demo", seed, ["outline"])
    lock = LP.read_lock(tmp_path, "demo")
    assert lock and lock["layout_seed"] == seed and lock["effects"] == ["outline"]
    again = LP.apply_lock(p0, lock, stills())
    assert json.dumps(again, sort_keys=True, default=str) == json.dumps(drawn, sort_keys=True, default=str)
    assert LP.read_lock(tmp_path, "nope") is None


def test_the_route_helper_writes_the_lock_inside_the_output_dir_only(tmp_path):
    import app_routes as AR
    out = tmp_path / "out"
    (out / "archive").mkdir(parents=True)
    r = AR.save_layout_lock("archive", "demo", 77, "grid", output_dir=out)
    assert r.get("ok") and (out / "archive" / "demo_layout_lock.json").is_file() and r["layout_seed"] == 77
    assert "error" in AR.save_layout_lock("../..", "demo", 77, output_dir=out)
    assert "error" in AR.save_layout_lock("archive", "../x", 77, output_dir=out)
    assert "error" in AR.save_layout_lock("archive", "demo", 0, output_dir=out)
    assert "error" in AR.save_layout_lock("missing", "demo", 5, output_dir=out)


def test_upgrade_applies_a_saved_lock(tmp_path):
    from tools.app import upgrade_still as US
    import cv2
    stem = "run_r01_still"
    p0 = piece()
    p0["still_preview"] = True
    p0["request"] = {"words": "STILL\nTHE", "seed": 3, "length": 16}
    (tmp_path / f"{stem}_piece.json").write_text(json.dumps(p0), encoding="utf-8")
    for k, im in stills(2).items():
        name = f"{stem}_still.png" if k == 1 else f"{stem}_still_{k}.png"
        cv2.imwrite(str(tmp_path / name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    (tmp_path / f"{stem}_music.wav").write_bytes(b"RIFF")
    plain = US.load_saved(str(tmp_path))
    assert plain["layout_lock"] is None and plain["piece"]["plan"].get("layout_seed") is None
    LP.write_lock(tmp_path, stem, 555)
    locked = US.load_saved(str(tmp_path))
    assert locked["layout_lock"]["layout_seed"] == 555 and locked["piece"]["plan"]["layout_seed"] == 555
    assert locked["piece"]["plan"]["piece_system"]["seed"] == 555 and locked["piece"]["still_preview"] is True and locked["piece"]["request"] == p0["request"]


# ------------------------------------------------------------------------------------------------------------------------ the planner hook (a request's layout_seed)
def test_request_layout_seed_redraws_only_the_layout_side():
    from test_fusion_core import synth_music
    from tools.fusion import plan_rules as PR
    req = {"words": "SLIDE\nLAUGH\nWIND\nUP", "scene": "a wide frozen lake under pale winter sun", "music": "calm piano", "length": 16, "look": "auto", "seed": 5,
           "quality": "preview", "format": "9:16", "shots": "cut"}
    assert PR.normalize_request(req)["layout_seed"] is None and PR.normalize_request(dict(req, layout_seed=0))["layout_seed"] is None
    assert PR.normalize_request(dict(req, layout_seed="123"))["layout_seed"] == 123
    typed = PR.plan_piece(dict(req, layout_seed=31), synth_music(100, 16), dry=False)          # a typed request (no concept system): the layout seed draws a whole system
    assert typed["plan"]["layout_seed"] == 31 and typed["plan"]["piece_system"]["seed"] == 31
    req = dict(req, system=PS.choose(5, "restless", 100, None, (), None, ("SLIDE", "LAUGH", "WIND", "UP"), 16.0))          # the concept's system
    base = PR.plan_piece(req, synth_music(100, 16), dry=False)
    lk = PR.plan_piece(dict(req, layout_seed=999), synth_music(100, 16), dry=False)
    lk2 = PR.plan_piece(dict(req, layout_seed=999), synth_music(100, 16), dry=False)
    a, b = base["plan"]["piece_system"], lk["plan"]["piece_system"]
    assert lk["plan"]["layout_seed"] == 999 and "layout_seed" not in base["plan"] and b["seed"] == 999 and b["layout_seed"] == 999
    for k in ("module", "modules", "io", "palette", "photo", "photo_key", "shot_moves", "camera", "motion"):          # what the stills and footage were made from stays
        assert a.get(k) == b.get(k), k
    assert (a["fonts"], a["cuts"]) != (b["fonts"], b["cuts"])
    assert lk["stills"] == base["stills"] and lk["shots"] == base["shots"], "same stills prompts: the layout seed never changes the pictures"
    assert json.dumps(lk["plan"]["piece_system"], sort_keys=True) == json.dumps(lk2["plan"]["piece_system"], sort_keys=True)
    assert lk["type"]["words"] and [w["font"] for w in lk["type"]["words"]] != [w["font"] for w in base["type"]["words"]] or lk["plan"]["piece_system"]["fonts"] != base["plan"]["piece_system"]["fonts"]


def test_with_layout_seed_keeps_the_picture_side():
    base = PS.choose(7, "restless", 100, None, (), None, ("RISE", "GO"), 16.0)
    new = LP.with_layout_seed(base, 31, ("RISE", "GO"), 16.0)
    for k in ("module", "io", "palette", "photo", "camera", "motion", "shot_moves", "mood", "tone"):
        assert new[k] == base[k], k
    assert new["seed"] == 31 and new["fonts"] != base["fonts"] and PS.choose(31, "restless", 100, None, (), None, ("RISE", "GO"), 16.0)["fonts"] == new["fonts"]
    assert LP.with_layout_seed(None, 31, ("RISE",), 16.0, mood="eerie", bpm=90)["seed"] == 31


# ------------------------------------------------------------------------------------------------------------------------ the graph + the node
def test_graph_builder():
    from tools.app import layout_preview_graph as G
    g = G.build("archive/motngrfx_shots", 12, "grid", stem="demo")
    assert g["layout"]["class_type"] == "MographLayoutPreview" and g["layout"]["inputs"]["archive"] == "archive/motngrfx_shots" and g["layout"]["inputs"]["seed"] == 12
    assert g["save"]["class_type"] == "SaveImage" and g["save"]["inputs"]["images"] == ["layout", 0] and g["save"]["inputs"]["filename_prefix"].startswith("layout_preview/demo_layout")
    assert g["out_system"]["class_type"] == "PreviewAny" and g["out_system"]["inputs"]["source"] == ["layout", 1]
    v = G.build("a", 1, via_load=True)
    assert v["load"]["class_type"] == "MographArchiveLoad" and v["layout"]["inputs"]["piece"] == ["load", 2] and v["layout"]["inputs"]["save"] is False
    tpl = json.loads(Path(G.TEMPLATE).read_text(encoding="utf-8"))
    assert set(tpl) == {"layout", "save", "out_system"} and tpl["layout"]["inputs"]["archive"] == ""
    import app_routes as AR
    assert all(not t["name"].startswith("_") for t in AR.list_graphs()), "the helper graph stays out of the template dropdown"
    assert AR.load_graph("_layout_preview")["graph"]["layout"]["class_type"] == "MographLayoutPreview"


def test_node_is_registered():
    pytest.importorskip("comfy_api.latest")
    import nodes_fusion
    names = [n.define_schema().node_id for n in nodes_fusion.NODES if hasattr(n, "define_schema")]
    assert "MographLayoutPreview" in names
    assert "layout_preview" in nodes_fusion.SUBMODULES and "layout_preview" not in nodes_fusion.FAILED


# ------------------------------------------------------------------------------------------------------------------------ one short scene per word
def test_an_isolated_word_scene_draws_the_same_frame_as_the_full_timeline(tmp_path):
    p, info = LP.redraw_piece(piece(), 21, stills())
    spec = LP.type_spec(p)
    rows = LP.word_timing(p)
    i = 2
    r = rows[i]
    t = LP.sample_times(p, [tuple(h) for h in info["holds"]], p["type"]["bpm"])[i]
    iso = LP.isolate_word(spec, i, float(r["S"]), float(r["end"]))
    assert len(iso["words"]) == 1 and iso["duration"] < spec["duration"] / 2 and iso["words"][0]["t0"] == 0.0
    tasks = [dict(spec=spec, frame=int(round(t * 24))), dict(spec=iso, frame=int(round((t - float(r["S"])) * 24)))]
    LP.render_frames(tasks, str(tmp_path), LP.TYPE_SIZE, procs=2, timeout=300)
    import cv2
    a = cv2.imread(str(tmp_path / "0" / "matte" / f"matte_{int(round(t * 24)):05d}.png"), cv2.IMREAD_GRAYSCALE).astype(int)
    f2 = int((tmp_path / "1" / "frame.txt").read_text())
    b = cv2.imread(str(tmp_path / "1" / "matte" / f"matte_{f2:05d}.png"), cv2.IMREAD_GRAYSCALE).astype(int)
    assert a.sum() > 1000, "the word is on screen in the full render"
    assert np.abs(a - b).mean() < 1.0, np.abs(a - b).mean()


def test_isolate_keeps_a_dict_camera_whole():
    spec = LP.type_spec(piece())
    spec["camera"] = {"chain": [], "plate": "multiplane", "cards": 3}
    assert LP.isolate_word(spec, 1, 3.0, 6.0) is spec


# ------------------------------------------------------------------------------------------------------------------------ the deadline: a slow word is drawn flat
def test_flat_frame_draws_the_words_layout_in_its_own_face():
    p, _ = LP.redraw_piece(piece(), 21, stills())
    seen = set()
    for w in p["type"]["words"]:
        face, a = LP.flat_frame(w, (250, 240, 220))
        assert face.shape == (1024, 576, 3) and a.shape == (1024, 576) and a.max() == 255 and (a > 128).sum() > 800, w["text"]
        ys, xs = np.nonzero(a > 128)
        assert 0.3 * 576 > abs(xs.mean() - 288) and 0.2 * 1024 > abs(ys.mean() - 512), "centred"
        assert xs.min() > 0 and xs.max() < 575 and ys.min() > 0 and ys.max() < 1023, "inside the frame"
        if LP.layout_of(w) in ("rot +90", "rot -90"):
            assert ys.max() - ys.min() > xs.max() - xs.min(), "a rolled word reads along the frame height"
        seen.add(LP.layout_of(w))
    for s in range(30, 50):                                       # grid and column words too
        p, _ = LP.redraw_piece(piece(), s, stills())
        for w in p["type"]["words"]:
            if LP.layout_of(w) in ("grid", "column"):
                face, a = LP.flat_frame(w)
                assert (a > 128).sum() > 800 and len(w["lines"]) >= 2
                return
    raise AssertionError("no grid / column word in 20 seeds")


def test_a_word_that_runs_out_of_time_is_drawn_flat_and_the_sheet_still_has_every_tile(tmp_path, monkeypatch):
    monkeypatch.setattr(LP, "TASK_TIMEOUT_S", 0.05)
    r = LP.preview(piece(), stills(), 6, out_dir=tmp_path, stem="slow", procs=3, work=str(tmp_path / "w"))
    n = len(piece()["type"]["words"])
    assert len(r["tiles"]) == n and r["info"]["summary"]["flat_words"] == list(range(n)), "every word timed out -> every word flat"
    assert r["timing"]["render"]["timeouts"] and r["timing"]["total_s"] < 20
    assert all(t.std() > 5 for t in r["tiles"])


def test_coarse_focus_patch_is_idempotent():
    from mograph.modules import sequence as MS
    LP._coarse_focus()
    first = MS.FocusLook
    LP._coarse_focus()
    assert MS.FocusLook is first and MS.FocusLook.__init__ is LP._coarse_focus_init            # patched in place (a local subclass did not pickle on Python 3.12)
    import pickle
    assert pickle.loads(pickle.dumps(MS.FocusLook.__new__(MS.FocusLook))).__class__ is MS.FocusLook


# ------------------------------------------------------------------------------------------------------------------------ 2026-10-30: the tiles outline what the render outlines
def _defeating_stills(n=5):
    """stills whose blocks (black / dark grey / mid grey / white) no single ink reads on: the colour stage wants an outline everywhere."""
    yy, xx = np.mgrid[0:1024, 0:576]
    lv = np.array([0, 77, 148, 255], np.uint8)
    im = np.repeat(lv[((yy // 48) + (xx // 48) * 2) % 4][..., None], 3, -1)
    return {k: im.copy() for k in range(1, n + 1)}


def _spy_tiles(monkeypatch, tmp_path, seed, stl):
    from tools.fusion import legible_color as LC
    calls, real = [], LC.legible_color

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(dict(t=k["t"], n=len(k["words"]), outline=k.get("outline"), key=k["state_key"], pal=k.get("ink_palette"),
                          support=list(LC.LC_LOG[k["state_key"]][0][0]["support"]), kind=LC.LC_LOG[k["state_key"]][0][0]["kind"]))
        return out
    monkeypatch.setattr(LC, "legible_color", spy)
    monkeypatch.setattr(LP, "render_frames", lambda tasks, out, size, procs=None, timeout=None: dict(wall_s=0, procs=0, failed=list(range(len(tasks))), timeouts=[]))      # flat words: fast
    r = LP.preview(piece(), stl, seed, procs=0, work=str(tmp_path / "w"))
    return r, calls


def test_preview_tiles_outline_only_the_drawn_words_plus_the_capped_last_resort(tmp_path, monkeypatch):
    for seed in (101, 102, 103):
        r, calls = _spy_tiles(monkeypatch, tmp_path / str(seed), seed, _defeating_stills())
        n = len(r["tiles"])
        assert len(calls) == n and r["timing"]["color_errors"] == []
        assert len({c["key"] for c in calls}) == 1 and all(c["n"] == n for c in calls), "one colour-stage state and the whole word list: the ceil(n / 3) cap counts across the tiles"
        wins = r["info"]["outline_windows"]
        assert all(c["outline"] == wins for c in calls), "exactly the drawn windows are passed, nothing forced onto every tile"
        holds = r["info"]["holds"]
        drawn = {i for i, (a, b) in enumerate(holds) if any(w[0] < b and w[1] > a for w in wins)}
        outlined = {i for i, c in enumerate(calls) if "outline" in c["support"]}
        assert len(outlined) <= -(-n // 3), (seed, n, outlined)
        assert drawn <= outlined or len(drawn) == 0 or all(calls[i]["support"] for i in drawn)
        assert all(c["kind"] == "extreme" for i, c in enumerate(calls) if i not in outlined and c["kind"] in ("extreme", "support")), "past the cap: extreme ink, no outline"


def test_preview_tiles_pass_no_outline_when_the_draw_has_none(tmp_path, monkeypatch):
    p0 = piece()
    seeds = [s for s in range(200, 260) if LP.redraw_piece(p0, s, stills())[1]["outline_windows"] == []]
    assert seeds, "some seed draws no outline (mode none)"
    r, calls = _spy_tiles(monkeypatch, tmp_path, seeds[0], stills())
    assert all(not c["outline"] for c in calls) and all("outline" not in c["support"] for c in calls)
