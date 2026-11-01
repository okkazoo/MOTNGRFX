"""The whole-composition set of the ingredients (16_app/INGREDIENTS.md, 2026-10-29): 3D shot-cut renders + their matte PNGs, depth PNGs, composed shots, clean plates,
window sources, type matte PNGs, the design sheet, places json, the full_archive switch, the size estimate and the pull of all of it. Fake small tensors only: no box,
no network, no GPU (the archive node needs a ComfyUI checkout for comfy_api; those tests skip without it).
Run: python -m pytest tests/test_ingredients_full.py -q"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools" / "app"), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from tools.fusion import ingredient_names as IN  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402
import ingredients as ING  # noqa: E402
from test_ingredients import AI, STEM, _png, make_version  # noqa: E402,F401  (AI = the module fixture that finds a ComfyUI checkout)


# ------------------------------------------------------------------------------------------------ names
def test_names_of_the_whole_composition_set():
    names = [IN.cut_file(STEM, "s1", "s2", "orbit:push"), IN.cut_file(STEM, "s1", "s2", "depth_pivot", "json"), IN.cut_matte_dir(STEM, "s1", "s2"),
             IN.cut_matte_frame(STEM, "s1", "s2", 3), IN.depth_dir(STEM, "s2"), IN.depth_frame(STEM, "s2", 4), IN.shot_file(STEM, "s1", "depth", "png"),
             IN.shot_file(STEM, "s1", "comp"), IN.shot_file(STEM, "s1", "plate"), IN.type_matte_dir(STEM), IN.type_matte_frame(STEM, 2)]
    assert all(IN.has_stem(STEM, n) for n in names)
    assert names[0] == f"{STEM}_cut_s1-s2_orbit_push.mp4" and names[2] == f"{STEM}_cut_s1-s2_matte" and names[3] == f"{STEM}_cut_s1-s2_matte_00003.png"
    assert names[5] == f"{STEM}_s2_depth_00004.png" and names[6] == f"{STEM}_s1_depth.png" and names[10] == f"{STEM}_type_matte_00002.png"
    for n in (names[3], names[5], names[10]):        # numbered PNGs follow the dashboard sequence rule
        assert re.match(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<suffix>_?)\.png$", n)


def test_heavy_means_only_what_full_archive_off_leaves_out():
    heavy = [IN.cut_file(STEM, "s1", "s2", "depth_pivot"), IN.cut_matte_dir(STEM, "s1", "s2") + "/", IN.shot_file(STEM, "s3", "comp"), IN.depth_dir(STEM, "s1") + "/",
             IN.type_matte_dir(STEM)]
    light = [IN.cut_file(STEM, "s1", "s2", "depth_pivot", "json"), IN.shot_file(STEM, "s1", "depth", "png"), IN.shot_file(STEM, "s1", "depth"), IN.shot_file(STEM, "s1", "plate"),
             IN.shot_file(STEM, "s1", "clip"), IN.matte_dir(STEM, "s1"), IN.type_pass_file(STEM, "matte"), IN.nm(STEM, "design_sheet.png"), IN.nm(STEM, "s1_sources.json")]
    assert all(IN.is_heavy(n) for n in heavy) and not any(IN.is_heavy(n) for n in light)


# ------------------------------------------------------------------------------------------------ the cut record (pivot_cut.py)
def _fake_cut_result(n_a=16, n_b=16, h=32, w=16, matte=True):
    """a CutResult of a 12-frame depth_pivot cut over two tiny clips (the engine replaced by a fake that also returns a matte pass)."""
    p = {"fps": 24, "shots": [{"id": "s1", "frames": n_a}, {"id": "s2", "frames": n_b}],
         "compose": {"crossfade": 0, "cuts": [{"after": "s1", "cut": "depth_pivot", "params": dict(short=64, travel=6, hold1=3, hold2=3)}]}}
    a = np.random.default_rng(1).integers(0, 255, (n_a, h, w, 3), dtype=np.uint8)
    b = np.random.default_rng(2).integers(0, 255, (n_b, h, w, 3), dtype=np.uint8)
    m = np.zeros((n_a, h, w), np.float32)
    m[:, 8:20, 4:12] = 1.0

    def fake(ca, ma, cb, mb, da, db, params):
        return SimpleNamespace(frames=np.full((12, h, w, 3), 77, np.uint8), info={"image2_start": 5, "depth": ["given", "depth-anything-v2-small"]},
                               matte=np.full((12, h, w), 200, np.uint8))
    return PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, mattes={"s1": m, "s2": m} if matte else None, render=fake)


def test_cut_record_holds_the_transition_frames_and_the_matte_through_it():
    rec = PC.cut_record(_fake_cut_result())
    assert (rec["a"], rec["b"], rec["kind"], rec["n"], rec["i2"]) == ("s1", "s2", "depth_pivot", 12, 5)
    assert rec["frames"][0].shape[0] == 5 and rec["frames"][1].shape[0] == 7 and rec["frames"][0].base is not None        # views of the new clips, no copy
    assert np.all(rec["frames"][0] == 77) and np.all(rec["frames"][1] == 77)
    assert [len(x) for x in rec["matte"]] == [5, 7] and rec["info"]["depth"] == ["given", "depth-anything-v2-small"]
    assert PC.cut_record(_fake_cut_result(matte=False))["matte"] is None          # no matte given = no matte render (the zero patches are not one)


# ------------------------------------------------------------------------------------------------ the writers
def test_archive_extras_of_the_whole_composition_set(AI, tmp_path):
    torch = pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    clips = [{"sid": "s1", "u8": torch.zeros(2, 4, 4, 3, dtype=torch.uint8), "cut_renders": [PC.cut_record(_fake_cut_result())]},
             {"sid": "s2", "u8": torch.zeros(2, 4, 4, 3, dtype=torch.uint8)}]
    out = AI.write_cuts(tmp_path, STEM, clips, heavy=True)
    assert out == [f"{STEM}_cut_s1-s2_depth_pivot.json", f"{STEM}_cut_s1-s2_depth_pivot.mp4", f"{STEM}_cut_s1-s2_matte/"]
    frames = sorted(os.listdir(tmp_path / f"{STEM}_cut_s1-s2_matte"))
    assert frames[0] == f"{STEM}_cut_s1-s2_matte_00001.png" and len(frames) == 12
    m = cv2.imread(str(tmp_path / f"{STEM}_cut_s1-s2_matte" / frames[0]), cv2.IMREAD_UNCHANGED)
    assert m.ndim == 2 and m.max() == 200                          # PNG, gray, exactly the rendered values (lossless)
    meta = json.loads((tmp_path / out[0]).read_text(encoding="utf-8"))
    assert meta["matte_through"] is True and meta["depth"] == ["given", "depth-anything-v2-small"] and meta["i2"] == 5
    (tmp_path / "x").mkdir()
    assert AI.write_cuts(tmp_path / "x", STEM, clips, heavy=False) == [f"{STEM}_cut_s1-s2_depth_pivot.json"]          # light: the plan json only
    assert AI.write_cuts(tmp_path, STEM, None) == [] and AI.write_cuts(tmp_path, STEM, [{"sid": "s1"}]) == []
    # depth: first frame always, per-frame sequence with heavy
    dep = torch.rand(20, 8, 4, 3)
    assert AI.write_depth_stills(tmp_path, STEM, "s1", dep) == [f"{STEM}_s1_depth.png", f"{STEM}_s1_depth/"]
    assert len(os.listdir(tmp_path / f"{STEM}_s1_depth")) == 20 and (tmp_path / f"{STEM}_s1_depth" / f"{STEM}_s1_depth_00020.png").is_file()
    assert AI.write_depth_stills(tmp_path, STEM, "s2", dep, heavy=False) == [f"{STEM}_s2_depth.png"] and not (tmp_path / f"{STEM}_s2_depth").exists()
    assert AI.write_depth_stills(tmp_path, STEM, "s3", None) == []
    # clean plate / composed shots
    pl = {"sid": "s1", "plate": True, "subject": torch.zeros(4, 8, 4, 3, dtype=torch.uint8), "u8": torch.randint(0, 255, (4, 8, 4, 3), dtype=torch.uint8)}
    assert AI.write_plate(tmp_path, STEM, "s1", pl) == [f"{STEM}_s1_plate.mp4"] and AI.write_plate(tmp_path, STEM, "s1", {"sid": "s1", "u8": pl["u8"]}) == []
    comp = [{"sid": "s1", "u8": torch.randint(0, 255, (4, 16, 8, 3), dtype=torch.uint8)}, {"sid": "s2", "u8": torch.randint(0, 255, (4, 16, 8, 3), dtype=torch.uint8)}]
    assert AI.write_composed(tmp_path, STEM, comp) == [f"{STEM}_s1_comp.mp4", f"{STEM}_s2_comp.mp4"] and AI.write_composed(tmp_path, STEM, None) == []
    # design sheet, type matte PNGs (renumbered from 1, lossless)
    assert AI.write_design_sheet(tmp_path, STEM, torch.rand(1, 12, 20, 3)) == [f"{STEM}_design_sheet.png"] and AI.write_design_sheet(tmp_path, STEM, None) == []
    assert cv2.imread(str(tmp_path / f"{STEM}_design_sheet.png")).shape == (12, 20, 3)
    d = tmp_path / "layers"
    for i in range(3):
        _png(d / "matte" / f"matte_{i:05d}.png", 50 * i)
    assert AI.write_type_matte_png(tmp_path, STEM, {"dir_layers": str(d)}) == [f"{STEM}_type_matte/"]
    assert sorted(os.listdir(tmp_path / f"{STEM}_type_matte")) == [f"{STEM}_type_matte_{i:05d}.png" for i in (1, 2, 3)]
    assert cv2.imread(str(tmp_path / f"{STEM}_type_matte" / f"{STEM}_type_matte_00003.png"))[0, 0, 0] == 100
    assert AI.write_type_matte_png(tmp_path, STEM, None) == [] and AI.write_type_matte_png(tmp_path, STEM, {"dir_layers": str(tmp_path / "none")}) == []
    # window sources (the compose ops' refs to other shots -> the archive files that hold them)
    from tools.fusion import window_cut
    orig = window_cut.other_refs
    window_cut.other_refs = lambda piece, sid: [("next", "s2", "footage"), ("next_matte", "s2", "matte"), ("next_subject", "s2", "subject")]
    try:
        assert AI.write_sources(tmp_path, STEM, "s1", {"shots": [{"id": "s1"}, {"id": "s2"}]}) == [f"{STEM}_s1_sources.json"]
        rows = json.loads((tmp_path / f"{STEM}_s1_sources.json").read_text(encoding="utf-8"))
        assert [r["file"] for r in rows] == [f"{STEM}_s2_clip.mp4", f"{STEM}_s2_matte/", f"{STEM}_s2_clean.mp4"]
        window_cut.other_refs = lambda piece, sid: []
        assert AI.write_sources(tmp_path, STEM, "s2", {"shots": [{"id": "s1"}, {"id": "s2"}]}) == []
    finally:
        window_cut.other_refs = orig
    # places: the angle-still source and edit prompt
    piece = {"stills": [{"id": "A", "prompt": "a hall"}, {"id": "B", "prompt": "b"}, {"id": "C", "prompt": "c", "angle_of": "A", "framing": "close-up"}]}
    assert AI.write_places_meta(tmp_path, STEM, piece) == [f"{STEM}_places.json"]
    pm = json.loads((tmp_path / f"{STEM}_places.json").read_text(encoding="utf-8"))
    assert [r["file"] for r in pm] == [f"{STEM}_still_{k}.png" for k in (1, 2, 3)] and pm[2]["angle_source_place"] == 1 and pm[2]["angle_source_file"] == f"{STEM}_still_1.png"
    assert pm[0]["angle_source_place"] is None and AI.write_places_meta(tmp_path, STEM, {}) == []
    # the size estimate counts folders recursively and separates the heavy part
    sz = AI.sizes(tmp_path, [f"{STEM}_s1_comp.mp4", f"{STEM}_s1_depth.png", f"{STEM}_s1_depth/", f"{STEM}_cut_s1-s2_matte/", "missing.bin"])
    assert sz["files"] == 1 + 1 + 20 + 12 and 0 < sz["heavy"] < sz["total"]


def test_contour_kinds_full_adds_the_two_handoff_traces(AI):
    p = {"type": {"words": [{"text": "A", "render": "handoff"}, {"text": "B"}]}}
    assert AI.contour_kinds_used(p) == [] and AI.contour_kinds_used(p, full=True) == ["matte_outline", "luma_edges"]
    assert AI.contour_kinds_used({"type": {"words": [{"text": "A", "render": "flat"}]}}, full=True) == []


# ------------------------------------------------------------------------------------------------ the node
S = "mg_r01"        # short stem here: cv2.imwrite fails past 260 chars on Windows and pytest's tmp dir is long


def _archive_run(A, tmp_path, monkeypatch, full, prefix_dir="v010"):
    torch = pytest.importorskip("torch")

    class FP:
        @staticmethod
        def get_output_directory():
            return str(tmp_path)
    monkeypatch.setattr(A, "folder_paths", FP)

    class Comp:
        audio = {"waveform": torch.zeros(1, 2, 4800), "sample_rate": 48000}

    class Vid:
        @staticmethod
        def get_components():
            return Comp
    n, h, w = 16, 32, 16
    clips = [{"sid": "s1", "u8": torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8), "start": 0},
             {"sid": "s2", "u8": torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8), "start": n,
              "plate": True, "subject": torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8)}]
    cuts = [dict(clips[0], cut_renders=[PC.cut_record(_fake_cut_result(n, n, h, w))]), clips[1]]
    composed = [{"sid": s, "u8": torch.randint(0, 255, (n, 2 * h, 2 * w, 3), dtype=torch.uint8), "start": 0} for s in ("s1", "s2")]
    ld = tmp_path / "typelayers"
    for i in range(n):
        _png(ld / "matte" / f"matte_{i:05d}.png", i)
    piece = {"id": "P", "layout": {"s1": {"start": 0, "frames": n}, "s2": {"start": n, "frames": n}}, "type": {"words": [{"text": "A"}]},
             "stills": [{"id": "A", "prompt": "x"}, {"id": "B", "prompt": "y"}]}
    prefix = f"projects/x/{prefix_dir}/{S}_out"
    imgs = {"matte_1": torch.rand(n, h, w, 3), "matte_2": torch.rand(n, h, w, 3), "depth_1": torch.rand(n, h, w, 3), "depth_2": torch.rand(n, h, w, 3)}
    A.MographArchive.execute(prefix, piece, clips, torch.rand(1, h, w, 3), video=Vid, places=torch.rand(2, h, w, 3), type={"dir_layers": str(ld)},
                             full_archive=full, composed=composed, cuts=cuts, sheet=torch.rand(1, 24, 40, 3), **imgs)
    return tmp_path / "projects" / "x" / prefix_dir / f"{S}_out_shots"


def test_mograph_archive_full_archive_on_writes_every_component(AI, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    from nodes_fusion import archive as A
    root = _archive_run(A, tmp_path, monkeypatch, True)
    names = set(os.listdir(root))
    want = {f"{S}_cut_s1-s2_depth_pivot.mp4", f"{S}_cut_s1-s2_depth_pivot.json", f"{S}_cut_s1-s2_matte", f"{S}_s1_comp.mp4", f"{S}_s2_comp.mp4",
            f"{S}_s1_depth.png", f"{S}_s2_depth.png", f"{S}_s1_depth", f"{S}_s2_depth", f"{S}_s1_depth.mp4", f"{S}_s2_plate.mp4", f"{S}_s2_clean.mp4",
            f"{S}_type_matte", f"{S}_design_sheet.png", f"{S}_places.json", f"{S}_still_1.png", f"{S}_still_2.png", f"{S}_music.wav",
            f"{S}_s1_matte", f"{S}_s2_matte", f"{S}_piece.json", f"{S}_index.json"}
    assert want <= names, sorted(want - names)
    legacy = {"still.png", "s1_matte", "s2_matte", "s1_clip.mp4", "s2_clean.mp4", "piece.json", "index.json"}
    assert all(n.startswith(S + "_") or n in legacy for n in names), sorted(names)
    assert len(os.listdir(root / f"{S}_type_matte")) == 16 and len(os.listdir(root / f"{S}_s1_depth")) == 16
    idx = json.loads((root / f"{S}_index.json").read_text(encoding="utf-8"))
    assert idx["full_archive"] is True and idx["bytes"]["heavy"] > 0 and idx["bytes"]["total"] > idx["bytes"]["heavy"]
    assert f"{S}_cut_s1-s2_matte/" in idx["files"] and f"{S}_s1_comp.mp4" in idx["files"] and idx["shots"]["s2"]["layers"]["plate"] == f"{S}_s2_plate.mp4"


def test_mograph_archive_full_archive_off_skips_the_heavy_extras(AI, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    from nodes_fusion import archive as A
    root = _archive_run(A, tmp_path, monkeypatch, False)
    names = set(os.listdir(root))
    assert not any(IN.is_heavy(n) for n in names), sorted(n for n in names if IN.is_heavy(n))
    assert {f"{S}_cut_s1-s2_depth_pivot.json", f"{S}_s1_depth.png", f"{S}_s2_plate.mp4", f"{S}_design_sheet.png", f"{S}_s1_matte", f"{S}_s1_clip.mp4"} <= names
    idx = json.loads((root / f"{S}_index.json").read_text(encoding="utf-8"))
    assert idx["full_archive"] is False and idx["bytes"]["heavy"] == 0
    monkeypatch.setenv("MOTNGRFX_ARCHIVE_FULL", "0")                                    # the environment switch does the same with the input left ON
    root2 = _archive_run(A, tmp_path, monkeypatch, True, prefix_dir="v011")
    assert not any(IN.is_heavy(n) for n in os.listdir(root2))


def test_the_light_archive_keeps_the_design_sheet_and_the_composed_shots(AI, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    from nodes_fusion import archive as A

    class FP:
        @staticmethod
        def get_output_directory():
            return str(tmp_path)
    monkeypatch.setattr(A, "folder_paths", FP)
    clips = [{"sid": "s1", "u8": torch.zeros(4, 8, 4, 3, dtype=torch.uint8), "start": 0}]
    comp = [{"sid": "s1", "u8": torch.randint(0, 255, (4, 16, 8, 3), dtype=torch.uint8)}]
    A.MographArchive.execute(f"redesign/{S}_out", {"id": "P"}, clips, torch.rand(1, 8, 4, 3), video=None, light=True, composed=comp, sheet=torch.rand(1, 6, 10, 3))
    root = tmp_path / "redesign" / f"{S}_out_shots"
    names = set(os.listdir(root))
    assert {f"{S}_s1_comp.mp4", f"{S}_design_sheet.png", f"{S}_piece.json", f"{S}_index.json"} <= names and f"{S}_s1_clip.mp4" not in names


def test_the_archive_asks_for_the_depth_and_matte_of_the_shots_a_cut_names(AI):
    from nodes_fusion import archive as A
    cuts = [{"sid": "s1", "cut_renders": [PC.cut_record(_fake_cut_result())]}]
    need = A.MographArchive.check_lazy_status("p", {"layout": {}}, None, [{"sid": "s1", "u8": None}], None, None, cuts=cuts)
    assert sorted(need) == ["depth_1", "depth_2", "matte_1", "matte_2"]


# ------------------------------------------------------------------------------------------------ pull / manifest of the whole set
def test_pull_links_the_new_components_and_the_manifest_has_names_and_size(tmp_path):
    v = make_version(tmp_path, legacy=False)
    sd = v / "out" / f"{STEM}_out_shots"
    for i in range(1, 19):
        _png(sd / f"{STEM}_cut_s1-s2_matte" / f"{STEM}_cut_s1-s2_matte_{i:05d}.png", i)
        _png(sd / f"{STEM}_type_matte" / f"{STEM}_type_matte_{i:05d}.png", i)
    for n, size in ((f"{STEM}_cut_s1-s2_depth_pivot.mp4", 5000), (f"{STEM}_s1_comp.mp4", 7000), (f"{STEM}_s1_depth.png", 40), (f"{STEM}_s2_plate.mp4", 300),
                    (f"{STEM}_design_sheet.png", 90), (f"{STEM}_cut_s1-s2_depth_pivot.json", 20), (f"{STEM}_s1_sources.json", 20), (f"{STEM}_places.json", 20)):
        (sd / n).write_bytes(b"x" * size)
    rep = ING.pull_run(str(v), ING.find_runs(str(v))[0], None, None, log=lambda *a: None)
    ing = v / "ingredients" / STEM
    got = set(os.listdir(ing))
    assert {f"{STEM}_cut_s1-s2_depth_pivot.mp4", f"{STEM}_cut_s1-s2_matte", f"{STEM}_s1_comp.mp4", f"{STEM}_s1_depth.png", f"{STEM}_s2_plate.mp4", f"{STEM}_type_matte",
            f"{STEM}_design_sheet.png", f"{STEM}_places.json"} <= got
    assert os.path.samefile(ing / f"{STEM}_s1_comp.mp4", sd / f"{STEM}_s1_comp.mp4") and os.stat(ing / f"{STEM}_s1_comp.mp4").st_nlink >= 2         # hardlink
    assert os.path.samefile(ing / f"{STEM}_cut_s1-s2_matte" / f"{STEM}_cut_s1-s2_matte_00018.png", sd / f"{STEM}_cut_s1-s2_matte" / f"{STEM}_cut_s1-s2_matte_00018.png")
    assert rep["stack"] and ING.collapse_check(str(ing))["matte_dirs"] == 4           # s1, s2 mattes + the cut matte + the type matte: all PNG sequences of one stack
    man = (ing / f"{STEM}_MANIFEST.md").read_text(encoding="utf-8")
    assert "3D shot cut s1 -> s2: the subject matte THROUGH" in man and "18 PNG frames" in man and "composited shot" in man and "CLEAN PLATE" in man
    assert "type matte pass as PNG" in man and "sheet image the vision model" in man and "depth of the FIRST frame" in man
    first = [ln for ln in man.splitlines() if ln.startswith("Size of this composition")][0]
    r = ING.size_report(STEM, str(ing))
    assert r["full_archive_extras"] >= 5000 + 7000 and r["total"] > r["full_archive_extras"] and "full_archive extras" in first and f"{r['files']} files" in first
    assert r["kinds"]["mp4"] >= 12300
