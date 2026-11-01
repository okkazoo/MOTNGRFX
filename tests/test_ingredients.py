"""Ingredients of a composition under one name: naming (tools/fusion/ingredient_names.py), MographArchive's extras (nodes_fusion/archive_ingredients.py,
archive.py), and the local gather (tools/app/ingredients.py: hardlinks, manifest, stack-collapse layout). No box, no network, no GPU: the box is faked.
Run: python -m pytest tests/test_ingredients.py -q"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools" / "app")):
    if _p not in sys.path:
        sys.path.append(_p)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from tools.fusion import ingredient_names as IN  # noqa: E402
import ingredients as ING  # noqa: E402

STEM = "motion-graphic_batch-24gb_v002_r01"


# ------------------------------------------------------------------------------------------------ naming
def test_run_stem_from_the_archive_prefix():
    assert IN.run_stem("projects/COMPETITIONS/motion-graphic/batch-24gb/v002/motion-graphic_batch-24gb_v002_r01_out") == STEM
    assert IN.run_stem(STEM + "_out_shots") == STEM
    assert IN.run_stem("archive/motngrfx") == "motngrfx"
    assert IN.run_stem("redesign/out") == "out" or IN.run_stem("redesign/out")           # never empty


def test_every_name_carries_the_stem():
    names = [IN.still(STEM), IN.still(STEM, 3), IN.shot_file(STEM, "s2", "clip"), IN.shot_file(STEM, "s2", "textbg"), IN.matte_dir(STEM, "s2"),
             IN.matte_frame(STEM, "s2", 7), IN.contour_file(STEM, "s1", "luma_edges"), IN.contour_file(STEM, "s1", "luma_edges", "npz"),
             IN.type_pass_file(STEM, "beauty"), IN.ltx_read_crop(STEM, "s2", 1), IN.nm(STEM, "piece.json"), IN.nm(STEM, "music.wav")]
    assert all(IN.has_stem(STEM, n) for n in names)
    assert IN.matte_frame(STEM, "s2", 7) == STEM + "_s2_matte_00007.png"


def test_legacy_names_map_to_prefixed_ones():
    assert IN.from_legacy(STEM, "still.png") == STEM + "_still.png"
    assert IN.from_legacy(STEM, "s3_clean.mp4") == STEM + "_s3_clean.mp4"
    assert IN.from_legacy(STEM, "piece.json") == STEM + "_piece.json"
    assert IN.from_legacy(STEM, "s1_matte/00012.png") == f"{STEM}_s1_matte/{STEM}_s1_matte_00012.png"
    assert IN.from_legacy(STEM, "something_else.bin") is None
    assert IN.to_new(STEM, f"{STEM}_s1_matte/{STEM}_s1_matte_00001.png") == f"{STEM}_s1_matte/{STEM}_s1_matte_00001.png"      # already prefixed: as is
    assert IN.to_new(STEM, "notes.txt") is None


def test_matte_frames_follow_the_dashboard_sequence_rule():
    m = re.match(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<suffix>_?)\.(?P<ext>png|jpe?g|webp)$", IN.matte_frame(STEM, "s1", 5), re.I)
    assert m and m["prefix"] == STEM + "_s1_matte_" and int(m["num"]) == 5


def test_plan_drops_a_legacy_name_when_its_prefixed_twin_exists():
    src = {"still.png": "a", STEM + "_still.png": "b", "s1_clip.mp4": "c", "unknown.txt": "d"}
    plan = ING.plan_files(STEM, src)
    assert plan == {STEM + "_still.png": "b", STEM + "_s1_clip.mp4": "c"}


def test_tar_member_names_are_checked():
    assert ING.safe_member("a/b.png") == "a/b.png" and ING.safe_member("./a/b.png") == "a/b.png"
    for bad in ("/etc/passwd", "../x", "a/../../x", "C:/x", ""):
        assert ING.safe_member(bad) is None


def test_remote_version_dir_drops_the_idea_order_prefix():
    v = os.path.join(ING.PROJECTS_ROOT, "COMPETITIONS", "motion-graphic", "22_batch-24gb", "v002")
    assert ING.remote_version_dir(v) == "/workspace/ComfyUI/output/projects/COMPETITIONS/motion-graphic/batch-24gb/v002"


# ------------------------------------------------------------------------------------------------ the local gather
def _png(path: Path, v: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((16, 8, 3), v, np.uint8))


def make_version(tmp: Path, legacy=True, shots=("s1", "s2"), frames=20) -> Path:
    """a synced version folder: out/<stem>_out_shots (short names as the dashboard sync lands an old archive) + out/<stem>_video_00001_.mp4."""
    v = tmp / "v002"
    sd = v / "out" / f"{STEM}_out_shots"
    for s in shots:
        for i in range(1, frames + 1):
            _png(sd / (f"{s}_matte/{i:05d}.png" if legacy else f"{STEM}_{s}_matte/{STEM}_{s}_matte_{i:05d}.png"), i)
        (sd / (f"{s}_clean.mp4" if legacy else f"{STEM}_{s}_clean.mp4")).write_bytes(b"clean" + s.encode())
    _png(sd / ("still.png" if legacy else f"{STEM}_still.png"))
    (v / "out" / f"{STEM}_video_00001_.mp4").write_bytes(b"not a real mp4")
    (v / "review").mkdir()
    (v / "review" / f"{STEM}_out_matte_preview.mp4").write_bytes(b"preview")
    return v


def same_file(a: Path, b: Path) -> bool:
    return os.path.samefile(a, b)


def test_pull_run_builds_hardlinked_prefixed_folder_without_a_box(tmp_path):
    v = make_version(tmp_path)
    runs = ING.find_runs(str(v))
    assert len(runs) == 1 and runs[0]["stem"] == STEM and runs[0]["part"] == "r01" and runs[0]["final"] and runs[0]["shots_dir"]
    rec = {"concept": {"title": "T", "words": ["A"]}, "design_reply": "raw reply", "design_check": "ok"}
    rep = ING.pull_run(str(v), runs[0], None, rec, log=lambda *a: None)
    ing = v / "ingredients" / STEM
    names = sorted(os.listdir(ing))
    assert all(n.startswith(STEM + "_") for n in names), names                          # every name carries the stem
    assert f"{STEM}_video.mp4" in names and f"{STEM}_still.png" in names and f"{STEM}_s1_clean.mp4" in names and f"{STEM}_MANIFEST.md" in names
    assert f"{STEM}_mattes_stacked.mp4" in names and f"{STEM}_concept.json" in names and f"{STEM}_design_reply.txt" in names
    assert json.loads((ing / f"{STEM}_concept.json").read_text(encoding="utf-8"))["title"] == "T"
    # hardlinks, not copies: same file as the origin in out/
    assert same_file(ing / f"{STEM}_video.mp4", v / "out" / f"{STEM}_video_00001_.mp4")
    assert same_file(ing / f"{STEM}_s1_clean.mp4", v / "out" / f"{STEM}_out_shots" / "s1_clean.mp4")
    assert same_file(ing / f"{STEM}_s1_matte" / f"{STEM}_s1_matte_00003.png", v / "out" / f"{STEM}_out_shots" / "s1_matte" / "00003.png")
    assert os.stat(ing / f"{STEM}_video.mp4").st_nlink >= 2
    # the outputs stay where they were (out/ untouched)
    assert (v / "out" / f"{STEM}_out_shots" / "s1_matte" / "00001.png").is_file()
    assert rep["stack"] and len(os.listdir(ing / f"{STEM}_s1_matte")) == 20
    man = (ing / f"{STEM}_MANIFEST.md").read_text(encoding="utf-8")
    assert f"{STEM}_s1_matte/" in man and "20 PNG frames" in man and "HARDLINKS" in man and "MographLayerClean" in man
    # idempotent
    rep2 = ING.pull_run(str(v), runs[0], None, rec, log=lambda *a: None)
    assert rep2["linked"] == 0 and not rep2["skipped"]


def test_prefixed_archive_needs_no_mapping(tmp_path):
    v = make_version(tmp_path, legacy=False)
    rep = ING.pull_run(str(v), ING.find_runs(str(v))[0], None, None, log=lambda *a: None)
    ing = v / "ingredients" / STEM
    assert rep["stack"] and (ing / f"{STEM}_s2_matte" / f"{STEM}_s2_matte_00020.png").is_file() and (ing / f"{STEM}_still.png").is_file()


def test_one_matte_folder_is_not_a_stack_and_the_manifest_says_so(tmp_path):
    v = make_version(tmp_path, shots=("s1",))
    rep = ING.pull_run(str(v), ING.find_runs(str(v))[0], None, None, log=lambda *a: None)
    assert not rep["stack"]
    assert "will not collapse" in (v / "ingredients" / STEM / f"{STEM}_MANIFEST.md").read_text(encoding="utf-8")


def test_model_replies_of_the_run_are_linked_in(tmp_path):
    v = make_version(tmp_path)
    md = v / "review" / "model_replies"
    md.mkdir()
    for n in ("r01_design_reply.txt", "r01_concept.json", "r01_design_prompt.reconstructed.txt", "r01_piece.json", "r02_design_reply.txt"):
        (md / n).write_text("x", encoding="utf-8")
    ING.pull_run(str(v), ING.find_runs(str(v))[0], None, None, log=lambda *a: None)
    got = set(os.listdir(v / "ingredients" / STEM))
    assert {f"{STEM}_design_reply.txt", f"{STEM}_concept.json", f"{STEM}_design_prompt.reconstructed.txt"} <= got
    assert f"{STEM}_piece.json" not in got and not any("r02" in n for n in got)         # r01's piece.json is the archive's, other runs stay out


def test_box_fetch_skips_what_the_sync_has_and_legacy_twins(tmp_path):
    v = make_version(tmp_path)                                # synced: legacy matte + clean + still
    sd = v / "out" / f"{STEM}_out_shots"
    size = (sd / "s1_clean.mp4").stat().st_size
    listing = {"s1_clean.mp4": size,                          # synced, same size -> not fetched
               "s1_matte/00001.png": (sd / "s1_matte" / "00001.png").stat().st_size,
               "piece.json": 10, "index.json": 5,             # json: the sync skips it -> fetched
               f"{STEM}_piece.json": 10,                      # prefixed twin of piece.json -> the legacy one is not fetched
               "s5_clean.mp4": 7,                             # not synced yet -> fetched
               "readme.txt": 3}                               # not an ingredient

    fetched = {}

    def fetcher(box, rdir, rels, dest, log):
        fetched["rels"] = list(rels)
        for r in rels:
            p = Path(dest, *r.split("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * listing[r])
        return len(rels)
    rep = ING.pull_run(str(v), ING.find_runs(str(v))[0], {"host": "h", "port": 1, "user": "root"}, None, log=lambda *a: None,
                       lister=lambda box, rdir: listing, fetcher=fetcher)
    assert sorted(fetched["rels"]) == sorted(["index.json", f"{STEM}_piece.json", "s5_clean.mp4"])
    ing = v / "ingredients" / STEM
    assert (ing / f"{STEM}_piece.json").is_file() and (ing / f"{STEM}_index.json").is_file() and (ing / f"{STEM}_s5_clean.mp4").is_file()
    assert rep["box"].endswith("3 fetched")
    # a second pull finds everything in the pull folder: nothing is fetched again
    fetched.clear()
    ING.pull_run(str(v), ING.find_runs(str(v))[0], {"host": "h", "port": 1, "user": "root"}, None, log=lambda *a: None,
                 lister=lambda box, rdir: listing, fetcher=fetcher)
    assert fetched == {}


def test_gone_box_falls_back_to_local_files(tmp_path):
    v = make_version(tmp_path)
    logs = []
    rep = ING.pull_run(str(v), ING.find_runs(str(v))[0], {"host": "h", "port": 1, "user": "root"}, None, log=logs.append,
                       lister=lambda box, rdir: None, fetcher=lambda *a: pytest.fail("no fetch without a listing"))
    assert rep["box"] == "unreachable" and (v / "ingredients" / STEM / f"{STEM}_s1_clean.mp4").is_file()


def test_dashboard_collapses_the_folder_into_one_stack(tmp_path):
    """the real output_collapse rule of the vastai-api-manager (skipped when that repo is not next to PROJECTS)."""
    api = Path.home() / "Documents" / "Dev" / "vastai-api-manager"
    if not (api / "core" / "output_collapse.py").is_file():
        pytest.skip("vastai-api-manager not found")
    sys.path.insert(0, str(api))
    try:
        from core import output_collapse as OC
    except Exception as e:       # noqa: BLE001
        pytest.skip(f"output_collapse not importable: {e}")
    v = make_version(tmp_path)
    ING.pull_run(str(v), ING.find_runs(str(v))[0], None, None, log=lambda *a: None)
    rows = []
    base = tmp_path
    for root, _, files in os.walk(v / "ingredients"):
        for f in files:
            p = Path(root, f)
            rel = p.relative_to(base).as_posix()
            ext = p.suffix.lower()
            rows.append({"root_id": "r", "relpath": rel, "ext": ext, "token": rel, "filename": f, "size": p.stat().st_size, "mtime": p.stat().st_mtime,
                         "width": 8 if ext == ".png" else None, "height": 16 if ext == ".png" else None, "day": "d", "row_key": rel})
    ents = OC.collapse(rows, lambda r: {"filename": r["filename"], "path": r["token"], "size": r["size"], "mtime": r["mtime"]}, "r")
    assert len(ents) == 1, [e["item"]["filename"] for e in ents]
    it = ents[0]["item"]
    assert it["kind"] == "stack" and it["filename"] == f"{STEM}.stack" and it["counts"]["sequences"] == 2
    assert any(m["filename"] == f"{STEM}_video.mp4" for m in it["members"])


# ------------------------------------------------------------------------------------------------ MographArchive (needs a ComfyUI checkout for comfy_api)
@pytest.fixture(scope="module")
def AI():
    for c in (str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if (Path(c) / "comfy_api" / "latest" / "__init__.py").is_file():
            sys.path.insert(0, c)
            break
    else:
        pytest.skip("no ComfyUI checkout")
    try:
        from nodes_fusion import archive_ingredients as m
    except Exception as e:       # noqa: BLE001
        pytest.skip(f"nodes_fusion not importable here: {e}")
    return m


def test_archive_extras_write_stem_prefixed_files(AI, tmp_path):
    torch = pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    places = torch.rand(2, 32, 16, 3)
    assert AI.write_places(tmp_path, STEM, places) == [f"{STEM}_still_1.png", f"{STEM}_still_2.png"]
    fl = torch.zeros(6, 16, 8, 2, dtype=torch.float16)
    fl[..., 0] = 1.0
    out = AI.write_shot_previews(tmp_path, STEM, "s1", {"flow": fl}, depth=torch.rand(6, 32, 16, 3))
    assert out == [f"{STEM}_s1_depth.mp4", f"{STEM}_s1_flow.mp4"] and all((tmp_path / n).stat().st_size > 0 for n in out)
    # type passes: png sequences -> one mp4 per pass, flow from flow_vis
    d = tmp_path / "layers"
    for p, sub in (("beauty", "beauty"), ("matte", "matte"), ("flow", "flow_vis")):
        for i in range(4):
            _png(d / sub / f"{sub}_{i:05d}.png", 40 * i)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    got = AI.write_type_passes(tmp_path, STEM, {"dir_layers": str(d)})
    assert sorted(got) == sorted([f"{STEM}_type_beauty.mp4", f"{STEM}_type_matte.mp4", f"{STEM}_type_flow.mp4", f"{STEM}_type_manifest.json"])
    assert AI.write_places(tmp_path, STEM, None) == [] and AI.write_type_passes(tmp_path, STEM, None) == []


def test_contour_kinds_come_from_the_ports_the_words_use(AI):
    piece = {"type": {"words": [{"text": "A", "entry": {"port": "matte_outline"}, "exit": "down"}, {"text": "B", "via": "luma_edges"}, {"text": "C"}]},
             "compose": {"default": {"ops": [{"op": "edge_run_type", "args": {"source": "depth_iso"}}]}}}
    assert AI.contour_kinds_used(piece) == ["matte_outline", "luma_edges", "depth_iso"]
    assert AI.contour_kinds_used({"type": {"words": [{"text": "A"}]}}) == []


def test_contour_export_writes_tracks_for_a_matte_shot(AI, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    n, h, w = 8, 96, 64
    matte = np.zeros((n, h, w), np.uint8)
    plate = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        cv2.circle(matte[i], (20 + 3 * i, 48), 18, 255, -1)
        plate[i] = matte[i][..., None]
    piece = {"type": {"words": [{"text": "A", "entry": {"port": "matte_outline"}}]}}
    out = AI.write_contours(tmp_path, STEM, "s1", piece, plate, matte=matte)
    assert out == [f"{STEM}_s1_contour_matte_outline.mp4", f"{STEM}_s1_contour_matte_outline.npz"]
    z = np.load(tmp_path / out[1])
    assert z["P"].shape[0] == n and z["valid"].any()
    assert AI.write_contours(tmp_path, STEM, "s2", {"type": {"words": [{"text": "A", "entry": {"port": "depth_iso"}}]}}, plate, matte=matte) == []     # no depth: skipped


def test_mograph_archive_node_writes_prefixed_and_legacy_names(AI, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    from nodes_fusion import archive as A

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
    n, h, w = 6, 32, 16
    clips = [{"sid": s, "u8": torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8), "start": 0} for s in ("s1", "s2")]
    piece = {"id": "P", "layout": {"s1": {"start": 0, "frames": n}, "s2": {"start": n, "frames": n}}, "type": {"words": [{"text": "A"}]}}
    prefix = f"projects/x/v002/{STEM}_out"
    imgs = {"matte_1": torch.rand(n, h, w, 3), "matte_2": torch.rand(n, h, w, 3)}
    A.MographArchive.execute(prefix, piece, clips, torch.rand(1, h, w, 3), video=Vid, places=torch.rand(2, h, w, 3), **imgs)
    root = tmp_path / "projects" / "x" / "v002" / f"{STEM}_out_shots"
    names = set(os.listdir(root))
    assert {f"{STEM}_still.png", f"{STEM}_still_1.png", f"{STEM}_still_2.png", f"{STEM}_s1_clip.mp4", f"{STEM}_s2_clip.mp4", f"{STEM}_s1_matte", f"{STEM}_s2_matte",
            f"{STEM}_music.wav", f"{STEM}_piece.json", f"{STEM}_index.json"} <= names
    assert {"still.png", "s1_clip.mp4", "s1_matte", "piece.json", "index.json"} <= names                    # the short names old readers use
    assert os.path.samefile(root / "s1_clip.mp4", root / f"{STEM}_s1_clip.mp4")                               # hardlinks: no extra disk
    assert len(os.listdir(root / f"{STEM}_s1_matte")) == n and (root / "s1_matte" / "00001.png").is_file()
    assert (root / f"{STEM}_s1_matte" / f"{STEM}_s1_matte_00001.png").is_file()
    idx = json.loads((root / f"{STEM}_index.json").read_text(encoding="utf-8"))
    assert idx["stem"] == STEM and f"{STEM}_music.wav" in idx["files"] and f"{STEM}_s1_matte/" in idx["files"]
    monkeypatch.setenv("MOTNGRFX_ARCHIVE_EXTRAS", "0")
    A.MographArchive.execute(prefix.replace("/v002/", "/v003/"), piece, clips, torch.rand(1, h, w, 3), video=Vid, places=torch.rand(2, h, w, 3))
    r3 = tmp_path / "projects" / "x" / "v003" / f"{STEM}_out_shots"
    assert not (r3 / f"{STEM}_music.wav").exists() and (r3 / f"{STEM}_s1_clip.mp4").is_file()
