"""scripts/download_models.py and scripts/check_install.py must stay in step with the app graphs (stdlib only, no ComfyUI)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _common as C  # noqa: E402


def test_every_model_in_the_gpu_graphs_has_a_download_entry():
    used = C.model_files(C.graph_files(["fusion*.json"], downloadable=True))
    assert used, "no model files found in app/graphs/fusion*.json"
    missing = sorted(f for f in used if f not in C.MODELS)
    assert not missing, f"add these to scripts/_common.py MODELS (name, folder, size, url): {missing}"


def test_model_table_is_well_formed():
    for name, m in C.MODELS.items():
        public = m["url"].startswith("https://huggingface.co/") or (m["url"].startswith("https://github.com/") and "/releases/download/" in m["url"])  # official GitHub releases (Real-ESRGAN)
        assert public and m["url"].endswith(name.split("/")[-1]), name
        assert m["folder"] and m["size"] > 1_000_000, name
        assert "pub-" not in m["url"], f"{name}: use the public Hugging Face URL, not a private mirror"


def test_every_non_core_node_class_maps_to_a_pack():
    """Classes that neither this pack nor ComfyUI core provide must be listed in web_app/node_packs.json (page + check_install)."""
    packs = C.load_packs()
    assert all(p.get("name") for p in packs["packs"])
    known_pack_classes = {c for p in packs["packs"] for c in p.get("classes", [])}
    # the three LTX IC-LoRA classes and the depth classes are the ones the box lacked without their packs
    for cls in ("LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "LoadVideoDepthAnythingModel", "VideoDepthAnythingProcess"):
        assert cls in known_pack_classes, cls
    assert C.pack_for_class("MographCompose", packs)["name"].startswith("MOTNGRFX")


def test_graph_classes_are_mograph_core_or_known_packs():
    """No graph references a class we cannot explain: Mograph*, a mapped pack class, or a core node (checked by name shape)."""
    packs = C.load_packs()
    for g in C.graph_files(["*.json"]):
        for cls in C.graph_classes(C.load_graph(g)):
            if cls.startswith("Mograph") or C.pack_for_class(cls, packs):
                continue
            assert re.match(r"^[A-Za-z][A-Za-z0-9_.]*$", cls), (g.name, cls)


def test_sidecars_optional_nodes_name_real_nodes():
    for g in C.graph_files(["*.json"]):
        side = C.GRAPHS_DIR / f"{g.stem}.app.json"
        if not side.is_file():
            continue
        graph, app = C.load_graph(g), json.loads(side.read_text(encoding="utf-8"))
        titles = {(n.get("_meta") or {}).get("title") for n in graph.values()}
        for spec in app.get("optional_nodes", []):
            if isinstance(spec, dict):
                for t in spec.get("titles", []):
                    assert t in titles, f"{g.stem}: optional_nodes title {t!r} is not in the graph"


# ---------------------------------------------------------------------------------------------------- download_models.py token + lock
def _dl():
    import download_models as DL
    return DL


def _clean_token_env(monkeypatch, tmp_path):
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))


def test_hf_token_falls_back_to_the_token_file_and_is_never_printed(tmp_path, monkeypatch, capsys):
    DL = _dl()
    _clean_token_env(monkeypatch, tmp_path)
    assert DL.hf_token_source() == (None, "none")
    (tmp_path / "hf_home").mkdir()
    secret = "hf_FAKE_secret_value_123"
    (tmp_path / "hf_home" / "token").write_text(secret + "\n", encoding="utf-8")
    assert DL.hf_token_source() == (secret, "token file")
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dry-run", "--models-dir", str(tmp_path / "models")])
    DL.main()
    out = capsys.readouterr()
    assert "HF token: set (token file)" in out.out
    assert secret not in out.out and secret not in out.err


def test_hf_token_env_wins_over_the_token_file(tmp_path, monkeypatch, capsys):
    DL = _dl()
    _clean_token_env(monkeypatch, tmp_path)
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text("hf_from_file", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert DL.hf_token_source() == ("hf_from_env", "env")
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dry-run", "--models-dir", str(tmp_path / "models")])
    DL.main()
    out = capsys.readouterr().out
    assert "HF token: set (env)" in out and "hf_from_env" not in out


def test_hf_token_not_set_message(tmp_path, monkeypatch, capsys):
    DL = _dl()
    _clean_token_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dry-run", "--models-dir", str(tmp_path / "models")])
    DL.main()
    assert "HF token: not set" in capsys.readouterr().out


def test_download_lock_blocks_a_second_downloader_and_replaces_a_stale_lock(tmp_path):
    import os
    import subprocess
    DL = _dl()
    models = tmp_path / "models"
    lock = DL.acquire_lock(models)
    assert lock.name == ".motngrfx_download.lock" and lock.read_text().strip() == str(os.getpid())
    try:
        DL.acquire_lock(models)                    # second acquire while the first is held
        raise AssertionError("second acquire should have failed")
    except DL.DownloadLocked as e:
        assert str(os.getpid()) in str(e)
    DL.release_lock(lock)
    assert not lock.exists()
    # a lock left by a dead PID is stale and gets replaced
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    lock.write_text(str(p.pid))
    again = DL.acquire_lock(models)
    assert again.read_text().strip() == str(os.getpid())
    DL.release_lock(again)
    # a lock held by a live process (the parent test runner, via a child that sleeps) blocks us
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(str(child.pid))
        try:
            DL.acquire_lock(models)
            raise AssertionError("a live holder should block")
        except DL.DownloadLocked as e:
            assert e.pid == child.pid
    finally:
        child.kill()
        child.wait()
