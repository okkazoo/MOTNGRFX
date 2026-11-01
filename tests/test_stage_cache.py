"""Stage cache (tools/app/stage_cache.py), its use by the own-video runner (matte key etc.), the image-mode node cache (nodes_fusion/node_cache.py) and the cancel route
(tools/app/own_video_route.py JobManager.cancel) with a dummy long-running subprocess. No ComfyUI, no GPU."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import own_video_job as J      # noqa: E402
from tools.app import own_video_route as R    # noqa: E402
from tools.app import stage_cache as SC       # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(d):
    return sorted((p.relative_to(d).as_posix(), sha(p)) for p in Path(d).rglob("*") if p.is_file())


@pytest.fixture(autouse=True)
def cache_on(monkeypatch):
    monkeypatch.delenv("MOTNGRFX_NO_CACHE", raising=False)


def make_job(tmp_path, job):
    """a job folder like the runner's: output/motngrfx_app/own/<job>, the shared input folder, the cache in <root>/motngrfx_cache beside output/."""
    out_root = tmp_path / "output"
    d = out_root / "motngrfx_app" / "own" / job
    (d / "work").mkdir(parents=True)
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    stem = f"motngrfx_own_{job.replace('-', '')}"
    rel = f"motngrfx_app/own/{job}"
    logs, marks = [], []
    jc = SC.JobCache(tmp_path / "motngrfx_cache", d, inp, job, stem, rel, log=logs.append, mark=marks.append)
    return jc, d, inp, stem, rel, logs, marks


# ---------------------------------------------------------------------------------------------------- keys
def test_key_is_stable_and_order_free(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    k1 = SC.make_key("matte", files=[f], ids=["x"], texts=["t"], params={"a": 1, "b": [1, 2]})
    k2 = SC.make_key("matte", files=[f], ids=["x"], texts=["t"], params={"b": [1, 2], "a": 1})
    assert k1 == k2 and len(k1) == 64
    assert SC.make_key("cut", files=[f], ids=["x"], texts=["t"], params={"a": 1, "b": [1, 2]}) != k1           # the stage name is in the key
    assert SC.make_key("matte", files=[f], ids=["x"], texts=["t"], params={"a": 2, "b": [1, 2]}) != k1
    assert SC.make_key("matte", files=[f], ids=["x"], texts=["t"], params={"a": 1, "b": [1, 2]}, version=SC.CACHE_VERSION + 1) != k1


def test_key_follows_file_content_not_mtime(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    k = SC.make_key("s", files=[f])
    os.utime(f, (1, 1))
    assert SC.make_key("s", files=[f]) == k                      # same content, other mtime: same key
    f.write_bytes(b"abd")
    assert SC.make_key("s", files=[f]) != k                      # content changed: new key
    f.write_bytes(b"abc")
    assert SC.make_key("s", files=[f]) == k


def test_code_fingerprint_changes_with_the_code(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    fp = SC.code_fingerprint(tmp_path, ("pkg/*.py",))
    SC._CODE_MEMO.clear()
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n")
    assert SC.code_fingerprint(tmp_path, ("pkg/*.py",)) != fp


def test_no_cache_env(tmp_path, monkeypatch):
    jc, d, inp, stem, rel, logs, marks = make_job(tmp_path, "20261101-120000-aaaa")
    f = d / "work" / "x.json"
    f.write_text("{}")
    assert jc.on and jc.key("s", ids=["1"])
    monkeypatch.setenv("MOTNGRFX_NO_CACHE", "1")
    assert not SC.enabled()
    off = SC.JobCache(tmp_path / "motngrfx_cache", d, inp, "20261101-120000-aaaa", stem, rel)
    assert off.key("s", ids=["1"]) == ""                         # nothing is hashed
    assert off.store("s", "k", [f]) is False
    assert off.restore("s", "k") is None
    assert not (tmp_path / "motngrfx_cache" / "s").exists()


def test_cache_root_is_beside_output_and_env_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("MOTNGRFX_CACHE_DIR", raising=False)
    root = SC.default_cache_root(tmp_path)
    assert root == tmp_path / "motngrfx_cache" and "output" not in root.relative_to(tmp_path).parts      # never inside output/ (the dashboard Outputs sync)
    monkeypatch.setenv("MOTNGRFX_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert SC.default_cache_root(tmp_path) == tmp_path / "elsewhere"


# ---------------------------------------------------------------------------------------------------- matte key
def alpha_like_graph(stem, rel, job_no=1, frames=97):
    """what V3.alpha_graph builds: the clip's file name and the output prefix carry the job's STEM / rel."""
    return {"ref": {"class_type": "LoadVideo", "inputs": {"file": f"{stem}_h{job_no}.mp4"}},
            "gen": {"class_type": "LTXAlpha", "inputs": {"frames": frames, "size": [576, 1024], "seed": 1, "model": "ltx-2.3"}},
            "out": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{rel}/{stem}_alpha_h{job_no}_image"}}}


def test_matte_key_ignores_words_fonts_seed_and_the_job_id_but_follows_the_clip(tmp_path):
    clip_bytes = b"matte clip bytes" * 100
    keys = []
    for i, job in enumerate(("20261101-120000-aaaa", "20261101-130000-bbbb")):
        jc, d, inp, stem, rel, *_ = make_job(tmp_path / f"j{i}", job)
        clip = inp / f"{stem}_h1.mp4"
        clip.write_bytes(clip_bytes)
        keys.append(J.matte_key(jc, clip, alpha_like_graph(stem, rel)))
    assert keys[0] == keys[1]                                    # another job id, another words / fonts / seed (none of them is an input): the same matte
    # the key function takes nothing but the clip and the Alpha-Gen graph
    import inspect
    assert list(inspect.signature(J.matte_key).parameters) == ["sc", "clip", "graph"]
    # a different picture or a different Alpha-Gen setting is a different matte
    jc, d, inp, stem, rel, *_ = make_job(tmp_path / "j2", "20261101-140000-cccc")
    clip = inp / f"{stem}_h1.mp4"
    clip.write_bytes(clip_bytes + b"!")
    assert J.matte_key(jc, clip, alpha_like_graph(stem, rel)) != keys[0]
    clip.write_bytes(clip_bytes)
    assert J.matte_key(jc, clip, alpha_like_graph(stem, rel)) == keys[0]
    assert J.matte_key(jc, clip, alpha_like_graph(stem, rel, frames=121)) != keys[0]


def test_design_key_follows_the_words_in_its_graph_but_a_default_font_never_enters(tmp_path):
    """the design key is the graph: words are in the concept (so they retire it), the page's random default font never reaches the graph (own_video_job.read_brief drops it)."""
    jc, d, inp, stem, rel, *_ = make_job(tmp_path, "20261101-120000-aaaa")
    brief_a = J.read_brief({"on_screen": ["SACRED NOISE"], "story": {"hits": [], "shots": [{"n": 1, "font": "Random Default"}]}})
    brief_b = J.read_brief({"on_screen": ["SACRED NOISE"], "story": {"hits": [], "shots": [{"n": 1, "font": "Another Default"}]}})
    g = lambda b: {"concept_in": {"inputs": {"value": json.dumps({"words": "\n".join(b["on_screen"])})}}, "seed": {"inputs": {"value": 7}}}
    assert jc.key("design", ids=["arch", "wav"], params=dict(graph=g(brief_a))) == jc.key("design", ids=["arch", "wav"], params=dict(graph=g(brief_b)))
    brief_c = J.read_brief({"on_screen": ["SACRED SIGNAL"]})
    assert jc.key("design", ids=["arch", "wav"], params=dict(graph=g(brief_c))) != jc.key("design", ids=["arch", "wav"], params=dict(graph=g(brief_a)))


# ---------------------------------------------------------------------------------------------------- reuse = new files, nothing of the old job touched
def fill_job_a(tmp_path):
    jc, d, inp, stem, rel, logs, marks = make_job(tmp_path, "20261101-120000-aaaa")
    pngs = []
    for i in range(1, 4):
        p = d / f"{stem}_alpha_h1_image_{i:05d}_.png"
        p.write_bytes(b"\x89PNG" + bytes([i]) * 300)
        pngs.append(p)
    sj = d / "work" / "segments.json"
    sj.write_text(json.dumps({"stem": stem, "dir": str(d), "rel": rel, "job": "20261101-120000-aaaa"}))
    h = inp / f"{stem}_h1.mp4"
    h.write_bytes(b"clip")
    return jc, d, inp, stem, rel, logs, marks, pngs, sj, h


def test_put_then_restore_makes_new_files_under_the_new_job_names(tmp_path):
    jc, d, inp, stem, rel, logs, marks, pngs, sj, h = fill_job_a(tmp_path)
    before = tree(d), sha(h)
    key = jc.key("matte", files=[h], params={"g": 1})
    assert jc.store("matte", key, pngs + [sj, h])
    assert (tmp_path / "motngrfx_cache" / "matte" / key / "manifest.json").is_file()
    assert jc.store("matte", key, pngs) is False                  # an existing entry stands

    jb, d2, inp2, stem2, rel2, logs2, marks2 = make_job(tmp_path, "20261101-130000-bbbb")
    jb.store_ = jc.store_                                          # same cache folder (same tmp_path/motngrfx_cache)
    got = jb.restore("matte", key)
    assert got is not None and marks2 == ["matte"] and any("cache hit matte" in m for m in logs2)
    for i in range(1, 4):                                         # the png stack under the NEW stem, same content
        assert (d2 / f"{stem2}_alpha_h1_image_{i:05d}_.png").read_bytes() == (d / f"{stem}_alpha_h1_image_{i:05d}_.png").read_bytes()
    seg = json.loads((d2 / "work" / "segments.json").read_text())
    assert seg == {"stem": stem2, "dir": str(d2), "rel": rel2, "job": "20261101-130000-bbbb"}       # ids / paths of the old job rewritten
    assert (inp2 / f"{stem2}_h1.mp4").read_bytes() == b"clip"      # input-folder file under the new name
    assert not any(stem in p.name for p in d2.rglob("*"))          # the old job's name is not recreated in the new job folder
    # the first job is exactly as it was
    assert (tree(d), sha(h)) == before
    assert not any(p.name.startswith(stem2) for p in d.rglob("*"))


def test_restore_refuses_to_overwrite_and_cleans_up_what_it_made(tmp_path):
    jc, d, inp, stem, rel, logs, marks, pngs, sj, h = fill_job_a(tmp_path)
    key = jc.key("matte", files=[h])
    jc.store("matte", key, pngs)
    jb, d2, inp2, stem2, *_ = make_job(tmp_path, "20261101-130000-bbbb")
    existing = d2 / f"{stem2}_alpha_h1_image_00002_.png"
    existing.write_bytes(b"MINE")
    assert jb.restore("matte", key) is None                       # a file is in the way: a miss
    assert existing.read_bytes() == b"MINE"                       # untouched
    assert sorted(p.name for p in d2.glob("*.png")) == [existing.name]      # the file it had linked first is removed again


def test_hardlink_store_does_not_copy_and_a_vanished_source_never_fails_the_job(tmp_path):
    jc, d, inp, stem, rel, logs, marks, pngs, sj, h = fill_job_a(tmp_path)
    key = jc.key("m", files=[h])
    assert jc.store("m", key, [pngs[0]])
    cached = next((tmp_path / "motngrfx_cache" / "m" / key / "files").rglob("*.png"))
    assert cached.read_bytes() == pngs[0].read_bytes()
    assert jc.store("m", "dead" * 16, [d / "nope.png"]) is False


def test_cancelled_job_dir_files_are_not_deleted_by_the_cache(tmp_path):
    """a stored entry survives its job folder being removed (hardlinks), and a stale temp folder never shows as an entry."""
    jc, d, inp, stem, rel, logs, marks, pngs, sj, h = fill_job_a(tmp_path)
    key = jc.key("matte", files=[h])
    jc.store("matte", key, pngs)
    import shutil
    shutil.rmtree(d)
    jb, d2, inp2, stem2, *_ = make_job(tmp_path, "20261101-130000-bbbb")
    assert jb.restore("matte", key) is not None
    assert len(list(d2.glob("*.png"))) == 3
    stale = tmp_path / "motngrfx_cache" / "matte" / ".tmp-xyz"
    stale.mkdir(parents=True)
    assert not SC.StageCache(tmp_path / "motngrfx_cache").has("matte", ".tmp-xyz")


# ---------------------------------------------------------------------------------------------------- image-mode node cache
def test_node_cache_reuses_a_reply_and_follows_the_inputs(tmp_path, monkeypatch):
    # nodes_fusion/__init__ pulls ComfyUI; load node_cache.py on its own with a stub package
    import importlib.util
    import types
    pkg = types.ModuleType("nodes_fusion_stub")
    pkg.__path__ = [os.path.join(ROOT, "nodes_fusion")]
    pkg.ensure_root_on_path = lambda: ROOT
    sys.modules["nodes_fusion_stub"] = pkg
    spec = importlib.util.spec_from_file_location("nodes_fusion_stub.node_cache", os.path.join(ROOT, "nodes_fusion", "node_cache.py"))
    NC = importlib.util.module_from_spec(spec)
    sys.modules["nodes_fusion_stub.node_cache"] = NC
    spec.loader.exec_module(NC)
    monkeypatch.setenv("MOTNGRFX_CACHE_DIR", str(tmp_path / "c"))

    class Out:
        def __init__(self, *args):
            self.args = args
    calls = []

    class Node:
        @classmethod
        def execute(cls, prompt, seed, keep_model_loaded=False, image=None):
            calls.append((prompt, seed))
            return Out(f"reply:{prompt}:{seed}", json.dumps({"total_s": 3}), "think")
    Node.execute = classmethod(NC.cached_text_outputs(Node.execute.__func__, "gemma", ignore=("keep_model_loaded",), make=lambda v: Out(*v),
                                                      on_hit=lambda v: [v[0], json.dumps(dict(json.loads(v[1]), cache_hit=True)), v[2]]))
    a = Node.execute(prompt="hello", seed=1)
    assert a.args[0] == "reply:hello:1" and len(calls) == 1
    b = Node.execute("hello", 1, keep_model_loaded=True)          # same inputs (a performance-only input differs): reused, no model call
    assert len(calls) == 1 and b.args[0] == "reply:hello:1" and json.loads(b.args[1])["cache_hit"] is True
    Node.execute(prompt="hello", seed=2)                          # another seed
    Node.execute(prompt="hello!", seed=1)                         # another prompt
    assert len(calls) == 3
    monkeypatch.setenv("MOTNGRFX_NO_CACHE", "1")
    Node.execute(prompt="hello", seed=1)
    assert len(calls) == 4                                        # disabled: the node runs


# ---------------------------------------------------------------------------------------------------- cancel route with a dummy long-running subprocess
DUMMY = textwrap.dedent("""
    import subprocess, sys, time, pathlib
    out = sys.argv[sys.argv.index("--out-dir") + 1]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    pathlib.Path(out, "child.pid").write_text(str(child.pid))
    time.sleep(600)
""")


def gone(pid: int) -> bool:
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return st in ("Z", "X")
    except OSError:
        return True


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_cancel_kills_the_runner_and_its_children_and_frees_the_lock(tmp_path):
    out, inp = tmp_path / "output", tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    (inp / "clip.mp4").write_bytes(b"x")
    script = tmp_path / "dummy_runner.py"
    script.write_text(DUMMY)
    mgr = R.JobManager(out, inp, tmp_path, port=1, python=sys.executable, script=script)
    calls = []

    def fake_req(method, path, body=None, timeout=10):
        calls.append((method, path, body))
        if path == "/queue" and method == "GET":
            return {"queue_running": [[0, "p-run", {}]], "queue_pending": [[1, "p-wait", {}], [2, "other-client", {}]]}
        return {}
    mgr._req = fake_req
    job = mgr.start({"video": "clip.mp4", "brief": {"on_screen": ["A"]}, "length": 12, "seed": 3})
    st = mgr._read_state(job)
    jd = mgr.job_dir(job)
    for _ in range(100):
        if (jd / "child.pid").is_file():
            break
        time.sleep(0.05)
    child = int((jd / "child.pid").read_text())
    runner = mgr.procs[job].pid
    assert st["pid"] == runner and mgr.running() == job
    with pytest.raises(R.Busy):
        mgr.start({"video": "clip.mp4"})                          # 409 while it runs
    st["prompt_ids"] = ["p-run", "p-wait"]                         # what Comfy.submit records
    mgr.write_state(job, st)

    res = mgr.cancel(job)
    assert res["cancelled"] and res["killed"]
    assert mgr.procs.get(job) is None or mgr.procs[job].poll() is not None
    t_end = time.time() + 5
    while time.time() < t_end and not (gone(runner) and gone(child)):
        time.sleep(0.05)
    assert gone(runner) and gone(child)                           # the whole process group
    assert res["dequeued"] == ["p-wait"] and res["interrupted"] == ["p-run"]
    assert ("POST", "/queue", {"delete": ["p-wait"]}) in calls and ("POST", "/interrupt", {"prompt_id": "p-run"}) in calls
    assert not any(c[2] and "other-client" in json.dumps(c[2]) for c in calls)          # another client's prompt is never touched
    after = mgr._read_state(job)
    assert after["cancelled"] is True and after["status"] == "cancelled" and after["stage"] == "cancelled" and not after["error"]
    assert mgr.status(job)["cancelled"] is True and not mgr.status(job).get("error")
    assert mgr.running() is None                                  # the lock is free
    job2 = mgr.start({"video": "clip.mp4", "brief": {}, "length": 12})
    assert job2 != job
    assert mgr.cancel(job2)["cancelled"]
    assert mgr.cancel(job)["already"] is True                     # cancelling twice is harmless
    assert mgr.cancel("20000101-000000-ffff") is None             # unknown job
    assert mgr.cancel("not-a-job") is None
