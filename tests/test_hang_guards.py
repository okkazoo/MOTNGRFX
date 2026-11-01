"""Hang guards of the render client chain (2026-10-29 incident: run_batch sat ~80 min after its prompt had finished). No network, no GPU, no box: fakes only.

  run_timing.follow / run_graph   a progress_standin that goes silent (or never closes its pipe) -> /history decides, the run is finalized within the bound
  run_batch.bounded / finalize    a hung ingredients pull / model-replies save / matte preview cannot hold the batch
  run_batch.run_ceiling           per-run ceiling = max(30 min, 3 x mean wall)
  ingredients.fetch / remote_list a stalled tar | ssh stream is killed, what arrived is kept
Run: python -m pytest tests/test_hang_guards.py -q"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "tools" / "app")):
    if _p not in sys.path:
        sys.path.append(_p)

import run_batch as RB  # noqa: E402
import run_timing as RT  # noqa: E402

PID = "96c93289-a1c6-406b-a333-50a0a22a5a93"


class FakeProc:
    """a progress_standin: yields the given lines, then blocks (never closes stdout) until kill() - the hung client."""

    def __init__(self, lines, hang=True):
        self._lines, self._hang = list(lines), hang
        self._released = threading.Event()
        self.killed = False
        self.stdout = self._gen()

    def _gen(self):
        for ln in self._lines:
            yield ln
        if self._hang:
            self._released.wait(60)          # blocked read on a pipe nobody writes to; kill() releases it like a closed pipe does

    def kill(self):
        self.killed = True
        self._released.set()

    def wait(self, timeout=None):
        return -9 if self.killed else 0


def relay(ty, **d):
    return "STANDIN_RELAY " + json.dumps({"type": ty, "data": d}, separators=(",", ":")) + "\n"


# ------------------------------------------------------------------------------------------------ run_timing.follow
def test_follow_ends_from_history_when_the_child_goes_silent():
    p = FakeProc([f"submitted {PID}  (client abcd1234)\n", "[  10s] SAMPLE 3/8\n"])
    seen, dumps, asked = [], [], []

    def hist(port, pid):
        asked.append(pid)
        return "success"
    t0 = time.time()
    r = RT.follow(p, seen.append, deadline=time.time() + 60, port=1, idle_s=0.3, check_every=0.1, history_fn=hist, dump=dumps.append)
    assert time.time() - t0 < 5
    assert r == {"ended": "history", "history": "success", "prompt_id": PID}
    assert len(seen) == 2 and len(dumps) == 1 and "no output" in dumps[0] and asked == [PID]
    p.kill()


def test_follow_keeps_waiting_while_history_says_running_then_hits_the_deadline():
    p = FakeProc([relay("execution_start", prompt_id=PID)])
    states = iter(["running"] * 100)
    t0 = time.time()
    r = RT.follow(p, lambda l: None, deadline=time.time() + 1.0, port=1, idle_s=0.2, check_every=0.1, history_fn=lambda port, pid: next(states))
    assert 0.9 < time.time() - t0 < 5
    assert r["ended"] == "deadline" and r["history"] == "running" and r["prompt_id"] == PID
    p.kill()


def test_follow_normal_eof_is_untouched():
    p = FakeProc([f"submitted {PID}\n", "RESULT OK 5s peakVRAM 1 MiB\n"], hang=False)
    r = RT.follow(p, lambda l: None, deadline=time.time() + 60, port=1, idle_s=5, history_fn=lambda *a: pytest.fail("must not ask /history on a normal run"))
    assert r["ended"] == "eof"


def test_a_chatty_child_is_not_a_stall():
    lines = [f"submitted {PID}\n"] + [f"[{i}s] SAMPLE\n" for i in range(5)]

    class Slow(FakeProc):
        def _gen(self):
            for ln in self._lines:
                time.sleep(0.15)
                yield ln
    p = Slow(lines, hang=False)
    r = RT.follow(p, lambda l: None, deadline=time.time() + 60, port=1, idle_s=0.4, check_every=0.05, history_fn=lambda *a: pytest.fail("not silent"))
    assert r["ended"] == "eof"


# ------------------------------------------------------------------------------------------------ run_timing.run_graph
GRAPH = {"1": {"class_type": "Thing", "inputs": {}, "_meta": {"title": "OUT final"}}}


def _run_graph(tmp_path, proc, history, timeout=60, idle_s=0.3):
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps(GRAPH), encoding="utf-8")
    return RT.run_graph(str(gp), "1", 9, {}, str(tmp_path / "run_01"), timeout=timeout, sample_vram=False, graph_obj=GRAPH, idle_s=idle_s,
                        history_fn=lambda port, pid: history, popen=lambda *a, **k: proc)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(RT.bench, "history_outputs", lambda base, pid, graph, **k: {"OUT final": ["v/x_00001_.mp4"]})


def test_run_graph_finalizes_from_history_when_the_child_hangs_after_the_prompt_finished(tmp_path):
    """the incident: prompt succeeded on the box, progress_standin never printed DONE / exited."""
    p = FakeProc([f"submitted {PID}  (client ab)\n", relay("execution_start", prompt_id=PID), relay("executing", node="1", prompt_id=PID),
                  "[   5s] PREP 1/1 11%\n"])
    t0 = time.time()
    s = _run_graph(tmp_path, p, "success")
    assert time.time() - t0 < 10
    assert p.killed and s["status"] == "done" and s["ended_by"] == "history" and s["prompt_id"] == PID
    assert s["outputs"] == {"OUT final": ["v/x_00001_.mp4"]}
    log = (tmp_path / "run_01.relay.log").read_text(encoding="utf-8")
    assert "WATCHDOG run ended by history" in log


def test_run_graph_records_an_error_when_history_says_error(tmp_path):
    p = FakeProc([f"submitted {PID}\n", relay("execution_start", prompt_id=PID)])
    s = _run_graph(tmp_path, p, "error")
    assert p.killed and s["status"] == "error" and s["ended_by"] == "history"


def test_run_graph_hard_ceiling_ends_a_run_nobody_can_decide(tmp_path):
    p = FakeProc([f"submitted {PID}\n", relay("execution_start", prompt_id=PID)])
    t0 = time.time()
    s = _run_graph(tmp_path, p, "running", timeout=1.0, idle_s=30)
    assert time.time() - t0 < 10
    assert p.killed and s["ended_by"] == "deadline" and s["status"] == "error" and s["error"]["type"] == "WatchdogTimeout"


def test_run_graph_normal_run_is_unchanged(tmp_path):
    p = FakeProc([f"submitted {PID}\n", relay("execution_start", prompt_id=PID), relay("executing", node="1", prompt_id=PID),
                  relay("execution_success", prompt_id=PID), "RESULT OK  9s  peakVRAM 100 MiB\n"], hang=False)
    s = _run_graph(tmp_path, p, "running")
    assert not p.killed and s["status"] == "done" and "ended_by" not in s


# ------------------------------------------------------------------------------------------------ run_batch
def test_bounded_returns_the_default_when_the_step_hangs_and_logs_it(tmp_path):
    logs = []
    ev = threading.Event()
    t0 = time.time()
    r = RB.bounded(lambda: ev.wait(30), 0.3, "ingredients pull", logs.append, default="dflt", out=str(tmp_path))
    assert time.time() - t0 < 5 and r == "dflt"
    assert any("WATCHDOG ingredients pull" in l for l in logs)
    assert "ingredients pull abandoned" in (tmp_path / "hang_stacks.log").read_text(encoding="utf-8")
    assert "Thread" in (tmp_path / "hang_stacks.log").read_text(encoding="utf-8")        # faulthandler's all-thread dump
    ev.set()
    RB.HANG_LOG[0].close()


def test_bounded_passes_results_and_swallows_errors():
    logs = []
    assert RB.bounded(lambda: 42, 5, "x", logs.append) == 42
    assert RB.bounded(lambda: 1 / 0, 5, "x", logs.append, default=7) == 7 and any("ZeroDivisionError" in l for l in logs)


def test_run_ceiling_is_max_30_min_or_3x_the_mean_wall():
    st = {"runs": [{"status": "done", "wall_s": 400.0}, {"status": "done", "wall_s": 800.0}, {"status": "error", "wall_s": 9999}]}
    assert RB.run_ceiling(st, {}) == 1800.0                      # 3 x 600 = 1800
    st["runs"].append({"status": "done", "wall_s": 1400.0})
    assert RB.run_ceiling(st, {}) == 3 * (400 + 800 + 1400) / 3
    assert RB.run_ceiling({"runs": []}, {}) == 3 * RB.DEFAULT_EXPECTED_WALL_S      # nothing measured yet
    assert RB.run_ceiling({"runs": [{"status": "done", "wall_s": 30.0}]}, {}) == RB.MIN_CEILING_S
    assert RB.run_ceiling(st, {"timeout": 77}) == 77.0           # an explicit --timeout wins


def test_finalize_is_not_held_by_a_hung_pull_or_replies_save(tmp_path, monkeypatch):
    ev = threading.Event()
    monkeypatch.setattr(RB, "STEP_S", {"matte_preview": 0.3, "model_replies": 0.3, "ingredients": 0.3})
    monkeypatch.setattr(RB, "version_dir", lambda target, runner=None: str(tmp_path / "v001"))
    monkeypatch.setattr(RB, "pull_ingredients", lambda st, vdir, log=print: ev.wait(30))
    monkeypatch.setattr(RB, "reconcile", lambda *a, **k: 0)
    (tmp_path / "v001" / "out").mkdir(parents=True)
    st = {"n": 1, "iid": "1", "port": 9, "target": "P/i/v001", "graph": "g", "length": 16, "started": "now", "version": None,
          "runs": [{"i": 1, "seed": 1, "status": "done", "video": "ltx", "part": "r01"}]}
    out = tmp_path / "bench"
    out.mkdir()
    logs = []
    t0 = time.time()
    RB.finalize(st, str(out), log=logs.append)
    assert time.time() - t0 < 10
    assert sum("WATCHDOG" in l for l in logs) == 1                # ingredients pull abandoned (the model-reply archive step is not in the export)
    assert (out / "BATCH.md").exists()                            # the reports were still written
    ev.set()
    RB.HANG_LOG[0].close()


def test_batch_passes_the_ceiling_and_goes_on_after_a_watchdog_ended_run(tmp_path):
    """run 1 is ended by the watchdog (error from /history), run 2 still starts; the runner gets timeout = ceiling."""
    calls = []

    def runner(graph, iid, port, sets, out, **kw):
        calls.append(kw["timeout"])
        if len(calls) == 1:
            return {"status": "error", "error": {"node": None, "type": "WatchdogTimeout", "message": "x"}, "prompt_id": "pid1", "ended_by": "deadline",
                    "history_state": "running", "wall_s": 1800.0, "stages": [], "nodes": [], "outputs": {}}
        return {"status": "done", "error": None, "prompt_id": "pid2", "wall_s": 100.0, "total_seconds": 90.0, "stages": [], "nodes": [], "outputs": {}}
    g = tmp_path / "g.json"
    g.write_text("{}", encoding="utf-8")
    a = dict(iid="1", port=1, n=2, length=16, graph=str(g), out=str(tmp_path / "out"), idea="app", project=False, resume=False, seed_base=0)
    logs = []
    st = RB.run_batch(a, run_graph=runner, texts_of=lambda *x: {}, log=logs.append)
    assert [r["status"] for r in st["runs"]] == ["error", "done"]
    assert calls[0] == 3 * RB.DEFAULT_EXPECTED_WALL_S and calls[1] == calls[0]      # an errored run has no usable wall: run 2 gets the default again
    assert any("did not finish by itself" in l for l in logs)
    RB.HANG_LOG[0] and RB.HANG_LOG[0].close()


# ------------------------------------------------------------------------------------------------ ingredients
class FakePipe:
    def __init__(self, ev, data=b""):
        self.ev, self.data = ev, data

    def read(self, n=-1):
        if self.data:
            d, self.data = self.data[:n if n > 0 else None], self.data[n:] if n > 0 else b""
            return d
        self.ev.wait(60)                 # a stalled ssh: no byte, no EOF until killed
        return b""

    def write(self, b):
        return len(b)

    def close(self):
        pass


class FakeSsh:
    def __init__(self, data=b""):
        self.ev = threading.Event()
        self.stdout, self.stderr, self.stdin = FakePipe(self.ev, data), FakePipe(self.ev), FakePipe(self.ev)
        self.killed = False
        self.returncode = None

    def kill(self):
        self.killed = True
        self.ev.set()

    def wait(self, timeout=None):
        self.returncode = -9 if self.killed else 0
        return self.returncode


def test_fetch_kills_a_stalled_ssh_and_keeps_what_arrived(tmp_path):
    import io
    import tarfile
    import ingredients as ING
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        b = b"hello"
        ti = tarfile.TarInfo("a.txt")
        ti.size = len(b)
        tf.addfile(ti, io.BytesIO(b))
    first = buf.getvalue()[:1024]                 # the first member complete, then the stream just stops (no end-of-archive marker, no EOF)
    p = FakeSsh(first)
    logs = []
    t0 = time.time()
    n = ING.fetch({"host": "h", "port": 1, "user": "u"}, "/r", ["a.txt", "b.txt"], str(tmp_path), logs.append, popen=lambda *a, **k: p, stall_s=0.4, max_s=60)
    assert time.time() - t0 < 10
    assert n == 1 and (tmp_path / "a.txt").read_bytes() == b"hello"
    assert p.killed and any("killed" in l for l in logs)


def test_fetch_total_ceiling(tmp_path):
    import ingredients as ING
    p = FakeSsh(b"")
    # bytes keep "arriving" would reset the stall clock: here the ceiling is what ends it
    t0 = time.time()
    ING.fetch({"host": "h", "port": 1, "user": "u"}, "/r", ["a"], str(tmp_path), lambda *a: None, popen=lambda *a, **k: p, stall_s=60, max_s=0.5)
    assert time.time() - t0 < 10 and p.killed


def test_remote_list_survives_a_ssh_timeout():
    import ingredients as ING

    def runner(argv, **kw):
        assert kw["timeout"] == 180 and kw["stdin"] is subprocess.DEVNULL        # no console stdin for ssh
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    assert ING.remote_list({"host": "h", "port": 1, "user": "u"}, "/r", runner=runner) is None
    assert "ServerAliveInterval=15" in " ".join(ING.ssh_argv({"host": "h", "port": 1, "user": "u"}, "true"))
