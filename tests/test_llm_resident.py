"""llm_resident: Gemma stays loaded between the read and story side graphs; unloaded after 10 min idle or when another prompt starts (tools/app/llm_resident.py). Fake runner, no GPU, no ComfyUI."""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.app import llm_resident as LR  # noqa: E402


class FakeRunner:
    def __init__(self):
        self.llm = None
        self.closed = 0

    def load(self):
        self.llm = object()

    def close(self):
        self.llm = None
        self.closed += 1


def side_prompt(keep=True):
    return {"read_llm": {"class_type": "MographGGUFVL", "inputs": {"keep_model_loaded": keep}}, "x": {"class_type": "PreviewAny", "inputs": {}}}


def app_prompt():
    return {"design": {"class_type": "MographGGUFVL", "inputs": {"keep_model_loaded": False}}, "ltx": {"class_type": "LTXVSampler", "inputs": {}}}


def call(res, keep):
    res.begin()
    try:
        res.runner.load() if res.runner.llm is None else None
    finally:
        res.end(keep)


def test_is_resident_prompt():
    assert LR.is_resident_prompt(side_prompt(True)) and not LR.is_resident_prompt(side_prompt(False)) and not LR.is_resident_prompt(app_prompt())
    assert not LR.is_resident_prompt(None) and not LR.is_resident_prompt({"a": 1})


def test_keep_then_other_prompt_unloads_but_side_graphs_do_not():
    r = FakeRunner()
    res = LR.Resident(r, idle_s=3600)
    call(res, True)
    assert r.llm is not None and res.kept
    assert res.on_prompt_start(side_prompt(True)) is False and r.llm is not None                    # a story after a read: stays
    assert res.on_prompt_start(app_prompt()) is True and r.llm is None and res.unloads == ["another prompt started"]
    assert res.on_prompt_start(app_prompt()) is False                                              # nothing left to unload
    res._cancel()


def test_non_keep_call_closes_after_itself():
    r = FakeRunner()
    res = LR.Resident(r, idle_s=3600)
    call(res, False)
    assert r.llm is None and r.closed == 1 and not res.kept


def test_idle_timer_unloads():
    r = FakeRunner()
    t = [0.0]
    res = LR.Resident(r, idle_s=0.2, clock=lambda: t[0])
    call(res, True)
    assert r.llm is not None
    t[0] = 1.0                                                                                     # the fake clock says the idle time has passed
    done = threading.Event()
    orig = res.unload
    res.unload = lambda why: (orig(why), done.set())[0]
    assert done.wait(3.0) and r.llm is None and res.unloads[-1].startswith("idle")


def test_new_call_resets_the_idle_timer():
    r = FakeRunner()
    t = [0.0]
    res = LR.Resident(r, idle_s=0.3, clock=lambda: t[0])
    call(res, True)
    first = res.timer
    call(res, True)
    assert res.timer is not first and first.finished.is_set() or not first.is_alive()
    res._cancel()


def test_executor_hook_runs_before_the_prompt_for_async_and_sync_executors():
    for kind in ("async", "sync"):
        r = FakeRunner()
        res = LR.Resident(r, idle_s=3600)
        call(res, True)
        seen = []
        if kind == "async":
            class PE:
                async def execute_async(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
                    seen.append(r.llm)
                    return "ok"
        else:
            class PE:
                def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
                    seen.append(r.llm)
                    return "ok"
        mod = types.SimpleNamespace(PromptExecutor=PE)
        assert LR.install_executor_hook(mod, res) and LR.install_executor_hook(mod, res)        # idempotent
        pe = PE()
        run = (lambda *a: asyncio.run(pe.execute_async(*a))) if kind == "async" else (lambda *a: pe.execute(*a))
        assert run(side_prompt(True), "p1") == "ok" and seen == [r.llm] and r.llm is not None       # a read / story prompt keeps it
        assert run(app_prompt(), "p2") == "ok" and seen[-1] is None and r.llm is None               # LTX etc.: gone BEFORE the first node ran
        res._cancel()


def test_hook_never_blocks_a_prompt():
    class Bad(LR.Resident):
        def on_prompt_start(self, prompt):
            raise RuntimeError("boom")
    class PE:
        def execute(self, prompt, prompt_id):
            return "ran"
    assert LR.install_executor_hook(types.SimpleNamespace(PromptExecutor=PE), Bad(FakeRunner()))
    assert PE().execute({}, "x") == "ran"


def test_node_module_wires_the_resident_object():
    src = (ROOT / "nodes_fusion" / "llm_gguf.py").read_text(encoding="utf-8")
    assert "_RES = LR.Resident(_RUNNER)" in src and "_RES.begin()" in src and "_RES.end(bool(keep_model_loaded))" in src and "install_executor_hook" in src
    assert "gemma resident:" in src and "reused (no load)" in src                                   # the measurement log line
