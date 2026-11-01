"""llm_resident: keep Gemma (the llama.cpp runner of nodes_fusion/llm_gguf.py) loaded between the `_media_read` and `_story` side graphs, so the page's read -> story -> story ...
before "Make it" pays the load once (16_app/MEDIA_BOX.md "STORYBOARD v2": GEMMA RESIDENT). Pure Python (no ComfyUI / torch import at module level); the node and the ComfyUI hook live in
nodes_fusion/llm_gguf.py.

    Resident(runner)                     one per node module; wraps the Runner
      .begin() / .end(keep, idle_s)      around a node call: the lock is held, after the call the idle timer is (re)armed when `keep`, else the model is closed
      .unload(reason)                    close + free now (idempotent; logged)
      .on_prompt_start(prompt)           ComfyUI is about to execute `prompt` (API graph dict): unload unless it is a resident (read / story) prompt
    is_resident_prompt(prompt)           a prompt holding a MographGGUFVL node with keep_model_loaded true (only the two side graphs set it; tools/app/llm_arms.RESIDENT_NODE_IDS)
    install_executor_hook(execution_module, resident)   wraps PromptExecutor.execute_async / execute once: the unload runs before the first node of any other prompt (LTX, Alpha-Gen ...)

Unload rules: (a) idle IDLE_S (600 s = 10 min) after the last resident call (a daemon timer thread); (b) any prompt that is not a read / story prompt starts executing;
(c) a MographGGUFVL call with keep_model_loaded false closes it after its own call as before (the app / batch graphs).
"""
from __future__ import annotations

import functools
import inspect
import threading
import time

IDLE_S = 600.0
NODE = "MographGGUFVL"


def _log(msg: str) -> None:
    print(f"[MographGGUFVL] {msg}")


def is_resident_prompt(prompt) -> bool:
    """True when the API-format graph has a MographGGUFVL node with keep_model_loaded true (the read / story side graphs)."""
    if not isinstance(prompt, dict):
        return False
    for n in prompt.values():
        if isinstance(n, dict) and n.get("class_type") == NODE and (n.get("inputs") or {}).get("keep_model_loaded") is True:
            return True
    return False


class Resident:
    def __init__(self, runner, idle_s: float = IDLE_S, clock=time.monotonic):
        self.runner = runner
        self.idle_s = float(idle_s)
        self.clock = clock
        self.lock = threading.RLock()
        self.last = 0.0
        self.timer: threading.Timer | None = None
        self.kept = False                       # True while the runner holds a model because of keep_model_loaded
        self.unloads: list[str] = []            # reasons, newest last (tests / the log)

    # --- around a node call
    def begin(self) -> None:
        self.lock.acquire()
        self._cancel()

    def end(self, keep: bool) -> None:
        try:
            if keep and getattr(self.runner, "llm", True) is not None:
                self.kept = True
                self.last = self.clock()
                self._arm()
            else:
                self.kept = False
                self.runner.close()
        finally:
            self.lock.release()

    # --- unloading
    def _cancel(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def _arm(self) -> None:
        self._cancel()
        t = threading.Timer(self.idle_s, self._idle)
        t.daemon = True
        self.timer = t
        t.start()

    def _idle(self) -> None:
        with self.lock:
            if self.kept and self.clock() - self.last >= self.idle_s - 0.5:
                self.unload(f"idle {self.idle_s:g} s")

    def unload(self, reason: str) -> bool:
        """close + free the resident model now. True when there was one."""
        with self.lock:
            self._cancel()
            had = self.kept or getattr(self.runner, "llm", None) is not None
            self.kept = False
            if had:
                self.runner.close()
                self.unloads.append(reason)
                _log(f"gemma resident: unloaded ({reason})")
            return had

    def on_prompt_start(self, prompt) -> bool:
        """ComfyUI starts executing `prompt`: any prompt that is not a read / story prompt frees Gemma first (LTX / Alpha-Gen would allocate next to it)."""
        if is_resident_prompt(prompt):
            return False
        return self.unload("another prompt started") if self.kept else False


def install_executor_hook(execution_module, resident: Resident) -> bool:
    """wrap execution.PromptExecutor.execute_async (coroutine, current ComfyUI) or .execute (older) so `resident.on_prompt_start(prompt)` runs before the prompt's first node. Once per process.
    Returns True when a hook was installed (or was already)."""
    cls = getattr(execution_module, "PromptExecutor", None)
    if cls is None:
        return False
    name = "execute_async" if hasattr(cls, "execute_async") else "execute" if hasattr(cls, "execute") else None
    if name is None:
        return False
    orig = getattr(cls, name)
    if getattr(orig, "_mograph_resident", False):
        return True

    def _before(args, kwargs):
        prompt = args[0] if args else kwargs.get("prompt")
        try:
            resident.on_prompt_start(prompt)
        except Exception as e:      # noqa: BLE001  the hook never stops a prompt
            _log(f"gemma resident hook error: {type(e).__name__}: {e}")

    if inspect.iscoroutinefunction(orig):
        @functools.wraps(orig)
        async def wrapped(self, *a, **kw):
            _before(a, kw)
            return await orig(self, *a, **kw)
    else:
        @functools.wraps(orig)
        def wrapped(self, *a, **kw):
            _before(a, kw)
            return orig(self, *a, **kw)
    wrapped._mograph_resident = True
    setattr(cls, name, wrapped)
    return True
