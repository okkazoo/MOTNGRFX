"""Live per-step waveform of the music sampler (tools/fusion/waveform_preview.py + nodes_fusion/music_preview.py), CPU, no ComfyUI.
The node wrapper is tested against a tiny stand-in of comfy_api.latest.io when ComfyUI is not importable.
Run: python -m pytest tests/test_waveform_preview.py -q"""
from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import music_graph  # noqa: E402
from tools.fusion import waveform_preview as WP  # noqa: E402


class FakeVae:
    audio_sample_rate = 48000

    def decode(self, lat):                       # [1, C, T] -> [1, N, 2]: a sine whose amplitude follows the latent mean
        n = 4800
        t = torch.arange(n, dtype=torch.float32) / 48000
        amp = float(lat.float().abs().mean())
        x = amp * torch.sin(2 * torch.pi * 220 * t)
        return torch.stack([x, x], -1).unsqueeze(0)


class FakeModel:
    def __init__(self):
        self.hooks = []

    def clone(self):
        c = FakeModel()
        c.hooks = list(self.hooks)
        return c

    def set_model_sampler_post_cfg_function(self, f):
        self.hooks.append(f)


def _hook():
    sent = []
    t = [0.0]

    def clock():
        t[0] += 0.001
        return t[0]
    h = WP.WaveformPreviewHook(FakeVae(), 64, send=lambda ev, d: sent.append((ev, d)), clock=clock, ids=lambda: ("p1", "n7"))
    return h, sent


def _args(i, sigmas, den):
    return {"denoised": den, "sigma": sigmas[i:i + 1], "model": None, "model_options": {"transformer_options": {"sample_sigmas": sigmas}}}


def test_event_name_is_unique_to_motngrfx():
    assert WP.EVENT == "motngrfx.waveform"


def test_peak_envelope_minmax_and_bytes():
    w = torch.tensor([[0.0, 1.0, -1.0, 0.5], [0.0, 0.0, 0.0, -0.5]])
    e = WP.peak_envelope(w, 2)
    assert e.dtype == torch.int8 and tuple(e.shape) == (2, 2)
    assert e.tolist() == [[0, 127], [-127, 64]]                  # bin0 = samples 0..1, bin1 = samples 2..3
    assert base64.b64decode(WP.envelope_b64(e)) == bytes([0, 127, 129, 64])


def test_hook_returns_denoised_untouched_and_sends_events():
    sig = torch.linspace(1.0, 0.0, 9)                              # 8 steps
    h, sent = _hook()
    for i in range(8):
        den = torch.full((1, 64, 30), 0.1 * (i + 1))
        assert h(_args(i, sig, den)) is den                        # the render is bit-identical
    assert sent and len(sent) + h.stats["skipped"] == 8
    ev, d = sent[0]
    assert ev == WP.EVENT and d["kind"] == "music" and d["prompt_id"] == "p1" and d["node"] == "n7"
    assert d["step"] == 1 and d["total"] == 8 and d["bins"] == 64 and d["sample_rate"] == 48000
    assert len(base64.b64decode(d["peaks"])) == 64 * 2


def test_hook_never_fails_the_render():
    h = WP.WaveformPreviewHook(object(), 64, send=lambda *a: None, ids=lambda: ("p", "n"))   # a vae without decode
    den = torch.zeros(1, 64, 10)
    assert h({"denoised": den, "sigma": torch.tensor([1.0]), "model_options": {}}) is den


def test_patch_model_disabled_is_a_passthrough_and_enabled_clones():
    m = FakeModel()
    assert WP.patch_model(m, FakeVae(), enabled=False) is m
    p = WP.patch_model(m, FakeVae(), enabled=True)
    assert p is not m and m.hooks == [] and len(p.hooks) == 1 and isinstance(p.hooks[0], WP.WaveformPreviewHook)


def test_t2m_chain_inserts_the_node_between_shift_and_sampler_and_can_leave_it_out():
    g = music_graph.t2m_chain(["request", 0], 1, None)
    assert g["t2m_live"]["class_type"] == "MographLiveWaveformPreview"
    assert g["t2m_live"]["inputs"]["model"] == ["t2m_shift", 0] and g["t2m_live"]["inputs"]["vae"] == ["t2m_vae", 0]
    assert g["t2m_live"]["inputs"]["enabled"] is True
    assert g["t2m_ks"]["inputs"]["model"] == ["t2m_live", 0]
    off = music_graph.t2m_chain(["request", 0], 1, None, live_preview=False)
    assert "t2m_live" not in off and off["t2m_ks"]["inputs"]["model"] == ["t2m_shift", 0]


def test_the_shipped_app_graphs_carry_the_node():
    gd = os.path.join(ROOT, "app", "graphs")
    names = [f for f in os.listdir(gd) if f.startswith("motngrfx_") and f.endswith(".json") and not f.endswith(".app.json")]
    assert names
    for f in names:
        g = json.load(open(os.path.join(gd, f), encoding="utf-8"))
        assert g["t2m_ks"]["inputs"]["model"] == ["t2m_live", 0], f
        assert g["t2m_live"]["class_type"] == "MographLiveWaveformPreview", f


def _stub_comfy_api():
    """a stand-in comfy_api.latest.io good enough to import and call the node wrapper."""
    class _In:
        def __init__(self, name, **kw):
            self.name, self.kw = name, kw

    class K:
        @staticmethod
        def Input(name, **kw):
            return _In(name, **kw)

        @staticmethod
        def Output(**kw):
            return _In("out", **kw)

    io = types.SimpleNamespace(ComfyNode=type("ComfyNode", (), {}), Schema=lambda **kw: types.SimpleNamespace(**kw), Model=K, Vae=K,
                               Boolean=K, Int=K, NodeOutput=lambda *a, ui=None: types.SimpleNamespace(args=a, ui=ui))
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    pkg = types.ModuleType("comfy_api")
    pkg.latest = latest
    return {"comfy_api": pkg, "comfy_api.latest": latest}


def test_node_is_registered_and_returns_the_model_patched(monkeypatch):
    try:
        import comfy_api.latest  # noqa: F401
        stub = {}
    except Exception:
        stub = _stub_comfy_api()
    for k, v in stub.items():
        monkeypatch.setitem(sys.modules, k, v)
    for k in [k for k in sys.modules if k == "nodes_fusion" or k.startswith("nodes_fusion.")]:
        monkeypatch.delitem(sys.modules, k)
    if stub:                                                       # nodes_fusion/__init__ imports every sub-module: skip it, load only ours
        t = types.ModuleType("nodes_fusion._types")
        t.CATEGORY = "Mograph"
        pkg = types.ModuleType("nodes_fusion")
        pkg.__path__ = [os.path.join(ROOT, "nodes_fusion")]
        monkeypatch.setitem(sys.modules, "nodes_fusion", pkg)
        monkeypatch.setitem(sys.modules, "nodes_fusion._types", t)
    mp = importlib.import_module("nodes_fusion.music_preview")
    assert [c.__name__ for c in mp.NODES] == ["MographLiveWaveformPreview"]
    m = FakeModel()
    out = mp.MographLiveWaveformPreview.execute(m, FakeVae(), True, 100)
    res = out.args[0] if stub else out[0]
    assert res is not m and len(res.hooks) == 1
    out_off = mp.MographLiveWaveformPreview.execute(m, FakeVae(), False, 100)
    assert (out_off.args[0] if stub else out_off[0]) is m
    src = open(os.path.join(ROOT, "nodes_fusion", "__init__.py"), encoding="utf-8").read()
    assert '"music_preview"' in src
