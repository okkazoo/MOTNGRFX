"""Moving sampler preview (tools/fusion/strip_preview.py + nodes_fusion/live_preview.py), CPU, no ComfyUI: fake latent_preview / server modules.
Run: python -m pytest tests/test_strip_preview.py -q"""
from __future__ import annotations

import base64
import io
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import strip_preview as SP  # noqa: E402


class FakeBase:
    """what comfy's Latent2RGBPreviewer looks like to us: transposed factors [3, C], bias, reshape, and the still-image method."""

    def __init__(self, c=8):
        g = torch.Generator().manual_seed(1)
        self.latent_rgb_factors = torch.randn(3, c, generator=g) * 0.2
        self.latent_rgb_factors_bias = None
        self.latent_rgb_factors_reshape = None
        self.calls = []

    def decode_latent_to_preview_image(self, fmt, x0):
        self.calls.append(tuple(x0.shape))
        return ("JPEG", "still", 512)


def _prev(**kw):
    sent = []
    t = [0.0]

    def clock():
        t[0] += 0.001
        return t[0]
    p = SP.StripPreviewer(FakeBase(), rate=6, send=lambda ev, d: sent.append((ev, d)), clock=clock, ids=lambda: ("pid1", "l1.39"), **kw)
    return p, sent


def test_event_and_flag_names_are_motngrfx():
    assert SP.EVENT == "motngrfx.strip" and SP.FLAG == "motngrfx_strip_preview"


def test_playback_rate_is_twice_real_time_capped_at_8():
    ltx = types.SimpleNamespace(temporal_downscale_ratio=8)
    assert SP.playback_rate(ltx) == 6 and SP.playback_rate(types.SimpleNamespace()) == 8 and SP.playback_rate(types.SimpleNamespace(temporal_downscale_ratio=4)) == 8
    assert SP.playback_rate(types.SimpleNamespace(temporal_downscale_ratio=100)) == 2


def test_a_5d_latent_becomes_one_event_with_every_frame_as_a_valid_small_jpeg():
    p, sent = _prev()
    x0 = torch.randn(1, 8, 13, 32, 18)                       # a 576x1024 LTX clip: 13 latent frames of 32 x 18
    assert p.decode_latent_to_preview_image("JPEG", x0) is None   # no binary still for these steps
    assert len(sent) == 1
    ev, d = sent[0]
    assert ev == SP.EVENT and d["kind"] == "strip" and d["prompt_id"] == "pid1" and d["node"] == "l1.39" and d["step"] == 1 and d["rate"] == 6 and d["fmt"] == "jpeg"
    assert len(d["frames"]) == 13 and (d["w"], d["h"]) == (18, 32)
    for f in d["frames"]:
        im = Image.open(io.BytesIO(base64.b64decode(f)))
        assert im.format == "JPEG" and im.size == (18, 32)
    assert p.base.calls == []


def test_frames_move_with_the_latent_not_a_single_repeated_frame():
    p, sent = _prev()
    x0 = torch.zeros(1, 8, 4, 16, 16)
    x0[:, :, 0] = -3.0
    x0[:, :, 3] = 3.0
    p.decode_latent_to_preview_image("JPEG", x0)
    fr = [Image.open(io.BytesIO(base64.b64decode(f))).convert("L").getpixel((4, 4)) for f in sent[0][1]["frames"]]
    assert len(set(fr)) >= 2


def test_long_clips_are_subsampled_and_big_frames_scaled_down():
    u8 = SP.decode_frames(torch.randn(1, 8, 40, 256, 128), torch.randn(3, 8) * 0.2)
    assert u8.shape[0] == SP.MAX_FRAMES and max(u8.shape[1:3]) == SP.MAX_SIDE and u8.dtype == torch.uint8


def test_steps_count_and_each_step_sends_a_strip():
    p, sent = _prev()
    for _ in range(5):
        p.decode_latent_to_preview_image("JPEG", torch.randn(1, 8, 5, 16, 16))
    assert len(sent) + p.stats["skipped"] == 5 and sent[0][1]["step"] == 1


def test_guard_skips_steps_while_decoding_costs_more_than_sampling():
    sent = []
    t = [0.0]
    ticks = {"decode": False}

    def clock():
        t[0] += 0.5 if ticks["decode"] else 0.0001                 # every clock read costs 0.5 s: decode looks slow, sampling (between calls) fast
        return t[0]
    p = SP.StripPreviewer(FakeBase(), rate=6, send=lambda ev, d: sent.append(d), clock=clock, ids=lambda: ("p", "n"))
    ticks["decode"] = True
    p.decode_latent_to_preview_image("JPEG", torch.randn(1, 8, 3, 8, 8))
    ticks["decode"] = False
    p.decode_latent_to_preview_image("JPEG", torch.randn(1, 8, 3, 8, 8))
    assert len(sent) == 1 and p.stats["skipped"] == 1


def test_a_4d_still_latent_keeps_the_original_previewer():
    p, sent = _prev()
    x0 = torch.randn(1, 8, 16, 16)
    r = p.decode_latent_to_preview_image("JPEG", x0)
    assert r == ("JPEG", "still", 512) and sent == [] and p.base.calls == [(1, 8, 16, 16)]


def test_a_failing_decode_never_raises():
    p, sent = _prev()
    p.factors = torch.randn(3, 99)                            # wrong channel count
    assert p.decode_latent_to_preview_image("JPEG", torch.randn(1, 8, 3, 8, 8)) is None and sent == []


def test_wrap_needs_rgb_factors_else_the_original_stays():
    assert SP.StripPreviewer.wrap(None) is None
    assert SP.StripPreviewer.wrap(types.SimpleNamespace(latent_rgb_factors=None)) is None
    assert isinstance(SP.StripPreviewer.wrap(FakeBase(), types.SimpleNamespace(temporal_downscale_ratio=8), send=lambda *a: None, ids=lambda: (None, None)), SP.StripPreviewer)


def _fake_server(flag):
    item = (1, "pid", {}, {SP.FLAG: True} if flag else {}, [])
    q = types.SimpleNamespace(currently_running={0: item})
    return types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(prompt_queue=q)))


def test_install_wraps_get_previewer_once_and_only_for_opted_in_prompts():
    base = FakeBase()
    lp = types.SimpleNamespace(get_previewer=lambda device, fmt, *a, **k: base)
    srv = _fake_server(False)
    assert SP.install(lp, srv) is True and SP.install(lp, srv) is False                 # idempotent
    assert lp.get_previewer("cpu", None) is base                                         # no flag: ComfyUI's own previewer
    lp2 = types.SimpleNamespace(get_previewer=lambda device, fmt, *a, **k: base)
    SP.install(lp2, _fake_server(True))
    p = lp2.get_previewer("cpu", types.SimpleNamespace(temporal_downscale_ratio=8))
    assert isinstance(p, SP.StripPreviewer) and p.base is base and p.rate == 6
    lp3 = types.SimpleNamespace(get_previewer=lambda *a, **k: None)                      # previews off (none): stays off
    SP.install(lp3, _fake_server(True))
    assert lp3.get_previewer("cpu", None) is None
    lp4 = types.SimpleNamespace(get_previewer=lambda *a, **k: types.SimpleNamespace(latent_rgb_factors=None))   # nothing to decode with: the original
    SP.install(lp4, _fake_server(True))
    assert lp4.get_previewer("cpu", None).latent_rgb_factors is None


def test_install_is_a_quiet_no_op_without_comfyui():
    assert SP.install(types.SimpleNamespace(), None) is False
    assert SP.flag_on(types.SimpleNamespace()) is False


def test_the_pack_registers_the_hook_module():
    src = open(os.path.join(ROOT, "nodes_fusion", "__init__.py"), encoding="utf-8").read()
    assert '"live_preview"' in src
    assert os.path.isfile(os.path.join(ROOT, "nodes_fusion", "live_preview.py"))
