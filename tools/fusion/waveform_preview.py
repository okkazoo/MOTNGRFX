"""waveform_preview.py - the live per-step waveform of the music sampler (pure torch, no ComfyUI import at module level).

Vendored from the owner's ACE15LiveWaveformPreview (ComfyUI-ACE15-Codes, composer 0.3.23), renamed for MOTNGRFX: the websocket event is
``motngrfx.waveform`` (EVENT). The node wrapper is ``nodes_fusion/music_preview.py`` (MographLiveWaveformPreview).

What it does: a MODEL patch (``ModelPatcher.set_model_sampler_post_cfg_function``) between the model source and the KSampler. At every
sampler step core's cfg_function calls ``WaveformPreviewHook`` with the step's x0 prediction (``denoised``). The hook returns that tensor
UNTOUCHED (the render is bit-identical with the node on, off or absent), VAE-decodes a copy exactly as VAEDecodeAudio does, reduces it to
``bins`` (min, max) peak pairs over the whole take and sends ``PromptServer.send_sync(EVENT, {...}, client_id)`` to the submitting client
only (the web page; progress_standin relays it with ``--relay-event motngrfx.waveform``).

Guard: a decode that took longer than the sampling since the last one skips the following step(s) until the sampling time since that
decode covers it (frames drop; the decodes cannot add more than the sampling time + one decode). A failing preview logs once per run and
never fails the render.

Event data (v 1): {v, kind "music", prompt_id, node, step, total, bins, sample_rate, seconds, peaks (base64 int8 [bins, 2]: min0, max0,
min1, max1 ... value / 127 = amplitude), decode_s, step_s, skipped, frame}.
"""
from __future__ import annotations

import base64
import logging
import time

EVENT = "motngrfx.waveform"       # websocket event type (the page listens for it; progress_standin --relay-event motngrfx.waveform)
VERSION = 1                       # data["v"]
BINS = 800                        # (min, max) pairs over the whole take: plenty for a timeline lane (the page draws <= ~1500 px)


def peak_envelope(wave, bins=BINS):
    """[C, N] (or [N]) float waveform -> int8 tensor [bins, 2] on the CPU: per bin the (min, max) over every channel and every sample in it,
    x 127, rounded half to even, clamped to +-127. Sample i belongs to bin i * bins // N (contiguous bins; with N < bins the empty bins stay 0)."""
    import torch
    w = wave.detach().float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2:
        raise ValueError(f"peak_envelope: expects [C, N] or [N], got {tuple(wave.shape)}")
    bins = int(bins)
    n = w.shape[-1]
    if n == 0:
        return torch.zeros(bins, 2, dtype=torch.int8)
    idx = torch.div(torch.arange(n, device=w.device, dtype=torch.int64) * bins, n, rounding_mode="floor")
    z = torch.zeros(bins, device=w.device, dtype=torch.float32)
    lo = z.scatter_reduce(0, idx, w.amin(0), "amin", include_self=False)
    hi = z.scatter_reduce(0, idx, w.amax(0), "amax", include_self=False)
    return torch.stack([lo, hi], 1).mul(127).round().clamp(-127, 127).to(torch.int8).cpu()


def envelope_b64(env):
    """int8 [bins, 2] -> base64 of the bytes min0, max0, min1, max1, ... (value / 127 = amplitude)."""
    return base64.b64encode(env.contiguous().numpy().tobytes()).decode("ascii")


def decode_x0(vae, model, denoised):
    """The x0 prediction decoded the way VAEDecodeAudio does it (comfy_extras/nodes_audio.py vae_decode_audio): model.process_latent_out,
    vae.decode -> [B, N, C] -> [B, C, N], divided by max(1, 5 x std). Batch item 0 only. -> ([C, N] tensor, sample rate)."""
    import torch
    lat = denoised.detach()
    if getattr(lat, "is_nested", False):
        lat = lat.unbind()[-1]
    if model is not None and hasattr(model, "process_latent_out"):
        lat = model.process_latent_out(lat)
    audio = vae.decode(lat[:1]).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sr = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    return audio[0], int(sr)


def current_ids():
    """(prompt_id, node_id) of the executing node: comfy_execution's context, else the server's last ids. (None, None) outside ComfyUI."""
    try:
        from comfy_execution.utils import get_executing_context
        c = get_executing_context()
        if c is not None:
            return c.prompt_id, c.node_id
    except Exception:                                                          # noqa: BLE001
        pass
    try:
        from server import PromptServer
        s = PromptServer.instance
        return getattr(s, "last_prompt_id", None), getattr(s, "last_node_id", None)
    except Exception:                                                          # noqa: BLE001
        return None, None


def send_preview(event, data):
    """To the client that submitted the prompt (as core's progress / executing events). False when there is none."""
    from server import PromptServer
    s = PromptServer.instance
    sid = getattr(s, "client_id", None)
    if sid is None:
        return False
    s.send_sync(event, data, sid)
    return True


def step_of(sample_sigmas, sigma):
    """-> (1-based step, total steps) from the run's sample_sigmas (CFGGuider puts them in transformer_options) and the step's sigma: the
    nearest schedule entry. (None, None) when the schedule is not there."""
    import torch
    if sample_sigmas is None or sigma is None or not torch.is_tensor(sample_sigmas) or sample_sigmas.numel() < 2:
        return None, None
    s = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
    sched = sample_sigmas.detach().flatten()[:-1].float().cpu()
    return int(torch.argmin((sched - s).abs())) + 1, int(sample_sigmas.numel()) - 1


class WaveformPreviewHook:
    """The post-CFG function MographLiveWaveformPreview installs. Returns `denoised` untouched. State resets per run (a new prompt id / sigma
    schedule, or step 1 again), so a cached patched model serves later prompts too. Guard (per run): after a decode of d seconds, the next
    decodes wait until the sampling time since it adds up to d (`debt`). clock / send / ids are injectable for tests."""

    def __init__(self, vae, bins=BINS, send=None, clock=None, ids=None):
        self.vae, self.bins = vae, int(bins)
        self.send = send or send_preview
        self.clock = clock or time.perf_counter
        self.ids = ids or current_ids
        self.runs = []                                     # one stats dict per run (tests, logging)
        self.key = object()
        self._reset(None)

    def _reset(self, key):
        self.key, self.last_step, self.t_ret, self.debt, self.warned = key, 0, None, 0.0, False
        self.stats = {"frames": 0, "skipped": 0, "decode_s": 0.0, "steps": [], "skipped_steps": []}
        self.runs.append(self.stats)

    def __call__(self, args):
        den = args["denoised"]
        try:
            self._step(args, den)
        except Exception as e:                             # noqa: BLE001 (InterruptProcessingException is a BaseException)
            if not self.warned:
                logging.warning(f"MographLiveWaveformPreview: preview failed, render unaffected: {type(e).__name__}: {e}")
                self.warned = True
            self.t_ret = self.clock()
        return den

    def _step(self, args, den):
        import torch
        if torch.is_tensor(den) and den.is_cuda:
            torch.cuda.synchronize(den.device)            # the step's GPU work is done: honest step / decode times
        t_in = self.clock()
        mo = args.get("model_options") or {}
        sched = (mo.get("transformer_options") or {}).get("sample_sigmas")
        step, total = step_of(sched, args.get("sigma"))
        pid, node = self.ids()
        key = (pid, id(sched))
        if key != self.key or (step == 1 and self.last_step >= 1):
            self._reset(key)
        if step is None:
            step, total = self.last_step + 1, 0
        if step <= self.last_step:                         # a second model call in the same step (2-eval samplers)
            return
        self.last_step = step
        step_s = None if self.t_ret is None else t_in - self.t_ret
        if step_s is not None:
            self.debt -= step_s
        if self.debt > 0:                                  # the last decode is not paid for yet: drop this frame
            self.stats["skipped"] += 1
            self.stats["skipped_steps"].append(step)
            self.t_ret = self.clock()
            return
        t0 = self.clock()
        audio, sr = decode_x0(self.vae, args.get("model"), den)
        env = peak_envelope(audio, self.bins)
        d = self.clock() - t0
        self.stats["frames"] += 1
        self.stats["steps"].append(step)
        self.stats["decode_s"] += d
        data = {"v": VERSION, "kind": "music", "prompt_id": pid, "node": node, "step": step, "total": total, "bins": self.bins,
                "sample_rate": sr, "seconds": round(audio.shape[-1] / sr, 4), "peaks": envelope_b64(env),
                "decode_s": round(d, 4), "step_s": None if step_s is None else round(step_s, 4),
                "skipped": self.stats["skipped"], "frame": self.stats["frames"]}
        self.send(EVENT, data)
        self.debt = self.clock() - t0                      # decode + envelope + send
        self.t_ret = self.clock()


def patch_model(model, vae, enabled=True, bins=BINS, hook=None):
    """-> the model with the preview hook installed (a clone; the input is untouched). enabled False -> the model itself."""
    if not enabled:
        return model
    m = model.clone()
    m.set_model_sampler_post_cfg_function(hook or WaveformPreviewHook(vae, bins))
    return m
