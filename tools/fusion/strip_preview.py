"""strip_preview.py - a MOVING sampler preview: every sampler step sends the whole latent clip as a short frame strip (pure torch + PIL).

Why: ComfyUI's own live preview (comfy/latent_preview.py, v0.43.2 `Latent2RGBPreviewer`) decodes ONE latent frame (`x0[0, :, 0]`) per step, so a shot slot
could only show one still refining. This module wraps `latent_preview.get_previewer` (the way ComfyUI-VideoHelperSuite does for its animated previews; KJNodes is not
on the box) and, for a 5D video latent [B, C, T, H, W], decodes ALL T latent frames with the same latent -> rgb factors, scales them down, JPEG-encodes them and sends
ONE websocket JSON event per step to the client that submitted the prompt:

    motngrfx.strip  {v, kind "strip", prompt_id, node, step, fmt "jpeg", w, h, rate, frames: [base64 jpeg x T], decode_s, skipped}

The page (web_app/buildview.js) loops the strip in the shot slot at `rate` fps (<= 8) and replaces it by the refined one at the next step.

Opt-in per prompt: `extra_data.motngrfx_strip_preview = true` (the page and own_video_job.py set it); without the flag, and for everything that is not a 5D video latent
(stills, audio), the original previewer runs unchanged. A latent format with no rgb factors (nothing to decode with) also keeps the original previewer = the single still.
The render is never touched: the hook only reads x0, returns None (no binary still is sent for those steps) and a failing decode logs once and never raises.
Guard: after a decode of d seconds the next steps are skipped until the sampling time since it adds up to d (decoding can never cost more than sampling).
"""
from __future__ import annotations

import base64
import io
import logging
import time

EVENT = "motngrfx.strip"           # websocket event type (the page listens for it)
FLAG = "motngrfx_strip_preview"    # extra_data key that turns it on for a prompt
VERSION = 1
MAX_FRAMES = 16                    # latent frames per strip (a 121 frame LTX clip has 16)
MAX_SIDE = 160                     # longest side of a strip frame in px (the latent grid is ~18x32 for a 576x1024 clip: scaled up smooth on the page)
MAX_RATE = 8                       # page playback fps cap


def playback_rate(latent_format) -> int:
    """fps to play the strip at: twice real time (a latent frame = `temporal_downscale_ratio` video frames at 24 fps), capped at MAX_RATE; 8 when the format does not say."""
    r = getattr(latent_format, "temporal_downscale_ratio", None)
    try:
        return max(2, min(MAX_RATE, int(round(2 * 24 / float(r))))) if r else MAX_RATE
    except Exception:                                                        # noqa: BLE001
        return MAX_RATE


def decode_frames(x0, factors, bias=None, reshape=None, max_frames=MAX_FRAMES, max_side=MAX_SIDE):
    """5D latent [B, C, T, H, W] (batch item 0) -> uint8 tensor [N, h, w, 3] on the CPU, N <= max_frames latent frames (evenly spaced when T is larger), the longest side
    scaled to <= max_side. `factors` = the previewer's latent_rgb_factors ([3, C], already transposed), `reshape` its latent_rgb_factors_reshape (applied to [T, C, H, W])."""
    import torch
    import torch.nn.functional as F
    x = x0[0].movedim(1, 0)                                                  # [T, C, H, W]
    t = x.shape[0]
    if t > max_frames:
        idx = torch.linspace(0, t - 1, max_frames).round().long()
        x = x[idx.to(x.device)]
    if reshape is not None:
        x = reshape(x)
    f = factors.to(dtype=x.dtype, device=x.device)
    b = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    img = F.linear(x.movedim(1, -1), f, bias=b)                              # [N, H, W, 3]
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    h, w = img.shape[1], img.shape[2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img = F.interpolate(img.movedim(-1, 1).float(), size=(max(1, round(h * s)), max(1, round(w * s))), mode="bilinear", align_corners=False).movedim(1, -1)
    return img.mul(255).round().clamp(0, 255).to(device="cpu", dtype=torch.uint8)


def encode_frames(frames_u8, quality=80):
    """uint8 [N, h, w, 3] -> list of base64 JPEG strings."""
    from PIL import Image
    out = []
    for fr in frames_u8:
        buf = io.BytesIO()
        Image.fromarray(fr.numpy()).save(buf, format="JPEG", quality=quality)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


class StripPreviewer:
    """Stands in for the original previewer of ONE sampler run. 5D latents -> a frame strip event; anything else -> the original previewer, unchanged."""

    def __init__(self, base, rate=MAX_RATE, send=None, clock=None, ids=None):
        self.base, self.rate = base, int(rate)
        self.factors = base.latent_rgb_factors
        self.bias = getattr(base, "latent_rgb_factors_bias", None)
        self.reshape = getattr(base, "latent_rgb_factors_reshape", None)
        if send is None or ids is None:
            from tools.fusion import waveform_preview as WP
            send, ids = send or WP.send_preview, ids or WP.current_ids
        self.send, self.ids = send, ids
        self.clock = clock or time.perf_counter
        self.step, self.t_ret, self.debt, self.warned = 0, None, 0.0, False
        self.stats = {"frames": 0, "skipped": 0, "decode_s": 0.0}

    @classmethod
    def wrap(cls, base, latent_format=None, **kw):
        """-> a StripPreviewer for `base`, or None when `base` has nothing to decode with (no rgb factors: keep the original = the single still)."""
        if base is None or getattr(base, "latent_rgb_factors", None) is None:
            return None
        return cls(base, rate=playback_rate(latent_format), **kw)

    def decode_latent_to_preview_image(self, preview_format, x0):
        if getattr(x0, "ndim", 0) != 5:
            return self.base.decode_latent_to_preview_image(preview_format, x0)
        self.step += 1
        try:
            self._strip(x0)
        except Exception as e:                                                # noqa: BLE001
            if not self.warned:
                logging.warning(f"MOTNGRFX strip preview failed, render unaffected: {type(e).__name__}: {e}")
                self.warned = True
            self.t_ret = self.clock()
        return None

    def __getattr__(self, name):                                              # anything else a caller asks the previewer (VHS-style attributes)
        return getattr(self.base, name)

    def _strip(self, x0):
        t_in = self.clock()
        if self.t_ret is not None:
            self.debt -= t_in - self.t_ret
        if self.debt > 0:                                                     # the last decode is not paid for yet: drop this step's strip
            self.stats["skipped"] += 1
            self.t_ret = self.clock()
            return
        t0 = self.clock()
        u8 = decode_frames(x0.detach(), self.factors, self.bias, self.reshape)
        frames = encode_frames(u8)
        d = self.clock() - t0
        pid, node = self.ids()
        self.stats["frames"] += 1
        self.stats["decode_s"] += d
        self.send(EVENT, {"v": VERSION, "kind": "strip", "prompt_id": pid, "node": node, "step": self.step, "fmt": "jpeg", "w": int(u8.shape[2]), "h": int(u8.shape[1]),
                          "rate": self.rate, "frames": frames, "decode_s": round(d, 4), "skipped": self.stats["skipped"]})
        self.debt = self.clock() - t0
        self.t_ret = self.clock()


def flag_on(server_module=None) -> bool:
    """is the prompt that is running now opted in (extra_data.motngrfx_strip_preview)? Same lookup as VideoHelperSuite's animated preview: the running queue item's extra data."""
    try:
        if server_module is None:
            import server as server_module
        q = server_module.PromptServer.instance.prompt_queue
        for item in list(q.currently_running.values()):
            return bool((item[3] or {}).get(FLAG))
    except Exception:                                                         # noqa: BLE001
        pass
    return False


def install(latent_preview_module=None, server_module=None) -> bool:
    """wrap latent_preview.get_previewer once (idempotent). False when ComfyUI's module is not there (tests, a very old server)."""
    try:
        lp = latent_preview_module
        if lp is None:
            import latent_preview as lp
    except Exception:                                                         # noqa: BLE001
        return False
    orig = getattr(lp, "get_previewer", None)
    if orig is None or getattr(orig, "_motngrfx_strip", False):
        return False

    def get_previewer(device, latent_format, *a, **k):
        p = orig(device, latent_format, *a, **k)
        try:
            if p is not None and flag_on(server_module):
                sp = StripPreviewer.wrap(p, latent_format)
                if sp is not None:
                    return sp
        except Exception as e:                                                # noqa: BLE001
            logging.warning(f"MOTNGRFX strip preview not installed for this run: {type(e).__name__}: {e}")
        return p

    get_previewer._motngrfx_strip = True
    get_previewer.__wrapped__ = orig
    lp.get_previewer = get_previewer
    return True
