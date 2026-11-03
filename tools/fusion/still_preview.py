"""STILL PREVIEW: the fast first look of a piece, built from the shots' own stills (no LTX video). Pure functions (numpy / cv2, no ComfyUI): the nodes in nodes_fusion/footage.py, layers.py,
plan.py and the graph tooling call them, so the rules can be tested on the CPU. Doc: 16_app/STILL_PREVIEW.md.

    app quality `still` (shown as "Preview"; the app has two choices, Preview and Final)  ->  piece.quality "draft" (draft sizes, flat type, draft encode: every `== "draft"` rule of the code base applies unchanged) + piece.still_preview True

What a still-preview run does differently from a draft run (everything else, incl. the Gemma concept + design and the MOSS stills, is the draft run):
  * footage of a shot = that shot's OWN still (the row of the first-frame batch its `first` names), held for the shot length, at the compose size. No clean i2v, no text pass, no plate.
  * matte of a shot = the existing Alpha-Gen IC-LoRA on a 9-frame clip of that still (`ALPHA_FRAMES`), frame 0 repeated over the shot (MographDecodeClip, `hold_len`).
  * depth of a shot = Video Depth Anything Small on that still (1 frame); compose / the 3D cuts read depth from one frame anyway.
  * the footage flow is zero (a held picture does not move).
  * the stills are the draft-size stills: a later draft render with real LTX (tools/app/upgrade_still.py) reuses the identical images and the identical design.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np

from tools.fusion import places as PL

APP_FINAL_TIER = "draft"          # the internal tier the app's "Final" renders at: LTX 576x1024 single stage, compose 540x960 (plan_rules.sizes_for). The real two-stage final (1152x2048 / 1080x1920) has never run as a
                                  # whole piece on the 24 GB box; when it has, change this ONE line to "final" (the upgrade lock then refuses: its stills are draft-size, see upgraded_piece)
APP_QUALITIES = (("still", "Preview"), ("final", "Final"))     # the two choices the graph's select shows (value, label); `preview` / `draft` stay internal tiers. v20: the page shows no select and sends `draft540` (PREVIEW_FULL)
PREVIEW_FULL = "draft540"         # v20 "Make it": the real LTX render at the draft size (compose 540x960) WITHOUT the Real-ESRGAN 1080 upscale (the lazy `OUT final 1080` stays idle)
NO_UPSCALE = "no_upscale"         # request / piece.request key set by PREVIEW_FULL; tools/app/final_upscale.wants_upscale reads it
PLATES_ONLY_Q = "draft540plates"  # 2026-11-03 image board edited like a video: the PREVIEW_FULL render, but MographCompose returns each shot's plain footage (no type, no compose ops); the page then sends that clip to /mograph/own_video (overlay)
PLATES_ONLY = "plates_only"       # request / piece.request key set by PLATES_ONLY_Q; nodes_fusion/compose.py reads it
QUALITY = "still"                 # the value of the app's Quality field for the still-based Preview
FLAG = "still_preview"            # piece / request key
LABEL = "Preview"
SUFFIX = "_still"                 # part / output label suffix
ALPHA_FRAMES = 9                  # Alpha-Gen reference length: 8k+1, the shortest clip with two latent frames
REQUEST_KEY = "request"           # piece key: the app inputs the piece was planned from (words, scene, music, look, shots, owners, length, seed, format), for the upgrade
REQUEST_FIELDS = ("words", "scene", "music", "length", "look", "seed", "format", "shots", "owners", "layout_seed", "direction", NO_UPSCALE, PLATES_ONLY)


def is_still(piece: dict | None = None, request: dict | None = None) -> bool:
    """did the app ask for the still preview? (piece flag, else the request's)"""
    return bool((piece or {}).get(FLAG) or (request or {}).get(FLAG))


def split_quality(value) -> tuple[str, bool]:
    """the app's quality string -> (quality the planner knows, still preview?): `still` (Preview) is a draft-size run with held stills; `final` is APP_FINAL_TIER; `preview` / `draft` are
    the internal tiers (batch drivers, tests) and pass through."""
    q = str(value or "").strip().lower()
    if q == QUALITY:
        return "draft", True
    if q in (PREVIEW_FULL, PLATES_ONLY_Q):
        return APP_FINAL_TIER, False                  # same tier as Final; the no-upscale flag is set by `is_preview_full`
    return (APP_FINAL_TIER, False) if q == "final" else (q, False)


def is_preview_full(value) -> bool:
    """the app's quality string asks for the full render at the draft size without the 1080 upscale (v20 Make it)."""
    return str(value or "").strip().lower() in (PREVIEW_FULL, PLATES_ONLY_Q)


def is_plates_only(value) -> bool:
    """the page asked for the plain footage (quality `draft540plates`): no type, no compose ops; the own-video overlay route puts the designed type on it."""
    return str(value or "").strip().lower() == PLATES_ONLY_Q


def request_record(raw: dict | None) -> dict:
    """the JSON-safe app inputs worth keeping on the piece (the upgrade re-runs the pipeline from them)."""
    raw = raw or {}
    return {k: (raw[k] if isinstance(raw[k], (int, float, bool)) else str(raw[k])) for k in REQUEST_FIELDS if raw.get(k) is not None}


def label(part: str) -> str:
    """`r01_gemma_ltx` -> `r01_gemma_ltx_still` (once)."""
    return part if part.endswith(SUFFIX) else part + SUFFIX


def shot_row(piece: dict, sid: str, n_rows: int) -> int:
    """row of the first-frame batch (MographFirstFrames: one row per place) that shot `sid` starts from = its own still."""
    shot = next((s for s in (piece or {}).get("shots") or [] if s.get("id") == sid), {})
    return PL.pick_row(piece, shot.get("first"), n_rows)


def cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """scale to cover (w, h) and centre-crop (float or uint8, same dtype back)."""
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    out = out[y0:y0 + h, x0:x0 + w]
    if out.dtype != np.uint8:
        out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out)


def hold_u8(img: np.ndarray, n: int, size: tuple[int, int]) -> np.ndarray:
    """one still (float32 HxWx3 0..1) -> uint8 [n, h, w, 3] at the compose size `size` (w, h): the same picture on every frame. No zoom: the matte and the depth are cut from this exact picture
    and held with it, so the subject stays on its matte (a scale ease would need the matte warped frame by frame: 2.7 GB per shot at 576x1024 floats)."""
    w, h = int(size[0]), int(size[1])
    one = np.clip(np.rint(cover(np.asarray(img, np.float32), w, h) * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(one[None], max(int(n), 1), axis=0)


def hold_len(piece: dict, sid: str, default: int = 1) -> int:
    """frames of shot `sid` in the layout (the length the held matte gets)."""
    return int(((piece or {}).get("layout") or {}).get(sid, {}).get("frames") or default)


def zero_flow_shape(u8_shape) -> tuple[int, int, int, int]:
    """fp16 flow [T, H/2, W/2, 2] of a held picture (all zeros)."""
    t, h, w = int(u8_shape[0]), int(u8_shape[1]), int(u8_shape[2])
    return (t, max(h // 2, 1), max(w // 2, 1), 2)


def prepare_piece(piece: dict, raw_request: dict | None, still: bool) -> dict:
    """the planned piece with the still-preview flag and the app inputs on it (what MographPiecePlan does after plan_piece). Not a copy."""
    if raw_request:
        piece[REQUEST_KEY] = request_record(raw_request)
    if still:
        piece[FLAG] = True
    return piece


def upgraded_piece(piece_json: dict, quality: str | None = None) -> dict:
    """a still-preview piece turned into the piece of the app's Final render (`APP_FINAL_TIER`): flag off, the same design. Only the draft tier is possible: the stills are draft-size stills."""
    quality = quality or APP_FINAL_TIER
    if quality != "draft":
        raise ValueError(f"the stills of a Preview are draft-size; only the draft tier re-uses them (asked for {quality!r}; APP_FINAL_TIER = {APP_FINAL_TIER!r})")
    p = copy.deepcopy(piece_json)
    p.pop(FLAG, None)
    p["quality"] = quality
    return p
