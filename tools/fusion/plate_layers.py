"""plate_layers: CLEAN PLATE + INDEPENDENT SUBJECT LAYER (probe 24_clean-plate/v001, user: "this way is way cleaner"; docs 16_app/CLEAN_PLATE.md section 5 and "App route" below).

A shot with `"plate": "clean"` is made of TWO footage layers that move on their own, with the type between them:

    plate    the CLEAN clip with the subject taken off the background (LTX-2.5 Clean Plate IC-LoRA, V2V of the clean clip, graph chain `p{k}.*` in tools/app/layered_shots.py).
             It has its own pan: `plate_pan`.
    type     the flat type layers (compose recipe: sandwich between the layers)
    subject  the CLEAN clip x ITS OWN matte (Alpha-Gen cut from that clip), moved by `subject_move`, edge feathered, colour-matched to the plate.

The compose recipes are written against `$footage` (the picture under the type) + `$subject` (what stands in front of the type) + `$matte` (the cut-out of the subject).
This module BAKES the independent motion before the recipe runs, so no recipe changes: `bake(...)` returns a new footage (= the moved plate), a new subject (= the moved,
colour-matched clean clip) and a new matte (= the same matte, feathered, moved with the SAME affine as the subject, at the clip's geometry). The matte follows its
picture: the subject and its matte are warped frame by frame by the identical matrix (tests/test_plate_layers.py::test_matte_follows_the_subject).

Shot keys (piece JSON, all optional except `plate`):

    "plate": "clean"
    "subject_move": {"dx": 0.08, "dy": 0.0, "scale": 1.10, "ease": "linear", "anchor": [0.5, 0.75], "feather": 0.0025}
        END values reached at the last frame, from the identity at the first: dx, dy as fractions of the frame width / height (+ = right / down), scale about `anchor`
        (fractions of the frame, default bottom-centre-ish [0.5, 0.75]), `ease` one of EASES, `feather` the matte softness as a fraction of the short side.
    "plate_pan": {"dx": 0.035, "dy": 0.0, "scale": 1.12, "ease": "linear"}
        the plate pans from (+dx, +dy) to (-dx, -dy) of the frame about the frame centre at a constant `scale`; the scale is raised to 1 + 2 * max(|dx|, |dy|) when it is lower
        (the pan must stay inside the plate, CLEAN_PLATE.md section 5.5). The probe's look: subject drifts right and grows while the plate pans left.
    "plate_prompt": "<full empty-scene prompt>" | "plate_subject": "<what to remove>"      (overrides of the template)

A bare `"plate": "clean"` takes DEFAULT_MOVE / DEFAULT_PAN (the numbers of the probe composite `c_k_independent.mp4`).

TYPE AS A MATTE (user 2026-10-29: "also using the type as a matte as an option"): a shot (or a word of the shot) may say `"type_matte": <mode>`. The letter matte is the type's own
matte pass (`<type_dir>/matte/matte_NNNNN.png`, frame-exact with the type's motion, soft edges), P = the moved plate, S = the moved subject (clean clip x its matte), C = S over P:

    reveal_subject   picture = P outside the letters, C inside them: the subject exists only inside the word (nothing stands in front of the type)
    reveal_plate     picture = C outside the letters, P inside them: the word erases the subject and shows the clean plate through it (nothing in front of the type)
    subject_through  picture = P, the type drawn over it, the subject layer in front of the type ONLY where it overlaps the letters (matte x letters): the word is a window onto the subject
    plate_in_subject (contrast moment, 2026-10-30) flat palette colour -> the word -> the subject's matte silhouette FILLED WITH THE CLEAN PLATE (the empty scene shows through the person's shape) on top
    plate_in_type    (contrast moment, 2026-10-30) flat palette colour -> the subject (clean clip x its matte) -> the word on top, its letters FILLED WITH THE CLEAN PLATE
    window_subject_front (contrast SECTION, 2026-10-30) flat ground (the piece's darkest palette colour; `flat: "light"` on the shot = the lightest) -> the letters are WINDOWS onto the footage
                     (the clean plate when the shot has one, else the shot's own clip) with a thin outline -> the subject (clip x its matte) stands IN FRONT of letters and ground
    text_as_mask     (2026-10-30, own-video v006) flat ground -> the letters' INTERIORS are the mask: filled with the footage (or the clean plate, `mask_fill`), the subject is NOT drawn on top
                     (pure knockout); the animated edge line runs along the letter edges
    window_other     the existing cut option (the letters show the NEXT shot, compose.cuts / window_cut); accepted here so one key names every mode, nothing is baked for it

The three modes need the plate, so they imply `"plate": "clean"`. All of them are baked into the same three arrays as the plain plate route (footage, subject, matte), so the matte follows
its picture exactly and the recipe's own guards (legibility, hold net, colour stage) still read the picture that is really under the type.

Colour match: the Clean Plate LoRA regenerates the WHOLE frame (3.6-7 of 255 drift in the background, CLEAN_PLATE.md section 3), so the subject, which comes from the original
clean render, would sit slightly off the plate. `fit_color` fits gain / offset per channel (plate = gain * clean + offset) over the BACKGROUND pixels (the matte, dilated, excluded)
on a few frames, clamps the gain, and `bake` applies it to the subject pixels. Not a local match: a subject lit differently from its surroundings keeps its own light.
Not done: a contact shadow / reflection for the moved subject (the plate has none: the LoRA removed them), motion blur, rotation.

Pure numpy / cv2 (torch only to wrap a clip dict in `bake_clip`); no ComfyUI. Frame arrays are uint8 [T,H,W,3]; mattes float 0..1 or uint8, any size (reference geometry).
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
PLATE_SIZE = (576, 1024)                  # the LoRA's trained bucket for 9:16 (CLEAN_PLATE.md section 2): the plate stage runs at the Alpha-Gen reference size
EASES = ("linear", "ease_in", "ease_out", "ease_in_out")
CONTRAST_MODES = ("plate_in_subject", "plate_in_type", "window_subject_front", "text_as_mask")   # flat ground + clean plate in the subject's shape / in the letters / letter windows with the subject in front (contrast moments / sections)
TYPE_MATTE_MODES = ("reveal_subject", "reveal_plate", "subject_through") + CONTRAST_MODES        # baked here; they need the plate
FLAT_RGB, WORD_RGB, OUTLINE_SIGMA = (196, 48, 40), (244, 238, 224), 0.004       # default flat ground / word colour (0..255) and the dark outline's blur (x short side) of a contrast moment
OCC_GUARDED = ("window_subject_front", "plate_in_type")     # modes whose word must read against the subject: the occlusion guard runs on them
OCC_WORD, OCC_LETTER = 0.35, 0.70                           # the subject may cover at most 35 % of the word's area and 70 % of any one letter over the section
EDGE_WIDTH, EDGE_DASH, EDGE_GAP, EDGE_SPEED = 2, 18.0, 12.0, 3.0            # animated edge line (px at the clip's size, crisp, no blur): dash / gap length along the contour and the travel per frame
FILL_MIN_TEXTURE = 5.0                                      # (2026-10-30, no flat fills) a fill inside the subject's silhouette must be TEXTURED: mean local std (7x7, 0..255 luma) inside the matte >= this ...
FILL_MIN_MOTION = 1.0                                       # ... and MOVING: mean abs luma difference between frames 2 apart inside the matte >= this (0..255 units); below either floor the candidate is a flat fill
FILL_SOURCES = ("plate", "footage", "other_clip")
FILL_TINT = 0.20                                            # share (0.15-0.25) of the word's ink mixed into the plate / window fill so the fills sit in the piece palette (fill_tint = 0 switches it off)
TYPE_MATTE_ALL = TYPE_MATTE_MODES + ("window_other",)                           # window_other = the compose.cuts window cut (window_cut.py), nothing to bake
LETTER_SOFT = 0.002                       # gaussian sigma of the letter matte (fraction of the short side): soft edges
DEFAULT_MOVE = {"dx": 0.08, "dy": 0.0, "scale": 1.10, "ease": "linear", "anchor": [0.5, 0.75]}      # composite_layers.py: subject drifts +8 % of the width, grows to 1.10 about bottom-centre
DEFAULT_PAN = {"dx": 0.035, "dy": 0.0, "scale": 1.12, "ease": "linear"}                             # the plate scaled 1.12 and panned +3.5 % -> -3.5 % of the width
FEATHER = 0.0025                          # matte softness (gaussian sigma) as a fraction of the short side; erode = 0.6 x that
GAIN_RANGE = (0.8, 1.25)                  # clamp of the colour-match gain (the measured drift is 1-3 %)
FIT_FRAMES = 6                            # frames the colour fit looks at
FIT_MIN_BG = 0.02                         # a colour fit needs this share of background pixels, else identity


# --------------------------------------------------------------------------------------------------------------- shot keys
SHOT_KEYS = ("plate", "subject_move", "plate_pan", "plate_subject", "plate_prompt", "type_matte", "flat", "fill_tint")        # the shot keys of this route
OVERRIDE_ENV = "MOGRAPH_SHOT_KEYS"        # PROBE HOOK: path of a JSON file {"*": {"s2": {...keys}}, "<piece slug or id>": {"s3": {...}}} merged over the piece's shots (read at every call, so it
                                          # can be edited between runs without a restart; the server needs the variable set once). Until plan_rules copies the keys from the concept's shot_plan
                                          # (`shot_keys` below) this is the way to ask a box run for a plate / type-matte shot. Unset = no effect.


def shot_keys(entry) -> dict:
    """the plate / type-matte keys of a concept `shot_plan` entry (or any dict), normalised, for plan_rules to copy into the piece's shot: `shots[k].update(PLT.shot_keys(sp_plan[k]))`."""
    e = entry if isinstance(entry, dict) else {}
    out: dict = {}
    if str(e.get("plate") or "").strip().lower() == "clean":
        out["plate"] = "clean"
    m = norm_type_matte(e.get("type_matte"))
    if m:
        out["type_matte"] = m
    if out.get("plate") or m in TYPE_MATTE_MODES:
        if isinstance(e.get("subject_move"), dict):
            out["subject_move"] = norm_move(e["subject_move"])
        if isinstance(e.get("plate_pan"), dict):
            out["plate_pan"] = norm_pan(e["plate_pan"])
        for k in ("plate_subject", "plate_prompt"):
            if str(e.get(k) or "").strip():
                out[k] = str(e[k]).strip()
    if m in CONTRAST_MODES:
        if str(e.get("flat") or "").strip().lower() in ("dark", "light"):
            out["flat"] = str(e["flat"]).strip().lower()
        try:
            if e.get("fill_tint") is not None:
                out["fill_tint"] = float(min(max(float(e["fill_tint"]), 0.0), 0.5))
        except (TypeError, ValueError):
            pass
    return out


def _override(piece: dict, sid: str) -> dict:
    path = os.environ.get(OVERRIDE_ENV)
    if not path or not os.path.isfile(path):
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    slug = str((piece or {}).get("slug") or (piece or {}).get("id") or "")
    o = {**((d.get("*") or {}).get(sid) or {}), **((d.get(slug) or {}).get(sid) or {})} if isinstance(d, dict) else {}
    return {k: v for k, v in o.items() if k in SHOT_KEYS}


def shot_of(piece: dict, sid: str) -> dict:
    sh = next((s for s in (piece or {}).get("shots") or [] if s.get("id") == sid), {})
    ov = _override(piece, sid)
    return {**sh, **ov} if ov else sh


def norm_type_matte(v) -> str | None:
    """a `type_matte` value -> one of TYPE_MATTE_ALL or None (unknown words are ignored)."""
    m = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
    return m if m in TYPE_MATTE_ALL else None


def type_matte_of(piece: dict, sid: str) -> str | None:
    """the type-matte mode of a shot: its own `type_matte`, else the first word whose hit falls inside the shot and carries one (`type.words[i].type_matte`; `hit` in seconds)."""
    m = norm_type_matte(shot_of(piece, sid).get("type_matte"))
    if m:
        return m
    lay = (piece or {}).get("layout") or {}
    L = lay.get(sid)
    if not L:
        return None
    fps = float((piece or {}).get("fps", 24) or 24)
    a, b = L["start"] / fps, (L["start"] + L["frames"]) / fps
    for w in ((piece or {}).get("type") or {}).get("words") or []:
        try:
            hit = float(w.get("hit"))
        except (TypeError, ValueError):
            continue
        mw = norm_type_matte(w.get("type_matte"))
        if mw and a <= hit < b:
            return mw
    return None


def wants_plate(piece: dict, sid: str) -> bool:
    """does shot `sid` take the clean-plate route? `plate: "clean"`, or a baked type-matte mode (they need the plate); unless the piece opts out with `layers.plate = false`."""
    if ((piece or {}).get("layers") or {}).get("plate") is False:
        return False
    return str(shot_of(piece, sid).get("plate") or "").strip().lower() == "clean" or type_matte_of(piece, sid) in TYPE_MATTE_MODES


def plate_sids(piece: dict) -> list[str]:
    """the active shots that ask for a plate (in layout order)."""
    lay = (piece or {}).get("layout")
    return [s["id"] for s in (piece or {}).get("shots") or [] if (lay is None or s["id"] in lay) and wants_plate(piece, s["id"])]


def _f(v, default, lo, hi) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(min(max(x, lo), hi)) if x == x else float(default)


def _ease_name(v, default) -> str:
    e = str(v or default).strip().lower().replace("-", "_").replace(" ", "_")
    return e if e in EASES else str(default)


def norm_move(move) -> dict:
    """a `subject_move` dict with every field present and clamped (dx / dy +-0.5, scale 0.5..3, anchor 0..1, feather 0..0.02); a missing / non-dict move = DEFAULT_MOVE."""
    m = dict(move) if isinstance(move, dict) else {}
    d = DEFAULT_MOVE
    anc = m.get("anchor")
    anc = [_f(anc[0], d["anchor"][0], 0.0, 1.0), _f(anc[1], d["anchor"][1], 0.0, 1.0)] if isinstance(anc, (list, tuple)) and len(anc) == 2 else list(d["anchor"])
    if not m:
        return dict(d, feather=FEATHER)
    return {"dx": _f(m.get("dx"), 0.0, -0.5, 0.5), "dy": _f(m.get("dy"), 0.0, -0.5, 0.5), "scale": _f(m.get("scale"), 1.0, 0.5, 3.0),
            "ease": _ease_name(m.get("ease"), d["ease"]), "anchor": anc, "feather": _f(m.get("feather"), FEATHER, 0.0, 0.02)}


def norm_pan(pan) -> dict:
    """a `plate_pan` dict, every field present: dx / dy +-0.25, scale raised to 1 + 2 x max(|dx|, |dy|) (+0.005) so the pan stays inside the plate, at most 2.0."""
    p = dict(pan) if isinstance(pan, dict) else {}
    if not p:
        p = dict(DEFAULT_PAN)
    dx, dy = _f(p.get("dx"), 0.0, -0.25, 0.25), _f(p.get("dy"), 0.0, -0.25, 0.25)
    need = 1.0 + 2.0 * max(abs(dx), abs(dy)) + (0.005 if (dx or dy) else 0.0)
    return {"dx": dx, "dy": dy, "scale": min(max(_f(p.get("scale"), DEFAULT_PAN["scale"], 1.0, 2.0), need), 2.0), "ease": _ease_name(p.get("ease"), DEFAULT_PAN["ease"])}


def shot_moves(piece: dict, sid: str) -> tuple[dict, dict]:
    """(subject_move, plate_pan) of a shot, normalised."""
    sh = shot_of(piece, sid)
    return norm_move(sh.get("subject_move")), norm_pan(sh.get("plate_pan"))


# --------------------------------------------------------------------------------------------------------------- motion
def ease(name: str, t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    if name == "ease_in":
        return t * t
    if name == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if name == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return t


def move_at(move: dict, t: float) -> tuple[float, float, float]:
    """(dx, dy, scale) of the subject at t in 0..1: identity at 0, the end values at 1."""
    e = ease(move["ease"], t)
    return move["dx"] * e, move["dy"] * e, 1.0 + (move["scale"] - 1.0) * e


def pan_at(pan: dict, t: float) -> tuple[float, float, float]:
    """(dx, dy, scale) of the plate at t in 0..1: from (+dx, +dy) to (-dx, -dy) at a constant scale."""
    e = ease(pan["ease"], t)
    return pan["dx"] * (1.0 - 2.0 * e), pan["dy"] * (1.0 - 2.0 * e), pan["scale"]


def affine(hw: tuple[int, int], dx: float, dy: float, scale: float, anchor=(0.5, 0.5)) -> np.ndarray:
    """2x3 float32 matrix (cv2.warpAffine, source -> destination): scale about the anchor (fractions of the frame), then shift by dx x W, dy x H."""
    h, w = int(hw[0]), int(hw[1])
    cx, cy = float(anchor[0]) * w, float(anchor[1]) * h
    return np.float32([[scale, 0.0, cx - scale * cx + dx * w], [0.0, scale, cy - scale * cy + dy * h]])


# --------------------------------------------------------------------------------------------------------------- geometry of the matte
def ref_to_clip(ref: np.ndarray, clip_hw: tuple[int, int]) -> np.ndarray:
    """a matte frame cut from the clip (Alpha-Gen reference = the clip cover-resized to the reference size, nodes_fusion/footage._u8_to_ref) -> the clip's own geometry. The
    reference shows the centred window of the clip it was cropped to; the window is scaled back and padded with the matte's own border where the crop left nothing
    (the same convention as matte_align.ref_to_frame). float32 HxW in 0..1."""
    m = np.asarray(ref)
    if m.ndim == 3:
        m = m[..., 0] if m.shape[2] in (1, 3, 4) else m.mean(axis=2)
    m = m.astype(np.float32)
    if m.size and m.max() > 1.5:
        m = m / (65535.0 if m.max() > 255.5 else 255.0)
    fh, fw = int(clip_hw[0]), int(clip_hw[1])
    mh, mw = m.shape[:2]
    s = max(mw / float(fw), mh / float(fh))                          # clip -> reference scale (cover)
    th, tw = max(1, min(fh, int(round(mh / s)))), max(1, min(fw, int(round(mw / s))))
    r = m if (th, tw) == (mh, mw) else cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA if tw < mw else cv2.INTER_LINEAR)
    if (th, tw) == (fh, fw):
        return np.clip(r, 0.0, 1.0)
    py0, px0 = (fh - th) // 2, (fw - tw) // 2
    return np.clip(np.pad(r, ((py0, fh - th - py0), (px0, fw - tw - px0)), mode="edge"), 0.0, 1.0)


def soften(m: np.ndarray, feather: float = FEATHER) -> np.ndarray:
    """feathered, slightly eroded matte (float32 0..1): erode so the clean clip's background fringe is not pasted, then a gaussian whose sigma is `feather` x the short side."""
    short = min(m.shape[:2])
    sigma = float(feather) * short
    r = int(round(0.6 * sigma))
    if r >= 1:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if sigma >= 0.3:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


# --------------------------------------------------------------------------------------------------------------- colour match
def fit_color(clean_frames, plate_frames, matte_frames, n: int = FIT_FRAMES) -> dict:
    """per-channel gain / offset that maps the clean clip onto the plate: plate = gain * clean + offset, fitted over the BACKGROUND pixels of `n` evenly spaced frames
    (matte, dilated by 2 % of the short side, excluded). clean_frames / plate_frames: uint8 [T,H,W,3] (same geometry), matte_frames: [T,H,W] float in the same geometry.
    -> {"gain": [3], "offset": [3] (0..1 units), "bg": share of background pixels used, "ok": bool}. Fewer than FIT_MIN_BG background pixels = the identity (ok False)."""
    T = min(len(clean_frames), len(plate_frames), len(matte_frames))
    if T <= 0:
        return {"gain": [1.0] * 3, "offset": [0.0] * 3, "bg": 0.0, "ok": False}
    sel = sorted({int(round(i * (T - 1) / max(n - 1, 1))) for i in range(n)})
    xs, ys, share = [], [], []
    for j in sel:
        m = np.asarray(matte_frames[j], np.float32)
        k = max(1, int(round(0.02 * min(m.shape[:2]))))
        bg = cv2.dilate((m > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) == 0
        share.append(float(bg.mean()))
        if bg.sum() < 64:
            continue
        sub = np.argwhere(bg)[:: max(1, int(bg.sum() // 6000))]
        c = np.asarray(clean_frames[j])[sub[:, 0], sub[:, 1], :3].astype(np.float32) / 255.0
        p = np.asarray(plate_frames[j])[sub[:, 0], sub[:, 1], :3].astype(np.float32) / 255.0
        xs.append(c)
        ys.append(p)
    if not xs or float(np.mean(share)) < FIT_MIN_BG:
        return {"gain": [1.0] * 3, "offset": [0.0] * 3, "bg": round(float(np.mean(share)) if share else 0.0, 4), "ok": False}
    x, y = np.concatenate(xs), np.concatenate(ys)
    gain, off = [], []
    for c in range(3):
        xc, yc = x[:, c], y[:, c]
        var = float(xc.var())
        g = float(np.cov(xc, yc, bias=True)[0, 1] / var) if var > 1e-8 else 1.0
        g = min(max(g, GAIN_RANGE[0]), GAIN_RANGE[1])
        gain.append(round(g, 5))
        off.append(round(float(yc.mean() - g * xc.mean()), 5))
    return {"gain": gain, "offset": off, "bg": round(float(np.mean(share)), 4), "ok": True}


# --------------------------------------------------------------------------------------------------------------- the letters
class LetterReader:
    """the type matte pass of a shot as float HxW frames at the clip's geometry: `<type_dir>/matte/*.png`, frame j of the shot = file `offset + j` (clamped; offset = the shot's
    piece start frame, the index the recipe's `$type_matte` uses), softened with a gaussian of LETTER_SOFT x the short side. A directory without frames reads as no letters."""

    def __init__(self, type_dir, offset: int, hw: tuple[int, int], soft: float = LETTER_SOFT):
        from pathlib import Path
        d = Path(str(type_dir)) / "matte" if type_dir else None
        self.files = sorted(str(f) for f in d.glob("*.png")) if d is not None and d.is_dir() else []
        self.offset, self.hw, self.soft = int(offset), (int(hw[0]), int(hw[1])), float(soft)

    def __call__(self, j: int) -> np.ndarray:
        h, w = self.hw
        if not self.files:
            return np.zeros((h, w), np.float32)
        im = cv2.imdecode(np.fromfile(self.files[min(max(self.offset + int(j), 0), len(self.files) - 1)], np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:
            return np.zeros((h, w), np.float32)
        if im.ndim == 3:
            im = im[..., 3] if im.shape[2] == 4 and im[..., 3].max() > 0 else im[..., :3].mean(axis=2)
        m = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA if m.shape[1] > w else cv2.INTER_LINEAR)
        if self.soft * min(h, w) >= 0.3:
            m = cv2.GaussianBlur(m, (0, 0), self.soft * min(h, w))
        return np.clip(m, 0.0, 1.0)


# --------------------------------------------------------------------------------------------------------------- bake
def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def edge_line_alpha(mask, phase: float = 0.0, width: int = EDGE_WIDTH, dash: float = EDGE_DASH, gap: float = EDGE_GAP, speed: float = EDGE_SPEED, thresh: float = 0.5) -> np.ndarray:
    """ANIMATED EDGE LINE: the contour of `mask` (the subject's Alpha-Gen matte, or the letters) at `thresh`, drawn as a thin crisp line (`width` px, anti-aliased, NO blur / glow) made of
    dashes of `dash` px with `gap` px between, that TRAVEL along the contour by `speed` px per frame (`phase` = the frame number: the dash pattern slides, the contour itself is that
    frame's own matte, so the line follows the matte frame by frame). -> float32 HxW 0..1 (the line's alpha)."""
    m = (np.asarray(mask, np.float32) > thresh).astype(np.uint8)
    out = np.zeros(m.shape[:2], np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    period = float(dash + gap)
    for c in cnts:
        pts = c[:, 0, :].astype(np.float32)
        if len(pts) < 30:
            continue
        seg = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        on = ((arc - float(phase) * float(speed)) % period) < float(dash)
        run = []
        for i in range(len(pts)):
            if on[i]:
                run.append(pts[i])
            elif run:
                if len(run) > 1:
                    cv2.polylines(out, [np.round(np.asarray(run)).astype(np.int32).reshape(-1, 1, 2)], False, 255, int(width), cv2.LINE_AA)
                run = []
        if len(run) > 1:
            cv2.polylines(out, [np.round(np.asarray(run)).astype(np.int32).reshape(-1, 1, 2)], False, 255, int(width), cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


EDGE_MODE = "sweep"      # default animation of the contrast-moment edge line (2026-10-30): "sweep" = brush strokes along the contour (tools/fusion/contour_sweep.edge_alpha: variable width, tapered ends, 2 strokes,
                         # wobble, breaks, draws on / off at the section's ends); "dash" = the earlier travelling dashes (edge_line_alpha), still reachable per call


def edge_alpha(mask, phase: float = 0.0, mode: str | None = None, seed: int = 0, section=None, guard: bool = True) -> np.ndarray:
    """the animated edge line of `mask` at frame `phase` in the chosen mode (None = EDGE_MODE). section = (first frame, last frame) of the contrast moment (the sweep draws on / off at its ends)."""
    if (mode or EDGE_MODE) == "dash":
        return edge_line_alpha(mask, phase)
    from tools.fusion import contour_sweep as CS
    return CS.edge_alpha(mask, phase, seed=seed, section=section, guard=guard)      # guard: no stroke along the frame border / long straight frame lines (False for letters)


def occlusion_share(letters_frames, matte_frames, min_area: int = 24) -> dict:
    """how much of the WORD the subject's matte covers over a section. letters_frames / matte_frames: sequences of float HxW (the letters at each sampled frame, the subject's matte
    moved like the subject). Only frames where the word is formed (letter area >= 60 % of the largest) count. -> {"word": mean share of the letter area under matte > 0.5,
    "letter": the worst single letter (connected blob of the letters) share over those frames, "frames": n}; a word with no letters reads as 0 / 0."""
    areas = [float((np.asarray(L) > 0.5).sum()) for L in letters_frames]
    top = max(areas) if areas else 0.0
    if top < min_area:
        return {"word": 0.0, "letter": 0.0, "frames": 0}
    words, worst, n = [], 0.0, 0
    for L, m, a in zip(letters_frames, matte_frames, areas):
        if a < 0.6 * top:
            continue
        Lb = (np.asarray(L) > 0.5).astype(np.uint8)
        Mb = np.asarray(m) > 0.5
        words.append(float((Lb.astype(bool) & Mb).sum()) / max(float(Lb.sum()), 1.0))
        k, lab = cv2.connectedComponents(Lb, connectivity=8)
        for c in range(1, k):
            blob = lab == c
            ar = float(blob.sum())
            if ar >= min_area:
                worst = max(worst, float((blob & Mb).sum()) / ar)
        n += 1
    return {"word": round(float(np.mean(words)) if words else 0.0, 4), "letter": round(worst, 4), "frames": n}


def occlusion_decision(share: dict, mode: str, can_relayout: bool = False) -> dict:
    """the guard's decision for a section (CLEAN_PLATE.md section 7). Over the limits (word > OCC_WORD or any letter > OCC_LETTER) a guarded mode first asks for a BIGGER LAYOUT (`layout`:
    "grid_fill": letters filling the frame, extending past the subject; only when `can_relayout`, i.e. the type stage can still redraw the word), else the mode becomes plate_in_subject.
    -> {"action": keep | bigger_layout | swap, "mode": the mode to bake, "layout": None | "grid_fill", "share", "limits", "reason"}; the caller records it in the piece."""
    d = {"action": "keep", "mode": mode, "layout": None, "share": dict(share), "limits": {"word": OCC_WORD, "letter": OCC_LETTER}, "reason": ""}
    if mode not in OCC_GUARDED:
        return d
    over = share.get("word", 0.0) > OCC_WORD or share.get("letter", 0.0) > OCC_LETTER
    if not over:
        return d
    why = f"subject covers {share.get('word', 0.0):.0%} of the word / {share.get('letter', 0.0):.0%} of its worst letter (limits {OCC_WORD:.0%} / {OCC_LETTER:.0%})"
    if can_relayout:
        return dict(d, action="bigger_layout", layout="grid_fill", reason=why + ": letters redrawn as a frame-filling grid")
    return dict(d, action="swap", mode="plate_in_subject", reason=why + ": no bigger layout available, mode swapped to plate_in_subject")


def fill_measure(frames, matte, n: int = 8, step: int = 2) -> dict:
    """TEXTURE + MOTION of a candidate fill over a section, measured INSIDE the matte. frames: uint8 [T,H,W,3] (any size: matte frames are resized to it), matte: float/uint8 [T,H,W] or a callable j -> HxW
    (0..1, the silhouette at the clip geometry). texture = mean local std (7x7 box, luma 0..255) over `n` sample frames; motion = mean |luma(j + step) - luma(j)| over the same samples.
    A flat (pale / blurred) clean plate reads low on BOTH: that is what a flat-looking fill is. -> {"texture", "motion", "area"} (area = mean matte share of the frame)."""
    fr = _np(frames)
    T = int(fr.shape[0])
    if T == 0:
        return {"texture": 0.0, "motion": 0.0, "area": 0.0}
    idx = sorted({int(round(i * (T - 1) / max(n - 1, 1))) for i in range(n)})
    tex, mot, area = [], [], []
    for j in idx:
        j2 = min(j + step, T - 1)
        j1 = j2 - step if j2 - step >= 0 else 0
        g1 = cv2.cvtColor(np.ascontiguousarray(fr[j1][..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(np.ascontiguousarray(fr[j2][..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = matte(j) if callable(matte) else _np(matte)[min(j, len(matte) - 1)]
        m = np.asarray(m, np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if m.max() > 1.5:
            m = m / 255.0
        if m.shape != g1.shape:
            m = cv2.resize(m, (g1.shape[1], g1.shape[0]), interpolation=cv2.INTER_LINEAR)
        sel = m > 0.5
        if int(sel.sum()) < 50:
            continue
        mu = cv2.blur(g2, (7, 7))
        sd = np.sqrt(np.maximum(cv2.blur(g2 * g2, (7, 7)) - mu * mu, 0.0))
        tex.append(float(sd[sel].mean()))
        mot.append(float(np.abs(g2 - g1)[sel].mean()))
        area.append(float(sel.mean()))
    if not tex:
        return {"texture": 0.0, "motion": 0.0, "area": 0.0}
    return {"texture": round(float(np.mean(tex)), 3), "motion": round(float(np.mean(mot)), 3), "area": round(float(np.mean(area)), 4)}


def pick_fill(measured: dict, order=FILL_SOURCES, min_texture: float = FILL_MIN_TEXTURE, min_motion: float = FILL_MIN_MOTION, force: str | None = None) -> dict:
    """which candidate fills the silhouette. measured = {source: {"texture", "motion"}} (a missing source = not available). The first source in `order` that is above BOTH floors wins (the clean plate
    when it is textured and moving, else the shot's own footage, else the other shot's moving clip); none above -> the best by min(texture / floor, motion / floor). NEVER a palette / flat fill.
    -> {"fill_source", "measured", "floors", "why"}."""
    floors = {"texture": min_texture, "motion": min_motion}
    have = [k for k in order if k in measured and measured[k] is not None]
    if force and force in have:
        return {"fill_source": force, "measured": measured, "floors": floors, "why": "forced by the shot / moment"}
    score = lambda k: min(measured[k]["texture"] / max(min_texture, 1e-6), measured[k]["motion"] / max(min_motion, 1e-6))      # noqa: E731
    ok = [k for k in have if score(k) >= 1.0]
    if ok:
        k = ok[0]
        why = "the clean plate is textured and moving" if k == "plate" else f"the clean plate reads flat (texture {measured.get('plate', {}).get('texture')}, motion {measured.get('plate', {}).get('motion')}): {k} instead"
    else:
        k = max(have, key=score)
        why = "no candidate is above both floors: the best of them"
    return {"fill_source": k, "measured": measured, "floors": floors, "why": why}


def _tint(img, ink, amount):
    """mix `amount` of the word's ink colour into a fill picture (float HxWx3, 0..255)."""
    if not amount or ink is None:
        return img
    return img * (1.0 - float(amount)) + np.asarray(ink, np.float32).reshape(1, 1, 3) * float(amount)


def contrast_picture(mode, plate, subject, matte, letters, flat_rgb=None, word_rgb=None, outline: bool = True, fill_tint: float = FILL_TINT,
                     edge_line: bool = False, edge_phase: float = 0.0, edge_rgb=None, mask_fill: str = "footage", fill=None, edge_mode: str | None = None, edge_seed: int = 0, edge_section=None) -> np.ndarray:
    """ONE picture (float32 HxWx3, 0..255) of a contrast moment from the moved layers of a frame: plate / subject float32 HxWx3, matte (the subject's, moved with it) and letters float HxW.
    plate_in_subject: flat -> word -> the subject's silhouette filled with the PLATE; plate_in_type: flat -> subject x matte -> the word, its letters filled with the PLATE.
    A thin dark outline hugs the letters (never over the silhouette that stands in front of the word) so the word stands out from the plate / flat ground."""
    flat = np.asarray(flat_rgb if flat_rgb is not None else FLAT_RGB, np.float32).reshape(1, 1, 3)
    word = np.asarray(word_rgb if word_rgb is not None else WORD_RGB, np.float32).reshape(1, 1, 3)
    m3, L3 = np.asarray(matte, np.float32)[..., None], np.asarray(letters, np.float32)[..., None]
    plate = _tint(np.asarray(plate, np.float32), word, fill_tint)                  # the fills (windows / silhouette / letters) lean toward the word's ink = the piece palette
    sil = plate if fill is None else np.asarray(fill, np.float32)                  # what fills the SUBJECT's silhouette (plate_in_subject): the plate, or (flat plate) the footage, never a flat colour
    ink = np.asarray(edge_rgb if edge_rgb is not None else word, np.float32).reshape(1, 1, 3)

    def _edge(pic_, on_mask, hide=None, guard=True):
        """the animated edge line along the contour of `on_mask`, crisp, in the ink colour; `hide` (HxW 0..1) = where something stands over it."""
        la = edge_alpha(on_mask, edge_phase, edge_mode, edge_seed, edge_section, guard)[..., None]
        if hide is not None:
            la = la * (1.0 - np.asarray(hide, np.float32)[..., None])
        return pic_ * (1.0 - la) + ink * la
    if mode == "text_as_mask":
        fill = _tint(np.asarray(plate if mask_fill == "plate" else subject, np.float32), word, 0.0) if mask_fill != "plate" else plate
        pic = flat * (1.0 - L3) + fill * L3                                        # the letters' interiors show the footage, the ground is flat, the subject is NOT drawn on top
        return _edge(pic, L3[..., 0], guard=False) if edge_line else pic       # the letters' own outline: letters have straight stems and may fill the frame
    if mode == "window_subject_front":
        base = flat * (1.0 - L3) + plate * L3                                      # letters = windows onto the footage, the ground is flat
        pic = base * (1.0 - m3) + subject * m3                                     # the subject, cut by its own matte, stands in front of both
        if outline:
            sig = max(OUTLINE_SIGMA * min(L3.shape[:2]), 0.8)
            edge = np.clip(cv2.GaussianBlur(L3[..., 0], (0, 0), sig) * 3.0, 0.0, 1.0)[..., None] * (1.0 - L3) * (1.0 - m3)
            ink_o = np.asarray((236, 236, 232) if float(flat.mean()) < 128.0 else (24, 24, 26), np.float32).reshape(1, 1, 3)       # light edge on a dark ground, dark edge on a light one
            pic = pic * (1.0 - 0.9 * edge) + ink_o * (0.9 * edge)
        return _edge(pic, m3[..., 0]) if edge_line else pic
    if mode == "plate_in_subject":
        pic = (flat * (1.0 - L3) + word * L3) * (1.0 - m3) + sil * m3
        front = m3
    else:
        pic = (flat * (1.0 - m3) + subject * m3) * (1.0 - L3) + plate * L3
        front = np.zeros_like(m3)
    if outline:
        sig = max(OUTLINE_SIGMA * min(L3.shape[:2]), 0.8)
        halo = np.clip(cv2.GaussianBlur(L3[..., 0], (0, 0), sig) * 3.0, 0.0, 1.0)[..., None] * (1.0 - L3) * (1.0 - front)
        pic = pic * (1.0 - 0.9 * halo)
    if edge_line:
        pic = _edge(pic, m3[..., 0], hide=(L3[..., 0] if mode == "plate_in_type" else None))      # the subject's silhouette edge; under the letters in plate_in_type
    return pic


def bake(plate, subject, matte, *, move=None, pan=None, depth=None, color: bool = True, mode: str | None = None, letters=None, flat_rgb=None, word_rgb=None,
         fill_tint: float = FILL_TINT, guard: bool = True, can_relayout: bool = False,
         edge_line: bool = True, edge_rgb=None, outline: bool = True, mask_fill: str = "footage", frame0: int = 0,
         other_clip=None, fill_source: str = "auto", subject_full_strength: bool = True, edge_mode: str | None = None, edge_seed: int = 0, window=None) -> dict:
    """the independent motion of a plate shot, baked into three arrays of the CLIP's geometry.
    plate, subject: uint8 [T,H,W,3] (the plate = Clean Plate V2V output, subject = the clean clip), matte: [Tm,h,w(,C)] the Alpha-Gen matte of the subject (reference geometry, any
    length: indexed by the same fraction of the clip), depth: optional [Td,h,w(,C)] float, warped with the PLATE's transform at its own resolution (the scene geometry follows
    the plate; the subject is cut by its matte, not by depth). move / pan: shot keys (None = defaults). mode: a TYPE_MATTE_MODES name (needs `letters`, a callable j -> float HxW
    letter matte at the clip's geometry, e.g. LetterReader): the letters decide which layer shows where, see the module docstring; the returned matte is then all zero for the two
    reveal modes (nothing stands in front of the type) and matte x letters for subject_through.
    OCCLUSION GUARD (guard=True, modes in OCC_GUARDED): before baking, `occlusion_share` measures how much of the word the moved subject matte covers over the section; over the limits
    `occlusion_decision` asks for a bigger layout (can_relayout) or swaps the mode to plate_in_subject; the decision is returned in info["occlusion"] (and the caller records it in the
    piece). fill_tint: share of the word's ink mixed into the plate fills (0 = off).
    NO FLAT FILLS (2026-10-30, user: "I dont want frames inside the silhouette to go flat colour"): what fills the SUBJECT's silhouette (plate_in_subject) must be moving, textured imagery.
    `fill_measure` scores the clean plate, the shot's own footage and `other_clip` (optional uint8 [T',H,W,3], the other shot's moving clip) over the section (texture = local std inside the matte,
    motion = frame difference inside the matte); `pick_fill` takes the plate only above both floors (FILL_MIN_TEXTURE / FILL_MIN_MOTION), else the footage, else other_clip, NEVER a flat fill;
    fill_source = "plate" | "footage" | "other_clip" forces one. The choice + the measurements are returned in info["fill"] (the caller records them per moment). subject_full_strength
    (default True): in the contrast modes the subject is drawn at full strength (the colour-fit gain / offset that matches it to the plate is not applied: no darkening / desaturation).
    window (2026-11-01, render R17): a per-frame weight 0..1 (sequence of T floats, see `window_weights`) for the CONTRAST modes: the flat-ground picture is drawn only where the weight is > 0 (blended
    over the plain plate + subject layers while it is < 1); weight 0 = the shot's real footage, the plate and subject layers and the matte untouched (the type is drawn between them by the recipe).
    None = every frame (the old behaviour).
    -> {"footage": uint8 [T,H,W,3] (the moved plate), "subject": uint8 [T,H,W,3] (colour-matched, moved), "matte": uint8 [T,H,W,1] (feathered, moved with the subject),
        "depth": warped depth | None, "info": {...}, "plate_moved", "subject_moved": the moved layers (for the archive)}. T = the plate's frame count."""
    from tools.fusion import matte_align as MA
    plate, subject = _np(plate), _np(subject)
    mv, pn = norm_move(move), norm_pan(pan)
    T = int(plate.shape[0])
    H, W = int(subject.shape[1]), int(subject.shape[2])
    ns = int(subject.shape[0])
    mt = _np(matte)
    nm = int(mt.shape[0])
    out_f = np.empty((T, H, W, 3), np.uint8)
    out_s = np.empty((T, H, W, 3), np.uint8)
    out_m = np.empty((T, H, W, 1), np.uint8)
    keep = mode in ("reveal_subject", "reveal_plate") + CONTRAST_MODES   # these overwrite the layers with ONE picture: keep the moved layers for the archive
    kept_p = np.empty((T, H, W, 3), np.uint8) if keep else None
    kept_s = np.empty((T, H, W, 3), np.uint8) if keep else None
    # matte at the clip's geometry (before the move), frame j of the plate <-> the same fraction of the subject / matte clips
    base_m = [soften(ref_to_clip(mt[MA.frame_index(j, nm, T)], (H, W)), mv["feather"]) for j in range(T)]
    sub_idx = [MA.frame_index(j, ns, T) for j in range(T)]

    def plate_at(j):
        p = plate[j]
        return p if p.shape[:2] == (H, W) else cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA)
    fit = {"gain": [1.0] * 3, "offset": [0.0] * 3, "bg": 0.0, "ok": False}
    if color:
        sel = sorted({int(round(i * (T - 1) / max(FIT_FRAMES - 1, 1))) for i in range(FIT_FRAMES)})
        fit = fit_color([subject[sub_idx[j]] for j in sel], [plate_at(j) for j in sel], [base_m[j] for j in sel], n=len(sel))
    gain = np.asarray(fit["gain"], np.float32) * 1.0
    off = np.asarray(fit["offset"], np.float32) * 255.0
    occ = None
    if guard and mode in OCC_GUARDED and letters is not None:
        sample = sorted({int(round(i * (T - 1) / 11.0)) for i in range(12)}) if T > 1 else [0]
        lf, mf = [], []
        for j in sample:
            t_ = j / float(T - 1) if T > 1 else 0.0
            mx_, my_, ms_ = move_at(mv, t_)
            lf.append(np.asarray(letters(j), np.float32))
            mf.append(cv2.warpAffine(base_m[j], affine((H, W), mx_, my_, ms_, mv["anchor"]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        occ = occlusion_decision(occlusion_share(lf, mf), mode, can_relayout)
        mode = occ["mode"]
        keep = mode in ("reveal_subject", "reveal_plate") + CONTRAST_MODES
        if keep and kept_p is None:
            kept_p = np.empty((T, H, W, 3), np.uint8)
            kept_s = np.empty((T, H, W, 3), np.uint8)
    fill_info = None
    other_T = None
    if mode == "plate_in_subject":                                              # the silhouette is filled: measure the candidates over the section, never a flat fill
        sub_T = subject[np.asarray(sub_idx)]
        mw_ = (lambda j: base_m[j]) if window is None else (lambda j: base_m[j] * (1.0 if float(window[j]) > 0.0 else 0.0))        # only the frames that really show the silhouette fill
        cand = {"plate": fill_measure(plate, mw_), "footage": fill_measure(sub_T, mw_)}
        if other_clip is not None:
            oc = _np(other_clip)
            other_T = oc[np.asarray([MA.frame_index(j, int(oc.shape[0]), T) for j in range(T)])]
            cand["other_clip"] = fill_measure(other_T, mw_)
        fill_info = pick_fill(cand, force=(fill_source if fill_source in FILL_SOURCES else None))
        fill_info["mode"] = mode
    shifts = []
    for j in range(T):
        t = j / float(T - 1) if T > 1 else 0.0
        mx, my, ms = move_at(mv, t)
        px, py, ps = pan_at(pn, t)
        M_sub = affine((H, W), mx, my, ms, mv["anchor"])
        M_plate = affine((H, W), px, py, ps, (0.5, 0.5))
        out_f[j] = cv2.warpAffine(plate_at(j), M_plate, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        s = subject[sub_idx[j]][..., :3].astype(np.float32)
        if fit["ok"] and not (subject_full_strength and mode in CONTRAST_MODES):
            s = s * gain + off
        s = np.clip(np.rint(s), 0, 255).astype(np.uint8)
        out_s[j] = cv2.warpAffine(s, M_sub, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mm = cv2.warpAffine(base_m[j], M_sub, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)      # the SAME matrix as the subject
        if mode in TYPE_MATTE_MODES:
            if letters is None:
                raise ValueError(f"bake: type_matte {mode} needs the letter matte (letters=)")
            L = np.asarray(letters(j), np.float32)
            if keep:
                kept_p[j], kept_s[j] = out_f[j], out_s[j]
            pf, sf = out_f[j].astype(np.float32), out_s[j].astype(np.float32)
            comp = pf * (1.0 - mm[..., None]) + sf * mm[..., None]                    # C = the moved subject over the moved plate
            wj_ = 1.0 if (window is None or mode not in CONTRAST_MODES) else min(max(float(window[j]), 0.0), 1.0)
            if mode in CONTRAST_MODES and wj_ <= 0.0:
                pass                                                                  # outside the section: the shot's real footage (plate + subject layers + matte as they are, the type goes between them)
            elif mode in CONTRAST_MODES:
                fl_ = None
                if fill_info is not None and fill_info["fill_source"] == "footage":
                    fl_ = sf
                elif fill_info is not None and fill_info["fill_source"] == "other_clip":
                    o_ = other_T[j]
                    fl_ = (o_ if o_.shape[:2] == (H, W) else cv2.resize(o_, (W, H), interpolation=cv2.INTER_AREA))[..., :3].astype(np.float32)
                pic_c = contrast_picture(mode, pf, sf, mm, L, flat_rgb, word_rgb, outline=outline, fill_tint=fill_tint,
                                         edge_line=edge_line, edge_phase=frame0 + j, edge_rgb=edge_rgb, mask_fill=mask_fill, fill=fl_, edge_mode=edge_mode, edge_seed=edge_seed, edge_section=(frame0, frame0 + T - 1))
                if wj_ < 1.0:                                                         # the ease in / out of the section: dissolve between the real footage (C) and the flat-ground picture
                    pic_c = comp * (1.0 - wj_) + pic_c * wj_
                out_f[j] = np.clip(np.rint(pic_c), 0, 255).astype(np.uint8)
                out_s[j] = out_f[j]                                                   # one picture: nothing stands in front of the type, the matte is empty
                mm = np.zeros_like(mm)
            elif mode == "subject_through":
                mm = mm * L                                                           # the subject layer shows (in front of the type) only where it overlaps the letters
            else:
                pic = pf * (1.0 - L[..., None]) + comp * L[..., None] if mode == "reveal_subject" else comp * (1.0 - L[..., None]) + pf * L[..., None]
                out_f[j] = np.clip(np.rint(pic), 0, 255).astype(np.uint8)
                out_s[j] = out_f[j]                                                   # nothing stands in front of the type: the subject layer IS the picture, its matte is empty
                mm = np.zeros_like(mm)
        out_m[j, ..., 0] = np.clip(np.rint(mm * 255.0), 0, 255).astype(np.uint8)
        shifts.append((round(mx * W, 2), round(my * H, 2), round(ms, 4), round(px * W, 2), round(py * H, 2), round(ps, 4)))
    dep = None
    if depth is not None:
        d = _np(depth)
        dep = np.empty_like(d, dtype=np.float32)
        nd = int(d.shape[0])
        for j in range(nd):
            t = j / float(nd - 1) if nd > 1 else 0.0
            px, py, ps = pan_at(pn, t)
            dep[j] = cv2.warpAffine(np.ascontiguousarray(d[j], dtype=np.float32), affine(d.shape[1:3], px, py, ps, (0.5, 0.5)), (d.shape[2], d.shape[1]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    info = {"move": mv, "pan": pn, "color": fit, "type_matte": mode, "frames": T, "first": shifts[0] if shifts else None, "last": shifts[-1] if shifts else None,
            "size": [W, H], "occlusion": occ, "fill_tint": fill_tint if mode in CONTRAST_MODES else 0.0,
            "fill": fill_info, "subject_strength": "full" if (subject_full_strength and mode in CONTRAST_MODES) else "colour-fitted",
            "window_frames": (None if (window is None or mode not in CONTRAST_MODES) else int(sum(1 for j in range(T) if float(window[j]) > 0.0)))}
    return {"footage": out_f, "subject": out_s, "matte": out_m, "depth": dep, "info": info,
            "plate_moved": kept_p if keep else out_f, "subject_moved": kept_s if keep else out_s}      # = the moved layers before a reveal mode merged them (the same arrays otherwise)


# --------------------------------------------------------------------------------------------------------------- flat-ground SECTIONS (render R17)
# A flat-ground contrast mode is a SECTION of one or two word holds (effects_menu.py), never a whole shot: R17 ran plate_in_subject over 109 + 83 frames and the final was a flat grey ground
# with a ghost drummer. Outside the section the shot shows its real footage; the sections of the whole piece add up to at most FLAT_SHARE_MAX of its duration.
FLAT_SHARE_MAX = 0.40            # all flat-ground sections together: at most this share of the piece's duration
FLAT_SECTION_MAX_S = 4.0         # one shot's section: at most this many seconds (effects_menu: 2-4 s)
FLAT_EASE_S = 0.2                # the dissolve in before the hit and out after the release (outside the hold, inside the shot)
FLAT_MERGE_S = 0.3               # two holds closer than this are one section
FLAT_FALLBACK_S = 2.0            # no word timing at all: a section this long, centred in the shot


def _layout_of(piece: dict) -> dict:
    try:
        from tools.fusion import recipe_eval as RE
        return RE.piece_layout(piece)[0]
    except Exception:       # noqa: BLE001
        return (piece or {}).get("layout") or {}


def _shot_span(piece: dict, sid: str):
    """(start_s, length_s) of a shot on the piece timeline, or None."""
    L = _layout_of(piece).get(sid)
    if not L:
        return None
    fps = float((piece or {}).get("fps", 24) or 24)
    return float(L["start"]) / fps, float(L["frames"]) / fps


def _piece_seconds(piece: dict) -> float:
    lay = _layout_of(piece)
    fps = float((piece or {}).get("fps", 24) or 24)
    end = max([float(L["start"]) + float(L["frames"]) for L in lay.values()] or [0.0]) / fps
    return end or float(((piece or {}).get("type") or {}).get("duration") or 0.0)


def _word_holds(piece: dict, type_dir=None) -> list:
    """[(hit_s, release_s)] of every word (piece seconds): the type pass manifest / plan holds first, else the words' own hit / release (t0 / t1, start / end) fields."""
    try:
        from tools.fusion import recipe_eval as RE
        h = RE._piece_word_holds(piece, type_dir)
        if h:
            return [(float(a), float(b)) for a, b in h]
    except Exception:       # noqa: BLE001
        pass
    out = []
    for w in ((piece or {}).get("type") or {}).get("words") or []:
        if not isinstance(w, dict):
            continue
        for ka, kb in (("hit", "release"), ("t0", "t1"), ("start", "end")):
            try:
                a, b = float(w[ka]), float(w[kb])
            except (KeyError, TypeError, ValueError):
                continue
            if b > a:
                out.append((a, b))
                break
    return out


def shot_sections(piece: dict, sid: str, type_dir=None) -> list:
    """the word-hold window(s) of shot `sid` that a flat-ground mode may cover, piece seconds [(a, b), ..] clipped to the shot, merged, at most FLAT_SECTION_MAX_S in all. A shot none of
    whose words hold in it gets none; a piece without any word timing gets one FLAT_FALLBACK_S window centred in the shot."""
    sp = _shot_span(piece, sid)
    if sp is None:
        return []
    s0, ln = sp
    holds = _word_holds(piece, type_dir)
    wins = []
    for a, b in sorted(holds):
        a, b = max(a, s0), min(b, s0 + ln)
        if b - a > 0.05:
            if wins and a - wins[-1][1] < FLAT_MERGE_S:
                wins[-1][1] = max(wins[-1][1], b)
            else:
                wins.append([a, b])
    if not holds:
        d = min(FLAT_FALLBACK_S, ln * 0.6)
        wins = [[s0 + (ln - d) / 2.0, s0 + (ln + d) / 2.0]]
    out, left = [], FLAT_SECTION_MAX_S
    for a, b in wins:
        if left <= 0.05:
            break
        b = min(b, a + left)
        out.append((round(a, 3), round(b, 3)))
        left -= b - a
    return out


def contrast_sections(piece: dict, type_dir=None) -> dict:
    """{sid: [(a, b), ..]} piece seconds of every shot whose (solid-letters) type-matte mode is a flat-ground CONTRAST mode: the shot sections above, in layout order, trimmed so that all of
    them together stay within FLAT_SHARE_MAX of the piece's duration (a shot that finds no budget left is absent = plain footage)."""
    from tools.fusion import letter_fill as _LF
    budget = FLAT_SHARE_MAX * _piece_seconds(piece)
    out = {}
    lay = _layout_of(piece)
    ids = list(lay) if lay else [s.get("id") for s in (piece or {}).get("shots") or []]
    for sid in ids:
        m = _LF.solid_mode(piece, type_matte_of(piece, sid))
        if m not in CONTRAST_MODES or budget <= 0.05:
            continue
        keep = []
        for a, b in shot_sections(piece, sid, type_dir):
            b = min(b, a + budget)
            if b - a > 0.05:
                keep.append((a, round(b, 3)))
                budget -= b - a
        if keep:
            out[sid] = keep
    return out


def window_weights(sections, shot_start_s: float, shot_len_s: float, T: int, ease_s: float = FLAT_EASE_S) -> list:
    """per frame j of a T-frame clip of the shot: 1.0 inside a section, a linear ramp over `ease_s` before its start and after its end, else 0.0 (the clip spans the shot's length)."""
    w = []
    for j in range(int(T)):
        t = shot_start_s + (j + 0.5) / max(int(T), 1) * shot_len_s
        v = 0.0
        for a, b in sections or []:
            if a <= t <= b:
                v = 1.0
            elif a - ease_s < t < a:
                v = max(v, (t - (a - ease_s)) / ease_s)
            elif b < t < b + ease_s:
                v = max(v, 1.0 - (t - b) / ease_s)
        w.append(round(min(max(v, 0.0), 1.0), 4))
    return w


def _rgb255(c):
    v = [float(x) for x in list(c)[:3]]
    return [x * 255.0 for x in v] if max(v) <= 1.0 else v


def _luma01(c) -> float:
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0


def _chroma01(c) -> float:
    return (max(c[:3]) - min(c[:3])) / 255.0


def shot_colours(frame, k: int = 5) -> list:
    """the dominant colours of a picture (uint8 HxWx3), most frequent first, rgb 0..255: k-means on a small copy, fixed seed (deterministic)."""
    f = np.asarray(frame)[..., :3]
    h, w = f.shape[:2]
    sc = 40.0 / max(h, w)
    small = cv2.resize(np.ascontiguousarray(f), (max(2, int(round(w * sc))), max(2, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    k = max(1, min(int(k), len(data)))
    cv2.setRNGSeed(0)
    _, lab, cen = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0), 1, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.reshape(-1), minlength=k)
    return [[float(x) for x in cen[i]] for i in np.argsort(-cnt)]


GREY_CHROMA, RICH_CHROMA = 0.12, 0.20                  # "low chroma" (a grey) / a "richer" colour: a grey candidate is dropped when any candidate is richer
GROUND_CR_MIN, GROUND_CR_GOOD = 3.0, 4.5               # WCAG contrast of the flat ground against the subject's mean colour and against the word ink: the floor / what is aimed for
GROUND_DL_MIN = 0.35                                   # and its OKLab lightness distance from the subject's mean colour
INK_CR_MIN, INK_CR_GOOD = 3.0, 4.5                     # the word ink (and the contour lines drawn in it) against the ground


def _lin255(c):
    v = np.clip(np.asarray(c[:3], np.float64) / 255.0, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _rel_lum(c) -> float:
    r, g, b = _lin255(c)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(a, b) -> float:
    """WCAG contrast ratio (1..21) of two rgb 0..255 colours."""
    la, lb = _rel_lum(a), _rel_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def oklab_l(c) -> float:
    """OKLab lightness (0..1) of an rgb 0..255 colour."""
    r, g, b = _lin255(c)
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return float(0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_)


def _mix(c, to, t):
    return [float(x) * (1.0 - t) + to * t for x in c[:3]]


def _push(c, ok):
    """the colour `c` (rgb 0..255) moved toward black or white (the nearer end that satisfies `ok`, the smaller step wins) until `ok(colour)` holds. -> (colour, t) or (None, 1)."""
    best = None
    for to in (0.0, 255.0):
        for t in np.linspace(0.0, 1.0, 41):
            v = _mix(c, to, float(t))
            if ok(v):
                if best is None or t < best[1]:
                    best = (v, float(t))
                break
    return best if best is not None else (None, 1.0)


def pick_ink(ground, word_rgb, cands=()) -> dict:
    """the word ink on a flat ground (the contour lines drawn along the letters use it too): the piece's own ink when it reaches INK_CR_GOOD on the ground, else the palette colour
    (cands, rgb 0..255) with the best contrast if it reaches INK_CR_MIN, else the most chromatic of them moved to the far end of lightness (hue kept) until it does. -> {"rgb", "cr", "pushed"}."""
    g = _rgb255(ground)
    w = _rgb255(word_rgb if word_rgb is not None else WORD_RGB)
    pool = [w] + [_rgb255(c) for c in cands or [] if c is not None]
    if contrast_ratio(w, g) >= INK_CR_GOOD:
        return {"rgb": [int(round(x)) for x in w], "cr": round(contrast_ratio(w, g), 2), "pushed": False}
    good = [c for c in pool if contrast_ratio(c, g) >= INK_CR_MIN]
    if good:
        c = max(good, key=lambda c: min(contrast_ratio(c, g), INK_CR_GOOD) + 1.5 * _chroma01(c))
        return {"rgb": [int(round(x)) for x in c], "cr": round(contrast_ratio(c, g), 2), "pushed": False}
    for need in (INK_CR_GOOD, INK_CR_MIN):
        for c in sorted(pool, key=lambda c: -_chroma01(c)):
            v, _ = _push(c, lambda x: contrast_ratio(x, g) >= need)
            if v is not None:
                return {"rgb": [int(round(x)) for x in v], "cr": round(contrast_ratio(v, g), 2), "pushed": True}
    far = [245.0, 243.0, 238.0] if _rel_lum(g) < 0.18 else [14.0, 14.0, 16.0]
    return {"rgb": [int(x) for x in far], "cr": round(contrast_ratio(far, g), 2), "pushed": True}


def pick_ground(cands, word_rgb, fill_rgb=None, ink_cands=()) -> dict:
    """the flat ground of a contrast section (user 2026-11-01 on R17: "it needs to be a contrasting colour or shade - not a close shade"). It must stand clearly apart from what fills the
    subject's silhouette (`fill_rgb`, the mean colour: WCAG >= GROUND_CR_MIN, aimed at GROUND_CR_GOOD, and >= GROUND_DL_MIN OKLab lightness away) and from the word's ink (>= INK_CR_MIN).
    Of the candidate colours (rgb 0..255, the user's palette first, then the shot's) the one that meets this with the least change wins (more contrast + more chroma break ties); a grey is
    dropped when any candidate is richer; a candidate that does not reach it is moved to the far end of lightness (hue kept). The ink is then picked on that ground (`pick_ink`: the
    piece's ink if it reads, else a palette colour, else a palette hue pushed light / dark), so a light ground under a dark photo gets a dark ink. -> {"rgb", "ink", "why", "cr_fill", "dl_fill", "cr_ink", "pushed"}."""
    word = _rgb255(word_rgb if word_rgb is not None else WORD_RGB)
    fill = _rgb255(fill_rgb) if fill_rgb is not None else None
    cs = [_rgb255(c) for c in cands or [] if c is not None]
    if not cs:
        cs = [list(FLAT_RGB)]
    if any(_chroma01(c) >= RICH_CHROMA for c in cs):
        cs = [c for c in cs if _chroma01(c) >= GREY_CHROMA] or cs

    def ok(c):
        return fill is None or (contrast_ratio(c, fill) >= GROUND_CR_MIN and abs(oklab_l(c) - oklab_l(fill)) >= GROUND_DL_MIN)

    def score(c):
        if fill is None:
            return 1.5 * _chroma01(c)
        return min(contrast_ratio(c, fill), 6.0) + 3.0 * min(abs(oklab_l(c) - oklab_l(fill)), 0.6) + 1.5 * _chroma01(c)
    opts = []
    for c in cs:
        if ok(c):
            opts.append((c, 0.0))
        else:
            v, t = _push(c, ok)
            if v is not None:
                opts.append((v, t))
    pushed = False
    if opts:
        best, t = max(opts, key=lambda o: (o[1] == 0.0, score(o[0])))
        pushed = t > 0.0
        why = "contrasts with the subject and the word ink" if not pushed else "no palette colour clearly apart from the subject: the best hue moved to the far end of lightness"
    else:                                                  # (cannot happen for a real fill colour; keep the most distant end)
        best = [0.0, 0.0, 0.0] if _rel_lum(fill) > 0.18 else [255.0, 255.0, 255.0]
        pushed, why = True, "no colour reaches the contrast: black / white"
    ink = pick_ink(best, word, ink_cands or cs)
    return {"rgb": [int(round(x)) for x in best], "ink": ink["rgb"], "why": why, "pushed": pushed or ink["pushed"],
            "cr_fill": round(contrast_ratio(best, fill), 2) if fill is not None else None,
            "dl_fill": round(abs(oklab_l(best) - oklab_l(fill)), 3) if fill is not None else None, "cr_ink": ink["cr"]}


def flat_ground(piece: dict, sid: str, word_rgb, plate_frame=None, silhouette_rgb=None) -> dict:
    """the flat ground colour (+ the word ink on it) of a plate_in_subject / plate_in_type section. The USER's palette (user_style.palette, direction text) first when it has one, else the
    shot's own colours (the shot key `palette`, the dominant colours of its plate, the piece palette) - picked by `pick_ground` against the subject silhouette's mean colour (the plate's
    mean when the silhouette cannot be measured) and the word ink. -> {"rgb", "ink", "source", "why", ...}."""
    cands, src = [], "shot"
    try:
        from tools.fusion import user_style as US
        up = US.parse_direction(piece or {}).get("palette")
    except Exception:       # noqa: BLE001
        up = None
    pal = ((piece or {}).get("type") or {}).get("palette") or {}
    if up:
        cands, src = [list(c) for c in up], "user_palette"
    else:
        sp = shot_of(piece, sid).get("palette")
        if isinstance(sp, (list, tuple)):
            cands += [list(c) for c in sp if isinstance(c, (list, tuple)) and len(c) >= 3]
        if plate_frame is not None:
            try:
                cands += shot_colours(plate_frame)
            except Exception:       # noqa: BLE001
                pass
        cands += [list(c) for c in list(pal.get("accents") or []) + [pal.get("bg")] if c]
    fill = silhouette_rgb
    if fill is None and plate_frame is not None:
        fill = [float(x) for x in np.asarray(plate_frame)[..., :3].reshape(-1, 3).mean(axis=0)]
    r = pick_ground(cands, word_rgb, fill, ink_cands=([list(c) for c in up] if up else []) + [list(c) for c in list(pal.get("accents") or []) if c])
    r["source"] = src
    return r


def _silhouette_rgb(subject, matte):
    """mean colour of the subject's pixels (middle frame, matte > 0.5), rgb 0..255; None when it cannot be measured."""
    try:
        sub, mt = _np(subject), _np(matte)
        T = int(sub.shape[0])
        j = T // 2
        from tools.fusion import matte_align as MA
        m = ref_to_clip(mt[MA.frame_index(j, int(mt.shape[0]), T)], tuple(int(x) for x in sub.shape[1:3])) > 0.5
        if int(m.sum()) < 50:
            return None
        return [float(x) for x in sub[j][..., :3][m].mean(axis=0)]
    except Exception:       # noqa: BLE001
        return None


def bake_clip(piece: dict, clip: dict, matte, depth=None, type_dir=None):
    """a plate shot's clip dict (nodes_fusion/layers.MographLayerGather: `u8` = the plate, `subject` = the clean clip, torch uint8 [T,H,W,3]) -> (clip2, matte2, depth2, info):
    the same dict with the moved plate / subject, the matte (torch uint8 [T,H,W,1]) that follows the subject and the plate-panned depth. A clip that is not a plate shot, or a
    plate shot without a matte, comes back unchanged (info {"skipped": why}). type_dir: the type layers directory (`type_t["dir_layers"]`): the letter matte of a type-matte shot.
    A type-matte mode without letters on disk is skipped the same way (the shot stays a plain plate shot)."""
    sid = clip.get("sid")
    if not clip.get("plate") or clip.get("subject") is None:
        return clip, matte, depth, {"skipped": "not a plate shot"}
    if matte is None:
        return clip, matte, depth, {"skipped": "no matte arrived (the plate route needs the Alpha-Gen matte of the clean clip)"}
    import torch
    mv, pn = shot_moves(piece, sid)
    from tools.fusion import letter_fill as _LF                       # 2026-11-01: solid letters by default (plate / footage INSIDE the letters only when the user asks)
    mode = _LF.solid_mode(piece, type_matte_of(piece, sid))
    mode = mode if mode in TYPE_MATTE_MODES else None
    letters = None
    if mode:
        from tools.fusion import recipe_eval as RE
        letters = LetterReader(type_dir, int(RE.piece_layout(piece)[0][sid]["start"]), tuple(int(x) for x in clip["subject"].shape[1:3]))
        if not letters.files:
            return clip, matte, depth, {"skipped": f"type_matte {mode}: no letter matte frames in {type_dir}"}
    pal = ((piece.get("type") or {}).get("palette")) or {}              # contrast moments: the flat ground is the palette's first accent, the word its fg
    acc = pal.get("accents") or []
    ground_info = None
    ink_rgb = pal.get("fg")
    wj = None
    if mode in CONTRAST_MODES:
        # FLAT-GROUND SECTIONS (render R17): the flat ground covers the word-hold window(s) of this shot only (<= 40 % of the piece in all), the rest of the shot is its real footage
        secs = contrast_sections(piece, type_dir).get(sid)
        sp_ = _shot_span(piece, sid)
        T_ = int(clip["u8"].shape[0])
        wj = window_weights(secs, sp_[0], sp_[1], T_) if (secs and sp_) else [0.0] * T_
        piece.setdefault("plan", {}).setdefault("flat_sections", {})[sid] = {"mode": mode, "sections": [list(x) for x in (secs or [])], "frames": int(sum(1 for x in wj if x > 0))}
    if mode in ("plate_in_subject", "plate_in_type") and any(x > 0 for x in (wj or [1])):
        ground_info = flat_ground(piece, sid, pal.get("fg"), plate_frame=_np(clip["u8"])[int(clip["u8"].shape[0]) // 2], silhouette_rgb=_silhouette_rgb(clip["subject"], matte))
        acc = [ground_info["rgb"]]
        ink_rgb = ground_info["ink"]                                    # the word ink that reads on that ground (the contour lines drawn along the letters use it too)
        piece["plan"]["flat_sections"][sid]["ground"] = ground_info
    if mode in ("window_subject_front", "text_as_mask"):                # the darkest (or, shot key flat = "light", the lightest) colour of the palette
        try:
            from tools.fusion import user_style as US
            _up = [_rgb255(c) for c in (US.parse_direction(piece).get("palette") or [])]          # the user's palette, when there is one, supplies the ground
        except Exception:       # noqa: BLE001
            _up = []
        cols = _up or [c for c in [pal.get("bg"), pal.get("fg")] + list(acc) if c]
        lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]       # noqa: E731
        acc = [(max if str(shot_of(piece, sid).get("flat") or "").lower() == "light" else min)(cols, key=lum)] if cols else []
    ft = shot_of(piece, sid).get("fill_tint")
    r = bake(clip["u8"], clip["subject"], matte, move=mv, pan=pn, depth=depth, mode=mode, letters=letters,
             flat_rgb=(acc[0] if acc else None), word_rgb=ink_rgb, fill_tint=(FILL_TINT if ft is None else float(ft)), window=wj)
    occ = r["info"].get("occlusion")
    if occ:                                                             # the guard's decision is part of the piece (plan.occlusion_guard.<shot>)
        piece.setdefault("plan", {}).setdefault("occlusion_guard", {})[sid] = occ
        mode = occ["mode"]
    if r["info"].get("fill"):                                           # the fill of the subject's silhouette (plate | footage | other_clip) + the measured texture / motion, per shot
        piece.setdefault("plan", {}).setdefault("fill_source", {})[sid] = r["info"]["fill"]
    c2 = dict(clip, u8=torch.from_numpy(r["footage"]), subject=torch.from_numpy(r["subject"]), textbg=False, plate_baked=True, type_matte=mode)
    partial = wj is not None and any(x <= 0.0 for x in wj)         # a flat-ground section inside a shot: the frames outside it keep the subject layer
    if mode in ("reveal_subject", "reveal_plate") + CONTRAST_MODES and not (mode in CONTRAST_MODES and partial):
        c2["subject"] = c2["u8"]                                    # one picture (no second copy in memory)
    elif partial:
        c2["subject"] = torch.from_numpy(r["subject"])
    # every component of the plate route rides on the clip so MographArchive can save it (<stem>_sK_plate / _plate_moved / _subject_moved / _matte_moved; uint8 [T,H,W,C] CPU tensors)
    pm = c2["u8"] if r["plate_moved"] is r["footage"] else torch.from_numpy(r["plate_moved"])
    sm = c2["subject"] if r["subject_moved"] is r["subject"] else torch.from_numpy(r["subject_moved"])
    mt = torch.from_numpy(r["matte"])
    c2["plate_layers"] = {"plate_raw": clip["u8"], "clean": clip["subject"], "plate_moved": pm, "subject_moved": sm, "matte_moved": mt,
                          "type_matte": mode, "move": r["info"]["move"], "pan": r["info"]["pan"], "color": r["info"]["color"]}
    return c2, mt, (torch.from_numpy(r["depth"]) if r["depth"] is not None else depth), r["info"]


# --------------------------------------------------------------------------------------------------------------- the empty-scene prompt
_PREP_W = {"across", "through", "along", "on", "onto", "in", "into", "toward", "towards", "over", "past", "under", "down", "up", "at", "from", "to", "beside", "by", "around",
           "inside", "outside", "against", "behind", "between", "near", "beneath", "above", "with", "of"}
_DET_W = {"a", "an", "the", "his", "her", "its", "their", "two", "three", "some", "this", "that"}
_NOT_VERB = {"glass", "grass", "dress", "moss", "across", "cross", "mass", "chaos", "gas", "lens", "bus", "this", "its", "his", "has", "was", "is", "as", "us", "mess", "cliss",
             "waist", "boots", "jeans", "pants", "shorts", "sunglasses", "goggles", "gloves", "clothes", "lights", "streets"}
_ATTIRE = re.compile(r"\b(vest|jacket|coat|dress|hat|shirt|hoodie|top|suit|clothes|clothing|skirt|scarf|cap|helmet|boots|jeans|gloves|uniform|robe|sweater|shorts|pants|hair|beard)\b", re.I)
_CAMERA = re.compile(r"\b(dolly|pan(?:s|ning)?|tilt|tracking|track|zoom|push(?:es)? in|pull(?:s)? back|crane|orbit|handheld|drone|camera|slow(?:ly)?|fast|static|steadicam|rack focus|"
                     r"angle|close[- ]?up|wide|medium|profile|top[- ]down|cinematic|35 ?mm|film|lens|bokeh)\b", re.I)
_PLATE_TEMPLATE = ("An empty clean plate of the exact same location, as in the source video: {scene}. No {subject}, no people, no figures, no body parts anywhere in the frame. "
                   "No wake, reflection or cast shadow of {subject}. Static photorealistic footage, natural light, high detail.")


def parse_clause(first_clause: str) -> dict:
    """split a shot prompt's first clause ('A woman in a red vest paddles a kayak across a calm lake') into {"subject": 'woman in a red vest', "objects": ['kayak'], "where": 'across a calm lake'}:
    the noun phrase before the first verb (a prepositional phrase of clothing stays with the subject, any other one is the place), the object after the verb up to the next
    preposition, the rest is the place. Heuristic (no model): the shot's `plate_subject` / `plate_prompt` override it."""
    c = re.sub(r"^(?:a|an|the)\s+", "", str(first_clause or "").strip(), flags=re.I)
    w = c.split()
    v = next((i for i in range(1, len(w)) if w[i - 1].lower() not in _DET_W and w[i].lower().strip(",.") not in _NOT_VERB and w[i].lower().strip(",.") not in _PREP_W
              and re.fullmatch(r"[a-z]{3,}(?:s|ing)", w[i].lower().strip(",."))), None)
    if v is None:
        return {"subject": " ".join(w[:4]) or "subject", "objects": [], "where": ""}
    p = next((i for i in range(1, v) if w[i].lower() in _PREP_W), v)
    subj, pp = w[:p], w[p:v]
    where = []
    if pp:
        (subj.extend if _ATTIRE.search(" ".join(pp)) else where.extend)(pp)
    rest = w[v + 1:]
    k = next((i for i, x in enumerate(rest) if x.lower() in _PREP_W), len(rest))
    obj = re.sub(r"^(?:a|an|the)\s+", "", " ".join(rest[:k][:4]), flags=re.I)
    where.extend(rest[k:])
    return {"subject": " ".join(subj), "objects": [obj] if obj else [], "where": " ".join(where)}


def subject_phrase(first_clause: str) -> str:
    """what to take off the plate: the subject and the objects it handles ('woman in a red vest, kayak')."""
    r = parse_clause(first_clause)
    return ", ".join([r["subject"]] + r["objects"])


def empty_scene_prompt(piece: dict, sid: str) -> str:
    """the Clean Plate prompt of a shot (CLEAN_PLATE.md section 5.3): the template with the shot's scene (the place named in the first clause + the prompt's other clauses without
    the camera / style ones) and the subject to remove. `plate_prompt` on the shot replaces all of it, `plate_subject` replaces the heuristic subject."""
    sh = shot_of(piece, sid)
    if str(sh.get("plate_prompt") or "").strip():
        return str(sh["plate_prompt"]).strip()
    try:
        from tools.app import layered_shots as LS
        base = LS.clean_prompt(str(sh.get("prompt", "")))
    except Exception:       # noqa: BLE001  (import cycle / missing optional deps in a bare test)
        base = str(sh.get("prompt", ""))
    from tools.fusion import angle_still as AS
    cl = AS._clauses(base)
    pc = parse_clause(cl[0] if cl else "")
    segs, cur = [], []                               # the place, one prepositional phrase at a time; 'toward the camera' is camera, not place
    for w_ in pc["where"].split():
        if w_.lower() in _PREP_W and cur:
            segs.append(" ".join(cur))
            cur = []
        cur.append(w_)
    if cur:
        segs.append(" ".join(cur))
    where = " ".join(x for x in segs if not _CAMERA.search(x))
    scene = ", ".join(x for x in [where] + [c for c in cl[1:] if not _CAMERA.search(c) and not AS._STYLE_RE.search(c)] if x)[:240].strip(" ,")
    subj = str(sh.get("plate_subject") or "").strip() or ", ".join([pc["subject"]] + pc["objects"])
    return _PLATE_TEMPLATE.format(scene=scene or "the same place without anyone in it", subject=subj.replace(", ", " or no "))
