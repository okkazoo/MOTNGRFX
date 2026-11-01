"""depth_pivot: the depth "pivot chunk" image transition as a reusable, seeded, parameterised module (17_depth-dolly v013, approved look).

Image A is a depth scene of chunky camera-facing cards (ONE cut-out card for the subject + `chunks` background cards, textured from the
shot, the subject rebuilt behind with an LDI back layer). One rigid camera PIVOTS round the middle of A's volume (A turns away by `out_angle`),
image B swings in over the top turned `in_angle` and lands flat. The camera does all the motion (quintic ease, no spring), each card is
smeared along its own screen motion (continuous blur). Frame 0 of the result is A exactly, the settled last frame is B exactly.

    from mograph.modules.transitions.depth_pivot import depth_pivot, depth_pivot_render
    frames = depth_pivot(a, matte_a, b, matte_b)                                   # uint8 RGB (N, H, W, 3), the approved look
    res = depth_pivot_render(a, matte_a, b, matte_b, depth_a, depth_b, {"chunks": 100, "seed": 3})
    res.frames, res.params (the FULL resolved dict, JSON-able), res.info (size, frame count, video used, card counts ...)
    depth_pivot(a, matte_a, b, matte_b, params=res.params)                         # re-makes the very same result

Inputs: `a` / `b` = one RGB image (H, W, 3) or a clip (n, H, W, 3), uint8 or float 0..1. `matte_*` = the subject matte(s), (H, W) or (n, H, W), 0..1
(Alpha-Gen output), None = no subject. `depth_*` = optional depth map (H, W) with BRIGHT = NEAR (any range; percentile-normalised like the engine
does); None = Depth-Anything-V2-Small on CPU on the FIRST frame (the geometry is ONE depth per clip).

LIVE VIDEO IS THE DEFAULT (`video="auto"`): when either input has more than one frame, every card is re-textured each output frame from the
current clip frame (image A plays from output frame 0, image B from the output frame it first becomes visible; a finished clip holds its last
frame) and the subject card's alpha is that frame's matte. A single image is a still. `video="on"` / `"off"` force it.

Density: ONE control, `chunks` = background cards per image (default 32) + the subject as one card. The card size scales with it so that the
frame stays covered: engine max_chunks = round(chunks * 50 / 32), card side = sqrt(W*H / max_chunks) x U(size_range). chunks=32 is v009's density
(50 reference, 32 background cards, size 1.5-4 = ~124-419 px at 432x762); more chunks -> smaller cards.

The engine is `tools/fusion/depth_orbit.py` (PivotScene, OrbitParams) + `tools/fusion/depth_dolly.py` (depth, layers); this module is the adapter:
it does not edit them. Registered as the hidden transition module `depth_pivot` (not in the type engine's pool: it works on images, not strands).
"""
from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from mograph.modules.core import TransitionModule, register

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VERSION = 1
REF_CHUNKS, REF_MAX_CHUNKS = 32, 50          # v009's density: 32 background cards at the card scale of 50 chunks (ref = sqrt(W*H/50))


def _p(default, lo=None, hi=None, kind="float", doc="", choices=None, advanced=False):
    return dict(default=default, lo=lo, hi=hi, kind=kind, doc=doc, choices=choices, advanced=advanced)


PARAMS: dict[str, dict] = {
    "chunks": _p(REF_CHUNKS, 10, 400, "int", "background cards per image (the subject is one more card); card size scales with it"),
    "video": _p("auto", kind="choice", choices=["auto", "on", "off"],
                doc="auto = live clip textures whenever an input has more than one frame (a single image is a still); on / off force it"),
    "in_angle": _p(45.0, 10.0, 80.0, doc="deg image B is turned when it starts to enter (lands flat)"),
    "out_angle": _p(40.0, 5.0, 80.0, doc="deg image A has turned when B starts to enter"),
    "travel": _p(36, 4, 120, "int", "frames of the eased camera move"),
    "hold1": _p(12, 0, 48, "int", "frames on A before the move (A's own clip frames keep playing through them: a0 + k)"),
    "hold2": _p(24, 0, 96, "int", "frames on B after the landing"),
    "depth_ratio": _p(6.0, 2.0, 12.0, doc="far/near depth ratio of the scene (more = more parallax)"),
    "blur": _p("continuous", kind="choice", choices=["continuous", "steps", "off"], doc="motion blur: per-card smear / sub-frame steps / none"),
    "subject_card": _p(1, 0, 1, "int", "1 = the subject is ONE cut-out card, 0 = jittered subject cards"),
    "short": _p(432, 32, 1080, "int", "short side (px) of the render; the output is this size with the aspect of the input"),
    "fps": _p(24, 12, 60, "int", "frame rate (metadata for the caller; the frame count comes from the holds + travel)"),
    "seed": _p(0, 0, 2 ** 31 - 1, "int", "card jitter / size seed"),
    "size_lo": _p(1.5, 0.5, 8.0, doc="advanced: smallest card side = ref x this", advanced=True),
    "size_hi": _p(4.0, 0.5, 8.0, doc="advanced: largest card side = ref x this", advanced=True),
    "spring": _p(False, kind="bool", doc="advanced: damped-spring landing instead of the plain ease (the approved look has none)", advanced=True),
    "mode": _p("pivot", kind="choice", choices=["pivot", "push", "pullback", "fly"],
               doc="camera path: pivot = the approved look (A turns away, B swings in); push / pullback / fly = one rigid camera path through two card volumes on one axis "
                   "(17_depth-dolly v021 / v022 / v023, tools/fusion/depth_orbit.py VolumeScene)"),
    "style": _p("auto", kind="choice", choices=["auto", "chunks", "smooth", "slide"],
                doc="how the picture changes: chunks = the card mosaic (the approved pivot); smooth = depth-parallax layers, a real camera move with no mosaic "
                    "(push / pullback / fly); slide = a lateral parallax slide, the subject faster than the background; auto = chunks for the pivot, smooth for push / pullback / fly. "
                    "The pivot has no smooth form (it stays on chunks)"),
    "dir": _p("right", kind="choice", choices=["left", "right", "up", "down"], doc="slide: the camera travels this way, the picture slides the other way"),
    "aim": _p(1.0, 0.0, 1.0, doc="smooth push / fly: how far the camera heads for the subject's centre (0 = straight down the axis, 1 = into the subject)"),
    "type": _p("", kind="str", doc="advanced (v014): 'P06:WORD ON A,WORD ON B' = the piece's words as flat type planes in the scene. NOT used by shot cuts: the type is a "
                                   "compose layer between the transition's background and its subject, never part of its pixels (pivot_cut drops it)", advanced=True),
    "type_depth": _p(0.8, 0.3, 1.2, doc="advanced: type plane depth = this x the subject's median depth", advanced=True),
}
LOCKED = {"mode": "pivot", "splat": "billboard", "ease": "quintic"}   # not parameters: what the approved look IS
DEFAULTS = {k: v["default"] for k, v in PARAMS.items()}
_INTERNAL = ("module", "version", "locked", "engine")                  # keys of a resolved dict that are not inputs


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


def resolve_params(params: dict | None = None) -> dict:
    """the FULL resolved parameter dict: defaults + overrides, clamped into the ranges, plus the locked look and the derived engine arguments.
    A resolved dict (from a previous result) is accepted back and gives the same dict. Unknown keys raise."""
    given = {k: v for k, v in (params or {}).items() if k not in _INTERNAL}
    bad = [k for k in given if k not in PARAMS]
    if bad:
        raise KeyError(f"depth_pivot: unknown parameter(s) {bad}; known: {sorted(PARAMS)}")
    out = {}
    for k, spec in PARAMS.items():
        v = given.get(k, spec["default"])
        kind = spec["kind"]
        if kind == "choice":
            if v not in spec["choices"]:
                raise ValueError(f"depth_pivot: {k}={v!r}, one of {spec['choices']}")
        elif kind == "bool":
            v = bool(v)
        elif kind == "str":
            v = str(v or "")
        elif kind == "int":
            v = int(min(max(int(round(float(v))), spec["lo"]), spec["hi"]))
        else:
            v = float(min(max(float(v), spec["lo"]), spec["hi"]))
        out[k] = v
    if out["size_lo"] > out["size_hi"]:
        out["size_lo"], out["size_hi"] = out["size_hi"], out["size_lo"]
    out["style"] = resolve_style(out["mode"], out["style"])
    return {"module": "depth_pivot", "version": VERSION, **out, "locked": dict(LOCKED), "engine": engine_args(out)}


def resolve_style(mode: str, style: str = "auto") -> str:
    """the concrete style of a camera mode: auto -> chunks for the pivot, smooth for push / pullback / fly; the pivot has no smooth / slide form (cards only)."""
    if mode == "pivot":
        return "chunks"
    return "chunks" if style in (None, "", "auto") else style          # 2026-10-30: push / pullback / fly default to the chunk mosaic too (user: no single flat card rotating in frame); `smooth` only when asked for


def engine_args(p: dict) -> dict:
    """OrbitParams keyword arguments for resolved public params `p`. `chunks` -> bg_chunks (background cards) and max_chunks (card scale,
    also the hard cap on subject + background): REF_CHUNKS -> REF_MAX_CHUNKS exactly (v009)."""
    c = int(p["chunks"])
    return dict(mode=p.get("mode", LOCKED["mode"]), splat=LOCKED["splat"], max_chunks=int(round(c * REF_MAX_CHUNKS / REF_CHUNKS)), bg_chunks=c,
                size_range=[float(p["size_lo"]), float(p["size_hi"])], blur=p["blur"], subject_cards=int(p["subject_card"]),
                in_angle=float(p["in_angle"]), out_angle=float(p["out_angle"]), travel=int(p["travel"]), hold1=int(p["hold1"]),
                hold2=int(p["hold2"]), depth_ratio=float(p["depth_ratio"]), short=int(p["short"]), fps=int(p["fps"]),
                spring=bool(p["spring"]), seed=int(p["seed"]), type=p["type"], type_depth=float(p["type_depth"]),
                style=p["style"], dir=p["dir"], aim=float(p["aim"]),
                pull_rise=0.0)           # 2026-10-30 (user: "a simple straight push in or pull out"): the pullback travels straight back along the axis, no mid-way rise; push is straight already (no yaw / pitch / roll)


def n_frames(params: dict | None = None) -> int:
    p = resolve_params(params)
    return p["hold1"] + p["travel"] + p["hold2"]


# ----------------------------------------------------------------------------------------------- engine adapter
def _engine():
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from tools.fusion import depth_dolly as dd
    from tools.fusion import depth_orbit as do
    return dd, do


def _op(do, eng: dict):
    """OrbitParams from the engine kwargs. bg_chunks is mapped here if the installed engine's OrbitParams does not have the field yet."""
    kw = dict(eng)
    kw["size_range"] = tuple(kw["size_range"])
    kw["video"] = True            # set per run below; the dataclass field only gates build_scene (we build the scene ourselves)
    fields = set(getattr(do.OrbitParams, "__dataclass_fields__", {}))
    if "bg_chunks" not in fields:
        kw.pop("bg_chunks")
    return do.OrbitParams(**{k: v for k, v in kw.items() if k in fields})


def as_frames(x, name="image") -> np.ndarray:
    """one image (H,W,3|4|1|None) or a clip (n,H,W,c), uint8 or float 0..1 -> uint8 RGB (n,H,W,3)."""
    a = np.asarray(x)
    if a.ndim == 2:
        a = a[None, ..., None]
    elif a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"depth_pivot: {name} must be (H,W,3) or (n,H,W,3), got {np.asarray(x).shape}")
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def as_mattes(m, n: int, h: int, w: int) -> np.ndarray:
    """matte (H,W) | (k,H,W) | (k,H,W,1) | None, 0..1 (or uint8) -> float32 (k,H,W) at the frame size; a single matte stands for a still subject."""
    if m is None:
        return np.zeros((1, h, w), np.float32)
    a = np.asarray(m)
    if a.ndim == 4:
        a = a[..., 0]
    if a.ndim == 2:
        a = a[None]
    a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
    if a.shape[1:] != (h, w):
        a = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in a])
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def as_depth(d):
    """depth map (H,W) | (H,W,c) | (1,H,W,c), BRIGHT = NEAR -> float32 (H,W), or None."""
    if d is None:
        return None
    a = np.asarray(d)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (1, 3, 4) else a[0]
    return a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)


def _to_mesh(frames_rgb, mattes, w, h):
    """clip -> (BGR uint8 (n,h,w,3) INTER_AREA, matte float (n,h,w) INTER_LINEAR), the very resize the engine's load_video does."""
    f = np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_AREA) for x in frames_rgb])
    m = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in mattes])
    return f, m


def _layer(dd, frame_rgb, matte, depth, w, h, prm):
    """the Layer of a clip's FIRST frame (depth model on the CPU unless `depth` is given) = depth_dolly.load_layer_inputs + build_layer on arrays."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    src = "given"
    if depth is None:
        depth, src = dd.estimate_depth(bgr), "depth-anything-v2-small"
    dn = cv2.resize(dd.normalize_disparity(depth), (w, h), interpolation=cv2.INTER_AREA)
    m = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
    return dd.build_layer(img, dn, m, prm), src


@dataclass
class PivotResult:
    frames: np.ndarray                       # uint8 RGB (N, H, W, 3)
    params: dict                             # the full resolved params: pass it back as `params=` to re-make this result
    info: dict = field(default_factory=dict)
    matte: np.ndarray | None = None          # uint8 (N, H, W): the subject matte THROUGH the transition (matte_pass=True), same cards / camera / blur as the frames


def build_scene(a, matte_a=None, b=None, matte_b=None, depth_a=None, depth_b=None, params: dict | None = None):
    """-> (PivotScene, OrbitParams, resolved params, info). Everything but the render loop (tests inspect the cards from here)."""
    dd, do = _engine()
    p = resolve_params(params)
    fa, fb_ = as_frames(a, "a"), as_frames(b, "b")
    h0, w0 = fa.shape[1:3]
    w, h = dd.mesh_size(w0, h0, p["short"])
    ma, mb = as_mattes(matte_a, len(fa), h0, w0), as_mattes(matte_b, len(fb_), h0, w0)
    video_on = {"on": True, "off": False, "auto": len(fa) > 1 or len(fb_) > 1}[p["video"]]
    op = _op(do, p["engine"])
    op.video = video_on
    prm = dd.Params(depth_ratio=op.depth_ratio)
    l1, s1 = _layer(dd, fa[0], ma[0], as_depth(depth_a), w, h, prm)
    l2, s2 = _layer(dd, fb_[0], mb[0], as_depth(depth_b), w, h, prm)
    from tools.fusion import depth_smooth as ds
    if ds.uses(op):                                  # smooth / slide: depth layers, no cards (a still image is a one-frame clip)
        op.spring = False
        video = []
        for fr, mt in ((fa, ma), (fb_, mb)):
            ff, mm = _to_mesh(fr, mt, w, h)
            mm = np.concatenate([mm, np.repeat(mm[-1:], len(ff) - len(mm), 0)]) if len(mm) < len(ff) else mm[:len(ff)]
            video.append((ff, mm))
        sc = ds.SmoothScene([l1, l2], op, video)
        info = {"size": [w, h], "input_size": [w0, h0], "frames_in": [len(fa), len(fb_)], "video": video_on, "depth": [s1, s2], "style": op.style, "cards": []}
        return sc, op, p, info
    video = None
    if video_on:
        video = []
        for fr, mt in ((fa, ma), (fb_, mb)):
            ff, mm = _to_mesh(fr, mt, w, h)
            mm = np.concatenate([mm, np.repeat(mm[-1:], len(ff) - len(mm), 0)]) if len(mm) < len(ff) else mm[:len(ff)]
            video.append((ff, mm))
    sc = (do.PivotScene if op.mode == "pivot" else do.VolumeScene)([l1, l2], op, video=video)
    if op.type:
        from tools.fusion import depth_type as dtp
        sc.typ = dtp.TypeLayer(sc, op.type, depth=op.type_depth)
    info = {"size": [w, h], "input_size": [w0, h0], "frames_in": [len(fa), len(fb_)], "video": video_on, "depth": [s1, s2], "style": "chunks",
            "cards": [{"subject": int(c["n_subject"]), "background": int(c["n_back"]), "ref_px": float(c["ref"]),
                       "size_px": [float(c["S"].min()), float(c["S"].max())]} for c in sc.cards.values()]}
    return sc, op, p, info


def depth_pivot_render(a, matte_a=None, b=None, matte_b=None, depth_a=None, depth_b=None, params: dict | None = None, on_frame=None,
                       matte_pass: bool = False) -> PivotResult:
    """render the transition. -> PivotResult(frames uint8 RGB (N,H,W,3), resolved params, info). on_frame(i, n) is called after each frame.
    matte_pass: also render the subject matte through the same scene (PivotResult.matte, uint8 (N,H,W); about doubles the time) so a type sandwich
    keeps the subject in front of the type while the background transforms (depth_orbit render --matte-pass)."""
    dd, do = _engine()
    sc, op, p, info = build_scene(a, matte_a, b, matte_b, depth_a, depth_b, params)
    fl = do.frame_list(op)
    smooth = hasattr(sc, "render_both")
    live = smooth or sc.video is not None
    i2 = sc.image2_start() if live else 0
    info["image2_start"] = int(i2)
    out, cache, mattes = [], {}, []
    for i, (t, _kind) in enumerate(fl):
        q = sc.q(t)
        if smooth:                                                # colour and matte come out of ONE composite (the matte is the subject cards' coverage)
            sc.set_frames(i, i - i2)
            fr, mt = sc.render_both(t)
            out.append(cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_BGR2RGB))
            if matte_pass:
                mattes.append(np.clip(mt * 255.0 + 0.5, 0, 255).astype(np.uint8))
            if on_frame:
                on_frame(i + 1, len(fl))
            continue
        if live:
            sc.set_frames(i, i - i2)
        if q in (0.0, 1.0) and not live:
            if q not in cache:
                cache[q] = sc.render_q(q)
            fr = cache[q]
        elif q in (0.0, 1.0) and sc.typ is None:                 # a rest pose of a clip: the clip's own frame (exact, no card pass, no grown-hole fill round the subject)
            k = 0 if q == 0.0 else 1
            fr_k, mt_k = sc.video[k]
            j = min(max(i if k == 0 else i - i2, 0), len(fr_k) - 1)
            fr = fr_k[j]
            if matte_pass:
                mattes.append(np.clip(mt_k[min(j, len(mt_k) - 1)] * 255.0 + 0.5, 0, 255).astype(np.uint8))
                out.append(cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_BGR2RGB))
                if on_frame:
                    on_frame(i + 1, len(fl))
                continue
        elif q in (0.0, 1.0):
            fr = sc.render_q(q)
        else:
            fr = sc.render_t(t, sc.nsub_at(t))
        out.append(cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_BGR2RGB))
        if matte_pass:
            mattes.append(np.clip(sc.render_matte_t(t, sc.nsub_at(t) if 0.0 < q < 1.0 else 1) * 255.0 + 0.5, 0, 255).astype(np.uint8))
        if on_frame:
            on_frame(i + 1, len(fl))
    info["n_frames"] = len(out)
    return PivotResult(np.stack(out), p, info, np.stack(mattes) if mattes else None)


def depth_pivot(a, matte_a=None, b=None, matte_b=None, depth_a=None, depth_b=None, params: dict | None = None) -> np.ndarray:
    """the pure function: (A, matte A, B, matte B, depth A, depth B, params) -> uint8 RGB frames (N, H, W, 3). See `depth_pivot_render` for the
    resolved params."""
    return depth_pivot_render(a, matte_a, b, matte_b, depth_a, depth_b, params).frames


@register("depth_pivot")
class DepthPivot(TransitionModule):
    """Registry entry for the image-domain pivot transition. It carries the parameter table and the render entry points; it does not draw
    strands (hidden: not in list_modules('transition'), so the type engine's pool and the batch coverage are unchanged)."""
    description = ("Depth 'pivot chunk' image transition: A and B are depth scenes of chunky camera-facing cards, one camera pivots round A's "
                   "volume, B swings in over the top and lands flat. Live clip textures by default. Image in, frames out.")
    hidden = True
    DEMO_DUR = 3.0
    PARAMS = PARAMS
    DEFAULTS = DEFAULTS

    def build(self):
        raise NotImplementedError("depth_pivot is an image transition: call depth_pivot() / depth_pivot_render()")

    def layers(self, t):
        raise NotImplementedError("depth_pivot is an image transition: call depth_pivot() / depth_pivot_render()")

    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        return (DEFAULTS["hold1"] + DEFAULTS["travel"] + DEFAULTS["hold2"]) / float(DEFAULTS["fps"])

    resolve = staticmethod(resolve_params)
    render = staticmethod(depth_pivot_render)
    apply = staticmethod(depth_pivot)
