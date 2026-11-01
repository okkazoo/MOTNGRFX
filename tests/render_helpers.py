"""Hand-built analysis dicts and plans for the renderer tests and demo (no W2/W3 code)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _blob_points(cx, cy, rx, ry, n=48, wobble=0.18, seed=0):
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = 1.0 + wobble * (0.6 * np.sin(3 * a + 0.4) + 0.4 * np.sin(5 * a + 1.3)) + rng.uniform(-0.02, 0.02, n)
    return np.stack([cx + rx * k * np.cos(a), cy + ry * k * np.sin(a)], 1)


def _image_entry(iid, img, contours, lines, focus, sal, empty):
    h, w = img.shape[:2]
    cs = []
    for k, (pts, closed) in enumerate(contours):
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
        cs.append({"id": f"{iid}:c{k}", "points": pts.tolist(), "closed": closed, "area": float(area),
                   "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]})
    return {
        "id": iid, "pixels": img.astype(np.float32), "w": w, "h": h,
        "palette": [{"hex": "#202020", "weight": 0.5}, {"hex": "#ff6a2e", "weight": 0.3}],
        "accent": "#ff6a2e", "luma_mean": float(img.mean()), "contrast": 0.5, "saturation": 0.5,
        "saliency_box": sal, "focus": focus, "empty_regions": empty,
        "contours": cs,
        "lines": [{"p0": list(p0), "p1": list(p1), "strength": 1.0} for p0, p1 in lines],
        "meaning": None,
    }


def make_analysis(scale: float = 1.0) -> dict:
    """Three synthetic images whose contours / lines are exactly what was drawn."""
    imgs = []
    # img0: landscape, warm blob on a blue gradient + a bold diagonal stripe
    W0, H0 = int(1024 * scale), int(768 * scale)
    g = np.linspace(0, 1, H0, dtype=np.float32)[:, None, None]
    img = (np.array([0.05, 0.08, 0.2]) * (1 - g) + np.array([0.1, 0.3, 0.55]) * g) * np.ones((H0, W0, 3), np.float32)
    blob = _blob_points(0.42, 0.5, 0.2, 0.27, seed=1)
    cv2.fillPoly(img, [np.round(blob * [W0, H0]).astype(np.int32)], (1.0, 0.45, 0.15), cv2.LINE_AA)
    inner = _blob_points(0.42, 0.5, 0.08, 0.1, n=24, wobble=0.1, seed=2)
    cv2.fillPoly(img, [np.round(inner * [W0, H0]).astype(np.int32)], (1.0, 0.85, 0.3), cv2.LINE_AA)
    l0 = ((0.62, 0.15), (0.92, 0.85))
    cv2.line(img, (int(l0[0][0] * W0), int(l0[0][1] * H0)), (int(l0[1][0] * W0), int(l0[1][1] * H0)),
             (0.95, 0.95, 0.95), max(2, int(10 * scale)), cv2.LINE_AA)
    imgs.append(_image_entry("img0", np.clip(img, 0, 1), [(blob, True), (inner, True)], [l0],
                             [0.45, 0.5], [0.2, 0.2, 0.45, 0.6], [[0.0, 0.0, 1.0, 0.14], [0.0, 0.86, 1.0, 0.14]]))
    # img1: portrait, stripes + a large circle subject
    W1, H1 = int(768 * scale), int(1024 * scale)
    img = np.zeros((H1, W1, 3), np.float32)
    img[:] = (0.9, 0.88, 0.82)
    for k in range(0, H1, max(4, int(64 * scale))):
        img[k:k + max(2, int(24 * scale))] = (0.15, 0.15, 0.18)
    circ = np.array([[0.5 + 0.3 * math.cos(a), 0.45 + 0.3 * (W1 / H1) * math.sin(a)]
                     for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)])
    cv2.fillPoly(img, [np.round(circ * [W1, H1]).astype(np.int32)], (0.18, 0.42, 1.0), cv2.LINE_AA)
    imgs.append(_image_entry("img1", img, [(circ, True)], [((0.05, 0.85), (0.95, 0.85))],
                             [0.5, 0.45], [0.2, 0.22, 0.6, 0.46], [[0.0, 0.75, 1.0, 0.25]]))
    # img2: noise texture with a bright square subject
    W2, H2 = int(900 * scale), int(900 * scale)
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.random((H2, W2, 3)).astype(np.float32), (0, 0), 6 * scale + 0.5) * 0.6 + 0.1
    sq = np.array([[0.35, 0.3], [0.7, 0.33], [0.66, 0.68], [0.32, 0.64]])
    cv2.fillPoly(img, [np.round(sq * [W2, H2]).astype(np.int32)], (1.0, 0.9, 0.2), cv2.LINE_AA)
    imgs.append(_image_entry("img2", np.clip(img, 0, 1), [(sq, True)], [],
                             [0.5, 0.5], [0.3, 0.28, 0.42, 0.42], [[0.0, 0.0, 1.0, 0.25]]))
    return {"text": "MAKE IT MOVE", "text_lines": ["MAKE IT MOVE"], "duration_hint": None,
            "images": imgs, "audio": None, "mood": {"energy": 0.7, "valence": 0.6, "keywords": []}}


STYLE = {
    "palette": ["#111111", "#f2f2f2", "#ff4d2e", "#2e6bff", "#ffd23f"],
    "background": "palette:0",
    "fonts": {"display": "Big Shoulders Display", "body": "Space Grotesk"},
    "motion": "snappy", "energy": 0.7, "grain": 0.12, "vignette": 0.25, "beat_flash": 0.25,
}


def base_plan(duration=10.0, fps=30, bpm=120.0, motion="snappy", seed=1234) -> dict:
    step = 60.0 / bpm
    beats = [round(k * step, 6) for k in range(int(duration / step) + 1) if k * step <= duration]
    return {
        "version": 1, "seed": seed, "duration": duration, "fps": fps,
        "style": dict(STYLE, motion=motion),
        "audio": {"source": "synth", "bpm": bpm, "key": "A", "scale": "minor", "mood": "driving",
                  "beats": beats, "downbeats": beats[::4], "sections": [{"t": 0.0, "label": "intro"}],
                  "risers": [], "sfx": []},
        "voice": None, "scenes": [],
    }


def box(a, b):
    return {"9:16": list(a), "16:9": list(b)}


def text_layer(text, t0, t1, verb_in="fade", verb_out="fade", **kw):
    d = {"type": "text", "text": text, "role": "headline", "font": "display", "case": "upper",
         "color": "palette:1", "highlight": None, "align": "center",
         "box": box([0.08, 0.62, 0.84, 0.16], [0.06, 0.6, 0.5, 0.25]),
         "t0": t0, "t1": t1, "in": {"verb": verb_in, "dur": 0.4}, "out": {"verb": verb_out, "dur": 0.25},
         "emphasis": []}
    d.update(kw)
    return d


def showcase_plan(duration=10.0, fps=30) -> dict:
    """3 scenes exercising every layer type, bg motion, treatment, mask, emphasis and echo."""
    p = base_plan(duration, fps)
    p["scenes"] = [
        {"id": "s0", "t0": 0.0, "t1": 4.0,
         "background": {"type": "image", "src": "img0", "focus": [0.45, 0.5],
                        "motion": {"verb": "ken_burns", "zoom": [1.0, 1.15], "pan": [[0, 0], [0.03, -0.01]]},
                        "treatment": {"duotone": None, "blur": 0.0, "darken": 0.2}},
         "layers": [
             {"type": "shape", "source": "contour", "ref": "img0:c0", "primitive": None, "box": None,
              "stroke": "palette:4", "stroke_width": 0.007, "fill": None, "fill_opacity": 0.0,
              "offset": [0, 0], "scale": 1.0, "t0": 0.2, "t1": 4.2,
              "in": {"verb": "stroke_draw", "dur": 1.0}, "out": {"verb": "fade", "dur": 0.3},
              "emphasis": [{"verb": "pulse", "on": "downbeats", "amount": 0.04}],
              "echo": {"count": 2, "spacing": 0.02, "decay": 0.5}},
             {"type": "shape", "source": "line", "ref": "img0:l0", "stroke": "palette:2", "stroke_width": 0.01,
              "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0, "t0": 0.6, "t1": 4.0,
              "in": {"verb": "stroke_draw", "dur": 0.6}, "out": {"verb": "fade", "dur": 0.2},
              "emphasis": [], "echo": {"count": 1, "spacing": 0.015, "decay": 0.5}},
             text_layer("Make it move", 0.5, 3.9, "word_stagger", "slide_up",
                        box=box([0.06, 0.05, 0.88, 0.16], [0.05, 0.06, 0.6, 0.2]),
                        **{"in": {"verb": "word_stagger", "dur": 0.45, "stagger_beats": 0.5},
                           "emphasis": [{"verb": "pulse", "on": "beats", "amount": 0.05}]}),
             text_layer("shapes borrowed from the image", 1.6, 3.9, "type_on", "fade",
                        role="sub", font="body", case="as_is",
                        box=box([0.1, 0.86, 0.8, 0.06], [0.05, 0.82, 0.5, 0.08]),
                        **{"in": {"verb": "type_on", "dur": 0.9}}),
         ],
         "transition_out": {"type": "shape_reveal", "dur": 0.6}},
        {"id": "s1", "t0": 4.0, "t1": 8.0,
         "background": {"type": "image", "src": "img1", "focus": [0.5, 0.45],
                        "motion": {"verb": "parallax_split", "zoom": [1.0, 1.2]},
                        "treatment": {"duotone": ["palette:0", "palette:3"], "blur": 0.0, "darken": 0.0}},
         "layers": [
             {"type": "shape", "source": "contour", "ref": "img1:c0", "stroke": "palette:1",
              "stroke_width": 0.005, "fill": "palette:2", "fill_opacity": 0.25,
              "offset": [0, 0], "scale": 1.0, "t0": 4.1, "t1": 8.0,
              "in": {"verb": "stroke_draw", "dur": 0.8}, "out": {"verb": "fade", "dur": 0.2},
              "emphasis": [{"verb": "shake", "on": "downbeats", "amount": 0.01}],
              "echo": {"count": 3, "spacing": 0.012, "decay": 0.55}},
             {"type": "image", "src": "img2", "mask": "rounded",
              "box": box([0.55, 0.06, 0.38, 0.22], [0.72, 0.08, 0.22, 0.38]),
              "t0": 4.6, "t1": 7.9, "in": {"verb": "scale_pop", "dur": 0.4}, "out": {"verb": "scale_down", "dur": 0.25},
              "emphasis": [{"verb": "pulse", "on": "downbeats", "amount": 0.05}],
              "border": {"color": "palette:1", "width": 0.004}},
             {"type": "image", "src": "img2", "mask": "contour:img2:c0",
              "box": box([0.07, 0.06, 0.3, 0.2], [0.05, 0.08, 0.18, 0.32]),
              "t0": 5.0, "t1": 7.9, "in": {"verb": "blur_in", "dur": 0.4}, "out": {"verb": "fade", "dur": 0.25},
              "emphasis": [], "border": {"color": "palette:4", "width": 0.004}},
             {"type": "shape", "source": "primitive", "primitive": "grid",
              "box": box([0.08, 0.3, 0.84, 0.3], [0.3, 0.2, 0.4, 0.6]), "stroke": "palette:1",
              "stroke_width": 0.002, "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0,
              "t0": 4.2, "t1": 5.6, "in": {"verb": "stroke_draw", "dur": 0.5}, "out": {"verb": "fade", "dur": 0.3},
              "emphasis": [], "echo": None},
             text_layer("Split the frame", 4.5, 7.9, "kinetic_split", "mask_wipe",
                        highlight="palette:2", color="palette:1",
                        box=box([0.08, 0.7, 0.84, 0.14], [0.06, 0.66, 0.55, 0.2]),
                        emphasis=[{"verb": "color_flip", "on": "downbeats"},
                                  {"verb": "underline_draw", "on": "beats"}]),
         ],
         "transition_out": {"type": "zoom_blur", "dur": 0.4}},
        {"id": "s2", "t0": 8.0, "t1": duration,
         "background": {"type": "shapes", "color": "palette:0", "motion": {"verb": "static"}},
         "layers": [
             {"type": "shape", "source": "primitive", "primitive": "wave",
              "box": box([0.05, 0.38, 0.9, 0.16], [0.1, 0.4, 0.8, 0.2]), "stroke": "palette:4",
              "stroke_width": 0.006, "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0,
              "t0": 8.0, "t1": duration, "in": {"verb": "stroke_draw", "dur": 0.5}, "out": {"verb": "cut"},
              "emphasis": [], "echo": {"count": 2, "spacing": 0.02, "decay": 0.5}},
             {"type": "shape", "source": "primitive", "primitive": "ring",
              "box": box([0.3, 0.12, 0.4, 0.22], [0.42, 0.08, 0.16, 0.3]), "stroke": "palette:2",
              "stroke_width": 0.008, "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0,
              "t0": 8.1, "t1": duration, "in": {"verb": "scale_pop", "dur": 0.4}, "out": {"verb": "cut"},
              "emphasis": [{"verb": "pulse", "on": "beats", "amount": 0.1}], "echo": None},
             {"type": "shape", "source": "primitive", "primitive": "dot_field",
              "box": box([0.1, 0.56, 0.8, 0.1], [0.1, 0.62, 0.3, 0.2]), "stroke": "palette:3",
              "stroke_width": 0.004, "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0,
              "t0": 8.2, "t1": duration, "in": {"verb": "stroke_draw", "dur": 0.5}, "out": {"verb": "cut"},
              "emphasis": [], "echo": None},
             text_layer("On the beat", 8.15, duration, "scale_pop", "cut", highlight="palette:2",
                        color="palette:0", box=box([0.1, 0.7, 0.8, 0.12], [0.5, 0.66, 0.42, 0.2])),
         ],
         "transition_out": {"type": "crossfade", "dur": 0.3}},
    ]
    return p
