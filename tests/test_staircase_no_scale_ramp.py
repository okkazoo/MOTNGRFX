"""staircase entry never draws a tiny copy of the word first (user 2026-11-01, own-video v001 r1 'FIRST WORD': the word was drawn ~1/10 size beside the head at
frames 18-26, then grew). The old entry (param tiny_copy=True) drew a hairline copy at 0.15 x 0.09 of the placement; the default now runs the lanes straight onto the
staircase and the strands spread into the outline of the word at its final size.

Frame sampling is the type scene's own (spec.build_scene + raster.collect, CPU, ~2 s). The bbox is that of every drawn polyline; before the outline forms the lanes
are a bundle of strands, not a word, so the checks are: (1) the drawn extent never falls to the tiny copy's size
(old: 0.10 of hold height, new: >= 0.25), (2) from the frame the traced word is wider than half its hold width, the height stays >= 0.85 of the hold height."""
import copy

import numpy as np
import pytest

SIZE = (576, 1024)
TYPE = {"bar": 4, "beats": [0.6417, 1.3092, 1.9802, 2.6415, 3.3072, 3.9743, 4.6431, 5.3109, 5.9779, 6.6405], "bpm": 90.0, "camera": "beat_orbit",
        "camera_params": {"bob": 0.55, "elev": 2.0, "fast_dps": 18.0, "intro_deg": 22.0, "read_dps": 4.2}, "downbeats": [1.9802, 4.6431],
        "duration": 8.0, "energy": 0.6, "fps": 24, "line": {"px": 4.0}, "mood": "Dark, Urban", "seed": 53764, "transitions": ["cut"],
        "palette": {"accents": [[200, 40, 40], [40, 80, 180], [255, 200, 50]], "bg": [15, 10, 20], "fg": [245, 240, 220]},
        "words": [{"entry": "right", "exit_s": 0.67, "font": "Blackout_Midnight.ttf", "form_in": "lines", "form_out": "flicker", "form_s": 1.98, "hit_on": "bar",
                   "module": "staircase", "placement": {"dx": 0.0, "dy": -0.1, "hf": 0.3, "pitch": 0.0, "roll": 0.0, "wf": 0.8, "yaw": 10.5},
                   "render": "flat", "t0": 0.0, "t1": 4.45, "text": "FIRST WORD"}]}


def _profile(tiny_copy):
    from mograph.render import cam3d
    from tools.fusion import raster, spec as S, type_frame as TF
    ts = copy.deepcopy(TYPE)
    if tiny_copy:
        ts["words"][0]["params"] = {"tiny_copy": True}
    world = (cam3d.W, cam3d.H)
    try:
        sc = S.build_scene(TF._clean_spec(ts), size=SIZE)
        cam = sc.chain if sc.chain is not None else sc.seq.demo_camera()
        hit = float(sc.info["words"][0]["hit"])
        rows = []
        for t in np.arange(0.0, hit + 0.4, 1 / 12):
            c = cam.at(t) if hasattr(cam, "at") else cam
            xs, ys = [], []
            for it in raster.collect(sc.seq.layers(t), c, SIZE):
                for xy, _q, _c in it.polys:
                    xs += [float(xy[:, 0].min()), float(xy[:, 0].max())]
                    ys += [float(xy[:, 1].min()), float(xy[:, 1].max())]
            rows.append((t, (max(ys) - min(ys)) / SIZE[1], (max(xs) - min(xs)) / SIZE[0]) if xs else (t, None, None))
        return hit, rows
    finally:
        cam3d.set_world(*world)


def test_entry_never_draws_a_tiny_copy():
    hit, rows = _profile(False)
    vis = [r for r in rows if r[1] is not None]
    assert vis, "nothing drawn"
    hold = vis[-1][1]
    assert hold > 0.2
    assert min(r[1] for r in vis) >= 0.25 * hold                           # the old tiny copy was 0.10 x hold tall
    wide = [r for r in vis if r[2] >= 0.5 * vis[-1][2]]
    assert wide
    first = wide[0][0]
    assert all(r[1] >= 0.85 * hold for r in vis if r[0] >= first)          # from the traced word on, never smaller than 0.85 of the hold height


def test_tiny_copy_param_brings_back_the_old_entry():
    hit, rows = _profile(True)
    vis = [r for r in rows if r[1] is not None]
    hold = vis[-1][1]
    assert min(r[1] for r in vis) < 0.15 * hold                            # the regression this test guards: the 1/10 copy
