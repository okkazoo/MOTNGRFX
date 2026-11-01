"""regenerates the type + compose blocks of P08.json (keeps stills/shots). python 15_fusion/pieces/_gen_P08.py"""
import json
from pathlib import Path
p = Path(__file__).with_name("P08.json")
j = json.loads(p.read_text(encoding="utf-8"))
j["type"]["words"] = [
 {"text": "ALL OF THIS", "font": "MajorMonoDisplay-Regular.ttf", "module": "lanes", "lines": ["ALL OF", "THIS"], "t0": 0.0, "t1": 13.5,
  "form_s": 8.0, "exit_s": 1.5, "hit_on": "bar", "form_in": "lines", "form_out": "lines",
  "placement": {"w": 470, "h": 250, "dy": -0.20}},
 {"text": "WAS ONCE", "font": "MajorMonoDisplay-Regular.ttf", "module": "lanes", "lines": ["WAS", "ONCE"], "t0": 0.0, "t1": 6.5,
  "form_s": 1.04, "exit_s": 1.5, "hit_on": "bar", "form_in": "lines", "form_out": "lines",
  "placement": {"w": 430, "h": 330, "dy": 0.02}},
 {"text": "LIGHT", "font": "Foldit[wght].ttf", "module": "lanes", "t0": 0.0, "t1": 5.6, "form_s": 1.1, "exit_s": 1.0,
  "hit_on": "bar", "form_in": "lines", "form_out": "lines", "lines": ["LIGHT"],
  "axes": {"wght": {"keys": [[23.5, 180], [27.5, 620]], "ease": "smooth"}}, "placement": {"w": 520, "h": 320}}]
j["type"]["camera"] = [
 {"module": "pullback_reveal", "t0": 0.0, "word_scale": 1.3, "from_scale": 0.55},
 {"module": "hold_drift", "t0": 9.2, "dolly": 0.04, "comp": [0.0, 0.0]},
 {"module": "follow", "t0": 12.3},
 {"module": "pullback_reveal", "t0": 15.0, "word_scale": 1.3, "from_scale": 0.6},
 {"module": "hold_drift", "t0": 17.2, "dolly": 0.04, "comp": [0.0, 0.0]},
 {"module": "follow", "t0": 20.5},
 {"module": "pullback_reveal", "t0": 22.9, "word_scale": 1.12, "from_scale": 0.65},
 {"module": "hold_drift", "t0": 25.2, "dolly": 0.03, "comp": [0.0, 0.0]}]
STARTS = {f"s{k+1}": 103 * k for k in range(6)}
def pw(points):
    def f(t):
        if t <= points[0][0]: return points[0][1]
        for (a, va), (b, vb) in zip(points[:-1], points[1:]):
            if t <= b:
                u = (t - a) / (b - a); u = u * u * (3 - 2 * u)
                return va + (vb - va) * u
        return points[-1][1]
    return f
gateA = pw([(0, 0), (5.0, 0), (7.6, 1), (12.3, 1), (13.3, 0), (14.6, 0), (15.2, 0), (15.9, 1), (20.3, 1), (21.3, 0), (30, 0)])
gateL = pw([(0, 0), (22.8, 0), (23.9, 1), (30, 1)])
def anim(sid, fn):
    s0 = STARTS[sid]
    ks = []
    for i in range(0, 121, 4):
        ks.append([round(i / 120.0, 4), round(float(fn((s0 + i) / 24.0)), 3)])
    if ks[-1][0] != 1.0:
        ks.append([1.0, round(float(fn((s0 + 120) / 24.0)), 3)])
    return {"anim": ks}
def ops(sid):
    return [
     {"op": "multiply", "out": "tmA", "args": {"a": "$type_matte", "b": anim(sid, gateA)}},
     {"op": "multiply", "out": "tmL", "args": {"a": "$type_matte", "b": anim(sid, gateL)}},
     {"op": "trace_edges", "out": "edges", "args": {"footage": "$footage", "low": 0.008, "high": 0.03, "temporal_smooth": 0.6, "blur": 2.5}},
     {"op": "type_from_edges", "out": "tfeA", "args": {"edges": "$edges", "type_a": "$tmA", "grow": 5, "line_width": 1.5, "density": 2.5, "fill": 0.55, "softness": 3}},
     {"op": "type_from_edges", "out": "tfeL", "args": {"edges": "$edges", "type_a": "$tmL", "grow": 6, "line_width": 2.5, "density": 2.5, "fill": 0.0, "softness": 3}},
     {"op": "tint_type", "out": "colA", "args": {"type_a": "$tmA", "footage": "$footage", "mode": "sample", "sample_blur": 18, "sat": 1.25, "lum_target": 0.96, "lum_amount": 1.0}},
     {"op": "tint_type", "out": "colL", "args": {"type_a": "$tmL", "footage": "$footage", "mode": "sample", "sample_blur": 24, "sat": 1.9, "lum_target": 0.97, "lum_amount": 1.0}},
     {"op": "over", "out": "image", "args": {"base": "$footage", "fg": "$colA", "alpha": "$tfeA"}},
     {"op": "translate", "out": "lifted", "args": {"img": "$image", "dx": 0, "dy": 672}},
     {"op": "refract", "out": "glass", "args": {"footage": "$lifted", "type_depth_or_matte": "$tmL", "strength": 16, "blur": 10, "chroma": 0.3, "rim": 1.0, "rim_color": [1.0, 0.85, 0.55]}},
     {"op": "blur", "out": "lbody", "args": {"img": "$tmL", "sigma": 5}},
     {"op": "multiply", "out": "lbody", "args": {"a": "$lbody", "b": 3.0}},
     {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$glass", "alpha": "$lbody"}},
     {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$colL", "alpha": "$tfeL"}},
     {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$tfeA", "radius": 16, "intensity": {"curve": "rms", "lo": 0.04, "hi": 0.25}, "color": [1.0, 0.8, 0.55]}},
     {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$tmL", "radius": 30, "intensity": {"curve": "rms", "lo": 0.10, "hi": 0.40}, "color": [1.0, 0.75, 0.4]}},
     {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0, "gamma": 1.0, "gain": 1.02, "sat": 1.05}},
     {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 8, "amount": 0.015, "size": 1.2}},
     {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.2}}]
j["compose"]["default"] = {"ops": ops("s1")}
j["compose"]["shots"] = {sid: {"ops": ops(sid)} for sid in STARTS}
p.write_text(json.dumps(j, indent=1), encoding="utf-8")
print("ok")
