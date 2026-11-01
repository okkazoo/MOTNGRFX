"""regenerates the compose + word blocks of P09.json (keeps stills/shots). python 15_fusion/pieces/_gen_P09.py"""
import json
from pathlib import Path
p = Path(__file__).with_name("P09.json")
j = json.loads(p.read_text(encoding="utf-8"))
j["type"]["words"] = [
 {"text": "GOLDEN", "font": "Monoton-Regular.ttf", "module": "lanes", "t0": 0.0, "t1": 4.0, "form_s": 1.5, "exit_s": 0.5,
  "hit_on": "bar", "form_in": "lines", "form_out": "lines"},
 {"text": "STAY", "font": "Monoton-Regular.ttf", "module": "lanes", "t0": 0.0, "t1": 3.9, "form_s": 0.6, "exit_s": 0.6,
  "hit_on": "bar", "form_in": "lines", "form_out": "lines"}]
def ops(gate):
    return [
     {"op": "multiply", "out": "tm", "args": {"a": "$type_matte", "b": gate}},
     {"op": "fill", "out": "black", "args": {"like": "$footage", "color": [0, 0, 0]}},
     {"op": "depth_occlude", "out": "vis", "args": {"footage": "$black", "footage_depth": "$depth", "type_rgb": [1, 1, 1], "type_a": "$tm",
        "plane": 0.80, "softness": 0.16, "type_opacity": 1.0}},
     {"op": "blur", "out": "visb", "args": {"img": "$vis", "sigma": 9}},
     {"op": "multiply", "out": "vism", "args": {"a": "$visb", "b": 2.5}},
     {"op": "refract", "out": "image", "args": {"footage": "$footage", "type_depth_or_matte": "$tm", "strength": 14, "blur": 12,
        "chroma": 0.25, "rim": 1.0, "rim_color": [1.0, 0.93, 0.72], "mask": "$vism"}},
     {"op": "tint_type", "out": "tcol", "args": {"type_a": "$tm", "footage": "$footage", "mode": "sample", "sample_blur": 40,
        "sat": 1.5, "lum_target": 0.92, "lum_amount": 0.7}},
     {"op": "depth_occlude", "out": "image", "args": {"footage": "$image", "footage_depth": "$depth", "type_rgb": "$tcol", "type_a": "$tm",
        "plane": 0.80, "softness": 0.16, "type_opacity": 0.28}},
     {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$vis", "radius": 24,
        "intensity": {"curve": "low", "lo": 0.10, "hi": 0.65}, "color": [1.0, 0.74, 0.3]}},
     {"op": "halation", "out": "image", "args": {"frame": "$image", "threshold": 0.72, "radius": 24, "strength": 0.3, "tint": [1.0, 0.6, 0.25]}},
     {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.02, "gamma": 1.08, "gain": 1.06, "sat": 1.12}},
     {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 9, "amount": 0.018, "size": 1.2}},
     {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.2}}]
# piece frame f: s1 local = f (u = f/120), s2 local = f - 109 (u = (f-109)/80). type is off from 3.9 s to 5.1 s (strand transition), ramps 3.6-3.95 / 5.1-5.35
def u1(t): return round(t * 24 / 120.0, 4)
def u2(t): return round((t * 24 - 109) / 80.0, 4)
g1 = {"anim": [[0, 1.0], [u1(3.6), 1.0], [u1(3.95), 0.0], [1, 0.0]]}
g2 = {"anim": [[0, 0.0], [u2(5.1), 0.0], [u2(5.35), 1.0], [1, 1.0]]}
j["type"]["camera"] = [{"module": "hold_drift", "window": "all", "dolly": 0.08}]   # one composed hold over the whole piece (word-to-word Follow relax made STAY wander)
j["compose"]["default"] = {"ops": ops(1.0)}
j["compose"]["shots"] = {"s1": {"ops": ops(g1)}, "s2": {"ops": ops(g2)}}
p.write_text(json.dumps(j, indent=1), encoding="utf-8")
print("ok")
