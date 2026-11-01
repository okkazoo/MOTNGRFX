"""P14 Lift (v004, 120 bpm grid, 12 s, 9:16) -> P14.json.

Two angles on one dancer = two shots (each its own MOSS still, each ONE continuous orbit), joined by a HARD cut on the 6.0 s
downbeat (crossfade 0, no last-frame chaining). 145 frames each: Alpha-Gen covers at most 145 frames (piece_runner._alpha), so
both shots keep a full solid dancer matte.
Weave: ONE type layer in front of AND behind the dancer at once (sandwich front=): the near half of each yawed word (type depth
pass, near = white) passes in front of the dancer, the far half behind her; solid matte (threshold 0.5 / soft 0.08).
s1 (0-6 s): footage flow pushes the type (flow_wake, settle 0.75 in holds) + scribble outlines around the dancer drawn on every
beat (contour_lines scribble). s2 (6-12 s): rising Motion-Track trails stream up off the dancer from the 8.0 s downbeat
(masktrack rise) -> NO flow_wake there (trails OR flow-on-type). Glow pulses on the kick (low band).

    python 15_fusion/pieces/_gen_P14.py [--song wav --grid json] [--reset-tc | --reset-compose]   (see _v004_grid.py)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _v004_grid as G  # noqa: E402

A = G.args("P14", "s2")                                       # s2 = 15_fusion/v004/audio/PICKS.md
GRID = G.load_grid(A.grid)
PLAN, IV = G.plan(GRID)                                       # 120 bpm: hits 2 / 5 / 8 / 11 s, 2-beat holds, 2-beat transitions
HOLDS = [w["hold"] for w in PLAN]
FPS = 24
NF = int(round(GRID["cut"] * FPS))
CUT = min(GRID["downbeats"], key=lambda d: abs(d - GRID["cut"] / 2))   # hard cut on the downbeat nearest the middle (6.0 s)
SPAN1 = int(round(CUT * FPS))
SPANS = [SPAN1, NF - SPAN1]
FRAMES = 145                                                  # rendered footage (alpha cap); a longer span needs new footage
for k, sp in enumerate(SPANS):
    if sp + 1 > FRAMES:
        print(f"WARNING s{k + 1}: span {sp} f > footage {FRAMES} f -> the last {sp + 1 - FRAMES} f hold (re-render the shot to avoid)")
TRAILS = min([w["hit"] for w in PLAN if w["hit"] >= CUT + 1.0] or [CUT + 2 * IV * 2])   # trails leave the dancer on the next hit


def settle(holds=HOLDS):
    return G.settle(holds)


def scribble_reveal(t0=0.0, t1=CUT):
    """draw-on in 4 frames after every beat, hold, draw-off before the next one -> the scribbles pulse on the kick."""
    keys = [[0.0, 0.0]]
    for b in GRID["beats"]:
        if t0 - 1e-6 <= b < t1 - 1e-6:
            keys += [[round(b, 3), 0.0], [round(b + 0.33 * IV, 3), 1.0], [round(b + 0.6 * IV, 3), 1.0], [round(b + 0.9 * IV, 3), 0.0]]
    return {"timeline": keys}


def words():
    common = dict(module="lanes", form_in="lines", form_out="lines")
    TM = [{k: w[k] for k in ("t0", "t1", "form_s", "exit_s", "hit_on")} for w in PLAN]
    return [
        dict(common, text="RISE", lines=["RISE"], font="FacadeVF.ttf", **TM.pop(0),
             placement={"yaw": -22}, axes={"wght": [500, 900, "beat"]}),
        dict(common, text="UP", lines=["UP"], font="Blackout", **TM.pop(0),
             placement={"yaw": 22}, axes={}),
        dict(common, text="INTO", lines=["INTO"], font="FacadeVF.ttf", **TM.pop(0),
             placement={"yaw": -22}, axes={"wght": [500, 900, "beat"]}),
        dict(common, text="THE LIGHT", lines=["THE", "LIGHT"], font="Blackout", **TM.pop(0),
              placement={"yaw": 20}, axes={}),
    ]


WARM = [1.0, 0.93, 0.8]


def ops(shot):
    flow = shot == "s1"
    o = [{"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [1080, 1920], "guide": "$footage"}}]
    ta = "$type_matte"
    if flow:                                       # footage flow on the type only where the trails are off
        o.append({"op": "flow_wake", "out": "wa", "args": {
            "type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m", "strength": {"curve": "onset", "lo": 3.0, "hi": 6.0},
            "radius": 70, "decay": 0.9, "trail": 10, "trail_gain": 1.0, "bg_drag": 0.12, "max_disp": 0.2, "smooth": 1.0,
            "settle": settle(), "frame_idx": "$frame_idx", "u": "$u", "state_key": "P14:s1"}})
        ta = "$wa"
    o += [
        # near half of the yawed word (type depth: near = white). luminosity_key needs RGB (compose._luma on HxW breaks): grey -> RGB
        {"op": "fill", "out": "blk", "args": {"like": "$footage", "color": [0.0, 0.0, 0.0]}},
        {"op": "over", "out": "tdep", "args": {"base": "$blk", "fg": [1.0, 1.0, 1.0], "alpha": "$type_depth"}},
        {"op": "luminosity_key", "out": "near", "args": {"img": "$tdep", "low": 0.54, "high": 0.6}},
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 1.0 if flow else 0.92,   # s2 is a pale room: darker
                                                 "gain": 0.9 if flow else 0.78, "sat": 1.05}},
        {"op": "tint_type", "out": "tcol", "args": {"type_a": ta, "footage": "$footage", "mode": "sample", "sample_blur": 50,
                                                     "sat": 1.6, "lum_target": 0.92, "lum_amount": 0.85}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_rgb": "$tcol", "type_a": ta, "matte": "$m",
                                                    "matte_threshold": 0.5, "matte_soft": 0.08, "front": "$near",
                                                    "type_opacity": 0.96}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": ta, "radius": 20,
                                                    "intensity": {"curve": "low", "lo": 0.06, "hi": 0.5}, "color": [1.0, 0.85, 0.6]}},
    ]
    if flow:
        o += [
            {"op": "contour_lines", "out": "cl", "args": {"src": "$m", "mode": "scribble", "strokes": 2, "jitter": 5.0, "width": 2,
                                                          "boil": 3, "frame_idx": "$frame_idx", "seed": 14, "max_contours": 2,
                                                          "min_len": 60.0, "smooth": 3.0, "reveal": scribble_reveal()}},
            {"op": "over", "out": "image", "args": {"base": "$image", "fg": WARM, "alpha": "$cl"}},
        ]
    o += [
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.04, "sat": 1.08}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 14, "amount": 0.02, "size": 1.2}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.26}},
    ]
    return o


NEG = ("text, watermark, logo, caption, cartoon, illustration, 3d render, oversaturated, extra limbs, deformed hands, "
       "deformed feet, blurry subject, second person")
LOOK = ("one contemporary dancer in loose pale grey clothing, empty raw concrete dance studio, one shaft of hard white sunlight "
        "from a high window, dust floating in the light beam, colour photograph, cinematic 35mm, tall vertical frame, "
        "the dancer is full body in frame, centred, filling about half the frame height")


def build():
    return {
        "id": "P14", "slug": "lift", "fps": 24, "version": "v004",
        "audio": {"src": G.rel(A.song), "bpm": round(GRID["bpm"], 3), "analysis": G.analysis("P14", A.song, GRID, write=False),
                  "cut": GRID["cut"], "fade": 0.4},
        "type": {"seed": 14, "energy": 0.9, "mood": "driving", "tier": "kinetic",
                 "palette": {"bg": [8, 8, 10], "fg": [255, 240, 215], "accents": [[255, 205, 140], [140, 190, 255], [255, 255, 255]]},
                 "camera": "beat_orbit", "camera_params": {"read_dps": 6.0, "intro_deg": 40.0, "fast_dps": 60.0, "elev": 5.0, "bob": 1.2},
                 "transitions": ["twist", "circuit_board", "fan"], "line": {"px": 4.0}, "words": words()},
        "type_passes": ["matte", "depth"],
        "stills": [
            {"id": "A", "model": "moss", "seed": 1401, "size": [1088, 1920], "negative": NEG,
             "prompt": "a contemporary dancer mid-leap seen from a low three-quarter front angle, legs split in the air, arms "
                       "sweeping up, " + LOOK},
            {"id": "B", "model": "moss", "seed": 1402, "size": [1088, 1920], "negative": NEG,
             "prompt": "a contemporary dancer seen from a high side angle, rising out of a deep crouch with both arms reaching "
                       "up into the light, " + LOOK}],
        "shots": [
            {"id": "s1", "kind": "i2v", "first": "A", "frames": FRAMES, "span": SPANS[0], "seed": 141, "size": [1088, 1920], "matte": True,
             "depth": False, "flow": True,
             "prompt": "the camera orbits slowly around the dancer in one continuous move from left to right, the dancer leaps, "
                       "lands and spins with arms sweeping upward, dust swirls in the shaft of hard light, the concrete walls "
                       "slide past behind, energetic, cinematic"},
            {"id": "s2", "kind": "i2v", "first": "B", "frames": FRAMES, "span": SPANS[1], "seed": 142, "size": [1088, 1920], "matte": True,
             "depth": False, "flow": False,
             "prompt": "the camera orbits around the dancer in one continuous move from right to left, the dancer rises from the "
                       "crouch and stretches up into the shaft of hard light, arms reaching high, dust glitters in the beam, "
                       "energetic, cinematic",
             "masktrack": {"style": "rise", "start": TRAILS, "denoise": 1.0, "corridor": False, "subject": "the dancer"}}],
        "compose": {"size": [1080, 1920], "crossfade": [0], "crf": 16,
                    "default": {"ops": ops("s1")}, "shots": {"s2": {"ops": ops("s2")}}},
        "signoff": {"off": True},
        "parent": {"piece": None, "why": "v004 brief P14 + reel-017 visual decisions: 2 orbit shots hard-cut on a downbeat, type "
                                         "in front of and behind the dancer at once, rise trails (never down), scribbles on beats"},
    }


def widen(j):
    """16:9 version P14_w: own wide stills (dancer on a third, more studio), both orbit shots re-generated wide (draft 960x544),
    hard cut on the same grid downbeat; shot frames = the smallest 8k+1 covering each span. Type / compose landscape
    re-layout is pending (contour-bridge redesign): the type block is the portrait plan and must not be rendered as is."""
    j = json.loads(json.dumps(j))
    j.update(id="P14_w", format="16:9")
    j["parent"] = {"piece": "P14", "why": "16:9 version (Both formats rule): own wide stills + own wide orbit shots, same cut"}
    wl = ("one contemporary dancer in loose pale grey clothing, empty raw concrete dance studio seen wide, one shaft of hard white "
          "sunlight from a high window, dust floating in the light beam, colour photograph, cinematic 35mm, wide horizontal frame, "
          "the dancer full body on the left third of the frame, the studio walls and the light shaft filling the rest")
    a, b = j["stills"]
    a.update(seed=1411, size=[1920, 1088], prompt="a contemporary dancer mid-leap seen from a low three-quarter front angle, legs "
             "split in the air, arms sweeping up, " + wl)
    b.update(seed=1412, size=[1920, 1088], prompt="a contemporary dancer seen from a high side angle, rising out of a deep crouch "
             "with both arms reaching up into the light, " + wl.replace("left third", "right third"))
    for s_, sp in zip(j["shots"], SPANS):
        s_.update(size=[1920, 1088], frames=G.snap8_up(sp + 1), span=sp, seed=s_["seed"] + 10)
        if sp + 1 > 145:
            print(f"NOTE {s_['id']}: {sp} f span > 145 -> Alpha-Gen covers only the first 145 f (piece_runner._alpha cap)")
    j["compose"]["size"] = [1920, 1080]
    j["type"]["format_note"] = "PENDING landscape re-layout"
    return j


if __name__ == "__main__":
    if not A.no_audio:
        G.analysis("P14", A.song, GRID, write=True)
    out = HERE / ("P14_w.json" if A.wide else "P14.json")
    out.write_text(json.dumps(widen(build()) if A.wide else build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, "| song", G.rel(A.song), "| hits", [w["hit"] for w in PLAN], "| cut", CUT, "spans", SPANS, "| trails", TRAILS)
    if A.reset_tc or A.reset_compose:
        G.reset("P14_w" if A.wide else "P14", keep_type=A.reset_compose and not A.reset_tc)
