"""P12 Night Market (v004, 9:16) -> P12.json. Timing is planned on the song's beat grid (_v004_grid: --song/--grid, else
t2m/P12.wav if present, else the current pick P12_s2), so a new song re-plans the type without new footage (--reset-tc).

One scene, one camera move: a single 289-frame i2v shot (forward dolly down a lantern street), no cut.
Weave: the type rides the dolly (camera track, recipe_eval.cam_ops) and is pushed by the footage flow (flow_wake, settle 0.75 in
holds). The glowing lanterns are keyed from the footage (luminosity_key, thresholded = SOLID matte). In front AND behind at once:
while a word forms / exits its orange fill hides behind the lanterns and its teal outline crosses in front of them; on the hold
it flips: the fill comes to the front and the outline drops behind the lanterns. Transition material (type_line between words)
is pale orange and always woven behind the lanterns. Glow pulses on the kick (low band).
Alpha-Gen is not used: the runner caps it at 145 frames (piece_runner._alpha), a 289-frame shot would lose its matte at 6 s.

    python 15_fusion/pieces/_gen_P12.py [--song wav --grid json] [--reset-tc | --reset-compose]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _v004_grid as G  # noqa: E402

A = G.args("P12", "s2")                                       # s2 = 15_fusion/v004/audio/PICKS.md
GRID = G.load_grid(A.grid)
PLAN, IV = G.plan(GRID)                                       # 120 bpm: hits 2 / 5 / 8 / 11 s, 2-beat holds, 2-beat transitions
HOLDS = [w["hold"] for w in PLAN]


def settle(holds=HOLDS):
    return G.settle(holds)


def front_gain(inv=False):
    """sandwich matte gain: 1 = lanterns cover (forming / transition), 0 = in front (hold); inv=True the other way round."""
    a_, b_ = (0.0, 1.0) if inv else (1.0, 0.0)
    keys = [[0.0, a_]]
    for a, b in HOLDS:
        keys += [[a, b_], [b, a_]]
    return {"timeline": keys, "step": True}


def word_gate():
    """1 while a word is on (form .. exit), 0 in the transition gaps: the transition strokes live in both the matte and the line
    pass, so without this gate the letter outline (edges_in_type) paints them solid teal in front of everything (g02)."""
    keys = [[0.0, 1.0]]
    for w in PLAN:
        if w["gap"]:
            keys += [[w["gap"][0], 0.0], [w["gap"][1], 1.0]]
    return {"timeline": keys, "step": True}


def word_on(k, on=1.0, off=0.0):
    """step timeline: `on` while word k is on screen (start .. end of its exit), `off` elsewhere (its transition excluded)."""
    a, b = PLAN[k]["t0"], PLAN[k]["t1"]
    keys = [[0.0, on if a <= 0.0 else off]] + ([[a, on]] if a > 0.0 else []) + [[b, off]]
    return {"timeline": keys, "step": True}


def section(k, on=1.0, off=0.0):
    """step timeline: `on` from word k's start to word k+1's start (its transition belongs to it), `off` elsewhere."""
    a = PLAN[k]["t0"] if k else 0.0
    b = PLAN[k + 1]["t0"] if k + 1 < len(PLAN) else 99.0
    keys = [[0.0, on if k == 0 else off]]
    if k:
        keys.append([a, on])
    keys.append([b, off])
    return {"timeline": keys, "step": True}


def words_window():
    """contrast chain (face / scale / treatment never repeats on consecutive lines):
    LANTERN  tall condensed (Big Shoulders), frame-filling, WINDOW into the market on a near-black teal plate, lanterns glow through
    LIGHT    dot face (Doto: a string of lights), small, hung between the lanterns, solid warm fill on the full-colour market,
             behind the lanterns while forming, in front on the hold
    CARRY ME wide racing face (Tourney), frame-wide, WINDOW on a flat lantern-orange plate (the ground flips bright)
    HOME     heavy rounded (Erica One), mid scale low in frame, glowing outline only over the full-colour market, behind lanterns"""
    common = dict(module="lanes", form_in="lines", form_out="lines")
    tm = [{k: w[k] for k in ("t0", "t1", "form_s", "exit_s", "hit_on")} for w in PLAN]
    return [
        dict(common, text="LANTERN", lines=["LAN", "TERN"], font="Big Shoulders Display", **tm[0],
             placement={"yaw": -12, "bleed": 1.1}, axes={"wght": [800, 900, "beat"]}),
        dict(common, text="LIGHT", lines=["LIGHT"], font="Doto[ROND,wght].ttf", **tm[1],
             placement={"yaw": 14, "dy": -0.17, "bleed": 0.85}, axes={"ROND": 100, "wght": [600, 900, "beat"]}),
        dict(common, text="CARRY ME", lines=["CARRY", "ME"], font="Tourney[wdth,wght].ttf", **tm[2],
             placement={"yaw": -10, "bleed": 1.05}, axes={"wght": 900}),
        dict(common, text="HOME", lines=["HOME"], font="Erica One", **tm[3],
             placement={"yaw": 10, "dy": 0.12, "bleed": 0.85}),
    ]


ORANGE, PALE, TEALBLACK, WARM = [1.0, 0.5, 0.12], [1.0, 0.78, 0.5], [0.015, 0.06, 0.07], [1.0, 0.9, 0.72]


def ops_window():
    sys.path.insert(0, str(ROOT))
    from tools.fusion import recipe_eval
    o = recipe_eval.cam_ops(layers=("type_matte", "type_line"), gain=1.0, ref="$flow", prefix="cam_")
    wake = {"radius": 70, "decay": 0.9, "trail": 8, "trail_gain": 0.8, "bg_drag": 0.1, "max_disp": 0.15, "smooth": 1.0}
    for src, out in (("$cam_type_matte", "wm"), ("$cam_type_line", "wl")):
        o.append({"op": "flow_wake", "out": out, "args": dict(wake, type_a=src, flow="$flow", matte="$flow_m",
                                                               strength={"curve": "onset", "lo": 2.0, "hi": 5.0},
                                                               settle=settle(), frame_idx="$frame_idx", u="$u",
                                                               state_key=f"P12:{out}")})
    sm = {"matte_threshold": 0.5, "matte_soft": 0.08}
    o += [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 1.05, "gain": 0.92, "sat": 1.08}},
        {"op": "blur", "out": "fb", "args": {"img": "$footage", "sigma": 2.0}},
        {"op": "luminosity_key", "out": "lk", "args": {"img": "$fb", "low": 0.58, "high": 0.72}},   # glowing lanterns (solid)
        # section 1: near-black teal ground, the letters are windows into the market, the lanterns glow through the ground
        {"op": "knockout", "out": "kA", "args": {"plate_rgb": TEALBLACK, "type_a": "$wm", "footage": "$foot", "fg_matte": "$lk"}},
        {"op": "crossfade", "out": "image", "args": {"a": "$foot", "b": "$kA", "t": section(0)}},
        # section 3: the ground flips to flat lantern orange, the dot letters are windows (no lanterns: pure flat)
        {"op": "knockout", "out": "kC", "args": {"plate_rgb": ORANGE, "type_a": "$wm", "footage": "$foot"}},
        {"op": "crossfade", "out": "image", "args": {"a": "$image", "b": "$kC", "t": section(2)}},
        # section 2: solid warm fill hung between the lanterns: behind them while forming, in front on the hold
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$image", type_rgb=WARM, type_a="$wm", matte="$lk",
                                                         matte_gain=front_gain(), type_opacity=word_on(1))},
        # section 4: outline only, woven behind the lanterns, glowing
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$image", type_rgb=WARM, type_a="$wl", matte="$lk",
                                                         type_opacity=word_on(3))},
        # strands (line pass) everywhere else = the letters forming / unravelling / the transitions: lantern-glow threads,
        # always behind the lanterns, in the type's own line language (no flat fills)
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$image", type_rgb=PALE, type_a="$wl", matte="$lk",
                                                         type_opacity=word_on(3, on=0.0, off=0.85))},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$wl", "radius": 18,
                                                    "intensity": {"curve": "low", "lo": 0.1, "hi": 0.55}, "color": ORANGE}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.03, "sat": 1.06}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 12, "amount": 0.02, "size": 1.2}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.24}},
    ]
    return o


# ---- g03 line (the rendered draft design): fill behind lanterns / outline in front, flipped on the hold. g04 = the same on the
# new t2m song, Monoton (no longer liked) -> Ultra, faces alternate tall / puffy so consecutive lines contrast.
O3, T3, P3 = [1.0, 0.56, 0.16], [0.16, 0.85, 0.82], [1.0, 0.78, 0.5]


def words_g03():
    common = dict(module="lanes", form_in="lines", form_out="lines")
    tm = [{k: w[k] for k in ("t0", "t1", "form_s", "exit_s", "hit_on")} for w in PLAN]
    return [
        dict(common, text="LANTERN", lines=["LAN", "TERN"], font="Big Shoulders Display", **tm[0],
             placement={"yaw": -14, "dy": -0.08}, axes={"wght": [700, 900, "beat"]}),
        dict(common, text="LIGHT", lines=["LIGHT"], font="Ultra", **tm[1], placement={"yaw": 14, "dy": -0.06}),
        dict(common, text="CARRY ME", lines=["CARRY", "ME"], font="Big Shoulders Display", **tm[2],
             placement={"yaw": -14, "dy": -0.06}, axes={"wght": [700, 900, "beat"]}),
        dict(common, text="HOME", lines=["HOME"], font="Ultra", **tm[3], placement={"yaw": 10, "dy": -0.04}),
    ]


def ops_g03():
    sys.path.insert(0, str(ROOT))
    from tools.fusion import recipe_eval
    o = recipe_eval.cam_ops(layers=("type_matte", "type_line"), gain=1.0, ref="$flow", prefix="cam_")
    wake = {"radius": 70, "decay": 0.9, "trail": 8, "trail_gain": 0.8, "bg_drag": 0.1, "max_disp": 0.15, "smooth": 1.0}
    for src, out in (("$cam_type_matte", "wm"), ("$cam_type_line", "wl")):
        o.append({"op": "flow_wake", "out": out, "args": dict(wake, type_a=src, flow="$flow", matte="$flow_m",
                                                               strength={"curve": "onset", "lo": 2.0, "hi": 5.0},
                                                               settle=settle(), frame_idx="$frame_idx", u="$u",
                                                               state_key=f"P12:{out}")})
    sm = {"matte_threshold": 0.5, "matte_soft": 0.08}
    o += [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 1.05, "gain": 0.92, "sat": 1.08}},
        {"op": "blur", "out": "fb", "args": {"img": "$footage", "sigma": 2.0}},
        {"op": "luminosity_key", "out": "lk", "args": {"img": "$fb", "low": 0.58, "high": 0.72}},
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$foot", type_rgb=P3, type_a="$wl", matte="$lk", type_opacity=0.85)},
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$image", type_rgb=O3, type_a="$wm", matte="$lk",
                                                         matte_gain=front_gain(), type_opacity=0.96)},
        {"op": "edges_in_type", "out": "ol", "args": {"edges": "$wl", "type_a": "$wm", "spread": 3.0, "gain": word_gate()}},
        {"op": "sandwich", "out": "image", "args": dict(sm, footage="$image", type_rgb=T3, type_a="$ol", matte="$lk",
                                                         matte_gain=front_gain(inv=True), type_opacity=1.0)},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$ol", "radius": 22,
                                                    "intensity": {"curve": "low", "lo": 0.08, "hi": 0.6}, "color": T3}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$wm", "radius": 14,
                                                    "intensity": {"curve": "onset", "lo": 0.03, "hi": 0.2}, "color": O3}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.03, "sat": 1.06}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 12, "amount": 0.02, "size": 1.2}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.26}},
    ]
    return o


DESIGN = "window" if "--design-window" in sys.argv else "g03"   # window = the paused P10-style redesign (not rendered)
words = words_window if DESIGN == "window" else words_g03
ops = ops_window if DESIGN == "window" else ops_g03


def build():
    return {
        "id": "P12", "slug": "night-market", "fps": 24, "version": "v004",
        "audio": {"src": G.rel(A.song), "bpm": round(GRID["bpm"], 3), "analysis": G.analysis("P12", A.song, GRID, write=False),
                  "cut": GRID["cut"], "fade": 0.4},
        "type": {"seed": 12, "energy": 0.6, "mood": "warm", "tier": "mid",
                 "palette": {"bg": [10, 8, 14], "fg": [255, 150, 50], "accents": [[40, 215, 210], [255, 120, 40], [255, 200, 120]]},
                 "camera": "beat_orbit", "camera_params": {"read_dps": 4.0, "intro_deg": 30.0, "fast_dps": 40.0, "elev": 4.0, "bob": 0.8},
                 "transitions": ["fan", "spiral_around", "braid"], "line": {"px": 4.0}, "words": words()},
        "type_passes": ["matte", "line"],
        "stills": [{
            "id": "A", "model": "moss", "seed": 1201, "size": [1088, 1920],
            "negative": "text, letters, signage text, watermark, logo, caption, cartoon, illustration, 3d render, oversaturated, "
                        "deformed faces, extra limbs",
            "prompt": "night market street seen from street level looking straight down the middle of the street, many rows of glowing "
                      "orange paper lanterns hanging overhead on strings across the street at different heights, dark blue night sky "
                      "visible between the lanterns, steam rising from food stalls on both sides, crowd silhouettes walking in the "
                      "distance, shallow depth of field, warm colour, deep perspective, cinematic 35mm photograph, tall vertical frame"}],
        "shots": [{
            "id": "s1", "kind": "i2v", "first": "A", "frames": 289, "seed": 121, "size": [1088, 1920], "matte": False,
            "flow": True,
            "prompt": "one continuous slow forward dolly down the night market street, the camera glides straight ahead under the "
                      "hanging paper lanterns, the lanterns sway gently and pass overhead, steam drifts up from the food stalls, "
                      "crowd silhouettes stroll, warm glowing light, shallow depth of field, cinematic"}],
        "compose": {"size": [1080, 1920], "crossfade": 0, "crf": 16, "default": {"ops": ops()}, "shots": {}},
        "signoff": {"off": True},
        "parent": {"piece": None, "why": "v004 brief P12 + reel-017 visual decisions: one continuous dolly, type rides the camera, "
                                         "solid lantern matte, front/behind at once, beat-paced holds"},
    }


def widen(j):
    """16:9 version P12_w: its own MOSS still composed for wide (never a crop of the 9:16 still), its own single 289-frame dolly at
    16:9 (draft 960x544). Type / compose re-layout for landscape is pending (contour-bridge redesign), so the type block is the
    portrait plan for now and must not be rendered as is."""
    j = json.loads(json.dumps(j))
    j.update(id="P12_w", format="16:9")
    j["parent"] = {"piece": "P12", "why": "16:9 version (Both formats rule): own wide still + own wide dolly shot"}
    st = j["stills"][0]
    st.update(seed=1211, size=[1920, 1088],
              prompt="wide cinematic view down a long night market street at street level, many rows of glowing orange paper lanterns "
                     "strung across the street overhead at different heights, filling the upper half of the frame, food stalls with "
                     "rising steam along both sides, crowd silhouettes walking, the vanishing point on the right third of the frame, "
                     "more of the stalls and side alleys visible left and right, dark blue night sky between the lanterns, shallow "
                     "depth of field, warm colour, cinematic 35mm photograph, wide horizontal frame")
    sh = j["shots"][0]
    sh.update(size=[1920, 1088], seed=1221,
              prompt="one continuous slow forward dolly down the wide night market street, the camera glides straight ahead under "
                     "the hanging paper lanterns, the lanterns sway gently and pass overhead, steam drifts up from the food stalls "
                     "on both sides, crowd silhouettes stroll, warm glowing light, shallow depth of field, cinematic")
    j["compose"]["size"] = [1920, 1080]
    j["type"]["format_note"] = "PENDING landscape re-layout"
    return j


if __name__ == "__main__":
    if not A.no_audio:
        G.analysis("P12", A.song, GRID, write=True)
    out = HERE / ("P12_w.json" if A.wide else "P12.json")
    out.write_text(json.dumps(widen(build()) if A.wide else build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, "| song", G.rel(A.song), "| hits", [w["hit"] for w in PLAN], "| holds", HOLDS)
    if A.reset_tc or A.reset_compose:
        G.reset("P12_w" if A.wide else "P12", keep_type=A.reset_compose and not A.reset_tc)
