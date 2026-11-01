#!/usr/bin/env python
"""(runs ON THE BOX, with the ComfyUI python: /venv/main/bin/python) retime a DESIGNED piece to the music reading's landing times + put chosen words BEHIND the man.

    /venv/main/bin/python own_video_v2_retime_box.py --piece designed.json --words words.json --behind "MIDDLE WORD,THIRD WORD" --out piece_v2.json --report report.json

Why on the box and not in the graph: the design step's piece rules (qwen_piece.fit_to_shots) keep word k's hit inside shot k and within 2 s of the shot start, so a word the
reading lands at 9.31 s in a shot that starts at 7.29 s is moved to the beat 8.64 s, and the last guard (`legible`) clears the matte over any letter a still man covers for the whole
hold, so no word can sit behind him. This script takes the designed piece (the model's design, every piece rule applied) and changes only:

  1. type.words[i].t0 / t1 / form_s / hit_on : the repo's own timing solver (qwen_piece._compile, spec.plan_timing as the simulator) is run on hit = the reading's beat-snapped `land`,
     hold = the beats to the next word, exit 1 beat; the hits it reaches are re-read with spec.plan_timing and written to the report.
  2. everything derived from the word holds: the flow_wake settle timeline (plan_rules.settle_timeline) and plan.flow_wake.holds, from a dry run of the new timing.
  3. compose (--behind): the sandwich op of every chain gets args legible=False (the op's own opt-out of the guard) and front=$wfront, a mask_gate of the type alpha that is 1 outside the
     chosen words' windows and 0 inside them: the chosen words are drawn between the footage and the man's matte (man in front), every other word stays in front of him.
Nothing else is touched (fonts, modules, io styles, palette, cuts, camera are the model's).
"""
import argparse
import copy
import json
import math
import numpy as np
import os
import sys

ROOT = os.environ.get("MOGRAPH_ROOT", "/workspace/ComfyUI/custom_nodes/motion-graphic")
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--piece", required=True)
ap.add_argument("--words", required=True)
ap.add_argument("--behind", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--bpm", type=float, default=90.0)
ap.add_argument("--hold-beats", default="", help="'WORD=beats,..': cap a word's hold (the designed cut needs the word gone before its window)")
ap.add_argument("--dy", default="", help="'WORD=dy,..': placement.dy of a word (negative = up); the behind words sit at head height so only part of the letters is hidden")
ap.add_argument("--outline", type=float, default=3.0, help="dark outline (blur sigma px of the type alpha, x3 gain) on SOME words only, see --outline-share; 0 = off")
ap.add_argument("--outline-share", type=float, default=0.34, help="share of the words that get the outline: the busiest-footage ones (Laplacian variance of the picture behind the word over its hold), at least 1, "
                                                                  "so 6 words -> 2 (user 2026-10-30: 'we dont need to outline everything')")
ap.add_argument("--busy-clip", default="/workspace/own_video/v3/seg/global_clip.mp4", help="the window's frames: where the outline words are chosen")
ap.add_argument("--colors", default="", help="global clip mp4 (keyframes -> a 3-5 colour palette, k-means in Lab) : per-word ink colour from the picture instead of the design's cream")
ap.add_argument("--still-palette", default="/workspace/own_video/scratch_tree/tools/fusion/still_palette.py")
ap.add_argument("--color-seed", type=int, default=1)
ap.add_argument("--transitions", default="cut", choices=["cut", "keep"], help="cut = word-to-word plain cuts (v003); keep = the design's word-to-word transitions (strands travel A exit -> B entry): seeded unique non-cut names, "
                                                                           "strand-carrying io on every gap, the dry-run drift of the engine's timeline corrected in form_s")
ap.add_argument("--pure-type", type=int, default=1, help="1 (route default): the type is drawn with a plain `over` of the module's face (no sandwich), so recipe_eval.legibility_pass adds NO hold_net (its scrim / "
                                                          "plain-version blend is the soft dark plate behind the letters); 0 = the design's sandwich")
ap.add_argument("--work-dir", default="/workspace/own_video/v5", help="segments.json (pieces with a subject) for the face check")
ap.add_argument("--alpha-dir", default="/workspace/ComfyUI/output/projects/COMPETITIONS/motion-graphic/own-video/v003", help="Alpha-Gen PNG stacks of the pieces (face check)")
ap.add_argument("--alpha-stem", default="motion-graphic_own-video_v003")
ap.add_argument("--max-sections", type=int, default=99, help="facefix: at most this many flat-ground sections (a grid that would cover the face but gets no section falls to another layout)")
ap.add_argument("--face-limit", type=float, default=0.35, help="a layout covers the face when more than this share of the face rectangle is inside its box")
ap.add_argument("--layouts", default="", help="'facefix' (v006): per word, a grid of <= 4 rows only where its box does not cover the subject's face for the hold, else stack / rot / line, else a grid inside a flat-ground section (subject in front); 'grid3': at least 3 of the words in the `grid` layout (rows of 2-4 letters filling the frame, no-orphan rule), the rest line / stack, seeded (--color-seed); empty = the design's own")
ap.add_argument("--stable-type", default="/workspace/own_video/scratch_tree/tools/fusion/stable_type.py", help="stable_type.py: the box tree may not have it yet (then this copy is loaded; it only needs design_defaults / design_options)")
ap.add_argument("--glow", default="off", help="off (route default: NO add_glow behind the type, user 2026-10-30) | keep (the design's own glow) | a number = add_glow intensity")
ap.add_argument("--cut3d", default="", help="the ONE designed 3D cut kept, 'after=kind' (every other cut is hard); empty = leave the design's cuts")
ap.add_argument("--edit", default="redesign", choices=["overlay", "redesign"], help="overlay (the own-video default, tools/app/overlay_mode.py): after every other edit the PICTURE LOCK runs on the piece - all cuts hard, no holds / "
                                                                                 "crossfade, no grade / grain / vignette / halation / plate treatments / plate pans / multiplane / hold net, no plate keys - so only the type can change a picture pixel; "
                                                                                 "redesign (the script's own default, the old route) leaves the design's cuts and effects")
ap.add_argument("--native-size", default="", help="'WxH' of the SOURCE: piece.sizes / compose.size are written at this size (compose = type = shot = source, control = the matte size, see --matte-size)")
ap.add_argument("--matte-size", default="", help="'WxH' of the Alpha-Gen mattes / control size (multiple of 32); default = --native-size")
ap.add_argument("--aspect", type=float, default=0.0, help="width / height of the frame the layouts are fitted to (default 9/16); the 16:9 orientation is used when > 1")
a = ap.parse_args()
ASPECT = float(a.aspect) if a.aspect and a.aspect > 0 else 9.0 / 16.0
FMT = "16:9" if ASPECT > 1.0 else "9:16"

from tools.fusion import design_step as DS      # noqa: E402
from tools.fusion import plan_rules as PR       # noqa: E402
from tools.fusion import qwen_piece as QP       # noqa: E402
from tools.fusion import spec as SP             # noqa: E402
from tools.app import overlay_mode as OVM       # noqa: E402

piece = json.load(open(a.piece, encoding="utf-8"))
_want0 = json.load(open(a.words, encoding="utf-8"))
if len(piece["type"]["words"]) < len(_want0):      # ONE WORD PER SHOT (2026-11-02): the design makes one word per planned design shot (6), the words file can hold more (a word for every source shot with sung lyrics): a designed slot that joined several wanted words ("A. B.") is split back into them, every copy keeps the slot's design (face, module, colours are re-drawn below); a missing tail reuses the designs in turn
    _ws0, _split = piece["type"]["words"], []
    for _src, _txt in OVM.expand_designed([w["text"] for w in _ws0], [x["text"] for x in _want0]):
        _c = copy.deepcopy(_ws0[_src])
        _c["text"] = _txt
        _c.pop("lines", None)
        _split.append(_c)
    piece["type"]["words"] = _split
if len(piece["type"]["words"]) > len(_want0):      # Gemma split a two-word slot ("BLACK SAND") into two designed words: merge consecutive words back to the slot text (first word's design, t1 of the last)
    _merged, _i, _ws0 = [], 0, piece["type"]["words"]
    for _lw in _want0:
        _w, _j = dict(_ws0[_i]), _i + 1
        while _i < len(_ws0) and " ".join(str(x["text"]) for x in _ws0[_i:_j]).upper() != str(_lw["text"]).upper() and _j < len(_ws0):
            _j += 1
        if _j - _i > 1 and " ".join(str(x["text"]) for x in _ws0[_i:_j]).upper() == str(_lw["text"]).upper():
            _w["text"], _w["t1"] = _lw["text"], _ws0[_j - 1].get("t1", _w.get("t1"))
            _w.pop("lines", None)
        else:
            _j = _i + 1
        _merged.append(_w)
        _i = _j
    piece["type"]["words"] = _merged
piece["type"]["camera_params"] = dict(piece["type"].get("camera_params") or {}, read_dps=0.0, intro_deg=0.0, elev=0.0, bob=0.0)      # beat_orbit that never moves (stable_type.STATIC_CAMERA; the box repo has no stable_type yet): the landed word holds perfectly still
report_freeze = piece["compose"].get("holds")
piece["compose"]["holds"] = []                  # no freeze-frames: compose.holds stops the footage dead for ~0.5 s at each music dropout = repeated frames = stutter (user: "can we not get the video stuttering")
TRANS_POOL = ["circuit_board", "braid", "fan", "ripple", "spiral_around", "twist"]       # mograph/modules/transitions: the strand transitions of the type engine
NO_MATERIAL_IO = ("flicker", "wipe")
trans_report = None
if a.transitions == "cut":
    piece["type"]["transitions"] = ["cut"] * max(len(piece["type"]["words"]) - 1, 0)      # word-to-word: plain cuts (the engine turned twist / ripple into cuts with a note and ran its timeline up to 0.18 s later than the plan)
else:
    import random
    rng_t = random.Random(a.color_seed)
    old_t = list(piece["type"].get("transitions") if isinstance(piece["type"].get("transitions"), list) else [])
    nwords = len(piece["type"]["words"])
    new_t, used = [], set()
    for g in range(max(nwords - 1, 0)):
        cur = old_t[g] if g < len(old_t) else None
        if cur in TRANS_POOL and cur not in used:
            pick = cur
        else:                                                    # missing / cut / a name used before: drawn from the pool, never the neighbour's (the schema also forbids a name twice)
            left = [x for x in TRANS_POOL if x not in used and x != (new_t[-1] if new_t else None)]
            pick = rng_t.choice(left or TRANS_POOL)
        new_t.append(pick)
        used.add(pick)
    piece["type"]["transitions"] = new_t
    io_fix = []
    for g, wd in enumerate(piece["type"]["words"][:-1]):        # every gap must carry strands: a flicker / wipe side makes the engine put a cut there
        nxt = piece["type"]["words"][g + 1]
        if wd.get("form_out") in NO_MATERIAL_IO:
            io_fix.append((g, "form_out", wd.get("form_out"), "lines"))
            wd["form_out"] = "lines"
        if nxt.get("form_in") in NO_MATERIAL_IO:
            io_fix.append((g + 1, "form_in", nxt.get("form_in"), "lines"))
            nxt["form_in"] = "lines"
    trans_report = dict(design=old_t, final=new_t, io_made_strand_carrying=io_fix)
want = json.load(open(a.words, encoding="utf-8"))
layout_report = None
# ---- font variety (user R3 "a single font", R4 "similar font choices over the frames", 2026-11-01): no two consecutive words in the same face; the replacement is the
# seeded liked face the piece does not use yet (never a risky / barcode one), else the design's other faces; before the layouts are fitted, so every fit uses the final face
import random as _rnd_f
from tools.app import fonts_route as _FR
_wf = piece["type"]["words"]
EXPRESSIVE_ROLES = ("heavy", "tall", "ornamental", "decon")       # readable display faces; pixel / blackletter / fine / mono faces carry length and cap rules (design_defaults.pixel_problem) that undo a swap, `text` is the plain sans
_roles = {e["file"]: e.get("role") for e in _FR.liked_list()}
_pool = [e["file"] for e in _FR.liked_list() if not e.get("risky") and "barcode" not in e["file"].lower() and e.get("role") in EXPRESSIVE_ROLES]       # video mode always carries the expressive liked faces (user 2026-11-01): never the plain `text` sans (Space Grotesk); replacements come from EXPRESSIVE_ROLES
_rng_f = _rnd_f.Random(f"fonts:{a.color_seed}")
_own = [f for f in dict.fromkeys(w.get("font") for w in _wf) if f and _roles.get(f) != "text"]
_extra = _rng_f.sample(_pool, len(_pool))
font_fix = []
for _i in range(1, len(_wf)):
    _f = _wf[_i].get("font")
    if _f and _f == _wf[_i - 1].get("font"):
        _nx = _wf[_i + 1].get("font") if _i + 1 < len(_wf) else None
        _c = next((f for f in [x for x in _extra if x not in _own] + _own if f not in (_f, _nx)), None)       # a liked face the piece does not use yet first
        if _c:
            _wf[_i]["font"] = _c
            _wf[_i].pop("axes", None)
            font_fix.append((_wf[_i].get("text"), _f, _c))
for _i in range(len(_wf)):
    _f = _wf[_i].get("font")
    if _f and _roles.get(_f) == "text":
        _pv = _wf[_i - 1].get("font") if _i else None
        _nx = _wf[_i + 1].get("font") if _i + 1 < len(_wf) else None
        _c = next((f for f in [x for x in _extra if x not in _own] + _own if f not in (_f, _pv, _nx)), None)
        if _c:
            _wf[_i]["font"] = _c
            _wf[_i].pop("axes", None)
            font_fix.append((_wf[_i].get("text"), _f, _c))
# ---- module variety (overlay, user R16b review e5 / e8: "ordinary looking text just over the top", "pretty static"; all four words had been `lanes`): every word gets its own approved Module Gallery
# word module (16_app/MODULES_MENU.md section A) so the entrances differ; the first `lanes` + `lines` word keeps lanes (the e1 look the user liked). The pool is the reading-first modules plus the
# effect-rich ones that act in the form-in / exit and settle for the hold (design_defaults.READING_FIRST / EFFECT_MODULES; never the echo / extruded / two-style ones, AVOID_MODULES). Seeded; no neighbour shares a module.
MODULE_POOL = OVM.module_pool(("slats", "gradient_fade", "staircase", "beat_bars", "particles", "outflow"))       # 2026-11-02: `particles` (a dot-cloud letter) is banned everywhere since 2026-11-01 (design_defaults.SPECKLE_MODULES); this pool re-introduced it on the own-video route (a word of a box render came out as a dot cloud)
module_fix = []
if a.edit == "overlay":
    _rng_m = _rnd_f.Random(f"modules:{a.color_seed}")
    _keep0 = next((i for i, w in enumerate(_wf) if w.get("module") == "lanes" and w.get("form_in") == "lines"), next((i for i, w in enumerate(_wf) if w.get("module") == "lanes"), None))
    _used = {_wf[_keep0].get("module")} if _keep0 is not None else set()
    _order = _rng_m.sample(MODULE_POOL, len(MODULE_POOL))
    for _i, _w in enumerate(_wf):
        if _i == _keep0:
            continue
        _m0 = _w.get("module")
        _nb = {_wf[_j].get("module") for _j in (_i - 1, _i + 1) if 0 <= _j < len(_wf)}
        if _m0 in MODULE_POOL and _m0 not in _used and _m0 not in _nb:
            _used.add(_m0)
            continue
        _c = next((m for m in _order if m not in _used and m not in _nb), None) or next((m for m in _order if m not in _nb), None)
        if _c:
            _w["module"] = _c
            _used.add(_c)
            module_fix.append((_w.get("text"), _m0, _c))
speckle_fix = OVM.mend_speckle(piece)            # the same plan-time swap as image mode (text_guard.fix_plan): a speckle module / io style the DESIGN or the pool above left is replaced; idempotent
if speckle_fix:
    print("speckle swap:", "; ".join(speckle_fix), flush=True)
if a.layouts:
    import importlib.util
    import random
    try:
        from tools.fusion import stable_type as ST
        st_src = "box tree"
    except ImportError:
        spl = importlib.util.spec_from_file_location("stable_type_overlay", a.stable_type)
        ST = importlib.util.module_from_spec(spl)
        sys.modules["stable_type_overlay"] = ST
        spl.loader.exec_module(ST)
        st_src = a.stable_type
    wl = piece["type"]["words"]
    rng_l = random.Random(f"layouts:{a.color_seed}")
    facefix = a.layouts == "facefix"
    ok_grid = [i for i, w in enumerate(wl) if "grid" in ST.allowed(str(w["text"]))]
    grid_idx = set(rng_l.sample(ok_grid, min(max(3, len(wl) // 2), len(ok_grid)))) if not facefix else set()
    pins = []
    for i, w in enumerate(wl):
        if i in grid_idx:
            pins.append("grid")
        else:
            pins.append("stack" if len(str(w["text"]).split()) > 1 else "line")
    lay_fixes: list = []
    if not facefix:
        ST.enforce(piece["type"], {"layouts": pins, "seed": a.color_seed}, a.bpm, lay_fixes, fmt=FMT)
    got = []
    for i, w in enumerate(wl):
        bl = ST.best_layout(str(w["text"]), w.get("font"), w.get("axes"), pins[i], FMT, ASPECT)
        got.append(dict(text=w["text"], pinned=pins[i], layout=bl["layout"], lines=bl["lines"], cap=round(bl["cap"], 3), ok=bool(bl["ok"]), orphan_free=bool(ST.orphan_free(bl["lines"], str(w["text"]))) if bl.get("split") else True))
    layout_report = dict(stable_type=st_src, words=got, grid_words=sum(1 for g in got if g["layout"] in ("grid", "column")), fixes=lay_fixes)
    if facefix:
        layout_report = None          # decided after the timing is solved (needs the hold frames), see below
ana = DS.ana_from_piece(piece)
b = 60.0 / a.bpm
ws = piece["type"]["words"]
assert len(ws) == len(want), (len(ws), len(want))
grid = PR.make_grid(ana, float(ana["cut"]))
report = {"words": [], "freeze_holds_removed": report_freeze}


def solve(hold_beats, form_beats, exit_beats):
    t = {"words": []}
    for i, (w, lw) in enumerate(zip(ws, want)):
        d = {k: v for k, v in w.items() if k not in ("t0", "t1", "hit_on", "_D", "_h0")}
        d.update(hit=round(lw["land"], 4), hold_s=round(hold_beats[i] * b, 3), form_s=round(form_beats[i] * b, 3), exit_s=round(exit_beats[i] * b, 3))
        t["words"].append(d)
    fixes: list = []
    done, end = QP._compile(t, ana, fixes, 0.0)
    return done, end, fixes


def reached(done):
    """the hit / release / end spec.plan_timing gives the solved words, and the words' own start S."""
    probe = {"type": {"words": [{k: v for k, v in d.items() if not k.startswith("_")} for d in done]}}
    PR.fill_type_timing(probe, ana)
    tm = SP.plan_timing(copy.deepcopy(probe["type"]["words"]), probe["type"], grid)
    return tm


n = len(ws)
for kv in [x for x in a.dy.split(",") if x.strip()]:
    nm, v = kv.rsplit("=", 1)
    for w in ws:
        if w["text"] == nm.strip():
            w.setdefault("placement", {})["dy"] = float(v)
form_b = [2] + [1] * (n - 1)
exit_b = [OVM.EXIT_BEATS] * n
# ---- WORD LIFETIME (user 2026-11-02: "why does a word stay on screen for more than its single shot?" / "It can overlap a bit if the transition needs time but not that long."):
# the hold used to run to the NEXT WORD (beats to the next land - 3), so a word sung in a 2 s shot stayed up for 10 s. Now a word lives inside the shot it is assigned to (words.json shot_a / shot_b,
# pipeline seconds = source frame / 24): form-in, hold and exit end by the shot's end cut plus at most EXIT_OVERLAP_S (the exit / transition handing over to the next shot). Only a shot too short to hold
# the readable minimum (max(spec.min_hold, spec.HOLD_FLOOR 1.2 s)) after the landing extends to the first following cut it can finish by (+ the same pad), never to the next word. No shot info
# (an old words.json): the hold is the slot's own hold_until - land, floor the readable minimum. Shots without a word stay clean. The solver below can only SHORTEN a hold to reach the next word's beat.
EXIT_OVERLAP_S = OVM.EXIT_OVERLAP_S            # longest a word may run past its shot's end cut (pipeline s); also capped at one beat
HOLD_QUANT = OVM.HOLD_QUANT                    # holds in quarter beats
cuts_p = sorted({float(c) for lw in want for c in (lw.get("cuts_p") or [])})


def need_s(i):
    return max(SP.min_hold(ws[i]), SP.HOLD_FLOOR)


def hold_beats_for(i, exit_s):
    lw = want[i]
    return OVM.word_hold_s(float(lw["land"]), need_s(i), exit_s, lw.get("shot_b"), cuts_p, b, lw.get("hold_until")) / b


HB_MIN = [OVM.quarter_beats(need_s(i), b) for i in range(n)]
hold_b = [hold_beats_for(i, exit_b[i] * b) for i in range(n)]
for kv in [x for x in a.hold_beats.split(",") if x.strip()]:
    nm, v = kv.rsplit("=", 1)
    for i, w in enumerate(ws):
        if w["text"] == nm.strip():
            hold_b[i] = int(v)
for attempt in range(8):
    done, end, fixes = solve(hold_b, form_b, exit_b)
    tm = reached(done)
    bad = [i for i in range(n) if abs(tm[i]["hit"] - want[i]["land"]) > 0.03]
    report["attempt_%d" % attempt] = dict(hold_beats=list(hold_b), bad=bad, hits=[round(x["hit"], 3) for x in tm], fixes=[f for f in fixes if "not reachable" in f])
    if not bad:
        break
    i = bad[0]
    if i == 0 or hold_b[i - 1] <= HB_MIN[i - 1] + 1e-9:
        form_b[i] = min(form_b[i] + 1, 3) if tm[i]["hit"] < want[i]["land"] else max(form_b[i] - 1, 1)
        if form_b[i] == 1 and tm[i]["hit"] > want[i]["land"] and i > 0 and hold_b[i - 1] > HB_MIN[i - 1] + 1e-9:
            hold_b[i - 1] = max(HB_MIN[i - 1], hold_b[i - 1] - 1)
    else:
        hold_b[i - 1] = max(HB_MIN[i - 1], hold_b[i - 1] - 1)
for i, (w, d) in enumerate(zip(ws, done)):
    for k in ("t0", "t1", "form_s", "hit_on"):
        w[k] = d[k]
    w["exit_s"] = round(exit_b[i] * b, 2)
tm = reached(done)
if a.transitions == "keep" and trans_report is not None:
    # a strand transition between two words takes ~2.6-2.9 s of its own (measured on the dry run: a word after a `circuit_board` / `ripple` gap landed 1.5-3 s after its beat, out of its shot): where the planned gap is shorter
    # than that, the word-to-word transition is a plain cut (the picture is never touched, this is the type engine's own join). Wider gaps keep their strands.
    cut_gaps = []
    for g_ in range(n - 1):
        if tm[g_ + 1]["S"] - tm[g_]["end"] < OVM.TRANSITION_MIN_GAP_S - 1e-6:
            piece["type"]["transitions"][g_] = "cut"
            cut_gaps.append(g_)
    trans_report["final"] = list(piece["type"]["transitions"])
    trans_report["cut_where_tight"] = cut_gaps
flat_sections = []
if a.layouts == "facefix":
    import cv2
    segs_f = json.load(open(os.path.join(a.work_dir, "segments.json"), encoding="utf-8"))
    cache_f = {}

    def face_rect(j):
        sg = next((x for x in segs_f if x["a"] <= j < x["b"] and x["person"]), None)
        if sg is None:
            return None
        pth = f"{a.alpha_dir}/{a.alpha_stem}_alpha_h{sg['job']}_image_{j - sg['a'] + 1:05d}_.png"
        m_ = cache_f.get(pth)
        if m_ is None:
            m_ = cache_f[pth] = cv2.imread(pth, 0)
        if m_ is None:
            return None
        ys, xs = np.where(m_ > 127)
        if len(ys) < 200:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h_, w_ = y1 - y0, x1 - x0
        cx = 0.5 * (x0 + x1)
        return (cx - 0.25 * w_) / m_.shape[1], y0 / m_.shape[0], (cx + 0.25 * w_) / m_.shape[1], (y0 + 0.25 * h_) / m_.shape[0]

    def covered(box, fr):
        ix0, iy0, ix1, iy1 = max(box[0], fr[0]), max(box[1], fr[1]), min(box[2], fr[2]), min(box[3], fr[3])
        return max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0) / (max(fr[2] - fr[0], 1e-6) * max(fr[3] - fr[1], 1e-6))
    FN_ROWS = frozenset("a an the and or but nor so of to in on at for with from by as into onto upon over under about than then if that this my your his our their its is are was were be am i'm".split())
    pins_f, face_rows = [], []
    for i, w in enumerate(ws):
        h0, h1 = int(tm[i]["hit"] * 24), int(tm[i]["rel"] * 24)                  # pipeline frame = source frame
        faces = [(j, face_rect(j)) for j in range(h0, h1, 2)]
        have = [(j, f) for j, f in faces if f is not None]
        cands = {}
        for lay in ST.allowed(str(w["text"])):
            bl = ST.fit_box(str(w["text"]), w.get("font"), w.get("axes"), lay, ASPECT)
            ov = [covered(bl["box"], f) for _, f in have]
            cands[lay] = dict(rows=len(bl["lines"]), lines=bl["lines"], cap=round(float(bl["cap"]), 3), ok=bool(bl["ok"]), face_overlap_mean=round(float(np.mean(ov)), 3) if ov else 0.0,
                              face_frames_covered=round(float(np.mean([o > a.face_limit for o in ov])), 3) if ov else 0.0)
        st_ = cands.get("stack")
        if st_ and len(cands) > 1 and len(str(w["text"]).split()) > 1 and any(len(r_.split()) == 1 and r_.lower().strip(".,;:!?&-'\"") in FN_ROWS for r_ in st_["lines"]):
            cands.pop("stack")                                                                 # a stack that leaves a small function word (ON / THE / IN ...) alone on a row reads as a fragment: another layout carries the phrase
        order = [c for c in ("grid", "stack", "rot90", "line") if c in cands]
        pick, why = None, ""
        g = cands.get("grid")
        if g and g["rows"] <= 4 and g["ok"] and g["face_frames_covered"] <= 0.35:             # 1. the grid (<= 4 rows, reads) where it keeps off the face
            pick, why = "grid", "grid, <= 4 rows, off the face"
        if pick is None and g and g["rows"] <= 4 and g["ok"] and have and len(flat_sections) < a.max_sections:                       # 2. the grid over the face -> a flat-ground section: the subject stands IN FRONT of the letters
            js = [j for j, _ in have]
            runs_, cur_ = [], [js[0]]
            for j_ in js[1:]:                                                                # contiguous stretches of frames with the subject (inserts split them)
                if j_ - cur_[-1] <= 2:
                    cur_.append(j_)
                else:
                    runs_.append(cur_)
                    cur_ = [j_]
            runs_.append(cur_)
            best_ = max(runs_, key=len)
            a_, b_ = best_[0], best_[-1] + 2
            if b_ - a_ >= 24:
                pick, why = "grid", "grid over the face -> flat-ground section (window_subject_front) on the frames with the subject"
                flat_sections.append(dict(word=w["text"], k=i, a=int(a_), b=int(b_)))
        if pick is None:                                                                     # 3. another layout that reads and keeps off the face
            for c in [c for c in order if c != "grid"]:
                d = cands[c]
                if d["ok"] and d["face_frames_covered"] <= 0.35:
                    pick, why = c, "reads, off the face (a grid would cover the face and no section fits)"
                    break
        if pick is None:                                                                     # 4. near the cap floor (>= 8.5 %) and off the face: the biggest
            near = [c for c in order if c != "grid" and cands[c]["cap"] >= 0.085 and cands[c]["face_frames_covered"] <= 0.35]
            if near:
                pick = max(near, key=lambda c: cands[c]["cap"])
                why = "near the cap floor, off the face (grid: > 4 rows or over the face with no room for a section)"
        if pick is None:                                                                     # 5. nothing keeps the face free: the biggest non-grid layout, flagged
            rest = [c for c in order if c != "grid"] or order
            pick = max(rest, key=lambda c: cands[c]["cap"])
            why = "FLAGGED: no layout is both readable and off the face (grid needs > 4 rows or has no section); the biggest non-grid layout"
        pins_f.append(pick)
        face_rows.append(dict(text=w["text"], hold_frames=[h0, h1], frames_with_subject=len(have), layout=pick, why=why, candidates=cands))
    lay_fixes_f: list = []
    ST.enforce(piece["type"], {"layouts": pins_f, "seed": a.color_seed}, a.bpm, lay_fixes_f, fmt=FMT)
    for i, w in enumerate(ws):                  # enforce() falls back to the biggest-cap layout when the pinned one is under the cap floor (THIRD WORD stack -> a 5-row grid): the pick stands
        bl = ST.fit_box(str(w["text"]), w.get("font"), w.get("axes"), pins_f[i], ASPECT)
        pl_ = w.setdefault("placement", {})
        if len(str(w["text"]).split()) > 1 or bl.get("split"):
            w["lines"] = list(bl["lines"])
        else:
            w.pop("lines", None)
        if bl.get("split"):
            w["ar_min"] = 0.85
        else:
            w.pop("ar_min", None)
        pl_["roll"], pl_["wf"], pl_["hf"] = bl["roll"], bl["wf"], bl["hf"]
        lay_fixes_f.append(f"type.words[{i}] layout {pins_f[i]} kept: {'/'.join(bl['lines'])}, cap ~{bl['cap']:.1%}")
    layout_report = dict(stable_type=st_src, mode="facefix", words=face_rows, grid_words=sum(1 for r_ in face_rows if r_["layout"] in ("grid", "column")), flat_sections=flat_sections, fixes=lay_fixes_f)
from tools.fusion import user_layout as _UL                       # 2026-11-01: after the facefix / grid3 layouts, EVERY piece: "more legible text" re-fits the layout (upright / bigger cap); else at most ONE rare mid-word split (about 1 piece in 4, seeded by --color-seed), the other grid words become whole words
_ul_fixes: list = []
if _UL.apply(piece, _ul_fixes, seed=a.color_seed):
    print("user legibility layout:", "; ".join(_ul_fixes), flush=True)
chk = PR.dry_run(piece)
if a.transitions == "keep":
    # the engine's own timeline (dry run) runs later than plan_timing where a transition really carries: shorten that word's form so its hit is back on the reading's beat (hold gets
    # longer by the same amount, release and end stay, so nothing else moves)
    corr = []
    for it in range(2 * len(ws) + 3):                      # one word per pass, the EARLIEST one off its beat (2026-11-02: correcting every word at once oscillated - a word's form moves every later hit - and left hits 1-3 s late, so words drifted out of their shots)
        errs = [float(cw["hit"]) - lw["land"] for cw, lw in zip(chk["words"], want)]
        corr.append([round(e, 3) for e in errs])
        bad_k = next((k for k, e in enumerate(errs) if abs(e) >= 0.02), None)
        if bad_k is None:
            break
        f_new = round(max(0.45, float(ws[bad_k]["form_s"]) - errs[bad_k]), 3)
        ws[bad_k]["t1"] = round(float(ws[bad_k]["t1"]) + (f_new - float(ws[bad_k]["form_s"])), 3)      # the word's window moves with its form: the NEXT word's gap (t0 - t1) stays what the plan made it (a stale t1 inflated the gap and pushed every later hit 1-3 s late)
        ws[bad_k]["form_s"] = f_new
        chk = PR.dry_run(piece)
    trans_report["dry_hit_error_per_pass"] = corr
holds = PR.word_holds(piece, ana, chk)
spans = PR.word_spans(piece, ana, chk)
for i, (w, lw) in enumerate(zip(ws, want)):
    report["words"].append(dict(text=w["text"], land=lw["land"], plan_hit=round(tm[i]["hit"], 3), plan_rel=round(tm[i]["rel"], 3), plan_end=round(tm[i]["end"], 3), plan_S=round(tm[i]["S"], 3),
                                dry_hit=round(holds[i][0], 3), dry_rel=round(holds[i][1], 3), dry_start=round(spans[i][0], 3), dry_end=round(spans[i][1], 3), t0=w["t0"], t1=w["t1"],
                                form_s=w["form_s"], exit_s=w["exit_s"], hit_on=w["hit_on"],
                                shot_a=lw.get("shot_a"), shot_b=lw.get("shot_b"),
                                start_before_shot_s=None if lw.get("shot_a") is None else round(spans[i][0] - float(lw["shot_a"]), 3),
                                end_past_shot_s=None if lw.get("shot_b") is None else round(spans[i][1] - float(lw["shot_b"]), 3)))
report["word_lifetime"] = dict(exit_overlap_s=EXIT_OVERLAP_S, hold_floor_s=SP.HOLD_FLOOR, rule="a word lives in its shot: ends by the shot's end cut + min(EXIT_OVERLAP_S, 1 beat); a shot too short for the readable hold runs to the first later cut it can finish by")
report["speckle_swap"] = speckle_fix
report["last_end"] = chk["last_end"]
report["layouts"] = layout_report
report["transitions"] = trans_report
report["transition_gaps"] = [dict(after=ws[g]["text"], to=ws[g + 1]["text"], kind=(piece["type"]["transitions"][g] if isinstance(piece["type"].get("transitions"), list) else None),
                                  a_end=report["words"][g]["dry_end"], b_start=report["words"][g + 1]["dry_start"], b_hit=report["words"][g + 1]["dry_hit"]) for g in range(len(ws) - 1)]
report["notes"] = chk.get("notes")

def chains_all():
    return ([piece["compose"]["default"]] if piece["compose"].get("default") else []) + list((piece["compose"].get("shots") or {}).values())


# ---- route default: no dissolving 3D cut. orbit:push / orbit:fly dissolve the cards of the shot (depth_orbit.py hole0 / hole1) and the background card is the frame with the subject REMOVED and
# push-pull filled (depth_orbit.subject_split): for ~1.5 s the singer is a blurred, filled silhouette (v002 r1, 13.7 s). Only a deliberate plate_in_subject moment may show that.
dissolve = [c for c in piece["compose"].get("cuts") or [] if str(c.get("cut", "")).lower() in ("orbit:push", "orbit:fly")]
for c in dissolve:
    c["why"] = f"{c.get('cut')} -> hard (own-video route default: the subject must not dissolve into a blurred fill outside a contrast moment); " + str(c.get("why", ""))
    c["cut"] = "hard"
report["dissolving_cuts_made_hard"] = [c.get("after") for c in dissolve]

# ---- cuts: one designed 3D cut where the words are clear of it, every other shot cut hard
if a.cut3d:
    after, kind = a.cut3d.split("=")
    for c in piece["compose"].get("cuts") or []:
        c["cut"] = kind if c.get("after") == after else "hard"
        if c.get("after") == after:
            c["why"] = "the one designed 3D cut of the piece: the words before it are gone, the hook word forms into it"
        elif "params" in c:
            c.pop("params")
def _subst(obj, old, new):
    """replace the buffer reference "$old" by "$new" in every string of an args structure."""
    if isinstance(obj, dict):
        return {k: _subst(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, old, new) for v in obj]
    return new if obj == old else obj


glow_removed = 0
if a.glow == "off":
    for ch in chains_all():
        ops = ch.get("ops", [])
        k = 0
        while k < len(ops):
            o = ops[k]
            if o["op"] == "add_glow":                       # drop the op, its readers read what it read (the image under the glow)
                src, out = o["args"].get("img"), "$" + str(o.get("out", "image"))
                for j in range(k + 1, len(ops)):
                    ops[j]["args"] = _subst(ops[j]["args"], out, src)
                if o.get("out", "image") == "image":          # the chain's final image: the op before now writes it
                    ops[k - 1]["out"] = "image"
                del ops[k]
                glow_removed += 1
                continue
            k += 1
    for key in ("glow_lo", "glow_hi"):
        if key in piece["compose"]:
            piece["compose"][key] = 0.0
elif a.glow != "keep":
    for ch in chains_all():
        for o in ch.get("ops", []):
            if o["op"] == "add_glow":
                o["args"]["intensity"] = float(a.glow)
report["glow"] = dict(mode=a.glow, add_glow_ops_removed=glow_removed)
report["cuts"] = [[c.get("after"), c.get("cut")] for c in piece["compose"].get("cuts") or []]
report["font_variety"] = font_fix
report["module_variety"] = module_fix

# ---- READABLE BEHIND (user 2026-11-02: "why are most of the text neither fully over or fully behind the main subject - we need moments like these to read the text correctly?"). Decided BEFORE the
# colours (the colour of a word depends on where it sits and on which pixels its letters are over). Per word, on the real letter shapes (PIL glyphs of the word's face / rows / cap at the
# placement) and the subject's Alpha-Gen matte (holes filled, as the compose draws it) over the word's hold: BEHIND only where every glyph keeps >= 70 % and the word >= 85 % visible
# (overlay_mode.VIS_RULE) in all but 25 % of the hold; else the smallest nudge that passes while the subject still overlaps it; else FRONT (whole, for the whole hold).
import cv2 as _cv2
_segs = json.load(open(os.path.join(a.work_dir, "segments.json"), encoding="utf-8")) if os.path.isfile(os.path.join(a.work_dir, "segments.json")) else []
_mc = {}


def _subject_mask(j):
    sg = next((x for x in _segs if x["a"] <= j < x["b"] and x.get("person")), None)
    if sg is None:
        return None
    pth = f"{a.alpha_dir}/{a.alpha_stem}_alpha_h{sg['job']}_image_{j - sg['a'] + 1:05d}_.png"
    if pth not in _mc:
        m_ = _cv2.imread(pth, 0)
        _mc[pth] = None if m_ is None else _cv2.resize(m_, (max(m_.shape[1] // 4, 1), max(m_.shape[0] // 4, 1)), interpolation=_cv2.INTER_AREA).astype(np.float32) / 255.0
    return _mc[pth]


def _rect_of_placement(w):
    pl = w.get("placement") or {}
    wf_, hf_ = float(pl.get("wf", 0.9)), float(pl.get("hf", 0.4))
    cx_, cy_ = 0.5 + float(pl.get("dx", 0.0) or 0.0), 0.5 + float(pl.get("dy", 0.0) or 0.0)
    return (cx_ - wf_ / 2, cy_ - hf_ / 2, cx_ + wf_ / 2, cy_ + hf_ / 2)


names = [s_.strip() for s_ in a.behind.split(",") if s_.strip()]
_wrect = [_rect_of_placement(w) for w in ws]          # the ink box of each word (frame fractions): the pixels the colour is chosen against
_behind = []
_vplan = []
if a.edit == "overlay":
    from tools.app import overlay_mode as OMB
    try:
        from tools.fusion import stable_type as _STB
    except ImportError:
        _STB = None
    _pins = globals().get("pins_f")
    _AH = 256
    _AW = int(round(_AH * ASPECT))
    _sc = []
    for i, w in enumerate(ws):
        h0, h1 = int(holds[i][0] * 24), int(holds[i][1] * 24)                       # pipeline frame = source frame
        occ = []
        for j in range(h0, max(h1, h0 + 1), 2):
            m_ = _subject_mask(j)
            occ.append(None if m_ is None else OMB.solid_mask(_cv2.resize(m_, (_AW, _AH), interpolation=_cv2.INTER_AREA), close=2))
        roll_ = float((w.get("placement") or {}).get("roll", 0.0) or 0.0)
        lay_ = (_pins[i] if _pins else None) or ("rot90" if roll_ > 45 else "rot-90" if roll_ < -45 else ("stack" if len(w.get("lines") or []) > 1 else "line"))
        try:
            bl_ = _STB.fit_box(str(w["text"]), w.get("font"), w.get("axes"), lay_, ASPECT)
            rows_, cap_ = list(bl_["lines"]), float(bl_["cap"])
        except Exception:
            rows_ = list(w.get("lines") or [str(w["text"])])
            cap_ = max(float((w.get("placement") or {}).get("hf", 0.3)) / max(len(rows_), 1) / 1.12 * 0.9, 0.05)
        cx0 = 0.5 + float((w.get("placement") or {}).get("dx", 0.0) or 0.0)
        cy0 = 0.5 + float((w.get("placement") or {}).get("dy", 0.0) or 0.0)
        st_ = OMB.glyph_stack(str(w["text"]), w.get("font"), w.get("axes"), rows_, cap_, (_AW, _AH), lay_, (cx0, cy0))
        res = OMB.plan_word(st_, occ, cap_)
        res["layout"], res["cap"], res["rows"] = lay_, round(cap_, 4), rows_
        res["stack"] = st_
        _vplan.append(res)
        _sc.append(dict(cover=round(1.0 - res["verdict"]["word_mean"], 4), overlap=res["overlap"], frames=len(occ), ok=res["status"] in ("behind", "nudged"), status=res["status"]))
    _named = [i for i, w in enumerate(ws) if w["text"] in names]
    _behind = _named or OMB.choose_behind(_sc)
    for i, w in enumerate(ws):
        r_ = _vplan[i]
        if i in _behind and r_["status"] == "nudged":                                # the nudge is kept only for a word that really goes behind; the others stay centred, in front
            pl_ = w.setdefault("placement", {})
            pl_["dx"], pl_["dy"] = round(float(pl_.get("dx", 0.0) or 0.0) + r_["dx"], 4), round(float(pl_.get("dy", 0.0) or 0.0) + r_["dy"], 4)
            if r_["scale"] != 1.0:
                pl_["wf"], pl_["hf"] = round(float(pl_.get("wf", 0.9)) * r_["scale"], 4), round(float(pl_.get("hf", 0.3)) * r_["scale"], 4)
            r_["stack"] = OMB.shift_stack(r_["stack"], r_["dx"], r_["dy"], r_["scale"])
        bb_ = OMB.stack_bbox(r_["stack"])
        if bb_ is not None:
            _share = (float(bb_[2]) - float(bb_[0])) * (float(bb_[3]) - float(bb_[1]))
            r_["bbox_share"] = round(_share, 4)
            if _share > OMB.MAX_BBOX_SHARE:                                           # a short word set 90 deg fills the frame: type over MAX_TYPE_SHARE of a frame is dropped as "broken" (restore_frame_ex), so the word vanished for its whole hold (FIVE, job 20261101-203838-e264) -> scale it down to the cap
                _s = (OMB.MAX_BBOX_SHARE / _share) ** 0.5
                _pl = w.setdefault("placement", {})
                _pl["wf"], _pl["hf"] = round(float(_pl.get("wf", 0.9)) * _s, 4), round(float(_pl.get("hf", 0.3)) * _s, 4)
                r_["stack"] = OMB.shift_stack(r_["stack"], 0.0, 0.0, _s)
                bb_ = OMB.stack_bbox(r_["stack"])
                r_["shrunk"] = round(_s, 4)
            _wrect[i] = tuple(float(v) for v in bb_)
    report["readable_behind"] = dict(rule=dict(OMB.VIS_RULE), behind=[ws[i]["text"] for i in _behind], words=[
        dict(text=ws[i]["text"], status=("front" if i not in _behind else _vplan[i]["status"]), found=_vplan[i]["status"], nudge=[_vplan[i]["dx"], _vplan[i]["dy"], _vplan[i]["scale"]] if i in _behind else [0.0, 0.0, 1.0],
             layout=_vplan[i]["layout"], bbox_share=_vplan[i].get("bbox_share"), shrunk=_vplan[i].get("shrunk"), overlap=_vplan[i]["overlap"], at_centre=_vplan[i]["base"], at_final=_vplan[i]["verdict"] if i in _behind else _vplan[i]["base"]) for i in range(len(ws))])
    for r_ in _vplan:
        r_.pop("stack", None)

# ---- type colour from the picture (user 2026-10-30: "why is the text all white? colour inspired by the surroundings or contrasted with it")
if a.colors:
    import colorsys
    import importlib.util
    import random
    import cv2
    spc = importlib.util.spec_from_file_location("still_palette_box", a.still_palette)
    SPAL = importlib.util.module_from_spec(spc)
    spc.loader.exec_module(SPAL)
    cap = cv2.VideoCapture(a.colors)
    gfr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        gfr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    keys = gfr[::15]
    mosaic = np.concatenate([cv2.resize(f, (48, 88), interpolation=cv2.INTER_AREA) for f in keys], axis=1) if False else np.concatenate([cv2.resize(f, (48, 88), interpolation=cv2.INTER_AREA) for f in keys], axis=1)
    pal = SPAL.extract(mosaic, k=5, seed=a.color_seed)
    rng = random.Random(a.color_seed)

    def fit_contrast(rgb, bgm, floor):
        """the palette colour with its Lab lightness moved (hue / chroma kept) until its WCAG contrast against the local background is >= floor; None when it cannot."""
        lab = np.array(SPAL.rgb_to_lab(np.asarray(rgb, float).reshape(1, 3))).reshape(3)
        up = SPAL.luminance(bgm) < 0.4
        for L in (np.arange(lab[0], 97, 4) if up else np.arange(lab[0], 6, -4)):
            c = [int(v) for v in np.asarray(SPAL.lab_to_rgb(np.array([L, lab[1], lab[2]]))).reshape(3)]
            if SPAL.contrast(c, bgm) >= floor:
                return c
        return None

    def lab_ab_dist(c, bgm):
        la = np.array(SPAL.rgb_to_lab(np.asarray(c, float).reshape(1, 3))).reshape(3)
        lb = np.array(SPAL.rgb_to_lab(np.asarray(bgm, float).reshape(1, 3))).reshape(3)
        return float(np.hypot(la[1] - lb[1], la[2] - lb[2]))
    picks = []
    # 2026-11-01 colour review: the ink is a hue of the frames UNDER THAT WORD (legible_color.shot_hues; the user's palette wins), the previous word's hue sent to
    # the back of the order, lightness searched until it reaches 4.5 (then 3.0) over the band behind the word; the outline is the complementary hue (support_color).
    from tools.fusion import legible_color as LCX
    from tools.fusion import user_style as USX
    ust = USX.parse_direction(piece)
    prev_h = None
    used_h = []                  # the hues of the last inks: a hue within LCX.LC_USED_NEAR of one goes behind the fresh ones (variety across shots)
    import zlib as _zl
    sup_base = _zl.crc32(str(piece.get("id", "piece")).encode()) % 97      # outline family varies word to word (legible_color.support_pick); a user palette ignores it
    for i, w in enumerate(ws):
        h0, h1 = int(holds[i][0] * 24), int(holds[i][1] * 24)
        idx = [j for j in range(h0, min(h1, len(gfr)), max(1, (h1 - h0) // 8))] or [min(h0, len(gfr) - 1)]
        H_, W_ = gfr[0].shape[:2]
        sub = np.stack([cv2.resize(gfr[j], (96, int(96 * H_ / W_)), interpolation=cv2.INTER_AREA) for j in idx]).astype(np.float32) / 255.0
        hs_ = sub.shape[1]
        # 2026-11-02: the subject's pixels are left out of the shot's hues (its skin / leather is not the shot's colour) and a behind word is measured only over the pixels its letters can show on
        sm_ = [_subject_mask(j) for j in idx]
        have_m = all(m_ is not None for m_ in sm_)
        subj = np.stack([cv2.resize(m_, (96, hs_), interpolation=cv2.INTER_AREA) for m_ in sm_]) if have_m else None
        rx0, ry0, rx1, ry1 = _wrect[i]
        rx0, rx1 = max(int(rx0 * 96), 0), min(max(int(np.ceil(rx1 * 96)), int(rx0 * 96) + 1), 96)
        ry0, ry1 = max(int(ry0 * hs_), 0), min(max(int(np.ceil(ry1 * hs_)), int(ry0 * hs_) + 1), hs_)
        reg = sub[:, ry0:ry1, rx0:rx1]
        vis_px = np.ones(reg.shape[:3], bool)
        if have_m and i in _behind:
            vis_px = subj[:, ry0:ry1, rx0:rx1] < 0.5
            if vis_px.mean() < 0.15:
                vis_px = np.ones(reg.shape[:3], bool)
        bgm = reg[vis_px].mean(0) * 255.0
        hist_w = LCX._hist(LCX.rel_lum(LCX.srgb_to_lin(reg[vis_px])).reshape(-1), np.ones(int(vis_px.sum())))
        plate_ok = (subj < 0.5) if have_m else None
        if plate_ok is not None and plate_ok.mean() < 0.08:
            plate_ok = None
        subj_h = [h_ for h_ in LCX.shot_hues(LCX.srgb_to_lin(sub), valid=(subj >= 0.5)) if h_["c"] >= 0.04][:2] if have_m and (subj >= 0.5).mean() > 0.01 else None      # the subject's own 1-2 coloured hues (skin, a coloured jacket); [] = none found
        if ust["palette"]:
            w_hues, skin_h, n_fresh = LCX.order_hues(LCX.hues_of_colors(ust["palette"]), prev_h), [], 0
        else:
            w_hues, skin_h, n_fresh = LCX.prefer_hues(LCX.shot_hues(LCX.srgb_to_lin(sub), valid=plate_ok), subj_h, used=used_h[-2:])
        lad = ((4.5 * 1.2, 0.85),) + LCX.LC_INK_LADDER if ust["legible_boost"] else LCX.LC_INK_LADDER
        ch_, info_ = LCX.pick_ink(hist_w, w_hues, skin_h, ring=None, ladder=lad, n_fresh=n_fresh)
        if ch_ is not None:
            chosen = [int(round(float(v) * 255)) for v in LCX.lin_to_srgb(ch_["lin"])]
            prev_h = ch_["hue"]
            used_h.append(prev_h)
            sup_ = LCX.support_color(ch_["lin"], w_hues, light=float(LCX.rel_lum(ch_["lin"])) <= 0.18, pick=sup_base + i)
            w["outline_rgb"] = [round(float(v), 4) for v in LCX.lin_to_srgb(sup_)]
            w["echo_rgb"] = [round(float(v), 4) for v in LCX.lin_to_srgb(LCX.echo_color(ch_["lin"], sup_, LCX.srgb_to_lin(np.asarray(bgm, np.float32) / 255.0)))]
            acc3 = SPAL.padded_accents(pal, 3)
            w["palette"] = {"bg": [int(v) for v in pal["dark"]], "fg": chosen, "accents": [[int(v) for v in c] for c in acc3]}
            picks.append(dict(text=w["text"], mode="shot_hue", hue=round(prev_h, 1), hues=[x["h"] for x in w_hues], skin_or_subject_hues=[x["h"] for x in skin_h], rgb=chosen, outline=w["outline_rgb"],
                              local_bg=[int(v) for v in bgm], contrast=round(float(SPAL.contrast(chosen, bgm)), 2), user_palette=bool(ust["palette"]), rung=info_["rung"], target=info_["target"], need=info_["need"],
                              from_subject_hue=bool(info_["skin"]), pick=info_.get("pass_"), subject_excluded=bool(plate_ok is not None), measured_on="visible pixels of the word box" if (have_m and i in _behind) else "the word box",
                              placed="behind" if i in _behind else "front"))
            continue
        def _pal_not_subject(c):          # the palette is the whole clip's k-means (skin + leather + wood in this job): a colour of the subject's own hue / the generic skin band is not a shot colour
            lab_ = np.array(SPAL.rgb_to_lab(np.asarray(c, float).reshape(1, 3))).reshape(3)
            hh_ = float(np.degrees(np.arctan2(lab_[2], lab_[1])) % 360.0)
            cc_ = float(np.hypot(lab_[1], lab_[2])) / 150.0
            if subj_h:
                return not any(LCX.hue_dist(hh_, s_["h"]) < LCX.LC_SUBJ_NEAR for s_ in subj_h)
            return not LCX.is_skin_hue(hh_, max(cc_, 0.05))
        pal_ok = [c for c in pal["colors"] if _pal_not_subject(c)] or list(pal["colors"])
        acc = [c for c in pal_ok[2:]] or [c for c in pal_ok] or [pal["light"]]
        sur = min(acc, key=lambda c: lab_ab_dist(c, bgm))                                  # harmonious: the palette accent nearest the surroundings' hue, lightness moved to read
        chrom = [c for c in pal_ok if lab_ab_dist(c, SPAL.lab_to_rgb(np.array([SPAL.rgb_to_lab(np.asarray(c, float).reshape(1, 3))[0][0], 0.0, 0.0])).reshape(3)) > 12.0]
        con = max(chrom or pal_ok, key=lambda c: lab_ab_dist(c, bgm))               # contrasted: the palette colour with the hue FARTHEST from the surroundings (their complement), lightness moved to read
        order = ["surround", "contrast"] if rng.random() < 0.5 else ["contrast", "surround"]
        chosen, how = None, None
        for mode in order:
            base_c = sur if mode == "surround" else con
            c = fit_contrast(base_c, bgm, 4.5) or fit_contrast(base_c, bgm, 3.0)
            if c is not None and SPAL.contrast(c, bgm) >= 3.0:
                chosen, how = c, mode
                break
        if chosen is None:
            chosen, how = list(max([pal["dark"], pal["light"]], key=lambda c: SPAL.contrast(c, bgm))), "fallback_dark_or_light"
        acc3 = SPAL.padded_accents(pal, 3)
        w["palette"] = {"bg": [int(v) for v in pal["dark"]], "fg": [int(v) for v in chosen], "accents": [[int(v) for v in c] for c in acc3]}
        w["outline_rgb"] = [round(float(v), 4) for v in LCX.lin_to_srgb(LCX.support_color(LCX.srgb_to_lin(np.asarray(chosen, np.float32) / 255.0), w_hues,
                                                                                            light=SPAL.luminance(chosen) <= 0.18, pick=sup_base + i))]      # the fallback ink still gets a shot-hue outline
        w["echo_rgb"] = [round(float(v), 4) for v in LCX.lin_to_srgb(LCX.echo_color(LCX.srgb_to_lin(np.asarray(chosen, np.float32) / 255.0),
                                                                              LCX.srgb_to_lin(np.asarray(w["outline_rgb"], np.float32)), LCX.srgb_to_lin(np.asarray(bgm, np.float32) / 255.0)))]
        picks.append(dict(text=w["text"], mode=how, drawn_order=order, outline=w["outline_rgb"], rgb=[int(v) for v in chosen], local_bg=[int(v) for v in bgm], contrast=round(float(SPAL.contrast(chosen, bgm)), 2)))
    piece["type"]["palette"] = {"bg": [int(v) for v in pal["bg"]], "fg": [int(v) for v in pal["fg"]], "accents": [[int(v) for v in c] for c in SPAL.padded_accents(pal, 3)]}
    piece["compose"]["legible_color"] = False       # the engine's own recolour stage would pull every word back to white / dark: the colours above already meet the 3:1 floor against the picture behind the word
    report["colors"] = dict(palette=pal["colors"], seed=a.color_seed, words=picks)

# ---- derived: flow_wake settle follows the holds the dry run reports
chains = [piece["compose"]["default"]] if piece["compose"].get("default") else []
chains += list((piece["compose"].get("shots") or {}).values())
for ch in chains:
    for o in ch.get("ops", []):
        if o.get("op") == "flow_wake":
            o["args"]["settle"] = {"ease": "smooth", **PR.settle_timeline(holds)} if "ease" in (o["args"].get("settle") or {}) else PR.settle_timeline(holds)
piece.setdefault("plan", {})["flow_wake"] = {"chains": len(chains), "holds": [[round(x, 3), round(y, 3)] for x, y in holds]}

# ---- route default (v007): the archive carries Alpha-Gen mattes, never depth frames, so a design chain that draws the type with `depth_occlude` ($depth -> compose s1: "$s1_depth needs depth
# frames") is turned into the matte form of the same drawer: upscale_matte -> sandwich (solid matte, the ghost type_opacity of the depth cover is dropped). The pure-type pass below then
# writes the sandwich as a plain `over` of the module face like every other chain.
n_dep = 0
for ch in chains_all():
    ops_ = ch["ops"]
    for k, o in enumerate(ops_):
        if o["op"] == "depth_occlude":
            ar = o["args"]
            mname = "m_dep"
            ops_[k:k + 1] = [{"op": "upscale_matte", "out": mname, "args": {"guide": ar["footage"], "matte": "$matte", "size": list(piece["compose"].get("size") or [360, 640])}},
                             {"op": "sandwich", "out": o["out"], "args": {"footage": ar["footage"], "type_rgb": ar.get("type_rgb", "$tcol"), "type_a": ar["type_a"], "matte": "$" + mname,
                                                                          "matte_threshold": 0.5, "matte_soft": 0.08}}]
            n_dep += 1
            break
report["depth_occlude_to_sandwich"] = n_dep

# ---- behind the man
names = [s.strip() for s in a.behind.split(",") if s.strip()]
if names:
    idx = [i for i, w in enumerate(ws) if w["text"] in names]
    win = []
    for i in idx:                                   # the chosen words' windows; windows less than 0.8 s apart are one window (the gate must not blip up between two neighbours)
        s0, e0 = spans[i]
        if win and s0 - win[-1][1] < 0.8:
            win[-1][1] = e0
        else:
            win.append([s0, e0])
    tl = [[0.0, 1.0]]
    for s0, e0 in win:
        tl += [[round(max(s0 - 0.25, tl[-1][0] + 0.01), 3), 1.0], [round(s0 - 0.05, 3), 0.0], [round(e0 + 0.05, 3), 0.0], [round(e0 + 0.25, 3), 1.0]]
    gate = {"ease": "smooth", "timeline": tl}
    nch = 0
    for ch in chains:
        ops = ch["ops"]
        for k, o in enumerate(ops):
            if o["op"] == "sandwich":
                ta = o["args"]["type_a"]
                ops.insert(k, {"op": "mask_gate", "out": "wfront", "args": {"a": ta, "gate": gate, "scale": 4.0}})
                o["args"]["front"] = "$wfront"
                o["args"]["legible"] = False
                nch += 1
                break
    # the hold net (recipe_eval.legibility_pass: hold_net) draws the plain word back over the picture wherever the matte hides more than ~15 % of its letters during a hold, and the
    # `legible` guard clears the matte over letters hidden for a whole hold: both would undo "behind". The piece-level opt-out is compose.beauty_face = false (it also drops the
    # beauty face + legible_color recolour, so the type is the flat matte silhouette tinted from the plate: push the tint light so it still reads).
    piece["compose"]["beauty_face"] = False
    for ch in chains:
        for o in ch["ops"]:
            if o["op"] == "tint_type":
                o["args"].update(lum_target=0.94, lum_amount=0.9, sat=0.5)
    report["behind"] = dict(words=[ws[i]["text"] for i in idx], gate=gate, sandwiches=nch)
if a.outline > 0:
    import cv2
    cap_o = cv2.VideoCapture(a.busy_clip)
    gray = []
    while True:
        ok_, f_ = cap_o.read()
        if not ok_:
            break
        gray.append(cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY))
    cap_o.release()
    busy = []
    for i in range(len(ws)):
        h0, h1 = int(holds[i][0] * 24), int(holds[i][1] * 24)           # pipeline frame = source frame
        idx_ = list(range(h0, min(h1, len(gray)), max(1, (h1 - h0) // 8))) or [min(h0, len(gray) - 1)]
        H_o = gray[0].shape[0]
        busy.append(float(np.mean([cv2.Laplacian(gray[j][int(0.30 * H_o):int(0.70 * H_o)], cv2.CV_64F).var() for j in idx_])))
    n_out = max(1, int(len(ws) * a.outline_share))
    chosen = sorted(sorted(range(len(ws)), key=lambda i: -busy[i])[:n_out])
    from tools.fusion import recipe_eval as _REo
    o_w = _REo.outline_weights(piece, len(ws))                 # 2026-11-01: per-word outline weight (thin .. bold), seeded; colour = the word's complementary hue
    tl_o = [[0.0, 0.0]]
    for i in chosen:
        s0, e0 = spans[i]
        tl_o += [[round(max(s0 - 0.15, tl_o[-1][0] + 0.01), 3), 0.0], [round(s0, 3), 1.0], [round(e0, 3), 1.0], [round(e0 + 0.15, 3), 0.0]]
    gate_o = {"ease": "smooth", "timeline": tl_o}
    for ch in chains_all():
        ops = ch["ops"]
        for k, o in enumerate(ops):
            if o["op"] == "sandwich":
                ta = o["args"]["type_a"]
                base_o, add = o["args"]["footage"], []
                for n_, i in enumerate(chosen):                 # one halo per outlined word: its own weight and colour, gated to its span
                    s0, e0 = spans[i]
                    g_i = {"ease": "smooth", "timeline": [[0.0, 0.0], [round(max(s0 - 0.15, 0.01), 3), 0.0], [round(s0, 3), 1.0], [round(e0, 3), 1.0], [round(e0 + 0.15, 3), 0.0]]}
                    add += [{"op": "blur", "out": f"ow0_{n_}", "args": {"img": ta, "sigma": round(float(a.outline) * (o_w[i] if i < len(o_w) else 1.0), 2)}},
                            {"op": "mask_gate", "out": f"ow_{n_}", "args": {"a": f"$ow0_{n_}", "gate": g_i, "scale": 3.0}},
                            {"op": "over", "out": "olay", "args": {"base": base_o, "fg": list(ws[i].get("outline_rgb") or [0.02, 0.02, 0.02]), "alpha": f"$ow_{n_}"}}]
                    base_o = "$olay"
                ops[k:k] = add
                o["args"]["fg"] = o["args"]["footage"]          # the man in front of the word is cut from the clean picture, the outline halo never lands on him
                o["args"]["footage"] = "$olay"
                break
    report["outline"] = dict(sigma=a.outline, share=a.outline_share, words=[ws[i]["text"] for i in chosen], busyness={ws[i]["text"]: round(busy[i], 1) for i in range(len(ws))}, gate=gate_o,
                             weights={ws[i]["text"]: o_w[i] for i in chosen if i < len(o_w)}, colours={ws[i]["text"]: ws[i].get("outline_rgb") for i in chosen})
if a.pure_type:
    # PURE TYPE: the module's beauty face is drawn by recipe_eval.legibility_pass only when the chain has a sandwich, and the same pass then appends `hold_net` whose scrim / plain-version
    # blend is the soft dark plate behind the letters (compose.hold_net, recipe_eval.py ~658). So the chain the pass would have built is written out here WITHOUT the net and WITHOUT the
    # `legible` guard: type_face -> tmat / tface, flow_wake + its face twin, and the sandwich replaced by a plain `over` of the face colour (no sandwich = nothing for the pass to wrap).
    import copy as _copy
    from tools.fusion import recipe_eval as RE

    def _sub(x):
        if isinstance(x, dict):
            return {k: _sub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sub(v) for v in x]
        return "$tmat" if x == "$type_matte" else x
    bg_ = RE._beauty_bg(piece, None)
    pure = 0

    def USX_plain(p_):
        from tools.fusion import user_style as _US
        return _US.parse_direction(p_)["plain"]
    for ch in chains_all():
        if not any(o["op"] == "sandwich" for o in ch["ops"]):
            continue
        new_ops = [{"op": "type_face", "out": "_tf4", "args": dict({"beauty": "$type_beauty", "type_a": "$tmat0", "bg": bg_} if bg_ else {"beauty": "$type_beauty", "type_a": "$tmat0"})},
                   {"op": "rgba_part", "out": "tmat", "args": {"rgba": "$_tf4", "part": "a"}},
                   {"op": "rgba_part", "out": "tface", "args": {"rgba": "$_tf4", "part": "rgb"}}]
        new_ops[0]["args"]["type_a"] = "$type_matte"
        face_of = {"tmat": "tface"}
        wake_src = {}
        ECHO_BACK = 0.0 if USX_plain(piece) else 0.4
        for o in ch["ops"]:
            o = _copy.deepcopy(o)
            o["args"] = _sub(o["args"])
            if o["op"] == "flow_wake":
                new_ops.append(o)
                tw = _copy.deepcopy(o)
                tw["out"] = "_%s4" % o["out"]
                tw["args"]["type_rgb"] = "$tface"
                tw["args"]["state_key"] = str(o["args"].get("state_key", "default")) + ":face"
                new_ops += [tw, {"op": "rgba_part", "out": "%s_face" % o["out"], "args": {"rgba": "$" + tw["out"], "part": "rgb"}}]
                face_of[o["out"]] = "%s_face" % o["out"]
                src_ = str(o["args"].get("type_a", "")).lstrip("$")
                if src_ == "tmat":
                    wake_src[o["out"]] = "tmat"
                continue
            if o["op"] == "hold_life" and str(o["args"].get("img", "")).lstrip("$") in wake_src:
                new_ops.append(o)                                   # the crisp front copy gets the same hold life (same `ref`)
                fr = _copy.deepcopy(o)
                fr["out"], fr["args"]["img"] = o["out"] + "_front", "$" + wake_src[str(o["args"]["img"]).lstrip("$")]
                new_ops.append(fr)
                wake_src[o["out"]] = fr["out"]
                continue
            if o["op"] == "sandwich":
                ar = o["args"]
                ta = ar["type_a"]
                base_t = ar["footage"]
                if str(ta).lstrip("$") in wake_src:
                    # 2026-11-01 (user R1: "the echo needs to have a more prominent version of the text in front while the echo is more subdued at the back"): the wake at
                    # ECHO_BACK opacity in its own face, then the crisp un-displaced word fully opaque on top
                    new_ops += [{"op": "mask_gate", "out": "_wake_back", "args": {"a": ta, "gate": ECHO_BACK}},
                                {"op": "over", "out": "_wake_lay", "args": {"base": base_t, "alpha": "$_wake_back",           # tinted: the first word's echo colour (legible_color.echo_color), else the face
                                                                            "fg": next((list(w_["echo_rgb"]) for w_ in ws if w_.get("echo_rgb")), "$" + face_of.get(str(ta).lstrip("$"), "tface"))}}]
                    base_t, ta = "$_wake_lay", "$" + wake_src[str(ta).lstrip("$")]
                new_ops.append({"op": "over", "out": o["out"] + "_t" if ar.get("matte") else o["out"], "args": {"base": base_t, "fg": "$" + face_of.get(str(ta).lstrip("$"), "tface"), "alpha": ta}})
                if ar.get("matte"):     # the sandwich's job survives: the SUBJECT goes back over the type (footage over the type layer by the subject matte), only the scrim / legible wrap is gone
                    new_ops.append({"op": "over", "out": o["out"], "args": {"base": "$" + o["out"] + "_t", "fg": ar["fg"] if isinstance(ar.get("fg"), str) else ar["footage"], "alpha": ar["matte"]}})   # the outline pass sets `fg` = the clean picture: the halo never lands on the subject (restore: type never over the subject)
                pure += 1
                continue
            new_ops.append(o)
        ch["ops"] = new_ops
    report["pure_type"] = dict(chains_rewritten=pure, why="beauty face kept, no hold_net scrim, no legible guard, no sandwich (so recipe_eval.legibility_pass does not wrap the chain)")
if a.edit == "overlay":
    # ---- TYPE BEHIND THE SUBJECT (user R16 review e5 / e6, 2026-11-01: "none of the text is being masked between the layers"). Per word: front or behind the subject (the Alpha-Gen matte). About half of the
    # words (at least 1) go behind: the ones whose box the subject covers a part of for most of the hold. The runtime floor still decides: a behind word the subject hides too much of
    # (overlay_mode.BEHIND_FLOOR) comes in front whole. Entry / exit of a front word stay behind the subject (the guard only clears the hold + a 0.25 s ramp).
    # (the front / behind decision, the nudges and the scores were made above, before the colours: `_behind`, `_sc`, report["readable_behind"])
    _solid = OMB.solid_occluder(piece, a.native_size and [int(x) for x in a.native_size.lower().split("x")] or None)       # the occluder is the subject's solid Alpha-Gen matte, never a luminosity key
    _rep_b = OMB.apply_behind(piece, [(float(x), float(y)) for x, y in holds], _behind, 24, spans=[(float(x), float(y)) for x, y in spans])
    report["overlay_behind"] = dict(words=[ws[i]["text"] for i in _behind], front=[ws[i]["text"] for i in range(len(ws)) if i not in _behind], scores={ws[i]["text"]: dict(_sc[i], score=round(OMB.behind_score(_sc[i]), 3)) for i in range(len(ws))},
                                    chains=_rep_b["chains"], floor=_rep_b["floor"], front_floor=_rep_b["front_floor"], solid_occluder=_solid,
                                    why="subject matte over the type: behind words pass the readable rule (every glyph >= 70 %, word >= 85 % visible) else come in front whole, front words are in front for the whole hold")
    if _rep_b["chains"] == 0:
        print("WARNING: apply_behind found no subject-over-type draw to guard (the occluder is not a matte)", flush=True)
    # ---- PICTURE LOCK (edit "overlay", user 2026-11-01: "no transitions and no change to my edit, just text animating over and behind, linked to the lyrics"). Runs LAST, on the finished chains
    # (pure type / outline / behind already written): cuts hard, no holds / crossfade, no footage-touching op, no plate keys, sizes = the source size (tools/app/overlay_mode.py).
    from tools.app import overlay_mode as OM
    _nat = None
    if a.native_size:
        _nw, _nh = (int(x) for x in a.native_size.lower().split("x"))
        _mw, _mh = (int(x) for x in (a.matte_size or a.native_size).lower().split("x"))
        _nat = dict(piece=dict(control=[_mw, _mh], shot=[_nw, _nh], type=list(OM.type_canvas(_nw, _nh)), compose=[_nw, _nh]), format="16:9" if _nw > _nh else "9:16")      # type = a multiple-of-64 canvas (the type engine's rule); compose cover-fits it back to the source size
    report["overlay_lock"] = OM.lock_piece(piece, _nat)
    report["overlay_violations"] = OM.picture_violations(piece)
    report["word_frames"] = [dict(text=w["text"], hit_frame=int(round(holds[i][0] * 24)), release_frame=int(round(holds[i][1] * 24)), start_frame=int(round(spans[i][0] * 24)), end_frame=int(round(spans[i][1] * 24)))
                             for i, w in enumerate(ws)]            # pipeline frame = source frame (stretched clock: t * 24 = real t * source fps)
    for i, wf_ in enumerate(report["word_frames"]):                  # WORD LIFETIME, the start: a word forms IN its shot. The engine's first word starts at t = 0 and any word can start a little early; the frames of its form-in before its shot's first frame show the source (job: restore_video inactive_frames)
        sa_ = want[i].get("shot_a")
        if sa_ is not None:
            lo_ = max(int(round(spans[i][0] * 24)), (int(round(spans[i - 1][1] * 24)) + 1) if i else 0)       # never the previous word's own exit
            hi_ = int(round(float(sa_) * 24)) - 1
            if lo_ <= hi_:
                wf_["suppress_before"] = [lo_, hi_]
    if report["overlay_violations"]:
        raise SystemExit("overlay lock left picture-touching parts: " + "; ".join(report["overlay_violations"]))
report["edit"] = a.edit
json.dump(piece, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
json.dump(report, open(a.report, "w", encoding="utf-8"), indent=1)
print(json.dumps(report["words"], indent=0)[:3000])
print("behind", report.get("behind"))
