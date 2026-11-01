"""shared helpers of the v004 hand pieces P11 / P13 / P15 (_gen_P11.py etc.): analysis json from the known score grid, timelines
sampled from python functions (beat pulses, gates), the flow_wake weave. Pure python, no ComfyUI.

    from _v004_lib import grid, analysis_for, tl, pulse, ramp_keys
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AUD = ROOT / "15_fusion" / "v004" / "audio"
FPS = 24


def song_paths(pid, song=None, grid_path=None):
    """(wav, grid json) for a piece: explicit CLI paths, else 15_fusion/v004/audio/t2m/<pid>.wav + <pid>_grid.json when present
    (the ACE text-to-music songs), else the s1 (PICKS.md: s2) score songs."""
    if song or grid_path:
        wav = Path(song) if song else Path(str(grid_path).replace("_grid.json", ".wav"))
        gp = Path(grid_path) if grid_path else wav.with_name(wav.stem + "_grid.json")
    elif (AUD / "t2m" / f"{pid}_grid.json").exists():
        wav, gp = AUD / "t2m" / f"{pid}.wav", AUD / "t2m" / f"{pid}_grid.json"
    else:
        seed = "s1"
        picks = AUD / "PICKS.md"
        if picks.exists() and f"{pid} s2" in picks.read_text(encoding="utf-8"):
            seed = "s2"
        wav, gp = AUD / f"{pid}_{seed}.wav", AUD / f"{pid}_{seed}_grid.json"
    wav = wav if wav.is_absolute() else (ROOT / wav)
    gp = gp if gp.is_absolute() else (ROOT / gp)
    return wav.resolve(), gp.resolve()


def cli_song(argv):
    """--song X.wav / --grid X_grid.json from argv -> (song, grid, rest)."""
    song = gridp = None
    rest, it = [], iter(argv)
    for a in it:
        if a == "--song":
            song = next(it)
        elif a == "--grid":
            gridp = next(it)
        else:
            rest.append(a)
    return song, gridp, rest


def load_grid(gp):
    return json.loads(Path(gp).read_text(encoding="utf-8"))


def retime(g, design_bpm, design_offset=0.0, anchor=None):
    """the generators' times are written on a DESIGN grid (beat = 60/design_bpm, beat 0 at design_offset). -> (at, dur): at(t) maps a
    design time onto the same beat position of grid g (interpolated on its real beats, extrapolated past them); dur(d) scales a
    duration by the beat-length ratio. On the design grid itself both are the identity."""
    import numpy as np
    iv0 = 60.0 / float(design_bpm)
    bs = np.asarray(g["beats"], float)
    iv = float(np.median(np.diff(bs))) if len(bs) > 1 else 60.0 / float(g["bpm"])

    # bar phase: design beat 0 is a downbeat; shift by the index of the grid's first downbeat (mod 4, nearest) so design downbeats
    # land on the grid's downbeats (t2m songs can start on beat 4)
    db0 = float(g["downbeats"][0]) if g.get("downbeats") else float(bs[0])
    k0 = int(np.argmin(np.abs(bs - db0))) % 4
    shift = k0 if k0 <= 2 else k0 - 4
    if anchor is not None:                       # anchor = design time that must land on the grid's first downbeat >= 1 s
        dbs = [i for i in range(len(bs)) if i % 4 == k0 and bs[i] >= 1.0]
        shift = dbs[0] - int(round((float(anchor) - design_offset) / iv0))

    def at(t):
        u = (float(t) - design_offset) / iv0 + shift
        if u <= len(bs) - 1:
            return round(float(np.interp(u, np.arange(len(bs)), bs)) if u >= 0 else bs[0] + u * iv, 4)
        return round(float(bs[-1] + (u - (len(bs) - 1)) * iv), 4)

    def dur(d):
        return round(float(d) * iv / iv0, 4)
    return at, dur


def analysis_for(wav, gp):
    """<grid stem>_analysis.json next to the grid = the known grid (exact beats / downbeats, no beat tracking) + band curves of the
    song, in the shape piece_runner._audio reads (bar_len, beats, downbeats, bpm, key.estimate, curves, chord_changes)."""
    wav, gp = Path(wav), Path(gp)
    out = gp.with_name(gp.name.replace("_grid.json", "") + "_analysis.json")
    g = json.loads(gp.read_text(encoding="utf-8"))
    if not out.exists() or out.stat().st_mtime < max(gp.stat().st_mtime, wav.stat().st_mtime):
        import soundfile as sf
        from tools.fusion import music_analysis as MA
        x, sr = sf.read(str(wav), always_2d=True)
        n = int(math.ceil(len(x) / sr * FPS))
        curves = {"fps": FPS, "n": n}
        curves.update(MA.band_curves(x, sr, FPS, n))
        ana = {"bpm": g["bpm"], "bar_len": g["bar_len"], "beats": g["beats"], "downbeats": g["downbeats"],
               "key": {"estimate": g.get("key", "C major")}, "chord_changes": g.get("chord_changes", []), "curves": curves,
               "src": "grid " + gp.name, "sections": g.get("sections", [])}
        out.write_text(json.dumps(ana), encoding="utf-8")
    return rel(out)


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def retime_obj(obj, at, keys=("timeline",)):
    """map the [t, v] keys of every {"timeline": ..} in a nested structure through at()."""
    if isinstance(obj, dict):
        if "timeline" in obj and isinstance(obj["timeline"], list):
            obj["timeline"] = [[at(k[0]), k[1]] for k in obj["timeline"]]
        for v in obj.values():
            retime_obj(v, at)
    elif isinstance(obj, list):
        for v in obj:
            retime_obj(v, at)
    return obj


BOX = dict(host="<box-ip>", port=16013, py="/venv/main/bin/python", root="/workspace/mograph")


def camtrack(pid, size=(540, 960), sid="s1", fetch=False):
    """per-frame camera track [[x, y], ..] (compose px) of the shot's footage flow, cached as _<pid>_camtrack.json next to the
    generators. fetch=True computes it on the box (recipe_eval.camera_track on out/<pid>/flow/<sid>_flow.npy, the same function
    the cam ops use) once the runner's flow stage has run. -> list or None."""
    import subprocess
    p = HERE / f"_{pid}_camtrack.json"
    if p.exists() and not fetch:
        return json.loads(p.read_text(encoding="utf-8"))
    if not fetch:
        return None
    code = ("import numpy as np, json; from tools.fusion import recipe_eval as RE; "
            f"f=np.load('{BOX['root']}/out/{pid}/flow/{sid}_flow.npy', mmap_mode='r'); "
            f"print(json.dumps([[round(float(a),2), round(float(b),2)] for a, b in RE.camera_track(f, {tuple(size)})]))")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", str(BOX["port"]), f"root@{BOX['host']}",
                        f"cd {BOX['root']} && {BOX['py']} -c \"{code}\""], capture_output=True, text=True, timeout=300)
    tr = json.loads(r.stdout.strip().splitlines()[-1])
    p.write_text(json.dumps(tr), encoding="utf-8")
    return tr


def cam_rebase(track, hits, bounds, fps=FPS, ramp=0.3):
    """f(t) -> (dx, dy) that cancels the camera track's value at each word's hit: with translate(cam) followed by translate(f), every
    word rides the scene around its hit but is AT its layout position at the hit (no cumulative drift off frame). bounds = times
    between words (mid-gap) where the offset ramps from one word's value to the next."""
    def val(t):
        i = min(max(int(round(t * fps)), 0), len(track) - 1)
        return track[i]
    vs = [val(h) for h in hits]

    def f(t, axis):
        k = 0
        for b in bounds:
            if t >= b + ramp / 2:
                k += 1
        v = -vs[min(k, len(vs) - 1)][axis]
        for j, b in enumerate(bounds):
            if abs(t - b) < ramp / 2 and j + 1 < len(vs):
                u = smooth((t - (b - ramp / 2)) / ramp)
                v = -(vs[j][axis] + (vs[j + 1][axis] - vs[j][axis]) * u)
        return v
    return f


def replan_state(pid, ver="v004"):
    """forget the audio / type / compose / final items of a piece in the runner state (shots, stills, depth, flow stay cached), so a
    re-planned piece (new song + grid) re-runs type + compose only on the SAME footage. Prints what it dropped."""
    sp = ROOT / "15_fusion" / ver / "local" / "pieces" / pid / "state.json"
    if not sp.exists():
        return []
    st = json.loads(sp.read_text(encoding="utf-8"))
    drop = [k for k in st.get("items", {}) if k.split(":")[0] in ("audio", "type", "compose", "assemble", "final")]
    for k in drop:
        st["items"].pop(k, None)
        st.get("pending", {}).pop(k, None)
    sp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    # the box type stage keeps finished frames (a re-run with a changed spec renders nothing): move them aside (never deleted)
    import subprocess
    import time
    tag = time.strftime("%H%M%S")
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", str(BOX["port"]), f"root@{BOX['host']}",
                    f"cd {BOX['root']}/out/{pid} 2>/dev/null && [ -d type ] && mv type type_old_{tag} || true"],
                   capture_output=True, text=True, timeout=120)
    return drop + [f"box out/{pid}/type -> type_old_{tag}"]


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp_keys(points):
    """[(t, v), ..] -> f(t): smoothstep between keys, held outside."""
    def f(t):
        if t <= points[0][0]:
            return points[0][1]
        for (a, va), (b, vb) in zip(points[:-1], points[1:]):
            if t <= b:
                return va + (vb - va) * smooth((t - a) / max(b - a, 1e-6))
        return points[-1][1]
    return f


def pulse(beats, downbeats, amp=1.0, amp_db=1.6, decay=0.22, until=99.0):
    """f(t): sharp attack on every beat (downbeats amp_db), exponential decay `decay` s."""
    bs = [b for b in beats if b < until]
    dset = {round(d, 3) for d in downbeats}

    def f(t):
        v = 0.0
        for b in bs:
            if b <= t + 1e-6 and t - b < 6 * decay:
                a = amp_db if round(b, 3) in dset else amp
                v = max(v, a * math.exp(-(t - b) / decay))
        return v
    return f


def tl(fn, dur, fps=FPS, nd=3):
    """sample f(t) at every frame (piece seconds) -> {"timeline": keys} (linear between frames = exact per frame)."""
    n = int(round(dur * fps))
    keys, last = [], None
    for i in range(n + 1):
        t = i / fps
        v = round(float(fn(t)), nd)
        keys.append([round(t, 4), v])
    # drop redundant middle keys of flat runs
    out = []
    for k, kv in enumerate(keys):
        if 0 < k < len(keys) - 1 and keys[k - 1][1] == kv[1] == keys[k + 1][1]:
            continue
        out.append(kv)
    return {"timeline": out}


def hold_gate(holds, lead=0.2, tail=0.25):
    """f(t) 0..1: 1 during every word hold (ramps in over `lead` s before the hit, out over `tail` s after the release). Used as the
    sandwich `front=` region (type alpha x gate) so mattes only occlude the type while it forms / exits, never while it is read
    (user rule 2026-10-26, after P15 g01)."""
    def f(t):
        v = 0.0
        for a, b in holds:
            if a - lead <= t <= b + tail:
                v = max(v, smooth((t - (a - lead)) / lead) if t < a else (1.0 if t <= b else 1.0 - smooth((t - b) / tail)))
        return v
    return f


def window_ops(sections, cut, near_thr, soft=0.08, type_a="$wm", line="$wl", foot="$foot", depth="$depth", front="$front",
               base="$image", out="image", line_px_gain=1.0):
    """P10's principle (learn, don't copy): in a WINDOW section the frame flips to a flat plate, the letters are windows into the
    footage, the near subject (depth above near_thr) steps out in front of the plate, and the letter outline is drawn in the
    plate's contrast colour. sections = [(t_on, t_off, plate_rgb, line_rgb), ..] piece seconds (flips land on hits / ends).
    Outside the sections `base` (the footage chain) is kept. During a hold the subject never covers the letters (front)."""
    keys_p, keys_l, keys_g = [[0.0, [0, 0, 0]]], [[0.0, [1, 1, 1]]], [[0.0, 0.0]]
    for a, b, pl, ln in sections:
        keys_p.append([round(a, 4), list(pl)])
        keys_l.append([round(a, 4), list(ln)])
        keys_g += [[round(a - 1e-3, 4), 0.0], [round(a, 4), 1.0], [round(b - 1e-3, 4), 1.0], [round(b, 4), 0.0]]
    keys_g.append([round(cut + 1, 4), 0.0])
    lo, hi = near_thr - soft / 2, near_thr + soft / 2
    return [
        {"op": "fill", "out": "wblk", "args": {"like": foot, "color": [0.0, 0.0, 0.0]}},
        {"op": "over", "out": "wdep", "args": {"base": "$wblk", "fg": [1.0, 1.0, 1.0], "alpha": depth}},
        {"op": "luminosity_key", "out": "wnear", "args": {"img": "$wdep", "low": lo, "high": hi}},
        {"op": "over", "out": "wfr", "args": {"base": "$wblk", "fg": [1.0, 1.0, 1.0], "alpha": front}},
        {"op": "luminosity_key", "out": "wkeep", "args": {"img": "$wfr", "low": 0.0, "high": 1.0, "invert": True}},
        {"op": "multiply", "out": "wnear", "args": {"a": "$wnear", "b": "$wkeep"}},
        {"op": "knockout", "out": "win", "args": {"plate_rgb": {"timeline": keys_p, "step": True}, "type_a": type_a, "footage": foot,
                                                  "fg_matte": "$wnear"}},
        {"op": "multiply", "out": "wline", "args": {"a": line, "b": line_px_gain}},
        {"op": "over", "out": "win", "args": {"base": "$win", "fg": {"timeline": keys_l, "step": True}, "alpha": "$wline"}},
        {"op": "crossfade", "out": out, "args": {"a": base, "b": "$win", "t": {"timeline": keys_g}}},
    ]


def reformat(piece, fmt, suffix, still_prompt, shot_prompt, seed_add=50):
    """the other-aspect version of a piece (user rule "Both formats"): own id (<id><suffix>), format, its OWN still generated at that
    aspect (reframed prompt, new seed; never a crop) and its own single continuous i2v shot; same song, grid and plan."""
    land = fmt == "16:9"
    piece["id"] = piece["id"] + suffix
    piece["format"] = fmt
    sz = [1920, 1088] if land else [1088, 1920]
    st = piece["stills"][0]
    st.update(seed=st["seed"] + seed_add, size=sz, prompt=still_prompt)
    sh = piece["shots"][0]
    sh.update(seed=sh["seed"] + seed_add, size=sz, prompt=shot_prompt)
    return piece
