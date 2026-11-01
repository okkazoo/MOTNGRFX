"""Piece spec (JSON) -> mograph.modules Sequence + CameraChain for the fusion renderer.

Spec (all keys optional except words):
  size [w, h] (multiples of 64), fps, duration (s), bpm, beats [s...], downbeats [s...], beat_offset, bar (beats per bar, 4), seed,
  energy, mood, tier (calm | mid | kinetic: transition gap / form / exit pace, see PACE; bpm <= 90 is always calm), palette {bg, fg, accents [..], panel}  (colours [r,g,b] or '#rrggbb'),
  words [ {text | lines, module ('lanes'), font (registry name / file name / key), axes {tag: number | [lo, hi, shape, {opts}] |
           {keys: [[t, v], ..]}}, ar (x squeeze, null = fit), t0, t1 (module window: form starts, exit ends), form_s, exit_s,
           hit_on ('bar' | 'beat' | 'auto'), form_in, form_out (io styles, str | [name, {opts}]), entry, exit (ports),
           placement {yaw, pitch, roll, dx, dy, z, bleed, w, h, wf, hf (box as fractions of the world frame, hold_layouts)}, params {..module params},
           case ('upper' | 'asis'), screen_offset [ox, oy] (frame fractions the word is shifted on screen, see ScreenOffsetCamera), hold_layout {..} (record, ignored here)} ],
  transitions 'auto' | [ name | {module, dur, ..params} | null per gap ], transition_names (engine pool, default twist / braid / fan),
  camera 'auto' | 'beat_orbit' (breath-echo orbit locked to the hits) | module name (+ camera_params {..}) ('spin_read_spin': that move on every word, follow in the gaps) | [ {module, word|window|t0,t1, ..params} ],
  camera_override {'word:1': module name | {module, ..params}} (with camera 'auto': that move on that window),
  v006 pan transitions (tools/fusion/pan_plan.py): word "transition": {"kind": "pan", "dir": down|up|right|left, "beats": 2, ..} (the gap after it; or a `transitions` entry
    {"module": "pan", ..}, or camera {"move": "pan", "dir": ..}): the camera travel IS the transition, words of a run sit one frame-span apart, B's hit on a downbeat,
  camera_F (1000), passes [..], flow {mode}, depth {near, far, mode}, clay {..}, line {px}.
Beats drive timing: every word's hit (form -> hold) lands on a downbeat (hit_on 'bar'; 'auto' = a downbeat within 1.25 beats of the wish, else a beat), hold / exit / transition boundaries on beats,
so a spin_read_spin camera (spin-in ends at the hit, spin-out starts at the release) spins on beats."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mograph import font_registry  # noqa: E402
from mograph.modules import core, Sequence, Timing, Placement, Palette  # noqa: E402
from mograph.modules.core import get_module  # noqa: E402
from mograph.modules.camera import CameraChain, get_camera  # noqa: E402
from mograph.render import cam3d  # noqa: E402
from tools.fusion.beats import Beats  # noqa: E402
from tools.fusion.axes import AxesAnim, AxesWarp, canon_axes  # noqa: E402
from tools.fusion import cameras  # noqa: E402,F401  (registers beat_orbit)
from tools.fusion import pan_plan as PAN  # noqa: E402  (v006: pan transitions = the camera travel; timing, layout, camera windows)

PASSES = ("beauty", "matte", "depth", "line", "clay", "flow")
_REG = {core._norm_font(k): v for k, v in font_registry.REGISTRY.items()}


def world_for(size):
    """frame plane (world units) for a render size: 540 wide for portrait, 960 for landscape, same aspect as the image."""
    w, h = size
    ww = 540 if h > w else (960 if w > h else 720)
    return ww, int(round(ww * h / w))


def parse_size(s):
    w, h = str(s).lower().split("x")
    return int(w), int(h)


def check_size(size):
    w, h = size
    if w % 64 or h % 64:
        raise ValueError(f"size {w}x{h}: both sides must be multiples of 64 (e.g. 576x1024 draft, 1088x1920 final)")
    return w, h


def colour(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(x) for x in c)


def make_palette(p):
    if not p:
        return Palette.get("ink")
    base = Palette.get("ink")
    acc = [colour(a) for a in p["accents"]] if p.get("accents") else base.accents
    return Palette(colour(p["bg"]) if "bg" in p else base.bg, colour(p["fg"]) if "fg" in p else base.fg, acc,
                   colour(p["panel"]) if p.get("panel") else None)


def load_spec(src):
    spec = json.load(open(src, encoding="utf-8")) if isinstance(src, (str, os.PathLike)) else dict(src)
    spec.setdefault("size", [576, 1024])
    spec.setdefault("fps", 24)
    spec.setdefault("duration", 6.0)
    spec.setdefault("camera", "auto")
    spec.setdefault("transitions", "auto")
    spec.setdefault("passes", list(PASSES))
    assert spec.get("words"), "spec needs a words list"
    return spec


# ------------------------------------------------------------------ line breaking for tall type
def _candidates(text, max_lines, landscape=False):
    """line-break candidates. Lines break ONLY at spaces: a single word is never cut (MIDN/IGHT, WA/KE/UP, GLA/SS, ESC/APE in the round-3 QC, 2026-10-29); it stays on one line and
    the squeeze `ar` / the box scale fit it instead. Several words: every contiguous grouping of the words."""
    if "\n" in text:
        return [[s for s in text.split("\n")]]
    words = text.split()
    out = []
    n = len(words)
    if n > 1:
        for mask in range(2 ** (n - 1)):                     # contiguous groupings of the words
            lines, cur = [], [words[0]]
            for i in range(1, n):
                if (mask >> (i - 1)) & 1:
                    lines.append(" ".join(cur))
                    cur = [words[i]]
                else:
                    cur.append(words[i])
            lines.append(" ".join(cur))
            if len(lines) <= max_lines:
                out.append(lines)
    else:
        out.append([words[0] if words else text])
    return out


def fit_lines(fontd, text, box_w, box_h, ar_min=None, max_lines=5, lines=None):
    """pick the line breaks + horizontal squeeze `ar` that fill a box_w x box_h box with the largest tall letters
    (reference: 10_finals F8_9x16: condensed faces, stacked lines). Pixel / dot faces (many contours per letter) are not squeezed
    unless ar_min says so (dots would turn into ellipses). -> (lines, ar, cap_px)."""
    best = None
    landscape = box_w > box_h
    cands = [list(lines)] if lines else _candidates(text, max_lines, landscape)
    for ls in cands:
        lay = core.layout_word(dict(fontd, ar=1.0), ls, box_w, box_h)
        pix = sum(len(l["cons"]) for l in lay["letters"]) > 6 * max(len(lay["letters"]), 1)
        lo = ar_min if ar_min is not None else (0.97 if pix else 0.5)
        s1 = lay["scale"]
        maxw, blockh, capu = lay["size"][0] / s1, lay["size"][1] / s1, lay["cap"] / s1
        need = (box_w / maxw) / (box_h / blockh)
        ar = min(1.0, max(lo, need))
        s = min(box_w / (maxw * ar), box_h / blockh)
        score = s * capu * ar ** 0.35
        if best is None or score > best[0]:
            best = (score, ls, ar, s * capu)
        if landscape and not lines and len(ls) == 1 and len(cands) > 1 and ar >= 0.7 and " " not in ls[0]:
            break                                             # landscape: a single word that fits on one line stays whole (DREAMING)
    return best[1], round(best[2], 4), best[3]


NO_MATERIAL = ("flicker", "wipe")


def _style_name(v):
    return (v[0] if isinstance(v, (list, tuple)) else v) or "lines"


# ------------------------------------------------------------------ timing plan
# pace by tempo / mood: (beats of a transition gap, gap min s, gap max s, form_s range, exit_s range)
PACE = {"calm": (4, 1.2, 3.2, (0.9, 2.6), (0.8, 2.2)),
        "mid": (3, 0.9, 2.2, (0.7, 2.0), (0.5, 1.6)),
        "kinetic": (2, 0.6, 1.4, (0.7, 1.8), (0.5, 1.4))}
_MOOD_PACE = {"calm": "calm", "luxury": "calm", "nostalgic": "calm", "warm": "mid", "dark": "mid", "playful": "mid",
              "techno": "kinetic", "hype": "kinetic"}
SLOW_BPM = 90.0


# ------------------------------------------------------------------ overlap mode (slow tempo)
# At <= 90 bpm a beat is >= 0.67 s, and a word cycle (form + hold + exit + transition) that snaps every boundary to a beat costs
# hold + ~3 beats, so a readable hold (>= 1.2 s) pushes every later word a beat past its downbeat. Overlap mode keeps ONLY the hits on the
# grid: rel / end / the transition gap are placed in seconds, and the next word's form may start while the previous exit / transition is still
# finishing -- the three phases are allowed to compress below their normal floors (overlap_s = how far below, at most one beat). The sequence
# still plays its items back to back: the overlap is realised by shortening exit / transition / form, never by drawing two items at once.
HOLD_FLOOR = 1.2                       # readable minimum hold (s) in overlap mode (on top of min_hold)
OVERLAP_FLOORS = dict(form=0.35, form_nomat=0.2, exit=0.4, gap=0.4)           # compressed floors (s)
NORMAL_FLOORS = dict(form=0.45, form_nomat=0.2, exit=0.5, gap=0.5)             # what the beat-snapped mode needs as a minimum (s)


def overlap_on(spec, grid) -> bool:
    """spec['overlap']: False / 'off' disables, True / 'on' forces, default 'auto' = slow tempo (<= SLOW_BPM) on a beat grid."""
    ov = (spec or {}).get("overlap", "auto")
    if ov is False or ov == "off":
        return False
    if ov is True or ov == "on":
        return grid is not None
    return bool(grid) and 60.0 / grid.iv <= SLOW_BPM


def overlap_split(C, hold_min, f_pref, x_pref, g_pref, f_min, g_fix=None, f_norm=None):
    """one cycle between two hits (C s) = hold + exit + gap + the next line's form. The hold gets all the room the other three do not need; they
    shrink from their preferred length toward OVERLAP_FLOORS only as far as the readable hold requires. g_fix: a gap that cannot move (a cut).
    -> dict(f, x, g, hold, overlap) or None when even the floors leave less than hold_min. overlap = how far below NORMAL_FLOORS the three sum."""
    pref = dict(f=f_pref, x=x_pref, g=g_fix if g_fix is not None else g_pref)
    flo = dict(f=f_min, x=OVERLAP_FLOORS["exit"], g=g_fix if g_fix is not None else OVERLAP_FLOORS["gap"])
    room = C - hold_min
    if sum(flo.values()) > room + 1e-9:
        return None
    if sum(pref.values()) <= room:
        use = dict(pref)
    else:
        slack = {k: pref[k] - flo[k] for k in pref}
        need = sum(pref.values()) - room
        tot = sum(slack.values())
        use = {k: pref[k] - (need * slack[k] / tot if tot > 0 else 0.0) for k in pref}
    nf = NORMAL_FLOORS["form"] if f_norm is None else f_norm
    norm_sum = nf + NORMAL_FLOORS["exit"] + (g_fix if g_fix is not None else NORMAL_FLOORS["gap"])
    return dict(f=use["f"], x=use["x"], g=use["g"], hold=C - sum(use.values()), overlap=max(0.0, norm_sum - sum(use.values())))


def pace_of(spec, grid):
    """'calm' | 'mid' | 'kinetic' for the transition pace: slow tempo (<= 90 bpm) or tier / mood calm -> calm; else the spec's
    `tier` ('mid' | 'kinetic'), else its `mood`, else medium."""
    spec = spec or {}
    tier = {"medium": "mid"}.get(spec.get("tier"), spec.get("tier"))
    mood = _MOOD_PACE.get(str(spec.get("mood") or "").lower())
    bpm = 60.0 / grid.iv if grid else spec.get("bpm")
    if tier == "calm" or (bpm and float(bpm) <= SLOW_BPM) or (tier is None and mood == "calm"):
        return "calm"
    return tier if tier in PACE else (mood or "mid")


def _line_chars(w):
    lines = w.get("lines") or str(w.get("text", "")).split("\n")
    return max((len(str(l).replace(" ", "")) for l in lines), default=1)


def min_hold(w):
    """seconds a formed line must stay on screen to be read: max(1.0, 0.6 + 0.07 * characters in its longest line)."""
    return max(1.0, 0.6 + 0.07 * _line_chars(w))


def plan_timing(words, spec, grid):
    """per word: start S (module start), hit (end of form = hold start), rel (end of hold), end, and the gap to the next word.
    Boundaries snap to the beat grid; hits to downbeats when each word has a bar to itself ('auto'), else to beats.
    The transition gap, form and exit lengths follow the tempo and mood (PACE); the hold is kept readable (min_hold) by shrinking
    the default form / exit first (an explicit form_s / exit_s is the caller's and stays)."""
    out, pos = [], 0.0
    n = len(words)
    bar = grid.iv * grid.bar if grid else None
    beats_n, gmin, gmax, frange, xrange_ = PACE[pace_of(spec, grid)]
    ov = overlap_on(spec, grid)
    pans = PAN.boundaries(spec, words)                     # v006: gap i is a camera travel (pan): A's hold -> travel -> B's hit on a downbeat
    for i, w in enumerate(words):
        c_in, c_out = (i - 1) in pans, i in pans
        t0 = float(w.get("t0", pos if i else 0.0))
        t1 = float(w.get("t1", t0 + 2.0))
        span = max(t1 - t0, 1.0)
        f = float(w.get("form_s", np.clip(0.34 * span, *frange)))
        x = float(w.get("exit_s", np.clip(0.26 * span, *xrange_)))
        h = max(span - f - x, 0.0)
        need = max(min_hold(w), HOLD_FLOOR) if ov else min_hold(w)
        f_floor, x_floor = (OVERLAP_FLOORS["form"], OVERLAP_FLOORS["exit"]) if ov else (0.45, 0.5)
        if h < need - 1e-9:                                # shrink the default form / exit (down to 0.45 / 0.5 s), never the hold
            spare_f = max(f - f_floor, 0.0) if "form_s" not in w else 0.0
            spare_x = max(x - x_floor, 0.0) if "exit_s" not in w else 0.0
            take = min(need - h, spare_f + spare_x)
            if take > 0:
                tf = take * spare_f / (spare_f + spare_x)
                f, x = f - tf, x - (take - tf)
                h = max(span - f - x, 0.0)
        if c_in:                                           # carried words: the form / exit windows are only EDGE_S long and frozen on the hold
            f = PAN.EDGE_S
        if c_out:
            x = PAN.EDGE_S
        S = pos
        hit_on = w.get("hit_on", "auto")
        minhit = S + (OVERLAP_FLOORS["form_nomat"] if _style_name(w.get("form_in")) in NO_MATERIAL else f_floor)
        if c_in:
            hit = S + f                                    # the previous word's lookahead put this exactly on the downbeat
        elif grid and hit_on != "beat":
            d = grid.snap(S + f, bars=True, minimum=minhit)
            hit = d if (hit_on == "bar" or abs(d - (S + f)) <= 1.25 * grid.iv) else grid.snap(S + f, minimum=minhit)
        else:
            hit = grid.snap(S + f, minimum=minhit) if grid else S + f
        if c_out:                                          # pan: the travel ENDS on the next word's hit, on the first downbeat the readable hold allows
            cfg = pans[i]
            tv = PAN.travel_s(cfg, grid)
            tgt = hit + max(h, need) + x + tv + PAN.EDGE_S
            bhit = grid.snap(tgt, bars=(cfg["land"] != "beat"), mode="ceil") if grid else tgt
            rel = bhit - PAN.EDGE_S - tv - x
            end = rel + x
        elif ov:                                           # overlap mode: only the hit is on the grid; the hold keeps its readable length
            rel = hit + max(h, need)
            end = rel + max(x, x_floor)
        else:
            rel = grid.snap(hit + h, minimum=hit + (0.0 if h <= 0.0 else grid.iv)) if grid else hit + h
            end = grid.snap(rel + x, minimum=rel + 0.5 * grid.iv) if grid else rel + x
        if i + 1 < n and "t0" in words[i + 1] and "t1" in w:
            gap = float(words[i + 1]["t0"]) - float(w["t1"])
        else:
            gap = None
        carries = _style_name(w.get("form_out")) not in NO_MATERIAL and (i + 1 >= n or _style_name(words[i + 1].get("form_in")) not in NO_MATERIAL)
        if not carries:                                   # a flicker / wipe side carries nothing: the engine puts a cut there
            gap = 0.5 * grid.iv if grid else 0.25
        elif gap is None or gap < 0.35:
            gap = float(np.clip(beats_n * grid.iv if grid else 0.9, gmin, gmax))
        if carries and not ov:
            gap = grid.snap(end + gap, minimum=end + 0.5 * grid.iv) - end if grid else gap
        if c_out:
            gap = tv
        out.append(dict(S=S, hit=hit, rel=rel, end=end, gap=gap, f=hit - S, h=rel - hit, x=end - rel))
        pos = end + gap
    return out


# ------------------------------------------------------------------ scene
class Scene:
    """everything a render needs: .seq (FusionSequence) .chain (CameraChain) .size .world .fps .nf .info (readout)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_fusion_sequence(items, bg, tail):
    from tools.fusion.sequence import FusionSequence
    return FusionSequence(items, bg=bg, tail=tail)


def build_scene(spec, size=None, verbose=False):
    spec = load_spec(spec)
    size = check_size(tuple(size) if size else tuple(spec["size"]))
    world = world_for(size)
    cam3d.set_world(*world)
    W, H = world
    fps, dur = int(spec["fps"]), float(spec["duration"])
    grid = Beats(spec.get("bpm"), spec.get("beats"), spec.get("bar", 4), spec.get("beat_offset", 0.0), dur + 4.0,
                 spec.get("downbeats"))
    pal = make_palette(spec.get("palette"))
    words = spec["words"]
    if spec.get("font_pins"):                                   # the user's per-line fonts win, whatever a rule / the model / an old piece wrote (tools/fusion/font_pins.py)
        from tools.fusion import font_pins as _FP
        words = _FP.applied(words, spec["font_pins"])
    mp_cfg = None
    if isinstance(spec.get("camera"), dict) and spec["camera"].get("plate") == "multiplane":     # multiplane plate: words sit between depth cards ("slot")
        from mograph.render import multiplane as MP
        mp_cfg = MP.plan(spec["camera"])
        words = MP.apply_slots(words, mp_cfg)
    pans = PAN.boundaries(spec, words)                          # v006: gaps that are a camera travel (pan); words of a run share one line in the world
    pan_notes = []
    if pans:
        words, pan_notes = PAN.prepare_words(words, pans)
    plan = plan_timing(words, spec, grid)
    pan_off = PAN.layout_offsets(words, pans, world, float(spec.get("camera_F", 1000.0))) if pans else {}

    # --- word modules
    mods, anims, lay_info = [], [], []
    for i, (w, pt) in enumerate(zip(words, plan)):
        fname = w.get("font", "Big Shoulders Display")
        key, _loc = core.resolve_font(fname)
        reg = _REG.get(core._norm_font(fname)) or _REG.get(key) or {}
        text = w.get("text", "")
        if w.get("case", reg.get("case", "upper")) == "upper":
            text = text.upper()
        anim = AxesAnim(fname, w.get("axes"), grid if grid else None, span=(pt["S"], pt["end"]),
                        quant=int(w.get("quant", 32)), base=reg.get("axes"))
        fontd = dict(key=key, axes=anim.nominal)
        p = dict(yaw=0.0, pitch=0.0, roll=0.0, dx=0.0, dy=0.0, z=0.0, bleed=1.0)
        p.update(w.get("placement") or {})
        rolled = abs(abs(float(p.get("roll") or 0.0)) - 90.0) < 1.0           # a word rolled +-90 reads vertically: wf is the box's LONG side as a fraction of the frame height, hf its short side as a fraction of the width
        pw = float(p.get("w") or (float(p["wf"]) * (H if rolled else W) if p.get("wf") else W * p["bleed"]))
        ph = float(p.get("h") or (float(p["hf"]) * (W if rolled else H) if p.get("hf") else 1.08 * H * p["bleed"]))
        if w.get("lines"):
            lines, ar, cap = fit_lines(fontd, text, pw * 0.94, ph * 0.9, w.get("ar_min"), lines=w["lines"])
        else:
            lines, ar, cap = fit_lines(fontd, text, pw * 0.94, ph * 0.9, w.get("ar_min"), w.get("max_lines", 5))
        if w.get("ar") is not None:
            ar = float(w["ar"])
        fontd["ar"] = ar
        origin = (W / 2.0 + p["dx"] * W, H / 2.0 + p["dy"] * H, float(p["z"]))
        if i in pan_off:                                  # next word of a pan run: one frame-span further along the travel, same plane
            o0 = run_origin
            origin = (o0[0] + pan_off[i][0], o0[1] + pan_off[i][1], o0[2])
        elif i in {a for a, _b in PAN.runs(pans, len(words))}:
            run_origin = origin
        pl = Placement.at(origin, p["yaw"], p["pitch"], p["roll"], w=pw, h=ph)
        entry, exit_ = w.get("entry", "up"), w.get("exit", "down")
        if isinstance(entry, dict) or isinstance(exit_, dict):               # contour ports (mograph/modules/contour_ports.py): the module gets its side
            try:                                                             # (or 'up' / 'down'); the ports are applied where plate data is bound
                from mograph.modules.contour_ports import split_port_spec
                entry, exit_, _ports = split_port_spec(w)
            except Exception:                                                # noqa: BLE001  module missing: plain sides
                entry = entry.get("side", "up") if isinstance(entry, dict) else entry
                exit_ = exit_.get("side", "down") if isinstance(exit_, dict) else exit_
        cls = get_module(w.get("module", "lanes"))
        tm = Timing(pt["f"], pt["h"], pt["x"])
        wpal = make_palette(w["palette"]) if w.get("palette") else pal

        def style(v):
            return tuple(v) if isinstance(v, list) else (v or "lines")
        mod = cls(lines, font=fontd, placement=pl, entry=entry, exit=exit_, timing=tm, palette=wpal,
                  form_in=style(w.get("form_in")), form_out=style(w.get("form_out")), **{**_hold_params(w.get("module", "lanes")), **(w.get("params") or {})})
        mods.append(mod)
        anims.append((anim, lines, key, ar, (pw * 0.94, ph * 0.9)))
        lay_info.append(dict(text=text, lines=lines, ar=ar, cap_px=round(float(cap), 1), font=key,
                             axes_nominal=anim.nominal, animated=sorted(anim.tracks)))

    # --- transitions
    tspec = spec.get("transitions", "auto")
    auto = None
    if len(mods) > 1:
        auto = Sequence.auto(list(mods), seed=int(spec.get("seed", 3)), energy=float(spec.get("energy", 0.5)), bpm=spec.get("bpm"),
                             mood=spec.get("mood"), bg=pal, names=spec.get("transition_names", ["twist", "braid", "fan"]))
    items, trinfo = [mods[0]], []
    for i in range(len(mods) - 1):
        want = tspec[i] if isinstance(tspec, list) and i < len(tspec) else None
        dT = plan[i]["gap"]                                  # tier / tempo paced (PACE) and beat snapped: later words are planned on it,
        if isinstance(want, str):                            # so it replaces engine.pick_duration (that one only knows the transition)
            want = dict(module=want)
        if i in pans:                                        # the camera travel: the pan transition (dur = the planned travel), B placed one frame-span on
            pc = pans[i]
            tr = get_module("pan")(dur=dT, dir=pc["dir"], ease=pc["ease"], exit_io=pc["exit_io"], entry_io=pc["entry_io"], fill=pc["fill"])
            tr.camera_hint, tr.engine = None, None
        elif want:
            want = dict(want)
            name = want.pop("module")
            d = float(want.pop("dur", dT))
            tr = get_module(name)(dur=d, **want)
        else:
            ch = auto.engine[i]
            tr = ch.tcls(dur=dT, **ch.params)
            tr.camera_hint, tr.engine = ch.camera_hint, ch
        items += [tr, mods[i + 1]]
    ctx, port_notes = _port_context(spec, words, world, fps, grid), []
    if ctx is not None:                                       # contour ports: words with entry / exit dicts get PortedWord, `via` swaps the transition
        try:
            from mograph.modules import contour_ports as CP
            items = CP.apply_ports(items, words, ctx)
            mods = [it for it in items if isinstance(it, core.WordModule)]            # the wrapped words (same order)
        except Exception as e:                                # noqa: BLE001  plates were given and ports are declared: fail loudly, never degrade
            raise RuntimeError(f"contour ports could not be applied: {type(e).__name__}: {e}") from e
    seq = _make_fusion_sequence(items, pal, 0.0)
    seq.tail = max(0.0, dur - seq.duration)
    if ctx is not None:
        CP.attach(seq, ctx)
    seq.warps = []
    for k, (it, a, b) in enumerate(seq.timeline):
        wp = None
        if it in mods and not (ctx is not None and isinstance(it, CP.PortedWord)):
            anim, lines, key, ar, box = anims[mods.index(it)]
            if anim.animated:
                wp = AxesWarp(it, anim, lines, key, ar, box, t_start=a)
                wp.precompute(np.arange(a, b + 1e-6, 1.0 / fps))
        seq.warps.append(wp)

    # --- camera
    chain = build_camera(seq, spec)
    offs = [list(w.get("screen_offset") or (0.0, 0.0)) for w in words]
    if chain is not None and any(abs(o[0]) + abs(o[1]) > 1e-9 for o in offs):
        chain = ScreenOffsetCamera(chain, screen_offset_keys(seq, mods, offs), world)
    if ctx is not None:                                       # ports are projected through the final camera: bind it, then let the rides rebuild
        if mp_cfg is not None:
            ctx.set_plane(mp_cfg["z"][mp_cfg["subject"]], mp_cfg["ref"])
        ctx.bind_camera(chain)
        CP.attach(seq, ctx)
    notes = list(port_notes) + list(pan_notes)
    if ctx is not None:                                       # the log line that proves the ports are bound to real plate data
        pw = [k for k, it in enumerate(mods) if isinstance(it, CP.PortedWord)]
        notes.append(f"contour ports BOUND: {len(pw)} ported word(s) {pw}, {len(spec['plates']) if isinstance(spec.get('plates'), (list, tuple)) else 'PlateData'} plate shot(s)")
    if seq.duration > dur + 1e-6:
        notes.append(f"timeline ({seq.duration:.2f}s) is longer than the clip ({dur:.2f}s): the end is cut")
    info = dict(words=[], transitions=[], notes=notes + list(getattr(seq, "notes", [])), camera=chain.plan_text() if chain else None,
                world=list(world), beat_interval=grid.iv if grid else None)
    wi = 0
    for it, a, b in seq.timeline:
        if it in mods:
            m = mods[mods.index(it)]
            k = mods.index(it)
            info["words"].append(dict(lay_info[k], index=k, module=m.name, start=round(a, 3), hit=round(a + m.timing.form, 3),
                                      release=round(a + m.timing.form + m.timing.hold, 3), end=round(b, 3),
                                      form_in=m.form_in.name if hasattr(m.form_in, "name") else str(m.form_in),
                                      form_out=m.form_out.name if hasattr(m.form_out, "name") else str(m.form_out)))
        else:
            info["transitions"].append(dict(module=it.name, start=round(a, 3), dur=round(b - a, 3)))
    if pans:
        info["pans"] = [dict(dir=d, start=round(a, 3), end=round(b, 3), words=[ia, ib]) for (a, b, d, ia, ib) in PAN.pan_windows(seq)]
    return Scene(seq=seq, chain=chain, size=size, world=world, fps=fps, nf=int(round(dur * fps)), duration=dur, info=info,
                 spec=spec, grid=grid, palette=pal)


class PlatesMissing(FileNotFoundError):
    """the spec declares contour ports and names plates, but a plate / matte / depth file is missing (on the box: it was not shipped)."""


def _plate_data(plates, size=(216, 384), strict=False):
    """spec["plates"] -> contour_ports.PlateData: a PlateData as is, or a list of {plate, matte, depth, frames} per shot (paths relative to the repo
    root or absolute; a video file or a PNG directory), joined in time. A missing file: None (ports stay module sides), or PlatesMissing when strict."""
    from mograph.modules import contour_ports as CP
    if isinstance(plates, CP.PlateData):
        return plates
    parts = []
    for sh in plates or []:
        def path(k):
            v = sh.get(k)
            return None if not v else (v if os.path.isabs(v) else os.path.join(ROOT, v))
        files = [path(k) for k in ("plate", "matte", "depth") if path(k)]
        gone = [f for f in files if not os.path.exists(f)]
        if gone or not files:
            if strict:
                raise PlatesMissing("contour ports declared but plate files missing: " + (", ".join(gone) or "(shot names no plate / matte / depth)"))
            return None
        parts.append(CP.PlateData.from_files(plate=path("plate"), matte=path("matte"), depth=path("depth"), fps=float(sh.get("fps", 24)),
                                             size=tuple(sh.get("size") or size), max_frames=sh.get("frames")))
    return CP.PlateData.concat(parts) if parts else None


def _port_context(spec, words, world, fps, grid):
    """a ContourContext when a word carries a contour port (dict entry / exit / via) and the spec has plate data; else None."""
    has = any(isinstance(w.get("entry"), dict) or isinstance(w.get("exit"), dict) or w.get("via") for w in words)
    if not has or not spec.get("plates"):
        return None
    try:
        from mograph.modules import contour_ports as CP
        data = _plate_data(spec["plates"], strict=True)       # plates are declared: a missing file is an error, not a quiet fallback
        if data is None:
            return None
        return CP.ContourContext(data, world=world, fps=fps, beats=list(grid.t) if grid is not None and hasattr(grid, "t") else None)
    except PlatesMissing:
        raise
    except Exception as e:                                    # noqa: BLE001
        raise RuntimeError(f"contour ports declared and plates given, but the plate data could not be built: {type(e).__name__}: {e}") from e


def _hold_params(module):
    """hold_layouts.MODULE_HOLD_PARAMS: the params a held line of this module settles to (readable hold pose); the word's own `params` win."""
    from tools.fusion import hold_layouts as _hl
    return _hl.hold_params(module)


def build_camera(seq, spec):
    cam = spec.get("camera", "auto")
    F = float(spec.get("camera_F", 1000.0))
    if cam is None or cam == "demo":
        return None
    pan_ov = PAN.window_overrides(seq)                        # v006: one PanDir over every run of words joined by pan transitions (the travel IS the transition)
    if isinstance(cam, dict) and cam.get("move") == "pan":    # {"move": "pan", "dir": ..}: the pan camera; everything outside a run is the auto chain
        cam = "auto"
    if pan_ov and cam != "auto" and not (isinstance(cam, dict) and cam.get("plate") == "multiplane"):
        cam = "auto"                                          # a pan run owns its words' camera; the rest follows the auto rules
    if isinstance(cam, dict):                                 # {"chain": .., "plate": "multiplane", "cards": 4, ..}: any gallery move through the cards
        from mograph.render import multiplane as MP
        return MP.camera_for(seq, cam, F, extra=pan_ov)
    if cam == "auto":
        ov = {}
        for key, v in (spec.get("camera_override") or {}).items():
            v = dict(v) if isinstance(v, dict) else dict(module=v)
            ov[key] = get_camera(v.pop("module"))(**v)
        if pan_ov:                                            # camera_override entries on words inside a run are replaced by the run's PanDir
            wins = {k: (next(a for it, a, b in seq.timeline if it is seq.words[int(k.split(":")[1])]),
                        next(b for it, a, b in seq.timeline if it is seq.words[int(k.split(":")[1])]))
                    for k in ov if isinstance(k, str) and k.startswith("word:")}
            for (t0, t1) in pan_ov:
                for key, (a, b) in wins.items():
                    if a >= t0 - 1e-6 and b <= t1 + 1e-6:
                        ov.pop(key, None)
            ov.update(pan_ov)
        return CameraChain.auto(seq, override=ov or None, F=F)
    hints = seq.camera_hints(F)
    if cam == "beat_orbit":                                # one module over the whole timeline
        return CameraChain(seq, [(get_camera(cam)(**dict(spec.get("camera_params") or {})), 0.0, hints.T)], F=F, hints=hints)
    if isinstance(cam, str):                              # one move on every word, follow through carrying gaps, tiled at cuts
        cls0 = get_camera(cam)
        cp = dict(spec.get("camera_params") or {})
        known = getattr(cls0, "DEFAULTS", None)
        if isinstance(known, dict):                       # params of another camera (e.g. beat_orbit's left on a hold_drift spec) are dropped, not fatal
            cp = {k: v for k, v in cp.items() if k in known}
        cls = lambda: cls0(**cp)
        from mograph.modules.camera import Follow
        plan, t = [], 0.0
        for k, w in enumerate(hints.words):
            if k >= len(hints.trans):
                plan.append((cls(), t, hints.T))
                break
            tr = hints.trans[k]
            if tr.mod.name == "cut":                      # nothing travels: the next word's move starts at the middle of the cut
                end = 0.5 * (tr.a + tr.b)
                plan.append((cls(), t, end))
                t = end
            else:
                f0, f1 = w.b - 0.45 * w.exit, tr.wb.a + 0.55 * tr.wb.form
                plan += [(cls(), t, f0), (Follow(), f0, f1)]
                t = f1
        return CameraChain(seq, plan, F=F, hints=hints)
    plan = []
    for e in cam:
        e = dict(e)
        name = e.pop("module")
        win = e.pop("window", None) or e.pop("at", None)
        t0, t1 = e.pop("t0", None), e.pop("t1", None)
        if "word" in e:
            win = f"word:{e.pop('word')}"
        if "trans" in e:
            win = f"trans:{e.pop('trans')}"
        m = get_camera(name)(**e)
        if win is not None:
            a, b = hints.window(win)
        else:
            a, b = float(t0), None if t1 is None else float(t1)
        plan.append((m, a, b))
    return CameraChain(seq, plan, F=F, hints=hints)


# ------------------------------------------------------------------ screen offset (hold layouts)
def screen_offset_keys(seq, mods, offs):
    """[(t, ox, oy)] keys of a word-by-word screen offset: the offset of a word is held from the start of its hold to its release and
    glides (smootherstep) to the next word's offset between release and the next hit; the first word's offset holds from t = 0."""
    keys = []
    for k, m in enumerate(mods):
        a = next(a for it, a, b in seq.timeline if it is m)
        hit, rel = a + m.timing.form, a + m.timing.form + m.timing.hold
        if k == 0:
            keys.append((0.0, *offs[k]))
        keys += [(hit, *offs[k]), (max(rel, hit + 1e-3), *offs[k])]
    return keys


class ScreenOffsetCamera:
    """wraps a camera path (anything with .at(t)): the picture is shifted on screen by (ox, oy) frame fractions (y down) by moving eye and
    target together along the camera's image axes, so a word framed at the centre lands at its layout position in EVERY type pass
    (beauty / matte / depth / line, the LTX control video too). Other attributes are those of the wrapped chain."""

    def __init__(self, inner, keys, world):
        self.inner, self.keys, self.world = inner, [tuple(map(float, k)) for k in keys], tuple(world)

    def __getattr__(self, name):
        if name == "inner" or (name.startswith("__") and name.endswith("__")):    # unpickling (spawn workers) looks attributes up before __init__ ran
            raise AttributeError(name)
        return getattr(self.inner, name)

    def offset(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1], ks[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(ks, ks[1:]):
            if t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-9)
                u = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
                return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        return ks[-1][1], ks[-1][2]

    def at(self, t):
        import copy
        cam = self.inner.at(t)
        ox, oy = self.offset(float(t))
        if abs(ox) + abs(oy) < 1e-12:
            return cam
        dist = float(np.linalg.norm(cam.target - cam.eye))
        k = dist / cam.F                                       # world units per frame px at the look target
        shift = -(cam.r * ox * self.world[0] + cam.d * oy * self.world[1]) * k     # camera moves opposite: the picture moves by (+ox, +oy)
        c2 = copy.copy(cam)
        c2.eye, c2.target = cam.eye + shift, cam.target + shift
        return c2


# ------------------------------------------------------------------ landing report
def landing_report(piece):
    """per line of a planned piece (piece['plan']['timing'] = the real scene readout): land time (end of form), the design beat it was meant
    for (`wanted_hit`), the nearest beat / downbeat of the song grid with the distance in frames, and the hold. -> [dict]."""
    t = piece["type"]
    fps = float(t.get("fps", 24))
    beats = np.asarray(t.get("beats") or [], float)
    downs = np.asarray(t.get("downbeats") or [], float)
    rows = []
    for r in piece["plan"]["timing"]:
        hit = float(r["hit"])
        nb = float(beats[np.argmin(np.abs(beats - hit))]) if len(beats) else hit
        nd = float(downs[np.argmin(np.abs(downs - hit))]) if len(downs) else hit
        want = float(r.get("wanted_hit", hit))
        rows.append(dict(text=r["text"], land=round(hit, 3), wanted=round(want, 3), d_wanted_f=round((hit - want) * fps, 2),
                         beat=round(nb, 3), d_beat_f=round((hit - nb) * fps, 2), downbeat=round(nd, 3), d_down_f=round((hit - nd) * fps, 2),
                         wanted_is_downbeat=bool(len(downs) and np.min(np.abs(downs - want)) * fps <= 1.0),
                         hold=round(float(r["release"]) - hit, 3)))
    return rows
