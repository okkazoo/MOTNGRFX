"""depth_transition_type: v018 - the P06 words in THEIR OWN layout, handed over by a transition MODULE from the library, flat 2D, behind the runner matte.

User (v017 review): "I dont want it wavy I want it to use one of the module transitions that are in the library - that was just to indicate direction" and
"keep the text like the original [P06] size and position wise" (reference `17_depth-dolly/v018/review/user_ref_P06_layout.png`: SEARCHING as SEA / RCH / ING in
Doto, big, right two-thirds of the frame, most of its height, the runner in front of it on the left).

Everything comes from the existing type engine (tools/fusion/spec.build_scene): the words of 15_fusion/pieces/P06_v003.json (SEARCHING: Doto, lanes, lines
SEA/RCH/ING; A LIGHT: GeistPixel, lanes, lines A/LIG/HT, the piece's own lines, seed / energy / mood / line px / palette / axes), io `lines`, and ONE transition
module of mograph/modules/transitions/ (braid, circuit_board, fan, ripple, spiral_around, twist) between them, in one Sequence. No wave, no squeeze, no
re-layout: the only placement choice is the box width (`wf` 0.76 = the reference's ~72 % of the frame) and the sides: word A leaves toward the RIGHT, word B
arrives from the LEFT (the hand-off runs left -> right where the module has a direction). The type camera is the static cam3d HOME camera (no beat_orbit, no
demo-camera zoom). The frame is the depth video of v016 (`--matte-pass` runner matte through the breakup) composited with compose.sandwich (video, type, the
video back over it by the matte), ink by legible_color measured per frame (depth_ribbon_type.InkTrack).

Timeline on the depth video (24 fps): A formed and frozen on image 1's hold (i 0-23); A's exit starts when the turn starts (i 24); the transition runs through the
turn and the swing; B has formed (its own entry) at `i_hit` = landing + 18 and is frozen (readable) to the end. The module's own durations are scaled to this
window: A exit 0.20 s, transition dur 1.6 s, B form 0.40 s.

  python tools/fusion/depth_transition_type.py --depth-mp4 17_depth-dolly/v016/out/..._v016_depth.mp4 --matte-dir 17_depth-dolly/v016/local/matte_depth \
      --out 17_depth-dolly/v018 --transition all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_flat_type as DF  # noqa: E402
from tools.fusion import depth_ribbon_type as RB  # noqa: E402
from tools.fusion import depth_type as DT  # noqa: E402

TRANSITIONS = ("braid", "circuit_board", "fan", "ripple", "spiral_around", "twist")
WHITE = DF.WHITE


class PinnedCamera:
    """the engine's own camera chain (CameraChain.auto: hold_drift on the words, FOLLOW through the hand-off - the approved Module Gallery chains' camera)
    with both holds pinned to the cam3d HOME pose: pose(t) = home + w(t) * (chain(t) - home), w = 0 outside the chain's follow window and eased in / out
    over `ease` of it, so the move lives only inside the transition and eases out of / into the home framing (the holds sit exactly at the reference
    size and position)."""

    def __init__(self, chain, home, ease=0.25):
        self.chain, self.home, self.ease = chain, home, float(ease)
        wins = [(a, b) for name, _p, a, b in chain.plan if name == "follow"]
        self.t0, self.t1 = (wins[0] if wins else (0.0, 0.0))

    def weight(self, t):
        if self.t1 <= self.t0 or t <= self.t0 or t >= self.t1:
            return 0.0
        L = self.t1 - self.t0
        u = (t - self.t0) / L
        e = max(self.ease, 1e-3)
        sm = lambda x: float(x * x * (3 - 2 * x)) if 0 < x < 1 else (0.0 if x <= 0 else 1.0)       # noqa: E731
        return sm(u / e) * sm((1.0 - u) / e)

    def at(self, t):
        from mograph.render import cam3d
        w = self.weight(t)
        if w <= 0.0:
            return self.home
        c = self.chain.at(t)
        h = self.home
        return cam3d.Camera(h.eye + w * (c.eye - h.eye), h.target + w * (c.target - h.target), 0.0, h.F + w * (c.F - h.F))


class TransitionType:
    """word A -> transition module -> word B in one engine Sequence, flat, home camera. Interface the composer reads: matte(i), words, typ, size."""

    def __init__(self, flag, n_frames, transition, fps=24, i_start=24, i_hit=77, wf=0.76, dx=0.07, exit_s=0.20, dur=1.6, form_s=0.40,
                 sides=("right", "left"), size=DT.MATTE_SIZE, shutter=0.55, sub=3, params=None, n_strands=None, camera="home", cam_ease=0.25,
                 b_dx=None, b_dy=0.0):
        from tools.fusion import spec as S
        from mograph.render import cam3d
        pid, texts = DT.parse_flag(flag)
        self.typ = json.load(open(DT.piece_file(pid), encoding="utf-8"))["type"]
        self.fps, self.n_frames, self.size, self.shutter, self.sub = int(fps), int(n_frames), tuple(size), float(shutter), int(sub)
        self.i_start, self.i_hit, self.transition = int(i_start), int(i_hit), transition
        self.dx = float(dx)                                          # word A's placement dx; B's base placement is the mirror (-dx) unless b_dx is given
        A, B = (dict(DT.find_word(self.typ, t)) for t in texts)
        for w in (A, B):
            w.pop("t0", None), w.pop("t1", None), w.pop("transition", None)
        pa, pb = dict(A.get("placement") or {}), dict(B.get("placement") or {})
        pa.update(yaw=0.0, wf=wf, dx=dx)                           # flat 2D: the piece's yaw (+16 / -16) is not used; the word sits right of centre like the reference
        pb.update(yaw=0.0, wf=wf, dx=-dx if b_dx is None else b_dx, dy=b_dy)                          # B mirrored (the piece alternates yaw +16 / -16; image 2's runner is on the right)
        A.update(placement=pa, t0=0.0, form_s=0.5, exit_s=exit_s, t1=0.5 + 1.2 + exit_s, exit=sides[0])
        B.update(placement=pb, t0=0.0, form_s=form_s, exit_s=0.3, t1=form_s + 6.0 + 0.3, entry=sides[1])
        if n_strands:                                                # fewer strands = a lighter cable (the module's own `n_strands` parameter)
            for w in (A, B):
                w["params"] = dict(w.get("params") or {}, n_strands=int(n_strands))
        spec = dict(size=list(self.size), fps=self.fps, duration=12.0, bpm=None, bar=4, camera=("auto" if camera == "engine" else None), words=[A, B],
                    transitions=[dict(module=transition, dur=float(dur), **dict(dict(handoff="gate"), **(params or {})))], palette=WHITE,
                    **{k: self.typ[k] for k in ("seed", "energy", "mood", "line") if k in self.typ})
        self.sc = S.build_scene(spec, self.size)                    # handoff "gate": the approved depth-slide look (g04) keeps the gate ribbons, not the glyph-outline hand-off
        self.words = self.sc.info["words"]
        self.tr = self.sc.info["transitions"][0]
        cam3d.set_world(*self.sc.world)
        self.cam = cam3d.Camera.home()
        self.camera_mode = camera
        if camera == "engine":
            self.cam = PinnedCamera(self.sc.chain, cam3d.Camera.home(), cam_ease)       # the chain's follow move inside the window, home on both holds
        assert self.words[0]["lines"] and self.words[1]["lines"]
        self.pa = self.words[0]
        self.pb = self.words[1]
        self.tau0 = self.pa["release"]                               # video frame i_start == A's release (its exit starts)
        self.exit_s = float(exit_s)
        self.warp = None
        self._cache = {}

    def tau(self, i):
        if i <= self.i_start:
            return self.pa["release"]                                # A frozen on its hold
        if self.warp is not None and i > self.warp[0]:               # v019: the transition's VISIBLE module time is spread evenly over the rest of the window
            i_x, ts, cum, k0 = self._clock
            u = min(max((float(i) - i_x) / max(self.i_hit - i_x, 1e-6), 0.0), 1.0)
            return float(np.interp(cum[k0] + u * (cum[-1] - cum[k0]), cum, ts))
        t = self.tau0 + (float(i) - self.i_start) / self.fps
        return min(t, self.pb["hit"])                                # B frozen once formed

    def fit_clock(self, thr=1.0e-3, size=(288, 512), step=1.0 / 48.0, pass_through=0.03):
        """v019: every module has stretches in which none of it is on screen (the strands gather outside the frame at the start of its phase; ripple has
        a gap between its rings and B's strands). Probe (small, no blur) the module time from A's exit end to B's hit for the moments the type is on
        screen, and re-time the window so the VISIBLE module time runs at a constant rate over the frames from A's exit end to B's hit; invisible
        stretches get `pass_through` of the rate (they are crossed, not shown), so there is never an empty frame. Visible phases are neither cut nor reordered.
        -> (i_x, first visible module time, B hit)."""
        from mograph.render import cam3d
        cam3d.set_world(*self.sc.world)
        i_x = self.i_start + self.exit_s * self.fps
        t0 = self.pa["release"] + self.exit_s
        ts = np.arange(t0, self.pb["hit"] + 1e-6, step)
        mass = np.array([float(self.sc.seq.frame(t, self.cam, size=size, S=1).max(-1).mean()) / 255.0 for t in ts])
        vis = mass > thr
        w = np.where(vis, 1.0, pass_through)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
        k = next((j for j in range(len(ts) - 3) if vis[j:j + 4].all()), 0)
        self._clock = (i_x, ts, cum, k)                                  # the invisible lead-in (before the first sustained visible moment) is skipped
        self.warp = (i_x, float(ts[k]), float(self.pb["hit"]))
        self.probe_mass = (ts, mass)
        self._cache.clear()
        return self.warp


    def matte(self, i, wmax=432):
        if i in self._cache:
            return self._cache[i]
        t0 = self.tau(i)
        moving = self.i_start < i < self.i_hit + 1
        n = self.sub if moving else 1
        acc = None
        from mograph.render import cam3d
        cam3d.set_world(*self.sc.world)
        for k in range(n):
            t = t0 + (self.shutter / self.fps * ((k + 0.5) / n - 0.5) if n > 1 else 0.0)
            f = self.sc.seq.frame(t, self.cam, size=self.size, S=1).max(-1).astype(np.float32) / 255.0
            acc = f if acc is None else acc + f
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[i] = acc / n
        return self._cache[i]


class TransitionComposer(RB.RibbonComposer):
    """RibbonComposer's per-frame ink + sandwich, with the type exactly as the engine drew it (no wave / squeeze)."""

    def __init__(self, tt: TransitionType, frames, mattes):
        self.rt, self.frames, self.mattes = tt, frames, mattes
        H, W = frames[0].shape[:2]
        self.W, self.H = W, H
        cap = float(tt.words[0].get("cap_px", 200.0)) * 1080.0 / 540.0
        self.track = RB.InkTrack(tt.typ.get("palette") or {}, cap)

    def alpha(self, i, ss=2):
        return DT.place_matte(self.rt.matte(i, self.W), self.W, self.H, ss)


def fit_b(tt, mattes, frames_at=(77, 92, 107), min_vis=0.6, dx_range=(-0.16, -0.02), dy_range=(-0.12, 0.12)):
    """v019: the smallest nudge (dx, dy; fractions of the frame, B's placement is dx -0.07, dy 0) of word B so that every letter keeps >= min_vis visible
    behind the runner on its whole hold (the runner matte at `frames_at`), B staying left of centre. -> (dx, dy, min_visible)."""
    H, W = mattes[0].shape
    a0 = DT.place_matte(tt.matte(tt.i_hit), W, H, 1)
    dx0 = -tt.dx
    best = None
    for d in np.arange(dx_range[0], dx_range[1] + 1e-9, 0.01):
        for e in np.arange(dy_range[0], dy_range[1] + 1e-9, 0.01):
            M = np.float32([[1, 0, (d - dx0) * W], [0, 1, e * H]])
            a = cv2.warpAffine(a0, M, (W, H))
            ys, xs = np.nonzero(a > 0.25)
            if not len(xs) or xs.min() < 4 or xs.max() > W - 4 or ys.min() < 4 or ys.max() > H - 4:
                continue
            mv = min(min(v for *_, v in RB.letter_visibility(a, mattes[i])) for i in frames_at)
            cost = abs(d - dx0) + abs(e)
            if mv >= min_vis and (best is None or cost < best[0]):
                best = (cost, float(d), float(e), float(mv))
    if best is None:
        return dx0, 0.0, None
    return best[1], best[2], best[3]


def run_one(a, name, frames, mattes, outd, review, local):
    t0 = time.time()
    kw = dict(camera=a.camera, cam_ease=a.cam_ease)
    sides = ("right", "left") if a.flow == "right" else ("left", "right")
    tt = TransitionType(a.type, len(frames), name, a.fps, a.i_start, a.i_hit, a.wf, a.dx, a.exit_s, a.dur, a.form_s, sides, **kw)
    nudge = None
    if a.fit_b:
        bdx, bdy, mv = fit_b(tt, mattes)
        nudge = dict(dx=round(bdx, 3), dy=round(bdy, 3), min_letter_visible=None if mv is None else round(mv, 3))
        print("[%s] word B nudge:" % name, nudge, flush=True)
        if mv is not None and (abs(bdx + a.dx) > 0.004 or abs(bdy) > 0.004):
            tt = TransitionType(a.type, len(frames), name, a.fps, a.i_start, a.i_hit, a.wf, a.dx, a.exit_s, a.dur, a.form_s, sides, b_dx=bdx, b_dy=bdy, **kw)
    if a.fit_clock:
        print("[%s] clock (A exit end frame, first visible module time, B hit):" % name, [round(x, 2) for x in tt.fit_clock()], flush=True)
    comp = TransitionComposer(tt, frames, mattes)
    print("[%s] type scene %.1fs | A %s lines %s release %.2f | transition %.2f-%.2f | B %s lines %s hit %.2f" % (
        name, time.time() - t0, tt.pa["text"], tt.pa["lines"], tt.pa["release"], tt.tr["start"], tt.tr["start"] + tt.tr["dur"], tt.pb["text"], tt.pb["lines"], tt.pb["hit"]), flush=True)
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), name)))
    proc = dd._ffmpeg_writer(mp4, comp.W, comp.H, a.fps)
    keep, mass = {}, []
    t1 = time.time()
    for i in range(len(frames)):
        fr = comp.frame(i)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        mass.append(round(float(tt.matte(i).mean()), 5))
        if i in SHEET_I:
            keep[i] = fr.copy()
        if i % 12 == 0:
            print("   %s frame %d/%d  %.0fs" % (name, i, len(frames), time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    import re
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    json.dump(dict(transition=name, mp4=mp4, camera=a.camera, pinned_window=(None if a.camera != "engine" else [tt.cam.t0, tt.cam.t1]), clock_warp=tt.warp, b_nudge=nudge, words=[{k: w[k] for k in ("text", "lines", "start", "hit", "release", "end", "cap_px")} for w in tt.words],
                   transition_window=tt.tr, i_start=a.i_start, i_hit=a.i_hit, wf=a.wf, dx=a.dx, type_mass=mass, ink_log=comp.track.log,
                   depth_mp4=a.depth_mp4, matte_dir=a.matte_dir),
              open(os.path.join(local, "transition_%s%s.json" % (name, ("_" + gm.group(1)) if gm else "")), "w"), indent=1)
    print("[%s] -> %s (%.0fs)" % (name, mp4, time.time() - t0), flush=True)
    return keep, mp4


SHEET_I = (0, 20, 26, 32, 40, 48, 56, 64, 72, 80, 100)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-mp4", required=True)
    ap.add_argument("--matte-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="P06:SEARCHING,A LIGHT")
    ap.add_argument("--transition", default="all", help="one of %s, a comma list, or 'all' (one take each, sequential)" % ", ".join(TRANSITIONS))
    ap.add_argument("--flow", default="right", choices=["right", "left"], help="A leaves toward this side, B arrives from the other")
    ap.add_argument("--i-start", type=int, default=24, help="video frame the turn starts (A's exit starts)")
    ap.add_argument("--i-hit", type=int, default=77, help="video frame word B is formed (landing + 18)")
    ap.add_argument("--wf", type=float, default=0.76, help="word box width: 0.76 = the P06 reference's ~72 %% of the frame")
    ap.add_argument("--dx", type=float, default=0.07, help="word A offset (fraction of the frame width, + = right); B mirrored")
    ap.add_argument("--exit-s", type=float, default=0.20)
    ap.add_argument("--dur", type=float, default=1.6, help="transition duration (s): the module's own timing scaled to the window")
    ap.add_argument("--form-s", type=float, default=0.40)
    ap.add_argument("--camera", default="engine", choices=["engine", "home"], help="engine = the transition engine's own camera (CameraChain.auto: hold_drift, FOLLOW through the hand-off) "
                    "pinned to the HOME pose on both holds and eased in / out inside the window (v019); home = static home camera throughout (v018)")
    ap.add_argument("--cam-ease", type=float, default=0.25, help="share of the follow window the camera eases out of / into the home pose")
    ap.add_argument("--fit-b", action=argparse.BooleanOptionalAction, default=True, help="nudge word B so no letter is > 40 %% hidden by the runner on its hold")
    ap.add_argument("--fit-clock", action=argparse.BooleanOptionalAction, default=True,
                    help="re-time the window so the module's visible part fills it (no empty frame between A's exit and B's hit)")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args(argv)
    outd, review, local = dd.version_dirs(a.out)
    frames = DF.read_frames(a.depth_mp4)
    mattes = DF.read_mattes(a.matte_dir, len(frames))
    names = TRANSITIONS if a.transition == "all" else tuple(a.transition.split(","))
    rows, mp4s = {}, {}
    for nm in names:
        keep, mp4s[nm] = run_one(a, nm, frames, mattes, outd, review, local)
        rows[nm] = [(keep[i], "f%d" % i) for i in sorted(keep)]
    sheet = dd.unique_path(os.path.join(review, "transitions_sheet.png"))
    dd._sheet(rows, sheet, tw=120, cols=len(SHEET_I))
    print("sheet ->", sheet, flush=True)


if __name__ == "__main__":
    main()
