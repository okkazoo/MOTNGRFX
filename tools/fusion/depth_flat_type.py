"""depth_flat_type: v016 - FLAT 2D type sandwiched between a depth-orbit video and the runner, built from the existing pieces only.

User (v015 review): "no not what I wanted - can we just use the text flat - not on a plane and transition it off to the right and reform
on the next video from the left - use the module chains that have already been written" + "use the matte though" + "the matte could also go
through the volume breakup as well ... which would follow the video correctly when it breaks up".

Pieces reused (nothing rebuilt):
  * the depth video + the runner's MATTE PASS = `depth_orbit.py render ... --matte-pass` (PivotScene.render_matte_t: the per-frame Alpha-Gen
    matte through the SAME cards / camera / blur / fades as the colour, so it follows the runner when the volume breaks up);
  * the words: 15_fusion/pieces/P06_v003.json type.words (SEARCHING: Doto, lanes; A LIGHT: GeistPixel, lanes, lines A / LIG / HT), rendered
    by tools/fusion/spec.build_scene with the `pan` transition (mograph/modules/transitions/pan.py + camera/pan_dir.py): ONE camera travel
    along the line the two words sit on. Camera dir 'left' = the picture slides to the RIGHT: A leaves off the right edge, B arrives from the left.
    A `gap` makes the two words sit several frame-spans apart, so with the quintic ease A is gone early in the travel and B only comes
    into view late (after the depth video has landed), where B's form-in (`Pan.entry_win`) plays as it arrives;
  * placement beside the runner + ink: depth_type.place_beside / choose_ink (tools/fusion/legible_color.choose against the depth video where
    each word is read, the runner excluded), the word is only translated in 2D (no plane, no perspective);
  * the 2D composite = compose.sandwich: the type over the depth video, then the video back OVER it by the runner matte (type between the
    background and the runner).

  python tools/fusion/depth_flat_type.py --depth-mp4 17_depth-dolly/v016/out/motion-graphic_depth-dolly_v016_depth.mp4 \
      --matte-dir 17_depth-dolly/v016/local/matte_depth --out 17_depth-dolly/v016 --type "P06:SEARCHING,A LIGHT"
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_type as DT  # noqa: E402

MATTE_SIZE = DT.MATTE_SIZE                    # (1152, 2048): the type engine's 9:16 size (multiples of 64), resampled to the video size
WHITE = dict(bg=[0, 0, 0], fg=[255, 255, 255], accents=[[255, 255, 255]] * 3)


def _static_axes(axes):
    """piece axes (scalar | [lo, hi, 'beat'] animated) -> constants (the middle of an animated range): the pan transition drops the animation."""
    out = {}
    for k, v in (axes or {}).items():
        out[k] = 0.5 * (float(v[0]) + float(v[1])) if isinstance(v, (list, tuple)) else float(v)
    return out


@contextlib.contextmanager
def static_registry_axes(words):
    """spec.prepare_words (pan) drops a word's animated `axes`, which would leave the face at the registry default (GeistPixel ELSH 0 = solid
    squares instead of the piece's dotted look). The piece's values are put in the font registry for the duration of the build instead."""
    from mograph.modules import core
    from tools.fusion import spec as S
    saved = []
    for w in words:
        reg = S._REG.get(core._norm_font(w["font"]))
        if reg is not None and w.get("axes"):
            saved.append((reg, dict(reg.get("axes") or {})))
            reg["axes"] = {**(reg.get("axes") or {}), **_static_axes(w["axes"])}
    try:
        yield
    finally:
        for reg, ax in saved:
            reg["axes"] = ax


class FlatType:
    """Two words of a piece on ONE pan travel. Video frame i <-> type clock tau(i) = tau_start + (i - i_start) / fps.
    i_start: the travel starts (the depth video's turn starts); i_hit: the camera has settled on word B and B is formed."""

    def __init__(self, flag, n_frames, fps=24, i_start=24, i_hit=77, direction="left", fill=0.7, gap=4.0, size=MATTE_SIZE,
                 a_form=0.5, a_exit=0.3, entry_win=(0.72, 1.0), shutter=0.55, ease="io"):
        from tools.fusion import spec as S
        pid, texts = DT.parse_flag(flag)
        self.typ = json.load(open(DT.piece_file(pid), encoding="utf-8"))["type"]
        self.fps, self.n_frames, self.i_start, self.i_hit, self.shutter = int(fps), int(n_frames), int(i_start), int(i_hit), float(shutter)
        self.size = tuple(size)
        A, B = (dict(DT.find_word(self.typ, t)) for t in texts)
        for w in (A, B):
            w.pop("t0", None), w.pop("t1", None), w.pop("transition", None)
        tv = (self.i_hit - self.i_start) / float(self.fps)
        A.update(t0=0.0, form_s=a_form, exit_s=a_exit, t1=a_form + 1.0 + a_exit,
                 transition=dict(kind="pan", dir=direction, dur=tv, ease=ease, gap=gap, fill=fill, entry_io=True))
        hold_b = (self.n_frames - self.i_hit) / float(self.fps) + 1.5
        B.update(t0=0.0, form_s=0.75, exit_s=0.375, t1=0.75 + hold_b + 0.375, entry=direction)     # B's strands stream in from the side the camera came from
        spec = dict(size=list(self.size), fps=self.fps, duration=a_form + 1.0 + tv + hold_b + 2.0, bpm=None, bar=4, camera="auto", words=[A, B],
                    palette=WHITE, **{k: self.typ[k] for k in ("seed", "energy", "mood", "line") if k in self.typ})
        with static_registry_axes([A, B]):
            self.sc = S.build_scene(spec, self.size)
        self.words = self.sc.info["words"]
        self.pan = self.sc.info["pans"][0]
        for it, _a, _b in self.sc.seq.timeline:
            if getattr(it, "name", "") == "pan":
                it.entry_win = tuple(entry_win)                               # B's form-in plays only as B comes into view
        self.tau_start = float(self.pan["start"])
        assert self.words[0]["hit"] <= self.tau(0) + 1e-6, "word A is still forming at video frame 0 (lengthen its hold)"
        self.origin = None
        words = [it for it, _, _ in self.sc.seq.timeline if hasattr(it, "placement")]
        self.origin = words[0].placement.origin[None, :]
        self._cache = {}

    # --- clock
    def tau(self, i):
        return self.tau_start + (float(i) - self.i_start) / self.fps

    def speed_px(self, tau, wmax):
        """on-screen travel of the word centre over one shutter (px, at video width wmax)."""
        h = 0.5 * self.shutter / self.fps
        ca, cb = self.sc.chain.at(tau - h), self.sc.chain.at(tau + h)
        pa, pb = ca.project(self.origin)[0][0], cb.project(self.origin)[0][0]
        return float(np.abs(pb - pa).max()) * wmax / self.size[0]

    def matte(self, i, wmax=432, step_px=2.0, max_sub=10):
        """float32 (h, w) 0..1 of the type at video frame i: the pan camera's own motion, blurred over the shutter (sub-frame average,
        ~step_px spacing at the video's width)."""
        if i in self._cache:
            return self._cache[i]
        tau = self.tau(i)
        n = int(np.clip(math.ceil(self.speed_px(tau, wmax) / step_px), 1, max_sub))
        acc = None
        for k in range(n):
            t = tau + self.shutter / self.fps * ((k + 0.5) / n - 0.5)
            f = self.sc.seq.frame(t, self.sc.chain, size=self.size, S=1).max(-1).astype(np.float32) / 255.0
            acc = f if acc is None else acc + f
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[i] = acc / n
        return self._cache[i]


def bbox_of(m, thr=0.1):
    ys, xs = np.nonzero(m > thr)
    return (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))


class Composer:
    """words -> placed (beside the runner) + inked flat type -> compose.sandwich with the runner matte."""

    def __init__(self, ft: FlatType, frames, mattes, read=(12, None)):
        self.ft, self.frames, self.mattes = ft, frames, mattes
        H, W = frames[0].shape[:2]
        self.W, self.H = W, H
        self.i_switch = (ft.i_start + ft.i_hit) // 2                          # nothing of either word is on screen here (the dead gap)
        self.read = (read[0], ft.i_hit + 12 if read[1] is None else read[1])
        self.place, self.ink, self.report = [], [], []
        k = W / ft.size[0]
        for which in (0, 1):
            ir = self.read[which]
            m = ft.matte(ir)
            bb = bbox_of(m)
            runner = mattes[ir]
            ww = (bb[1] - bb[0]) * k

            def alpha_at(cx, bb=bb, m=m):
                return cv2.resize(DT.place_matte(m, W, H, 2, 1.0, (cx, cy_home), bb), (W, H), interpolation=cv2.INTER_AREA)
            cy_home = 0.5 * (bb[2] + bb[3]) * k + (H - ft.size[1] * k) / 2.0
            cx, right = DT.place_beside(runner, ww, W, lambda cx: float(((runner > 0.5) & (alpha_at(cx) > 0.25)).sum() / max((alpha_at(cx) > 0.25).sum(), 1)))
            a_home = alpha_at(cx)
            info = ft.words[which]
            cap = float(info.get("cap_px", 200.0)) * 1080.0 / 540.0
            ink = DT.choose_ink(frames[ir], a_home, ft.typ.get("palette") or {}, cap, hide=runner)
            self.place.append((cx, cy_home, bb))
            self.ink.append(ink)
            self.report.append(dict(word=ft.words[which]["text"], read_frame=ir, centre_px=[round(cx, 1), round(cy_home, 1)], free_side="right" if right else "left",
                                    hidden_by_runner=round(float(((runner > 0.5) & (a_home > 0.25)).sum() / max((a_home > 0.25).sum(), 1)), 3),
                                    ink=ink["hex"], palette_pick=ink["name"], kind=ink["kind"], support=ink["support"],
                                    contrast_mean=round(ink["ratio_mean"], 2), share_over_target=round(ink["share_ok"], 3), target=ink["target"]))

    def word_of(self, i):
        return 0 if i < self.i_switch else 1

    def type_layer(self, i):
        """(premultiplied BGR 0..255 (H,W,3), alpha (H,W)) of the flat type at frame i (placed, inked, support under it)."""
        w = self.word_of(i)
        cx, cy, bb = self.place[w]
        a2 = DT.place_matte(self.ft.matte(i, self.W), self.W, self.H, 2, 1.0, (cx, cy), bb)
        pm = cv2.resize(DT.premult_tex(a2, self.ink[w]), (self.W, self.H), interpolation=cv2.INTER_AREA)
        return pm[..., :3], np.clip(pm[..., 3], 0.0, 1.0)

    def frame(self, i):
        """BGR uint8: the depth video frame with the type between it and the runner (compose.sandwich: type over the footage, then the
        footage over the type by the runner matte)."""
        pm, a = self.type_layer(i)
        foot = self.frames[i][..., ::-1].astype(np.float32) / 255.0
        rgb = (pm[..., ::-1] / 255.0) / np.maximum(a, 1e-6)[..., None]
        out = C.sandwich(foot, rgb.astype(np.float32), a, self.mattes[i].astype(np.float32), matte_threshold=None)
        return np.clip(np.asarray(out)[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- io
def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def read_mattes(d, n):
    return [cv2.imread(os.path.join(d, "%04d.png" % i), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-mp4", required=True)
    ap.add_argument("--matte-dir", required=True, help="the --matte-pass PNGs of the same depth render (NNNN.png, 8-bit)")
    ap.add_argument("--out", required=True, help="version folder, e.g. 17_depth-dolly/v016")
    ap.add_argument("--type", default="P06:SEARCHING,A LIGHT")
    ap.add_argument("--dir", default="left", choices=["left", "right", "up", "down"], help="CAMERA travel; 'left' = the picture (type) slides right")
    ap.add_argument("--fill", type=float, default=0.5, help="how much of the frame a word fills at its hold (engine default 0.7; 0.5 keeps the type small)")
    ap.add_argument("--gap", type=float, default=4.0, help="extra frame-spans between the two words (dead air in the middle of the travel)")
    ap.add_argument("--i-start", type=int, default=24, help="video frame the depth turn (and the type travel) starts")
    ap.add_argument("--i-hit", type=int, default=77, help="video frame the camera has settled on word B and B is formed (landing + 18)")
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args(argv)
    outd, review, local = dd.version_dirs(a.out)
    frames = read_frames(a.depth_mp4)
    mattes = read_mattes(a.matte_dir, len(frames))
    t0 = time.time()
    ft = FlatType(a.type, len(frames), a.fps, a.i_start, a.i_hit, a.dir, a.fill, a.gap)
    comp = Composer(ft, frames, mattes)
    print("type scene %.1fs | pan window %.2f-%.2f s (video frames %d-%d), words %s" % (
        time.time() - t0, ft.pan["start"], ft.pan["end"], a.i_start, a.i_hit, [(w["text"], w["hit"]) for w in ft.words]), flush=True)
    for r in comp.report:
        print("word:", r, flush=True)
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), a.tag)))
    proc = dd._ffmpeg_writer(mp4, comp.W, comp.H, a.fps)
    keep = {}
    t1 = time.time()
    sheet_i = sorted(set(int(x) for x in np.linspace(a.i_start - 2, a.i_hit + 14, 12)) | {0, len(frames) - 1})
    for i in range(len(frames)):
        fr = comp.frame(i)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        if i in sheet_i:
            keep[i] = fr.copy()
        if i % 6 == 0:
            print("  frame %d/%d  %.0fs" % (i, len(frames), time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    import re
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    gen = ("_" + gm.group(1)) if gm else ""
    tiles = [(keep[i], "f%d" % i) for i in sorted(keep)]
    sheet = dd.unique_path(os.path.join(review, "flat_type_sheet%s.png" % gen))
    dd._sheet({"flat type + runner matte": tiles}, sheet, tw=140, cols=len(tiles))
    json.dump(dict(flag=a.type, dir=a.dir, fill=a.fill, gap=a.gap, i_start=a.i_start, i_hit=a.i_hit, i_switch=comp.i_switch, pan=ft.pan,
                   words=[{k: w[k] for k in ("text", "start", "hit", "release", "end")} for w in ft.words], report=comp.report,
                   depth_mp4=a.depth_mp4, matte_dir=a.matte_dir),
              open(os.path.join(local, "flat_type%s.json" % gen), "w"), indent=1)
    print("flat type -> %s (%.0fs) sheet %s" % (mp4, time.time() - t0, sheet), flush=True)


if __name__ == "__main__":
    main()
