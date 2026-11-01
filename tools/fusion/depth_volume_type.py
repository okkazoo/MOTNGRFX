"""depth_volume_type: the flat 2D P06 words (SEARCHING Doto -> A LIGHT Tourney, centred, v020 g04 framing) for the camera-path-through-the-volume transitions
v021 push / v022 pullback / v023 fly (tools/fusion/depth_orbit.py --mode push|pullback|fly).

The words stay FLAT on screen (no plane, no perspective on the glyphs): each word is a camera-facing billboard attached to a point of the world on the optical axis.
Per frame (and per shutter sub-sample) the SAME rigid camera pose as the cards (depth_orbit.volume_pose) gives the billboard's screen position, its scale z_ref / z_cam
and the frame roll; the word's 2D alpha is warped by that similarity (scale + shift + roll, no shear) - so the type travels with the camera exactly, and at both holds
(scale 1, shift 0, roll 0) it is the untouched v020 g04 hold, pixel for pixel.
  push      SEARCHING hangs just in front of image 1 (z 0.9): the dolly takes the camera through it, it scales up and passes the lens (fades as it gets huge);
            A LIGHT hangs just in front of image 2's content (z dz + 0.9): small and far at the start, hidden behind image 1's cards (their per-frame coverage
            `front_alpha`), it is REVEALED as the chunks part and grows to its hold.
  pullback  SEARCHING hangs in image 1's plane and recedes (shrinks) as the camera backs away, hidden by image 2's cards as they arrive in front (front_alpha);
            A LIGHT hangs 2.5 in front of image 2's final camera pose: it flies in from the lens and settles at its hold.
  fly       the push planes, seen through the rolling / banking camera (the words roll with the frame).
The runner matte pass of the same path puts the runner in front (compose.sandwich), legible ink per frame (depth_ribbon_type.InkTrack).

  python tools/fusion/depth_volume_type.py --mode push --depth-mp4 17_depth-dolly/v021/out/..._depth.mp4 --matte-dir .../local/matte_depth --front-dir .../local/front_depth --out 17_depth-dolly/v021
"""
from __future__ import annotations

import argparse
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

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_flat_type as DF  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402
from tools.fusion import depth_ribbon_type as RB  # noqa: E402
from tools.fusion import depth_slide_type as SL  # noqa: E402
from tools.fusion import depth_type as DT  # noqa: E402


def _sm(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


class Billboard:
    """a flat word as a camera-facing billboard at the world point (0, 0, z_w), native size at camera distance z_ref."""

    def __init__(self, z_w, z_ref):
        self.z_w, self.z_ref = float(z_w), float(z_ref)

    def xform(self, pose, f, cx, cy):
        """-> (scale, screen x, screen y, roll deg, z_cam) of the billboard seen from `pose` (volume_pose dict), or None behind the camera."""
        R = do.cam_R(pose["yaw"], pose["pitch"])
        pc = R.T @ (np.array([0.0, 0.0, self.z_w]) - np.asarray(pose["pos"], float))
        if pc[2] <= 0.04:
            return None
        return self.z_ref / pc[2], cx + f * pc[0] / pc[2], cy + f * pc[1] / pc[2], pose["roll"], float(pc[2])


class VolumeType:
    def __init__(self, mode, op, st, f, cx, cy, z_a=0.9, z_b=None):
        self.mode, self.op, self.st = mode, op, st
        self.f, self.cx, self.cy = f, cx, cy
        dz = op.vol_dz
        if mode == "pullback":
            self.A = Billboard(z_a, z_a)
            zb = 2.5 if z_b is None else z_b
            self.B = Billboard(-dz + zb, zb)
        else:
            self.A = Billboard(z_a, z_a)
            zb = 0.9 if z_b is None else z_b
            self.B = Billboard(dz + zb, zb)
        self.aspect = cy * 2.0 / (cx * 2.0)

    def pose(self, q):
        return do.volume_pose(self.mode, q, self.op, self.aspect)


def _warp(a, s, sx, sy, roll, cx, cy, ss=2):
    """similarity warp of the ss-times alpha `a`: scale s about the frame centre, then the billboard shift, then the frame roll about the centre."""
    if abs(s - 1.0) < 1e-9 and abs(sx - cx) < 1e-9 and abs(sy - cy) < 1e-9 and abs(roll) < 1e-9:
        return a
    H1 = np.array([[s, 0.0, sx - s * cx], [0.0, s, sy - s * cy], [0.0, 0.0, 1.0]])
    R = np.vstack([cv2.getRotationMatrix2D((cx, cy), roll, 1.0), [0, 0, 1]])
    M = R @ H1
    D = np.array([[ss, 0.0, (ss - 1) / 2.0], [0.0, ss, (ss - 1) / 2.0], [0.0, 0.0, 1.0]])
    M2 = D @ M @ np.linalg.inv(D)
    return cv2.warpAffine(a, M2[:2], (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class VolumeComposer(RB.RibbonComposer):
    def __init__(self, vt: VolumeType, frames, mattes, fronts, sub=4, shutter=0.55):
        self.vt, self.frames, self.mattes, self.fronts = vt, frames, mattes, fronts
        H, W = frames[0].shape[:2]
        self.W, self.H = W, H
        cap = float(vt.st.words[0].get("cap_px", 200.0)) * 1080.0 / 540.0
        self.track = RB.InkTrack(vt.st.typ.get("palette") or {}, cap)
        self.a0 = DT.place_matte(vt.st._hold(vt.st.wa), W, H, 2)
        self.b0 = DT.place_matte(vt.st._hold(vt.st.wb), W, H, 2)
        self.sub, self.shutter = int(sub), float(shutter)

    def _sub_times(self, i):
        st = self.vt.st
        t = st.t_of(i)
        q0, q1 = st.q_t(t - 0.5 * self.shutter), st.q_t(t + 0.5 * self.shutter)
        moving = abs(q1 - q0) > 1e-9
        n = self.sub if moving else 1
        return [st.q_t(t + self.shutter * ((k + 0.5) / n - 0.5)) if n > 1 else st.q(i) for k in range(n)]

    def parts(self, i, ss=2):
        """(A, B) alpha on the ss grid at frame i: billboard-warped, fades applied, shutter averaged; B (push / fly) or A (pullback) hidden behind the front volume."""
        vt = self.vt
        a0 = self.a0 if ss == 2 else DT.place_matte(vt.st._hold(vt.st.wa), self.W, self.H, ss)
        b0 = self.b0 if ss == 2 else DT.place_matte(vt.st._hold(vt.st.wb), self.W, self.H, ss)
        A = np.zeros_like(a0)
        B = np.zeros_like(b0)
        qs = self._sub_times(i)
        for q in qs:
            pose = vt.pose(q)
            ta = vt.A.xform(pose, vt.f, vt.cx, vt.cy)
            if ta is not None:
                s, sx, sy, roll, z = ta
                g = 1.0
                if vt.mode == "pullback":
                    g = _sm((s - 0.22) / 0.2)                                   # receding: gone by the time image 1 has gone (no ghost left on the final hold)
                else:
                    g = _sm((7.0 - s) / 3.0)                                    # passing the lens: gone before it is a wall
                if g > 0.0:
                    A += g * _warp(a0, s, sx, sy, roll, vt.cx, vt.cy, ss)
            tb = vt.B.xform(pose, vt.f, vt.cx, vt.cy)
            if tb is not None:
                s, sx, sy, roll, z = tb
                g = _sm((s - 0.10) / 0.12) * _sm((8.0 - s) / 3.0)
                if g > 0.0:
                    B += g * _warp(b0, s, sx, sy, roll, vt.cx, vt.cy, ss)
        n = float(len(qs))
        A, B = A / n, B / n
        fa = cv2.resize(self.fronts[i], (A.shape[1], A.shape[0]), interpolation=cv2.INTER_LINEAR) if ss != 1 else self.fronts[i]
        if vt.mode == "pullback":
            A = A * (1.0 - fa)
        else:
            B = B * (1.0 - fa)
        return A, B

    def alpha(self, i, ss=2):
        A, B = self.parts(i, ss)
        return np.clip(np.maximum(A, B), 0.0, 1.0)


def read_alphas(d, n):
    return [cv2.imread(os.path.join(d, "%04d.png" % i), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for i in range(n)]


def build_op(a):
    return do.OrbitParams(mode=a.mode, vol_dz=a.vol_dz, roll_deg=a.roll_deg, fly_amp=(a.fly_amp_x, a.fly_amp_y), fly_yaw=a.fly_yaw, fly_pitch=a.fly_pitch,
                          pull_rise=a.pull_rise, hold1=a.i_start, hold2=108 - a.i_start - 36)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["push", "pullback", "fly"])
    ap.add_argument("--depth-mp4", required=True)
    ap.add_argument("--matte-dir", required=True)
    ap.add_argument("--front-dir", required=True, help="the front volume's per-frame coverage (depth_orbit writes local/front_<tag>/ for these modes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="P06:SEARCHING,A LIGHT")
    ap.add_argument("--i-start", type=int, default=24)
    ap.add_argument("--vol-dz", type=float, default=5.0)
    ap.add_argument("--roll-deg", type=float, default=28.0)
    ap.add_argument("--fly-amp-x", type=float, default=0.9)
    ap.add_argument("--fly-amp-y", type=float, default=0.4)
    ap.add_argument("--fly-yaw", type=float, default=14.0)
    ap.add_argument("--fly-pitch", type=float, default=7.0)
    ap.add_argument("--pull-rise", type=float, default=0.3)
    ap.add_argument("--z-a", type=float, default=0.9)
    ap.add_argument("--z-b", type=float, default=None)
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args(argv)
    outd, review, local = dd.version_dirs(a.out)
    frames = DF.read_frames(a.depth_mp4)
    mattes = DF.read_mattes(a.matte_dir, len(frames))
    fronts = read_alphas(a.front_dir, len(frames))
    t0 = time.time()
    op = build_op(a)
    st = SL.SlideType(a.type, len(frames), a.fps, a.i_start, lines="scan")           # words only: centred, Tourney B (the v020 g04 defaults)
    vt = VolumeType(a.mode, op, st, 414.9321, frames[0].shape[1] / 2.0, frames[0].shape[0] / 2.0, a.z_a, a.z_b)
    comp = VolumeComposer(vt, frames, mattes, fronts)
    print("words %.1fs | mode %s dz %.2f | A %s | B %s" % (time.time() - t0, a.mode, a.vol_dz, st.words[0]["lines"], st.words[1]["lines"]), flush=True)
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), a.tag)))
    proc = dd._ffmpeg_writer(mp4, comp.W, comp.H, a.fps)
    keep, mass = {}, []
    sheet_i = (0, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 70, 100)
    t1 = time.time()
    for i in range(len(frames)):
        fr = comp.frame(i)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        A, B = comp.parts(i, 1)
        mass.append([round(float(A.mean()), 5), round(float(B.mean()), 5)])
        if i in sheet_i:
            keep[i] = fr.copy()
        if i % 12 == 0:
            print("  frame %d/%d  %.0fs" % (i, len(frames), time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    import re
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    gen = ("_" + gm.group(1)) if gm else ""
    sheet = dd.unique_path(os.path.join(review, "volume_sheet%s.png" % gen))
    dd._sheet({a.mode: [(keep[i], "f%d" % i) for i in sorted(keep)]}, sheet, tw=120, cols=len(keep))
    json.dump(dict(mode=a.mode, args=vars(a), mass_A_B=mass, ink_log=comp.track.log), open(os.path.join(local, "volume_type%s.json" % gen), "w"), indent=1)
    print("volume type -> %s (%.0fs) sheet %s" % (mp4, time.time() - t0, sheet), flush=True)


if __name__ == "__main__":
    main()
