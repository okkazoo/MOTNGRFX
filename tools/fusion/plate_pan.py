"""plate_pan - compose op: the footage plate travels with the type camera (user v005 review: "the text transitions through the camera travel
were unrelated to the camera movement").

    plate_pan(img, dx, dy, vx, vy, ...)   numpy op registered in compose._ops_registry (tools/fusion/pan_plan.py bakes the arguments)

The plate is a tiled world plane at the type's depth. Its content moves by (dx, dy) (fractions of the frame, +dy = down, the cumulative on-screen
displacement of the type's plane taken from the REAL type camera, pan_plan.plate_track) so a picture point and a letter at the same depth move the
same number of pixels. `vx, vy` = displacement per frame: the shutter integrates it into a directional motion blur whose length is proportional to the
speed (taps along the motion, shutter 0.5 = 180 deg). Outside the plate the tile wraps (a travel of exactly one frame-span lands on the same
picture), with a soft seam: while the travel is under way (`seam_amt` 1) the rows/columns within `seam` of the tile edge are cross-faded to the
reflected picture, so the wrap edge does not scroll through the frame as a hard line; at the holds `seam_amt` is 0 and the plate is exactly the
shot's frame. Optional depth parallax (`depth` near = 1, `parallax` 0..1): near pixels move more than the type plane, far ones less
(factor eye / (eye + z)); the default 0 keeps one plane (the multiplane plate does real parallax with the same camera).

Convention: out(x, y) = img(x - dx, y - dy) like compose `translate`; every argument is a number (frames fractions) at the frame being composed.
"""
from __future__ import annotations

import cv2
import numpy as np


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sample(a, mx, my, mode, seam, seam_amt, moving_x, moving_y):
    """bilinear sample of `a` at the float coordinates (mx, my) with the tile mode. wrap = periodic tile (+ soft seam), mirror, clamp."""
    h, w = a.shape[:2]
    mx32, my32 = mx.astype(np.float32), my.astype(np.float32)
    if mode == "wrap":
        wx, wy = np.mod(mx32, w).astype(np.float32), np.mod(my32, h).astype(np.float32)
        out = cv2.remap(a, wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if seam_amt > 1e-4 and seam > 0:
            mir = cv2.remap(a, mx32, my32, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            b = np.zeros(wx.shape, np.float32)
            if moving_x:
                b = np.maximum(b, 1.0 - _smooth(np.minimum(wx, w - 1 - wx) / max(seam * w, 1.0)))
            if moving_y:
                b = np.maximum(b, 1.0 - _smooth(np.minimum(wy, h - 1 - wy) / max(seam * h, 1.0)))
            b = (b * float(seam_amt))[..., None] if a.ndim == 3 else b * float(seam_amt)
            out = out * (1.0 - b) + mir * b
        return out
    border = cv2.BORDER_REFLECT_101 if mode == "mirror" else cv2.BORDER_REPLICATE
    return cv2.remap(a, mx32, my32, cv2.INTER_LINEAR, borderMode=border)


def plate_pan(img, dx=0.0, dy=0.0, vx=0.0, vy=0.0, depth=None, parallax=0.0, eye=1400.0, z_near=-120.0, z_far=260.0, mode="wrap",
              shutter=0.5, seam=0.16, seam_amt=0.0, max_taps=17):
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    sx, sy = float(dx) * w, float(dy) * h                     # px
    px, py = float(vx) * w, float(vy) * h                     # px / frame
    speed = float(np.hypot(px, py))
    if abs(sx) + abs(sy) < 1e-4 and speed < 1e-4:
        return a
    taps = int(min(max_taps, max(1, np.ceil(speed * float(shutter) / 1.5) + 1))) if speed * float(shutter) > 0.75 else 1
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fac = None
    if depth is not None and float(parallax) > 0:
        d = np.asarray(depth, np.float32)
        d = d[..., 0] if d.ndim == 3 else d
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if d.shape != (h, w) else d
        z = float(z_far) + (float(z_near) - float(z_far)) * np.clip(d, 0.0, 1.0)         # near = 1
        fac = 1.0 + float(parallax) * (float(eye) / np.maximum(float(eye) + z, 50.0) - 1.0)
    acc = np.zeros_like(a, np.float32)
    mvx, mvy = abs(sx) + abs(px) > 0.5, abs(sy) + abs(py) > 0.5
    for k in range(taps):
        o = (k / (taps - 1) - 0.5) if taps > 1 else 0.0
        ox, oy = sx + px * float(shutter) * o, sy + py * float(shutter) * o
        if fac is None:
            mx, my = xx - ox, yy - oy
        else:
            mx, my = xx - ox * fac, yy - oy * fac
        acc += _sample(a, mx, my, mode, float(seam), float(seam_amt), mvx, mvy)
    out = acc / taps
    return out if out.ndim == a.ndim else out[..., None]


OPS = {"plate_pan": plate_pan}
