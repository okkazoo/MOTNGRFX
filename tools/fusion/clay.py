"""Clay pass: the type as light-grey shaded solids on a mid-grey ground with a soft horizon, like a 3D viewport playblast.
Everything is grey (R = G = B), no text-like overlays. Shading: Lambert from a camera-attached key light on the per-pixel normal
(planar faces tilt with the camera, so an orbit reads), a bevel from the signed distance to the type edge (rounded clay edges),
depth fog toward the horizon tone, and a soft contact shadow. The ground is a world plane (y = ground_y) with a faint large
checker that fades with distance; above the horizon a soft gradient. Both are computed from the real camera, so every camera
move reads in the backdrop too."""
import numpy as np
import cv2

from mograph.render import cam3d

LIGHT = np.array([-0.45, -0.55, -0.70], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
HORIZON = 0.62            # backdrop value on the horizon
SKY_TOP = 0.46
GROUND_NEAR = 0.40
TYPE_AMB, TYPE_DIFF = 0.34, 0.70


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rays(cam, size, k):
    """world-space ray directions (h, w, 3) (not normalised; camera z = 1) for pixel centres."""
    w, h = size
    F = cam.F * k
    xs = (np.arange(w, dtype=np.float32) + 0.5 - w / 2.0) / F
    ys = (np.arange(h, dtype=np.float32) + 0.5 - h / 2.0) / F
    return (xs[None, :, None] * cam.r.astype(np.float32) + ys[:, None, None] * cam.d.astype(np.float32)
            + cam.f.astype(np.float32))


def ground_hits(cam, d, ground_y):
    """per pixel: (t, hit mask) of the ray eye + t d with the plane y = ground_y (double sided; t clamped for the far field)."""
    dy = d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_y - cam.eye[1]) / dy
    hit = (np.abs(dy) > 1e-5) & (t > 0) & (t < 2e5)
    return np.where(hit, t, 0.0).astype(np.float32), hit


def backdrop(cam, size, k, ground_y, pivot, grid=True):
    w, h = size
    d = rays(cam, size, k)
    dn = d / np.linalg.norm(d, axis=2, keepdims=True)
    t, hit = ground_hits(cam, d, ground_y)
    elev = -dn[..., 1]
    sky = SKY_TOP + (HORIZON - SKY_TOP) * np.exp(-np.maximum(elev, 0.0) / 0.45)
    P = cam.eye.astype(np.float32) + t[..., None] * d
    rr = np.hypot(P[..., 0] - cam.eye[0], P[..., 2] - cam.eye[2])
    gnd = GROUND_NEAR + (HORIZON - GROUND_NEAR) * (1.0 - np.exp(-rr / 1500.0))
    if grid:
        s = 180.0
        c = np.sin(np.pi * P[..., 0] / s) * np.sin(np.pi * P[..., 2] / s)
        gnd = gnd + 0.035 * np.tanh(4.0 * c) * np.exp(-rr / 2200.0)
    v = np.where(hit, gnd, sky)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2.0) / (w / 2.0)) ** 2 + ((yy - h / 2.0) / (h / 2.0)) ** 2
    return (v * (1.0 - 0.12 * r2 / 2.0)).astype(np.float32)


def render(G, cam, size, k, near, far, ground_y, pivot, bevel=0.0075, fog=0.30, shadow=0.30, grid=True):
    """GBuffer -> uint8 grey (h, w)."""
    w, h = size
    A = G.A
    bd = backdrop(cam, size, k, ground_y, pivot, grid)
    if shadow > 0 and A.max() > 0:
        sh = cv2.GaussianBlur(A, (0, 0), 0.012 * w)
        M = np.float32([[1, 0, 0.010 * w], [0, 1, 0.018 * w]])
        sh = cv2.warpAffine(sh, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        bd = bd * (1.0 - shadow * np.clip(sh, 0, 1))
    if A.max() <= 1e-4:
        return np.clip(bd * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mask = (A > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    bw = max(2.0, bevel * w)
    Hh = bw * _ss(dist / bw)
    gy, gx = np.gradient(Hh)
    n = G.normal()
    n[..., 0] += -gx * 0.9
    n[..., 1] += -gy * 0.9
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
    lam = np.clip(n @ LIGHT, 0.0, 1.0)
    v = TYPE_AMB + TYPE_DIFF * lam
    v = v * (1.0 - 0.14 * (1.0 - _ss(dist / (3.0 * bw))))
    Dd = G.depth()
    f = fog * _ss((Dd - near) / max(far - near, 1e-6))
    v = v + (HORIZON - v) * f
    out = bd * (1.0 - A) + v * A
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
