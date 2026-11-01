"""Flow pass: dense forward optical flow between consecutive frames of the type animation.

1. ANALYTIC camera flow (exact for everything that is fixed in the world): every pixel is lifted to a world point (type pixels:
   from the G-buffer depth; ground pixels: the ray / ground-plane hit; sky: the ray direction at infinity) and projected with the
   camera of the next frame. Flow = p1 - p0 in full-res pixels.
2. OBJECT residual (mode 'hybrid', default): what the type itself does (strands travelling, words forming) is not camera motion.
   The next frame's coverage is warped back by the camera flow; cv2 DIS flow on the blurred coverage of frame t vs the warped
   frame t+1 gives the residual, kept only where the coverage actually changed, smoothed with a mask-weighted Gaussian, and added.
Written at half resolution (H/2 x W/2 x 2, float16, units = half-res pixels, x right / y down) plus a colour-wheel video."""
import numpy as np
import cv2

from tools.fusion import clay


def project(cam, k, size, P):
    """world points (h, w, 3) -> (xy (h, w, 2), valid)."""
    w, h = size
    Q = (P - cam.eye.astype(np.float32)) @ cam.R.T.astype(np.float32)
    z = Q[..., 2]
    ok = z > 1e-3
    zz = np.where(ok, z, 1.0)
    F = cam.F * k
    xy = np.stack([w / 2.0 + F * Q[..., 0] / zz, h / 2.0 + F * Q[..., 1] / zz], -1)
    return xy.astype(np.float32), ok


def camera_flow(G, cam0, cam1, size, k, ground_y, bg="camera"):
    w, h = size
    d = clay.rays(cam0, size, k)
    t, hit = clay.ground_hits(cam0, d, ground_y)
    eye0 = cam0.eye.astype(np.float32)
    Pg = eye0 + t[..., None] * d
    # sky: a point far along the ray (rotation-only motion in the limit)
    Ps = eye0 + d * np.float32(1e6)
    P = np.where(hit[..., None], Pg, Ps)
    cov = G.A > 1e-3
    if cov.any():
        Dd = G.depth()
        Pt = eye0 + d * Dd[..., None]                       # camera z = depth, ray has camera z = 1
        P = np.where(cov[..., None], Pt, P)
    xy1, ok = project(cam1, k, size, P.astype(np.float32))
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    fl = xy1 - np.stack(np.meshgrid(xs, ys), -1)
    fl = np.where(ok[..., None], fl, 0.0)
    if bg == "zero":
        fl = np.where(cov[..., None], fl, 0.0)
    return fl.astype(np.float32)


def _smooth_masked(F, m, sigma):
    num = cv2.GaussianBlur(F * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def residual(A0, A1, flow_cam, size):
    """object motion left after the camera flow, from the coverage of frame t (A0) and t+1 (A1). -> (h, w, 2) float32."""
    w, h = size
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    g = np.stack(np.meshgrid(xs, ys), -1)
    m = g + flow_cam - 0.5                                   # sample frame t+1 where the camera says the pixel went
    A1w = cv2.remap(A1, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    diff = cv2.GaussianBlur(np.abs(A0 - A1w), (0, 0), 3.0)
    if diff.max() < 0.02:
        return np.zeros((h, w, 2), np.float32)
    s = 2.5
    I0 = np.clip(cv2.GaussianBlur(A0, (0, 0), s) * 255.0, 0, 255).astype(np.uint8)
    I1 = np.clip(cv2.GaussianBlur(A1w, (0, 0), s) * 255.0, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    res = dis.calc(I0, I1, None).astype(np.float32)
    support = cv2.GaussianBlur(np.maximum(A0, A1w), (0, 0), 6.0)
    conf = np.clip(diff * 8.0, 0, 1) * np.clip(support * 4.0, 0, 1)
    res = _smooth_masked(res, conf, 5.0) * np.clip(conf * 1.5, 0, 1)[..., None]
    return res


def half_res(flow):
    h, w = flow.shape[:2]
    return (cv2.resize(flow, (w // 2, h // 2), interpolation=cv2.INTER_AREA) * 0.5).astype(np.float16)


def colour_wheel(flow, ref):
    """flow (h, w, 2) -> BGR uint8: hue = direction (red = +x / right, cyan = left, green-ish = down), value = sqrt(magnitude / ref)."""
    fx, fy = flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
    mag = np.hypot(fx, fy)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.sqrt(np.clip(mag / max(ref, 1e-6), 0, 1)) * 255           # sqrt: small flows stay visible
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
