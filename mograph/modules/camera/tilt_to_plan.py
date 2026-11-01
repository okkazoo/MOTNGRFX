"""tilt_to_plan: rise to top-down so the 3D structure reads as a flat pattern. The eye keeps its radius around a slowly moving
pivot (the strand focus), raises its elevation to `elev` degrees (looking down onto the plan), stays up with a slow drift, and
comes back down to the formed word's face-on view. The image-right hint (the horizon-level right of the track's azimuth) fades in as the view nears vertical so the
image cannot spin; roll stays 0. Rise and return are quintic eases of the elevation (a single continuous bump, no tilt-then-correct)."""
import math
import numpy as np
from mograph.modules.camera.base import (CameraModule, register_camera, smoother, smooth_series, lerp, vdir)
from mograph.modules.camera.follow import Follow, anchor_path
from mograph.render import cam3d


@register_camera("tilt_to_plan")
class TiltToPlan(CameraModule):
    description = ("Rises to a top-down view so the 3D layering of the strands collapses into a flat plan pattern, rides there "
                   "with a slow drift, then comes back down to face-on. The image-right hint fades in near vertical so the picture "
                   "never spins.")
    DEFAULTS = dict(elev=68.0, rise=0.45, down=0.4, radius_scale=1.15, fill=0.7, gain=0.5, smooth=0.5)

    def _bump(self, ts, t0, t1):
        d = t1 - t0
        up = smoother((ts - t0) / (self.p("rise") * d))
        dn = 1.0 - smoother((ts - (t1 - self.p("down") * d)) / (self.p("down") * d))
        return up * dn

    def curves(self, hints, ts, t0, t1):
        fol = Follow()
        el = math.radians(self.p("elev"))
        bump = self._bump(ts, t0, t1)

        def pivot(tt):
            return lerp(anchor_path(hints, tt), hints.focus(tt), self.p("gain"))

        def nd(tt):
            nrm, dist, hw, he, hl = fol._track(hints, tt, entry=False)
            return np.c_[nrm, dist]
        piv = smooth_series(pivot, ts, self.p("smooth"))
        X = smooth_series(nd, ts, 0.25)
        nrm, dist = X[:, :3], X[:, 3] * self.p("radius_scale") / fol.p("close")
        trs = [tr for tr in hints.trans if tr.b > t0 and tr.a < t1]
        if trs:                                    # up in the plan the whole transition fits: centre on it and back off to take it all in
            tr = max(trs, key=lambda q: min(q.b, t1) - max(q.a, t0))
            B = tr.bounds()
            ext = max(float((B.max(0) - B.min(0))[0]), float((B.max(0) - B.min(0))[2])) * 1.05
            d_plan = hints.F * ext / (0.85 * cam3d.W)
            piv = piv + (0.5 * (B.min(0) + B.max(0)) - piv) * (0.8 * bump)[:, None]
            dist = dist * (1.0 + bump * float(np.clip(d_plan / float(dist.mean()) - 1.0, 0.0, 0.25)))      # one constant factor: no kinks
        yaw = np.arctan2(nrm[:, 0], nrm[:, 2])
        pit0 = np.arcsin(np.clip(nrm[:, 1] / np.maximum(np.linalg.norm(nrm, axis=1), 1e-9), -1, 1))
        pitch = pit0 + (el - pit0) * bump
        dist = dist * (1.0 + 0.06 * bump * smoother((ts - t0) / (t1 - t0)))      # a slow pull-back while up in the plan (steady, no wobble)
        eye = piv - dist[:, None] * vdir(yaw, pitch)
        # near the vertical the horizon-level image-right is undefined, so the image-right hint is the horizon-level right of the
        # track's own azimuth (cos yaw, 0, -sin yaw): identical to the horizon-level right at oblique views, so fading it in
        # rolls nothing, whatever azimuth the camera is looking along
        rw = smoother((np.degrees(pitch) - 50.0) / 25.0)
        rv = np.stack([np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)], 1)
        return dict(eye=eye, look=piv, right_w=rw, right_v=rv)
