"""ECHO CAP on compose.flow_wake (user 2026-11-01: "the echo is too strong and covers too much of the image").

The offending rules args from the WINTER CRANE render are bounded, and the wake stays in a small band around the letters.
"""
import cv2
import numpy as np

from tools.fusion import compose as C

OFFENDING = {"bg_drag": 0.12, "decay": 0.9, "flow": "$flow", "matte": "$flow_m", "max_disp": 0.2, "radius": 70,
             "smooth": 1.0, "strength": {"curve": "onset", "hi": 6.0, "lo": 3.0}, "trail": 10, "trail_gain": 1.0,
             "type_a": "$type_matte", "u": "$u"}


def test_offending_args_are_capped():
    c = C.cap_wake_args(OFFENDING, short_side=540)
    assert c["trail"] == 3.0 and c["trail_gain"] == 0.4 and c["max_disp"] == 0.04
    assert c["decay"] == 0.85 and c["bg_drag"] == 0.05
    assert abs(c["radius"] - 0.03 * 540) < 1e-6                                   # 16.2 px at 540x960
    assert c["strength"] == {"curve": "onset", "hi": 2.0, "lo": 2.0}
    assert c["flow"] == "$flow" and c["smooth"] == 1.0                            # other args untouched
    assert OFFENDING["trail"] == 10                                              # input not mutated
    small = C.cap_wake_args({"trail": 2, "trail_gain": 0.2, "radius": 5, "strength": 1.0}, 540)
    assert small == {"trail": 2.0, "trail_gain": 0.2, "radius": 5.0, "strength": 1.0}   # values under the cap pass through


def _run(capped, n=16, w=270, h=480):
    saved = dict(C.ECHO_CAP)
    if not capped:
        C.ECHO_CAP.update({k: 1e9 for k in C.ECHO_CAP})
    try:
        C.set_backend("numpy")
        ta = np.zeros((h, w), np.float32)
        for x in range(30, 240, 36):                                             # a row of block letters
            ta[200:260, x:x + 22] = 1.0
        st, outs = C.FlowWake(), []
        args = {k: OFFENDING[k] for k in ("bg_drag", "decay", "max_disp", "smooth", "trail", "trail_gain")}
        for i in range(n):
            yy, xx = np.mgrid[0:h // 2, 0:w // 2]
            m = np.exp(-(((xx - (-15 + 5 * i)) / 28) ** 2 + ((yy - 115) / 45) ** 2) * 2.0).astype(np.float32)
            fl = np.zeros((h // 2, w // 2, 2), np.float32)
            fl[..., 0] = 0.5 + 10.0 * m
            outs.append(st.step(ta, fl, cv2.resize(m, (w, h)), strength=6.0, radius=70 * w / 540, settle=0.0, **args))
        return ta, outs
    finally:
        C.ECHO_CAP.clear()
        C.ECHO_CAP.update(saved)


def test_wake_stays_in_a_small_band_around_the_type():
    band = 0.03 * 270
    res = {}
    for capped in (False, True):
        ta, outs = _run(capped)
        dist = cv2.distanceTransform((ta < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        cov = max(float(((o - ta) > 0.05).mean()) for o in outs)
        far = max(float(((o > 0.05) & (dist > band)).mean()) for o in outs)
        res[capped] = (cov, far)
    (cov0, far0), (cov1, far1) = res[False], res[True]
    assert far0 > 0.005                                       # uncapped: the offending args smear well outside the band
    assert far1 < 0.002 and far1 < far0 / 5                   # capped: almost nothing beyond 3 % of the short side
    assert cov1 < cov0 and cov1 < 0.06                        # and the wake covers less of the frame
