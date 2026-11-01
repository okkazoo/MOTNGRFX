"""No soft mask / scrim / feathered window / dark or light surround around the type (user 2026-11-01: "ive said before not to use soft masks around text").
Trigger: render R15, piece mvxc7w6vqz6 shot s1: the x10 `outline_to_word` effect draws a dark ground over the shot; the word does not read in it, so compose.hold_net fired and
blended the picture toward the plate inside a gaussian-feathered dilation of the word = a soft rounded window of footage round STILLNESS on a black frame.
Run: python -m pytest --rootdir=tests -q -p no:cacheprovider tests/test_no_soft_surround.py     (CPU, synthetic data)"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W, N = 160, 96, 24
GROUND = (0.04, 0.045, 0.05)                       # the dark ground of the effect (outline_to_word fills the frame with it)
PLATE = 0.45
FACE = np.array([0.95, 0.8, 0.7], np.float32)


def _word():
    """a filled word: three bars (hard letter shapes)."""
    a = np.zeros((H, W), np.float32)
    for x0 in (10, 38, 66):
        cv2.rectangle(a, (x0, 50), (x0 + 20, 110), 1.0, -1)
    return a


def _write_type(tmp, a):
    beauty = np.zeros((H, W, 3), np.float32) + np.array(GROUND, np.float32)
    beauty = beauty * (1 - a[..., None]) + FACE * a[..., None]
    for k, im in (("beauty", beauty), ("matte", a), ("line", a)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        u8 = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        if u8.ndim == 3:
            u8 = u8[..., ::-1]
        for i in range(N):
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), u8)


def _ops():
    """the s1 chain shape of the R15 piece: sandwich image, then the effect's dark ground crossfaded over the whole picture (the word is NOT drawn in it), then the tail."""
    return [
        {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 1.0}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "matte": "$m", "type_a": "$type_matte", "type_rgb": "$tcol", "matte_threshold": 0.5,
                                                    "type_opacity": 0.9}},
        {"op": "fill", "out": "fx", "fx": True, "args": {"like": "$footage", "color": list(GROUND)}},
        {"op": "crossfade", "out": "image", "fx": True, "args": {"a": "$image", "b": "$fx", "t": 1.0}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0}},
    ]


def _run(tmp, holds=((0.3, 0.9),)):
    a = _word()
    _write_type(tmp, a)
    piece = {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]},
             "shots": [{"id": "s1", "frames": N, "matte": True}],
             "compose": {"crossfade": 0.0, "default": {"ops": _ops()}, "color": "fixed"},
             "plan": {"flow_wake": {"holds": [list(h) for h in holds]}}}
    plate = np.clip(np.full((N, H, W, 3), PLATE, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)
    mid = int(round((holds[0][0] + holds[0][1]) / 2 * 24))
    u8, st = RE.run_compose(piece, "s1", type_dir=str(tmp), footage=plate, matte=np.zeros((N, H, W), np.float32), backend="numpy", threads=0, frames=(mid, mid + 1))
    return a, np.asarray(u8, np.uint8)[0].astype(np.float32) / 255.0, st


def test_no_surround_darkening_or_lightening_outside_the_type(tmp_path):
    a, out, st = _run(tmp_path / "t")
    hn = st.get("hold_net") or []
    assert hn and hn[0]["fired"], "the net must have fired in this scenario (word hidden by the dark ground), else the test measures nothing"
    near = cv2.dilate((a > 0.25).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0       # the word + ~4 px of antialias / outline room
    ground = np.array(GROUND, np.float32)
    dev = np.abs(out - ground).max(-1)
    assert float(dev[~near].max()) < 0.02, "outside the word the frame must stay the effect's ground: no soft window of plate (0.45) round the type"
    letters = a > 0.9
    assert float(np.abs(out[letters] - FACE).mean()) < float(np.abs(ground - FACE).mean()), "the net still pulls the letters toward the face colour"


def test_net_region_is_the_hard_letter_shape(tmp_path):
    a = _word()
    img = np.zeros((H, W, 3), np.float32) + np.array(GROUND, np.float32)
    plate = np.full((H, W, 3), PLATE, np.float32)
    C.NET_LOG.clear()
    out = C.hold_net(img, plate, FACE, a, holds=[[0.5, 1.5]], t=1.0, frame_idx=24, state_key="soft0")
    assert not C.net_report("soft0", [[0.5, 1.5]])[0]["max_k"] == 0.0
    changed = np.abs(out - img).max(-1) > 1e-4
    assert not (changed & (a < 1e-6)).any(), "nothing outside the letter shape changes (no dilation, no blur)"
    assert changed[a > 0.99].all()


def test_legible_clear_is_hard_edged():
    ta = _word()
    plan = {"size": [W, H], "holds": [{"a": 0.5, "b": 1.5, "n": 3, "boxes": [[8, 48, 32, 112]]}]}
    c = C.legible_clear(ta, plan, 1.0)
    assert c is not None and c.shape == (H, W)
    vals = np.unique(np.round(c, 4))
    assert set(vals.tolist()) <= {0.0, 1.0}, f"the clear mask is the dilated letter, hard-edged (no gaussian feather): {vals[:8]}"
