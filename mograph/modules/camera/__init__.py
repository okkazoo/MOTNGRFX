"""mograph.modules.camera: the camera module library (phase 3b). One file per move; base.py has CameraModule + CameraChain.
    from mograph.modules.camera import CameraChain, Follow, OrbitEven, ...
    chain = CameraChain.auto(seq)  |  CameraChain(seq, [Follow().at('trans:0'), HoldDrift().at('word:1'), ...])
    seq.render(path, camera=chain)"""
from .base import (CameraModule, CameraChain, Segment, Hints, carve, list_cameras, get_camera, register_camera, CAMERA_REG)  # noqa: F401
from .hold_drift import HoldDrift  # noqa: F401
from .follow import Follow  # noqa: F401
from .orbit_even import OrbitEven  # noqa: F401
from .spin_read_spin import SpinReadSpin  # noqa: F401
from .push_through import PushThrough  # noqa: F401
from .tilt_to_plan import TiltToPlan  # noqa: F401
from .graze import Graze  # noqa: F401
from .ease_reversal import EaseReversal  # noqa: F401
from .pullback_reveal import PullbackReveal  # noqa: F401
from .pan_dir import PanDir  # noqa: F401  (v006: the directional camera travel that is the transition)
