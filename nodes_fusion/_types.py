"""Custom ComfyUI io types of the fusion node pack (DESIGN.md section 1) plus TypedDicts for the dict shapes.

The values travelling between nodes are plain dicts / lists (no classes), so ComfyUI's output cache and the nodes'
unit tests need nothing from this module except the type names. The TypedDicts below are documentation (and help
editors); nothing validates against them at run time.

Usage in a node::

    from ._types import MographRequest, MographPiece
    inputs=[MographRequest.Input("request"), ...]
    outputs=[MographPiece.Output(display_name="piece")]

Heavy data crosses node boundaries as uint8 CPU tensors (``MOGRAPH_CLIP.u8``), never float32 (a float32 1080x1920
shot is 3.4 GB and ComfyUI caches node outputs).
"""
from __future__ import annotations

from typing import Any, TypedDict

from comfy_api.latest import io

# --------------------------------------------------------------------------------------------- io types
MographRequest = io.Custom("MOGRAPH_REQUEST")
MographMusic = io.Custom("MOGRAPH_MUSIC")
MographPiece = io.Custom("MOGRAPH_PIECE")
MographType = io.Custom("MOGRAPH_TYPE")
MographClip = io.Custom("MOGRAPH_CLIP")
MographClips = io.Custom("MOGRAPH_CLIPS")

TYPE_NAMES = ("MOGRAPH_REQUEST", "MOGRAPH_MUSIC", "MOGRAPH_PIECE", "MOGRAPH_TYPE", "MOGRAPH_CLIP", "MOGRAPH_CLIPS")

CATEGORY = "MOTNGRFX/Fusion"

# shared constants (shot slots, sizes) so every node module agrees
MAX_SHOTS = 6
FPS = 24
LENGTHS = (8, 12, 16, 24)
LOOK_AUTO = "auto"
QUALITIES = ("preview", "draft", "final")


# --------------------------------------------------------------------------------------------- dict shapes
class Request(TypedDict, total=False):
    """MOGRAPH_REQUEST: the app inputs after validation (MographRequest)."""
    words: str            # multiline story text, one idea per line
    scene: str            # scene prompt; "" = written from the words
    music: str            # music prompt; "" = default tags
    length: int           # seconds: 8 | 12 | 16 | 24
    look: str             # "auto" or a module name / family (GET /mograph/api/looks)
    seed: int
    quality: str          # "draft" | "final"
    image: Any            # IMAGE tensor [1,H,W,3] or None (user first frame)
    audio: Any            # AUDIO dict or None (user music)


class Music(TypedDict, total=False):
    """MOGRAPH_MUSIC: same keys as piece_runner ``analysis.json``."""
    cut: float            # seconds kept (on a downbeat)
    nframes: int          # round(cut * 24)
    bpm: float
    bar_len: float        # seconds per bar
    key: str
    beats: list           # seconds
    downbeats: list       # seconds
    chord_changes: list   # seconds
    curves: dict          # name -> per-frame list (energy, onset ...)
    music_end: float


class ShotLayout(TypedDict):
    """``MOGRAPH_PIECE['layout'][sid]``."""
    start: int            # first frame in the piece timeline
    frames: int           # LTX frames (8k+1)
    xf: int               # crossfade frames into the next shot


class Piece(TypedDict, total=False):
    """MOGRAPH_PIECE: the P*.json schema (15_fusion/pieces/P02.json) plus planner additions."""
    id: str
    slug: str
    layout: dict          # sid -> ShotLayout, sid in "s1".."s6"
    sizes: dict           # {"control": [w,h], "type": [w,h], "compose": [w,h]}
    technique: str        # built_in | behind | traced
    quality: str          # draft | final
    brief: dict           # parsed Qwen brief (or {})
    shots: list           # P*.json shot entries (sid, prompt, motion line, seed, first)
    recipe: dict          # compose recipe


class TypePasses(TypedDict, total=False):
    """MOGRAPH_TYPE: result of MographTypePasses; ``job`` is a subprocess.Popen (async layers) or None."""
    dir_control: str
    dir_layers: str
    manifest: dict
    job: Any


class Clip(TypedDict, total=False):
    """MOGRAPH_CLIP: one shot (or one composed shot)."""
    u8: Any               # torch.uint8 [T,H,W,C] on CPU
    sid: str              # "s1".."s6"
    start: int            # first frame in the piece timeline
    fps: int


def new_clip(u8, sid: str, start: int = 0, fps: int = FPS) -> Clip:
    return {"u8": u8, "sid": sid, "start": int(start), "fps": int(fps)}


def sid_of(index: int) -> str:
    """Slot index 1..6 -> shot id 's1'..'s6'."""
    return f"s{int(index)}"


def active_sids(piece: dict) -> list[str]:
    """Shot ids present in ``piece['layout']`` ordered s1, s2, ... (the active slots)."""
    lay = (piece or {}).get("layout") or {}
    return sorted(lay, key=lambda s: int(str(s).lstrip("s") or 0))
