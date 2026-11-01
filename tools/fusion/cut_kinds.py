"""cut_kinds - the vocabulary of SHOT transitions (compose.cuts[].cut), pure strings, no engine imports (design_options and pivot_cut both read it).

    kind = family or family:direction
    plain      crossfade | hard                                  (the piece's soft cut / a hard cut on the beat)
    window     window_cut | window_swap | window_other | window_zoom   (type as a window onto the next shot, tools/fusion/window_cut.py; window_zoom = the camera scales into ONE letter of the holding
               word until the next shot fills the frame)
    explore    circle_grid | depth_lines                          (2026-10-30 promoted experiments, tools/fusion/explore_cuts2.py via pivot_cut.apply_cut(render=make_render); B's word is OWNED BY THE CUT:
               circle_grid has a word-less form and a word form (params.word), depth_lines always forms B's word from the depth lines, B playing inside the letters on a flat ground)
    3d         depth_pivot | orbit:push | orbit:pullback | orbit:fly | depth_dolly:down|up|left|right     (tools/fusion/pivot_cut.py)
    pan        pan:down | pan:up | pan:left | pan:right          (the type camera travels, tools/fusion/pan_plan.py; compiled to the word before the cut)
    normalize(kind)    -> the canonical kind, a bare family ('orbit' | 'depth_dolly' | 'pan': the direction is chosen later from the footage / music) or None
    family_of(kind)    -> 'plain' | 'window' | '3d' | 'pan' | 'explore'
    direction_of(kind) -> 'left' | ... | None;   sides(kind, style=None, dir=None) -> (exit side of word A, entry side of word B) or None
    style_of(kind, style=None) -> how the picture changes at a 3D cut: 'chunks' (card mosaic) | 'smooth' (depth-parallax camera move, no mosaic) | 'slide' (lateral parallax
                                  slide); None for kinds that have no style. Defaults: depth_pivot = chunks (the approved look), orbit:push / pullback / fly = smooth.
                                  The per-piece random system (random_system.draw_cuts) draws a style for each orbit kind; the draw wins over the default.
                                  Written as `style` on the cut entry or in its `params` (compose.cuts[].params.style).
"""
from __future__ import annotations

import re

PLAIN = ("crossfade", "hard")
WINDOW = ("window_cut", "window_swap", "window_other", "window_zoom")
PIVOT = "depth_pivot"
ORBIT_MODES = ("push", "pullback", "fly")
DOLLY_DIRS = ("down", "up", "left", "right")
PAN_DIRS = DOLLY_DIRS
SHOT_3D = (PIVOT,) + tuple(f"orbit:{m}" for m in ORBIT_MODES) + tuple(f"depth_dolly:{d}" for d in DOLLY_DIRS)
PANS = tuple(f"pan:{d}" for d in PAN_DIRS)
EXPLORE = ("circle_grid", "depth_lines")        # promoted from explore_cuts2 (round 2); NOT in SHOT_3D: no depth_pivot params / orbit styles apply to them
ALL = PLAIN + WINDOW + SHOT_3D + PANS + EXPLORE
_BARE = ("orbit", "depth_dolly", "pan")
_SYN = {"pivot": PIVOT, "depth_pivot": PIVOT, "pivot_chunks": PIVOT, "camera_orbit": "orbit", "orbit": "orbit", "dolly": "depth_dolly", "depth_dolly": "depth_dolly",
        "push": "orbit:push", "push_in": "orbit:push", "pullback": "orbit:pullback", "pull_back": "orbit:pullback", "fly": "orbit:fly", "fly_through": "orbit:fly",
        "pan": "pan", "window": "window_cut", "swap": "window_swap", "cut": "hard", "none": "crossfade", "dissolve": "crossfade", "fade": "crossfade",
        "text_fill": "window_other", "fill": "window_other", "window_next": "window_other",
        "circles": "circle_grid", "circle_grid": "circle_grid", "dot_grid": "circle_grid", "circle_grid_word": "circle_grid", "dot_grid_word": "circle_grid",
        "depth_lines": "depth_lines", "depth_line": "depth_lines", "depth_lines_window": "depth_lines", "depth_lines_word": "depth_lines", "lines": "depth_lines", "iso_lines": "depth_lines",
        "zoom": "window_zoom", "letter_zoom": "window_zoom", "zoom_through": "window_zoom", "zoom_letter": "window_zoom", "window_zoom": "window_zoom"}
_DIR_SYN = {"l": "left", "r": "right", "u": "up", "d": "down", "in": "push", "out": "pullback", "forward": "push", "back": "pullback", "backward": "pullback", "pull_back": "pullback", "push_in": "push", "fly_through": "fly", "flythrough": "fly"}


def normalize(kind) -> str | None:
    """any spelling -> canonical kind ('window_cut', 'orbit:push', 'depth_dolly:down', 'pan:left', 'depth_pivot', ...), a bare family, or None."""
    k = re.sub(r"\s*:\s*", ":", str(kind or "").strip().lower()).replace("-", "_").replace(" ", "_")
    if k in ALL:
        return k
    fam, _, arg = k.partition(":")
    arg = _DIR_SYN.get(arg, arg)
    f = _SYN.get(fam)
    if f is None:
        return None
    if f in ALL:
        return f if not arg else None
    if f == "orbit":
        return f"orbit:{arg}" if arg in ORBIT_MODES else ("orbit" if not arg else None)
    if f == "depth_dolly":
        return f"depth_dolly:{arg}" if arg in DOLLY_DIRS else ("depth_dolly" if not arg else None)
    if f == "pan":
        return f"pan:{arg}" if arg in PAN_DIRS else ("pan" if not arg else None)
    return None


def family_of(kind: str) -> str:
    k = str(kind)
    return "plain" if k in PLAIN else "window" if k in WINDOW else "pan" if k.startswith("pan") else "explore" if k in EXPLORE else "3d"


def direction_of(kind: str):
    a = str(kind).partition(":")[2]
    return a or None


# the sides a word slides to / from under a directional cut: the camera travels `dir`, the picture slides the other way, so word A leaves on the opposite
# side and word B arrives from the travel side (g04: the picture turned LEFT, word A left, word B came in from the RIGHT, strands flowed left -> right).
_OPP = {"left": "right", "right": "left", "up": "down", "down": "up"}
STYLES = ("chunks", "smooth", "slide")


def style_of(kind, style=None):
    """the concrete style of a 3D cut kind: 'auto' / None / unknown -> the default (chunks for the pivot AND the orbit modes; smooth / slide only when named); the pivot has no smooth or slide form;
    depth_dolly (shards) and every non-3D kind -> None."""
    k = str(kind)
    if k == PIVOT:
        return "chunks"
    if k.startswith("orbit:"):
        st = str(style or "").strip().lower()
        return st if st in STYLES else "chunks"          # 2026-10-30 (user: "we shouldnt see simple flat planes rotating like this in frame - they should be broken up into smaller chunks"): `smooth` (one card) only when asked for
    return None


def sides(kind: str, style=None, dir=None):
    """(exit side of word A, entry side of word B) for a cut that has a direction; None for kinds without one (push / pullback / fly / window / plain).
    An orbit kind in the `slide` style has one: the camera travels `dir` (default right), A leaves the other way."""
    k = str(kind)
    if k.startswith("orbit:") and style_of(k, style) == "slide":
        d = dir if dir in _OPP else "right"
        return _OPP[d], d
    if k == PIVOT:
        return "left", "right"
    d = direction_of(k)
    if d in _OPP and (k.startswith("depth_dolly") or k.startswith("pan")):
        return _OPP[d], d
    return None


def owns_word(kind, params=None) -> bool:
    """does the cut draw B's word itself ('owned_by_cut': the type stage must not draw it during the cut's window)? depth_lines always; circle_grid only in its word form
    (params.word truthy: True | 'auto' | the word text)."""
    k = normalize(kind)
    if k == "depth_lines":
        return True
    return k == "circle_grid" and bool((params or {}).get("word"))
