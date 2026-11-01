"""ingredient_names: ONE naming scheme for everything a composition is made of, so a search for the run stem finds it all.

Pure python (no torch / cv2): used on the box by nodes_fusion/archive.py (MographArchive) and on this PC by tools/app/ingredients.py.

    stem = run_stem("projects/COMPETITIONS/motion-graphic/batch-24gb/v002/motion-graphic_batch-24gb_v002_r01_out")
         = "motion-graphic_batch-24gb_v002_r01"

Every file / folder of a run carries `<stem>_` in front (the tails are listed in `TAILS`), e.g.
    <stem>_video.mp4  <stem>_music.wav  <stem>_still.png  <stem>_still_2.png  <stem>_s1_clip.mp4  <stem>_s1_clean.mp4  <stem>_s1_textbg.mp4
    <stem>_s1_matte/<stem>_s1_matte_00001.png (PNG sequence = one stack member)  <stem>_s1_depth.mp4  <stem>_s1_flow.mp4
    <stem>_s1_contour_luma_edges.mp4 / .npz  <stem>_type_beauty.mp4 (matte, depth, line, clay, flow)  <stem>_ltx_read_s2_1.png
    <stem>_piece.json  <stem>_index.json  <stem>_concept.json  <stem>_design_reply.txt ...
The pre-2026-10-29 archives used short names (still.png, s1_clip.mp4, s1_matte/00001.png, piece.json, index.json): `from_legacy` maps them.
`LEGACY_NAMES` = MographArchive also keeps the short names (hardlinks on the box: no extra box disk; the dashboard sync copies both) because
nodes_fusion/redesign.py (MographArchiveLoad) and tools/fusion/generations.py read them. Set False once those readers take the prefixed names.
"""
from __future__ import annotations

import os
import re

LEGACY_NAMES = True

# the tails of the run-level files (after `<stem>_`); per-shot / per-pass tails are built by the helpers below
TAILS = ("video.mp4", "music.wav", "still.png", "piece.json", "index.json", "concept.json", "concept_reply.txt", "concept_check.txt",
         "design_reply.txt", "design_check.txt", "design_final.txt", "ltx_reads.json", "MANIFEST.md")
TYPE_PASSES = ("beauty", "matte", "depth", "line", "clay", "flow")
CONTOUR_KINDS = ("matte_outline", "depth_iso", "luma_edges")

_OUT_SUFFIX = re.compile(r"_out(_shots)?$")


def run_stem(prefix: str) -> str:
    """archive filename_prefix (`.../<stem>_out`, with or without `_shots`) -> `<stem>`; `archive/motngrfx` -> `motngrfx`."""
    base = str(prefix).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return _OUT_SUFFIX.sub("", base) or "motngrfx"


def nm(stem: str, tail: str) -> str:
    return f"{stem}_{tail}"


def still(stem: str, k: int | None = None) -> str:
    """`<stem>_still.png` (the piece's first frame) or `<stem>_still_<k>.png` (place k, 1-based)."""
    return nm(stem, "still.png" if k is None else f"still_{int(k)}.png")


def shot_file(stem: str, sid: str, kind: str, ext: str = "mp4") -> str:
    """`<stem>_s1_clip.mp4` (kind clip | clean | textbg | depth | flow)."""
    return nm(stem, f"{sid}_{kind}.{ext}")


def matte_dir(stem: str, sid: str) -> str:
    return nm(stem, f"{sid}_matte")


def matte_frame(stem: str, sid: str, i: int) -> str:
    """frame i (1-based) of a matte folder: `<stem>_s1_matte_00001.png` (dashboard sequence rule: prefix + number + .png)."""
    return nm(stem, f"{sid}_matte_{int(i):05d}.png")


def contour_file(stem: str, sid: str, kind: str, ext: str = "mp4") -> str:
    return nm(stem, f"{sid}_contour_{kind}.{ext}")


def type_pass_file(stem: str, name: str) -> str:
    return nm(stem, f"type_{name}.mp4")


def ltx_read_crop(stem: str, sid: str, j: int) -> str:
    return nm(stem, f"ltx_read_{sid}_{int(j)}.png")


def kind_slug(kind: str) -> str:
    """`orbit:push` -> `orbit_push`, `depth_dolly:down` -> `depth_dolly_down` (a transition kind as a file-name part)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(kind)).strip("_") or "cut"


def cut_file(stem: str, a: str, b: str, kind: str, ext: str = "mp4") -> str:
    """the rendered 3D shot cut between shots a and b: `<stem>_cut_s1-s2_<kind>.mp4` (nodes_fusion/depth_transition.py MographPivotCut)."""
    return nm(stem, f"cut_{a}-{b}_{kind_slug(kind)}.{ext}")


def cut_matte_dir(stem: str, a: str, b: str) -> str:
    return nm(stem, f"cut_{a}-{b}_matte")


def cut_matte_frame(stem: str, a: str, b: str, i: int) -> str:
    return nm(stem, f"cut_{a}-{b}_matte_{int(i):05d}.png")


def depth_dir(stem: str, sid: str) -> str:
    """per-frame depth of a shot as a PNG sequence folder (`<stem>_s1_depth/<stem>_s1_depth_00001.png`)."""
    return nm(stem, f"{sid}_depth")


def depth_frame(stem: str, sid: str, i: int) -> str:
    return nm(stem, f"{sid}_depth_{int(i):05d}.png")


def type_matte_dir(stem: str) -> str:
    """the type matte pass as a PNG sequence (mattes are never lossy): `<stem>_type_matte/<stem>_type_matte_00001.png`."""
    return nm(stem, "type_matte")


def type_matte_frame(stem: str, i: int) -> str:
    return nm(stem, f"type_matte_{int(i):05d}.png")


# What `MographArchive(full_archive=False)` leaves out (the heavy extras); matched on the top-level name inside `<stem>/`.
# Everything else is the light set that is always written. Used by the MANIFEST size estimate (tools/app/ingredients.py).
_HEAVY = re.compile(r"_(cut_s\d+-s\d+_.+\.mp4|cut_s\d+-s\d+_matte|s\d+_comp\.mp4|s\d+_depth|type_matte)$")


def is_heavy(name: str) -> bool:
    """True for the files / folders only written with full_archive on (cut renders + cut mattes, composed shots, depth PNG sequences, type matte PNGs)."""
    return bool(_HEAVY.search(os.path.basename(name.replace("\\", "/").rstrip("/"))))


def has_stem(stem: str, name: str) -> bool:
    return os.path.basename(name).startswith(stem + "_")


_LEG_FILE = re.compile(r"^(still\.png|piece\.json|index\.json|s\d+_(?:clip|clean|textbg)\.mp4)$")
_LEG_MATTE = re.compile(r"^(s\d+)_matte/(\d+)\.png$")


def from_legacy(stem: str, rel: str) -> str | None:
    """short archive name -> prefixed relative path (None: not a legacy name; already prefixed names pass through unchanged)."""
    rel = rel.replace("\\", "/")
    if _LEG_FILE.match(rel):
        return nm(stem, rel)
    m = _LEG_MATTE.match(rel)
    if m:
        return f"{matte_dir(stem, m.group(1))}/{matte_frame(stem, m.group(1), int(m.group(2)))}"
    return None


def to_new(stem: str, rel: str) -> str | None:
    """relative path inside a `_shots` archive dir -> its name in the ingredients folder (prefixed names as they are, legacy mapped, others None)."""
    rel = rel.replace("\\", "/")
    top = rel.split("/", 1)[0]
    if top.startswith(stem + "_"):
        return rel
    return from_legacy(stem, rel)
