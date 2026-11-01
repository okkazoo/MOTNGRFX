"""One still per PLACE (app graph, 2026-10-29). Pure functions, no ComfyUI: the nodes (`MographFirstFrames`, `MographStillPrompt`, `MographShotSlot`) and the graph
builder (`tools/app/build_graphs.py`) both read the plan through these, and `tests/test_places.py` checks the wiring on a plan without a server.

The plan (tools/fusion/plan_rules.py) writes `stills` = one entry per NEW PLACE (ids A, B, C ...) and, per shot, `first` = a still id (the shot starts a new place) or
`last:sK` (the shot continues the previous shot K = its immediate predecessor from that shot's last frame). The graph renders every still the piece names, up to
MAX_STILLS, each as its own lazy chain; the shot slot picks its first frame from the batch the chains are gathered into.
"""
from __future__ import annotations

MAX_STILLS = 6                    # = MAX_SHOTS: at most one new place per shot


def still_ids(piece: dict) -> list[str]:
    return [str(s.get("id", chr(ord("A") + i))) for i, s in enumerate((piece or {}).get("stills") or [])] or ["A"]


def is_last_ref(ref) -> bool:
    return isinstance(ref, str) and ref.startswith("last:")


def still_index(piece: dict, ref) -> int | None:
    """index (0-based, into piece['stills']) of the still a shot's `first` names; None for `last:sK` / no ref. An unknown id falls back to its letter position, then to 0."""
    if ref is None or is_last_ref(ref):
        return None
    ids = still_ids(piece)
    r = str(ref)
    if r in ids:
        return ids.index(r)
    if len(r) == 1 and r.isalpha() and 0 <= ord(r.upper()) - ord("A") < min(len(ids), MAX_STILLS):
        return ord(r.upper()) - ord("A")
    return 0


def active_shots(piece: dict) -> list[dict]:
    """the shots of the piece that are in its layout (all shots when the piece has no layout yet), in order."""
    lay = (piece or {}).get("layout")
    return [s for s in (piece or {}).get("shots") or [] if lay is None or s.get("id") in lay]


def needed_stills(piece: dict) -> list[int]:
    """sorted indices of the stills some active shot starts from (always including 0 when no shot names any still)."""
    idx = {still_index(piece, s.get("first")) for s in active_shots(piece)}
    idx.discard(None)
    out = sorted(i for i in idx if i < MAX_STILLS)
    return out or [0]


def batch_size(piece: dict) -> int:
    """rows of the first-frame batch the slots index: the highest needed still + 1 (1 for every single-still piece)."""
    return max(needed_stills(piece)) + 1


def shot_sources(piece: dict) -> list[dict]:
    """what each active shot's first frame is: {"sid", "ref", "kind": "still" | "last", "still": index | None, "prev": sid | None}."""
    out = []
    for s in active_shots(piece):
        ref = s.get("first")
        if is_last_ref(ref):
            out.append({"sid": s["id"], "ref": ref, "kind": "last", "still": None, "prev": ref[5:]})
        else:
            out.append({"sid": s["id"], "ref": ref, "kind": "still", "still": still_index(piece, ref) or 0, "prev": None})
    return out


def pick_row(piece: dict, ref, n_rows: int) -> int:
    """row of the first-frame batch (n_rows rows) that a shot with `first` = ref starts from; a batch of one row is always row 0."""
    if n_rows <= 1:
        return 0
    i = still_index(piece, ref)
    return i if i is not None and 0 <= i < n_rows else 0


def batch_rows(piece: dict, first, stills: dict) -> list:
    """the rows of the first-frame batch: row 0 = `first`, row i = stills[i] (the still of place i+1, 0-based i >= 1) where a shot starts from it, else a copy of row 0
    (a place no shot uses is never read; the batch stays dense). `stills` maps the 0-based still index -> image or None. -> [first] when the piece has one place."""
    need = set(needed_stills(piece))
    rows = [first]
    for i in range(1, batch_size(piece)):
        src = stills.get(i) if i in need else None
        rows.append(first if src is None else src)
    return rows


MAX_USER_IMAGES = 4              # the user's images 2..5 (`IN images`); image 1 is the first frame (`IN image`)


def user_image_rows(piece: dict, n_user: int) -> dict[int, int]:
    """USER IMAGES AS STILLS (16_app/MEDIA_BOX.md): user image k (k = 2..5, index k-2 of `IN images`) is the first frame of shot k. -> {0-based still index (>= 1): index into the user images}
    for every shot k >= 2 that starts from a still of its own (`first` = a still id); that still is the user's picture instead of a generated one. Place A (index 0) is `first` (`IN image`
    or still A) and never replaced here; a shot that continues another (`last:sK`) has no still of its own; when two shots start from the same still the earlier shot's image wins."""
    out: dict[int, int] = {}
    for pos, src in enumerate(shot_sources(piece)):
        ui = pos - 1
        if pos == 0 or ui >= min(int(n_user), MAX_USER_IMAGES):
            continue
        if src["kind"] == "still" and src["still"] and src["still"] >= 1 and src["still"] not in out:
            out[src["still"]] = ui
    return out


def prev_is_adjacent(piece: dict) -> bool:
    """True when every `last:sK` names the shot right before it (the graph wires `prev` of slot k to the clip of shot k-1 only)."""
    ids = [s.get("id") for s in (piece or {}).get("shots") or []]
    for i, s in enumerate((piece or {}).get("shots") or []):
        if is_last_ref(s.get("first")) and (i == 0 or s["first"][5:] != ids[i - 1]):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------- same-subject new-angle stills
# A place `{"angle_of": "A", "framing": "close-up", ...}` is an EDIT of another place's still (MOSS-Image + the moss_image_edit LoRA, tools/fusion/angle_still.py,
# 16_app/ANGLE_STILLS.md), not a text-to-image still. The graph keeps both chains per place (lazy): `MographFirstFrames` asks for `angle_k` where `angle_source` says so
# (and the graph's `angles` switch is on: the 24 GB tier), else for `still_k`. plan_rules writes `angle_of` + `framing` + `angle_fallback` (the fresh-still prompt that is
# used while the edit is off).
def angle_source(piece: dict, idx: int) -> int | None:
    """0-based index of the place whose ORIGINAL still place `idx` is an edit of (`stills[idx].angle_of`); None = a normal text-to-image still.
    Never an angle place for place 0 (its image is the first frame: the user's own image or still A), never an edit of an edit (a source that is itself an angle place),
    never an unknown id, never itself: all of these fall back to a normal still (the `angle_fallback` prompt)."""
    stills = (piece or {}).get("stills") or []
    if not 1 <= int(idx) < min(len(stills), MAX_STILLS):
        return None
    ref = stills[idx].get("angle_of")
    if not ref:
        return None
    ids = still_ids(piece)
    r = str(ref)
    j = ids.index(r) if r in ids else (ord(r.upper()) - ord("A") if len(r) == 1 and r.isalpha() and ord(r.upper()) - ord("A") < len(ids) else None)
    if j is None or j == idx or j >= MAX_STILLS or stills[j].get("angle_of"):
        return None
    return j


def angle_sources(piece: dict, only_needed: bool = True) -> dict[int, int]:
    """{place index: source place index} of the angle places (of the places some shot starts from when `only_needed`)."""
    idx = needed_stills(piece) if only_needed else range(1, min(len((piece or {}).get("stills") or []), MAX_STILLS))
    return {i: j for i in idx if (j := angle_source(piece, i)) is not None}


def angle_prompt(piece: dict, idx: int) -> str:
    """the ANCHORED edit prompt of angle place `idx` ('' when it is not an angle place): `angle_still.build_edit_prompt` over the SOURCE place's still prompt (subject clause +
    light / colour words), the place's framing, face-hidden wording where the framing would show a face of a person. The place may carry its own `subject`, `palette`, `keep`,
    `hide_face`, `person` to override the heuristics (a design that has looked at the still can name the clothing)."""
    j = angle_source(piece, idx)
    if j is None:
        return ""
    from tools.fusion import angle_still as AS
    stills = piece["stills"]
    st, src = stills[idx], str(stills[j].get("prompt", ""))
    person = st["person"] if isinstance(st.get("person"), bool) else AS.is_person(src)
    return AS.build_edit_prompt(src, st.get("framing") or "close-up", subject=st.get("subject"), palette=st.get("palette"),
                                hide_face=st["hide_face"] if isinstance(st.get("hide_face"), bool) else None, keep=st.get("keep"), person=person)
