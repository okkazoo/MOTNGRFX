"""batch_plan.py - the batch-aware choice of EVERYTHING the planner can pick from (user 2026-10-26: "only 2 of the 24 word modules are used
in any piece, we should be using it all"; "all generations choosing from all modules that we have"; "use the layouts that we worked on").

    assign_batch(pieces, seed=0)      -> {"pieces": [{id, format, subject, lines: [{module, io_in, io_out, transition, camera, font, layout, ..}]}],
                                          "coverage": coverage_report(..)}
    coverage_report(assignment)       -> what the batch used / missed, in-piece repeats, contrast violations, font / layout checks
    plan_batch(requests, musics, seed)-> (pieces, assignment): plan_rules.plan_piece for every request with its assignment applied

A batch of N pieces together uses every registered word module, transition (cut included), io style, camera move and hold layout, with
the rules of the notes inside each piece: no repeated module or transition (cuts are not a visible transition), consecutive lines contrast
(module family, font role / weight / fill, or scale), fonts only from mograph/fonts/liked.json, layouts valid for the line (hold_layouts).
Deterministic in (pieces, seed): every choice is "least used so far, ties by a crc32 rank of (seed, tag, slot, item)", rarest items placed first.

Pairings (all checked in tests/test_plan_coverage.py):
  * a boundary whose io styles are flicker / wipe carries no strands -> the transition is the cut (spec.NO_MATERIAL); every other boundary gets a
    strand transition; at most 1-2 cut boundaries per piece, flicker / wipe otherwise only open the first / close the last line.
  * a layout with backing shapes is screen-space, so that line's camera is hold_drift; graze / push_through / tilt_to_plan leave the framed
    box (traverse / dive / top-down) so they go on full-frame holds, one per piece at most; counters_window needs fly_into / depth_tunnel.
Contour keys (entry / exit / via / matte, the contour-port spec of tools/fusion/contour_ports.py) are chosen for every line of a piece that
has a subject, varied line to line; plan_rules writes them into the spec words once contour_ports exists.
"""
from __future__ import annotations

import importlib.util
from collections import Counter

from tools.fusion import blocks as bk
from tools.fusion import hold_layouts as hl
from tools.fusion import plan_rules as pr

BEAT_ONLY = ("beat_orbit", "pan")                                               # tools/fusion/cameras.py: one beat-locked orbit over the whole timeline, not a per-word move
FREE_CAMERAS = ("graze", "push_through", "tilt_to_plan")                   # leave the framed box: full-frame holds only
LAYOUT_CAMERAS = ("hold_drift", "follow", "pullback_reveal", "orbit_even", "spin_read_spin", "ease_reversal")   # keep the box in frame
ANCHOR_CAMERA = "hold_drift"
LAYOUT_WEIGHT = {"full": 2.0}                                              # the full-frame hold is also what the free cameras need
MATTE_MODES = ("front", "window", "fill", "wipe", "glow")                  # spec "matte" key of the contour-port spec
PORT_FALLBACK = ("matte_outline", "depth_iso", "luma_edges")                # mograph/modules/contour_ports.py KINDS
PORT_DIRS = ("up", "along", "around")                                       # contour_ports.DIRS
ANCHORED = {n for n in hl.NAMES if hl.build(n, 0, "a", hl.LAYOUTS[n]["fmts"][0])["shapes"]}     # layouts with backing shapes (screen space)
REPEATABLE = ("small_line",)                                              # layouts a piece may use on several lines (never on neighbours)
PAN_DIRS = ("down", "up", "right", "left")                                   # mograph.modules.camera.pan_dir.DIRS: the directions of the camera_pan_transition block
MAX_CUTS = (0, 0, 0, 1, 1, 1, 2)                                           # boundaries -> cut boundaries per piece (index = boundaries, last = more)


# ================================================================================================== registry views
def word_modules() -> list[str]:
    pr._spec()                                                              # imports mograph.modules (registers words / transitions / io / cameras)
    from mograph.modules.core import list_modules
    from tools.fusion import design_defaults as DD
    return [m for m in list_modules("word") if m not in DD.BANNED_MODULES]            # 2026-11-01: `particles` (a dot-cloud letter) is retired, 2026-11-02 `outflow` too, so no batch line is assigned them


def io_styles() -> list[str]:
    pr._spec()
    from mograph.modules import io
    from tools.fusion import design_defaults as DD
    return [s for s in io.list_styles() if s not in DD.SPECKLE_IO and s != "outline"]   # 2026-11-01: `particles` / `blobs` are banned; `outline` is the library transitions' glyph hand-off (sequence.py), not a style a word names


def cameras() -> list[str]:
    pr._spec()
    from mograph.modules.camera import list_cameras
    return [c for c in list_cameras() if c not in BEAT_ONLY]


def family_of(module: str) -> str:
    for fam, ms in pr.LOOKS.items():
        if module in ms:
            return fam
    return "other:" + module                                                # a new module is its own family until LOOKS lists it


def no_material() -> tuple:
    return tuple(pr._spec().NO_MATERIAL)


def _cp():
    """the contour-port module (mograph/modules/contour_ports.py) or None."""
    try:
        from mograph.modules import contour_ports
        return contour_ports
    except Exception:                                                       # not built yet / missing scipy: the keys are still planned
        return None


def contour_ports_available() -> bool:
    return _cp() is not None


def contour_port_names() -> list[str]:
    cp = _cp()
    return list(getattr(cp, "KINDS", PORT_FALLBACK)) if cp else list(PORT_FALLBACK)


def contour_dirs() -> list[str]:
    cp = _cp()
    return list(getattr(cp, "DIRS", PORT_DIRS)) if cp else list(PORT_DIRS)


def matte_modes() -> list[str]:
    cp = _cp()
    return list(getattr(cp, "MATTE_MODES", MATTE_MODES)) if cp else list(MATTE_MODES)


def piece_ports(piece: dict) -> tuple[list[str], list[str]]:
    """(ports, matte modes) a planned piece supports: luma_edges always, matte_outline with a matte shot, depth_iso with a depth shot
    (contour_ports.available); every port / mode when the module is absent."""
    cp = _cp()
    if cp is None or not piece.get("shots"):
        return contour_port_names(), matte_modes()
    av = cp.available(piece)
    return list(av["ports"] or ["luma_edges"]), list(av["matte"])


# ================================================================================================== input
def _norm_pieces(pieces, fmt_default="9:16") -> list[dict]:
    out = []
    for k, p in enumerate(pieces):
        d: dict = {"id": f"B{k + 1}", "texts": None, "n": 0, "format": fmt_default, "subject": True, "blocks": [], "pan": None}
        if isinstance(p, int):
            d["n"] = int(p)
        elif isinstance(p, (list, tuple)):
            d["texts"], d["n"] = [str(t) for t in p], len(p)
        elif isinstance(p, dict) and "type" in p and "words" in p["type"]:          # a planned piece
            d["id"] = str(p.get("id", d["id"]))
            d["texts"] = [w["text"] for w in p["type"]["words"]]
            d["n"] = len(d["texts"])
            d["format"] = pr.format_of(p)
            d["subject"] = pr.piece_has_subject(p)
            d["ports"], d["mattes"] = piece_ports(p)
        elif isinstance(p, dict):
            d["id"] = str(p.get("id", d["id"]))
            if p.get("lines"):
                d["texts"] = [str(t) for t in p["lines"]]
            elif p.get("words"):
                d["texts"] = list(pr.fb.lines_of(str(p["words"])))
            d["n"] = len(d["texts"]) if d["texts"] else int(p.get("n_lines") or 0)
            d["format"] = p.get("format", fmt_default)
            d["subject"] = bool(p.get("subject", True))
            d["blocks"] = [b for b in (p.get("blocks") or []) if b in bk.BLOCKS]
            d["ports"], d["mattes"] = p.get("ports"), p.get("mattes")
            d["pan"] = p.get("pan")
        else:
            raise TypeError(f"piece {k}: expected int | list[str] | dict, got {type(p).__name__}")
        if d["n"] < 1:
            raise ValueError(f"piece {k} has no lines")
        if d["format"] not in hl.FORMATS:
            raise ValueError(f"piece {k}: format {d['format']!r}")
        out.append(d)
    return out


# ================================================================================================== the generic coverage-first assigner
def _rank(seed, tag, *parts) -> int:
    return pr.sub_seed(seed, tag + ":" + ":".join(str(x) for x in parts))


def _assign(slots, universe, ok, seed, tag, fixed=None, weights=None, soft=None, first=()):
    """slot -> item. ``ok(slot, item, A)`` = may this item sit in this slot given the others assigned in A (A[slot] itself is ignored).
    1. fixed slots are kept.  2. Coverage: items in order of how few slots take them (rarest first) are placed at an eligible free slot,
    the piece with the fewest assigned slots first.  3. The other slots take the least used eligible item (count / weight), ties by crc32
    rank; when nothing is eligible ``soft`` (a weaker ok) is used, and the violation shows in the coverage report.
    4. Repair: an item still missing replaces a duplicate wherever ok allows."""
    weights = weights or {}
    A = dict(fixed or {})
    cnt = Counter(A.values())
    fill = Counter(s[0] for s in A)
    free = [s for s in slots if s not in A]
    for s in first:                                  # slots a block constrains are filled first, least used allowed item, so the coverage prepass cannot use up what they need
        cands = [x for x in universe if ok(s, x, A)]
        if s not in A and cands:
            x = min(cands, key=lambda x: (cnt[x] / weights.get(x, 1.0), _rank(seed, tag + "/c", x, *s)))
            A[s] = x
            cnt[x] += 1
            fill[s[0]] += 1

    def elig(x):
        return [s for s in free if s not in A and ok(s, x, A)]

    for x in sorted(universe, key=lambda x: (len(elig(x)), _rank(seed, tag + "/u", x))):
        if cnt[x]:
            continue
        cands = elig(x)
        if cands:
            s = min(cands, key=lambda s: (fill[s[0]], _rank(seed, tag + "/s", x, *s)))
            A[s] = x
            cnt[x] += 1
            fill[s[0]] += 1
    for s in sorted(free, key=lambda s: (s[0], s[1:])):
        if s in A:
            continue
        cands = [x for x in universe if ok(s, x, A)] or [x for x in universe if (soft or (lambda *_: True))(s, x, A)] or list(universe)
        x = min(cands, key=lambda x: (cnt[x] / weights.get(x, 1.0), _rank(seed, tag + "/f", x, *s)))
        A[s] = x
        cnt[x] += 1
    for x in [u for u in universe if not cnt[u]]:
        for s in sorted(free, key=lambda s: _rank(seed, tag + "/r", x, *s)):
            old = A[s]
            if cnt[old] > 1:
                del A[s]
                if ok(s, x, A):
                    A[s] = x
                    cnt[old] -= 1
                    cnt[x] += 1
                    break
                A[s] = old
    return A


# ================================================================================================== pan boundaries (block camera_pan_transition)
def _cut_boundaries(P, seed) -> dict:
    """{piece: [boundary indices that are a cut]}: flicker / wipe sides carry no strands (at most MAX_CUTS per piece, never two neighbours)."""
    cut_at: dict[int, list[int]] = {}
    for p, d in enumerate(P):
        nb = d["n"] - 1
        k = MAX_CUTS[min(nb, len(MAX_CUTS) - 1)] if nb >= 0 else 0
        bs = sorted(range(nb), key=lambda b: _rank(seed, "cutb", p, b))[:k]
        if k == 2 and abs(bs[0] - bs[1]) < 2:
            bs = bs[:1]
        cut_at[p] = sorted(bs)
    return cut_at


def _pan_request(v):
    """the piece key `pan`: True | int (how many) | 'down' | ['down', 'right'] (the directions, in order) -> (count | None, [dirs])"""
    if v is None or isinstance(v, bool):
        return None, []
    if isinstance(v, int):
        return int(v), []
    ds = [v] if isinstance(v, str) else list(v)
    ds = [x for x in ds if x in PAN_DIRS]
    return (len(ds) or None), ds


def _pan_boundaries(P, cut_at, seed) -> dict:
    """{(piece, boundary): {dir, beats}} of the pieces that carry the camera_pan_transition block: 1-3 non-cut boundaries (spread out, the
    block's `count` param unless the request names a number / the directions), the four directions rotated least-used first across the batch
    (a batch with >= 4 pans covers down / up / right / left), neighbouring pans of one piece differ."""
    out: dict = {}
    cnt: Counter = Counter()
    for p, d in enumerate(P):
        if bk.PAN_BLOCK not in d["blocks"]:
            continue
        prm = bk.resolve_params(bk.PAN_BLOCK, seed, f"{p}.pan")
        n_req, want = _pan_request(d.get("pan"))
        cands = [b for b in range(d["n"] - 1) if b not in cut_at[p]]
        k = min(len(cands), n_req or prm["count"])
        order = sorted(cands, key=lambda b: _rank(seed, "panb", p, b))
        chosen: list[int] = []
        for b in order:                                                     # spread out first (no two neighbouring boundaries), then fill up
            if len(chosen) < k and all(abs(b - c) > 1 for c in chosen):
                chosen.append(b)
        for b in order:
            if len(chosen) < k and b not in chosen:
                chosen.append(b)
        prev = None
        for j, b in enumerate(sorted(chosen)):
            if j < len(want):
                dr = want[j]
            else:
                pool = [x for x in PAN_DIRS if x != prev] or list(PAN_DIRS)
                dr = min(pool, key=lambda x: (cnt[x], _rank(seed, "pand", p, b, x)))
            cnt[dr] += 1
            prev = dr
            out[(p, b)] = dict(dir=dr, beats=int(prm["beats"]))
    return out


# ================================================================================================== the assignment
def assign_batch(pieces, seed: int = 0, fmt: str = "9:16", subject: bool = True) -> dict:
    """pieces: list of int (line count) | list[str] (the lines) | dict {id, lines | n_lines | words, format, subject} | a planned piece dict.
    -> {"seed", "pieces": [{"id", "format", "subject", "n_lines", "lines": [...], "transitions": [...]}], "coverage": {...}}.
    Every line: index, text, module, family, io_in, io_out, transition (into the next line, None on the last), camera, camera_params,
    font (family), font_file, role, layout (name), layout_record, entry / exit / via / matte (contour keys, only with a subject)."""
    P = _norm_pieces(pieces, fmt)
    for d in P:
        d["subject"] = bool(d["subject"] and subject)
    S = [(p, i) for p, d in enumerate(P) for i in range(d["n"])]
    N = {p: d["n"] for p, d in enumerate(P)}
    text = {(p, i): (d["texts"][i] if d["texts"] else "") for p, d in enumerate(P) for i in range(d["n"])}
    fmts = {(p, i): d["format"] for p, d in enumerate(P) for i in range(d["n"])}

    def mates(A, s):
        return [A.get((s[0], j)) for j in range(N[s[0]]) if j != s[1]]

    def around(A, s):
        return A.get((s[0], s[1] - 1)), A.get((s[0], s[1] + 1))

    # ---- blocks (lego passages, tools/fusion/blocks.py): the lines they constrain may only take what they allow
    cons: dict = {}
    for p, d in enumerate(P):
        for i, c in enumerate(bk.line_constraints(d["blocks"], d["n"])):
            cons[(p, i)] = c

    def allow(s, key, x):
        v = cons[s].get(key)
        return v is None or x in v

    # ---- cut boundaries (flicker / wipe sides carry nothing) and the pan boundaries of the camera_pan_transition block (the camera travel is the transition)
    cut_at = _cut_boundaries(P, seed)
    pan_at = _pan_boundaries(P, cut_at, seed)
    ok_l = [n for n in hl.NAMES if n not in ANCHORED and n not in hl.RESERVED]
    for (p, b) in pan_at:                                                   # a pan carries the words world-locked, in screen-free layouts (a backing shape / screen offset would not travel)
        for i in (b, b + 1):
            cons[(p, i)]["layout"] = ok_l if cons[(p, i)]["layout"] is None else ([n for n in cons[(p, i)]["layout"] if n in ok_l] or ok_l)

    # ---- word modules: all of them, none twice in a piece, neighbouring lines from different families
    mods = word_modules()

    def ok_mod(s, x, A):
        pv, nx = around(A, s)
        return (allow(s, "module", x) and x not in mates(A, s)
                and family_of(x) not in {family_of(pv) if pv else None, family_of(nx) if nx else None})
    M = _assign(S, mods, ok_mod, seed, "module", soft=lambda s, x, A: x not in mates(A, s), first=[s for s in S if cons[s]["module"] is not None])

    # ---- hold layouts: each valid for its line; unique in a piece; shapes never on two neighbouring lines
    names = list(hl.NAMES)

    def reserved(s, x):                                                     # small_line (cap >= 4 %) is the edge_run block's: no other line may take it
        return x in hl.RESERVED and cons[s].get("layout") is None

    def ok_lay(s, x, A):
        if (x in mates(A, s) and x not in REPEATABLE) or not allow(s, "layout", x) or reserved(s, x) or not hl.eligible(x, text[s], fmts[s], M.get(s)):
            return False
        pv, nx = around(A, s)
        if x in (pv, nx):
            return False
        return not (x in ANCHORED and (pv in ANCHORED or nx in ANCHORED))
    L = _assign(S, names, ok_lay, seed, "layout", weights=LAYOUT_WEIGHT, first=[s for s in S if cons[s]["layout"] is not None],
                soft=lambda s, x, A: not reserved(s, x) and hl.eligible(x, text[s], fmts[s], M.get(s)) and (x not in mates(A, s) or x in REPEATABLE))
    recs = {s: hl.build(L[s], seed, f"{s[0]}.{s[1]}", fmts[s]) for s in S}

    # ---- cameras: all of them
    cams = cameras()
    fixed_cam = {s: ANCHOR_CAMERA for s in S if recs[s]["anchored"]}

    def ok_cam(s, x, A):
        pv, nx = around(A, s)
        if x in (pv, nx) and s not in fixed_cam:
            return False
        if not allow(s, "camera", x) and not recs[s]["anchored"]:
            return False
        if recs[s]["anchored"]:
            return x == ANCHOR_CAMERA
        if x in FREE_CAMERAS:
            return L[s] == "full" and not any(m in FREE_CAMERAS for m in mates(A, s))
        return L[s] == "full" or x in LAYOUT_CAMERAS
    C = _assign(S, cams, ok_cam, seed, "camera", fixed=fixed_cam, first=[s for s in S if cons[s]["camera"] is not None],
                soft=lambda s, x, A: x not in (around(A, s)) and (x not in FREE_CAMERAS or L[s] == "full" or not recs[s]["box"]))

    # ---- fonts: liked display faces only; neighbours contrast in weight / fill (font role) or in scale (layout box area)
    from mograph import fonts_liked as fl
    fonts = sorted(pr._liked_display())

    def role(f):
        return fl.role_of(f)

    def ok_font(s, f, A):
        if f in mates(A, s):
            return False
        if text[s] and not hl.cap_ok(recs[s], text[s], pr.fr.REGISTRY[f]["file"], fmts[s]):      # this face must still reach the layout's letter height
            return False
        for o_slot in ((s[0], s[1] - 1), (s[0], s[1] + 1)):
            o = A.get(o_slot)
            if o is None:
                continue
            a, b = recs[s]["area"], recs[o_slot]["area"]
            if not (fl.contrast_axes(role(o), role(f)) or max(a, b) / max(min(a, b), 1e-6) >= 1.25):
                return False
        return True
    F = _assign(S, fonts, ok_font, seed, "font", soft=lambda s, f, A: f not in around(A, s))

    # ---- io styles and transitions
    styles = io_styles()
    nm = no_material()
    strand_styles = [x for x in styles if x not in nm]
    trans = [t for t in pr.transition_names()]
    fixed_io: dict = {}
    nm_cnt: Counter = Counter()
    for p, d in enumerate(P):
        for b in cut_at[p]:
            st = min(nm, key=lambda x: (nm_cnt[x], _rank(seed, "nm", p, b, x))) if nm else None
            if st is None:
                continue
            nm_cnt[st] += 1
            side = ("out", "in", "both")[_rank(seed, "side", p, b) % 3]
            if side in ("out", "both"):
                fixed_io[(p, b, "out")] = st
            if side in ("in", "both"):
                fixed_io[(p, b + 1, "in")] = st
    io_slots = [(p, i, sd) for p, d in enumerate(P) for i in range(d["n"]) for sd in ("in", "out")]

    def boundary(s):
        p, i, sd = s
        return (sd == "out" and i < N[p] - 1) or (sd == "in" and i > 0)

    def ok_io(s, x, A):
        if boundary(s) and x in nm:
            return False                                                    # flicker / wipe on a boundary only where the cut was planned
        c = cons[(s[0], s[1])]
        if c.get("io") is not None and c.get("io_side") == s[2] and x not in c["io"]:
            return False
        other = A.get((s[0], s[1], "out" if s[2] == "in" else "in"))
        return x != other
    IO = _assign(io_slots, styles, ok_io, seed, "io", fixed=fixed_io, soft=lambda s, x, A: not (boundary(s) and x in nm))

    def is_cut(p, b):
        return IO[(p, b, "out")] in nm or IO[(p, b + 1, "in")] in nm
    t_slots = [(p, b) for p, d in enumerate(P) for b in range(d["n"] - 1) if not is_cut(p, b) and (p, b) not in pan_at]
    tcons = {p: bk.transition_constraints(d["blocks"], max(d["n"] - 1, 0)) for p, d in enumerate(P)}

    def ok_tr(s, x, A):
        c = tcons[s[0]][s[1]]
        return (c is None or x in c) and x not in [A.get((s[0], b)) for b in range(N[s[0]] - 1) if b != s[1]]
    T = _assign(t_slots, trans, ok_tr, seed, "transition", soft=lambda s, x, A: True) if t_slots else {}

    # ---- contour keys (pieces with a subject)
    contour = _contour_keys(P, S, contour_port_names(), seed)

    # ---- assemble
    out_pieces = []
    for p, d in enumerate(P):
        lines, trs = [], []
        for i in range(d["n"]):
            s = (p, i)
            f = F[s]
            ln = {"index": i, "text": text[s] or None, "module": M[s], "family": family_of(M[s]),
                  "io_in": IO[(p, i, "in")], "io_out": IO[(p, i, "out")], "transition": None,
                  "camera": C[s], "camera_params": hl.camera_params(recs[s], C[s]),
                  "font": f, "font_file": pr.fr.REGISTRY[f]["file"], "role": role(f),
                  "layout": L[s], "layout_record": recs[s]}
            if i < d["n"] - 1:
                ln["transition"] = "cut" if is_cut(p, i) else ("pan" if (p, i) in pan_at else T[(p, i)])
                if (p, i) in pan_at:
                    ln["pan"] = dict(pan_at[(p, i)])
                trs.append(ln["transition"])
            if d["subject"]:
                ln.update(contour[s])
            lines.append(ln)
        out_pieces.append({"id": d["id"], "format": d["format"], "subject": d["subject"], "n_lines": d["n"], "lines": lines,
                           "transitions": trs, "blocks": list(d["blocks"])})
    res = {"seed": int(seed), "pieces": out_pieces}
    res["coverage"] = coverage_report(res)
    return res


def _contour_keys(P, S, ports, seed) -> dict:
    """entry / exit / via / matte per line: ports and directions vary line to line (never the same entry port twice in a row when the piece
    has two or more), matte modes rotate through front | window | fill | wipe | glow, least used first. A piece dict may carry its own
    ``ports`` / ``mattes`` (what its shots support, contour_ports.available)."""
    out: dict = {}
    pcnt: Counter = Counter()
    mcnt: Counter = Counter()
    dcnt: Counter = Counter()
    dirs = contour_dirs()
    for p, d in enumerate(P):
        if not d["subject"]:
            continue
        pp = list(d.get("ports") or ports)
        mm_ = list(d.get("mattes") or matte_modes())
        prev_e = prev_m = None
        for i in range(d["n"]):
            s = (p, i)
            e = min((x for x in pp if x != prev_e or len(pp) == 1), key=lambda x: (pcnt[x], _rank(seed, "ce", *s, x)))
            pcnt[e] += 1
            xo = min((x for x in pp if x != e or len(pp) == 1), key=lambda x: (pcnt[x], _rank(seed, "cx", *s, x)))
            pcnt[xo] += 1
            ed = min(dirs, key=lambda x: (dcnt[x], _rank(seed, "de", *s, x)))
            dcnt[ed] += 1
            xd = min((x for x in dirs if x != ed), key=lambda x: (dcnt[x], _rank(seed, "dx", *s, x)))
            dcnt[xd] += 1
            mm = min((x for x in mm_ if x != prev_m or len(mm_) == 1), key=lambda x: (mcnt[x], _rank(seed, "cm", *s, x)))
            mcnt[mm] += 1
            out[s] = {"entry": {"port": e, "dir": ed}, "exit": {"port": xo, "dir": xd}, "matte": mm}
            if i:
                out[(p, i - 1)]["via"] = {"port": out[(p, i - 1)]["exit"]["port"]}
            prev_e, prev_m = e, mm
    return out


# ================================================================================================== report
def coverage_report(assignment: dict) -> dict:
    """what the batch covers and which rules hold. ``ok`` is True when every universe is covered and nothing is violated."""
    from mograph import fonts_liked as fl
    pcs = assignment["pieces"]
    lines = [(p, ln) for p in pcs for ln in p["lines"]]
    nm = set(no_material())

    def cov(universe, used):
        cnt = Counter(used)
        return {"universe": len(universe), "used": sum(1 for u in universe if cnt[u]), "missing": [u for u in universe if not cnt[u]],
                "counts": {u: cnt[u] for u in universe if cnt[u]}}
    rep = {
        "modules": cov(word_modules(), [ln["module"] for _, ln in lines]),
        "transitions": cov(pr.transition_names() + ["cut"], [t for p in pcs for t in p["transitions"]]),
        "io_styles": cov(io_styles(), [ln[k] for _, ln in lines for k in ("io_in", "io_out")]),
        "cameras": cov(cameras(), [ln["camera"] for _, ln in lines]),
        "layouts": cov([n for n in hl.NAMES if n not in hl.RESERVED or any(ln["layout"] == n for _, ln in lines)], [ln["layout"] for _, ln in lines]),
        "matte_modes": cov(matte_modes(), [ln["matte"] for _, ln in lines if "matte" in ln]),
        "contour_ports": cov(contour_port_names(), [ln[k]["port"] for _, ln in lines for k in ("entry", "exit") if k in ln]),
        "pan_dirs": cov(list(PAN_DIRS), [ln["pan"]["dir"] for _, ln in lines if ln.get("pan")]),
    }
    fonts_used = [ln["font"] for _, ln in lines]
    rep["fonts"] = {"used": len(set(fonts_used)), "liked_display": len(pr._liked_display()),
                    "not_liked": sorted({f for f in fonts_used if not fl.is_liked(f)})}
    rep_ = {"modules": [], "transitions": [], "layouts": []}
    contrast, layout_bad, pairing = [], [], []
    for p in pcs:
        L = p["lines"]
        for key, name in (("module", "modules"), ("layout", "layouts")):
            c = Counter(ln[key] for ln in L)
            rep_[name] += [f"{p['id']}: {k} x{n}" for k, n in c.items() if n > 1 and not (key == "layout" and k in REPEATABLE)]
        rep_["layouts"] += [f"{p['id']}[{a['index']}]: {a['layout']} on neighbouring lines" for a, b in zip(L, L[1:]) if a["layout"] == b["layout"]]
        c = Counter(t for t in p["transitions"] if t not in ("cut", "pan"))
        rep_["transitions"] += [f"{p['id']}: {k} x{n}" for k, n in c.items() if n > 1]
        for a, b in zip(L, L[1:]):
            tag = f"{p['id']}[{a['index']}->{b['index']}]"
            if a["family"] == b["family"]:
                contrast.append(f"{tag}: module family {a['family']}")
            if a["font"] == b["font"]:
                contrast.append(f"{tag}: same font {a['font']}")
            else:
                ratio = max(a["layout_record"]["area"], b["layout_record"]["area"]) / max(min(a["layout_record"]["area"], b["layout_record"]["area"]), 1e-6)
                if not fl.contrast_axes(a["role"], b["role"]) and ratio < 1.25:
                    contrast.append(f"{tag}: fonts {a['font']} / {b['font']} differ in neither weight, fill nor scale")
            if a["layout_record"]["anchored"] and b["layout_record"]["anchored"]:
                contrast.append(f"{tag}: shapes on both lines")
        for ln in L:
            tag = f"{p['id']}[{ln['index']}]"
            layout_bad += [f"{tag} {m}" for m in hl.validate(ln["layout_record"], p["format"])]
            if ln["text"] and not hl.eligible(ln["layout"], ln["text"], p["format"], ln["module"]):
                layout_bad.append(f"{tag}: layout {ln['layout']} not valid for {ln['text']!r} / {ln['module']}")
            if ln["layout_record"]["anchored"] and ln["camera"] != ANCHOR_CAMERA:
                pairing.append(f"{tag}: shapes need {ANCHOR_CAMERA}, got {ln['camera']}")
            if ln["camera"] in FREE_CAMERAS and ln["layout_record"]["box"]:
                pairing.append(f"{tag}: {ln['camera']} leaves the {ln['layout']} box")
            if ln["transition"] is not None:
                carries = ln["io_out"] not in nm and L[ln["index"] + 1]["io_in"] not in nm
                if carries == (ln["transition"] == "cut"):
                    pairing.append(f"{tag}: transition {ln['transition']} with io {ln['io_out']} -> {L[ln['index'] + 1]['io_in']}")
        n_free = Counter(ln["camera"] in FREE_CAMERAS for ln in L)[True]
        if n_free > 1:
            pairing.append(f"{p['id']}: {n_free} free-form camera moves")
    doms = [p["blocks"][0] for p in pcs if p.get("blocks")]
    rep["blocks"] = {"dominant": doms, "repeated_dominant": sorted({b for b in doms if doms.count(b) > 1}),
                     "used": sorted({b for p in pcs for b in p.get("blocks", [])})}
    for p in pcs:                                               # lines the blocks constrain must satisfy them
        cs = bk.line_constraints(p.get("blocks", []), p["n_lines"])
        for ln in p["lines"]:
            c = cs[ln["index"]]
            for key, val in (("module", ln["module"]), ("layout", ln["layout"]), ("camera", ln["camera"])):
                if c.get(key) is not None and val not in c[key]:
                    pairing.append(f"{p['id']}[{ln['index']}]: block constraint {key} {val} not in {c[key]}")
    rep["in_piece_repeats"] = rep_
    rep["contrast_violations"] = contrast
    rep["layout_problems"] = layout_bad
    rep["pairing_problems"] = pairing
    cover = ("modules", "transitions", "io_styles", "cameras", "layouts")
    rep["complete"] = all(not rep[k]["missing"] for k in cover)
    rep["ok"] = bool(rep["complete"] and not rep["blocks"]["repeated_dominant"] and not any(rep_.values()) and not contrast and not layout_bad and not pairing and not rep["fonts"]["not_liked"])
    return rep


def format_report(rep: dict) -> str:
    """one screen of text for a coverage report."""
    rows = []
    for k in ("modules", "transitions", "io_styles", "cameras", "layouts", "matte_modes", "contour_ports") + (("pan_dirs",) if rep.get("pan_dirs", {}).get("counts") else ()):
        r = rep[k]
        rows.append(f"{k:14s} {r['used']:2d}/{r['universe']:2d}" + (f"  MISSING {', '.join(r['missing'])}" if r["missing"] else ""))
    f = rep["fonts"]
    rows.append(f"{'fonts':14s} {f['used']:2d} of {f['liked_display']} liked display faces; not liked: {f['not_liked'] or 'none'}")
    rows.append(f"in-piece repeats: {sum(len(v) for v in rep['in_piece_repeats'].values())}  contrast violations: {len(rep['contrast_violations'])}  "
                f"layout problems: {len(rep['layout_problems'])}  pairing problems: {len(rep['pairing_problems'])}  ok: {rep['ok']}")
    return "\n".join(rows)


# ================================================================================================== plan a batch
def plan_batch(requests, musics, seed: int = 0, dry: bool = True, size=None, rounds: int = 3, briefs=None, use_blocks: bool = True):
    """plan_rules.plan_piece for every request with its slice of ONE batch assignment applied. The final line count of a plan depends on the
    timing (lines merge / split to fit the bars), so the plan is first made without an assignment (cheap, no dry run) to learn the lines,
    the lego blocks are chosen for every piece (no two pieces with the same dominant block) from what that plan offers (shots, technique,
    lines, holds) and the brief's blocks + direction, the batch is assigned on them, and the pieces are planned again; if a plan still ends
    on another line count the batch is re-assigned on the new counts (up to ``rounds``).
    briefs: optional raw Qwen brief text / dict per request (plan_piece's brief_text). -> (pieces, assignment)."""
    from mograph import fusion_brief as fb
    reqs = list(requests)
    ms = [musics] * len(reqs) if isinstance(musics, dict) else list(musics)
    brs = list(briefs) if briefs else [None] * len(reqs)
    if len(ms) != len(reqs) or len(brs) != len(reqs):
        raise ValueError("one music analysis (and brief) per request (or a single music dict for all)")
    base = [pr.plan_piece(r, m, b, dry=False, size=size) for r, m, b in zip(reqs, ms, brs)]
    parsed = [fb.parse_brief(b, r) for b, r in zip(brs, reqs)]
    for br, r in zip(parsed, reqs):                                                   # request key `pan` (True | n | 'down' | [dirs]): the camera_pan_transition block
        if r.get("pan") not in (None, False):
            br["blocks"] = [bk.PAN_BLOCK] + [x for x in (br.get("blocks") or []) if x != bk.PAN_BLOCK][:fb.MAX_BLOCKS - 1]
    descr = [{"id": b["id"], "lines": [w["text"] for w in b["type"]["words"]], "format": pr.format_of(b), "subject": pr.piece_has_subject(b),
              "ports": piece_ports(b)[0], "mattes": piece_ports(b)[1], "pan": r.get("pan")} for b, r in zip(base, reqs)]
    asg, pieces = None, base
    for rnd in range(max(1, rounds)):
        names = bk.batch_blocks([pr.block_ctx(b) for b in pieces], seed, parsed) if use_blocks else [[] for _ in base]
        for d, nm in zip(descr, names):
            d["blocks"] = nm
        asg = assign_batch(descr, seed)
        for a_, br, nm, b in zip(asg["pieces"], parsed, names, pieces):
            a_["direction"] = br.get("direction") or {}
            ctx = pr.block_ctx(b)
            a_["block_notes"] = [f"block {n}: named by the brief, not used ({'; '.join(bk.eligible(n, dict(ctx, opt_in=(n,)))[1]) or 'a better fit was chosen'})"
                                 for n in (br.get("blocks") or []) if n not in nm]
        pieces = [pr.plan_piece(r, m, b, dry=dry, size=size, assignment=a) for r, m, b, a in zip(reqs, ms, brs, asg["pieces"])]
        got = [[w["text"] for w in p["type"]["words"]] for p in pieces]
        if all(len(g) == d["n_lines"] for g, d in zip(got, asg["pieces"])):
            break
        for d, g in zip(descr, got):
            d["lines"] = g
    for p in pieces:                                   # the blocks against the FINAL plan: flag any whose needs the last timing no longer meets
        ctx = pr.block_ctx(p)
        for b in p["plan"].get("blocks", []):
            ok, why = bk.eligible(b["name"], dict(ctx, opt_in=(b["name"],)))          # a block the piece carries was named, so opt-in blocks are checked on their other needs
            b["eligible_on_final_plan"] = ok
            if not ok:
                p["plan"]["notes"].append(f"block {b['name']}: needs not met by the final plan ({'; '.join(why)})")
    return pieces, asg
