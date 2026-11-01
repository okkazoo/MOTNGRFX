"""transition_rubric - numbers on the transitions of ONE piece JSON, for the open-model par tables (20_open-model-design, rubric columns of round r7).
A presence / variety check like piece_brief.rubric, not a judge of how the transition looks. Pure functions on the piece dict, no engine, no network.

    stats(piece)        -> dict: text (names per gap, used, distinct, cuts, repeats), io (styles beyond lines), video (kind per shot boundary, beyond),
                           linked (window cuts / window_other words / pans / contour ports), repeats (text + video), gaps, boundaries
    columns(piece)      -> the table columns: text_used, text_distinct, video_beyond, linked, repeats + the 0-2 scores text / video / linked / repeats
                           (`text_auto` = the piece says "auto": the engine's pick is not in the JSON, so its text score is 0 and must be read with this flag; a score is None where the piece cannot have one: a single-shot piece has no video boundary, a one-word piece no text gap)

Scores (0-2): text = distinct non-cut transitions in the gaps (2: >= min(3, gaps) distinct and no repeat; 1: >= 1 or an `auto`/explicit mix; 0: none);
video = boundaries made by something other than a plain hard cut / crossfade (2: a window cut / swap / pivot / pan; 1: both hard cuts AND crossfades
used on purpose, or an `io` style carrying the boundary; 0: all boundaries the same plain kind); linked = 2 when a window cut / swap, a pan or a contour
`via` port exists, 1 when only window_source words / entry-exit ports exist, else 0; repeats = 2 when nothing repeats, 1 for one repeat, 0 for more.
"""
from __future__ import annotations

from tools.fusion import cut_kinds as CK

LINKED_CUTS = ("window_cut", "window_swap", "window_zoom")
VIDEO_BEYOND = ("window_cut", "window_swap", "window_zoom", "window_other") + CK.SHOT_3D + CK.PANS
PLAIN = ("hard", "crossfade")


def _name(x):
    """a `type.transitions` entry -> its name (str / {module | kind} -> str, null -> None)."""
    if isinstance(x, dict):
        x = x.get("module") or x.get("kind")
    return str(x) if x else None


def _text(piece):
    t = piece.get("type") or {}
    words = t.get("words") or []
    gaps = max(len(words) - 1, 0)
    tr = t.get("transitions", "auto")
    names: list = []
    if isinstance(tr, list):
        names = [_name(x) for x in tr][:gaps] + [None] * max(gaps - len(tr), 0)
    elif tr == "pan":
        names = ["pan"] * gaps
    else:
        names = [None] * gaps                                   # "auto": the engine picks, nothing is written
    for i, w in enumerate(words[:-1] if words else []):
        wt = w.get("transition") if isinstance(w, dict) else None
        if wt and i < len(names):
            names[i] = _name(wt)                               # a word-level pan replaces the gap after it
    cam = t.get("camera")
    if isinstance(cam, dict) and cam.get("move") == "pan":
        names = [n or "pan" for n in names]
    named = [n for n in names if n]
    nocut = [n for n in named if n != "cut"]
    seen, rep = set(), 0
    for n in nocut:
        rep += n in seen
        seen.add(n)
    return dict(names=names, gaps=gaps, used=len(named), distinct=len(set(nocut)), cuts=len(named) - len(nocut), repeats=rep, auto=(tr == "auto"),
                pans=sum(1 for n in names if n == "pan"))


def _io(piece):
    st = set()
    for w in (piece.get("type") or {}).get("words") or []:
        for k in ("form_in", "form_out"):
            v = w.get(k)
            v = v[0] if isinstance(v, (list, tuple)) and v else v
            if v and v != "lines":
                st.add(str(v))
    return sorted(st)


def _video(piece):
    shots = piece.get("shots") or []
    n = max(len(shots) - 1, 0)
    comp = piece.get("compose") or {}
    cf = comp.get("crossfade", 0)
    cfl = list(cf) if isinstance(cf, (list, tuple)) else [cf] * n
    cfl = (cfl + [cfl[-1] if cfl else 0] * n)[:n]
    kinds = ["crossfade" if (c or 0) > 0 else "hard" for c in cfl]
    ids = [s.get("id") for s in shots]
    for c in comp.get("cuts") or []:
        if c.get("after") in ids and ids.index(c["after"]) < n and c.get("cut"):
            kinds[ids.index(c["after"])] = str(c.get("was") or c["cut"])           # a window_other is compiled to the word's window_source + a hard cut: `was` keeps its name
    beyond = [k for k in kinds if k in VIDEO_BEYOND]
    rep = len(beyond) - len(set(beyond))
    three_d = [k for k in kinds if k in CK.SHOT_3D]
    return dict(kinds=kinds, boundaries=n, beyond=len(beyond), repeats=rep, distinct=len(set(kinds)), three_d=len(three_d),
                windows=sum(1 for k in kinds if k in CK.WINDOW), pans=sum(1 for k in kinds if k in CK.PANS))


def _chain_ops(piece, sid):
    comp = piece.get("compose") or {}
    ch = (comp.get("shots") or {}).get(sid) or comp.get("default") or {}
    return [o.get("op") for o in ch.get("ops") or [] if isinstance(o, dict)]


def _subject_front(piece):
    """share of the 3D cuts whose subject stays crisp in front of the type while the background transforms: matte_pass not switched off AND the type is
    sandwiched behind the matte (a `sandwich` / `depth_occlude` op) in the chains of both shots. None without a 3D cut."""
    ids = [s.get("id") for s in piece.get("shots") or []]
    cuts = [c for c in (piece.get("compose") or {}).get("cuts") or [] if c.get("cut") in CK.SHOT_3D and c.get("after") in ids]
    if not cuts:
        return None
    ok = 0
    for c in cuts:
        k = ids.index(c["after"])
        if k + 1 >= len(ids):
            continue
        behind = all(any(o in ("sandwich", "depth_occlude", "depth_cover") for o in _chain_ops(piece, sid)) for sid in (ids[k], ids[k + 1]))
        ok += bool(behind and (c.get("params") or {}).get("matte_pass", True))
    return round(ok / len(cuts), 2)


def _shared_direction(piece):
    """share of the directional 3D cuts whose neighbouring words slide the same way (word A `exit`, word B `entry` = cut_kinds.sides). The design step writes the
    sides (design_options.shot_cut_words), so this checks the compile, not the model. None without a directional 3D cut."""
    ids = [s.get("id") for s in piece.get("shots") or []]
    words = (piece.get("type") or {}).get("words") or []
    cuts = [c for c in (piece.get("compose") or {}).get("cuts") or [] if CK.sides(str(c.get("cut"))) and not str(c.get("cut")).startswith("pan") and c.get("after") in ids]
    if not cuts:
        return None
    ok = 0
    for c in cuts:
        sd = CK.sides(c["cut"])
        ok += any(words[i].get("exit") == sd[0] and words[i + 1].get("entry") == sd[1] for i in range(len(words) - 1))
    return round(ok / len(cuts), 2)


def _linked(piece, text):
    comp = piece.get("compose") or {}
    cuts = [c for c in comp.get("cuts") or [] if c.get("cut") in LINKED_CUTS]
    words = (piece.get("type") or {}).get("words") or []
    src = [w for w in words if w.get("window_source") not in (None, "", "self")]
    ports = [(w.get("text"), k) for w in words for k in ("entry", "exit", "via") if isinstance(w.get(k), dict)]
    via = [p for p in ports if p[1] == "via"]
    return dict(window_cuts=len(cuts), window_words=len(src), pans=text["pans"], ports=len(ports), via=len(via))


def stats(piece: dict) -> dict:
    tx, vd = _text(piece), _video(piece)
    return dict(text=tx, io=_io(piece), video=vd, linked=_linked(piece, tx), repeats=tx["repeats"] + vd["repeats"], gaps=tx["gaps"], boundaries=vd["boundaries"],
                subject_front=_subject_front(piece), shared_direction=_shared_direction(piece))


def columns(piece: dict) -> dict:
    s = stats(piece)
    tx, vd, lk, io = s["text"], s["video"], s["linked"], s["io"]
    text = None if tx["gaps"] == 0 else (2 if tx["distinct"] >= min(3, tx["gaps"]) and tx["repeats"] == 0 else 1 if tx["used"] >= 1 else 0)
    if vd["boundaries"] == 0:
        video = None
    elif vd["beyond"] or lk["pans"]:
        video = 2
    elif len(set(vd["kinds"])) > 1 or io:
        video = 1
    else:
        video = 0
    linked = 2 if (lk["window_cuts"] or lk["pans"] or lk["via"]) else 1 if (lk["window_words"] or lk["ports"]) else 0
    rep = 2 if s["repeats"] == 0 else 1 if s["repeats"] == 1 else 0
    return dict(shot_kinds=vd["kinds"], shot_distinct=vd["distinct"], three_d=vd["three_d"], windows=vd["windows"] + max(lk["window_words"] - sum(1 for k in vd["kinds"] if k == "window_other"), 0),
                subject_front=s["subject_front"], shared_direction=s["shared_direction"],
                text_used=tx["used"], text_distinct=tx["distinct"], video_beyond=vd["beyond"] + max(lk["pans"] - vd["pans"], 0), linked=lk["window_cuts"] + lk["window_words"] + lk["pans"] + lk["via"],
                repeats=s["repeats"], text_auto=bool(tx["auto"]), text=text, video=video, linked_score=linked, repeats_score=rep)
