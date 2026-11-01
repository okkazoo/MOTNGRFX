"""ltx_check: did LTX really form the letters of an LTX-owned word? If not, the word stays flat (pure functions + numpy / cv2, no ComfyUI, no GPU). NO CLAUDE.

The bug it fixes (batch 3060 v005 r01): the text-rendered background (`sK_textbg.mp4`, the union-control V2V pass) formed NO letters, but every `handoff` word's
flat type still faded out at 50 % of its span (tools/fusion/owner.py `flat_gain`): the word simply vanished.

Check (three-layer shots only: the shot has BOTH the clean clip and the text-rendered clip, nodes_fusion/compose.py compose_clips): per LTX-owned word (render
ltx / handoff), over the frames from the moment LTX's letters should be fully in (handoff time + the control ramp; the word's start for `ltx`) to the end of the word,
sample up to MAX_SAMPLES frames and measure the mean absolute difference between the text-rendered clip and the clean clip

    inside   = inside the word's mask (the flat type matte of that frame, dilated: LTX draws its letters roughly where the control pass put them)
    outside  = everywhere else (the V2V pass drifts the whole scene a little: grade, small motion; that drift is the baseline)
    excess   = inside - outside

excess >= MIN_EXCESS -> the letters are there. Below -> `ltx letters missing -> flat`: the word's render becomes "flat" for its whole span (compose then draws it
as ordinary flat type, no fade-out) and the decision is logged into piece.plan.owner. Fewer than MIN_FRAMES usable frames (no matte pass, the word is not in the shot,
the mask is empty) -> unknown: nothing changes.

Threshold (measured on the real batch-3060 v005 r01 clips s1_clean.mp4 vs s1_textbg.mp4, 360x640, which show NO letters, tests/test_owner_concept.py; the type matte
is not archived, so a stand-in word mask is drawn in the middle of the frame, 5 words at 5 heights): no-letters excess per word window = -0.010, +0.011, +0.015, +0.040,
+0.043 (the V2V drift: grade + small motion; the highest values are where the mask sits over the moving hands). Solid stone-grey letters drawn into the same mask give
+0.26 .. +0.28. MIN_EXCESS = 0.08: ~1.9x above the worst no-letters window, ~3.3x below solid letters, so weaker, translucent LTX letters (excess ~0.10+) still pass.
A false "missing" leaves LTX's letters AND the flat word on screen (two copies, the old bug); a false "formed" lets the word vanish: the threshold sits on the no-letters side.
`piece.layers.ltx_check = false` switches the check off, `piece.layers.ltx_min_excess` overrides the threshold.
"""
# ------------------------------------------------------------------------------------------------ VLM reading check (2026-10-28, the DECIDER)
# The pixel test above cannot judge legibility: batch 3060 v007 r01 passed icicles / a blue slab with no letters (excess 0.097 > 0.08) and "BRAAKER" for "STORM BREAKER"
# (0.112). The decider is now a vision-language reading: after the text pass, up to 3 frames of the text-rendered clip after the handoff are cropped to the word's dilated
# matte box and the design VLM (the same arm: nodes_fusion/layers.py MographLtxRead, llama.cpp) is asked READ_PROMPT. The reading is compared with the word, letters only,
# upper case: EXACT match by default (an LTX word is <= owner.MAX_LTX_LETTERS letters; a misspelled real word is worse than flat); `piece.layers.ltx_read_dist` = an int
# allows that edit distance, "fuzzy" = the tiers of fuzzy_dist (<= 1 for <= 6 letters, else <= 2). A word passes when a majority of its crops read right. A word without
# readings (no VLM node, no matte, no crop) falls back to the pixel decision; the pixel numbers are always logged in plan.owner.ltx_check. `piece.layers.ltx_read = false`
# opts out of the reading.
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from tools.fusion import owner as OW

MIN_EXCESS = 0.08
MAX_SAMPLES = 12
MIN_FRAMES = 3                  # fewer usable frames than this: unknown, nothing changes
MIN_COVER = 0.004               # a frame counts only when the word's mask covers at least this share of the picture
DILATE_FRAC = 0.03              # dilation of the mask: this share of the short side
MASK_ON = 32                    # matte level (0..255) that counts as letter


def _u8_frame(clip, i: int) -> np.ndarray:
    """clip[i] (numpy or torch, uint8 or float) -> float32 HxWx3 in 0..1."""
    f = clip[int(i)]
    if hasattr(f, "detach"):
        f = f.detach().cpu().numpy()
    f = np.asarray(f)
    if f.dtype == np.uint8:
        return f.astype(np.float32) / 255.0
    return f.astype(np.float32)


def _dilated(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """gray matte (any size) -> bool mask at hw = (H, W), dilated."""
    H, W = hw
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
    m = (mask >= MASK_ON).astype(np.uint8)
    k = max(5, int(round(DILATE_FRAC * min(H, W))) | 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def score_frames(pairs) -> dict | None:
    """pairs = iterable of (clean HxWx3, text HxWx3, gray matte HxW uint8) -> {inside, outside, excess, n} (means over the usable frames) or None (< MIN_FRAMES)."""
    ins, out = [], []
    for clean, text, mask in pairs:
        if clean is None or text is None or mask is None:
            continue
        if text.shape != clean.shape:
            text = cv2.resize(text, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
        m = _dilated(mask, clean.shape[:2])
        cover = float(m.mean())
        if cover < MIN_COVER or cover > 0.9:
            continue
        d = np.abs(text - clean).mean(-1)
        ins.append(float(d[m].mean()))
        out.append(float(d[~m].mean()))
    if len(ins) < MIN_FRAMES:
        return None
    i, o = float(np.mean(ins)), float(np.mean(out))
    return dict(inside=round(i, 4), outside=round(o, 4), excess=round(i - o, 4), n=len(ins))


# ------------------------------------------------------------------------------------------------ LAYOUT check (2026-10-29): the rendered letters must sit where the graphic sits
# v005 r02 "DRIFT": the control pass was a dotted word on one line, LTX drew its OWN stacked letters elsewhere; the VLM could not read them and the word fell back to flat, but a reading
# alone does not say whether letters that DO read are the graphic's layout. The layout check compares the letters LTX drew (where the text-rendered clip differs from the clean clip, inside
# a wide box round the word) with the flat type matte: IoU of the two masks, the flat matte dilated by LAYOUT_TOL of the short side (LTX extrudes and shades its letters). An LTX word only
# counts as LTX-owned when the reading passes AND the median IoU is >= LAYOUT_IOU_MIN (`piece.layers.ltx_iou` = a number overrides, false switches the check off).
LAYOUT_IOU_MIN = 0.5
LAYOUT_TOL = 0.02
LAYOUT_BOX = 2.5                # the search box = the word's mask dilated by this many times DILATE_FRAC
LAYOUT_DIFF = 0.07              # a pixel counts as a letter when the clips differ by this much (0..1 mean over channels) after smoothing


def letter_mask(clean: np.ndarray, text: np.ndarray, flat_matte: np.ndarray) -> np.ndarray:
    """bool HxW: where the text-rendered frame differs from the clean frame inside the wide box round the flat word (the letters the video model drew)."""
    if text.shape != clean.shape:
        text = cv2.resize(text, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
    H, W = clean.shape[:2]
    m = (cv2.resize(flat_matte, (W, H), interpolation=cv2.INTER_AREA) if flat_matte.shape[:2] != (H, W) else flat_matte) >= MASK_ON
    k = max(5, int(round(DILATE_FRAC * LAYOUT_BOX * min(H, W))) | 1)
    box = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    d = cv2.GaussianBlur(np.abs(text - clean).mean(-1), (0, 0), max(1.0, 0.004 * min(H, W)))
    return (d >= LAYOUT_DIFF) & box


def layout_iou_frames(triples) -> dict | None:
    """triples = (clean, text, flat gray matte uint8) per frame -> {iou (median), n}: IoU of the drawn letters with the flat matte (dilated by LAYOUT_TOL). None: < MIN_FRAMES usable."""
    vals = []
    for clean, text, mask in triples:
        if clean is None or text is None or mask is None:
            continue
        H, W = clean.shape[:2]
        flat = (cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA) if mask.shape[:2] != (H, W) else mask) >= MASK_ON
        if not MIN_COVER <= float(flat.mean()) <= 0.9:
            continue
        k = max(3, int(round(LAYOUT_TOL * min(H, W))) | 1)
        fat = cv2.dilate(flat.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        drawn = letter_mask(clean, text, mask)
        union = float((drawn | fat).sum())
        if union > 0:
            vals.append(float((drawn & fat).sum()) / union)
    if len(vals) < MIN_FRAMES:
        return None
    return dict(iou=round(float(np.median(vals)), 4), n=len(vals))


def _matte_files(type_dir) -> list[str]:
    from tools.fusion.recipe_eval import natural
    d = Path(type_dir) / "matte" if type_dir else None
    return sorted((str(p) for p in d.glob("*.png")), key=natural) if d is not None and d.is_dir() else []


def _read_gray(path: str):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    return im


def word_frames(w: dict, fps: int, shot_start: int, shot_frames: int) -> list[int]:
    """piece frames to measure for LTX-owned word window `w` inside the shot [shot_start, shot_start + shot_frames): from where LTX's letters are fully in to the
    end of the word, at most MAX_SAMPLES evenly spaced."""
    t_from = OW.handoff_t(w) + OW.CONTROL_RAMP_S if w["render"] == "handoff" else w["start"] + OW.CONTROL_RAMP_S
    a = max(int(math.ceil(t_from * fps)), int(shot_start))
    b = min(int(math.floor(w["end"] * fps)), int(shot_start) + int(shot_frames) - 1)
    if b < a:
        return []
    n = b - a + 1
    if n <= MAX_SAMPLES:
        return list(range(a, b + 1))
    return sorted({int(round(a + k * (n - 1) / (MAX_SAMPLES - 1))) for k in range(MAX_SAMPLES)})


def check_shot(piece: dict, sid: str, type_dir, clean, text, wins=None) -> list[dict]:
    """per LTX-owned word of shot `sid`: {i, text, render, inside, outside, excess, n, formed (bool | None = unknown)}. `clean` / `text` = the shot's clean and
    text-rendered clips [T,H,W,3] (frame j = piece frame shot_start + j)."""
    if wins is None:
        from tools.fusion.recipe_eval import owner_windows
        wins = owner_windows(piece, type_dir)
    if wins is None:
        return []
    from tools.fusion.recipe_eval import piece_layout
    lay, _ = piece_layout(piece)
    L = lay[sid]
    fps = int(piece.get("fps") or 24)
    n_avail = min(int(clean.shape[0]), int(text.shape[0]), int(L["frames"]))
    files = _matte_files(type_dir)
    thr = float(((piece.get("layers") or {}).get("ltx_min_excess")) or MIN_EXCESS)
    res = []
    for i, w in enumerate(wins):
        if w["render"] not in ("ltx", "handoff"):
            continue
        in_frames = word_frames(w, fps, L["start"], n_avail)
        fr = [f for f in in_frames if f < len(files)]
        r = None
        li = None
        if files and fr:
            r = score_frames((_u8_frame(clean, f - L["start"]), _u8_frame(text, f - L["start"]), _read_gray(files[f])) for f in fr)
            li = layout_iou_frames((_u8_frame(clean, f - L["start"]), _u8_frame(text, f - L["start"]), _read_gray(files[f])) for f in fr)
        row = dict(i=i, text=w["text"], render=w["render"], thr=thr, in_shot=bool(in_frames))
        row.update(r or dict(inside=None, outside=None, excess=None, n=0))
        row["formed"] = None if r is None else bool(r["excess"] >= thr)
        row["iou"] = None if li is None else li["iou"]
        iou_min = (piece.get("layers") or {}).get("ltx_iou", LAYOUT_IOU_MIN)
        row["layout_ok"] = None if (li is None or iou_min is False) else bool(li["iou"] >= float(iou_min))
        res.append(row)
    return res


READ_PROMPT = "Read the text in this image exactly. Reply with the text only, or NONE."
READ_FRAMES = 3                 # crops per word
READ_PAD = 0.15                 # padding of the crop box, share of the box's size
READ_MIN_SIDE = 384             # the crop is upscaled until its long side is at least this (a VLM reads small crops badly)


def norm_letters(s) -> str:
    """letters only, upper case: 'Storm  Braaker!' -> 'STORMBRAAKER'."""
    return re.sub(r"[^A-Z]", "", str(s or "").upper())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_dist(n_letters: int) -> int:
    """the fuzzy tiers: <= 1 for words of <= 6 letters, <= 2 for longer ones."""
    return 1 if n_letters <= 6 else 2


def allowed_dist(piece: dict, word: str) -> int:
    v = (piece.get("layers") or {}).get("ltx_read_dist", 0)
    if v == "fuzzy":
        return fuzzy_dist(len(norm_letters(word)))
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def read_matches(word: str, reading: str, dist: int = 0) -> bool:
    w, r = norm_letters(word), norm_letters(reading)
    if not w or not r or (r == "NONE" and w != "NONE"):
        return False
    return edit_distance(w, r) <= int(dist)


def read_verdict(word: str, readings: list, dist: int = 0) -> dict:
    """{ok, matches, n}: ok when more than half of the readings (>= n // 2 + 1) match the word."""
    n = len(readings)
    m = sum(read_matches(word, r, dist) for r in readings)
    return dict(ok=bool(n and m >= n // 2 + 1), matches=m, n=n)


def crop_jobs(piece: dict, sid: str, type_dir, text, wins=None, n: int = READ_FRAMES) -> list[dict]:
    """per LTX-owned word of shot `sid`: up to `n` crops of the text-rendered clip `text` [T,H,W,3] (frame j = piece frame shot_start + j) after the handoff, cut to the
    bounding box of the word's dilated type matte of that frame -> [{i, text, render, frame, image (uint8 HxWx3, upscaled)}]. No matte pass / no usable frame: no job."""
    if wins is None:
        from tools.fusion.recipe_eval import owner_windows
        wins = owner_windows(piece, type_dir)
    if wins is None:
        return []
    from tools.fusion.recipe_eval import piece_layout
    lay, _ = piece_layout(piece)
    L = lay[sid]
    fps = int(piece.get("fps") or 24)
    n_avail = int(text.shape[0])
    files = _matte_files(type_dir)
    jobs = []
    for i, w in enumerate(wins):
        if w["render"] not in ("ltx", "handoff") or not files:
            continue
        good = []
        for f in word_frames(w, fps, L["start"], n_avail):
            if f >= len(files):
                continue
            m = _dilated(_read_gray(files[f]), tuple(text.shape[1:3]))
            if MIN_COVER <= float(m.mean()) <= 0.9:
                good.append((f, m))
        if not good:
            continue
        pick = sorted({int(round(k * (len(good) - 1) / max(n - 1, 1))) for k in range(min(n, len(good)))})
        for k in pick:
            f, m = good[k]
            ys, xs = np.where(m)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            py, px = int(round((y1 - y0) * READ_PAD)), int(round((x1 - x0) * READ_PAD))
            H, W = m.shape
            y0, y1, x0, x1 = max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)
            fr = text[f - L["start"]]
            fr = fr.detach().cpu().numpy() if hasattr(fr, "detach") else np.asarray(fr)
            if fr.dtype != np.uint8:
                fr = np.clip(np.rint(fr * 255.0), 0, 255).astype(np.uint8)
            crop = np.ascontiguousarray(fr[y0:y1, x0:x1, :3])
            sc = READ_MIN_SIDE / float(max(crop.shape[:2]))
            if sc > 1.0:
                crop = cv2.resize(crop, (int(round(crop.shape[1] * sc)), int(round(crop.shape[0] * sc))), interpolation=cv2.INTER_CUBIC)
            jobs.append(dict(i=i, text=w["text"], render=w["render"], frame=int(f), image=crop))
    return jobs


def reads_from(jobs: list[dict], replies: list[str]) -> list[dict]:
    """crop jobs + the VLM's replies (same order) -> [{i, text, frames, readings}], one row per word (JSON-able; rides on the clip as `ltx_reads`)."""
    rows: dict[int, dict] = {}
    for j, r in zip(jobs, replies):
        row = rows.setdefault(j["i"], dict(i=j["i"], text=j["text"], frames=[], readings=[]))
        row["frames"].append(j["frame"])
        row["readings"].append(str(r).strip())
    return list(rows.values())


def e_is_missing(by_i: dict, r: dict) -> bool:
    return r["i"] not in by_i


def apply_fallback(piece: dict, sid: str, type_dir, clean, text, reads=None) -> list[str]:
    """run check_shot and, for every word whose LTX letters are missing, set its render to "flat" IN PLACE on piece["type"]["words"] (ltx_at / ltx_t / material
    dropped, `ltx_missing` = True, the asked render kept in `render_asked`); log into piece["plan"]["owner"]: `fallback` (one text per demoted word) and `ltx_check`
    (every measurement). -> the fallback texts (empty = nothing changed). Not layered (no clean or no text clip) / opted out (piece.layers.ltx_check false): nothing."""
    if clean is None or text is None or ((piece.get("layers") or {}).get("ltx_check") is False):
        return []
    rows = check_shot(piece, sid, type_dir, clean, text)
    if not rows:
        return []
    words = (piece.get("type") or {}).get("words") or []
    po = piece.setdefault("plan", {}).setdefault("owner", {})
    msgs: list[str] = []
    by_i = {int(e["i"]): e for e in (reads or []) if e.get("readings")} if (piece.get("layers") or {}).get("ltx_read") is not False else {}
    for r in rows:
        pixel_formed = r["formed"]
        unverified = r["formed"] is None and r.get("in_shot") and e_is_missing(by_i, r) and (piece.get("layers") or {}).get("ltx_strict") is not False
        e = by_i.get(r["i"])
        if e is not None:                                  # the VLM reading decides; the pixel numbers stay in the log
            v = read_verdict(r["text"], e["readings"], allowed_dist(piece, r["text"]))
            r["formed"] = v["ok"]
            r["read"] = list(e["readings"])
            r["read_matches"] = f"{v['matches']}/{v['n']}"
        if unverified:                                     # the word is in this shot but nothing could be measured (no matte pass, too few frames) and no reading: it did not PASS, so it is flat
            r["formed"] = False
        layout_veto = r["formed"] is not False and r.get("layout_ok") is False          # the letters read (or the pixels differ) but they are not the graphic's layout
        if layout_veto:
            r["formed"] = False
        po.setdefault("ltx_check", []).append({k: r[k] for k in ("i", "text", "render", "inside", "outside", "excess", "n", "formed", "thr", "iou", "layout_ok") if k in r}
                                              | ({"pixel_formed": pixel_formed, "read": r["read"], "read_matches": r["read_matches"]} if "read" in r else {}) | {"shot": sid})
        if r["formed"] is not False:
            continue
        w = words[r["i"]]
        w["render_asked"] = w.get("render")
        w["render"] = "flat"
        w["ltx_missing"] = True
        for k in ("ltx_at", "ltx_t", "material"):
            w.pop(k, None)
        if unverified:
            msg = (f"ltx letters not verified -> flat: type.words[{r['i']}] {r['text']!r} ({r['render']} -> flat, shot {sid}): the text-rendered clip could not be measured for this word (no readable matte frames) "
                   f"and no reading arrived; a word counts as LTX-owned only when it passed (piece.layers.ltx_strict = false trusts it); the word stays flat for its whole span")
        elif layout_veto:
            msg = (f"ltx letters not the graphic's layout -> flat: type.words[{r['i']}] {r['text']!r} ({r['render']} -> flat, shot {sid}): the letters the video model drew overlap the flat graphic by IoU "
                   f"{r['iou']:.2f} (< {float((piece.get('layers') or {}).get('ltx_iou', LAYOUT_IOU_MIN)):g}); the word stays flat for its whole span")
        elif "read" in r:
            px = f"pixel excess {r['excess']:.3f} (log only)" if r.get("excess") is not None else "no pixel measurement"
            msg = (f"ltx letters unreadable -> flat: type.words[{r['i']}] {r['text']!r} ({r['render']} -> flat, shot {sid}): the vision model read {r['read']} "
                   f"({r['read_matches']} crops match, edit distance <= {allowed_dist(piece, r['text'])}); {px}; the word stays flat for its whole span")
        else:
            msg = (f"ltx letters missing -> flat: type.words[{r['i']}] {r['text']!r} ({r['render']} -> flat, shot {sid}): the text-rendered clip differs from the clean clip by "
                   f"{r['inside']:.3f} inside the word's mask vs {r['outside']:.3f} outside (excess {r['excess']:.3f} < {r['thr']:g}, {r['n']} frames); the word stays flat for its whole span")
        msgs.append(msg)
    if msgs:
        po.setdefault("fallback", []).extend(msgs)
        piece["plan"].setdefault("notes", []).extend(msgs)
    return msgs


# ------------------------------------------------------------------------------------------------ the background of a layered shot (2026-10-29)
# The matte of a three-layer shot is cut from the CLEAN clip and the subject layer is the clean clip. The text-rendered clip (union-control V2V of the clean one) is only the
# background under the type, and only worth having while it holds LTX letters that survived the checks above. Two cases put it at odds with the matte, and in both the clean
# clip becomes the background (background == subject == the picture the matte was cut from, so no second person can show):
#   1. no LTX-owned word is left in the shot (every one fell back to flat, or the design had none): the text pass may hold a half-formed word or LTX's own layout of the letters;
#   2. the subject of the text-rendered clip sits more than matte_align.DRIFT_MAX of the short side off the subject of the clean clip (V2V re-draws the scene).
def shot_ltx_words(piece: dict, sid: str, type_dir=None) -> list[int]:
    """indices of the words LTX still owns somewhere in shot `sid` (render ltx / handoff with a control gain > 0 inside the shot)."""
    from tools.fusion.recipe_eval import owner_windows
    wins = owner_windows(piece, type_dir)
    if wins is None:
        return []
    sp = OW.shot_span_s(dict(piece, layout=piece.get("layout") or _layout(piece)), sid)
    if sp is None:
        return []
    own = OW.shot_words(wins, sp[0], sp[1])
    return [i for i, w in enumerate(wins) if any(w is o for o in own)]


def _layout(piece: dict) -> dict:
    from tools.fusion.recipe_eval import piece_layout
    return piece_layout(piece)[0]


def demote_shot_words(piece: dict, idx: list[int], why: str) -> list[str]:
    """the words `idx` of the piece become flat for their whole span (like apply_fallback): ltx_at / ltx_t / material dropped, `ltx_missing`, the asked render kept. -> fix texts."""
    words = (piece.get("type") or {}).get("words") or []
    out = []
    for i in idx:
        w = words[i]
        if w.get("render") == "flat":
            continue
        w["render_asked"] = w.get("render_asked") or w.get("render")
        out.append(f"type.words[{i}] {w.get('text')!r}: render {w.get('render')!r} -> 'flat': {why}")
        w["render"] = "flat"
        w["ltx_missing"] = True
        for k in ("ltx_at", "ltx_t", "material"):
            w.pop(k, None)
    return out


def choose_background(piece: dict, sid: str, clip: dict, matte=None, type_dir=None) -> tuple[dict, list[str]]:
    """-> (clip to compose, notes). A layered clip whose text-rendered background cannot stay (see above) comes back with u8 = the clean clip and textbg False; the shot's
    remaining LTX words are made flat in place (the letters they would hand over to are not in the clean clip). Everything is logged into piece.plan.owner.background."""
    if clip.get("subject") is None or not clip.get("textbg"):
        return clip, []
    from tools.fusion import matte_align as MA
    notes: list[str] = []
    left = shot_ltx_words(piece, sid, type_dir)
    why = None
    drift = None
    if not left:
        why = "no LTX-owned word is left in the shot (all flat): the text pass may hold a half-formed word, so the clean clip is the background"
    elif (piece.get("layers") or {}).get("drift_check") is not False:
        drift = MA.subject_drift(clip["subject"], clip["u8"], matte) if matte is not None else None
        if drift is not None and drift["ok"] is False:
            why = (f"the subject of the text-rendered clip is {drift['px']:g} px ({drift['rel']:.1%} of the short side, limit {MA.DRIFT_MAX:.1%}) off the clean clip the matte was cut from: "
                   f"it would show two people, so the clean clip is the background")
        elif drift is None or drift["ok"] is None:       # nothing to verify against (no matte / no subject / a weak match): the text clip is not trusted as the picture under the type
            why = "the subject of the text-rendered clip could not be checked against the clean clip (no matte or a weak match), so the clean clip is the background (the text pass is used only when its word passed)"
    po = piece.setdefault("plan", {}).setdefault("owner", {})
    row = dict(shot=sid, kept_textbg=why is None, ltx_words=left, drift=drift)
    if why is None:
        po.setdefault("background", []).append(row)
        return clip, notes
    notes.append(f"background {sid}: {why}")
    notes += demote_shot_words(piece, left, "the text-rendered background was dropped for this shot, so there are no LTX letters to hand over to")
    row["why"] = why
    po.setdefault("background", []).append(row)
    po.setdefault("fallback", []).extend(notes)
    piece["plan"].setdefault("notes", []).extend(notes)
    return dict(clip, u8=clip["subject"], textbg=False), notes


def prepare_backgrounds(piece: dict, type_dir, clips: list[dict], mattes: dict | None = None, log=print) -> list[dict]:
    """the layered clips of a piece, ready for compose (nodes_fusion/compose.py calls this before any shot is composed). Pass 1: every shot with a text-rendered clip is judged
    (apply_fallback: reading + pixel + layout; a word can span shots: HOLD of v005 r01 sat in s3, s5 and s6, so no shot is decided before ALL words are). Pass 2: choose_background per shot:
    the text clip stays the picture under the type only where an LTX word passed AND its subject was verified to sit on the clean clip's. Changes `piece` in place (plan.owner), logs through `log`.
    -> the clips to compose (same order)."""
    for clip in clips:
        if clip.get("subject") is not None and clip.get("textbg"):
            for m_ in apply_fallback(piece, clip["sid"], type_dir, clip["subject"], clip["u8"], reads=clip.get("ltx_reads")):
                log(f"[MographCompose] {m_}")
    out = []
    for clip in clips:
        if clip.get("subject") is not None and clip.get("textbg"):
            clip, notes = choose_background(piece, clip["sid"], clip, (mattes or {}).get(clip["sid"]), type_dir)
            for m_ in notes:
                log(f"[MographCompose] {m_}")
        out.append(clip)
    return out
