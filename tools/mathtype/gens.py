"""Generation naming and lineage log for the 11_mathtype renders (user naming rule).

Every render gets a unique numbered name `<stem>_g<NN>_<draft|final>[_chords].mp4`; a draft and its
final from the same code share NN. The code state is identified by a hash of the script files; a
new hash means a new generation. Lineage goes into `local/GENERATIONS.md` as a markdown table
(gen | time NZ | test | parent | trigger | change | files | still), one row per generation, upserted.
Per-test state lives in `local/.gens_<test>.json`. Nothing is ever overwritten or deleted.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re

try:
    from zoneinfo import ZoneInfo
    _NZ = ZoneInfo("Pacific/Auckland")
except Exception:  # pragma: no cover
    _NZ = None

HEADER = ("| gen | time NZ | test | parent | trigger | change | files | still |\n"
          "|---|---|---|---|---|---|---|---|\n")


def code_hash(paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _state_path(local, test):
    return os.path.join(local, f".gens_{test}.json")


def _load(local, test) -> dict:
    p = _state_path(local, test)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(local, test, st):
    with open(_state_path(local, test), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


def gen_floor() -> int:
    """Lowest NN a NEW generation may take on this host (0 = no floor).

    Box and local number independently (each sees only its own local/ dir), so they used to
    clash. Rule: the GPU box sets a floor of 100 (mathtype-gpu card writes ~/.mathtype_gen_floor),
    so box generations are g100+ and local ones stay below; names keep the `_g<NN>_` form.
    Env MATHTYPE_GEN_FLOOR overrides the file. Reusing an existing generation is unaffected."""
    raw = os.environ.get("MATHTYPE_GEN_FLOOR")
    if raw is None:
        try:
            with open(os.path.expanduser("~/.mathtype_gen_floor"), encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return 0
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


def video_name(stem, nn, kind, chords=False):
    return f"{stem}_g{nn:02d}_{kind}{'_chords' if chords else ''}.mp4"


def gen_for(local, test, stem, kind, h, parent=None, trigger="", change="") -> int:
    """NN for a render of `kind` (draft/final) with code hash h: reuse the latest generation if its
    hash matches and that kind is not written yet, otherwise open a new generation
    max(max + 1, gen_floor())."""
    os.makedirs(local, exist_ok=True)
    st = _load(local, test)
    if st:
        last = max(st, key=int)
        if st[last]["hash"] == h and not os.path.exists(os.path.join(local, video_name(stem, int(last), kind))):
            return int(last)
    nums = [int(k) for k in st]
    for f in os.listdir(local):
        m = re.match(re.escape(stem) + r"_g(\d+)_", f)
        if m:
            nums.append(int(m.group(1)))
    nn = max(max(nums + [0]) + 1, gen_floor())
    while os.path.exists(os.path.join(local, video_name(stem, nn, kind))):
        nn += 1
    prev = max([int(k) for k in st] + [0]) if st else 0
    st[str(nn)] = dict(hash=h, time=now_nz(), parent=parent or f"{test} g{prev:02d}", trigger=trigger,
                       change=change, files=[], stills=[])
    _save(local, test, st)
    upsert_row(local, test, nn)
    return nn


def add_files(local, test, nn, files=(), stills=()):
    st = _load(local, test)
    e = st[str(nn)]
    for f in files:
        if f not in e["files"]:
            e["files"].append(f)
    for s in stills:
        if s not in e["stills"]:
            e["stills"].append(s)
    _save(local, test, st)
    upsert_row(local, test, nn)


def backfill(local, test, nn, time, parent, trigger, change, files, stills):
    """Record an earlier (unnumbered = g00) generation."""
    st = _load(local, test)
    st[str(nn)] = dict(hash="", time=time, parent=parent, trigger=trigger, change=change,
                       files=list(files), stills=list(stills))
    _save(local, test, st)
    upsert_row(local, test, nn)


def now_nz() -> str:
    t = datetime.datetime.now(_NZ) if _NZ else datetime.datetime.now()
    return t.strftime("%H:%M")


def upsert_row(local, test, nn):
    e = _load(local, test)[str(nn)]
    cells = [f"g{nn:02d}", e["time"], test, e["parent"], e["trigger"], e["change"],
             ", ".join(e["files"]), ", ".join(e["stills"])]
    row = "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |\n"
    path = os.path.join(local, "GENERATIONS.md")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    if not any(ln.startswith("| gen |") for ln in lines):
        lines = lines + (["\n"] if lines and lines[-1].strip() else []) + HEADER.splitlines(keepends=True)
    key = re.compile(rf"^\| g{nn:02d} \| [^|]* \| {re.escape(test)} \|")
    for i, ln in enumerate(lines):
        if key.match(ln):
            lines[i] = row
            break
    else:
        lines.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
