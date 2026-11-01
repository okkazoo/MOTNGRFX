"""generations: the lineage of one piece across Refine / redesign passes. Pure Python, no ComfyUI. Docs: 16_app/REFINE_PLAN.md.

One ``generations.json`` per store (a PROJECTS version folder, or the app's own folder under ComfyUI's output dir). It holds every piece of that
store, and for each piece the list of its generations:

    {"schema": 1,
     "pieces": {"r03": {"generations": [
         {"gen": "g01", "parent": null,  "kind": "initial", "status": "done", ...},
         {"gen": "g02", "parent": "g01", "kind": "refine",  "status": "done", "arm": "design_llm_8b", "note": "...", "critique": "...",
          "edits": [...], "salvage": {...}, "final": "out/..._r03_g02_video_00001_.mp4", "archive": "out/..._r03_g02_out_shots", ...}]}}}

Rules (the user's "never overwrite renders; gNN names + lineage"):
  * a generation number is allocated by ``reserve`` BEFORE the render starts and is never reused, not even by a failed or interrupted one;
  * ``complete`` fills a pending entry once; a ``done`` entry is never changed again (a second ``complete`` raises);
  * the file prefixes handed out by ``reserve`` carry the gNN, so a refine can never write over an earlier generation's mp4 / archive folder;
  * every path is stored relative to the store's ``base`` (the version folder, or ComfyUI's output dir), with forward slashes.

PROJECTS naming (``~/.claude/skills/project/proj.py``, ``progress_standin.py submit --idea ... --part LABEL``): the part label of generation gNN of
piece ``r03`` is ``r03_g02``; progress_standin re-points the save nodes to ``<stem>_<part>_<role>`` so the files are
``<project>_<idea>_<vNNN>_r03_g02_video_00001_.mp4`` / ``..._r03_g02_out_shots/`` next to the earlier ``..._r03_...`` ones.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = 1
FILE = "generations.json"
KINDS = ("initial", "refine", "redesign")
STATUS = ("pending", "done", "failed")
GEN_RE = re.compile(r"^g(\d{2,})$")
_LOCK = threading.RLock()


class GenerationError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_nz() -> str:
    """NZDT (UTC+13) like the rest of the repo's notes; the date is what matters."""
    return (datetime.now(timezone.utc) + timedelta(hours=13)).strftime("%Y-%m-%d %H:%M NZDT")


def gen_name(n: int) -> str:
    return f"g{n:02d}"


def gen_number(name: str) -> int:
    m = GEN_RE.match(str(name or ""))
    if not m:
        raise GenerationError(f"bad generation id {name!r} (expected gNN)")
    return int(m.group(1))


def slug(text: str, n: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s[:n].strip("-") or "piece"


def new_piece_key(words: str = "", t: float | None = None) -> str:
    """a key for a piece made in the app: p<yymmdd-hhmmss>-<first words>."""
    tm = time.localtime(t if t is not None else time.time())
    return f"p{time.strftime('%y%m%d-%H%M%S', tm)}-{slug(words, 16)}"


def part_label(piece: str, gen: str) -> str:
    """the --part label of progress_standin / proj.py: r03 + g02 -> r03_g02."""
    gen_number(gen)
    return f"{piece}_{gen}"


def rel(base: Path, p: str | Path | None) -> str | None:
    """a path under ``base`` as 'a/b/c' (forward slashes); a path outside base is kept as given."""
    if p is None or p == "":
        return None
    q = Path(p)
    try:
        return q.resolve().relative_to(Path(base).resolve()).as_posix() if q.is_absolute() else q.as_posix()
    except ValueError:
        return q.as_posix()


class Store:
    """``generations.json`` of one version folder (or of the app output folder). ``base`` resolves the stored relative paths."""

    def __init__(self, base: str | Path, file: str | Path | None = None):
        self.base = Path(base)
        self.path = Path(file) if file else self.base / FILE

    # ------------------------------------------------------------------ io
    def load(self) -> dict:
        if not self.path.is_file():
            return {"schema": SCHEMA, "pieces": {}}
        d = json.loads(self.path.read_text(encoding="utf-8"))
        d.setdefault("schema", SCHEMA)
        d.setdefault("pieces", {})
        return d

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".gen_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)           # atomic: a reader never sees half a file
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    @contextlib.contextmanager
    def _txn(self):
        with _LOCK:
            d = self.load()
            yield d
            self._save(d)

    # ------------------------------------------------------------------ read
    def pieces(self) -> list[str]:
        return list(self.load()["pieces"])

    def generations(self, piece: str) -> list[dict]:
        return list((self.load()["pieces"].get(piece) or {}).get("generations") or [])

    def get(self, piece: str, gen: str) -> dict | None:
        return next((g for g in self.generations(piece) if g["gen"] == gen), None)

    def lineage(self, piece: str, gen: str) -> list[str]:
        """gen and its ancestors, root first."""
        by = {g["gen"]: g for g in self.generations(piece)}
        out, cur = [], gen
        while cur:
            if cur not in by or cur in out:
                raise GenerationError(f"{piece}: broken lineage at {cur!r}")
            out.append(cur)
            cur = by[cur].get("parent")
        return out[::-1]

    def children(self, piece: str, gen: str) -> list[str]:
        return [g["gen"] for g in self.generations(piece) if g.get("parent") == gen]

    # ------------------------------------------------------------------ write
    def reserve(self, piece: str, kind: str, parent: str | None = None, *, arm: str = "", note: str = "", prefix_root: str = "",
                **extra) -> dict:
        """allocate the next gNN of ``piece`` (status pending) and hand out the file prefixes to render with. Never reuses a number."""
        if kind not in KINDS:
            raise GenerationError(f"bad kind {kind!r} (one of {', '.join(KINDS)})")
        if not piece or "/" in piece or "\\" in piece or piece.startswith("."):
            raise GenerationError(f"bad piece key {piece!r}")
        if (kind == "initial") != (parent is None):
            raise GenerationError("an initial generation has no parent; a refine / redesign needs one")
        with self._txn() as d:
            P = d["pieces"].setdefault(piece, {"generations": []})
            gens = P["generations"]
            if parent is not None and parent not in {g["gen"] for g in gens}:
                raise GenerationError(f"{piece}: parent {parent!r} does not exist")
            if parent is not None and next(g for g in gens if g["gen"] == parent)["status"] != "done":
                raise GenerationError(f"{piece}: parent {parent!r} is not finished")
            n = max([gen_number(g["gen"]) for g in gens] + [0]) + 1
            gen = gen_name(n)
            root = (prefix_root.strip("/") + "/") if prefix_root else ""
            part = part_label(piece, gen)
            e = {"gen": gen, "parent": parent, "kind": kind, "status": "pending", "arm": arm, "note": note, "critique": "", "edits": [],
                 "salvage": {}, "final": None, "archive": None, "part": part,
                 "prefix_video": f"{root}{part}_video", "prefix_archive": f"{root}{part}_out",
                 "created": now_iso(), "created_nz": now_nz(), "completed": None, **extra}
            gens.append(e)
            return json.loads(json.dumps(e))

    def complete(self, piece: str, gen: str, **fields) -> dict:
        """fill a pending entry (final, archive, critique, edits, salvage, seconds, prompt_id ...) and mark it done. A done entry is immutable."""
        with self._txn() as d:
            e = self._entry(d, piece, gen)
            if e["status"] == "done":
                raise GenerationError(f"{piece}/{gen} is already complete; generations are never overwritten")
            for k in ("gen", "parent", "kind", "created", "part"):
                fields.pop(k, None)
            for k in ("final", "archive"):
                if k in fields:
                    fields[k] = rel(self.base, fields[k])
            e.update(fields)
            e["status"] = "done"
            e["completed"] = now_iso()
            return json.loads(json.dumps(e))

    def fail(self, piece: str, gen: str, error: str = "") -> dict:
        with self._txn() as d:
            e = self._entry(d, piece, gen)
            if e["status"] == "done":
                raise GenerationError(f"{piece}/{gen} is already complete")
            e["status"], e["error"], e["completed"] = "failed", str(error)[:500], now_iso()
            return json.loads(json.dumps(e))

    def register_existing(self, piece: str, final: str | Path, archive: str | Path | None = None, *, kind: str = "initial", parent: str | None = None,
                          **fields) -> dict:
        """add a render that already exists (e.g. the batch run r03) as a done generation. Idempotent: the same ``final`` is returned, not duplicated."""
        fin, arc = rel(self.base, final), rel(self.base, archive)
        for g in self.generations(piece):
            if g.get("final") == fin:
                return g
        e = self.reserve(piece, kind, parent, **{k: v for k, v in fields.items() if k in ("arm", "note", "prefix_root")})
        extra = {k: v for k, v in fields.items() if k not in ("arm", "note", "prefix_root")}
        return self.complete(piece, e["gen"], final=fin, archive=arc, **extra)

    @staticmethod
    def _entry(d: dict, piece: str, gen: str) -> dict:
        P = d["pieces"].get(piece)
        e = next((g for g in (P or {}).get("generations") or [] if g["gen"] == gen), None)
        if e is None:
            raise GenerationError(f"{piece}/{gen}: no such generation")
        return e

    # ------------------------------------------------------------------ files
    def files_of(self, entry: dict) -> dict:
        """the on-disk state of one entry: does the final mp4 / the archive piece.json exist."""
        fin = self.base / entry["final"] if entry.get("final") else None
        arc = self.base / entry["archive"] if entry.get("archive") else None
        return {"final": fin, "final_exists": bool(fin and fin.is_file()), "archive": arc, "piece_json": (arc / "piece.json") if arc else None,
                "archive_exists": bool(arc and (arc / "piece.json").is_file())}

    def find_output(self, prefix: str, suffix: str = ".mp4") -> str | None:
        """the newest ``<prefix>_NNNNN_<suffix>`` ComfyUI wrote (SaveVideo appends a counter), relative to base; None if not there (yet)."""
        pre = Path(prefix)
        folder = self.base / pre.parent
        if not folder.is_dir():
            return None
        c = sorted((p for p in folder.glob(pre.name + "*" + suffix) if p.is_file()), key=lambda p: p.stat().st_mtime)
        return rel(self.base, c[-1]) if c else None


# ---------------------------------------------------------------------------------------------- CLI
def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="generations.json of a version folder / app output folder (REFINE_PLAN.md)")
    ap.add_argument("base", help="the version folder (PROJECTS) or ComfyUI's output dir")
    ap.add_argument("--file", help="generations.json location (default <base>/generations.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("register", help="add an existing render as a done generation (idempotent)")
    s.add_argument("piece"); s.add_argument("final"); s.add_argument("--archive"); s.add_argument("--kind", default="initial", choices=KINDS)
    s.add_argument("--parent"); s.add_argument("--note", default=""); s.add_argument("--arm", default="")
    s = sub.add_parser("reserve", help="allocate the next gNN and print the part label / prefixes to render with")
    s.add_argument("piece"); s.add_argument("--kind", default="refine", choices=KINDS); s.add_argument("--parent", required=True)
    s.add_argument("--note", default=""); s.add_argument("--arm", default="design_llm_8b")
    s = sub.add_parser("complete")
    s.add_argument("piece"); s.add_argument("gen"); s.add_argument("--final"); s.add_argument("--archive"); s.add_argument("--critique", default="")
    s.add_argument("--edits", help="JSON list"); s.add_argument("--salvage", help="JSON object")
    a = ap.parse_args(argv)
    st = Store(a.base, a.file)
    if a.cmd == "list":
        for p in st.pieces():
            print(p)
            for g in st.generations(p):
                print(f"  {g['gen']} <- {g.get('parent') or '-':4} {g['kind']:8} {g['status']:7} {g.get('final') or ''}")
    elif a.cmd == "register":
        print(json.dumps(st.register_existing(a.piece, a.final, a.archive, kind=a.kind, parent=a.parent, note=a.note, arm=a.arm), indent=1))
    elif a.cmd == "reserve":
        print(json.dumps(st.reserve(a.piece, a.kind, a.parent, arm=a.arm, note=a.note), indent=1))
    elif a.cmd == "complete":
        f = {k: v for k, v in dict(final=a.final, archive=a.archive, critique=a.critique).items() if v}
        if a.edits:
            f["edits"] = json.loads(a.edits)
        if a.salvage:
            f["salvage"] = json.loads(a.salvage)
        print(json.dumps(st.complete(a.piece, a.gen, **f), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
