"""Stage cache of the MOTNGRFX app: a finished stage is stored once under a CONTENT-HASH key and reused (as NEW files in the new job folder) when its inputs have not changed.

    <ComfyUI root>/motngrfx_cache/<stage>/<key>/manifest.json + files/<root>/<relative name>      (env MOTNGRFX_CACHE_DIR overrides the folder)

The cache sits BESIDE output/ (same filesystem, so hardlinks work), never inside it: the dashboard Outputs sync must not see cached duplicates.

key = sha256 of (stage name, CACHE_VERSION, the content hashes of the input files, ids of earlier stages, hashes of input texts, canonical JSON of the relevant parameters).
Nothing about a user's pick that a stage does not read may enter its key (the matte key has no words / fonts / seed; fonts reach a key only through a design graph that
carries a user-pinned font, never through the page's random default faces).

Rules (user 2026-11-01, "reuse them if they haven't changed"):
  * reuse NEVER overwrites or deletes anything: `JobCache.restore` links (hardlink, copy as fallback) the cached files into the new job folder under the new job's names and refuses
    to touch a file that is already there (a refused / failed restore removes only what it just made and reports a miss);
  * `put` stores hardlinks of the finished files (copy as fallback), written to a temp folder and renamed in one step (a half-written entry is never visible);
  * the job id lives in many names / JSON texts (STEM `motngrfx_own_<id>`, the ComfyUI-relative output prefix, the absolute job folder): they are swapped for tokens when stored and
    back for the new job when restored, in file names and in text files (.json / .txt / .sh);
  * env MOTNGRFX_NO_CACHE=1 disables get and put (every stage computes).

Used by tools/app/own_video_job.py (every stage) and nodes_fusion/node_cache.py (image-mode LTX shot / Alpha-Gen / Gemma nodes). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

CACHE_DIR_NAME = "motngrfx_cache"
CACHE_VERSION = 1                     # bump to retire every entry (a change in how a stage's files are made that its key cannot see)
TEXT_SUFFIXES = (".json", ".txt", ".sh")
MAX_TEXT_BYTES = 64 * 1024 * 1024
_OFF = ("1", "true", "yes", "on")


def enabled() -> bool:
    """False while MOTNGRFX_NO_CACHE is 1 / true / yes / on."""
    return os.environ.get("MOTNGRFX_NO_CACHE", "").strip().lower() not in _OFF


def default_cache_root(comfy_root) -> Path:
    """env MOTNGRFX_CACHE_DIR, else <ComfyUI root>/motngrfx_cache (next to output/ and input/)."""
    env = os.environ.get("MOTNGRFX_CACHE_DIR", "").strip()
    return Path(env) if env else Path(comfy_root) / CACHE_DIR_NAME


def canonical(obj) -> str:
    """stable JSON (sorted keys, no spaces) of anything JSON-like; unknown objects by str()."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def text_hash(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


_FILE_MEMO: dict = {}
MEMO_MIN_BYTES = 32 * 1024 * 1024


def file_hash(path) -> str:
    """sha256 of the file's CONTENT (files of 32 MB or more are memoised per process on path + size + mtime + ctime + inode)."""
    p = Path(path)
    st = p.stat()
    mk = (str(p), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino) if st.st_size >= MEMO_MIN_BYTES else None          # small files are just hashed again (ms): a same-size rewrite in one clock tick can never return a stale hash
    h = _FILE_MEMO.get(mk) if mk else None
    if h is None:
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                d.update(chunk)
        h = d.hexdigest()
        if mk:
            _FILE_MEMO[mk] = h
    return h


_CODE_MEMO: dict = {}


def code_fingerprint(root, patterns) -> str:
    """sha256 over (relative path, content) of every file matching the glob `patterns` under `root` (memoised per process): a stage whose output depends on the repo's code puts this
    in its key, so an edit to that code retires the entry."""
    root = Path(root)
    mk = (str(root), tuple(patterns))
    if mk not in _CODE_MEMO:
        d = hashlib.sha256()
        files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts})
        for p in files:
            d.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
            d.update(p.read_bytes() + b"\0")
        _CODE_MEMO[mk] = d.hexdigest()
    return _CODE_MEMO[mk]


def make_key(stage: str, files=(), ids=(), texts=(), params=None, version: int = CACHE_VERSION) -> str:
    """the cache key: files = input paths (content hash), ids = hashes / keys of earlier stages, texts = input texts (hash), params = canonical JSON."""
    payload = dict(stage=str(stage), v=int(version), files=[file_hash(f) for f in files], ids=[str(i) for i in ids], texts=[text_hash(t) for t in texts], params=params)
    return text_hash(canonical(payload))


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _expand(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            out.append(p)
    return out


class StageCache:
    """the store: <root>/<stage>/<key>/{manifest.json, files/<root name>/<relative name>}."""

    def __init__(self, root):
        self.root = Path(root)

    def entry(self, stage: str, key: str) -> Path:
        return self.root / stage / key

    def manifest(self, stage: str, key: str) -> dict | None:
        try:
            m = json.loads((self.entry(stage, key) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return m if m.get("version") == CACHE_VERSION and m.get("key") == key else None

    def has(self, stage: str, key: str) -> bool:
        return self.manifest(stage, key) is not None

    @staticmethod
    def _tok(s: str, subst) -> str:
        for real, tok in subst:
            if real:
                s = s.replace(real, tok)
        return s

    @staticmethod
    def _untok(s: str, subst) -> str:
        for real, tok in subst:
            if real:
                s = s.replace(tok, real)
        return s

    def put(self, stage: str, key: str, entries, subst=(), inputs=None, meta=None) -> bool:
        """entries = [(root name, relative path under that root, absolute path)]. Atomic; an existing entry is left as it is. False if nothing was stored."""
        final = self.entry(stage, key)
        if (final / "manifest.json").is_file():
            return False
        tmp = self.root / stage / f".tmp-{key[:12]}-{uuid.uuid4().hex[:8]}"
        files = []
        try:
            for rname, rel, src in entries:
                src = Path(src)
                if not src.is_file():
                    continue
                rel_t = self._tok(str(rel).replace("\\", "/"), subst)
                dst = tmp / "files" / rname / rel_t
                text = src.suffix.lower() in TEXT_SUFFIXES and src.stat().st_size <= MAX_TEXT_BYTES
                if text:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(self._tok(src.read_bytes().decode("utf-8", "surrogateescape"), subst).encode("utf-8", "surrogateescape"))
                else:
                    _link(src, dst)
                files.append(dict(root=rname, rel=rel_t, text=bool(text)))
            if not files:
                return False
            (tmp / "manifest.json").write_text(json.dumps(dict(version=CACHE_VERSION, stage=stage, key=key, created=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs=inputs or {}, meta=meta or {},
                                                               files=files), ensure_ascii=False, indent=1), encoding="utf-8")
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(tmp, final)
            except OSError:                                   # a parallel writer won the rename: its entry stands
                return False
            return True
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def put_meta(self, stage: str, key: str, meta: dict, inputs=None) -> bool:
        """an entry with no files, only `meta` (a node's text outputs). Atomic; an existing entry stands."""
        final = self.entry(stage, key)
        if (final / "manifest.json").is_file():
            return False
        tmp = self.root / stage / f".tmp-{key[:12]}-{uuid.uuid4().hex[:8]}"
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "manifest.json").write_text(json.dumps(dict(version=CACHE_VERSION, stage=stage, key=key, created=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs=inputs or {}, meta=meta,
                                                               files=[]), ensure_ascii=False, indent=1), encoding="utf-8")
            try:
                os.rename(tmp, final)
            except OSError:
                return False
            return True
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def get_meta(self, stage: str, key: str) -> dict | None:
        m = self.manifest(stage, key)
        return None if m is None else (m.get("meta") or {})

    def get(self, stage: str, key: str, roots: dict, subst=()) -> dict | None:
        """link the cached files into `roots` (name -> folder) as NEW files; returns the manifest, or None (miss, or a file in the way / missing: what was made is removed again)."""
        m = self.manifest(stage, key)
        if m is None:
            return None
        made: list[Path] = []
        try:
            for f in m["files"]:
                src = self.entry(stage, key) / "files" / f["root"] / f["rel"]
                root = roots.get(f["root"])
                if root is None or not src.is_file():
                    raise FileNotFoundError(str(src))
                dst = Path(root) / self._untok(f["rel"], subst)
                if dst.exists():
                    raise FileExistsError(str(dst))
                if f.get("text"):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(self._untok(src.read_bytes().decode("utf-8", "surrogateescape"), subst).encode("utf-8", "surrogateescape"))
                else:
                    _link(src, dst)
                made.append(dst)
            return m
        except OSError:
            for p in made:
                try:
                    p.unlink()
                except OSError:
                    pass
            return None


class JobCache:
    """a StageCache bound to ONE job: the token swap of its ids / paths, the two file roots (the job folder = "out", the ComfyUI input folder = "in") and the log hooks."""

    def __init__(self, cache_root, out_dir, input_dir, job_id: str, stem: str, rel: str, log=None, mark=None, on: bool | None = None):
        self.on = enabled() if on is None else bool(on)
        self.store_ = StageCache(cache_root)
        self.out_dir, self.input_dir = Path(out_dir), Path(input_dir)
        self.roots = {"out": self.out_dir, "in": self.input_dir}
        pairs = [(self.out_dir.as_posix(), "@OUT@"), (self.out_dir.resolve().as_posix(), "@OUT@"), (str(rel).replace("\\", "/"), "@REL@"), (str(stem), "@STEM@"), (str(job_id), "@JOB@")]
        self.subst = sorted([p for p in pairs if p[0]], key=lambda p: -len(p[0]))
        self.log = log or (lambda m: None)
        self.mark = mark or (lambda stage: None)
        self.hits: dict[str, str] = {}

    # ------------------------------------------------------------------------------------------------ keys
    def norm(self, s: str) -> str:
        """a text with this job's paths / ids swapped for tokens (so two jobs that differ only by id hash the same)."""
        return self.store_._tok(str(s), self.subst)

    def norm_json(self, obj) -> str:
        return self.norm(canonical(obj))

    def key(self, stage: str, files=(), ids=(), texts=(), params=None) -> str:
        """the stage's key ("" while the cache is off: nothing is hashed then)."""
        if not self.on:
            return ""
        return make_key(stage, files=files, ids=ids, texts=[self.norm(t) for t in texts], params=json.loads(self.norm_json(params)) if params is not None else None)

    def file_text_hash(self, path) -> str:
        """hash of a small text file with the job tokens swapped in (a JSON that names the job's own files)."""
        return text_hash(self.norm(Path(path).read_text(encoding="utf-8", errors="replace")))

    # ------------------------------------------------------------------------------------------------ restore / store
    def restore(self, stage: str, key: str, mark: bool = True) -> dict | None:
        """a hit links the cached files into the job and returns the manifest (meta in `["meta"]`); a miss (or a disabled cache) returns None. mark=False: the caller tags the stage itself."""
        if not self.on or not key:
            return None
        m = self.store_.get(stage, key, self.roots, self.subst)
        if m is None:
            return None
        self.hits[stage] = key
        self.log(f"cache hit {stage} {key[:10]} ({len(m['files'])} files reused)")
        if mark:
            self.mark(stage)
        return m

    def entries(self, paths) -> list:
        out = []
        for p in _expand(paths):
            for rname, root in self.roots.items():
                try:
                    out.append((rname, p.resolve().relative_to(root.resolve()).as_posix(), p))
                    break
                except ValueError:
                    continue
        return out

    def store(self, stage: str, key: str, paths, meta=None, inputs=None) -> bool:
        if not self.on or not key:
            return False
        try:
            ok = self.store_.put(stage, key, self.entries(paths), self.subst, inputs=inputs, meta=meta)
        except OSError as e:                                  # a full disk / a vanished file must never fail the job
            self.log(f"cache store {stage} skipped: {e}")
            return False
        if ok:
            self.log(f"cache stored {stage} {key[:10]}")
        return ok
