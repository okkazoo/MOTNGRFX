"""box_ssh: persistent ssh sessions to the rented GPU box for piece_runner (16_app/BOX_OVERHEAD.md).

Why: every `ssh host cmd` / `scp` from Windows costs a full handshake (about 4 s against the vast.ai proxy), and ControlMaster does not
work from either Windows ssh (Git's MSYS ssh fails with "mux_client_request_session: send fds failed", System32 OpenSSH has no mux). So one
long-lived `ssh host bash -s` carries many commands: a call is one round trip (a few hundred ms) instead of a handshake.

Protocol (one session = one remote bash reading commands from stdin, strictly one call at a time):
    client -> bash   one line: runs the command (base64, so any quoting survives) in a SUBSHELL with stdout / stderr to temp files, then
                     prints `@@END <rc> <len stdout> <len stderr>` and cats both files (binary safe, byte counts).
    put (data)       the command reads its stdin (`head -c N | tar x`), the N payload bytes follow the command line on the same pipe.
Each call runs in a fresh subshell, so cwd / env never leak between calls (same as separate ssh invocations).
A SessionPool hands out an idle session or opens another (parallel stages each get their own); a dead session is replaced and the
call retried once. Everything raises BoxSSHError when ssh itself is unusable, and the caller falls back to plain ssh / scp."""
from __future__ import annotations

import base64
import json
import io
import queue
import shutil
import subprocess
import tarfile
import threading
import time


class BoxSSHError(RuntimeError):
    pass


class Session:
    READY = b"@@READY"

    def __init__(self, argv, start_timeout=60.0):
        self.argv = list(argv)
        self.dead = False
        self.q: "queue.Queue[bytes | None]" = queue.Queue()
        try:
            self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as e:
            raise BoxSSHError(f"cannot start ssh: {e}") from e
        threading.Thread(target=self._reader, daemon=True, name="boxssh-reader").start()
        self.buf = b""
        self._write(b"echo @@READY\n")
        try:
            self._until(self.READY + b"\n", start_timeout)
        except BoxSSHError:
            self.close()
            raise
        self.buf = b""

    # ---- plumbing
    def _reader(self):
        try:
            while True:
                b = self.p.stdout.read(1 << 20)
                if not b:
                    break
                self.q.put(b)
        except Exception:      # noqa: BLE001
            pass
        self.q.put(None)

    def _write(self, data: bytes):
        try:
            self.p.stdin.write(data)
            self.p.stdin.flush()
        except (OSError, ValueError) as e:
            self.dead = True
            raise BoxSSHError(f"ssh session write failed: {e}") from e

    def _fill(self, deadline):
        left = deadline - time.time()
        if left <= 0:
            self.close()
            raise subprocess.TimeoutExpired("box ssh call", 0)
        try:
            b = self.q.get(timeout=left)
        except queue.Empty:
            self.close()
            raise subprocess.TimeoutExpired("box ssh call", 0)
        if b is None:
            self.dead = True
            raise BoxSSHError("ssh session closed by the remote side")
        self.buf += b

    def _until(self, marker: bytes, timeout):
        deadline = time.time() + timeout
        while marker not in self.buf:
            self._fill(deadline)
        i = self.buf.index(marker) + len(marker)
        out, self.buf = self.buf[:i], self.buf[i:]
        return out

    def alive(self):
        return not self.dead and self.p.poll() is None

    def close(self):
        self.dead = True
        try:
            self.p.stdin.close()
        except Exception:      # noqa: BLE001
            pass
        try:
            self.p.kill()
        except Exception:      # noqa: BLE001
            pass

    # ---- one call
    def call(self, cmd: str, timeout=120.0, data: bytes | None = None):
        """-> (rc, stdout bytes, stderr bytes). data: bytes fed to the command's stdin (None: stdin is /dev/null)."""
        if not self.alive():
            raise BoxSSHError("ssh session is dead")
        b64 = base64.b64encode(cmd.encode("utf-8")).decode()
        redir = "" if data is not None else " </dev/null"
        line = (f'__o=$(mktemp); __e=$(mktemp); ( eval "$(printf %s {b64} | base64 -d)" ) >"$__o" 2>"$__e"{redir}; __r=$?; '
                f'echo "@@END $__r $(stat -c%s "$__o") $(stat -c%s "$__e")"; cat "$__o" "$__e"; rm -f "$__o" "$__e"\n')
        self.buf = b""
        deadline = time.time() + timeout
        self._write(line.encode() + (data if data is not None else b""))
        while b"\n" not in self.buf or b"@@END " not in self.buf:
            self._fill(deadline)
        i = self.buf.index(b"@@END ")
        j = self.buf.index(b"\n", i)
        try:
            _, rc, so, se = self.buf[i:j].decode().split()
            rc, so, se = int(rc), int(so), int(se)
        except ValueError as e:
            self.close()
            raise BoxSSHError(f"bad ssh session framing: {self.buf[:200]!r}") from e
        need = j + 1 + so + se
        while len(self.buf) < need:
            self._fill(deadline)
        body = self.buf[j + 1:need]
        self.buf = self.buf[need:]
        return rc, body[:so], body[so:]


class SessionPool:
    def __init__(self, argv, max_sessions=8):
        self.argv, self.max = list(argv), max_sessions
        self.idle: list[Session] = []
        self.all: list[Session] = []
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(max_sessions)
        self.broken = False         # ssh itself unusable: callers use plain ssh / scp

    def _acquire(self):
        self.sem.acquire()
        try:
            while True:
                with self.lock:
                    s = self.idle.pop() if self.idle else None
                if s is None:
                    break
                if s.alive():
                    return s
            s = Session(self.argv)
            with self.lock:
                self.all.append(s)
            return s
        except BaseException:
            self.sem.release()
            raise

    def _release(self, s):
        with self.lock:
            if s.alive():
                self.idle.append(s)
            elif s in self.all:
                self.all.remove(s)
        self.sem.release()

    def call(self, cmd, timeout=120.0, data=None):
        last = None
        for attempt in range(2):
            s = self._acquire()
            try:
                return s.call(cmd, timeout, data)
            except BoxSSHError as e:
                last = e
                s.close()
            finally:
                self._release(s)
        raise BoxSSHError(f"ssh session failed twice: {last}")

    def warm(self):
        s = self._acquire()
        self._release(s)

    def close(self):
        with self.lock:
            ss, self.all, self.idle = list(self.all), [], []
        for s in ss:
            s.close()


# ---------------------------------------------------------------------------------------------------- tar helpers
def tar_bytes(items, level=1) -> bytes:
    """items: [(arcname, local_path | bytes)] -> .tar.gz bytes (gzip level `level`; 0 = plain tar)."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz" if level else "w", **({"compresslevel": level} if level else {})) as t:
        for arc, src in items:
            if isinstance(src, (bytes, bytearray)):
                ti = tarfile.TarInfo(arc)
                ti.size, ti.mtime, ti.mode = len(src), time.time(), 0o644
                t.addfile(ti, io.BytesIO(bytes(src)))
            else:
                t.add(str(src), arcname=arc)
    return bio.getvalue()


def untar_to(data: bytes, pairs):
    """pairs: [(arcname, local_dst)]: extract those members of a plain tar to the given local paths."""
    import os
    want = dict(pairs)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            dst = want.get(m.name)
            if dst is None or not m.isfile():
                continue
            os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
            with open(dst, "wb") as f:
                shutil.copyfileobj(t.extractfile(m), f)


# ---------------------------------------------------------------------------------------------------- plates on the box
# The TYPE stage builds the contour ports (mograph/modules/contour_ports.py) from the shot plates / depth / mattes named in
# piece["type"]["plates"] = [{plate, matte, depth, frames}, ...]. Those are PC paths (repo-relative or absolute). The box needs the
# files AND paths that resolve there: repo-relative paths keep their place under the box root, absolute paths (any host) go to
# <root>/work/plates_ext/<sha1 of the path string>/<name>. The mapping depends only on the path string, so the PC launcher
# (shipping) and the on-box runner (rewriting) agree without talking to each other.
PLATE_KEYS = ("plate", "matte", "depth")
PLATES_EXT = "work/plates_ext"
INPUTS_MANIFEST = ".inputs_manifest.json"


class PlatesMissing(FileNotFoundError):
    """a piece declares contour ports but a plate / matte / depth file it names does not exist (locally or on the box)."""


def _is_abs(p: str) -> bool:
    import re
    return bool(re.match(r"^([A-Za-z]:[\/]|/|\\)", p))


def plate_box_rel(path) -> str:
    """box-root-relative posix path of one plate string (deterministic, host independent)."""
    import hashlib
    import posixpath
    p = str(path).replace("\\", "/")
    if _is_abs(p) or posixpath.normpath(p).startswith(".."):
        return f"{PLATES_EXT}/{hashlib.sha1(p.encode()).hexdigest()[:8]}/{posixpath.basename(p.rstrip('/'))}"
    return posixpath.normpath(p)


def plate_strings(plates):
    """[(shot index, key, path string)] of every plate / matte / depth a `type.plates` list names (PlateData objects name none)."""
    out = []
    if not isinstance(plates, (list, tuple)):
        return out
    for i, sh in enumerate(plates):
        if isinstance(sh, dict):
            out += [(i, k, str(sh[k])) for k in PLATE_KEYS if sh.get(k)]
    return out


def rewrite_plates(plates, box_root: str):
    """deep copy of `type.plates` with every path replaced by its absolute box path (<box_root>/<plate_box_rel>)."""
    import copy
    if not isinstance(plates, (list, tuple)):
        return plates
    new = copy.deepcopy(list(plates))
    for i, k, p in plate_strings(plates):
        new[i][k] = f"{box_root.rstrip('/')}/{plate_box_rel(p)}"
    return new


def plate_pairs(plates, local_root):
    """-> (pairs [(local Path, box-root-relative posix path)], missing [path strings]). A PNG directory is expanded to its files."""
    from pathlib import Path
    pairs, missing = [], []
    for _, _, p in plate_strings(plates):
        loc = Path(p) if _is_abs(p.replace("\\", "/")) else Path(local_root) / p
        rel = plate_box_rel(p)
        if loc.is_file():
            pairs.append((loc, rel))
        elif loc.is_dir() and any(loc.iterdir()):
            pairs += [(f, f"{rel}/{f.relative_to(loc).as_posix()}") for f in sorted(loc.rglob("*")) if f.is_file()]
        else:
            missing.append(p)
    return pairs, sorted(set(missing))


def declares_ports(type_spec) -> bool:
    """True when a type spec has a word with a contour port (dict entry / exit, or a `via`)."""
    return any(isinstance(w.get("entry"), dict) or isinstance(w.get("exit"), dict) or w.get("via")
               for w in (type_spec or {}).get("words") or [])


def missing_on_disk(box_plates):
    """box-path plates (rewrite_plates output) -> the file / directory paths that do not exist on THIS machine."""
    import os
    return sorted({p for _, _, p in plate_strings(box_plates) if not os.path.exists(p)})


def delta_put(rx, pairs, manifest=INPUTS_MANIFEST):
    """pairs: [(local Path, absolute remote path)] -> upload only files whose sha1 differs from the box's manifest (<root>/<manifest>, keyed
    by remote path) or that are gone from the box; ONE `rx.sh` round trip to compare + ONE `rx.put_many` tar stream. -> dict(sent, skipped, bytes)."""
    import hashlib
    from pathlib import Path
    pairs = [(Path(s), r) for s, r in pairs]
    if not pairs:
        return dict(sent=0, skipped=0, bytes=0)
    sha = {r: hashlib.sha1(Path(s).read_bytes()).hexdigest() for s, r in pairs}
    mf = f"{rx.root}/{manifest}"
    have, there = {}, set()
    if not getattr(rx, "dry", False):
        lst = " ".join(rx.q(r) for _, r in pairs)
        out = rx.sh(f"cat {rx.q(mf)} 2>/dev/null; echo; echo @@EXIST; for f in {lst}; do [ -f \"$f\" ] && echo \"$f\"; done; true",
                    check=False).stdout or ""
        head, _, tail = out.partition("@@EXIST")
        try:
            have = json.loads(head.strip()) if head.strip() else {}
        except ValueError:
            have = {}
        there = {x.strip() for x in tail.splitlines() if x.strip()}
    todo = [(s, r) for s, r in pairs if not (have.get(r) == sha[r] and r in there)]
    if todo:
        rx.put_many(todo)
    new = dict(have)
    new.update({r: sha[r] for _, r in pairs})
    rx.put_many([(json.dumps(new).encode(), mf)])
    return dict(sent=len(todo), skipped=len(pairs) - len(todo), bytes=sum(Path(s).stat().st_size for s, _ in todo))
