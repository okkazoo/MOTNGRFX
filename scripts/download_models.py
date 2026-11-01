#!/usr/bin/env python3
"""Download every model file the MOTNGRFX app graphs need into ComfyUI/models/<folder>/.

    python scripts/download_models.py --comfy /path/to/ComfyUI            # everything used by app/graphs/fusion*.json
    python scripts/download_models.py --comfy ... --dry-run               # list files, sizes, what exists
    python scripts/download_models.py --comfy ... --skip-gated            # only the files that need no Hugging Face login
    python scripts/download_models.py --comfy ... --extras angles         # + the optional MOSS edit LoRA (undeclared licence, off by default)

The graphs are scanned for model file names, so the list never drifts from what the app loads. Existing files of the
right size are skipped; partial downloads resume (``<name>.part``). Several LTX 2.5 files and the Alpha-Gen IC-LoRA live in
gated Hugging Face repos: accept the licence on each repo page once, then export a read token
(``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``) before running, or run ``huggingface-cli login`` once (the token file in
``$HF_HOME/token``, default ``~/.cache/huggingface/token``, is read when the variables are unset; the token is never printed).
Only one downloader may run per models folder (lock file ``.motngrfx_download.lock``): two copies would corrupt ``.part`` files.
Sizes are 1024-based and labelled GiB. Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXTRAS, MODELS, find_comfy, graph_files, human_size, model_files  # noqa: E402


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects to a signed CDN URL; sending our Authorization header there makes the CDN reject it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnRedirect)


def _token_file() -> Path:
    """Where ``huggingface-cli login`` writes the token ($HF_TOKEN_PATH, else $HF_HOME/token, else ~/.cache/huggingface/token)."""
    if os.environ.get("HF_TOKEN_PATH"):
        return Path(os.environ["HF_TOKEN_PATH"]).expanduser()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home).expanduser() / "token"


def hf_token_source() -> tuple[str | None, str]:
    """(token, where it came from): "env", "token file" or "none". Callers must never print the token."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env and env.strip():
        return env.strip(), "env"
    try:
        tok = _token_file().read_text(encoding="utf-8").strip()
    except OSError:
        tok = ""
    if tok:
        return tok, "token file"
    return None, "none"


def hf_token() -> str | None:
    return hf_token_source()[0]


LOCK_NAME = ".motngrfx_download.lock"


class DownloadLocked(RuntimeError):
    """Another live downloader already holds the lock for this models folder."""

    def __init__(self, pid: int, path: Path):
        super().__init__(f"another download_models.py (PID {pid}) is already running on this models folder (lock {path}); "
                         "wait for it to finish, or stop it first. Two copies would corrupt the .part files.")
        self.pid, self.path = pid, path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":                                      # os.kill(pid, 0) would TERMINATE the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)             # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(models_dir: Path) -> Path:
    """Create ``<models_dir>/.motngrfx_download.lock`` holding our PID. A lock whose PID is dead is stale and replaced;
    a lock held by a live process raises DownloadLocked."""
    models_dir.mkdir(parents=True, exist_ok=True)
    lock = models_dir / LOCK_NAME
    for _ in range(3):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise DownloadLocked(pid, lock)
            if pid == os.getpid():                          # we hold it already (second acquire in one process)
                raise DownloadLocked(pid, lock)
            try:
                lock.unlink()                                # stale: its process is gone
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return lock
    raise DownloadLocked(0, lock)


def release_lock(lock: Path) -> None:
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def download(url: str, dest: Path, size: int, token: str | None) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "motngrfx-download/1.0"}
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = _OPENER.open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have == size:          # already complete, just not renamed
            part.replace(dest)
            return
        raise
    if have and resp.status != 206:                 # server ignored Range: start over
        have = 0
    mode = "ab" if have else "wb"
    done, t0, last = have, time.time(), 0.0
    with open(part, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 1.0:
                rate = (done - have) / max(now - t0, 1e-6)
                eta = (size - done) / rate if rate > 0 and size else 0
                print(f"\r    {human_size(done)} / {human_size(size)}  {human_size(rate)}/s  eta {eta:5.0f}s   ", end="", flush=True)
                last = now
    print("\r" + " " * 70 + "\r", end="")
    if size and done != size:
        raise IOError(f"size mismatch: got {done} bytes, expected {size}")
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", help="ComfyUI folder (default: $COMFYUI_PATH, or the folder this pack is installed under)")
    ap.add_argument("--graph", action="append", help="graph file or glob inside app/graphs (default fusion*.json); repeatable")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    ap.add_argument("--skip-gated", action="store_true", help="skip files in gated Hugging Face repos")
    ap.add_argument("--models-dir", help="override <ComfyUI>/models (e.g. when extra_model_paths.yaml points elsewhere)")
    ap.add_argument("--extras", action="append", choices=sorted(EXTRAS), default=[],
                    help="also download an optional extra (never downloaded by default): " + "; ".join(f"{k} = {v['what']} [{v['licence']}]" for k, v in EXTRAS.items()))
    args = ap.parse_args()

    graphs = graph_files(args.graph, downloadable=True)
    if not graphs:
        print("No graphs found in app/graphs", file=sys.stderr)
        return 2
    used = model_files(graphs)

    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        comfy = find_comfy(args.comfy)
        if comfy is None:
            print("Could not find ComfyUI. Pass --comfy /path/to/ComfyUI (the folder with main.py) or --models-dir.", file=sys.stderr)
            return 2
        models_dir = comfy / "models"

    token, token_src = hf_token_source()
    print(f"Graphs: {', '.join(g.name for g in graphs)}")
    print(f"Models folder: {models_dir}")
    print(f"HF token: {'set (' + token_src + ')' if token else 'not set (gated repos will fail with 401)'}\n")

    table = dict(MODELS)
    for key in args.extras:                                  # an extra is keyed by the file name the graphs use
        x = EXTRAS[key]
        table[x["dest"]] = x
        used.setdefault(x["dest"], [f"--extras {key}"])
        print(f"NOTE --extras {key}: {x['licence']}")
    unknown = [f for f in used if f not in table]
    todo, rows = [], []
    for fname in sorted(used, key=lambda f: (table.get(f, {}).get("folder", "?"), f)):
        m = table.get(fname)
        if m is None:
            continue
        dest = models_dir / m["folder"] / fname
        exists = dest.is_file() and (dest.stat().st_size == m["size"])
        wrong = dest.is_file() and not exists
        state = "ok" if exists else ("WRONG SIZE" if wrong else "missing")
        gated = " (gated)" if m.get("gated") else ""
        rows.append((m["folder"], fname, human_size(m["size"]), state + gated))
        if not exists and not (args.skip_gated and m.get("gated")):
            todo.append((fname, m, dest))
    w = max(len(r[1]) for r in rows) if rows else 10
    for folder, fname, size, state in rows:
        print(f"  {folder:<22} {fname:<{w}}  {size:>10}  {state}")
    total = sum(m["size"] for _f, m, _d in todo)
    print(f"\n{len(rows)} model files used; {len(todo)} to download = {human_size(total)}")
    if unknown:
        print("\nNot in the download table (get them yourself or add them to scripts/_common.py MODELS):")
        for f in unknown:
            print(f"  {f}  (used by {', '.join(used[f])})")
    if args.dry_run or not todo:
        return 1 if unknown else 0

    free = _free_bytes(models_dir)
    if free is not None and free < total * 1.02:
        print(f"\nNot enough free disk: need {human_size(total)}, have {human_size(free)} on {models_dir}", file=sys.stderr)
        return 3

    try:
        lock = acquire_lock(models_dir)
    except DownloadLocked as e:
        print(f"\n{e}", file=sys.stderr)
        return 4
    try:
        return _download_all(todo, token, unknown, lock)
    finally:
        release_lock(lock)


def _download_all(todo, token, unknown, lock) -> int:
    failed = []
    for i, (fname, m, dest) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {fname}  ({human_size(m['size'])}) -> {dest.parent}")
        if m.get("gated") and not token:
            print("    gated repo and no HF token: skipping. Accept the licence on the repo page, then export HF_TOKEN or run huggingface-cli login.")
            failed.append(fname)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            download(m["url"], dest, m["size"], token)
            print(f"    done {human_size(dest.stat().st_size)}")
        except urllib.error.HTTPError as e:
            hint = " (gated repo: accept the licence on its Hugging Face page and check HF_TOKEN)" if e.code in (401, 403) else ""
            print(f"    FAILED: HTTP {e.code} {e.reason}{hint}\n    {m['url']}")
            failed.append(fname)
        except (OSError, urllib.error.URLError) as e:
            print(f"    FAILED: {e}\n    re-run the script to resume.")
            failed.append(fname)
    if failed:
        print(f"\n{len(failed)} file(s) not downloaded: {', '.join(failed)}")
        return 1
    print("\nAll models present. Restart ComfyUI, then run: python scripts/check_install.py --url http://127.0.0.1:8188")
    return 1 if unknown else 0


def _free_bytes(path: Path) -> int | None:
    import shutil
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())
