#!/usr/bin/env python3
"""Download the model files for one VRAM tier into a ComfyUI install.

Reads models.json (repo root). Each entry lands at <comfy>/<dest>. Files already
present with the expected size are skipped. Downloads stream to <dest>.part with
HTTP Range resume and retries, then are size-checked and renamed into place.

  python tools/download_models.py --tier 24 --comfy /workspace/ComfyUI
  python tools/download_models.py --source r2 --tier 16 --only video,audio
  python tools/download_models.py --dry-run --tier 32

--source origin  Hugging Face URLs. Gated repos (Lightricks/LTX-2.5) need an HF
                 token in $HF_TOKEN from an account that accepted the license.
--source r2      Public R2 mirror (<R2_PUBLIC_BASE>/<r2_key>); no token needed.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

R2_PUBLIC_BASE = "https://pub-ad01ab5c49c349979d7ee68a67eb34c9.r2.dev"
MANIFEST = Path(__file__).resolve().parent.parent / "models.json"
TIERS = (12, 16, 24, 32)
GROUPS = ("planner", "image", "video", "audio", "voice")
CHUNK = 8 * 1024 * 1024
RETRIES = 6
USER_AGENT = "motion-graphic-download_models/1"


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never forward Authorization to another host (HF -> CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
                new.headers.pop("Authorization", None)
                new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnHostChange)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


def load_manifest() -> list[dict]:
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != 1:
        sys.exit(f"Unsupported models.json version: {data.get('version')}")
    return data["models"]


def select(models: list[dict], tier: int, only: set[str] | None) -> list[dict]:
    return [m for m in models if tier in m["tiers"] and (only is None or m["group"] in only)]


def source_url(m: dict, source: str) -> str:
    if source == "r2":
        return f"{R2_PUBLIC_BASE}/{urllib.parse.quote(m['r2_key'])}"
    return m["origin_url"]


def local_state(path: Path, expected: int) -> str:
    if path.is_file():
        return "present" if path.stat().st_size == expected else "wrong-size"
    part = path.with_name(path.name + ".part")
    if part.is_file():
        return f"partial {human(part.stat().st_size)}"
    return "missing"


def download(m: dict, url: str, dest: Path, token: str | None) -> None:
    """Stream url -> dest with Range resume. Raises on final failure."""
    expected = m["size_bytes"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected:
            part.unlink()
            have = 0
        if have == expected:
            break
        headers = {"User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with _OPENER.open(req, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # server ignored Range: restart from scratch
                mode = "ab" if have else "wb"
                done = have
                t0 = t_last = time.monotonic()
                with open(part, mode) as out:
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                        done += len(buf)
                        now = time.monotonic()
                        if now - t_last >= 5:
                            rate = (done - have) / max(now - t0, 1e-6)
                            print(f"    {human(done)} / {human(expected)}  {human(rate)}/s", flush=True)
                            t_last = now
            if part.stat().st_size == expected:
                break
            last_err = IOError(f"short read: {part.stat().st_size} of {expected} bytes")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and e.headers.get("X-Error-Code") in ("GatedRepo", "GatedRepoError"):
                raise RuntimeError("gated repo: set HF_TOKEN (license accepted) or use --source r2") from e
            if e.code == 416:  # range not satisfiable: .part is complete or corrupt
                if part.exists() and part.stat().st_size == expected:
                    break
                part.unlink(missing_ok=True)
            if e.code in (401, 403, 404):
                raise RuntimeError(f"HTTP {e.code} for {url}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        if attempt < RETRIES:
            wait = min(60, 2 ** attempt)
            print(f"    attempt {attempt}/{RETRIES} failed ({last_err}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed after {RETRIES} attempts: {last_err}")

    size = part.stat().st_size
    if size != expected:
        raise RuntimeError(f"size mismatch: got {size}, expected {expected}")
    os.replace(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("origin", "r2"), default="origin")
    ap.add_argument("--tier", type=int, choices=TIERS, default=24)
    ap.add_argument("--comfy", default="/workspace/ComfyUI", help="ComfyUI root (default /workspace/ComfyUI)")
    ap.add_argument("--only", help=f"comma-separated groups: {','.join(GROUPS)}")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    args = ap.parse_args()

    only = None
    if args.only:
        only = {g.strip() for g in args.only.split(",") if g.strip()}
        unknown = only - set(GROUPS)
        if unknown:
            ap.error(f"unknown group(s): {', '.join(sorted(unknown))}")

    comfy = Path(args.comfy)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    todo = select(load_manifest(), args.tier, only)
    if not todo:
        print("Nothing selected.")
        return 0

    total = sum(m["size_bytes"] for m in todo)
    print(f"Tier {args.tier} GB | source {args.source} | groups {','.join(sorted(only)) if only else 'all'} | "
          f"{len(todo)} files, {human(total)} | ComfyUI root {comfy}")
    if args.source == "origin" and not token and any(m["gated"] for m in todo):
        print("WARNING: gated files selected and HF_TOKEN is not set; they will fail (use --source r2).")

    rows = []  # (status, size, id, dest)
    for m in todo:
        dest = comfy / m["dest"]
        state = local_state(dest, m["size_bytes"])
        url = source_url(m, args.source)
        if args.dry_run:
            status = "skip (present)" if state == "present" else f"would get ({state})"
            if args.source == "origin" and m["gated"] and state != "present" and not token:
                status += " NEEDS HF_TOKEN"
            rows.append((status, m["size_bytes"], m["id"], url))
            continue
        if state == "present":
            rows.append(("skipped", m["size_bytes"], m["id"], m["dest"]))
            continue
        if state == "wrong-size":
            print(f"  {m['dest']}: size differs from manifest, re-downloading")
            dest.unlink()
        print(f"==> {m['id']}  ({human(m['size_bytes'])})\n    {url}\n    -> {dest}", flush=True)
        use_token = token if (args.source == "origin" and urllib.parse.urlsplit(url).netloc == "huggingface.co") else None
        try:
            download(m, url, dest, use_token)
            rows.append(("downloaded", m["size_bytes"], m["id"], m["dest"]))
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            print(f"    FAILED: {e}", flush=True)
            rows.append((f"FAILED: {e}", m["size_bytes"], m["id"], m["dest"]))

    w = max(len(r[0]) for r in rows)
    print("\n" + "-" * 100)
    print(f"{'status':<{w}}  {'size':>10}  {'id':<55} {'url' if args.dry_run else 'dest'}")
    for status, size, mid, where in rows:
        print(f"{status:<{w}}  {human(size):>10}  {mid:<55} {where}")
    print("-" * 100)
    need = sum(r[1] for r in rows if not r[0].startswith(("skip", "skipped")))
    failed = [r for r in rows if r[0].startswith("FAILED")]
    print(f"Selected {len(rows)} files, {human(total)} total; "
          f"{'to download' if args.dry_run else 'fetched/failed'}: {human(need)}"
          + (f"; FAILED {len(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
