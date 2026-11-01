"""SoundFont bank registry + one-time download into a cache folder (never into the repo).

Bank list, versions, hashes and fallback sources follow op55 (github.com/ynginkim/op55, MIT,
skills/op55/scripts/lib/banks.mjs). Licence texts: ``licenses/banks/README.md``.

    from mograph.banks import ensure_banks
    paths = ensure_banks(["gm", "upright"])      # -> {"gm": Path(...), "upright": Path(...)}

Cache dir: ``$MOGRAPH_BANKS`` if set, else ``~/.cache/mograph/banks``.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

__all__ = ["BANKS", "bank_dir", "bank_path", "ensure_banks", "sha256_file"]

# op55's release hosts gzip copies of gm + upright (GeneralUser's author asks redistributors to serve their own copy).
_OP55_RELEASE = "https://github.com/ynginkim/op55/releases/download/banks-v1"

# Sources are tried in order. unpack: "none" (raw .sf2) | "gzip" | "archive" (7z / tar.xz via system `tar`, bsdtar).
# `sha256` of the final .sf2: op55's pinned values (it computes them from the same files); we re-verify on every download.
BANKS: dict[str, dict] = {
    "gm": {
        "name": "gm",
        "title": "GeneralUser GS 2.0.3",
        "file": "GeneralUser-GS-2.0.3.sf2",
        "size": 32319396,
        "sha256": "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
        "url": "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/97049183643d5fc5a9322a69c5b09efb667c6c3a/GeneralUser-GS.sf2",
        "sources": [  # author's own GitHub repo at a pinned commit first (not his website download files), op55 mirror second
            {"url": "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/97049183643d5fc5a9322a69c5b09efb667c6c3a/GeneralUser-GS.sf2", "unpack": "none"},
            {"url": f"{_OP55_RELEASE}/GeneralUser-GS-2.0.3.sf2.gz", "unpack": "gzip"},
        ],
        "license": "GeneralUser GS License v2.0: free for private and commercial music",
        "credit_required": False,
        "credit": None,
        "auto": True,
        "notes": "128 GM programs, variation banks, GS drum kits (kit 0 = Standard). Author notes some sample origins are uncertain.",
    },
    "upright": {
        "name": "upright",
        "title": "Upright Piano KW (FreePats, 2022-02-21)",
        "file": "UprightPianoKW-20220221.sf2",
        "size": 57377848,
        "sha256": "d9f5157720963671906727ca2e12b3293fd822c831bb0de477dd1c5f3ad37108",
        "url": f"{_OP55_RELEASE}/UprightPianoKW-20220221.sf2.gz",
        "sources": [  # op55 gzip copy first (plain Python can unpack it), official FreePats 7z second (needs `tar` with 7z support)
            {"url": f"{_OP55_RELEASE}/UprightPianoKW-20220221.sf2.gz", "unpack": "gzip"},
            {"url": "https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-SF2-20220221.7z", "unpack": "archive"},
        ],
        "license": "CC0 1.0 (public domain)",
        "credit_required": False,
        "credit": None,
        "auto": True,
        "notes": "Kawai upright, stereo, 2 velocity layers. Preset 0:0 'Upright Piano KW'.",
    },
    "grand": {
        "name": "grand",
        "title": "Salamander Grand Piano V3 (2020-06-02)",
        "file": "SalamanderGrandPiano-V3-20200602.sf2",
        "size": 1266462334,
        "sha256": "712d0e681efbe5203a8014e9b3e84168f1908c82f2f6fb13bd2c77d6d72c70b7",
        "url": "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3%2B20200602.tar.xz",
        "sources": [
            {"url": "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3%2B20200602.tar.xz", "unpack": "archive"},
        ],
        "license": "CC BY 3.0 (credit required)",
        "credit_required": True,
        "credit": "Salamander Grand Piano by Alexander Holm (CC BY 3.0)",
        "auto": False,  # 1.2 GB on disk: registered only; ensure_banks skips it unless allow_large=True
        "notes": "Yamaha C5, 16 velocity layers. NOT downloaded by default.",
    },
}


def bank_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return Path(os.environ.get("MOGRAPH_BANKS") or Path.home() / ".cache" / "mograph" / "banks")


def bank_path(name: str, cache_dir: str | os.PathLike | None = None) -> Path:
    if name not in BANKS:
        raise KeyError(f"unknown bank {name!r} (have: {', '.join(BANKS)})")
    return bank_dir(cache_dir) / BANKS[name]["file"]


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, to: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mograph-banks/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(to, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def _fetch(b: dict, dest: Path, tmp_dir: Path) -> str:
    """Try each source until one gives a file with the pinned sha256. Returns the url used."""
    last = "no sources"
    for src in b["sources"]:
        part = tmp_dir / f".{b['name']}.part"
        try:
            print(f"  from {src['url']}", file=sys.stderr)
            if src["unpack"] == "none":
                _download(src["url"], part)
            elif src["unpack"] == "gzip":
                gz = part.with_suffix(".gz")
                _download(src["url"], gz)
                with gzip.open(gz, "rb") as fi, open(part, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                gz.unlink()
            else:  # archive: unpack with system tar into a scratch dir, take the first .sf2
                arc = tmp_dir / f".{b['name']}.arc{Path(src['url']).suffix}"
                _download(src["url"], arc)
                x = Path(tempfile.mkdtemp(dir=tmp_dir, prefix=f".{b['name']}."))
                subprocess.run(["tar", "-xf", str(arc), "-C", str(x)], check=True)
                found = next(x.rglob("*.sf2"), None)
                if found is None:
                    raise RuntimeError("no .sf2 in the archive")
                shutil.move(str(found), part)
                shutil.rmtree(x, ignore_errors=True)
                arc.unlink(missing_ok=True)
            h = sha256_file(part)
            if h != b["sha256"]:
                raise RuntimeError(f"sha256 mismatch (got {h})")
            part.replace(dest)
            return src["url"]
        except Exception as e:  # noqa: BLE001 - try the next source
            last = f"{src['url']}: {e}"
            print(f"  failed: {last}", file=sys.stderr)
            part.unlink(missing_ok=True)
    raise RuntimeError(f"bank {b['name']!r}: every source failed (last: {last})")


def ensure_banks(names=("gm", "upright"), cache_dir: str | os.PathLike | None = None,
                 allow_large: bool = False) -> dict[str, Path | None]:
    """Download missing banks once into cache_dir and verify sha256. Returns {name: path}.

    Banks with auto=False (Salamander grand, 1.2 GB) map to None unless allow_large=True.
    """
    d = bank_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path | None] = {}
    for name in names:
        b = BANKS[name]
        p = d / b["file"]
        if not b["auto"] and not allow_large:
            print(f"skip {name}: {b['title']} is optional and large (pass allow_large=True)", file=sys.stderr)
            out[name] = p if p.exists() else None
            continue
        if p.exists() and p.stat().st_size == b["size"]:
            out[name] = p
            continue
        print(f"download {name}: {b['title']} [{b['license']}]", file=sys.stderr)
        _fetch(b, p, d)
        out[name] = p
        if b["credit_required"]:
            print(f"note: using {name!r} requires a credit line: {b['credit']}", file=sys.stderr)
    return out


if __name__ == "__main__":
    for k, v in ensure_banks(sys.argv[1:] or ["gm", "upright"]).items():
        print(k, v)
