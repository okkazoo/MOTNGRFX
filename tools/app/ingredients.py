#!/usr/bin/env python
"""Everything one composition is made of, on this PC, together, under ONE searchable name (16_app/INGREDIENTS.md).

    python tools/app/ingredients.py pull --iid 58135115 --version COMPETITIONS/motion-graphic/22_batch-24gb/v002 [--run r01]
    python tools/app/ingredients.py pull --version COMPETITIONS/motion-graphic/21_batch-3060/v005 --no-box     # destroyed box: gather what is local
    python tools/app/ingredients.py check <version>/ingredients/<stem>                                          # will the dashboard collapse it?

For every run (`<stem>` = `motion-graphic_batch-24gb_v002_r01`, tools/fusion/ingredient_names.py) it builds `<version>/ingredients/<stem>/`:
the final mp4, the music wav, the stills, per shot clip / clean / textbg / depth / flow / contour previews, the matte PNG sequences (one folder per shot),
the type passes as mp4 (+ type matte PNGs), LTX-read crops, the 3D shot cuts (mp4 + matte PNGs), depth PNGs, composed shots, clean plates, the design sheet, places json,
piece / concept / design json + text, MANIFEST.md (starts with the size of the composition and its full_archive share). Every name starts with `<stem>_`: search the stem in Outputs.

How the bytes get there
  * the box archive dir `<remote>/<stem>_out_shots/` is listed over ssh (the box address comes from the dashboard API); a file the dashboard sync already put
    in `<version>/out/<stem>_out_shots/` (same size) is NOT fetched again; the rest (json, wav, whatever the sync skipped or has not reached) comes with
    `tar | ssh` into `<version>/local/ingredients_pull/<stem>/`. READ ONLY on the box (ls / find / tar -c).
  * `ingredients/<stem>/` is made of NTFS HARDLINKS of those files and of the final mp4 in `<version>/out/`: no extra space, never a copy (a link that
    fails is reported and skipped). The matte PNGs stay PNG, one folder per shot (>= 2 folders of >= 16 frames = the dashboard shows ONE stack tile,
    core/output_collapse.py in the vastai-api-manager); the stacked-mattes mp4 of review/ is linked in as `<stem>_mattes_stacked.mp4`.
  * old archives (before 2026-10-29) have short names (still.png, s1_clip.mp4, s1_matte/00001.png, piece.json): they are linked in under the prefixed
    names. What an old run never saved (type passes, places, depth / flow / contour previews) is simply absent; the wav is cut from the final mp4.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.fusion import ingredient_names as IN      # noqa: E402

API = "http://127.0.0.1:8000"
KEY = os.path.expanduser("~/.ssh/vastai_comfyui_rsa")
REMOTE_OUT = "/workspace/ComfyUI/output"
PROJ = os.path.expanduser("~/.claude/skills/project/proj.py")
PROJECTS_ROOT = os.path.dirname(os.path.dirname(ROOT))          # .../PROJECTS
MIN_SEQ = 16                                                     # the dashboard's SEQ_MIN_FRAMES
FPS = 24


# ----------------------------------------------------------------------------------------------- version / box
def resolve_version(target: str, runner=subprocess.run) -> str:
    """a directory, or a PROJECTS target `P/idea/vNNN` -> the version directory."""
    if os.path.isdir(target):
        return os.path.abspath(target)
    p = runner([sys.executable, PROJ, "resolve", "--target", target], capture_output=True, text=True)
    try:
        d = (json.loads(p.stdout).get("target") or {}).get("dir")
    except ValueError:
        d = None
    if not d:
        raise SystemExit(f"cannot resolve {target!r}: {(p.stdout + p.stderr).strip()[:300]}")
    return d


def remote_version_dir(vdir: str) -> str:
    """local `.../PROJECTS/COMPETITIONS/motion-graphic/22_batch-24gb/v002` -> `/workspace/ComfyUI/output/projects/COMPETITIONS/motion-graphic/batch-24gb/v002`
    (the idea folder loses its `NN_` order prefix on the box)."""
    rel = os.path.relpath(os.path.abspath(vdir), PROJECTS_ROOT).replace("\\", "/").split("/")
    if len(rel) >= 3:
        rel[-2] = re.sub(r"^\d+_", "", rel[-2])
    return f"{REMOTE_OUT}/projects/" + "/".join(rel)


def box_info(iid: str, api: str = API, opener=urllib.request.urlopen) -> dict | None:
    """ssh endpoint of a box from the dashboard API, None when the box is gone / not ready."""
    try:
        d = json.load(opener(f"{api}/api/instances/{iid}", timeout=20))
    except Exception:      # noqa: BLE001
        return None
    if not d.get("ssh_host") or not d.get("ssh_port") or d.get("actual_status") not in (None, "running"):
        return None
    return {"host": d["ssh_host"], "port": int(d["ssh_port"]), "user": d.get("ssh_user") or "root"}


def ssh_argv(box: dict, remote_cmd: str) -> list[str]:
    return ["ssh", "-i", KEY, "-p", str(box["port"]), "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",      # a dead link ends the session in ~60 s instead of never
            f"{box['user']}@{box['host']}", remote_cmd]


def remote_list(box: dict, remote_dir: str, runner=subprocess.run) -> dict[str, int] | None:
    """{relative path: bytes} of every file under remote_dir (None: ssh failed or no such folder). Read only."""
    cmd = f"cd {shlex.quote(remote_dir)} 2>/dev/null && find . -type f -printf '%P\\t%s\\n'"
    try:
        p = runner(ssh_argv(box, cmd), capture_output=True, timeout=180, stdin=subprocess.DEVNULL)      # no stdin: ssh must not wait on the parent's console
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode:
        return None
    out = {}
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if "\t" in line:
            rel, size = line.rsplit("\t", 1)
            if size.isdigit() and rel:
                out[rel] = int(size)
    return out


def safe_member(name: str) -> str | None:
    n = name.replace("\\", "/")
    if n.startswith("/"):
        return None
    n = n[2:] if n.startswith("./") else n
    parts = n.split("/")
    if not n or ".." in parts or ":" in parts[0]:
        return None
    return n


FETCH_STALL_S = 120.0       # no byte from the box for this long = dead link: ssh killed, what arrived is kept
FETCH_MAX_S = 1800.0        # hard ceiling of one fetch


class _Watched:
    """file-like read side of the ssh pipe that stamps the time of every read, for the stall watchdog."""

    def __init__(self, raw, state):
        self.raw, self.state = raw, state

    def read(self, n=-1):
        b = self.raw.read(n)
        self.state["t"] = time.time()
        return b


def fetch(box: dict, remote_dir: str, rels: list[str], dest: str, log=print, popen=subprocess.Popen, stall_s: float = FETCH_STALL_S,
          max_s: float = FETCH_MAX_S) -> int:
    """`tar -cf - -T -` on the box piped into this process, extracted member by member under dest (plain files only, names checked). -> files written.
    Bounded: a watchdog thread kills ssh when no byte arrives for stall_s or the whole fetch passes max_s (the 2026-10-29 run_batch hang sat in an unbounded
    read of this pipe); the files extracted before that are kept and the caller carries on with them."""
    if not rels:
        return 0
    want = set(rels)
    cmd = f"cd {shlex.quote(remote_dir)} && tar -cf - -T -"
    p = popen(ssh_argv(box, cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    state = {"t": time.time(), "killed": None}
    t_start, done = time.time(), threading.Event()
    err_chunks: list[bytes] = []

    def feed():
        try:
            p.stdin.write(("\n".join(rels) + "\n").encode("utf-8"))
            p.stdin.close()
        except OSError:
            pass

    def drain():                 # stderr read concurrently: a chatty tar must not fill the pipe and stall the stream
        try:
            err_chunks.append(p.stderr.read())
        except Exception:       # noqa: BLE001
            pass

    def watch():
        while not done.wait(min(5.0, max(0.05, stall_s / 4))):
            now = time.time()
            why = (f"no data for {now - state['t']:.0f}s" if now - state["t"] > stall_s else f"over {max_s:.0f}s" if now - t_start > max_s else None)
            if why:
                state["killed"] = why
                try:
                    p.kill()
                except OSError:
                    pass
                return
    for fn in (feed, drain, watch):
        threading.Thread(target=fn, daemon=True).start()
    n = 0
    try:
        with tarfile.open(fileobj=_Watched(p.stdout, state), mode="r|") as tf:
            for m in tf:
                rel = safe_member(m.name)
                if not m.isreg() or rel is None or rel not in want:
                    continue
                dst = os.path.join(dest, *rel.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                src = tf.extractfile(m)
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                os.replace(tmp, dst)
                n += 1
    except (tarfile.TarError, OSError, ValueError) as e:
        log(f"  tar | ssh stream broke after {n} file(s): {type(e).__name__}: {str(e)[:200]}")
    finally:
        done.set()
    if state["killed"]:
        log(f"  tar | ssh killed ({state['killed']}); {n} file(s) kept")
    try:
        rc = p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -9
    if rc and not state["killed"]:
        log(f"  tar | ssh exit {rc}: {b''.join(err_chunks).decode('utf-8', 'replace')[-300:]}")
    return n


# ----------------------------------------------------------------------------------------------- local discovery
def find_runs(vdir: str, part: str | None = None) -> list[dict]:
    """runs found in <version>/out: {stem, part, final, shots_dir}. Stem from the `<stem>_out_shots` folder or the `<stem>_video_00001_.mp4` name."""
    runs: dict = {}
    out = os.path.join(vdir, "out")
    for root, dirs, files in os.walk(out):
        for d in dirs:
            if d.endswith("_shots"):
                stem = IN.run_stem(d[: -len("_shots")])
                runs.setdefault(stem, {})["shots_dir"] = os.path.join(root, d)
        for n in files:
            m = re.match(r"^(?P<stem>.+)_video_\d+_\.mp4$", n)
            if m:
                runs.setdefault(m.group("stem"), {})["final"] = os.path.join(root, n)
    res = []
    for stem, r in sorted(runs.items()):
        m = re.search(r"_(r\d+[a-z]?)$", stem)
        pt = m.group(1) if m else None
        if part and pt != part:
            continue
        res.append({"stem": stem, "part": pt, "final": r.get("final"), "shots_dir": r.get("shots_dir")})
    return res


def ffprobe(path: str) -> dict:
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path], capture_output=True, text=True, timeout=120)
        s = (json.loads(p.stdout).get("streams") or [{}])[0]
        num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
        return {"w": s.get("width"), "h": s.get("height"), "fps": round(float(num) / float(den or 1), 3) if float(den or 1) else None,
                "frames": int(s["nb_read_frames"]) if s.get("nb_read_frames") else None}
    except Exception:      # noqa: BLE001
        return {}


def extract_wav(mp4: str, wav: str) -> bool:
    """the audio of the final mp4 as 16-bit wav (what the old archives never saved)."""
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vn", "-c:a", "pcm_s16le", wav + ".part.wav"], capture_output=True, timeout=300,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        p = subprocess.CompletedProcess([], 1)
    if p.returncode or not os.path.isfile(wav + ".part.wav") or os.path.getsize(wav + ".part.wav") < 1000:
        if os.path.isfile(wav + ".part.wav"):
            os.remove(wav + ".part.wav")
        return False
    os.replace(wav + ".part.wav", wav)
    return True


# ----------------------------------------------------------------------------------------------- linking
def link(src: str, dst: str) -> str:
    """NTFS hardlink src at dst. 'new' | 'same' (already the same file) | 'exists' (a different file is there: left alone) | 'fail:<why>'. Never copies."""
    try:
        if os.path.lexists(dst):
            return "same" if os.path.samefile(src, dst) else "exists"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.link(src, dst)
        return "new"
    except OSError as e:
        return f"fail:{e}"


def plan_files(stem: str, sources: dict[str, str]) -> dict[str, str]:
    """{path in the ingredients folder (relative): absolute source}. `sources` maps archive-relative names (prefixed or legacy) to files; a legacy name
    whose prefixed twin exists is dropped (same bytes)."""
    out: dict[str, str] = {}
    for rel in sorted(sources):
        new = IN.to_new(stem, rel)
        if new is None:
            continue
        if new != rel and new in sources:
            continue
        out.setdefault(new, sources[rel])
    return out


# ----------------------------------------------------------------------------------------------- manifest
_DESC = [
    (r"_video\.mp4$", "the finished composition (final mp4, audio muxed)", "SaveVideo `OUT final` <- MographAssemble"),
    (r"_music\.wav$", "the music mix of the piece", "the audio of the assembled video (MographArchive); old runs: cut from the final mp4 with ffmpeg"),
    (r"_still\.png$", "the first frame of the piece (the MOSS still)", "MographFirstFrame -> MographArchive"),
    (r"_still_(\d+)\.png$", "still of place {1} (the first frame of the shots that start there)", "MographFirstFrames batch row {1}-1 -> MographArchive"),
    (r"_(s\d+)_clip\.mp4$", "shot {1}: the LTX footage as it enters compose", "MographDecodeClip / MographFootageGather -> clips"),
    (r"_(s\d+)_clean\.mp4$", "shot {1}: the CLEAN clip (three-layer shot: the subject pixels the matte was cut from)", "MographLayerClean (plain image-to-video)"),
    (r"_(s\d+)_textbg\.mp4$", "shot {1}: the text-rendered background (union-control V2V of the clean clip)", "MographLayerGather / text pass"),
    (r"_(s\d+)_matte$", "shot {1}: Alpha-Gen IC-LoRA matte, one PNG per frame (stack member)", "MographDecodeClip (gray) after the Alpha-Gen IC-LoRA sampler"),
    (r"_cut_(s\d+)-(s\d+)_matte$", "3D shot cut {1} -> {2}: the subject matte THROUGH the transition, one PNG per frame (stack member)", "MographPivotCut (pivot_cut.cut_record) -> MographArchive, full_archive"),
    (r"_cut_(s\d+)-(s\d+)_(\w+)\.mp4$", "3D shot cut {1} -> {2} ({3}): the rendered transition frames (A's tail then B's head)", "MographPivotCut (tools/fusion/pivot_cut.py) -> MographArchive, full_archive"),
    (r"_cut_(s\d+)-(s\d+)_(\w+)\.json$", "3D shot cut {1} -> {2} ({3}): plan, resolved params, windows, which depth the engine used", "MographPivotCut -> MographArchive"),
    (r"_(s\d+)_depth\.png$", "shot {1}: depth of the FIRST frame (what the 3D cuts / parallax read), 8-bit gray PNG, bright = near", "Video-Depth-Anything -> MographArchive"),
    (r"_(s\d+)_depth$", "shot {1}: per-frame depth, 8-bit gray PNG per frame (stack member)", "Video-Depth-Anything -> MographArchive, full_archive"),
    (r"_(s\d+)_plate\.mp4$", "shot {1}: the CLEAN PLATE (the clean clip with the subject taken off; clean-plate route)", "MographLayerGather (clip['plate']) -> MographArchive"),
    (r"_(s\d+)_comp\.mp4$", "shot {1}: the composited shot as compose made it (type woven in, graded), before assemble", "MographCompose -> MographArchive, full_archive"),
    (r"_(s\d+)_sources\.json$", "shot {1}: what its compose ops read from OTHER shots (window_other / $next / $prev) and which files here hold it", "tools/fusion/window_cut.other_refs"),
    (r"_type_matte$", "type matte pass as PNG per frame (the lossless twin of `_type_matte.mp4`; stack member)", "type render dir_layers/matte -> MographArchive, full_archive"),
    (r"_design_sheet\.png$", "the sheet image the vision model was shown for the design", "MographDesignPackage 'sheet' -> MographArchive"),
    (r"_places\.json$", "per place: id, prompt, angle-still source + edit prompt (`_still_<k>.png` = row k)", "tools/fusion/places.py -> MographArchive"),
    (r"_(s\d+)_depth\.mp4$", "shot {1}: Video-Depth-Anything depth (gray preview, cheap encode)", "VideoDepthAnythingOutput -> MographArchive"),
    (r"_(s\d+)_flow\.mp4$", "shot {1}: footage optical flow, colour wheel (hue = direction)", "MographFootageFlow (RAFT / DIS) -> clip['flow']"),
    (r"_(s\d+)_contour_(\w+)\.mp4$", "shot {1}: traced contour `{2}` over the plate (the lines the type words enter / exit / ride)", "mograph/modules/contour_ports.build_track"),
    (r"_(s\d+)_contour_(\w+)\.npz$", "shot {1}: raw tracks of contour `{2}` (P[F,L,N,2] px of `size`, valid, closed, fps)", "mograph/modules/contour_ports.build_track"),
    (r"_type_manifest\.json$", "manifest of the type passes (timeline, camera plan, depth range)", "tools/fusion/type_passes.py"),
    (r"_type_(\w+)\.mp4$", "type pass `{1}`: the graphic-design animation by itself (1088x1920 compose layers)", "MographTypePasses / tools/fusion/type_passes.py"),
    (r"_ltx_read_(s\d+)_(\d+)\.png$", "shot {1}: crop {2} of the text-rendered clip the VLM read (LTX legibility check)", "MographLtxRead / tools/fusion/ltx_check.py"),
    (r"_ltx_reads\.json$", "what the VLM read in the LTX crops, per shot", "MographLtxRead"),
    (r"_piece\.json$", "the designed piece: words, fonts, modules, camera, transitions, compose ops, layout", "MographPieceFromJSON (design_piece) -> MographArchive"),
    (r"_index\.json$", "archive index: fps, per shot frames / sizes, file list", "MographArchive"),
    (r"_concept\.json$", "the concept used (title, subject, words, scene, music, look, shots)", "concept nodes (OUT concept)"),
    (r"_concept_reply\.txt$", "raw model reply to the concept prompt", "OUT concept reply"),
    (r"_concept_check\.txt$", "parse / repair verdict of the concept", "OUT concept check"),
    (r"_design_reply\.txt$", "raw model reply to the design prompt", "OUT design reply"),
    (r"_design_check\.txt$", "parse verdict of the design", "OUT design check"),
    (r"_design_final\.txt$", "final verdict of the design after repair", "OUT design final check"),
    (r"_(concept|design)_prompt(\.reconstructed)?\.txt$", "the {1} prompt sent to the model" + " (rebuilt offline with the code of the save date)", "model_replies.py (prompt_src reconstructed / exact)"),
    (r"_mattes_stacked\.mp4$", "all matte frames stacked shot after shot into ONE gray mp4 (24 fps)", "tools/app/matte_preview.py (review/)"),
]


def describe(stem: str, name: str) -> tuple[str, str]:
    base = name[len(stem):] if name.startswith(stem) else name
    for pat, what, origin in _DESC:
        m = re.search(pat, base)
        if m:
            g = (None,) + m.groups()
            return what.format(*g), origin.format(*g)
    return ("", "")


def _png_size(path: str):
    try:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return None if im is None else (im.shape[1], im.shape[0])
    except Exception:      # noqa: BLE001
        return None


def size_report(stem: str, ing: str) -> dict:
    """per-composition size estimate of an ingredients folder: total, by kind (mp4 / png / wav / text+json / other) and the share only written with MographArchive
    full_archive on (`ingredient_names.is_heavy`: cut renders + cut mattes, composed shots, per-frame depth PNGs, type matte PNGs). Hardlinks: no extra disk."""
    kinds: dict[str, int] = {}
    tot = heavy = nfiles = 0
    for name in sorted(os.listdir(ing)):
        p = os.path.join(ing, name)
        if os.path.isdir(p):
            sub = [(os.path.join(r, f), f) for r, _, fs in os.walk(p) for f in fs]
        else:
            sub = [(p, name)]
        for fp, fn in sub:
            try:
                b = os.path.getsize(fp)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            k = ext if ext in ("mp4", "png", "wav") else ("text" if ext in ("json", "txt", "md", "npz") else "other")
            kinds[k] = kinds.get(k, 0) + b
            tot += b
            nfiles += 1
            if IN.is_heavy(name):
                heavy += b
    return {"total": tot, "files": nfiles, "kinds": kinds, "full_archive_extras": heavy}


def size_line(rep: dict) -> str:
    mb = lambda b: f"{b / 1e6:.1f} MB"      # noqa: E731
    parts = ", ".join(f"{k} {mb(v)}" for k, v in sorted(rep["kinds"].items(), key=lambda kv: -kv[1]))
    return (f"Size of this composition: {mb(rep['total'])} in {rep['files']} files ({parts}); of which {mb(rep['full_archive_extras'])} are the full_archive extras "
            f"(rendered 3D cuts + cut mattes, composed shots, per-frame depth PNGs, type matte PNGs: `MographArchive full_archive=False` / `MOTNGRFX_ARCHIVE_FULL=0` leaves them out).")


def manifest_md(stem: str, vdir: str, ing: str, notes: list[str] | None = None, probe=ffprobe) -> str:
    """MANIFEST.md text: one line per file / folder: what it is, origin node, frames / size / fps."""
    L = [f"# MANIFEST — ingredients/{stem}", "",
         f"Everything of this composition in one folder (generated {time.strftime('%Y-%m-%d %H:%M')} by `python tools/app/ingredients.py`). Every name starts with "
         f"`{stem}_`: search that in Outputs. All files are NTFS HARDLINKS of the pulled box archive (`local/ingredients_pull/{stem}/`, or the dashboard sync in `out/`) "
         f"and of the final mp4 in `out/` (same bytes, no extra space). Mattes stay PNG, one folder per shot (>= 2 folders of >= 16 frames = one stack tile).", ""]
    L += [size_line(size_report(stem, ing)), ""]
    for n in notes or []:
        L.append(f"- {n}")
    if notes:
        L.append("")
    for name in sorted(os.listdir(ing)):
        p = os.path.join(ing, name)
        if name == f"{stem}_MANIFEST.md":
            continue
        what, origin = describe(stem, name)
        if os.path.isdir(p):
            fr = sorted(f for f in os.listdir(p) if f.lower().endswith(".png"))
            sz = _png_size(os.path.join(p, fr[0])) if fr else None
            L.append(f"- `{name}/` · {what or 'folder'} · {len(fr)} PNG frames `{fr[0] if fr else ''}`..`{fr[-1] if fr else ''}`"
                     + (f", {sz[0]}x{sz[1]}" if sz else "") + f", {FPS} fps · origin: {origin or '-'}")
            continue
        extra = ""
        if name.lower().endswith(".mp4"):
            i = probe(p)
            if i:
                extra = f" · {i.get('frames')} frames, {i.get('w')}x{i.get('h')}, {i.get('fps')} fps"
        elif name.lower().endswith(".png"):
            sz = _png_size(p)
            extra = f" · {sz[0]}x{sz[1]}" if sz else ""
        elif name.lower().endswith(".wav"):
            extra = f" · {os.path.getsize(p) / 1e6:.1f} MB"
        L.append(f"- `{name}` · {what or 'file'}{extra} · origin: {origin or '-'}")
    return "\n".join(L) + "\n"


def collapse_check(ing: str) -> dict:
    """does the dashboard rule (core/output_collapse.py) see a stack here? -> {'matte_dirs', 'stack': bool, 'frames': {dir: n}}."""
    frames = {}
    for d in sorted(os.listdir(ing)):
        p = os.path.join(ing, d)
        if os.path.isdir(p):
            frames[d] = len([f for f in os.listdir(p) if f.lower().endswith(".png")])
    ok = [d for d, n in frames.items() if n >= MIN_SEQ]
    return {"matte_dirs": len(ok), "stack": len(ok) >= 2, "frames": frames}


# ----------------------------------------------------------------------------------------------- text files from the run record
TEXTS = (("concept_reply", "txt"), ("concept", "json"), ("concept_check", "txt"), ("design_reply", "txt"), ("design_check", "txt"), ("design_final", "txt"))


def write_texts(stem: str, part: str | None, vdir: str, stage: str, rec: dict | None) -> dict[str, str]:
    """{ingredient name: source file}. From <version>/review/model_replies/<part>_<tail> (run_batch / model_replies.save) when present, else
    written from the run record (state.json) into the pull folder."""
    out = {}
    mdir = os.path.join(vdir, "review", "model_replies")
    if part and os.path.isdir(mdir):                          # every saved text of the run: replies, checks, prompts, repair replies, readings
        for f in sorted(os.listdir(mdir)):
            if f.startswith(part + "_") and f != f"{part}_piece.json" and os.path.isfile(os.path.join(mdir, f)):
                out[f"{stem}_{f[len(part) + 1:]}"] = os.path.join(mdir, f)
    for tail, ext in TEXTS:
        name = f"{stem}_{tail}.{ext}"
        if name in out:
            continue
        text = None
        if rec:
            if tail == "concept" and rec.get("concept"):
                text = json.dumps(rec["concept"], indent=1, ensure_ascii=False)
            elif tail in ("design_reply", "design_check", "concept_check") and rec.get(tail):
                text = rec[tail]
            elif tail == "design_final" and (rec.get("fixes") or {}).get("design_final"):
                text = "\n".join(rec["fixes"]["design_final"])
        if text:
            dst = os.path.join(stage, name)
            os.makedirs(stage, exist_ok=True)
            open(dst, "w", encoding="utf-8").write(text)
            out[name] = dst
    return out


# ----------------------------------------------------------------------------------------------- one run
def pull_run(vdir: str, run: dict, box: dict | None = None, rec: dict | None = None, log=print, lister=remote_list, fetcher=fetch) -> dict:
    """gather one run into <vdir>/ingredients/<stem>/. run: {stem, part, final, shots_dir}. -> report dict."""
    stem, part = run["stem"], run.get("part")
    stage = os.path.join(vdir, "local", "ingredients_pull", stem)
    sync = run.get("shots_dir")
    rep = {"stem": stem, "fetched": 0, "linked": 0, "skipped": [], "box": None}
    sources: dict[str, str] = {}
    # 1. files the dashboard sync already landed in out/<stem>_out_shots/
    if sync and os.path.isdir(sync):
        for root, _, files in os.walk(sync):
            for f in files:
                p = os.path.join(root, f)
                sources[os.path.relpath(p, sync).replace("\\", "/")] = p
    # 2. the box
    if box:
        rdir = remote_version_dir(vdir) + "/" + os.path.basename(sync or f"{stem}_out_shots")
        lst = lister(box, rdir)
        if lst is None:
            log(f"  {stem}: box archive {rdir} not listable (gone?): local files only")
            rep["box"] = "unreachable"
        else:
            need = []
            for r, size in sorted(lst.items()):
                new = IN.to_new(stem, r)
                if new is None or (new != r and new in lst):          # not an ingredient / a legacy name whose prefixed twin is on the box
                    continue
                if r in sources and os.path.getsize(sources[r]) == size:
                    continue                                           # the dashboard sync has it
                have = os.path.join(stage, *r.split("/"))
                if os.path.isfile(have) and os.path.getsize(have) == size:
                    continue                                           # an earlier pull has it
                need.append(r)
            rep["fetched"] = fetcher(box, rdir, need, stage, log) if need else 0
            rep["box"] = f"{len(lst)} files, {len(need)} fetched"
            for r in lst:
                p = os.path.join(stage, *r.split("/"))
                if os.path.isfile(p):
                    sources[r] = p
    elif os.path.isdir(stage):                                # earlier pull, box gone
        for root, _, files in os.walk(stage):
            for f in files:
                if not f.endswith(".part"):
                    p = os.path.join(root, f)
                    sources.setdefault(os.path.relpath(p, stage).replace("\\", "/"), p)
    plan = plan_files(stem, sources)
    # 3. the final mp4, the wav, texts, stacked mattes
    notes = []
    if run.get("final"):
        plan[f"{stem}_video.mp4"] = run["final"]
        wav = f"{stem}_music.wav"
        if wav not in plan:
            w = os.path.join(stage, wav)
            if os.path.isfile(w) or extract_wav(run["final"], w):
                plan[wav] = w
                notes.append(f"`{wav}` was cut from the final mp4 (this run's archive predates the music export)")
    else:
        notes.append("no final mp4 found in out/ yet")
    plan.update(write_texts(stem, part, vdir, stage, rec))
    prev = os.path.join(vdir, "review", f"{stem}_out_matte_preview.mp4")
    if os.path.isfile(prev):
        plan[f"{stem}_mattes_stacked.mp4"] = prev
    # 4. hardlinks
    ing = os.path.join(vdir, "ingredients", stem)
    os.makedirs(ing, exist_ok=True)
    for rel, src in sorted(plan.items()):
        r = link(src, os.path.join(ing, *rel.split("/")))
        if r.startswith("fail") or r == "exists":
            rep["skipped"].append(f"{rel}: {r}")
        elif r == "new":
            rep["linked"] += 1
    if not any(n.endswith("_piece.json") for n in plan):
        notes.append("no piece.json (json is not synced by the dashboard and the box archive was not reachable)")
    c = collapse_check(ing)
    rep["stack"] = c["stack"]
    if not c["stack"]:
        notes.append(f"only {c['matte_dirs']} matte folder(s) of >= {MIN_SEQ} frames: the dashboard will not collapse this folder into one stack tile")
    for s in rep["skipped"]:
        notes.append(f"NOT linked: {s}")
    open(os.path.join(ing, f"{stem}_MANIFEST.md"), "w", encoding="utf-8").write(manifest_md(stem, vdir, ing, notes))
    rep["dir"] = ing
    rep["files"] = len(os.listdir(ing))
    log(f"  {stem}: {rep['linked']} linked, {rep['fetched']} fetched, stack={rep['stack']}, {rep['files']} entries -> {ing}")
    return rep


def pull(version: str, iid: str | None = None, part: str | None = None, no_box: bool = False, state: str | None = None, api: str = API, log=print) -> list[dict]:
    vdir = resolve_version(version)
    runs = find_runs(vdir, part)
    if not runs:
        log(f"no run found under {vdir}/out" + (f" for {part}" if part else ""))
        return []
    box = None if no_box or not iid else box_info(iid, api)
    if iid and not no_box and box is None:
        log(f"box {iid} is not reachable through the dashboard API: gathering what is local")
    recs = {}
    if state and os.path.isfile(state):
        for r in json.load(open(state, encoding="utf-8")).get("runs", []):
            recs[r.get("part") or f"r{r['i']:02d}"] = r
    return [pull_run(vdir, r, box, recs.get(r.get("part")), log) for r in runs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pull", help="gather the ingredients of finished runs")
    a.add_argument("--iid", help="box instance id (dashboard API gives the ssh endpoint); omit with --no-box")
    a.add_argument("--version", required=True, help="PROJECTS target (COMPETITIONS/motion-graphic/<idea>/vNNN) or a version directory")
    a.add_argument("--run", help="one run, e.g. r01 (default: every run in out/)")
    a.add_argument("--no-box", action="store_true", help="local files only (destroyed box)")
    a.add_argument("--state", help="batch state.json (concept / design text of the runs when review/model_replies has none)")
    a.add_argument("--api", default=API)
    c = sub.add_parser("check", help="will the dashboard collapse this ingredients folder into one stack tile?")
    c.add_argument("folder")
    x = ap.parse_args(argv)
    if x.cmd == "check":
        print(json.dumps(collapse_check(x.folder), indent=1))
        return 0
    res = pull(x.version, x.iid, x.run, x.no_box, x.state, x.api)
    return 0 if res else 1


if __name__ == "__main__":
    raise SystemExit(main())
