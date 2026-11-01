"""Module library: render the library clip of any registered module, plus delivery helpers used by chain tests.

  python -m mograph.modules.library lanes
  python -m mograph.modules.library pipework --words "PIPEWORK" --font anton --entry left --exit right
  python -m mograph.modules.library lanes --check-dirs        # direction check sheet (up/down, 3D vectors, angle)
  python -m mograph.modules.library twist                      # transition clip: straight ribbon in, twisted, out
  python -m mograph.modules.library --list

Clip = centred placement, fixed demo camera, ~4 s (form from the left, hold ~1 s, exit to the right) + a contact sheet.
Outputs go to 14_modules/v002/local/ as <id>_gNN_draft.mp4 / <id>_sheet_gNN.jpg / <id>_dirs_gNN.jpg (never overwritten,
lineage appended to GENERATIONS.md). Manifest: with --batch X (or env MOGRAPH_BATCH=X) the entry is written to
14_modules/library/manifest.d/X.json (manifest.json is the base and is never touched by phase-2 batches); with no batch
nothing is published (a note is printed). build_library.py merges manifest.json + manifest.d/*.json."""
import os, sys, json, argparse, subprocess, time
import numpy as np
import cv2

from mograph.modules import core
from mograph.modules.core import Placement, Timing, Port, list_modules, get_module
from mograph.modules.sequence import Sequence
from mograph.render import cam3d

ol = cam3d.ol
ROOT = cam3d.ROOT
VERSION = "14_modules/v002"
OUT = ROOT + VERSION + "/local/"
MANIFEST = ROOT + "14_modules/library/manifest.json"           # base manifest (phase 1)
MANIFEST_D = ROOT + "14_modules/library/manifest.d/"           # one json per batch (phase 2)
BATCH = os.environ.get("MOGRAPH_BATCH") or None


def manifest_path(batch=None):
    """the manifest file a publish goes to: manifest.d/<batch>.json, or None when no batch is set."""
    b = batch or BATCH
    return None if not b else MANIFEST_D + b + ".json"
W, H, FPS = 960, 540, 30


def _setup():
    os.makedirs(OUT, exist_ok=True)
    ol.OUTDIR, ol.GEN = OUT, OUT + "GENERATIONS.md"
    if not os.path.exists(ol.GEN):
        open(ol.GEN, "w", encoding="utf-8").write("# 14_modules v002 local GENERATIONS\n\n")


def next_gen(stem, kinds=("draft.mp4",)):
    g = 1
    while any(os.path.exists(f"{OUT}{stem}_g{g:02d}_{k}") for k in kinds):
        g += 1
    return g


# ------------------------------------------------------------------ sheets / checks
def decode(path, w=W, h=H):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def tile(im, label, w=320):
    t = cv2.resize(np.ascontiguousarray(im), (w, int(w * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(t, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 148, 140), 1, cv2.LINE_AA)
    cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (70, 68, 64), 1)
    return t


def grid(tiles, cols):
    rows = []
    for r in range(0, len(tiles), cols):
        row = list(tiles[r:r + cols])
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def sheet_from_frames(fr, out, step=6, cols=6, fps=FPS):
    tiles = [tile(fr[i], f"f{i} {i / fps:.2f}s") for i in range(0, len(fr), step)]
    cv2.imwrite(out, grid(tiles, cols)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def checks(fr, nf):
    F = fr.astype(np.int16)
    D = np.array([np.abs(F[i] - F[i - 1]).mean() for i in range(1, len(F))])
    blank = [i for i in range(len(F)) if F[i].std() < 0.5]
    out = [f"frames {len(F)} (expected {nf})",
           f"frame diff: mean {D.mean():.2f} median {np.median(D):.2f} p95 {np.percentile(D, 95):.2f} max {D.max():.2f} at f{D.argmax() + 1}",
           f"blank frames (flat ground): {len(blank)}" + (f" first {blank[:3]} last {blank[-3:]}" if blank else "")]
    return "\n".join(out)


# ------------------------------------------------------------------ manifest
def publish(id, kind, title, description, clip, sheet, dircheck, gen, version=VERSION, batch=None):
    """add / update one entry of 14_modules/library/manifest.d/<batch>.json (batch = argument, else module BATCH / env
    MOGRAPH_BATCH; no batch: nothing is written). Other entries untouched. Paths relative to the repo root."""
    rel = lambda p: None if p is None else os.path.relpath(p, ROOT).replace("\\", "/")
    ent = dict(id=id, kind=kind, title=title, description=description, clip=rel(clip), sheet=rel(sheet),
               dircheck=rel(dircheck), version=version, gen=int(gen))
    mp = manifest_path(batch)
    if mp is None:
        print("publish: no batch set (--batch X / MOGRAPH_BATCH); manifest not written for", id)
        return ent
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    data = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else []
    for i, e in enumerate(data):
        if e["id"] == id:
            if ent["dircheck"] is None:
                ent["dircheck"] = e.get("dircheck")           # a clip re-render keeps its direction sheet
            data[i] = ent
            break
    else:
        data.append(ent)
    json.dump(data, open(mp, "w", encoding="utf-8"), indent=2)
    return ent


def latest_gen_file(stem_kind):
    """newest existing path of the gNN series, e.g. latest_gen_file('lanes_dirs_g{:02d}.jpg')"""
    g, last = 1, None
    while os.path.exists(OUT + stem_kind.format(g)):
        last = OUT + stem_kind.format(g)
        g += 1
    return last


# ------------------------------------------------------------------ delivery
def deliver(name, seq, camera=None, parent="-", change="", why="", sheet_step=6, sheet_cols=6, tail=None, procs=8, sub=3):
    """render <name>_gNN_draft.mp4 + <name>_sheet_gNN.jpg + checks + lineage line. -> dict(gen, mp4, sheet)."""
    _setup()
    g = next_gen(name)
    stem = f"{name}_g{g:02d}_draft"
    mp4 = OUT + stem + ".mp4"
    secs, nf = seq.render(mp4, camera=camera, fps=FPS, size=(W, H), S=2, procs=procs, sub=sub, tail=tail)
    fr = decode(mp4)
    sheet = sheet_from_frames(fr, f"{OUT}{name}_sheet_g{g:02d}.jpg", sheet_step, sheet_cols)
    rep = checks(fr, nf)
    st = ol.LAST_STATS
    sts = f"frame-diff(q-res) mean {st.mean():.2f} p95 {np.percentile(st, 95):.2f} max {st.max():.2f}"
    ol.log_gen(stem, parent, change + " [" + sts + "]", why, secs)
    open(f"{OUT}{name}_checks_g{g:02d}.txt", "w", encoding="utf-8").write(rep + "\n")
    print(stem, f"render {secs:.1f}s", mp4)
    print("sheet", sheet)
    print(rep)
    return dict(gen=g, mp4=mp4, sheet=sheet, secs=secs, report=rep)


# ------------------------------------------------------------------ direction check
DIR_CASES = [("entry up / exit down", "up", "down"),
             ("entry 3D (-0.6,-0.3,0.8) from behind upper-left / exit 3D (0.5,0.3,-0.6) toward the viewer, lower right",
              [-0.6, -0.3, 0.8], [0.5, 0.3, -0.6]),
             ("entry angle 35 deg / exit angle 215 deg", 35, 215)]


def dir_times(m):
    t = m.timing
    return [0.28 * t.form, 0.6 * t.form, t.form + 0.5 * t.hold, t.form + t.hold + 0.3 * t.exit,
            t.form + t.hold + 0.62 * t.exit, t.form + t.hold + t.exit + 0.1, t.form + t.hold + t.exit + 0.35]


def check_dirs(name, words=None, font=None, timing=None, palette=None, **params):
    """small sheet of the same module with other entry / exit directions (one row per case)."""
    _setup()
    cls = get_module(name)
    rows = []
    for label, entry, exit_ in DIR_CASES:
        m = cls(words, font=font, entry=entry, exit=exit_, timing=timing or Timing(1.5, 1.0, 1.3), palette=palette, **params)
        seq = Sequence([m], tail=0.5)
        tiles = []
        for k, t in enumerate(dir_times(m)):
            tiles.append(tile(seq.frame(t), f"{t:.2f}s" + (f"  {label}" if k == 0 else ""), 320))
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    g = 1
    while os.path.exists(f"{OUT}{name}_dirs_g{g:02d}.jpg"):
        g += 1
    out = f"{OUT}{name}_dirs_g{g:02d}.jpg"
    cv2.imwrite(out, sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    ol.log_gen(f"{name}_dirs_g{g:02d}", "-", "direction check sheet: " + "; ".join(c[0] for c in DIR_CASES),
               "prove entry / exit direction handling (up/down, 3D vectors, angle)", 0)
    print("dirs sheet", out)
    return out, g


# ------------------------------------------------------------------ library clips
def title_of(name, cls):
    return name.replace("_", " ").capitalize()


def render_clip(name, words=None, font=None, entry="left", exit="right", timing=None, publish_it=True, **params):
    _setup()
    cls = get_module(name)
    if cls.kind == "transition":
        a, b = core.TransitionModule.demo_bundles()
        tp = dict(cls.DEMO_PARAMS)
        tp.update(params)
        tr = cls(a, b, dur=float(tp.pop("dur", cls.DEMO_DUR)), **tp)
        seq = Sequence([tr], bare=True, tail=0.0)
        info = deliver(name, seq, camera=cam3d.Camera.home(),
                       change=f"library clip: straight tube ribbon in from the left, {name} ({tp}), out to the right",
                       why=f"library clip of the {name} transition")
    else:
        m = cls(words, font=font, entry=entry, exit=exit, timing=timing or Timing(1.5, 1.0, 1.3),
                placement=Placement.centred(), **params)
        seq = Sequence([m], tail=0.5)
        info = deliver(name, seq, camera=None,
                       change=f"library clip: {m.words!r} font {m.font!r}, centred placement, entry {m.entry} exit {m.exit}, {m.timing}, N={getattr(m, 'N', '?')} strands",
                       why=f"library clip of the {name} word module")
    if publish_it:
        d = latest_gen_file(f"{name}_dirs_g{{:02d}}.jpg")
        publish(name, cls.kind, title_of(name, cls), cls.description, info["mp4"], info["sheet"], d, info["gen"])
    return info


def _parse_port(s):
    return None if s is None else Port.coerce(s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mograph.modules.library")
    ap.add_argument("module", nargs="?")
    ap.add_argument("--words")
    ap.add_argument("--font")
    ap.add_argument("--entry", default="left")
    ap.add_argument("--exit", default="right")
    ap.add_argument("--check-dirs", action="store_true", help="render the direction-check sheet instead of the clip")
    ap.add_argument("--timing", help="form,hold,exit seconds (default 1.5,1.0,1.3)")
    ap.add_argument("--batch", help="manifest batch name -> manifest.d/<batch>.json (default env MOGRAPH_BATCH; none = no manifest write)")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.module:
        print("modules:", ", ".join(f"{n} ({get_module(n).kind})" for n in list_modules()))
        return
    global BATCH
    if a.batch:
        BATCH = a.batch
    tm = Timing(*[float(x) for x in a.timing.split(",")]) if a.timing else None
    words = a.words.replace("\\n", "\n") if a.words else None
    if a.check_dirs:
        out, g = check_dirs(a.module, words, a.font, tm)
        mp = manifest_path()
        if not a.no_manifest and mp and os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            e = next((e for e in m if e["id"] == a.module), None)
            if e:
                e["dircheck"] = os.path.relpath(out, ROOT).replace("\\", "/")
                json.dump(m, open(mp, "w", encoding="utf-8"), indent=2)
        elif not a.no_manifest:
            print("note: direction sheet not linked into a manifest (no batch file yet for", a.module + "); rendering the clip afterwards picks it up")
    else:
        render_clip(a.module, words, a.font, a.entry, a.exit, tm, publish_it=not a.no_manifest)


if __name__ == "__main__":
    main()
