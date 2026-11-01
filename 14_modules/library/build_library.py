#!/usr/bin/env python3
"""Build the "Module Library" review page as a static folder.

One page showing every lego type module: chains (several modules linked, camera on top), word
modules and transition modules. Each card is a figure keyed by its clip file name, so the shared
review snippets (likes/notes in db collection `reviews`, overall box + past overall reviews) work
exactly like the review hub. Media helpers (web_video/web_image, probing, size limits) and the two
snippets are reused from ../../12_motionflow/review_hub (build_hub.py and its _parts/), not forked.

Output (next to this script): index.html (an HTML FRAGMENT for the Artifact tool, like the hub) and
web/ (media). Re-runnable: web files are only re-encoded when the source is newer; web files no
longer referenced are removed. Stdlib + ffmpeg/ffprobe only.

    python build_library.py [--round "Module library v001"] [--manifest path/to/manifest.json]

Manifest: library/manifest.json is the base; every library/manifest.d/*.json (one per work batch, sorted by name) is merged
on top. Entries are deduped by id: the later version wins (version "14_modules/v002" > "14_modules/v001", then the higher
gen; on a tie the later file wins). Use --no-merge for the base alone. Each file is a JSON list of {id, kind: word|transition|chain|camera, title, description, clip, sheet,
dircheck (or null), version, gen}; paths are relative to the motion-graphic folder. Default is
library/manifest.json; if that does not exist yet, manifest.sample.json is used (with a warning).
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MG = HERE.parent.parent                                   # .../motion-graphic
HUB_DIR = MG / "12_motionflow" / "review_hub"
sys.path.insert(0, str(HUB_DIR))
import build_hub as bh                                    # noqa: E402  (reuse, don't fork)

WEB = HERE / "web"
bh.WEB = WEB                                              # the hub helpers write to bh.WEB at call time
bh.web_used.clear()
bh.web_src.clear()
esc = bh.esc
TITLE = "Module Library"
DEFAULT_MANIFEST = HERE / "manifest.json"
MANIFEST_D = HERE / "manifest.d"
SAMPLE_MANIFEST = HERE / "manifest.sample.json"
KINDS = ("chain", "word", "style", "transition", "camera")

EXTRA_STYLE = """<style>
.intro{display:flex;flex-direction:column;gap:8px;margin-top:10px}
.intro p{margin:0;max-width:84ch;color:var(--muted)} .intro b{color:var(--fg);font-weight:400}
.mod-tag{display:inline-block;align-self:flex-start;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#111;background:var(--accent);padding:1px 6px}
.mod-tag.chain{background:#8fb8a8} .mod-tag.transition{background:#b79ad6} .mod-tag.camera{background:#e0b46a} .mod-tag.style{background:#d6b79a}
figure details{margin-top:6px}
figure details img{width:100%;height:auto;display:block;margin-top:8px;background:#000}
.soon{border:1px dashed var(--line);padding:14px;color:var(--muted)}
@media (max-width:600px){.wrap{padding-top:14px;gap:20px} h2{font-size:26px}}
</style>"""

INTRO = (
    '<div class="intro">'
    '<p><b>Word module.</b> Forms any words from nothing: material arrives from an entry direction '
    '(left by default), holds, then deconstructs and leaves along an exit direction (right by default). '
    'It works from any direction and at any position and angle. Here each one is shown centred so you can '
    'see how it works.</p>'
    '<p><b>Form-in / form-out style.</b> How a word module materialises and dissolves is its own part: the body only says what '
    'the word looks like; the style (lines, blobs, flicker, particles, wipe) says how it arrives and leaves. Any style works on any '
    'body, and each style hands the next module its own material (strands, points, blobs, or nothing, in which case the hand-off '
    'becomes a cut).</p>'
    '<p><b>Transition module.</b> Takes the strands one module releases and routes them to the next '
    'through an animation (a twist, a circuit board, a spiral around an object).</p>'
    '<p><b>Camera module.</b> One camera move (follow, orbit, spin-read-spin, push-through, tilt-to-plan, graze, eased reversal, '
    'pull-back reveal, hold drift) that works over the top of any chain. Every clip here is the same test chain '
    '(SIGNAL, twist, BENDS, circuit board, FLOW) so the moves compare directly; modules join with continuous velocity and acceleration.</p>'
    '<p><b>Chains</b> show modules and transitions snapped together, with the camera on top.</p>'
    '</div>')

SECTIONS = [
    ("chain", "Chains", "Several modules linked: word, transition, word, with the camera working over the whole run."),
    ("word", "Word modules", "One card per module: library clip (form from the left, hold, exit right), then the contact sheet and the direction check."),
    ("style", "Form-in / form-out styles", "One card per style, all on the same body (lanes, FORM): the style forms the word, it holds, the style takes it away. Styles work on any word module."),
    ("transition", "Transition modules", "One card per transition: it carries the strands released by one word module into the next."),
    ("camera", "Camera modules", "One card per camera move, all on the same test chain: the move is applied where it makes sense, the default camera (follow through transitions, composed holds on words) elsewhere. Speed numbers in each description."),
]


def _read_list(p):
    try:
        data = json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        sys.exit(f"cannot read manifest {p}: {e}")
    if isinstance(data, dict):
        data = data.get("modules") or data.get("entries") or []
    if not isinstance(data, list):
        sys.exit(f"manifest {p} must be a JSON list")
    return data


def _vkey(e):
    m = re.search(r"(\d+)\s*$", str(e.get("version") or ""))
    try:
        g = int(e.get("gen") or 0)
    except (TypeError, ValueError):
        g = 0
    return (int(m.group(1)) if m else 0, g)


def merge_entries(layers):
    """layers: [(source name, [entries])] in priority order (base first). Dedupe by id, later version wins (version number,
    then gen, then the later source). The first occurrence keeps its position. -> (entries, notes)."""
    out, pos, notes = [], {}, []
    for src, entries in layers:
        for e in entries:
            i = e.get("id")
            if i is None:
                out.append(e)
                continue
            if i not in pos:
                pos[i] = len(out)
                out.append(dict(e, _src=src))
                continue
            old = out[pos[i]]
            if _vkey(e) >= _vkey(old):
                notes.append(f"{i}: {src} ({e.get('version')} g{e.get('gen')}) replaces {old['_src']} ({old.get('version')} g{old.get('gen')})")
                out[pos[i]] = dict(e, _src=src)
            else:
                notes.append(f"{i}: {src} ({e.get('version')} g{e.get('gen')}) is older than {old['_src']}; ignored")
    for e in out:
        e.pop("_src", None)
    return out, notes


def load_manifest(arg, merge=True):
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            sys.exit(f"manifest not found: {p}")
        merge = False
    elif DEFAULT_MANIFEST.exists():
        p = DEFAULT_MANIFEST
    elif SAMPLE_MANIFEST.exists():
        bh.warn(f"{DEFAULT_MANIFEST.name} not found; using the SAMPLE manifest {SAMPLE_MANIFEST.name}")
        p = SAMPLE_MANIFEST
        merge = False
    else:
        sys.exit(f"no manifest ({DEFAULT_MANIFEST} or {SAMPLE_MANIFEST})")
    layers = [(p.name, _read_list(p))]
    if merge and MANIFEST_D.is_dir():
        for f in sorted(MANIFEST_D.glob("*.json")):
            layers.append((f"manifest.d/{f.name}", _read_list(f)))
    data, notes = merge_entries(layers)
    print("manifest:", p, "(" + " + ".join(f"{n}: {len(l)}" for n, l in layers) + f" -> {len(data)} entries)")
    for n in notes:
        print("  merge:", n)
    return data, p


def rel(path):
    return (MG / path) if path else None


def kind_of(e):
    return e.get("kind") if e.get("kind") in KINDS else "word"


def card(e):
    kind = kind_of(e)
    title = e.get("title") or e.get("id") or "(untitled)"
    clip = rel(e.get("clip"))
    if not clip or not clip.exists():
        bh.warn(f"{e.get('id')}: clip not found ({e.get('clip')}); placeholder")
        return bh.pending(title, "clip not rendered yet")
    name = bh.web_video(clip)
    if not name:
        return bh.pending(title, "clip not a finished mp4 yet")
    stamp = " · ".join(str(x) for x in (e.get("version"), e.get("gen")) if x)
    fig = bh.fig_video(name, title, [e.get("description"), stamp])
    fig = fig.replace("<figcaption>", f'<figcaption><span class="mod-tag {kind}">{kind}</span>', 1)
    stem = Path(name).stem
    extra = ""
    for field, label, suffix in (("sheet", "Contact sheet", "sheet"), ("dircheck", "Direction check", "dircheck")):
        src = rel(e.get(field))
        if not src:
            continue
        if not src.exists():
            bh.warn(f"{e.get('id')}: {field} not found ({e.get(field)})")
            continue
        if src.suffix.lower() in (".mp4", ".webm", ".mov"):
            bh.warn(f"{e.get('id')}: {field} is a video; only images are supported, skipped")
            continue
        iname = bh.web_image(src, f"{stem}__{suffix}.jpg")
        if iname:
            extra += (f'<details><summary>{label}</summary><img class="sheet" src="web/{esc(iname)}" '
                      f'loading="lazy" alt="{label.lower()} {esc(title)}"></details>')
    return fig[:-len("</figure>")] + extra + "</figure>"


def build_sections(entries):
    out, counts = [], {}
    seen = set()
    for kind, heading, sub in SECTIONS:
        cells = []
        for e in entries:
            if kind_of(e) != kind:
                continue
            key = Path(e.get("clip") or "").name
            if key in seen:
                bh.warn(f"{e.get('id')}: duplicate clip file name {key}; skipped (data-key must be unique)")
                continue
            seen.add(key)
            cells.append(card(e))
        counts[kind] = len(cells)
        inner = bh.grid(cells) if cells else '<div class="soon">coming next</div>'
        out.append(bh.sec(heading, inner, sub))
    return out, counts


def verify(page):
    problems = []
    if not page.startswith(f"<title>{TITLE}</title>"):
        problems.append(f"fragment must start with <title>{TITLE}</title>")
    p = bh.Tags()
    p.feed(page)
    p.close()
    problems += p.bad
    if p.stack:
        problems.append(f"unclosed tags: {p.stack}")
    srcs = re.findall(r'\bsrc="web/([^"]+)"', page)
    missing = [s for s in srcs if not (WEB / html.unescape(s)).exists()]
    if missing:
        problems.append(f"missing src: {missing}")
    keys = re.findall(r'<figure data-key="([^"]+)"', page)
    if len(keys) != len(re.findall(r"<figure\b", page)):
        problems.append("figure without data-key")
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        problems.append(f"duplicate data-keys: {dups}")
    vids = re.findall(r"<video[^>]*>", page)
    nofps = [t[:90] for t in vids if 'data-fps="' not in t or 'data-frames="' not in t]
    if nofps:
        problems.append(f"videos without data-fps/data-frames: {nofps}")
    for h in ("Chains", "Word modules", "Form-in / form-out styles", "Transition modules", "Camera modules"):
        if f"<h2>{h}</h2>" not in page:
            problems.append(f"section missing: {h}")
    print(f"videos: {len(vids)} total, {len(vids) - len(nofps)} with data-fps/data-frames, {len(nofps)} missing")
    return problems, len(keys), len(srcs)


def main():
    ap = argparse.ArgumentParser(description="Build the Module Library index.html")
    ap.add_argument("--round", dest="round_label", default="Module library v001",
                    help='round label shown in the overall panel heading (default "Module library v001")')
    ap.add_argument("--manifest", default=None,
                    help="manifest JSON, used alone (default library/manifest.json + manifest.d/*.json merged, else manifest.sample.json with a warning)")
    ap.add_argument("--no-merge", action="store_true", help="base manifest only, ignore manifest.d/")
    args = ap.parse_args()
    WEB.mkdir(exist_ok=True)
    for t in WEB.glob("_tmp_*"):
        t.unlink(missing_ok=True)
    entries, mpath = load_manifest(args.manifest, merge=not args.no_merge)
    sections, counts = build_sections(entries)

    header = (f'<header><h1>{TITLE}</h1>' + INTRO +
              '<p class="muted">Like what works, note what to change. Open a card\'s contact sheet or direction check below its clip.</p></header>')
    history = ('<section class="grp" id="ov-history"><h2>Past overall reviews</h2>'
               '<p class="muted">Earlier rounds\' overall notes, newest first. Read-only; the box on the side is for the current round.</p>'
               '<div class="ovh-list"><p class="muted ovh-empty">Loading…</p></div></section>')
    snippet_rv = bh.load_part(bh.PART_NAMES[0])
    snippet_ov = bh.load_part(bh.PART_NAMES[1])
    page = (f"<title>{TITLE}</title>\n" + bh.HEAD_STYLE + "\n" + EXTRA_STYLE + "\n"
            f'<div class="wrap" data-ov-round="{esc(args.round_label.strip())}">' + header + "\n"
            + "\n".join(sections) + "\n" + history + "</div>\n" + snippet_rv + "\n" + snippet_ov + "\n")
    (HERE / "index.html").write_text(page, encoding="utf8")

    for f in sorted(WEB.iterdir()):
        if f.is_file() and f.name not in bh.web_used:
            print("removed stale web file:", f.name)
            f.unlink()

    problems, nfig, nsrc = verify(page)
    files = sorted(f for f in WEB.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in files)
    if total > bh.MAX_TOTAL:
        problems.append(f"web/ is {total / 1e6:.1f} MB, over the 60 MB limit")
    print(f"overall panel round: {args.round_label!r}")
    print(json.dumps({"index": str(HERE / "index.html"), "manifest": str(mpath), "items_per_section": counts,
                      "figures": nfig, "media_refs": nsrc, "warnings": len(bh.warnings),
                      "problems": problems, "web_total_mb": round(total / 1e6, 2)}, indent=1))
    print("web files (publisher list):")
    print(json.dumps([{"path": f"web/{f.name}"} for f in files]))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
