#!/usr/bin/env python3
"""Build the "Module Gallery": a simple one-page gallery of the lego type modules (words, chains, form styles).

No review UI, no notes. A strip of small square looping previews at the top (each with a heart, hearted ones first),
one main player below for the selected module (hash route #<id>). Hearts persist in the artifact database
(collection `favs`, doc id = module id) and fall back to localStorage when window.claude is not available.
Each module has a one-click copy of its exact reference string "<id> · <version> g<NN>" (main page and each preview).

Entries come from ../library (manifest.json + manifest.d/*.json) through build_library.load_manifest, so the merge/dedupe
rules are exactly the library's (imported, not copied). Kinds shown: chain, word, style. Output (next to this script):
index.html (an HTML FRAGMENT for the Artifact tool, like the library/hub) and web/ (media).

    python build_gallery.py

Stdlib + ffmpeg/ffprobe only. Re-runnable: media is only rebuilt when the source (or this script, for previews) is newer;
web files no longer referenced are removed.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MG = HERE.parent.parent                                   # .../motion-graphic
LIB = MG / "14_modules" / "library"
sys.path.insert(0, str(LIB))
import build_library as bl                                # noqa: E402  (reuse the manifest merge, don't fork)

WEB = HERE / "web"
LIB_WEB = LIB / "web"
TITLE = "Module Gallery"
KIND_ORDER = ("chain", "word", "style")                   # strip order when nothing is hearted
KIND_LABEL = {"chain": "chain", "word": "word", "style": "form style"}
PREV_SIZE = 240
PREV_SECONDS = 2.5
PREV_FPS = 20
PREV_MAX = 150 * 1024
MAX_VIDEO = 4 * 1024 * 1024
MAX_TOTAL = 60 * 1024 * 1024
warnings = []


def warn(msg):
    warnings.append(msg)
    print("WARN:", msg)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def duration(p):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def fresh(src, dst, *extra):
    if not (dst.exists() and dst.stat().st_size > 0):
        return False
    t = dst.stat().st_mtime
    return all(t >= s.stat().st_mtime for s in (src,) + extra)


def reference(e):
    """'<id> · <version> g<NN>' from the manifest entry (gen zero-padded to 2)."""
    ref = str(e["id"])
    if e.get("version"):
        ref += f" · {e['version']}"
        try:
            ref += f" g{int(e.get('gen')):02d}"
        except (TypeError, ValueError):
            pass
    return ref


# ---- main clip ------------------------------------------------------------------------------------------------
def main_clip(src):
    """web/<clip name>: reuse the library's encoded copy when it exists and is not older than the source, else transcode."""
    name = src.name
    dst = WEB / name
    lib = LIB_WEB / name
    if lib.exists() and lib.stat().st_size > 0 and lib.stat().st_mtime >= src.stat().st_mtime:
        if not fresh(lib, dst):
            shutil.copy2(lib, dst)
        return name, "library copy"
    if fresh(src, dst):
        return name, "kept"
    tmp = WEB / ("_tmp_" + name)
    for crf in (26, 30):
        r = run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                 "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", str(tmp)])
        if r.returncode != 0:
            warn(f"{name}: ffmpeg failed: {r.stderr.decode(errors='replace').strip()[:200]}")
            tmp.unlink(missing_ok=True)
            return None, "failed"
        if tmp.stat().st_size <= MAX_VIDEO:
            break
    if tmp.stat().st_size > MAX_VIDEO:
        warn(f"{name}: {tmp.stat().st_size / 1e6:.1f} MB even at crf 30")
    tmp.replace(dst)
    return name, "transcoded"


# ---- preview --------------------------------------------------------------------------------------------------
def pick_start(src, dur):
    """Start time (s) of the PREV_SECONDS window with the most going on: material on screen (ink) plus motion.
    Samples the centre square at 8 fps, 32x32 gray."""
    if dur <= PREV_SECONDS + 0.2:
        return 0.0
    sfps, n = 8, 32 * 32
    r = run(["ffmpeg", "-v", "error", "-i", str(src), "-vf",
             f"fps={sfps},crop=min(iw\\,ih):min(iw\\,ih),scale=32:32:flags=area,format=gray", "-f", "rawvideo", "-"])
    raw = r.stdout
    frames = [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)]
    w = int(PREV_SECONDS * sfps)
    if len(frames) <= w:
        return 0.0
    allpix = sorted(b for f in frames for b in f)
    bg = allpix[len(allpix) // 2]                         # background level = median pixel of the clip
    ink = [sum(max(0, b - bg - 20) for b in f) / n / 255 for f in frames]
    mot = [0.0] + [sum(abs(a - b) for a, b in zip(f, p)) / n / 255 for p, f in zip(frames, frames[1:])]
    mi, mm = max(ink) or 1.0, max(mot) or 1.0
    score = [ink[i] / mi + mot[i] / mm for i in range(len(frames))]
    best, best_i = -1.0, 0
    for i in range(0, len(frames) - w + 1):
        s = sum(score[i:i + w])
        if s > best:
            best, best_i = s, i
    return best_i / sfps


def preview(src, eid):
    name = f"{eid}_p.mp4"
    dst = WEB / name
    if fresh(src, dst, Path(__file__).resolve()):
        return name
    dur = duration(src)
    if dur <= 0:
        warn(f"{eid}: cannot read duration of {src.name}")
        return None
    t0 = pick_start(src, dur)
    tmp = WEB / ("_tmp_" + name)
    vf = f"crop=min(iw\\,ih):min(iw\\,ih),scale={PREV_SIZE}:{PREV_SIZE}:flags=lanczos,fps={PREV_FPS},format=yuv420p"
    for crf in (23, 27, 31, 35, 40):
        r = run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.2f}", "-t", str(PREV_SECONDS), "-i", str(src), "-an", "-vf", vf,
                 "-c:v", "libx264", "-profile:v", "main", "-preset", "slow", "-crf", str(crf),
                 "-movflags", "+faststart", str(tmp)])
        if r.returncode != 0:
            warn(f"{eid}: preview ffmpeg failed: {r.stderr.decode(errors='replace').strip()[:200]}")
            tmp.unlink(missing_ok=True)
            return None
        if tmp.stat().st_size <= PREV_MAX:
            break
    if tmp.stat().st_size > PREV_MAX:
        warn(f"{eid}: preview {tmp.stat().st_size / 1024:.0f} KB even at crf 40")
    tmp.replace(dst)
    print(f"  preview {eid}: start {t0:.2f}s, {dst.stat().st_size / 1024:.0f} KB")
    return name


# ---- page -----------------------------------------------------------------------------------------------------
PAGE = r"""<title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0c0c0b;--fg:#ece6da;--muted:#9d968a;--line:#2b2a26;--accent:#d9a441;--heart:#ff6f61;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f5f0e6;--fg:#1d1c19;--muted:#6c675d;--line:#d9d2c3;--accent:#a8721a;--heart:#e5483a;color-scheme:light}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--fg);font:13px/1.55 "IBM Plex Mono",ui-monospace,Consolas,monospace;margin:0}
.mg{max-width:1100px;margin:0 auto;padding:20px 16px 64px;display:flex;flex-direction:column;gap:22px;min-width:0}
.mg h1{font:800 clamp(30px,7vw,46px)/1 "Big Shoulders Display","Arial Narrow",Impact,sans-serif;text-transform:uppercase;margin:0;letter-spacing:.01em}
.strip{display:flex;flex-wrap:wrap;gap:10px 8px;min-width:0}
.tile{position:relative;width:118px;flex:0 0 auto}
.thumb{display:block;color:inherit;text-decoration:none;outline-offset:3px}
.thumb .sq{display:block;width:118px;height:118px;background:#000;overflow:hidden;border:2px solid transparent}
.thumb video{display:block;width:100%;height:100%;object-fit:cover}
.thumb .lb{display:block;margin-top:4px;padding-right:22px;font-size:10px;line-height:1.25;color:var(--muted);height:25px;overflow:hidden;overflow-wrap:anywhere}
.tile:hover .lb,.tile.cur .lb{color:var(--fg)}
.tile.cur .sq{border-color:var(--accent)}
.heart{display:inline-flex;align-items:center;justify-content:center;border:0;background:transparent;padding:0;margin:0;cursor:pointer;color:var(--fg);-webkit-tap-highlight-color:transparent}
.heart svg{display:block;width:100%;height:100%;fill:none;stroke:currentColor;stroke-width:2;stroke-linejoin:round}
.heart[aria-pressed="true"] svg{fill:var(--heart);stroke:var(--heart)}
.heart:focus-visible,.cp:focus-visible,.copy:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.tile .heart{position:absolute;top:4px;right:4px;width:32px;height:32px;color:#fff;align-items:flex-start;justify-content:flex-end;padding:4px;background:radial-gradient(circle at 19px 13px,rgba(0,0,0,.5) 0 11px,transparent 12px)}
.tile .heart svg{width:18px;height:18px;filter:drop-shadow(0 0 2px rgba(0,0,0,.85))}
.tile .heart[aria-pressed="true"] svg{stroke:#fff;stroke-width:1.5}
.cp,.copy{border:0;background:transparent;padding:0;margin:0;cursor:pointer;color:var(--muted);font:inherit;-webkit-tap-highlight-color:transparent}
.cp:hover,.copy:hover{color:var(--accent)}
.cp{position:absolute;right:0;bottom:-2px;width:28px;height:28px;display:flex;align-items:flex-end;justify-content:flex-end;padding:0 2px 6px 0}
.cp svg,.copy svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;flex:0 0 auto}
.ck,.copy.done .cc,.cp.done .cc{display:none}
.copy.done .ck,.cp.done .ck{display:block}
.cp.done,.copy.done{color:var(--accent)}
.main{display:flex;flex-direction:column;gap:12px;min-width:0;scroll-margin-top:12px}
.main video{width:100%;height:auto;display:block;background:#000;aspect-ratio:16/9}
.head{display:flex;align-items:center;gap:12px;min-width:0}
.head h2{font:800 clamp(24px,5.5vw,34px)/1.05 "Big Shoulders Display","Arial Narrow",Impact,sans-serif;text-transform:uppercase;margin:0;color:var(--accent);flex:1 1 auto;min-width:0;overflow-wrap:anywhere}
.head .heart{width:36px;height:36px;padding:5px;flex:0 0 auto}
.copy{display:inline-flex;align-items:center;gap:7px;align-self:flex-start;max-width:100%;font-size:11px;line-height:1.4;text-align:left;margin-top:-6px;min-height:24px}
.copy .ref{overflow-wrap:anywhere;min-width:0}
.copy .msg{display:none}
.copy.done .msg{display:inline}
.tag{align-self:flex-start;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#111;background:var(--accent);padding:1px 7px}
.desc{margin:0;max-width:78ch;color:var(--muted)}
.nav{display:flex;gap:10px}
.nav button{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);padding:8px 14px;cursor:pointer;min-height:40px}
.nav button:hover{border-color:var(--accent);color:var(--accent)}
@media (max-width:700px){
  .mg{padding-top:14px;gap:18px}
  .strip{flex-wrap:nowrap;overflow-x:auto;overflow-y:hidden;padding-bottom:8px;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain}
}
</style>
<div class="mg" id="mg">
<h1>__TITLE__</h1>
<nav class="strip" id="strip" aria-label="Modules"></nav>
<section class="main" id="main" aria-live="polite">
<video id="mv" controls autoplay muted loop playsinline preload="auto"></video>
<div class="head"><h2 id="mt"></h2><button class="heart" id="mh" type="button" aria-pressed="false" aria-label="Heart this module">__HEART__</button></div>
<button class="copy" id="mc" type="button" title="Copy this reference"><span class="ref" id="mr"></span>__COPYICON__<span class="msg">copied</span></button>
<span class="tag" id="mk"></span>
<p class="desc" id="md"></p>
<div class="nav"><button type="button" id="pv" aria-label="Previous module">&larr; prev</button><button type="button" id="nx" aria-label="Next module">next &rarr;</button></div>
</section>
</div>
<script type="application/json" id="gdata">__DATA__</script>
<script>
(function () {
  var ITEMS = JSON.parse(document.getElementById("gdata").textContent);
  var byId = {}; ITEMS.forEach(function (it) { byId[it.id] = it; });
  var HEART = '__HEART__';
  var COPYICON = '__COPYICON__';
  var strip = document.getElementById("strip");
  var mv = document.getElementById("mv"), mt = document.getElementById("mt"), mk = document.getElementById("mk"),
      md = document.getElementById("md"), mh = document.getElementById("mh"), mainEl = document.getElementById("main"),
      mc = document.getElementById("mc"), mr = document.getElementById("mr");
  var favs = {};            // id -> true
  var pending = {};         // id -> bool, local writes the database has not echoed yet
  var store = null;         // artifact db collection, or null -> localStorage
  var LS = "module-gallery-favs";
  var cur = null, locked = false, tiles = {}, fromTile = false;

  function lsLoad() { try { var a = JSON.parse(localStorage.getItem(LS) || "[]"); (a || []).forEach(function (i) { if (byId[i]) favs[i] = true; }); } catch (e) {} }
  function lsSave() { try { localStorage.setItem(LS, JSON.stringify(Object.keys(favs))); } catch (e) {} }

  function order() {
    var f = [], r = [];
    ITEMS.forEach(function (it) { (favs[it.id] ? f : r).push(it); });
    return f.concat(r);
  }
  function defaultId() {
    var o = order();
    if (o.length && favs[o[0].id]) return o[0].id;
    for (var i = 0; i < ITEMS.length; i++) if (ITEMS[i].kind === "word") return ITEMS[i].id;
    return ITEMS.length ? ITEMS[0].id : null;
  }
  function hashId() { var h = ""; try { h = decodeURIComponent((location.hash || "").replace(/^#/, "")); } catch (e) {} return byId[h] ? h : null; }

  // ---- copy
  function copyText(text) {
    function fallback() {
      try {
        var ta = document.createElement("textarea");
        ta.value = text; ta.setAttribute("readonly", "");
        ta.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none";
        document.body.appendChild(ta); ta.focus(); ta.select();
        try { ta.setSelectionRange(0, text.length); } catch (e) {}
        var ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return ok;
      } catch (e) { return false; }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try { return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return fallback(); }); }
      catch (e) { return Promise.resolve(fallback()); }
    }
    return Promise.resolve(fallback());
  }
  function copyBtn(btn, text) {
    copyText(text).then(function (ok) {
      if (ok === false) return;
      btn.classList.add("done");
      clearTimeout(btn._t); btn._t = setTimeout(function () { btn.classList.remove("done"); }, 1000);
    });
  }

  // ---- strip
  function buildStrip() {
    ITEMS.forEach(function (it) {
      var d = document.createElement("div"); d.className = "tile"; d.dataset.id = it.id;
      var a = document.createElement("a"); a.className = "thumb"; a.href = "#" + encodeURIComponent(it.id); a.title = it.title;
      var sq = document.createElement("span"); sq.className = "sq";
      var v = document.createElement("video");
      v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true; v.preload = "metadata";
      v.setAttribute("muted", ""); v.setAttribute("loop", ""); v.setAttribute("autoplay", ""); v.setAttribute("playsinline", ""); v.setAttribute("preload", "metadata");
      v.src = "web/" + it.preview; v.tabIndex = -1;
      sq.appendChild(v);
      var lb = document.createElement("span"); lb.className = "lb"; lb.textContent = it.title;
      a.appendChild(sq); a.appendChild(lb);
      a.addEventListener("click", function () { fromTile = true; });
      var h = document.createElement("button"); h.type = "button"; h.className = "heart"; h.innerHTML = HEART;
      h.setAttribute("aria-label", "Heart " + it.title);
      h.addEventListener("click", function (e) { e.stopPropagation(); e.preventDefault(); toggle(it.id); });
      var c = document.createElement("button"); c.type = "button"; c.className = "cp"; c.innerHTML = COPYICON;
      c.title = "Copy " + it.ref; c.setAttribute("aria-label", "Copy reference for " + it.title);
      c.addEventListener("click", function (e) { e.stopPropagation(); e.preventDefault(); copyBtn(c, it.ref); });
      d.appendChild(a); d.appendChild(h); d.appendChild(c); strip.appendChild(d);
      tiles[it.id] = { el: d, heart: h, video: v };
    });
  }
  function paintStrip() {
    order().forEach(function (it, i) {
      var t = tiles[it.id]; t.el.style.order = i;
      t.heart.setAttribute("aria-pressed", favs[it.id] ? "true" : "false");
      t.el.classList.toggle("cur", it.id === cur);
    });
  }

  // previews only play while on screen
  var io = null;
  function setupIO() {
    if (!("IntersectionObserver" in window)) { return; }
    io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else { v.pause(); }
      });
    }, { rootMargin: "80px" });
    Object.keys(tiles).forEach(function (id) { tiles[id].video.pause(); io.observe(tiles[id].video); });
  }

  // ---- main
  function paintMain() {
    var it = byId[cur]; if (!it) return;
    mt.textContent = it.title; mk.textContent = it.kindLabel; md.textContent = it.description || "";
    md.style.display = it.description ? "" : "none";
    mr.textContent = it.ref; mc.title = "Copy " + it.ref; mc.classList.remove("done");
    mh.setAttribute("aria-pressed", favs[it.id] ? "true" : "false");
    mh.setAttribute("aria-label", (favs[it.id] ? "Remove heart from " : "Heart ") + it.title);
    document.title = it.title + " · __TITLE__";
  }
  function show(id, scroll) {
    if (!byId[id]) return;
    var changed = id !== cur; cur = id;
    if (changed) {
      mv.src = "web/" + byId[id].clip;
      mv.muted = true;
      var p = mv.play(); if (p && p.catch) p.catch(function () {});
      paintMain();
    } else { paintMain(); }
    paintStrip();
    if (scroll) {
      var r = mainEl.getBoundingClientRect();
      if (r.top < 0 || r.top > window.innerHeight * 0.55) mainEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  function route() {
    var h = hashId();
    if (h) { locked = true; show(h, fromTile); }
    else show(cur || defaultId(), false);
    fromTile = false;
  }
  function go(delta) {
    var o = order(), i = 0;
    for (; i < o.length; i++) if (o[i].id === cur) break;
    var n = o[(i + delta + o.length) % o.length];
    locked = true; fromTile = false;
    location.hash = encodeURIComponent(n.id);
  }

  // ---- hearts
  function toggle(id) {
    var on = !favs[id];
    if (on) favs[id] = true; else delete favs[id];
    paintStrip(); paintMain();
    if (store) {
      pending[id] = on;
      var body = { fav: on, updated: new Date().toISOString() };
      try { store.doc(id).set(body).then(function () {}, function () { delete pending[id]; }); } catch (e) { delete pending[id]; }
    } else lsSave();
  }
  function settleDefault() {
    if (!locked && !hashId()) { locked = true; show(defaultId(), false); }   // first data in: default = first hearted, else first word
    else { paintStrip(); paintMain(); }
  }
  function applySnapshot(snap) {
    var next = {};
    snap.docs.forEach(function (d) { if (d.exists !== false) { var v = d.data(); if (v && v.fav === true && byId[d.id]) next[d.id] = true; } });
    Object.keys(pending).forEach(function (id) {                 // keep a local write the database has not echoed yet
      if ((!!next[id]) === pending[id]) delete pending[id];
      else if (pending[id]) next[id] = true; else delete next[id];
    });
    favs = next;
    settleDefault();
  }
  function useLocal() { store = null; lsLoad(); settleDefault(); }
  function connect() {
    var c = window.claude;
    if (!c || !c.use) { useLocal(); return; }
    var settled = false;
    function fb() { if (!settled) { settled = true; clearTimeout(fallback); useLocal(); } }
    var fallback = setTimeout(fb, 2500);
    try {
      Promise.resolve(c.use("db")).then(function (d) {
        if (settled) return;
        if (!d || !d.collection) { fb(); return; }
        var col = d.collection("favs");
        store = col;
        col.onSnapshot(function (snap) { settled = true; clearTimeout(fallback); applySnapshot(snap); }, fb);
      }, fb);
    } catch (e) { fb(); }
  }

  mh.addEventListener("click", function () { if (cur) toggle(cur); });
  mc.addEventListener("click", function () { if (cur) copyBtn(mc, byId[cur].ref); });
  document.getElementById("pv").addEventListener("click", function () { go(-1); });
  document.getElementById("nx").addEventListener("click", function () { go(1); });
  document.addEventListener("keydown", function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var t = e.target && e.target.tagName;
    if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return;
    if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
  }, true);
  window.addEventListener("hashchange", route);

  buildStrip();
  if (hashId()) { locked = true; show(hashId(), false); } else show(defaultId(), false);   // paint at once
  setupIO();
  connect();
})();
</script>
"""

HEART_SVG = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20.6s-7.6-4.7-9.4-9.4C1.4 8 3.2 4.8 6.5 4.8'
             'c2 0 3.7 1.1 5.5 3.3 1.8-2.2 3.5-3.3 5.5-3.3 3.3 0 5.1 3.2 3.9 6.4-1.8 4.7-9.4 9.4-9.4 9.4z"/></svg>')
COPY_SVG = ('<svg class="cc" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="1.5"/>'
            '<path d="M5 15V5.5A1.5 1.5 0 0 1 6.5 4H15"/></svg>'
            '<svg class="ck" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>')


def render(items):
    data = json.dumps(items, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", TITLE).replace("__HEART__", HEART_SVG).replace("__COPYICON__", COPY_SVG)
            .replace("__DATA__", data))


def verify(page, items):
    problems = []
    if not page.startswith(f"<title>{TITLE}</title>"):
        problems.append("fragment must start with the <title>")
    if re.search(r"<script[^>]+\bsrc=", page):
        problems.append("external script")
    for it in items:
        for k in ("clip", "preview"):
            if not (WEB / it[k]).exists():
                problems.append(f"{it['id']}: missing web/{it[k]}")
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    return problems


def main():
    WEB.mkdir(exist_ok=True)
    for t in WEB.glob("_tmp_*"):
        t.unlink(missing_ok=True)
    entries, _ = bl.load_manifest(None, merge=True)
    picked = []
    for kind in KIND_ORDER:
        for e in entries:
            if bl.kind_of(e) != kind:
                continue
            src = (MG / e["clip"]) if e.get("clip") else None
            if not src or not src.exists() or src.stat().st_size < 1000:
                warn(f"{e.get('id')}: clip missing/not finished ({e.get('clip')}); skipped")
                continue
            picked.append((kind, e, src))
    names = [s.name for _, _, s in picked]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        sys.exit(f"clip file name clash: {clash}")
    items, used, counts = [], set(), {k: 0 for k in KIND_ORDER}
    for kind, e, src in picked:
        eid = e["id"]
        print(f"{kind:5s} {eid}")
        name, how = main_clip(src)
        pv = preview(src, eid)
        if not name or not pv:
            warn(f"{eid}: skipped (clip {how}, preview {'ok' if pv else 'failed'})")
            continue
        used.update((name, pv))
        counts[kind] += 1
        items.append({"id": eid, "kind": kind, "kindLabel": KIND_LABEL[kind], "title": e.get("title") or eid,
                      "description": e.get("description") or "", "ref": reference(e), "clip": name, "preview": pv})
    page = render(items)
    (HERE / "index.html").write_text(page, encoding="utf8")
    for f in sorted(WEB.iterdir()):
        if f.is_file() and f.name not in used:
            print("removed stale web file:", f.name)
            f.unlink()
    problems = verify(page, items)
    files = sorted(f for f in WEB.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in files)
    pvs = [WEB / it["preview"] for it in items]
    if total > MAX_TOTAL:
        problems.append(f"web/ is {total / 1e6:.1f} MB, over the 60 MB limit")
    print(json.dumps({"index": str(HERE / "index.html"), "items_per_kind": counts, "items": len(items),
                      "preview_total_kb": round(sum(p.stat().st_size for p in pvs) / 1024),
                      "preview_max_kb": round(max(p.stat().st_size for p in pvs) / 1024) if pvs else 0,
                      "web_total_mb": round(total / 1e6, 2), "warnings": len(warnings), "problems": problems}, indent=1))
    print("web files (publisher list):")
    print(json.dumps([{"path": f"web/{f.name}"} for f in files]))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
