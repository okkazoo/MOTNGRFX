#!/usr/bin/env python3
"""Make archived claude.ai review boards work from disk (file://).

    python tools/archive_artifact.py relink _info/artifacts/artifact-NNN-<slug>
    python tools/archive_artifact.py relink --all

For each artifact folder:
  * index.html (as published) is kept once as index.published.html; index.html is
    always rebuilt from it, so reruns are idempotent.
  * every relative media reference (src/href/poster/url()/JSON-in-attribute/JS string)
    is re-pointed to the matching render already on disk (linked, never copied).
  * saved likes/notes from db/reviews/*.json (and db/overall/*.json) are injected as a
    read-only overlay; db/overall_history/*.json (one doc per past round) fills the
    "Past overall reviews" section at the bottom, newest first, read-only, plus a small "Archived copy" banner.
  * relink_report.json records resolved / ambiguous / unresolved references.
"""
import argparse
import html
import json
import os
import re
import shutil
import sys
from urllib.parse import quote, unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "_info", "artifacts")
EXTS = "mp4|webm|mov|mp3|wav|m4a|jpg|jpeg|png|gif|webp|svg|woff2|woff|ttf|otf"
EXT_SET = set(EXTS.split("|"))

# A media path token wholly enclosed by quotes (", ', &quot;) or url( ), optionally
# followed by #fragment / ?query. Tokens containing ':' (http:, data:) never match.
TOKEN = r"(?:\.{1,2}/)*(?:[\w.%~+@-]+/)*[\w.%~+@-]+\.(?:" + EXTS + ")"
REF_RE = re.compile(
    r"(?P<open>&quot;|\"|'|\()(?P<path>" + TOKEN + r")(?P<suffix>(?:[#?][^\"'&)\s]*)?)(?=&quot;|\"|'|\))",
    re.I,
)
# Bare filenames used as identifiers must not be rewritten.
SKIP_PREFIXES = ('data-key="', "data-key='", 'alt="', 'title="', 'aria-label="', 'data-item="', 'data-id="')


# ---------------------------------------------------------------- file index
def build_index(root):
    """lowercased filename -> [absolute paths], for media extensions only."""
    idx = {}
    for dirpath, dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        dirs[:] = [d for d in dirs if d != ".git" and not (rel == "." and d == "_info")]
        for f in files:
            if f.rsplit(".", 1)[-1].lower() in EXT_SET:
                idx.setdefault(f.lower(), []).append(os.path.join(dirpath, f))
    return idx


def resolve(pub, idx):
    """Return (path, status, candidates). status: exact|filename|unresolved; ambiguity flagged."""
    rel = unquote(pub)
    while rel.startswith("./") or rel.startswith("../"):
        rel = rel[2:] if rel.startswith("./") else rel[3:]
    name = rel.rsplit("/", 1)[-1]
    cands = idx.get(name.lower(), [])
    if not cands:
        return None, "unresolved", []
    suffix = "/" + rel.lower()
    exact = [c for c in cands if c.replace("\\", "/").lower().endswith(suffix)]
    pool, kind = (exact, "exact") if exact else (cands, "filename")
    if len(pool) == 1:
        return pool[0], kind, pool
    sizes = {os.path.getsize(c) for c in pool}
    newest = lambda lst: max(lst, key=os.path.getmtime)
    if len(sizes) == 1:  # identical-size copies: any will do, no ambiguity
        return newest(pool), kind, pool
    pub_dir = rel.rsplit("/", 1)[0].rsplit("/", 1)[-1].lower() if "/" in rel else None
    if pub_dir:
        same_dir = [c for c in pool if os.path.basename(os.path.dirname(c)).lower() == pub_dir]
        if len(same_dir) == 1:
            return same_dir[0], kind + "+dir", pool
        if same_dir:
            pool = same_dir
    return newest(pool), kind + "+newest", pool


# ---------------------------------------------------------------- rewriting
def collect_refs(text):
    refs = []
    for m in REF_RE.finditer(text):
        before = text[max(0, m.start() - 12):m.start()]
        if any((before + m.group("open")).endswith(p) for p in SKIP_PREFIXES):
            continue
        refs.append(m)
    return refs


def rel_url(art_dir, target):
    r = os.path.relpath(target, art_dir).replace("\\", "/")
    return quote(r, safe="/-_.~")


# ---------------------------------------------------------------- overlay
OVERLAY_JS = r"""
(function () {
  if (window.claude) return;
  var R = window.__ARCHIVED_REVIEWS || {}, O = window.__ARCHIVED_OVERALL || null, H = window.__ARCHIVED_HISTORY || [], STAMP = %(stamp)s;
  function q(s) { return document.querySelector(s); }
  function key(el) {
    var dk = el.getAttribute("data-key"); if (dk) return dk;
    var m = el.querySelector("img[src],video[src]");
    if (!m) return null;
    var s = decodeURIComponent((m.getAttribute("src") || "").split("#")[0].split("?")[0].split("/").pop());
    return s.replace(/[^A-Za-z0-9_.~:@+-]/g, "_").slice(0, 180) || null;
  }
  var ta = q(".ov-text"), st = q(".ov-st");
  if (ta) { if (O) ta.value = O; ta.readOnly = true; }
  if (st) st.textContent = STAMP;
  // past overall reviews: fill (or create) the bottom section, read-only, newest first
  if (H.length) {
    var hs = document.getElementById("ov-history");
    if (!hs) {
      hs = document.createElement("section"); hs.id = "ov-history"; hs.className = "grp";
      hs.innerHTML = '<h2>Past overall reviews</h2><div class="ovh-list"></div>';
      (q(".wrap") || document.body).appendChild(hs);
    }
    var hl = hs.querySelector(".ovh-list");
    if (!hl) { hl = document.createElement("div"); hl.className = "ovh-list"; hs.appendChild(hl); }
    hl.textContent = "";
    H.slice().sort(function (a, b) { return a.updated < b.updated ? 1 : a.updated > b.updated ? -1 : (a.id < b.id ? 1 : -1); }).forEach(function (r, i) {
      var det = document.createElement("details"), sum = document.createElement("summary"), b = document.createElement("b"), body = document.createElement("div");
      b.textContent = r.round || r.id; sum.appendChild(b);
      sum.appendChild(document.createTextNode(" · " + (r.updated ? r.updated.slice(0, 10) : r.id.slice(0, 10))));
      body.className = "ovh-text"; body.style.whiteSpace = "pre-wrap"; body.textContent = r.text || "(empty)";
      det.appendChild(sum); det.appendChild(body); det.open = i === 0;
      hl.appendChild(det);
    });
  }
  // Newer pages (original stays in place, liked board holds clones) expose window.__rvHub:
  // it fills notes/likes in the originals and the clones and keeps the board in sync.
  if (window.__rvHub) { window.__rvHub.apply(R, STAMP); return; }
  var items = [];
  document.querySelectorAll("figure, article.piece").forEach(function (el) {
    var k = key(el), like = el.querySelector(".rv-like"), note = el.querySelector(".rv-note");
    if (!k || !like || !note) return;
    var ph = el.previousSibling; if (!(ph && ph.nodeType === 8)) ph = null;
    items.push({ el: el, ph: ph, k: k, like: like, note: note });
  });
  function paint() {
    var onlyBtn = q(".rv-only"), colBtn = q(".rv-collect"), grid = q(".rv-bg");
    var onlyLiked = onlyBtn && onlyBtn.getAttribute("aria-pressed") === "true";
    var collect = colBtn ? colBtn.getAttribute("aria-pressed") === "true" : false;
    var n = 0;
    items.forEach(function (it) {
      var d = R[it.k] || {}, liked = !!d.liked;
      if (liked) n++;
      it.like.setAttribute("aria-pressed", liked ? "true" : "false");
      it.like.textContent = liked ? "♥ Liked" : "♡ Like";
      it.like.disabled = true;
      it.note.value = d.note || "";
      it.note.disabled = false; it.note.readOnly = true;
      it.note.style.height = "auto"; if (it.note.scrollHeight) it.note.style.height = (it.note.scrollHeight + 4) + "px";
      it.el.classList.toggle("rv-hide", !!(onlyLiked && !liked && !d.note));
      if (grid) {
        if (collect && liked) { if (it.el.parentNode !== grid) grid.appendChild(it.el); }
        else if (it.el.parentNode === grid && it.ph && it.ph.parentNode) it.ph.parentNode.insertBefore(it.el, it.ph.nextSibling);
      }
    });
    var c = q(".rv-count"); if (c) c.textContent = n + " liked";
    if (grid) {
      var bn = q(".rv-bn"); if (bn) bn.textContent = collect ? grid.children.length + " here" : "off";
      var be = q(".rv-be"); if (be) be.style.display = grid.children.length ? "none" : "";
    }
    var msg = q(".rv-msg"); if (msg) msg.textContent = STAMP;
  }
  paint();
  [".rv-only", ".rv-collect", ".rv-foldb"].forEach(function (s) {
    var b = q(s); if (b) b.addEventListener("click", paint);
  });
})();
"""

BANNER_CSS = (
    "html{padding-top:calc(28px + env(safe-area-inset-top,0px))!important}"
    "#__archived_banner{position:fixed;top:0;left:0;right:0;z-index:2147483647;height:28px;line-height:28px;"
    "padding:0 12px;background:#d9a441;color:#111;font:12px/28px ui-monospace,Consolas,monospace;"
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;box-sizing:border-box}"
    "#__archived_banner a{color:#111}"
    ".rv-note[readonly]{cursor:default}"
)


def js_json(obj):
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def load_db(art_dir):
    reviews, overall, latest, history = {}, None, "", []
    rdir = os.path.join(art_dir, "db", "reviews")
    if os.path.isdir(rdir):
        for f in sorted(os.listdir(rdir)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(rdir, f), encoding="utf8") as fh:
                d = json.load(fh)
            data = d.get("data", d)  # tolerate {"id","data":{...}} wrapper
            item = data.get("item") or d.get("id") or f[:-5]
            reviews[item] = {"note": data.get("note", ""), "liked": bool(data.get("liked")),
                             "updated": data.get("updated", "")}
            latest = max(latest, data.get("updated", ""))
    odir = os.path.join(art_dir, "db", "overall")
    if os.path.isdir(odir):
        for f in sorted(os.listdir(odir)):
            if f.endswith(".json"):
                with open(os.path.join(odir, f), encoding="utf8") as fh:
                    d = json.load(fh)
                data = d.get("data", d)
                overall = data.get("text", "")
                latest = max(latest, data.get("updated", ""))
    hdir = os.path.join(art_dir, "db", "overall_history")
    if os.path.isdir(hdir):
        for f in sorted(os.listdir(hdir)):
            if f.endswith(".json"):
                with open(os.path.join(hdir, f), encoding="utf8") as fh:
                    d = json.load(fh)
                data = d.get("data", d)
                history.append({"id": d.get("id") or f[:-5], "text": data.get("text", ""),
                                "round": data.get("round", ""), "updated": data.get("updated", "")})
                latest = max(latest, data.get("updated", ""))
    history.sort(key=lambda h: (h["updated"], h["id"]), reverse=True)
    return reviews, overall, latest, history


def index_urls(path):
    """'NNN' -> URL from INDEX.md table rows."""
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf8"):
            m = re.match(r"\|\s*(\d{3})\s*\|.*?(https://\S+)\s*\|\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def inject(text, title, url, reviews, overall, latest, history=()):
    date = latest[:10] if latest else ""
    has_db = bool(reviews) or overall is not None or bool(history)
    banner = "Archived copy of %s &middot; live: <a href=\"%s\">%s</a>" % (
        html.escape(title), html.escape(url or "#"), html.escape(url or "n/a"))
    if has_db:
        banner += " &middot; notes as of %s" % html.escape(date)
    block = "<style>%s</style><div id=\"__archived_banner\">%s</div>" % (BANNER_CSS, banner)
    if has_db:
        stamp = "Archived copy — notes as of %s, read-only" % date
        block += "<script>window.__ARCHIVED_REVIEWS=%s;window.__ARCHIVED_OVERALL=%s;window.__ARCHIVED_HISTORY=%s;%s</script>" % (
            js_json(reviews), js_json(overall), js_json(list(history)), OVERLAY_JS % {"stamp": js_json(stamp)})
    pos = text.lower().rfind("</body>")
    return text + block if pos < 0 else text[:pos] + block + text[pos:]


# ---------------------------------------------------------------- driver
def relink(art_dir, idx, urls):
    art_dir = os.path.abspath(art_dir)
    pub = os.path.join(art_dir, "index.published.html")
    cur = os.path.join(art_dir, "index.html")
    if not os.path.exists(pub):
        shutil.copyfile(cur, pub)
    with open(pub, encoding="utf8", newline="") as fh:
        text = fh.read()

    cache, report = {}, {"resolved": {}, "ambiguous": {}, "unresolved": []}
    out, last = [], 0
    for m in collect_refs(text):
        p = m.group("path")
        if p not in cache:
            path, status, cands = resolve(p, idx)
            cache[p] = (path, status)
            if path is None:
                report["unresolved"].append(p)
            else:
                report["resolved"][p] = {"path": os.path.relpath(path, ROOT).replace("\\", "/"), "match": status}
                if status.endswith("+newest") or status.endswith("+dir"):
                    report["ambiguous"][p] = {
                        "chosen": os.path.relpath(path, ROOT).replace("\\", "/"),
                        "candidates": [os.path.relpath(c, ROOT).replace("\\", "/") for c in cands]}
        path, _ = cache[p]
        out.append(text[last:m.start()])
        if path is None:
            out.append(m.group(0))
        else:
            out.append(m.group("open") + rel_url(art_dir, path) + m.group("suffix"))
        last = m.end()
    out.append(text[last:])
    new = "".join(out)

    mt = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    title = html.unescape(mt.group(1).strip()) if mt else os.path.basename(art_dir)
    nnn = re.match(r"artifact-(\d{3})", os.path.basename(art_dir))
    url = urls.get(nnn.group(1)) if nnn else None
    reviews, overall, latest, history = load_db(art_dir)
    new = inject(new, title, url, reviews, overall, latest, history)

    with open(cur, "w", encoding="utf8", newline="") as fh:
        fh.write(new)
    report["counts"] = {"refs": len(cache), "resolved": len(report["resolved"]),
                        "ambiguous": len(report["ambiguous"]), "unresolved": len(report["unresolved"])}
    with open(os.path.join(art_dir, "relink_report.json"), "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    c = report["counts"]
    print("%s: refs %d / resolved %d / ambiguous %d / unresolved %d" % (
        os.path.basename(art_dir), c["refs"], c["resolved"], c["ambiguous"], c["unresolved"]))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rl = sub.add_parser("relink", help="re-point media refs to local renders and inject read-only notes")
    rl.add_argument("artifact", nargs="?", help="artifact folder (e.g. _info/artifacts/artifact-014-...)")
    rl.add_argument("--all", action="store_true", help="every artifact-* folder")
    a = ap.parse_args(argv)
    if a.all:
        dirs = sorted(os.path.join(ARTIFACTS, d) for d in os.listdir(ARTIFACTS)
                      if d.startswith("artifact-") and os.path.isdir(os.path.join(ARTIFACTS, d)))
    elif a.artifact:
        dirs = [a.artifact]
    else:
        ap.error("give an artifact folder or --all")
    idx = build_index(ROOT)
    urls = index_urls(os.path.join(ARTIFACTS, "INDEX.md"))
    for d in dirs:
        relink(d, idx, urls)


if __name__ == "__main__":
    sys.exit(main())
