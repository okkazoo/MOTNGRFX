/* MOTNGRFX app front-end. Vanilla JS, no build step, no external requests. Binding logic lives in bind.js. */
(() => {
  "use strict";
  const B = window.MographBind;

  // ------------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const ICONS = {
    film: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 3v18M17 3v18M3 8h4M3 16h4M17 8h4M17 16h4"/>',
    image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="1.6"/><path d="m21 15-4.5-4.5L6 21"/>',
    music: '<path d="M9 18V5l11-2v13"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
    chev: '<path d="m6 9 6 6 6-6"/>',
    left: '<path d="m15 18-6-6 6-6"/>',
    dice: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.1" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.1" fill="currentColor"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    download: '<path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14"/>',
    play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>',
    pause: '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/>',
    full: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    vol: '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/>',
    mute: '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="m17 9 5 6m0-6-5 6"/>',
    zin: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.2-4.2M8.5 11h5M11 8.5v5"/>',
    zout: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.2-4.2M8.5 11h5"/>',
    undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    shuffle: '<path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/>',
    check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
  };
  const svg = (name, size = 16) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;

  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== undefined && kid !== null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
  }
  /** fetch with a deadline: a request through a dead tunnel/proxy never settles on its own, and resync() (the run's only safety net) waits on these. */
  const fetchT = async (url, opts = {}, ms = 15000) => {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), ms);
    try { return await fetch(url, { ...opts, signal: ac.signal }); }
    finally { clearTimeout(timer); }
  };
  const getJSON = async (url) => {
    const r = await fetchT(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  };
  const now = () => performance.now() / 1000;
  const fmtS = (s) => (s === null || s === undefined ? "-" : s.toFixed(s < 10 ? 2 : 1) + " s");
  const mmss = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  };
  const fmtDur = (s) => (isFinite(s) ? mmss(s) : "");
  /** the toast; v17: every non-info toast is also written into the work area (addNotice: an amber row in the progress list, then a notice line on the storyboard). opts.info = true for plain confirmations ("copied"). */
  function toast(msg, ms = 4000, opts) {
    const t = $("#toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (t.hidden = true), ms);
    if (!(opts && opts.info)) addNotice(msg);
  }

  // ------------------------------------------------------------------ state
  // v20: Make it renders the Preview = the full render at the draft size (540x960 for 9:16) WITHOUT the Real-ESRGAN upscale; "Final (1080)" upscales that finished clip afterwards (side graph `_final_upscale`).
  const PREVIEW_QUALITY = "draft540";      // image route: the app graph's IN quality value for "full render, no upscale" (still_preview.PREVIEW_FULL; the lazy `OUT final 1080` stays idle)
  const OWN_PREVIEW_QUALITY = "draft";     // own-video route: POST /mograph/own_video quality (the draft tier = 540x960; that job never upscales)
  function newUp() { return { forUrl: "", busy: false, detail: "", err: "", out: null, show: "preview" }; }
  const S = {
    templates: [], tpl: null, examples: [],
    values: {}, files: {},            // field id -> UI value / File
    objectInfo: null, packs: null, pre: null,   // server nodes (/object_info), node_packs.json, template-vs-server check
    setters: {},                      // field id -> fn(value) updating the DOM control
    redraw: {},                       // field id -> fn() re-render a drop zone
    view: "examples",                 // examples | example | run | result
    exampleShown: null,
    outputs: [],                      // {title, label, kind, url, final, filename}
    up: newUp(),                      // v20 "Final (1080)" of the finished clip: {forUrl, busy, detail, err, out, show: "preview" | "final"}
    run: null,                        // active/last run {tracker, promptId, ws, ticker, ...}
    lastTimings: null,
    zoom: 1,
    archive: null,                    // {dir, stem} of the last run's MographArchive folder (its ui text): what the Layout panel previews
    layout: { seed: null, url: "", busy: false, err: "", effects: "", lock: null },   // Randomise-layout panel; lock = {seed} set by "Use this layout"
    // "Add media" box (16_app/MEDIA_BOX.md): items = [{uid, kind, file, name (sanitised upload name), key, url, poster, duration, uploaded}]
    media: { items: [], text: "", read: null, readSig: null, busy: false, making: false, stage: "idle", soundtrack: "video", edit: "overlay", err: "", msg: "", msgT: null, edited: false },
    // STORY step (16_app/MEDIA_BOX.md): data = STORY JSON v1 (the user's edits are made in place), sig = what it was written for, edited = which fields the user touched
    story: { data: null, sig: null, busy: false, stage: "idle", detail: "", err: "", unavailable: false, failSig: null, timer: null, again: false, promise: null,
             edited: {}, vers: {}, gemma: null, selText: "", openNotes: {}, active: null, seed: Math.floor(Math.random() * 4294967296) },
    // STORYBOARD v2: liked fonts for the cards (GET /mograph/fonts/liked), state idle | loading | ready | none; every model-written field is editable and has a revert to Gemma's latest value (story.gemma / media.readGemma)
    fonts: { list: [], state: "idle", loaded: {} },
    runDetails: false,                // run view: the per-node list behind the "Details" toggle
  };

  // ------------------------------------------------------------------ BUILD VIEW (web_app/buildview.js): the live timeline of the piece being made (shots, cuts, words, music, progress, stack)
  const PREVIEW_METHOD = (() => { const v = new URLSearchParams(location.search).get("preview"); return ["none", "auto", "latent2rgb", "taesd"].includes(v) ? v : "auto"; })();   // extra_data.preview_method of every main prompt: ComfyUI streams sampler preview frames (binary ws messages) to this page (none | auto | latent2rgb | taesd); ?preview=none turns it off
  const BV = window.MographBuild.create({
    el, B, mmss,
    story: () => (S.story.data && Array.isArray(S.story.data.shots) ? S.story.data : null),
    length: () => Number(S.values.length) || 12,
    picUrl: (shot) => { const sh = S.story.data && (S.story.data.shots || []).find((x) => x.n === shot.n); return sh ? shotPicUrl(sh) : null; },
    cameraLabel: (c) => B.cameraLabel(c),
    viewUrl: (f) => B.viewUrl(f),
    finalVideo: () => document.querySelector(".player video"),
    ownDetail: () => (S.run && S.run.own ? ownProgress(S.run).detail : ""),
    audioSource: () => {                                       // the user's track, or the video's own sound: drawn as a real waveform from the first second
      const track = S.media.items.find((i) => i.kind === "audio") || (S.files.audio ? { file: S.files.audio } : null), video = S.media.items.find((i) => i.kind === "video"), st = soundtrackNow();
      if (track && track.file && st !== "generate" && (st === "track" || !video)) return { blob: track.file, offset: 0, kind: "your track" };
      if (video && video.file && st === "video" && video.hasAudio !== false) { const e = readEntry(video); return { blob: video.file, offset: (e && Array.isArray(e.window) && Number(e.window[0])) || 0, kind: "the video's sound" }; }
      return null;
    },
  });

  // ------------------------------------------------------------------ boot
  async function boot() {
    { const b = $("#build"); if (b) { const ours = /^18\d\d\d$/.test(location.port); b.hidden = !ours || document.documentElement.dataset.shots === "1"; b.textContent = ours ? "v21 · " + (location.port === "18188" ? "box" : "dev :" + location.port) : ""; } }   // our dev / box ports only (a user's ComfyUI never shows it); data-shots="1" = README captures   // DEV BUILD BADGE: delete this line + #build in index.html when shipping
    $("#btn-home").innerHTML = svg("left", 14);
    $("#btn-download").innerHTML = svg("download", 14) + "<span>Download</span>";
    $("#zoom-out").innerHTML = svg("zout", 14);
    $("#zoom-in").innerHTML = svg("zin", 14);
    $("#btn-close-drawer").innerHTML = svg("x", 14);

    $("#btn-home").onclick = () => { if (!activeRun()) { S.view = "examples"; renderCanvas(); } };
    $("#btn-run").onclick = run;
    $("#btn-stop").onclick = cancelRun;
    $("#btn-setup").onclick = backToSetup;
    $("#btn-download").onclick = download;
    $("#btn-timings").onclick = () => { const d = $("#drawer"); d.hidden = !d.hidden; renderDrawer(); };
    $("#btn-close-drawer").onclick = () => ($("#drawer").hidden = true);
    $("#btn-copy-timings").onclick = copyTimings;
    $("#zoom-in").onclick = () => setZoom(S.zoom + 0.1);
    $("#zoom-out").onclick = () => setZoom(S.zoom - 0.1);
    window.addEventListener("online", () => { const r = activeRun(); if (r) resync(r); });
    try { S.layout.lock = JSON.parse(localStorage.getItem("mograph.layout.lock") || "null"); } catch (_) { S.layout.lock = null; }
    $("#dev-graph").onchange = (e) => {
      if (activeRun()) { e.target.value = S.tpl.name; toast("Wait for the current run, or cancel it."); return; }
      localStorage.setItem("mograph.graph.v2", e.target.value);
      loadTemplate(e.target.value).catch((err) => toast(String(err)));
    };

    try { S.templates = await getJSON("/mograph/graphs"); } catch (e) { toast("Could not list templates: " + e.message); return; }
    const sel = $("#dev-graph");
    // the shipped `motngrfx_*` templates show their friendly label alone; the long fusion_design_* names stay as aliases (name - title)
    for (const t of S.templates) sel.append(el("option", { value: t.name }, t.name.startsWith("motngrfx_") && t.title ? t.title : (t.title && t.title !== t.name ? `${t.name} - ${t.title}` : t.name)));
    const want = new URLSearchParams(location.search).get("graph");   // the template picker is hidden: default graph unless ?graph= (dev); no remembered choice
    const name = S.templates.some((t) => t.name === want) ? want : (S.templates.some((t) => t.name === "motngrfx_default") ? "motngrfx_default" : (S.templates[0] || {}).name);
    if (!name) { toast("No templates in app/graphs"); return; }
    sel.value = name;
    try { S.examples = await getJSON("/mograph/examples"); } catch (_) { S.examples = []; }
    await loadTemplate(name);
  }

  async function loadTemplate(name) {
    const t = await getJSON("/mograph/graphs/" + encodeURIComponent(name));
    const prevValues = S.values, prevFiles = S.files;
    S.tpl = t;
    S.values = {}; S.files = {};
    for (const f of B.visibleFields(t.app, t.graph)) {
      if (f.id in prevValues && prevValues[f.id] !== "" ) S.values[f.id] = prevValues[f.id];
      if (prevFiles[f.id]) S.files[f.id] = prevFiles[f.id];
    }
    { const cl = String(t.app.cost_line || "");                          // v19: the footer sentence "Runs on the GPU box" is gone; a real cost line (any other text) still shows
      $("#cost-line").textContent = /^\s*Runs on the GPU box\.?\s*$/i.test(cl) ? "" : cl; }
    // one short user line; the graph's developer note stays as a tooltip
    $("#note").textContent = "";                       // v19: no footer sentence under Make it (the note area and its tooltip stay)
    $("#note").title = t.app.note || "";
    renderPanel();
    S.pre = null;
    S.pruned = t.pruned || [];                       // optional nodes the server already removed (app_routes.load_graph)
    renderPreflight();
    updateRunButton();
    if (!activeRun()) { S.view = "examples"; S.exampleShown = null; S.outputs = []; S.up = newUp(); renderCanvas(); }   // a result of another template does not apply
    preflight(false);
  }

  // ------------------------------------------------------------------ left panel
  function renderPanel() {
    const host = $("#fields");
    host.textContent = "";
    S.setters = {}; S.redraw = {};
    const { app, graph } = S.tpl;
    for (const f of B.visibleFields(app, graph)) {
      if (f.id === "quality") { S.values[f.id] = PREVIEW_QUALITY; continue; }          // v19: no Quality select; v20: every Make it is the Preview (full render at the draft size, no upscale), Final (1080) is a button on the finished clip
      if (f.kind === "seed") { if (S.values[f.id] === undefined) S.values[f.id] = ""; continue; }       // v19: no Random seed field: every Make it / story draws a fresh seed (recorded in the run: S.run.used)
      if (f.hidden || f.id === "previous") { if (S.values[f.id] === undefined) S.values[f.id] = f.default ?? ""; continue; }   // "Earlier concepts" is batch plumbing: hidden, its default still goes to the graph
      host.append(buildField(f));
    }
  }

  function buildField(f) {
    switch (f.kind) {
      case "textarea": case "text": return buildText(f);
      case "select": return buildSelect(f);
      case "seed": case "number": return buildSeed(f);
      case "image": case "audio": return buildDrop(f);
      case "media": return buildMediaField(f);
      default: return el("div", { class: "field" }, `Unsupported field kind: ${f.kind}`);
    }
  }

  function buildText(f) {
    const common = { placeholder: f.placeholder || "", "data-id": f.id };
    const input = f.kind === "textarea"
      ? el("textarea", { ...common, rows: f.rows || 3, spellcheck: "false" })
      : el("input", { ...common, type: "text" });
    input.value = S.values[f.id] ?? f.default ?? "";
    S.values[f.id] = input.value;
    input.addEventListener("input", () => { S.values[f.id] = input.value; updateRunButton(); });
    S.setters[f.id] = (v) => { input.value = v ?? ""; S.values[f.id] = input.value; };
    return el("div", { class: "field" }, el("label", { class: "f-label" }, f.label + (f.optional ? "" : " *")), input);
  }

  function buildSelect(f) {
    const sel = el("select", { "data-id": f.id, title: f.hint || "" });
    for (const o of f.options || []) sel.append(el("option", { value: String(B.optionValue(o)) }, B.optionLabel(o)));
    const init = S.values[f.id] ?? f.default ?? B.optionValue((f.options || [])[0]);
    sel.value = String(init);
    S.values[f.id] = sel.value;
    sel.addEventListener("change", () => { S.values[f.id] = sel.value; if (f.id === "length") { paintLength(); scheduleRead(); scheduleStory(2000); } });
    S.setters[f.id] = (v) => { sel.value = String(v); if (sel.value !== String(v)) sel.selectedIndex = 0; S.values[f.id] = sel.value; if (f.id === "length") paintLength(); };
    const note = f.id === "length" ? el("div", { class: "len-note", hidden: true }) : null;
    const box = el("div", { class: "field" },
      el("div", { class: "row-select" }, el("span", { class: "lbl" }, f.label), sel, el("span", { class: "chev", html: svg("chev", 14) })), note);
    if (note) setTimeout(paintLength, 0);
    return box;
  }

  function buildSeed(f) {
    const input = el("input", { type: "text", inputmode: "numeric", placeholder: f.placeholder || "Random seed", "data-id": f.id });
    input.value = S.values[f.id] ?? "";
    input.addEventListener("input", () => { input.value = input.value.replace(/[^0-9]/g, "").slice(0, 10); S.values[f.id] = input.value; });
    const dice = el("button", { class: "icon-btn", title: "Random seed", type: "button", html: svg("dice", 16),
      onclick: () => { input.value = String(B.randomSeed()); S.values[f.id] = input.value; } });
    S.setters[f.id] = (v) => { input.value = v === undefined || v === null ? "" : String(v); S.values[f.id] = input.value; };
    return el("div", { class: "field" }, el("div", { class: "row-select row-seed" }, input, dice));
  }

  function buildDrop(f) {
    const isImg = f.kind === "image";
    const box = el("div", { class: "drop", "data-id": f.id });
    const input = el("input", { type: "file", hidden: true, accept: isImg ? "image/*" : "audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus" });
    let objUrl = null, audio = null;

    const clear = () => { delete S.files[f.id]; draw(); updateRunButton(); };
    const draw = () => {
      if (objUrl) { URL.revokeObjectURL(objUrl); objUrl = null; }
      if (audio) { audio.pause(); audio = null; }
      box.textContent = "";
      const file = S.files[f.id];
      box.className = "drop";
      if (!file) {
        box.append(el("div", { html: svg(isImg ? "image" : "music", 22) }),
          el("div", { class: "d-title" }, f.title || (isImg ? "Add an image" : "Add audio")),
          el("div", { class: "d-hint" }, f.hint || (f.optional ? "Optional" : "")));
        return;
      }
      objUrl = URL.createObjectURL(file);
      const xbtn = el("button", { class: "x", type: "button", title: "Remove", html: svg("x", 12), onclick: (e) => { e.stopPropagation(); clear(); } });
      if (isImg) {
        box.classList.add("filled");
        const cap = el("div", { class: "cap" }, file.name);
        const img = el("img", { src: objUrl, alt: "" });
        img.onload = () => (cap.textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight}`);
        box.append(img, cap, xbtn);
      } else {
        box.classList.add("audio-filled");
        const sub = el("small", {}, "reading...");
        audio = new Audio(objUrl);
        audio.addEventListener("loadedmetadata", () => (sub.textContent = isFinite(audio.duration) ? `${audio.duration.toFixed(1)} s` : ""));
        const play = el("button", { class: "a-play", type: "button", html: svg("play", 12),
          onclick: (e) => { e.stopPropagation(); if (audio.paused) audio.play(); else audio.pause(); } });
        audio.addEventListener("play", () => (play.innerHTML = svg("pause", 12)));
        audio.addEventListener("pause", () => (play.innerHTML = svg("play", 12)));
        audio.addEventListener("ended", () => (play.innerHTML = svg("play", 12)));
        box.append(play, el("div", { class: "a-name" }, file.name, sub), xbtn);
      }
    };
    S.redraw[f.id] = draw;
    box.addEventListener("click", () => { if (!S.files[f.id]) input.click(); });
    input.addEventListener("change", () => { if (input.files[0]) setFile(f.id, input.files[0]); input.value = ""; });
    box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("over"); });
    box.addEventListener("dragleave", () => box.classList.remove("over"));
    box.addEventListener("drop", (e) => {
      e.preventDefault(); box.classList.remove("over");
      const file = e.dataTransfer.files[0];
      if (file) setFile(f.id, file);
    });
    draw();
    return el("div", { class: "field" }, box, input);
  }

  function setFile(id, file) {
    S.files[id] = file;
    if (S.redraw[id]) S.redraw[id]();
    updateRunButton();
  }

  function currentMissing() {
    const { app, graph } = S.tpl;
    return B.visibleFields(app, graph).filter((f) => {
      if (f.optional || f.kind === "seed" || f.kind === "media") return false;
      const v = f.kind === "image" || f.kind === "audio" ? S.files[f.id] : S.values[f.id];
      return B.isEmpty(f, v);
    });
  }

  function updateRunButton() {
    if (!S.tpl) return;
    const app = S.tpl.app, btn = $("#btn-run");
    const running = !!activeRun();
    const miss = currentMissing();
    if (running) { btn.textContent = (app.running_verb || "Working") + "..."; btn.disabled = true; }
    else if (S.media.making) { btn.textContent = S.story.busy ? "Writing the story..." : "Reading media..."; btn.disabled = true; }
    else if (S.pre && S.pre.checked && !S.pre.ok) { btn.textContent = "Install the missing nodes first"; btn.disabled = true; }
    else if (miss.length) { btn.textContent = (app.missing || {})[miss[0].id] || `Add ${miss[0].label.toLowerCase()} to start`; btn.disabled = true; }
    else { btn.textContent = app.verb || "Run"; btn.disabled = false; }
    const stop = $("#btn-stop");
    stop.hidden = !running;
    const ar = activeRun();
    if (!ar || !ar.cancelled) { stop.textContent = "Cancel"; stop.disabled = false; }
    paintSetupBtn();
    $("#dev-graph").disabled = !!running;
  }

  // ------------------------------------------------------------------ media box + automatic read (16_app/MEDIA_BOX.md)
  const mediaField = () => (S.tpl ? B.visibleFields(S.tpl.app, S.tpl.graph).find((f) => f.kind === "media") || null : null);
  const mediaHasInput = () => S.media.items.length > 0 || S.media.text.trim() !== "";
  // Length matters to the READ only with a video (its window is clamped to it); images and text read the same at any length (the STORY follows Length on its own)
  // v19: the soundtrack ("video" | "track" | "generate") picks which audio is read (lyrics / beat grid), so it is part of the signature: a change re-reads.
  const soundtrackPlan = () => { const v = S.media.items.find((i) => i.kind === "video"); return B.soundtrackPlan(S.media.items, S.media.soundtrack, v ? v.hasAudio : undefined); };
  const soundtrackNow = () => soundtrackPlan().value;
  const mediaSig = () => B.mediaSignature(S.media.items.map((i) => ({ kind: i.kind, key: i.key })), S.media.text, S.media.items.some((i) => i.kind === "video") ? S.values.length : "", soundtrackNow());
  const errText = (e) => String((e && (e.err ? e.err.message : e.message)) || e);
  const ed = () => (S.media.edited && typeof S.media.edited === "object" ? S.media.edited
    : (S.media.edited = { on_screen: false, concept: false, look: false, media: {} }));
  const itemOfFile = (file) => S.media.items.find((i) => i.uploaded && (i.uploaded === file || i.uploaded.endsWith("/" + file) || file.endsWith("/" + i.uploaded)));
  let mediaUid = 0;

  function setMediaMsg(text) {
    const m = S.media;
    m.msg = text; clearTimeout(m.msgT);
    if (text) addNotice(text);                                  // v17: also a warning row in the work area
    if (text) m.msgT = setTimeout(() => { m.msg = ""; paintMediaStatus(); }, 6000);
    paintMediaStatus();
  }

  /** drop / pick: validate against the limits, keep the cheap local bits (thumbnail, poster, duration) now, schedule the read. */
  async function dropMedia(dt) {   // a drag out of a browser page hands over a link / .url shortcut, not the file: fetch it when the browser may, else say where to drag from
    const all = [...dt.files], links = new Set((dt.getData("text/uri-list") || "").split(/\r?\n/).filter((u) => /^(https?|blob):/.test(u)));   // read before the first await
    const real = all.filter((f) => !/\.url$/i.test(f.name)), failed = [];
    for (const f of all.filter((f) => /\.url$/i.test(f.name))) { const u = /^URL=(.+)$/m.exec(await f.text()); if (u && /^https?:/.test(u[1].trim())) links.add(u[1].trim()); else failed.push(f.name.replace(/(-\d+)?\.url$/i, "")); }
    for (const u of links) {
      const name = decodeURIComponent(u.split(/[?#]/)[0].split("/").pop() || "media");
      try { const r = await fetch(u); if (!r.ok) throw new Error(r.status); const b = await r.blob(); real.push(new File([b], name, { type: b.type })); } catch (_) { failed.push(name); }
    }
    if (failed.length) setMediaMsg(`${failed.join(", ")}: the browser handed over a link, not the file. Drag it from File Explorer instead.`);
    if (real.length) addMediaFiles(real);
  }
  async function addMediaFiles(files) {
    const mf = mediaField();
    if (!mf || activeRun()) return;
    const m = S.media, limits = B.mediaLimits(mf);
    const list = [...files];                                                           // the first file that fits decides the mode of an empty box
    const upName = (x) => "motngrfx_" + x.name.replace(/[^A-Za-z0-9._-]+/g, "_");
    const plan = B.planMediaDrop(m.items, list, limits);                                // audio / video replace, images fill free slots (bind.js)
    let added = 0;
    for (const x of [...plan.removes].sort((p, q) => q - p)) {                          // the replaced file goes first: its url, readout notes and name slot are freed
      const old = m.items[x];
      releaseFrames(old);
      try { URL.revokeObjectURL(old.url); } catch (_) {}
      if (m.edited && typeof m.edited === "object" && old.uploaded) {
        for (const k of Object.keys(m.edited.media || {})) if (old.uploaded === k || old.uploaded.endsWith("/" + k) || k.endsWith("/" + old.uploaded)) delete m.edited.media[k];
      }
      m.items.splice(x, 1); added++;
    }
    for (const { i, kind } of plan.adds) {
      const file = list[i];
      let f = file;
      for (let n = 2; m.items.some((it) => upName(it.file) === upName(f)); n++) {       // two files with the same name must not overwrite each other on the server
        const dot = file.name.lastIndexOf("."), stem = dot > 0 ? file.name.slice(0, dot) : file.name;
        f = new File([file], stem + "-" + n + (dot > 0 ? file.name.slice(dot) : ""), { type: file.type });
      }
      const item = { uid: ++mediaUid, kind, file: f, key: `${f.name}|${f.size}|${f.lastModified}`, url: URL.createObjectURL(f),
                     poster: null, posterFail: false, duration: NaN, uploaded: null };
      m.items.push(item); added++;
      if (item.kind === "video") videoPoster(item); else if (item.kind === "audio") audioInfo(item);
    }
    if (plan.adds.some((x) => x.kind === "audio")) m.soundtrack = "track";             // v19: dropping an audio file selects "My track"
    setMediaMsg(plan.msg);
    if (added) { redrawMedia(); updateRunButton(); scheduleRead(); }
  }

  function removeMedia(uid) {
    const m = S.media, it = m.items.find((i) => i.uid === uid);
    if (!it || activeRun()) return;
    releaseFrames(it);
    try { URL.revokeObjectURL(it.url); } catch (_) {}
    m.items = m.items.filter((i) => i !== it);
    if (it.kind === "audio" && m.soundtrack === "track") m.soundtrack = m.items.some((i) => i.kind === "video") ? "video" : "generate";   // v19: no file left -> Video sound (video) / New music (images)
    if (it.kind === "video" && m.soundtrack === "video") m.soundtrack = m.items.some((i) => i.kind === "audio") ? "track" : "generate";
    setMediaMsg("");
    redrawMedia(); updateRunButton(); scheduleRead();
  }

  function clearMedia() {
    for (const it of S.media.items) { releaseFrames(it); try { URL.revokeObjectURL(it.url); } catch (_) {} }
    S.media.items = []; S.media.text = ""; S.media.soundtrack = "video"; S.media.edit = "overlay"; setMediaMsg("");
    if (S.mediaUI) S.mediaUI.ta.value = "";
    if (S.mediaTextSync) S.mediaTextSync();
    redrawMedia(); updateRunButton(); scheduleRead();
  }

  /** image order = shot order: image 1 -> IN image, 2..5 -> IN images. Moves one image a step left/right among the images. */
  function moveImage(uid, delta) {
    const m = S.media, imgs = m.items.filter((i) => i.kind === "image");
    const a = imgs.findIndex((i) => i.uid === uid), b = a + delta;
    if (a < 0 || b < 0 || b >= imgs.length) return;
    const ia = m.items.indexOf(imgs[a]), ib = m.items.indexOf(imgs[b]);
    [m.items[ia], m.items[ib]] = [m.items[ib], m.items[ia]];
    redrawMedia(uid); scheduleRead();
  }

  /** commit a drag: put the image `uid` at index `to` among the images (video / audio keep their places). */
  function setImageOrder(uid, to) {
    const m = S.media, imgs = m.items.filter((i) => i.kind === "image");
    const from = imgs.findIndex((i) => i.uid === uid);
    if (from < 0 || to < 0 || to >= imgs.length || to === from) return false;
    const [d] = imgs.splice(from, 1); imgs.splice(to, 0, d);
    let k = 0; const next = m.items.map((i) => (i.kind === "image" ? imgs[k++] : i));
    m.items.splice(0, m.items.length, ...next);
    return true;
  }

  /** Drag-to-reorder with pointer events. The dragged tile follows the pointer; the other tiles slide (CSS transform) into the slots they
   *  would take, which opens a gap at the insertion point. Slots are measured once at the start, so nothing depends on hit-testing
   *  (no flicker) and wrapped rows work. Release inside the box commits; release outside, Esc or pointercancel puts everything back. */
  function startImageDrag(e, tile, grid, uid) {
    if (e.button !== 0 || e.target.closest(".x")) return;
    const m = S.media;
    if (m.drag) m.drag.abort();
    const tiles = [...grid.querySelectorAll(".m-tile.image")], from = tiles.indexOf(tile);
    if (tiles.length < 2 || from < 0) return;
    const g0 = grid.getBoundingClientRect(), r0 = tile.getBoundingClientRect();
    const slots = tiles.map((t) => { const r = t.getBoundingClientRect(); return { x: r.left - g0.left, y: r.top - g0.top, w: r.width, h: r.height }; });
    const sx = e.clientX, sy = e.clientY, gx = sx - r0.left, gy = sy - r0.top, pid = e.pointerId, SCALE = 1.06;
    let active = false, to = from, lastTo = -1;
    const slide = () => {                                    // every other tile -> the slot it takes with the dragged one at `to`
      const pre = tiles.map((_, j) => j).filter((j) => j !== from), order = pre.slice(0, to).concat([from], pre.slice(to));
      tiles.forEach((t, j) => {
        if (j === from) return;
        const a = slots[j], b = slots[order.indexOf(j)];
        t.style.transform = a === b ? "" : `translate(${b.x - a.x}px, ${b.y - a.y}px)`;
      });
      tiles.forEach((t, j) => { const n = t.querySelector(".m-num"); if (n) n.textContent = String(order.indexOf(j) + 1); });   // badges show the order it would have
    };
    const place = (ev) => {
      const gr = grid.getBoundingClientRect(), dx = ev.clientX - gx, dy = ev.clientY - gy;
      tile.style.transform = `translate(${dx - (gr.left + slots[from].x)}px, ${dy - (gr.top + slots[from].y)}px) scale(${SCALE})`;
      const cx = dx - gr.left + slots[from].w / 2, cy = dy - gr.top + slots[from].h / 2;
      let best = Infinity;
      slots.forEach((s, j) => { const d = (s.x + s.w / 2 - cx) ** 2 + (s.y + s.h / 2 - cy) ** 2; if (d < best) { best = d; to = j; } });
      if (to !== lastTo) { lastTo = to; slide(); }
    };
    const onMove = (ev) => {
      if (ev.pointerId !== pid) return;
      if (!active) {
        if (Math.hypot(ev.clientX - sx, ev.clientY - sy) < 5) return;
        active = true; tile.classList.add("dragging"); grid.classList.add("sorting");
        try { tile.setPointerCapture(pid); } catch (_) {}
      }
      ev.preventDefault(); place(ev);
    };
    const detach = () => {
      tile.removeEventListener("pointermove", onMove); tile.removeEventListener("pointerup", onUp); tile.removeEventListener("pointercancel", onCancel);
      document.removeEventListener("keydown", onKey, true);
      try { tile.releasePointerCapture(pid); } catch (_) {}
      grid.classList.remove("sorting"); m.drag = null;
    };
    const putBack = () => {                                  // everything glides home
      detach();
      if (!active) return;
      tile.classList.remove("dragging"); tile.classList.add("settling");
      tiles.forEach((t, j) => { t.style.transform = ""; const n = t.querySelector(".m-num"); if (n) n.textContent = String(j + 1); });
      setTimeout(() => tile.classList.remove("settling"), 220);
    };
    const onCancel = (ev) => { if (ev.pointerId === pid) putBack(); };
    const onKey = (ev) => { if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); putBack(); } };
    const onUp = (ev) => {
      if (ev.pointerId !== pid) return;
      const br = S.mediaUI.box.getBoundingClientRect();
      const inside = ev.clientX >= br.left - 8 && ev.clientX <= br.right + 8 && ev.clientY >= br.top - 8 && ev.clientY <= br.bottom + 8;
      if (!active || !inside || to === from) { putBack(); return; }
      const vis = tile.getBoundingClientRect();              // where the tile is on screen right now
      detach();
      setImageOrder(uid, to); redrawMedia(uid); scheduleRead();
      const nt = S.mediaUI.box.querySelector(`.m-tile[data-uid="${uid}"]`);
      if (!nt) return;
      const nr = nt.getBoundingClientRect();                 // FLIP: start from the drop position, glide into the slot
      nt.classList.add("settling"); nt.style.transition = "none";
      nt.style.transform = `translate(${vis.left + vis.width / 2 - (nr.left + nr.width / 2)}px, ${vis.top + vis.height / 2 - (nr.top + nr.height / 2)}px) scale(${SCALE})`;
      void nt.offsetWidth;
      nt.style.transition = ""; nt.style.transform = "";
      setTimeout(() => nt.classList.remove("settling"), 220);
    };
    tile.addEventListener("pointermove", onMove); tile.addEventListener("pointerup", onUp); tile.addEventListener("pointercancel", onCancel);
    document.addEventListener("keydown", onKey, true);
    m.drag = { abort: () => { detach(); } };
  }

  const redrawMedia = (focusUid) => { if (S.mediaDraw) S.mediaDraw(focusUid); paintMediaStatus(); paintStoryImages(); };

  /** poster frame of a video: a frame drawn from a <video> onto a canvas, plus its duration. */
  function videoPoster(item) {
    const v = document.createElement("video");
    v.muted = true; v.preload = "metadata"; v.playsInline = true; v.src = item.url;
    const release = () => { try { v.removeAttribute("src"); v.load(); } catch (_) {} };
    v.addEventListener("loadedmetadata", () => {
      item.duration = v.duration; item.dim = [v.videoWidth, v.videoHeight];
      v.currentTime = Math.min(Math.max(v.duration * 0.1, 0.1), 1.5) || 0.1;
      redrawMedia();
      if (S.media.items.includes(item)) scheduleRead();         // video mode: the Length can never be longer than the video (syncLength)
    });
    v.addEventListener("seeked", () => {
      try {
        const c = document.createElement("canvas"), w = 320, h = Math.max(1, Math.round(w * v.videoHeight / (v.videoWidth || 1)));
        c.width = w; c.height = h;
        c.getContext("2d").drawImage(v, 0, 0, w, h);
        item.poster = c.toDataURL("image/jpeg", 0.82);
      } catch (_) { item.posterFail = true; }
      release(); redrawMedia();
    });
    v.addEventListener("error", () => { item.posterFail = true; release(); redrawMedia(); });
  }

  /** VIDEO MODE: the video's frame at time t (s), grabbed from the dropped file with a hidden <video> + canvas -> JPEG data URL, cached per time (0.01 s); null when the browser cannot decode it.
   *  The seeks are queued one after another on one hidden element per video. */
  const frameKey = (t) => (Math.round(Number(t) * 100) / 100).toFixed(2);
  const frameCached = (item, t) => (item && item.frames ? item.frames[frameKey(t)] || null : null);
  function videoFrameAt(item, t) {
    const key = frameKey(t);
    item.frames = item.frames || {}; item.fpend = item.fpend || {};
    if (item.frames[key]) return Promise.resolve(item.frames[key]);
    if (item.fpend[key]) return item.fpend[key];
    const run = () => new Promise((resolve) => {
      const v = item.fvid || (item.fvid = Object.assign(document.createElement("video"), { muted: true, preload: "auto", playsInline: true }));
      let tm = null, off = false;
      const done = (url) => { if (off) return; off = true; clearTimeout(tm); v.onseeked = null; v.onerror = null; resolve(url); };
      const draw = () => {
        try {
          const w = Math.min(480, v.videoWidth || 480), h = Math.max(1, Math.round(w * (v.videoHeight || 1) / (v.videoWidth || 1))), c = document.createElement("canvas");
          c.width = w; c.height = h; c.getContext("2d").drawImage(v, 0, 0, w, h);
          done(c.toDataURL("image/jpeg", 0.85));
        } catch (_) { done(null); }
      };
      const go = () => {
        clearTimeout(tm); tm = setTimeout(() => done(null), 8000);
        const to = Math.min(Math.max(0, Number(t) || 0), Math.max(0, (v.duration || Number(t) || 0) - 0.05));
        if (Math.abs(v.currentTime - to) < 0.002 && v.readyState >= 2) return draw();
        v.onseeked = draw; v.onerror = () => done(null);
        v.currentTime = to;
      };
      tm = setTimeout(() => done(null), 12000);
      if (!v.src) { v.src = item.url; v.addEventListener("loadedmetadata", go, { once: true }); v.addEventListener("error", () => done(null), { once: true }); }
      else if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
    });
    const p = (item.fq = (item.fq || Promise.resolve()).then(run)).then((url) => { delete item.fpend[key]; if (url) item.frames[key] = url; return url; });
    item.fpend[key] = p;
    return p;
  }
  function releaseFrames(item) {
    if (item && item.fvid) { try { item.fvid.removeAttribute("src"); item.fvid.load(); } catch (_) {} item.fvid = null; }
    if (item) { item.frames = {}; item.fpend = {}; item.fq = null; }
  }

  function audioInfo(item) {
    const a = new Audio(item.url);
    a.addEventListener("loadedmetadata", () => { item.duration = a.duration; redrawMedia(); });
  }

  function buildMediaField(f) {
    const m = S.media, limits = B.mediaLimits(f);
    const box = el("div", { class: "drop media-drop", "data-id": f.id });
    const input = el("input", { type: "file", hidden: true, multiple: true, accept: "video/*,image/*,audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.mkv" });
    const msg = el("div", { class: "media-msg", role: "status" });
    const ta = el("textarea", { rows: 4, spellcheck: "false", "data-id": "media_text",
      placeholder: 'Words, lyrics, a concept or a feel for the look. "Quoted text" is shown on screen exactly.' });
    ta.value = m.text;
    // v19: the words textarea is collapsed behind a "+ Words or direction" link (it stays in the DOM, hidden); it starts open when it has text and collapses again when emptied and blurred
    const tlabel = el("label", { class: "f-label" }, "Paste words, lyrics or direction");
    const textWrap = el("div", { class: "media-text" }, tlabel, ta);
    const tlink = el("button", { class: "link-btn text-link", type: "button", "data-id": "media_text_link", onclick: () => openText(true) }, "+ Words or direction");
    const syncText = () => { textWrap.hidden = false; tlink.hidden = true; };      // 2026-11-02 user: always expanded, no collapse link
    const openText = (focus) => { textWrap.dataset.open = "1"; syncText(); if (focus) ta.focus(); };
    ta.addEventListener("input", () => { m.text = ta.value; if (m.text.trim() !== "") textWrap.dataset.open = "1"; syncText(); scheduleRead(); updateRunButton(); });
    ta.addEventListener("blur", () => { if (ta.value.trim() === "") { delete textWrap.dataset.open; syncText(); } });
    const stat = el("span", { class: "m-stat" });
    const reread = el("button", { class: "link-btn", type: "button", hidden: true, title: "Read the media and text again", onclick: () => { m.unavailable = false; m.failSig = null; S.story.unavailable = false; S.story.failSig = null; doRead(true); } }, "Re-read");
    const show = el("button", { class: "link-btn", type: "button", hidden: true, onclick: () => { if (!activeRun()) { S.view = "read"; renderCanvas(); } } }, "Show storyboard");
    S.mediaUI = { box, msg, ta, stat, reread, show };

    const canAdd = () => { const p = B.planMediaDrop(m.items, [{ name: "x.mp4" }, { name: "x.png" }, { name: "x.wav" }], limits); return p.adds.length > 0; };   // a full box still takes a replacement audio / video
    const draw = (focusUid) => {
      if (m.drag) m.drag.abort();                                 // a redraw during a drag drops the drag (the tiles are replaced)
      box.textContent = "";
      box.classList.toggle("filled", m.items.length > 0);
      if (!m.items.length) {
        box.append(el("div", { html: svg("film", 22) }), el("div", { class: "d-title" }, f.title || "Add media"),
          el("div", { class: "d-hint" }, f.hint || `Video, images or audio: drop or click. Up to ${limits.video} video, ${limits.image} images, ${limits.audio} audio.`));
        return;
      }
      const ids = B.mediaIds(m.items);
      const grid = el("div", { class: "m-grid" });
      let imgN = 0, focusEl = null;
      const rank = { video: 0, image: 1, audio: 2 };
      const display = m.items.map((it, i) => [it, i]).sort((a, b) => rank[a[0].kind] - rank[b[0].kind]);       // video, images (in order), audio
      for (const [it, idx] of display) {
        const tile = el("div", { class: `m-tile ${it.kind}`, tabindex: "0", "data-uid": it.uid, title: it.file.name,
          "aria-label": `${ids[idx]}, ${it.kind}, ${it.file.name}. Delete removes it.` + (it.kind === "image" ? " Left and right arrows change its order." : "") });
        const rm = el("button", { class: "x", type: "button", title: "Remove (Delete key)", "aria-label": `Remove ${it.file.name}`, html: svg("x", 12),
          onclick: (e) => { e.stopPropagation(); removeMedia(it.uid); } });
        if (it.kind === "video") {
          tile.append(it.poster ? el("img", { src: it.poster, alt: "", draggable: "false" }) : el("div", { class: "m-ph", html: svg("film", 26) }),
            el("span", { class: "m-id" }, ids[idx]), isFinite(it.duration) ? el("span", { class: "m-dur" }, fmtDur(it.duration)) : null,
            el("div", { class: "m-cap" }, it.file.name));
        } else if (it.kind === "image") {
          imgN++;
          tile.append(el("img", { src: it.url, alt: "", draggable: "false" }), el("span", { class: "m-num", title: imgN === 1 ? "Image 1 is the first frame" : `Image ${imgN} (shot ${imgN})` }, String(imgN)));
          tile.addEventListener("pointerdown", (e) => startImageDrag(e, tile, grid, it.uid));
        } else {
          tile.append(el("div", { class: "m-aicon", html: svg("music", 18) }), el("span", { class: "m-id" }, ids[idx]),
            el("div", { class: "m-aname" }, it.file.name, isFinite(it.duration) ? el("small", {}, fmtDur(it.duration)) : null));
        }
        tile.append(rm);
        tile.addEventListener("keydown", (e) => {
          if (e.target !== tile) return;
          if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); removeMedia(it.uid); }
          else if (it.kind === "image" && e.key === "ArrowLeft") { e.preventDefault(); moveImage(it.uid, -1); }
          else if (it.kind === "image" && e.key === "ArrowRight") { e.preventDefault(); moveImage(it.uid, 1); }
        });
        if (focusUid === it.uid) focusEl = tile;
        grid.append(tile);
      }
      if (canAdd()) grid.append(el("button", { class: "m-add", type: "button", title: "Add more media", onclick: () => input.click() }, "+ Add"));
      box.append(grid);
      {                                                              // v19 SOUNDTRACK: video mode = Video sound | My track | New music, image mode = My track | New music
        const sp = soundtrackPlan();
        const seg = el("div", { class: "m-sound", role: "radiogroup", "aria-label": "Soundtrack from" }, el("span", { class: "m-sound-l" }, "Soundtrack from"));
        for (const o of sp.options) {
          seg.append(el("button", { class: "seg" + (sp.value === o.val ? " on" : ""), type: "button", role: "radio", "data-val": o.val, "aria-checked": sp.value === o.val ? "true" : "false",
            disabled: !o.enabled, title: o.title,
            onclick: (e) => { e.stopPropagation(); if (!o.enabled || m.soundtrack === o.val) return; m.soundtrack = o.val; draw(); updateRunButton(); scheduleRead(); } }, o.label));
        }
        box.append(seg);
      }
      if (m.items.some((i) => i.kind === "video")) {                  // EDIT MODE (video only): Overlay only (default) | With transitions. Rides in the job request + brief as `edit`.
        const eseg = el("div", { class: "m-sound m-edit", role: "radiogroup", "aria-label": "Edit" }, el("span", { class: "m-sound-l" }, "Edit"));
        for (const o of B.EDIT_OPTIONS) {
          const on = B.normEdit(m.edit) === o.val;
          eseg.append(el("button", { class: "seg" + (on ? " on" : ""), type: "button", role: "radio", "data-val": o.val, "aria-checked": on ? "true" : "false", title: o.title,
            onclick: (e) => { e.stopPropagation(); if (on) return; m.edit = o.val; draw(); } }, o.label));
        }
        box.append(eseg);
      }
      if (focusEl) focusEl.focus();
    };
    S.mediaDraw = draw;
    box.addEventListener("click", () => { if (!m.items.length) input.click(); });
    input.addEventListener("change", () => { addMediaFiles(input.files); input.value = ""; });
    box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("over"); });
    box.addEventListener("dragleave", () => box.classList.remove("over"));
    box.addEventListener("drop", (e) => { e.preventDefault(); box.classList.remove("over"); dropMedia(e.dataTransfer); });
    draw();
    const wrap = el("div", { class: "field media-field" }, box, input, msg,
      tlink, textWrap,
      el("div", { class: "media-read" }, stat, reread, show));
    S.mediaTextSync = () => { if (m.text.trim() === "") delete textWrap.dataset.open; syncText(); };
    syncText();
    paintMediaStatus();
    return wrap;
  }

  /** a read exists and its story is still to come (waiting / queued / writing): the work area shows the progress rows, never "Not read yet.". */
  function storyPending() {
    const st = S.story;
    return !!S.media.read && !st.unavailable && !st.err && (st.busy || ["waiting", "queued", "writing"].includes(st.stage));
  }

  function mediaStatusText() {
    const m = S.media;
    if (!mediaHasInput()) return "";
    if (m.err) return "Read failed: " + m.err;
    if (m.unavailable) return "Reading is not available on this server; your media and text are still used.";
    switch (m.stage) {
      case "waiting": return "Will read in a moment...";
      case "uploading": return "Uploading...";
      case "queued": return "Reading: queued behind another job on the box...";
      case "reading": return "Reading..." + (m.detail ? " " + m.detail : "");
      default: break;
    }
    if (m.read && (m.readSig === mediaSig() || storyPending())) {
      const n = m.read.on_screen.length, sTxt = storyStatusText();
      return `Read: ${n} on-screen line${n === 1 ? "" : "s"}${m.read.concept ? ", concept found" : ""}.` + (sTxt ? " " + sTxt : "");
    }
    return "Not read yet.";
  }

  function paintMediaStatus() {
    const m = S.media, ui = S.mediaUI;
    const text = mediaStatusText(), bad = !!(m.err || (S.story.err && m.read)), working = ((["waiting", "uploading", "queued", "reading"].includes(m.stage) && !m.err) || (storyWorking() && !!m.read)) && !bad;
    if (ui) {       // (no isConnected test: at boot the box is built before it is attached, and Re-read / Show storyboard must be hidden from the first paint)
      ui.msg.textContent = m.msg; ui.msg.hidden = true;   // user v18: no double-up, media messages show only in the work area (step list / notices, addNotice)
      ui.stat.textContent = text; ui.stat.className = "m-stat" + (working ? " working" : "") + (bad ? " bad" : "");
      ui.stat.hidden = true;   // user v19: no double-up, the read / story status shows only in the work area (step list); the Re-read link stays
      ui.reread.hidden = !mediaHasInput() || m.busy;
      ui.show.hidden = !m.read || !mediaHasInput() || S.view === "read";
    }
    for (const ro of document.querySelectorAll(".ro-state")) { ro.textContent = text; ro.className = "ro-state" + (ro.classList.contains("sb-state") ? " sb-state" : "") + (working ? " working" : "") + (bad ? " bad" : ""); }
    paintSteps();
  }

  // ---- PROGRESS STEPS (v17, 16_app/MEDIA_BOX.md): before the storyboard exists the work area lists the pipeline as rows, derived from the real state
  // (S.media.stage / m.read / S.story.stage) and appended as it advances: done rows go grey with a check, the active one is bright with a pulsing dot, a failed one is red.
  // Warnings (every non-info toast + media-box message) are amber rows at the point they happened; on the storyboard they become a dismissable notice stack.
  let noticeUid = 0;
  function readSummary() {
    const r = S.media.read;
    if (!r) return "";
    const n = (r.on_screen || []).length, bits = [`Read: ${n} on-screen line${n === 1 ? "" : "s"}`];
    if (r.concept) bits.push("concept found");
    const cuts = Array.isArray(r.video_cuts) ? r.video_cuts.length : 0;
    if (cuts) bits.push(`${cuts} cut${cuts === 1 ? "" : "s"} found`);
    if (Array.isArray(r.lyrics) && r.lyrics.length) bits.push("lyrics heard");
    return bits.join(", ");
  }
  /** the pipeline rows right now: [{key, state: done|active|fail|note, text, sub?, act?: {label, fn}}] (empty = nothing has started). */
  function stepRows() {
    const m = S.media, st = S.story, rows = [];
    if (!mediaHasInput()) return rows;
    if (m.unavailable) return [{ key: "read", state: "note", text: "Reading is not available on this server", sub: "Your media and text are still used." }];
    if (m.stage === "waiting" && !m.err) return [{ key: "wait", state: "active", text: "Will read in a moment" }];
    const files = m.items.length, fileTxt = files + (files === 1 ? " file" : " files");
    const live = ["uploading", "queued", "reading"].includes(m.stage) && !m.err;
    const cur = !!(m.read && (m.readSig === mediaSig() || (storyPending() && !live)));   // a pending story keeps the list on screen even for the instant the read's signature is stale
    const reread = { label: "Re-read", fn: () => { if (S.mediaUI) S.mediaUI.reread.click(); } };
    if (files) {
      if (m.stage === "uploading" && !m.err) rows.push({ key: "up", state: "active", text: "Uploading your media", sub: fileTxt });
      else if (m.err && !m.items.every((i) => i.uploaded)) { rows.push({ key: "up", state: "fail", text: "Uploading your media failed", sub: m.err, act: reread }); return rows; }
      else if (live || cur || m.err) rows.push({ key: "up", state: "done", text: `Uploaded ${fileTxt}` });
    }
    if (m.stage === "queued" && !m.err) rows.push({ key: "read", state: "active", text: "Reading your media", sub: "queued behind another job" });
    else if (m.stage === "reading" && !m.err) rows.push({ key: "read", state: "active", text: "Reading your media", sub: String(m.detail || "").replace(/^\((.*)\)$/, "$1") });
    else if (m.err) rows.push({ key: "read", state: "fail", text: "Reading your media failed", sub: m.err, act: reread });
    else if (cur) rows.push({ key: "read", state: "done", text: readSummary() });
    if (cur && !live && !m.err && !st.unavailable) {
      if (st.err) rows.push({ key: "story", state: "fail", text: "Writing the story failed", sub: st.err, act: { label: "Try again", fn: rewriteStory } });
      else if (st.stage === "waiting") rows.push({ key: "story", state: "active", text: "Writing the story", sub: "starting in a moment" });
      else if (st.stage === "queued") rows.push({ key: "story", state: "active", text: "Writing the story", sub: "queued behind another job" });
      else if (st.stage === "writing") rows.push({ key: "story", state: "active", text: "Writing the story" });
      else if (hasCurrentStory()) rows.push({ key: "story", state: "done", text: `Wrote the story: ${st.data.shots.length} shot${st.data.shots.length === 1 ? "" : "s"}` });
    }
    return rows;
  }
  const liveNotices = () => (S.notices || []).filter((n) => !n.dismissed);
  /** write a warning into the work area. The same text again is the same notice, refreshed. */
  function addNotice(text) {
    try {
      const t = String(text || "").trim();
      if (!t) return;
      const list = S.notices || (S.notices = []);
      const hit = list.find((n) => !n.dismissed && n.text === t);
      if (hit) hit.at = Date.now();
      else {
        const rows = stepRows();
        list.push({ id: ++noticeUid, text: t, at: Date.now(), after: rows.length ? rows[rows.length - 1].key : null, dismissed: false });   // `after`: the last row when it happened
        if (list.length > 30) list.splice(0, list.length - 30);
      }
      paintNotices();
    } catch (_) { /* a notice must never break its caller */ }
  }
  function dismissNotice(id) {
    const n = (S.notices || []).find((x) => x.id === id);
    if (n) { n.dismissed = true; paintNotices(); }
  }
  function paintNotices() {
    paintSteps();
    if (document.querySelector(".work .storyboard")) renderWork();
  }
  /** rows + warnings in the order they happened. */
  function stepItems() {
    const rows = stepRows(), ns = liveNotices(), out = [];
    const note = (n) => ({ key: "n" + n.id, state: "warn", text: n.text, dismiss: () => dismissNotice(n.id) });
    ns.filter((n) => n.after === null).forEach((n) => out.push(note(n)));
    rows.forEach((r) => { out.push(r); ns.filter((n) => n.after === r.key).forEach((n) => out.push(note(n))); });
    ns.filter((n) => n.after !== null && !rows.some((r) => r.key === n.after)).forEach((n) => out.push(note(n)));
    return out;
  }
  function fillStep(node, it) {
    const sig = JSON.stringify([it.state, it.text, it.sub || "", it.act ? it.act.label : "", !!it.dismiss]);
    if (node.dataset.sig === sig) return;
    node.dataset.sig = sig;
    node.className = "step " + it.state + (node.classList.contains("enter") ? " enter" : "");
    node.textContent = "";
    const ic = el("span", { class: "step-ic", "aria-hidden": "true" });
    if (it.state === "done") ic.innerHTML = svg("check", 15);
    else if (it.state === "fail") ic.textContent = "×";
    else if (it.state === "warn") ic.textContent = "!";
    else if (it.state === "active") ic.append(el("span", { class: "step-dot" }));
    const body = el("div", { class: "step-body" }, el("div", { class: "step-t" }, it.text));
    if (it.sub) body.append(el("div", { class: "step-sub" }, it.sub));
    if (it.act) body.append(el("button", { class: "link-btn step-act", type: "button", onclick: it.act.fn }, it.act.label));
    node.append(ic, body);
    if (it.dismiss) node.append(el("button", { class: "step-x", type: "button", title: "Dismiss", "aria-label": "Dismiss", onclick: it.dismiss }, "×"));
  }
  /** keep the .steps list in step with the state: rows are keyed, so an existing row keeps its node (no restarted animation) and a new row slides in. */
  function paintSteps() {
    const box = document.querySelector(".work .steps");
    if (!box) return;
    const items = stepItems(), keep = new Set(items.map((i) => i.key));
    for (const node of [...box.children]) if (!keep.has(node.dataset.key)) node.remove();
    let prev = null;
    for (const it of items) {
      let node = [...box.children].find((c) => c.dataset.key === it.key);
      if (!node) { node = el("div", { "data-key": it.key }); if (box.dataset.live) node.classList.add("enter"); }
      fillStep(node, it);
      const want = prev ? prev.nextSibling : box.firstChild;
      if (node !== want) box.insertBefore(node, want);
      prev = node;
    }
    box.dataset.live = "1";
  }
  function buildSteps() {
    const box = el("div", { class: "steps", role: "status", "aria-live": "polite" });
    for (const it of stepItems()) { const node = el("div", { "data-key": it.key }); fillStep(node, it); box.append(node); }
    box.dataset.live = "1";
    return box;
  }
  /** the storyboard's notice stack: the 3 newest warnings, newest first, each dismissable. */
  function buildNotices() {
    const list = liveNotices().sort((a, b) => b.at - a.at || b.id - a.id).slice(0, 3);
    if (!list.length) return null;
    return el("div", { class: "sb-notices", role: "status" }, list.map((n) => el("div", { class: "sb-notice", "data-id": String(n.id) },
      el("span", { class: "step-ic" }, "!"), el("span", { class: "sb-notice-t" }, n.text),
      el("button", { class: "step-x", type: "button", title: "Dismiss", "aria-label": "Dismiss", onclick: () => dismissNotice(n.id) }, "×"))));
  }

  // ---- STORYBOARD v12: shots never exceed images, so a Length is only offered when every shot's clip fits (B.shotsFor / B.feasibleLengths; 16_app/MEDIA_BOX.md)
  const imageCount = () => S.media.items.filter((i) => i.kind === "image").length;
  // ---- VIDEO MODE (16_app/MEDIA_BOX.md "VIDEO MODE STORYBOARD"): the Length can never be longer than the video
  const videoItem = () => S.media.items.find((i) => i.kind === "video") || null;
  /** the video's length in s: the dropped file's own duration, else the READ's, else the story's capacity; NaN while unknown. */
  function videoDur() {
    const v = videoItem();
    if (!v) return NaN;
    const r = S.media.read, cap = S.story.data && S.story.data.capacity;
    for (const d of [v.duration, r && r.video_duration, cap && cap.video_duration]) if (Number(d) > 0 && Number.isFinite(Number(d))) return Number(d);
    return NaN;
  }
  const baseLengthOptions = (sel) => [...sel.options].filter((o) => !o.dataset.vlen).map((o) => Number(o.value));
  /** grey out the Lengths the images cannot fill, say what would unlock them (small line under the select, only when it matters).
   *  With a video: Lengths longer than the video are greyed ("the video is 12 s"); a video shorter than the smallest Length adds its own duration as the Length. */
  /** video mode renders 9:16 only (own_video_route FORMATS): 16:9 is greyed with the reason, and a 16:9 choice drops back to 9:16. */
  function paintFormat() {
    const fs = document.querySelector("select[data-id=format]"), vid = !!videoItem();
    if (!fs) return;
    for (const o of fs.options) { if (o.dataset.lbl === undefined) o.dataset.lbl = o.textContent; const off = vid && o.value !== "9:16"; o.disabled = off; o.textContent = off ? `${o.dataset.lbl} (video is 9:16 only)` : o.dataset.lbl; }
    if (vid && S.values.format !== "9:16") { fs.value = "9:16"; S.values.format = "9:16"; }
  }
  function paintLength() {
    const sel = document.querySelector("select[data-id=length]"), note = document.querySelector(".len-note");
    paintFormat();
    if (!sel) return;
    const lines = [], dur = videoDur(), opts = baseLengthOptions(sel);
    if (videoItem()) {
      const plan = B.videoLengthPlan(opts, dur, S.values.length);
      for (const o of [...sel.options]) if (o.dataset.vlen && (plan.extra === null || Number(o.value) !== plan.extra)) o.remove();
      if (plan.extra !== null && ![...sel.options].some((o) => o.dataset.vlen)) {            // v16: the clip's own duration when it is not one of the standard Lengths, in numeric order
        const o = document.createElement("option"); o.value = String(plan.extra); o.dataset.vlen = "1"; o.dataset.lbl = `${plan.extra} s`; o.textContent = o.dataset.lbl;
        const after = [...sel.options].find((x) => Number(x.value) > plan.extra);
        sel.insertBefore(o, after || null);
      }
      for (const o of sel.options) {
        if (o.dataset.lbl === undefined) o.dataset.lbl = o.textContent;
        const off = !o.dataset.vlen && !!plan.disabled[Number(o.value)];                      // longer than the clip: visible, disabled, the reason in the label
        o.disabled = off; o.title = off ? `the video is ${B.fmtDur(dur)} s` : "";
        o.textContent = off ? `${o.dataset.lbl} (video is ${B.fmtDur(dur)} s)` : o.dataset.lbl;
      }
      sel.disabled = false;
      if (Number.isFinite(dur)) lines.push(`The video is ${B.fmtDur(dur)} s: the story is built from its own cuts.`);
      if (note) { note.textContent = lines.join(" "); note.hidden = !lines.length; }
      return;
    }
    for (const o of [...sel.options]) if (o.dataset.vlen) o.remove();
    const n = imageCount();
    for (const o of sel.options) {
      if (o.dataset.lbl !== undefined) o.textContent = o.dataset.lbl;                         // image mode: the plain label again (v16: video mode appended "(video is N s)")
      const L = Number(o.value), need = B.imagesNeeded(L, n);
      o.disabled = need > 0;
      o.title = need > 0 ? `add ${need} image${need === 1 ? "" : "s"} for ${L} s` : "";
    }
    const off = opts.filter((L) => B.imagesNeeded(L, n) > 0);
    if (off.length) lines.push(off.sort((a, b) => b - a).map((L) => `${L} s needs ${B.imagesNeeded(L, n)} more image${B.imagesNeeded(L, n) === 1 ? "" : "s"}`).join(", ") + ".");
    const cur = Number(S.values.length) || 0, shots = B.shotsFor(cur, n);
    if (n >= 2 && cur && shots < n) lines.push(`${cur} s holds ${shots} of your ${n} images.`);
    if (note) { note.textContent = lines.join(" "); note.hidden = !lines.length; }
  }
  /** an image was added / removed (or the video's length became known): if the current Length is no longer possible drop to the longest that is, and say so. */
  function syncLength() {
    const sel = document.querySelector("select[data-id=length]");
    if (!sel) return;
    const cur = Number(S.values.length);
    paintLength();
    if (videoItem()) {
      const plan = B.videoLengthPlan(baseLengthOptions(sel), videoDur(), cur);
      if (!plan.changed || !Number.isFinite(plan.length)) return;
      sel.value = String(plan.length); S.values.length = sel.value;
      paintLength();
      toast(plan.extra !== null ? `The video is ${B.fmtDur(plan.duration)} s; Length is now ${plan.length} s.` : `${cur} s is longer than the video (${B.fmtDur(plan.duration)} s); Length is now ${plan.length} s.`, 6000);
      scheduleStory(2000);
      return;
    }
    if (sel.value !== String(cur) || !Number.isFinite(cur)) { sel.value = "12"; S.values.length = sel.value; paintLength(); scheduleStory(2000); return; }   // the video's own Length option went away with it
    const n = imageCount();
    if (!Number.isFinite(cur) || B.lengthFeasible(cur, n)) return;
    const opts = baseLengthOptions(sel), to = B.bestFeasibleLength(cur, n, opts);
    if (to === cur) return;
    sel.value = String(to); S.values.length = sel.value;
    paintLength();
    toast(`${cur} s needs more images than you have (${n}); Length is now ${to} s. Add ${B.imagesNeeded(cur, n)} more for ${cur} s.`, 7000);
    scheduleStory(2000);
  }

  /** debounced (2.5 s after the last add / remove / reorder / text edit); runs only when the MEDIA JSON changed since the last completed read. */
  function scheduleRead() {
    const m = S.media;
    syncLength();
    if (!mediaField()) return;
    clearTimeout(m.timer); m.timer = null;
    m.err = "";
    if (!mediaHasInput()) { clearRead(); return; }
    if (m.busy) { m.again = true; paintMediaStatus(); return; }          // let the running read finish, then read once more
    m.stage = "waiting";
    m.timer = setTimeout(() => { m.timer = null; maybeRead(); }, 2500);
    paintMediaStatus();
  }

  function maybeRead() {
    const m = S.media;
    if (m.busy) { m.again = true; return null; }
    const sig = mediaSig();
    if (!mediaHasInput() || m.unavailable || sig === m.readSig || sig === m.failSig) { m.stage = "idle"; paintMediaStatus(); return null; }
    return doRead(false);
  }

  function clearRead() {
    const m = S.media;
    m.read = null; m.readGemma = null; m.readSig = null; m.failSig = null; m.err = ""; m.stage = "idle"; m.edited = false; m.again = false;
    S.notices = [];
    clearStory();
    if (S.view === "read" && !activeRun()) { S.view = S.exampleShown ? "example" : "examples"; renderCanvas(); }
    paintMediaStatus();
  }

  async function uploadPending() {
    for (let guard = 0; guard < 30; guard++) {
      const it = S.media.items.find((i) => !i.uploaded);
      if (!it) return;
      it.uploaded = await upload(it.file);
    }
  }

  /** keep the user's edited fields, refresh the untouched ones. */
  function mergeRead(fresh) {
    const old = S.media.read, e = ed();
    if (!old) return fresh;
    const out = fresh;
    if (e.on_screen) out.on_screen = old.on_screen;
    if (e.concept) out.concept = old.concept;
    if (e.look) out.look = old.look;
    out.media = out.media.map((x) => {
      const o = e.media[x.file] && old.media.find((y) => y.file === x.file);
      return o ? Object.assign({}, x, { shows: o.shows, window: o.window, mood: o.mood }) : x;
    });
    return out;
  }

  /** upload what is pending, queue `_media_read`, read `OUT read`. Resolves when done; the error (if any) is in S.media.err. */
  function doRead(force) {
    const m = S.media;
    if (m.busy) { m.again = true; return m.promise; }
    if (!mediaHasInput()) return Promise.resolve();
    clearTimeout(m.timer); m.timer = null;
    m.busy = true; m.err = ""; m.again = false; m.detail = "";
    m.stage = "uploading";
    if (["examples", "example", "read"].includes(S.view) && !activeRun()) { S.view = "read"; renderCanvas(); }
    paintMediaStatus(); updateRunButton();
    m.promise = (async () => {
      let sig = null;
      try {
        await uploadPending();
        const items = m.items.map((i) => ({ kind: i.kind, name: i.uploaded }));
        sig = mediaSig();                                            // what this read is for: a later change makes the result stale
        const media = B.buildMediaJson(items, m.text, S.values.length, S.values.format, soundtrackNow());
        let t;
        try { t = await getJSON("/mograph/graphs/_media_read"); }
        catch (e) { if (/: 404$/.test(String(e.message))) { m.unavailable = true; m.stage = "idle"; return; } throw e; }
        const g = t.graph, tm = B.titleMap(g);
        if (!("IN media" in tm) || !("OUT read" in tm)) throw new Error("the _media_read graph has no IN media / OUT read node");
        g[tm["IN media"]].inputs.value = JSON.stringify(media);
        for (const n of Object.values(g)) if (n.inputs && "filename_prefix" in n.inputs) n.inputs.filename_prefix = "media_read/read";   // queueSide labels it motngrfx_app/<run>_read
        m.stage = "queued"; paintMediaStatus();
        const reply = await queueSide(g, {
          node: tm["OUT read"], label: "the media read", timeout: 900,
          pick: (out) => (out && Array.isArray(out.text) && out.text.length ? out.text[0] : null),
          onStatus: (txt) => { m.stage = "reading"; m.detail = txt === "queued" ? "" : `(${txt})`; paintMediaStatus(); },
        });
        const fresh = B.normalizeRead(reply);
        if (mediaSig() === sig) {
          m.readGemma = B.clone(fresh);                               // Gemma's latest read per field (the storyboard's revert targets)
          S.story.edited.use = false;                                // image ids are positions: a new read drops a frame swap
          m.read = mergeRead(fresh); m.readSig = sig; m.failSig = null;
          { const vi = videoItem(), ve = vi && fresh.media.find((x) => x.kind === "video");       // v19: the read tells whether the clip has an audio stream ("Video sound" is off when it has none)
            if (vi && ve && typeof ve.has_audio === "boolean" && vi.hasAudio !== ve.has_audio) {
              vi.hasAudio = ve.has_audio;
              m.readSig = mediaSig();                                    // the pick may have moved (no sound -> New music): reads of "video" and "generate" are the same for a silent clip
              redrawMedia();
            } }
          const len0 = S.values.length; syncLength();                // video mode: the READ's video_duration may cap the Length
          if (S.values.length !== len0) m.again = true;              // the Length changed: the read (its window) is stale, read once more
          else scheduleStory(300);                                   // the storyboard follows every completed read
        }                                                            // else: inputs changed while reading; ignore it (m.again re-reads)
        else m.again = true;
      } catch (e) {
        m.err = errText(e); m.failSig = sig;
      } finally {
        m.busy = false; if (m.stage !== "idle" || m.err) m.stage = "idle";
        const again = m.again; m.again = false;
        if (S.view === "read" && !activeRun()) renderCanvas();
        paintMediaStatus(); updateRunButton();
        if (again) scheduleRead();
      }
    })();
    return m.promise;
  }

  /** Make it: wait for a pending / running read, re-read if the inputs changed since. false = the read failed. */
  async function ensureRead() {
    const m = S.media;
    for (let i = 0; i < 5; i++) {
      clearTimeout(m.timer); m.timer = null;
      if (!mediaHasInput() || m.unavailable) return true;
      if (m.busy) { await m.promise; continue; }
      if (m.readSig === mediaSig()) return true;
      await doRead(true);
      if (m.err) return false;
    }
    return m.readSig === mediaSig();
  }

  /** the edited READ JSON (text), ids following the current image order. */
  function readBrief() {
    const m = S.media;
    if (!m.read) return "";
    const r = B.normalizeRead(m.read), ids = B.mediaIds(m.items);
    for (const e of r.media) { const it = itemOfFile(e.file); if (it) e.id = ids[m.items.indexOf(it)]; }
    return B.briefString(r);
  }

  /** the brief to send: the edited READ; with a current story, on_screen / concept / look come from the (edited) storyboard + `story` (B.briefFromStory). */
  function currentBrief() {
    const base = readBrief(), typed = S.media.text;      // the typed "Words or direction" text rides along as `direction` (B.withDirection): the user's style wishes override the defaults
    if (!base || !hasCurrentStory()) return B.withDirection(base, typed);
    const userFonts = S.story.data.shots.filter((x) => S.story.edited[`${x.n}.font`] && x.font).map((x) => x.n);      // only fonts the user changed are pins (font_user); the random default is not
    try { return B.withDirection(B.briefFromStory(base, S.story.data, { lookEdited: !!ed().look, userFonts }), typed); } catch (_) { return B.withDirection(base, typed); }
  }

  // ------------------------------------------------------------------ STORY step (16_app/MEDIA_BOX.md "STORY step")
  const storyOn = () => !!(mediaField() && S.media.read && !S.story.unavailable);
  const storySeed = () => (B.isEmpty({ kind: "seed" }, S.values.seed) ? S.story.seed : Number(S.values.seed));
  const storySig = () => B.storySignature(readBrief(), S.values.length, storySeed(), S.values.format);
  const hasCurrentStory = () => !!(S.story.data && S.story.sig && S.story.sig === storySig());
  const storyWorking = () => ["waiting", "queued", "writing"].includes(S.story.stage);

  function clearStory() {
    const st = S.story;
    clearTimeout(st.timer);
    Object.assign(st, { timer: null, data: null, gemma: null, sig: null, failSig: null, err: "", stage: "idle", again: false, edited: {}, vers: {}, openNotes: {}, active: null });
  }

  /** debounced (default 2 s) after a completed read, a Length change or an edit of the readout; runs only when what the story depends on changed. */
  function scheduleStory(delay = 2000) {
    const st = S.story;
    if (!mediaField()) return;
    clearTimeout(st.timer); st.timer = null;
    if (!mediaHasInput()) { clearStory(); return; }
    if (!S.media.read || st.unavailable) return;
    st.err = "";
    if (st.busy) { st.again = true; paintMediaStatus(); return; }
    st.stage = "waiting";
    st.timer = setTimeout(() => { st.timer = null; maybeStory(); }, delay);
    paintMediaStatus();
  }

  function maybeStory() {
    const st = S.story, m = S.media;
    if (st.busy) { st.again = true; return null; }
    if (!storyOn() || m.busy || m.readSig !== mediaSig()) { st.stage = "idle"; paintMediaStatus(); return null; }   // a read is still due: its end schedules the story
    const sig = storySig();
    if (sig === st.sig || sig === st.failSig) { st.stage = "idle"; paintMediaStatus(); return null; }
    return doStory();
  }

  /** the context the pure story helpers need: the image ids in the user's order, the Length and its options, the pasted text's tokens, the read's note per image. */
  /** the READ's beat grid (window seconds in video mode) for landing hand-picked words on a beat; [] when the read has none. */
  const gridBeats = () => { const g = S.media.read && S.media.read.beat_grid; return g && Array.isArray(g.beats) ? g.beats.map(Number).filter(Number.isFinite) : []; };
  function storyCtx() {
    const m = S.media, ids = B.mediaIds(m.items), sel = document.querySelector("select[data-id=length]");
    const shows = {};
    m.items.forEach((it, i) => { if (it.kind !== "image" || !m.read) return; const e = readEntry(it); if (e && e.shows) shows[ids[i]] = e.shows; });
    return { imageIds: m.items.map((it, i) => (it.kind === "image" ? ids[i] : null)).filter(Boolean), length: Number(S.values.length) || 12,
             lengthOptions: sel ? [...sel.options].filter((o) => !o.dataset.vlen).map((o) => Number(o.value)).filter(Number.isFinite) : null, tokens: textSource().tokens, times: textSource().times, shows, beats: gridBeats() };
  }
  /** the words the "Your text" panel lists: the pasted text; when none was pasted and the read heard lyrics (Whisper, the video's audio or the audio file), the lyrics (each token has its sung time). */
  function textSource() {
    const m = S.media;
    const typedWords = B.splitTyped(m.text).words;                // the box text minus its direction clauses (palette / legibility / plain / instructions): only words are story candidates; the server splits the same way (user_style.split_typed)
    if (typedWords.trim()) return { tokens: B.textTokens(typedWords), times: null, lyrics: false, from: "" };
    const ly = m.read && m.read.lyrics;
    if (Array.isArray(ly) && ly.length) { const t = B.lyricTokens(ly); if (t.tokens.length) return { tokens: t.tokens, times: t.times, lyrics: true, from: videoItem() && soundtrackNow() !== "track" ? "video" : "audio" }; }
    return { tokens: [], times: null, lyrics: false, from: "" };
  }
  const readEntry = (it, read) => ((read || S.media.read) ? (read || S.media.read).media.find((x) => it.uploaded && (x.file === it.uploaded || it.uploaded.endsWith("/" + x.file) || x.file.endsWith("/" + it.uploaded))) : null);

  /** give every shot without a valid font one from the liked list (random, not used by another line, role unlike the line before). `reserved` = fonts the user already picked. The result is Gemma's default. */
  function fillFonts(story, reserved) {
    const list = S.fonts.list;
    if (!story || !list.length) return story;
    const valid = (f) => list.some((x) => x.file === f);
    const taken = [...story.shots.filter((x) => valid(x.font)).map((x) => x.font), ...(reserved || [])];
    story.shots.forEach((x, i) => {
      if (valid(x.font)) return;
      const prev = story.shots[i - 1];
      x.font = B.fontPick(list, taken, prev && prev.font ? [B.fontRole(list, prev.font)] : []);
      taken.push(x.font);
    });
    return story;
  }

  /** a fresh story from the model arrives: it becomes the new Gemma value for every field; what the user edited is kept on top (until they revert). */
  function mergeStory(fresh) {
    const st = S.story, old = st.data, e = st.edited, ctx = storyCtx();
    if (old) syncAllVers(false);                                   // Gemma's value that is about to be replaced becomes a version of its field
    const mine = old ? old.shots.filter((x) => e[`${x.n}.font`]).map((x) => x.font) : [];
    if (ctx.imageIds.length >= 2) fresh = B.capShots(fresh, B.shotsFor(ctx.length, ctx.imageIds.length));   // v12: never more shots than images (the page enforces it whatever the server sent)
    if (videoItem()) { const r = S.media.read, ids = B.mediaIds(S.media.items); fresh = B.videoStoryFix(fresh, { cuts: r && r.video_cuts, duration: videoDur() }, Number(S.values.length) || 12, ids[S.media.items.indexOf(videoItem())]); }   // video mode: never invented shots
    fresh = fillFonts(B.completeStory(fresh, ctx), mine);
    const og = st.gemma, gnew = B.clone(fresh);
    // the frames / words the user chose came back honoured by the model: Gemma's own pick (what revert restores) stays the previous one
    if (og && e.use && og.shots.length === gnew.shots.length) { gnew.shots.forEach((x, i) => { if (x.image !== og.shots[i].image) { x.image = og.shots[i].image; x.beat = og.shots[i].beat; } }); gnew.candidates = B.clone(og.candidates); }
    if (og && e.sel && st.selText === S.media.text && og.shots.length === gnew.shots.length) { gnew.text_select = B.clone(og.text_select); gnew.shots.forEach((x, i) => { x.line = og.shots[i].line; }); }
    st.gemma = gnew;
    if (!old) { st.edited = {}; return fresh; }
    const sameN = old.shots.length === fresh.shots.length;
    if (!sameN) for (const k of Object.keys(st.vers)) if (/^\d+\./.test(k)) delete st.vers[k];   // other shots now: their version lists start over
    let out = fresh;
    if (e.use && B.usedIds(old).every((id) => ctx.imageIds.includes(id))) out = B.deriveStory(out, { use: B.usedIds(old), imageIds: ctx.imageIds, shows: ctx.shows }).story;
    else delete e.use;
    if (e.sel && st.selText === S.media.text) out = B.deriveStory(out, { select: old.text_select, tokens: ctx.tokens, times: ctx.times, beats: ctx.beats }).story;
    else delete e.sel;
    for (const k of ["title", "logline", "look", "music"]) if (e[k]) out[k] = old[k];
    for (const key of Object.keys(e)) {
      const m = /^(\d+)\.(line|beat|camera|font)$/.exec(key);
      if (!m) continue;
      const o = old.shots.find((x) => x.n === Number(m[1])), x = out.shots.find((y) => y.n === Number(m[1]));
      if (!o || !x || (m[2] !== "font" && !sameN)) { delete e[key]; continue; }
      if (m[2] === "beat" && o.image !== x.image) { delete e[key]; continue; }
      x[m[2]] = o[m[2]];
    }
    return out;
  }

  /** what the user decided that the next model call must honour (STORYBOARD v2: req.use / req.select / req.text / req.fonts). */
  function storyExtras() {
    const st = S.story, d = st.data, e = st.edited, x = { text: S.media.text };
    if (d && e.use) x.use = B.usedIds(d);
    if (d && e.sel) x.select = d.text_select;
    if (d) { const f = {}; for (const s of d.shots) if (e[`${s.n}.font`] && s.font) f[String(s.n)] = s.font; if (Object.keys(f).length) x.fonts = f; }
    return x;
  }

  /** queue `_story` (like the read) and read `OUT story`. Resolves when done; the error (if any) is in S.story.err. 404 on the graph = no story on this server. */
  function doStory() {
    const st = S.story;
    if (st.busy) { st.again = true; return st.promise; }
    clearTimeout(st.timer); st.timer = null;
    st.busy = true; st.err = ""; st.again = false; st.stage = "writing";
    if (["examples", "example"].includes(S.view) && !activeRun()) { S.view = "read"; renderCanvas(); }
    paintMediaStatus(); updateRunButton(); paintVers();            // v14: the › arrows of title / logline / look / music show busy while the story is written
    st.promise = (async () => {
      let sig = null;
      try {
        sig = storySig();
        const req = B.buildStoryRequest(readBrief(), S.values.length, storySeed(), S.values.format, storyExtras());
        let t;
        try { t = await getJSON("/mograph/graphs/_story"); }
        catch (e) { if (/: 404$/.test(String(e.message))) { st.unavailable = true; st.stage = "idle"; return; } throw e; }
        const g = t.graph, tm = B.titleMap(g);
        if (!("IN story" in tm) || !("OUT story" in tm)) throw new Error("the _story graph has no IN story / OUT story node");
        g[tm["IN story"]].inputs.value = JSON.stringify(req);
        for (const n of Object.values(g)) if (n.inputs && "filename_prefix" in n.inputs) n.inputs.filename_prefix = "story/story";
        const reply = await queueSide(g, {
          node: tm["OUT story"], label: "the story", timeout: 900,
          pick: (out) => (out && Array.isArray(out.text) && out.text.length ? out.text[0] : null),
          onStatus: () => { if (st.stage !== "writing") { st.stage = "writing"; paintMediaStatus(); } },
          onQueued: () => { if (st.stage !== "queued") { st.stage = "queued"; paintMediaStatus(); } },     // only while it really waits in the box queue
        });
        const fresh = B.normalizeStory(reply);
        if (storySig() === sig && S.media.read) { st.data = mergeStory(fresh); st.sig = sig; st.failSig = null; syncAllVers(false); }
        else st.again = true;                                          // the inputs changed while it was written
      } catch (e) {
        st.err = errText(e); st.failSig = sig;
      } finally {
        st.busy = false; st.stage = "idle";
        const again = st.again; st.again = false;
        renderWork();
        paintMediaStatus(); updateRunButton();
        if (again) scheduleStory(300);
      }
    })();
    return st.promise;
  }

  /** Make it: wait for a pending / running story, write it again if the inputs changed since. false = it failed (Make it then goes on without it). */
  async function ensureStory() {
    const st = S.story, m = S.media;
    for (let i = 0; i < 5; i++) {
      clearTimeout(st.timer); st.timer = null;
      if (!mediaHasInput() || !m.read || st.unavailable) return true;
      if (st.busy) { await st.promise; continue; }
      if (hasCurrentStory()) return true;
      await doStory();
      if (st.err) return false;
    }
    return hasCurrentStory();
  }

  /** "Write again": a new story for the same inputs (a new seed unless the Seed field is set). */
  function rewriteStory() {
    const st = S.story;
    if (st.busy || !S.media.read || activeRun()) return;
    if (B.isEmpty({ kind: "seed" }, S.values.seed)) st.seed = Math.floor(Math.random() * 4294967296);
    st.sig = null; st.failSig = null;                  // the user's edits stay on top of the new story (revert brings Gemma's back)
    doStory();
  }

  function storyStatusText() {
    const st = S.story;
    if (st.unavailable || !S.media.read) return "";
    if (st.err) return "Story failed: " + st.err;
    if (st.stage === "waiting") return "Story in a moment...";
    if (st.stage === "queued") return "Story: queued behind another job on the box...";
    if (st.stage === "writing") return "Writing the story...";
    if (hasCurrentStory()) return `Story: ${st.data.shots.length} shot${st.data.shots.length === 1 ? "" : "s"}.`;
    return "";
  }

  /** the bundle a media field puts into the graph (bind.js applyMedia): images in box order. */
  function mediaBundle() {
    const m = S.media;
    return { images: m.items.filter((i) => i.kind === "image").map((i) => i.uploaded), audio: soundtrackNow() === "track" ? ((m.items.find((i) => i.kind === "audio") || {}).uploaded || "") : "",       // v19: the file only rides along when "My track" is chosen
             brief: currentBrief() };
  }

  // ---- readout (editable) on the canvas
  function buildReadout() {
    const m = S.media, rd = m.read;
    const card = el("div", { class: "readout" });
    card.append(el("div", { class: "ro-head" }, el("h3", {}, "Readout"), el("span", { class: "ro-state" })));
    if (!rd) {
      card.append(el("div", { class: "ro-empty" }, mediaHasInput()
        ? (m.err ? "The read did not work. Press Re-read in the panel, or Make it without it." : "Reading your media and text...")
        : "Add media or paste text and it is read here."));
      setTimeout(paintMediaStatus, 0);
      return card;
    }
    card.append(el("div", { class: "ro-sub" }, "Edit anything. Make it uses exactly this."));

    // On screen: one input per line
    const lines = el("div", { class: "ro-lines" });
    const drawLines = () => {
      lines.textContent = "";
      rd.on_screen.forEach((t, i) => {
        const inp = el("input", { type: "text", value: t, "aria-label": `On-screen line ${i + 1}` });
        inp.addEventListener("input", () => { rd.on_screen[i] = inp.value; ed().on_screen = true; scheduleStory(); });
        lines.append(el("div", { class: "ro-line" }, inp,
          el("button", { class: "icon-btn", type: "button", title: "Remove line", html: svg("x", 12), onclick: () => { rd.on_screen.splice(i, 1); ed().on_screen = true; scheduleStory(); drawLines(); } })));
      });
      if (!rd.on_screen.length) lines.append(el("div", { class: "ro-none" }, "None: the words are invented."));
      lines.append(el("button", { class: "link-btn", type: "button", onclick: () => { rd.on_screen.push(""); ed().on_screen = true; scheduleStory(); drawLines(); lines.querySelectorAll("input").forEach((x, k, a) => { if (k === a.length - 1) x.focus(); }); } }, "+ add line"));
    };
    drawLines();
    card.append(el("div", { class: "ro-sec" }, el("label", { class: "f-label" }, "On screen"), lines));

    const area = (key, label, rows) => {
      const ta = el("textarea", { rows, spellcheck: "false", placeholder: "Empty: invented." });
      ta.value = rd[key];
      ta.addEventListener("input", () => { rd[key] = ta.value; ed()[key] = true; scheduleStory(); });
      return el("div", { class: "ro-sec" }, el("label", { class: "f-label" }, label), ta);
    };
    card.append(area("concept", "Concept", 2), area("look", "Look", 2));

    // per-media notes, in the current box order, keyed by file
    const ids = B.mediaIds(m.items), notes = el("div", { class: "ro-media" });
    m.items.forEach((it, idx) => {
      const e = rd.media.find((x) => it.uploaded && (x.file === it.uploaded || it.uploaded.endsWith("/" + x.file) || x.file.endsWith("/" + it.uploaded)));
      if (!e) return;
      const thumb = it.kind === "image" ? el("img", { src: it.url, alt: "" }) : it.poster ? el("img", { src: it.poster, alt: "" })
        : el("div", { class: "ro-ph", html: svg(it.kind === "audio" ? "music" : "film", 16) });
      const shows = el("input", { type: "text", value: e.shows, placeholder: "What it shows", "aria-label": `${ids[idx]} shows` });
      shows.addEventListener("input", () => { e.shows = shows.value; ed().media[e.file] = true; scheduleStory(); });
      const row = el("div", { class: "ro-m" }, el("div", { class: "ro-thumb" }, thumb, el("span", { class: "m-id" }, ids[idx])),
        el("div", { class: "ro-m-body" }, shows));
      if (Array.isArray(e.window)) {
        const num = (k) => {
          const n = el("input", { type: "number", step: "0.1", min: "0", value: String(e.window[k]), "aria-label": k ? "Window end" : "Window start" });
          n.addEventListener("input", () => { const v = Number(n.value); if (Number.isFinite(v)) { e.window[k] = v; ed().media[e.file] = true; scheduleStory(); } });
          return n;
        };
        row.querySelector(".ro-m-body").append(el("div", { class: "ro-win" }, "Window ", num(0), " to ", num(1), " s"));
      }
      if (e.mood) row.querySelector(".ro-m-body").append(el("div", { class: "ro-mood" }, e.mood));
      notes.append(row);
    });
    if (notes.children.length) card.append(el("div", { class: "ro-sec" }, el("label", { class: "f-label" }, "Media"), notes));
    if (rd.music && (rd.music.bpm || rd.music.key)) card.append(el("div", { class: "ro-note" }, "Music: " + [rd.music.bpm ? `${Number(rd.music.bpm).toFixed(0)} bpm` : "", rd.music.key || ""].filter(Boolean).join(" · ")));
    if (rd.notes) card.append(el("div", { class: "ro-note" }, rd.notes));
    setTimeout(paintMediaStatus, 0);
    return card;
  }

  // ---- work area: ONE storyboard page (16_app/MEDIA_BOX.md "STORYBOARD v2"): text panel, frames row, shot cards with the words over the picture in the chosen font.
  // Every model-written field is editable and has a "revert" that puts Gemma's latest value back (S.story.gemma, S.media.readGemma); before the first story
  // (or on a box without `_story`) the read itself is shown (buildReadout).

  /** re-render the work area; while the user types in it the redraw waits for the field to lose focus (so the caret never jumps). */
  function renderWork() {
    if (S.view !== "read" || activeRun()) return;
    const a = document.activeElement;
    if (a && a.closest && a.closest(".work") && /^(INPUT|TEXTAREA)$/.test(a.tagName)) { S.workDefer = true; return; }
    S.workDefer = false;
    rerender();
  }
  document.addEventListener("focusout", () => { if (S.workDefer) setTimeout(renderWork, 60); });
  /** redraw the page keeping everything where it was: the clicks on words / frames / fonts rebuild the whole page, so every scroller (canvas, storyboard, the word list, the sideways card row)
   *  is put back, and the control that was clicked is pinned to the same screen y (the head's textareas autosize a frame later and would otherwise push the panel). */
  let rrAnchor = null, revealWant = null, rrGen = 0;
  /** selectors (most exact first) that find the clicked word chip / next-best / clear / Auto-pick again after a rebuild. */
  function anchorSelsOf(t) {
    const b = t && t.closest ? t.closest(".sb-text .w, .sb-text .sbw-btn, .sb-text .sbw-shot") : null;
    if (!b) return null;
    const row = b.closest(".sbw-shot"), rs = row ? (row.dataset.n ? `.sbw-shot[data-n="${row.dataset.n}"]` : ".sbw-shot.other") : "";
    if (b.classList.contains("sbw-auto")) return [".sb-text .sbw-auto"];
    if (b.classList.contains("w")) return rs ? [`${rs} .w[data-i="${b.dataset.i}"]`, rs] : [`.sb-text .w[data-i="${b.dataset.i}"]`];
    if (b.classList.contains("sbw-btn") && row) return [`${rs} .sbw-act .sbw-btn:nth-child(${[...b.parentNode.children].indexOf(b) + 1})`, rs];
    return rs ? [rs] : null;
  }
  const noteAnchor = (ev) => { const sels = anchorSelsOf(ev.target); if (sels) rrAnchor = { sels, t: Date.now() }; };
  document.addEventListener("pointerdown", noteAnchor, true);
  document.addEventListener("click", noteAnchor, true);
  function reanchor(a) {
    const sb = document.querySelector(".storyboard"), sw = document.querySelector(".storyboard .sb-words");
    for (let k = 0; k < a.sels.length; k++) {
      const e = a.tops[k] == null ? null : document.querySelector(a.sels[k]);
      if (!e) continue;
      let d = e.getBoundingClientRect().top - a.tops[k];
      if (Math.abs(d) < 0.5) return;
      if (sb) sb.scrollTop += d;                                       // layout above the panel moved: the storyboard takes it
      d = e.getBoundingClientRect().top - a.tops[k];
      if (Math.abs(d) >= 0.5 && sw) sw.scrollTop += d;                 // the rest (e.g. the list was clamped) goes to the word list
      return;
    }
  }
  function rerender() {
    const c = $("#canvas"), y = c ? c.scrollTop : 0, sb = document.querySelector(".storyboard"), sy = sb ? sb.scrollTop : 0, wy0 = window.scrollY;
    const sw = document.querySelector(".storyboard .sb-words"), wy = sw ? sw.scrollTop : 0, gr = document.querySelector(".storyboard .sb-grid"), gx = gr ? gr.scrollLeft : 0;
    let anc = null, focusSel = null;
    if (rrAnchor && Date.now() - rrAnchor.t < 1500) {
      const els = rrAnchor.sels.map((s) => document.querySelector(s));
      if (els.some(Boolean)) anc = { sels: rrAnchor.sels, tops: els.map((e) => e ? e.getBoundingClientRect().top : null) };
      const ai = els.indexOf(document.activeElement); if (ai >= 0) focusSel = rrAnchor.sels[ai];
    }
    rrAnchor = null;
    const want = revealWant && Date.now() - revealWant.t < 1500 ? revealWant : null; revealWant = null;
    const gen = ++rrGen;
    renderCanvas();
    if (c) c.scrollTop = y;
    if (wy0 && window.scrollY !== wy0) window.scrollTo(0, wy0);
    const nb = document.querySelector(".storyboard"); if (nb) nb.scrollTop = sy;
    const nw = document.querySelector(".storyboard .sb-words"); if (nw) nw.scrollTop = wy;
    const ng = document.querySelector(".storyboard .sb-grid"); if (ng) ng.scrollLeft = gx;
    if (focusSel) { const f = document.querySelector(focusSel); if (f) try { f.focus({ preventScroll: true }); } catch (_) { /* no focus */ } }
    const settle = () => { if (gen === rrGen && anc) reanchor(anc); };
    settle();
    requestAnimationFrame(() => { settle(); requestAnimationFrame(() => { settle(); if (gen === rrGen && want) revealShot(want.n, want.still); }); });
  }
  /** a shot's lyric row was clicked: its card (below) scrolls into view and flashes a yellow outline. Only the card row / storyboard scroll; the storyboard only when the clicked row stays visible too, and never (still) when a word / button was clicked, so that control stays exactly where it is. */
  function revealShot(n, still) {
    const sb = document.querySelector(".storyboard"), card = sb && sb.querySelector(`.sb-card[data-n="${n}"]`);
    if (!card) return;
    const grid = card.parentElement, row = sb.querySelector(`.sbw-shot[data-n="${n}"]`);
    const behavior = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    if (grid && grid.scrollWidth > grid.clientWidth + 1) {
      const g = grid.getBoundingClientRect(), cr = card.getBoundingClientRect();
      if (cr.left < g.left || cr.right > g.right) grid.scrollTo({ left: grid.scrollLeft + (cr.left + cr.width / 2) - (g.left + g.width / 2), behavior });
    }
    const s = sb.getBoundingClientRect(), cr = card.getBoundingClientRect();
    let need = still ? 0 : cr.bottom > s.bottom ? cr.bottom - s.bottom + 8 : cr.top < s.top ? cr.top - s.top - 8 : 0;
    if (need) {
      const max = sb.scrollHeight - sb.clientHeight;
      need = Math.max(0, Math.min(max, sb.scrollTop + need)) - sb.scrollTop;
      const r = row && row.getBoundingClientRect();
      if (Math.abs(need) >= 1 && (!r || (r.top - need >= s.top && r.bottom - need <= s.bottom))) sb.scrollTo({ top: sb.scrollTop + need, behavior });   // never at the cost of the clicked row
    }
    card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1700);
  }

  // ---- fonts: the liked ones only (GET /mograph/fonts/liked -> [{file, name, family_css, role, risky}]; the file at /mograph/fonts/file/<file>), loaded on demand as CSS @font-face
  function loadFonts() {
    const f = S.fonts;
    if (f.state !== "idle") return f.promise;
    f.state = "loading";
    f.promise = (async () => {
      try {
        const list = await getJSON("/mograph/fonts/liked");
        f.list = (Array.isArray(list) ? list : []).filter((x) => x && x.file).map((x) => ({ file: String(x.file), name: String(x.name || x.family || x.file), family_css: x.family_css ? String(x.family_css) : "", role: String(x.role || ""), risky: !!x.risky }));
        f.state = f.list.length ? "ready" : "none";
      } catch (_) { f.list = []; f.state = "none"; }
      const st = S.story;
      if (st.data && f.list.length) {                          // fonts arrived after the story: the same random default goes into the story and Gemma's copy
        const rnd = B.fontPickAll(f.list, st.data.shots.length);
        st.data.shots.forEach((x, i) => {
          if (f.list.some((y) => y.file === x.font)) return;
          x.font = rnd[i];
          const g = st.gemma && st.gemma.shots.find((y) => y.n === x.n);
          if (g && !f.list.some((y) => y.file === g.font)) g.font = rnd[i];
        });
      }
      renderWork();
    })();
    return f.promise;
  }
  const fontByFile = (file) => S.fonts.list.find((f) => f.file === file) || null;
  let fontStyle = null;
  function ensureFontFace(f) {
    if (!f || S.fonts.loaded[f.file]) return;
    S.fonts.loaded[f.file] = "pending";
    if (!fontStyle) { fontStyle = document.createElement("style"); fontStyle.id = "mf-faces"; document.head.append(fontStyle); }
    const fam = B.fontCss(f);
    fontStyle.append(document.createTextNode(`@font-face{font-family:"${fam}";src:url("/mograph/fonts/file/${encodeURIComponent(f.file)}");font-weight:100 900;font-display:swap}\n`));
    if (document.fonts && document.fonts.load) document.fonts.load(`40px "${fam}"`).then(() => { S.fonts.loaded[f.file] = true; fitAllOverlays(); }, () => { S.fonts.loaded[f.file] = "failed"; });
  }

  // ---- ink: light or dark words by the picture's luminance under the text band (B.legibleInk), a subtle scrim only if needed
  const inkCache = new Map();
  function sampleLum(url) {
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => {
        const W = 36, H = 64, cv = document.createElement("canvas");
        cv.width = W; cv.height = H;
        const cx = cv.getContext("2d", { willReadFrequently: true }), iw = im.naturalWidth, ih = im.naturalHeight, tr = W / H;
        let sw = iw, sh = ih;
        if (iw / ih > tr) sw = ih * tr; else sh = iw / tr;              // object-fit: cover into 9:16
        cx.drawImage(im, (iw - sw) / 2, (ih - sh) / 2, sw, sh, 0, 0, W, H);
        const d = cx.getImageData(0, Math.round(H * 0.3), W, Math.round(H * 0.4)).data, lin = (c) => Math.pow(c / 255, 2.2);
        let n = 0, sum = 0, sum2 = 0, cl = 0, cd = 0;
        for (let i = 0; i < d.length; i += 4) {
          const L = 0.2126 * lin(d[i]) + 0.7152 * lin(d[i + 1]) + 0.0722 * lin(d[i + 2]);
          sum += L; sum2 += L * L; n++;
          if (1.05 / (L + 0.05) >= 3) cl++;                       // white words readable on this pixel
          if ((L + 0.05) / 0.0533 >= 3) cd++;                     // dark words readable on this pixel
        }
        const mean = sum / n;
        res({ lum: mean, sd: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), cover: { light: cl / n, dark: cd / n } });
      };
      im.onerror = rej; im.src = url;
    });
  }
  function applyInk(ov, url) {
    const set = (r) => { const k = B.legibleInk(r.lum, r.sd, r.cover); ov.style.color = k.color; ov.classList.toggle("scrim", k.scrim); ov.classList.toggle("dark", k.ink === "dark"); ov.dataset.contrast = k.contrast.toFixed(1); };
    if (!url) return set({ lum: 0.03, sd: 0 });
    const c = inkCache.get(url);
    if (c && c.lum !== undefined) return set(c);
    if (!c) { inkCache.set(url, { pending: true }); const done = (r) => { inkCache.set(url, r); paintAllCards(); }; sampleLum(url).then(done, () => done({ lum: 0.3, sd: 0 })); }
    set({ lum: 0.3, sd: 0 });
  }

  /** the picture of a story shot: the image item with the shot's id (I1.. in the user's current order), or null. */
  function storyImage(shot) {
    if (!shot || !shot.image) return null;
    const i = B.mediaIds(S.media.items).indexOf(shot.image), it = S.media.items[i];
    return it && it.kind === "image" ? it : null;
  }
  /** VIDEO MODE: the video item a story shot is cut from (a shot with a source span), or null. */
  const shotVideo = (shot) => (B.isVideoShot(shot) ? videoItem() : null);
  /** the first frame of a cut = the video frame at src_start + 0.04 s (the cut strip and the card show the same one). */
  const cutFrameT = (shot) => Number(shot.src_start) + B.VIDEO_FRAME_OFF_S;
  /** the url of the picture under a shot's words: its image, or (video mode) the cut's first frame once grabbed, else the video's poster. */
  const shotPicUrl = (shot) => { const it = storyImage(shot), v = shotVideo(shot); return it ? it.url : v ? (frameCached(v, cutFrameT(shot)) || v.poster || null) : null; };
  const storyImgNode = (shot) => {
    const it = storyImage(shot), v = shotVideo(shot);
    if (v) { const u = shotPicUrl(shot); return [el("img", { src: u || null, alt: "", draggable: "false", "data-vframe": frameCached(v, cutFrameT(shot)) ? "1" : null })]; }
    return it ? [el("img", { src: it.url, alt: "", draggable: "false" }), el("span", { class: "m-id" }, shot.image)]
              : [el("div", { class: "sb-noimg", html: svg("film", 24) })];
  };
  /** load a video cut's first frame into its card (and the strip) and re-ink the words over it. */
  function loadCutFrame(card, shot, thumb) {
    const v = shotVideo(shot);
    if (!v) return;
    videoFrameAt(v, cutFrameT(shot)).then((url) => {
      if (!url) return;
      const im = card && card.isConnected ? card.querySelector(".sb-img img") : null;
      if (im) { im.src = url; im.dataset.vframe = "1"; }
      if (thumb && thumb.isConnected) { thumb.src = url; thumb.dataset.vframe = "1"; }
      if (card && card.isConnected) { const x = S.story.data && S.story.data.shots[Number(card.dataset.i)]; if (x) paintCard(card, x); }
    });
  }
  /** the box order changed: the cards / frames follow their image id at once (the text follows when the story is written again). */
  function paintStoryImages() { if (S.story.data && S.view === "read") renderWork(); }

  /** v14: the words over a frame never break inside a word. The font size starts at the length-based size and shrinks until the longest word fits the overlay's inner width
   *  (measured per word with a Range, text on one line) and the wrapped block fits 84% of the frame height; wraps only happen between words (CSS word-break: normal). */
  function fitOverlay(ov) {
    const fr = ov.parentElement;
    if (!fr || ov.hidden || !ov.firstChild || ov.firstChild.nodeType !== 3) return;
    const fw = fr.clientWidth, fh = fr.clientHeight;
    if (!fw || !fh) return;
    const cs = getComputedStyle(ov), px = (v) => parseFloat(v) || 0;
    const maxW = ov.clientWidth - px(cs.paddingLeft) - px(cs.paddingRight), padV = px(cs.paddingTop) + px(cs.paddingBottom), maxH = fh * 0.84 - padV;
    if (maxW <= 0) return;
    const t = ov.firstChild, rs = [], re = /\S+/g;
    for (let m; (m = re.exec(t.data));) { const r = document.createRange(); r.setStart(t, m.index); r.setEnd(t, m.index + m[0].length); rs.push(r); }
    if (!rs.length) return;
    let size = fw * (ov.dataset.size === "l" ? 0.21 : ov.dataset.size === "m" ? 0.15 : 0.105);
    for (let k = 0; k < 16; k++) {
      ov.style.fontSize = size + "px";
      ov.style.whiteSpace = "nowrap";
      const ww = Math.max(...rs.map((r) => r.getBoundingClientRect().width));
      ov.style.whiteSpace = "";
      const hh = ov.scrollHeight - padV;
      const kw = ww > maxW ? maxW / ww : 1, kh = hh > maxH ? maxH / hh : 1;
      if (kw === 1 && kh === 1) break;
      size = Math.max(5, size * Math.min(kw, kh) * 0.99);
    }
  }
  const fitAllOverlays = () => document.querySelectorAll(".sb-ov").forEach(fitOverlay);
  const frameRO = typeof ResizeObserver !== "undefined" ? new ResizeObserver((es) => es.forEach((e) => { const o = e.target.querySelector(".sb-ov"); if (o) fitOverlay(o); })) : null;
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", fitAllOverlays);

  /** v14: card column width = min(work width / shots, the width at which the whole card (frame at the chosen format + rows) fits the visible work height), no fixed cap. */
  let boardRO = null, boardLast = "";
  function fitBoard() {
    const sb = document.querySelector(".storyboard"), grid = sb && sb.querySelector(".sb-grid");
    if (!grid || !grid.firstElementChild) return;
    const n = grid.children.length, GAP = 12, PAD = 18, MIN = 150, COMFY = 320;
    const fm = /^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/.exec(String(S.values.format || "9:16")), ratio = fm ? Number(fm[2]) / Number(fm[1]) : 16 / 9;
    grid.style.setProperty("--fa", fm ? `${fm[1]} / ${fm[2]}` : "9 / 16");
    const availW = grid.clientWidth || sb.clientWidth, wW = Math.max(MIN, Math.floor((availW - GAP * (n - 1)) / n));
    const top = grid.getBoundingClientRect().top - sb.getBoundingClientRect().top + sb.scrollTop, availH = sb.clientHeight - top - 17;
    const chromeOf = () => { const c = grid.firstElementChild, im = c.querySelector(".sb-img"); return c.offsetHeight - im.offsetHeight; };
    const isV = grid.classList.contains("vid");                      // v16 video mode: one row that scrolls sideways, so the width is set by the work height only (never below COMFY)
    const floor = isV ? COMFY : Math.min(wW, COMFY);                 // a short window never shrinks the cards below a readable width: the board scrolls instead
    let w = isV ? COMFY : wW;
    for (let k = 0; k < 4; k++) {
      grid.style.setProperty("--colw", w + "px");
      const hw = Math.floor((availH - chromeOf()) / ratio) + PAD, nw = isV ? Math.max(floor, hw) : Math.max(floor, Math.min(wW, hw));
      if (Math.abs(nw - w) < 1) break;
      w = nw;
    }
    grid.style.setProperty("--colw", w + "px");
    for (let k = 0; k < 4 && w > floor && sb.scrollHeight > sb.clientHeight + 1; k++) {        // text wrapped a little more at this width: shave the overflow
      w = Math.max(floor, w - Math.ceil((sb.scrollHeight - sb.clientHeight) / ratio) - 1);
      grid.style.setProperty("--colw", w + "px");
    }
    fitAllOverlays();
  }
  const fitBoardSoon = () => requestAnimationFrame(fitBoard);

  /** the words over the picture: the shot's line in its font, ink by the picture under it. */
  function paintCard(card, x) {
    const ov = card.querySelector(".sb-ov"), f = fontByFile(x.font), L = x.line.trim();
    ov.textContent = L; ov.hidden = !L;
    ov.style.fontFamily = f ? `"${B.fontCss(f)}", system-ui, sans-serif` : "";
    ov.dataset.size = L.length <= 7 ? "l" : L.length <= 13 ? "m" : "s";
    if (f) ensureFontFace(f);
    fitOverlay(ov);
    const nm = card.querySelector(".sb-fname");
    if (nm) { nm.textContent = f ? f.name : x.font; nm.title = f ? `${f.name}${f.role ? " (" + f.role + ")" : ""}: liked font` : ""; }
    applyInk(ov, shotPicUrl(x));
  }
  function paintAllCards() {
    const d = S.story.data;
    if (!d) return;
    document.querySelectorAll(".sb-card").forEach((c) => { const x = d.shots[Number(c.dataset.i)]; if (x) paintCard(c, x); });
  }

  // ---- versions of the model-written text fields (user 2026-10-30: no revert links, ‹ › arrows step a field through Gemma's version(s) and the user's edit)
  // S.story.vers[key] = {g: [Gemma's values], u: the user's latest edit | null, at}; keys: title | logline | look | music | <n>.line|beat|camera | note:<file>:shows|mood. The shown value is what Make it sends.
  const noteOfKey = (key) => {
    const m = /^note:(.*):(shows|mood)$/.exec(key), e = m && S.media.read && S.media.read.media.find((x) => x.file === m[1]);
    return e ? { e, f: m[2] } : null;
  };
  const noteGemma = (e) => (S.media.readGemma ? S.media.readGemma.media.find((x) => x.file === e.file) : null);
  /** a read note differs from Gemma's (shows, mood or window)? Drives the "keep my edit" flag the next read honours. */
  function noteDiffers(e) {
    const g = noteGemma(e);
    return !g || e.shows !== g.shows || e.mood !== g.mood || JSON.stringify(e.window) !== JSON.stringify(g.window);
  }
  const verGet = (key) => { const n = noteOfKey(key); return n ? n.e[n.f] : (S.story.data ? B.fieldGet(S.story.data, key) : undefined); };
  const verGemma = (key) => { const n = noteOfKey(key); if (n) { const g = noteGemma(n.e); return g ? g[n.f] : undefined; } return S.story.gemma ? B.fieldGet(S.story.gemma, key) : undefined; };
  function verSet(key, val) {
    const n = noteOfKey(key);
    if (n) { n.e[n.f] = val; ed().media[n.e.file] = noteDiffers(n.e); }
    else if (S.story.data) B.fieldSet(S.story.data, key, val);
  }
  /** bring one field's versions in line with its shown value and Gemma's latest (typing = the user is typing in it now). */
  function verSync(key, typing) {
    const cur = verGet(key);
    if (cur === undefined) return null;
    return (S.story.vers[key] = B.verSync(S.story.vers[key], cur, verGemma(key), typing));
  }
  function syncAllVers(typing) {
    const d = S.story.data;
    if (!d) return;
    for (const k of ["title", "logline", "look", "music", ...d.shots.flatMap((x) => ["line", "beat", "camera"].map((f) => `${x.n}.${f}`))]) verSync(k, typing);
  }
  /** the user changed a text field (typing, camera select): its value is the user's version (or back at one of Gemma's). */
  function typed(key, val) {
    const st = S.story;
    verSet(key, val);
    const v = verSync(key, true);
    if (v && !noteOfKey(key)) st.edited[key] = v.at === "u";
    paintVers();
  }
  /** v14: title / logline / look / music ALWAYS show ‹ n/m ›; the other fields (beat, read notes) show them once they have a second version. At the newest version › writes the story again. */
  const ALWAYS_VER = new Set(["look", "music"]);              // v19: title / logline are plain text now (no ‹ n/m ›)
  function paintVer(box) {
    const key = box.dataset.ver, always = ALWAYS_VER.has(key), v = verSync(key);
    if (!v && !always) { box.hidden = true; return; }
    const list = v ? B.verList(v) : [], n = Math.max(1, list.length), i = v ? B.verIndex(v) : 0, st = S.story;
    box.hidden = !always && list.length < 2;
    box.querySelector(".sb-vn").textContent = `${i + 1}/${n}`;
    const prev = box.querySelector("[data-dir='-1']"), next = box.querySelector("[data-dir='1']"), newest = i >= n - 1, writing = always && newest && st.busy;
    prev.disabled = i <= 0;
    next.disabled = newest && (!always || st.busy || !S.media.read);
    next.classList.toggle("busy", !!writing);
    next.title = writing ? "Gemma is writing another version..." : newest && always ? "Newest version: write the story again for another" : "Next version";
    box.title = !v ? "Gemma's version" : i >= v.g.length ? `Your edit (version ${i + 1} of ${n})` : `Gemma's version (${i + 1} of ${n})`;
  }
  const paintVers = () => document.querySelectorAll(".work [data-ver]").forEach(paintVer);
  function stepVer(key, dir, input) {
    const st = S.story, v = verSync(key), r = v && B.verStep(v, dir);
    if (!r) { if (dir > 0 && ALWAYS_VER.has(key)) { rewriteStory(); paintVers(); } return; }   // › at the newest version: Gemma writes another
    st.vers[key] = r.v;
    verSet(key, r.value);
    if (!noteOfKey(key)) st.edited[key] = r.user;
    if (input) input.value = r.value;
    const m = /^(\d+)\.line$/.exec(key);
    if (m) { const card = document.querySelector(`.sb-card[data-n="${m[1]}"]`), x = st.data.shots.find((s) => s.n === Number(m[1])); if (card && x) paintCard(card, x); }
    if (noteOfKey(key)) scheduleStory();
    paintVers();
    if (input) { autosize(input); fitBoardSoon(); }                // the text got shorter / longer: the cards refit the work height
  }
  const verCtl = (key, input) => {
    const btn = (dir, ch, title) => el("button", { class: "sb-fbtn sb-vb", type: "button", "data-dir": dir, title, "aria-label": title, onclick: () => stepVer(key, dir, input) }, ch);
    const box = el("span", { class: "sb-ver", "data-ver": key, hidden: true }, btn(-1, "‹", "Previous version"), el("span", { class: "sb-vn" }), btn(1, "›", "Next version"));
    paintVer(box);
    return box;
  };
  const fld = (input, key) => el("div", { class: "sb-fld" }, input, verCtl(key, input));

  // ---- revert: only what is not a text field (the words chosen in the text panel, a card's font) keeps one
  function isEd(key) { return B.fieldEdited(S.story.edited, S.story.data, S.story.gemma, key); }
  /** show / hide the revert controls that are left (words, fonts) without redrawing the page. */
  function paintRevs() { document.querySelectorAll(".work [data-rev]").forEach((b) => { b.hidden = !isEd(b.dataset.rev); }); }

  function revertKey(key) {
    const st = S.story;
    if (!st.data || !st.gemma) return;
    st.data = B.revertField(st.data, st.gemma, key);
    delete st.edited[key];
    if (key === "sel") for (const s of st.data.shots) delete st.edited[`${s.n}.line`];
    syncAllVers(true);
    rerender();
  }

  /** a change of frames / words that needs no model call: re-derive the story locally (the next model call gets it as req.use / req.select).
   *  A picture change gives the shot the read's note for the new picture as its beat, unless the user edited that beat. */
  function applyOverride(o) {
    const st = S.story, ctx = storyCtx(), beats = new Map(st.data.shots.map((s) => [s.n, s.beat]));
    const r = B.deriveStory(st.data, Object.assign({ tokens: ctx.tokens, times: ctx.times, beats: ctx.beats, imageIds: ctx.imageIds, shows: ctx.shows }, o));
    st.data = r.story;
    if (o.select) { st.edited.sel = true; st.selText = S.media.text; for (const n of r.lineChanged) delete st.edited[`${n}.line`]; }
    if (o.use) {
      st.edited.use = true;
      for (const n of r.imageChanged) if (st.edited[`${n}.beat`]) { const sh = st.data.shots.find((s) => s.n === n); if (sh) sh.beat = beats.get(n); }
    }
    syncAllVers(true);
    rerender();
  }

  // ---- the editor for one file's read notes (what it shows, mood, window); used in a card's info toggle and in the "all notes" list
  function noteEditor(it, e, id) {
    const keyOf = (f) => `note:${e.file}:${f}`, wrap = el("div", { class: "sb-note" });
    const shows = el("input", { type: "text", value: e.shows, placeholder: "What it shows", "aria-label": `${id} shows` });
    shows.addEventListener("input", () => { typed(keyOf("shows"), shows.value); scheduleStory(); });
    wrap.append(el("div", { class: "sb-nrow" }, el("span", { class: "sb-k" }, id), fld(shows, keyOf("shows"))));
    const mood = el("input", { type: "text", value: e.mood, placeholder: "Mood", "aria-label": `${id} mood` });
    mood.addEventListener("input", () => { typed(keyOf("mood"), mood.value); scheduleStory(); });
    wrap.append(el("div", { class: "sb-nrow" }, el("span", { class: "sb-k" }, "Mood"), fld(mood, keyOf("mood"))));
    if (Array.isArray(e.window)) {
      const num = (k) => {
        const n = el("input", { type: "number", step: "0.1", min: "0", value: String(e.window[k]), "aria-label": k ? "Window end" : "Window start" });
        n.addEventListener("input", () => { const v = Number(n.value); if (Number.isFinite(v)) { e.window[k] = v; ed().media[e.file] = true; scheduleStory(); } });
        return n;
      };
      wrap.append(el("div", { class: "sb-nrow sb-win" }, el("span", { class: "sb-k" }, "Window"), num(0), " to ", num(1), " s"));
    }
    return wrap;
  }

  // ---- TEXT panel (v18 SHOT-FOCUSED WORDS): the pasted text / lyrics (display cap 400 words). Every chosen word belongs to a shot. With a shot ACTIVE (click its frame) its words are strong and a click puts
  //      a word on / takes it off THAT shot (a word of another shot moves here); the other shots' words are faint with their shot number. Without an active shot a click gives the word to the nearest empty shot.
  const LONG_WORDS = 3;                                                        // a shot with more words than this is shown small on screen: a small "long" hint
  function buildTextPanel(d) {
    const src = textSource(), tokens = src.tokens, sel = d.text_select, act = S.story.active;
    const shotOf = new Map(sel.map((t) => [t.i, t.shot]));
    function shotTokens(n) { return B.shotTokens(sel, n); }
    const withWords = d.shots.filter((x) => shotTokens(x.n).length).length;
    const actShot = act != null ? d.shots.find((x) => x.n === act) : null;
    const byShot = !!(src.lyrics && src.times && src.times.length);                // WORD PICK: lyrics with sung times -> one row per shot with the words sung during it
    const head = el("div", { class: "sb-th" }, el("b", {}, "Your text"), src.lyrics ? el("span", { class: "sb-lyr", title: "No text was pasted: these are the words heard in your media, with the time each is sung" }, `Lyrics (heard in the ${src.from})`) : null,
      el("span", { class: "sb-tsum" }, byShot ? `${withWords} of ${d.shots.length} shot${d.shots.length === 1 ? "" : "s"} ${withWords === 1 ? "has" : "have"} words. Click a word to put it on its shot or take it off; shift-click or drag across words for a phrase; ↻ tries the next-best pick`
        : actShot ? `Shot ${actShot.n}: click words to put them on this shot or take them off (Esc or its frame again to finish)`
        : withWords ? `${withWords} of ${d.shots.length} shot${d.shots.length === 1 ? "" : "s"} ${withWords === 1 ? "has" : "have"} words; click a frame to change its words` : "no words chosen; click a frame to pick its words"),
      el("button", { class: "sbw-btn sbw-auto", type: "button", title: "Back to the automatic word choice for every shot (drops your edits to the words)", onclick: autoPickWords }, "Auto-pick words"));
    const words = byShot ? buildShotWords(d, src, sel, act) : el("div", { class: "sb-words" });
    const lastOf = new Map(d.shots.map((x) => [x.n, Math.max(-1, ...shotTokens(x.n))]));
    (byShot ? [] : tokens.slice(0, B.TEXT_CAP)).forEach((t, i) => {
      const sh = shotOf.get(i) || 0, has = sh >= 1, mine = has && sh === act, faint = has && act != null && sh !== act, on = has && (act == null || mine);
      const firstOfRun = has && shotOf.get(i - 1) !== sh;
      const at = src.times ? Number(src.times[i]) : NaN, sung = Number.isFinite(at) ? ` Sung at ${at.toFixed(1)} s.` : "";
      const title = (act != null ? (mine ? `On shot ${act}. Click to take it off this shot.` : has ? `On shot ${sh}. Click to move it to shot ${act}.` : `Click to put it on shot ${act}.`)
        : has ? `On shot ${sh}. Click to take it off.` : "Click to give it to the nearest shot without words.") + sung;
      words.append(el("button", { class: "w" + (on ? " on" : "") + (faint ? " other" : "") + (mine ? " mine" : ""), type: "button", "data-i": i, "data-shot": has ? sh : null, "data-t": Number.isFinite(at) ? at.toFixed(2) : null, title,
        onclick: () => clickWord(i) },
        firstOfRun ? el("sup", { class: "w-n" }, String(sh)) : null, t,
        has && lastOf.get(sh) === i && shotTokens(sh).length > LONG_WORDS ? el("sup", { class: "w-long", title: `Shot ${sh} has more than ${LONG_WORDS} words: they will be small on screen` }, "long") : null));
    });
    const panel = el("div", { class: "sb-text" + (act != null ? " focus" : "") }, head, words);
    const hidden = Math.max(0, tokens.length - B.TEXT_CAP), hiddenSel = sel.filter((t) => t.i >= B.TEXT_CAP).length;
    if (hidden) panel.append(el("div", { class: "sb-more" }, `... ${hidden} more word${hidden === 1 ? "" : "s"}` + (hiddenSel ? ` (${hiddenSel} of them chosen)` : "")));
    return panel;
  }
  /** a click on a word of the text panel: the active shot gets / loses it; with no active shot it goes to the nearest EMPTY shot (by sung time, hit land or position), and when every shot has words it is refused with a notice. */
  function clickWord(i) {
    const st = S.story, d = st.data;
    if (!d) return;
    const sel = d.text_select;
    if (st.active != null) return applyOverride({ select: B.toggleWordShot(sel, i, st.active) });
    const cur = sel.find((t) => t.i === i);
    if (cur) return applyOverride({ select: B.toggleWordShot(sel, i, cur.shot) });              // already on a shot: a click takes it off
    const ctx = storyCtx(), k = B.nearestEmptyShot(d, i, { tokens: ctx.tokens, times: ctx.times });
    if (k < 0) { toast("every shot has words: click a frame first to change its words"); return; }
    applyOverride({ select: B.toggleWordShot(sel, i, k) });
  }
  // ---- WORD PICK editor (lyrics with sung times): one row per shot = the words sung DURING that shot as chips; chosen words yellow; click toggles a word on / off THAT shot (a word of another shot moves here),
  //      shift-click or a drag across chips makes a phrase, ↻ steps through the backend's ranked picks (story.word_alts), "clear" empties the shot. Every edit goes through applyOverride (pins story.edited.sel).
  let wordDrag = null, wordAnchor = null, wordDragDone = false;
  document.addEventListener("pointerup", () => {
    const g = wordDrag; wordDrag = null;
    if (!g || !g.moved) return;
    wordDragDone = true; setTimeout(() => { wordDragDone = false; }, 0);
    const d = S.story.data;
    if (d) { wordAnchor = { i: g.to, shot: g.shot }; S.story.wordRow = g.shot; revealWant = { n: g.shot, t: Date.now(), still: true }; applyOverride({ select: B.rangeWordShot(d.text_select, g.from, g.to, g.shot) }); }
  });
  function buildShotWords(d, src, sel, act) {
    const tokens = src.tokens, times = src.times, cap = B.TEXT_CAP, shotOf = new Map(sel.map((t) => [t.i, t.shot]));
    const { by, outside } = B.sungByShot(d.shots, times), gem = S.story.gemma;
    const wrap = el("div", { class: "sb-words by-shot" });
    const chip = (i, n) => {
      const sh = shotOf.get(i) || 0, mine = sh === n, elsewhere = sh >= 1 && !mine, at = Number(times[i]);
      const sung = Number.isFinite(at) ? ` Sung at ${at.toFixed(1)} s.` : "";
      const b = el("button", { class: "w" + (mine ? " on" : "") + (elsewhere ? " other" : ""), type: "button", "data-i": i, "data-shot": sh >= 1 ? sh : null, "data-ts": Number.isFinite(at) ? `${at.toFixed(1)}s` : null,
        title: (n == null ? "Click to give it to the nearest shot without words." : mine ? `On shot ${n}. Click to take it off.` : elsewhere ? `On shot ${sh}. Click to move it to shot ${n}.` : `Click to put it on shot ${n}.`) + sung },
        elsewhere ? el("sup", { class: "w-n" }, String(sh)) : null, tokens[i]);
      if (n == null) { b.addEventListener("click", () => clickWord(i)); return b; }
      b.addEventListener("pointerdown", (ev) => { if (ev.button === 0 && !ev.shiftKey) wordDrag = { shot: n, from: i, to: i, moved: false }; });
      b.addEventListener("pointerenter", () => {
        if (!wordDrag || wordDrag.shot !== n || wordDrag.to === i) return;
        wordDrag.to = i; wordDrag.moved = true;
        const lo = Math.min(wordDrag.from, i), hi = Math.max(wordDrag.from, i);
        wrap.querySelectorAll(`.sbw-shot[data-n="${n}"] .w`).forEach((c) => c.classList.toggle("drag", Number(c.dataset.i) >= lo && Number(c.dataset.i) <= hi));
      });
      b.addEventListener("click", (ev) => {
        if (wordDragDone) return;
        const dd = S.story.data;
        if (!dd) return;
        if (ev.shiftKey && wordAnchor && wordAnchor.shot === n && wordAnchor.i !== i) { const a = wordAnchor.i; wordAnchor = { i, shot: n }; applyOverride({ select: B.rangeWordShot(dd.text_select, a, i, n) }); return; }
        wordAnchor = { i, shot: n };
        applyOverride({ select: B.toggleWordShot(dd.text_select, i, n) });
      });
      return b;
    };
    for (const sh of d.shots) {
      const [a, z] = B.shotSpan(sh), mineIdx = B.shotTokens(sel, sh.n), sung = (by.get(sh.n) || []).filter((i) => i < cap);
      const idx = [...new Set([...sung, ...mineIdx.filter((i) => i < cap)])].sort((p, q) => p - q);
      const alts = (d.word_alts && d.word_alts[String(sh.n)]) || [];
      const curKey = mineIdx.join(","), at = alts.findIndex((x) => x.t.join(",") === curKey);
      const edited = !!gem && B.shotTokens(gem.text_select || [], sh.n).join(",") !== curKey;
      const cyc = el("button", { class: "sbw-btn", type: "button", disabled: alts.length < 1 || (alts.length === 1 && at === 0), "aria-label": `Shot ${sh.n}: next-best pick`,
        title: alts.length ? `Next-best auto pick for shot ${sh.n} (${at >= 0 ? at + 1 : 0} of ${alts.length} tried)` : "No other pick for this shot",
        onclick: () => { const ns = B.cycleShotWords(S.story.data, sh.n); if (ns) applyOverride({ select: ns }); else toast("no other pick for this shot"); } }, "↻");
      const clr = el("button", { class: "sbw-btn", type: "button", disabled: !mineIdx.length, "aria-label": `Shot ${sh.n}: clear words`, title: "Take every word off this shot",
        onclick: () => applyOverride({ select: B.clearShotWords(S.story.data.text_select, sh.n) }) }, "clear");
      const chips = el("div", { class: "sbw-chips" });
      if (idx.length) idx.forEach((i) => { const c = chip(i, sh.n); if (!sung.includes(i)) { c.classList.add("out"); c.title += " (outside this shot)"; } chips.append(c); });
      else chips.append(el("span", { class: "sbw-none" }, "nothing sung in this shot"));
      const row = el("div", { class: "sbw-shot" + (act === sh.n ? " act" : "") + (S.story.wordRow === sh.n ? " cur" : ""), "data-n": sh.n },
        el("div", { class: "sbw-h" }, el("b", {}, `Shot ${sh.n}`), el("span", {}, `${a.toFixed(1)}-${z.toFixed(1)} s`), edited ? el("i", { title: "You changed this shot's words" }, "edited") : null),
        chips, el("div", { class: "sbw-act" }, cyc, clr));
      row.addEventListener("click", (ev) => {                                    // capture: before a chip rebuilds the page. The row is now the current one; its card is revealed once the page has settled
        S.story.wordRow = sh.n; revealWant = { n: sh.n, t: Date.now(), still: !!(ev.target.closest && ev.target.closest(".w, .sbw-btn")) };
        wrap.querySelectorAll(".sbw-shot").forEach((r) => r.classList.toggle("cur", r === row));
      }, true);
      row.addEventListener("click", () => { const rw = revealWant; if (rw && rw.n === sh.n) { revealWant = null; revealShot(sh.n, rw.still); } });   // bubble: no rebuild happened (header / gap click)
      wrap.append(row);
    }
    const rest = outside.filter((i) => i < cap);
    if (rest.length) wrap.append(el("div", { class: "sbw-shot other" }, el("div", { class: "sbw-h" }, el("b", {}, "Not in a shot"), el("span", {}, "sung outside the cuts")), el("div", { class: "sbw-chips" }, ...rest.map((i) => chip(i, null))), el("div", {})));
    return wrap;
  }
  /** "Auto-pick words": every shot back to the backend's own word choice and the words are no longer pinned (the next story run picks them again). */
  function autoPickWords() {
    const st = S.story;
    if (!st.data || !st.gemma) return;
    if (!st.edited.sel) { toast("the words are already the automatic pick"); return; }
    revertKey("sel");
  }
  /** the active shot (clicked frame): highlight its card, dim the others, and redraw the text panel for it. n = the shot number, null/same again = none. */
  function setActive(n) {
    const st = S.story;
    st.active = n == null || st.active === n ? null : n;
    paintFocus();
  }
  function paintFocus() {
    const st = S.story, d = st.data, sb = document.querySelector(".storyboard");
    if (!sb || !d) return;
    if (st.active != null && !d.shots.some((x) => x.n === st.active)) st.active = null;
    sb.classList.toggle("has-active", st.active != null);
    sb.querySelectorAll(".sb-card").forEach((c) => {
      const on = st.active != null && Number(c.dataset.n) === st.active;
      c.classList.toggle("active", on); c.classList.toggle("dim", st.active != null && !on);
      const f = c.querySelector(".sb-img"); if (f) f.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const old = sb.querySelector(".sb-text");
    if (old && textSource().tokens.length) { const ps = old.querySelector(".sb-words"), top = ps ? ps.scrollTop : 0; old.replaceWith(buildTextPanel(d)); const nw = sb.querySelector(".sb-words"); if (nw) nw.scrollTop = top; }
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || S.story.active == null || popOff) return;
    setActive(null);
  });

  // ---- picture picker: click a shot card's picture -> every dropped image (user order) in a small popover; a picture already used by another shot swaps with it, an unused one replaces this shot's
  let popOff = null;
  const closePop = () => { if (popOff) { popOff(); popOff = null; } };
  function pickImage(k, id) {
    const d = S.story.data;
    closePop();
    if (!d || !d.shots[k]) return;
    const use = d.shots.map((s) => s.image), other = use.indexOf(id);
    if (other === k) return;
    if (other >= 0) { use[other] = use[k]; use[k] = id; } else use[k] = id;     // swap with the shot that has it, else replace (the replaced picture becomes unused)
    applyOverride({ use });
  }
  function openPicker(k, anchor) {
    const wasOpen = popOff && anchor.classList.contains("picking");
    closePop();
    if (wasOpen) return;
    const d = S.story.data, x = d && d.shots[k];
    if (!x || !x.image) return;
    const ids = B.mediaIds(S.media.items), imgs = S.media.items.map((it, i) => ({ it, id: ids[i] })).filter((q) => q.it.kind === "image");
    const pop = el("div", { class: "sb-pop", role: "dialog", "aria-label": `Picture for shot ${x.n}` }, el("div", { class: "sb-pop-h" }, `Picture for shot ${x.n}`));
    const grid = el("div", { class: "sb-pick" });
    imgs.forEach(({ it, id }) => {
      const owner = d.shots.find((s) => s.image === id), cur = owner === x;
      grid.append(el("button", { class: "sb-pk" + (cur ? " cur" : owner ? " used" : " free"), type: "button", "data-id": id, "data-shot": owner ? owner.n : null,
        title: cur ? `${id}: this shot's picture` : owner ? `${id} is shot ${owner.n}'s picture: pick it to swap the two` : `${id} is not used: pick it to use it here`,
        onclick: () => pickImage(k, id) },
        el("img", { src: it.url, alt: "", draggable: "false" }), el("span", { class: "m-id" }, id), owner ? el("span", { class: "fr-n" }, String(owner.n)) : null));
    });
    pop.append(grid);
    document.body.append(pop);
    anchor.classList.add("picking");
    const r = anchor.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8)) + "px";
    pop.style.top = (r.bottom + h + 12 > window.innerHeight ? Math.max(8, r.top - h - 6) : r.bottom + 6) + "px";
    const away = (ev) => { if (!pop.contains(ev.target) && !anchor.contains(ev.target)) closePop(); }, key = (ev) => { if (ev.key === "Escape") closePop(); };
    const scrolled = (ev) => { if (!pop.contains(ev.target)) closePop(); };
    document.addEventListener("pointerdown", away, true); document.addEventListener("keydown", key, true); window.addEventListener("scroll", scrolled, true);
    popOff = () => { pop.remove(); anchor.classList.remove("picking"); document.removeEventListener("pointerdown", away, true); document.removeEventListener("keydown", key, true); window.removeEventListener("scroll", scrolled, true); };
    const c = pop.querySelector(".cur"); if (c) c.focus();
  }

  // ---- font actions (a font choice is a user choice: it has a revert too)
  function setFont(x, file) {
    const st = S.story;
    x.font = file; st.edited[`${x.n}.font`] = true;
    const card = document.querySelector(`.sb-card[data-n="${x.n}"]`);
    if (card) paintCard(card, x);
    paintRevs();
  }
  function shuffleAll() {
    const st = S.story, list = S.fonts.list;
    if (!st.data || !list.length) return;
    const f = B.fontPickAll(list, st.data.shots.length);
    st.data.shots.forEach((x, i) => { x.font = f[i]; st.edited[`${x.n}.font`] = true; });
    paintAllCards(); paintRevs();
  }

  function buildCard(x, i, d) {
    const st = S.story, key = (f) => `${x.n}.${f}`, edit = (f, v) => typed(key(f), v);
    const it = storyImage(x), e = it ? readEntry(it) : null, list = S.fonts.list, ids = B.mediaIds(S.media.items);
    const line = el("textarea", { class: "sb-line", rows: 2, spellcheck: "false", placeholder: "No words in this shot", "aria-label": `Shot ${x.n} on-screen line` });
    line.value = x.line;
    line.addEventListener("input", () => { edit("line", line.value); if (B.isVideoShot(x)) syncHitsOfShot(i); paintCard(card, x); });
    const beat = el("textarea", { class: "sb-beat", rows: 3, spellcheck: "false", placeholder: "What happens", "aria-label": `Shot ${x.n} beat` });
    beat.value = x.beat;
    beat.addEventListener("input", () => edit("beat", beat.value));
    const cv = B.verSync(st.vers[key("camera")], x.camera, st.gemma ? B.fieldGet(st.gemma, key("camera")) : undefined);   // every version of the camera is an option (the arrows can land on it)
    const cams = [...new Set(["", ...B.CAMERAS, x.camera, ...B.verList(cv)])];
    const cam = el("select", { class: "sb-cam", "aria-label": `Shot ${x.n} camera` }, cams.map((c) => el("option", { value: c }, c ? B.cameraLabel(c) : "no move")));
    cam.value = x.camera;
    cam.addEventListener("change", () => edit("camera", cam.value));
    const vid = B.isVideoShot(x);                                              // v18: video mode cards = frame, font stepper, editable line (no transition / camera row)
    const iconBtn = (act, title, inner, fn) => el("button", { class: "sb-fbtn", type: "button", "data-act": act, title, "aria-label": title, onclick: fn }, inner);
    const fontRow = list.length ? el("div", { class: "sb-font" },
      iconBtn("prev", "Previous liked font", "‹", () => setFont(x, B.fontStep(list, x.font, -1))),
      el("span", { class: "sb-fname" }),
      iconBtn("next", "Next liked font", "›", () => setFont(x, B.fontStep(list, x.font, 1)))) : el("div", { class: "sb-font-none" });
    const info = e && !vid ? el("button", { class: "icon-btn sb-info" + (st.openNotes[x.n] ? " on" : ""), type: "button", title: "What the read saw in this image", "aria-label": "Image notes", html: svg("info", 13),
      onclick: () => { st.openNotes[x.n] = !st.openNotes[x.n]; notes.hidden = !st.openNotes[x.n]; info.classList.toggle("on", !!st.openNotes[x.n]); } }) : null;
    const notes = e && !vid ? el("div", { class: "sb-notebox", hidden: !st.openNotes[x.n] }, noteEditor(it, e, x.image || ids[S.media.items.indexOf(it)])) : null;
    const ov = el("div", { class: "sb-ov" });
    const swap = it ? el("button", { class: "sb-swap", type: "button", title: "Swap this shot's picture", "aria-label": `Shot ${x.n}: swap the picture`, onclick: (ev) => { ev.stopPropagation(); openPicker(i, img); } }, "swap") : null;
    const img = el("div", { class: "sb-img", tabIndex: 0, role: "button", "aria-pressed": st.active === x.n ? "true" : "false", "aria-label": `Shot ${x.n}: pick this shot's words`,
      title: "Click to work on this shot's words (click again or press Esc to finish)" }, ...storyImgNode(x), ov, swap);
    if (frameRO) frameRO.observe(img);
    img.addEventListener("click", () => setActive(x.n));
    img.addEventListener("keydown", (ev) => { if (ev.target === img && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); setActive(x.n); } });
    const act = st.active === x.n, dimmed = st.active != null && !act;
    const card = el("div", { class: "sb-card" + (act ? " active" : "") + (dimmed ? " dim" : ""), "data-i": i, "data-n": x.n },
      el("div", { class: "sb-top" + (vid ? " vid" : "") }, el("b", {}, `Shot ${x.n}`),
        el("span", { class: "sb-span", title: vid ? "Output time in the piece; the video's own time of this cut in brackets" : null }, B.fmtSpan(x.start, x.end) + (vid ? ` (video ${x.src_start.toFixed(1)}-${x.src_end.toFixed(1)} s)` : "")), info),
      img, fontRow, el("div", { class: "sb-fld sb-plain" }, line),
      ...(vid ? [] : [el("div", { class: "sb-camrow" }, el("span", { class: "sb-k", title: "Camera: the cut that carries this shot into the next one" }, "Camera"), el("div", { class: "sb-fld sb-plain" }, cam)),
                      fld(beat, key("beat")), notes || el("div", { class: "sb-notebox", hidden: true })]));            // video mode: no beat / notes (v16) and no transition / camera (v18) rows
    paintCard(card, x);
    if (vid) loadCutFrame(card, x, null);
    return card;
  }

  /** textareas grow with their text (CSS field-sizing where the browser has it, scrollHeight otherwise). */
  const autosize = (ta) => { if (window.CSS && CSS.supports && CSS.supports("field-sizing", "content")) return; ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; };

  /** video mode: a card's words were edited by hand: the hit(s) landing in that shot follow (one hit per shot; emptied -> no hit). */
  function syncHitsOfShot(i) {
    const d = S.story.data, x = d && d.shots[i];
    if (!x) return;
    if (!d.hits) d.hits = B.hitsFromLines(d.shots, d.shots.map((s, k) => (k === i ? "" : s.line)));
    const last = i === d.shots.length - 1, inside = (h) => h.land >= x.start - 1e-6 && (h.land < x.end - 1e-6 || (last && h.land <= x.end + 1e-6));
    const mine = d.hits.filter(inside), txt = x.line.trim(), rest = d.hits.filter((h) => !inside(h));
    if (txt) rest.push(mine.length ? Object.assign({}, mine[0], { text: txt }) : { text: txt, land: x.start, hold_until: x.end });
    d.hits = rest.sort((a, b) => a.land - b.land);
  }

  /** video mode (v16): the row of cards scrolls sideways: the mouse wheel (and shift + wheel) moves it while it can still move that way, and dragging the row (not a field) pans it. */
  function wireRowScroll(grid) {
    grid.addEventListener("wheel", (ev) => {
      if (ev.ctrlKey || grid.scrollWidth <= grid.clientWidth + 1) return;
      const d = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
      if (!d || (d < 0 && grid.scrollLeft <= 0) || (d > 0 && grid.scrollLeft >= grid.scrollWidth - grid.clientWidth - 1)) return;
      grid.scrollLeft += d; ev.preventDefault();
    }, { passive: false });
    let drag = null;
    grid.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || ev.target.closest("input, textarea, select, button, option")) return;
      drag = { x: ev.clientX, left: grid.scrollLeft, moved: false, id: ev.pointerId };
    });
    grid.addEventListener("pointermove", (ev) => {
      if (!drag || ev.pointerId !== drag.id) return;
      const dx = ev.clientX - drag.x;
      if (!drag.moved && Math.abs(dx) < 4) return;
      if (!drag.moved) { drag.moved = true; try { grid.setPointerCapture(ev.pointerId); } catch (_) { /* no capture */ } grid.classList.add("dragging"); }
      grid.scrollLeft = drag.left - dx;
    });
    const end = () => { drag = null; grid.classList.remove("dragging"); };
    grid.addEventListener("pointerup", end); grid.addEventListener("pointercancel", end);
  }

  function buildStoryboard() {
    const st = S.story, d = st.data, m = S.media;
    const stale = !hasCurrentStory() || storyWorking();
    if (st.active != null && !d.shots.some((x) => x.n === st.active)) st.active = null;           // v18: the active shot is gone (other Length / cuts)
    const wrap = el("div", { class: "storyboard" + (stale ? " stale" : "") + (st.active != null ? " has-active" : "") });
    const nb = buildNotices(); if (nb) wrap.append(nb);                  // v17: the warnings (the toasts) stay visible on the board
    const edit = (key, v) => typed(key, v);
    const title = el("div", { class: "sb-title", role: "heading", "aria-level": "2" }, d.title);        // v19: title + logline are plain text (no input, no version arrows); LOOK / MUSIC stay editable
    const logline = el("div", { class: "sb-logline" }, d.logline);
    const meta = el("div", { class: "sb-meta" });
    for (const [k, label] of [["look", "Look"], ["music", "Music"]]) {             // stacked, full text: wrapping, editable, auto-height
      const inp = el("textarea", { class: "sb-" + k, rows: 1, spellcheck: "false", placeholder: label, "aria-label": label });
      inp.value = d[k];
      inp.addEventListener("input", () => { edit(k, inp.value); autosize(inp); fitBoardSoon(); });
      meta.append(el("div", { class: "sb-m" }, el("span", { class: "sb-k" }, label), fld(inp, k)));
    }
    wrap.append(el("div", { class: "sb-head" }, title, logline, meta));
    requestAnimationFrame(() => { wrap.querySelectorAll(".sb-look, .sb-music").forEach(autosize); fitBoard(); });
    if (textSource().tokens.length) wrap.append(buildTextPanel(d));

    closePop();
    const vgrid = d.shots.length > 0 && d.shots.every(B.isVideoShot) && !!videoItem();                  // v16: video mode = ONE horizontal row of cards that scrolls sideways (no cut strip)
    const grid = el("div", { class: "sb-grid" + (vgrid ? " vid" : ""), style: `--n:${Math.max(1, Math.min(6, d.shots.length))}` });
    if (vgrid) wireRowScroll(grid);
    if (frameRO) frameRO.disconnect();
    d.shots.forEach((x, i) => grid.append(buildCard(x, i, d)));
    wrap.append(grid);
    if (boardRO) boardRO.disconnect();
    if (typeof ResizeObserver !== "undefined") { boardLast = ""; boardRO = new ResizeObserver(() => { const k = wrap.clientWidth + "x" + wrap.clientHeight; if (k !== boardLast) { boardLast = k; fitBoard(); } }); boardRO.observe(wrap); }
    wrap.addEventListener("focusout", (ev) => { if (ev.target.tagName === "TEXTAREA") fitBoardSoon(); });   // no "notes for other files" (user v13): an unused image's read note becomes the shot's beat when it is picked (applyOverride)
    return wrap;
  }

  function buildWork() {
    const st = S.story, d = st.data;
    if (S.fonts.state === "idle") loadFonts();
    const showBar = !!d || storyOn();
    let bar = null;
    if (showBar) {
      bar = el("div", { class: "work-bar" }, el("span", { class: "ro-state sb-state" }));
      if (d) bar.append(
        el("button", { class: "link-btn", type: "button", "data-act": "shuffle-all", title: "A different random liked font for every line", disabled: !S.fonts.list.length, onclick: shuffleAll }, "Shuffle all"));
      bar.append(el("button", { class: "link-btn w-again", type: "button", title: "Write a different story for the same media (your edits stay on top)", disabled: st.busy || !S.media.read, onclick: rewriteStory }, "Write again"));
    }
    // storyboard only (user 2026-10-30: "just have the storyboard - no readout"): before the story exists, a plain status line
    const wrap = el("div", { class: "work" }, bar, d ? buildStoryboard()
      : stepItems().length ? buildSteps()                                // v17: the progress step list (left-aligned rows, done ones grey) replaces the centred caption
      : el("div", { class: "caption story-wait" }, mediaStatusText() || (st.busy ? "Writing the story..." : "The storyboard appears here once your media is read.")));   // the status / Shuffle all bar sits at the top of the box
    setTimeout(paintMediaStatus, 0);
    return wrap;
  }

  // ---- own video (/mograph/own_video): POST, then poll every 3 s
  async function runOwnVideo(r, mf, opts) {
    const route = mf.own_video_route || "/mograph/own_video";
    r.own = { stage: "uploading", log: [], jobId: null, poll: null, fails: 0, soundtrack: soundtrackNow() };
    r.clientId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
    BV.start({ mode: "video", shots: 0, soundtrack: soundtrackNow() });
    BV.setProgress(() => bvOwnProgress(r));
    renderCanvas();
    await uploadPending();
    if (r.cancelled) return endOwn(r, "cancelled");
    const video = S.media.items.find((i) => i.kind === "video");
    const values = { ...S.values };
    let seed = values.seed;
    if (B.isEmpty({ kind: "seed" }, seed) && opts.seeds && opts.seeds.seed !== undefined) seed = opts.seeds.seed;
    seed = B.isEmpty({ kind: "seed" }, seed) ? B.randomSeed() : Number(seed);
    r.used = { seed };
    const brief = currentBrief();
    const edit = B.normEdit(S.media.edit);
    const body = { edit, video: video.uploaded, brief: brief ? Object.assign(JSON.parse(brief), { edit }) : { edit }, length: Number(values.length) || 12, seed,
                   format: values.format || "9:16", quality: OWN_PREVIEW_QUALITY,
                   soundtrack: soundtrackNow(), ...(soundtrackNow() === "track" ? { audio: (S.media.items.find((i) => i.kind === "audio") || {}).uploaded } : {}) };
    r.own.soundtrack = body.soundtrack;
    body.client_id = r.clientId;                                // the job submits its prompts with this id: ComfyUI streams their previews + the music waveform to this page
    await connectWS(r);
    if (r.cancelled) return endOwn(r, "cancelled");
    let resp;
    try { resp = await fetchT(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }, 60000); }
    catch (_) { throw new RunError({ message: "Could not reach the server to start the video job." }); }
    const j = await resp.json().catch(() => ({}));
    if (!resp.ok || !j.job) throw new RunError({ head: "The video job did not start", message: String(j.error || `HTTP ${resp.status}`) });
    if (r.cancelled) { await cancelOwnJob(j.job); return endOwn(r, "cancelled"); }      // cancelled while the POST was in flight
    r.own.jobId = j.job; r.own.stage = "starting"; r.status = "running";
    paintOwn(r);
    r.own.poll = setInterval(() => pollOwn(r), 3000);
    pollOwn(r);
  }

  async function pollOwn(r) {
    if (!r.active || !r.own || r.own.busy) return;
    r.own.busy = true;
    try {
      const j = await getJSON(`${(mediaField() || {}).own_video_route || "/mograph/own_video"}/${encodeURIComponent(r.own.jobId)}`);
      r.own.fails = 0;
      r.own.stage = j.stage || r.own.stage;
      if (Array.isArray(j.stages) && j.stages.length) r.own.stages = j.stages;
      if (Number.isFinite(Number(j.stage_n)) && j.stage_n !== null && j.stage_n !== "") r.own.stageN = Number(j.stage_n);
      if (Array.isArray(j.log)) r.own.log = j.log.slice(-12);
      r.own.cached = j.cached || {};                                          // stages that came back from the stage cache (shown as "reused")
      BV.ownState(j);
      for (const o of j.outputs || []) {
        if (!o || !o.url || S.outputs.some((x) => x.url === o.url)) continue;
        S.outputs.push({ title: "own video", label: o.label || (o.kind === "video" ? "Video" : o.kind), kind: o.kind || B.fileKind({ filename: o.url.split("?")[0] }),
                         url: o.url, final: !!o.final, filename: decodeURIComponent((o.url.split("filename=")[1] || o.url).split("&")[0]).split("/").pop() });
      }
      if (j.cancelled) return endOwn(r, "cancelled");
      if (j.error) return failRun(r, { head: "The video job failed", message: cleanLines(typeof j.error === "string" ? j.error : JSON.stringify(j.error)), details: (r.own.log || []).join("\n") });
      if (j.done || /^(done|finished|complete|completed)$/i.test(String(j.stage || ""))) return endOwn(r, "done");
      paintOwn(r);
    } catch (_) {
      if (++r.own.fails >= 20) failRun(r, { head: "Lost the video job", message: "The server stopped answering for a minute." });
      else setConn("offline");
    } finally { if (r.own) r.own.busy = false; }
  }

  function endOwn(r, status) {
    if (!r.active) return;
    r.active = false; r.status = status;
    stopTimers(r);
    BV.end(status === "done" ? "done" : status === "cancelled" ? "cancelled" : "error");
    if (status === "done" && finalOutput()) BV.decodeFinal(finalOutput().url);
    if (status === "done" && !finalOutput()) {
      const vids = S.outputs.filter((o) => o.kind === "video");
      if (vids.length) vids[vids.length - 1].final = true;
    }
    S.lastTimings = null;
    S.view = status === "done" && finalOutput() ? "result" : "run";
    renderCanvas(); updateRunButton();
    if (status === "done" && !finalOutput()) toast("The job finished, but returned no video.");
  }

  function paintOwn(r) {
    if (S.run !== r || S.view !== "run") return;
    paintOwnSteps(r);
    const prev = $(".previews"); if (prev) prev.replaceWith(buildPreviews());
    $("#btn-download").disabled = !finalOutput();
  }

  // v21: the own-video card uses the same plain step list as the image-mode card (.ov-list / .ov-st rows + bar + %), the job's raw stage names map onto 7 plain steps
  // (16_app/MEDIA_BOX.md "PROGRESS STEPS"); the job log sits behind the "Details" toggle (closed by default).
  const OWN_STEPS = [
    { id: "sound", label: "Sound", stages: ["ingest", "audio", "music"] },
    { id: "beats", label: "Beats & words", stages: ["analyse", "words"] },
    { id: "shots", label: "Cutting the shots", stages: ["prep", "cut"] },
    { id: "matte", label: "Cutting out the subject", stages: ["matte", "archive"] },
    { id: "design", label: "Design", stages: ["design", "parse"] },
    { id: "type", label: "Type & transitions", stages: ["retime", "render"] },
    { id: "finish", label: "Finish", stages: ["finish"] },
  ];
  const OWN_STAGES = ["queued", "ingest", "audio", "music", "analyse", "words", "prep", "cut", "matte", "archive", "design", "parse", "retime", "render", "finish", "done"];
  const ownStepLabel = (s, own) => (s.id === "sound" && own.soundtrack === "generate" ? "Sound & new music" : s.label);
  /** {steps: [{id, label, state: done|current|pending|error}], pct, detail} from the job's stage / stage_n (never the raw names). */
  function ownProgress(run) {
    const own = run.own, stages = own.stages || OWN_STAGES, finished = run.status === "done" || /^(done|finished|complete|completed)$/i.test(String(own.stage || ""));
    let idx = OWN_STEPS.findIndex((s) => s.stages.includes(own.stage));
    if (idx >= 0) own.stepIdx = Math.max(own.stepIdx || 0, idx);
    idx = finished ? OWN_STEPS.length : (own.stepIdx || 0);                    // queued / starting / uploading: the first step; an unknown stage keeps the last one
    const ended = run.status === "error" || run.status === "cancelled";
    const steps = OWN_STEPS.map((s, i) => ({ id: s.id, label: ownStepLabel(s, own),
      state: i < idx ? "done" : i === idx ? (run.status === "error" ? "error" : ended ? "pending" : "current") : "pending" }));
    const at = Number.isFinite(own.stageN) ? own.stageN : Math.max(0, stages.indexOf(own.stage));
    const pct = finished ? 100 : Math.max(0, Math.min(99, Math.round((at / Math.max(1, stages.length - 1)) * 100)));
    const cur = steps[idx];
    const detail = run.status === "error" ? "" : run.status === "cancelled" ? "Stopped." : finished ? "" : own.stage === "uploading" ? "Uploading your media..."
      : own.stage === "starting" || own.stage === "queued" ? "Starting..." : cur ? `${cur.label}...` : "Working...";
    return { steps, pct, detail };
  }
  /** video mode: the Build view's progress bar = the 7 plain steps weighted by how long they take on the box (matte cutting and the render are the long ones). */
  const OWN_W = { sound: 10, beats: 5, shots: 12, matte: 33, design: 12, type: 22, finish: 6 };
  function bvOwnProgress(r) {
    const p = ownProgress(r), finished = r.status === "done";
    const segments = p.steps.map((s) => {
      const st = OWN_STEPS.find((x) => x.id === s.id), k = st.stages.indexOf(r.own.stage);
      const f = finished || s.state === "done" ? 1 : s.state === "current" ? (k < 0 ? 0.08 : (k + 0.5) / st.stages.length) : 0;
      return { id: s.id, label: s.label, w: OWN_W[s.id] || 5, f, state: s.state === "done" ? "done" : s.state === "current" ? "running" : "pending" };
    });
    const W = segments.reduce((a, s) => a + s.w, 0);
    return { segments, overall: finished ? 1 : segments.reduce((a, s) => a + s.w * s.f, 0) / W, label: p.detail };
  }
  function paintOwnSteps(r) {
    const p = ownProgress(r);
    for (const s of p.steps) {
      const row = document.querySelector(`.ov-st[data-step="${s.id}"]`);
      if (!row) continue;
      row.className = "ov-st " + s.state;
      const lab = row.querySelector(".lab"); if (lab && lab.textContent !== s.label) lab.textContent = s.label;
      const sec = row.querySelector(".sec"), cd = r.own.cached || {};
      if (sec) sec.textContent = OWN_STEPS.find((x) => x.id === s.id).stages.some((st) => cd[st]) ? "reused" : "";          // a stage that came back from the stage cache
    }
    const bar = $(".own-bar"), pct = $(".own-pct"), det = $(".own-detail"), lg = $(".own-log");
    if (bar) bar.style.width = p.pct + "%";
    if (pct) pct.textContent = p.pct + "%";
    if (det) det.textContent = p.detail;
    if (lg) { lg.textContent = (r.own.log || []).join("\n"); lg.scrollTop = lg.scrollHeight; }
  }

  function buildOwnRunView(run) {
    const heading = { error: "Something went wrong", cancelled: "Stopped" }[run.status] || (run.active ? "Working on it" : "Done");
    const p = ownProgress(run);
    const ovList = el("div", { class: "ov-list" }, p.steps.map((s) => el("div", { class: "ov-st " + s.state, "data-step": s.id }, el("span", { class: "dot" }), el("span", { class: "lab" }, s.label), el("span", { class: "sec" }, ""))));
    const detBtn = el("button", { class: "link-btn ov-toggle", type: "button", "aria-expanded": S.runDetails ? "true" : "false" }, S.runDetails ? "Hide details" : "Details");
    const log = el("pre", { class: "own-log" }, (run.own.log || []).join("\n"));            // the job log: shown behind the Build panel's Details link (BV.setDetails)
    detBtn.addEventListener("click", () => { BV.toggleDetails(); });
    const card = el("div", { class: "run-card own-card" }, el("h3", {}, heading, el("span", { class: "total" }, run.own.jobId ? `job ${run.own.jobId}` : "")),
      ovList, el("div", { class: "ov-bar" }, el("div", { class: "ov-track" }, el("i", { class: "own-bar", style: `width:${p.pct}%` })), el("span", { class: "ov-pct own-pct" }, p.pct + "%")),
      el("div", { class: "ov-detail own-detail" }, p.detail), detBtn);
    const endNodes = [];                                          // v22: while the Build panel is up it carries the error, the notes and the re-run buttons; the step card is hidden (its rows still update, kept in the DOM)
    if (run.error) endNodes.push(buildErrorBox(run.error));
    if (run.status === "cancelled") endNodes.push(el("div", { class: "note-line" }, "Stopped. The job was cancelled on the box; what it had already made is kept and reused by the next run."));
    if (!run.active) endNodes.push(buildActions(run));
    BV.setDetails(log); BV.setEnd(endNodes);
    const wrap = el("div", { class: "player-wrap has-bv" }, card, buildPreviews());
    BV.mount(wrap, card);
    return wrap;
  }

  // ------------------------------------------------------------------ canvas
  function setZoom(z) {
    S.zoom = Math.max(0.4, Math.min(2, Math.round(z * 10) / 10));
    $("#stage").style.transform = `scale(${S.zoom})`;
    $("#zoom-val").textContent = Math.round(S.zoom * 100) + "%";
  }

  function renderCanvas() {
    const stage = $("#stage");
    stage.textContent = "";
    S.stageRows = null;
    $("#btn-download").disabled = !finalOutput();
    paintSetupBtn();
    if (S.view === "examples") return;   // user 2026-10-30: no example cards; the canvas stays empty until media is added or a run starts
    if (S.view === "example" && S.exampleShown) stage.append(buildExampleView(S.exampleShown));
    else if (S.view === "run") stage.append(buildRunView());
    else if (S.view === "result") stage.append(buildResultView());
    else if (S.view === "read") stage.append(buildWork());
  }

  const finalOutput = () => S.outputs.find((o) => o.final);

  function buildExamples() {
    const grid = el("div", { class: "ex-grid" });
    for (const ex of S.examples) {
      const media = ex.poster && /\.(mp4|webm)$/i.test(ex.poster)
        ? el("video", { src: `/mograph/examples/${ex.poster}`, poster: ex.poster_image ? `/mograph/examples/${ex.poster_image}` : null,
          muted: true, loop: true, playsinline: true, preload: "metadata" })
        : el("img", { class: "poster", src: `/mograph/examples/${ex.poster || ex.inset || ""}`, alt: "" });
      const tile = el("button", { class: "ex", type: "button", title: ex.title || ex.id, onclick: () => loadExample(ex) },
        media,
        ex.inset ? el("div", { class: "inset" }, el("img", { src: `/mograph/examples/${ex.inset}`, alt: "" })) : null,
        el("div", { class: "t" }, ex.title || ex.id, ex.subtitle ? el("small", {}, ex.subtitle) : null));
      if (media.tagName === "VIDEO") {
        media.muted = true;
        tile.addEventListener("mouseenter", () => media.play().catch(() => {}));
        tile.addEventListener("mouseleave", () => { media.pause(); media.currentTime = 0; });
      }
      grid.append(tile);
    }
    if (!S.examples.length) grid.append(el("div", { class: "caption" }, "No examples in app/examples. Add inputs on the left."));
    return el("div", { class: "examples" }, el("h2", {}, "Start with an example"), grid);
  }

  async function loadExample(ex) {
    if (activeRun()) return;
    const { app, graph } = S.tpl;
    const fields = B.visibleFields(app, graph);
    // reset to defaults, then apply the example's inputs
    for (const f of fields) {
      if (f.kind === "image" || f.kind === "audio") { delete S.files[f.id]; if (S.redraw[f.id]) S.redraw[f.id](); }
      else if (f.kind === "media") clearMedia();
      else if (S.setters[f.id]) S.setters[f.id](f.kind === "seed" ? "" : (f.default ?? ""));
    }
    for (const [id, v] of Object.entries(ex.inputs || {})) {
      const mf = fields.find((x) => x.kind === "media");
      if (mf && (id === "image" || id === "audio")) {            // old example ids -> the media box
        try {
          const r = await fetch(`/mograph/examples/${v}`);
          if (!r.ok) throw new Error(r.status);
          const blob = await r.blob();
          await addMediaFiles([new File([blob], v, { type: blob.type })]);
        } catch (e) { toast(`Could not load example file ${v}`); }
        continue;
      }
      const f = fields.find((x) => x.id === id);
      if (!f) continue;
      if (f.kind === "image" || f.kind === "audio") {
        try {
          const r = await fetch(`/mograph/examples/${v}`);
          if (!r.ok) throw new Error(r.status);
          const blob = await r.blob();
          setFile(id, new File([blob], v, { type: blob.type }));
        } catch (e) { toast(`Could not load example file ${v}`); }
      } else if (S.setters[id]) S.setters[id](v);
    }
    S.outputs = []; S.up = newUp();
    S.exampleShown = ex;
    S.view = "example";
    renderCanvas();
    updateRunButton();
  }

  function buildExampleView(ex) {
    const isVid = ex.poster && /\.(mp4|webm)$/i.test(ex.poster);
    const wrap = el("div", { class: "player-wrap" });
    if (isVid) wrap.append(makePlayer(`/mograph/examples/${ex.poster}`, "video"));
    else wrap.append(el("div", { class: "player" }, el("img", { src: `/mograph/examples/${ex.poster || ex.inset}`, alt: "" })));
    wrap.append(el("div", { class: "caption" }, el("b", {}, ex.title || ex.id), " - example inputs loaded. Press ",
      el("b", {}, S.tpl.app.verb || "Run"), " to render your own."));
    return wrap;
  }

  // ---- player with yellow controls
  function makePlayer(url, kind) {
    const wrap = el("div", { class: "player paused" });
    if (kind !== "video") { wrap.append(el("img", { src: url, alt: "" })); return wrap; }
    const v = el("video", { src: url, playsinline: true, preload: "auto", loop: true });
    const play = el("button", { class: "cb", type: "button", title: "Play / pause", html: svg("play", 14) });
    const fill = el("i");
    const bar = el("div", { class: "bar" }, fill);
    const tm = el("span", { class: "tm" }, "00:00");
    const full = el("button", { class: "cb", type: "button", title: "Fullscreen", html: svg("full", 14) });
    const mute = el("button", { class: "cb", type: "button", title: "Mute", html: svg("vol", 14) });
    const toggle = () => (v.paused ? v.play().catch(() => {}) : v.pause());
    play.onclick = toggle;
    v.addEventListener("click", toggle);
    v.addEventListener("play", () => { play.innerHTML = svg("pause", 14); wrap.classList.remove("paused"); });
    v.addEventListener("pause", () => { play.innerHTML = svg("play", 14); wrap.classList.add("paused"); });
    v.addEventListener("timeupdate", () => {
      fill.style.width = (v.duration ? (v.currentTime / v.duration) * 100 : 0) + "%";
      tm.textContent = mmss(v.currentTime);
    });
    bar.addEventListener("click", (e) => {
      const r = bar.getBoundingClientRect();
      if (v.duration) v.currentTime = ((e.clientX - r.left) / r.width) * v.duration;
    });
    full.onclick = () => (document.fullscreenElement ? document.exitFullscreen() : wrap.requestFullscreen && wrap.requestFullscreen());
    mute.onclick = () => { v.muted = !v.muted; mute.innerHTML = svg(v.muted ? "mute" : "vol", 14); };
    v.addEventListener("loadedmetadata", () => wrap.classList.toggle("landscape", v.videoWidth > v.videoHeight));
    wrap.append(v, el("div", { class: "ctrl" }, play, bar, tm, full, mute));
    v.play().catch(() => {});
    return wrap;
  }

  // ---- run view
  function buildRunView() {
    const run = S.run;
    if (run && run.own) return buildOwnRunView(run);
    const card = el("div", { class: "run-card" });
    const total = el("span", { class: "total" }, "");
    const heading = { error: "Something went wrong", cancelled: "Cancelled" }[run && run.status] || "Working on it";
    card.append(el("h3", {}, heading, total));
    const p0 = B.stageProgress(run.tracker.nodes, { shots: run.storyShots });
    const ov = { rows: {}, bar: el("i"), pct: el("span", { class: "ov-pct" }, "0%"), detail: el("div", { class: "ov-detail" }, "Starting...") };
    const ovList = el("div", { class: "ov-list" });
    for (const sg of p0.stages) {
      const sec = el("span", { class: "sec" }, "");
      const row = el("div", { class: "ov-st pending", "data-stage": sg.id }, el("span", { class: "dot" }), el("span", { class: "lab" }, sg.label), sec);
      ov.rows[sg.id] = { row, sec };
      ovList.append(row);
    }
    card.append(ovList, el("div", { class: "ov-bar" }, el("div", { class: "ov-track" }, ov.bar), ov.pct), ov.detail);
    const detBtn = el("button", { class: "link-btn ov-toggle", type: "button", "aria-expanded": S.runDetails ? "true" : "false" }, S.runDetails ? "Hide details" : "Details");
    const list = el("div", { class: "stage-list" });               // the per-node list: shown behind the Build panel's Details link (BV.setDetails)
    detBtn.addEventListener("click", () => { BV.toggleDetails(); });
    card.append(detBtn);
    S.stageRows = { total, rows: {}, ov };
    const { app } = S.tpl;
    for (const n of Object.values(run.tracker.nodes)) {
      const label = (app.stage_labels || {})[n.title] || n.title;
      const sec = el("span", { class: "sec" }, "");
      const bar = el("div", { class: "pbar" });
      const row = el("div", { class: "st" }, el("span", { class: "dot" }),
        el("span", { class: "lab" }, label, label !== n.title ? el("small", {}, n.title) : null), sec, bar);
      S.stageRows.rows[n.id] = { row, sec, bar };
      list.append(row);
    }
    const endNodes = [];                                          // v22: the Build panel carries the error, the notes and the re-run buttons; this step card is hidden while it is up
    if (run.error) endNodes.push(buildErrorBox(run.error));
    if (run.status === "cancelled") endNodes.push(el("div", { class: "note-line" }, "The run was stopped. Nothing was kept."));
    if (run.dropped && run.dropped.length && run.active) endNodes.push(el("div", { class: "note-line" }, `Skipped (optional, not installed): ${run.dropped.join(", ")}`));
    if (!run.active && run.tracker) endNodes.push(buildActions(run));
    BV.setDetails(list); BV.setEnd(endNodes);
    const wrap = el("div", { class: "player-wrap has-bv" }, card, buildPreviews());
    BV.mount(wrap, card);
    paintStages();
    return wrap;
  }

  /** {head, message, hint, details, items} -> readable block: stage title, plain message, fix hint, collapsible traceback. */
  function buildErrorBox(e) {
    const box = el("div", { class: "err-box" });
    if (e.head) box.append(el("div", { class: "err-head" }, e.head, e.classType ? el("small", {}, e.classType) : null));
    if (e.items && e.items.length) {
      const ul = el("ul", { class: "err-list" });
      for (const it of e.items) ul.append(el("li", {}, it.title ? el("b", {}, it.title + ": ") : null, it.message));
      box.append(ul);
    } else box.append(el("div", { class: "err-msg" }, e.message || "Unknown error."));
    if (e.hint) box.append(el("div", { class: "err-hint" }, e.hint));
    if (e.details) {
      const pre = el("pre", {}, e.details);
      box.append(el("details", {}, el("summary", {}, "Technical details"), pre,
        el("button", { class: "link-btn", type: "button", onclick: async () => {
          try { await navigator.clipboard.writeText(e.details); toast("Details copied", 4000, { info: true }); } catch (_) { toast("Clipboard blocked", 4000, { info: true }); }
        } }, "Copy")));
    }
    return box;
  }

  /** Re-run buttons shown after a run ended (done, failed or cancelled). */
  function buildActions(run) {
    const box = el("div", { class: "actions" });
    box.append(el("button", { class: "ghost-btn", type: "button", title: "Same inputs and the same seed", onclick: runAgain },
      run.status === "error" ? "Try again" : "Run again"),
      el("button", { class: "ghost-btn", type: "button", title: "Same inputs, a new random seed", onclick: runNewSeed }, "New seed"));
    return box;
  }

  function buildPreviews() {
    const box = el("div", { class: "previews" });
    for (const o of S.outputs.filter((x) => !x.final)) {
      const media = o.kind === "video"
        ? Object.assign(el("video", { src: o.url, controls: true, loop: true, playsinline: true, autoplay: true }), { muted: true })
        : o.kind === "audio" ? el("audio", { src: o.url, controls: true }) : el("img", { src: o.url, alt: "" });
      box.append(el("div", { class: "prev" }, media, el("div", { class: "pl" }, o.label)));
    }
    return box;
  }

  function buildResultView() {
    const fin = shownOutput();
    const wrap = el("div", { class: "player-wrap" });
    BV.setEnd([]);
    if (BV.plan && BV.status !== "idle") BV.mount(wrap);       // the build timeline stays above the finished clip (collapsible); its playhead follows the player
    const player = fin ? makePlayer(fin.url, fin.kind) : null;
    if (player && fin !== finalOutput()) player.append(el("span", { class: "res-tag", title: "Real-ESRGAN upscale of the Preview" }, "1080"));
    if (player) wrap.append(player);
    if (fin) wrap.append(buildMeta(fin, player));
    { const fb = buildFinalBar(finalOutput()); if (fb) wrap.append(fb); }
    const t = S.lastTimings, seed = S.run && S.run.used && Object.entries(S.run.used).find(([k]) => /seed/i.test(k));
    wrap.append(el("div", { class: "caption" }, t && t.total_seconds !== null ? ["Made in ", el("b", {}, fmtS(t.total_seconds))] : "",
      (seed ? ` · seed ${seed[1]}` : "") + (S.run && S.run.layoutSeed ? ` · layout seed ${S.run.layoutSeed}` : "")));
    const lp = buildLayoutPanel();
    if (lp) wrap.append(lp);
    if (S.run && S.run.tracker) wrap.append(buildActions(S.run));
    wrap.append(buildPreviews());
    return wrap;
  }

  // ---- Final (1080): upscale THIS finished clip (v20). Side graph `_final_upscale` (tools/app/final_upscale_graph.py) = LoadVideo of the output file -> Real-ESRGAN (x2plus for 540x960, x4plus for 360x640)
  // -> lanczos to 1080x1920 (16:9: 1920x1080) -> same fps + audio -> SaveVideo next to the original as <stem>_up1080 (never overwrites). Nothing is re-rendered.
  const upFor = (fin) => (fin && S.up.out && S.up.forUrl === fin.url ? S.up.out : null);
  function shownOutput() { const fin = finalOutput(), u = upFor(fin); return u && S.up.show === "final" ? u : fin; }

  function buildFinalBar(fin) {
    if (!fin || fin.kind !== "video") return null;
    const u = S.up, done = upFor(fin), bar = el("div", { class: "up-bar" });
    if (done) {
      const seg = (label, key, title) => el("button", { class: "seg" + (u.show === key ? " on" : ""), type: "button", "data-show": key, title, onclick: () => { u.show = key; renderCanvas(); } }, label);
      bar.append(el("div", { class: "up-toggle" }, seg("Preview 540", "preview", "The 540x960 Preview, as rendered"), seg("Final 1080", "final", "The 1080x1920 Real-ESRGAN upscale of this Preview")));
    } else {
      bar.append(el("button", { class: "primary up-final", type: "button", "data-act": "final-1080", disabled: u.busy || !!activeRun(), onclick: finalThisClip,
        title: "Upscale this clip to 1080x1920 with Real-ESRGAN; nothing is re-rendered" }, "Final (1080)"));
    }
    if (u.busy) bar.append(el("div", { class: "up-prog" }, el("span", { class: "dot live" }), el("span", { class: "up-t" }, "Upscaling to 1080..."), el("span", { class: "up-d" }, u.detail || "")));
    if (u.err) bar.append(el("div", { class: "err-box" }, u.err));
    return bar;
  }

  function refreshUp() { const old = $(".up-bar"); if (!old || S.view !== "result") return; const neu = buildFinalBar(finalOutput()); if (neu) old.replaceWith(neu); else old.remove(); }

  /** width x height of a finished clip from its metadata (the 9:16 / 16:9 draft size when the browser cannot say). */
  function probeDims(url, fmt) {
    return new Promise((resolve) => {
      const v = document.createElement("video");
      const fallback = String(fmt || "9:16") === "16:9" ? [960, 540] : [540, 960];
      let tm = null;
      const end = (d) => { clearTimeout(tm); v.removeAttribute("src"); try { v.load(); } catch (_) {} resolve(d); };
      tm = setTimeout(() => end(fallback), 8000);
      v.preload = "metadata"; v.muted = true;
      v.onloadedmetadata = () => end(v.videoWidth && v.videoHeight ? [v.videoWidth, v.videoHeight] : fallback);
      v.onerror = () => end(fallback);
      v.src = url;
    });
  }

  async function finalThisClip() {
    const fin = finalOutput(), u = S.up;
    if (!fin || fin.kind !== "video" || u.busy || activeRun()) return;
    u.busy = true; u.err = ""; u.detail = ""; u.forUrl = fin.url; refreshUp();
    try {
      const dims = await probeDims(fin.url, S.values.format);
      let t;
      try { t = await getJSON("/mograph/graphs/_final_upscale"); }
      catch (e) { throw new Error(/: 404$/.test(String(e.message)) ? "This server has no Final (1080) step yet (the _final_upscale graph is missing)." : e.message); }
      const built = B.buildFinalUpscale(t.graph, fin.url, dims[0], dims[1]);
      if (!built) throw new Error(`This clip is already ${dims[0]}x${dims[1]}; nothing to upscale.`);
      const tm = B.titleMap(built.prompt);
      const f = await queueSide(built.prompt, {
        node: tm["OUT upscaled"], label: "the 1080 upscale", timeout: 3600,
        pick: (out) => B.outputFiles(out)[0] || null,
        onStatus: (txt) => { if (S.up !== u) return; u.detail = txt === "queued" ? "waiting for the GPU" : txt; const d = $(".up-prog .up-d"); if (d) d.textContent = u.detail; },
      });
      if (S.up !== u) return;                                   // a new run / template replaced the page state meanwhile
      u.out = { title: "OUT upscaled", label: "Final 1080", kind: "video", url: B.viewUrl(f), final: false, filename: f.filename, up: true };
      u.show = "final"; u.busy = false; u.detail = "";
      renderCanvas(); $("#btn-download").disabled = !finalOutput();
    } catch (e) {
      if (S.up !== u) return;
      u.busy = false; u.detail = ""; u.err = "Final (1080) failed: " + errText(e);
      refreshUp();
    }
  }

  // ---- Layout panel: one frame per word of the piece, drawn again with a new seed (no Gemma, no LTX: about 5-10 s on the box)
  function buildLayoutPanel() {
    const a = S.archive, L = S.layout;
    if (!a) return null;
    const head = el("div", { class: "lp-head" }, el("b", {}, "Layout"), L.seed ? el("span", { class: "lp-seed" }, `seed ${L.seed}`) : null,
      L.lock ? el("span", { class: "lp-lock" }, `locked: ${L.lock.seed}`, el("button", { class: "link-btn", type: "button", onclick: layoutClear }, "clear")) : null);
    const body = L.url
      ? el("a", { href: L.url, target: "_blank", rel: "noreferrer", title: "Open the full sheet" }, el("img", { src: L.url, alt: "Layout preview, one frame per word" }))
      : el("div", { class: "lp-empty" }, L.busy ? "Drawing the layout..." : "Press Randomise to see the text layout of every word on its own still.");
    const busy = L.busy || !!activeRun();
    const btns = el("div", { class: "actions" },
      el("button", { class: "ghost-btn", type: "button", disabled: busy, title: "A new random layout seed: fonts, layouts, palette, outline, hold life and cuts drawn again on the same words", onclick: layoutRandomise },
        L.busy ? "Drawing..." : "Randomise"),
      el("button", { class: "ghost-btn", type: "button", disabled: busy || !L.url, title: "The next Preview / Final of this piece uses this layout seed", onclick: layoutUse }, "Use this layout"));
    return el("div", { class: "layout-panel" }, head, body, L.err ? el("div", { class: "err-box" }, L.err) : null, btns);
  }

  function refreshLayoutPanel() {
    const old = $(".layout-panel");
    const neu = S.view === "result" ? buildLayoutPanel() : null;
    if (old && neu) old.replaceWith(neu);
  }

  /**
   * Queue a side graph through /prompt, follow it on /ws (history polling as the fallback) and resolve with pick(output) of the node
   * `node` (its `executed` output) once pick returns something. opts: {node, pick, label, timeout (s), onStatus(text)}.
   */
  function queueSide(prompt, opts) {
    const { node, pick, label = "the side graph", timeout = 180, onStatus, onQueued } = opts;
    B.labelPrompt(prompt, B.newRunId());                       // App label: its saves go to motngrfx_app/<run>_<name>
    return new Promise((resolve, reject) => {
      const clientId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
      let ws = null, pid = null, done = false, poll = null, shownRunning = false, qChecks = 0;
      const t0 = now();
      const finish = (fn, v) => { if (done) return; done = true; clearInterval(poll); try { if (ws) ws.close(); } catch (_) {} fn(v); };
      const titleOf = (id) => (prompt[id] && prompt[id]._meta && prompt[id]._meta.title) || id;
      try {
        ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${clientId}`);
        ws.onmessage = (ev) => {
          if (typeof ev.data !== "string") return;
          let m; try { m = JSON.parse(ev.data); } catch (_) { return; }
          const d = m.data || {};
          if (pid && d.prompt_id && d.prompt_id !== pid) return;
          if (m.type === "executed" && String(d.node) === String(node)) { const u = pick(d.output); if (u) finish(resolve, u); }
          else if (m.type === "execution_error") finish(reject, new Error(d.exception_message || `${label} failed`));
          else if (onStatus && m.type === "executing" && d.node) onStatus(titleOf(String(d.node)));
          else if (onStatus && m.type === "progress" && d.node && d.max) onStatus(`${titleOf(String(d.node))} ${d.value}/${d.max}`);
        };
      } catch (_) { ws = null; }
      fetchT("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, client_id: clientId }) }, 30000)
        .then((r) => r.json().then((b) => ({ ok: r.ok, b })))
        .then(({ ok, b }) => {
          if (!ok) return finish(reject, new Error(describePromptError(b, prompt).message));
          pid = b.prompt_id;
          poll = setInterval(async () => {
            if (now() - t0 > timeout) return finish(reject, new Error(`${label} timed out`));
            try {
              const h = (await getJSON("/history/" + pid))[pid];
              if (!h) {                                          // "queued" only while this prompt really waits in the box queue (the dev proxy's reads never do)
                if (onStatus && !shownRunning && (++qChecks % 3 === 1)) {
                  const q = await getJSON("/queue").catch(() => ({}));
                  if (!(q.queue_pending || []).some((it) => it[1] === pid)) { shownRunning = true; onStatus("queued"); }
                  else if (onQueued) onQueued();                 // really waiting behind another job in the box queue
                }
                return;
              }
              const u = pick((h.outputs || {})[node]);
              if (u) finish(resolve, u);
              else if (h.status && h.status.status_str === "error") finish(reject, new Error(`${label} failed (see the ComfyUI console)`));
              else if (h.status && h.status.completed) finish(reject, new Error(`${label} finished without a result`));
            } catch (_) { /* retry */ }
          }, 700);
        })
        .catch((e) => finish(reject, e));
    });
  }

  /** the layout graph: resolves with the /view url of the saved sheet. */
  function queueLayout(prompt) {
    return queueSide(prompt, { node: "save", label: "the layout preview",
      pick: (out) => { const f = B.outputFiles(out)[0]; return f ? B.viewUrl(f) : null; } });
  }

  async function layoutRandomise() {
    const L = S.layout, a = S.archive;
    if (!a || L.busy || activeRun()) return;
    L.busy = true; L.err = ""; L.seed = Math.max(1, B.randomSeed());
    refreshLayoutPanel();
    try {
      const t = await getJSON("/mograph/graphs/_layout_preview");      // app/graphs/_layout_preview.json, written by tools/app/layout_preview_graph.py
      const g = t.graph;
      g.layout.inputs.archive = a.dir;
      g.layout.inputs.seed = L.seed;
      g.layout.inputs.effects = L.effects || "";
      g.save.inputs.filename_prefix = `layout_preview/${a.stem}_layout_s${L.seed}`;
      L.url = await queueLayout(g);
    } catch (e) { L.err = "Layout preview failed: " + String((e && e.message) || e); }
    L.busy = false;
    refreshLayoutPanel();
  }

  async function layoutUse() {
    const L = S.layout, a = S.archive;
    if (!a || !L.url || !L.seed) return;
    try {
      const r = await fetchT("/mograph/layout_lock", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archive: a.dir, stem: a.stem, seed: L.seed, effects: L.effects || "" }) });
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).error) || r.status);
    } catch (e) { L.err = "Could not save the layout: " + String((e && e.message) || e); refreshLayoutPanel(); return; }
    L.lock = { seed: L.seed };
    try { localStorage.setItem("mograph.layout.lock", JSON.stringify(L.lock)); } catch (_) {}
    toast(`Layout seed ${L.seed} will be used for the next Preview / Final.`, 4000, { info: true });
    refreshLayoutPanel();
  }

  function layoutClear() {
    S.layout.lock = null;
    try { localStorage.removeItem("mograph.layout.lock"); } catch (_) {}
    refreshLayoutPanel();
  }

  /** resolution / duration / size under the player (read from the loaded video and a HEAD request). */
  function buildMeta(fin, player) {
    const meta = el("div", { class: "meta" });
    const parts = { dim: "", dur: "", size: "" };
    const draw = () => (meta.textContent = [parts.dim, parts.dur, parts.size, (fin.filename || "").split(".").pop().toUpperCase()].filter(Boolean).join(" · "));
    draw();
    const v = player && player.querySelector("video");
    if (v) v.addEventListener("loadedmetadata", () => {
      parts.dim = `${v.videoWidth}×${v.videoHeight}`;
      parts.dur = isFinite(v.duration) ? `${v.duration.toFixed(1)} s` : "";
      draw();
    });
    if (fin.kind === "image" && player) {
      const img = player.querySelector("img");
      if (img) img.addEventListener("load", () => { parts.dim = `${img.naturalWidth}×${img.naturalHeight}`; draw(); });
    }
    fetch(fin.url, { method: "HEAD" }).then((r) => {
      const n = Number(r.headers.get("content-length"));
      if (n) { parts.size = n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`; draw(); }
    }).catch(() => {});
    return meta;
  }

  function paintStages() {
    const run = S.run;
    if (!run || !S.stageRows) return;
    const t = now();
    for (const n of Object.values(run.tracker.nodes)) {
      const r = S.stageRows.rows[n.id];
      if (!r) continue;
      r.row.className = "st " + n.state;
      if (n.state === "running") r.sec.textContent = fmtS(t - n.start);
      else if (n.state === "done") r.sec.textContent = fmtS(n.seconds);
      else if (n.state === "cached") r.sec.textContent = "cached";
      else if (n.state === "error") r.sec.textContent = "failed";
      else r.sec.textContent = "";
      r.bar.style.width = n.state === "running" && n.max ? Math.min(100, (n.value / n.max) * 100) + "%" : "0";
    }
    const t0 = run.tracker.run.t0 ?? run.clickT;
    const t1 = run.tracker.run.t1 ?? t;
    S.stageRows.total.textContent = fmtS(t1 - t0);
    paintOverview(run);
  }

  /** the ~6 plain stages + one overall bar + one detail line (B.stageProgress over the tracker's nodes). */
  function paintOverview(run) {
    const ov = S.stageRows && S.stageRows.ov;
    if (!ov) return;
    const p = B.stageProgress(run.tracker.nodes, { shots: run.storyShots, label: stageLabel });
    const ended = run.status === "done" || run.status === "cancelled" || run.status === "error";
    for (const sg of p.stages) {
      const r = ov.rows[sg.id];
      if (!r) continue;
      const cls = sg.state === "running" && ended ? "pending" : sg.state;
      r.row.className = "ov-st " + cls + (sg.id === p.current && !ended ? " current" : "") + (sg.id === p.current && run.status === "error" ? " current" : "");
      r.sec.textContent = sg.state === "done" ? fmtS(sg.seconds) : sg.state === "error" ? "failed" : sg.state === "running" && !ended ? `${Math.round(sg.frac * 100)}%` : "";
    }
    const pct = run.status === "done" ? 100 : Math.round(p.overall * 100);
    ov.bar.style.width = pct + "%";
    ov.pct.textContent = pct + "%";
    ov.detail.textContent = run.status === "error" ? "" : run.status === "cancelled" ? "Stopped." : (p.detail || (run.tracker.run.t0 === null ? "Starting..." : "Working..."));
  }

  // ------------------------------------------------------------------ server check (/object_info)
  async function loadObjectInfo(force) {
    if (S.objectInfo && !force) return S.objectInfo;
    try { S.objectInfo = await getJSON("/object_info"); } catch (_) { S.objectInfo = null; }
    return S.objectInfo;
  }
  async function loadPacks() {
    if (S.packs) return S.packs;
    try { S.packs = await getJSON("/mograph/static/node_packs.json"); } catch (_) { S.packs = { packs: [] }; }
    return S.packs;
  }

  /** Compare the open template with the nodes this server has; shows what is missing under the Run button. */
  async function preflight(force) {
    const tpl = S.tpl;
    if (!tpl) return null;
    const [info, packs] = await Promise.all([loadObjectInfo(force), loadPacks()]);
    if (S.tpl !== tpl) return null;                     // template changed while we were fetching
    S.pre = B.analyzeAvailability(tpl.graph, tpl.app, info, packs);
    S.flagsWarning = await checkServerFlags(tpl, force);
    renderPreflight();
    updateRunButton();
    return S.pre;
  }

  /** sidecar server_flags {comfyui: [...], why}: warn when ComfyUI was not started with them (/system_stats argv). */
  async function checkServerFlags(tpl, force) {
    const want = ((tpl.app.server_flags || {}).comfyui || []).filter((f) => String(f).startsWith("--") && f !== "--use-sage-attention");   // sage attention is optional
    if (!want.length) return "";
    try {
      if (!S.stats || force) S.stats = await getJSON("/system_stats");
    } catch (_) { return ""; }
    const argv = ((S.stats.system || {}).argv || []).map(String);
    if (!argv.length) return "";
    const lacking = want.filter((f) => !argv.includes(f));
    return lacking.length ? `ComfyUI was not started with ${lacking.join(" ")}. ${tpl.app.server_flags.why || ""}`.trim() : "";
  }

  function renderPreflight() {
    const box = $("#preflight"), pre = S.pre;
    box.textContent = "";
    const extras = [];
    for (const p of S.pruned || []) extras.push(`Skipped (optional, not installed): ${p.removed.join(", ")}${p.note ? ". " + p.note : ""}`);
    if (S.flagsWarning) extras.push(S.flagsWarning);
    const addExtras = () => { for (const t of extras) box.append(el("div", { class: "pf-line" }, t)); };
    if (!pre) { if (extras.length) { box.hidden = false; box.className = "info"; addExtras(); } else box.hidden = true; return; }
    if (!pre.checked) {
      box.hidden = false; box.className = "warn";
      box.append("Could not read /object_info, so installed nodes were not checked.");
      addExtras();
      return;
    }
    if (pre.ok && !pre.droppable.length) { if (extras.length) { box.hidden = false; box.className = "info"; addExtras(); } else box.hidden = true; return; }
    box.hidden = false;
    box.className = pre.ok ? "info" : "bad";
    if (!pre.ok) {
      box.append(el("b", {}, "This server is missing nodes this template needs."));
      const byPack = new Map();
      for (const m of pre.missing) {
        const key = m.pack ? m.pack.name : "(unknown pack)";
        if (!byPack.has(key)) byPack.set(key, { pack: m.pack, items: [] });
        byPack.get(key).items.push(m);
      }
      for (const [name, g] of byPack) {
        const classes = g.items.map((m) => m.class_type).join(", ");
        const line = el("div", { class: "pf-line" }, el("b", {}, g.pack ? name : "Unknown pack"), ": ", classes);
        if (g.pack && g.pack.url) line.append(" ", el("a", { href: g.pack.url, target: "_blank", rel: "noreferrer" }, "repo"));
        if (g.pack && g.pack.install) line.append(el("code", {}, g.pack.install));
        if (!g.pack) line.append(el("small", {}, "Update ComfyUI to the latest release, or install the custom node pack that provides it."));
        for (const m of g.items) for (const bl of m.blockers || []) {
          line.append(el("small", {}, `Optional, but "${bl.node}" needs it on its required input ${bl.input}.`));
        }
        box.append(line);
      }
      box.append(el("small", {}, "Install, restart ComfyUI, then re-check. scripts/check_install.py lists everything."),
        el("button", { class: "link-btn", type: "button", onclick: () => preflight(true) }, "Re-check"));
    }
    for (const m of pre.droppable) {
      box.append(el("div", { class: "pf-line" },
        `${m.pack ? m.pack.name : m.class_type} is not installed: "${m.titles.join('", "')}" will be skipped (optional).${m.note ? " " + m.note : ""}`));
    }
    addExtras();
  }

  // ------------------------------------------------------------------ connection badge
  function setConn(kind) {
    const c = $("#conn");
    if (!kind) { c.hidden = true; return; }
    c.hidden = false;
    c.className = "conn " + kind;
    c.textContent = kind === "offline" ? "Server unreachable - retrying. The job keeps running on the server."
      : "Live link lost - reconnecting...";
  }

  // ------------------------------------------------------------------ running
  const activeRun = () => (S.run && S.run.active ? S.run : null);

  class RunError extends Error { constructor(err) { super(err.message); this.err = err; } }

  async function upload(file) {
    const name = "motngrfx_" + file.name.replace(/[^A-Za-z0-9._-]+/g, "_");
    const fd = new FormData();
    fd.append("image", file, name);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    let r;
    try { r = await fetch("/upload/image", { method: "POST", body: fd }); }
    catch (_) { throw new RunError({ message: `Could not reach the server to upload ${file.name}.` }); }
    if (!r.ok) throw new RunError({ message: `Upload of ${file.name} failed (${r.status}).` });
    const j = await r.json();
    return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
  }

  /** Open the /ws socket of a run. Resolves true when open, false on failure/timeout (the run then polls and retries). */
  function connectWS(r) {
    return new Promise((resolve) => {
      let opened = false, settled = false;
      const done = (v) => { if (!settled) { settled = true; resolve(v); } };
      let ws;
      try { ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${r.clientId}`); }
      catch (_) { done(false); scheduleReconnect(r); return; }
      ws.binaryType = "arraybuffer";
      const timer = setTimeout(() => { if (!opened) { try { ws.close(); } catch (_) {} } }, 8000);
      ws.onopen = () => { opened = true; r.ws = ws; r.retry = 0; r.lastMsg = now(); setConn(null); done(true); };
      ws.onmessage = (ev) => {
        r.lastMsg = now();
        if (typeof ev.data !== "string") {                       // a sampler preview frame (ComfyUI binary message): the Build view draws it in the shot being made
          if (r.own) BV.binary(ev.data); else if (r.promptId) BV.binary(ev.data); else r.pending.push({ bin: ev.data, t: now() });
          return;
        }
        let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
        if (r.own) { BV.ownEvent(msg); return; }                 // video mode: the job's own prompts run on this client id; the page follows the job by polling, the events only feed the Build view
        const item = { msg, t: now() };
        if (!r.promptId) r.pending.push(item); else handleMessage(r, item);
      };
      ws.onerror = () => {};
      ws.onclose = () => {
        clearTimeout(timer);
        if (r.ws === ws) r.ws = null;
        done(false);
        if (r.active) scheduleReconnect(r);
      };
    });
  }

  function scheduleReconnect(r) {
    if (!r.active || r.reconnectTimer) return;
    if (r.promptId) setConn("reconnect");
    const delay = Math.min(15000, 1000 * 2 ** Math.min(r.retry++, 4));
    r.reconnectTimer = setTimeout(() => { r.reconnectTimer = null; if (r.active) connectWS(r); }, delay);
  }

  function stopTimers(r) {
    clearInterval(r.ticker); clearInterval(r.watch); clearTimeout(r.reconnectTimer);
    if (r.own) clearInterval(r.own.poll);
    r.reconnectTimer = null;
    if (r.ws) { r.ws.onclose = null; try { r.ws.close(); } catch (_) {} r.ws = null; }
    setConn(null);
  }

  async function run(opts = {}) {
    if (activeRun() || S.media.making || currentMissing().length) return;
    const mf = mediaField();
    const hasVideo = !!(mf && S.media.items.some((i) => i.kind === "video"));
    if (!hasVideo && S.pre && S.pre.checked && !S.pre.ok) { toast("This server is missing nodes the template needs (see the note under the button)."); return; }
    if (mf && mediaHasInput()) {                                  // a pending / running read finishes first; stale inputs are read again
      S.media.making = true; updateRunButton();
      let ok = false;
      try {
        ok = await ensureRead();
        if (ok) {                                                  // the storyboard Make it renders: a pending story finishes first (video mode too: the story rides in the brief)
          const sOk = await ensureStory();
          if (!sOk) toast("The story could not be written (" + (S.story.err || "unknown error") + "). Making it without a story.", 9000);
        }
      } finally { S.media.making = false; }
      updateRunButton();
      if (!ok) { toast("Could not read your media: " + (S.media.err || "unknown error") + ". Press Re-read to try again.", 9000); return; }
    }
    const { app, graph } = S.tpl;
    const r = { active: true, status: "uploading", storyShots: hasCurrentStory() ? S.story.data.shots.length : 0, clickT: now(), tracker: null, promptId: null, ws: null, ticker: null, watch: null,
                reconnectTimer: null, pending: [], error: null, errorText: "", retry: 0, lost: 0, lastMsg: now(), cancelled: false };
    S.run = r;
    S.outputs = []; S.up = newUp();
    S.archive = null; S.layout.url = ""; S.layout.seed = null; S.layout.err = "";
    S.view = "run";
    S.stageRows = null;
    updateRunButton();
    try {
      if (hasVideo) return await runOwnVideo(r, mf, opts);
      const uploads = {};
      const tU = now();
      for (const f of B.visibleFields(app, graph)) {
        if ((f.kind === "image" || f.kind === "audio") && S.files[f.id]) {
          uploads[f.id] = await upload(S.files[f.id]);
          if (r.cancelled) return endCancelled(r);
        }
      }
      if (mf) { await uploadPending(); uploads[mf.id] = mediaBundle(); if (r.cancelled) return endCancelled(r); }
      r.uploadSeconds = now() - tU;
      const values = { ...S.values };
      for (const f of B.visibleFields(app, graph)) {      // "Run again": an empty seed field reuses the last run's seed
        if (f.kind === "seed" && B.isEmpty(f, values[f.id]) && opts.seeds && opts.seeds[f.id] !== undefined) values[f.id] = opts.seeds[f.id];
      }
      const built = B.buildPrompt(graph, app, values, uploads);
      const prompt = built.prompt;
      r.used = built.used; r.runId = built.runId;
      if (S.layout.lock) {                                  // "Use this layout": the request draws the layout side of the piece system from that seed (MographRequest.layout_seed)
        for (const n of Object.values(prompt)) if (n.class_type === "MographRequest") n.inputs.layout_seed = S.layout.lock.seed;
        r.layoutSeed = S.layout.lock.seed;
      }
      // nodes the server does not have: drop the optional ones, refuse with a clear message for the rest
      const info = await loadObjectInfo(false);
      const pre = B.analyzeAvailability(graph, app, info, await loadPacks());
      if (!pre.ok) {
        S.pre = pre; renderPreflight();
        throw new RunError({ message: "This server is missing nodes the template needs: " + pre.missing.map((m) => m.class_type).join(", ") + ". See the note under the button." });
      }
      r.dropped = B.applyAvailability(prompt, pre, info).map((id) => (graph[id] && graph[id]._meta && graph[id]._meta.title) || id);
      r.prompt = prompt;
      r.tracker = B.createTracker(prompt);
      r.clientId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
      BV.start({ mode: "image", prompt, nodes: () => r.tracker.nodes, shots: r.storyShots, soundtrack: soundtrackNow() });
      renderCanvas();
      r.ticker = setInterval(paintStages, 100);
      await connectWS(r);                                  // false = poll only; the reconnect loop keeps trying
      if (r.cancelled) return endCancelled(r);
      let resp;
      try {
        resp = await fetchT("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, client_id: r.clientId, extra_data: { preview_method: PREVIEW_METHOD, ...(PREVIEW_METHOD !== "none" ? { motngrfx_strip_preview: true } : {}) } }) }, 120000);
      } catch (_) { throw new RunError({ message: "Could not reach the server to start the run. Check the connection and try again." }); }
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new RunError(describePromptError(body, prompt));
      r.promptId = body.prompt_id;
      BV.setPromptId(r.promptId);
      r.status = "running";
      r.watch = setInterval(() => watchdog(r), 3000);
      for (const item of r.pending.splice(0)) handleMessage(r, item);
      if (r.cancelled) cancelRemote(r).then(() => { if (r.active) endCancelled(r); });
    } catch (e) {
      failRun(r, e instanceof RunError ? e.err : { message: String((e && e.message) || e) });
    }
  }

  /** Run again with the same inputs: an empty seed field reuses the seed of the last run. */
  function runAgain() { const r = S.run; run({ seeds: r && r.used }); }
  /** Same inputs, new random seed (a seed typed into the field is replaced). */
  function runNewSeed() {
    for (const f of B.visibleFields(S.tpl.app, S.tpl.graph)) {
      if (f.kind === "seed" && S.setters[f.id] && !B.isEmpty(f, S.values[f.id])) S.setters[f.id](String(B.randomSeed()));
    }
    run();
  }

  // ---- cancel
  async function cancelRun() {
    const r = activeRun();
    if (!r || r.cancelled) return;
    r.cancelled = true;
    if (r.own && r.own.jobId) {                                         // POST <route>/<job>/cancel: the server kills the runner, clears our ComfyUI prompts, frees the lock; what the stages made stays cached for the next run
      const b0 = $("#btn-stop"); b0.textContent = "Cancelling..."; b0.disabled = true;
      await cancelOwnJob(r.own.jobId);
      endOwn(r, "cancelled");
      return;
    }
    const b = $("#btn-stop");
    b.textContent = "Cancelling..."; b.disabled = true;
    if (r.promptId) {
      await cancelRemote(r);
      setTimeout(() => { if (r.active) endCancelled(r); }, 4000);     // no interrupt message arrives when the socket is down
    }
    // before the job exists: run() sees the flag at its next checkpoint
  }

  async function cancelOwnJob(job) {
    const route = (mediaField() || {}).own_video_route || "/mograph/own_video";
    try { await fetchT(`${route}/${encodeURIComponent(job)}/cancel`, { method: "POST" }, 30000); }
    catch (_) { toast("Could not reach the server to stop the video job; it may keep running.", 6000); }
  }

  /** "Back to setup": the setup view (storyboard: words, shots, fonts) with the media, story, word picks and pinned fonts exactly as they were (S is never cleared), so a run can be
   *  changed and started again. While a run is going it is cancelled first (one click). */
  async function backToSetup() {
    if (activeRun()) {
      await cancelRun();
      for (let i = 0; i < 80 && activeRun(); i++) await new Promise((res) => setTimeout(res, 100));      // image mode ends on the server's interrupt message (or after 4 s)
    }
    if (activeRun()) return;
    S.view = mediaHasInput() ? "read" : "examples";
    renderCanvas(); updateRunButton();
  }

  function paintSetupBtn() {
    const b = $("#btn-setup");
    if (!b) return;
    const ar = activeRun();
    b.hidden = !(S.view === "run" || S.view === "result");
    b.textContent = ar ? "Cancel and back to setup" : "Back to setup";
    b.disabled = !!(ar && ar.cancelled);
  }

  /** Remove our job from the queue if it is still pending, and interrupt the server only if our job is the running one. */
  async function cancelRemote(r) {
    const json = { "Content-Type": "application/json" };
    try { await fetchT("/queue", { method: "POST", headers: json, body: JSON.stringify({ delete: [r.promptId] }) }); } catch (_) { /* offline */ }
    let running = true;
    try {
      const q = await getJSON("/queue");
      running = (q.queue_running || []).some((it) => it[1] === r.promptId);
    } catch (_) { /* cannot tell: interrupt anyway, the user asked to stop */ }
    if (running) {
      try { await fetchT("/interrupt", { method: "POST", headers: json, body: JSON.stringify({ prompt_id: r.promptId }) }); } catch (_) { /* offline */ }
    } else {
      await resync(r);                                     // it may have finished meanwhile
      if (r.active) endCancelled(r);
    }
  }

  // ---- describing errors
  const cleanLines = (s, max = 400) => {
    const lines = String(s || "").split("\n").map((x) => x.trim()).filter(Boolean);
    const first = lines.slice(0, 3).join(" ");
    return first.length > max ? first.slice(0, max - 1) + "…" : first;
  };

  function hintFor(text) {
    if (/out of memory|OutOfMemory|CUDA error: out of memory|MPS backend out of memory/i.test(text))
      return "The GPU ran out of memory. Try a shorter Length, or start the smaller graph (add ?graph=motngrfx_16gb or ?graph=motngrfx_3060 to the page address). For big models start ComfyUI with --mmap-torch-files --cache-ram 16 (see the README).";
    if (/Killed|MemoryError|Cannot allocate memory/i.test(text))
      return "The machine ran out of RAM. Start ComfyUI with --mmap-torch-files --cache-ram 16 (see the README).";
    if (/(value_not_in_list|Value not in list|No such file|FileNotFoundError|not found)[\s\S]{0,300}\.(safetensors|pth|gguf|ckpt)/i.test(text))
      return "A model file is missing or has the wrong name. Run python scripts/download_models.py, then python scripts/check_install.py.";
    if (/latent_preview|preview_method|Latent2RGBPreviewer|latent_rgb_factors/i.test(text)) return "The live sampler preview failed for this model. Reload the page with ?preview=none added to the address (the render is the same, only the live pictures in the Build view go away).";
    if (/ffmpeg|ffprobe/i.test(text)) return "ffmpeg must be installed and on PATH for the ComfyUI process.";
    if (/ModuleNotFoundError|No module named/i.test(text)) return "A Python package is missing. Run pip install -r requirements.txt in the ComfyUI environment and restart ComfyUI.";
    return "";
  }

  function stageLabel(title) { return ((S.tpl && S.tpl.app.stage_labels) || {})[title] || title; }

  /** execution_error payload (websocket message or history) -> {head, message, hint, details}. */
  function describeExecError(r, d) {
    const n = d.node_id !== undefined && r.tracker && r.tracker.nodes[String(d.node_id)];
    const msg = d.exception_message || d.exception_type || "execution error";
    const tb = Array.isArray(d.traceback) ? d.traceback.join("") : String(d.traceback || "");
    return {
      head: n ? `Failed in "${stageLabel(n.title)}"` : `Failed in ${d.node_type || "a node"}`,
      node: n ? n.title : null, classType: d.node_type || (n && n.class_type) || null,
      message: cleanLines(msg), hint: hintFor(msg + "\n" + tb),
      details: [d.exception_type ? `${d.exception_type}: ${msg}` : msg, tb].filter(Boolean).join("\n\n"),
    };
  }

  function describePromptError(body, prompt) {
    const items = [];
    if (body.error && (!body.node_errors || !Object.keys(body.node_errors).length)) {
      items.push({ title: null, message: cleanLines(body.error.message || body.error.type || JSON.stringify(body.error)),
                   details: body.error.details || "" });
    }
    for (const [id, ne] of Object.entries(body.node_errors || {})) {
      const title = (prompt[id] && prompt[id]._meta && prompt[id]._meta.title) || ne.class_type || id;
      for (const er of ne.errors || []) items.push({ title, message: cleanLines(er.message + (er.details ? ": " + er.details : "")), details: JSON.stringify(er, null, 1) });
    }
    const text = items.map((i) => (i.title ? i.title + ": " : "") + i.message).join("\n");
    return { head: "The server rejected the workflow", message: text || "The server rejected the workflow.", items,
             hint: hintFor(JSON.stringify(body)), details: JSON.stringify(body, null, 1) };
  }

  function handleMessage(r, item) {
    if (item.bin) { BV.binary(item.bin); return; }                // a preview frame that arrived before the prompt id was known
    const { msg, t } = item;
    const d = msg.data || {};
    if (d.prompt_id && d.prompt_id !== r.promptId) return;
    if (!["execution_start", "execution_cached", "executing", "progress", "progress_state", "executed",
          "execution_success", "execution_error", "execution_interrupted", "motngrfx.waveform", "motngrfx.strip"].includes(msg.type)) return;
    if (!r.active) return;
    r.tracker.feed(msg, t);
    BV.event(msg);
    if (msg.type === "executed") collectOutputs(r.tracker.nodes[String(d.node)], d.output);
    if (msg.type === "execution_success") finishRun(r);
    else if (msg.type === "execution_error") failRun(r, describeExecError(r, d));
    else if (msg.type === "execution_interrupted") {
      if (r.cancelled) endCancelled(r);
      else failRun(r, { head: "Stopped", message: "The run was interrupted on the server (someone pressed Interrupt, or the server restarted)." });
    } else paintStages();
  }

  function collectOutputs(node, output) {
    if (!node) return;
    BV.output(node.title, output);                              // the Build view: pictures, shot clips (Archive), the song, the final video
    if (output && Array.isArray(output.text)) {              // MographArchive's ui text = {dir, stem, ...}: the folder the Layout panel reads
      for (const t of output.text) {
        try { const j = JSON.parse(t); if (j && j.dir && j.stem) S.archive = { dir: j.dir, stem: j.stem }; } catch (_) { /* not JSON */ }
      }
    }
    const spec = ((S.tpl.app.outputs) || []).find((o) => o.node_title === node.title)
      || (node.title.startsWith("OUT ") ? { kind: null, label: node.title.slice(4), final: node.title === "OUT final" } : null);
    if (!spec) return;
    for (const f of B.outputFiles(output)) {
      const url = B.viewUrl(f);
      if (S.outputs.some((o) => o.url === url)) continue;
      const kind = spec.kind || B.fileKind(f);
      S.outputs.push({ title: node.title, label: spec.label || node.title, kind, url, final: !!spec.final, filename: f.filename });
    }
    if (S.view === "run") { const prev = $(".previews"); if (prev) prev.replaceWith(buildPreviews()); }
    $("#btn-download").disabled = !finalOutput();
  }

  async function finishRun(r) {
    if (!r.active) return;
    r.active = false;
    stopTimers(r);
    // history is the source of truth for outputs (covers any websocket message we missed)
    // (the entry is written just after execution_success, so retry briefly)
    for (let i = 0; i < 20; i++) {
      try {
        const h = (await getJSON("/history/" + r.promptId))[r.promptId];
        if (h) { for (const [id, out] of Object.entries(h.outputs || {})) collectOutputs(r.tracker.nodes[id], out); break; }
      } catch (_) { /* retry */ }
      await new Promise((res) => setTimeout(res, 250));
    }
    r.status = "done";
    BV.end("done");
    { const fin = finalOutput(); if (fin) BV.decodeFinal(fin.url); }
    const timings = r.tracker.summary({
      template: S.tpl.name, prompt_id: r.promptId, finished_at: new Date().toISOString(),
      upload_seconds: Math.round((r.uploadSeconds || 0) * 1000) / 1000,
      wall_seconds: Math.round((now() - r.clickT) * 1000) / 1000, inputs: r.used, skipped_nodes: r.dropped || [],
    });
    S.lastTimings = timings;
    window.__mographTimings = timings;
    try { localStorage.setItem("mograph.lastTimings", JSON.stringify(timings)); } catch (_) {}
    console.log("MOGRAPH_TIMINGS " + JSON.stringify(timings));
    S.view = finalOutput() ? "result" : "run";
    renderCanvas();
    updateRunButton();
    renderDrawer();
    if (!finalOutput()) toast("Finished, but no 'OUT final' output was found.");
  }

  /** err: string or {head, message, hint, details, items}. */
  function failRun(r, err) {
    if (!r.active) return;
    const e = typeof err === "string" ? { message: err } : err;
    r.active = false; r.status = "error"; r.error = e; r.errorText = e.message;
    stopTimers(r);
    BV.end("error");
    if (r.tracker) {
      const timings = r.tracker.summary({ template: S.tpl.name, prompt_id: r.promptId, error: e.message, inputs: r.used });
      S.lastTimings = timings; window.__mographTimings = timings;
      console.log("MOGRAPH_TIMINGS " + JSON.stringify(timings));
    }
    if (!r.tracker && !r.own) { toast(e.message, 7000); S.view = S.exampleShown ? "example" : "examples"; }
    else S.view = "run";
    renderCanvas();
    updateRunButton();
    renderDrawer();
  }

  function endCancelled(r) {
    if (!r.active) return;
    r.active = false; r.status = "cancelled";
    stopTimers(r);
    BV.end("cancelled");
    if (r.tracker) {
      S.lastTimings = r.tracker.summary({ template: S.tpl.name, prompt_id: r.promptId, cancelled: true, inputs: r.used });
      S.view = "run";
    } else {
      toast("Cancelled.", 4000, { info: true });
      S.view = S.exampleShown ? "example" : "examples";
    }
    renderCanvas();
    updateRunButton();
    renderDrawer();
  }

  // ---- connection loss: the job runs on the server, so recover its state from /history and /queue
  async function watchdog(r) {
    if (!r.active || !r.promptId) return;
    const wsUp = r.ws && r.ws.readyState === WebSocket.OPEN;
    if (!wsUp || now() - r.lastMsg > 10) await resync(r);
  }

  async function resync(r) {
    if (!r.active || !r.promptId || r.syncing) return;
    r.syncing = true;
    try {
      const q = await getJSON("/queue");                   // queue first, history second: a job that finishes in between is then in history
      const h = (await getJSON("/history/" + r.promptId))[r.promptId];
      const wsUp = r.ws && r.ws.readyState === WebSocket.OPEN;
      setConn(wsUp ? null : "reconnect");              // the server answers, only the live link is down
      if (h && h.status) {
        const msgs = h.status.messages || [];
        const errMsg = msgs.find((m) => m[0] === "execution_error");
        if (errMsg) return failRun(r, describeExecError(r, errMsg[1] || {}));
        if (msgs.some((m) => m[0] === "execution_interrupted")) {
          return r.cancelled ? endCancelled(r) : failRun(r, { head: "Stopped", message: "The run was interrupted on the server." });
        }
        if (h.status.status_str === "error") return failRun(r, "Execution failed (see the ComfyUI console).");
        if (h.status.completed) { r.tracker.feed({ type: "execution_success", data: {} }, now()); return finishRun(r); }
      }
      const known = [...(q.queue_running || []), ...(q.queue_pending || [])].some((it) => it[1] === r.promptId);
      if (known || h) r.lost = 0;
      else if (++r.lost >= 5) {
        failRun(r, { head: "Lost the job", message: "The server no longer knows this job (it probably restarted). Press Run again." });
      }
    } catch (_) {
      setConn("offline");
    } finally {
      r.syncing = false;
    }
  }

  // ------------------------------------------------------------------ download + timings drawer
  function download() {
    const fin = shownOutput();
    if (!fin) return;
    const a = el("a", { href: fin.url, download: fin.filename || "motngrfx.mp4" });
    document.body.append(a); a.click(); a.remove();
  }

  function renderDrawer() {
    const body = $("#drawer-body");
    const t = S.lastTimings;
    body.textContent = "";
    $("#drawer-total").textContent = t && t.total_seconds !== null ? `total ${fmtS(t.total_seconds)}` : "";
    if (!t) { body.append(el("div", { class: "empty" }, "No run yet. Per-node durations of the last run appear here and are logged to the console as JSON.")); return; }
    const max = Math.max(0.001, ...t.nodes.map((n) => n.seconds || 0));
    const tb = el("table");
    for (const n of t.nodes) {
      tb.append(el("tr", {}, el("td", {}, n.title, " ", el("small", {}, n.class_type),
        el("div", { class: "bar", style: `width:${Math.round(((n.seconds || 0) / max) * 100)}%` })),
        el("td", {}, n.state === "cached" ? "cached" : fmtS(n.seconds))));
    }
    body.append(tb);
    if (t.wall_seconds !== undefined) body.append(el("div", { class: "empty" }, `Wall time ${fmtS(t.wall_seconds)} incl. uploads ${fmtS(t.upload_seconds)}`));
    if (t.error) body.append(el("div", { class: "err-box" }, t.error));
  }

  async function copyTimings() {
    const txt = JSON.stringify(S.lastTimings || {}, null, 2);
    try { await navigator.clipboard.writeText(txt); toast("Timings JSON copied", 4000, { info: true }); }
    catch (_) { console.log(txt); toast("Clipboard blocked; JSON is in the console", 4000, { info: true }); }
  }

  // ------------------------------------------------------------------ custom dropdown menu
  /** Every <select> opens a menu in the app look instead of the OS popup. The native select stays the closed control and the source of
   *  truth (value, options, disabled and option text are read from it each time the menu opens); a MutationObserver enhances selects added later. */
  const coarsePointer = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;   // phones: the native sheet is fine
  let selMenu = null;                                                                                    // { sel, pop, items, hi, typed, tTyped, off }
  function closeSelMenu(refocus) {
    const m = selMenu; if (!m) return;
    selMenu = null;
    m.off();
    m.pop.remove();
    m.sel.setAttribute("aria-expanded", "false");
    if (refocus && m.sel.isConnected) m.sel.focus({ preventScroll: true });
  }
  function openSelMenu(sel) {
    if (selMenu && selMenu.sel === sel) return closeSelMenu(true);
    closeSelMenu(false);
    if (sel.disabled || !sel.isConnected) return;
    const pop = document.createElement("div");
    pop.className = "mg-menu"; pop.setAttribute("role", "listbox"); pop.tabIndex = -1;
    if (sel.getAttribute("aria-label")) pop.setAttribute("aria-label", sel.getAttribute("aria-label"));
    const items = [];
    const addOpt = (o, parent, group) => {
      const it = document.createElement("div");
      it.className = "mg-opt"; it.setAttribute("role", "option"); it.id = `mg-opt-${items.length}`;
      const txt = o.textContent, cut = o.disabled ? /^(.*\S)\s+(\([^()]*\))$/.exec(txt) : null;       // "Horizontal 16:9 (video is 9:16 only)": the reason is muted and smaller
      const lb = document.createElement("span"); lb.className = "mg-lbl"; lb.textContent = cut ? cut[1] : txt; it.append(lb);
      if (cut) { const w = document.createElement("small"); w.textContent = cut[2]; it.append(w); }
      const ck = document.createElement("span"); ck.className = "mg-ck"; ck.innerHTML = svg("check", 13); it.append(ck);
      const on = sel.selectedIndex === o.index;
      it.setAttribute("aria-selected", on ? "true" : "false"); if (on) it.classList.add("sel");
      if (o.disabled || (group && group.disabled)) { it.setAttribute("aria-disabled", "true"); it.classList.add("dis"); }
      it.dataset.index = String(o.index);
      parent.append(it); items.push(it);
    };
    for (const c of sel.children) {
      if (c.tagName === "OPTGROUP") {
        const g = document.createElement("div"); g.className = "mg-group"; g.setAttribute("role", "group");
        const h = document.createElement("div"); h.className = "mg-ghead"; h.textContent = c.label; g.append(h);
        for (const o of c.children) if (o.tagName === "OPTION" && !o.hidden) addOpt(o, g, c);
        pop.append(g);
      } else if (c.tagName === "OPTION" && !c.hidden) addOpt(c, pop, null);
    }
    if (!items.length) return;
    document.body.append(pop);
    // position: under the control (the whole row when the select sits in one), flipped above when there is no room
    const r = (sel.closest(".row-select") || sel).getBoundingClientRect(), vh = window.innerHeight, vw = window.innerWidth;
    pop.style.minWidth = `${Math.round(r.width)}px`;
    const below = vh - r.bottom - 10, above = r.top - 10, want = Math.min(pop.scrollHeight, 320), up = below < want && above > below;
    pop.style.maxHeight = `${Math.max(90, Math.min(320, up ? above : below))}px`;
    const w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = `${Math.max(8, Math.min(r.left, vw - w - 8))}px`;
    pop.style.top = `${Math.max(8, up ? r.top - h - 4 : r.bottom + 4)}px`;
    if (up) pop.classList.add("up");
    sel.setAttribute("aria-expanded", "true");
    const m = selMenu = { sel, pop, items, hi: -1, typed: "", tTyped: 0, off: () => {} };
    const enabled = (i) => i >= 0 && i < items.length && !items[i].classList.contains("dis");
    const hilite = (i, scroll) => {
      if (m.hi >= 0 && items[m.hi]) items[m.hi].classList.remove("hi");
      m.hi = i;
      if (i < 0) { pop.removeAttribute("aria-activedescendant"); return; }
      items[i].classList.add("hi"); pop.setAttribute("aria-activedescendant", items[i].id);
      if (scroll) items[i].scrollIntoView({ block: "nearest" });
    };
    const step = (from, dir) => { for (let i = from + dir; i >= 0 && i < items.length; i += dir) if (enabled(i)) return i; return from; };
    const choose = (i) => {
      if (!enabled(i)) return;
      const v = sel.options[Number(items[i].dataset.index)].value, changed = sel.value !== v;
      closeSelMenu(true);
      if (changed) sel.value = v;
      sel.dispatchEvent(new Event("input", { bubbles: true }));
      if (changed) sel.dispatchEvent(new Event("change", { bubbles: true }));
    };
    items.forEach((it, i) => {
      it.addEventListener("mousedown", (e) => e.preventDefault());                    // keep focus in the menu
      it.addEventListener("click", () => choose(i));
      it.addEventListener("mousemove", () => { if (enabled(i) && m.hi !== i) hilite(i, false); });
    });
    pop.addEventListener("keydown", (e) => {
      const k = e.key;
      if (k === "ArrowDown") hilite(m.hi < 0 ? step(-1, 1) : step(m.hi, 1), true);
      else if (k === "ArrowUp") hilite(m.hi < 0 ? step(items.length, -1) : step(m.hi, -1), true);
      else if (k === "Home") hilite(step(-1, 1), true);
      else if (k === "End") hilite(step(items.length, -1), true);
      else if (k === "Enter" || k === " ") { if (m.hi >= 0) choose(m.hi); }
      else if (k === "Escape") closeSelMenu(true);
      else if (k === "Tab") { closeSelMenu(true); return; }
      else if (k.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {            // type-ahead
        const now = Date.now(); m.typed = now - m.tTyped > 700 ? k.toLowerCase() : m.typed + k.toLowerCase(); m.tTyped = now;
        const pick = (from) => { for (let n = 0; n < items.length; n++) { const i = (from + n) % items.length; if (enabled(i) && items[i].textContent.toLowerCase().startsWith(m.typed)) return i; } return -1; };
        const i = pick(m.typed.length > 1 ? Math.max(m.hi, 0) : m.hi + 1); if (i >= 0) hilite(i, true);
      } else return;
      e.preventDefault(); e.stopPropagation();
    });
    const onDown = (e) => { if (!pop.contains(e.target) && e.target !== sel) closeSelMenu(false); };
    const onScroll = (e) => { if (!pop.contains(e.target)) closeSelMenu(false); };
    const onOut = (e) => { if (!pop.contains(e.relatedTarget) && e.relatedTarget !== sel) closeSelMenu(false); };
    const onClose = () => closeSelMenu(false);
    document.addEventListener("mousedown", onDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onClose);
    window.addEventListener("blur", onClose);
    pop.addEventListener("focusout", onOut);
    m.off = () => { document.removeEventListener("mousedown", onDown, true); window.removeEventListener("scroll", onScroll, true); window.removeEventListener("resize", onClose); window.removeEventListener("blur", onClose); };
    pop.focus({ preventScroll: true });
    const cur = items.findIndex((x) => x.dataset.index === String(sel.selectedIndex));
    hilite(enabled(cur) ? cur : step(-1, 1), true);
  }
  function enhanceSelect(sel) {
    if (coarsePointer || sel.dataset.mg === "1") return;
    sel.dataset.mg = "1";
    sel.setAttribute("aria-haspopup", "listbox"); sel.setAttribute("aria-expanded", "false");
    sel.addEventListener("mousedown", (e) => { if (e.button !== 0) return; e.preventDefault(); if (!sel.disabled) { sel.focus({ preventScroll: true }); openSelMenu(sel); } });
    sel.addEventListener("click", (e) => e.preventDefault());
    sel.addEventListener("keydown", (e) => {
      if (selMenu) return;
      if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown" || (e.altKey && e.key === "ArrowUp")) { e.preventDefault(); openSelMenu(sel); }
    });
  }
  function enhanceSelectsIn(root) {
    if (root.nodeType !== 1) return;
    if (root.tagName === "SELECT") enhanceSelect(root);
    else if (root.querySelectorAll) root.querySelectorAll("select").forEach(enhanceSelect);
  }
  if (typeof MutationObserver !== "undefined") {
    new MutationObserver((muts) => {
      for (const mu of muts) mu.addedNodes.forEach(enhanceSelectsIn);
      if (selMenu && !selMenu.sel.isConnected) closeSelMenu(false);
    }).observe(document.body || document.documentElement, { childList: true, subtree: true });
  }
  enhanceSelectsIn(document.body);

  boot().catch((e) => { console.error(e); toast(String(e)); });
  window.MographApp = { state: S, soundtrackNow, addMediaFiles, removeMedia, moveImage, doRead, ensureRead, currentBrief, readBrief, storyIsCurrent: hasCurrentStory, doStory, ensureStory, rewriteStory, scheduleStory, mediaBundle, run, runAgain, runNewSeed, cancelRun, backToSetup, loadExample, preflight, toast, build: BV };
})();
