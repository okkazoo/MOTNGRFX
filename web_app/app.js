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
  const PLATES_QUALITY = "draft540plates";   // image board edited like a video (2026-11-03): the same Preview render but the graph hands back each shot's plain footage (still_preview.PLATES_ONLY_Q); the page then sends that clip to the own-video overlay route
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
    { const nb = $("#btn-new"); let armed = false, t = 0;                                 // 2026-11-03 user: start again on a fresh page. The page's work lives in memory, so a reload clears it; a second click confirms, and a running render is never abandoned
      const say = (txt, on) => { armed = on; nb.textContent = txt; clearTimeout(t); t = setTimeout(() => { armed = false; nb.textContent = "Start again"; }, 3000); };
      let at = 0; nb.onclick = () => { if (activeRun()) return say("Stop the run first", false); if (armed) { if (performance.now() - at > 400) location.replace(location.pathname + "?new"); return; } at = performance.now(); say("Clear everything?", true); }; }   // a quick double-click is not a confirmation
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
    sel.addEventListener("change", () => { if (f.id === "format" && sel.dataset.vscale === "1") { vidScale = sel.value === "half" ? "half" : "full"; return; } S.values[f.id] = sel.value; if (f.id === "length") { paintLength(); scheduleRead(); scheduleStory(2000); } });
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
      paintFormat(); fitBoardSoon();                            // the Format options name the clip's real sizes, the board frames take its real aspect
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
  const isVidBoard = (d) => !!d && d.shots.length > 0 && d.shots.every(B.isVideoShot) && !!videoItem();
  /** IMAGE FRAME BOARD (2026-11-03 user: images + your own music are edited like a video): no video item, every shot has its dropped image. The shots keep start / end (the piece clock, the song plays from 0),
   *  they get NO src_start / src_end (that would make them video shots: card text, story fix, route). Everything the video board does (strip, open cards, on-frame type tools, layer preview, lyrics + wave) runs on these. */
  const isImgBoard = (d) => !!d && d.shots.length > 0 && !videoItem() && d.shots.every((x) => !B.isVideoShot(x) && !!storyImage(x));
  const isFrameBoard = (d) => isVidBoard(d) || isImgBoard(d);
  const isFrameShot = (x) => (B.isVideoShot(x) ? !!videoItem() : !videoItem() && !!storyImage(x));
  /** the shot's own clock: a video cut's source seconds, an image shot's piece seconds (B.shotSpan). */
  const shotT0 = (x) => B.shotSpan(x)[0], shotT1 = (x) => B.shotSpan(x)[1];
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
  /** VIDEO MODE Format (2026-11-03): the select offers the clip's own size ("Original (W×H)", the default) or half of it ("Half scale (w×h)"); the job gets it as `scale` ("full" | "half").
   *  S.values.format stays a valid aspect string = the clip's orientation ("16:9" | "9:16"), so the story request / signature and the frame ratio never see the scale (Original <-> Half re-reads and
   *  re-writes nothing). Sizes mirror tools/app/overlay_mode.scaled_size: the clip's display size (rotation applied by the browser) floored to even, half = that / 2 floored to even.
   *  Leaving video mode puts the normal options and value back. */
  let vidScale = "full", fmtNormal = null;
  const evenDown = (n) => Math.max(2, Math.floor(Number(n) / 2) * 2);
  function vidSizes() {
    const v = videoItem(), d = v && v.dim;
    if (!d || !(d[0] > 0) || !(d[1] > 0)) return null;
    const w = evenDown(d[0]), h = evenDown(d[1]);
    return { full: [w, h], half: [evenDown(w / 2), evenDown(h / 2)], format: w > h ? "16:9" : "9:16" };
  }
  function paintFormat() {
    const fs = document.querySelector("select[data-id=format]"), v = videoItem();
    if (!fs) return;
    if (!v) {
      if (fmtNormal) {
        const n = fmtNormal; fmtNormal = null;
        fs.textContent = "";
        for (const o of n.opts) { const op = el("option", { value: o.value }, o.text); op.disabled = !!o.disabled; fs.append(op); }
        delete fs.dataset.vscale; delete fs.dataset.vsig;
        fs.value = n.value; if (fs.value !== n.value) fs.selectedIndex = 0;
        S.values.format = fs.value;
      }
      return;
    }
    if (fs.dataset.vscale !== "1") {
      if (!fmtNormal) fmtNormal = { opts: [...fs.options].map((o) => ({ value: o.value, text: o.textContent, disabled: o.disabled })), value: S.values.format };
      fs.dataset.vscale = "1";
    }
    const z = vidSizes(), sig = z ? `${z.full.join("x")}/${z.half.join("x")}` : "?";
    if (fs.dataset.vsig !== sig) {
      const lab = (k, t) => (z ? `${t} (${z[k][0]}×${z[k][1]})` : t);
      fs.textContent = "";
      fs.append(el("option", { value: "full" }, lab("full", "Original")), el("option", { value: "half" }, lab("half", "Half scale")));
      fs.dataset.vsig = sig;
    }
    fs.value = vidScale;
    if (z) S.values.format = z.format; else if (!/^\d+:\d+$/.test(String(S.values.format || ""))) S.values.format = "9:16";
  }
  function paintLength() {
    const sel = document.querySelector("select[data-id=length]"), note = document.querySelector(".len-note");
    paintFormat();
    if (!sel) return;
    const lines = [], dur = videoDur(), opts = baseLengthOptions(sel);
    if (videoItem()) {
      // 2026-11-03 user: the piece keeps the video's own length, always: one option, the exact duration (rounded up so the last frame stays in; the job's window stops at the clip's end)
      for (const o of [...sel.options]) if (o.dataset.vlen) o.remove();
      for (const o of sel.options) { o.hidden = true; o.disabled = true; }
      if (Number.isFinite(dur)) {
        const o = document.createElement("option"); o.value = String(Math.ceil(dur * 100) / 100); o.dataset.vlen = "1"; o.textContent = `Whole video (${B.fmtDur(dur)} s)`;
        sel.append(o); sel.value = o.value;
        if (S.values.length !== o.value) { S.values.length = o.value; scheduleStory(2000); }
      }
      sel.disabled = true;
      if (Number.isFinite(dur)) lines.push(`The story is built from the video's own cuts.`);
      if (note) { note.textContent = lines.join(" "); note.hidden = !lines.length; }
      return;
    }
    for (const o of [...sel.options]) if (o.dataset.vlen) o.remove();
    for (const o of sel.options) o.hidden = false;
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
      return;                                                                         // paintLength above set it to the whole video
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
  /** corrected lyrics: m.lyricFix[heard word index] = the user's spelling. The read itself stays untouched (so no re-read / re-write fires). Several words in place of one share its sung time. */
  function applyLyricFix(t, m) {
    if (m.lyricFixFor !== m.read) { m.lyricFix = {}; m.lyricFixFor = m.read; }
    const fix = m.lyricFix, out = { tokens: [], times: [], base: [], piece: [] };
    t.tokens.forEach((tok, b) => {
      const ps = String(fix[b] != null ? fix[b] : tok).trim().split(/\s+/).filter(Boolean), t0 = Number(t.times[b]), nx = Number(t.times[b + 1]), t1 = Number.isFinite(nx) && nx > t0 ? nx : t0 + 0.3;
      ps.forEach((p, k) => { out.tokens.push(p); out.times.push(Math.round((t0 + (t1 - t0) * k / ps.length) * 100) / 100); out.base.push(b); out.piece.push(k); });
    });
    return out;
  }
  /** correct one word of the lyrics: it keeps its sung time, and every shot using it updates its words (their stacks and layout stay). */
  function correctToken(o, text) {
    const st = S.story, d = st.data, src = textSource();
    if (!d || !src.base) return;
    const nm = (w) => String(w).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ""), b = src.base[o], k = src.piece[o], oldTok = src.tokens[o], nw = String(text).trim().split(/\s+/).filter(Boolean);
    if (nw.join(" ") === oldTok) return;
    const parts = src.tokens.filter((_, j) => src.base[j] === b); parts.splice(k, 1, ...nw); S.media.lyricFix[b] = parts.join(" ");
    const users = new Set(), sel = [], add = nw.length - 1;
    for (const t of d.text_select || []) {
      if (t.i < o) sel.push(t); else if (t.i > o) sel.push(Object.assign({}, t, { i: t.i + add }));
      else { users.add(t.shot); nw.forEach((_, q) => sel.push(Object.assign({}, t, { i: o + q }))); }
    }
    d.text_select = sel;
    for (const x of d.shots) {
      if (users.has(x.n)) {
        const up = x.line === x.line.toUpperCase();
        x.line = x.line.split(/(\s+)/).map((w) => (nm(w) && nm(w) === nm(oldTok) ? (up ? nw.join(" ").toUpperCase() : nw.join(" ")) : w)).join("").trim();
        delete st.edited[`${x.n}.line`];
      }
      if (users.has(x.n) || (x.custom && x.custom.length)) reconcileShot(x.n, true);
    }
    syncAllVers(true); rerender();
  }
  /** a shot's words were typed: link them to the lyrics. A word sung in this shot lights up there and lands when it is sung; a free word sung elsewhere lights up there; anything else is a custom word
   *  (dashed chip at the start of the shot's words, lands at the shot's start). A word another shot uses is not taken from it. One or two letters off a word of this shot offers "Fix lyric?". */
  function reconcileShot(n, quiet) {
    const st = S.story, d = st.data, x = d && d.shots.find((s) => s.n === n), src = textSource();
    if (!x || !src.lyrics || !src.times || !isFrameBoard(d)) return;
    const nm = (w) => String(w).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ""), sung = B.sungByShot(d.shots, src.times).by.get(n) || [], where = new Map();
    src.tokens.forEach((t, i) => { const k = nm(t); if (k) { if (!where.has(k)) where.set(k, []); where.get(k).push(i); } });
    const taken = new Set((d.text_select || []).filter((t) => t.shot !== n).map((t) => t.i)), bits = x.line.trim().split(/(\s+)/), words = [];
    for (let k = 0; k < bits.length; k += 2) {                                       // a line break inside a word (PRES / SURE) still reads as that word
      let w = bits[k];
      while (k + 2 < bits.length && /\n/.test(bits[k + 1]) && !where.has(nm(w)) && [...where.keys()].some((q) => q.startsWith(nm(w + bits[k + 2])))) { w += bits[k + 2]; k += 2; }
      if (w) words.push(w);
    }
    const used = new Set(), picked = [], custom = []; let last = -1;
    for (const w of words) {
      const c = (where.get(nm(w)) || []).filter((i) => !used.has(i) && !taken.has(i)), inS = c.filter((i) => sung.includes(i)), t0 = shotT0(x);
      const i = inS.find((q) => q > last) ?? inS[0] ?? c.slice().sort((p, q) => Math.abs(src.times[p] - t0) - Math.abs(src.times[q] - t0))[0];
      if (i != null) { picked.push(i); used.add(i); last = i; } else custom.push({ t: w, near: null, after: picked.length ? picked[picked.length - 1] : null });   // remembers which lyric it follows on the frame
    }
    const lev = (a, b) => { let p = Array.from({ length: b.length + 1 }, (_, j) => j); for (let i = 1; i <= a.length; i++) { const r = [i]; for (let j = 1; j <= b.length; j++) r[j] = Math.min(p[j] + 1, r[j - 1] + 1, p[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)); p = r; } return p[b.length]; };
    for (const c of custom) {
      const k = nm(c.t); if (k.length < 3) continue;
      let best = (k.length >= 6 ? 2 : 1) + 1; for (const i of sung) { if (used.has(i) || taken.has(i)) continue; const e = lev(k, nm(src.tokens[i])); if (e < best) { best = e; c.near = i; } }   // a real typo: one letter off, two on longer words
      if (c.near != null) used.add(c.near);
    }
    let lead = null;                                                                 // words with no lyric before them: beside the lyric a typo is close to, else before the first lyric used
    for (const c of custom) { if (c.after != null) continue; if (c.near != null) lead = c.near; c.before = lead != null ? lead : picked.length ? picked[0] : null; }
    let sel = B.clearShotWords(d.text_select, n); picked.forEach((i) => { sel = B.toggleWordShot(sel, i, n); }); d.text_select = sel;
    x.custom = custom;
    const idx = d.shots.indexOf(x), lastShot = idx === d.shots.length - 1, text = words.join(" ");
    syncHitsOfShot(idx);
    if (text && d.hits) { const h = d.hits.find((q) => q.land >= x.start - 1e-6 && (q.land < x.end - 1e-6 || (lastShot && q.land <= x.end + 1e-6))); if (h) Object.assign(h, B.landHit(x, text, picked, src.times, storyCtx().beats)); d.hits.sort((a, b) => a.land - b.land); }
    st.edited.sel = true; st.selText = S.media.text;
    if (!quiet) paintFocus();
  }
  /** take one custom word off its shot (the frame's words follow). */
  function removeCustom(n, word) {
    const x = S.story.data.shots.find((s) => s.n === n), card = document.querySelector(`.sb-card[data-n="${n}"]`), line = card && card.querySelector(".sb-line");
    if (!x || !line) return;
    const bits = x.line.split(/(\s+)/), at = bits.indexOf(word);
    if (at >= 0) bits.splice(at, 2);
    line.value = bits.join("").trim(); line.dispatchEvent(new Event("input", { bubbles: true }));
    reconcileShot(n);
  }
  function textSource() {
    const m = S.media;
    const typedWords = B.splitTyped(m.text).words;                // the box text minus its direction clauses (palette / legibility / plain / instructions): only words are story candidates; the server splits the same way (user_style.split_typed)
    if (typedWords.trim()) return { tokens: B.textTokens(typedWords), times: null, lyrics: false, from: "" };
    const ly = m.read && m.read.lyrics;
    if (Array.isArray(ly) && ly.length) { const t = applyLyricFix(B.lyricTokens(ly), m); if (t.tokens.length) return { tokens: t.tokens, times: t.times, base: t.base, piece: t.piece, lyrics: true, from: videoItem() && soundtrackNow() !== "track" ? "video" : "audio" }; }
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
    if (b.classList.contains("sbw-btn") && b.dataset.act) return [`.sb-text .sbw-btn[data-act="${b.dataset.act}"]`];
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
    const nw = document.querySelector(".storyboard .sb-words"); if (nw) { nw.scrollTop = wy; if (nw.classList.contains("by-run")) nw.scrollLeft = stripX; }
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
  /** the picture a frame-board shot is previewed on and matted from: a video cut's first frame (once grabbed), or the shot's own dropped image. -> {url, key} or null. key names it for the matte cache. */
  function shotFrame(shot) {
    const v = shotVideo(shot);
    if (v) { const t = cutFrameT(shot), u = frameCached(v, t); return u ? { url: u, key: `${v.uid}:${t.toFixed(3)}` } : null; }
    const it = isFrameShot(shot) ? storyImage(shot) : null;
    return it ? { url: it.url, key: `img${it.uid}` } : null;
  }
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
    if (fr && fr.closest(".sb-grid.vid")) return fitVidType(ov);
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
  const frameRO = typeof ResizeObserver !== "undefined" ? new ResizeObserver((es) => es.forEach((e) => { const g = e.target.closest(".sb-grid.vid"); if (g && g.dataset.moving) return; const o = e.target.querySelector(".sb-ov"); if (o) fitOverlay(o); })) : null;
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", fitAllOverlays);

  /** video mode (2026-11-02 user: "full height to a maximum ... when you scroll your mouse over them they expand to show the full frame"):
   *  every frame has the SAME height (fills the work height, capped at the height whose full-frame width is COMFY = 320 px); each card's WIDTH is proportional to its shot's length
   *  (water-filled between MINW and the full-frame width so the row fills the work width exactly); the picture is a centre crop; hover / focus pins one card at the full frame. */
  // 2026-11-02 user: gaps between frames 2 px (cards have no side padding / border), a card shows number + frame only until it is OPEN (hovered, keyboard-focused or selected = pinned at the full frame)
  if (window.__MG_DEV) Object.assign(window.__MG_DEV, { S, rerender });               // dev proxy only (16_app/app_check/dev_proxy.mjs DEV_CLIP): live reload keeps the session
  const VID = { GAP: 2, PAD: 0, MINW: 28, COMFY: 320, STDH: 568, MINFH: 280, MINL: 52, LOCK: 240 };
  const vidPin = { hover: null, focus: null, skip: null, edit: null, wasEdit: false };
  let stripX = 0, stripShot = null, stripAnim = 0, stripDragged = false, stripJump = null;           // the single lyrics line: scroll position, the shot it last eased to, its animation, a drag just happened
  /** 2026-11-03 user: drag (or wheel) the lyrics line to see other words; a drag never counts as a word click. */
  function wireStrip(w) {
    let dr = null;
    w.addEventListener("wheel", (ev) => { const d = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY; if (!d) return; cancelAnimationFrame(stripAnim); w.scrollLeft += d; stripX = w.scrollLeft; ev.preventDefault(); }, { passive: false });
    w.addEventListener("pointerdown", (ev) => { if (ev.button) return; dr = { x: ev.clientX, left: w.scrollLeft, id: ev.pointerId, moved: false }; stripDragged = false; });
    w.addEventListener("pointermove", (ev) => {
      if (!dr || ev.pointerId !== dr.id) return;
      const dx = ev.clientX - dr.x;
      if (!dr.moved) { if (Math.abs(dx) < 4) return; dr.moved = stripDragged = true; cancelAnimationFrame(stripAnim); try { w.setPointerCapture(ev.pointerId); } catch (_) { /* no capture */ } w.classList.add("dragging"); }
      w.scrollLeft = dr.left - dx; stripX = w.scrollLeft;
    });
    const end = () => { if (!dr) return; dr = null; w.classList.remove("dragging"); setTimeout(() => { stripDragged = false; }, 0); };
    w.addEventListener("pointerup", end); w.addEventListener("pointercancel", end);
  }
  /** the open cards: the selected shot first, then the hovered one, then the keyboard-focused one (indices into the row). */
  function vidPinsFor(grid) {
    const out = [], add = (i) => { if (i != null && !out.includes(i)) out.push(i); }, act = S.story.active;
    if (act != null) { const c = grid.querySelector(`.sb-card[data-n="${act}"]`); if (c) add(Number(c.dataset.i)); }
    add(vidPin.edit); add(vidPin.hover); add(vidPin.focus);
    return out;
  }
  /** click the type on a frame: edit it right there. Typing feeds the card's line field, Enter finishes, Esc puts the old words back. */
  const ALIGN_ICON = { l: "M3 4h10M3 8h6M3 12h8", c: "M3 4h10M5 8h6M4 12h8", r: "M3 4h10M7 8h6M5 12h8", j: "M3 4h10M3 8h10M3 12h10" }, ALIGN_NAME = { l: "start edge", c: "centre", r: "end edge", j: "spread across" };
  const tlOf = (x) => x.tl || (x.tl = { rot: 0, align: "c", anch: false, cols: 0, x: 0.5, y: 0.5, s: 1, layer: "front" });      // the shot's type layout as the user set it on the frame
  const layerOf = (t) => { const l = (t && (t.layer || (t.behind ? "behind" : "front"))) || "front"; return l === "inverse" ? "mask" : l; };       // front | behind (matte) | mask (image through outlined letters on a solid colour) | maskm (the same with the matted subject behind the letters)
  const LAYERS = { front: ["In front of the subject", "M2 5.5L8 2.5l6 3-6 3-6-3zM2 10.5l6 3 6-3"], behind: ["Behind the subject", "M2 5.5L8 2.5l6 3-6 3-6-3zM2 10.5l6 3 6-3"], mask: ["Image through the letters on a solid colour", "M2.5 2.5h11v11h-11zM5 5.5h6M8 5.5v5.5"], maskm: ["Image through the letters, subject behind them", "M2.5 2.5h11v11h-11zM5 5h6M8 5v4M4.5 13c.6-2.2 6.4-2.2 7 0"], maskb: ["Image through the letters, letters behind the subject", "M2.5 2.5h11v11h-11zM5 4.5h6M8 4.5v3M4.5 13c.2-4 6.8-4 7 0"] };
  /** the solid colour of a masked shot: the strongest colour family of its frame (skin / wood hues only as a last resort), never a washed-out or peach default. */
  /** 2026-11-04: the type's fill comes from the shot's own picture, one of its strongest colours (shots take different ones, so a warm film does not repeat one peach), and the page shows and sends the colour the render uses. */
  function sampleInk(url, n) {
    return new Promise((res) => {
      const im = new Image();
      im.onerror = () => res(null);
      im.onload = () => {
        const W = 48, H = Math.max(1, Math.round(W * im.naturalHeight / im.naturalWidth)), c = document.createElement("canvas"); c.width = W; c.height = H;
        const a = c.getContext("2d", { willReadFrequently: true }); a.drawImage(im, 0, 0, W, H);
        const d = a.getImageData(0, 0, W, H).data, bins = new Array(12).fill(0);
        for (let i = 0; i < d.length; i += 4) {
          const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255, mx = Math.max(r, g, b), mn = Math.min(r, g, b), dl = mx - mn;
          if (dl < 0.1 || mx < 0.15) continue;
          const h = ((mx === r ? (g - b) / dl % 6 : mx === g ? (b - r) / dl + 2 : (r - g) / dl + 4) * 60 + 360) % 360;
          bins[Math.floor(h / 30) % 12] += dl * mx;
        }
        const top = bins.map((w, k) => [w, k]).filter((q) => q[0] > 0).sort((p, q) => q[0] - p[0]).slice(0, 3);
        const hue = top.length ? top[n % top.length][1] * 30 + 15 : 45, fill = hslHex(hue, 88, 66);
        res({ fill, ring: outlineFor(fill) });
      };
      im.src = url;
    });
  }
  function sampleSolid(url) {
    return new Promise((res) => {
      const im = new Image();
      im.onerror = () => res("#1c1a28");
      im.onload = () => {
        const W = 48, H = Math.max(1, Math.round(W * im.naturalHeight / im.naturalWidth)), c = document.createElement("canvas"); c.width = W; c.height = H;
        const a = c.getContext("2d", { willReadFrequently: true }); a.drawImage(im, 0, 0, W, H);
        const d = a.getImageData(0, 0, W, H).data, bins = Array.from({ length: 12 }, () => ({ w: 0, h: 0, s: 0 }));
        for (let i = 0; i < d.length; i += 4) {
          const r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255, mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, dl = mx - mn;
          if (dl < 0.08 || l < 0.1 || l > 0.85) continue;
          const s = dl / (1 - Math.abs(2 * l - 1)), h = ((mx === r ? (g - b) / dl % 6 : mx === g ? (b - r) / dl + 2 : (r - g) / dl + 4) * 60 + 360) % 360, k = bins[Math.floor(h / 30) % 12];
          k.w += s; k.h += h * s; k.s += s * s;
        }
        const skin = (k) => k === 0 || k === 1, order = bins.map((b, k) => [b, k]).filter(([b]) => b.w > 0).sort((p, q) => (skin(p[1]) - skin(q[1])) || q[0].w - p[0].w), top = order[0];
        if (!top) return res("#1c1a28");
        const hue = Math.round(top[0].h / top[0].w), sat = Math.round(Math.max(55, Math.min(90, 100 * top[0].s / top[0].w)));
        res(hslHex(hue, sat, 34));
      };
      im.src = url;
    });
  }
  const hslHex = (h, s, l) => { const f = (n) => { const k = (n + h / 30) % 12, A = (s / 100) * Math.min(l / 100, 1 - l / 100), v = l / 100 - A * Math.max(-1, Math.min(k - 3, 9 - k, 1)); return Math.round(255 * v).toString(16).padStart(2, "0"); }; return `#${f(0)}${f(8)}${f(4)}`; };
  /** the outline colour contrasts the solid colour and comes from its own family (not black). */
  const outlineFor = (hex) => { const n = parseInt(hex.slice(1), 16), r = n >> 16 & 255, g = n >> 8 & 255, b = n & 255, mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn, h = d ? ((mx === r ? (g - b) / d % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4) * 60 + 360) % 360 : 0, lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255; return lum < 0.45 ? hslHex(h, 85, 88) : hslHex(h, 75, 16); };
  /** the two copies of the type a masked shot needs (solid colour with the letters cut, outlines on top) follow the real type. */
  function syncGhosts(fr) {
    const ov = fr.querySelector(".sb-ov"); if (!ov) return;
    if (!document.getElementById("mg-outer")) document.body.append(el("div", { html: '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><filter id="mg-outer" x="-5%" y="-20%" width="110%" height="140%" color-interpolation-filters="sRGB"><feMorphology in="SourceGraphic" operator="dilate" radius="1" result="fat"/><feComposite in="fat" in2="SourceAlpha" operator="out"/></filter></svg>' }));   // the outline: the letters grown, minus the letters = their outer edge only (a stroke also draws where a font's shapes overlap inside a letter)
    fr.querySelectorAll(".sb-gh").forEach((g) => { g.innerHTML = ov.innerHTML; g.style.cssText = ov.style.cssText; g.hidden = ov.hidden; if (g.parentElement.classList.contains("sb-to")) g.style.filter = "url(#mg-outer)"; });
  }
  /** video cards: the type is laid out on the FULL frame (a collapsed card shows a true crop of it): size fits the full frame, then the user's position, rotation and scale. */
  function fitVidType(ov) {
    const fr = ov.parentElement, grid = fr.closest(".sb-grid.vid"), g = grid && grid._vid, card = fr.closest(".sb-card"), x = card && S.story.data && S.story.data.shots[Number(card.dataset.i)];
    if (!g || !x || ov.hidden) return;
    const fw = g.fullW || fr.clientWidth, fh = fr.clientHeight, t = x.tl || {}, deg = t.rot || 0, rot = deg % 180 !== 0, al = t.align || "c";
    if (!fw || !fh) return;
    const jl = ov.querySelectorAll(":scope > .jl"); if (jl.length) ov.textContent = [...jl].map((e) => e.textContent).join("\n");     // back to plain lines to measure
    ov.style.width = ""; ov.style.textAlignLast = ""; ov.style.justifyContent = ""; ov.style.whiteSpace = "";
    if (!ov.classList.contains("grid") && !ov.isContentEditable) ov.textContent = ov.textContent.split("\n").map((s) => s.trim()).join("\n");   // every alignment keeps new lines and the spaces between words; spaces at a line's edges don't count, so lines sit flat to the edge
    ov.style.textAlign = { l: "left", c: "center", r: "right", j: "center" }[al]; ov.style.justifyItems = { l: "start", c: "center", r: "end", j: "center" }[al];
    ov.style.transform = "none"; ov.style.fontSize = "100px";
    if (ov.classList.contains("grid")) {
      ov.style.setProperty("--cell", "auto"); ov.style.setProperty("--gj", { l: "flex-start", c: "center", r: "flex-end", j: "space-between" }[al]);
      ov.style.setProperty("--cell", Math.max(1, ...[...ov.querySelectorAll(".gr > span")].map((c) => c.offsetWidth)) / 100 + "em");   // every cell as wide as the widest letter
    }
    const w = ov.offsetWidth, h = ov.offsetHeight;
    const fs = w && h ? Math.min(fw * 0.26, 100 * Math.min(0.88 * fw / (rot ? h : w), 0.84 * fh / (rot ? w : h))) : fw * 0.15;   // empty (just deleted, about to type): a sensible size at its position
    ov.style.fontSize = fs + "px";
    if (al === "j" && w && h && !ov.classList.contains("grid")) {                  // spread: every line of the stack runs the full width of the frame (along the reading direction), letters spaced out to fill it. A grid keeps its own width: --gj space-between spreads a short row across the grid
      const W = Math.max(w * fs / 100, 0.88 * (rot ? fh : fw) / (t.s || 1));
      ov.style.width = W + "px";
      if (ov.isContentEditable) { ov.style.whiteSpace = "pre-wrap"; ov.style.textAlign = "justify"; ov.style.textAlignLast = "justify"; }   // while typing: words spread (per-letter spacing returns when the edit ends)
      else {
        ov.replaceChildren(...ov.textContent.split("\n").map((s) => el("span", { class: "jl" }, s)));
        ov.querySelectorAll(":scope > .jl").forEach((sp) => { const n = [...sp.textContent].length, ls = n > 1 ? (W - sp.offsetWidth) / (n - 1) : 0; sp.style.letterSpacing = ls + "px"; sp.style.marginRight = -ls + "px"; });
      }
    }
    if (t.anch && !ov.classList.contains("grid")) {                                // a grid stays put: alignment only moves its leftover letters. Otherwise pre-aligned: the block sits against the frame edge its reading direction starts (left) or ends (right) at, whatever the rotation
      const L = w * (fs / 100) * (t.s || 1), span = rot ? fh : fw, m = fw * 0.06, flip = deg === 180 || deg === 270;
      const pos = al === "c" || al === "j" ? span / 2 : ((al === "l") !== flip) ? m + L / 2 : span - m - L / 2;
      if (rot) t.y = pos / fh; else t.x = pos / fw;
    }
    ov.style.left = `calc(50% + ${((t.x == null ? 0.5 : t.x) - 0.5) * fw}px)`; ov.style.top = `${(t.y == null ? 0.5 : t.y) * 100}%`;
    ov.style.transform = `translate(-50%, -50%) rotate(${deg}deg) scale(${t.s || 1})`;
    if (ov._edit) drawCaret(ov);
    if (fr.closest(".sb-img.tmk")) syncGhosts(fr.closest(".sb-img"));
  }
  /** type between background and subject: the shot's frame again above the type, cut out by its matte (dev: the proxy's stand-in mattes; 404 = no matte for this frame yet). */
  function paintFg(card, x) {
    const fr = card.querySelector(".sb-img"), old = card.querySelector(".sb-fg"), sf = shotFrame(x), url = sf && sf.url;
    if (!["behind", "maskm", "maskb"].includes(layerOf(x.tl)) || !url) { if (old) old.remove(); return; }
    if (old && old.dataset.src === `${matteEngine}:${url}`) return;
    const eng = matteEngine, load = (u) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = u; });
    Promise.all([load(url), frameMatte(x).then(load)]).then(([f, m]) => {
      if (!["behind", "maskm", "maskb"].includes(layerOf(x.tl)) || !card.isConnected || eng !== matteEngine) return;
      const c = document.createElement("canvas"); c.width = f.naturalWidth; c.height = f.naturalHeight; c.className = "sb-fg"; c.dataset.src = `${eng}:${url}`;
      const a = c.getContext("2d", { willReadFrequently: true }); a.drawImage(m, 0, 0, c.width, c.height);
      const mp = a.getImageData(0, 0, c.width, c.height); a.drawImage(f, 0, 0, c.width, c.height);
      const fp = a.getImageData(0, 0, c.width, c.height); for (let i = 0; i < fp.data.length; i += 4) fp.data[i + 3] = mp.data[i];
      a.putImageData(fp, 0, 0);
      const cur = card.querySelector(".sb-fg"); if (cur) cur.replaceWith(c); else fr.append(c);
    }, () => { /* no subject matte for this frame: the look still shows, just without the subject (the background through the letters, the type over the picture); the render makes its own */ });
  }
  /** 2026-11-03 user: try MediaPipe selfie segmentation for the storyboard's mattes: it runs in this browser (no box, no upload), people only, rough edges; a preview, the render makes its own with Alpha-Gen. Only shots with words ask; the key is the frame, so changing the words never redoes it. */
  const matteEngine = "modnet";                                                       // 2026-11-03 user: a frame's own saved matte when there is one (savedMatte), else MODNet in this browser
  const grayMask = (w, h, valueAt) => { const c = document.createElement("canvas"); c.width = w; c.height = h; const a = c.getContext("2d"), d = a.createImageData(w, h); for (let i = 0; i < w * h; i++) { const g = valueAt(i), j = i * 4; d.data[j] = d.data[j + 1] = d.data[j + 2] = g; d.data[j + 3] = 255; } a.putImageData(d, 0, 0); return c.toDataURL("image/png"); };
  const MATTE_LOAD = {
    modnet: async () => {
      const T = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2"), p = await T.pipeline("background-removal", "Xenova/modnet", { device: navigator.gpu ? "webgpu" : "wasm", dtype: "fp32" });
      return async (im) => { const c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight; c.getContext("2d").drawImage(im, 0, 0); const r = await p(await T.RawImage.fromCanvas(c)), o = Array.isArray(r) ? r[0] : r; return grayMask(o.width, o.height, (i) => o.data[i * o.channels + o.channels - 1]); };
    },
  };
  const matteRun = {}; let matteQueue = Promise.resolve();
  const matteLoaded = (e) => matteRun[e] || (matteRun[e] = MATTE_LOAD[e]().catch((err) => { delete matteRun[e]; throw err; }));
  const frameMatteDone = new Map();
  function frameMatte(x) { return savedMatte(x).then((u) => u || browserMatte(x)); }
  function browserMatte(x) {
    const sf = shotFrame(x), src = sf && sf.url, e = matteEngine, key = src ? `${e}:${sf.key}` : "";
    if (!src) return Promise.reject(new Error("the frame is not loaded yet"));
    if (frameMatteDone.has(key)) return frameMatteDone.get(key);
    const p = (matteQueue = matteQueue.catch(() => {}).then(async () => {                 // one at a time
      const run = await matteLoaded(e), im = new Image();
      await new Promise((res, rej) => { im.onload = res; im.onerror = rej; im.src = src; });
      return run(im);
    }));
    frameMatteDone.set(key, p); p.then(() => { const card = document.querySelector(`.storyboard .sb-card[data-n="${x.n}"]`); if (card && matteEngine === e) paintFg(card, x); }, (err) => { frameMatteDone.delete(key); toast(`MODNet: ${String(err.message || err).slice(0, 120)}`, 4000, { info: true }); });
    return p;
  }
  /** the piece keeps the video's own length: whatever window the read suggested, the job gets the whole clip. */
  function wholeWindow(brief) {
    const dur = videoDur(), v = Array.isArray(brief.media) && brief.media.find((m) => m && m.kind === "video");
    if (v && Number.isFinite(dur)) v.window = [0, Math.ceil(dur * 100) / 100];
    return brief;
  }
  /** a matte already made for this frame (the full run's): its url, or null when there is none. Only looks, never starts a render (peek). */
  const savedAsked = new Map();
  function savedMatte(x) {
    if (!shotVideo(x)) return Promise.resolve(null);                              // an image has no saved matte (the full run makes its own): MODNet on the picture itself
    const v = shotVideo(x), url = `/mograph/own_video/matte_frame?video=${encodeURIComponent((v && v.uploaded) || "")}&t=${cutFrameT(x).toFixed(3)}&end=${Number(x.src_end).toFixed(3)}&peek=1`;
    if (!savedAsked.has(url)) savedAsked.set(url, fetch(url, { cache: "no-store" }).then((r) => (r.status === 200 ? r.blob().then((b) => URL.createObjectURL(b)) : null)).catch(() => null));
    return savedAsked.get(url);
  }
  function syncTools(card, x) {
    const t = x.tl || {}, on = (a, v) => { const b = card.querySelector(`.sb-tool[data-act="${a}"]`); if (b) b.classList.toggle("on", !!v); };
    const lay = layerOf(t), lb = card.querySelector('.sb-tool[data-act="behind"]'), tl = card.querySelector(".sb-tl"), hasText = !!x.line.trim();
    on("behind", lay !== "front"); on("align", t.anch);
    if (lb) { lb.title = `${LAYERS[lay][0]}. Click for the next: in front, behind the subject, image through the letters, image around the letters`; }
    const masked = lay === "mask" || lay === "maskm" || lay === "maskb", fr = card.querySelector(".sb-img"), bg = t.bg || "#1c1a28", cb = card.querySelector(".sb-tool-color");
    if (tl) {
      fr.classList.toggle("tmk", hasText && masked); fr.classList.toggle("tbh", lay === "maskb"); fr.style.setProperty("--mbg", bg); fr.style.setProperty("--olc", outlineFor(bg));
      if (masked && !t.bg && !t.bgBusy) { const u = shotPicUrl(x); if (u) { t.bgBusy = true; sampleSolid(u).then((c) => { delete t.bgBusy; t.bg = c; t.bgA = true; syncTools(card, x); }); } }
      if (!masked && hasText && !t.color && !t.colorBusy) { const all = S.story.data.shots; let hd = all.indexOf(x); while (hd > 0 && all[hd - 1].line === x.line && all[hd - 1].font === x.font) hd--; const u = shotPicUrl(all[hd] || x); if (u) { const tt = tlOf(x); tt.colorBusy = true; sampleInk(u, Number((all[hd] || x).n) || 0).then((c) => { delete tt.colorBusy; if (c) { tt.color = c.fill; tt.outline = c.ring; tt.colorA = true; syncTools(card, x); } }); } }
      { const all = S.story.data.shots; let hd = all.indexOf(x); while (hd > 0 && all[hd - 1].line === x.line && all[hd - 1].font === x.font) hd--; const h = all[hd] && all[hd].tl; if (hd >= 0 && all[hd] !== x && h && h.color && t.colorA !== false && t.color !== h.color && (!t.color || t.colorA)) { t.color = h.color; t.outline = h.outline; t.colorA = true; } }   // a word held across cuts keeps its first shot's colour
      const ovc = card.querySelector(".sb-ov");
      if (ovc && !masked && t.color) { ovc.style.color = t.color; ovc.style.webkitTextStroke = `${(fr.clientWidth * 0.0056).toFixed(1)}px ${t.outline || "transparent"}`; ovc.style.paintOrder = "stroke fill"; } else if (ovc) ovc.style.webkitTextStroke = "";
      syncGhosts(fr);
    }
    if (cb) { cb.style.visibility = masked ? "" : "hidden"; cb.querySelector("input").value = bg; cb.querySelector("span").style.background = bg; }
    const ab = card.querySelector('.sb-tool[data-act="align"]'), cur = t.anch ? t.align : "c";
    if (ab) { ab.querySelector("svg").style.transform = `rotate(${t.rot || 0}deg)`; /* the icon turns with the type, so it shows the edge it will align to */ ab.querySelector("path").setAttribute("d", ALIGN_ICON[cur]); ab.title = `Align: ${ALIGN_NAME[cur]}${t.anch ? "" : " (not set)"}. Click for the next: start edge, centre, end edge, spread`; } on("stack", /\n/.test(x.line)); on("grid", t.cols);
  }
  /** the icons at the bottom of the selected frame + dragging the type to move it, wheel over it to scale. */
  function buildTools(getCard, x, line, ov) {
    const feed = () => line.dispatchEvent(new Event("input", { bubbles: true }));
    const ic = (d) => `<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;
    const tool = (act, title, d, fn) => el("button", { class: "sb-tool", type: "button", "data-act": act, title, "aria-label": title, html: ic(d), onclick: (ev) => { ev.stopPropagation(); fn(); const c = getCard(); paintCard(c, x); } });
    const scale = (k) => { const t = tlOf(x); t.s = Math.max(0.2, Math.min(4, (t.s || 1) * k)); };
    ov.addEventListener("wheel", (ev) => { if (!getCard().classList.contains("active")) return; ev.preventDefault(); scale(ev.deltaY < 0 ? 1.06 : 1 / 1.06); fitVidType(ov); }, { passive: false });
    let dr = null;
    ov.addEventListener("pointerdown", (ev) => { if (ov._edit) { ev.preventDefault(); ev.stopPropagation(); return; } if (ev.button || ov.isContentEditable || !getCard().classList.contains("active")) return; ev.stopPropagation(); dr = { px: ev.clientX, py: ev.clientY, x: tlOf(x).x, y: tlOf(x).y, moved: false, id: ev.pointerId }; ov.setPointerCapture(ev.pointerId); });
    ov.addEventListener("pointermove", (ev) => {
      if (!dr || ev.pointerId !== dr.id) return;
      const dx = ev.clientX - dr.px, dy = ev.clientY - dr.py;
      if (!dr.moved && Math.hypot(dx, dy) < 3) return;
      dr.moved = true;
      const fr = ov.parentElement, g = fr.closest(".sb-grid.vid")._vid, t = tlOf(x);
      if (t.anch) { t.anch = false; syncTools(getCard(), x); }
      t.x = dr.x + dx / (g.fullW || fr.clientWidth); t.y = dr.y + dy / fr.clientHeight;   // free: it may leave the frame (reset brings it back, delete removes it)
      fitVidType(ov);
    });
    ov.addEventListener("pointerup", () => { if (dr && dr.moved) ov.dataset.dragged = "1"; dr = null; });
    ov.addEventListener("click", (ev) => { if (ov.dataset.dragged) { delete ov.dataset.dragged; ev.stopImmediatePropagation(); } }, true);
    return el("div", { class: "sb-tools" },
      tool("edit", "Edit the words (Enter = new line, click away to finish)", "M3 13l.6-2.6L10.8 3.2l2 2L5.6 12.4 3 13z", () => { setTimeout(() => startOvEdit(ov, line, x), 0); }),
      tool("font", "Font: hover or scroll the list to try them, click to keep, Esc to cancel", "M3 5V3h10v2M8 3v10M6 13h4", () => openFontMenu(getCard, x)),
      tool("behind", LAYERS.front[0], LAYERS.front[1], () => { const t = tlOf(x), order = ["front", "behind", "maskm", "maskb", "mask"]; delete t.noMatte; t.layer = order[(order.indexOf(layerOf(t)) + 1) % order.length]; delete t.behind; }),
      (() => {                                                                      // the solid colour of a masked shot (starts from the shot's own colour)
        const inp = el("input", { type: "color", "aria-label": "Background colour" }), lab = el("label", { class: "sb-tool sb-tool-color", title: "Background colour" }, el("span", { class: "sb-swatch" }), inp);
        inp.addEventListener("input", () => { const t = tlOf(x); t.bg = inp.value; delete t.bgA; syncTools(getCard(), x); });
        lab.addEventListener("click", (ev) => ev.stopPropagation());
        return lab;
      })(),
      tool("rotate", "Rotate 90 degrees clockwise (keep clicking to go round)", "M13 8a5 5 0 1 1-1.5-3.5L13 6M13 2.5V6H9.5", () => { const t = tlOf(x), was = (t.rot || 0) % 180 !== 0; t.rot = ((t.rot || 0) + 90) % 360; if (t.anch) { if (was) t.y = 0.5; else t.x = 0.5; } }),
      tool("align", "Align", ALIGN_ICON.c, () => { const t = tlOf(x), order = ["l", "c", "r", "j"]; t.align = t.anch ? order[(order.indexOf(t.align) + 1) % 4] : "l"; t.anch = true; }),
      tool("grid", "Letters in a grid: 2 across, then wider until it is square, then off", "M3 3h4v4H3zM9 3h4v4H9zM3 9h4v4H3zM9 9h4v4H9z", () => { const t = tlOf(x), n = [...String(x.line).replace(/\s+/g, "")].length, square = Math.max(2, Math.ceil(Math.sqrt(n))); t.cols = t.cols >= square ? 0 : Math.max(2, (t.cols || 1) + 1); }),   /* the last step is the first width with as many across as down */
      tool("smaller", "Smaller (or wheel over the type)", "M4 8h8", () => scale(1 / 1.12)),
      tool("bigger", "Bigger (or wheel over the type)", "M4 8h8M8 4v8", () => scale(1.12)),
      tool("reset", "Back to the centre, full size, not rotated", "M8 2v3M8 11v3M2 8h3M11 8h3M8 6.5v3M6.5 8h3", () => { const t = tlOf(x); t.x = t.y = 0.5; t.s = 1; t.rot = 0; t.align = "c"; t.anch = false; t.cols = 0; }),
      tool("delete", "Delete the words from this shot", "M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.5h5.6l.7-8.5M7 7v3.5M9 7v3.5", () => {
        const t = tlOf(x), d = S.story.data; t.x = t.y = 0.5; t.s = 1; t.rot = 0; t.align = "c"; t.anch = false; t.cols = 0;
        line.value = ""; feed();
        if (d) reconcileShot(x.n);
      }));
  }
  /** the font list of the selected frame: hover, scroll or arrow keys show each font on the frame at once; click or Enter keeps it; Esc or a click elsewhere puts the old one back. */
  let closeFontMenu = null;
  function openFontMenu(getCard, x) {
    if (closeFontMenu) { closeFontMenu(false); return; }
    const list = S.fonts.list, btn = getCard().querySelector('.sb-tool[data-act="font"]');
    if (!list.length || !btn) return;
    const orig = x.font, menu = el("div", { class: "sb-fontmenu", role: "listbox", "aria-label": "Fonts" }), items = [];
    let hi = -1;
    const show = (k, scroll) => { if (k === hi) return; if (items[hi]) items[hi].classList.remove("hi"); hi = k; items[k].classList.add("hi"); if (scroll) items[k].scrollIntoView({ block: "nearest" }); x.font = list[k].file; paintCard(getCard(), x); };
    list.forEach((f, k) => {
      ensureFontFace(f);
      const it = el("div", { class: "sb-fopt" + (f.file === orig ? " cur" : ""), role: "option", style: `font-family:"${B.fontCss(f)}", system-ui, sans-serif` }, f.name);
      it.addEventListener("mouseover", () => show(k));
      it.addEventListener("click", () => { x.font = orig; close(true, f.file); });
      items.push(it); menu.append(it);
    });
    const key = (ev) => {
      if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); close(false); }
      else if (ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); if (hi >= 0) { const f = list[hi].file; x.font = orig; close(true, f); } else close(false); }
      else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") { ev.preventDefault(); ev.stopPropagation(); const base = hi >= 0 ? hi : Math.max(0, list.findIndex((f) => f.file === orig)); show((base + (ev.key === "ArrowDown" ? 1 : list.length - 1)) % list.length, true); }
    };
    const away = (ev) => { if (!menu.contains(ev.target) && !btn.contains(ev.target)) close(false); };
    function close(keep, file) {
      document.removeEventListener("keydown", key, true); document.removeEventListener("pointerdown", away, true); menu.remove(); closeFontMenu = null; btn.classList.remove("on");
      if (keep) setFont(x, file); else { x.font = orig; paintCard(getCard(), x); }
    }
    closeFontMenu = close;
    document.body.append(menu); btn.classList.add("on");
    const r = getCard().querySelector(".sb-img").getBoundingClientRect(), mw = menu.offsetWidth;     // beside the frame, never over the type being tried
    menu.style.left = (r.right + 6 + mw <= innerWidth - 8 ? r.right + 6 : Math.max(8, r.left - 6 - mw)) + "px";
    menu.style.top = r.top + "px"; menu.style.maxHeight = r.height + "px";
    const at = list.findIndex((f) => f.file === orig); if (items[at]) items[at].scrollIntoView({ block: "center" });
    document.addEventListener("keydown", key, true); document.addEventListener("pointerdown", away, true);
  }
  /** grid and spread can't live inside editable text (equal cells, per-letter spacing), so while editing them the real layout stays on screen, an invisible field takes the typing and the caret is drawn on the letters. */
  function shapedIndexAt(ov, ev) {
    const v = ov._edit.ta.value, px = ev.clientX, py = ev.clientY, rad = (ov._edit.deg || 0) * Math.PI / 180, dx = Math.round(Math.cos(rad)), dy = Math.round(Math.sin(rad));   // measured from the point, not ev.target: the drag handler's pointer capture makes the target the whole type
    const dist = (b) => Math.hypot(Math.max(b.left - px, 0, px - b.right), Math.max(b.top - py, 0, py - b.bottom));   // 0 inside the box
    const past = (cx, cy) => (px - cx) * dx + (py - cy) * dy > 0;                  // beyond this point along the way the text reads, whatever the rotation
    const cells = [...ov.querySelectorAll(".gr > span")];
    if (cells.length) {
      let best = 0, bd = Infinity; cells.forEach((c, i) => { const d = dist(c.getBoundingClientRect()); if (d < bd) { bd = d; best = i; } });
      const r = cells[best].getBoundingClientRect(), want = best + (past(r.left + r.width / 2, r.top + r.height / 2) ? 1 : 0);
      let n = 0; for (let i = 0; i < v.length; i++) { if (n === want) return i; if (!/\s/.test(v[i])) n++; }
      return v.length;
    }
    ov.querySelectorAll(".ocaret").forEach((c) => c.remove());                     // measure the plain text: the caret is drawn again by the caller
    const jl = [...ov.querySelectorAll(":scope > .jl")]; if (!jl.length) return v.length;
    const box = ov.getBoundingClientRect(), s = Math.max(box.width, box.height) / Math.max(ov.offsetWidth, ov.offsetHeight, 1), rg = document.createRange();
    let at = 0, want = 0, bd = Infinity;
    jl.forEach((row, ri) => {
      row.normalize(); const tn = row.firstChild;
      if (!tn || tn.nodeType !== 3) { const d = dist(row.getBoundingClientRect()); if (d < bd) { bd = d; at = ri; want = 0; } return; }
      const ls = (parseFloat(row.style.letterSpacing) || 0) * s;                   // each letter's box also holds the spacing after it: halve the letter itself, not the box
      for (let c = 0; c < tn.length; c++) {
        rg.setStart(tn, c); rg.setEnd(tn, c + 1); const b = rg.getBoundingClientRect(), d = dist(b);
        if (d < bd) { bd = d; at = ri; want = c + (past(b.left + b.width / 2 - dx * ls / 2, b.top + b.height / 2 - dy * ls / 2) ? 1 : 0); }
      }
    });
    const rows = v.split("\n"), row = rows[at] || "", lead = row.length - row.trimStart().length;   // the display leaves out the spaces at the line's edges
    return rows.slice(0, at).reduce((n, t) => n + t.length + 1, 0) + lead + Math.min(want, row.trim().length);
  }
  function drawCaret(ov) {
    ov.querySelectorAll(".ocaret").forEach((c) => c.remove());
    ov.querySelectorAll(":scope > .jl").forEach((r) => r.normalize());             // the last caret split the text: join it again before placing this one
    const e = ov._edit; if (!e || document.activeElement !== e.ta) return;
    const before = e.ta.value.slice(0, e.ta.selectionStart), mk = el("span", { class: "ocaret" }), cells = ov.querySelectorAll(".gr > span"), rows = ov.querySelectorAll(":scope > .jl");
    if (cells.length) { const n = before.replace(/\s+/g, "").length; if (n < cells.length) cells[n].prepend(mk); else cells[cells.length - 1].append(mk); return; }
    if (rows.length) {
      const k = before.split("\n"), row = rows[Math.min(k.length - 1, rows.length - 1)], tn = row.firstChild, col = k[k.length - 1].trimStart().length;
      if (tn && tn.nodeType === 3 && col < tn.length) row.insertBefore(mk, tn.splitText(col)); else { row.append(mk); mk.style.left = -(parseFloat(row.style.letterSpacing) || 0) + "px"; }   // at the line's end: against the last letter, not after its spacing
      return;
    }
    ov.append(mk);
  }
  function startShapedEdit(ov, line, x, ev) {
    const old = line.value, fr = ov.closest(".sb-img"), ta = el("textarea", { class: "sb-oe", spellcheck: "false", "aria-label": `Shot ${x.n} words` }), ac = new AbortController(), on = { signal: ac.signal }, feed = () => line.dispatchEvent(new Event("input", { bubbles: true }));
    ta.value = old; fr.append(ta); ov.classList.add("editing"); ov._edit = { ta, deg: (x.tl && x.tl.rot) || 0 };
    ta.focus({ preventScroll: true });
    const at = ev ? shapedIndexAt(ov, ev) : old.length; ta.setSelectionRange(at, at);
    const caret = () => { drawCaret(ov); syncGhosts(fr); };
    ta.addEventListener("input", () => { line.value = ta.value; feed(); caret(); }, on);
    ["keyup", "select", "click"].forEach((n) => ta.addEventListener(n, caret, on));
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && x.tl && x.tl.cols && !(e.ctrlKey || e.metaKey)) e.preventDefault(); else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ta.blur(); } else if (e.key === "Escape") { e.stopPropagation(); line.value = old; ta.value = old; ta.blur(); } }, on);
    ta.addEventListener("blur", () => { ac.abort(); ov._edit = null; ta.remove(); ov.classList.remove("editing"); ov.querySelectorAll(".ocaret").forEach((c) => c.remove()); feed(); if (line.value !== old) reconcileShot(x.n); }, on);
    caret();
  }
  function startOvEdit(ov, line, x, ev) {
    if (ov.isContentEditable) return;
    if (ov._edit) { if (ev) { const i = shapedIndexAt(ov, ev); ov._edit.ta.setSelectionRange(i, i); drawCaret(ov); syncGhosts(ov.closest(".sb-img")); } return; }
    if (x && x.tl && (x.tl.cols || x.tl.align === "j") && line.value.trim()) return startShapedEdit(ov, line, x, ev);
    const cols = (x && x.tl && x.tl.cols) || 0, old = line.value, letters = old.replace(/\s+/g, ""), shown = cols ? (letters.match(new RegExp(`.{1,${cols}}`, "gu")) || []).join("\n") : old;   // edit it as it looks: a grid edits as its rows
    const ac = new AbortController(), on = { signal: ac.signal }, feed = () => line.dispatchEvent(new Event("input", { bubbles: true }));
    ov.classList.remove("grid"); ov.hidden = false;
    const stand = !old.trim();                                                       // an empty shot: a stand-in word shows where the text will go; it is never saved and goes the moment you type
    ov.textContent = stand ? "YOUR WORDS" : shown; ov.classList.toggle("ph", stand); ov.contentEditable = "plaintext-only"; ov.spellcheck = false; ov.classList.add("editing"); fitVidType(ov); ov.focus();
    let standing = stand;
    if (stand) ov.addEventListener("beforeinput", () => { if (!standing) return; standing = false; ov.classList.remove("ph"); ov.textContent = ""; }, on);
    const s = getSelection(), hit = ev && document.caretRangeFromPoint ? document.caretRangeFromPoint(ev.clientX, ev.clientY) : null;   // the caret goes where you clicked, nothing is selected, so the type looks exactly as it did
    if (hit && !stand && ov.contains(hit.startContainer)) { s.removeAllRanges(); s.addRange(hit); } else { const r = document.createRange(); r.selectNodeContents(ov); r.collapse(false); s.removeAllRanges(); s.addRange(r); }
    ov.addEventListener("input", () => { if (cols) { const flat = ov.textContent.replace(/\n/g, ""); line.value = flat.replace(/\s+/g, "") === letters ? old : flat; } else line.value = ov.textContent; feed(); }, on);
    ov.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); ov.blur(); } else if (ev.key === "Escape") { ev.stopPropagation(); line.value = old; ov.blur(); } }, on);   // plain Enter = a new line, anywhere (between words or mid-word)
    ov.addEventListener("blur", () => { ac.abort(); ov.removeAttribute("contenteditable"); ov.classList.remove("editing", "ph"); if (standing) ov.textContent = ""; feed(); if (x && line.value !== old) reconcileShot(x.n); }, on);
  }
  function waterFill(w, total, lo, hi) {                              // widths proportional to w[], each clamped to [lo, hi], summing to total
    const n = w.length, out = new Array(n).fill(0);
    if (!n) return out;
    if (n * lo >= total) return out.fill(lo);                         // even all-minimum overflows: the row scrolls sideways
    if (n * hi <= total) return out.fill(hi);
    const fixed = new Array(n).fill(false), tent = new Array(n).fill(0);
    for (let it = 0; it <= n; it++) {
      let rem = total, sw = 0;
      for (let i = 0; i < n; i++) { if (fixed[i]) rem -= out[i]; else sw += w[i]; }
      let delta = 0;
      for (let i = 0; i < n; i++) if (!fixed[i]) { tent[i] = sw > 0 ? (w[i] / sw) * rem : rem / n; out[i] = Math.min(hi, Math.max(lo, tent[i])); delta += out[i] - tent[i]; }
      if (Math.abs(delta) < 1e-6) break;
      for (let i = 0; i < n; i++) if (!fixed[i] && (delta > 0 ? tent[i] < lo : tent[i] > hi)) fixed[i] = true;
    }
    return out;
  }
  function vidApply(grid, g, pins, animate) {                         // g = { durs, total, fullW }; pins = indices of the OPEN cards (each at the full frame, the rest water-filled into what remains; if that cannot fit the row scrolls); animate: overlays are refitted once the columns have settled
    const n = g.durs.length, MINW = VID.MINW, P = (pins || []).filter((i, k, a) => i >= 0 && i < n && a.indexOf(i) === k);
    let ws;
    if (P.length >= n) ws = new Array(n).fill(g.fullW);
    else if (P.length) {
      const room = g.total - P.length * g.fullW, lo = Math.max(8, Math.min(MINW, room / (n - P.length)));   // many cuts: the closed cards get narrower instead of the row scrolling (a scrollbar would also push the lyrics down)
      const rest = waterFill(g.durs.filter((_, i) => !P.includes(i)), room, lo, g.fullW);
      let r = 0; ws = g.durs.map((_, i) => (P.includes(i) ? g.fullW : rest[r++]));
      if (P.length === 2) {                                            // 2026-11-03 user: a second frame opens: every closed frame keeps the width it has with one open. Two full frames can be wider than the row, so the row runs past the edge and slides (see below)
        const oroom = g.total - g.fullW, oneLo = Math.max(8, Math.min(MINW, oroom / (n - 1))), one = waterFill(g.durs.filter((_, i) => i !== P[0]), oroom, oneLo, g.fullW);
        ws = g.durs.map((_, i) => (P.includes(i) ? g.fullW : one[i > P[0] ? i - 1 : i]));
      }
    } else ws = waterFill(g.durs, g.total, MINW, g.fullW);
    ws = ws.map((v) => Math.floor(v * 100) / 100);
    const next = ws.map((v) => v + "px").join(" ");
    if (animate && next !== grid.style.gridTemplateColumns) { grid.dataset.moving = "1"; clearTimeout(grid._mvT); grid._mvT = setTimeout(() => { delete grid.dataset.moving; fitAllOverlays(); }, 240); }
    grid.style.gridTemplateColumns = next;
    clearTimeout(grid._slideT);                                        // the frame just opened comes into view, the row slides back when it closes (no scrollbar, css)
    grid._slideT = setTimeout(() => {
      const k = P.length >= 2 ? P[P.length - 1] : -1, x0 = k >= 0 ? ws.slice(0, k).reduce((a, v) => a + v + VID.GAP, 0) : 0;
      const left = k < 0 ? 0 : (x0 < grid.scrollLeft ? x0 : x0 + ws[k] > grid.scrollLeft + grid.clientWidth ? x0 + ws[k] - grid.clientWidth : grid.scrollLeft);
      if (Math.abs(left - grid.scrollLeft) > 1) grid.scrollTo({ left, behavior: "smooth" });
    }, animate ? 190 : 0);
    Array.from(grid.children).forEach((c, i) => c.classList.toggle("open", P.includes(i)));        // the details (Shot n, times, font picker, words) show only on an open card
  }
  /** video board layout (2026-11-02 user: shots at the top and taller, nothing bumps): [cards][lyrics section][head]. Every text row of a card has a fixed height (css), so the card chrome is one number;
   *  frame height = visible storyboard height - chrome - the lyrics section's height - gaps, capped at the 320 px-wide full frame. The section keeps its natural height (strips) and only scrolls inside
   *  itself when giving it that much would push the frame under MINFH. */
  function fitVid(sb, grid, ratio) {
    const n = grid.children.length, V = VID, d = S.story.data, panel = sb.querySelector(".sb-text");
    grid.classList.add("nofx");                                       // no width transition while measuring
    const durs = Array.from(grid.children).map((c) => { const s = d && d.shots[Number(c.dataset.i)]; return s ? Math.max(0.05, Number(s.end) - Number(s.start)) : 1; });
    const gcs = getComputedStyle(grid), px = (v) => parseFloat(v) || 0;
    const g = grid._vid = { durs, total: grid.clientWidth - px(gcs.paddingLeft) - px(gcs.paddingRight) - V.GAP * (n - 1), fullW: 0 };
    const capH = Math.min(V.STDH, Math.floor(g.total * 0.75 * ratio)), minH = 120;   // 2026-11-03 user: a standard size: every clip's frames get the vertical frames' height (not a width cap, which left landscape at 180 px); a wide open frame leaves a quarter of the row to the others
    const setH = (h) => { grid.style.setProperty("--fh", h + "px"); g.fullW = Math.floor(h / ratio) + V.PAD; vidApply(grid, g, null); };
    if (panel) panel.style.height = "";
    setH(capH);
    const c0 = grid.firstElementChild, chrome = c0.offsetHeight - c0.querySelector(".sb-img").offsetHeight;
    const top = grid.getBoundingClientRect().top - sb.getBoundingClientRect().top + sb.scrollTop;
    const lastPad = px(gcs.paddingBottom), tmt = panel ? px(getComputedStyle(panel).marginTop) : 0;
    const base = sb.clientHeight - top + (panel ? panel.offsetHeight : 0) - chrome - lastPad - tmt - 8;   // height left for the frame + the lyrics section (the section sits above the grid, so `top` already counts it)
    let L = panel ? panel.offsetHeight : 0, fh = Math.min(capH, base - L);
    if (panel && fh < V.MINFH) { L = Math.max(V.MINL, Math.min(L, base - V.MINFH)); fh = Math.min(capH, base - L); }
    fh = Math.max(minH, Math.floor(fh));
    if (panel) panel.style.height = L + "px";                         // fixed from here on: toggling a word never changes a frame height
    setH(fh);
    vidApply(grid, g, vidPinsFor(grid));
    void grid.offsetWidth; grid.classList.remove("nofx");
    runIntoView(panel && panel.querySelector(".sb-words"));
  }
  /** hover tint of the lyrics run of the hovered card (colour only, no layout change). */
  function paintRunHover() {
    const grid = document.querySelector(".storyboard .sb-grid.vid"); if (!grid) return;
    const c = vidPin.hover != null ? grid.children[vidPin.hover] : null, n = c ? c.dataset.n : null;
    document.querySelectorAll(".storyboard .sbw-run").forEach((r) => r.classList.toggle("hov", n != null && r.dataset.n === n));
    paintWave(document.querySelector(".storyboard .sb-words.by-run"));
  }
  /** hovering a card (mouse) or keyboard-focusing inside it pins it at the full frame; leaving the row goes back to the shot-length layout.
   *  Steady: once pinned a card stays pinned for LOCK ms (the width transition), and afterwards only a REAL pointer move (> 2 px from where it was pinned) onto another card re-pins; layout shifts under a
   *  stationary pointer (browsers fire a synthetic mousemove) never change the pin. */
  function wireCardHover(grid) {
    vidPin.hover = vidPin.focus = vidPin.skip = null;
    const T = { lock: 0, ax: -9999, ay: -9999, px: 0, py: 0, moved: false, timer: 0 };
    const go = () => { if (grid._vid) vidApply(grid, grid._vid, vidPinsFor(grid), true); paintRunHover(); };
    const idxOf = (t) => { const c = t && t.closest ? t.closest(".sb-card") : null; return c && grid.contains(c) ? Number(c.dataset.i) : null; };
    const settle = () => {                                            // the pointer moved while locked: now look at what is under it
      if (!T.moved) return;
      T.moved = false;
      const i = idxOf(document.elementFromPoint(T.px, T.py));
      if (i != null && i !== vidPin.hover) pin(i);
    };
    const pin = (i) => { vidPin.hover = i; T.lock = performance.now() + VID.LOCK; T.ax = T.px; T.ay = T.py; T.moved = false; go(); clearTimeout(T.timer); T.timer = setTimeout(settle, VID.LOCK + 20); };
    grid.addEventListener("mousemove", (ev) => {
      T.px = ev.clientX; T.py = ev.clientY;
      const i = idxOf(ev.target);
      if (i == null) return;
      if (vidPin.skip != null) { if (i === vidPin.skip) return; vidPin.skip = null; }                // the card just collapsed by a click on its frame: no re-open until the pointer has left it
      if (i === vidPin.hover) return;
      if (Math.hypot(T.px - T.ax, T.py - T.ay) <= 2) return;          // stationary pointer (layout moved under it): keep the pin
      if (performance.now() < T.lock) { T.moved = true; return; }     // mid transition: decide when it has ended
      pin(i);
    });
    grid.addEventListener("mouseleave", () => { clearTimeout(T.timer); T.moved = false; T.ax = T.ay = -9999; vidPin.skip = null; if (vidPin.hover != null) { vidPin.hover = null; go(); } });
    grid.addEventListener("pointerdown", () => { const a = document.activeElement; vidPin.wasEdit = !!(a && a.matches && a.matches(".sb-line, .sb-ov, .sb-oe")); }, true);
    grid.addEventListener("focusin", (ev) => { if (ev.target.matches && ev.target.matches(".sb-line, .sb-ov, .sb-oe")) { const i = idxOf(ev.target); if (i != null) { vidPin.edit = i; const c = grid.children[i]; if (c && S.story.active !== Number(c.dataset.n)) { S.story.active = Number(c.dataset.n); setTimeout(paintFocus, 0); } go(); } } });
    grid.addEventListener("focusout", (ev) => { if (vidPin.edit != null && ev.target.matches && ev.target.matches(".sb-line, .sb-ov, .sb-oe")) { vidPin.edit = null; go(); } });
    grid.addEventListener("focusin", (ev) => { const i = idxOf(ev.target); if (i != null && i !== vidPin.focus && ev.target.matches && ev.target.matches(":focus-visible")) { vidPin.focus = i; go(); } });
    grid.addEventListener("focusout", (ev) => { if (!ev.relatedTarget || !grid.contains(ev.relatedTarget)) { if (vidPin.focus != null) { vidPin.focus = null; go(); } } });
  }

  /** v14: card column width = min(work width / shots, the width at which the whole card (frame at the chosen format + rows) fits the visible work height), no fixed cap. */
  let boardRO = null, boardLast = "";
  function fitBoard() {
    const sb = document.querySelector(".storyboard"), grid = sb && sb.querySelector(".sb-grid");
    if (!grid || !grid.firstElementChild) return;
    const n = grid.children.length, GAP = 12, PAD = 18, MIN = 150, COMFY = 320;
    const vs = vidSizes(), fm = vs ? [null, String(vs.full[0]), String(vs.full[1])] : /^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/.exec(String(S.values.format || "9:16")), ratio = fm ? Number(fm[2]) / Number(fm[1]) : 16 / 9;
    grid.style.setProperty("--fa", fm ? `${fm[1]} / ${fm[2]}` : "9 / 16");
    if (grid.classList.contains("vid")) { fitVid(sb, grid, ratio); fitAllOverlays(); return; }      // 2026-11-02 user: video mode = same-height frames, widths by shot length (below)
    const availW = grid.clientWidth || sb.clientWidth, wW = Math.max(MIN, Math.floor((availW - GAP * (n - 1)) / n));
    const top = grid.getBoundingClientRect().top - sb.getBoundingClientRect().top + sb.scrollTop, availH = sb.clientHeight - top - 17;
    const chromeOf = () => { const c = grid.firstElementChild, im = c.querySelector(".sb-img"); return c.offsetHeight - im.offsetHeight; };
    const floor = Math.min(wW, COMFY);
    let w = wW;
    for (let k = 0; k < 4; k++) {
      grid.style.setProperty("--colw", w + "px");
      const hw = Math.floor((availH - chromeOf()) / ratio) + PAD, nw = Math.max(floor, Math.min(wW, hw));
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
    if (!ov.isContentEditable) {                                                  // never rewrite the type while it is being edited on the frame (keeps the caret)
      const cols = isFrameShot(x) && x.tl && x.tl.cols ? x.tl.cols : 0;
      ov.classList.toggle("grid", !!(cols && L));
      if (cols && L) {                                                              // grid = rows of equal cells; alignment moves a short row inside the block (fitVidType)
        const rows = [], ch = [...L.replace(/\s+/g, "")];                           // all the letters, evenly through the grid: spaces and new lines don't count here
        for (let k = 0; k < ch.length; k += cols) rows.push(el("div", { class: "gr" }, ...ch.slice(k, k + cols).map((c) => el("span", {}, c))));
        ov.replaceChildren(...rows);
      } else ov.textContent = ov._edit ? x.line : L;                                // while editing keep a just-typed new line, so it shows at once
      ov.hidden = !L;
    }
    ov.style.fontFamily = f ? `"${B.fontCss(f)}", system-ui, sans-serif` : "";
    ov.dataset.size = L.length <= 7 ? "l" : L.length <= 13 ? "m" : "s";
    if (f) ensureFontFace(f);
    fitOverlay(ov);
    const nm = card.querySelector(".sb-fname");
    if (nm) { nm.textContent = f ? f.name : x.font; nm.title = f ? `${f.name}${f.role ? " (" + f.role + ")" : ""}: liked font` : ""; }
    applyInk(ov, shotPicUrl(x));
    if (isFrameShot(x)) { if (L) frameMatte(x).catch(() => {}); paintFg(card, x); syncTools(card, x); }
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
    const inPlace = !!o.select && !o.use && !!document.querySelector(".storyboard .sb-grid.vid");   // video mode: picking words changes only the words and their timing, so the frames stay on screen (a rebuild reloads them and the type jumps)
    if (inPlace) { const cur = st.data; cur.text_select = r.story.text_select; cur.hits = r.story.hits; cur.shots.forEach((s, k) => { s.line = r.story.shots[k].line; }); } else st.data = r.story;
    if (o.select) { st.edited.sel = true; st.selText = S.media.text; for (const n of r.lineChanged) { delete st.edited[`${n}.line`]; const sh = st.data.shots.find((s) => s.n === n); if (sh) delete sh.custom; } }
    if (o.use) {
      st.edited.use = true;
      for (const n of r.imageChanged) if (st.edited[`${n}.beat`]) { const sh = st.data.shots.find((s) => s.n === n); if (sh) sh.beat = beats.get(n); }
    }
    syncAllVers(true);
    if (inPlace) {
      for (const n of r.lineChanged) { const card = document.querySelector(`.storyboard .sb-card[data-n="${n}"]`), x = st.data.shots.find((s) => s.n === n); if (!card || !x) continue; const ta = card.querySelector(".sb-line"); if (ta) ta.value = x.line; paintCard(card, x); }
      paintFocus();
      return;
    }
    rerender();
  }

  // ---- UNDO / REDO of the storyboard (Ctrl+Z, Ctrl+Y or Ctrl+Shift+Z; Cmd on a Mac): snapshots taken when an edit settles. Inside a text field the browser's own undo works on the typing.
  const hist = { past: [], future: [], last: null, timer: 0, n: 0 };
  const inTextEdit = () => { const a = document.activeElement; return !!a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName)) && a.type !== "color"; };
  function histSnap() {
    const st = S.story;
    return JSON.stringify({ data: st.data, edited: st.edited, vers: st.vers, selText: st.selText, fix: S.media.lyricFix || {} }, function (k, v) {
      if (k === "bgBusy" || k === "bgA" || (k === "bg" && this.bgA) || k === "colorBusy" || k === "colorA" || ((k === "color" || k === "outline") && this.colorA)) return undefined;             // a colour sampled from the shot is not an edit (it comes back by itself)
      return (k === "x" || k === "y") && typeof v === "number" ? Math.round(v * 1000) / 1000 : v;
    });
  }
  function histTick() {
    const st = S.story, d = st.data;
    if (!d || st.busy || closeFontMenu || inTextEdit()) return;
    if (hist.n !== d.shots.length || !hist.last) { hist.past = []; hist.future = []; hist.n = d.shots.length; hist.last = histSnap(); return; }   // a new story starts a new history
    const now = histSnap();
    if (now === hist.last) return;
    hist.past.push(hist.last); if (hist.past.length > 200) hist.past.shift();
    hist.future = []; hist.last = now;
  }
  function histGo(from, to) {
    histTick();
    if (!from.length) { toast("nothing to " + (from === hist.past ? "undo" : "redo")); return; }
    to.push(hist.last); hist.last = from.pop();
    const before = JSON.parse(to[to.length - 1]), v = JSON.parse(hist.last), st = S.story, name = from === hist.past ? "Undo" : "Redo", same = (p, q) => JSON.stringify(p) === JSON.stringify(q);
    const wordsOf = (o, n) => (o.data.text_select || []).filter((t) => t.shot === n).map((t) => t.i), SOFT = ["line", "tl", "font", "custom"];
    const sameRow = before.data.shots.length === v.data.shots.length && v.data.shots.every((x, k) => before.data.shots[k].n === x.n);
    const changed = sameRow ? v.data.shots.filter((x, k) => !same(before.data.shots[k], x) || !same(wordsOf(before, x.n), wordsOf(v, x.n))) : [];
    const shotsOff = (d) => Object.fromEntries(Object.entries(d).filter(([k]) => !["shots", "text_select", "hits"].includes(k))), softShot = (x) => Object.fromEntries(Object.entries(x).filter(([k]) => !SOFT.includes(k)));
    const inPlace = sameRow && !!document.querySelector(".storyboard .sb-grid.vid") && same(shotsOff(before.data), shotsOff(v.data)) && v.data.shots.every((x, k) => same(softShot(before.data.shots[k]), softShot(x)));   // only words, font or layout changed: the frames stay on screen
    st.edited = v.edited; st.vers = v.vers; st.selText = v.selText; S.media.lyricFix = v.fix; S.media.lyricFixFor = S.media.read;
    if (inPlace) {
      const cur = st.data; cur.text_select = v.data.text_select; cur.hits = v.data.hits;
      for (const x of changed) { const live = cur.shots.find((q) => q.n === x.n); Object.keys(live).forEach((k) => { delete live[k]; }); Object.assign(live, x); }   // the cards hold these objects: update them, don't swap them
    } else st.data = v.data;
    if (st.active != null && !st.data.shots.some((x) => x.n === st.active)) st.active = null;
    if (changed.length) {                                                             // go to the shot this step changed: it opens and the lyrics line eases to it
      const go = changed.find((x) => x.n === st.active) || changed[0], o = before.data.shots.find((q) => q.n === go.n);
      st.active = go.n; stripShot = null;
      const what = (!same(o.line, go.line) || !same(wordsOf(before, go.n), wordsOf(v, go.n)) || !same(o.custom, go.custom)) ? "words" : !same(o.font, go.font) ? "font" : "layout";
      toast(`${name}: shot ${go.n}, ${what}`, 2500, { info: true });                  // info: not added to the notices (that repaints the page)
    } else stripShot = st.active;
    if (inPlace) {
      for (const x of changed) { const card = document.querySelector(`.storyboard .sb-card[data-n="${x.n}"]`), live = st.data.shots.find((q) => q.n === x.n); if (!card || !live) continue; const ta = card.querySelector(".sb-line"); if (ta) ta.value = live.line; paintCard(card, live); }
    } else rerender();
    paintFocus();
    hist.last = histSnap();                                                           // the restored state, as repainted, is this step: not a new edit
  }
  if (window.__MG_DEV) Object.assign(window.__MG_DEV, { hist, histSnap, histTick });   // dev proxy only
  document.addEventListener("pointerdown", histTick, { capture: true, passive: true });   // a new action is starting: the last one becomes its own undo step, however quickly they follow each other
  document.addEventListener("focusout", histTick, { capture: true, passive: true });      // a text edit just finished: it is a step of its own
  ["pointerup", "click", "change", "input", "keyup", "wheel", "focusout"].forEach((n) => document.addEventListener(n, () => { clearTimeout(hist.timer); hist.timer = setTimeout(histTick, 300); }, { capture: true, passive: true }));
  document.addEventListener("keydown", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey) || ev.altKey || !S.story.data || inTextEdit() || closeFontMenu) return;
    const k = ev.key.toLowerCase();
    if (k === "z" && !ev.shiftKey) { ev.preventDefault(); histGo(hist.past, hist.future); }
    else if (k === "y" || (k === "z" && ev.shiftKey)) { ev.preventDefault(); histGo(hist.future, hist.past); }
  });

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
    const vidB = byShot && isFrameBoard(d);                                           // 2026-11-02: video mode = ONE continuous lyrics section (strips), the selected shot's run highlighted
    let vidSum = null, vidBtns = [];
    if (vidB) {
      const sungN = actShot ? (B.sungByShot(d.shots, src.times).by.get(actShot.n) || []).length : 0, mineIdx = actShot ? shotTokens(actShot.n) : [];
      const alts = actShot ? (d.word_alts && d.word_alts[String(actShot.n)]) || [] : [], curKey = mineIdx.join(","), at = alts.findIndex((x) => x.t.join(",") === curKey);
      vidSum = actShot ? `Shot ${actShot.n}: click a word to use it or take it off, then shift-click another to do the same to every word between, double-click to correct a lyric; drag to see the rest${sungN ? "" : " (nothing is sung in this shot)"}`
        : `${withWords} of ${d.shots.length} shot${d.shots.length === 1 ? "" : "s"} ${withWords === 1 ? "has" : "have"} words. Click a shot to go to its words; drag the line to see the rest`;
      vidBtns = [
        el("button", { class: "sbw-btn", type: "button", "data-act": "cyc", disabled: !actShot || alts.length < 1 || (alts.length === 1 && at === 0), "aria-label": "Next-best word pick for the selected shot",
          title: actShot ? (alts.length ? `Next-best auto pick for shot ${actShot.n} (${at >= 0 ? at + 1 : 0} of ${alts.length} tried)` : "No other pick for this shot") : "Select a shot first",
          onclick: () => { const ns = actShot && B.cycleShotWords(S.story.data, actShot.n); if (ns) applyOverride({ select: ns }); else toast("no other pick for this shot"); } }, "↻"),
        el("button", { class: "sbw-btn", type: "button", "data-act": "clr", disabled: !actShot || !mineIdx.length, "aria-label": "Clear the words of the selected shot", title: actShot ? "Take every word off this shot" : "Select a shot first",
          onclick: () => { if (actShot) applyOverride({ select: B.clearShotWords(S.story.data.text_select, actShot.n) }); } }, "clear")];
    }
    const head = el("div", { class: "sb-th" }, el("b", vidB ? { title: vidSum } : {}, vidB ? "Text" : "Your text"), src.lyrics && !vidB ? el("span", { class: "sb-lyr", title: "No text was pasted: these are the words heard in your media, with the time each is sung" }, `Lyrics (heard in the ${src.from})`) : null,
      el("span", { class: "sb-tsum", title: vidSum }, vidB ? vidSum : byShot ? `${withWords} of ${d.shots.length} shot${d.shots.length === 1 ? "" : "s"} ${withWords === 1 ? "has" : "have"} words. Click a word to put it on its shot or take it off; shift-click or drag across words for a phrase; ↻ tries the next-best pick`
        : actShot ? `Shot ${actShot.n}: click words to put them on this shot or take them off (Esc or its frame again to finish)`
        : withWords ? `${withWords} of ${d.shots.length} shot${d.shots.length === 1 ? "" : "s"} ${withWords === 1 ? "has" : "have"} words; click a frame to change its words` : "no words chosen; click a frame to pick its words"),
      ...vidBtns,
      el("button", { class: "sbw-btn sbw-auto", type: "button", title: "Back to the automatic word choice for every shot (drops your edits to the words)", onclick: autoPickWords }, "Auto-pick words"));
    const words = byShot ? buildShotWords(d, src, sel, act, vidB) : el("div", { class: "sb-words" });
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
    const panel = el("div", { class: "sb-text" + (act != null ? " focus" : "") + (vidB ? " by-run" : "") }, head, words);
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
  let wordDrag = null, wordAnchor = null, wordDragDone = false, vidWords = false, wordClickT = 0;
  document.addEventListener("pointerup", () => {
    const g = wordDrag; wordDrag = null;
    if (!g || !g.moved) return;
    wordDragDone = true; setTimeout(() => { wordDragDone = false; }, 0);
    const d = S.story.data;
    if (d) { wordAnchor = { i: g.to, shot: g.shot }; S.story.wordRow = g.shot; if (vidWords) S.story.active = g.shot; revealWant = { n: g.shot, t: Date.now(), still: true }; applyOverride({ select: B.rangeWordShot(d.text_select, g.from, g.to, g.shot) }); }
  });
  function buildShotWords(d, src, sel, act, vidB) {
    vidWords = !!vidB;
    const tokens = src.tokens, times = src.times, cap = B.TEXT_CAP, shotOf = new Map(sel.map((t) => [t.i, t.shot]));
    const { by, outside } = B.sungByShot(d.shots, times), gem = S.story.gemma;
    const wrap = el("div", { class: "sb-words by-shot" });
    /** the single lyrics line: with a shot selected, a click puts this word on THAT shot in place of its words (click its own word = off, shift-click = a phrase); with none selected, a click selects the word's shot. */
    const stripWord = (i, n) => {
      const sh = shotOf.get(i) || 0, at = Number(times[i]), mine = act != null && sh === act;
      const usedOn = sh >= 1 ? d.shots.find((s) => s.n === sh) : null, m = tokens[i].match(/^([^\p{L}\p{N}]*)(.*?)([^\p{L}\p{N}]*)$/su), low = tokens[i].toLowerCase();
      const trimmed = usedOn && m && m[2] && (m[1] || m[3]) && !String(usedOn.line).split(/\s+/).some((w) => w.toLowerCase() === low);   // the shot uses this word without its punctuation: only what's used lights up
      const b = el("button", { class: "w" + (sh >= 1 ? " on" : "") + (mine ? " mine" : "") + (trimmed ? " part" : ""), type: "button", "data-i": i, "data-t": tokens[i], "data-shot": sh >= 1 ? sh : null, "data-ts": Number.isFinite(at) ? `${at.toFixed(1)}s` : null,
        title: act == null ? (n != null ? `Sung in shot ${n}. Click to select that shot.` : "Select a shot first.") : mine ? `On shot ${act}. Click to take it off, then shift-click another word to take off everything between.` : `Click to use this on shot ${act}, then shift-click another word to use everything between.` }, trimmed ? el("span", { class: "w-in" }, m[1], el("span", { class: "w-u" }, m[2]), m[3]) : tokens[i]);
      b.addEventListener("click", (ev) => {
        const dd = S.story.data;
        if (stripDragged || !dd) return;
        if (ev.detail >= 2) { clearTimeout(wordClickT); fixWord(b, i); return; }       // double-click = correct the lyric
        if (act == null) { if (n != null) { S.story.active = n; paintFocus(); } return; }
        if (ev.shiftKey && (!wordAnchor || wordAnchor.shot !== act || wordAnchor.i === i)) { clearTimeout(wordClickT); wordAnchor = { i, shot: act, off: mine }; applyOverride({ select: B.toggleWordShot(dd.text_select, i, act) }); return; }   // nothing to run from: this word flips, the shot's other words stay
        if (ev.shiftKey) {                                                              // from the last clicked word to this one: a word turned on lights everything between, a word turned off clears it
          const a = wordAnchor.i, off = wordAnchor.off; wordAnchor = { i, shot: act, off };
          if (!off) { applyOverride({ select: B.rangeWordShot(dd.text_select, a, i, act) }); return; }
          let next = dd.text_select; for (let j = Math.min(a, i); j <= Math.max(a, i); j++) if (next.some((t) => t.i === j && t.shot === act)) next = B.toggleWordShot(next, j, act);
          applyOverride({ select: next }); return;
        }
        clearTimeout(wordClickT);
        wordClickT = setTimeout(() => {                                                // a moment's wait: this may be the first click of a double-click
          const cur = S.story.data; if (!cur) return;
          wordAnchor = { i, shot: act, off: mine };
          applyOverride({ select: B.toggleWordShot(mine ? cur.text_select : B.clearShotWords(cur.text_select, act), i, act) });
        }, 220);
      });
      return b;
    };
    const fixWord = (b, i) => {
      const inp = el("input", { class: "w-edit", type: "text", value: tokens[i], size: Math.max(3, tokens[i].length + 1), spellcheck: "false", "aria-label": "Correct this lyric" });
      let done = false; const end = (save) => { if (done) return; done = true; if (save && inp.value !== tokens[i]) correctToken(i, inp.value); else paintFocus(); };
      inp.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      inp.addEventListener("click", (ev) => ev.stopPropagation());
      inp.addEventListener("input", () => { inp.size = Math.max(3, inp.value.length + 1); });
      inp.addEventListener("keydown", (ev) => { ev.stopPropagation(); if (ev.key === "Enter") end(true); else if (ev.key === "Escape") end(false); });
      inp.addEventListener("blur", () => end(true));
      b.replaceWith(inp); inp.focus(); inp.select();
    };
    const chip = (i, n) => {
      if (vidB) return stripWord(i, n);
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
        wrap.querySelectorAll(`.sbw-shot[data-n="${n}"] .w, .sbw-run[data-n="${n}"] .w`).forEach((c) => c.classList.toggle("drag", Number(c.dataset.i) >= lo && Number(c.dataset.i) <= hi));
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
    if (vidB) {                                                                       // ONE continuous text, time order: each shot's sung words are one run (data-n), a faint shot number marks where it starts
      const owner = new Map(); for (const [n, arr] of by) for (const i of arr) owner.set(i, n);
      wrap.classList.add("by-run");
      let cur = null, curN;
      for (let i = 0, tot = Math.min(cap, tokens.length); i < tot; i++) {
        const n = owner.has(i) ? owner.get(i) : null;
        if (!cur || curN !== n) {
          curN = n; cur = el("span", { class: "sbw-run" + (n == null ? " other" : "") + (n != null && n === act ? " sel" : ""), "data-n": n == null ? null : n }, n == null ? null : el("sup", { class: "sbw-sn", title: `Shot ${n} starts here` }, String(n)));
          if (n != null) {
            const run = cur;
            run.addEventListener("click", (ev) => { if (stripDragged || (ev.target.closest && ev.target.closest(".w"))) return; S.story.active = n; paintFocus(); });                  // a gap in the line selects that run's shot
          }
          wrap.append(cur, " ");
        }
        cur.append(chip(i, n), " ");
      }
      for (const sh of d.shots) {                                                     // custom words: dashed chips at the start of their shot's words (a shot with nothing sung gets its own run, in shot order)
        if (!sh.custom || !sh.custom.length) continue;
        let run = wrap.querySelector(`.sbw-run[data-n="${sh.n}"]`);
        if (!run) {
          run = el("span", { class: "sbw-run" + (sh.n === act ? " sel" : ""), "data-n": sh.n }, el("sup", { class: "sbw-sn", title: `Shot ${sh.n} starts here` }, String(sh.n)));
          const next = [...wrap.querySelectorAll(".sbw-run[data-n]")].find((r) => Number(r.dataset.n) > sh.n);
          wrap.insertBefore(run, next || null);
        }
        const at = run.querySelector(".sbw-sn").nextSibling, tails = new Map();
        // a custom word sits where it reads on the frame: after the lyric it follows, or before the first one, in this shot's run; else at the start of the run
        const put = (c, node) => {
          const ref = c.after != null ? run.querySelector(`.w[data-i="${c.after}"]`) : null;
          if (ref) { const from = tails.get(c.after) || ref, nx = from.nextSibling; run.insertBefore(document.createTextNode(" "), nx); run.insertBefore(node, nx); tails.set(c.after, node); return; }
          const bf = c.before != null ? run.querySelector(`.w[data-i="${c.before}"]`) : null, point = bf || at;
          run.insertBefore(node, point); run.insertBefore(document.createTextNode(" "), point);
        };
        for (const c of sh.custom) {
          const chipC = el("button", { class: "w cust" + (sh.n === act ? " mine" : ""), type: "button", "data-t": c.t, title: sh.n === act ? "Your own word, not in the lyrics. Click to take it off." : `Your own word on shot ${sh.n}. Click to select that shot.` }, c.t);
          chipC.addEventListener("click", (ev) => { ev.stopPropagation(); if (stripDragged) return; if (S.story.active !== sh.n) { S.story.active = sh.n; paintFocus(); } else removeCustom(sh.n, c.t); });
          put(c, chipC);
          if (c.near != null && c.near < tokens.length) {
            const fx = el("button", { class: "w-fix", type: "button", title: `Change "${tokens[c.near]}" in the lyrics to "${c.t}"` }, "Fix lyric?");
            fx.addEventListener("click", (ev) => { ev.stopPropagation(); if (!stripDragged) correctToken(c.near, c.t); });
            put(c, fx);
          }
        }
      }
      wireStrip(wrap);
      return wrap;
    }
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
    if (old && textSource().tokens.length) {
      const ps = old.querySelector(".sb-words"), top = ps ? ps.scrollTop : 0, h = old.style.height, np = buildTextPanel(d);
      if (h) np.style.height = h;                                                  // video mode: the section keeps the height fitVid reserved (nothing moves)
      old.replaceWith(np); const nw = np.querySelector(".sb-words"); if (nw) nw.scrollTop = top;
      if (sb.classList.contains("vidboard")) { runIntoView(nw); paintRunHover(); }
    }
    const vg = sb.querySelector(".sb-grid.vid");
    if (vg && vg._vid) vidApply(vg, vg._vid, vidPinsFor(vg), true);                   // video mode: the selected card is pinned open (or collapses when deselected)
  }
  /** selected shot's run: when the lyrics section scrolls inside itself, bring the run into view (the section only; the page never moves). */
  /** 2026-11-03 user: the soundtrack's waveform behind the lyrics line, each word over the part of it where it is sung: gaps in the singing are real time on the line (WAVE_PX px a second), and the wave is stretched between the words' own edges. */
  const WAVE_PX = 34, WAVE_H = 22, WAVE_MAX_BYTES = 400e6, wave = { blob: null, busy: false, peaks: null, dur: 0 };
  function waveBlob() {
    const video = S.media.items.find((i) => i.kind === "video"), track = S.media.items.find((i) => i.kind === "audio"), st = soundtrackNow();
    return track && track.file && st === "track" ? track.file : video && video.file && st === "video" && video.hasAudio !== false ? video.file : null;
  }
  function loadWave(blob) {
    if (wave.blob === blob || !blob || blob.size > WAVE_MAX_BYTES) return;
    wave.blob = blob; wave.peaks = null; wave.busy = true;
    (async () => {
      const ctx = new (window.AudioContext || window.webkitAudioContext)(), buf = await ctx.decodeAudioData(await blob.arrayBuffer()); ctx.close();
      const R = 100, n = Math.ceil(buf.duration * R), peaks = new Float32Array(n), step = buf.sampleRate / R;
      for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let k = 0; k < n; k++) { const a = Math.floor(k * step), b = Math.min(d.length, Math.floor((k + 1) * step)); let m = peaks[k]; for (let i = a; i < b; i += 4) { const v = Math.abs(d[i]); if (v > m) m = v; } peaks[k] = m; } }
      let top = 0; for (const v of peaks) if (v > top) top = v; if (top) for (let k = 0; k < n; k++) peaks[k] /= top;
      wave.peaks = peaks; wave.dur = buf.duration; wave.busy = false;
      const w = document.querySelector(".storyboard .sb-words.by-run"); if (w) { const left = w.scrollLeft; layoutWave(w); w.scrollLeft = left; paintWave(w); }
    })().catch(() => { wave.busy = false; });
  }
  function layoutWave(words) {
    const times = textSource().times; if (!times) return;
    words.querySelectorAll(".sbw-gap, .sbw-wave, .sbw-cutn").forEach((e) => e.remove());
    const dur = wave.dur || videoDur() || 0, chips = [...words.querySelectorAll(".w[data-i]")].filter((e) => Number.isFinite(times[Number(e.dataset.i)]));
    if (!chips.length || !dur) return;
    const gap = (sec) => el("span", { class: "sbw-gap", style: `width:${Math.round(sec * WAVE_PX)}px` }), spans = [];
    let prev = 0;
    chips.forEach((e, k) => {
      const i = Number(e.dataset.i), t = times[i], nx = k + 1 < chips.length ? times[Number(chips[k + 1].dataset.i)] : Infinity, end = Math.max(t + 0.05, Math.min(nx, t + 0.7));
      if (t - prev > 0.2) { const r = e.parentElement; (r.querySelector(".w[data-i]") === e ? r : e).before(gap(t - prev)); }
      spans.push([e, t, end]); prev = end;
    });
    if (dur - prev > 0.05) words.append(gap(dur - prev));
    const box = words.getBoundingClientRect(), X = (x) => x - box.left + words.scrollLeft, gaps = words.querySelectorAll(".sbw-gap");
    const first = spans[0][0].getBoundingClientRect(), x0 = X(gaps.length && gaps[0].getBoundingClientRect().left < first.left ? gaps[0].getBoundingClientRect().left : first.left), lastEl = gaps.length && gaps[gaps.length - 1].getBoundingClientRect().left >= spans[spans.length - 1][0].getBoundingClientRect().right - 1 ? gaps[gaps.length - 1] : spans[spans.length - 1][0], x1 = X(lastEl.getBoundingClientRect().right);
    const pts = [[x0, 0]]; for (const [e, t, end] of spans) { const r = e.getBoundingClientRect(); pts.push([X(r.left), t], [X(r.right), end]); } pts.push([x1, dur]);
    const timeAt = (x) => { for (let k = 1; k < pts.length; k++) if (x <= pts[k][0]) { const [a, ta] = pts[k - 1], [b, tb] = pts[k]; return b > a ? ta + ((x - a) / (b - a)) * (tb - ta) : tb; } return dur; };
    words._xAt = (t) => { for (let k = 1; k < pts.length; k++) if (t <= pts[k][1]) { const [a, ta] = pts[k - 1], [b, tb] = pts[k]; return tb > ta ? a + ((t - ta) / (tb - ta)) * (b - a) : b; } return x1; };
    if (!wave.peaks) return;
    const W = Math.ceil(x1 - x0), dpr = window.devicePixelRatio || 1, cv = el("canvas", { class: "sbw-wave", style: `left:${x0}px;width:${W}px;height:${WAVE_H}px` });
    cv.width = Math.round(W * dpr); cv.height = Math.round(WAVE_H * dpr);
    words._wave = { x0, W, dpr, timeAt, spans: spans.map(([e]) => { const r = e.getBoundingClientRect(); return [X(r.left) - x0, X(r.right) - x0, e]; }) };
    cv.addEventListener("click", (ev) => {                                         // 2026-11-03 user: click a section of the wave = jump to its frame (selected and open); a click on the selected section keeps it
      if (stripDragged) return;
      const x = x0 + ev.offsetX, sh = S.story.data.shots.find((q) => x >= words._xAt(shotT0(q)) && x < words._xAt(shotT1(q)));
      if (!sh) return;
      S.story.active = stripJump = sh.n; paintFocus();
      setTimeout(() => revealShot(sh.n), 260);                                    // once the columns have settled
    });
    words.prepend(cv);
    for (const q of S.story.data.shots) words.append(el("span", { class: "sbw-cutn", style: `left:${words._xAt(shotT0(q))}px` }, String(q.n)));   // 2026-11-03 user: each shot's number sits above its cut line
  }
  /** 2026-11-03 user: the selected and hovered shots light their own stretch of the wave (white on grey), and the line follows: it always travels to the shot in hand (hovered, else selected). */
  function paintWave(words) {
    if (!words || !words._xAt) return;
    const d = S.story.data, grid = document.querySelector(".storyboard .sb-grid.vid"), hc = grid && vidPin.hover != null ? grid.children[vidPin.hover] : null, hov = hc ? Number(hc.dataset.n) : null, act = S.story.active, shotOf = (n) => (n == null ? null : d.shots.find((q) => q.n === n));
    const wv = words._wave, cv = words.querySelector(".sbw-wave");
    if (wv && cv) {
      const secs = [shotOf(act), shotOf(hov)].filter(Boolean).map((q) => [words._xAt(shotT0(q)) - wv.x0, words._xAt(shotT1(q)) - wv.x0]), g = cv.getContext("2d");
      g.setTransform(wv.dpr, 0, 0, wv.dpr, 0, 0); g.clearRect(0, 0, wv.W, WAVE_H);
      for (let x = 0; x < wv.W; x += 2) {
        const a = Math.floor(wv.timeAt(wv.x0 + x) * 100), b = Math.max(a + 1, Math.floor(wv.timeAt(wv.x0 + x + 2) * 100)); let m = 0; for (let k = a; k < b && k < wave.peaks.length; k++) if (wave.peaks[k] > m) m = wave.peaks[k];
        const on = secs.some((q) => x >= q[0] && x < q[1]), h = Math.max(1, m * (WAVE_H - 2));
        g.fillStyle = on ? "rgba(255,255,255,.95)" : "rgba(166,161,182,.38)";
        g.fillRect(x, (WAVE_H - h) / 2, 1.5, h);
      }
      g.fillStyle = "rgba(120,170,255,.8)"; for (const q of d.shots.slice(1)) g.fillRect(words._xAt(shotT0(q)) - wv.x0, 0, 1, WAVE_H);   // the cuts
    }
    const n = hov != null ? hov : act;
    if (n == null) { stripShot = null; return; }
    if (n === stripShot) return;
    stripShot = n;
    const sh = shotOf(n); if (!sh) return;
    const from = words.scrollLeft, t0 = performance.now(), D = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 320;
    const to = Math.max(0, Math.min(words.scrollWidth - words.clientWidth, (words._xAt(shotT0(sh)) + words._xAt(shotT1(sh))) / 2 - words.clientWidth / 2));
    cancelAnimationFrame(stripAnim);
    const step = (t) => { const w = document.querySelector(".storyboard .sb-words.by-run"); if (!w) return; const p = Math.min(1, Math.max(0, (t - t0) / D)), e = p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; w.scrollLeft = stripX = from + (to - from) * e; if (p < 1) stripAnim = requestAnimationFrame(step); };   // ease in and out
    stripAnim = requestAnimationFrame(step);
  }
  function runIntoView(words) {
    if (!words || !words.classList.contains("by-run")) return;
    loadWave(waveBlob()); layoutWave(words);
    words.scrollLeft = stripX;                                                     // a rebuild keeps where the line was
    paintWave(words);
  }
  /** video mode: a click anywhere on a card (its controls too) selects its shot = its lyrics run is highlighted and the card stays OPEN (full frame + details) when the mouse leaves;
   *  a click on the selected card's FRAME again deselects it (it collapses at once, and stays collapsed until the pointer has left it). Fonts / text-field clicks never deselect. */
  function selectShot(n, idx, frame) {
    const st = S.story, jumped = frame && stripJump === n;                          // 2026-11-03 user: the first click on a frame reached from the wave is a first click (cursor if it has no words), not the deselecting second one
    stripJump = null;
    if (st.active === n && !jumped) {
      if (!frame || vidPin.wasEdit) return;                                        // a click that ends a text edit keeps the shot selected and open
      st.active = null; vidPin.hover = null; vidPin.skip = idx;
    } else {
      st.active = n;
      if (frame) setTimeout(() => {                                                 // a frame with no words: open a text box on it straight away
        const x = st.data && st.data.shots.find((q) => q.n === n), card = document.querySelector(`.storyboard .sb-card[data-n="${n}"]`), ov = card && card.querySelector(".sb-ov"), line = card && card.querySelector(".sb-line");
        if (st.active === n && x && !x.line.trim() && ov && line) startOvEdit(ov, line, x);
      }, 0);
    }
    setTimeout(paintFocus, 0);                                                      // after the click's own handlers (a control may rebuild the page; paintFocus is idempotent)
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
    line.addEventListener("input", () => { edit("line", line.value); if (isFrameShot(x)) syncHitsOfShot(i); paintCard(card, x); });
    line.addEventListener("change", () => { if (isFrameShot(x)) reconcileShot(x.n); });
    const beat = el("textarea", { class: "sb-beat", rows: 3, spellcheck: "false", placeholder: "What happens", "aria-label": `Shot ${x.n} beat` });
    beat.value = x.beat;
    beat.addEventListener("input", () => edit("beat", beat.value));
    const cv = B.verSync(st.vers[key("camera")], x.camera, st.gemma ? B.fieldGet(st.gemma, key("camera")) : undefined);   // every version of the camera is an option (the arrows can land on it)
    const cams = [...new Set(["", ...B.CAMERAS, x.camera, ...B.verList(cv)])];
    const cam = el("select", { class: "sb-cam", "aria-label": `Shot ${x.n} camera` }, cams.map((c) => el("option", { value: c }, c ? B.cameraLabel(c) : "no move")));
    cam.value = x.camera;
    cam.addEventListener("change", () => edit("camera", cam.value));
    const vid = isFrameShot(x);                                              // v18: video mode cards = frame, font stepper, editable line (no transition / camera row)
    const iconBtn = (act, title, inner, fn) => el("button", { class: "sb-fbtn", type: "button", "data-act": act, title, "aria-label": title, onclick: fn }, inner);
    const fontRow = list.length ? el("div", { class: "sb-font" },
      iconBtn("prev", "Previous liked font", "‹", () => setFont(x, B.fontStep(list, x.font, -1))),
      el("span", { class: "sb-fname" }),
      iconBtn("next", "Next liked font", "›", () => setFont(x, B.fontStep(list, x.font, 1)))) : el("div", { class: "sb-font-none" });
    const info = e && !vid ? el("button", { class: "icon-btn sb-info" + (st.openNotes[x.n] ? " on" : ""), type: "button", title: "What the read saw in this image", "aria-label": "Image notes", html: svg("info", 13),
      onclick: () => { st.openNotes[x.n] = !st.openNotes[x.n]; notes.hidden = !st.openNotes[x.n]; info.classList.toggle("on", !!st.openNotes[x.n]); } }) : null;
    const notes = e && !vid ? el("div", { class: "sb-notebox", hidden: !st.openNotes[x.n] }, noteEditor(it, e, x.image || ids[S.media.items.indexOf(it)])) : null;
    const ov = el("div", { class: "sb-ov" });
    const vidTools = vid ? buildTools(() => card, x, line, ov) : null;
    if (vidTools) { ov.title = "Click to edit, drag to move, wheel to scale"; ov.addEventListener("click", (ev) => startOvEdit(ov, line, x, ev)); }
    const swap = it ? el("button", { class: "sb-swap", type: "button", title: "Swap this shot's picture", "aria-label": `Shot ${x.n}: swap the picture`, onclick: (ev) => { ev.stopPropagation(); openPicker(i, img); } }, "swap") : null;
    const img = el("div", { class: "sb-img", tabIndex: 0, role: "button", "aria-pressed": st.active === x.n ? "true" : "false", "aria-label": `Shot ${x.n}: pick this shot's words`,
      title: "Click to work on this shot's words (click again or press Esc to finish)" }, ...storyImgNode(x), vidTools ? el("div", { class: "sb-tl" }, ov) : ov, ...(vidTools ? [el("div", { class: "sb-tc" }, el("div", { class: "sb-gh" })), el("div", { class: "sb-to" }, el("div", { class: "sb-gh" }))] : []), swap, vid ? el("span", { class: "sb-n" }, String(x.n)) : null);
    if (frameRO) frameRO.observe(img);
    img.addEventListener("click", () => { if (!vid) setActive(x.n); });                                  // video mode: the whole card selects (below), never toggles
    img.addEventListener("keydown", (ev) => { if (ev.target === img && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); if (vid) selectShot(x.n, i, true); else setActive(x.n); } });
    const act = st.active === x.n, dimmed = st.active != null && !act;
    const card = el("div", { class: "sb-card" + (act ? " active" : "") + (dimmed ? " dim" : ""), "data-i": i, "data-n": x.n },
      el("div", { class: "sb-top" + (vid ? " vid" : "") }, el("b", {}, el("span", { class: "sb-shotw" }, "Shot "), String(x.n)),
        el("span", { class: "sb-span", title: B.isVideoShot(x) ? "Output time in the piece; the video's own time of this cut in brackets" : null }, B.fmtSpan(x.start, x.end) + (B.isVideoShot(x) ? ` (video ${x.src_start.toFixed(1)}-${x.src_end.toFixed(1)} s)` : "")), info),
      img, fontRow, el("div", { class: "sb-fld sb-plain" }, line),
      ...(vid ? [vidTools] : [el("div", { class: "sb-camrow" }, el("span", { class: "sb-k", title: "Camera: the cut that carries this shot into the next one" }, "Camera"), el("div", { class: "sb-fld sb-plain" }, cam)),
                      fld(beat, key("beat")), notes || el("div", { class: "sb-notebox", hidden: true })]));            // video mode: no beat / notes (v16) and no transition / camera (v18) rows
    if (vid) card.addEventListener("click", (ev) => selectShot(x.n, i, !!(ev.target.closest && ev.target.closest(".sb-img") && !ev.target.closest(".sb-swap, .sb-ov, .sb-tools"))), true);                                // anywhere on the card (its controls too) selects the shot: its lyrics run is highlighted
    paintCard(card, x);
    if (vid) loadCutFrame(card, x, null);
    return card;
  }

  /** textareas grow with their text (CSS field-sizing where the browser has it, scrollHeight otherwise). */
  const autosize = (ta) => { if (window.CSS && CSS.supports && CSS.supports("field-sizing", "content")) return; ta.style.height = "auto"; ta.style.height = ta.scrollHeight + 2 + "px"; };

  /** video mode: a card's words were edited by hand: the hit(s) landing in that shot follow (one hit per shot; emptied -> no hit). */
  const sungIn = (x, txt) => { const src = textSource(), tk = src.tokens || [], tm = src.times || [], want = txt.toUpperCase().split(/\s+/).filter(Boolean); return want.some((w) => tk.some((t, k) => String(t).toUpperCase().replace(/[^A-Z0-9']/g, "") === w.replace(/[^A-Z0-9']/g, "") && Number(tm[k]) >= Number(x.src_start ?? x.start) - 0.3 && Number(tm[k]) < Number(x.src_end ?? x.end))); };
  function syncHitsOfShot(i) {
    const d = S.story.data, x = d && d.shots[i];
    if (!x) return;
    if (!d.hits) d.hits = B.hitsFromLines(d.shots, d.shots.map((s, k) => (k === i ? "" : s.line)));
    const last = i === d.shots.length - 1, inside = (h) => h.land >= x.start - 1e-6 && (h.land < x.end - 1e-6 || (last && h.land <= x.end + 1e-6));
    const mine = d.hits.filter(inside), txt = x.line.trim().replace(/\s*\n\s*/g, " "), rest = d.hits.filter((h) => !inside(h));
    if (txt) rest.push(mine.length && mine.map((h) => h.text).join(" ").toUpperCase() === txt.toUpperCase() ? mine[0] : mine.length && B.lyricTokens && sungIn(x, txt) ? Object.assign({}, mine[0], { text: txt }) : { text: txt, land: x.start, hold_until: x.end });   // 2026-11-04: new words do not inherit the old words' landing time (a typed word arrived up to 2.9 s late) unless they are sung in this shot
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
    const vgrid = isFrameBoard(d);                                                                        // v16: video mode = ONE horizontal row of cards (no cut strip); 2026-11-02: cards first, then the lyrics section, then the head
    const wrap = el("div", { class: "storyboard" + (stale ? " stale" : "") + (st.active != null ? " has-active" : "") + (vgrid ? " vidboard" : "") });
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
    const headEl = el("div", { class: "sb-head" }, title, logline, meta);
    if (!vgrid) wrap.append(headEl);
    requestAnimationFrame(() => { wrap.querySelectorAll(".sb-look, .sb-music").forEach(autosize); fitBoard(); });
    if (!vgrid && textSource().tokens.length) wrap.append(buildTextPanel(d));

    closePop();
    const grid = el("div", { class: "sb-grid" + (vgrid ? " vid" : ""), style: `--n:${Math.max(1, Math.min(6, d.shots.length))}` });
    if (vgrid) { wireRowScroll(grid); wireCardHover(grid); }
    if (frameRO) frameRO.disconnect();
    d.shots.forEach((x, i) => grid.append(buildCard(x, i, d)));
    if (vgrid && textSource().tokens.length) wrap.append(buildTextPanel(d));         // 2026-11-03 user: video mode: the lyrics at the top, then the cards, then title / logline / LOOK / MUSIC
    wrap.append(grid);
    if (vgrid) wrap.append(headEl);
    if (boardRO) boardRO.disconnect();
    if (typeof ResizeObserver !== "undefined") { boardLast = ""; boardRO = new ResizeObserver(() => { const k = wrap.clientWidth + "x" + wrap.clientHeight; if (k !== boardLast) { boardLast = k; fitBoard(); } }); boardRO.observe(wrap); }
    wrap.addEventListener("focusout", (ev) => { if (ev.target.tagName === "TEXTAREA") fitBoardSoon(); });   // no "notes for other files" (user v13): an unused image's read note becomes the shot's beat when it is picked (applyOverride)
    return wrap;
  }

  function buildWork() {
    const st = S.story, d = st.data;
    if (S.fonts.state === "idle") loadFonts();
    const showBar = (!!d || storyOn()) && !S.media.items.some((m) => m.kind === "video");   // 2026-11-03 user: no status / matte / Shuffle all / Write again row on a video's board
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
    const body = { edit, video: video.uploaded, brief: wholeWindow(brief ? Object.assign(JSON.parse(brief), { edit }) : { edit }), length: Number(values.length) || 12, seed,
                   format: values.format || "9:16", scale: vidScale === "half" ? "half" : "full", quality: OWN_PREVIEW_QUALITY,
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

  /** IMAGE BOARD, step 2 (2026-11-03): the graph's plain footage (no type) is the "video": upload it, then POST /mograph/own_video with edit "overlay", the sound it carries kept, and the board's story where each shot
   *  is a cut of that footage (src_start = start, src_end = end, image V1) with its line / font / tl. tools/app/frame_render.py then sets the type and the mattes, exactly as for a dropped video. */
  function platesBrief(total) {
    const b = JSON.parse(currentBrief() || "{}");
    if (b.story && Array.isArray(b.story.shots)) b.story.shots.forEach((x) => { x.image = "V1"; x.src_start = Number(x.start) || 0; x.src_end = Number(x.end) || 0; });
    b.media = [{ id: "V1", kind: "video", shows: "", mood: "", window: [0, Math.ceil(total * 100) / 100] }];
    b.edit = "overlay";
    return b;
  }
  async function chainPlates(r) {
    try {
      const fin = finalOutput(), d = S.story.data, total = Math.max(...d.shots.map((x) => Number(x.end) || 0), 0);
      if (!fin || !(total > 0)) throw new RunError({ message: "The footage finished but no video came back to put the type on." });
      fin.final = false; fin.label = "Footage without type";
      r.own = { stage: "uploading", log: [], jobId: null, poll: null, fails: 0, soundtrack: "video" };
      r.tracker = null; r.promptId = null; r.pending = [];
      r.clientId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
      BV.start({ mode: "video", shots: 0, soundtrack: "video" });
      BV.setProgress(() => bvOwnProgress(r));
      S.view = "run"; renderCanvas(); updateRunButton();
      const blob = await (await fetchT(fin.url, {}, 600000)).blob();
      if (r.cancelled) return endOwn(r, "cancelled");
      const up = await upload(new File([blob], `plates_${r.clientId.slice(0, 8)}.mp4`, { type: "video/mp4" }));
      if (r.cancelled) return endOwn(r, "cancelled");
      const seed = Number(r.used && r.used.seed), route = (mediaField() || {}).own_video_route || "/mograph/own_video";
      const body = { edit: "overlay", video: up, brief: platesBrief(total), length: Math.ceil(total * 100) / 100, seed: Number.isFinite(seed) && seed > 0 ? seed : B.randomSeed(), format: S.values.format || "9:16", scale: "full",
                     quality: OWN_PREVIEW_QUALITY, soundtrack: "video", client_id: r.clientId };
      await connectWS(r);
      if (r.cancelled) return endOwn(r, "cancelled");
      let resp;
      try { resp = await fetchT(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }, 60000); }
      catch (_) { throw new RunError({ message: "Could not reach the server to start the type job." }); }
      const j = await resp.json().catch(() => ({}));
      if (!resp.ok || !j.job) throw new RunError({ head: "The type job did not start", message: String(j.error || `HTTP ${resp.status}`) });
      if (r.cancelled) { await cancelOwnJob(j.job); return endOwn(r, "cancelled"); }
      r.own.jobId = j.job; r.own.stage = "starting"; r.status = "running";
      paintOwn(r);
      r.own.poll = setInterval(() => pollOwn(r), 3000);
      pollOwn(r);
    } catch (e) { failRun(r, e instanceof RunError ? e.err : { message: String((e && e.message) || e) }); }
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
    const plates = !hasVideo && !!mf && hasCurrentStory() && isImgBoard(S.story.data) && S.story.data.shots.some((x) => x.line.trim());      // an image board with type on it: the footage first, the type from the video route; no type at all = the old image route (the graph's own words)
    const r = { active: true, status: "uploading", plates, storyShots: hasCurrentStory() ? S.story.data.shots.length : 0, clickT: now(), tracker: null, promptId: null, ws: null, ticker: null, watch: null,
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
      if (plates) values.quality = PLATES_QUALITY;
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
    if (r.plates && finalOutput()) { r.active = true; return chainPlates(r); }           // image board: the footage is not the result, the type job below is
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
