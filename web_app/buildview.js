/* MOTNGRFX Build view: a live timeline of the piece while it is made (window.MographBuild; the pure helpers also load in node).
 *
 * What it shows (v1):
 *   - SHOTS lane   one slot per shot at its start-end. The storyboard picture (image mode) / the cut's first frame (video mode) first, then the
 *                  generated still, then the SAMPLER PREVIEW frames of that shot's LTX render as ComfyUI streams them (binary websocket messages,
 *                  extra_data.preview_method), with a progress ring, then the finished shot clip as a muted loop.
 *   - CUTS lane    the transition between two shots, labelled with its cut type.
 *   - WORDS lane   each on-screen line at its land time; lights up when its type pass is done.
 *   - MUSIC lane   full length from the start. While the music sampler runs its waveform FORMS and refines step by step (the
 *                  `motngrfx.waveform` event of the MographLiveWaveformPreview node: the x0 prediction of every step, VAE-decoded, as a peak
 *                  envelope), then the real decoded waveform of the track. A user's track / a video's own sound is drawn at once.
 *   - a stage-weighted progress bar (real node progress events), elapsed time + estimate, a playhead for the finished video, and a STACK of
 *     everything that has landed (newest first).
 * Nothing is faked: a lane with no data yet shows an empty placeholder. Frames are drawn at most PREVIEW_FPS per slot; no blob URL outlives a run.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MographBuild = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const PREVIEW_FPS = 8;                       // max preview redraws per slot per second
  const STRIP_EVENT = "motngrfx.strip";        // a whole latent clip as a few small frames per sampler step (tools/fusion/strip_preview.py): the shot slot loops it = a MOVING preview
  const WAVE_EVENT = "motngrfx.waveform";      // websocket event type of nodes_fusion/music_preview.py (tools/fusion/waveform_preview.py EVENT)
  const SAMPLER_CLASSES = /^(SamplerCustomAdvanced|SamplerCustom|KSampler|KSamplerAdvanced)$/;

  // ------------------------------------------------------------------ pure helpers
  /** one ComfyUI binary websocket message -> {mime, bytes, meta} (image previews only), else null.
   *  type 1 PREVIEW_IMAGE: [u32 1][u32 format 1 jpeg | 2 png][bytes]; type 4 PREVIEW_IMAGE_WITH_METADATA: [u32 4][u32 n][n bytes JSON {node_id, real_node_id, prompt_id, image_type...}][bytes]. */
  function parseFrame(buf) {
    if (!buf || buf.byteLength < 9) return null;
    const dv = new DataView(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength), u8 = new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength);
    const type = dv.getUint32(0, false);
    let meta = null, off;
    if (type === 1) off = 8;
    else if (type === 4) {
      const n = dv.getUint32(4, false);
      if (8 + n > u8.length) return null;
      try { meta = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n))); } catch (_) { meta = null; }
      off = 8 + n;
    } else return null;
    const bytes = u8.subarray(off);
    if (bytes.length < 4) return null;
    let mime = "image/jpeg";
    if (bytes[0] === 0x89 && bytes[1] === 0x50) mime = "image/png";
    else if (bytes[0] === 0xff && bytes[1] === 0xd8) mime = "image/jpeg";
    else if (bytes[0] === 0x52 && bytes[1] === 0x49) mime = "image/webp";
    else if (type === 1 && dv.getUint32(4, false) === 2) mime = "image/png";
    return { mime, bytes, meta };
  }

  /** a graph node (title, class) -> what it produces for the timeline, or null.
   *  {kind: "still"|"clip"|"music", shot, layer} - layer: motion (the clean LTX clip) | type (the text pass) | matte (Alpha-Gen) | plate (clean plate) | angle */
  function classify(title, cls) {
    const t = String(title || ""), c = String(cls || "");
    if (!SAMPLER_CLASSES.test(c)) return null;
    let m;
    if (/ACE sampler|text-to-music/i.test(t)) return { kind: "music", shot: null, layer: "music" };
    if ((m = /^Angle sampler \[angle place (\d+)\]/.exec(t))) return { kind: "still", shot: +m[1], layer: "angle" };
    if ((m = /^Still sampler(?: \[place (\d+)\])?\s*$/.exec(t))) return { kind: "still", shot: m[1] ? +m[1] : 1, layer: "still" };
    if ((m = /\[(alpha |plate )?s(\d+)(?: (clean|text))?\]\s*$/.exec(t))) {
      const layer = m[1] === "alpha " ? "matte" : m[1] === "plate " ? "plate" : m[3] === "text" ? "type" : "motion";
      return { kind: "clip", shot: +m[2], layer };
    }
    return null;
  }

  /** {nodeId: classify(...)} over an API-format prompt (only the nodes that make a timeline item), + the highest shot number seen. */
  function mapPrompt(prompt) {
    const byNode = {};
    let maxShot = 0;
    for (const [id, n] of Object.entries(prompt || {})) {
      const c = classify(n && n._meta && n._meta.title, n && n.class_type);
      if (!c) continue;
      byNode[String(id)] = c;
      if (c.shot) maxShot = Math.max(maxShot, c.shot);
    }
    return { byNode, maxShot };
  }

  /** time weight of a graph node (a sampler or a model call costs far more than plumbing): the progress bar is weighted by this, so it moves like the wall clock. */
  function nodeWeight(n) {
    const c = String((n && n.class_type) || "");
    if (SAMPLER_CLASSES.test(c) || /^(MographGGUFVL|AILab_QwenVL_Advanced|MographLtxRead)$/.test(c)) return 20;
    if (/^(MographCompose|MographAssemble|MographTypePasses|MographArchive|MographFootageFlow|MographPivotCut|ImageUpscaleWithModel|MographSaveFinalUpscaled|VideoDepthAnythingProcess|MographDecodeClip|MographMusicT2MCut|VAEDecodeAudio)$/.test(c)) return 6;
    return 1;
  }
  const nodeFrac = (n) => (n.state === "done" || n.state === "cached" ? 1 : n.state === "running" ? (n.max ? Math.min(1, Math.max(0, n.value / n.max)) : 0.05) : 0);

  /** tracker nodes -> {segments: [{id, label, w, f, state}], overall 0..1}. Segments are the plain run stages (B.STAGES) weighted by their nodes' time weight. */
  function weightedStages(nodes, B) {
    const seg = {};
    for (const s of B.STAGES) seg[s.id] = { id: s.id, label: s.label, w: 0, wf: 0, run: false, n: 0, done: 0 };
    for (const n of Object.values(nodes || {})) {
      const s = seg[B.stageOf(n.title, n.class_type)], w = nodeWeight(n);
      s.w += w; s.wf += w * nodeFrac(n); s.n++;
      if (n.state === "done" || n.state === "cached") s.done++;
      if (n.state === "running") s.run = true;
    }
    const segments = B.STAGES.map((s) => seg[s.id]).filter((s) => s.w > 0).map((s) => ({ id: s.id, label: s.label, w: s.w, f: s.wf / s.w, state: s.done === s.n ? "done" : s.run || s.done ? "running" : "pending" }));
    const W = segments.reduce((a, s) => a + s.w, 0);
    return { segments, overall: W ? segments.reduce((a, s) => a + s.w * s.f, 0) / W : 0 };
  }

  /** estimate: elapsed s and overall 0..1 -> seconds left, or null while it would be a guess (too early). */
  function etaSeconds(elapsed, overall) {
    if (!(overall >= 0.06) || !(elapsed >= 15) || overall >= 1) return null;
    return Math.round((elapsed * (1 - overall)) / overall);
  }

  /** base64 of int8 (min, max) pairs -> Int8Array [min0, max0, min1, max1, ...]. */
  function envFromB64(b64) {
    const bin = atob(String(b64 || "")), out = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) { const v = bin.charCodeAt(i); out[i] = v > 127 ? v - 256 : v; }
    return out;
  }

  /** decoded audio channels (arrays) -> the same (min, max) pair layout as the live envelope, for seconds [t0, t1) in `bins` bins. */
  function envFromChannels(chans, sr, bins, t0, t1) {
    const out = new Int8Array(bins * 2), n = chans[0] ? chans[0].length : 0;
    const a = Math.max(0, Math.floor(t0 * sr)), b = Math.min(n, Math.max(a + 1, Math.floor(t1 * sr)));
    for (let k = 0; k < bins; k++) {
      const i0 = a + Math.floor(((b - a) * k) / bins), i1 = Math.max(i0 + 1, a + Math.floor(((b - a) * (k + 1)) / bins));
      let lo = 0, hi = 0;
      for (const ch of chans) for (let i = i0; i < i1 && i < n; i++) { const v = ch[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      out[2 * k] = Math.max(-127, Math.round(lo * 127)); out[2 * k + 1] = Math.min(127, Math.round(hi * 127));
    }
    return out;
  }

  /** the timeline model of a story: shots [{n, start, end, line, camera, transition}], words [{text, land, hold, shot}], cuts [{t, label, from, to}]. Falls back to `nShots` even shots over `length`. */
  function planTimeline(story, length, nShots, cameraLabel) {
    const L = Math.max(1, Number(length) || 12);
    let shots = [];
    if (story && Array.isArray(story.shots) && story.shots.length) {
      shots = story.shots.map((s, i) => ({ n: Number(s.n) || i + 1, start: Math.max(0, Number(s.start) || 0), end: Math.min(L, Number(s.end) || L), line: String(s.line || ""), camera: s.camera || "", transition: s.transition || "", image: s.image || null }));
    } else if (nShots > 0) {
      for (let i = 0; i < nShots; i++) shots.push({ n: i + 1, start: (L * i) / nShots, end: (L * (i + 1)) / nShots, line: "", camera: "", transition: "", image: null });
    }
    const words = [];
    const hits = story && Array.isArray(story.hits) ? story.hits.filter((h) => h && h.text && Number.isFinite(Number(h.land))) : [];
    if (hits.length) {
      for (const h of hits) {
        const land = Number(h.land), sh = shots.find((s) => land >= s.start - 1e-6 && land < s.end - 1e-6) || shots[shots.length - 1];
        words.push({ text: String(h.text), land, hold: Number.isFinite(Number(h.hold_until)) ? Number(h.hold_until) : land + 0.75, shot: sh ? sh.n : null });
      }
    } else {
      for (const s of shots) if (s.line) words.push({ text: s.line, land: s.start + (s.end - s.start) * 0.15, hold: s.end, shot: s.n });
    }
    const cuts = [];
    for (let i = 0; i + 1 < shots.length; i++) {
      const a = shots[i], b = shots[i + 1], raw = b.transition || a.camera || "";
      cuts.push({ t: a.end, from: a.n, to: b.n, label: raw && raw !== "none" && raw !== "hold" && raw !== "as filmed" ? (cameraLabel ? cameraLabel(raw) : raw) : "cut" });
    }
    return { length: L, shots, words, cuts };
  }

  // ------------------------------------------------------------------ the view
  /** host = {el, svg, mmss, B, story(), length(), picUrl(shot), cameraLabel(c), audioSource() -> {blob, offset} | null, finalVideo() -> HTMLVideoElement | null, viewUrl(f)} */
  function create(host) {
    const { el, B } = host;
    const V = { root: null, active: false, mode: "image", status: "idle", plan: null, slots: {}, map: { byNode: {}, maxShot: 0 }, cur: null, t0: 0, tEnd: 0, stack: [], urls: [],
                music: null, collapsed: false, userToggled: false, nodes: null, ownProg: null, live: null, tickTimer: null, headText: "", frames: 0, previewDraws: 0, archive: null };

    // ---- DOM
    const root = el("section", { class: "bv", "data-state": "idle", hidden: true, "aria-label": "Build view" });
    const head = el("div", { class: "bv-head" });
    const title = el("b", { class: "bv-title" }, "Build");
    const what = el("span", { class: "bv-what" }, "");
    const clock = el("span", { class: "bv-clock" }, "");
    const liveTile = el("canvas", { class: "bv-live", width: 54, height: 96, title: "the picture being made right now" });
    const toggle = el("button", { class: "link-btn bv-toggle", type: "button", "aria-expanded": "true" }, "Hide");
    const detailsBtn = el("button", { class: "link-btn bv-detailsbtn", type: "button", "aria-expanded": "false", title: "The job log / the per-step list" }, "Details");
    head.append(title, what, el("span", { class: "spacer" }), clock, liveTile, detailsBtn, toggle);
    const detailsWrap = el("div", { class: "bv-details", hidden: true });
    const endWrap = el("div", { class: "bv-end" });
    const bar = el("div", { class: "bv-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100" });
    const pct = el("span", { class: "bv-pct" }, "0%");
    const barRow = el("div", { class: "bv-barrow" }, bar, pct);
    const body = el("div", { class: "bv-body" });
    const tl = el("div", { class: "bv-tl" });
    const ruler = el("div", { class: "bv-track bv-ruler" });
    const laneShots = el("div", { class: "bv-track bv-shots" });
    const laneCuts = el("div", { class: "bv-track bv-cuts" });
    const laneWords = el("div", { class: "bv-track bv-words" });
    const laneMusic = el("div", { class: "bv-track bv-music" });
    const mcanvas = el("canvas", { class: "bv-wave", width: 900, height: 64 });
    const mfill = el("i", { class: "bv-mfill" });
    const mlabel = el("span", { class: "bv-mlabel" }, "");
    laneMusic.append(mcanvas, mfill, mlabel);
    const playhead = el("div", { class: "bv-playhead", hidden: true });
    const lane = (name, cls, track) => el("div", { class: "bv-lane " + cls }, el("span", { class: "bv-lab" }, name), track);
    tl.append(lane("", "bv-l-ruler", ruler), lane("Shots", "bv-l-shots", laneShots), lane("Cuts", "bv-l-cuts", laneCuts), lane("Words", "bv-l-words", laneWords), lane("Music", "bv-l-music", laneMusic), playhead);
    const stackRow = el("div", { class: "bv-stack" });
    const stackWrap = el("div", { class: "bv-stackwrap" }, el("span", { class: "bv-lab" }, "Made"), stackRow);
    body.append(tl, stackWrap);
    root.append(head, barRow, endWrap, detailsWrap, body);
    V.root = root;

    toggle.addEventListener("click", () => { V.userToggled = true; setCollapsed(!V.collapsed); });
    detailsBtn.addEventListener("click", () => toggleDetails());
    function toggleDetails(on) {
      V.details = on === undefined ? !V.details : !!on;
      detailsWrap.hidden = !V.details || !detailsWrap.firstChild;
      detailsBtn.textContent = V.details ? "Hide details" : "Details";
      detailsBtn.setAttribute("aria-expanded", V.details ? "true" : "false");
    }
    /** the node shown behind Details (the per-node list / the job log); the page rebuilds it per view, the panel just hosts it. */
    function setDetails(node) { detailsWrap.textContent = ""; if (node) detailsWrap.append(node); detailsWrap.hidden = !V.details || !detailsWrap.firstChild; }
    /** the error box, notes and re-run buttons of an ended run: they stay visible even when the panel is collapsed. */
    function setEnd(nodes) { endWrap.textContent = ""; for (const n of nodes || []) endWrap.append(n); endWrap.hidden = !endWrap.firstChild; }
    function setCollapsed(c) {
      V.collapsed = !!c;
      root.classList.toggle("collapsed", V.collapsed);
      toggle.textContent = V.collapsed ? "Show" : "Hide";
      toggle.setAttribute("aria-expanded", V.collapsed ? "false" : "true");
    }

    // ---- lifecycle
    function reset() {
      for (const u of V.urls) { try { URL.revokeObjectURL(u); } catch (_) { /* gone */ } }
      V.urls = [];
      for (const s of Object.values(V.slots)) { clearTimeout(s.timer); s.pendingFrame = null; closeStrip(s); }
      closeStrip(V.liveT); V.liveT = { strip: null, gen: 0 }; clearInterval(V.stripTimer); V.stripTimer = null;
      V.slots = {}; V.stack = []; V.music = null; V.musicFetch = false; V.cur = null; V.archive = null; V.live = null; V.frames = 0; V.previewDraws = 0; V.nodes = null; V.ownProg = null;
      clearInterval(V.tickTimer); V.tickTimer = null;
      renderStack();
    }

    /** a run starts. opts: {mode: "image" | "video", prompt?, nodes? (tracker nodes), shots?, soundtrack?} */
    function start(opts) {
      reset();
      V.mode = opts.mode || "image"; V.active = true; V.status = "running"; V.t0 = performance.now() / 1000; V.tEnd = 0; V.userToggled = false;
      V.nodes = opts.nodes || null;
      setEnd([]); setDetails(null);
      V.map = opts.prompt ? mapPrompt(opts.prompt) : { byNode: {}, maxShot: 0 };
      V.plan = planTimeline(host.story(), host.length(), Math.max(opts.shots || 0, V.map.maxShot), host.cameraLabel);
      V.soundtrack = opts.soundtrack || "";
      V.music = { env: null, prev: null, bins: 0, seconds: 0, step: 0, total: 0, real: false, samplerDone: false, hasLive: false, source: "" };
      root.hidden = false; root.dataset.state = "running";
      setCollapsed(false);
      buildTimeline();
      title.textContent = "Build";
      what.textContent = "Starting...";
      V.tickTimer = setInterval(tick, 250);
      tick();
      const src = host.audioSource && host.audioSource();                       // the user's track / the video's own sound: a real waveform from the first second
      if (src && src.blob) decodeSource(src.blob, src.offset || 0, src.kind || "your track");
    }

    function end(status) {
      if (!V.active && V.status !== "running") return;
      V.active = false; V.status = status; V.tEnd = performance.now() / 1000;
      clearInterval(V.tickTimer); V.tickTimer = null;
      root.dataset.state = status;
      clearInterval(V.stripTimer); V.stripTimer = null;                          // the strips stop looping; every slot keeps its last frame
      for (const s of Object.values(V.slots)) flushFrame(s);
      tick();
      what.textContent = status === "done" ? "Done" : status === "cancelled" ? "Stopped" : status === "error" ? "Stopped by an error (details below)" : "";
      if (status === "error") setCollapsed(false);
      if (status === "done" && !V.userToggled) setCollapsed(true);
      bindPlayhead();
    }

    // ---- timeline construction
    const pctOf = (t) => (100 * t) / V.plan.length;
    function buildTimeline() {
      const P = V.plan;
      for (const t of [ruler, laneShots, laneCuts, laneWords]) t.textContent = "";
      const step = P.length <= 12 ? 1 : P.length <= 30 ? 2 : 5;
      for (let t = 0; t <= P.length + 1e-6; t += step) ruler.append(el("span", { class: "bv-tick", style: `left:${pctOf(t)}%` }, host.mmss(t)));
      for (const s of P.shots) {
        const slot = { n: s.n, shot: s, phase: "", frac: 0, done: {}, state: "empty", dom: null, canvas: null, last: 0, timer: null, pendingFrame: null, hasPreview: false, clip: null, still: null };
        const pic = el("img", { class: "bv-pic", alt: "", draggable: "false" });
        const canvas = el("canvas", { class: "bv-prev", width: 2, height: 2 });
        const vid = el("video", { class: "bv-clipv", muted: true, loop: true, playsinline: true, preload: "none" });
        vid.muted = true;
        const ring = svgRing();
        const chip = el("span", { class: "bv-chip" }, "");
        const num = el("span", { class: "bv-num" }, "S" + s.n);
        const dom = el("div", { class: "bv-slot", "data-shot": s.n, "data-state": "empty", style: `left:${pctOf(s.start)}%;width:${pctOf(s.end - s.start)}%` },
          el("div", { class: "bv-slotin" }, pic, canvas, vid, num, chip, ring.node));
        Object.assign(slot, { dom, pic, canvas, vid, ring, chip });
        const url = host.picUrl(s);
        if (url) { pic.src = url; slot.state = "pic"; dom.dataset.state = "pic"; }
        V.slots[s.n] = slot;
        laneShots.append(dom);
      }
      for (const c of P.cuts) laneCuts.append(el("span", { class: "bv-cut", style: `left:${pctOf(c.t)}%`, "data-from": c.from, "data-to": c.to, title: `S${c.from} to S${c.to}: ${c.label}` }, el("i", {}), el("em", {}, c.label)));
      for (const w of P.words) laneWords.append(el("span", { class: "bv-word", "data-shot": w.shot == null ? "" : w.shot, style: `left:${pctOf(w.land)}%;max-width:${Math.max(8, pctOf(Math.max(1.2, w.hold - w.land)))}%`, title: `${w.text} lands at ${w.land.toFixed(1)} s` }, w.text));
      V.music.bins = 0;
      musicFill();
      drawWave();
      mlabel.textContent = V.soundtrack === "track" ? "your track" : "waiting for the music";
      laneMusic.classList.toggle("empty", true);
    }
    function svgRing() {
      const C = 2 * Math.PI * 10;
      const node = el("span", { class: "bv-ring", html: `<svg viewBox="0 0 26 26" width="26" height="26"><circle cx="13" cy="13" r="10" class="bg"/><circle cx="13" cy="13" r="10" class="fg" stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${C.toFixed(2)}" transform="rotate(-90 13 13)"/></svg><b></b>` });
      const fg = node.querySelector(".fg"), b = node.querySelector("b");
      return { node, set(f, done) { fg.setAttribute("stroke-dashoffset", (C * (1 - Math.max(0, Math.min(1, f)))).toFixed(2)); node.classList.toggle("done", !!done); b.textContent = done ? "✓" : ""; } };
    }

    // ---- slots
    const PHASE_TEXT = { still: "picture", angle: "picture", motion: "motion", type: "type pass", matte: "matte", plate: "plate" };
    function setSlot(slot, patch) {
      Object.assign(slot, patch);
      const live = slot.phase && !slot.done[slot.phase];
      slot.dom.dataset.state = slot.clip ? "clip" : slot.hasPreview ? (live ? "live" : "made") : slot.state;
      slot.dom.dataset.phase = slot.phase || "";
      slot.chip.textContent = slot.phase ? (PHASE_TEXT[slot.phase] || slot.phase) + (slot.done[slot.phase] ? " done" : slot.steps ? ` ${slot.stepNow || 0}/${slot.steps}` : "") : "";
      slot.ring.set(slot.done[slot.phase] ? 1 : slot.frac, !!(slot.phase && slot.done[slot.phase]));
      slot.dom.classList.toggle("active", !!live && V.active);
    }
    function slotFor(info) { return info && info.shot ? V.slots[info.shot] : null; }

    /** draw the newest preview frame of a slot, at most PREVIEW_FPS times a second (the rest are dropped; only the newest is kept). */
    function queueFrame(slot, blob, whole) {
      slot.pendingFrame = { blob, whole };
      const wait = Math.max(0, 1000 / PREVIEW_FPS - (performance.now() - slot.last));
      if (wait <= 0) flushFrame(slot);
      else if (!slot.timer) slot.timer = setTimeout(() => { slot.timer = null; flushFrame(slot); }, wait);
    }
    function flushFrame(slot) {
      const p = slot.pendingFrame;
      if (!p) return;
      slot.pendingFrame = null; slot.last = performance.now();
      if (typeof createImageBitmap !== "function") return;
      createImageBitmap(p.blob).then((bmp) => {
        paintBitmap(slot.canvas, bmp);
        if (p.whole) paintBitmap(liveTile, bmp, true);
        V.previewDraws++;
        slot.dom.dataset.frames = String((+slot.dom.dataset.frames || 0) + 1);
        if (!slot.hasPreview) setSlot(slot, { hasPreview: true });
        try { bmp.close(); } catch (_) { /* old browser */ }
      }).catch(() => {});
    }
    function paintBitmap(canvas, bmp, fit) {
      if (!fit) { if (canvas.width !== bmp.width || canvas.height !== bmp.height) { canvas.width = bmp.width; canvas.height = bmp.height; } canvas.getContext("2d").drawImage(bmp, 0, 0); return; }
      const g = canvas.getContext("2d"), s = Math.max(canvas.width / bmp.width, canvas.height / bmp.height), w = bmp.width * s, h = bmp.height * s;
      g.fillStyle = "#000"; g.fillRect(0, 0, canvas.width, canvas.height); g.drawImage(bmp, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    }

    /** a binary websocket message (a sampler preview). `current` = the id of the executing node (type-1 frames carry no node). */
    function binary(buf) {
      const f = parseFrame(buf);
      if (!f) return false;
      if (f.meta && f.meta.prompt_id && V.promptId && f.meta.prompt_id !== V.promptId) return false;     // somebody else's run
      V.frames++;
      const id = String((f.meta && (f.meta.real_node_id || f.meta.node_id || f.meta.display_node_id)) || V.cur || "");
      const info = V.map.byNode[id] || null;
      const blob = new Blob([f.bytes], { type: f.mime });
      if (info && info.kind === "music") return true;                          // an audio sampler has no picture
      const slot = slotFor(info);
      if (slot) { if (!slot.phase || slot.phase !== info.layer) setSlot(slot, { phase: info.layer }); queueFrame(slot, blob, true); }
      else {                                                                     // video mode / unmapped sampler: the live tile only
        createImageBitmap(blob).then((bmp) => { paintBitmap(liveTile, bmp, true); V.previewDraws++; try { bmp.close(); } catch (_) {} }).catch(() => {});
      }
      liveTile.classList.add("on");
      return true;
    }

    // ---- MOVING previews: the strip event = all latent frames of the shot's clip at this sampler step; the slot loops them (<= PREVIEW_FPS) until the next, more refined strip arrives
    const b64bytes = (b) => { const bin = atob(b), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
    function closeStrip(o) { if (o && o.strip) { for (const b of o.strip.bmps) { try { b.close(); } catch (_) { /* old browser */ } } o.strip = null; } }
    function stripEvent(d) {
      if (!d || !Array.isArray(d.frames) || !d.frames.length || typeof createImageBitmap !== "function") return;
      if (d.prompt_id && V.promptId && d.prompt_id !== V.promptId) return;
      const id = String(d.node === null || d.node === undefined ? V.cur || "" : d.node), info = V.map.byNode[id] || null, slot = slotFor(info), target = slot || V.liveT;
      const gen = (target.gen = (target.gen || 0) + 1), mime = d.fmt === "png" ? "image/png" : "image/jpeg";
      V.strips = (V.strips || 0) + 1;
      Promise.all(d.frames.map((f) => createImageBitmap(new Blob([b64bytes(f)], { type: mime })))).then((bmps) => {
        if (V.liveT !== undefined && target.gen !== gen) { bmps.forEach((b) => { try { b.close(); } catch (_) {} }); return; }     // an older strip that finished decoding late
        if (!V.active && V.status !== "running") { bmps.forEach((b) => { try { b.close(); } catch (_) {} }); return; }
        const keep = target.strip && target.strip.idx % target.strip.bmps.length;
        closeStrip(target);
        target.strip = { bmps, idx: keep && keep < bmps.length ? keep : 0, rate: Math.max(1, Math.min(PREVIEW_FPS, Number(d.rate) || PREVIEW_FPS)), last: 0 };       // keep the loop position: no jump back to frame 0 at every step
        if (slot && info && slot.phase !== info.layer) setSlot(slot, { phase: info.layer });
        if (!V.stripTimer) V.stripTimer = setInterval(stripLoop, 40);
      }).catch(() => {});
    }
    function stripLoop() {
      const t = performance.now();
      for (const s of Object.values(V.slots)) {
        const st = s.strip;
        if (!st || s.clip || t - st.last < 1000 / st.rate) continue;
        st.last = t;
        const bmp = st.bmps[st.idx]; st.idx = (st.idx + 1) % st.bmps.length;
        paintBitmap(s.canvas, bmp);
        if (s.dom.classList.contains("active")) { paintBitmap(liveTile, bmp, true); liveTile.classList.add("on"); }
        V.previewDraws++; s.dom.dataset.frames = String((+s.dom.dataset.frames || 0) + 1); s.dom.dataset.strip = String(st.bmps.length);
        if (!s.hasPreview) setSlot(s, { hasPreview: true });
      }
      const lt = V.liveT.strip;
      if (lt && t - lt.last >= 1000 / lt.rate) { lt.last = t; const bmp = lt.bmps[lt.idx]; lt.idx = (lt.idx + 1) % lt.bmps.length; paintBitmap(liveTile, bmp, true); liveTile.classList.add("on"); V.previewDraws++; }
    }

    // ---- JSON events (image mode)
    function event(msg) {
      const d = (msg && msg.data) || {};
      if (msg.type === STRIP_EVENT) return stripEvent(d);
      if (msg.type === WAVE_EVENT) return waveEvent(d);
      const id = d.node === null || d.node === undefined ? null : String(d.node);
      if (msg.type === "executing") {
        if (V.cur && V.cur !== id) finishNode(V.cur);
        V.cur = id;
        const info = id && V.map.byNode[id];
        if (info && info.kind === "music") { V.music.running = true; mlabel.textContent = "making the music"; }
        const slot = slotFor(info);
        if (slot) { if (slot.phase !== info.layer) closeStrip(slot); setSlot(slot, { phase: info.layer, frac: 0, steps: 0, stepNow: 0 }); }     // a new phase: its own strip replaces the last one
      } else if (msg.type === "progress" && id) {
        const info = V.map.byNode[id];
        if (!info) return;
        const f = d.max ? d.value / d.max : 0;
        if (info.kind === "music") { V.music.step = d.value; V.music.total = d.max; musicFill(); if (d.max && d.value >= d.max) finishNode(id); return; }
        const slot = slotFor(info);
        if (slot) { setSlot(slot, { phase: info.layer, frac: f, steps: d.max, stepNow: d.value }); if (d.max && d.value >= d.max) finishNode(id); }
      } else if (msg.type === "execution_cached") {
        for (const nid of d.nodes || []) { const info = V.map.byNode[String(nid)]; if (info) markDone(info); }
      }
    }
    function finishNode(id) { const info = V.map.byNode[id]; if (info) markDone(info); }
    function markDone(info) {
      if (info.kind === "music") { V.music.samplerDone = true; V.music.running = false; V.music.step = V.music.total || V.music.step; musicFill(); mlabel.textContent = V.music.real ? "track ready" : "song made, finishing"; return; }
      const slot = slotFor(info);
      if (!slot || slot.done[info.layer]) return;
      slot.done[info.layer] = true;
      setSlot(slot, { frac: 1 });
      if (info.layer === "type") for (const w of laneWords.querySelectorAll(`.bv-word[data-shot="${slot.n}"]`)) w.classList.add("on");
      if (slot.hasPreview && (info.layer === "motion" || info.layer === "still")) addStackFromCanvas(slot, info.layer);
    }

    /** executed outputs by node title (called for ws events and for history). */
    function output(title, out) {
      const t = String(title || ""), files = (B.outputFiles ? B.outputFiles(out) : []) || [];
      if (/^OUT stills$/.test(t)) {
        files.forEach((f, i) => { const slot = V.slots[i + 1], url = host.viewUrl(f); if (slot) { slot.still = url; slot.pic.src = url; setSlot(slot, { state: "pic" }); } pushStack({ kind: "still", label: "Picture " + (i + 1), url }); });
      } else if (/^OUT still$/.test(t)) {
        files.forEach((f) => { const url = host.viewUrl(f), slot = V.slots[1]; if (slot && !slot.still) { slot.still = url; slot.pic.src = url; setSlot(slot, { state: "pic" }); } pushStack({ kind: "still", label: "First frame", url }); });
      } else if (/^OUT type$/.test(t)) {
        files.forEach((f, i) => pushStack({ kind: "type", label: "Type " + (i + 1), url: host.viewUrl(f) }));
      } else if (/^OUT final/.test(t)) {
        files.forEach((f) => pushStack({ kind: "final", label: /1080/.test(t) ? "Final 1080" : "Video", url: host.viewUrl(f), video: true }));
      } else if (out && Array.isArray(out.text)) {
        for (const x of out.text) {
          let j = null; try { j = JSON.parse(x); } catch (_) { /* plain text */ }
          if (j && j.dir && Array.isArray(j.files)) archive(j);
        }
      }
    }
    /** MographArchive's ui text {dir, stem, files, shots}: the finished shot clips and the song are files under dir. */
    function archive(j) {
      V.archive = j;
      const sub = j.dir;
      for (const name of j.files) {
        const m = /_s(\d+)_(clean|clip)\.mp4$/.exec(name);
        if (m) {
          const slot = V.slots[+m[1]], url = host.viewUrl({ filename: name, subfolder: sub, type: "output" });
          if (slot && !slot.clip) { slot.clip = url; slot.vid.src = url; slot.vid.preload = "metadata"; slot.vid.play().catch(() => {}); setSlot(slot, {}); }
          pushStack({ kind: "clip", label: "Shot " + m[1], url, video: true });
        }
      }
      const mus = j.files.find((n) => /_music\.(wav|flac|mp3|ogg)$/.test(n));
      if (mus && !V.music.real && !V.musicFetch) {
        V.musicFetch = true;
        const url = host.viewUrl({ filename: mus, subfolder: sub, type: "output" });
        pushStack({ kind: "music", label: "Music", url });
        fetch(url).then((r) => (r.ok ? r.blob() : null)).then((b) => { if (b) decodeSource(b, 0, "the song"); }).catch(() => {});
      }
    }

    // ---- stack
    function pushStack(item) {
      if (V.stack.some((s) => s.url === item.url)) return;
      V.stack.unshift(item);
      renderStack();
    }
    function addStackFromCanvas(slot, layer) {
      if (!slot.canvas.toBlob) return;
      slot.canvas.toBlob((b) => {
        if (!b) return;
        const url = URL.createObjectURL(b);
        V.urls.push(url);
        pushStack({ kind: "frame", label: `Shot ${slot.n} ${layer === "still" ? "picture" : "motion"}`, url });
      }, "image/png");
    }
    function renderStack() {
      stackRow.textContent = "";
      for (const it of V.stack.slice(0, 40)) {
        const media = it.video ? Object.assign(el("video", { src: it.url, muted: true, loop: true, playsinline: true, preload: "metadata" }), { muted: true })
          : it.kind === "music" ? el("span", { class: "bv-note" }, "♪") : el("img", { src: it.url, alt: "", draggable: "false" });
        if (it.video) { media.addEventListener("mouseenter", () => media.play().catch(() => {})); media.addEventListener("mouseleave", () => media.pause()); }
        stackRow.append(el("div", { class: "bv-item " + it.kind, title: it.label }, media, el("small", {}, it.label)));
      }
      stackWrap.dataset.count = String(V.stack.length);
    }

    // ---- music
    function waveEvent(d) {
      const M = V.music;
      if (!M || !d.peaks) return;
      if (d.prompt_id && V.promptId && d.prompt_id !== V.promptId) return;
      M.prev = M.env; M.env = envFromB64(d.peaks); M.bins = M.env.length / 2; M.seconds = Number(d.seconds) || M.seconds; M.hasLive = true;
      M.step = d.step || M.step; M.total = d.total || M.total;
      laneMusic.classList.remove("empty");
      mlabel.textContent = "making the music " + (M.total ? `${M.step}/${M.total}` : `step ${M.step}`);
      musicFill(); drawWave();
    }
    function musicFill() { const M = V.music; mfill.style.width = M.total ? Math.min(100, (100 * M.step) / M.total) + "%" : "0"; laneMusic.dataset.phase = M.real ? "real" : M.samplerDone ? "made" : M.hasLive ? "live" : "wait"; }
    function drawWave() {
      const M = V.music, c = mcanvas, w = (c.width = Math.max(300, Math.round(laneMusic.clientWidth || 900))), h = (c.height = Math.max(40, Math.round(laneMusic.clientHeight || 64))), g = c.getContext("2d");
      g.clearRect(0, 0, w, h);
      const mid = h / 2;
      if (!M || !M.env) {                                                       // placeholder: a flat dashed centre line, not a fake wave
        g.strokeStyle = "rgba(138,133,153,.45)"; g.setLineDash([4, 5]); g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke(); g.setLineDash([]);
        return;
      }
      const L = V.plan.length, frac = M.real || !M.seconds ? 1 : Math.min(1, L / M.seconds), nb = Math.max(1, Math.floor(M.bins * frac));   // generated song = Length + 4 s: draw the first Length seconds
      const draw = (env, style) => {
        g.fillStyle = style;
        for (let x = 0; x < w; x++) {
          const k = Math.min(nb - 1, Math.floor((x / w) * nb)), lo = env[2 * k] / 127, hi = env[2 * k + 1] / 127;
          const y0 = mid - hi * (mid - 2), y1 = mid - lo * (mid - 2);
          g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
        }
      };
      if (M.prev && !M.real && M.prev.length === M.env.length) draw(M.prev, "rgba(138,133,153,.55)");     // the step before, dim: the change between steps is the refinement
      const live = !M.real && !M.samplerDone;
      draw(M.env, live ? "rgba(242,255,68,.92)" : "#f2ff44");
    }
    function setRealWave(env, seconds, label) {
      const M = V.music;
      M.prev = null; M.env = env; M.bins = env.length / 2; M.seconds = seconds; M.real = true; M.hasLive = M.hasLive || false;
      laneMusic.classList.remove("empty"); mlabel.textContent = label; musicFill(); drawWave();
    }
    function decodeSource(blob, offset, label) {
      const Ctx = typeof OfflineAudioContext !== "undefined" ? OfflineAudioContext : (typeof webkitOfflineAudioContext !== "undefined" ? webkitOfflineAudioContext : null);
      if (!Ctx) return Promise.resolve(false);
      return blob.arrayBuffer().then((ab) => new Ctx(1, 1, 44100).decodeAudioData(ab)).then((buf) => {
        const chans = []; for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
        const L = V.plan.length, t0 = Math.min(offset, Math.max(0, buf.duration - 1)), t1 = Math.min(buf.duration, t0 + L);
        setRealWave(envFromChannels(chans, buf.sampleRate, 900, t0, t1), t1 - t0, label);
        V.music.source = label;
        return true;
      }).catch(() => false);
    }
    // the final file's own sound, once there is a final video and no better source
    function decodeFinal(url) {
      if (V.music.real || V.musicFetch) return;
      V.musicFetch = true;
      fetch(url).then((r) => (r.ok ? r.blob() : null)).then((b) => (b ? decodeSource(b, 0, "the finished track") : null)).catch(() => {});
    }

    // ---- video mode (the own-video job is polled, its sampler events carry no titles)
    function ownState(j) {                                                      // j = the job status {stage, ...}; called by the poll
      V.ownStage = j && j.stage;
      if (j && j.stage === "music" && !V.music.hasLive) mlabel.textContent = "making the music";
    }
    function setProgress(prog) { V.ownProg = prog; }                            // video mode: {segments, overall, label}
    function setNodes(n) { V.nodes = n; }
    function ownEvent(msg) {
      const d = (msg && msg.data) || {};
      if (msg.type === STRIP_EVENT) return stripEvent(d);
      if (msg.type === WAVE_EVENT) return waveEvent(d);
      if (msg.type === "progress" && d.max) { V.live = { value: d.value, max: d.max }; }
    }

    // ---- progress bar + clock
    function paintBar(prog) {
      const segs = prog.segments;
      if (bar.children.length !== segs.length || [...bar.children].some((c, i) => c.dataset.id !== segs[i].id)) {
        bar.textContent = "";
        for (const s of segs) bar.append(el("span", { class: "bv-seg", "data-id": s.id, style: `flex:${Math.max(1, Math.round(s.w))} 1 0` }, el("i", {}), el("small", {}, s.label)));
      }
      segs.forEach((s, i) => { const c = bar.children[i]; c.firstChild.style.width = Math.round(100 * s.f) + "%"; c.dataset.state = s.state; });
      const p = V.status === "done" ? 100 : Math.round(100 * prog.overall);
      pct.textContent = p + "%"; bar.setAttribute("aria-valuenow", String(p));
      root.style.setProperty("--bv-p", String(p / 100));
    }
    function tick() {
      let prog = null;
      if (V.mode === "image" && V.nodes) { const nodes = typeof V.nodes === "function" ? V.nodes() : V.nodes; prog = weightedStages(nodes, B); prog.label = ""; }
      else if (V.ownProg) prog = typeof V.ownProg === "function" ? V.ownProg() : V.ownProg;
      if (prog) paintBar(prog);
      const el_s = ((V.tEnd || performance.now() / 1000) - V.t0), left = prog && V.active ? etaSeconds(el_s, prog.overall) : null;
      clock.textContent = host.mmss(el_s) + (V.active ? (left !== null ? ` · about ${host.mmss(left)} left (estimate)` : "") : (V.status === "done" ? " · took" : ""));
      if (V.active) {
        const slotActive = Object.values(V.slots).find((s) => s.dom.classList.contains("active"));
        what.textContent = V.mode === "video" && host.ownDetail ? host.ownDetail()
          : slotActive ? `Shot ${slotActive.n}: ${PHASE_TEXT[slotActive.phase] || slotActive.phase}${slotActive.steps ? ` ${slotActive.stepNow}/${slotActive.steps}` : ""}`
          : V.music && V.music.running ? "Making the music" : (prog && prog.label) || "Working...";
      }
    }

    // ---- playhead
    let phTimer = null, phVideo = null;
    function bindPlayhead() {
      clearInterval(phTimer);
      phTimer = setInterval(() => {
        const v = host.finalVideo && host.finalVideo();
        if (!v || !V.plan) { playhead.hidden = true; return; }
        if (v !== phVideo) { phVideo = v; }
        const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : V.plan.length;
        playhead.hidden = false; playhead.style.setProperty("--p", String(Math.min(1, v.currentTime / dur)));
      }, 100);
    }

    if (typeof ResizeObserver === "function") { let lastW = 0; new ResizeObserver(() => { const w = laneMusic.clientWidth; if (w && w !== lastW) { lastW = w; drawWave(); } }).observe(laneMusic); }   // the lane grows with the window: redraw sharp
    function mount(parent, before) {
      if (before) parent.insertBefore(root, before); else parent.append(root);
      if (V.plan) { drawWave(); if (!V.active && V.status === "done") bindPlayhead(); }
    }
    function currentNodeId(id) { V.cur = id == null ? null : String(id); }
    function setPromptId(id) { V.promptId = id; }

    Object.assign(V, { start, end, event, ownEvent, binary, output, mount, setDetails, setEnd, toggleDetails, ownState, setProgress, setNodes, tick, currentNodeId, setPromptId, decodeFinal, decodeSource, setCollapsed, drawWave });
    Object.defineProperty(V, "slotsList", { get() { return Object.values(V.slots); } });
    return V;
  }

  return { create, STRIP_EVENT, parseFrame, classify, mapPrompt, nodeWeight, weightedStages, etaSeconds, envFromB64, envFromChannels, planTimeline, PREVIEW_FPS, WAVE_EVENT };
});
