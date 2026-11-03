/* MOTNGRFX app: generic template binding + run tracking. No DOM; usable in the page (window.MographBind) and in node.
 *
 * A template is an API-format ComfyUI workflow whose nodes carry _meta.title. The sidecar (*.app.json) lists fields
 * ({id, kind, node_title, input_name, optional, default, options, remove_when_empty, max, min}) and outputs.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MographBind = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const clone = (o) => JSON.parse(JSON.stringify(o));

  function titleMap(graph) {
    const m = {};
    for (const [id, n] of Object.entries(graph)) {
      const t = n && n._meta && n._meta.title;
      if (t) m[t] = id;
    }
    return m;
  }

  /** Fields whose node exists in the template, in sidecar order. */
  function visibleFields(app, graph) {
    const tm = titleMap(graph);
    return (app.fields || []).filter((f) => f.kind === "media" || f.node_title in tm);   // hidden fields stay in the list (they still bind); the page just does not draw them   // media: the page box works even if some of its nodes are missing (old graphs)
  }

  function optionValue(o) { return o !== null && typeof o === "object" ? o.value : o; }
  function optionLabel(o) { return o !== null && typeof o === "object" ? (o.label ?? String(o.value)) : String(o); }

  function randomSeed() { return Math.floor(Math.random() * 4294967296); }

  function isEmpty(field, value) {
    if (field.kind === "image" || field.kind === "audio") return !value;
    if (field.kind === "media") return !value || (!value.video && !(value.images || []).length && !value.audio && !String(value.text || "").trim());
    if (field.kind === "seed") return value === "" || value === null || value === undefined;
    if (field.kind === "number") return value === "" || value === null || value === undefined || Number.isNaN(value);
    return value === undefined || value === null || String(value).trim() === "";
  }

  /** Coerce a UI value to what the node input expects. */
  function coerce(field, value) {
    switch (field.kind) {
      case "seed": {
        const n = Math.floor(Number(value));
        return Math.max(0, Math.min(4294967295, Number.isFinite(n) ? n : randomSeed()));
      }
      case "number": case "select": {
        const opt = (field.options || []).map(optionValue).find((o) => String(o) === String(value));
        let v = opt !== undefined ? opt : value;
        if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v)) && typeof opt === "number") v = Number(v);
        if (typeof v === "number") {
          if (field.max !== undefined) v = Math.min(v, field.max);
          if (field.min !== undefined) v = Math.max(v, field.min);
        }
        return v;
      }
      default:
        return String(value);
    }
  }

  /** Drop nodes by title and every input link that points at them. */
  function removeNodes(prompt, titles) {
    const tm = titleMap(prompt);
    const ids = new Set(titles.filter((t) => t in tm).map((t) => tm[t]));
    for (const id of ids) delete prompt[id];
    for (const n of Object.values(prompt)) {
      for (const [k, v] of Object.entries(n.inputs || {})) {
        if (Array.isArray(v) && v.length === 2 && ids.has(String(v[0]))) delete n.inputs[k];
      }
    }
    return [...ids];
  }

  // ------------------------------------------------------------------ media box (16_app/MEDIA_BOX.md)
  const MEDIA_LIMITS = { video: 1, image: 5, audio: 1 };
  const VIDEO_EXT = ["mp4", "mov", "webm", "mkv", "m4v", "avi"];
  const IMAGE_EXT = ["png", "jpg", "jpeg", "webp", "bmp", "gif"];
  const AUDIO_EXT = ["wav", "mp3", "flac", "m4a", "ogg", "opus", "aac"];

  /** "video" | "image" | "audio" | null, from the MIME type, then the extension. */
  function mediaKindOf(name, type) {
    const t = String(type || "").toLowerCase();
    if (t.startsWith("video/")) return "video";
    if (t.startsWith("image/")) return "image";
    if (t.startsWith("audio/")) return "audio";
    const ext = (String(name || "").split(".").pop() || "").toLowerCase();
    if (VIDEO_EXT.includes(ext)) return "video";
    if (IMAGE_EXT.includes(ext)) return "image";
    if (AUDIO_EXT.includes(ext)) return "audio";
    return null;
  }

  function mediaLimits(field) {
    return Object.assign({}, MEDIA_LIMITS, (field && field.limits) || {});
  }

  /**
   * Can `cand` ({name, type}) join `items` ([{kind}])? -> {ok, kind, reason}. The reason is a short message for the box.
   * v19: pictures are up to 5 still images XOR one video (never both); plus at most ONE audio file in either mode (the user's own track).
   * The first image / video decides the mode (an audio file alone leaves it open); removing the last picture frees it.
   */
  function checkMediaAdd(items, cand, limits) {
    limits = Object.assign({}, MEDIA_LIMITS, limits || {});
    const kind = mediaKindOf(cand.name, cand.type);
    const n = (k) => items.filter((i) => i.kind === k).length;
    if (!kind) return { ok: false, kind, reason: `${cand.name}: only video, image or audio files.` };
    if (kind === "video") {
      if (n("video") >= limits.video) return { ok: false, kind, reason: `Only ${limits.video} video allowed. Remove the current one first.` };
      if (n("image")) return { ok: false, kind, reason: "Remove the images to add a video." };
    } else if (kind === "image") {
      if (n("video")) return { ok: false, kind, reason: "Remove the video to add images." };
      if (n("image") >= limits.image) return { ok: false, kind, reason: `Only ${limits.image} images allowed.` };
    } else if (kind === "audio") {
      if (n("audio") >= limits.audio) return { ok: false, kind, reason: `Only ${limits.audio} audio file allowed.` };
    }
    return { ok: true, kind, reason: "" };
  }

  /**
   * Plan a whole drop / pick of `cands` ([{name, type}]) into `items` ([{kind}]) -> {adds, removes, msg}.
   *   adds    = [{i, kind}] candidate indexes to add, in drop order;
   *   removes = indexes into `items` that the drop REPLACES (the existing audio / video);
   *   msg     = one short message for the box ("" when everything went in).
   * Rules (user 2026-10-30): AUDIO and VIDEO replace the one already in (the FIRST of each kind in a drop is used, the rest is ignored with a note);
   * IMAGES fill the free slots in drop order, the ones that do not fit are refused with ONE warning (existing images are never replaced);
   * the two modes stay exclusive (a video vs still images; v19: ONE audio file is allowed with either, a second audio file replaces the first);
   * an empty box (or one holding only audio) takes the mode of the first picture file.
   */
  function planMediaDrop(items, cands, limits) {
    limits = Object.assign({}, MEDIA_LIMITS, limits || {});
    const has = (k) => items.some((i) => i.kind === k);
    let mode = has("video") ? "video" : (has("image") ? "images" : null);
    let nImg = items.filter((i) => i.kind === "image").length;
    const adds = [], removes = [], refused = [];
    let pickedV = false, pickedA = false, extraV = 0, extraA = 0, overflow = 0;
    cands.forEach((c, i) => {
      const kind = mediaKindOf(c.name, c.type);
      if (!kind) { refused.push(`${c.name}: only video, image or audio files.`); return; }
      if (kind === "video") {
        if (mode === "images") { refused.push("Remove the images to add a video."); return; }
        if (pickedV) { extraV++; return; }
        pickedV = true; mode = "video";
        items.forEach((it, x) => { if (it.kind === "video") removes.push(x); });
        adds.push({ i, kind });
      } else if (kind === "audio") {
        if (pickedA) { extraA++; return; }
        pickedA = true;
        items.forEach((it, x) => { if (it.kind === "audio") removes.push(x); });
        adds.push({ i, kind });
      } else {
        if (mode === "video") { refused.push("Remove the video to add images."); return; }
        mode = "images";
        if (nImg >= limits.image) { overflow++; return; }
        nImg++; adds.push({ i, kind });
      }
    });
    const msgs = [...new Set(refused)];
    if (overflow) msgs.push(`Box holds ${limits.image} images — ${overflow} not added. Remove some to add more.`);
    if (extraV) msgs.push(`One video at a time — used the first, ${extraV} ignored.`);
    if (extraA) msgs.push(`One audio file at a time — used the first, ${extraA} ignored.`);
    return { adds, removes: [...new Set(removes)], msg: msgs.join(" ") };
  }

  /** Ids V1 / I1..I5 / A1 for items in order. */
  function mediaIds(items) {
    const c = { video: 0, image: 0, audio: 0 };
    return items.map((i) => ({ video: "V", image: "I", audio: "A" }[i.kind]) + (++c[i.kind]));
  }

  /** MEDIA JSON (page -> `IN media`). items: [{kind, name}] with name = the uploaded file name. */
  function buildMediaJson(items, text, length, format, soundtrack) {
    const of = (k) => items.filter((i) => i.kind === k).map((i) => i.name);
    const n = Number(length);
    const st = soundtrack ? normSoundtrack(soundtrack) : "";
    return { video: of("video")[0] || null, images: of("image").slice(0, MEDIA_LIMITS.image), audio: of("audio")[0] || null,
             text: String(text || ""), length: Number.isFinite(n) && n > 0 ? n : 12, format: format || "9:16", ...(st ? { soundtrack: st } : {}) };
  }

  /** What a Read depends on: when this changes, the READ JSON is stale. items: [{kind, key}]. v19: the soundtrack ("video" | "track" | "generate") picks which audio is read. */
  function mediaSignature(items, text, length, soundtrack) {
    return JSON.stringify([items.map((i) => [i.kind, i.key]), String(text || "").trim(), String(length || "")].concat(soundtrack ? [normSoundtrack(soundtrack)] : []));
  }

  /** v19 SOUNDTRACK: "video" (the clip's own sound) | "track" (the user's audio file) | "generate" (new music). The old "keep" is "video". Anything else -> "". */
  function normSoundtrack(v) {
    const s = String(v === undefined || v === null ? "" : v).trim().toLowerCase();
    return s === "keep" ? "video" : (s === "video" || s === "track" || s === "generate" ? s : "");
  }

  /** EDIT MODE of a video job (2026-11-01): "overlay" (default: the picture is untouched, only type is composited) | "redesign" (with transitions: 3D cuts, effects). Anything else -> "overlay". */
  function normEdit(v) {
    const s = String(v === undefined || v === null ? "" : v).trim().toLowerCase().replace(/[\s-]+/g, "_");
    return s === "redesign" || s === "with_transitions" || s === "transitions" ? "redesign" : "overlay";
  }
  const EDIT_OPTIONS = [
    { val: "overlay", label: "Overlay only", title: "Your video stays exactly as it is: same picture, same cuts, same size and frame rate. Words animate over and behind it, landing when they are sung." },
    { val: "redesign", label: "With transitions", title: "Our full edit: 3D cuts, word-to-word transitions and picture effects. Still your size and frame rate." },
  ];

  /**
   * The soundtrack choice the box offers. items = [{kind}], chosen = the user's pick (any of normSoundtrack's values or ""), videoSound = false when the video is known to have no audio stream.
   * -> {options: [{val, label, enabled, title}], value}. Video mode: Video | My track | New music; image mode (no video): My track | New music.
   * "My track" needs an audio file in the box; "Video sound" needs the video to have sound. A pick that is not available falls back: video -> Video sound (else New music), images -> New music.
   */
  function soundtrackPlan(items, chosen, videoSound) {
    const hasV = items.some((i) => i.kind === "video"), hasA = items.some((i) => i.kind === "audio");
    const vOk = videoSound !== false;
    const options = [];
    if (hasV) options.push({ val: "video", label: "Video", enabled: vOk, title: vOk ? "Keep the video's own sound" : "this video has no sound" });
    options.push({ val: "track", label: "My track", enabled: hasA, title: hasA ? "Use the audio file you added" : "add an audio file to use it" });
    options.push({ val: "generate", label: "New music", enabled: true, title: "Write new music for the piece" });
    const c = normSoundtrack(chosen), ok = (v) => options.some((o) => o.val === v && o.enabled);
    const value = ok(c) ? c : (hasV && vOk ? "video" : "generate");
    return { options, value };
  }

  /** The reply of `OUT read` (JSON text, maybe in a code fence, maybe an object) -> a READ JSON v1 with every field present. Throws on garbage. */
  function normalizeRead(raw) {
    let o = raw;
    if (typeof raw === "string") {
      let t = raw.trim();
      const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
      if (fence) t = fence[1].trim();
      if (t[0] !== "{") { const a = t.indexOf("{"), b = t.lastIndexOf("}"); if (a >= 0 && b > a) t = t.slice(a, b + 1); }
      o = JSON.parse(t);
    }
    if (!o || typeof o !== "object" || Array.isArray(o)) throw new Error("the read reply is not a JSON object");
    const str = (v) => (v === undefined || v === null ? "" : String(v));
    const lines = Array.isArray(o.on_screen) ? o.on_screen : (o.on_screen ? [o.on_screen] : []);
    return {
      v: 1,
      on_screen: lines.map(str).map((x) => x.trim()).filter(Boolean),
      concept: str(o.concept), look: str(o.look),
      media: (Array.isArray(o.media) ? o.media : []).filter((m) => m && typeof m === "object").map((m) => Object.assign({}, m, {
        id: str(m.id), kind: str(m.kind), file: str(m.file), shows: str(m.shows),
        window: Array.isArray(m.window) && m.window.length === 2 ? m.window.map(Number) : null,
        mood: str(m.mood), palette: Array.isArray(m.palette) ? m.palette.map(str) : [] })),
      music: o.music && typeof o.music === "object" ? o.music : null,
      notes: str(o.notes),
      ...(normSoundtrack(o.soundtrack) ? { soundtrack: normSoundtrack(o.soundtrack) } : {}),     // v19: which audio the read analysed (video | track | generate)
      // VIDEO MODE (MEDIA_BOX.md): the video's length, its real cuts, the sung words; carried only when the read has them
      ...(o.beat_grid && typeof o.beat_grid === "object" && !Array.isArray(o.beat_grid) && Array.isArray(o.beat_grid.beats) ? { beat_grid: o.beat_grid } : {}),     // v19: the DSP grid of the AUDIO THE SOUNDTRACK PICKED (it rode along nowhere before: the story fell back to a synthetic grid)
      ...(Number(o.video_duration) > 0 ? { video_duration: Number(o.video_duration) } : {}),
      ...(Array.isArray(o.video_cuts) ? { video_cuts: o.video_cuts.map((c) => ({ start: Number(c && c.start), end: Number(c && c.end) })).filter((c) => Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start) } : {}),
      ...(Array.isArray(o.lyrics) ? { lyrics: o.lyrics.filter((l) => l && typeof l === "object" && str(l.text).trim() && Number.isFinite(Number(l.t)))
        .map((l) => Object.assign({}, l, { t: Number(l.t), end: Number.isFinite(Number(l.end)) ? Number(l.end) : Number(l.t), text: str(l.text).trim() })) } : {}),
      ...(o.story && typeof o.story === "object" && !Array.isArray(o.story) ? { story: o.story } : {}),   // the edited STORY JSON rides along in the brief (briefFromStory)
    };
  }

  /** The edited READ JSON that goes into `IN brief`: blank lines dropped, text trimmed. "" when there is nothing to say. */
  function briefString(read) {
    if (!read) return "";
    const r = normalizeRead(read);
    r.on_screen = r.on_screen.map((x) => x.trim()).filter(Boolean);
    r.concept = r.concept.trim(); r.look = r.look.trim();
    return JSON.stringify(r);
  }

  /** the brief JSON with the text the USER typed in "Words or direction" riding along as `direction` (it is NOT part of the READ the model writes: normalizeRead drops it, so a model reply can never carry one).
   *  Nothing typed = the string unchanged. The server turns it into piece.user_style (tools/fusion/user_style.py): palette / legibility / plain wishes override the automated defaults. */
  function withDirection(briefJson, text) {
    const t = typeof text === "string" ? text.replace(/\r\n?/g, "\n").trim() : "";
    if (!briefJson || !t) return briefJson;
    let o;
    try { o = JSON.parse(briefJson); } catch (_) { return briefJson; }
    if (!o || typeof o !== "object" || Array.isArray(o)) return briefJson;
    o.direction = t.slice(0, 2000);
    return JSON.stringify(o);
  }

  /** splitTyped(text) -> { words, direction }: the page's port of tools/fusion/user_style.py split_typed (the Python is the source of truth; tests/test_split_typed.py runs both on the same cases).
   *  Directive clauses / lines (palette: / colours: lines, "palette of ...", "in red and gold", legibility asks, plain / simple asks, "make the text ...", "use ...") are the DIRECTION and never
   *  on-screen words; everything else is the words. Pure direction -> words "". The "Your text" list tokenises `words` so its token numbers are the server's. */
  const TYPED_NAMES = ["turquoise", "lavender", "crimson", "scarlet", "emerald", "magenta", "orange", "golden", "yellow", "cobalt", "indigo", "purple", "violet", "silver", "amber", "green", "coral", "peach", "cream", "ivory", "white", "black", "brown", "olive", "gold", "lime", "teal", "cyan", "aqua", "blue", "navy", "pink", "rose", "grey", "gray", "rust", "mint", "neon", "red"];
  const TX = (() => {
    const NAME = "(?:" + TYPED_NAMES.join("|") + ")";
    const SEP = "(?:\\s*,\\s*(?:and\\s+)?|\\s*/\\s*|\\s*&\\s*|\\s*\\+\\s*|\\s+and\\s+|\\s+with\\s+)";
    const LIST = `\\b${NAME}\\b(?:${SEP}\\b${NAME}\\b)*`;
    const LEAD = "(?:(?:use|using|with|in|a|an|the|set|keep|go|choose|pick)\\s+)*";
    const VERB = "(?:make|use|using|keep|set|render|show|give|put|add|try|have|write|go|style|paint|colou?r|apply|choose|pick)";
    const PRE = "(?:(?:please|also|and|then|just|now|maybe|can\\s+you|could\\s+you|let'?s|we\\s+want|it\\s+should|should|must|i\\s+want(?:\\s+to)?|i\\s+need|i'?d\\s+like(?:\\s+to)?|i\\s+would\\s+like(?:\\s+to)?)\\s+)*";
    const OBJ = "(?:text|type|typeface|fonts?|letters?|lettering|colou?rs?|palette|background|effects?|echo(?:es)?|trails?|shimmer|glow|outline|shadow|contrast|animation|captions?|titles?|serif|sans|bold|italic|hues?|tones?)";
    return {
      HEX6: /#[0-9a-fA-F]{6}\b/,
      P_LINE: /^[ \t]*(?:palette|colou?rs?)[ \t]*:[ \t]*(.+)$/im,
      P_CUE: new RegExp(`\\b(?:palette|colou?rs?|tones?|hues?)\\b\\s*(?:of|is|are|being|=|:|-)?\\s*(?:a\\s+|the\\s+|only\\s+)?(${LIST})`, "i"),
      P_TAIL: new RegExp(`(${LIST})\\s+(?:colou?r\\s+)?(?:palette|colou?rs?|scheme|tones?|theme)\\b`, "ig"),
      P_IN: new RegExp(`\\b(?:make|do|render|set|keep|show|write|type|use|using|go|style|styled|colou?red|paint|painted|text|words?|letters)\\b[^.\\n]{0,24}?\\bin\\s+(?:an?\\s+)?(${LIST})`, "ig"),
      P_BARE: new RegExp(`(?:^|[,;.]\\s*)(?:in|using|with)\\s+(${NAME}(?:${SEP}${NAME})+)[ \\t]*[.!]?[ \\t]*$`, "img"),
      DIRECTIVE: /^\s*(palette|colou?rs?|style|look|text|type|font|direction)\s*:/i,
      LEGIBLE: /\b(more\s+)?(legib\w*|readab\w*|clear(er)?\s+(text|type|words|letters)|(bigger|larger)\s+(text|type|words|letters|font)|easier\s+to\s+read)\b/i,
      PLAIN: /\b((keep|make|go|stay|want|prefer|have)\s+(it\s+|this\s+|the\s+\w+\s+|everything\s+)?(very\s+|really\s+|more\s+)?(plain|simple|simpler|minimal(ist)?|clean)\b|(plain|simple|minimal(ist)?)\s+(text|type|look|style|design|version)\b|no\s+(effects?|echo(es)?|trails?|shimmer)\b|clean\s+(text|type)\b)/i,
      QUOTED: /"[^"]*"|“[^”]*”/g,
      CUT: new RegExp(`${LEAD}\\b(?:palette\\s*(?::|=|of\\b|is\\b|are\\b)|colou?rs?\\s*[:=])[^\\n]*`, "ig"),
      TAIL_LEAD: new RegExp(`${LEAD}$`, "i"),
      INSTR: new RegExp(`^\\s*${PRE}${VERB}\\b(?:\\s+\\S+){0,5}?\\s+(?:the\\s+)?(?:${OBJ}|words?)\\b|^\\s*${PRE}(?:make|use|using|set|paint|apply)\\s+(?:\\S+\\s+){0,3}?${NAME}\\b`, "i"),
      NEG: new RegExp(`^\\s*${PRE}(?:don'?t|do\\s+not|never|avoid|without)\\b(?:\\s+\\S+){0,4}?\\s+${OBJ}\\b`, "i"),
      DELIM: /([,;.!?]+\s*|\n)/,
      EDGE: /^[\s,;.:\-–—]+|(?:[\s,;.:\-–—]|\b(?:and|with|plus|in)\b|&)+$/ig,
      NAME_RE: new RegExp(`\\b(${NAME})\\b`, "i"),
    };
  })();
  const txAll = (rx, s) => [...s.matchAll(new RegExp(rx.source, rx.flags.includes("g") ? rx.flags : rx.flags + "g"))];
  function txColours(c) {                         // colours_in(c) non-empty
    if (TX.HEX6.test(c)) return true;
    for (const m of txAll(TX.P_LINE, c)) if (/#[0-9a-fA-F]{3}\b/.test(m[1]) || TX.NAME_RE.test(m[1])) return true;
    return [TX.P_CUE, TX.P_TAIL, TX.P_IN, TX.P_BARE].some((rx) => txAll(rx, c).some((m) => TX.NAME_RE.test(m[1])));
  }
  const txDirective = (c) => c.trim() !== "" && (txColours(c) || TX.LEGIBLE.test(c) || TX.PLAIN.test(c) || TX.INSTR.test(c) || TX.NEG.test(c));
  const txStrip = (s, chars) => { let a = 0, b = s.length; while (a < b && chars.includes(s[a])) a++; while (b > a && chars.includes(s[b - 1])) b--; return s.slice(a, b); };
  function txChunk(chunk) {                       // an unquoted run inside one line -> [words, [direction pieces]]
    const words = [], dirs = [], pieces = [];
    let pos = 0;
    for (const m of txAll(TX.CUT, chunk)) { pieces.push([chunk.slice(pos, m.index), false]); pieces.push([m[0], true]); pos = m.index + m[0].length; }
    pieces.push([chunk.slice(pos), false]);
    for (const [txt, d] of pieces) {
      if (d) { dirs.push(txt.trim()); continue; }
      const spans = [];
      for (const rx of [TX.P_TAIL, TX.P_IN, TX.P_BARE]) for (const m of txAll(rx, txt)) {
        let s = m.index;
        if (rx === TX.P_TAIL) { const lead = TX.TAIL_LEAD.exec(txt.slice(0, s)); if (lead) s = s - lead[0].length; }
        spans.push([s, m.index + m[0].length]);
      }
      spans.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      const merged = [];
      for (const [a, b] of spans) { if (merged.length && a <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(b, merged[merged.length - 1][1]); else merged.push([a, b]); }
      let p = 0; const segs = [];
      for (const [a, b] of merged) { segs.push([txt.slice(p, a), false]); segs.push([txt.slice(a, b), true]); p = b; }
      segs.push([txt.slice(p), false]);
      for (const [seg, sd] of segs) {
        if (sd) { dirs.push(txStrip(seg, " ,;.\t")); continue; }
        const parts = seg.split(TX.DELIM);
        for (let i = 0; i < parts.length; i += 2) {
          const cl = parts[i], dl = i + 1 < parts.length ? parts[i + 1] : "";
          if (txDirective(cl)) dirs.push(cl.trim()); else words.push(cl + dl);
        }
      }
    }
    return [words.join(""), dirs.filter(Boolean)];
  }
  function splitTyped(text) {
    const t = String(text == null ? "" : text).replace(/\r\n?/g, "\n").trim();
    const wl = [], dl = [];
    for (const line of t.split("\n")) {
      if (!line.trim()) { wl.push(""); continue; }
      if (TX.DIRECTIVE.test(line)) { dl.push(line.trim()); continue; }
      const q = txAll(TX.QUOTED, line);
      let w, d;
      if (!q.length) [w, d] = txChunk(line);
      else {
        w = ""; d = []; let pos = 0;
        for (const m of q) { const [cw, cd] = txChunk(line.slice(pos, m.index)); w += cw + m[0]; d = d.concat(cd); pos = m.index + m[0].length; }
        const [cw, cd] = txChunk(line.slice(pos)); w += cw; d = d.concat(cd);
      }
      if (!d.length) { wl.push(line); continue; }
      w = q.length ? w.trim() : w.trim().replace(TX.EDGE, "").trim();
      if (!/[\p{L}\p{N}_]/u.test(w)) w = "";
      dl.push(w ? d.join(" ") : line.trim());
      if (w) wl.push(w);
    }
    if (!dl.length) return { words: t, direction: "" };
    return { words: wl.join("\n").replace(/\n{3,}/g, "\n\n").trim(), direction: dl.join("\n") };
  }

  // ------------------------------------------------------------------ STORY step (16_app/MEDIA_BOX.md "STORY step")
  const num = (v, d) => { const n = Number(v); return Number.isFinite(n) ? n : d; };

  /** What `IN story` gets (as an object, JSON-stringify it): the (edited) READ, the length, the seed, the format. */
  function buildStoryRequest(read, length, seed, format, extra) {
    const r = typeof read === "string" ? (read.trim() ? JSON.parse(read) : null) : read;
    const n = Number(length), s = Math.floor(Number(seed));
    const req = { read: r ? normalizeRead(r) : null, length: Number.isFinite(n) && n > 0 ? n : 12, seed: Number.isFinite(s) && s >= 0 ? s : 0, format: format || "9:16" };
    // STORYBOARD v2 (the backend honours these verbatim): use = image ids in shot order, select = [{i, phrase, shot}] over the whitespace tokens of `text`, text = the pasted text, fonts = liked font file per shot ("" = free)
    const x = extra || {};
    if (Array.isArray(x.use) && x.use.length) req.use = x.use.map(String);
    if (Array.isArray(x.select)) req.select = x.select.filter((t) => Number(t.shot) >= 1).map((t) => ({ i: t.i, phrase: t.shot, shot: t.shot }));      // v18: each token names its shot (the backend puts the phrase on THAT shot, no time-based placement)
    if (typeof x.text === "string" && x.text.trim()) req.text = x.text;
    if (x.fonts && typeof x.fonts === "object") {                       // {"1": file, ...} for the shots the user picked a font for (liked files only)
      const f = {}; for (const [k, v] of Object.entries(x.fonts)) if (v) f[String(k)] = String(v);
      if (Object.keys(f).length) req.fonts = f;
    }
    return req;
  }

  /** What a story depends on: when this changes the STORY JSON is stale. `readJson` = the edited READ as text. */
  function storySignature(readJson, length, seed, format) {
    return JSON.stringify([String(readJson || ""), String(length || ""), String(seed === undefined || seed === null ? "" : seed), String(format || "")]);
  }

  /** `OUT story` reply (JSON text, maybe fenced, maybe an object) -> STORY JSON v1 with every field present. Throws on garbage / no shots. */
  function normalizeStory(raw) {
    let o = raw;
    if (typeof raw === "string") {
      let t = raw.trim();
      const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
      if (fence) t = fence[1].trim();
      if (t[0] !== "{") { const a = t.indexOf("{"), b = t.lastIndexOf("}"); if (a >= 0 && b > a) t = t.slice(a, b + 1); }
      o = JSON.parse(t);
    }
    if (!o || typeof o !== "object" || Array.isArray(o)) throw new Error("the story reply is not a JSON object");
    const str = (v) => (v === undefined || v === null ? "" : String(v));
    const shots = (Array.isArray(o.shots) ? o.shots : []).filter((x) => x && typeof x === "object");
    if (!shots.length) throw new Error("the story has no shots");
    const cap = o.capacity && typeof o.capacity === "object"
      ? Object.assign({}, o.capacity, { shots: Math.max(1, Math.floor(num(o.capacity.shots, shots.length))), why: str(o.capacity.why).trim() }) : null;   // + images, min_length, pick_length, per_shot_s as sent
    return {
      v: num(o.v, 1) >= 2 ? 2 : 1, title: str(o.title).trim(), logline: str(o.logline).trim(), look: str(o.look).trim(), music: str(o.music).trim(),
      shots: shots.map((x, i) => ({ n: Math.max(1, Math.floor(num(x.n, i + 1))), image: x.image ? str(x.image) : null,
                                     start: num(x.start, 0), end: num(x.end, num(x.start, 0)), beat: str(x.beat).trim(), line: str(x.line).trim(),
                                     camera: str(x.camera).trim(), font: str(x.font).trim(), ...(x.tl && typeof x.tl === "object" && !Array.isArray(x.tl) ? { tl: x.tl } : {}),   /* 2026-11-03: the frame's layout and look ride to the render */
                                     ...(x.src_start !== undefined && x.src_start !== null && x.src_end !== undefined && x.src_end !== null && Number.isFinite(Number(x.src_start)) && Number.isFinite(Number(x.src_end))
                                         ? { src_start: Number(x.src_start), src_end: Number(x.src_end) }     // video shot: a real cut of the clip (v18: no transition field, the design step picks the cuts)
                                         : {}) })),
      candidates: (Array.isArray(o.candidates) ? o.candidates : []).filter((c) => c && typeof c === "object" && c.image)
        .map((c) => ({ image: str(c.image), use: !!c.use, reason: str(c.reason).trim() })),
      text_select: (Array.isArray(o.text_select) ? o.text_select : []).filter((t) => t && Number.isFinite(Number(t.i)) && Number(t.i) >= 0)
        .map((t) => ({ i: Math.floor(Number(t.i)), phrase: Math.max(0, Math.floor(num(t.phrase, 0))), shot: Math.max(0, Math.floor(num(t.shot, 0))) })),      // v18: shot = the shot (n) that carries the token, 0 = unknown/none
      capacity: cap,
      notes: str(o.notes).trim(),
      // WORD PICK: per shot (key = n) the ranked other word choices [{t: [token index, ...], s: score}] of the lyrics sung inside the shot (the page's cycle button steps through them)
      ...(o.word_alts && typeof o.word_alts === "object" && !Array.isArray(o.word_alts) ? { word_alts: Object.fromEntries(Object.entries(o.word_alts)
        .map(([k, v]) => [String(k), (Array.isArray(v) ? v : []).filter((a) => a && Array.isArray(a.t) && a.t.length && a.t.every((i) => Number.isFinite(Number(i)) && Number(i) >= 0))
          .map((a) => ({ t: a.t.map((i) => Math.floor(Number(i))), s: Number.isFinite(Number(a.s)) ? Number(a.s) : 0 }))]).filter(([, v]) => v.length)) } : {}),
      // VIDEO MODE: the text hits {text, land, hold_until} in window seconds (decided on beat-snapped slots, independent of the cuts); only when the story has them
      ...(Array.isArray(o.hits) ? { hits: o.hits.filter((h) => h && typeof h === "object" && str(h.text).trim() && Number.isFinite(Number(h.land)))
        .map((h) => ({ text: str(h.text).trim(), land: Number(h.land), hold_until: Number.isFinite(Number(h.hold_until)) ? Number(h.hold_until) : Number(h.land), ...(Number.isFinite(Number(h.sung)) ? { sung: Number(h.sung) } : {}) })).sort((a, b) => a.land - b.land) } : {}),
    };
  }

  /** "0.0-4.0 s" */
  function fmtSpan(a, b) { return `${num(a, 0).toFixed(1)}-${num(b, 0).toFixed(1)} s`; }

  /**
   * The brief Make it sends when a story exists: the edited READ with on_screen = shots[].line (non-empty, in order),
   * concept = title + ". " + logline + " " + the beats, look = story.look (unless the user edited the READ's Look: opts.lookEdited),
   * plus `story` = the edited STORY JSON. Returns the JSON text (like briefString).
   */
  function briefFromStory(read, story, opts) {
    const r = normalizeRead(read), st = normalizeStory(story);
    r.on_screen = st.shots.map((x) => x.line.trim()).filter(Boolean);
    const head = st.title && st.logline ? `${st.title}. ${st.logline}` : (st.title || st.logline);
    r.concept = [head, ...st.shots.map((x) => x.beat.trim()).filter(Boolean)].filter(Boolean).join(" ").trim();
    if (!(opts && opts.lookEdited) && st.look) r.look = st.look;
    r.look = r.look.trim();
    // R17b: a shot's `font` is the page's random DEFAULT (fillFonts) unless the user changed it; only the user's own picks are marked `font_user` (opts.userFonts = their shot numbers), and only
    // those become font pins on the server (concept.story_fonts). A default font rides along for display but is never a pin.
    const mine = new Set(((opts && opts.userFonts) || []).map(Number));
    st.shots.forEach((x) => { if (x.font && mine.has(x.n)) x.font_user = true; else delete x.font_user; });
    r.story = st;
    return JSON.stringify(r);
  }

  // ------------------------------------------------------------------ STORYBOARD v2 (16_app/MEDIA_BOX.md "STORYBOARD v2"): words, frames, fonts, legible ink, revert-to-Gemma
  /** shots[].camera = the cut that carries shot k into shot k+1 (the last shot is `hold`): the backend's list (MEDIA_BOX.md "STORYBOARD v2 backend"); the select also offers whatever the story returns. */
  const CAMERAS = ["hold", "depth_pivot", "orbit:push", "orbit:pullback", "orbit:fly", "pan:down", "pan:up", "pan:left", "pan:right"];
  const cameraLabel = (c) => String(c || "").replace(/[_:]/g, " ");
  // v18: video mode has no transition / camera per shot (user 2026-10-31: the transitions are the design step's choice, not the storyboard's)
  /** the planner's shot count for a Length: 6 -> 2, 8 -> 2, 12 -> 3, 16 -> 4, 24 -> 6 (max 6). */
  const shotCountFor = (len) => Math.max(1, Math.min(6, Math.ceil((Number(len) * 24 - 12) / 109 - 1e-9)));
  const TEXT_CAP = 400;

  // ---- STORYBOARD v12 (16_app/MEDIA_BOX.md): shots never exceed images. Mirrors tools/fusion/story.py shots_for / feasible_lengths.
  const CLIP_MAX_S = 5.0;                                   // one LTX clip per shot
  const LENGTHS = [6, 8, 12, 16, 24];
  const lockupCount = shotCountFor;                         // 8 -> 2, 12 -> 3, 16 -> 4, 24 -> 6
  /** the shots a Length gets with n images: n >= 2 -> min(n, lockups); n = 0 (text only) and n = 1 -> the lockup count as before. */
  const shotsFor = (length, n) => (Number(n) >= 2 ? Math.min(Math.floor(Number(n)), lockupCount(length)) : lockupCount(length));
  /** n < 2: always feasible; else the clip each shot needs (length / shots) must fit CLIP_MAX_S. */
  const lengthFeasible = (length, n) => Number(n) < 2 || Number(length) / shotsFor(length, n) <= CLIP_MAX_S + 1e-9;
  const feasibleLengths = (n, options) => (options || LENGTHS).filter((L) => lengthFeasible(L, n));
  /** how many more images a Length needs (0 = feasible). */
  const imagesNeeded = (length, n) => (lengthFeasible(length, n) ? 0 : Math.max(0, Math.ceil(Number(length) / CLIP_MAX_S - 1e-9) - Math.floor(Number(n))));
  /** the Length to use when `length` is not feasible: the longest feasible one (the shortest option if none is). */
  function bestFeasibleLength(length, n, options) {
    const opts = (options && options.length ? options : LENGTHS).slice().sort((a, b) => a - b), ok = opts.filter((L) => lengthFeasible(L, n));
    if (lengthFeasible(length, n)) return Number(length);
    return ok.length ? ok[ok.length - 1] : opts[0];
  }
  /** cut a story down to `cap` shots (cap < 1 = no cap): the later shots go, the kept spans stretch to cover the same total, the last cut is `hold`. */
  function capShots(story, cap) {
    if (!(cap >= 1) || story.shots.length <= cap) return story;
    const s = clone(story), end = Math.max(...s.shots.map((x) => num(x.end, 0)));
    s.shots = s.shots.slice(0, cap);
    const kept = Math.max(...s.shots.map((x) => num(x.end, 0))), f = kept > 0 && end > kept ? end / kept : 1;
    const r1 = (v) => Math.round(v * 10) / 10;
    for (const x of s.shots) { x.start = r1(num(x.start, 0) * f); x.end = r1(num(x.end, 0) * f); }
    s.shots[s.shots.length - 1].camera = "hold";
    s.shots[s.shots.length - 1].end = r1(end);
    s.capacity = Object.assign({}, s.capacity || {}, { shots: cap, why: "" });
    s.candidates = (s.candidates || []).map((c) => (s.shots.some((x) => x.image === c.image) ? c : Object.assign({}, c, { use: false })));
    return s;
  }

  // ---- VIDEO MODE (16_app/MEDIA_BOX.md "VIDEO MODE STORYBOARD"): the storyboard is built from the video's real cuts, never invented
  const VIDEO_CLIP_MAX_S = 5.04, VIDEO_MERGE_S = 0.2, VIDEO_FRAME_OFF_S = 0.04, VIDEO_LEN_SLACK_S = 0.1;       // a cut's picture = the video's frame at src_start + 0.04 s (its first frame)
  const r2 = (v) => Math.round(v * 100) / 100;
  /** "12" / "11.9": a duration for a label. */
  const fmtDur = (d) => { const n = Number(d), r = Math.round(n); return Math.abs(n - r) <= 0.15 ? String(r) : n.toFixed(1); };       // 12.07 -> "12"
  /** Length select with a video of `duration` s: which options are too long (disabled), the Length to use, and the extra option when the video is shorter than the smallest one
   *  (its duration rounded down to 0.5 s). `current` fits -> kept, else the longest option that fits. Unknown duration -> no limit. */
  function videoLengthPlan(options, duration, current) {
    const opts = (options && options.length ? options : LENGTHS).map(Number).filter(Number.isFinite).sort((a, b) => a - b), cur = Number(current), dur = Number(duration);
    if (!(dur > 0)) return { disabled: {}, extra: null, length: cur, changed: false, duration: NaN };
    const disabled = {}; for (const L of opts) if (L > dur + VIDEO_LEN_SLACK_S) disabled[L] = true;      // a "12 s" clip that measures 11.98 s still offers 12
    // v16: the clip's own duration (rounded down to 0.5 s) is an extra option when it is not one of the standard ones (a 10 s clip: 8, 10, then 12 / 16 / 24 greyed) and below the longest option
    const half = Math.max(0.5, Math.floor(dur * 2 + 1e-9) / 2), near = opts.some((L) => Math.abs(L - dur) <= VIDEO_LEN_SLACK_S || L === half);
    const extra = !near && dur < opts[opts.length - 1] ? half : null;
    const fit = opts.filter((L) => !disabled[L]);
    // an extra that is the only option (clip shorter than the smallest Length) is the Length; otherwise `current` is kept when it fits, else the longest STANDARD option that fits
    const length = !fit.length ? extra : (opts.includes(cur) && !disabled[cur]) || (extra !== null && cur === extra) ? cur : fit[fit.length - 1];
    return { disabled, extra, length, changed: length !== cur, duration: dur };
  }
  /** the shot spans (source seconds) of a video inside the Length window: the READ's cuts (v18: EVERY real cut stays; only a piece under 6 frames = 0.2 s at 30 fps merges into the shorter neighbour), pieces > 5.04 s split evenly;
   *  no cuts -> an even split by the planner's shot count (never a piece > 5.04 s). -> [{start, end}] covering [0, window]. */
  function deriveVideoSpans(cuts, duration, length) {
    const L = Number(length), D = Number(duration), W = r2(D > 0 && L > 0 ? Math.min(D, L) : (D > 0 ? D : L));
    if (!(W > 0)) return [];
    const marks = [];
    for (const c of cuts || []) for (const v of [Number(c && c.start), Number(c && c.end)]) if (Number.isFinite(v) && v > 0.05 && v < W - 0.05 && !marks.some((m) => Math.abs(m - v) < 0.05)) marks.push(v);
    marks.sort((a, b) => a - b);
    let sp;
    if (marks.length) {
      const edges = [0, ...marks, W];
      sp = edges.slice(0, -1).map((a, i) => ({ start: a, end: edges[i + 1] }));
      for (;;) {                                                      // merge the shortest piece under 6 frames (0.2 s) into its shorter neighbour
        let k = -1, len = VIDEO_MERGE_S;
        sp.forEach((x, i) => { if (x.end - x.start < len - 1e-9) { len = x.end - x.start; k = i; } });
        if (k < 0 || sp.length < 2) break;
        const a = sp[k - 1], b = sp[k + 1], j = !a ? k + 1 : !b ? k - 1 : (a.end - a.start <= b.end - b.start ? k - 1 : k + 1);
        const lo = Math.min(j, k), hi = Math.max(j, k);
        sp.splice(lo, 2, { start: sp[lo].start, end: sp[hi].end });
      }
    } else {
      const n = Math.max(lockupCount(W), Math.ceil(W / VIDEO_CLIP_MAX_S - 1e-9), 1);
      sp = Array.from({ length: n }, (_, i) => ({ start: W * i / n, end: W * (i + 1) / n }));
    }
    const out = [];
    for (const x of sp) {
      const n = Math.max(1, Math.ceil((x.end - x.start) / VIDEO_CLIP_MAX_S - 1e-9));
      for (let i = 0; i < n; i++) out.push({ start: r2(x.start + (x.end - x.start) * i / n), end: r2(x.start + (x.end - x.start) * (i + 1) / n) });
    }
    out[out.length - 1].end = W;
    return out;
  }
  const isVideoShot = (x) => !!x && x.src_start !== undefined && x.src_start !== null && x.src_end !== undefined && x.src_end !== null && Number.isFinite(Number(x.src_start)) && Number.isFinite(Number(x.src_end));
  /** the text hits that land inside a shot's window span [start, end) (the last shot also takes a hit at its end). */
  function hitsInShot(story, k) {
    const sh = story.shots[k], last = k === story.shots.length - 1;
    return (story.hits || []).filter((h) => h.land >= sh.start - 1e-6 && (h.land < sh.end - 1e-6 || (last && h.land <= sh.end + 1e-6)));
  }
  /** video mode: a shot with no words gets the hit(s) landing inside its span (the backend's `hits` are the truth for what lands where). */
  function fillLinesFromHits(story) {
    if (!story.hits || !story.hits.length || !story.shots.every(isVideoShot)) return story;
    const s = clone(story);
    s.shots.forEach((sh, k) => { if (!sh.line) sh.line = hitsInShot(s, k).map((h) => h.text).join(" "); });
    return s;
  }
  /** hits for a set of per-shot lines (the words were chosen by hand): one hit per non-empty line, landing at the shot's start, held to its end. */
  const hitsFromLines = (shots, lines) => shots.map((sh, k) => (lines[k] ? { text: lines[k], land: sh.start, hold_until: sh.end } : null)).filter(Boolean);
  /** a story whose shots all carry a source span is kept; an old story (shots without spans) gets its shots rebuilt on the READ's real cuts
   *  (shot k keeps the old shot k's words / beat / font while there is one). v = {cuts, duration}; id = the video's id ("V1"). Without a duration nothing is invented: returned as is. */
  function videoStoryFix(story, v, length, id) {
    let s = story;
    if (!story.shots.every(isVideoShot)) {
      const spans = deriveVideoSpans((v && v.cuts) || [], v && v.duration, length);
      if (!spans.length) return story;
      const t0 = spans[0].start;
      s = clone(story);
      s.shots = spans.map((sp, k) => Object.assign({ line: "", beat: "", font: "" }, story.shots[k] || {}, { n: k + 1, image: id || "V1", start: r2(sp.start - t0), end: r2(sp.end - t0), src_start: sp.start, src_end: sp.end, camera: "as filmed" }));
      s.capacity = Object.assign({}, s.capacity || {}, { shots: spans.length });
    }
    return fillLinesFromHits(s);
  }
  /** the shot (index) whose source span holds time t (the last span includes its end); -1 = none. */
  function shotAtTime(shots, t) {
    const T = Number(t);
    if (!Number.isFinite(T)) return -1;
    return shots.findIndex((x, i) => isVideoShot(x) && T >= x.src_start - 1e-6 && (T < x.src_end - 1e-6 || (i === shots.length - 1 && T <= x.src_end + 1e-6)));
  }
  /** READ lyrics [{t, end, text, ts?, words?}] -> the token list (whitespace words of each line's text, in sung order, the same tokens the backend counts) and each token's sung time.
   *  WORD PICK: the real Whisper word times are used whenever the line has them (`words` [{text, t, end, p}], else `ts` [start per word]); only a line without them spreads its words
   *  over t..end (or over ~0.45 s a word when it has no end). Also returns `ends` (sung end of each token) and `probs` (Whisper's word probability, null when unknown). */
  function lyricTokens(lyrics) {
    const tokens = [], times = [], ends = [], probs = [];
    const lines = [...(lyrics || [])].sort((a, b) => num(a.t, 0) - num(b.t, 0));
    lines.forEach((l, li) => {
      const t0 = num(l.t, 0), t1 = Math.max(t0, num(l.end, t0)), nextT = li + 1 < lines.length ? num(lines[li + 1].t, null) : null;
      const W = Array.isArray(l.words) ? l.words.filter((w) => w && typeof w === "object") : [];
      const tt = textTokens(l.text), n = tt.length;
      if (!n) {                                                     // an older shape: a line with words[] only
        for (const w of W) for (const piece of String(w.text || w.word || "").trim().split(/\s+/).filter(Boolean)) { tokens.push(piece); times.push(r2(num(w.t, t0))); ends.push(r2(num(w.end, num(w.t, t0)))); probs.push(null); }
        return;
      }
      const fin = (x) => x !== null && x !== undefined && x !== "" && Number.isFinite(Number(x));
      let st, en = null, pr = null;
      if (W.length === n && W.every((w) => fin(w.t))) {
        st = W.map((w) => Number(w.t)); en = W.map((w) => (fin(w.end) ? Number(w.end) : null)); pr = W.map((w) => (fin(w.p) ? Number(w.p) : null));
      } else if (Array.isArray(l.ts) && l.ts.length === n && l.ts.every(fin)) {
        st = l.ts.map(Number);
      } else {
        const span = t1 > t0 ? t1 - t0 : Math.min(0.45 * n, nextT !== null && nextT > t0 ? nextT - t0 : 0.45 * n);
        st = tt.map((_, i) => t0 + span * i / Math.max(1, n));
      }
      tt.forEach((w, i) => {
        tokens.push(w); times.push(r2(st[i]));
        ends.push(r2(en && en[i] !== null ? en[i] : (i + 1 < n ? st[i + 1] : (t1 > st[i] ? t1 : st[i] + 0.5))));
        probs.push(pr ? pr[i] : null);
      });
    });
    return { tokens, times, ends, probs };
  }
  /** video mode: the chosen phrases -> the shot each lands on (the one whose span holds the phrase's first word's sung time); a second phrase in the same shot, or one outside the window, lands nowhere.
   *  -> {lines: [text per shot], placed: Map(phrase value -> shot index | -1)} */
  function phrasePlacement(shots, tokens, sel, times, hits) {   // hits (STORY video mode): a phrase the backend placed lands at its hit's `land` (beat slot), a phrase the user added at its sung time
    const by = new Map(), hitAt = new Map((hits || []).map((h) => [String(h.text || "").toUpperCase(), Number(h.land)]));
    for (const t of sel || []) { if (!by.has(t.phrase)) by.set(t.phrase, []); by.get(t.phrase).push(t.i); }
    const lines = shots.map(() => ""), placed = new Map();
    for (const k of [...by.keys()].sort((a, b) => a - b)) {
      const idx = by.get(k).filter((i) => i < tokens.length).sort((a, b) => a - b);
      if (!idx.length) continue;
      const text = stripEdge(idx.map((i) => tokens[i]).join(" ")), hl = hitAt.get(text.toUpperCase()), at = shotAtTime(shots, Number.isFinite(hl) ? hl : (times || [])[idx[0]]);
      if (text && at >= 0 && !lines[at]) { lines[at] = text; placed.set(k, at); } else placed.set(k, -1);
    }
    return { lines, placed };
  }

  const textTokens = (text) => { const t = String(text || "").trim(); return t ? t.split(/\s+/) : []; };
  const stripEdge = (t) => String(t).replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");

  /** selection [{i, phrase, shot}] -> sorted by token, de-duplicated (the last entry of a token wins). v18: `shot` = the shot (n) that carries the token (0 = not known / none), `phrase` is kept as sent
   *  (the page itself writes phrase = shot). Adjacency is never inferred: two neighbouring tokens on different shots stay on their own shots. */
  function normSelect(sel) {
    const m = new Map();
    for (const t of sel || []) if (t && Number.isFinite(Number(t.i)) && Number(t.i) >= 0) m.set(Math.floor(Number(t.i)), { phrase: Number.isFinite(Number(t.phrase)) ? Math.floor(Number(t.phrase)) : 0, shot: Number.isFinite(Number(t.shot)) ? Math.max(0, Math.floor(Number(t.shot))) : 0 });
    return [...m.keys()].sort((a, b) => a - b).map((i) => ({ i, phrase: m.get(i).phrase, shot: m.get(i).shot }));
  }
  /** phrase value -> its rank 1.. in screen order (a backend that numbers from 0 or skips numbers still reads 1, 2, 3 on the page). */
  function phraseRanks(sel) {
    const vals = [...new Set((sel || []).map((t) => t.phrase))].sort((a, b) => a - b);
    return new Map(vals.map((v, k) => [v, k + 1]));
  }
  /** v18 SHOT-FOCUSED WORDS: click on word i for shot `shot`: it is on that shot -> off; on another shot (or on none) -> on THIS shot. Only the clicked token changes. */
  function toggleWordShot(sel, i, shot) {
    const cur = (sel || []).find((t) => t.i === i), rest = (sel || []).filter((t) => t.i !== i);
    if (!(cur && cur.shot === shot)) rest.push({ i, phrase: shot, shot });
    return normSelect(rest);
  }
  /** the tokens (indices, ascending) a shot carries. */
  const shotTokens = (sel, shot) => (sel || []).filter((t) => t.shot === shot).map((t) => t.i).sort((a, b) => a - b);
  /** the clock a shot's words are sung on: a video cut = the clip's source seconds, a picture shot = the piece (the song plays from its start). -> [from, to] */
  const shotSpan = (sh) => (isVideoShot(sh) ? [Number(sh.src_start), Number(sh.src_end)] : [num(sh.start, 0), num(sh.end, 0)]);
  /** WORD PICK editor: {by: Map(shot n -> [token index sung inside the shot's span, ascending]), outside: [token index in no shot's span]}. The last shot also owns a word sung at its very end. */
  function sungByShot(shots, times) {
    const by = new Map((shots || []).map((s) => [s.n, []])), outside = [];
    (times || []).forEach((t, i) => {
      const T = Number(t);
      const k = Number.isFinite(T) ? shots.findIndex((sh, q) => { const [a, b] = shotSpan(sh); return T >= a - 1e-6 && (T < b - 1e-6 || (q === shots.length - 1 && T <= b + 1e-6)); }) : -1;
      if (k >= 0) by.get(shots[k].n).push(i); else outside.push(i);
    });
    return { by, outside };
  }
  /** every token from a to b (inclusive, either order) goes on `shot` (a shift-click / drag over word chips = a phrase); tokens of other shots move here. */
  function rangeWordShot(sel, a, b, shot) {
    const lo = Math.min(a, b), hi = Math.max(a, b), m = new Map((sel || []).map((t) => [t.i, t]));
    for (let i = lo; i <= hi; i++) m.set(i, { i, phrase: shot, shot });
    return normSelect([...m.values()]);
  }
  /** the shot's words off (the "clear" button). */
  const clearShotWords = (sel, shot) => normSelect((sel || []).filter((t) => t.shot !== shot));
  /** the next-best auto pick for a shot (the cycle button): story.word_alts[shot] is ranked best first; the one after the shot's current words (the first when the current words are not an alternative);
   *  an alternative that uses a word another shot holds is skipped. -> the new selection, or null when there is no other idea. */
  function cycleShotWords(story, shot) {
    const alts = (story && story.word_alts && story.word_alts[String(shot)]) || [], sel = (story && story.text_select) || [];
    if (!alts.length) return null;
    const cur = shotTokens(sel, shot).join(","), at = alts.findIndex((a) => a.t.join(",") === cur);
    for (let q = 1; q <= alts.length; q++) {
      const alt = alts[(at + q) % alts.length];
      if (alt.t.join(",") === cur || sel.some((t) => t.shot !== shot && alt.t.includes(t.i))) continue;
      return normSelect(sel.filter((t) => t.shot !== shot).concat(alt.t.map((i) => ({ i, phrase: shot, shot }))));
    }
    return null;
  }
  /** a hit for a hand-chosen line on a video shot: it lands when its first word is sung (piece second = shot start + (sung - cut start), snapped forward to a beat within 0.25 s, never before it is sung),
   *  is held to its last word + 0.7 s, all inside the shot. No sung time (or sung outside this cut) = the shot's start, held to its end. times = sung second per token (source clock), beats = the grid (piece clock). */
  function landHit(sh, text, idx, times, beats) {
    const base = { text, land: sh.start, hold_until: sh.end };
    const ts = (idx || []).map((i) => Number(times && times[i])).filter(Number.isFinite);
    if (!ts.length || !isVideoShot(sh)) return base;
    const t0 = Math.min(...ts), t1 = Math.max(...ts);
    if (t0 < sh.src_start - 1e-6 || t0 >= sh.src_end - 1e-6) return base;
    const toPiece = (t) => sh.start + (t - sh.src_start);
    let land = toPiece(t0);
    const ahead = (beats || []).filter((b) => b >= land - 1e-6 && b <= land + 0.25 + 1e-6).sort((a, b) => a - b)[0];
    if (ahead !== undefined) land = ahead;
    land = Math.max(sh.start, Math.min(land, sh.end - 0.7));
    const hold = Math.min(sh.end, Math.max(land + 0.7, toPiece(Math.min(t1, sh.src_end)) + 0.7));
    return { text, land: r2(land), hold_until: r2(hold), sung: t0 };
  }
  /** v18: the shot to give a word clicked while no shot is active: the EMPTY shot (no words, no line) nearest in time to where the word is sung / lands / sits in the text; -1 = every shot has words.
   *  Time of the word: its backend hit's `land` (a hit sung at this word's time), else its sung time (`times[i]`, seconds in the media file: compared with a video cut's source span, else with the
   *  shot's span), else (pasted text) its position in the text as a proportion of the piece. ctx = {tokens, times}. */
  function nearestEmptyShot(story, i, ctx) {
    const c = ctx || {}, sel = story.text_select || [], taken = new Set(sel.map((t) => t.shot));
    const empty = story.shots.filter((sh) => !taken.has(sh.n) && !String(sh.line || "").trim());
    if (!empty.length) return -1;
    const tm = Array.isArray(c.times) ? Number(c.times[i]) : NaN, total = Math.max(...story.shots.map((sh) => num(sh.end, 0)), 0);
    const hit = Number.isFinite(tm) ? (story.hits || []).find((h) => Number.isFinite(h.sung) && Math.abs(h.sung - tm) < 0.06) : null;
    let t, kind;
    if (hit) { t = hit.land; kind = "piece"; }
    else if (Number.isFinite(tm)) { t = tm; kind = "src"; }
    else { t = ((i + 0.5) / Math.max(1, (c.tokens || []).length)) * total; kind = "piece"; }
    const dist = (sh) => {
      const vid = isVideoShot(sh) && kind === "src", a = vid ? sh.src_start : sh.start, b = vid ? sh.src_end : sh.end;
      return (t < a ? a - t : t > b ? t - b : 0) + 1e-3 * Math.abs(t - (a + b) / 2);
    };
    return empty.reduce((best, sh) => (dist(sh) < dist(best) - 1e-9 ? sh : best), empty[0]).n;
  }
  /** the text of each phrase (edge punctuation / quotes stripped), in screen order. */
  function phraseTexts(tokens, sel) {
    const by = new Map();
    for (const t of sel || []) { if (!by.has(t.phrase)) by.set(t.phrase, []); by.get(t.phrase).push(t.i); }
    return [...by.keys()].sort((a, b) => a - b).map((k) => stripEdge(by.get(k).filter((i) => i < tokens.length).map((i) => tokens[i]).join(" "))).filter(Boolean);
  }
  /** fallback when the backend sends no text_select: find each on-screen line, in order, in the token sequence. */
  function selectFromLines(tokens, lines) {
    const norm = tokens.map((t) => stripEdge(t).toLowerCase()), out = [];
    let pos = 0, k = 0;
    for (const [li, line] of (lines || []).entries()) {
      const w = String(line || "").split(/\s+/).map((x) => stripEdge(x).toLowerCase()).filter(Boolean);
      if (!w.length) continue;
      let at = -1;
      for (let j = pos; j + w.length <= norm.length && at < 0; j++) if (w.every((x, q) => norm[j + q] === x)) at = j;
      if (at < 0) continue;
      k++; for (let q = 0; q < w.length; q++) out.push({ i: at + q, phrase: k, shot: li + 1 });
      pos = at + w.length;
    }
    return out;
  }

  const usedIds = (story) => (story ? story.shots.map((x) => x.image).filter(Boolean) : []);
  /** "12 s fits 3 shots - each line needs ~4.0 s to land and hold; pick 24 s to use all 5" */
  function capacityWhy(nImages, shots, length, options) {
    if (nImages <= shots) return "";
    const need = (options || [6, 8, 12, 16, 24]).filter((L) => shotsFor(L, nImages) >= Math.min(nImages, 6) && lengthFeasible(L, nImages)).sort((a, b) => a - b)[0];
    return `${length} s fits ${shots} shot${shots === 1 ? "" : "s"} - each line needs ~${(length / shots).toFixed(1)} s to land and hold` +
      (need ? `; pick ${need} s to use all ${nImages}` : "; the longest Length fits 6 shots");
  }
  /** v18: entries without a shot (an older backend / a stub): video mode -> the shot where the phrase lands (backend hit, else sung time), image mode -> shot = the phrase's rank. 0 = on no shot. */
  function assignShots(story, tokens, times) {
    const sel = story.text_select || [];
    if (sel.every((t) => t.shot >= 1)) return sel;
    const N = story.shots.length, rank = phraseRanks(sel);
    const vid = N > 0 && story.shots.every(isVideoShot) && Array.isArray(times);
    const placed = vid ? phrasePlacement(story.shots, tokens || [], sel, times, story.hits).placed : null;
    return sel.map((t) => {
      if (t.shot >= 1) return t;
      const at = placed ? placed.get(t.phrase) : rank.get(t.phrase) - 1;
      return Object.assign({}, t, { shot: at >= 0 && at < N ? story.shots[at].n : 0 });
    });
  }
  /** make a v1 or v2 story complete: capacity, a candidate per image (use = the shots' truth), text_select. ctx: {imageIds, length, lengthOptions, tokens} */
  function completeStory(story, ctx) {
    const s = clone(story), c = ctx || {}, ids = c.imageIds || [], N = s.shots.length;
    if (!s.capacity) s.capacity = { shots: N, why: "" };
    if (!s.capacity.why) s.capacity.why = capacityWhy(ids.length, N, Number(c.length) || 12, c.lengthOptions);
    const old = new Map(s.candidates.map((x) => [x.image, x.reason])), used = usedIds(s);
    const all = ids.length ? ids : [...old.keys()];
    s.candidates = all.map((id) => ({ image: id, use: used.includes(id), reason: used.includes(id) ? "" : (old.get(id) || s.capacity.why || "not used") }));
    const tokens = c.tokens || [];
    s.text_select = s.text_select.length || !tokens.length ? normSelect(s.text_select) : normSelect(selectFromLines(tokens, s.shots.map((x) => x.line)));
    s.text_select = assignShots(s, tokens, c.times);
    return s;
  }
  /** re-derive a story WITHOUT a model call after the user changed which frames (o.use = ids in shot order) or which words (o.select) are used.
   *  A swapped-in frame gets the read's note for it as its beat; a changed phrase becomes the shot's line. Returns {story, imageChanged:[n], lineChanged:[n]}. */
  function deriveStory(story, o) {
    const s = clone(story), x = o || {}, N = s.shots.length, imageChanged = [], lineChanged = [];
    if (Array.isArray(x.use)) {
      const ids = x.imageIds || s.candidates.map((c) => c.image), list = [];
      for (const id of x.use) if (ids.includes(id) && !list.includes(id)) list.push(id);
      list.length = Math.min(list.length, N);
      for (const id of ids) if (list.length < N && !list.includes(id)) list.push(id);
      while (list.length < N) list.push(null);
      s.shots.forEach((sh, k) => {
        if (sh.image === list[k]) return;
        sh.image = list[k]; sh.beat = (x.shows && list[k] && x.shows[list[k]]) || ""; imageChanged.push(sh.n);
      });
      const why = s.capacity ? s.capacity.why : "";
      const prev = new Map(s.candidates.map((c) => [c.image, c.reason]));
      s.candidates = ids.map((id) => ({ image: id, use: list.includes(id), reason: list.includes(id) ? "" : (prev.get(id) || why || "not used") }));
    }
    if (Array.isArray(x.select)) {                                   // v18: the words of shot n = the tokens whose `shot` is n, in token order; no time-based placement
      const prevSel = s.text_select || [];
      s.text_select = normSelect(x.select).filter((t) => s.shots.some((sh) => sh.n === t.shot));
      // a shot whose tokens did not change keeps its line exactly as it was (the backend's cleaned UPPER CASE, a hand edit); only the shots the click touched are re-derived
      const lines = s.shots.map((sh) => (shotTokens(prevSel, sh.n).join() === shotTokens(s.text_select, sh.n).join() && sh.line
        ? sh.line : stripEdge(shotTokens(s.text_select, sh.n).filter((i) => i < (x.tokens || []).length).map((i) => x.tokens[i]).join(" "))));
      if (s.shots.length > 0 && s.shots.every(isVideoShot)) {         // the hits follow: an unchanged line keeps its beat-snapped hit, a changed one lands at its shot's start
        const prior = new Map(s.shots.map((sh, k) => [k, hitsInShot(s, k)])), hits = [];
        s.shots.forEach((sh, k) => {
          if (!lines[k]) return;
          const old = prior.get(k);
          if (old.length && old.map((h) => h.text).join(" ").toUpperCase() === lines[k].toUpperCase()) hits.push(...old); else hits.push(landHit(sh, lines[k], shotTokens(s.text_select, sh.n), x.times, x.beats));
        });
        s.hits = hits.sort((a, b) => a.land - b.land);
      }
      s.shots.forEach((sh, k) => { const t = lines[k] || ""; if (sh.line !== t) { sh.line = t; lineChanged.push(sh.n); } });
    }
    return { story: s, imageChanged, lineChanged };
  }

  // editable-with-revert fields: "title" | "logline" | "look" | "music" | "<n>.line|beat|camera|font" | "sel" | "use"
  const SHOT_FIELD = /^(\d+)\.(line|beat|camera|font)$/;
  function fieldGet(story, key) {
    const m = SHOT_FIELD.exec(key);
    if (m) { const sh = story.shots.find((x) => x.n === Number(m[1])); return sh ? sh[m[2]] : undefined; }
    return story[key];
  }
  function fieldSet(story, key, val) {
    const m = SHOT_FIELD.exec(key);
    if (m) { const sh = story.shots.find((x) => x.n === Number(m[1])); if (sh) sh[m[2]] = val; }
    else story[key] = val;
  }
  /** is the field different from Gemma's latest value AND touched by the user? (edited = {key: true}) */
  function fieldEdited(edited, story, gemma, key) {
    if (!edited || !edited[key] || !story || !gemma) return false;
    if (key === "sel") return JSON.stringify(story.text_select) !== JSON.stringify(gemma.text_select);
    if (key === "use") return usedIds(story).join() !== usedIds(gemma).join();
    const a = fieldGet(story, key), b = fieldGet(gemma, key);
    return a !== undefined && b !== undefined && a !== b;
  }
  /** a copy of `story` with one field back at Gemma's value ("use" / "sel" also take back the beats / lines that followed them). */
  function revertField(story, gemma, key) {
    const s = clone(story);
    if (key === "sel") {
      s.text_select = clone(gemma.text_select);
      for (const sh of s.shots) { const g = gemma.shots.find((x) => x.n === sh.n); if (g) sh.line = g.line; }
      if (gemma.hits) s.hits = clone(gemma.hits);                     // video mode: the landings go back with the words
    } else if (key === "use") {
      for (const sh of s.shots) { const g = gemma.shots.find((x) => x.n === sh.n); if (g && g.image !== sh.image) { sh.image = g.image; sh.beat = g.beat; } }
      s.candidates = clone(gemma.candidates);
    } else {
      const v = fieldGet(gemma, key);
      if (v !== undefined) fieldSet(s, key, v);
    }
    return s;
  }

  // VERSIONS of a model-written text field (STORYBOARD v3, replaces the per-field revert): v = {g: [Gemma's values, one per story that brought a new one], u: the user's latest edit | null, at: index into g | "u"}
  const verList = (v) => (v.u === null || v.u === undefined ? v.g.slice() : [...v.g, v.u]);
  const verIndex = (v) => (v.at === "u" && v.u !== null && v.u !== undefined ? v.g.length : Math.max(0, Math.min(Number(v.at) || 0, v.g.length - 1)));
  /** a copy of v brought in line with what the field shows now (cur) and Gemma's latest value: Gemma's value joins g when new; a value that is neither is the user's version
   *  (typing = the user is typing right now: a value that is back at one of Gemma's versions drops the user's version). */
  function verSync(v, cur, gemma, typing) {
    const s = v ? { g: v.g.slice(), u: v.u === undefined ? null : v.u, at: v.at } : { g: [], u: null, at: 0 };
    if (gemma !== undefined && gemma !== null && !s.g.includes(gemma)) s.g.push(gemma);
    if (s.u !== null && s.g.includes(s.u)) s.u = null;
    const gi = s.g.lastIndexOf(cur);
    if (cur === s.u) s.at = "u";
    else if (gi >= 0) { s.at = gi; if (typing) s.u = null; }
    else { s.u = cur; s.at = "u"; }
    return s;
  }
  /** one arrow step (dir -1 | +1): null at the ends, else {v, value, user} (user = the shown version is the user's edit). */
  function verStep(v, dir) {
    const list = verList(v), i = verIndex(v) + dir;
    if (i < 0 || i >= list.length) return null;
    const user = i >= v.g.length;
    return { v: { g: v.g.slice(), u: v.u, at: user ? "u" : i }, value: list[i], user };
  }

  // fonts (the liked ones only: GET /mograph/fonts/liked -> [{file, name, family_css, role}])
  const fontRole = (fonts, file) => { const f = fonts.find((x) => x.file === file); return f ? f.role : ""; };
  /** one liked font file, random (never barcode / risky unless nothing else), not in `taken`, preferably not of a role in `avoidRoles` (lines must contrast). rnd: () => [0,1). */
  function fontPick(fonts, taken, avoidRoles, rnd) {
    const r = rnd || Math.random;
    // never a random default: a barcode face (its glyphs are stripes) or a `risky` one (dotted / thin / outline-only; a default is shown on the storyboard only, it is not a font pin: briefFromStory marks the user's own picks `font_user`); ‹ › (fontStep) still reaches every face
    const good = fonts.filter((f) => !/barcode/i.test(f.file) && !f.risky);
    const pool = good.length ? good : fonts;
    for (const ok of [(f) => !taken.includes(f.file) && !(avoidRoles || []).includes(f.role), (f) => !taken.includes(f.file), () => true]) {
      const c = pool.filter(ok);
      if (c.length) return c[Math.floor(r() * c.length) % c.length].file;
    }
    return "";
  }
  /** n fonts for n lines: all different, and consecutive lines of different roles. */
  function fontPickAll(fonts, n, rnd) {
    const out = [];
    for (let k = 0; k < n; k++) out.push(fontPick(fonts, out, k ? [fontRole(fonts, out[k - 1])] : [], rnd));
    return out;
  }
  /** a new font for line k of `current` (array of files): not already used, role unlike its neighbours'. */
  function fontReroll(fonts, current, k, rnd) {
    const others = current.filter((_, q) => q !== k);
    const near = [current[k - 1], current[k + 1]].filter(Boolean).map((f) => fontRole(fonts, f));
    return fontPick(fonts, [...others, current[k]], near, rnd);
  }
  function fontStep(fonts, file, dir) {
    if (!fonts.length) return "";
    const i = fonts.findIndex((f) => f.file === file);
    return fonts[i < 0 ? (dir > 0 ? 0 : fonts.length - 1) : (i + dir + fonts.length) % fonts.length].file;
  }
  /** CSS family name for a liked font (the box sends family_css; else derived from the file name). */
  const fontCss = (f) => (f && (f.family_css || ("mf-" + String(f.file).replace(/[^A-Za-z0-9]/g, ""))));

  /** the words over a picture: ink (light or dark) from the picture's luminance under the text (lum = mean relative luminance 0..1, sd = its spread 0..1);
   *  a subtle scrim only when neither ink reaches ~4.5:1 or the area is busy. */
  function legibleInk(lum, sd, cover) {
    const L = Math.min(1, Math.max(0, Number(lum) || 0)), light = 1.05 / (L + 0.05), dark = (L + 0.05) / 0.0533;
    let ink = light >= dark ? "light" : "dark";
    const best = Math.max(light, dark), c = cover && Number.isFinite(cover.light) && Number.isFinite(cover.dark) ? cover : null;
    if (c && Math.abs(c.light - c.dark) > 0.15) ink = c.light > c.dark ? "light" : "dark";       // cover = share of the text band's pixels where that ink reaches 3:1
    const own = c ? c[ink] : 1, ratio = ink === "light" ? light : dark;
    return { ink, color: ink === "light" ? "#ffffff" : "#0b0b0b", scrim: Math.max(best, ratio) < 4.5 || own < 0.8 || (!c && (Number(sd) || 0) > 0.27), contrast: ratio };
  }

  // ------------------------------------------------------------------ run stages (the run view shows ~6 plain stages, not the node list)
  const STAGES = [{ id: "story", label: "Story" }, { id: "music", label: "Music" }, { id: "pictures", label: "Pictures" },
                  { id: "motion", label: "Motion (LTX)" }, { id: "type", label: "Type" }, { id: "finish", label: "Finish" }];

  /** Which run stage a graph node belongs to (by title, then class). Anything unrecognised is LTX plumbing -> "motion". */
  function stageOf(title, cls) {
    const t = String(title || ""), c = String(cls || "");
    if (/\[s\d+ text\]/.test(t) || /^(Decode text clip|Type passes|Text-pass negative|Read the LTX letters|OUT type|OUT design|Piece from design|Design )/.test(t) || /\bdesign\b/i.test(t)
        || /^(MographDesign|MographPieceFromJSON|MographTypePasses|MographLtxRead)/.test(c)) return "type";
    if (/^(Compose|Assemble|Archive|UP |OUT final)/.test(t)
        || /^(MographCompose|MographAssemble|MographArchive|MographSaveFinalUpscaled|SaveVideo|CreateVideo|GetVideoComponents|ImageUpscaleWithModel|UpscaleModelLoader)$/.test(c)) return "finish";
    if (/\bACE\b|\bT2M\b|^(Music|Master|Tame highs|Live waveform|IN audio|Load audio)/.test(t) || /^(MographMusic|MographMaster|MographTameHighs|MographLiveWaveform|ModelSamplingAuraFlow|EmptyAceStep|VAEDecodeAudio)/.test(c)) return "music";
    if (/^(IN images?$|Still\b|First frames?$|MOSS-|Qwen3\.5|Angle sampler|OUT still|User images)/.test(t)
        || /^(MographFirstFrame|MographStillPrompt|MographUserImages|MographOptionalImage)/.test(c)) return "pictures";
    if (/^(IN |Concept\b|Request$|Plan the piece|Format sizes|OUT concept)|\bconcept\b/.test(t) || /^(MographConcept|MographRequest|MographPiecePlan|MographPieceSizes)/.test(c)) return "story";
    return "motion";
  }

  /** The shot / picture number a node title carries in its last [bracket] ("Shot slot [s2]", "Still sampler [place 3]", "... [alpha s1]", "... [s2 clean]"), else null. */
  function shotOf(title) {
    const m = String(title || "").match(/\[([^\]]*)\]\s*$/);
    const d = m && m[1].match(/(\d+)/);
    return d ? Number(d[1]) : null;
  }

  /**
   * Run stages from the tracker's nodes {id: {title, class_type, state, value, max, seconds}}.
   * -> {stages: [{id, label, total, done, frac, state: pending|running|done|error, seconds}], overall: 0..1, current: stage id|null, detail: "Motion: shot 2 of 3, step 5/8"}
   * overall = the mean of the stage fractions (equal stage weights; a stage with no nodes in this graph is left out).
   * opts: {shots: N from the story (else counted from the node titles), label(title) -> friendly text of the running node}
   */
  function stageProgress(nodes, opts) {
    opts = opts || {};
    const by = {};
    for (const s of STAGES) by[s.id] = { id: s.id, label: s.label, total: 0, done: 0, frac: 0, state: "pending", seconds: 0, _run: null, _started: 0, _shots: new Set() };
    for (const n of Object.values(nodes || {})) {
      const st = by[stageOf(n.title, n.class_type)];
      st.total++;
      const k = shotOf(n.title); if (k !== null) st._shots.add(k);
      if (n.state === "done" || n.state === "cached") { st.done++; st._started++; st.seconds += n.seconds || 0; }
      else if (n.state === "running") { st._started++; st._run = n; st.frac += n.max ? Math.min(1, Math.max(0, n.value / n.max)) : 0; }
      else if (n.state === "error") { st.state = "error"; st._started++; }
    }
    const out = [];
    for (const s of STAGES) {
      const st = by[s.id];
      if (!st.total) continue;
      st.frac = Math.min(1, (st.done + st.frac) / st.total);
      if (st.state !== "error") st.state = st.done === st.total ? "done" : (st._run ? "running" : "pending");
      out.push(st);
    }
    // between two nodes nothing "runs": the first unfinished stage that has started (or the one after the last finished) is current
    let cur = out.find((s) => s.state === "running") || out.find((s) => s.state === "error");
    if (!cur) {
      const open = out.filter((s) => s.state !== "done");
      cur = open.find((s) => s._started > 0) || (out.some((s) => s.state === "done") ? open[0] : null) || null;
      if (cur && cur.state === "pending") cur.state = "running";
    }
    const overall = out.length ? out.reduce((a, s) => a + s.frac, 0) / out.length : 0;
    let detail = "";
    if (cur) {
      const n = cur._run, label = cur.label.replace(/ \(LTX\)$/, ""), bits = [];
      if (n) {
        const k = shotOf(n.title);
        if (k !== null) bits.push(`${cur.id === "pictures" ? "picture" : "shot"} ${k} of ${Math.max(opts.shots || cur._shots.size, k)}`);
        else if (opts.label) bits.push(opts.label(n.title));
        if (n.max) bits.push(`step ${n.value}/${n.max}`);
      }
      detail = bits.length ? `${label}: ${bits.join(", ")}` : (cur.state === "error" ? `${label}: failed` : label);
    }
    for (const s of out) { delete s._run; delete s._shots; delete s._started; }
    return { stages: out, overall, current: cur ? cur.id : null, detail };
  }

  const isLinkLike = (v) => Array.isArray(v) && v.length === 2;
  /** The one input name to write on a media node: the field's override, else "value" (string primitives), else the node's only non-link input. */
  function mediaInputName(node, preferred, fallback) {
    const ins = node.inputs || {};
    if (preferred && (preferred in ins)) return preferred;
    if (fallback in ins) return fallback;
    const keys = Object.keys(ins).filter((k) => !isLinkLike(ins[k]));
    return keys.length === 1 ? keys[0] : fallback;
  }

  /**
   * Media field -> graph inputs. bundle = {images: [names], audio: name|"", brief: string|object|"" } (uploaded names).
   * image_node gets the first image, images_node the rest comma-joined, audio_node the audio, brief_node the READ JSON string.
   * A node the graph does not have (an old graph) is skipped silently.
   */
  function applyMedia(prompt, tm, field, bundle) {
    bundle = bundle || {};
    const put = (title, preferred, fallback, value) => {
      if (!title || !(title in tm)) return false;
      const node = prompt[tm[title]];
      node.inputs = node.inputs || {};
      node.inputs[mediaInputName(node, preferred, fallback)] = value;
      return true;
    };
    const images = (bundle.images || []).filter(Boolean);
    const brief = typeof bundle.brief === "string" ? bundle.brief : briefString(bundle.brief);
    put(field.image_node, field.image_input, "image", images[0] || "");
    put(field.images_node, field.images_input, "value", images.slice(1).join(","));
    put(field.audio_node, field.audio_input, "audio", bundle.audio || "");
    put(field.brief_node, field.brief_input, "value", brief);
  }

  // ------------------------------------------------------------------ App label (16_app/MEDIA_BOX.md "App label")
  const APP_DIR = "motngrfx_app";

  /** short unique id of one Make it / one read: base36 timestamp + 3 random chars, e.g. "mh3k9x2fa1". */
  function newRunId() {
    return Date.now().toString(36) + Math.floor(Math.random() * 46656).toString(36).padStart(3, "0");
  }

  const isPrefixInput = (k) => k === "prefix" || k === "output_prefix" || k.endsWith("_prefix");

  /**
   * Every file the page submits is labelled as App output: each prefix-like string input (filename_prefix, output_prefix, prefix ...)
   * becomes `motngrfx_app/<original subfolder>/<runId>_<original basename>` (video/motngrfx -> motngrfx_app/video/<run>_motngrfx), so the
   * stage name and its folder stay recognisable. Idempotent. Returns how many inputs were rewritten.
   */
  function labelPrompt(prompt, runId) {
    let n = 0;
    for (const node of Object.values(prompt)) {
      for (const [k, v] of Object.entries((node && node.inputs) || {})) {
        if (!isPrefixInput(k) || typeof v !== "string" || v.startsWith(APP_DIR + "/")) continue;
        const parts = v.replace(/\\/g, "/").split("/").filter(Boolean);
        const stem = parts.pop() || "out";
        node.inputs[k] = [APP_DIR, ...parts, `${runId}_${stem}`].join("/");
        n++;
      }
    }
    return n;
  }

  /**
   * values: field id -> UI value. uploads: field id -> name returned by /upload/image ("sub/name" or "name").
   * A media field takes uploads[id] = {images, audio, brief} (see applyMedia).
   * extra.runId = the run id of the App label (default: a new one); every save prefix is rewritten by labelPrompt.
   * Returns {prompt, missing: [field], used, runId}. Seed fields left empty get a random seed (reported back).
   */
  function buildPrompt(template, app, values, uploads, extra) {
    const prompt = clone(template);
    const runId = (extra && extra.runId) || newRunId();
    const tm = titleMap(prompt);
    const missing = [];
    const drop = [];
    const used = {};
    uploads = uploads || {};
    for (const f of app.fields || []) {
      if (f.kind === "media") { applyMedia(prompt, tm, f, uploads[f.id]); continue; }
      if (!(f.node_title in tm)) continue;
      const node = prompt[tm[f.node_title]];
      const isFile = f.kind === "image" || f.kind === "audio";
      const raw = isFile ? uploads[f.id] : values[f.id];
      if (isEmpty(f, raw)) {
        if (f.kind === "seed") {
          const s = randomSeed();
          node.inputs[f.input_name] = s;
          used[f.id] = s;
        } else if (!f.optional) {
          missing.push(f);
        } else if ((f.remove_when_empty || []).length) {
          drop.push(...f.remove_when_empty);
        } else if (f.kind === "textarea" || f.kind === "text") {
          node.inputs[f.input_name] = "";
        } else if (f.default !== undefined && f.default !== "" && f.kind !== "image" && f.kind !== "audio") {
          node.inputs[f.input_name] = coerce(f, f.default);
        }
        continue;
      }
      const v = coerce(f, raw);
      node.inputs[f.input_name] = v;
      used[f.id] = v;
    }
    removeNodes(prompt, drop);
    labelPrompt(prompt, runId);
    return { prompt, missing, used, runId };
  }

  // ------------------------------------------------------------------ server availability (/object_info)
  const nodeTitle = (n) => (n && n._meta && n._meta.title) || null;

  /** The node_packs.json entry that provides a class, or null. */
  function packFor(cls, packs) {
    for (const p of (packs && packs.packs) || []) {
      if ((p.classes || []).includes(cls)) return p;
      for (const rx of p.patterns || []) { try { if (new RegExp(rx).test(cls)) return p; } catch (_) { /* bad pattern */ } }
    }
    return null;
  }

  /**
   * Optional-node specs of a sidecar (app.optional_nodes). Entry forms: {class_type, titles: [...], note} (the form
   * tools/app/optional_nodes.py uses), or a node title / node id / class type string, or {title|node_title|id}.
   * Returns [{class_type|null, ids: Set of node ids, note}]: the ids are every node of that class plus the titled nodes.
   */
  function optionalSpecs(graph, app) {
    const out = [];
    for (const e of (app && app.optional_nodes) || []) {
      const spec = { class_type: null, ids: new Set(), note: "" };
      const keys = [];
      if (e && typeof e === "object") {
        spec.class_type = e.class_type || null; spec.note = e.note || "";
        keys.push(...(e.titles || []), e.title, e.node_title, e.id);
      } else keys.push(e);
      for (const [id, n] of Object.entries(graph)) {
        if ((spec.class_type && n.class_type === spec.class_type) || keys.some((k) => k !== undefined && k !== null && (k === id || k === nodeTitle(n) || (!spec.class_type && k === n.class_type)))) spec.ids.add(id);
      }
      if (!spec.class_type && spec.ids.size) {
        const cls = new Set([...spec.ids].map((id) => graph[id].class_type));
        if (cls.size === 1) spec.class_type = [...cls][0];
      }
      out.push(spec);
    }
    return out;
  }

  const isLink = (v) => Array.isArray(v) && v.length === 2 && (typeof v[0] === "string" || typeof v[0] === "number");

  /**
   * Compare a template with the server's /object_info. A missing class is droppable when a sidecar optional_nodes spec covers
   * all its nodes (or its pack is marked required:false in node_packs.json) AND every consumer takes the output on an optional
   * input; anything else is missing-required. A spec whose class is missing also drops its titled nodes (e.g. the brief prompt).
   * Returns {checked, ok, missing: [{class_type, ids, titles, pack, blockers}], droppable: [... + note], dropIds}.
   */
  function analyzeAvailability(graph, app, info, packs) {
    const out = { checked: !!info, ok: true, missing: [], droppable: [], dropIds: [] };
    if (!info) return out;
    const specs = optionalSpecs(graph, app);
    const byClass = {};
    for (const [id, n] of Object.entries(graph)) {
      if (n && n.class_type && !(n.class_type in info)) (byClass[n.class_type] = byClass[n.class_type] || []).push(id);
    }
    const cand = {};          // missing class -> {ids, extra: other nodes dropped with it, note}
    const candIds = new Set();
    for (const [cls, ids] of Object.entries(byClass)) {
      const pack = packFor(cls, packs);
      const spec = specs.find((sp) => ids.every((id) => sp.ids.has(id)));
      if (spec || (pack && pack.required === false)) {
        const extra = spec ? [...spec.ids].filter((id) => !(graph[id].class_type in byClass)) : [];
        cand[cls] = { extra, note: spec ? spec.note : "" };
        ids.forEach((i) => candIds.add(i)); extra.forEach((i) => candIds.add(i));
      }
    }
    for (const [cls, ids] of Object.entries(byClass)) {
      const pack = packFor(cls, packs);
      const titles = ids.map((id) => nodeTitle(graph[id]) || id);
      const blockers = [];
      if (cls in cand) {
        const mine = new Set([...ids, ...cand[cls].extra]);
        for (const [cid, cn] of Object.entries(graph)) {
          if (candIds.has(cid) || !cn.class_type || !(cn.class_type in info)) continue;
          const required = (info[cn.class_type].input || {}).required || {};
          for (const [k, v] of Object.entries(cn.inputs || {})) {
            if (isLink(v) && mine.has(String(v[0])) && k in required) blockers.push({ node: nodeTitle(cn) || cid, input: k });
          }
        }
      }
      const entry = { class_type: cls, ids, titles, pack, blockers, note: (cand[cls] || {}).note || "" };
      if (cls in cand && !blockers.length) {
        const extra = cand[cls].extra;
        entry.titles = titles.concat(extra.map((id) => nodeTitle(graph[id]) || id));
        out.droppable.push(entry); out.dropIds.push(...ids, ...extra);
      } else out.missing.push(entry);
    }
    out.ok = out.missing.length === 0;
    return out;
  }

  /** Remove the droppable nodes from a prompt: unlink their outputs from consumers, then prune upstream nodes only they fed. */
  function applyAvailability(prompt, analysis, info) {
    const drop = new Set((analysis && analysis.dropIds) || []);
    if (!drop.size) return [];
    const sources = new Set();
    for (const id of drop) {
      for (const v of Object.values((prompt[id] && prompt[id].inputs) || {})) if (isLink(v)) sources.add(String(v[0]));
      delete prompt[id];
    }
    for (const n of Object.values(prompt)) {
      for (const [k, v] of Object.entries(n.inputs || {})) if (isLink(v) && drop.has(String(v[0]))) delete n.inputs[k];
    }
    const removed = [...drop];
    const referenced = (id) => Object.values(prompt).some((n) => Object.values(n.inputs || {}).some((v) => isLink(v) && String(v[0]) === id));
    const queue = [...sources];
    while (queue.length) {
      const id = queue.pop();
      const n = prompt[id];
      if (!n || drop.has(id)) continue;
      const isOut = !!(info && info[n.class_type] && info[n.class_type].output_node);
      if (isOut || referenced(id)) continue;
      for (const v of Object.values(n.inputs || {})) if (isLink(v)) queue.push(String(v[0]));
      delete prompt[id];
      removed.push(id);
    }
    return removed;
  }

  /** Node ids in a dependency-respecting order (inputs before consumers; ties by numeric id). */
  function stageOrder(prompt) {
    const depth = {};
    const visit = (id, seen) => {
      if (depth[id] !== undefined) return depth[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      let d = 0;
      for (const v of Object.values((prompt[id] && prompt[id].inputs) || {})) {
        if (Array.isArray(v) && v.length === 2 && prompt[String(v[0])]) d = Math.max(d, visit(String(v[0]), seen) + 1);
      }
      depth[id] = d;
      return d;
    };
    const ids = Object.keys(prompt);
    ids.forEach((id) => visit(id, new Set()));
    return ids.sort((a, b) => depth[a] - depth[b] || Number(a) - Number(b));
  }

  /** Output files of one executed message: [{filename, subfolder, type, kind}]. */
  function outputFiles(output) {
    const files = [];
    if (!output) return files;
    for (const key of ["images", "gifs", "video", "videos", "audio", "files"]) {
      for (const f of output[key] || []) {
        if (f && f.filename) files.push({ filename: f.filename, subfolder: f.subfolder || "", type: f.type || "output", key });
      }
    }
    return files;
  }

  function fileKind(f) {
    const ext = (f.filename.split(".").pop() || "").toLowerCase();
    if (["mp4", "webm", "mov", "mkv"].includes(ext)) return "video";
    if (["wav", "mp3", "flac", "ogg", "m4a", "opus"].includes(ext)) return "audio";
    return "image";
  }

  function viewUrl(f) {
    const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: f.type || "output" });
    return "/view?" + q.toString();
  }

  /**
   * Tracks per-node state and durations from ComfyUI websocket messages. Times are seconds (any monotonic clock).
   * feed(msg, t) with msg = {type, data}. Idempotent per node: first start and first end win.
   */
  function createTracker(prompt) {
    const nodes = {};
    for (const id of stageOrder(prompt)) {
      const n = prompt[id];
      nodes[id] = { id, title: (n._meta && n._meta.title) || n.class_type, class_type: n.class_type,
                    state: "pending", start: null, end: null, seconds: null, value: 0, max: 0, outputs: null };
    }
    const run = { nodes, t0: null, t1: null, status: "idle", error: null, current: null };

    const start = (id, t) => {
      const n = nodes[id];
      if (!n || n.start !== null || n.state === "cached") return;
      n.start = t; n.state = "running"; run.current = id;
    };
    const end = (id, t) => {
      const n = nodes[id];
      if (!n || n.state !== "running") return;
      n.end = t; n.seconds = Math.max(0, t - n.start); n.state = "done";
    };
    const endAll = (t) => { for (const id of Object.keys(nodes)) end(id, t); };

    function feed(msg, t) {
      const d = msg.data || {};
      switch (msg.type) {
        case "execution_start":
          run.t0 = t; run.status = "running"; break;
        case "execution_cached":
          for (const id of d.nodes || []) { const n = nodes[String(id)]; if (n && n.state === "pending") { n.state = "cached"; n.seconds = 0; } }
          break;
        case "executing": {
          const id = d.node === null || d.node === undefined ? null : String(d.node);
          if (id === null) { endAll(t); break; }
          for (const other of Object.keys(nodes)) if (other !== id) end(other, t);
          start(id, t);
          break;
        }
        case "progress": {
          const id = d.node === null || d.node === undefined ? null : String(d.node);
          if (id && nodes[id]) { start(id, t); nodes[id].value = d.value; nodes[id].max = d.max; }
          break;
        }
        case "progress_state":
          for (const [key, s] of Object.entries(d.nodes || {})) {
            const id = String(s.display_node_id ?? s.node_id ?? key);
            const n = nodes[id];
            if (!n) continue;
            if (s.state === "running") { start(id, t); n.value = s.value; n.max = s.max; }
            else if (s.state === "finished") { start(id, t); end(id, t); }
          }
          break;
        case "executed": {
          const id = String(d.node);
          if (nodes[id]) { nodes[id].outputs = d.output || null; end(id, t); }
          break;
        }
        case "execution_success":
          endAll(t); run.t1 = t; run.status = "done"; break;
        case "execution_error":
          run.status = "error"; run.t1 = t;
          run.error = { node: d.node_id !== undefined ? String(d.node_id) : null, type: d.exception_type, message: d.exception_message };
          endAll(t);
          if (run.error.node && nodes[run.error.node]) nodes[run.error.node].state = "error";
          break;
        case "execution_interrupted":
          run.status = "interrupted"; run.t1 = t; endAll(t); break;
        default: break;
      }
      return run;
    }

    function summary(extra) {
      const list = Object.values(nodes).map((n) => ({
        id: n.id, title: n.title, class_type: n.class_type, state: n.state,
        seconds: n.seconds === null ? null : Math.round(n.seconds * 1000) / 1000,
      }));
      const total = run.t0 !== null && run.t1 !== null ? Math.round((run.t1 - run.t0) * 1000) / 1000 : null;
      return Object.assign({ status: run.status, total_seconds: total, nodes: list }, extra || {});
    }

    return { run, nodes, feed, summary };
  }

  // ------------------------------------------------------------------ FINAL (1080) of a finished clip (v20): side graph `_final_upscale` (tools/app/final_upscale_graph.py)
  const FINAL_LONG = [1080, 1920];                                  // 9:16; 16:9 swaps the sides (plan_rules.sizes_for("final")["compose"])
  const UP_X2 = "RealESRGAN_x2plus.pth", UP_X4 = "RealESRGAN_x4plus.pth";

  /** the same rule as final_upscale.pick_model / final_size: {target:[w,h], model, scale} for a clip of w x h; model null = the clip already is that size or bigger. */
  function finalUpscalePlan(w, h) {
    w = Number(w); h = Number(h);
    const target = w > h ? [FINAL_LONG[1], FINAL_LONG[0]] : FINAL_LONG.slice();
    const need = Math.max(target[0] / w, target[1] / h);
    if (!(need > 1)) return { target, model: null, scale: 1 };
    return need <= 2 + 1e-9 ? { target, model: UP_X2, scale: 2 } : { target, model: UP_X4, scale: 4 };
  }

  /** a /view?filename=&subfolder=&type= url -> {filename, subfolder, type}. */
  function parseViewUrl(url) {
    const q = new URLSearchParams(String(url).split("?")[1] || "");
    return { filename: q.get("filename") || "", subfolder: q.get("subfolder") || "", type: q.get("type") || "output" };
  }

  /**
   * Patch the `_final_upscale` template for ONE finished clip (the template is the 9:16 540x960 form): IN video = "<subfolder>/<name> [output]", the model and the lanczos target by size,
   * the save prefix = the clip's own folder + stem + "_up1080" (a NEW file next to it; labelPrompt keeps an App folder, else adds motngrfx_app/). -> {prompt, plan, file, prefix} or null (nothing to upscale).
   */
  function buildFinalUpscale(template, view, w, h) {
    const pl = finalUpscalePlan(w, h);
    if (!pl.model) return null;
    const g = clone(template), tm = titleMap(g);
    if (!("IN video" in tm) || !("OUT upscaled" in tm)) throw new Error("the _final_upscale graph has no IN video / OUT upscaled node");
    const f = typeof view === "string" ? parseViewUrl(view) : view;
    const sub = String(f.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
    const file = (sub ? sub + "/" : "") + f.filename + " [" + (f.type || "output") + "]";
    const stem = f.filename.replace(/\.[^.]+$/, "").replace(/_?\d{5}_?$/, "").replace(/_$/, "");
    const prefix = (sub ? sub + "/" : "") + stem + "_up1080";
    g[tm["IN video"]].inputs.file = file;
    for (const [id, n] of Object.entries(g)) {
      if (n.class_type === "UpscaleModelLoader") { n.inputs.model_name = pl.model; n._meta.title = "UP model " + pl.model; }
      else if (n.class_type === "ImageUpscaleWithModel") n._meta.title = "UP Real-ESRGAN x" + pl.scale;
      else if (n.class_type === "ImageScale") { n.inputs.width = pl.target[0]; n.inputs.height = pl.target[1]; n._meta.title = `UP lanczos to ${pl.target[0]}x${pl.target[1]}`; }
      else if (n.class_type === "SaveVideo") n.inputs.filename_prefix = prefix;
    }
    return { prompt: g, plan: pl, file, prefix };
  }

  return { clone, titleMap, visibleFields, optionValue, optionLabel, isEmpty, coerce, removeNodes, buildPrompt,
           stageOrder, outputFiles, fileKind, viewUrl, createTracker, randomSeed, packFor, optionalSpecs,
           analyzeAvailability, applyAvailability,
           MEDIA_LIMITS, mediaKindOf, mediaLimits, checkMediaAdd, planMediaDrop, mediaIds, buildMediaJson, mediaSignature, normSoundtrack, soundtrackPlan, normEdit, EDIT_OPTIONS, normalizeRead,
           briefString, applyMedia, newRunId, labelPrompt, APP_DIR, finalUpscalePlan, parseViewUrl, buildFinalUpscale,
           buildStoryRequest, storySignature, normalizeStory, fmtSpan, briefFromStory, withDirection, splitTyped,
           CAMERAS, cameraLabel, shotCountFor, CLIP_MAX_S, LENGTHS, lockupCount, shotsFor, lengthFeasible, feasibleLengths, imagesNeeded, bestFeasibleLength, capShots, VIDEO_CLIP_MAX_S, VIDEO_FRAME_OFF_S, fmtDur, videoLengthPlan, deriveVideoSpans, isVideoShot, videoStoryFix, hitsInShot, hitsFromLines, shotAtTime, lyricTokens, phrasePlacement, TEXT_CAP, textTokens, stripEdge, normSelect, phraseTexts, selectFromLines, usedIds, capacityWhy, completeStory, deriveStory,
           fieldGet, fieldSet, fieldEdited, revertField, verList, verIndex, verSync, verStep, fontRole, fontPick, fontPickAll, fontReroll, fontStep, fontCss, legibleInk, phraseRanks, toggleWordShot, shotTokens, nearestEmptyShot, assignShots, shotSpan, sungByShot, rangeWordShot, clearShotWords, cycleShotWords, landHit, STAGES, stageOf, shotOf, stageProgress };
});
