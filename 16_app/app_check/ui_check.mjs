// Headless Edge UI check of the MOTNGRFX page (Node 22+, no dependencies).
//   node 16_app/app_check/ui_check.mjs [http://127.0.0.1:8190] [outDir]
// Drives the real page over the DevTools protocol: loads /motngrfx, screenshots each state, runs the local_v1 template,
// cancels one run, drops the websocket mid-run (reconnect), injects a node error message and a validation error.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.argv[2] || "http://127.0.0.1:8190";
const OUT = process.argv[3] || dirname(fileURLToPath(import.meta.url));
const EDGE = process.env.EDGE || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9333;
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, extra = "") => { results.push({ name, ok: !!ok, extra }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); };

const prof = mkdtempSync(join(tmpdir(), "edge_ui_"));
const edge = spawn(EDGE, [`--headless=new`, `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`, "--disable-gpu",
  "--hide-scrollbars", "--window-size=1280,800", "--no-first-run", "about:blank"], { stdio: "ignore" });

async function targetWs() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch (_) { /* not up yet */ }
    await sleep(200);
  }
  throw new Error("Edge DevTools did not come up");
}

const ws = new WebSocket(await targetWs());
await new Promise((r) => (ws.onopen = r));
let nextId = 1; const pending = new Map(); const logs = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
  else if (m.method === "Runtime.exceptionThrown") logs.push("EXCEPTION " + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
};
const cdp = (method, params = {}) => new Promise((res) => { const id = nextId++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
const js = async (expr) => {
  const r = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result.exceptionDetails) throw new Error("page eval failed: " + JSON.stringify(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text));
  return r.result.result.value;
};
const shot = async (name) => {
  const r = await cdp("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(OUT, name), Buffer.from(r.result.data, "base64"));
  console.log("screenshot", name);
};
const waitFor = async (expr, ms = 60000, step = 250) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await js(expr).catch(() => false)) return true; await sleep(step); }
  return false;
};

try {
  await cdp("Page.enable"); await cdp("Runtime.enable");
  await cdp("Emulation.setDeviceMetricsOverride", { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false });
  await cdp("Page.navigate", { url: BASE + "/motngrfx" });
  check("page boots (template list loaded)", await waitFor(`!!(window.MographApp && MographApp.state.tpl)`, 20000));
  check("title is MOTNGRFX", (await js(`document.title`)) === "MOTNGRFX");
  check("app name is MOTNGRFX", (await js(`document.querySelector('.app-name').textContent`)) === "MOTNGRFX");
  await sleep(1500);
  await shot("01_examples_default_template.png");
  const defaultTpl = await js(`MographApp.state.tpl.name`);
  const pre = await js(`JSON.stringify({ok: MographApp.state.pre && MographApp.state.pre.ok, missing: (MographApp.state.pre?.missing||[]).map(m=>m.class_type), dropped: (MographApp.state.pre?.droppable||[]).map(m=>m.class_type), btn: document.querySelector('#btn-run').textContent, disabled: document.querySelector('#btn-run').disabled})`);
  console.log("default template", defaultTpl, pre);

  // switch to the CPU template that runs anywhere
  await js(`(()=>{const s=document.querySelector('#dev-graph'); s.value='local_v1'; s.dispatchEvent(new Event('change'));})()`);
  check("local_v1 loads", await waitFor(`MographApp.state.tpl.name==='local_v1' && MographApp.state.pre!==null`, 15000));
  await js(`document.querySelector('.ex').click()`);
  await sleep(1000);
  await shot("02_example_loaded_local_v1.png");
  check("Run enabled for local_v1", await js(`!document.querySelector('#btn-run').disabled`));

  // run 1: full run
  await js(`document.querySelector('#btn-run').click()`);
  await sleep(300);
  check("Run disabled while running", await js(`document.querySelector('#btn-run').disabled`));
  check("Cancel visible while running", await js(`!document.querySelector('#btn-stop').hidden`));
  await waitFor(`MographApp.state.run && MographApp.state.run.status==='running'`, 15000);
  await sleep(1500);
  await shot("03_running.png");
  // drop the websocket mid-run: the page must reconnect and still finish
  await js(`MographApp.state.run.ws && MographApp.state.run.ws.close()`);
  await sleep(300);
  const connShown = await js(`!document.querySelector('#conn').hidden`);
  const t0 = Date.now();
  const done = await waitFor(`MographApp.state.run && MographApp.state.run.status==='done'`, 120000);
  console.log("run 1 took", ((Date.now() - t0) / 1000).toFixed(1), "s after the drop; status", await js(`MographApp.state.run.status`));
  check("run finishes after a websocket drop", done, `conn badge shown during drop: ${connShown}`);
  await sleep(1500);
  check("result view with player", await js(`!!document.querySelector('.player video')`));
  await sleep(1200);
  const meta = await js(`document.querySelector('.meta') && document.querySelector('.meta').textContent`);
  check("resolution/duration shown under the player", /\d+×\d+ · [\d.]+ s/.test(meta || ""), meta);
  check("Run again / New seed buttons", await js(`document.querySelectorAll('.actions .ghost-btn').length===2`));
  await shot("04_result.png");

  // run 2: cancel
  await js(`MographApp.runNewSeed()`);
  await waitFor(`MographApp.state.run && MographApp.state.run.status==='running'`, 15000);
  await sleep(700);
  await js(`document.querySelector('#btn-stop').click()`);
  const cancelled = await waitFor(`MographApp.state.run.status==='cancelled'`, 15000);
  check("Cancel ends the run as cancelled", cancelled);
  await sleep(400);
  const q = await (await fetch(BASE + "/queue")).json();
  check("server queue is empty after cancel", (q.queue_running || []).length === 0 && (q.queue_pending || []).length === 0);
  check("Run enabled again after cancel", await js(`!document.querySelector('#btn-run').disabled`));
  await shot("05_cancelled.png");

  // run 3: injected node error (simulated execution_error websocket message)
  await js(`MographApp.run()`);
  await waitFor(`MographApp.state.run && MographApp.state.run.status==='running'`, 15000);
  await sleep(500);
  await js(`(()=>{const r=MographApp.state.run; r.ws.onmessage({data: JSON.stringify({type:'execution_error', data:{prompt_id:r.promptId, node_id:'4', node_type:'MographPlanRules', exception_type:'torch.OutOfMemoryError', exception_message:'CUDA out of memory. Tried to allocate 2.00 GiB', traceback:['Traceback (most recent call last):\\n','  File "x.py", line 1\\n']}})});})()`);
  await sleep(500);
  const errText = await js(`document.querySelector('.err-box') && document.querySelector('.err-box').innerText`);
  check("node error shows node title + message + hint", /Failed in "Plan the cut"/.test(errText || "") && /MographPlanRules/.test(errText || "") && /out of memory/i.test(errText || "") && /Draft/.test(errText || ""), (errText || "").split("\n").slice(0, 3).join(" | "));
  await shot("06_node_error.png");
  await js(`fetch('/interrupt',{method:'POST'})`);
  await sleep(2500);

  // validation error from the server (bad combo value): readable list, not a raw dump
  await js(`(()=>{MographApp.state.tpl.graph['6'].inputs.format='bogus-format';})()`);
  await js(`MographApp.run()`);
  await waitFor(`MographApp.state.run && MographApp.state.run.status==='error'`, 15000);
  await sleep(500);
  const verr = await js(`document.querySelector('.err-box') && document.querySelector('.err-box').innerText`);
  check("validation error names the node", /Render/.test(verr || ""), (verr || "").split("\n").slice(0, 3).join(" | "));
  await shot("07_validation_error.png");

  // fusion template on this box (no GPU nodes): missing nodes are listed with the pack to install
  await js(`(()=>{const s=document.querySelector('#dev-graph'); s.value='fusion'; s.dispatchEvent(new Event('change'));})()`);
  await waitFor(`MographApp.state.tpl.name==='fusion' && MographApp.state.pre!==null`, 15000);
  await sleep(800);
  const pf = await js(`document.querySelector('#preflight').innerText`);
  console.log("fusion preflight:\n" + pf);
  await shot("08_fusion_preflight.png");
  check("missing-node message lists the pack", /ComfyUI-LTXVideo/.test(pf) && /Run button|Install/i.test(await js(`document.querySelector('#btn-run').textContent`)) );
} catch (e) {
  check("script ran to the end", false, String(e && e.stack || e));
} finally {
  writeFileSync(join(OUT, "ui_check_console.log"), logs.join("\n"));
  writeFileSync(join(OUT, "ui_check_results.json"), JSON.stringify(results, null, 1));
  try { await cdp("Browser.close"); } catch (_) { /* closing */ }
  edge.kill();
  const bad = results.filter((r) => !r.ok).length;
  console.log(`\n${results.length - bad}/${results.length} checks passed`);
  process.exit(bad ? 1 : 0);
}
