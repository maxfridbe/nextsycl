// Headless Chrome against the wfe Conversation page with a fake microphone (a WAV played as the capture device);
// CHROME names the binary (default: Playwright's Chrome for Testing under ~/.cache/ms-playwright).
//   node page_test.mjs URL WAV SECONDS OUTDIR [VOICE]
// Instruments the page before it loads: every WebSocket message both ways (types, sizes, the session.update), every
// AudioBufferSourceNode.start (when, length, the context's time then, the buffer's level) - so the play-out can be
// checked for gaps - and the microphone's track settings. Clicks the voice chip and "Start talking", waits, clicks
// Stop, writes OUTDIR/page.json and OUTDIR/heard.wav (what the page scheduled, laid out at its start times).
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";

const [url, wav, secs, out, voice] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const chrome = process.env.CHROME || `${process.env.HOME}/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`;
const prof = `${out}/profile`;
rmSync(prof, { recursive: true, force: true });
const port = 9333 + Math.floor(Math.random() * 500);
const proc = spawn(chrome, [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, "--no-first-run", "--no-default-browser-check",
  "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav}%noloop`,
  "--ignore-certificate-errors", "--autoplay-policy=no-user-gesture-required", "--mute-audio", "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let errlog = "";
proc.stderr.on("data", (d) => { errlog += d; });
console.log("chrome pid", proc.pid);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => { ws.onopen = r; });
let id = 0;
const waiting = new Map();
const consoleLines = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled") consoleLines.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
  if (m.method === "Runtime.exceptionThrown") consoleLines.push("EXC " + JSON.stringify(m.params.exceptionDetails).slice(0, 400));
};
const cdp = (method, params = {}) => new Promise((r) => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

const HOOK = `(() => {
  const L = window.__ns = { sent: [], recv: [], starts: [], update: null, track: null, ctxRate: null, errors: [] };
  const OW = window.WebSocket;
  window.WebSocket = function (u, p) {
    const s = p ? new OW(u, p) : new OW(u);
    L.url = u;
    const send = s.send.bind(s);
    s.send = (d) => { try { const v = JSON.parse(d); if (v.type === "session.update" && !L.update) L.update = v;
      L.sent.push({ t: performance.now(), type: v.type, n: v.audio ? v.audio.length : 0 }); } catch {} send(d); };
    s.addEventListener("message", (e) => { try { const v = JSON.parse(e.data);
      L.recv.push({ t: performance.now(), type: v.type, n: v.delta ? v.delta.length : 0, text: v.type && v.type.includes("transcript") ? v.delta : undefined,
                    err: v.error ? v.error.message : undefined }); } catch {} });
    return s;
  };
  window.WebSocket.prototype = OW.prototype; Object.assign(window.WebSocket, { OPEN: 1, CLOSED: 3, CONNECTING: 0, CLOSING: 2 });
  const OS = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (when) {
    const b = this.buffer; const c = b.getChannelData(0); let e = 0; for (const x of c) e += x * x;
    L.starts.push({ when, len: b.length, rate: b.sampleRate, now: this.context.currentTime, db: 10 * Math.log10(e / c.length + 1e-12),
                    pcm: Array.from(c, (x) => Math.round(x * 32767)) });
    L.ctxRate = this.context.sampleRate; L.ctxState = this.context.state;
    return OS.apply(this, arguments);
  };
  const OG = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => { const s = await OG(c); L.constraints = c; L.track = s.getAudioTracks()[0]?.getSettings(); return s; };
})();`;
await cdp("Page.enable");
await cdp("Runtime.enable");
await cdp("Page.addScriptToEvaluateOnNewDocument", { source: HOOK });
await cdp("Page.navigate", { url });
await sleep(4000);
const clickText = (txt) => ev(`(() => { const b = [...document.querySelectorAll("button")].find((x) => x.textContent.includes(${JSON.stringify(txt)})); if (!b) return false; b.click(); return true; })()`);
// the Conversation panel may sit behind a tab
const tabs = await ev(`[...document.querySelectorAll("button")].map((b) => b.textContent.trim()).filter((t) => t.length < 40)`);
console.log("buttons", JSON.stringify(tabs));
if (voice) console.log("voice chip", await clickText("★ " + voice));
await sleep(300);
console.log("start", await clickText("Start talking"));
const t0 = Date.now();
await sleep(Number(secs) * 1000);
const phase = await ev(`document.querySelector(".converse")?.innerText?.slice(0, 2000)`);
console.log("stop", await clickText("Stop"));
await sleep(500);
const L = await ev(`window.__ns`);
// what the page scheduled, at its start times (seconds of context time from the first start)
const st = L.starts;
let heard = [];
if (st.length) {
  const z = st[0].when, rate = st[0].rate;
  const n = Math.ceil((st[st.length - 1].when - z) * rate) + st[st.length - 1].len + 1;
  heard = new Int16Array(n);
  for (const s of st) { const o = Math.round((s.when - z) * rate); s.pcm.forEach((x, i) => { if (o + i < n) heard[o + i] += x; }); }
  const hdr = Buffer.alloc(44);
  hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + n * 2, 4); hdr.write("WAVEfmt ", 8); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22);
  hdr.writeUInt32LE(rate, 24); hdr.writeUInt32LE(rate * 2, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34); hdr.write("data", 36); hdr.writeUInt32LE(n * 2, 40);
  writeFileSync(`${out}/heard.wav`, Buffer.concat([hdr, Buffer.from(heard.buffer)]));
}
// play-out: gaps between one buffer's end and the next one's start, how far ahead each was scheduled
const gaps = [], ahead = [];
for (let i = 0; i < st.length; i++) {
  ahead.push(st[i].when - st[i].now);
  if (i) gaps.push(st[i].when - (st[i - 1].when + st[i - 1].len / st[i - 1].rate));
}
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null; };
const appends = L.sent.filter((s) => s.type === "input_audio_buffer.append");
const deltas = L.recv.filter((r) => r.type === "response.audio.delta");
const res = {
  dropped: deltas.length - st.length,
  url: L.url, update: L.update, constraints: L.constraints, track: L.track, ctxRate: L.ctxRate, ctxState: L.ctxState,
  appends: appends.length, append_b64_total: appends.reduce((a, s) => a + s.n, 0),
  mic_seconds_sent: appends.reduce((a, s) => a + s.n, 0) * 3 / 4 / 2 / (L.ctxRate || 48000),
  deltas: deltas.length, starts: st.length, frame_len: st[0]?.len, frame_rate: st[0]?.rate,
  gaps_ms: { n_over_1ms: gaps.filter((g) => g > 0.001).length, total_gap_ms: 1000 * gaps.filter((g) => g > 0.001).reduce((a, g) => a + g, 0),
             overlaps: gaps.filter((g) => g < -0.001).length, max_ms: 1000 * Math.max(0, ...gaps) },
  ahead_ms: { min: 1000 * Math.min(...ahead), median: 1000 * q(ahead, 0.5), p95: 1000 * q(ahead, 0.95), max: 1000 * Math.max(...ahead), last: 1000 * ahead[ahead.length - 1] },
  loud_frames: st.filter((s) => s.db > -40).length,
  events: Object.entries(L.recv.reduce((a, r) => { a[r.type] = (a[r.type] || 0) + 1; return a; }, {})),
  errors: L.recv.filter((r) => r.err).map((r) => r.err),
  text: L.recv.filter((r) => r.text !== undefined).map((r) => r.text).join(""),
  panel: phase, console: consoleLines.slice(0, 40), wall_s: (Date.now() - t0) / 1000,
};
writeFileSync(`${out}/page.json`, JSON.stringify(res, null, 1));
console.log(JSON.stringify({ ...res, panel: res.panel?.slice(0, 600) }, null, 1));
ws.close();
proc.kill("SIGTERM");
setTimeout(() => process.exit(0), 500);
