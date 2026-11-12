// The web front end's smoke test (H3's check.mjs, for nextsycl's apps): against a running server (NS_WFE, default
// http://localhost:8086 - the image page; an audio server's page works the same), fetch the page and every ES module
// it imports, check that each parses, then render the app headlessly from a made-up state - App() must build a vnode
// tree with its panels, no DOM needed until patch().
import { mkdir, writeFile, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";

const base = process.env.NS_WFE || "http://localhost:8086";
const root = "/tmp/nswfecheck";
await rm(root, { recursive: true, force: true });
await mkdir(root, { recursive: true });
await writeFile(join(root, "package.json"), '{"type":"module"}');

const page = await (await fetch(base + "/")).text();
if (page.includes("__CSS__")) throw new Error("the page's stylesheet was not inlined");
// the page's paths are relative (it is also served under /open/<id>/): the module's own, from the root
const entry = "/" + /src="([^"]+)"/.exec(page.slice(page.indexOf("<script")))[1].replace(/^\.?\//, "");
const seen = new Set();
const norm = (dir, spec) => {
  const out = [];
  for (const p of (dir + "/" + spec).split("/")) { if (p === "..") out.pop(); else if (p !== "." && p !== "") out.push(p); }
  return "/" + out.join("/");
};
async function walk(path) {
  if (seen.has(path)) return;
  seen.add(path);
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  const src = await r.text();
  const file = join(root, path.replace(/^\/ui\//, ""));
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, src);
  const dir = path.slice(0, path.lastIndexOf("/"));
  for (const m of src.matchAll(/from\s+"([^"]+)"/g)) if (m[1].startsWith(".")) await walk(norm(dir, m[1]));
}
await walk(entry);
console.log(`${seen.size} modules fetched`);
let bad = 0;
for (const p of seen) {
  try { execFileSync(process.execPath, ["--check", join(root, p.replace(/^\/ui\//, ""))], { stdio: "pipe" }); }
  catch (e) { bad++; console.log("PARSE FAIL", p, String(e.stderr).split("\n")[2] ?? ""); }
}
console.log(bad ? `${bad} modules failed to parse` : "all modules parse as ES modules");

globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.document = { getElementById: () => null };
// the app the page loads: /ui/<app>/src/main.js
const app = entry.split("/")[2];
const { App } = await import(join(root, `${app}/src/components/App.js`));
const { state } = await import(join(root, `${app}/src/state.js`));
state.info = await (await fetch(base + "/api/info")).json();
state.gpu = await (await fetch(base + "/api/gpu")).json();
state.busy = true;
if (app === "audio") {
  state.progress = { busy: true, phase: "tokens", at: 300, of: 750, seconds: 15.2 };
  const song = { url: "/v1/audio/files/x.wav", prompt: "Genre: folk", lyrics: "[verse]\nla la", seed: 7, steps: 30, cfg: 1.7, seconds: 30, took: 60, wh: 3.6, created: 0 };
  state.result = song;
  state.history = [song, { ...song, seed: 8 }];
} else {
  state.progress = { busy: true, picture: 1, pictures: 2, at: 12, of: 40, seconds: 8.4 };
  state.form.loras = { [state.info.loras?.[0]?.id ?? "x"]: 1 };
  const pic = { url: "/v1/images/files/x.png", prompt: "a fox", seed: 7, steps: 40, width: 1024, height: 1024, loras: [], seconds: 17, wh: 1.3, created: 0 };
  state.results = [pic, { ...pic, seed: 8 }];
  state.history = [pic];
  state.zoom = pic;
  await describeChecks(state);
}
let panels = 0, nodes = 0;
const walkTree = (n) => { nodes++; if (n.sel?.startsWith("section") && n.data?.class?.panel) panels++; for (const c of n.children ?? []) if (typeof c === "object" && c) walkTree(c); };
walkTree(App());
console.log(`${app}: rendered ${nodes} nodes, ${panels} panels`);
if (bad || panels < 3) process.exit(1);

// the image page's Describe panel: its rules (describe.ts) and the panel rendered from a made-up answer
async function describeChecks(state) {
  const D = await import(join(root, `${app}/src/describe.js`));
  const eq = (a, b, what) => { if (JSON.stringify(a) !== JSON.stringify(b)) { console.log("DESCRIBE FAIL", what, JSON.stringify(a), "!=", JSON.stringify(b)); process.exit(1); } };
  const z = { model: "z-image-turbo", arch: "z-image", guidance: false, grid: 16 };
  const h3 = { model: "minimax-h3-image", arch: "minimax-h3-image", guidance: false, grid: 32 };
  const qi = { model: "qwen-image-2.1-q8", arch: "qwen-image-2.1", guidance: true, grid: 16 };
  eq(D.targetOf("", z), "z-image-turbo", "the page's model is the default target");
  eq(D.targetOf("tags", z), "tags", "a chosen family");
  eq([D.familyOf("z-image-turbo", z), D.familyOf("h3", z), D.familyOf("qwen-image-2.1-q8", qi), D.familyOf("some-other-model", z)], ["z-image", "h3", "qwen-image", "natural"], "families");
  eq([D.takesNegative("z-image-turbo", z), D.takesNegative("qwen-image-2.1-q8", qi), D.takesNegative("natural", z), D.takesNegative("h3", z), D.takesNegative("tags", z)],
     [false, true, true, false, true], "negative prompts");
  // the size is applied only for this page's model or family, on its grid
  eq(D.sizeToApply({ width: 1184, height: 880 }, "z-image-turbo", z), [1184, 880], "size on the grid");
  eq(D.sizeToApply({ width: 1184, height: 896 }, "h3", h3), [1184, 896], "size on the h3 grid (32)");
  eq(D.sizeToApply({ width: 1184, height: 880 }, "h3", h3), null, "off the h3 grid (32)");
  eq(D.sizeToApply({ width: 1024, height: 1024 }, "qwen-image", z), null, "another family's size is not ours");
  eq(D.sizeToApply(null, "z-image-turbo", z), null, "no suggestion");
  eq(D.errorText({ code: "conflict", message: "target h3: no model of the family is enabled here" }).includes("409"), true, "conflict explained");
  eq(D.errorText({ code: "invalid_params", message: "x", field: "describer" }), "invalid_params: x (field describer)", "refusal shape");
  eq(D.errorText({ code: "cancelled", message: "cancelled" }), "cancelled", "cancelled");
  eq(D.jobStage({ state: "queued", waiting_for: "2 jobs ahead" }, 1), "waiting: 2 jobs ahead", "queued");
  eq(D.jobStage({ state: "running", stage: "loading" }, 20).includes("45 s"), true, "cold note");
  // the panel itself, with an answer to show
  const d = state.describe;
  d.list = { default: "best", best: "qwen3.8-27b-q4-describe", describers: [{ id: "qwen3.8-27b-q4-describe", name: "q", kind: "chat", size_gb: 18, vram_gib: 20, image_models: [], downloaded: true, enabled: true, best: true }] };
  d.picture = { url: "data:image/png;base64,AA==", name: "barn.png", w: 1024, h: 768 };
  d.result = { prompt: "A red barn.", negative_prompt: null, target: "z-image", target_model: "z-image-turbo", describer: "best", describer_model: "q", model: "q", words: 3, width: 1024, height: 768,
               suggested_size: { width: 1184, height: 880, size: "1184x880" }, timings: { pp_tok_s: 797, tg_tok_s: 60, seconds: 3.2 }, finish: "stop", notes: [] };
  d.prompt = d.result.prompt;
  state.panels.describe = true;
  const A = await import(join(root, `${app}/src/actions.js`));
  state.info = { ...state.info, ...z };
  A.useDescription();
  eq([state.form.prompt, state.form.width, state.form.height], ["A red barn.", 1184, 880], "use as prompt fills the form");
  eq(Object.keys(A.describeBody()).sort(), ["detail", "image", "target"], "the request carries only what differs");
}
