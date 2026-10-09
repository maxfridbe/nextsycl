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
const entry = /src="([^"]+)"/.exec(page.slice(page.indexOf("<script")))[1];
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
}
let panels = 0, nodes = 0;
const walkTree = (n) => { nodes++; if (n.sel?.startsWith("section") && n.data?.class?.panel) panels++; for (const c of n.children ?? []) if (typeof c === "object" && c) walkTree(c); };
walkTree(App());
console.log(`${app}: rendered ${nodes} nodes, ${panels} panels`);
if (bad || panels < 3) process.exit(1);
