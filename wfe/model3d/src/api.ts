/** The only module that talks to the server. */
import type { Answer, Gpu, Info, Made, Progress } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

async function post(path: string, body: Record<string, unknown>): Promise<Answer> {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = (await r.json()) as Answer & { error?: { message: string } };
  if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
  return Array.isArray(j.data) ? { ...j, data: j.data.map(local) } : j;
}

/** a mesh's links relative to the page */
const local = (m: Made): Made => ({ ...m, url: here(m.url), ...(m.picture ? { picture: here(m.picture) } : {}) });

export const api = {
  info: () => getJSON<Info>("api/info"),
  progress: () => getJSON<Progress>("api/progress"),
  gpu: () => getJSON<Gpu>("api/gpu"),
  history: async () => (await getJSON<{ data: Made[] }>("api/history")).data.map(local),
  cancel: async () => { await fetch("api/cancel", { method: "POST" }); },
  /** a mesh from pictures; its answer links the GLB (an error carries the server's message) */
  generate: (body: Record<string, unknown>) => post("v1/model3d/generate", body),
  /** a mesh rigged or animated */
  rig: (body: Record<string, unknown>) => post("v1/model3d/rig", body),
};

/** A link the server gave, made relative to the page (which may be served under the front door's /open/<id>/, where
 *  the server's own absolute links would name the door's token-protected /v1/): a data: or blob: URL as it is, an
 *  http(s) one by its path, and no leading "/" */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.replace(/^\//, "");
}
