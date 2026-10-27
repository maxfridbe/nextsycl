/** The only module that talks to the server. */
import type { GenAnswer, Gpu, Info, Picture, Progress } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("api/info"),
  progress: () => getJSON<Progress>("api/progress"),
  gpu: () => getJSON<Gpu>("api/gpu"),
  history: async () => (await getJSON<{ data: Picture[] }>("api/history")).data.map((p) => ({ ...p, url: here(p.url) })),

  /** OpenAI's images endpoint (an edit: the edits one, its pictures as data URLs), with our fields; an error carries
   *  the server's message. */
  async generate(body: Record<string, unknown>, edit = false): Promise<GenAnswer> {
    const r = await fetch(edit ? "v1/images/edits" : "v1/images/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as GenAnswer & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return { ...j, data: (j.data ?? []).map((d) => (d.url ? { ...d, url: here(d.url) } : d)) };
  },
};

/** A link the server gave, made relative to the page (which may be served under the front door's /open/<id>/, where
 *  the server's own absolute links would name the door's token-protected /v1/): a data: or blob: URL as it is, an
 *  http(s) one by its path, and no leading "/" */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.replace(/^\//, "");
}

