/** The only module that talks to the server. */
import type { Gpu, Info, Progress, Reading } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("/api/info"),
  progress: () => getJSON<Progress>("/api/progress"),
  gpu: () => getJSON<Gpu>("/api/gpu"),
  history: async () => (await getJSON<{ data: Reading[] }>("/api/history")).data.filter((x) => x.kind === "ocr"),

  /** A reading: the pages as data URLs; an error carries the server's message. */
  async ocr(body: Record<string, unknown>): Promise<Reading> {
    const r = await fetch("/v1/ocr", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const j = (await r.json()) as Reading & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
};
