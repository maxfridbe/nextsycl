/** The only module that talks to the server. */
import type { GenAnswer, Gpu, Info, Picture, Progress } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("/api/info"),
  progress: () => getJSON<Progress>("/api/progress"),
  gpu: () => getJSON<Gpu>("/api/gpu"),
  history: async () => (await getJSON<{ data: Picture[] }>("/api/history")).data,

  /** OpenAI's images endpoint, with our fields; an error carries the server's message. */
  async generate(body: Record<string, unknown>): Promise<GenAnswer> {
    const r = await fetch("/v1/images/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as GenAnswer & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
};
