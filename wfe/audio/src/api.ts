/** The only module that talks to the server. */
import type { GenAnswer, Gpu, Info, Progress, SavedVoice, Song } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("/api/info"),
  progress: () => getJSON<Progress>("/api/progress"),
  gpu: () => getJSON<Gpu>("/api/gpu"),
  history: async () => (await getJSON<{ data: Song[] }>("/api/history")).data,
  cancel: async () => { await fetch("/api/cancel", { method: "POST" }); },
  voices: async () => (await getJSON<{ data: SavedVoice[] }>("/api/voices")).data,

  /** Save a voice: a recording (sample, a data: URL) or a file made here; an error carries the server's message. */
  async addVoice(body: Record<string, unknown>): Promise<SavedVoice> {
    const r = await fetch("/api/voices/add", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const j = (await r.json()) as SavedVoice & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
  async removeVoice(name: string): Promise<void> {
    const r = await fetch("/api/voices/remove", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) });
    if (!r.ok) throw new Error(((await r.json()) as { error?: { message: string } }).error?.message ?? `HTTP ${r.status}`);
  },

  /** The speech endpoint (the reference server's), its answer as JSON with the song's link; an error carries the
   *  server's message. */
  async generate(body: Record<string, unknown>): Promise<GenAnswer> {
    const r = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as GenAnswer & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
};
