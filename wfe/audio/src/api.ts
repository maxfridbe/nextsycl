/** The only module that talks to the server. */
import type { EnrollJob, EnrollState, GenAnswer, Gpu, Info, Progress, SavedVoice, Song } from "./types.js";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  info: () => getJSON<Info>("api/info"),
  progress: () => getJSON<Progress>("api/progress"),
  gpu: () => getJSON<Gpu>("api/gpu"),
  history: async () => (await getJSON<{ data: Song[] }>("api/history")).data.map((s) => ({ ...s, url: here(s.url) })),
  cancel: async () => { await fetch("api/cancel", { method: "POST" }); },
  voices: async () => (await getJSON<{ data: SavedVoice[] }>("api/voices")).data,

  /** Save a voice: a recording (sample, a data: URL) or a file made here; an error carries the server's message. */
  async addVoice(body: Record<string, unknown>): Promise<SavedVoice> {
    const r = await fetch("api/voices/add", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const j = (await r.json()) as SavedVoice & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
  /** The front door's enrolments (through /open/<id>/ only: none when the page is the server's own) */
  async enrollState(): Promise<EnrollState | null> {
    try {
      const r = await fetch("api/voices/enroll");
      return r.ok ? ((await r.json()) as EnrollState) : null;
    } catch {
      return null;
    }
  },
  /** Enroll a saved voice (a front-door job that waits for a card); an error carries the door's message. */
  async enroll(name: string, model?: string): Promise<EnrollJob> {
    const r = await fetch("api/voices/enroll", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, model }) });
    const j = (await r.json()) as EnrollJob & { error?: { message: string } | string };
    if (!r.ok) throw new Error(typeof j.error === "string" ? j.error : j.error?.message ?? `HTTP ${r.status}`);
    return j;
  },
  async removeVoice(name: string): Promise<void> {
    const r = await fetch("api/voices/remove", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) });
    if (!r.ok) throw new Error(((await r.json()) as { error?: { message: string } }).error?.message ?? `HTTP ${r.status}`);
  },

  /** The speech endpoint (the reference server's), its answer as JSON with the song's link; an error carries the
   *  server's message. */
  async generate(body: Record<string, unknown>): Promise<GenAnswer> {
    const r = await fetch("v1/audio/speech", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await r.json()) as GenAnswer & { error?: { message: string } };
    if (!r.ok) throw new Error(j.error?.message ?? `HTTP ${r.status}`);
    return { ...j, data: j.data.map((d) => (d.url ? { ...d, url: here(d.url) } : d)) };
  },
};

/** Plays 16-bit PCM pieces as they come, one after the other (Web Audio) */
export class Player {
  private ctx: AudioContext;
  private at = 0;
  constructor(private rate: number) {
    this.ctx = new AudioContext({ sampleRate: rate });
  }
  /** a piece of base64 16-bit PCM, queued after the ones before */
  push(b64: string): void {
    const bin = atob(b64);
    const n = bin.length >> 1;
    const buf = this.ctx.createBuffer(1, n, this.rate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) {
      const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      ch[i] = (v >= 0x8000 ? v - 0x10000 : v) / 32768;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    // a little ahead of now when it ran dry (or at the start)
    this.at = Math.max(this.at, this.ctx.currentTime + 0.05);
    src.start(this.at);
    this.at += buf.duration;
  }
  stop(): void {
    void this.ctx.close();
  }
}

/** The speech endpoint as server-sent events: each piece of sound to `piece` as it comes, then the answer (the
 *  file's link and the numbers); an error carries the server's message. */
export async function generateStreamed(body: Record<string, unknown>, piece: (b64: string) => void): Promise<GenAnswer> {
  const r = await fetch("v1/audio/speech", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, response_format: "wav", stream_format: "sse" }),
  });
  if (!r.ok || !r.body) {
    const j = (await r.json().catch(() => ({}))) as { error?: { message: string } };
    throw new Error(j.error?.message ?? `HTTP ${r.status}`);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (value) text += dec.decode(value, { stream: true });
    let i;
    while ((i = text.indexOf("\n\n")) >= 0) {
      const ev = text.slice(0, i);
      text = text.slice(i + 2);
      if (!ev.startsWith("data: ")) continue;
      const j = JSON.parse(ev.slice(6)) as { type: string; audio?: string; error?: { message: string } } & Partial<GenAnswer>;
      if (j.type === "speech.audio.delta" && j.audio) piece(j.audio);
      else if (j.type === "error") throw new Error(j.error?.message ?? "failed");
      else if (j.type === "speech.audio.done") {
        return { created: j.created ?? 0, data: (j.data ?? []).map((d) => (d.url ? { ...d, url: here(d.url) } : d)),
                 nextsycl: j.nextsycl ?? { took: 0, seconds: 0, seed: 0, steps: 0, cfg: 0, wh: null } };
      }
    }
    if (done) throw new Error("the answer ended early");
  }
}

/** A link the server gave, made relative to the page (which may be served under the front door's /open/<id>/, where
 *  the server's own absolute links would name the door's token-protected /v1/): a data: or blob: URL as it is, an
 *  http(s) one by its path, and no leading "/" */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.replace(/^\//, "");
}

