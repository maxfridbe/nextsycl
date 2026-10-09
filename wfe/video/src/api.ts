/** The only module that talks to the server. Every call is typed by Methods. */
import type {
  Envelope, Film, Gpu, Job, JobDetail, ListRow, Methods, Plan, Scene, Status, Summary, TimelineClip,
} from "./types.js";

const base = "";

export async function rpc<M extends keyof Methods>(
  method: M,
  params: Methods[M][0],
): Promise<Methods[M][1]> {
  const r = await fetch(`${base}/rpc/${String(method)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  const e = (await r.json()) as Envelope<Methods[M][1]>;
  if (!e.ok) throw new Error(`${e.error.code}: ${e.error.message}`);
  return e.result;
}

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  status: () => getJSON<Status>("/api/status"),
  gpu: () => getJSON<Gpu>("/api/gpu"),
  summary: () => getJSON<Summary>("/api/summary"),
  list: () => getJSON<ListRow[]>("/api/list"),
  films: () => getJSON<Film[]>("/api/films"),
  timeline: (project: string) =>
    getJSON<{ clips: TimelineClip[] }>(`/api/timeline?project=${encodeURIComponent(project)}`),
  job: (name: string) => getJSON<JobDetail>(`/api/job/${encodeURIComponent(name)}`),
  clip: (label: string) => getJSON<Job>(`/api/clip?label=${encodeURIComponent(label)}`),
  plan: () => getJSON<Plan>("/api/plan"),

  async post<T>(path: string, body: unknown): Promise<T> {
    const r = await fetch(base + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    return (await r.json()) as T;
  },

  hold: (scope: { project?: string; batch?: string; on: boolean }) =>
    api.post<{ ok: boolean }>("/api/queue/pause", scope),
  clearQueue: (scope: { project?: string; batch?: string } = {}) =>
    rpc("queue.clear", scope),
  importScene: (scene: Scene, project: string, paused: boolean) =>
    api.post<{ imported?: number; project?: string; error?: string }>(
      "/api/scene/import", { scene, project, paused }),
};
