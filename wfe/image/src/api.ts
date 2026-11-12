/** The only module that talks to the server. */
import type { DescribeJob, Describers, GenAnswer, Gpu, Info, Picture, Progress } from "./types.js";

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

  /** The Describe panel's four calls: the front door answers them under /open/<id>/ (image.describers, image.describe as
   *  a job, image.status, image.cancel; a server run alone has no such route: the 404 says so). An error is thrown as
   *  the server's error object. */
  describers: () => rpc<Describers>("api/describe"),
  describe: (body: Record<string, unknown>) => rpc<DescribeJob>("api/describe", body),
  describeJob: (label: string) => rpc<DescribeJob>(`api/describe/status?label=${encodeURIComponent(label)}`),
  describeCancel: (label: string) => rpc<{ cancelled?: number }>("api/describe/cancel", { label }),
};

/** The error object a describe call carries (the /rpc envelope), thrown; `status` is the HTTP one (404: no such route). */
export class RpcError extends Error {
  constructor(public status: number, public code: string, message: string, public field: string | null = null) {
    super(message);
  }
}

/** A GET, or a POST with a body, answered as the front door's envelope {ok, result | error}. */
async function rpc<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const r = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = (await r.json().catch(() => ({}))) as { ok?: boolean; result?: T; error?: { code?: string; message?: string; field?: string | null } };
  if (!r.ok || j.ok === false) throw new RpcError(r.status, j.error?.code ?? "", j.error?.message ?? `${path}: HTTP ${r.status}`, j.error?.field ?? null);
  return j.result as T;
}

/** A link the server gave, made relative to the page (which may be served under the front door's /open/<id>/, where
 *  the server's own absolute links would name the door's token-protected /v1/): a data: or blob: URL as it is, an
 *  http(s) one by its path, and no leading "/" */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.replace(/^\//, "");
}

