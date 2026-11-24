/** The only module that talks to the server: the front door's generated client (rule 7, wfe/home/src/gen) over the
 *  studio that serves this page. A front-door method `video.X` is the studio's `X` (the door strips the kind the same
 *  way, home/api.rs studio_rpc); video.roll is the studio's `roll`, which starts the film as the door's handler does.
 *  Reached through the front door (its /open/<model>/roll), the relative paths go through the door to the same studio. */
import { client } from "../../home/src/gen/nextsycl.js";

async function post(method: string, body: unknown): Promise<any> {
  const m = method.startsWith("video.") ? method.slice("video.".length) : method;
  const r = await fetch(`rpc/${m}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = (await r.json().catch(() => ({}))) as { ok?: boolean; result?: unknown; error?: { code?: string; message?: string } | string };
  if (j.ok === false || !r.ok) {
    const e = typeof j.error === "string" ? { message: j.error } : j.error ?? {};
    throw new Error(`${e.code ? e.code + ": " : ""}${e.message ?? `HTTP ${r.status}`}`);
  }
  return j.result;
}

export const api = client(post);

/** A file the studio serves (/out/...) as this page reaches it: relative, so the door's /open/ prefix stays */
export function fileUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return path.startsWith("/out/") ? path.slice(1) : path;
}
