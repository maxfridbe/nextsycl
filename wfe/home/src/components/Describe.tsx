/** The Describe tab: a picture to the prompt that would draw it (image.describe through the generated client), for the
 *  image model picked as the target; the prompt, its negative and the suggested size to copy. */
import { jsx } from "../jsx.js";
import { client } from "../gen/nextsycl.js";
import type { ImageDescribeReq } from "../gen/nextsycl.js";
import { copy, loadSpec, render, ui } from "../main.js";

const TARGETS = ["natural", "z-image", "qwen-image", "h3", "hunyuanimage", "moss-image", "ming-image", "tags"];

export const describe = {
  picture: null as string | null,
  name: "",
  target: "natural",
  detail: "medium" as "short" | "medium" | "long",
  negative: false,
  busy: false,
  result: null as Record<string, any> | null,
  error: null as string | null,
};

/** The front door's /rpc with the page's token (the spec carries it) */
async function post(method: string, body: unknown): Promise<any> {
  if (!ui.spec) await loadSpec();
  const r = await fetch(`/rpc/${method}`, { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${ui.spec?.token ?? ""}` }, body: JSON.stringify(body) });
  const j = (await r.json()) as { ok: boolean; result?: unknown; error?: { code: string; message: string } };
  if (!j.ok) throw new Error(`${j.error?.code}: ${j.error?.message}`);
  return j.result;
}

function pick(ev: Event): void {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { describe.picture = String(rd.result); describe.name = f.name; describe.result = null; render(); };
  rd.readAsDataURL(f);
}

async function run(): Promise<void> {
  if (!describe.picture) return;
  describe.busy = true;
  describe.error = null;
  render();
  try {
    const req: ImageDescribeReq = { image: describe.picture, target: describe.target, detail: describe.detail, include_negative: describe.negative,
                                    label: `page-describe-${Date.now()}` };
    describe.result = (await client(post)["image.describe"](req)) as Record<string, any>;
  } catch (e) {
    describe.error = String(e);
  }
  describe.busy = false;
  render();
}

export function Describe() {
  const r = describe.result;
  const t = r?.timings ?? {};
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf03e;" }} /><span class="ptitle">Describe a picture</span>
        <span class="hint">the prompt that would draw it, written for the image model you pick (image.describe)</span></div>
      <div class="pbody">
        <div class="row">
          <input attrs={{ type: "file", accept: "image/png,image/jpeg,image/webp" }} on={{ change: pick }} />
          <select on={{ change: (e: Event) => { describe.target = (e.target as HTMLSelectElement).value; } }}>
            {TARGETS.map((x) => <option attrs={{ value: x, selected: x === describe.target }}>{x}</option>)}
          </select>
          <select on={{ change: (e: Event) => { describe.detail = (e.target as HTMLSelectElement).value as "short" | "medium" | "long"; } }}>
            {["short", "medium", "long"].map((x) => <option attrs={{ value: x, selected: x === describe.detail }}>{x}</option>)}
          </select>
          <label><input attrs={{ type: "checkbox", checked: describe.negative }} on={{ change: (e: Event) => { describe.negative = (e.target as HTMLInputElement).checked; } }} /> negative</label>
          <button class="btn" attrs={{ disabled: !describe.picture || describe.busy }} on={{ click: () => void run() }}>{describe.busy ? "describing…" : "Describe"}</button>
        </div>
        {describe.picture ? <img attrs={{ src: describe.picture, alt: describe.name }} style={{ maxWidth: "320px", maxHeight: "240px", marginTop: "8px", borderRadius: "6px" }} /> : null}
        {describe.error ? <pre class="err">{describe.error}</pre> : null}
        {r
          ? <div>
              <div class="curl"><pre>{r.prompt}</pre>
                <button class="ghost small" on={{ click: () => void copy(String(r.prompt), "describe-prompt") }}>{ui.copied === "describe-prompt" ? "copied" : "copy"}</button></div>
              {r.negative_prompt ? <div class="curl"><pre>negative: {r.negative_prompt}</pre></div> : null}
              <div class="hint">{r.target} · {r.model} · {r.words} words · size {r.suggested_size?.size} · pp {t.pp_tok_s ?? "?"} tok/s · tg {t.tg_tok_s ?? "?"} tok/s · {t.seconds} s
                {(r.notes ?? []).length ? ` · ${(r.notes as string[]).join("; ")}` : ""}</div>
            </div>
          : null}
      </div>
    </section>
  );
}
