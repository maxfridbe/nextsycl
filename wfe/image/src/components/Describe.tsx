/** Describe: a picture dropped, picked or pasted (or one made here) to the prompt that would draw it. The request is the
 *  front door's image.describe as a job (api/describe), for a prompt style (the model this page serves, or a family)
 *  and a describer; the answer is an editable text with copy and "use as prompt". */
import { jsx } from "../jsx.js";
import { cancelDescribe, describePicture, runDescribe, useDescription } from "../actions.js";
import { FAMILIES, jobStage, takesNegative, targetOf } from "../describe.js";
import { render, saveDescribe, state } from "../state.js";
import type { DescribeForm } from "../types.js";
import { fmtT } from "../util.js";
import { Spinner } from "./Spinner.js";

/** the Copy button says so for a moment */
let copied = false;

async function copy(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
    render();
    setTimeout(() => { copied = false; render(); }, 1500);
  } catch {
    /* no clipboard (an http page off localhost): the text is selectable */
  }
}

function set<K extends keyof DescribeForm>(k: K, v: DescribeForm[K]): void {
  state.describe.form[k] = v;
  saveDescribe();
  render();
}

export function Describe() {
  const d = state.describe;
  const f = d.form;
  const info = state.info;
  if (d.list === null) return <Spinner label="Asking the front door for its describers…" />;
  if (d.list === "no") {
    return (
      <div class="hint">
        Describe needs the front door: a chat model that reads pictures lives there, not in this server. Open this page through it
        (http://&lt;box&gt;:8000/open/&lt;model&gt;/), or run nextsycl image describe picture.png --target &lt;family&gt;.
      </div>
    );
  }
  const list = d.list.describers;
  const target = targetOf(f.target, info);
  const running = !!d.label;
  const elapsed = d.started !== null ? (performance.now() - d.started) / 1000 : 0;
  const add = (files: FileList | null | undefined) => {
    const g = Array.from(files ?? []).find((x) => x.type.startsWith("image/"));
    if (g) void describePicture(g, g.name);
  };
  const r = d.result;
  const select = (value: string, options: [string, string][], change: (v: string) => void) => (
    <select on={{ change: (e: Event) => change((e.target as HTMLSelectElement).value) }}>
      {options.map(([v, label]) => <option attrs={{ value: v }} props={{ selected: v === value }}>{label}</option>)}
    </select>
  );
  return (
    <div class="create describe">
      <div class="drop" on={{ dragover: (e: DragEvent) => e.preventDefault(), drop: (e: DragEvent) => { e.preventDefault(); add(e.dataTransfer?.files); } }}>
        <div class="row">
          {d.picture ? <img class="dpic" attrs={{ src: d.picture.url, alt: d.picture.name }} /> : null}
          <label class="inline">
            <input attrs={{ type: "file", accept: "image/png,image/jpeg,image/webp", disabled: running }}
                   on={{ change: (e: Event) => { add((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ""; } }} />
          </label>
          {d.picture ? <span class="hint">{d.picture.name} · {d.picture.w}×{d.picture.h}</span> : null}
        </div>
        <div class="hint">Drop a picture here, pick a file or paste one (Ctrl+V anywhere on the page); "describe" on a picture below does the same.</div>
      </div>
      <div class="row">
        <label attrs={{ title: "the model the prompt is for: it decides how the prompt is written" }}>prompt for
          {select(f.target, [["", `${info?.model ?? "this page's model"} (this page)`], ...FAMILIES.map((x): [string, string] => [x.id, `${x.id}: ${x.about}`])],
                  (v) => set("target", v))}
        </label>
        <label attrs={{ title: "who reads the picture: best is a chat model that reads pictures; native is the target model's own text encoder" }}>describer
          {select(f.describer, [["best", `best${d.list.best ? ` (${d.list.best})` : " (none enabled)"}`], ["native", "native (the model's own text encoder)"],
                                 ...list.map((x): [string, string] => [x.id, `${x.id} (${x.kind}${x.enabled ? "" : x.downloaded ? ", disabled" : ", not registered"})`])],
                  (v) => set("describer", v))}
        </label>
        <label>length
          {select(f.detail, [["short", "short (~35 words)"], ["medium", "medium (~90)"], ["long", "long (~170)"]], (v) => set("detail", v as DescribeForm["detail"]))}
        </label>
        <label attrs={{ title: "the words aimed for, in place of the length's" }}>words
          <input attrs={{ type: "number", min: 8, max: 400, placeholder: "by length" }} props={{ value: f.maxWords === null ? "" : String(f.maxWords) }}
                 on={{ change: (e: Event) => {
                   const t = (e.target as HTMLInputElement).value.trim();
                   const x = Number(t);
                   set("maxWords", t === "" || !Number.isFinite(x) ? null : Math.max(8, Math.min(400, Math.round(x))));
                 } }} />
        </label>
        <label>language
          <input attrs={{ type: "text", placeholder: "English" }} props={{ value: f.language }} on={{ change: (e: Event) => set("language", (e.target as HTMLInputElement).value) }} />
        </label>
        {takesNegative(target, info)
          ? <label class="inline"><input attrs={{ type: "checkbox" }} props={{ checked: f.negative }} on={{ change: () => set("negative", !f.negative) }} /> also a negative prompt</label>
          : null}
      </div>
      <div class="row">
        <label class="wide">more for the describer
          <input attrs={{ type: "text", placeholder: "focus on the lighting; the person is a child ...", maxlength: 2000 }} props={{ value: f.instructions }}
                 on={{ change: (e: Event) => set("instructions", (e.target as HTMLInputElement).value) }} />
        </label>
      </div>
      <div class="row">
        <button type="button" attrs={{ disabled: !d.picture || running }} on={{ click: () => void runDescribe() }}>
          <span class="i" props={{ innerHTML: "&#xf002;" }} />{running ? "Describing…" : "Describe"}
        </button>
        {running ? <button type="button" class="tbtn" attrs={{ title: "stop this job (by its label)" }} on={{ click: () => void cancelDescribe() }}>cancel</button> : null}
        <span class="hint">The first describe after a while loads the describer: about 45 s cold, a few seconds after.</span>
      </div>
      {running
        ? <div class="clipprog">
            <div class="bar"><div class="fill" style={{ width: d.job?.state === "running" ? "60%" : "10%" }} /></div>
            <div class="stage"><span class="i" props={{ innerHTML: "&#xf110;" }} /> {jobStage(d.job, elapsed)} <span class="hint">{fmtT(elapsed)} elapsed</span></div>
          </div>
        : null}
      {d.error ? <pre class="err">{d.error}</pre> : null}
      {r
        ? <div>
            <label class="wide">prompt (edit it before using it)
              <textarea props={{ value: d.prompt }} on={{ input: (e: Event) => { d.prompt = (e.target as HTMLTextAreaElement).value; d.used = null; } }} />
            </label>
            {r.negative_prompt !== null || d.negative
              ? <label class="wide">negative prompt
                  <textarea class="short" props={{ value: d.negative }} on={{ input: (e: Event) => { d.negative = (e.target as HTMLTextAreaElement).value; d.used = null; } }} />
                </label>
              : null}
            <div class="row">
              <button type="button" attrs={{ disabled: !d.prompt.trim() }} on={{ click: useDescription }}>Use as prompt</button>
              <button type="button" class="tbtn" on={{ click: () => void copy(d.negative.trim() ? `${d.prompt}\n\nNegative: ${d.negative}` : d.prompt) }}>{copied ? "copied" : "copy"}</button>
              {d.used ? <span class="hint">{d.used}</span> : null}
            </div>
            <div class="hint">
              {`${r.target}${r.target_model ? ` (${r.target_model})` : ""} · read by ${r.model} · ${r.words} words · picture ${r.width}×${r.height}` +
                (r.suggested_size ? ` · suggested size ${r.suggested_size.size}` : "") +
                ` · pp ${r.timings.pp_tok_s ?? "?"} tok/s · tg ${r.timings.tg_tok_s ?? "?"} tok/s · ${fmtT(r.timings.seconds)}`}
            </div>
            {r.finish === "length" ? <div class="hint warn">cut short: the word limit ended the answer (raise "words")</div> : null}
            {(r.notes ?? []).map((n) => <div class="hint">{n}</div>)}
          </div>
        : null}
    </div>
  );
}
