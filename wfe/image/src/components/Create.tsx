/** The request, every option the engine has: prompt, negative, size (aspect presets), steps, pictures, seed, the
 *  sampler and schedule, guidance, shift, transparency, LoRAs. Each field writes straight into state.form. */
import { jsx } from "../jsx.js";
import { addPicture, generate, movePicture, randomSeed, removePicture, setAspect } from "../actions.js";
import { panelOpen, render, saveForm, state, togglePanel } from "../state.js";
import type { Form } from "../types.js";
import { ASPECTS, clamp, r16 } from "../util.js";

type Key = keyof Form;

/** A field bound to the form: its value from state, each input written back. */
function set<K extends Key>(k: K, v: Form[K]): void {
  state.form[k] = v;
  saveForm();
  render();
}

const num = (k: Key, opts: { min?: number; max?: number; step?: number; placeholder?: string; nullable?: boolean; snap16?: boolean }) => {
  const v = state.form[k] as number | null;
  return (
    <input
      attrs={{ type: "number", min: opts.min ?? "", max: opts.max ?? "", step: opts.step ?? 1, placeholder: opts.placeholder ?? "" }}
      props={{ value: v === null || v === undefined ? "" : String(v) }}
      on={{
        change: (e: Event) => {
          const t = (e.target as HTMLInputElement).value.trim();
          if (t === "" && opts.nullable) return set(k, null as Form[typeof k]);
          let x = Number(t);
          if (!Number.isFinite(x)) return render();
          if (opts.min !== undefined || opts.max !== undefined) x = clamp(x, opts.min ?? -Infinity, opts.max ?? Infinity);
          if (opts.snap16) x = r16(x);
          set(k, x as Form[typeof k]);
        },
      }}
    />
  );
};

function Aspects() {
  const f = state.form;
  return (
    <div class="chips">
      {ASPECTS.map(([name, a, b]) => (
        <button
          type="button"
          class={{ chip: true, on: Math.abs(f.width / f.height - a / b) < 0.02 }}
          on={{ click: () => setAspect(a, b) }}
        >{name}</button>
      ))}
    </div>
  );
}

function Loras() {
  const list = state.info?.loras ?? [];
  if (!list.length) return <div class="hint">No LoRAs for this model are registered (nextsycl models search --kind lora).</div>;
  return (
    <div>
      {list.map((l) => {
        const on = l.id in state.form.loras;
        const scale = state.form.loras[l.id] ?? 1;
        return (
          <div class="lora">
            <input
              attrs={{ type: "checkbox" }}
              props={{ checked: on }}
              on={{ change: () => {
                if (on) delete state.form.loras[l.id];
                else state.form.loras[l.id] = 1;
                // another set may bring its own steps (a few-step LoRA): the model's then
                state.form.steps = null;
                saveForm(); render();
              } }}
            />
            <span class="lt" attrs={{ title: l.title }}>{l.title}{l.loaded !== null ? <span class="hint"> · merged ×{l.loaded}</span> : null}</span>
            <input
              attrs={{ type: "number", step: 0.05, min: -4, max: 4, disabled: !on }}
              props={{ value: String(scale) }}
              on={{ change: (e: Event) => {
                const x = Number((e.target as HTMLInputElement).value);
                if (Number.isFinite(x)) state.form.loras[l.id] = x;
                saveForm(); render();
              } }}
            />
          </div>
        );
      })}
      <div class="hint">A change of LoRAs reloads the model with them merged (a few seconds); a few-step LoRA brings its own steps.</div>
    </div>
  );
}

/** An edit's pictures: added from files (or dropped here, or "edit" on a picture made here); the first is the one
 *  changed, the others are references the instructions can name ("the coat from picture 2"). */
function Pictures() {
  const list = state.pictures;
  const max = state.info?.max_pictures ?? 8;
  const add = (files: FileList | null | undefined) => {
    for (const f of Array.from(files ?? [])) void addPicture(f, f.name);
  };
  return (
    <div
      class="drop"
      on={{
        dragover: (e: DragEvent) => e.preventDefault(),
        drop: (e: DragEvent) => { e.preventDefault(); add(e.dataTransfer?.files); },
      }}
    >
      <div class="pics small">
        {list.map((p, i) => (
          <figure class="pic">
            <img attrs={{ src: p.url, alt: p.name }} />
            <figcaption>
              <span class="cap" attrs={{ title: p.name }}>{i === 0 ? "picture 1 (changed)" : `picture ${i + 1}`} · {p.w}×{p.h}</span>
              <span class="acts">
                <button type="button" class="tbtn" attrs={{ title: "earlier", disabled: i === 0 }} on={{ click: () => movePicture(i, -1) }}>↑</button>
                <button type="button" class="tbtn" attrs={{ title: "later", disabled: i === list.length - 1 }} on={{ click: () => movePicture(i, 1) }}>↓</button>
                <button type="button" class="tbtn" attrs={{ title: "remove" }} on={{ click: () => removePicture(i) }}>×</button>
              </span>
            </figcaption>
          </figure>
        ))}
      </div>
      <div class="row">
        <label class="inline">
          <input attrs={{ type: "file", accept: "image/png,image/jpeg,image/webp", multiple: true, disabled: list.length >= max }}
                 on={{ change: (e: Event) => { add((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ""; } }} />
        </label>
        {list.length
          ? <label class="inline">
              <input attrs={{ type: "checkbox" }} props={{ checked: state.sizeFromPictures }}
                     on={{ change: () => { state.sizeFromPictures = !state.sizeFromPictures; render(); } }} /> size from the pictures
            </label>
          : null}
      </div>
      <div class="hint">{`Drop pictures here or pick files (up to ${max}). With pictures the prompt becomes instructions: the first picture is changed, the others are references - name them ("put the hat from picture 2 on the fox in picture 1").`}</div>
    </div>
  );
}

/** A sub-section that opens and closes like a panel (its flag in state, not the DOM's). */
function Section(props: { id: string; title: string; children?: unknown }) {
  const open = panelOpen(props.id);
  return (
    <div class="sub">
      <div class="subhead" on={{ click: () => togglePanel(props.id) }}>
        <span class="chev">{open ? "▾" : "▸"}</span> {props.title}
      </div>
      {open ? <div class="subbody">{props.children as never}</div> : null}
    </div>
  );
}

export function Create() {
  const f = state.form;
  const info = state.info;
  const d = info?.defaults;
  const ready = !!info && !info.loading;
  const select = (k: "sampler" | "schedule", list: string[]) => (
    <select on={{ change: (e: Event) => set(k, (e.target as HTMLSelectElement).value) }}>
      {list.map((x) => <option attrs={{ value: x }} props={{ selected: x === f[k] }}>{x}</option>)}
    </select>
  );
  return (
    <div class="create">
      <textarea
        attrs={{ placeholder: "a red fox in fresh snow, morning light, photograph" }}
        props={{ value: f.prompt }}
        on={{
          input: (e: Event) => { state.form.prompt = (e.target as HTMLTextAreaElement).value; saveForm(); render(); },
          keydown: (e: KeyboardEvent) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void generate(); },
        }}
      />
      {info?.guidance === false ? null : <div class="row">
        <label class="wide">negative prompt
          <input attrs={{ type: "text", placeholder: d && d.cfg <= 1 ? "needs guidance (cfg above 1)" : "" }} props={{ value: f.negative }}
                 on={{ change: (e: Event) => set("negative", (e.target as HTMLInputElement).value) }} />
        </label>
      </div>}
      <div class="row"><label>size</label><Aspects /></div>
      <div class="row">
        <label>width {num("width", { min: 256, max: 4096, step: 16, snap16: true })}</label>
        <label>height {num("height", { min: 256, max: 4096, step: 16, snap16: true })}</label>
        <label>steps {num("steps", { min: 1, max: 200, placeholder: d ? String(d.steps) : "", nullable: true })}</label>
        <label>pictures {num("n", { min: 1, max: 8 })}</label>
        <label>seed
          <span class="seed">
            {num("seed", { min: 0, placeholder: "random", nullable: true })}
            <button type="button" class="tbtn" attrs={{ title: "a new random seed" }} on={{ click: randomSeed }}>
              <span class="i" props={{ innerHTML: "&#xf074;" }} />
            </button>
            {f.seed !== null
              ? <button type="button" class="tbtn" attrs={{ title: "random each time" }} on={{ click: () => set("seed", null) }}>clear</button>
              : null}
          </span>
        </label>
      </div>
      <Section id="sampling" title={`Sampling (${f.sampler || "—"} · ${f.schedule || "—"})`}>
        <div class="row">
          <label>sampler {select("sampler", info?.samplers ?? [])}</label>
          <label attrs={{ title: "karras, exponential and polyexponential crowd the steps at the clean end: flow models want shift, simple, beta or normal" }}>
            schedule {select("schedule", info?.schedules ?? [])}</label>
          {info?.guidance === false
            ? null
            : <label>guidance (cfg) {num("cfg", { min: 1, max: 20, step: 0.1, placeholder: d ? String(d.cfg) : "", nullable: true })}</label>}
          <label>shift {num("shift", { min: 0, max: 20, step: 0.1, placeholder: "model's", nullable: true })}</label>
          <label class="inline">
            <input attrs={{ type: "checkbox" }} props={{ checked: f.rgba }} on={{ change: () => set("rgba", !f.rgba) }} /> transparent (RGBA)
          </label>
        </div>
        {(info?.options ?? []).length
          ? <div class="row">
              {(info?.options ?? []).map((o) => (
                <label attrs={{ title: o.help }}>{o.name}
                  <input
                    attrs={{ type: "text", placeholder: o.value || "on" }}
                    props={{ value: f.options[o.name] ?? "" }}
                    on={{ change: (e: Event) => {
                      state.form.options[o.name] = (e.target as HTMLInputElement).value;
                      saveForm(); render();
                    } }}
                  />
                </label>
              ))}
            </div>
          : null}
      </Section>
      {info?.edits
        ? <Section id="pictures" title={`Pictures (${state.pictures.length ? `${state.pictures.length}: an edit` : "none: a new picture"})`}>
            <Pictures />
          </Section>
        : null}
      <Section id="loras" title={`LoRAs (${Object.keys(f.loras).length} on)`}>
        <Loras />
      </Section>
      <div class="row">
        <button type="button" attrs={{ id: "go", disabled: !ready || state.busy || !f.prompt.trim() }} on={{ click: () => void generate() }}>
          <span class="i" props={{ innerHTML: "&#xf0d0;" }} />{state.busy ? "Generating…" : state.pictures.length ? "Edit" : "Generate"}
        </button>
      </div>
    </div>
  );
}
