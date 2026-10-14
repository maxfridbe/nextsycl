/** The pages (dropped, chosen or pasted; ordered, removed) and the instruction: a preset or one's own, tiles, the
 *  longest reading, temperature, the repetition guard. */
import { jsx } from "../jsx.js";
import { addFiles, movePage, read, removePage } from "../actions.js";
import { PRESETS, render, saveForm, state } from "../state.js";
import type { Form } from "../types.js";
import { clamp } from "../util.js";

function set<K extends keyof Form>(k: K, v: Form[K]): void {
  state.form[k] = v;
  saveForm();
  render();
}

/** A file input inside a label: a tap on the label opens the phone's picker (or camera) natively - no scripted
 *  click, which mobile browsers may refuse. The input itself is kept out of sight, not out of the page. */
function Picker(props: { camera?: boolean }) {
  return (
    <input
      class="pick"
      attrs={{ type: "file", accept: "image/png,image/jpeg,image/webp", multiple: !props.camera, ...(props.camera ? { capture: "environment" } : {}) }}
      on={{
        change: (e: Event) => {
          const t = e.target as HTMLInputElement;
          if (t.files) void addFiles(t.files);
          t.value = "";
        },
      }}
    />
  );
}

function Drop() {
  return (
    <div>
      <label
        class={{ drop: true, over: state.dragging }}
        on={{
          dragover: (e: DragEvent) => {
            e.preventDefault();
            if (!state.dragging) { state.dragging = true; render(); }
          },
          dragleave: () => { state.dragging = false; render(); },
          drop: (e: DragEvent) => {
            e.preventDefault();
            state.dragging = false;
            if (e.dataTransfer?.files.length) void addFiles(e.dataTransfer.files);
            render();
          },
        }}
      >
        <Picker />
        <span class="i" props={{ innerHTML: "&#xf093;" }} /> Tap or click to choose page pictures - or drop or paste them (PNG, JPEG,
        WebP); several pages are read as one document, in this order
      </label>
      <label class="tbtn camera"><Picker camera={true} /><span class="i" props={{ innerHTML: "&#xf030;" }} /> Take a photo of a page</label>
    </div>
  );
}

function Pages() {
  if (!state.pages.length) return null;
  return (
    <div class="pages">
      {state.pages.map((p, i) => (
        <figure class="page">
          <img attrs={{ src: p.url, alt: p.name }} />
          <figcaption>
            <span class="cap" attrs={{ title: `${p.name} · ${p.width}×${p.height}` }}>{i + 1}. {p.name}</span>
            <span class="acts">
              <button type="button" class="tbtn" attrs={{ title: "earlier", disabled: i === 0 }} on={{ click: () => movePage(i, -1) }}>◀</button>
              <button type="button" class="tbtn" attrs={{ title: "later", disabled: i === state.pages.length - 1 }} on={{ click: () => movePage(i, 1) }}>▶</button>
              <button type="button" class="tbtn" attrs={{ title: "remove" }} on={{ click: () => removePage(i) }}>✕</button>
            </span>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

export function Read() {
  const f = state.form;
  const one = state.pages.length <= 1;
  const ready = !!state.info && !state.info.loading && state.info.reads !== false;
  return (
    <div class="read">
      <Drop />
      <Pages />
      <div class="row">
        <div class="chips">
          {PRESETS.map(([id, , title]) => (
            <button type="button" class={{ chip: true, on: f.preset === id }} on={{ click: () => set("preset", id) }}>{title}</button>
          ))}
        </div>
      </div>
      {state.pages.length > 1
        ? <div class="hint">{`${state.pages.length} pages are read with the model's own multi-page instruction ("Multi page parsing."), each page in turn - the presets are for a single page.`}</div>
        : null}
      {f.preset === "custom"
        ? <textarea attrs={{ placeholder: "Your instruction (\"<|grounding|>\" first for boxes)" }} props={{ value: f.prompt }}
                    on={{ input: (e: Event) => set("prompt", (e.target as HTMLTextAreaElement).value) }} />
        : null}
      <div class="row">
        <label class="inline" attrs={{ title: "A single page: read its 640-pixel tiles too (more detail, more tokens). Several pages are read whole." }}>
          <input attrs={{ type: "checkbox", disabled: !one }} props={{ checked: f.tiles && one }} on={{ change: () => set("tiles", !f.tiles) }} /> tiles
        </label>
        <label class="inline" attrs={{ title: "The model marks pictures inside a page (screenshots, figures) without reading them: read each again on its own and put its text in" }}>
          <input attrs={{ type: "checkbox" }} props={{ checked: f.figures !== false }} on={{ change: () => set("figures", f.figures === false) }} /> read inside pictures
        </label>
        <label class="inline" attrs={{ title: "The reference's rule: no 35 tokens repeated (a looping reading stops)" }}>
          <input attrs={{ type: "checkbox" }} props={{ checked: f.guard }} on={{ change: () => set("guard", !f.guard) }} /> repetition guard
        </label>
        <label>max tokens
          <input attrs={{ type: "number", min: 64, max: 32768, step: 64 }} props={{ value: String(f.maxTokens) }}
                 on={{ change: (e: Event) => set("maxTokens", clamp(Number((e.target as HTMLInputElement).value) || 8192, 64, 32768)) }} />
        </label>
        <label>temperature
          <input attrs={{ type: "number", min: 0, max: 2, step: 0.1 }} props={{ value: String(f.temperature) }}
                 on={{ change: (e: Event) => set("temperature", clamp(Number((e.target as HTMLInputElement).value) || 0, 0, 2)) }} />
        </label>
        <button type="button" attrs={{ id: "go", disabled: !ready || !state.pages.length || state.busy }} on={{ click: () => void read() }}>
          <span class="i" props={{ innerHTML: state.busy ? "&#xf110;" : "&#xf02d;" }} />
          {state.busy ? "Reading…" : `Read ${state.pages.length > 1 ? `${state.pages.length} pages` : "the page"}`}
        </button>
      </div>
    </div>
  );
}
