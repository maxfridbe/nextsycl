/** The request, every option the engine has: the description, the lyrics (structure tags a click away), the length,
 *  the seed, and under Advanced the flow stage's steps and guidance and the engine's own options. Each field writes
 *  straight into state.form. */
import { jsx } from "../jsx.js";
import { addTag, cancel, generate, randomSeed } from "../actions.js";
import { panelOpen, render, saveForm, state, togglePanel } from "../state.js";
import type { Form } from "../types.js";
import { LENGTHS, TAGS, clamp, mss } from "../util.js";

type Key = keyof Form;

function set<K extends Key>(k: K, v: Form[K]): void {
  state.form[k] = v;
  saveForm();
  render();
}

const num = (k: Key, opts: { min?: number; max?: number; step?: number; placeholder?: string; nullable?: boolean }) => {
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
          set(k, x as Form[typeof k]);
        },
      }}
    />
  );
};

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

const area = (k: "prompt" | "lyrics", cls: string, placeholder: string) => (
  <textarea
    class={cls}
    attrs={{ placeholder }}
    props={{ value: state.form[k] }}
    on={{
      input: (e: Event) => { state.form[k] = (e.target as HTMLTextAreaElement).value; saveForm(); render(); },
      keydown: (e: KeyboardEvent) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void generate(); },
    }}
  />
);

export function Create() {
  const f = state.form;
  const info = state.info;
  const d = info?.defaults;
  const max = d?.max_seconds ?? 360;
  const ready = !!d;
  return (
    <div class="create">
      <div class="lab">description <span class="hint">genre, tempo and key, mood, the voice (say who sings), instruments, arrangement</span></div>
      {area("prompt", "", "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. Vocals: soft female lead. Arrangement: fingerpicked guitar, soft piano, brushed drums in the chorus.")}
      {info?.lyrics === false ? null : <div>
        <div class="lab">lyrics
          <span class="chips">{TAGS.map((t) => <button type="button" class="chip" on={{ click: () => addTag(t) }}>[{t}]</button>)}</span>
        </div>
        {area("lyrics", "lyrics", "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe")}
        <div class="hint">Tags on lines of their own; text after a tag on its line is dropped. Empty lyrics: an instrumental.</div>
      </div>}
      <div class="row">
        <label>length <span class="hint">at most; the song may end sooner</span>
          <span class="chips">
            {LENGTHS.filter(([, s]) => s <= max).map(([n, s]) => (
              <button type="button" class={{ chip: true, on: f.seconds === s }} on={{ click: () => set("seconds", s) }}>{n}</button>
            ))}
          </span>
        </label>
        <label>seconds {num("seconds", { min: 1, max, step: 1 })}</label>
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
      <Section id="advanced" title={`Advanced (${f.steps ?? d?.steps ?? "—"} steps · guidance ${f.cfg ?? d?.cfg ?? "—"})`}>
        <div class="row">
          <label>steps (each window) {num("steps", { min: 1, max: 100, placeholder: d ? String(d.steps) : "", nullable: true })}</label>
          <label>guidance (cfg) {num("cfg", { min: 1, max: 10, step: 0.1, placeholder: d ? String(d.cfg) : "", nullable: true })}</label>
        </div>
        {(info?.options ?? []).length
          ? <div class="row">
              {(info?.options ?? []).map((o) => (
                <label attrs={{ title: o.help }}>{o.name}
                  <input
                    attrs={{ type: "text", placeholder: o.value || "on" }}
                    props={{ value: f.options[o.name] ?? "" }}
                    on={{ change: (e: Event) => { state.form.options[o.name] = (e.target as HTMLInputElement).value; saveForm(); render(); } }}
                  />
                </label>
              ))}
            </div>
          : null}
        <div class="hint">{`The language model draws a frame 25 times a second of sound; the flow stage then makes ${mss(f.seconds)} in windows of 8 s, 4 s apart.`}</div>
      </Section>
      <div class="row">
        <button type="button" attrs={{ id: "go", disabled: !ready || state.busy || !f.prompt.trim() }} on={{ click: () => void generate() }}>
          <span class="i" props={{ innerHTML: "&#xf0d0;" }} />{state.busy ? "Generating…" : "Generate"}
        </button>
        {state.busy || state.progress?.busy
          ? <button type="button" attrs={{ id: "stop" }} on={{ click: () => void cancel() }}><span class="i" props={{ innerHTML: "&#xf04d;" }} />Cancel</button>
          : null}
      </div>
    </div>
  );
}
