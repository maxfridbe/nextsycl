/** Everything to do with making a new clip. Deliberately separate from Jobs. */
import { jsx } from "../jsx.js";
import { rpc } from "../api.js";
import { render, state } from "../state.js";
import { navigate } from "../router.js";
import type { Engine, GenerateRequest } from "../types.js";
import { CanvasPicker } from "./CanvasPicker.js";
import { Panel } from "./Panel.js";

interface Form {
  prompt: string;
  seconds: number;
  steps: number;
  seed: number;
  te: "teacher" | "student";
  label: string;
  width: number;
  height: number;
  engine: Engine;
  upscale: number;
  /** "latent" (default), or an ESRGAN-type network on the decoded frames */
  upscaler: string;
  chain: boolean;
}

export const form: Form = {
  prompt:
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames " +
    "a middle-aged baker with a calm, slightly raspy voice (S1) opening the shutters of a small " +
    "street bakery at dawn, and he says: <d>[English] Morning. The bread is still warm.</d>" +
    "\n\noverall_soundscape: Wooden shutters scrape open over a quiet street, trays clink inside." +
    "\n\nnon_diegetic_music: None.",
  seconds: 10,
  steps: 8,
  seed: 0,
  te: "teacher",
  label: "",
  width: 768,
  height: 576,
  engine: "INT8",
  upscale: 1.5,
  upscaler: "esrgan-general",
  chain: false,
};

/** Fill the form from an existing job - the "use these settings" path. */
export function useSettings(j: GenerateRequest): void {
  if (j.prompt && j.prompt !== "(recovered)") form.prompt = j.prompt;
  if (j.seconds) form.seconds = j.seconds;
  if (j.steps) form.steps = j.steps;
  if (j.seed != null) form.seed = j.seed;
  if (j.te) form.te = j.te;
  if (j.width) form.width = j.width;
  if (j.height) form.height = j.height;
  if (j.engine) form.engine = j.engine === "Q8_0" ? "INT8" : j.engine;
  if (j.upscale) form.upscale = j.upscale;
  form.upscaler = j.upscaler ?? "esrgan-general";
  form.label = j.label ?? "";
  form.chain = !!j.first_frame;
  navigate({ tab: "create" });
}

async function generate(queue: boolean): Promise<void> {
  const body: GenerateRequest = {
    prompt: form.prompt,
    seconds: form.seconds,
    steps: form.steps,
    seed: form.seed,
    te: form.te,
    label: form.label,
    width: form.width,
    height: form.height,
    engine: form.engine,
    upscale: form.upscale,
    upscaler: form.upscaler,
    chain_mode: "png",
    queue,
  };
  if (form.chain) body.first_frame = "prev";
  try {
    await rpc("generate", body);
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  render();
}

const num = (
  label: string, key: "seconds" | "steps" | "seed" | "upscale",
  min: number, max: number, step: number,
) => (
  <label>
    {label}
    <input
      attrs={{ type: "number", min, max, step, value: String(form[key]) }}
      on={{ input: (e: Event) => { form[key] = Number((e.target as HTMLInputElement).value); render(); } }}
    />
  </label>
);

export function Creation() {
  return (
    <div class="create">
      <textarea
        attrs={{ spellcheck: false }}
        on={{ input: (e: Event) => { form.prompt = (e.target as HTMLTextAreaElement).value; } }}
      >{form.prompt}</textarea>
      <div class="hint">
        Dialogue: <code>{"(S1) says: <d>[English] line</d>"}</code> · no negations (cfg is 1.0,
        so every word renders) · 4–15 s per clip
      </div>
      <div class="row">
        {num("seconds", "seconds", 1, 15, 0.01)}
        {num("steps", "steps", 1, 40, 1)}
        {num("seed", "seed", 0, 2147483647, 1)}
        {num("upscale", "upscale", 1, 4, 0.5)}
        <label attrs={{ title: "ESRGAN: decoded at the sampled size, the frames enlarged by a network on the GPU - about half the decode stage's time (a 5 s clip at 1152x864 on a B65: 19.6 s against 41.9). latent: the latents enlarged, then decoded at the larger size - the decoder's own fine texture." }}>
          upscaler
          <select on={{ change: (e: Event) => { form.upscaler = (e.target as HTMLSelectElement).value; render(); } }}>
            <option attrs={{ value: "esrgan-general", selected: form.upscaler === "esrgan-general" }}>ESRGAN general-x4v3 (default)</option>
            <option attrs={{ value: "esrgan-anime", selected: form.upscaler === "esrgan-anime" }}>ESRGAN animevideov3 (sharper, drawn look)</option>
            <option attrs={{ value: "latent", selected: form.upscaler === "latent" }}>latent (slowest, most natural texture)</option>
          </select>
        </label>
        <label>
          encoder
          <select on={{ change: (e: Event) => { form.te = (e.target as HTMLSelectElement).value as "teacher" | "student"; render(); } }}>
            <option attrs={{ value: "teacher", selected: form.te === "teacher" }}>teacher 32B</option>
            <option attrs={{ value: "student", selected: form.te === "student" }}>student 4B</option>
          </select>
        </label>
        <label>
          label
          <input
            attrs={{ type: "text", placeholder: "Name 01/10: …", value: form.label }}
            on={{ input: (e: Event) => { form.label = (e.target as HTMLInputElement).value; } }}
          />
        </label>
        <label class="inline">
          <input
            attrs={{ type: "checkbox", checked: form.chain }}
            on={{ change: (e: Event) => { form.chain = (e.target as HTMLInputElement).checked; render(); } }}
          />
          chain from previous clip
        </label>
      </div>
      <Panel
        id="canvas"
        icon="&#xf0b2;"
        title="Canvas and length"
        hint={`${form.width}×${form.height} · ${form.seconds}s · ${form.steps} steps`}
      >
        <CanvasPicker
          width={form.width}
          height={form.height}
          engine={form.engine}
          seconds={form.seconds}
          steps={form.steps}
          onPick={(w, h, e, sec) => {
            form.width = w;
            form.height = h;
            form.engine = e as Engine;
            form.seconds = sec;
            render();
          }}
        />
      </Panel>
      <div class="row">
        <button type="button" on={{ click: () => void generate(false) }}>
          <span class="i" props={{ innerHTML: "&#xf0d0;" }} />Generate
        </button>
        <button type="button" class="tbtn" on={{ click: () => void generate(true) }}>
          <span class="i" props={{ innerHTML: "&#xf03a;" }} />Queue
        </button>
      </div>
      {state.error ? <pre class="err">{state.error}</pre> : null}
    </div>
  );
}
