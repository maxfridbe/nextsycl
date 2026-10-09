/** What the buttons do: build the request from the form, send it, keep the results. */
import { api } from "./api.js";
import { render, saveForm, state } from "./state.js";
import type { Picture } from "./types.js";
import { r16 } from "./util.js";

/** The first time the model is known: the form's empty fields take its defaults. */
export function adoptDefaults(): void {
  const d = state.info?.defaults;
  if (!d) return;
  const f = state.form;
  if (!f.sampler || !(state.info?.samplers ?? []).includes(f.sampler)) f.sampler = d.sampler;
  if (!f.schedule || !(state.info?.schedules ?? []).includes(f.schedule)) f.schedule = d.schedule;
  // LoRAs the server no longer offers drop out; one merged at start shows ticked
  const known = new Set((state.info?.loras ?? []).map((l) => l.id));
  for (const id of Object.keys(f.loras)) if (!known.has(id)) delete f.loras[id];
  if (!Object.keys(f.loras).length) {
    for (const l of state.info?.loras ?? []) if (l.loaded !== null) f.loras[l.id] = l.loaded;
  }
}

export function setAspect(a: number, b: number): void {
  const d = state.info?.defaults;
  const area = (d?.width ?? 1024) * (d?.height ?? 1024);
  const h = Math.sqrt((area * b) / a);
  state.form.width = r16((h * a) / b);
  state.form.height = r16(h);
  saveForm();
  render();
}

export function randomSeed(): void {
  state.form.seed = Math.floor(Math.random() * 1_000_000);
  saveForm();
  render();
}

/** Use a picture's prompt and seed again. */
export function reuse(p: Picture): void {
  state.form.prompt = p.prompt;
  state.form.seed = p.seed;
  saveForm();
  render();
}

export async function refreshHistory(): Promise<void> {
  try {
    state.history = await api.history();
    render();
  } catch {
    /* transient */
  }
}

function body(): Record<string, unknown> {
  const f = state.form;
  const d = state.info?.defaults;
  const b: Record<string, unknown> = {
    model: state.info?.model, prompt: f.prompt.trim(), size: `${f.width}x${f.height}`, n: f.n, response_format: "url",
  };
  if (f.steps) b.steps = f.steps;
  if (f.seed !== null) b.seed = f.seed;
  if (state.info?.guidance !== false) {
    if (f.negative.trim()) b.negative_prompt = f.negative.trim();
    if (f.cfg !== null && f.cfg !== d?.cfg) b.cfg = f.cfg;
  }
  if (f.shift !== null) b.shift = f.shift;
  if (f.sampler && f.sampler !== d?.sampler) b.sampler = f.sampler;
  if (f.schedule && f.schedule !== d?.schedule) b.schedule = f.schedule;
  if (f.rgba) b.background = "transparent";
  if ((state.info?.loras ?? []).length) b.loras = Object.entries(f.loras).map(([name, scale]) => ({ name, scale }));
  return b;
}

export async function generate(): Promise<void> {
  if (state.busy || !state.info || state.info.loading || !state.form.prompt.trim()) return;
  saveForm();
  state.busy = true;
  state.started = performance.now();
  state.error = null;
  state.status = "starting";
  render();
  try {
    const b = body();
    const j = await api.generate(b);
    const f = state.form;
    state.results = j.data.filter((d) => d.url).map((d) => ({
      url: d.url as string, prompt: d.revised_prompt, seed: d.seed, steps: j.nextsycl.steps, width: f.width, height: f.height,
      loras: j.nextsycl.loras, seconds: j.nextsycl.seconds, wh: j.nextsycl.wh, created: j.created,
    }));
    const wh = j.nextsycl.wh != null ? ` · ${j.nextsycl.wh.toFixed(2)} Wh` : "";
    state.status = `done in ${j.nextsycl.seconds.toFixed(1)} s${wh}`;
    await refreshHistory();
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
    state.status = "failed";
  }
  state.busy = false;
  state.started = null;
  render();
}
