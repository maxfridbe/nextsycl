/** What the buttons do: build the request from the form, send it, keep the result. */
import { api } from "./api.js";
import { render, saveForm, state } from "./state.js";
import type { Song } from "./types.js";

export function randomSeed(): void {
  state.form.seed = Math.floor(Math.random() * 1_000_000);
  saveForm();
  render();
}

/** Use a song's description, lyrics and seed again. */
export function reuse(s: Song): void {
  state.form.prompt = s.prompt;
  state.form.lyrics = s.lyrics;
  state.form.seed = s.seed;
  saveForm();
  render();
}

/** A structure tag on a line of its own, at the end of the lyrics. */
export function addTag(tag: string): void {
  const l = state.form.lyrics.replace(/\s+$/, "");
  state.form.lyrics = `${l}${l ? "\n" : ""}[${tag}]\n`;
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
    model: state.info?.model, instructions: f.prompt.trim(), input: f.lyrics.trim(), seconds: f.seconds, response_format: "url",
  };
  if (f.seed !== null) b.seed = f.seed;
  if (f.steps !== null && f.steps !== d?.steps) b.steps = f.steps;
  if (f.cfg !== null && f.cfg !== d?.cfg) b.cfg = f.cfg;
  const opts = Object.fromEntries(Object.entries(f.options).filter(([k, v]) =>
    v.trim() !== "" && (state.info?.options ?? []).some((o) => o.name === k)));
  if (Object.keys(opts).length) b.options = opts;
  return b;
}

export async function generate(): Promise<void> {
  if (state.busy || !state.info || !state.form.prompt.trim()) return;
  saveForm();
  state.busy = true;
  state.error = null;
  state.status = "starting";
  render();
  try {
    const j = await api.generate(body());
    state.result = j.data[0] ?? null;
    const n = j.nextsycl;
    const wh = n.wh != null ? ` · ${n.wh.toFixed(2)} Wh` : "";
    state.status = `done: ${n.seconds.toFixed(1)} s of sound in ${n.took.toFixed(1)} s (${(n.seconds / n.took).toFixed(2)}× real time)${wh}`;
    await refreshHistory();
  } catch (e) {
    const m = e instanceof Error ? e.message : String(e);
    state.error = m === "cancelled" ? null : m;
    state.status = m === "cancelled" ? "cancelled" : "failed";
  }
  state.busy = false;
  render();
}

export async function cancel(): Promise<void> {
  state.status = "cancelling";
  render();
  await api.cancel();
}
