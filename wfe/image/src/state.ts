/** The whole client state in one object (H3's scheme). Components read it; actions mutate it and call render(). */
import type { Form, Given, Gpu, Info, Picture, Progress } from "./types.js";

export interface AppState {
  info: Info | null;
  progress: Progress | null;
  gpu: Gpu | null;
  history: Picture[];
  form: Form;
  /** the last request's pictures */
  results: Picture[];
  /** the picture opened large */
  zoom: Picture | null;
  /** an edit's pictures: the first is changed, the others are references (not kept across a reload: they are big) */
  pictures: Given[];
  /** an edit's size follows its pictures (the last one's aspect) */
  sizeFromPictures: boolean;
  busy: boolean;
  /** seconds since the click, while a request runs */
  started: number | null;
  status: string;
  error: string | null;
  /** which collapsible regions are open, by panel id. Persisted. */
  panels: Record<string, boolean>;
}

export const state: AppState = {
  info: null,
  progress: null,
  gpu: null,
  history: [],
  form: {
    prompt: "", negative: "", width: 1024, height: 1024, steps: null, n: 1, seed: null,
    sampler: "", schedule: "", cfg: null, shift: null, rgba: false, loras: {}, options: {},
  },
  results: [],
  zoom: null,
  pictures: [],
  sizeFromPictures: true,
  busy: false,
  started: null,
  status: "idle",
  error: null,
  panels: {},
};

type Listener = () => void;
let listener: Listener | null = null;

export function onRender(fn: Listener): void {
  listener = fn;
}

/** Ask for a repaint. Cheap to call; snabbdom diffs. */
export function render(): void {
  listener?.();
}

/* -- persistence ----------------------------------------------------------------------
 * Every read and write is wrapped: a private window throws on localStorage access, and
 * the page has to work there too. */

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private window */
  }
}

/** Panels default to open, but for those listed here; an explicit flag wins. */
const CLOSED_BY_DEFAULT = new Set(["sampling"]);

export function panelOpen(id: string): boolean {
  const v = state.panels[id];
  return v === undefined ? !CLOSED_BY_DEFAULT.has(id) : v;
}

export function togglePanel(id: string): void {
  state.panels[id] = !panelOpen(id);
  save("nsimg.panels", state.panels);
  render();
}

/** The form survives a reload (the prompt and every setting). */
export function saveForm(): void {
  save("nsimg.form", state.form);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("nsimg.panels", {});
  const f = load<Partial<Form> | null>("nsimg.form", null);
  if (f) state.form = { ...state.form, ...f, loras: { ...(f.loras ?? {}) }, options: { ...(f.options ?? {}) } };
}
