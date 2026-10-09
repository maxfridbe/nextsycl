/** The whole client state in one object (H3's scheme). Components read it; actions mutate it and call render(). */
import type { Form, Gpu, Info, Progress, Song } from "./types.js";

export interface AppState {
  info: Info | null;
  progress: Progress | null;
  gpu: Gpu | null;
  history: Song[];
  form: Form;
  /** the last request's song */
  result: Song | null;
  busy: boolean;
  status: string;
  error: string | null;
  /** which collapsible regions are open, by panel id. Persisted. */
  panels: Record<string, boolean>;
  /** songs whose lyrics are shown, by url */
  shown: Record<string, boolean>;
}

export const state: AppState = {
  info: null,
  progress: null,
  gpu: null,
  history: [],
  form: { prompt: "", lyrics: "", seconds: 60, steps: null, cfg: null, seed: null, options: {} },
  result: null,
  busy: false,
  status: "idle",
  error: null,
  panels: {},
  shown: {},
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
const CLOSED_BY_DEFAULT = new Set(["advanced"]);

export function panelOpen(id: string): boolean {
  const v = state.panels[id];
  return v === undefined ? !CLOSED_BY_DEFAULT.has(id) : v;
}

export function togglePanel(id: string): void {
  state.panels[id] = !panelOpen(id);
  save("nsaud.panels", state.panels);
  render();
}

/** The form survives a reload (the description, the lyrics, every setting). */
export function saveForm(): void {
  save("nsaud.form", state.form);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("nsaud.panels", {});
  const f = load<Partial<Form> | null>("nsaud.form", null);
  if (f) state.form = { ...state.form, ...f, options: { ...(f.options ?? {}) } };
}
