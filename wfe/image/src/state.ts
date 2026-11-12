/** The whole client state in one object (H3's scheme). Components read it; actions mutate it and call render(). */
import type { Described, DescribeForm, DescribeJob, Describers, Form, Given, Gpu, Info, Picture, Progress } from "./types.js";

/** The Describe panel: a picture in, the prompt that would draw it out (actions.ts runs it). */
export interface DescribeState {
  /** the front door's describers; null: not asked yet; "no": this page's server stands alone (no route) */
  list: Describers | "no" | null;
  /** the picture to describe (a data: URL) */
  picture: Given | null;
  form: DescribeForm;
  /** the job's label while one runs (a cancel names it) */
  label: string | null;
  job: DescribeJob | null;
  started: number | null;
  /** the answer, and its prompts as they are edited */
  result: Described | null;
  prompt: string;
  negative: string;
  error: string | null;
  /** what "use as prompt" did, in words */
  used: string | null;
}

export interface AppState {
  info: Info | null;
  progress: Progress | null;
  gpu: Gpu | null;
  history: Picture[];
  /** false until the history's first answer (a spinner stands in for the list) */
  historyLoaded: boolean;
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
  describe: DescribeState;
}

export const state: AppState = {
  info: null,
  progress: null,
  gpu: null,
  history: [],
  historyLoaded: false,
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
  describe: {
    list: null, picture: null,
    form: { target: "", describer: "best", detail: "medium", maxWords: null, negative: false, language: "", instructions: "" },
    label: null, job: null, started: null, result: null, prompt: "", negative: "", error: null, used: null,
  },
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
const CLOSED_BY_DEFAULT = new Set(["sampling", "describe"]);

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

/** The Describe panel's choices survive a reload (the picture and the answer do not: they are big). */
export function saveDescribe(): void {
  save("nsimg.describe", state.describe.form);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("nsimg.panels", {});
  state.describe.form = { ...state.describe.form, ...load<Partial<DescribeForm>>("nsimg.describe", {}) };
  const f = load<Partial<Form> | null>("nsimg.form", null);
  if (f) state.form = { ...state.form, ...f, loras: { ...(f.loras ?? {}) }, options: { ...(f.options ?? {}) } };
}
