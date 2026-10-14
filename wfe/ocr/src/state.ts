/** The whole client state in one object (H3's scheme). Components read it; actions mutate it and call render(). */
import type { Form, Gpu, Info, Page, Progress, Reading } from "./types.js";

/** The instructions the model reads by: the server's (each model its own), else Unlimited-OCR's; and one's own */
const FALLBACK: [string, string, string][] = [
  ["parse", "document parsing.", "Document → Markdown with boxes (the model's own)"],
  ["free", "Free OCR.", "Plain text"],
];

export function presets(): [string, string, string][] {
  const p = state.info?.prompts;
  const list: [string, string, string][] = p && p.length ? p.map((x) => [x.id, x.prompt, x.title]) : FALLBACK;
  return [...list, ["custom", "", "Your own"]];
}

export type View = "doc" | "boxes" | "raw" | "text";

export interface AppState {
  info: Info | null;
  progress: Progress | null;
  gpu: Gpu | null;
  history: Reading[];
  form: Form;
  pages: Page[];
  /** the reading shown (the last one, or one picked from the history) and the pages it read */
  result: Reading | null;
  resultPages: Page[];
  view: View;
  /** the page whose boxes are drawn (several pages) */
  boxPage: number;
  dragging: boolean;
  busy: boolean;
  /** what is being done besides the reading itself (rendering a PDF, page 3 of 12) */
  note: string | null;
  /** page by page: the pages read so far (with their <PAGE> marks), and which page is being read */
  partial: string;
  pageAt: number;
  started: number | null;
  error: string | null;
  panels: Record<string, boolean>;
}

export const state: AppState = {
  info: null,
  progress: null,
  gpu: null,
  history: [],
  form: { preset: "parse", prompt: "", tiles: true, maxTokens: 8192, temperature: 0, guard: true, figures: true, perPage: true, columns: true },
  pages: [],
  result: null,
  resultPages: [],
  view: "doc",
  boxPage: 0,
  dragging: false,
  busy: false,
  note: null,
  partial: "",
  pageAt: 0,
  started: null,
  error: null,
  panels: {},
};

type Listener = () => void;
let listener: Listener | null = null;

export function onRender(fn: Listener): void {
  listener = fn;
}

export function render(): void {
  listener?.();
}

/* Every storage access is wrapped: a private window throws, and the page has to work there too. */
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

export function panelOpen(id: string): boolean {
  return state.panels[id] ?? true;
}

export function togglePanel(id: string): void {
  state.panels[id] = !panelOpen(id);
  save("nsocr.panels", state.panels);
  render();
}

export function saveForm(): void {
  save("nsocr.form", state.form);
  save("nsocr.view", state.view);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("nsocr.panels", {});
  state.form = { ...state.form, ...load<Partial<Form>>("nsocr.form", {}) };
  state.view = load<View>("nsocr.view", "doc");
}

/** The instruction the form asks for */
export function promptText(): string {
  const all = presets();
  // a preset this model does not have: its default
  const p = all.find((x) => x[0] === state.form.preset) ?? all[0];
  return !p || p[0] === "custom" ? state.form.prompt : p[1];
}
