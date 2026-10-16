/** The whole client state in one object (H3's scheme). Components read it; actions mutate it and call render(). */
import type { Form, Gpu, Info, Made, Progress } from "./types.js";

export interface AppState {
  info: Info | null;
  progress: Progress | null;
  gpu: Gpu | null;
  history: Made[];
  form: Form;
  /** "make" (pictures to a mesh) or "rig" (a mesh rigged or animated) */
  mode: "make" | "rig";
  /** the mesh in the viewer */
  shown: Made | null;
  /** what the viewer says about it once loaded */
  stats: string;
  busy: boolean;
  status: string;
  error: string | null;
  panels: Record<string, boolean>;
  view: { wire: boolean; flat: boolean; skeleton: boolean; playing: boolean };
}

export const state: AppState = {
  info: null,
  progress: null,
  gpu: null,
  history: [],
  form: { pictures: [], texture: true, seed: null, steps: null, cfg: null, resolution: null, faces: null, textureSize: null, removeBackground: true,
          mesh: "", meshName: "", file: "", prompt: "", frames: null, options: {} },
  mode: "make",
  shown: null,
  stats: "",
  busy: false,
  status: "idle",
  error: null,
  panels: {},
  view: { wire: false, flat: false, skeleton: true, playing: true },
};

type Listener = () => void;
let listener: Listener | null = null;

export function onRender(fn: Listener): void {
  listener = fn;
}

export function render(): void {
  listener?.();
}

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

const CLOSED_BY_DEFAULT = new Set(["advanced"]);

export function panelOpen(id: string): boolean {
  const v = state.panels[id];
  return v === undefined ? !CLOSED_BY_DEFAULT.has(id) : v;
}

export function togglePanel(id: string): void {
  state.panels[id] = !panelOpen(id);
  save("ns3d.panels", state.panels);
  render();
}

/** The settings survive a reload (not the pictures or the mesh: too big to keep) */
export function saveForm(): void {
  save("ns3d.form", { ...state.form, pictures: [], mesh: "", meshName: "" });
  save("ns3d.view", state.view);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("ns3d.panels", {});
  const f = load<Partial<Form> | null>("ns3d.form", null);
  if (f) state.form = { ...state.form, ...f, pictures: [], mesh: "", meshName: "", options: { ...(f.options ?? {}) } };
  state.view = { ...state.view, ...load<Partial<AppState["view"]>>("ns3d.view", {}) };
}
