/** The whole client state in one object. Components read it; actions mutate it and call
 *  render(). No component keeps its own copy, which is what made the old page drift. */
import type {
  Film, Gpu, Job, JobDetail, ListRow, Plan, Status, Summary, TimelineClip,
} from "./types.js";

export type Tab = "create" | "jobs";

export interface AppState {
  tab: Tab;
  status: Status | null;
  summary: Summary | null;
  gpu: Gpu | null;
  plan: Plan | null;
  /** the GPU the canvas table is timed for (its switch); null: the fastest measured */
  pickGpu: number | null;
  list: ListRow[];
  films: Film[];
  filter: string;

  timelineProject: string | null;
  timelineClips: TimelineClip[];
  selected: number | null;
  pxPerSecond: number;

  /** Which collapsible regions are open, by panel id. Persisted. */
  panels: Record<string, boolean>;
  /** Human names for cameras, keyed "<project>|<camera>". Persisted. */
  cameraNames: Record<string, string>;

  detail:
    | { kind: "job"; name: string; data: JobDetail }
    | { kind: "clip"; label: string; data: Job }
    | null;
  /** Cued video. `playing` stays false until the poster is clicked, so nothing autoplays. */
  video: { file: string; poster: string | null; playing: boolean } | null;
  error: string | null;
}

export const state: AppState = {
  tab: "create",
  status: null,
  summary: null,
  gpu: null,
  plan: null,
  pickGpu: null,
  list: [],
  films: [],
  filter: "",
  timelineProject: null,
  timelineClips: [],
  selected: null,
  pxPerSecond: 14,
  panels: {},
  cameraNames: {},
  detail: null,
  video: null,
  error: null,
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

/** Panels default to open; only an explicit false collapses one. */
export function panelOpen(id: string): boolean {
  return state.panels[id] !== false;
}

export function togglePanel(id: string): void {
  state.panels[id] = !panelOpen(id);
  save("h3panels", state.panels);
  render();
}

export function cameraKey(project: string | null, camera: string | null): string {
  return `${project ?? ""}|${camera ?? ""}`;
}

/** "Camera A" until someone renames it to what it actually is. */
export function cameraName(project: string | null, camera: string | null): string {
  const custom = state.cameraNames[cameraKey(project, camera)];
  if (custom) return custom;
  if (!camera || camera === "—") return "Single camera";
  return `Camera ${camera}`;
}

export function setCameraName(project: string | null, camera: string | null, name: string): void {
  const k = cameraKey(project, camera);
  if (name.trim()) state.cameraNames[k] = name.trim();
  else delete state.cameraNames[k];
  save("h3cameras", state.cameraNames);
  render();
}

export function rememberTab(tab: Tab): void {
  save("h3tab", tab);
}

export function restorePreferences(): void {
  state.panels = load<Record<string, boolean>>("h3panels", {});
  state.cameraNames = load<Record<string, string>>("h3cameras", {});
  const t = load<Tab | null>("h3tab", null);
  if (t === "create" || t === "jobs") state.tab = t;
}
