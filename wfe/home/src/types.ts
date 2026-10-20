/** /api/state and /api/spec (glue/serve/src/home). */

export interface Proc {
  pid: number;
  vram_mb: number;
  program: string;
  kind?: string;
  command?: string;
  model?: string | null;
  /** its model's registry id, when its file is a registered one */
  model_id?: string | null;
  port?: string | null;
  gpus?: string[];
  args?: string;
}

/** a model vouched for on a card: `mb` all of it, `headroom_mb` of that its working memory beyond the weights (KV for
 *  its context, arenas, a song's or an image's activations) - from its model card when given, else from its files */
export interface Vouched { model: string; kind: string; mb: number; headroom_mb: number; headroom_given: boolean; measured: boolean; state: string; of?: number }

export interface Card {
  index: number;
  name: string;
  pci: string;
  vram_used_mb: number;
  vram_total_mb: number;
  busy_pct: number;
  power_w: number | null;
  power_cap_w: number | null;
  temp_pkg: number | null;
  temp_vram: number | null;
  pcie: { cur: { gen: number | null; width: number } } | null;
  procs: Proc[];
  vouched: Vouched[];
  vouched_mb: number;
  vouched_headroom_mb: number;
  overbooked: boolean;
}

export type ModelState = "disabled" | "unloaded" | "loading" | "loaded" | "busy" | "ready";

export interface Model {
  id: string;
  kind: "llm" | "image" | "audio" | "video" | "model3d";
  title: string | null;
  arch: string | null;
  enabled: boolean;
  gpus: number[];
  /** a chat model spread over every card (its model card: gpus all): its cards are not a choice */
  spans?: boolean;
  state: ModelState;
  vouch_mb: Record<string, number>;
  /** of vouch_mb, the headroom (Vouched) */
  headroom_mb: Record<string, number>;
  headroom_given: boolean;
  measured: boolean;
  page: number;
}

export interface Service { id: string; title: string; what: string; port: number; up: boolean; detail?: string | null; link?: string; api?: string }

export interface Run { id: number; label: string; started: number; rc: number | null; out: string }

export interface State {
  ts: number;
  host: { mem_total_mb: number; mem_avail_mb: number; load1: number | null; cpus: number };
  cards: Card[];
  models: Model[];
  services: Service[];
  runs: Run[];
  idle_minutes: number;
  ports: { video: number; chat_ui: number };
  /** maintenance mode while it is on (system.maintenance): the notice callers read, the bypass key */
  maintenance: Maintenance | null;
  /** the drives the models are on, each model's place on them, the move under way (system.drives) */
  drives?: Drives;
  /** the models a card offers that are not downloaded (system.models' downloaded: false) */
  available: Available[];
  /** the downloads under way and the last ones, newest first (system.models.pulls) */
  pulls: Pull[];
  /** NS_AUTO_PULL: a request for a model not downloaded fetches it and waits */
  auto_pull: boolean;
}

export interface Available {
  id: string; kind: string; title?: string; about?: string; bytes: number;
  /** the drive and folder it would go to, whether it has the room, its free bytes */
  drive: string; folder: string; fits: boolean; free: number;
  pull: Pull | null;
}

export interface Pull {
  model: string; kind: string; title: string; drive: string; folder: string;
  state: "queued" | "downloading" | "verifying" | "done" | "failed" | "cancelled";
  error: string | null; bytes_total: number; bytes_done: number; pct: number; speed: number; eta_s: number | null;
  started: number; ended: number | null; run: number;
}

export interface Measured { mb_s: number; bytes: number; seconds: number; at: number; file: string }

export interface Drive {
  id: string;
  disk: string;
  model: string;
  link: string | null;
  device: string;
  fstype: string;
  mount: string;
  /** where models moved here go */
  folder: string;
  size: number;
  free: number;
  /** what a move leaves free */
  margin: number;
  models_bytes: number;
  models: string[];
  measured: Measured | null;
  measuring: boolean;
  /** as measured, else its link's */
  rate_gb_s: number | null;
}

/** a model's files: bytes by drive, and for each other drive what a move needs or why it cannot move now */
export interface DiskModel {
  bytes: number;
  files: number;
  missing: number;
  drives: Record<string, number>;
  streams: boolean;
  moving: boolean;
  move: Record<string, { need?: number; why?: string }>;
}

export interface Move {
  model: string;
  to: string;
  from: string[];
  state: "copying" | "registry" | "removing" | "done" | "failed" | "cancelled";
  running: boolean;
  files: number;
  file_index: number;
  file: string | null;
  bytes: number;
  done: number;
  work: number;
  pct: number;
  rate_mb_s: number;
  eta_s: number | null;
  started: number;
  seconds: number;
  error: string | null;
  warning: string | null;
}

export interface Drives { drives: Drive[]; models: Record<string, DiskModel>; move: Move | null; measuring: string | null; folders: string[] }

export interface Maintenance { on: boolean; message: string; since: number; until: number | null; retry_after_s: number; notice: string; bypass?: string }

export interface Operation {
  tags: string[];
  operationId?: string;
  summary: string;
  requestBody?: { content: { "application/json": { example?: unknown; schema?: { properties?: Record<string, { type: string; description?: string; enum?: string[] }>; required?: string[] } } } };
  parameters?: unknown[];
  "x-curl"?: string;
}

export interface Spec {
  token: string;
  spec: { info: { description: string; version: string }; servers: { url: string }[]; paths: Record<string, Record<string, Operation>> };
}

export interface SettingRow {
  name: string;
  group?: string;
  default?: string;
  what: string;
  readers: string[];
  file: string | null;
  env?: string | null;
}

export interface Config {
  path: string;
  text: string;
  settings: SettingRow[];
  other: SettingRow[];
  prefixes: { prefix: string; what: string; readers: string[] }[];
}

export interface ConfigSaved { changed: string[]; restart: string[]; note?: string | null }
