/** /api/state and /api/spec (glue/serve/src/home). */

export interface Proc {
  pid: number;
  vram_mb: number;
  program: string;
  kind?: string;
  command?: string;
  model?: string | null;
  port?: string | null;
  gpus?: string[];
  args?: string;
}

export interface Vouched { model: string; kind: string; mb: number; measured: boolean; state: string; of?: number }

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
  overbooked: boolean;
}

export type ModelState = "disabled" | "unloaded" | "loading" | "loaded" | "busy" | "ready";

export interface Model {
  id: string;
  kind: "llm" | "image" | "audio" | "video";
  title: string | null;
  arch: string | null;
  enabled: boolean;
  gpus: number[];
  state: ModelState;
  vouch_mb: Record<string, number>;
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
}

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
