/** /api/state (glue/serve/src/home.rs). */

export interface Proc {
  pid: number;
  vram_mb: number;
  program: string;
  /** a nextsycl command: its kind, command, model, port and GPUs */
  kind?: string;
  command?: string;
  model?: string | null;
  port?: string | null;
  gpus?: string[];
  args?: string;
}

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
  freq_mhz: number | null;
  pcie: { cur: { gen: number | null; width: number } } | null;
  procs: Proc[];
}

export interface ChatModes {
  mode: string | null;
  choices: Record<string, string> | null;
  starting: string | null;
  up: boolean | null;
}

export interface Service {
  id: string;
  kind?: string;
  title: string;
  what: string;
  port: number;
  up: boolean;
  model?: string | null;
  detail?: string | null;
  busy?: boolean | null;
  waiting?: number | null;
  /** the server has its web page (started with --wfe) */
  page?: boolean | null;
  link?: string;
  api?: string;
  startable?: boolean;
  chat?: ChatModes | null;
}

export interface Model { id: string; title: string | null; kind: string; gpus: string | null }

export interface Run { id: number; label: string; started: number; rc: number | null; out: string }

export interface State {
  ts: number;
  host: { mem_total_mb: number; mem_avail_mb: number; load1: number | null; cpus: number };
  cards: Card[];
  services: Service[];
  models: Model[];
  runs: Run[];
  gpustat: { pci: string; age: number | null } | null;
}
