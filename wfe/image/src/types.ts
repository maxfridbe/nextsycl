/** What the image server answers (glue/serve/src/image.rs). */

export interface Defaults {
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  schedule: string;
  shift: number;
}

export interface LoraInfo {
  id: string;
  title: string;
  /** the scale it is merged at now; null: not loaded */
  loaded: number | null;
}

export interface Info {
  model: string;
  loading?: boolean;
  arch?: string;
  edits?: boolean;
  /** whether it takes guidance: cfg above 1 and a negative prompt */
  guidance?: boolean;
  defaults?: Defaults;
  samplers?: string[];
  schedules?: string[];
  loras?: LoraInfo[];
  /** the engine's own per-request options (sent as `options`) */
  options?: { name: string; value: string; help: string }[];
  saves?: boolean;
}

export interface Progress {
  busy: boolean;
  loading?: boolean;
  prompt?: string;
  picture?: number;
  pictures?: number;
  at?: number;
  of?: number;
  seconds?: number;
  waiting?: number;
}

export interface Gpu {
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  power_w: number | null;
  temp_pkg: number | null;
  temp_vram: number | null;
}

export interface Picture {
  url: string;
  prompt: string;
  seed: number;
  steps: number;
  width: number;
  height: number;
  loras: string[];
  seconds: number;
  wh?: number | null;
  created: number;
}

export interface GenAnswer {
  created: number;
  data: { url?: string; b64_json?: string; seed: number; revised_prompt: string }[];
  nextsycl: { seconds: number; steps: number; seed: number; loras: string[]; wh: number | null };
}

/** The form: every option a request can carry. */
export interface Form {
  prompt: string;
  negative: string;
  width: number;
  height: number;
  steps: number | null;
  n: number;
  seed: number | null;
  sampler: string;
  schedule: string;
  cfg: number | null;
  shift: number | null;
  rgba: boolean;
  /** id -> scale, for the LoRAs ticked */
  loras: Record<string, number>;
  /** the engine's own options, by name (empty: not sent) */
  options: Record<string, string>;
}
