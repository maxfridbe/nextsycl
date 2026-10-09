/** What the audio server answers (glue/serve/src/audio.rs). */

export interface Defaults {
  seconds: number;
  max_seconds: number;
  steps: number;
  cfg: number;
  /** samples a second of what it returns */
  rate: number;
}

export interface Info {
  model: string;
  /** the engine is running a request: what changes is not reported */
  busy?: boolean;
  arch?: string;
  /** whether it sings words */
  lyrics?: boolean;
  defaults?: Defaults;
  /** the engine's own per-request options (sent as `options`) */
  options?: { name: string; value: string; help: string }[];
  report?: string[];
  saves?: boolean;
}

export interface Progress {
  busy: boolean;
  prompt?: string;
  /** prompt | tokens | flow | decode */
  phase?: string;
  at?: number;
  of?: number;
  seconds?: number;
  /** the length asked for */
  want?: number;
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

export interface Song {
  url: string;
  prompt: string;
  lyrics: string;
  seed: number;
  steps?: number;
  cfg?: number;
  /** seconds of sound */
  seconds: number;
  /** seconds it took to make */
  took?: number;
  wh?: number | null;
  model?: string;
  created: number;
}

export interface GenAnswer {
  created: number;
  data: Song[];
  nextsycl: { took: number; seconds: number; seed: number; steps: number; cfg: number; wh: number | null };
}

/** The form: every option a request can carry. */
export interface Form {
  prompt: string;
  lyrics: string;
  seconds: number;
  steps: number | null;
  cfg: number | null;
  seed: number | null;
  /** the engine's own options, by name (empty: not sent) */
  options: Record<string, string>;
}
