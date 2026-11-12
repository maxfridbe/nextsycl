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
  /** the grid the picture's sides are multiples of */
  grid?: number;
  /** whether "transparent" gives an alpha channel */
  transparent?: boolean;
  /** the aspect ratios ("W:H") and pixel budgets it offers for a size (a size is a budget at a ratio) */
  sizes?: { ratios: string[]; pixels: number[] } | null;
  /** whether the schedule's shift can be set */
  shift?: boolean;
  /** the model writes pictures as SVG (POST /v1/images/vectorize) instead of drawing */
  vectorize?: boolean;
  /** the model enlarges pictures (POST /v1/images/upscale): its variants */
  upscale?: { variant: string; about: string; scales: number[] }[];
  defaults?: Defaults;
  samplers?: string[];
  schedules?: string[];
  loras?: LoraInfo[];
  /** the engine's own per-request options (sent as `options`) */
  options?: { name: string; value: string; help: string }[];
  saves?: boolean;
  /** the most pictures an edit takes */
  max_pictures?: number;
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
  /** the pictures it was made from (an edit) */
  pictures?: number;
  created: number;
}

export interface GenAnswer {
  created: number;
  data: { url?: string; b64_json?: string; seed: number; revised_prompt: string; width?: number; height?: number }[];
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

/** A picture given to an edit: a data: URL and its size */
export interface Given {
  url: string;
  name: string;
  w: number;
  h: number;
}

/** What the front door answers the page's Describe panel (glue/serve/src/home/describe_page.rs; the contract is
 *  docs/api/image-describe.md). */

/** One of image.describers' list: who can read a picture. */
export interface Describer {
  id: string;
  name: string;
  /** a chat model, or an image model's own text encoder */
  kind: "chat" | "native";
  size_gb: number | null;
  vram_gib: number | null;
  image_models: string[];
  downloaded: boolean;
  enabled: boolean;
  /** the one "best" runs now */
  best: boolean;
}

export interface Describers {
  default: string;
  best: string | null;
  describers: Describer[];
}

export interface Described {
  prompt: string;
  negative_prompt: string | null;
  target: string;
  target_model: string | null;
  describer: string;
  describer_model: string | null;
  model: string;
  words: number;
  width: number;
  height: number;
  suggested_size: { width: number; height: number; size: string } | null;
  timings: { pp_tok_s?: number; tg_tok_s?: number; seconds: number };
  /** "stop", or "length" when max_tokens cut the answer */
  finish: string;
  notes: string[];
}

/** image.status's view of the job (the part the panel shows). */
export interface DescribeJob {
  label: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  stage?: string;
  waiting_for?: string | null;
  position?: number;
  result?: Described;
  error?: { code: string; message: string; field?: string | null };
}

/** The Describe panel's fields (the request) and its answer. */
export interface DescribeForm {
  /** "" the model this page serves; else a family or a model's id */
  target: string;
  /** best, native or a describer's id */
  describer: string;
  detail: "short" | "medium" | "long";
  maxWords: number | null;
  negative: boolean;
  language: string;
  instructions: string;
}
