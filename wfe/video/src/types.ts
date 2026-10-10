/** Shapes the server actually returns. Mirrors wfe/server.py; also served at /api/types.ts. */

export type Envelope<T> =
  | { ok: true; result: T }
  | { ok: false; error: { code: ErrorCode; message: string } };

export type ErrorCode =
  | "invalid_request" | "invalid_param" | "not_found"
  | "busy" | "not_running" | "unknown_method" | "internal";

/** "prev" = previous clip of this series, "first" = the series hub, "*_cam" = same camera only. */
export type Anchor = "prev" | "prev_cam" | "first" | "first_cam" | (string & {});
export type ChainMode = "png" | "video" | "latent" | "none";
/** a denoiser the engine has: INT8 (the default), or a GGUF k-quant it was given (Q6_K, Q4_K_M) */
export type Engine = string;
export type ClipState = "done" | "running" | "queued" | "held";

export interface GenerateRequest {
  prompt: string;
  seconds?: number;
  steps?: number;
  seed?: number;
  te?: "teacher" | "student";
  label?: string;
  width?: number;
  height?: number;
  engine?: Engine;
  chain_mode?: ChainMode;
  first_frame?: Anchor | null;
  last_frame?: Anchor | null;
  exposure_ref?: Anchor | null;
  first_audio?: Anchor | null;
  first_audio_s?: number;
  cond_noise_aug?: number | null;
  camera?: string | null;
  upscale?: number;
  upscaler?: string;
  /** ComfyUI's sampler and schedule names (none: Euler on the shifted schedule) */
  sampler?: string | null;
  schedule?: string | null;
  loras?: string[];
  /** references (Ref2VA): files in the studio's output directory, named in the prompt as <Picture i>, <Video k>,
   *  <Audio j> (a clip's own sound takes the next <Audio j> before its <Video k>) */
  ref_images?: string[] | null;
  ref_videos?: string[] | null;
  ref_audios?: string[] | null;
  ref_video_sound?: boolean | null;
  ref_image_size?: "match" | "max" | null;
  /** the keyframe pictures shown to the text encoder too (default on) */
  te_pictures?: boolean | null;
  /** the Fun ControlNet: a control video (canny / depth / HED / MLSD / pose frames), a mask (white: regenerate) over
   *  a source video; files in the studio's output directory */
  control_video?: string | null;
  control_mask?: string | null;
  control_source?: string | null;
  control_strength?: number | null;
  project?: string;
  batch?: string;
  queue?: boolean;
}

export interface Job extends GenerateRequest {
  id: string;
  name: string;
  started: number;
  restore_vllm?: boolean;
  energy_wh?: number | null;
  dialogue?: string;
  state?: ClipState;
}

export interface QueueItem {
  label: string;
  seconds: number;
  steps: number;
  engine?: Engine;
  size: string;
  chain: boolean;
  project: string;
  batch: string;
  held: boolean;
}

export interface Batch { batch: string; clips: number; video_s: number; paused: boolean }

export interface Project {
  project: string;
  clips: number;
  video_s: number;
  paused: boolean;
  first_index: number;
  batches: Batch[];
}

/** Which LLM owns the card when no render does: vLLM, the Strata Coder (llama.cpp), or none. */
export interface LlmStatus {
  mode: string;   // a key of choices, or "none"
  up: boolean;
  running: string[];
  starting: string | null;
  choices: Record<string, string>;
  url: string | null;
}

export interface Status {
  idle: boolean;
  llm?: LlmStatus;
  paused: boolean;
  paused_projects: string[];
  paused_batches: string[];
  job?: Job;
  stage?: string;
  pct?: number;
  done?: boolean;
  exited?: boolean;
  error?: string | null;
  eta?: number;
  elapsed?: number;
  rms?: number | null;
  times?: Record<string, number>;
  queue: string[];
  queue_items: QueueItem[];
  projects: Project[];
  all_projects: string[];
  version?: string;
  changed?: boolean;
}

export interface TimelineClip {
  n: number;
  name: string | null;
  state: ClipState;
  seconds: number;
  camera: string | null;
  label: string;
  dialogue: string;
  thumb: string | null;
}

export interface Film { file: string; mb: number; seconds: number | null; mtime: number }

export interface ListRow {
  file: string;
  label?: string;
  project?: string;
  camera?: string | null;
  seconds?: number;
  steps?: number;
  total?: number;
  energy_wh?: number | null;
  speech_pct?: number | null;
  engine?: string;
  size?: string;
  size2?: number;
  mtime?: number;
}

/** /api/plan: everything the canvas grid needs to price and disqualify a combination. */
export interface Canvas {
  width: number;
  height: number;
  aspect: string;
  note: string;
  mpx: number;
  cost: number;
  default: boolean;
}

export interface EngineInfo {
  quant: string;
  file: string;
  path: string;
  gib: number;
  ready: boolean;
  default: boolean;
}

export interface MeasuredCell {
  peak: number;
  resv: number | null;
  s_per_step: number | null;
  steps: number | null;
  tokens: number;
  name: string;
}

/** One GPU's measured step cost (`nextsycl video plan measure`): the 50 blocks at each token count, and the factor a
 *  real clip's step is of them. */
export interface GpuPlan {
  gpu: number;
  /** the denoiser measured (INT8 when absent: a plan from before engines) */
  engine?: string;
  name: string;
  shared: boolean;
  step_scale: number;
  clip_tokens: number | null;
  points: [number, number][];
  step_a: number;
  step_b: number;
  /** the canvas table's cells measured on this GPU, keyed "WxH|seconds": a clip step's time and peak VRAM, or
   *  `over` when the engine refused the size */
  cells?: Record<string, { tokens: number; s_per_step?: number; gib?: number; over?: boolean } | undefined>;
}

export interface Plan {
  engines: EngineInfo[];
  canvases: Canvas[];
  defaults: { seconds: number; steps: number; width: number; height: number; engine: string };
  cap_gib: number;
  gib_per_token: number;
  margin_gib: number;
  step_a: number;
  step_b: number;
  /** the sampler and schedule names the engine takes */
  samplers?: string[];
  schedules?: string[];
  chain_modes: Record<string, string>;
  /** keyed "<engine>|<W>x<H>|<seconds>" - cells this box has actually run */
  measured: Record<string, MeasuredCell | undefined>;
  /** measured per GPU; empty until `nextsycl video plan measure` has run on the box */
  gpus?: GpuPlan[];
}

export interface Disk { path: string; total_gb: number; used_gb: number; free_gb: number; pct: number }

export interface ProjectStats {
  project: string;
  clips: number;
  video_s: number;
  gpu_s: number;
  ratio: number | null;
  wh: number;
}

export interface Summary {
  rate_wall_per_video_s: number | null;
  rate_wh_per_video_s: number | null;
  running: boolean;
  queued: number;
  queued_video_s: number;
  remaining_wall_s: number;
  eta_ts: number | null;
  remaining_kwh: number | null;
  series: string | null;
  series_done: number;
  series_video_s: number;
  series_final_s: number;
  done_total: number;
  done_video_s: number;
  done_wall_s: number;
  done_wh: number;
  disk: Disk | null;
  project_stats: ProjectStats | null;
}

export interface Gpu {
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  busy_pct: number;
  temp_pkg: number | null;
  power_w: number | null;
  fan_rpm: number | null;
  freq_mhz: number | null;
  pkg_power_w?: number | null;
  power_cap_w?: number | null;
  temp_vram?: number | null;
  temp_vram_max?: number | null;
  temp_pcie?: number | null;
  temp_mctrl?: number | null;
  pcie?: { cur: PcieLink; card_max: PcieLink; slot_max: PcieLink } | null;
  host_load1?: number | null;
  host_cpus?: number | null;
  error?: string;
}

export interface PcieLink {
  gen: number | null;
  gts: string;
  width: number;
}

export interface JobDetail {
  job?: Job;
  times?: Record<string, number>;
  frames?: number;
  rms?: number;
  energy_wh?: number | null;
  error?: string;
}

export interface Scene {
  version: 1;
  title?: string;
  project: string;
  batch?: string;
  notes?: string;
  exported?: number;
  defaults: Partial<GenerateRequest>;
  clips: (Partial<GenerateRequest> & { n: number; prompt: string })[];
}

export interface Scope { project?: string; batch?: string }

export interface HoldResult {
  queued: number;
  held: number;
  paused_all: boolean;
  paused_projects: string[];
  paused_batches: string[];
}

/** RPC method -> [request, response]. Keeps rpc() honest at the call site. */
export interface Methods {
  status: [Record<string, never>, Status];
  wait: [{ version?: string; timeout_s?: number }, Status];
  generate: [GenerateRequest, { queued: boolean; position?: number; job?: Job }];
  cancel: [Record<string, never>, { stopped: true }];
  "queue.list": [Record<string, never>, { queue: GenerateRequest[] }];
  "queue.clear": [Scope, { cleared: number }];
  "queue.pause": [Scope, HoldResult];
  "queue.resume": [Scope, HoldResult];
  "projects.list": [Record<string, never>, { projects: Project[]; paused_all: boolean }];
  timeline: [{ project: string }, { clips: TimelineClip[] }];
  "scene.export": [{ project: string }, Scene];
  "scene.import": [
    { scene: Scene; project?: string; paused?: boolean },
    { imported: number; project: string; queued: number; held: number },
  ];
  "films.list": [Record<string, never>, { films: Film[] }];
  summary: [Record<string, never>, Summary];
  gpu: [Record<string, never>, Gpu];
  "llm.mode": [{ mode?: string }, LlmStatus];
}
