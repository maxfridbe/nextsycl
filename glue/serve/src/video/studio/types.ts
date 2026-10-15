// Types for the H3 web front end RPC layer. Served at /api/types.ts and generated from the
// running server, so it matches the build you are talking to.
//
//   const r = await fetch("http://127.0.0.1:8095/rpc/status", {
//     method: "POST", headers: { "content-type": "application/json" }, body: "{}",
//   });
//   const { result } = (await r.json()) as Envelope<Status>;

export type Envelope<T> =
  | { ok: true; result: T }
  | { ok: false; error: { code: ErrorCode; message: string } };

export type ErrorCode =
  | "invalid_request" | "invalid_param" | "not_found"
  | "busy" | "not_running" | "unknown_method" | "internal";

/** "prev" = previous clip of this series, "first" = the series hub, "*_cam" = same camera only. */
export type Anchor = "prev" | "prev_cam" | "first" | "first_cam" | (string & {});
export type ChainMode = "png" | "video" | "latent" | "none";
/** engines.list's quant: INT8 (default; Q8_0 is read as it), Q6_K, Q4_K_M, REF2VA (chosen by itself for ref_images / ref_videos) */
export type Engine = "INT8" | "Q6_K" | "Q4_K_M" | "REF2VA" | (string & {});
export type ClipState = "done" | "running" | "queued" | "held";

export interface GenerateRequest {
  prompt: string;
  seconds?: number;          // 1-15.1, snapped up to 17k + 5 frames; the model is trained on 124-362 frames at 24 fps
  frames?: number;           // instead of seconds: exactly 17k + 5 (22-362), else 400 naming the nearest
  reentrant_id?: string;     // idempotent submit: a clip of this id queued, running or made is answered instead
  chain?: Chain;             // sets guide_clip, first_audio(_s), first_frame, chain_mode, exposure_ref (flat keys win)
  negative_prompt?: string;  // ignored (cfg 1: nothing to steer); listed in the answer's ignored
  steps?: number;            // 1-40; below 8 only with a turbo LoRA
  seed?: number;
  te?: "teacher" | "student";
  label?: string;            // "Name NN/MM: ..." - the NN drives edit order
  width?: number; height?: number;  // 256-1344, multiples of 32, width * height <= 1032192
  engine?: Engine;
  chain_mode?: ChainMode;
  first_frame?: Anchor | null;   // also accepts a still dropped into out/
  last_frame?: Anchor | null;
  exposure_ref?: Anchor | null;
  first_audio?: Anchor | null;
  first_audio_s?: number;
  cond_noise_aug?: number | null;  // harmful here: decodes as visible confetti
  camera?: string | null;
  upscale?: number;          // latent upscale at decode, 1.5 = free resolution
  loras?: string[];          // "path:strength"
  ref_audios?: string[];     // at most 3; voices to speak in: <Audio 1>, <Audio 2> ... (files in out/)
  guide_clip?: string | null;  // motion guide: "<anchor or file>[:frames[:at]]" - the last frames of a clip as one moving keyframe
  shift_video?: number; shift_audio?: number;  // flow shifts (12, 3)
  source?: Anchor | null;    // masked run: a clip whose latents are kept outside the regenerated part
  regen?: string | null;     // "from-to" seconds to regenerate; a source shorter than the clip is extended
  regen_box?: string | null; // "x0,y0,x1,y1" pixels: regenerate only inside
  project?: string; batch?: string;
  queue?: boolean;           // busy without it: 409 busy
  priority?: number;         // -9..9: a queued clip goes ahead of lower ones
  upscaler?: "esrgan-general" | "esrgan-anime" | "latent";
  ref_images?: string[];     // at most 9 (Ref2VA)
  ref_videos?: string[];     // at most 3
  ref_image_size?: "match" | "max";
  ref_video_sound?: boolean; te_pictures?: boolean;
  control_video?: string; control_mask?: string; control_source?: string;
  control_strength?: number; control_start?: number; control_end?: number;
  sampler?: string; schedule?: string;
}

export interface Chain {
  from?: Anchor;             // default prev
  context_frames?: number;   // 0 or 17k + 5: the last N frames as a moving keyframe at frame 0 (trim N/2 each side when joining)
  sound?: boolean;           // with context_frames: the last N/24 s of sound too (default true)
  still?: boolean;           // from's last frame as first_frame (default: when context_frames is 0)
  mode?: "video" | "png" | "latent";
  color_reanchor?: Anchor | boolean;  // exposure_ref: the carried pictures scaled to its brightness
}

export interface GenerateAnswer {
  queued: boolean; position?: number; job?: Job; label?: string; project?: string; batch?: string; held?: boolean;
  frames: number; seconds: number; ignored: string[];
  reentered: boolean; state?: "queued" | "held" | "running" | "done"; name?: string; file?: string;
}

export interface Job extends GenerateRequest {
  id: string; name: string; started: number;
  restore_vllm?: boolean; energy_wh?: number | null;
}

export interface Status {
  idle: boolean;
  paused: boolean;
  paused_projects: string[];
  paused_batches: string[];
  job: Job | null;            // null: nothing made since the studio started
  stage?: string; pct?: number; done?: boolean; error?: string | null;
  eta?: number; elapsed?: number;
  queue: string[];
  queue_items: QueueItem[];
  projects: Project[];
  all_projects: string[];
  version?: string;
  preview?: string | null;    // while it samples: a picture of the shot so far (PNG URL, new each step)
}

export interface QueueItem {
  label: string; seconds: number; steps: number;
  engine?: Engine; size: string; chain: boolean;
  project: string; batch: string; held: boolean;
}

export interface Project {
  project: string; clips: number; video_s: number; paused: boolean; first_index: number;
  batches: { batch: string; clips: number; video_s: number; paused: boolean }[];
}

export interface TimelineClip {
  n: number; name: string | null; state: ClipState;
  seconds: number; camera: string | null; label: string;
  dialogue: string;              // spoken words only, for alt text
  thumb: string | null;          // "/thumb/<name>.webp"
}

export interface Film { file: string; mb: number; seconds: number | null; mtime: number; }

/** Portable scene document: prompts are final, a runner just merges defaults and posts. */
export interface Scene {
  version: 1;
  title?: string; project: string; batch?: string; notes?: string; exported?: number;
  defaults: Partial<GenerateRequest>;
  clips: (Partial<GenerateRequest> & { n: number; prompt: string })[];
}

export interface Methods {
  "status": [Record<string, never>, Status];
  "wait": [{ version?: string; timeout_s?: number /* 0-60 */ }, Status & { changed: boolean }];
  "files.stat": [{ name: string }, { name: string; bytes: number; sha256: string; mtime: number }];
  "files.delete": [{ name: string }, { deleted: string[] }];
  "generate": [GenerateRequest, GenerateAnswer];
  "cancel": [{ label?: string }, { stopped: boolean; removed: number; label?: string }];
  "queue.move": [{ label: string; position?: number; priority?: number }, { label: string; position: number; queue: string[] }];
  "queue.remove": [{ label: string }, { removed: number; label: string }];
  "queue.list": [Record<string, never>, { queue: GenerateRequest[] }];
  "queue.clear": [Scope, { cleared: number }];
  "queue.pause": [Scope, HoldResult];
  "queue.resume": [Scope, HoldResult];
  "projects.list": [Record<string, never>, { projects: Project[]; paused_all: boolean }];
  "timeline": [{ project: string }, { clips: TimelineClip[] }];
  "scene.export": [{ project: string }, Scene];
  "scene.import": [{ scene: Scene; project?: string; paused?: boolean },
                   { imported: number; project: string; queued: number; held: number }];
  "films.list": [Record<string, never>, { films: Film[] }];
  "jobs.list": [{ limit?: number }, { jobs: unknown[] }];
  "jobs.get": [{ name: string }, unknown];
  "plan": [Record<string, never>, unknown];
  "engines.list": [Record<string, never>, unknown];
  "canvases.list": [Record<string, never>, unknown];
  "templates.list": [Record<string, never>, unknown];
  "gpu": [Record<string, never>, unknown];
  "summary": [Record<string, never>, unknown];
}

export interface Scope { project?: string; batch?: string; }
export interface HoldResult {
  queued: number; held: number; paused_all: boolean;
  paused_projects: string[]; paused_batches: string[];
}

export async function rpc<M extends keyof Methods>(
  base: string, method: M, params: Methods[M][0],
): Promise<Methods[M][1]> {
  const r = await fetch(`${base}/rpc/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  const e = (await r.json()) as Envelope<Methods[M][1]>;
  if (!e.ok) throw new Error(`${e.error.code}: ${e.error.message}`);
  return e.result;
}
