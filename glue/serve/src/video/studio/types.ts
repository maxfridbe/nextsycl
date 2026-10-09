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
export type Engine = "Q4_K_M" | "Q6_K" | "Q8_0";
export type ClipState = "done" | "running" | "queued" | "held";

export interface GenerateRequest {
  prompt: string;
  seconds?: number;          // 1-15.1; the model is trained on 124-362 frames at 24 fps
  steps?: number;            // 1-40; below 8 only with a turbo LoRA
  seed?: number;
  te?: "teacher" | "student";
  label?: string;            // "Name NN/MM: ..." - the NN drives edit order
  width?: number; height?: number;
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
  ref_audios?: string[];     // voices to speak in: <Audio 1>, <Audio 2> ... (files in out/)
  guide_clip?: string | null;  // motion guide: "<anchor or file>[:frames[:at]]" - the last frames of a clip as one moving keyframe
  shift_video?: number; shift_audio?: number;  // flow shifts (12, 3)
  source?: Anchor | null;    // masked run: a clip whose latents are kept outside the regenerated part
  regen?: string | null;     // "from-to" seconds to regenerate; a source shorter than the clip is extended
  regen_box?: string | null; // "x0,y0,x1,y1" pixels: regenerate only inside
  project?: string; batch?: string;
  queue?: boolean;
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
  job?: Job;
  stage?: string; pct?: number; done?: boolean; error?: string | null;
  eta?: number; elapsed?: number;
  queue: string[];
  queue_items: QueueItem[];
  projects: Project[];
  all_projects: string[];
  version?: string;
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
  "wait": [{ version?: string; timeout_s?: number }, Status & { changed: boolean }];
  "generate": [GenerateRequest, { queued: boolean; position?: number; job?: Job }];
  "cancel": [Record<string, never>, { stopped: true }];
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
