/** What the 3D server answers (glue/serve/src/model3d.rs). */

export interface Abilities { generate: boolean; texture: boolean; views: number; rig: boolean; animate: boolean }

export interface Info {
  model: string;
  arch?: string;
  busy?: boolean;
  abilities?: Abilities;
  defaults?: { steps: number; cfg: number; resolution: number; faces: number; texture_size: number; frames: number; fps: number };
  options?: { name: string; value: string; help: string }[];
  report?: string[];
  saves?: boolean;
}

export interface Progress {
  busy: boolean;
  waiting?: number;
  prompt?: string;
  phase?: string;
  at?: number;
  of?: number;
  seconds?: number;
}

export interface Gpu {
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  power_w: number | null;
  temp_pkg: number | null;
  temp_vram: number | null;
}

/** A mesh made here (its settings from the sidecar JSON) */
export interface Made {
  url: string;
  picture?: string;
  model?: string;
  arch?: string;
  seed?: number;
  steps?: number;
  cfg?: number;
  resolution?: number;
  faces?: number;
  triangles?: number;
  texture?: boolean;
  textured?: boolean;
  rigged?: number | null;
  animated?: number | null;
  prompt?: string | null;
  views?: (string | null)[];
  bytes?: number;
  took?: number;
  wh?: number | null;
  created?: number;
}

export interface Answer { created: number; data: Made[] }

/** A picture chosen for the request */
export interface Pic { name: string; data: string; side: string }

export interface Form {
  pictures: Pic[];
  texture: boolean;
  seed: number | null;
  steps: number | null;
  cfg: number | null;
  resolution: number | null;
  faces: number | null;
  textureSize: number | null;
  removeBackground: boolean;
  /** rig: a mesh file sent (a data: URL), or one made here */
  mesh: string;
  meshName: string;
  file: string;
  prompt: string;
  frames: number | null;
  options: Record<string, string>;
}
