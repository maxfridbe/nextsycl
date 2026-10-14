/** What the server sends (glue/serve/src/image.rs) and what the page keeps. */

export interface Info {
  model: string;
  arch?: string;
  reads?: boolean;
  max_pages?: number;
  loading?: boolean;
}

export interface Progress {
  busy: boolean;
  reading?: boolean;
  pages?: number;
  text?: string;
  tokens?: number;
  seconds?: number;
  waiting?: number;
}

export interface Gpu {
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  power_w?: number | null;
  temp_pkg?: number | null;
  temp_vram?: number | null;
}

/** A reading: what /v1/ocr answers (and /api/history keeps) */
export interface Reading {
  text: string;
  prompt?: string;
  pages: number;
  tokens: number;
  prompt_tokens?: number;
  seconds: number;
  tok_s?: number;
  read_tok_s?: number;
  wh?: number | null;
  created?: number;
  kind?: string;
}

/** A page given: its name and data URL (kept for the boxes view) */
export interface Page {
  name: string;
  url: string;
  width: number;
  height: number;
}

export interface Form {
  preset: string;
  prompt: string;
  tiles: boolean;
  maxTokens: number;
  temperature: number;
  guard: boolean;
  /** read the text inside the pictures the model marks */
  figures: boolean;
  /** several pages: each read on its own (tiles, finer print) rather than as one document (the model's multi-page
   *  reading, each page seen at 1024) */
  perPage: boolean;
  /** find blocks set side by side and read each on its own */
  columns: boolean;
}
