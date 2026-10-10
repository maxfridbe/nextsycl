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
  /** a speech engine: its voices and what it takes (none: songs) */
  speech?: Speech | null;
}

export interface Speech {
  voices: string[];
  languages: string[];
  /** styles the read by instructions */
  instructions: boolean;
  /** makes the voice from instructions alone */
  design: boolean;
  /** clones a recording */
  clone: boolean;
  clone_needs_text: boolean;
  /** a transcript of the recording makes a closer clone */
  clone_takes_text?: boolean;
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
  /** speech: the voice, language, instructions, whether it was cloned */
  voice?: string | null;
  language?: string | null;
  instructions?: string | null;
  cloned?: boolean;
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
  /** speech: the text to say, the voice, the language ("" auto), how to say it / the voice to make up */
  text: string;
  voice: string;
  language: string;
  instructions: string;
  /** speech: the recording to clone (not kept across reloads) and its transcript */
  refAudio: string;
  refName: string;
  refText: string;
}
