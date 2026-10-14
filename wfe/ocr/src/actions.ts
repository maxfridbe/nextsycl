/** What the buttons do. */
import { api } from "./api.js";
import { promptText, render, state } from "./state.js";
import type { Page } from "./types.js";

/** Pictures added as pages (PNG, JPEG, WebP), in the order given */
export async function addFiles(files: FileList | File[]): Promise<void> {
  const max = state.info?.max_pages ?? 24;
  for (const f of Array.from(files as ArrayLike<File>)) {
    if (/heic|heif/i.test(f.type) || /\.(heic|heif)$/i.test(f.name)) {
      state.error = `${f.name}: HEIC is not read - share it as JPEG (or set the camera to "Most Compatible")`;
      continue;
    }
    if (!f.type.startsWith("image/")) {
      state.error = `${f.name}: a picture is needed (PNG, JPEG, WebP) - export a PDF's pages as pictures`;
      continue;
    }
    if (state.pages.length >= max) {
      state.error = `at most ${max} pages a reading`;
      break;
    }
    const url = await new Promise<string>((ok, no) => {
      const r = new FileReader();
      r.onload = () => ok(String(r.result));
      r.onerror = () => no(r.error);
      r.readAsDataURL(f);
    });
    const size = await new Promise<[number, number]>((ok) => {
      const im = new Image();
      im.onload = () => ok([im.naturalWidth, im.naturalHeight]);
      im.onerror = () => ok([0, 0]);
      im.src = url;
    });
    state.pages.push({ name: f.name, url, width: size[0], height: size[1] });
  }
  render();
}

export function removePage(i: number): void {
  state.pages.splice(i, 1);
  render();
}

export function movePage(i: number, d: number): void {
  const j = i + d;
  if (j < 0 || j >= state.pages.length) return;
  const p = state.pages.splice(i, 1)[0] as Page;
  state.pages.splice(j, 0, p);
  render();
}

export async function refreshHistory(): Promise<void> {
  try {
    state.history = await api.history();
  } catch {
    /* transient */
  }
  render();
}

export async function read(): Promise<void> {
  if (!state.pages.length || state.busy) return;
  state.busy = true;
  state.started = Date.now();
  state.error = null;
  state.result = null;
  state.resultPages = [...state.pages];
  render();
  const f = state.form;
  try {
    state.result = await api.ocr({
      images: state.pages.map((p) => p.url),
      prompt: promptText(),
      tiles: f.tiles,
      max_tokens: f.maxTokens,
      temperature: f.temperature,
      ...(f.guard ? {} : { ngram: 0 }),
    });
  } catch (e) {
    state.error = String(e instanceof Error ? e.message : e);
  }
  state.busy = false;
  state.started = null;
  render();
  void refreshHistory();
}

export function show(r: (typeof state.history)[number]): void {
  state.result = r;
  state.resultPages = [];
  render();
}

export function download(text: string, name: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
