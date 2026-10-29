/** What the buttons do. */
import { api } from "./api.js";
import { isPdf, pdfPages } from "./pdf.js";
import { promptText, render, state } from "./state.js";
import type { Page, Reading } from "./types.js";

/** Pictures added as pages (PNG, JPEG, WebP), in the order given */
export async function addFiles(files: FileList | File[]): Promise<void> {
  const max = state.info?.max_pages ?? 24;
  for (const f of Array.from(files as ArrayLike<File>)) {
    if (isPdf(f)) {
      // a PDF: its pages rendered here, as many as fit
      const room = max - state.pages.length;
      if (room <= 0) {
        state.error = `at most ${max} pages a reading`;
        break;
      }
      try {
        state.note = `opening ${f.name}…`;
        render();
        const { pages, total } = await pdfPages(f, room, (done, of) => {
          state.note = `rendering ${f.name}: page ${done} of ${of}`;
          render();
        });
        state.pages.push(...pages);
        if (total > pages.length) state.error = `${f.name}: ${total} pages - the first ${pages.length} taken (at most ${max} a reading)`;
      } catch (e) {
        state.error = `${f.name}: not read as a PDF (${String(e instanceof Error ? e.message : e)})`;
      }
      state.note = null;
      render();
      continue;
    }
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
  state.historyLoaded = true;
  render();
}

export async function read(): Promise<void> {
  if (!state.pages.length || state.busy) return;
  state.busy = true;
  state.started = Date.now();
  state.error = null;
  state.result = null;
  state.boxPage = 0;
  state.partial = "";
  state.pageAt = 0;
  state.resultPages = [...state.pages];
  render();
  const f = state.form;
  const body = (images: string[]) => ({
    images,
    prompt: promptText(),
    tiles: f.tiles,
    figures: f.figures !== false,
    columns: f.columns !== false,
    max_tokens: f.maxTokens,
    temperature: f.temperature,
    ...(f.guard ? {} : { ngram: 0 }),
  });
  try {
    if (state.pages.length > 1 && f.perPage !== false) {
      // page by page: each read on its own (its tiles too), the readings joined with <PAGE> marks
      const all: Reading = { text: "", pages: state.pages.length, tokens: 0, prompt_tokens: 0, seconds: 0 };
      for (const [i, p] of state.pages.entries()) {
        state.pageAt = i + 1;
        render();
        const r = await api.ocr(body([p.url]));
        all.text += `<PAGE>${r.text}\n`;
        all.tokens += r.tokens;
        all.prompt_tokens = (all.prompt_tokens ?? 0) + (r.prompt_tokens ?? 0);
        all.seconds += r.seconds;
        all.wh = (all.wh ?? 0) + (r.wh ?? 0);
        state.partial = all.text;
      }
      all.tok_s = all.tokens / Math.max(all.seconds, 1e-9);
      state.result = all;
    } else {
      state.result = await api.ocr({ ...body(state.pages.map((p) => p.url)), each_page: false });
    }
  } catch (e) {
    state.error = String(e instanceof Error ? e.message : e);
  }
  state.busy = false;
  state.started = null;
  state.pageAt = 0;
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
