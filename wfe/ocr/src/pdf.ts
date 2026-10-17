/** A PDF's pages as pictures, rendered here with the vendored pdf.js (loaded on first use): the server reads pictures
 *  only. 150 dpi (a letter page: 1275 x 1650), as the reference tests render them. */
import type { Page } from "./types.js";

// absolute, from the page's address: import() resolves against this module's own URL, and the page may be served
// under the front door's /open/<id>/
const BASE = new URL("ui/vendor/pdfjs/", document.baseURI).href;
export const DPI = 150;

interface Viewport { width: number; height: number }
interface PdfPage {
  getViewport(o: { scale: number }): Viewport;
  render(o: { canvasContext: CanvasRenderingContext2D; viewport: Viewport }): { promise: Promise<void> };
  cleanup(): void;
}
interface PdfDoc { numPages: number; getPage(n: number): Promise<PdfPage>; destroy(): Promise<void> }
interface PdfLib {
  GlobalWorkerOptions: { workerSrc: string };
  getDocument(o: Record<string, unknown>): { promise: Promise<PdfDoc> };
}

let lib: Promise<PdfLib> | null = null;

function load(): Promise<PdfLib> {
  if (!lib) {
    const url = `${BASE}pdf.min.js`;
    lib = (import(url) as Promise<PdfLib>).then((m) => {
      m.GlobalWorkerOptions.workerSrc = `${BASE}pdf.worker.min.js`;
      return m;
    });
  }
  return lib;
}

export const isPdf = (f: File): boolean => f.type === "application/pdf" || /\.pdf$/i.test(f.name);

/** Up to `max` of the PDF's pages as PNG pictures, in order; `progress` hears each page done */
export async function pdfPages(f: File, max: number, progress: (done: number, of: number) => void): Promise<{ pages: Page[]; total: number }> {
  const m = await load();
  const doc = await m.getDocument({ data: new Uint8Array(await f.arrayBuffer()), cMapUrl: `${BASE}cmaps/`, cMapPacked: true,
                                    standardFontDataUrl: `${BASE}standard_fonts/` }).promise;
  const n = Math.min(doc.numPages, max);
  const pages: Page[] = [];
  const stem = f.name.replace(/\.pdf$/i, "");
  for (let i = 1; i <= n; i++) {
    const p = await doc.getPage(i);
    const vp = p.getViewport({ scale: DPI / 72 });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(vp.width);
    canvas.height = Math.round(vp.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2D canvas here");
    // a white sheet first: a PDF page without a background would come out transparent
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    await p.render({ canvasContext: ctx, viewport: vp }).promise;
    pages.push({ name: `${stem} p${i}`, url: canvas.toDataURL("image/png"), width: canvas.width, height: canvas.height });
    p.cleanup();
    progress(i, n);
  }
  const total = doc.numPages;
  await doc.destroy();
  return { pages, total };
}
