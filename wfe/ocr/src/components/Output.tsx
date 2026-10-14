/** The reading: as a document (headings, paragraphs, tables), the page with its boxes, the model's raw text, or the
 *  plain text; while it is written, the text so far. */
import { jsx } from "../jsx.js";
import { download } from "../actions.js";
import { blocks, colour, pages, plain, render as block } from "../ocr.js";
import { render, saveForm, state, type View } from "../state.js";
import { fmtT } from "../util.js";

const VIEWS: [View, string][] = [["doc", "Document"], ["boxes", "Boxes"], ["raw", "Markdown"], ["text", "Text"]];

function Boxes(raw: string) {
  const all = pages(raw);
  const n = Math.min(all.length, state.resultPages.length);
  if (!n) return <div class="hint">The boxes are drawn over the pages read here (a reading from the history has no pictures).</div>;
  const at = Math.min(state.boxPage, n - 1);
  const page = state.resultPages[at];
  const text = all[at] ?? "";
  if (!page) return null;
  return (
    <div>
      {n > 1
        ? <div class="tabs">{Array.from({ length: n }, (_, i) => (
            <button type="button" class={{ tab: true, on: i === at }} on={{ click: () => { state.boxPage = i; render(); } }}>{`page ${i + 1}`}</button>
          ))}</div>
        : null}
      <div class="boxes">
        <img attrs={{ src: page.url, alt: page.name }} />
        {blocks(text).filter((b) => b.box).map((b) => {
          const [x1, y1, x2, y2] = b.box as [number, number, number, number];
          const c = colour(b.label);
          return (
            <div class="bx" attrs={{ title: `${b.label}: ${b.text.replace(/<[^>]+>/g, " ").slice(0, 300)}` }}
                 style={{ left: `${(x1 / 999) * 100}%`, top: `${(y1 / 999) * 100}%`, width: `${((x2 - x1) / 999) * 100}%`,
                          height: `${((y2 - y1) / 999) * 100}%`, borderColor: c }}>
              <span style={{ background: c }}>{b.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The document: page by page when there are several */
function Doc(raw: string) {
  const all = pages(raw);
  if (all.length === 1) return <div class="doc">{blocks(raw).map(block)}</div>;
  return (
    <div class="doc">
      {all.flatMap((p, i) => [<div class="pagehead">{`Page ${i + 1}`}</div>, ...blocks(p).map(block)])}
    </div>
  );
}

export function Output() {
  const r = state.result;
  const p = state.progress;
  if (!r) {
    const so = p?.reading ? p.text ?? "" : "";
    return (
      <div>
        <div class="live">{p?.reading ? `${p.tokens ?? 0} tokens · ${fmtT(p.seconds)} · ${p.seconds ? ((p.tokens ?? 0) / p.seconds).toFixed(0) : "—"} tok/s` : "seeing the pages…"}</div>
        <pre class="raw">{so}<span class="cursor" /></pre>
      </div>
    );
  }
  const raw = r.text;
  const tools = (
    <div class="row">
      <div class="tabs">
        {VIEWS.map(([v, t]) => (
          <button type="button" class={{ tab: true, on: state.view === v }} on={{ click: () => { state.view = v; saveForm(); render(); } }}>{t}</button>
        ))}
      </div>
      <span class="hint">{`${r.pages} page${r.pages === 1 ? "" : "s"} · ${r.prompt_tokens ?? "?"} read${r.read_tok_s ? ` (${r.read_tok_s.toFixed(0)} tok/s)` : ""} · ${r.tokens} written in ${fmtT(r.seconds)}${r.tok_s ? ` (${r.tok_s.toFixed(0)} tok/s)` : ""}`}</span>
      <button type="button" class="tbtn" on={{ click: () => void navigator.clipboard?.writeText(state.view === "raw" ? raw : plain(raw)) }}>copy</button>
      <button type="button" class="tbtn" on={{ click: () => download(raw, "reading.md") }}>.md</button>
      <button type="button" class="tbtn" on={{ click: () => download(plain(raw), "reading.txt") }}>.txt</button>
    </div>
  );
  const body =
    state.view === "raw" ? <pre class="raw">{raw}</pre>
    : state.view === "text" ? <pre class="raw">{plain(raw)}</pre>
    : state.view === "boxes" ? Boxes(raw)
    : Doc(raw);
  const empty = !raw.trim()
    ? <pre class="err">{`The model wrote nothing for this (it ended at once). ${r.pages > 1 ? "" : "A small picture or one that is not a page can do this; try \"Free OCR.\" or \"Describe this image in detail.\", or a larger, sharper picture."}`}</pre>
    : null;
  return <div>{tools}{empty}{body}</div>;
}
