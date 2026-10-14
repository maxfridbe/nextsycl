/** The model's output, read: its blocks (`<|det|>label [x1, y1, x2, y2]<|/det|>text`, boxes in thousandths of the
 *  page - the reference draws them at x / 999 of the width), the text without markup, and its HTML tables as a
 *  safe tree (only table tags and colspan / rowspan pass; everything else becomes text). */
import { h } from "../../vendor/snabbdom/h.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";

export interface Block {
  label: string;
  box: [number, number, number, number] | null;
  text: string;
}

const DET = /<\|det\|>([\s\S]*?)<\|\/det\|>/g;

/** The blocks in order; text before any box is a block of its own (no box) */
export function blocks(raw: string): Block[] {
  const out: Block[] = [];
  let last = 0;
  let label = "text";
  let box: Block["box"] = null;
  const push = (t: string) => {
    const text = t.replace(/<\|\/?(ref|grounding)\|>/g, "").trim();
    if (text || box) out.push({ label, box, text });
  };
  for (const m of raw.matchAll(DET)) {
    push(raw.slice(last, m.index));
    const inner = m[1] ?? "";
    const b = inner.match(/^\s*([^[\]]*?)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/);
    label = b?.[1]?.trim() || "text";
    box = b ? [Number(b[2]), Number(b[3]), Number(b[4]), Number(b[5])] : null;
    last = (m.index ?? 0) + m[0].length;
  }
  push(raw.slice(last));
  return out;
}

/** The text alone: boxes, tags and HTML gone, table cells tab-separated, rows on lines */
export function plain(raw: string): string {
  return blocks(raw).map((b) => {
    if (!b.text.includes("<")) return b.text;
    const d = new DOMParser().parseFromString(b.text, "text/html");
    const rows = Array.from(d.querySelectorAll("tr"));
    if (rows.length) return rows.map((r) => Array.from(r.children).map((c) => c.textContent?.trim() ?? "").join("\t")).join("\n");
    return d.body.textContent ?? "";
  }).filter(Boolean).join("\n\n");
}

const KEEP = new Set(["table", "thead", "tbody", "tr", "td", "th", "b", "i", "br", "sup", "sub", "p", "span"]);

function tree(n: Node): (VNode | string)[] {
  if (n.nodeType === Node.TEXT_NODE) return [n.textContent ?? ""];
  if (n.nodeType !== Node.ELEMENT_NODE) return [];
  const el = n as Element;
  const kids = Array.from(el.childNodes).flatMap(tree);
  const tag = el.tagName.toLowerCase();
  if (!KEEP.has(tag)) return kids;
  const attrs: Record<string, string> = {};
  for (const a of ["colspan", "rowspan"]) {
    const v = el.getAttribute(a);
    if (v && /^\d{1,2}$/.test(v)) attrs[a] = v;
  }
  return [h(tag, { attrs }, kids)];
}

/** A block's text as nodes: HTML tables kept (sanitized), headings for titles, paragraphs otherwise */
export function render(b: Block): VNode {
  const lbl = h("span.lbl", {}, b.label);
  if (b.text.includes("<table") || b.text.includes("<tr")) {
    const d = new DOMParser().parseFromString(b.text, "text/html");
    return h("div", {}, [lbl, ...Array.from(d.body.childNodes).flatMap(tree)]);
  }
  if (b.text === "[Non-Text]" || b.label === "image" || b.label === "figure") return h("p.fig", {}, [lbl, b.text || "(a figure)"]);
  const md = b.text.match(/^(#{1,6})\s+(.*)$/s);
  if (md) return h(`h${Math.min(3, md[1]?.length ?? 1)}`, {}, [md[2] ?? ""]);
  if (b.label === "title") return h("h2", {}, [b.text]);
  if (b.label === "sub_title" || b.label === "header") return h("h3", {}, [lbl, b.text]);
  return h("p", {}, [lbl, ...b.text.split("\n").flatMap((l, i) => (i ? [h("br", {}), l] : [l]))]);
}

/** A colour a label (the same label, the same colour) */
export function colour(label: string): string {
  let x = 0;
  for (const c of label) x = (x * 31 + c.charCodeAt(0)) >>> 0;
  return `hsl(${x % 360} 75% 48%)`;
}
