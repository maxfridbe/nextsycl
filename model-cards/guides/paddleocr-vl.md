# Reading pages with PaddleOCR-VL

PaddleOCR-VL reads pictures of pages - scans, photos, screenshots, rendered PDF pages - or an element cut from one,
and writes what they say. It sees the picture at its own size (resized to between 0.1 and 1 megapixel, sides in
multiples of 28), so small print on a large page is better cropped. Send pages as pictures (PNG, JPEG, WebP) or
PDFs (split into pages at 150 dpi); several pages are read each on its own, a `<PAGE>` mark before each. The
instruction is fixed; the model was trained on these:

| Instruction | What comes back |
|---|---|
| `OCR:` (the default) | the text, line by line, in reading order (no boxes, no Markdown) |
| `Table Recognition:` | a table as HTML (the model writes OTSL cell tokens; spans kept) |
| `Formula Recognition:` | a formula as LaTeX |
| `Chart Recognition:` | a chart's data as a table |
| `Seal Recognition:` | the text of a seal or stamp, including curved text |
| `Spotting:` | each line with its box, `<|det|>text [x1, y1, x2, y2]<|/det|>` in thousandths of the page (read up to 1.6 megapixels) |

Blocks set side by side (two school years of a transcript, a two-column article) are found first (`columns`, on by
default) and each read on its own, top to bottom - otherwise the model reads straight across them.

- Tables, formulas and charts read best cropped to the element: the model was trained on elements a layout model cut
  from the page. A whole page under `Table Recognition:` comes back as one table.
- It does not take a question about the page - only the instructions above; read the page, then ask a chat model.
- A reading that falls into repeating one short piece over and over is stopped and cut to one copy; `max_tokens`
  (default 4096) bounds the rest.
