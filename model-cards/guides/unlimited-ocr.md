# Reading pages with Unlimited-OCR

Unlimited-OCR reads pictures of pages - scans, photos, screenshots, rendered PDF pages - and writes what they say.
Send the pages as pictures (PNG, JPEG, WebP), in reading order; a PDF's pages are rendered to pictures first
(`pdftoppm -r 150 -png doc.pdf page`). The instruction is short and fixed; the model was trained on these:

| Instruction | What comes back |
|---|---|
| `document parsing.` (the default) | the page as blocks - title, text, table, figure, header, footer - each with its box, tables as HTML |
| `<|grounding|>Convert the document to markdown.` | Markdown with boxes |
| `Free OCR.` | the text alone |
| `Parse the figure.` | a chart or diagram described and its numbers read |
| `Describe this image in detail.` | a description |
| `<|grounding|>Locate <|ref|>Total Due<|/ref|> in the image.` | where a phrase is |

Boxes come as `<|det|>label [x1, y1, x2, y2]<|/det|>` before each block, in thousandths of the page (x / 999 of
its width). One page is also read through its 640-pixel tiles (`tiles`, on by default): finer print, more tokens.
Several pages in one request are read whole, one after another, as one document.

- Pages of 150-200 dpi read best; a phone photo is fine if the page fills it.
- It does not take a question about the page - only the instructions above; read the page, then ask a chat model.
- A dense page can run long: `max_tokens` (default 8192) bounds it; the repetition guard (no 35 tokens repeated
  within the last 128) stops a loop - leave it on.
