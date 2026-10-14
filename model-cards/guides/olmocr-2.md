# Reading pages with olmOCR-2

olmOCR-2 reads pictures of whole pages - scans, photos, screenshots, rendered PDF pages - into Markdown in natural
reading order: headings and paragraphs, tables as HTML (spans kept), equations as LaTeX, figures labelled
`![what it shows](page_x_y_w_h.png)`. Send pages as pictures (PNG, JPEG, WebP) or PDFs (split into pages at 150 dpi);
each page is fitted to 1288 pixels on its longest side, as olmOCR renders its pages. Several pages are read each on
its own, a `<PAGE>` mark before each.

The model was trained on one instruction (sent by default; leave the prompt empty). It writes a short front matter
first - the page's language, whether it is upright, tables, diagrams - which is dropped from the text; a page the
model says is turned is turned as it says and read again.

- Blocks set side by side (two school years of a transcript) it would read across as one wide table: `columns` (on
  by default) finds the gutter between them first and reads above, each side and below on its own. A flowing
  two-column article it reads in order either way.
- It does not take a question about the page; read the page, then ask a chat model.
- A page of ~2000 words writes ~3000 tokens: about a minute on a B70 (~58 tokens a second); `max_tokens` (default
  8000, olmOCR's own) bounds it, and a reading that loops on one short piece is stopped.
