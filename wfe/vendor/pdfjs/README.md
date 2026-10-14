# pdf.js (vendored)

Mozilla's pdf.js 4.10.38 (npm `pdfjs-dist`, Apache-2.0: LICENSE), as published, for the OCR page: a PDF chosen there
is rendered in the browser, a page at a time, to the pictures the model reads (the server takes pictures only).

- `pdf.min.js`, `pdf.worker.min.js`: `build/pdf.min.mjs` and `build/pdf.worker.min.mjs` renamed (the servers send
  `.js` as JavaScript; the files are ES modules, loaded with `import()` and as a module worker)
- `standard_fonts/`, `cmaps/`: the data for PDFs that use the 14 standard fonts unembedded, and CJK encodings
- source: https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-4.10.38.tgz (sha256
  1011b38553532d7078c59f26b15a471f8dae00f101b60e2add9b8511737a1ce0)
