# StarVector-1B im2svg

A picture written as SVG code, by a model trained on icons, logos and simple illustrations (flat shapes, few colours).
Photographs and detailed scenes come out as long, approximate paths.

- **image.vectorize** `{image, max_length?, beams?, sample?, top_p?, temperature?, seed?}`: the SVG text (`svg`), and
  `closed` - whether `</svg>` ended it. At `max_length` (default 5000, the picture's 257 rows and the prompt counted
  in) it stops wherever it is, often inside a path: close or cut it before rendering (keep the finished elements).
- The model's own settings are the defaults: 2 beams sampled with top_p 0.9. `sample: false, beams: 1` is greedy
  (the same SVG every time); a `seed` makes a sampled run repeatable here.
- Transparency is flattened on white first (the model was trained on pictures on white). A square picture with the
  subject filling it works best: anything else is padded with white to a square and seen at 224 x 224.
- Longer `max_length` gives detailed pictures room to finish: about 125 tokens a second with 2 beams on an RTX 5070.
