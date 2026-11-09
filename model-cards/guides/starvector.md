# StarVector-1B im2svg

A picture written as SVG code, by a model trained on icons, logos and simple illustrations (flat shapes, few colours).
Photographs and detailed scenes come out as long, approximate paths.

- **image.vectorize** `{image, max_length?, beams?, sample?, top_p?, temperature?, seed?}`: the SVG text (`svg`), always
  complete and well-formed XML. `truncated` says `max_length` (the picture's 257 rows and the prompt counted in; cap 8192,
  StarCoder's positions) stopped the model before it wrote `</svg>`: the SVG is then what it had written, cut back to its
  last whole element or attribute and closed, so it renders - but the picture may be unfinished (raise `max_length`).
  `repaired` says the text was changed to make it so, `closed` that it ends with `</svg>` and parses, `max_length` the
  limit the request was held to (without one it is derived from the picture: 4096 for a plain logo, up to 8192 for a
  busy one).
- A job is stopped with `image.cancel {label}` (or `{reentrant_id}`); a bare id is refused. Dropping the call does not
  stop it (the job goes on). The model stops within a token or two.
- The model's own settings are the defaults: 2 beams sampled with top_p 0.9. `sample: false, beams: 1` is greedy
  (the same SVG every time); a `seed` makes a sampled run repeatable here.
- Transparency is flattened on white first (the model was trained on pictures on white). A square picture with the
  subject filling it works best: anything else is padded with white to a square and seen at 224 x 224.
- A picture it cannot draw can send the model into a loop (the same rule written again and again) that runs to the limit;
  it is closed cleanly but empty of drawing - a simpler picture, or a different `seed`, helps.
- Speed (Arc Pro B70, 2 beams sampled, tg): about 120 tokens a second to 4,700 tokens (it was 7 at that length, a
  quarter hour a run, before the silo's attention and norm).
