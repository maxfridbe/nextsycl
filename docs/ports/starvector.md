# StarVector-1B im2svg (image.vectorize)

CLIP ViT-L/14 (224 x 224, the picture flattened on white) -> adapter -> StarCoder-1B writing the SVG after `<svg`, 2 beams
sampled as the starvector package does. Engine: `image/starvector`, silo: `kernels/image/starvector/silo`
(`dist/silo/libnextsycl-starvector.so`; `NS_SILO=0` runs the shared kernels). Card: `model-cards/image/starvector-1b-im2svg.yml`.

## 2026-11-08: a quarter hour a logo, a cut-off SVG, a cancel that did nothing

The platform's report: a 1024 x 1024 logo with transparency took 689.8 s on the B70, answered `"closed": false`, and a client
cancel at 4 m 50 s did not stop the job.

### Speed

Reproduced exactly with a busy flat-colour logo (36 coloured wedges, a white disc, text; PIL-made, 1024 x 1024 RGBA):
`image vectorize --max-length 5000 --seed 1` wrote 4,741 tokens (the limit) in **689.2 s, 6.9 tg tok/s**, 0.7 s of it for the
picture and prompt. The model had fallen into a loop (`.cls-N{fill:#...}` rules written to the end of the budget), so any
long run is this case. The same input now:

| B70, 2 beams sampled, seed 1, max_length 5000 | tokens | time | tg |
|---|---|---|---|
| before (shared kernels) | 4,741 | 689.2 s | 6.9 tok/s |
| silo attention only | 4,741 | 96.8 s | 49 tok/s |
| silo attention + layer norm | 4,741 | **38.9 s** | **122 tok/s** |
| a flat logo that ends on its own (913 tokens), before / after | 913 | 48.1 s / 6.8 s | 18.9 / 135 tok/s |

Greedy (`--greedy`, 1,241 tokens) is **token-for-token identical** with and without the silo (`NS_SILO=0`: 25.9 tok/s; silo:
151 tok/s). Sampled runs are not bit-identical to before (the kernels sum in another order, so a draw can fall the other way
at a near-tie): same model, same settings, a different sample from the same seed.

Where the time went (decode step timed op by op with the queue drained after each, 1,241 greedy tokens, 24 ms a token then):

- **Layer norm, 13 ms of 18 ms a token.** The shared `nsd_layer_norm` sums a row in a single work-item (two serial passes over
  2048 values: 270 us a call), 48 calls a token. The silo's is a work-group a row with group reductions.
- **Decode attention**, growing with the position: the shared `nsd_attention_causal_rows` walks the keys one at a time in one
  work-group a head with a group-wide reduction a key (about 1 us a key, 48 launches a token: 0.12 s a token at 3,000 keys -
  the 6.9 tok/s). The silo's splits the keys in chunks of 128 across work-groups (flash-decoding: a partial softmax each, a
  combine pass) and does both beams in one launch.
- The rest, 6.6 ms a token: the linears (fc/proj about 245 GB/s, the model's 2.3 GB of weights are read once a token), the
  head's 49,156-row product, a 400 KB logit read-back. The host is not the limit: submitting a token's 400 launches takes
  1 ms, the beam warpers 0.6 ms, the SVG text is decoded every 64 tokens (it was every token: O(n^2) host work).
- Not a factor: the encoder's input is fixed at 224 x 224 by the model (a 1024 picture is resized, nothing to save), the KV
  cache is per beam in half and copied only when a beam is taken over, the weights are half (the checkpoint's float32 halved
  at load).

Left: the linears at 2 rows (oneDNN reading the weight transposed) and a fused norm + residual; each worth a few ms a token.

### Output

A finished request returns a complete, well-formed SVG (`close.rs`, checked with roxmltree - a small crate with no
dependencies - on every answer):

- the length: a request's `max_length` (the picture's 257 rows and the prompt counted) up to **8192**, StarCoder's
  positions (the front door clamps to it); without one it is derived from the picture, `prefix + edge pixels` of the 224 x 224
  input (pixel pairs differing by more than 24 grey levels; a flat logo of 4,677 of them wrote 913 tokens) between **4096 and
  8192** (it was 5000);
- when the limit stops the model: the text is cut back to the last whole tag, the partial token dropped; inside a tag the
  attributes that were whole stay and the cut one goes - except path `d` and `points` (cut back to the last whole command or
  coordinate set) and `style` (to its last whole declaration); a cut `<style>` sheet keeps its whole rules; every element still
  open is closed in order, `</svg>` last; a missing `xmlns` is added; text that still does not parse (a stray end tag, an
  undeclared prefix) is repaired once more and failing that cut back to an empty `<svg>` with the root's size;
- the answer: `data[0]` keeps `svg`, `closed`, `tokens` and adds `truncated` (the limit stopped the model), `repaired` (the text
  was changed) and `max_length` (the limit used). **`closed` now means the returned `svg` is complete** (ends with `</svg>`,
  parses): true even when `truncated` is; read `truncated` for "the picture may be unfinished". The history line says
  "cut at max_length";
- a closed text that parses is returned untouched (`a_closed_svg_is_untouched`).

The looping case above comes back as a valid SVG with the style rules it wrote and no drawing. A loop guard (stop on a rule
repeated many times) would save the remaining time but changes what a run writes; not done.

### Cancel

`image.vectorize` is a front-door job: the call can drop and the job goes on (a client timeout is not a cancel - that is what
the platform saw). To stop it:

```
POST /rpc/image.cancel   {"label": "<the label the job was submitted with>"}      (or {"reentrant_id": "<its reentrant_id>"})
```

The label is the one sent with `image.vectorize` (the job's `id`, `image-<unix>-<n>`, when none was sent - then the label *is*
that id). The front door marks the job, tells the image server (`/api/cancel {id: <job id>}`, the id it sent as `request_id`),
and the decode loop - which checks the flag **every token** - returns 499 `cancelled` within a token (about 8 ms now). The answer:
`{cancelled: <jobs>, stopped: <running requests stopped>, already_ended, jobs: [...]}`; a job that ended already answers
`already_ended: true` with its state; no such label is 404. `image.status {label}` then says `cancelled`.

An **id alone is refused** (400 `invalid_params`, `image.cancel names its job by label or reentrant_id`): a body with no label and
no reentrant_id stops *whatever the kind's servers are making* (the pages' stop button), so a client sending only an id would stop
other people's jobs. Refused for `image`, `audio` and `model3d` cancels (`jobs::bare_id`, tested). A body with nothing at all keeps
its meaning (stop everything), as the pages use it.

Tested through a build copy's front door (own registry, ports 18000+, B70 only, `NS_GPU_LOCK` of its own): a looping
`image.vectorize` (max_length 8192, `wait: false`, label `svtest-cancel-1`) running at 800 tokens: `image.cancel {id}` refused
and the job kept running; `image.cancel {label}` answered `{cancelled: 1, stopped: 1}`, the job was `cancelled` and `system.busy`
false **0.5 s** later (the model stays loaded, idle: 3 GB on the card until its idle unload); the next `image.vectorize` through the
same door ran at once (911 tokens in 6.8 s, `closed: true, truncated: false`), and a `max_length: 700` one came back
`closed: true, truncated: true, repaired: true`, 441 tokens, ending `...</g></g></svg>`. A job still *waiting* is cancelled at once
(`stopped: 0`).
