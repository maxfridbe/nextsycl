# image.describe: a picture to the prompt that would draw it

[← nextsycl](../../README.md) · [images](../images.md) · [models](../models.md)

`POST /rpc/image.describe` takes a picture and answers the text prompt that would make one of our image models draw
it - written the way the image model it is for reads prompts (Midjourney calls the same thing `/describe`). This page
is the whole contract: enough to wire it into a UI, or to implement the same method on another runtime (nextcuda)
without reading nextsycl's code.

## Design

- **The describer is a chat model that reads pictures, already served**: Qwen3.8-27B with its vision tower (Qwen3-VL's
  ViT, the `mmproj` file next to the model; the `qwen35` engine). No new engine, no new download. The H3 text encoder
  (Qwen3-VL 32B in `crates/h3` / `crates/qwen3vl`) was the other candidate: it is an *encoder* there - it returns hidden
  states, it has no decode loop, KV cache or sampler - and giving it one would mean a second LLM runtime inside the image
  kind. The OCR models (olmOCR-2, a Qwen2.5-VL-7B fine-tune; Unlimited-OCR; PaddleOCR-VL) are trained to transcribe
  pages, not to caption scenes.
- **A model card, not code, sizes it**: `qwen3.8-27b-q4-describe` (model-cards/llm) is `qwen3.8-27b-q4`'s own files
  (nothing new downloaded) with a 16K + 8K context instead of 128K + 32K, so it takes 19.2 GiB of a card (measured, B70) instead of 29 and
  fits beside an image model or another card's work. Any chat model whose card carries a `vision` file (or a `sidecars`
  pattern naming a vision tower) can describe; `system.models` lists `vision` and `describe` in its capabilities and
  `image.describe` in its methods.
- **One instruction per image-model family** (`target`): each family's guide (model-cards/guides) says how its model
  reads prompts, and the instruction asks for exactly that (table below). The answer is cut at `PROMPT:` /
  `NEGATIVE:`.
- **No thinking**: the chat server's request field `think_budget: 0` (new, both `/api/chat` and
  `/v1/chat/completions`) closes Qwen's thinking before it starts (`<think>\n\n</think>\n\n`, its non-thinking form).
  A describe is ~100 tokens; the thinking would be several hundred.
- **Name**: `image.describe` (not `image.prompt`): the picture is the input and a description the output, as
  Midjourney's `/describe`; "prompt" as a verb reads as "prompt the image model". Not to be confused with
  `image.describePromptGuideMD` (how to prompt a model, as Markdown).
- **A job in the image queue**, like `image.upscale` / `image.remove_background`: `label`, `reentrant_id`, `wait`,
  `image.status`, `image.cancel`, priority and holds - the front door's job machinery as it is, so cancel-by-label works
  and a bare id is refused. It runs on the chat side (the switcher): the job waits in the image queue for an image
  slot like the others, then asks the chat switcher, which loads the describer (room made first, idle models stopped,
  pinned or held ones never; `no_room` otherwise, as llm.chat) or uses it when loaded.

## Request

`POST /rpc/image.describe`, `Authorization: Bearer <token>`, a JSON object:

| field | type | default | what |
|---|---|---|---|
| `image` | string, **required** | | the picture: PNG, JPEG or WebP as base64, a `data:` URL, or `upload:<name>` (from `image.upload`); at most 40 MP. Transparency is put on white |
| `target` | string | `natural` | a family (`natural`, `qwen-image`, `z-image`, `h3`, `hunyuanimage`, `moss-image`, `ming-image`, `tags`), or an image model's id (`z-image-turbo`, `minimax-h3-image`, `p-h3-image-q8`, `qwen-image-2.1-turbo-q8` ...: its family, by its architecture when registered here, else by its id's prefix). Anything else: 400 |
| `detail` | `short` \| `medium` \| `long` | `medium` | the length aimed for: ~35 / ~90 / ~170 words (tags: 12 / 25 / 45) |
| `max_words` | integer 8-400 | | the words (tags) aimed for, in place of `detail`'s |
| `include_negative` | boolean | false | also a negative prompt - only for targets whose models read one (`natural`, `qwen-image`, `hunyuanimage`, `moss-image`, `tags`); for the others none is written and `notes` says so |
| `language` | string, ≤ 40 chars | `English` | the prompt's language, by name |
| `instructions` | string, ≤ 2,000 chars | | more for the describer ("focus on the lighting", "the person is a child") |
| `max_pixels` | integer 65,536-4,194,304 | 1,048,576 | the picture is shown at most this large (Lanczos); 1 MP = ~1,024 vision positions, 4 MP = 4,096 (slower: see Speed) |
| `max_tokens` | integer 16-4,096 | 2 x words + 40 (+80 with a negative) | the answer's tokens at most |
| `temperature` | number 0-2 | 0 | 0: greedy - the same picture gives the same prompt; above 0 for variety (`seed` then fixes it) |
| `seed` | integer ≥ 0 | | the describer's sampling seed |
| `model` | string | see below | the describer: a chat model that reads pictures. Default: the loaded chat model when it reads pictures (no swap), else the enabled one whose card takes the least memory (the `-describe` card before the long-context ones) |
| `label`, `title`, `reentrant_id`, `wait`, `priority`, `held`, `queue`, `project`, `batch`, `rsvp_s`, `max_willing_to_wait_s` | | | as every image job (`image.generate`): `wait: false` answers the job at once, `image.status {label}` later |

## Answer

`{"ok": true, "result": {...}}` (HTTP 200), `result`:

```json
{
  "prompt": "A flat vector illustration of a simple red barn with a dark brown triangular roof and a small white rectangular door in the center, standing on a solid green field that occupies the lower third of the frame. ...",
  "negative_prompt": null,
  "target": "z-image",
  "target_model": "z-image-turbo",
  "model": "qwen3.8-27b-q4-describe",
  "detail": "medium",
  "language": "English",
  "words": 94,
  "width": 1024, "height": 768,
  "seen": {"width": 1024, "height": 768},
  "suggested_size": {"width": 1184, "height": 880, "size": "1184x880"},
  "tokens": {"prompt": 1070, "reused": 0, "generated": 109},
  "timings": {"pp_tok_s": 797.4, "tg_tok_s": 59.8, "prompt_seconds": 1.342, "generate_seconds": 1.823, "seconds": 3.184},
  "finish": "stop",
  "energy_wh": 0.21,
  "notes": ["z-image takes no negative prompt (no guidance): none written"],
  "job": {"id": "image-1794470000-3", "label": "barn-prompt", "state": "done", "...": "image.status's view"}
}
```

- `prompt` - feed it to `image.generate` with `size: suggested_size.size` and the target model.
- `negative_prompt` - a string only with `include_negative` and a target that reads one; else `null`.
- `target` - the family written for; `target_model` - the model the request named (else `null`).
- `seen` - the size the describer saw (after `max_pixels`); `width` / `height` - the picture as sent.
- `suggested_size` - the target's best area at the picture's aspect (clamped 1:4 .. 4:1), sides on its grid:
  1 MP / multiples of 16 for natural, qwen-image, z-image (256-2048), moss-image, ming-image, tags; 4 MP / multiples
  of 32 for h3 (256-5024) and hunyuanimage (768-3072).
- `tokens.prompt` - the describer's whole prompt (instruction + the picture's vision positions + the request);
  `tokens.reused` - from its prompt cache (the instruction after the first describe for a family: the system turn is the
  same for the same settings).
- `timings` - `pp_tok_s` prompt processing including the vision tower, `tg_tok_s` generation, `seconds` the whole
  describe (a swap to the describer, when it was not loaded, is in it).
- `finish` - `stop`, or `length` when `max_tokens` cut the answer (a note says so). Qwen sometimes goes on after a whole prompt (writes `PROMPT:` and the prompt again until the cap): the prompt is cut where it first ended and `finish` is `stop`, with a note that the describer kept writing.
- `rsvp_granted_s` when the request asked for `rsvp_s`.

`wait: false` answers the job (`{"id", "label", "state": "queued", ...}`); `image.status {"label"}` gives it with the
`result` above once `state` is `done`.

## Errors

The envelope `{"ok": false, "error": {"code", "message", "retryable", "status", "method", "field"}}`, HTTP 200:

| code (status) | when | retryable |
|---|---|---|
| `invalid_params` (400) | no `image`; a picture that does not decode or is over 40 MP; a `target` that is neither a family nor an image model that draws ("target ocr-model: it does not draw pictures from prompts"); a value out of its range (`max_words 3: from 8 to 400`); a `model` that does not read pictures | no |
| `not_found` (404) | `model` names no model; `image: upload:<name>` not here | no |
| `conflict` (409) | no chat model that reads pictures is enabled (`system.enable qwen3.8-27b-q4-describe`); the `model` named is disabled; the chat switcher refuses to swap away from a chat model in use ("... is loaded and in use; switching ... would cut that conversation off") | no (the last: later) |
| `model_not_downloaded` (409) | `model` is a card's that is not downloaded (with `NS_AUTO_PULL` it is fetched instead) | no |
| `busy` (409) | `queue: false` and the job would not start now | yes |
| `cancelled` (499) | `image.cancel {label}` / `{reentrant_id}` stopped it | no |
| `unavailable` (502) | the chat server did not answer, or its stream ended early | yes |
| `no_room` (503) | the describer's card has no room (work that is not idle, a pin or a hold there) | yes |
| `wait_too_long` (503) | `max_willing_to_wait_s` would be passed | yes |
| `maintenance` (503) | the box is in maintenance | yes |
| `timeout` (504) | `wait: true` and still running after an hour (the job goes on), or the describe itself over 30 minutes | yes |

## Cancel

`POST /rpc/image.cancel {"label": "barn-prompt"}` (or `{"reentrant_id": "..."}`): a queued describe ends at once
(`cancelled before it started`); a running one is stopped within a second - the front door closes its stream to the chat
server, which stops at its next token (while the picture is still being read, at the end of that reading: ~1-10 s). A
cancel naming only `id` / `job_id` / `request_id` is refused (400): without a label it would stop whatever the image
servers run. A cancel of a job that has ended answers `already_ended: true` with its state.

## Prompt styles by family

| `target` | models (ids) | how the prompt is written | negative |
|---|---|---|---|
| `natural` | any | one paragraph of plain, concrete sentences, the important things first | yes |
| `qwen-image` | qwen-image-2.1-* | as told to a painter: subject and look, setting, light, camera (lens, angle, distance), style; one long paragraph; text in double quotes ("a shop sign that reads \"OPEN LATE\"") | yes (cfg above 1, not the turbo) |
| `z-image` | z-image-turbo, -int8 | sentences, never tags; names the kind of picture ("a close-up product photo"); subject, what it wears/holds, setting, light; each text in quotes with its place and lettering; only what is there | no (distilled, no guidance) |
| `h3` | p-h3-image*, minimax-h3-image* | opens with one sentence naming the real object the picture is ("a photograph by an experienced street photographer", "a flat vector illustration"); one scene: viewpoint, subject, a few supporting elements, light with source and direction; every word of text quoted with its place, then "no other text"; no hype words (stunning, epic, cinematic, 8k, masterpiece) | no |
| `hunyuanimage` | hunyuanimage-2.1-* | as qwen-image; quoted text spelled exactly (its glyph encoder reads quotes) | yes (the base model; the distilled one ignores it) |
| `moss-image` | moss-image-sft-* | natural sentences (it takes tags and a JSON layout too, sentences are its everyday way) | yes (best left empty, its guide says) |
| `ming-image` | ming-image-0.1-design | a design: the layout first (header, columns, cards, a centred title), then each block's exact words in quotes, colours, lettering, position | no |
| `tags` | tag-style models | comma-separated tags, most important first: kind of picture, subject and count, attributes, colours, setting, style, lighting, framing | yes |

Every instruction also says: describe only what is visible; the kind of picture; subjects with colours, shapes,
materials, counts and places in the frame; background; light; viewpoint and framing; palette; copy text exactly in
double quotes; about N words; the language; and the answer's shape (`PROMPT: ...` and, with a negative,
`NEGATIVE: ...`). The exact text is `glue/serve/src/describe.rs` (`Family::style`, `Ask::instruction`). A
reimplementation should keep the families' names and the answer's fields; the wording may differ.

## OpenAI-compatible route

There is no separate `/v1` describe route. What exists already and works: `POST /v1/chat/completions` naming a chat
model that reads pictures, with the picture as an `image_url` content part (a `data:` URL; remote URLs are refused),
your own instruction as text - and `"think_budget": 0` to skip the thinking. That is the raw call `image.describe`
makes (through `/api/chat`, streamed); it returns OpenAI's `chat.completion` and you cut the prompt out yourself.

## Examples (curl)

```sh
T=<token>; D=http://box:8000
# a picture, for Z-Image-Turbo
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d "{\"image\": \"data:image/png;base64,$(base64 -w0 barn.png)\", \"target\": \"z-image-turbo\", \"label\": \"barn-prompt\"}"
# upload once, describe for two families, a negative where it is read
curl -s $D/rpc/image.upload -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d "{\"name\": \"barn.png\", \"data\": \"$(base64 -w0 barn.png)\"}"            # -> {"ref": "upload:up_1794470000_barn.png", ...}
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "qwen-image", "include_negative": true, "detail": "long"}'
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "tags", "detail": "short"}'
# as a job: answer at once, poll, cancel
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "h3", "wait": false, "label": "barn-h3"}'
curl -s $D/rpc/image.status -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d '{"label": "barn-h3"}'
curl -s $D/rpc/image.cancel -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d '{"label": "barn-h3"}'
# and back: draw the prompt
curl -s $D/rpc/image.generate -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"model": "z-image-turbo", "prompt": "<result.prompt>", "size": "<result.suggested_size.size>", "seed": 7}'
# the raw OpenAI-style call (your own instruction)
curl -s $D/v1/chat/completions -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d "{\"model\": \"qwen3.8-27b-q4-describe\",
  \"think_budget\": 0, \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$(base64 -w0 barn.png)\"}},
  {\"type\": \"text\", \"text\": \"Write a text-to-image prompt for this picture.\"}]}]}"
```

The command line does the same against a chat server: `nextsycl image describe barn.png --target z-image [--negative]
[--detail long] [--json] [--server HOST:PORT] [--model ID]` (default: the box's switcher).

## Measured (hardac1, 2026-11-10)

All through `nextsycl image describe --json` (the same `describe::describe` the front door runs) from a build copy,
against that copy's `nextsycl llm serve` of Qwen3.8-27B UD-Q4_K_M + mmproj-F16 + its MTP drafts (`--ctx 16384,8192`,
the card's settings); the front door's job wrapper (queue, label, cancel) is covered by unit tests, not run on the box.

### Speed

The B70, one describe each, `target: z-image`, the barn picture drawn at each size, `max_pixels` its own area. "Cold"
is a new picture (the instruction from the prompt cache, the picture read); "again" the same picture a second time
(the whole prompt from the cache: what a repeat or a re-roll costs).

| picture | vision positions (prompt tokens) | pp tok/s (cold) | prompt s | tg tok/s | generated | whole describe, cold | again |
|---|---|---|---|---|---|---|---|
| 512 x 512 | 256 (558) | 332 (first request after load) | 1.68 | 56.4 | 134 | 4.07 s | 2.45 s |
| 1024 x 1024 (the default cap) | 1,024 (1,326) | 582 | 1.80 | 48.9 | 114 | 4.15 s | 2.18 s |
| 2048 x 2048 (`max_pixels` 4194304) | 4,096 (4,398) | 451 | 9.13 | 47.1 | 111 | 11.53 s | 2.23 s |

- pp counts the tokens read (prompt less reused) over the prompt's time, the vision tower included; tg 44-60 tok/s over
  all describes (MTP drafts: ~70 % accepted in the server's log). A medium prompt is 90-160 tokens: ~2-3 s of writing.
- The five round-trip pictures (0.8-1 MP): 3.4-4.3 s each, pp 600-680, tg 46-60 (first table below); the styles test
  (16 describes, 8 families x 2 pictures): 2.4-6.0 s, pp 800-1,020 once the family's instruction was cached.
- Thinking left on (no `think_budget`) on the teapot: 286 tokens written (563 characters of thinking first) against
  120 without - about twice the writing time for the same prompt, which is why it is closed.
- **B65: not measured** - its card was busy with production song jobs (acestep) for the whole session and the rule is
  not to share a busy card. `tests/api/test-describe.py --speed 512,1024,2048` measures it when the card is free.

Memory: 19.1-19.2 GiB on the card with the model loaded and after describes (front door's `system.busy` view: 19,595
MB loaded, 19,617 MB after; the card's other tenants not counted); host: the server container 0.45 GB after load,
1.6-6.0 GB after a run (the prompt cache's checkpoints in RAM and the page cache of the mmproj; no pinned weights). The
card says `vram_gib: 20.0`.

### Round trip

Five pictures made for this (none downloaded, nothing committed but the script that draws them): `shapes` (red circle,
blue square, green triangle on white), `barn` (flat barn, sun, sky gradient), `sign` (blocky yellow "OPEN" on navy) -
drawn by tests/api/test-describe.py in pure Python -, and `teapot` (a product photo) and `lighthouse` (a flat vector
illustration), rendered by Z-Image-Turbo from the script's fixed prompts (seeds 1234, 4321). Each was described,
drawn again from the prompt alone (seed 7, `suggested_size`), and compared:

- **colour**: histogram intersection, 4 x 4 x 4 bins (coarse: a navy 10 levels darker lands in another bin - the
  sign's 0.003 on Z-Image is that, the letters and ground are right);
- **tint**: 1 - mean RGB distance of the 16 x 16 thumbnails, cell by cell (colour in place);
- **layout**: correlation of the 16 x 16 grey thumbnails;
- **judge**: the describer itself shown both pictures, "0-10, how closely picture 2 reproduces picture 1" (the same model
  that wrote the prompt: a lenient judge, here as a sanity check, not a measurement);
- **baseline** (in brackets): the same original against the *other four* pictures' copies - what "unrelated" scores.

**`target: z-image-turbo`, drawn by Z-Image-Turbo** (8 steps, ~1 MP):

| picture | colour (baseline) | tint (baseline) | layout (baseline) | judge |
|---|---|---|---|---|
| barn | 0.389 (0.038) | 0.829 (0.602) | +0.539 (-0.078) | 6 |
| shapes | 0.834 (0.151) | 0.901 (0.534) | +0.844 (-0.057) | 9 |
| sign | 0.003 (0.064) | 0.884 (0.465) | +0.488 (-0.044) | 6 |
| teapot | 0.768 (0.172) | 0.826 (0.643) | +0.315 (+0.055) | 9 |
| lighthouse | 0.821 (0.083) | 0.918 (0.621) | +0.903 (+0.002) | 7 |

**`target: p-h3-image` (family h3), drawn by P-H3-Image** (8 steps, ~4 MP: 2368x1760, 2048x2048, 2720x1536):

| picture | colour (baseline) | tint (baseline) | layout (baseline) | judge |
|---|---|---|---|---|
| barn | 0.211 (0.062) | 0.825 (0.612) | +0.613 (-0.034) | 7 |
| shapes | 0.658 (0.139) | 0.753 (0.481) | +0.592 (-0.052) | 9 |
| sign | 0.891 (0.053) | 0.926 (0.532) | +0.780 (-0.054) | 8 |
| teapot | 0.675 (0.167) | 0.795 (0.597) | +0.357 (+0.100) | 9 |
| lighthouse | 0.586 (0.139) | 0.905 (0.610) | +0.955 (+0.074) | 7 |

Every copy beats its baseline on tint and layout, most by a wide margin; by eye the copies are the same scene - same
objects, colours, places and style - not the same picture (the barn came back with a gambrel roof, the sign's letters
smooth rather than blocky, the teapot's background shelf moved). That is as tight as a prompt can hold a picture:
nothing here measures fine detail, and the judge is the describer grading its own work. Prompts written for the z-image
round trip (medium):

- shapes: *A flat vector graphic on a plain white background featuring three solid geometric shapes arranged in a
  triangular layout. In the upper left quadrant, a large bright red circle sits with smooth edges. In the upper right
  quadrant, a solid royal blue square is positioned at the same height as the circle. Centered in the lower half of
  the frame, a vibrant green equilateral triangle points upward, with its base aligned horizontally. ...*
- sign: *A flat vector graphic featuring the word "OPEN" centered on a solid dark navy blue background. The text is
  rendered in a bold, blocky, pixelated font made of thick golden-yellow rectangular segments ...*
- teapot: *A close-up product photo of a glossy turquoise ceramic teapot with a rounded body, a curved spout on the
  left, a C-shaped handle on the right, and a domed lid topped with a small spherical knob. The teapot sits on a light
  oak wooden table, with a folded cream linen napkin resting to its left. ...*

The same pictures for every family (styles test): qwen-image / hunyuanimage open on "A flat vector illustration ..."
in painter's terms with a negative ("3D render, photograph, realistic, shading ..."); h3 opens on what the picture is
and ends "No other text."; ming-image turned the sign into "A minimalist logo design ..."; tags gave
`flat vector graphic, single word "OPEN" centered, bold geometric sans-serif font, golden yellow letters, dark navy
blue background, ...`. A wart: the h3 instruction says not to mention text when there is none, and the describer still
ends some text-free prompts with "no other text" - harmless to the model, not what the guide asks.

Reproduce (on a box with the method deployed): `tests/api/test-describe.py --target z-image-turbo --render
z-image-turbo --judge <chat server>`; on a box where the describer and the image model share one card,
`--stage render|describe|draw|score` runs the parts one at a time.
