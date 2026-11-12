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
- **Or the target's own text encoder (`describer: native`)**: an image model whose text encoder file is a whole
  vision-language model - every layer, the final norm, the output head and the vision tower - can describe with it.
  On nextsycl that is Qwen-Image 2.1 (Qwen3-VL-8B-Instruct) and P-H3-Image (Qwen3-VL-4B): the image engines use those
  files as encoders only, so a small engine of its own (`image/vldescribe`, arch `qwen3vl-describe`: the same layers
  plus a key / value cache and the head, the shared kernels) runs them, each from a *describer card*
  (`qwen3-vl-8b-instruct`, `qwen3-vl-4b-instruct`: the image model's own files, nothing downloaded). Smaller
  models than `best`'s: see Measured. [Describers](#describers-best-native-or-an-id) has the rules and which targets can.
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
| `describer` | string | `best` | who reads the picture: `best` (a chat model that reads pictures: the loaded chat model when it reads pictures - no swap -, else the enabled one whose card takes the least memory, the `-describe` card before the long-context ones), `native` (the target image model's own text encoder) or a describer's id from `image.describers` (a chat model's or a native describer's). [Describers](#describers-best-native-or-an-id) |
| `model` | string | | `describer`'s older name, kept: a describer's id, the same as `describer: "<id>"` (with `describer` absent or `best`); beside `describer: "native"`, or a different id in `describer`: `invalid_params` (field `describer`) |
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
  "describer": "best",
  "describer_model": "qwen3.8-27b-q4-describe",
  "memory_mb": 19617,
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
- `describer` - the request's: `"best"`, `"native"`, or the id it named (in `describer`, or in `model`).
- `model` - what read the picture, in words: the chat model's id; for a native describer the image model and its
  encoder, `"qwen-image-2.1-q8 text encoder (Qwen3-VL-8B-Instruct)"` (a family target: the family's model chosen;
  a native describer named by its id: the first image model, by id, whose encoder it is).
- `describer_model` - the id of the describer that ran (nextsycl's addition: a chat model's or a native describer
  card's id, as `image.describers` lists them).
- `memory_mb` - the describer's memory on its card, MB: what this box measured once it has run, else its model card's
  `vram_gib` (null when neither is known). Same field as nextcuda's.
- `seen` - the size the describer saw (after `max_pixels`); `width` / `height` - the picture as sent.
- `suggested_size` - the target's best area at the picture's aspect (clamped 1:4 .. 4:1), sides on its grid:
  1 MP / multiples of 16 for natural, qwen-image, z-image (256-2048), moss-image, ming-image, tags; 4 MP / multiples
  of 32 for h3 (256-5024) and hunyuanimage (768-3072).
- `tokens.prompt` - the describer's whole prompt (instruction + the picture's vision positions + the request);
  `tokens.reused` - from its prompt cache (the instruction after the first describe for a family: the system turn is the
  same for the same settings; a native describer keeps no cache: always 0).
- `timings` - `pp_tok_s` prompt processing including the vision tower, `tg_tok_s` generation, `seconds` the whole
  describe (a swap to the describer, when it was not loaded, is in it).
- `finish` - `stop`, or `length` when `max_tokens` cut the answer (a note says so). Qwen sometimes goes on after a whole prompt (writes `PROMPT:` and the prompt again until the cap): the prompt is cut where it first ended and `finish` is `stop`, with a note that the describer kept writing.
- `rsvp_granted_s` when the request asked for `rsvp_s`.

`wait: false` answers the job (`{"id", "label", "state": "queued", ...}`); `image.status {"label"}` gives it with the
`result` above once `state` is `done`.

## Describers: best, native or an id

`describer` (default `best`) decides what reads the picture. The instruction (per `target` family, no thinking,
`PROMPT:` / `NEGATIVE:`), the limits and the answer's shape are the same for all three.

| `describer` | runs | resolution |
|---|---|---|
| `best` (or absent) | a chat model that reads pictures | `model` given: that one (as `describer: <id>`). Else the loaded chat model when it reads pictures, else the enabled one whose card's `vram_gib` is least. None enabled: `conflict` |
| `native` | the target image model's own text encoder | a **model** target (`target: "qwen-image-2.1-q8"`, registered or a card's id): its card's text encoder, run by the native describer card that reads the same text-encoder file. A **family** target (`target: "h3"`): of that family's enabled, registered image models, the one whose native describer's card takes the least memory (`vram_gib`) among those that have one; none has: the first one's reason. `natural` and `tags` are prompt styles, not models: no encoder (`invalid_params`). A family with no enabled model here: `conflict` |
| `<id>` | that describer | an id from `image.describers`: a registered chat model that reads pictures, or a native describer. A card's that is not registered: `model_not_downloaded` (with `NS_AUTO_PULL` it is pulled first) |

Which targets have a native describer (verified against the model cards' files and engines, 2026-11-11):

| target (family) | its text encoder | `describer: native` |
|---|---|---|
| `qwen-image-2.1-*` (`qwen-image`) | Qwen3-VL-8B-Instruct, int8 ConvRot (`qwen3vl_8b_int8_convrot.safetensors`: 36 layers, final norm, `lm_head`, vision tower, deepstack mergers) | **works**: `qwen3-vl-8b-instruct` |
| `p-h3-image*` (`h3`) | Qwen3-VL-4B, int8 ConvRot (`p-h3-image_text_encoder_int8_convrot.safetensors`: 36 layers, final norm, vision tower; no `lm_head` - Qwen3-VL-4B ties it to the token embedding, which the file keeps) | **works**: `qwen3-vl-4b-instruct` |
| `minimax-h3-image*` (`h3`) | MiniMax H3's Qwen3-VL-32B (`qwen3vl_32b_minimax_h3-Q4_K_M.gguf` + its vision tower) | `invalid_params`: the file stops after layer 50 of the 64 with no final norm or output head (`minimax_h3_te: unnormalized_hidden_after_layer_50`) - hidden states only, it cannot write. (A family target `h3` picks P-H3-Image's when that is enabled) |
| `hunyuanimage-2.1-*` (`hunyuanimage`) | Qwen2.5-VL-7B-Instruct, Q4_0 GGUF (`qwen2.5-vl-7b-test-q4_0.gguf`: 28 layers, `output.weight`, and the 32-block vision tower in the same file) | `invalid_params` for now: the file reads pictures but carries no model settings or tokenizer, and no engine here writes with it (the olmOCR-2 engine is a Qwen2.5-VL writer, but for Q8_0 with an mmproj and its own settings). A card + a Q4_0 reader for that engine would do it |
| `z-image-turbo*` (`z-image`) | Qwen3-4B (text only) | `invalid_params`: does not read pictures |
| `ming-image-*` (`ming-image`) | Ling-Mini-2.0 (text only) | `invalid_params`: does not read pictures |
| `moss-image-*` (`moss-image`) | Qwen3.5-9B Q8_0, its text weights only (no vision tower in the GGUF) | `invalid_params`: does not read pictures as shipped |
| `natural`, `tags` | - | `invalid_params`: a prompt style, no encoder |

The peer list said Qwen-Image 2512 too: there is no Qwen-Image 2512 model on nextsycl (no card, no engine), so nothing
here; its encoder would be Qwen2.5-VL-7B as HunyuanImage's.

How it is told (model cards, read at run time - a new encoder is a card, not code):

- an image model's card carries `encoder: {name, sees, why}` - its text encoder's name, whether it reads pictures, and
  (one that reads but cannot write) why;
- a native describer card (`kind: image`, `arch: qwen3vl-describe`, `base_model`) names the image model's own
  `text-encoder` and `tokenizer` files by key; an image model whose card names the same `text-encoder` key is the one
  it is native for. Registering it (`system.models.pull {"model": "qwen3-vl-8b-instruct"}`) links the files
  already on the box: nothing is downloaded.

A native describer runs on an image server like any image model (its card's `gpus`, room made first, the image
queue's turn); a `best` describer on the chat switcher as before.

### Listing: `image.describers`

`POST /rpc/image.describers` (no parameters) - what a "Describe with" list offers: the two words `best` and `native`
are fixed, the rest is this list.

```json
{"ok": true, "result": {
  "default": "best",
  "best": "qwen3.8-27b-q4-describe",
  "describers": [
    {"id": "qwen3.8-27b-q4-describe", "name": "Qwen3.8-27B UD-Q4_K_M, pictures to prompts (16K)", "base_model": "Qwen3.8-27B",
     "kind": "chat", "size_gb": 18.76, "vram_gib": 20.0, "image_models": [], "downloaded": true, "enabled": true, "best": true},
    {"id": "qwen3-vl-8b-instruct", "name": "Qwen-Image 2.1 text encoder (Qwen3-VL-8B-Instruct) as a describer",
     "base_model": "Qwen3-VL-8B-Instruct", "kind": "native", "size_gb": 9.36, "vram_gib": 13.5,
     "image_models": ["qwen-image-2.1-q4km", "qwen-image-2.1-q6k", "qwen-image-2.1-q8", "qwen-image-2.1-turbo-q8",
                      "qwen-image-2.1-uncensored-q4km", "qwen-image-2.1-uncensored-q8"],
     "downloaded": true, "enabled": true, "best": false},
    {"id": "qwen3-vl-4b-instruct", "name": "P-H3-Image text encoder (Qwen3-VL-4B) as a describer", "base_model": "Qwen3-VL-4B",
     "kind": "native", "size_gb": 5.13, "vram_gib": 10.5, "image_models": ["p-h3-image", "p-h3-image-lora", "p-h3-image-q8"],
     "downloaded": false, "enabled": false, "best": false}
  ]}}
```

- `describers[]` - every chat model that reads pictures and every native describer, registered first (registry
  order), then the cards not registered. `id` is what `describer` takes. Native describers are named for the language
  model they are (`qwen3-vl-8b-instruct`, `qwen3-vl-4b-instruct`), as nextcuda names its own, so a UI can keep an id
  across the two runtimes; a chat describer's id is its model's.
- `name` - the card's (or entry's) title; `base_model` - the language model it is (the card's `base_model`, else its
  title); `kind` - `chat` or `native`.
- `size_gb` - its files, GB of 10^9 bytes (a native describer's are its image models' own: nothing more on disk);
  `vram_gib` - its card's memory figure (null when the card has none).
- `image_models` - native: the image models (registered or cards) whose text encoder it is - `describer: native`
  with any of them as `target` runs it; chat: `[]`.
- `downloaded` - registered here (false: a card's; describe answers `model_not_downloaded` until it is pulled);
  `enabled`; `best` - the one `best` runs now (the top-level `best` the same, null when none is enabled).

And in `system.models`, an image model that draws gets the capability **`describe_native`** when a native describer
card reads its text encoder (enabled or not registered yet; a disabled one: not). A native describer's own entry has
capabilities `["vision", "describe"]` and methods `["image.describe"]`; it is never an `image.generate` model (named
there: `invalid_params`) nor a `target`. `system.version` features: `image.describe.describer`, `image.describers`.

## Errors

The envelope `{"ok": false, "error": {"code", "message", "retryable", "status", "method", "field"}}`, HTTP 200:

| code (status) | when | retryable |
|---|---|---|
| `invalid_params` (400) | no `image`; a picture that does not decode or is over 40 MP; a `target` that is neither a family nor an image model that draws ("target ocr-model: it does not draw pictures from prompts"); a value out of its range (`max_words 3: from 8 to 400`); a `describer` / `model` that does not read pictures ("describer glm-5.3: an llm model that does not read pictures; ..."); `describer` not a string; `describer: native` beside `model`, or `describer` and `model` naming two describers; **`describer: native` and a target whose encoder cannot describe** - field `describer`, the message names the target and its encoder: "target z-image-turbo: its text encoder (Qwen3-4B) does not read pictures; use describer best", "target minimax-h3-image: its text encoder (Qwen3-VL-32B, MiniMax H3's) reads pictures but cannot write text here (the file stops after layer 50 ...); use describer best", "target natural is a prompt style, not an image model: it has no text encoder of its own; ..." | no |
| `not_found` (404) | `describer` / `model` names no describer here or in the cards ("describer x: no such describer here (image.describers lists them)"); `image: upload:<name>` not here | no |
| `conflict` (409) | `best` and no chat model that reads pictures is enabled (`system.enable qwen3.8-27b-q4-describe`); the describer named is disabled; `native` with a family target and no model of that family enabled ("target h3: no model of the family is enabled here (system.models); name one as the target, or use describer best"); the chat switcher refuses to swap away from a chat model in use ("... is loaded and in use; switching ... would cut that conversation off") | no (the last: later) |
| `model_not_downloaded` (409) | the describer (named, or `native`'s) is a card's that is not registered - for a native describer its files are the image model's: `system.models.pull` links them, nothing is downloaded; for a target that is itself not downloaded its encoder files are missing too. The error carries `model` (the describer card's id), `bytes`, `fits`, `pull`. With `NS_AUTO_PULL` it is fetched instead | no |
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
server, which stops at its next token (while the picture is still being read, at the end of that reading: ~1-10 s).
A running **native** describe is not stopped part-way: the image server's reading has no stop between tokens, so it
ends with its writing (bounded by `max_tokens`; see Measured for its speed) and the job then ends `cancelled`. A
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

## The image page's routes

The image page (`wfe/image`, served through the door at `/open/<model>/`) has no token, so the door answers four routes
under `/open/<model>/api/describe` with the methods above (glue/serve/src/home/describe_page.rs; docs/images.md has the
panel): `GET api/describe` -> `{"ok": true, "result": <image.describers>}`; `POST api/describe` (the fields `image`,
`target`, `describer`, `detail`, `max_words`, `include_negative`, `language`, `instructions`; the job is always
`wait: false` with a label `page-describe-<ms>`) -> the job; `GET api/describe/status?label=` -> `image.status`;
`POST api/describe/cancel {"label"}` -> `image.cancel`, refused (400) for a label that does not start with
`page-describe-`. Errors are the `/rpc` envelope `{"ok": false, "error": {...}}` with the error's HTTP status. Not
part of the cross-runtime contract.

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
# the describers a "Describe with" list offers (best and native besides)
curl -s $D/rpc/image.describers -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d '{}'
# the target's own text encoder reads the picture (Qwen-Image 2.1's Qwen3-VL-8B-Instruct)
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "qwen-image-2.1-q8", "describer": "native"}'
#   -> result.describer "native", result.model "qwen-image-2.1-q8 text encoder (Qwen3-VL-8B-Instruct)",
#      result.describer_model "qwen3-vl-8b-instruct"
# a family: its enabled model whose encoder describer is smallest (h3: P-H3-Image's Qwen3-VL-4B)
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "h3", "describer": "native"}'
# a target whose encoder cannot see -> {"ok": false, "error": {"code": "invalid_params", "field": "describer",
#   "message": "target z-image-turbo: its text encoder (Qwen3-4B) does not read pictures; use describer best", ...}}
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "z-image-turbo", "describer": "native"}'
# a describer by its id (any from image.describers; "model" is the older name for the same)
curl -s $D/rpc/image.describe -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"image": "upload:up_1794470000_barn.png", "target": "p-h3-image", "describer": "qwen3-vl-4b-instruct"}'
# a native describer registered once (its files are the image model's: nothing is downloaded)
curl -s $D/rpc/system.models.pull -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d '{"model": "qwen3-vl-4b-instruct"}'
# and back: draw the prompt
curl -s $D/rpc/image.generate -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"model": "z-image-turbo", "prompt": "<result.prompt>", "size": "<result.suggested_size.size>", "seed": 7}'
# the raw OpenAI-style call (your own instruction)
curl -s $D/v1/chat/completions -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d "{\"model\": \"qwen3.8-27b-q4-describe\",
  \"think_budget\": 0, \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$(base64 -w0 barn.png)\"}},
  {\"type\": \"text\", \"text\": \"Write a text-to-image prompt for this picture.\"}]}]}"
```

The command line does the same against a chat server: `nextsycl image describe barn.png --target z-image [--negative]
[--detail long] [--json] [--server HOST:PORT] [--model ID]` (default: the box's switcher); `--native DESCRIBER_ID
--server HOST:PORT` asks an image server running that native describer (`nextsycl image serve DESCRIBER_ID`).

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

### Native describers (hardac1, 2026-11-11)

Through `nextsycl image describe --native ID --json` (the same `describe::describe_native` the front door runs) against
the build copy's `nextsycl image serve ID` (`qwen3-vl-8b-instruct`, `qwen3-vl-4b-instruct`), on the **B65** (the B70
was running production work); the front door's routing, listing and errors are covered by unit tests, not run on the
box. Pictures: the five round-trip pictures above, `detail: medium`, target the describer's own image model; speed rows:
the barn (2368 x 1760) shown at `max_pixels` 0.26 / 1 / 4.2 MP, `target: natural`. Times are whole describes with the
describer loaded (its load: 10.7 s for the 4B, 13.6 s for the 8B, not included); no prompt cache, so "again" costs the
same as cold.

| describer | picture | prompt tokens | pp tok/s | tg tok/s | written | whole describe |
|---|---|---|---|---|---|---|
| `qwen3-vl-4b-instruct` (target p-h3-image) | the five pictures, ~1 MP | 1,118-1,374 | 361-513 | 48-60 | 46-81 | 3.1-4.8 s |
| | 512 x 512 (0.26 MP) | 443 | 887 | 63.4 | 75 | 1.74 s |
| | 1 MP | 1,208 | 455 | 60.0 | 74 | 3.96 s |
| | 4 MP | 4,279 | 163 | 39.7 | 77 | 28.2 s |
| `qwen3-vl-8b-instruct` (target qwen-image-2.1-q8) | the five pictures, ~1 MP | 1,049-1,305 | 330-458 | 36-42 | 53-88 | 4.1-5.9 s |
| | 512 x 512 | 443 | 707 | 45.0 | 41 | 1.59 s |
| | 1 MP | 1,208 | 410 | 35.4 | 46 | 4.32 s |
| | 4 MP | 4,279 | 151 | 31.5 | 47 | 30.0 s |
| `best` (qwen3.8-27b-q4-describe, B70, from the tables above) | 1 MP, cold | 1,326 | 582 | 48.9 | 114 | 4.15 s (2.2 s again, cached) |
| | 4 MP | 4,398 | 451 | 47.1 | 111 | 11.5 s |

- Memory on the card: the 4B 4.9 GiB loaded, ~6.5 GiB after 1 MP describes, 10.1 GiB after a 4 MP one; the 8B 8.6
  GiB loaded, 13.5 GiB after a 4 MP one (the cache is float32 q | k | v rows, ~0.9 MB a row over 36 layers, plus the
  prompt pass's work). Host: 0.4-0.8 GB. The cards say 10.5 / 13.5 GiB (the 4 MP peaks).
- tg: a written token's attention runs on the engine's silo (flash decoding over chunks of 256 cached rows,
  `kernels/image/vldescribe/silo`); on the shared attention by rows alone the 4B wrote 13-16 tok/s at 1 MP and 4.9 at 4
  MP (B70, before the silo) - the silo is 3.7x / 8x that.
- pp is the weak spot at 4 MP (150-163 tok/s against best's 451): the prompt's causal attention is the shared kernel
  that gives each query row one work-group walking its keys - fine for ~1,200 rows, slow for 4,300. Not tuned: the
  default `max_pixels` is 1 MP, where a native describe is as fast as best's cold one.
- At 1 MP a native describe is about as fast as best's first one (3-6 s), slower than best's repeats (best keeps a
  prompt cache; the native ones do not). Neither swaps a 20 GiB chat model in, which is the native describers' real
  saving: they load in 11-14 s into ~5-9 GiB.

Quality, the same five pictures (the prompts in full are in the build copy's run; excerpts):

- **4B (`qwen3-vl-4b-instruct`, for P-H3-Image)**: right kind of picture every time, the h3 form kept (what the picture
  is first, "no other text" at the end); short - 36-63 words where `medium` asks ~90 - and a few wrong details: the
  barn is "a simple, stylized house" with "a single white rectangular window" (a cream door), the shapes' blue square a
  "rectangle". The teapot and lighthouse prompts are as good as best's in content ("a glossy, light teal ceramic
  teapot ... a white textured fabric napkin lies to its left; soft, diffused daylight from a window to the left").
- **8B (`qwen3-vl-8b-instruct`, for Qwen-Image 2.1)**: more precise - the shapes' places ("a solid red circle
  positioned in the upper left, a solid blue square in the upper right, and a solid green triangle centered below"),
  the sign's "pixelated block letters" on "dark navy blue", the teapot's spout and handle, three seagulls - still
  40-65 words for `medium`, and the barn is "a simple red house" with "a cream-colored rectangular door".
- **best (27B)** writes the asked length (90-160 tokens), names the barn a barn, and is the better describer by eye;
  native is the smaller, encoder-matched one. No round trip (drawing the native prompts back) was run: the image
  models' cards were busy with production work.
