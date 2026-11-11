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
  (nothing new downloaded) with a 16K + 8K context instead of 128K + 32K, so it takes 19.6 GiB of a card instead of 29 and
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
- `finish` - `stop`, or `length` when `max_tokens` cut the answer (a note says so).
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

See [Speed](#speed) and [Round trip](#round-trip) below.
