# Model cards

One YAML file a model, `<kind>/<id>.yml` (kind: `llm`, `image`, `video`, `audio`, `model3d`, `lora`): everything nextsycl knows
about a model it offers, read when it is needed - not built into the program. `nextsycl models search` lists the
cards, `nextsycl models pull <id>` downloads a card's files, checks them and registers the model with the card's
settings, and `nextsycl models refresh <id>` brings an installed model's entry up to its changed card without touching
its files (docs/registry.md: A changed model card); the front door's `describePromptGuideMD` answers with the card's guide. A release carries this folder
beside the program; `NS_MODEL_CARDS` names another one. A new model is a new card: no code changes unless its
architecture is new (an engine: CONTRIBUTING.md).

```yaml
# Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
id: qwen3.8-27b-q4              # the file's name; what the registry, the API and the CLI call it
kind: llm                       # the file's folder
arch: qwen35                    # the engine that runs it (GGUF general.architecture, or the engine's ARCH)
title: Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
about: Qwen's dense Qwen3.8 27B ...
license: apache-2.0             # optional: the weights' licence (SPDX id, else its name); the front door's model rows show it
commercial: true                # optional: whether the weights may be used commercially (false: research / non-commercial only)
license_note: '...'             # optional: what the one word leaves out (other parts' licences, gated terms)
gpus: '0'                       # the GPUs it loads on: '0', '1', '0,1' (one copy over both), 'all' (weakest first), or '0|1' (one of
                                #   them, the whole model on the first with room, in that order - llm cards that fit one card)
vram_gib: 29.0                  # its memory on each card when it runs, as measured (the front door keeps the card
                                #   free for it before it has measured the model itself; else from the files' sizes)
spans: true                     # optional, image / audio / 3D: ONE copy spans its cards (gpus '0,1'): its server starts on all
                                #   of them (`--gpus 0,1`; an engine that splits itself - Sophon-Realtime's brain and voice), the
                                #   front door frees room on each and never runs a second copy. Without it an image, audio or 3D
                                #   model's cards are where its copies may go, one card a copy (a chat model spreads by gpus alone)
vram_gib_per_card: [23.0, 5.5]  # optional, with spans: its memory on each card, in gpus' order (else vram_gib on each); vram_gib
                                #   stays its largest card's figure for what reads one number
ram_gib: 46                     # optional: its host memory when it runs, not counting reclaimable cache, as measured (pinned tiers, process); the
                                #   front door checks it against MemAvailable before a load, idle models go for it, else it waits or answers no_room
ram_cache_gib: 12.7             # optional: the reclaimable page cache it keeps warm: MemAvailable counts it, other models' loads do not
headroom_gib: 3.3               # optional: of vram_gib, the working memory beyond the weights (KV for its context, arenas,
                                #   a song's or an image's activations) - the front door shows it apart (else: vram_gib less
                                #   the files' share of the card; none when the files do not all sit on the card)
streams: true                   # optional: it reads weights from its files while it serves (experts or rows from the
                                #   file as decode needs them), so its drive's speed is its speed - the front door warns
                                #   before moving it to a slower drive (README: Drives)
sidecars: ["*.vision.gguf", engram/]  # optional: files its engine finds beside the model file by convention, no registry
                                #   path naming them (a draft block, a vision tower, a folder of tables) - names or *
                                #   patterns relative to the model file's folder; a drive move carries them with it
ctx: 131072,32768               # llm: each session's context
tools: true                     # llm: tool calls offered to it (false: dropped)
tasks: true                     # llm: a front end's background tasks (titles, tags) accepted
env: {}                         # settings its engine reads; '{role}' is that file's path once pulled
silo:                           # its tuned kernels (CONTRIBUTING rule 6): kernels/<kind>/<name>/silo, built as
  name: qwen35                  #   dist/silo/libnextsycl-<name>.so (default: the architecture's own, when built)
  required: true                # no load without it (default false: the shared kernels for what it lacks)
  variant: b65                  # optional: that variant of the silo (silo/variants/<name>) on every card, or none;
                                #   default: each card the variant whose manifest matches it, else the silo alone
files:                          # by role: a file defined here, or another card's file by its key
  model:
    key: q27-q4                 # the file's id among all cards (a file is downloaded once, defined once)
    url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/<revision>/Qwen3.8-27B-UD-Q4_K_M.gguf
    size: 16464440224           # bytes
    sha256: 322e194f...         # checked after the download
    path: qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf   # under NS_MODELS
  mtp: q27-mtp                  # defined in qwen3.8-27b-q8.yml
guide: guides/chat.md           # how to prompt it (Markdown, relative to this folder)
```

`silo` also takes a name alone (`silo: qwen35`) or `none` (the shared kernels only); without it a model runs on its
architecture's silo when one is built. `nextsycl models search` shows each model's (`-`: the shared kernels). The
registry entry carries it as the loader's settings (`NS_SILO_NAME`, `NS_SILO_REQUIRED`, `NS_SILO=0`,
`NS_SILO_VARIANT`). A silo's variants are its ops tuned for other cards (CONTRIBUTING rule 6): a card pins one only to
measure it; left alone, each GPU takes the variant made for it. A variant may also carry engine options for its cards
(`settings:` in its `variant.yml`); a card's own `env` setting the same variable wins over the variant's, as `--opt`
and the environment do - set one in a card only when that model needs it on every card.

A file on a gated Hugging Face repository names the page whose terms its account must accept under `gated` (`gated:
https://huggingface.co/<repo>`): `pull` downloads it with the account's token (`HF_TOKEN`, else huggingface_hub's token
file), and without one stops, saying which terms to accept (`model3d/budget3d-idol.yml`'s DINOv3).

A file published as byte ranges of itself (`name.gguf.part01-of-40` ...) lists them under `parts`, in order, each
with its `url`, `size` and `sha256` (the file keeps its own `size`, `sha256` and `path`, no `url`): `pull` downloads
and checks each, joins them, checks the whole and removes the parts (`llm/deepseek-v4.1-flash-vq.yml`).

Other keys: `local` (role: what it is - a part made on the machine rather than downloaded, found in nextsycl.conf
or another registry entry), `base` (a lora: the model it rides on), `defaults` (image / video / audio: the
request's defaults - steps, cfg, size, seconds ...), `max_seconds` (audio: the longest a request may ask, seconds - the engine's own limit, which refuses past it; system.models rows carry it as `max_seconds`, so a client knows it before the model loads; null when the card states none), `speech` (audio: `voices`, `design`, `clone`, `direction` - a clone steered by `instructions` -, `events` - vocal events in the text, `transcript` - a clone needs what the recording says, `speakers` - a script of several speakers, a voice each: `voices` / `ref_audios`, `enrolled` - a saved voice fitted for it is spoken: a model without an encoder, `edit` - it edits a recording's words: audio.edit, `realtime` - a live conversation over /v1/realtime: capability realtime, and with nothing else no audio.speak, `lockstep` - a frame in, a frame out: the row's lockstep: true), `voices` / `languages` (audio: a speech model's built-in voices and languages when its files hold them
where the front door cannot read them - Voxtral TTS's GGUF: a request naming one goes to a model that has it), `does` (model3d: `generate`, `texture`, `paint` (paints a mesh it is given), `rig`, `animate` - what it is
asked for; video: `regen2k` - it makes a draft again at 2K, and only that: capabilities regenerate_2k, video_to_video, sound).

The rules, checked when the cards load and by `cargo test` (`the_model_cards_are_whole`): the id is the file's
name and the kind its folder; a file key is defined once (defining it again the same way is allowed) and every key
a card names is defined; links are https and pinned, with a size and a SHA-256; a model's guide exists; every
`{role}` in `env` is one of its files or local parts. An entry registered without a card (`models add`) gets the
guide of a card of its architecture, else of its kind.
## Concurrency and coexistence

Optional; a card with neither block is *unmeasured*: one copy, shared with anything that fits. Bad values refuse the whole catalog at load, naming the file
(`crates/scheduler/src/profile.rs` is the schema). `nextsycl models bench` proposes the blocks; do not write them by guess.

```yaml
concurrency:
  max_copies_per_card: 2          # >= 1, default 1
  recommended_copies: 1           # 1..max, default 1: extra copies start only when requests wait
  scaling: {"1": 1.0, "2": 1.18}  # total throughput of n copies relative to one (quote the keys)
  measured: {date: "2026-10-31", card: B70, method: nextsycl models bench, notes: ""}
coexistence:
  mode: shared                    # shared | exclusive
  shared_with: ["kind:audio", "model:whisper-large-v3-turbo"]   # empty: anything that fits
  exclusive_reason: ""            # required when exclusive
  peak_extra_gib: 1.5
```

Recommended is where one more copy still gains at least 25% total; exclusive when co-running fails or either side drops below 50%.

## Image controls (schema 1)

Every picture-making image card (an `image` card with `defaults.steps`: text to image, edits) states its request
rules under `controls`; describers, OCR, upscalers, vectorizers and background removers have none. The front door
copies the block into the model's row of `system.models` and `system.state` (its registry entry's `controls`, else
its card's; null for the others), and the platform's one image control reads it to refuse a request outside them.
The block is nextcuda's (docs/models.md, "Image controls (schema 1)"), field for field. It says what the model makes
**well**, not all its engine accepts: full Qwen-Image 2.1 runs any step count but stops half made under 20, so it says
20-100; a few-step model gives its real, narrow range. Bad values refuse the whole catalog at load, naming the file
(`glue/models/src/controls.rs` is the check; `cargo test`: `every_picture_making_card_states_its_request_rules`).

```yaml
controls:
  version: 1                     # integer; a reader treats a version above its own as unknown rules; fields added within a version are optional
  steps: {min: 6, max: 6, default: 6}          # min = max: fixed (the box's own count: a client leaves steps out)
  cfg: {min: 1, max: 30, default: 4}           # null: no guidance (cfg 1 only)
  negative: true                 # a negative prompt is read (only with cfg)
  size:
    min_side: 256                # each side
    max_side: 4096
    multiple: 16                 # each side a multiple of this (refused otherwise - a client does not rely on rounding)
    max_pixels: 4194304          # width x height at most (optional)
    default: 1024x1024
    sizes: []                    # optional: only these WxH
    presets: {ratios: ['16:9'], pixels: [1048576]}   # optional: what the model was trained around, for a picker (capped by max_side / max_pixels)
  seed: true                     # the seed is honoured (picture i: seed + i)
  n: 8                           # pictures one request makes at most
  loras: true                    # per-request LoRAs are applied (false: the model takes none)
  edit:                          # null: text to image only (no pictures, no img2img)
    images: 8                    # pictures one edit takes (the first is the one changed)
    mask: false                  # a mask picture (no engine receives one today)
    strength: true               # img2img strength below 1
    reference: true              # the pictures condition the model (the instruction edit)
    strength_with_reference: false   # strength below 1 also applies with reference true (else it needs reference false)
  transparent: true              # background "transparent" gives a picture with real transparency (RGBA)
  sampler: {values: [euler], default: euler}         # optional
  shift: {min: 0.1, max: 20, default: 3}             # optional: the schedule's shift a client may set; absent: its own
  options:                       # optional: the engine's own per-request options (`options: {name: value}`)
    - {name: eta, type: number, min: 0, max: 1, default: 0, help: '...'}   # type number | bool | text
  note: '...'                    # optional: one sentence for a picker
```

The values come from the engine (image/<arch>/src: its step, guidance and size checks, edits, LoRAs, samplers,
shift, options) and, for what it makes well, from pictures made and looked at; a model nextcuda serves too states the
same rules unless the engine here differs (the card's comment then says where and why).


## Annotations

What a model reads inline in its text (a speech model's tags, a song model's section markers), as nextcuda states
them (its model-cards/annotations.yml, field for field): `annotations`, a list, and `annotations_note`, said under it.

```yaml
annotations_note: 'Only these tags are trained; other bracketed text is not a tag.'
annotations:
- syntax: '[laughter]'          # what to write
  keep: '\[laughter\]'          # a regex of the spans the server's speech text pass leaves as written (absent: it may
                                #   rewrite them - brackets become pauses, symbols words)
  does: 'a laugh where the tag stands'
  tested: 'verified 2026-11-16: ...'   # "verified <date>: <what was measured>", or "unverified: <why>"
```

The front door copies both into the model's row of `system.models` (its registry entry's, else its card's; null when
neither states them): the platform's legend where the text is written. A speech engine's `keep` patterns are also its
own (`AudioEngine::markup`): the audio server's speech text pass keeps the spans they match, so a tag reaches the model
as written - the engine's test checks that its patterns are the card's (`audio/omnivoice`: `the_kept_spans_are_the_cards`).
`tested` is from the real server path (`/v1/audio/speech`), each take heard by Whisper large-v3-turbo.
