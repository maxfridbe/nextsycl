# Model cards

One YAML file a model, `<kind>/<id>.yml` (kind: `llm`, `image`, `video`, `audio`, `model3d`, `lora`): everything nextsycl knows
about a model it offers, read when it is needed - not built into the program. `nextsycl models search` lists the
cards, `nextsycl models pull <id>` downloads a card's files, checks them and registers the model with the card's
settings; the front door's `describePromptGuideMD` answers with the card's guide. A release carries this folder
beside the program; `NS_MODEL_CARDS` names another one. A new model is a new card: no code changes unless its
architecture is new (an engine: CONTRIBUTING.md).

```yaml
# Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
id: qwen3.8-27b-q4              # the file's name; what the registry, the API and the CLI call it
kind: llm                       # the file's folder
arch: qwen35                    # the engine that runs it (GGUF general.architecture, or the engine's ARCH)
title: Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
about: Qwen's dense Qwen3.8 27B ...
gpus: '0'                       # the GPUs it loads on: '0', '1', '0,1', or 'all' (weakest first)
vram_gib: 29.0                  # its memory on each card when it runs, as measured (the front door keeps the card
                                #   free for it before it has measured the model itself; else from the files' sizes)
headroom_gib: 3.3               # optional: of vram_gib, the working memory beyond the weights (KV for its context, arenas,
                                #   a song's or an image's activations) - the front door shows it apart (else: vram_gib less
                                #   the files' share of the card; none when the files do not all sit on the card)
ctx: 131072,32768               # llm: each session's context
tools: true                     # llm: tool calls offered to it (false: dropped)
tasks: true                     # llm: a front end's background tasks (titles, tags) accepted
env: {}                         # settings its engine reads; '{role}' is that file's path once pulled
silo:                           # its tuned kernels (CONTRIBUTING rule 6): kernels/<kind>/<name>/silo, built as
  name: qwen35                  #   dist/silo/libnextsycl-<name>.so (default: the architecture's own, when built)
  required: true                # no load without it (default false: the shared kernels for what it lacks)
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
registry entry carries it as the loader's settings (`NS_SILO_NAME`, `NS_SILO_REQUIRED`, `NS_SILO=0`).

Other keys: `local` (role: what it is - a part made on the machine rather than downloaded, found in nextsycl.conf
or another registry entry), `base` (a lora: the model it rides on), `defaults` (image / video / audio: the
request's defaults - steps, cfg, size, seconds ...), `speech` (audio: `voices`, `design`, `clone`, `direction` - a clone steered by `instructions` -, `events` - vocal events in the text), `does` (model3d: `generate`, `texture`, `rig`, `animate` - what it is
asked for).

The rules, checked when the cards load and by `cargo test` (`the_model_cards_are_whole`): the id is the file's
name and the kind its folder; a file key is defined once (defining it again the same way is allowed) and every key
a card names is defined; links are https and pinned, with a size and a SHA-256; a model's guide exists; every
`{role}` in `env` is one of its files or local parts. An entry registered without a card (`models add`) gets the
guide of a card of its architecture, else of its kind.
