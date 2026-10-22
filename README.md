<p align="center"><img src="docs/logo-card.svg" alt="nextsycl - the swiss-army compute engine: video, chat, image, music" width="520"></p>

We stand on the shoulders of giants.

# nextsycl

- **A curated Ollama for everything a local machine needs** - chat, images, video, music and speech
- **Use our kernels for your engine** - every engine is a library, its kernels SYCL you can build on
- **No CUDA, no PyTorch, no Vulkan** - SYCL only, optimized for Intel Arc
- **No Python**
- **No Node**
- **Every useful model, optimized**
- **Silo architecture** - only do what is best for model X ([CONTRIBUTING.md](CONTRIBUTING.md) rule 6)
- **Test driven** - nothing is committed until it beats the base, with the same results
- **A single API control plane** - the front door on :8000
- **A Docker-style CLI** - `nextsycl <kind> start | stop | ps | logs | ...`
- **A porting methodology** - [CONTRIBUTING.md](CONTRIBUTING.md) and [the SYCL porting skill](skills/sycl-porting/SKILL.md)

## RUN

`nextsycl models pull <id>`, then pick it in the front door (:8000) or the chat front end - or `nextsycl llm start <id>`
for a chat model. Measured speeds: tg (token generation, decode) / pp (prompt processing):

**On two 32 GB cards** (measured on an Arc Pro B70 + B65, 61 GB of host RAM, an NVMe; two B70s at least as fast):

| id | model | speed |
|---|---|---|
| `deepseek-r1-iq1s` | **DeepSeek-R1 671B** (unsloth UD-IQ1_S, 131 GiB) - the experts that do not fit VRAM in pinned RAM (multiplied on the CPU) and on the NVMe | ~8 tok/s / ~135 tok/s, 32K context |
| `glm-5.3-flash-uncensored-256k` | **GLM-5.3-Flash** uncensored IQ2 | ~18 tok/s / 450-1,100 tok/s, to 256K |
| `deepseek-v4-flash-iq2m`, `-256k` | **DeepSeek-V4-Flash 284B** (unsloth UD-IQ2_M, 84.7 GiB) - the experts beyond VRAM in pinned RAM; llama.cpp pages it from the disk (no result in 100 minutes) | ~27 tok/s with its MTP draft block (26 at 3.7K, 17 at 256K) / 635-1,000 tok/s (943 at 256K), 128K or 256K context (1M tested: 287 tok/s prompt, 12 tok/s decode) |
| `deepseek-v4.1-flash-iq2xxs`, `-fast` | **DeepSeek-V4.1-Flash-1017** (Unsloth's UD-IQ2_XXS: 146 GiB of experts in IQ1_S-Q2_K by layer, its 126 GiB Engram table inside the file read a row at a time) - the experts beyond VRAM in pinned RAM and the NVMe. `-fast`: cold low-weight experts left out, attention in q4_K (perplexity +3-5%); a 12,000-token novel without a loop | exact: tg 7-9 tok/s after a prompt (~14 sustained) / pp 125 (2.5K) to 440-470 tok/s (25K-140K). `-fast`: tg 11-13 tok/s after a prompt (25.3 sustained over 12K tokens of prose; 21.4 over 512) / pp 118-187 (2.5K) to ~475 tok/s (140K); 128K context. [Ladder](docs/ladders/deepseek-v4.1-flash-iq2xxs_ladder_stats.md) |
| `deepseek-v4.1-flash-q2kxl`, `deepseek-v4.1-flash-iq1m` (each also `-fast`), `deepseek-v4.1-flash-q3kxl` | other quants of the same model, secondary: UD-Q2_K_XL (178 GiB of experts) keeps a 12,000-token novel clean where UD-IQ2_XXS drifts from ~8,000 words; UD-IQ1_M (experimental) is faster but perplexity +9% and repeats from ~5,000 words (llama.cpp too); UD-Q3_K_XL is near the original. [Port notes](docs/ports/deepseek41-unsloth.md#other-quants) | Q2_K_XL: tg 8.8 exact / 19.2 `-fast` (512 tokens), 22.3 `-fast` sustained. IQ1_M: tg 16.2 / 25.8. Q3_K_XL: tg ~4.4 |
| `nemotron-3.5-super-q2kxl` | **Nemotron 3.5 Super 120B-A12B** (unsloth UD-Q2_K_XL; hybrid Mamba-2 / latent MoE / attention; text only) | ~40 tok/s with its MTP drafts (30 without) / ~790-1,140 tok/s, 32K (128K tested: 868 / 34) |
| `nemotron-3.5-super-iq1m` | **Nemotron 3.5 Super** UD-IQ1_M (the same; or on the B70 alone, experts in pinned RAM: ~21 tok/s / ~350 tok/s at 2K) | ~46 tok/s with MTP (32 without) / ~880-1,470 tok/s, 32K (128K tested: 1,120 / 38); `-b70`: the B70 alone, ~21 / 450-560 |

**On one 32 GB card** (measured on an Arc Pro B70):

| id | model | speed |
|---|---|---|
| `qwen3.8-flash-next-iq2_xs` | **Qwen3.8-Flash-Next** 125B MoE IQ2_XS (MTP drafts) | ~80 tok/s / ~1,200 tok/s, 128K |
| `qwen3.8-flash-next-coder-iq1_m` | **Qwen3.8-Flash-Next Coder** IQ1_M | ~77 tok/s / ~1,400 tok/s, 128K |
| `qwen3.8-27b-q4` | **Qwen3.8-27B** dense UD-Q4_K_M (MTP drafts) | 43-49 tok/s, 128K |
| `muse-glimmer-30b-q8` | **Muse Glimmer 30B** abliterated, dense Q8_0 with its DFlash drafter (reasons on its to=self channel) | 30-77 tok/s by text (18.7 without drafts) / ~1,350 tok/s (1,215 at 35K), 128K |
| `muse-glimmer-30b-q8-b70`, `-b65` | the same on one card alone, without the drafter (it does not fit beside the model) | 18.7 (B70) / 16.9 (B65) tok/s, 128K |
| `muse-glimmer-30b-q4`, `-q6` | **Muse Glimmer 30B** Q4_K_M / Q6_K on the B70 alone, with its drafter | 43 / 35 tok/s (25.4 / 20 without drafts) / ~1,400 tok/s, 128K |
| `qwen-image-2.1-q8` | **Qwen-Image 2.1** text to image, edits | 1024x1024, 40 steps: 23.8 s (16.5 s int8) |
| `hunyuanimage-2.1-distilled-q8` | **HunyuanImage 2.1** text to image at 2K (quoted text spelled by its glyph encoder) | 2048x2048, 8 steps: 7.8 s; 1024x1024: 2.5 s |
| `unlimited-ocr-q8`, `-q4` | **Unlimited-OCR** pictures and PDFs to Markdown (tables as HTML, boxes), its own page | a page: ~2-3 s to see, ~130 tok/s written |
| `paddleocr-vl-1.6` | **PaddleOCR-VL** text, tables (HTML), formulas (LaTeX), charts, seals, spotting; the same OCR page | a page: ~1.2 s to see, ~200-250 tok/s written |
| `olmocr-2-7b-q8` | **olmOCR-2** whole pages to Markdown (tables HTML, equations LaTeX), turned pages righted; the same OCR page | a page: ~3 s to see, ~55 tok/s written |
| `unirig` | **UniRig** a skeleton and skin weights for any mesh (`model3d.rig`) | a 7.6K-vertex humanoid: 3.2 s |
| `rignet` | **RigNet** a skeleton and skin weights for a character mesh (`model3d.rig`); its own crate, **GPLv3** as RigNet's code | a 7.6K-vertex humanoid: 0.6 s |
| `animateanymesh` | **AnimateAnyMesh** a mesh set in motion by a sentence, as morph targets (`model3d.rig` with a prompt) | a 7.6K-vertex humanoid, 16 frames: 2.8 s |
| `minimax-h3` | **MiniMax H3** video with sound | 896x672, 4.5 s: ~90 s |
| `minimax-music3-int8` | **MiniMax Music 3** songs | a minute of song: 89 s |
| `qwen3-tts-custom` | **Qwen3-TTS** speech | a minute of speech: 10.7 s (8.2 s int8); a sentence: 0.64 s |

The earlier VQ GGUF of V4.1 (1.6-bit experts, tg ~23 tok/s) repeated itself past ~2K tokens and was dropped for Unsloth's: [docs/ports/deepseek41-unsloth.md](docs/ports/deepseek41-unsloth.md).

## What it is

A Rust runtime for language, image, video, audio and 3D models on Intel Arc GPUs, with SYCL kernels. Each kind of model
has a contract its server and command line drive, and each model architecture its own engine behind it, tuned end to end -
a library another program can use without the server. New models are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) walks
through a port, from each kind's template engine. What each engine is, how it was checked and what it measures:
[docs/engines.md](docs/engines.md). A skill for porting to SYCL on Arc - the safety rules, migration traps, parity method,
profiling and what made things fast: [skills/sycl-porting/SKILL.md](skills/sycl-porting/SKILL.md).

## Highlights

- **Two cards:** the layers split over the GPUs, the experts in an exclusive VRAM / pinned-RAM / NVMe store.
- **Speculative decoding that is exact:** verify passes are bit-identical to one-token decode (`nextsycl spec-check`).
- **Kernels tuned per model, in silos** (`kernels/<kind>/<arch>/silo/`), the shared ones left as they were.
- **Long context** to 256K (1M on DeepSeek V4), a prompt cache that survives restarts, several requests at once.
- **An OpenAI-compatible server** with streaming, tool calls, logprobs and the energy of every request.

All of it, with the commands and the numbers: [docs/llm-server.md](docs/llm-server.md).

## Quick start

New to it, on an Arc Pro B70? [B70_Startup_guide.md](B70_Startup_guide.md) goes from a fresh clone to chat,
pictures, speech, songs and video, with what each step takes.

```sh
./build.sh                          # the kernel libraries and the program, in a container with oneAPI (podman)
echo "NS_MODELS=$HOME/models" > nextsycl.conf
dist/nextsycl models add glm ~/models/GLM-5.3-Flash-IQ2.gguf --gpu all   # register a model (or models pull <id> from the catalog)
dist/nextsycl llm start glm         # load it; the OpenAI API on 127.0.0.1:8085
dist/nextsycl llm status            # live: the GPUs, the request running, the prompt cache
dist/nextsycl llm inspect 4         # one request as JSON (settings, timings, previews; NS_KEEP_REQUESTS are kept)
dist/nextsycl llm chat "Hello"
dist/nextsycl llm stop
```

`dist/nextsycl help` lists every command and setting. What `llm status`, `ps` and `inspect` print: [docs/llm-server.md](docs/llm-server.md).

## Documentation

| Topic | Page |
|---|---|
| Every supported model: id, engine, size, status (what `models search` lists) | [docs/models.md](docs/models.md) |
| The registry, `models pull / add / move`, drives, engine options (`--opt-NAME`) | [docs/registry.md](docs/registry.md) |
| The language-model server: features, status / ps / inspect, the OpenAI API | [docs/llm-server.md](docs/llm-server.md) |
| The front door (:8000), its API, maintenance mode, the switcher, gpustat | [docs/host-services.md](docs/host-services.md) |
| More language-model engines: Qwen3.8-27B, Nemotron 3.5 Super | [docs/llm-engines.md](docs/llm-engines.md) |
| Images: Qwen-Image 2.1, HunyuanImage 2.1, the image server and its page | [docs/images.md](docs/images.md) |
| Video: MiniMax H3 as `nextsycl video`; making films | [docs/video.md](docs/video.md), [docs/video/h3.md](docs/video/h3.md) |
| Audio: songs (Music 3) and speech (Qwen3-TTS, Breeze, VibeVoice, S2 Pro, Voxtral, VoxCPM2) | [docs/audio.md](docs/audio.md) |
| 3D: meshes from pictures, rigging and animation | [docs/3d.md](docs/3d.md) |
| Speed: every model at each context length, benchy | [docs/speed.md](docs/speed.md), [docs/benchy/](docs/benchy/) |
| Context ladders: pp and tg by prompt length, with the machine and silo each ran on | [docs/ladders/](docs/ladders/) |
| Speed tricks: what each silo does, what was tried, what is open | [docs/tricks.md](docs/tricks.md) |
| Long context: memory a token, scaling, what 256K takes | [docs/256k-context.md](docs/256k-context.md) |
| How the code is laid out; every environment variable | [docs/architecture.md](docs/architecture.md), [docs/environment.md](docs/environment.md) |
| Porting a model; the rules | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Checking an engine; LogProbChain; lessons learned | [docs/checking.md](docs/checking.md), [docs/logprobchain.md](docs/logprobchain.md), [docs/lessons.md](docs/lessons.md) |
| Per-model port notes (DeepSeek V4.1, HunyuanImage, TRELLIS.2, ...); GLM-5.3's math | [docs/ports/](docs/ports/), [docs/glm5next.md](docs/glm5next.md) |
| Model cards | [model-cards/README.md](model-cards/README.md) |
| Where it is going, what is next | `PLAN.md`, `TODO.md`, `NOTES.md` |

## Standing on

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp):** the GGUF format, the quantization formats and their
  dot products, the reference every layer is checked against (`reference/llama-dump`).
- **[Strata](https://github.com/Niko1221/Strata):** the SYCL kernels in `kernels/strata` (MIT; see
  `kernels/strata/PROVENANCE.md`) - its Intel Arc port is this project author's own (Strata PR #423, merged in
  0.1.39) - and the ideas behind the expert store and the host mirror.
- **[ds4](https://github.com/antirez/ds4)** by antirez: how GLM-5.3's MTP block runs, and the file this runtime
  serves first.
- **Zhipu AI's GLM-5.3**, and the people who quantized it: the uncensored GSQ-RCO and IQ2 files.
- **Intel's oneAPI:** SYCL, oneMKL, and SYCLomatic, which migrated the kernels from CUDA.

## Releases

The version is `yy.mmdd.###`: the commit's date (UTC), then its number among that day's commits, from git
(`./version.sh`; `nextsycl version` prints the one built in). Every push to `main` is built and tested by
`.github/workflows/build.yml` and published as a release `v<version>` with `nextsycl-<version>-linux-x86_64.tar.gz`
(the program, the kernel libraries `libnextsycl-<kind>.so`, `ns.h`, the docs). Other branches and pull requests keep the tarball as a workflow
artifact.

MIT licensed, as the kernels it imports.

[Buy me a cup of tokens](https://buymeacoffee.com/maxfridbe)
