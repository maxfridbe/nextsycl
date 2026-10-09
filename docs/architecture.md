# Architecture

nextsycl runs language, image and video models on Intel Arc GPUs. Each **kind** of model has a contract - a Rust
trait its server and command line drive - and each model **architecture** has its own engine behind it: a complete
runtime for that model on these GPUs, with its own kernels. There is no generic forward pass parameterised by a config:
the speed here comes from decisions that only make sense for one model (GLM-5.3's KDA scan sized per card, its MLA
indexer's ring, its expert store; Qwen3.8-Flash-Next's verify window and drafter graphs), and every new model gets
the same freedom. The engines are libraries: another program can load one without the server or the CLI.

All GPU work is SYCL (oneAPI, Level Zero). That is the one hard rule: no other GPU runtime anywhere (CONTRIBUTING.md).

## The layout

```
crates/                    THE FOUNDATION - every kind uses it, it knows no kind
  sys        nextsycl-sys        the kernel libraries' shared C ABI (kernels/ns/ns.h), opened at run time
  core       nextsycl-core       GPUs (one SYCL context each), device / host buffers, arenas, copies; which kind's
                                 kernel library this process uses (use_kind)
  gguf       nextsycl-gguf       the GGUF and safetensors file formats
  tok        nextsycl-tok        tokenizers and chat templates
  diffusion  nextsycl-diffusion  what every diffusion engine shares: samplers and schedules by name, the sigma
                                 schedules, LoRA references, pictures (PNG); the diffusion kernels' bindings (Nsd)
  qwen3vl    nextsycl-qwen3vl    the Qwen3-VL text tower (int8 ConvRot): a text encoder image and video engines share

llm/                       LANGUAGE MODELS
  contract   nextsycl-llm        the contract: Engine, Session / Decoder / Checkpoint, Sampler, EngineKind; host sampling
  glm5next   nextsycl-llm-glm5next   GLM-5.3-Flash
  qwen4exp   nextsycl-llm-qwen4exp   the Qwen3.8-Flash-Next family (IQ2_XS, Coder IQ1_M, Swift 1.5)
  example    nextsycl-llm-example    a template: the contract filled with documented stubs

image/                     IMAGES
  contract   nextsycl-image      the contract: ImageEngine, ImageRequest (+ Edit), Step, ImageKind
  qwenimage21 nextsycl-image-qwenimage21  Qwen-Image 2.1: DiT, VAE, pipeline, stage checks
  example    nextsycl-image-example  a template

video/                     VIDEO
  contract   nextsycl-video      the contract: VideoEngine, VideoRequest (+ Keyframe), Progress, Clip, VideoKind
  example    nextsycl-video-example  a template

glue/                      WHAT MAKES ENGINES A PRODUCT (libraries too)
  models     nextsycl-models     the registry of what a machine serves, settings, the catalog of supported models
  serve      nextsycl-serve      the servers: OpenAI-compatible APIs over a kind's contract, the prompt cache, telemetry

cli/
  nextsycl   nextsycl            the program: argument parsing and output; lists the engines it has, by kind

kernels/                   SYCL ONLY - one library a kind (dist/libnextsycl-<kind>.so)
  ns/                     the shared part, linked into every kind's library: GPUs, memory, queues, copies, tickets
  strata/                 the imported Strata kernels (MIT), linked into the llm library
  diffusion/              the shared diffusion kernels (H3's, on oneDNN), linked into the image and video libraries
  llm/<arch>/  image/<arch>/  video/<arch>/   an engine's own kernels and its header (ns_<kind>_<arch>_* symbols)
```

### The rules (checked by `cli/nextsycl/tests/architecture.rs` on every `cargo test`)

- **Dependencies point down.** foundation <- a kind's contract <- that kind's engines; glue <- contracts (never
  an engine); the program <- everything. An engine never depends on another kind, the glue or the program; two
  kinds never depend on each other - what both need moves down into the foundation (as `nextsycl-diffusion`).
- **One GPU runtime: SYCL.** No crate that brings another (Vulkan, CUDA, OpenCL, Metal, wgpu, candle, PyTorch,
  ONNX Runtime...) in Cargo.lock; no other runtime's headers in the project's kernels; the kernel libraries link
  against no other runtime (`./build.sh test`). oneAPI's own libraries (oneMKL, oneDNN) run on the same SYCL queue
  and are fine.
- **One library a kind.** A process of one kind (`nextsycl_core::use_kind`, default llm) opens only that kind's
  kernel library: an llm server never loads image or video code. An engine binds its own symbols by name
  (`nextsycl_sys::Api::symbol`, in its `ffi.rs`); the shared tables hold only `ns.h`.

## The kinds' contracts

**llm** (`nextsycl-llm`) - what the server needs from a language model, and nothing about how it is computed:

- **Sessions:** `session(max_ctx)`, `reset_session`, `copy_session`, `save` / `restore` / `read_checkpoint` - a
  conversation's whole state, opaque (`Session`, `Checkpoint` box the engine's own types; only it reads them).
- **Reading:** `feed`, `feed_until` (stops between chunks for a newcomer), `forward`, `forward_rows`.
- **Decoding:** `decoder` / `decoder_after`, `step` (one or more committed tokens: drafts are the engine's affair),
  `rollback`, `forward_batch` (a row a conversation), drawing through a `Sampler` (`dist` + `uniform` let an engine
  do speculative sampling; `device` lets it draw on the GPU; a plain closure is a sampler too).
- **Reporting:** `gpu_info`, `expert_stats` (MoE engines), `report`, `cache_fingerprint` (what a disk checkpoint
  must match), `save_profile` (what the next load should know).
- **Optional:** `capture_attention` / `take_attention` (LogProbChain), `hold_arena`.
- A file is served by the engine whose `EngineKind::archs` holds its `general.architecture` (`kind_for`).

Every llm engine is held to the same checks: `spec-check` (verify passes equal one-token decode), `batch-check`
(a batch's rows equal each conversation alone), `check` against a reference dump, `bench`, `bench --needle`.

**image** (`nextsycl-image`) - `generate(request, progress) -> pictures`: the prompt, size, steps, guidance, seed,
count, sampler / schedule / shift, per-request LoRAs, an optional `Edit` (the picture, reference pictures, a mask,
a strength), RGBA; `defaults()` for what a request leaves out; `Step` progress. A model is several files by role
(`ModelFiles`: transformer, text encoder, VAE...); `ImageKind::roles` says which, `archs` which registry entries.

**video** (`nextsycl-video`) - `generate(request, progress, cancel) -> Clip`: the clip's prompt and settings,
keyframes, an audio reference, the output path; progress by stage; `cancel` stops it at a step boundary;
`unload` gives the GPUs back. The queue, the studio and the tools built on clips sit above the contract.

## Adding a model

CONTRIBUTING.md walks through a port step by step; each kind's template (`<kind>/example` and
`kernels/<kind>/example`) is the starting point.

## The llm engines

- **glm5next** - GLM-5.3-Flash: KDA + MLA with a DSA indexer, 288 experts (VRAM / pinned host store), MTP.
- **qwen4exp** - Qwen3.8-Flash-Next (Strata's family: the IQ2_XS general model; the Coder IQ1_M and Swift 1.5 files
  are the same architecture): Gated DeltaNet + QSA, hyper-connections, the hashed PLE, 512 experts, all in VRAM over
  the cards. Its glue (`kernels/llm/qwen4exp/qwen.cpp`) is the Strata port's verify window, a stage per GPU; its
  Rust side the memory plan, the PLE rows, sessions and checkpoints. It binds its own ABI from the library by name
  (`nextsycl_sys::Api::symbol`): an engine's symbols are not in the shared tables. Its settings: `NS_QW_MTP` (Strata's MTP
  runtime directory), `NS_QW_EXPERT_PROFILE` (its expert profile: which experts stay in VRAM on one card),
  `NS_QW_CHUNK` (prompt chunk, 2048; Strata serves 4096), `NS_QW_SPEC` / `NS_QW_SPEC_MIN_P` (4 / 0.5),
  `NS_QW_EAGER=1` (no graphs), `NS_QW_WINDOWS=1` (prompts through windows), `NS_QW_PROFILE=1` (round timings), `NS_QW_PLE_PREFETCH=0` (a window's PLE rows read one after another, not asked of the
  kernel together first), `NS_QW_COUPLED=0` (a sampled request's draws on
  the host, its drafts the draft layer's argmax; by default on the GPU with coupled drafts, `ns_qw_set_sampling`).
  The projection modes (llama.cpp's control vectors with the experimental-speed-projection package's
  `--cvec-mode project --cvec-dir per-layer`, Strata's cvec kernel on the stage's own context, `cvec.cpp`):
  `NS_QW_CVEC=<file.gguf>:<scale>`, `NS_QW_CVEC_LAYERS=4,44`, `NS_QW_CVEC_MODE=project|add`,
  `NS_QW_CVEC_DIR=per-layer`. The Coder has 256 experts (the kernels take the count from the file); Swift's router is
  F32 in the file and is converted to BF16 at load (exact).

## The model registry

`nextsycl models` (`glue/models`) keeps what a machine serves: an entry's id, kind, files, GPUs, contexts and engine
settings, so an engine's settings are given once, at `models add --set`, and `nextsycl llm start <id>` applies them.
The README's Models section has the commands.

## What stays shared on purpose

- **One kernel library a kind**, not one an engine: one toolchain run builds them all, and an engine's code reaches
  only the processes of its kind.
- **The expert-store and pipeline ideas** are not a shared library yet: GLM's are tuned to its shapes. When a second
  MoE engine wants the same, factor what both actually share - not before.
