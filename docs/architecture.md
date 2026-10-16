# Architecture

nextsycl runs language, image, video, audio and 3D models on Intel Arc GPUs. Each **kind** of model has a contract - a Rust
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
  mesh       nextsycl-mesh       meshes (PBR textures, skins, morph animation), GLB and OBJ, and the CPU tools: weld,
                                 decimate (quadric), remesh (isotropic), re-UV (charts, packing, baking), closest points

llm/                       LANGUAGE MODELS
  contract   nextsycl-llm        the contract: Engine, Session / Decoder / Checkpoint, Sampler, EngineKind; host sampling
  glm5next   nextsycl-llm-glm5next   GLM-5.3-Flash
  qwen4exp   nextsycl-llm-qwen4exp   the Qwen3.8-Flash-Next family (IQ2_XS, Coder IQ1_M, Swift 1.5)
  qwen35     nextsycl-llm-qwen35     dense Qwen3.5 / Qwen3.8 (Qwen3.8-27B): gated DeltaNet + gated full attention
  example    nextsycl-llm-example    a template: the contract filled with documented stubs

image/                     IMAGES
  contract   nextsycl-image      the contract: ImageEngine, ImageRequest (+ Edit), Step, ImageKind; OcrRequest /
                                 OcrOutput for an engine that reads pictures (reads, ocr)
  qwenimage21 nextsycl-image-qwenimage21  Qwen-Image 2.1: DiT, VAE, pipeline, stage checks
  unlimitedocr nextsycl-image-unlimitedocr  Unlimited-OCR: pages to Markdown (SAM-B + CLIP-L, a DeepSeek-V2 MoE decoder)
  example    nextsycl-image-example  a template

video/                     VIDEO
  contract   nextsycl-video      the contract: VideoEngine (generate; jobs / run_job), VideoRequest (+ Keyframe),
                                 Progress, Clip, VideoKind
  h3         nextsycl-video-h3   MiniMax H3: H3's engine, jobs and mp4 writer
  example    nextsycl-video-example  a template

audio/                     AUDIO
  contract   nextsycl-audio      the contract: AudioEngine, AudioRequest (description, lyrics, length), Step, Audio
                                 (a waveform; WAV with an INFO chunk), AudioKind
  minimaxmusic3 nextsycl-audio-minimaxmusic3  MiniMax Music 3: the language model and depth decoder, the flow
                                 transformer and its windows, the decoder, stage checks
  qwen3tts   nextsycl-audio-qwen3tts  Qwen3-TTS: the prompt per voice kind, the talker and code predictor, the 12 Hz
                                 codec's decoder and encoder, the speaker encoder (host), stage checks
  voxcpm2    nextsycl-audio-voxcpm2  VoxCPM2: MiniCPM4 language models, the local encoder and flow-matching DiT,
                                 AudioVAE V2, a .pth reader, stage checks
  example    nextsycl-audio-example  a template

model3d/                   3D
  contract   nextsycl-model3d    the contract: Model3dEngine (generate: pictures to a mesh; rig: a skeleton and skin
                                 weights, or motion from words), MeshRequest / RigRequest, Abilities, Step, Model3dKind
  example    nextsycl-model3d-example  a template

glue/                      WHAT MAKES ENGINES A PRODUCT (libraries too)
  models     nextsycl-models     the registry of what a machine serves, settings, the catalog of supported models
  serve      nextsycl-serve      the servers: OpenAI-compatible APIs over a kind's contract (llm, image, audio), the prompt
                                 cache, telemetry; video: the daemon (queue, a worker process a GPU) and the studio

cli/
  nextsycl   nextsycl            the program: argument parsing and output; lists the engines it has, by kind

kernels/                   SYCL ONLY - one library a kind (dist/libnextsycl-<kind>.so)
  ns/                     the shared part, linked into every kind's library: GPUs, memory, queues, copies, tickets
  strata/                 the imported Strata kernels (MIT), linked into the llm library
  diffusion/              the shared diffusion kernels (H3's, on oneDNN), linked into the image, video and audio libraries;
                          diffusion/flash: flash attention (ARK on sycl-tla) as libnextsycl-flash.so, loaded on use
  llm/<arch>/  image/<arch>/  video/<arch>/  audio/<arch>/  model3d/<arch>/   an engine's own kernels and its header
                          (ns_<kind>_<arch>_* symbols)
  <kind>/<arch>/silo/     the engine's silo: kernels tuned for that model alone, replacing shared ones for what they
                          cover - their own library, dist/silo/libnextsycl-<arch>.so, opened by the engine at load
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
- **Tuning lives in silos** (CONTRIBUTING.md rule 6). The shared kernels (`kernels/ns`, `kernels/strata`,
  `kernels/diffusion`, an engine's kernels another engine calls) stay as the engines on them were tuned; a model's
  faster version of one goes in its silo, `kernels/<kind>/<arch>/silo/`, built as `dist/silo/libnextsycl-<arch>.so`
  (no shared part inside: its undefined symbols resolve against the kind's library, opened first). The engine opens
  it at load (`nextsycl_core::silo(ARCH)`, once a process; `NS_SILO=0` or `NS_SILO_DIR` to change where), binds its
  symbols by name, asks each op's `_supported` per call and falls back to the shared kernel otherwise - with no
  silo file it runs on the shared kernels alone, just slower. Its load log says which.

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

**audio** (`nextsycl-audio`) - `generate(request, progress) -> Audio`: the description, the lyrics, a length limit,
steps, guidance, seed; `defaults()` (with the longest it makes and its sample rate); progress by the engine's phases,
and an error from `progress` cancels. Files by role, a sharded file as `<role>-1`, `<role>-2` ... (`shards`).

**model3d** (`nextsycl-model3d`) - `generate(request, progress) -> Mesh` (one or several pictures of an object, each
with the side it shows; texture or not; seed, steps, guidance, resolution, a triangle budget) and `rig(request,
progress) -> Mesh` (a mesh in; a skeleton and skin weights, or with a prompt a morph animation); `abilities()` says
which it does, how many views it takes and whether it paints; `defaults()`. Meshes are `nextsycl_mesh::Mesh`, written
as GLB. The server (`nextsycl model3d serve`) answers `/v1/model3d/generate` and `/rig`; the front door runs them as
jobs (`model3d.generate`, `model3d.rig`, as image.generate's) and the CPU tools itself (`model3d.tool`).

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
- **qwen35** - dense Qwen3.5 / Qwen3.8 (Qwen3.8-27B): gated DeltaNet (glm5next's conv, L2 norm and delta-rule scan)
  and gated full attention (its own, `kernels/llm/qwen35/q35.cpp`), a dense SwiGLU; the weights in their GGUF blocks.
  The first engine with a **silo** (`kernels/llm/qwen35/silo/`): its decode products for the types its files hold
  (Q3_K, Q4_K, Q5_K, IQ4_NL, IQ3_S, IQ4_XS), started from Strata's wide kernels and tuned for its shapes - scales from
  one aligned load, the IQ4 codebook and IQ3_S grid in local memory - 1.3-4.3x Strata's per type on the B70 and the
  same results; Q8_0 / Q6_K and the rest go to the shared ones. Settings: `NS_Q35_CHUNK` (prompt rows a pass: 2,048 when they fit beside the weights and a session, else 1,024 or 512),
  `NS_Q35_PROFILE=1` with `NS_PROFILE=gpu` (every kernel timed on the device between queue timestamps, with its bytes
  and arithmetic: time, GB/s and TFLOP/s against the card's ceilings by phase and kernel, decode and prompt passes apart,
  and the device time between kernels - `llm/qwen35/src/probe.rs`), `NS_Q35_MMVQ_ROWS`
  (rows a sub-group, every silo type) and `NS_Q35_MMVQ_GROUPS` (work-groups of the kernels with a table, 512).
  MTP: the draft block (the GGUF's `MTP/` file, `NS_Q35_MTP`) reads each position's normed hidden state with the token
  after it (eh_proj split into its two halves at load), a gated full-attention layer with its own cache, a SwiGLU,
  its head; it runs behind every pass over that pass's rows. `step` drafts `NS_Q35_DRAFTS` (3) tokens chained on the
  block's own output, verifies them in one pass, keeps what the sampled tokens match and rolls the DeltaNet states back
  from the pass's snapshots (`spec-check`: verify rows bit for bit the one-token decode's - the decode attention splits
  each row's keys as a one-row pass at that position would).

- **deepseek2** - DeepSeek-V3 / R1 (unsloth's dynamic GGUFs: IQ1_S experts): MLA absorbed over an fp16 cache of
  [latent | rotated key] (576 a token and layer; YaRN as llama.cpp runs it), the 256 experts in three tiers - VRAM
  slots, pinned host slots, the file (read by a pool of threads, 1 MiB jobs, what a layer waits for ahead of what is
  read ahead) - exclusive, LRU. Decode multiplies the VRAM experts in one launch a layer (its own IQ1_S / IQ2_XXS
  kernels, `kernels/llm/deepseek2`) and the pinned ones on the CPU where they are (AVX-512 VNNI, `cpu.rs`): PCIe moves
  ~28 GB/s both directions together, so a swap costs more than the CPU's product; a few of the CPU's experts go up to
  VRAM behind it while the copy lane is idle. Q4_K / Q5_K decode products are the qwen35 silo's wide kernels, copied.
  Small uploads and readbacks go through pinned rings copied by kernels (the runtime's memcpy waits behind the
  store's copies on the copy engine). Group-limited routing (8 groups, best 4) on by default (`NS_DS2_GROUPS=0`: as
  llama.cpp runs this file, which has no group metadata). Checked against llama.cpp layer by layer (cosine >= 0.998 to
  layer 59); 1.58-bit R1 on a B70 + B65 with 61 GB of host memory: ~8 tok/s decode, ~135 tok/s prompt reading - a prompt runs a
  layer at a time over spans of 4,096 tokens (attention a chunk at a time, every expert loaded once a span, the next
  layer's file-only experts read behind the current one's) (llama.cpp with its experts
  mmapped: 1.0, 2.8-3.5 with madvise prefetch).

- **deepseek4** - DeepSeek-V4-Flash (unsloth's UD-IQ2_M: IQ2_XXS gate / up, IQ3_XXS and MXFP4 down): 64 MQA heads of
  512 (one key head, the key the value too, a rotated tail of 64, learned sinks) over a window of the last 128 raw
  keys plus compressed ones - ratio 4 with overlapping blocks and the lightning indexer's top 512 on the even layers
  from 2, ratio 128 on the odd ones (each block a per-value softmax pool, normed and rotated at its start; YaRN x16 on
  the compressed layers). Grouped output LoRA (8 groups), mHC (4 streams, GLM-5.3's), sqrt-softplus routing with hash
  routing (the file's token table) on layers 0-2. The rest - experts in VRAM and pinned host slots, the prompt path,
  checkpoints - is the glm5next engine's, forked. Decode multiplies a layer's experts in two launches (gate | up, then
  down: the deepseek2 silo's split-Q8 scheme with IQ3_XXS, IQ2_S and MXFP4 lanes - every layer, `mmvq.cpp`), attention on XMX (`attn.cpp`: 8 heads of a
  row as the DPAS rows); MXFP4 has its own kernels (`ds4.cpp`) - all in its silo, `kernels/llm/deepseek4/silo`
  (dist/silo/libnextsycl-deepseek4.so). Checked against llama.cpp layer by layer (5 tokens: cosine >= 0.98, logits 0.998, the same
  next token; long prompts drift only where near-tied experts swap). B70 + B65, 61 GB of host memory: ~25.6 tok/s
  decode (21.8 at 3.7K, 17 at 140K), ~630 tok/s prompt reading at 3.7K tokens, ~870 at 128K, 943 at 256K (the
  indexer's keys fp16; its prompt scores fused: queries packed once as XMX B tiles in global memory, each key tile
  against a row's 64 heads, relu x weight summed in registers - the GEMM writing every head's score was half a 256K
  prompt). The passes' small tables go up and the routers'
  logits come down through pinned rings copied by kernels (the runtime's memcpy waits behind the expert copies on
  the copy engine), and the shared expert runs while the host routes; the layers split by the cards' speed.
  MTP: the original checkpoint's draft block (its last shard; unsloth's GGUFs leave it out) converted after the pull
  (`llm prepare`, mtp.rs: the FP4 experts losslessly MXFP4, the FP8 matrices Q8_0) into a sidecar opened as a further
  shard - block 43, window-only attention and its own mHC head, on the last card. Its 256 MXFP4 experts are 3.2 GiB, so
  only the most recently used 128 stay in VRAM: one not there is left out of that draft (verification is exact; ~3
  points of acceptance) and a reader thread brings it from the file through pinned staging for the passes after. With
  a 256K reserve that keeps ~200 main experts off the file tier: 23.6 -> 24.7 tok/s. It runs over
  each pass's last 136 rows (a key needs only its row; the window is 128), drafts one token, and `step` verifies it
  with the next in a 2-row pass that is bit for bit two one-row passes (attention splits by a row's own cells, a
  token's experts summed in expert order): spec-check 0 difference, the same text with and without. 64-93% of the
  drafts accepted; B70 + B65 +8-16% (26 tok/s at 3.7K).
- **muse-glimmer** (`llm/museglimmer`) - Muse Glimmer 30B (Meta's dense agentic model; the abliterated Q8_0): 52
  layers of gated attention (32 / 2 heads of 128, q / k norms, a sigmoid gate from its own projection) and a SwiGLU,
  each followed by its own RMS norm (eps 1e-8) before the residual; 39 layers over a 2,048 sliding window with rotary
  positions (llama.cpp's NORM: adjacent pairs), every fourth over everything without positions; the embedding
  RMS-normed, the logits x 0.196 then soft-capped at 20. A windowed layer's cache is a ring of its window and a pass,
  a full layer's the context. Its kernels (`kernels/llm/museglimmer/mg.cpp`): the norms with the rotation, the ring
  stores, the windowed attention, the gate, the post norm into the residual, the soft cap and argmax on the GPU. Its
  silo: `silo/mmvq.cpp` repacks each Q8_0 matrix at load (quants apart from scales), q | k | v | gate and gate | up
  stacked, and runs decode's products (1..16 rows) on XMX - int8 DPAS of 8 weight rows x a block against the 16
  columns' quants, which the norm / gate / SwiGLU kernels write straight in the packed operand layout: flat in the rows
  (B70: ~550-590 GB/s for 1 or 16), and a column's arithmetic the same however many share the pass; `silo/attn.cpp`
  is flash attention on XMX (the qwen35 silo's, for heads of 128 and the window / ring), for prompt rows and, split at
  absolute key positions with chunks at absolute multiples, for decode rows - so a verify row is its one-row pass bit
  for bit (spec-check: 0 difference; drafted text = plain text).
  DFlash (llama.cpp's src/models/dflash.cpp, common/speculative.cpp; the drafter GGUF beside the model): a 5-layer
  drafter fed the target's hidden states going into layers 2, 14, 26, 38, 50 (fc, a norm, each layer's k / v into its
  own ring caches at every position a pass covers); a draft is one non-causal pass of [last token, MASK x 15] through
  the target's embedding and head, the 15 argmaxes; `step` verifies [token, drafts] in one pass (greedy: the argmaxes
  on the GPU, 64 bytes back). Its K-quant matrices are requantized to the silo's Q8 at load. With two GPUs it runs on
  the first (the target's 52 layers stay on the B70; 6 of them on a B65 cost more than the drafter there). Checked
  against llama.cpp layer by layer (14 tokens: every layer >= 0.9999; 6,153 tokens, the ring wrapped: the same next
  token). B70 + B65: 30-77 tok/s decode with drafts by text (prose ~31-33, code ~75; a step: verify ~62 ms + draft
  ~12.5 ms, 2.5-4.5 tokens), 18.7 without; sampled requests drawn on the GPU (ns_mg_sample: the host sampler's
  distribution), as fast a step as greedy; ~1,350 tok/s prompt at 4.2K, 1,215 at 35K.
  Q4_K_M and Q6_K (`silo/kq.cpp`): K-quant matrices repacked at load a row at a time, losslessly (4-bit parts in Q4_K's
  order, Q6_K's 2 high bits laid out so a mask and a shift land them in a dword's bytes, the scales unpacked); decode's
  products put the weights in DPAS's B operand (a lane a row, expanded by masks into local memory; its scales its own)
  against the rows' token-major quants (the xq layout carries them and each block's sum, for Q4_K's minimum): exact
  as Q8_0's (spec-check 0). Prompt passes expand them to half. With the drafter on the B70 alone: Q4_K_M 25.4 tok/s
  plain, 43 drafted (front door, temperature 1: ~34 prose, ~74 code), 1,404 tok/s prompt at 6.7K; Q6_K 20 plain, 35
  drafted (~31 / ~57). Products ~420-520 GB/s (Q8_0's ~570: the unpack, not the reads, bounds them). A model whose
  matrices the silo does not all take gets float rows from the decode norms (the shared products need them). The chat template's channels (a `to=self` message, then `to=user`) come back as
  `reasoning_content` and `content` (nextsycl_tok::muse_channels).

- **nemotronh** - NVIDIA Nemotron 3.5 Super VL 120B-A12B (`nemotron_h_moe`, llama.cpp's nemotron-h graph): 88
  pre-norm residual blocks of three kinds - 40 Mamba-2 mixers (an input projection to z | x B C | dt, a causal conv +
  SiLU, the selective scan with 128 heads of 64 in 8 groups and a state of 128, a SiLU(z)-gated group RMS norm), 40
  latent-MoE layers (a float32 sigmoid router whose selection bias picks 22 of 512 experts, weights renormalized and
  scaled by 5; the token projected to a 1024-wide latent where each expert is relu^2(up) then down, the weighted sum
  projected back; a relu^2 shared expert on the full width) and 8 attention layers (32 / 2 heads of 128, no
  positions). Its own kernels (`kernels/llm/nemotronh/nh.cpp`): the conv with its bias, the scan, the gated norm,
  relu^2 and the head-128 attention; every product is Strata's Q8_1 kernels (they read the file's Q2_0 and IQ4_NL
  expert downs, whose 2,688 inputs no 256-block type divides). unsloth's UD-Q2_K_XL (53.8 GiB) lies all in VRAM, the
  layers split over the cards by their memory (`NS_SPLIT` overrides). First version: passes of at most 8 rows, the
  routing on the host each MoE layer, an expert a launch; no MTP drafts yet (its block is in the file), no vision
  tower (mmproj). Checked against llama.cpp layer by layer (`reference/llama-dump`): the same next token and top
  logit, logits cosine 0.999997; a token fed alone after the rest gives the same logits as the prompt at once. B65 +
  B70: decode 18.2 tok/s, prompt 27 tok/s (58 tokens).

## The image engines

- **qwen-image-2.1** (`image/qwenimage21`) - Qwen-Image 2.1's DiT, VAE and pipeline (text to image, edits).
- **unlimited-ocr** (`image/unlimitedocr`) - Baidu's Unlimited-OCR (the DeepSeek-OCR architecture): it reads pictures
  rather than makes them (`ImageEngine::reads`, `ocr`; the image server's `POST /v1/ocr`, the front door's `image.ocr`,
  the page wfe/ocr). From two GGUFs: the projector (SAM ViT-B + CLIP-L + a linear, F16, float32 here) and the
  decoder (12 layers of 1280: MHA of 10 x 128 with NEOX rotary positions, layer 0 a SwiGLU, then 64 experts of 896
  - softmax gate, top 6 - beside 2 shared; expanded to half at load, Q8_0 / Q4_K / Q6_K / Q5_0). The reading is the
  reference's `infer` step by step: the global view (ImageOps.pad to 1024) and, a side over 640, tiles of 640 on the
  grid of 2 .. 32 closest to the page's aspect (PIL's bicubic resampling: `Picture::resize_bicubic`, bit-exact with
  PIL), image tokens tiles first (a newline after each patch row), then the global view, then the separator; greedy
  with no 35-gram repeated in the last 128 tokens; the cache keeps the prompt and only the last 128 tokens written
  (the reference's ring: a page of any length in fixed memory). Its kernels (`kernels/image/unlimitedocr`): SAM's
  attention with decomposed relative positions, the patch / 3x3 / window rearrangements, quick GELU, the experts'
  gathers and scatters, Q5_0 to half; the linears, norms and attention are the diffusion kernels'. Checked: the
  preprocessing against PIL (identical), the vision tower against the reference's own deepencoder.py on the
  original weights (projector output rel 3e-4, tiles and global view), the decoder against llama.cpp
  (reference/llama-dump/ns-ocr-dump.cpp: every layer cos >= 0.9999; llama.cpp's tile path resamples SAM's and CLIP's
  positions differently from the reference - plain bicubic, CLIP's CLS row from the end - so tiles compare with
  the reference, not llama.cpp). Several pages: each read on its own (each_page, default; joined with <PAGE> marks) or
  under the model's own "Multi page parsing." (single-page instructions stop after page 1); pictures inside a page
  (empty image blocks) read again cropped (figures); columns found first (layout.rs: a gutter free of ink for two
  characters down a quarter of the page splits it - above, each side, below - the sides not split again) and read
  region by region; PDFs split into pages by hayro (pure Rust, Picture::from_pdf) on
  the server and by a vendored pdf.js on the page. B70: a page (tiles) seen and read at ~600-900 tok/s, written at ~120-145 tok/s;
  tests/api/test-ocr.py (three public transcripts, pictures only) passes on Q8_0 and Q4_K_M.
- **paddleocr-vl** (`image/paddleocrvl`) - PaddlePaddle's PaddleOCR-VL 1.6: an element reader (text, tables, formulas,
  charts, seals, spotting - each its instruction, `ImageEngine::ocr_prompts` names them for the page). From two BF16
  GGUFs and the original tokenizer.json (ERNIE's SentencePiece-shaped BPE: spaces as "▁", no prefix - nextsycl-tok
  reads it): the vision tower (SigLIP NaViT, 27 blocks of 1152, at the page's own size - smart_resize to multiples of
  28 within 0.1 .. 1 megapixel, PIL bicubic; the learned 27 x 27 position grid resized bilinear with align_corners as
  transformers does (the repo's remote code and llama.cpp use half-pixel centres); axial 2D rotary positions, 18
  frequencies of the row and 18 of the column a head of 72; float32 weights) and the projector (2 x 2 merge, two
  linears), then the decoder (ERNIE-4.5 0.3B: 18 layers of 1024, 16 query heads over 2 key / value heads, M-RoPE with
  Qwen2-VL's positions - sections 16 / 24 / 24, base 500000; weights in half). Its kernels
  (`kernels/image/paddleocrvl`): the 2 x 2 merge, cache row copies, and one-row attention over grouped heads split over
  chunks of 256 keys (flash decoding: 90 -> 245 tok/s on a page's 1300-token prompt). Tables come back from OTSL as
  HTML (spans kept), spotting as `<|det|>` boxes like Unlimited-OCR's; a plain reading that loops (a dense grade grid)
  is cut and the picture read again as a table, which goes where the loop began. Checked against transformers on the
  original weights (every stage cos >= 0.99999, 40 of 40 greedy tokens). B70: a page seen at ~1000 tok/s, written at
  ~200-250; test-ocr.py: cairn and hslda pass, wtamu reads 84% of its words (the header's right column is skipped by
  the reference too).
- **olmocr2** (`image/olmocr2`) - Ai2's olmOCR-2-7B-1025, a Qwen2.5-VL-7B fine-tune that reads whole pages into
  Markdown (tables HTML, equations LaTeX). From two GGUFs: the projector (F16, float32 here: the two temporal patch
  kernels summed, 32 blocks of 1280 with RMS norms, SwiGLU and 2D rotary positions - 20 frequencies of the row, 20 of
  the column a head of 80 - attending within windows of 4 x 4 units, blocks 7 / 15 / 23 / 31 over the whole page; the
  merger's GELU MLP) and the decoder (Qwen2-7B: 28 layers of 3584, 28 query over 4 key / value heads with biases,
  M-RoPE 16 / 24 / 24 at base 1e6). The decoder's Q8_0 stays Q8_0 on the card, split at load into aligned int8 rows
  and scales: the prompt's pass expands each matrix to half for oneDNN, a written token's reads 16 bytes a lane
  (q8r_gemv: 16 -> 58 tok/s, ~460 GB/s). Patches are laid out in window order on the host (prep.rs), the merger's rows
  gathered back to row order. The reading is olmOCR's pipeline: the page fitted to 1288 on its longest side, its one
  instruction (text first, then the image, in Qwen's chat template), the front matter dropped and a page the model
  says is turned turned (PIL's ROTATE_r) and read again. Its kernels (`kernels/image/olmocr2`): the window attention,
  the Q8 products, row gathers and the one-row grouped attention (flash decoding). Checked against llama.cpp on the
  same GGUFs (ns-ocr-dump with the prompt placing the image: every tower block cos >= 0.99996, the decoder >= 0.998 -
  llama.cpp's CPU products quantize activations to 8 bits - and 32 of 32 greedy tokens). B70: a page seen at ~600
  tok/s, written at ~55-58; test-ocr.py passes all three transcripts.

## The audio engines

- **minimaxmusic3** - MiniMax Music 3 from its diffusers checkpoint (bf16 language model and depth decoder, f32 flow
  transformer, condition encoder and decoder). Its kernels (`kernels/audio/minimaxmusic3`): a small-batch product
  that reads a matrix once for up to 8 rows (half, or int8 with a scale a row against float activations), causal
  attention against a half cache with 4 query heads a key head (split over the keys for one-row decode), Qwen3's
  q / k norm with RoPE, the flow transformer's partial RoPE, the codebooks' embeddings, the frames' layer mix, the
  sampler's step and overlap blend, and the decoder's 1-D convolutions on oneDNN (`conv.cpp`). The shared diffusion
  kernels do the rest (norms, SwiGLU, the flow transformer's GEMMs and attention). Settings: `NS_MM3_INT8=1`
  (`--opt-int8`: the language model and depth decoder in int8; a prompt's wide products expand them to half),
  `NS_MM3_FPMATH` (the decoder's convolutions: strict float32 by default, or f16 / bf16 / tf32),
  `NS_MM3_NSD_CONV=1` (the shared plain-loop convolutions instead).
- **qwen3tts** - Qwen3-TTS from Qwen's checkpoints (the model's bf16 safetensors, the speech tokenizer's f32 one, the
  tokenizer's vocab.json / merges.txt). The language models run on the MiniMax Music engine's kernels (the same Qwen3
  layer shapes; its attention also takes 2 query heads a key head, its convolutions a left / right padding each:
  `ns_audio_mm3_conv1d_lr`); its own (`kernels/audio/qwen3tts`) are the codec's: SnakeBeta, the depthwise causal
  convolution, attention over a causal window, float32 RoPE, layer-scaled residuals, ELU (the encoder's); the
  encoder's strided causal convolutions go through `ns_audio_mm3_conv1d_lr` too. The text embedding stays
  on disk (a row read a token). Settings: `NS_Q3T_INT8=1` (`--opt-int8`).

## The model registry

`nextsycl models` (`glue/models`) keeps what a machine serves: an entry's id, kind, files, GPUs, contexts and engine
settings, so an engine's settings are given once, at `models add --set`, and `nextsycl llm start <id>` applies them.
The README's Models section has the commands.

## What stays shared on purpose

- **One kernel library a kind**, not one an engine: one toolchain run builds them all, and an engine's code reaches
  only the processes of its kind.
- **The expert-store and pipeline ideas** are not a shared library yet: GLM's are tuned to its shapes. When a second
  MoE engine wants the same, factor what both actually share - not before.
