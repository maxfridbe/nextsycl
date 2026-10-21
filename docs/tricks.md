# Speed tricks

[← nextsycl](../README.md)

Every speed trick the engines and their silos use, what was tried and gave nothing, and what is still open - by silo.
Written 2026-10-21 from the repository's own notes; every number here comes from the file it cites (paths and line
numbers as of that day). Speeds say what they measure: **pp** is prompt processing (a prompt read), **tg** token
generation (decode).

## What a silo is

A silo is one model's own kernel library: `kernels/<kind>/<arch>/silo/*.cpp`, built as
`dist/silo/libnextsycl-<arch>.so`, opened by the engine at load. Each silo op has a `_supported` check; what the silo
does not cover runs on the shared kernels (`kernels/ns`, `kernels/strata`, `kernels/diffusion`), and `NS_SILO=0` runs
the engine on the shared kernels alone. A kernel tuned for one model goes in that model's silo, never into the shared
ones (CONTRIBUTING.md rule 6), so a silo starts as a copy of a shared or another silo's kernel and is changed there.
That is why the same trick shows up copied from silo to silo, and why carrying a trick to another model means copying
it into that model's silo.

The silos today (`ls -d kernels/*/*/silo`): llm `deepseek41`, `deepseek4`, `museglimmer`, `nemotronh`, `qwen35`,
`qwen4exp`; audio `qwen3tts`, `voxcpm2`; image `hunyuanimage21`; model3d `animateanymesh`, `hunyuan3d`, `trellis2`,
`unirig`. Two language engines have no silo but carry many of the tricks in their own kernels: `glm5next`
(`kernels/llm/glm5next`) and `deepseek2` (`kernels/llm/deepseek2`); they get columns too. Engine-level tricks (expert
tiers, drafts, pipelines) live in `llm/<arch>/src` and count for the engine's column.

## How to read the tables

- **✓** used (on by default, or the engine's normal path).
- **✓°** built and working but off by default (an opt-in switch, usually an approximation or a room-for-speed trade).
- **✗** tried and gave no gain or lost; the measurement is in [Tried, no gain](#tried-no-gain).
- **open** listed as a next step in the docs; see [Still open](#still-open).
- blank: not applicable to the model, or not tried.

The last column says where the trick is described; the numbers are in the sections below.

## The language models

`g5` glm5next (no silo), `q4x` qwen4exp, `q35` qwen35, `ds2` deepseek2 (no silo), `ds4` deepseek4, `ds41` deepseek41,
`mg` museglimmer, `nh` nemotronh.

### Kernels

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Decode products read weights in aligned 16-byte loads, scales from one aligned load (no byte loads) | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | qwen35 silo `mmvq.cpp:5-14`; lessons.md:43; TODO.md:548 |
| IQ grids / codebooks staged in local memory | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | `mmvq.cpp` (qwen35, ds4), `std.cpp:56`; TODO.md:119, 140 |
| Activations in a split Q8 layout (quants contiguous, 16-byte aligned) against dp4a | | | ✓ | ✓ | ✓ | | | ✓ | qwen35 `silo.h`; ds4 `mmvq.cpp:1-7`; nh `nhs.cpp:1-5` |
| Float32 activations straight into the product (no quantize launch) | | | | | | ✓ | | | ds41 `q4k.cpp:1-3, 28`, `std.cpp:1-9` |
| Norm (or SwiGLU) and the activations' quantization in one launch | | ✓ | ✓ | | | | ✓ | ✓ | qwen35 `norm.cpp`; mg `silo.h`; environment.md:368 |
| Several rows a sub-group sharing the activations | | | ✗ | | | ✓ | | | qwen35 `mmvq.cpp:9`; ds41 `q4k.cpp`, TODO.md:28-29 |
| Decode products on XMX (int8 DPAS), a column's arithmetic the same at any pass width | | | | | | | ✓ | | mg `mmvq.cpp`, `kq.cpp` |
| Weights repacked losslessly at load (aligned rows, scales apart) | | | | | | | ✓ | ✓ | mg `silo.h:1-4`, `kq.cpp`; nh `nhs.h` |
| Weights decoded once for all columns of a verify pass | | | | | | ✓ | ✓ | | environment.md:571 (`NS_DS41_Q4K_COLS`) |
| A layer's chosen experts in one or two decode launches | ✓ | ✓ | | ✓ | ✓ | ✓ | | ✓ | architecture.md:233, 251; llm-engines.md:92 |
| Routing on the GPU, the products queued before the host reads the routing | | ✓ | | | | ✓ | | ✓ | deepseek41.md:104; llm-engines.md:92; TODO.md:368 |
| Flash attention on XMX (joint_matrix) for prompt rows | | | ✓ | | ✓ | ✓ | ✓ | ✓ | qwen35 `attn.cpp`; copies in ds4, ds41, mg, nh |
| Decode attention split over the keys, merged after (verify rows exact) | | | ✓ | | ✓ | ✓ | ✓ | ✓ | qwen35 `attn.cpp:13` |
| Prompt experts decoded straight into the matrix units (no fp16 copy) | ✓ | ✓ | | | | ✓ (VQ file) | | ✓ | TODO.md:114-146; architecture.md:191; deepseek41.md:63 |
| Faster weight expansion to half for prompt GEMMs (wide stores, several blocks a work-item) | ✓ | | ✓ | | | | | | qwen35 `dequant.cpp`; TODO.md:152-156 |
| Small decode launches fused into chains | ✓ | | | | | ✓ | ✓ | | deepseek41.md:108; TODO.md:93; mg `silo.h` |
| Lightning indexer scored on XMX in one kernel, two-level top-k | open | | | | ✓ | ✓ | | | ds4 `idx.cpp`; deepseek41.md:75, 113; TODO.md:447 |
| Kernel shape picked per card or timed at load | ✓ | | | | | ✓ | | | TODO.md:173-176; deepseek41.md:83-89 |
| Decode as recorded SYCL graphs | open | ✓ | ✗ | | | open | | | environment.md:201; TODO.md:618; deepseek41.md:175 |
| Sampling on the GPU | | ✓ | | | | | ✓ | | TODO.md:391-395; architecture.md:286 |
| Top-k for sampling in linear time on the host | ✓ | | | | | | | | TODO.md:555-557 |

### Expert offload

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Experts in tiers: VRAM, pinned host memory, the file | ✓ | ✓ (one card) | | ✓ | ✓ | ✓ | | ✓ (one card) | architecture.md:230; deepseek41.md:139 |
| VRAM filled at load from an expert profile of earlier runs | ✓ | ✓ | | | ✓ | ✓ | | | TODO.md:533-540; deepseek41-unsloth.md:70 |
| Hot experts pinned by a usage profile | ✗ | | | | | | | | TODO.md:536-537 |
| LRU on the VRAM tier | ✓ | | | ✓ | ✓ | ✓ | | | TODO.md:95; architecture.md:230 |
| LRU on the pinned host tier | | | | ✓ | | ✗ | | | `llm/deepseek41/src/engine.rs:149`; deepseek41-unsloth.md:89 |
| Swaps on two copy lanes, spare free VRAM slots (no copy down before a copy up) | ✓ | | | | | ✓ | | | TODO.md:93; environment.md:553 |
| The next layer's experts fetched ahead by its router's guess | ✓ | | | ✓° | ✓ | ✓ | | | TODO.md:98-99; deepseek41.md:92 |
| The guess's threshold from measured hit rates | ✓ | | | | | ✓ | | | TODO.md:98; deepseek41-unsloth.md:63-65 |
| Guessing two layers ahead | ✗ | | | | | | | | TODO.md:99 |
| A thread pool reading the file, a job a slab | | | | ✓ | | ✓ | | | deepseek41-unsloth.md:63; environment.md:481 |
| O_DIRECT expert reads | | | | | | ✗ | | | deepseek41-unsloth.md:86-87 |
| Missed experts read in place over PCIe at decode | ✗ | | | | | | | | TODO.md:608-609 |
| Missed experts multiplied on the CPU | ✗ | | | ✓ | | | | | TODO.md:541-547; architecture.md:233-236 |
| Cold experts left out of a token (the "cold skip") | | | | | ✓ (drafts) | ✓° | | | deepseek41-unsloth.md:74-76; architecture.md:262-266 |
| An arriving expert's gate / up launched before its down is in | ✓ | | | | | | | | TODO.md:458 |

### Memory and data movement

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| The prompt arena's VRAM lent to experts during decode | ✓ | | | ✓ | | | | | TODO.md:93; environment.md:469 |
| A VRAM staging ring: host and file experts copied up while the resident ones compute | open | | | | | ✓ | | | deepseek41.md:66; TODO.md:83 |
| Expert-major placement (every layer the same share of each tier) | | | | | | ✓ | | | deepseek41.md:69 |
| The plan's spare VRAM trimmed | | | | | | ✓ | | | deepseek41.md:76 |
| Pass buffers budgeted per chunk, a VRAM guard | ✓ | ✓ | ✓ | | | | | | TODO.md:195-199; environment.md:102; llm-engines.md:57-58 |
| Small uploads and read-backs through pinned rings copied by kernels | | | | ✓ | ✓ | | | | architecture.md:237, 259-261 |
| Rows needed next asked of the page cache together (fadvise WILLNEED) | | ✓ | | | | | | | TODO.md:405-410 |
| Level Zero v2 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`) | | | | | | ✓° | | | deepseek41.md:121 |
| Matrices requantized at load for faster products | | | | ✓ (kv_b to fp16) | | ✓° (dense to q4_K) | ✓ (drafter to Q8) | | environment.md:468, 544; architecture.md:290 |

### Decoding

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| MTP draft block, verify rows bit-exact | ✓ | ✓ | ✓ | | ✓ | ✗ (DSpark) | | ✓ | TODO.md:48-50; llm-engines.md:32; architecture.md:262 |
| DFlash block drafter (15 drafts a step) | | | | | | | ✓ | | architecture.md:286-295 |
| More than one chained draft a step | ✗ | ✓ | ✓ | | ✗ | ✗ | ✓ | ✓ | TODO.md:615; environment.md:505 |
| Prompt-lookup (n-gram) drafts | ✗ | | | ✓° | | | | | TODO.md:558-563; environment.md:474 |
| Speculative sampling at a temperature (drafts drawn, coupled) | ✓ | ✓ | | | | | | | TODO.md:569-575, 391-395 |
| Several requests decoded in one pass | ✓ | | ✓ | | | | | | speed.md (Benchy); architecture.md:220 |
| The shared expert run while the host routes | | | | | ✓ | | | | architecture.md:261 |

### Prompt processing

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Bigger prompt chunks (each expert's weights serve more tokens) | ✓ 6,144 | ✓ 2,048 | ✓ 2,048 | ✓ | ✓ | ✓ 16,384 | ✓ 2,048 | ✓ 2,048 | TODO.md:195; deepseek41-unsloth.md:66 |
| A layer at a time over a span of many tokens (each expert read once a span) | | | | ✓ 4,096 | | open | | | architecture.md:240-243; TODO.md:36 |
| The next layer's cold experts read while this one computes | | | | ✓ | | ✓ | | | environment.md:480, 545 |
| Cold experts left out of a prompt pass | | | | | | ✓° | | | deepseek41-unsloth.md:80 |
| Dense products in fp16 on XMX past a row count | ✓ | | | ✓ | ✓ | ✓ | | | environment.md:136, 465, 502, 576 |
| Short prompts' dense products through the decode kernel, 8 rows a launch | | | | | | ✓ | | | deepseek41.md:73 |
| A long read stops at a chunk's end for a newcomer | ✓ | | | | | | | | TODO.md:522-527 |

### Two GPUs

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Prompt chunks pipelined over the two GPUs | ✓ | | | | ✓ | ✓ | | ✓ | TODO.md:165-169; lessons.md:19 |
| Layers split by the cards' speed, the strongest last | ✓ | | ✓ | | ✓ | | | ✓ | environment.md:497; llm-engines.md:97 |
| A GPU's unused expert share passed to the next | ✓ | | | | | | | | TODO.md:178-181 |
| Two cards for decode speed (not only room) | | ✗ | ✗ | | | | ✗ | | TODO.md:414-417; llm-engines.md:38-51 |
| More layers on the B70 | | | | | | ✗ | | | environment.md:516 |
| The cards' slots swapped (B70 on x16) | open | | | | | | | | TODO.md:504-507 |

### Numerics

| Trick | g5 | q4x | q35 | ds2 | ds4 | ds41 | mg | nh | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Quantized attention cache (q8 latents; q8_0 / q4_0 rows) | ✓ | | ✓° | | | | | | TODO.md:441-445; llm-engines.md:66-78 |
| Quantized cache rotated by a Hadamard transform | | | ✓° | | | | | | environment.md:203 |
| q4_K nibbles to floats by their bits (no conversion) | | | | | | ✗ | | | environment.md:572; deepseek41.md:84 |
| Cold-skip weights rescaled to the kept ones | | | | | | ✗ | | | deepseek41-unsloth.md:76 |

## The other kinds

`q3t` qwen3tts, `vcp` voxcpm2 (audio); `hi21` hunyuanimage21 (image); `hy3d` hunyuan3d, `t2` trellis2, `aam`
animateanymesh, `ur` unirig (3D).

| Trick | q3t | vcp | hi21 | hy3d | t2 | aam | ur | Where |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| A frame / patch recorded once as a SYCL graph and replayed | ✓ | ✓ | | | | | | q3t `q3s.h:1-7`; vcp `vcs.h` |
| Sampling / the sampler's step on the device (bit-exact with the host's) | ✓ | ✓ | | | | | | q3t `q3s.cpp:220-240`; vcp `vcs.cpp:23-30` |
| RMS norm read into the product, SwiGLU taken out of gate / up | ✓ | | | | | | | q3t `q3s.cpp:18-23` |
| q / k norm, RoPE, cache store and attention in one kernel | ✓ | | | | | | | q3t `q3s.h:38-46` |
| int8 weights with a half scale every 32 | ✓° | | | | | | | environment.md:268-269 |
| Q8_0 read as int8 inside oneDNN's product (no half copy) | | | ✓ | | | | | hi21 `mm.cpp:1-8` |
| Epilogues (GELU, the gated residual) as oneDNN post-ops | | | ✓ | | ✓ | | | hi21 `mm.cpp:5-8`; t2 `mm.cpp:1-6` |
| Guidance's two branches as one batch | | | ✓ | | | | | images.md:73-77 |
| int8 x int8 products after a 256-block Hadamard rotation | | | ✓° | ✓° | | | | environment.md:215, 316 |
| Norms re-parallelized for the model's shapes (a sub-group a row, many work-groups a group norm, 8-half streams) | | | ✓ | ✓ | ✓ | | ✓ | hy3d `h3s.cpp:1-13`; t2 `norm.cpp`; ur `norm.cpp`; hi21 `gn.cpp` |
| Elementwise kernels 4 values a lane with 32-bit indices | | | | ✓ | | | | hy3d `h3s.cpp:10-11` |
| Flash attention in float32, the scores in registers | | | | | | ✓ | ✓ | aam `attn.cpp:62-66`; ur `attn.cpp:63-67` |
| Attention as two oneDNN products and a softmax | | | | | | ✓ | ✓ | aam `attn.cpp:1-6`; 3d.md:68 |
| Half flash attention for self-attention | | | | ✓ | ✓ | | | environment.md:315; architecture.md:510 |
| A device-wide reduction a launch (farthest points) | | | | | | ✓ | | aam `fps.cpp:1-5` |
| Decode attention over the beams' slot pool, whole rows a sub-group | | | | | | | ✓ | ur `decode.cpp` |
| CPU work beside the GPU (mesh, voxel prior, grid) | | | | ✓ | ✓ | | ✓ | 3d.md:37-39, 95, 154-158 |
| oneDNN primitives made once a length (bucketed) | ✓ | | | | ✓ | | | architecture.md:418-420; t2 `mm.cpp:6` |

## What the tricks bought

The measured gains, so a trick can be weighed before it is copied. Before -> after.

**deepseek41** (DeepSeek-V4.1-Flash, B70 + B65, experts partly on the NVMe):

- Read-ahead on 16 threads, a job a slab: UD-Q3_K_XL tg 2.7 -> 4.4 tok/s; read-ahead only on guesses with measured
  chance >= 0.2: IQ2_XXS tg 8.7 -> 14 tok/s (deepseek41-unsloth.md:63-65).
- Prompt chunks of 16,384: pp at 10K tokens 258 -> 391 tok/s, at 24K 311 -> 470; short prompts lose ~8%
  (deepseek41-unsloth.md:66-67).
- Cold skip 0.2 (`-fast`, approximate): tg 12.5 -> 18 tok/s, perplexity +1.8% (deepseek41-unsloth.md:74-75).
- Dense weights requantized to q4_K (`-fast`, approximate): tg ~+2 tok/s, perplexity +1.7% (deepseek41-unsloth.md:77-79).
- On the VQ file (deepseek41.md:63-117): the indexer on XMX pp at 35K 476 -> 581 tok/s; 0.75 GiB free instead of
  1.5: tg +4-9%; next-layer file read-ahead tg 20.4 / 19.9 -> 22.9 / 21.0 tok/s (story / code); GPU routing tg 21.97
  -> 22.44, 20.93 -> 22.33; fused decode chains (11 launches a layer fewer) tg 22.05 -> 23.19, 21.22 -> 22.45; the
  lookup-table VQ variant on the B65 328 -> 263 us a gate | up; candidate-only index scoring kept the B65's 2K chunks
  at 3.3-3.7 s over a 318K prompt (3.5 -> 5.1 before).

**glm5next** (GLM-5.3-Flash, no silo; TODO.md):

- MTP: greedy tg 13.1 -> 15.7 tok/s (TODO.md:48-50); guesses by measured chance tg 18.4 -> 19.4 (TODO.md:98-99); expert
  profile at load: a new topic's first 256 tokens 3,951 -> 3,561 swaps, tg 23.2-23.5 -> 24.0-24.6 (TODO.md:536-540);
  linear-time top-k: tg at temperature 0.7 21.3-21.6 -> 22.7-23.0 (TODO.md:555-557); speculative sampling: tg 20.37 ->
  20.59 average, acceptance 61.6% -> 65.0% (TODO.md:569-575); two requests in one pass: tg 18.1 -> 20.3 in all
  (TODO.md:508-509).
- Prompt: the two GPUs pipelined pp 12K 542 -> 721 tok/s, 36K 527 -> 744 (TODO.md:165-167); KDA scan columns per card
  pp 36K 914 -> 989 (TODO.md:173-176); a GPU's spare share to the next, benchy pp 2K 345 -> 435, 8K 408 -> 762, 40K
  720 -> 1,011 (TODO.md:178-181); 6,144-token chunks benchy pp 40K 1,011 -> 1,096 (TODO.md:195-199); the ESIMD fused
  gate | up pp 36K 1,052-1,063 -> 1,096-1,097 (TODO.md:142-146); the joint_matrix fused down 3.3 -> 2.8 s of GPU time
  at 12K (TODO.md:126-128).
- 128K decode: 32 histogram copies and a decode-width attention kernel, tg 13.0 -> 17.1 tok/s (TODO.md:427-433).

**qwen35** (Qwen3.8-27B): the silo's decode products tg Q4 17.7 -> 22.5-23.5 tok/s at 2K, Q3 12.3 -> 20.4 (TODO.md:334-336),
1.3-4.3x Strata's per type (Q4_K 507 against 366 GB/s, Q3_K 380 / 89, IQ3_S 439 / 118; llm-engines.md:52-58); MTP tg at
2K 26 -> 43 (Q4), 18.5 -> 41.8 (Q8) (TODO.md:337-338); split decode attention on XMX tg at 40K 36 -> 47 (qwen35
`attn.cpp:13`); the shared kernels alone (`NS_SILO=0`, no drafts) pp 441 / tg 18.5 against the silo's 1,184 / 57.4
(llm-engines.md:25, 30-31); two requests at once tg 58.4 in all against 11 + 11 before (architecture.md:226-228).

**qwen4exp** (Qwen3.8-Flash-Next): the silo's int8 DPAS prompt experts pp +40% (Coder) / +20-28% (IQ2_XS) at 2K, +8-10%
at 8K, +4-5% at 40K; B65 2K +49% (architecture.md:191-195); the prompt path pp 12K 142 -> 1,023 tok/s (TODO.md:371-377);
PLE rows asked together: tg 71.8 -> 83.9 with the files out of the page cache (TODO.md:405-410); coupled drafts at
temperature 0.7: tg 56.6 -> 59.6, acceptance 63.6% -> 68.9% (TODO.md:391-395).

**deepseek4**: the fused prompt indexer pp at 256K 700 -> 943 tok/s (environment.md:507); its packed queries 35K
indexer 3.05 -> 1.90 s (environment.md:508); MTP +8-16% (tg 26 at 3.7K), 128 draft experts in VRAM tg 23.6 -> 24.7
(architecture.md:262-271); decode attention chunk of 16 cells 3.3 ms a token against 32 cells 6.0 (ds4 `attn.cpp:35`).

**museglimmer**: DFlash tg 18.7 -> 30-77 tok/s by text (architecture.md:292-295); Q4_K_M tg 25.4 plain, 43 drafted
(architecture.md:300-303); prompt flash attention on XMX (the engine's own kernel was half a 4K prompt's time, mg
`attn.cpp:6-7`).

**nemotronh**: two cards pp 2K 884 / tg 47.0 (IQ1_M) against the B70 alone with host experts pp 447 / tg 21.3
(llm-engines.md:106-113); without drafts tg 30.0-31.8 (llm-engines.md:115).

**deepseek2** (no silo): kv_b in fp16 tg 7.9 against 7.0 kept as Q6_K (environment.md:468); overall tg ~8, pp ~135
against llama.cpp's mmapped 1.0 / 2.8-3.5 (architecture.md:240-243).

**The other kinds**:

- qwen3tts: the frame as one graph with the draws on the GPU, B70 70 -> 82 frames/s in half, 93 -> 125 in int8
  (TODO.md:347-348); the codec's primitives once a length, a short answer 1.1 -> 0.6 s (TODO.md:348).
- hunyuanimage21 (B70): distilled 1024x1024 3.9 -> 2.5 s, 2048x2048 10.3 -> 7.8 s; base 1024 31.8 -> 20.2 s, 2048
  101.1 -> 82.7 s; int8 x int8 (approximate) distilled 2.0 / 5.9 s, base 14.4 / 59.2 s (images.md:72-79).
- hunyuan3d (B70): 2mini 4.1 -> 2.1 s, 2.1 shape 26.6 -> 21.4 s, textured 48.5 -> 36.4 s (3d.md:37-45); int8 steps
  0.38 -> 0.32 s (environment.md:316).
- trellis2 (B70): 1024 cascade 78.7 -> 60.1 s, 512 19.7 -> 11.8 s, Pixal3D 44.9 -> 30.3 s (3d.md:154-158).
- animateanymesh: attention 13x the engine's own kernel; the humanoid 3.1 -> 2.8 s (3d.md:68-72).
- unirig: the humanoid 6.3 -> 3.2 s, a 20K-vertex fox 11.6 -> 3.9 s (3d.md:95-96).
- voxcpm2: B70 22 patches a second (audio.md:345); the docs give no before / after for its silo's graph.

## Tried, no gain

Each with the measurement and where it is written down. A trick here may still be right for another model, but it
should be re-measured, not assumed.

### deepseek41

- **An LRU pinned host tier** (`NS_DS41_HOST_LRU=1`, 2026-10-21): a full host tier gives its least recently used expert
  back to the file to take one moved down from VRAM, instead of keeping the set the profile chose at load. A 2,561-token
  prompt (pp2k), the `-fast` card, 1,024 tokens: tg 12.2 tok/s (LRU) against 12.9 (static). Opt-in, off by default.
  `llm/deepseek41/src/engine.rs:149`; deepseek41-unsloth.md:89; environment.md:523; TODO.md:33; the ladder,
  docs/ladders/deepseek-v4.1-flash-iq2xxs_ladder_stats.md:53.
- **The DSpark drafter** on the Unsloth files: 55% of drafts accepted, but a verify pass touches twice the experts and
  the towers take 6.7 GB of VRAM: tg 5.5 against 10 tok/s, then 15.5-16.3 against 18.3 with cheap cold reads
  (deepseek41-unsloth.md:85-86; TODO.md:23). On the VQ file: code tg 21.2 without drafts, 20.6 with 1, 20.4 with 2;
  prose 22.5 without, 16.8 with 1 (deepseek41.md:131-137).
- **O_DIRECT expert reads**: one cold expert 3.7 -> 2.5 ms, but decode loses the page cache's hits on repeating cold
  experts: tg 13.6 -> 12.7 tok/s (deepseek41-unsloth.md:86-87; environment.md:536).
- **More reader threads, more pinned memory, a smaller VRAM reserve, other prefetch thresholds**: all within the
  run-to-run noise of +-5% (deepseek41-unsloth.md:88-89).
- **The grouped `o_a` in one launch**: 98 against 138 us alone, no gain end to end (deepseek41-unsloth.md:89;
  environment.md:542).
- **Unrolling the expert kernel over blocks**: register spills (deepseek41-unsloth.md:89-90).
- **Rescaling the kept experts' weights after a cold skip** (`NS_DS41_COLD_RENORM`): worse perplexity; a skip
  threshold of 0.4 is a cliff (+11%) (deepseek41-unsloth.md:76).
- **More layers on the B70** (`NS_SPLIT=24` / `28`): pp at 8.7K 409-493 / 382-412 against 485 tok/s, tg 17-18 / 16
  against 19 (environment.md:516; deepseek41.md:86-87).
- On the VQ file (deepseek41.md:83-89, 124-127): the XMX expert product with 2 or 4 row tiles and 8 / 16 K slices
  (within noise end to end); the q4_K nibbles to floats by their bits (the B65, one column, no faster); a short
  prompt's next-layer file reads from its router's guess (0.6-0.7 s either way); the VQ lookup table's run count;
  decode's file reads after the resident experts' products (no gain: a read outlasts those products); spin-waiting on
  the router's read-back (no gain); the indexer's queries packed in global memory with 16 key tiles (pp at 35K 490 against
  599 tok/s); a grouped one-column q4_K product for q_b (B65 93 -> 124 us); `UR_L0_USE_DRIVER_INORDER_LISTS=1`
  (nothing, deepseek41.md:121).
- **What does not show in tg after a prompt** (2026-10-21, not a trick but the reason several of the above did not
  move the first tokens): after a technical-docs prompt, tg is ~11-13 tok/s (`-fast`) in the first 256-1,024 tokens,
  against ~23 sustained over a 12,000-token prose novel. The expert cache has to warm to the text; until it has, the
  drive's reads dominate (6-12 s of waits per 256 tokens; attention is ~2 ms of a ~70 ms token).
  deepseek41-unsloth.md:42-43; ladder `Reading it` (line 37); TODO.md:34-35.

### glm5next

- **Fused IQ2_XXS gate | up on joint_matrix, first form**: 223 us a call against 215 for expand + oneMKL in a 12K
  prompt (TODO.md:114-115). The cheaper decode that followed won only up to 64 tokens an expert (TODO.md:122-125).
- **The grid table in local memory; B staged through local memory**: no help in the fused kernels (164-196 us against
  144-146 for the register fill) (TODO.md:119-120, 140).
- **ESIMD fused decoding, a tile a thread**: exact but slower - B70 64 / 128 / 256 tokens 181 / 278 / 402 us against
  expand + oneMKL 318 / 244 / 242 and joint_matrix's 110 / 135 / 200 (TODO.md:134-139). The work-group-shared form won
  (above).
- **The ESIMD down projection (Q2_K)**: B70 48 / 52 / 69 us against the joint_matrix down's 39 / 40 / 55 (TODO.md:147-150).
- **The expansion beside the GEMMs on a side queue**: pp at 36K 919-927 against 995 tok/s; reverted (TODO.md:157-161).
- **More column tiles a sub-group in the fused gate / up** (NB 2 slower, NB 4 spills) or limits past 64 (128 the same
  7.5 s, 192-256 8.1) (TODO.md:200-201).
- **Smaller prompt chunks** (at 12K, 2,048-token chunks pp 585, 3,072 pp 609, against the pipeline's 721 at 4,096) and **equal chunks** (pp at
  40K 975 against 1,011) (TODO.md:168-169, 188).
- **Prompt-lookup drafts** (`NS_NGRAM`): ~90% accepted on a copy-heavy answer, yet tg 17.46 tok/s without against
  16.85 / 17.13 / 16.87 at K = 2 / 3 / 4: a 5-row pass ~280 ms against a 2-row one's ~110 (TODO.md:558-563; lessons.md:26).
- **Two chained drafts** (`NS_DRAFTS=2`): 2.36 tokens a pass against 1.83, but tg 18.4 against 19.1 (TODO.md:615-617).
- **A sharper draft distribution** (`NS_SPEC_DRAFT_TEMP` 0.5 / 0.25): acceptance 65.8% / 64.1%, within noise
  (TODO.md:574).
- **Pinning hot experts by a usage profile** (simulated on another topic's profile): 25% pinned +12% misses, 90% +140%
  (TODO.md:536-537).
- **Guessing two layers ahead**: 93% at rank 1 against 96% one layer ahead (TODO.md:99).
- **Decode reading host-slot experts in place** (`NS_DECODE_DIRECT=1`): tg 11.2 against 12.5 tok/s (TODO.md:608-609).
- **Missed experts on the CPU** (a 9950X, AVX2): 138 us an expert-token against ~270 us for its copy, but every miss
  made free (`NS_FREE_MISSES=1`) is only tg 18.8-19.0 -> 21.3-21.6 (+13%); not built (TODO.md:541-547; lessons.md:7).
- **Lifting the B65's Q8_0 products to the B70's share of bandwidth**: ~1-2% of decode, not pursued (TODO.md:548-552).
- **4 heads a work-group in prompt MLA attention** (`NS_MLA_HG=4`): measured slower (environment.md:125).
- **Arriving experts launched in two phases** (`NS_SPLIT_COPY`): +1% only, kept on (TODO.md:458-459).

### qwen35

- **SYCL graphs for decode** (`NS_Q35_GRAPHS=1`): the same tokens, no faster on the B70 / B65 - the host keeps ahead of
  the GPU (environment.md:201).
- **4 or 5 drafts a step**: slower than 3; 2 as fast (llm-engines.md:34-35).
- **Several rows a sub-group in the decode products**: one row wins, per type (qwen35 `mmvq.cpp:9`).
- **Bigger flash-attention chunks**: a 40K prompt with 32 keys a chunk 76 s, 64 keys 106 s, 16 keys 50 s; padding the
  packed K rows no faster at 16 (qwen35 `attn.cpp:31-32, 40`).
- **Two cards for speed**: they are for room - the B65 runs a layer ~1.5x slower (llm-engines.md:38-51).
- A quantized cache costs speed for room: at 43K pp 673 / tg 38.9 (q8) against 742 / 43.0 (f16) (llm-engines.md:72-73).

### qwen4exp

- **Both cards** (every expert in VRAM): slower than the B70 alone - Coder tg 70.8 / 69.6 / 70.7 (B70 first) and 58.2 /
  54.7 / 53.5 (B65 first) against 77.5 / 75.9 / 77.2; IQ2_XS 70.6 / 68.5 / 66.8 against 83.0 / 81.3 / 75.9
  (TODO.md:414-417).
- **Touching every expert at load** (`ns_touch`) for the first pass: tried and removed - the cost was the PLE rows
  (TODO.md:410).
- Strata's own A/B switches left off by default: `STRATA_GR_DOWN_MAX4` (no measured gain),
  `STRATA_IQ_STAGE_GRID_MMVQ` (slower at few columns) (environment.md:382, 399).

### deepseek4

- **A second chained draft** (`NS_DS4_DRAFTS=2`): a poem tg 27.7 -> 22.7 tok/s, a 3.7K summary 26.0 -> 25.5; the
  second draft kept ~45% of the time (environment.md:505).

### museglimmer

- **dp4a decode products** for a 16-row verify pass: compute-bound, a 15-draft step cost 1.7 one-row passes - replaced
  by the XMX products (mg `mmvq.cpp:8-9`).
- **The target's layers on the B65**: moving 6 of its 52 layers there cost more than putting the DFlash drafter there; the
  target stays on the B70 (architecture.md:290-291).

### deepseek2

- **kv_b kept as Q6_K**: 1.2 GB more for experts and 7% fewer file reads, but tg 7.0 against fp16's 7.9 tok/s
  (environment.md:468).

### The other kinds

- **hunyuan3d**: int8 on 2mini, no faster (environment.md:316).
- **hunyuanimage21**: `NS_HI_INT8` is faster but approximate (PSNR 23.5 dB in TODO.md:479, 24.7 dB at 2048 in
  environment.md:215): "fix or drop it".
- **qwenimage21** (no silo): ARK's persistent flash schedule (sycl-tla rejects the shape) and SageAttention int8 q / k
  (4.0 ms with its quantize passes, 10x the error) (TODO.md:249-251).
- **minimaxmusic3** (no silo): the flow stage beside the frames on a second queue of the same card, 118 -> 114.5 s;
  dropped (TODO.md:291-292).
- **breezetts2** (no silo): two servers on one card, 1.54x real time alone against 0.65x for the pair (audio.md:172-174).

## Still open

Ranked by the payoff the docs give, highest first. Where a trick already works in one silo, it says which and why it
might carry. lessons.md:7 applies to all of it: bound the prize (make the cost free and time it) before building.

### Carried from one silo to another

1. **deepseek41: one MoE pass over a group of chunks, as deepseek2 does with its spans.** deepseek2 runs a prompt a
   layer at a time over spans of 4,096 tokens, every expert loaded once a span (architecture.md:240-243,
   `NS_DS2_SPAN`). deepseek41's prompt time is the non-resident experts' bytes over the drive, re-read every chunk
   (2,561 tokens read ~72 GB; deepseek41-unsloth.md:57-59). TODO.md:36-37 expects long prompts 2-4x; the group's
   streams past 8K tokens need pinned host spill.
2. **deepseek41: warm the tiers from the prompt's own routing.** tg is ~11-13 tok/s (`-fast`) for the first ~1K tokens
   after a technical prompt and ~23 sustained on prose once the cache has warmed (2026-10-21; TODO.md:34-35). The prompt
   passes already read those experts from the file. The LRU host tier did not do it (tg 12.2 against 12.9), so the
   lever is what the tiers hold when decode starts, not the eviction rule. Up to the ~2x gap between the two numbers.
3. **The cold skip to deepseek2 and glm5next / deepseek4.** deepseek41's `NS_DS41_COLD_SKIP=0.2` took tg 12.5 -> 18
   (+1.8% perplexity; deepseek41-unsloth.md:74-75). deepseek2 also streams experts from the file and decodes at tg ~8
   (architecture.md:240), so the same skip should pay most there. On glm5next the prize is bounded by
   `NS_FREE_MISSES`: +13% (TODO.md:541-547); deepseek4 already leaves a missing expert out of its drafts
   (architecture.md:262-266). An approximation: measure with `generate --score`.
4. **deepseek41: the deepseek4 silo's expert decode kernels for the Unsloth types.** deepseek41's `std.cpp` reads its
   2-byte aligned blocks a byte pair at a time (`std.cpp:47-49`) against float32 activations. deepseek4's `mmvq.cpp`
   (from deepseek2's: split Q8 activations, dp4a, aligned 16-byte loads, grids in local memory) already takes IQ2_XXS,
   IQ2_S, IQ3_XXS and MXFP4 - the IQ2_XXS file's main types. On Xe2 the same IQ3_S kernel ran at 132 GB/s with small
   loads and 439 with aligned 16-byte loads (lessons.md:43). The prize: the expert products, ~8-13 ms of a ~72 ms token
   (TODO.md:25-27; deepseek41-unsloth.md:55-56) - a larger share once the cache is warm.
5. **deepseek41: fuse the small dense launches of a layer** (q_a | kv | compressor | indexer, the shared expert's gate |
   up): ~5 ms a token (TODO.md:38). The VQ file's fused chains took tg 22.05 -> 23.19 (deepseek41.md:108-112).
6. **The q4_K requantization to glm5next and deepseek4's dense matrices.** deepseek41's `NS_DS41_DENSE_Q4K` halves
   the bytes a token reads (tg ~+2, perplexity +1.7%; deepseek41-unsloth.md:77-79). On glm5next, KDA is the
   largest single part of a decode token (10.6-10.9 ms; TODO.md:455-456, 576) and its Q8_0 projections already run at
   80-90% of the B70's bandwidth (TODO.md:548-550), so only fewer bytes make them faster: the products' share of those
   ms is the prize (time it with `NS_PROFILE=gpu` first). An approximation; opt-in, as on deepseek41.
7. **nemotronh on the B70 alone: deepseek41's VRAM staging ring and one XMX launch for host experts.** Its prompt
   passes there still expand experts one at a time (llm-engines.md:116-117): pp 447 at 2K against 884 on two cards.
   deepseek41's ring copies host experts up a wave at a time while the resident ones compute (deepseek41.md:66-68).
8. **nemotronh: its expert up products at decode** run at ~150 GB/s on the B70, a third of its dense ones, and neither
   the grid lookups nor the index math is the cause (llm-engines.md:117-118). museglimmer's XMX int8 decode products
   (`mg mmvq.cpp`) and the decode-once-per-tile of qwen4exp's `i8moe.cpp` are the two forms in the repo that move the
   arithmetic off the vector units; measure first with `NS_NH_PROFILE=1`.
9. **glm5next: deepseek41's staging ring (and Strata's lend mirror) for the prompt.** TODO.md:79-88 lists "borrow
   expert slots" (Strata: pp +15-40%, refill 720-2,240 -> 200-380 ms) and "a stager" (the in-place PCIe reads sit on the
   critical path); deepseek41 has the ring working (deepseek41.md:66-68).
10. **glm5next: deepseek4's fused indexer for decode at depth.** At 128K a layer's indexer projections take ~300 us and
    attention 444 us (TODO.md:434); the fused indexer is open there (TODO.md:447). deepseek4's took pp at 256K 700 ->
    943 tok/s (environment.md:507) and deepseek41's copy pp 35K 476 -> 581 (deepseek41.md:75).
11. **trellis2 (and qwenimage21): hunyuanimage21's int8 weight read inside oneDNN.** trellis2 still expands Q8_0 to half
    per product (architecture.md:512); hunyuanimage21's silo reads the Q8_0 matrix as int8 and widens it in the product,
    "as fast as the half product itself", the expansion pass gone (hi21 `mm.cpp:1-5`). hunyuanimage21's whole silo took
    distilled 1024x1024 3.9 -> 2.5 s (images.md:72).
12. **The audio engines without a silo: qwen3tts's frame graph and GPU draws.** qwen3tts got B70 70 -> 82 frames/s
    (half) and 93 -> 125 (int8) from them (TODO.md:347-348). Open in the others: Breeze TTS 2 "int8 / graphs" (22 / 28
    frames/s; TODO.md:308-309), S2 Pro "sampling on the GPU, graphs for the fast model" (42 frames/s; TODO.md:323),
    Voxtral "the flow in int8 or a per-frame graph" (30 frames/s; TODO.md:329-330), VibeVoice "SYCL graphs for the
    tokenizers' ~500 small launches a frame" (graphs <= 0.5 ms of 12.6 ms a frame; TODO.md:317-319), Music "the frame's
    eight host round trips" (TODO.md:289-292).
13. **qwen35: the nemotronh silo's Q6_K decode kernel for the head.** "Q6_K head in the silo" is open (TODO.md:342);
    nemotronh's silo added Q6_K to the qwen35 kernels it copied, ~520 GB/s on the B70 (llm-engines.md:92-96).

### Open in one silo

14. **glm5next: missed experts read straight from pinned host memory in the expert kernel** (Strata: tg 2.6 -> 40.9 on
    its B70; TODO.md:610-614). Here the prize is bounded by the +13% of free misses (TODO.md:541-547); the swaps were ~1/3
    of decode time (TODO.md:97).
15. **glm5next: swap the cards' slots** (the B70 on x16, `NS_SPLIT=15`): simulated copy time a pass 5.9 ms against 8.8
    at today's best (TODO.md:504-507). Same +13% ceiling.
16. **deepseek41: re-quantize the cold tail of the experts** (an imatrix from Unsloth) to fit the resident tiers - big,
    a quality risk (TODO.md:40). The VQ file showed how far is too far: ~1.6-bit experts degraded long answers
    (deepseek41-unsloth.md:10-15).
17. **deepseek41: SYCL graphs for the attention half of decode** (deepseek41.md:175-176): ~700 kernels a token a GPU,
    ~6 ms of gaps between them (deepseek41.md:119-120). qwen4exp and qwen3tts use graphs; on qwen35 they gave nothing
    because the host kept ahead (environment.md:201) - check device time against wall time first (lessons.md:33).
18. **deepseek41: a better guess than GUESS_HIT for read-ahead**, the profile from many more prompts (TODO.md:39); one
    recalibration already took tg 8.7 -> 14 (deepseek41-unsloth.md:63-65).
19. **deepseek41: one decode for all of an expert's tokens in the prompt's XMX product** (local memory, many token tiles;
    deepseek41.md:180-181). glm5next's work-group-shared ESIMD decode is the model (TODO.md:140-146), and qwen4exp's
    int8 DPAS (pp +20-49% at 2K) another. Only worth it once item 1 stops the drive from bounding the prompt.
20. **glm5next: the B65's prompt stage**: the B65's stage is the longer one (its MLA attention 4.2 s at
    36K; TODO.md:177), the ESIMD gate | up past 128 tokens on the B65 (TODO.md:151), the shares by need both ways
    (TODO.md:182).
21. **glm5next: SYCL graph replay for the decode step** (TODO.md:618) - but the GPUs are busy ~all of a decode, so
    routing on the device or graph capture "would save little" (TODO.md:533-535).
22. **glm5next: the indexer's pooled keys in fp16, a 4-bit latent** (TODO.md:446).
23. **qwen4exp: the decode window, 2 ms behind Strata's** (here ~35 ms a round against Strata's 36.6 ms with drafting
    included, the window 30.5 ms against ~28.7; TODO.md:401-404).
24. **qwen35: two cards for speed** (each matrix split over both, two crossings a layer; TODO.md:342-343).
25. **qwen3tts: the predictor's attention inside its o-projection**, ~4% a frame (TODO.md:349-350).
26. **voxcpm2: the guided and unguided rows batched through the products, the local attention fused** (TODO.md:344-345;
    the graph part of that item is done in its silo, `vcs.h`).
27. **Qwen-Image 2.1 (no silo): the small passes fused** (norm + modulate, gate add, SwiGLU, RMS + RoPE: ~1.7 ms a
    block, ~13%; TODO.md:252). trellis2's and hunyuanimage21's oneDNN post-ops are the pattern.
28. **trellis2: measure `--opt-sage`** and fix its help text, which quotes unmeasured numbers (TODO.md:470).
29. **hunyuanimage21: fix or drop the int8 mode** (TODO.md:479).
30. **deepseek41: the Level Zero v2 adapter in the cards** (+1-5% measured, the server not yet run with it;
    deepseek41.md:121-122).

A small mismatch found on the way: environment.md:543 gives `NS_DS41_STD_KU` a default of 2, while `std.cpp:351` reads
1 when the variable is unset.
