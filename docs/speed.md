# Speed

[← nextsycl](../README.md)

# Speed by model

Benchy v1 through the server (`nextsycl llm bench --sizes 20,2185,8000,40000,128000 --parallel 1`; 256 greedy tokens a
size, the prompt cache cleared before each), 8 October 2026. The Qwen3.8-Flash-Next family on one Arc Pro B70 (its
cold experts in pinned host memory, the MTP draft layer); GLM-5.3-Flash on the B70 and an Arc Pro B65. Input tokens
as each model's tokenizer counts them (Qwen 30 / 2,195 / 7,914 / 39,783 / 127,303; GLM 31 / 2,216 / 7,975 / 39,758 /
127,196 / 248,342). Strata's rows are its own numbers on the same B70 (its INTEL_PERFORMANCE.md, benchy v1).

Prompt read (PP), tokens/s:

| Model | 20 | 2K | 8K | 40K | 128K | 250K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next IQ2_XS | 84 | 789 | 1,154 | 1,226 | 1,210 | - |
| IQ2_XS, refusal projection | 83 | 786 | 1,143 | 1,212 | 1,199 | - |
| Qwen3.8-Flash-Next Coder IQ1_M | 80 | 938 | 1,389 | 1,495 | 1,471 | - |
| Coder, refusal projection | 80 | 930 | 1,371 | 1,473 | 1,450 | - |
| Swift 1.5 IQ2_XS | 83 | 782 | 1,147 | 1,217 | 1,202 | - |
| GLM-5.3-Flash IQ2 (64K / 256K modes) | 27 | 456 | 759 | 1,108 | 1,134 | 1,026 |
| Strata: IQ2_XS | 60 | 745 | 1,050 | 1,117 | - | - |
| Strata: Coder IQ1_M | 57 | 875 | 1,285 | 1,392 | - | - |
| Strata: Swift 1.5 | 59 | 742 | 1,046 | 1,108 | - | - |

Decode (TG), tokens/s: warm (the same prompts a second time on one load), and in brackets the first pass after the
model loads:

| Model | 20 | 2K | 8K | 40K | 128K | 250K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next IQ2_XS | 77.2 (75.6) | 82.7 (78.3) | 81.2 (74.4) | 75.6 (72.2) | 75.5 (69.9) | - |
| IQ2_XS, refusal projection | 78.3 (76.9) | 84.7 (77.6) | 82.5 (75.6) | 77.8 (72.0) | 72.3 (67.4) | - |
| Qwen3.8-Flash-Next Coder IQ1_M | 78.0 (71.5) | 77.5 (71.0) | 75.9 (70.9) | 77.0 (71.3) | 67.9 (62.9) | - |
| Coder, refusal projection | 78.4 (72.1) | 78.2 (72.0) | 69.8 (65.1) | 76.0 (70.8) | 71.6 (67.1) | - |
| Swift 1.5 IQ2_XS | 79.3 (77.7) | 85.8 (79.7) | 86.1 (78.7) | 82.4 (78.3) | 75.2 (69.4) | - |
| GLM-5.3-Flash IQ2 (64K / 256K modes) | (18.8) | (18.2) | (18.7) | (18.1) | (18.7) | (16.9) |
| Strata: IQ2_XS | 70.0 | 77.5 | 78.5 | 71.8 | - | - |
| Strata: Coder IQ1_M | 77.3 | 74.6 | 71.9 | 67.6 | - | - |
| Strata: Swift 1.5 | 77.1 | 74.8 | 75.5 | 63.1 | - | - |

The prompt speeds are the warm pass's (the first pass reads the same, but for its very first request: ~50 tokens/s
on the 20-token prompt). A model just loaded decodes 2-9% slower than the warm pass on new text: each
token's 16 per-layer-embedding rows come from a 27 GiB table in the file, and rows not in the page cache are read
from the disk (asked for together a window: `docs/architecture.md`, NS_QW_PLE_PREFETCH). The
projection modes cost nothing measurable. Draft acceptance varies with the text (61-86% here), and decode with it. On two cards the Qwen models decode slower than on
the B70 alone (Coder 2K 70.8 tok/s with the B70 first, 58.2 with the B65 first; `TODO.md`).

# Speed

GLM-5.3-Flash IQ2 (the ds4 file, 80 GB of experts) on an Arc Pro B70 and an Arc Pro B65, 7-8 October 2026, through
the server:

| | |
|---|---|
| prompt, 2K tokens (one chunk) | ~430 tokens/s |
| prompt, 8K tokens | ~740 tokens/s |
| prompt, 12K tokens | ~820 tokens/s |
| prompt, 36-40K tokens | ~1,100 tokens/s |
| prompt, 128K tokens | ~1,000 tokens/s (126 s) |
| prompt, 253K tokens | ~1,020 tokens/s (249 s) |
| decode, short context, MTP on | ~18-23 tokens/s by the text (70-90% of drafts accepted), greedy or at a temperature |
| decode at 128K / 253K of context | ~17 / ~15 tokens/s |
| a follow-up on a 253K document | ~4.5 s (its checkpoint mounted, in memory or from disk) |
| a chat sent while a 253K prompt is read | answered in ~10 s (the read stops for it at a chunk's end) |
| power while decoding | ~175-185 W for both cards (~1.6-1.9 kJ for a 170-200-token answer) |
| idle | ~9 W for both cards |
| load | ~18 s |

About 60% of the experts fit in VRAM (plus ~380 slots a GPU lent by the prompt arena while decode runs); the rest sit
in pinned host memory. At load VRAM takes the experts decode asked for most in earlier runs first (the expert profile
a stop saves beside the prompt cache: a new topic's first answer ~10% fewer swaps). At decode a missed expert is
swapped in over PCIe (both directions at once, on two copy queues, while the resident experts compute; the next
layer's likely experts prefetched). The copies are mostly hidden now: with every miss made free (a measurement,
`NS_FREE_MISSES=1`) decode would be at most 13% faster - its time is the GPUs' own work, spread over the experts'
kernels, KDA (~11 ms a token), MLA (~5), the draft block and the shared expert (`TODO.md`). `--gpu all` puts the GPU
that computes most last (it takes the head and the draft block); `NS_SPLIT` sets the layers per GPU. A prompt longer
than a chunk (6,144 tokens) runs the two GPUs as a pipeline: the first reads chunk n+1 while the second finishes
chunk n.

# Benchy

`nextsycl llm bench` runs Strata's benchy v1 (its prompts as text in `bench/v1`) against the running server, and sends
several requests at once. The latest run, `docs/benchy/v1-2026-10-07-q8.md` (B65 + B70, the B70 last; prompt chunks of
6,144, the q8 latent cache, the GPUs in a pipeline for prompts over a chunk):

| Input tokens | PP (tok/s) | TTFT (s) | TG (tok/s) | Drafts accepted | Energy (J) | Avg power (W) |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | - | 1.3 | 18.0 | 91% | 1,359 | 159 |
| 2,216 | 456 | 5.2 | 18.1 | 76% | 3,380 | 178 |
| 7,975 | 759 | 10.9 | 18.7 | 77% | 5,199 | 215 |
| 39,758 | 1,108 | 36.3 | 18.1 | 76% | 14,978 | 299 |

Earlier: `v1-2026-10-07-mirror.md` (chunks of 4,096, fp16 latents) 435 / 762 / 1,011 tokens/s at 2K / 8K / 40K;
`v1-2026-10-06-prefetch.md` 332 / 314 / 416. Benchy decodes greedily; sampling at a temperature runs within a few
percent of it since the sampler selects its top 256 in linear time.

Several at once, before and after batched decode (`docs/benchy/v1-2026-10-06-parallel.md`; the short prompt, 256
tokens each):

| Clients at once | Tokens/s, all: one at a time | batched (NS_PARALLEL=2) | First token mean / max: one at a time | batched |
|---:|---:|---:|---:|---:|
| 1 | 17.6 | 17.0 | 1.1 / 1.1 s | 1.0 / 1.0 s |
| 2 | 18.1 | 20.3 | 5.7 / 10.4 s | 2.0 / 2.0 s |
| 4 | 17.9 | 19.8 | 15.8 / 29.1 s | 14.8 / 27.5 s (two at a time) |

`nextsycl batch-check` (the engine alone, no draft block): 2 conversations together 23-30 tokens/s in all against
16-18 one at a time (over 32 to 128 tokens: the longer, the more their experts differ), the tokens and logits
exactly those of each alone.

With NS_PARALLEL=2 two requests decode in one pass; a third waits for a free session. The batch's limits today: MLA
runs per session (its projections read once per conversation), and two conversations want more distinct experts a
layer (more swaps).
