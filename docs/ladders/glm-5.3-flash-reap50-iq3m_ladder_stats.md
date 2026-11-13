# glm-5.3-flash-reap50-iq3m: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for GLM-5.3-Flash REAP-50 IQ3_M (patrickbdevaney's
REAP-pruned GGUF, 144 of 288 routed experts, 67.2 GiB, its MTP block), engine `llm/glm5next` with its silo, measured
2026-11-13 on hardac1. Port notes and what bounds decode: [docs/ports/glm53-reap50.md](../ports/glm53-reap50.md).

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (layers 0-21) + Arc Pro B65 (layers 22-44), 32 GB each, `--gpu 0,1` as served (`gpus: all`) |
| model file | Crucial P310 4 TB NVMe, `glm53-reap50/GLM-5.3-Flash-REAP50-IQ3_M.gguf` |
| code | nextsycl main at `2ce40c7` (the silo IQ3_S decode product) |
| expert places at 64K | B70 1,877 in VRAM (18.9 GiB) + layers 3-4 whole (3.2 GiB) + 571 pinned; B65 2,117 in VRAM (21.3 GiB) + 1,339 pinned; none on the file only |
| settings | `llm generate --temp 0 --max 256`, MTP drafts on, each rung a fresh process (the load, 15-19 s, not counted), behind an idle check of both cards |

Prompts: this repository's documentation (`pp2k`, `pp8k`, `pp24k`) and the same with a buried secret code and a question
at the end (`needle*`), the prompts of the uncensored GLM ladder.

## Results

| prompt tokens | context | pp | tg | drafts accepted | needle |
|---:|---:|---:|---:|---:|---|
| 13 | 64K | 15.9 tok/s (0.8 s) | 17.48 tok/s | 86 / 169 | - |
| 2,533 | 64K | 472 tok/s (5.4 s) | 19.67 tok/s | 113 / 142 | - |
| 9,744 | 64K | 816 tok/s (11.9 s) | 17.57 tok/s | 97 / 158 | - |
| 24,594 | 64K | 960 tok/s (25.6 s) | 18.40 tok/s | 109 / 146 | - |
| 9,785 | 64K | 820 tok/s (11.9 s) | (21-token answer) | 10 / 11 | found |
| 34,776 | 64K | 1,022 tok/s (34.0 s) | 19.07 tok/s (`--ignore-eos`) | 112 / 143 | found |
| 248,495 | 256K | 871 tok/s (285 s) | 15.62 tok/s (`--ignore-eos`) | 107 / 148 | found |
| 1,018,142 | 1M | 482 tok/s (2,111 s) | (11-token answer) | 4 / 7 | found |

The context reserve alone (a 31-token prompt, 256 generated): tg 19.03 tok/s with 256K held back, 16.52 with 1M. The
1M reserve (7.9 GiB over both cards) leaves 3,261 experts in VRAM against 3,994 at 64K. One run of each rung; the
needle answers are the code alone and are not a decode rate.

## Reading it

- **pp** climbs from 472 tok/s at 2.5K to ~1,000 at 35K, holds 871 at 248K and falls to 482 at 1M: a 256K prompt
  reads in under 5 minutes, a 1M one in 35.
- **tg** is 17.5-19.7 tok/s to 35K, 15.6 after a 248K prompt. The 1M card is ~13% slower than the 64K one even on short
  chats (fewer experts fit beside its reserve): pick it for whole-codebase work, the 64K card for chat.
- The secret code was found at 10K, 35K, 248K and 1,018K tokens.
- Against the uncensored IQ2 card's ladder: pp 472 / 960 vs 471 / 872 at 2.5K / 25K, tg 17.5-19.7 vs 15.6-18.9.

## The other REAP-50 files (same day, `--gpu 0,1`)

| file | tg short, no MTP | 1K pp / tg | 8K pp / tg | perplexity (512 tokens) | experts on the file only |
|---|---:|---:|---:|---:|---:|
| IQ3_M (67.2 GiB) | 19.17 | 338 / 20.68 | 799 / 18.39 | 46.75 | 0 |
| Q4_K_S (87.1 GiB) | 16.95 | 313 / 14.45 | 794 / 12.89 | 42.95 | 0 |
| Q4_K_M (92.5 GiB) | 15.17 | 230 / 7.63 | 212 / 7.80 | 43.20 | 94-259 |

All three are exact (spec-check: 0 of 94 rows differ). The IQ3_M rows were measured with `--gpu 1,0`. Q4_K_S scores
8% lower perplexity at ~30% slower decode with drafts (its 13.5 MiB experts fit 2,982 in VRAM against 3,994); Q4_K_M
spills experts to the file and halves again.
