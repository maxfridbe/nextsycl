# deepseek-v4.1-flash-iq2xxs: context ladder

Prompt processing (pp) and token generation (tg) by prompt length, for the cards `deepseek-v4.1-flash-iq2xxs` (exact)
and `deepseek-v4.1-flash-iq2xxs-fast`, measured 2026-10-21. Port notes: [docs/ports/deepseek41-unsloth.md](../ports/deepseek41-unsloth.md).

## The machine and the silo

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (PCI id 0xe223, layers 0-19) + Arc Pro B65 (0xe222, layers 20-39), 32 GB each, xe driver |
| host | AMD Ryzen 9 9950X (16 cores / 32 threads), 61 GB RAM, Fedora 44, kernel 7.2.8 |
| model files | Crucial P310 4 TB NVMe (CT4000P310SSD8), ~5 GB/s reads; 146 GiB of experts + the 126 GiB Engram table in the last shard |
| stack | oneAPI DPC++ 2026.1.0 (20260617), Level Zero 1.28.6, compute runtime 26.31.39395.13, in the `h3-build` image |
| silo | `dist/silo/libnextsycl-deepseek41.so` (14.9 MB) built from `kernels/llm/deepseek41/silo/` (`std.cpp` the ggml-type expert kernels, `q4k.cpp`, `attn.cpp`, `ds41.cpp`, `k41.cpp`), `NS_SILO_REQUIRED=1` |
| code | nextsycl main at `5a2f514` |
| expert tiers | ~2,340 experts in VRAM (~21.6 GiB over both cards), ~4,270 in pinned host memory (~40 GiB), ~8,750 on the file only (of 15,360); VRAM filled from the expert profile `prof-base.txt` (prose) |
| settings | `NS_PREFILL_CHUNK=16384`, `NS_DS41_FREE_GIB=1.5`, `NS_DS41_VISION=0`, greedy (`--temp 0`), no drafter (`--no-mtp`), `llm generate --gpu 0,1` |
| `-fast` | `NS_DS41_COLD_SKIP=0.2 NS_DS41_DENSE_Q4K=1 NS_DS41_COLD_SKIP_PROMPT=0.3 NS_DS41_COLD_SKIP_PROMPT_MAX=8` |

Each rung is a fresh process (the load, ~50 s, not counted). Prompts: documentation and source of this repo (`pp2k`,
`pp8k`, `pp24k`), and the same with a buried secret code and a question at the end (`needle*`).

## Results

| prompt tokens | pp exact | pp `-fast` | tg exact | tg `-fast` | tokens generated | needle |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 5.6 tok/s (2.8 s) | 9.8 tok/s (1.6 s) | 6.1 tok/s | 11.8 tok/s | 7 | - |
| 2,561 | 125 tok/s (20.5 s) | 96-187 tok/s (13.7-26.6 s) | 7.7-9.4 tok/s | 7.3-12.9 tok/s | 256-1,024 | - |
| 9,901 | 397 tok/s (24.9 s) | 378-413 tok/s (24-26 s) | 7.1 tok/s | 12.1-12.3 tok/s | 256 | - |
| 25,142 | 442 tok/s (56.8 s) | 478 tok/s (52.6 s) | 8.3 tok/s | 11.6 tok/s | 256 | - |
| 35,391 | 393-430 tok/s (82-90 s) | 460-471 tok/s (75-77 s) | 6.9 tok/s | 11.7 tok/s | 320 | found (both) |
| 139,561 | 448-473 tok/s (295-311 s) | 474 tok/s (294.5 s) | 7.7 tok/s | not run with a long answer | 320 | found (both) |

Ranges are repeated runs of the same rung (the 2,561 rung six times for `-fast`, twice exact): the drive's page cache and which
experts the previous run left warm move a run by up to ±25%.

## Reading it

- **pp** rises with the prompt to ~450-475 tok/s from ~10K tokens on and stays there at 140K: a prompt touches nearly
  every expert of every layer, so a chunk's time is the non-resident experts' bytes over the drive (~5 GB/s). Short
  prompts are slow for the same reason (2,561 tokens still read ~72 GB).
- **tg** in the first few hundred tokens after a prompt is 7-9 tok/s exact and 11-13 tok/s `-fast`, almost flat with
  depth (attention is ~2 ms of a ~70 ms token): it is bound by experts read from the drive (6-12 s of waits per 256
  tokens), not by context.
- The higher tg figures elsewhere (~14 exact, ~23-24 `-fast`) are **sustained** decode of prose (a 12,000-token novel:
  tg 23.1 tok/s), where the expert cache has warmed to the text and the VRAM tier matches the prose profile. A fresh
  prose prompt starts at tg ~14.5 (`-fast`, first 256 tokens).
- The secret code was found at 35K and 140K tokens in both modes.
- Not run: 256K and 1M. With tg under 15 tok/s at every measured depth, the deep rungs were stopped.
- Memory, chunk 16384: from 25K tokens on both arenas sit at their 8 GiB peak (prompt passes spill past them); at 25K
  the cards peaked at 27.9 GB (4.8 GB free).

## Tried during the ladder

- `NS_DS41_HOST_LRU=1` (the pinned host tier evicts its least recently used expert to take one moved down from VRAM,
  instead of keeping the profile's set): 2,561-token prompt, `-fast`, 1,024 tokens: tg 12.2 against 12.9 tok/s static.
  No gain; off by default.

## Since the ladder (2026-10-22, rungs not rerun)

The split-Q8 dp4a expert products (`mmvq.cpp`), the prompt passes' VRAM reserve lent to the expert tiers during decode
(`NS_DS41_LEND_RESERVE`), layers split by compute units (24 / 16) and read-ahead only on guesses above 0.4
(`NS_DS41_FILE_AHEAD_P`): tg over 512 tokens after a short prose prompt 9.7 -> 12.7 tok/s exact and 16.1 -> 21.4
`-fast` (medians of 3), and 23.1 -> 25.3 tok/s `-fast` sustained over the 12,000-token novel. A 35K-token needle prompt:
pp 418-442 tok/s `-fast`, the code found.

## Rerun

`~/src/run_tg.sh RESULTS "name file ctx max" ...` on hardac1 (exact then `-fast` for each rung, through
`~/src/ds41urun.sh`, under `/tmp/ds2run.lock`, the front door in maintenance).
