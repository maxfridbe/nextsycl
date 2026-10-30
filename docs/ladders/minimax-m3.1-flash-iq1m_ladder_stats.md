# minimax-m3.1-flash-iq1m: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for MiniMax-M3.1-Flash UD-IQ1_M (129 GB on disk,
111.3 GiB of routed experts), engine `llm/minimaxm3`, measured 2026-10-28 and 2026-10-29 on hardac1. Port notes, profile and what
bounds it: [docs/ports/minimax-m31.md](../ports/minimax-m31.md).

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (layers 0-29) + Arc Pro B65 (layers 30-59), 31.9 GiB each; the single-card rows use the B70 alone |
| host | 61 GiB RAM, Fedora, in the `h3-build` image; the front door idle (`system.busy` false) |
| model file | Crucial P310 4 TB NVMe (~5.2 GB/s at 8 parallel direct reads), the 4 shards of `UD-IQ1_M/` |
| second drive | the system NVMe (`NS_MM31_STREAM_DIR`): a copy of half of the off-card experts, written once (5 s, 17-19 GiB) |
| expert places, 2K and 8K | B70 23.5 GiB in VRAM + 13.0 GiB pinned, B65 23.2 GiB in VRAM + 18.0 GiB pinned (77.7 of 111.3 GiB); during decode the places are a recency cache (see the port notes) |
| expert places, 32K | 21.7 + 13.0 and 21.3 + 18.0 GiB (74.0 GiB; the 32K KV reserve takes the difference) |
| expert places, B70 alone | 18.9 GiB in VRAM + 32.0 GiB pinned (50.9 GiB) |
| settings | `llm generate --gpu 0,1` (or `--gpu 0`) `--temp 0 --ignore-eos`, prompt chunk 4096, read-ahead 3, no draft |

Each rung is a fresh process (the load, 22-33 s on two cards, not counted). Prompts: the repository text repeated
to length (`p2k` 7,600 characters, `p8k` 30,000, `p32k` 120,000), chat template applied.

## Results (2026-10-29)

| prompt tokens | cards | pp | tg | tokens generated |
|---:|---|---:|---:|---:|
| 2,115 | B70 + B65 | 112.7 tok/s (18.8 s) | 7.38 tok/s | 200 |
| 7,925 | B70 + B65 | 120.9 tok/s (65.5 s) | 7.51 tok/s | 200 |
| 31,211 | B70 + B65 | 88.7 tok/s (352.0 s) | 5.99 tok/s | 100 |
| 2,115 | B70 alone | 123.1 tok/s (17.2 s) | 5.18 tok/s | 200 |
| 7,925 | B70 alone | 145.5 tok/s (54.5 s) | 5.06 tok/s | 100 |

One run of each rung (the stream file already written). tg is the average over the generated tokens and **rises as the
decode cache warms up** (see below): over the first 120 tokens of the 2K prompt it was 6.4 tok/s, over 200 7.4, and the
32K rung's 100 tokens (6.0) are still warming. The context itself costs little: the sparse attention reads 16 blocks of
128 keys however long the context is.

### The same rungs before the 2026-10-29 work (2026-10-28 state, main 2306522)

| prompt tokens | cards | pp | tg | tokens generated |
|---:|---|---:|---:|---:|
| 2,115 | B70 + B65 | 105.5 tok/s | 3.26 tok/s | 64 |
| 7,925 | B70 + B65 | 115.5 tok/s | 3.21 tok/s | 64 |
| 31,211 | B70 + B65 | 85.8 tok/s | 2.82 tok/s | 32 |
| 2,115 | B70 alone | 112.8 tok/s | 2.60 tok/s | 64 |
| 7,925 | B70 alone | 137.0 tok/s | 2.52 tok/s | 32 |

And before any of the work (the first engine, B70 alone, the 2,115-token prompt): pp 8.1 tok/s (259.8 s), tg 1.13 tok/s.

## Reading it

- **tg** went from 3.3 to 7.4 tok/s on two cards (2.6 to 5.2 on the B70 alone) because the experts are no longer placed
  once at load: each layer's most recently used experts are moved into VRAM as the tokens go by (hit rate of the cards 70% -> 93%),
  the copies run on the copy lanes beside the compute queue, and the experts of a token are one fused kernel pair.
- **pp** is bound by streaming the experts no card holds, once per prompt pass: at chunk 4096 a 2K prompt is one pass, and
  the figure grows with the prompt to 8K and falls at 32K, where the sparse attention and a smaller VRAM share take over. The B70 alone is
  not slower in prompt passes (123 against 113 tok/s at 2K); the second card's layer hop and reads were not separated.
- Greedy text is deterministic for a given build (the expert sums add in rank order) but, since the decode products read
  float32 activations (the shared kernels' Q8_1 before), it differs from the 2026-10-28 text after a few tokens at near ties.
