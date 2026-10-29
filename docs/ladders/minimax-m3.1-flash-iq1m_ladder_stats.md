# minimax-m3.1-flash-iq1m: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for MiniMax-M3.1-Flash UD-IQ1_M (129 GB on disk,
111.3 GiB of routed experts), engine `llm/minimaxm3`, measured 2026-10-28 on hardac1. Port notes and profile:
[docs/ports/minimax-m31.md](../ports/minimax-m31.md).

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (layers 0-29) + Arc Pro B65 (layers 30-59), 31.9 GiB each; the single-card rows use the B70 alone |
| host | 61 GiB RAM, Fedora, in the `h3-build` image; the front door idle (`system.busy` false) |
| model file | Crucial P310 4 TB NVMe (~5.2 GB/s at 8 parallel direct reads), the 4 shards of `UD-IQ1_M/` |
| second drive | the system NVMe (`NS_MM31_STREAM_DIR`): a copy of half of the off-card experts, written once (5 s, 17-19 GiB) |
| expert tiers, 2K and 8K | B70 23.5 GiB in VRAM + 13.0 GiB pinned, B65 23.2 GiB in VRAM + 18.0 GiB pinned (77.7 of 111.3 GiB; 69% of the expert uses hit the cards) |
| expert tiers, 32K | 21.7 + 13.0 and 21.3 + 18.0 GiB (74.0 GiB; the 32K KV reserve takes the difference) |
| expert tiers, B70 alone | 18.9 GiB in VRAM + 32.0 GiB pinned (50.9 GiB) |
| settings | `llm generate --gpu 0,1` (or `--gpu 0`) `--temp 0 --ignore-eos`, prompt chunk 4096, read-ahead 3, no draft |

Each rung is a fresh process (the load, 22-26 s on two cards, not counted). Prompts: the repository text repeated
to length (`p2k` 7,600 characters, `p8k` 30,000, `p32k` 120,000), chat template applied.

## Results

| prompt tokens | cards | pp | tg | tokens generated |
|---:|---|---:|---:|---:|
| 2,115 | B70 + B65 | 105.5 tok/s (20.0 s) | 3.26 tok/s | 64 |
| 7,925 | B70 + B65 | 115.5 tok/s (68.6 s) | 3.21 tok/s | 64 |
| 31,211 | B70 + B65 | 85.8 tok/s (363.7 s) | 2.82 tok/s | 32 |
| 2,115 | B70 alone | 112.8 tok/s (18.8 s) | 2.60 tok/s | 64 |
| 7,925 | B70 alone | 137.0 tok/s (57.8 s) | 2.52 tok/s | 32 |

One run of each rung (the stream file already written). The best decode seen in the work, with the 172-token prompt, was
tg 3.98 tok/s (3.85 profiled) on two cards; the rungs above run lower (3.26 at 2K) and the cause was not isolated
(longer context; the 64-token runs include the first tokens' cold expert reads).

Before the work (the first engine, B70 alone, the same machine): pp 2.9 tok/s (179 tokens) and 15.9 tok/s (4,044
tokens), tg 0.89-0.96 tok/s; a re-run on the 2,115-token prompt: pp 8.1 tok/s (259.8 s), tg 1.13 tok/s.

## Reading it

- **pp** is bound by streaming the experts that no card holds, once per prompt pass: at chunk 4096 a 2K prompt is one
  pass and each of the 57 MoE layers reads its off-card experts a single time, so the figure grows with the prompt up to 8K
  and falls at 32K, where the sparse attention (a 31K context in 16-block selections) and a smaller VRAM share take over.
  The B70 alone is not slower in prompt passes (113 against 106 tok/s at 2K, 137 against 116 at 8K; measured, not
  explained: the second card's layer hop and its own reads were not separated in the profile).
- **tg** is one token at a time: 4 of 128 experts per layer, 30% of them off the cards, each read from the SSD
  or copied from pinned memory. A second card lifts it from 2.6 to 3.3 tok/s at 2K by holding 19 GiB more experts in VRAM.
- Greedy text at 2K matches the first engine's token for token over 24 generated tokens; longer runs differ only
  where two logits are within fp32 summation order of each other.
