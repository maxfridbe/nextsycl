# DeepSeek-V4.1-Flash-1017 (Unsloth GGUFs): port notes

The same engine as the [VQ file's](deepseek41.md) (`llm/deepseek41`, silo `kernels/llm/deepseek41/silo`), reading the
llama.cpp form of the model ([llama.cpp PR 30817](https://github.com/ggml-org/llama.cpp/pull/30817), architecture
`deepseek41`) as Unsloth publishes it. Cards: `deepseek-v4.1-flash-iq2xxs`, `deepseek-v4.1-flash-iq2xxs-fast`,
`deepseek-v4.1-flash-q3kxl`.

## Why

The VQ file (experts at ~1.6 bits) degraded on long answers: a story's unique-word ratio fell from 0.6 to 0.2 over the
first ~2,000 tokens and a chat looped at ~7,300 tokens, at temperature 0.6 as at 1.0. The sampler (it draws only inside the
nucleus), the decode path against the prompt path (cosine 0.9965 at 7.3K tokens, the same top token) and the old V4's
clean 4,100-word story on the same prompt pointed at the quantization. Unsloth's UD-Q3_K_XL (DeepSeek's own MXFP4
experts in most layers) wrote a clean 2,903-word story, UD-IQ2_XXS a clean 12,000-token novel (5-gram repeats <= 0.02,
unique-word ratio 0.41-0.59 throughout). The VQ file was removed from the box; its card stays.

## The form

- Metadata: `deepseek41.*` (`ple.*` for the Engram's constants, `dspark.*`, `indexer.*`), a token-type table (the VQ file
  had none).
- Tensors: merged `ffn_{gate,up,down}_exps` [384, rows, cols] in a ggml row type that varies by layer (UD-IQ2_XXS: IQ1_S,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, Q2_K; UD-Q3_K_XL: mostly MXFP4, Q3_K, IQ3_S, IQ3_XXS, IQ2_S), attention and the shared
  expert in Q8_0 / Q6_K / Q5_K / BF16, `engram_gate` (the product of the VQ file's q and k) and `engram_kv`.
- The Engram's table is a tensor of the last shard, `per_layer_token_embd.weight` [768,022,850 rows, 256] in MXFP8 /
  Q5_K / Q6_K / IQ4_NL (126-189 GB), read a row at a time like the safetensors shards.
- The DSpark drafter is a file of its own (`dspark-*.gguf`, architecture `dflash`), loaded with its tensors renamed to the
  towers' (`Gguf::open_with_map`, `model::dflash_name`); `NS_DS41_MTP=1`.

## Kernels (`kernels/llm/deepseek41/silo/std.cpp`)

One scheme for every ggml type: a 256-value block is 32 groups of 8 values, a function per type makes group g (scales,
signs and grid read for that group alone); the decode product is a sub-group a row, lane g the block's group g, float32
activations; the expansion (for prompts) a work-item a group. Types: MXFP4, Q2_K, Q3_K, Q5_K, Q6_K, Q8_0, IQ1_S,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S. The grids are staged in local memory by each work-group. Checked **bit-exact
against ggml's own `dequantize_row_*`** (a C harness built from `ggml-quants.c`; `nextsycl llm kernels <gguf>` with
`NS_DS41_STD_DIR`), expansions and decode products, on real experts of every type in both files.

## Speed (B70 + B65, the files on the 4 TB NVMe, IQ2_XXS; experts 146 GiB against ~85 GiB resident)

| | exact | `-fast` |
|---|---|---|
| tg, sustained (prose, a warm expert cache) | ~14 tok/s | ~24 tok/s (23.1 over 12,000 tokens) |
| tg, the first 256-1,024 tokens after a 2.5K-140K technical prompt | 7-9 tok/s | 11-13 tok/s |
| pp, 580 tokens | 12.6 s (46 tok/s) | 6.3 s (92 tok/s) |
| pp, 2,561 tokens | 133 tok/s | 166 tok/s |
| pp, 9,901 tokens | 392 tok/s | 425 tok/s |
| pp, 25,142 tokens | 470 tok/s | 445 tok/s |
| pp, 139,561 tokens | 448-473 tok/s | 474 tok/s |
| quality: perplexity of two prose texts (exact 3.42 / 3.91) | - | 3.60 / 4.02 (+3-5%) |

UD-Q3_K_XL (253 GiB of experts, two thirds streamed from the drive): tg ~4.4 tok/s.

By prompt length, with the machine and silo: [the ladder](../ladders/deepseek-v4.1-flash-iq2xxs_ladder_stats.md).

**Where the time goes** (`NS_PROFILE=1`, a device-time trace): per decode token ~41 ms of GPU kernels (dense Q8_0 / Q6_K
~16, expert products ~13, attention 2, the rest small ones) and, without the cold-expert skip, ~22 ms waiting on ~6 cold
experts read from the file (3.7 ms each, three slabs in parallel). A prompt touches nearly every expert of every layer
(even 100 tokens reach ~80%), so its time is the non-resident experts' bytes over the drive's ~5 GB/s (2,561 tokens:
72 GB in 14.4 s).

## What helped

- Read-ahead on 16 reader threads, a job a slab (Q3_K_XL tg 2.7 -> 4.4 tok/s); speculating only on guesses likely
  enough (`NS_DS41_FILE_AHEAD_P` 0.2 against the guess's *measured* hit rates, which were half the table inherited from
  other models': tg 8.7 -> 14 tok/s).
- Prompt chunks of 16384 (the card's setting; `NS_PREFILL_CHUNK`, to 32768): each chunk re-reads every non-resident
  expert, so pp at 10K tokens went 258 -> 391 tok/s and at 24K 311 -> 470; short prompts lose ~8% (fewer resident experts).
  Per-expert asynchronous file reads and, for passes of 4096+ tokens, the next layer's cold experts read while this one
  computes (below that the early reads only contend).
- The expert profile (`NS_EXPERT_PROFILE`; `generate` loads and saves it too): VRAM filled with the experts used most.

## The fast setting (approximations, off by default, on in the `-fast` card)

- `NS_DS41_COLD_SKIP=0.2`: decode leaves out a routed expert held by neither VRAM nor pinned memory whose share of the
  layer's routing weight is under 0.2 (75% of cold experts are under 0.15). tg 12.5 -> 18 tok/s, perplexity +1.8%.
  Rescaling the kept weights (`NS_DS41_COLD_RENORM`) made it worse; 0.4 is a cliff (+11%).
- `NS_DS41_DENSE_Q4K=1`: attention's and the shared expert's Q8_0 / Q6_K weights requantized to q4_K at load (ggml's
  `quantize_row_q4_K_ref` algorithm, ~20 s more at load): half the bytes a token reads and the fused q4_K kernels
  apply. +1.7% perplexity, tg ~+2 tok/s.
- `NS_DS41_COLD_SKIP_PROMPT=0.3` (<= 8 tokens an expert): the same for prompt passes. A secret code buried in a 10K
  prompt is still found with 0.5 / 16 and the fast decode settings together.

## Tried, no gain

The DSpark drafter (55% of drafts accepted, but a verify pass touches twice the experts and the towers take 6.7 GB of
VRAM: tg 5.5 against 10 tok/s, then 15.5-16.3 against 18.3 even with cheap cold reads); O_DIRECT reads (one cold expert
3.7 -> 2.5 ms, but decode loses the page cache's hits on repeating cold experts: tg 13.6 -> 12.7; `NS_DS41_ODIRECT`);
more reader threads, more pinned memory, a smaller VRAM reserve, prefetch thresholds (all within the run-to-run noise
of +-5%); an LRU pinned host tier (`NS_DS41_HOST_LRU`: tg 12.2 against 12.9 tok/s over 1,024 tokens); the grouped `o_a` in one launch (98 against 138 us alone, no gain end to end); unrolling the expert kernel
over blocks (register spills).

## Open

One MoE pass over a group of chunks per layer (each expert read once a prompt, not a chunk; the group's streams over
8K tokens need pinned spill); fusing the small dense launches; re-quantizing the cold tail of the experts. See `TODO.md`.
