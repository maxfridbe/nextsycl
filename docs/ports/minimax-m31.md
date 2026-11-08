# MiniMax-M3.1-Flash (Unsloth GGUFs): port notes and assignment

Engine written and exact (2026-10-28): `llm/minimaxm3` has the model description (geometry from the metadata, every
tensor by role and shape) and the engine (`engine.rs`: MiniMax Sparse Attention, routed experts kept in
VRAM / pinned host memory or read from the file per pass); its kernels are `kernels/llm/minimaxm3/mm3.cpp` (rope,
SwiGLU-OAI, indexer scores, top-K, sparse attend) on the glm5next engine's generic dequant / GEMM / mmvq kernels. The
speed work (two cards, read-ahead, second drive, the silo's prompt attention; draft not done) is under "Speed work" at the end.
Card: `model-cards/llm/minimax-m3.1-flash-iq1m.yml`. Files: `unsloth/MiniMax-M3.1-Flash-GGUF`, `UD-IQ1_M/` (4 shards, 129 GB) at
`/var/mnt/4TBSSD/nextsycl-models/minimax-m3.1-flash-iq1m` on hardac1 (license `minimax-community`; read it before
serving).

## The model (read from the file's metadata and tensor table)

- Architecture `minimax-m3`, ~426B parameters, 128 experts of 16B-class size ("128x16B"), 23B active; native
  multimodal (text, image, video in; text out), context 1,048,576. Tokenizer gpt2/BPE, pre `minimax-m2`, 200,064 tokens,
  merges 199,744, bos 200019, eos 200020.
- 60 layers, hidden 6144, 64 query heads, 4 KV heads (GQA 16), head 128; q and k each RMS-normed per head
  (`attn_q_norm`, `attn_k_norm`, [128]); RoPE base 1e7, 64 of the 128 dimensions rotated. No biases.
- Layers 0-2: dense SwiGLU, 12288. Layers 3-59: MoE, 128 experts of 3072, top 4, sigmoid gating (`expert_gating_func` 2)
  with `exp_probs_b` added to choose (not to weigh), weights normalized then x2.0, plus 1 shared expert of 3072.
- MiniMax Sparse Attention on every layer: an indexer (`indexer.q_proj` [6144, 4x128], `k_proj` [6144, 128], both
  normed, F32 in the file) scores key blocks of 128; a query attends the best `top_k` 16 blocks plus `local_blocks` 1.
  The metadata gives the sizes only; **the scoring (how heads combine, whether RoPE applies, how the causal edge block
  is handled) is not in the file** - take it from the reference implementation (the model repo's modeling code and
  llama.cpp's `minimax-m3` graph) and check against it before writing a kernel.
- Types in UD-IQ1_M (the file, by role: `NS_MM31_FILE=... cargo test -p nextsycl-llm-minimaxm3 -- --nocapture` prints them): experts gate/up IQ1_M (28 layers) or IQ2_XXS (29), down IQ3_XXS (54) or IQ4_XS (3); attention Q5_K (one layer Q6_K); dense and shared FFN
  Q5_K/Q6_K; embeddings and head Q4_K; norms and the indexer F32. The other quants of the repo (IQ2_XXS, Q2_K_XL, IQ3_*,
  Q3_K_XL, Q4_K_XL, Q8_K_XL) change the expert/attention types per layer, as DeepSeek V4.1's did.
- A DSpark draft exists (`dspark-MiniMax-M3.1-Flash-Q8_0.gguf`, 2 GB; llama.cpp `draft-dspark`, one draft token: ~1.5x
  on code in Unsloth's tests). deepseek41 already has DSpark towers (`llm/deepseek41`); not part of the first pass.

## Size on the box

IQ1_M is 129 GB: B70 32 GB + B65 32 GB leave ~65 GB for experts of ~110 GB (the experts are ~100 GB at these types), so
the rest is host-pinned/file-streamed like deepseek4/deepseek41 (`spare` chain, pin-gib, NVMe). Expect it to load like
DeepSeek V4.1 IQ2_XXS; KV is small (4 KV heads x 128 x 60 layers).

## Reuse

- Dequant/matvec for IQ1_M and IQ3_XXS: `kernels/llm/deepseek41/silo/std.cpp` (bit-exact against ggml for V4.1's
  quants), Q4_K/Q5_K/Q6_K from `kernels/llm/qwen35/silo/mmvq.cpp`. Expert store, pinning, prefetch, two-card split:
  `llm/deepseek4` / `llm/deepseek41`. Split-file opening: `Gguf::open_with`.
- Flash-style decode attention with GQA: qwen35's. New: the block indexer and the block-sparse gather.

## Order of work

1. Reference: llama.cpp (a build with `minimax-m3`) or the HF modeling code on a small prompt, dumping per-layer outputs
   (read-only; a ref.py of our own, as the other ports).
2. Engine on one prompt token to token: embeddings, a dense layer, an MoE layer, the indexer, with each checked against
   the dump (cosine/abs per layer), then 60 layers and the head; greedy tokens identical.
3. Silo: the indexer scoring and block gather first (the new kernel), then expert matvecs per type.
4. Card, registry entry (`minimax-m3.1-flash-iq1m`), chat template (in the GGUF), tool format.
5. Speed: pp/tg per context length in a ladder doc, like GLM's.

`NS_MM31_FILE=<shard 1>` makes `cargo test -p nextsycl-llm-minimaxm3 (the crate sits in crates/ until the engine exists; then it moves to llm/ with its checkpoints)` check the real file against the description.

## Spec points verified from sources (2026-10-28)

Sources read (data only, nothing executed): llama.cpp's `src/models/minimax-m3.cpp` (the checkout on hardac1,
`~/src/llama.cpp-glm5`, and upstream master, which agree), `conversion/minimax.py`, `src/llama-graph.cpp`
(`build_moe_ffn`, `build_ffn`), `ggml/src/ggml-cpu/ops.cpp` (`swiglu_oai`), `src/llama-model.cpp` (rope type), and the
model's `config.json` on Hugging Face.

- **SwiGLU-OAI**: `alpha` 1.702, `limit` 7.0 (llama.cpp's constants, "the same as gpt-oss"; config `swiglu_alpha` / `swiglu_limit`
  agree). `x = min(gate, limit)`, `y = clamp(up, -limit, limit)`, `out = x * sigmoid(alpha x) * (y + 1)`. Dense FFN, routed
  experts and the shared expert all use it; gate and up are separate tensors (no interleaving).
- **RoPE**: NEOX (`LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_MINIMAX_M3`): value i pairs with i + 32 in the first 64 of 128
  values; base 1e7; no scaling. The indexer's q and k use the same RoPE (same `n_rot`, base) after their norm.
- **Norms**: plain RMS with the weights as stored (the converter adds the Gemma-style +1 into every `*norm.weight`);
  q and k are normed per head (one [128] weight for all heads), eps 1e-6.
- **Indexer scoring** (graph, both the decode and the batch path): `score[j] = iq_h . ik_j` per indexer head h (4 heads, one per
  key/value group), **unscaled**, no softmax, no relu; keys not visible to the query (causal, holes) are -inf; the block
  score is the **max** over the block's 128 positions (`sparse_score_type: "max"` in the config); the query's own block (and
  the `local_blocks` - 1 before it) get +1e30 so they always rank in; top-k (K = min(16, blocks)) per (token, group); the
  attention then reads the positions of those blocks that the ordinary causal mask lets through. The scale is irrelevant to
  the order. Indexer keys are cached as fp16 like K and V.
- **Causal edge block**: the current block is always selected; its positions after the query are masked (causal), so a
  block is attended "partially" only at the edge. Blocks with no visible key score -inf and are never picked.
- **Router**: sigmoid of the logits; the choice is top-4 of sigmoid + `exp_probs_b`; the weights are the sigmoids (without the
  bias) of the chosen, divided by their sum (clamped at 6.1035e-5) and multiplied by 2.0.
- **Differences between the model and the reference, left as found**: the config says `sparse_attention_freq` is 1 on all 60
  layers, `sparse_init_block` 0 (a block that is always kept?) and the file carries indexer tensors for layers 0-2, but
  llama.cpp's graph (the reference every number below is checked against) runs layers 0-2 with plain dense attention, loads no
  indexer for them and knows no init block. They differ only past 2,176 tokens of context (up to 16 blocks + the local
  one attend everything). `NS_MM31_SPARSE_ALL=1` makes the first three layers sparse too; the default is llama.cpp's.
- **The reference build**: the llama.cpp checkout on hardac1 stops at a GGUF with 12 tensors more than its graph loads (the
  dense layers' indexer, which `llama_model_loader` counts and refuses). The reference dump uses a copy of shard 2 without
  those 12 tensors and a shard 1 with `split.tensors.count` 1176 (`reference/minimaxm3/` scripts), the CPU only
  (`llama-ns-dump --device none -ngl 0`), the prebuilt binary: nothing was built or served.

## Results (hardac1, B70, one card, 2026-10-28)

- **Exactness** (`nextsycl llm check`, against llama.cpp CPU dumps; the activations into each layer taken from the reference with
  `NS_MM31_FORCE=<dump>` so the quantization noise of IQ1_M does not accumulate; cosine against the reference):
  - 19-token prompt (`/var/mnt/2TBSSD/m3ref/short`): every l_out >= 0.99985, kqv_out >= 0.99997, result_output 0.99996, the next token the
    same; the last token fed alone after the rest: logits cosine 0.99998 (against the whole prompt at once).
  - 3,881-token prompt (`long`, 31 blocks, so the top-16 selection really chooses): every kqv_out >= 0.99998, every l_out >= 0.99985,
    result_output 0.99976, the next token the same (" +"), decode-after-prefill logits cosine 0.99997.
  - Free-running (each layer on its own earlier output) the same checks drift to cosine 0.97-0.98 by layer 30 to 58 (the IQ1_M weights against
    the CPU reference's Q8_K activations, and a near-tie in the indexer's or the router's choice flips a block or an expert) and the next token can differ
    where the top two are 0.04 apart (long: "-" 13.21 against " +" 13.17); the teacher-forced runs show it is noise, not a layer that is wrong.
  - Layer 59 of the dump has one row only (llama.cpp prunes to the output rows there), so its taps compare a shape artifact (l_out-59 0.73-0.79).
  - `msa_bs` sizes differ (the reference pads the block grid); the block selection is verified through kqv_out, not the raw scores.
- **Speed**: see the ladder below; the engine today is input/output bound (the experts do not fit one card), the kernels untuned.

### Speed, first measurement (untuned, B70 alone; pp = prompt, tg = generation)

| context | pp | tg | notes |
| --- | --- | --- | --- |
| 179 tokens, 160 generated | 2.9 tok/s | 0.89 tok/s | 19 GiB experts in VRAM + 40 GiB pinned, the rest read from the file |
| 4,044 tokens, 48 generated | 15.9 tok/s | 0.96 tok/s | 19 GiB VRAM + 19 GiB pinned (the default at the time), chunk 1024 |
| 3,881 tokens (check run) | 24.6 tok/s | - | 36 GiB pinned, chunk 4096 |
| 2,115 tokens, 24 generated (re-run) | 8.1 tok/s (259.8 s) | 1.13 tok/s | the baseline of the table below |

## Speed work (2026-10-28): profile, second card, read-ahead, silo

### Profile of the first engine (2,115-token prompt + 24 tokens, B70 alone)

`NS_PROFILE=1` times each layer's sections with a GPU wait at each boundary (`--opt-profile`; the passes are slower than
unprofiled by that much). Where the time went: expert file reads 68% (201.8 s), expert products 19% (57.6 s), attention
4%, the rest 5%. **The bound was the SSD**: each pass read its experts one at a time, one request in flight (~3 GB/s) with
a GPU sync per expert.

### Profile after the work (B70 + B65, stream drive, read-ahead 3; the engine prints these tables with `NS_PROFILE=1`)

Decode, 172-token prompt, 63 one-token passes (16.6 s, tg 3.85 profiled / 3.98 unprofiled): expert products 51%
(8.35 s), expert read wait 33% (5.36 s), attention projections 8% (1.28 s), indexer 2%, attend 2%, router 2%, shared/dense
1%, other 1%. Per expert, timed alone on the GPU: VRAM 0.30 ms (x 5,772), pinned (copied to VRAM first) 0.84 ms (x 4,274),
staged from the file 0.87 ms (x 1,566). One-token stages (summed over 64 tokens, earlier run): gate 6.54 s (it also
holds the preceding host-to-VRAM copy), up 0.86 s, swiglu + quantize 0.14 s, down 1.31 s, scatter 0.10 s.
Per token (~250 ms): read wait ~85 ms, expert products ~130 ms (of which the PCIe copy of the pinned and staged experts is
the largest part: ~0.55 ms each, serialized with the compute in one in-order queue), attention ~25 ms. **What bounds
decode now**: the SSD bytes of the ~30% of expert uses that no tier holds (~0.9 GB a token; read-ahead hides about
half of them: 2,752 of the 5,663 needed) plus the PCIe copy of pinned / staged experts. The kernels (up / down ~0.15-0.2 ms)
are not the limit.

Prompt pass, 2,115 tokens, 4 wide passes of 512 (earlier chunk; 25.1 s at the time): expert read wait 27%
(6.7 s), expert products 35% (8.8 s), attend 20% (5.1 s with the silo; 16.1 s = 44% before it), attention projections 12%
(3.1 s), router / indexer / other 3%.

### What was done, each measured (pp = prompt, tg = generation)

| step | measure | before | after |
| --- | --- | --- | --- |
| second card (layers split), experts in both cards' VRAM and pinned tiers, O_DIRECT parallel placement, waves overlapping reads with products | 172-token prompt, decode | tg 1.08 tok/s, pp 8.7 | (first version, pinned read in place: 5.8 ms an expert) |
| pinned and file-read experts copied to VRAM before the products | the same | tg 1.08, pp 8.7 | tg 2.97, pp 25.4 |
| prompt chunk 512 -> 2048, second drive for half of the off-card experts | 2,115 tokens | pp 12.8 (165.0 s) | pp 57.7 (36.6 s); tg 1.04 -> 3.4-3.5 |
| silo prompt-pass attention | 2,115 tokens | pp 57.7 | pp 84.4 (25.1 s); chunk 4096 later: 105.5 |
| read-ahead of the next layer (top 3 of its guessed routing; 96% right at top 2, 82% at top 4) | 172 tokens decode | tg 2.98 (0) | 3.28 (2), 3.38 (3), 3.31 (4), 3.09 (5), 2.68 (6, profiled): wasted reads take SSD bandwidth |
| second drive (50% of the off-card experts) with read-ahead 3 | decode | tg 3.33 | tg 3.9-4.05 |
| expert-profile placement (`NS_EXPERT_PROFILE`, a profile from three unrelated prompts) | decode | 2.95 | 2.79-2.98: routing is flat, no gain |
| one-token fast path (activation quantized once a layer) | decode | tg 3.98 | tg 3.97-3.99: launches are not the limit |

Totals against the first engine on the same 2,115-token prompt: pp 8.1 -> 105.5 tok/s (13x), tg 1.13 -> 3.26 tok/s (2.9x);
ladder: [docs/ladders/minimax-m3.1-flash-iq1m_ladder_stats.md](../ladders/minimax-m3.1-flash-iq1m_ladder_stats.md).

### Exactness after the work

Teacher-forced `check` against the llama.cpp CPU dumps (two cards, silo attention, chunk 4096 for `long`):

| prompt | min l_out | min kqv_out | result_output | next token | decode-after-prefill logits cosine |
| --- | --- | --- | --- | --- | --- |
| `short` (19 tokens) | 0.999849 | 0.999966 | 0.999957 | same as the reference | 0.9999773 |
| `long` (3,881 tokens) | 0.999847 | 0.999988 | 0.999756 | same (" +") | 0.9999729 |

No drop against the first engine (0.99985 / 0.99997 / 0.99996 and 0.99985 / 0.99998 / 0.99976). Greedy: the 24-token
output on the 2,115-token prompt is byte-identical to the first engine's; longer runs differ only through fp32
summation order at near ties, and are the same across read-ahead, stream-drive and residency settings (the expert sums
add in rank order). `NS_SILO=0` (shared attention) gives the same 24 tokens.

### Counters

The expert-use line used to print zeros; it now counts VRAM hits, pinned (host memory) uses, file reads ("swapped in"),
prefetched and prefetch-used experts (`NS_PROFILE` is not needed), and `NS_EXPERT_PROFILE` / `save_profile` writes the
same profile file the other expert engines use.

### Left (as of 2026-10-28; see "Second round" below)

- Overlap the pinned-to-VRAM copies with compute (a second queue / the copy engine): the copy is the largest part of the
  products' time.
- Silo mmvq for IQ1_M / IQ2_XXS / IQ3_XXS / IQ4_XS (not done; the shared kernels run the expert matvecs, ~0.15-0.2 ms each)
  and a batched expert GEMM for prompt passes (today each expert dequantizes to half and runs the generic GEMM: 8.8 s of 25 s
  at 2K); an indexer-score kernel.
- The DSpark draft (`dspark-MiniMax-M3.1-Flash-Q8_0.gguf`, 2 GB) was not started.
- A larger pinned tier (capped at 32 GiB of 61 GiB RAM) would move the miss rate, at the box's hang risk.
- Not in the front door's registry yet (the card exists; no `registry` entry or chat template wiring was changed).

## Second round (2026-10-29): toward 20 tok/s

Target from the user: tg 20 tok/s. Reached: tg 7.4 tok/s on two cards (from 3.3), pp 113-121 tok/s at 2K-8K. What each step bought
(B70 + B65, the 2,115-token prompt, greedy; tg averaged over the generated tokens, so the longer runs include the cache's
warm-up; ladder: [docs/ladders/minimax-m3.1-flash-iq1m_ladder_stats.md](../ladders/minimax-m3.1-flash-iq1m_ladder_stats.md)):

| step | before | after | what changed |
| --- | --- | --- | --- |
| start of the round (main 2306522) | pp 105.5, tg 3.26 (64 tokens) | | |
| pinned / file-read experts copied into VRAM on the copy lane beside the compute queue (tickets order the two on the device; a ring of 4 VRAM slots, copies 2 experts ahead) | tg 3.26 | tg 3.54 (64 tokens); pp 112.6 | the copy no longer blocks the products; PCIe is the limit (lane 0, 1 or alternating: 4.04 / 4.02 / 4.03) |
| silo decode attention (the keys split in chunks of 128 over work-groups, merged by a second kernel) | tg 3.54 | tg 4.02 (64 tokens) | the shared kernel took 0.9 ms a layer at a 2K context (53 ms a token); now 0.58 s of 63 passes |
| decode places as a recency cache (each layer: the most recently used experts in VRAM, the next in pinned memory; a used expert that was pinned or in the file swaps into the VRAM place of the least recently used, which goes to the pinned place it leaves) | tg 3.92 (300 tokens, same build without it) | tg 6.12 (300 tokens), text identical | hit rate of the cards 70% -> 92.9%; routing is flat across unrelated prompts but strongly local inside one generation (offline replay of a traced 300-token run: the recency stack gives 87.5% VRAM + 9.2% pinned + 3.3% miss) |
| silo fused expert products (gate+up+SwiGLU-OAI in one kernel, down in one, up to four experts a launch, float32 activations, ggml's row types IQ1_M / IQ2_XXS / IQ3_XXS / IQ4_XS) | tg 6.12 | tg 7.26 (300 tokens) | the shared kernels took three launches an expert on Q8_1 activations; text differs from the earlier engine after a few tokens (float activations) |
| no per-layer syncs in one-token passes (tickets for the moves), stale read-ahead not waited for, matrices read in 8 pieces at once | tg 7.26 | tg 7.38 (200 tokens) | within noise except the reads: the host waited 6.2 s -> 5.4 s for expert reads over 200 tokens |

Read-ahead of the next layer (`NS_MM31_AHEAD`) no longer matters much: 0 / 2 / 3 -> tg 6.95 / 7.18 / 7.22 (200 tokens).

### Where a token's time goes now (2K context, 200 tokens, 27 s; `NS_PROFILE=gpu` stamps, no syncs)

Device time by part, per token: attention (norm, q/k/v, norms, rope, KV store, indexer, split attention, o projection) **31 ms**,
router 4 ms, host turnaround from the router read-back to the shared expert queued 3 ms, **experts 90 ms** (queue time between the shared expert
and the last expert: the products themselves, the waits for pinned / file experts' copies and the host's waits for the reads),
adds and layer end 4 ms. The host side of the 138 ms a token: 32 ms waiting for the routers' read-backs (the GPU running attention),
31 ms waiting for expert reads (a miss costs ~4 ms: 16 MB from the SSD, 7.5 misses a token), the rest launching and waiting for
the GPU's expert work. The fused kernels alone are fast (a gate+up launch of one IQ1_M expert: 37 us, 224 GB/s, scaling linearly with
the experts in the launch; the profile's per-launch figures with a sync around each are 0.5 ms higher than that).

### Why 20 tok/s is not within reach of this engine (the second round's estimate; the third round's measured floor is below)

20 tok/s is 50 ms a token. What is left is not the experts' bytes any more but the number of dependent steps: 60 layers, each with
~30 small kernel launches for attention (31 ms in all, launch- and latency-bound; no kernel runs more than 0.1 ms), a router
read-back that makes the host decide per layer (3 syncs removed; the GPU still waits for the host's routing round trip), and
copies / reads for the 7% of expert uses that are not in VRAM (pinned 9%: 0.45 ms each on PCIe; file 3.3%: ~4 ms each). With every
expert hit in VRAM the floor is attention + experts' compute + the 60 router round trips, about 60-70 ms (15 tok/s); 20 would need
the attention and router steps fused into few launches (or recorded as a command graph) and a draft that lands two tokens a pass.

### Not done, and why

- **DSpark draft** (`dspark-MiniMax-M3.1-Flash-Q8_0.gguf`): not started. A verify of k tokens runs the generic multi-row path (the silo's fused pair
  is one-token) and reads the union of the k tokens' experts; with 7% of uses off the VRAM, the union of 2 tokens' 8 experts a layer
  costs about twice the misses of one. Estimate (not measured): a pass of 2 rows costs 1.4-1.7x a pass of 1, so one draft token needs more than 40% acceptance
  to break even and the gain at the quoted ~1.5x of Unsloth's code tests would be 10-30%, before the draft's own cost. Worth doing only
  after the one-token path is cheaper.
- **Batched expert GEMM for prompts, indexer-score kernel**: not done (pp is bound by the experts' reads and the dequant-to-half GEMMs; pp moved only through the read splitting and the copy lane, 105 -> 113 at 2K).
- **Larger pinned tier**: not changed (32 GiB of 61 GiB RAM, the box hangs when it spills). With the recency cache the pinned tier serves 9%
  of the uses, so more of it buys little; a cold generation (new topic) warms in ~100 tokens.
- **Pinned experts prefetched for the guessed next layer**: the largest thing left in the experts' time (~10 ms a token by the copy waits above); not done.

### Exactness after the second round

Teacher-forced `check` (two cards, silo attention, chunk 4096 for `long`), unchanged to the printed digits:

| prompt | min l_out | min kqv_out | result_output | next token | decode-after-prefill logits cosine |
| --- | --- | --- | --- | --- | --- |
| `short` (19 tokens) | 0.999849 | 0.999966 | 0.999957 | same as the reference | 0.9999773 |
| `long` (3,881 tokens) | 0.999847 | 0.999988 | 0.999756 | same (" +") | 0.9999729 |

The check's decode line is the same with `NS_MM31_FUSED=0` and `NS_MM31_DYNAMIC=0` (it compares one token against the whole prompt's wide pass, which
reads float activations too; the float products are the closer of the two to it). **Greedy text changed**: the fused products use float32 activations instead
of ggml's Q8_1, so a 300-token generation diverges from the 2026-10-28 engine's after 19 characters (near ties); the cache itself
changes no byte of the text (300 tokens, `NS_MM31_DYNAMIC=0` against 1, measured with the unfused products), and the fused text is the same
before and after the sync removal (300 tokens, byte-identical). `NS_MM31_FUSED=0` gives the earlier arithmetic.

## Third round (2026-10-30): both speeds, by what the profile showed

Reached on two cards: **pp 199 / 332 / 289** at 2K / 8K / 31K (from 113 / 121 / 89), **tg 9.5 / 8.5 / 7.1** (from 7.4 / 7.5 / 6.0); tg 20 was not reached: floor below. The
prompts of the earlier rounds were gone from the box; the new ones (`README.md + docs/architecture.md + CONTRIBUTING.md` repeated) are 2,137 / 7,990 / 31,085 tokens (was 2,115 / 7,925 / 31,211).
Main measured on them: pp 112.5, tg 7.06 (300 tokens).

### The order changed: what the profile said

The assignment's order was device-side routing, fused attention, silo mmvq, miss reduction. A profile by layer kind (`NS_PROFILE=gpu`, the `[card N: ...]` lines) said otherwise. On main, per token:

| | layers a token | experts' device time a layer |
| --- | --- | --- |
| all four in VRAM | 34 (B70 0.40 ms, B65 0.70 ms) | |
| some pinned, none read | 13 | 1.86 / 2.29 ms |
| some read from the file | 10 | 4.68 / 4.58 ms |

so misses cost ~65 of the ~140 ms, and a pinned miss cost three copies (1.5 ms) rather than one (0.45 ms). The host turnaround between router and shared expert was 3.7 ms a token.

**Step 1, device-side routing: skipped.** It removes at most that turnaround, and 22 of 57 layers then had a miss (21 now) and would have to stop the queued work: a stop every ~2.7 layers. It cannot pay until misses are rare.

### What each step bought (B70 + B65, 2,137 tokens unless said; tg over 300 tokens)

| step | pp before | pp after | tg before | tg after | text |
| --- | --- | --- | --- | --- | --- |
| a miss lands directly in an empty VRAM place of the layer, one copy (`NS_MM31_FREE` kept empty, the least recently used demoted on the other lane beside the compute); no end-of-layer sync | 112.5 | 112.7 | 7.06 | 7.91 | identical |
| reads from a pool of threads, not 24 spawned an expert; pieces go round the drives; second drive holds all the experts off the cards at load (share 100, 34 GiB) | 112.7 | 110.1 | 7.91 | 8.06 | identical |
| silo steps of a token: indexer projections in one launch, norms + RoPE + stores in one, add + norm in one, layer end in one, router on float; q, k, v, o and the shared gate/up on a copy of the qwen35 silo's Q5_K kernels (525 GB/s against ~214) | 110.1 | 115.6 | 8.06 | 8.59 | identical |
| prompt passes: q, k, v, o and shared expert on half (`NS_MM31_HALF_PROJ`; the float GEMM was 5x slower), silo dequantize-to-half; 60% of the layers on the B70 | 115.6 | 150.0 (8K 172) | 8.59 | 8.43-8.60 | below |
| long prompts: both cards at once (a chunk on the B70 while the B65 has the one before; `NS_MM31_PIPE`), logits bit-identical to chunk after chunk | - | 8K 121 -> 159 (with 60% split 172) | | | |
| attention on XMX, prompt passes (`attnx.cpp`: the qwen35 flash kernel shaped for 16 heads of 128 and selected blocks; a 2K pass 14.2 -> 10.8 s) | 150.0 | 197.7 (8K 332, 31K 288) | 8.43 | 8.44 | from byte 232, with the row above |
| attention on XMX, decode (a work-group a split of 128 keys) | 197.7 | 197.9 | 8.59 | 9.48 | changes at byte 49: half queries |
| no read-ahead by default (`NS_MM31_AHEAD` 0: guesses on the CPU in a thread, at 1, 2, 3 and 6 layers ahead, never gained; they cost a second read-back a layer) | 198.3 | 197.8 | 9.71 | 9.93 | |

### Tried and dropped, measured

- **Warming the cache from the prompt's last 64 rows**: pp 198 -> 160 (2.6 s more), tg 9.47 -> 9.50: the decode routes differently from the prompt's end. Not kept.
- **Chunks**: 4096 stays. 2K: 4096 198, 1024 91 (before XMX); 8K: 4096 333, 8192 235; 31K: 4096 288, 8192 303, at 2 GiB more scratch a card. Every pass moves all the off-card experts once, so small chunks lose.
- **Third drive** (the 2 TB, 1.8 GB/s): a stripe goes at its slowest drive, no gain.
- **Global LRU over a card's layers** (replay of a 300-token trace): 85.5 against 84.8% in VRAM, not worth it. **1.5x the cache** in the same replay moves pinned into VRAM but leaves file misses at 4% (first touches in the long tail: 100 of 128 experts a layer are touched in 300 tokens).
- **Fused multi-row expert kernels for prompts**: not written. 92% of the experts of a 2K pass take more than 8 rows and a direct product costs 37 us a row against ~0.9 ms for half + GEMM: break-even ~12-14 rows, 8% of the work at best.
- **Float Q5_K products at decode** (`mv3`): 214 GB/s, no gain over the shared kernel, and they moved the text; replaced by the tuned Q8_1 ones.

### Where a token's time goes now, and the floor

300 tokens, 2K: 100 ms average; 138 ms over the first 50 (cold: 25 file reads a token), 85-109 ms after (10.5 tg), ~7 file reads and ~20 pinned copies a token. Device time a token: attention 20 ms, router 0.7, host turnaround 7 (profiled), experts 67 (all-VRAM layers 0.38 / 0.67 ms; pinned 0.95 / 1.02; read 4.2 / 4.1). Host: 61 ms waiting for read-backs (the GPU's work up to the router), 29 ms for expert reads.

**The floor with no misses at all is ~52 ms, tg 19**: attention 20 + 33 layers x 0.38 + 24 x 0.67 + router + turnaround. Misses add ~35 ms in steady state (7 x ~3.7 ms read + 20 x 0.55 ms copied), and those are physical: 16 MB from a drive at 4.5-6 GB/s, 16 MB over PCIe; hit rates are bound by the long tail of routing, not the policy. So tg 20 needs either no misses (VRAM for all 111 GiB) or a draft.

- **Every expert on a second drive** (`NS_MM31_STREAM_ALL=1`, 111 GiB on the system NVMe, written once in 78 s): tg 9.93 -> 10.49 (300 tokens), pp unchanged. Available, not default.
- **DSpark draft: not done, on measurement.** A 2-row verify costs 2.07x a 1-row pass (255.5 against 123.1 ms, `NS_SCORE_ROWS`, the unfused multi-row path) and touches 2.4x the file reads, so one draft token gains nothing at any acceptance. A fused 2-row path would have to come first; even at 1.6x and Unsloth's ~1.5x on code that is <10%, before the draft's own passes and with a 2 GB model to serve.

### Exactness

`check` (two cards, chunk 4096), the same columns as before:

| prompt | min l_out | min kqv_out | result_output | next token | decode-after-prefill logits cosine |
| --- | --- | --- | --- | --- | --- |
| `short` (19 tokens) | 0.999849 | 0.999966 | 0.999957 | same as the reference | 0.9555839 |
| `long` (3,881 tokens) | 0.999848 | 0.999988 | 0.999756 | same (" +") | 0.9795699 |

Teacher-forced columns: unchanged or better in the sixth digit, and this is the prompt path (half projections, half expert route, XMX attention).

**The decode column of the earlier tables measured nothing.** `NS_MM31_FORCE` replaced a layer's output when the dump's size matched; the dump's layer 59 has one row, so at one token the last layer's output *was* the reference's, whatever the engine had done (a broken XMX decode kernel, cosine ~0.05 per layer and garbage text, still printed 0.9999729). The force now applies to wide passes only and the column is a real free-running decode against the forced whole prompt. On that measure, the earlier states: fused products and shared attention (the second round's engine, `NS_MM31_ATTX=0 NS_MM31_FUSE1=0 NS_MM31_FUSED=0`) 0.9628573 / 0.9800251; vector-unit attention with this round's other steps 0.9632365 / 0.9802953; XMX 0.9555839 / 0.9795699. So the XMX decode attention lowers this one-token column slightly (half queries, a free-running token). Over 400 teacher-forced decode rows of a text the model has not seen, its perplexity is no worse: **16.42 with XMX, 16.63 with the vector-unit attention**, the likeliest token the text's 43.25% against 43.00%.

**Greedy text changed.** The cache, reads and silo steps leave a 300-token text byte-identical to main's (`cmp`). The half route of prompt passes and XMX prompt attention move it from byte 232, XMX decode (half queries) from byte 49: near ties, both readable and on topic. `NS_MM31_ATTX=0 NS_MM31_HALF_PROJ=0` gives the earlier arithmetic, at pp 121, tg 8.6.

Also found: demotions could pick a pinned place a fallback copy was still reading (more experts in a layer than empty places), so a text could differ between runs; fixed, three runs byte-identical.

### Left

- Second drive for all experts (user's decision: 111 GiB). A multi-row fused path, then DSpark. Not in the front door's registry yet.
- A stream file is per placement: `NS_MM31_STREAM_DIR` now deletes another placement's file first (36 GiB each; they filled the system drive during this round's benches).

## Fourth round (2026-10-30): the user's list

Every number on the same prompts as the third round (2,137 / 7,990 / 31,085 tokens). "ms a token" is on one fixed 300-token text fed one token at a time (`NS_SCORE_ROWS=1 --score`), so numerics that change greedy text do not change the measure.

### What each step bought (ms a token over 300 tokens; first 50 tokens in brackets)

| step | pp 2K | pp 8K | ms a token | greedy tg 300 |
|---|---|---|---|---|
| end of round 3 | 198.8 | 331.9 | 99.5 (139.8), on the new kernels below | 9.93 |
| row-type kernels through word loads (byte loads were a message each), shared down back on Q8_1 (float Q6_K 76 GB/s against 444) | 198.8 -> 221 | 332 -> 347 | | 9.93 -> 9.87 |
| doorbell: router writes to pinned memory, rings, the shared expert runs behind it | | | 99.3 -> 95.1 | |
| B65 first, B70 the rest and the head | 216-218 | 337-357 | 95.1 -> 89.4 | 10.15 |
| prompt scratch freed in decode (1.3 GiB a card as expert places), spare 2.5 -> 1.9 GiB | | | 89.4 -> 83.8 | 11.22 |
| small tables by kernel from a pinned ring, not the runtime's memcpy | | | 84.0 -> 82.0 | |
| expert profile: seed placement, hottest counted most recent | | | 82.0 -> 74.9 (115 -> 85) | 12.26 |
| page cache at run time | | 373 | 74.9 -> 73.4 | |
| table work-groups 16 row pairs, not 4 | | | 73.4 -> 71.9 | 13.3 |

tg 20 was not reached: tg 11.4 at 2K over 200 tokens (12.5 with the expert profile); 72 ms a token on the fixed text with the profile. pp 230 / 373 / 305 (2K / 8K / 31K). Numerics change the text, so ms on the fixed text is given beside greedy tg. Repeats vary: with the page cache, three 100-token runs gave tg 10.1 / 13.0 / 10.9 (O_DIRECT: 10.2 / 10.2 / 10.2), and a 2K pp sometimes reads 147-160 where 213-230 is usual; the cause of that outlier was not found.

### 1. Layer order (B70+B65, default arena, before the later steps)

| order | first card's layers | pp 2K | pp 8K | ms a token (first 50) | greedy tg 300 | file reads a token |
|---|---|---|---:|---|---:|---:|
| B70 first | 36 | 221 | 347 | 99.5 (140) | 9.92 | 10.3 |
| B65 first | 3 | 179 | 264 | 155.9 (203) | 6.77 | 23.7 |
| B65 first | 8 | 189 | 285 | 132.1 (173) | 7.43 | 16.3 |
| B65 first | 12 | 200 | 309 | 114.5 (155) | 7.84 | 11.8 |
| B65 first | 18 | 216 | 341 | 94.3 (131) | 9.63 | 8.6 |
| B65 first | 24 | 218 | 357 | 95.3 (151) | 10.15 | 9.0 |
| B65 first | 30 | | 337 | 92.8 (134) | | 9.6 |

The B65 taking only the dense layers leaves its VRAM idle: experts cannot live on the other card, so hit rate is what decides. Per card at 18/42: B65 98.0 / 0.7 / 1.3%, B70 80.1 / 14.1 / 5.8%. Default now: B65 first, 24 layers, B70 the rest and the head.

### 2. Memory

During decode (`NS_MM31_SPARE_GIB`, gpustat sampled every second): each card 30.1 GiB used of 31.9, 1.6-1.9 free. In them: weights 3.3 / 4.5 GiB, experts 24.7 / 22.6 GiB. Idle before this round: the prompt arena 1.5 GiB and ~1 GiB of spare above the floor.

| VRAM for experts a card | 14 | 18 | 21 | 24 | spare 1.7 |
|---|---|---|---|---|---|
| ms a token | 124.1 | 105.8 | 95.9 | 89.4 | 87.6 |
| VRAM / pinned / file a token | 175 / 35 / 18 | 189 / 25 / 14 | 197 / 20 / 11 | 202 / 17 / 10 | 203 / 16 / 9 |

About 2.2 ms a token per GiB on both cards. Taken: prompt scratch freed in decode, 192 MiB kept (the slab comes back as empty places in every layer; dropped again before the next prompt pass), spare 1.9 GiB (peak gpustat free 1.45 GiB at 1.7, so 1.9 keeps the 1.5 GiB floor). Not worth it: indexer to half (0.25 GiB a card, ~0.5 ms); KV at 2K is 0.2 GiB.

| pinned GiB | 8 | 16 | 24 | 31 |
|---|---|---|---|---|
| ms a token | 118.2 | 105.7 | 96.4 | 89.4 |

About 1.3 ms per GiB. The 32 GiB cap holds. The page cache now does real work: run-time reads go through it (O_DIRECT only at load) and it is reclaimable, 74.9 -> 73.4 ms.

### 3. Placement

Six prompts of different kinds, 200 tokens each: a layer's top 54 experts carry 79% of the uses across the corpus, but only 45-62% of a held-out text's (static top 54 / 84 against recency's 88%), and experts touched a layer are 121-128 of 128. So a profile cannot replace recency; as a seed it removes most of the cold start: first 50 tokens 115 -> 85 ms, mean 82.0 -> 74.9 (`NS_EXPERT_PROFILE`, hottest counted most recent; the service sets it already). The same ranking across a card's layers is already the unequal budget (misses by layer 688-1468). Replay: LFU, recency + frequency and LRU-2 give 86.9-87.0% VRAM against LRU's 86.7, file reads the same: dropped. Which reads will happen: guesses on the CPU ahead 3 experts: 73.4 -> 78.9 ms, dropped.

### 4. Strata's tricks

**Loads** (`NS_MM31_KBENCH=1`, B70, GB/s of weights; the ceiling ~520-570 on these shapes with dp4a): the row kernels read bytes one message each. Rewritten as word loads: gu IQ1_M 228 -> 268, IQ2_XXS 291; down IQ3_XXS 315-331; float Q6_K 76 -> 349; Q5_K float 382 -> 442; Q8_1 Q5_K 524, Q6_K 444; half GEMM 528; to_half 38 MB out in 76 us (a 500 GB/s store limit), router F32 300. B65 about half (140-170 on gu). Benched numbers vary with the idle clock (B70 123-252 us for a gu), so ms a token is the verdict: 16-row groups for table kernels 73.4 -> 71.9.

**Device plan**: skipped, measured. Host from the doorbell to the VRAM experts queued is 28 us a layer; the shared expert queued behind the router takes 100-200 us, so the GPU does not idle for the host: nothing is left to save. Launches: queuing one costs the host 1.3-7 us, 0.85-1.45 us on the device (stamps with profiling on read 30 us, misleadingly); ~24 launches a layer are not a limit. Doorbell itself: 99.3 -> 95.1 ms.

**Small tables by kernel**: 84.0 -> 82.0 ms.

**Draft**: DSpark ported (`NS_MM31_DRAFT`, `NS_MM31_DRAFTS`, `NS_MM31_DRAFT_PMIN`, 1.75 GiB on the last card), verify rows on the fused path, `spec-check` bit-exact against one-token decode (0 of 30 rows differ). Acceptance 72% at 1 draft, 90% at 6 with confidence 0.8. Results, draft against none: 2K 300 tokens 12.48 and 14.05 (two builds) against 12.69 and 13.33; 8K 200 tokens 12.50 against 11.56; a code prompt 13.48 against 11.65. A draft pass 8.8 ms; consecutive tokens share only 1.43 of 4 experts a layer, so a verify row costs nearly a pass. Gain ~0-16%, and it takes 2.2 ms a token back in lost VRAM: optional, not default.

**Prompt**: wave 1 read early, half route, XMX attention; dequant vector stores done. Now (profiled, 11 s with syncs): expert products 63%, read wait 12%, attend 11%, projections 4%, indexer 3%, router 4%. Fused indexer on XMX: at most the 3%, not done. Routed-only staging: 92% of a 2K pass's experts have over 8 rows, nearly all needed. Direct products below 8 rows: 8% of the experts. Reader threads: the read wait is 12%. Page cache in place of WILLNEED. Chunk: 4096 stays (2K 1100: 181; 8K 2048: 252, 3000: 322, 4096: 373). Split-K flash kernel in the fused decode. What bounds pp: each expert writes then reads 38 MB of half three times at the card's ~500 GB/s; on Arc the fused decode-in-local XMX GEMM has been slower (skill notes), not retried here.

**Decode overlap**: a token's layers depend on each other; a hop is one 24 KB round through the host (read-back 10-16 us). The only parallel work would be the draft beside a verify pass, and a verify pass needs the draft first. Nothing to overlap.

### Floor

Device a token (all-VRAM layers 0.42 ms B65, 0.24 ms B70; attention 20 ms): **no misses at all: ~45 ms, tg 22**. Misses in steady state: 5.5 file x 3.4 ms + 15 pinned x 0.55 ms = ~27 ms: ms 72. tg 20 needs more residency (VRAM + pinned) than 80 of 111 GiB, about 8 more GiB of pinned for each 10 ms, past the 32 GiB cap, or a smaller quant.

### Exactness

| prompt | min l_out | min kqv_out | result_output | next token | decode-after-prefill logits cosine |
|---|---|---|---|---|---|
| `short` | 0.999849 | 0.999966 | 0.999957 | same | 0.9540083 |
| `long` | 0.999848 | 0.999988 | 0.999756 | same (" +") | 0.9797325 |

Against the third round's 0.9555839 / 0.9795699 the decode column moves -0.0016 / +0.0002. Greedy text changes again at near ties (Q8_1 on the shared down from byte 656, B65 first from byte 49). Two runs of the final path byte-identical; `spec-check`: verify rows bit-identical to one-token decode.

## Fifth round (2026-10-30): more residency, and why runs differed

B65 + B70, the same prompts, with the expert profile. ms a token is the fixed 300-token text. Start of the round (151d754): pp 235 at 2K, 371 at 8K, tg 13.21 (300 tokens), 72.9 ms. End: **pp 264 / 429 / 333** (2K / 8K / 31K), **tg 14.19 over 300 tokens, 14.08 / 12.69 / 12.42** over 200 / 200 / 100, **66.9 ms** (three runs 66.8-66.9). tg 20 not reached.

### The cause of the varying runs

Decode's reads went through the page cache, including experts not in any tier. Over 300 tokens that filled the free memory (MemFree under 1 GiB, Cached up to 58 GiB in the slow runs), woke reclaim and compaction, and the runs came out at 67-96 ms a token. With only the warm tier cached and everything else O_DIRECT, five runs gave 67.3-67.5 ms and six greedy runs tg 13.82-14.04, pp 246-257, no outlier, the same text. Three runs with decode's reads cached again: 70.5, 85.8, 76.1 ms. Not the cause: read wait (13-15 ms in slow and fast runs alike), CPU placement (both CCDs still gave a slow run), GPU clock. The mechanism between reclaim and a slower token was not traced; the correlation and the fix are measured. The 147-160 pp outliers have not come back in the 6 repeats and the ladder; that is all that was checked.

### Warm tier and pinned memory

Warm tier: at load the model's stale cache is dropped (`posix_fadvise`, own files only), the next-ranked experts are read twice (so they are on the active list), up to what leaves `NS_MM31_CACHE_MARGIN_GIB` available. No mlock, no tmpfs; Mlocked stayed 0. Other reads in prompt passes and decode are O_DIRECT.

| pinned GiB | warm GiB | ms a token | file reads a token | MemAvailable min, MiB | |
|---|---|---|---|---|---|
| 32 | 0 | 74.3 | 5.5 | 24,783 | margin 10 |
| 32 | 14.5 | 70.7 | 5.5 | 24,746 | margin 10 |
| 36 | 10.4 | 69.3 | 4.6 | 20,540 | margin 10 |
| 40 | 7.3 | 67.4 | 3.6 | 16,643 | margin 10, five runs |
| 40 | 0 | 68.9-70.1 | 3.6-4.2 | 16,432-18,533 | two runs |
| 40 | 12.4 | **66.8-66.9** | 3.6 | 16,392-16,732 | margin 4, three runs: the default |
| 44 | 2.2 | 66.5 | 2.9 | 12,314 | margin 10, one run |
| 48 | | 68.6-69.5 | 2.1 | 8,092 | before the fix: below 8 GiB, stop |

Per GiB: pinned 32 -> 40 gives 0.4 ms a GiB; the warm tier 0.25 ms (14.5 GiB: 3.6 ms) at 32 and ~0.2 (12.4 GiB: 2-3 ms) at 40. **Pinned is worth about twice the cache per GiB**, and the cache is reclaimable. **Default: pinned at most 40 GiB** (44 was the last clean step, one back), cap `PIN_DEFAULT_GIB`, no other clamp (`NS_MM31_PIN_GIB` is taken as given); it still leaves 16 GiB available at load, so with memory held elsewhere it takes less. 48 is the rule's stop and was not rerun after the fix. No swap activity above 34 MiB, no GPU reset, in any run.

### Peaks at the default (for the front door's gate)

| | |
|---|---|
| VRAM a card | B70 31,114 MiB, B65 31,211 MiB at a 31K prompt (1,557 MiB free), gpustat at 50 ms samples (it refreshes every 3 s); 1,654 / 1,669 free at 2K. `vram_gib` in the card: 30.5 |
| pinned | 39 GiB of experts + 1.04 GiB of staging, 40.0 GiB |
| host, without reclaimable cache | MemTotal less the lowest MemAvailable: 62,864 - 16,392 = 46,472 MiB, 45.4 GiB for the whole box, of which about 3 GiB with production idle |

### Round 6: the draft and pinned-only guesses, again (seed profile, interleaved, greedy, 300 tokens, no second drive)

| prompt | none pp / tg (3 runs each) | draft, 6 tokens, confidence 0.8, pp / tg | accepted |
|---|---|---|---|
| 2K | 254.9 / 13.95 | 243.7 / 14.25 | 125 of 138 |
| 8K | 417.8 / 11.89 | 395.6 / 13.20 | 179 of 189 |
| code, 190 tokens | 58.2 / 12.48 | 55.0 / 13.76 | 163 of 173 |
| 31K (1 run) | 330.1 / 11.22 | 325.1 / 13.34 | 175 of 178 |

One draft a step loses: 2K tg 13.3, code tg 12.0 (before the reserve below). The text is the same in every run of a prompt. Spreads are about 0.03 tg. So the draft at 6 / 0.8 gains tg 2% at 2K and 10-19% beyond, costs pp 1.5-5% and about 2.0 GiB of B70 experts plus a 2 GB file. It is not on by default: `NS_MM31_DRAFT` stays unset, and the card does not name the file. Through a server and with two sessions: round 7, below.

The first set (without the reserve) peaked at 31,716 MiB on the B70, about 1,050 free, under the 1.5 GiB floor: 2K tg 14.52, 8K tg 13.50, code tg 13.9. The sizing now keeps 700 MiB more on the last card with a draft: at most 30,989 MiB, 1,779 free. Those are the numbers in the table.

Guesses that only copy pinned experts (`NS_MM31_AHEAD=3`, `NS_MM31_AHEAD_READS=0`): pinned swaps a token 17.7 -> 9.4 and fixed text 67.8 -> 71.7 ms, tg 13.94 -> 13.15; with 4 empty places (`NS_MM31_FREE=4`) 75.0 ms, tg 12.55. The routers' memory takes VRAM hits back (206.7 -> 214.8 hits a token is only part of it), and the second read-back and host work cost more than the copies saved. Off.

### Round 6: second drive, slices, CPU products (none kept)

**Second drive, all experts** (`NS_MM31_STREAM_ALL=1`, 111.3 GiB on the 2TB, 7 minutes to write): tg 13.91 -> 13.44 at 2K, 11.72 -> 11.16 at 8K, 11.19 -> 10.72 at 31K; pp no better. **A 50% share** (2K, 8K, warmed baseline): pp 254 -> 257 and 415 -> 411, tg 13.94 -> 13.67 and 11.90 -> 11.55. Both copies deleted. A run straight after a run that read another file can lose up to half its pp (2K 118.8, 8K 270), probably the page cache; a throwaway run first removes it. So compare pp only on warmed runs.

**Half slices of 8 MB in prompt passes**: pp 257 -> 227 at 2K, 416 -> 384 at 8K, 327 -> 354 at 31K, and the greedy text changes at every length. Removed.

**What a pinned expert costs** (fixed text, 279 tokens, 3 runs, least squares by token): ms = 46.7 + 0.68 pinned + 2.6 file, about 11 ms a token for 16.4 pinned swaps. A pure GPU product is ~0.12 ms of that (4 experts: gu 276 us, down 203 us, B70).

**Products on the host** (commit b73bc9d, since reverted; AVX2 and FMA on 8-float groups, not the AVX-512 VNNI of deepseek2, since these products are float). One expert alone, Ryzen 9 9950X, ms by threads:

| types | MB | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| IQ1_M / IQ3_XXS | 15.5 | 7.85 | 2.01 | 1.06 | 0.77-1.01 | 0.68-0.70 |
| IQ2_XXS / IQ3_XXS | 17.0 | 5.32 | 1.38 | 0.72-0.80 | 0.49-0.53 | 0.46 (one median 1.16) |
| IQ2_XXS / IQ4_XS | 19.8 | 5.93 | 1.57 | 0.81 | 0.56 | 0.50 |

On the same bytes and synthetic activations the output matches the silo's kernels to cosine 1.00000000, rel err 6-8e-7. In the engine (pinned experts on the host while the card runs the VRAM group, promotion copies still issued, the output row uploaded by a kernel), 0.78 ms each at 16 threads and 0.74 at 12: fixed text 66.5 -> 68.1 and 67.7 ms after the first 50 tokens, tg 13.95 -> 13.41 and 13.36 at 2K. The host's time from doorbell to queued VRAM experts went from 16 to 50 us a layer, about 2 ms a token. Not exact either: perplexity 1.6961 -> 1.7062 and the greedy text changes (same across runs and thread counts). That is more than the rel err of one product explains, and the cause was not found. The 8K tg with it (12.7-12.9 against 11.90) is on different text, so it is no gain.

### Round 7: the draft through a server (2026-10-31)

The build copy's own engine server on a spare port, 32K context, 2 sessions, as production; request by request in `~/src/m3/a7.py` (OpenAI streaming); VRAM from the process's DRM fdinfo at 50 ms (the gpustat file refreshes more slowly). Draft 6 tokens at confidence 0.8 against none, medians of 3 servers:

| prompt | tg none | tg draft | read s none | read s draft | accepted |
|---|---|---|---|---|---|
| code, 300 | 10.74 | 11.42 | 3.5 | 3.7 | 163 of 173 |
| prose, 300 | 10.53 | 10.80 | 4.0 | 4.3 | 160 of 169 |
| 2K, 300 | 11.29 | 12.07 | 9.8 | 10.2 | 150 of 158 |
| 8K, 200 | 13.01 | 14.14 | 22.8 | 24.1 | 104 of 106 |
| 31K, 300 (1 server) | 9.51 | 10.52 | 101 | 104 | 179 of 194 |
| free prose, 150, repeated | 16.4 | 14.2 | 3.5-4.2 | 3.9-4.0 | 21 of 21 |

Slower here than `generate` because the server's two 32K sessions take VRAM from the experts. With pinned at 40 (below) one server: code 11.20, prose 10.46, 2K 11.75, 8K 13.95, 31K 10.60, reads 3.9 / 4.5 / 10.5 / 25.5 / 108 s: tg about +4 / -1 / +4 / +7 / +11%, pp -6 to -11%.

Correct: the text equals no draft for streaming, two at once, a request joining mid-decode, a third queued, `max_tokens` 1-200 (counts exact), a natural stop, a client gone after 1 / 40 / 41 / 97 chunks (the same prompt after gives the uncut text), one of two cut, a client gone during a read, logprobs, `logprob_chain`, the end of a 2,400-token context. No GPU reset. VRAM peak at 50 ms: B65 1,538 MiB free, B70 1,754 (none: 1,549 / 1,554).

What is not free or not right:

- **Host memory**: through a server pinned is 43 GiB, not 40 (`mirror_bytes` from the server skips `PIN_DEFAULT_GIB`; the plain production card too). With the drafter MemAvailable fell to 6,373 and 7,778 MiB, under the 8 GiB floor (none: 8,134-12,394). 2 GiB more cache margin did not help (7,020: the warm cache is reclaimable). `NS_MM31_PIN_GIB=40` did: low 11,415 MiB, on the single chats, interleaved turns and long prompts only; the two-at-once and cancel cases were not rerun at 40. So 59.4 - 11.4: `ram_gib` 48.
- **Stalls**: one solo request with the drafter at 43 GiB paused 23.5 s (tg 4.45, the sampler itself stalled 3.1 s); baseline had 16-20 s on small requests. Memory pressure is the suspect, not traced. At 40 GiB no solo chat paused, in one shortened run, but a small request right after the 31K prompt still took 18 s, as with no draft, so the small-request stall is not the drafter's.
- **Two sessions take no drafts** (`step_together`) but still pay for the drafter's caches on each pass: one session tg 5.58 -> 4.25, the other level. The drafter's pass also runs when its confidence stops it, so free prose loses 13%.
- **Live reuse at a length stop**: when a step's accepted drafts run past `max_tokens`, the session holds tokens past the answer, the next prompt is no extension of it, and the turn reads from a checkpoint: reused 190 against 309, read 3.0 against 1.6 s, and that turn's text changes (a different split of prompt and decode). The fix is in `glue/serve` (trim to the common prefix with a rollback), not done here.
- **Checkpoints do not hold the drafter's caches**: after a restore they hold another conversation's. Measured: no cost, 125 of 137 accepted with the caches overwritten by two 8K prompts and with its own, the same text. Why it does not matter was not traced.

Result: correct, within the limits at 40 GiB pinned, a small gain. The card `minimax-m3.1-flash-iq1m-draft` says what it needs (`NS_MM31_DRAFT: '{draft}'`, `NS_MM31_DRAFTS: '6'`, `NS_MM31_DRAFT_PMIN: '0.8'`, `NS_MM31_PIN_GIB: '40'`, the file, `ram_gib: 48`); it is not registered, and the plain card is unchanged.

### Round 7: prompt speed, measured (none kept)

Baseline (warmed, 3 runs) pp 256-261 at 2K, 420-424 at 8K. **No gain was kept.** Text below is the greedy 300 tokens.

**Threads**: the lead was gaps in a static schedule (Intel reports up to 15% of MoE GEMM time). The prompt path's GEMM is oneMKL's, closed, so it cannot be read or changed; the silo's kernels are decode's and `to_half`. Per-thread timing could not be taken: these cards have no device clock in kernels (`ext_oneapi_clock_device` is false; `to_half` with stamps refused to run), and OA counters need root. A bound instead, kernel bench on real weights, 38 MB half matrix, microseconds, best of 8:

| card | bare read of it, all threads | GEMM 8 / 32 / 64 rows | GEMM 128 / 256 / 512 rows | 64 rows in 2 launches |
|---|---|---|---|---|
| B70 | 59-63 | 65-91 / 56-76 / 70-105 | 101-179 / 141 / 158 | 76 |
| B65 | 65-66 | 87 / 68-78 / 91-95 | 185-190 / 179 / 270 | 128-130 |

So at the typical 64 rows a GEMM is 16-20% above the card's read time on a B70 and 28-40% on a B65, compute included, and the GEMM is 27% of a pass: **perfect scheduling could give 4-7% of pp at the very most**. Prepacking does not apply to weights dequantized a pass. Not pursued. `vllm-xpu-kernels` (Apache 2.0) was read and nothing copied: its persistent work-groups are Xe3 kernels pulling (expert, slice, tile) from one launch.

**Where a 2K pass goes** (`NS_PROFILE=1`, 10.1 s with syncs): expert products 70%, attention 12% + projections 4% + indexer 3%, router 4%, read wait 3% (12% in round 4's profile; this run is warmed and has the warm tier). Of the products: 4.05 s `to_half`, 2.25 s GEMM. 21.6K products at the bench's 60-96 us would be about 1.6-2 s of `to_half`, so about half of it is the compute queue waiting for copies of 4,077 off-card experts (16 MB over PCIe, 0.65 ms each): inferred, not measured apart.

**Idle by card** (each engine's busy at 50 ms from fdinfo): 2K: only the B70 58%, only the B65 40%, both 0%, neither 2%. 8K: 38 / 28 / both 33 / 1. Half the box idles at 2K: the real load problem, not threads.

| tried, warmed and interleaved | pp 2K | text | verdict |
|---|---|---|---|
| the idle card takes the file's experts (reads into its ring, its GEMM, rows back by host), one-chunk passes, 3 runs | 217 -> 217-224 | changes, and differs between runs | noise; dropped |
| also 50% / 100% of the pinned experts (host copy into its ring) | 217 -> 172 / 136 | changes, 50% differs between runs | slower; dropped |
| both copy lanes (`NS_MM31_LANE=2`), 3 runs | 217 -> 216-218 | same | nothing |
| small tables by kernel from a pinned ring, not memcpy, 3 runs | 256-258 -> 256-261; 8K 417-418 -> 420-424 | same | about 1%; dropped |

The baseline of the first three rows reads 217 because their build, with `expert_compute` split into two functions (more arena marks and one more upload an expert), lost 15% with the helper off (254 -> 217, six runs, text the same). Upload count is not why: the last row's one upload and three uploads read the same. Traced below: it was where the upload sat, not the arena.

### Round 7: alignment (none kept)

Pinned 40 GiB as production, 2K prompt, warmed, interleaved. No idea gained; no code stayed. Text was the same in every run below (40 and 300 tokens).

**The 15% of the split**, one binary, 3 runs each, 40 tokens:

| order in `expert_compute` | pp 2K | tg |
|---|---|---|
| main: tables, then the products, then scatter | 255.2, 256.2, 256.2 | 13.17-13.23 |
| the split: rows' table, products, rewind, three tables, scatter | 215.9, 202.7, 217.3 | 13.17-13.21 |
| the same without the rewind | 215.8, 216.9, 215.8 | 13.14-13.18 |
| result buffer first, one set of tables before the products | 253.3, 257.0, 257.6 | 13.20-13.22 |
| main with one more upload before the products | 254.3, 254.6, 254.7 | 13.15-13.22 |

So arena order and rewinds do not matter. Uploading tables between an expert's GEMM and its scatter costs about 1.5 s over 21K products' experts (about 200 us an expert); the same uploads before the products cost nothing. Why that one place costs was not traced (a runtime memcpy behind a just-finished GEMM is the suspect). Main already uploads first and the kernel copy of the tables above gained 1%, so no better order is left to gain: rule, keep runtime uploads ahead of the products.

**Expert start offsets**, kernel bench on real experts copied to each start offset from 2 MB alignment (0, 8, 16, 32, 48, 64, 128, 192, 256, 512, 1K, 2K, 4K, 8K, 64K, 1M, 2M), one expert a launch, best of 8, 3 rounds; GB/s of weights:

| kernel | B70 | B65 |
|---|---|---|
| gu IQ1_M | 227-235 | 105-115 |
| gu IQ2_XXS | 243-249 | 106-114 |
| down IQ3_XXS | 209-220 | 144-146 |
| down IQ4_XS | 144-146 | not on this card |
| to_half IQ1_M | 59-61 | 46-47 |
| to_half IQ2_XXS | 70-73 | 57-60 |
| to_half + half GEMM, by the half matrix's offset (0 to 2M) | 269-272 | 184-186 |

The spread is no larger than a repeat at one offset (B65 gu at offset 0: 101-110) and follows no pattern. Row and block alignment were not swept apart: blocks are 56-136 bytes and rows are not aligned to anything these offsets could fix together. So the ceiling of these kernels is not alignment; if Nemotron's expert-up has the same limit, alignment will not lift it (not tested on Nemotron).

**VRAM pages**: no root, so no debugfs. `zeVirtualMemQueryPageSize` says 64 KB under 2 MB and 2 MB from 2 MB up, on both cards; that is for a reservation, not proof for `malloc_device`. A read probe (ns a read by stride over 1 GiB, B70 fresh buffer / expert slab): 64 KB 0.04 / 0.04, 256 KB 0.05 / 0.05, 1 MB 0.10 / 0.10, 2 MB 0.17 / 0.17, 4 MB 0.33 / 0.34, 16 MB 0.32 / 0.33; B65 the same shape. No step at 64 KB and one from 1-4 MB: it looks like 2 MB pages already, in fresh buffers and slabs alike; indirect. Expert slabs are 1 GiB, their starts aligned to bits 21-25; slots inside them to bits 14-16. The same expert in its slot and 2 MB aligned in a fresh buffer, 4 rounds: B70 gu IQ1_M 228-229 / 228-230, IQ2_XXS 240-245 / 243-245, down IQ3_XXS 209-215 / 208-213; B65 gu 104-105 / 105-110, 100-105 / 101-106, down 140-145 / 140-146. Nothing to gain.

**Pinned on 2 MB pages**: `madvise(MADV_HUGEPAGE)` on all 41 slabs was accepted, but the process's `AnonHugePages` stayed 0 (sampled every 2 s), so **2 MB pages were not obtained** and this does not test them; these pages seem not to be plain anonymous memory. It would not matter: a pinned expert into VRAM runs at 28 GB/s on both copy lanes with or without the call (the x8 link), and `to_half` reading pinned memory at 23 GB/s. With it, 3 runs against 3: pp 244.8, 258.2, 253.9 against 257.0, 233.8, 254.5; tg 13.53-13.94 against 13.58-13.94.

### Round 7: other alignments (none kept)

Pinned 40 GiB, 2K prompt unless said, warmed, interleaved, 3 runs each, 300 tokens. Baseline pp 254.6-257.4, tg 13.93-13.95.

**Rows and blocks in the weights** (real gate and up rows repacked losslessly, a copy of the gu kernel with the strides as arguments, us for one expert, three rounds; only gu was bounded, the down kernel not):

| layout | size | B70 IQ1_M | B70 IQ2_XXS | B65 IQ1_M | B65 IQ2_XXS |
|---|---|---|---|---|---|
| as stored | | 37-38 | 44 | 81-83 | 97-99 |
| rows to 64 bytes | +0-1% | 37-38 | 43-45 | 79-82 | 97-99 |
| rows to 256 | +13-14% | 38 | 44 | 81 | 97-100 |
| rows to 4096 | +159-205% | 38 | 44 | 82 | 96 |
| blocks to 8 bytes | +0 / +9% | 37 | 46 | 81-83 | 104-105 |
| blocks to 64 bytes | +14 / +94% | 38 | 52-59 | 82-84 | 115-118 |
| the same bytes kept in cache: one block | | 44 | 52-53 | 96-99 | 117-119 |
| the same bytes kept in cache: 4 rows | | 47-48 | 50-51 | 94 | 115-116 |

(A sixth, cold sample of 61-74 us at the start of the B70 IQ2_XXS rows is left out.) Reads that never leave the cache are no faster, slower if anything: this kernel is not bound by memory or by how rows meet lines, so no padding can bring it towards 520-570 GB/s. And padding would cost residency: +14% on the IQ1_M matrices alone is several GiB of VRAM at 2.2 ms a token per GiB, for nothing back. Not tried: a layout that changes the decode itself (a record a group, one load a lane): at least 2.3x the bytes for IQ1_M, computed.

**Batches**: oneMKL's half GEMM goes in steps by rows, best of three rounds, us:

| rows | 32 | 33 | 64 | 65 | 128 | 129 | 192 | 256 | 320 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| B70, gate and up | 70 | 67 | 72 | 83 | 99 | 139 | 139 | 142 | 128 | 159 |
| B65, gate and up | 78 | 88 | 93 | 140 | 178 | 137 | 146 | 178 | 223 | 271 |
| B70, down | 66 | 62 | 74 | 76 | 91 | 82 | 90 | 100 | 112 | 170 |
| B65, down | 78 | 77 | 83 | 136 | 164 | 141 | 149 | 185 | 217 | 266 |

So a larger batch can be quicker. Padding where the bench says (B65 gate and up 65-128 -> 129, B70 129-319 -> 320, down 97-128 -> 129; the extra rows' results unread) in the engine: pp 2K 239.5-257.2 against 254.6-257.4, pp 8K 420.2-420.6 against 414.4-416.8, about +1%. **The text changes** (different GEMM paths, near ties); 8K tg 13.47 against 11.88 is on different text, so it is no gain. Dropped. Chunk sizes as multiples were not retried: rows an expert takes depend on routing, not on the chunk, and earlier rounds swept chunks.

Fused decode rows (`NS_MM3S_RPS`, `NS_MM3S_KU`; 4 experts a launch, GB/s): RPS 2, KU 1 stays best for gu (B70 IQ1_M 276; RPS 1 219, RPS 4 226, RPS 4 KU 2 5). Down alone likes RPS 4 (B70 IQ3_XXS 316 -> 347, B65 172 -> 188), and with gu left at 2: **tg 14.00-14.02 against 13.93-13.95, +0.5%, text the same, pp the same** (253.8-255.4). Real but too small to keep; a test-only variable, not on main.

**Drive** (Crucial P310 on ext4: `max_sectors_kb` 128 hardware limit, optimal I/O 4 KB, no scheduler; 16 MB O_DIRECT at random offsets of a shard, median ms over 40, three rounds, Python threads):

| reads | ms |
|---|---|
| 16 MB, one read | 3.05-3.29 |
| 8 MB x 2 / 4 MB x 4 | 3.04-3.49 / 3.08-3.50 |
| 2 MB x 8 (today) | 3.26-3.48 |
| 1 MB x 8 / x 16 / x 32 | 3.14-3.38 / 3.37-3.46 / 3.28-3.48 |
| 512 KB x 16 / x 32 | 3.24-3.37 / 3.08-3.56 |
| 128 KB x 32 / x 64 / x 128 | 3.25-3.49 / 3.24-3.57 / 4.53-5.15 |
| 2 MB x 8 starting aligned to 64 KB / 128 KB / 1 MB / 2 MB / 16 MB | 3.11-3.46 / 3.08-3.51 / 3.14-3.44 / 3.18-3.40 / 3.13-3.34 |
| 2 MB x 8 starting 4 KB past a 128 KB edge | 3.09-3.36 |

The drive reads about 5 GB/s however it is asked; no start alignment shows. One read of a matrix in the engine (`NS_MM31_SPLITREAD=1`): pp 251.2-256.2, tg 13.86-13.90, slightly worse. Stripe and NAND geometry are not exposed, so boundaries could only be tested by these offsets. The reader's bounce copy (O_DIRECT into its own buffer, because GGUF offsets are not page aligned) is 0.34-0.38 ms of 16 MB in this bench: about 1.3 ms of a 67 ms token at 3.6 file reads, a bound of 2%, needing a slot layout matched to each matrix's file offset. Not attempted.

**CPU**: the 9950X's CCDs are CPUs 0-7 + 16-23 and 8-15 + 24-31, one I/O die; sysfs gives no NUMA node for either card. `podman --cpuset-cpus` fails rootless (no cpuset controller) and `taskset` on the runner does not reach the engine (read back: 0-31), so affinity was set from the host on the running process and read back. Same order, all threads:

| set | pp 2K | tg | launch on the queue, B70, us | read-back, B70, us |
|---|---|---|---|---|
| all | 229.6, 255.1, 257.0 | 13.85, 13.91, 13.36 | 9.6 | 9.1 |
| CCD 0 | 253.2, 254.9, 257.6 | 13.92, 13.45, 13.35 | 10.6 | 8.2 |
| CCD 1 | 256.4, 257.1, 256.8 | 13.66, 13.18, 13.34 | 9.9 | 10.6 |

No gain; the later rounds drop together on every set, which is the box, not the set (not traced). One launch-bench run a set. Threads pinned one by one to a matched card were not tried: nothing here suggests a side to match.

### Through the front door

A private `serve` from the build copy (own port, registry and config) with production idle, 8K context:

| | |
|---|---|
| load | 26.6 s in the engine; the first chat answers in 31-38 s |
| VRAM, loaded and between requests | B65 30,579, B70 30,698 MiB |
| VRAM during a 2,144-token prompt | 31,069 / 31,076 MiB (about 1,595 free, gpustat refreshes every 3 s), back to 30,579 / 30,698 after: the prompt scratch is freed |
| speeds through serve | pp 223 (2,144 tokens in 9.6 s), tg 13.20 over 24 tokens, two sessions at once answered |
| host | MemAvailable 59,447 MiB idle, lowest 12,435: 45.9 GiB, so `ram_gib: 46` (the 45.4 above is the whole box, on a different baseline). The warm cache, `ram_cache_gib: 12.7`, is on top and counted as available by the kernel |
| unload | 0.7 s; VRAM 0 / 0, MemAvailable back to 59,405 MiB |

Eviction, with PaddleOCR-VL as the other load: without a header MiniMax goes at once and the load takes 3 s; asking only host memory (`vram_gib` 0.05, `ram_gib` 20) evicts it too; with `X-Keep-Up-S: 150` it stays, the load gets `no_room` after 91.7 s (for VRAM and, separately, naming host memory), and after the 150 s the same load evicts it and succeeds in 6 s. No GPU resets, lowest MemAvailable 12.1 GiB. Not run this way: a video job (the room rule for a waiting worker has unit tests only).

### What did not help

- **GPU clock**: active samples sit at the maximum (B70 2800, B65 2400 MHz, p50 and nearly p10 too); the card is simply idle during the other card's layers. A keep-alive kernel every 200 us: 84.1 and 74.1 ms, measured before the variance fix, no sign of a gain; not rerun. Pinning the minimum frequency has no gain to report: there is no lower clock to lift. Not tested with root.
- **Attention** (20 ms) by step, seconds over 300 tokens, B65 24 layers / B70 36: q k v 0.84 / 1.10, o 0.65 / 0.81, attend 0.41 / 0.60, indexer 0.16 / 0.24, top-k 0.10 / 0.14, scores 0.05 / 0.07, norms rope stores 0.04 / 0.05. q, k, v and o are 11 of the 20 ms, on the kernels already at 520 GB/s on the B70 (ceiling about 570), and no form of the Q5_K weights is cheaper to read without changing them. The rest are single milliseconds. No kernel written; no VRAM-for-ms trade found.
- **VRAM from weights**: nothing done. The embeddings were never in VRAM (a row from the file each token); the head lives on the last card only. Decode reads every other weight. The indexer as half would free ~0.22 GiB a card, ~0.5 ms by the slope: computed, not run. gpustat keeps 1.5 GiB free (1,557 MiB at the tightest), so the spare stays 1.9.
- **Read path**: the shared expert and the VRAM experts are queued before the reads are waited for (code, and read layers cost 3.3-3.6 ms against 0.24-0.41 clean with a 2-3 ms wait), the misses of a layer are read at once. Second copy on the 2 TB: not rerun, it read at 1.8 GB/s in round 3 and a stripe goes at its slowest drive. Second-drive copy on the system drive for off-card experts is kept as an option.

### Exactness

| prompt | min l_out | min kqv_out | result_output | next token | decode cosine |
|---|---|---|---|---|---|
| `short` | 0.999849 | 0.999966 | 0.999957 | same | 0.9540083 |
| `long` | 0.999848 | 0.999988 | 0.999756 | same (" +") | 0.9797325 |

Unchanged from round 4. Greedy text: the same md5 across six runs at margin 10 and the final margin 4 run. The round 4 note on a byte-838 difference was wrong about its timing: that pair came before the demotion fix, and no later pair has differed.

## Round 8 (2026-11-08): profile again, CPU products, kernels, drafter, copies

B65 + B70 as deployed (B65 first with 24 layers, expert profile seeded, pinned 40 GiB, warm tier 11.5 GiB), main 19bd250, the ladder prompts and
the corpus prompts c1 (code), c2 (prose), c4 (chat). Greedy, 300 tokens, warm runs; "ms a pass" is the token log's mean over the 299 one-token
passes (`NS_MM31_TOKLOG`). Another agent's CPU reference runs (python, 16 threads) and AOT builds ran on the host through most of the round
(load 5-15, logged with each run); a repeat of one binary moved 74.9 -> 80.7 and 76.0 -> 91.9 ms a pass, so every comparison below is interleaved and
repeated, and nothing smaller than the spread of its own repeats is called a gain.

### Where a token goes (2K, 300 tokens, `NS_PROFILE=gpu` device stamps; the profiled run reads tg 12.30, 76.1 ms a pass)

| | ms a token |
|---|---|
| attention (B65 24 layers 7.4, B70 36 layers 12.4; q k v 6.5, o 4.9, attend 3.3, indexer 1.3, top-k 0.8, scores 0.4, norms/rope/stores 0.3) | 19.8 |
| router | 1.1 |
| host turnaround, router read-back to the shared expert queued (profiled) | 7.0 |
| experts, layers with all four in VRAM (B65 0.405 ms x 16.5 layers, B70 0.238 ms x 20.7) | 11.6 |
| experts, layers with a pinned expert and no read (B65 0.643 ms x 3.3, B70 0.910 ms x 12.4) | 13.3 |
| experts, layers with a file read (B65 4.94 ms x 1.2, B70 4.01 ms x 2.8) | 17.2 |

Uses a token: 202.8 VRAM, 20.9 pinned, 4.3 file (B65 94.0 / 4.6 / 1.5%, B70 86.0 / 11.9 / 2.1%). The host waited 49.8 ms a token for the routers'
read-backs (the GPU working) and 14.2 ms for reads. So: misses (~9 ms of pinned copies, ~15 ms of file layers) and attention (20 ms, q/k/v/o at
~400 GB/s) bound a token; the VRAM experts' compute is 12 ms. The same run unprofiled: pp 238, tg 13.14, 76.0 ms a pass (fifth round's 66.9 ms was the
fixed-text measure; today's greedy 2K text uses 20.9 pinned experts a token, not 16.4).

### CPU products on AVX-512 (`NS_MM31_CPU_EXPERTS`, the user's approval of 2026-10-30): exact, no gain, removed

Bound first. Host read bandwidth (Ryzen 9 9950X, DDR5): 17 MB at 54 GB/s on 1 thread, 58 GB/s at 4-16 threads: an expert can not cost less than
~0.29 ms on the host, against 0.68 ms through PCIe, so at most ~0.3 ms x 20.9 = ~6 ms a token. The products (`cpu.rs`: IQ1_M, IQ2_XXS gate/up and
IQ3_XXS down, a 64-bit gather a group, the GPU's weight formula bit for bit, a spinning pool) match a scalar port of the silo's formulas at relative error
2e-7 (unit test). Standalone, 96 cold experts, an expert in ms by threads: 8: 0.63, 12: 0.44, 16: 0.42, 24: 0.38, 32: 0.38 (single-thread 4.2 ms).

The hook bug of the AVX2 attempt (b73bc9d): the reverted pool let a worker late from one job take an index of the next and run the *old* closure on it,
writing into the previous expert's freed buffers (here: `free(): invalid pointer`, a crash at 8 tokens). The new pool tags parts with the job's
generation; a 3,000-job unit test checks that no part runs twice or on another job. Also fixed: an expert multiplied on the host and promoted into
VRAM now leaves its copy's ticket on the place, so its next use waits for the copy.

| 2K, 300 tokens, ms a pass (2 rounds) | round 1 | round 2 | text |
|---|---|---|---|
| CPU off | 74.9 | 80.7 | f6ff55e5 (main's) |
| 16 threads | 77.0 | 73.9 | 764eca8a |
| 24 threads | 75.7 | 77.6 | 764eca8a |

In the engine a host product took 0.66 ms (16 threads) / 0.59 ms (24) against the bench's 0.42 / 0.38. Pinned memory is not the cause: the same
experts from their pinned places and from heap copies, with the cards idle, 0.36-0.47 ms both (`NS_MM31_KBENCH=1` with the switch). During decode the
products share DRAM with the cards' own DMA from host memory (promotions and demotions, 28 GB/s a card) and the drive's reads, and the GPU path already
overlaps the next copy with the current product, so 0.6 ms on the host buys nothing. The text differs from the GPU path's after a few dozen tokens
(the sums in another order at near ties) and is the same for 16 and 24 threads. Kept in history (the round's first commit), removed from the engine.

### Silo kernels: int8 grid table and sign bits (exact, slower, not kept)

The fused gu/down kernels are ALU-bound (reads that stay in cache are no faster, round 7). Two exact rewrites of the group decode:

- IQ1_M's grid in local memory as eight signed bytes (value - 1) a grid entry, so a weight is a byte-to-float, an add and a multiply (16 KB of local memory a
  work-group instead of 8);
- IQ2_XXS / IQ3_XXS signs by flipping the sign bit (a multiply by +-1 is exact) instead of a select and a multiply.

The greedy text was byte-identical to main's (300 tokens, md5 f6ff55e5): the weights are the same bits. Kernel bench (`NS_MM31_KBENCH=1`, 4 experts,
us; B65 repeats within 2%, the B70's swing +-50% run to run - its gu IQ1_M, the same code, read 153, 232 and 122 us - so only the B65 decides):

| B65 | main | both rewrites | sign bits only |
|---|---|---|---|
| gu IQ1_M | 231 | 489 | 231 (main's) |
| gu IQ2_XXS | 254-258 | 511 | 250-253 (main's) |
| down IQ3_XXS | 168-171 | 186 | 186-191 |

Both slower on the B65; the sign-bit form also halved pp in two of three runs (122, 109, 167 against 238, 174: the prompt path's dequantize-to-half
shares the group code). Removed; the silo is main's. `NS_MM31_KBENCH` now also prints the time per launch over 16 back-to-back launches.

### Drafter: backoff kept, doorbell for verify rows dropped

The DSpark pass costs ~9 ms whether its confidence head lets a draft through or not. **Backoff** (`NS_MM31_DRAFT_BACKOFF`): after a pass
that drafts nothing (with 2, the default, after two in a row), the next 1, 3, 7, 15 steps go without one; a pass that drafts ends it. **Doorbell for verify rows**: the router of a 2-7 row pass rings the
pinned bell like a one-token pass (the shared expert queued behind it). Both exact (verify is exact: every run of a prompt has the same text, also
without the drafter).

| tg, 300 tokens, 2 rounds | no draft | drafter (main) | + backoff + verify doorbell | + verify doorbell only |
|---|---|---|---|---|
| code c1 | 11.80 | 12.75 / 12.67 (163 of 173) | 12.43 / 12.39 (141 of 148) | 12.15 / 12.69 |
| prose c2 | 16.20 | 12.82 / 14.44 (46 of 48) | 14.15 / 14.89 (5 of 6) | 14.32 / 14.52 |
| chat c4 | 15.64 | 14.13 / 14.16 (60 of 70) | 14.39 / 14.23 (10 of 11) | 14.17 / 13.87 |

The verify doorbell is neutral (-2% to +5% by prompt, the same drafts): removed. The backoff gains 1-3% on prose and chat and costs ~2% on code
(it skips some passes that would have drafted). Backing off only after **two** empty passes in a row (`NS_MM31_DRAFT_BACKOFF=2`, now the default)
keeps both: code 12.71 / 12.52 (150 of 160 accepted; main's drafter 12.75 / 12.67), prose 14.72 / 14.99 (5 of 5; main's drafter 12.82 / 14.44,
backoff 1 14.15 / 14.89). So the draft card gains ~5-10% on prose and chat against main's draft card and stays within ~1% on code.

What the drafter costs when it drafts nothing: on prose a one-row pass takes ~70 ms with the drafter against ~62 without (its caches fed every pass, and
the 2 GiB of B70 VRAM it holds back from experts), so even with no drafter pass the draft card stays below the plain card on prose and chat. It pays
on code and long prompts only (round 7's server numbers).

### File reads copied piece by piece (`NS_MM31_PIECE_COPY`, default on): kept

A file-read expert was copied to VRAM after all its pieces (8 a matrix) were in. Now each piece goes to the copy lane as it lands, so the card waits for
the last piece's copy, not the read and then a 16 MB copy (~0.57 ms). Exact (the same bytes; the same text in every run).

| ms a pass (tg) | whole copy | piece by piece |
|---|---|---|
| 2K, 3 rounds | 76.0, 74.9, 74.5 (tg 13.14-13.39) | 73.9, 74.0, 73.9 (tg 13.50-13.52) |
| 8K, 3 rounds | 90.8, 89.3, 89.4 (tg 11.01-11.19) | 90.4, 87.8, 89.5 (tg 11.05-11.38) |
| 31K, 2 rounds | 93.8, 93.5 (tg 10.65, 10.69) | 96.1, ~93 (tg 10.34, 10.83) |
| code c1, 2 rounds (tg, no draft) | tg 11.76, 11.60 | tg 11.93, 11.97 |

31K round 1 piece-copy was a slow first-50-token start (111.5 ms) on a loaded box; round 2 was tg 10.83 vs 10.69. Same text md5 in every run (31K cf42207b, c1 6dbfdf2c).

-1.6% a pass at 2K, -0.7% at 8K, lower in every pair; pp unchanged (240-246 at 2K, 402-405 at 8K).

### Other things measured

- **Build copies**: rsync keeps the source's times, so after a sync cargo may keep a binary built from older sources (the first CPU-tier run ran
  without the tier). Touch the synced sources before `./build.sh rust`.

### Left

- The verify-pass rows still cost ~0.7 of a pass each (round 6): decode the shared experts of a verify pass once for all their rows (the fused kernels
  decode per (expert, row)); worth it only with a cheaper drafter.
- Attention's projections (11 ms) are at ~400 GB/s of the ~520 the same kernels reach alone: the gap is launch ramp-up on 35-74 MB matrices.
- tg 20 still needs more residency or fewer misses: the floor without misses is ~45 ms (round 4), today's misses ~24 ms.
