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

One draft a step loses: 2K tg 13.3, code tg 12.0 (before the reserve below). The text is the same in every run of a prompt. Spreads are about 0.03 tg. So the draft at 6 / 0.8 gains tg 2% at 2K and 10-19% beyond, costs pp 1.5-5% and about 2.0 GiB of B70 experts plus a 2 GB file. It is not on by default: `NS_MM31_DRAFT` stays unset, and the card does not name the file. Not run through a server or with two sessions.

The first set (without the reserve) peaked at 31,716 MiB on the B70, about 1,050 free, under the 1.5 GiB floor: 2K tg 14.52, 8K tg 13.50, code tg 13.9. The sizing now keeps 700 MiB more on the last card with a draft: at most 30,989 MiB, 1,779 free. Those are the numbers in the table.

Guesses that only copy pinned experts (`NS_MM31_AHEAD=3`, `NS_MM31_AHEAD_READS=0`): pinned swaps a token 17.7 -> 9.4 and fixed text 67.8 -> 71.7 ms, tg 13.94 -> 13.15; with 4 empty places (`NS_MM31_FREE=4`) 75.0 ms, tg 12.55. The routers' memory takes VRAM hits back (206.7 -> 214.8 hits a token is only part of it), and the second read-back and host work cost more than the copies saved. Off.

### Round 6: second drive, slices, CPU products (none kept)

**Second drive, all experts** (`NS_MM31_STREAM_ALL=1`, 111.3 GiB on the 2TB, 7 minutes to write): tg 13.91 -> 13.44 at 2K, 11.72 -> 11.16 at 8K, 11.19 -> 10.72 at 31K; pp no better. **A 50% share** (2K, 8K, warmed baseline): pp 254 -> 257 and 415 -> 411, tg 13.94 -> 13.67 and 11.90 -> 11.55. Both copies deleted. A run straight after a run that read another file can lose up to half its pp (2K 118.8, 8K 270), probably the page cache; a throwaway run first removes it. So compare pp only on warmed runs.

**Half slices of 8 MB in prompt passes**: pp 257 -> 227 at 2K, 416 -> 384 at 8K, 327 -> 354 at 31K, and the greedy text changes at every length. Removed.

**What a pinned expert costs** (fixed text, 279 tokens, 3 runs, least squares by token): ms = 46.7 + 0.68 pinned + 2.6 file, about 11 ms a token for 16.4 pinned swaps. A pure GPU product is ~0.12 ms of that (4 experts: gu 276 us, down 203 us, B70).

**Products on the host** (commit "expert products on the host's cores", reverted in the next commit; AVX2 and FMA on 8-float groups, not the AVX-512 VNNI of deepseek2, since these products are float). One expert alone, Ryzen 9 9950X, ms by threads:

| types | MB | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| IQ1_M / IQ3_XXS | 15.5 | 7.85 | 2.01 | 1.06 | 0.77-1.01 | 0.68-0.70 |
| IQ2_XXS / IQ3_XXS | 17.0 | 5.32 | 1.38 | 0.72-0.80 | 0.49-0.53 | 0.46 (one median 1.16) |
| IQ2_XXS / IQ4_XS | 19.8 | 5.93 | 1.57 | 0.81 | 0.56 | 0.50 |

On the same bytes and synthetic activations the output matches the silo's kernels to cosine 1.00000000, rel err 6-8e-7. In the engine (pinned experts on the host while the card runs the VRAM group, promotion copies still issued, the output row uploaded by a kernel), 0.78 ms each at 16 threads and 0.74 at 12: fixed text 66.5 -> 68.1 and 67.7 ms after the first 50 tokens, tg 13.95 -> 13.41 and 13.36 at 2K. The host's time from doorbell to queued VRAM experts went from 16 to 50 us a layer, about 2 ms a token. Not exact either: perplexity 1.6961 -> 1.7062 and the greedy text changes (same across runs and thread counts). That is more than the rel err of one product explains, and the cause was not found. The 8K tg with it (12.7-12.9 against 11.90) is on different text, so it is no gain.

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
