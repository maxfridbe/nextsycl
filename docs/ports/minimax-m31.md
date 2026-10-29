# MiniMax-M3.1-Flash (Unsloth GGUFs): port notes

Engine written and exact (2026-10-28): `llm/minimaxm3` has the model description (geometry from the metadata, every
tensor by role and shape) and the engine (`engine.rs`: single GPU, MiniMax Sparse Attention, routed experts kept in
VRAM / pinned host memory or read from the file per pass); its kernels are `kernels/llm/minimaxm3/mm3.cpp` (rope,
SwiGLU-OAI, indexer scores, top-K, sparse attend) on the glm5next engine's generic dequant / GEMM / mmvq kernels. The
speed work (silo `kernels/llm/minimaxm3/silo`, second GPU, draft) is ahead; see "Results" at the end.
Card to come: `minimax-m3.1-flash-iq1m` (needs the shards' SHA-256 from the Hugging Face tree). Files: `unsloth/MiniMax-M3.1-Flash-GGUF`, `UD-IQ1_M/` (4 shards, 129 GB) at
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
- Types in UD-IQ1_M (shard 2's table): experts gate/up IQ1_M, down IQ3_XXS; attention Q5_K/Q4_K; dense and shared FFN
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

### Speed (first measurement, untuned, B70 alone; pp = prompt, tg = generation)

| context | pp | tg | notes |
| --- | --- | --- | --- |
| 179 tokens, 160 generated | 2.9 tok/s | 0.89 tok/s | 19 GiB experts in VRAM + 40 GiB pinned, the rest read from the file |
| 4,044 tokens, 48 generated | 15.9 tok/s | 0.96 tok/s | 19 GiB VRAM + 19 GiB pinned (the default at the time), chunk 1024 |
| 3,881 tokens (check run) | 24.6 tok/s | - | 36 GiB pinned, chunk 4096 |

What bounds it: the 120 GB of routed experts do not fit one 32 GiB card plus 40 GiB of pinned memory (61 GiB of RAM on the box),
so about half of each token's expert reads (4 of 128 per layer, 16 MB each) come from the SSD at ~3 GB/s - tg is a file-read
rate, not a kernel rate. Next: the B65 as a second card holding experts, a draft (DSpark), the silo kernels for the
IQ1_M/IQ2_XXS/IQ3_XXS mmvq and the sparse attend, and reading ahead the next layer's experts while this one computes.
