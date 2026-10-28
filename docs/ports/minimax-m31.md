# MiniMax-M3.1-Flash (Unsloth GGUFs): port notes

Prepared, not run. `llm/minimaxm3` has the model description (geometry from the metadata, every tensor by role and
shape, the row types the checked file uses); the engine and its silo (`kernels/llm/minimaxm3/silo`) are the work ahead.
Card to come: `minimax-m3.1-flash-iq1m`. Files: `unsloth/MiniMax-M3.1-Flash-GGUF`, `UD-IQ1_M/` (4 shards, 129 GB) at
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

`NS_MM31_FILE=<shard 1>` makes `cargo test -p nextsycl-llm-minimaxm3` check the real file against the description.
