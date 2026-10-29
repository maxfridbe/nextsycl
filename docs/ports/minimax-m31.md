# MiniMax-M3.1-Flash (Unsloth GGUFs): port notes and assignment

Prepared, not run. `crates/minimaxm3` has the model description (geometry from the metadata, every tensor by role and
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

## What the references say (read 2026-10-28; verify each against the model's own code before relying on it)

- MiniMax Sparse Attention (paper arXiv 2606.13392; vLLM `vllm/models/minimax_m3/common/sparse_attention` and
  `indexer.py`; llama.cpp PR 24908, `src/models/minimax-m3.cpp`): block-sparse GQA. A lightweight indexer scores the
  causal context, the scores are **max-pooled over key blocks of 128 tokens**, and the **top 16 blocks** per query are
  kept per GQA group (4 indexer heads = the 4 KV heads, each shared by that group's 16 query heads). The query's own
  (local) block is always kept: llama.cpp adds a 1e30 bias to the `local_blocks` most recent blocks including its own.
  The main attention then runs over only the kept blocks (at most 16 x 128 = 2048 keys a query, whatever the context).
  Running it dense gives wrong answers (trained with the sparsity).
- Indexer: `indexer.q_proj` [6144 -> 4 x 128], `indexer.k_proj` [6144 -> 1 x 128] (one K head shared by the 4 indexer
  heads), each RMS-normed (`indexer.q_norm`, `indexer.k_norm`, [128]) and then **RoPE'd like the main q/k** (llama.cpp);
  scores are unscaled F32 dot products idx_q . idx_k, masked causally to -inf. The indexer keeps its own per-layer K
  cache (F32 in llama.cpp; keep the indexer projections at full precision: they are F32 in the file).
- Main attention: per-head RMSNorm of q and k (one [128] weight shared over heads), partial RoPE (first 64 of 128
  dims; base 1e7; the rope layout - NEOX or NORM - and any YaRN factors are NOT in the GGUF metadata: take them from
  the model's modeling code), scale 1/sqrt(128). Layers 0-2 attend densely (their FFN is dense too); the sparse path
  applies to layers 3-59 in llama.cpp ("the first three full-attention layers": the HF card), a 1M context.
- FFN activation is llama.cpp's **SwiGLU-OAI** (the gpt-oss form: clamped gate/up, `x * sigmoid(alpha * x) * (y + 1)`,
  alpha 1.702, limit 7.0 in ggml's `swiglu_oai` - confirm the constants and the clamp sides from ggml-cpu/ops.cpp and
  the HF code), for the dense layers, the routed experts and the shared expert. Routing: sigmoid gating, `exp_probs_b`
  added for the choice only, top 4, weights normalized then x2.0.
- Output: final RMSNorm, then the Q4_K head. The tokenizer is the GGUF's (gpt2/BPE, pre `minimax-m2`).
- References to read (all read-only - never execute downloaded code): the HF repo `MiniMaxAI/MiniMax-M3.1-Flash`
  (`modeling_*.py` with trust_remote_code, config.json), the vLLM source for `minimax_m3` (models/minimax_m3/*), the
  llama.cpp sources `src/models/minimax-m3.cpp`, `ggml/src/ggml-cpu/ops.cpp` (swiglu_oai), `gguf-py` constants.

## Assignment (make it, then make it fast)

Phase 1 - correctness, in this order, each step checked before the next:
1. Move `crates/minimaxm3` to `llm/minimaxm3` as the engine crate (`ARCH = "minimaxm3"`), keep `Geometry` and the
   NS_MM31_FILE test. The architecture tests then apply (cli/nextsycl/tests/architecture.rs: prompt-cache checkpoints
   `cache_fingerprint` / `save` / `restore` / `read_checkpoint`, an entry in main.rs's `engines()`, every env var in
   docs/environment.md) - read CONTRIBUTING.md and an existing engine (llm/deepseek41, llm/glm5next) first; follow them.
2. A CPU reference of one layer and then the whole model in Rust or Python written from the reference code (no
   llama.cpp build: it would be downloaded code), reading the GGUF with `nextsycl-gguf` and the repo's dequant
   (kernels/llm/deepseek41/silo/std.cpp is bit-exact against ggml for IQ1_M, IQ2_XXS, IQ3_XXS; Q4_K/Q5_K/Q6_K in
   kernels/llm/qwen35/silo/mmvq.cpp; IQ4_XS in nemotronh's nhs.cpp; Q8_0). Dump per-layer activations for a short prompt.
3. The engine on the GPUs (B70 + B65, 32 GB each; the 129 GB file streams like DeepSeek V4.1 IQ2_XXS: non-expert
   weights ~7 GB resident, experts in VRAM / pinned host memory / the file, per llm/deepseek4 and llm/deepseek41 -
   reuse their expert store, `spare` chain, prefetch and two-card split rather than writing a new one). Compare each
   layer's output with the reference (cosine, max abs), then greedy tokens identical to the reference for a prompt.
4. The sparse attention (indexer, block scores, top-k, gather, flash decode over the selected blocks) in the model's
   silo `kernels/llm/minimaxm3/silo/`, never in shared kernels (CONTRIBUTING.md; docs on silos).
5. Chat: the GGUF's chat template (Jinja; see how other engines render theirs), the prompt cache, the model card
   `model-cards/llm/minimax-m3.1-flash-iq1m.yml` (cards are data, not code; licence minimax-community), tool format.
Phase 2 - speed (only after phase 1 is exact): measure pp and tg (label them so; never a bare tok/s) at 2K / 8K /
32K contexts with `--ignore-eos --max 256` rungs like docs/ladders/glm-5.3-flash-uncensored_ladder_stats.md; tune the
expert placement and prefetch, the MoE matvec for IQ1_M / IQ2_XXS / IQ3_XXS / IQ4_XS, the decode attention, and the
two-card split; per-GPU parallel load (see llm/glm5next engine.rs `NS_LOAD_SEQ`); then an MTP/DSpark draft
(`dspark-MiniMax-M3.1-Flash-Q8_0.gguf`) if the base is solid. Record the table in docs/ladders/.

Rules for this work (the user's): commit to the worktree branch only, never push, no Claude attribution in commits or
comments; tune only inside the model's silo; exact before fast; test files and containers cleaned by exact name; never
kill an engine mid-run; keep >= 1.5 GB VRAM free on each card; hardac1 `sudo` needs a password (do not use it); no
llama.cpp server; the production registry and `/var/lib/nextsycl` are not yours to touch - build and test in your own
copy on hardac1 (`~/src/nextsycl-m3`, rsync with --checksum, touch changed files before cargo builds, runner scripts
under the shared lock `/tmp/ds2run.lock` like ~/src/glmrun.sh / mossrun.sh, mounting the model dir read-only), and use
a registry of your own (NS_REGISTRY in your copy's nextsycl.conf, never the production NS_STUDIO_MODES). The production
chat models and the audio server (Qwen3-TTS on the B70, 4 GB) may be loaded: check `system.busy` / the card's free
memory before loading anything, and say so if the cards are not free enough instead of unloading others' work.
