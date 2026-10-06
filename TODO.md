# TODO

Paused 2026-10-06. State: GLM-5.3-Flash (ds4 IQ2 file) on B70 + B65, served as the box's chat mode `glm53`, MTP on
by default (`--no-mtp` turns it off).

## Done

- [x] Server (`nextsycl serve`): OpenAI API, streaming, thinking split out, prompt-prefix reuse, graceful stop (7931d08)
- [x] Wired into the box: studio llm mode `glm53` + model-switch entry (LIGHT: no tools, no background tasks)
- [x] MTP speculative decoding (8faf2e9): draft block over the whole conversation, 2-row verify, KDA snapshot
      rollback, exact acceptance. Greedy 13.1 -> 15.7 tok/s, 87% of drafts accepted on a code prompt; verify rows
      bit-identical to one-token decode (`nextsycl spec-check`: 0 difference over 94 rows)

## Next: long context (in progress - reading the reference)

Today MLA attends to every earlier token, which is exact only up to ~2,048 tokens of context.

- [ ] Read the indexer's exact semantics from llama.cpp de25343 `src/models/glm5-next.cpp` (`build_kpool_select`,
      ~line 567; `set_input_kpool` in `llama-memory-hybrid-idx.cpp`): which pools count as complete for a query,
      the tail (`kpool_select_tail`), mask / tie handling, `indexer_index_share_mtp`, `indexer.types`
- [ ] Indexer cache per MLA layer: per-token ik / ig of the incomplete pool, pooled keys [ctx / 4, 128]
- [ ] Indexer scoring: iq = W_iqb . qr [32 x 128], w = W_proj . x / sqrt(128 * 32), score = sum_h relu(iq_h . pool) * w_h;
      top 512 pools per query (as a GEMM for prefill rows)
- [ ] Sparse MLA kernel: attend over the selected pools' tokens + the tail
- [ ] The MTP block's attention through its own indexer too (it has the weights)
- [ ] VRAM budget: reserve the latent caches (f32 [ctx, 512] per MLA layer, 11 + 1) and indexer caches before the
      expert store takes "free less 3 GiB"; keep >= 1.5 GB free (the VRAM spill hang). Consider f16 latents.
- [ ] Parity against llama.cpp past 2,048 tokens (a long-prompt dump), then `--ctx` 32K / 128K in serve and the mode
- [ ] Long-prompt prefill speed (256-token chunks; the indexer GEMM)

## Speed, later

- [ ] Expert swaps: 32 ms/token with MTP (2-row passes touch more experts). Ideas: compute misses straight from pinned
      host memory (no victim D2H), admit to VRAM only on repeat use, hot-expert placement from usage counts
- [ ] Deeper drafts (chain the MTP block; MAX_VERIFY > 2)
- [ ] Launch overhead: SYCL graph replay for the decode step
- [ ] Prefill speed

## Later

- [ ] qwen4exp (the Coder family) and a `--model` switch between runtimes (Strata does 75 tok/s on it today)
- [ ] IQ3 variants of GLM (NOTES.md)
