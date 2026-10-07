# TODO

Paused 2026-10-06. State: GLM-5.3-Flash (ds4 IQ2 file) on B70 + B65, served as the box's chat mode `glm53`, MTP on
by default (`--no-mtp` turns it off).

## Done

- [x] Server (`nextsycl serve`): OpenAI API, streaming, thinking split out, prompt-prefix reuse, graceful stop (7931d08)
- [x] Wired into the box: studio llm mode `glm53` + model-switch entry (LIGHT: no tools, no background tasks)
- [x] MTP speculative decoding (8faf2e9): draft block over the whole conversation, 2-row verify, KDA snapshot
      rollback, exact acceptance. Greedy 13.1 -> 15.7 tok/s, 87% of drafts accepted on a code prompt; verify rows
      bit-identical to one-token decode (`nextsycl spec-check`: 0 difference over 94 rows)

## Long context (done 2026-10-06)

- [x] The DSA lightning indexer (llama.cpp de25343 `build_kpool_select` / `set_input_kpool`): pools of 4, a pool
      visible once complete, top 512 pools per row plus the row's incomplete pool, dense below 512 pools
- [x] Pooled keys in the pass that completes them (a 4-token ring of ik | ig per layer); scores as GEMMs; a radix
      top-k with a scan compaction (deterministic); attention scores in local memory (<= 2,051 cells a row)
- [x] The MTP block through its own indexer
- [x] VRAM held for the sessions' attention caches before the expert store sizes itself
- [x] Parity: 3,180-token prompt against llama.cpp - layer 3's attention output 0.99999 with selection active, the
      same next token; deeper layers drift smoothly (0.98 by layer 23, no layer jumps); spec-check 0 difference
- [x] 12,202-token prompt runs (prompt 60 tok/s, decode 12.1 tok/s - flat with context); the chat mode serves 64K
- [ ] 128K: the latents are float32 (2 KB a token a layer; 6.5 GB for two 128K sessions) - fp16 latents first
- [ ] Prompt speed (2026-10-06): 396 tok/s at 3K, 425 at 12K (from 23 at the start of the long-context work; 4,096
      chunks; experts, dense matrices and MLA attention in fp16 on XMX; host experts copied on a copy queue beside
      the compute; the KDA scan a sub-group per column; hc mix a GEMM). Left at 12K: the experts' fp16 GEMMs and
      expansion (~11 s), MLA's per-row gather of its cells (2.1 s: score a row block against the union of its
      cells), KDA 3.7 s. Next: a fused dequant + XMX GEMM; MLA over row-block unions
- [ ] Decode past 2K: 12.1 vs 15.7 tok/s short - the indexer's per-token GEMM + the attention kernel

## Prompt speed: what Strata's prompt path teaches (its numbers on the B70, docs/INTEL.md there)

- [ ] **Bigger chunks.** Strata reads prompts in 4,096-token chunks; at 2,048 an 80K prompt took 278 s, at 4,096 77 s
      (its experts not in VRAM are streamed once per chunk). Here a 512-token chunk routes ~14 tokens to an expert;
      at 4,096 it would be ~114, so every expert's weights (from VRAM or over PCIe) serve 8x the tokens, and the
      expanded fp16 GEMM path (measured slower at 512: 22.6 tok/s) becomes a real GEMM. What stops it: the arena
      (~1.5 MB of temporaries a token: 6 GB at 4,096). Trim the per-token temporaries (fp16 activations, in-place
      KDA/MoE buffers), and borrow VRAM for the rest.
- [ ] **Borrow expert slots for the prompt** (Strata's "prompt path borrows N cache slots"): during a long prompt,
      lend part of the VRAM expert cache to the chunk's temporaries / a streaming ring, give it back for decode.
      Strata mirrors the lent slots in pinned host memory so the refill after the prompt is a RAM copy (its "lend
      mirror": prompt +15-40%, refill 720-2,240 -> 200-380 ms).
- [ ] **A stager: stream the next layer's experts while this one computes** (Strata's 96-slot streamed ring, a copy
      queue, a polled sequence flag - never host waits on queue events: they deadlocked under Level Zero v2). Today
      the in-place PCIe reads sit on the compute critical path.
- [ ] **Group by expert across the chunk, one launch a layer** - done (the grouped kernels); prompt lanes 32 a row.
- [ ] **QSA / indexer scores as GEMM tiles** (Strata's QSA block scores: a GEMM tile per block pair) - the indexer
      here already scores by GEMM; the MLA prompt attention (12 s of 47 at 3K) is the per-row kernel: tile it.
- [ ] **Dequant with vector stores** (Strata: 575 -> 841 tok/s at 2K / 8K) for any path that still expands weights.

## Decode (2026-10-06 night: 18.5-20 tok/s)

- [x] hc pre fused, async swaps on two copy lanes, 4 lanes a row, prefetch by the next layer's router (2 a layer),
      the prompt arena's VRAM lent to experts at decode, the strongest GPU last
- [x] Expert cache policy studied offline (reference/expert-cache): LRU is the best simple policy; the rest of the
      gap to Belady needs better prediction
- [ ] Swaps are still ~1/3 of decode time (both cards on Gen5 x8)
- [x] Better expert prediction: guesses by chance of use (rank hit rates; reference/expert-cache) - p >= 0.6,
      read at the router's wait: decode 18.4 -> 19.4 tok/s. Two layers ahead predicts worse (93% at rank 1 vs 96%)
- [ ] The rest of the gap to Belady: a guess from a later state (needs the hidden state before the layer's attention
      on the host - another wait), or the copies of a wrong guess cancelled
- [ ] Two requests at once would use both cards at the same time (each is ~55% busy at decode today)

## Decode, what is left (2026-10-07: 18-19 tok/s, MTP on)

Per token: routed experts 27.7 ms (the kernel ~0.3 ms a layer-pass of the ~1.2: the rest waits on PCIe copies of the
misses), KDA 10.9 (its products at ~613 GB/s), MLA 4.7, the draft block 3.2, the shared expert 2.1, hc 1.5.

- [ ] An expert's gate/up computed as soon as its gate/up bytes are in (2/3 of the copy), its down while the down
      copies (split moe_grouped into its two launches with a wait between) - a third of each miss's wait
- [ ] The card swap (below) - the misses' copies at x16

## To try

- [ ] Swap the cards' slots: the B70 to the x16 slot, the B65 to Gen4 x4 - then NS_SPLIT=15 (the B65's 12 MoE layers
      all resident, no swaps over its slow link; the B70's swaps on x16). Simulated (reference/expert-cache/splits.py)
      copy time a pass 5.9 ms vs 8.8 at today's best; with today's even split it would be 18.6 (worse). Measure with
      `nextsycl bench` after.
- [x] Several requests at once (branch `batching`): forward_batch + an engine thread with NS_PARALLEL sessions -
      bit-exact; 2 clients 18.1 -> 20.3 tok/s in all, the second's first token 10.4 -> 2.0 s
- [x] Batch: MLA's projections once for all rows (mla_batch) - +3% (24.1 -> 24.9 tok/s for 2): the batch is bound by
      its experts - two conversations want ~15 distinct experts a layer, not 8 (49 ms a 2-row pass of 90)
- [ ] Batch: a prompt read
      in chunks between decode steps (a long prompt now stalls the others' decode); the draft block in batches;
      per-request energy (concurrent requests share the counters - each counts both)

## LogProbChain (an API option)

A request option `logprob_chain` (beside `logprobs` / `top_logprobs`): logprobs corrected for what the model is
only echoing from earlier in its own turn. The thinking tokens skew the answer's logprobs: a token the thinking
already wrote comes out near-certain because the model copies it, not because it is sure of it.

- [x] v1 (2026-10-06): `logprob_chain: true`; the attention = the mean over heads and MLA layers of a one-token
      pass's softmax (Glm::capture_attention); chained = logprob + sum over equal turn tokens of attention x their
      own logprob; `chain` on each entry (with the top 3 echoes) and `chained_logprob` on each alternative
- [ ] The mean dilutes: an echoed token gets 1-3% of the attention while the turn gets 20-30% - try the max over
      heads, the heads that attend most to the turn, the last layers only
- [ ] For each output token X, one attention vector over the context: combine the heads of the attention layers -
      experiment: all MLA layers averaged, the most informative layers only (by entropy or by how peaked they are),
      the last layers, weighted by head; MLA's absorbed scores (q~ . c) give per-position weights cheaply; KDA layers
      have no attention matrix (only a state) - leave them out or approximate
- [ ] For X's candidates (its top logprobs), any candidate that equals a token at a position of this turn (thinking
      or answer generated so far) that X attends to: skew its logprob in proportion to THAT earlier token's own
      logprob times the attention weight on it (an echo of a confident earlier token is discounted; of an uncertain
      one, kept low) - the exact formula is part of the experiment
- [ ] Keep each generated token's own logprob (already computed for `logprobs`) so the chain can look it up
- [ ] Return the chained logprobs beside the raw ones (e.g. `chained_logprob` on each entry), and say which earlier
      positions dominated
- [ ] Validate: questions with known answers, with and without thinking; does the chained confidence of the final
      answer track correctness better than the raw one (calibration)

## Speed, later

- [x] Prompt chunks read host-slot experts in place over PCIe (no swaps); decode still swaps (faster there: 12.5 vs
      11.2 tok/s with NS_DECODE_DIRECT=1)
- [ ] Expert swaps at decode: 32 ms/token with MTP (2-row passes touch more experts). Strata's answer (its 2.6 -> 40.9 tok/s on
      the B70): the GPU reads missed experts straight from pinned host memory in the expert kernel (the grouped
      kernel already takes a pointer per expert - pass the host slot's address), a share of the misses (pcie_frac
      ~0.55) and LRU admission for the rest. Pinned memory belongs to the context that allocated it: each part reads
      only its own mirror (Strata #1054 hung two B70s on a cross-context fill)
- [x] Deeper drafts: NS_DRAFTS=2 chains the draft block on its own output, a 3-row verify - exact (the same greedy
      output), 2.36 tokens a pass vs 1.83 (the 2nd draft accepted 71%), but a 3-row pass costs 33% more (3 tokens'
      experts, the block run twice): 18.4 vs 19.1 tok/s. One draft stays the default
- [ ] Launch overhead: SYCL graph replay for the decode step
- [ ] Prefill speed

## Later

- [ ] qwen4exp (the Coder family) and a `--model` switch between runtimes (Strata does 75 tok/s on it today)
- [ ] IQ3 variants of GLM (NOTES.md)
