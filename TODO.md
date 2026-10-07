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

## Prompt: experts without the fp16 copy (started 2026-10-07)

The prompt path expands each expert to fp16 (13.8 s of a 36K prompt) and oneMKL multiplies it (17 s). At 128 tokens
oneMKL is bound by reading the 33.5 MB fp16 expert (92 us at 128 tokens, 94 at 256), so a kernel that reads the 2 MB
of IQ2_XXS instead could win twice.

- [x] joint_matrix (XMX) GEMMs on Xe2 measured (scratchpad, kept in kernels/ns/fused.cpp's notes): A and a col-major B
      straight from global memory reach oneMKL (63-82 TFLOPS); staging B through local memory (any layout) or a
      global scratch is 2-3x slower; a B fragment filled in registers works - lane n holds column n's 16 k in order
      (probed by loading a known matrix; the coordinate form of joint_matrix_apply faulted the GPU)
- [x] fused.cpp: each lane decodes its 16 IQ2_XXS weights into B (two 8-value groups), 32 tokens a sub-group -
      exact against expand + oneMKL; in a real 12K prompt 223 us a call vs 215 for expand + GEMM: no gain
- [x] Cheaper decode: 32 k a lane a step (the group's 8 bytes read once - grid indices, scale, signs - for two B
      tiles), 8 row tiles a sub-group above 64 tokens (large register file). Alone (4096 x 4096, 64 experts
      rotated so the weights come cold): B70 117 us vs 202 for expand + oneMKL at 64 tokens, 135 / 214 at 128; B65
      150 / 344, 224 / 358. Not helping: the grid table in local memory; each sub-group staging B through its own
      local-memory tile and a block load (164-196 us vs 144-146 for the register fill); -fp-model=precise costs
      nothing here
- [x] In the engine (12K, NS_PROFILE=gpu): 172 us a call for the experts of <= 64 tokens vs ~200 for the
      engine's own expand (109 us, faster than the bench's) + oneMKL; above that oneMKL's larger tiles win (with no
      limit gate/up took 9.5 s vs 7.9). NS_FUSED_MAX now defaults to 64: gate/up 7.3 s vs 7.9 of GPU time over
      both cards, the same text; end to end within the run-to-run noise (504-541 tok/s either way)
- [x] The down projection (Q2_K: 16 k share a scale byte and a shift of 16 quant bytes, no table) the same way:
      alone 39-40 us at 64-128 tokens (B70); in the engine the down took 2.8 s of GPU time at 12K vs 3.3 expanded
      with NS_FUSED_DOWN_MAX=128 (the default; 2.9 at 64, 3.25 at 256), the same text
- [ ] 2D block loads for A; the B fill (joint_matrix_apply, ~55 us of the 128-token call with plain fp16 weights) is
      the gap to oneMKL's ceiling

## Prompt: the two GPUs in a pipeline (2026-10-07)

- [x] A prompt of several chunks ran the GPUs in turn (each idle through the other's layers). `feed` now runs the
      first GPU's layers on a thread a chunk ahead (NS_PIPELINE=0: in turn): 12K 542 -> 721 tok/s, 36K 527 -> 744
      (B70 first), the same text; in the server's order (B65 first) 12K at 786. Splits (12K / decode): B70 first 22
      731 / 21.1, 26 802 / 20.4; B65 first 22 786 / 20.9, 20 740 / 21.7, 18 502. Smaller chunks lose (2048: 585,
      3072: 609 - the experts' weights serve fewer tokens)
- [x] Measured the stages (NS_PIPE_TRACE=1; NS_PROFILE=gpu NS_PROFILE_PART=i profiles one GPU): 36K, B65 first, a
      chunk 4.1 s on the B65 vs 3.45 on the B70 (with the draft block's 0.22). Splits do not balance it: a layer moves
      its experts too (21: 3.9 / 3.6, the same total; 20: the B70 4.7). The hand-off is ~17 ms a chunk
- [x] The KDA scan was latency bound by occupancy: a sub-group a value column, H x 128 long-running sub-groups - ~3
      waves on the B65 (twice the B70's scan time for the same layers). Now 4 (B70) or 8 (B65) columns of a head a
      sub-group (shared loads, overlapping reductions; NS_KDA_COLS): the B65's scan 1.46 -> 0.54 s at 12K, the B70's
      0.25; 36K 914 -> 989 tok/s, the same text (each column's arithmetic unchanged)
- [ ] The B65's stage is still the longer one: its MLA attention (4.2 s at 36K) and the experts' expand + GEMM
- [x] The server read 36K in 82 s against generate's 37: its two 64K sessions' VRAM pushed experts to host memory,
      whose share went by layer count - the B65 left part of its share unused while 268 of the B70's experts stayed
      on the file only (read from it each prompt pass). A GPU's unused share now goes to the next: 0 on the file,
      36K through the server in 38.4 s; benchy 2K 345 -> 435, 8K 408 -> 762, 40K 720 -> 1,011 tok/s
- [ ] The shares by need both ways (only a later GPU gets an earlier one's spare today)
- [ ] Bigger prompt chunks: 6144 read 36K at 1,072 tok/s vs 961 (generate; 2 x 6K = 3 x 4K at 12K, decode the
      same), but in the server (two 64K sessions reserved) the 40K prompt faulted the B70 (an engine reset, the host
      heap corrupted, exit 139). Suspect the driver's VRAM spill path (the box's known hang): the expert budget keeps
      a fixed 2 GiB besides the arenas, while a pass's own buffers (the stream ring and work buffers, t x ~64 KB each,
      ~1.5 GiB at 6144) grow with the chunk. Budget those per chunk, then retry - with VRAM headroom checked before,
      never by reproducing the fault. Cutting a prompt into equal chunks did not help at 4096 (40K: 975 vs 1,011)
- [ ] Fused gate/up: no gain from more column tiles a sub-group (NB 2: slower, NB 4: spills) or limits past 64 (128:
      the same 7.5 s, 192-256: 8.1)

## 256K context (plan: docs/256k-context.md, 2026-10-07)

- [x] 1. Per-slot context (`--ctx 262144,32768`: a session each; a request goes to the smallest free one it fits,
        else waits; a checkpoint restores into any session as long; its tokens end at its session's context). 256K
        served: 253,308 tokens read in 271 s (933 tok/s, 29.6 Wh), decode at that depth 14.5-15.6 tok/s, a short
        request in the 32K session at 21 tok/s; the attention reserve 3.8 GiB (B65 1.58, B70 2.21), ~320 experts
        fewer in VRAM, 0 on the file; the checkpoint 3.5 GiB; a follow-up on the document in 4.4 s (reused from it)
  - [x] 128K in generate: the prompt at 998 tok/s (126 s), the answer right, stages ~3.9 s a chunk to the end, 0
        experts on the file. Decode at that depth was 13.0 tok/s (21-22 short): per layer and pass the indexer's
        top-k 330 us a row (one histogram's counters took every add; the selection written with four collectives
        a tile) and MLA's attention 1,047 us (a work-group walking 2,051 cells a load at a time). Now 32 histogram
        copies and a one-pass selection (45 us, the same selection), and a decode-width attention kernel (a
        sub-group a cell, 16-byte loads, 4 cells a step: 444 us): 17.1 tok/s at 128K, exact (spec-check,
        batch-check, the same text)
  - [ ] Decode at depth, what is left a layer: the indexer's projections (~300 us), attention 444 us (each of the
        64 heads' work-groups reads the same 2 MB of cells)
- [x] 2. A needle check: `nextsycl bench --needle` - 9 of 9 found at 10 / 50 / 90% of 32K, 130K and 249K
        (docs/benchy/needle-2026-10-07-f16.md; 249K prompts read in ~267 s). Indices: positions are int32 cells,
        every size and product in the kernels int64 - no limit below 2^31 tokens
- [ ] 2b. Logits of one long prompt read in other chunk sizes (a chunking-independence check)
- [x] 3. NS_KV=q8: the latents as [16 fp16 scales][512 int8] a row (544 bytes vs 1,024), written by ns_to_q8row,
        read by the decode kernel (a lane's 32 values one block) and gathered to fp16 for the prompt's GEMMs. Exact
        within itself (spec-check, batch-check). Against llama.cpp at 3K: final logits cosine 0.99521 vs fp16's
        0.99640, the same top 5. 12K prompt 852 tok/s (= fp16); 128K: 1,019 tok/s, decode 17.2 at depth (fp16
        17.1), the attention reserve 0.95 GiB vs 1.63. The needle on q8: below
- [ ] 3b. The indexer's pooled keys in fp16 (128 -> 64 bytes a token and layer); a 4-bit latent form
- [ ] 4. The indexer fused (XMX scores, the heads' sum and the top-k in one kernel; no T x H x pools scores)
- [ ] 5. Long prompts' checkpoints on the NVMe (~3.5 GiB at 256K)
- [ ] 6. Serve it: NS_CTX per slot, the HTTP limit, the tokenizer on ~1 MB, the client's context setting

## Decode, what is left (2026-10-07: 18-19 tok/s, MTP on)

Per token: routed experts 27.7 ms (the kernel ~0.3 ms a layer-pass of the ~1.2: the rest waits on PCIe copies of the
misses), KDA 10.9 (its products at ~613 GB/s), MLA 4.7, the draft block 3.2, the shared expert 2.1, hc 1.5.

- [x] An arriving expert's gate/up launched once its gate | up bytes are in, its down once the rest is (two copies up,
      moe_grouped in two phases) - +1% (most misses are prefetched now; little demand wait is left)
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
- [x] Batch: a prompt read in groups between rounds - 4 chunks (16K tokens, ~17 s) while others decode or wait, 8
      alone (a request arriving meanwhile waits one group); the GPUs shared by time while others decode (after a
      group of t seconds they decode for t). A 256K prompt read at once held every other request for 4.5 minutes;
      one decode step a group gave a chat 0.23 tok/s. Measured: a chat sent 20 s after a 253K prompt answered in
      55 s (it waited one group, 19 s; then 2.9 tok/s), the long prompt 304 s vs 271 alone. Checkpoints at the stops
      as before
- [ ] The chat's decode beside a prompt read: 2.9 tok/s, not the ~10 of a half share - suspect the big arena moving
      between the expert store and the prompt at each switch (NS_LEND); take new requests within a group too
- [ ] Batch: the draft block in batches; per-request energy (concurrent requests share the counters - each counts
      both)

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
