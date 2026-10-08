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

- [x] joint_matrix (XMX) GEMMs on Xe2 measured (scratchpad, kept in kernels/engines/glm5next/fused.cpp's notes): A and a col-major B
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
- [x] The B fill, other ways (plain fp16 weights, B70, 128 / 256 tokens: oneMKL 113 / 101 us, a direct block load of
      B 125 / 123, the apply fill 170 / 148): get_wi_data element writes 159 / 149 - the same; apply without the
      fill first is removed by the compiler (3 us: no work). With joint_matrix a register-filled B costs ~30-45 us
      over a block load whatever the form, and B staged through local memory was 2-3x slower: fused decoding cannot
      beat expand + oneMKL for the large experts this way
- [x] Fused decoding in ESIMD (reference/esimd-fused.cpp: xmx::dpas, B decoded with vector selects straight into its
      VNNI layout, A by 2D block loads, the weight words a step ahead): exact, but slower - B70, 64 / 128 / 256
      tokens: best 181 / 278 / 402 us vs expand + oneMKL 318 / 244 / 242 and joint_matrix's fused 110 / 135 / 200.
      Few row tiles a thread: the decode dominates (each thread decodes its B tile for one dpas - RM=1 scales with
      the tokens: 1,372 us at 256); many: too few threads to hide the chain (gathered words -> gathered grid ->
      decode -> dpas); the weights' 16 rows 1 KB apart are a scattered gather a step.
- [x] The three: the grid in local memory, one 8-byte gather for a row's group (no gain over two 4-byte), and a
      work-group sharing the decode - its threads split a 256-k block's 8 groups, write the B tiles into local
      memory (two buffers, a barrier a block), each loads every tile for its own rows. One work-group over all of
      an expert's rows (16 a thread) decodes every weight once: B70 103 / 117 / 157 us at 64 / 128 / 256 tokens
      (joint_matrix 117 / 135 / 200), B65 165 / 213 / 327 (206 / 224 / -); past 256 (B70) or 128 (B65) expand +
      oneMKL wins (512: 335 vs 294 on the B70). In the engine (fused.cpp, ns_moe_fused_gu_esimd; NS_ESIMD_MAX):
      36K 1,052-1,063 -> 1,096-1,097 tok/s
- [x] The down projection (Q2_K) the same way (reference/esimd-down.cpp): exact, slower than the joint_matrix down
      already in the engine - B70 48 / 52 / 69 us at 64 / 128 / 256 tokens vs 39 / 40 / 55, B65 105 / 143 / 132
      vs 47 / 66 / 119. K is 2,048 (8 blocks a row) and Q2_K's decode needs no table: little to share, and the
      work-group's barriers and local-memory round trips cost more than they save. Not used
- [ ] The B65's ESIMD gate | up past 128 tokens (decode-bound: its fewer units)
- [x] The expansion itself: alone (reference/expand.cpp, 2048 x 4096, cold) IQ2_XXS took 67 us on a B70 and 100
      on a B65 a work-item 8 values of one block; two blocks a work-item (their grid loads overlap) 40 / 58 us,
      exact; bigger work-groups or 4 blocks no better; Q2_K (36-37 us) already at its bound. In the engine the gain
      is small: the B65's gate/up expansion at 36K 4.17 -> 3.92 s (its build of the old kernel was already faster
      than the bench's), 36K 989 -> 992 tok/s
- [x] The expansion beside the GEMMs (a side compute queue, expert k + 1 expanded while expert k multiplied, two fp16
      buffers): slower - 36K 919-927 vs 995 tok/s. The GEMMs at these widths are memory bound too (they read the
      33.5 MB fp16 expert): the expansion still waited 4.5 s and the gate/up GEMMs went 4.0 -> 4.5 s. Reverted. The
      fp16 copy itself is the cost: only not making it (the fused kernels, wider) or bigger chunks (fewer per
      token) remove it

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
      Measured 2026-10-07 (NS_PIPE_TRACE now prints each GPU's free VRAM after a chunk): at 36K in generate the
      free VRAM after a chunk is about the same at 4096 and 6144 (B65 1.85 / 1.85 GiB, B70 1.54 / 1.38) - the store
      gives the bigger arena its room - and 6144 read 1,060 vs 938 tok/s. 7168 failed with a clean allocation error
      (no silent spill), so the server's fault at 6144 may not be memory at all: its chunks were 5,680 tokens then
      (the equal split, since reverted). Next: a server run at 6144 with plain chunks, watching free VRAM - only
      with a margin of >= 1.5 GiB at every chunk end, stopped at once otherwise
- [x] Done: in the server at 6144 the B70 then ran out cleanly (a 100 MB allocation failed, no fault): the expert
      budget's fixed 2 GiB was the pass buffers at 4096. Now 0.6 GiB + ~352 KB a chunk token (2 GiB at 4096 as
      before, 2.7 at 6144: ~100 experts fewer a GPU); NS_VRAM_GUARD_GIB stops a read before a GPU runs short. 6144
      is the default: benchy 40K 1,011 -> 1,096 tok/s, 2K 435 / 428, 8K 762 / 736, decode 17.7-18.5; the last two
      chunks split evenly when the last is under half (8K: 711 -> 736); free VRAM >= 1.79 GiB at every chunk end
- [ ] Fused gate/up: no gain from more column tiles a sub-group (NB 2: slower, NB 4: spills) or limits past 64 (128:
      the same 7.5 s, 192-256: 8.1)

## Several models (docs/engines.md, 2026-10-08)

- [x] The split: ns-runtime (the contract: Engine, opaque Session / Decoder / Checkpoint, Sampler, the registry by
      architecture), engines/glm5next (GLM's model description + engine + tools, moved as they were), its kernels
      in kernels/engines/glm5next; the server, the CLI and the prompt cache drive only `dyn Engine`. The same
      numbers before and after: spec-check (2 / 3 / 5 rows, short and long) and batch-check exact, llama.cpp
      parity 0.995209, generate's text identical
- [x] kernels/strata refreshed to the user's intel-arc-0.1.40 (3f37281): GLM unchanged (spec-check, batch-check
      exact, parity 0.995209)
- [x] engines/qwen4exp (the files say `qwen4exp`): Qwen3.8-Flash-Next IQ2_XS on both cards, every weight and expert
      in VRAM (B70 layers 0-38, B65 39-47; 36.5 GiB, 4.7 s from the page cache). The glue
      (kernels/engines/qwen4exp/qwen.cpp) is Strata's verify window (verify.cpp at 0.1.40) per stage, eager, all
      experts planned on the device; the PLE rows hashed and read in Rust. spec-check 2 / 4 / 8 rows (short, 3K,
      12K) and batch-check exact (max diff 0). Decode 37 tok/s, prompt 142 tok/s (8-token windows)
- [x] The chat templates by engine (`EngineKind::chat`: glm_chat, qwen_chat); ns-tok's `qwen35` pre-tokenizer
- [ ] qwen4exp: the prompt path (Strata's prefill.cpp: GEMM chunks, the grouped experts), then parity with
      Strata's own output (its prefill + first window; a like-for-like run there needs the same path)
- [ ] qwen4exp: MTP (Strata's draft layer from the base checkpoint, mtp/rt), the window graph captured, the split
      by compute (the B65 has 9 layers), the Coder IQ1_M and Swift files
- [ ] The GLM-only kernels' declarations out of ns.h's shared part

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
- [x] 2b. Chunk sizes: the 12K prompt read in chunks of 2048 gives the same text as 4096; 3072 differs at one
        word ten tokens in (oneMKL tiles by the row count: the sums' order moves, a near tie flips)
- [x] 3. NS_KV=q8: the latents as [16 fp16 scales][512 int8] a row (544 bytes vs 1,024), written by ns_to_q8row,
        read by the decode kernel (a lane's 32 values one block) and gathered to fp16 for the prompt's GEMMs. Exact
        within itself (spec-check, batch-check). Against llama.cpp at 3K: final logits cosine 0.99521 vs fp16's
        0.99640, the same top 5. 12K prompt 852 tok/s (= fp16); 128K: 1,019 tok/s, decode 17.2 at depth (fp16
        17.1), the attention reserve 0.95 GiB vs 1.63. The needle on q8: below
- [ ] 3b. The indexer's pooled keys in fp16 (128 -> 64 bytes a token and layer); a 4-bit latent form
- [ ] 4. The indexer fused (XMX scores, the heads' sum and the top-k in one kernel; no T x H x pools scores)
- [x] 5. A disk tier for the prompt cache: checkpoints pushed out of memory written to NS_CACHE_DIR (32 GiB),
        mounted from there when they are the best prefix; emptied at start (a checkpoint fits only its process's
        sessions)
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
- [x] The chat's decode beside a prompt read was 2.9 tok/s: a short prompt read after a long group reset the
      decode share to its own second (now the round's reading in all). Now 17.9 tok/s beside a 253K read, the chat
      done 35 s after it was sent (19 of them waiting for the group in progress). The big arena also stays with the
      prompt while one is read (`hold_arena`: no lending back and forth at each switch)
- [x] A request arriving mid-group waited for the group: a prompt read alone now goes in one pipelined read that
      stops at a chunk's end once a request waits (`feed_until`; groups of 8 alone had cost 8%: 294 vs 273 s at
      253K); the shortest read goes first and a request that starts decoding gets a group's time before the next
      group. A chat sent 20 s into a 253K read: answered in 9.8 s (was 258 s that morning), 18.4 tok/s; the long
      prompt 280 s vs 273 alone; the window holds back long reads only (it had made a second short request wait 11 s for
      its first token)
- [x] The disk tier, measured: a 256K prompt's checkpoint pushed out by a 128K one, its follow-up mounted from the
      file - 4.5 s, the right answer
- [x] The disk tier outlives the server: a subdirectory per fingerprint (the model file and size, the cache's form,
      MTP), the tokens beside each file, a stop writing the checkpoints still in memory; unused for 24 h (or past
      32 GiB, least recently used first) removed. A 12K checkpoint from the 64K mode mounted by the 256K mode
- [x] Decode, from Strata's lessons (2026-10-07): the GPUs are busy ~all of a decode (summed device time = the wall
      time) - routing on the device or graph capture would save little; the misses' copies are the cost (~16
      swaps a token). The q8 latent cache by default (~100 more experts in VRAM in the 2 x 64K server: benchy the
      same or a little faster - docs/benchy/v1-2026-10-07-q8.md). Pinning experts by a usage profile LOST in
      simulation (reference/expert-cache/pinning.py, the other topic's profile: 25% pinned +12% misses, 90% +140%);
      filling VRAM by the profile at load won the first answer: decode's requests counted, saved at a stop
      (NS_EXPERT_PROFILE, ~/.cache/nextsycl/prompts/expert-profile.txt, older halved each save) - a new topic's first
      256 tokens 3,951 -> 3,561 swaps, 23.2-23.5 -> 24.0-24.6 tok/s
- [x] Decode: missed experts on the CPU (reference/cpu-expert: ggml's AVX2 IQ2_XXS / Q2_K dots in Rust, the token
      quantized to int8 per 256, cosine 0.99997 to float). A 9950X, 8 threads of a spinning pool, the expert cold
      from DRAM: 138 us an expert-token (a copy up ~270 us), 227 / 341 us for a 2 / 3-row verify pass. But the
      ceiling is small: with every miss free (NS_FREE_MISSES=1, a measurement - wrong output) decode without MTP
      goes 18.8-19.0 -> 21.3-21.6 tok/s, +13%: prefetch and the async swaps already hide most of the copies. A CPU
      share would win a part of that (CPU-computed experts never come up, 8 cores spinning through decode) - not
      built. The card swap has the same ceiling
- [x] Decode's Q8_0 products (the KDA projections): Strata's wide32 kernel already has the aligned loads (load16_a2).
      Measured alone (reference/mmvq-bw.cpp, 8192 x 4096, random weights, cold): B70 488 / 551 / 538 GB/s at 1 / 2 /
      3 rows - 80-90% of its ~608; B65 374 / 384 / 356 (~62%: its fewer units' decode). Lifting the B65 to the
      B70's share would save ~1-2% of decode: not pursued. (The ~330 GB/s seen in the profile was both cards and
      all six projections, the small ones included.) Lesson: a constant fill is compressed by the GPU's memory -
      bandwidth tests need random data; and ns_copy_to is queued - its host buffer must outlive it (a bench that
      freed it faulted the B70, engine reset 4, recovered)
- [x] Sampling at a temperature sorted the whole vocabulary (154,880) for its top 256 a token (~2.5 ms): now a linear
      selection, then the 256 sorted (the same order, the same tokens - the drafts accepted matched exactly). Decode
      at temperature 0.7 21.3-21.6 -> 22.7-23.0 tok/s, at 1.0 20.4-20.9 -> 21.6-22.2 (greedy 23.0-23.8)
- [x] Prompt-lookup drafts (NS_NGRAM=K, off by default): the up to K tokens that followed the last 3 where they
      occurred before, verified in one pass. Exact, and ~90% accepted on a copy-heavy answer (adding comments to a
      file: 902 of 979 at K=4) - but slower at every K (17.46 tok/s without, 16.85 / 17.13 / 16.87 at K = 2 / 3 / 4):
      a 5-row pass took ~280 ms against a 2-row one's ~110 (2.5x for 2.4x the tokens) - each row brings its own 8
      experts (swaps +11-15%), the pass's cost grows almost with its rows. The draft block's 2-row passes (93%
      accepted on that answer) are the better trade on an MoE
- [x] A real bug on the way: verify passes of 3+ rows drifted at long context (spec-check --rows 3 at 3K: 3.2e-2).
      The indexer's ring of the last 4 tokens' keys (slot pos % 4) - a 3-row pass's rejected rows overwrote slots
      of positions the next pool still needed. Now 8 slots, every row of a pass written: 2 / 3 / 5-row passes exact
      at short and long context. NS_DRAFTS=2 was affected too. Disk checkpoints of the old layout: fingerprint nsck2.
      spec-check: --rows R (2..5), --layers --at N --row k, tensors matched by name
- [x] Speculative sampling at a temperature (on; NS_SPEC_SAMPLING=0 for the argmax drafts): the draft sampled from
      the draft block's distribution (the request's temperature and top-p), kept with min(1, p/q), else a token
      from max(0, p - q) - what is committed is distributed exactly as plain sampling; greedy is unchanged (the
      same text). Three prompts x temperature 0.7 / 1.0 x 384 tokens (docs/benchy/spec-sampling-2026-10-08.txt):
      acceptance 61.6% -> 65.0%, 20.37 -> 20.59 tok/s on average; the free-running story most (54% -> 66% at 1.0),
      a short poem at 0.7 lost (noisy). A sharper draft distribution (NS_SPEC_DRAFT_TEMP 0.5 / 0.25: 65.8% / 64.1%)
      the same within the noise. The sampler is a trait now (ns-runtime's Sampler: sample, dist, uniform, record)
- [ ] Decode is spread out now (per token: experts' kernels, KDA 10.6 ms, MLA 4.9, the draft 2.5, the shared expert
      2.2): no single kernel holds a big share. Levers left: more VRAM (fewer misses: <= 13%), MTP acceptance
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
