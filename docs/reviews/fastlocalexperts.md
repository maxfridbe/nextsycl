# Review: FastLocalExperts (FLE) against our DeepSeek engines (2026-11-11)

[sirfyyn/FastLocalExperts](https://github.com/sirfyyn/FastLocalExperts) at `e6909b7` (2026-11-06, FLE v1.2.6). Read only:
README, `docs/fle/ENV.md`, `EXPERT-SPLIT.md`, `STATUS.md`, the NOTICE files, the technical report PDF
(`docs/fle/fle-paper-v1.0.1.pdf`, pages 1-13) and greps of `src/llama-moecache.cpp`, `src/llama-expertstore.cpp`,
`src/models/deepseek41.cpp`. Nothing of it was built or run. Also read: ISTA-DASLab's tensor-allocation list for the
GSQ-RCO IQ2_XXS file FLE runs (a text file on Hugging Face).

## Summary

- **What it is.** A llama.cpp fork (MIT) for big MoE GGUFs on **one 32 GB NVIDIA GPU and 120 GB of RAM**. Hot experts are
  cached in VRAM, misses are **computed on the CPU** from pinned RAM while the GPU computes the hits, and the model's own
  MTP / DSpark head drafts. For DeepSeek-V4.1 it adds a 90 GB RAM expert store (O_DIRECT reads, sampled LFU) and a
  **routing bonus for resident experts**. Every FLE path is CUDA-only: `ggml/src/ggml-sycl` has no FLE code, so it would
  not run its features on our Arc cards.
- **Its DeepSeek numbers are not ours to expect.** It claims V4.1 Flash (ISTA GSQ-RCO IQ2_XXS) at tg 37.6 exact and
  50.6 with DSpark on a Ryzen 9950X3D + RTX PRO 4500 + 120 GB. With 120 GB, ~86% of that file's experts are held in VRAM or
  RAM: a 22 GB cache plus a 90 GB store over ~130 GB of experts, per their ENV.md. On our 61 GB box only ~58% of UD-IQ2_XXS's 146 GiB fits. Our exact decode waits 22-43 ms a token on expert reads
  from the NVMe, and FLE has no software answer to that except RAM. The DeepSeek claims come only as prose in the README
  and ENV.md: there are no brackets or raw data for them in the repo. The paper's paired brackets cover only Qwen3.8-Flash-Next.
- **What we already do.** Most of FLE is already in our engines: tiered experts, the profile warm start, hits and misses
  overlapped, CPU products with an (expert, 64-row) atomic-ticket pool (deepseek2), next-layer read-ahead, VRAM lent
  between prompt and decode, and a low-bit DSpark drafter.
- **Verdict.** Three ideas are worth our time. Only one is exact and cheap.
  1. **Residency-aware routing bias** (FLE's `FLE_MOE_CACHE_BIAS` / `FLE_MOE_HOT_BIAS`). This is the one large software
     lever. It would take ds41 tg from 12.7 exact to an estimated **13.8-16.3** (range below). It **changes output text**,
     so it needs the user's sign-off and our audit.
  2. **ISTA GSQ-RCO IQ2_XXS as a new ds41 card.** It is a different model file, so it **needs sign-off**. Its 128.75 GiB of
     backbone shards are the size of our UD-IQ1_M (130.2 GiB), which measured tg 16.2 against 12.7 exact. It needs an IQ1_M
     kernel in the silo.
  3. **deepseek2's CPU pool spinning and pinned to cores** (FLE / Strata's lane pool). Exact, ~60-100 lines in one file (the
     pool is 90 lines today), **+1-5%** tg.
  4. A cheap exact test: send ds41's speculative read-aheads past the page cache (O_DIRECT), a one-line switch;
     expected −3% to +8%.

  The rest is already done, does not fit the 61 GB box, or computes to under 3%.
- **The biggest exact lever is hardware.** Doubling RAM to ~128 GB would let the pinned tier hold ~100 GiB instead of 40.
  Exact ds41 tg would go from 10-13 to an estimated 17-23. That is FLE's real advantage. Arithmetic in the
  [cost model](#the-cost-model-of-our-expert-streaming).
- **Measured.** Nothing new. Production was busy (an ACE-Step job on the B65), and our ds41 runner needs the front door in
  maintenance (`ds41urun.sh`), which is a production impact. All our numbers below come from earlier runs on hardac1
  (`~/src/ds41/bt_*`, `~/src/ds2/sweep.log`, `~/src/ds4/cmp_v4_*`), cited by file.

## 1. What FLE is

| | |
|---|---|
| Form | llama.cpp fork (base `6a1a922d`, 2026-09-05), all features opt-in by `FLE_*` variables (~150 listed in ENV.md) |
| License | MIT (LICENSE). Notices: ik_llama.cpp (MIT, the IQ*_KT types), FreeToken (Apache-2.0, a CUDA pager), Strata (MIT, the expert pager order, AVX-512 VNNI IQ1/IQ2 dot products, the lane pool). Model files: Qwen Community License (Qwen), MIT (Rin, from ISTA-DASLab's quants) |
| Hardware | one machine: Ryzen 9 9950X3D (16 cores), 120 GB DDR5, one RTX PRO 4500 Blackwell 32 GB, Linux |
| Backends | CUDA only for every FLE path (async CPU subgraphs, fused hc kernels, the pager, top-k). No FLE code in ggml-sycl or Vulkan |
| Evidence | Qwen numbers: the report (PDF) with a paired-bracket method (reference / candidate / reference, ≤5% drift, bootstrap 95% CIs) and per-bracket tables. Its scripts (`m2_gepaart.py`, `end_kette.sh`, ...) and raw data are **not** in the repo. DeepSeek numbers: README and ENV.md prose only (dates, settings, a few medians of 3) |

### Techniques

1. **GPU expert cache.** A byte budget is planned across layers by hits per byte from a usage profile, with a per-layer
   floor. Admission is by aged request frequency (counts halved every 2,000 observations). Eviction takes the least frequent
   of the 32 oldest slots. A profile warm-starts the cache. Option: `--moe-expert-cache-inserts 0`, a static cache with no
   uploads while decoding.
2. **CPU computes the misses.** The CPU computes misses from pinned RAM (`--load-mode none`) while the GPU computes the hits
   (`FLE_MC_OVERLAP`). An async scheduler lets the GPU stream run ahead (`FLE_SCHED_ASYNC_CPU`). A fused CPU MoE op runs on a
   worker pool pinned to cores 2-15 that spins between layers, with jobs of (expert, 64 rows) taken by one atomic ticket
   (`FLE_MOE_POOL`). It uses AVX-512 VNNI kernels for IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS (from Strata).
3. **RAM expert store** (DeepSeek-V4.1, `src/llama-expertstore.cpp`). It is an arena with one slot per expert. Reads are
   O_DIRECT. Victims are the least used of 64 sampled slots, with counts halved every 8,192 layer calls. Missing experts are
   computed last, each when its read lands (Strata's order). Option `FLE_MOE_HOT_MLOCK_GB` instead mlocks the hottest
   experts and leaves the rest to the page cache.
4. **Routing bonus for resident experts.** `FLE_MOE_CACHE_BIAS` per layer range, and `FLE_MOE_HOT_BIAS` for experts held
   in RAM rather than on the SSD. In `deepseek41.cpp` the bonus is added to the **selection** bias (`exp_probs_b`). The
   weights stay unbiased. It is lossy: it changes which experts are chosen.
5. **Speculative decoding.** For Qwen it uses the model's MTP head (experts requantized to IQ4_NL, a 32K draft vocabulary).
   For V4.1 it uses the DSpark drafter (`FLE-DSpark-V4.1-1017-Q2K-Q8.gguf`), with `FLE_DRAFT_UBATCH` capping the
   drafter's compute buffer.
6. **Prefill.**
   - Ubatch lending: big prefill batches borrow expert-cache layers.
   - A 3-slot staging ring copies the next weight split while one computes.
   - Prefill windows: prompts of 2-2,047 tokens go through the decode path 8 tokens at a time (`FLE_PREFILL_WINDOW`).
7. **Model files.**
   - Per-layer hybrids spliced from Unsloth and ISTA quants, layers chosen by measured sensitivity (FLE-IQ3_M).
   - REAP pruning.
   - "Rin": a V4.1 file split into a hot half fixed on the GPU and a cold half in RAM (`EXPERT-SPLIT.md`, lossy, KL 0.29
     against ISTA's 0.25).
8. **Measured and dropped** (report section 6).
   - Pre-gating prefetch: −10% to −27%.
   - Copying a share of misses over PCIe (Strata's way): −4.5%.
   - N-gram drafting.
   - A rebase onto newer upstream: −5.7%.
   - Expert deferral: lossy and slower.
   - Engram preview / cold: neutral.

### What it claims

| Model, file | Claim (FLE's "tokens/s" for decode is tg) | Basis in the repo |
|---|---|---|
| Qwen3.8-Flash-Next, FLE-IQ3_M 88.87 GB | ~100 tg (bracket B1e, 24 requests), 2.49x upstream llama.cpp, +58.5% vs their earlier setup, decode 91.2 vs Strata 0.1.39's 74.0 tg (unpaired medians); prefill **slower** than Strata (2.3-2.7x) | report with CIs; no raw data |
| DeepSeek-V4.1-Flash, ISTA GSQ-RCO IQ2_XXS (232 GiB) | exact 26.8 tg (mlock recipe, 2026-11-02), 37.7 with a static profile cache and `inserts 0`, 29.7 with bias 0.1 on layers 15-42 (KL +0.003 against the quant's ~0.17), 35-36 with bias on all layers (KL 0.021); **50.6 tg with DSpark** (v1.2.5, three starts 49.8 / 50.8 / 50.6); prefill 74-83 tok/s pp (ISTA recipe) | README / ENV.md prose; no brackets |
| Rin (split 2-bit V4.1) | 53.4 tg, KL 0.29 vs 0.25 for ISTA IQ2_XXS | README; model card not read |

## 2. Our engines today (the numbers this review works from)

| Engine | File | tg | Where a token's routed experts come from (measured) |
|---|---|---|---|
| deepseek41 | Unsloth UD-IQ2_XXS, 146 GiB experts (15,360 x 9.73 MiB avg, 7.3-11.5 by layer), shards on the P41 | exact 12.7, `-fast` 21.4 (512 tokens, medians of 3); sustained `-fast` 25.3; pp 46 (580 tokens) to 470 (25K) | `bt_nomtp_code3.log` (2026-11-03, 256 tokens, no drafter, 9.92 tg): of 240 a token, **215.9 VRAM (89.9%)**, **11.9 pinned host (4.9%)**, **14.1 file (5.9%)**; 23.7 file reads a token with the read-ahead; miss waits 11.0 s = **43 ms a token** of 100.8. Port doc profile (13.4 tg): GPU chain ~44 ms, ~6 file experts x 3.7 ms = **22 ms** |
| deepseek2 | Unsloth UD-IQ1_S R1, 8.6 MB experts, on the 4 TB P310 | 7.06 (sweep.log) → 8.06 (round 2) | `sweep.log` (256 tokens): of 464 a token, 339.6 VRAM (73%), **96.5 computed on the CPU (21%)**, 15.9 up from pinned, **25.4 from the file (5.5%)** |
| deepseek4 | UD-IQ2_M V4-Flash, 85 GB, no file tier | 24.2 without MTP, 28.8 with (cmp_v4_dec*.log); 25.6 in the port notes | `cmp_v4_dec0.log`: of 258 a token, 251.9 VRAM (97.6%), ~9.4 swapped from host |

## 3. FLE's techniques mapped onto ours

| FLE | Ours | Status |
|---|---|---|
| Tiers VRAM / pinned RAM / NVMe | ds41 `Store` (exclusive VRAM + pinned cache, file below; `llm/deepseek41/src/engine.rs` `Store`, `Part::ensure_in`, `swap_in`); ds2 `llm/deepseek2/src/store.rs` | have |
| Profile warm start (`FLE_MOE_PROFILE`) | `NS_EXPERT_PROFILE` (`engine.rs` `expert_profile`, `read_profile`); XQ stores sized by the profile's ranking | have |
| Byte budget across layers by hits per byte | ds41: one LRU over each store's slots across its layers; stores by expert size. VRAM placement is uniform per layer at load ("expert-major placement") | mostly have. A per-layer share by profile is untried for the Unsloth form, worth ≤ a few % (FLE: +1.49% for budget + profile) |
| Hits on the GPU while the CPU computes misses | ds2 `Ds::moe_hybrid` (resident GPU products queued first, pinned on the CPU, file experts on the CPU as reads land); ds41 queues resident products before the host reads the routing (`NS_DS41_GPU_ROUTE`, `ns_ds41_route1` in `kernels/llm/deepseek41/silo/k41.cpp`) | have |
| CPU expert products, AVX-512 VNNI | ds2 `cpu.rs` (`dot_avx512_v2`, IQ1_S); not in ds41 or ds4 (they copy pinned experts up over PCIe) | ds2 only; for ds41 see "not worth it" |
| Worker pool, (expert, 64 rows) jobs, one atomic ticket, two sections a layer | ds2 `cpu::Pool::work`, `cpu::experts` (`ROWS = 64`): the same scheme | have. **Differences:** ours sleeps on a Condvar between runs (FLE spins, sleeping after 20 ms idle) and is not pinned to physical cores → proposal 3 |
| Async CPU subgraphs (GPU stream runs ahead) | ds41 GPU routing queues products before the host reads routing; ds2 host-synchronous a layer | partly. A SYCL `host_task` analogue is not worth it before graphs (TODO) |
| Next-layer prefetch from the router (`FLE_MOE_PREFETCH`) | ds41 `read_ahead` / `NS_DS41_FILE_AHEAD(_P)`; ds2 `NS_DS2_PREFETCH` (measured slower, off) | have. FLE's report also found pre-gating prefetch a loss on Qwen |
| O_DIRECT RAM store, sampled LFU with aging | ds41 `NS_DS41_ODIRECT` (lost: 13.6 → 12.7 tg, page-cache hits gone); `NS_DS41_HOST_LRU` (lost: 12.2 vs 12.9); LFU tried in ds4 (−23% swaps on free cards, +16% beside a 256K reserve) and in ds2 sims (LRU best) | tried; FLE's exact policy (LFU of 64 sampled, halving) untried. Not worth it on 61 GB (below) |
| `--moe-expert-cache-inserts 0` (static cache) | our misses must reach the GPU; ds41's swaps are already off the critical path (0.05 s of copy waits per 256 tokens = 0.2 ms a token) | not applicable |
| **Routing bonus for resident experts** | nothing: `NS_DS41_COLD_SKIP` drops cold experts instead; `NS_DS41_SUBST` substitutes only in draft rows | **new → proposal 1** |
| DSpark drafter, low-bit | `NS_DS41_MTP`, `NS_DS41_DRAFTER`, `llm requant-draft` (Q2_K); loses on prose (verify rows double the file misses) | have; revisit after 1/2 |
| Prefill: lend expert VRAM to big batches, staging ring | inverse lend (`NS_DS41_LEND_RESERVE`, `NS_DS2_LEND`), VRAM staging ring (deepseek41.md item 2), chunks of 16,384, next-layer prompt prefetch | have; our pp (258-470) is far above FLE's DeepSeek pp (74-83) |
| Prefill windows through the decode path | our prompt path already reads only routed experts; FLE's window path (45 tok/s pp) fixed upstream's habit of streaming every expert over PCIe | not applicable (our 580-token pp is 46 exact / 92 fast) |
| Fused hyper-connection kernels | `NS_DS41_KV_FUSE` (`ns_ds41_hc_pre_in`, ...) | have |
| CUDA graphs | SYCL graphs for decode: open (TODO "Launch overhead") | open, not FLE-specific |
| Hybrid files spliced by layer sensitivity; ISTA GSQ-RCO quants | one Unsloth file a card | **new → proposal 2** (ISTA file); hybrids later |
| Split hot/cold GGUF (Rin) | dynamic tiers do this at run time | not worth it (lossier than ISTA, needs new tensor layout) |

## The cost model of our expert streaming

Bandwidths used (measured on hardac1, from our notes):
- PCIe Gen5 x8 per card: ~28 GB/s for both directions together.
- NVMe: P41 6.3-6.6 GB/s sequential with O_DIRECT; P310 5.1 sequential, 2.7-4.0 for 16 threads of 4 MiB random reads
  (by data age).
- CPU products: ~60 GB/s on 16 threads (AVX-512 VNNI, ds2).
- One cold ds41 expert: 3.7 ms buffered, 2.5 ms O_DIRECT, read alone.

**deepseek41, UD-IQ2_XXS, exact** (bt_nomtp_code3 / port-doc profile):

| Traffic a token | Count x size | Bytes | Time it would take | What it costs now |
|---|---|---|---|---|
| VRAM experts | 215.9 x 10.2 MB | 2.2 GB from VRAM | part of the GPU chain | GPU chain 41-44 ms (device time), all of the rest outside file waits ~58 ms in the bt run |
| pinned → VRAM swaps | 11.9 x 10.2 MB | 121 MB over PCIe | 121 MB / 28 GB/s / 2 cards = 2.2 ms a card | **0.2 ms** on the critical path (copies wait 0.05 s per 256 tokens): hidden |
| file → VRAM | 14.1 needed (23.7 read with read-ahead) x 10.2 MB | 144 MB needed, 242 MB read | 144 MB / 5-6.5 GB/s = 22-29 ms if bandwidth-bound | **43 ms** of waits (bt run), **22 ms** (port-doc profile, ~6 file experts) |

tg = 1000 / (GPU chain and host + file waits): 1000 / (58 + 43) = 9.9 (bt run), 1000 / (57 + 22) = 12.7 (port doc).

What FLE's design would change on this box (each is a time a token):
- **CPU computes the pinned misses instead of swapping them.** 121 MB / 60 GB/s = 2.0 ms of CPU time, in place of 0.2 ms of
  visible copy waits. It saves at most 0.2 ms of 79-101 ms, **under 0.3%**.
- **CPU computes the file misses after their read.** This saves the copy up: 10.2 MB / ~14 GB/s per direction ≈ 0.7 ms an
  expert, but those copies already overlap other reads. The visible part is at most 1-3 ms → **1-3%**. Excluded: building
  AVX-512 kernels for nine ggml expert types.
- **FLE's memory (120 GB).** Pinned tier 40 → ~100 GiB, which assumes ~10 GiB for the OS and process and the 8 GiB
  MemAvailable floor. Resident = ~21.6 VRAM + ~11 lent + 100 = ~133 of 146 GiB (91%), so file picks fall to a few a token or
  fewer. tg exact ≈ 1000 / (44-58 + 0-5) = **17-23**, from 10-13. Excluded: whether the board takes 128 GB, and DDR5 speed
  with four DIMMs. Not checked.
- **Routing bonus.** File picks fall by a fraction α → see proposal 1.

**deepseek2, R1 UD-IQ1_S** (sweep.log, 142 ms a token then, 124 ms now):
- File: 25.4 x 8.6 MB = 218 MB a token → 44-55 ms at 4-5 GB/s if bandwidth-bound. The wait split for ds2 is not in the log.
- CPU: 96.5 x 8.6 MB = 830 MB → 14 ms at 60 GB/s, overlapped with the GPU's 340 hits.
- PCIe: 15.9 x 8.6 MB = 137 MB → 2.4 ms a card at 28 GB/s / 2.

The file is the largest single cost here too.

**deepseek4, V4-Flash UD-IQ2_M**:
- No file tier. ~9.4 host swaps x ~9 MB = 85 MB → 1.5 ms a card of link time, mostly hidden.
- 24.2 tg = 41 ms a token of GPU chain.
- Nothing in FLE moves this engine materially. The analogous ceiling measured on glm5next (every miss free) was +13%.

## 4. Proposals, ranked

### Try now (exact)

**3. deepseek2's CPU pool spins and is pinned** (FLE `FLE_MOE_POOL`, Strata's lane pool). Ranked below 1 and 2 by size of gain; first to build, being exact

- **Change.** In `cpu::Pool`, workers spin on the generation counter between runs and fall back to the Condvar after ~20 ms
  idle (FLE's rule). Pin workers to distinct physical cores, leaving core 0-1 for the host thread that feeds the GPUs.
  `NS_DS2_CPU_PIN=0` turns it off.
- **Files.** `llm/deepseek2/src/cpu.rs` (Pool), `docs/environment.md` row, a unit test that the pool's results are
  identical with and without spinning. The MiniMax CPU products (approved 2026-10-30) could take the same pool later.
- **Gain.** A token runs ≤ 2 pool runs x 58 MoE layers = 116 wake-ups of 15 sleeping workers. If a Condvar wake to all
  workers running costs 10-50 µs (typical Linux futex wake; **not measured** on hardac1), that is 1.2-5.8 ms of 124 ms
  → **+1% to +5%** tg (8.06 → 8.1-8.5). FLE's pool measured 29.5 → 31 tg (+5%), but together with its fused op, so not
  isolated. Excluded: SMT placement effects, either way.
- **Exactness.** Exact: the same parts write the same rows in the same arithmetic. Check that the 256 generated tokens are
  byte-identical.
- **Cost.** ~60-100 lines in one file. Basis: the pool is 90 lines today.
- **Test.** ds2 `generate`, 256 tokens on the sweep prompt, A/B/A, medians of 3, with `NS_PROFILE=1` "routed experts
  (CPU)" times. Gate: keep only if the median gain is outside the ±5% run-to-run spread, else close as measured-no-gain.

**4. ds41: speculative read-aheads past the page cache, demand reads through it** (from FLE's store reading "past the page
cache"; our own variant)

- **Change.** Read-aheads that are only guesses use the O_DIRECT path. In the bt run 2,038 of 4,226 started were not taken,
  which is 8 a token x 10.2 MB = 81 MB a token of page-cache churn. A layer's own demand reads stay buffered, so repeats
  still hit. `read_ahead_in` sends `RJob { direct: false }` to the reader pool, and the field already exists, so this is a
  switch on one line plus its environment row.
- **Files.** `llm/deepseek41/src/engine.rs` (`read_ahead_in`, `FileReader` jobs), an environment.md row.
- **Gain.** Bounded on both sides by earlier measurements. All-O_DIRECT lost 7% (13.6 → 12.7 tg) because repeats no
  longer hit, and a guess read alone is 32% faster O_DIRECT (3.7 → 2.5 ms). Expected **−3% to +8%**: the waits on taken
  read-aheads (3.55 s of the 11 s) x 0.32 = 1.1 s of 25.8 s = +4.5% at best, plus whatever the 81 MB a token of junk was
  evicting. Excluded: the drive's mixed-mode behaviour.
- **Exactness.** Exact: the same bytes.
- **Cost.** ~10-20 lines with the switch and its row.
- **Test.** 512 tokens on the B-tree and novel prompts, exact and `-fast`, medians of 3, warm and cold page cache. Counters:
  "experts read from the file", "waited ... for read-aheads".

### Needs the user's sign-off (changes output text)

**1. Residency-aware routing bias** (FLE `FLE_MOE_CACHE_BIAS` + `FLE_MOE_HOT_BIAS`). The largest software lever found.

- **Change.** At decode, add β to the **selection** score of each expert that is not file-only, on layers ≥ L0. The
  weights stay the unbiased sqrt(softplus) ones (ds41) or sigmoid ones (ds2), as FLE does in `deepseek41.cpp:1114`.
  - GPU route: `ns_ds41_route1` already gets the per-GPU slot table (`tab`), so the kernel adds β_v where
    `tab[e] != dummy`, plus a per-layer β_h mask for the pinned tier. The mask is uploaded at load and updated on the rare
    host-tier changes; ds41's host tier is exclusive, so it changes with swaps.
  - Host route: the same from `Store::loc`.
  - Switches `NS_DS41_RES_BIAS=β`, `NS_DS41_RES_BIAS_LAYERS`. Same for ds2 in `Ds::route`.
- **Files.** `llm/deepseek41/src/engine.rs` (route, `router_bias`, `Store` loc updates), silo
  `kernels/llm/deepseek41/silo/k41.cpp` + `ds41.h` (`ns_ds41_route1`: one extra pointer and β), `llm/deepseek2/src/engine.rs`
  (`Ds::route`), environment.md rows, a new `-bias` card (cards, not code).
- **Gain (ds41).** tg = 1000 / (T − α·W), where α is the share of file picks the bias steers to resident experts.
  - The bias would move 30-80% of the file picks. This is **an assumption, to be measured**: FLE's +11% (bias on layers 15-42)
    to +33% (all layers) was on a box whose misses go mostly to the CPU path, not the file.
  - Port-doc basis, T = 79 ms, W = 22: **13.8-16.3 tg** exact (+9% to +28%).
  - bt basis, T = 100.8, W = 43: **11.4-15.1 tg** (+15% to +52%).
  - With `-fast`'s q4_K dense on top, the bias could replace or soften the cold skip at similar speed.
  - Excluded: the cost of the bias itself (~384 adds a layer), and extra host swaps if β_v > β_h shifts picks between tiers.
- **Gain (ds2).** W is not in the log. If it is 30-55 ms of 124, then α = 0.3-0.8 gives 8.7-12.5 tg (from 8.06).
- **Exactness.** **Changes output** in every biased layer.
  - FLE measured KL +0.003 (β = 0.1, layers 15-42) against its quant's ~0.17, and 0.021 for all layers.
  - Our `-fast` cold skip costs +1.8% perplexity and ~2 points of audit agreement.
  - Output also depends on cache state, so the same prompt can differ between runs. The cold skip already has this property.
- **Cost.** ~150-250 lines in two engines and one kernel. Basis: the cold skip (2d72756) was 565 lines with its counters
  and docs.
- **Test.**
  - Quality: `llm audit` against the Q8_K_XL reference record (agreement %, as in the Promethevs review);
    `generate --score` perplexity on the two prose texts (exact 3.44 / 4.02); the 12,000-token novel (unique-word share,
    5-grams).
  - Speed: 512-token tg, medians of 3.
  - Sweep: β ∈ {0.02, 0.05, 0.1}, L0 ∈ {0, 15}.
  - Report file picks a token from "swapped in (... from host)" and "read from the file".

**2. ISTA-DASLab GSQ-RCO IQ2_XXS as a ds41 card** (the file FLE runs)

- **What it is.** `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, 5 shards of 128.75 GiB + a separate 103 GiB IQ4_NL
  Engram GGUF.
  - Experts per its allocation list: 60 tensors IQ1_M, 42 IQ2_XXS, 18 IQ2_XS (2.0 bpw backbone target).
  - Dense: Q5_K / Q8_0 / BF16.
  - Also on the hub: IQ1_M (112 GiB), IQ2_XS (145 GiB) and a "Coder" calibration.
- **Change.**
  - IQ1_M in the silo's std kernels: a group decoder in `std.cpp` (it shares IQ1_S's grid) and the split-Q8 decode product in
    `mmvq.cpp`.
  - A card mapping the separate Engram file as the "last shard".
  - Download to the 4 TB P310 (1.3 TB free, the disk rule for streaming models).
- **Gain.** Its backbone (128.75 GiB) is the size of UD-IQ1_M's (130.2 GiB). On our engine UD-IQ1_M measured **tg 16.2 vs
  12.7 exact** (512 tokens, medians of 3, +28%) and sustained `-fast` 26.4 vs 25.3. So expect ~16 tg exact,
  ±10% for the different type mix (IQ1_M's decode cost on the B65 is unmeasured). Compared on the same drive: the served
  UD-IQ2_XXS shards are on the P41 (6.3-6.6 GB/s), the P310 is 5.1, so an A/B must put both on one drive.
- **Exactness.** A **different model file**: quality unknown on our meter.
  - FLE: KL 0.25 to the original, on their meter.
  - Ours: UD-IQ1_M was +9% perplexity and repeats from ~5,000 words; UD-IQ2_XXS holds to ~8,000.
  - GSQ-RCO at the same size may do better, or not. It must pass the audit and the novel before any card is served.
- **Cost.**
  - Kernel ~50-150 lines. Basis: IQ4_NL was 18 lines in `std.cpp` (841f2ff); IQ1_M also needs the mmvq path.
  - Bit-exact check against ggml's `dequantize_row_iq1_m` with the existing harness (`NS_DS41_STD_DIR`).
  - Card ~40 lines.
  - Download: 248 GB / link rate. At 100 MB/s ≈ 41 min, at 50 MB/s ≈ 83 min. The link rate is not measured.
- **Test.** `llm kernels` bit-exact; `llm audit` agreement against Q8_K_XL beside UD-IQ2_XXS's 91.2%; the novel; perplexity;
  tg 512 tokens and a ladder rung (2,561 / 9,901 pp).

**5. Revisit DSpark after 1 or 2** (FLE: 37.6 → 50.6 tg, +35%)

Our drafter lost because a verify pass doubles file misses (prose 15.7 vs 20.3 tg). FLE's gain comes on a box where
misses come from RAM. Once file picks fall (proposal 1 or 2, or more RAM), the same settings deserve one re-run:
`NS_DS41_MTP=1`, the Q2_K drafter, and `NS_DS41_SUBST`, which is already lossy in draft rows. Cost: runs only, 4 arms x 3
runs x ~2 min ≈ 24 min of GPU time in a maintenance window.

### Not worth it (with the calculation)

- **CPU products for ds41 or ds4.** Pinned-tier swaps are hidden: 0.2 ms a token visible, so <0.3%. File-miss copies are
  1-3% at most. The cost is AVX-512 kernels for nine-plus ggml types (MXFP4, Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S). FLE's CPU path pays because its 120 GB holds the misses in RAM. Our misses are on the drive.
- **An O_DIRECT LFU RAM store as FLE builds it.** The arena is not reclaimable. With 40 GiB pinned and the 8 GiB
  MemAvailable rule, it can be at most ~4-6 GiB (~400-600 experts, +3-4% of 15,360), and it replaces a page cache of about
  that size. Our all-O_DIRECT test already lost 7%. Revisit with more RAM.
- **A static VRAM cache (`inserts 0`), or a share of misses over PCIe.** Our swaps are off the critical path. FLE itself
  measured the PCIe share at −4.5%.
- **Per-expert pipelining of a layer's file misses** (copy up each as it lands). With 14.1 file misses a token over 40
  layers (Poisson λ = 0.35), P(≥2 in a layer) = 1 − e^−0.35·1.35 = 4.9%, about 2 layers a token. Saving one ~0.5 ms copy
  each gives ~1 ms of 79-101, **<1.3%**.
- **Prefill windows.** Not applicable (section 3). Our 580-token pp of 46 tok/s exact equals FLE's window path's 45, and
  longer prompts are 3-10x faster than FLE's.
- **Rin-style split files, expert deferral, Engram preview / cold.** Split files are lossier than ISTA (KL 0.29 vs 0.25).
  FLE measured deferral slower and lossy, and the Engram tricks neutral. Our Engram wait is 0.11 ms a token.
- **Running FLE itself on Arc.** Its features are CUDA-only.

## 5. What could not be verified

- No FLE number was reproduced. Nothing was built or run. The DeepSeek figures have no raw data in the repo, and the paper's
  scripts are not in the repo either.
- No micro-measurement was run on hardac1. The front door was busy (ACE-Step on the B65), and the ds41 runner needs
  maintenance mode, which is a production impact.
- The wake-up cost of deepseek2's Condvar pool is not measured on hardac1; proposal 3's range assumes 10-50 µs.
- deepseek2's split of file waits is not in its log. Proposal 1's ds2 range assumes 30-55 ms.
- α, the share of file picks a bias moves, is an assumption (30-80%). FLE's figures come from a different miss mix.
- ISTA's file was not downloaded. That our loader accepts its separate Engram GGUF and its metadata is unchecked.
- ISTA's quality on our audit is unknown. Its expert share of the 128.75 GiB is estimated from the shard total less a
  dense part of the size Unsloth's has.
- "120 GB RAM → 17-23 tg exact" assumes file picks fall to near zero and ~10 GiB of OS and process memory. Whether the
  ProArt B850 board takes 128 GB, and at what DDR5 speed, is not checked.
- The DSpark acceptance behind FLE's 50.6 is not stated, apart from an illustrative status JSON.
