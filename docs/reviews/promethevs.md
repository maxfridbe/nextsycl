# Review: PROMETHEVS against our deepseek41 engine (2026-11-02)

[Magnvs7/Promethevs](https://github.com/Magnvs7/Promethevs) at 9e06221, release v0.2: DeepSeek-V4.1-Flash-1017 (510 GB, the
user's "deepseek 2017") on a Ryzen 3950X, 64 GB, one RTX 3060 12 GB, two NVMe drives. Read only: the report, INSTALL.md, the
format notes and greps of the partial source (`v41_core.h`, `v41_serve2.cpp`). Nothing of it was built or run. Engine Apache-2.0,
report CC BY 4.0. Their numbers are theirs: one PC, 160-token passes, no comparison with DeepSeek's reference.

| | PROMETHEVS | nextsycl (B70 + B65, measured 2026-10-21) |
|---|---|---|
| tg | 4.4-5.8 cold, 7.4 warm on a short code prompt | IQ2_XXS 8.5 cold, 11.9 with a profile; Q3_K_XL 4.4 |
| pp | 1,163 tokens in 126 s (about 9) | IQ2_XXS 258 at chunk 8192 |
| Weights | 2-bit copy, exact FP4 on doubted tokens | one GGUF quant; never audited against FP4 |
| Sampling | greedy only | sampled |

We are faster on pp and on tg at the low quant. What they have and we lack is a measure of quality and a way to buy it back.

## What we do not have, most useful first

1. **An audit against the exact weights.** They count, token by token, how often the cheap copy picks another word than FP4
   (2-bit alone 97.40% over 960 tokens, corrected 98.33%). We have only the story test for gross failure. This is the cheapest
   item to copy and the rest depends on it. It needs an exact path: the 510 GB, or Q3_K_XL as a stand-in.
2. **Greedy draft acceptance.** They accept 94.6% of drafts with DeepSeek's own draft head; ours accepted 55% and lost
   (5.5 vs 10 tg). Our `step` samples each draft (`engine.rs` `smp.sample`), they are greedy only, and their quant of the head
   differs, so the two figures may not be comparable. Measure ours greedy before concluding the towers are not worth it.
3. **A low-bit draft head.** Their head in 2 bits took 2.85-2.98 s a pass against 3.8-4.0 s. Ours is Q8_0 and holds 6.7 GB of VRAM
   that experts could use, which is a large part of why MTP loses on us.
4. **Margin-gated correction.** Tokens whose top-two logit gap is under 0.45 are redone on exact experts; about 5% slower than
   no correction (7.39 vs 7.74, separate builds). Gains under one point of agreement on their audit. It is defined on greedy
   decoding and the chat server samples, so it needs a rule for sampled tokens first. Not worth building before item 1 says how
   far IQ2_XXS is from exact.
5. **Resident substitutes in speculative rows.** A row checking a guess may use a resident expert for a missing one (kept 3): without
   it their misses rose 55% and acceptance fell 98 to 91 (earlier weights). The exact row stays exact. We can only leave one out
   (`NS_DS41_COLD_SKIP`). Only matters if MTP comes back.

## What we already have

Prompt state saved to disk (`glue/serve/src/cache.rs`, their "saved state"), expert skip (`NS_DS41_COLD_SKIP` + `_RENORM`, default off;
theirs 0.12 on cheap lanes only), a prompt skip like their encoder-only read (`NS_DS41_COLD_SKIP_PROMPT`, default off, unmeasured),
O_DIRECT reads, a second-drive replica (`NS_DS41_REPLICA`), a usage profile for hot experts (`NS_EXPERT_PROFILE`), read-ahead.

Their 0.12 skip needs the exact path behind it; their own uniform raises cut the reasoning prompt from 4.49 to 2.65 (0.15). Treat a
global default on our side with the same care.

## Not worth trying: their negative results

Learned cache controllers (1.1-1.2% over a counting rule); generic compression of experts (0.905); about 1.6 bits a weight
re-quantised (1.7-2.1x their error at 2 bits, which fits our VQ repetition); drafting 8 instead of 5; reusing products between tokens
(0 of 61,824 blocks repeated); exact skips with certificates (0% certified). Their one combined pass losing to five lanes
(5.9 vs 7.7) is on a CPU-heavy 12 GB machine, so it says nothing about our batched verify.

## Downloads

Repo and v0.2 bundle (sha256 matches) in the session scratchpad, unrun. The bundle holds Windows programs and a partial source; its
own notes say it is not rebuildable (gaps G1, G2), and it has no SYCL or Linux path, so it is a reference, not something to run here.

Not downloaded: DeepSeek's 510 GB FP4 weights (revision 1bae0e1 in their INSTALL.md; 83 files on Hugging Face). Only item 1 with a
true exact path needs them, or a store build (about 1 TB more with the stores; the 4TB drive has 1.3 TB free).
