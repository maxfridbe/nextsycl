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

## Measured on our side (2026-11-02): how close to the reference, at what cost

Reference: **Q8_K_XL (routed experts original)**, UD-Q8_K_XL: every routed expert MXFP4, DeepSeek's own format, dense BF16. Not called exact: whether BF16 holds the
original FP8 exactly is not checked. `nextsycl llm audit`: 5 prompts, 200 tokens each (1,000 positions), its greedy run recorded, each arm following it teacher-forced
through `step` on B70 + B65. Agreement = the arm's best token is the reference's, at each position given the reference's history. With 1,000 positions a point or less apart
is not a result. tg is the cost, in the same runs, over one pass; pp was 2.5-14 over 132 tokens and is not a comparison.

| Arm (`--opt`) | IQ2_XXS agreement | tg | IQ1_M agreement | tg |
|---|---|---|---|---|
| alone | 91.2% | 11.32 | 89.0% | 8.83 |
| `gate 0.45` (Q8_K_XL's routed experts) | 91.8% (46 redone) | 5.04 | 90.1% (42) | 4.64 |
| `gate 0.9` | 92.7% (92 redone) | 3.12 | 91.3% (92) | 2.99 |
| `gate 0.9`, `gate-near 0.5` | 92.4% (15 let stand) | 3.15 | not run | |
| `gate inf` (every token) | 97.0% | 0.41 | 96.1% | 0.40 |
| the reference, again, by another numeric path (`NS_DS41_GPU_ROUTE=0`) | **97.3%**: the floor | 1.27 (reference run 1.24) | | |

**The floor.** The same weights agree 97.3% with themselves when one kernel path changes. The IQ2_XXS records show the same: host routing 98.2%, float32 products 97.9-98.1%, no
fused kernels 98.1%, no reads ahead 97.7%, while a repeat and another pinned size are identical to the bit (100%). So the file's agreement cannot pass about 97-98% against a record made by another
path, and every word in the tables carries that. `gate inf` on the Q8 experts (97.0%) reaches it. The cause of the sensitivity is not found: residency is ruled out, `NS_DS41_QSIM=0` shrinks the
logit differences but leaves 17 of 1,000 top tokens different. A decode row and its prompt-path row also differ. Worth finding before quoting any figure above the floor as a gain.

**What a gate buys.** Against what the records say the same gate could reach if each redone token took the reference's word (IQ2_XXS: 94.2% at 0.45, 96.1% at 0.9, 97.6% at 2, 99.5% at 4; IQ1_M 91.9, 93.5, 97.5, 99.2),
the measured gates give about a fifth to a half of it. The recomputation has the cheap model's dense weights and history around it. Most differences sit at wide margins: IQ2_XXS's 88 are 49 under 0.9 and 39 above it; none at 8 or above. A high gate is where the agreement is: no run between 0.9 and inf was made.
A recomputation costs about 2.4 s, 80% of it reading 18 MiB experts from the drive (buffered). Promethevs' 5% slower does not hold here: their corrections read a plane against a resident 2-bit copy.

**The rule with sampling.** The gate acts on the sampler's input logits: the top-two gap of the raw row decides, the sampler then draws from the corrected row, at any temperature (`--temp 0.7`: 17 redone, 4.34 tg over 256 tokens, no check of its quality). It works with drafts too:
a row under the gate drops the rows after it (IQ2_XXS, Q2_K head, gate 0.45: 92.4%, 4.67 tg). Without a second set, or with another geometry, the load fails. Check: gate `inf` on IQ2_XXS's own experts read through the second-set path gives the rows to the bit of the same pass reading the store (1,000 positions).

**Drafts (IQ2_XXS).** Same records, teacher-forced: towers on 91.3% (552 of 776 drafts kept, 9.71 tg), Q2_K head 91.4% (542 of 778, 10.81), `subst 3` 91.4% (549 of 781, 11.38), `NS_DS41_COLD_SKIP=0.2` **89.4%** (547 of 778, 15.59). Neither the Q2_K head nor substitutes lowers agreement measurably; leaving experts out does, by 1.9.
Free generation, 256 tokens `--ignore-eos`, story then code then the story again (tg; drafts kept):

| Arm | story | code | story again |
|---|---|---|---|
| no towers | 12.91 | 10.90 | 12.70 |
| towers, `--temp 0` | 7.01; 106 of 224 | 10.52; 159 of 183 | 10.31; 106 of 224 |
| towers, `--temp 0.7` | 7.57; 110 of 229 | 10.64; 160 of 182 | 10.18; 110 of 229 |
| `mtp-greedy` at `--temp 0.7` | 8.95; 106 of 224 | 8.87; 159 of 183 | 10.28; 106 of 224 |
| Q2_K head, `--temp 0` | 10.46; 104 of 225 | 6.63; 155 of 187 | 10.93; 104 of 225 |
| `subst 3`, `--temp 0` | 11.83; 111 of 225 | 13.11; 163 of 181 | 12.38; 111 of 225 |
| `COLD_SKIP=0.2`, `--temp 0` | 11.82; 108 of 233 | 12.66; 157 of 184 | 11.96; 108 of 233 |

- **Item 2: greedy does not explain the 55%.** Acceptance follows the text (47% story, 87% code), and sampling at 0.7 keeps the same share. `generate`'s default was already `--temp 0`, so the 55% was likely greedy. `mtp-greedy`'s text is byte-identical to `--temp 0`, as it must be; it only makes a request's temperature not matter. Promethevs' 94.6% is on another head quant and a code prompt.
- **Item 3:** the drafter's experts were already MXFP4, not Q8_0 despite the file's name, so Q2_K saves less than hoped: 6.72 to 4.15 GiB of VRAM (file 8.0 to 5.2 GB). `nextsycl llm requant-draft`; `NS_DS41_DRAFTER` picks it, and its name, `lowbit-dspark-*`, keeps it out of the default search. Acceptance drops 1-4 points. tg varied more between runs than between arms (6.63 to 10.93).
- **Item 5:** `subst 3` keeps the anchor's row and each row's best three on routed experts. Their report does not define "keep 3"; this reading is ours. Timings are single passes.

The story-and-code tg columns ran after the download ended. Nothing here was run through the chat server. Earlier Q3_K_XL figures (a stand-in, moved off the box): IQ2_XXS 92.20%, Q3_K_XL itself 1.77 tg while the download ran.
