# DSpine drafter for Qwen3.8-27B (llm/qwen35)

Tencent's `Qwen3.8-27B-DSpine-b16` (huggingface.co/tencent/Qwen3.8-27B-DSpine-b16, Apache-2.0, 2.03 B parameters in bfloat16,
`DFlashDraftModel`): a speculative-decoding *draft* for the dense Qwen3.8-27B, run by the `qwen35` engine in MTP's place.
Opt-in (`NS_Q35_DSPINE=DIR`), exact (the target's own sampler verifies every draft). **It does not beat MTP here** except
on code against the Q8_0 file; numbers below. Reference: Tencent's `dspine_mlx` (read, not run); the numpy reference in
`tooling/dspine-ref.py` redoes the draft from the checkpoint and checks the engine's.

Files: `llm/qwen35/src/dspine.rs` (config, weights, tables, pure logic with tests), `llm/qwen35/src/engine.rs`
(`dspine_rows`, `dspine_draft`), `kernels/llm/qwen35/silo/dspine.cpp` (six small silo kernels). The checkpoint's format:
the original safetensors, read directly (a third-party GGUF conversion, aj9o9's, has the same tensors - names mapped, the
zero `out` projection dropped - but its BF16 file is the same size and its Q8_0 file quantizes the bridge's codebook, which the
reference keeps exact; not fetched).

## What it is

- **Context**: for every committed position, `ctx = hidden_norm(fc(concat of the target's residual stream after layers 1,
  16, 31, 46, 61))` (fc 25,600 -> 5,120). Each of the five draft layers keeps k / v of `ctx` (its own k_proj / v_proj, k_norm,
  rotary at the position) in its cache; the engine writes them behind every target pass (prompt chunks and verify rows;
  20 KiB a position, `NS_Q35_DSPINE_CTX` positions, 32,768 by default).
- **Block**: rows `[anchor, mask, mask, ...]` (the last committed token, not yet fed to the target; mask token 248070) at the
  positions after the context, embedded with the *target's* embedding. Attention: context fully visible (window 8,192), the
  block causal. 32 query / 8 key heads of 128, q / k norm, NeoX rotary at theta 1e7, SwiGLU of 17,408.
- **Code bridge**, after every layer, over the block's rows: `rms = |raw| / sqrt(H)`, `code = sqrt(1024) unit(read_l (raw /
  (rms + eps)))` (row 0: `sqrt(1024) codebook[anchor]`), `packed = [G; Gp; A] code` (q | s | v, rank 512),
  `message[i] = sigmoid(q[i] + s[i-1] + bias) v[i-1]`, `out = raw + rms W_l message` (rows 1..). `out` is the next layer's
  residual stream: a position's predicted feature reaches the next position through depth, so the block's tokens depend
  on each other without being drafted one at a time.
- **Head and refinement**: the final norm, the target's head -> each row's 16 best tokens; then each row's pick among them is
  conditioned on the previous row's pick (the anchor for row 0) through two vocabulary tables built at load from the target's
  head rows and the checkpoint's codebook (`TG` V x 512, `TSV` V x 1,024, half; 0.76 GiB): `score[k] = raw . norm_w .
  head_row[cand k] + rms . m . TG[cand k]`, `m = sigmoid(q + TSV[sender][:512] + bias) TSV[sender][512:]`. The drafts are the
  chain of those picks.
- **Block size**: the block is causal, so drafting `NS_Q35_DRAFTS` (at most 7: a verify pass is at most 8 rows) rows gives those
  rows' values exactly - the reference's 16-row block, cut to what the target verifies. A 16-row verify would leave the
  decode products (at most 8 rows) for the dequantize-to-half GEMM path.
- **Engine**: the layers' products are the engine's own (Q8_0 kept on the GPU: made on the host at load, as EAGLE-4's);
  the attention is the engine's D=128 kernel (it covers one key head a query head: the four query heads of a key head are
  four calls, rope over a strided gather, and the o projection's columns split four ways to match - no new attention
  kernel). The bridge is float32. One GPU only (the aux states come from one device); `steps_together` is false, as EAGLE's.
  GPU memory: 2.58 GiB with the default 32K cache (1.71 GiB weights, 0.76 tables, 0.1 bridge, 0.64 cache).

## Checked

- Against the numpy reference (`tooling/dspine-ref.py`, float32, from the dumped aux states of a real run, 7 drafts on Q4):
  final rows cosine 0.9998-0.99995, last layer's raw 0.9999, packed 0.99999, the candidates are the logits' top 16 in every
  row, the refinement's picks match in 112 of 112 (row, sender) cases in each of the 7 drafts, and the drafts are the same
  tokens. (The reference's bfloat16 roundings are not copied; ours are float32 with Q8_0 layers.)
- Exactness: greedy output (256 tokens, `--ignore-eos`) with the drafter is byte for byte the plain greedy output, for 3, 5
  and 7 drafts, on code, prose and a 13.6K prompt, Q4_K_M and (code, prose) Q8_0.

## Speed (hardac1, B70, 2026-11-09, greedy 256 tokens; tg tok/s, accepted = drafts kept of drafts compared; tokens per verify round = 256 / (256 - accepted))

UD-Q4_K_M, 32K context, MTP 3 drafts (the default) vs DSpine with 3 / 5 / 7 drafts:

| prompt | plain tg | MTP tg (tokens/round) | DSpine 3 | DSpine 5 | DSpine 7 |
|---|---:|---:|---:|---:|---:|
| code (34 prompt tokens) | 27.3 | **58.5** (3.3) | 56.7 (3.4) | 56.9 (4.1) | 54.6 (4.7) |
| prose (32) | 27.3 | **52.4** (2.9) | 47.7 (2.8) | 41.6 (3.1) | 37.0 (3.3) |
| 13.6K-token document | 24.2 | **49.8** (3.1) | 35.4 (2.8) | 34.6 (3.6) | 33.2 (3.9) |

Prompt reading is unchanged (13.6K tokens: pp 1,043 plain, 1,111 MTP, 1,064-1,126 DSpine: its context rows cost little).
Q8_0 (4,096-token context and draft cache: the file, the draft and the verify snapshots do not fit a B70 with more):

| prompt | plain tg | MTP tg | DSpine 3 | DSpine 5 | DSpine 7 |
|---|---:|---:|---:|---:|---:|
| code | 18.3 | 42.0 / 40.4 (two runs) | 41.3 | **48.2 / 47.9** | 39.3 / 39.7 |
| prose | 18.5 | **36.7** | 33.8 | 33.0 | 27.6 |

Why it does not win: DSpine's blocks are accepted further (4.7 tokens a round with 7 drafts on code against MTP's 3.3), but a
verify row costs ~6.5 ms on the Q4 file (the silo's decode products are compute-bound at 8 columns: 84 ms for an 8-row pass
against 37 for one row; the device-side probe, `NS_Q35_PROFILE=1`), and the draft itself is ~11 ms (the five layers' 2 GB of
weights at ~300 GB/s, the head's 7 rows 3 ms). A round costs ~59 / 73 / 85 ms at 3 / 5 / 7 drafts against MTP's 56 for 3: the
extra rows buy ~0.3 tokens each, ~21 ms a token against MTP's ~17. Where a row is cheap (Q8_0 is bandwidth-bound: its rows are
nearly free) and the text is predictable (code), 5 drafts win by ~15-19%; on prose and on a long document the drafts are
accepted less and MTP's cheap round wins. With DSpine's first three drafts accepted as often as MTP's (87% against 86% on code)
the draft's cost is all that is left.

What would change it: a verify that is cheaper per row past 4 (XMX decode products for 8-16 columns in the silo), a
confidence cut of the verified rows, a cheaper draft (the bridge's float32 products and the 130 launches of a draft).

## Env

`NS_Q35_DSPINE=DIR|auto` (auto: a `dspine/` folder beside the model), `NS_Q35_DSPINE_CTX`, `NS_Q35_DSPINE_DUMP=DIR` (debug dump
for the reference), `NS_Q35_DRAFTS` (the drafts a step, at most 7). No model card: not a default and not a clear win.
