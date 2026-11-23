# Inkling-2-Mini (Unsloth GGUFs): port notes

Engine `llm/inkling2` (architecture `inkling2_loop`), kernels `kernels/llm/inkling2/ink.cpp` (its own, in the llm
library) and its silo `kernels/llm/inkling2/silo/` (`dist/silo/libnextsycl-inkling2.so`: `mv.cpp` repacked products,
`fmv.cpp` float products, `dec.cpp` fused decode steps, `attn.cpp` decode attention). Cards
`model-cards/llm/inkling-2-mini-q4kxl.yml` (the fast one) and `inkling-2-mini-q8kxl.yml` (reference quality). Files
on hardac1: `/var/mnt/4TBSSD/nextsycl-models/inkling-2-mini/`.

## Which GGUF layout

Unsloth publishes the same tensors twice: the root files (`general.architecture = inkling2`, 48 `blk` slots with the
loop's per-pass queries in Unsloth's own names - `loop_res_query`, `loop_kv_res_query`, `loop_attn_*`) and
`Mainline_Support/*.gguf_file` (`inkling2_loop`: 24 physical layers, the loop in metadata `loop.start / end / count /
gate_rank`, its controllers as `blk.N.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` and `loop_msg_query`). The engine
reads **Mainline_Support**: it is the layout mainline llama.cpp's graph (`src/models/inkling2.cpp`, `inkling2-loop.cpp`
after PRs 32694 + 32704) is written against, so the reference and the engine name every tensor the same way, and the
weights are stored once (24 layers, not 48 slots). The card renames `.gguf_file` to `.gguf` (`path`).

## The model (read from the metadata, the llama.cpp graph and Thinking Machines' modelling code)

Sources: llama.cpp 690b87d (PR 32694 merged 2026-11-20, PR 32704 2026-11-21), transformers'
`models/inkling2_loop/modeling_inkling2_loop.py`, SGLang's `models/inkling2*.py` (for the drafter).

- 24 layers of 2,304; layers 6-17 run three times with the same weights (48 applications a token, each its own KV
  cache and convolution state: slots 0-23 the physical layers and pass 1, 24-35 pass 2, 36-47 pass 3; the order a
  token runs them is 0-5, 6-17, 24-35, 36-47, 18-23).
- **Attention** (24 query heads, 8 key heads of 128; layers 5, 11, 17, 23 full, the rest a window of 1,024 - a key
  less than 1,024 positions back): logit = q.k / sqrt(128) + r . relbias[distance] (r [16] a head from its own
  projection, relbias [extent = 1,024][16]; 0 past the extent), plus a per-row sink logit (its own projection) with no
  value; q, k and r clipped to an L2 norm (22.63, 8) after a per-channel scale (k without), v RMS-normed without a
  weight; the full layers scale q and r by 1 + 0.1 ln(max(1, (pos + 1) / 8192)). The output is gated by sigmoid of its
  own projection of the input. No RoPE.
- **Short convolutions** (4 taps, a state of the last three inputs): on k and v after their projections (the full
  layers add input-dependent taps a key head, `attn_{k,v}_conv_proj`), on the attention's output and on the FFN's output;
  out = (x + conv(x)) / sqrt(taps + 1) (sqrt(2 taps + 1) with the dynamic taps).
- **Attention over the depth** (block residuals, blocks of 6): the stream restarts at layers 0, 6, 12, 18 (the old one
  joins the blocks); each layer's attention input and FFN input are softmax mixes of the blocks and the stream, each
  RMS-normed (no weight), by learned queries (3 for a full layer's q / k / v inputs, 1 otherwise; none on layer 0);
  the head's input the same with `output_res_query`.
- **The loop's messages**: passes 2 and 3 start from the prelude's output again; each looped layer mixes the previous
  pass's blocks after its first and its output by `loop_msg_query` (4 messages: q, k, v, FFN) and a controller per
  input: out = x - tanh(F u) x + tanh(G u) msg, u = D [rms(x) | rms(msg)] (rank 64, F32 weights).
- **MoE** (128 experts of 1,152, 8 a token, no shared expert): p = sigmoid(router . x) (router F32), the 8 largest of p
  + bias, weights p / sum x 8; gate | up one matrix, both scaled by the expert's `.scale` (NVFP4's per-expert factor),
  SiTU: 4 tanh(g / 4) sigmoid(g) x 25 tanh(u / 25); down scaled likewise.
- Embedding RMS-normed with a weight (`token_embd_norm`); logits soft-capped at 30, the padded vocabulary (201,024)
  cut to the tokenizer's 200,058.
- Types in UD-Q4_K_XL: experts NVFP4 (layers 0-19; Q8_0 in 20-23), attention / embedding / head Q8_0, sink and
  convolution-tap projections BF16, norms / queries / controllers / router F32. NVFP4: blocks of 64 - four UE4M3 scales
  (one a 16 values; ggml halves them, its E2M1 table being doubled), then 32 bytes of codes, sub-block s's value j in
  the low nibble of byte 8s + j and value j + 8 in its high nibble.
- Tokenizer: byte-level BPE, pre `inkling2` = GPT-4o's split. Chat template: `<|message_ROLE|><|content_text|>...
  <|end_message|>`, the effort as a system message before the first other message, no generation prompt (the model
  opens its turn with `<|message_model|>`, thinks in `<|content_thinking|>`, answers in `<|content_text|>`, ends with
  `<|content_model_end_sampling|>` = eos); tool calls `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":..}`.
  `nextsycl_tok::inkling_chat` / `inkling_channels` (the server splits `reasoning_content` and turns the calls into
  `tool_calls`); the user-turn marker `<|message_user|>` is in `USER_TURNS` (prompt cache checkpoints).

## Parity

Reference: llama.cpp 690b87d built for the CPU (`reference/inkling2/ink-dump.cpp`: one pass's named tensors as float32,
then greedy tokens with each one's top-1 / top-2 gap; `-fa off`, one ubatch of up to 2,048), UD-Q4_K_XL. Engine:
`nextsycl llm check` (B70).

| prompt | tokens | layer outputs (`l_out`, cosine) | logits (cosine) | next token | greedy run against llama.cpp's |
|---|---|---|---|---|---|
| short | 27 | 0.9987 - 0.99998 | 0.99998 | same | 48 of 48 the same |
| medium | 315 | drift to ~0.983 in the coda (result_norm 0.96) | 0.9967 | same | the same to token 35, where the reference's own top-1 / top-2 gap is 0.061 (a near tie) |
| short, UD-Q8_K_XL | 27 | 0.9992 - 0.99998 | 0.99999 | same | 48 of 48 the same |
| long | 2,410 (last 362 rows compared) | from layer 7, to ~0.83-0.86 in the coda (result_norm 0.929) | 0.9936 | same | 32 of 32 the same |

The long prompt's drift is the model's sensitivity, not a fault: two numeric paths of this engine (the silo's - XMX
flash attention in half, Q8_0-quantized activations - and its own kernels' - float32 throughout) differ from each
other by about as much (layer 29's output cosine 0.970, the coda's 0.85-0.92, result_norm 0.985, the logits 0.998),
and each is as far from llama.cpp (0.937 / 0.934 at layer 29, result_norm 0.929 / 0.929). The 48 applications a token
- the same weights three times, the loop's messages, the depth mixes - amplify a rounding difference; a 512-row chunk
instead of 2,048 changes nothing (not the chunking). Both cards give the same tokens (B70 and B65: 48 / 128 / 32 rows,
the medium prompt's near tie at 35 on both).

- **Kernels against ggml**: every expansion (Q8_0, NVFP4, BF16 - `llm kernels` with `NS_INK2_DEQ_REF` =
  `reference/inkling2/deq-ref.cpp`'s rows of ggml's `dequantize_row_*`) bit for bit; the silo's repacked products
  within the Q8_0 activation rounding (rel err 4.5e-4 / 6.3e-4 on NVFP4 experts, the float products 1e-8), and a
  16-row pass's columns bit for bit its 1-row passes.
- **spec-check** (a verify pass of [token, draft] against one-row decode): 0 difference, the top token the same in 46
  of 46 rows, on both cards and both quants.
- **Paths against each other** (`llm generate --score`, 256 tokens of prose, KL of one against the other): the silo's
  decode against its own kernels 0.11 (top-1 89%), the silo's prompt rows (64 a pass) against its decode rows 0.07 -
  the same sensitivity again; 8-row passes against 1-row passes 0.00000 (bit for bit). UD-Q4_K_XL against
  UD-Q8_K_XL (512 tokens, both on the silo): KL 0.142, the same top token 92.8%, perplexity 35.4 against 34.4 - within
  the paths' own spread on this text, so the bigger file buys little here (Unsloth's own KLD 0.0107 is measured on one
  numeric path).

## Speed

UD-Q4_K_XL, one card, greedy, the prompt then 128 generated tokens (`llm generate --ids`: `pp` the prompt's
tokens a second, `tg` the generated tokens a second after it). No drafter (see below), so prose and code
generate at the same tg (nothing in a decode step depends on the text).

| card | pp 512 | pp 4K | pp 32K | tg after 512 | tg after 4K | tg after 32K | tg, short chat |
|---|---|---|---|---|---|---|---|
| B70 | 1,138 | 2,085 | 1,657 | 57.3 | 54.6 | 41.8 | 57.6 |
| B65 | 832 | 1,470 | 1,080 | 51.7 | 45.7 | 33.1 | 52.4 |
| B70, UD-Q8_K_XL | 925 | 1,931 | - | 46.9 | 45.0 | - | 47.4 |
| llama.cpp 690b87d, CPU (Ryzen 7 7700X, 12 threads) | 39.1 | - | - | 10.6 (tg128) | - | - | - |

Both cards together: the model fits one card with 256K of context, so a split would only add a hop a layer
application (48 a token); the cards serve two conversations instead (`gpus: '0|1'`). Not measured.

**The bound.** A token reads 4.00 GB (UD-Q4_K_XL; the loop counted three times): attention, router and tap projections
1.43, 8 experts x 48 applications 1.85, the controllers 0.23, the head 0.49. At the silo's measured DRAM rate on the
B70 (the head product, 548 GB/s) that is 7.3 ms = 137 tok/s (152 at 608 GB/s); tg 57 is ~42% of it. Where the 17.4 ms
go (device time, `NS_INK2_PROFILE=gpu NS_PROFILE=gpu`): the experts' gate | up 3.6 ms and down 1.8 (~330 GB/s), the
dense products 2.1 (five a layer in one launch) + 1.8, the depth mixes 1.8 (117 a token, one work-group a row), the
attention 1.2, the controllers 0.9 + 0.6, the post-projection step 0.8, the remaining small launches; ~2 ms of launch
gaps (about 1,000 launches a token). UD-Q8_K_XL reads 5.77 GB a token (bound 95 tok/s).

What moved it (B70 tg on the short chat): the engine on its own kernels 20 -> the silo's repacked products and a
sub-group router 41.7 -> fused decode steps (depth mix + norm + quantization; convolutions + clip / norm + cache
write; merge + gate + quantization; convolution + residual) 44.5 -> the controllers' down product rewritten (z in
local memory, 4 rows a sub-group) 57.1. pp: XMX flash attention for prompts (museglimmer's, with the relative bias
and the sink) took pp 4K from 945 to 2,100 and pp 32K from 517 to 1,657.

Tried and dropped: the router's choice made inside the experts' gate | up launch (every work-group recomputing it:
gate | up 3.6 -> 4.8 ms for 0.6 saved); loading an NVFP4 line's raw bytes before the row loop (3.6 -> 4.0 ms); a
512-wide depth mix (no change).

## Round 2: decode toward the bound, and the drafter

**Where a token's time goes** (B70, UD-Q4_K_XL, `NS_INK2_PROFILE=gpu NS_PROFILE=gpu`, 256 tokens; device ms a token).
The device time adds up to the wall time (17.3 ms against 1 / 58.0 tok/s): the host stays ahead and the launches run
back to back, so SYCL graphs or fewer launches alone buy nothing here (the kernels' own time is the cost).

| kernel | before | after | what changed |
|---|---|---|---|
| experts gate / up | 3.66 | 3.69 | (unchanged; a two-way split of the lines was slower: 4.9 on the B65) |
| experts down | 1.80 | 1.82 | |
| dense products (q k v r gate in one launch / the rest) | 2.14 + 1.84 | 2.22 + 1.86 | |
| depth mixes (117 a token) | 1.83 | 1.18 | templated on its states and queries, 576 threads |
| decode attention | 1.39 | 0.99 | a lane a key, the split's softmax at once (no reduction a key) |
| router product + choice | 0.44 + 0.65 | 0.75 | rows in quarters; the K largest by a parallel rank |
| controllers (down, apply, norms) | 0.94 + 0.63 + 0.61 | 0.44 + 0.40 + 0.43 | three controllers a launch each |
| post-projection step | 0.84 | 0.84 | (the sink and tap projections folded in) |
| B70 tg, short chat | 58.0 | 66.1 | (device time 15.8 ms) |

B65 the same way: tg 52.4 -> 59.4. The bound is unchanged (4.0 GB a token, 7.3 ms on the B70): tg 66 is ~48% of it.
pp is unchanged (B70 1,147 / 2,089 / 1,655 at 512 / 4K / 32K; tg after them 65.9 / 58.9 / 43.4).
The rest is in the expert products (~330 GB/s of the ~550 the dense ones reach): with the NVFP4 decoding table
removed (a wrong-answer experiment) they stream at 465 GB/s, so the code decode costs ~30%; the rest is latency.

**The DSpark drafter** (`dspark-Inkling-2-Mini-Q8_0.gguf` beside the model, 470 MB with the fold). Reference: SGLang's
`models/inkling2_mtp.py`, `dspark.py`, `dflash.py`, `kernels/ops/speculative/inkling2_mtp.py` (llama.cpp's mainline
and Unsloth's fork b12588 have DSpark, but neither captures Inkling's looped features); the original repository
ships only the weights. The target keeps four BF16-rounded features a position (the stream going into applications 5,
23, 41 and the head's normed input); each drafter layer's context is their softmax(`dspark_fuse`) mix normed, its k
(normed, NEOX at theta 1e6) and v cached; a draft is one non-causal pass of [last token, MASK x n] (layers 0 and 2
over a 1,024 window), the drafter's norm, the target's head; the Markov head's RNN then runs a position at a time
(state, w1[previous token], hidden row, f_rnn) and its latent goes through `dspark_hidden_out` folded into the head at
load (vocab x 256 BF16) as a bias; drafts are the argmaxes of softcap(base + bias). A draft the drafter gives less
than `NS_INK2_DRAFT_PMIN` (default 0.7) ends the block. Verify passes keep the engine's identity: greedy text drafted
= plain (short / medium / long the same tokens), sampled text drafted = plain (the same seed, the same 200 tokens),
spec-check 0.

What a verify costs (B70, `--score` passes of the model's own text): 1 row 16.4 ms, 2 rows 21.3, 3 rows 24.9, 4 rows
28.4, 5 rows 32.2, 8 rows 42.0 - ~4 ms a row. The experts the rows choose (`NS_INK2_UNION=1`, distinct experts a
layer application; prose and code alike): 8.0 / 12.6 / 16.2 / 19.2 / 22.0 / 28.5 for 1 / 2 / 3 / 4 / 5 / 8 rows. A
verify reads each distinct expert once for all its rows (`moe_union`, `moe_gu_union`, `moe_down_union`; bit for bit
the per-pair kernels' sums) - little faster in practice (the pairs of a shared expert already met in L2).

tg on the B70 (400 tokens, the chat template; `drafts` 3 and p_min 0.7 the defaults):

| text | temperature | plain | 2 drafts | 3 drafts | 5 drafts | 3 drafts, no p_min |
|---|---|---|---|---|---|---|
| prose | 0 | 64.6 | 66.7 | 66.9 | 65.1 | 56.6 |
| code | 0 | 65.0 | 85.1 | 88.4 | 89.8 | 86.3 |
| prose | 1.0 | 62.3 | 68.2 | 67.5 | 64.8 | 58.2 |
| code | 1.0 | 62.0 | 94.3 | 99.2 | 104.2 | 98.2 |

UD-Q8_K_XL on the B70: prose 51.5 -> 53.4 greedy, 49.7 -> 52.5 at 1.0; code 51.5 -> 87.7, 49.6 -> 90.2; pp 876 / 1,892.

and on the B65:

| text | temperature | plain | 2 drafts | 3 drafts | 5 drafts | 3 drafts, no p_min |
|---|---|---|---|---|---|---|
| prose | 0 | 58.5 | 60.1 | 59.6 | 57.7 | 49.8 |
| code | 0 | 58.7 | 78.0 | 80.1 | 80.8 | 76.0 |
| prose | 1.0 | 55.8 | 56.8 | 56.6 | 55.0 | 45.9 |
| code | 1.0 | 55.6 | 77.7 | 83.8 | 85.7 | 79.6 |

UD-Q8_K_XL on the B65: plain 46.8 / 45.4 (prose, greedy / 1.0), drafted 49.3 / 45.9; code 46.8 -> 65.4 greedy, 44.9 ->
68.9 at 1.0; pp 725 / 1,374 at 512 / 4K. Acceptance at temperature 1.0 with the defaults: code 90%, prose 65% of
the drafts verified (the gate verifies about half as many prose drafts).

GPU sampling (`ns_ink2_sample`): the 256 likeliest are now found among the values within 1 .. 64 of the largest
(6 scans of the vocabulary, not 34); the same draws (a 200-token sampled run identical before and after).

## Round 3: the NVFP4 products without a table, the drafter retuned

**The expert products.** The decode table in local memory was the cost, not the reads: 18 gathers a 32-code line
(16 code bytes, 2 scales) per row. Filling it per work-group was only 2.5% (B65: 283 against 276 GB/s without the
fill); a register table read by sub-group shuffles was 4x slower (66 GB/s). The codes are now decoded in the word's
arithmetic - four codes a 32-bit word: magnitude c = code & 7 is the value up to 4, and from 4 on c + (c & 3) +
2 b0 b1 gives 4 6 8 12; the sign by a borrow-free subtraction ((m | 0x80) - 2m when negative, ^ 0x80) - and the UE4M3
scales by their bits; the same bytes and floats as ggml's table (all 65,536 code quads and 256 scales checked), so
every product is bit for bit what it was (spec-check 0, the silo-against-own-kernels KL unchanged to the digit).

| | B70 before | B70 after | B65 before | B65 after |
|---|---|---|---|---|
| 8 experts' gate / up + down, streamed (GB/s) | 329 | 382 | 276 | 314 |
| without any decode (the bound of this kernel shape) | 465 | - | - | 446 |
| experts, device ms a token | 5.50 | 5.05 | 6.35 | 5.82 |

What else was tried: rows a sub-group (2 / 4 / 8 values of gate | up, 2 / 4 / 8 down rows: within 4%); unrolling the
line loop by 2 or 4 (no change); a two-way split of the lines (slower). Dense products of few rows (under 16K a
launch: q k v r gate, the output projection) now take a row a sub-group - the attention output (2,304 rows of 3,072)
still streams at ~220 GB/s alone, latency-bound; mv-many 2.2 -> 2.06 ms, mv 1.86 -> 1.77.

GPU sampling: the windows are now 1/8 .. 22.6 by factors of sqrt 2 and up to 8K candidates (the K-th largest is the
same for any window): temperature 1.0 plain tg on the B70 65.3 -> 67.4.

**The drafter.** A draft below p_min now stops its block on the GPU: the later steps' fold products and picks do
nothing (a 103 MB read each). That made longer blocks nearly free when unsure; retuned at temperature 1.0 (B70 /
B65, tg, 400 tokens):

| drafts, p_min | prose B70 | code B70 | prose B65 | code B65 |
|---|---|---|---|---|
| plain | 67.2 | 65.1 | 59.6 | 57.3 |
| 3, 0.7 (round 2) | 71.7 | 102.0 | 59.2 | 84.6 |
| 4, 0.8 | 71.9 | 107.7 | 60.0 | 87.9 |
| **5, 0.8 (default)** | 71.7 | 110.7 | 59.7 | 89.4 |
| 5, 0.6 | 70.4 | 108.1 | 57.4 | 85.4 |

A first version left the stopped picks' tokens unwritten, and the next Markov step indexed w1 by arena garbage: a GPU
page fault on both cards (15:14, the runs stopped by name). The stopped pick now writes token 0 and the Markov input
clamps its token to w1's rows.

What a 4-row verify costs (28 ms against 16): the experts the rows choose (19 distinct a layer application against 8)
- the union kernels decode each expert's lines once for all its rows, so the cheaper decode helps there in full.

**Round 3 tables** (400 tokens, the chat template, plain / drafted with the defaults):

| card, quant | prose, 0 | code, 0 | prose, 1.0 | code, 1.0 | pp 512 / 4K / 32K |
|---|---|---|---|---|---|
| B70 Q4 | 68.7 / 69.3 | 68.7 / 91.3 | 67.4 / 71.8 | 65.1 / 110.8 | 1,135 / 2,056 / 1,648 |
| B65 Q4 | 61.3 / 61.5 | 61.2 / 83.5 | 59.7 / 59.8 | 57.3 / 89.5 | 836 / 1,477 / 1,084 |
| B70 Q8 | 52.9 / 54.9 | 52.1 / 95.1 | 51.6 / 54.0 | 50.3 / 95.4 | 859 / 1,851 / - |
| B65 Q8 | 47.5 / 49.9 | 47.7 / 66.2 | 46.7 / 47.7 | 45.1 / 71.8 | 719 / 1,362 / - |

Short chat tg (256 tokens, greedy): B70 66.1 -> 70.0, B65 59.4 -> 62.0. The greedy references unchanged (short 48 / 48,
medium the near tie at 35, long 32 / 32), spec-check 0 on both cards and quants, drafted text = plain text greedy and
sampled.

## Round 4: a shorter decode; what did not pay

**The NVFP4 decode in 16 operations a word** (from 20): the repacking now stores -0 (code 8) as +0 (code 0) - the same
value - so a decoded magnitude negates as (m ^ 0xFF) + 1 with no carry out of its byte, and the b0 b1 term takes one
fewer step. The same bytes, so every product is what it was (the silo-against-own-kernels KL and the sampled tokens
of round 3 reproduced to the bit on both cards). Experts a token: B70 5.05 -> 4.83 ms (gate | up 3.33 -> 3.22, down
1.71 -> 1.62; 383 GB/s by the profile), B65 5.82 -> 5.49 (337 GB/s).

Tried and dropped (each measured, then reverted):

| change | what happened |
|---|---|
| the attention-output product (2,304 rows of 3,072) with its row's lines split over 4 sub-groups and an SLM sum | slower: the projections 2.1 -> 2.9 ms (B65) - four times the work-groups, each too short |
| two lines in flight a lane (separate sums) for one-row products | no change (2.12 / 1.70 ms) |
| the attention's convolution and residual made inside the FFN's depth-mix launch (one row) | the convolution's launch saved 0.11 ms, the mix grew 0.18 (one work-group a row doing 2,304 channels serially) |

The depth mixes (1.2 ms, 117 a token) and the post-projection step (0.83) stay latency-bound single-work-group launches;
neither fusion above nor more threads helped. The 4-row verify still reads 19 distinct experts a layer against 8 (the
union kernels decode each once for all its rows).

**Round 4 tables** (400 tokens, the chat template, plain / drafted with the defaults; short chat tg greedy):

| card, quant | short chat | prose, 0 | code, 0 | prose, 1.0 | code, 1.0 | pp 512 / 4K / 32K |
|---|---|---|---|---|---|---|
| B70 Q4 | 71.0 | 69.8 / 70.3 | 70.1 / 93.3 | 67.6 / 72.5 | 65.7 / 112.0 | 1,134 / 2,095 / 1,650 |
| B65 Q4 | 63.1 | 61.8 / 62.0 | 61.8 / 85.5 | 59.7 / 60.7 | 57.9 / 90.9 | 831 / 1,458 / 1,084 |
| B70 Q8 | - | 53.7 / 55.6 | 53.8 / 95.2 | 52.2 / 54.5 | 50.9 / 96.1 | 1,013 / 1,910 / - |
| B65 Q8 | - | 48.4 / 50.5 | 48.5 / 67.2 | 46.9 / 48.3 | 45.6 / 73.2 | 720 / 1,361 / - |

## Quantization choice

- **UD-Q4_K_XL** (17.6 GB, the fast card): the experts in their native NVFP4 (the checkpoint's own format: no loss),
  the rest Q8_0; Unsloth's KLD 0.0107 against UD-Q8_K_XL, top-1 98.0%. Smaller files (Q4_K_M, IQ4_XS) differ only in
  a few tensors (layers 20-23's gate | up Q5_K, the head Q6_K) for at most 6% fewer bytes and would need K-quant
  kernels: not worth it.
- **UD-Q8_K_XL** (22.9 GB, the reference card): the same NVFP4 experts, BF16 attention / embedding / head / last
  four layers' experts / controllers; through the silo's float products.

| card | vram (measured, the B70, front door's system.busy) | context that fits with 1.5 GiB free |
|---|---|---|
| UD-Q4_K_XL | 18.7 GiB at 32K, 25.7 GiB at 256K (32 KiB a token: 8 full-attention slots) | the model's 256K, plus a 32K session (~27 GiB) |
| UD-Q8_K_XL | 26.1 GiB at 128K | 128K plus a 32K session (~27.6 GiB); 256K alone (~30.1 GiB) just fits |

## Not done

- **The drafter against a reference**: no runnable reference produces its tokens for this model (see round 2); its
  acceptance (90% of code drafts at temperature 1.0) is the check. The checkpoint's drafter state is saved and
  restored with the session's.
- **Vision** (the mmproj ViT): TODO.md.
- **llama.cpp on the GPU**: not built for SYCL here (its NVFP4 support there unchecked); only the CPU numbers above.

## Porting it to another backend (nextcuda and the like)

Everything above the kernels is backend-neutral and can be copied or read as the spec; only the kernels are SYCL.

**Reuse as is (Rust, no SYCL):**
- `crates/gguf` - the NVFP4 type (ggml code 40, blocks of 64 values in 36 bytes: four UE4M3 scales, then 32 bytes
  of E2M1 codes; see "The model" for the nibble order and ggml's halved scales).
- `crates/tok` - `inkling_chat` (the template, the effort's system message, tool calls) and `inkling_channels` (raw
  text to reasoning / content / tool calls); `<|message_user|>` goes in the prompt cache's user-turn markers.
- `glue/serve/src/serve.rs` - the `ink` channel detection (how a reply is split into `reasoning_content`, `content`
  and `tool_calls`).
- `model-cards/llm/inkling-2-mini-q4kxl.yml` / `-q8kxl.yml` - the files (Mainline_Support, pinned revision), the
  drafter key, the measured VRAM.

**Read as the spec:**
- `llm/inkling2/src/engine.rs` - the whole forward pass in order: the loop's slot numbering (0-23, 24-35, 36-47),
  where the stream restarts, the depth mixes, the controllers, the router, the coda and the head. Every kernel call is
  one step, so it is the checklist for a new backend.
- `llm/inkling2/src/engine/dspark.rs` - the drafter: the four kept features, the non-causal draft pass, the Markov
  RNN, the head bias folded at load, and the p_min gate.
- `kernels/llm/inkling2/ink.cpp` - plain float32 kernels for every op (the slow path, but the semantics to match).

**Rewrite for the backend:** the silo (`mv.cpp` repacked NVFP4 / Q8_0 products, `fmv.cpp`, `dec.cpp` fused decode
steps, `attn.cpp` / `pattn.cpp` decode and prompt attention, `draft.cpp`). NVFP4 is the checkpoint's native expert
format. On a GPU with FP4 tensor cores, the expert products can run on them directly; elsewhere, decode the codes in
registers. Read "Where a token's time goes": experts plus dense products are ~2/3 of a token.

**Order that worked here:**
1. Load the Mainline_Support file and check every expansion bit for bit against ggml (`reference/inkling2/deq-ref.cpp`).
2. Run one prompt through float32 kernels and compare each layer's output with llama.cpp
   (`reference/inkling2/ink-dump.cpp` dumps the named tensors; `gguf_np.py` reads them). Expect the long-prompt drift
   described in "Parity". It is the model's sensitivity, so judge by the next token and greedy runs, not by a cosine
   alone.
3. Then the fast products and fused steps, each scored against the float32 path, on **each quant** (Q8_K_XL keeps
   the controllers in BF16).
4. Then the drafter. A verify must keep greedy text identical to plain text.

**Gotchas:** layer 0 pushes no stream (`l > cs`); 48 applications a token means 48 KV / convolution-state slots, not
24; the logits are soft-capped at 30, and the padded vocabulary (201,024) is cut to 200,058; the full layers' q and r
scale with position (the log term); an arena that spills must refuse, not free buffers the queue still reads.

## Lessons

- An arena that spills into owned buffers frees them while the queue still reads them: a GPU page fault on the
  2,410-token prompt (B70 engine reset, 2026-11-22 07:26). The arena now refuses (`ab()` errors), every helper rewinds
  its scratch, and `arena_bytes` is sized from the chunk.
- A block's stream restarts at layers 0, 6, 12, 18 - but layer 0 has no earlier stream to push (`l > cs`, not
  `l != cs`): one wrong push gave layer 12 a cosine of 0.97, found by dumping the states.
- A quant's other types reach other paths: UD-Q8_K_XL stores the loop's controllers in BF16, which the fused
  controller kernels read as float32 (fixed by widening them at load, exact). spec-check and 1-vs-8-row checks cannot
  see that (both sides wrong the same way): score each quant's silo path against its own kernels (`NS_SILO=0`).
