# More language-model engines

[← nextsycl](../README.md)

## Qwen3.8-27B (dense)

`qwen35` (`llm/qwen35`, `kernels/llm/qwen35`): Qwen's dense Qwen3.8 / Qwen3.5 family as llama.cpp's `qwen35` graph has
it - 64 layers, three of every four gated DeltaNet (the conv + SiLU, the delta-rule recurrence, a SiLU-gated norm),
the fourth gated full attention (24 / 4 heads of 256, a sigmoid gate a head in the query projection, q / k norms,
rotary positions on 64 of the 256 features at theta 1e7), each with a dense SwiGLU of 17,408. The weights stay in
their GGUF blocks: decode multiplies them through the Q8_1 products (the same as GLM's and Qwen3.8-Flash-Next's), a
prompt chunk expands each matrix to half for the GEMM; the delta-rule scan, conv, norms and SwiGLU are the glm5next
engine's kernels (GDN is KDA with one decay a head), the attention and the gates this engine's.

- **Checked** against llama.cpp (CPU, the same Q8_0 file, `reference/llama-dump`): every layer's output cosine
  0.99993 or better, the final norm 0.99993, the same next token at the same logit; decoding the last token alone
  after the rest agrees with the whole prompt at once (0.99997). The greedy continuation of a prompt agrees with
  llama.cpp's until two candidates sit within 1% of each other.
- **Speed** (one B70, 2026-10-11, 1fecf2b; `llm generate --ids` of the benchy v1 prompts, 128 greedy tokens, MTP on - the
  default), prompt read / decode, tokens/s:

  | File | 20 | 2,185 | 8,000 | 40,000 | 2,185 without MTP |
  |---|---:|---:|---:|---:|---:|
  | Q8_0 | 57 / 42.9 | 1,014 / 44.9 | 1,044 / 39.9 | (32K context) | 1,178 / 18.5 |
  | UD-Q4_K_M | 60 / 59.5 | 1,184 / 57.4 | 1,245 / 55.6 | 772 / 46.9 | 1,195 / 27.3 |
  | UD-Q3_K_XL | 60 / 55.1 | 1,180 / 58.9 | 1,243 / 57.3 | 772 / 48.5 | 1,193 / 29.0 |

  A coding answer (400 tokens) decodes at 61 tok/s on UD-Q4_K_M and UD-Q3_K_XL. 84-89% of the drafts are accepted;
  prompts read a little slower with MTP (the draft block follows every pass; Q8_0's chunks are 1,024 rows beside it).
  Q8_0 decode is bandwidth-bound (27 GiB a token). With the shared kernels alone (`NS_SILO=0`, no drafts) UD-Q4_K_M
  reads 2,185 tokens at 441 tok/s and decodes at 18.5.
- **MTP:** the draft block (block 64 of the MTP file - `MTP/` beside the model, the catalog's `q27-mtp`, or
  `NS_Q35_MTP`) reads each position's normed hidden state with the token after it. It runs behind every pass over
  that pass's rows, so its cache follows the conversation; `step` drafts `NS_Q35_DRAFTS` tokens (3: 2 is as fast, 4
  and 5 slower), verifies [token, drafts] in one pass and keeps what the sampled tokens match, so what is committed
  is distributed as plain sampling. Verify rows are bit for bit the one-token decode's (`spec-check --rows 2..4`).
  `--no-mtp` turns it off. Q8_0 with MTP fits its catalog 32K context on one card, not 40K.
- **EAGLE-4** (`llm/qwen35/src/eagle.rs`, opt-in: `NS_Q35_EAGLE=DIR`, or `auto` for an `eagle4/` folder beside the
  model): the EAGLE-4 draft for Qwen3.8-27B (`LlamaForCausalLMEagle4`, safetensors, kept as Q8_0) in MTP's place on
  one GPU. One Llama layer (40 heads of 128, a 4,096-key window) over `fc` of the residual stream after layers 3, 31
  and 59; its low-rank head's 128 candidates (`ns_q35_topk`) scored by the full head's rows. It drafts no better than
  MTP here (2026-10-23, B70, Q4, 512 greedy tokens, 3 drafts: code 57.6 tok/s with 81% of drafts kept against MTP's
  55.8 and 83%; prose 43.1 / 42.8 at 63%; a 14K prompt read 960 / 1,111 tok/s - its layer runs over every prompt
  row), and it does not draft past its cache (`NS_Q35_EAGLE_CTX`, 32K) or step conversations together, so MTP stays
  the default.
- **Two cards** (`--gpu all`, or `--gpu 1,0`: the B65, then the B70; the `-256k` catalog entries): the layers split
  so the last GPU (the server orders them weakest first) takes as many as fit beside the head, the draft block and
  the planned context's caches, and the one before it the rest - a GPU the layers do not reach is not used, so a
  context that fits the B70 runs there alone. Each GPU has its own work memory; a pass's rows cross once (through
  host memory, ~0.5 ms a token). `NS_SPLIT=N` puts layers N.. on the last GPU. The same tokens as one card
  (`spec-check --rows 4`: 0 difference). It is for room, not speed (the B65 runs a layer ~1.5x slower than the B70),
  prompt read / decode, tokens/s, MTP on:

  | File, context planned | 2,185 | 8,000 | 40,000 |
  |---|---:|---:|---:|
  | Q8_0, 256K + 32K (B65 layers 0-26) | 990 / 37.8 | | 588 / 29.0 |
  | UD-Q4_K_M, 256K + 32K (B65 layers 0-15) | 1,061 / 52.1 | 1,099 / 48.8 | 666 / 39.3 |

  Q8_0's 40K figure is from a split with layers 0-32 on the B65 (before the last GPU filled first).
- **Its silo** (`kernels/llm/qwen35/silo`, kernels tuned for this model): decode products for the types these files
  hold (Q4_K 507 GB/s against Strata's 366, IQ4_XS ~490 / 361, Q5_K 525 / 415, Q3_K 380 / 89, IQ3_S 439 / 118 - the
  same outputs), flash attention on XMX for prompts and, its keys split over work-groups, for decode, the
  weight expansion to half at ~300 GB/s for every type (Q6_K's went through float32 at 13-15), a norm that hands the
  products their Q8_1 blocks; prompt chunks of 2,048 when they fit. `NS_PROFILE=gpu NS_Q35_PROFILE=1` times every
  kernel on the device against the card's ceilings. Sessions check their cache against the free VRAM before
  allocating (1.5 GiB spare).
- **Pictures:** the model's `mmproj-F16.gguf` (Qwen3-VL's ViT + merger; the cards' vision file, found beside the
  model, or `NS_Q35_VISION`) reads pictures in chat - OpenAI `image_url` data URLs, Anthropic base64 parts, `/api/chat`
  `images` - and `llm generate --image`. A picture is up to 4,096 prompt positions (`NS_Q35_IMAGE_TOKENS`), ~0.6 s to
  see on the B70; its rotary positions are Qwen3-VL's (row and column), and the prompt cache holds across turns with
  pictures. Against llama.cpp's mtmd on the same file: the picture's rows cos 0.999998, the prompt's last logits as
  close as text alone (0.989 on Q4 against the CPU), the same greedy answer.
- **The attention cache's form** (`NS_Q35_KV`, a card at a time; `NS_Q35_KV_ROT=1` rotates a quantized one by a
  32-value Hadamard transform): half by default, q8_0 or q4_0 (llama.cpp's blocks) for room. Q4 on the B70 with MTP,
  measured with `llm generate --score` (256 tokens teacher-forced after a 12.8K prompt, against half; the noise floor -
  half read in other chunks - KL 0.00014):

  | NS_Q35_KV | KL mean (max) | top-1 the same | perplexity | one session's most context | 43K prompt read / decode |
  |---|---:|---:|---:|---:|---:|
  | f16 (default) | - | - | 2.688 | 196K | 742 / 43.0 tok/s |
  | q8 | 0.00024 (0.019) | 99.2% | 2.698 | 368K | 673 / 38.9 |
  | q8, rotated | 0.00018 (0.009) | 99.6% | 2.696 | 368K | |
  | q4 | 0.0032 (0.041) | 99.6% | 2.706 | 704K | 718 / 40.9 |
  | q4, rotated | 0.0023 (0.104) | 99.6% | 2.705 | 704K | |

  `auto` puts q8_0 on the card the plan fills first and half on the others (`auto-q4`: q4_0 there).
- **Two requests at once** decode together: each one's verify rows and drafts in the same passes (the same tokens as
  alone). Q4, 256-token answers: B70 one 47 tok/s, two 29 + 29; both cards (256K) one 44, two 26 + 26.

## Nemotron 3.5 Super (hybrid Mamba-2 / latent MoE)

`nemotronh` (`llm/nemotronh`, `kernels/llm/nemotronh`, its silo `kernels/llm/nemotronh/silo`): NVIDIA's
Nemotron-H MoE as llama.cpp's `nemotron_h_moe` graph has it - 88 layers: 40 Mamba-2 mixers (conv 4, 128 heads of 64,
8 groups, state 128, a SiLU-gated group norm), 40 latent MoE layers (a float32 sigmoid router with a selection bias,
22 of 512 experts, the experts in a 1,024-wide latent with ReLU-squared, a shared expert on the full width) and 8
attention layers without positions (32 / 2 heads of 128); its MTP block drafts. Served; text only (the vision tower not yet).

- **Checked** against llama.cpp (CPU, the same files, `llm check`): the same next token, the output logits cosine
  0.99998 (Q2_K_XL) / 0.99999 (IQ1_M); verify rows the one-token decode's (`spec-check`: 0 of 66 flips).
- **Decode**: the silo routes on the GPU and multiplies all of a layer's chosen experts in one launch per projection
  (the 2,688-wide downs repacked into aligned rows); the dense matrices (Q6_K Mamba projections, K-quant shared
  experts) through the silo's own decode products (the qwen35 silo's, copied, with a Q6_K kernel: ~520 GB/s on the
  B70), one quantize feeding both activation layouts. MTP drafts 2 a step (`NS_NH_DRAFTS`), the Mamba states
  snapshotted for the verify's rollback.
- **Prompts**: chunks of 2,048 rows when they fit; dense matrices expanded to half for the GEMM; the routed experts on
  XMX in one launch a projection (rows sorted by expert after a GPU routing pass; a work-group takes 64 rows of one
  expert, its lanes decode the quantized weights 128 inputs at a time into a half tile in local memory, each sub-group
  multiplies its 16 rows against all 64 columns; the up's output written as ReLU-squared half); the Mamba scan in
  64-row chunks (Mamba-2's chunked form: the in-chunk part as small products, the state carried over the chunks);
  flash attention on XMX; on two cards the layers split by measured speed (`NS_NH_SPEED`), the GPUs pipelined over
  the chunks.
- **One card**: `--gpu 0` keeps the MoE layers' experts that do not fit in pinned host memory and gathers the chosen
  ones into VRAM before each product (`NS_NH_HOST`).
- **Speed** (2026-10-15, 6913713 + the conv; `nemo-sweep`: benchy v1 prompts, 128 greedy tokens, MTP on), prompt
  read / decode, tokens/s:

  | File, GPUs | 20 | 2K | 8K | 40K | 128K |
  |---|---:|---:|---:|---:|---:|
  | UD-IQ1_M, B65 + B70 | 62 / 46.1 | 884 / 47.0 | 1,431 / 45.7 | 1,472 / 44.0 | 1,120 / 37.6 |
  | UD-Q2_K_XL, B65 + B70 | 54 / 41.7 | 789 / 39.6 | 1,141 / 41.9 | 1,134 / 38.5 | 868 / 34.3 |
  | UD-IQ1_M, B70 alone (host-memory experts) | 44 / 20.9 | 447 / 21.3 | 562 / 19.6 | | |

  Without drafts: Q2_K_XL 30.0, IQ1_M 31.8 tok/s decode. Strata on the same B70 reads ~880 (2K) / 1,400 (8K) and
  decodes ~30. On the B70 alone (`nemotron-3.5-super-iq1m-b70`, the B65 left free) 13 MoE layers' experts (15.5 GiB)
  sit in pinned host memory, gathered into VRAM each step; its prompt passes still expand experts one at a time. Open: the expert up products at decode (IQ3_XXS / IQ2_S / IQ1_S at ~150 GB/s on the B70, a third of the dense
  ones - neither the grid lookups nor the index math; `NS_NH_PROFILE=1` times every kernel), the vision tower.
