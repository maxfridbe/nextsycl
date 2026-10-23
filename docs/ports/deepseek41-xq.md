# DeepSeek-V4.1-Flash-1017 XQ v3 (IndrasMirror)

Cards: `deepseek-v4.1-flash-xq` (exact) and `deepseek-v4.1-flash-xq-fast`. The file is
[IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)
`v3/` (revision `15e786c`, MIT), 16 byte ranges of one 163 GB GGUF, made with the
[Xolotl](https://github.com/Indras-Mirror/Xolotl) tools from dealignai's UNCENSORED-FP8 (a rank-2 refusal-direction edit
of the attention output and the shared expert's down projection). It runs on the `deepseek41` engine beside the Unsloth
quants ([deepseek41-unsloth.md](deepseek41-unsloth.md)).

## The file

- arch `deepseek41` with Hugging Face config names (`num_hidden_layers`, `moe_intermediate_size`, ...): `model.rs`
  maps the llama.cpp names to them (`xq_name`).
- Each layer's routed experts are split by type into `blk.N.ffn_{gate,up,down}_exps.{q3_k,q2_0,ptq1_0}.weight`
  (experts in id order inside each), with I32 maps `ffn_exps_gu_type` / `ffn_exps_down_type` [384] of ggml type codes.
  Gate and up share a type, down has its own. 2.09 bits a weight on average, 142 GiB of experts:

  | type | values / bytes | value | gate+up | down |
  |---|---|---|---|---|
  | Q3_K | 256 / 110 | ggml's | 2,753 | 2,697 |
  | Q2_0 (ggml 42) | 64 / 18 | d (q - 1) | 5,665 | 7,060 |
  | PTQ1_0 (143) | 128 / 28 | d (t - 1), ternary, 5 trits a byte | 6,942 | 5,603 |

- Attention, the shared expert and the head are Q8_0, the hyper-connection and Engram projections F16.
- The Engram tables are not in the file: DeepSeek's own two shards (`deepseek-engram/`, the VQ card's) beside it; the
  file has `engram_kv` and the q / k norms (multiplied at load, as for the VQ file).

## The engine

- An expert's slot holds its own three matrices back to back (`LayerVq::lay`), in a store of its size: 8 stores by
  type pair (7.4-14.5 MiB) instead of slots of the layer's widest (195.6 GiB of slots for 142.4 GiB of experts). With
  a profile, the stores get the slots their experts take in its ranking over all of them.
- Kernels (silo `deepseek41`): Q2_0 / PTQ1_0 expansion (std.cpp, against the host's to 0); the float32 mixed product
  (codes 1000 gate | up, 1001 down) for prompt passes, decode's split-Q8 dp4a one (mmvq.cpp `Mixed`). Each entry
  carries its type in bits 8-15 of its first meta word: a tag in the pointer's top byte failed, the B70's device
  addresses use those bits. PTQ1_0's trits come four at a time (two bytes a 16-bit-lane multiply). `nextsycl llm
  kernels FILE` checks every type: float32 ~1e-7, split-Q8 3.7-3.8e-3.
- GPU routing gets each expert's type and up offset from a [layer][expert] table.
- `-fast` (`NS_DS41_DENSE_Q4K`) also requantizes the F16 Engram projection to q4_K (as float32 it was 630 MB read a
  token at each of the two Engram layers).

## Measured (2026-10-23, both GPUs)

| | XQ v3 | UD-IQ2_XXS | UD-Q2_K_XL |
|---|---|---|---|
| tg, 512 tokens after a short prompt, exact / `-fast` | 10.7 / 21.6-22.4 tok/s | 12.7 / 21.4 | 8.8 / 19.2 |
| pp exact, 580 / 2,561 / 9,901 tokens | 78 / 162 / 339 tok/s | - | 42 / 92 / 280 |
| pp `-fast`, 580 / 2,561 / 9,901 tokens | 123 / 225 / 410 tok/s | - | 121 / 138 / 300 |

Prompts are bound by the experts read from the drive (only ~40% of the 142 GiB fit VRAM and pinned memory; a 2,561-token
prompt reads ~7,400 of them, ~70 GB). The file sits on the box's fastest drive (SK Hynix PC801-class, 7.3 GB/s O_DIRECT
at 16 x 8 MiB; the 4 TB Crucial P310 (QLC, DRAM-less) gave 4.8, the 2 TB Phison E13T (Gen3, DRAM-less) 1.2). How pp got
here (exact, 2,561 / 9,901 tokens): 115 / 269 from the P310; 72-154 / 436 from the fast drive (bimodal: the prompt's
reads went through the page cache, and its reclaim beside ~45 GiB of pinned memory stalled whole layers for seconds);
162 / 339 with them O_DIRECT (2026-10-23 evening, after f3221ce). Earlier numbers that day (154-193 / 514-525) were
inflated by a bug: from 4,096-token passes (2,048 for a few hours) a layer's own file reads overwrote the experts the
layer before had read ahead for it, so long prompts read less and computed with wrong weights (a 5K-token chat looped
on nonsense; a 2,561-token prompt's logits NaN). Fixed in f3221ce, checked against 8-token passes (KL 0.0325).
| perplexity, 700 tokens of Gutenberg prose (gut1 / gut2), exact | 1.281 / 1.375 | 1.228 / - | 1.211 / 1.281 |

The quant is a little worse than UD-IQ2_XXS on human text (Xolotl's own numbers put it ~15% above the FP8 original).
How tg got there (512 tokens, exact / `-fast`): the float32 mixed kernel 6.75 / 11.0; split-Q8 products 8.2 / 15.7;
stores by expert size 9.35 / 16.3; q4_K Engram projection and PTQ1_0 four trits at a time 10.1 / 20.6; the mixed
product on the large register file (`grf_size<256>`: the one kernel with three loaders spilled) and Q3_K's scales in two
loads - / 21.8-22.4 (runs of the same build differ by ~0.5).

Tried and left out: a launch a type (each type ~2x faster alone, but a 30-60 us floor a launch with a token's 1-3
experts of it: no faster); the rows staged through local memory (slower: the loaders' arithmetic, not the loads, bound
it); 2-byte loads for Q3_K (2-3x slower); q_b on the split-Q8 kernel (70 against 106 us alone, no faster end to end);
the file's Q8_0 attention instead of q4_K (faster a product cold, but twice the VRAM: 17-20 against 22 tok/s).

## Open

- The vision sidecar is untested with this file (the cards set `NS_DS41_VISION=0`).
- The MTP / DSpark draft sidecar (`V41-MTP-Vision-1017.gguf`) is not read.
