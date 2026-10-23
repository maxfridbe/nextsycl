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
| tg, 512 tokens after a short prompt, exact / `-fast` | 10.1 / 20.7 tok/s | 12.7 / 21.4 | 8.8 / 19.2 |
| pp exact, 580 / 2,561 / 9,901 tokens | 67 / 99 / 254 tok/s | - | 42 / 92 / 280 |
| pp `-fast`, 2,561 / 9,901 tokens | 187 / 257 tok/s | - | 138 / 300 |
| perplexity, 700 tokens of Gutenberg prose (gut1 / gut2), exact | 1.281 / 1.375 | 1.228 / - | 1.211 / 1.281 |

The quant is a little worse than UD-IQ2_XXS on human text (Xolotl's own numbers put it ~15% above the FP8 original).
How tg got there (512 tokens, exact / `-fast`): the float32 mixed kernel 6.75 / 11.0; split-Q8 products 8.2 / 15.7;
stores by expert size 9.35 / 16.3; q4_K Engram projection and PTQ1_0 four trits at a time 10.1 / 20.6. `-fast` decode
GPU time a token: attention 16.4 ms (as UD-IQ2_XXS), expert products ~4.6 ms (3.4), Engram 0.6 ms.

## Open

- The vision sidecar is untested with this file (the cards set `NS_DS41_VISION=0`).
- The MTP / DSpark draft sidecar (`V41-MTP-Vision-1017.gguf`) is not read.
