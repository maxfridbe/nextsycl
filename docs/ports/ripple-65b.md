# Ripple-65B-Preview (llm/ricursive)

Ricursive's Ripple-65B-Preview (huggingface.co/ricursiveAI, arch `ricursive`, dense, Apache-2.0), run from AtomicChat's
AD-Q4_K_M GGUF (`AtomicChat/Ripple-65B-Preview-GGUF` @ df550a71, `Ripple-65B-Preview-AD-Q4_K_M.gguf`, 40,426,079,712
bytes, sha256 9016e260...ca7bfc7, checked after the download). The references are llama.cpp's `src/models/ricursive.cpp`
(the PR's patch) and `modeling_ricursive.py`, both read, neither run. Card: `model-cards/llm/ripple-65b-ad-q4km.yml`
(`ripple-65b-ad-q4km`). Engine: `llm/ricursive`; kernels `kernels/llm/ricursive` (the engine's) and
`kernels/llm/ricursive/silo` (tuned); reference `reference/ricursive-65b`.

## The model (read from the file and the modeling code)

- 80 dense blocks of 5,120; 80 query heads over 8 key heads of 128 (10 a key head); a SwiGLU of 40,960 whose gate and up
  are clamped at 20 (`min(gate, 20)` and `clamp(up, -20, 20)`, then `silu(gate) * up`); embeddings scaled by sqrt(5120);
  vocabulary 100,352 (the cl100k pre-tokenizer: llama.cpp's `dbrx`, added to `nextsycl_tok`); RMS eps 1e-6.
- Four plain RMS norms a block (before and after the attention, before and after the SwiGLU: the post norms sit before
  the residual add); each head's q and k normed (the weight per feature, shared by the heads).
- An elementwise sigmoid gate on the attention's output, from its own projection (`attn_gate`, from the normed input).
- A value residual: the value projection reads `x0 + sigmoid(lambda) (h - x0)`, x0 the scaled token embeddings (one
  lambda a block, `attn_vres_lambda`).
- Four layers of five (64 of 80) attend a 1,024-key sliding window (the query and itself included) with rotary
  positions (NEOX pairs, theta 1e4, all 128 features); every fifth layer (4, 9, ... : 16) attends everything and has no
  positions at all (NoPE). The context is 2,097,152 in the metadata.
- Tensor types of the AD file (a "dynamic" quant: the norms and gate-side tensors are exact, the feed-forwards mixed):

  | tensor | type | note |
  |---|---|---|
  | `token_embd` | IQ4_XS | read row by row (a pass needs its tokens' rows only), dequantized on the host |
  | `attn_q`, `attn_gate` | Q4_K | 80 layers each |
  | `attn_k`, `attn_v` | Q8_0 | 80 layers each |
  | `attn_output` | Q6_K | 80 layers |
  | `ffn_gate`, `ffn_up`, `ffn_down` | Q4_K in 50 layers, Q5_K in 30 | |
  | `output` | Q6_K | the head, 100,352 rows |
  | all norms, the q / k norms, `attn_vres_lambda` | F32 | |

  37.65 GiB of tensors. The AD file's card quotes mean KLD 0.0136 against BF16 and 95.9% the same top token.

## What runs where

- **Engine** (`llm/ricursive`): the contract of `llm/contract` complete - `Engine`, sessions, checkpoints (save, restore,
  read; fingerprint `ricursive f16-kv nsrc-1 w1024`: rule 1, prompt caching, covered by `architecture.rs` once registered),
  rollback, `copy_session`. No draft, no MTP (`max_verify` 1). The chat template (`chat.rs`) is the model's jinja, written out:
  `<|im_start|>role\n...<|im_end|>`, tools as `<tools>` JSON and calls as `<tool_call>` JSON objects (the parser is
  `nextsycl_tok::json_tool_calls`), `<think>\n...\n</think>` reasoning kept in earlier assistant turns when present.
- **Kernels** (`kernels/llm/ricursive/rc.cpp`, `ns_rc_*`): q / k norm with the rotation (sliding layers only), the ring
  store, the attention (the qwen35 engine kernel's template: heads of 128, 10 query heads a key head, window and ring),
  the output gate, the scale, the value residual, the post norm into the residual. Everything else is the shared
  kernels (RMS norm, the Q8_1 products for Q6_K / Q8_0, the half GEMM, SwiGLU).
- **Silo** (`kernels/llm/ricursive/silo`, `dist/silo/libnextsycl-ricursive.so`; the shared kernels run it too, slower):
  - `mmvq.cpp`: the decode products for Q4_K and Q5_K (a copy of the qwen35 silo's);
  - `dequant.cpp`: the weight expansion to half for prompt GEMMs (Q4_K, Q5_K, Q6_K, Q8_0, ...), a copy of qwen35's;
  - `norm.cpp`: RMS norm with the Q8_1 blocks of its output, qwen35's;
  - `attn.cpp`: flash attention on XMX for prompt rows and (split over the keys) decode rows, a copy of the museglimmer
    silo's changed to 10 query heads a key head (a work-group is 8 rows and the 10 heads: 20 sub-groups).
- **Layers over the cards.** The last card given is filled first, the first card takes the rest, with 1.5 GiB of VRAM
  left free on each (`NS_SPLIT` overrides; `--gpu 1,0` is B65 then B70, so the B70 holds more: the default).
  A sliding layer's cache is a ring of window + chunk positions, a global layer's the whole context (4,096 bytes a
  position a layer). A layer that fits no card stays in pinned host memory and its products read it **in place over
  PCIe** (default) or through a staging copy in VRAM (`NS_RC_STREAM=copy`); host memory stays at least 8 GiB free.
  There is no disk tier (a dense layer has no hot subset: every layer is read once a token, so a drive is slower than
  pinned memory by 10x or more and was not built).

## Exactness

Rule 5: a hand-written float32 CPU reference (numpy; `reference/ricursive-65b/ripple_ref.py`) first, no PyTorch, no
`trust_remote_code`, none of the downloaded code run. Its two weight backends:

- `--gguf`: the file's tensors dequantized by ggml's formulas (Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, F32), float32 math - what
  an engine running the file must reproduce: the check of the kernels;
- `--st`: the BF16 safetensors of the model itself, fetched by HTTP range for the tensors a check needs
  (`fetch_tensors.py`, the pinned revision; 3 layers and the head are 6.4 GB, not the 16 shards).

`compare.py` takes the engine's `NS_RC_DUMP=DIR` (every layer's output, float32) and a reference dump.

| check | against | result |
|---|---|---|
| 3 layers, 27 tokens, prompt path (half GEMM, XMX attention) | GGUF reference | every layer cosine 0.9999986 or better (worst row 0.9999975), rel RMS 0.0008-0.0018, final norm 0.9999994 |
| 5 layers (4 sliding + the first global), 1,343 tokens (the window wrapped), XMX attention | GGUF reference | worst row of any layer 0.9999965; last-row logits: same top token (89888), KL 0.000166 |
| the same with the shared attention (`NS_RC_FLASH=0`) | GGUF reference | worst row 0.9999965, logits KL 0.000169 (the XMX kernel is as close as the vector one) |
| 3 layers, the engine | BF16 reference | cosine 0.9955-0.9964 at the embedding and final norm, 0.9886-0.9933 at layers 0-2 (rel RMS 0.08-0.16) |
| 3 layers, the GGUF reference | BF16 reference | the same numbers to four digits: all of the engine's distance from BF16 is the file's quantization (the IQ4_XS embedding alone is 0.0843 rel RMS); the logits KL of the truncated stack 0.024, same top token |
| the whole model, XMX attention vs the shared kernel, 8,454-token prompt, 256 teacher-forced tokens | each other | KL mean 0.00099 (max 0.17 at one near tie), top token the same 99.2%, perplexity 26.93 vs 26.80 |

The truncated models are checked, not the whole 80 layers against the CPU (a 65B in float32 on the CPU is hours a
token; the box has 61 GiB of RAM). The whole-model numbers the model card quotes for the file (mean KLD 0.0136, 95.9%) are
the maker's; this port did not reproduce them (no whole BF16 model was fetched and no llama.cpp run was made). What the
whole-model run shows: greedy text is coherent (answers a question, thinks in `<think>`), and the two attention kernels
agree to KL 0.001. Decoding against the prompt path was not compared (a gap).

## Speed (B65 + B70, `llm generate --ids`, 64 tokens `--ignore-eos`, greedy, 2026-11-08; pp = prompt, tg = generation)

Layers 0-20 on the B65, 21-79 on the B70 (the default), 37.4 GiB of weights, the context 32,768.

| prompt | pp, shared attention (first engine) | pp, XMX attention (silo) | tg, XMX attention |
|---|---:|---:|---:|
| 2,000 tokens | 188 tok/s | **531 tok/s** | **10.5 tok/s** |
| 8,000 | 122 | **514** | 9.8 |
| 31,000 | 57 | **409** | 9.4 |
| 17,098 (this repository's `architecture.md`, real text) | 85 | **442** | 9.3 |

tg with the first engine was 9.8 (2K) and 7.45 (31K): the XMX attention is most of the gain at depth.

What bounds them:

- **tg** reads 40.4 GB a token across two cards in series: the B70 holds 59 layers (~29 GB), the B65 21 (~10 GB). The
  decode products run at 450-500 GB/s (Q4_K 497, Q5_K 455 GB/s measured per call with `NS_RC_PROFILE=1`; Q6_K 320,
  Q8_0 295), so the memory floor is ~75 ms (13 tok/s) and the engine takes ~96 ms. Profile of a decode pass, 2K context
  (milliseconds, a sync after each operation so a little inflated): Q4_K products 45, Q5_K 29, Q6_K 11, Q8_0 3, the
  small kernels between them 13 (about 640 groups of them a pass, 20 us each as timed), attention with its norms and cache store 11.
  The remaining gap is those small kernels and the Q6_K / Q8_0 products (the shared kernel, not yet in the silo).
- **pp** is compute: the half GEMM after expanding each matrix (the expansion reads the 40 GB again per chunk of 2,048
  rows: 2K tokens in 3.8 s) and, at depth, the 16 global layers' attention; 409 tok/s at 31K.
- **Split.** The B70 first is the fastest of the splits tried (tg at 2K: first layer of the B70 at 21: 10.6; at 30: 10.2;
  at 40: 9.9; at 12, where 9 layers no longer fit its VRAM and stream over PCIe: 3.4).

### Offload (one card, the rest in pinned host memory, 2026-11-08)

The B65 alone (the B70 was held by the front door's other models while this was measured): 20 layers (9.8 GiB) in pinned
host memory, 60 on the card, chunk 512, context 8,704 so that the caches fit.

| stream | pp 2,000 | pp 8,000 | tg |
|---|---:|---:|---:|
| in place over PCIe (default) | 343 tok/s | 296 tok/s | 1.9 tok/s |
| through a staging copy (context 2,304: the staging buffer needs its VRAM) | 385 tok/s | not enough VRAM | 2.0 tok/s |

- Streaming is bound by PCIe: 9.8 GiB a token at ~21 GB/s is 0.5 s; adding the 60 resident layers' 63 ms gives the 1.9
  measured. Direct and copy are the same speed; copy costs a layer of VRAM, so direct is the default.
- Prompts are far less hurt (a chunk of 512 reads the host layers once, 296-343 tok/s against 531): a prompt-heavy,
  one-card use works, a chat does not. **Two cards are the answer for chat: 10 tok/s against 1.9.**
- `MemAvailable` stayed above 40 GiB through these (the 8 GiB guard is the engine's: it refuses to pin past it).

## Memory

See the card's `vram_gib` (the maximum over the two cards, from the engine's own log line, "card memory in use before any
session", plus the caches at 32K: 16 global layers x 32,768 x 4,096 B = 2.0 GiB and 64 rings of 3,072 positions =
0.77 GiB). MEMORY_TABLE

## Known, open

- Q6_K and Q8_0 products are the shared kernels (320 and 295 GB/s against the silo's ~500): ~4 ms of 96.
- ~2,000 small launches a token (norm, quantize, rotation, store, gate, post norm ...); fusing them, or a graph, is the
  next ~10 ms.
- No draft: Ripple ships none; the engine verifies one token a pass.
- The plain `Ripple-65B-Preview-Q4_K_M.gguf` was not downloaded; the engine reads any file whose tensors are among the
  types above (IQ4_XS embeddings are the only one the shared dequantizer lacks, handled on the host).
- A disk tier was not built: see above.
- Not measured: the B70 alone with offload (its VRAM was busy), 31K on one card.
