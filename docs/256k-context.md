# 256K context: the plan

Today the server holds two sessions of 64K tokens. GLM-5.3-Flash is trained to 1,048,576 tokens (`context_length`
in the file) and its MLA has no RoPE (`rope_dimension_count 0`), so 256K needs no position scaling - the work is
memory, the indexer's quadratic part, the prompt cache, and checking it at depth.

## What grows with the context

Only 11 of the 45 layers are MLA (the other 34 are KDA, whose state is fixed), plus the draft block's attention.
Per token, per session, each of those 12 keeps:

| | bytes | at 256K (262,144 tokens), one session |
|---|---:|---:|
| MLA latents, fp16 [512] | 1,024 | 3.0 GiB |
| indexer pooled keys, fp32 [128] per 4 tokens | 128 | 0.4 GiB |
| **in all** | **13,824 (13.5 KB)** | **3.4 GiB** (B65 1.4, B70 2.0 with the draft block) |

Today's two 64K sessions hold 1.7 GiB. Every GiB more is ~150 experts less in VRAM (6.8 MiB each), and decode is
bound by the misses' PCIe copies.

| Sessions | VRAM for attention | Experts out of VRAM vs today |
|---|---:|---:|
| 2 x 64K (today) | 1.7 GiB | - |
| 256K + 32K | 3.8 GiB | ~315 of ~6,500 |
| 2 x 256K | 6.8 GiB | ~760 |
| 2 x 256K, q8 latents + fp16 keys | 3.6 GiB | ~280 |
| 2 x 256K, 4-bit latents + fp16 keys | 2.1 GiB | ~60 |

KDA's state per session is 142 MB however long the context.

## Measured scaling of the prompt (B65's stage, 2026-10-07)

36.6K -> 62.6K tokens (x1.71): routed experts x1.67, KDA x1.72, MLA attention x1.71 (sparse: each query attends to
its 2,048 selected tokens), the indexer x2.51 (it scores every earlier pool). The indexer fits 0.0147 s/K tokens +
0.00078 s/K^2: 1.6 s at 36K, 4.0 at 62K, **~57 s at 256K** (B65; the B70 has 7 of the 12 attention layers, so its
stage grows faster - the pipeline's balance moves at long context). Everything else is linear: a 256K prompt should
read in ~300 s (~850 tokens/s) as things stand; 62.6K reads at 1,001 tokens/s today.

The indexer's cost is its layout: per head scores for every (token, pool) - 4,096 x 32 heads x 64K pools at 256K,
34 GB per chunk and layer, scored in 64 MiB pieces - then summed over the heads and top-k'd.

At decode the indexer reads each layer's pooled keys once a token (32 MB a layer at 256K: ~0.4 GB a token over both
cards, under a millisecond each) - decode should barely slow with depth; to be measured.

## Measured (2026-10-07)

- 128K, generate: the prompt at 998 tok/s (126 s); decode at that depth 13.0 tok/s at first - the indexer's top-k
  and MLA's decode attention were slow at depth - now 17.1.
- 256K, through the server (`--ctx 262144,32768`): 253,308 tokens in 271 s (933 tok/s, 29.6 Wh), a right answer,
  decode at that depth 14.5-15.6 tok/s; the B65's stage 4.3 s a chunk at the start, 4.6 at the end (the indexer's
  quadratic part far below the fit above); a follow-up on the document 4.4 s (the checkpoint, 3.5 GiB, restored).

## Steps

1. **Fit and measure, no new kernels.** Per-slot context (`--ctx 262144,32768`: one long session, one short,
   instead of equal ones). Run 128K, then 256K in `generate` with the VRAM computed first (the attention reserve is
   in the expert budget; keep >= 1.5 GiB free - the box's known hang is the driver's VRAM spill path, and a
   262K-context llama.cpp run hit it once). Record TTFT, decode speed at depth (16K / 64K / 128K / 256K), VRAM.
2. **Correctness at depth.** A needle-in-a-haystack check (`nextsycl bench --needle`: a fact placed at 10 / 50 /
   90% of 32K-256K of filler, greedy, the answer must carry it). llama.cpp parity stops being practical here
   (its 262K run is what hung the box), so: the needle, logits of a long prompt against the same prompt read in
   other chunk sizes, and an int32 audit of the attention and indexer kernels (positions are int32 cells today -
   fine to 2^31 - but every `t x n` product must be 64-bit).
3. **Smaller cache entries** (what you asked about): `NS_KV=f16|q8|q4`. Latents in q8_0 (544 B a layer-token,
   half) or a 4-bit form (288 B), pooled keys in fp16 (DeepSeek's DSA keeps its indexer keys in fp8; Xe2 has no fp8
   matrix math, but fp8/int8 as storage widened on read works). Each query reads only its 2,048 selected cells, so
   decoding them on gather is cheap. Quality: logits against fp16 on long prompts, the needle at 256K, MTP's
   acceptance rate (a sensitive sign of drift).
4. **The indexer without the per-head scores.** One kernel per row block: q . pooled keys on the XMX units (fp16
   keys), the heads' weighted sum in registers, the top-k over the row's pools - never writing T x H x pools.
   Target: the quadratic part from ~57 s to a few at 256K.
5. **Prompt cache for long prompts.** A 256K checkpoint is ~3.5 GiB (latents + pooled keys + KDA states); the 4 GiB
   cache holds one, and host RAM is the expert mirror's (it takes what is free at start less 10 GiB). Keep long
   checkpoints on the NVMe instead: ~3.5 GB written or read in ~1 s, against 5 minutes to re-read the prompt.
6. **Serve it.** `NS_CTX` per slot in nextsycl.conf and the chat mode; check the HTTP body limit (a 256K-token prompt
   is ~1 MB of text) and the tokenizer's speed on it; the client's context setting (Open WebUI's) to match; the
   default max_tokens is already the rest of the context.

Order: 1 -> 2 (know it works and what it costs) -> 3 (memory) -> 4 (speed) -> 5 -> 6. The bigger prompt chunks of
TODO.md (6144: +11%) wait for the pass buffers in the VRAM budget - more pressing once the attention reserve grows.
