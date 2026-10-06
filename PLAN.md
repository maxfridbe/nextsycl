# nextsycl: plan

A Rust runtime for large hybrid-attention MoE models on Intel Arc GPUs (SYCL), with the experts that do not fit the
cards streamed from host RAM and NVMe. First model: **GLM-5.3-Flash** (`glm5-next`). Second: the **qwen4exp** family
Strata serves today (Qwen3.8-Flash-Next, Coder, Swift), so the box can switch between them in one system.

Not a fork of Strata. The SYCL kernels of the Strata port (`intel-arc-0.1.39` / `up139-fix`, MIT) come in as a
kernel library; every host-side decision - loader, memory plan, expert cache, scheduler, server - is made here,
for these models and this box.

## Why its own runtime

Measured 2026-10-05 on the test box (B70 + B65, 61 GB RAM, Gen4 NVMe) with llama.cpp de25343 (PR 27773) and the GCSA
GSQ-RCO 3.5-bit file (137 GB):

| | llama.cpp SYCL, experts of 16 layers in VRAM, the rest mmapped |
|---|---|
| decode | 4.7 tok/s warm |
| prompt | 2-4 tok/s |

Prompts are the problem: llama.cpp runs the RAM-side experts on the CPU, and a batch of tokens touches nearly all
288 experts of every such layer (~80 GB, more than the page cache). The runtime here does what Strata does
instead - every expert computes on a GPU; experts not resident are DMA'd from pinned RAM or read from NVMe once
per prompt chunk and grouped by expert - and plans memory for two cards and this host from the start.

## Layout

```text
kernels/strata/   the Strata SYCL port's kernels, imported unchanged (PROVENANCE.md, LICENSE)
kernels/ns/       this project's kernels and the C ABI the Rust side calls (ns.h): new layer types, wrappers
crates/ns-sys     the C ABI, loaded at run time (dlopen, like the H3 engine's h3-sys)
crates/ns-gguf    GGUF v3: every tensor type the files use, split files, mmap; no ggml
crates/ns-core    devices, buffers, the memory plan, expert store (VRAM cache / pinned RAM / NVMe), scheduler
crates/ns-model   the model trait and the architectures: glm5next, then qwen4exp
crates/nextsycl   the command line and the OpenAI-compatible server
reference/        Python only: dumps from llama.cpp (and ds4) for parity checks
```

Language rule (as in the H3 engine): non-kernel code in Rust; kernels in SYCL C++; Python only in `reference/`.

## GLM-5.3-Flash, as the files describe it

45 layers: 3 dense (FFN 12288), 42 MoE (288 experts of 2048, top-8, sigmoid gate with a bias, normalized, x2.5,
plus one shared expert; SwiGLU clamped at 10). Hidden 4096, vocab 154,880. Every 4th layer (3, 7, ... 43) is full
attention: MLA (q LoRA 1536, kv LoRA 512, absorbed `k_b` / `v_b`, 64 heads) with a DSA lightning indexer (32 heads
x 128, top-2048, a 4-way key compressor). The others are KDA (Kimi delta attention): per-channel decay from a
low-rank gate (`ssm_f_a/b`), q/k/v short convolutions (kernel 4), an output gate (`ssm_g_a/b`), 64 heads x 128.
Hyper-connections around every block: 4 streams, `hc_*_fn` [24, 16384] (pre 4 + post 4 + residual 4x4), the 4x4
made doubly stochastic by 20 Sinkhorn iterations. Native MTP (absent from the GCSA file, present in the ds4 IQ2).

## Files

| file | size | types | reference runtime | role |
|---|---:|---|---|---|
| GCSA uncensored GSQ-RCO 3.5-bit (`~/models/glm53`) | 137 GB | Q2_K / Q3_K / Q4_K / Q8_0 / BF16 | llama.cpp (on the box) | bring-up and parity |
| DogContext uncensored IQ2 + MTP, ds4 (`~/models/glm53-iq2`) | 96.5 GB | IQ2_XXS / Q2_K, imatrix | antirez ds4 | the first served model |
| later: IQ3 (NOTES.md) | ~118-125 GB | IQ3_XXS / IQ3_S | | quality comparison |

## Phases

Each phase ends with a parity check against the reference (cosine per tensor, then logits and greedy tokens).

1. **Load and describe.** `nextsycl info <gguf>`: the geometry from metadata, every tensor named and its shape
   checked against the layer types above, the bytes by type and by placement. Both files.
2. **Kernel library.** The imported kernels built as `libnextsycl.so` with a C ABI; per-device SYCL contexts from
   the first line (a shared multi-GPU context makes xe mirror device memory into host RAM - measured 45 GB).
3. **Dense path.** Embedding, mHC, the 3 dense layers, KDA, MLA + indexer, final norm and head - against
   llama.cpp dumps of the GCSA file, layer by layer.
4. **Experts.** Router, shared expert, K-quant and IQ2_XXS decode matvecs and grouped prompt GEMMs; the expert
   store: resident sets per card, a VRAM cache, pinned RAM, NVMe reads; grouped per prompt chunk.
5. **Two cards.** Layer placement by memory plan; activations between cards through host memory.
6. **Serve.** OpenAI-compatible server (chat template with reasoning effort), the box's model switch.
7. **MTP** speculative decoding (the ds4 file's draft head). Done 2026-10-06: one draft per cycle, 87% accepted on
   code, 13.1 -> 15.7 tok/s greedy; verify rows bit-identical to one-token decode. Next: the swaps a 2-row pass adds
   (32 ms/token in the profile), deeper drafts.
8. **qwen4exp.** The Strata models through the same runtime; `--model` chooses.
