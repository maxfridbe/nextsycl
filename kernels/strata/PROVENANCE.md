# Imported kernels

The SYCL kernels of the Strata Intel port, copied unchanged from Strata_B70 branch `intel-arc-0.1.40` at
`3f37281` (2026-10-06, the port refreshed to upstream 0.1.40; first imported from `up139-fix` 79ad9d5 and refreshed by a
3-way merge, base = the previous import). Only kernel
code: Strata's host runtime (loader, sessions, expert source, scheduler, server) is not here - nextsycl's is in
Rust (`crates/`).

| here | from |
|---|---|
| `src/kernels/*.dp.cpp` | `sycl/src/kernels/cuda/*.dp.cpp` |
| `src/prefill/{gemm,kernels}.dp.cpp` | `sycl/src/prefill/` (not `prefill.cpp`, the host orchestration; not the MMQ path `moe_mmq` / `ggml_cuda_host`, which needs ggml-cuda's `common.cuh` and measured 6-10x slower) |
| `include-sycl/` (dpct helpers, `strata/sycl_*.hpp`, `strata/kernels/`, `strata/core/coupled_draft.hpp`) | `sycl/include/` minus the rest of `strata/core/` (host runtime; `coupled_draft.hpp` is the sampler's contract) |
| `include/strata/{kernels,prefill}`, `include/strata/core/emulate.hpp` | `include/` |
| `third_party/ggml/` | `third_party/ggml/` (the root copy, as Strata's kernel target uses; `sycl/third_party/ggml` is a dpct-migrated variant whose tables do not compile here) |

Include order as in Strata's CMake: `include-sycl/` before `include/` (the migrated headers win).

Licences: Strata is MIT (`LICENSE` here, copyright Niko1221 and the Strata contributors); `third_party/ggml` is
MIT (its own `LICENSE`); the `dpct/` helpers are Intel's, Apache-2.0 WITH LLVM-exception (their headers).

New code goes in `kernels/ns/`. The changes made here, each marked `nextsycl` in the file:

- `src/kernels/iq_kernels.dp.cpp`: Q2_K and Q3_K dot products (from ggml-sycl's `vecdotq.hpp`, llama.cpp
  de25343, in this file's (row, kbx) form) and their `Fmt` traits; Q2_K / Q3_K added to the gate-up, down and
  matvec format lists, Q4_K / Q5_K to the down list; Q2_K in `is_iq` / `iq_row_bytes` - GLM-5.3's expert files.
  The grouped experts' SwiGLU takes an optional clamp (`native_expert_set_swiglu_limit`, 0 = none as before;
  GLM-5.3's `silu(min(g, 10)) * clamp(u, -10, 10)`).
- `iq_kernels.dp.cpp`: multi-entry dots for IQ2_XXS (`Multi<16>`) and Q2_K (`Multi<10>`, a `custom` per-token dot),
  and 8-entry passes in the grouped expert kernels: GLM-5.3's expert formats decoded once per 8 tokens of a prompt
  chunk instead of once per token. Same arithmetic as the single-entry dots.
- Since the 0.1.40 refresh: `native_expert_set_phase` sets upstream's `g_exp_phase` (its bench switch, the same
  meaning), and the opt-in S26 grouped path (`STRATA_EXPERT_V2`, Qwen shapes only) runs the SwiGLU unclamped.
