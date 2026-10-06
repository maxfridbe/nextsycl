# Imported kernels

The SYCL kernels of the Strata Intel port, copied unchanged from Strata_B70 branch `up139-fix` at
`79ad9d5aff716292602895b19860c7661bacc5ed` (2026-10-05; upstream PR #809, branch `intel-arc-0.1.39`). Only kernel
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

Changes to these files are made in `kernels/ns/` (new files or wrappers), so this directory stays a clean copy
that can be refreshed from the branch.
