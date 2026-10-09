---
name: sycl-porting
description: Port GPU inference code to SYCL on Intel Arc (Xe2 - B580, B70, B65) - from CUDA (SYCLomatic), from Vulkan/compute shaders, or from a PyTorch / diffusers reference - with host code in Rust. Covers the safety rules that keep the machine up, the migration traps, the parity-first test method (stage dumps, teacher forcing, bit-exact checks), profiling (device timestamps, flame graphs, bandwidth math) and the techniques that measurably cut time and raised throughput. Use when porting a model or kernel to SYCL / oneAPI, when a SYCL port is wrong, hangs or wedges the GPU, or when tuning SYCL kernels on Arc.
---

# Porting to SYCL on Intel Arc

Lessons from porting, end to end, a CUDA LLM engine (dpct + hand fixes, then kept in step with upstream), two MoE
language models, a text-to-image DiT, a video+audio diffusion model and a music model (an 8B autoregressive model +
a flow-matching transformer + a vocoder) to SYCL with Rust host code. Every number below was measured on Arc Pro
B70 / B65 (Xe2, BMG-G31, 32 GB). The order of the sections is the order to work in.

## 1. Safety first: what takes an Arc box down

Read this before the first kernel runs. Each item cost a reboot or a wedged card at least once.

- **There is no GPU out-of-memory error.** An allocation past VRAM is evicted by the xe driver into host RAM
  (TTM). Those pages are unswappable and charged to no process: the OOM killer kills bystanders and the host
  livelocks. Budget memory *before* allocating (weights + caches + activations + work buffers), compare with the
  driver's free memory, keep >= 1.5 GB free, and refuse with a message otherwise. Test every VRAM-size change through
  the real server (which holds more), not only a one-shot run.
- **One SYCL context per GPU.** The platform's default context spans all GPUs and makes xe mirror each card's
  allocations into host RAM (measured: ~45 GB of host RAM for 40 GB of model on two cards). Open each device with
  `sycl::context(dev)`; copies between GPUs go through host memory.
- **Unbounded device spins wedge the GT.** A kernel spinning on a flag that never comes (its producer was killed)
  cannot be preempted; the driver resets thousands of graph nodes one by one and the card may need a reboot. Bound
  every spin, and never kill an engine mid-kernel - stop it through its own exit path.
- **A queue copy with no device end hangs the copy engine** (pinned host <-> pageable host via `q.memcpy`). Do
  host-only copies with `std::memcpy` (check `sycl::get_pointer_type`).
- **Queued copies are asynchronous.** The host source must outlive the copy: free a `std::vector` after queuing
  copies from it and the GPU faults.
- **Host waits on another queue's events can deadlock** under the Level Zero v2 adapter (a stager thread waiting on
  copy-queue events while another queue's barrier waits on the same event). Have the copy queue `fill()` a sequence
  number into host USM and poll it instead.
- **oneDNN's graph-API attention silently falls back** to materializing the whole S x S score table when it cannot
  use the fused kernel (30 GiB at 16.5k tokens - a spill, a reboot). Check `ONEDNN_VERBOSE=1` at <= 2k tokens that
  the fused implementation runs, step sizes up one at a time, and prefer kernels whose memory is bounded by
  construction (or patch oneDNN to error instead of falling back).
- **After every GPU test**, check the kernel log before the next: `journalctl -k -b | grep -cE "Engine reset|wedged"`.
  A count that moved means stop and find out why.
- Never overlap a big AOT build (icpx ~2 GB a job) with a GPU run on a small-RAM host; build with fewer jobs.

## 2. The migration path

### From CUDA (SYCLomatic / dpct)

- Generate a compile database, run `dpct` (`--cuda-gpu-arch=sm_80`), keep the migrated tree separate from the
  original (fall back per file), and put **every hand fix in one idempotent script** (one entry per dpct mistake) so
  a re-migration replays them.
- dpct mistakes seen, all silent or nearly so:
  - CUDA's null stream becomes a null `sycl::queue*` (crash in submit) - route through a queue getter.
  - `__fadd_rn(a, b)` becomes `a + b` **without parentheses**: `__fadd_rn(s, c ? x : y)` turns into
    `s + c ? x : y`. Audit every line after a DPCT1013 note.
  - The cast in `__ldg((const float*) p)` is dropped.
  - `cudaMemcpy` is synchronous; the migrated `q.memcpy` is not - add `.wait()` where the code relied on it.
  - Launches get pinned to sub-group sizes; templated launchers lose template parameters in kernel names
    ("redefinition of KernelInfo"); macros get expanded into unrelated call sites.
  - Type punning through non-char types (`(const uint16_t*) &x` read as `int2`) miscompiles on the device -
    use shifts or `sycl::bit_cast`.
  - `volatile` does not bypass Intel's caches: a host-mapped doorbell needs system-scope atomics.
- Keeping a port in step with upstream: never re-port by hand. dpct the old upstream tree and the new one, normalize
  both (canonicalize dpct's kernel-name hashes, which change run to run), and three-way merge each file
  (`git merge-file --diff-algorithm=histogram`, base = the commit the port was *migrated from*, not the branch
  point). Only re-merge files whose upstream source changed. Afterwards **count the port's own symbols before and
  after** - a dropped setter or a lost spin bound shows up there before it shows up as a hang.

### From Vulkan / compute shaders

The concepts map one to one; the traps are in the memory model and the sub-group size.

| Vulkan / GLSL | SYCL |
|---|---|
| workgroup, `local_size_x` | `nd_range` local size |
| `shared` | `sycl::local_accessor` |
| `subgroupAdd`, `subgroupShuffle` | `reduce_over_group(sg, ...)`, `select_from_group`, `shift_group_left` |
| push constants, specialization constants | kernel arguments, C++ templates (compile-time shapes) |
| `barrier()` | `group_barrier(it.get_group())` |
| buffers / descriptor sets | USM device pointers passed as arguments |
| `coherent` / memory barriers | `sycl::atomic_ref` with explicit scope and order |

Pin the sub-group size explicitly (`[[sycl::reqd_sub_group_size(16)]]` - 16 is Xe2's native width) wherever code
assumes one, and re-derive any lane-count constants.

### From a PyTorch / diffusers reference (a model port)

1. Read the reference implementation completely before writing a kernel: the pipeline blocks, the scheduler's
   exact sigma/timestep arithmetic, the prompt template (whitespace changes outputs), the special tokens, which
   tensor is normed where, the dtypes it computes in.
2. Write a **stage dumper** in Python that runs the reference on the CPU in float32 on **the same files** the port
   loads (same quantized weights), and saves each stage's input and output (`.npy`). Keep it in the repo
   (`reference/<model>/ref.py`) - it is the only Python.
3. Port stage by stage, each stage **started from the reference's own input**, so an error is born where it is seen.
4. Fill in the pipeline, then make it fast (section 5) - never the other way round.

## 3. Testing: exact before fast

- **Compare with three numbers**: relative RMS error, max absolute error, cosine. Typical, healthy values: half
  weights against an f32 reference 1e-4 - 2e-3; int8 weights 1e-2 - 3e-2 (cosine >= 0.999); f32 kernels 1e-6.
  A 0.1+ relative error or a cosine below 0.99 is a bug or a broken quantization scheme, not "precision".
- **Autoregressive models: teacher forcing.** The reference samples with its own RNG; you will not match the
  draws. Dump the reference's sampled tokens/codes, force them in the port, and compare every hidden state and the
  logits frame by frame. Unit-test the sampling semantics (guidance, top-k with ties, masks) separately.
- **Language models**: verify passes must equal one-token decode bit-exactly (`spec-check`: decode-width products
  row by row), a batch's rows must equal each conversation alone (`batch-check`), and a layer-by-layer check against
  a reference dump. **Compare output tokens, never just timings** - a "faster path" that silently dropped prompt
  tokens looked 2x faster for three hours. Near-ties (two logits < 0.05 apart) flip with summation order; judge
  those by cosines, not the token.
- **Non-deterministic references**: measure the reference against itself first (two runs gave cosine 0.996 for a
  video DiT); the port passes when it is inside that spread at the same speed.
- **Canary kernels**: a tiny parity binary per numerically sensitive kernel (activation quantization must be
  byte-exact). It caught a lost compiler flag that nothing else noticed.
- **Daemons and servers**: a bug that appears only on the second job is cross-job cached state - e.g. reordered
  weights cached by device pointer, and the allocator reused the address. Always test two jobs in one fresh process.
- **Hangs**: add a "sync and log after every phase" switch - if the hang vanishes, it is an ordering bug. Bisect by
  reverting files, not by staring. `gdb-oneapi` attaches in a container with `--cap-add SYS_PTRACE`. Toggle the
  Level Zero adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) as an A/B. Fill new buffers with NaN (a poison switch) to
  find uninitialized reads; a NaN detector on GEMM inputs finds where NaNs start (token 0 forever = NaN logits).
- **Bench hygiene**: drop the model files from the page cache for cold runs (`posix_fadvise DONTNEED`); measure
  enough work (256 decode tokens, not 64 - one stall swamps a short run); run several times; include the smallest
  shape in every A/B (buffer-size bookkeeping bugs show only there); remember the profiler itself costs (9% here).

## 4. Profiling

- **Device timestamps, not host timers**: a queue with `enable_profiling`, a stamp event at each section boundary,
  the elapsed time read at the end - no syncs inside the measured work. Print one line per section
  (`[profile NAME  S s  N calls]`).
- **Flame graphs from those lines**: nest the sections (attention under its layer, experts under MoE) into JSON
  and render it with a small HTML template - it shows at a glance that "experts 45%, half of it expanding 2-bit
  weights" where a flat list hid it.
- **Bandwidth math before tuning**: bytes a kernel must move / its time, against the card's peak (~600 GB/s on the
  B70). Decode-sized products should reach 70-85%. A kernel far below that is mis-shaped (alignment, occupancy),
  not "slow hardware". FLOPs / time against peak for compute-bound work.
- An eager mode (no graphs) for profiling; per-op timing switches (`NS_*_PROFILE=1`); `ONEDNN_VERBOSE` to see which
  oneDNN implementation runs. Profile without JIT-compiled libraries (the first JIT call lands in your numbers).

## 5. What made it fast (measured)

**Build**
- **AOT-compile for the card** (`-fsycl-targets=spir64_gen`, `-device bmg-g31`): the first window cost 47 s of JIT
  otherwise. `-fsycl-device-code-split=per_kernel` cut first-launch latency to 245 ms.
- Correctness flags that also hold in AOT: `-fp-model=precise`, and `-cl-fp32-correctly-rounded-divide-sqrt` -
  which must go *inside* the AOT backend options (`-Xsycl-target-backend=spir64_gen "-device bmg-g31 -options ..."`)
  or it is silently lost.

**Memory-bound kernels (decode, GEMV, dequant)**
- **Check alignment first.** Q6_K's 210-byte blocks made 16-byte loads 2-byte aligned; the B70 splits them (145 GB/s
  vs 640 aligned). Two aligned 16-byte loads plus a shift: decode 45 -> 54 tok/s, then the same for every block
  format: 70.6 tok/s.
- A small-batch product that reads the matrix once for up to 8 rows (a sub-group per rows, 16-byte vector loads,
  the reduction at the end) instead of a GEMM library for M = 1-8.
- int8 weights with a scale per row, against float activations, for decode (half the bytes: 21 -> 35 frames/s on an
  8B model). For wide products (prompts) expand the int8 matrix to half and use the half GEMM: quantizing the
  *activations* per row as well (W8A8) destroyed Qwen3's outlier features (relative error 0.37 -> 0.02).
- W8A8 does work with a rotation: a normalized Hadamard ("ConvRot") along the inputs in groups of 256 spreads the
  outliers, and the int8 GEMM then runs at the card's int8 rate - 1.4x on a 7B DiT, PSNR 39 dB against half.

**Compute-bound kernels**
- Use oneDNN (on the same SYCL queue) for GEMMs and convolutions, and cache primitives per shape and reordered
  weights per buffer. A plain-loop 1-D convolution took 4.7 s a window of a vocoder; oneDNN took 0.27 s, exact.
- Flash attention from sycl-tla (Intel's CUTLASS port) through a small wrapper library: 4.95 -> 2.46 ms a block at 4k
  tokens (~110 TFLOPS) against oneDNN's fused SDPA. Read q, k, v in place through strides - no copies.
- Fuse the small passes: a layernorm + modulation done one work-group a row (the row in registers): 1.69 -> 0.37 ms.

**Latency and the host**
- **SYCL graphs** for the decode window and the drafter, recorded when a session is created (capturing inside
  decode made the first run slow).
- Speculative decoding (the model's own MTP draft layer, verify passes exact) was worth more than all kernel
  micro-tuning together (< 15%).
- Keep sampling on the device (argmax / draws) to avoid a host round trip a token.
- Host I/O shows up as GPU slowness: hashed per-token embedding rows read one at a time from disk cost 15% of decode
  on new text. `posix_fadvise(WILLNEED)` on all of a window's rows first: 71.8 -> 83.9 tok/s. Random reads of many
  small rows: parallel readers up to the drive's IOPS, and a small first chunk to hide them.
- Weights streamed per chunk: the chunk size sets the speed at long context (2k-token chunks 278 s, 4k 77 s).

**Multi-GPU**
- Split layers by each card's measured speed; often one fast card beats two (a B65 is ~1.6x slower per layer, and
  PCIe swaps cost more than they save). Pipeline long prompts across cards.

**What did not help (so you do not repeat it)**
- XMX `joint_matrix` fused dequant + GEMM and a joint_matrix prompt attention: correct but slower - those paths were
  dequant/staging-bound, not compute-bound. (On the B70: fp16/bf16 16x16x16 and 32x64x16, s8 16x16x32, packed B,
  sub-group 16 only.)
- SIMD16 variants of already bandwidth-bound kernels; SageAttention int8 q/k at 4k tokens (slower with its quantize
  passes, 10x the error).
- A second in-order queue on the same card to overlap a bandwidth-bound stage with a compute-bound one: the card
  takes the two queues in turns (118 -> 114.5 s).

## 6. Host code in Rust

- Kernels behind a plain C ABI (`int fn(gpu*, ...)` returning 0 or -1 with the reason in a thread-local error),
  one header per engine, bound by symbol name at run time (`dlsym`) into a table of function pointers; every
  `unsafe` call carries a `// SAFETY:` line naming the sizes it relies on.
- A device buffer type owning its allocation (freed through its GPU's context), with explicit sync points; async
  uploads only through a pinned ring.
- One kernel library per model kind, so a process loads only its kind's code; an engine per model architecture,
  free to be specialized; a contract (trait) per kind that the server and CLI drive.
- Engine options declared by each engine (name, environment variable, help, load or request time) and forwarded from
  any command (`--opt-NAME VALUE`) and API request, so new architectures need no CLI changes.
- Check the plan before loading: the size of every file and buffer against free VRAM, with an error that says what
  to change.

## 7. Checklist for a new port

1. Safety: memory budget function, per-device context, bounded spins, kernel-log check after each run.
2. Reference dumper on the CPU, f32, same files; stage list written down.
3. Kernels AOT with the precise-FP flags; a canary parity test.
4. Stage-by-stage parity from the reference's inputs; teacher forcing for sampled models; record the numbers.
5. Whole pipeline; then two jobs in one process; then the server path with its real memory.
6. Profile with device timestamps; bandwidth/FLOP math per kernel; flame graph.
7. Tune the biggest bar first: alignment, then the small-batch product, int8 weights, oneDNN / flash attention,
   fusion, graphs, host I/O. Re-run parity after every change and quote before/after numbers.
