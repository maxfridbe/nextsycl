# Engines: one a model, tuned end to end

nextsycl runs more than one model architecture by giving each its own **engine** - a complete runtime for that
model on these GPUs - behind one small contract the server and the command line drive. There is no generic forward
pass parameterised by a config: the speed in this repository came from decisions that only make sense for one
model (GLM-5.3's KDA scan sized per card, its MLA indexer's ring, its expert store, its draft block, its prompt
pipeline), and a second model gets the same freedom.

## The layout

```
crates/ns-sys       the C ABI of libnextsycl (the kernel library)            shared
crates/ns-core      GPUs, device / host buffers, an arena, the shared ops     shared
crates/ns-gguf      the file format                                          shared
crates/ns-tok       the tokenizer, the chat templates                        shared
crates/ns-runtime   the contract: Engine, Session / Decoder / Checkpoint,     shared
                    Sampler, Tap, GpuInfo, LoadOptions, EngineKind
engines/<arch>      ONE ENGINE A MODEL: its model description (geometry, tensors by role, checked), its engine
                    (forward passes, memory plan, decode loop, drafts), its tools (`info`, `kernels`)
kernels/ns          the shared kernels: queues and copies, norms, GEMMs, the quantized decode products
kernels/engines/<arch>   that engine's own kernels (in the same library; its own symbols)
kernels/strata      the imported Strata kernels (MIT), used by any engine
crates/nextsycl     the program: the server, the CLI, bench - only `dyn Engine`, picked by the file
```

`crates/nextsycl/src/main.rs` lists the engines it has (`engines()`); a file is served by the one whose
`EngineKind::archs` holds its `general.architecture` (`ns_runtime::kind_for`). `nextsycl info` names it.

## The contract (`ns-runtime`)

What the server needs from any model, and nothing about how it is computed:

- **Sessions:** `session(max_ctx)`, `reset_session`, `copy_session`, `save` / `restore` / `read_checkpoint` - a
  conversation's whole state, opaque (`Session`, `Checkpoint` box the engine's own types; only it reads them).
- **Reading:** `feed`, `feed_until` (stops between chunks for a newcomer), `forward`, `forward_rows`.
- **Decoding:** `decoder` / `decoder_after`, `step` (one or more committed tokens: drafts are the engine's affair),
  `rollback`, `forward_batch` (a row a conversation), drawing through a `Sampler` (`dist` + `uniform` let an engine
  do speculative sampling; a plain closure is a sampler too).
- **Reporting:** `gpu_info`, `expert_stats` (MoE engines), `report` (its profile lines), `cache_fingerprint` (what a
  disk checkpoint must match), `save_profile` (what the next load should know).
- **Optional:** `capture_attention` / `take_attention` (LogProbChain), `hold_arena`.

Every engine is held to the same checks: `spec-check` (verify passes equal one-token decode), `batch-check`
(a batch's rows equal each conversation alone), `check` against a llama.cpp dump, `bench`, `bench --needle`.

## Adding a model

1. `engines/<arch>/` with `Cargo.toml` (crate `ns-<arch>`: ns-core, ns-gguf, ns-runtime), `src/model.rs` (the
   geometry and the tensors by role, checked at load - copy GLM's pattern, not its roles), `src/engine.rs`,
   `src/lib.rs` (`impl Engine`, the opaque handles' `*State` traits, `pub fn kind() -> EngineKind`),
   `src/tools.rs` (`info`, and `kernels` if it has a kernel test).
2. Its kernels in `kernels/engines/<arch>/` (`ns_<arch>_*` symbols in a header of their own there, included at the end
   of `kernels/ns/ns.h`; bound by the engine's crate through `ns_sys::Api::symbol`, as qwen4exp's `ffi.rs`); reuse
   `kernels/ns` and `kernels/strata` where they fit.
3. Add `ns_<arch>::kind()` to `engines()` in `crates/nextsycl/src/main.rs` and the crate to the workspace.
4. Its chat template (ns-tok) as `EngineKind::chat`, and its pre-tokenizer in ns-tok if it is new.
5. Make `spec-check`, `batch-check` and `check` pass at short and long context before it is tuned; then tune it as
   its own thing.

## The engines

- **glm5next** - GLM-5.3-Flash: KDA + MLA with a DSA indexer, 288 experts (VRAM / pinned host store), MTP.
- **qwen4exp** - Qwen3.8-Flash-Next (Strata's family: the IQ2_XS general model; the Coder IQ1_M and Swift 1.5 files
  are the same architecture): Gated DeltaNet + QSA, hyper-connections, the hashed PLE, 512 experts, all in VRAM over
  the cards. Its glue (`kernels/engines/qwen4exp/qwen.cpp`) is the Strata port's verify window, a stage per GPU; its
  Rust side the memory plan, the PLE rows, sessions and checkpoints. It binds its own ABI from the library by name
  (`ns_sys::Api::symbol`): an engine's symbols are not in the shared tables. Its settings: `NS_QW_MTP` (Strata's MTP
  runtime directory), `NS_QW_EXPERT_PROFILE` (its expert profile: which experts stay in VRAM on one card),
  `NS_QW_CHUNK` (prompt chunk, 2048; Strata serves 4096), `NS_QW_SPEC` / `NS_QW_SPEC_MIN_P` (4 / 0.5),
  `NS_QW_EAGER=1` (no graphs), `NS_QW_WINDOWS=1` (prompts through windows), `NS_QW_PROFILE=1` (round timings).

## What stays shared on purpose

- **The kernel library** (one `libnextsycl.so`): one build, one ABI, symbols per engine. Splitting it per engine
  would buy nothing at load and cost a second toolchain run.
- **The expert-store and pipeline ideas** are not a shared library yet: GLM's are tuned to its shapes. When a second
  MoE engine wants the same, factor what both actually share - not before.
