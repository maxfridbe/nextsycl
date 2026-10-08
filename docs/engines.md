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
2. Its kernels in `kernels/engines/<arch>/` (`ns_<arch>_*` symbols, declared in `kernels/ns/ns.h` for now, bound in
   ns-sys); reuse `kernels/ns` and `kernels/strata` where they fit.
3. Add `ns_<arch>::kind()` to `engines()` in `crates/nextsycl/src/main.rs` and the crate to the workspace.
4. Its chat template in ns-tok if it is new.
5. Make `spec-check`, `batch-check` and `check` pass at short and long context before it is tuned; then tune it as
   its own thing.

## Next models

Strata's port serves the Qwen3.8-Flash-Next family on these cards (Coder IQ1_M, the IQ2_XS general model, Swift
1.5): hybrid attention (Gated DeltaNet + gated full attention, QSA selection), many small experts, a per-layer
embedding (PLE), an MTP draft layer. That is a separate engine here (`engines/qwen38next`), with its own memory plan
(PLE rows, the expert store) and kernels - the Strata port's are in `kernels/strata` already (GDN, QSA, PLE, the
grouped experts).

## What stays shared on purpose

- **The kernel library** (one `libnextsycl.so`): one build, one ABI, symbols per engine. Splitting it per engine
  would buy nothing at load and cost a second toolchain run.
- **The expert-store and pipeline ideas** are not a shared library yet: GLM's are tuned to its shapes. When a second
  MoE engine wants the same, factor what both actually share - not before.
