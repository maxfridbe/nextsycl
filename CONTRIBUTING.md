# Contributing

nextsycl welcomes new models - language, image, video and audio - and work on the ones it has. Each model architecture
is its own **engine**: a complete runtime for that model, free to be as specialised as it needs to be fast. This
guide is how to port one, and the few rules every change follows.

## The rules

1. **Every language model caches prompts.** A conversation's next turn reads only what is new: the server checkpoints
   each prompt at the start of its last user turn (and its first, and its end) and mounts the longest checkpoint
   that is a prefix of the next prompt. So an engine implements `save` / `restore` / `read_checkpoint` with its
   `cache_fingerprint`, and its chat template keeps what comes before the last user turn unchanged on the next turn
   (rewrite only what follows it - effort notes, dropped thinking) and begins user turns with a marker in
   `nextsycl_tok::USER_TURNS` (a new template's marker is added there). The CPU-side check runs for every engine;
   check a port on the box too: the second turn's request shows `reused` > 0.
2. **SYCL is the only GPU runtime.** Every kernel is SYCL (oneAPI DPC++, Level Zero), on the GPU's own queue. No
   Vulkan, CUDA, OpenCL, HIP, Metal, wgpu, or a framework that brings one (candle, PyTorch, ONNX Runtime, burn...).
   oneAPI's own libraries (oneMKL, oneDNN) run on the same SYCL queue and are fine. Host code is Rust.
3. **Third-party crates are fine** - anything that does not bring another GPU runtime (rule 2). Prefer a few
   well-known ones; say why in the pull request.
4. **Keep the architecture** (`docs/architecture.md`):
   - an engine is its own crate under its kind (`llm/<arch>`, `image/<arch>`, `video/<arch>`, `audio/<arch>`), with its kernels in
     `kernels/<kind>/<arch>/`;
   - it depends only on the foundation (`crates/`) and its own kind's contract (`<kind>/contract`) - never on
     another kind, the glue (`glue/`) or the program (`cli/`);
   - what two kinds both need moves down into the foundation, not sideways;
   - its kernels' C functions are named `ns_<kind>_<arch>_*`, declared in its own header, and bound by its own crate
     by name (its `ffi.rs`) - never added to the shared `kernels/ns/ns.h`.
5. **Exact before fast.** An engine is checked against a reference (the model's own implementation, or llama.cpp
   for an llm) before it is tuned, and every speed claim comes with the measurement that made it.
6. **Tune in the model's silo, never in the shared kernels.** The shared kernels (`kernels/ns`, `kernels/strata`,
   `kernels/diffusion`, and one engine's kernels another engine calls) are tuned for the engines already on them;
   changing one to speed up another model can slow or break those. A kernel tuned for one model goes in that model's
   **silo**, `kernels/<kind>/<arch>/silo/` - its own library, `dist/silo/libnextsycl-<arch>.so`, which the engine
   opens at load (`nextsycl_core::silo(ARCH)`) and calls for what it covers (each silo op has a `_supported` check),
   the shared kernel for everything else; without the file (or with `NS_SILO=0`) the engine runs on the shared
   kernels alone. Start a silo kernel as a copy of the shared one, then change the copy. A change to a shared kernel
   itself (a bug fix, a new type) re-runs every engine that calls it - its check and its bench - before it lands.
   **Tuning for other hardware goes in a variant** of the silo, never in the default silo or the shared kernels:
   `kernels/<kind>/<arch>/silo/variants/<name>/` holds a `variant.yml` (a one-line `description`, who `measured` it on
   what, and a `match` rule on what the device reports - `vram_mb_min` / `_max`, `units_min` / `_max`, `mhz_min` /
   `_max`, `sub_group`, a `name` substring only as the last resort) and only the ops that differ, under the silo's C
   names (copy the op, then change the copy; its kernels in a namespace of its own). It builds to
   `dist/silo/libnextsycl-<arch>@<name>.so`; each GPU then takes its variant's op where it has one, else the default
   silo's, else the shared kernel (`docs/architecture.md`). A variant gives the same outputs as its silo
   (`NS_SILO_VARIANT=none` against it, with the silo's own check), its pull request quotes before / after numbers on
   its hardware, and it never changes a memory plan (memory is planned from what the device reports).
   **A variant may also set its engine's options for its cards**: an optional `settings:` block in `variant.yml`,
   option name -> value, the names the engine declares (`EngineOption`, the ones `--opt-NAME` takes, `nextsycl <kind>
   engines`). Where the variant applies to every GPU the engine loads on, each becomes the engine's default - below a
   request's options, `--opt` / the environment and the model card's settings, above the engine's own default
   (`nextsycl silo` and the load's log say what applied). **An engine's alternative host-side paths** (another
   chunking, a fusion, an expert placement) are declared options with enumerated values (`value: "spread|pack"`):
   the variant picks one by name, and a new path is itself an engine change under the gate, never a variant's own
   code. A setting never sizes memory (an option whose value is `TOKENS`, `GIB`, `MIB` or `BYTES` is refused): an
   option may choose a placement strategy, the sizes still come from what the device reports. **Two variants of one
   silo never match the same card**: their numeric `match` ranges must not meet (a sub-group size or a name never
   tells two rules apart) - give one a tighter `match`. And a variant has a **gate baseline on a card it matches**,
   recorded on that card and committed in its pull request (the merge gate below).

7. **A front-door method is its handler.** `glue/serve/src/home/methods.rs`: `#[rpc(name = "image.generate", tags
   = "image", example = r#"{...}"#)]` on a fn with a doc comment (its summary), and a request struct with
   `#[derive(Rpc)]` (each field's doc comment its description, `#[schema(min, max, default, one_of ...)]` its
   limits). The table, the OpenAPI document's operations, system.version's list and the wfe client
   (`nextsycl gen-api` -> `wfe/home/src/gen/nextsycl.ts`) are made from that; never describe a method a second
   time. The generator, rpcgen, is vibe-creative-platform's, vendored in `vendor/rpcgen`: change it there, then
   `tooling/sync_rpcgen.sh` (a test fails on a copy edited here, and on a stale client). What happened is in the
   answer's envelope (`error.code`, `retryable`), never only in the HTTP status.
8. **The cards are one backplane.** Every enabled model of every kind loads on demand; only an idle holder neither
   pinned nor held is stopped for room, running work never; a method that runs a model takes `rsvp_s` and
   `max_willing_to_wait_s` (`home/backplane.rs`, `RUNS_MODEL`), and the page has no start or load: the page's only stop
   is a force unload of an idle model (`system.unload` with `force_when_idle`: refused while it answers a request or
   runs a job, over its pin and hold once idle). The decisions are
   `crates/scheduler` (`backplane-scheduler`, nextcuda's, copied): pure code the front door drives through its `Host` traits (`Sched`, `home/life.rs`),
   never a second copy of the logic beside it; a change to the crate is written in its CHANGELOG.md, so the next copy can carry it.

Every rule a machine can check is a test in `cli/nextsycl/tests/architecture.rs` (its header maps each rule to its
tests; rule 5 is a review rule), run by `cargo test`; `./build.sh test` checks the kernel libraries link against no
other runtime. CI runs both. A new rule comes with its test. Rule 7 is checked by the compiler, `tsc` and `cargo test` (`home::api_files_tests`).

## Building

```sh
./build.sh            # the kernel libraries and the program, in a container with oneAPI (podman); output in dist/
./build.sh test       # clippy (warnings are errors), the unit tests, the architecture's rules
cargo build --release # the Rust side alone builds anywhere (the kernels are loaded at run time)
```

A GPU is needed only to run: an Intel Arc (Xe2: B580 / B70 / B65), Level Zero, and the oneAPI runtime (in the
build image). `nextsycl llm selftest` (and `image` / `video` / `audio`) checks the whole path on your machine: the kind's
kernel library opens, an engine's own symbol binds, a kernel runs on the GPU and its result comes back right.

## Porting a model, step by step

Each kind has a template that compiles, runs its example kernel to prove the path to the GPU, and fails with a
clear message where the port begins: `llm/example`, `image/example`, `video/example`, `audio/example`, with their kernels in
`kernels/<kind>/example/`. Every method of the contract is there with a comment saying what it must do.

1. **Copy the template.** `cp -r llm/example llm/<arch>` and `cp -r kernels/llm/example kernels/llm/<arch>` (or
   image / video / audio). Rename the crate (`nextsycl-<kind>-<arch>`), `ARCH`, and the kernel symbols
   (`ns_<kind>_<arch>_*`). Add the crate to the workspace (`Cargo.toml`: `members` and `[workspace.dependencies]`).
   Keep its `selftest` working as you go: it is the quickest proof that your kernels are built and bound.
2. **Describe the model.** Read the file's metadata and find every tensor by role, checked at load, so a wrong file
   fails with a reason before any pass runs (`llm/qwen4exp/src/model.rs` is the pattern). Image and video models
   are several files by role (`ROLES`: transformer, text encoder, VAE...).
3. **Get a reference.** Run the model's own implementation on a fixed input and dump what each stage produces (the
   embeddings, a layer's output, the logits or latents, the decoded picture). Keep the dumping script in
   `reference/` - it is the only place Python lives.
4. **Port the forward pass, checked stage by stage.** Write the kernels in `kernels/<kind>/<arch>/` (SYCL; reuse
   `kernels/ns` and `kernels/strata` where they fit), bind them in `ffi.rs`, and compare each stage with the dump.
   For an llm: `nextsycl llm check <file> <dump>` compares every layer; then `spec-check` and `batch-check` must
   hold exactly. For image / video: compare the latents step by step and the final picture or frames.
5. **Fill in the contract.** Sessions, decode and checkpoints (llm), or the sampler loop over
   `nextsycl_diffusion::sigmas` and the decode (image / video) - the template's comments say what each method owes.
6. **Declare its options.** Anything the engine reads beyond its kind's common settings - a draft layer, a chunk
   size, a schedule, a debug switch - goes in its kind's `options` (`EngineOption`: a name, the environment variable
   it sets, what the value looks like - `a|b|c` for a choice between paths the engine has, which a silo variant's
   `settings` can pick for its cards (rule 6) -, a line of help, when it applies). The program then forwards
   `--opt-NAME VALUE` from every command (and an API request's `"options"`) to it, rejects a typo with the list,
   and shows them under `nextsycl <kind> engines` - no change to the command line for a new architecture.
   A new variable (an option's, a setting's, a kernel's getenv) gets a row in `docs/environment.md`; the
   architecture test (`cli/nextsycl/tests/architecture.rs`) fails on one that has none.
7. **Register it.** Add `nextsycl_<kind>_<arch>::kind()` to the program's list of that kind
   (`cli/nextsycl/src/main.rs`: `engines()`, `video_engines()`; `cli/nextsycl/src/image.rs`: `engines()`). `nextsycl <kind> engines`
   lists it.
8. **Then make it fast** - as its own thing: graphs, fused kernels, its own memory plan, and kernels tuned for its
   shapes in its silo (rule 6: `kernels/<kind>/<arch>/silo/`, never by editing the shared ones). Check that the silo
   gives the same results as the shared kernel (same outputs, the same greedy text with `NS_SILO=0` and without), and
   measure with the kind's bench before and after (`nextsycl llm bench`, ...); quote both in the pull request.
9. **Document it**: its settings and numbers in `docs/architecture.md` (its engine section) and the README's speed
   tables; a model card (`model-cards/<kind>/<id>.yml`, model-cards/README.md) when its files are public - a model
   of an architecture already served needs only its card, no code.

## Silos and variants: how tuning stays out of each other's way

Every kernel an engine runs comes from one of three layers. Each GPU picks an op from the most specific layer that has
it when the engine loads: **variant → silo → shared kernel**.

| | shared kernel | silo | variant |
|---|---|---|---|
| **Scope** | every engine of a kind | one model architecture | one model on one class of hardware |
| **Where** | `kernels/<kind>/*.cpp`, `kernels/ns`, `kernels/strata` | `kernels/<kind>/<arch>/silo/` | `kernels/<kind>/<arch>/silo/variants/<name>/` + `variant.yml` |
| **Builds to** | the kind's library | `dist/silo/libnextsycl-<arch>.so` | `dist/silo/libnextsycl-<arch>@<name>.so` |
| **Holds** | the plain, correct op | the model's ops tuned for its shapes | only the ops that differ from the silo (a delta) |
| **Settings** | - | - | optional `settings:` in `variant.yml`: engine options (the names `--opt-NAME` takes) as the engine's defaults on its cards - a path the engine already has (an option with enumerated values), never a memory size |
| **Chosen by** | always there (the fallback) | the engine's arch | its `match` against what the GPU reports: `vram_mb_min/max`, `units_min/max`, `mhz_min/max`, `sub_group`; a `name` substring as a last resort. Two variants of one silo never both match a card |
| **Override** | `NS_SILO=0` (silo off) | `NS_SILO=0` | `NS_SILO_VARIANT=<name>\|none`, or a model card's `silo.variant` |
| **Who may change it** | a bug fix or new type, with every engine that calls it re-checked | that model's porter | whoever tunes that hardware |
| **Must prove** | every caller's check and bench | the same output as the shared kernel; before and after numbers | the same output as the silo (`NS_SILO_VARIANT=none` against it); before and after numbers on its hardware; a gate baseline recorded on a card it matches, and the gate's table from that card |

**What it protects against**
- **Pull requests fighting.** Someone tuning for a 16 GB card, or a card with fewer units, adds a variant. The default
  silo that the 32 GB B70 / B65 run doesn't change, so two tunings for different hardware never edit the same file.
- **One model's speed-up slowing another.** Tuning goes in that model's silo, never in a shared kernel, so another
  engine's speed and output can't move.
- **Silent output changes.** Each layer must give the same output as the layer below it, and the merge gate
  (`tooling/gate.sh`) compares logits and greedy tokens exactly. pp and tg must be within 3% on each card a change
  touches.
- **Hard-coded memory.** A variant never changes a memory plan. Memory is planned from what the device reports
  (`Gpu::free_or_guess`), so a smaller card gets a smaller plan without a fork. A variant's `settings` may not set an
  option that sizes memory (`TOKENS`, `GIB`, `MIB`, `BYTES`).
- **Forking an engine for one card.** A card that wants another host-side path - a chunking, a fusion, an expert
  placement - gets it as an engine option with enumerated values (`value: "spread|pack"`), and its variant picks one
  in `settings`. A new path is an engine change under the gate; the variant only chooses. Precedence, highest first:
  a request's options, `--opt` / the environment, the model card's settings, the variant's, the engine's default.
- **Two tunings claiming one card.** The architecture test fails when two variants of one silo can match the same
  card (their numeric ranges meet; a sub-group size or a name never separates them). The fix is a tighter `match`.
  At run time the first by name is taken, with a warning.
- **Hardware we don't own.** The gate runs on a contributor's own box (`NS_GATE_HOST=local`), and its baselines are
  per card (`arc-pro-b65-160u-32g`: the name, units and memory the card reports). A variant's pull request records
  its card's baseline (`NS_GATE_RECORD=1`), commits it, and pastes the gate's table from that card; the
  architecture test fails on a variant with no baseline on a card it matches.
- **Mixed boxes.** The layer is chosen per GPU (`nextsycl_core::silo_for(arch, gpu)`). A B70 beside a 16 GB card each
  get their own ops. An engine that binds once per process takes a variant only when every GPU matches it, and a
  variant's settings (process-wide) apply only when every GPU the engine loads on matches it.
  `nextsycl silo [ARCH]` shows what each GPU takes and the settings that come with it.

The rules are enforced by tests in `cli/nextsycl/tests/architecture.rs` and by the gate, not only by review
(CONTRIBUTING.md rule 6). The example variant is `kernels/llm/qwen35/silo/variants/b65/`.

## Code style

- Write like the code around you: the same comment density, naming and idioms. Comments say what something is and
  why, in plain words.
- `cargo clippy --all-targets -- -D warnings` is clean; `unsafe` blocks carry a `// SAFETY:` line.
- C++ kernels: SYCL 2020 with the oneAPI extensions; `-fsycl-default-sub-group-size=32` is the default
  (`kernels/build.sh`); every exported function is `extern "C"`, returns 0 or -1 with `ns_fail(reason)`
  (`NS_TRY` / `NS_CATCH`), and queues its work on the GPU's queue.
- A kernel that can read past a buffer or spin forever can wedge the GPU (an engine reset, sometimes a reboot):
  bound every loop and every allocation by construction, and test new kernels at small sizes first.

## Pull requests

- One change a pull request, with what it does, why, and how it was checked (the commands and their output).
- Speed changes: the numbers before and after, on which GPU, with which command.
- Correctness changes: the check that failed before and passes after.
- **The merge gate**: `tooling/gate.sh <base> <head>` works out from the paths changed which engines they touch (an
  engine's own code, kernels or silo: that engine; a silo variant: that engine on the cards it matches; a shared part:
  the engines `tooling/gate-baselines.json` maps it to), builds `<head>` in a build copy on the GPU box
  (`NS_GATE_HOST`, `NS_GATE_DIR`), runs each engine's quick run on each card under the GPU lock, and compares with
  the baselines: the logits and greedy tokens exactly, pp and tg within 3%. Paste its table into the pull request; a
  change that moves a number on purpose records new baselines (`NS_GATE_RECORD=1`) and says why. An engine with no
  baseline is listed as such - add its run to the file when you port one.
- **Gating hardware the project does not own.** The baselines are per card, keyed by the id `nextsycl silo` prints
  (`arc-pro-b65-160u-32g`: the name, units and memory it reports, with the `device` recorded beside it), so the gate
  runs on your own box: `NS_GATE_HOST=local` (no ssh; no GPU lock unless `NS_GATE_LOCK` names one), `NS_GATE_GPUS`
  your GPU indexes, `NS_GATE_PATHS=FROM=TO` where the baselines' model files are on your disk. Record your card's
  baseline on the base commit (`NS_GATE_RECORD=1 tooling/gate.sh <base> <base> <kind>/<arch>`), commit it in the same
  pull request, then run the gate on the head and **paste its table from your hardware**. A variant's pull request
  always does this: the architecture test fails on a variant with no baseline on a card its `match` accepts.
