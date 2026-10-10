# Contributing

nextsycl welcomes new models - language, image, video and audio - and work on the ones it has. Each model architecture
is its own **engine**: a complete runtime for that model, free to be as specialised as it needs to be fast. This
guide is how to port one, and the few rules every change follows.

## The rules

1. **SYCL is the only GPU runtime.** Every kernel is SYCL (oneAPI DPC++, Level Zero), on the GPU's own queue. No
   Vulkan, CUDA, OpenCL, HIP, Metal, wgpu, or a framework that brings one (candle, PyTorch, ONNX Runtime, burn...).
   oneAPI's own libraries (oneMKL, oneDNN) run on the same SYCL queue and are fine. Host code is Rust.
2. **Third-party crates are fine** - anything that does not bring another GPU runtime (rule 1). Prefer a few
   well-known ones; say why in the pull request.
3. **Keep the architecture** (`docs/architecture.md`):
   - an engine is its own crate under its kind (`llm/<arch>`, `image/<arch>`, `video/<arch>`, `audio/<arch>`), with its kernels in
     `kernels/<kind>/<arch>/`;
   - it depends only on the foundation (`crates/`) and its own kind's contract (`<kind>/contract`) - never on
     another kind, the glue (`glue/`) or the program (`cli/`);
   - what two kinds both need moves down into the foundation, not sideways;
   - its kernels' C functions are named `ns_<kind>_<arch>_*`, declared in its own header, and bound by its own crate
     by name (its `ffi.rs`) - never added to the shared `kernels/ns/ns.h`.
4. **Exact before fast.** An engine is checked against a reference (the model's own implementation, or llama.cpp
   for an llm) before it is tuned, and every speed claim comes with the measurement that made it.
5. **Tune in the model's silo, never in the shared kernels.** The shared kernels (`kernels/ns`, `kernels/strata`,
   `kernels/diffusion`, and one engine's kernels another engine calls) are tuned for the engines already on them;
   changing one to speed up another model can slow or break those. A kernel tuned for one model goes in that model's
   **silo**, `kernels/<kind>/<arch>/silo/` - its own library, `dist/silo/libnextsycl-<arch>.so`, which the engine
   opens at load (`nextsycl_core::silo(ARCH)`) and calls for what it covers (each silo op has a `_supported` check),
   the shared kernel for everything else; without the file (or with `NS_SILO=0`) the engine runs on the shared
   kernels alone. Start a silo kernel as a copy of the shared one, then change the copy. A change to a shared kernel
   itself (a bug fix, a new type) re-runs every engine that calls it - its check and its bench - before it lands.

Rules 1 and 3 are checked by `cargo test` (`cli/nextsycl/tests/architecture.rs`) and `./build.sh test` (the kernel
libraries link against no other runtime); CI runs both.

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
   it sets, what the value looks like, a line of help, when it applies). The program then forwards
   `--opt-NAME VALUE` from every command (and an API request's `"options"`) to it, rejects a typo with the list,
   and shows them under `nextsycl <kind> engines` - no change to the command line for a new architecture.
7. **Register it.** Add `nextsycl_<kind>_<arch>::kind()` to the program's list of that kind
   (`cli/nextsycl/src/main.rs`: `engines()`, `video_engines()`; `cli/nextsycl/src/image.rs`: `engines()`). `nextsycl <kind> engines`
   lists it.
8. **Then make it fast** - as its own thing: graphs, fused kernels, its own memory plan, and kernels tuned for its
   shapes in its silo (rule 5: `kernels/<kind>/<arch>/silo/`, never by editing the shared ones). Check that the silo
   gives the same results as the shared kernel (same outputs, the same greedy text with `NS_SILO=0` and without), and
   measure with the kind's bench before and after (`nextsycl llm bench`, ...); quote both in the pull request.
9. **Document it**: its settings and numbers in `docs/architecture.md` (its engine section) and the README's speed
   tables; a catalog entry when its files are public.

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
