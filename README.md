We stand on the shoulders of giants.

# nextsycl

A Rust runtime for large hybrid-attention MoE models on Intel Arc GPUs, with SYCL kernels. First model:
**GLM-5.3-Flash** (`glm5-next`): 45 layers of KDA and MLA attention with a DSA lightning indexer, 288 experts, and
hyper-connections. The experts that do not fit the cards live in pinned host memory, and the GPU reads them over
PCIe. It runs on one card or splits the layers over several.

## What it does

- **Two cards:** the layers split over the GPUs (each with a SYCL context of its own), the experts in an exclusive
  VRAM / pinned-RAM store.
- **MTP speculative decoding:** the model's own draft block. Verify passes are bit-identical to one-token decode,
  so greedy output with MTP equals greedy output without it.
- **Long context:** the DSA indexer past 2,048 tokens (each row attends to its top 512 pools of 4 tokens and its
  own unfinished pool), checked against llama.cpp.
- **A prompt cache:** checkpoints of the whole conversation state in host memory, at the end of a prompt's first
  turn, at the start of its last user turn, and at its end.
- **An OpenAI-compatible server** with streaming and the thinking split out, run as a service: `nextsycl start`,
  `stop`, `status`, `ps`, `cache`, `chat`, `logs`, over a control socket.
- **`POST /api/chat` for web pages:** the same chat as JSON lines (`{"thinking": ...}`, `{"content": ...}`, then a
  `{"done": true, ...}` line with the timings), with CORS for loopback pages and the origins in `NS_CORS`.

## Quick start

```sh
./build.sh                      # the kernel library and the program, in a container with oneAPI (podman)
echo "NS_MODELS=$HOME/models" > nextsycl.conf
dist/nextsycl start             # loads the model on every GPU; the OpenAI API on 127.0.0.1:8085
dist/nextsycl status            # live: the GPUs, the request running, the prompt cache
dist/nextsycl chat "Hello"
dist/nextsycl stop
```

`dist/nextsycl help` lists every command and setting.

## Standing on

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp):** the GGUF format, the quantization formats and their
  dot products, the reference every layer is checked against (`reference/llama-dump`).
- **[Strata](https://github.com/Niko1221/Strata):** the SYCL kernels in `kernels/strata` (MIT; see
  `kernels/strata/PROVENANCE.md`), and the ideas behind the expert store and the host mirror.
- **[ds4](https://github.com/antirez/ds4)** by antirez: how GLM-5.3's MTP block runs, and the file this runtime
  serves first.
- **Zhipu AI's GLM-5.3**, and the people who quantized it: the uncensored GSQ-RCO and IQ2 files.
- **Intel's oneAPI:** SYCL, oneMKL, and SYCLomatic, which migrated the kernels from CUDA.

## Releases

The version is `yy.mmdd.###`: the commit's date (UTC), then its number among that day's commits, from git
(`./version.sh`; `nextsycl version` prints the one built in). Every push to `main` is built and tested by
`.github/workflows/build.yml` and published as a release `v<version>` with `nextsycl-<version>-linux-x86_64.tar.gz`
(the program, `libnextsycl.so`, `ns.h`, the docs). Other branches and pull requests keep the tarball as a workflow
artifact.

## Docs

- `docs/glm5next.md`: the model's math, as this runtime computes it.
- `PLAN.md`, `TODO.md`, `NOTES.md`: where it is going, what is next, the files tried.

MIT licensed, as the kernels it imports.
