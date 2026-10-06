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
  own unfinished pool), checked against llama.cpp; the latent cache in fp16 (64K of context per session by default).
- **A prompt path on the XMX units:** prompts are read in chunks of 4,096 tokens, with the experts, the dense
  matrices and MLA's attention in fp16 through oneMKL's half GEMMs, and the experts that live in host memory copied
  on a second queue while the previous ones compute.
- **A prompt cache:** checkpoints of the whole conversation state in host memory, at the end of a prompt's first
  turn, at the start of its last user turn, and at its end.
- **An OpenAI-compatible server** with streaming and the thinking split out, run as a service: `nextsycl start`,
  `stop`, `status`, `ps`, `cache`, `chat`, `logs`, over a control socket.
- **Energy and logprobs in every answer:** `usage.energy_wh` (and `energy_wh` on `/api/chat`'s last line) - the watt-hours
  both cards drew for the request; `logprobs: true` (+ `top_logprobs`, up to 20) returns each answer token's
  log-probability and the likeliest alternatives, as OpenAI's `choices[0].logprobs.content`, streamed or not.
- **A record of each request:** `nextsycl inspect <id>` prints one as JSON (settings, timings, previews of the prompt
  and the answer); the server keeps the last `NS_KEEP_REQUESTS` (100).
- **`POST /api/chat` for web pages:** the same chat as JSON lines (`{"thinking": ...}`, `{"content": ...}`, then a
  `{"done": true, ...}` line with the timings), with CORS for loopback pages and the origins in `NS_CORS`.

## Quick start

```sh
./build.sh                      # the kernel library and the program, in a container with oneAPI (podman)
echo "NS_MODELS=$HOME/models" > nextsycl.conf
dist/nextsycl start             # loads the model on every GPU; the OpenAI API on 127.0.0.1:8085
dist/nextsycl status            # live: the GPUs, the request running, the prompt cache
dist/nextsycl inspect 4         # one request as JSON (settings, timings, previews; NS_KEEP_REQUESTS are kept)
dist/nextsycl chat "Hello"
dist/nextsycl stop
```

`dist/nextsycl help` lists every command and setting.

What `status` shows during a chat (GLM-5.3-Flash IQ2 on an Arc Pro B70 and a B65), and `ps` after it:

```
$ nextsycl status
glm-5.3-flash-uncensored - up 15s, context 65536, MTP on, 1 request(s) served

GPU  CARD                      VRAM USED      FREE   LAYERS EXPERTS VRAM/HOST   TEMP   VRAM   POWER
0    Arc Pro B70        30.1GiB / 31.9GiB    1.8GiB     0-21      3447 / 2026    36C    36C     78W
1    Arc Pro B65        30.1GiB / 31.9GiB    1.8GiB    22-44      3301 / 3474    34C    36C     86W
                                                                       the GPUs draw 164 W

request #1 (socket): generating, prompt 18 tokens, 0 reused (none), 184 / 300 generated at 16.9 tok/s, 12s, 1863 J so far

$ nextsycl ps
ID     VIA     STATE        PROMPT           REUSED    READ  GENERATED   TOK/S    ENERGY  AVG W FINISH       AGO
#1     socket  done             18           0 none    1.0s        300    16.9    2994 J    160 length       27s
```

TEMP and VRAM are the cards' package and memory temperatures, POWER each card's draw over the last second, and
ENERGY what both cards drew while the request ran (idle power included) - from the xe driver's sensors.

## Speed

GLM-5.3-Flash IQ2 (the ds4 file, 80 GB of experts) on an Arc Pro B70 and an Arc Pro B65, 6 October 2026:

| | |
|---|---|
| prompt, 3K tokens | ~400 tokens/s |
| prompt, 12K tokens | ~475 tokens/s |
| decode, short context, MTP on | ~18.5-19.5 tokens/s (87-90% of drafts accepted) |
| load | ~18 s |

About 60% of the experts fit in VRAM (plus ~380 slots a GPU lent by the prompt arena while decode runs); the rest sit
in pinned host memory. At decode a missed expert is swapped in over PCIe (both directions at once, on two copy
queues, while the resident experts compute; the next layer's likely experts prefetched) - on these cards' x8 links
those swaps are still the largest share of decode time (`TODO.md`). `--gpu all` puts the GPU that computes most last
(it takes the head and the draft block); `NS_SPLIT` sets the layers per GPU.

## Checking it

- `nextsycl check <model> <dump dir>`: the forward pass against a llama.cpp dump (`reference/llama-dump`), every
  step compared by cosine, the next token and the top logits.
- `nextsycl spec-check <model> --prompt-file F`: verify passes against one-token decode - must stay 0 difference
  (greedy output with MTP equals greedy output without).
- `NS_PROFILE=1 nextsycl generate ...`: seconds per section, the GPU synced at each boundary; `NS_PROFILE=gpu`: device
  timestamps instead (no syncs - the honest view of decode, where sections are tens of microseconds).
- Switches for A/B measurements: `NS_DENSE_F16=0`, `NS_PROMPT_F16_MIN`, `NS_PREFILL_CHUNK`, `NS_HC_FUSED=0`,
  `NS_DECODE_LANES`, `NS_DECODE_DIRECT=1`, `NS_ARENA_MIB`, `NS_PREFETCH`, `NS_LEND=0`, `NS_SPLIT`.

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
