# Environment variables

Every environment variable nextsycl reads - the program, the servers, the engines, the kernels, the build. One row
each: its default (what happens when it is unset), what it does, what reads it (`cli`, `serve`, `switch`, `studio`,
an engine as `llm <arch>` / `image <arch>` / `audio <arch>` / `video <arch>`, `kernels:<lib>`, `build`).
`cli/nextsycl/tests/architecture.rs` checks that every name the code reads is here.

- **nextsycl.conf or the environment.** The program's settings (`cfg.get`) are read from the environment, then
  `nextsycl.conf` beside the repository (`dist/../nextsycl.conf`, the file `nextsycl serve`'s Config tab edits),
  then `~/.config/nextsycl.conf`: the first that has the name wins - the environment over the files. An empty value
  counts as unset. The engine and kernel variables are read with getenv where they run (in the container): set them
  in the file - the program forwards `NS_QW_*`, `NS_DS2_*`, `NS_QI_*`, `NS_MM3_*`, `NS_Q3T_*`, `NS_VCP_*`, `NS_M3D_*`, `NSD_*` and `H3_*` to
  the engine's container - or with `--opt-NAME`.
- **`--opt-NAME VALUE`** sets an engine's declared variable for one command (README "Engine options"; `nextsycl
  <kind> engines` lists them). Rows with `(--opt-NAME)` are those.
- **Debug:** and **Experiment:** mark switches for measuring and A/B runs, not for serving.

## Runtime and paths

Where the program, its libraries, its models and its sockets are.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NEXTSYCL_LIB` | the library beside the executable, else dist/ of the working directory | the kind's kernel library to load (a `libnextsycl-<kind>.so` path) | runtime (every kind) |
| `NS_CONTAINER_ENGINE` | podman, else docker (whichever is on PATH) | the container program | cli (llm, image, audio, video start) |
| `NS_DIST` | the program's own directory | dist/: where the program, the kernel libraries and wfe/ are looked for; nextsycl.conf is read from its parent | cli, serve |
| `NS_IMAGE` | localhost/nextsycl-build (`./build.sh image`) | the container image with the oneAPI runtime the servers run in; build.sh builds in it too | cli (llm, image, audio, video start), build (build.sh) |
| `NS_IN_IMAGE` | unset | set inside the image when the program re-runs a GPU command there (a host without oneAPI: `gpus`, `llm generate`, the selftests ...); set, the command runs in this process | cli |
| `NS_MODELS` | none (llm start fails: set it) | host directory with the model files, mounted as /models in the containers; `models pull` puts files there | cli (llm, image, audio, video start; models pull) |
| `NS_MODEL_DIRS` | none | more models folders, one a drive, comma-separated: a model moved to that drive (`system.move`, `nextsycl models move`) goes there, else to `<mount>/nextsycl-models`; NS_MODELS is the first models folder | serve (drives), cli (drives, models move) |
| `NS_MOUNTS` | none | more read-only mounts for the chat server's container, `host:inside[,...]` (another model folder, a draft layer, a profile) | cli (llm start) |
| `NS_SILO` | silos on | 0: no model silo, every model on the shared kernels (the A/B for a silo) | runtime (every kind) |
| `NS_SILO_NAME` | the architecture's own | another silo of the engine's ABI to load (dist/silo/libnextsycl-<name>.so) - a model card's `silo.name` | runtime (every kind) |
| `NS_SILO_REQUIRED` | 0 | 1: refuse to load when the silo is not built (no fall back to the shared kernels) - a model card's `silo.required` | runtime (every kind) |
| `NS_SILO_DIR` | silo/ beside the kind's kernel library | directory holding the silos (`libnextsycl-<arch>.so`) | runtime (every kind) |
| `NS_SOCKET_DIR` | $XDG_RUNTIME_DIR/nextsycl (serve's pid and marker files: /tmp) | where the control sockets live (mounted into the containers) | cli, serve, llm, video, studio |

## Front door and services

`nextsycl serve` (:8000), `nextsycl switch` and the servers' web access (README "Host services").

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_API_TOKEN` | made once, kept in serve.json beside the registry | the front door API's bearer token; `audio voices design` uses it to call the front door | serve, cli (audio) |
| `NS_AUDIO_PORT` | 8087 | the audio server's port, as the front door reaches it | serve |
| `NS_AUDIO_PORT2` | 8097 | the second audio server's port (`audio start --slot 1`): a model enabled on two cards runs on both while its work keeps the first busy; 0: one server | serve |
| `NS_MODEL3D_PORT` | 8088 | the 3D server's port, as the front door reaches it | serve |
| `NS_MODEL3D_PORT2` | 8098 | the second 3D server's port (`model3d start --slot 1`), as `NS_AUDIO_PORT2` | serve |
| `NS_SPEECH_NORMALIZE` | 1 | 0: speech's text as sent (by default math and code symbols are said as words - √ "square root of", = "equals", d_k "d sub k" - and markdown's marks dropped; a request's `normalize: false` too) | serve (audio) |
| `NS_AUTO_PULL` | 0 | 1: a request naming a model a card offers that is not downloaded starts its download (`system.models.pull`) and waits for it, a job saying `downloading <model>: N%`; 0: it is refused 409 `model_not_downloaded` (error.model, bytes, fits, pull). Read on every request, so the page's checkbox needs no restart | serve |
| `NS_CHAT_UI_PORT` | 8080 | Open WebUI's port (the page's link) | serve |
| `NS_CORS` | loopback origins only | more web origins that may call the servers from a browser, comma-separated (`*`: any) | cli (llm, image, audio start) |
| `NS_GPU_LOCK` | /tmp/ds2run.lock | the lock file (flock) GPU work the front door does not manage holds while it runs - tests, benchmarks, builds: while it is held no model loads (requests get the retryable error `cards_taken`, 503, retry_after_s; /v1 Retry-After), image / audio / 3D jobs wait, the video queue pauses, the switcher swaps in no chat model. `off`: none | serve, switch |
| `NS_IDLE_MINUTES` | 10 | an enabled model unloads after this long without a request (0: never); the switcher uses the same unless --idle-minutes | serve, switch |
| `NS_IMAGE_PORT` | 8086 | the image server's port, as the front door reaches it | serve |
| `NS_IMAGE_PORT2` | 8096 | the second image server's port (`image start --slot 1`), as `NS_AUDIO_PORT2` | serve |
| `NS_PORT` | 8085 | the chat server's port (also the url the registry gives chat entries) | cli (llm start), serve, switch |
| `NS_SERVE_PORT` | 8000 | the front door's port; `audio voices design` calls it there | serve, cli (audio) |
| `OWUI_TOKEN` | a sign-in (works as is with WEBUI_AUTH=False) | Open WebUI's admin token for `glue/openwebui/install.sh` (it installs the pp / tg speed filter) | glue (openwebui install) |
| `NS_SWITCH_PORT` | 8001 | the model switcher's port, as the front door reaches it (the switcher itself listens on --port, 8001) | serve |

## Model registry and catalog

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_MODEL_CARDS` | dist/model-cards, else dist/../model-cards | the model cards' directory (`<kind>/<id>.yml`): what `models search` and `pull` offer, the prompt guides | cli (models search, pull), serve |
| `NS_REGISTRY` | ~/.config/nextsycl/models.json | the model registry (`nextsycl models`) | cli, serve, switch, studio, llm, image, audio, video |
| `NS_STUDIO_MODES` | none (no sync) | the H3 studio's llm-modes file: each registry change rewrites its nextsycl modes (one a model) | cli (models), serve |

## llm engines

### The chat server: `nextsycl llm start`

Settings the program turns into the chat server's flags and container; a registry entry's own values win.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_CACHE_DIR` | ~/.cache/nextsycl/prompts | where prompt-cache checkpoints go when pushed out of host memory (mounted as /cache) | cli (llm start) |
| `NS_CACHE_DISK_GIB` | 32 | the prompt cache's disk budget (0: no disk tier) | cli (llm start) |
| `NS_CACHE_TTL_HOURS` | 24 | how long disk checkpoints are kept | cli (llm start) |
| `NS_CTX` | 65536 | tokens of context a session, or one a session: 262144,32768 (--ctx) | cli (llm start) |
| `NS_EFFORT` | low | default reasoning effort (--effort) for requests that name none; a model's registry entry (its card's env) may set its own | cli (llm start) |
| `NS_GPUS` | all | the chat server's GPUs when the command and the model do not say ("0 1" or "0,1") | cli (llm start) |
| `NS_HOST` | 127.0.0.1 | the chat server's address (0.0.0.0: the network) (--host) | cli (llm start) |
| `NS_KEEP_REQUESTS` | 100 | ended requests kept for ps and inspect (--keep-requests) | cli (llm start) |
| `NS_MAX_TOKENS` | to the context's end | tokens a request without max_tokens may make (--max-tokens) | cli (llm start) |
| `NS_THINK_BUDGET` | none | a thinking request's tokens of thought at most: then `</think>` is its next token, forced, and it answers (a model's registry entry sets it; needs `</think>` to be one token of the vocabulary) | serve (llm server) |
| `NS_MODEL` | /models/glm53-iq2/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf | the model file when no registry entry names one (--model) | cli (llm start) |
| `NS_NAME` | glm-5.3-flash-uncensored | the model name the chat server answers as (--name) | cli (llm start) |
| `NS_NO_MTP` | off | 1: decode without the draft block (--no-mtp) | cli (llm start) |
| `NS_PARALLEL` | 2 | requests decoded together (--parallel) | cli (llm start) |
| `NS_PROMPT_CACHE_MIB` | 4096 | host memory for the prompt cache, MiB (0: off) (--prompt-cache-mib) | cli (llm start) |
| `NS_QW_*` | none | every NS_QW_ setting in nextsycl.conf or the environment is passed into the chat server's container (a registry entry's own value wins) | cli (llm start) |
| `NS_DS2_*` | none | every NS_DS2_ setting (the deepseek2 engine's) is passed into the chat server's container the same way | cli (llm start) |

### Every llm engine

The llm kind's common options (`nextsycl_llm::COMMON_OPTIONS`); what each engine makes of them differs, as noted.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_PROFILE` | off | Time the passes by section (1: GPU synced at each boundary; gpu: device timestamps, no syncs; deepseek2: 1 its decode passes, prompt its prompt passes) (--opt-profile) | llm glm5next, llm qwen35 (NS_Q35_PROFILE needs gpu), llm deepseek2, llm minimaxm3, kernels:ns (gpu: profiling queues) |
| `NS_SPLIT` | even by layer count (glm5next); planned by free VRAM (qwen4exp, qwen35) | Layers per GPU. glm5next: A,B,... layer counts of each GPU but the last; qwen4exp, qwen35: the first layer of each GPU after the first (same for two GPUs) (--opt-split) | llm glm5next, llm qwen4exp, llm qwen35 |
| `NS_VRAM_GUARD_GIB` | glm5next: off; qwen4exp: 1.5 | VRAM (GiB) to keep free. glm5next: a prompt read stops with an error once a GPU has less free after a chunk; qwen4exp: the load plan's margin a GPU (--opt-vram-guard-gib) | llm glm5next, llm qwen4exp; cli (passes it to the container) |

### glm5next

GLM-5.3 (`llm/glm5next`, `kernels/llm/glm5next`; docs/glm5next.md).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_ARENA_MIB` | max(1 GiB, ~660 KiB a token of the prompt chunk) | MiB of each GPU's arena for the prompt path's temporaries (--opt-arena-mib) | llm glm5next |
| `NS_DECODE_DIRECT` | 0 | Experiment: 1 = decode reads host-slot experts in place instead of making them resident | llm glm5next |
| `NS_DECODE_LANES` | 4 | Experiment: lanes a row in decode's grouped MoE kernels (also a verify pass's) | llm glm5next |
| `NS_DENSE_F16` | 1 | Experiment: 0 = only the experts take the fp16 XMX prompt path, not the dense matrices | llm glm5next |
| `NS_DRAFTS` | 1 | Drafts a verify pass checks: 1, or 2 (the draft block chained on its own output) (--opt-drafts) | llm glm5next |
| `NS_ESIMD_MAX` | the GPU's limit (B70 256, B65 128) | Experiment: most tokens of an expert the ESIMD gate/up kernel takes; 0 = never | llm glm5next |
| `NS_EXPERT_PROFILE` | none (cli service sets /cache/expert-profile-<model id>.txt, a new one first copied from the shared expert-profile.txt) | File of layer expert weight lines: which experts fill VRAM first at load; the server writes decode's requests to it at stop (--opt-expert-profile) | llm glm5next, serve (llm server, saves it) |
| `NS_FREE_MISSES` | 0 | Debug: 1 = decode's expert misses read resident slots instead (WRONG output; the speed with no copies) | llm glm5next |
| `NS_FUSED_DOWN_MAX` | 128 | Experiment: most tokens of an expert the fused down kernel takes in prompt chunks (Q2_K only); 0 = never | llm glm5next |
| `NS_FUSED_MAX` | 64 | Experiment: most tokens of an expert the fused gate/up kernel takes in prompt chunks (oneMKL above); 0 = never | llm glm5next |
| `NS_HC_FUSED` | 1 | Experiment: 0 = the hyper-connection pre-mix as separate steps, not the fused two-launch kernel | llm glm5next |
| `NS_KDA_COLS` | by GPU size (4 at 200+ compute units, else 8) | Experiment: columns a sub-group in the KDA scan kernel: 1, 2, 4 or 8 | kernels:llm glm5next |
| `NS_KV` | q8 | The MLA latent cache's form: q8 (544 bytes a token and layer) or f16 (1,024) (--opt-kv) | llm glm5next; cli (passes it to the container) |
| `NS_LEND` | 1 | 0 = the prompt arena does not hold experts while no chunk needs it (--opt-lend) | llm glm5next |
| `NS_MLA_DEC` | 1 | Experiment: 0 = decode widths take the generic MLA attention kernel, not the 512-latent decode one | kernels:llm glm5next |
| `NS_MLA_HG` | 1 head a work-group | Experiment: 4 = 4 heads a work-group in prompt-chunk MLA attention (measured slower) | kernels:llm glm5next |
| `NS_MLA_KERNEL` | 0 | Experiment: 1 = prompt chunks take the per-row MLA attention kernel, not the GEMM path | llm glm5next |
| `NS_MMVQ_IQ_FIRST` | 0 | Experiment: 1 = decode mat-vec tries the IQ kernels before the tuned dense ones | kernels:llm glm5next |
| `NS_NGRAM` | 0 | Up to K (1..4) tokens drafted from the context's repeats (prompt lookup); 0 = the draft block's (--opt-ngram) | llm glm5next |
| `NS_PIPELINE` | 1 | 0 = a long prompt's chunks on two GPUs run in turn, not pipelined (--opt-pipeline) | llm glm5next |
| `NS_PIPE_TRACE` | 0 | Debug: each pipeline stage's time a chunk and the GPUs' free VRAM | llm glm5next; cli (passes it to the container) |
| `NS_PREFETCH` | 8 | Experts swapped in ahead of their layer, at most, a layer; 0 = none (--opt-prefetch) | llm glm5next |
| `NS_PREFETCH_P` | 0.6 | Experiment: chance of use from which a guessed expert is fetched ahead | llm glm5next |
| `NS_PREFILL_CHUNK` | 6144 (64..8192) | Tokens a prompt chunk (--opt-prefill-chunk) | llm glm5next; cli (passes it to the container) |
| `NS_PROFILE_PART` | all parts | Debug: with NS_PROFILE, only part (GPU) i's sections | llm glm5next |
| `NS_PROMPT_DEQUANT` | 0 | Experiment: 1 = prompt chunks take the float32 expanded-GEMM MoE path | llm glm5next |
| `NS_PROMPT_F16_MIN` | 1024 | Experiment: prompt chunks of this many tokens or more multiply in fp16 on the XMX units; 0 = never | llm glm5next |
| `NS_PROMPT_LANES` | 32 | Experiment: lanes a row in prompt chunks' grouped MoE kernels | llm glm5next |
| `NS_SPEC_DRAFT_TEMP` | 1.0 | The draft distribution's temperature relative to the request's (--opt-spec-draft-temp) | llm glm5next |
| `NS_SPEC_SAMPLING` | 1 | 0 = argmax drafts, accepted when the sampled token is them, instead of speculative sampling (--opt-spec-sampling) | llm glm5next |
| `NS_SPLIT_COPY` | 1 | Experiment: 0 = an arriving expert's gate/up waits until its down is in too | llm glm5next |
| `NS_LOAD_SEQ` | 0 | 1 = the GPUs' parts load one after the other; 0: all at once, a thread each (the pinned host mirror is one pool they claim from as they need it) | llm glm5next |
| `NS_SWAP_WAIT` | 0 | Experiment: 1 = decode's expert swaps wait for the GPU's queued work first (not needed) | llm glm5next |
| `NS_TRACE_EXPERTS` | off | Debug: FILE to append every decode expert request to (GPU, tick, layer, experts) for offline policy studies | llm glm5next |
| `NS_TRACE_GUESS` | off | Debug: FILE to append the next two layers' router guesses on this layer's input (16 best a row), for scoring prefetch | llm glm5next |

### qwen4exp

Qwen3.8-Flash-Next on the Strata kernels (`llm/qwen4exp`, `kernels/llm/qwen4exp`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_QW_CHUNK` | 2048 | a prompt chunk's tokens (at least 16; ~0.4 MiB a token on each GPU) (--opt-chunk) | llm qwen4exp |
| `NS_QW_COUPLED` | 1 | 0: a sampled request's draws on the host and the drafts the draft layer's argmax (Strata's default); else on the GPU with drafts coupled to the draws (--opt-coupled) | llm qwen4exp |
| `NS_QW_CVEC` | none | control vectors applied: FILE[:SCALE],... (controlvector GGUFs: the refusal or speed projection) (--opt-cvec) | llm qwen4exp |
| `NS_QW_CVEC_DIR` | per-layer | per-layer: each layer's own direction; single:L: layer L's for all (--opt-cvec-dir) | llm qwen4exp |
| `NS_QW_CVEC_LAYERS` | 1 to the last layer | FIRST,LAST: the layers the control vector steers (--opt-cvec-layers) | llm qwen4exp |
| `NS_QW_CVEC_MODE` | project | how the control vector applies: project or add (--opt-cvec-mode) | llm qwen4exp |
| `NS_QW_EAGER` | graphs on | Debug: 1 launches every kernel as it runs, no SYCL graphs (the A/B) | kernels:qwen4exp |
| `NS_QW_I8` | 1 | 0: the prompt experts dequantized to FP16 and multiplied by oneMKL; else the silo's fused int8 DPAS products where the card runs them (Battlemage; gate/up IQ3_XXS/IQ3_S/IQ2_S, down IQ4_NL/Q2_0) | llm qwen4exp |
| `NS_QW_I8_ROWS` | 128 | an expert group with this many rows an expert or more takes the FP16 GEMMs (faster there); 0 = int8 always | llm qwen4exp |
| `NS_QW_I8_DBG`, `NS_QW_I8_MT`, `NS_QW_I8_WG`, `NS_QW_I8_GRF`, `NS_QW_I8_FORCE` | off / auto | the int8 kernels' bring-up switches (Strata's STRATA_I8_*): debug skips, rows a thread, work-group kernels, the register mode, FORCE=0 off | kernels:qwen4exp silo |
| `NS_QW_EXPERT_PROFILE` | layer order | expert profile file ranking which experts fill VRAM first when not all fit (--opt-expert-profile) | llm qwen4exp |
| `NS_QW_MTP` | no drafts | the MTP draft layer: Strata's runtime directory (--opt-mtp) | llm qwen4exp |
| `NS_QW_PLE_PREFETCH` | 1 | 0: a window's per-layer embedding rows read one after another, not asked for together first (fadvise WILLNEED) (--opt-ple-prefetch) | llm qwen4exp |
| `NS_QW_PROFILE` | off | Debug: 1 prints each decode round's time by part on stderr (costs ~9%) (--opt-qw-profile) | llm qwen4exp, kernels:qwen4exp |
| `NS_QW_SELECT_GEMM` | from 8K cells | the prompt's QSA block scores as oneMKL GEMM tiles from 2,048 blocks of context (below, the warp kernel is faster); 1 always, 0 never | kernels:qwen4exp |
| `NS_QW_SPEC` | 4 | the speculative window: the token and up to N - 1 drafts (2 to 8; Strata's --spec) (--opt-spec) | llm qwen4exp |
| `NS_QW_SPEC_MIN_P` | 0.5 | a draft enters the window while at least this likely under the draft layer (--opt-spec-min-p) | llm qwen4exp |
| `NS_QW_WINDOWS` | off | Debug: 1 reads a prompt in decode windows instead of chunks (a check) (--opt-windows) | llm qwen4exp |

### qwen35

Qwen3.8-27B dense (`llm/qwen35`, its silo `kernels/llm/qwen35/silo`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_NH_CHUNK` | auto | A prompt pass's rows in the nemotronh engine (default the largest of 2048, 1024, 512 that fits; with its silo - without it passes of 8) (--opt-chunk) | llm nemotronh |
| `NS_NH_DENSE` | 1 | 0: the nemotronh engine's dense decode products on the shared mmvq instead of its silo's | llm:nemotronh |
| `NS_NH_DENSE_GROUPS` | 512 | Debug: the work-groups of the nemotronh silo's dense products with a table | kernels:nemotronh silo |
| `NS_NH_DENSE_ROWS` | 0 | Debug: 1, 2 or 4 rows a sub-group in every 256-block type of the nemotronh silo's dense products (0: each type's own) | kernels:nemotronh silo |
| `NS_NH_MOE_MT` | 64 | Debug: rows a tile of the nemotronh silo's prompt expert products (32 or 64) | kernels:nemotronh silo |
| `NS_NH_MOE_XMX` | 1 | 0: the nemotronh prompt pass's experts expanded to half one at a time, then the GEMM (else one XMX launch a projection) | llm:nemotronh |
| `NS_NH_DRAFTS` | 2 | Tokens the nemotronh engine drafts a step with its MTP block (at most 7; 0: none; the MTP block needs its silo) (--opt-drafts) | llm nemotronh |
| `NS_NH_HOST` | auto | MoE layers (from the last back) whose experts the nemotronh engine keeps in pinned host memory on one GPU (default: as few as fit; the draft block's first) | llm nemotronh |
| `NS_NH_SCAN` | 1 | 0: the nemotronh engine's prompt rows through the Mamba scan row after row instead of in 64-row chunks | llm:nemotronh |
| `NS_NH_SPEED` | units x clock | The nemotronh engine's relative speed of each GPU, in order ('1,1.6'): its layers split by it (as far as each GPU's memory goes) | llm nemotronh |
| `NS_NH_PIPELINE` | 1 | 0: a prompt of several chunks over several GPUs chunk after chunk, not as a pipeline (each GPU on its own chunk at once) | llm nemotronh |
| `NS_NH_PROFILE` | off | Debug: 1 waits for each kernel of the nemotronh engine and times it by name (the wall time it costs, launch overhead included; the waits slow the pass), the table printed after a run | llm nemotronh |
| `NS_NH_MULTI_GROUPS` | 1024 | Debug: the work-groups of a MoE layer's experts in one launch (the nemotronh silo) | kernels:nemotronh silo |
| `NS_PEAK_GBS` | 608 | Debug: memory bandwidth (GB/s) the NS_Q35_PROFILE tables show each kernel's share of | llm qwen35 |
| `NS_PEAK_TF16` | 183 | Debug: f16 TFLOP/s peak the NS_Q35_PROFILE tables compare the f16 GEMMs with | llm qwen35 |
| `NS_PEAK_TF32` | 22.9 | Debug: f32 TFLOP/s peak the NS_Q35_PROFILE tables compare the other kernels with | llm qwen35 |
| `NS_Q35_CHUNK` | 2048 rows when they fit beside the weights and a session, else 1024 or 512 | a prompt pass's rows (at least 16) (--opt-chunk) | llm qwen35 |
| `NS_Q35_DRAFTS` | 3 | tokens drafted a step with MTP or EAGLE-4 (at most 7; 0: none) (--opt-drafts) | llm qwen35 |
| `NS_Q35_EAGLE` | unset (MTP) | the EAGLE-4 draft's folder (config.json, model.safetensors: Qwen3.8-27B's `LlamaForCausalLMEagle4`), or `auto`: an `eagle4/` folder beside the model; loaded in MTP's place on one GPU (--opt-eagle) | llm qwen35 |
| `NS_Q35_EAGLE_HEAD` | low | the EAGLE-4 draft's head: low (its low-rank head's 128 candidates scored by the full head, as trained: 35 MB read a draft step) or full (the full head's argmax: 1.35 GB) | llm qwen35 |
| `NS_Q35_EAGLE_CTX` | 32768 | positions the EAGLE-4 draft's cache holds (20 KiB each); no drafts past them (--opt-eagle-ctx) | llm qwen35 |
| `NS_Q35_MMVQ_GROUPS` | 512 | Experiment: work-groups of the silo's matrix-vector kernels with a lookup table | kernels:qwen35 silo |
| `NS_Q35_MMVQ_ROWS` | 0 (each type's own) | Experiment: rows a sub-group (1, 2 or 4) for every 256-block type of the silo's matrix-vector kernels | kernels:qwen35 silo |
| `NS_Q35_MTP` | the first .gguf in an MTP/ folder beside the model | the MTP draft block's GGUF (--opt-mtp) | llm qwen35 |
| `NS_Q35_VISION` | an `mmproj*.gguf` beside the model | the vision tower's file (the model's mmproj GGUF, Qwen3-VL's ViT and merger: pictures in chat); 0: none (the model reads no pictures) (--opt-vision) | llm qwen35 |
| `NS_Q35_IMAGE_TOKENS` | 4096 | the most prompt positions a picture takes (2 x 2 patches of 16 pixels each; larger pictures are scaled down to fit, and to what the prompt arena holds) (--opt-image-tokens) | llm qwen35 |
| `NS_Q35_VISION_DUMP` | unset | PREFIX: a picture's patches and rows written as raw float32 to PREFIX.patches.f32 and .rows.f32 (a check against llama.cpp's mtmd: reference/llama-dump/ns-mtmd-dump.cpp) | llm qwen35 |
| `NS_Q35_CHAIN_HEADS` | mean | LogProbChain (`logprob_chain`): how a full-attention layer's heads combine into a position's attention - their mean, or `max` | llm qwen35 |
| `NS_Q35_GRAPHS` | 0 | 1: decode passes' position-free stretches (all but the full-attention layers' rotary positions, cache writes and attention) as SYCL graphs, recorded once a session and row count, replayed after - the same tokens, no faster on the B70 / B65 (the host keeps ahead of the GPU) | llm qwen35 |
| `NS_Q35_KV` | f16 | the attention cache's form a card: f16 - half (512 bytes a head's row) everywhere, the fastest; auto - q8_0 (272) on the card the plan fills first (the last: one card, that one), half on the others where it fits; auto-q4 - q4_0 (144) there instead; q8 (q8_0), q4 (q4_0) - that on every card. A quantized cache needs the silo (without it: half). B70, Q4 with MTP: one session holds 196K tokens in half, 368K in q8_0, 704K in q4_0; against half after a 12.8K prompt, KL q8_0 0.00024, q4_0 0.0032 (noise 0.00014: `llm generate --score`) | llm qwen35 |
| `NS_Q35_KV_ROT` | 0 | 1: a quantized cache rotated (a 32-value Hadamard transform a block on keys, values and queries, the attention's output turned back) | llm qwen35 |
| `NS_Q35_PROFILE` | off | Debug: 1 times every kernel on the device with its bytes and arithmetic (needs NS_PROFILE=gpu; one GPU only) | llm qwen35 |

## Image

Qwen-Image 2.1 (`image/qwenimage21`), HunyuanImage 2.1 (`image/hunyuanimage21`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_IMAGE_OUT` | ~/.local/share/nextsycl/images | where pictures are saved | cli (image) |
| `NS_HI_*` | - | every NS_HI_ setting in nextsycl.conf is passed to the image server | cli (image serve) |
| `NS_HI_GUIDANCE` | apg | the HunyuanImage base model's guidance: apg (the pipeline's MIX_mode_0: CFG through step 5, or 38 when the negative quotes text, then adaptive projected guidance) or cfg (plain CFG every step) (--opt-guidance; per request too) | image hunyuanimage21 |
| `NS_HI_INT8` | 0 | 1: the DiT's products in int8 x int8 on the silo (rows of both factors turned by a 256-block Hadamard, one scale a row). Approximate: the distilled 512 check's latents 2.0e-2 from the reference against 2.5e-3 exact; the same picture with other details at 2048x2048 (PSNR 24.7 dB against the exact one; 1024x1024 39 dB). B70: distilled 1024 2.5 -> 2.0 s, 2048 7.8 -> 5.9 s; base 1024 20.2 -> 14.4 s (--opt-int8) | image hunyuanimage21 |
| `NS_HI_PROFILE` | off | Debug: set: a GPU wait after each named step of the DiT, the totals printed after the picture (slower; shows where the time goes) | image hunyuanimage21 |
| `NS_MOSS_*` | - | every NS_MOSS_ setting in nextsycl.conf is passed to the image server | cli (image serve) |
| `NS_MOSS_EXPAND` | auto | MOSS-Image's DiT blocks expanded from their quantized form to half on the card once at load (the per-forward expansion then goes: ~9 % of a forward; the same values): auto keeps 8 GiB free after each block (a 32 GiB card holds all 42: 10.2 GB of half beside the 9.5 GB text encoder), 0 never (the quantized matrices stay, 3.4 GB), 1 all of them | image mossimage |
| `NS_MOSS_STEPS` | 50 | MOSS-Image's default steps for a request that names none (a turbo LoRA's card sets 4 or 8) | image/mossimage |
| `NS_MOSS_CFG` | 3 | MOSS-Image's default guidance (a turbo LoRA's card sets 1: no unconditional branch) | image/mossimage |
| `NS_MOSS_BENCH_SIZE` | 1024 | Debug: the picture side the `image check --stages bench` stage times a forward and the VAE at (a multiple of 16) | image mossimage (check) |
| `NS_MOSS_BENCH_REPEAT` | 1 | Debug: the bench stage's prompt said this many times over (a long prompt through the text encoder, up to 4096 tokens) | image mossimage (check) |
| `NS_MOSS_PROFILE` | off | Debug: set: a GPU wait after each part of a text-encoder layer, the seconds in full attention, DeltaNet and the FFN printed on stderr (slower) | image mossimage |
| `NS_QI_*` | - | every NS_QI_ setting in nextsycl.conf is passed to the image server | cli (image serve) |
| `NS_QI_DUMP` | off | Debug: a directory: what the denoiser is given (embeds, pads, cond latents, noise) written there as raw float32 files | image qwenimage21 |
| `NS_QI_INT8` | 0 | 1: the DiT's block matrices in int8 ConvRot (~1.4x a step, half the VRAM) (--opt-int8) | image qwenimage21 |
| `NS_QI_PROFILE` | off | Debug: set: a GPU wait after each named step, the totals printed (slower; shows where the time goes) | image qwenimage21 |
| `NS_QI_SIGMAS` | the model's own schedule | a few-step preset: noise levels from 1 down, comma separated (their count is the steps) (--opt-sigmas; per request too) | image qwenimage21 |
| `NS_QI_SIGMA_SHIFT` | dynamic | dynamic: the picture size's shift on the NS_QI_SIGMAS levels; none: as given (--opt-sigma-shift; per request too) | image qwenimage21 |

## Video

The video daemon and the studio (`nextsycl video`) and the H3 engine (`video/h3`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `H3_*` | - | every H3_ setting in nextsycl.conf is passed to the video daemon's container (and set before the GPU opens) | cli (video serve), video |
| `H3_STALL_S` | 1200 | the studio's watchdog: a running job that writes nothing this many seconds is stopped | studio |
| `H3_TE_PIN` | 1 | 0: no pinned host copy of the text encoder between clips (two pinned layer slots instead) (--opt-te-pin) | video h3 |
| `H3_TE_PIN_LOCK` | /tmp/h3-te-pin.lock | the lock file that lets one worker a host hold the pinned text-encoder copy | video h3 |
| `H3_THREADS` | 8 | threads reading the checkpoint at load (--opt-threads) | video h3 |
| `H3_VAE_ATTN_PER_TILE` | off | 1: the video decoder's attention a tile at a time (--opt-vae-attn-per-tile) | video h3 |
| `NS_VIDEO_CHARACTERS` | docs/video/characters.example.json | characters for the speech tool | cli (video tools) |
| `NS_VIDEO_ENGINES` | none | the video daemon's other models, space-separated registry ids (serve: the enabled video models decide) | cli (video start), serve |
| `NS_VIDEO_GPUS` | all | the video daemon's GPUs, e.g. "0 1" (serve: the enabled video models' GPUs) | cli (video start) |
| `NS_VIDEO_GPUSTAT` | /run/gpustat.json | the GPU telemetry file (`nextsycl gpustat`, root: every user's processes) the studio reads; the front door plans loads with its `vram_used_mb` per card where it sees more than its own sampler | cli (video serve), studio, serve |
| `NS_VIDEO_GPU_LOCK` | none | a lock file shared with the card's other users (mounted, passed as --gpu-lock) | cli (video start) |
| `NS_VIDEO_IDLE` | 600 | seconds without a job before a video worker unloads and lets its card go (0: never) | cli (video start) |
| `NS_VIDEO_LISTEN` | 127.0.0.1 | the studio's address | cli (video serve) |
| `NS_VIDEO_LLM_MODES` | none | the studio's chat modes file (the language models it switches) | cli (video serve), studio |
| `NS_VIDEO_LLM_SWITCHER` | none | a model switcher URL: its chat model stops before a video load on a shared card and comes back after | cli (video start) |
| `NS_VIDEO_MODEL` | the first video model | the model jobs get unless they name one: a registry id (serve: the enabled video models decide) | cli (video start), serve |
| `NS_VIDEO_MODELS_DIR` | none | a directory mounted as /models in the video container: what clips, templates and scene files name as /models/... (upscalers, LoRAs) | cli (video start) |
| `NS_VIDEO_OUT` | ~/.local/share/nextsycl/video | where clips go | cli (video), studio |
| `NS_VIDEO_PORT` | 8095 (`video serve`); the front door assumes 8090 | the studio's port | cli (video serve), serve |
| `NS_VIDEO_SHARED_GPUS` | none | GPUs shared with the chat model: a clip there stops chat first | cli (video start) |
| `NS_VIDEO_STUDIO` | http://127.0.0.1:8095 | the studio the video tools talk to | cli (video tools) |
| `NS_VIDEO_STUDIO_DIR` | ~/.local/share/nextsycl/studio | the studio's state (queue, projects, plan.json) | cli (video serve, plan), studio |
| `NS_VIDEO_TEMPLATES` | none | prompt templates for the studio | cli (video serve), studio |

## Audio

ACE-Step 1.5 (`NS_ACE_`), MiniMax Music 3 (`NS_MM3_`), Qwen3-TTS (`NS_Q3T_`), VoxCPM2 (`NS_VCP_`), Whisper (`NS_WH_`). A per-request option's variable is only its name: it rides in the request (`"options"`), the engine does not read the environment for it.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_AUDIO_OUT` | ~/.local/share/nextsycl/audio | where songs, speech and saved voices are kept | cli (audio), serve |
| `NS_ACE_CHECKPOINT` | acestep-v15-turbo | the ACE-Step DiT checkpoint a model directory loads (acestep-v15-base: the base model, for lego) | audio acestep |
| `NS_ACE_DCW` | the turbo's: 1, the base's: 0 | the wavelet-domain correction of each step (DCW; as the reference: on for turbo) (--opt-dcw) | audio acestep |
| `NS_ACE_DECODE_CHUNK` | by the card | the VAE decode's window in latent frames, 128-512 (the reference's for the card's free memory: 256 from 12 GB, 128 below) (--opt-decode-chunk) | audio acestep |
| `NS_ACE_NORMALIZE` | 1 | 0: the song is not brought to a -1 dB peak (--opt-normalize) | audio acestep |
| `NS_ACE_PLANNER` | on | 0: the 5 Hz planner LM is not loaded (no `planner` requests; less VRAM) | audio acestep |
| `NS_ACE_SHIFT` | 3.0 | the schedule's shift (t' = s t / (1 + (s - 1) t)) (--opt-shift) | audio acestep |
| `NS_ACE_VAE_SAMPLE` | 1 | 0: a recording's VAE posterior taken at its mean, not sampled as the reference (--opt-vae-sample) | audio acestep |
| `NS_MM3_*` | - | every NS_MM3_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_MM3_DIT_INT8` | 0 | 1: the flow transformer's block matrices in int8 ConvRot (--opt-dit-int8) | audio minimaxmusic3 |
| `NS_MM3_FPMATH` | strict | the decoder's oneDNN convolutions' math: strict (float32), f16, bf16 or tf32 | kernels:audio (minimaxmusic3) |
| `NS_MM3_INT8` | 0 | 1: the language model and depth decoder in int8 (half the VRAM, ~1.7x the frames a second) (--opt-int8) | audio minimaxmusic3 |
| `NS_MM3_NSD_CONV` | 0 | Debug: 1: the vocoder on the shared plain-loop convolutions (exact float32, slow) instead of oneDNN's | audio minimaxmusic3 |
| `NS_Q3T_*` | - | every NS_Q3T_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_Q3T_GREEDY` | 0 | the per-request option greedy's name: most likely code, no repetition penalty; a request's options only, the variable itself is not read (--opt-greedy) | audio qwen3tts |
| `NS_Q3T_INT8` | 0 | 1: the talker's and code predictor's matrices in int8 (less VRAM, faster frames) (--opt-int8) | audio qwen3tts |
| `NS_Q3T_INT8_ROWS` | 0 | 1: int8 with a scale a row (as without the silo), not a scale every 32 weights | audio qwen3tts |
| `NS_Q3T_PROFILE` | 0 | 1: each request's stages timed on stderr (the request, the prompt's build and prefill, the frames - with the silo the graph's recording and the time a frame; without it each stage synchronised -, the codec) | audio qwen3tts |
| `NS_Q3T_STREAM_FIRST` | 8 | a streamed answer (stream / stream_format) hands out its first sound after this many frames (8: 0.64 s of speech); then at 25 frames and where the codec's chunk lengths fall | audio qwen3tts |
| `NS_LIB_FROM_DISK` | unset: the kernel libraries are read from inside the binary | `1`: open `libnextsycl-<kind>.so`, `libdnnl.so.3` and `libnextsycl-flash.so` from the files beside the executable instead (rebuild a kernel without relinking the program) | runtime (every kind) |
| `NS_LIB_DIST` | `dist/` beside the repository | build time: the directory `crates/sys/build.rs` embeds the kernel libraries from | build |
| `NS_FLASH_LIB` | set by the program to the embedded flash-attention library's `/proc/self/fd/N` path; unset: the file beside the kernel library | the path of `libnextsycl-flash.so` the diffusion kernels open | kernels:diffusion |
| `NS_WFE_FROM_DISK` | unset: the front ends are read from inside the binary | `1`: read the pages and modules from `dist/wfe` instead of the embedded copy (edit a page and reload, no rebuild) | serve, studio, image/audio/model3d servers |
| `NS_WFE_DIST` | `dist/wfe` beside the repository | build time: the directory `glue/serve/build.rs` embeds into the binary | build |
| `CARGO_MANIFEST_DIR`, `OUT_DIR` | set by cargo | where `glue/serve/build.rs` finds the repository and writes the list of embedded front-end files | build |
| `NS_MM31_CHUNK` | 4096 | The minimaxm3 engine's prompt pass rows: every pass reads the experts that are off the cards once, so a wider pass reads less per token (--opt-chunk) | llm:minimaxm3 |
| `NS_MM31_SPARSE_ALL` | 0 | 1: the minimaxm3 engine's first three layers take the sparse-attention indexer too (the model config says every layer is sparse; llama.cpp's graph leaves them dense, the reference) (--opt-sparse-all) | llm:minimaxm3 |
| `NS_MM31_VRAM_GIB` | per card: free VRAM less the 1.5 GiB arena, the sessions' caches (the load's token count x 138 KB) and 2.5 GiB | The minimaxm3 engine's budget, per card, for routed experts kept in VRAM (placed at load, hottest first by the expert profile; the rest are read from the file for each pass) (--opt-vram-gib) | llm:minimaxm3 |
| `NS_MM31_PIN_GIB` | MemAvailable less 16 GiB, at most 32 (in all, shared by the cards in proportion to what their VRAM cannot hold) | The minimaxm3 engine's budget for routed experts kept in pinned host memory (the kernels read them over PCIe) (--opt-pin-gib) | llm:minimaxm3 |
| `NS_MM31_SPLIT` | half the layers | With two GPUs, how many layers the first runs (the minimaxm3 engine's layer split) (--opt-split) | llm:minimaxm3 |
| `NS_MM31_DIRECT` | 1 | 0: the minimaxm3 engine reads experts from the file through the page cache; 1: with O_DIRECT, many reads in flight (--opt-direct) | llm:minimaxm3 |
| `NS_MM31_SPLITREAD` | 8 | Pieces an expert matrix is read in at once by the minimaxm3 engine (each its own request, one thread each), so a miss reads with several requests in flight (--opt-splitread) | llm:minimaxm3 |
| `NS_MM31_LANE` | 1 | The copy lane (0 or 1) the minimaxm3 engine copies pinned and file-read experts into VRAM on, beside the compute queue (--opt-lane) | llm:minimaxm3 |
| `NS_MM31_SPLIT_ATTEND` | 1 | 0: decode passes run the prompt-pass attention kernel instead of the silo's key-split one (--opt-split-attend) | llm:minimaxm3 |
| `NS_MM31_FUSED` | 1 | 0: the minimaxm3 decode experts run as the shared kernels' three products each instead of the silo's fused gate+up and down launches (--opt-fused) | llm:minimaxm3 |
| `NS_MM3S_RPS` | 2 | Debug: output rows a sub-group of the minimaxm3 silo's expert products works at once (1 or 2) | llm:minimaxm3 |
| `NS_MM3S_KU` | 1 | Debug: blocks of a row the minimaxm3 silo's expert products decode before multiplying (1 or 2) | llm:minimaxm3 |
| `NS_MM31_DYNAMIC` | 1 | 0: the minimaxm3 engine keeps the experts where the load put them; by default decode passes keep the most recently used experts of each layer in VRAM and the next in pinned memory (the used ones swap into VRAM, copied on the copy lanes) | llm:minimaxm3 |
| `NS_MM31_TRACE` | unset | Debug: a file the minimaxm3 engine writes one line a layer of every one-token pass to ("layer expert expert ..."), for cache studies | llm:minimaxm3 |
| `NS_MM31_AHEAD` | 3 | Decode: how many experts of each layer ahead the minimaxm3 engine guesses (their routers on the CPU, in a thread, on this layer's residual stream) and starts reading or copying while this layer computes; 0 = none, at most 12 (--opt-ahead) | llm:minimaxm3 |
| `NS_MM31_AHEAD_LAYERS` | 2 | Decode: how many layers ahead the minimaxm3 engine guesses the routing of (at most 16; 0 = none, and the routers stay off the host) | llm:minimaxm3 |
| `NS_MM31_PRECOPY` | 1 | Decode: 0 = a guessed expert in pinned memory waits for its layer instead of being copied into an empty VRAM place ahead | llm:minimaxm3 |
| `NS_MM31_FREE` | 2 | Decode: empty VRAM places the minimaxm3 engine keeps per layer (1 to 8): a pinned or read expert lands there in one copy, the least recently used are demoted beside the compute | llm:minimaxm3 |
| `NS_MM31_FUSE1` | 1 | 0 = one-token passes run the separate launches, not the silo's fused attention steps | llm:minimaxm3 |
| `NS_MM31_ATTX` | 1 | 0 = a prompt pass's attention on the silo's vector-unit kernel, not its XMX one (half queries) | llm:minimaxm3 |
| `NS_MM31_SILO_HALF` | 1 | 0 = a prompt pass's matrices dequantize to half on the shared kernel, not the silo's | llm:minimaxm3 |
| `NS_MM31_HALF_PROJ` | 1 | 0 = a prompt pass's q, k, v, o and shared-expert products in float32, not half (the half GEMM) | llm:minimaxm3 |
| `NS_MM31_PIPE` | 1 | 0 = a prompt longer than a chunk runs chunk after chunk, with the stages' taps; by default two cards work at once (a chunk on the first while the second has the one before; the taps are not called) | llm:minimaxm3 |
| `NS_MM31_TOKLOG` | unset | Debug: a file the minimaxm3 engine writes a line to for every pass of up to 8 rows: milliseconds, experts from VRAM, pinned memory and the file, rows | llm:minimaxm3 |
| `NS_MM31_STREAM_DIR` | unset | Directories on other drives (a `:` list) for the minimaxm3 engine: a copy of the experts the cards do not keep at load is written to each once (`mm31-stream-<hash>.bin`, ~34 GiB; a copy for another placement in the directory is deleted first) and a read's pieces go round the drives (--opt-stream-dir) | llm:minimaxm3 |
| `NS_MM31_STREAM_ALL` | 0 | 1 = with `NS_MM31_STREAM_DIR`, every expert, not only those off the cards at load (~111 GiB a drive) | llm:minimaxm3 |
| `NS_MM31_STREAM_SHARE` | 100 | With `NS_MM31_STREAM_DIR`, the percentage of the off-card experts read from that drive (--opt-stream-share) | llm:minimaxm3 |
| `NS_MM31_FORCE` | unset | Debug: a llama.cpp dump's directory (`nextsycl llm check`): after each layer the minimaxm3 engine's hidden state is replaced by the dump's `l_out-N`, so every layer is checked on the reference's own input | llm:minimaxm3 |
| `NS_MM31_FILE` | unset: the test is skipped | the first shard of a MiniMax-M3.1-Flash GGUF: `cargo test -p nextsycl-llm-minimaxm3` checks the file against the model description | test minimaxm3 |
| `NS_Q3T_EAGER` | 0 | 1: the silo's frame launched kernel by kernel, not replayed as a recorded graph (the A/B for the graph) | audio qwen3tts |
| `NS_Q3T_STREAMING` | clone 1, built-in and designed voices 0 | the per-request option streaming's name: the text fed a token a frame; a request's options only, the variable itself is not read (--opt-streaming) | audio qwen3tts |
| `NS_Q3T_TEMPERATURE` | 0.9 | the per-request option temperature's name: the talker's sampling temperature; a request's options only, the variable itself is not read (--opt-temperature) | audio qwen3tts |
| `NS_Q3T_TOP_K` | 50 | the per-request option top-k's name: the talker's top-k; a request's options only, the variable itself is not read (--opt-top-k) | audio qwen3tts |
| `NS_BZ2_*` | - | every NS_BZ2_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_BZ2_GREEDY` | 0 | the per-request option greedy's name: the most likely code, backbone and depth decoder (deterministic; may not end); a request's options only, the variable itself is not read (--opt-greedy) | audio breezetts2 |
| `NS_BZ2_BATCH` | 8 | the rows a frame takes for requests made together (a guided request - with an instruction - two, a plain one one); the server takes that many requests at once; 1: one at a time (--opt-batch) | audio breezetts2 |
| `NS_BZ2_TEMPERATURE` | 0.9 | the per-request option temperature's name: the sampling temperature, backbone and depth decoder; a request's options only, the variable itself is not read (--opt-temperature) | audio breezetts2 |
| `NS_BZ2_TOP_K` | 50 | the per-request option top-k's name: the sampling top-k, backbone and depth decoder; a request's options only, the variable itself is not read (--opt-top-k) | audio breezetts2 |
| `NS_S2P_BATCH` | 8 | the requests a frame takes together (the server takes that many at once); 1: one at a time (--opt-batch) | audio s2pro |
| `NS_S2P_GREEDY` | 0 | the per-request option greedy's name: the most likely code, slow and fast models (deterministic; may not end); a request's options only, the variable itself is not read (--opt-greedy) | audio s2pro |
| `NS_S2P_TEMPERATURE` | 0.7 | the per-request option temperature's name: the sampling temperature, slow and fast models; a request's options only, the variable itself is not read (--opt-temperature) | audio s2pro |
| `NS_S2P_TOP_P` | 0.7 | the per-request option top-p's name: the sampling top-p, slow and fast models; a request's options only, the variable itself is not read (--opt-top-p) | audio s2pro |
| `NS_S2P_TOP_K` | 30 | the per-request option top-k's name: the sampling top-k, slow and fast models; a request's options only, the variable itself is not read (--opt-top-k) | audio s2pro |
| `NS_VXT_STEPS` | 7 | the option steps's name: the flow's Euler steps a frame (upstream's 7; fewer is faster, the model is trained for 7); read from a request's options (`audio gen --opt-steps` puts it there too), the variable itself is not read | audio voxtraltts |
| `NS_VXT_ENROLLED` | - | the option enrolled's name: an enrolled voice's fitted codes (experimental; reference/voxtraltts/enroll.py) as the voice; read from a request's options (`audio gen --opt-enrolled` puts it there too), the variable itself is not read | audio voxtraltts |
| `NS_VXT_CHECKPOINT` | - | `audio voice enroll`: the mistralai/Voxtral-4B-TTS-2603 checkout (consolidated.safetensors, params.json) whose codec decoder the fit runs through | cli (audio voice enroll) |
| `NS_VXT_VLLM_OMNI` | - | `audio voice enroll`: a vllm-omni checkout (enroll.py runs its voxtral_tts decoder code) | cli (audio voice enroll) |
| `NS_VXT_TORCH` | - | `audio voice enroll`: PyTorch with XPU installed with `pip install --target` for the build image's Python (its PYTHONPATH) | cli (audio voice enroll) |
| `NS_VXT_ENROLL_TOOL` | <dist>/../reference/voxtraltts | `audio voice enroll`: where enroll.py and ref.py are | cli (audio voice enroll) |
| `NS_VV_GEMM_SIMT` | 0 | Debug: 1 runs the VibeVoice tokenizers' bfloat16 products on the plain tiled kernel instead of the matrix engines (an A/B) | kernels:audio (vibevoice) |
| `NS_VV_GEMV_KS` | by shape | Debug: the VibeVoice products' split of K between a work-group's sub-groups (1, 2 or 4), for every shape (a tuning switch) | kernels:audio (vibevoice) |
| `NS_VV_PROFILE` | 0 | Debug: 1 times a run's stages (logits, head, decoder, semantic encoder, connectors, language model; the GPU waited for at each boundary), the table on stderr at its end; 2 the tokenizers' operations one by one | audio vibevoice |
| `NS_VCP_*` | - | every NS_VCP_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_VCP_CONTINUE` | 0 | 1: VoxCPM2 continues a saved voice's recording (its transcript the prompt's) instead of taking its timbre alone - continuing led the recording's last words into each sentence | audio voxcpm2 |
| `NS_VCP_GRAPH` | 1 | 0: the silo's sampler queued an evaluation at a time instead of replayed as a recorded graph (the A/B for the graph; the host still waits once a patch) | audio voxcpm2 |
| `NS_VCP_SAMPLER_CHECK` | 0 | Debug: 1 (with NS_VCP_GRAPH=0): each of the silo's sampler steps checked against the host's on the same values (stderr: the worst difference; 0 expected) | audio voxcpm2 |
| `NS_VCP_INT8` | 0 | 1: the language models' matrices in int8 (less VRAM, faster patches) (--opt-int8) | audio voxcpm2 |
| `NS_VCP_MAX_RATIO` | 6 | the per-request option max-ratio's name: the most patches a text token (a runaway's bound); a request's options only, the variable itself is not read (--opt-max-ratio) | audio voxcpm2 |
| `NS_WH_*` | - | every NS_WH_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_WH_CONDITION` | 0 | 1: Whisper prompts each 30 s window with the text before it (openai's condition_on_previous_text: a steadier style, but a mistake carries on); also a request's option condition (--opt-condition) | audio whisper |
| `NS_WH_FALLBACK` | 1 | 0: a window that runs on without ending, or decodes unsure (average log-probability under -1), is kept as greedy decoding left it instead of decoded again at rising temperatures (openai's fallback); also a request's option fallback (--opt-fallback) | audio whisper |
| `NS_WH_NO_SPEECH` | 0.6 | a window is skipped as silence when <\|nospeech\|> is likelier than this and its text unsure; also a request's option no-speech (--opt-no-speech) | audio whisper |

## 3D

The `model3d` kind (meshes from pictures, rigging, animation) and its engines' settings, `NS_M3D_*`.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_MODEL3D_OUT` | ~/.local/share/nextsycl/model3d | where meshes (GLB, with their picture and settings beside them) and 3D uploads are kept | cli (model3d), serve |
| `NS_M3D_*` | - | every NS_M3D_ setting in nextsycl.conf is passed to the 3D server | cli (model3d serve) |
| `NS_M3D_HY_TOKENS` | 3072 (2mini), 4096 (2.1) | the shape's latent tokens (Tencent's turbo example samples the turbo decoder's 512: coarser) (--opt-tokens) | model3d hunyuan3d |
| `NS_M3D_HY_DECODE` | fast | fast: the grid decoded coarse to fine, only near the surface (FlashVDM's hierarchical decoding); dense: every point (--opt-decode) | model3d hunyuan3d |
| `NS_M3D_HY_CHUNK` | by the decoder's width | points the shape decoder queries at once (--opt-chunk) | model3d hunyuan3d |
| `NS_M3D_HY_PAINT` | 1 | 0: 2.1 loads without its paint model (no textures; ~6.5 GiB less) even when the paint files are there (--opt-paint) | model3d hunyuan3d |
| `NS_M3D_HY_PAINT_STEPS` | 15 | the texture's UniPC steps (--opt-paint-steps) | model3d hunyuan3d |
| `NS_M3D_HY_PAINT_KEEP_MIB` | 3072, or 30 % of the free memory on a smaller card | the paint run's arena for what outlives a block (MiB); past it a run spills to plain allocations | model3d hunyuan3d |
| `NS_M3D_HY_PAINT_TMP_MIB` | 4096, or 45 % of the free memory on a smaller card | the paint run's arena for a block's temporaries (MiB) | model3d hunyuan3d |
| `NS_M3D_HY_ATTN` | flash | self-attention (DINOv2, the DiT, the shape decoder's transformer, the paint UNets): flash - the strided flash kernel the shared kernels use for cross-attention (half precision, q / k / v read in place); shared - the shared kernels' own choice (oneDNN's fused attention under 8,192 rows, SageAttention's int8 q and k from there: slower and less exact) (--opt-attn) | model3d hunyuan3d |
| `NS_M3D_HY_INT8` | 0 | 1: 2.1's DiT block matrices as int8 ConvRot (rotated by a 256 x 256 Hadamard matrix along their inputs, a scale a row; the activations quantized a row at a time): steps 0.38 -> 0.32 s on the B70, 2.7 GiB less VRAM. Approximate: a CFG step's velocity 4e-2 - 1.6e-1 from Tencent's (half: 2e-3 - 1.4e-2), the shape alike to the eye (2mini: ignored - no faster) (--opt-int8) | model3d hunyuan3d |
| `NS_M3D_HY_PROFILE` | off | Debug: 1 waits for the GPU after each stage of a request (the picture, DINOv2, each DiT step, the decoder's levels and their CPU parts, marching cubes, the paint model's stages) and prints the totals on stderr; 2 also times the parts of the DiT's and the paint UNet's blocks (more waits: slower) | model3d hunyuan3d |
| `NS_M3D_AAM_PROFILE` | off | Debug: set: AnimateAnyMesh waits for the GPU after each named step of the flow and prints the totals (slower; shows where the time goes) | model3d animateanymesh |
| `NS_M3D_AAM_TRAJ` | max(512, vertices / 8) | AnimateAnyMesh: the trajectory tokens DyMeshVAE samples (more: finer motion, slower) (--opt-traj; per request too) | model3d animateanymesh |
| `NS_M3D_UNIMATE_FACING` | auto | UniMate: which way the rig looks (auto, +z, -z, +x, -x): auto reads the sides in the joint names (Left / Right), else finds the rig's mirror symmetry and the feet's direction, else takes +Z; a wrong facing turns the walk sideways or backwards (--opt-facing; per request too) | model3d unimate |
| `NS_M3D_UNIMATE_NAMES` | auto | UniMate: auto cleans the joint names by rules (a name nothing recognizes is Bone); structure leaves the names out, the model reading the skeleton's graph, rest pose and depths alone (--opt-names; per request too) | model3d unimate |
| `NS_M3D_UNIRIG_BEAMS` | 15 | UniRig: the skeleton decoder's beams, sampled as UniRig's inference (1: greedy, the same skeleton every time) (--opt-beams; per request too) | model3d unirig |
| `NS_M3D_UNIRIG_PROFILE` | off | Debug: set: UniRig waits for the GPU after each named step and prints the totals (slower; shows where the time goes) | model3d unirig |
| `NS_M3D_SKINTOKENS_BEAMS` | 10 | SkinTokens: the decoder's beams, sampled as the demo (1: greedy, the same rig every time) (--opt-beams; per request too) | model3d skintokens |
| `NS_M3D_SKINTOKENS_SKIN_END` | upstream | SkinTokens: where the skin tokens end - upstream (the demo's: the last joint's fourth token is the end token, code 0) or fix (the model says it) (--opt-skin-end; per request too) | model3d skintokens |
| `NS_M3D_RIGNET_FACES` | 4000 | RigNet: the triangles the mesh is decimated to before rigging (the weights go back to every vertex of the input) (--opt-faces; per request too) | model3d rignet |
| `NS_M3D_RIGNET_BANDWIDTH` | the learned 0.0429 | RigNet: the joints' mean-shift bandwidth (smaller: more, closer joints) (--opt-bandwidth; per request too) | model3d rignet |
| `NS_M3D_RIGNET_THRESHOLD` | 1e-5 | RigNet: the joints' density threshold (lower: more joints) (--opt-threshold; per request too) | model3d rignet |
| `NS_M3D_RIGNET_PROFILE` | off | Debug: set: RigNet prints each step's time on stderr (the geometry, each network, the mean-shift, the volumetric geodesics) | model3d rignet |
| `NS_M3D_T2_PROFILE` | off | TRELLIS.2 / Pixal3D: set: each stage's time and the card's memory on stderr, and the run's peak (the card's free memory watched every 50 ms) (--opt-profile) | model3d trellis2 |
| `NS_M3D_T2_RESIDENT` | off | TRELLIS.2 / Pixal3D: set: every model stays loaded after its stage (faster repeats; ~11 GiB of weights on the card) - by default each stage loads its model and drops it after (--opt-resident) | model3d trellis2 |

## Diffusion kernels: `NSD_*`

`kernels/diffusion`, shared by the image, audio and video engines. Read with getenv when the kernels first need them: set them in nextsycl.conf (forwarded as `NSD_*`) or with `--opt-NAME` where the engine declares one.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NSD_*` | - | every NSD_ setting in nextsycl.conf is passed to the image, audio and video servers | cli (image serve, audio serve, video serve) |
| `NSD_ATTN` | sage | onednn: long sequences' attention on oneDNN's fused kernel, not SageAttention (int8 q, k) (--opt-attn, video) | kernels:diffusion (image, audio, video) |
| `NSD_ATTN_ROWS` | 8192 | query rows a fused attention call takes (at least 16) (--opt-attn-rows, video) | kernels:diffusion (image, audio, video) |
| `NSD_ATTN_TABLE_MB` | 1536 | the most score-table memory (MiB, at least 64) a chunk of unfused attention may take | kernels:diffusion (image, audio, video) |
| `NSD_FLASH` | on | 0: no flash attention (libnextsycl-flash.so); oneDNN's fused SDPA instead | kernels:diffusion (image, audio, video) |
| `NSD_MEM_FRACTION` | 0.94 | the share of the card (0.05-0.97) the kernels' allocations and scratch may hold (--opt-mem-fraction, video) | kernels:diffusion (image, audio, video) |
| `NSD_NORM_SCALAR` | off | Debug: set: the element-a-work-item norm and RMS-RoPE kernels, not the 16-byte vector ones | kernels:diffusion (image, audio, video) |
| `NSD_POISON` | off | Debug: set: every allocation filled with 0xff (finds reads of unwritten memory) | kernels:diffusion (image, audio, video) |
| `NSD_PROFILE` | off | set: the kernels' times by part (int8 linears, attention) and the video decoder's, on stderr; waits after each (--opt-profile, video) | kernels:diffusion (image, audio, video), video h3 |
| `NSD_ROTQ_SPLIT` | off | Debug: set: the int8 linear's rotation, scales and quantize as separate passes, not one fused kernel | kernels:diffusion (image, audio, video) |
| `NSD_SAGE_MIN_S` | 8192 | the shortest sequence SageAttention takes; shorter stay on oneDNN (--opt-sage-min-s, video) | kernels:diffusion (image, audio, video) |
| `NSD_SWIGLU_SCALAR` | off | Debug: set: the plain SwiGLU kernel, not the 16-byte vector one | kernels:diffusion (image, audio, video) |

## Strata kernels: `STRATA_*`

The vendored Strata kernels (`kernels/strata`: shared, tuned - read only here) under qwen4exp and qwen35. Upstream's tuning and A/B switches, read once per process. Many are upstream's HIP or CUDA paths, compiled out of the SYCL build: those rows say so and setting them does nothing here. Rows that mention a compute capability key off what the SYCL port reports for the Intel card.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `STRATA_ARGMAX_MULTI` | on | Greedy sampling across a cluster of work-groups a token; 0: always the one-block kernel. No effect on the SYCL build (no clusters: the plain greedy kernel runs) | kernels:strata (sampler) |
| `STRATA_ATTN_LANECELL` | off | Experiment: 1 = the warp decode-attention kernel scores one cell a thread (same bits, no shuffle reductions); only with STRATA_ATTN_PERCELL=0 | kernels:strata (decode attention) |
| `STRATA_ATTN_PERCELL` | on | 0 = decode attention's warp score kernel instead of the SYCL port's per-cell kernel (128-cell chunks) | kernels:strata (decode attention) |
| `STRATA_ATTN_PRE75` | on (where built) | 0 = PR #540's pre-sm_75 decode-attention kernel off (A/B); CUDA STRATA_EXPERIMENTAL_SM60 builds only, not in the SYCL build | kernels:strata (decode attention) |
| `STRATA_BF16_TC` | by compute capability | Experiment: the BF16 prompt GEMM's path, 0 native BF16, 1 through FP16, 2 the FP16 path on any card (a test mode); keyed to CUDA compute capabilities | kernels:strata (prompt GEMM) |
| `STRATA_DBG_NAN` | off | Debug: set = count NaN/zero rows after each FP16 GEMM (a sync a GEMM); HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_DEC_BATCH` | on | Shared expert gate for several tokens as one bf16 gemv (fused sigmoid and scale); 0: one gemv a token | kernels:strata (shared expert) |
| `STRATA_DENSE_MMQ` | off | 1 = dense prompt GEMMs through the MMQ kernels; HIP builds with STRATA_PREFILL_MMQ only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_DOWN_LANES` | 8 (STRATA_EXPERT_LANES at build) | Lanes a row in the grouped expert down kernels: 4, 8, 16 or 32 (256 / lanes rows a work-group); a per-call count from the engine wins | kernels:strata (experts) |
| `STRATA_EMULATE_CC` | off | Test: 75 to 119 = answer compute-capability and shared-memory queries as that CUDA card would (CUDA-keyed paths) | kernels:strata (device queries) |
| `STRATA_EXPERT_SPLIT` | off | 1: upstream's multi kernels (one warp a row, own grid) for the split i-quant formats instead of the port's grouped expert kernels | kernels:strata (experts) |
| `STRATA_EXPERT_V2` | off | Experiment: 1: the S26 grouped expert kernels for IQ3_S gate/up + IQ4_NL down at 2560/640 (AMD-tuned) | kernels:strata (experts) |
| `STRATA_EXPERT_V2K` | off | Experiment: 1: the S27 grouped expert kernels for Q4_K/Q5_K gate/up + Q5_1/Q8_0 down at 2560/640 (UD-Q4_K_XL) | kernels:strata (experts) |
| `STRATA_EXP_MODE` | 7 | Grouped expert kernel mode for the bench. HIP gfx906 only: not compiled into the SYCL build | kernels:strata (experts) |
| `STRATA_FUSED_SWIGLU_Q81` | on | Shared expert SwiGLU fused with the q8_1 quantize of its result; 0: separate passes | kernels:strata (shared expert) |
| `STRATA_GDN_COMMIT_SPLIT` | on | 0 = the gated-delta-net step's commit (whole window kept) in the fused kernel instead of the split commit kernel | kernels:strata (gated delta net, verify) |
| `STRATA_GDN_CONVL2` | off | Experiment: nonzero = the fused tiled gated-delta-net conv kernel for long prompts | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_CONV_SERIAL` | off | Experiment: set = the prompt's gated-delta-net conv as the old serial walk (A/B) | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_HEAD` | off | 1 = the four-lanes-a-column recurrence (gfx1151); it selects HIP-only quad kernels, so no effect in the SYCL build | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_KEYHEAD` | off | Experiment: 1 = the prompt recurrence's key-head kernel (a key head's value heads in one thread), an A/B on Xe | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_NOY` | off | Experiment: nonzero = the output norm as a separate loop kernel after the quad recurrence; HIP builds only, not in the SYCL build | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_PIPELINE` | on | 0 = without the key-head recurrence (which also needs STRATA_GDN_KEYHEAD=1 on SYCL) | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_PP` | 0 | Experiment: 1 or 2 = the pipelined quad recurrence kernels; HIP builds only, not in the SYCL build | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_REC_HEADS` | off | Experiment: set = the prompt recurrence as the one-work-group-a-head kernel (A/B); also turns STRATA_GDN_HEAD off | kernels:strata (gated delta net, prompt) |
| `STRATA_GDN_SPLIT` | off | Experiment: 1 = the verify window's gated-delta-net step split (step, then norm) for windows of 2+ rows; same bits | kernels:strata (gated delta net, verify) |
| `STRATA_GROUPED_PAIR_MIN_HITS` | 0 (always the new kernels) | S2 grouped experts: below N hits the per-hit path keeps the old one-warp-a-row kernels (bitwise the same; bench knob) | kernels:strata (s2 grouped experts) |
| `STRATA_GROUPED_V1` | off | 1: the v1 grouped expert path - a block row per possible group and separate SwiGLU and q8_1 passes (A/B baseline) | kernels:strata (experts) |
| `STRATA_GR_DOWN_DIRECT` | on | Hyper-connection read: the direct down kernel; 0: the tiled one (13-19% slower at 3-6 tokens) | kernels:strata (hc read) |
| `STRATA_GR_DOWN_MAX4` | off | Experiment: 1: launches of up to 4 tokens keep 4 tokens' sums a thread instead of the max (same bits; no measured gain) | kernels:strata (hc read) |
| `STRATA_GR_DOWN_SLICED` | on | Hyper-connection read: the sliced down kernel (108.6 -> 76.5 us at 6 tokens); 0: direct | kernels:strata (hc read) |
| `STRATA_GR_FAST` | on | Latency-hidden norm/up of the hyper-connection read. HIP gfx906 only: not compiled into the SYCL build | kernels:strata (hc read) |
| `STRATA_GR_NORM_SPLIT` | on | Hyper-connection read: the norm split over work-groups; 0: one work-group a token | kernels:strata (hc read) |
| `STRATA_GR_SPLIT` | off | 1: K-split down in the multi-token hyper-connection read (not bitwise equal to single-token). HIP gfx906 only: not compiled into the SYCL build | kernels:strata (hc read) |
| `STRATA_GR_V3` | off | Experiment: 1: the v3 two-kernel stream-split hyper-connection read (another summation order) | kernels:strata (hc read) |
| `STRATA_GU_LANES` | 8 (STRATA_EXPERT_LANES at build) | Lanes a row in the grouped expert gate/up kernels: 4, 8, 16 or 32; a per-call count from the engine wins | kernels:strata (experts) |
| `STRATA_HC_CHECK` | off (the plain read) | Set: run upstream's hyper-connection read self-test at load and pick the variant it passes (crashes on the B70: opt-in) | kernels:strata (hc read) |
| `STRATA_HC_SPLIT` | the plain read (with STRATA_HC_CHECK: 2, staged) | Hyper-connection read variant: 0 plain, 1 at most split, 2 staged (the newest the check accepts); 1 or 2 without STRATA_HC_CHECK forces it (tests such as gr_parity) | kernels:strata (hc read) |
| `STRATA_HIPBLASLT_TUNING` | none | a hipBLASLt tuning table (path) for the prompt GEMMs; HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_HIPBLASLT_VERBOSE` | off | Debug: set = log hipBLASLt lookups, fallbacks and a summary; HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_HIP_PROMPT_F16` | off | 1 = the prompt's 16-bit GEMMs rounded through FP16 (faster on gfx103x, other numbers); HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_HIP_ROUTER_FAST` | off | 1: the S6 router top-k kernel on gfx906. HIP only: not compiled into the SYCL build | kernels:strata (router) |
| `STRATA_HIP_ROUTER_OLD` | off | Set: the old router top-k kernel on AMD RDNA. HIP only: not compiled into the SYCL build | kernels:strata (router) |
| `STRATA_HIP_SWIGLU_FUSED` | off | 1: the one-pass SwiGLU + q8_1 kernel for grouped experts on gfx906. HIP gfx906 only: not compiled into the SYCL build | kernels:strata (experts) |
| `STRATA_HIP_WMMA` | off | 1 = RDNA4's int8-KV WMMA prompt-attention kernel; HIP builds only, not in the SYCL build | kernels:strata (prompt attention) |
| `STRATA_IQ_STAGE_GRID` | on | Stage the 64-bit i-quant codebook tables in shared local memory in the grouped expert kernels; 0: read from global | kernels:strata (experts) |
| `STRATA_IQ_STAGE_GRID_MMVQ` | off | 1: the single-matrix i-quant mmvq stages the codebook tables too (bitwise the same; slower at few columns) | kernels:strata (mmvq) |
| `STRATA_MMVF_ROWS` | off | Experiment: 1: the bf16 multi-token gemv as the rows kernel (several rows a work-group, every token) at n_out >= 64 | kernels:strata (bf16 gemv) |
| `STRATA_MMVQ_A2` | on | Wide Q6_K mmvq with aligned 16-byte loads; 0: the misaligned-load path (also restores the old wide-kernel gate) | kernels:strata (mmvq) |
| `STRATA_MMVQ_LOOP` | off | Experiment: 1: the wide Q6_K kernel runs 5+ columns as a runtime column loop | kernels:strata (mmvq) |
| `STRATA_MMVQ_PF` | 0 (off) | Experiment: weight-load steps issued ahead in the wide Q6_K kernel | kernels:strata (mmvq) |
| `STRATA_MMVQ_RPW` | 0 (the shared kernel) | Bench knob: rows a warp for the row-warp mmvq kernel (1, 2 or 4); 0 the shared kernel | kernels:strata (mmvq) |
| `STRATA_MMVQ_SG` | 32 | Decode mmvq sub-group: 32 all 32-lane; 16 all SIMD16; 1 SIMD16 for IQ4_XS only; 2 for short outputs (<= 1024) only; 3 both | kernels:strata (mmvq) |
| `STRATA_MMVQ_WAVE` | on | 0: the CUDA layout instead of the wave64 mmvq layout. HIP gfx906 only: not compiled into the SYCL build | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE` | on | The wide 16-byte-load Q6_K kernel for decode shapes; 0: the older Q6_K kernels | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE_32` | on | Q8_0 / IQ4_NL through the wide32 kernel (1-8 columns); 0: the small/multi kernels (old path) | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE_K` | on | Q4_K / Q5_K / IQ4_XS through the wide kernel (1-8 columns); 0: the multi kernel (old path) | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE_MAXC` | 8 (4 with STRATA_MMVQ_A2=0) | Most columns (tokens) the wide Q6_K kernel takes | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE_MIN` | 0 (4096 with STRATA_MMVQ_A2=0) | Fewest output rows for the wide Q6_K kernel | kernels:strata (mmvq) |
| `STRATA_MMVQ_WIDE_RPW` | 1 | Experiment: rows a warp in the wide Q6_K kernel | kernels:strata (mmvq) |
| `STRATA_MULTI_BLOCK_ARGMAX` | on | 0 = multi_block_head_ops() off (the head ops in one work-group); nothing in the vendored tree calls it, so no effect here | kernels:strata (verify) |
| `STRATA_NORM_ROPE` | off on AMD, on elsewhere | 1: the fused norm + RoPE kernel on AMD. HIP only: the SYCL build always uses it (see STRATA_NO_NORM_ROPE) | kernels:strata (rope) |
| `STRATA_NO_MMVQ_ROWS2` | off | 1: one row a block for large K in the multi mmvq (default two rows a block for every K) | kernels:strata (mmvq) |
| `STRATA_NO_MULTI_GR` | off | 1: the hyper-connection read token by token instead of the multi-token kernels (exact-T variants off) | kernels:strata (hc read) |
| `STRATA_NO_NORM_ROPE` | off | 1: separate norm and RoPE instead of the fused norm + RoPE kernel | kernels:strata (rope) |
| `STRATA_NO_SUB16_GU` | off | 1: with STRATA_EXPERT_SPLIT=1, upstream's 8-row kernels instead of the 16-lane ones on 2560-wide split formats and the 16/32-row down | kernels:strata (experts) |
| `STRATA_OLD_GROUPED` | off | 1: the previous S2 grouped expert kernels (A/B; read once, chosen at each launch) | kernels:strata (s2 grouped experts) |
| `STRATA_OLD_IQ_MMVQ` | off | 1: the per-column i-quant mmvq kernels (bitwise equal; A/B timing) | kernels:strata (mmvq) |
| `STRATA_OLD_SAMPLER` | off | 1: the 0.1.20 sampler kernel instead of the split top-k (wins over STRATA_SAMPLER_ONE_BLOCK) | kernels:strata (sampler) |
| `STRATA_PA_FAST` | off | 1 = the WMMA prompt-attention kernel on chunks of STRATA_PF_SWITCH_MIN_T+ queries; HIP builds only, not in the SYCL build | kernels:strata (prompt attention) |
| `STRATA_PF_GEMM` | off | 1 = the S23 FP16 prompt GEMM on chunks of STRATA_PF_SWITCH_MIN_T+ tokens; HIP builds only (STRATA_USE_HIP), not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_PF_PAD` | off | 1 = with STRATA_PF_GEMM=1, the weight dequantized with row stride K + 64 (no 4 KB-multiple strides); HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_PF_SWITCH_MIN_T` | 0 | the fewest tokens a chunk needs for STRATA_PF_GEMM / STRATA_PA_FAST; HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM, prompt attention) |
| `STRATA_PLAN_PARALLEL` | on | 0 = the resident plan's entries grouped by thread 0 alone (the old way) | kernels:strata (verify) |
| `STRATA_PROMPT_ATTN_Q4` | on | 0 = Q4_0-KV prompt attention takes the old kernel (A/B); its code is behind the SYCL port's early return, so no effect in the SYCL build | kernels:strata (prompt attention) |
| `STRATA_PROMPT_ATTN_V1` | off | Debug: set = the first int8-KV prompt-attention kernel (another summation order); behind the SYCL port's early return, so no effect in the SYCL build | kernels:strata (prompt attention) |
| `STRATA_PROMPT_ATTN_XMX` | 0 (the older kernel) | 1 = the XMX prompt-attention port with 64-cell chunks, 32 = 32-cell chunks; unset or 0 = the caller's fallback kernel | kernels:strata (prompt attention) |
| `STRATA_Q6P_GRID` | 640 | Persistent work-groups of the packed Q6_K head kernel (STRATA_Q6_PACKED=1) | kernels:strata (mmvq) |
| `STRATA_Q6P_SELFTEST` | off | Debug: 1: bitwise check of the packed Q6_K head against the GGUF-layout kernels (1-4 columns) plus timings when packing; 2: also a grid sweep | kernels:strata (mmvq) |
| `STRATA_Q6_PACKED` | off | 1: repack the Q6_K output heads (main, MTP) into aligned planes at load (+VRAM; bitwise the same outputs) | kernels:strata (mmvq) |
| `STRATA_Q8_PACKED` | off | 1: lossless load-time repack of Q8_0 matrices into qs/d planes for aligned streaming decode loads | kernels:strata (mmvq) |
| `STRATA_QSA_CLUSTER` | on | 0 = top-k selection without the CTA-cluster kernel for few-query calls (sm_90+ path) | kernels:strata (QSA select) |
| `STRATA_QSA_EARLY_EXIT` | on | 0 = the multi-query score kernel's old staging (every work-group loads the queries first); same scores | kernels:strata (QSA select) |
| `STRATA_QSA_WARP` | off | 1, attn or select = the pre-sm_80 kernels on any card (A/B); the SYCL port discards the result, so no effect in the SYCL build | kernels:strata (QSA select, prompt attention) |
| `STRATA_ROPE_TABLE` | off | 1: RoPE from a precomputed cos/sin table (last-bit differences from the fast-math angles: outputs move) | kernels:strata (rope) |
| `STRATA_ROUTER_BLOCK` | off | Set: the block router top-k kernel instead of the one-wavefront one. HIP gfx906 only: not compiled into the SYCL build | kernels:strata (router) |
| `STRATA_S2_SWIGLU_Q8` | on | S2 grouped experts: SwiGLU and the scaled Q8_0 quantize in one kernel; 0: two launches | kernels:strata (s2 grouped experts) |
| `STRATA_SAMPLER_ONE_BLOCK` | off | 1: the one-block sampler kernel instead of the split top-k | kernels:strata (sampler) |
| `STRATA_SCORES_MULTI` | on | 0 = without the multi-query block-score kernel (decode and the pooled prompt path) | kernels:strata (QSA select) |
| `STRATA_SELECT_WMMA` | off | 1 = RDNA4's WMMA block scorer (selects slightly differently); HIP builds only, not in the SYCL build | kernels:strata (QSA select) |
| `STRATA_SPEC_COUPLED` | off | 1 = coupled draft sampling in Strata's own server; nothing in nextsycl reads the flag (qwen4exp uses NS_QW_COUPLED) | kernels:strata (sampler) |
| `STRATA_TOPK_ACTIVE_ANY` | off | Test: 1 = the Turing active-count top-k dispatch on any card (qsa_topk_active_parity) | kernels:strata (QSA select) |
| `STRATA_TOPK_CAPACITY_GUARD` | off | set = top-k ignores the prompt's active block count and keeps the capacity rule | kernels:strata (QSA select) |
| `STRATA_TOPK_OLD` | off | Experiment: set = the original top-k kernel (the reference kernel) | kernels:strata (QSA select) |
| `STRATA_TOPK_STREAM` | on | 0 = Turing prompt top-k above ~90K cells keeps the register kernel instead of the wide coalesced one | kernels:strata (QSA select) |
| `STRATA_TSUM` | off | Experiment: 1: sums as one transposed butterfly a warp in the S26/S27 grouped experts and the q8 hyper-connection read (bitwise the same) | kernels:strata (experts, hc read) |
| `STRATA_WMMA_BF16` | on (with STRATA_WMMA_GEMM) | 0 = STRATA_WMMA_GEMM leaves out the bf16 GEMMs; HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |
| `STRATA_WMMA_GEMM` | off | 1 = RDNA3 / RDNA3.5 WMMA dense prompt GEMM before hipBLASLt; HIP builds only, not in the SYCL build | kernels:strata (prompt GEMM) |

### deepseek2

DeepSeek-V3 / R1 (`llm/deepseek2`, its kernels `kernels/llm/deepseek2`): the experts in VRAM, pinned host memory and the
file; decode multiplies the pinned ones on the CPU.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_DS2_ARENA_MIB` | 256 MiB and ~1.4 MiB a token of the prompt chunk | the passes' temporaries on each GPU (--opt-arena-mib) | llm deepseek2 |
| `NS_DS2_CHUNK` | 1024 | a prompt pass's tokens (1..4096) (--opt-chunk) | llm deepseek2 |
| `NS_DS2_CPU` | on | 0: the pinned experts go up to VRAM over PCIe instead of being multiplied on the CPU | llm deepseek2 |
| `NS_DS2_CPU_AVX512` | on where the CPU has AVX-512 VNNI | Experiment: 0 runs the CPU's products in plain code | llm deepseek2 |
| `NS_DS2_CPU_THREADS` | 16 | the CPU's workers for the pinned experts | llm deepseek2 |
| `NS_DS2_CPU_V1` | off | Experiment: 1 runs the first AVX-512 product (indices and scales a sub-block at a time) | llm deepseek2 |
| `NS_DS2_F16_MIN` | 64 | prompt passes of at least this many tokens multiply the dense matrices in fp16 | llm deepseek2 |
| `NS_DS2_GROUPS` | 8,4 | routing within the best K of G expert groups (DeepSeek-V3's own; 0: every expert, as llama.cpp runs this file) (--opt-groups) | llm deepseek2 |
| `NS_DS2_KEEP_PAGES` | off | 1: the expert reads stay in the page cache (by default dropped: the pinned slots are the cache) | llm deepseek2 |
| `NS_DS2_KVB16` | on | 0: kv_b kept as the file stores it (Q6_K, decoded where read: 1.2 GB more for experts, 7% fewer file reads, but 7.0 tok/s against fp16's 7.9) instead of fp16 k_b / v_b | llm deepseek2 |
| `NS_DS2_LEND` | on | 0: the prompt arena's memory is not lent to the experts while decode runs | llm deepseek2 |
| `NS_DS2_MLA_FUSED_MAX` | 8192 | decode passes take the one-launch MLA kernel up to this many cells (0: never) | llm deepseek2 |
| `NS_DS2_MMVQ_GROUPS` | 512 | Experiment: work-groups of the IQ1_S / IQ2_XXS matrix-vector kernels | kernels:llm deepseek2 |
| `NS_DS2_MULTI_GROUPS` | 1024 | Experiment: work-groups of the kernel that multiplies a layer's experts at once | kernels:llm deepseek2 |
| `NS_DS2_MULTI_MAX` | 32 | passes of up to this many tokens run the experts in the decode kernels (--opt-multi-max) | llm deepseek2 |
| `NS_DS2_NGRAM` | 0 | up to K tokens (at most 7) drafted from the context's repeats and verified in one pass (--opt-ngram) | llm deepseek2 |
| `NS_DS2_OWN_K` | on | 0: Q4_K / Q5_K decode products on the shared kernels instead of this engine's (the qwen35 silo's, copied) | llm deepseek2 |
| `NS_DS2_PREFETCH` | 0 | experts a layer starts loading ahead of the next, from its router's guess (--opt-prefetch) | llm deepseek2 |
| `NS_DS2_PREFETCH_P` | 0.5 | the guessed chance of use from which an expert is loaded ahead (--opt-prefetch-p) | llm deepseek2 |
| `NS_DS2_PREFETCH_UP` | off | 1: the look ahead also copies guessed pinned experts up to VRAM | llm deepseek2 |
| `NS_DS2_PROMOTE` | 3 | pinned experts the CPU multiplied that a layer copies up to VRAM behind it (while the copy lane is idle) | llm deepseek2 |
| `NS_DS2_READ_AHEAD` | 256 | prompt passes of at least this many tokens read the next layer's file-only experts behind the current layer's (0: never) | llm deepseek2 |
| `NS_DS2_READERS` | 16 | threads reading experts from the file (--opt-readers) | llm deepseek2 |
| `NS_DS2_SPAN` | 4096 | a prompt's tokens a pass takes a layer at a time: its experts loaded once for all of them, its attention a chunk at a time | llm deepseek2 |
| `NS_DS2_TAP_EXPERTS` | off | Debug: 1 shows the checks each row's experts' gate, up and down outputs (ffn_moe_gate / _up / _down) | llm deepseek2 |
| `NS_DS2_TRACE_EXPERTS` | off | Debug: FILE gets every decode request (GPU, layer, experts) for offline cache studies | llm deepseek2 |
| `NS_DS2_TRACE_GUESS` | off | Debug: FILE gets the next layer's router guesses (top 32) beside the requests | llm deepseek2 |
| `NS_DS2_TRACE_ROUTE` | off | Debug: L prints layer L's choices and weights a row | llm deepseek2 |

### deepseek4

DeepSeek-V4-Flash (`llm/deepseek4`, its kernels its silo `kernels/llm/deepseek4/silo`): a fork of glm5next, which also reads that
engine's settings (NS_PREFETCH, NS_NGRAM, NS_PREFILL_CHUNK, NS_PROFILE, ...).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_DS4_MULTI` | on | 0: decode runs each expert on its own instead of a layer's experts in two launches | llm deepseek4 |
| `NS_PROFILE_SPLIT` | off | 1: with NS_PROFILE, decode passes' sections counted apart from the prompt's ("decode: ...") | llm deepseek4 |
| `NS_SPLIT` | by speed | layers a GPU, all but the last (default: by the square root of each card's units x clock - B70 + B65: 25 / 18) | llm deepseek4 |
| `NS_DS4_ATT_ROWS` | 2048 | a prompt pass's query rows the attention block takes at a time (q_b, attention, o_a; 64..8192) | llm deepseek4 |
| `NS_DS4_ATTN_KB` | 32 | Experiment: cells a decode attention split takes (more: fewer, longer work-groups) | kernels:llm deepseek4 |
| `NS_DS4_ATTN_SPLIT` | on | 0: decode rows' attention in one work-group a head group instead of split over the cells and merged | kernels:llm deepseek4 |
| `NS_DS4_BENCH_LAYER` | 5 | the layer whose experts `llm kernels` times in the multi-expert launches | llm deepseek4 (kernels) |
| `NS_DS4_DENSE_F16_MIN` | 64 | dense products of at least this many rows in fp16 on XMX (0: never; the experts keep NS_PROMPT_F16_MIN) | llm deepseek4 |
| `NS_DS4_ATTN_XMX` | on | 0: attention on the vector units (the first kernel) instead of XMX | kernels:llm deepseek4 |
| `NS_DS4_ATTN_G` | 8 from 64 rows, else 2 | Experiment: groups of 8 heads an attention work-group takes (1, 2, 4, 8) | kernels:llm deepseek4 |
| `NS_DS4_DRAFTS` | 1 | 2: a second draft, the draft block chained on its own output, verified in a 3-row pass (measured slower: a poem 27.7 -> 22.7 tok/s, a 3.7K summary 26.0 -> 25.5; the second draft is kept ~45% of the time) | llm deepseek4 |
| `NS_DS4_MTP_EXPERTS` | 128 | the draft block's experts kept in VRAM (of 256, 13 MB each; the most recently used); one missing is left out of that draft and read in for the next (256: all, 3.2 GiB) | llm deepseek4 |
| `NS_DS4_IDX_PRE` | on | 0: prompt rows' indexer as the fp16 GEMM writing every head's score and a pass summing them, instead of the fused score kernel (256K prompt 700 -> 943 tok/s) | llm deepseek4 |
| `NS_DS4_IDX_PACKED` | on | 0: the fused prompt indexer packs the queries into local memory a work-group instead of once into global memory (35K: indexer 3.05 -> 1.90 s) | llm deepseek4 |
| `NS_DS4_IDX_ROWS` | 1 | Experiment: query rows a fused prompt indexer work-group takes (1, 2, 4) | kernels:llm deepseek4 |
| `NS_DS4_IDX_KT` | 2 | Experiment: key tiles sharing each query tile loaded in the local-memory prompt indexer (1, 2) | kernels:llm deepseek4 |
| `NS_DS4_ROUTE_LOG` | off | path: each decode pass's experts a layer written there ("layer e1,e2,..."), for offline cache studies | llm deepseek4 |
| `NS_DS4_IDX_FUSED` | on | 0: decode's indexer as a GEMM, a score pass and one radix select a row instead of the fused score kernel and two-level top-k | llm deepseek4 |
| `NS_DS4_MULTI_CACHED` | on | 0: the IQ experts' multi-matrix kernel loads each row's activations again instead of holding them over a sub-group's run of rows | kernels:llm deepseek4 |
| `NS_DS4_MULTI_DENSE` | on | 0: decode's dense products of one input a launch a matrix (shared kernels) instead of one multi-matrix launch a type | llm deepseek4 |
| `NS_DS4_MULTI_GROUPS` | 1024 | Experiment: work-groups of the multi-expert decode kernels | kernels:llm deepseek4 |
| `NS_SPLIT` | by compute units (B70 + B65: 24) | deepseek41: layers on the first GPU - a part begins at a kv source (2, 8, 14, 20) or, on two GPUs, at an index source (24, 28, 32, 36: its kv source's keys and the candidate blocks mirrored on the second GPU each pass). By default the boundary nearest each card's share of the compute units (B70 256, B65 160 -> 24.6 -> 24). Unsloth IQ2_XXS with the reserve lent (2026-10-21, 512 tokens, medians of three): 24 against 20 tg fast 17.4 -> 20.7, exact 12.2 -> 12.7, a 35K prompt pp 418 -> 442 tok/s, its secret code found. (On the VQ file, before the lending, 24 and 28 had measured slower: an 8.7K prompt 409-493 / 382-412 tok/s against 485, decode 17-18 / 16 against 19) | llm deepseek41 |
| `NS_DS41_ENGRAM_DIR` | `deepseek-engram` beside the model | the directory of DeepSeek-V4.1's two Engram shards (model-00047/48-of-00048.safetensors, read in rows) | llm deepseek41 |
| `NS_DS41_ENGRAM_ALIGN` | 512 | the Engram reads' unit in bytes (a power of two from 512: the drives' logical block, what O_DIRECT needs; 4096 reads whole pages) | llm deepseek41 |
| `NS_DS41_ENGRAM_READERS` | 48 | the Engram's reader threads: its row reads in flight (O_DIRECT, a 512-byte block each: a row's weights, then its scales) | llm deepseek41 |
| `NS_DS41_ENGRAM_CACHE` | 262144 | the Engram rows kept in host memory (264 B each; 0: none) | llm deepseek41 |
| `NS_DS41_MTP` | 0 | 1: DeepSeek-V4.1's three DSpark draft towers loaded (2.9 GiB on the last GPU) and drafting - a gain on code (+3%, 84% kept), a loss on prose (59% kept: 15.7 against 20.3 tok/s) | llm deepseek41 |
| `NS_DS41_DRAFTS` | 2 | the DSpark towers' drafts a verify pass checks (1..7; a third is rarely kept) | llm deepseek41 |
| `NS_DS41_HOST_LRU` | 0 | 1: a full pinned host tier gives its least recently used expert back to the file to take a VRAM expert moved down (the loaded tier's least profiled first), so both tiers follow the text being read; no gain measured (tg 12.2 against 12.9 tok/s over 1,024 tokens after a 2.5K prompt), so off: the tier keeps the experts it was loaded with | llm deepseek41 |
| `NS_DS41_PIN_GIB` | 40 | pinned host memory for the experts that do not fit VRAM (at most what is available less 12 GiB); set, it also caps the server's own estimate (what is available less 10 GiB and the prompt cache), which leaves out the engine's staging slots (XQ: ~12 GB more) | llm deepseek41 |
| `NS_DS41_FREE_GIB` | 0.75 | VRAM each GPU keeps free beyond the plan (the xe driver has no out-of-memory path; the plan's own reserve for a pass is generous: at 0.75 an 8.7K prompt and 256 tokens peaked at 30.3 of 32 GiB - 2.4 GiB free -, at 1.5 at 29.5) | llm deepseek41 |
| `NS_DS41_FILE_AHEAD` | 8 | decode reads ahead (`NS_DS41_READERS` threads, a job a slab, into the staging slots) at most this many of the next layer's likely experts that only the file holds, from its router's guess (0: none) | llm deepseek41 |
| `NS_DS41_IDX_BENCH` | unset | `llm kernels` only: N1,N2,.. key counts - times the indexer's scores, its top-k and the two-level top-k at each (random keys, a prompt step's rows) and checks the two pick the same; nothing else runs | llm deepseek41 |
| `NS_DS41_IDX_ORDER` | 1 | the indexer's full scoring runs the rows of one span of keys side by side (the keys read from L2 by all of them); 0: each row's spans in turn (every key read from memory once a row) | llm deepseek41 |
| `NS_DS41_IDX_TWO` | 196608 | from this many keys an index source's top-k is taken in two levels (the blocks' maxima, the best blocks, their keys scored again, the top-k among them: the same picks); 0: never | llm deepseek41 |
| `NS_DS41_VISION` | the `*.vision.gguf` beside the model | the vision sidecar's file (DeepSeek-V4.1-Flash.vision.gguf: the ViT, the aligner, the image span's rows and the routers' image biases); 0: none (the model reads no images) | llm deepseek41 |
| `NS_DS41_VISION_DUMP` | unset | PREFIX: an image's patches, ViT output and rows written as raw float32 to PREFIX.patches.f32, .vit.f32, .rows.f32 (a check against the reference) | llm deepseek41 |
| `NS_DS41_KV_FUSE` | 1 | decode's fused launches: a layer's window keys normed, rotated, simulated and written to the ring in one; the attention output rotated back and simulated in one (both only with the simulation on); an mHC half's mixes, collapse and norm in two; two products of the same input in one at one row (q_a with kv, the shared gate with up, the router with the next layer's guess); 0: the separate kernels | llm deepseek41 |
| `NS_DS41_GPU_ROUTE` | 1 | decode routes one token on the GPU (the router's best, their weights, the products' tables from a slot table of VRAM-resident experts) and queues those experts' products before the host reads the routing; the host then does the rest (experts not in VRAM, the bookkeeping); off with the draft towers; 0: the host routes first | llm deepseek41 |
| `NS_DS41_FILE_AHEAD_P` | 0.4 | the chance of use (the guess's, from the measured rank hit rates) from which decode reads a file expert ahead (guesses that miss take the drive from the reads that matter). 512 tokens, IQ2_XXS fast: 0.2 started 13,200 reads ahead and took 672 (~280 MB a token, the drive's whole rate), tg 18.4-18.9; 0.4 started 6,281, took 670, tg 21.4; 0.6 / 0.9 the same tg. Exact: 0.2 10.6-11.7, 0.4 11.7-11.8 (on the VQ file, 0.2 against the old table had taken 8.7 -> 14.4) | llm deepseek41 |
| `NS_DS41_STD_RPS` | 2 | rows a sub-group's decode product of an Unsloth-form expert works at once (1, 2 or 4; their loads issued together) | llm deepseek41 |
| `NS_DS41_ODIRECT` | 0 | 1: the experts are read from the model's files with O_DIRECT (an aligned span into a thread's own buffer, the wanted bytes copied out) - one cold read is faster, but decode loses the page cache's hits (13.6 -> 12.7 tok/s) | llm deepseek41 |
| `NS_DS41_READ_BENCH` | none | `llm kernels` on an Unsloth-form file times the experts' file reads instead (one expert at a time, buffered and O_DIRECT, then 4 / 16 / 32 in flight) | llm deepseek41 |
| `NS_DS41_MMV_BENCH` | none | `llm kernels` on an Unsloth-form file times the decode products instead: layers 0 and 20's dense ones (the file's type on the shared mmvq, and as q4_K; hot in the GPU's cache, so small matrices read high) and each expert type's six-expert launch over 48 different experts (float32 against split Q8) | llm deepseek41 |
| `NS_DS41_COLD_SKIP` | 0 | decode leaves out a routed expert held by neither VRAM nor host memory (nor being read ahead) whose share of the layer's routing weight is under this (an approximation: measure `generate --score`) | llm deepseek41 |
| `NS_SCORE_ROWS` | 8 | `generate --score`: rows (1-8) fed a pass (1: as decode runs them) | llm |
| `NS_DS41_COLD_RENORM` | 0 | with `NS_DS41_COLD_SKIP`, the experts kept have their weights scaled by (all / kept) | llm deepseek41 |
| `NS_DS41_COLD_SKIP_AHEAD` | 1 | with `NS_DS41_COLD_SKIP`, also leave out an expert whose read ahead is on its way (0: wait for those) | llm deepseek41 |
| `NS_DS41_AHEAD_SKIP_BAR` | 0.75 | with `NS_DS41_COLD_SKIP`, decode reads a file expert ahead only when its guessed share of the routing weight is at least this many times the skip bar (those below would be left out when picked); 0: every guess | llm deepseek41 |
| `NS_DS41_STD_OA` | 0 | 1: decode's grouped o_a (a Q8_0 / Q6_K one) in one launch of the std kernel (alone faster, end to end no gain) | llm deepseek41 |
| `NS_DS41_STD_KU` | 1 | blocks a sub-group's decode product of an Unsloth-form expert decodes before multiplying (1, 2 or 4; their loads in flight together) | llm deepseek41 |
| `NS_DS41_DENSE_Q4K` | 0 | 1: attention's and the shared expert's Q8_0 weights are requantized to q4_K when loaded (Q6_K ones too, and the XQ file's F16 Engram projection; half the bytes a decode token reads, the fused q4_K kernels apply; an approximation, a few seconds more at load) | llm deepseek41 |
| `NS_DS41_DENSE_Q4K_ROLES` | qa,qb,kv,oa,ob,shgate,shup,shdown | which Q8_0 weights `NS_DS41_DENSE_Q4K` requantizes (a comma list; a Q6_K one of the eight always: its shared product is slow). All eight are fastest in decode (GPU time a token 35.0 ms against 37.4 with only oa,ob; keeping qa,qb,kv,ob in Q8_0: the B70's dense products 6.5 -> 7.8 ms) - though measured alone, even cold from VRAM, Q8_0 looked as fast for qa, kv, qb and ob | llm deepseek41 |
| `NS_DS41_DRY` | 0 | DRY against repeated runs (llama.cpp's sampler, in this engine): a token that would extend a run of this turn's generated tokens already seen earlier loses this x `NS_DS41_DRY_BASE`^(run - `NS_DS41_DRY_ALLOWED`) of its logit; runs stop at tokens with a newline, `:`, `"` or `*`, which are never penalized; the prompt's tokens never count (0: off) | llm deepseek41 |
| `NS_DS41_DRY_BASE` | 1.75 | with `NS_DS41_DRY`, how fast the penalty grows with the run's length | llm deepseek41 |
| `NS_DS41_DRY_ALLOWED` | 2 | with `NS_DS41_DRY`, the shortest run penalized | llm deepseek41 |
| `NS_DS41_ENGRAM_SHARED` | 0 | 1: the Engram's Q5_K / Q6_K rows through the shared dequant kernels instead of the silo's std ones (a measurement switch) | llm deepseek41 |
| `NS_DS41_FILL_LENT` | 0 | 1: the big prompt arena's slots, lent to an expert store while no prompt pass needs them, filled at load with the profile's next experts (else empty until decode's misses fill them; measured no fewer file reads) | llm deepseek41 |
| `NS_DS41_STD_Q8` | 1 | decode's Unsloth-form expert products against the activations quantized to int8 (split Q8 layout, dp4a: `mmvq.cpp`, the deepseek4 silo's scheme with IQ1_S, IQ2_XS and Q2_K added); expert products ~11% less GPU time, perplexity unchanged (3.4457 / 4.0464 against 3.4454 / 4.0367); 0: `std.cpp`'s float32 kernel | llm deepseek41 |
| `NS_DS41_SKIP_HOST` | 0 | 1 (a measurement - wrong output): `NS_DS41_COLD_SKIP` also leaves out experts only the pinned host tier holds, so decode never waits for a swap: the ceiling of free misses | llm deepseek41 |
| `NS_DS41_LEND_RESERVE` | 1 | decode's expert stores also take the VRAM the prompt passes' reserve leaves free (what is free at decode's first step less `NS_DS41_FREE_GIB`, the sessions' held-back caches and 1 GiB; ~5.6-6.1 GiB a card at a 16384-token chunk, ~1,200 more experts over two cards), filled at once from the pinned host tier; given back (its experts moved down, the buffer freed) before a pass that needs the big arena, lent again after. Swaps a 256-token decode: exact 8,061 -> 5,149, fast 3,691 -> 2,176; perplexity unchanged (3.4434 / 4.0182). 0: only the big arena is lent | llm deepseek41 |
| `NS_DS41_FILE_AHEAD_FAR` | 0 | decode's file reads further ahead: the routers of this many layers after the next run on a layer's input, and up to `NS_DS41_FILE_AHEAD_FAR_TOP` (3) of each one's best `NS_DS41_FILE_AHEAD_FAR_RANK` (8) that only the file holds read ahead. Measured worse (2: tg 10.3 against 18.3 tok/s, 32K more reads in 512 tokens; the one-layer reads ahead are already 5% used: 672 of 13,207) | llm deepseek41 |
| `NS_DS41_FILE_AHEAD_FAR_TOP` | 3 | see `NS_DS41_FILE_AHEAD_FAR` | llm deepseek41 |
| `NS_DS41_FILE_AHEAD_FAR_RANK` | 8 | see `NS_DS41_FILE_AHEAD_FAR` | llm deepseek41 |
| `NS_DS41_MMVQ_GROUPS` | 1024 | work-groups of a `NS_DS41_STD_Q8` launch (each fills its grid table once; 256 and 4096 measured slower) | llm deepseek41 |
| `NS_DS41_MMVQ_ROWS` | 2 | rows a sub-group of a `NS_DS41_STD_Q8` launch works at once (1, 2 or 4; 4 measured slower) | llm deepseek41 |
| `NS_DS41_PROMPT_PREFETCH` | 1 | a prompt pass starts reading the next layer's file experts while it computes this one | llm deepseek41 |
| `NS_DS41_COLD_SKIP_PROMPT` | 0 | a prompt pass leaves out a file-only expert that gets at most `NS_DS41_COLD_SKIP_PROMPT_MAX` tokens, all with a routing-weight share under this (an approximation: measure the perplexity after such a prompt) | llm deepseek41 |
| `NS_DS41_COLD_SKIP_PROMPT_MAX` | 4 | the most tokens such an expert may get | llm deepseek41 |
| `NS_DS41_PROMPT_PREFETCH_MIN` | 4096 | the fewest tokens in a pass for that prefetch (below, the early reads only contend with the layer's own: XQ 2,561 tokens 15.8 -> 19.2 s with it; 9,901 tokens fast 28.3 -> 24.1 s) | llm deepseek41 |
| `NS_DS41_READERS` | 16 | the read-ahead's reader threads (a job a slab of an expert in ggml's row types) | llm deepseek41 |
| `NS_DS41_REPLICA` | unset | a folder holding copies of the model's files (the same names and sizes; any of them) on another drive: the experts' file reads split between each file and its copy (each expert always from the same one), so both drives read at once | llm deepseek41 |
| `NS_DS41_REPLICA_SHARE` | 40 | percent of the experts read from the `NS_DS41_REPLICA` copies (40: a 4.8 GB/s drive beside a 7.3 GB/s one) | llm deepseek41 |
| `NS_DS41_PROMPT_READERS` | 32 | the reader pool's threads at least this many (a prompt pass's file-only experts, a job a slab: 32 reads in flight keep the drive busier than 16) | llm deepseek41 |
| `NS_DS41_PROMPT_ODIRECT` | 1 | a prompt pass's expert reads O_DIRECT (0: through the page cache, which a long prompt's tens of GB churned beside the pinned memory - reclaim stalled the reads for seconds) | llm deepseek41 |
| `NS_DS41_DRAFTER` | none | the DSpark drafter file (Unsloth's `dspark-*.gguf`) of an Unsloth-form model, with `NS_DS41_MTP=1`; else `dspark-*.gguf` in the model's folder or its parent's | llm deepseek41 |
| `NS_DS41_STD_DIR` | none | `llm kernels` on an Unsloth-form file writes each expert type's raw bytes and its GPU expansion here, and compares with `<tag>.ref.f32` (ggml's own dequantize_row_*) when there | llm deepseek41 |
| `NS_DS41_SPARE` | 8 | decode's empty VRAM slots kept a store: an expert swapped in takes one (its copy up waits for no copy down), the least recently used are moved down afterwards (0: a swap moves its victim down first) | llm deepseek41 |
| `NS_DS41_QSIM` | on | 0: the model's low-precision simulations (fp8 window keys and attention output, fp4 compressed and indexer keys) left out - a check against a reference without them | llm deepseek41 |
| `NS_DS41_VQ_V` | measured | 0..5: the VQ decode products' kernel variant (0 a sub-group a row on the float32 codebook, 1 a lane a row on the bytes, 2 a lane a row on fp16, 3 a sub-group a row on fp16, 4 a lane a row two bytes at once, 5 by lookup table: each 8 values of x against every codebook entry in local memory) instead of the fastest of them timed at load | llm deepseek41 |
| `NS_DS41_LUT_RUNS` | 32 | Experiment: the runs of 64-column units a row is split into by the VQ lookup-table variant (a work-group a run; their partials added after) | llm deepseek41 |
| `NS_DS41_PROMPT_MMV` | 0 | a prompt pass's experts with at most this many of its tokens multiplied by the decode kernels (8 tokens an entry) instead of the XMX product from the indices (vq_gemm, faster at every count measured) | llm deepseek41 |
| `NS_DS41_PROMPT_MMV_V` | 3 | the decode kernels' variant (NS_DS41_VQ_V's numbers) for those experts | llm deepseek41 |
| `NS_DS41_PROMPT_RING` | 24 | a prompt pass's VRAM staging slots (a slot an expert, from each GPU's expert memory) for the experts in host memory or only in the file: copied up on the copy lane, half the ring a wave, while the others are multiplied (0: host-memory experts read in place, the file's swapped in) | llm deepseek41 |
| `NS_DS41_PROMPT_FILE_SLOTS` | 320 | a prompt pass's pinned slots (each GPU, two halves for two layers) for the experts only the file holds: read on the reader pool while the GPU multiplies the others (0: a wave at a time on the host thread) | llm deepseek41 |
| `NS_DS41_PIPE_LOG` | 0 | 1: each prompt chunk's time on each GPU of the two-GPU pipeline (and the second's wait for the first), on stderr; 2: and each layer's attention and experts apart (a GPU sync between them) | llm deepseek41 |
| `NS_DS41_PLACE` | expert | Experiment: `layer` fills VRAM, then pinned memory, with the experts in layer order (the last layers of a GPU then live in the file alone) instead of every layer the same share | llm deepseek41 |
| `NS_DS41_VQ_LUT` | 1 | 0: the VQ lookup-table variant left out of the decode products' timing at load | llm deepseek41 |
| `NS_DS41_IDX_XMX` | 1 | 0: the indexer's scores in the scalar kernel instead of on XMX (deepseek4's form for 32 heads; the queries rounded to fp16) | kernels:llm deepseek41 |
| `NS_DS41_IDX_CAND` | 1 | the index sources after the candidate source score only each row's candidate blocks (2,048 blocks of 8 keys) on XMX, then pick among them; 0: every key scored, those outside the blocks masked | llm deepseek41 |
| `NS_DS41_VQ_GEMM_MAX` | 100000 | a prompt pass's experts past NS_DS41_PROMPT_MMV tokens and up to this many multiplied on XMX straight from their indices; more expanded to fp16 for a GEMM (0: all expanded) | llm deepseek41 |
| `NS_DS41_VQ_GEMM_V` | 5 | Experiment: the prompt's XMX expert product's form - 0 its B tiles through local memory, 1 decoded straight into the tiles' registers (8 K slices a work-group), 2 / 3 the same with 4 K slices, 4 / 5 as 2 / 3 and past 32 tokens 8 token tiles a sub-group, 6 / 7 two or four row tiles a sub-group (4 K slices), 8 / 9 two with 8 / 16 K slices | kernels:llm deepseek41 |
| `NS_DS41_GEMM_TEST_N` | 40 | `llm kernels`: the tokens of its vq_gemm check (1..64) | llm deepseek41 |
| `NS_DS41_VQ_BENCH` | 0,1 | `llm kernels`: the VQ decode variants it times | llm deepseek41 |
| `NS_DS41_MMVQ_GRF` | 0 | Experiment: 1: every split-Q8 expert product on the large register file (grf_size 256; the XQ file's mixed one always is) | kernels:llm deepseek41 |
| `NS_DS41_Q4K_CACHE` | none (cli service: /cache/q4k) | a directory: the q4_K blocks `NS_DS41_DENSE_Q4K` requantizes are kept there and read back at the next load (XQ fast load 42.3 -> ~17 s; ~4.7 GiB for XQ) | llm deepseek41 |
| `NS_DS41_Q4K_R` | 4 | Experiment: rows a sub-group of decode's q4_K products (1, 2 or 4) | kernels:llm deepseek41 |
| `NS_DS41_Q4K_COLS` | 2 | decode's q4_K products decode each weight once for all of a pass's columns from this many columns (a verify pass; the same arithmetic a column as the one-column kernel, so verify rows stay bit-identical); 1 one column too, 0 never | kernels:llm deepseek41 |
| `NS_DS41_Q4K_MAGIC` | 0 | Experiment: 1: decode's q4_K products turn the nibbles into floats by their bits (1 + q/16, no conversion; the sums rounded differently) | kernels:llm deepseek41 |
| `NS_DS41_VQ_RPG` | 8 | Experiment: sub-groups a work-group of the VQ decode products | kernels:llm deepseek41 |
| `NS_DS41_VQ_RPS` | 1 | Experiment: rows a sub-group of the VQ decode products takes in turn | kernels:llm deepseek41 |
| `NS_DS41_Q4K` | on | 0: decode's q4_K products through the shared Q8_1 quantization and product (o_a a launch a group) instead of this engine's float32 kernel | llm deepseek41 |
| `NS_DS41_DENSE_F16_MIN` | 64 | dense products of at least this many rows in fp16 on XMX | llm deepseek41 |
| `NS_DS41_ATTN_XMX` | on | 0: attention on the vector units instead of XMX (deepseek4's kernels, copied) | kernels:llm deepseek41 |
| `NS_DS41_ATTN_SPLIT` | on | 0: decode rows' attention in one work-group a head group instead of split over the cells | kernels:llm deepseek41 |
| `NS_DS41_ATTN_KB` | 32 | Experiment: cells a decode attention split takes | kernels:llm deepseek41 |
| `NS_DS41_ATTN_G` | 8 from 64 rows, else 2 | Experiment: groups of 8 heads an attention work-group takes | kernels:llm deepseek41 |

### museglimmer

Muse Glimmer 30B (`llm/museglimmer`, its kernels `kernels/llm/museglimmer`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_MG_CHUNK` | 2048 rows when they fit beside the weights and a session, else 1024 or 512 | a prompt pass's rows (at least 16) (--opt-chunk) | llm museglimmer |
| `NS_MG_DRAFT` | the smallest dflash-*.gguf beside the model | the DFlash drafter's GGUF (--opt-draft) | llm museglimmer |
| `NS_MG_DRAFTS` | 15 (the drafter's block less one) | tokens drafted a step (0: none) (--opt-drafts) | llm museglimmer |
| `NS_MG_XMX_SHAPE` | 7 | Experiment: the silo's XMX decode products' shape (tiles of 8 rows a sub-group, sub-groups a work-group, blocks an iteration) - 7: 1, 4, 2; 0: 2, 4, 1; 1: 4, 4, 1; 2: 2, 8, 1; 3: 1, 8, 1; 4: 2, 4, 2; 5: 1, 8, 2; 6: 2, 8, 2; 8: 1, 4, 4; 9: 1, 2, 4; 10: 1, 8, 4; 11: 2, 2, 2 (`llm kernels` times them) | kernels:llm museglimmer (silo) |
| `NS_MG_KQ_SHAPE` | 0 | Experiment: the silo's K-quant (Q4_K, Q6_K) decode products' shape (tiles of 16 rows a sub-group, sub-groups a work-group, groups of 64 a step) - 0: by the width (16, 13 or 8 sub-groups splitting a row's groups evenly, 2 a step); 1: 1, 8, 2; 2: 1, 4, 2; 3: 1, 8, 4; 4: 2, 8, 2; 5: 1, 16, 2; 6: 1, 13, 2; 7: 1, 13, 4 (`llm kernels` times them) | kernels:llm museglimmer (silo) |
| `NS_MG_PROFILE` | off | Debug: 1 times every kernel (the queue drained before and after each) and lists them after the run | llm museglimmer |

## Tests and checks

Inputs of the checks and unit tests that need files outside the repository (unset: the test is skipped), and the parity tools.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_CHECK_COND` | the reference's cond_latents.npy | edit-dit check: a raw float32 file of condition latents to use instead | image qwenimage21 (check) |
| `NS_CHECK_EMBEDS` | the reference's embeddings | edit-dit check: a raw float32 file of embeddings to use instead (the engine's own, to see their effect) | image qwenimage21 (check) |
| `NS_CHECK_LATENTS` | the reference's last latents_N.npy | vae check (HunyuanImage 2.1): a .npy of latents [h * w, 64] to decode instead | image hunyuanimage21 (check) |
| `NS_CHECK_NOISE` | the reference's noise.npy | edit-dit check: a raw float32 noise file to use instead | image qwenimage21 (check) |
| `NS_CHECK_PICTURE` | edit-pre check fails | edit-pre check: the original picture (PNG) resized and compared with the reference's | image qwenimage21 (check) |
| `NS_CHECK_PROMPT` | no ids compared | edit-te check: an edit prompt whose token ids are compared with the reference's | image qwenimage21 (check) |
| `NS_CHECK_SAVE` | off | llm check: DIR gets this engine's tensors too (<name>.f32), beside the reference's | cli (llm check) |
| `NS_DUMP_DIR` | ns-dump | Directory llama-ns-dump writes its float32 tensors and index.tsv to | reference/llama-dump (ns-dump.cpp, llama.cpp parity tool) |
| `NS_DUMP_RE` | the GLM layer outputs (inp_embd, attn_norm-N, l_out-N, result_output, ...) | Regex of the llama.cpp graph tensor names llama-ns-dump writes | reference/llama-dump (ns-dump.cpp, llama.cpp parity tool) |
| `NS_DUMP_PROMPT` | unset (-p) | a file whose bytes are the prompt as they are (-p and -f drop a trailing newline) | reference/llama-dump (ns-mtmd-dump.cpp, llama.cpp mtmd parity tool) |
| `NS_DUMP_GEN` | 48 | greedy tokens llama-ns-mtmd-dump generates after the prompt (next.txt) | reference/llama-dump (ns-mtmd-dump.cpp, llama.cpp mtmd parity tool) |
| `NS_MM3_CHECK_STEPS` | 3 | chunks check: flow steps a window | audio minimaxmusic3 (check) |
| `NS_TEST_H3_TOKENIZER` | test skipped | the H3 tokenizer's directory (vocab.json, merges.txt, tokenizer_config.json) for its reference-match test | video h3 (cargo test) |
| `NS_TEST_STARVECTOR_TOKENIZER` | test skipped | StarVector's tokenizer.json, for its prompt / stop tokens and decoding test | image starvector (cargo test) |
| `NS_UP_DUMP` | unset | Debug: a directory: the upscalers write each layer's output there as raw halves (its blob, layer and shape in the name), and the tile's input - for comparing with a reference layer by layer | image upscale |
| `NS_TEST_WHISPER_CONFIG` | test skipped | whisper-large-v3-turbo's config.json for its sizes' test | audio whisper (cargo test) |
| `NS_TEST_WHISPER_GENERATION` | test skipped | whisper-large-v3-turbo's generation_config.json (with NS_TEST_WHISPER_TOKENIZER) for the vocabulary, rules and alignment tests | audio whisper (cargo test) |
| `NS_TEST_WHISPER_REF` | test skipped | a folder of reference/whisper/ref.py dumps (short/, long/, song/): the mel, greedy-rule, alignment and placement tests against openai-whisper | audio whisper (cargo test) |
| `NS_TEST_WHISPER_TOKENIZER` | test skipped | whisper-large-v3-turbo's tokenizer.json for the byte-level tokenizer test and the vocabulary tests | tests (crates/tok), audio whisper (cargo test) |
| `NS_TEST_QWEN_TOKENIZER` | test skipped | a Qwen3-VL tokenizer.json for the tokenizer test (with NS_TEST_QWEN_TOKENS) | tests (crates/tok) |
| `NS_TEST_QWEN_TOKENS` | test skipped | the reference pipeline's tokens.json the tokenizer test compares against | tests (crates/tok) |
| `NS_X_INT8` | - | fixture: an option table's variable in the --opt-NAME unit tests (never read by an engine) | tests (crates/core options) |
| `NS_X_SIGMAS` | - | fixture: an option table's variable in the --opt-NAME unit tests (never read by an engine) | tests (crates/core options) |

## Build

Read by `build.sh`, `kernels/build.sh`, `wfe/build.sh` and the scripts beside them; `NS_VERSION` also at compile time.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `AOT` | bmg-g31 | the GPU the kernels are compiled ahead for (empty: SPIR-V, JIT at first use) | build (kernels/build.sh) |
| `DNNL` | /opt/onednn | oneDNN's install (headers, libdnnl) for the diffusion, image, video and audio kernels | build (kernels/build.sh) |
| `JOBS` | 8 | compiles run at a time | build (kernels/build.sh) |
| `LINK_ANYWAY` | off | set: link the libraries even when a source failed to compile | build (kernels/build.sh) |
| `NODE` | node on PATH, else ~/.local/bin/node | the node that runs the vendored TypeScript compiler | build (wfe/build.sh) |
| `NS_ARK` | /opt/ark/auto_round_kernel | ARK's checkout for the flash-attention library (skipped without its headers) | build (kernels/build.sh) |
| `NS_SHOT_PICTURES` | none | two pictures, comma-separated: shoot.py also shoots an edit (not checked: Python) | docs/screenshots/shoot.py |
| `NS_SYCL_TLA` | /opt/sycl-tla | sycl-tla's headers for the flash-attention library | build (kernels/build.sh) |
| `NS_VERSION` | ./version.sh (yy.mmdd.###); "dev" if cargo built without it | the version compiled in (read at compile time, option_env!) | build (build.sh), cli, serve |
| `NS_WFE` | http://localhost:8086 | the running server `wfe/build.sh --check` smoke-tests (not checked: JavaScript) | build (wfe/check.mjs) |
| `ONLY` | every source | compile only these kernel sources, space-separated | build (kernels/build.sh) |

## System variables read

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | level_zero:* when the program re-runs a command in the image | which devices the SYCL runtime offers (oneAPI's own variable) | cli, the oneAPI runtime |
| `HOME` | /tmp | the base of the default paths (~/.config/nextsycl.conf, the registry, the output and cache directories) | cli, serve, models |
| `USER` | empty | whose processes `restart` looks for (pgrep -u) | serve |
| `XDG_RUNTIME_DIR` | /tmp | the base of the socket directory (nextsycl/) and the front door's run logs (nextsycl-serve/) | cli, serve |
