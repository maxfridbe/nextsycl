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
| `NS_CHAT_UI_PORT` | 8080 | Open WebUI's port (the page's link) | serve |
| `NS_CORS` | loopback origins only | more web origins that may call the servers from a browser, comma-separated (`*`: any) | cli (llm, image, audio start) |
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
| `NS_EFFORT` | low | default reasoning effort (--effort) | cli (llm start) |
| `NS_GPUS` | all | the chat server's GPUs when the command and the model do not say ("0 1" or "0,1") | cli (llm start) |
| `NS_HOST` | 127.0.0.1 | the chat server's address (0.0.0.0: the network) (--host) | cli (llm start) |
| `NS_KEEP_REQUESTS` | 100 | ended requests kept for ps and inspect (--keep-requests) | cli (llm start) |
| `NS_MAX_TOKENS` | to the context's end | tokens a request without max_tokens may make (--max-tokens) | cli (llm start) |
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
| `NS_PROFILE` | off | Time the passes by section (1: GPU synced at each boundary; gpu: device timestamps, no syncs; deepseek2: 1 its decode passes, prompt its prompt passes) (--opt-profile) | llm glm5next, llm qwen35 (NS_Q35_PROFILE needs gpu), llm deepseek2, kernels:ns (gpu: profiling queues) |
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
| `NS_EXPERT_PROFILE` | none (cli service sets /cache/expert-profile.txt) | File of layer expert weight lines: which experts fill VRAM first at load; the server writes decode's requests to it at stop (--opt-expert-profile) | llm glm5next, serve (llm server, saves it) |
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
| `NS_QW_EXPERT_PROFILE` | layer order | expert profile file ranking which experts fill VRAM first when not all fit (--opt-expert-profile) | llm qwen4exp |
| `NS_QW_MTP` | no drafts | the MTP draft layer: Strata's runtime directory (--opt-mtp) | llm qwen4exp |
| `NS_QW_PLE_PREFETCH` | 1 | 0: a window's per-layer embedding rows read one after another, not asked for together first (fadvise WILLNEED) (--opt-ple-prefetch) | llm qwen4exp |
| `NS_QW_PROFILE` | off | Debug: 1 prints each decode round's time by part on stderr (costs ~9%) (--opt-qw-profile) | llm qwen4exp, kernels:qwen4exp |
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
| `NS_Q35_DRAFTS` | 3 | tokens drafted a step with MTP (at most 7; 0: none) (--opt-drafts) | llm qwen35 |
| `NS_Q35_MMVQ_GROUPS` | 512 | Experiment: work-groups of the silo's matrix-vector kernels with a lookup table | kernels:qwen35 silo |
| `NS_Q35_MMVQ_ROWS` | 0 (each type's own) | Experiment: rows a sub-group (1, 2 or 4) for every 256-block type of the silo's matrix-vector kernels | kernels:qwen35 silo |
| `NS_Q35_MTP` | the first .gguf in an MTP/ folder beside the model | the MTP draft block's GGUF (--opt-mtp) | llm qwen35 |
| `NS_Q35_PROFILE` | off | Debug: 1 times every kernel on the device with its bytes and arithmetic (needs NS_PROFILE=gpu; one GPU only) | llm qwen35 |

## Image

Qwen-Image 2.1 (`image/qwenimage21`), HunyuanImage 2.1 (`image/hunyuanimage21`).

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_IMAGE_OUT` | ~/.local/share/nextsycl/images | where pictures are saved | cli (image) |
| `NS_HI_*` | - | every NS_HI_ setting in nextsycl.conf is passed to the image server | cli (image serve) |
| `NS_HI_GUIDANCE` | apg | the HunyuanImage base model's guidance: apg (the pipeline's MIX_mode_0: CFG through step 5, or 38 when the negative quotes text, then adaptive projected guidance) or cfg (plain CFG every step) (--opt-guidance; per request too) | image hunyuanimage21 |
| `NS_HI_PROFILE` | off | Debug: set: a GPU wait after each named step of the DiT, the totals printed after the picture (slower; shows where the time goes) | image hunyuanimage21 |
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
| `NS_VIDEO_GPUSTAT` | /run/gpustat.json | the GPU telemetry file the studio reads | cli (video serve), studio |
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

MiniMax Music 3 (`NS_MM3_`), Qwen3-TTS (`NS_Q3T_`), VoxCPM2 (`NS_VCP_`). A per-request option's variable is only its name: it rides in the request (`"options"`), the engine does not read the environment for it.

| Variable | Default | What it does | Read by |
|---|---|---|---|
| `NS_AUDIO_OUT` | ~/.local/share/nextsycl/audio | where songs, speech and saved voices are kept | cli (audio), serve |
| `NS_MM3_*` | - | every NS_MM3_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_MM3_DIT_INT8` | 0 | 1: the flow transformer's block matrices in int8 ConvRot (--opt-dit-int8) | audio minimaxmusic3 |
| `NS_MM3_FPMATH` | strict | the decoder's oneDNN convolutions' math: strict (float32), f16, bf16 or tf32 | kernels:audio (minimaxmusic3) |
| `NS_MM3_INT8` | 0 | 1: the language model and depth decoder in int8 (half the VRAM, ~1.7x the frames a second) (--opt-int8) | audio minimaxmusic3 |
| `NS_MM3_NSD_CONV` | 0 | Debug: 1: the vocoder on the shared plain-loop convolutions (exact float32, slow) instead of oneDNN's | audio minimaxmusic3 |
| `NS_Q3T_*` | - | every NS_Q3T_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_Q3T_GREEDY` | 0 | the per-request option greedy's name: most likely code, no repetition penalty; a request's options only, the variable itself is not read (--opt-greedy) | audio qwen3tts |
| `NS_Q3T_INT8` | 0 | 1: the talker's and code predictor's matrices in int8 (less VRAM, faster frames) (--opt-int8) | audio qwen3tts |
| `NS_Q3T_STREAMING` | clone 1, built-in and designed voices 0 | the per-request option streaming's name: the text fed a token a frame; a request's options only, the variable itself is not read (--opt-streaming) | audio qwen3tts |
| `NS_Q3T_TEMPERATURE` | 0.9 | the per-request option temperature's name: the talker's sampling temperature; a request's options only, the variable itself is not read (--opt-temperature) | audio qwen3tts |
| `NS_Q3T_TOP_K` | 50 | the per-request option top-k's name: the talker's top-k; a request's options only, the variable itself is not read (--opt-top-k) | audio qwen3tts |
| `NS_BZ2_*` | - | every NS_BZ2_ setting in nextsycl.conf is passed to the audio server | cli (audio serve) |
| `NS_BZ2_GREEDY` | 0 | the per-request option greedy's name: the most likely code, backbone and depth decoder (deterministic; may not end); a request's options only, the variable itself is not read (--opt-greedy) | audio breezetts2 |
| `NS_BZ2_TEMPERATURE` | 0.9 | the per-request option temperature's name: the sampling temperature, backbone and depth decoder; a request's options only, the variable itself is not read (--opt-temperature) | audio breezetts2 |
| `NS_BZ2_TOP_K` | 50 | the per-request option top-k's name: the sampling top-k, backbone and depth decoder; a request's options only, the variable itself is not read (--opt-top-k) | audio breezetts2 |
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
| `NS_VCP_INT8` | 0 | 1: the language models' matrices in int8 (less VRAM, faster patches) (--opt-int8) | audio voxcpm2 |
| `NS_VCP_MAX_RATIO` | 6 | the per-request option max-ratio's name: the most patches a text token (a runaway's bound); a request's options only, the variable itself is not read (--opt-max-ratio) | audio voxcpm2 |

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
| `NS_M3D_AAM_PROFILE` | off | Debug: set: AnimateAnyMesh waits for the GPU after each named step of the flow and prints the totals (slower; shows where the time goes) | model3d animateanymesh |
| `NS_M3D_AAM_TRAJ` | max(512, vertices / 8) | AnimateAnyMesh: the trajectory tokens DyMeshVAE samples (more: finer motion, slower) (--opt-traj; per request too) | model3d animateanymesh |
| `NS_M3D_UNIRIG_BEAMS` | 15 | UniRig: the skeleton decoder's beams, sampled as UniRig's inference (1: greedy, the same skeleton every time) (--opt-beams; per request too) | model3d unirig |
| `NS_M3D_UNIRIG_PROFILE` | off | Debug: set: UniRig waits for the GPU after each named step and prints the totals (slower; shows where the time goes) | model3d unirig |
| `NS_M3D_RIGNET_FACES` | 4000 | RigNet: the triangles the mesh is decimated to before rigging (the weights go back to every vertex of the input) (--opt-faces; per request too) | model3d rignet |
| `NS_M3D_RIGNET_BANDWIDTH` | the learned 0.0429 | RigNet: the joints' mean-shift bandwidth (smaller: more, closer joints) (--opt-bandwidth; per request too) | model3d rignet |
| `NS_M3D_RIGNET_THRESHOLD` | 1e-5 | RigNet: the joints' density threshold (lower: more joints) (--opt-threshold; per request too) | model3d rignet |
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
| `NS_MM3_CHECK_STEPS` | 3 | chunks check: flow steps a window | audio minimaxmusic3 (check) |
| `NS_TEST_H3_TOKENIZER` | test skipped | the H3 tokenizer's directory (vocab.json, merges.txt, tokenizer_config.json) for its reference-match test | video h3 (cargo test) |
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
