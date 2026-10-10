# TODO

Paused 2026-10-06. State: GLM-5.3-Flash (ds4 IQ2 file) on B70 + B65, served as the box's chat mode `glm53`, MTP on
by default (`--no-mtp` turns it off).

## Done

- [x] Server (`nextsycl serve`): OpenAI API, streaming, thinking split out, prompt-prefix reuse, graceful stop (7931d08)
- [x] Wired into the box: studio llm mode `glm53` + model-switch entry (LIGHT: no tools, no background tasks)
- [x] MTP speculative decoding (8faf2e9): draft block over the whole conversation, 2-row verify, KDA snapshot
      rollback, exact acceptance. Greedy 13.1 -> 15.7 tok/s, 87% of drafts accepted on a code prompt; verify rows
      bit-identical to one-token decode (`nextsycl spec-check`: 0 difference over 94 rows)

## Long context (done 2026-10-06)

- [x] The DSA lightning indexer (llama.cpp de25343 `build_kpool_select` / `set_input_kpool`): pools of 4, a pool
      visible once complete, top 512 pools per row plus the row's incomplete pool, dense below 512 pools
- [x] Pooled keys in the pass that completes them (a 4-token ring of ik | ig per layer); scores as GEMMs; a radix
      top-k with a scan compaction (deterministic); attention scores in local memory (<= 2,051 cells a row)
- [x] The MTP block through its own indexer
- [x] VRAM held for the sessions' attention caches before the expert store sizes itself
- [x] Parity: 3,180-token prompt against llama.cpp - layer 3's attention output 0.99999 with selection active, the
      same next token; deeper layers drift smoothly (0.98 by layer 23, no layer jumps); spec-check 0 difference
- [x] 12,202-token prompt runs (prompt 60 tok/s, decode 12.1 tok/s - flat with context); the chat mode serves 64K
- [ ] 128K: the latents are float32 (2 KB a token a layer; 6.5 GB for two 128K sessions) - fp16 latents first
- [ ] Prompt speed (2026-10-06): 396 tok/s at 3K, 425 at 12K (from 23 at the start of the long-context work; 4,096
      chunks; experts, dense matrices and MLA attention in fp16 on XMX; host experts copied on a copy queue beside
      the compute; the KDA scan a sub-group per column; hc mix a GEMM). Left at 12K: the experts' fp16 GEMMs and
      expansion (~11 s), MLA's per-row gather of its cells (2.1 s: score a row block against the union of its
      cells), KDA 3.7 s. Next: a fused dequant + XMX GEMM; MLA over row-block unions
- [ ] Decode past 2K: 12.1 vs 15.7 tok/s short - the indexer's per-token GEMM + the attention kernel

## Prompt speed: what Strata's prompt path teaches (its numbers on the B70, docs/INTEL.md there)

- [ ] **Bigger chunks.** Strata reads prompts in 4,096-token chunks; at 2,048 an 80K prompt took 278 s, at 4,096 77 s
      (its experts not in VRAM are streamed once per chunk). Here a 512-token chunk routes ~14 tokens to an expert;
      at 4,096 it would be ~114, so every expert's weights (from VRAM or over PCIe) serve 8x the tokens, and the
      expanded fp16 GEMM path (measured slower at 512: 22.6 tok/s) becomes a real GEMM. What stops it: the arena
      (~1.5 MB of temporaries a token: 6 GB at 4,096). Trim the per-token temporaries (fp16 activations, in-place
      KDA/MoE buffers), and borrow VRAM for the rest.
- [ ] **Borrow expert slots for the prompt** (Strata's "prompt path borrows N cache slots"): during a long prompt,
      lend part of the VRAM expert cache to the chunk's temporaries / a streaming ring, give it back for decode.
      Strata mirrors the lent slots in pinned host memory so the refill after the prompt is a RAM copy (its "lend
      mirror": prompt +15-40%, refill 720-2,240 -> 200-380 ms).
- [ ] **A stager: stream the next layer's experts while this one computes** (Strata's 96-slot streamed ring, a copy
      queue, a polled sequence flag - never host waits on queue events: they deadlocked under Level Zero v2). Today
      the in-place PCIe reads sit on the compute critical path.
- [ ] **Group by expert across the chunk, one launch a layer** - done (the grouped kernels); prompt lanes 32 a row.
- [ ] **QSA / indexer scores as GEMM tiles** (Strata's QSA block scores: a GEMM tile per block pair) - the indexer
      here already scores by GEMM; the MLA prompt attention (12 s of 47 at 3K) is the per-row kernel: tile it.
- [ ] **Dequant with vector stores** (Strata: 575 -> 841 tok/s at 2K / 8K) for any path that still expands weights.

## Decode (2026-10-06 night: 18.5-20 tok/s)

- [x] hc pre fused, async swaps on two copy lanes, 4 lanes a row, prefetch by the next layer's router (2 a layer),
      the prompt arena's VRAM lent to experts at decode, the strongest GPU last
- [x] Expert cache policy studied offline (reference/expert-cache): LRU is the best simple policy; the rest of the
      gap to Belady needs better prediction
- [ ] Swaps are still ~1/3 of decode time (both cards on Gen5 x8)
- [x] Better expert prediction: guesses by chance of use (rank hit rates; reference/expert-cache) - p >= 0.6,
      read at the router's wait: decode 18.4 -> 19.4 tok/s. Two layers ahead predicts worse (93% at rank 1 vs 96%)
- [ ] The rest of the gap to Belady: a guess from a later state (needs the hidden state before the layer's attention
      on the host - another wait), or the copies of a wrong guess cancelled
- [ ] Two requests at once would use both cards at the same time (each is ~55% busy at decode today)

## Prompt: experts without the fp16 copy (started 2026-10-07)

The prompt path expands each expert to fp16 (13.8 s of a 36K prompt) and oneMKL multiplies it (17 s). At 128 tokens
oneMKL is bound by reading the 33.5 MB fp16 expert (92 us at 128 tokens, 94 at 256), so a kernel that reads the 2 MB
of IQ2_XXS instead could win twice.

- [x] joint_matrix (XMX) GEMMs on Xe2 measured (scratchpad, kept in kernels/engines/glm5next/fused.cpp's notes): A and a col-major B
      straight from global memory reach oneMKL (63-82 TFLOPS); staging B through local memory (any layout) or a
      global scratch is 2-3x slower; a B fragment filled in registers works - lane n holds column n's 16 k in order
      (probed by loading a known matrix; the coordinate form of joint_matrix_apply faulted the GPU)
- [x] fused.cpp: each lane decodes its 16 IQ2_XXS weights into B (two 8-value groups), 32 tokens a sub-group -
      exact against expand + oneMKL; in a real 12K prompt 223 us a call vs 215 for expand + GEMM: no gain
- [x] Cheaper decode: 32 k a lane a step (the group's 8 bytes read once - grid indices, scale, signs - for two B
      tiles), 8 row tiles a sub-group above 64 tokens (large register file). Alone (4096 x 4096, 64 experts
      rotated so the weights come cold): B70 117 us vs 202 for expand + oneMKL at 64 tokens, 135 / 214 at 128; B65
      150 / 344, 224 / 358. Not helping: the grid table in local memory; each sub-group staging B through its own
      local-memory tile and a block load (164-196 us vs 144-146 for the register fill); -fp-model=precise costs
      nothing here
- [x] In the engine (12K, NS_PROFILE=gpu): 172 us a call for the experts of <= 64 tokens vs ~200 for the
      engine's own expand (109 us, faster than the bench's) + oneMKL; above that oneMKL's larger tiles win (with no
      limit gate/up took 9.5 s vs 7.9). NS_FUSED_MAX now defaults to 64: gate/up 7.3 s vs 7.9 of GPU time over
      both cards, the same text; end to end within the run-to-run noise (504-541 tok/s either way)
- [x] The down projection (Q2_K: 16 k share a scale byte and a shift of 16 quant bytes, no table) the same way:
      alone 39-40 us at 64-128 tokens (B70); in the engine the down took 2.8 s of GPU time at 12K vs 3.3 expanded
      with NS_FUSED_DOWN_MAX=128 (the default; 2.9 at 64, 3.25 at 256), the same text
- [x] The B fill, other ways (plain fp16 weights, B70, 128 / 256 tokens: oneMKL 113 / 101 us, a direct block load of
      B 125 / 123, the apply fill 170 / 148): get_wi_data element writes 159 / 149 - the same; apply without the
      fill first is removed by the compiler (3 us: no work). With joint_matrix a register-filled B costs ~30-45 us
      over a block load whatever the form, and B staged through local memory was 2-3x slower: fused decoding cannot
      beat expand + oneMKL for the large experts this way
- [x] Fused decoding in ESIMD (reference/esimd-fused.cpp: xmx::dpas, B decoded with vector selects straight into its
      VNNI layout, A by 2D block loads, the weight words a step ahead): exact, but slower - B70, 64 / 128 / 256
      tokens: best 181 / 278 / 402 us vs expand + oneMKL 318 / 244 / 242 and joint_matrix's fused 110 / 135 / 200.
      Few row tiles a thread: the decode dominates (each thread decodes its B tile for one dpas - RM=1 scales with
      the tokens: 1,372 us at 256); many: too few threads to hide the chain (gathered words -> gathered grid ->
      decode -> dpas); the weights' 16 rows 1 KB apart are a scattered gather a step.
- [x] The three: the grid in local memory, one 8-byte gather for a row's group (no gain over two 4-byte), and a
      work-group sharing the decode - its threads split a 256-k block's 8 groups, write the B tiles into local
      memory (two buffers, a barrier a block), each loads every tile for its own rows. One work-group over all of
      an expert's rows (16 a thread) decodes every weight once: B70 103 / 117 / 157 us at 64 / 128 / 256 tokens
      (joint_matrix 117 / 135 / 200), B65 165 / 213 / 327 (206 / 224 / -); past 256 (B70) or 128 (B65) expand +
      oneMKL wins (512: 335 vs 294 on the B70). In the engine (fused.cpp, ns_moe_fused_gu_esimd; NS_ESIMD_MAX):
      36K 1,052-1,063 -> 1,096-1,097 tok/s
- [x] The down projection (Q2_K) the same way (reference/esimd-down.cpp): exact, slower than the joint_matrix down
      already in the engine - B70 48 / 52 / 69 us at 64 / 128 / 256 tokens vs 39 / 40 / 55, B65 105 / 143 / 132
      vs 47 / 66 / 119. K is 2,048 (8 blocks a row) and Q2_K's decode needs no table: little to share, and the
      work-group's barriers and local-memory round trips cost more than they save. Not used
- [ ] The B65's ESIMD gate | up past 128 tokens (decode-bound: its fewer units)
- [x] The expansion itself: alone (reference/expand.cpp, 2048 x 4096, cold) IQ2_XXS took 67 us on a B70 and 100
      on a B65 a work-item 8 values of one block; two blocks a work-item (their grid loads overlap) 40 / 58 us,
      exact; bigger work-groups or 4 blocks no better; Q2_K (36-37 us) already at its bound. In the engine the gain
      is small: the B65's gate/up expansion at 36K 4.17 -> 3.92 s (its build of the old kernel was already faster
      than the bench's), 36K 989 -> 992 tok/s
- [x] The expansion beside the GEMMs (a side compute queue, expert k + 1 expanded while expert k multiplied, two fp16
      buffers): slower - 36K 919-927 vs 995 tok/s. The GEMMs at these widths are memory bound too (they read the
      33.5 MB fp16 expert): the expansion still waited 4.5 s and the gate/up GEMMs went 4.0 -> 4.5 s. Reverted. The
      fp16 copy itself is the cost: only not making it (the fused kernels, wider) or bigger chunks (fewer per
      token) remove it

## Prompt: the two GPUs in a pipeline (2026-10-07)

- [x] A prompt of several chunks ran the GPUs in turn (each idle through the other's layers). `feed` now runs the
      first GPU's layers on a thread a chunk ahead (NS_PIPELINE=0: in turn): 12K 542 -> 721 tok/s, 36K 527 -> 744
      (B70 first), the same text; in the server's order (B65 first) 12K at 786. Splits (12K / decode): B70 first 22
      731 / 21.1, 26 802 / 20.4; B65 first 22 786 / 20.9, 20 740 / 21.7, 18 502. Smaller chunks lose (2048: 585,
      3072: 609 - the experts' weights serve fewer tokens)
- [x] Measured the stages (NS_PIPE_TRACE=1; NS_PROFILE=gpu NS_PROFILE_PART=i profiles one GPU): 36K, B65 first, a
      chunk 4.1 s on the B65 vs 3.45 on the B70 (with the draft block's 0.22). Splits do not balance it: a layer moves
      its experts too (21: 3.9 / 3.6, the same total; 20: the B70 4.7). The hand-off is ~17 ms a chunk
- [x] The KDA scan was latency bound by occupancy: a sub-group a value column, H x 128 long-running sub-groups - ~3
      waves on the B65 (twice the B70's scan time for the same layers). Now 4 (B70) or 8 (B65) columns of a head a
      sub-group (shared loads, overlapping reductions; NS_KDA_COLS): the B65's scan 1.46 -> 0.54 s at 12K, the B70's
      0.25; 36K 914 -> 989 tok/s, the same text (each column's arithmetic unchanged)
- [ ] The B65's stage is still the longer one: its MLA attention (4.2 s at 36K) and the experts' expand + GEMM
- [x] The server read 36K in 82 s against generate's 37: its two 64K sessions' VRAM pushed experts to host memory,
      whose share went by layer count - the B65 left part of its share unused while 268 of the B70's experts stayed
      on the file only (read from it each prompt pass). A GPU's unused share now goes to the next: 0 on the file,
      36K through the server in 38.4 s; benchy 2K 345 -> 435, 8K 408 -> 762, 40K 720 -> 1,011 tok/s
- [ ] The shares by need both ways (only a later GPU gets an earlier one's spare today)
- [ ] Bigger prompt chunks: 6144 read 36K at 1,072 tok/s vs 961 (generate; 2 x 6K = 3 x 4K at 12K, decode the
      same), but in the server (two 64K sessions reserved) the 40K prompt faulted the B70 (an engine reset, the host
      heap corrupted, exit 139). Suspect the driver's VRAM spill path (the box's known hang): the expert budget keeps
      a fixed 2 GiB besides the arenas, while a pass's own buffers (the stream ring and work buffers, t x ~64 KB each,
      ~1.5 GiB at 6144) grow with the chunk. Budget those per chunk, then retry - with VRAM headroom checked before,
      never by reproducing the fault. Cutting a prompt into equal chunks did not help at 4096 (40K: 975 vs 1,011)
      Measured 2026-10-07 (NS_PIPE_TRACE now prints each GPU's free VRAM after a chunk): at 36K in generate the
      free VRAM after a chunk is about the same at 4096 and 6144 (B65 1.85 / 1.85 GiB, B70 1.54 / 1.38) - the store
      gives the bigger arena its room - and 6144 read 1,060 vs 938 tok/s. 7168 failed with a clean allocation error
      (no silent spill), so the server's fault at 6144 may not be memory at all: its chunks were 5,680 tokens then
      (the equal split, since reverted). Next: a server run at 6144 with plain chunks, watching free VRAM - only
      with a margin of >= 1.5 GiB at every chunk end, stopped at once otherwise
- [x] Done: in the server at 6144 the B70 then ran out cleanly (a 100 MB allocation failed, no fault): the expert
      budget's fixed 2 GiB was the pass buffers at 4096. Now 0.6 GiB + ~352 KB a chunk token (2 GiB at 4096 as
      before, 2.7 at 6144: ~100 experts fewer a GPU); NS_VRAM_GUARD_GIB stops a read before a GPU runs short. 6144
      is the default: benchy 40K 1,011 -> 1,096 tok/s, 2K 435 / 428, 8K 762 / 736, decode 17.7-18.5; the last two
      chunks split evenly when the last is under half (8K: 711 -> 736); free VRAM >= 1.79 GiB at every chunk end
- [ ] Fused gate/up: no gain from more column tiles a sub-group (NB 2: slower, NB 4: spills) or limits past 64 (128:
      the same 7.5 s, 192-256: 8.1)

## Kinds: language, image, video, audio (docs/architecture.md, CONTRIBUTING.md, 2026-10-09)

- [x] The layout by kind: the foundation (crates/: sys, core, gguf, tok, diffusion), each kind's contract and engines
      (llm/, image/, video/), the glue as libraries (glue/models: registry and settings; glue/serve: the servers,
      prompt cache, telemetry), the program (cli/nextsycl); crates renamed nextsycl-*; one kernel library a kind
      (libnextsycl-llm|image|video.so; libnextsycl.so a link to the llm one); the architecture's rules as tests
- [x] The image contract (ImageEngine: request, edit, defaults, steps) and the video contract (VideoEngine: clip
      request, keyframes, stages, cancel, unload); samplers / schedules / sigmas / LoRA references / pictures shared
      in nextsycl-diffusion; a template engine of each kind (llm/example, image/example, video/example) with its own
      kernel, and `nextsycl <kind> selftest`
- [x] `nextsycl llm ...` (the old top-level names still work); `nextsycl image|video engines | selftest`
- [x] The catalog of supported models (glue/models/catalog.json: direct HF links, sizes, SHA-256) and
      `nextsycl models search | pull` (resumes, checks SHA-256, hard-links equal files, `--from` adopts copies)
- [ ] The catalog's files mirrored somewhere of our own (a backup of the links)
- [x] Qwen-Image 2.1 (image/qwenimage21) on SYCL: the Qwen3-VL text encoder in int8 ConvRot (crates/qwen3vl, shared),
      the DiT from the Q8_0 GGUF in half (the text's K/V once a prompt, image rows each step), the RGBA VAE; the
      diffusion kernels (kernels/diffusion = H3's h3sycl.cpp, oneDNN). `nextsycl image check` against
      reference/qwenimage21/ref.py run on the same quantized files: text encoder layer 0 rel 2.8e-3 (36 layers:
      cosine 0.998, int8 activation rounding compounding), DiT block 0 / velocity 2e-4 / 3e-4, 20 Euler steps 2.1e-3,
      VAE max 3 / 255. `nextsycl image gen` (in-process): 1024x1024, 40 steps in 44 s on the B65 (1.07 s / step)
- [x] Qwen-Image 2.1: edits and compositions (up to 8 pictures: the vision tower, the VAE encoder, the block-causal
      prefix; checked at 512 and 1024), true guidance, ComfyUI's 29 samplers and 9 schedulers (crates/diffusion
      samplers.rs, checked against ComfyUI's code); /v1/images/edits (multipart / JSON) and the page's Pictures
- [x] `image start | ps | logs | stop` (the server's container in the background; a stop waits for the pictures in
      progress) and the images API through the switcher (`--images`: /v1/images/* passed on, its model listed);
      ComfyUI's samplers and schedules in the video engine (`sampler` / `schedule` in a job, the studio's Create form)
- [x] `nextsycl serve`: the box at a glance (glue/serve home, wfe/home): every GPU's VRAM by process, the services with
      their links, start / stop image and music servers on a GPU (a held card refused unless forced), the chat model
      through llm.mode; `audio start | ps | logs | stop` (cli served.rs, shared with image)
- [ ] Qwen3.8-Flash-Next's own vision input for chat (transformers has qwen4_exp's vision tower); `image inspect | bench`
- [x] H3's other modes (ComfyUI nodes_minimax_h3.py): Ref2VA (the vision tower in the text encoder, references in the
      denoiser, the ref2va model), keyframe pictures to the text encoder, the effect embeddings, the Fun ControlNet
      Union 2.0 (control videos, masked inpainting); the studio's References and Control panels, its uploads
- [ ] H3: several guides at once (ComfyUI chains MiniMaxH3AddGuide: "multiframe reference"); built-in control
      preprocessors (canny at least; depth / pose need networks); the ref2v turbo LoRA's 4 steps in the studio
- [x] Qwen-Image speed, round 1 (B70, 1024x1024, 40 steps 30.0 -> 26.5 s): layernorm + modulation fused, one
      work-group a row (1.69 -> 0.37 ms a call); prefix + block attention read in place through strides (5.9 -> 5.2 ms)
- [x] Qwen-Image speed, round 2 (B70, 1024x1024, 40 steps): `NS_QI_INT8=1` int8 ConvRot block matrices (rotated and
      quantized per row at load; 26.5 -> 18.5 s, PSNR 39.2 dB against half); the linears' outputs half before the
      gated add; the VAE's single-head attention (D 1152) as two GEMMs + a softmax (228 -> 35 ms). Half: 25.7 s
- [x] Qwen-Image attention: ARK's half flash kernel on sycl-tla (kernels/diffusion/flash -> libnextsycl-flash.so,
      compiled ahead for bmg-g31, loaded on first use; q, k, v read in place through strides): 4.95 -> 2.46 ms a block
      at 4k tokens (~110 TFLOPS); 1024x1024 40 steps half 25.7 -> 24.1 s, int8 18.5 -> 16.9 s. Tried and dropped:
      ARK's persistent schedule (sycl-tla rejects the shape and exits), SageAttention int8 q / k (4.0 ms with its
      quantize passes, 10x the error)
- [ ] Qwen-Image speed, next: the small passes (norm + modulate, gate add, SwiGLU, RMS + RoPE: ~1.7 ms a block,
      ~13%) fused into fewer; the VAE's 3x3 convolutions (0.45 s a picture)
- [ ] Qwen-Image 2.1 Q6_K / Q4_K_M: the dequant reads Q4_K / Q6_K; Q4_K_M also has Q5_K tensors
- [x] LoRAs merged at load (nextsycl-diffusion's `lora`: PEFT / kohya / diffusers files; W += s B A on the GPU before
      half or int8; `--lora NAME[:SCALE]`): the uncensored LoRA checked against ref.py `--lora` (20 steps 8.4e-4;
      without it 3.3e-3)
- [x] Few-step Qwen-Image 2.1 (its "Lightning"): Viggle's turbo (the merged Q8_0, its fused gate_up, and the LoRA)
      and Pruna's 8 / 5-step LoRAs, their sigma presets in the catalog; LoRAs into the small linears too (the
      timestep and modulation ones); checked against the reference (turbo 2.9e-3, Pruna 2.2e-3 after all steps)
- [x] `nextsycl image serve` (OpenAI images API, files, history, progress, GPU; LoRAs by request = a reload) and
      `--wfe` (wfe/image: TSX on vendored snabbdom, H3's scheme; README screenshots, docs/screenshots/shoot.py)
- [x] Engine options: `--opt-NAME` on every command and `options` in API requests, declared per engine
      (`EngineOption` in nextsycl-core: name, variable, value, help, load / request), checked (a typo lists them),
      forwarded as variables in-process and into containers; `<kind> engines` lists them; the image page shows the
      per-request ones. qwen4exp, glm5next and Qwen-Image declare theirs
- [ ] LoRAs per request as a side path (no reload); `image start | stop | ps`; the switcher's images route; edits;
      Qwen-Image (20B) and lightx2v's Lightning LoRAs
- [x] H3 into video/ (from the deployed H3 studio, newer than its sycl-port branch): the engine (video/h3) and its
      jobs on libnextsycl-video.so, the daemon (glue/serve video::daemon), the studio (glue/serve video::studio) and
      its front end (wfe/video), the tools; `nextsycl video start | serve | job | ps | inspect | cancel | rm |
      status | gpus | unload | logs | speech | scene | join | speechpct | plan | stop`; the catalog's minimax-h3
      entries; checked against H3's h3d (same spread, same speed)
- [x] The GPU box's production on nextsycl video (2026-10-09): nextsycl-video.service (the daemon) and
      nextsycl-video-studio.service (the studio on :8090, which also switches chat); h3-engine / h3-studio disabled
      (kept for a rollback); the H3 studio's state, clips, LLM modes and templates carried over
- [ ] Video: dedupe against the foundation (its gguf / safetensors / tokenizer / LoRA readers -> crates/), the text
      encoder into crates/qwen3vl (the 32B GGUF form), the ESRGAN weights in the catalog
- [x] The audio kind (2026-10-09): its contract (description, lyrics, length; a waveform out, WAV with an INFO chunk;
      progress that cancels), template (audio/example + kernels), libnextsycl-audio.so, `nextsycl audio engines |
      selftest | gen | check | serve`
- [x] MiniMax Music 3 (audio/minimaxmusic3): the 8B language model and depth decoder (half; int8 opt-in), the
      condition encoder, the 2.4B flow transformer over 200-frame windows with their overlap, the Flow-VAE decoder
      (oneDNN convolutions); checked stage by stage against diffusers' own code on the same files
      (reference/minimaxmusic3/ref.py: 51 forced frames 4e-4, 30 flow steps 1.2e-3, decoder 8e-7, three windows
      stitched 3.6e-3). Served: the reference's /v1/audio/speech + the page (wfe/audio). B70: a minute of song in
      118 s half, 89 s int8; B65 155 / 136 s
- [x] Music: the flow transformer in int8 ConvRot (`--opt-dit-int8`): flow 42 -> 33.5 s a minute of song, rel 8.6e-3
- [ ] Music speed: the
      frame's eight host round trips (draws on the GPU, the frame as one graph); the depth decoder's seven passes
      read 8 GB a frame in half - int8 only, or a smaller form. Tried: the flow stage beside the frames on a second
      queue of the same card (the card takes the two in turns: 118 -> 114.5 s; dropped)
- [x] The last Python on the serving path ported (2026-10-09): the model switcher (`nextsycl switch`, glue/serve
      switch.rs) and the GPU telemetry sampler (`nextsycl gpustat`, gpustat.rs), checked side by side with the scripts
      (the same JSON field for field; models list, aliases, plain and streamed answers) and in service on the GPU box
      (the Python units kept as *.bak-python). What is left outside Rust + SYCL there: the chat front end (Open WebUI,
      third-party), and dev tools (reference/ dumpers, the screenshot script, the page smoke test)
- [ ] Music: the two cards together as the reference runs (the language model on one, flow + decoder on the other,
      the windows streamed between them) when chat is not on the B65; `audio start | stop | ps`; 32 kHz output (the
      reference server resamples); the caption rewriter (MiniMax's music-caption-rewriter skill) as an option

## Several models (docs/engines.md, 2026-10-08)

- [x] The split: ns-runtime (the contract: Engine, opaque Session / Decoder / Checkpoint, Sampler, the registry by
      architecture), engines/glm5next (GLM's model description + engine + tools, moved as they were), its kernels
      in kernels/engines/glm5next; the server, the CLI and the prompt cache drive only `dyn Engine`. The same
      numbers before and after: spec-check (2 / 3 / 5 rows, short and long) and batch-check exact, llama.cpp
      parity 0.995209, generate's text identical
- [x] kernels/strata refreshed to the user's intel-arc-0.1.40 (3f37281): GLM unchanged (spec-check, batch-check
      exact, parity 0.995209)
- [x] engines/qwen4exp (the files say `qwen4exp`): Qwen3.8-Flash-Next IQ2_XS on both cards, every weight and expert
      in VRAM (B70 layers 0-38, B65 39-47; 36.5 GiB, 4.7 s from the page cache). The glue
      (kernels/engines/qwen4exp/qwen.cpp) is Strata's verify window (verify.cpp at 0.1.40) per stage, eager, all
      experts planned on the device; the PLE rows hashed and read in Rust. spec-check 2 / 4 / 8 rows (short, 3K,
      12K) and batch-check exact (max diff 0). Decode 37 tok/s, prompt 142 tok/s (8-token windows)
- [x] The chat templates by engine (`EngineKind::chat`: glm_chat, qwen_chat); ns-tok's `qwen35` pre-tokenizer
- [x] qwen4exp: the prompt path (kernels/engines/qwen4exp/prefill.cpp = Strata's prefill.cpp default path: the HC
      reads as BF16 GEMMs, Gemm::native projections, the GDN / QSA chunk kernels, the selection as GEMM tiles, the
      experts grouped on the host and run dequantized to FP16 + oneMKL, the writes fused with the next norm); a
      prompt reads as Strata's does (all but the last token in chunks, the first 256, then a window). 3K 702 tok/s
      (2048 chunks) / 805 (4096), 12K 1,023 tok/s (was 142). Strata's own output on the same ids: identical text
      on a short prompt; 3K identical for ~65 tokens (Strata on one B70 computes its non-resident experts on the
      CPU, so not bitwise there). spec-check 8 rows and batch-check still exact
- [x] qwen4exp at Strata's own numbers (INTEL_PERFORMANCE.md, benchy v1 on the B70), one card: the MTP draft layer
      (kernels/engines/qwen4exp/mtp.cpp = Strata's MtpDrafter, NS_QW_MTP=<mtp rt dir>; --spec 4 --spec-min-p 0.5 as
      NS_QW_SPEC / NS_QW_SPEC_MIN_P), the window / commit / drafter as SYCL graphs a session (recorded when it is
      made), the cold experts in pinned host memory read over PCIe by the plan (resident_plan_set_mirror; ranked by
      Strata's expert profile, NS_QW_EXPERT_PROFILE), greedy picks on the GPU (Sampler::greedy). 20 / 2,185 / 8,000 /
      40,000 tokens: prompt 62 / 795 / 1,180 / 1,254 tok/s (Strata 60 / 745 / 1,050 / 1,117), decode 76 / 80-87 /
      82-89 / 70-75 (70 / 77 / 78 / 72). Two cards are slower (the B65 runs a layer at half the B70's speed)
- [x] qwen4exp served (2026-10-08): the chat template, the prompt cache (the drafter's K/V in the session state),
      the Coder IQ1_M (256 experts: the count a parameter; its expert blobs to 2.5 MiB: 4 MiB staging) and Swift 1.5
      (its F32 router converted to BF16 at load), the projection modes (Strata's cvec kernel, `cvec.cpp`; the same
      text as Strata's). Strata no longer serves: the Open WebUI switcher and the H3 studio start nextsycl
- [x] `nextsycl models` (list / add / download / remove / enable / disable): one registry the switcher and the
      studio's mode file read
- [x] qwen4exp's sampled decoding coupled (Strata's STRATA_SPEC_COUPLED): a sampled request's rows drawn on the
      GPU, Philox(seed, position), and each draft drawn from the draft layer with the same chain and the uniform of
      the row that verifies it. 18 requests (3 prompts x 3 seeds, 400 tokens), host draws vs coupled: 0.7 63.6% ->
      68.9% of drafts accepted, 56.6 -> 59.6 tok/s; 1.0 59.9% -> 70.1%, 55.4 -> 58.7. Greedy unchanged (the same
      text); a request's `seed` replays its text. NS_QW_COUPLED=0: the host draws
- [x] Tool calls on Qwen3.8-Flash-Next: the template's tool path (ns_tok::qwen_chat with the request's tools, the
      `tool` role, earlier calls), the calls parsed out of the answer (qwen_tool_calls) into OpenAI's tool_calls,
      arguments typed by the schema, held back from the stream; a weather round trip (call, result, answer) through
      the switcher, streamed and not
- [ ] GLM-5.3's tool path (its template's <tool_call> form)
- [ ] qwen4exp's decode window, 2 ms behind Strata's. The same 2,185 ids, greedy, fresh, 169 drafts accepted in
      both: Strata 87 rounds in 3,182 ms (36.6 ms a round: its window ~28.7, commit 3.6, drafting 3.6; 80.4 tok/s);
      here the window 30.5 on average (T=4 31-33, T=3 28, T=2 25, T=1 22), commit 0.8, drafting 3.1 - about 35 ms a
      round, 79-86 tok/s. The window's 2 ms need a kernel-by-kernel profile of both
- [x] qwen4exp's first pass (a model just loaded decoded 10-20% slower than the same text again): the PLE rows. A
      window reads 16 rows a token (64 at T=4) from the 27 GiB table, hashed from the token's n-grams; read one after
      another, the rows not in the page cache came off the SSD in turn (~6 ms a window) - and this box's page cache
      cannot keep 63 GB of model. posix_fadvise(WILLNEED) on all of a window's rows first: with the files dropped
      from the page cache 71.8 -> 83.9 tok/s (2K benchy ids), warm 86.0 / 86.1. NS_QW_PLE_PREFETCH=0 off. (Not the
      IOMMU, not the GPU: an ns_touch of every expert at load was tried and removed)
- [x] qwen4exp's VRAM on one card: 6.2 GiB free after the weights load, 1.2 GiB once the sessions (two, 128K + 32K),
      the draft layer and the prompt path's buffers are made and a 40K prompt has run - the planner's 1.5 GiB guard.
      Nothing left to give the experts (the Coder's 2.5 GiB in host memory are those buffers)
- [x] qwen4exp on both cards (every expert in VRAM; the split now leaves each stage its prompt buffers, and the head
      and draft layer a card that takes every layer): slower than the B70 alone - Coder warm 2K / 8K / 40K 70.8 /
      69.6 / 70.7 tok/s (B70 first) and 58.2 / 54.7 / 53.5 (B65 first) vs 77.5 / 75.9 / 77.2; IQ2_XS 70.6 / 68.5 /
      66.8 vs 83.0 / 81.3 / 75.9. One card stays the default
- [x] The GLM-only kernels out of the shared part: glm.h, engines/glm5next's ffi.rs and ops.rs

## 256K context (plan: docs/256k-context.md, 2026-10-07)

- [x] 1. Per-slot context (`--ctx 262144,32768`: a session each; a request goes to the smallest free one it fits,
        else waits; a checkpoint restores into any session as long; its tokens end at its session's context). 256K
        served: 253,308 tokens read in 271 s (933 tok/s, 29.6 Wh), decode at that depth 14.5-15.6 tok/s, a short
        request in the 32K session at 21 tok/s; the attention reserve 3.8 GiB (B65 1.58, B70 2.21), ~320 experts
        fewer in VRAM, 0 on the file; the checkpoint 3.5 GiB; a follow-up on the document in 4.4 s (reused from it)
  - [x] 128K in generate: the prompt at 998 tok/s (126 s), the answer right, stages ~3.9 s a chunk to the end, 0
        experts on the file. Decode at that depth was 13.0 tok/s (21-22 short): per layer and pass the indexer's
        top-k 330 us a row (one histogram's counters took every add; the selection written with four collectives
        a tile) and MLA's attention 1,047 us (a work-group walking 2,051 cells a load at a time). Now 32 histogram
        copies and a one-pass selection (45 us, the same selection), and a decode-width attention kernel (a
        sub-group a cell, 16-byte loads, 4 cells a step: 444 us): 17.1 tok/s at 128K, exact (spec-check,
        batch-check, the same text)
  - [ ] Decode at depth, what is left a layer: the indexer's projections (~300 us), attention 444 us (each of the
        64 heads' work-groups reads the same 2 MB of cells)
- [x] 2. A needle check: `nextsycl bench --needle` - 9 of 9 found at 10 / 50 / 90% of 32K, 130K and 249K
        (docs/benchy/needle-2026-10-07-f16.md; 249K prompts read in ~267 s). Indices: positions are int32 cells,
        every size and product in the kernels int64 - no limit below 2^31 tokens
- [x] 2b. Chunk sizes: the 12K prompt read in chunks of 2048 gives the same text as 4096; 3072 differs at one
        word ten tokens in (oneMKL tiles by the row count: the sums' order moves, a near tie flips)
- [x] 3. NS_KV=q8: the latents as [16 fp16 scales][512 int8] a row (544 bytes vs 1,024), written by ns_to_q8row,
        read by the decode kernel (a lane's 32 values one block) and gathered to fp16 for the prompt's GEMMs. Exact
        within itself (spec-check, batch-check). Against llama.cpp at 3K: final logits cosine 0.99521 vs fp16's
        0.99640, the same top 5. 12K prompt 852 tok/s (= fp16); 128K: 1,019 tok/s, decode 17.2 at depth (fp16
        17.1), the attention reserve 0.95 GiB vs 1.63. The needle on q8: below
- [ ] 3b. The indexer's pooled keys in fp16 (128 -> 64 bytes a token and layer); a 4-bit latent form
- [ ] 4. The indexer fused (XMX scores, the heads' sum and the top-k in one kernel; no T x H x pools scores)
- [x] 5. A disk tier for the prompt cache: checkpoints pushed out of memory written to NS_CACHE_DIR (32 GiB),
        mounted from there when they are the best prefix; emptied at start (a checkpoint fits only its process's
        sessions)
- [ ] 6. Serve it: NS_CTX per slot, the HTTP limit, the tokenizer on ~1 MB, the client's context setting

## Decode, what is left (2026-10-07: 18-19 tok/s, MTP on)

Per token: routed experts 27.7 ms (the kernel ~0.3 ms a layer-pass of the ~1.2: the rest waits on PCIe copies of the
misses), KDA 10.9 (its products at ~613 GB/s), MLA 4.7, the draft block 3.2, the shared expert 2.1, hc 1.5.

- [x] An arriving expert's gate/up launched once its gate | up bytes are in, its down once the rest is (two copies up,
      moe_grouped in two phases) - +1% (most misses are prefetched now; little demand wait is left)
- [ ] The card swap (below) - the misses' copies at x16

## To try

- [ ] Swap the cards' slots: the B70 to the x16 slot, the B65 to Gen4 x4 - then NS_SPLIT=15 (the B65's 12 MoE layers
      all resident, no swaps over its slow link; the B70's swaps on x16). Simulated (reference/expert-cache/splits.py)
      copy time a pass 5.9 ms vs 8.8 at today's best; with today's even split it would be 18.6 (worse). Measure with
      `nextsycl bench` after.
- [x] Several requests at once (branch `batching`): forward_batch + an engine thread with NS_PARALLEL sessions -
      bit-exact; 2 clients 18.1 -> 20.3 tok/s in all, the second's first token 10.4 -> 2.0 s
- [x] Batch: MLA's projections once for all rows (mla_batch) - +3% (24.1 -> 24.9 tok/s for 2): the batch is bound by
      its experts - two conversations want ~15 distinct experts a layer, not 8 (49 ms a 2-row pass of 90)
- [x] Batch: a prompt read in groups between rounds - 4 chunks (16K tokens, ~17 s) while others decode or wait, 8
      alone (a request arriving meanwhile waits one group); the GPUs shared by time while others decode (after a
      group of t seconds they decode for t). A 256K prompt read at once held every other request for 4.5 minutes;
      one decode step a group gave a chat 0.23 tok/s. Measured: a chat sent 20 s after a 253K prompt answered in
      55 s (it waited one group, 19 s; then 2.9 tok/s), the long prompt 304 s vs 271 alone. Checkpoints at the stops
      as before
- [x] The chat's decode beside a prompt read was 2.9 tok/s: a short prompt read after a long group reset the
      decode share to its own second (now the round's reading in all). Now 17.9 tok/s beside a 253K read, the chat
      done 35 s after it was sent (19 of them waiting for the group in progress). The big arena also stays with the
      prompt while one is read (`hold_arena`: no lending back and forth at each switch)
- [x] A request arriving mid-group waited for the group: a prompt read alone now goes in one pipelined read that
      stops at a chunk's end once a request waits (`feed_until`; groups of 8 alone had cost 8%: 294 vs 273 s at
      253K); the shortest read goes first and a request that starts decoding gets a group's time before the next
      group. A chat sent 20 s into a 253K read: answered in 9.8 s (was 258 s that morning), 18.4 tok/s; the long
      prompt 280 s vs 273 alone; the window holds back long reads only (it had made a second short request wait 11 s for
      its first token)
- [x] The disk tier, measured: a 256K prompt's checkpoint pushed out by a 128K one, its follow-up mounted from the
      file - 4.5 s, the right answer
- [x] The disk tier outlives the server: a subdirectory per fingerprint (the model file and size, the cache's form,
      MTP), the tokens beside each file, a stop writing the checkpoints still in memory; unused for 24 h (or past
      32 GiB, least recently used first) removed. A 12K checkpoint from the 64K mode mounted by the 256K mode
- [x] Decode, from Strata's lessons (2026-10-07): the GPUs are busy ~all of a decode (summed device time = the wall
      time) - routing on the device or graph capture would save little; the misses' copies are the cost (~16
      swaps a token). The q8 latent cache by default (~100 more experts in VRAM in the 2 x 64K server: benchy the
      same or a little faster - docs/benchy/v1-2026-10-07-q8.md). Pinning experts by a usage profile LOST in
      simulation (reference/expert-cache/pinning.py, the other topic's profile: 25% pinned +12% misses, 90% +140%);
      filling VRAM by the profile at load won the first answer: decode's requests counted, saved at a stop
      (NS_EXPERT_PROFILE, ~/.cache/nextsycl/prompts/expert-profile.txt, older halved each save) - a new topic's first
      256 tokens 3,951 -> 3,561 swaps, 23.2-23.5 -> 24.0-24.6 tok/s
- [x] Decode: missed experts on the CPU (reference/cpu-expert: ggml's AVX2 IQ2_XXS / Q2_K dots in Rust, the token
      quantized to int8 per 256, cosine 0.99997 to float). A 9950X, 8 threads of a spinning pool, the expert cold
      from DRAM: 138 us an expert-token (a copy up ~270 us), 227 / 341 us for a 2 / 3-row verify pass. But the
      ceiling is small: with every miss free (NS_FREE_MISSES=1, a measurement - wrong output) decode without MTP
      goes 18.8-19.0 -> 21.3-21.6 tok/s, +13%: prefetch and the async swaps already hide most of the copies. A CPU
      share would win a part of that (CPU-computed experts never come up, 8 cores spinning through decode) - not
      built. The card swap has the same ceiling
- [x] Decode's Q8_0 products (the KDA projections): Strata's wide32 kernel already has the aligned loads (load16_a2).
      Measured alone (reference/mmvq-bw.cpp, 8192 x 4096, random weights, cold): B70 488 / 551 / 538 GB/s at 1 / 2 /
      3 rows - 80-90% of its ~608; B65 374 / 384 / 356 (~62%: its fewer units' decode). Lifting the B65 to the
      B70's share would save ~1-2% of decode: not pursued. (The ~330 GB/s seen in the profile was both cards and
      all six projections, the small ones included.) Lesson: a constant fill is compressed by the GPU's memory -
      bandwidth tests need random data; and ns_copy_to is queued - its host buffer must outlive it (a bench that
      freed it faulted the B70, engine reset 4, recovered)
- [x] Sampling at a temperature sorted the whole vocabulary (154,880) for its top 256 a token (~2.5 ms): now a linear
      selection, then the 256 sorted (the same order, the same tokens - the drafts accepted matched exactly). Decode
      at temperature 0.7 21.3-21.6 -> 22.7-23.0 tok/s, at 1.0 20.4-20.9 -> 21.6-22.2 (greedy 23.0-23.8)
- [x] Prompt-lookup drafts (NS_NGRAM=K, off by default): the up to K tokens that followed the last 3 where they
      occurred before, verified in one pass. Exact, and ~90% accepted on a copy-heavy answer (adding comments to a
      file: 902 of 979 at K=4) - but slower at every K (17.46 tok/s without, 16.85 / 17.13 / 16.87 at K = 2 / 3 / 4):
      a 5-row pass took ~280 ms against a 2-row one's ~110 (2.5x for 2.4x the tokens) - each row brings its own 8
      experts (swaps +11-15%), the pass's cost grows almost with its rows. The draft block's 2-row passes (93%
      accepted on that answer) are the better trade on an MoE
- [x] A real bug on the way: verify passes of 3+ rows drifted at long context (spec-check --rows 3 at 3K: 3.2e-2).
      The indexer's ring of the last 4 tokens' keys (slot pos % 4) - a 3-row pass's rejected rows overwrote slots
      of positions the next pool still needed. Now 8 slots, every row of a pass written: 2 / 3 / 5-row passes exact
      at short and long context. NS_DRAFTS=2 was affected too. Disk checkpoints of the old layout: fingerprint nsck2.
      spec-check: --rows R (2..5), --layers --at N --row k, tensors matched by name
- [x] Speculative sampling at a temperature (on; NS_SPEC_SAMPLING=0 for the argmax drafts): the draft sampled from
      the draft block's distribution (the request's temperature and top-p), kept with min(1, p/q), else a token
      from max(0, p - q) - what is committed is distributed exactly as plain sampling; greedy is unchanged (the
      same text). Three prompts x temperature 0.7 / 1.0 x 384 tokens (docs/benchy/spec-sampling-2026-10-08.txt):
      acceptance 61.6% -> 65.0%, 20.37 -> 20.59 tok/s on average; the free-running story most (54% -> 66% at 1.0),
      a short poem at 0.7 lost (noisy). A sharper draft distribution (NS_SPEC_DRAFT_TEMP 0.5 / 0.25: 65.8% / 64.1%)
      the same within the noise. The sampler is a trait now (ns-runtime's Sampler: sample, dist, uniform, record)
- [ ] Decode is spread out now (per token: experts' kernels, KDA 10.6 ms, MLA 4.9, the draft 2.5, the shared expert
      2.2): no single kernel holds a big share. Levers left: more VRAM (fewer misses: <= 13%), MTP acceptance
- [ ] Batch: the draft block in batches; per-request energy (concurrent requests share the counters - each counts
      both)

## LogProbChain (an API option)

A request option `logprob_chain` (beside `logprobs` / `top_logprobs`): logprobs corrected for what the model is
only echoing from earlier in its own turn. The thinking tokens skew the answer's logprobs: a token the thinking
already wrote comes out near-certain because the model copies it, not because it is sure of it.

- [x] v1 (2026-10-06): `logprob_chain: true`; the attention = the mean over heads and MLA layers of a one-token
      pass's softmax (Glm::capture_attention); chained = logprob + sum over equal turn tokens of attention x their
      own logprob; `chain` on each entry (with the top 3 echoes) and `chained_logprob` on each alternative
- [ ] The mean dilutes: an echoed token gets 1-3% of the attention while the turn gets 20-30% - try the max over
      heads, the heads that attend most to the turn, the last layers only
- [ ] For each output token X, one attention vector over the context: combine the heads of the attention layers -
      experiment: all MLA layers averaged, the most informative layers only (by entropy or by how peaked they are),
      the last layers, weighted by head; MLA's absorbed scores (q~ . c) give per-position weights cheaply; KDA layers
      have no attention matrix (only a state) - leave them out or approximate
- [ ] For X's candidates (its top logprobs), any candidate that equals a token at a position of this turn (thinking
      or answer generated so far) that X attends to: skew its logprob in proportion to THAT earlier token's own
      logprob times the attention weight on it (an echo of a confident earlier token is discounted; of an uncertain
      one, kept low) - the exact formula is part of the experiment
- [ ] Keep each generated token's own logprob (already computed for `logprobs`) so the chain can look it up
- [ ] Return the chained logprobs beside the raw ones (e.g. `chained_logprob` on each entry), and say which earlier
      positions dominated
- [ ] Validate: questions with known answers, with and without thinking; does the chained confidence of the final
      answer track correctness better than the raw one (calibration)

## Speed, later

- [x] Prompt chunks read host-slot experts in place over PCIe (no swaps); decode still swaps (faster there: 12.5 vs
      11.2 tok/s with NS_DECODE_DIRECT=1)
- [ ] Expert swaps at decode: 32 ms/token with MTP (2-row passes touch more experts). Strata's answer (its 2.6 -> 40.9 tok/s on
      the B70): the GPU reads missed experts straight from pinned host memory in the expert kernel (the grouped
      kernel already takes a pointer per expert - pass the host slot's address), a share of the misses (pcie_frac
      ~0.55) and LRU admission for the rest. Pinned memory belongs to the context that allocated it: each part reads
      only its own mirror (Strata #1054 hung two B70s on a cross-context fill)
- [x] Deeper drafts: NS_DRAFTS=2 chains the draft block on its own output, a 3-row verify - exact (the same greedy
      output), 2.36 tokens a pass vs 1.83 (the 2nd draft accepted 71%), but a 3-row pass costs 33% more (3 tokens'
      experts, the block run twice): 18.4 vs 19.1 tok/s. One draft stays the default
- [ ] Launch overhead: SYCL graph replay for the decode step
- [ ] Prefill speed

## Later

- [ ] IQ3 variants of GLM (NOTES.md)
