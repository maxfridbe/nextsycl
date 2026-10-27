## MOSS-Image-SFT (OpenMOSS): the port

Text to image, 5.1B, Apache-2.0, released 2026-10-16. Served here from unsloth's GGUFs
(`unsloth/MOSS-Image-SFT-GGUF`: the DiT only, Q3_K_XL ... BF16, 2.8 - 10.2 GB). Upstream:
`OpenMOSS-Team/MOSS-Image-Base` / `-SFT` (diffusers `MOSSImagePipeline`; the whole model is one file,
`pipeline_moss_image.py`, 440 lines - kept beside the reference as `reference/mossimage/pipeline_moss_image.py`).
Prompts are natural language, booru-style tags, or a JSON layout (a summary and items with bounding boxes: where
text and objects land, and what the text says).

### Files

| Part | File | Size |
|---|---|---|
| DiT | `unsloth/MOSS-Image-SFT-GGUF` `MOSS-Image-SFT-UD-Q4_K_XL.gguf` (recommended), `-Q8_0`, `-UD-Q5_K_XL`, `-UD-Q6_K_XL`, `-Q4_K_M`, `-UD-Q3_K_XL`, `-BF16` | 3.42 GB |
| text encoder | `unsloth/Qwen3.5-9B-GGUF` `Qwen3.5-9B-Q8_0.gguf` (the text model; no mmproj) | ~9.5 GB |
| VAE | `unsloth/MOSS-Image-SFT-FP8` `vae/diffusion_pytorch_model.safetensors` (FLUX.2 VAE, fp32) | 250 MB |
| tokenizer | the same repo's `tokenizer/` (Qwen2 tokenizer.json + chat_template.jinja) | 20 MB |

The GGUF's metadata is four keys (`general.architecture = moss_image`, ...); its tensors are the PyTorch state-dict
names (`layers.N.proj.q.weight`, ...), 463 of them, `ne` = [in, out]. Types: Q4_K / Q5_K / Q6_K / BF16 matrices, F32
norms and biases. `rope.bands` [16] is a derived constant (recomputed: `1 / 10000^(i/16)`).

### Architecture (from `pipeline_moss_image.py`)

- **Latents**: the FLUX.2 VAE's, packed 2x2: a 1024x1024 picture is 128 channels x 64 x 64 (patch 16 px). Sizes are
  multiples of 16. Noise is fp32.
- **DiT** (`MOSSImageTransformer`): dim 3072, 24 heads of 128, 42 blocks, ffn 8192, one stream: the image tokens
  (`video_embedder`: Linear 128 -> 3072) followed by the text tokens (`text_embedder`: Linear 4096 -> 3072 and two
  small text blocks with biases and fused qkv, run on the text alone), then 42 `_Block`s over both.
  - block: `x += o(attn(q,k,v of (rmsnorm1(x) * (1 + scale_a) + shift_a))) * gate_a`; `x += down(situ(up, gate)) *
    gate_f` with `rmsnorm2` and `(shift_f, scale_f)`. q / k have a per-head RMSNorm (128), no biases in the blocks.
  - the modulation is **shared**: `time -> shared_block_modulation (Linear 3072 -> 6 x 3072)` once a step, the same
    six vectors for every block (no per-block adaLN).
  - `situ(h, g) = 4 tanh(g/4) sigmoid(g) * 25 tanh(h/25)` (a soft-clipped SwiGLU); the FFN is one `ffn_up_gate`
    matrix (3072 -> 16384) split in two.
  - RoPE: 4 axes (r, f, h, w) over 16 bands (`head_dim / 8`): image tokens (0, text_len, h - (H-1)/2, w - (W-1)/2),
    text tokens (0, i, 0, 0); phase = coordinate x band, `[sin | cos]` per axis, applied half-split (`rotate_half`).
  - time: sinusoid (256, cos then sin, float64) -> Linear -> SiLU -> Linear; the head: `rmsnorm(x_img) * (1 + s) + t`
    from `video_head_modulation`, then Linear 3072 -> 128.
  - **skip layers**: a forward can skip whole blocks by index; the negative branch of CFG skips block 22 (that, not a
    negative prompt, is the default: `negative_prompt = ""`, `uncond_skip_layers = (22,)`).
- **Text encoder**: Qwen3.5-9B (`Qwen3_5Model` text side): 32 layers, hidden 4096, 3 gated-DeltaNet layers then 1 gated
  full-attention (16 heads, 4 kv, head 256, output gate, 25 % partial rotary, mrope-interleaved theta 1e7), FFN 12288.
  The pipeline feeds `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
  enable_thinking=True)` and takes `last_hidden_state` (after the final norm) of **every** token: the DiT's text
  sequence is the whole prompt (up to 4096 tokens), unpadded, one prompt at a time. No LM head.
- **Sampler**: `MOSSImageScheduler`: a shifted grid (shift 5), snapped to the nearest of the 1000 training sigmas
  (`unique_consecutive`), Euler on the velocity: `x += (sigma_next - sigma) * v`. CFG 3 over fp32 predictions,
  50 steps default. Optional SLG (an extra skipped positive forward) - not needed at first.
- **VAE decode**: `z * sqrt(bn.running_var + 1e-4) + bn.running_mean`, unpatchify (128 -> 32 channels, 2x2),
  `AutoencoderKLFlux2.decode` (decoder channels 96/192/384/384, 2 resnets a block, mid attention, group norm 32,
  post_quant_conv), clamp [-1, 1]. VAE runs in fp32 upstream.

### What exists in the repo to reuse

| Need | Have |
|---|---|
| quantised GEMM Q4_K / Q5_K / Q6_K / Q8_0 / BF16, rmsnorm + modulation, rope tables, attention, gate_add, conv2d | `nextsycl_diffusion::kernels::Nsd` (check Q5_K) |
| GGUF + safetensors readers, tokenizer | `nextsycl_gguf`, `nextsycl_tok` |
| a 2D VAE decoder | `image/mingimage/src/vae.rs` is Wan's (RMS norm); FLUX.2's is a group-norm SD-style VAE - new |
| gated DeltaNet prefill (conv, L2 norm, delta-rule scan, gates) | `kernels/llm/qwen35` + glm5next's scan: an image engine may not call another kind's library - copy into `kernels/image/mossimage` |
| Qwen3.5 layer layout | `llm/qwen35/src/engine.rs` (read for the tensor roles), the 27B shares the architecture |

### Plan (each step checked against the reference before the next)

0. Reference (`reference/mossimage/ref.py`, CPU float32, venv on hardac1): the DiT's classes from the upstream file
   with the GGUF dequantised into them, the text encoder from the original bf16 shards, the VAE from diffusers.
   Dumps: ids, te_final, sigmas, lat_0, per-step (lat_i, vel_i; vel with the skip), per-block outputs at step 0,
   lat_final, the picture.
1. Engine crate `image/mossimage`, kernels `kernels/image/mossimage`: loader + role check; the DiT; check
   `nextsycl image check` vs the dump (block by block, then step by step).
2. VAE (FLUX.2) and the sampler: latents -> picture.
3. Text encoder (GDN + gated attention) and the tokenizer/template; check vs `te_final`.
4. Registry, model cards (one per quant), `image serve`, the front door (it is `image.generate`; layout JSON is
   just a prompt), docs, README speed table.
5. Silo tuning only after 1-4 are exact.

### Status (2026-10-26)

Steps 0-3 done and checked on the B70 (`nextsycl image check <dump dir> --model moss-image-sft-q4kxl`, dumps made by
`reference/mossimage/ref.py`: 512 px, 12 steps, cfg 3, seed 7; stages `sigmas,tokens,te,dit,steps,vae,full,bench`):

| Stage | Against the reference (CPU float32) |
|---|---|
| sigmas / tokens | 3e-8 / the same 21 ids |
| text encoder (Qwen3.5-9B Q8_0, F32 recurrences) | final hidden rel 1.4e-2 (cos 0.9999; Q8_0 weights, half activations) |
| text embedder, video embedder | 2.8e-4 / 2.9e-4 |
| DiT, block by block (step 0) | 1e-4 .. 3e-4, the last block 2e-3 (cos 0.999998) |
| velocity, every sampled step | 4e-4 .. 2e-3 (cos 0.999998 +) |
| 12 CFG steps, latents | rel 3e-2 .. 4e-2, cos 0.999 |
| VAE | 2.9e-4 |
| whole picture, all ours | cos 0.997 to the reference's; the same fox |

The last-latents number is a noise floor, not a bug: every product reads half activations (rounding ~5e-4 a step),
and 12 steps of CFG 3 amplify that ~30x. A forward on the same latents is bit-identical run to run; what once made two
runs differ (rel 2e-2 to 1.4e-1 in the latents) was the step modulation's float32 product over all the steps' rows
(summed in another order each call, ~1e-6): `Dit::mods` now does one step a product and the sampler is
reproducible - the same seed, the same picture. Judge a change by the velocities and blocks, which do not move.

### Speed (B70, 2026-10-26; a forward = one DiT pass, a step = two with CFG)

| | 1024 x 1024 (4,096 image tokens) | 2048 x 2048 (16,384) |
|---|---|---|
| DiT forward, first port | 0.669 s | |
| + the silo's fused passes | 0.575 s | |
| + whole-matrix products (no slicing) | 0.509 s | |
| + matrices expanded to half at load (`NS_MOSS_EXPAND`) | 0.464 s | 2.82 s |
| 50 steps with CFG | ~46 s | ~290 s |
| text encoder, 32-token prompt / 3,970 tokens | 0.11 s / 0.92 s (3.4 s before the silo's attention) | |
| VAE decode | 0.32 s | 1.5 s |

What a forward holds now, from `--stages bench` (the per-op list drains the queue between ops, so it adds ~0.25 ms an
op; the sum of its rows runs ~30 % over the forward): the products run at 143-162 TFLOPS on the expanded half
matrices (the B70's practical peak); the attention at ~110 TFLOPS (oneDNN's fused SDPA, 16 % of a 1024 forward, 36 %
at 2048); the silo's three fused passes together cost 0.6 ms a block (add + norm + modulation 0.07 ms, q/k norm +
rope 0.13 ms, the soft-clipped gate 0.35 ms: the residual add rides along in the next norm, the up / gate product is
written in half); a forward's rotary table and buffers are kept per shape (10 ms a forward saved); the host queues a
forward in 12 ms, so it is GPU bound.

- **Silo** (`kernels/image/mossimage/silo`, `dist/silo/libnextsycl-mossimage.so`, `NS_SILO=0` for the A/B): add +
  norm + modulation, q/k RMS norm + rope in one launch, the soft-clipped gate from half rows; `_supported` gates it
  (sub-groups of 16, 3,072-wide rows). Without it the engine runs the shared kernels, ~8 % slower.
- **Whole-matrix products** (engine, not a kernel): the shared quantized linear expands ~64 MiB of rows at a time and
  copies each slice's products into their columns; the DiT's matrices (16384 x 3072 fused up / gate, the packed qkv)
  now expand once into one scratch and run as one product straight into the output (4.4 -> 3.1 ms for the up / gate
  at 4,128 rows). The values are the same; q, k, v are expanded side by side and make the packed rows in one product.
- **Resident half** (`NS_MOSS_EXPAND`, auto): with 8 GiB to spare after each block the quantized matrices are
  replaced by their half expansion at load (10.2 GB for the Q4_K_XL file; the text encoder's 9.5 GB fits beside it on
  a 32 GiB card), so a forward skips ~1.1 ms of expansion a block.
- **Left** (measured, not chased): the attention at ~110 TFLOPS (a hand-written flash kernel would have to beat
  oneDNN's by a third to matter below 2048 px); the text encoder at long prompts, now
  1.0 s at 3,970 tokens (3.4 s with the shared causal attention, a work-group a (head, query) walking the keys one by
  one). `NS_MOSS_PROFILE=1` splits it: 0.08 s in the 8 full-attention layers, 0.34 s in the 24 DeltaNet layers (the
  delta-rule scan is sequential in the tokens), 0.43 s the FFN and norms (now whole-matrix products too; was 0.55 s with the shared quantized product's slicing and
  float32 outputs: the DiT's whole-matrix trick is not applied here yet). The silo's attention takes query blocks of
  512 rows against the keys up to their end as half products on oneDNN with a masked float32 softmax between (shapes
  fixed per block, so primitives are reused across prompts); the text encoder's parity is unchanged (final hidden
  rel 1.4e-2). The 8 GiB reserve keeps the 2048 px forward inside the card with the expanded matrices.

Still to do: model cards for the other quants, the front door / `image serve` smoke test, README and INTEL.md rows.
