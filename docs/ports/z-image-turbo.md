## Z-Image-Turbo (Tongyi-MAI): the port

Text to image, 6B, Apache-2.0 (`Tongyi-MAI/Z-Image-Turbo` @ f332072a). A diffusers `ZImagePipeline`: the 6B
single-stream DiT (`ZImageTransformer2DModel`), Qwen3-4B as the text encoder (`Qwen3Model`), Flux.1's VAE
(`AutoencoderKL`, 16 channels), `FlowMatchEulerDiscreteScheduler` at shift 3. Distilled (Decoupled-DMD + DMDR) to 8
steps without guidance. Engine `image/zimage` (arch `z-image`), kernels `kernels/image/zimage` and its silo, card
`z-image-turbo`, reference `reference/z-image-turbo/ref.py`.

### Files (and why these)

| Part | File | Size |
|---|---|---|
| DiT | `unsloth/Z-Image-Turbo-GGUF` @ b40f712c `z-image-turbo-BF16.gguf` (ComfyUI's layout, `general.architecture = lumina2`) | 12.3 GB |
| text encoder | `Comfy-Org/z_image_turbo` @ 6fc90a3b `split_files/text_encoders/qwen_3_4b.safetensors` (one bf16 file) | 8.0 GB |
| VAE | `Tongyi-MAI/Z-Image-Turbo` `vae/diffusion_pytorch_model.safetensors` (diffusers' names, bf16) | 168 MB |
| tokenizer | `Tongyi-MAI/Z-Image-Turbo` `tokenizer/tokenizer.json` | 11 MB |

Checked by `ref.py --files` against the official repository's shards:

- The official transformer is stored float32 (24.6 GB), but **every one of its values is a bfloat16** (the low 16 bits
  are zero throughout): BF16 loses nothing. unsloth's BF16 GGUF holds all 521 official tensors **bit for bit** (its
  small tensors as F32, the same values). It is the default card.
- unsloth's Q8_0 (the same file as `jayn7/Z-Image-Turbo-GGUF`'s: one sha256) keeps 281 tensors as they are (the
  refiners, the embedders, the final layer, every norm) and quantizes the 30 blocks' matrices, their adaLN included
  (worst relative difference 7.2e-3, `layers.29.adaLN_modulation.0`). See "Q8_0" below for whether it earns a card.
- ComfyUI's `qwen_3_4b.safetensors`: all 398 tensors equal the official text encoder's three shards.
- Comfy-Org also has the DiT as bf16 safetensors (the same values in ComfyUI's names, not needed beside the GGUF),
  int8 ConvRot (every block matrix int8, activations quantized on the fly) and NVFP4.

### Architecture (from transformer_z_image.py and pipeline_z_image.py, diffusers 84532a05)

- **Text**: the prompt in Qwen3's chat template with thinking on: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`,
  tokenized to at most 512 tokens (`max_sequence_length`; longer is cut, the template's end with it), run through
  Qwen3-4B with the padding masked; the condition is `hidden_states[-2]` - the residual stream after layer 35 of 36,
  before any norm - of the prompt's tokens alone. The engine loads and runs 35 layers.
- **DiT** (dim 3840, 30 heads of 128, FFN 10240, patch 2): the caption `cap_embedder` (RMSNorm 2560, Linear 2560 ->
  3840) then two unmodulated context-refiner blocks; the image's 2 x 2 patches `x_embedder` (64 -> 3840) then two
  modulated noise-refiner blocks; then 30 blocks over [image | caption]; the final layer (LayerNorm without affine,
  times 1 + adaLN(SiLU(t)), Linear 3840 -> 64) on the image's rows. Block: `x += tanh(g1) rmsnorm(attn(rmsnorm(x)(1 +
  s1)))`, `x += tanh(g2) rmsnorm(w2(silu(w1 h) w3 h))`, q / k RMS-normed per head, RoPE on three axes (32 / 48 / 48
  features, theta 256), the adaLN a Linear 256 -> 4 x 3840 of the timestep embedding (no SiLU before it).
- **Padding** (what the port first got wrong): both sequences are padded to a multiple of 32 with learned pad tokens
  (`x_pad_token`, `cap_pad_token`) that every attention sees (a batch of one has no mask). The image's pad rows sit at
  position (0, 0, 0); the caption's pad rows keep counting (1 + j, 0, 0): `_pad_with_ids` gives the caption a grid of its
  padded length and then appends origin rows for the padding, which the rotary table's cut to the rows drops. The
  image's frame is the padded caption length + 1. 1024 x 1024 has no image padding (4,096 tokens); 1280 x 720 has 16
  pad rows (3,600 tokens).
- **Sampler**: `get_default_z_image_sigmas(n)` = linspace(1, 1/n, n), the static shift 3 (`3 s / (1 + 2 s)`), a final 0;
  the model's timestep is (1000 - 1000 s) / 1000; Euler on minus the model's output. 8 steps = 8 forwards (the README's
  `num_inference_steps=9` "results in 8 DiT forwards" was the same schedule on the pipeline of its day, whose levels
  ended at 0). Guidance (`guidance_scale > 0`: pos + g (pos - neg)) is off for the Turbo; the engine refuses it.
- **VAE**: `z / 0.3611 + 0.1159`, Flux.1's decoder (no post-quant conv), clamp to [-1, 1].

### What was reused

- **The DiT**: Ming-Image's (`image/mingimage/src/dit.rs`, the same transformer): copied and changed rather than shared,
  because an engine may not depend on another engine (CONTRIBUTING rule 4), moving it into the foundation would make
  Ming's kernels shared (frozen, rule 6), and the two differ where it matters - Z-Image's caption is Qwen3-4B's states
  (no query tokens, no direct condition), its pad tokens are attended, ComfyUI's layout fuses q | k | v. The kernels it
  needs (modulated RMS / layer norms, gated adds, scaled half conversions, the scaled SwiGLU) are copies of Ming's in
  `kernels/image/zimage/zi.cpp`; the VAE's group norm, softmax, upsample and picture bytes copies of MOSS-Image's.
- **The VAE**: MOSS-Image's FLUX.2 decoder (`image/mossimage/src/vae.rs`, diffusers' blocks) without its batch norm,
  unpatchify and post-quant conv. One change: Flux.1's mid-block q and k reach 2.5e3, so their scores pass half's range
  (NaN everywhere on the first try) - q, k and the scores are float32 products.
- **The text encoder**: Qwen3 dense layers in the shape of `crates/qwen3vl` (same layer, Qwen3-VL's), on bf16
  matrices made half at load instead of int8 ConvRot (the residual stream and attention float32, the gated product made
  half at 1/256).

### Parity (`nextsycl image check <dump> --model z-image-turbo`, B70)

The reference runs diffusers' classes on the CPU in float32 on the same files (each GGUF matrix expanded to float32 for
its own product). Two dumps: the README's prompt at 1024 x 1024, seed 42, and the demo's first example ("一位男士和他的
贵宾犬...") at 1280 x 720 (padded image), seed 7; 8 steps each. Each stage starts from the reference's own input.

| Stage | 1024 x 1024 | 1280 x 720 |
|---|---|---|
| tokens | identical (101) | identical (28) |
| text encoder after layer 1 / 18 / 35 (the caption) | 3.7e-4 / 6.1e-5 / 2.5e-4 | 4.0e-4 / 5.8e-5 / 1.8e-4 |
| sigmas, t_emb | exact, 1e-7 | exact, 1e-7 |
| caption embedded (+ pad), context refiner | 2.4e-4, 1.5e-4 | 2.5e-4, 1.6e-4 |
| image embedded (+ pad), noise refiner | exact, 1.2e-4 | exact, 1.2e-4 |
| layer 0, layer 29 (image rows) | 1.4e-4, 4e-3 .. 7e-3 | 1.4e-4, 6e-3 |
| velocity, each of the 8 steps (teacher-forced) | 0.8e-3 .. 2.6e-3, cos >= 0.999997 | 0.8e-3 .. 4.3e-3, cos >= 0.999991 |
| latents after 8 steps (free-running) | 1.6e-2 .. 1.9e-2, cos 0.99985 | 4.3e-2 .. 4.9e-2, cos 0.9988 |
| VAE | 3.2e-4 | 3.4e-4 |
| picture, all ours vs ref.png | rel 2.0e-2, cos 0.9998 | rel 4.4e-2, cos 0.9990 |

The free-running latents carry the half rounding of every product (~1e-3 a step) amplified by the trajectory; the
pictures are the reference's (side by side: the same woman, lamp and pagoda; the same man and poodle). The layer-29
number moves between 4e-3 and 7e-3 with summation order alone (the silo's fused norms against the unfused ones: the
1024 dump 7.2e-3 / 3.8e-3, the 720 dump 6.3e-3 / 6.3e-3). Activations: `stats.json` of the reference gives the largest
magnitudes at step 0 - q / k / v 5.6e3, the gated product 6.5e4 (half's limit is 65,504), the out and w2 products
1.4e5 and 2.5e5 - so q | k | v are made half at 1/16, the gated product at 1/256, and the out and w2 products stay
float32 into their norms (whose epsilon takes the scale).

### Speed

Measured with `--stages bench` (a forward at 1024 x 1024, warm; the per-part list waits after each part, so its sum
runs a little over the forward) and `--stages sizes` (the whole request - text encoder, 8 steps, VAE - at each size,
made twice, the second timed), the card's memory sampled every second from the front door.

| B70, 1024 x 1024 (4,096 image + 128 caption tokens) | DiT forward | a picture |
|---|---|---|
| first exact port (Ming's passes, q / k / v float32 then scaled into half) | 0.596 s | 5.5 s |
| + q / k / v scale folded into the half matrices (the product writes half; no float32 round trip) | 0.560 s | |
| + the silo (`kernels/image/zimage/silo`: the two residual steps of a block one fused pass each, the next block's input norm in the second; q / k norm + rope in one launch; a vectorized SwiGLU) | 0.511 s | 4.5 s |

Where a forward goes now (B70, BF16): the four products 360 ms at 124-152 TFLOPS (q | k | v 3.0 ms, out 1.0, w1 | w3
4.4, w2 2.2 a block - near the card's practical half peak), attention 81 ms (sycl-tla's flash kernel through
`nsd_attention_qk`, ~112 TFLOPS), the norms and gates 50 ms (98 before the silo), SwiGLU 16 (29), q / k norm + rope 8
(15). The text encoder takes 0.03 s for 101 tokens, the VAE 0.39 s at 1024 x 1024.

A picture, the whole request, BF16 (`z-image-turbo`):

| size | B70 | B65 |
|---|---|---|
| 1024 x 1024 | 4.52 s | 6.80 s |
| 1152 x 896, 896 x 1152 | 4.47 s | 6.72 s |
| 1152 x 864, 864 x 1152 | 4.32 s | 6.39 s |
| 1248 x 832, 832 x 1248 | 4.55 s | 6.89 s |
| 1280 x 720, 720 x 1280 | 4.04 s | 6.02 s |
| 1344 x 576, 576 x 1344 | 3.39 s | 5.05 s |
| 1280 x 1280 | 7.55 s | 11.50 s |
| 1536 x 1024 | 7.21 s | 10.93 s |
| 1536 x 1536 | 11.66 s | 17.81 s |
| 2048 x 1152 | 11.67 s | 17.82 s |
| 2048 x 2048 | 24.67 s | 38.05 s |

The card's memory while it runs (the whole model on one card: the DiT 11.2 GiB half, Qwen3-4B's 35 layers 6.6 GiB, the
VAE, the work): 19.4 GiB at 1024 x 1024, 20.6 at 1280 x 1280, 21.3 at 1536 x 1536, 23.7 GiB at 2048 x 2048 (B70; the
B65 21.8). Through the server (`image serve --here`, two 1024 x 1024 requests in one process, then a 1280 x 720): 4.48
s, 4.45 s, 3.99 s, 0.30 Wh a 1024 picture; the refusals answer at once (a negative prompt, cfg 3, 1000 x 1000). Load:
the DiT 5.7 s warm (29 s from a cold page cache), the text encoder 2.8 s (15 s cold).

### Q8_0 and int8

- **Q8_0 (unsloth, = jayn7)**: not offered. Its matrices stay as stored for the shared quantized product (slices made
  half per call), so it is slower - a forward 0.712 s against 0.511 - and less exact: velocities 2.3e-2 at step 0 (BF16:
  2.6e-3), the picture cos 0.994 to the reference. It would save 4.6 GiB of card memory (6.6 GiB of DiT), which a
  32 GiB card does not need for this model.
