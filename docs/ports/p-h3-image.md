# P-H3-Image (image/ph3image)

PrunaAI's P-H3-Image (huggingface.co/PrunaAI/P-H3-Image @ 7f98acef; ComfyUI files: PrunaAI/P-H3-Image-Comfy @ 44a0fb73):
text to image and edits with up to 8 pictures in 8 steps without guidance, on MiniMax H3's 50-block denoiser. The
reference is diffusers' modular pipeline at commit 84532a058702c74ae5d1dd6ceb05683fc5d5e768
(`src/diffusers/modular_pipelines/p_h3_image`, `models/transformers/transformer_minimax_h3.py`,
`models/autoencoders/autoencoder_kl_p_h3_image.py`, `models/condition_embedders/condition_embedder_p_h3_image.py`,
`schedulers/scheduling_minimax_h3.py`), dumped by `reference/p-h3-image/ref.py`.

## What runs where

- **The denoiser is H3's.** The video engine's code moved into the foundation (`crates/h3`, `nextsycl-h3`); `video/h3` is
  now H3 behind the video contract and `image/ph3image` P-H3-Image behind the image contract. The image is H3's packed
  sequence for one latent frame: `[text | the edit's pictures | 2 x 2 audio rows | 2 x 2 patches]` - t2va's layout for
  text to image, ref2va's (image references only) for an edit; positions, tags and per-row timesteps are H3's `layout`
  code unchanged (checked: `pos` exact, both layouts). The audio rows are denoised beside the image and dropped.
- **Schedule.** The checkpoint's grid `1, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0` is
  used as it is: the scheduler config's `shift: 12` is not applied to given sigmas (`shift_provided_sigmas: false`), the
  same as ComfyUI's graph (ModelSamplingMiniMaxH3 at 1.0). Image and audio share it. Euler as MiniMaxH3Scheduler.step:
  x0 = x + (1 - t) v, x = r x + (1 - r) x0. Edit conditioning rows: 0.999 latent + 0.001 noise, held at max(t, 0.999).
- **Noise** is torch's (H3's Mt19937 draws, checked exact): each edit picture's augmentation, then the image, then the
  audio, from the request's seed; the edit's VAE posterior is sampled under seed 42 and rounded to half (diffusers'
  `encode_vae_condition`).
- **Text encoder (new):** Qwen3-VL-4B from Pruna's ComfyUI int8 ConvRot file, through `crates/qwen3vl` (generalised: the
  `model.language_model.` prefix, the 4B vision tower - 24 blocks of 1,024, deepstack after 5, 11, 17 -, hidden states
  after chosen layers and the final norm). The text connector (hidden states 13, 19, 25, 31, 36 concatenated,
  standardised, 12,800 -> 5,120, the sink row) is folded into one float32 product. The prompt is tokenized as written (no
  chat template); an edit is `"<Picture i>: "` + vision tokens per picture, then `"Edit <Picture 1>. "` + the prompt. The
  vision tower loads only when an edit comes (1.6 GiB).
- **VAE (new decoder):** H3's conv encoder (`crates/h3` venc, a new `moments_of_picture` for the posterior) and P-H3's own
  transformer decoder (image/ph3image/src/vae.rs: H3's video decoder adapted to one frame - its time coordinate
  2 (0.5 / 7) - 1, a 768-wide output, the residual stream in float32), 256-pixel tiles cross-faded as the reference's.
- **Silo** (`kernels/image/ph3image/silo`, `dist/silo/libnextsycl-ph3image.so`): the denoiser's attention. Its own copy
  of MiniMax-H3-Image's Sage kernel (`silo/tla` -> `dist/silo/libnextsycl-ph3image-tla.so`; see "The silo's own Sage
  kernel" below), ARK's int8 prefill in libnextsycl-flash.so with `NS_PH3_SAGE_TLA=0` or without that library; the
  passes around it rewritten to stream. The
  engine's H3 context is opened on the core GPU's queue (`Device::on_queue`, `nsd_create`) so the text encoder, the
  denoiser and the silo share one queue and one context.

## LoRA or full model

Pruna offers a LoRA (2 GB) that "runs P-H3-Image on the H3 FL2VA transformer". Compared by HTTP range reads (no download):
`p-h3-image_bf16.safetensors` against `Comfy-Org/MiniMax-H3 minimax_h3_fl2va_pruned_bf16.safetensors` + the LoRA:

- 221 of the 272 tensors the LoRA does not touch are bit-identical. The other 51 are the AdaLN projections and the
  folded timestep table: P-H3's is a rank-16 float32 table `[4097, 16]`, FL2VA's rank-8 half `[1025, 8]`; the tables they
  give differ by ~2e-4 relative (biases 1.6e-4 - 2.2e-4).
- The 208 LoRA'd matrices: P-H3 is 1.2e-3 - 3.3e-3 from the base; base + LoRA is 1.2e-3 - 1.9e-3 from P-H3 (bf16 rounding
  alone is ~1.6e-3); after rounding base + LoRA to bf16, 77 - 87 % of the values equal P-H3's. So the LoRA is close to the
  fine-tune but not it: the full model was not saved as base + LoRA.
- What matters is the output. Each denoiser against the reference's velocities (the 576 x 384 run below, every step fed
  the reference's latents):

| Denoiser | file | velocity rel. error, steps 0 / 3 / 7 | final latents | 8 MP steps, B65 |
|---|---|---|---|---|
| Pruna's int8 ConvRot (default) | 21.2 GB | 2.9e-2 / 2.3e-2 / 1.5e-2 | 3.7e-2 | 29.5 s |
| AtomicChat Q8_0 | 21.7 GB | 1.7e-2 / 1.5e-2 / 1.0e-2 | 3.3e-2 | 47.1 s |
| AtomicChat AD-Q6_K (Q6_K/Q5_K/Q8_0) | 16.8 GB | 3.8e-2 / 4.2e-2 / 1.8e-2 | 5.9e-2 | - |
| H3 FL2VA int8 + Pruna's LoRA | (H3's) + 2 GB | 3.0e-2 / 2.6e-2 / 1.5e-2 | 4.5e-2 | 32.7 s |
| H3 FL2VA Q6_K + LoRA | (H3's) + 2 GB | 3.3e-2 / 3.5e-2 / 2.2e-2 | 4.8e-2 | - |
| H3 FL2VA Q4_K_M + LoRA | (H3's) + 2 GB | 1.3e-1 / 1.2e-1 / 6.5e-2 | 2.2e-1 | - |

  The LoRA route equals the full model within int8's own error (3.0e-2 against 2.9e-2): the fine-tune's 1.5e-3 weight
  residual is under the quantization. The default is Pruna's int8 file (fastest; Q8_0 is closest but 1.6x slower here,
  its k-quant matrices expanded to bf16 for every product); the LoRA card serves people who already have H3's files.
  AtomicChat's files are made with stable-diffusion.cpp's converter from Pruna's bf16 file (their README: per-tensor
  layouts from a sensitivity scan, an importance matrix from bf16 renders; the AD-Q5_K / Q4_K / Q3_K files use Q3_K,
  which the dequant does not read). realrebelai's GGUFs keep 220 tensors (the refiner included) in F16 and were not taken.
- ComfyUI's folded AdaLN table against the full model's time MLP (ref.py `fold`): 1.1e-3 - 1.3e-3 relative on the blocks'
  six tables, 5e-7 on the final layer's.

## Parity (reference: diffusers on the CPU, float32, the full bf16 model; 576 x 384, seed 3, the model's own prompt)

| Stage | Text to image | Edit (1 picture, canvas 576 x 384) |
|---|---|---|
| tokens | 33 of 33 equal | - |
| vision tower (4B) | - | merged tokens 3.2e-4, deepstack 1.7e-4 - 4.2e-4 |
| hidden states 13 .. 31 (int8 TE) | 5.9e-2 - 6.6e-2 rel., cos 0.99975 - 0.99996 (one outlier channel ~300) | 2.5e-2 - 3.4e-2, cos >= 0.9998 |
| hidden state 36 (final norm) | 7.4e-2, cos 0.9973 | 4.4e-2, cos 0.9990 |
| connector output | 1.6e-3 | 5.9e-3 |
| VAE encoder (conditioning latent) | - | 2.4e-3 |
| refined text (int8 file's refiner) | 6.3e-3 | 9.5e-3 |
| noise, positions, sigmas | exact | exact |
| velocities (int8) | 1.5e-2 - 2.9e-2, cos >= 0.9996 | 2.3e-2 - 5.5e-2, cos >= 0.9985 |
| final latents, 8 steps from the reference's noise | 3.7e-2, cos 0.9993 | 3.4e-2, cos 0.9994 |
| VAE decoder, bf16 file | 1.4e-3; 94.6 % of bytes equal, 99.99 % within 1 | 1.8e-3; 93.1 %, 99.93 % |
| VAE decoder, int8 file (default) | 5.2e-3; 79.8 % equal, 99.2 % within 1 | 6.2e-3; 77.2 %, 97.8 % |

With the silo's SageAttention from 1,024 tokens (the default) the 576 x 384 run (253 tokens) stays on oneDNN; forced
onto Sage it reads 2.8e-2 / 3.2e-2 / 1.7e-2 and 5.5e-2. The silo against the shared Sage on the same 8,318-token rows:
8.6e-6 (the order of k's mean). Like H3, the whole 8-step run is chaotic: two runs of the shared kernels at 8 MP give
pictures 38.9 dB apart, the shared and the silo attention 26.4 dB, Sage and bf16 flash attention 27.9 dB (the same
scene and composition, details differ). Judge changes by the velocity / block parity, not the end picture.

## Speed (B65, one picture after the first, `image gen --local`; int8 denoiser, the text encoder resident)

| | before (shared kernels, bf16 VAE) | after (silo, int8 VAE) |
|---|---|---|
| 1024 x 1024 | 4.8 s (8 steps 4.0, decode 0.57) | 4.6 s (3.7, 0.46) |
| 2048 x 2048 | 18.3 s (15.4, 2.7) | 15.6 s (13.3, 2.2) |
| 2912 x 2912 (8 MP) | 38.4 s (33.1, 5.1) | 34.0 s (29.5, 4.1) |
| text encoder | loaded for every request: 1.6 - 9.8 s | loaded once (2 s), then 0.03 - 0.3 s |
| peak card memory, 8 MP | 27.4 GiB own (the text encoder loading beside the denoiser) | 26.2 GiB |

The refiner adds 0.2 - 0.7 s a picture (loaded per picture). An 8 MP step: int8 products 2.0 s (184 TOPS: near the
B65's int8 rate), attention 1.15 s (was 1.7: the shared Sage passes took 14 ms of 34 ms a block), the rest 0.5 s. The
decoder at 8 MP (225 tiles of 261 tokens): 5.3 s in bf16 at ~85 TFLOPS (its matrices; the batch of tiles makes no
difference), 4.1 s in int8. An edit at 4 MP (one picture): text 14 - 20 s the first time (the vision tower and the VAE
encoder loaded), the picture's latents 7.4 s, 8 steps 33 s (~9,300 tokens).

B70: see the next section (int8, 8 steps: 1 MP 2.50 s, 8 MP 18.94 - 19.00 s; ~1.53x the B65 a step).

## The silo's own Sage kernel (2026-11-17)

The silo now has its own copy of the Sage kernel MiniMax-H3-Image's silo got on 2026-11-16 (docs/ports/minimax-h3-image.md,
"Third pass": what it is, what was tried): `kernels/image/ph3image/silo/tla/` (`sage.cpp`, `ark/sage_mainloop.hpp`) ->
`dist/silo/libnextsycl-ph3image-tla.so`, opened by the silo's `sage.cpp` beside itself. ARK's int8 prefill kernel around a
copy of its mainloop with the scores' scale folded into the exponent's fma, the result written as bfloat16 straight into
the activations: no half result buffer (0.11 GiB at 8 MP), no conversion pass. The code is h3image's under this silo's
names (`ns_image_ph3_tla_*`, namespace `nsph3`); silos stay apart, neither engine calls the other's. `NS_PH3_SAGE_TLA=0`
(--opt-sage-tla) or a missing library: ARK's kernel and the conversion pass, as before. `NS_PH3_PROFILE` now also prints
the silo's preparation / kernel / result times every 50 calls (a step), naming the kernel.

**Exact, the kernel alone** (`reference/h3-image/sagebench.cpp` built against this silo's copy - its header has the two
flags; 56 heads of 128, random q / k / v with a per-column k offset, quantized as the silo does; "int8 math" = float32
attention of the dequantized int8 q / k and half v; ARK's half result rounded to bfloat16 as the conversion pass does).
The same digits on the B65 and the B70:

| tokens | kernel | vs int8 math: max / mean abs | vs float32 q k v: max / mean abs, relative |
|---|---|---|---|
| 1,061 (1 MP) | ARK | 8.772e-3 / 2.897e-4 | 1.118e-1 / 4.864e-3, 2.437e-2 |
| 1,061 | silo | 7.812e-3 / 2.853e-4 | 1.118e-1 / 4.863e-3, 2.437e-2 |
| 8,318 (8 MP) | ARK | 8.773e-3 / 1.583e-4 | 1.014e-1 / 2.984e-3, 2.700e-2 |
| 8,318 | silo | 7.802e-3 / 1.559e-4 | 1.014e-1 / 2.983e-3, 2.700e-2 |

The silo rounds once (float -> bfloat16) where ARK's path rounds twice (float -> half -> bfloat16): no looser anywhere.

**Parity, 576 x 384 seed 3, Sage forced** (`NS_PH3_SAGE_MIN_S=64`: the default runs oneDNN at 253 tokens and the table
above is unchanged; `image check --stages dit,steps`, each step from the reference's latents), int8 file, B65, the same
build, ARK by `NS_PH3_SAGE_TLA=0`; ARK's text-to-image velocities repeat the forced
row above (2.8e-2 / 3.2e-2 / 1.7e-2; its free-run final latents read 6.3e-2 here against 5.5e-2 then):

| | text to image: silo kernel | text to image: ARK | edit: silo kernel | edit: ARK |
|---|---|---|---|---|
| velocity, steps 0 / 3 / 7 | 3.35e-2 / 3.17e-2 / 1.82e-2 | 2.85e-2 / 3.23e-2 / 1.72e-2 | 4.10e-2 / 4.79e-2 / 3.35e-2 | 4.07e-2 / 4.34e-2 / 3.45e-2 |
| velocity, steps 0 .. 7 | 1.62e-2 - 3.99e-2 (mean 2.79e-2) | 1.67e-2 - 3.62e-2 (mean 2.79e-2) | 2.66e-2 - 6.11e-2 (mean 3.94e-2) | 2.60e-2 - 5.15e-2 (mean 3.72e-2) |
| velocity cosine, worst | 0.99921 | 0.99935 | 0.99818 | 0.99870 |
| final latents (8 steps from the reference's noise) | 4.57e-2, cos 0.99896 | 6.29e-2, cos 0.99803 | 3.30e-2, cos 0.99946 | 4.11e-2, cos 0.99916 |

Refined text, noise, positions: identical in all four. Step by step the two kernels trade places (the rounding paths
differ by one rounding); the edit's mean velocity error is 6 % higher with the silo kernel and its final latents 20 %
lower, the text-to-image means equal: inside the per-step spread, as MiniMax-H3-Image's table found. At the attention
itself the silo kernel is no looser (above).

**Speed** (kernel alone: the bench above, median of 20, order ARK / silo / silo / ARK, TFLOPS as 4 S^2 D H; in the engine:
`NS_PH3_PROFILE`, a step's 50 calls at 8 MP = 8,335 tokens; pictures: `image gen --local`, int8, seed 7, the second picture
of n = 2 (the text encoder resident), 8 steps, runs alternated silo / ARK / silo / ARK, both cards the same night):

| | B65: ARK | B65: silo kernel | B70: ARK | B70: silo kernel |
|---|---|---|---|---|
| kernel, 1,061 tokens | 0.452 / 0.454 ms (71.3 TFLOPS) | 0.430 / 0.430 ms (75.0) | 0.300 / 0.299 ms (107.7) | 0.292 / 0.293 ms (110.4) |
| kernel, 8,318 tokens | 20.85 / 20.85 ms (95.1) | 20.15 / 20.17 ms (98.5) | 13.08 / 13.12 ms (151.6) | 12.87 / 12.88 ms (154.1) |
| attention in a step, 8 MP (preparation + kernel + result) | 89 + 1,028 + 25 = 1,142 ms | 89 + 988 + 0 = 1,077 ms | 85 + 634 + 23 = 742 ms | 85 + 608 + 0 = 693 ms |
| 1024 x 1024: 8 steps / decode | 3.74 / 3.74 s; 0.47 | 3.68 / 3.68 s; 0.46 | 2.52 / 2.52 s; 0.31 | 2.50 / 2.50 s; 0.31 |
| 1024 x 1024: whole picture (refiner + steps + decode) | 4.42 / 4.40 s | 4.30 / 4.32 s | 3.00 / 3.01 s | 2.99 / 2.99 s |
| 2912 x 2912 (8 MP): 8 steps / decode | 29.51 / 29.59 s; 4.08 / 4.07 | 28.92 / 29.08 s; 4.08 / 4.08 | 19.36 / 19.36 s; 2.77 / 2.77 | 18.94 / 19.00 s; 2.77 / 2.83 |
| 2912 x 2912: whole picture | 33.79 / 33.84 s | 33.18 / 33.34 s | 22.32 / 22.34 s | 21.89 / 22.04 s |

So: the kernel 3 - 5 % faster on the B65 and about 2 % on the B70, the attention op 6 - 7 % at 8 MP (the conversion pass
gone too), the 8 steps 1.6 - 2.0 % faster (B65 1 / 8 MP) and 0.8 - 2.0 % (B70), a whole 8 MP picture 1.6 % on either card. Exact and not slower: **on by default** (`NS_PH3_SAGE_TLA` 1 when the library is built).

## Open

- **The prompt upsampler** (Qwen3.5-9B, `Qwen3_5ForConditionalGeneration`, on by default upstream; sees an edit's
  pictures) is not run: prompts are rendered as written. nextsycl's qwen35 llm engine reads GGUF files of the dense
  Qwen3.5/3.8 family, but Pruna ships the upsampler only as safetensors (bf16, int8 ConvRot, w4a8) of a fine-tuned 9B
  with a vision tower; serving it needs a conversion to GGUF and the vision path, and the front door's image methods
  would need to call an llm first. Not started.
- B70 numbers beyond the Sage table above (2048 x 2048, edits, the first-pass table).
- The edit's VAE encoder (H3's half-precision convolutions, loaded per request) and the per-picture refiner load are
  the slow parts of an edit; keep both resident when the card has room.
- w4a8 files (Pruna's, realrebelai's w6a8) need a 4-bit codebook kernel; AD-Q3_K-based GGUFs need Q3_K in the dequant.
- The decoder's attention over 261-token tiles (0.95 s at 8 MP) still goes through the shared fused kernel's copies.
