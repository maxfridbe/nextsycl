## HunyuanVideo-1.5 (Tencent): the port

Text to video and image to video at 480p, 8.3B, Tencent Hunyuan Community License (`tencent/HunyuanVideo-1.5` @
9b49404b, code github.com/Tencent-Hunyuan/HunyuanVideo-1.5 @ 60783e7 - the commit MovieRoll forked). Engine
`video/hunyuanvideo15` (arch `hunyuanvideo-1.5`), kernels `kernels/video/hunyuanvideo15` and its silo, cards
`hunyuanvideo-1.5-480p-t2v-distilled-q8` (cfg-distilled, one pass a step, 50 steps), `hunyuanvideo-1.5-480p-t2v-q8`
(the base model, CFG 6) and `hunyuanvideo-1.5-480p-i2v-step-distilled-q8` (MeanFlow, 8 steps), reference
`reference/hunyuanvideo15/ref.py` (+ `fetch.sh`). Phase 0 + 1 of [movieroll.md](movieroll.md).

### Files (and why these)

| Part | File | Size |
|---|---|---|
| DiT | `transformer/480p_t2v_distilled` / `480p_t2v` / `480p_i2v_step_distilled` `diffusion_pytorch_model.safetensors` (float32, official) | 33.3 GB each |
| VAE | `vae/diffusion_pytorch_model.safetensors` (float32: encoder and decoder) | 5.0 GB |
| text encoder | HunyuanImage 2.1's `qwen2.5-vl-7b-test-q4_0.gguf` (calcuis) + Qwen2.5-VL's `tokenizer.json` | 5.0 GB |
| glyph encoder | HunyuanImage 2.1's `byt5_small_glyphxl_fp32-f16.gguf` (Glyph-SDXL-v2's byT5) | 0.44 GB |
| image encoder (I2V) | `google/siglip-so400m-patch14-384` `model.safetensors` (its vision tower read) | 3.5 GB |

- **The text encoders are HunyuanImage 2.1's, unchanged.** Both pipelines take Qwen2.5-VL-7B's
  `hidden_state_skip_layer=2` (hidden_states[-3]: layer 25's output, no final norm) and Glyph-SDXL-v2's `byt5_model.pt`
  (the same checkpoint both READMEs name). What differs is the template (the video one: a JSON system prompt through
  Qwen2.5-VL's chat template, the generation prompt added, the crop computed - 108 tokens), the quote rule (`"..."` and
  `“...”` in order, deduplicated; HunyuanImage also reads ‘...’, by kind) and the glyph length (256, not 128). The
  code moved into the foundation (`crates/hunyuan-text`) with those as parameters; HunyuanImage 2.1 runs through it
  with its own kernels (the image library's), this engine with its copies (the video library's: each kind's library
  holds its own engines' kernels).
- **The DiT is quantized as it loads**: the double blocks' matrices (qkv, proj, MLP and the modulations: 8.1B of the
  8.3B values) to Q8_0 (ggml's `quantize_row_q8_0_ref`), everything else float32. No converted file to publish or
  keep: the cards name the official files. Load: 26 - 57 s on hardac1 (the 33 GB file read and quantized; ~26 s warm).
- **The I2V image encoder**: the pipeline loads `FLUX.1-Redux-dev`'s `image_encoder` (SigLIP so400m-patch14-384,
  `SiglipVisionModel` in fp16; gated). **Checked 2026-11-23: it is google's `siglip-so400m-patch14-384` vision tower
  rounded to bfloat16** - all 448 of its tensors (Redux rev 11deb8ff, `image_encoder/model.safetensors`, 856,506,120
  bytes, sha256 d769e3a3...) equal google's float32 `vision_model.*` tensors rounded to bfloat16, bit for bit; no
  tensor missing either side (`~/src/hv15/redcmp.py` on hardac1). So the card keeps google's open file (Apache-2.0):
  the same weights, at float32 rather than Redux's bfloat16 (upstream then runs them in fp16) - the difference from
  upstream is the rounding only. Redux's file stays on the drive (`hunyuanvideo15/image_encoder/`), unused.
- MovieRoll (`yunyangge/MovieRoll`): still 403 (checked 2026-11-23); its code (github, public) was read for phase 2.

### Architecture (hunyuanvideo_1_5_transformer.py, hunyuanvideo_15_vae.py, hunyuan_video_pipeline.py)

- **DiT**: hidden 2048, 16 heads of 128, 54 double-stream blocks, no single-stream ones, FFN 8192 (GELU tanh), q / k
  RMS-normed per head, biases everywhere; 3-D RoPE over the image tokens only (16 / 56 / 56 features for frame / row
  / column, theta 256, interleaved pairs); `img_in` a 1x1x1 conv of 65 channels: the 32 noisy latents, 32 condition
  latents (I2V: frame 0's latent, zeros after) and a mask (I2V: 1 on frame 0); the token refiner (2 blocks) over
  Qwen's tokens; text rows `[SigLIP via vision_in (I2V) + e2 | byt5_in(glyph) + e1 | refined Qwen + e0]` (e the
  `cond_type_embedding`), only the valid ones - upstream's flash attention runs the unpadded sequence, the same thing;
  T2V's vision tokens are zeros with a zero mask, i.e. absent. vec = time_in(t) (+ time_r_in(t_next): MeanFlow, the
  step-distilled card). Final: LayerNorm * (1 + scale) + shift, linear to 32.
- **Sampler**: `linspace(1, 0, n + 1)`, shift `s' = k s / (1 + (k - 1) s)` (k 5 for 480p T2V, 7 for the step-distilled
  I2V), t = 1000 s', Euler `x += (s'_{i+1} - s'_i) v`; CFG `v_u + g (v_c - v_u)` over an empty negative on the base
  card. The noise: `torch.randn((1, 32, F, h, w), generator, dtype=bfloat16)` - reproduced value for value
  (crates/diffusion's mt19937 + Box-Muller, rounded to bfloat16), so a seed gives upstream's clip.
- **Sizes**: 480p buckets `generate_crop_size_list(640, 16)`; the request's shape picks the nearest (16:9: 848 x
  480; I2V: the picture's shape); frames 4k + 1 (5 s = 121: 31 latent frames, 30 x 53 x 31 = 49,290 tokens).
- **VAE**: 3-D causal, 32 channels, 16x space / 4x time, RMS norms over the channels (x sqrt(C) gamma), causal
  3x3x3 convs with replicate padding, one-head attention in the middle causal by frame, DC-AE-style up/down sampling
  with channel-repeat / channel-mean shortcuts and a first frame of its own. Scaling 1.03682. The pipeline decodes
  under `memory_efficient_context`: tiles of 16 x 16 latents every 14, blended over 32 pixels, kept to 224 - done the
  same here (480p: 3 x 4 tiles, each the whole clip in time).

### The port

- Kernels (`kernels/video/hunyuanvideo15/hv15.cpp`, `mm.cpp`): HunyuanImage's text-encoder kernels under this
  engine's names, the modulations' Q8_0 expansion, and the VAE's: causal replicate padding (then the shared oneDNN
  `conv3d_ex`), channel RMS norm (+ SiLU), the upsample / downsample rearrangements with their shortcuts, the
  frame-causal softmax of the middle attention (scores and the weighted sum on oneDNN with float32 accumulation, query
  rows in blocks of whole frames under 512 MB). The silo (`silo/mm.cpp`): HunyuanImage 2.1's Q8_0 product (oneDNN
  int8 weights with grouped half scales, GELU and the gated residual as post-ops), copied (rule 6). Without the silo
  each matrix is expanded to half before its product (slower, the same results within rounding).
- Activations: the residual stream float32, products' inputs half, attention over `[image | text]` the silo's (ARK's
  half flash on sycl-tla, the shared kernels' own; SageAttention as an option - "Faster" below); the VAE half (as
  upstream's float16 autocast).
- The attention call takes queries and keys / values apart: MovieRoll's window over a cached history is the same
  call with the cache's rows before the window's (below).

### Parity (`nextsycl video job check --dir DIR --stages ... --here`; reference float32 on the CPU, the same Q8_0 values)

Fox prompt, seed 7, 128 x 224 x 9 frames (336 tokens) and 320 x 320 x 9 (1,200 tokens: the VAE's 2 x 2 tiles), 2
steps of the cfg-distilled DiT; I2V: the step-distilled DiT, a 1024 x 1024 picture to 128 x 224.

| Stage | rel RMS | max abs | cosine |
|---|---|---|---|
| tokens (143 / 114) and crop (108) | identical | | |
| text encoder layer 0 | 1.9e-4 | 1.6e-3 | 1.000000 |
| text encoder hidden_states[-3] (prompt / empty negative) | 2.6e-4 / 5.6e-4 | 5.1e-2 / 6.6e-2 | 1.000000 |
| noise from the seed | 0 | 0 | 1 |
| vec (timestep embedding) | 2.9e-6 | 5.7e-8 | 1.000000 |
| token refiner output | 7.4e-7 | 5.5e-5 | 1.000000 |
| double block 0 (image / text) | 6.2e-5 / 9.4e-5 | 1.3e-4 / 9.4e-3 | 1.000000 |
| after the 54 blocks (image) | 9.1e-4 | 1.1e-2 | 1.000000 |
| velocity, step 0 | 3.9e-4 | 3.8e-3 | 1.000000 |
| latents after step 1 / 2 | 1.4e-4 / 6.0e-4 | 6.3e-4 / 4.2e-3 | 1.000000 |
| 320 x 320: latents after 2 steps | 1.2e-3 | 2.3e-2 | 0.999999 |
| VAE decode (one tile / 2 x 2 tiles) | 1.4e-3 / 2.7e-3 | 1.6e-2 / 5.1e-2 | 0.999999 / 0.999996 |
| VAE decode as a stream (latent frames 1 + 1 + 1, carried state) against the whole decode's reference | 1.35e-3 | 1.6e-2 | 0.999999 |
| I2V: SigLIP tokens (from the reference's 384 x 384 pixels) | 6.2e-5 | 1.1e-2 | 1.000000 |
| I2V: the first frame's latent (VAE encoder) | 7.9e-4 | 4.2e-3 | 1.000000 |
| I2V: after the 54 blocks / velocity / latents after 2 steps / decode | 9.0e-4 / 2.9e-4 / 5.3e-4 / 3.5e-4 | 6.2e-3 / 4.1e-3 / 5.3e-3 / 3.9e-3 | 1.000000 |
| *the quantization itself*: the reference on Q8_0 against the reference on the float32 file - after the 54 blocks / velocity / latents after 1 and 2 steps | 1.5e-2 / 6.2e-3 / 2.3e-3 / 1.4e-2 | 0.13 / 6.1e-2 / 1.0e-2 / 0.19 | 0.99989 / 0.99998 / 0.999997 / 0.99990 |

Every stage is the reference's to within the engine's rounding (half products, oneDNN's sums); the gap Q8_0 itself
opens against the float32 weights is ten times larger - the usual for int8 weights (1e-2 - 3e-2) - and the 50-step
fox clip is clean (a coherent fox walking through the snow, the camera tracking it).

### Speed and memory (one card; the clip 848 x 480, 121 frames, 50 steps, the cfg-distilled card)

| Card | Load | 50 steps | a step (49,290 tokens) | VAE decode (12 tiles) | Clip | Peak on the card |
|---|---|---|---|---|---|---|
| Arc Pro B70 | 26 - 66 s | 642.5 s | 12.85 s | 34.5 s | 677.6 s (11.3 min) | 18.7 GiB |
| Arc Pro B65 | 40 s | 1015.8 s | 20.32 s | 61.8 s | 1078.3 s (18.0 min) | 18.5 GiB |

The other cards on the B70 (one run each): the base card with CFG 6, 3 steps: 25.5 s a step (two passes), peak 19.4
GiB - 50 steps would be ~21 min; the step-distilled I2V card from a 1024 x 1024 picture (640 x 640, 121 frames, 49,600
tokens): 8 steps 105.8 s (13.2 s a step), decode 33.9 s, the clip 141 s, peak 21.8 GiB (SigLIP's tower and the VAE
encoder loaded beside the rest: 17.2 GiB before the first step). The I2V clip starts on the picture (the fox sitting
in the snow), turns its head and trots off.

Measured with `video job generate --seconds 5 --steps 50 --seed 7` (the fox prompt), one run each; the load is the
33 GB float32 file read and quantized (66 s cold, 26 s with the file in the page cache). The B65 is 1.58x the B70 a
step, as for H3 (1.5x). The base card (CFG 6) runs two passes a step: twice the step time.

The card's memory ("vouch"): the cfg-distilled card states `vram_gib: 19.5` (18.7 GiB measured + margin); the engine's
own check before it allocates asks for the files' share plus 5 GiB of work (~20.6 GiB) and refuses with the numbers
when the card has less free. The worker's room check reads the card's `vram_gib` (its float32 file would otherwise ask
for 41 GiB: more than a card).

The memory figure is the card's whole use while the clip ran (total less the driver's free): the DiT's 8.2 GiB, the
text encoders' 4.2 GiB and the VAE's 1.6 GiB stay loaded (15.1 GiB before the first step), the rest is the 49K-token
pass's activations and the decoder's tiles.

### Faster (2026-11-23): where a step goes, the attention, step caching, fewer steps

**Where a step goes** (`NS_HV15_PROFILE=1`: the queue drained at each mark, device time; the 5 s clip, 49,290
picture tokens + 256 text, the cfg-distilled card, 3 steps):

| Part of a step | B70 | B65 |
|---|---|---|
| attention (54 calls) | 10.16 s (79.3%) | 16.05 s (80.0%) |
| the picture's Q8 products: fc1 / fc2 / qkv / proj | 0.68 / 0.69 / 0.50 / 0.23 s (16.4%) | 1.08 / 1.07 / 0.78 / 0.30 s (16.2%) |
| norm + modulation / RMS + RoPE | 0.34 / 0.16 s (3.9%) | 0.54 / 0.18 s (3.6%) |
| the text stream (its four products), embedding + refiner, final layer | 0.05 s (0.4%) | 0.05 s (0.3%) |
| the step | 12.81 s | 20.05 s |
| VAE decode (once a clip) | 35.2 s | 62.0 s |

**What the cards give** (`video job check --stages bench`, oneDNN's plain matmul on scratch buffers, mean of 10):
B70 half 177 - 182 TFLOPS, int8 306 - 354 TOPS; B65 half 95 - 97 TFLOPS, int8 164 - 189 TOPS (the same five shapes,
8192³ to the DiT's 49,290 x 6,144 x 2,048). The attention before this work was the shared kernels' flash (ARK's half
prefill on sycl-tla, libnextsycl-flash.so; oneDNN's SDPA only without it): **108 TFLOPS on the B70** (186 ms a call,
counted 4 S² D H, 49,546 tokens, 16 heads), 60% of the half matmul's rate; the B65 68 TFLOPS (295 ms).

**The attention in the silo** (`kernels/video/hunyuanvideo15/silo/attn.cpp` + `silo/tla/attn.cpp` ->
`dist/silo/libnextsycl-hunyuanvideo15-tla.so`, opened by the silo beside itself): q, k, v read in place from the qkv
rows and the result written as half into the activations, two kernels -

- **half** (the default): ARK's half prefill on this silo's choice of tiles. Five were tried at 1,024 / 4,096 /
  8,192 / 49,546 tokens: ARK's own (256 query rows, 32 keys a block, two prefetch stages) and a third prefetch stage
  are the fastest and give the same bits as the shared kernel (rel 0); 64 keys a block (the sage kernel's tile) or
  128 query rows run at 55 - 70% of it (register pressure: a sub-group's 16 x 128 float32 accumulator is half the
  register file). Keeping q in registers across the key blocks - the one change that would cut its reloads - does not
  fit beside that accumulator. So the default path is the same arithmetic as before: the clip is **bit-identical**
  (the gate's hash did not move) and as fast.
- **sage** (`--opt-attention sage`, a request's `"attention": "sage"`): SageAttention v1 - q and k int8 (a scale per
  head per 64 rows, k less its mean over the sequence), the scores' int32 products in int8 DPAS, P V in half; the
  kernel is the MiniMax-H3-Image silo's (ARK's int8 prefill with the scale folded into the exponent) with a half
  result. The quantization pass is that silo's on half input. **B70 126 ms (160 TFLOPS), 1.48x the half; B65 197 ms
  (102 TFLOPS), 1.50x.** On random q / k (x 2.5, sharp scores) the result is rel 3.1e-2 (cosine 0.9995) from the half.

**Step caching** (`--opt-cache deepcache`, a request's `"cache"`): upstream's DeepCache (`generate.py
--enable_cache --cache_type deepcache`, defaults `--cache_start_step 11 --cache_end_step 45
--cache_step_interval 4 --no_cache_block_id 53`): steps 0 - 10 and 45 - 49 and every 4th from 11 run in full, the
other 25 of 50 reuse the rows double block 53 got on the last full step and run only block 53 and the final layer
with their own step's modulation (`Dit::forward_cached`; one float32 copy of a pass's rows, 0.4 GiB, the card's peak
18.7 -> 18.9 GiB). `deepcache` for other step counts scales start and end (30 steps: 7 to 26, 15 cached);
`deepcache:START,END,INTERVAL[,BLOCKS]` sets them (BLOCKS: the last blocks run on a cached step). Upstream's
TeaCache / TaylorCache need angelslim's code and its calibrations; MagCache (ComfyUI's, 1.7x at 20 steps) its
magnitude ratios - neither is here.

**Fewer steps**: the cfg-distilled card was trained for 50; 30 and 20 are request `steps` (the clip below).

**Quality** (the fox prompt, seed 7, the 5 s clip; each against the default 50-step clip of the same card: the last
latents' relative RMS and cosine, every frame's PSNR and SSIM (luma, 11 x 11 Gaussian), and two whole-clip numbers -
detail: the mean |Laplacian| of the luma, motion: the mean |difference| of neighbouring frames):

| Clip | Card | latents rel / cos | PSNR mean / min (dB) | SSIM mean / min | detail | motion |
|---|---|---|---|---|---|---|
| default, 50 steps (the reference) | B70 | - | - | - | 6.31 | 22.04 |
| the same on the other card (the noise floor) | B65 vs B70 | 0.0096 / 0.99995 | 53.2 / 46.8 | 0.9993 / 0.9983 | 6.31 | 22.04 |
| sage | B70 | 0.133 / 0.9911 | 26.5 / 21.6 | 0.904 / 0.849 | 6.34 | 21.99 |
| sage | B65 | 0.085 / 0.9964 | 30.3 / 26.0 | 0.947 / 0.916 | | |
| deepcache | B65 | 0.087 / 0.9962 | 30.3 / 27.6 | 0.954 / 0.930 | | |
| sage + deepcache | B70 | 0.095 / 0.9955 | 29.5 / 27.3 | 0.948 / 0.933 | 6.93 | 20.85 |
| sage + deepcache | B65 | 0.102 / 0.9947 | 29.1 / 26.4 | 0.937 / 0.907 | 6.90 | 20.87 |
| 30 steps | B65 | 0.459 / 0.891 | 16.1 / 13.7 | 0.614 / 0.540 | | |
| 30 steps + deepcache | B70 | 0.454 / 0.893 | 16.2 / 13.7 | 0.609 / 0.539 | 8.67 | 19.47 |
| 30 steps + deepcache + sage | B70 | 0.454 / 0.893 | 16.2 / 13.6 | 0.611 / 0.538 | 8.70 | 19.47 |
| 30 steps + deepcache + sage | B65 | 0.455 / 0.892 | 16.2 / 13.5 | 0.611 / 0.538 | 8.69 | 19.47 |
| 30 steps + deepcache + sage, against 30 steps + deepcache | B70 | 0.088 / 0.9962 | 30.0 / 24.5 | 0.952 / 0.932 | 8.70 | 19.47 |
| 20 steps | B65 | 0.623 / 0.801 | 13.9 / 11.8 | 0.490 / 0.441 | | |

Read: sage and deepcache each move the clip well past the noise floor (PSNR ~30 against 53) but keep it - the same
fox, pose, gait, framing and background a frame for frame; looked at side by side (frames 0, 30, 60, 90, 120 and a
full-size crop of frame 60) nothing was seen lost - no blur, no blocks - though one prompt and a few frames are not a
proof. The two together are no further off than either alone. Neither is "within noise", so neither is the default. 30 or 20 steps make a **different** clip from the
same seed (the trajectory changes: PSNR 14 - 16, the fox's legs and the trees elsewhere), not a broken one - by eye
still sharp and steady at full size, the fur a little smoother; it has more fine detail by the Laplacian (8.7 against
6.3: fewer steps leave more texture / grain) and a little less motion (19.5 against 22.0). Whether 30 is "as good" is
for eyes on more prompts than one; the card's guide calls 50 the model's own and offers the others as faster takes.
At 30 steps sage moves the clip as much as it does at 50 (PSNR 30 against 30 steps + deepcache); against the 50-step
clip, 30 steps alone (B65), with deepcache and with both (B70, B65) all land at PSNR ~16: the step change dominates.

**Speed, before and after** (one run each, the 5 s clip; load not counted):

| Card | Setting | Denoise (a step) | Decode | The 5 s clip | x | Peak |
|---|---|---|---|---|---|---|
| B70 | before / the default (half, 50 steps) | 640.7 s (12.81 s) | 35.2 s | 676.7 s | 1.00 | 18.7 GiB |
| B70 | sage | 481.7 s (9.63 s) | 34.7 s | 517.3 s | 1.31 | 18.9 GiB |
| B70 | sage + deepcache | 245.6 s (4.91 s) | 35.4 s | 281.8 s | 2.40 | 18.9 GiB |
| B70 | 30 steps + deepcache | 195.9 s (6.53 s) | 35.3 s | 232.0 s | 2.92 | 18.7 GiB |
| B70 | 30 steps + deepcache + sage | 147.6 s (4.92 s) | 34.8 s | 183.2 s | 3.69 | 18.9 GiB |
| B65 | before / the default | 1010.8 s (20.22 s) | 61.8 s | 1073.4 s | 1.00 | 18.5 GiB |
| B65 | sage | 748.0 s (14.96 s) | 62.0 s | 810.9 s | 1.32 | 18.9 GiB |
| B65 | deepcache | 515.8 s (10.32 s) | 61.9 s | 578.6 s | 1.86 | 18.5 GiB |
| B65 | sage + deepcache | 381.3 s (7.63 s) | 62.1 s | 444.2 s | 2.42 | 18.7 GiB |
| B65 | 30 steps + deepcache + sage | 229.0 s (7.63 s) | 61.9 s | 291.7 s | 3.68 | 18.7 GiB |
| B65 | 30 steps | 608.7 s (20.29 s) | 62.2 s | 671.7 s | 1.60 | 18.7 GiB |
| B65 | 20 steps | 405.3 s (20.27 s) | 62.3 s | 468.5 s | 2.29 | 18.7 GiB |

The ≥3x asked for is there only with the step change (30 steps, deepcache, sage: 3.69x on the B70 - 183 s for the 5 s
clip -, 3.68x on the B65); at 50 steps the most is sage + deepcache, 2.4x, a clip that is the same one by eye but not
by the noise floor. The default stays the lossless path (the gate's hashes unchanged on both cards).

**Not done, and why**:
- An own half flash kernel beyond ARK's tiles: the q-in-registers form does not fit (above); ARK's half already folds
  the softmax scale into the exponent's multiply-add.
- Int8 P V (ARK's UseInt8PV): P is quantized as P x 127 with no scale of its own, so every probability under 1/254 -
  most of a 49K-key row - becomes 0.
- Int8 products (the silo's opt-in int8 x int8, from HunyuanImage 2.1): the products are 16% of a step; 1.7x on them
  is ~7% of the clip, an approximation for little.
- Both cards on one clip: CFG passes split over the B70 + B65 would make a CFG step max(12.8, 20.3) = 20.3 s against
  25.5 (1.26x) and hold both cards (production serves on both); sequence parallel needs the other half's K and V every
  block, ~11 GB each way a step over PCIe. Not built.
- The VAE in larger tiles: 12 tiles of 256 px (26% overlap); a tile of the whole height (480 x 256) is ~12% less work
  for ~2x the decoder's memory (~22 GiB peak) - 4 s of a 35 s decode; left as it is.

### Served

`video.generate` with `engine: "HV15"` (the cfg-distilled card), `"HV15-CFG"` (the base model) or `"HV15-I2V"` (a
`first_frame` picture), once the model is in `NS_VIDEO_ENGINES`: the studio maps the registry ids to those names
(`quant_of`), applies their own rules (`hv15_rules`: 4k + 1 frames from `seconds` - 5 s is 121 -, default steps 50 (I2V
8), at most 100; sides multiples of 16, default 848 x 480) and the engine's `generate` job reads `prompt`, `width`,
`height`, `seconds` / `frames`, `steps`, `seed`, `first_frame`. `video.status / wait / cancel / timeline` work on
it as on any clip (the daemon's queue). The front door's `steps` limit went from 40 to 100 for it (H3's own limit,
40, stays in the studio). No sound: the mp4 is silent.

### Gate

`tooling/gate-baselines.json` `video/hunyuanvideo15`: the cfg-distilled card, 848 x 480, 1 s (25 frames, 11,130
tokens), 4 steps, seed 7. `exact` hashes the last latents (float32) and the frames (8-bit); its two `[prompt ...]`
speeds are the DiT's latent tokens a second over the steps and the decoder's thousands of pixels a second (tok/s by
the gate's grammar). The run resolves `--model` from the build copy's registry: the gate's copy needs the model
pulled (`nextsycl models pull hunyuanvideo-1.5-480p-t2v-distilled-q8 --dir <the models' drive>`).

Re-recorded at 2d646fc (2026-11-23), the attention moved into the silo: the same hashes on both cards (the default path is
the same arithmetic), pp B70 9,684.6 - 9,687.9 / B65 6,401.7 - 6,403.4, tg 1,355 - 1,368 / 728 - 729. First recorded
at 12844f8 on hardac1 (gate.sh's per-card step run by hand in the build copy - two runs a card under the card's
lock, the same commands; gate.sh itself would have relinked every kind's library and built without a memory cap):

| Card | exact (both runs) | pp (DiT tokens/s) | tg (decoder kpixels/s) |
|---|---|---|---|
| arc-pro-b70-256u-32g | 1225588c2ce6e65a | 9684.6 | 1355.2 |
| arc-pro-b65-160u-32g | 9d80ad04dc6c5baa | 6401.7 | 727.6 |

### Lessons

- **oneDNN is not deterministic by default on this GPU**: two runs of the same clip differed (latents up to 7e-3 after
  one step) - the shared `nsd_linear`'s small float32 products (the timestep embedders: M = 1) and the half products
  pick split reductions whose order varies. `attr.set_deterministic(true)` on every primitive this engine makes (its
  `linear32` / `linear_f32`, the silo's products) and its float32 products moved off the shared `nsd_linear` made the
  clip bit-identical run to run (the attention, the VAE's convolutions and its 1x1 products, still shared, were
  already) - what lets the merge gate hash it. The speed did not move measurably. MiniMax H3's "not bit-reproducible
  run to run" (docs/video.md, cosine 0.996) is likely the same.

- Read the template, not the README: the video encoder's system prompt is a Python string with backslash line
  continuations, so its text holds runs of eight spaces - and the crop is computed from the tokenized head (108), not
  fixed as in HunyuanImage (34).
- `torch.randn(..., dtype=bfloat16)` on the CPU draws float32 normals (mt19937, Box-Muller in blocks of 16) and rounds
  them: the shared generator reproduced it bit for bit.
- `models pull --from` finds a copy by file name and size: the three 480p transformers share both
  (`diffusion_pytorch_model.safetensors`, 33,306,632,192 bytes for two of them), so it hashed the wrong one, refused
  it and started downloading 31 GB. `--dir <the drive the files are on>` finds each at its own path ("have it").
- Relinking every kind's library for one engine's kernels takes most of an hour (AOT at link): a build copy that only
  needs the video library links that one (and its silo).

### What MovieRoll (phase 2) needs on top of this engine

Not started: MovieRoll's weights are gated (403, checked again 2026-11-23), and its rolling window only means
anything with them - HunyuanVideo-1.5's own weights were never trained on mixed noise levels or a causal history, so
machinery built against them could be run but not checked. What its public code (github Tencent-Hunyuan/MovieRoll,
`movieroll_transformer.py` `forward_inference`, `streaming_attention.py` `chunk_inference_attention`,
`movieroll_dmd_pipeline.py`) asks of this engine, in order:

1. **Per-chunk timesteps - built**: a window of 5 chunks (4 latent frames each) at rising noise levels, `vec` (hence
   every block's modulations) a row per chunk. `Dit::forward_spans` takes the image rows in spans, each at its own row
   of `Steps` (the norms, gated products and final layer called per span); one span is `forward` (every check and
   clip here ran through it). Several spans at different levels need MovieRoll's weights to mean anything.
2. **Attention over a cached history**: the window's image queries over `[cached history K, V | window image | text]`,
   the text queries over `[window image | text]` only (the history unseen by the text). `nsd.attention_qk` takes
   queries and keys / values apart already; the cache is a per-block buffer of post-RoPE K and V rows the window's own
   are written after (one call for the image queries, one for the text's). RoPE positions run on with the frame index
   (the table's frame axis offset by the history's length).
3. **The cache (HieraKV)**: blocks 0 - 5 keep every finished chunk; blocks 6 - 53 a sink (the first chunk) and the
   recent ones; at 18 cached chunks a recompute re-encodes the sink, representative middle chunks and the 3 most
   recent at t = 0 without text (`forward_kv_recompute`, the K / V only, no text stream). Memory (movieroll.md): ~13 GB
   bf16 at 480 x 864 - a second card or int8 K / V when it does not fit beside the 8.2 GiB DiT.
4. **Events**: one prompt a window; a new prompt re-encodes the text (cheap: 0.1 s here) and triggers a recompute.
5. **Streaming decode - built and checked**: the decoder is causal in time, so `Vae::stream_begin / stream_chunk /
   stream_end` decode a video a few latent frames at a time - each causal convolution carries its last two padded
   input frames, the middle attention its keys and values, the temporal upsamples treat only the first chunk's first
   frame as the video's - and give the whole decode's frames (the check above: 1.35e-3 against the reference, the same
   as one decode). MovieRoll's `streaming_vae.py` ("causal RMSNorm 3D VAE with T-cache streaming", the same parameters
   and keys as its `causal_vae.py`) is this design; whether its decoder weights are HunyuanVideo-1.5's is for when the
   weights can be read.
6. **Phase 3** (`video.roll`, `video.rollAppend`, `video.rollEnd`, fragmented MP4 + HLS while it runs) sits on 1 - 5.
