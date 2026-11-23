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
- **The I2V image encoder**: the pipeline loads `FLUX.1-Redux-dev`'s `image_encoder` (SigLIP so400m-patch14-384),
  **gated** - 403 for our token. Google's open `siglip-so400m-patch14-384` is the same architecture; whether Redux's
  weights equal it is **not checked** (the reference ran on google's file too). Accepting Redux's licence on the
  token's account would settle it (one tensor compared).
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
- Activations: the residual stream float32, products' inputs half, attention the shared `attention_qk` (oneDNN's
  fused SDPA, half) over `[image | text]`; the VAE half (as upstream's float16 autocast).
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

Recorded at 12844f8 on hardac1 (gate.sh's per-card step run by hand in the build copy - two runs a card under the card's
lock, the same commands; gate.sh itself would have relinked every kind's library and built without a memory cap):

| Card | exact (both runs) | pp (DiT tokens/s) | tg (decoder kpixels/s) |
|---|---|---|---|
| arc-pro-b70-256u-32g | 1225588c2ce6e65a | 9689.0 | 1375.6 |
| arc-pro-b65-160u-32g | 9d80ad04dc6c5baa | 6467.1 | 727.3 |

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
