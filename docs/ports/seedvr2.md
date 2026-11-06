## SeedVR2 (ByteDance-Seed): the port

One-step diffusion restoration and upscaling of video, 3B and 7B, released 2025-06. The engine is `image/seedvr2`
(kernels `kernels/image/seedvr2`), served by the image server as `image.upscale` and `video.upscale`
(`{"file": ..., "model": "seedvr2", "variant": "seedvr2-3b" | "seedvr2-7b", "scale": 2, "crf": 16}`); the card is
`model-cards/image/seedvr2.yml` - an image-kind card because the image server is what serves `video.upscale` (the
video kind is H3's daemon).

### Licence

Code (github.com/ByteDance-Seed/SeedVR: LICENSE is Apache-2.0) and weights (huggingface.co/ByteDance-Seed/SeedVR2-3B
and -7B: `license: apache-2.0`, "SeedVR and SeedVR2 are licensed under the Apache 2.0") are Apache-2.0, without a
non-commercial clause. The files served are numz's fp16 safetensors of the official checkpoints
(huggingface.co/numz/SeedVR2_comfyUI, Apache-2.0) and the official `pos_emb.pt` (the fixed positive prompt's
embedding: its tensor storage, bfloat16 [58, 5120], taken out of the zip - no unpickling). The wavelet colour fix is
numz's ComfyUI node's (Apache-2.0).

### Files

| Part | File | Size |
|---|---|---|
| VAE | `numz/SeedVR2_comfyUI` `ema_vae_fp16.safetensors` | 501 MB |
| text | `ByteDance-Seed/SeedVR2-3B` `pos_emb.pt`, member `pos_emb/data/0` | 594 KB |
| DiT 3B | `numz/SeedVR2_comfyUI` `seedvr2_ema_3b_fp16.safetensors` | 6.8 GB |
| DiT 7B | `numz/SeedVR2_comfyUI` `seedvr2_ema_7b_fp16.safetensors` | 16.5 GB |

### The pipeline (projects/inference_seedvr2_3b.py)

- frames / 255, resized (torchvision bicubic, antialiased: PyTorch's a = -0.5 filter) to the size asked (each side x
  scale), clamped, centre-cropped to multiples of 16, to [-1, 1];
- the VAE (`models/video_vae_v3`, s8 c16 t4): causal 3-D, the time padded by repeating the first frame, GroupNorm per
  frame, one attention head per frame in the middle; the encoder's posterior **mean** is used (the reference samples:
  its std is ~2.5 % of the latent's spread - the mean is deterministic and steadier), times 0.9152;
- the DiT once at t = 1000 (uniform trailing timesteps, one step: the shift leaves 1000 as it is), its input
  [noise 16 | condition 16 | 1], the positive prompt only (cfg 1); Euler's v_lerp endpoint: x0 = noise - v;
- the VAE decode of x0 / 0.9152, then the wavelet colour fix against the resized input (out = content +
  blur5(style - content), clamped).

The 3B is `models/dit_v2` (MMDiT blocks with separate video / text weights for blocks 0-9, shared after; SwiGLU;
mmrope3d - the text rotated too, the video's time positions after the text's; an output RMS norm), the 7B `models/dit`
(separate weights everywhere, GELU, the rotary over each window's own positions in [-1, 1], no output norm). Two
details of the reference that the trained weights depend on, carried over: the 3B's output norm takes the *first
block's attention* shift and scale from the shared cache (the key `emb_repeat_0_vid` collides) plus its own; the 3B's
last block modulates only the video (its ada is `vid_only`: the text goes into the last attention normed alone). The
attention is per window (`window.rs`: 720p-equivalent 3 x 3 windows, time in 4, shifted by half on odd blocks), every
window seeing the whole text; the text's outputs are the mean over the windows.

### Clips: chunks with overlap

The reference upscales a clip whole, which a film of 5,000 frames cannot be. As numz's ComfyUI node does, a clip goes
through in chunks of `NS_SVR_CHUNK` frames (33: 9 latent frames, 3 time windows), each chunk the whole reference
pipeline on its own (a short last chunk padded with its last frame, as the reference pads), neighbouring chunks sharing
`NS_SVR_OVERLAP` frames (4) crossfaded with a raised cosine across the overlap (the node's Hann window fades over the
middle third only: for 4 frames a cut, and the detail jumped at every seam). The DiT's windows tie a chunk's frames together in
time; the overlap carries it over the seams. Chunk n's noise is seed + n.

### Memory

The DiT's matrices live in pinned host memory and go up to the card for each call (6.8 GB / 16.5 GB, ~0.3 / 0.7 s),
let go after: the VAE's causal convolutions keep their last two input frames from slice to slice (exactly the
reference's slicing), ~17 GB of them at 1728x960 in the decoder, and the full-resolution part of each slice runs a
frame or two at a time (`NS_SVR_PIECE`; the same frames whatever the piece: the kept frames carry over). At 864x480 ->
1728x960 the card peaks at 23.9 GiB (3B) and 24.4 GiB (7B) of the B70's 31.9.

### Checked (exact before fast)

`reference/seedvr2/ref.py` runs the official code on the CPU in float32 (numz's fp16 weights upcast; apex and flash-attn
stood in for by torch's RMSNorm and SDPA), 9 frames of an H3 clip at 384x224, and dumps each stage;
`nextsycl image check DUMP --model seedvr2 [--variant seedvr2-7b] --frames PNG_DIR` feeds each stage the reference's own
input:

| stage | 3B: cosine / relative error | 7B |
|---|---|---|
| resize (vs torchvision) | 0.999999 / 1.3e-3 | (same) |
| VAE encode (latent) | 0.999963 / 8.6e-3 | (same) |
| DiT block 0 / 15 / 31 (7B: 0 / 17 / 35) | 0.999997 / 2.3e-3; 0.999968 / 8.0e-3; 0.999792 / 2.0e-2 | 0.999999 / 1.7e-3; 0.999964 / 8.4e-3; 0.999714 / 2.4e-2 |
| DiT output v | 0.999930 / 1.2e-2 | 0.999911 / 1.3e-2 |
| x0 = noise - v (vs the reference's sampler) | 0.999880 / 1.6e-2 | 0.999850 / 1.7e-2 |
| VAE decode | 0.999994 / 3.4e-3 | (same) |
| colour fix | 0.999997 / 2.6e-3 | (same) |

The engine is bfloat16 as the official inference (autocast): the reference's own bfloat16-autocast run is 1.9e-2 from
its float32 run at v, the engine 1.2e-2. The VAE in pieces of 1 and 2 frames gives the same numbers to every digit.

### Speed (B70, 864x480 -> 1728x960, chunks of 33)

| | encode | DiT | decode | a chunk | a frame |
|---|---|---|---|---|---|
| 3B | 10.8 s | 5.3 s | 29.5 s | 45.7 s | 1.38 s (~1.5 s with the overlap) |
| 7B | 10.8 s | 9.6 s | 29.5 s | 50 s | 1.5 s (~1.7 s with the overlap) |

The VAE is most of it (oneDNN's convolutions on the shared path, no silo yet): the open work is its silo.

### Optimisation log

Each step measured at 864x480 -> 1728x960 on the 243-frame H3 clip (or one 33-frame chunk where marked), s/frame of
the whole request (`video.upscale` through the image server, ffmpeg included), B70 unless said.

| # | change | 3B s/frame | 7B s/frame | notes |
|---|---|---|---|---|
| 0 | the first working port (one card, everything serial) | 1.54 (243 frames: 373.6 s) | 1.51 (33-frame chunk) | VAE 87 % of a chunk |
