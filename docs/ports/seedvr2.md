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

numz's files against the official checkpoints (`convcheck`: the official fp32 tensor cast to fp16, bit for bit): the VAE
identical (250 tensors); the 3B's 603 of 635 tensors differ by at most 0.00195, one fp16 step at their size (another
rounding of the same weights). The parity below runs the reference code on numz's files.

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

### Speed (864x480 -> 1728x960, chunks of 33, the 243-frame clip through `video.upscale`)

| | one card (B70) | two cards (B70 + B65) | peak VRAM a card |
|---|---|---|---|
| 3B, step 2 | 0.98 s/frame (1.02 frames/s) | 0.57 s/frame (1.75 frames/s) | 25.1 GiB B70, 23.0 GiB B65 |
| 7B, step 1 | 1.28 s/frame (0.78 frames/s; 50 s of it the switch from the 3B) | 0.875 s/frame (1.14 frames/s) | 26.3 GiB B70, 23.1 GiB B65 |
| first port (step 0) | 3B 1.51, 7B 1.51 (a 33-frame chunk) | 3B 0.93 | 23.9 GiB |

The 7B's DiT is 16.5 GB: it fits beside the VAE because its weights stay in pinned host memory and go up for each call
(0.7 s at 28 GB/s), let go before the decode (the VAE's kept frames at full resolution take ~17 GB). Two cards each keep
their own pinned copy (2 x 16.5 GB of host memory for the 7B). A film of 5,500 frames: ~1 h 40 (3B, one card), ~52 min
(3B, two cards).

### Optimisation log

Each step measured at 864x480 -> 1728x960 on the 243-frame H3 clip (or one 33-frame chunk where marked), s/frame of
the whole request (`video.upscale` through the image server, ffmpeg included), B70 unless said.

| # | change | 3B s/frame | 7B s/frame | notes |
|---|---|---|---|---|
| 0 | the first working port (one card; chunks in a worker thread, the host's frames beside it) | 1.51 (367.5 s); two cards 0.93 (225.5 s) | 1.51 (one 33-frame chunk, 50 s) | VAE 87 % of a chunk; B70 compute engine busy 77 % (fdinfo) |
| 1 | silo: the causal convolutions as 2-D taps over the frames where they are (the copies of the kept frames gone: 9 % of a chunk); group norm, shuffle and residual add vectorised (16-byte loads), group norm's statistics reduced in parallel; the DiT's window tables kept across chunks | 1.03 (250.3 s); two cards 0.60 (146.5 s) | | convolutions 51-57 % -> 61-67 % of the XMX peak; group norm 143 -> 269 GB/s |
| 2 | a let-go volume freed at the card's next wait instead of a wait at every free (the queue was drained thousands of times a chunk); the DiT's attention fed half (the shared attention's in-place path) | 0.98 (239.0 s); two cards 0.57 (139.0 s) | | B70 compute busy 73 -> 80 %; DiT attention 1.54 -> 1.09 s a chunk. The half q, k, v overflowed on the 7B (see below): replaced in step 3 |
| 3 | the DiT's window attention in bfloat16 on the silo (oneDNN's fused graph; the half path made chunks NaN); group norm's statistics over 4096 parts | parity: v 1.16e-2 (3B); speed queued behind the films | | not yet run on a whole clip: whether the NaN chunks are gone is open |

### Non-finite chunks (found and fixed)

Now and then a whole chunk came out black: the DiT's output all NaN (counted: 0 in the condition latent, every value
of the DiT's output), in different chunks from run to run, the 7B more than the 3B. Ruled out by measurement: the frees
(made safe; still NaN), the weights' upload (read back after it: 0 bytes differ; made in pieces), a race within a call
(a second call of the same chunk went NaN again, from the same block). The shared attention runs in half: its scores
are float32, but v, the probabilities and the output are half, and its bfloat16 input is copied to half first - a
value past 65,504 is inf, and through the text tokens every window shares, NaN everywhere after that block. Feeding it
half straight from the qkv projection made it worse (the 7B NaN from block 3). The reference runs this attention in
bfloat16 (flash attention); the silo's `ns_image_seedvr2_silo_attention` is oneDNN's fused attention graph in bfloat16
(float32 scores), read in place from the packed rows. A chunk that still comes back non-finite is made again once, the
first block that goes wrong logged and the weights on the card compared with the host's.

### Where the time goes (B70, one 33-frame chunk of the 3B at 864x480 -> 1728x960, step 2)

`NS_SVR_PROFILE=1`: each step timed alone (a wait before and after: the chunk takes 34.1 s so instead of ~31), its
FLOPs counted from the shapes - a convolution 2 To Ho Wo Co Ci kt kh kw, a GEMM 2 M K N, attention 4 S^2 D a head
and window (q k^T and the weighted sum; S the window's tokens plus the text's) - and its bytes as the least it must
move (inputs, outputs and weights once). Peaks: XMX bfloat16 = Xe cores (compute units / 8) x 2048 FLOPs a clock x
the clock: the B70's 32 cores at 2.8 GHz 183.5 TFLOPS (Intel's B580: 233 int8 TOPS from 20 cores at 2.85 GHz,
bfloat16 half that); a GEMM of 8192^3 reaches 170.6 (93 %: `image check --stages bench`). Memory 608 GB/s (256-bit
GDDR6 at 19 Gbps). PCIe: pinned host -> card 28.0 GB/s measured (`nextsycl gpus`: 25.8 GB/s with its own buffers).

| step | share | s | TFLOPS (% of 183.5) | GB/s (% of 608) | why it sits where it does |
|---|---|---|---|---|---|
| VAE 3x3x3 convolutions (silo taps), all | 59 % | 16.6 | 112-122 (61-66 %) at 864x480 and up; 87 (47 %) at 216x120 | 74-294 | compute-bound: oneDNN's 2-D convolutions; the small first and last layers (3 or 16 channels) 1-6 TFLOPS but 2 % of the time |
| VAE group norm (+SiLU) | 10.7 % | 3.0 | - | 265 (44 %) | three passes (statistics, groups, apply); the next to fuse |
| DiT MLP | 5.6 % | 1.5 | 128 (70 %) | | GEMMs |
| DiT window attention | 3.9 % | 1.1 | 20 (11 %) | | windows of ~870 tokens, 20 heads: short sequences for the fused kernel (the bfloat16 silo graph since step 3) |
| VAE 1x1x1 convolutions | 3.3 % | 0.9 | 26 (14 %) | 140 | the upsamplers' channel expansion: the shared linear with a bias; open |
| VAE mid attention | 3.1 % | 0.9 | 30 (16 %) | | one head of 512 over 25,920 pixels: GEMM + softmax + GEMM a block of rows |
| VAE residual add | 2.8 % | 0.8 | - | 456 (75 %) | bandwidth |
| DiT qkv / out linears | 3.4 % | 0.9 | 99-125 (54-68 %) | | GEMMs |
| DiT qk norm, rope, gathers, gates, norms | 3.9 % | 1.1 | - | 236-424 (39-70 %) | bandwidth |
| DiT weights up (PCIe) | 0.8 % | 0.24 | - | 28.3 GB/s (101 % of the measured link) | |
| colour fix + RGB bytes, resize, frames up / down | 0.3 % | 0.08 | - | 48-695 | |
| ffmpeg decode / x264 encode (host) | - | - | - | - | on the calling thread while the card works: the run takes 0.8 cores (one card), 1.6 (two) |

Per chunk (wall, unprofiled): encode 9.6 s, DiT 5.2 s (its weights up included), decode 19.2 s.

Utilisation (`gpumon.py`: the xe driver's per-client engine counters in /proc/PID/fdinfo, every 0.5 s): one card, the
B70's compute engine busy 73 % (step 1) -> 80 % (step 2) of the run; the rest is the host between kernels (a wait for
each slice's frames, the DiT's input and output, noise) - the next thing to overlap. Two cards: B70 64-69 %, B65 87-97 %
(the B70 finishes its chunks first and waits for the last one); the copy engines 1-3 %. CPU 0.8 / 1.6 cores. The host's
work (ffmpeg, crossfade, the frames' order) runs on the calling thread beside the cards' workers.
