## FlashVSR v1.1 (OpenImagingLab): the port

One-step diffusion video super-resolution, streamed: Wan 2.1 T2V-1.3B distilled to one step, 8 frames a step, each step
attending to the 24 before it through a KV cache, a causal LQ projection in place of a VAE encoder and a tiny
conditional decoder in place of the VAE decoder. The engine is `image/flashvsr` (kernels `kernels/image/flashvsr`, its
silo `kernels/image/flashvsr/silo`), served by the image server as `image.upscale` and `video.upscale`
(`{"file": ..., "model": "flashvsr", "variant": "flashvsr-v1.1", "scale": 2, "crf": 16}`, with the optional `seed` (0),
`sparse_ratio` (1.5), `local_range` (11) and `color_fix` (1), at the top level or in `options`); the card is
`model-cards/image/flashvsr.yml`. A port of nextcuda's engine (image/flashvsr there, main 5733ec6), its names and
request keys kept.

### Licence

Code (github.com/OpenImagingLab/FlashVSR @ 0d403ee3: LICENSE is Apache-2.0) and weights
(huggingface.co/JunhaoZhuang/FlashVSR-v1.1 @ 2aa7e1f3: `license: apache-2.0`) are Apache-2.0, without a non-commercial
clause; the base model (Wan 2.1 T2V-1.3B) is Apache-2.0 too. The silo's attention kernels are Intel's ARK
(auto-round's auto_round_kernel: the sycl-tla wrappers MIT, the host code Apache-2.0) on sycl-tla (BSD-3-Clause), from
the build image.

### Files

| Part | File | Size |
|---|---|---|
| DiT | `JunhaoZhuang/FlashVSR-v1.1` `diffusion_pytorch_model_streaming_dmd.safetensors` (float32, made half at load) | 5.7 GB |
| LQ projection | `LQ_proj_in.ckpt` (bfloat16, `Causal_LQ4x_Proj`, one linear layer) | 576 MB |
| decoder | `TCDecoder.ckpt` (float32, channels [512, 256, 128, 128], 784 in) | 189 MB |
| prompt | `OpenImagingLab/FlashVSR` `examples/WanVSR/prompt_tensor/posi_prompt.pth` (bfloat16 [1, 512, 4096]) | 4.2 MB |

The checkpoints are read by `nextsycl_gguf::pth` (the pickle's tensors only, no code run). On hardac1 they are in
/var/mnt/2TBSSD/nextsycl-models/flashvsr (sha256 checked against the card).

### The pipeline (diffsynth/pipelines/flashvsr_tiny_long.py, v1.1's scripts)

- each frame Pillow-bicubic to scale x its size, mirrored (numpy 'reflect') out to multiples of 128 - the official script
  centre-crops to them instead, which would cut 32 pixels from every side of 1728 x 960 (nextcuda's choice, kept) - to
  [-1, 1]; the clip's last frame repeated 4 to 11 times past its end so every frame comes out;
- chunk 0: frames 0-24 through the LQ projection (7 causal calls: the first only fills its memory) -> 6 latent frames of
  tokens; chunk k: frames 8k + 17 .. 8k + 24 -> 2;
- the DiT once at t = 1000 on the noise's patches plus the LQ tokens (before block 0): 30 blocks of 1,536 (12 heads of
  128), self-attention block-sparse over windows of 2 x 8 x 8 tokens - per head the means of each 128-token block of q
  and k score every key block inside an 11 x 11 window of blocks (`local_range`), the softmax of those scores keeps its
  top k across each query group (`int(nb^2 sparse_ratio 768 1280 / (H W)) - 1`), and each query block attends to the
  key blocks it kept, over the chunk's own pair and the three cached ones; cross-attention to the fixed prompt (its keys
  and values made once at load); a GELU MLP of 8,960; the head; latents = noise - output;
- the tiny decoder on the latents with the chunk's frames (8 x 8 x 4 unshuffled) as its condition, its MemBlocks carrying
  the frame before; the stream's first 3 frames dropped; chunk 0 gives frames 0-20, chunk k 8k + 13 .. 8k + 20;
- the colour fix: AdaIN per channel against the upscaled input frame (the reference's `method='adain'`, over the picture's
  box here: the mirrored margins are not the picture's), clamped, bytes truncated as tensor2video does.

### The engine

- **Tokens in window order** throughout (nextcuda's): everything but the attention is row-wise, so the 2 x 8 x 8 windows
  are runs of 128 rows from the patches in to the head out; the LQ tokens are added and the RoPE table built in that order.
- **The KV ring**: four pair slots a layer; chunk n's pair goes to slot (n + 2) % 4, the slot of the pair four before it,
  which chunk n no longer sees, so the attention reads the ring in place (its lists name slots) and nothing moves. The
  draft's threshold rank is the reference's (min(topk, nb x seen - 1) + 1 over the blocks it sees: three pairs for
  chunk 0, four after).
- **Products on oneDNN** (`dnn.cpp`), half in, float32 accumulation, with what follows each fused: bias, GELU, the gate
  and the residual sum into the float32 stream (`binary_mul` + `sum` post-ops); the convolutions likewise (a MemBlock's
  first convolution as two summed, its last summed onto its input in place, ReLU fused). The LQ projection's causal 3-D
  convolutions are one 2-D convolution each over their four input frames side by side in the channels around a
  replicated border.
- **The silo** (`silo/tla/*.cpp`, `dist/silo/libnextsycl-flashvsr.so`): ARK's native-precision block-sparse SDPA on
  sycl-tla for the self-attention (a LUT row a 64-query tile, delta-encoded lists of 128-key route blocks; a row with no
  block zeroed after it, as the reference's kernel gives 0) and ARK's dense prefill kernel for the cross-attention. Built
  apart with sycl-tla's flags (`kernels/build.sh`'s silo/tla stanza). Without it the engine's own plain kernel and
  oneDNN's fused attention run (slow; the same results).

### Checked (exact before fast)

`reference/flashvsr/ref.py` (nextcuda's, a thread limit added) runs the official pipeline on the CPU in float32
(block_sparse_attn stood in for by its exact equivalent: per head, softmax over the selected blocks' keys, a row with
none 0) and dumps each stage; `nextsycl image check DIR --model flashvsr` feeds the engine the reference's prepared
frames and noise and compares as it goes (relative L2 error and cosine, the residual stream in the reference's token
order). The input: 33 frames of an H3 film (h3_20261105_134939, scaled to 512 x 384), x2 -> 1024 x 768 (nextcuda's
size; its frames are not published, so a clip of ours), sparse_ratio 2.0, local_range 11 - the reference's run took 12
minutes on 16 threads of the 9950X. The dumps and our frames stay on hardac1 (/var/mnt/2TBSSD/nsref/flashvsr/d33).

| stage | chunk 0 (6 latent frames): rel / cos | chunk 1 (2): rel / cos |
|---|---|---|
| LQ projection (block 0's residual) | 2.16e-4 / 1.000000 | 2.28e-4 / 1.000000 |
| after block 0 | 1.51e-4 / 1.000000 | 1.52e-4 / 1.000000 |
| after block 15 | 1.35e-3 / 0.999999 | 1.26e-3 / 0.999999 |
| after block 29 | 8.25e-4 / 1.000000 | 8.74e-4 / 1.000000 |
| latents (noise - the DiT's output) | 1.19e-3 / 0.999999 | 1.18e-3 / 0.999999 |
| decoded frames (37, before the colour fix) | worst 8.9e-4 (max 1.5e-2) | |
| the bytes (after the colour fix) | PSNR 61.5 dB mean, 61.2 dB worst, against the reference's | |

nextcuda's figures on its own clip: the LQ projection 3.8e-4, block 0 2e-4, block 15 2.3e-3, block 29 1.9e-3, the
latents 1.8e-3, the decoded frames 9e-4, 64.1 dB. Ours are as close or closer at every stage; the bytes' PSNR is
content-dependent (it counts the pixels a rounding moves across a byte boundary) and comes from the same check code.

Half precision (nextcuda's choice, with float32 accumulation and a float32 residual stream) holds: no non-finite value
at any stage (SeedVR2's fp16 attention overflowed; here q and k are RMS-normed before the attention and the residual
stream stays float32). The first port (ARK's 64-query tile) and every tile since (128 queries with 64 or 32 keys a
step, 256) give the same figures to the digits shown (the summation order moves the latents' error between 1.13e-3
and 1.19e-3); the decoder in bands gives the same frames as whole.

### Speed: the optimisation log

Each figure is a whole `video.upscale` request through the image server (ffmpeg decode and x264 encode included) of
the 243-frame H3 clip h3_20261105_134939 (864 x 480; the 1280 x 704 rows are the same clip scaled up with Lanczos -
none of the platform's clips is that size), seconds / frames, sparse_ratio 1.5, local_range 11. One card is the B70
(gpu 0); two are the B70 + B65 (`NS_FVSR_GPUS=0,1`, the decoder on the B70). s/frame; nextcuda: 0.225 at 864x480 x2
on its RTX 5070 + 3060, 1.16 at x4.

| # | change | 864x480 x2, one card | two cards | 1280x704 x2, one card | two cards |
|---|---|---|---|---|---|
| 0 | the first working port (ARK's 64-query sparse tile; 17 / 13 blocks) | 0.192 | 0.139 | 0.539 (did not fit whole: 2 x 2 tiles) | 0.288 |
| 1 | the attention a work-group a 128-query block, 32 keys a step (7.05 -> 4.62 s of 65 frames' attention); K and V one product straight into the ring (rows of 3,072); the RoPE as two small tables (not 22 MB a chunk made on the host); the decoder's full-size end in bands of 256 rows | 0.158 | 0.116; 15 / 15: 0.110 | out of memory (oneDNN's reordered copy of each weight, a copy a layout) | 0.227 |
| 2 | the linear layers' weights read as they are (no reordered copies: 2.4 GiB less, and the products 2.88 -> 2.71 s); the two cards' split tried | 0.157 | 14 / 16: 0.107 (both 86-87 % busy); 16 / 14: 0.113 | 0.310 (whole, 30.3 of 31.9 GiB) | |
| 3 | frames and bytes through pinned memory by the queue's copies (one wait a chunk, not a copy and a wait a frame); 14 / 16 the default; 2560 x 1408's buffers trimmed (the head's rows in the LQ tokens' buffer, TGrow 21 a frame at a time, the MLP 8,192 rows at a time) | 0.155 | 0.107 | 0.309 (29.9 GiB) | 0.214 |
| 4 | the sparse attention's prefetch: each 32-key step of a 128-key block prefetched the whole next block (4 x its 32 KB a block); now one step's tile of it (ARK's mainloop copied into the silo, `silo/tla/ark/`). The clip's buffers let go when it ends (below) | 0.140 (95 % busy) | 0.099 (86 / 80 %) | 0.279 (96 %, 29.9 GiB) | 0.194 (85 / 84 %) |

Tried after step 4 (B70, 65 frames' attention, 3.58 s for the default 129): skipping the output's rescale when no
row's maximum moved in a sub-group (exact) - 3.58 s, no gain, taken out; 64 keys a step (128) 5.78 s. **Known bad: a
P V tile 64 columns wide (`NS_FVSR_QTILE` 134, and 136 with 64 keys a step) hung the B70** - the compute engine at
100 % for 8 minutes on the first chunk of 1728 x 960 (it takes 11 s for 65 frames), SIGTERM did not end it, the
process was killed (2026-11-06 15:28). Both shapes are gone from the silo. 4 sub-groups of 32 rows (135), on the B65:
26.0 s against the default's 5.7 (0.555 s a frame against 0.247) - also dropped; the silo is step 4's.

Step 4's attention: 4.65 -> 3.54 s of 65 frames' (58.5 -> 76.8 TFLOPS, 31.9 -> 41.9 % of the peak; 42.3 -> 35.5 % of
the time); the check unchanged (decoded worst 8.9e-4, 61.49 dB).

A detour: the pinned memory read and written by the kernels in place (over PCIe) made a frame 0.210 s - a kernel's
small reads across the bus; the queue's copies (the copy engines) are what step 3 keeps.

### Speed now (the final build: the whole 243-frame clip unless said; the x4 rows are step 3's)

| size | one card (B70) | two cards (B70 + B65) |
|---|---|---|
| 864x480 -> 1728x960 (x2) | **0.140 s/frame** (step 4; 0.156 at step 3) - B70 95 % busy, 17.3 GiB | **0.099** (0.107) - 86 / 80 % busy, 10.7 / 7.8 GiB |
| 1280x704 -> 2560x1408 (x2) | **0.279** (step 4; 0.310 at step 3), whole - 96 %, 29.9 GiB (2.0 free) | **0.194** (0.213) - 85 / 84 %, 18.2 / 13.6 GiB |
| 864x480 -> 3456x1920 (x4: 2 x 2 tiles of 1792 x 1024) | **0.753** - 97 %, 1.0 core, 17.3 GiB | **0.547** - 81 / 83 %, 1.4 cores, 10.7 / 7.8 GiB |
| 1280x704 -> 5120x2816 (x4: 3 x 3 tiles; 65 frames) | **1.542** - 98 %, 1.0 core, 17.3 GiB | **1.286** - 70 / 73 % (each tile's stream fills the pipeline anew), 1.3 cores, 10.7 / 7.8 GiB |

Busy is the compute engine's share of the run (the xe driver's per-client counters in /proc/PID/fdinfo, every 0.5 s);
the memory the process's resident VRAM at its peak (fdinfo's drm-resident-vram0, sampled), the same as the card's least
free memory the engine saw. CPU: the server process's cores (ffmpeg's own processes not counted). A 5,665-frame film at
2x of 864 x 480: about 15 minutes on the B70, 10 on both cards.

### Where the time goes (one card; `NS_FVSR_PROFILE=1`, 65 frames, each step waited for)

Peaks: the B70's XMX 183.5 TFLOPS (32 Xe cores x 2048 FLOPs a clock x 2.8 GHz; half and bfloat16 alike), its memory
608 GB/s, pinned host -> card ~28 GB/s. FLOPs from the shapes (a product 2 M K N, a convolution 2 px Co Ci k^2, the
sparse attention 4 x 128^3 for each 128 x 128 block a query block selected - counted from its lists); bytes as the least
each step must move (the attention's: each selected K / V block read once a query block, an upper bound - neighbouring
query blocks share most of theirs in the cache).

| step | 1728x960: share | s | TFLOPS (% of peak) | GB/s | 2560x1408: share | TFLOPS (%) |
|---|---|---|---|---|---|---|
| DiT sparse attention (silo, ARK on sycl-tla, 128-query tile) | 42.3 % | 4.65 | 58.5 (31.9 %) | 462 (block reads) | 44.0 % | 57.3 (31.2 %) |
| DiT products (oneDNN, half, epilogues fused) | 24.7 % | 2.72 | 119.8 (65.3 %) | 138 | 25.2 % | 119.9 (65.3 %) |
| LQ projection convolutions | 6.3 % | 0.69 | 129.2 (70.4 %) | 23 | 5.5 % | 150.2 (81.8 %) |
| decoder convolutions at 1/2 | 5.4 % | 0.60 | 113.1 (61.6 %) | 231 | 4.9 % | 128.1 (69.8 %) |
| decoder convolutions at full size (bands) | 5.3 % | 0.59 | 135.3 (73.8 %) | 235 | 5.3 % | 139.2 (75.9 %) |
| decoder convolutions at 1/8 | 3.7 % | 0.40 | 87.0 (47.4 %) | 48 | 3.0 % | 108.5 (59.1 %) |
| DiT norms, RoPE (bandwidth) | 3.4 % | 0.38 | - | 412 (68 %) | 3.4 % | - |
| decoder convolutions at 1/4 | 3.3 % | 0.36 | 92.7 (50.5 %) | 95 | 2.7 % | 116.4 (63.4 %) |
| DiT cross-attention (silo, ARK dense) | 1.6 % | 0.18 | 68.2 (37.2 %) | 133 | 1.6 % | 70.9 (38.7 %) |
| DiT draft (block means, scores, top k, lists) | 1.4 % | 0.15 | - | 155 | 2.0 % | - |
| decoder's last convolution (128 -> 3) | 1.4 % | 0.15 | 6.2 (3.4 %) | 235 | 1.2 % | 7.0 (3.8 %) |
| decoder upsample, input; colour fix and bytes | 1.3 % | 0.13 | - | 490 (81 %) | 1.3 % | - |

The attention is the next lever: at a third of the peak it is the largest share, and its kernel reads each selected
K / V block from memory for every 128 queries (ARK's 256-query tile, which shares them between two query blocks,
walks the union of their lists and was no faster; tiles of 16 sub-groups or 16 keys a step were slower).

### Memory (one card)

At 2560 x 1408 the KV ring alone is 19.4 GiB (30 layers x 4 pair slots x 28,160 tokens x 3,072 halves): the steps that
made the size fit whole are the weights read in place (oneDNN's reordered copies were 2.4 GiB, and a second copy a
layout), the decoder's full-size end in bands (three 0.9 GB buffers become 0.17 GB ones), and the head's rows, TGrow 21
and the MLP's hidden rows trimmed. The run peaks at 29.9 of 31.9 GiB; `room_for` (lib.rs) refuses a size whose stream,
weights, 1.5 GiB margin and 1 GiB of runtime pass the card's memory with a plain error (400), and a size that fits an
empty card but not beside what holds it now with no_room (503, retryable) - before anything is allocated. A size
that cannot fit whole runs in tiles instead (x4; `NS_FVSR_TILE=0` forbids them, which is how the 400 is shown).

When a clip ends its buffers go (the KV ring, the decoder's past, the projection's state): the server's VRAM falls from
17.7 to 3.9 GiB within half a second of the answer (fdinfo, 1728 x 960). Kept, an idle FlashVSR held its card from
SeedVR2 on the same card (no_room in production); `NS_FVSR_KEEP=1` keeps them for the next clip.

### Checked through the server

`video.upscale` with the platform's keys at the top level (`seed: 3, sparse_ratio: 2.0, local_range: 9,
color_fix: 0`) and in `options`; `image.upscale` of a picture (a still clip); a size that can never fit (x4 of
1280 x 704 whole: 400, "needs 103.3 GiB ... and the card has 31.9 in all"); an unknown option (400, the list of
options). The plain kernels (`NS_SILO=0`) give the reference's figures too (latents 1.08e-3, 61.6 dB).

### The attention, measured further (B65, 2026-11-06; main a3fcd0a's build)

The B65 (98.3 TFLOPS half: fewer Xe cores than the B70, the same memory), 65 frames of 864 x 480 x2, each step
timed alone: the sparse attention 5.62 s, 37.3 % of the timed work, 48.4 TFLOPS (49 % of the B65's peak); the DiT's
products 4.06 s (27 %, 80 TFLOPS, 82 %); the decoder's convolutions 2.83 s together (19 %, 65-89 %); the LQ
projection 1.16 s (7.7 %, 78 %); norms and RoPE 0.53 s; the cross-attention 0.26 s (47 %); the decoder's 128 -> 3
convolution 0.25 s (1.7 %, 3.8 %); the draft (block means, scores, top k, lists) 0.17 s (1.1 %). At 2560 x 1408 the
same shares (attention 39 %, 47 %). Whole clips on the B65: 864x480 x2 0.215 s/frame, 1280x704 x2 0.434, 864x480 x4
1.040; the check as before (decoded 8.9e-4, 61.52 dB).

What limits the attention:

| try | attention, 65 frames | what it says |
|---|---|---|
| main (129: 128 query rows a work-group, 8 sub-groups of 16, 32 keys a step) | 5.62 s (49 %) | |
| every LUT row walking the same blocks 0, 1, 2, ... (a probe, `NS_FVSR_ATTN_PROBE`, not kept: wrong output, the same work, the K / V blocks perfectly reused in L2 and L1) | 5.65 s | **not bound by memory**: the blocks' reads from L2 / DRAM cost nothing measurable. Keeping K / V tiles in shared local memory and sharing them between the work-group's sub-groups attacks traffic that is not the limit: each sub-group still has to bring the whole K and V tile into its own registers for its matrix products, from SLM or from L1 alike (the same load path on Xe2, and SLM has no 2-D block loads with the VNNI transform the B operand needs) |
| the query rows held in registers instead of reloaded for every key tile (an `NS_FVSR_QREG` build of the mainloop) | 10.40 s (27 %) - 1.85x slower; 864x480 0.277 s/frame, 1280x704 0.550, x4 1.333 | **bound by the register file**: the sub-group's output accumulator (16 rows x 128 float32 = 8 KiB) already holds half of a thread's 256 registers; 4 KiB more of query halves the threads an Xe core keeps resident (or spills) |
| 4 sub-groups of 32 rows (135) | 26.0 s | the same, worse: 16 KiB of accumulator a sub-group |
| a P V tile 64 columns wide (134, 136) | hung the B70 | known bad |

So the lever is not K / V reuse but registers per sub-group. The design that would move it: the output's 128 columns
split between two sub-groups that share one block of query rows (each 4 KiB of accumulator: twice the resident
threads, or room for the query rows), the scores computed once and the probabilities passed through shared local
memory - a new kernel (the sycl-tla collective rewritten), not a shape of this one. The other parts are near their
peaks (the products 82 %, the decoder's large convolutions 78-89 %); after the attention the next levers are small:
the 128 -> 3 convolution (1.7 % of the time at 3.8 % of peak: a direct kernel, ~1.4 % back) and the decoder's 1/8
and 1/4 convolutions (7 % at 66-68 %).

### The decoder's small convolutions (B65, 2026-11-06)

`NS_FVSR_PROFILE=shapes` names the decoder's convolutions by shape (65 frames of 864 x 480 x2, oneDNN):

| convolution | s | TFLOPS (% of 98.3) |
|---|---|---|
| full size 3x3 128 -> 128 | 0.90 | 88 (90 %) |
| 1/2 3x3 128 -> 128 (plain / summed onto x) | 0.37 / 0.39 | 80 / 75 |
| 1/2 3x3 256 -> 128 | 0.13 | 78 |
| 1/4 3x3 256 -> 256 (plain / summed) | 0.21 / 0.21 | 71 / 66 |
| 1/4 3x3 512 -> 256 | 0.09 | 57 |
| 1/8 3x3 512 -> 512 (plain / summed) | 0.25 / 0.21 | 68 / 68 |
| 1/8 3x3 784 -> 512 | 0.08 | 47 |
| the last, 3x3 128 -> 3 | 0.25 | 3.7 |

Tried for the last one, both slower, not kept: a direct kernel (a work-item a pixel, the weights in local memory,
float32 sums) 0.57 s; four pixels a work-item 5.6 s (private arrays spilled); the convolution made with 16 outputs (the
weights zero-padded, a shape for the matrix engines) and its 3 channels taken after 0.30 s - oneDNN's 3-output
convolution is not compute-bound at 3.7 % but bound by its 288 band-sized calls (0.9 ms each, the same with 16
outputs). The 1/8 and 1/4 convolutions together are 1.05 s of the 15.1 timed (7 %); at the large ones' 85-90 % they
would save ~0.2 s (1.3 %). They run a frame at a time (the decoder's causal past); a chunk's frames through each layer
together (the past is the frame before's input to the same layer) would give oneDNN larger problems - a restructuring
of tcd.rs, not a shape, left for later. None of the decoder's parts is the lever the attention is.

### The split-output attention: tried, stopped (B65, 2026-11-06)

The design: a work-group of 16 sub-groups for one 128-row query block, eight pairs of 16 query rows; each 64-key
tile in local memory once for the work-group (K transposed, the B operand of q k^T row-major); in a pair, one
sub-group scores keys 0-31 and the other 32-63, their row maxima and sums meet in local memory, the probabilities
(half) are written there, and each multiplies all 64 by its 64 of V's 128 columns - 16 x 64 float32 of output a
sub-group (4 KiB, not ARK's 8) with its 16 query rows held in registers (4 KiB). Written as plain SYCL with
joint_matrix (8x16 accumulators, 8x16 A, 16x16 B; the B65 offers M <= 8, N 16, K 16 for half) behind
`NS_FVSR_ATTN=split`.

| | result |
|---|---|
| a unit test of the layouts it relies on (an accumulator's wi_data[i] = row i of the lane's column; A and B row-major from global and local memory) | as assumed |
| its scores against the CPU (a 128 x 128 head) | exact (worst 3.8e-3 on values ~10) |
| its probabilities | wrong for every row but the first of each 8-row tile - the per-row maxima and exponentials read through `get_wi_data` (an internal API) after `reduce_over_group` do not land on the rows the scores did; not resolved |
| the whole pipeline (`image check`) | 8.5 dB (wrong, as above) |
| its speed (65 frames, 864 x 480 x2) | **121.1 s against ARK's 5.62 - 2.2 % of the B65's peak, 21x slower** |

Why it stopped (the coordinator's rule: report if a solid attempt does not beat 49 %): the bytes a sub-group must
bring into its registers per FLOP do not shrink much when the output is split. Per 32 keys, ARK's sub-group (16 rows x
128 columns) loads Q 4 KiB + K 8 + V 8 = 20 KiB for 262k FLOPs (0.076 B/FLOP); a split sub-group (its 16 rows, 16 of
the keys for q k^T, 64 of the columns for P V) loads K 4 + V 4 + P 1 = 9 KiB for 131k FLOPs (0.069 B/FLOP, Q held) -
10 % less, before the exchange through local memory and its barriers. A 64-key tile's work-group does ~2.1 MFLOP and
reads ~144 KiB from local memory into registers; at ~128 B a clock an Xe core's local memory needs ~1,150 clocks for
that, about as long as the matrix engines need for the FLOPs (~1,000): the local-memory path is the same wall the L1
path is, and Xe2 has no 2-D block loads (with the VNNI / transpose the B operand wants) from local memory - the
loads ARK's kernel gets from L1. The ceiling of this design is about ARK's, not past it; reaching even that needs
the sycl-tla collective rewritten with 2-D block loads and prefetch, not joint_matrix over local memory (this first
version's scattered transposing stores and three work-group barriers a tile cost it 20x). The attempt's source (a
patch, not merged: wrong output) is kept beside the build copy on hardac1 (~/src/nextsycl-flashvsr/split-attempt.patch).

### Open

- The attention: 37 % of a frame at 49 % of the peak, bound by registers and the on-core load path; the split-output
  design measured and analysed above does not lift that ceiling. What would: fewer bytes per FLOP into registers -
  larger query tiles a sub-group (needs the accumulator somewhere other than registers), or fp8/int8 K/V (changes the
  numbers: not exact).
- The decoder's 128 -> 3 convolution on oneDNN is at 3 % of peak (1.4 % of the time): a direct kernel would do.
- No clip of the platform is 1280 x 704: those rows are the 864 x 480 clip scaled up. x4 seams were not examined by eye.
- Through the production front door (:8000, 2026-11-06): x4 of 864 x 480 0.753 s/frame; a 1344 x 768 film clip
  (`file`) at x2, 2688 x 1536, 0.526 s/frame, 19.2 GiB; no black frame in any.
