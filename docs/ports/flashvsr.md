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

`reference/flashvsr/ref.py` (nextcuda's, a thread limit added) runs the official pipeline on the CPU in float32 (block
_sparse_attn stood in for by its exact equivalent: per head, softmax over the selected blocks' keys, a row with none 0)
and dumps each stage; `nextsycl image check DIR --model flashvsr` feeds the engine the reference's prepared frames and
noise and compares as it goes. The input: 33 frames of an H3 film (h3_20261105_134939), 512 x 384 -> 1024 x 768
(nextcuda's size), sparse_ratio 2.0, local_range 11 (the reference's run; 12 minutes on 16 threads of the 9950X).

(results below)

### Speed: the optimisation log

Each figure is a whole `video.upscale` request through the image server (ffmpeg decode and x264 encode included) of
the 243-frame H3 clip h3_20261105_134939 (864 x 480; the 1280 x 704 rows are the same clip scaled up with Lanczos -
none of the platform's clips is that size), seconds / frames, sparse_ratio 1.5, local_range 11. One card is the B70
(gpu 0); two are the B70 + B65 (`NS_FVSR_GPUS=0,1`, blocks 17 / 13, the decoder on the B70).

| # | change | 864x480 x2, one card | two cards | 1280x704 x2, one card | two cards | 864x480 x4, one card |
|---|---|---|---|---|---|---|
| 0 | the first working port | 0.192 | 0.139 | 0.539 (did not fit whole: 2 x 2 tiles) | 0.288 | 0.824 (97 frames, 2 x 2 tiles) |
