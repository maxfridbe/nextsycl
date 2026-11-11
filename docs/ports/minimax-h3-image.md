# MiniMax-H3-Image (image/h3image)

MiniMax's image model (MiniMaxAI/MiniMax-H3-Image, released 2026-11-10) as the GGUF files of Hearthscale
(Hearthscale/MiniMax-H3-Image-GGUF @ 167c5e18 for Q3_K_M / Q4_K_M / Q5_K_M, @ 2cd99b1e for Q8_0, converted by
stable-diffusion.cpp master-1088 from Comfy-Org's pruned bf16 file) and Comfy-Org's own int8 ConvRot file
(Comfy-Org/MiniMax-H3-Image @ 892de7cb). Text to image and edits, 1 to 16 MP, 6 steps, cfg 1, Euler on the model's own
grid. The reference is ComfyUI (comfyanonymous/ComfyUI dba377b9: `comfy/ldm/minimax/model.py` MiniMaxH3ImageModel,
FinalLayer and `_pdd_head`; `sampling.py`; `vae.py` MiniMaxH3ImageVAE; `comfy_extras/nodes_minimax_h3.py`
MiniMaxH3ImageConditioning) and stable-diffusion.cpp 32d516c (the "minimax-h3-image" schedule), read, not run;
`reference/h3-image/ref.py` writes it out in torch on the CPU in float32 from the bf16 file.

## What it is, and what was reused

The checkpoint is MiniMax H3's 50-block denoiser (5,376 wide, 56 heads, the 2-block token refiner, the same AdaLN
timestep-curve tables, the same packed layout and rotations) with three differences: **no audio stream**
(`audio_patch_proj` / `audio_out` gone), **a bank of six output heads** (`video_out` is [6 x 96, 5376]: head 0 a full
head, heads 1-5 offsets distilled on the grid 1, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0 - a step adds the heads of the
grid intervals it spans, weighted by their share), and **its own VAE decoder** (a 2D ViT: 24 blocks, 1,024 wide, 16
heads, rotation on half of each head over height and width; the file also carries H3's conv encoder but no latent
statistics - H3's are used). The text encoder is H3's own (Qwen3-VL 32B cut to 50 layers, the hidden state after
layer 50), presented as written; an edit shows each picture as `"Picture i: "` + its vision block.

So the engine is P-H3-Image's shape on H3's foundation (`crates/h3`), reused unchanged where it matches:

- `dit::Blocks` (the 50 blocks, int8 or k-quant expand-then-bf16 products), `TextRefiner`, `layout::Layout` /
  `Timesteps` (audio length 0), `jobs::encode_pieces` (the streamed 32B text encoder + the vision tower),
  `venc::VideoEncoder` (the edit's latents), the noise generator. The draft's own block stack (`stack.rs`, 826 lines)
  and host k-quant decoder (`weights.rs`) were dropped in favour of these.
- New in `image/h3image`: `net.rs` (the no-audio sequence, the head bank, the final layer, Euler as ComfyUI's
  sample_euler), `vae.rs` (the 2D decoder: P-H3-Image's decoder code adapted, ComfyUI's tiling that blends each tile
  with its neighbours as already blended), `lib.rs` (the image contract, sizes, edits), `check.rs`.
- Additive in `crates/h3`: **Q3_K** (`kquant.rs`). A Q3_K block (d, sixteen 6-bit scales s, values q in -4..3:
  d (s - 32) q) is exactly a Q6_K block with the same d, int8 scales s - 32 and 6-bit values q + 32; the GGUF reader
  transcodes Q3_K matrices to Q6_K as it reads them (bit-for-bit the same floats, unit-tested), so the shared Q6_K
  expansion serves them - no kernel touched, at Q6_K's size on the card. And `venc` takes H3's latent statistics when a
  VAE file has none.
- Silo `kernels/image/h3image/silo` (`dist/silo/libnextsycl-h3image.so`): P-H3-Image's attention silo copied with its
  own symbols (SageAttention's preparation passes rewritten to stream; Sage from 1,024 tokens).

## Parity (reference: ref.py, the bf16 file in float32 on the CPU; 576 x 384, seed 3, 6 steps)

The prompt's conditioning comes from the engine's text encoder (`check --stages te`, then the reference reads it): the
text encoder is the video engine's, checked in its own port. Each step is fed the reference's own latents
(teacher-forced), so a step's error is its own. B70.

| Stage | int8 ConvRot | Q8_0 | Q5_K_M | Q4_K_M | Q3_K_M (as Q6_K) |
|---|---|---|---|---|---|
| refined text | 5.8e-3 | 7.2e-3 | 3.0e-2 | 5.8e-2 | 1.2e-1 |
| stream after block 49, step 0 | 5.4e-2 | 4.3e-2 | 1.1e-1 | 1.9e-1 | 3.2e-1 |
| velocity, steps 0 .. 5 | 1.9e-2 - 5.9e-2 | 2.1e-2 - 5.3e-2 | 5.6e-2 - 1.2e-1 | 9.2e-2 - 1.3e-1 | 1.6e-1 - 2.5e-1 |
| velocity cosine, worst | 0.99825 | 0.99860 | 0.99330 | 0.99104 | 0.96938 |
| final latents, 6 steps from the reference's noise | 1.4e-1 | 1.4e-1 | 2.5e-1 | 4.2e-1 | 5.5e-1 |

- Positions, table rows, noise (torch's draws), the heads' weights each step: exact.
- VAE decoder (fp16 file, half products, float32 stream) on the reference's final latents: 2.1e-4 relative, 98.95 % of
  the bytes equal, 100 % within 1.
- The error falls with the bits as it should (Q3 > Q4 > Q5 > Q8 ~ int8); int8's step-0 velocity (1.9e-2) is P-H3-Image's
  int8 figure (2.9e-2). The 6-step end latents differ more than P-H3-Image's 8-step run did (the 576 x 384 test is far
  from the model's 4 MP and its steps are long): judge by the per-step numbers, as for H3.
- The silo's Sage forced onto the 237-token test (NS_H3I_SAGE_MIN_S=64): int8 velocities 3.5e-2 - 7.3e-2 (oneDNN's bf16
  attention 1.9e-2 - 5.9e-2) - Sage's int8 q / k cost, as in P-H3-Image; real sizes (>= 1,024 tokens) run it.
- An edit (`image edit`, one of our own renders, Q4_K_M, 1536 x 1024): the picture is kept and changed as asked (a
  scarf added). Not checked against the reference (ref.py has no edit stage).

## Speed

`image gen --local`, the second picture of a request (n = 2: no text encoder), 6 steps; "steps" is the 6 denoiser
steps, "all" adds the refiner (0.1 - 0.2 s) and the VAE decode. The text encoder (Qwen3-VL 32B Q4_K_M streamed from
pinned host memory) adds 1.8 - 2.8 s for a short prompt once the process has it pinned (12 s for the very first,
reading the file); an edit adds the vision tower and its long prompt (1,559 tokens: 26.6 s the first time) and the
picture's latents (5.2 s). Card memory: the front door's per-card reading, sampled each second (peak).

**Before / after the silo** (B70, 6 steps, seconds):

| | Q4_K_M before | Q4_K_M after | int8 before | int8 after |
|---|---|---|---|---|
| 1024 x 1024 (1 MP) | 3.85 | 3.72 | 1.95 | 1.84 |
| 2048 x 2048 (4 MP) | 11.46 | 10.52 | 7.50 | 6.54 |
| 2912 x 2912 (8.5 MP) | 23.96 | 22.27 | 16.39 | 14.57 |
| 4096 x 4096 (16.8 MP) | 53.57 | 50.27 | 39.32 | 35.77 |

**Now, by card and file** (steps / all, seconds; peak card memory GiB at 16 MP):

| | B70 Q4_K_M | B70 int8 | B65 Q4_K_M | B65 int8 | B70 Q5_K_M | B70 Q8_0 | B70 Q3_K_M |
|---|---|---|---|---|---|---|---|
| 1 MP | 3.72 / 3.9 | 1.84 / 2.1 | 5.45 / 5.7 | 2.75 / 3.1 | - | - | - |
| 4 MP | 10.52 / 11.0 | 6.54 / 7.2 | 15.88 / 16.6 | 9.94 / 10.8 | 10.58 / 11.1 | 10.83 / 11.4 | 10.66 / 11.2 |
| 8.5 MP | 22.27 / 23.2 | 14.57 / 15.6 | 34.63 / 35.9 | 22.34 / 23.7 | - | - | - |
| 16.8 MP | 50.27 / 52.0 | 35.77 / 37.6 | 79.01 / 81.4 | 55.32 / 57.8 | 50.36 / 52.3 | 50.58 / 52.3 | 50.08 / 52.0 |
| card, 16 MP | 14.4 | 21.7 | 14.4 | 21.7 | 16.5 | 22.9 | 18.8 |

Where the time goes:

- Every GGUF runs at the same speed (4 MP 10.5 - 10.8 s, 16 MP 50.1 - 50.6 s): each matrix is expanded to bf16 and
  multiplied at the 16-bit rate; the expansion is a few percent. A GGUF's size buys memory, not speed. The int8 file is
  1.4 - 2x faster (int8 products) and as close to bf16 as Q8_0.
- Attention grows with the square of the tokens (4,096 at 4 MP, 16,384 at 16 MP): from 8.5 to 16.8 MP the steps take
  2.3 - 2.5x for 2x the pixels. The silo's Sage passes saved 6 - 13 % a picture.
- The decoder: 0.09 s at 1 MP, 1.6 s (B70) / 2.3 s (B65) at 16 MP (441 tiles); the refiner, reloaded per picture,
  0.1 - 0.2 s.
- Against P-H3-Image's int8 path (the baseline asked for: 8 MP, B70 22 s): MiniMax-H3-Image int8 at 8.5 MP is 15.6 s on
  the B70 (6 steps against P-H3-Image's 8: the same per step, ~2.4 s), 23.7 s on the B65 (P-H3-Image's B65 8 MP: 34 s).

## Kept and lost

- Kept: text to image and edits (up to 16 pictures), 1 - 16.9 MP, the heads' 6-step grid and other step counts read off
  it as stable-diffusion.cpp does, every file the model has (Q3_K_M, Q4_K_M, Q5_K_M, Q8_0 GGUFs; int8 ConvRot).
- Lost / not run: MiniMax's prompt enhancer (MiniMaxAI/MiniMax-H3-Image-PE, a Qwen3.8-27B fine-tune) - prompts are
  rendered as written; Comfy-Org's w4a8 file (needs a 4-bit codebook kernel); the bf16 file (40 GB: does not fit a card;
  it is the reference). Q3_K_M holds Q6_K's size on the card (14.7 GiB of blocks, more than Q4_K_M's 10.1): it saves
  download, not memory.

## Open

- Faster GGUF products: a fused Q4_K / Q5_K / Q8_0 x bf16 XMX kernel in the silo would only remove the few-percent
  expansion; the real lever for GGUF speed is int8 products, which the int8 file already gives with less error. Not done.
- Attention at 16 MP (half of a step by P-H3-Image's split): a better long-sequence kernel would help most there.
- Edit parity against the reference (ref.py has no edit stage); the vision tower and VAE encoder are H3's, checked in
  the video port.
- Keep the refiner and the VAE encoder resident between requests.
- The license: hosted service to third parties needs MiniMax's written authorization (see the cards).
