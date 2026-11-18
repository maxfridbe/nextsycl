# Open-H3-Regenerate-2K-IC: a 768p H3 draft made again at 2560x1440

NVlabs' Open-H3-Regenerate-2K-IC (Sana sol-engine; weights huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC
at 54a7104b; ComfyUI nodes github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC) regenerates a MiniMax H3 draft at 2K
while keeping its faces, lip sync, lettering and layout: the draft sits in the context of H3's Ref2VA denoiser as
`<Video 1>`, and an in-context (IC) LoRA makes the high-resolution version of the same clip. In nextsycl it is the
`regen2k` job of the H3 engine (`crates/h3/src/regen.rs`), the `minimax-h3-regen2k-ic` model card, and
`video.generate`'s `regen2k` (the studio's "Regenerate 2K" button on a made clip).

```text
draft .mp4 -> 124 frames at 24 fps, cover-fitted to 1344x768 (>= 113 real ones; the last repeated)
  -> H3 VAE encode, a seeded posterior draw (as the release)                       z  [24, 37, 48, 84]
  -> the 3D-conv latent upscaler to the target's latent size, whole clip in one piece   up [24, 37, 90, 160]
"<Video 1>: " + the draft at 2 fps in frame pairs ("<0.2 seconds>" ...) + the release's task prompt
  + "\n\nScene description: " + caption -> Qwen3-VL 32B (Q4_K_M) with its vision tower
denoiser: the int8 Ref2VA DiT + HyperFlow 8-step (block LoRAs + the two-time embedding) + LMS Sharp (incl. its AdaLN
  curve branches) + the IC LoRA (rank 384)
  rows: [text | z as the reference clip (0.999 z + 0.001 noise, t >= 0.999) | silence latent (kept, t = 1) | target]
  up3: x = (1 - 0.834942) up + 0.834942 noise; Euler on 0.834942, 0.696852, 0.468753, 0 (CFG 1)
  (up4: from 0.923077, 4 steps; hf8: the whole 8-point grid from noise)
-> the release's fast decoder at 2560x1440 (the H3 video decoder with regen_vae standard) -> cut to the draft's real
   frames -> the draft's own sound muxed back
```

## What we reuse, what we downloaded

The release ships its own `base_model/` (MiniMax-H3 @ 42ed227: `transformer_ref` 66 GB bf16, Qwen3-VL bf16, the VAE
fp32). We checked it against what hardac1 already had before downloading it:

| Part | Release | Ours | Check |
|---|---|---|---|
| Ref2VA denoiser | `transformer_ref` (bf16, Diffusers) | Comfy-Org `minimax_h3_ref2va_pruned_int8_convrot` (card `minimax-h3-ref2va`) | block 0's q matrix range-fetched from the release; ours un-rotated (W = (q s) H): rel 0.0088 - the int8 rounding: the same weights |
| latent upscaler | `latent_upsampler/..._3d_conv_v1_bf16` | `h3-upscaler` | same SHA-256 (4f57821f...) |
| LMS Sharp | `lora/minimax_h3_lms_v1.0_r64` | `h3-lora-lms` | same SHA-256 (16f3195b...) |
| text encoder | Qwen3-VL 32B bf16 | the Q4_K_M GGUF + the bf16 vision tower (`h3-te-32b-q4km`, `h3-te-visual`) | the engine's TE against ComfyUI: cosine 0.99994 (h3-modes) |
| video VAE | fp32 Diffusers | Comfy-Org fp16 (`h3-vae`) | the same model (the decoder's parity in docs/video) |
| sharp curve | `adaln_t_table` by byte range from Comfy-Org's int8 Ref2VA | in our int8 file | the same tensor |

Downloaded (4TB drive, `open-h3-regenerate-2k-ic/`): the IC LoRA (8.3 GB, fp32), HyperFlow (2.8 GB), the silence
latent, the fast VAE (6.5 GB, float32; the card's `fast-vae`, below), and **26 GB of the bf16 Ref2VA checkpoint: only its time
embedder and full AdaLN projections** (Comfy-Org `minimax_h3_ref2va_bf16.safetensors`, card file `h3-ref2va-bf16-adaln`
with `tensors: ['time_embedder.', 'blocks.*.adaln_proj.', 'final_layer.adaln_proj.']` - the card's `tensors` now takes a
list of prefixes with `*`). Why that is needed:

## HyperFlow's two-time embedding on the curve-form checkpoint

ComfyUI's pruned checkpoints drop the time embedder: `silu(time_embedder(t))` (2688 wide) lies on an 8-dimensional
curve, stored as `adaln_t_table` [1025, 8], and every AdaLN projection is stored on those 8 coordinates (fit residual
1.4e-5; the curve form against the full projection on the grid: rel 2e-4). HyperFlow's step embedding is
`e_t(t) + 0.25 (e_r(r) - e_t(t))` with LoRAs on the time embedder and on an endpoint copy of it (r = 1 - the next
sigma for the generated rows' timestep, r = t for the others). That embedding is off the curve: projected back on it
the modulation loses 3.2% (reference/regen2k/timecurve.py). So the engine (`crates/h3/src/hyperflow.rs`) builds the
run's tables from the full projections: once per run it lists every (t, r) a step will ask for (`hyperflow::points`),
computes `silu(e(t, r))` on the host (the time embedder and its two LoRAs merged in float32), and reads the 50 blocks'
and the final layer's full AdaLN weights once (26 GB, 32 threads: ~15 s), adding LMS Sharp's curve branch
(strength x B A c(t), on the table at t, as `sharp.py`'s CurveLinear does). The net then takes its tables from there.
Without HyperFlow nothing changes; LMS Sharp's 50 AdaLN branches (A [64, 8]) were dropped by our LoRA loader before and
are now applied in every H3 clip that names the LoRA (`LoraSet::adaln`, `Step::new_with`).

## Fitting the 2K sequence on one 32 GB card

124 frames at 2560x1440 are 133,200 target rows + 37,296 reference rows + ~6.3k text and vision tokens: ~177k tokens.
The int8 blocks (19.5 GiB) plus that sequence do not fit; the release shards it over 8 GPUs. Here:

- **the blocks stream** from the int8 file (`H3_RESIDENT_BLOCKS=0` on the card: `Blocks::load_streamed`; each block's
  four matrices read and sent to one card copy as it runs, ~0.4 GB; the page cache keeps the file hot);
- **the LoRAs stay in host memory** (HyperFlow + LMS + IC: 882 layers, 9.3 GiB as bf16; `LoraSet::load_to(.., on_host)`)
  and go to the card a block at a time; q, k, v of the Diffusers-layout LoRAs are three LoRAs on qkv's outputs;
- **the MLP runs in row chunks** (16,384 rows: `Scratch::chunked`, `Conditions::mlp_rows`; the job checks the card has room for the sequence first) so its two widest
  buffers are not per token.

The whole clip is one sequence, as in the release (no windows, no seams).

## Parity (stage by stage, same inputs)

| Stage | Against | Result |
|---|---|---|
| int8 base = the release's base | `transformer_ref` block 0 `to_q` (range-fetched) | rel 0.0088 (int8 rounding) |
| time tables (HyperFlow two-time + LMS curve), blocks 0 and 49 and the final layer, 8 (t, r) points | `hyperflow.TwoTimeEmbedder` + Diffusers `TimestepEmbedding` + the full projection + `sharp.CurveLinear` (reference/regen2k/time_ref.py) | rel 7e-7, worst point 1e-6 |
| text encoder input: token ids and modality tags of `<Video 1>` + task prompt | Diffusers' `MiniMaxH3Ref2VATextEncoderStep` presentation with the release's processor (presentation_ref.py) | identical but for the placeholder id: the release's `<\|video_pad\|>`, ours `<\|image_pad\|>` (ComfyUI's); the encoder replaces both by the vision rows |
| one denoiser call at sigma 0.834942 (448x256 draft, 896x512 target, 22 frames: 4,194 tokens), everything on: HyperFlow merged + two-time embedder, LMS merged + curve branches, IC LoRA unmerged, Diffusers' Ref2VA packed layout and per-row timesteps | Diffusers' `MiniMaxH3Transformer3DModel` with the release's `infer/ref2va_ic` code, CPU float32, the same int8 weights un-rotated (reference/regen2k/dit_ref.py) | velocity cosine 0.999877, rel 0.0157; worst frame 0.999853; the denoised estimate rel 0.0195. (The engine's velocity is the release's negated, as everywhere in our H3.) |
| latent upscaler (target dimensions, whole clip) | the author's node as the release runs it (upscale_ref.py) | cosine 0.999930, rel 0.0119 (both bfloat16; at the check's x2, 28x16 -> 56x32 latents, 7 frames) |

H3 is not bit-reproducible (two H3 runs: latents cosine 0.996), so the clip itself is judged by looking (below).

## Timing and memory (hardac1)

One 5.17 s clip (124 frames), draft 1344x768 -> 2560x1440, up3, on the **B65** (gpu1; the B70 was busy with other
work), `nextsycl video job regen2k --here` in the build copy:

| Stage | Seconds |
|---|---|
| draft read + H3 VAE encode (124 frames at 1344x768, posterior draw) | 98.5 |
| latent upscaler, 84x48 -> 160x90, whole clip | 6.9 |
| vision tower (6 blocks, 6,048 tokens) + Q4_K_M text encoder (6,197 tokens; layers from the file into pinned memory) | 6.8 + 149.7 |
| LoRAs to host memory (HyperFlow + LMS + IC, 882 layers, 9.31 GiB) | 20.2 |
| HyperFlow's time tables (8 points, 26 GB of projections) | 15.6 |
| denoiser, 177,107 tokens, 3 steps | 1880.5 (632.7, 627.5, 620.4) |
| H3 video decoder, 124 frames at 2560x1440 (182 tile batches) | 124.3 |
| mux the draft's sound | <1 |
| **whole clip** | **2310.9 s (38.5 min) for 5.17 s of video** |

Card memory (B65, 32 GiB): the engine's own peak 26.2 GiB; the driver's view (`/run/gpustat.json`, sampled every
second by the runner) 28,835 MB peak, so ~3.9 GB free at the worst moment (the 1.5 GiB guard never fired). The job
checks the room first: "177107 tokens need about 23.9 GiB on the card, 30.4 GiB free". Only one card is used (the
B70 holds nothing of this run). Host: MemAvailable fell to 23.7 GB at the lowest (the TE's 14.4 GiB pinned copy + 9.3
GiB of LoRAs).

Per denoiser block ~12.6 s at 177k tokens; the streaming itself (block read + upload, LoRA upload) is ~0.9 s a block
(measured at 4k tokens: one call 46.5 s for 50 blocks), i.e. ~7% of a 2K step. The B70 is the faster card and was not
measured.

## Looking at it

Draft: our own H3 (minimax-h3 int8 FL2VA, 1344x768, 124 frames, 8 steps, seed 7; 232 s on the B70, 27.6 GB peak): a
woman in a yellow raincoat on a harbour pier at dusk speaking to camera, a "HARBOUR CAFE" sign, boats, lamps.
Regenerated with the caption above. Frames 10, 60 and 115 extracted from both (draft enlarged with bicubic to 2560x1440;
reference/regen2k/look.py), viewed side by side and as full-resolution crops of the face and the sign:

- composition, camera, boats, lamps, the woman's pose and mouth shape at each frame: the same as the draft;
- the face: visibly sharper - skin texture, wrinkles, eyelashes, separate hair strands at the hairline; the identity is
  kept (slightly more contrast, the eyes a touch darker);
- the lettering: "HARBOUR CAFE" kept letter for letter, edges crisp where the draft's are soft;
- sharpness (variance of the luma Laplacian), draft enlarged vs regen: frame 10 1.9 vs 6.2, frame 60 2.1 vs 7.2,
  frame 115 1.8 vs 6.2; on the face crop 4.8 vs 17.6, 5.3 vs 19.2, 3.9 vs 14.1; on the sign 3.0 vs 11.1. Mean absolute
  difference to the enlarged draft 3.5-4.0 (of 255): a refinement, not a new clip;
- sound: the regenerated .mp4 has an AAC track of 5.15 s (the draft's, muxed back), the video 124 frames at 24 fps.

Frames on hardac1: `~/src/regen2k/look1/` (pair_N.png, crop_N.png), `~/src/regen2k/look1sign/`; clips
`~/src/regen2k/draft1.mp4`, `~/src/regen2k/regen1.mp4`.

## Drafts of any length, in production (2026-11-16)

The first live production job (a platform clip, 864x480) was stopped by the studio's watchdog and had been cut to 124
frames. Since e6f0d14:

- **The whole draft is regenerated.** 4.7 to 15.1 s, landscape 16:9 within 10%, any size (cover-fitted to 1344x768;
  864x480 works). Past 124 frames it goes in overlapping windows of the model's own clip (`regen::windows`: 37 latent
  frames each, starting on 17-frame groups, spread evenly, the overlap - at least 2 latent frames - kept from the window
  before as masked frames). Each window has its own part of the draft as `<Video 1>`; the joined latents are decoded
  once, cut to the draft's length, and the draft's sound is muxed back. Anything else is a 400 from video.generate
  naming the limit (the studio probes the draft). The kernels' scratch is given back between windows (`Device::trim`,
  `nsd_trim`): without it the second window found 22.1 GiB free, not 30.4.
- **The watchdog sees progress.** It stopped a clip after 20 min without a new log line; it now also counts the engine's
  progress (every block, ~13 s at 2K). The job logs each step as it ends.
- **Status:** the studio's item carries the draft's frames, its windows and all their steps (`sampling 4/6` over two
  windows), and a regen ETA: the measured figures until it samples, then the blocks' pace.
- **ETA at the door:** a regen request runs on `minimax-h3-regen2k-ic` (its own history, not H3's 45 s clips); with no
  history the answer's eta is a first figure from these measurements (`basis: "estimate"`).

End to end as production runs it (2026-11-16, B65, under gpu_lock): a test video daemon (`regen2k-e2e-nextsycl-video`,
GPU 1 only) and a test studio (127.0.0.1:8197) from the build copy, with their own sockets and out dir, and
video.generate's body `{"prompt": <the clip's own prompt>, "regen2k": "h3_20261113_150714.mp4", "regen_sampler":
"up3", "regen_resolution": "2k", "queue": true}` sent to the studio. The production clip that had failed is **864x480
and 7.71 s (185 frames)**; its record says 10.125 s, but that is the length asked for, not the length made.

- First try: window 1 went through without the watchdog firing; window 2 was refused by the memory check (22.1 GiB
  free). The kernels' scratch had not been given back, which `Device::trim` now does.
- Second try: **done in 4,395 s (73 min) from the request**: two windows, 6 steps of 177k tokens (628, 630, 607,
  599, 581, 581 s), the text encoder 162 s a window, the 2K decode of 57 latent frames 196 s. The card peak was
  28,668 MB (driver); the second window started on a trimmed card.
- The studio showed `starting` -> `text encoder` -> `conditioning done` -> `sampling 1/6` ... `sampling 6/6` ->
  `sampling done, decoding` -> `done`, with pct 2 -> 100. Its ETA at the request was 4,394 s (the measured figure), then
  from the blocks' pace 3,843 s at 13 min, 2,154 s at 37 min, 229 s at 69 min.
- Frames 20, 90 and 150 against the draft enlarged to 2560x1440: the same shot, pose and mouth. The face, hair, the
  microphone's mesh and the jacket's embroidery are sharper. Laplacian variance, whole frame 2.1-2.2 -> 8.4-8.6; face
  crop 2.2-3.3 -> 9.8-13.0. The change from frame to frame follows the draft's all through, including frames 68-80
  where window 2 takes over: no seam.
- The sound is the draft's (7.71 s). **But the picture came out at 181 frames**: muxing with `-shortest` and the
  picture copied ends a few frames early. Fixed after the run (`mux_draft_sound`: the output cut at the clip's own
  length). The ffmpeg command was checked on the same 185-frame clip (185 frames out, sound 7.708 s); the 73-min run
  was not repeated.

Files on hardac1: `~/src/regen2k/e2e/out/h3_20261116_082306.mp4`, frames `~/src/regen2k/e2e/look/`, the
studio's progress `~/src/regen2k/e2e/run.log`.

## The fast decoder (2026-11-17)

The release decodes with `vae_fast/fast_h3_vae_v2_24l.safetensors` (`infer/ref2va_ic/fast_vae/decoder.py`, from
NVIDIA's StreamRender-H3): the H3 decoder's transformer pruned to 24 of its 36 layers and distilled to decode a whole
clip in one piece - temporal windows of 5 latent frames (a native block), spatial windows of 16x16 tokens, both shifted
on odd layers (by 2 latent frames, by 8 tokens), and 5 global tokens a block (the 4 registers + a zero token) that attend
to their whole block while every window attends to the global tokens of the blocks it overlaps; window-local rotary ids;
no tiles, no cross-fades. In nextsycl it is `crates/h3/src/fastvae.rs` with the H3 video engine's first silo
(`kernels/video/h3/silo`, `dist/silo/libnextsycl-h3.so`, bound in `video/h3/src/silo.rs`):

- the clip's residual stream stays whole on the card (float32, a row a token); each layer runs **a temporal window at a
  time** (72,000 tokens at 2K): its norm, qkv, the window attention, the projection and the MLP (16,384 rows at a time),
  written back in place. A window's keys are its own tokens and its blocks' global rows, both made before it changes
  anything, so this is the release's layer exactly, in ~5 GB of buffers whatever the length (the stream: 8 KB a token,
  4.4 GB for 124 frames, 12.6 GB for 362);
- **window attention**: the windows' rows gathered (an indexed row copy) into variable-length sequences - every window
  its tokens, its keys those plus 5 or 10 global rows - and ARK's bfloat16 prefill kernel on sycl-tla with cu_seqlens
  (`ns_video_h3_silo_varlen`; upstream runs FlashAttention-4's varlen form the same way: no padding, no mask);
- **global tokens**: their attention over the whole block takes the block's keys a temporal window at a time into a
  running softmax (max, sum, weighted values in float32; 256 keys a work-group, `ns_video_h3_silo_glob`), finished with
  the block's own 5 keys after the last window; the last layer's are skipped (never read), as upstream does;
- bfloat16 matrices and activations (the weights converted from the release's float32 at load, 6.0 GiB read in ~3.4 s),
  the residual in float32, the input map (de-normalization, the 1x1x1 convolution, the embedding) folded into one
  [2048, 24] product and rounded to bfloat16 as the release does, the head in float32 (the release's reference engine).

Parity (`reference/regen2k/fastvae_ref.py`: the release's `NonCausalViTDecoder`, CPU, float32, crops of a real 2K
clip's latents, through the decode job's `check`):

| Latents (T x H x W) | What it covers | PSNR | cosine |
|---|---|---|---|
| 7 x 20 x 36 (22 frames) | 2 blocks, shifted windows across them | 76.2 dB | 1.000000 (rel 5.0e-4) |
| 12 x 33 x 40 (39 frames), from latent frame 5 | 3 blocks, partial windows on every side | 77.8 dB | 1.000000 (rel 3.1e-4) |

A whole clip (regen1's 37 x 90 x 160 latents, 124 frames at 2560x1440) against our H3 video decoder (`H3_DECODE_FRAMES`
dumps, SSIM on luma, Gaussian 11x11): **PSNR 44.4 dB (min 43.7), SSIM 0.9944 (min 0.9923)** over 34 frames - the
distillation's own distance from the full decoder (upstream's target is 30 dB+). Frames 10, 60, 115 side by side and as
face crops (`~/src/regenvae/look1/`): no visible difference, no window or block seams; the sign's lettering and the
face's texture the same.

| Decoder, 124 frames at 2560x1440 (regen1's latents) | B65 | B70 | Card peak (driver) |
|---|---|---|---|
| H3 video decoder (182 tile batches) | 127.4 s | 90.2 s | 5.9 GB |
| fast decoder | 32.4 s (+3.3 load) | 22.7 s | 11.5 GB |

(The B70's fast decode gives the same frames: PSNR 44.4 dB against the H3 decoder there too. 192 frames, B65: 50.3 s;
362 frames, B65: 91.5 s - linear in the length.)

It is the card's default (`H3_REGEN_VAE: 'fast'`); `H3_REGEN_VAE: 'standard'` on the card (or `--regen_vae standard`
on an engine job; video.generate does not take it - the card decides) keeps the H3 video decoder, and the job falls back to it, saying so, when the silo is not built or the card lacks the room.

## Drafts of three lengths (2026-11-17)

All on the **B65** (gpu1), up3, seed 7, `nextsycl video job regen2k --here` in a build copy under `gpu_lock 1`, the card's
files. 5 s and 15 s: 1344x768 H3 drafts with sound from the studio's out dir (a man seated in a uniform on a ship's
bridge, a book on his knee, talking to camera; the same prompt at two lengths), the **fast decoder**. 7.7 s: the
2026-11-16 production test above (864x480, end to end through a test studio, the **H3 video decoder**), plus its saved
latents through the fast decoder alone.

| | 5.17 s | 7.71 s | 15.08 s |
|---|---|---|---|
| draft frames in / out | 124 / 124 | 185 / 185 (decoded 192, cut) | 362 / 362 |
| windows x steps | 1 x 3 | 2 x 3 | 3 x 3 |
| **request to finished clip** | **2,294 s** (job 2,258) | **4,395 s** (studio) | **6,609 s** (job 6,573) |
| draft read + VAE encode | 98.7 | - | 268.1 |
| LoRAs to host + time tables | 19.8 + 18.1 | - | 19.7 + 17.5 |
| per window: upscale + vision + text encoder | 6.8 + 6.7 + 164.2 | 162 (TE) | 6.8 + 6.1-6.6 + 165-166 |
| steps (s) | 634.7, 630.6, 632.4 | 628, 630, 607, 599, 581, 581 | 637, 638, 636 / 638, 636, 635 / 629, 584, 583 |
| decode | fast 31.2 (+9.5 load, mp4: 42.1) | H3 decoder 196; fast 50.3 (62.8 with load and mp4) | fast 91.5 (104.8 with load and mp4) |
| card peak (driver; engine) | 28,678 MB; 26.1 GiB | 28,668 MB | 28,696 MB; 26.1 GiB |
| sound | the draft's AAC copied: its first 162 of 163 packets, identical (pts, size, md5); the last lies past the picture's 5.17 s | 7.71 s | first 472 of 473 packets identical |

The job's figures exclude ~36 s of container start in `job --here` (in the request column). The card peak is the
sampling's (177k tokens); the fast decode peaks at 11.5 GB (124 frames) and 13.7 GB (192), on a context of its own
after the sampling's scratch is given back.

Looking: frames 10/60/115 (5 s) and 60/121/180/240/330 (15 s) against the draft enlarged to 2560x1440
(`~/src/regenvae/look5`, `look15`): the same shot, pose and mouth at every frame; fabric, skin and the insignia sharper.
Frame-to-frame change of the regen against the draft's (`ref/seam.py`, 640x360 luma): median ratio 0.97 (5 s), 0.96
(15 s); the largest excess is at the 15 s clip's second window seam (frames 242-246: 0.46-0.82 vs the draft's
0.29-0.59 of 255) - not visible in the frames around it. Both decoders show a slightly larger change at the 17-frame
block boundaries (regen1's latents: +19% over the rest with the H3 decoder, +31% with the fast one; ~0.05 of 255).

**The ETA's first estimate** (`regen_estimate_s`, studio/runner.rs), before / after recalibration:

| Draft | Old: 40 + W (160 + 627 S) + 1.8 F | Measured (job) | New: 40 + W (180 + 625 S) + 1.05 F |
|---|---|---|---|
| 5.17 s (F 124, W 1, S 3) | 40 + 2,041 + 223 = 2,304 (+2.0%) | 2,258 | 40 + 2,055 + 130 = 2,225 (-1.5%) |
| 7.71 s (F 185, W 2), H3 decoder | 40 + 4,082 + 333 = 4,455 (+1.4%) | 4,395 | (fast decoder: 40 + 4,110 + 194 = 4,344) |
| 15.08 s (F 362, W 3, S 3) | 40 + 6,123 + 652 = 6,815 (+3.7%) | 6,573 | 40 + 6,165 + 380 = 6,585 (+0.2%) |

(F frames, W windows, S steps a window.) The old figure was within 4% but carried the H3 decoder's ~1.1 s a frame; with
the fast decoder the frame term is the encode (0.74-0.80 s a frame) and the decode with its load (0.29-0.34): 1.05. A
window's text encoder, vision tower and upscaler are ~180 s, a step 625 s (the mean of 12). The running ETA's decode term
is now 0.3 s a frame (was 1.1), its text encoder 180 s a window.

## Use

```sh
nextsycl models pull minimax-h3-regen2k-ic        # IC LoRA, HyperFlow, silence, 26 GB of AdaLN tensors; the rest is shared
nextsycl video job regen2k --here --model minimax-h3-regen2k-ic --draft draft.mp4 --caption "<the draft's prompt>" --out out.mp4
    [--sampler up3|up4|hf8] [--resolution 2k|1080p] [--sharp 1] [--seed N] [--regen_vae fast|standard]
```

Front door / studio: `video.generate {"prompt": "<the draft's prompt>", "regen2k": "<clip>.mp4" | "prev", "regen_sampler": "up3",
"regen_resolution": "2k", "sharp": 1}` (feature `video.regen2k`; the model in NS_VIDEO_ENGINES). The studio's clip
panel has "Regenerate 2K".

## Open

- Speed: the streamed blocks and host LoRAs cost about 45 s per step (~7%); double-buffered uploads from pinned memory,
  or merging HyperFlow + LMS + IC into the int8 weights once (requantized), would remove most of it. The time tables
  include the audio stream's timesteps, which no row uses (pinned silence): a few wasted points.
- 1080p (computed at 1088 rows, centre 1080 kept) and up4 / hf8 are wired but were not run on the box.
- Host memory: ~10 GiB of LoRAs + the text encoder's pinned copy (14 GiB) while a clip runs; the backplane gates VRAM,
  not host RAM (`ram_gib` on the card).
