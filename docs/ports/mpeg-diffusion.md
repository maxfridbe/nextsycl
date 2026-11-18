# MPEG-Diffusion: half resolution, then full resolution on the subject (MiniMax H3)

MPEG-Diffusion ("MPEG-Diffusion: Mixed-Resolution Video Generation", Dennis Menn, Tianpei Gu, Guande He, Aryan V S,
Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu; arXiv 2611.12069; the authors published no
code) renders the region of a clip that matters at full resolution and the rest at half. This is a port of
**scrya-com/mpeg-diffusion** (an independent port of the paper to MiniMax H3; code at 8e91323, `lot/` Apache-2.0) and of
its weights **johndpope/MiniMax-H3-MPEG-Diffusion** (at e2cc718: the fitted K/V maps and region head v3, MiniMax H3
Community License). Upstream calls it "research code, it needs training"; so is this.

In nextsycl it is the H3 engine's generate job with the two-track sampler on (`crates/h3/src/mpeg.rs`), the
`minimax-h3-mpeg` model card (`env: H3_MPEG: '1'`), and through the front door simply that model id in
`video.generate`. No new request option: a request may send `"mpeg": false` to that model to run it at full resolution.

## What runs

Upstream's quality recipe (`two_track_h3.py render --preset quality`), on our H3:

```text
text -> Qwen3-VL 32B (Q4_K_M) -> the text refiner                                   (as every H3 clip)
noise at half resolution [24, T, H/32, W/32] (and the sound's)
steps 0..15 of 20 (H3's shift-12 schedule): the unmodified model on the whole clip at half resolution, with its sound
step 15: block 25's half-res video states -> region head s = w . LN(h) + b (5,376 weights) -> spatial Gaussian (sigma 1,
         reflected) + [1, 2, 1] / 4 across latent frames -> M = s > m + 0.5 (p99 - m) over the whole clip
switch:  start = latent x2 of the clean half-res estimate; inside M's bounding box (all frames, + 2 latent px) the VAE
         round trip instead: decode, bicubic x2, encode. hr = (1 - 0.75) start + 0.75 fresh noise (sigma 0.75)
6 steps, sigma 0.75 -> 0 linear: one sequence [text | audio | half-res video (all) | full-res video (inside M only)]
         text, audio, half-res queries: keys text + audio + half-res (exactly the base model at half resolution)
         full-res queries: keys text + audio + the full-res grid: their own inside M; outside M every half-res token's
         normalized, unrotated key and value expanded to its 2x2 cells by the fitted map (k + D_c k, rank 128 a block),
         rotated at the cells' own positions
end:     full-res latent = hr inside M, latent x2 of the half-res track outside -> H3 decode at full size;
         the half-res track decoded at half size, bicubic x2; pasted under the full-res frames with a 4-px ramp inside
         M's edge (bilinear x16 to pixels, each latent frame to its pixel frames); the sound decoded as usual
```

H3's rotary positions are normalised by the frame's area, so the two grids share one coordinate frame: a half-res token
sits on the top-left cell of its 2x2 (both upstream's convention and our `layout.rs`'s, unchanged).

### How the attention is assembled on the card

The block's qkv linear, norms, rotation, MLP and AdaLN are the stock ones over the packed sequence; only the attention
is split (`dit.rs` `SplitAttention`, `mpeg::TwoTrack`):

1. before q and k are normalized: the raw k and v rows of the half-res tokens outside M are gathered (the silo's
   indexed row copy), k gets its per-head RMS norm (no rotation);
2. for each of the four cells c: copy, then `+= x D_c^T` (the shared `nsd_linear_acc`, one [rows x heads, 128] x
   [128, 128] product a cell; D = down . up folded once on the host in float64, bfloat16 on the card) into the key /
   value set; the expanded keys rotated at their cells' positions (`nsd_rope`);
3. after the stock rotation: the prefix's (text, audio) and the full-res rows' keys and values gathered next to them;
4. attention 1: rows [0, split) over [0, split) with the stock `nsd_attention` (SageAttention past 8,192 tokens, as in
   every H3 step); attention 2: the full-res rows over their key set with `nsd_attention_qk` (oneDNN's fused kernel).

Key order does not matter to attention, so the full-res queries' key set is [prefix | own rows | cell 0 of every
token outside M | cell 1 | ...] rather than the raster grid upstream builds: the same set, the same positions.

The one new kernel is `ns_video_h3_silo_gather` (kernels/video/h3/silo/tla/gather.cpp, `dist/silo/libnextsycl-h3.so`:
rows picked by an index from a strided buffer, 16 bytes a work-item, an exact copy). It shares the H3 video silo with
Open-H3-Regenerate-2K's fast decoder: that silo is sycl-tla sources alone, built as one library with sycl-tla's flags,
so the gather takes the GPU's SYCL queue (`Gpu::queue`) rather than the shared part's `ns_gpu` and reports through
`ns_video_h3_silo_error`; video/h3/src/silo.rs binds both. Without the silo the same rows are copied one
`nsd_copy` a row (the same bits, slower). The engine opens the silo only with H3_MPEG=1, and then puts its device on the
core's queue of the card (`Device::on_queue`, as the image engines on H3 do) so the silo's work is ordered with it;
without H3_MPEG nothing changes for H3.

### Decisions (where this port differs from upstream, or chooses)

- **Sound.** Upstream feeds H3 silent audio noise every step and throws the audio away. H3 here makes the clip's sound:
  the audio rows are denoised with the half-res track (text, audio and half-res queries are exactly the base model at
  half resolution) through all 22 calls, the two-track tail on the same linear sigmas.
- **The Level-of-Token layout** (full-size tokens inside, 2x2 on a one-token ring, 4x4 elsewhere) is what upstream's
  `layout` command draws for a region and what its separate Level-of-Token port (`mixed_res_h3.py`, "smeared" by
  upstream's own account) runs on. The two-track sampler, the one upstream measured and recommends, does not use it.
  It is ported (`mpeg::region_layout`, checked exactly against upstream) and each clip's log and report give its token
  count; it does not drive the sampler.
- **No region**: upstream finishes the half-res schedule and upscales with the VAE; here the half-res track finishes,
  is decoded at half size and enlarged (bicubic) - the same path as the paste with an empty region.
- **Noise.** Our H3 draws its noise from the seed on the CPU (`noise::clip_noise`), the half-res track's at the half-res
  shape, the switch's fresh noise from seed + 2 (upstream: CUDA generators seeded seed, seed + 1, seed + 2). So a
  two-track clip and a full-res clip of the same seed start from different noise and compose different pictures -
  upstream says the same of its own ("Full res and two-track compose different pictures from the same seed").
- **Requests it takes**: text (and the effect embeddings) only - no keyframes, guides, references, sound inputs,
  ControlNet, masked runs, refine, other samplers or knots: the job fails at once, naming the limit. Upstream's node takes text
  only too. Canvas sides multiples of 64.
- **Settings** (card env or job keys, docs/environment.md): H3_MPEG_SWITCH 16, H3_MPEG_HR_STEPS 6, H3_MPEG_ALPHA 0.5,
  H3_MPEG_FLOOR none, H3_MPEG_FEATHER 4 - upstream's quality preset.
- **Turbo**: upstream's "Turbo LoRA (v4), 8 steps" is an 8-step distillation LoRA for the FL2VA base (Comfy-Org's
  repository has `minimax_h3_fl2v_turbo_8step_v1.0`); our `minimax-h3-ref2v-turbo-lora` is Comfy-Org's 4-step **Ref2V**
  LoRA for the Ref2VA denoiser - not the same LoRA, and not for this base. No Turbo card (untested here).
- **Q6_K / Q4_K_M**: the maps were fitted on the int8 checkpoint's keys. The GGUF denoisers are the same model
  re-quantized (their keys differ by quantization noise), so the maps should transfer, but that is untested: no cards
  for them.

## Files

| Part | Where | Size | sha256 |
|---|---|---|---|
| K/V up-maps (`down.l{i}_2x2_{k\|v}` [128, 128], `up.*` [128, 512], 50 blocks, float32) | HF johndpope/MiniMax-H3-MPEG-Diffusion @ e2cc71831ff468374024d78e2b53d6607daab84d `kv_maps/up_maps_tt.safetensors` | 32,784,896 | a105eae0aa4f30d9a4cf1c861e7d78e8efd6393957ed6cfde09e34378984c648 |
| region head v3 (`score.weight` [1, 5376], `score.bias`; block 25, switch 18 in its metadata, held-out mIoU 0.528) | same, `region_head_v3/region_head_tt.safetensors` | 21,780 | 9a43b94c84b12c422cbf0d303c2145c87f0ed11d7ecd0e74b8bd5fa3a9715984 |
| the base: denoiser int8 FL2VA, VAE fp16, audio VAE, text encoder | the `minimax-h3` card's files (the same files upstream names) | | |

On hardac1: `/var/mnt/4TBSSD/nextsycl-models/minimax-h3-mpeg/` (downloaded 2026-11-17, checksums as above).

## Parity with upstream's code (same inputs)

Upstream's own Python driven by `reference/mpeg-diffusion/tt_ref.py` (its checkout imported, nothing copied; CPU,
float32; the int8 checkpoint dequantized - W = (q s) H per 256 group - one block in memory at a time). The inputs are
the engine's: a generate job with `"mpeg_dump"` writes what its first two-track call saw and gave (the refined text,
both tracks' x, the sound as the network sees it, the plan, block 0's raw qkv and attention output, the region head's
block-25 states and scores). Dump: a 22-frame 1344x768 clip (`p1`, seed 42; 7 latent frames, 1,764 half-res tokens,
region 17.1% = 302 tokens -> 1,208 full-res rows; the two-track sequence 3,128 rows; the full-res queries over 7,212
keys), sigma 0.75 (the first two-track step), on the B65.

| Part | Ours | Upstream | Result |
|---|---|---|---|
| region head scores, the same block-25 features (1,764 tokens x 5,376) | `RegionHead::scores` (float64 sums) | `mixed_res.RegionHead` | rel 3.2e-7, max abs 7.6e-6 |
| the region mask from those scores (smoothing, threshold) | `region_mask` | `mixed_res.region_mask` | identical, 0 of 1,764 cells differ (from upstream's own scores too) |
| the token layout: full-res rows (`hr_index`), `cell_order`, `grid_source` | `mpeg::Plan` | `TwoTrackPlan` | identical (the dump, and 3 fixture cases in `cargo test`) |
| RoPE positions of the packed sequence (3,128 rows; fixtures: 453-8,133 rows) | `Plan::positions` on `layout.rs` | `build_position_grid` of the half- and full-res layouts | identical in float32 (what the rotation reads); in float64 a quarter differ in the last bit (max 1.4e-14: the time spans summed in another order) |
| the Level-of-Token layout the region implies (1x1 / 2x2 ring / 4x4) | `region_layout` | `mixed_res.region_layout` | identical rectangles (3 fixture cases) |
| the K/V up-map on the same keys (block 0, the 1,462 half-res keys outside M, normalized, unrotated; 42 heads) | `UpMaps::expand_host` (D folded in float64) | `UpMaps.expand` float32 | rel 1.2e-7 (upstream's own bfloat16 path against its float32: rel 2.5e-3) |
| block 0's dual-resolution attention on the engine's own raw qkv (3,128 rows) | the card: gather, map products in bfloat16, `nsd_attention` + `nsd_attention_qk` | `two_track_attention` float32 | cos 0.9999975, rel 2.3e-3 (half-res rows 2.3e-3, full-res rows 2.1e-3) |
| the whole two-track network call (50 blocks, int8 GEMMs and bfloat16 activations on the card) | `Denoiser::net_two_track` | `two_track_h3.two_track_forward` float32 | half-res velocity cos 0.999934, rel 0.0115; full-res velocity inside M cos 0.999867, rel 0.0163 (per latent frame 0.99977-0.99993); outside M both exactly 0 |

The whole call's 1.6% is the int8 activation quantization and bfloat16 of our H3 against float32 weights-only
dequantization in the reference - the same size as the Regenerate-2K port's one-call check (rel 0.0157) and as our H3
against ComfyUI. Upstream's CPU call took 290 s (+ 61 s dequantizing blocks; 8 threads).

Upstream's own CPU test `lot/test_lot.py` passes in the reference's environment ("ok 33", torch 2.15 CPU): the code
the checks drove is upstream's as it tests it.

## Two-track against our own full-resolution H3

The `minimax-h3-mpeg` card against the `minimax-h3` card (the same int8 denoiser, text encoder and VAE): the same
prompt, seed (42), steps (20) and size (1344x768), 73 frames (3.04 s) and 141 frames (5.88 s; the "145 frames" of
upstream's table is 141 there too: its `latent_frames` snaps 145 down onto 17k + 5, ours rounds to the same 141).
Prompts (`~/src/mpeg/prompts/` on hardac1): **p1** a red-haired woman speaking to camera on a city street at dusk,
**p2** two old fishermen mending a net on a dock, laughing, **p3** an aerial mountain valley at sunrise (landscape, no
people), **p4** a red sports car racing along a coastal road, the camera tracking it (fast motion).

Measured by `reference/mpeg-diffusion/metrics.py` (frames from the mp4s; the region is the run's own, its job report's
`mpeg.region_mask`, a half-res token = 64 x 64 px over the pixel frames its latent frame decodes to). PSNR over RGB,
SSIM on luma (Gaussian 11 x 11, sigma 1.5, every 2nd frame), LPIPS AlexNet from the `lpips` pip package on CPU in its
own venv (every 4th frame; inside the region: its spatial map averaged there). Flicker: the mean absolute
frame-to-frame change of luma (0-255) of each clip. Sharpness: the variance of the luma Laplacian (every 4th frame).
Pairs are written "full res / two-track".

| Prompt | Frames | Region | PSNR dB frame / region | SSIM frame / region | LPIPS frame / region | Flicker full / tt | Flicker in region full / tt | Sharpness in region full / tt | Sharpness outside full / tt |
|---|---|---|---|---|---|---|---|---|---|
| p1 | 73 | 16.5% | 9.9 / 10.7 | 0.411 / 0.363 | 0.705 / 0.631 | 4.63 / 5.53 | 6.75 / 7.73 | 31 / 24 | 14 / 16 |
| p1 | 141 | 18.5% | 10.7 / 10.5 | 0.401 / 0.320 | 0.698 / 0.655 | 4.67 / 3.09 | 7.38 / 5.43 | 74 / 29 | 48 / 21 |
| p2 | 73 | 20.8% | 10.5 / 13.2 | 0.357 / 0.347 | 0.663 / 0.604 | 2.76 / 3.04 | 5.06 / 3.10 | 134 / 22 | 168 / 42 |
| p2 | 141 | 26.9% | 11.1 / 10.6 | 0.288 / 0.258 | 0.725 / 0.698 | 4.70 / 3.46 | 6.11 / 4.75 | 172 / 59 | 162 / 59 |
| p3 | 73 | 16.6% | 13.1 / 20.8 | 0.655 / 0.546 | 0.506 / 0.360 | 0.61 / 0.41 | 0.94 / 0.67 | 17 / 3 | 36 / 11 |
| p3 | 141 | 14.2% | 13.4 / 19.9 | 0.747 / 0.755 | 0.535 / 0.379 | 0.47 / 0.46 | 0.83 / 0.83 | 18 / 5 | 35 / 12 |
| p4 | 73 | 14.4% | 10.5 / 9.8 | 0.544 / 0.324 | 0.672 / 0.730 | 12.85 / 11.91 | 14.81 / 8.73 | 245 / 46 | 91 / 29 |
| p4 | 141 | 14.9% | 11.4 / 10.3 | 0.564 / 0.342 | 0.632 / 0.753 | 6.85 / 9.21 | 10.75 / 9.42 | 201 / 32 | 71 / 24 |

**Read the PSNR / SSIM / LPIPS columns as "a different picture", not as error.** The two runs start from different
noise (the half-res track draws its noise at the half-res shape), so they compose different shots of the same prompt -
upstream says the same of its own. That makes frame-by-frame agreement meaningless across the two; it is reported
because it was asked for. What does carry across is the flicker (the two-track clips move about as much as the full-res
ones: 0.66-1.34x, no extra frame-to-frame noise) and the sharpness: **inside the region the two-track clip is 1.3-6.3x less
sharp than a full-res render on every pair** (p2 73 frames: 22 against 134; p4: 46 against 245).

### The same noise: two-track against full-resolution tokens everywhere

To compare like with like, the same two-track run with the region forced to the whole frame (`mpeg_alpha -1000`, "all":
the same half-res track, the same switch, then full-res tokens on every cell - the two-track method's own upper bound)
at 73 frames on the B70. Pairs "all / two-track":

| Prompt | Region | PSNR dB frame / region | SSIM frame / region | LPIPS frame / region | Flicker all / tt | Sharpness in region all / tt | Sharpness outside all / tt |
|---|---|---|---|---|---|---|---|
| p1 | 16.5% | 17.2 / 22.4 | 0.727 / 0.782 | 0.325 / 0.191 | 4.55 / 5.53 | 28 / 24 | 13 / 16 |
| p2 | 20.8% | 18.3 / 23.1 | 0.671 / 0.785 | 0.318 / 0.204 | 2.31 / 3.04 | 25 / 22 | 29 / 42 |
| p3 | 16.6% | 28.0 / 32.2 | 0.932 / 0.942 | 0.118 / 0.083 | 0.34 / 0.41 | 3 / 3 | 7 / 11 |
| p4 | 14.4% | 22.9 / 23.6 | 0.830 / 0.788 | 0.173 / 0.142 | 9.01 / 11.91 | 62 / 46 | 20 / 29 |

The same shot, frame for frame (the people and objects in the same places). Inside the region the two-track clip
agrees with "all" at 22-32 dB, LPIPS 0.08-0.20; its region is 0.73-0.89x as sharp as all's. **"All" is as soft as
the two-track region**: the softness against a full-res render is the method's - a switch at sigma 0.75 from an
enlarged half-res estimate leaves 6 steps to draw detail, too few untrained (upstream: "faces are only as good as the
LR track draws them"; the paper fine-tunes the whole DiT so the full-res track can redraw) - not this port's region,
maps or paste.

### Looking at the frames

Frames 10, 36, 70 (and 120) of every pair, side by side with the region outlined, and 2x crops (hardac1
`~/src/mpeg/out/`, the PNGs made locally by metrics.py `--png`):

- **p1 woman**: the region is her face and hair (16-19%). The face is clean - eyes, teeth, skin - and steady over the
  clip; at 2x the hair inside the region has separate strands, outside it the same hair is softer but continuous: the
  edge of the region shows as a change of sharpness, **no seam or step**. A full-res render of the same prompt is
  sharper still (pores, single hairs). One isolated one-token region on a background man's head (frame 36, 73 frames)
  shows a **visible rectangular edge** in his grey hair: the paste's 4-px ramp cannot hide a 64-px island.
- **p2 fishermen**: the region covers both men's heads and upper bodies (21-27%). Faces read well at full size; at 2x
  the beards and hair are smooth texture where the full-res render has strands. The boats, the net and the water
  outside are the enlarged half-res track: soft, no blocking. No seam at the region's edge.
- **p3 valley (no person)**: the head finds no subject and picks scattered tokens along the bottom edge of the frame
  (14-17%) - full-res tokens spent on dark forest. The sky and the mist are fine at half resolution; the treetops are
  soft. `H3_MPEG_FLOOR` (upstream's floor, off by default) is the knob for "no subject, stay half-res"; not tuned here.
- **p4 car (fast motion)**: the region is the car (14-15%), clean edges and wheels while it moves; road markings,
  guardrail and spray outside are soft. Region flicker is lower than full res (8.7-9.4 against 10.7-14.8): the car
  holds together.
- Washed-out region: not seen - colour and exposure match across the region's edge in every pair.

## Speed (hardac1, wall clock, clip from text, 1344x768, 20 steps)

Times are sampling + VAE decode (+ the two-track paste), the user's wait after the text encoder (12-19 s for these
prompts, the same for both) and the model load (10-14 s). One card each, under the GPU lock; the switch's VAE round
trip is inside "sampling".

| Card | Frames | Full res: s/step, sampling + decode = total | Two-track: half-res s/step, two-track s/step, round trip, sampling + decode + paste = total | Region | Speedup (of the two totals) |
|---|---|---|---|---|---|
| B70 | 73 | 10.1, 201.9 + 14.3 = **216.2 s** (p1) | 1.85, 4.51, 43.0 s; 100.9 + 13.8 + 4.1 = **118.8 s** (p1) | 16.5% | 216.2 / 118.8 = 1.82x |
| B70 | 73 | 10.1, 202.3 + 14.3 = 216.6 s (p2) | 1.86, 4.98, 31.7 s; 92.8 + 13.9 + 4.2 = 110.9 s (p2) | 20.8% | 1.95x |
| B70 | 73 | 10.1, 202.4 + 14.2 = 216.6 s (p3) | 1.86, 4.51, 34.3 s; 92.6 + 13.7 + 4.1 = 110.4 s (p3) | 16.6% | 1.96x |
| B70 | 73 | 10.1, 202.4 + 14.3 = 216.7 s (p4) | 1.86, 4.21, 24.9 s; 81.4 + 14.5 + 4.1 = 100.0 s (p4) | 14.4% | 2.17x |
| B65 | 73 | 15.6, 312.0 + 20.2 = **332.2 s** (p1) | 2.88, 7.06, 71.4 s; 161.0 + 19.4 + 5.7 = **186.1 s** (p1) | 16.6% | 332.2 / 186.1 = 1.79x |
| B70 | 141 | 27.4, 548.5 + 28.2 = **576.7 s** (p1) | 3.84, 12.18, 75.3 s; 211.0 + 27.5 + 8.2 = **246.7 s** (p1) | 19.2% | 576.7 / 246.7 = 2.34x |
| B70 | 141 | 27.5, 549.3 + 28.4 = 577.7 s (p2) | 3.89, 15.22, 45.9 s; 201.0 + 27.6 + 8.2 = 236.8 s (p2) | 26.9% | 2.44x |
| B70 | 141 | 27.4, 548.2 + 28.3 = 576.5 s (p3) | 3.89, 10.34, 74.8 s; 200.5 + 27.6 + 8.2 = 236.3 s (p3) | 14.2% | 2.44x |
| B70 | 141 | 27.5, 549.3 + 28.1 = 577.4 s (p4) | 3.89, 10.72, 56.3 s; 184.3 + 27.6 + 8.2 = 220.1 s (p4) | 14.9% | 2.62x |
| B65 | 141 | 42.6, 851.9 + 40.1 = **892.0 s** (p1) | 5.87, 19.18, 122.2 s; 332.6 + 39.1 + 11.5 = **383.2 s** (p1) | 18.5% | 892.0 / 383.2 = 2.33x |

Two-track sampling = 16 half-res steps + the round trip + 6 two-track steps, e.g. B70 141 frames p1: 16 x 3.84 + 75.3
(round trip, in step 16) + 6 x 12.18 = 211 s. The round trip is mostly the VAE **encode** of the box at full size (19-65 s
on the B70, 60 s on the B65 at 73 frames; decode 2-8 s, bicubic under 3 s, loads 1-4 s): with clips the region's box,
the union over all frames, is often the whole frame (passers-by picked for a few frames widen it). It is 20-40% of the
two-track time and the first thing to speed up (Open).

Upstream's numbers, **an RTX 5090, not ours** (its README, early measurements, decode included, base model, 20 steps,
`--preset quality`): 22 frames 27.1 s against 14.4 s (1.9x), 73 frames 138 s against 56 s (2.5x), 145 frames 398 s
against 128 s (3.1x); ComfyUI, 73 frames 132.4 s against 68.3 s (1.94x); Turbo LoRA 8 steps, 15 s: 857 s against 244.5 s
(3.5x). Same shape here - the gain grows with length - at about 0.8x upstream's ratio (our full-res steps are relatively
cheaper against our round trip).

## Memory (the card's peak, the driver's view, /run/gpustat.json each second; 32 GiB = 32,768 MB a card)

| Configuration | Peak | Free at the peak |
|---|---|---|
| full res, 73 frames (B70 / B65) | 25,210-25,292 MB | 7.3 GiB |
| full res, 141 frames (B70 / B65) | 28,561-28,705 MB | 3.97 GiB |
| two-track, 22 frames (B65, with the dump) | 24,313 MB | 8.3 GiB |
| two-track, 73 frames (B70 / B65) | 25,483-25,592 MB | 7.0 GiB |
| two-track, 141 frames (B70 / B65) | 26,922-27,915 MB | 4.7 GiB |
| two-track with the region the whole frame, 73 frames (B70) | 26,346-26,382 MB | 6.2 GiB |

Every configuration leaves well over 1.5 GiB free. The two-track sequence adds 2.2-5.2 GiB (its log line: the packed
stream and scratch, the full-res queries' key and value set) on top of the half-res denoiser; the switch's VAE is
loaded and freed (`Device::trim`) before it. The 19.5 GiB of int8 blocks stay on the card as for H3.

## Use

```sh
nextsycl models pull minimax-h3-mpeg      # the two small files; the rest is minimax-h3's
nextsycl video job generate --here --model minimax-h3-mpeg --prompt "..." --width 1344 --height 768 --seconds 3.04 --steps 20 --out clip.mp4
    [--mpeg_switch 16 --mpeg_hr_steps 6 --mpeg_alpha 0.5 --mpeg_floor X --mpeg_feather 4] [--mpeg false] [--mpeg_dump F]
```

Front door: `video.generate {"model": "minimax-h3-mpeg", "prompt": ..., "width": 1344, "height": 768, "seconds": 3.04,
"steps": 20}`. The report (`video.jobs.get`) carries `mpeg`: the region's share and mask, the round trip's box, the
times.

## Open

- **The round trip at the switch**: a full-size VAE encode of the region's box over the whole clip (up to 65 s). A
  per-window box, the encoder only on the tiles that hold the region, or a learned latent upsampler (upstream's
  `LatentUpscaler`, untrained) would cut most of it; upstream's `latent` switch (no VAE) stripes the region.
- **Quality**: the region is softer than a full-res render, by the method (the "all" check). Upstream's LoRA that lets
  the full-res track start earlier is unpublished ("changes the whole picture"); an earlier switch without it seams.
- **No subject** (p3): the head spends 14-17% on random edge tokens; `H3_MPEG_FLOOR` untuned.
- **Isolated one-token regions** show a block edge; dropping islands below a few tokens (not upstream's rule) would
  avoid it.
- Turbo (an 8-step FL2V LoRA we do not have as a card), Q6_K / Q4_K_M with the int8-fitted maps, other sizes and step
  counts, keyframes and references: untested.
