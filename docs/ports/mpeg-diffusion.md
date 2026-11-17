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

The one new kernel is `ns_video_h3_silo_gather` (kernels/video/h3/silo, `dist/silo/libnextsycl-h3.so`: rows picked by
an index from a strided buffer, 16 bytes a work-item, an exact copy). Without the silo the same rows are copied one
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

Upstream's `lot/test_lot.py` ("ok 33") is a CPU test of upstream's own code; it is not run here (it does not touch
ours).
