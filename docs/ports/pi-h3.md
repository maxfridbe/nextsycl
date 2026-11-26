# pi-H3: MiniMax H3 in 2 or 4 steps, the frames without a VAE decoder

[← video](../video.md)

**pi-H3** (Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas; Stanford / Luma AI;
weights huggingface.co/Lakonik/pi-H3 at 4a864589, code github.com/Lakonik/LakonLab at 2ee2b334 and
github.com/Lakonik/ComfyUI-piFlow at 4355cc47) distils MiniMax H3's FL2VA model with pi-Flow v2 to 2 or 4 sampling
steps, and with **DWD** ("Decoding Without a Decoder") makes the frames from the denoiser's own last hidden states, so
H3's video decoder does not run. Its adapters are Model Derivatives of MiniMax H3 under the MiniMax H3 Community
License (the same terms as H3's; keep its NOTICE).

In nextsycl it is the H3 engine with an `adapter` file (`crates/h3/src/piflow.rs`, `crates/h3/src/dwd.rs`), four model
cards, and four `video.generate` engines:

| Card | Engine | Adapter file | Video out |
|---|---|---|---|
| `minimax-h3-pi-2step` | `PI2` | `pi_h3_2step_dwd_adapter_int8_convrot.safetensors` (2.19 GB, ComfyUI int8 ConvRot) | DWD head |
| `minimax-h3-pi-4step` | `PI4` | `pi_h3_4step_dwd_adapter_int8_convrot.safetensors` (2.19 GB) | DWD head |
| `minimax-h3-pi-2step-vae` | `PI2-VAE` | `pi_h3_2step_adapter/diffusion_pytorch_model.safetensors` (503 MB, Diffusers float16) | H3's VAE |
| `minimax-h3-pi-4step-vae` | `PI4-VAE` | `pi_h3_4step_adapter/diffusion_pytorch_model.safetensors` (503 MB) | H3's VAE |

Everything else is the `minimax-h3` card's files (the int8 FL2VA denoiser, the Q4_K_M text encoder and its vision
tower, the VAEs): `nextsycl models pull minimax-h3-pi-2step` on a box with H3 registers it from the files there and
downloads the adapter alone. On hardac1 the adapters are in `/var/mnt/4TBSSD/nextsycl-models/pi-h3/` (SHA-256 as the
cards; `SHA256SUMS` beside them).

## What it takes - and what it does not

| Input | pi-H3 |
|---|---|
| a prompt (H3's three-field format) | yes |
| a first frame, a last frame, both (`first_frame`, `last_frame`) | yes - what the adapters were distilled for (pi-H3's README: "text, first-frame, and last-frame conditioning") |
| reference pictures, clips, voices (`ref_images`, `ref_videos`, `ref_audios`: **ref2va**) | **no.** The adapters are trained on the FL2VA transformer, not Ref2VA's (`transformer_ref`); LakonLab and ComfyUI-piFlow load them only into the FL2VA model. The engine and the studio refuse references with pi-H3 (a 400 that names REF2VA). Experimentally, `H3_PI_REFS=1` over the Ref2VA denoiser lets them through: the subject holds, the camera and the place drift ("Over Ref2VA" below) |
| motion guides, a carried sound, a soundtrack (`guide_clip`, `first_audio`, `soundtrack`) | they run (the same keyframe rows as a first frame) but were not checked against pi-H3's reference |
| ControlNet, masked regeneration, MPEG-Diffusion, a refine pass, knots, other samplers | refused |
| extra LoRAs (`loras`) | run beside the adapter's |
| steps | the adapter's (2 or 4) whatever a request asks |
| the latent upscaler | skipped with DWD (there are no latents to enlarge before a decode); the pixel upscaler (ESRGAN) works |

**The platform's films mostly use ref2va: pi-H3 cannot make those clips.** It is a fast text / first-frame /
last-frame model: a draft of a shot, a quick take, a clip between two pictures.

## What an adapter holds

Read from the files' headers (the ComfyUI int8 file under ComfyUI's names, the Diffusers files under Diffusers'):

| Part | DWD adapters | VAE adapters |
|---|---|---|
| LoRA, rank 64, no alpha (scale 1) | `attn.out_proj`, `mlp.fc1`, `mlp.fc2` of blocks 0-45 (138 pairs) | the same on blocks 0-49 (150) |
| AdaLN projection on the timestep curve [96768, 8] | every block's, in place of the base's (the base's bias kept) | every block's |
| blocks tuned whole | 46-49: out_proj, fc1, fc2 (int8 ConvRot in the ComfyUI file, as our base's) and their AdaLN bias; q, k, v stay the base's | none |
| the final layer whole | norm, AdaLN projection, video_out, audio_out | the same |
| `dwd_norm_out` | a LayerNorm without affine, (scale, shift) = linear(silu(curve point)) | - |
| `dwd_pixel_refiner` (SolH3DWDHead) | `feature_expand` [8192, 5376], 4 registers, 8 Sol-H3-VAE decoder layers (int8 ConvRot in the ComfyUI file), norm_out, `proj_out` (4 frames a latent) and `first_frame_proj_out` (the one frame of a block's first latent) | - |

The base the adapters were trained on is multimodalart/MiniMax-H3-Pruned (1a0ef5e6); ours is Comfy-Org's int8
FL2VA at e5eb578a. `reference/pi-h3/base_check.py` range-fetched theirs: the AdaLN curve and the output head are
identical, block 0's / 30's / 49's matrices within int8's rounding (rel 0.012 / 0.009 / 0.010), block 0's AdaLN
projection rel 0.0017 (their bfloat16 against our float32) - the same weights.

## How it runs

```text
load:   the int8 FL2VA denoiser as for minimax-h3, then the adapter merged (piflow.rs apply):
        every block's AdaLN projection replaced; blocks 46-49's three int8 matrices replaced; the final layer replaced;
        the LoRAs in host memory (bf16, 0.4 GB), each block's sent to the card as it runs, beside the int8 products;
        the DWD head's norm and expansion on the host, its layers read from the adapter file for each decode
sample: LakonLab's FlowEulerODEScheduler(num_train_timesteps=1, shift 12, max_raw_t 0.93 | 0.96, final_step_size_scale 0.6)
        2 steps: sigmas 0.993767, 0.865340, 0     4 steps: 0.996540, 0.964451, 0.899297, 0.695652, 0
        the audio on the same raw times through shift 3 (what H3's sampler derives); Euler, no CFG;
        from the noise itself (LakonLab's pipeline), not the noise at the first sigma (ComfyUI's)
DWD:    the last call keeps the video rows' hidden states after block 49 (to the host, in chunks) and the curve point
        f = LayerNorm(h) (1 + scale) + shift, folded into feature_expand once a clip (W diag(1 + scale), b + W shift)
        each 2x2-latent token -> its 4 Sol tokens; the 8 Sol layers (shifted 5-latent x 16x16 windows, per-block global
        tokens: the fast decoder's code, fastvae.rs Sol, on a context on core's queue: the H3 video silo's kernels),
        loaded after the denoiser's buffers are given back; norm_out; proj_out / first_frame_proj_out -> frames
audio:  H3's audio decoder on the last latents, as every H3 clip
```

The DWD head is Open-H3-Regenerate-2K's fast decoder's architecture (the same Sol-H3-VAE layers): `fastvae.rs` now
holds those layers and the pixel head as `Sol` (bfloat16 or int8 matrices, an optional first-frame projection), used by
both. The fast decoder's frames are unchanged by that (448x256, 22 frames: bit-identical before and after, PSNR 200 dB).

## Parity

`reference/pi-h3/ref.py` runs LakonLab's own code - its `MiniMaxH3Transformer3DModel` (pruned, with `dwd_head_config`),
`SolH3DWDHead`, `FlowEulerODEScheduler` and its pipeline's loop - on the CPU in float32, with the weights the engine
runs (the int8 base and adapter un-rotated), the engine's text conditioning (the text encoder is H3's, unchanged) and
the engine's noise; it writes a dump the engine's `denoise` job replays. 448x256, 22 frames (7 latent frames, 966
tokens), seed 42, the fox prompt; B65:

| Stage | 2-step DWD | 4-step DWD | 2-step VAE adapter (Diffusers fp16) | base H3, no adapter (control) |
|---|---|---|---|---|
| the stream after block 0 / 15 / 25 (video rows) | rel 6.5e-3 / 7.2e-3 / 1.6e-2 | - | - | 6.4e-3 / 7.2e-3 / 1.4e-2 |
| after block 35 / 45 / 49 | cosine 0.9939 / 0.9779 / 0.9803 | - | - | 0.9975 / 0.9972 / 0.9994 |
| step 1 velocity (the same x on both sides) | cosine 0.99569 | 0.99660 | 0.99564 | 0.99983 |
| denoised estimate, step 1 .. last | 0.99270, 0.98256 | 0.99518, 0.99046, 0.99292, 0.99101 | 0.99262, 0.98447 | 0.99979, 0.99783 |
| audio estimate, every step | 0.99934, 0.99917 | 0.99918 .. 0.99735 | 0.99933, 0.99941 | 0.99953, 0.99961 |
| the DWD head on the reference's hidden states | **PSNR 66.6 dB** (cosine 0.999999) | **65.7 dB** | - | - |
| the frames, end to end | PSNR 25.4 dB (cosine 0.9934) | 27.9 dB (0.9957) | (VAE decoded: the H3 decoder's own parity) | - |
| the DWD curve point | identical | identical | - | - |

Reading it:

- **The adapter is applied as LakonLab applies it.** Its LoRAs and AdaLN projections act on every one of blocks 0-45,
  and through block 25 the stream is as close to the reference as the base model's is (rel 6.5e-3 against 6.4e-3 after
  block 0, 1.6e-2 against 1.4e-2 after block 25): a wrong LoRA, scale or table would show from block 0.
- **The DWD head is exact** to its int8 matrices: 66 dB on the reference's own hidden states.
- **The gap after block 25 is the distilled model's sensitivity, not the port.** The engine runs int8 W8A8 products in
  bfloat16; the reference float32. The same perturbation added after block 0 in the reference (`ref.py --perturb`,
  cmp.py) moves the 2-step model's first estimate 9x as far as the base model's (cosine 0.9697 against 0.9968) and its
  finished latents 7.6x (0.9454 against 0.9928): a 2-step distilled model carries a small error a long way. LakonLab's
  own pipeline runs bfloat16 with FP8 W8A8 linears (`quantize_frozen_linears`), so it is in the same place.
- **By eye the clips are the same** (`reference/pi-h3/sheet.py` against the engine's frames): the same fox, pose, gait,
  trees and light, frame by frame.
- **First and last frame** (832x480, 5 s, 2 steps): frame 0 against the given picture PSNR 31.0 dB, the last frame
  32.4 dB (H3's own pinned ends: ~33.8 dB) - the clip goes from one to the other.

## Speed

hardac1, the fox prompt, seed 7, each clip once through `video job generate --here` (the engine loaded first, ~5 s, not
counted): the job's time, from the prompt to the written .mp4 - text encoder (2-14 s: the Q4_K_M layers from the page
cache or the disk, whichever it was that time), sampling, the DWD head or H3's video decoder, the audio decoder, the mp4.
Against H3 as it is served (the int8 FL2VA `minimax-h3`, 15 steps - the low end of the platform's 15-20 - the same
prompt, seed, size and length). Peak: the card's own figure (`/run/gpustat.json`, every second; the card idle before).
Frames 17k + 5: 5 s is 124 frames (5.17 s), 10 s is 243 (10.12 s).

**Arc Pro B70** (`arc-pro-b70-256u-32g`)

| Clip | pi-H3 2 steps (DWD) | pi-H3 4 steps (DWD) | H3 INT8, 15 steps (VAE) | 2 / 4 steps against H3 |
|---|---|---|---|---|
| 832x480, 5 s | **32.0 s** (sampling 12.4, DWD 1.6), 22.3 GB | 41.0 s (24.5, 1.4), 22.3 GB | 114.1 s (86.5, decoder 11.2), 24.9 GB | 3.6x / 2.8x |
| 832x480, 10 s | **43.1 s** (31.3, 2.3), 25.5 GB | 76.0 s (61.9, 2.4), 25.5 GB | 259.6 s (222.8, 21.4), 25.5 GB | 6.0x / 3.4x |
| 1344x768, 5 s | **63.1 s** (46.6, 3.1), 27.6 GB | 104.5 s (92.9, 3.5), 27.5 GB | 368.0 s (337.0, 24.6), 27.5 GB | 5.8x / 3.5x |
| 1344x768, 10 s | **166.1 s** (141.0, 6.3), 30.3 GB | 298.5 s (281.1, 5.7), 30.3 GB | 1091.0 s (1031.1, 49.2), 30.3 GB | 6.6x / 3.7x |

**Arc Pro B65** (`arc-pro-b65-160u-32g`)

| Clip | pi-H3 2 steps (DWD) | pi-H3 4 steps (DWD) | H3 INT8, 15 steps (VAE) | 2 / 4 steps against H3 |
|---|---|---|---|---|
| 832x480, 5 s | **38.3 s** (18.8, 2.4), 22.4 GB | 48.0 s (37.1, 2.3), 22.4 GB | 159.2 s (132.5, 15.1), 24.9 GB | 4.2x / 3.3x |
| 832x480, 10 s | **73.8 s** (47.2, 3.3), 25.5 GB | 111.1 s (94.6, 3.5), 25.5 GB | 389.3 s (342.8, 29.5), 25.6 GB | 5.3x / 3.5x |
| 1344x768, 5 s | **91.6 s** (71.0, 4.4), 27.5 GB | 161.5 s (142.8, 4.8), 27.5 GB | 567.0 s (519.0, 34.5), 27.4 GB | 6.2x / 3.5x |
| 1344x768, 10 s | **241.1 s** (217.9, 7.7), 30.3 GB | 459.3 s (435.5, 8.1), 30.3 GB | 1686.2 s (1605.2, 68.6), 30.4 GB | 7.0x / 3.7x |

- A pi-H3 step costs 5-7% more than an H3 step (832x480, 5 s: B70 6.16 s against 5.76, B65 9.35 against 8.82): the
  adapter's rank-64 LoRAs beside the int8 products, their weights sent to the card a block at a time. The rest of the
  gain is the step count (15 -> 2 or 4) and the decoder: the DWD head takes 1.4-8 s where H3's video decoder takes
  11-69 s (5-10x).
- **1344x768 x 10 s fits** only because the adapter holds nothing extra on the card while it samples: the LoRAs stay in
  host memory and the DWD head's layers go to the card for the decode (first try: 11.7 GiB needed, 11.4 there; now 12.0).
  The clip's 73,494 tokens run the MLP 8,192 rows at a time, as H3 does at that size; the card peaks at 30.3 GB.
- Run to run the text encoder's time moves the 832x480 5 s totals by up to ~10 s (it is 30-40% of those clips).
- The frames were looked at for every clip (contact sheets of frames 0 / middle / last): the fox, the snow, the trees
  and the light hold through every clip; the 2-step clips are a little softer, the 4-step ones sharper in the fur; next
  to H3's 15-step clip at 1344x768 the pi-H3 clips carry more contrast and fur texture and a closer framing (a different
  sample of the same prompt). The sound: stereo 32 kHz, -19.6 LUFS integrated; its spectrogram shows short bird chirps
  and the steps' broadband hits over a low bed, as the prompt's soundscape asks.

## Merge gate

`tooling/gate-baselines.json` has the H3 engine now (`video/h3`, which a change to `crates/h3` or `video/h3` touches): its
quick run replays the 2-step DWD reference dump (`video job denoise --model minimax-h3-pi-2step --dump
/var/mnt/4TBSSD/nsref-pih3/out/dwd2.safetensors`; the denoise job's `--logits-out` / `--gen-out`: the finished latents and
the DWD frames hashed, `pp` the denoiser's tokens x steps a second, `tg` the DWD head's thousands of pixels a second). A
replayed dump, not a generate job: H3's generate job is not bit-reproducible run to run (the base model's latents cosine
0.9988 between two 8-step clips, the 2-step adapter's 0.95-0.98 - it carries the difference further), while the replay
is, on the B65.

| Card | exact | pp | tg | recorded |
|---|---|---|---|---|
| `arc-pro-b65-160u-32g` (GPU 1) | `4789d7d6901960da` (two runs, the same) | 1880.3 | 6528.3 | cdb35b0 |
| `arc-pro-b70-256u-32g` (GPU 0) | none: two replays differ on the B70 (the base model too: latents cosine 0.99777 / 0.99805 against the reference in two runs of the same dump) - H3's kernels are not bit-reproducible there, so the gate takes the B65 only | 2435.7-2442.9 | 6331.7-6552.3 | - |

The baseline was taken by running the gate's command twice in the build copy (`~/src/pih3/gaterec.sh`), not through
`gate.sh` itself (it builds the commit again in its own copy); `gate.sh`'s copy needs `minimax-h3-pi-2step` in its
registry (`nextsycl models pull minimax-h3-pi-2step` there finds every file on the box) and the dump where the entry's
mounts say.

## Lessons

- **Start from the noise itself.** LakonLab's pipeline (Diffusers' `MiniMaxH3PrepareLatentsStep`) draws the latents and
  steps from there whatever the first sigma is (0.9938 for 2 steps); ComfyUI's sampler scales the noise by the first
  sigma. The engine follows LakonLab (`Denoiser::pure_noise`); in the sampler's audio space that start is the noise times
  sigma0 / sigma0_audio (the sampler carries the audio at sigma / sigma_audio).
- **ComfyUI's H3 sampling is LakonLab's exactly.** H3's sampler carries the audio as 4x its latent on the video's sigmas;
  that is an exact reparametrisation of an Euler run on the audio's own shift-3 schedule (x_s = (1 - s) 4 a0 + s n), so
  one engine sampler serves both; the reference's audio estimates are compared as 4 x0.
- **A distilled model magnifies the engine's own error**: compare block by block against the base model under the same
  check before blaming the adapter (the taps), and measure the sensitivity in the reference (`--perturb`).
- **LakonLab without its training stack**: its package `__init__`s import mmcv 1.6, triton and the rest; its leaf modules
  load with three of mmcv's init helpers and its registry stood in (ref.py `load_lakonlab`).
- **DWD's norm folds into the expansion**: one timestep for every video row, so the LayerNorm's scale and shift become
  feature_expand's weights and bias for the clip - one product instead of a norm, a modulation and a product.

## Over Ref2VA (experimental)

The platform's films use reference pictures (ref2va), which the FL2VA adapters were not trained for. The experiment: the
2- and 4-step DWD adapters loaded over the **Ref2VA** denoiser (`minimax_h3_ref2va_pruned_int8_convrot`) with two
experimental engine options - `H3_PI_REFS=1` lets references through, `H3_PI_CURVE=<the FL2VA checkpoint>` puts FL2VA's
timestep curve under the adapter's AdaLN projections - in registry entries of the build copy only (no card).

### What the adapter replaces, and how far Ref2VA is from FL2VA there

`reference/pi-h3/ref2va_diff.py`, both bases un-rotated, relative distances:

| Part | Ref2VA vs FL2VA | adapter vs FL2VA |
|---|---|---|
| AdaLN projections on the curve [96768, 8], 50 blocks | 1.86 (median) - **but in another basis**: the modulation they give, W c(t) + b at t = 0.007 / 0.13 / 0.5 / 0.9 / 0.999, differs by 0.035 (median; max 0.053) | modulation 0.003 (median), up to 0.21 (blocks 46-49, the last steps' t) |
| `adaln_t_table` (the 8-point curve) | 0.018 | (none in the adapter: FL2VA's) |
| blocks 46-49 out_proj / fc1 / fc2 (replaced) | 0.020-0.034 | 0.028-0.24 (block 49's fc2 the most) |
| the other blocks' matrices (LoRA'd) | 0.025-0.036 | the LoRA update 0.006-0.016 |
| final layer (replaced) | AdaLN 1.92 in the basis / bias 0.033, norm 0.033, video_out 0.050, audio_out 0.078 | 0.028-0.079 |

So Ref2VA is a fine-tune of FL2VA about 3% away in every matrix - two to five times the adapter's LoRA updates (0.6-1.6%),
which were trained against FL2VA's matrices. The AdaLN projections look 186% apart only because each pruned checkpoint
fits its own 8-coordinate curve; what they produce is 3.5% apart. The adapter replaces every projection, so its own (and
FL2VA's curve, `H3_PI_CURVE`) is used throughout: the timestep conditioning is wholly the distilled model's, the matrices
Ref2VA's plus the FL2VA-trained LoRAs.

### The runs

Five shots of the platform's recent ref2va jobs (two films: the space skimmer chase - four shots with a subject and a
place picture, keyframes and motion guides - and a singer on a ship's deck - a person's picture, a motion guide; 10.125 s,
832x480 / 896x512), copied read-only, each run on the B65 with its own prompt, references, keyframes and seed as
production made it: Ref2VA at its served 15 steps, pi-H3 2 steps over Ref2VA, 4 steps over Ref2VA. Production's own clip
of each (Ref2VA, made on the B70) is scored too: the spread of Ref2VA against itself.

**Reference similarity** (`reference/pi-h3/refsim.py`: DINOv3 ViT-H+/16 CLS and SigLIP so400m pooled, each frame of 30, 60
... 240 against each reference picture, cosine averaged over frames and references):

| Shot (references) | Ref2VA (production, B70) | Ref2VA 15 steps (B65) | pi 2 steps | pi 4 steps |
|---|---|---|---|---|
| skimmer 222002 (ship) | 0.739 / 0.798 | 0.675 / 0.778 | 0.747 / 0.802 | 0.743 / 0.793 |
| skimmer 214805 (ship, place) | 0.598 / 0.766 | 0.597 / 0.785 | 0.605 / 0.777 | 0.607 / 0.768 |
| skimmer 124128 (ship, place) | 0.570 / 0.705 | 0.564 / 0.700 | 0.583 / 0.727 | 0.610 / 0.733 |
| skimmer 123702 (ship, place) | 0.586 / 0.685 | 0.585 / 0.694 | 0.586 / 0.723 | 0.588 / 0.727 |
| singer 212510 (the man) | **0.937 / 0.897** | **0.917 / 0.888** | 0.871 / 0.880 | 0.891 / 0.882 |

(DINOv3 / SigLIP.) The ship's own picture alone: Ref2VA 0.662-0.758, pi 0.704-0.765 (DINOv3).

**The motion guide** (PSNR of the first 22 frames against the guide clip's last 22, which they carry in): Ref2VA 40.4 /
30.3 / 35.2 dB (shots 124128 / 123702 / 212510), pi 2 steps 33.8 / 27.2 / 29.5, pi 4 steps 37.1 / 29.2 / 32.6.

**Time** (B65, the whole job): Ref2VA 442 / 445 / 496 / 497 / 588 s; pi 2 steps 78 / 78 / 94 / 98 / 107 s (5.5x); pi 4
steps 133 / 133 / 154 / 155 / 173 s (3.4x). Card peak 26.0-27.5 GB for all three.

**Sound**: present in every clip, -11.9 to -14.8 LUFS (Ref2VA -12.5 to -14.5); the singer's song has the same two sung
phrases with their harmonics at the same places in all three versions (spectrograms stacked).

**By eye** (contact sheets on hardac1, `~/src/pih3/look/sheet_<shot>.png`: the references, then frames 30 / 100 / 170 / 240
of each version):

- **The subject holds.** The skimmer's design (wings, twin engines, cyan and magenta edge lights) is the reference's in
  every pi frame; the singer's face, hair and purple shirt stay his.
- **The scene and the camera drift.** Ref2VA keeps the locked chase camera and the place picture (the ringed planet with
  its moons comes back in shot 123702's last frames); pi pushes in toward the planet, the rings and moons drop out, the
  night-side city lights turn to a daylit surface of high-contrast "swirl" clouds - a texture pi adds where Ref2VA's
  planet is photographic. On the singer, Ref2VA holds the medium shot as asked ("one continuous shot", the frame
  matching the reference); pi moves the camera in to a closer shot and puffs his hair up (the DINOv3 drop: framing, not
  identity).
- **The carried frames are looser** (2-6 dB below Ref2VA), the 4-step adapter closer than the 2-step.

### Verdict

pi-H3 over Ref2VA **keeps the subject references** - the scores are inside the spread between two Ref2VA runs of the same
shot for the skimmer, a few points under for the singer, and by eye the ship and the man are theirs - at 5.5x (2 steps)
and 3.4x (4 steps) the speed. It **follows the prompt's camera and the place reference less**: shots drift toward a
closer, livelier framing, the place picture's details go, and the motion guide is carried in less exactly. The
mismatch has a cause in the weights: Ref2VA's matrices are ~3% from FL2VA's, the adapter's LoRAs ~1%, so its distilled
corrections are applied to a model they were not fitted to.

Added as two **experimental** cards, `minimax-h3-pi-4step-ref2va` and `minimax-h3-pi-2step-ref2va` (engines `PI4-REF2VA`,
`PI2-REF2VA`): the Ref2VA files with `H3_PI_REFS: '1'` and `H3_PI_CURVE: '{curve}'` (a `curve` role naming the FL2VA
denoiser file - `models pull` finds it on a box that has `minimax-h3`), for fast drafts of reference shots: the subject
holds, the camera, place pictures and carried guides drift. Ref2VA's inputs (reference pictures, clips, voices) are taken;
ControlNet and masked runs are refused. Not for the films' final renders, where Ref2VA at 15-20 steps keeps the camera,
the place and the cut's carried frames.

## Open

- Ref2VA: the adapters over the Ref2VA denoiser keep the subject references but drift in camera and place ("Over Ref2VA");
  adapters trained on `transformer_ref` would be the real fix.
- Guide clips, a carried sound and a soundtrack run but were not compared with a pi-H3 reference.
- The DWD head reads the last hidden states through host memory (1.5 GB at 1344x768 x 10 s): kept on the card it would
  save that copy at the cost of room during the last step.
- Sol sparse attention (LakonLab / Spark-H3 on CUDA) is not ported; the denoiser runs dense (Sage) attention.
