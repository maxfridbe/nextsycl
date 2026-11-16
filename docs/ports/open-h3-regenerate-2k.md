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
-> the H3 video decoder at 2560x1440 -> cut to the draft's real frames -> the draft's own sound muxed back
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
latent, the fast VAE (6.5 GB; not used yet, see Open), and **26 GB of the bf16 Ref2VA checkpoint: only its time
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

## Use

```sh
nextsycl models pull minimax-h3-regen2k-ic        # IC LoRA, HyperFlow, silence, 26 GB of AdaLN tensors; the rest is shared
nextsycl video job regen2k --here --model minimax-h3-regen2k-ic --draft draft.mp4 --caption "<the draft's prompt>" --out out.mp4
    [--sampler up3|up4|hf8] [--resolution 2k|1080p] [--sharp 1] [--seed N]
```

Front door / studio: `video.generate {"prompt": "<the draft's prompt>", "regen2k": "<clip>.mp4" | "prev", "regen_sampler": "up3",
"regen_resolution": "2k", "sharp": 1}` (feature `video.regen2k`; the model in NS_VIDEO_ENGINES). The studio's clip
panel has "Regenerate 2K".

## Open

- **The fast VAE v2** (`vae_fast/fast_h3_vae_v2_24l`, downloaded, card file not yet): a 24-layer non-causal distillation
  of the H3 decoder with shifted 5x16x16 windows and per-block global tokens. It needs a windowed attention with
  padded (masked) keys - a new kernel (or a varlen gather) we do not have; the release's `--vae standard` path (our
  H3 decoder) is used instead.
- Speed: the streamed blocks and host LoRAs cost about 45 s per step (~7%); double-buffered uploads from pinned memory,
  or merging HyperFlow + LMS + IC into the int8 weights once (requantized), would remove most of it. The time tables
  include the audio stream's timesteps, which no row uses (pinned silence): a few wasted points.
- 1080p (computed at 1088 rows, centre 1080 kept) and up4 / hf8 are wired but were not run on the box.
- Host memory: ~10 GiB of LoRAs + the text encoder's pinned copy (14 GiB) while a clip runs; the backplane gates VRAM,
  not host RAM (`ram_gib` on the card).
