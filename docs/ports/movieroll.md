# MovieRoll (and HunyuanVideo-1.5 under it): port plan

Status: **plan** (2026-11-23, the user asked for a full port; HunyuanVideo-1.5 first, on the traditional video methods;
MovieRoll's long form may need a new kind of endpoint).

## What the two models are

**HunyuanVideo-1.5** (`tencent/HunyuanVideo-1.5`, Tencent Hunyuan Community License; open, 200 with our token):
- DiT `HunyuanVideo_1_5_DiffusionTransformer`: 8.3B, 54 MM double blocks (a video stream and a text stream, joint
  attention), hidden 2,048, 16 heads x 128, qk RMS norm, qkv bias, GELU-tanh MLP x4, 3D RoPE (16 / 56 / 56, theta
  256), patch 1x1x1, in 32 / out 32 channels (`concat_condition`: noise + the condition latents and a mask),
  `use_cond_type_embedding`, text refiner (`single_refiner`), text states 3,584 (Qwen2.5-VL-7B) + glyph byT5, vision
  states 1,152 (I2V: a SigLIP-type image encoder - which one: to read in the code).
- VAE: 3D causal, 16x spatial / 4x temporal (to verify), 5.0 GB. SR upsamplers (720p / 1080p) separate, later.
- Checkpoints, 33.3 GB each (fp32): 480p t2v, t2v cfg-distilled, i2v, i2v cfg-distilled, i2v step-distilled (8 or 12
  steps), 720p t2v / i2v (+ distilled, sparse), 720p / 1080p SR. 121 frames (5 s at 24 fps) is the native clip.
- Text: Qwen2.5-VL-7B-Instruct hidden states + byT5-small with Glyph-SDXL-v2 (quoted text rendered as glyphs).
  **Both encoders are already ported for HunyuanImage 2.1** (`image/hunyuanimage21`): reuse, don't redo.

**MovieRoll** (`yunyangge/MovieRoll`, gated: **403 for our token - the user must accept its licence on HF**; Tencent
Hunyuan Community License, territory excludes the EU, UK and South Korea; arXiv 2611.21339, code
github.com/Tencent-Hunyuan/MovieRoll forked from HunyuanVideo-1.5 at 60783e7):
- A full fine-tune of HunyuanVideo-1.5-480P-T2V (same 8.3B, 33.3 GB files): `base/` (50 steps, AnnealCFG 1 -> 4,
  flow shift 5), `distilled/t2v` and `distilled/i2v` (5 steps, no CFG). 480p (distilled 480x864 / 864x480), 24 fps,
  lengths 16k+1 frames (241 = 10 s, 721 = 30 s, 1441 = 60 s; evaluated to 60 s).
- Long form by **rolling diffusion**: latent chunks of 4 frames, a window of 5 chunks denoised together at rising
  noise levels (bidirectional inside the window, causal to the clean history); a finished chunk leaves the window
  and its KV is cached. **HieraKV**: the first 6 blocks keep all history, the other 48 a sink + recent chunks. **KV
  recompute** at 18 cached chunks (sink, representative mid-event chunks, the 3 most recent, re-encoded at t = 0
  without text). Context noise on cached history (0.05 base, 0.15 distilled t2v, 0 i2v).
- **Events**: an ordered list of prompts, one per window, played one after another ("a new prompt every 10 s");
  an event switch recomputes the KV. I2V: a first image. Streaming playback is the upstream TODO.
- Decoder: PS-VAE, decoding a chunk at a time (`vae_stream_decode_max_chunk_size: 4`).
- Upstream needs a 96 GB GPU (no offload) and FlashAttention 2 + 3 (H100). Paper: 60 s at 480x864 on one H20 948 s
  (FA3) / 707 s (SageAttention); PS-VAE 176 s of that.

## Will it fit a 32 GB Arc card (estimates - to measure)

- Weights: 8.3B in int8 / Q8 8.3 GB (the files are fp32; we convert once, as for other DiTs).
- Tokens: 480x864 at 16x is 30 x 54 = 1,620 a latent frame; a chunk 6,480; the window 32,400.
- KV, bf16, 8 KB a token a block: 6 full-history blocks x ~117K tokens (18 chunks, the recompute bound) ~5.6 GB +
  48 blocks x ~20K (sink + recent) ~7.5 GB = **~13 GB** (int8 KV: ~6.5).
- Text encoder (7B, Q4 as HunyuanImage's ~4.5 GB) runs once per event: loaded, used, freed - or kept on the other
  card.
- So one card ~25 GB with bf16 KV; two cards (text encoder + VAE on the B65, the DiT on the B70, or the DiT's layers
  split) leave room. HunyuanVideo-1.5 alone (121 frames: 31 latent frames, ~50K tokens) fits one card easily.

**Speed, a rough guess from FLOPs (not a measurement):** a window step is ~1.2e15 FLOPs (linear ~0.3e15, attention
~0.9e15); the distilled model makes about one window pass per chunk of 16 frames, so 60 s (90 chunks) is ~1.1e17.
At ~40-50 TFLOPS sustained bf16 on the B70 that is **~40 min of compute per minute of video** (the H20, ~3x the
B70's dense bf16, took 16 min). int8 XMX products and a Sage-style attention (we have both) would cut it. The
50-step base is ~10x that: a reference, not a served card.

## Phase 0 - access, files, a reference (days 1-2)

1. **User**: accept MovieRoll's licence on huggingface.co with the account of hardac1's token (403 today).
2. Download to `/var/mnt/4TBSSD/nextsycl-models/` (741 GB free; the 2 TB has 78 GB): HunyuanVideo-1.5 `vae/`,
   `transformer/480p_t2v`, `480p_t2v_distilled`, `480p_i2v_step_distilled` (~105 GB), the I2V image encoder;
   MovieRoll `distilled/t2v`, `distilled/i2v`, `base` (~100 GB). Encoders from HunyuanImage 2.1's files where they
   are the same (check: Qwen2.5-VL-7B states at the same layer, the same byT5 glyph checkpoint).
3. Reference: no NVIDIA GPU here, and upstream wants FA3. Run the reference in PyTorch on the CPU (sdpa in place of
   flash attention) at tiny sizes - 128x224, 2 windows, a few steps - in a capped container (`--memory 48g`; the fp32
   DiT is 33 GB, the text encoder run separately first), dumping each stage. If the platform peer's NVIDIA box can
   run it at real sizes, ask it for dumps (its user's call).

## Phase 1 - HunyuanVideo-1.5 on the traditional video methods (~1-2 weeks)

- New engine `video/hunyuanvideo15` from `video/example` (the video kind's contract), kernels in
  `kernels/video/hunyuanvideo15` + silo. Reuse: `crates/diffusion` (flow-match Euler, shift), HunyuanImage 2.1's
  Qwen2.5-VL + byT5 glyph encoders, H3's video plumbing (mp4 out, jobs), the XMX / Sage attention kernels.
- Order: VAE decode (bit for bit vs the reference on a latent) -> DiT one block -> full DiT one step -> T2V 480p 121
  frames end to end -> I2V (image encoder + concat condition) -> CFG-distilled and step-distilled cards -> VAE encode
  (I2V's first frame).
- Served through what exists: `video.generate` (prompt, image, seconds, size, seed, steps, cfg), `video.status /
  wait / cancel / timeline`; cards `hunyuanvideo-1.5-480p-t2v-distilled-q8`, `-i2v-step-distilled-q8`; vouch
  memory measured per card; merge-gate baselines on both cards. 720p and the SR upsamplers after.

## Phase 2 - MovieRoll on that engine (~1-2 weeks)

- Same DiT code with MovieRoll's weights (`distilled/t2v`, `distilled/i2v`; base kept for checks): the rolling
  window scheduler (per-chunk noise levels, bidirectional in window / causal to history attention mask), the KV
  cache with HieraKV retention (6 full / 48 sink+recent), KV recompute (threshold 18, recent 3, event switch),
  context noise, AnnealCFG for base, the event list (one prompt per window; text encoded once per event), the
  streaming VAE decode of each finished chunk.
- Checks: one window against the reference dump, then a 3-window rollout (KV carried), then an event switch and a
  recompute; a 60 s film on both cards: time, peak VRAM, no seams at chunk joins.

## Phase 3 - the long-form endpoint (~1 week)

A film is minutes of GPU time, many prompts and a growing output: the one-shot `video.generate` does not fit it.
Proposed (rule 7: each method is its handler, the client generated):
- `video.roll` - start: `{model, events: [{prompt, seconds}], image?, seed, size, live?: bool}` -> a job (the jobs
  machinery: `video.status / wait / cancel / timeline` work on it). `seconds` per event sets when the next prompt
  takes over (whole chunks: 16 frames = 0.67 s).
- `video.rollAppend` - `{job, events: [...]}`: more prompts sent in while it runs (direct the film live); with
  `live: true` the job waits for more (holding its card and KV) up to `max_idle_s`, then ends; `video.rollEnd`
  ends it now.
- Output as it is made: each decoded chunk appended to a fragmented MP4 and an HLS playlist
  (`/video/jobs/<id>/live.m3u8`), so a browser plays while it generates; `video.status` lists the segments and the
  seconds done; the finished file at the end as today.
- Backplane (rule 8): a live film holds its card like a reservation (rsvp_s / max_willing_to_wait_s apply); the ETA
  is seconds x the measured seconds-per-chunk.
- Later: a WebSocket for pushed progress / frames (with Haru-4B's back-and-forth socket, TODO.md).

## Phase 4 - speed

int8 XMX products for the DiT, Sage-style int8 attention over the window and history, int8 KV, the two cards
(DiT layers split, or one film a card), a cheaper VAE path; measure first (pp/tg do not apply: seconds of video a
minute).

## Risks

- Gated weights and a territorial licence (non-EU/UK/KR); not for redistribution in our cards' URLs without care.
- No GPU reference: CPU dumps at tiny sizes only; the rollout's long-range behaviour is checked by eye and by
  continuity metrics, not against the reference at 60 s.
- Speed: tens of minutes of compute per minute of film on one card until phase 4.
