# MovieRoll (and HunyuanVideo-1.5 under it): port plan

Status: phases 0 - 3 **built** (2026-11-24): HunyuanVideo-1.5 (docs/ports/hunyuanvideo15.md), MovieRoll on its engine
(arch `movieroll`, cards `movieroll-distilled-t2v-q8`, `-i2v-q8`, `movieroll-base-t2v-q8`) through video.generate, and
the long-form methods video.roll / rollAppend / rollEnd with a live HLS playlist. The plan as written first is kept
below the built sections.

## Phase 2 - built (2026-11-23)

**Files** (`yunyangge/MovieRoll` @ fe3150d9, gated - the licence accepted on the token's account):
`/var/mnt/4TBSSD/nextsycl-models/movieroll/{distilled/t2v, distilled/i2v, base}/model.safetensors` (33.3 GB each,
float32; `REVISION` beside them). The config's `MovieRoll-16164-MMDiT` is HunyuanVideo-1.5-480P's DiT to the
tensor - 54 double blocks of 2,048, 16 heads x 128, RoPE 16 / 56 / 56 at theta 256, `img_in` over 65 channels, the
2-block token refiner, byT5's mapper, `cond_type_embedding` [3, 2048] - with q, k and v **fused** in the file
(`img_attn_qkv` [6144, 2048], where HunyuanVideo-1.5 has `img_attn_q` / `_k` / `_v`), and no `vision_in` and no
`time_r_in` (the image of image to video enters through the KV cache, not SigLIP). `Dit::load` reads either. The
text encoders are HunyuanVideo-1.5's (the same video template, `hidden_states[-3]`) and the VAE is its VAE.

**Code read** (github Tencent-Hunyuan/MovieRoll @ 0b5f73e2): `movieroll_transformer.py` (`forward` with
`training=False`, `forward_inference`, `_forward_kv_recompute`), `streaming_attention.py`
(`chunk_inference_attention`), `kv_cache.py`, `rope.py`, `movieroll_dmd_pipeline.py` (`_execute_rollout`, the
distilled chain), `movieroll_pipeline.py` (`_select_recompute_chunks`, `_kv_recompute`), `context_noise.py`,
`scheduling_flow_match_streaming.py`, and upstream's own port notes (`agent/movieroll-port`) and SGLang /
vLLM-Omni ports (`sglang_movieroll/rolling`: the loop with no framework in it, "every function matches its movieroll
original" - the reference drives the official transformer with it).

**Engine** (`video/hunyuanvideo15/src/roll.rs` the three passes, `movieroll.rs` the loop and the engine, arch
`movieroll`):

- *denoise*: the window's chunks each at their own timestep - the image stream modulated chunk by chunk (spans), the
  text stream and the token refiner at the window's first (cleanest) chunk's timestep; image queries over
  `[history K, V | window | text]`, text queries over `[window | text]` (the text never sees the history, so one cache
  serves every prompt).
- *encode*: a clean chunk at t = 0 without text, its queries over `[history | chunk]`; its K (after RoPE) and V
  appended to the block's cache. A sink-recent block trims its cache to the sink first.
- *recompute*: the chosen chunks at t = 0, block-causal (chunk i over chunks 0..i); the 6 full-history blocks over all
  of them, the 48 sink-recent blocks over the first and the last as block 5 left them.
- RoPE compact, per block group: a block's cached chunks at frames 0.., the window after them; bounded by the grid's
  120 frames (checked). The cache keeps K after its RoPE (a chunk's position never changes while it is cached).
- HieraKV: blocks 0-5 the whole history (up to the threshold's chunks, or what an event-heavy recompute keeps), 6-53
  the sink and one recent chunk. KV recompute at 18 cached chunks and at every event switch (distilled: even when it
  drops nothing), keeping the first chunk, every event's middle and the 3 newest. Context noise on everything that
  enters the cache (0.15 t2v, 0 i2v, 0.05 base). AnnealCFG for base (`w = 1 + 3 t / 1000` a chunk, over the empty
  negative prompt). Events: the film split evenly between them (upstream's even split); the prompt of a round is the
  cleanest chunk's event.
- The loop is SGLang-MovieRoll's `run_rolling` (window schedule, Euler a chunk at its own sigma, the padding chunks
  after the last). Noise: torch's CPU generator in bfloat16 from the seed, in the loop's order (the distilled chain's
  stream - the check below finds it the same draws).
- Each chunk leaving the window goes through the streaming VAE decode (`stream_chunk`, one latent frame a call) and on
  to ffmpeg, so a film never holds its frames.
- Memory: the cache `half` or `int8` (K and V int8 with a float32 scale a row and head: `kv_q8` / `kv_dq8`, applied as
  the cache is written and as a block's history is read back); `auto` takes half when the film fits with it, else
  int8 (at 864 x 480: half 11.5 GiB, int8 5.8 GiB). The text encoders (4.2 GiB) are off the card while the film rolls:
  a steered film loads them for an event it is sent (the kernels' scratch given back first) and drops them after. The
  engine plans the room before it allocates (the driver has no out-of-memory error): see "Memory" below.
- The attention: the silo's ARK half flash over queries and keys apart (`ns_video_hv15_silo_attention2`), or its
  SageAttention (`sage`: q and k int8, k less its mean over the keys).

**Checks** (`reference/movieroll/ref.py`: the official transformer in float32 on the CPU at 128 x 224, its double
blocks read one at a time, the matrices rounded to Q8_0 as the engine rounds them; SDPA for FlashAttention; driven by
SGLang-MovieRoll's loop; every call dumped). One rollout covers it all: 7 chunks, 2 events (switch at chunk 3), the
threshold lowered to 4 - 11 denoise rounds, 4 encodes, a recompute at the event switch and one at the threshold. The
engine (`video job check --dir DIR`) makes the same film with the reference's noise, and replays every call on the
reference's own input (teacher-forced: the cache built from the reference's encodes):

| | velocity rel (each call, teacher-forced) | the film's latents |
|---|---|---|
| half cache | 2.7e-3 - 5.2e-3 (cos >= 0.999986) | rel 0.186, cos 0.983 |
| int8 cache | 3.7e-3 - 6.7e-3 (cos >= 0.999977) | rel 0.193, cos 0.982 |
| the reference against itself, Q8_0 weights vs float32 | 4.2e-2 at the first call | rel 0.419, cos 0.913 |

Every call's error stays flat through the cache's growth, both recomputes and the event switch (teacher-forced); the
free-running film drifts because the 5-step distilled chain amplifies any difference (the reference's own Q8_0 vs
float32 weights drift twice as far as the engine from the reference). The calls, their order, the recomputes (at
chunks 3 and 6, the same chunks kept) and the noise (the seed's stream, the same draws) match.

**Films** (the fox prompt in upstream's style - one event: 10 s; four events, every 15 s: 60 s; seed 7; 864 x 480):

| Card | Film | Chunks (rounds) | Denoise (a chunk) | Decode | The film | Cache | Peak on the card |
|---|---|---|---|---|---|---|---|
| B65 | 10 s, 1 event | 15 (20) | 252.9 s (16.86 s) | 67.9 s | 321.3 s | half (13.4 GiB with buffers) | 28.8 GiB |
| B65 | 60 s, 4 events (switches at chunks 22, 45, 67; recomputes at 19, 22, 36, 45, 58, 67, 79) | 90 (95) | 1562.4 s (17.36 s) | 422.1 s | 1985.5 s (33 min) | int8 (9.0 GiB with buffers) | 27.3 GiB |
| B65 | the steered roll below (6.7 s asked, one 3 s event added, then ended) | 15 (20) | 249.9 s (16.66 s) | 67.9 s | 318.3 s | int8 (9.1 GiB) | 28.8 GiB |
| B70 | 10 s, 1 event | 15 (20) | 159.5 s (10.63 s) | 44.6 s | 204.5 s | half (13.4 GiB with buffers) | 28.9 GiB |
| B70 | 60 s, 4 events (the same recomputes) | 90 (95) | 967.6 s (10.75 s) | 279.8 s | 1248.2 s (20.8 min) | int8 (9.0 GiB with buffers) | 27.3 GiB |
| B70 | the steered roll (as on the B65) | 15 (20) | 156.5 s (10.43 s) | 44.6 s | 201.5 s | int8 (9.1 GiB) | 28.9 GiB |

The load (the 33 GB float32 file to Q8_0, the text encoders, the VAE) is 31-77 s and not counted. A chunk is 16 frames
(0.67 s): a second of film takes ~16 s of denoising on the B70 and ~26 s on the B65 (the paper's H20: ~12 s with
FlashAttention-3); the whole 60 s film 20.8 min on the B70, 33 min on the B65. The parity check on the B70: every call
within 2.9e-3 (teacher-forced), the film rel 0.134. The decode's time grows with the film (its middle
attention sees every frame before: 4.5 s a chunk at 10 s of film, 4.7 s averaged over 60 s); the first 60 s attempt ran
out of card memory at chunk 36 and a second took 3 h for 82 chunks because the stream copied and transposed the
whole history's keys and values for every latent frame (fixed: they grow in place; the stream check is unchanged).

Memory (2026-11-24). The first film through the front door (video.roll, 2 events of 5 s, steered, the B70) failed at
its last chunk: the card refused 204 MB to the decoder. Again on the B65 with the card's use sampled every 2 ms: the
films ran at **31.6 of 32 GiB** - the old check counted 5.5 GiB over the cache and buffers, read at chunk ends, which
miss a pass's and a decoded frame's transients (the tables above read 28.8 GiB the same way), and it kept the text
encoders on the card for a steered film. This film's 65th latent frame then doubled the decoder's middle-attention keys
(64 -> 128 frames: 850 MB made beside the old 425) and the card had no more. Now (`Room` / `plan` in movieroll.rs, tested):
- the text encoders are off the card for every film; a steered one loads them for its added events and drops them after;
- the plan counts the cache, the passes' buffers, `ROLL_OVER` over them (7.35 GiB measured with half attention: the
  decoder's carried frames, the kernels' scratch, the transients; since round 3 7.875: the decoder's keys apart, sage's
  0.4 in, 0.5 to spare), the decoder's keys a frame (6.6 MB, made at once for the film's frames; a steered film's grow
  with its events, the card asked at each growth - round 3), and the card's last 1.5 GiB (`NSD_MEM_RESERVE_MB`) never
  touched;
- `auto` takes half when that fits, else int8; a film that fits neither is refused before it starts, "... too long for
  this card ...; at most N frames (S s) fit" (the studio answers it `too_long`, not retryable);
- (since 2026-11-24's second round: no length limit - the memory below is bounded whatever the film's length);
- the peak a film reports is the sampled one (every 5 ms), not the chunk ends'; the plan's need and the card's free memory
  are in the film's first lines.

After, on the B65, as video.roll runs them (steered: a control file, HLS), peaks sampled every 5 ms:

| Film | Chunks | Cache | Keys made for | Denoise (a chunk) | Decode | Total | Peak (free) | Seams: join / inside, worst |
|---|---|---|---|---|---|---|---|---|
| 10 s, 2 events of 5 s (the failed one) | 16 | int8 | 433 frames | 267.5 s (16.72 s) | 73.0 s | 341.1 s | 29.7 GiB (2.3) | 8.40 / 8.49, 1.20x |
| 61 s, 4 events of 15 s | 92 | int8 | 429 frames | 1598.0 s (17.37 s) | 462.2 s | 2061.2 s (34 min) | 29.7 GiB (2.3) | 7.58 / 7.62, 3.04x |

The 61 s film's worst join (66, frame ~1073) is the fox's head and leg sweeping into the frame close to the camera: the
background is continuous across it (looked at), motion, not a seam. A steered film's added event (the encoders loaded
for it, then dropped; a test of 2 x 3 s + 3 s added): the card at 27.10 GiB while it encoded, under the roll's 27.24.
The 10 s film with its keys made for its own 65 frames peaked at 27.4 GiB; a steered film's are made for what it may
reach (here ~2.4 GB more).

Seams (`seams.sh`: the mean |luma difference| of neighbouring frames, at the joins of chunks against inside them):
the 10 s film's 14 joins: mean |luma difference| across a join 10.08 against 10.11 between frames inside a chunk, the
worst join 1.11x the median of its chunk's own pairs; the 60 s film's 89 joins: 7.66 against 7.74, the worst 1.62x (join
55, frame 897 - the fox walking back into the frame from the edge, looked at: motion, not a seam); on the B70 10.22 /
10.30 (worst 1.11x) and 6.31 / 6.44 (worst 1.54x). No seams at the joins,
and none to see in the frames (sheets of every 10 s and the joins looked at).

**Not done**: int8 KV quality beyond the check above; the base chain was not run as a film (50 steps: ~10x the
distilled film's work - the reference card); MovieRoll's own SageAttention variant upstream (SageAttention 2, FP8 P V
on Hopper) - ours is v1.

## Phase 3 - built (2026-11-24): video.roll, video.rollAppend, video.rollEnd

- **video.roll** `{model, events: [{prompt, seconds}], first_frame?, live?, max_idle_s?, seed?, width?, height?, label?,
  project?, queue?, rsvp_s?, max_willing_to_wait_s?}`: a MovieRoll film. Each event holds for its seconds (whole chunks
  of 16 frames, at least one; no limit on their sum since round 3). It is made as a video.generate clip with its events - the same
  admission (a known wait over max_willing_to_wait_s declined), queue, reservation after it ends (rsvp_s), and
  video.status / wait / cancel by its label - so the backplane treats it as any clip: it holds its card while it runs,
  a live film included (rule 8).
- **video.rollAppend** `{label | name, events}`: more events. A running film reads them from its control file
  (`<name>.ctl.json` beside it, written atomically by the studio) between rounds (~17 s apart on the B65): they extend
  the chunk layout and the event boundaries, and the switch to each recomputes the cache as upstream's does. A queued
  film takes them before it starts.
- **video.rollEnd** `{label | name}`: no chunk is begun after it; the window's (up to 5) finish and the mp4 closes. A
  queued film leaves the queue. video.cancel stops a film at once.
- **live**: out of events, the film waits for more up to `max_idle_s` (default 300, at most 3,600) before its window
  flushes - holding its card and its cache - saying so once a minute (the studio's watchdog stops a job silent for
  20 min). The text encoders are loaded for the events it is sent and dropped after (since the memory fix).
- **Watching it as it is made**: ffmpeg's tee writes the mp4 and an HLS event playlist of fragmented-MP4 segments
  (`<name>.live.m3u8`, `<name>.live.init.mp4`, `<name>.live.NNNN.m4s`: 2 s each, a key frame a second) in the clips
  directory as the chunks are decoded; video.status's `live` (and its running_clips') is the playlist's URL
  (`/v1/video/files/<name>.live.m3u8`), served uncached by the door and the studio. A browser plays it with hls.js (or
  natively on Safari), ffmpeg / VLC directly.
- **ETA**: video.status's, from the film's own pace once it has done a chunk (seconds a chunk x the chunks left + the
  decode); before that the history's estimate for the model and length (system.eta).
- **The limit on events**: a KV recompute keeps the middles of the latest 4 events (upstream: every event's) - the same
  for up to 4, and a long or live film's cache stays bounded.

**Checked** (the engine directly, as the studio's job runs it - the B65): a live film of 2 events of 3 s (10 chunks);
at its 2nd chunk an event of 3 s was written into its control file, at its 6th the end. The film took the event at
chunk 10 (15 chunks), recomputed its cache at both switches (chunks 5 and 10, as upstream's event switch does), ended at
15 and closed its mp4 (241 frames, 318 s); its playlist grew a segment as each chunk was decoded (`roll_b65.live.m3u8`,
6 segments, ENDLIST at the end) and plays in ffmpeg. The front door's three methods and the studio's steering are
covered by the serve crate's tests (the events' checks, the rules); the studio was not run against the daemon here
(a build-copy studio never dispatches to production's daemon).

## Round 3 (2026-11-24): I2V, live mode and its page, no length limit, sage, speed

**Image to video** (`movieroll-distilled-i2v-q8`, `first_frame`): the picture scaled to cover the canvas, centre-cropped,
through the VAE's encoder (its mode x the scaling: upstream's MovieRollImageVAEEncodingStage), is the film's first chunk
(one latent frame) - encoded into the cache at t = 0 before the first window (context noise 0: the i2v folder's), the
layout [1, 4, 4, ...]. The reference (`ref.py roll --anchor cond_latent.npy --context-noise 0`, the i2v weights, 128 x 224,
7 chunks, 2 events) and the check (`video job check --dir ... --json '{"image": PICTURE}'`): the picture through the engine's
resize, crop and VAE encoder against the reference's latent rel 5.2e-3 (cos 0.999987: the resize's Lanczos is the image
crate's, not PIL's); every model call teacher-forced 2.6e-3 - 5.8e-3 (cos >= 0.99998, the t2v check's range); the film's
latents rel 0.29 (t2v: 0.19; the reference against itself in Q8 and float32 weights drifts to 0.42); the recomputes where
the reference's are (chunks 4, 6, 7).
A 10 s film from a 1024 x 1024 picture of a fox on the B65: 16 chunks [1, 4, ...], int8, denoise 230.1 s (14.38 s a
chunk), decode 68.4 s, 299.6 s, peak 27.2 GiB; the first frame against the picture covered and cropped to 864 x 480:
PSNR 31.7 dB, SSIM 0.893 (the VAE's round trip); seams: joins 11.71 against 11.67 inside, worst 1.30x; by eye the fox
gets up from the picture's pose and trots on, the camera following.

**Live mode, exercised** (the build copy's own studio and daemon on the B65, their own socket and clips):
- a live film of one 3 s event ran out of events and waited (`live - out of events at chunk 5`, its window held), took an
  event from video.rollAppend (`event 1 added: 5 chunks from chunk 5`, the switch's recompute), ran out again and, sent
  nothing more, ended at its max_idle_s of 600 s (`no new event in 600 s: the film ends`): 161 frames, 838 s;
- a live film with max_idle_s 60 timed out the same way (33 frames);
- the page drove a third: started it, added a 5 s event while it waited, ended it with video.rollEnd (`ended: 13 chunks`,
  209 frames) - its stage, chunks, ETA and log on the page as it went.
Found and fixed on the way: the studio's status said "starting, 0 / 5" for a whole film (its stage and progress were
read from H3's log lines) - a film now says `rolling: N of M chunks made` / `live: N of M chunks made, waiting for the
next event`, its percent from its chunks and its ETA at its own pace from its first chunk; and a live film's stream
showed nothing while it waited - x264's 40-frame look-ahead held its first 80 frames back - so the HLS encode is
`-tune zerolatency` now (a segment is written when its frames are). And with sage the page's film ran out of the card
after its added event twice: the plan kept no room for sage's int8 q and k (ROLL_OVER raised to 7.875 GiB, the
decoder's keys counted apart), it reserved a steered film's whole decoder window from the start (now its keys grow with
its events, the card asked at each growth), and the added event's encode left its shapes' primitives in the kernels'
caches (given back after it now).

**The page** (`wfe/roll`, served by the studio at `/roll`; through the front door `/open/<model>/roll`): a film's model,
events, seed, live and max_idle_s; its stage, chunks, ETA and card; the HLS stream playing as it grows (hls.js from
jsDelivr where the browser has no HLS of its own, Safari's native), the mp4 once done; events added while it runs, end and
cancel. It calls the front door's generated client (rule 7), its `video.*` methods over the studio's /rpc - the studio
has `roll` now, video.roll's film as the door's handler makes it. Checked with a headless Firefox (Marionette) on the box:
every control worked (screenshots). The two Firefoxes at hand (the box's, this desk's) have no H.264 decoder (Fedora's
ffmpeg-free / noopenh264), so they show "can't be played" for the stream and the mp4 alike; a browser with H.264
(Chrome, Edge, Safari) plays both.

**video.status's `live`**: running_clips' rows carried no `live`, and the top-level one was set before the playlist
existed. Both say the playlist now, from its first segment (`#EXTINF`) on; the door rewrites both to its
`/v1/video/files/...` (no-store). Checked on the private studio: the row's and the status's the same, the playlist served
200 with `Cache-Control: no-store`.

**No length limit.** A film runs as long as its card holds it, and what it holds no longer grows with its length:
- the DiT's cache: HieraKV and the recompute, bounded already;
- the decoder's middle-attention keys: upstream's streaming VAE keeps the full history up to 361 latent frames (a
  minute) and the latest 361 past it (DEFAULT_STREAMING_EXACT_ATTENTION_FRAMES: "a genuine approximation, since the
  bottleneck attention is near-uniform over history") - so do we now: a window in buffers of 361 + 64 frames (2.8 GB),
  moved back to their front every 64 frames; a film up to a minute is exact as before;
- the finished chunks on the host: only those a later recompute may take (`kept_chunks`: the first, the latest
  events' middles, the newest, the event under way from its middle on), not every chunk;
- the events' encodings: let go once their chunks have all left the window.
The plan counts the decoder's keys for the film's own frames (up to the window and its slack, 361 + 64). A steered
film's grow with the events it is sent: the card is asked at each growth (the new buffers beside the old while they are
copied, what comes and goes between chunks, the last 1.5 GiB), and an event it has no room for ends the film there, said
so in its log. The 2-minute STEER_MAX_FRAMES and the 60 s cap on video.roll's start are gone. Kept, with their reasons in the code: 64 events a call
(each is encoded before its chunk, the card waiting), 60 s an event (its chunks stay on the host from its middle on until
it ends), 4,000 characters a prompt (the encoder reads 1,000 tokens), max_idle_s 3,600 (a waiting live film holds its
card).

**Sage over the window and the history** (the silo's SageAttention v1 for queries and keys apart): the default in the
three cards now (`NS_MR_ATTN: sage`, as HunyuanVideo-1.5's cards). The 10 s 2-event film, seed 7, B65:

| Attention | Denoise (a chunk) | Film | PSNR to half (mean / first s / last s) | SSIM mean | Seams (worst) |
|---|---|---|---|---|---|
| half (tile 0) | 252.3 s (16.82 s) | 321.5 s | - | - | - |
| half, tile 1 (the noise floor: the last bits) | 378.3 s (25.22 s) | 447.4 s | 25.5 / 32.7 / 14.7 dB | 0.819 | - |
| sage | 201.3 s (13.42 s) | 270.3 s | 22.2 / 25.0 / 18.1 dB | 0.775 | 1.10x |

A rolling film diverges chaotically from any change - half against itself with only the order of its running maximum
changed falls to 14.7 dB by its tenth second - and sage stays above that floor at the end; frame by frame the same scene
(the fox leaving the frame and coming back at the same moments), no seams.

**Speed** (the 60 s 4-event film, seed 7, the B65; NS_MR_PROFILE=1 drains the queue at each stage - the totals are
within 2% of an unprofiled run's):

| Part (s) | Before | After |
|---|---|---|
| denoise passes: attention | 1016.2 | 720.7 (sage) |
| denoise passes: the history's and the window's rows into the attention's keys ("history in") | 136.2 | 7.8 |
| denoise passes: modulate + qkv + RoPE / proj + MLP | 82.4 / 167.7 | 74.8 / 163.8 |
| encodes / recomputes | 134.9 / 28.7 | 101.4 / 19.2 |
| decode (middle 61, full-size levels 3 + 4: 221) | 450.3 | 448.6 |
| **the film** | **2019.0 (17.42 s a chunk)** | **1538.9 (12.10 s a chunk), 1.31x** |
| peak | 29.1 GiB | 29.5 GiB |

Unprofiled after: 1539.3 s (yesterday's unprofiled before: 1985.5 s - 1.29x); seams: joins 6.87 against 7.00 inside,
the worst 1.83x at join 87 (frame ~1409: the fox's legs passing close to the camera, the background continuous -
looked at). What did it: "history in" was the shared `ns_copy_2d` - a work-item a row, copying its 4 KiB a byte at a
time, 35,000 rows twice a block a pass - now the engine's `copy_rows` (16 bytes a work-item): 136 s to 8 s; sage over the
window and the history: the attention 1.41x. The decoder's convolutions no longer drain the queue and allocate their
carried frames every call - no change; RoPE rows built from per-axis angles - no measurable change (the host was not the
limit: "the rest" 0.4 s). Overlapping the decode with the denoising was not tried: both are the card's arithmetic (and
the music engine's second queue gained nothing).

## Round 4 (2026-11-24): the decode, a steered film's memory, the B70

**A steered film filled the card** after its decoder's keys grew (a live film sent an event): its use climbed ~1.3 GiB
a chunk from the growth on and the page's film failed at chunk 11 on the B65. The decoder's attention products (mm.cpp)
kept a oneDNN primitive for every shape - one a latent frame, its keys longer each time - and the grown keys' pitch
(51,030 rows: not a multiple of 8) made each primitive copy the values into a scratch of its own (~330 MB). The keys'
rows are a multiple of 64 now, mm.cpp keeps 64 shapes at most (emptied whole), and an added event's encode gives its
kernels' caches back. The same film after: 13 chunks, 217.6 s, peak 27.6 GiB, the card's use between chunks 24.5 -> 25.1.

**The decode.** oneDNN's verbose profile of a decode: the convolutions run at ~82 TFLOPS on the B65 (jit:ir, 17.8 ms each
at full size) - near the card's half rate; a full-size level took 325 ms a latent frame for 118 ms of convolutions. The
rest was the decoder's own elementwise kernels: the causal padding and the upsample an element a work-item with three
64-bit divisions each, the add an element a work-item, the norm a work-group of 64 with a barrier a voxel (1.7 million
voxels a frame at full size). Now 16 bytes a work-item with 32-bit positions (the padding, the add: the same bits), the
upsample's positions in 32 bits, the norm a sub-group a voxel up to 512 channels. A latent frame, B65: level 4 325 ->
202 ms, level 3 288 -> 217, level 2 222 -> 187, level 1 141 -> 134 (the 10 s film's decode 68.4 -> 51.6 s). The VAE
checks against the reference unchanged: the whole decode rel 1.36e-3 (1.4e-3 before), the stream 1.35e-3 (the same). The middle attention got faster too
(168 -> 86 ms a frame over a minute's keys): its values' pitch is aligned now.

**Before and after, both cards** (the 60 s 4-event film, seed 7; the 10 s 2-event film):

| Card | Film | Before (round 2) | After (round 4) | Speed-up | Peak | Seams (worst) |
|---|---|---|---|---|---|---|
| B65 | 60 s | 1985.5 s (17.36 s a chunk, decode 422 s) | 1416.7 s (12.11 s a chunk, decode 325.7 s) | 1.40x | 29.5 GiB | 1.41x |
| B65 | 10 s | 321.5 s (16.82 s a chunk, half) | 231.2 s (11.93 s a chunk, decode 51.6 s) | 1.39x | 26.9 GiB | - |
| B70 | 60 s | 1248.2 s (10.75 s a chunk, decode 280 s) | 944.2 s (7.89 s a chunk, decode 233.6 s) | 1.32x | 29.6 GiB | 1.67x |
| B70 | 10 s | 204.5 s (10.63 s a chunk, 1 event) | 154.4 s (7.79 s a chunk, decode 37.0 s) | 1.32x | 27.0 GiB | 1.10x |

**Next**, by size (B65, 60 s): the attention (~720 s: the window's queries over the 6 full-history blocks' keys are the
bulk; sage runs near half the card's int8 rate), the decode's convolutions (~240 s at ~82 TFLOPS: only int8 would go
faster) and its remaining elementwise passes (~70 ms a frame at the two larger levels: a norm fused into the padding
would save one pass of each), the MLP's products (164 s at ~87 TFLOPS).



**All Rust, no leftovers.** The runtime path - the model card, the Rust engine (`video/hunyuanvideo15`: movieroll.rs,
roll.rs, dit.rs, vae.rs, the text encoders' crates), the SYCL kernels (`kernels/video/hunyuanvideo15`, its silo, and the
silo's sycl-tla part built on Intel's ARK headers - Intel's, as CONTRIBUTING allows), oneDNN - has no Python and no other
third-party C++; ffmpeg is a program it runs for the mp4 and the playlist, as every video engine here. Removed: the
NS_MR_MEMLOG switch (its numbers are in the film's lines now: the plan's need and the card's free memory at the start, the
peak sampled every 5 ms), STEER_MAX_FRAMES and the steered growth factor. Kept as tools beside the runtime, as the other
ports keep theirs: `reference/movieroll/ref.py` (the official transformer on the CPU, now with `--anchor` for I2V),
`reference/hunyuanvideo15/` (ref.py and fetch.sh), and the engine's `check` job; NS_MR_PROFILE (a pass's stages and the
decoder's levels, for work like the above).

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

## Phase 0 - access, files, a reference (the plan)

1. **User**: accept MovieRoll's licence on huggingface.co with the account of hardac1's token (403 today).
2. Download to `/var/mnt/4TBSSD/nextsycl-models/` (741 GB free; the 2 TB has 78 GB): HunyuanVideo-1.5 `vae/`,
   `transformer/480p_t2v`, `480p_t2v_distilled`, `480p_i2v_step_distilled` (~105 GB), the I2V image encoder;
   MovieRoll `distilled/t2v`, `distilled/i2v`, `base` (~100 GB). Encoders from HunyuanImage 2.1's files where they
   are the same (check: Qwen2.5-VL-7B states at the same layer, the same byT5 glyph checkpoint).
3. Reference: no NVIDIA GPU here, and upstream wants FA3. Run the reference in PyTorch on the CPU (sdpa in place of
   flash attention) at tiny sizes - 128x224, 2 windows, a few steps - in a capped container (`--memory 48g`; the fp32
   DiT is 33 GB, the text encoder run separately first), dumping each stage. If the platform peer's NVIDIA box can
   run it at real sizes, ask it for dumps (its user's call).

## Phase 1 - HunyuanVideo-1.5 on the traditional video methods (the plan)

- New engine `video/hunyuanvideo15` from `video/example` (the video kind's contract), kernels in
  `kernels/video/hunyuanvideo15` + silo. Reuse: `crates/diffusion` (flow-match Euler, shift), HunyuanImage 2.1's
  Qwen2.5-VL + byT5 glyph encoders, H3's video plumbing (mp4 out, jobs), the XMX / Sage attention kernels.
- Order: VAE decode (bit for bit vs the reference on a latent) -> DiT one block -> full DiT one step -> T2V 480p 121
  frames end to end -> I2V (image encoder + concat condition) -> CFG-distilled and step-distilled cards -> VAE encode
  (I2V's first frame).
- Served through what exists: `video.generate` (prompt, image, seconds, size, seed, steps, cfg), `video.status /
  wait / cancel / timeline`; cards `hunyuanvideo-1.5-480p-t2v-distilled-q8`, `-i2v-step-distilled-q8`; vouch
  memory measured per card; merge-gate baselines on both cards. 720p and the SR upsamplers after.

## Phase 2 - MovieRoll on that engine (the plan)

- Same DiT code with MovieRoll's weights (`distilled/t2v`, `distilled/i2v`; base kept for checks): the rolling
  window scheduler (per-chunk noise levels, bidirectional in window / causal to history attention mask), the KV
  cache with HieraKV retention (6 full / 48 sink+recent), KV recompute (threshold 18, recent 3, event switch),
  context noise, AnnealCFG for base, the event list (one prompt per window; text encoded once per event), the
  streaming VAE decode of each finished chunk.
- Checks: one window against the reference dump, then a 3-window rollout (KV carried), then an event switch and a
  recompute; a 60 s film on both cards: time, peak VRAM, no seams at chunk joins.

## Phase 3 - the long-form endpoint (the plan)

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
