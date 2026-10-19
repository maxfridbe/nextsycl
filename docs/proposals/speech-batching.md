# Proposal: several clips at once in one speech engine (batched decode)

Status: **done for Breeze TTS 2 and S2 Pro** (2026-10-19): `audio/breezetts2/src/batch.rs`, `audio/s2pro/src/batch.rs` - one
thread a card advances every request in hand a frame at a time (the products over all rows, each row's own cache and
position, continuous batching); the audio server runs as many requests as the engine batches (`AudioEngine::batch`,
`NS_BZ2_BATCH` / `NS_S2P_BATCH`, 8). B70, `audio serve`, clips of 5-17 s, real time a second of wall time:

| engine, mode | one at a time | 2 together | 4 together | 8 together |
|---|---|---|---|---|
| Breeze, design (guided: 2 rows a request) | 1.61x | 2.58x | 3.68x | - |
| Breeze, plain | 1.83x | 3.58x | 5.55x | 7.63x |
| S2 Pro, sampled | 3.01x | 4.43x | 5.31x | - |
| S2 Pro, greedy | 3.1x | - | 6.6x | - |

Parity: a Breeze request's codes in a batch are its solo run's (every clip the same length, samples identical or 1 in
32,768 apart - the codec's own run-to-run rounding), and a solo run is the previous engine's to the bit. S2 Pro is not
repeatable run to run on its own (two solo runs of one greedy request differ, as two runs of the old build do); a
batched request matches its solo run as often as two solo runs match. Qwen3-TTS (its frame is one recorded graph of
one request), VoxCPM2 and VibeVoice: not batched yet.

## The measurement that prompted it

Two Breeze TTS 2 servers on one card, against one (2026-10-17, the same 8 clips of ~8 s, design mode, cards idle):

| card | 1 server | 2 servers on the card |
|---|---|---|
| B70 | 1.54x real time | 0.65x in total (each clip ~5x slower: 24 s against 5) |
| B65 | 1.43x | 0.68x |

With the second server loaded but idle the first ran at its usual 1.5-1.7x, so the loss is the sharing itself. The
xe driver time-slices the card between processes, and a speech engine is many small kernels with a host wait every
frame: each wait sits through the other process's slice. More workers per card is a loss, so the front door keeps
one speech server per card (the UI's per-card checkbox stays a checkbox, not a worker count).

## Where a frame's time goes

Breeze makes speech a frame at a time (12.5 frames a second of audio). Each frame reads the backbone's weights (28
layers of 2,048) once and the depth decoder's (12 layers of 1,024) fifteen times, to compute about one row of numbers
each step. The work is bound by reading weights from card memory, not by the arithmetic: the compute units mostly
wait. The cached-decode product the engine already uses (`ns_audio_mm3_gemv`) takes up to 8 rows and reads the
weights once for all of them, so 8 rows cost about what 1 does.

## The proposal

One engine process advances several requests by one frame in the same step:

```
now:      [clip A frame 1] [clip A frame 2] ... [clip A done] [clip B frame 1] ...
batched:  [A1 B1 C1 D1]    [A2 B2 C2 D2]    ...   each step ~ the cost of one clip's
```

- **Rows**: a request with an instruction (guidance) is already two rows (the guided and the plain branch), so a
  step holds 4 guided requests or 8 plain ones within the 8-row product.
- **Joining and leaving**: requests end at different lengths. A finished one leaves the batch, and a waiting one
  joins at the next frame after its prompt pass (continuous batching). Each keeps its own KV cache, position,
  sampler state (seed, temperature, top-k, repetition penalty) and cfg.
- **The codec**: decoding frames to the waveform stays per request after its last frame. It is a small share of a
  clip's time, and could move to a second queue.
- **Server**: `audio serve` accepts up to N requests at once (today: one, the rest wait on the engine lock). N comes
  from the model card (rows the kernel takes, memory for each request's cache) and the card's free memory. The
  front door and the UI need no change: a card's queue just drains faster.
- **Attention**: `ns_audio_mm3_attn` already takes a batch of caches of one length. Requests at different
  positions need per-row lengths (a ragged form), or caches padded to the batch's longest with a mask.

## Expected gain

An estimate from the product's shape, not measured: about 3-4x the audio per second from one card with four
guided requests in flight. A single clip takes about the same time as today. Memory: weights once (5.4 GiB), plus
~0.5-1 GiB of caches and buffers a request.

## Cost and risks

- About a few days in `audio/breezetts2` (frames() becomes a step over a set of requests) and `glue/serve/src/audio.rs`
  (a scheduler instead of the one-at-a-time lock; progress, cancel and the retire guard per request).
- Parity: every request in a batch must give what it gives alone (same seed -> same codes); a test runs a batch
  against the requests one by one.
- The same shape fits Qwen3-TTS (talker + code predictor) and VoxCPM2; Breeze first, then the others if it pays.

## Decision needed

Go / no-go, and whether Breeze alone or the speech engines as a set.
