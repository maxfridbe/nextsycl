# ACE-Step 1.5 on nextsycl

ACE-Step 1.5 (`audio/acestep`, model cards `acestep-1.5`, `acestep-1.5-base`) makes songs from a description and lyrics, or
from a recording (cover, repaint, edit): a 2B flow DiT over the Oobleck VAE's latents (25 a second, 48 kHz stereo). The
stages and their checks against upstream's pipeline are `audio/acestep/src/check.rs` (dumps by `reference/acestep/ref.py`).
This page records the VAE's speed-up (2026-11-18).

## The VAE: from 95% of a song's time to a few seconds

A 262 s cover (8 turbo steps) took 329 s on the B65, about 130 s of it encoding the source and 186 s decoding the song:
the VAE's 1-D convolutions ran on the shared kernels' plain loops (`nsd_conv1d`, `nsd_conv_transpose1d`: one work-item
an output sample, every input channel and tap read from memory), about 2% of the card's float32 rate.

They now run in ACE-Step's silo, `kernels/audio/acestep/silo/acs.cpp` -> `dist/silo/libnextsycl-acestep.so`:

- every convolution is one oneDNN primitive (float32 in, out and accumulated: `fpmath strict`), made once a shape and
  kept, its weight reordered once to the primitive's layout (started from MiniMax Music 3's decoder convolutions,
  copied: a card's state is locked for a whole call and emptied when the VAE goes, `ns_acs_release`, because the weights
  are keyed by their addresses);
- a transposed (upsampling) convolution is made as its `stride` output phases: each phase is a plain convolution with
  the taps `k = r + stride j`, all phases one convolution (Ci -> stride x Co channels, 2 taps) into the free activation
  buffer, then interleaved. oneDNN's own deconvolution ran these shapes at 0.8-1.3 TFLOPS against 7.7 for the plain
  convolutions; as phases they take a third of the time;
- the scratchpad and the weights' copies are given back at the end of each encode and decode (`ns_acs_trim`), so the
  DiT's phase does not carry them.

Without the silo (or `NS_ACE_VAE_SILO=0`) the VAE runs on the shared loops as before. The windows are unchanged: the
reference handler's (encode 30 s windows with 2 s of overlap each side; decode `NS_ACE_DECODE_CHUNK` / the card's
free memory: 384 frames on a B65 holding the engine, 64 of overlap each side), so the tiling is the reference's.

### Speed (seconds; 384-frame decode windows, posterior mean; the B65 is GPU 1, the B70 GPU 0)

| Card | Clip | Encode before | Encode after | Decode before | Decode after |
|---|---|---|---|---|---|
| B65 | 60 s (3 encode / 6 decode windows) | 31.30 | 1.52 | 41.17 | 2.27 |
| B65 | 262 s (11 / 26 windows) | 137.73 | 5.29 | 186.27 | 7.06 |
| B70 | 60 s | 16.92 | 1.51 | 22.87 | 2.01 |
| B70 | 262 s | 74.52 | 4.04 | 103.36 | 5.08 |

With oneDNN's own deconvolution for the upsampling (before the phases) the decode of 262 s took 8.90 s (B65) / 6.31 s
(B70).

The whole cover (262 s source, new lyrics, 8 turbo steps, seed 7, the planner loaded, the card's decode window: 384),
B65: **327.1 s before, 15.4 s after** (encode 137.8 -> 5.4, prompt 0.6 -> 0.5, flow 2.4 -> 2.4, decode 186.3 -> 7.1).

The first window of each length makes its primitives (a few hundred ms); a served engine keeps them (up to 512).

### Profile

`NS_ACE_VAE_PROFILE=1` waits for and times each operation by kind (the `vaebench` stage prints the table). Before, the
convolutions were 99.5% of the time (a k7 128->128 at 48 kHz: 370 ms a window; the transposed ones 200-240 ms); snake,
residual adds, uploads and read-backs together under 1%, host work nil (timed operations 41.12 s of a 41.14 s decode).
After (60 s, B65, decode 2.84 s with the waits): the three k7 128->128 convolutions at 48 kHz 22 ms each (7.7 TFLOPS,
about 63% of the B65's float32 peak), the transposed ones 15-16 ms, snake 4%, residual adds 2%. The VAE is now bound by
float32 arithmetic.

`NS_ACE_VAE_FPMATH=tf32` or `bf16` (oneDNN computing in a narrower type) was tried: decode 7.0 -> 5.9 s, encode no
faster, latents at 29.8 dB SNR and the sound at 45.7 dB against float32 - not taken; strict float32 is the default.

### Exactness

Against the shared loops (the same build, `NS_ACE_VAE_SILO=0`), the whole 262 s song (B65; the B70's figures the same):

| | max abs diff | mean abs diff | SNR | spectra (dB, bins within 80 dB of the peak) |
|---|---|---|---|---|
| encode, latents [6561, 64] | 2.3e-4 | 2.6e-6 | 105.5 dB (cos 1.000000000) | - |
| decode, sound [2, 12.6 M] (from the same latents) | 1.8e-5 (below a 16-bit step, 3.1e-5) | 1.5e-7 | 115.3 dB | mean 0.0001, max 0.01 |

At the seams: the latents within 8 frames of an encode window's edge differ by at most 2.8e-5 (elsewhere 2.3e-4); the
sound within 4 frames of a decode window's edge by at most 3.3e-6 (elsewhere 1.8e-5); the largest step across a seam
is 0.64x the largest within a frame either side (no click).

Against upstream (diffusers' `AutoencoderOobleck` on the CPU, float32, the checkpoint's VAE; the first window of the
60 s clip - the encode of [0, 28 s), its 650 core frames; the decode of latent frames [0, 320), its 256-frame core):

| | encode SNR (max abs diff) | decode SNR (max abs diff) |
|---|---|---|
| before (shared loops) | 102.2 dB (2.3e-4) | 107.1 dB (1.5e-6) |
| after (silo) | 103.0 dB (1.9e-4) | 107.2 dB (1.2e-6) |

The silo is as close to upstream as the loops were.

**A whole song made twice is not always the same song.** The 262 s cover with the same seed on the same build came out
different between some runs (waveform SNR about -1 dB, spectra 10 dB apart) - the old build's two runs, and one of
three of the new one's (the other two bit-identical) - and two 30 s text2music runs (planner off) differed too (19.9
dB); 60 s covers, planner loaded or not, were bit-identical twice. The VAE alone is bit-for-bit the same run to run (the
vaebench runs). So the end-to-end comparison is of speed only; the VAE's exactness is the tables above.

### VRAM

A pass's buffers, as before (the engine's scratch, given back after the pass): decode, three activations of 128 channels
at the audio rate - 1.06 GiB for a 384-frame window (1.41 for 512); encode, the same for a 30 s window - 2.07 GiB.
During a pass the silo adds its copies of that half's weights in oneDNN's layouts (encoder 0.31 GiB; decoder 0.50 with
the phases') and a scratchpad of about one activation (0.69 GiB at a 30 s encode window), all given back at the pass's
end (`ns_acs_trim`; the next pass reorders the weights again, milliseconds). So, above the loaded engine (6.3 GiB on the
card without the planner, 9.7 with it): encode ~3.1 GiB (was ~2.1), decode ~2.3 GiB (was ~1.1), nothing between
passes. The encode's 30 s window is taken only when that fits in what the card has free (else the reference's 15 s
window, as a smaller card's); keep 1.5 GiB free beside it.

### Measuring it

```
nextsycl audio check OUT --model acestep-1.5 --stages vaebench --source SONG.wav [--against EARLIER-OUT] [--gpu N]
```

encodes the song (the posterior's mean), decodes it (the earlier run's latents when given, so both decodes see the same
input), writes `latents.f32`, `wav.f32` and `bench.json` to OUT and compares with the earlier run. The A/B's other side
is the same build with `NS_ACE_VAE_SILO=0`. Upstream's side: `reference/acestep/vae_ref.py` (the checkpoint's VAE on
diffusers, CPU; `python -I`).

### Open

- The run-to-run difference of a whole song (above): not the VAE's; intermittent, before and after this change; not
  chased here (something outside the VAE varies from run to run; text2music, which has no encode, shows it too).
- Bigger decode windows or one pass would save the overlap (a third of the decode at 384 frames, ~2 s of 7), but change
  the sound at the seams against the reference's tiling; one pass of 262 s would need ~19 GiB. Left as the reference
  tiles.
