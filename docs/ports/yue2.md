# Porting YuE2 (M-A-P YuE2-3B) to nextsycl

YuE2 turns a style prompt and lyrics into a whole song with vocals, 48 kHz stereo; non-commercial (CC BY-NC 4.0). This
engine is a port of nextcuda's (`audio/yue2`, its `docs/ports/yue2.md` at f6f57a5 records the model and the protocol: what
is said there about the prompts, the phases and the VAE holds here) from CUDA to SYCL on the Arc Pro B70 and B65.

## Status: not yet checked against a reference

**Rule 5 is open.** `reference/yue2/ref.py` needs upstream's wheel, which is not run on this box; its dumps have not arrived. The
stage checks (`nextsycl audio check DIR --stages tokens,ar,neg,noise,vel,steps,solve,vae`, and SheetSage2's) are ported
and have never been run here. Everything below is the engine against itself and real songs judged by ear-free measures.
The silo was therefore written before parity: it is checked against the engine's own kernels, not against a reference.

| Claim | State | How |
|---|---|---|
| prompts, tokenizer | unproved here | no reference; Whisper hears the sung words |
| AR logits, NAR velocities, VAE vs upstream | unproved here | needs the dumps |
| SheetSage2 vs upstream | unproved here | ran end to end on a song of ours (a score, a cover) |
| kernels run, no GPU reset | proved | every run below; reset count stayed 10 |
| same request, same card: same WAV | proved | 8 s song twice, abc, codes, latents and samples identical (once the flow heads were fixed) |
| silo = engine's kernels | proved | 81 s song, `NS_SILO=0` vs the silo on the final build: 0 samples differ |
| fast arithmetic vs float32 | measured | below |
| real songs in English and Mandarin | measured | below; not listened to |

## What changed from nextcuda

| nextcuda | here | why |
|---|---|---|
| bf16 product inputs, float accumulation | half | oneDNN's product keeps the input type here, so bf16 means 8-bit results (table below) |
| flash attention with grouped query heads | the shared `attention_qk` (flash), the 16 query heads regrouped as 2 rows of 8 (`ns_audio_yue2_heads`) | the shared kernel's heads are one to one |
| device cos / sin of float32 angles | a host table of the same angles | audio kernels build with fast math |
| float32 GEMM for the flow heads | `ns_audio_yue2_linear_f32`, one fixed sum order | oneDNN's float32 GEMM changes its order each call; two runs' latents differed by rel 5.6e-3 |
| nsd's conv1d | MiniMax Music 3's oneDNN convolutions | nsd's is a plain loop here: 26.2 s to 2.4 s for 81 s of song, samples within 1 LSB |
| int8 ConvRot flow | none | nextsycl has no `int8_prepare`; the flow is already fast (below), nothing needs it |
| two-card split by the card (`split: true`) | the front door starts one copy a card; `audio gen --gpus A,B` splits | nextsycl's cards have no split key |

## The silo (kernels/audio/yue2/silo)

A Q8_0 block leaves its quants 2-aligned, so the engine's kernel reads them two bytes at a time. The silo repacks each matrix but
the embedding table at load (lossless), then reads aligned 16-byte vectors: the same sums in the same order, and expands
the whole matrix to half for a prompt or flow pass. Without it (`NS_SILO=0`) the matrices stay as stored.

| Decode kernel, B70 | engine's | silo |
|---|---|---|
| q | k | v, 2 rows | 70 us | 18 us |
| gate | up, 2 rows | 213 us | 51 us |
| down, 2 rows | 97 us | 25 us |

| 81 s song, seed 7, B70 + B65 | score | codes, tg | flow | VAE | total |
|---|---|---|---|---|---|
| nsd's convolutions, no silo | 17.4 s | 70 codes/s | 7.5 s | 26.2 s | 80 s |
| oneDNN convolutions, no silo | 17.5 s | 70 codes/s | 7.5 s | 2.4 s | 56 s |
| with the silo | 5.3 s | 217.6 codes/s | 7.3 s | 2.4 s | 24 s |

## Self-check (`nextsycl audio check DIR --stages self`)

No reference needed: the fast arithmetic against the slow one (every product float32 from the Q8_0 blocks, each attention
summed in float32). It sees rounding and kernel faults; it cannot see a structural error both paths share. 257 prompt rows, 202 flow
rows, one layer stack each.

| rel L2 against float32 | half, B70 | half, B65 | bf16, B70 |
|---|---|---|---|
| prefill logits | 1.78e-4 (silo 2.30e-4) | 1.86e-4 | 1.56e-3 |
| velocity | 9.54e-4 (silo 8.67e-4) | 8.68e-4 | 4.16e-3 |
| shared attention alone, in the velocity | 3.99e-4 | 3.71e-4 | 3.81e-4 |

Cosines are 1.000000 for half, 0.999991 at worst for bf16. The decode path's logits equal the float32 prefill's exactly
(rel 0). The velocity twice in one process is identical. Prefill twice in one process differed once on the B70 (rel 1e-4)
and not on the B65, unexplained; two whole runs are identical.

## Songs

Audio checks by a standard-library script (duration, peak, RMS, non-silence, spectrum) and Whisper large-v3-turbo. For scale,
MiniMax Music 3 takes on this box have a centroid of 546 to 608 Hz and a median flatness of 0.0005 to 0.004.

| Take | Length | Peak | RMS dBFS | Non-silent | Centroid | Flatness | Whisper |
|---|---|---|---|---|---|---|---|
| English folk, full | 81.0 s | 0.922 | -16.8 | 99% | 335 Hz | 0.0000 | every line ("along" heard as "alone") |
| English rock, full | 162.6 s | 1.000 | -15.7 | 99% | 758 Hz | 0.0002 | every line from verse 1's end on; verse 1's start not heard |
| Mandarin ballad, full | 142.1 s | 0.996 | -16.4 | 99% | 477 Hz | 0.0000 | every line of both verses and choruses |
| English folk, melody | 82.4 s | 1.000 | -15.5 | 100% | 305 Hz | 0.0000 | second verse line and the chorus |
| English folk, off (cfg 1.01) | 63.2 s | 0.971 | -16.0 | 100% | 410 Hz | 0.0000 | the chorus only |
| English folk, cfg 1.2 | 81.5 s | 1.000 | -16.3 | 100% | 366 Hz | 0.0000 | every line |
| Same score, new words | 82.1 s | 1.000 | -15.1 | 99% | 374 Hz | 0.0000 | new words in the same places; last line not heard |
| Cover of the rock take, synthwave | 163.7 s | 1.000 | -16.1 | 99% | 820 Hz | 0.0003 | as the rock take |

Whisper invents text over instrumental passages ("University of Minnesota", a composer's credit); those lines are not counted.

## Speed

Token stages are tg; a whole song as real-time factor (seconds of song a second). nextcuda's published numbers are an RTX 5070 + RTX 3060 (224 s song in 72.5 s).

| Song | Cards | Score | Codes tg | Flow, 32 steps | VAE | Total | Real time |
|---|---|---|---|---|---|---|---|
| Mandarin 142.1 s | B70 + B65 | 4.1 s | 207.3 codes/s | 14.4 s | 3.2 s | 39.0 s | 3.6x |
| Mandarin 142.1 s | B70 | 4.1 s | 207.3 codes/s | 14.4 s | 2.9 s | 38.8 s | 3.7x |
| Mandarin 142.2 s | B65 | 5.9 s | 152.1 codes/s | 20.9 s | 3.3 s | 53.7 s | 2.6x |
| English 233.7 s | B70 + B65 | 12.5 s | 169.2 codes/s | 28.3 s | 4.6 s | 80.3 s | 2.9x |
| English 233.7 s | B70 | 12.5 s | 168.8 codes/s | 28.1 s | 3.8 s | 79.4 s | 2.9x |
| English 234.4 s | B65 | 17.4 s | 123.8 codes/s | 42.1 s | 4.9 s | 112.1 s | 2.1x |
| nextcuda 223.8 s | 5070 + 3060 | 11.7 s | 190 codes/s | 26.8 s | 4.2 s | 72.5 s | 3.1x |

The split gains nothing in time on these cards; it only moves memory. A guided song (cfg 1.2) draws at 157.4 codes/s. SheetSage2 transcribes 162.6 s in 8.3 s.

| Peak VRAM, 234 s song at the six-minute cache | First card | Second |
|---|---|---|
| B70 + B65 | 6.62 GiB | 2.83 GiB |
| one card, B70 / B65 | 8.13 / 8.14 GiB | |

No song here reached the 360 s cap, so its peak is not measured. `concurrency` and `coexistence` are not on the card: not measured.

## Known

- One seed makes different songs on the B70 and the B65 (3552 and 3555 codes for one request). The VAE agrees byte for byte across them.
- The score given back with the same words is a different take: a prompt pass and the decode round differently, and the codes are drawn from that.
