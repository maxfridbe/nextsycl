# VibeVoice-1.5B: port brief

VibeVoice-1.5B (Microsoft, August 2025) makes long-form, multi-speaker speech in English and Chinese at 24 kHz: a
script of up to four speakers, each speaker's voice cloned from a recording without its transcript. nextsycl runs it
as the `audio/vibevoice` engine (arch `vibevoice`).

## Sources

- **Upstream checkpoint**: https://huggingface.co/microsoft/VibeVoice-1.5B (revision c00898d2), bf16 safetensors
  shards, MIT.
- **Upstream code**: Microsoft withdrew its GitHub repository in 2025. The code lives on in two places.
  - https://github.com/vibevoice-community/VibeVoice (commit 07cb79f): `modular/modeling_vibevoice_inference.py` is
    upstream's generate loop.
  - transformers 5.19 `models/vibevoice` and `models/vibevoice_acoustic_tokenizer`: the Microsoft team's port, with
    the same weights renamed in https://huggingface.co/vibevoice/VibeVoice-1.5B-hf (revision d4cd82eb).
- **GGUF**: https://huggingface.co/gguf-org/vibevoice-gguf (revision 197bcd95), the q8_0 and f16 files.
  - It keeps the checkpoint's own tensor names and has no metadata (`general.architecture` is `pig`).
  - The q8_0 file quantizes only the language model's 196 matrices. The acoustic and semantic tokenizers, the
    connectors, the diffusion head and the token embedding stay BF16 (a few small tensors F32).
  - Neither file holds a tokenizer or a configuration, so the card takes `tokenizer.json` (Qwen2.5's) and
    `config.json` from the -hf checkpoint. The configuration gives the eps, RoPE theta, heads, ratios, depths,
    vae_std and token ids. The engine reads the rest from the tensors' shapes.

## License and responsible use

- The weights, the GGUF and upstream's code are MIT.
- Microsoft's model card restricts use: no voice impersonation without the speaker's consent, no disinformation or
  passing synthetic speech off as real, and disclose AI-generated audio where it is shared.
- It is meant for research and development.

## Architecture

| part | shape |
|---|---|
| language model | Qwen2.5-1.5B: 28 layers of 1,536; 12 query / 2 key-value heads of 128 with q / k / v biases; SwiGLU 8,960; RoPE θ 1e6; ε 1e-6; the output head tied to the 151,936-row embedding |
| acoustic tokenizer | σ-VAE: encoder and decoder of ConvNeXt-style 1-D blocks; ratios 2, 2, 4, 5, 5, 8 (3,200 samples a frame: 7.5 Hz); depths 3 x 6 + 8; channels 32 to 2,048; latents of 64; every convolution causal |
| semantic tokenizer | the same encoder with 128 outputs (no decoder) |
| connectors | Linear (+ bias), RMS norm (ε 1e-6), Linear (+ bias): acoustic 64 → 1,536, semantic 128 → 1,536 |
| diffusion head | 4 adaLN layers of 1,536 (RMS norm, shift / scale / gate from SiLU(c), SwiGLU 4,608), a final adaLN layer to 64; c = cond_proj(hidden) + timestep MLP(sinusoids of 256) |
| latents | speech_scaling_factor and speech_bias_factor (scalars in the file) |

A ConvNeXt block is:

- RMS norm over channels, a depthwise causal convolution of 7, and the layer scale γ;
- RMS norm, then Linear(C → 4C), exact GELU, Linear(4C → C), and the layer scale.

## Prompt and generation (upstream's)

- **The prompt** is the -hf chat template (upstream's processor builds the same string).
  - It starts with the system line, then ` Voice input:\n`.
  - Each voiced speaker follows as ` Speaker N:<audio start><audio> x frames<audio end>\n`.
  - Then ` Text input:\n` and the script, ` Speaker N: text\n` a turn.
  - It ends with ` Speech output:\n<audio start>`.
  - Script speakers numbered from 1 are made 0-based.
- **A voice's recording** is resampled to 24 kHz and brought to -25 dBFS (and below full scale). It is zero-padded to
  whole frames and run through the acoustic encoder.
  - Upstream's `encode(sample=True)` then adds noise: a std of 0.625 times a normal draw for the recording, times a
    normal draw a value.
  - The latents are shifted by the bias, scaled, and passed through the acoustic connector. They replace the audio
    tokens' rows.
- **Each step**, the language model's next token is the argmax of its raw logits over four ids: audio start, audio
  end, audio, and end of text.
- **An audio token** makes one frame:
  1. The negative branch steps. It starts from an audio start alone, then reads the same inputs as the positive one,
     but only on audio steps.
  2. The diffusion head draws a latent from N(0, I) in 10 DPM-Solver++ steps (2M midpoint, v-prediction,
     squaredcos_cap_v2 betas, linspace timesteps 999 to 100, first order at both ends). Guidance is cfg 1.3 between
     the two branches' last hidden states.
  3. The latent is unscaled and run through the streaming acoustic decoder, giving 3,200 samples.
  4. The semantic encoder (streaming) reads those samples back.
  5. The next input is acoustic_connector(latent) + semantic_connector(features).
- **The other tokens**:
  - An audio end clears both streaming caches.
  - An audio start restarts the negative branch: its audio start at position 0, the next input at 1.
  - The end of text ends the run.
- **Bounds**: upstream takes at most twice the prompt's length in steps. That cuts a long script read without voice
  recordings short, because speech takes about 2.4 frames a text token. The engine allows four times the prompt's
  length, and the request's `seconds` bounds it as well.

Transformers' `model.generate` differs from upstream in one way. Its generation config's `guidance_scale` 1.3 adds
an `UnbatchedClassifierFreeGuidanceLogitsProcessor` to the token choice, which upstream does not have. The two then
end a frame or two apart, while the frames they share are bit-identical. The engine and the reference follow
upstream.

## Port

1. **Weights**: each part keeps the file's values exactly where it can.
   - The language model's matrices stay Q8_0 on the card: int8 codes [N, K] and the block scales [N, K / 32] in half,
     split at load (1.06 bytes a weight: the decode step reads 1.39 GB, not 2.6 GB of half). The f16 file's stay half.
   - The tokenizers' matrices and convolutions are the file's bfloat16 (a few small float32 ones as float32).
   - The diffusion head and the connectors are expanded to half (BF16 to half is exact here, apart from subnormals).
   - The token embedding stays on disk and is read a row a token. Of the tied head, only the four tokens' rows are
     on the card.
2. **Language model** (`lm.rs`): a decode step (`Lm::decode`: one row of one sequence, or the paired step's two rows,
   each its own cache) runs on the engine's own kernels (`kernels/audio/vibevoice/vvlm.cpp`), its positions read from
   device memory:
   - a layer is 8 launches: RMS norm, q / k / v product (+ biases), RoPE + key / value store for both rows in one
     launch, the attention split over the keys in a fixed number of pieces (64 keys a piece when the cache is full; the
     GPU kept busy at any length) and their merge, the output product added in, RMS norm, the gate / up product with
     SwiGLU in its epilogue (the gate and up rows interleaved at load), the down product added in.
   - The products are the engine's small-batch product (a copy of mm3's, CONTRIBUTING rule 6) for half, bfloat16, Q8_0
     and float32 weights, up to 20 rows, the K reduction split between a work-group's sub-groups for matrices of few
     rows (chosen by shape from `check --stages bench`).
   - **Paired step**: both branches read the same input row, so one pass of two rows reads the weights once for both
     (`Lm::pair`). The negative row is used if the next token is audio; otherwise its position is written again later.
   - The prompt runs in passes of 256 rows on oneDNN's half GEMM (a Q8_0 matrix expanded to half for it), the shared
     RoPE and this engine's prompt attention (`vv.cpp`).
3. **Diffusion head** (`head.rs`, `vvhead.cpp`): the two branches are two rows. The adaLN modulations depend on the step
   only through its timestep, so a frame's conditions for all ten steps (silu(cond + temb), 20 rows) go through the
   14 modulation matrices in one product - read once a frame, not ten times. A step is then the input product, four
   layers of norm / gate-up with SwiGLU / down with the gated residual in its epilogue, the final norm and output, and
   DPM-Solver++'s step on the device (`sched.rs`'s coefficients, the same arithmetic). Nothing is read back: the last
   step writes the latent where the acoustic connector and (unscaled) the decoder read it.
4. **Tokenizers** (`vae.rs`, `vvcodec.cpp`): one streaming coder for the encoder and decoder alike, over time rows [L, C].
   - Every convolution reads a buffer of its carried rows followed by its new input ([ctx; x]: the stage before writes
     its rows there), so a window of k rows is k * C consecutive floats and the convolution is a product over windows:
     a strided one's windows start s rows apart (weights [co, k * ci]); a transposed one of kernel 2s makes s output
     rows from each pair of input rows (weights [s * co, 2 * ci]). Its carried rows then move to the front.
   - A ConvNeXt block is two launches and its MLP's two products: the first norm into the depthwise convolution's
     carried rows (two buffers that alternate), then a row at a time the depthwise convolution, the layer-scaled
     residual and the second norm; GELU and the layer-scaled residual ride in the products' epilogues.
   - A product of up to 8 rows (the low stages, where the weights are) is the small-batch product; more rows run on the
     matrix engines (joint_matrix, 64 x 64 a work-group) with x split into two bfloat16 halves (hi, x - hi) against the
     exact bfloat16 weights, summed in float32: float32-accurate (the decoder at 2-3e-6 of the reference) at the
     engines' rate; K split over work-groups when the output has few tiles.
   - A frame decoded alone is what the whole clip would give. A voice runs through the encoder eight frames a piece.
5. **Requests**:
   - The contract's `AudioRequest` gains `references` (the other speakers' recordings) and `Speech` gains `speakers`
     (4 here).
   - CLI: `--ref-audio` once a speaker, or `--voices NAME,NAME`.
   - Audio server and front door: `voices` (saved voices) or `ref_audios` (recordings), a speaker each. The front door
     routes them to a model whose card lists speech `speakers`.

## Parity

`reference/vibevoice/ref.py` runs the transformers implementation in float32 on the CPU and writes a dump.

- Its loop is checked against `model.generate` with the same seed: the frames they share are bit-identical (max
  |difference| 0.0).
- The Whisper transcripts of the reference's own clips are word for word.

`nextsycl audio check DUMP --dir DIR` compares each stage from the reference's own inputs. Four dumps were used:

- **plain**: one speaker, no voice, 19 frames.
- **two**: two speakers with the en-Alice and en-Frank voices, 251 voice frames, 49 frames.
- **four**: the same voices, the four-turn dialogue, 91 frames.
- **zh**: a Chinese two-speaker script with the en-Maya and en-Carter voices, 47 frames. The model closed the first
  speaker's audio and opened a new segment between the turns (audio end, audio start). That exercises the negative
  branch's restart and the streams' reset.

Numbers are relative RMS errors, measured 2026-10-18 after the speed work (Speed below; the numbers before it are kept
where they differ).

| stage | Q8_0 (served) | f16 file |
|---|---|---|
| token ids | identical (43 / 324 / 486, Chinese included) | identical |
| voice latents (acoustic encoder) | 2.1e-6 to 9.4e-6 (before: 3.4e-4 / 1.0e-3) | about the same |
| voice rows (connector) | 3.0e-4 | 3.0e-4 |
| prompt rows | 3.0e-4 (0 without voices) | 3.0e-4 |
| logits, teacher forced (21-93 steps) | 2.2-2.6e-3; the best of four agrees on every step | 1.9-4.9e-4; every step |
| hidden state, positive / negative | 1.4-1.6e-2 / 1.9-2.1e-2 | 0.7-2.2e-3 / 4.9-6.3e-4 |
| diffusion head: first output / latents (10 steps, the reference's conditions and noise) | 4.2-4.3e-7 / 3.7e-7 to 5.1e-6 | the same |
| acoustic decoder (speech) | 2.2-3.0e-6 (before: 1.5-2.4e-4) | 1.0-1.1e-6 |
| semantic features | 3.7e-6 to 7.7e-5 (before: 7.4e-4 to 3.8e-3) | 1.0e-6 |
| next inputs (connectors) | 3.0e-4 | the same |

What the numbers show:

- **Q8_0** accounts for the language model's 1e-2: the f16 file is at 2e-3.
- **The diffusion head** runs at float32 precision (its BF16 weights are exact in half).
- **The tokenizers** run on the file's exact bfloat16 weights with float32 activations (the matrix-engine products on
  x split into two bfloat16 halves): 2-3e-6 against the reference, 50-100x closer than the first port's half GEMMs. The
  f16 file's tokenizers are F16, kept as float32 (the plain tiled product): 1e-6.
- **The paired step** (`Lm::pair`) holds the same numbers as single steps: with f16, logits 4.2e-4 and hidden
  2.2e-3 / 5.0e-4.
- **The free run** (`--stages run`, with the reference's noise) departs from the reference gradually, as an
  autoregressive sampler does.
  - With the f16 file it is 1e-2 off by frame 4 and 1e-1 off by frame 20.
  - Its clip still says the script word for word.
  - The reference's clip and the engine's run transcribe identically (Whisper small): "Welcome back to the show.
    Today we talk about gardens. Thanks for having me. I love tomatoes."
  - The Chinese run transcribes as the script, apart from Whisper writing it in traditional characters and a
    homophone (戴 for 带).

## Clips

On the B70, from the served Q8_0 file:

- A plain voice, a single clone (en-Maya), a two-speaker dialogue (Alice / Frank) and a 46.5 s two-speaker show
  (Carter / Maya) all transcribe word for word with Whisper small.
- Through the audio server, these also transcribe correctly: a plain script, a `ref_audio`, a saved voice without a
  transcript, `voices` of two saved voices, and `ref_audios` of two recordings.
- In five seeds of the four-turn dialogue, the speakers alternate as cast. The median pitch a second is 200-290 Hz on
  Alice's turns and 95-140 Hz on Frank's. The reference shows the same pattern.
- After the speed work (2026-10-18): the four clips (seed 7) transcribe word for word on the B70; on the B65 too but
  for a word or two ("it to spring", a garbled turn start - the first port's B70 clip of the show had the same).
  Twelve seeds of the 46.5 s show: mean word error 1.2%, 10 of 12 within 3% (the first port: 2.3%, 9 of 12; in both a
  seed now and then repeats a phrase or garbles a turn's first words - the model's own, the runs vary from run to run in
  either build). The dialogue's speakers alternate by pitch as before (300 / 208-244 Hz on Alice's turns, 145-150 on
  Frank's first).

## Speed

Measured 2026-10-18 (the speed work's branch) against the first port (2026-10-17), the same commands on the same cards:
`nextsycl audio gen`, the long show (seed 7: about 350 frames, 46.5 s of speech), and `nextsycl audio serve` (each
request three times after a warm-up; the wall time of the HTTP request).

| | B70 before | B70 now | B65 before | B65 now |
|---|---|---|---|---|
| frames a second, the 46.5 s show | 44.0 | 79.3 | 39.3 | 64.0 |
| real time | x5.9 | x10.6 | x5.2 | x8.5 |
| frames a second, plain / clone / two-speaker clips | 40 / 30 / 41 | 79 / 79 / 79 | | 65 / 64 / 64 |
| server: a short sentence (2.7 s of speech) | 0.45 s | 0.31 s | 0.50 s | 0.38 s |
| server: a short clone (`ref_audio`, 3.2 s of speech) | 1.76 s | 1.55 s | 1.90 s | 1.70 s |
| server: the 46.5 s show (`ref_audios` of two) | 8.85 s | 5.9 s | 10.3 s | 7.1 s |
| on the card | 4.9 GiB of weights, 5.6 GiB peak | 3.6 GiB of weights, 4.0 GiB peak | | |
| load (file in the page cache) | 4 s | 2 s | | |

### A frame on the B70, and the floor

`NS_VV_PROFILE=1` (the GPU waited for at each stage's end) and unitrace's device time a kernel; the floor is the
weights a frame reads at the B70's 608 GB/s (the largest products run at 560-630 GB/s).

| stage | before | now | launches now | bytes a frame | floor |
|---|---|---|---|---|---|
| language model, the paired step | 10.46 ms | 3.92 ms | 253 | 1.39 GB (Q8_0; was 2.6 GB of half) | 2.29 ms |
| diffusion head, 10 steps | 5.79 ms | 3.91 ms | 163 | 1.78 GB (the modulations once a frame) | 2.93 ms |
| acoustic decoder | 3.40 ms | 2.29 ms | ~190 | 0.37 GB (bfloat16) | 0.6 ms |
| semantic encoder | 2.94 ms | 2.32 ms | ~190 | 0.38 GB | 0.6 ms |
| read-backs, connectors | 0.08 ms | 0.08 ms | 8 | | |
| a frame | 22.7 ms | 12.6 ms | ~800 (was ~1,300 and 12 host waits) | 3.9 GB | 6.4 ms |

The GPU is busy 12.0 ms of the 12.6 (one host wait a frame: the next token's four logits). The steps, in order of
payoff (each commit has its numbers):

1. **The language model's decode step on its own kernels** (44.0 -> 64.0 frames a second): Q8_0 kept on the card (half
   the bytes), RoPE and the key / value store in one launch (the shared RoPE took 16 us a call, a work-item a head:
   1.9 ms a frame), the attention split over the keys in pieces (it ran 2-4 work-groups: 1.3 ms a frame), SwiGLU in
   the gate / up product.
2. **The head without read-backs** (-> 70.4): every step's modulations in one product (the 14 matrices read once a
   frame, not ten times), DPM-Solver++ on the device, the gated residual in the down product; the samples read back
   256 frames at a time (each frame had 12 host waits: owned buffers' "async" writes waited).
3. **The tokenizers on their own kernels** (-> 75.2): every convolution a product over windows of [carried rows; input]
   (no transposes, no oneDNN: each oneDNN call cost tens of microseconds of host time), a block in two launches and its
   MLP's products, the file's exact bfloat16 weights, many rows on the matrix engines with x split into two bfloat16
   halves (float32-accurate: the tokenizers 50-100x closer to the reference than before).
4. **Tuned products** (-> 79.3): the K split by shape from a bench that sweeps every layer's matrices (`audio check
   --stages bench`), 4 output rows a sub-group for 8+ rows of x, the matrix-engine tiles 64 x 64.

### What is left, and why it stays

- **The language model, 1.6 ms over its floor**: the small products run at 380-420 GB/s (q / k / v 3.3 MB in 7.9 us,
  o 2.5 MB in 6.6 us: 0.25 ms a frame over the rate of the big ones), the norms 56 launches of 4.7 us (0.26 ms), the
  attention 0.33 ms, and the gaps between 253 launches. The norm inside the products (each work-group norming its
  rows into local memory) was tried: slower (3.99 -> 4.87 ms), each work-group's own reduction costs more than the
  launch it saves.
- **The head, 1.0 ms over**: its modulations' product of 20 rows runs at 270 GB/s (0.25 ms where 0.11 would read its
  66 MB; on the matrix engines it was slower, 381 us), norms 0.24 ms, gate / up and down at ~555 GB/s. Its weights in
  Q8_0 would halve the bytes but move the head from 4e-7 to ~1e-3 of the reference: not taken (exact before fast).
- **The tokenizers, 3.4 ms over their weights**: they are latency-bound - the matrix-engine products of the high
  stages (68 calls and 42 merges of K pieces, 1.2 ms), the blocks' mixing kernels (0.6 ms), 8-row products (0.3 ms) and
  ~420 launches. A whole ConvNeXt block (norm, depthwise, MLP, residual) in one kernel for the narrow stages would save
  perhaps 1 ms; a large rewrite for 8% of a frame.
- **SYCL graphs**: the GPU idles 0.6 ms of the 12.6 (one host wait a frame); graphs would save part of that and of the
  launch gaps (est. <= 0.5 ms, 4%). Recording needs every kernel's inputs in device memory (the decode positions
  already are) and two graphs for the alternating buffers.
- **Overlap**: none to be had - the decoder's samples feed the semantic encoder, whose features feed the next language
  model step, whose hidden state conditions the head, whose latent feeds the decoder. Speculating the next token
  (audio, most often) would need every stream's state rolled back on a miss.
- **Outside the engine**: through the server a request with recordings spends ~0.5 s a MB of request body outside the
  engine (the short clone: 1.55 s through the server, 0.55 s in the engine - 0.15 s encoding the voice, 0.07 s the
  prompt, 0.3 s the frames): the server's request handling (glue/serve), not this engine's.

Debug switches: `NS_VV_PROFILE=1` (stages, and the prompt's voice encoding and prefill), `NS_VV_PROFILE=2` (the
tokenizers' operations one by one), `NS_VV_GEMV_KS`, `NS_VV_GEMM_SIMT` (docs/environment.md).
