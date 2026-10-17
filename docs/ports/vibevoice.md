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
- **Bounds**: a run takes at most twice the prompt's length in steps (upstream's), and the request's seconds.

Transformers' `model.generate` differs from upstream in one way. Its generation config's `guidance_scale` 1.3 adds
an `UnbatchedClassifierFreeGuidanceLogitsProcessor` to the token choice, which upstream does not have. The two then
end a frame or two apart, while the frames they share are bit-identical. The engine and the reference follow
upstream.

## Port

1. **Weights**: the GGUF's tensors are expanded to half on the host (F32, F16, BF16 and Q8_0 blocks).
   - BF16 to half is exact here, apart from subnormals.
   - The convolutions are float32, as oneDNN's convolutions take them.
   - The token embedding stays on disk and is read a row a token. Of the tied head, only the four tokens' rows are
     on the card.
2. **Language model** (`lm.rs`): it runs on shared kernels and one of its own.
   - Shared: the MiniMax Music engine's small-batch product (with the biases), its key / value store, and Breeze's
     table RoPE.
   - Its own: attention for 6 query heads a key head (`kernels/audio/vibevoice/vv.cpp`, a copy of mm3's: CONTRIBUTING
     rule 6). The prompt's wide products are oneDNN's half GEMM.
   - **Paired step**: both branches read the same input row, so one pass of two rows reads the weights once for both
     (`Lm::pair`). The negative row is used if the next token is audio; otherwise its position is written again later.
3. **Diffusion head** (`head.rs`): the two branches are two rows of the small-batch product (K padded to 256 for the
   64-wide input). The timestep embeddings are computed once a request. DPM-Solver++ runs on the host (`sched.rs`):
   64 values, read back after every step.
4. **Tokenizers** (`vae.rs`): one streaming coder for the encoder and decoder alike, over time rows [L, C].
   - The norms and the MLPs use the shared kernels and the codec's layer-scaled residual.
   - The depthwise convolution carries its last 6 rows (`ns_audio_vv_dwconv`, `ns_audio_vv_ctx_push`).
   - Each channel-mixing convolution carries k - s input rows (k / s - 1 for a transposed one). It is transposed
     around mm3's oneDNN convolution, and a transposed one is cut to its new outputs.
   - A frame decoded alone is what the whole clip would give.
   - A voice runs through the encoder eight frames a piece.
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

`nextsycl audio check DUMP --dir DIR` compares each stage from the reference's own inputs. Two dumps were used:

- **plain**: one speaker, no voice, 19 frames.
- **two**: two speakers with the en-Alice and en-Frank voices, 251 voice frames, 49 frames.

Numbers are relative RMS errors.

| stage | Q8_0 (served) | f16 file |
|---|---|---|
| token ids | identical (43 / 324) | identical |
| voice latents (acoustic encoder) | 3.4e-4 / 1.0e-3 | 3.5e-4 / 1.0e-3 |
| voice rows (connector) | 3.0e-4 | 3.0e-4 |
| prompt rows | 3.0e-4 (0 without voices) | 3.0e-4 |
| logits, teacher forced (51 steps) | 2.2-2.3e-3; the best of four agrees 51 / 51 | 4.2-4.8e-4; 51 / 51 |
| hidden state, positive / negative | 1.4-1.6e-2 / 1.9-2.1e-2 | 2.2e-3 / 5.0e-4 |
| diffusion head: first output / latents (10 steps, the reference's conditions and noise) | 4e-7 / 4-6e-7 | the same |
| acoustic decoder (speech) | 1.7-2.4e-4 | the same |
| semantic features | 7.4-9.8e-4 | the same |
| next inputs (connectors) | 3.0e-4 | the same |

What the numbers show:

- **Q8_0** accounts for the language model's 1e-2: the f16 file is at 2e-3.
- **The diffusion head** runs at float32 precision (its BF16 weights are exact in half).
- **The paired step** (`Lm::pair`) holds the same numbers as single steps: with f16, logits 4.2e-4 and hidden
  2.2e-3 / 5.0e-4.
- **The free run** (`--stages run`, with the reference's noise) departs from the reference gradually, as an
  autoregressive sampler does.
  - With the f16 file it is 1e-2 off by frame 4 and 1e-1 off by frame 20.
  - Its clip still says the script word for word.
  - The reference's clip and the engine's run transcribe identically (Whisper small): "Welcome back to the show.
    Today we talk about gardens. Thanks for having me. I love tomatoes."

## Clips

On the B70, from the served Q8_0 file:

- A plain voice, a single clone (en-Maya), a two-speaker dialogue (Alice / Frank) and a 46.5 s two-speaker show
  (Carter / Maya) all transcribe word for word with Whisper small.
- Through the audio server, these also transcribe correctly: a plain script, a `ref_audio`, a saved voice without a
  transcript, `voices` of two saved voices, and `ref_audios` of two recordings.
- In five seeds of the four-turn dialogue, the speakers alternate as cast. The median pitch a second is 200-290 Hz on
  Alice's turns and 95-140 Hz on Frank's. The reference shows the same pattern.

## Speed

Measured 2026-10-17 with `nextsycl audio gen`, the long show: 349 frames, 46.5 s of speech.

| card | frames / s | real time | the whole request | on the card |
|---|---|---|---|---|
| B70 | 43.5 | x5.8 | 8.8 s for 46.5 s | 4.9 GiB of weights, 5.6 GiB peak |
| B65 | 38.9 | x5.2 | 10.0 s | 5.6 GiB peak |

- Shorter clips run 39-41 frames a second on the B70. Before the paired step it was 31-36.
- A frame on the B70 takes about 23 ms:
  - the paired language model step, about 9 ms;
  - the diffusion head, about 6 ms (ten steps, each read back);
  - the acoustic decoder and the semantic encoder, about 10 ms. That is mostly small kernels, about 500 launches a
    frame, so SYCL graphs are the next step.
- Loading takes 4-5 s with the file in the page cache.
