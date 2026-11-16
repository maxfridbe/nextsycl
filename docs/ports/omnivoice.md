# OmniVoice: port brief

OmniVoice (k2-fsa / Xiaomi, 2026; arXiv 2604.00688) is a zero-shot text-to-speech model for 646 languages at 24 kHz,
built as a diffusion language model. nextsycl runs it as the `audio/omnivoice` engine (arch `omnivoice`, card
`model-cards/audio/omnivoice.yml`).

## Sources

- **GGUF (what nextsycl loads)**: https://huggingface.co/Serveurperso/OmniVoice-GGUF (revision 01709416), omnivoice.cpp's
  conversion. Two files load together, each in F32 / BF16 / Q8_0 / Q4_K_M:
  - `omnivoice-base-*.gguf` (`general.architecture` omnivoice-lm): the Qwen3-0.6B backbone, the audio embeddings
    [8 x 1,025, 1,024] and heads [8 x 1,025, 1,024], the GPT-2-style vocabulary (151,676 ids; no `tokenizer.ggml.pre`,
    no token types) and the special tokens' ids (`omnivoice.special.*`). Q8_0: every matrix Q8_0, norms F32;
    Q4_K_M: Q4_K with Q6_K for v / down / the embeddings.
  - `omnivoice-tokenizer-*.gguf` (omnivoice-tokenizer): the Higgs Audio v2 codec - HuBERT, the semantic encoder, DAC's
    encoder and decoder (weight norm folded), fc / fc2, the 8 quantizers. The quantizers, fc, fc2 and Snake alphas are
    F32 in every variant; the convolutions F16 in the Q* variants; HuBERT's linears follow the variant. The semantic
    *decoder* (training only) is not in the file.
- **Upstream checkpoint**: https://huggingface.co/k2-fsa/OmniVoice (revision c5fdb5cc): model.safetensors (float32),
  tokenizer.json, audio_tokenizer/ (the codec, transformers' HiggsAudioV2TokenizerModel). Used for the reference only;
  the F32 GGUF's tensors equal it (checked for the semantic encoder: max difference 0).
- **Upstream code**: https://github.com/k2-fsa/OmniVoice (commit 08be0b4): `omnivoice/models/omnivoice.py` (prompt,
  iterative unmasking, post-processing), `omnivoice/utils/{text,duration,audio,voice_design,lang_map}.py`.
- **Community engine**: https://github.com/ServeurpersoCom/omnivoice.cpp (commit a5bd24f; GGML): not read for code; its
  GGUF layout is read from the files themselves.

## License

- Weights: **CC BY-NC 4.0 - non-commercial** (upstream: "The pre-trained model is licensed under the CC-BY-NC due to
  constraints from its training data (e.g., Emilia)"); the GGUFs are conversions and carry the same terms.
- Code: upstream Apache 2.0; the Higgs Audio v2 codec (Boson AI) Apache 2.0; omnivoice.cpp MIT. The language table
  (`audio/omnivoice/src/languages.tsv`) is upstream's `docs/lang_id_name_map.tsv` (Apache 2.0).
- Upstream prohibits unauthorized voice cloning, impersonation, fraud and scams.

## Architecture

| part | shape |
|---|---|
| backbone | Qwen3-0.6B: 28 layers of 1,024; 16 query / 8 key-value heads of 128; q / k RMS norms; SwiGLU 3,072; RoPE (rotate-half) base 1e6; ε 1e-6; **bidirectional** (a full mask within each sequence), no LM head |
| input rows | a text token: its `embed_tokens` row; an audio frame: the sum of its 8 codes' rows of `audio_embeddings` (codebook c's code k: row k + 1,025 c; 1,024 is the mask) |
| head | `audio_heads` [8,200, 1,024] on the final-normed hidden state: 8 codebooks x 1,025 logits a frame |
| codec | 25 frames a second of 8 codebooks of 1,024 (64-dimensional, Euclidean nearest), 960 samples a frame at 24 kHz |
| decoder | each codebook's row through its projection out (64 -> 1,024), summed, fc2 (-> 256), DAC's decoder: conv k7 -> 1,024, five blocks (Snake, transposed conv x8 / x5 / x4 / x2 / x3 with output padding stride % 2, residual units at dilations 1, 3, 9), Snake, conv k7 -> 1; no tanh |
| encoder | semantic: 16 kHz (torchaudio's sinc resampler), 160 zeros a side, HuBERT-base (7 convs with a GroupNorm a channel on the first, 12 post-norm layers), the mean of its 13 hidden states (the normed input and each layer's output), every other frame, the semantic encoder (conv k3, two blocks of two ELU residual units and a conv k3); acoustic: DAC's encoder (conv k7, five blocks /8 /5 /4 /2 /3, Snake, conv k3 -> 256); [acoustic; semantic] -> fc (1,024) -> the residual quantizer |

## Generation (upstream's, kept)

- **Prompt**: `[<|denoise|> with a recording] <|lang_start|>LANG<|lang_end|><|instruct_start|>ATTRS<|instruct_end|>`
  (`None` for a missing one) then `<|text_start|>TRANSCRIPT TEXT<|text_end|>` - each non-verbal tag tokenized on its
  own - each token repeated over the 8 codebooks; then the recording's codes; then T mask frames. The tokenizer is
  the base GGUF's vocabulary (Qwen2's split) with Qwen's added tokens (151,643 on) matched in the text and NFC first,
  as the checkpoint's tokenizer.json does.
- **Length**: upstream's RuleDurationEstimator (a weight a character by script; the target's weight at the
  recording's pace, or "Nice to meet you." in 25 frames; under 50 frames boosted by a cube root), divided by `speed`;
  `duration` fixes it.
- **Steps** (32): one pass over the conditional sequence (prompt + target) and the unconditional one (the target
  alone), each from position 0. For each masked position and codebook: log_softmax of both, `lc + 2 (lc - lu)`,
  log_softmax again, the mask code excluded; its best code and that log probability; minus 5 x the codebook's index;
  divided by 5 plus Gumbel noise; the top k unmasked, k from the shifted schedule (t_shift 0.1).
- **Long text**: over 30 s by the estimate, chunks of about 15 s at sentence ends; each chunk with the recording, or
  (no recording) the first chunk's codes and text as the voice; decoded one by one and cross-faded (0.1 s fades,
  0.1 s of silence).
- **Post**: pydub's silence removal (-50 dBFS, 10 ms steps: middle silences over 0.5 s cut to 0.5 s a side, the
  ends to 0.1 s; the samples pass through 16 bits), the level (a clone: scaled by its RMS / 0.1 when that was under
  0.1; otherwise a peak of 0.5), 0.1 s fades and 0.1 s padding. A recording: its RMS raised to 0.1 when quieter,
  middle silences over 0.2 s shortened, the ends trimmed, cut to whole frames.

## What the weights support (and what is exposed)

| | |
|---|---|
| built-in voices | none (`voice` is refused unless the front door resolves a saved voice to its recording) |
| clone | a recording + its transcript (required: upstream would run Whisper to make one; the engine does not) |
| design | `instructions` = attributes: male / female; child, teenager, young adult, middle-aged, elderly; five pitches; whisper; ten English accents; twelve Chinese dialects (or their Chinese names). Anything else is refused with the list, as upstream |
| direction | a clone plus attributes |
| events | `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en/ah/oh/ei/yi]`, `[surprise-ah/oh/wa/yo]`, `[dissatisfaction-hnn]`; pronunciation by CMU phonemes or pinyin in the text |
| languages | 646 (names or ids, upstream's table); auto by default; an unknown one is refused |

## The port

- `audio/omnivoice/src`: `lm.rs` (the backbone), `codec.rs` (decoder, encoder, quantizer), `text.rs` (prompt,
  attributes, languages, length, chunks), `post.rs` (resampler, silence, fades), `lib.rs` (the unmasking loop, the
  request), `check.rs` (stages against the reference).
- The backbone's matrices are expanded to half at load by the shared dequant (any of the release's types), the
  residual stream float32, the products half in / half or float32 out (upstream runs in half on CUDA). The attention
  is the shared fused one, a call a sequence, after `ns_audio_ov_expand_qkv` lays the 8 key / value heads beside the
  16 query heads. The step's scores are `ns_audio_ov_score` (one sub-group a position and codebook); the host reads
  2 x 8 x T numbers a step and picks the positions.
- The codec in float32: the convolutions on the MiniMax Music engine's oneDNN wrapper (`conv1d_lr`: the transposed
  ones need their output padding), Snake / GELU / layer norms / attention / linears shared; the decoder's eight
  projections and fc2 folded at load into one [256, 512] product; HuBERT's grouped positional convolution as 16
  convolutions; the quantizer's nearest codes on the host.
- A trap found on the way (the shared wrapper, left as it is): mm3's `conv1d_lr` keeps one oneDNN primitive a shape,
  made with or without a bias by the shape's first call. The semantic encoder has bias-less and biased k3
  convolutions of the same shape, so the biased ones ran without their bias (the semantic encoder 5.9e-2 off, 2% of a
  recording's codes wrong). The engine gives every bias-less convolution a zero bias; the wrapper's key should carry
  the bias (a shared-kernel fix: every engine that calls it re-checked first).
- No silo: nothing is tuned yet (rule 6 - a speed pass would start one in `kernels/audio/omnivoice/silo`).

## Parity

`reference/omnivoice/ref.py` runs upstream's code (CPU, float32) with position_temperature 0 (deterministic) and dumps
each stage; `nextsycl audio check DUMP --model ID` compares, each stage from the reference's own input. Five cases:
English design (`female, young adult, british accent`), English auto voice, Chinese design (`male, middle-aged, low
pitch`), a `[laughter]` tag, and an English clone (LibriSpeech dev-clean 1272, 9.9 s, with its transcript).

With the F32 GGUFs (the checkpoint's float32 weights):

| stage | result |
|---|---|
| prompt tokens, T, L | identical in all five (the duration estimate exact) |
| recording preprocessing (resample, level, pydub trim) | same length; rel 4.0e-6 |
| HuBERT mean (semantic) | rel 1.0e-4 |
| DAC encoder (acoustic) | rel 3.1e-6 |
| semantic encoder | rel 1.3e-6 from the reference's HuBERT features; 1.5e-4 end to end |
| fc (the quantizer's input) | rel 1.6e-5 |
| the recording's codes | 1,912 / 1,912 identical |
| step 0 input rows | exact |
| layer 0, final hidden | rel 1.5-2.2e-4 |
| logits, step 0 and step 16 (teacher forced) | rel 2.3-2.5e-4, cosine 1.000000 |
| guided best code per position, step 0 / 16 | 78-92% / 90-95% agree (step 0's distributions are nearly flat: near ties) |
| codec decoder (the reference's codes) | rel 1.0-1.7e-6 |
| post-processing (the reference's raw sound) | exact (0) / 4.6e-8 for the clone |

A whole run is not bit-comparable: the unmasking order follows the top-k of scores that differ in the 4th digit, and
one different choice changes every later step (4-24% of codes agree). The runs are judged by listening instead.

## Listening

Our runs' sound (`ours.wav`, F32, deterministic) transcribed by Whisper large-v3-turbo (CPU), against the text:

| case | heard | f0 (median) |
|---|---|---|
| design-en | "The quick brown fox jumps over the lazy dog. Omnivoice speaks more than 600 languages." | 173 Hz (the reference's run: 164) |
| auto-en | "Good morning. The weather today is sunny with a light breeze from the west." | 203 Hz |
| design-zh | "今天天气很好,我们一起去公园散步吧!" | 167 Hz |
| tag-en | "That is the funniest thing I have heard all week. You have to tell me again." | 129 Hz |
| clone-en | "This voice was cloned from 10 seconds of an old audiobook, and now it reads whatever we give it." | 114 Hz (the recording: 120, the reference's clone 121) |

All at 24 kHz mono, peak 0.5 (design / auto) or the recording's level (clone: peak 0.46, RMS 0.067 against the
recording's 0.071).
