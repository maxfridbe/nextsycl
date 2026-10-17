# Voxtral 4B TTS 2603: port brief

Voxtral 4B TTS 2603 (Mistral, March 2026) is an open-weight speech model in nine languages at 24 kHz with 20 preset
voices. nextsycl runs it as the `audio/voxtraltts` engine (arch `voxtral-tts`, model card `voxtral-4b-tts`).

## Sources

- **Upstream checkpoint**: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 (revision b81be46c, not gated):
  `consolidated.safetensors` (bf16, 8.0 GB), `params.json`, `tekken.json`, `voice_embedding/<voice>.pt` (20 presets).
- **Reference code**: vllm-omni (github.com/vllm-project/vllm-omni, commit b164fea7):
  `vllm_omni/model_executor/models/voxtral_tts/` (the acoustic transformer, the codec, the generation loop) and its
  config parser; mistral-common (commit b5416172) for the prompt (`encode_speech_request`) and the Tekken tokenizer.
  Paper: arXiv 2603.25551.
- **GGUF**: https://huggingface.co/cstr/voxtral-4b-tts-GGUF (revision aec6a93f), CrispASR's conversion
  (`models/convert-voxtral-tts-to-gguf.py`, CrispASR commit 369029c9), in f16, q8_0 and q4_k.
  - The codec's weight-normed convolutions are fused (`g v / |v|`); the semantic codebook is stored as
    `embedding_sum / max(cluster_usage, 1e-5)`; both F32 in every file, as are the 20 voices (`voice.<name>`,
    [N, 3,072] each) and the codec's convolutions.
  - Tekken's vocabulary is in the file (`tokenizer.tekken.vocab_tensor`: a u16 length and the bytes of each piece, a
    byte a float32) with the 1,000 specials (`tokenizer.tekken.specials`).
  - sha256: q8_0 75c8049e…, f16 116e5646… (checked against Hugging Face's).
- The served file is the q8_0 (4.3 GB); parity is measured with the f16.

## What the open release supports

| mode | supported | why |
|---|---|---|
| preset voices (20) | yes | `voice_embedding/*.pt`: each a reference clip's frames as their summed code embeddings |
| voice cloning (a recording, saved voices) | **no** | the checkpoint has the codec's decoder but no encoder (below) |
| voice design (a description) | **no** | the prompt has no instruction or description input |
| direction (tone, emotion, pace by instruction) | **no** | the same: no instruction input; style comes from the preset (casual, cheerful, neutral) |
| vocal events | **no** | none in the model's training format or upstream's code |
| guidance (`cfg`) | yes | the flow's classifier-free guidance, upstream 1.2 |
| Euler steps | yes | 7 upstream; `--steps` / `--opt-steps` |

The missing encoder, with the evidence:

- The safetensors' `audio_tokenizer.*` tensors are `decoder_blocks.*` (112), `output_proj` (2) and
  `quantizer.semantic_codebook.{embedding_sum, cluster_usage}`; there is no `input_proj.*` and no `encoder_blocks.*`.
- vllm-omni marks those prefixes "may be absent in open-source checkpoints"; its `encode_waveforms` raises "requires
  encoder weights which are not available in the open-source checkpoint", and its speech adapter refuses `ref_audio`:
  "The open-source variant only supports preset voices via the 'voice' parameter."
- A voice prompt is the encoder's codes of a clip, embedded and summed (`encode_tokens`); without the encoder there are
  no codes for a new recording. (Mistral's hosted service does adaptation.)

The engine refuses `ref_audio` and `instructions` with these reasons; its card lists `speech: [voices]`, so the front
door routes saved voices and clones to models that clone.

## Architecture

| part | shape |
|---|---|
| backbone | Ministral 3B: 26 layers of 3,072; 32 query / 8 key-value heads of 128; SwiGLU 9,216; RMS ε 1e-5; RoPE θ 1e6 on adjacent pairs; no output head (the hidden state feeds the acoustic transformer) |
| embeddings | text 131,072 x 3,072 (tied, unused as a head); audio 9,088 x 3,072: codebook 0 (8,192 semantic codes + 2 specials) then 36 acoustic codebooks of 21 levels + 2 specials, padded to 128 |
| acoustic transformer | 3 bidirectional layers of 3,072, 32 / 8 heads, SwiGLU 9,216, no positions; inputs: the 36 values projected, the time embedding projected, the hidden state projected (3 rows); outputs at row 0 |
| semantic head | 3,072 -> 8,320 (8,192 codes, [EMPTY_AUDIO], [END_AUDIO], padding); argmax with [EMPTY] and the padding masked |
| flow | x0 ~ N(0, 1) [36]; 7 Euler steps over linspace(0, 1, 8); v = cfg v(h) + (1 - cfg) v(0); clamp to [-1, 1], 21 levels, rounded (half to even) |
| codec decoder | 292 = 256 (semantic VQ codeword) + 36 (levels to [-1, 1]); causal conv 292 -> 1,024 (k 3); 4 stages of 2 layers (1,024, 8 heads of 128, RMS ε 0.01, q / k RMS over the whole row ε 1e-6, ALiBi, layer scales, SwiGLU 4,096) with ConvTranspose (k 4, s 2) after the first three; causal conv 1,024 -> 240 (k 7) |

A frame is 12.5 a second and 240 samples.

## The loop (vllm-omni's)

1. The prompt: `<s> [BEGIN_AUDIO] [AUDIO] x N [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]` (mistral-common
   ids 1, 25, 24, 36, …, 35, 25), the voice's N rows at the [AUDIO] positions.
2. The prefill's last hidden state makes frame 0: the semantic code, then (unless [END_AUDIO]) the 36 levels.
3. The frame's 37 codes, each offset into its codebook's block, embedded and summed: the next position's input.
4. [END_AUDIO] (semantic id 1) ends the speech; the codec decodes it in chunks of 375 frames, each on its own.

## Where this follows vllm-omni rather than CrispASR

CrispASR's runtime (`src/voxtral_tts.cpp`, after github.com/mudler/voxtral-tts.c) differs from upstream in five
places; the engine and `reference/voxtraltts/ref.py` follow vllm-omni:

| | vllm-omni (here) | CrispASR |
|---|---|---|
| codec ALiBi slopes (8 heads) | 2^(-8h/8) = 1, 1/2 … 1/128 (get_alibi_slopes) | 2^-(h+1) = 1/2 … 1/256 |
| codec window | keys i - W .. i: W + 1 of them (flash-attn window (W, 0); the SDPA fallback's mask) | W keys |
| the decoder's first convolution's padding | replicate (pad_mode="replicate") | reflect |
| codec decode | chunks of 375 frames (decode_helper_batch_async) | the whole run in one graph |
| frame 0 | the last prompt position's hidden state (compute_mm_logits on the prefill) | one more step with the [AUDIO] token's text embedding first |

## Port

1. The GGUF's matrices are expanded to half at load (F16, Q8_0, Q4_0, Q4_K on the host); q and k are loaded with
   their rows reordered in each head (even rows, then odd) so the rotate-half RoPE kernel rotates the checkpoint's
   adjacent pairs. The text embedding and the voices stay in the file (a row read a token).
2. The backbone runs on the MiniMax Music engine's cached-decode kernels (gemv for up to 8 rows, half cache,
   attention: 4 query heads a key head) and Breeze TTS 2's table-driven RoPE; wide products (the prefill, the codec)
   are oneDNN's half GEMM.
3. The acoustic transformer runs both guidance passes as one batch of six rows; the whole ODE stays on the GPU (the
   input projection's 36 columns padded to 256; mm3's guided Euler step); two syncs a frame (the semantic logits, the
   36 results).
4. Its own kernels (`kernels/audio/voxtraltts/vxt.cpp`): the codec's causal ALiBi attention over a window, an RMS
   norm of strided rows (q / k over the whole row), left padding (replicate / reflect), the flow's six input rows.
   Layer-scaled residuals are Qwen3-TTS's `scale_add`, the convolutions mm3's oneDNN ones.
5. The tokenizer is `nextsycl_tok`'s new Tekken mode (`Tokenizer::from_tekken`: tiktoken's BPE - the adjacent pair
   whose joined piece ranks lowest merges first - over Tekken's split pattern; the first 130,072 pieces in use).
6. The noise is the engine's own (splitmix64 + Box-Muller; a seed repeats here, not torch's draws). The semantic code
   is always the argmax, as upstream.

## Parity

`reference/voxtraltts/ref.py` runs upstream's own code on the CPU in float32 from the safetensors: vllm-omni's
`FlowMatchingAudioTransformer` and `VoxtralTTSAudioTokenizer` taken out of the checkout by name (ast) and run as they
are, vLLM's Mistral backbone written out, mistral-common's prompt. `nextsycl audio check DIR --dir <f16 GGUF dir>`
(B65, 2026-10-17; the f16 GGUF):

| stage | English (neutral_female, 121 frames) | French (fr_female, 44 frames) |
|---|---|---|
| token ids | 244 identical | 118 identical |
| prompt rows | 2.5e-9 | 3.5e-9 |
| hidden states (teacher forced) | 2.6e-4 | 5.5e-4 |
| semantic logits | 1.3e-4 (0 of 122 codes differ) | 2.1e-4 (0 of 45) |
| flow results (teacher forced, the reference's noise) | 6.4e-4 (3 of 4,356 levels differ) | 4.9e-4 (2 of 1,584) |
| flow alone (the reference's hidden states and noise, 16 frames) | velocities 1.7e-6, results 2.0e-6, 0 of 576 levels | 4.7e-7, 4.3e-7, 0 |
| codec (the reference's codes) | 5.5e-4 | 5.4e-4 |
| a free run with the reference's noise | identical for 34 frames | identical for 19 frames |

A free run parts from the reference at the first FSQ level that rounds the other way (a value within ~1e-3 of a
level's edge): a level feeds the next frame's input. Judge parity by the teacher-forced stages.

Real clips (q8_0, `audio gen`, seed 3) transcribe word for word with Whisper small in all nine languages (English in
three voices, French, German, Spanish, Italian, Portuguese, Dutch, Arabic; Hindi missed its first word, "नमस्ते").
The reference's own clips too.

## Speed and memory

B70, q8_0 file, 2026-10-17 (`audio gen`): 30 frames a second (2.4x real time) in the frame loop, steady over a long
passage: 34.6 s of speech in 15.2 s with the prompt and decode (432 frames in 14.4 s, decode 0.4 s); short sentences
4.7-9.8 s of speech in 2.2-4.4 s. B65: 22 frames a second (1.8x; 16.9 s in 9.9 s through the server).

Memory: 6.8 GiB of weights on the card (half; the f16 file the same), 7.2 GiB in a server run (peak, B65, the front door's card count sampled every 0.5 s).
`--opt-steps 4` cuts the flow's share (about half of a frame) for rougher audio.

Where the time goes: per frame the backbone reads its 3.0B weights once (~6 GB in half) and the flow its 0.35B seven
times (~5 GB). Next for speed: the flow in int8, or a graph per frame.

## License

CC BY-NC 4.0 (the weights, the preset voices from EARS, CML-TTS, IndicVoices-R and the Arabic Natural Audio dataset,
and what they make): non-commercial use only. The vllm-omni code is Apache-2.0, CrispASR's converter MIT.
