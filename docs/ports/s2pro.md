# Fish Audio S2 Pro: port brief

Fish Audio S2 Pro (Fish Audio / 39 AI, released 2026-03) is an open-weight speech model in about 80 languages at
44.1 kHz. nextsycl runs it as the `audio/s2pro` engine (arch `s2-pro`).

## Sources

- **Upstream checkpoint**: https://huggingface.co/fishaudio/s2-pro (revision 1de9996b), about 4.56B parameters.
  - Two bf16 safetensors shards (9.1 GB), `tokenizer.json` (Qwen3's BPE, 151,643 ids plus 4,131 added: the chat
    tokens, `<|voice|>`, the 4,096 `<|semantic:N|>`), `config.json` (`fish_qwen3_omni`) and `codec.pth` (float32,
    weight-normed convolutions).
- **Upstream code**: https://github.com/fishaudio/fish-speech (commit 54156693, which also carries Drama 3; S2 Pro's
  path is `fish_speech/models/text2semantic/llama.py` + `inference.py`, the codec `fish_speech/models/dac/`).
  Fish Audio Community License.
- **GGUF**: https://huggingface.co/rodrigomt/s2-pro-gguf (revision a7320690), s2.cpp's conversion
  (`quantize/unified_export_gguf.py`): the Dual-AR transformer and the codec in one file, `general.architecture`
  `fish-speech`.
  - The transformer's matrices and embeddings are Q8_0 in the served file (F16 in `s2-pro-f16.gguf`, used for
    parity), the norms F16.
  - The codec (`c.`-prefixed) is F16 in both: its weight norm folded in float32 first, then stored in half.
  - The tokenizer.json in the GGUF repo is the checkpoint's byte for byte (sha256 f24e0809...).
- **Community engine**: https://github.com/rodrigomatta/s2.cpp (commit 2c332619; GGML / Vulkan): read for the GGUF's
  tensor names and its prompt layout, which matches upstream's.

## License

- The weights, derivatives (the GGUF) and the audio they make are under the Fish Audio Research License: research and
  non-commercial use free; commercial use needs a written license from Fish Audio (business@fish.audio).
- Attribution: "This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights
  Reserved."

## Architecture

S2 Pro is a Dual-AR model over Fish's RVQ codec.

| part | shape |
|---|---|
| slow model | Qwen3: 36 layers of 2,560; 32 query / 8 key-value heads of 128; q / k RMS norms; SwiGLU 9,728; RoPE base 1e6 on interleaved pairs; ε 1e-6; context 32,768 |
| slow input | a text token's embedding row; a frame's row = (its semantic token's row + its ten codes' rows of a [10 x 4,096, 2,560] codebook embedding) / sqrt(11) |
| slow head | the tied embedding (155,776 ids); a frame may be only a semantic id (151,678 + code) or `<\|im_end\|>` (the end) |
| fast model | 4 layers of 2,560, the same heads and SwiGLU, no q / k norms, RoPE base 1e6 over 10 positions |
| fast input | position 0: the slow model's normed last hidden state; position i: the embedding of code i - 1 ([4,096, 2,560]) |
| fast head | 2,560 -> 4,096: codes 1..9 |
| codec | 10 codebooks (a 4,096-entry semantic one, nine 1,024-entry residual ones, 8-dimensional, cosine nearest); 2,048 samples a frame (21.5 frames a second) |
| codec decoder | the codebooks' 1x1 projections to 1,024, summed -> an 8-layer transformer (16 heads of 64, a 128-frame causal window, layer scales) -> 2 x (transposed conv x2 + ConvNeXt) -> conv k7 to 1,536 -> 4 DAC blocks (Snake, transposed conv x8 / x8 / x4 / x2, residual units at dilations 1, 3, 9) -> Snake -> conv k7 -> tanh; every convolution causal |
| codec encoder | conv k7 to 64 -> 4 DAC blocks (x2 / x4 / x8 / x8, the last with a 4-layer transformer, a 512-step window) -> conv k3 -> 2 x (strided conv x2 + ConvNeXt) -> an 8-layer transformer -> the quantizers |

RoPE everywhere (slow, fast, codec) is fish-speech's `precompute_freqs_cis`: the cosines and sines are stored in
bfloat16, and the port keeps that (a table rounded the same way).

## Prompt and modes

The prompt is fish-speech's `Conversation` as `generate_long` builds it. Each text part is tokenized on its own.

```text
<|im_start|>system\n convert the provided text to speech <|im_end|>\n                                   (plain)
<|im_start|>system\n convert the provided text to speech reference to the following:\n\nText:\n
    <|speaker:0|>TRANSCRIPT \n\nSpeech:\n [the recording's frames] <|im_end|>\n                      (a clone)
<|im_start|>user\n TEXT <|im_end|>\n <|im_start|>assistant\n<|voice|>  -> frames until <|im_end|>
```

| mode | what S2 Pro has | here |
|---|---|---|
| plain | the model's own voice for the text (varies with the seed) | `input` alone |
| clone | a recording's codes and its transcript in the system turn (no transcript: no clone - upstream drops the reference) | `ref_audio` + `ref_text`, or a saved voice with its transcript |
| inline control | free-form tags in square brackets in the text, applying from where they stand (15,000+ in training: `[whisper]`, `[laughing]`, `[excited tone]`, `[pause]`, `[pitch up]` ...) | the text as written; `instructions` becomes such a tag at the start of each turn (`sad` -> `[sad] ...`) |
| several speakers | `<\|speaker:N\|>` turns in the text, read a batch at a time (five speakers or 200 bytes), each batch a user turn after the previous batches' audio | the same, the cache carried from batch to batch |
| voice design | **none**: no description input, no built-in voices | not offered (the card has no `design`) |

- Draws: upstream's `logits_to_probs` - top-k 30 and top-p 0.7 cut on the plain softmax of the sorted logits (the best
  kept), then temperature 0.7.
- The frame's first code is repetition-aware: a draw among the last 10 semantic ids is redrawn at temperature 1.0,
  top-p 0.9.
- The fast model's codes are drawn the same way, without RAS.
- No guidance (cfg).

## Port

1. The matrices stay as the file has them. Q8_0 goes on the card as int8 codes plus a half scale a block, split at
   load so a lane reads 8 aligned codes; F16 goes on as half.
2. The decode's product (`ns_audio_s2p_gemv_q8`, up to 8 rows) reads 1.06 bytes a weight. A prompt's wide products
   expand the matrix to float32 and use oneDNN's float32 GEMM.
   - Half is not used there: some prompts' activations pass half's range (the down projection's input reaches tens
     of thousands).
   - Converted to half, a 170-token prompt's logits came out wrong (cosine 0.49); in float32 they match (1.1e-4).
3. The embeddings stay in the file. A prompt's token rows, and a frame's 11 rows, are read and summed on the host.
4. The attention, the half cache and the half embedding table of the fast model use the MiniMax Music engine's kernels
   (32 query heads over 8).
5. The engine's own kernels (`kernels/audio/s2pro`) are the Q8_0 product, the float32 expansion and the
   interleaved-pair RoPE from a table, with an optional per-head RMS norm.
6. The codec runs in float32.
   - The Qwen3-TTS codec's windowed attention, depthwise causal conv and layer-scaled residuals; the MiniMax Music
     engine's oneDNN convolutions; the shared Snake, norms, GELU, SwiGLU and linear layers.
   - The quantizers run on the host: each codebook's 1x1 projection precomputed at load, a nearest search in 8
     dimensions.
   - The decoder's convolution stack runs in chunks of 200 frames with 16 frames of context. It is causal with a reach
     of a few frames, so the chunks join exactly.
7. A clone's recording is resampled to 44.1 kHz (windowed sinc), up to 30 s.

## Checks

`reference/s2pro/ref.py` runs fish-speech's own code on the bf16 checkpoint and `codec.pth`, in float32 on the CPU,
greedy. `--codec-half` additionally rounds the codec's weights to half, as the GGUF holds them. Then:

```sh
nextsycl audio check DUMP --dir DIR --stages prompt,frames,codec
```

Four prompts: English plain (en1), a clone of a 9.9 s recording with `[whisper]` (clone1), Chinese with `[excited]`
(zh1), and a 170-token English paragraph with `[sigh]` / `[laughing]` (long1). All on the B65 / B70, 2026-10-17.

| stage | F16 file | Q8_0 file (served) |
|---|---|---|
| prompt rows (ids and codes) | identical (38, 302, 36, 170) | identical |
| the recording's codes (encoder) | 2,137 of 2,140 agree, every semantic code | the same (the codec is F16 in both) |
| prompt input rows | 4e-9 .. 5e-8 | 3.6e-3 .. 7.1e-3 (Q8_0 embeddings) |
| slow logits, teacher forced | 1.1e-4 .. 1.3e-4 | 5.6e-3 .. 6.5e-3 |
| fast logits, teacher forced | 4.1e-5 .. 1.3e-4 | 1.4e-3 .. 4.9e-3 |
| argmax differing from the reference's choice | 0-1 first codes, 0-1 of 270-540 others | 0-3 first codes, 4-11 of 270-540 others |
| greedy frames, identical to the reference's | en1 40 / 40, long1 30 / 30, clone1 46 / 60, zh1 19 / 50 (each from a near tie) | diverge after 0-2 frames (quantization; sampled speech is unaffected) |
| codec latents | 6.1e-4 .. 9.6e-4 against codec.pth (its half rounding) | the same |
| speech | 5.3e-4 .. 8.4e-4 against codec.pth; **7e-7 .. 2e-6** against the reference with the GGUF's half weights | the same |

Every real clip transcribes word for word with Whisper small on the CPU:

- English plain, with tags
- the long paragraph (42.8 s)
- clones of a saved voice (`narrator`, 88 Hz median pitch; the clones 83-88 Hz)
- Chinese and Japanese
- a six-turn, three-batch dialogue
- the API's saved-voice, instruction and `continue_from` requests

Tags move the voice measurably. The same seed and text:

| tag | spectral flatness | loudness | peak |
|---|---|---|---|
| none | 0.08 | -24.0 dBFS | 0.28 |
| `[whisper]` | 0.21 (breathy) | -26.7 dBFS | 0.18 |
| `[shouting]` | 0.18 | -20.4 dBFS | 0.43 |

## Speed and memory (B70, 2026-10-17)

- 41.5-43.9 frames a second for the frames (21.5 a second of audio: 1.9-2.0x real time).
- A 42.8 s paragraph took 24.4 s in all: 22.1 s of frames, 2.2 s of decode (1.75x real time).
- A short sentence answers in 2-4 s through the server.
- A clone's 10 s recording adds about 1 s: the encoder, and the prompt read from the file.
- Q8_0: 5.6 GiB on the card loaded, 6.5 GiB at the peak of the 42.8 s clip. F16: 9.1 GiB loaded.
- Where the time goes: the slow model reads 3.9 GB a frame, the fast model 0.43 GB nine times (a frame at 42 frames
  a second is about 330 GB/s); host draws between the fast model's codes.
- Room left: sampling on the GPU, graphs over the fast model's nine passes.
