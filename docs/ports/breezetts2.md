# Breeze TTS 2: port brief

Breeze TTS 2 (BreezeBlue / RESONIA, released 2026-08-25) is an open-weight speech model in English and Chinese at
24 kHz. nextsycl runs it as the `audio/breezetts2` engine (arch `breeze-tts-2`).

## Sources

- **Upstream checkpoint**: https://huggingface.co/BreezeBlue/Breeze-TTS-2 (revision 3e28c515), about 3B parameters.
  - Two bf16 safetensors shards (7.0 GB), `tokenizer.json` (Gemma3's, 262,158 ids with the added tokens) and the
    `audio_tokenizer/` folder.
- **Upstream code**: https://github.com/breezeblue-ai/breeze-tts (commit 6eba901d), Apache-2.0.
  - `models/breeze.py`, `models/t5gemma2_compat.py`, `breeze_infer/templates.py`.
  - The eager runtime is `models/fast_streaming.py` with `cudagraph/depth_decoder_graph.py`.
- **GGUF**: https://huggingface.co/cstr/breeze-tts-2-GGUF (revision 5314fe9a), CrispASR's conversion, in f16, q8_0
  and q4_k.
  - It drops two inference-dead tensors: `embed_text_tokens` (1.07 GB) and the Mimi `codec_model.*`.
  - It splits `codebooks_head` into 15 heads `[2051, 1024]`.
  - It folds Gemma's `1 + w` into the text encoder's norms (`breeze.te.norm_weights_pre_offset`).
  - Its metadata states the real hyperparameters, so the top-level `config.json` decoys need no special handling.
- **Codec**: the bundled `audio_tokenizer/model.safetensors` is byte-identical to Qwen3-TTS-Tokenizer-12Hz, the
  `speech_tokenizer/` of the Qwen3-TTS checkpoints (sha256 836b7b35..., checked with `cmp` on hardac1).
  - The card names the existing `q3t-codec` file.
  - The decoder and encoder moved from `audio/qwen3tts` into the foundation crate `crates/codec12hz`, which both
    engines use.

## License

- The weights, derivatives (the GGUF) and self-hosted outputs are under the BreezeBlue Research and Non-Commercial
  License 1.1: research and non-commercial use only.
- The code is Apache-2.0.
- The codec is Qwen's, Apache-2.0.

## Architecture

Breeze TTS 2 is a CSM fork.

| part | shape |
|---|---|
| text encoder | T5Gemma2's encoder: 26 layers of 1,152; 4 query heads and 1 key-value head of 256; q / k norms; a GELU-tanh gated MLP of 6,912; four norms a layer; bidirectional attention |
| text encoder attention | sliding layers attend to keys in (i - 256, i + 257) at θ 1e4; every sixth layer attends to the whole text at θ 1e6 with linear scaling ÷ 8; scores × 1/16; embedding × sqrt(1,152) |
| projection | 1,152 → 2,048, no bias |
| backbone | Qwen3: 28 layers of 2,048; 16 query / 8 key-value heads of 128; SwiGLU 6,144; θ 1e6; ε 1e-6 |
| backbone head | 2,052 ids: the codec's 2,048 codes, three reserved ids, and the end of speech |
| depth decoder | Llama: 12 layers of 1,024; 8 / 2 heads of 128; SwiGLU 8,192; ε 1e-5; llama3 RoPE (θ 5e5, factor 32, 16 original positions) |
| depth decoder input | a 2,048 → 1,024 projection of [the backbone's hidden state, then each code's embedding] |
| depth decoder heads | 15, one per codebook 1..15 |
| embedding table | 16 × 2,051 rows of 2,048, tied between the backbone (a frame is the sum of its 16 rows) and the depth decoder |

A frame comes out at 12.5 Hz and decodes to 1,920 samples.

## Prompts and modes

The prompt layout follows upstream's `templates.py`. Each text segment starts with `<bos>[S0]` and runs through the
text encoder as its own sequence.

| mode | prompt | guidance |
|---|---|---|
| plain | `[S0]text` | none |
| design | `[S0]<ins_bos>instruction<ins_eos>text` | cfg 4 against plain |
| clone | `[S0]ref_text`, the recording's frames as `<\|AUDIO\|>`, `<\|audio_eos\|>`, then `[S0]text` | none |
| direction | the clone's prompt with the instruction before the text | cfg 4 against the clone |

- The guided logits are `uncond + cfg (cond - uncond)`.
- The backbone draws with a repetition penalty of 1.1 over the frames' first codes.
- Ids 2048-2050 are masked out.
- Both the backbone and the depth decoder sample at temperature 0.9 with top-k 50.
- Speech ends at id 2051. A run stops at 1,500 frames, or at 2,048 positions counting the prompt.
- The clone mode needs the recording's exact transcript.
- Vocal events are plain text: `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)` in English, and `[笑]`, `[叹气]`,
  `[咳嗽]`, `[清嗓子]` in Chinese.

## Port plan

1. The GGUF's matrices are expanded to half at load. The host dequantizes F16, Q8_0, Q4_0 and Q4_K. The text
   embedding stays on disk and is read one row per token.
2. The backbone and depth decoder use the MiniMax Music engine's cached-decode kernels (mm3: gemv, half cache,
   attention).
3. The engine's own kernels are in `kernels/audio/breezetts2`: windowed bidirectional attention, `gelu_tanh(g) * u`,
   and per-head norm plus RoPE from a frequency table.
4. The depth decoder runs the guided pair as one batch of two sequences.
5. The tokenizer is `nextsycl_tok`'s SentencePiece-style BPE. Gemma's `Split(" ")` pre-tokenizer after the
   space-replacing normalizer is a no-op, so the tokenizer accepts it.
6. Parity follows `reference/breezetts2/ref.py` (upstream on the CPU in float32, greedy), and
   `nextsycl audio check` reports these stages:
   - prompt: ids, text encoder, and the rows
   - teacher-forced frames: backbone and depth logits
   - greedy frames
   - codec
