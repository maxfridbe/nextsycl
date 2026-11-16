# Sophon-Realtime (Flash): port brief

Sophon-Realtime (Adventists AI, v1, November 2026) is a full-duplex voice conversation system: a 32B "brain" hears
the user's speech directly, decides when they have finished, writes a reply with sound tags, and a voice speaks it
while it is written; the user can cut in. nextsycl runs the Flash build as the `audio/sophon` engine (arch
`sophon-realtime`) behind a new endpoint, `/v1/realtime` (OpenAI Realtime's events: [../api/realtime.md](../api/realtime.md)),
which Haru-4B will share. The wfe audio page has a live Conversation panel.

## Sources

- **Code**: https://github.com/adventists-ai/sophon-realtime (commit 2c48aa54, Apache-2.0): `flash/plugin/sophon_realtime`
  (duplex.py: the conversation; talker.py and proc.py: the talker fed as the reply is written; tools.py),
  `finetune/sophon_swift` (the thinker in transformers: modeling_duplexjev.py, modeling_qwen3_asr_encoder.py,
  processing.py), `mac/mlx_ear.py` (the ear on MLX, the same maths). Cloned to
  `/var/mnt/4TBSSD/nextsycl-models/sophon-realtime-code` on hardac1 (Python run only with `-I`).
- **Weights**: https://huggingface.co/adventists-ai/Sophon-Realtime-32B-Flash @ 13b33529 (Apache-2.0; not gated):
  `thinker/` 16 safetensors shards (33 GB: the language model FP8 E4M3 with a scale an output channel -
  compressed-tensors -, the audio tower, projector, embedding and head bf16), `thinker/visual.safetensors` (the
  vision tower: not downloaded, the audio path does not read it), `talker/model.safetensors` (Qwen3-TTS-12Hz-1.7B-Base
  fine-tuned, bf16; its config.json and `speech_tokenizer/` are the Base checkpoint's byte for byte), `voices/*.pt`
  (x-vectors: chelsie, brand; en / zh). On hardac1: `/var/mnt/4TBSSD/nextsycl-models/sophon-realtime/Flash`.
- **Card**: `model-cards/audio/sophon-realtime-32b-flash.yml` (the talker's config and the codec reuse the
  qwen3-tts-base card's keys `q3t-base-config`, `q3t-codec`).

## License

Apache-2.0 code and weights, commercial use allowed. Derived from Qwen3-VL-32B-Instruct, Qwen3-ASR and
Qwen3-TTS-12Hz-1.7B-Base (Qwen, Apache-2.0); NOTICE in the repository. The release asks that only one's own voice, or
one used with permission, be cloned.

## The pipeline

```text
  the client's audio (any rate) -> the listener (duplex.rs): 20 ms energy frames, an adaptive floor
     (threshold max(0.012, 3.5 x noise)); 100 ms of voice starts the user's turn (0.6 s before it kept)
  a pause -> the turn question (text.rs, duplex.py's): "The user said: <|audio|> ... Has the user finished
     speaking? A. ... D. ..." -> the brain's next-token logits over the four letters: P(finished) against a
     threshold that falls as the pause grows (0.2 / 0.35 / 0.5 / 0.8 / 1.3 / 2.0 s: 0.95 ... 0.25); 2 s answers anyway
  the reply: system prompt + earlier turns (the user's speech, Sophon's replies as the audio it spoke - text when
     cut off or calling a tool) + this turn's speech, the thinker's chat template (Qwen3-VL's, no thinking)
     -> the ear (ear.rs): Whisper features (WhisperFeatureExtractor: 128 bands, hop 160; the clip padded to 0.16 s)
        -> Qwen3-ASR's encoder (three stride-2 3x3 convolutions of 480, a linear to 896 plus sinusoids, 18 layers of
        896 with attention in 104-frame windows) -> the projector (pairs stacked, RMS norm, 1,792 -> 4,096,
        x[:2048] * silu(x[2048:]), RMS norm, -> 5,120): 6.25 rows a second in place of <|audio|>
     -> the brain (brain.rs): Qwen3-32B, 64 layers of 5,120, 64 / 8 heads, SwiGLU 25,600, q / k norms, RoPE θ 5e6
        (Qwen3-VL's M-RoPE with three equal positions); prompt cache: the longest common prefix with the last prompt
        is not computed again (audio rows keyed by their clip)
     -> the reply's tokens (sampled: temperature 0.7, top-k 20, top-p 0.8; 120 at most, 240 with tools), the text
        out as written, its speakable part (proc.py's tag_clean: the sound tags kept, other asides and markdown
        dropped; nothing from a <tool_call> on) tokenized and handed to the talker, the last token held back
  -> the talker (speak.rs + talker.rs, on the other card): Qwen3-TTS's streaming prompt (the role, think / language
     / x-vector / pad / bos codec rows, the first text token), then a frame (80 ms, 16 codes) a step, fed the next
     text token's projection - waiting when the brain has none yet (never padding), tts_eos once the reply is whole,
     the end of speech held back until then; the 12 Hz codec decodes as frames come (24 kHz), the reply's leading
     silence trimmed
  the user speaks over a reply (barge-in): it stops, the history keeps the share of it that was played; the user
     speaking again within 2 s of a reply: it came too early - dropped, the two stretches made one turn
```

## Files

| file | what |
|---|---|
| `audio/contract/src/lib.rs` | `Realtime`, `RealtimeSession`, `RealtimeEvent`, `Conversation`; `AudioEngine::realtime` / `converse` |
| `audio/sophon/src/lib.rs` | the engine: roles, options, the memory plan checked before loading, `generate` (the talker alone) |
| `audio/sophon/src/duplex.rs` | the conversation: listener, turn checks, replies (brain thread + talker thread), barge-in, history, tools |
| `audio/sophon/src/brain.rs` | Qwen3-32B: q4 / q8 made on the card, prompt passes (oneDNN on matrices expanded to half a slice at a time, row buckets warmed at load), decode, cache slots with prefix reuse, a two-card layer split |
| `audio/sophon/src/ear.rs` | Whisper features, the Qwen3-ASR encoder, the projector |
| `audio/sophon/src/speak.rs` | the talker fed text as it comes, voices (.pt x-vectors; recordings by the speaker encoder), the codec |
| `audio/sophon/src/talker.rs`, `sample.rs`, `speaker.rs` | copies of the Qwen3-TTS engine's (an engine depends on no other), without its silo |
| `audio/sophon/src/text.rs` | prompts, the turn question (its option permutation precomputed), tag_clean, tool calls |
| `audio/sophon/src/check.rs` | stage checks against reference/sophon/ref.py's dumps |
| `kernels/audio/sophon/sp.{h,cpp}` | FP8 / bf16 -> q4 / q8 on the card; their decode product; their half expansion; cache attention; the ear's patches and window attention |
| `glue/serve/src/realtime.rs` | the WebSocket (RFC 6455, SHA-1 of our own) and OpenAI Realtime's events over a `Conversation` |
| `glue/serve/src/audio.rs`, `http.rs`, `home/{api,mod}.rs` | `/v1/realtime` on the audio server; the front door's splice (`/v1/realtime?model=`, `/open/<id>/v1/realtime`) |
| `wfe/audio/src/converse.ts`, `components/Converse.tsx` | the page's live conversation: microphone (AudioWorklet), playback, barge-in truncation, transcript |
| `reference/sophon/ref.py`, `client.py` | the reference dumps; a standard-library /v1/realtime client for checks |
| `model-cards/audio/sophon-realtime-32b-flash.yml`, `guides/sophon-realtime.md`, `docs/api/realtime.md` | the card, its guide, the protocol |

## Parity

The reference (`reference/sophon/ref.py`) runs the release's own audio code (processing.py, the Qwen3-ASR encoder,
the DuplexJev projector) and the language model's maths in float32 on the CPU, a layer at a time (each layer's FP8
matrices made float32 with their scales). The served model also quantizes activations to FP8 per token (vLLM w8a8);
the reference and nextsycl keep them float32. Input: the release's `live/cases/en_know.wav` ("why is the sky blue"),
resampled to 16 kHz. `nextsycl audio check <dump> --dir <release> --stages mel,ear,lm,check,write` (q4, B65):

| stage | rel RMS | cos |
|---|---|---|
| mel (272 frames) | 5.2e-7 | 1.000000 |
| ear: encoder (35 rows) | 6.5e-4 | 1.000000 |
| ear: projected rows (17 x 5,120) | 4.5e-4 | 1.000000 |
| brain after layer 0 / 1 / 15 / 31 / 47 | 2.8e-2 / 3.3e-2 / 3.7e-2 / 3.9e-2 / 4.0e-2 | 0.99964 / 0.99948 / 0.99953 / 0.99948 / 0.99938 |
| brain after layer 62 / 63 | 1.8e-1 / 2.9e-1 | 0.9846 / 0.9586 |
| logits (the prompt's last row) | 2.6e-1 | 0.9655 |

- The next token is the reference's ("Because"); every prompt row's argmax agrees on 125 of 147.
- Greedy, nextsycl writes "Because sunlight is made up of different colors, and blue light scatters more in the
  atmosphere than other colors, making the sky appear blue to us." Teacher-forced through the reference
  (`ref.py --follow`): 28 of its 30 tokens are the reference's own argmax.
- The turn question over the same clip: P(finished) 0.9980 here, 0.9976 in the reference (logits cos 0.99976).
- The error is q4's (4-bit codes over FP8 weights, least-squares fitted a group of 32): it is ~3-4 % through the
  stack and grows in the last two layers, where Qwen3's few huge activation channels (|x| ~ 900) meet the
  quantization. q8 shows the maths is the reference's:

| q8 (both cards, layers 0-29 on the B70, 30-63 and the head on the B65) | rel RMS | cos |
|---|---|---|
| brain after layer 0 / 1 / 15 / 31 / 47 | 2.2e-3 / 2.4e-3 / 2.3e-3 / 2.4e-3 / 2.7e-3 | 0.999998 / 0.999997 / 0.999998 / 0.999997 / 0.999997 |
| brain after layer 62 / 63 | 1.3e-2 / 2.1e-2 | 0.999917 / 0.999776 |
| logits (the prompt's last row) | 1.7e-2 | 0.999876 |

  Every row's argmax agrees on 141 of 147 (q4: 125); the same next token. Served q4 keeps layers 62 and 63 in q8
  (`NS_SOPHON_Q8_LAYERS`, +0.4 GiB): logits cos 0.966 -> 0.980, argmax 125 -> 127 of 147 (with layers 0 and 1 in q8
  too, their residual matches at rel 2.2e-3 / cos 0.999998 - the q8 numbers - on one card).

## Heard

The answers' audio, written down by the box's Whisper (`nextsycl audio transcribe`, whisper-large-v3-turbo):

- the talker alone, "Oh, that is wonderful news! [laughter] I am so happy for you. Tell me, how did it happen?" ->
  "Oh, that is wonderful news. I am so happy for you. Tell me, how did it happen?" (the tag performed, not read)
- a two-turn conversation over /v1/realtime (en_know, then en_chat "how are you"): "because sunlight is made up of
  different colors and blue light scatters more in the atmosphere than other colors, making the sky appear blue to
  us." / "I'm doing well thank you for asking, how about you?" - each the text the brain wrote.
- a barge-in (en_weather spoken 1.5 s into the first answer, no tools offered): the first answer cancelled, then "I
  don't know the weather in Melbourne tomorrow. You can check a weather app or website for the most accurate
  information."
- a tool call (en_weather with a get_weather function; client.py answers the call): "Let me check that." ->
  get_weather({"city": "Melbourne"}) -> "No, it's not going to rain in Melbourne tomorrow. The forecast is sunny with
  a high of 21 degrees and a low of 12 degrees."
- a saved voice (cloned from a 4 s recording): the first answer, word for word.
- every answer's text and audio agree (`reference/sophon/client.py` sessions on hardac1: ~/src/sophon-out/convo5).

## Speed and memory (hardac1)

Measured on the B65 (GPU 1) with the brain and the talker on the same card (the B70 was serving), q4, 4K context:

| what | B65 |
|---|---|
| load (the release on the 4 TB P310) | 27-60 s: the brain 13 s (page cache warm) to 43 s (cold), the talker 5-7 s, the prompt shapes warmed 4 s |
| memory planned / measured | 27.4 GiB planned on one card (brain matrices 18.9 GiB, ear 0.9, KV for 4K + 1K, work, talker 5.5); brain alone (2K context) 21.6 GiB |
| brain decode | tg 12.3-13.0 tok/s (q4: 19.5 GB read a token, ~250 GB/s) |
| brain prompt pass (`--stages bench`) | 32 rows 417 ms, 128 rows 456 ms (pp ~280 tok/s), 512 rows 872 ms (pp ~590 tok/s); a layer at 32 rows: the q4 matrices expanded to half 4.6 ms (~210 GB/s written), oneDNN's half products 1.7 ms. Before: 2.7 s a pass whatever its rows (the expansion in byte loads and 2-byte stores), then 1.6 s (products in slices stitched by ~900 2-D copies a pass) |
| brain decode product (`bench`) | q4 at ~280 GB/s for one row (gate|up 0.57 ms); a pass of 1 row 73 ms |
| prompt cache | a second turn computes only its new rows (191 positions, 66 computed; 462, 48 after a tool result) |
| ear | a 2.7 s clip in ~50 ms (warm) |
| talker | 98 frames (7.8 s of speech) in 2.64 s: real-time factor 0.34; first audio 0.18 s after its first text |
| turn check | 0.49 s (the ear and a ~100-position pass); on a clip with mid-sentence pauses P(finished) 0.03 / 0.10 / 0.01 at 0.2 / 0.35 / 0.5 s (waits), 1.00 at its end |
| first audio after the user's speech ended (one card) | ~3.2 s: 0.2 s pause + 0.49 s check, the reply's prompt (144 positions, 0.54 s) and 3 tokens, then the talker - which feeds a text token a frame and waits for the brain (13 tok/s, on the same card) for its first frames, the reply's leading silence trimmed. From the reply's start to its first audio 1.7-2.9 s (`reference/sophon/client.py`: five sessions). Before the prompt-pass fixes: 4.5-6.6 s (the turn check lost to the 2 s fallback) |

The card's own layout - the brain on the B70, the talker on the B65 (`--gpus 0,1`), q4 with layers 62-63 in q8:

| what | B70 + B65 |
|---|---|
| memory planned (checked before loading, 1.5 GiB kept free) | B70 23.0 GiB (brain matrices 19.3 GiB, ear 0.9, KV 4K + 1K, work 1.6); B65 5.2 GiB (talker 3.9, codec 0.7, work) |
| load | 19 s (the release in page cache; prompt shapes 3.4 s) |
| brain decode | tg 20.4-21.4 tok/s |
| brain prompt | pp 306 tok/s (144 new positions), 421 tok/s (267) |
| turn check | 0.46-0.51 s |
| from the reply's start to its first audio | 0.80-0.95 s (five sessions); ~1.5 s after the user's last word (0.2 s pause, 0.46 s check, then that) |
| talker | its own card: frames as the brain's text comes, then 3x real time to the end |

q8 on both cards (no talker): tg 12.5 tok/s (each token hops through the host at the split).

## Open items

- The last two layers' q4 error: they are kept in q8 by default now (`NS_SOPHON_Q8_LAYERS=62,63`, +0.4 GiB); the
  logits with them not yet measured.
- Speed: the brain's decode product reaches about half the B65's bandwidth (tg ~13 tok/s); the prompt pass expands
  every matrix to half per pass (fine at a few hundred rows, wasteful at 30); a q4 XMX prompt product, decode graphs.
- First audio: the talker takes a text token a frame (Qwen3-TTS's streaming layout) and waits for the brain when it
  has none, so its first frames come at the brain's pace (13 tok/s on the B65, sharing the card); a faster brain (the
  B70, its own card), fewer frames before the first codec chunk, and drafting (next item) would each cut it.
- Speculative drafting (duplex.py's SOPHON_SPEC_AT: a reply drafted from the first 50 ms of a pause, released when
  the turn is judged finished) - not ported; replies start after the turn check.
- Long replies in segments (segreg.py: a fresh talker context past ~25 s of speech) - not ported (replies are a
  sentence or two; 190 s is the frame cap).
- Back-channel resume (duplex.py's BC_RESUME: a cut reply continued after an "mm-hmm") - not ported.
- Camera frames (the thinker's vision tower, VISION in duplex.py) - not ported, the file is not downloaded.
- The Tag build (CosyVoice3 talker: llm.pt, flow, HiFT; .pt / ONNX files) and the E2E build (a 350M talker of
  continuous vectors) are not ported; only Flash.
- The 4B builds (v1.4, released 2026-11-16) would fit one 24 GB card; not looked at.
- The talker is a copy of audio/qwen3tts's (without its silo's frame graph, ~1.2x slower): moving the Qwen3-TTS
  talker into a foundation crate both engines use would end the copy.
- Haru-4B's lockstep frames on the same socket: designed (api/realtime.md), not built.
- Not deployed or registered in production (a card is in model-cards/; `nextsycl models pull sophon-realtime-32b-flash`).
