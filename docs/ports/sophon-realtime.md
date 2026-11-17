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
- **Cards**: `model-cards/audio/sophon-realtime-32b-flash.yml` (Base's talker, tags out) and `-tags.yml` (Flash's talker, tags performed): Talkers below (the talker's config and the codec reuse the
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
| `audio/sophon/src/check.rs` | stage checks against reference/sophon/ref.py's and talker_ref.py's dumps; `speak` takes several voices (a recording's clone or a release voice) and a temperature for A/Bs |
| `kernels/audio/sophon/sp.{h,cpp}` | FP8 / bf16 -> q4 / q8 on the card; their decode product; their half expansion; cache attention; the ear's patches and window attention |
| `glue/serve/src/realtime.rs` | the WebSocket (RFC 6455, SHA-1 of our own) and OpenAI Realtime's events over a `Conversation` |
| `glue/serve/src/audio.rs`, `http.rs`, `home/{api,mod}.rs` | `/v1/realtime` on the audio server; the front door's splice (`/v1/realtime?model=`, `/open/<id>/v1/realtime`) |
| `wfe/audio/src/converse.ts`, `components/Converse.tsx` | the page's live conversation: microphone (AudioWorklet), playback, barge-in truncation, transcript |
| `reference/sophon/ref.py`, `client.py` | the reference dumps; a standard-library /v1/realtime client for checks |
| `reference/sophon/talker_ref.py` | the talker as upstream feeds it (proc.py's tag_clean, the thinker's tokenizer, talker.py's prompt through qwen-tts on the CPU) for `audio check --stages talker`; `--stages speak` speaks lines.json's lines x seeds with the talker alone |
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
  - 2026-11-16, through the front door's `/v1/audio/speech` (voice chelsie, seeds 1-3, each of the 7 tags in a line):
    every tag was read aloud - "Laughter.", "Breath,", "Sigh.", "Cough.", "Quick breath.", "less than ... laughter"
    for `<laughter>` and "strong blue" for `<strong>blue</strong>` (21 of 21 takes) - and `<strong>` stressed
    nothing (the word +0.3 dB on average against the line, +2.1 without the tags). The speech text pass keeps the
    tags as written (the server's markup); the talker gets them as plain text tokens (`speakable_ids`: tag_clean, then
    the thinker's tokenizer).
  - **The cause (found 2026-11-16): the wrong talker.** The registry's `talker` was
    `/var/mnt/2TBSSD/qwen3tts/models/qwen3-tts/base/model.safetensors` - Qwen3-TTS-12Hz-1.7B-Base, which reads tags
    as text. `models pull --from` had linked it for the card's `talker/model.safetensors` because a copy is found
    by name and size, and a fine-tune keeps its base model's size to the byte (3,857,413,744; SHA-256 38fc7fc5...
    against the card's b1d57164...); only copies under 256 MiB were hashed. The first check above ran `--dir` on the
    release, with Flash's own talker - hence performed then, read aloud through the registry. Our input path was
    already upstream's: proc.py feeds the talker the thinker tokenizer's plain ids of tag_clean's text (no special
    tokens, no tag table: `[laughter]` is " [", "laughter", "]"), and talker.py builds Qwen3-TTS's streaming
    x-vector prompt - nothing else carries the tags. Fixed: pull hashes every name-and-size copy before linking it
    (glue/models catalog.rs), and the talker gets the tags again by default (`NS_SOPHON_TAGS=0` takes them out).
  - **Parity with upstream's talker path** (`reference/sophon/talker_ref.py`: upstream's tag_clean and cut_tool from
    the repository's own files, the thinker's tokenizer in transformers, the prompt talker.py builds run through the
    qwen-tts package in float32 on the CPU, greedy; then `nextsycl audio check DIR --stages talker --gpu 1`, the
    talker alone on the B65), "They would be bagels! [laughter] Oh, that is a classic.", voice chelsie: the 16
    speakable ids identical; the 10 prompt rows rel 4.8e-8 (cos 1.000000); the first logits rel 3.9e-4 (cos
    1.000000), the same first code (1221); greedy 71 frames in both, all 1,136 codes identical.
  - **Heard with Flash's talker** (`--stages speak`, voice chelsie, seeds 1-3, the sampled defaults; Whisper
    large-v3-turbo; the gap: Whisper's word before the tag to the word after, its 10 ms frames above -40 dBFS, against
    the same line without the tag):

    | tag | tag word written | gap, s (sound, s) | without the tag |
    |---|---|---|---|
    | `[laughter]` "bagels! [laughter] Oh" | 0 of 3 | 1.20 (0.81) / 1.34 (0.43) / 1.22 (0.48) | 0.28 (0.08) / 0.48 (0.10) / 0.58 (0.11) |
    | `[sigh]` "over. [sigh] Fine" | 0 of 3 | 1.48 (0.70) / 1.18 (0.32) / 0.96 (0.36) | 0.50 (0.06) / 0.62 (0.07) / 0.34 (0.08) |
    | `[cough]` "me, [cough] as" | 0 of 3 | 0.32 (0.03) / 0.28 (0.02) / 0.76 (0.15) | 0.16 (0.05) / 0.14 (0.06) / 0.22 (0.08) |
    | `[breath]` "go. [breath] Let" | 0 of 3 | 1.72 (0.29) / 1.54 (0.54) / 0.88 (0.42) | 0.24 (0.04) / 0.26 (0.01) / 0.68 (0.04) |
    | `[quick_breath]` (same line) | 1 of 3 ("Quick breath.") | - / 0.78 (0.10) / 1.84 (0.12) | as above |
    | `<laughter>...</laughter>`, "person.</laughter> Well" | 0 of 3 | 0.84 (0.13) / 1.18 (0.30) / 0.74 (0.21) | 0.58 (0.10) / 0.36 (0.06) / 0.50 (0.06) |
    | `<strong>blue</strong>` | 0 of 3 | "blue" -1.9 / +4.1 / +0.8 dB against the line | -0.7 / +1.0 / +0.9 dB |

    `[cough]` is weak in English (upstream: 50%), `<strong>` weak (upstream says so). Upstream's own greedy take of
    the bagels line: "they would be bagels. Oh, that is a classic.", 2.1 s between with short bursts of sound.
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

## Talkers (2026-11-17): Base clones closer, Flash performs the tags

Two registry entries, the same brain:
- `sophon-realtime-32b-flash`: Qwen3-TTS-12Hz-1.7B-Base's model.safetensors as the talker, with `NS_SOPHON_TAGS=0`.
- `sophon-realtime-32b-flash-tags`: the release's own talker, which performs the tags.

The release's talker is Base fine-tuned. NOTICE says "the language model was fine-tuned to perform sound tags"; the
training data is not published. Compared tensor by tensor with Base's:
- the speaker encoder (74 tensors) is identical;
- the talker's layers differ (rel 1.8e-3), as do the code predictor (7.3e-3), the codec head and the text projection.

So a recording gives the same x-vector under either talker, and only what the talker makes of it differs.

**The clone path is upstream's.** Upstream's talker.py `_load_voice` takes `voices/<name>.wav`:
- the channels are averaged and the first `SOPHON_MAX_REF_SECONDS` (15) kept;
- vLLM-Omni's `extract_speaker_embedding` (librosa to 24 kHz, qwen-tts's mel, the loaded talker's ECAPA encoder in bf16) makes the x-vector;
- the prompt is the x-vector-only one. There is no ICL, no transcript, no loudness normalisation, and no `.pt` written.

Ours is `Talk::clone_voice` (15 s, our resample, speaker.rs). `reference/sophon/talker_ref.py --ref-wav` on Skippy1's
15.9 s recording, then `audio check --stages talker` (B65), gives:
- the x-vector: rel 1.1e-6 to upstream's float32 one, and rel 1.6e-3 (cos 0.9999996) to the bf16 one upstream feeds;
- the first code 1995 in both, and the first 51 greedy frames identical.

**The A/B.** Settings:
- `audio check DIR --stages speak` (lines.json: `voices` and `temperature`), the talker alone on the B65;
- two lines with no tags, seeds 1-3, the four saved voices and chelsie;
- each take scored with ZONOS3-Mini's speaker encoder (`spk-encoder-cli`, the platform's; 192-d, cosine), and Whisper large-v3-turbo through `/rpc/audio.transcribe`.

At the served temperature (0.9), speaker similarity to the recording (mean of 6 takes):

| voice (recording) | Base | Flash | Base, in-context | among takes Base / Flash | WER Base / Flash |
|---|---|---|---|---|---|
| Skippy1 (15.9 s, a man's audiobook voice) | 0.606 | 0.520 | 0.696 | 0.74 / 0.75 | 0 / 0 |
| Meeseeks (11.7 s, a TV clip) | 0.553 | 0.471 | 0.707 | 0.76 / 0.80 | 0 / 0 |
| mission (10.0 s, a film clip) | 0.659 | 0.600 | 0.773 | 0.80 / 0.80 | 0 / 0 |
| keeper (6.4 s, a voice-design model's sample) | 0.835 | 0.721 | 0.834 | 0.87 / 0.89 | 0 / 0 |

What the A/B shows:
- **Flash's talker drifts.** It is 0.06-0.11 further from every recording, human or synthetic, as stable from take to take as Base, and only 0.63-0.79 like Base's own takes.
- **It matches the platform's measurement.** Skippy1 0.61 then 0.52 here, against the platform's 0.61 / 0.54 / 0.48 then 0.50.
- **Temperature 0.7 (upstream's deploy config for the talker and its predictor) does not close the gap:** Flash 0.45-0.72.
- **Pitch moves no more one way than the other** (median F0 of the takes against the recording).
- **Chelsie** (a release voice) is equally stable under both talkers (0.86-0.89 among takes).

**In-context cloning.** Qwen3-TTS's ICL prompt puts the recording's codes and its transcript in the prompt. It was run
through the qwen3-tts engine, with each talker's weights:
- **With Base it clones closest of all** (Skippy1 0.70). It still reads tags aloud ("bagels, laughter. Oh").
- **With Flash's talker it fails.** Speech ends early (WER 0.26-0.74; Skippy1 "Thank you." in 3 of 9 takes), though Keeper's "[laughter]" was performed and not read.

So no measured path gives both a faithful clone and performed tags. The fine-tune learned the x-vector prompt alone.
ICL with Base is now the plain entry's way (next section).

## In-context clones (2026-11-17): the plain entry's default

`NS_SOPHON_CLONE=icl` (sophon-realtime-32b-flash's card; -tags keeps `xvector`): a cloned voice whose transcript is
known speaks through Qwen3-TTS's own ICL prompt (qwen-tts `generate_icl_prompt`, streaming), built in speak.rs:
- the codec rows as before (x-vector included), then `proj(ref text ++ reply ++ tts_eos)` beside codec
  `[bos, the recording's frames]` (the 12 Hz codec's encoder, crates/codec12hz); a shorter text is padded with
  tts_pad, a longer one fed a token a frame after the prompt;
- the codec decodes the new frames after the recording's (their context), as qwen-tts does;
- the transcript's ids are qwen-tts's `ref_ids[3:-2]` (the reference template, NFC, the thinker's tokenizer).

**Where the transcript comes from.** A request's `ref_text`, or a saved voice's record. The front door hears every
saved voice once with Whisper (glue/serve home/transcribe.rs, `NS_VOICE_TRANSCRIBE`): `text` filled when empty,
`words` (their times) and `transcribed` kept. With word times the engine cuts the recording between two words (after
the last sentence ending in the second half of its first `NS_SOPHON_ICL_MAX_S`, 16 s) and takes that part's
transcript from the words. No transcript: the x-vector; over the limit with no word times: the x-vector.

Why the words, not the stored text: Skippy1's stored transcript is "Heidi, hold their monkeys. Tis eye, ...", Whisper
hears "Hidey-ho there, monkeys. Tis I, ...", and the 15.9 s recording ends on a clipped "Since...". With the stored
text the talker ran on past a 30-token line (to the frame guard, 28-34 s) in 5 of 8 seeds - and so does the
qwen3-tts engine with the same inputs (4 of 8 to its 60 s cap): the model, not the port. With Whisper's words: 0 of 8
whole, 0 of 8 cut at 15.2 s, 0 of 8 cut at 5.9 s. Meeseeks: 0 of 8 either way.

**Caching.** `Talk::icl` keeps the last 16 prepared recordings (x-vector, frames, codec rows, the transcript's
projected rows) by a fingerprint of the samples kept and the text: 0.4-0.9 s to make (B65), then nothing; a
/v1/realtime session makes it at session.update. The prompt itself (~210 rows for 15 s) is prefilled each reply.

**Exactness** (`reference/sophon/talker_ref.py --ref-wav --ref-text`: qwen-tts's ICL path on the CPU in float32,
Base, Skippy1's 15.9 s at 24 kHz; `audio check --stages talker`, B65): the transcript's 48 ids identical; the x-vector
rel 9.2e-7; the recording's codes 3177 of 3184 (every first codebook); prompt rows rel 2.4e-2 from our frames (those
7 codes), rel 7.2e-5 from the reference's; first logits rel 3.1e-4, the same first code; forced through the
reference's 122 greedy frames, 0 first-code misses and 8 of 1830 other codes. Free-running greedy parts after 3
frames on a predictor near-tie (as the qwen3-tts engine's ICL check found).

**Quality** (the A/B's method: `audio check --stages speak`, Base on the B65, two lines x seeds 1-3, T 0.9; ZONOS3
similarity to the recording, Whisper large-v3-turbo WER; mean of 6 takes):

| voice | x-vector | qwen3-tts engine ICL | Sophon ICL, stored text | Sophon ICL, Whisper words, 16 s (default) | 10 s | WER (default) |
|---|---|---|---|---|---|---|
| Skippy1 | 0.606 | 0.696 | 0.684 (1 run-on, WER 0.14) | 0.714 (cut 15.2 s) | 0.601 (5.9 s) | 0 |
| Meeseeks | 0.553 | 0.707 | 0.753 | 0.699 (cut 11.0 s) | 0.681 (5.6 s) | 0 |
| mission | 0.659 | 0.773 | 0.764 | 0.746 (9.4 s) | 0.746 | 0 |
| keeper | 0.835 | 0.834 | 0.850 | 0.872 (6.2 s) | 0.872 | 0 |

**/v1/realtime, first audio** (reply start to its first audio, `reference/sophon/client.py --options`, a build copy
serving both cards, saved voice Skippy1 with Whisper's words; two sessions each; the brain at tg ~20 tok/s):

| question (reply) | x-vector | in context, 16 s (default) | in context, `icl-max-s` 10 |
|---|---|---|---|
| en_know (30 tokens, 7.9 s spoken) | 0.82 / 1.05 s | 2.22 / 2.75 s | 2.04 / 2.06 s |
| en_chat (13 tokens, 3.3 s spoken) | 1.48 / 1.50 s | 1.61 / 1.81 s | 1.43 / 1.44 s |

In upstream's layout the reply's text sits beside the recording's frames, so the talker starts when the reply covers
them (190 frames here: never) or is whole - a reply's whole writing time, +1.2-1.7 s on a 30-token answer, little
on a short one. No run-on in these sessions (7.84 s spoken, as the x-vector's 7.86); before the words cut the stored
transcript's prompt ran on to 33.7 s in all three sessions.

`NS_SOPHON_ICL_LIVE=stream` (the reference's text over its frames, the reply fed as written) fails: WER 1.0 in all 24
takes ("Thank you.", "Hmm"). It stays an option for experiments only.

Files: hardac1 `~/src/talkerab-out` (ab/<config>/*.wav, transcripts.json, scores.json; emb/), scripts
`~/src/talkerab-scripts`.

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
