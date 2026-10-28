# Audio

[← nextsycl](../README.md)

MiniMax Music 3 (`audio/minimaxmusic3`): a song from its lyrics and a description of it - genre, tempo and key, mood,
the voice, the instruments, the arrangement.

```sh
nextsycl models pull minimax-music3 [--dir DIR] [--from DIR]   # --from: the checkpoint's directory, if downloaded
nextsycl audio gen "Genre: indie folk. BPM: 84. Vocals: male lead, gentle. Arrangement: acoustic guitar, banjo." \
                   --lyrics-file song.txt [--seconds 60] [--seed N] [--steps 30] [--cfg 1.7] [--out FILE|DIR] [--opt-int8 1]
nextsycl audio serve [minimax-music3] [--wfe] [--port 8087] [--host H] [--out DIR] [--gpu N]
nextsycl audio check <reference dump dir> [--stages ar,dit,voc,chunks] [--gpu N]
nextsycl audio engines | selftest [--gpu N]
```

Lyrics take structure tags on lines of their own (`[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`,
`[instrumental]`, `[solo]`, `[outro]`; text after a tag on its line is dropped, as the model was trained); no lyrics
makes an instrumental. `--seconds` is a limit: the model ends a song when it is done. The WAV carries the description,
the lyrics and the settings (its INFO chunk).

- **The pipeline**, one card: the prompt in the checkpoint's template and its classifier-free twin; the language
  model draws a semantic code a frame (25 a second, guidance 1.5, the best 50) and the depth decoder the seven residual
  codes after it, two rows a frame on the engine's own small-batch products and cached attention, the draws on the
  host; each frame's eight hidden states mixed into one conditioning row; the flow transformer makes the Flow-VAE
  latents in 200-frame windows 100 apart (30 Euler steps, guidance 1.7), each blended into the previous over their
  overlap; the decoder (DAC's, its convolutions on oneDNN) a window at a time, the overlaps cropped.
- **Checked** against MiniMax's own diffusers code (`reference/minimaxmusic3/ref.py`, float32 on the CPU, the same
  files), each stage from the reference's input: the prompt's tokens equal; the language model and the depth decoder
  over 51 frames forced to the reference's codes rel 4e-4 (half; int8 2-3e-2, cosine 0.9997); the condition 2e-6; the
  flow transformer's first velocity 1.4e-3 and the latents after 30 steps 1.2e-3; the decoder 8e-7; three windows
  with their overlaps, stitched, 3.6e-3.
- **The server** (`nextsycl audio serve`, the container `nextsycl-audio`): the reference server's endpoint - `POST
  /v1/audio/speech` with `input` (the lyrics), `instructions` (the description), `seed`, `max_new_tokens` (frames,
  25 a second) or `seconds`, and ours: `steps`, `cfg`, `options` - answering the WAV, or with `"response_format":
  "url"` JSON with its link; `/api/progress` (the phase: prompt, tokens, flow, decode), `/api/cancel`,
  `/api/history`, `/v1/audio/files/<f>` (byte ranges, for the player).

```sh
curl localhost:8087/v1/audio/speech -H 'content-type: application/json' -o song.wav -d '{"input": "[verse]\nCity
  lights are calling out my name\n[chorus]\nTonight we run", "instructions": "Genre: funk pop. BPM: 112. Slap bass,
  brass stabs. Vocals: confident male lead.", "seed": 11, "max_new_tokens": 1500}'
```

With `--wfe` the page at `/`: the description, the lyrics with the structure tags a click away, the length, the
seed, the steps and guidance; the phase and its progress while a song is made, a cancel; each song with its player,
settings, lyrics and the card's energy (`docs/screenshots/shoot.py audio http://localhost:8087 docs/screenshots`
retakes these):

| Idle | Composing | Done |
|---|---|---|
| ![idle](screenshots/audio-wfe-idle.webp) | ![composing](screenshots/audio-wfe-running.webp) | ![done](screenshots/audio-wfe-done.webp) |

<img src="screenshots/audio-wfe-phone.webp" width="260" alt="the page at phone width">

A minute of song (`nextsycl audio gen --seconds 60`, a pop-rock description and lyrics long enough not to end
sooner, 30 steps; 2026-10-09), each card alone; its phases as `gen` reports them. VRAM: the card's use with the
model loaded, and its peak over a 20 s song (the cache and the flow stage's buffers), through the server:

| card, language model | 60 s of song | tokens (1,500 frames) | flow (14 windows x 30 steps) | decode | VRAM, loaded / peak |
|---|---|---|---|---|---|
| B70, half | 117.8 s | 71.8 s · 20.9 frames/s | 42.0 s · 0.10 s/step | 3.8 s | 19.4 / 22.7 GiB |
| B70, int8 (`--opt-int8 1`) | 88.8 s | 42.7 s · 35.2 frames/s | 42.2 s | 3.8 s | 12.3 / 15.4 GiB |
| B70, int8 + flow int8 (`--opt-dit-int8 1`) | 80.0 s | 42.5 s · 35.3 frames/s | 33.5 s · 0.08 s/step | 3.7 s | |
| B65, half | 154.8 s | 89.1 s · 16.8 frames/s | 60.7 s · 0.14 s/step | 4.8 s | the same |
| B65, int8 | 136.2 s | 70.5 s · 21.3 frames/s | 60.7 s | 4.8 s | the same |

Real time is 25 frames a second: the B70 composes faster than real time in int8. `--opt-dit-int8 1` keeps the flow
transformer's block matrices as int8 ConvRot (rotated by the 256 x 256 Hadamard matrix along their inputs, quantized
per row; activations quantized on the fly): the latents after 30 steps 8.6e-3 off the reference (half 1.2e-3), cosine
0.99996. The language model's frame reads
all of its weights and, seven times, the depth decoder's (14 GB and 8 GB a frame in half) - ~78% of the B70's
memory bandwidth. A load takes 6 s with the files in the page cache (20 s from the disk). Running the flow stage
beside the frames on a second queue of the same card was tried: the card takes the two in turns (114.5 s, not less).

## Speech

Qwen3-TTS (`audio/qwen3tts`, the 12 Hz 1.7B checkpoints): text to speech in Chinese, English, Japanese, Korean, German,
French, Russian, Portuguese, Spanish and Italian, 24 kHz mono. One engine, three checkpoints: **CustomVoice** (nine
built-in voices - `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee` - and an
instruction for the read), **VoiceDesign** (the voice made up from a description), **Base** (the voice of a
recording: its x-vector alone, or with its transcript the recording itself as the example the model continues).

```sh
nextsycl models pull qwen3-tts-custom qwen3-tts-design qwen3-tts-base
nextsycl audio gen "Good morning. The bread is still warm." --model qwen3-tts-custom --voice ryan --language english \
                   [--instructions "cheerful, a little hurried"] [--seed N] [--out FILE|DIR] [--opt-int8 1]
nextsycl audio gen "The lighthouse keeper counted the ships." --model qwen3-tts-design \
                   --instructions "A calm elderly woman with a soft, slightly raspy voice, speaking slowly."
nextsycl audio gen "I never thought the garden would grow this fast." --model qwen3-tts-base --ref-audio voice.wav \
                   [--ref-text "what the recording says"]
nextsycl audio check <reference dump dir> --model qwen3-tts-custom [--stages prompt,frames,codec]

# saved voices: kept by name, spoken by any model that clones (--voice NAME, "voice": "NAME")
nextsycl audio voice add grandpa --sample grandpa.mp3 --text "Well, back in my day we walked to school."
nextsycl audio voice design narrator --instructions "A deep, warm male narrator in his fifties, unhurried." [--seed N]
nextsycl audio voice list | rm NAME
nextsycl audio gen "Chapter one." --model qwen3-tts-base --voice narrator
```

A **saved voice** is a recording and what it says, in `<audio output>/voices/<name>/`: cloned from a sample (WAV,
MP3, FLAC, Ogg Vorbis or M4A, decoded by symphonia), or designed - the front door has the voice-design model read a
ten-second line in the voice described and keeps that. A request naming it (`audio.speak`, `/v1/audio/speech`) with
no model goes to an enabled model that clones, as an in-context clone; `audio.voices.add | design | list | remove`
on the API, chips and "save voice" buttons on the page.

- **The pipeline**, one card: the prompt as qwen-tts builds it (projected text tokens plus codec control tokens: the
  language, the speaker's row - a built-in voice's embedding or the recording's x-vector - and the instruction ahead);
  the talker (a 28-layer Qwen3 of 2,048) draws a frame's first code, 12.5 frames a second, its code predictor (5 layers
  of 1,024) the other fifteen, each frame fed back with the next text token - the whole frame on the GPU as one
  recorded SYCL graph, fused kernels and the draws on the device (the engine's silo; without it the MiniMax Music
  engine's kernels with the draws on the host); the codec's decoder (an 8-layer windowed transformer,
  ConvNeXt upsampling, SnakeBeta convolution blocks on oneDNN) turns 16 codes a frame into 1,920 samples. The speaker
  encoder (ECAPA-TDNN on a 128-band log mel) runs on the host. The in-context clone encodes the recording with the
  codec's encoder (Mimi's: SEANet convolutions, an 8-layer causal transformer, split-RVQ nearest codes) and puts its
  frames under the reference transcript's tokens; the recording's frames are decoded ahead of the new ones and cut
  off, as qwen-tts does.
- **Checked** against qwen-tts' own code (`reference/qwen3tts/ref.py`, float32 on the CPU, the same files): the prompt
  rows rel 9e-5 to 2e-4, the first frame's logits 3e-4 to 8e-4; greedy frames identical to the reference's (VoiceDesign
  46 of 46, a clone 59 of 59; CustomVoice all 16 codes through frame 143 and 2,046 of 2,047 first codes, half-precision
  drift after); with the reference's frames forced, the first code differs in 0 of 2,047 frames and the others in 1 of
  30,705 (a near tie - the greedy runs part there); the codec 1e-6 to 1.7e-3; the clone's speaker embedding 8e-7; the
  recording's codes from the encoder 1,152 of 1,152. Every mode's speech transcribes word for word (Whisper); a clone
  is 0.990 (x-vector) / 0.992 (in context) like its recording by the speaker encoder's cosine, another voice 0.943.
- **Speed** (2026-10-18; a 70-word paragraph, sampled, seed 7; the frames on the device as one graph): the B70 makes
  82 frames/s in half (6.6x real time), 125 with `--opt-int8 1` (10x); the B65 72 / 94 (before the graph, 2026-10-09
  path: B70 70 / 93, B65 66 / 71). Through the audio server (B70, half): "Good morning. The bread is still warm."
  answers in 0.64 s (3.6 s of speech), a minute of speech takes 0.18x its length (0.14x int8); B65 0.65 s, 0.22x
  (0.18x). A frame reads 5.3 GB of weights in half (2.8 GB int8); at the 530 GB/s a product reaches on the B70 that
  is 10 ms (5.3 ms) a frame - the frame takes 12.3 ms (8.3 ms), the rest the ~550 small kernels' ramps, the
  attention and the draws. The codec decodes a minute in ~0.9 s after its chunk lengths' first use (two are made at
  load: +1.4 s). A model loads in 2-3 s from the page cache.
- **Served** by the same audio server (`nextsycl audio serve qwen3-tts-custom --wfe`) and `:8000`'s `audio.speak`:
  `/v1/audio/speech` with `input` (the text), `voice`, `language`, `instructions`, `ref_audio` (a WAV, base64 or a
  data: URL), `ref_text`, `seconds` (the most), `seed`, `options` (`greedy`, `temperature`, `top-k`, `streaming`).
  The page takes the text, the voice (chips), the description or instruction, a recording to clone and its
  transcript, the language.
- **OpenAI's shape** (`/v1/audio/speech`): `response_format` `wav`, `pcm` (16-bit mono at 24 kHz, `X-Sample-Rate`
  says), `mp3`, `opus` (Ogg), `aac` (ADTS), `flac` or `url`; `speed` 0.25 - 4 (pitch kept); `sample_rate` (8000 -
  96000). `pcm` and the encoded formats are sent as the sound is made, chunked, the first sentence in under a second:
  `curl localhost:8000/v1/audio/speech -d '{"model": "qwen3-tts-custom", "input": "Hello.", "voice": "ryan",
  "response_format": "pcm"}' --output - | ffplay -f s16le -ar 24k -ac 1 -nodisp -`. wav is whole unless `stream:
  true`. Anything but native wav/pcm goes through the server's `ffmpeg` (a 400 says when it is missing).
- **Merging**: `input` as a list (`["One.", "Two."]`) is said in turn as one stream - the same voice and seed, `gap`
  seconds (0.3) of silence between - and arrives as one file or one chunked stream. Parts that fail end the stream
  uncleanly (the chunks are not terminated), so a client sees a cut, not a short file. `response_format: url` takes
  one text at normal speed (a list or speed asks for a stream).

## Breeze TTS 2

Breeze TTS 2 (`audio/breezetts2`, BreezeBlue's CSM fork, ~3B; CrispASR's GGUF of it): English and Chinese speech at
24 kHz. A T5Gemma2 text encoder (26 layers, bidirectional, a symmetric sliding window) reads each text segment of the
prompt; a Qwen3 backbone (28 layers) draws a frame's first code, a 12-layer depth decoder the other fifteen of the
Qwen3-TTS 12 Hz codec (the same codec file, shared through `crates/codec12hz`). Four ways: **design** (`--instructions`:
the voice described, in the language of the text), **clone** (`--ref-audio` + `--ref-text`, the exact transcript),
**direction** (a clone + `--instructions`: tone, emotion, pace), plain; guidance `--cfg` 4 with an instruction. Vocal
events inline: `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)`; `[笑]`, `[叹气]`, `[咳嗽]`, `[清嗓子]`. **License:
the weights and their outputs are research and non-commercial only** (BreezeBlue Research and Non-Commercial License
1.1); the upstream code is Apache-2.0.

```sh
nextsycl models pull breeze-tts-2
nextsycl audio gen "(sigh) Welcome aboard. Your journey begins now." --model breeze-tts-2 \
                   --instructions "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."
nextsycl audio gen "[叹气] 没想到过了这么久，你还记得我的声音。" --model breeze-tts-2 --ref-audio r.wav --ref-text "..." \
                   [--instructions "用激动的语气，语速稍快。"] [--cfg 4] [--seed N] [--opt-temperature 0.9]
nextsycl audio check <reference dump dir> --model breeze-tts-2 [--stages prompt,frames,codec]
```

- **Ported**: the GGUF's matrices expanded to half at load (F16 / Q8_0 / Q4_0 / Q4_K on the host; the text embedding
  read from the file a row a token); the backbone and depth decoder on the music engine's cached decode (the depth
  decoder's guided pair as one batch of two); this engine's kernels (`kernels/audio/breezetts2`) for the text encoder's
  windowed bidirectional attention, its GELU-tanh gate and the table-driven RoPE; the prompt of upstream's four
  templates; Gemma's tokenizer (`nextsycl_tok`, SentencePiece-style).
- **Checked** against upstream's own code (`reference/breezetts2/ref.py`, float32 on the CPU, greedy; prompts of all
  four templates, English and Chinese): token ids identical; the recording's codes 1,152 of 1,152; with the f16 GGUF
  the text encoder 4-5e-4, the prompt rows 7-9e-4, each frame's backbone logits 4e-4 and depth logits 8-9e-4 (teacher
  forced: no first code differs); with the served Q8_0 file 6e-3, 7-1e-2, 3-5e-3 and 8e-3 (its quantization); the
  codec 1.4-5e-6. Real clips in every mode, English and Chinese, transcribe word for word (Whisper small).
- **Speed** (B70, 2026-10-16): 22 frames a second with guidance (design, direction: 1.8x real time), 28 without
  (clone: 2.2x); 5.4 s of designed speech in 3.9 s with the decode. 5.4 GiB of weights, 6.1 GiB in a run.
- **One server a card**: two Breeze servers on one card make less than half the audio one does (B70 1.54x real time
  alone, 0.65x for the pair: the driver time-slices them). Several clips in one engine instead (batched decode, ~3-4x
  estimated) is a proposal for review: `docs/proposals/speech-batching.md`.

## VibeVoice

VibeVoice-1.5B (`audio/vibevoice`, Microsoft; gguf-org's GGUF of it): long-form, multi-speaker speech in English and
Chinese at 24 kHz. A script of up to four speakers (`Speaker 1: ...`, a turn a line) is read by a Qwen2.5-1.5B
language model; for each 7.5 Hz frame a diffusion head draws a 64-value acoustic latent (10 DPM-Solver++ steps, guided
by a second, negative run of the language model: `--cfg`, 1.3), the σ-VAE acoustic decoder turns it into 3,200
samples, and the semantic encoder reads them back for the next step. Each speaker's voice is cloned from a recording
(no transcript needed: `--ref-audio` once a speaker in order, or saved voices `--voices NAME,NAME`); a speaker without
one gets a voice of the model's own. The model ends the speech itself (up to 90 minutes). **License: MIT; Microsoft's
model card restricts its use - no imitation of a real person's voice without their consent, no passing its speech off
as real, disclose that the audio is AI-generated.**

```sh
nextsycl models pull vibevoice-1.5b
nextsycl audio gen "Speaker 1: Welcome back to the show.\nSpeaker 2: Thanks for having me." --model vibevoice-1.5b \
                   --ref-audio host.wav --ref-audio guest.wav [--cfg 1.3] [--steps 10] [--seed N]
nextsycl audio gen "Good morning. The bread is still warm." --model vibevoice-1.5b [--voice SAVED]
nextsycl audio check <reference dump dir> --model vibevoice-1.5b [--stages prompt,lm,head,codec,run]
```

The audio server and the front door take a script's voices as `voices` (saved voices) or `ref_audios` (recordings), a
speaker each in order; `audio.speak` picks a model of several speakers (card `speech: speakers`) for them.

- **Ported**: the language model's matrices kept Q8_0 on the card (codes and block scales), the tokenizers' the file's
  bfloat16, the head and connectors half; the token embedding read from the file a row a token, the tied head's four
  tokens on the card; the tokenizer and config.json from the -hf checkpoint. All on the engine's own kernels
  (`kernels/audio/vibevoice`): the language model's decode step (8 launches a layer, its two guidance branches in one
  paired pass, the attention split over the keys), the diffusion head's ten steps without a read-back (every step's
  modulations in one product a frame, DPM-Solver++ on the device), the tokenizers streaming a frame at a time (each
  convolution a product over windows of its carried rows and input, the high stages on the matrix engines with x split
  into two bfloat16 halves: float32-accurate).
- **Checked** against transformers' implementation (`reference/vibevoice/ref.py`, float32 on the CPU; its loop
  bit-identical to `generate`'s frames): token ids identical; with the f16 GGUF the language model's logits 2-5e-4 and
  hidden states 0.7-2.2e-3 / 5-6e-4 teacher forced (every token agrees), with the served Q8_0 2e-3 and 1.4-2e-2 (its
  quantization); the diffusion head 4e-7, the acoustic decoder 2-3e-6, the semantic encoder 4e-6 to 8e-5, the voice
  encoder 2-9e-6; Chinese token ids identical, and a run that splits into two audio segments (the negative branch's
  restart, the streams' reset) as close. Clips plain, cloned, two-speaker and a 46 s show transcribe word for word
  (Whisper small; twelve seeds of the show: 1.2% word error); the speakers alternate as cast (pitch).
  `docs/ports/vibevoice.md` has the details.
- **Speed** (2026-10-18): 79.3 frames a second on the B70 (10.6x real time: 46.5 s of a two-speaker show in 4.9 s with
  the prompt; 5.9 s through the server), 64.0 on the B65 (8.5x; 7.1 s through the server). A short sentence through
  the server 0.31 s (B70) / 0.38 s (B65). A frame on the B70 is 12.6 ms against a floor of 6.4 ms of weights read
  (`docs/ports/vibevoice.md`: what is left). 3.6 GiB of weights, 4.0 GiB in a run.

## S2 Pro

Fish Audio S2 Pro (`audio/s2pro`, ~4.6B; s2.cpp's GGUF of it, the codec in the same file): speech in about 80
languages at 44.1 kHz (best in English, Chinese and Japanese). A Dual-AR model: a 36-layer Qwen3 slow model draws each
frame's semantic code, a 4-layer fast model the other nine of Fish's RVQ codec (21.5 frames a second). What it really
does:

- **clone** (`--ref-audio` + `--ref-text`, the exact transcript; or a saved voice with one): yes. Without the
  transcript it does not clone (upstream drops the recording).
- **direction**: yes, by S2 Pro's own mechanism - free-form tags in square brackets anywhere in the text (`[whisper]`,
  `[laughing]`, `[excited tone]`, `[pause]`, `[pitch up]` ...; 15,000+ in training). `--instructions` becomes such a
  tag at the start of the text (and of each speaker turn).
- **vocal events**: yes, as tags (`[laughing]`, `[sigh]`, `[inhale]`, `[clearing throat]`, `[chuckle]`, `[panting]` ...).
- **several speakers**: yes, `<|speaker:0|>`, `<|speaker:1|>` turns in one text, read in batches that carry the
  conversation on.
- **design**: no (no description input, no built-in voices). Without a recording the voice is the model's pick for
  the text, and it changes with the seed.

**License: the weights and their outputs are research and non-commercial only** (Fish Audio Research License;
commercial use needs Fish Audio's written license).

```sh
nextsycl models pull s2-pro
nextsycl audio gen "Hello! [laughing] Isn't that something?" --model s2-pro [--seed N]
nextsycl audio gen "I have a secret to tell you." --model s2-pro --ref-audio r.wav --ref-text "..." [--instructions "whisper"]
nextsycl audio gen "<|speaker:0|>Did you hear that? <|speaker:1|>[whisper] Hear what?" --model s2-pro
                   [--opt-temperature 0.7] [--opt-top-p 0.7] [--opt-top-k 30] [--opt-greedy 1]
nextsycl audio check <reference dump dir> --model s2-pro [--stages prompt,frames,codec]
```

- **Ported**: the GGUF's matrices kept as Q8_0 on the card (int8 codes and half scales), read by this engine's own
  decode product (`kernels/audio/s2pro`). A prompt's wide products expand them to float32 for oneDNN's GEMM - half
  overflows on some prompts' activations.
  - The embeddings stay in the file (a row a token; a frame's 11 rows summed).
  - fish-speech's RoPE on interleaved pairs with its bfloat16 table.
  - The attention, cache and embeddings on the music engine's kernels.
  - Fish's codec (decoder, encoder, quantizers) in float32 on the Qwen3-TTS codec's and the shared kernels.
  - fish-speech's conversation prompt, speaker batches, inline-tag instructions, and its draws (top-p before the
    temperature, repetition-aware resampling).
- **Checked** against fish-speech's own code (`reference/s2pro/ref.py`, float32 on the CPU, greedy; plain, clone,
  Chinese, a 170-token paragraph):
  - prompts identical; the recording's codes 2,137 of 2,140 (every semantic code)
  - with the F16 file, the slow logits 1.1-1.3e-4 and the fast logits 0.4-1.3e-4, teacher forced; greedy frames
    identical (40 / 40, 30 / 30) up to a near tie
  - with the served Q8_0 file, 6e-3 and 1-5e-3 (its quantization)
  - the codec 7e-7 .. 2e-6 against the reference with the GGUF's half weights (5-8e-4 against codec.pth's float32)
  - real clips in every mode, English, Chinese, Japanese, a dialogue, transcribe word for word (Whisper small)
  - details: [docs/ports/s2pro.md](ports/s2pro.md)
- **Speed** (B70, 2026-10-17): 42 frames a second (2.0x real time for the frames); a 42.8 s paragraph in 24.4 s with
  the decode (1.75x). 5.6 GiB on the card loaded, 6.5 GiB at a long clip's peak.
- **Served** like the other speech models: `nextsycl audio serve s2-pro` (`/v1/audio/speech`: `input`, `ref_audio` +
  `ref_text` or a saved `voice` with its transcript, `instructions` as a tag, `seconds`, `seed`, `options`) and
  `:8000`'s `audio.speak`.

## Voxtral TTS

Voxtral 4B TTS 2603 (`audio/voxtraltts`, Mistral; CrispASR's GGUF of it): speech at 24 kHz in English, French,
German, Spanish, Italian, Portuguese, Dutch, Arabic and Hindi, in 20 **preset voices** - `neutral_female` (the
default), `neutral_male`, `casual_female`, `casual_male`, `cheerful_female`, and a female and a male voice for each
other language (`fr_female`, `de_male`, ... ; Arabic `ar_male` only). A Ministral 3B backbone (26 layers) makes a hidden
state per 12.5 Hz frame; a 3-layer flow-matching transformer turns it into the frame's semantic code and 36 acoustic
levels (7 guided Euler steps, `--cfg` 1.2); the Voxtral codec's decoder makes 240 samples a frame. `--language` alone
picks that language's voice. **What it does not do**: the open checkpoint ships the codec's decoder but not its encoder
(no `audio_tokenizer.encoder_blocks` / `input_proj` tensors; vllm-omni refuses `ref_audio` for it), and its prompt has
no instruction - so **no cloning from a recording, no voice design or direction, no vocal events**. **Experimental:**
a saved voice can be **enrolled** (`nextsycl audio voice enroll NAME`: its codes fitted through the released decoder
with PyTorch XPU, ~1-2 minutes a voice on a B65) and is then spoken as `--voice NAME`; it measured closer to the
speaker than the nearest preset for all four voices tried, a little below Breeze's and VoxCPM2's clones
(`docs/ports/voxtraltts.md`). **License: CC BY-NC 4.0** - the weights, the preset voices and the speech they make are
non-commercial.

```sh
nextsycl models pull voxtral-4b-tts
nextsycl audio gen "The quick brown fox jumps over the lazy dog." --model voxtral-4b-tts --voice casual_male
nextsycl audio gen "Le soleil se couchait lentement sur la mer." --model voxtral-4b-tts --language french \
                   [--cfg 1.2] [--steps 7] [--seed N]
nextsycl audio check <reference dump dir> --dir <f16 GGUF dir> [--stages prompt,frames,flow,codec]
nextsycl audio voice enroll NAME [--gpu N]          # experimental; NS_VXT_CHECKPOINT / NS_VXT_VLLM_OMNI / NS_VXT_TORCH
nextsycl audio gen "..." --model voxtral-4b-tts --voice NAME
```

- **Ported**: the GGUF's matrices expanded to half at load (q and k rows reordered so the rotate-half RoPE kernel
  turns the checkpoint's adjacent pairs); the backbone on the music engine's cached decode and Breeze's RoPE; the
  acoustic transformer's two guidance passes as one batch of six rows with the whole ODE on the GPU; this engine's
  kernels (`kernels/audio/voxtraltts`) for the codec's causal ALiBi attention, its whole-row q / k norms and causal
  padding; Tekken (tiktoken BPE) in `nextsycl_tok`; vllm-omni's loop and codec where CrispASR's runtime differs (ALiBi
  slopes, the window's W + 1 keys, replicate padding, 375-frame decode chunks, frame 0 from the prompt's last position:
  `docs/ports/voxtraltts.md`).
- **Checked** against upstream's own code (`reference/voxtraltts/ref.py`: vllm-omni's acoustic transformer and codec
  run as they are, float32 on the CPU from the safetensors; English and French): token ids identical; with the f16
  GGUF the prompt rows 3e-9, each frame's hidden state 3-6e-4 and semantic logits 1-2e-4 teacher forced (no semantic
  code differs, 2-3 of thousands of acoustic levels), the flow alone 5e-7..2e-6 (no level differs), the codec 5.5e-4.
  Clips in all nine languages transcribe word for word (Whisper small).
- **Speed** (B70, Q8_0, 2026-10-17): 30 frames a second, 2.4x real time; 34.6 s of speech in 15.2 s with the prompt
  and decode. 6.8 GiB of weights, 7.2 GiB in a run. B65: 22 frames a second.

## VoxCPM2

VoxCPM2 (`audio/voxcpm2`, OpenBMB's 2B checkpoint): tokenizer-free speech in 30 languages at 48 kHz. No audio tokens:
each step makes a patch of four AudioVAE latents by flow matching (a 12-layer local DiT, 10 Euler steps, guidance 2.0
with CFG-zero*), conditioned on a MiniCPM4 base language model (28 layers; its outputs at audio positions through a
scalar quantizer) and a residual one (8 layers), the patch fed back through a 12-layer local encoder; a stop head ends
it. A description in `instructions` designs the voice (put in parentheses ahead of the text, as the model was
trained); a recording is the reference (isolated by its own tokens); with its transcript also the prompt it continues
(the "ultimate" clone).

```sh
nextsycl models pull voxcpm2
nextsycl audio gen "Bonjour à tous." --model voxcpm2 [--instructions "A cheerful young French woman"] [--ref-audio r.mp3 [--ref-text "..."]]
                   [--voice SAVED] [--steps 10] [--cfg 2] [--seed N] [--opt-int8 1]
nextsycl audio check <reference dump dir> --model voxcpm2 [--stages prompt,prefill,patches,vae]
```

- **Ported**: the two language models and the local transformers on the music engine's kernels (attention now also 8
  query heads a key head) and this engine's (`kernels/audio/voxcpm2`: the long-RoPE table, attention within short
  groups of rows, Snake, the dilated causal depthwise convolution); AudioVAE V2 (weight-normalized causal convolutions,
  the 48 kHz sample-rate conditioning) on oneDNN; its weights from `audiovae.pth` read by a small ZIP + pickle reader
  (`pth.rs`: no Python); the tokenizer's SentencePiece-style BPE (byte fallback, Chinese words split into characters as
  the reference does).
- **Checked** against voxcpm's own code (`reference/voxcpm2/ref.py`, float32 on the CPU, its noise recorded): prompt
  tokens identical (plain, design, clone); the recording's latents rel 1.6e-6; both language models' prefill 4e-4 to
  1e-3; each patch from the reference's history and noise 1e-3 to 4e-2 (cos 0.99996+); the VAE 1.3e-6 to 2.8e-6.
  Run free, the half-precision differences grow through the quantizer and the guidance until a run takes another
  (equally valid) path - as two runs of the reference in two precisions do. Every mode's speech transcribes word for
  word (Whisper; Chinese and French too); a clone is 0.98 like its recording, another voice 0.92.
- **Speed** (B70, 2026-10-10): 22 patches a second (6.25 is real time): 9.9 s of speech in 3.2 s, the decoder 0.4 s.
