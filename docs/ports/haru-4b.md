# Haru-4B: port brief

Haru-4B (higurashi.ai, November 2026) is a full-duplex spoken dialogue model: it listens and speaks at the same time
in lockstep - every 133 ms (7.5 Hz) one frame of the user's audio in, one frame of its own audio and text out,
silence included. nextsycl runs it as the `audio/haru` engine (arch `haru`, card `haru-4b`) behind `/v1/realtime`
with the lockstep extension ([../api/realtime.md](../api/realtime.md), "Lockstep models"), and the wfe audio page's
Conversation panel talks with it (microphone in, speaker out, the running transcript, a character card).

## Sources

- **Code**: https://github.com/higurashi-ai/Haru (Apache-2.0): `haru/haru` (the PyTorch implementation: `models/haru.py`,
  `models/meanflow_head.py`, `models/vibevoice_codec.py`, `session.py`, `prompt.py`, `sampling.py`, `assets/` - the card
  templates and the "chat" cards), `haru_mlx` (the same in MLX), `haru_vllm`. Cloned to
  `/var/mnt/2TBSSD/nextsycl-ref/haru-src/Haru` on hardac1 (Python run only with `-I`).
- **Weights**: https://huggingface.co/higurashi-ai/Haru-4B @ 8d6e83bc (Apache-2.0; not gated): `model.safetensors`
  (10.5 GB, bfloat16: the language model `lm.*`, the two audio projectors, the MeanFlow head `head.*`, the codec
  `codec.*`, `latent_scale` / `latent_bias`), `config.json`, `tokenizer.json`. On hardac1:
  `/var/mnt/4TBSSD/nextsycl-models/haru-4b`.
- **GGUF**: https://huggingface.co/sakasegawa/Haru-4B-GGUF (F16, Q8_0, Q6_K, Q5_K, Q4_K; made for speech.cpp). Not used:
  the safetensors are read as they are (bfloat16 matrices are the checkpoint's own values, and the B70 keeps up with
  real time on them - below); a quantized file would only matter for memory or speed, neither of which binds.
- **Card**: `model-cards/audio/haru-4b.yml`, guide `model-cards/guides/haru.md`.

## License

Apache-2.0 code and weights, commercial use allowed. A pretrained base model released for research: it chats
naturally in character but is not an assistant and not a source of facts. It imitates the voice of a reference clip:
clone only your own voice or one you have permission to use. The card templates and default cards (`audio/haru/src/
cards.json`) are the release's assets, Apache-2.0.

## The model

```text
  the user's frame (3,200 samples at 24 kHz) -> the codec's encoder (causal VibeVoice acoustic tokenizer: a stem conv,
     6 stride convs 2,2,4,5,5,8 with ConvNeXt stages of 3,3,3,3,3,3,8 blocks, 32 -> 2,048 channels, a head conv) ->
     a 64-value mean; drawn around it (one N(0, 1) scale a frame x 0.5 / 0.8 x N(0, 1) noise)
  position t = embed(text token t) + model_proj((model latent t + bias) * scale) + user_proj((user latent t + bias) * scale)
     (a projector: fc1 64 -> 2,560 with bias, RMS norm, fc2 with bias)
  -> Qwen3.5-4B's text layers: 32 of 2,560 (SwiGLU 9,216), zero-centred RMS norms (1 + w); three of every four Gated
     DeltaNet (in_proj qkv 8,192 / z 4,096 / b, a 32; a causal conv of 4 + SiLU; q, k L2-normed, 16 key heads read by
     32 value heads of 128; decay exp(-exp(A_log) softplus(a + dt_bias)), beta sigmoid(b); a [128 x 128] state a head;
     a gated RMS norm with silu(z); out 4,096 -> 2,560), the fourth full attention (16 query heads of 256 with an output
     gate, 4 key / value heads, q / k RMS norms, RoPE θ 1e7 on the first 64 of each head; sigmoid(gate) x attention)
  -> the final norm: the hidden state h
     -> the tied output embedding: logits (248,320; float32) -> DRY (0.8, 1.75, allowed 1, lookback 1,024), top-k 25,
        temperature 0.7 -> text token t + 1 (PAD 248076 on silent frames, EOP 248071 the frame before it speaks)
     -> the MeanFlow head (4 adaptive-norm SwiGLU layers of 2,560 / 7,680; the condition cond_proj(h) plus the
        sinusoidal embeddings of t = 0, the interval constant 7, h = 1 and the guidance's 1 - 1/w; one step:
        z = noise + u) -> latent t + 1 = z / scale - bias -> the codec's decoder (mirror of the encoder, transposed
        convs of 2s) -> 3,200 samples
  the prompt: the card as a ChatML system turn (<|im_start|>system\n...<|im_end|>\n), then the voice reference's
     latents on the model's stream (text PAD ... PAD EOP, no user audio); speak-first adds PAD, EOP over the
     reference's quietest frame (room tone) and encoded silence on the user's side, frame 0 then forced to a word
```

The model is one frame ahead: frame 0 comes from the prompt's last position, user frame k gives frame k + 1.

## Files

- `audio/haru/src/lib.rs` - the engine: load (memory check, the parts), `generate` (offline: the model's side of a
  recording, as haru-inference does), `realtime` / `converse`, options.
- `weights.rs` - the safetensors (bfloat16 matrices stacked / interleaved for the fused products) and config.json.
- `lm.rs` - the language model: rows in groups of 16 (a frame: one), the conversation's `State` (each DeltaNet layer's
  convolution rows and recurrence, each attention layer's half cache).
- `head.rs` - the MeanFlow head; its scalar embeddings are constants of a guidance, made on the host once.
- `codec.rs` - the streaming codec: the VibeVoice engine's tokenizer code (audio/vibevoice/src/vae.rs) on its kernels,
  reading Haru's tensor names.
- `session.rs` - one conversation frame by frame (the release's DuplexSession): the prompt, `start`, `step`, the
  sampling, the reference's preparation (silence trimmed, the last 3 minutes).
- `duplex.rs` - the `Conversation`: the lockstep framing, the speech gate's events, cancel, restarts on new settings.
- `card.rs` + `cards.json` - the character card (render_card) and the release's default cards in nine languages.
- `sample.rs` - DRY, top-k / top-p, the incremental text, the random numbers. `resample.rs` - whole and streaming.
- `check.rs` - the stage checks (below). `kernels/audio/haru/haru.{h,cpp}` - its kernels: the token rows, the
  DeltaNet's convolution, recurrence and gated norm, attention's q / k norm + partial RoPE + half cache store, the
  attention split over pieces of the keys and its gated merge. The products, norms and codec blocks are the VibeVoice
  engine's (vv.h: bfloat16 weights read as they are), the add MiniMax Music's, SiLU Qwen3-TTS's - shared, called,
  not changed. A frame runs on these alone (it keeps up with real time on the shared products).
- `cache.rs` - the voices' cache: a call's prompt (the state after it) kept on the host, restored by a repeat call.
- `kernels/audio/haru/silo/hs.{h,cpp}` - the silo (`dist/silo/libnextsycl-haru.so`): the prompt's wide passes' gated
  delta rule (the state held in registers across the rows) and attention (a sub-group a row, no pieces). Without the
  file, or with `NS_SILO=0`, the engine's own kernels run them.
- `glue/serve/src/realtime.rs` - `nextsycl.card`; the lockstep extension needs nothing else of the server (the engine
  makes the events); a lockstep mock test. `wfe/audio/src/converse.ts`, `components/Converse.tsx` - lockstep play-out
  behind a 0.2 s jitter buffer that catches up after a burst, no barge-in cut, the transcript split where you spoke,
  the character card's fields.
- `reference/haru/ref.py` (the dumps), `reference/haru/client.py` (a lockstep conversation over the socket).

## Checks

`reference/haru/ref.py` runs the release's own PyTorch code in float32 on the CPU (the same bfloat16 weights, upcast)
and dumps every stage; `nextsycl audio check DUMP --dir .../haru-4b --stages codec,prompt,frames` compares, each stage
from the reference's own input. The codec's latents are its means there (no noise), the frames teacher-forced (the
reference's argmax token and its head's latent from a fixed noise).

| stage | here vs the reference (B65) |
|---|---|
| codec encoder (32 frames of speech, a frame at a time) | rel 2.3e-5, cos 1.000000 |
| codec decoder (68 frames) | rel 3.4e-6, cos 1.000000 |
| the card's ChatML ids (268) | identical |
| prompt embeddings (336 positions: card + 68 voice frames) | rel 2.8e-7 |
| after DeltaNet layers 0-2 | rel 3.5e-7 - 3.9e-7 |
| after the first attention layer (3) .. layer 31 | rel 1.2e-4 .. 2.6e-4 (the half key / value cache) |
| prompt logits | rel 1.1e-4, cos 1.000000, same argmax |
| 12 frames teacher-forced: hidden / logits / head latent | worst rel 4.8e-3 / 3.4e-3 / 5.0e-4; argmax 13 of 13 |
| a second dump, speak-first (`ref.py --speak-first`, words on the text stream): 17 frames | argmax 17 of 17 ("I", "'m", then PAD); hidden worst rel 2.2e-3, logits 1.6e-3; the head's latent rel 2e-4 - 4e-2 (largest on the frames right after speech, where it amplifies the hidden state's 1e-3) |
| the head alone (from the reference's own hidden state and noise), 17 frames | worst rel 7.2e-6 |
| a long run (`ref.py --frames 300`: a 32 s TED monologue after the default card and a 10 s voice; greedy, teacher-forced) | argmax 301 of 301; hidden worst rel 1.2e-2, logits 8.1e-3; P(PAD or EOP) at temperature 0.7 mean 0.9007 here and 0.9007 in the reference, the worst frame 3.0e-4 apart |

Run: `reference/haru/ref.py` (venv `/var/mnt/2TBSSD/nextsycl-ref/venv`, `python -I`, CPU, ~6 min), dumps in
`/var/mnt/2TBSSD/nextsycl-ref/haru/dump` (user en_chat.wav, voice a 9 s saved voice) and `dump2` (en_know.wav,
speak-first).

## Speed

`nextsycl audio check DUMP --stages speed`: the card as the prompt, then 120 frames of a live session (the user's clip,
then silence), each timed from its samples in to its samples out (encode, the language model, logits to the host,
sampling, the head, decode):

| card | a 133 ms frame | real time | the prompt (268 positions) |
|---|---|---|---|
| B70 (GPU 0) | 23.4 ms (median 23.4, worst 23.5) | 5.7x | - |
| B65 (GPU 1) | 25.0 ms (median 25.2, worst 25.3) | 5.3x | 0.9 s |

Over the socket (the audio server on the B65, `reference/haru/client.py` streaming at real time): each frame's answer
comes back 33 ms (median) after the frame was sent. The prompt delays the first frames (below: "The start of a
call"); the frames that came meanwhile are then answered at once.

Where a frame goes (`NS_HARU_PROFILE=1`, the B65, each operation waited for - so slower than the 25 ms): the codec's
encoder is the largest share (its first stages run 3,200 rows of 32-128 channels through the tiled product), the
language model's MLPs next, then attention, the DeltaNet layers, the logits (248,320 x 2,560 bfloat16) and the head.
Both cards land within 7% of each other: a frame is a few hundred small launches and host round trips (token ids,
latents, logits and the head's 64 values cross each frame), not the 10 GiB of weights it reads. Not tuned (the silo
has only the prompt's kernels).

## The start of a call

A Talk call sends a saved voice's recording (up to 60 s; a 39 s one measured) and the platform measured 1.77 s from
the session's ready to the first audio, against 0.41 s on nextcuda (its port of this engine: the reference encoded 16
frames a pass, the prompt on its qwen35 engine's wide prefill). Measured here on the B70 (GPU 0) with two saved voices
from the voice library - 38.8 s (291 frames, 559 positions with the default card) and 60.0 s (450 frames, 718
positions) - by `nextsycl audio check DIR --stages start` (voice*.wav, card.txt, user.wav in DIR: each path timed three
times, then the old path's frames against the new one's) and over the socket (`reference/haru/client.py`, a server
of the build copy, each call's session.update then 3 s of quiet).

**Where the 2 s went** (the old path, the 39 s voice; `NS_HARU_PROFILE=1` for the split, each operation waited for):

| part | time | |
|---|---|---|
| the reference through the codec's encoder | 697 ms | a frame at a time (291 x 2.4 ms: a few hundred small launches each) |
| the prompt's 559 positions through the language model | 1,347 ms | 35 passes of 16 rows on the small-batch product: MLPs 672, attention layers 237, DeltaNet layers 446 |
| frame 0 (logits, sampling, head, decoder) | 6 ms | |
| the conversation's state (28,672 positions) allocated | 24 ms | once a call |
| total, `start` | 2,050 ms | the 60 s voice: 2,860 (encode 1,075, prefill 1,780) |

**What changed** (each measured alone; time to frame 0, 39 s / 60 s voice):

| path | 39 s voice | 60 s voice |
|---|---|---|
| old: the encoder a frame at a time, positions 16 a pass | 2,050 ms | 2,860 ms |
| the reference 8 frames a pass through the encoder (a wide stream of its own) | 1,548 (encode 697 -> 195) | 2,086 (1,075 -> 301) |
| the positions 128 a pass on the matrix engines (vv.h's tiled product, input split in two bfloat16 halves) | 1,261 (prefill 1,347 -> 557) | 1,846 (1,779 -> 763) |
| both | 760 | 1,070 |
| + the silo's gated delta rule (state in registers; DeltaNet layers 240 -> 113 ms) | 627 | 898 |
| + the silo's attention (a sub-group a row; attention layers ~154 -> ~40 ms) | **515** (encode 194, prefill 313) | **718** (301, 409) |
| a repeat call: the voices' cache (state restored: 5-6 ms, frame 0: 6 ms) | **13** | **13** |

A larger attention piece for the wide passes without the silo (256 keys and more) changed nothing (755 vs 760 ms).
16 frames a pass through the encoder would cut ~60 ms more for another 0.12 GiB on the card; not taken.

**The prompt ahead**: the conversation now runs the prompt as soon as a session names its voice (session.update), not
at the first audio, so a client that waits after its update finds it done; and a session that names no seed draws the
reference's latent noise from a fixed seed (the frames still from the session's own), so its prompt is the same call
after call - the cache's key. A seeded session keeps one stream from its seed, as before.

**Over the socket** (time from session.updated to the first audio; the client sends audio at once):

| | before | after, first call (cold) | after, repeat call (cache) |
|---|---|---|---|
| 39 s voice | 2.09 s | 0.58 s | 0.06 s |
| 60 s voice | 2.90 s | 0.77 s | 0.06 s |

The frames that queued meanwhile: the first call's worst turnaround 2.11 s -> 0.59 s (39 s voice), the repeat call's
74 ms, its median 30.8 ms either way. A frame alone did not change: 23.5 ms (median 23.5) on the B70 by the speed
stage, 23.4 before.

**Exactness** (`--stages start`): the new path against the old over 41 frames (frame 0 and 40 user frames: user.wav,
then silence), seed 41, the new session teacher-forced (each frame given the old one's token and latent), with and
without speak-first, both voices: the logits' argmax 41 of 41 every time, silent (PAD) vs speaking 41 of 41, the
sampled tokens 41 of 41, audible vs quiet audio 41 of 41; the audio's relative difference median 3e-5 - 7e-5, worst
5e-4 - 1.5e-3. What they said: "Yeah" / "Wow" (39 s voice), "I'm doing well" / "I'm Haru twenty two a senior in
college I go running every morning" (60 s), the same words on both paths. A cache hit against the cold start: the
same tokens and argmax 41 of 41 and the same audio, bit for bit (max difference 0), all four runs.

**Memory**: on the card +0.18 GiB (9.88 -> 10.06 GiB loaded: the encoder's wide stream 0.12, the wide passes' work
buffers); with a conversation of 28,672 positions 11.13 GiB on the B70, under the card's 11.5. The cache is host
memory: 71 MB a prompt with the 39 s voice, 76 MB with the 60 s one (the DeltaNet states ~50 MB whatever the voice,
keys and values 32 KB a position), bounded by `NS_HARU_VOICE_CACHE` (MiB, default 1024: about a dozen voice and card
pairs; the least recently used leaves first; 0: none), plus the encoded references of the last 64 voices (256 bytes a
frame) so a new card on a known voice skips the encoder. It goes with the model when it unloads.

## Memory

Measured on the B65 (the card's free memory before and after): 9.9 GiB in use once loaded (the language model 7.8,
the head 0.6, the codec's encoder and decoder 0.6 each, the work buffers), a conversation of the default 28,672
positions (an hour) adds 1.0 GiB (its half key / value cache 0.9, the DeltaNet states 50 MB, its codec streams): 10.9
GiB of the 31.9 with one conversation going - either card, with room for another model beside it. Each conversation
holds its own state (two at once: 12 GiB); their frames take turns on the card. The card says `vram_gib: 11.5`.

## Conversations

Held over the socket with `reference/haru/client.py` (the audio server on the B65, `audio serve --dir ... --here`):
recordings streamed at real time a frame at a time with quiet room noise between them, both sides recorded and the
model's side written down by the box's Whisper (`nextsycl audio transcribe`, whisper-large-v3-turbo). The voice: a
9 s saved voice; the card: the default unless named. What came back (Whisper's words, the times on the call):

- **A chat** (three questions with 6-7 s pauses): it introduced itself at once, as the default card's situation
  says ("Okay, my name is Haru. I'm 22. I go running every morning to clear my head ... I'm very curious about a lot
  of things."); to "why is the sky blue" - "What's blue? Why blue?"; to "is it going to rain in Melbourne tomorrow" -
  "Oh, that's a really good question." Answers come 2-4 s after the question ends.
- **Questions back to back** (11 s of three clips, 0.4 s apart): it introduced itself over the start of the first
  ("Hello, I'm Haru, age 22, senior in college. I go for runs most mornings."), kept quiet through most of the
  second, then answered it before the third ended: "It's going to rain, but it's not going to be a heavy rain. It's
  more likely to be an afternoon drizzle." The text runs one frame ahead of its audio, word by word.
- **Cut in on** (it opens - speak-first -, the user talks over it): it carried on through a 1.8 s question
  ("I'm Haru. I'm 22. I'm a senior in college, and I'm really interested in a lot of things, technology, startups,
  business, economics."); over a 16 s reading it stopped in mid-thought several times (pauses of 0.4 - 0.9 s while
  the user spoke), answered with "Yeah. Yeah." / "Yes, yes, exactly." and went on - it yields the floor only
  partly. Whether it stops is its own decision; the socket cuts nothing.
- **Listening to a reading** (16 s of invented names, card back-channels "high"): "Okay" at the start, then silent
  throughout - no back-channels (Whisper, and the frames' levels).
- **Without a voice** (the card alone): silent for the whole call - the release's server refuses a session without a
  reference clip, and so does this engine now (an `error` asking for a saved voice).

A second set with natural user speech (the lines spoken by Qwen3-TTS, voice ryan):

- **A 28 s story** ("last weekend I finally went on that hiking trip ... my sister twisted her ankle ... still a
  pretty good day"): it said "Hi." at the start, stayed quiet through the whole story, and answered when it ended:
  "Oh, that sounds like a really tough thing to happen. I am so sorry to hear that."
- **Cut in on, properly** (it opens with "What do you think you could use it for?"; the user: "Wait, wait, sorry to
  cut you off, can I ask you something?"): it stopped and handed over - "Well, sure, but you go ahead." - then, when
  the user said nothing more, "I'm sorry, I didn't mean to..."
- **A card of its own** (Kai, a retired sea captain running a bait shop; the user Max): "Hi there! Who am I talking
  to?" - "Hello, Max. It's me, Kai."; "Are the fish biting this week?" - "Ah, I see. Well, not much is biting. The
  fish are not very big. You know..." (Whisper heard "wish" for "fish" once).
- **No voice**: the session did not start; the client got the `error` asking for a saved voice.

Per-frame turnaround over the socket: 33 ms median from a frame sent to the model's answer to it (the B65), so on a
call the model's audio trails the user's by one frame (133 ms) plus 33 ms plus the network and the jitter buffer.
The prompt (1.4 s with a 9 s voice) delays the first frames: the client's first ~10 frames are answered together
once it is done.

## Back-channels (2026-11-17)

Does ours say "mm-hm" while you talk, as the release says the model does ("backchannels and overlapping speech",
blog.higurashi.ai/melte/haru4b)? Three 32-36 s monologues with their natural pauses (TED-LIUM long-form, validation:
Al Gore on the Arctic and water, Barry Schwartz on choice, Blaise Agüera y Arcas on Photosynth), each after 3 s of
quiet and followed by 8 s; Haru's frames over -40 dBFS while the monologue runs counted, each stretch of at most 1.6 s
written down by Whisper large-v3-turbo (the front door's), a back-channel when every word is one ("yeah", "mm",
"mm-hmm", "oh", "okay", "I see", "wow", "really" ...). Three cards: the default ("chat", back-channels "mid"), the same
with "high", and a listening one (female, "high", a friend on the phone telling Haru at length about something, Haru
"mostly listens, reacts with short responses"; the default persona kept). Voices: narrator (a 10 s designed male
voice) for the first two, keeper (6 s, a woman) for the third.

| | runs | sound over the monologue (share of its frames: mean / median) | silent throughout | back-channels | runs with one |
|---|---|---|---|---|---|
| ours over the socket (production :8000, `reference/haru/client.py` at real time; seeds 11/22/33, 44/55) | 24 | 0.33 / 0.32 | 3 | 6 | 4 |
| ours offline (the build copy, `/v1/audio/speech`, the same user audio; seeds 1-8) | 72 | 0.31 / 0.32 | 8 | 43 | 18 |
| upstream's own PyTorch session on the CPU (`DuplexSession`, float32, the same user audio, card and voice) | 24 | 0.28 / 0.18 | 5 | 6 | 5 |

What ours says back: "Yeah." (the most), "Mm.", "Mm-hmm.", "Oh.", "Okay.", "Wow.", "I see.", "Really?", "Sure.",
"Aha."; the release: "Yeah." (twice), "Yes.", "Okay.", "Oh!", "Ah". A lone "No.", "I" or "Um" is not counted.
Per card the numbers are too few to tell apart (the release: 0 / 3 / 3 for high / mid / listening; ours offline
11 / 14 / 18). Most come over the monologue's voiced frames or its short pauses, not only at sentence ends.
In the release's own demo video (investment-caption-en.mp4: user and Haru on separate Whisper on each) Haru says "Yeah." twice in 95 s, both in the user's 1-3 s pauses mid-turn. So: a back-channel
in about one 30 s monologue in four or five, ours and the release's alike.

Just as often it talks over the monologue, in the release as here: the default card's situation has the two "start
by introducing themselves", so it introduces itself whenever it likes ("I'm Haru, twenty-two, a senior in college, I
go running most mornings ..."), sometimes repeats the user's words ("the more freedom they have"), and now and then
says the voice recording's own words ("This is how I sound when I read aloud", the narrator clip's script; "...
wrote each name in a small blue book", the end of keeper's): 6 of our 24 socket runs - three of them the same seed
(55, keeper, from its first frames) -, 1 of the release's 24. When the monologue ends it answers within 1.3 s in 11
of the 12 socket runs where it was quiet at that moment (once 4.8 s); the release within 0.1-4.3 s.

Same model, same sampling: the frame-by-frame check above runs 300 frames of a monologue against the release's code
with P(silent) 3e-4 apart at worst, and the sampling is the release's line for line (DRY 0.8 / 1.75 / 1 / 1,024 on
words only, temperature 0.7, top-k 25, top-p 1, no EOP bias, no other logit bias; the release has no back-channel or
silence bias to miss; its PAD / EOP logits in float32 - ours are all float32). The socket and the offline path give
the same words for the same seed (their first sentences identical, then sampling's chaos). No port difference
found, nothing changed in the engine.

Tools (reference/haru/): `client.py` (a socket session: `--say clip.wav --lead 3 --gap 8 --seed S --card c.json`),
`upstream_session.py` (the release's session on the CPU on a socket run's user.wav: ~0.8 s a frame, 20 GB of host
memory - one at a time), `offline_answer.py` (ours on the same audio through an audio server), `backchannels.py`
(levels, stretches, the text by frame), `backchannel_table.py` (Whisper on the short stretches, the table above);
`page_test.mjs` (below). The runs: hardac1 `~/src/harubc-work` (runs/, off/, up/, summary-all.json, compare.json).

## The conversation page with a microphone (2026-11-17)

Tested headless: Chrome for Testing 153 with a fake microphone (`--use-fake-device-for-media-stream
--use-file-for-fake-audio-capture=<the TED clip>`), the page driven over DevTools, its WebSocket and every
`AudioBufferSourceNode.start` recorded (`reference/haru/page_test.mjs`: `node page_test.mjs URL WAV SECONDS OUT
[VOICE]`). Against production over https://127.0.0.1:8443/open/haru-4b/:
the microphone at 48 kHz with echo cancellation, noise suppression and AGC, `input_sample_rate: 48000`, 45 s sent in
562 appends, 338 frames back, every one played, no gap between them - but queued 450 ms ahead (median; 200 at the
first): the frames answered together when the prompt finished stayed queued, so everything Haru said came 0.25 s
late for the whole call. Fixed: past 0.35 s queued a quiet frame (under -40 dBFS) is dropped, past 1 s any (the
release's client drops its oldest audio too); the AudioContext is made in the click, before the permission prompt,
and resumed if a browser started it suspended. After the fix, against the build copy's server (B65, the page from
disk): 45 s - 338 frames, 3 quiet ones dropped, queued 205 ms median (p95 243), no gaps; 30 s - 225 frames, none
dropped, 211 ms. Haru heard the fake microphone and answered it ("wait, what are you, a libertarian?").

## Open

- **Back-channels**: occasional, as in the release (above); not yet heard by a person on a headset.
- **Speed**: a frame is not tuned (the silo has only the prompt's wide passes). It is bound by a few hundred small
  launches and host round trips, not bandwidth; the codec encoder's first stages (3,200 rows) are its largest part.
  Graphs or fused launches would cut the 23-25 ms further; nothing needs it for real time.
- **The GGUF** (sakasegawa/Haru-4B-GGUF) is not read: bfloat16 already keeps up; a Q8_0 path (vv.h's gemv reads Q8_0)
  would halve the weights' memory if a card needs the room.
- **One conversation per voice and card change**: a session.update that changes them starts over (the prompt again,
  0.5-0.7 s, or the cache's 13 ms), as the release does.
- **The start** was measured on the B70 only; the B65 not yet.
- **The wfe panel**: checked headless with a fake microphone (above); not yet by a person with a real one.
- Served on hardac1's front door (:8000, :8443; the page /open/haru-4b/).
