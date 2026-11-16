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
  not changed. No silo: nothing was tuned (it keeps up with real time on the shared products).
- `glue/serve/src/realtime.rs` - `nextsycl.card`; the lockstep extension needs nothing else of the server (the engine
  makes the events); a lockstep mock test. `wfe/audio/src/converse.ts`, `components/Converse.tsx` - lockstep play-out
  behind a 0.2 s jitter buffer, no barge-in cut, the transcript split where you spoke, the character card's fields.
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
comes back 33 ms (median) after the frame was sent. The prompt with a 9 s voice (336 positions, 68 frames encoded)
takes 1.4 s; the frames that came meanwhile are then answered at once (the client's first 10 frames wait up to 1.4 s).

Where a frame goes (`NS_HARU_PROFILE=1`, the B65, each operation waited for - so slower than the 25 ms): the codec's
encoder is the largest share (its first stages run 3,200 rows of 32-128 channels through the tiled product), the
language model's MLPs next, then attention, the DeltaNet layers, the logits (248,320 x 2,560 bfloat16) and the head.
Both cards land within 7% of each other: a frame is a few hundred small launches and host round trips (token ids,
latents, logits and the head's 64 values cross each frame), not the 10 GiB of weights it reads. Not tuned: no silo.

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

## Open

- **Back-channels**: none heard in these tests (it listens silently, then answers); interruptions it handles when
  addressed ("sorry to cut you off" - "go ahead"), not when talked over by a reading. Worth a session with a person
  on a headset in the wfe.
- **Speed**: not tuned (no silo). The frame is bound by a few hundred small launches and host round trips, not
  bandwidth; the codec encoder's first stages (3,200 rows) are its largest part. Graphs or fused launches would cut
  the 23-25 ms further; nothing needs it for real time.
- **The GGUF** (sakasegawa/Haru-4B-GGUF) is not read: bfloat16 already keeps up; a Q8_0 path (vv.h's gemv reads Q8_0)
  would halve the weights' memory if a card needs the room.
- **One conversation per voice and card change**: a session.update that changes them starts over (the prompt again,
  1-2 s), as the release does.
- **The wfe panel** was checked by its smoke test against the Haru server (the page and its 27 modules load, the
  app renders) and the protocol by the lockstep mock test and the client - not yet by a person with a microphone.
- Not deployed (no card registered on the front door).
