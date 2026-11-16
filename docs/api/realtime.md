# Live conversations: `/v1/realtime`

A WebSocket for talking with a full-duplex speech model: the microphone streams in, the model listens, decides when
you have finished, and answers with speech (and its text) as it makes it; you can cut in. It speaks OpenAI Realtime's
events, so an OpenAI Realtime client works by changing the URL and key. One API for every full-duplex model here:
Sophon-Realtime (docs/ports/sophon-realtime.md) and Haru-4B (docs/ports/haru-4b.md: lockstep frames, the extension below).

```text
ws://<box>:8000/v1/realtime?model=<id>          the front door: Authorization: Bearer <token>, or the browser's
                                                sub-protocol "openai-insecure-api-key.<token>"; no model: the loaded or
                                                first enabled model whose card's speech list says realtime
ws://<box>:8000/open/<id>/v1/realtime           the model's own page's (no token: as every /open/ page)
ws://127.0.0.1:<port>/v1/realtime               the model's audio server itself (nextsycl audio serve)
```

The front door loads the model as for any request (rsvp, room, a card), passes the upgrade on to its audio server and
copies both ways while the socket lasts; the server counts as busy meanwhile (its progress says `phase:
conversation`), so it is neither stopped for room nor idled out. Maintenance mode refuses new conversations.

## Audio

| format | in | out |
|---|---|---|
| `pcm16` (default) | 16-bit little-endian mono, 24 kHz unless `nextsycl.input_sample_rate` (or the GA shape's `rate`) says another | 24 kHz unless `nextsycl.output_sample_rate` |
| `g711_ulaw`, `g711_alaw` | 8 kHz | 8 kHz |
| `pcm_f32le` | float32 mono, at `sample_rate_hz` (Sophon's own server's) | float32 |

What comes in is resampled for the model (Sophon hears at 16 kHz, speaks at 24 kHz; Haru hears and speaks at 24 kHz). `input_audio_buffer.append`
carries base64; a binary WebSocket message is taken as raw audio in the session's input format (an extension).

## Client events

| event | what it does |
|---|---|
| `session.update {session}` | the settings, beta or GA shape: `instructions` (the system prompt; empty: the model's own), `voice` (a built-in one, or a saved voice's name: its recording cloned), `input_audio_format` / `output_audio_format` (GA: `audio.input.format`, `audio.output.format`, `audio.output.voice`), `turn_detection` (`{"type": "server_vad"}` - the model decides when you have finished; `silence_duration_ms`: the longest pause before it answers anyway; `null`: only your commit ends a turn), `tools` (functions it may call), `tool_choice: "none"`, `temperature`, `max_response_output_tokens` (a number, or `"inf"`: as much as the context holds; default 1,024 - Sophon's whole spoken answer), `modalities` (`["text"]`: no audio out). Extensions: `extra_body` as Sophon's server takes it (`language`, `turn_fallback_s`, `history`, `sample_rate_hz`) and `nextsycl` (`language`, `input_sample_rate`, `output_sample_rate`, `voice`, `history`, `seed`, `options`: the engine's own) |
| `input_audio_buffer.append {audio}` | your audio (base64) |
| `input_audio_buffer.commit` | your turn is over now (with or without turn detection) |
| `input_audio_buffer.clear` | your audio not yet taken is dropped |
| `response.create` | answer now (a text turn, a function's result) |
| `response.cancel` | stop the answer being made |
| `conversation.item.create {item}` | `{"type": "message", "role": "user" or "assistant", "content": [{"type": "input_text", "text": ...}]}` - a turn as text; `{"type": "function_call_output", "call_id", "output"}` - a function's result (it answers after `response.create`, or by itself a second later) |
| `conversation.item.truncate {item_id, audio_end_ms}` | you heard only this much of an answer (the client stopped playing it): its history keeps that share |
| `session.close` | ends it (Sophon's own; closing the socket does the same) |

## Server events

| event | when |
|---|---|
| `session.created`, `session.updated` | at the start, after each update; `session.nextsycl` tells the model's own: `input_rate`, `output_rate`, `frame_ms` and `lockstep`, `voices`, `clone`, `languages`, `tools` |
| `input_audio_buffer.speech_started {audio_start_ms, item_id}` | you started speaking (an answer being made or played stops: barge-in) |
| `input_audio_buffer.speech_stopped {audio_end_ms}` | the model judged your turn finished (`audio_end_ms`: where your pause began) |
| `input_audio_buffer.committed`, `conversation.item.created` | your turn taken (Sophon hears it directly: `transcript` is null) |
| `response.created`, `response.output_item.added`, `conversation.item.created`, `response.content_part.added` | an answer begins (its item `msg_<response id>`) |
| `response.audio_transcript.delta {delta}` | its text as written, sound tags kept (`[laughter]`, `<strong>...</strong>`) |
| `response.audio.delta {delta}` | its sound as spoken (base64, the session's output format) |
| `response.output_item.added` / `response.function_call_arguments.delta` / `.done {call_id, name, arguments}` / `response.output_item.done` | a function call it makes (an item of type function_call) |
| `response.audio.done`, `response.audio_transcript.done {transcript}`, `response.content_part.done`, `response.output_item.done`, `response.done {response: {id, status, output, nextsycl: {audio_ms}}}` | the answer ended: `completed`, `cancelled` (cut in, cancelled) or `failed` |
| `error {error: {type, code, message}}` | a bad event or a failure (the session goes on) |

A client using the GA shape (`session.type: "realtime"`, or `?flavor=ga` in the URL) gets GA's names:
`response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.output_text.delta`,
`conversation.item.added`.

## Lockstep models (Haru-4B)

OpenAI's events assume a model that answers a turn; Haru-4B is full duplex in lockstep: every 133 ms (7.5 Hz) one
3,200-sample 24 kHz frame of yours in, one of its own out (silence included), its text alongside. It hears and
speaks at once: when to answer, back-channel ("mm-hm" while you talk) or stop because you cut in is the model's own
doing, not a turn rule's. On this socket:

| | |
|---|---|
| `session.created` | `nextsycl.frame_ms: 133, lockstep: true` (Sophon: 0, false), `voices: []` (its voice is a recording: `clone: true`), `languages` (the card's), `tools: false` |
| the card | `session.update {session: {nextsycl: {card: {...}}}}` - `persona`, `name`, `gender` (`female` / `male`), `backchannel` (`low` / `mid` / `high`), `other_name`, `relationship`, `situation`; fields left out keep the model's own card (the release's "chat" character), `null` goes back to it; a string is a whole card's text. `instructions` alone becomes the persona. `nextsycl.language` picks the card's wording (`en` `ja` `zh` `ko` `es` `fr` `de` `pt` `ru`); `session.nextsycl.card` echoes it |
| the voice | `voice`: a saved voice by name - its recording (3 s - 3 min, trimmed of silence, the last 3 min) is the voice reference. Needed: without one Haru stays silent (the release's server refuses such a session), so nothing starts and an `error` asks for one |
| options | `nextsycl.options`: `guidance` (1-2), `eop-bias`, `speak-first` (1: it opens), `top-k`, `dry`; `temperature`; `nextsycl.seed` (0: a new one) |
| your audio | `input_audio_buffer.append` (or a binary message) of any size as it is recorded; the server cuts it into frames (resampled to 24 kHz first). The prompt (card, voice) runs when the first audio comes (or a `response.create`); a later `session.update` that changes the card, the voice, the language or the sampling starts the conversation over |
| its audio | one long-running response (`response.created` once): each frame of yours is answered by exactly one `response.audio.delta` of one frame (133 ms) in the session's output format, in order, and its new text (`response.audio_transcript.delta`: word pieces as said; a run of silent frames is one space). It is one frame ahead: the first delta (frame 0) comes from the prompt, frame k + 1 answers your frame k. Sent as fast as made: a client that sends faster than real time gets the answers faster; one that stops sending stops it with it |
| turn events | none (`input_audio_buffer.committed` / `response.done` do not come while it runs); `input_audio_buffer.speech_started` / `speech_stopped` as an energy gate hears you (for a display: a frame above max(0.01, 3 x the noise floor) starts, 4 quiet frames end) - a client does not stop its playback on them, the model stops itself |
| `response.cancel` | ends the response (`response.done` cancelled) and silences the model: its text held at PAD and its own audio stream at silence while it keeps listening; nothing goes out until `response.create`, which starts a new response |
| `input_audio_buffer.commit` / `clear` | commit: nothing (no turns); clear: your audio not yet framed is dropped |
| not taken | `conversation.item.create` (a turn as text: it hears audio only), function calls (`tools`: it calls none) |
| the end | the context (`NS_HARU_CTX`: about an hour) full: an `error`, `response.done` failed; a new session starts over |

A client plays the deltas back to back behind a short jitter buffer (the wfe's Conversation panel keeps 0.2 s) and
records with echo cancellation (or headphones): the model hears whatever the microphone picks up, its own voice
included. `reference/haru/client.py` streams a recording at real time and records both sides.

## Example

```sh
python3 -I reference/sophon/client.py --url ws://localhost:8000/v1/realtime --token $TOKEN --wav question.wav --out DIR
```

(the client streams a recording at real time, then silence, and writes what the model said - DIR/answer-1.wav -, its
text, the events and the timings; `--then` a second turn, `--barge-at S` that turn spoken S seconds into the answer,
`--tools` a get_weather function offered and answered).

A lockstep conversation with Haru-4B (a saved voice, a card of its own; `--say` / `--gap` the user's script):

```sh
python3 -I reference/haru/client.py --url "ws://localhost:8000/v1/realtime?model=haru-4b" --token $TOKEN --voice MYVOICE \
    --card card.json --say hello.wav --gap 5 --say question.wav --gap 8 --out DIR
```

(it writes DIR/model.wav, DIR/user.wav, DIR/call.wav - you left, it right -, its text with times and the frames'
turnaround; `--barge-at S` speaks the last line S seconds into its talk, `--speak-first` lets it open).
