# Live conversations: `/v1/realtime`

A WebSocket for talking with a full-duplex speech model: the microphone streams in, the model listens, decides when
you have finished, and answers with speech (and its text) as it makes it; you can cut in. It speaks OpenAI Realtime's
events, so an OpenAI Realtime client works by changing the URL and key. One API for every full-duplex model here:
Sophon-Realtime now (docs/ports/sophon-realtime.md), Haru-4B next (its lockstep frames are the extension below).

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

What comes in is resampled for the model (Sophon hears at 16 kHz, speaks at 24 kHz). `input_audio_buffer.append`
carries base64; a binary WebSocket message is taken as raw audio in the session's input format (an extension).

## Client events

| event | what it does |
|---|---|
| `session.update {session}` | the settings, beta or GA shape: `instructions` (the system prompt; empty: the model's own), `voice` (a built-in one, or a saved voice's name: its recording cloned), `input_audio_format` / `output_audio_format` (GA: `audio.input.format`, `audio.output.format`, `audio.output.voice`), `turn_detection` (`{"type": "server_vad"}` - the model decides when you have finished; `silence_duration_ms`: the longest pause before it answers anyway; `null`: only your commit ends a turn), `tools` (functions it may call), `tool_choice: "none"`, `temperature`, `max_response_output_tokens`, `modalities` (`["text"]`: no audio out). Extensions: `extra_body` as Sophon's server takes it (`language`, `turn_fallback_s`, `history`, `sample_rate_hz`) and `nextsycl` (`language`, `input_sample_rate`, `output_sample_rate`, `voice`, `history`, `seed`, `options`: the engine's own) |
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
3,200-sample 24 kHz frame of yours in, one of its own out (silence included), its text alongside. Such a model's
`session.created` says `nextsycl.frame_ms: 133, lockstep: true`; the plan for it on this socket: frames are sent as
they are recorded (`input_audio_buffer.append`, any size - the server cuts them), each input frame answered by one
`response.audio.delta` of the same length within one long-running response, `response.audio_transcript.delta` as its
text comes, no turn events (speech_started / stopped only as it hears them); `response.cancel` silences it, a session
holds the character card it runs on (`nextsycl.card`) and its voice (a saved voice's recording). Not built yet
(TODO.md: Haru-4B).

## Example

```sh
python3 -I reference/sophon/client.py --url ws://localhost:8000/v1/realtime --token $TOKEN --wav question.wav --out DIR
```

(the client streams a recording at real time, then silence, and writes what the model said - DIR/answer-1.wav -, its
text, the events and the timings; `--then` a second turn, `--barge-at S` that turn spoken S seconds into the answer,
`--tools` a get_weather function offered and answered).
