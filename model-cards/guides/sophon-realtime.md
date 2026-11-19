# Talking with Sophon-Realtime

A live voice conversation, not a request and an answer: open `/v1/realtime` (a WebSocket of OpenAI Realtime's
events, docs/api/realtime.md) or the model's page (Conversation), stream your microphone, and talk.

## How it takes turns
- It hears your speech itself (no transcription): tone, hesitation and laughter reach it. When you pause it asks
  itself whether you have finished; a short pause after a clear question answers in about a second, a hesitation
  ("um, so...") waits. Two seconds of silence answer anyway (`turn_detection.silence_duration_ms` sets that).
- Speak over it to stop it: what you heard of its answer is what it remembers having said.
- Starting again within two seconds of its answer means it answered too early: that answer is dropped and your two
  stretches of speech are taken as one turn.

## The session
- `instructions`: a system prompt of your own (a persona, a task). Without one it is Sophon, answering in one or two
  natural spoken sentences - and in full when asked for something long (a story, an explanation, a list: up to ~11
  minutes spoken, 2,048 tokens unless max_response_output_tokens says otherwise). Upstream's prompt held every answer
  to a sentence or two (asked for a story, it declined); a prompt of your own should say long requests are welcome
  if they are. It was tuned with its own; keep yours short and spoken in style, and say it may use sound
  tags, or it uses fewer.
- `voice`: `chelsie` (default) or `brand`, or a saved voice by its name (its recording cloned: 3-15 s of one speaker,
  clean). Only clone your own voice or one you have permission to use.
- `nextsycl.language`: `en` (default), `zh`, `de`, `fr`, `es`, `it`, `pt`, `ja`, `ru` - the language it speaks;
  Chinese and English are its best.
- `tools`: functions it may call (OpenAI's shape). It says a short phrase ("let me check"), calls, and speaks the
  result when your `function_call_output` comes back.

## What it writes
Its text keeps the sound tags it performs: `[laughter]`, `[breath]`, `[sigh]`, `[cough]`, `[quick_breath]`,
`<laughter>...</laughter>` (laughing while speaking) and `<strong>...</strong>` (emphasis). Strip them for display.

## Speaking a text
`audio.speak` / `/v1/audio/speech` with this model speaks `input` in its voice (the talker alone, no brain). The
sound tags work here too, as in a conversation: `[laughter]`, `[sigh]` and `[breath]` are performed (a laugh, a sigh,
a breath where the tag stands; the word is not said), `<laughter>...</laughter>` is not read; `[cough]` is weak in
English, `[quick_breath]` is sometimes said as words, and `<strong>` hardly stresses anything (upstream says so of
this build). Heard 2026-11-16, voice chelsie, three seeds each. That is the `-tags` entry; the plain entry's voice says
nothing for a tag (below).

## Which entry: the voice or the tags
Two entries run the same brain and differ only in the voice that speaks:
- `sophon-realtime-32b-flash`: the voice is Qwen3-TTS-12Hz-1.7B-Base. Choose it for a character with a saved, cloned
  voice. It clones in context: the saved voice's recording and what it says go into the voice's prompt (the front
  door has Whisper hear each saved voice once for that; a voice not heard yet, or a recording sent without
  `ref_text`, is cloned from its timbre alone). That sounds closest to the recording: 0.70-0.87 speaker similarity
  to it, against 0.55-0.84 by timbre and 0.47-0.74 with the tags entry's talker (four saved voices, measured
  2026-11-17). The price is a later first word in a conversation: the voice starts once the reply is written (about
  2.2 s after the reply starts for a 30-word answer, against 0.8 s; little on a short one). Set
  `nextsycl.options: {"clone": "xvector"}` in the session for the quick start. It cannot perform the sound tags. The
  brain still writes them in its text, but they are taken out of what the voice reads, so they are silent.
- `sophon-realtime-32b-flash-tags`: the voice is the release's own talker, a fine-tune of Base. Choose it for a
  built-in voice (`chelsie`, `brand`) or when the laughs and sighs matter more than the likeness. It performs the
  tags, as listed above. A cloned voice keeps its words and its manner, but it drifts towards a generic voice: the
  saved voice "Skippy1" measured 0.52 to its recording, against 0.61 with Base.
- A good recording for either is 5-15 s of one speaker, clean, in a steady voice (upstream asks for 3-10 s).
