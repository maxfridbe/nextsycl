# Talking with Haru-4B

A live voice conversation in lockstep, not a request and an answer: open `/v1/realtime` (a WebSocket of OpenAI
Realtime's events, docs/api/realtime.md: "Lockstep models") or the model's page (Conversation), stream your
microphone, and talk. Wired headphones help: it hears whatever the microphone picks up, its own voice included.

## How it talks
- Every 133 ms it takes one frame of your audio and gives one of its own, silence included: it listens while it
  speaks. It decides itself when to answer, when to say "mm-hm" while you talk, and when to stop because you cut in -
  there are no turns and no turn detection.
- What to expect (measured against the release's own code: the same): back-channels come now and then, not on every
  pause - a "yeah", "mm", "mm-hmm", "oh" or "I see" in about one 30 s monologue of yours in four or five. Just as often it
  talks over a long turn: the default card has the two of you introduce yourselves, so it may tell you who it is
  while you talk, echo your words, or now and then say the words of its voice recording. It answers within about a
  second of you stopping. The card's `backchannel` setting made no clear difference in tests.
- Its text comes with its audio (`response.audio_transcript.delta`), word pieces as it says them; a run of silent
  frames shows as one space.
- It is a pretrained base model: it chats naturally in character but is not an assistant - no facts to rely on, no
  instructions followed beyond the card.

## The session
- `nextsycl.card`: the character - an object of `persona` (who it is, how it talks), `name`, `gender` (`female` /
  `male`), `backchannel` (`low` / `mid` / `high`: how often it says "mm-hm"), `other_name` (yours),
  `relationship`, `situation`; fields left out keep the default card's (the release's "Haru", a college senior
  meeting you for the first time). `instructions` alone is taken as the persona.
- `voice`: a saved voice by name - its recording is the voice it speaks with (3 s to 3 minutes of one speaker,
  clean). It is needed: without a voice the model stays silent (the release's own server refuses such a session), so
  the conversation does not start and an `error` says so. Only clone your own voice or one you have permission to use.
  The voice and card are read as soon as the session names them (the start then needs only the first frame), and kept:
  the next call with the same voice and card starts at once.
- `nextsycl.language`: the card's language - `en` (default), `ja`, `zh`, `ko`, `es`, `fr`, `de`, `pt`, `ru`. It
  answers in the language you speak.
- `nextsycl.options`: `guidance` (1-2, default 1.5: higher keeps the voice closer to the recording), `eop-bias`
  (above 0: it starts speaking sooner), `speak-first` (1: it opens the conversation), `top-k`, `dry`.
- `temperature`: its words' sampling (default 0.7).

## Offline
`audio.generate` with this model answers a recording (`source`: your side of a conversation) with its side, as long
as yours (`options: {"stereo": "1"}`: you left, it right); without a recording it talks alone for `seconds`. The
voice is the request's reference recording.
