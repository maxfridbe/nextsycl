# Prompting VibeVoice (long-form, multi-speaker speech in English and Chinese, 24 kHz)

VibeVoice-1.5B reads a **script** and speaks it - one voice, or a conversation of up to four speakers (a podcast, an
interview, a drama), for seconds or tens of minutes in one go. No built-in voices and no instructions: each speaker's
voice comes from a recording (cloned without a transcript), or, without one, the model picks a voice.

| mode | what you send | what you get |
|---|---|---|
| **plain** | `input` alone (a text, or a script) | voices the model picks (they vary with the `seed`) |
| **clone** | `input` + `ref_audio` (or a saved `voice`) | the recording's voice for speaker 1 |
| **multi-speaker** | a script + `voices` (saved voices) or `ref_audios` (recordings), one a speaker in order | each speaker in their own voice |

## The script
One turn a line, `Speaker N:` in front (numbered from 1, up to 4):

```
Speaker 1: Welcome back to the show. Today we talk about gardens.
Speaker 2: Thanks for having me. I love tomatoes, so this is a good day.
Speaker 1: Then let's start with tomatoes.
```

A line without a speaker continues the turn before it; a text without any `Speaker N:` is speaker 1's. Speaker N's
voice is the Nth of `voices` / `ref_audios` (or `ref_audio` / `voice` for speaker 1); a speaker without one gets a
voice of the model's own. Write naturally punctuated text: the model takes its pauses and intonation from it.
English and Chinese are what it was trained on; other languages are unreliable.

## Voices
A clean recording of one speaker, 5-30 s, no music or noise; a transcript is not needed (`ref_text` is ignored). The
model copies the voice's timbre and its recording conditions, so a studio-quality sample gives studio-quality speech.
Saved voices (`audio.voices.add`) work with or without their transcript: `voice: NAME` for one speaker,
`voices: ["anna", "ben"]` for a script. The CLI takes `--ref-audio` once a speaker, or `--voices anna,ben`.

## Settings
`cfg` (1.3: the diffusion head's guidance - higher holds the voice tighter, too high sounds harsh), `steps` (10: the
diffusion steps a frame), `seed` (the voices made up and the details of delivery), `seconds` (the most; the model ends
the speech itself - a script of a few minutes takes a few minutes of frames). The speed is about 7.5 frames (one
second) of speech for each frame the card makes - see the README for the measured real-time factor.

## License and responsible use
MIT (Microsoft's weights; gguf-org's conversion; this engine). Microsoft's model card restricts its use: do not use
it to imitate a real person's voice without their explicit consent (impersonation, fraud, disinformation), do not
present the speech it makes as a real recording, and disclose that the audio is AI-generated where it is shared. It
is meant for research and development; it may produce unexpected or biased output, and it does not make music,
background sounds or overlapping speech.
