# Prompting Fish Audio S2 Pro (speech in ~80 languages, 44.1 kHz)

No built-in voices and no language setting: `input` is the text to say, in any of about 80 languages (best: English,
Chinese, Japanese; then Korean, Spanish, Portuguese, Arabic, Russian, French, German) - the model reads the script.
What it does:

| mode | what you send | what you get |
|---|---|---|
| **voice clone** | `ref_audio` + `ref_text` (or a saved `voice` with its transcript) | that recording's voice |
| **direction** | a clone (or plain) + `instructions`, or tags in the text | the voice steered by the tag: tone, emotion, pace, volume |
| plain | `input` alone | a voice the model picks for the text (it changes with the `seed`) |
| several speakers | `<\|speaker:0\|>`, `<\|speaker:1\|>` ... in the text | a dialogue in one clip (with a clone: speaker 0 is the recording's voice) |

**What it does not do**: design a voice from a description (there is no description input - a tag like `[deep male
voice]` steers the delivery, not a reliable timbre), and there are no named voices. Use a voice-design model
(qwen3-tts-design, voxcpm2, breeze-tts-2) to make a voice, save it, and clone it here.

## Inline tags (S2 Pro's control)
Free-form words in square brackets, anywhere in the text, apply from where they are: S2 Pro was trained on
15,000+ different tags, so plain descriptions work, not only a fixed list.

- Emotion and tone: `[excited]`, `[excited tone]`, `[sad]`, `[angry]`, `[surprised]`, `[shocked]`, `[delight]`,
  `[laughing tone]`, `[professional broadcast tone]`
- Voice and volume: `[whisper]`, `[low voice]`, `[shouting]`, `[screaming]`, `[loud]`, `[volume up]`, `[volume down]`,
  `[low volume]`, `[whisper in small voice]`, `[pitch up]`, `[with strong accent]`, `[echo]`, `[singing]`
- Sounds and pauses: `[laughing]`, `[chuckle]`, `[chuckling]`, `[sigh]`, `[inhale]`, `[exhale]`, `[panting]`,
  `[clearing throat]`, `[tsk]`, `[moaning]`, `[audience laughter]`, `[pause]`, `[short pause]`, `[emphasis]`,
  `[interrupting]`

Example: `I can't believe it. [laughing] You actually did it! [whisper] Don't tell anyone.`

`instructions` (the API's field) is put in as one such tag at the start of the text (and of each speaker's turn):
`instructions: "sad, slow"` reads like `[sad, slow] ...`. Keep it short - a few words, as the tags above.

## Voice clone
A clean recording of one speaker (10-30 s is best, at most 30 s; no music or noise) **and its exact transcript** in
`ref_text` - S2 Pro conditions on both, and without the transcript it does not clone. Saved voices
(`audio.voices.add` with `text`) carry their transcript, so `voice: NAME` is a clone.

## Several speakers
Mark each turn: `<|speaker:0|>Did you hear that? <|speaker:1|>[whisper] Hear what?`. Turns are read in batches
(five speakers or 200 bytes at a time), each continuing the conversation before it, so voices stay consistent. With a
recording, speaker 0 speaks in its voice; the other speakers are the model's.

## Settings
`seed`, `seconds` (the most, up to 120 - the model ends the speech itself). `options`: `temperature` (0.7), `top-p`
(0.7), `top-k` (30), `greedy` (1: always the most likely code - deterministic but flat, and it may not end). No `cfg`.

## License
The weights (and this GGUF conversion of them) and **everything they make - the audio - are for research and
non-commercial use only**, under the Fish Audio Research License (https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md;
commercial use needs a written license from Fish Audio: business@fish.audio). Attribution: "This model is licensed under
the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved." Do not clone a voice without its owner's
consent.
