# Prompting Breeze TTS 2 (speech in English and Chinese, 24 kHz)

No built-in voices: `input` is the text to say, in English or Chinese (no language setting: the model reads the
script). Four ways to use it:

| mode | what you send | what you get |
|---|---|---|
| **voice design** | `instructions`: a description of the voice, no recording | a voice made up from the description |
| **voice clone** | `ref_audio` + `ref_text` (or a saved `voice`) | that recording's voice: timbre, rhythm, emotion and style |
| **voice direction** | `ref_audio` + `ref_text` (or a saved `voice`) + `instructions` | the recording's voice, steered by the instruction |
| plain | `input` alone | a voice the model picks for the text |

Vocal events go inline in the text, in any mode (see below).

## Voice design
Describe the speaker and the delivery in a sentence: who (age, gender), the voice (warm, clear, raspy, deep), the
delivery (calm, brisk, playful, reflective). **Write the description in the language of the text**: an English
description for English text, a Chinese one for Chinese text.

- `A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.`
- `一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。`

`cfg` 4 (the default with an instruction) holds the voice to the description; lower is looser, 1 turns guidance off.
A design varies with the `seed`: try 2-3 and keep the one you like (save it with `audio.voices.design`).

## Voice clone
Give a clean recording of one speaker (3-15 s, no music or noise, no looping) **and its exact transcript** in
`ref_text` - every word said, repetitions included; a transcript that does not match makes a worse clone. The clone
takes no instruction: adding `instructions` makes it voice direction. Saved voices (`audio.voices.add`) carry their
transcript, so `voice: NAME` is a clone.

## Voice direction
A clone (recording + transcript, or a saved voice) plus `instructions` that steer tone, emotion, pace and delivery
while keeping the speaker: `Speak slowly with a restrained, serious tone.` / `用激动的语气，语速稍快。` Guided by `cfg`
(4 by default) against the plain clone.

## Vocal events
Put them in the text where they happen:

- English, in parentheses: `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)` - `(sigh) It is good to hear your voice again.`
- Chinese, in square brackets: `[笑]`, `[叹气]`, `[咳嗽]`, `[清嗓子]` - `[叹气] 没想到过了这么久，你还记得我的声音。`

## Settings
`cfg` (4 with an instruction; unused without one), `seed`, `seconds` (the most; up to 120 - the model ends the speech
itself). `options`: `temperature` (0.9), `top-k` (50), `greedy` (1: always the most likely code - deterministic but
flat, and it may not end on its own).

## License
The weights (and this GGUF conversion of them) and **everything they make - the audio - are for research and
non-commercial use only**, under the BreezeBlue Research and Non-Commercial License 1.1
(https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). The upstream inference code is Apache-2.0; the
codec is Qwen's (Apache-2.0). Do not clone a voice without its owner's consent.
