# Prompting OmniVoice (speech in 646 languages, 24 kHz)

`input` is the text to say. OmniVoice has no named voices; it speaks in one of three ways:

| mode | what you send | what you get |
|---|---|---|
| **voice clone** | `ref_audio` + `ref_text` (or a saved `voice` with its transcript) | that recording's voice |
| **voice design** | `instructions`: attributes, comma-separated | a new voice with those attributes |
| clone + attributes | both | the recording's voice; attributes that agree with it make it steadier (a dialect clone: a dialect recording and the dialect) |
| auto | `input` alone | a voice the model picks (it changes with the `seed`) |

## Voice design: the attributes
`instructions` takes only these, at most one of each kind, separated by commas (`,` or `，`), any case:

- gender: `male`, `female`
- age: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- pitch: `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`
- style: `whisper`
- English accent (English text only): `american accent`, `british accent`, `australian accent`, `canadian accent`,
  `indian accent`, `chinese accent`, `korean accent`, `japanese accent`, `portuguese accent`, `russian accent`
- Chinese dialect (Chinese text only): 河南话, 陕西话, 四川话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话

The Chinese names work too (男, 女, 青年, 低音调, 耳语 ...). Example: `female, young adult, high pitch, british accent`.
Anything else - "happy", "speak slowly", "a pirate" - is refused with the list: the model was trained on these
attributes only. Leave out what you do not care about; some combinations are ignored (the training data had few).

## Non-verbal tags
In the text, where they happen: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`,
`[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`,
`[dissatisfaction-hnn]`. Example: `That is the funniest thing I have heard all week [laughter] tell me again.`
Pronunciation can be forced in brackets: English by CMU phonemes (`[B EY1 S]`), Chinese by pinyin with tone digits
(`ZHE4`).

## Language
`language`: a name (`english`, `cantonese`, `swahili` ...) or its id (`en`, `yue`, `sw`) of the 646 it was trained on;
`auto` or none lets it read the script (slightly worse than naming it). Min Nan (Hokkien) only in Tai-lo romanization.

## Voice clone
A clean recording of one speaker, 3-10 s is best (at most 30 s), **and its transcript** in `ref_text`: OmniVoice
reads the transcript and the recording's codes together and continues in that voice. Saved voices carry their
transcript, so `voice: NAME` is a clone. A clone keeps the recording's level.

## Length and settings
The length is set before the speech is made, from the text at the recording's pace (or a standard pace): `options`
`speed` (1.2: 20% faster) or `duration` (seconds, exact). `seconds` caps the estimate. A text estimated over 30 s is
said in chunks of about 15 s, split at sentence ends, cross-faded. `steps` (32; 16 is about twice as fast, a little
worse), `cfg` (guidance, 2.0), `seed`. Other `options`: `position-temperature` (5.0; 0: deterministic),
`layer-penalty` (5.0), `t-shift` (0.1), `denoise` (1: with a recording, ask for clean speech), `postprocess` (1:
silences over 0.5 s shortened), `sigh-seconds` (0.8: the time each `[sigh]` gets in the estimate), `min-seconds` (2.0:
the shortest length without a recording - the trailing silence is trimmed). Without these two (0: upstream's estimate)
short lines often came out as a hum or silence instead of their words. A tag with a single word and no recording
(`No [sigh]`) can still lose the word: give a recording or a few more words.

## License
The weights - and omnivoice.cpp's GGUF conversion of them - are **CC BY-NC 4.0: no commercial use** (the training
data's terms, Emilia among them). Upstream's code is Apache 2.0, the Higgs Audio v2 codec Apache 2.0. Upstream forbids
unauthorized voice cloning, impersonation, fraud and scams: do not clone a voice without its owner's consent.
