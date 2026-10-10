# Prompting Qwen3-TTS (speech)

Three models, one method (`audio.speak`, or `/v1/audio/speech`); `input` is always the text to say.

| model | the voice | what steers it |
|---|---|---|
| `qwen3-tts-custom` | one of nine built-in voices (`voice`) | `instructions`: how to say it |
| `qwen3-tts-design` | made up from `instructions` | the description is the voice |
| `qwen3-tts-base` | cloned from a recording (`ref_audio`) | the recording alone |

## The text
- Plain sentences with normal punctuation; commas and full stops set the pauses. Write numbers, dates and
  abbreviations the way they should be read when it matters ("twenty twenty-six", "doctor").
- `language`: english, chinese, japanese, korean, german, french, russian, portuguese, spanish, italian, or `auto`
  (the default: the model's guess). Name it when the text is short or mixes languages.
- Up to ~10 minutes in one request (`seconds` caps it); long texts read best a paragraph a request.

## Built-in voices (`qwen3-tts-custom`)
`vivian`, `serena` (Chinese, female), `uncle_fu` (Chinese, older male), `dylan` (Beijing dialect), `eric` (Sichuan
dialect), `ryan`, `aiden` (English, male), `ono_anna` (Japanese, female), `sohee` (Korean, female). Every voice speaks
every language; its own is its most natural. `instructions` styles the read:
`Cheerful and a little hurried, as if late for a train.` / `Whispering, slow, a bedtime story.`

## Designed voices (`qwen3-tts-design`)
`instructions` describes the speaker: age, gender, timbre, pace, emotion, accent, the setting.
`A calm elderly woman with a soft, slightly raspy voice, speaking slowly and warmly.`
`A young male sports commentator, fast and excited, rising pitch at the end of each sentence.`
The same description with another `seed` gives another speaker of that kind.

## Cloned voices (`qwen3-tts-base`)
`ref_audio`: 3-15 s of one person speaking clearly, little noise or music (WAV, base64 or a data: URL). Alone, the
voice's timbre is taken from it (its x-vector). With `ref_text` - exactly what the recording says - the model also
hears the recording itself (its codec frames) and continues it: a closer clone, the delivery too. The words to say
can be anything, in any of the languages.

## Saved voices
Keep a voice by name and say `"voice": "NAME"` to `audio.speak` (no model needed: a model that clones takes it):
- `audio.voices.add` `{name, sample, text}` - cloned from a recording (WAV, MP3, FLAC, Ogg, M4A), with what it says;
- `audio.voices.design` `{name, instructions, language, seed}` - the voice-design model reads a ten-second line in
  the voice described, and that recording is kept (with its text: the closer, in-context clone). Try a few seeds;
- `audio.voices.list`, `audio.voices.remove`. The command line: `nextsycl audio voice add|design|list|rm`.

## Settings
`seed` repeats a read. `options`: `temperature` (0.9; lower is steadier, flatter), `top-k` (50), `greedy` (1: always
the likeliest - can loop on long texts), `streaming` (1: the text fed as it is spoken).
