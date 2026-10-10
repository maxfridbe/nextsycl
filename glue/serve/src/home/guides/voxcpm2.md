# Prompting VoxCPM2 (speech, 48 kHz)

One model, no built-in voices: `input` is the text, in any of 30 languages (no language setting - it reads the
script: Arabic, Burmese, Chinese and its dialects, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi,
Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili,
Swedish, Tagalog, Thai, Turkish, Vietnamese).

| what you send | what you get |
|---|---|
| `input` alone | a voice the model picks for the text |
| `instructions` | a voice made up from the description: `A young woman, gentle and sweet voice` |
| `ref_audio` (or a saved `voice`) | that recording's voice; `instructions` then steers its style: `slightly faster, cheerful` |
| `ref_audio` + `ref_text` | the closest clone: the recording continued (its exact transcript) |

## The text
Write it as it should be read; punctuation sets the pauses. Context matters: the model infers the delivery from the
words (a question rises, an exclamation lifts). Read 1-3 takes of a design (`seed`): results vary between runs.

## Settings
`steps` (10; more: finer, slower), `cfg` (2.0; higher follows the text and the voice more tightly), `seed`.
`options`: `max-ratio` (6: at most this many patches a text token - a runaway's bound).
Saved voices (`audio.voices.add | design`) work here as with any model that clones.
