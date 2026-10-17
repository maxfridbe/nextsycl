# Prompting Voxtral 4B TTS (speech in nine languages, 24 kHz)

`input` is the text to say; `voice` is one of the 20 preset voices. The model reads the language from the text: pick
a voice of that language for a native accent (a voice of another language speaks it with that voice's accent).

| voice | language |
|---|---|
| `neutral_female` (the default), `neutral_male`, `casual_female`, `casual_male`, `cheerful_female` | English |
| `fr_female`, `fr_male` | French |
| `de_female`, `de_male` | German |
| `es_female`, `es_male` | Spanish |
| `it_female`, `it_male` | Italian |
| `pt_female`, `pt_male` | Portuguese |
| `nl_female`, `nl_male` | Dutch |
| `ar_male` | Arabic |
| `hi_female`, `hi_male` | Hindi |

Without a `voice`, `language` picks that language's first voice above (`ar_male` for Arabic); without either, the
voice is `neutral_female`.

## What it does not do
The open release (mistralai/Voxtral-4B-TTS-2603) ships the codec's decoder but not its encoder, and its prompt has no
instruction: **no voice cloning** (`ref_audio`, saved voices - those go to a model that clones), **no voice design or
direction** (`instructions` is refused), **no vocal events**. Style comes from the voice: the casual and cheerful
voices speak more loosely than the neutral ones.

## Settings
`cfg` (1.2, upstream's): how strongly the flow follows the backbone - 1 is unguided, higher is more articulated and
eventually harsher. `seed`: the flow's noise (the words and rhythm come from the backbone, which always takes its most
likely code, so seeds change the voice's texture more than the reading). `seconds`: the most (up to 160; the model
ends the speech itself). `options`: `steps` (7): the flow's Euler steps a frame - fewer is faster but rougher.

Long texts: a paragraph or two at a time reads best; split books into chunks.

## License
The weights, this GGUF conversion, the preset voices (from EARS, CML-TTS, IndicVoices-R and the Arabic Natural Audio
dataset) and **everything they make are CC BY-NC 4.0: non-commercial use only**, with attribution to Mistral AI.
