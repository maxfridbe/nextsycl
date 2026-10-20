# Whisper large-v3-turbo: speech to text

`audio.transcribe` writes a recording down: its text, its **segments** (stretches timed by the model's own timestamp
tokens) and, unless `words: false`, every **word** with its start, end and the model's probability for it.

```json
{"audio": "upload:up_1792400000_take2.wav", "language": "english", "words": true}
```

- **audio**: WAV, MP3, FLAC, Ogg or M4A, at any rate (mixed to mono, resampled to 16 kHz) - base64, a data URL, or
  `upload:<name>` after `audio.upload` (send a long recording once).
- **language**: a code (`en`, `de`, `ja`) or a name (`english`, `german`); leave it out (or `auto`) and the first
  30 s decide. Naming it is faster and steadier, above all for sung or noisy audio.
- **translate**: `true` writes English whatever is spoken.
- **prompt**: text that leads the first window - names, spellings, punctuation style ("Dr. Okonkwo, Zürich").
- **options**: `condition` (`1`: each window is prompted with the text before it - steadier style across windows,
  but a mistake can carry on; off by default), `no-speech` (0.6: how sure the model must be that a window is
  silent to skip it), `fallback` (`0` turns off decoding an unsure window again at a higher temperature).

What to expect:

- Long recordings go in 30 s windows; each window moves on from its last timestamp, so nothing is cut mid-word.
- Word times come from the decoder's cross-attention (six alignment heads, dynamic time warping), as
  openai-whisper's `word_timestamps`: good to a few tenths of a second on speech; sung words over a full mix are
  harder - a stem (the vocals alone) times far better.
- Silence and music without words give few or no segments; a hallucinated line over silence is possible on long
  quiet stretches (a known Whisper habit): check `words`' probabilities.
