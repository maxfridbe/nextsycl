# Prompting ZONOS3-Mini (speech)

`audio.speak` (or `/v1/audio/speech`) with `model: zonos3-mini`; `input` is the text to say. 44.1 kHz mono, 43
languages (no language tag needed: the text decides), many times faster than real time on an Arc card.

## The text
- Plain sentences with normal punctuation; full stops, commas, `!` and `?` shape the read. Write numbers, dates and
  abbreviations the way they should be said when it matters.
- A long text is said a stretch of whole sentences at a time (up to ~400 characters each, a quarter second between),
  in the same voice; `seconds` caps the whole (600 s at most).

## Markup
- `[pause]`, `[pause 0.25|0.5|1|2]`: a pause after a word (the model says long ones shorter, a 2 s one as about half a
  second); `[start]` (first): speech from the first frame.
- `[ipa ...]`: a word said as written in IPA (one of the model's 136 phone symbols each; stress marks, syllable dots and
  spaces are dropped). Write it **in the notation the model was trained on**, Zyphra's pronunciation lexicon: for
  English that is American, CMUdict-style IPA - no length marks (`ː`), `ɝ` / `ɚ` for r-coloured vowels, no flaps or
  glottal stops - e.g. `[ipa təmeɪtoʊ]`, `[ipa ɛləfənt]`, `[ipa kəmpjutɚ]`. Spelled that way the word comes out as
  written (18 of 18 takes of those three words); other notations are not reliable (British `[ipa təmɑːtoʊ]` was heard
  as "tomato" in 2 of 6 takes, and in 1 of 6 in Zyphra's own reference runtime: the model's, not this port's).
- `[steer ID=DOSE, ...]words[/steer]`: steering over those words alone (the controls below).

## Voices
- **Preset** (`voice`): `savannah` (American, female; the default), `wyatt` (American, male), `rupert`, `imogen`
  (British), `cillian` (Irish, male), `jett` (energetic, male), `morgana` (a witch), `sakura` (anime, female),
  `thaddeus` (announcer), `willow` (ASMR); `gaspard` (French), `liesel`, `wolfgang` (German), `ananya`, `arjun`
  (Hindi), `ginevra`, `lorenzo` (Italian), `minjun`, `seoyeon` (Korean), `haoran`, `meilin` (Mandarin), `leonor`,
  `thiago` (Portuguese), `rodrigo`, `ximena` (Spanish), `chaitanya`, `sravya` (Telugu). Any voice speaks any
  language; its own is its most natural. A preset comes in as its recording's codes (the reference prefix, the model's
  default); `options: {"speaker-mode": "embedding"}` uses its speaker embedding instead.
- **Cloned** (`ref_audio`, or a saved voice): 5-30 s of one person speaking clearly. Its timbre is taken by the
  MECT-B2 speaker encoder (a 192-d embedding; a transcript is not needed and not used). With
  `options: {"speaker-mode": "reference"}` the recording itself is the voice's prefix instead (raw-reference cloning,
  the model's own way: its first 60 s encoded to the codec's codes; about 6.3 GiB more on the card for some 6 s, once
  per recording).
- `cfg` (1 = off): speaker guidance against a pass without the voice; 1.5 holds a voice closer, at twice the cost.

## Steering (`options.controls`)
Doses of named directions, `"controls": "delivery.tempo=2, emotions.happy=4"` (about -4 to 4; 0 = none):
- pace and delivery: `delivery.tempo` (+ faster, - slower), `delivery.whisper`, `delivery.loud`,
  `delivery.conversational`, `expressive.expressive`, `disfluency.disfluent`
- mood axes: `mood.valence`, `mood.arousal`, `mood.dominance`, `mood.loudness`, `mood.tempo`
- emotions: `emotions.happy`, `.sad`, `.angry`, `.fearful`, `.surprised`, `.serene`, `.proud`, `.disappointed`, ...
- the voice itself (it moves the speaker embedding, so a cloned voice or `speaker-mode: embedding`):
  `identity.gender`, `identity.age`, `pitch.pitch`, `accents_en.british` (`.american`, `.irish`, `.indian` ...),
  `texture.husky` (`.raspy`, `.deep`, `.soft` ...), `room.clean`

## Other options
**Editing** (`audio.edit`): a recording of up to 30 s, what it says (`original_text`; `audio.transcribe` makes it)
and what it should say (`target_text`): the words that differ (at most three places) are said anew in the recording's
voice, everything else is kept as it was.
`short-text` (auto): a text of a word or two is said with a pause and a filler sentence after it and cut where the
filler starts (alone, the model often says it wrong or runs on); `laci` (off; needs cfg 1): the reading of the text is
checked at every step and a skipped or repeated stretch is said again. `stream: true` on /v1/audio/speech: the sound
comes as it is made.
`level` (on): every take is brought to -19 dBFS of speech and held there (the model drifts quieter through a long
take with some voices); `off` keeps the model's own level, 3 dB down.
`speaking-rate` (UTF-8 bytes a second, ~14 is ordinary English), `loudness` (LUFS), `fmax` (Hz): the model's
condition rows (unset: its own choice). `decoder: full` - the full-size codec head (4 GiB more on the card, loaded on
first use; keep2, the default, is what Zyphra serves). `temperature` (0.75), `top-k` (80), `top-p` (0.69),
`repetition-penalty` (2.4), `silence` (3 frames of codec silence after the text).
