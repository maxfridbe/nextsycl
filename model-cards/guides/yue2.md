# Prompting YuE2 (songs with vocals)

YuE2-3B is M-A-P's song model, **non-commercial (CC BY-NC 4.0)**: attribution, no commercial use (a commercial
licence comes from the YuE2 team). Two texts make a song - a **style** (`instructions`) and the **lyrics** (`input`) -
and it writes a score of the melody (and chords) before the music, so the words land on a tune.

## The style
A comma-separated list of tags: genre, mood, tempo feel, the voice, the instruments, the production. Upstream's own:

`City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night`
`Indie folk, warm male vocal, acoustic guitar, soft drums, melancholic, heartfelt`
`Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, no guitar, spacious modern harmony`

Name the voice (male / female vocal, its character) and leave out what should not be there ("no guitar").

## The lyrics
- Section tags on lines of their own: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Interlude]`,
  `[Outro]`; a blank line between sections. An empty `[Intro]` or `[Interlude]` is an instrumental passage.
- Start with an empty `[Intro]` (then a blank line), as upstream's own examples do - especially for short lyrics. The
  score is drawn first; without the tag it sometimes gives the intro a single bar and the music then runs the verse into
  the first seconds, and the chorus is never sung. That is the model and the score it drew, not this engine: upstream's
  own code, given the same score, sang no chorus for 2 of 3 seeds. With `[Intro]` the same request (seed 7) sang every
  line, and so did 5 other seeds. If a take loses its words anyway, change the seed; `cot: melody` or `cot: off` draw
  other plans.
- One sung line per line; repeat a chorus in full where it comes back (the model sings what is written).
- English and Mandarin are what it was made for (upstream's demos: English rock, metal, folk; Mandarin pop, funk; here: English folk and rock, Mandarin ballad); write
  Mandarin in characters, phrases separated by spaces.

```
[Verse]
I walked along the river when the city lights were low
Every window held a story that I used to know

[Chorus]
Carry me home, carry me home
Over the water, I am not alone
```

## The score (cot)
- `cot: full` (default): a chord-annotated ABC score first, then the music.
- `cot: melody`: a melody-only score - upstream's recommendation for covers.
- `cot: off`: no score; the codes from the words alone (guided by 1.01 by default).
- `abc`: a score to sing over instead of one drawn (with `cot` melody or full). Every song made with a score says it
  back (`score` in the answer): give it again with the **same seed and style and new lyrics** to keep the melody and
  sing new words; edit its chords for a reharmonization. A cover's score is a recording's melody as ABC (no chord
  symbols for `cot: melody`), its sections matching the lyrics'.

## A cover (a recording sung anew)
Give the recording (`audio`: WAV, MP3, FLAC, Ogg, M4A), **its lyrics** (`input`: the words it sings, with section tags
as above) and the **new style** (`instructions`). SheetSage2 (the model's `sheetsage` file, M-A-P's transcriber, also
CC BY-NC 4.0) writes the recording's lead sheet as ABC - its beats, key, sections and melody - and YuE2 sings that
score in the new style; the recording itself is never played back into the song.

- `cot: melody` (the default for a cover, upstream's recommendation): the melody alone - the new style brings its own
  harmony. `cot: full`: the recording's chords too (closer to the original's harmony).
- **Match the lyrics' sections to the recording's**: YuE2 fits the words to the score's sections (`% intro`, `% verse`,
  `% chorus` ... in the score; `score-only: true` shows them), so tag the lyrics the same way and give a long `[Verse]`
  every verse and chorus it holds. Words under tags the score lacks are scatted, not sung.
- **An instrumental recording**: SheetSage2 writes a played lead line as the score's `Ins` voice; the engine moves it to
  the `Vocal` voice, the one YuE2 sings (`sing-lead: 0` keeps the score as transcribed - an instrumental cover).
- **New words over the recording's tune**: give your own lyrics instead of its; keep their lines and syllables close to
  the original's, section for section, so the notes have words to carry.
- The score comes back (`score` in the answer); sing it again with `abc` and the same seed and style to make the song
  again with other words. `score-only: true` gives the transcription alone (no song) - read it, edit it, then use it as
  `abc`.
- A whole song transcribes in seconds (300 s windows; a longer recording in overlapping windows). A clear lead melody
  transcribes best; a busy mix can put the tune in the `Ins` voice rather than `Vocal` - YuE2 reads both.

```
nextsycl audio gen "Synthwave, female vocal, analog synths, driving drums" --model yue2-3b-q8 --source song.mp3 --lyrics-file its-words.txt
nextsycl audio gen "-" --model yue2-3b-q8 --source song.mp3 --opt-score-only true     # its score alone: song.abc
```

## Length, guidance, steps
`seconds` is the most it makes (up to 360; the model ends the song when the lyrics are sung - a verse and two choruses
run about a minute and a half). `cfg` guides the codes against the instruction alone (1: none; try 1.2 for a closer
reading of the style). `steps` are the flow's (32, upstream's). Each song is one seed: the same request and seed make the
same WAV on the same card; the B70 and the B65 draw different songs from one seed (their rounding differs, and the codes are
drawn from it). A score sung again with new words keeps its melody; the same score and the same words do not repeat the
first take, whose score was decoded a token at a time.
