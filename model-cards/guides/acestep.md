# ACE-Step 1.5

A song from a **caption** (`instructions`: genre, mood, instruments, the voice, production) and **lyrics** (`input`),
`seconds` long (1 to 600; 30 when not said). It sings the lyrics in the time it is given - the length is the request's,
not the lyrics' - so give a long song enough words, and a short one few.

## The caption

Comma-separated tags and short phrases, the way ACE-Step was trained: `upbeat indie pop, bright electric guitars,
punchy drums, warm bass, female lead vocal, summery`. Name the voice (`warm melodic alto female lead vocal`), the
instruments, the mood and the production; tempo and key are better given as options than in words.

## The lyrics

Structure tags on lines of their own - `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`,
`[Instrumental]` - and the words under them, a line a phrase. No lyrics (or `[Instrumental]`): an instrumental.

## Options

| option | what |
|---|---|
| `bpm` | the tempo (beats a minute) |
| `key` | key and scale: `C major`, `A minor`, `F# major` |
| `time-signature` | beats a bar: `4`, `3`, `6` |
| `language` | the lyrics' language as a code (`en`, `zh`, `ja`, `es` ...); `unknown` when not given |
| `shift` | the schedule's shift (3) |
| `dcw` | the per-step wavelet correction (on) |
| `normalize` | the peak to -1 dB (on) |

`steps` (8) can go up to 20; turbo is guidance-distilled, so `cfg` is not used.

## The planner

A song from words alone first goes through the 5 Hz planner LM (as the platform's `use_lm`): it writes the BPM, key,
time signature and a fuller caption (the ones you give are kept), then the song's audio codes, five a second; the
song is made in their shape. `planner: false` (option `planner 0`) skips it: faster, less structured.

## From a recording (`audio`: base64, a data: URL or `upload:<name>`)

| `task` | what | its fields |
|---|---|---|
| `cover` | the recording performed anew: `instructions` its new sound, `input` its words | `cover_strength` (0.5: the share of the steps that follow the recording), `cover_noise` |
| `repaint` | `repaint_start` to `repaint_end` (seconds) made again, the rest kept; an end past the recording extends it | `repaint_mode` (balanced, conservative, aggressive), `repaint_strength` (0.5) |
| `edit` | new words (`input`) sung over the same recording | `source_lyrics` (the words it sings now), `edit_n_min` / `edit_n_max` (0 / 1) |
| `lego` | (acestep-1.5-base) a track - the vocals - made over an instrumental, to be mixed back | `track` (vocals), `steps` (50), `cfg` (7) |

**Not working on SYCL yet (2026-10-24):** the base checkpoint, so `lego` and base text-to-music, makes loud noise (rms 0.6, against 0.13 for the turbo; Whisper hears no speech) at any guidance, 50 steps. The turbo model - text2music, cover, repaint and edit - is checked on the B70. The latent statistics of both models match (final rms 1.03 / 1.06) and the planner is not the cause (turbo without it sings), so it is in how the base weights are driven; the next step is the reference's per-step dumps (`reference/acestep/ref.py`, `audio check --stages guided`).

The song is as long as the recording (a repaint: to its end).
